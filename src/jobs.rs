//! Bounded batch preparation, cancellable image sequences and validated resume.

use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use ghostline_red::assets::ResourceKey;
use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    AssemblyOptions, BUILD_ID, Error, GameSource, Node, PreparedScene, RenderOptions, Renderer,
    Scene,
};

/// Cooperative cancellation, checked between preparation and output work units.
#[derive(Clone, Debug, Default)]
pub struct Cancellation {
    flag: Arc<AtomicBool>,
    file: Option<PathBuf>,
}

impl Cancellation {
    /// Also cancels when the supplied control file exists.
    pub fn with_file(path: PathBuf) -> Self {
        Self {
            file: Some(path),
            ..Self::default()
        }
    }
    /// Requests cancellation from another thread or application control handler.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }
    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire) || self.file.as_ref().is_some_and(|p| p.exists())
    }
}

/// One timestamped image request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameRequest {
    /// Source time in seconds, preserved in the output receipt.
    pub time: f32,
    /// Camera and output dimensions for this image.
    pub render: RenderOptions,
}

/// Result of writing or resuming an image sequence.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SequenceReport {
    /// Images rendered and committed during this call.
    pub written: usize,
    /// Images whose receipts and encoded file hashes were validated and reused.
    pub reused: usize,
    /// Whether processing ended at a cancellation boundary.
    pub cancelled: bool,
    /// Wall time preparing native geometry/materials and checking scene budgets.
    #[serde(default)]
    pub source_seconds: f64,
    /// Wall time decoding textures and preparing the GPU scene.
    #[serde(default)]
    pub gpu_prepare_seconds: f64,
    /// Wall time rendering and reading back newly rendered frames.
    #[serde(default)]
    pub render_seconds: f64,
    /// Wall time encoding, hashing and committing new images and frame receipts.
    #[serde(default)]
    pub image_output_seconds: f64,
}

#[derive(Serialize, Deserialize)]
struct FrameReceipt {
    identity: String,
    index: usize,
    time: f32,
    size: u32,
    png_sha256: String,
}

impl Renderer {
    /// Renders timestamped frames while retaining the prepared scene.
    ///
    /// `input_identity` must identify all source selections and application-side
    /// inputs used by `update`. The output fingerprint also includes this build,
    /// GPU/driver, timestamps and cameras. Resume verifies each PNG and receipt;
    /// incomplete or corrupted frames are rendered again. `update` runs only for
    /// frames that need rendering and must establish the full state for that time.
    /// Use a fixed camera for animation comparisons without automatic reframing.
    ///
    /// # Errors
    /// Returns [`Error`] for invalid requests, conflicting output, update/render
    /// failures or filesystem errors. Already committed frames remain resumable.
    pub fn render_sequence(
        &self,
        scene: &mut PreparedScene,
        frames: &[FrameRequest],
        output: &Path,
        input_identity: &str,
        cancellation: &Cancellation,
        mut update: impl FnMut(&mut PreparedScene, f32) -> Result<(), Error>,
    ) -> Result<SequenceReport, Error> {
        if frames.is_empty()
            || frames.len() > 10_000
            || input_identity.is_empty()
            || frames.iter().any(|f| {
                !f.time.is_finite() || f.time < 0. || !(64..=4096).contains(&f.render.size)
            })
        {
            return Err(Error::InvalidInput("invalid frame sequence request".into()));
        }
        let adapter = self.adapter_info();
        let fingerprint = hash(
            &serde_json::to_vec(&(
                BUILD_ID,
                input_identity,
                frames,
                &adapter.name,
                &adapter.driver_info,
                format!("{:?}", adapter.backend),
            ))
            .map_err(json_error)?,
        );
        fs::create_dir_all(output)?;
        let manifest_path = output.join("sequence.json");
        let manifest = serde_json::json!({"identity":fingerprint,"build":BUILD_ID,"input_identity":input_identity,"frames":frames});
        if manifest_path.exists() {
            let saved: serde_json::Value =
                serde_json::from_slice(&fs::read(&manifest_path)?).map_err(json_error)?;
            if saved["identity"].as_str() != Some(&fingerprint) {
                return Err(Error::InvalidInput(
                    "sequence output belongs to different inputs or rendering settings".into(),
                ));
            }
        } else {
            if fs::read_dir(output)?.next().is_some() {
                return Err(Error::InvalidInput(
                    "sequence output directory is not empty".into(),
                ));
            }
            atomic_json(&manifest_path, &manifest)?;
        }
        let mut report = SequenceReport::default();
        for (index, frame) in frames.iter().enumerate() {
            if cancellation.is_cancelled() {
                report.cancelled = true;
                break;
            }
            let image_path = output.join(format!("{index:06}.png"));
            let receipt_path = output.join(format!("{index:06}.frame.json"));
            if valid_frame(&image_path, &receipt_path, &fingerprint, index, frame) {
                report.reused += 1;
                continue;
            }
            update(scene, frame.time)?;
            if cancellation.is_cancelled() {
                report.cancelled = true;
                break;
            }
            let render_started = std::time::Instant::now();
            let image = self.render(scene, frame.render)?.image;
            if !image.pixels().any(|pixel| pixel.0[3] != 0) {
                return Err(Error::InvalidInput(format!(
                    "frame {index} contains no visible pixels; check native geometry, material coverage and camera"
                )));
            }
            report.render_seconds += render_started.elapsed().as_secs_f64();
            let output_started = std::time::Instant::now();
            let temporary = tempfile::Builder::new()
                .suffix(".png")
                .tempfile_in(output)?;
            image::codecs::png::PngEncoder::new_with_quality(
                temporary.as_file(),
                image::codecs::png::CompressionType::Fast,
                image::codecs::png::FilterType::Adaptive,
            )
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgba8,
            )?;
            temporary.as_file().sync_all()?;
            let encoded = hash(&fs::read(temporary.path())?);
            persist_output(temporary, &image_path)?;
            atomic_json(
                &receipt_path,
                &FrameReceipt {
                    identity: fingerprint.clone(),
                    index,
                    time: frame.time,
                    size: frame.render.size,
                    png_sha256: encoded,
                },
            )?;
            report.written += 1;
            report.image_output_seconds += output_started.elapsed().as_secs_f64();
        }
        atomic_json(&output.join("result.json"), &report)?;
        Ok(report)
    }
}

/// Source selection for one independently prepared batch scene.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BatchSource {
    /// One exact mesh or morphtarget appearance.
    Mesh {
        /// Native path or hash.
        resource: ResourceKey,
        /// Exact mesh appearance.
        appearance: String,
    },
    /// An entity/appearance assembly with explicit component overrides.
    Assembly {
        /// Native entity or appearance resource.
        resource: ResourceKey,
        /// Exact authored selection, absent for simple entities.
        appearance: Option<String>,
        /// Explicit per-component mesh appearances.
        #[serde(default)]
        options: Box<AssemblyOptions>,
    },
}

/// One source with one or more rendered catalogue views.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchJob {
    /// Human-readable job identity, recorded without use as a filesystem path.
    pub name: String,
    /// Native source selection.
    pub source: BatchSource,
    /// Requested views in output order.
    pub views: Vec<RenderOptions>,
    /// Native bone-local axes for rigid standalone subjects. Skinned assets retain their bind frame.
    #[serde(default)]
    pub native_frame: Option<crate::NativeBoneFrame>,
}

/// Bounds on a serial batch's preparation and GPU residency.
#[derive(Clone, Copy, Debug)]
pub struct BatchLimits {
    /// Maximum jobs accepted in a single manifest.
    pub jobs: usize,
    /// Maximum vertices across unique assets in one scene.
    pub vertices_per_scene: usize,
    /// Maximum views per catalogue entry.
    pub views_per_job: usize,
    /// Maximum transform/mesh nodes resident in one scene.
    pub nodes_per_scene: usize,
    /// Conservative decoded RGBA texture budget for one prepared scene.
    pub texture_bytes_per_scene: u64,
}

impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            jobs: 10_000,
            vertices_per_scene: 2_000_000,
            views_per_job: 16,
            nodes_per_scene: 4096,
            texture_bytes_per_scene: 512 * 1024 * 1024,
        }
    }
}

/// Per-job result, including actionable preparation/render errors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobResult {
    /// Native visibility excludes this assembly from ordinary 3D previews.
    #[serde(default)]
    pub non_visual: bool,
    /// Caller-supplied job name.
    pub name: String,
    /// Content-addressed directory beneath the batch output root.
    pub directory: String,
    /// Written/reused image counts when output succeeded.
    pub output: Option<SequenceReport>,
    /// Native preparation, validation or GPU failure, if any.
    pub error: Option<String>,
}

/// Aggregate batch results in input order.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BatchReport {
    /// Completed or failed jobs; pending jobs are absent after cancellation.
    pub jobs: Vec<JobResult>,
    /// Whether a cancellation request ended processing.
    pub cancelled: bool,
}

impl Renderer {
    /// Prepares one scene at a time, renders its views and releases it before the next job.
    ///
    /// Fingerprints include all source archive payload identities, schema, build,
    /// native selections and output settings. Source changes invalidate prior cache
    /// entries. Per-job failures are recorded and do not discard completed work.
    /// `on_result` receives each committed result for progress reporting.
    ///
    /// # Errors
    /// Returns [`Error`] for invalid manifests/output roots or batch receipt I/O.
    pub fn run_batch(
        &self,
        source: &GameSource,
        jobs: &[BatchJob],
        output: &Path,
        limits: BatchLimits,
        cancellation: &Cancellation,
        mut on_result: impl FnMut(&JobResult),
    ) -> Result<BatchReport, Error> {
        if jobs.len() > limits.jobs
            || limits.vertices_per_scene == 0
            || limits.views_per_job == 0
            || jobs.iter().any(|j| {
                j.name.is_empty() || j.views.is_empty() || j.views.len() > limits.views_per_job
            })
        {
            return Err(Error::InvalidInput(
                "batch manifest exceeds limits or contains empty jobs".into(),
            ));
        }
        let output = source.validate_output(output)?;
        fs::create_dir_all(&output)?;
        let marker = output.join(".ghostline-renderer-batch.json");
        if marker.exists() {
            let saved: serde_json::Value =
                serde_json::from_slice(&fs::read(&marker)?).map_err(json_error)?;
            if saved["format"] != "ghostline-renderer-batch" || saved["version"] != 1 {
                return Err(Error::InvalidInput("invalid batch output marker".into()));
            }
        } else {
            if fs::read_dir(&output)?.next().is_some() {
                return Err(Error::InvalidInput(
                    "existing output directory is not a renderer batch".into(),
                ));
            }
            atomic_json(
                &marker,
                &serde_json::json!({"format":"ghostline-renderer-batch","version":1}),
            )?;
        }
        for view in jobs.iter().flat_map(|job| &job.views) {
            if !(64..=4096).contains(&view.size) {
                return Err(Error::InvalidInput(
                    "batch view size must be 64..4096".into(),
                ));
            }
            view.camera.matrices(glam::Vec3::ZERO, 1.)?;
        }
        let source_identity = source.content_identity();
        let mut report = BatchReport::default();
        for job in jobs {
            if cancellation.is_cancelled() {
                report.cancelled = true;
                break;
            }
            let adapter = self.adapter_info();
            let identity = hash(
                &serde_json::to_vec(&(
                    BUILD_ID,
                    &source_identity,
                    job,
                    &adapter.name,
                    &adapter.driver_info,
                    format!("{:?}", adapter.backend),
                ))
                .map_err(json_error)?,
            );
            let directory = output.join(&identity);
            fs::create_dir_all(&directory)?;
            let rendered =
                self.render_batch_job(source, job, &directory, &identity, limits, cancellation);
            let result = match rendered {
                Err(Error::Cancelled) => {
                    report.cancelled = true;
                    JobResult {
                        non_visual: false,
                        name: job.name.clone(),
                        directory: identity,
                        output: Some(SequenceReport {
                            cancelled: true,
                            ..SequenceReport::default()
                        }),
                        error: None,
                    }
                }
                Ok(frames) => {
                    report.cancelled = frames.cancelled;
                    JobResult {
                        non_visual: false,
                        name: job.name.clone(),
                        directory: identity,
                        output: Some(frames),
                        error: None,
                    }
                }
                Err(Error::NonVisual) => JobResult {
                    non_visual: true,
                    name: job.name.clone(),
                    directory: identity,
                    output: None,
                    error: None,
                },
                Err(error) => JobResult {
                    non_visual: false,
                    name: job.name.clone(),
                    directory: identity,
                    output: None,
                    error: Some(error.to_string()),
                },
            };
            atomic_json(&directory.join("job.json"), &result)?;
            on_result(&result);
            report.jobs.push(result);
            atomic_json(&output.join("batch.json"), &report)?;
            if report.cancelled {
                break;
            }
        }
        atomic_json(&output.join("batch.json"), &report)?;
        Ok(report)
    }

    fn render_batch_job(
        &self,
        source: &GameSource,
        job: &BatchJob,
        directory: &Path,
        identity: &str,
        limits: BatchLimits,
        cancellation: &Cancellation,
    ) -> Result<SequenceReport, Error> {
        let frames: Vec<_> = job
            .views
            .iter()
            .map(|&render| FrameRequest { time: 0., render })
            .collect();
        // A fully valid sequence is reused without extracting or allocating its scene.
        if let Some(report) =
            completed_sequence(&directory.join("images"), identity, &frames, self)?
        {
            return Ok(report);
        }
        let source_started = std::time::Instant::now();
        let mut scene = match &job.source {
            BatchSource::Mesh {
                resource,
                appearance,
            } => {
                let asset = source.load_shared_mesh_resource(resource, appearance)?;
                let mut scene = Scene::new();
                let asset = scene.add_shared_asset(asset);
                scene.add_node(Node::new(&job.name, Some(asset)))?;
                scene
            }
            BatchSource::Assembly {
                resource,
                appearance,
                options,
            } => {
                let native = source.preview_assembly(resource, appearance.as_deref(), options)?;
                let mut assembly =
                    crate::SceneAssembly::from_native(native, options, |resource, appearance| {
                        if cancellation.is_cancelled() {
                            return Err(Error::Cancelled);
                        }
                        source.load_assembly_mesh_resource(resource, appearance, options)
                    })?;
                source.add_neutral_references(&mut assembly.scene, options)?;
                atomic_json(
                    &directory.join("assembly-inspection.json"),
                    &assembly.inspection(&job.views)?,
                )?;
                assembly.scene
            }
        };
        if let Some(frame) = &job.native_frame {
            source.apply_native_bone_frame(&mut scene, frame)?;
        }
        let vertices: usize = scene
            .assets
            .iter()
            .flat_map(|a| a.chunks.iter())
            .map(|c| c.positions.len())
            .sum();
        if vertices > limits.vertices_per_scene {
            return Err(Error::InvalidInput(format!(
                "scene has {vertices} vertices, exceeding batch limit {}",
                limits.vertices_per_scene
            )));
        }
        if scene.node_count() > limits.nodes_per_scene {
            return Err(Error::InvalidInput("scene exceeds batch node limit".into()));
        }
        let mut textures = std::collections::HashSet::new();
        let mut texture_bytes = 0_u64;
        for material in scene
            .assets
            .iter()
            .flat_map(|asset| &asset.appearance.materials)
        {
            for path in [
                &material.base_color_texture,
                &material.metallic_roughness_texture,
                &material.normal_texture,
                &material.emissive_texture,
            ]
            .into_iter()
            .flatten()
            .chain(material.hair.as_ref().and_then(|h| h.flow_texture.as_ref()))
            .chain(material.decal.iter().flat_map(|decal| {
                [&decal.gradient_texture, &decal.secondary_texture]
                    .into_iter()
                    .flatten()
            })) {
                if textures.insert(path) {
                    let (width, height) = image::image_dimensions(path)?;
                    texture_bytes = texture_bytes
                        .checked_add(u64::from(width) * u64::from(height) * 4)
                        .ok_or_else(|| Error::InvalidInput("texture budget overflow".into()))?;
                    if texture_bytes > limits.texture_bytes_per_scene {
                        return Err(Error::InvalidInput(
                            "scene exceeds batch texture budget".into(),
                        ));
                    }
                }
            }
        }
        if cancellation.is_cancelled() {
            return Ok(SequenceReport {
                cancelled: true,
                ..SequenceReport::default()
            });
        }
        let source_seconds = source_started.elapsed().as_secs_f64();
        let prepare_started = std::time::Instant::now();
        let mut prepared = self.prepare_scene(&scene)?;
        let gpu_prepare_seconds = prepare_started.elapsed().as_secs_f64();
        let mut report = self.render_sequence(
            &mut prepared,
            &frames,
            &directory.join("images"),
            identity,
            cancellation,
            |_, _| Ok(()),
        )?;
        report.source_seconds = source_seconds;
        report.gpu_prepare_seconds = gpu_prepare_seconds;
        Ok(report)
    }
}

fn completed_sequence(
    output: &Path,
    input: &str,
    frames: &[FrameRequest],
    renderer: &Renderer,
) -> Result<Option<SequenceReport>, Error> {
    let adapter = renderer.adapter_info();
    let fingerprint = hash(
        &serde_json::to_vec(&(
            BUILD_ID,
            input,
            frames,
            &adapter.name,
            &adapter.driver_info,
            format!("{:?}", adapter.backend),
        ))
        .map_err(json_error)?,
    );
    if frames.iter().enumerate().all(|(i, frame)| {
        valid_frame(
            &output.join(format!("{i:06}.png")),
            &output.join(format!("{i:06}.frame.json")),
            &fingerprint,
            i,
            frame,
        )
    }) {
        Ok(Some(SequenceReport {
            written: 0,
            reused: frames.len(),
            cancelled: false,
            ..Default::default()
        }))
    } else {
        Ok(None)
    }
}

fn valid_frame(
    image: &Path,
    receipt: &Path,
    identity: &str,
    index: usize,
    frame: &FrameRequest,
) -> bool {
    let Ok(receipt) = fs::read(receipt) else {
        return false;
    };
    let Ok(receipt) = serde_json::from_slice::<FrameReceipt>(&receipt) else {
        return false;
    };
    if receipt.identity != identity
        || receipt.index != index
        || receipt.time != frame.time
        || receipt.size != frame.render.size
    {
        return false;
    }
    let Ok(bytes) = fs::read(image) else {
        return false;
    };
    if hash(&bytes) != receipt.png_sha256 {
        return false;
    }
    // The complete encoded payload hash already protects every pixel and PNG
    // chunk. Read dimensions without decoding and hashing the same pixels again.
    let Ok((width, height)) =
        image::ImageReader::with_format(std::io::Cursor::new(&bytes), image::ImageFormat::Png)
            .into_dimensions()
    else {
        return false;
    };
    width == frame.render.size && height == frame.render.size
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn json_error(error: serde_json::Error) -> Error {
    Error::InvalidInput(format!("invalid job/receipt JSON: {error}"))
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), Error> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::InvalidInput("receipt has no parent directory".into()))?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|source| Error::FileIo {
        operation: "create receipt temporary file for",
        path: path.to_owned(),
        source,
    })?;
    {
        let mut writer = BufWriter::with_capacity(64 * 1024, &mut file);
        serde_json::to_writer_pretty(&mut writer, value).map_err(json_error)?;
        writer.flush()?;
    }
    file.as_file().sync_all()?;
    persist_output(file, path)
}

fn persist_output(mut file: tempfile::NamedTempFile, path: &Path) -> Result<(), Error> {
    for attempt in 0..8 {
        match file.persist(path) {
            Ok(_) => return Ok(()),
            Err(error)
                if cfg!(windows)
                    && attempt < 7
                    && matches!(error.error.raw_os_error(), Some(5 | 32 | 33)) =>
            {
                file = error.file;
                std::thread::sleep(std::time::Duration::from_millis(20 << attempt));
            }
            Err(error) => {
                return Err(Error::FileIo {
                    operation: "commit renderer output",
                    path: path.to_owned(),
                    source: error.error,
                });
            }
        }
    }
    unreachable!("output persistence returns or retries")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_output_resumes_after_cancellation_and_repairs_corrupt_images() {
        let renderer = pollster::block_on(Renderer::new()).unwrap();
        let mut scene = Scene::new();
        let asset = scene.add_asset(crate::gpu::tests::triangle());
        scene.add_node(Node::new("mesh", Some(asset))).unwrap();
        let mut prepared = renderer.prepare_scene(&scene).unwrap();
        let frames: Vec<_> = [0., 0.5, 1.]
            .into_iter()
            .map(|time| FrameRequest {
                time,
                render: RenderOptions {
                    size: 64,
                    ..RenderOptions::default()
                },
            })
            .collect();
        let directory = tempfile::tempdir().unwrap();
        let cancel = Cancellation::default();
        let first = renderer
            .render_sequence(
                &mut prepared,
                &frames,
                directory.path(),
                "fixture",
                &cancel,
                |_, time| {
                    if time > 0. {
                        cancel.cancel();
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert!(first.cancelled);
        assert_eq!(first.written, 1);
        let resumed = renderer
            .render_sequence(
                &mut prepared,
                &frames,
                directory.path(),
                "fixture",
                &Cancellation::default(),
                |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!((resumed.written, resumed.reused), (2, 1));
        fs::write(directory.path().join("000000.png"), b"corrupt").unwrap();
        let repaired = renderer
            .render_sequence(
                &mut prepared,
                &frames,
                directory.path(),
                "fixture",
                &Cancellation::default(),
                |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!((repaired.written, repaired.reused), (1, 2));
        assert!(
            renderer
                .render_sequence(
                    &mut prepared,
                    &frames,
                    directory.path(),
                    "different source",
                    &Cancellation::default(),
                    |_, _| Ok(())
                )
                .is_err()
        );
    }
}
