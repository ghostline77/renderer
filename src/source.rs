use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use ghostline_red::{
    assets::ResourceKey,
    material::{ArchiveSet, export_mesh_materials_from_document},
    mesh::{MeshChunk, MeshSkin, decode_geometry, decode_skin},
    pbr::{BakedAppearance, bake_appearance},
    schema::RedSchema,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::Error;

/// Decoded native geometry and material-source document before map preparation.
#[derive(Clone, Debug)]
pub struct GeometryAsset {
    shadow_only: bool,
    chunks: Arc<Vec<MeshChunk>>,
    skin: Option<MeshSkin>,
    morphs: Arc<Vec<ghostline_red::morph::MorphTarget>>,
    /// SHA-256 of native geometry and base-mesh source bytes.
    pub source_sha256: String,
    /// Requested native resource identity.
    pub mesh: String,
    material_document: Arc<serde_json::Value>,
}

impl GeometryAsset {
    /// Replaces decoded chunks while retaining their native skin palette.
    /// Material attachment validates the resulting geometry before GPU upload.
    pub fn with_chunks(mut self, chunks: Vec<MeshChunk>) -> Self {
        self.chunks = Arc::new(chunks);
        self
    }
    /// Uses authored native geometry and skin streams without preparing textures.
    ///
    /// # Errors
    /// Returns native geometry or skin decoding errors.
    pub fn with_document(mut self, document: &serde_json::Value) -> Result<Self, Error> {
        self.chunks = Arc::new(decode_geometry(document, true)?);
        self.skin = decode_skin(document)?;
        self.material_document = Arc::new(document.clone());
        self.source_sha256 = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(document).map_err(|e| Error::InvalidInput(e.to_string()))?
            )
        );
        Ok(self)
    }
    /// Whether native metadata marks this as a shadow-only mesh.
    pub fn is_shadow_only(&self) -> bool {
        self.shadow_only
    }
    /// Exact authored appearances on the material-source mesh.
    pub fn appearance_names(&self) -> Vec<&str> {
        self.material_document["Data"]["RootChunk"]["appearances"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|a| {
                let appearance = a.get("Data").unwrap_or(a);
                appearance.get("name").map_or(Some("default"), |name| {
                    name.get("$value").unwrap_or(name).as_str()
                })
            })
            .collect()
    }
    /// Highest-detail native render chunks in Y-up coordinates.
    pub fn chunks(&self) -> &[MeshChunk] {
        &self.chunks
    }
    /// Named source bone palette and inverse bind matrices.
    pub fn skin(&self) -> Option<&MeshSkin> {
        self.skin.as_ref()
    }
    /// Named native morph shapes.
    pub fn morph_targets(&self) -> &[ghostline_red::morph::MorphTarget] {
        &self.morphs
    }

    /// Attaches a texture-free matte grey material for reference mannequins.
    /// Native geometry, chunk indices and morphs are retained without baking skin.
    ///
    /// # Errors
    /// Returns [`Error`] for invalid geometry material bindings.
    pub fn with_neutral_material(self) -> Result<MeshAsset, Error> {
        self.with_diagnostic_material("Diagnostic neutral reference; native materials omitted")
    }

    /// Attaches a matte diagnostic surface while retaining native geometry.
    ///
    /// # Errors
    /// Returns errors for invalid geometry or material bindings.
    pub fn with_diagnostic_material(self, reason: &str) -> Result<MeshAsset, Error> {
        self.with_materials(
            "neutral-reference",
            BakedAppearance {
                chunk_materials: vec!["neutral-reference".into()],
                materials: vec![ghostline_red::pbr::PbrMaterial {
                    name: "neutral-reference".into(),
                    base_color_texture: None,
                    base_color_factor: [0.32, 0.32, 0.32, 1.],
                    uv_transform: [1., 1., 0., 0.],
                    metallic_roughness_texture: None,
                    metallic_factor: 0.,
                    roughness_factor: 0.85,
                    normal_texture: None,
                    emissive_texture: None,
                    emissive_factor: [0.; 3],
                    emissive_strength: 1.,
                    alpha_mode: "OPAQUE",
                    alpha_cutoff: None,
                    stochastic_coverage: false,
                    hair: None,
                    decal: None,
                    transmission: 0.,
                    double_sided: true,
                    preview_notes: vec![reason.into()],
                }],
                summary: Default::default(),
            },
        )
    }

    /// Attaches explicitly prepared materials to this geometry.
    ///
    /// # Errors
    /// Returns [`Error`] for missing or ambiguous material bindings.
    pub fn with_materials(
        mut self,
        name: &str,
        mut appearance: BakedAppearance,
    ) -> Result<MeshAsset, Error> {
        let mut names = std::collections::HashSet::new();
        if name.is_empty()
            || appearance.chunk_materials.is_empty()
            || appearance
                .materials
                .iter()
                .any(|m| !names.insert(m.name.as_str()))
            || self.chunks.iter().any(|chunk| {
                !names.contains(
                    appearance.chunk_materials
                        [chunk.chunk_index % appearance.chunk_materials.len()]
                    .as_str(),
                )
            })
        {
            return Err(Error::InvalidInput(
                "invalid geometry material bindings".into(),
            ));
        }
        if self.chunks.iter().any(|chunk| {
            chunk
                .tex_coords
                .iter()
                .any(|uv| uv.iter().any(|v| !v.is_finite()))
        }) {
            for chunk in Arc::make_mut(&mut self.chunks) {
                if chunk.tex_coords.len() != chunk.positions.len() {
                    return Err(Error::InvalidInput(format!(
                        "mesh {} chunk {} has mismatched UV and position counts",
                        self.mesh, chunk.chunk_index
                    )));
                }
                let original = chunk.tex_coords.clone();
                let mut minimum = [f32::INFINITY; 3];
                let mut maximum = [f32::NEG_INFINITY; 3];
                for position in &chunk.positions {
                    for axis in 0..3 {
                        minimum[axis] = minimum[axis].min(position[axis]);
                        maximum[axis] = maximum[axis].max(position[axis]);
                    }
                }
                let mut axes = [0, 1, 2];
                axes.sort_by(|a, b| {
                    (maximum[*b] - minimum[*b]).total_cmp(&(maximum[*a] - minimum[*a]))
                });
                let mut repaired = 0;
                let mut projected = 0;
                for (index, uv) in chunk.tex_coords.iter_mut().enumerate() {
                    if uv.iter().all(|v| v.is_finite()) {
                        continue;
                    }
                    // Keep native geometry intact. Prefer a finite UV on a
                    // duplicate position; otherwise use an adjacent vertex.
                    // This is an explicit preview repair, never a source edit.
                    let replacement = original
                        .iter()
                        .enumerate()
                        .find(|(other, candidate)| {
                            *other != index
                                && chunk.positions[*other] == chunk.positions[index]
                                && candidate.iter().all(|v| v.is_finite())
                        })
                        .map(|(_, value)| *value)
                        .or_else(|| {
                            chunk
                                .indices
                                .chunks_exact(3)
                                .filter(|triangle| triangle.iter().any(|v| *v as usize == index))
                                .flat_map(|triangle| triangle.iter())
                                .filter_map(|v| original.get(*v as usize))
                                .find(|candidate| candidate.iter().all(|v| v.is_finite()))
                                .copied()
                        });
                    let replacement = replacement
                        .or_else(|| {
                            chunk
                                .tex_coords_1
                                .get(index)
                                .filter(|candidate| candidate.iter().all(|v| v.is_finite()))
                                .copied()
                        })
                        .unwrap_or_else(|| {
                            projected += 1;
                            // Invalid source UVs cannot be passed to the GPU.
                            // Preserve every triangle and use an explicitly
                            // reported planar preview mapping for these vertices.
                            std::array::from_fn(|i| {
                                let axis = axes[i];
                                (chunk.positions[index][axis] - minimum[axis])
                                    / (maximum[axis] - minimum[axis]).max(f32::EPSILON)
                            })
                        });
                    *uv = replacement;
                    repaired += 1;
                }
                if repaired > 0 {
                    for material in &mut appearance.materials {
                        material.preview_notes.push(format!("Preview repaired {repaired} nonfinite native UVs in chunk {} using finite duplicate/adjacent or secondary UVs; {projected} vertices required planar preview projection. Original geometry and source bytes are preserved", chunk.chunk_index));
                    }
                }
            }
        }
        Ok(MeshAsset {
            _preparation: None,
            _material_repository: None,
            shadow_only: self.shadow_only,
            chunks: self.chunks,
            skin: self.skin,
            morphs: self.morphs,
            source_sha256: self.source_sha256,
            mesh: self.mesh,
            appearance,
            appearance_name: name.to_owned(),
        })
    }
}

/// Decoded mesh geometry and the exact selected appearance's baked materials.
#[derive(Debug)]
pub struct MeshAsset {
    pub(crate) _preparation: Option<Arc<tempfile::TempDir>>,
    pub(crate) _material_repository: Option<Arc<tempfile::TempDir>>,
    pub(crate) shadow_only: bool,
    pub(crate) chunks: Arc<Vec<MeshChunk>>,
    pub(crate) appearance: BakedAppearance,
    pub(crate) skin: Option<MeshSkin>,
    pub(crate) morphs: Arc<Vec<ghostline_red::morph::MorphTarget>>,
    /// SHA-256 of the native mesh bytes used for preparation.
    pub source_sha256: String,
    /// Native mesh depot path.
    pub mesh: String,
    /// Exact authored appearance name.
    pub appearance_name: String,
}

impl MeshAsset {
    /// Reuses prepared native materials and textures with an edited rigid mesh.
    /// Material slots are rebound from the edited document, including split LOD chunks.
    ///
    /// # Errors
    /// Returns an error for malformed native streams or an absent appearance binding.
    pub fn with_document(&self, document: &serde_json::Value) -> Result<Self, Error> {
        let geometry = decode_geometry(document, true)?;
        let base = GeometryAsset {
            shadow_only: self.shadow_only,
            chunks: Arc::new(geometry),
            skin: decode_skin(document)?,
            morphs: Arc::new(Vec::new()),
            source_sha256: format!(
                "{:x}",
                Sha256::digest(
                    serde_json::to_vec(document).map_err(|e| Error::InvalidInput(e.to_string()))?
                )
            ),
            mesh: self.mesh.clone(),
            material_document: Arc::new(document.clone()),
        };
        let index = base
            .appearance_names()
            .iter()
            .position(|name| *name == self.appearance_name)
            .ok_or_else(|| Error::InvalidInput("edited native appearance is absent".into()))?;
        let mut appearance = self.appearance.clone();
        appearance.chunk_materials = base
            .chunks
            .iter()
            .map(|chunk| {
                chunk.material_names.get(index).cloned().ok_or_else(|| {
                    Error::InvalidInput("edited native material slot is absent".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut edited = base.with_materials(&self.appearance_name, appearance)?;
        edited._preparation = self._preparation.clone();
        edited._material_repository = self._material_repository.clone();
        Ok(edited)
    }
    /// Returns native material inputs retained by a shared preparation, when present.
    ///
    /// # Errors
    /// Returns an I/O or JSON error if a retained material manifest cannot be read.
    pub fn material_provenance(&self) -> Result<Option<serde_json::Value>, Error> {
        let Some(directory) = &self._preparation else {
            return Ok(None);
        };
        let path = directory.path().join("asset/appearance.Material.json");
        Ok(Some(
            serde_json::from_slice(&fs::read(path)?)
                .map_err(|e| Error::InvalidInput(e.to_string()))?,
        ))
    }
    /// Shares the native mesh and prepared materials with authored morph targets.
    ///
    /// The retained-scene preparation validates the target indices and streams.
    pub fn with_morph_targets(&self, targets: Vec<ghostline_red::morph::MorphTarget>) -> Self {
        Self {
            _preparation: self._preparation.clone(),
            _material_repository: self._material_repository.clone(),
            shadow_only: self.shadow_only,
            chunks: Arc::clone(&self.chunks),
            appearance: self.appearance.clone(),
            skin: self.skin.clone(),
            morphs: Arc::new(targets),
            source_sha256: self.source_sha256.clone(),
            mesh: self.mesh.clone(),
            appearance_name: self.appearance_name.clone(),
        }
    }
    pub(crate) fn surface_indices<'a>(&self, chunk: &'a MeshChunk) -> Cow<'a, [u16]> {
        let name = &self.appearance.chunk_materials
            [chunk.chunk_index % self.appearance.chunk_materials.len()];
        if !self
            .appearance
            .materials
            .iter()
            .any(|m| m.name == *name && m.double_sided)
        {
            return Cow::Borrowed(&chunk.indices);
        }
        let mut orientations = HashMap::<[u16; 3], u8>::new();
        for tri in chunk.indices.chunks_exact(3) {
            let mut key = [tri[0], tri[1], tri[2]];
            let inversions =
                u8::from(tri[0] > tri[1]) + u8::from(tri[0] > tri[2]) + u8::from(tri[1] > tri[2]);
            key.sort_unstable();
            *orientations.entry(key).or_default() |= 1 << (inversions % 2);
        }
        if !orientations.values().any(|&flags| flags == 3) {
            return Cow::Borrowed(&chunk.indices);
        }
        let mut emitted = HashSet::new();
        let mut indices = Vec::with_capacity(chunk.indices.len());
        for tri in chunk.indices.chunks_exact(3) {
            let mut key = [tri[0], tri[1], tri[2]];
            key.sort_unstable();
            let mut face = [tri[0], tri[1], tri[2]];
            if orientations[&key] == 3 {
                if !emitted.insert(key) {
                    continue;
                }
                // Native cards can encode both windings using the same vertices.
                // A two-sided GPU material needs only one of those surfaces, with
                // winding aligned to its authored normals for back-face shading.
                let [a, b, c] = face.map(|i| glam::Vec3::from(chunk.positions[usize::from(i)]));
                let normal = face
                    .iter()
                    .filter_map(|&i| chunk.normals.get(usize::from(i)))
                    .copied()
                    .map(glam::Vec3::from)
                    .sum::<glam::Vec3>();
                if (b - a).cross(c - a).dot(normal) < 0. {
                    face.swap(0, 1);
                }
            }
            indices.extend_from_slice(&face);
        }
        Cow::Owned(indices)
    }

    /// Whether native metadata marks this as a shadow-only mesh.
    pub fn is_shadow_only(&self) -> bool {
        self.shadow_only
    }
    /// Prepared material parameters and texture paths, in GPU material index order.
    pub fn materials(&self) -> &[ghostline_red::pbr::PbrMaterial] {
        &self.appearance.materials
    }
    /// Authored morph targets and their native regions, in target index order.
    pub fn morph_targets(&self) -> &[ghostline_red::morph::MorphTarget] {
        &self.morphs
    }
    /// Number of visible highest-detail render chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Number of decoded triangles.
    pub fn triangle_count(&self) -> usize {
        self.chunks.iter().map(|c| c.indices.len() / 3).sum()
    }

    /// Notes about effects approximated or omitted by native material conversion.
    pub fn material_notes(&self) -> impl Iterator<Item = &str> {
        self.appearance
            .materials
            .iter()
            .flat_map(|m| m.preview_notes.iter().map(String::as_str))
    }
}

#[derive(Serialize)]
struct SourceReceipt<'a> {
    mesh: &'a str,
    appearance: &'a str,
    source_sha256: &'a str,
}

/// A retained archive index and reflected schema for native mesh preparation.
///
/// Open once and reuse for multiple meshes/appearances. Loaded assets own their
/// geometry and remain renderable after this source is dropped.
#[derive(Debug)]
pub struct GameSource {
    game_root: PathBuf,
    archives: ArchiveSet,
    schema: RedSchema,
    schema_sha256: String,
    content_identity: String,
    geometry_cache: ResourceCache<GeometryAsset>,
    mesh_cache: ResourceCache<MeshAsset>,
    material_repository: Arc<tempfile::TempDir>,
    documents: ghostline_red::assets::DocumentCache,
}

type CacheSlot<T> = Arc<Mutex<Option<Arc<T>>>>;

#[derive(Debug)]
struct CacheEntry<T> {
    slot: CacheSlot<T>,
    touched: u64,
}

/// Bounded, per-resource single-flight preparation shared by render workers.
#[derive(Debug)]
struct ResourceCache<T> {
    entries: Mutex<HashMap<(u64, String), CacheEntry<T>>>,
    clock: std::sync::atomic::AtomicU64,
}

impl<T> Default for ResourceCache<T> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            clock: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl<T> ResourceCache<T> {
    fn get(
        &self,
        key: (u64, String),
        load: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Arc<T>, Error> {
        let slot = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| Error::InvalidInput("resource cache poisoned".into()))?;
            while !entries.contains_key(&key) && entries.len() >= 64 {
                // Keep the working set, and never evict a slot while another
                // thread is preparing/waiting for it: that duplicates work.
                if let Some(old) = entries
                    .iter()
                    .filter(|(_, entry)| Arc::strong_count(&entry.slot) == 1)
                    .min_by_key(|(_, entry)| entry.touched)
                    .map(|(key, _)| key.clone())
                {
                    entries.remove(&old);
                } else {
                    break;
                }
            }
            let touched = self
                .clock
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let entry = entries.entry(key).or_insert_with(|| CacheEntry {
                slot: Arc::new(Mutex::new(None)),
                touched,
            });
            entry.touched = touched;
            Arc::clone(&entry.slot)
        };
        let mut value = slot
            .lock()
            .map_err(|_| Error::InvalidInput("resource preparation poisoned".into()))?;
        if let Some(asset) = &*value {
            return Ok(Arc::clone(asset));
        }
        let asset = Arc::new(load()?);
        *value = Some(Arc::clone(&asset));
        Ok(asset)
    }
}

impl GameSource {
    /// Opens base/expansion archives and loads reflection metadata once.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] for missing sources, malformed archives or invalid schemas.
    pub fn open(game_root: &Path, schema_path: Option<&Path>) -> Result<Self, Error> {
        Self::open_with_overlays(game_root, schema_path, &[])
    }

    /// Opens vanilla archives plus authored archives outside the game installation.
    /// This allows native previews without installing generated mods.
    ///
    /// # Errors
    /// Returns errors for unreadable archives or invalid reflection metadata.
    pub fn open_with_overlays(
        game_root: &Path,
        schema_path: Option<&Path>,
        overlays: &[PathBuf],
    ) -> Result<Self, Error> {
        let game_root = game_root.canonicalize()?;
        let schema = RedSchema::load(Some(&game_root), schema_path)?;
        let schema_bytes = serde_json::to_vec(&schema).map_err(|error| {
            Error::InvalidInput(format!("cannot fingerprint reflection metadata: {error}"))
        })?;
        let schema_sha256 = format!("{:x}", Sha256::digest(schema_bytes));
        let mut paths = Vec::new();
        for folder in ["archive/pc/content", "archive/pc/ep1"] {
            let directory = game_root.join(folder);
            if !directory.exists() && folder.ends_with("ep1") {
                continue;
            }
            for entry in fs::read_dir(directory)? {
                let path = entry?.path();
                if path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("archive"))
                {
                    paths.push(path);
                }
            }
        }
        if paths.is_empty() {
            return Err(Error::InvalidInput("no vanilla game archives found".into()));
        }
        paths.extend_from_slice(overlays);
        let archives = ArchiveSet::from_paths(paths)?;
        let content_identity = archives.content_identity();

        Ok(Self {
            game_root,
            archives,
            schema,
            schema_sha256,
            content_identity,
            geometry_cache: ResourceCache::default(),
            mesh_cache: ResourceCache::default(),
            documents: ghostline_red::assets::DocumentCache::default(),
            material_repository: Arc::new(
                tempfile::Builder::new()
                    .prefix("ghostline-materials-")
                    .tempdir()?,
            ),
        })
    }

    /// SHA-256 of the reflected schema held by this source.
    pub fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }

    /// Reflection metadata used by this archive source, including native overlays.
    pub fn schema(&self) -> &RedSchema {
        &self.schema
    }

    /// Prepares native geometry and materials from an authored mesh document.
    ///
    /// # Errors
    /// Returns native material decoding, texture preparation or I/O errors.
    pub fn load_material_override(
        &self,
        resource: &ResourceKey,
        appearance: &str,
        document: &serde_json::Value,
    ) -> Result<Arc<MeshAsset>, Error> {
        let directory = Arc::new(
            tempfile::Builder::new()
                .prefix("ghostline-authored-")
                .tempdir()?,
        );
        // Geometry and its inverse bind matrices form one authored asset.
        let geometry = self.load_geometry(resource)?.with_document(document)?;
        let work = directory.path().join("asset");
        fs::create_dir(&work)?;
        let repository = self
            .material_repository
            .path()
            .join(&geometry.source_sha256);
        let sidecar = work.join("appearance.Material.json");
        let indices: Vec<_> = geometry.chunks.iter().map(|c| c.chunk_index).collect();
        export_mesh_materials_from_document(
            document,
            &self.schema,
            &self.archives,
            &repository,
            &sidecar,
            Some(appearance),
            Some(&indices),
            true,
        )?;
        let materials = bake_appearance(&sidecar, appearance, &indices, Some(512))?;
        let mut asset = geometry.with_materials(appearance, materials)?;
        asset._preparation = Some(directory);
        asset._material_repository = Some(Arc::clone(&self.material_repository));
        Ok(Arc::new(asset))
    }

    /// Identity of the selected archive payload index and reflection schema.
    pub fn content_identity(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(self.content_identity.as_bytes());
        hash.update(self.schema_sha256.as_bytes());
        format!("{:x}", hash.finalize())
    }

    /// Fingerprint of the effective native archive payloads, independent of schema metadata.
    #[must_use]
    pub fn archive_identity(&self) -> &str {
        &self.content_identity
    }

    /// Resolves an output location and rejects writes inside the game installation.
    ///
    /// # Errors
    /// Returns an error when the path cannot be resolved or overlaps game files.
    pub fn validate_output(&self, path: &Path) -> Result<PathBuf, Error> {
        let path = canonical_destination(path)?;
        if path.starts_with(&self.game_root) {
            return Err(Error::InvalidInput(
                "output directory must be outside the game installation".into(),
            ));
        }
        Ok(path)
    }

    /// Loads highest-detail geometry and an exact named appearance.
    ///
    /// Geometry and material resolution share one decoded mesh document.
    /// `work` is a new directory outside the game installation for prepared maps
    /// and material diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] for invalid selections, absent resources/appearances,
    /// unsupported data or preparation I/O errors.
    pub fn load_mesh(&self, mesh: &str, appearance: &str, work: &Path) -> Result<MeshAsset, Error> {
        let mesh = mesh_selection(mesh, appearance)?;
        self.load_mesh_resource(&ResourceKey::Path(mesh), appearance, work)
    }

    /// Loads a mesh by exact depot path or native hash.
    ///
    /// # Errors
    /// Returns [`Error`] for absent/unsupported resources, appearances or invalid output paths.
    pub fn load_mesh_resource(
        &self,
        resource: &ResourceKey,
        appearance: &str,
        work: &Path,
    ) -> Result<MeshAsset, Error> {
        self.prepare_mesh(resource, appearance, work, None, 1)
    }

    fn prepare_mesh(
        &self,
        resource: &ResourceKey,
        appearance: &str,
        work: &Path,
        shared_repository: Option<&Path>,
        lod_mask: u64,
    ) -> Result<MeshAsset, Error> {
        if appearance.trim().is_empty() {
            return Err(Error::InvalidInput("mesh appearance must be named".into()));
        }
        let work = canonical_destination(work)?;
        if work.starts_with(&self.game_root) {
            return Err(Error::InvalidInput(
                "preparation directory must be outside the game installation".into(),
            ));
        }
        if work.exists() {
            return Err(Error::InvalidInput(
                "preparation directory already exists".into(),
            ));
        }
        let mut geometry = self.load_geometry(resource)?;
        if lod_mask != 1 {
            if !geometry.morphs.is_empty() {
                return Err(Error::InvalidInput(
                    "LOD selection for morph targets requires matching native morph data".into(),
                ));
            }
            let chunks = decode_geometry(&geometry.material_document,false)?.into_iter().filter(|chunk| {
                geometry.material_document["Data"]["RootChunk"]["renderResourceBlob"]["Data"]["header"]["renderChunkInfos"][chunk.chunk_index]["lodMask"].as_u64() == Some(lod_mask)
            }).collect::<Vec<_>>();
            if chunks.is_empty() {
                return Err(Error::InvalidInput(format!(
                    "mesh {resource} has no native LOD mask {lod_mask}"
                )));
            }
            geometry.chunks = Arc::new(chunks);
        }
        // Shadow geometry can be a hierarchy dependency, but its placeholder
        // material is never drawn. Preserve the geometry and native flag without
        // trying to bake a surface that has no material template.
        if geometry.is_shadow_only() {
            return geometry.with_neutral_material();
        }
        if geometry.material_document["Data"]["RootChunk"]
            .get("materialEntries")
            .and_then(serde_json::Value::as_array)
            .is_none_or(Vec::is_empty)
        {
            return geometry.with_diagnostic_material(
                "Native geometry has no authored material assignments; diagnostic neutral preview",
            );
        }
        if !geometry.appearance_names().contains(&appearance) {
            return Err(Error::InvalidInput(format!(
                "appearance {appearance:?} is absent on {resource}; authored names: {}",
                geometry.appearance_names().join(", ")
            )));
        }
        let GeometryAsset {
            shadow_only,
            chunks,
            skin,
            morphs,
            source_sha256,
            mesh,
            material_document,
        } = geometry;
        if let Some(parent) = work.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(&work)?;
        // A shared depot repository lets the native exporter and PBR baker reuse
        // textures across different appearances, not just identical mesh jobs.
        // Embedded resources are scoped to their owning mesh to avoid collisions.
        let embedded = material_document["Data"]["EmbeddedFiles"]
            .as_array()
            .is_some_and(|files| !files.is_empty());
        let repository = match shared_repository {
            Some(root) if embedded => root.join(&source_sha256),
            Some(root) => root.to_path_buf(),
            None => work.join("materials"),
        };
        let sidecar = work.join("appearance.Material.json");
        let selected_indices: Vec<_> = chunks.iter().map(|c| c.chunk_index).collect();
        export_mesh_materials_from_document(
            &material_document,
            &self.schema,
            &self.archives,
            &repository,
            &sidecar,
            Some(appearance),
            Some(&selected_indices),
            true,
        )?;
        let (chunks, animation_notes) =
            fluid_preview_frame(&material_document, chunks, &sidecar, appearance)?;
        let indices: Vec<_> = chunks.iter().map(|c| c.chunk_index).collect();
        let mut baked = bake_appearance(&sidecar, appearance, &indices, Some(512))?;
        for material in &mut baked.materials {
            material
                .preview_notes
                .extend(animation_notes.iter().cloned());
        }
        let identity = SourceReceipt {
            mesh: &mesh,
            appearance,
            source_sha256: &source_sha256,
        };
        fs::write(
            work.join("source.json"),
            serde_json::to_vec_pretty(&identity)
                .map_err(|error| Error::InvalidInput(error.to_string()))?,
        )?;
        GeometryAsset {
            shadow_only,
            chunks,
            skin,
            morphs,
            source_sha256,
            mesh,
            material_document,
        }
        .with_materials(appearance, baked)
    }

    /// Reuses prepared geometry and materials across batch jobs and workers.
    /// Backing texture files live until the last scene using the asset is released.
    ///
    /// # Errors
    /// Returns the same resource/preparation errors as [`Self::load_mesh_resource`].
    pub fn load_shared_mesh_resource(
        &self,
        resource: &ResourceKey,
        appearance: &str,
    ) -> Result<Arc<MeshAsset>, Error> {
        self.mesh_cache.get(
            (resource.native_hash(), format!("appearance:{appearance}")),
            || {
                let directory = Arc::new(
                    tempfile::Builder::new()
                        .prefix("ghostline-prepared-")
                        .tempdir()?,
                );
                let mut asset = self.prepare_mesh(
                    resource,
                    appearance,
                    &directory.path().join("asset"),
                    Some(self.material_repository.path()),
                    1,
                )?;
                asset._preparation = Some(directory);
                asset._material_repository = Some(Arc::clone(&self.material_repository));
                Ok(asset)
            },
        )
    }

    /// Selects an explicitly requested native LOD while sharing prepared materials.
    ///
    /// # Errors
    /// Returns native preparation errors or an error when the requested LOD is absent.
    pub fn load_shared_mesh_resource_lod(
        &self,
        resource: &ResourceKey,
        appearance: &str,
        lod_mask: u64,
    ) -> Result<Arc<MeshAsset>, Error> {
        if lod_mask == 1 {
            return self.load_shared_mesh_resource(resource, appearance);
        }
        self.mesh_cache.get(
            (
                resource.native_hash(),
                format!("appearance:{appearance}:lod:{lod_mask}"),
            ),
            || {
                let directory = Arc::new(
                    tempfile::Builder::new()
                        .prefix("ghostline-prepared-")
                        .tempdir()?,
                );
                let mut asset = self.prepare_mesh(
                    resource,
                    appearance,
                    &directory.path().join("asset"),
                    Some(self.material_repository.path()),
                    lod_mask,
                )?;
                asset._preparation = Some(directory);
                asset._material_repository = Some(Arc::clone(&self.material_repository));
                Ok(asset)
            },
        )
    }

    pub(crate) fn load_shared_neutral(
        &self,
        resource: &ResourceKey,
    ) -> Result<Arc<MeshAsset>, Error> {
        self.mesh_cache
            .get((resource.native_hash(), "neutral".into()), || {
                self.load_geometry(resource)?.with_neutral_material()
            })
    }

    /// Decodes native geometry, skin palettes and morphs without extracting material maps.
    ///
    /// # Errors
    /// Returns [`Error`] for missing resources or malformed native geometry.
    pub fn load_geometry(&self, resource: &ResourceKey) -> Result<GeometryAsset, Error> {
        self.geometry_cache
            .get((resource.native_hash(), String::new()), || {
                self.decode_geometry_resource(resource)
            })
            .map(|asset| (*asset).clone())
    }

    fn decode_geometry_resource(&self, resource: &ResourceKey) -> Result<GeometryAsset, Error> {
        let mesh = resource.to_string();
        let bytes = match resource {
            ResourceKey::Path(path) => self.archives.read_resource(path)?,
            ResourceKey::Hash(hash) => self.archives.read_resource_hash(*hash)?,
        };
        let mut digest = Sha256::new();
        digest.update(&bytes);
        let document = self.resources().document(resource)?;
        let (chunks, skin, morphs, material_document) =
            if document["Data"]["RootChunk"]["$type"] == "MorphTargetMesh" {
                let base = ghostline_red::morph::base_mesh(&document)?;
                let bytes = match &base {
                    ResourceKey::Path(path) => self.archives.read_resource(path)?,
                    ResourceKey::Hash(hash) => self.archives.read_resource_hash(*hash)?,
                };
                digest.update(&bytes);
                let base = self.resources().document(&base)?;
                let morph = ghostline_red::morph::decode(&document, &base)?;
                (morph.chunks, morph.skin, morph.targets, base)
            } else {
                (
                    decode_geometry(&document, true)?,
                    decode_skin(&document)?,
                    Vec::new(),
                    document,
                )
            };
        let source_sha256 = format!("{:x}", digest.finalize());
        let root = &material_document["Data"]["RootChunk"];
        let shadow_only = ["isShadowMesh", "isPlayerShadowMesh"]
            .iter()
            .any(|key| root[*key].as_bool().unwrap_or(false));
        Ok(GeometryAsset {
            shadow_only,
            chunks: Arc::new(chunks),
            skin,
            morphs: Arc::new(morphs),
            source_sha256,
            mesh,
            material_document,
        })
    }

    /// Creates a cached native resource reader sharing this source's archive index.
    pub fn resources(&self) -> ghostline_red::assets::AssetReader<'_> {
        ghostline_red::assets::AssetReader::with_cache(
            &self.archives,
            &self.schema,
            &self.documents,
        )
    }

    /// Resolves native component definitions and rigs without preparing GPU assets.
    ///
    /// # Errors
    /// Returns [`Error`] for absent resources or invalid assembly data.
    pub fn resolve_assembly(
        &self,
        resource: &ResourceKey,
        appearance: Option<&str>,
    ) -> Result<ghostline_red::assets::Assembly, Error> {
        Ok(self.resources().assembly(resource, appearance)?)
    }
}

fn fluid_preview_frame(
    document: &serde_json::Value,
    mut chunks: Arc<Vec<MeshChunk>>,
    sidecar: &Path,
    appearance: &str,
) -> Result<(Arc<Vec<MeshChunk>>, Vec<String>), Error> {
    if !chunks.iter().all(|chunk| {
        chunk
            .positions
            .iter()
            .all(|p| Some(p) == chunk.positions.first())
    }) {
        return Ok((chunks, Vec::new()));
    }
    let materials: serde_json::Value = serde_json::from_slice(&fs::read(sidecar)?)
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
    let names = materials["Appearances"][appearance]
        .as_array()
        .ok_or_else(|| Error::InvalidInput("fluid appearance bindings absent".into()))?;
    if names.is_empty() {
        return Err(Error::InvalidInput(
            "fluid appearance has no material bindings".into(),
        ));
    }
    let records = materials["Materials"]
        .as_array()
        .ok_or_else(|| Error::InvalidInput("fluid material records absent".into()))?;
    let mut notes = Vec::new();
    for chunk in Arc::make_mut(&mut chunks) {
        let name = names.get(chunk.chunk_index % names.len());
        let Some(record) = records.iter().find(|record| Some(&record["Name"]) == name) else {
            continue;
        };
        if record["MaterialTemplate"] != r"base\materials\ver_mov_glass.mt" {
            continue;
        }
        let data = &record["Data"];
        let Some(texture) = data["vertex_paint_tex"].as_str() else {
            continue;
        };
        if data["YAxisUp"].as_f64().unwrap_or(0.) != 0. {
            return Err(Error::InvalidInput(
                "fluid VAT Y-up conversion is unsupported".into(),
            ));
        }
        let frames = data["n_frames"].as_f64().unwrap_or(0.);
        let pieces = data["n_pieces"].as_f64().unwrap_or(0.);
        if frames < 1.
            || pieces < 1.
            || frames > f64::from(u32::MAX)
            || pieces > f64::from(u32::MAX)
        {
            return Err(Error::InvalidInput(
                "invalid native fluid frame/point counts".into(),
            ));
        }
        let repository = record["MaterialRepo"]
            .as_str()
            .or_else(|| materials["MaterialRepo"].as_str())
            .ok_or_else(|| Error::InvalidInput("fluid material repository absent".into()))?;
        let path = Path::new(repository)
            .join(texture.replace('\\', "/"))
            .with_extension("png");
        let texture = image::open(path)
            .map_err(|e| Error::InvalidInput(e.to_string()))?
            .into_rgba8();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "validated native integer counts"
        )]
        let (frame, pieces) = ((frames as u32) / 2, pieces as u32);
        let range = [
            data["trans_min"].as_f64().unwrap_or(0.) as f32,
            data["trans_max"].as_f64().unwrap_or(0.) as f32,
        ];
        ghostline_red::mesh::apply_fluid_vertex_frame(
            document, chunk, &texture, frame, pieces, range,
        )?;
        notes.push(format!("Collapsed native fluid geometry reconstructed from authored vertex-animation frame {frame} of {frames}, using native point indices and translation range; playback is frozen for the preview"));
    }
    Ok((chunks, notes))
}

fn mesh_selection(mesh: &str, appearance: &str) -> Result<String, Error> {
    let mesh = mesh.replace('/', "\\");
    if mesh.starts_with('\\')
        || mesh.contains(':')
        || mesh.split('\\').any(|p| p == "..")
        || !(mesh.ends_with(".mesh") || mesh.ends_with(".morphtarget"))
        || appearance.trim().is_empty()
    {
        return Err(Error::InvalidInput(
            "an exact mesh depot path and named appearance are required".into(),
        ));
    }
    Ok(mesh)
}

fn canonical_destination(path: &Path) -> Result<std::path::PathBuf, std::io::Error> {
    let absolute = std::path::absolute(path)?;
    let mut parent = absolute.as_path();
    let mut suffix = Vec::new();
    while !parent.exists() {
        let name = parent
            .file_name()
            .ok_or_else(|| std::io::Error::other("destination has no existing ancestor"))?;
        suffix.push(name);
        parent = parent
            .parent()
            .ok_or_else(|| std::io::Error::other("destination has no parent"))?;
    }
    let mut result = parent.canonicalize()?;
    for name in suffix.into_iter().rev() {
        result.push(name);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn resource_cache_keeps_recent_hits() {
        let cache = super::ResourceCache::<usize>::default();
        for key in 0..64 {
            cache
                .get((key, String::new()), || Ok(key as usize))
                .unwrap();
        }
        cache
            .get((0, String::new()), || panic!("recent resource was lost"))
            .unwrap();
        cache.get((64, String::new()), || Ok(64)).unwrap();
        cache
            .get((0, String::new()), || panic!("recent resource was evicted"))
            .unwrap();
        assert_eq!(*cache.get((1, String::new()), || Ok(101)).unwrap(), 101);
    }

    #[test]
    fn resource_cache_does_not_duplicate_inflight_preparation() {
        use std::sync::{Arc, Barrier};
        let cache = Arc::new(super::ResourceCache::<usize>::default());
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        std::thread::scope(|scope| {
            let cache_owner = Arc::clone(&cache);
            let owner_started = Arc::clone(&started);
            let owner_release = Arc::clone(&release);
            let owner = scope.spawn(move || {
                cache_owner
                    .get((0, String::new()), || {
                        owner_started.wait();
                        owner_release.wait();
                        Ok(7)
                    })
                    .unwrap()
            });
            started.wait();
            for key in 1..100 {
                cache
                    .get((key, String::new()), || Ok(key as usize))
                    .unwrap();
            }
            release.wait();
            assert_eq!(*owner.join().unwrap(), 7);
            assert_eq!(*cache.get((0, String::new()), || Ok(99)).unwrap(), 7);
        });
    }
    use super::*;

    #[test]
    fn two_sided_cards_preserve_native_indices_and_draw_one_surface() {
        let mut asset = crate::gpu::tests::triangle();
        Arc::make_mut(&mut asset.chunks)[0].indices = vec![0, 1, 2, 2, 1, 0];
        let indices = asset.surface_indices(&asset.chunks[0]);
        assert_eq!(indices.len(), 3);
        let [a, b, c] = [indices[0], indices[1], indices[2]]
            .map(|i| glam::Vec3::from(asset.chunks[0].positions[usize::from(i)]));
        assert!(
            (b - a)
                .cross(c - a)
                .dot(glam::Vec3::from(asset.chunks[0].normals[0]))
                > 0.
        );
        assert_eq!(asset.chunks[0].indices, vec![0, 1, 2, 2, 1, 0]);
        asset.appearance.materials[0].double_sided = false;
        assert_eq!(
            asset.surface_indices(&asset.chunks[0]).as_ref(),
            asset.chunks[0].indices
        );
    }

    #[test]
    fn preparation_rejects_output_beneath_game_without_creating_it() {
        let game = tempfile::tempdir().unwrap();
        let work = game.path().join("new/subdirectory");
        let source = GameSource {
            game_root: game.path().canonicalize().unwrap(),
            archives: ArchiveSet::from_paths(Vec::new()).unwrap(),
            schema: RedSchema::default(),
            schema_sha256: String::new(),
            content_identity: String::new(),
            geometry_cache: ResourceCache::default(),
            mesh_cache: ResourceCache::default(),
            documents: ghostline_red::assets::DocumentCache::default(),
            material_repository: Arc::new(tempfile::tempdir().unwrap()),
        };
        let result = source.load_mesh("base/test.mesh", "default", &work);
        assert!(matches!(result, Err(Error::InvalidInput(_))));
        assert!(!work.exists());
    }

    #[test]
    fn invalid_resource_selection_is_rejected_before_source_access() {
        for mesh in ["../bad.mesh", "C:/bad.mesh", "/bad.mesh", "base/test.ent"] {
            let result = mesh_selection(mesh, "default");
            assert!(matches!(result, Err(Error::InvalidInput(_))), "{mesh}");
        }
    }
}
