use std::sync::Arc;

use super::*;
use crate::{NodeId, Scene, scene::chunk_visible};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceUniform {
    model: [[f32; 4]; 4],
    normals: [[f32; 4]; 4],
    orientation: [f32; 4],
}

struct InstanceBinding {
    buffer: wgpu::Buffer,
    group: wgpu::BindGroup,
    world: Option<Mat4>,
}

pub(super) struct AssetBuffers {
    draws: Vec<Draw>,
    materials: Vec<wgpu::BindGroup>,
    resident_bytes: u64,
}

#[derive(Default)]
pub(super) struct AssetCache {
    entries: std::collections::HashMap<usize, (Arc<MeshAsset>, Arc<AssetBuffers>)>,
    bytes: u64,
}

struct DeformedDraw {
    vertices: wgpu::Buffer,
    bounds: [Vec3; 8],
    side_bounds: [Option<[Vec3; 8]>; 2],
    centroid: Vec3,
}

struct Deformation {
    palette: Vec<Mat4>,
    weights: Vec<f32>,
    draws: Vec<DeformedDraw>,
}

struct MaterialOverride {
    material: ghostline_red::pbr::PbrMaterial,
    group: wgpu::BindGroup,
}

/// GPU allocation counts for a prepared scene.
#[derive(Clone, Copy, Debug)]
pub struct SceneStatistics {
    /// Number of unique loaded assets.
    pub assets: usize,
    /// Number of geometry chunks uploaded, independent of instance count.
    pub geometry_chunks: usize,
    /// Number of material bindings.
    pub materials: usize,
    /// Number of mesh-bearing component instances.
    pub instances: usize,
}

/// Retained GPU geometry, materials, instance transforms and render targets.
pub struct PreparedScene {
    owner: Arc<()>,
    scene: Scene,
    assets: Vec<Arc<AssetBuffers>>,
    instances: Vec<InstanceBinding>,
    camera: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    targets: Option<Targets>,
    deformed: Vec<Option<Deformation>>,
    morph_weights: std::collections::HashMap<NodeId, Vec<f32>>,
    material_overrides: std::collections::HashMap<(NodeId, usize), MaterialOverride>,
    lighting: crate::Lighting,
}

impl PreparedScene {
    /// Encodes the current pose, morph weights, visibility and material overrides for a browser.
    /// # Errors
    /// Returns geometry, material, texture I/O and GLB size errors.
    pub fn browser_glb(&self) -> Result<Vec<u8>, Error> {
        let overrides = self
            .material_overrides
            .iter()
            .map(|(key, value)| (*key, &value.material))
            .collect();
        crate::browser::encode(&self.scene, &self.morph_weights, &overrides)
    }
    /// Evaluates posed skeletal geometry for collision inspection, without morphs.
    ///
    /// # Errors
    /// Returns errors for invalid skin bindings or scene transforms.
    pub fn evaluated_geometry(&self) -> Result<Vec<crate::EvaluatedChunk>, Error> {
        self.scene.evaluated_geometry()
    }
    /// Replaces preview illumination without rebuilding geometry or materials.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for non-finite or negative values,
    /// zero directional vectors, invalid radii or more than 64 lights.
    pub fn set_lighting(&mut self, lighting: crate::Lighting) -> Result<(), Error> {
        lighting.validate()?;
        self.lighting = lighting;
        Ok(())
    }
    /// Returns the authored morph targets for an editable mesh instance.
    ///
    /// # Errors
    /// Returns an error when the node has no mesh asset in this scene.
    pub fn morph_targets(
        &self,
        node: NodeId,
    ) -> Result<&[ghostline_red::morph::MorphTarget], Error> {
        let asset = self
            .scene
            .nodes
            .get(node.0)
            .and_then(|node| node.asset)
            .ok_or_else(|| Error::InvalidInput("morph node has no mesh asset".into()))?;
        Ok(self.scene.assets[asset.0].morph_targets())
    }
    /// Removes a control-rig driver while preserving the current deformation pose.
    pub fn unlink_rig(&mut self, destination: crate::RigId) -> bool {
        self.scene.unlink_rig(destination)
    }
    /// Removes a per-instance material override and restores the asset material.
    pub fn reset_material(&mut self, node: NodeId, index: usize) -> bool {
        self.material_overrides.remove(&(node, index)).is_some()
    }
    /// Samples and applies a native clip, retaining all prepared GPU assets.
    ///
    /// # Errors
    /// Returns [`Error`] for an absent rig, incompatible clip or invalid sample.
    pub fn apply_animation(
        &mut self,
        rig: crate::RigId,
        clip: &ghostline_red::animation::AnimationClip,
        time: f32,
        additive_base: Option<&crate::Pose>,
        root_motion: crate::RootMotion,
    ) -> Result<crate::PoseSample, Error> {
        let instance = self
            .scene
            .rigs
            .get(rig.0)
            .ok_or_else(|| Error::InvalidInput("absent animation rig".into()))?;
        let sample = instance
            .skeleton
            .sample_animation(clip, time, additive_base, root_motion)?;
        self.scene.set_pose(rig, &sample.pose)?;
        Ok(sample)
    }
    /// Sets all morph weights for one mesh instance in authored target order.
    ///
    /// Morph deltas are combined before skinning, without replacing material or
    /// index buffers. Other instances of the same asset keep their own weights.
    ///
    /// # Errors
    /// Returns [`Error`] for absent mesh nodes, mismatched counts or nonfinite weights.
    pub fn set_morph_weights(&mut self, node: NodeId, weights: &[f32]) -> Result<(), Error> {
        let asset = self
            .scene
            .nodes
            .get(node.0)
            .and_then(|n| n.asset)
            .ok_or_else(|| Error::InvalidInput("morph node has no mesh asset".into()))?;
        if weights.len() != self.scene.assets[asset.0].morphs.len()
            || weights.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidInput("invalid morph weights".into()));
        }
        self.morph_weights.insert(node, weights.to_vec());
        Ok(())
    }
    /// Updates skeleton bones, skin deformation and rigid attachments for the next frame.
    ///
    /// # Errors
    /// Returns [`Error`] for an absent rig or invalid pose.
    pub fn set_pose(&mut self, rig: crate::RigId, pose: &crate::Pose) -> Result<(), Error> {
        self.scene.set_pose(rig, pose)
    }
    /// Reports allocated asset/chunk counts.
    pub fn statistics(&self) -> SceneStatistics {
        SceneStatistics {
            assets: self.assets.len(),
            geometry_chunks: self.assets.iter().map(|a| a.draws.len()).sum(),
            materials: self.assets.iter().map(|a| a.materials.len()).sum::<usize>()
                + self.material_overrides.len(),
            instances: self
                .scene
                .nodes
                .iter()
                .filter(|n| n.asset.is_some())
                .count(),
        }
    }

    /// Updates a local transform while retaining uploaded geometry and materials.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for absent nodes or invalid transforms.
    pub fn set_transform(&mut self, node: NodeId, transform: Mat4) -> Result<(), Error> {
        self.scene.set_transform(node, transform)
    }

    /// Updates native chunk visibility.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for an absent node.
    pub fn set_chunk_mask(&mut self, node: NodeId, mask: u64) -> Result<(), Error> {
        self.scene.set_chunk_mask(node, mask)
    }

    /// Shows or hides a component hierarchy.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for an absent node.
    pub fn set_visible(&mut self, node: NodeId, visible: bool) -> Result<(), Error> {
        self.scene.set_visible(node, visible)
    }

    /// Changes a component parent without reuploading its asset.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for absent references or hierarchy cycles.
    pub fn set_parent(&mut self, node: NodeId, parent: Option<NodeId>) -> Result<(), Error> {
        self.scene.set_parent(node, parent)
    }
}

impl Renderer {
    /// Changes one instance's material without rebuilding geometry or other instances.
    ///
    /// Identical texture pixels share GPU storage, including across prepared assets.
    ///
    /// # Errors
    /// Returns [`Error`] for incompatible devices, absent materials or invalid textures.
    pub fn set_material(
        &self,
        scene: &mut PreparedScene,
        node: NodeId,
        index: usize,
        material: ghostline_red::pbr::PbrMaterial,
    ) -> Result<(), Error> {
        if !Arc::ptr_eq(&scene.owner, &self.owner) {
            return Err(Error::InvalidInput(
                "scene belongs to another renderer".into(),
            ));
        }
        let asset = scene
            .scene
            .nodes
            .get(node.0)
            .and_then(|n| n.asset)
            .ok_or_else(|| Error::InvalidInput("material node has no mesh".into()))?;
        if index >= scene.assets[asset.0].materials.len() {
            return Err(Error::InvalidInput("material index out of range".into()));
        }
        let group = self.material(&material)?;
        scene
            .material_overrides
            .insert((node, index), MaterialOverride { material, group });
        Ok(())
    }
    /// Uploads each shared asset once and creates editable component bindings.
    ///
    /// # Errors
    /// Returns [`Error`] for invalid geometry, material dependencies or GPU limits.
    pub fn prepare_scene(&self, scene: &Scene) -> Result<PreparedScene, Error> {
        scene.world_transforms()?;
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let assets = scene
            .assets
            .iter()
            .map(|asset| {
                let mut cache = self
                    .assets
                    .lock()
                    .map_err(|_| Error::Gpu("asset cache poisoned".into()))?;
                let key = Arc::as_ptr(asset) as usize;
                if let Some((_, buffers)) = cache.entries.get(&key) {
                    return Ok(Arc::clone(buffers));
                }
                let buffers = Arc::new(self.upload_asset(asset)?);
                const BUDGET: u64 = 64 * 1024 * 1024;
                while cache.bytes.saturating_add(buffers.resident_bytes) > BUDGET
                    || cache.entries.len() >= 16
                {
                    let Some(old) = cache.entries.keys().next().copied() else {
                        break;
                    };
                    if let Some((_, old)) = cache.entries.remove(&old) {
                        cache.bytes = cache.bytes.saturating_sub(old.resident_bytes);
                    }
                }
                if buffers.resident_bytes <= BUDGET {
                    cache.bytes += buffers.resident_bytes;
                    cache
                        .entries
                        .insert(key, (Arc::clone(asset), Arc::clone(&buffers)));
                }
                Ok(buffers)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let instances = scene
            .nodes
            .iter()
            .map(|_| {
                let buffer = self.buffer(
                    "instance",
                    bytemuck::bytes_of(&InstanceUniform::zeroed()),
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                );
                let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("instance"),
                    layout: &self.instance_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });
                InstanceBinding {
                    buffer,
                    group,
                    world: None,
                }
            })
            .collect();
        let camera = self.buffer(
            "camera",
            bytemuck::bytes_of(&Globals::zeroed()),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let camera_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: &self.globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            }],
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(Error::Gpu(error.to_string()));
        }
        Ok(PreparedScene {
            lighting: crate::Lighting::default(),
            material_overrides: std::collections::HashMap::new(),
            morph_weights: std::collections::HashMap::new(),
            deformed: (0..scene.nodes.len()).map(|_| None).collect(),
            owner: Arc::clone(&self.owner),
            scene: scene.clone(),
            assets,
            instances,
            camera,
            camera_group,
            targets: None,
        })
    }

    fn upload_asset(&self, asset: &MeshAsset) -> Result<AssetBuffers, Error> {
        let table = &asset.appearance.chunk_materials;
        if table.is_empty() {
            return Err(Error::InvalidInput(
                "empty appearance material table".into(),
            ));
        }
        let mut draws = Vec::new();
        for chunk in asset.chunks.iter() {
            if chunk.positions.is_empty()
                || chunk.indices.is_empty()
                || chunk.indices.len() % 3 != 0
                || chunk.normals.len() != chunk.positions.len()
                || chunk.tex_coords.len() != chunk.positions.len()
                || chunk
                    .indices
                    .iter()
                    .any(|&i| usize::from(i) >= chunk.positions.len())
                || chunk
                    .positions
                    .iter()
                    .flatten()
                    .chain(chunk.normals.iter().flatten())
                    .chain(chunk.tex_coords.iter().flatten())
                    // The GPU vertex interface consumes UV0 only. Native UV1
                    // padding can contain NaNs; it is retained for inspection
                    // but is never uploaded or sampled by this pipeline.
                    .any(|v| !v.is_finite())
            {
                return Err(Error::InvalidInput(format!(
                    "invalid geometry in mesh {} chunk {}: {} vertices, {} indices, {} normals, {} UVs; nonfinite positions={}, normals={}, UVs={}, secondary UVs={}",
                    asset.mesh,
                    chunk.chunk_index,
                    chunk.positions.len(),
                    chunk.indices.len(),
                    chunk.normals.len(),
                    chunk.tex_coords.len(),
                    chunk
                        .positions
                        .iter()
                        .flatten()
                        .filter(|v| !v.is_finite())
                        .count(),
                    chunk
                        .normals
                        .iter()
                        .flatten()
                        .filter(|v| !v.is_finite())
                        .count(),
                    chunk
                        .tex_coords
                        .iter()
                        .flatten()
                        .filter(|v| !v.is_finite())
                        .count(),
                    chunk
                        .tex_coords_1
                        .iter()
                        .flatten()
                        .filter(|v| !v.is_finite())
                        .count()
                )));
            }
            let name = &table[chunk.chunk_index % table.len()];
            let material = asset
                .appearance
                .materials
                .iter()
                .position(|m| m.name == *name)
                .ok_or_else(|| Error::InvalidInput(format!("missing material {name}")))?;
            let vertices: Vec<_> = chunk
                .positions
                .iter()
                .enumerate()
                .map(|(i, &position)| Vertex {
                    position,
                    normal: chunk.normals[i],
                    uv: chunk.tex_coords[i],
                    tangent: chunk.tangents.get(i).copied().unwrap_or([0.0; 4]),
                    color: chunk.colors.get(i).copied().unwrap_or([0.; 4]),
                })
                .collect();
            let indices = asset.surface_indices(chunk);
            let (bounds, centroid) = vertex_bounds(&vertices, indices.as_ref())?;
            draws.push(Draw {
                vertices: self.buffer(
                    "vertices",
                    bytemuck::cast_slice(&vertices),
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                ),
                indices: self.buffer(
                    "indices",
                    bytemuck::cast_slice(indices.as_ref()),
                    wgpu::BufferUsages::INDEX,
                ),
                count: u32::try_from(indices.len())
                    .map_err(|_| Error::InvalidInput("too many indices".into()))?,
                material,
                chunk: chunk.chunk_index,
                centroid,
                bounds,
                side_bounds: vertex_side_bounds(&vertices, indices.as_ref()),
            });
        }
        let materials = asset
            .appearance
            .materials
            .iter()
            .map(|m| self.material(m))
            .collect::<Result<_, _>>()?;
        // Bind groups retain their textures even after the texture lookup cache
        // evicts them. Account for that ownership, not only vertex buffers.
        let mut paths = std::collections::BTreeSet::new();
        for m in &asset.appearance.materials {
            paths.extend(
                [
                    &m.base_color_texture,
                    &m.normal_texture,
                    &m.metallic_roughness_texture,
                    &m.emissive_texture,
                ]
                .into_iter()
                .flatten(),
            );
            paths.extend(m.hair.iter().filter_map(|h| h.flow_texture.as_ref()));
            paths.extend(m.decal.iter().flat_map(|d| {
                [&d.gradient_texture, &d.secondary_texture]
                    .into_iter()
                    .flatten()
            }));
        }
        let mut resident_bytes = draws
            .iter()
            .map(|d| d.vertices.size() * 2 + d.indices.size())
            .sum::<u64>();
        for path in paths {
            let (width, height) = image::image_dimensions(path)?;
            resident_bytes =
                resident_bytes.saturating_add(u64::from(width) * u64::from(height) * 16 / 3);
        }
        Ok(AssetBuffers {
            draws,
            materials,
            resident_bytes,
        })
    }

    /// Renders retained assets with their current hierarchy and visibility.
    ///
    /// # Errors
    /// Returns [`Error`] for incompatible devices, invalid dimensions or GPU errors.
    pub fn render(
        &self,
        scene: &mut PreparedScene,
        options: RenderOptions,
    ) -> Result<RenderedImage, Error> {
        if !Arc::ptr_eq(&scene.owner, &self.owner) {
            return Err(Error::InvalidInput(
                "scene belongs to a different renderer".into(),
            ));
        }
        if !(64..=4096).contains(&options.size) {
            return Err(Error::InvalidInput("size must be 64..4096".into()));
        }
        scene.scene.update_rig_links()?;
        let worlds = scene.scene.world_transforms()?;
        self.update_deformations(scene, &worlds)?;
        let mut points = Vec::new();
        let mut side_fallback = Vec::new();
        let mut subject_part = 0;
        for (index, node) in scene.scene.nodes.iter().enumerate() {
            let (world, visible) = worlds[index];
            if scene.instances[index].world != Some(world) {
                let uniform = InstanceUniform {
                    model: world.to_cols_array_2d(),
                    normals: world.inverse().transpose().to_cols_array_2d(),
                    orientation: [world.determinant().signum(), 0.0, 0.0, 0.0],
                };
                self.queue.write_buffer(
                    &scene.instances[index].buffer,
                    0,
                    bytemuck::bytes_of(&uniform),
                );
                scene.instances[index].world = Some(world);
            }
            let subject = visible
                && !node.reference
                && node.asset.is_some_and(|asset| {
                    scene.assets[asset.0]
                        .draws
                        .iter()
                        .any(|draw| chunk_visible(node.chunk_mask, draw.chunk) && draw.count > 0)
                });
            let fit = match options.camera {
                Camera::FrameSubject(_)
                | Camera::FrameSubjectSide { .. }
                | Camera::FrameSubjectRegion { .. } => subject,
                Camera::FrameSubjectCrop { part, .. } => {
                    subject && part.is_none_or(|part| subject_part == part)
                }
                Camera::FrameSubjectPart { part, .. } => subject && subject_part == part,
                _ => visible,
            };
            subject_part += usize::from(subject);
            if let Some(asset) = node.asset.filter(|_| fit) {
                for (chunk_index, chunk) in scene.assets[asset.0].draws.iter().enumerate() {
                    if chunk_visible(node.chunk_mask, chunk.chunk) {
                        let bounds = scene.deformed[index]
                            .as_ref()
                            .map_or(&chunk.bounds, |d| &d.draws[chunk_index].bounds);
                        if let Camera::FrameSubjectSide { positive, .. } = options.camera {
                            side_fallback.extend(bounds.iter().map(|p| world.transform_point3(*p)));
                            let sides = scene.deformed[index]
                                .as_ref()
                                .map_or(&chunk.side_bounds, |d| &d.draws[chunk_index].side_bounds);
                            if let Some(bounds) = &sides[usize::from(positive)] {
                                points.extend(bounds.iter().map(|p| world.transform_point3(*p)));
                            }
                        } else {
                            points.extend(bounds.iter().map(|p| world.transform_point3(*p)));
                        }
                    }
                }
            }
        }
        // A standalone left cuff or glove has no positive-side counterpart.
        // Frame the actual visible subject instead of failing that preview.
        if points.is_empty() && !side_fallback.is_empty() {
            points = side_fallback;
        }
        if points.is_empty()
            && matches!(
                options.camera,
                Camera::FrameSubjectPart { .. }
                    | Camera::FrameSubjectSide { .. }
                    | Camera::FrameSubjectCrop { part: Some(_), .. }
            )
        {
            return Err(Error::InvalidInput(
                "camera subject part has no visible geometry".into(),
            ));
        }
        let (center, radius) = if points.is_empty() {
            (Vec3::ZERO, 1.0)
        } else {
            options.camera.subject_bounds(&mut points)?
        };
        let (projection, eye) = options.camera.matrices(center, radius)?;
        let projection = options.camera.fit_projection(projection, &points)?;
        let mut lights = [GpuLight::zeroed(); 64];
        for (target, light) in lights.iter_mut().zip(&scene.lighting.lights) {
            *target = match light {
                crate::Light::Directional { direction, color } => GpuLight {
                    position: [direction[0], direction[1], direction[2], 0.],
                    color: [color[0], color[1], color[2], 0.],
                },
                crate::Light::Point {
                    position,
                    color,
                    radius,
                } => GpuLight {
                    position: [position[0], position[1], position[2], 1.],
                    color: [color[0], color[1], color[2], *radius],
                },
            };
        }
        self.queue.write_buffer(
            &scene.camera,
            0,
            bytemuck::bytes_of(&Globals {
                projection: projection.to_cols_array_2d(),
                eye: eye.extend(1.0).to_array(),
                ambient: [
                    scene.lighting.ambient[0],
                    scene.lighting.ambient[1],
                    scene.lighting.ambient[2],
                    scene.lighting.sky,
                ],
                counts: [scene.lighting.lights.len() as u32, 0, 0, 0],
                lights,
            }),
        );
        let mut draws = Vec::new();
        for (node_index, node) in scene.scene.nodes.iter().enumerate() {
            let Some(asset) = node.asset.filter(|_| worlds[node_index].1) else {
                continue;
            };
            for (chunk, draw) in scene.assets[asset.0].draws.iter().enumerate() {
                if !chunk_visible(node.chunk_mask, draw.chunk) {
                    continue;
                }
                let material = scene
                    .material_overrides
                    .get(&(NodeId(node_index), draw.material))
                    .map_or(
                        &scene.scene.assets[asset.0].appearance.materials[draw.material],
                        |m| &m.material,
                    );
                let centroid = scene.deformed[node_index]
                    .as_ref()
                    .map_or(draw.centroid, |d| d.draws[chunk].centroid);
                let distance =
                    eye.distance_squared(worlds[node_index].0.transform_point3(centroid));
                draws.push((
                    material.alpha_mode == "BLEND" && !material.stochastic_coverage,
                    distance,
                    node_index,
                    asset.0,
                    chunk,
                ));
            }
        }
        draws.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)));
        let coverage_sampling = scene.scene.assets.iter().any(|asset| {
            asset
                .materials()
                .iter()
                .any(|material| material.stochastic_coverage)
        });
        let sampling = if coverage_sampling && options.size <= 2048 {
            4
        } else {
            2
        };
        let deferred = scene
            .scene
            .assets
            .iter()
            .flat_map(|asset| asset.materials())
            .chain(
                scene
                    .material_overrides
                    .values()
                    .map(|value| &value.material),
            )
            .any(|material| {
                material.decal.is_some()
                    && !material.stochastic_coverage
                    && material.alpha_mode == "BLEND"
            });
        if scene.targets.as_ref().is_none_or(|t| {
            t.size != options.size * sampling || t.frame_group.is_some() != deferred
        }) {
            scene.targets = Some(Targets::new(
                &self.device,
                &self.frame_layout,
                options.size * sampling,
                deferred,
                &self.resolver,
                options.size,
            ));
        }
        let targets = scene
            .targets
            .as_ref()
            .ok_or_else(|| Error::Gpu("missing render targets".into()))?;
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for transparent in [false, true] {
            let mut attachments = vec![Some(wgpu::RenderPassColorAttachment {
                view: &targets.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if transparent {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    },
                    store: wgpu::StoreOp::Store,
                },
            })];
            if !transparent && deferred {
                attachments.extend([&targets.normal_view, &targets.surface_view].map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view: view.as_ref().expect("deferred attachment"),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })
                }));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth_view,
                    depth_ops: (!transparent).then_some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &scene.camera_group, &[]);
            if transparent && let Some(frame_group) = &targets.frame_group {
                pass.set_bind_group(3, frame_group, &[]);
            }
            for &(blended, _, node, asset, chunk) in &draws {
                if blended != transparent {
                    continue;
                }
                let draw = &scene.assets[asset].draws[chunk];
                let replacement = scene.material_overrides.get(&(NodeId(node), draw.material));
                let material = replacement.map_or(
                    &scene.scene.assets[asset].appearance.materials[draw.material],
                    |m| &m.material,
                );
                let pipeline = usize::from(deferred) * 8
                    + usize::from(worlds[node].0.determinant() < 0.0) * 4
                    + usize::from(blended) * 2
                    + usize::from(material.double_sided);
                pass.set_pipeline(&self.pipelines[pipeline]);
                pass.set_bind_group(
                    1,
                    replacement.map_or(&scene.assets[asset].materials[draw.material], |m| &m.group),
                    &[],
                );
                pass.set_bind_group(2, &scene.instances[node].group, &[]);
                let vertices = scene.deformed[node]
                    .as_ref()
                    .map_or(&draw.vertices, |d| &d.draws[chunk].vertices);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(draw.indices.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..draw.count, 0, 0..1);
            }
        }
        self.resolver.encode(&mut encoder, &targets.resolved);
        encoder.copy_texture_to_buffer(
            targets.resolved.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &targets.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(targets.stride),
                    rows_per_image: Some(options.size),
                },
            },
            targets.resolved.texture.size(),
        );
        self.queue.submit([encoder.finish()]);
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(Error::Gpu(error.to_string()));
        }
        Ok(RenderedImage {
            image: targets.read(&self.device, options.size)?,
            adapter: self.adapter.clone(),
        })
    }

    fn update_deformations(
        &self,
        scene: &mut PreparedScene,
        worlds: &[(Mat4, bool)],
    ) -> Result<(), Error> {
        for node_index in 0..scene.scene.nodes.len() {
            let node = NodeId(node_index);
            let binding = scene.scene.skins.get(&node);
            let weights = scene
                .morph_weights
                .get(&node)
                .map_or(&[][..], Vec::as_slice);
            if binding.is_none() && weights.is_empty() {
                continue;
            }
            let palette: Vec<_> = if let Some(binding) = binding {
                let from_world = worlds[node.0].0.inverse();
                binding
                    .palette
                    .iter()
                    .enumerate()
                    .map(|(i, &bone)| from_world * worlds[bone.0].0 * binding.bind_from_mesh[i])
                    .collect()
            } else {
                Vec::new()
            };
            if scene.deformed[node.0]
                .as_ref()
                .is_some_and(|d| d.palette == palette && d.weights == weights)
            {
                continue;
            }
            if palette
                .iter()
                .any(|matrix| !matrix.is_finite() || matrix.determinant().abs() < 1e-12)
            {
                return Err(Error::InvalidInput("invalid skin matrix".into()));
            }
            let normal_matrices: Vec<_> = palette.iter().map(|m| m.inverse().transpose()).collect();
            let asset = scene.scene.nodes[node.0]
                .asset
                .ok_or_else(|| Error::InvalidInput("skin has no asset".into()))?;
            let mut draws = Vec::new();
            for (chunk_index, chunk) in scene.scene.assets[asset.0].chunks.iter().enumerate() {
                let mut vertices: Vec<_> = chunk
                    .positions
                    .iter()
                    .enumerate()
                    .map(|(index, &position)| Vertex {
                        position,
                        normal: chunk.normals[index],
                        uv: chunk.tex_coords[index],
                        tangent: chunk.tangents.get(index).copied().unwrap_or([0.0; 4]),
                        color: chunk.colors.get(index).copied().unwrap_or([0.; 4]),
                    })
                    .collect();
                for (target, &weight) in scene.scene.assets[asset.0]
                    .morphs
                    .iter()
                    .zip(weights)
                    .filter(|(_, weight)| **weight != 0.0)
                {
                    if let Some(change) =
                        target.chunks.iter().find(|c| c.chunk == chunk.chunk_index)
                    {
                        for (i, &vertex) in change.vertices.iter().enumerate() {
                            let vertex = &mut vertices[usize::from(vertex)];
                            for axis in 0..3 {
                                vertex.position[axis] += change.positions[i][axis] * weight;
                                vertex.normal[axis] += change.normals[i][axis] * weight;
                                vertex.tangent[axis] += change.tangents[i][axis] * weight;
                            }
                        }
                    }
                }
                for (index, vertex) in vertices.iter_mut().enumerate() {
                    if let Some(skin) = chunk.skin.get(index).filter(|_| !palette.is_empty()) {
                        let mut position = Vec3::ZERO;
                        let mut normal = Vec3::ZERO;
                        let mut tangent = Vec3::ZERO;
                        for (&joint, &weight) in skin
                            .joints
                            .iter()
                            .zip(&skin.weights)
                            .filter(|(_, weight)| **weight > 0.0)
                        {
                            position += palette[usize::from(joint)]
                                .transform_point3(vertex.position.into())
                                * weight;
                            normal += normal_matrices[usize::from(joint)]
                                .transform_vector3(vertex.normal.into())
                                * weight;
                            tangent += palette[usize::from(joint)].transform_vector3(Vec3::new(
                                vertex.tangent[0],
                                vertex.tangent[1],
                                vertex.tangent[2],
                            )) * weight;
                        }
                        let normal = normal.normalize_or_zero();
                        let tangent = (tangent - normal * normal.dot(tangent)).normalize_or_zero();
                        vertex.position = position.to_array();
                        vertex.normal = normal.to_array();
                        vertex.tangent[..3].copy_from_slice(&tangent.to_array());
                    }
                    let normal = Vec3::from(vertex.normal).normalize_or_zero();
                    let tangent =
                        Vec3::new(vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]);
                    vertex.normal = normal.to_array();
                    vertex.tangent[..3].copy_from_slice(
                        &(tangent - normal * normal.dot(tangent))
                            .normalize_or_zero()
                            .to_array(),
                    );
                }
                let indices = scene.scene.assets[asset.0].surface_indices(chunk);
                let (bounds, centroid) = vertex_bounds(&vertices, indices.as_ref())?;
                let bytes = bytemuck::cast_slice(&vertices);
                let buffer = if let Some(previous) = &scene.deformed[node.0] {
                    let buffer = previous.draws[chunk_index].vertices.clone();
                    self.queue.write_buffer(&buffer, 0, bytes);
                    buffer
                } else {
                    self.buffer(
                        "deformed vertices",
                        bytes,
                        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    )
                };
                draws.push(DeformedDraw {
                    vertices: buffer,
                    bounds,
                    side_bounds: vertex_side_bounds(&vertices, indices.as_ref()),
                    centroid,
                });
            }
            scene.deformed[node.0] = Some(Deformation {
                palette,
                weights: weights.to_vec(),
                draws,
            });
        }
        Ok(())
    }
}

fn vertex_side_bounds(vertices: &[Vertex], indices: &[u16]) -> [Option<[Vec3; 8]>; 2] {
    std::array::from_fn(|side| {
        let mut low = Vec3::splat(f32::INFINITY);
        let mut high = Vec3::splat(f32::NEG_INFINITY);
        for &index in indices {
            let p = Vec3::from_array(vertices[usize::from(index)].position);
            if (p.x >= 0.) == (side == 1) {
                low = low.min(p);
                high = high.max(p);
            }
        }
        low.is_finite().then(|| {
            std::array::from_fn(|i| {
                Vec3::new(
                    if i & 1 == 0 { low.x } else { high.x },
                    if i & 2 == 0 { low.y } else { high.y },
                    if i & 4 == 0 { low.z } else { high.z },
                )
            })
        })
    })
}

fn vertex_bounds(vertices: &[Vertex], indices: &[u16]) -> Result<([Vec3; 8], Vec3), Error> {
    let mut lower = Vec3::splat(f32::INFINITY);
    let mut upper = Vec3::splat(f32::NEG_INFINITY);
    let mut center = Vec3::ZERO;
    let mut count = 0_u32;
    for triangle in indices.chunks_exact(3) {
        let points =
            [0, 1, 2].map(|i| Vec3::from_array(vertices[usize::from(triangle[i])].position));
        // Native meshes retain unused vertices and collapsed helper triangles.
        // They must not move the camera away from the drawable surface.
        if (points[1] - points[0])
            .cross(points[2] - points[0])
            .length_squared()
            <= 1e-20
        {
            continue;
        }
        for point in points {
            if !point.is_finite() {
                return Err(Error::InvalidInput("nonfinite deformed position".into()));
            }
            lower = lower.min(point);
            upper = upper.max(point);
            center += point;
            count += 1;
        }
    }
    if count == 0 {
        return Err(Error::InvalidInput(
            "mesh has no nondegenerate surface bounds".into(),
        ));
    }
    let bounds = std::array::from_fn(|i| {
        Vec3::new(
            if i & 1 == 0 { lower.x } else { upper.x },
            if i & 2 == 0 { lower.y } else { upper.y },
            if i & 4 == 0 { lower.z } else { upper.z },
        )
    });
    Ok((bounds, center / count as f32))
}

struct Targets {
    size: u32,
    stride: u32,
    resolved: super::resolve::Resolved,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    normal_view: Option<wgpu::TextureView>,
    surface_view: Option<wgpu::TextureView>,
    frame_group: Option<wgpu::BindGroup>,
    readback: wgpu::Buffer,
}

impl Targets {
    fn new(
        device: &wgpu::Device,
        frame_layout: &wgpu::BindGroupLayout,
        size: u32,
        deferred: bool,
        resolver: &super::resolve::Resolver,
        output_size: u32,
    ) -> Self {
        let extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let make = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = make(
            "color",
            COLOR,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let color_view = color.create_view(&Default::default());
        let depth_view = make(
            "depth",
            DEPTH,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .create_view(&Default::default());
        let (normal_view, surface_view, frame_group) = if deferred {
            let normal_view = make(
                "receiving normal and roughness",
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .create_view(&Default::default());
            let surface_view = make(
                "receiving albedo and metalness",
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .create_view(&Default::default());
            let frame_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("receiving surface"),
                layout: frame_layout,
                entries: &[(&normal_view, 0), (&surface_view, 1), (&depth_view, 2)].map(
                    |(view, binding)| wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                ),
            });
            (Some(normal_view), Some(surface_view), Some(frame_group))
        } else {
            (None, None, None)
        };
        let resolved = resolver.targets(device, &color_view, size, output_size);
        let stride = (output_size * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(stride) * u64::from(output_size),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            size,
            stride,
            resolved,
            color_view,
            depth_view,
            normal_view,
            surface_view,
            frame_group,
            readback,
        }
    }

    fn read(&self, device: &wgpu::Device, size: u32) -> Result<RgbaImage, Error> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(e.to_string()))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(e.to_string()))?
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let mapped = self.readback.slice(..).get_mapped_range();
        let pixels = mapped
            .chunks_exact(self.stride as usize)
            .flat_map(|row| row[..size as usize * 4].iter().copied())
            .collect();
        drop(mapped);
        self.readback.unmap();
        RgbaImage::from_raw(size, size, pixels)
            .ok_or_else(|| Error::Gpu("invalid framebuffer dimensions".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Node;

    #[test]
    fn decal_surface_properties_are_independent_of_diffuse_alpha() {
        let renderer = pollster::block_on(Renderer::new()).unwrap();
        let mut base = super::super::tests::triangle();
        base.appearance.materials[0].base_color_factor = [0.25, 0.25, 0.25, 1.];
        base.appearance.materials[0].roughness_factor = 0.5;
        let mut scene = Scene::new();
        let asset = scene.add_asset(base);
        scene
            .add_node(Node::new("receiving surface", Some(asset)))
            .unwrap();
        let options = RenderOptions {
            size: 64,
            camera: Camera::Frame(View::Hero),
        };
        let mut prepared = renderer.prepare_scene(&scene).unwrap();
        let baseline = renderer.render(&mut prepared, options).unwrap().image;
        let mut overlay = super::super::tests::triangle();
        let material = &mut overlay.appearance.materials[0];
        material.base_color_factor = [1., 0., 0., 0.];
        material.alpha_mode = "BLEND";
        material.roughness_factor = 0.08;
        material.metallic_factor = 1.;
        material.decal = Some(ghostline_red::pbr::DecalShading {
            gradient_texture: None,
            gradient_intensity: 1.,
            gradient_uv: 1.,
            secondary_texture: None,
            secondary_color: [1.; 4],
            secondary_alpha: 0.,
            normal_alpha: 0.,
            roughness_metalness_alpha: 1.,
            depth_threshold: 1.,
            normal_blending_mode: 1.,
            normal_alpha_texture: false,
        });
        let asset = scene.add_asset(overlay);
        let node = scene
            .add_node(Node::new("surface-only decal", Some(asset)))
            .unwrap();
        let mut prepared = renderer.prepare_scene(&scene).unwrap();
        let changed = renderer.render(&mut prepared, options).unwrap().image;
        assert_ne!(
            changed, baseline,
            "surface overlay must work with zero colour coverage"
        );
        let mut disabled = scene.assets[asset.0].appearance.materials[0].clone();
        disabled.decal.as_mut().unwrap().roughness_metalness_alpha = 0.;
        renderer
            .set_material(&mut prepared, node, 0, disabled)
            .unwrap();
        let unchanged = renderer.render(&mut prepared, options).unwrap().image;
        assert_eq!(
            unchanged, baseline,
            "zero independent blend weights must preserve the receiving surface"
        );
    }

    #[test]
    fn subject_camera_ignores_distant_reference_geometry() {
        let renderer = pollster::block_on(Renderer::new()).unwrap();
        let asset = Arc::new(super::super::tests::triangle());
        let mut scene = Scene::new();
        let mesh = scene.add_shared_asset(Arc::clone(&asset));
        scene.add_node(Node::new("subject", Some(mesh))).unwrap();
        let mut original = renderer.prepare_scene(&scene).unwrap();
        let expected = renderer
            .render(
                &mut original,
                RenderOptions {
                    size: 64,
                    camera: Camera::Frame(View::Hero),
                },
            )
            .unwrap()
            .image;
        let mut reference = Node::new("reference", Some(mesh))
            .with_transform(Mat4::from_translation(Vec3::new(50., 0., 0.)));
        reference.reference = true;
        scene.add_node(reference).unwrap();
        let mut contextual = renderer.prepare_scene(&scene).unwrap();
        let actual = renderer
            .render(
                &mut contextual,
                RenderOptions {
                    size: 64,
                    camera: Camera::FrameSubject(View::Hero),
                },
            )
            .unwrap()
            .image;
        assert_eq!(actual, expected);
    }

    #[test]
    fn morph_and_material_edits_leave_shared_instances_independent() {
        use ghostline_red::morph::{MorphChunk, MorphTarget};
        let renderer = pollster::block_on(Renderer::new()).unwrap();
        let mut asset = super::super::tests::triangle();
        let mut green = asset.appearance.materials[0].clone();
        green.base_color_factor = [0.05, 0.9, 0.05, 1.];
        asset.morphs = Arc::new(vec![MorphTarget {
            name: "wider".into(),
            region: "shape".into(),
            chunks: vec![MorphChunk {
                chunk: 0,
                vertices: vec![2],
                positions: vec![[0.7, 0., 0.]],
                normals: vec![[0.; 3]],
                tangents: vec![[0.; 3]],
            }],
        }]);
        let mut scene = Scene::new();
        let mesh = scene.add_asset(asset);
        let left = scene
            .add_node(
                Node::new("left", Some(mesh))
                    .with_transform(Mat4::from_translation(Vec3::new(-1.3, 0., 0.))),
            )
            .unwrap();
        scene
            .add_node(
                Node::new("right", Some(mesh))
                    .with_transform(Mat4::from_translation(Vec3::new(1.3, 0., 0.))),
            )
            .unwrap();
        let mut prepared = renderer.prepare_scene(&scene).unwrap();
        let options = RenderOptions {
            size: 64,
            camera: Camera::Orthographic {
                eye: [0., 0., -8.],
                target: [0.; 3],
                up: [0., 1., 0.],
                half_height: 4.,
                near: 0.01,
                far: 20.,
            },
        };
        let rest = renderer.render(&mut prepared, options).unwrap().image;
        prepared.set_morph_weights(left, &[1.]).unwrap();
        let changed = renderer.render(&mut prepared, options).unwrap().image;
        assert_ne!(rest, changed);
        assert!(
            rest.enumerate_pixels()
                .filter(|(x, _, _)| *x < 24)
                .all(|(x, y, p)| changed.get_pixel(x, y) == p)
        );
        prepared.set_morph_weights(left, &[0.]).unwrap();
        assert_eq!(rest, renderer.render(&mut prepared, options).unwrap().image);
        renderer
            .set_material(&mut prepared, left, 0, green)
            .unwrap();
        let painted = renderer.render(&mut prepared, options).unwrap().image;
        assert!(
            painted
                .pixels()
                .any(|p| p.0[3] > 0 && u16::from(p.0[1]) > u16::from(p.0[0]) * 2)
        );
        assert_ne!(rest, painted);
        assert!(prepared.reset_material(left, 0));
        assert_eq!(rest, renderer.render(&mut prepared, options).unwrap().image);
        assert_eq!(prepared.statistics().geometry_chunks, 1);
    }

    #[test]
    fn skeleton_pose_deforms_skin_and_moves_rigid_attachments() {
        use ghostline_red::{
            assets::{Bone, Rig, Transform},
            mesh::{MeshSkin, VertexSkin},
        };
        let renderer = pollster::block_on(Renderer::new()).unwrap();
        let mut mesh = super::super::tests::triangle();
        mesh.skin = Some(MeshSkin {
            names: vec!["root".into(), "tip".into(), "None".into()],
            inverse_bind: vec![Mat4::IDENTITY; 3],
        });
        Arc::make_mut(&mut mesh.chunks)[0].skin = vec![
            VertexSkin {
                joints: [0; 8],
                weights: [1., 0., 0., 0., 0., 0., 0., 0.]
            };
            3
        ];
        Arc::make_mut(&mut mesh.chunks)[0].skin[2].joints[0] = 1;
        let skeleton = Arc::new(
            crate::Skeleton::from_rig(&Rig {
                a_pose: None,
                source: None,
                bones: vec![
                    Bone {
                        name: "root".into(),
                        parent: None,
                        rest: Transform::default(),
                    },
                    Bone {
                        name: "tip".into(),
                        parent: Some(0),
                        rest: Transform::default(),
                    },
                ],
            })
            .unwrap(),
        );
        let mut scene = Scene::new();
        let asset = scene.add_asset(mesh);
        let root = scene.add_node(Node::new("actor", None)).unwrap();
        let skin = scene
            .add_node(Node::new("skin", Some(asset)).with_parent(root))
            .unwrap();
        let rig = scene.add_rig(root, Arc::clone(&skeleton)).unwrap();
        scene.bind_skin(skin, rig).unwrap();
        let bone = scene.bone_node(rig, "tip").unwrap();
        let attachment = scene
            .add_node(
                Node::new("attachment", None)
                    .with_parent(bone)
                    .with_transform(Mat4::from_translation(Vec3::X)),
            )
            .unwrap();
        let mut pose = skeleton.rest_pose().clone();
        pose.local[1] = Mat4::from_translation(Vec3::new(0.6, 0.3, 0.0));
        scene.set_pose(rig, &pose).unwrap();
        assert!(
            (scene.world_transforms().unwrap()[attachment.0]
                .0
                .w_axis
                .truncate()
                - Vec3::new(1.6, 0.3, 0.0))
            .length()
                < 1e-6
        );
        scene.set_pose(rig, skeleton.rest_pose()).unwrap();
        let mut prepared = renderer.prepare_scene(&scene).unwrap();
        let options = RenderOptions {
            size: 64,
            camera: Camera::Frame(View::Front),
        };
        let rest = renderer.render(&mut prepared, options).unwrap().image;
        prepared.set_pose(rig, &pose).unwrap();
        let moved = renderer.render(&mut prepared, options).unwrap().image;
        assert_ne!(rest, moved);
        prepared.set_pose(rig, skeleton.rest_pose()).unwrap();
        assert_eq!(rest, renderer.render(&mut prepared, options).unwrap().image);
        assert_eq!(prepared.statistics().geometry_chunks, 1);
    }

    #[test]
    fn retained_instances_share_geometry_and_apply_visibility_and_reflections() {
        let renderer = pollster::block_on(Renderer::new()).unwrap();
        let mut asset = super::super::tests::triangle();
        Arc::make_mut(&mut asset.chunks)[0].indices = vec![0, 2, 1];
        asset.appearance.materials[0].double_sided = false;
        let mut scene = Scene::new();
        let mesh = scene.add_asset(asset);
        let left = scene
            .add_node(
                Node::new("left", Some(mesh))
                    .with_transform(Mat4::from_translation(Vec3::new(-1.3, 0.0, 0.0))),
            )
            .unwrap();
        let right = scene
            .add_node(Node::new("right", Some(mesh)).with_transform(
                Mat4::from_translation(Vec3::new(1.3, 0.0, 0.0))
                    * Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0)),
            ))
            .unwrap();
        let mut prepared = renderer.prepare_scene(&scene).unwrap();
        let options = RenderOptions {
            size: 64,
            camera: Camera::Frame(View::Front),
        };
        let first = renderer.render(&mut prepared, options).unwrap().image;
        assert!(
            first
                .enumerate_pixels()
                .any(|(x, _, p)| x < 32 && p.0[3] > 0)
        );
        assert!(
            first
                .enumerate_pixels()
                .any(|(x, _, p)| x > 32 && p.0[3] > 0)
        );
        assert_eq!(prepared.statistics().geometry_chunks, 1);
        assert_eq!(prepared.statistics().instances, 2);
        assert_eq!(
            first,
            renderer.render(&mut prepared, options).unwrap().image
        );
        prepared
            .set_transform(left, Mat4::from_rotation_z(0.4))
            .unwrap();
        assert_ne!(
            first,
            renderer.render(&mut prepared, options).unwrap().image
        );
        prepared.set_chunk_mask(left, 0).unwrap();
        prepared.set_visible(right, false).unwrap();
        assert!(
            renderer
                .render(&mut prepared, options)
                .unwrap()
                .image
                .pixels()
                .all(|p| p.0[3] == 0)
        );
        assert_eq!(prepared.statistics().geometry_chunks, 1);
    }
}
