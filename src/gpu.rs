use std::{path::Path, sync::mpsc};

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use image::RgbaImage;
use wgpu::util::DeviceExt;

use crate::{Camera, Error, MeshAsset};

mod resolve;
mod retained;
mod textures;
pub use retained::{PreparedScene, SceneStatistics};

const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Camera direction in the mesh's Y-up coordinates; character front is -Z.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum View {
    /// Three-quarter front view.
    #[default]
    Hero,
    /// Front view.
    Front,
    /// Rear view.
    Back,
    /// Left profile view.
    Left,
    /// Right profile view.
    Right,
    /// Profile from the side occupied by the framed subject.
    Side,
}

/// Settings for a square transparent preview.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct RenderOptions {
    /// Final width and height in pixels, between 64 and 4096.
    pub size: u32,
    /// Camera direction.
    pub camera: Camera,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            size: 512,
            camera: Camera::default(),
        }
    }
}

/// Top-left-origin RGBA preview and the adapter used to produce it.
pub struct RenderedImage {
    /// Final image with transparent background.
    pub image: RgbaImage,
    /// Adapter and backend information.
    pub adapter: wgpu::AdapterInfo,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    projection: [[f32; 4]; 4],
    eye: [f32; 4],
    ambient: [f32; 4],
    counts: [u32; 4],
    lights: [GpuLight; 64],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLight {
    position: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Material {
    color: [f32; 4],
    surface: [f32; 4],
    emission: [f32; 4],
    settings: [f32; 4],
    decal: [f32; 4],
    decal_blend: [f32; 4],
    decal_layers: [f32; 4],
    secondary_color: [f32; 4],
    uv_transform: [f32; 4],
}

struct Draw {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    count: u32,
    material: usize,
    chunk: usize,
    centroid: Vec3,
    bounds: [Vec3; 8],
    side_bounds: [Option<[Vec3; 8]>; 2],
}

/// Persistent offscreen GPU device for mesh previews.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::AdapterInfo,
    globals_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::RenderPipeline>,
    instance_layout: wgpu::BindGroupLayout,
    frame_layout: wgpu::BindGroupLayout,
    owner: std::sync::Arc<()>,
    textures: std::sync::Mutex<textures::TextureCache>,
    resolver: resolve::Resolver,
    assets: std::sync::Mutex<retained::AssetCache>,
}

impl Renderer {
    /// GPU, backend and driver used by this retained renderer.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter
    }
    /// Creates a hardware GPU device and compiles the preview shader.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Gpu`] if no compatible adapter/device exists or shader
    /// validation fails.
    pub async fn new() -> Result<Self, Error> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Ghostline renderer"),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT)],
        });
        let mut entries = vec![uniform_entry(0, wgpu::ShaderStages::FRAGMENT)];
        for binding in 1..=4 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        for binding in [7, 8] {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &entries,
        });
        let instance_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("receiving surface"),
            entries: &(0..3)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: if binding == 2 {
                            wgpu::TextureSampleType::Depth
                        } else {
                            wgpu::TextureSampleType::Float { filterable: false }
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview"),
            bind_group_layouts: &[&globals_layout, &material_layout, &instance_layout],
            immediate_size: 0,
        });
        let blended_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("surface decals and transparency"),
            bind_group_layouts: &[
                &globals_layout,
                &material_layout,
                &instance_layout,
                &frame_layout,
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("studio"),
            source: wgpu::ShaderSource::Wgsl(include_str!("studio.wgsl").into()),
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4, 4 => Float32x4];
        let mut pipelines = Vec::new();
        for deferred in [false, true] {
            for mirrored in [false, true] {
                for blended in [false, true] {
                    for double_sided in [false, true] {
                        let mut targets = vec![Some(wgpu::ColorTargetState {
                            format: COLOR,
                            blend: blended.then_some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })];
                        if !blended && deferred {
                            targets.extend((0..2).map(|_| {
                                Some(wgpu::ColorTargetState {
                                    format: wgpu::TextureFormat::Rgba16Float,
                                    blend: None,
                                    write_mask: wgpu::ColorWrites::ALL,
                                })
                            }));
                        }
                        pipelines.push(device.create_render_pipeline(
                            &wgpu::RenderPipelineDescriptor {
                                label: Some("studio"),
                                layout: Some(if blended && deferred {
                                    &blended_layout
                                } else {
                                    &layout
                                }),
                                vertex: wgpu::VertexState {
                                    module: &shader,
                                    entry_point: Some("vertex_main"),
                                    compilation_options: Default::default(),
                                    buffers: &[wgpu::VertexBufferLayout {
                                        array_stride: size_of::<Vertex>() as u64,
                                        step_mode: wgpu::VertexStepMode::Vertex,
                                        attributes: &attributes,
                                    }],
                                },
                                primitive: wgpu::PrimitiveState {
                                    front_face: if mirrored {
                                        wgpu::FrontFace::Cw
                                    } else {
                                        wgpu::FrontFace::Ccw
                                    },
                                    cull_mode: if double_sided {
                                        None
                                    } else {
                                        Some(wgpu::Face::Back)
                                    },
                                    ..Default::default()
                                },
                                depth_stencil: Some(wgpu::DepthStencilState {
                                    format: DEPTH,
                                    depth_write_enabled: !blended,
                                    depth_compare: wgpu::CompareFunction::LessEqual,
                                    stencil: Default::default(),
                                    bias: Default::default(),
                                }),
                                multisample: Default::default(),
                                fragment: Some(wgpu::FragmentState {
                                    module: &shader,
                                    entry_point: Some(if !deferred {
                                        "forward_main"
                                    } else if blended {
                                        "fragment_main"
                                    } else {
                                        "opaque_main"
                                    }),
                                    compilation_options: Default::default(),
                                    targets: &targets,
                                }),
                                multiview_mask: None,
                                cache: None,
                            },
                        ));
                    }
                }
            }
        }
        let resolver = resolve::Resolver::new(&device);
        if let Some(error) = error_scope.pop().await {
            return Err(Error::Gpu(error.to_string()));
        }
        Ok(Self {
            resolver,
            assets: std::sync::Mutex::new(retained::AssetCache::default()),
            textures: std::sync::Mutex::new(textures::TextureCache::default()),
            device,
            queue,
            adapter: info,
            globals_layout,
            material_layout,
            pipelines,
            instance_layout,
            frame_layout,
            owner: std::sync::Arc::new(()),
        })
    }

    fn buffer(&self, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
    }

    fn material(
        &self,
        material: &ghostline_red::pbr::PbrMaterial,
    ) -> Result<wgpu::BindGroup, Error> {
        if material
            .base_color_factor
            .iter()
            .chain(&material.emissive_factor)
            .any(|v| !v.is_finite() || *v < 0.)
            || !material.roughness_factor.is_finite()
            || !(0. ..=1.).contains(&material.roughness_factor)
            || !material.metallic_factor.is_finite()
            || !(0. ..=1.).contains(&material.metallic_factor)
            || !material.emissive_strength.is_finite()
            || material.emissive_strength < 0.
            || material.uv_transform.iter().any(|v| !v.is_finite())
            || material.hair.as_ref().is_some_and(|hair| {
                [hair.flow_strength, hair.vertex_color_strength]
                    .iter()
                    .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
            })
            || material.decal.as_ref().is_some_and(|decal| {
                [
                    decal.normal_alpha,
                    decal.roughness_metalness_alpha,
                    decal.depth_threshold,
                    decal.normal_blending_mode,
                    decal.gradient_intensity,
                    decal.gradient_uv,
                    decal.secondary_alpha,
                ]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.)
                    || decal.normal_alpha > 1.
                    || decal.roughness_metalness_alpha > 1.
                    || decal.secondary_alpha > 1.
                    || decal
                        .secondary_color
                        .iter()
                        .any(|value| !value.is_finite() || *value < 0.)
            })
            || material
                .alpha_cutoff
                .is_some_and(|v| !v.is_finite() || !(0. ..=1.).contains(&v))
        {
            return Err(Error::InvalidInput("invalid material parameters".into()));
        }
        let mode = match material.alpha_mode {
            "OPAQUE" => 0.0,
            "MASK" => 1.0,
            "BLEND" => 2.0,
            other => {
                return Err(Error::InvalidInput(format!(
                    "unsupported alpha mode {other}"
                )));
            }
        };
        let parameters = Material {
            uv_transform: material.uv_transform,
            decal_layers: material.decal.as_ref().map_or([0.; 4], |d| {
                [
                    f32::from(d.gradient_texture.is_some()),
                    d.gradient_intensity,
                    d.gradient_uv,
                    d.secondary_alpha * f32::from(d.secondary_texture.is_some()),
                ]
            }),
            secondary_color: material
                .decal
                .as_ref()
                .map_or([1.; 4], |d| d.secondary_color),
            color: material.base_color_factor,
            surface: [
                material.roughness_factor,
                material.metallic_factor,
                material.alpha_cutoff.unwrap_or(0.5),
                if material.stochastic_coverage && mode == 2.0 {
                    3.0
                } else {
                    mode
                },
            ],
            emission: [
                material.emissive_factor[0],
                material.emissive_factor[1],
                material.emissive_factor[2],
                material.emissive_strength,
            ],
            settings: [
                f32::from(material.normal_texture.is_some()),
                f32::from(material.hair.is_some()),
                material
                    .hair
                    .as_ref()
                    .map_or(0., |h| h.vertex_color_strength),
                material.hair.as_ref().map_or(0., |h| h.flow_strength),
            ],
            decal: material.decal.as_ref().map_or([0.; 4], |d| {
                [
                    1.,
                    d.normal_alpha,
                    d.roughness_metalness_alpha,
                    d.depth_threshold,
                ]
            }),
            decal_blend: material.decal.as_ref().map_or([0.; 4], |d| {
                [
                    d.normal_blending_mode,
                    f32::from(d.normal_alpha_texture),
                    0.,
                    0.,
                ]
            }),
        };
        let buffer = self.buffer(
            "material",
            bytemuck::bytes_of(&parameters),
            wgpu::BufferUsages::UNIFORM,
        );
        let maps = [
            self.texture(material.base_color_texture.as_deref(), [255; 4], true)?,
            self.texture(
                material.metallic_roughness_texture.as_deref(),
                [255; 4],
                false,
            )?,
            self.texture(
                material.normal_texture.as_deref(),
                [128, 128, 255, 255],
                false,
            )?,
            self.texture(material.emissive_texture.as_deref(), [255; 4], true)?,
        ];
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }];
        for (index, view) in maps.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: index as u32 + 1,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::Sampler(&sampler),
        });
        let flow = self.texture(
            material
                .hair
                .as_ref()
                .and_then(|h| h.flow_texture.as_deref()),
            [128, 255, 128, 255],
            false,
        )?;
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::TextureView(&flow),
        });
        let gradient = self.texture(
            material
                .decal
                .as_ref()
                .and_then(|d| d.gradient_texture.as_deref()),
            [255; 4],
            true,
        )?;
        let secondary = self.texture(
            material
                .decal
                .as_ref()
                .and_then(|d| d.secondary_texture.as_deref()),
            [255, 255, 255, 0],
            true,
        )?;
        entries.push(wgpu::BindGroupEntry {
            binding: 7,
            resource: wgpu::BindingResource::TextureView(&gradient),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::TextureView(&secondary),
        });
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: &self.material_layout,
            entries: &entries,
        }))
    }

    fn texture(
        &self,
        path: Option<&Path>,
        fallback: [u8; 4],
        srgb: bool,
    ) -> Result<wgpu::TextureView, Error> {
        let file_key = path
            .map(|path| textures::FileKey::new(path, srgb))
            .transpose()?;
        if let Some(key) = &file_key
            && let Some(view) = self
                .textures
                .lock()
                .map_err(|_| Error::Gpu("texture cache lock poisoned".into()))?
                .get_file(key)
        {
            return Ok(view);
        }
        let image = if let Some(path) = path {
            image::open(path)?.into_rgba8()
        } else {
            RgbaImage::from_pixel(1, 1, image::Rgba(fallback))
        };
        let size = wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        };
        let key = textures::TextureKey::new(&image, srgb);
        {
            let mut cache = self
                .textures
                .lock()
                .map_err(|_| Error::Gpu("texture cache lock poisoned".into()))?;
            if let Some(view) = cache.get(&key) {
                if let Some(file) = file_key {
                    cache.remember_file(file, key);
                }
                return Ok(view);
            }
        }
        if size.width > self.device.limits().max_texture_dimension_2d
            || size.height > self.device.limits().max_texture_dimension_2d
        {
            return Err(Error::InvalidInput(
                "material texture exceeds GPU limits".into(),
            ));
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material map"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if srgb {
                COLOR
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            image.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(image.width() * 4),
                rows_per_image: Some(image.height()),
            },
            size,
        );
        let view = texture.create_view(&Default::default());
        let mut cache = self
            .textures
            .lock()
            .map_err(|_| Error::Gpu("texture cache lock poisoned".into()))?;
        cache.insert(key, image.as_raw().len(), view.clone());
        if let Some(file) = file_key {
            cache.remember_file(file, key);
        }
        Ok(view)
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ghostline_red::{
        mesh::MeshChunk,
        pbr::{BakeSummary, BakedAppearance, PbrMaterial},
    };

    pub(crate) fn triangle() -> MeshAsset {
        MeshAsset {
            _preparation: None,
            _material_repository: None,
            shadow_only: false,
            morphs: std::sync::Arc::new(Vec::new()),
            skin: None,
            chunks: std::sync::Arc::new(vec![MeshChunk {
                skin: Vec::new(),
                chunk_index: 0,
                name: "triangle".into(),
                positions: vec![[-1.0, -0.5, 0.0], [0.8, -0.5, 0.1], [-0.3, 1.0, 0.0]],
                normals: vec![[0.0, 0.0, -1.0]; 3],
                tangents: vec![],
                colors: vec![],
                tex_coords: vec![[0.0, 0.0]; 3],
                tex_coords_1: vec![],
                indices: vec![0, 1, 2],
                material_names: vec!["solid".into()],
            }]),
            appearance: BakedAppearance {
                chunk_materials: vec!["solid".into()],
                materials: vec![PbrMaterial {
                    name: "solid".into(),
                    base_color_texture: None,
                    base_color_factor: [0.8, 0.2, 0.1, 1.0],
                    uv_transform: [1., 1., 0., 0.],
                    metallic_roughness_texture: None,
                    metallic_factor: 0.0,
                    roughness_factor: 0.5,
                    normal_texture: None,
                    emissive_texture: None,
                    emissive_factor: [0.0; 3],
                    emissive_strength: 1.0,
                    alpha_mode: "OPAQUE",
                    alpha_cutoff: None,
                    stochastic_coverage: false,
                    hair: None,
                    decal: None,
                    transmission: 0.0,
                    double_sided: true,
                    preview_notes: vec![],
                }],
                summary: BakeSummary::default(),
            },
            source_sha256: String::new(),
            mesh: "test.mesh".into(),
            appearance_name: "solid".into(),
        }
    }
}
