//! Linear premultiplied Catmull–Rom reduction and straight-alpha PNG conversion.

pub(super) struct Resolver {
    layouts: [wgpu::BindGroupLayout; 2],
    pipelines: [wgpu::ComputePipeline; 2],
}

pub(super) struct Resolved {
    pub texture: wgpu::Texture,
    groups: [wgpu::BindGroup; 2],
    size: u32,
    source_size: u32,
}

impl Resolver {
    pub fn new(device: &wgpu::Device) -> Self {
        let layouts = [
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
        .map(|format| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("image resolve"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            })
        });
        let pipelines = std::array::from_fn(|i| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("image resolve"),
                source: wgpu::ShaderSource::Wgsl(if i == 0 {
                    include_str!("resolve.wgsl").into()
                } else {
                    include_str!("resolve_output.wgsl").into()
                }),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("image resolve"),
                bind_group_layouts: &[&layouts[i]],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("image resolve"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self { layouts, pipelines }
    }

    pub fn targets(
        &self,
        device: &wgpu::Device,
        source: &wgpu::TextureView,
        source_size: u32,
        size: u32,
    ) -> Resolved {
        let make = |height, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("resolved image"),
                size: wgpu::Extent3d {
                    width: size,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let intermediate = make(
            source_size,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let texture = make(
            size,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        let middle = intermediate.create_view(&Default::default());
        let output = texture.create_view(&Default::default());
        let groups = [(source, &middle), (&middle, &output)];
        let groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("image resolve"),
                layout: &self.layouts[i],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(groups[i].0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(groups[i].1),
                    },
                ],
            })
        });
        Resolved {
            texture,
            groups,
            size,
            source_size,
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, targets: &Resolved) {
        for (i, height) in [targets.source_size, targets.size].into_iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("image resolve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines[i]);
            pass.set_bind_group(0, &targets.groups[i], &[]);
            pass.dispatch_workgroups(targets.size.div_ceil(8), height.div_ceil(8), 1);
        }
    }
}
