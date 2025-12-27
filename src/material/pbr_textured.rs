//! Textured PBR (Physically Based Rendering) material.

use crate::core::Id;
use crate::geometry::Vertex;
use crate::math::Color;
use crate::texture::{Sampler, Texture2D};
use bytemuck::{Pod, Zeroable};

/// Material uniform for textured PBR.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct TexturedPbrMaterialUniform {
    /// Base color (albedo) - used when no texture.
    pub base_color: [f32; 4],
    /// Metallic factor.
    pub metallic: f32,
    /// Roughness factor.
    pub roughness: f32,
    /// Ambient occlusion.
    pub ao: f32,
    /// Whether to use albedo map (1.0 = yes).
    pub use_albedo_map: f32,
    /// Whether to use normal map (1.0 = yes).
    pub use_normal_map: f32,
    /// Whether to use metallic-roughness map (1.0 = yes).
    pub use_metallic_roughness_map: f32,
    /// Padding.
    pub _padding: [f32; 2],
}

impl Default for TexturedPbrMaterialUniform {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            use_albedo_map: 0.0,
            use_normal_map: 0.0,
            use_metallic_roughness_map: 0.0,
            _padding: [0.0; 2],
        }
    }
}

/// A textured PBR material supporting albedo, normal, and metallic-roughness maps.
pub struct TexturedPbrMaterial {
    /// Unique ID.
    id: Id,
    /// Base color (used when no albedo texture).
    pub color: Color,
    /// Metallic factor (used when no metallic-roughness texture).
    pub metallic: f32,
    /// Roughness factor (used when no metallic-roughness texture).
    pub roughness: f32,
    /// Ambient occlusion.
    pub ao: f32,
    /// Albedo texture.
    albedo_texture: Option<Texture2D>,
    /// Normal map texture.
    normal_texture: Option<Texture2D>,
    /// Metallic-roughness texture (G=roughness, B=metallic per glTF spec).
    metallic_roughness_texture: Option<Texture2D>,
    /// Render pipeline.
    pipeline: Option<wgpu::RenderPipeline>,
    /// Camera bind group layout.
    camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Model bind group layout.
    model_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Material bind group layout.
    material_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Texture + shadow bind group layout (combined due to WebGPU 4 bind group limit).
    texture_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Whether the material needs to rebuild.
    needs_update: bool,
}

impl Default for TexturedPbrMaterial {
    fn default() -> Self {
        Self::new()
    }
}

impl TexturedPbrMaterial {
    /// Shader source.
    const SHADER_SOURCE: &'static str = include_str!("../shaders/pbr_textured.wgsl");

    /// Create a new textured PBR material.
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            color: Color::WHITE,
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            albedo_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            pipeline: None,
            camera_bind_group_layout: None,
            model_bind_group_layout: None,
            material_bind_group_layout: None,
            texture_bind_group_layout: None,
            needs_update: true,
        }
    }

    /// Set the albedo texture.
    pub fn set_albedo_texture(&mut self, texture: Texture2D) {
        self.albedo_texture = Some(texture);
        self.needs_update = true;
    }

    /// Set the normal map texture.
    pub fn set_normal_texture(&mut self, texture: Texture2D) {
        self.normal_texture = Some(texture);
        self.needs_update = true;
    }

    /// Set the metallic-roughness texture.
    pub fn set_metallic_roughness_texture(&mut self, texture: Texture2D) {
        self.metallic_roughness_texture = Some(texture);
        self.needs_update = true;
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the render pipeline.
    #[inline]
    pub fn pipeline(&self) -> Option<&wgpu::RenderPipeline> {
        self.pipeline.as_ref()
    }

    /// Get camera bind group layout.
    #[inline]
    pub fn camera_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.camera_bind_group_layout.as_ref()
    }

    /// Get model bind group layout.
    #[inline]
    pub fn model_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.model_bind_group_layout.as_ref()
    }

    /// Get material bind group layout.
    #[inline]
    pub fn material_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.material_bind_group_layout.as_ref()
    }

    /// Get texture bind group layout (includes shadow bindings).
    #[inline]
    pub fn texture_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.texture_bind_group_layout.as_ref()
    }

    /// Get the material uniform data.
    pub fn uniform(&self) -> TexturedPbrMaterialUniform {
        TexturedPbrMaterialUniform {
            base_color: [self.color.r, self.color.g, self.color.b, 1.0],
            metallic: self.metallic,
            roughness: self.roughness,
            ao: self.ao,
            use_albedo_map: if self.albedo_texture.is_some() { 1.0 } else { 0.0 },
            use_normal_map: if self.normal_texture.is_some() { 1.0 } else { 0.0 },
            use_metallic_roughness_map: if self.metallic_roughness_texture.is_some() { 1.0 } else { 0.0 },
            _padding: [0.0; 2],
        }
    }

    /// Build the render pipeline.
    pub fn build_pipeline(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Textured PBR Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SOURCE.into()),
        });

        // Camera bind group layout (group 0)
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Textured PBR Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Model bind group layout (group 1)
        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Textured PBR Model Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Material bind group layout (group 2)
        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Textured PBR Material Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Texture + Shadow bind group layout (group 3)
        // Combined to stay within WebGPU's 4 bind group limit
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Textured PBR Texture+Shadow Bind Group Layout"),
                entries: &[
                    // Albedo texture (binding 0)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Albedo sampler (binding 1)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Normal texture (binding 2)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Normal sampler (binding 3)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Metallic-roughness texture (binding 4)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Metallic-roughness sampler (binding 5)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Shadow uniform buffer (binding 6)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Shadow map texture (binding 7)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Shadow comparison sampler (binding 8)
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Textured PBR Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &model_bind_group_layout,
                &material_bind_group_layout,
                &texture_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Textured PBR Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        self.pipeline = Some(pipeline);
        self.camera_bind_group_layout = Some(camera_bind_group_layout);
        self.model_bind_group_layout = Some(model_bind_group_layout);
        self.material_bind_group_layout = Some(material_bind_group_layout);
        self.texture_bind_group_layout = Some(texture_bind_group_layout);
        self.needs_update = false;
    }

    /// Create a combined texture + shadow bind group.
    pub fn create_texture_shadow_bind_group(
        &self,
        device: &wgpu::Device,
        albedo: &Texture2D,
        albedo_sampler: &Sampler,
        normal: &Texture2D,
        normal_sampler: &Sampler,
        metallic_roughness: &Texture2D,
        metallic_roughness_sampler: &Sampler,
        shadow_uniform_buffer: &wgpu::Buffer,
        shadow_map_view: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
    ) -> Option<wgpu::BindGroup> {
        self.texture_bind_group_layout.as_ref().map(|layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Textured PBR Texture+Shadow Bind Group"),
                layout,
                entries: &[
                    // Textures (bindings 0-5)
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(albedo.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(albedo_sampler.wgpu_sampler()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(normal.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(normal_sampler.wgpu_sampler()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(metallic_roughness.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(metallic_roughness_sampler.wgpu_sampler()),
                    },
                    // Shadow (bindings 6-8)
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: shadow_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(shadow_map_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Sampler(shadow_sampler),
                    },
                ],
            })
        })
    }

    /// Create a camera bind group.
    pub fn create_camera_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        self.camera_bind_group_layout.as_ref().map(|layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Textured PBR Camera Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
    }

    /// Create a model bind group.
    pub fn create_model_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        self.model_bind_group_layout.as_ref().map(|layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Textured PBR Model Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
    }

    /// Create a material bind group.
    pub fn create_material_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        self.material_bind_group_layout.as_ref().map(|layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Textured PBR Material Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
    }


    /// Check if needs update.
    #[inline]
    pub fn needs_update(&self) -> bool {
        self.needs_update || self.pipeline.is_none()
    }
}
