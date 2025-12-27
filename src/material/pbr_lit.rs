//! PBR material with dynamic lighting support.

use crate::core::Id;
use crate::geometry::Vertex;

/// A PBR material with dynamic lighting support.
pub struct LitPbrMaterial {
    /// Unique ID.
    id: Id,
    /// Render pipeline.
    pipeline: Option<wgpu::RenderPipeline>,
    /// Camera bind group layout (group 0).
    camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Model bind group layout (group 1).
    model_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Material bind group layout (group 2).
    material_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Lights bind group layout (group 3).
    lights_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Whether the material needs to rebuild.
    needs_update: bool,
}

impl Default for LitPbrMaterial {
    fn default() -> Self {
        Self::new()
    }
}

impl LitPbrMaterial {
    /// Shader source.
    const SHADER_SOURCE: &'static str = include_str!("../shaders/pbr_lit.wgsl");

    /// Create a new lit PBR material.
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            pipeline: None,
            camera_bind_group_layout: None,
            model_bind_group_layout: None,
            material_bind_group_layout: None,
            lights_bind_group_layout: None,
            needs_update: true,
        }
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

    /// Get lights bind group layout.
    #[inline]
    pub fn lights_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.lights_bind_group_layout.as_ref()
    }

    /// Build the render pipeline.
    pub fn build_pipeline(
        &mut self,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lit PBR Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SOURCE.into()),
        });

        // Camera bind group layout (group 0)
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lit PBR Camera Bind Group Layout"),
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
                label: Some("Lit PBR Model Bind Group Layout"),
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
                label: Some("Lit PBR Material Bind Group Layout"),
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

        // Lights bind group layout (group 3)
        let lights_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lit PBR Lights Bind Group Layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lit PBR Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &model_bind_group_layout,
                &material_bind_group_layout,
                &lights_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lit PBR Render Pipeline"),
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
        self.lights_bind_group_layout = Some(lights_bind_group_layout);
        self.needs_update = false;
    }

    /// Create a camera bind group.
    pub fn create_camera_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        self.camera_bind_group_layout.as_ref().map(|layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Lit PBR Camera Bind Group"),
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
                label: Some("Lit PBR Model Bind Group"),
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
                label: Some("Lit PBR Material Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
    }

    /// Create a lights bind group.
    pub fn create_lights_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        self.lights_bind_group_layout.as_ref().map(|layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Lit PBR Lights Bind Group"),
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
