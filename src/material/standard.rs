//! Standard material with lighting.

use crate::core::Id;
use crate::geometry::Vertex;
use crate::math::Color;
use bytemuck::{Pod, Zeroable};

/// Camera uniform data.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct CameraUniform {
    /// Combined view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
}

/// Model uniform data.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ModelUniform {
    /// Model matrix.
    pub model: [[f32; 4]; 4],
    /// Normal matrix (inverse transpose of model, padded to 3x4 for alignment).
    pub normal: [[f32; 4]; 3],
}

/// A standard material with lighting support.
pub struct StandardMaterial {
    /// Unique ID.
    id: Id,
    /// Base color.
    pub color: Color,
    /// Render pipeline.
    pipeline: Option<wgpu::RenderPipeline>,
    /// Camera bind group layout.
    camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Model bind group layout.
    model_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Whether the material needs to rebuild the pipeline.
    needs_update: bool,
}

impl Default for StandardMaterial {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardMaterial {
    /// Shader source.
    const SHADER_SOURCE: &'static str = include_str!("../shaders/standard.wgsl");

    /// Create a new standard material.
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            color: Color::WHITE,
            pipeline: None,
            camera_bind_group_layout: None,
            model_bind_group_layout: None,
            needs_update: true,
        }
    }

    /// Create with a specific color.
    pub fn with_color(color: Color) -> Self {
        Self {
            color,
            ..Self::new()
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

    /// Get the camera bind group layout.
    #[inline]
    pub fn camera_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.camera_bind_group_layout.as_ref()
    }

    /// Get the model bind group layout.
    #[inline]
    pub fn model_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.model_bind_group_layout.as_ref()
    }

    /// Build the render pipeline.
    pub fn build_pipeline(
        &mut self,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Standard Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SOURCE.into()),
        });

        // Camera uniform bind group layout
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
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

        // Model uniform bind group layout
        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Model Bind Group Layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Standard Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &model_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Standard Render Pipeline"),
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
                label: Some("Camera Bind Group"),
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
                label: Some("Model Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
    }

    /// Check if the pipeline needs rebuilding.
    #[inline]
    pub fn needs_update(&self) -> bool {
        self.needs_update || self.pipeline.is_none()
    }

    /// Mark the material as needing an update.
    pub fn mark_needs_update(&mut self) {
        self.needs_update = true;
    }
}
