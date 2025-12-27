//! Basic material with vertex colors.

use crate::core::Id;
use crate::geometry::ColorVertex;

/// A basic material using vertex colors.
pub struct BasicMaterial {
    /// Unique ID.
    id: Id,
    /// Render pipeline.
    pipeline: Option<wgpu::RenderPipeline>,
    /// Whether the material needs to rebuild the pipeline.
    needs_update: bool,
}

impl BasicMaterial {
    /// Shader source.
    const SHADER_SOURCE: &'static str = include_str!("../shaders/basic.wgsl");

    /// Create a new basic material.
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            pipeline: None,
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

    /// Build the render pipeline.
    pub fn build_pipeline(
        &mut self,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Basic Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SOURCE.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Basic Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Basic Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[ColorVertex::layout()],
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
        self.needs_update = false;
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

impl Default for BasicMaterial {
    fn default() -> Self {
        Self::new()
    }
}
