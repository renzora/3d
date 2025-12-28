//! Skybox rendering pass.
//!
//! Renders a cubemap as the scene background before other objects.

use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// Skybox uniform data.
/// Must be 96 bytes to match WGSL alignment (vec3 requires 16-byte alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SkyboxUniform {
    /// Inverse view-projection matrix (64 bytes).
    pub inv_view_proj: [[f32; 4]; 4],
    /// Exposure multiplier (4 bytes).
    pub exposure: f32,
    /// Padding to reach 96 bytes (28 bytes).
    pub _padding: [f32; 7],
}

impl Default for SkyboxUniform {
    fn default() -> Self {
        Self {
            inv_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            exposure: 1.0,
            _padding: [0.0; 7],
        }
    }
}

/// Skybox rendering pass.
pub struct SkyboxPass {
    enabled: bool,
    exposure: f32,
    pipeline: wgpu::RenderPipeline,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl SkyboxPass {
    /// Create a new skybox pass.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        // Load skybox shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/skybox.wgsl").into()),
        });

        // Uniform bind group layout (group 0)
        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skybox Uniform Bind Group Layout"),
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

        // Texture bind group layout (group 1)
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skybox Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffer - full-screen triangle
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None, // No blending - skybox is first
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for full-screen triangle
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Write depth = 1.0 (far plane) but don't test
                // This ensures skybox is at the back
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create uniform buffer
        let uniform = SkyboxUniform::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Skybox Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create uniform bind group
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skybox Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            enabled: true,
            exposure: 1.0,
            pipeline,
            uniform_bind_group_layout,
            texture_bind_group_layout,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Create a texture bind group for the skybox cubemap.
    pub fn create_texture_bind_group(
        &self,
        device: &wgpu::Device,
        cube_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skybox Texture Bind Group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(cube_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Update the uniform buffer with the inverse view-projection matrix.
    pub fn update_uniform(&self, queue: &wgpu::Queue, inv_view_proj: [[f32; 4]; 4]) {
        let uniform = SkyboxUniform {
            inv_view_proj,
            exposure: self.exposure,
            _padding: [0.0; 7],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Render the skybox.
    /// This should be called at the start of the main render pass.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        texture_bind_group: &'a wgpu::BindGroup,
    ) {
        if !self.enabled {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_bind_group(1, texture_bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Full-screen triangle
    }

    /// Check if the skybox is enabled.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable the skybox.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get the current exposure.
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    /// Set the exposure multiplier.
    pub fn set_exposure(&mut self, exposure: f32) {
        self.exposure = exposure.max(0.0);
    }
}
