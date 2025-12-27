//! Shadow rendering pass (depth-only).

use crate::geometry::Vertex;
use crate::math::{Matrix4, Vector3};

use super::ShadowConfig;

/// Depth-only shader source for shadow pass.
const SHADOW_SHADER: &str = r#"
// Shadow depth shader - renders depth from light's perspective

struct LightCamera {
    view_proj: mat4x4<f32>,
}

struct Model {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> light_camera: LightCamera;

@group(1) @binding(0)
var<uniform> model: Model;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model.model * vec4<f32>(in.position, 1.0);
    out.clip_position = light_camera.view_proj * world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) {
    // Depth is automatically written
}
"#;

/// Shadow pass for rendering shadow maps.
pub struct ShadowPass {
    /// Depth-only render pipeline.
    pipeline: wgpu::RenderPipeline,
    /// Light camera bind group layout.
    light_camera_layout: wgpu::BindGroupLayout,
    /// Model bind group layout.
    model_layout: wgpu::BindGroupLayout,
    /// Light camera uniform buffer.
    light_camera_buffer: wgpu::Buffer,
    /// Light camera bind group.
    light_camera_bind_group: wgpu::BindGroup,
    /// Configuration.
    config: ShadowConfig,
}

impl ShadowPass {
    /// Create a new shadow pass.
    pub fn new(device: &wgpu::Device, config: ShadowConfig) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        });

        // Light camera bind group layout (group 0)
        let light_camera_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow Light Camera Layout"),
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

        // Model bind group layout (group 1)
        let model_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Model Layout"),
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
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&light_camera_layout, &model_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
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
                targets: &[], // No color targets
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
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,      // Constant depth bias
                    slope_scale: 2.0, // Slope-scale bias
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create light camera uniform buffer
        let light_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Light Camera Buffer"),
            size: 64, // mat4x4
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create light camera bind group
        let light_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Light Camera Bind Group"),
            layout: &light_camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_camera_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            light_camera_layout,
            model_layout,
            light_camera_buffer,
            light_camera_bind_group,
            config,
        }
    }

    /// Calculate light-space matrix for a directional light.
    ///
    /// Creates an orthographic projection that encompasses the scene.
    pub fn calculate_directional_matrix(
        light_direction: &Vector3,
        scene_center: &Vector3,
        scene_radius: f32,
    ) -> Matrix4 {
        // Position light far enough to cover the scene
        let light_pos = *scene_center - *light_direction * scene_radius * 2.0;

        // Create view matrix looking at scene center
        let view = Matrix4::look_at(&light_pos, scene_center, &Vector3::UP);

        // Create orthographic projection that encompasses the scene
        let proj = Matrix4::orthographic(
            -scene_radius,
            scene_radius,
            -scene_radius,
            scene_radius,
            0.1,
            scene_radius * 4.0,
        );

        proj.multiply(&view)
    }

    /// Calculate light-space matrix for a spot light.
    pub fn calculate_spot_matrix(
        position: &Vector3,
        direction: &Vector3,
        outer_angle: f32,
        range: f32,
    ) -> Matrix4 {
        let target = *position + *direction;
        let view = Matrix4::look_at(position, &target, &Vector3::UP);
        let proj = Matrix4::perspective(outer_angle * 2.0, 1.0, 0.1, range);
        proj.multiply(&view)
    }

    /// Update the light camera matrix.
    pub fn set_light_matrix(&self, queue: &wgpu::Queue, matrix: &Matrix4) {
        let data = matrix.to_cols_array_2d();
        queue.write_buffer(&self.light_camera_buffer, 0, bytemuck::bytes_of(&data));
    }

    /// Get the render pipeline.
    #[inline]
    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    /// Get the light camera bind group layout.
    #[inline]
    pub fn light_camera_layout(&self) -> &wgpu::BindGroupLayout {
        &self.light_camera_layout
    }

    /// Get the model bind group layout.
    #[inline]
    pub fn model_layout(&self) -> &wgpu::BindGroupLayout {
        &self.model_layout
    }

    /// Get the light camera bind group.
    #[inline]
    pub fn light_camera_bind_group(&self) -> &wgpu::BindGroup {
        &self.light_camera_bind_group
    }

    /// Get the configuration.
    #[inline]
    pub fn config(&self) -> &ShadowConfig {
        &self.config
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: ShadowConfig) {
        self.config = config;
    }
}
