//! Vignette post-processing effect.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// Vignette settings.
#[derive(Debug, Clone)]
pub struct VignetteSettings {
    /// Intensity of the vignette (0.0 = none, 1.0 = strong).
    pub intensity: f32,
    /// Smoothness of the falloff (higher = softer edge).
    pub smoothness: f32,
    /// Roundness (0.0 = rectangular, 1.0 = circular).
    pub roundness: f32,
}

impl Default for VignetteSettings {
    fn default() -> Self {
        Self {
            intensity: 0.3,
            smoothness: 0.5,
            roundness: 1.0,
        }
    }
}

/// Vignette uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VignetteUniform {
    /// intensity, smoothness, roundness, aspect
    params: [f32; 4],
}

/// Vignette post-processing pass.
pub struct VignettePass {
    enabled: bool,
    settings: VignetteSettings,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    sampler: Option<wgpu::Sampler>,
}

impl VignettePass {
    /// Create a new vignette pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: VignetteSettings::default(),
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            pipeline: None,
            bind_group_layout: None,
            uniform_buffer: None,
            quad_buffer: None,
            sampler: None,
        }
    }

    /// Get settings.
    pub fn settings(&self) -> &VignetteSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: VignetteSettings) {
        self.settings = settings;
    }

    /// Set intensity.
    pub fn set_intensity(&mut self, intensity: f32) {
        self.settings.intensity = intensity.clamp(0.0, 1.0);
    }

    /// Set smoothness.
    pub fn set_smoothness(&mut self, smoothness: f32) {
        self.settings.smoothness = smoothness.clamp(0.01, 2.0);
    }

    /// Set roundness.
    pub fn set_roundness(&mut self, roundness: f32) {
        self.settings.roundness = roundness.clamp(0.0, 1.0);
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.format = format;

        // Create sampler
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Vignette Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Create bind group layout
        self.bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vignette Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        }));

        // Create uniform buffer
        let aspect = width as f32 / height as f32;
        let uniform = VignetteUniform {
            params: [self.settings.intensity, self.settings.smoothness, self.settings.roundness, aspect],
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vignette Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vignette Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipeline
        self.create_pipeline(device, format);
    }

    fn create_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vignette Pipeline Layout"),
            bind_group_layouts: &[self.bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vignette Shader"),
            source: wgpu::ShaderSource::Wgsl(VIGNETTE_SHADER.into()),
        });

        self.pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Vignette Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        }));
    }

    fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let aspect = self.width as f32 / self.height as f32;
            let uniform = VignetteUniform {
                params: [self.settings.intensity, self.settings.smoothness, self.settings.roundness, aspect],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }
}

impl Default for VignettePass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for VignettePass {
    fn name(&self) -> &str {
        "vignette"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn resize(&mut self, width: u32, height: u32, _device: &wgpu::Device) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let Some(ref pipeline) = self.pipeline else { return };
        let Some(ref bind_group_layout) = self.bind_group_layout else { return };
        let Some(ref sampler) = self.sampler else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };

        self.update_uniforms(queue);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vignette Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Vignette Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, quad_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }
}

const VIGNETTE_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    // intensity, smoothness, roundness, aspect
    params: vec4<f32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let intensity = params.params.x;
    let smoothness = params.params.y;
    let roundness = params.params.z;
    let aspect = params.params.w;

    // Sample input
    let color = textureSample(input_texture, input_sampler, in.uv);

    // No effect when intensity is 0
    if (intensity <= 0.0) {
        return color;
    }

    // Calculate distance from center
    var uv = in.uv * 2.0 - 1.0;

    // Adjust for aspect ratio based on roundness
    uv.x *= mix(1.0, aspect, roundness);

    // Calculate vignette
    let dist = length(uv);
    // Radius where darkening starts (smaller = more vignette)
    let radius = 1.5 - intensity;
    // Smoothstep from radius to radius+smoothness
    let vignette = 1.0 - smoothstep(radius, radius + smoothness + 0.01, dist);

    return vec4<f32>(color.rgb * vignette, color.a);
}
"#;
