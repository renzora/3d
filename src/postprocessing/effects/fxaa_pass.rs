//! FXAA (Fast Approximate Anti-Aliasing) post-processing effect.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// FXAA quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FxaaQuality {
    /// Low quality, fastest.
    Low,
    /// Medium quality, balanced.
    #[default]
    Medium,
    /// High quality, best results.
    High,
}

impl FxaaQuality {
    fn as_u32(&self) -> u32 {
        match self {
            FxaaQuality::Low => 0,
            FxaaQuality::Medium => 1,
            FxaaQuality::High => 2,
        }
    }
}

/// FXAA settings.
#[derive(Debug, Clone)]
pub struct FxaaSettings {
    /// Quality preset.
    pub quality: FxaaQuality,
    /// Edge threshold - minimum contrast to apply AA (lower = more edges).
    pub edge_threshold: f32,
    /// Edge threshold minimum - for dark areas.
    pub edge_threshold_min: f32,
}

impl Default for FxaaSettings {
    fn default() -> Self {
        Self {
            quality: FxaaQuality::Medium,
            edge_threshold: 0.166,
            edge_threshold_min: 0.0833,
        }
    }
}

/// FXAA uniform data.
/// Must match WGSL struct layout (48 bytes with proper alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FxaaUniform {
    /// 1.0 / screen_width, 1.0 / screen_height
    inverse_screen_size: [f32; 2],  // offset 0, size 8
    /// Edge detection threshold.
    edge_threshold: f32,             // offset 8, size 4
    /// Minimum edge threshold.
    edge_threshold_min: f32,         // offset 12, size 4
    /// Quality preset.
    quality: u32,                    // offset 16, size 4
    /// Padding to align vec3 to 16 bytes.
    _pad1: [u32; 3],                 // offset 20, size 12 (brings us to 32)
    /// vec3 padding (aligned to 16 bytes).
    _padding: [u32; 3],              // offset 32, size 12
    /// Final pad to 48 bytes.
    _pad2: u32,                      // offset 44, size 4
}

/// FXAA anti-aliasing pass.
pub struct FxaaPass {
    enabled: bool,
    settings: FxaaSettings,
    width: u32,
    height: u32,
    pipeline: Option<wgpu::RenderPipeline>,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    sampler: Option<wgpu::Sampler>,
}

impl FxaaPass {
    /// Create a new FXAA pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: FxaaSettings::default(),
            width: 1,
            height: 1,
            pipeline: None,
            blit_pipeline: None,
            bind_group_layout: None,
            uniform_buffer: None,
            quad_buffer: None,
            sampler: None,
        }
    }

    /// Create with custom settings.
    pub fn with_settings(settings: FxaaSettings) -> Self {
        let mut pass = Self::new();
        pass.settings = settings;
        pass
    }

    /// Get settings.
    pub fn settings(&self) -> &FxaaSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: FxaaSettings) {
        self.settings = settings;
    }

    /// Set quality preset.
    pub fn set_quality(&mut self, quality: FxaaQuality) {
        self.settings.quality = quality;
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        // Create sampler (linear filtering for edge detection)
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FXAA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FXAA Bind Group Layout"),
            entries: &[
                // Input texture
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
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Uniforms
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
        });
        self.bind_group_layout = Some(bind_group_layout);

        // Create uniform buffer
        let uniform = FxaaUniform {
            inverse_screen_size: [1.0 / width as f32, 1.0 / height as f32],
            edge_threshold: self.settings.edge_threshold,
            edge_threshold_min: self.settings.edge_threshold_min,
            quality: self.settings.quality.as_u32(),
            _pad1: [0; 3],
            _padding: [0; 3],
            _pad2: 0,
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FXAA Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FXAA Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipeline
        self.create_pipeline(device, format);
    }

    fn create_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FXAA Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        // FXAA shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FXAA Shader"),
            source: wgpu::ShaderSource::Wgsl(FXAA_SHADER.into()),
        });

        self.pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FXAA Pipeline"),
            layout: Some(&pipeline_layout),
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

        // Blit shader (simple copy when FXAA disabled)
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        self.blit_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
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

    /// Update uniform buffer.
    fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let uniform = FxaaUniform {
                inverse_screen_size: [1.0 / self.width as f32, 1.0 / self.height as f32],
                edge_threshold: self.settings.edge_threshold,
                edge_threshold_min: self.settings.edge_threshold_min,
                quality: self.settings.quality.as_u32(),
                _pad1: [0; 3],
                _padding: [0; 3],
                _pad2: 0,
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }
}

impl Default for FxaaPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for FxaaPass {
    fn name(&self) -> &str {
        "fxaa"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn resize(&mut self, width: u32, height: u32, _device: &wgpu::Device) {
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
        let Some(ref sampler) = self.sampler else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };
        let Some(ref bind_group_layout) = self.bind_group_layout else { return };

        // Choose pipeline based on enabled state
        let pipeline = if self.enabled {
            self.pipeline.as_ref()
        } else {
            self.blit_pipeline.as_ref()
        };
        let Some(pipeline) = pipeline else { return };

        // Update uniforms
        self.update_uniforms(queue);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FXAA Bind Group"),
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

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FXAA Pass"),
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

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
    }
}

const FXAA_SHADER: &str = r#"
// Simplified FXAA implementation for WebGPU
// Based on FXAA 3.11 Quality preset

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    inverse_screen_size: vec2<f32>,
    edge_threshold: f32,
    edge_threshold_min: f32,
    quality: u32,
    _padding: vec3<u32>,
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

// Convert RGB to luminance (using perceived brightness)
fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

// Sample texture at explicit LOD 0
fn tex(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(input_texture, input_sampler, uv, 0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let texel = params.inverse_screen_size;

    // Sample the center pixel and its 4 neighbors
    let rgbM = tex(uv).rgb;
    let rgbN = tex(uv + vec2<f32>(0.0, -texel.y)).rgb;
    let rgbS = tex(uv + vec2<f32>(0.0, texel.y)).rgb;
    let rgbE = tex(uv + vec2<f32>(texel.x, 0.0)).rgb;
    let rgbW = tex(uv + vec2<f32>(-texel.x, 0.0)).rgb;

    // Calculate luminance for edge detection
    let lumaM = luma(rgbM);
    let lumaN = luma(rgbN);
    let lumaS = luma(rgbS);
    let lumaE = luma(rgbE);
    let lumaW = luma(rgbW);

    // Find the maximum and minimum luminance in the neighborhood
    let lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
    let lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));
    let lumaRange = lumaMax - lumaMin;

    // If the contrast is below threshold, skip anti-aliasing
    if (lumaRange < max(params.edge_threshold_min, lumaMax * params.edge_threshold)) {
        return vec4<f32>(rgbM, 1.0);
    }

    // Sample corner pixels for better edge detection
    let rgbNW = tex(uv + vec2<f32>(-texel.x, -texel.y)).rgb;
    let rgbNE = tex(uv + vec2<f32>(texel.x, -texel.y)).rgb;
    let rgbSW = tex(uv + vec2<f32>(-texel.x, texel.y)).rgb;
    let rgbSE = tex(uv + vec2<f32>(texel.x, texel.y)).rgb;

    let lumaNW = luma(rgbNW);
    let lumaNE = luma(rgbNE);
    let lumaSW = luma(rgbSW);
    let lumaSE = luma(rgbSE);

    // Compute edge direction using Sobel-like filter
    let edgeH = abs((lumaNW + lumaN + lumaNE) - (lumaSW + lumaS + lumaSE));
    let edgeV = abs((lumaNW + lumaW + lumaSW) - (lumaNE + lumaE + lumaSE));
    let isHorizontal = edgeH > edgeV;

    // Compute blend direction based on local contrast
    let luma1 = select(lumaW, lumaN, isHorizontal);
    let luma2 = select(lumaE, lumaS, isHorizontal);
    let gradient1 = abs(luma1 - lumaM);
    let gradient2 = abs(luma2 - lumaM);

    // Determine the steeper gradient direction
    let steepest = select(gradient2, gradient1, gradient1 >= gradient2);
    let stepDir = select(
        select(texel.x, -texel.x, gradient1 >= gradient2),
        select(texel.y, -texel.y, gradient1 >= gradient2),
        isHorizontal
    );

    // Calculate blend factor based on local contrast
    let lumaL = 0.25 * (lumaN + lumaS + lumaE + lumaW);
    let rangeL = abs(lumaL - lumaM);
    var blendL = max(0.0, (rangeL / lumaRange) - 0.25) * 1.3333;
    blendL = min(0.75, blendL);

    // Sample with subpixel offset
    var blendUV = uv;
    if (isHorizontal) {
        blendUV.y += stepDir * blendL;
    } else {
        blendUV.x += stepDir * blendL;
    }

    let rgbBlend = tex(blendUV).rgb;

    return vec4<f32>(rgbBlend, 1.0);
}
"#;

const BLIT_SHADER: &str = r#"
// Simple blit shader - copies input texture to output

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(input_texture, input_sampler, in.uv);
}
"#;
