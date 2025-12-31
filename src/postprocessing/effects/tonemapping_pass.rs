//! HDR tonemapping post-processing effect.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// Tonemapping operator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TonemappingMode {
    /// Linear (no tonemapping).
    Linear,
    /// Reinhard operator.
    Reinhard,
    /// Reinhard with white point.
    ReinhardLuminance,
    /// ACES filmic curve.
    Aces,
    /// Uncharted 2 filmic curve.
    Uncharted2,
    /// AgX (Blender 4.0+ default) - best for both HDR and SDR.
    #[default]
    AgX,
}

impl TonemappingMode {
    fn as_u32(&self) -> u32 {
        match self {
            TonemappingMode::Linear => 0,
            TonemappingMode::Reinhard => 1,
            TonemappingMode::ReinhardLuminance => 2,
            TonemappingMode::Aces => 3,
            TonemappingMode::Uncharted2 => 4,
            TonemappingMode::AgX => 5,
        }
    }

    /// Create from u32 value.
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => TonemappingMode::Linear,
            1 => TonemappingMode::Reinhard,
            2 => TonemappingMode::ReinhardLuminance,
            3 => TonemappingMode::Aces,
            4 => TonemappingMode::Uncharted2,
            5 => TonemappingMode::AgX,
            _ => TonemappingMode::Aces, // Default to ACES
        }
    }
}

/// Tonemapping settings.
#[derive(Debug, Clone)]
pub struct TonemappingSettings {
    /// Tonemapping operator.
    pub mode: TonemappingMode,
    /// Exposure adjustment.
    pub exposure: f32,
    /// Gamma correction.
    pub gamma: f32,
    /// Contrast adjustment.
    pub contrast: f32,
    /// Saturation adjustment.
    pub saturation: f32,
}

impl Default for TonemappingSettings {
    fn default() -> Self {
        Self {
            mode: TonemappingMode::AgX,
            exposure: 1.0,
            gamma: 2.2,
            contrast: 1.0,
            saturation: 1.0,
        }
    }
}

/// Tonemapping uniform data.
/// WGSL struct alignment: vec3<f32> requires 16-byte alignment.
/// Total size must be 48 bytes (multiple of 16).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TonemappingUniform {
    exposure: f32,      // offset 0
    gamma: f32,         // offset 4
    contrast: f32,      // offset 8
    saturation: f32,    // offset 12
    mode: u32,          // offset 16
    _pad1: u32,         // offset 20 - padding to align vec3 to 16 bytes
    _pad2: u32,         // offset 24
    _pad3: u32,         // offset 28
    _padding: [f32; 3], // offset 32 (aligned to 16)
    _align_pad: f32,    // offset 44 - pad struct to 48 bytes
}

/// HDR tonemapping pass.
pub struct TonemappingPass {
    enabled: bool,
    settings: TonemappingSettings,
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    sampler: Option<wgpu::Sampler>,
}

impl TonemappingPass {
    /// Create a new tonemapping pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: TonemappingSettings::default(),
            pipeline: None,
            bind_group_layout: None,
            uniform_buffer: None,
            quad_buffer: None,
            sampler: None,
        }
    }

    /// Create with custom settings.
    pub fn with_settings(settings: TonemappingSettings) -> Self {
        let mut pass = Self::new();
        pass.settings = settings;
        pass
    }

    /// Get settings.
    pub fn settings(&self) -> &TonemappingSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: TonemappingSettings) {
        self.settings = settings;
    }

    /// Set exposure.
    pub fn set_exposure(&mut self, exposure: f32) {
        self.settings.exposure = exposure;
    }

    /// Set tonemapping mode.
    pub fn set_mode(&mut self, mode: TonemappingMode) {
        self.settings.mode = mode;
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        // Create sampler
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Tonemapping Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tonemapping Bind Group Layout"),
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
        let uniform = TonemappingUniform {
            exposure: self.settings.exposure,
            gamma: self.settings.gamma,
            contrast: self.settings.contrast,
            saturation: self.settings.saturation,
            mode: self.settings.mode.as_u32(),
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _padding: [0.0; 3],
            _align_pad: 0.0,
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tonemapping Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tonemapping Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipeline
        self.create_pipeline(device, format);
    }

    fn create_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tonemapping Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tonemapping Shader"),
            source: wgpu::ShaderSource::Wgsl(TONEMAPPING_SHADER.into()),
        });

        self.pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tonemapping Pipeline"),
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
    }

    /// Update uniform buffer with current settings.
    pub fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let uniform = TonemappingUniform {
                exposure: self.settings.exposure,
                gamma: self.settings.gamma,
                contrast: self.settings.contrast,
                saturation: self.settings.saturation,
                mode: self.settings.mode.as_u32(),
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
                _padding: [0.0; 3],
                _align_pad: 0.0,
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }
}

impl Default for TonemappingPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for TonemappingPass {
    fn name(&self) -> &str {
        "tonemapping"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn resize(&mut self, _width: u32, _height: u32, _device: &wgpu::Device) {
        // Tonemapping doesn't need to resize - it works on any resolution
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
        let Some(ref sampler) = self.sampler else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };
        let Some(ref bind_group_layout) = self.bind_group_layout else { return };

        // Update uniforms with current settings
        self.update_uniforms(queue);

        // Create bind group for this frame's input texture
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tonemapping Bind Group"),
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
            label: Some("Tonemapping Pass"),
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

const TONEMAPPING_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    exposure: f32,
    gamma: f32,
    contrast: f32,
    saturation: f32,
    mode: u32,
    _padding: vec3<f32>,
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

// Reinhard tonemapping
fn tonemap_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

// Reinhard with luminance
fn tonemap_reinhard_luminance(color: vec3<f32>) -> vec3<f32> {
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let mapped_luminance = luminance / (1.0 + luminance);
    return color * (mapped_luminance / luminance);
}

// ACES Filmic Tone Mapping (Narkowicz approximation - fast)
fn tonemap_aces_fast(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
fn aces_input_matrix(color: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777)
    );
    return m * color;
}

// ODT_SAT => XYZ => D60_2_D65 => sRGB
fn aces_output_matrix(color: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>( 1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108,  1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605,  1.07602)
    );
    return m * color;
}

// RRT and ODT fit
fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

// Full ACES Filmic Tone Mapping (Stephen Hill's fit)
// This is the proper RRT+ODT used in film production
fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    var c = aces_input_matrix(color);
    c = rrt_and_odt_fit(c);
    c = aces_output_matrix(c);
    return saturate(c);
}

// Uncharted 2 tonemapping helper
fn uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn tonemap_uncharted2(color: vec3<f32>) -> vec3<f32> {
    let exposure_bias = 2.0;
    let curr = uncharted2_partial(color * exposure_bias);
    let W = vec3<f32>(11.2);
    let white_scale = vec3<f32>(1.0) / uncharted2_partial(W);
    return curr * white_scale;
}

// AgX tonemapping (Blender 4.0+ / Troy Sobotka)
// Attempt to be more perceptually uniform and preserve hues

// AgX inset matrix (sRGB to AgX working space)
fn agx_inset_matrix(color: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>(0.842479062253094,  0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772,  0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104)
    );
    return m * color;
}

// AgX outset matrix (AgX working space to sRGB)
fn agx_outset_matrix(color: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>( 1.19687900512017,   -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368,  1.15190312990417,   -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433,  1.15107367264116)
    );
    return m * color;
}

// AgX sigmoid curve approximation
fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
    // 6th order polynomial approximation of AgX sigmoid
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn tonemap_agx(color: vec3<f32>) -> vec3<f32> {
    let agx_min_ev = -12.47393;
    let agx_max_ev = 4.026069;

    // Apply inset matrix (approximates rec2020 / wide gamut handling)
    var c = agx_inset_matrix(color);

    // Log2 encoding
    c = max(c, vec3<f32>(1e-10));
    c = log2(c);
    c = (c - agx_min_ev) / (agx_max_ev - agx_min_ev);
    c = saturate(c);

    // Apply sigmoid curve
    c = agx_default_contrast_approx(c);

    // Apply outset matrix to return to sRGB-ish
    c = agx_outset_matrix(c);

    // Clamp to valid range
    return saturate(c);
}

// Saturation adjustment
fn adjust_saturation(color: vec3<f32>, saturation: f32) -> vec3<f32> {
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(vec3<f32>(luminance), color, saturation);
}

// Contrast adjustment
fn adjust_contrast(color: vec3<f32>, contrast: f32) -> vec3<f32> {
    return (color - 0.5) * contrast + 0.5;
}

// Interleaved gradient noise for dithering (avoids banding artifacts)
fn interleaved_gradient_noise(pos: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pos, magic.xy)));
}

// Apply dithering to break up color banding
fn dither(color: vec3<f32>, pos: vec2<f32>) -> vec3<f32> {
    // Noise in range [-0.5/255, 0.5/255] to randomize quantization
    let noise = (interleaved_gradient_noise(pos) - 0.5) / 255.0;
    return color + vec3<f32>(noise);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(input_texture, input_sampler, in.uv).rgb;

    // Apply exposure
    color = color * params.exposure;

    // Apply tonemapping based on mode
    switch params.mode {
        case 0u: {
            // Linear - no tonemapping
        }
        case 1u: {
            color = tonemap_reinhard(color);
        }
        case 2u: {
            color = tonemap_reinhard_luminance(color);
        }
        case 3u: {
            color = tonemap_aces(color);
        }
        case 4u: {
            color = tonemap_uncharted2(color);
        }
        case 5u: {
            color = tonemap_agx(color);
        }
        default: {
            color = tonemap_aces(color);
        }
    }

    // Apply contrast
    color = adjust_contrast(color, params.contrast);

    // Apply saturation
    color = adjust_saturation(color, params.saturation);

    // Apply gamma correction
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / params.gamma));

    // Apply dithering to break up banding (before 8-bit quantization)
    color = dither(color, in.position.xy);

    return vec4<f32>(color, 1.0);
}
"#;
