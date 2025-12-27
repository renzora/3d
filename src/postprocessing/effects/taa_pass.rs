//! TAA (Temporal Anti-Aliasing) post-processing effect.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// TAA settings.
#[derive(Debug, Clone)]
pub struct TaaSettings {
    /// Blend factor for history (0.0-1.0, higher = more temporal stability).
    pub blend_factor: f32,
    /// Enable neighborhood clamping to reduce ghosting.
    pub clamp_history: bool,
    /// Sharpening amount (0.0-1.0).
    pub sharpness: f32,
}

impl Default for TaaSettings {
    fn default() -> Self {
        Self {
            blend_factor: 0.9,
            clamp_history: true,
            sharpness: 0.2,
        }
    }
}

/// TAA uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaUniform {
    /// 1.0 / screen_width, 1.0 / screen_height, screen_width, screen_height
    resolution: [f32; 4],
    /// Jitter offset for current frame (x, y) and blend factor, sharpness
    jitter_blend: [f32; 4],
    /// Flags: clamp_history as float
    flags: [f32; 4],
}

/// Halton sequence for jitter offsets (provides good temporal distribution).
/// 16 samples for better convergence (industry standard).
const HALTON_SEQUENCE: [[f32; 2]; 16] = [
    // Halton(2, 3) sequence
    [0.5, 0.333333],
    [0.25, 0.666667],
    [0.75, 0.111111],
    [0.125, 0.444444],
    [0.625, 0.777778],
    [0.375, 0.222222],
    [0.875, 0.555556],
    [0.0625, 0.888889],
    [0.5625, 0.037037],
    [0.3125, 0.370370],
    [0.8125, 0.703704],
    [0.1875, 0.148148],
    [0.6875, 0.481481],
    [0.4375, 0.814815],
    [0.9375, 0.259259],
    [0.03125, 0.592593],
];

/// TAA anti-aliasing pass.
pub struct TaaPass {
    enabled: bool,
    settings: TaaSettings,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    frame_index: u32,
    // Pipelines
    resolve_pipeline: Option<wgpu::RenderPipeline>,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    copy_pipeline: Option<wgpu::RenderPipeline>,
    // Bind group layouts
    resolve_bind_group_layout: Option<wgpu::BindGroupLayout>,
    copy_bind_group_layout: Option<wgpu::BindGroupLayout>,
    // History buffer (ping-pong)
    history_textures: Vec<wgpu::Texture>,
    history_views: Vec<wgpu::TextureView>,
    // Buffers
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    // Sampler
    sampler: Option<wgpu::Sampler>,
}

impl TaaPass {
    /// Create a new TAA pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: TaaSettings::default(),
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            frame_index: 0,
            resolve_pipeline: None,
            blit_pipeline: None,
            copy_pipeline: None,
            resolve_bind_group_layout: None,
            copy_bind_group_layout: None,
            history_textures: Vec::new(),
            history_views: Vec::new(),
            uniform_buffer: None,
            quad_buffer: None,
            sampler: None,
        }
    }

    /// Get settings.
    pub fn settings(&self) -> &TaaSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: TaaSettings) {
        self.settings = settings;
    }

    /// Set blend factor.
    pub fn set_blend_factor(&mut self, factor: f32) {
        self.settings.blend_factor = factor.clamp(0.0, 0.99);
    }

    /// Set sharpness.
    pub fn set_sharpness(&mut self, sharpness: f32) {
        self.settings.sharpness = sharpness.clamp(0.0, 1.0);
    }

    /// Get the current jitter offset for camera projection.
    /// Returns (x, y) offset in clip space (-1 to 1).
    pub fn get_jitter_offset(&self) -> (f32, f32) {
        if !self.enabled {
            return (0.0, 0.0);
        }
        let idx = (self.frame_index as usize) % HALTON_SEQUENCE.len();
        let jitter = HALTON_SEQUENCE[idx];
        // Convert from [0,1] to [-0.5, 0.5] then to clip space
        let x = (jitter[0] - 0.5) / self.width as f32;
        let y = (jitter[1] - 0.5) / self.height as f32;
        (x * 2.0, y * 2.0)
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.format = format;

        // Create sampler
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Create bind group layouts
        self.create_bind_group_layouts(device);

        // Create uniform buffer
        let uniform = TaaUniform {
            resolution: [1.0 / width as f32, 1.0 / height as f32, width as f32, height as f32],
            jitter_blend: [0.0, 0.0, self.settings.blend_factor, self.settings.sharpness],
            flags: [if self.settings.clamp_history { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TAA Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TAA Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipelines
        self.create_pipelines(device, format);

        // Create history textures
        self.create_textures(device);
    }

    fn create_bind_group_layouts(&mut self, device: &wgpu::Device) {
        // Resolve: current + history + sampler + uniforms
        self.resolve_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA Resolve Bind Group Layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        // Copy: just texture + sampler
        self.copy_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA Copy Bind Group Layout"),
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
            ],
        }));
    }

    fn create_pipelines(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        // Resolve pipeline
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TAA Resolve Pipeline Layout"),
            bind_group_layouts: &[self.resolve_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TAA Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_SHADER.into()),
        });

        self.resolve_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TAA Resolve Pipeline"),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
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

        // Copy pipeline (for updating history)
        let copy_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TAA Copy Pipeline Layout"),
            bind_group_layouts: &[self.copy_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TAA Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        self.copy_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TAA Copy Pipeline"),
            layout: Some(&copy_layout),
            vertex: wgpu::VertexState {
                module: &copy_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &copy_shader,
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

        // Blit pipeline (for when disabled)
        self.blit_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TAA Blit Pipeline"),
            layout: Some(&copy_layout),
            vertex: wgpu::VertexState {
                module: &copy_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &copy_shader,
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

    fn create_textures(&mut self, device: &wgpu::Device) {
        self.history_textures.clear();
        self.history_views.clear();

        // Create two history textures for ping-pong
        for i in 0..2 {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("TAA History Texture {}", i)),
                size: wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.history_views.push(view);
            self.history_textures.push(texture);
        }
    }

    fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let jitter = self.get_jitter_offset();
            let uniform = TaaUniform {
                resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
                jitter_blend: [jitter.0, jitter.1, self.settings.blend_factor, self.settings.sharpness],
                flags: [if self.settings.clamp_history { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }

    /// Advance to next frame (call after render).
    pub fn next_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl Default for TaaPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for TaaPass {
    fn name(&self) -> &str {
        "taa"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn resize(&mut self, width: u32, height: u32, device: &wgpu::Device) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        self.create_textures(device);
        // Reset frame index on resize to clear history artifacts
        self.frame_index = 0;
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
        let Some(ref quad_buffer) = self.quad_buffer else { return };

        self.update_uniforms(queue);

        // If disabled, just blit
        if !self.enabled {
            let Some(ref blit_pipeline) = self.blit_pipeline else { return };
            let Some(ref copy_layout) = self.copy_bind_group_layout else { return };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA Blit Bind Group"),
                layout: copy_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Blit Pass"),
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

            pass.set_pipeline(blit_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
            return;
        }

        let Some(ref resolve_pipeline) = self.resolve_pipeline else { return };
        let Some(ref copy_pipeline) = self.copy_pipeline else { return };
        let Some(ref resolve_layout) = self.resolve_bind_group_layout else { return };
        let Some(ref copy_layout) = self.copy_bind_group_layout else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };

        if self.history_views.len() < 2 {
            return;
        }

        // Ping-pong index
        let read_idx = (self.frame_index as usize) % 2;
        let write_idx = (self.frame_index as usize + 1) % 2;

        // TAA resolve: blend current frame with history -> write to history buffer
        let resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TAA Resolve Bind Group"),
            layout: resolve_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.history_views[read_idx]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Render TAA result to history buffer (write_idx)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Resolve Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.history_views[write_idx],
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

            pass.set_pipeline(resolve_pipeline);
            pass.set_bind_group(0, &resolve_bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Blit from history buffer to final output
        let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TAA Output Bind Group"),
            layout: copy_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.history_views[write_idx]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Output Pass"),
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

            pass.set_pipeline(copy_pipeline);
            pass.set_bind_group(0, &output_bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
    }
}

// TAA resolve shader - Industry standard implementation
// Based on techniques from DICE/Frostbite and UE4
const RESOLVE_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    resolution: vec4<f32>,      // 1/w, 1/h, w, h
    jitter_blend: vec4<f32>,    // jitterX, jitterY, blendFactor, sharpness
    flags: vec4<f32>,           // clampHistory, 0, 0, 0
}

@group(0) @binding(0) var current_texture: texture_2d<f32>;
@group(0) @binding(1) var history_texture: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

// Luminance for weighting
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// Tonemap for HDR-aware blending (Reinhard)
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    return c / (1.0 + luminance(c));
}

fn tonemap_inverse(c: vec3<f32>) -> vec3<f32> {
    return c / max(1.0 - luminance(c), 0.001);
}

// Clip history to AABB (industry standard technique from "Temporal AA and the quest for the holy trail")
fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, history: vec3<f32>, current: vec3<f32>) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + 0.001;

    let v_clip = history - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    if (ma_unit > 1.0) {
        return p_clip + v_clip / ma_unit;
    }
    return history;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = params.resolution.xy;
    let blendFactor = params.jitter_blend.z;
    let sharpness = params.jitter_blend.w;
    let clampHistory = params.flags.x > 0.5;

    // Sample current frame (already rendered with camera jitter applied)
    let current = textureSampleLevel(current_texture, tex_sampler, in.uv, 0.0).rgb;

    // Sample history at same location
    var history = textureSampleLevel(history_texture, tex_sampler, in.uv, 0.0).rgb;

    // Neighborhood clamping/clipping to reduce ghosting (industry standard)
    if (clampHistory) {
        // Sample 3x3 neighborhood (plus pattern for efficiency)
        let c0 = current;
        let c1 = textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(-texel.x, 0.0), 0.0).rgb;
        let c2 = textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(texel.x, 0.0), 0.0).rgb;
        let c3 = textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(0.0, -texel.y), 0.0).rgb;
        let c4 = textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(0.0, texel.y), 0.0).rgb;

        // Corner samples for better coverage
        let c5 = textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(-texel.x, -texel.y), 0.0).rgb;
        let c6 = textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(texel.x, -texel.y), 0.0).rgb;
        let c7 = textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(-texel.x, texel.y), 0.0).rgb;
        let c8 = textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(texel.x, texel.y), 0.0).rgb;

        // Compute AABB of neighborhood in RGB space
        var minC = min(c0, min(c1, min(c2, min(c3, c4))));
        var maxC = max(c0, max(c1, max(c2, max(c3, c4))));

        // Expand with corners (weighted less)
        let minCorner = min(c5, min(c6, min(c7, c8)));
        let maxCorner = max(c5, max(c6, max(c7, c8)));
        minC = (minC + minCorner) * 0.5;
        maxC = (maxC + maxCorner) * 0.5;

        // Clip history to AABB (better than simple clamp)
        history = clip_aabb(minC, maxC, history, current);
    }

    // Temporal blend with HDR-aware weighting
    let current_weight = 1.0 - blendFactor;
    let history_weight = blendFactor;

    // Tonemap before blending for HDR stability
    let current_mapped = tonemap(current);
    let history_mapped = tonemap(history);

    var result = current_mapped * current_weight + history_mapped * history_weight;
    result = tonemap_inverse(result);

    // Sharpening pass (optional)
    if (sharpness > 0.0) {
        let neighbors = textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(-texel.x, 0.0), 0.0).rgb
                      + textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(texel.x, 0.0), 0.0).rgb
                      + textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(0.0, -texel.y), 0.0).rgb
                      + textureSampleLevel(current_texture, tex_sampler, in.uv + vec2<f32>(0.0, texel.y), 0.0).rgb;
        let blur = neighbors * 0.25;
        let sharp = result + (result - blur) * sharpness;
        result = max(sharp, vec3<f32>(0.0));
    }

    return vec4<f32>(result, 1.0);
}
"#;

// Simple blit shader
const BLIT_SHADER: &str = r#"
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
