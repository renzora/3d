//! Bloom post-processing effect.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// Bloom effect settings.
#[derive(Debug, Clone)]
pub struct BloomSettings {
    /// Bloom intensity (0.0 - 1.0+).
    pub intensity: f32,
    /// Brightness threshold for bloom.
    pub threshold: f32,
    /// Soft knee for threshold.
    pub soft_threshold: f32,
    /// Number of blur iterations.
    pub blur_iterations: u32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            intensity: 0.5,
            threshold: 0.8,
            soft_threshold: 0.5,
            blur_iterations: 5,
        }
    }
}

/// Bloom uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomUniform {
    threshold: f32,
    soft_threshold: f32,
    intensity: f32,
    _padding: f32,
    resolution: [f32; 4],
}

/// Bloom post-processing pass.
pub struct BloomPass {
    enabled: bool,
    settings: BloomSettings,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    // Pipelines
    threshold_pipeline: Option<wgpu::RenderPipeline>,
    blur_h_pipeline: Option<wgpu::RenderPipeline>,
    blur_v_pipeline: Option<wgpu::RenderPipeline>,
    combine_pipeline: Option<wgpu::RenderPipeline>,
    // Bind group layouts
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    // Intermediate textures
    bright_texture: Option<wgpu::Texture>,
    bright_view: Option<wgpu::TextureView>,
    blur_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    // Buffers
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    // Sampler
    sampler: Option<wgpu::Sampler>,
}

impl BloomPass {
    /// Create a new bloom pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: BloomSettings::default(),
            width: 0,
            height: 0,
            format: wgpu::TextureFormat::Bgra8UnormSrgb, // Default, will be set in init()
            threshold_pipeline: None,
            blur_h_pipeline: None,
            blur_v_pipeline: None,
            combine_pipeline: None,
            bind_group_layout: None,
            bright_texture: None,
            bright_view: None,
            blur_textures: Vec::new(),
            uniform_buffer: None,
            quad_buffer: None,
            sampler: None,
        }
    }

    /// Create with custom settings.
    pub fn with_settings(settings: BloomSettings) -> Self {
        let mut pass = Self::new();
        pass.settings = settings;
        pass
    }

    /// Get settings.
    pub fn settings(&self) -> &BloomSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: BloomSettings) {
        self.settings = settings;
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.format = format;

        // Create sampler
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bloom Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Bind Group Layout"),
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
        let uniform = BloomUniform {
            threshold: self.settings.threshold,
            soft_threshold: self.settings.soft_threshold,
            intensity: self.settings.intensity,
            _padding: 0.0,
            resolution: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bloom Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bloom Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipelines
        self.create_pipelines(device, format);

        // Create intermediate textures
        self.create_textures(device, format);
    }

    fn create_pipelines(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bloom Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        // Threshold shader
        let threshold_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Threshold Shader"),
            source: wgpu::ShaderSource::Wgsl(THRESHOLD_SHADER.into()),
        });

        self.threshold_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Threshold Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &threshold_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &threshold_shader,
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

        // Blur shaders
        let blur_h_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Blur H Shader"),
            source: wgpu::ShaderSource::Wgsl(BLUR_H_SHADER.into()),
        });

        self.blur_h_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Blur H Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blur_h_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_h_shader,
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

        let blur_v_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Blur V Shader"),
            source: wgpu::ShaderSource::Wgsl(BLUR_V_SHADER.into()),
        });

        self.blur_v_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Blur V Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blur_v_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_v_shader,
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

        // Combine shader
        let combine_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Combine Shader"),
            source: wgpu::ShaderSource::Wgsl(COMBINE_SHADER.into()),
        });

        self.combine_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Combine Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &combine_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &combine_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
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

    fn create_textures(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        // Bright pixels texture
        let bright_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Bright Texture"),
            size: wgpu::Extent3d {
                width: self.width / 2,
                height: self.height / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.bright_view = Some(bright_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.bright_texture = Some(bright_texture);

        // Blur ping-pong textures
        self.blur_textures.clear();
        for i in 0..2 {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Blur Texture {}", i)),
                size: wgpu::Extent3d {
                    width: self.width / 2,
                    height: self.height / 2,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.blur_textures.push((texture, view));
        }
    }
}

impl Default for BloomPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for BloomPass {
    fn name(&self) -> &str {
        "bloom"
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

        // Note: uniform buffer gets updated in render() with new resolution

        // Recreate textures at new size using the stored format
        if self.bind_group_layout.is_some() {
            self.create_textures(device, self.format);
        }
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if !self.enabled {
            return;
        }

        let Some(threshold_pipeline) = &self.threshold_pipeline else { return };
        let Some(blur_h_pipeline) = &self.blur_h_pipeline else { return };
        let Some(blur_v_pipeline) = &self.blur_v_pipeline else { return };
        let Some(combine_pipeline) = &self.combine_pipeline else { return };
        let Some(bind_group_layout) = &self.bind_group_layout else { return };
        let Some(uniform_buffer) = &self.uniform_buffer else { return };
        let Some(quad_buffer) = &self.quad_buffer else { return };
        let Some(sampler) = &self.sampler else { return };
        let Some(bright_view) = &self.bright_view else { return };

        if self.blur_textures.len() < 2 {
            return;
        }

        // Update uniform buffer
        let uniform = BloomUniform {
            threshold: self.settings.threshold,
            soft_threshold: self.settings.soft_threshold,
            intensity: self.settings.intensity,
            _padding: 0.0,
            resolution: [self.width as f32, self.height as f32, 2.0 / self.width as f32, 2.0 / self.height as f32],
        };
        queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        // Step 1: Extract bright pixels (threshold pass)
        let input_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom Threshold Bind Group"),
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

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Threshold Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: bright_view,
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

            render_pass.set_pipeline(threshold_pipeline);
            render_pass.set_bind_group(0, &input_bind_group, &[]);
            render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }

        // Step 2: Blur passes (ping-pong between two textures)
        let blur_view_0 = &self.blur_textures[0].1;
        let blur_view_1 = &self.blur_textures[1].1;

        // First blur: bright -> blur0 (horizontal)
        let bright_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom Blur Bright Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(bright_view),
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

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Blur H Pass 0"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: blur_view_0,
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

            render_pass.set_pipeline(blur_h_pipeline);
            render_pass.set_bind_group(0, &bright_bind_group, &[]);
            render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }

        // Create bind groups for ping-pong
        let blur0_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom Blur0 Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(blur_view_0),
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

        let blur1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom Blur1 Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(blur_view_1),
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

        // Blur iterations (ping-pong)
        for i in 0..self.settings.blur_iterations {
            // Vertical blur: blur0 -> blur1
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Blur V Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: blur_view_1,
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

                render_pass.set_pipeline(blur_v_pipeline);
                render_pass.set_bind_group(0, &blur0_bind_group, &[]);
                render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
                render_pass.draw(0..6, 0..1);
            }

            // Horizontal blur: blur1 -> blur0 (if not last iteration)
            if i < self.settings.blur_iterations - 1 {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Blur H Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: blur_view_0,
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

                render_pass.set_pipeline(blur_h_pipeline);
                render_pass.set_bind_group(0, &blur1_bind_group, &[]);
                render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
                render_pass.draw(0..6, 0..1);
            }
        }

        // Step 3: Combine bloom with scene (additive blend)
        // The output already has the scene rendered to it, so we add bloom on top
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Combine Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing scene
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(combine_pipeline);
            render_pass.set_bind_group(0, &blur1_bind_group, &[]); // Use final blurred result
            render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }
    }
}

// Shader sources
const THRESHOLD_SHADER: &str = r#"
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
@group(0) @binding(2) var<uniform> params: vec4<f32>; // threshold, soft_threshold, intensity, _

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(input_texture, input_sampler, in.uv);
    let brightness = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let threshold = params.x;
    let soft = params.y;

    let soft_threshold = threshold * soft;
    let contribution = smoothstep(threshold - soft_threshold, threshold + soft_threshold, brightness);

    return vec4<f32>(color.rgb * contribution, 1.0);
}
"#;

const BLUR_H_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    threshold: f32,
    soft_threshold: f32,
    intensity: f32,
    _padding: f32,
    resolution: vec4<f32>,
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
    let pixel_size = params.resolution.zw;
    var color = vec3<f32>(0.0);

    // 9-tap Gaussian blur
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    color += textureSample(input_texture, input_sampler, in.uv).rgb * weights[0];

    for (var i = 1; i < 5; i++) {
        let offset = vec2<f32>(f32(i) * pixel_size.x, 0.0);
        color += textureSample(input_texture, input_sampler, in.uv + offset).rgb * weights[i];
        color += textureSample(input_texture, input_sampler, in.uv - offset).rgb * weights[i];
    }

    return vec4<f32>(color, 1.0);
}
"#;

const BLUR_V_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    threshold: f32,
    soft_threshold: f32,
    intensity: f32,
    _padding: f32,
    resolution: vec4<f32>,
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
    let pixel_size = params.resolution.zw;
    var color = vec3<f32>(0.0);

    // 9-tap Gaussian blur
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    color += textureSample(input_texture, input_sampler, in.uv).rgb * weights[0];

    for (var i = 1; i < 5; i++) {
        let offset = vec2<f32>(0.0, f32(i) * pixel_size.y);
        color += textureSample(input_texture, input_sampler, in.uv + offset).rgb * weights[i];
        color += textureSample(input_texture, input_sampler, in.uv - offset).rgb * weights[i];
    }

    return vec4<f32>(color, 1.0);
}
"#;

const COMBINE_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    threshold: f32,
    soft_threshold: f32,
    intensity: f32,
    _padding: f32,
    resolution: vec4<f32>,
}

@group(0) @binding(0) var bloom_texture: texture_2d<f32>;
@group(0) @binding(1) var bloom_sampler: sampler;
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
    let bloom = textureSample(bloom_texture, bloom_sampler, in.uv).rgb;
    return vec4<f32>(bloom * params.intensity, 1.0);
}
"#;
