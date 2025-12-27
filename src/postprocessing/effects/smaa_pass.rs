//! SMAA (Subpixel Morphological Anti-Aliasing) post-processing effect.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// SMAA quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmaaQuality {
    /// Low quality, fastest.
    Low,
    /// Medium quality, balanced.
    #[default]
    Medium,
    /// High quality, best results.
    High,
    /// Ultra quality, maximum quality.
    Ultra,
}

/// SMAA settings.
#[derive(Debug, Clone)]
pub struct SmaaSettings {
    /// Quality preset.
    pub quality: SmaaQuality,
    /// Edge detection threshold.
    pub threshold: f32,
    /// Maximum search steps for edge detection.
    pub max_search_steps: u32,
}

impl Default for SmaaSettings {
    fn default() -> Self {
        Self {
            quality: SmaaQuality::Medium,
            threshold: 0.1,
            max_search_steps: 16,
        }
    }
}

/// SMAA uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SmaaUniform {
    /// 1.0 / screen_width, 1.0 / screen_height, screen_width, screen_height
    resolution: [f32; 4],
    /// Edge detection threshold and settings
    threshold: f32,
    max_search_steps: f32,
    _padding: [f32; 2],
}

/// SMAA anti-aliasing pass.
pub struct SmaaPass {
    enabled: bool,
    settings: SmaaSettings,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    // Pipelines
    edge_pipeline: Option<wgpu::RenderPipeline>,
    blend_pipeline: Option<wgpu::RenderPipeline>,
    resolve_pipeline: Option<wgpu::RenderPipeline>,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    // Bind group layouts
    edge_bind_group_layout: Option<wgpu::BindGroupLayout>,
    blend_bind_group_layout: Option<wgpu::BindGroupLayout>,
    resolve_bind_group_layout: Option<wgpu::BindGroupLayout>,
    // Intermediate textures
    edge_texture: Option<wgpu::Texture>,
    edge_view: Option<wgpu::TextureView>,
    blend_texture: Option<wgpu::Texture>,
    blend_view: Option<wgpu::TextureView>,
    // Buffers
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    // Sampler
    sampler: Option<wgpu::Sampler>,
    point_sampler: Option<wgpu::Sampler>,
}

impl SmaaPass {
    /// Create a new SMAA pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: SmaaSettings::default(),
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            edge_pipeline: None,
            blend_pipeline: None,
            resolve_pipeline: None,
            blit_pipeline: None,
            edge_bind_group_layout: None,
            blend_bind_group_layout: None,
            resolve_bind_group_layout: None,
            edge_texture: None,
            edge_view: None,
            blend_texture: None,
            blend_view: None,
            uniform_buffer: None,
            quad_buffer: None,
            sampler: None,
            point_sampler: None,
        }
    }

    /// Get settings.
    pub fn settings(&self) -> &SmaaSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: SmaaSettings) {
        self.settings = settings;
    }

    /// Set quality preset.
    pub fn set_quality(&mut self, quality: SmaaQuality) {
        self.settings.quality = quality;
        // Adjust parameters based on quality
        match quality {
            SmaaQuality::Low => {
                self.settings.threshold = 0.15;
                self.settings.max_search_steps = 4;
            }
            SmaaQuality::Medium => {
                self.settings.threshold = 0.1;
                self.settings.max_search_steps = 8;
            }
            SmaaQuality::High => {
                self.settings.threshold = 0.1;
                self.settings.max_search_steps = 16;
            }
            SmaaQuality::Ultra => {
                self.settings.threshold = 0.05;
                self.settings.max_search_steps = 32;
            }
        }
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.format = format;

        // Create samplers
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMAA Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.point_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMAA Point Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        // Create bind group layouts
        self.create_bind_group_layouts(device);

        // Create uniform buffer
        let uniform = SmaaUniform {
            resolution: [1.0 / width as f32, 1.0 / height as f32, width as f32, height as f32],
            threshold: self.settings.threshold,
            max_search_steps: self.settings.max_search_steps as f32,
            _padding: [0.0; 2],
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SMAA Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SMAA Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipelines
        self.create_pipelines(device, format);

        // Create intermediate textures
        self.create_textures(device);
    }

    fn create_bind_group_layouts(&mut self, device: &wgpu::Device) {
        // Edge detection: input texture + sampler + uniforms
        self.edge_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SMAA Edge Bind Group Layout"),
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

        // Blend weight: edge texture + sampler + uniforms
        self.blend_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SMAA Blend Bind Group Layout"),
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

        // Resolve: input + blend texture + samplers + uniforms
        self.resolve_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SMAA Resolve Bind Group Layout"),
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
    }

    fn create_pipelines(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        // Edge detection pipeline
        let edge_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SMAA Edge Pipeline Layout"),
            bind_group_layouts: &[self.edge_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SMAA Edge Shader"),
            source: wgpu::ShaderSource::Wgsl(EDGE_SHADER.into()),
        });

        self.edge_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SMAA Edge Pipeline"),
            layout: Some(&edge_layout),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg8Unorm,
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

        // Blend weight pipeline
        let blend_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SMAA Blend Pipeline Layout"),
            bind_group_layouts: &[self.blend_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SMAA Blend Shader"),
            source: wgpu::ShaderSource::Wgsl(BLEND_SHADER.into()),
        });

        self.blend_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SMAA Blend Pipeline"),
            layout: Some(&blend_layout),
            vertex: wgpu::VertexState {
                module: &blend_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blend_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
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

        // Resolve pipeline
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SMAA Resolve Pipeline Layout"),
            bind_group_layouts: &[self.resolve_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SMAA Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_SHADER.into()),
        });

        self.resolve_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SMAA Resolve Pipeline"),
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

        // Blit pipeline for when disabled
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SMAA Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        self.blit_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SMAA Blit Pipeline"),
            layout: Some(&edge_layout),
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

    fn create_textures(&mut self, device: &wgpu::Device) {
        // Edge texture (RG8 for horizontal/vertical edges)
        let edge_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMAA Edge Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.edge_view = Some(edge_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.edge_texture = Some(edge_texture);

        // Blend weight texture (RGBA8 for blend weights)
        let blend_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMAA Blend Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.blend_view = Some(blend_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.blend_texture = Some(blend_texture);
    }

    fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let uniform = SmaaUniform {
                resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
                threshold: self.settings.threshold,
                max_search_steps: self.settings.max_search_steps as f32,
                _padding: [0.0; 2],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }
}

impl Default for SmaaPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for SmaaPass {
    fn name(&self) -> &str {
        "smaa"
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

        self.update_uniforms(queue);

        // If disabled, just blit
        if !self.enabled {
            let Some(ref blit_pipeline) = self.blit_pipeline else { return };
            let Some(ref edge_layout) = self.edge_bind_group_layout else { return };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SMAA Blit Bind Group"),
                layout: edge_layout,
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
                label: Some("SMAA Blit Pass"),
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

        let Some(ref edge_pipeline) = self.edge_pipeline else { return };
        let Some(ref blend_pipeline) = self.blend_pipeline else { return };
        let Some(ref resolve_pipeline) = self.resolve_pipeline else { return };
        let Some(ref edge_layout) = self.edge_bind_group_layout else { return };
        let Some(ref blend_layout) = self.blend_bind_group_layout else { return };
        let Some(ref resolve_layout) = self.resolve_bind_group_layout else { return };
        let Some(ref edge_view) = self.edge_view else { return };
        let Some(ref blend_view) = self.blend_view else { return };

        // Pass 1: Edge detection
        let edge_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SMAA Edge Bind Group"),
            layout: edge_layout,
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
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SMAA Edge Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: edge_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(edge_pipeline);
            pass.set_bind_group(0, &edge_bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 2: Blend weight calculation
        let blend_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SMAA Blend Bind Group"),
            layout: blend_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(edge_view),
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
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SMAA Blend Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: blend_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(blend_pipeline);
            pass.set_bind_group(0, &blend_bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 3: Neighborhood blending (resolve)
        let resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SMAA Resolve Bind Group"),
            layout: resolve_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(blend_view),
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

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SMAA Resolve Pass"),
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

            pass.set_pipeline(resolve_pipeline);
            pass.set_bind_group(0, &resolve_bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
    }
}

// SMAA Edge Detection Shader - Industry standard luma-based detection
// Based on SMAA: Enhanced Subpixel Morphological Antialiasing (Jimenez et al.)
const EDGE_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    resolution: vec4<f32>,  // 1/w, 1/h, w, h
    threshold: f32,
    max_search_steps: f32,
    _padding: vec2<f32>,
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

// Luminance calculation (Rec. 709)
fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<f32> {
    let texel = params.resolution.xy;
    let threshold = params.threshold;

    // Sample center and 4 neighbors for edge detection
    let C = luma(textureSampleLevel(input_texture, input_sampler, in.uv, 0.0).rgb);
    let L = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(-texel.x, 0.0), 0.0).rgb);
    let T = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(0.0, -texel.y), 0.0).rgb);
    let R = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(texel.x, 0.0), 0.0).rgb);
    let B = luma(textureSampleLevel(input_texture, input_sampler, in.uv + vec2<f32>(0.0, texel.y), 0.0).rgb);

    // Calculate deltas
    let deltaL = abs(C - L);
    let deltaT = abs(C - T);
    let deltaR = abs(C - R);
    let deltaB = abs(C - B);

    // Find maximum delta for local contrast adaptation
    let maxDelta = max(max(deltaL, deltaT), max(deltaR, deltaB));

    // Edges are detected where contrast exceeds threshold
    // Using local contrast adaptation: threshold * maxDelta provides better edge detection
    var edges = vec2<f32>(0.0);

    // Left edge (horizontal)
    if (deltaL >= threshold) {
        edges.x = 1.0;
    }

    // Top edge (vertical)
    if (deltaT >= threshold) {
        edges.y = 1.0;
    }

    return edges;
}
"#;

// SMAA Blend Weight Calculation Shader
// Computes blend weights based on edge patterns and crossing edges
const BLEND_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    resolution: vec4<f32>,
    threshold: f32,
    max_search_steps: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var edge_texture: texture_2d<f32>;
@group(0) @binding(1) var edge_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

// Search for edge end in horizontal direction
fn searchXLeft(uv: vec2<f32>, texel: vec2<f32>, max_steps: i32) -> f32 {
    var coord = uv;
    var i = 0;
    for (; i < max_steps; i++) {
        coord.x -= texel.x;
        let e = textureSampleLevel(edge_texture, edge_sampler, coord, 0.0).rg;
        // Stop at edge end (no left edge) or crossing edge (top edge found)
        if (e.r < 0.5) { break; }
        if (e.g > 0.5) { break; } // Crossing edge
    }
    return f32(i);
}

fn searchXRight(uv: vec2<f32>, texel: vec2<f32>, max_steps: i32) -> f32 {
    var coord = uv;
    var i = 0;
    for (; i < max_steps; i++) {
        coord.x += texel.x;
        let e = textureSampleLevel(edge_texture, edge_sampler, coord, 0.0).rg;
        if (e.r < 0.5) { break; }
        if (e.g > 0.5) { break; }
    }
    return f32(i);
}

fn searchYUp(uv: vec2<f32>, texel: vec2<f32>, max_steps: i32) -> f32 {
    var coord = uv;
    var i = 0;
    for (; i < max_steps; i++) {
        coord.y -= texel.y;
        let e = textureSampleLevel(edge_texture, edge_sampler, coord, 0.0).rg;
        if (e.g < 0.5) { break; }
        if (e.r > 0.5) { break; }
    }
    return f32(i);
}

fn searchYDown(uv: vec2<f32>, texel: vec2<f32>, max_steps: i32) -> f32 {
    var coord = uv;
    var i = 0;
    for (; i < max_steps; i++) {
        coord.y += texel.y;
        let e = textureSampleLevel(edge_texture, edge_sampler, coord, 0.0).rg;
        if (e.g < 0.5) { break; }
        if (e.r > 0.5) { break; }
    }
    return f32(i);
}

// Calculate blend area based on distances (approximation of area texture lookup)
fn area(d1: f32, d2: f32, crossing: f32) -> vec2<f32> {
    // Simplified area calculation based on edge distances
    // In full SMAA, this uses a precomputed area texture
    let total = d1 + d2;
    if (total < 0.001) {
        return vec2<f32>(0.0);
    }

    // Blend factor based on position along edge
    let t = d1 / total;

    // Smooth step for sub-pixel blending
    let blend = smoothstep(0.0, 1.0, t);

    // Adjust for crossing edges
    let factor = 0.5 * (1.0 - crossing * 0.5);

    return vec2<f32>(factor * (1.0 - blend), factor * blend);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = params.resolution.xy;
    let max_steps = i32(params.max_search_steps);

    let e = textureSampleLevel(edge_texture, edge_sampler, in.uv, 0.0).rg;

    var weights = vec4<f32>(0.0);

    // Horizontal edge processing (for horizontal edges, we blend vertically)
    if (e.r > 0.5) {
        // Search for edge ends
        let dL = searchXLeft(in.uv, texel, max_steps);
        let dR = searchXRight(in.uv, texel, max_steps);

        // Check for crossing edges at ends
        let eL = textureSampleLevel(edge_texture, edge_sampler, in.uv - vec2<f32>((dL + 1.0) * texel.x, 0.0), 0.0).g;
        let eR = textureSampleLevel(edge_texture, edge_sampler, in.uv + vec2<f32>((dR + 1.0) * texel.x, 0.0), 0.0).g;
        let crossing = (eL + eR) * 0.5;

        // Calculate blend weights
        let blendArea = area(dL, dR, crossing);
        weights.x = blendArea.x; // Up blend
        weights.y = blendArea.y; // Down blend
    }

    // Vertical edge processing (for vertical edges, we blend horizontally)
    if (e.g > 0.5) {
        let dT = searchYUp(in.uv, texel, max_steps);
        let dB = searchYDown(in.uv, texel, max_steps);

        let eT = textureSampleLevel(edge_texture, edge_sampler, in.uv - vec2<f32>(0.0, (dT + 1.0) * texel.y), 0.0).r;
        let eB = textureSampleLevel(edge_texture, edge_sampler, in.uv + vec2<f32>(0.0, (dB + 1.0) * texel.y), 0.0).r;
        let crossing = (eT + eB) * 0.5;

        let blendArea = area(dT, dB, crossing);
        weights.z = blendArea.x; // Left blend
        weights.w = blendArea.y; // Right blend
    }

    return weights;
}
"#;

// Neighborhood blending (resolve) shader
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
    resolution: vec4<f32>,
    threshold: f32,
    max_search_steps: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var blend_texture: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = params.resolution.xy;

    // Sample blend weights
    let blend = textureSampleLevel(blend_texture, tex_sampler, in.uv, 0.0);

    // Check if we need to blend (any non-zero weight)
    let blendSum = blend.x + blend.y + blend.z + blend.w;
    if (blendSum < 0.001) {
        return textureSampleLevel(input_texture, tex_sampler, in.uv, 0.0);
    }

    // Sample neighbors based on blend weights
    var color = vec4<f32>(0.0);

    // Horizontal blending (left/right weights in x/y)
    if (blend.x + blend.y > 0.001) {
        let weightH = blend.x + blend.y;
        let offsetH = (blend.y - blend.x) * texel.x;
        color += textureSampleLevel(input_texture, tex_sampler, in.uv + vec2<f32>(offsetH, 0.0), 0.0) * weightH;
    }

    // Vertical blending (up/down weights in z/w)
    if (blend.z + blend.w > 0.001) {
        let weightV = blend.z + blend.w;
        let offsetV = (blend.w - blend.z) * texel.y;
        color += textureSampleLevel(input_texture, tex_sampler, in.uv + vec2<f32>(0.0, offsetV), 0.0) * weightV;
    }

    // Normalize and blend with original
    let totalWeight = blendSum;
    color = color / max(totalWeight, 0.001);

    // Mix with original based on edge strength
    let original = textureSampleLevel(input_texture, tex_sampler, in.uv, 0.0);
    let mixFactor = min(totalWeight * 0.5, 1.0);

    return mix(original, color, mixFactor);
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
