//! Color correction post-processing effect.
//!
//! Provides brightness, contrast, saturation, gamma, and color temperature adjustments.

use wgpu::util::DeviceExt;

/// Color correction settings.
#[derive(Debug, Clone, Copy)]
pub struct ColorCorrectionSettings {
    /// Brightness adjustment (-1.0 to 1.0, default 0.0)
    pub brightness: f32,
    /// Contrast adjustment (0.0 to 2.0, default 1.0)
    pub contrast: f32,
    /// Saturation adjustment (0.0 to 2.0, default 1.0)
    pub saturation: f32,
    /// Gamma correction (0.5 to 2.5, default 1.0)
    pub gamma: f32,
    /// Color temperature (-1.0 cool to 1.0 warm, default 0.0)
    pub temperature: f32,
    /// Tint adjustment (-1.0 green to 1.0 magenta, default 0.0)
    pub tint: f32,
}

impl Default for ColorCorrectionSettings {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            gamma: 1.0,
            temperature: 0.0,
            tint: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorCorrectionUniform {
    /// x: brightness, y: contrast, z: saturation, w: gamma
    params1: [f32; 4],
    /// x: temperature, y: tint, z: unused, w: unused
    params2: [f32; 4],
}

/// Color correction post-processing pass.
pub struct ColorCorrectionPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    settings: ColorCorrectionSettings,
    enabled: bool,
}

impl ColorCorrectionPass {
    /// Create a new color correction pass.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Color Correction Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("color_correction.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Color Correction Bind Group Layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Color Correction Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Color Correction Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let settings = ColorCorrectionSettings::default();

        let uniform = ColorCorrectionUniform {
            params1: [settings.brightness, settings.contrast, settings.saturation, settings.gamma],
            params2: [settings.temperature, settings.tint, 0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Color Correction Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Color Correction Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            sampler,
            settings,
            enabled: false,
        }
    }

    /// Check if enabled.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get current settings.
    pub fn settings(&self) -> &ColorCorrectionSettings {
        &self.settings
    }

    /// Set brightness (-1.0 to 1.0).
    pub fn set_brightness(&mut self, brightness: f32) {
        self.settings.brightness = brightness.max(-1.0).min(1.0);
    }

    /// Set contrast (0.0 to 2.0).
    pub fn set_contrast(&mut self, contrast: f32) {
        self.settings.contrast = contrast.max(0.0).min(2.0);
    }

    /// Set saturation (0.0 to 2.0).
    pub fn set_saturation(&mut self, saturation: f32) {
        self.settings.saturation = saturation.max(0.0).min(2.0);
    }

    /// Set gamma (0.5 to 2.5).
    pub fn set_gamma(&mut self, gamma: f32) {
        self.settings.gamma = gamma.max(0.5).min(2.5);
    }

    /// Set color temperature (-1.0 cool to 1.0 warm).
    pub fn set_temperature(&mut self, temperature: f32) {
        self.settings.temperature = temperature.max(-1.0).min(1.0);
    }

    /// Set tint (-1.0 green to 1.0 magenta).
    pub fn set_tint(&mut self, tint: f32) {
        self.settings.tint = tint.max(-1.0).min(1.0);
    }

    /// Reset all settings to defaults.
    pub fn reset(&mut self) {
        self.settings = ColorCorrectionSettings::default();
    }

    /// Render color correction effect.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Color Correction Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Update uniforms
        let uniform = ColorCorrectionUniform {
            params1: [
                self.settings.brightness,
                self.settings.contrast,
                self.settings.saturation,
                self.settings.gamma,
            ],
            params2: [self.settings.temperature, self.settings.tint, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Color Correction Pass"),
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

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
