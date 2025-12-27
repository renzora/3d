//! Motion blur post-processing effect.
//!
//! Implements camera-based motion blur by computing per-pixel velocity
//! from the difference between current and previous frame's view-projection matrices.

use wgpu::util::DeviceExt;

/// Motion blur quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MotionBlurQuality {
    /// 4 samples - fast but visible banding
    Low,
    /// 8 samples - good balance
    #[default]
    Medium,
    /// 16 samples - high quality
    High,
}

impl MotionBlurQuality {
    pub fn sample_count(&self) -> u32 {
        match self {
            MotionBlurQuality::Low => 4,
            MotionBlurQuality::Medium => 8,
            MotionBlurQuality::High => 16,
        }
    }
}

/// Motion blur settings.
#[derive(Debug, Clone, Copy)]
pub struct MotionBlurSettings {
    /// Blur intensity/strength (0.0 - 2.0, default 1.0)
    pub intensity: f32,
    /// Maximum blur distance in pixels (default 32.0)
    pub max_blur: f32,
    /// Velocity scale multiplier (default 1.0)
    pub velocity_scale: f32,
}

impl Default for MotionBlurSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            max_blur: 32.0,
            velocity_scale: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionBlurUniform {
    /// Previous view-projection matrix
    prev_view_proj: [[f32; 4]; 4],
    /// Current inverse view-projection matrix
    inv_view_proj: [[f32; 4]; 4],
    /// x: intensity, y: max_blur, z: velocity_scale, w: sample_count
    params: [f32; 4],
    /// x: width, y: height, z: 1/width, w: 1/height
    resolution: [f32; 4],
}

/// Motion blur post-processing pass.
pub struct MotionBlurPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    settings: MotionBlurSettings,
    quality: MotionBlurQuality,
    enabled: bool,
    width: u32,
    height: u32,
    // Store previous frame's view-projection matrix
    prev_view_proj: [[f32; 4]; 4],
}

impl MotionBlurPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Motion Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("motion_blur.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Motion Blur Bind Group Layout"),
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
                // Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Depth sampler (non-filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("Motion Blur Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Motion Blur Pipeline"),
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

        let settings = MotionBlurSettings::default();
        let quality = MotionBlurQuality::default();

        let uniform = MotionBlurUniform {
            prev_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            inv_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            params: [settings.intensity, settings.max_blur, settings.velocity_scale, quality.sample_count() as f32],
            resolution: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Motion Blur Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Motion Blur Sampler"),
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
            quality,
            enabled: false,
            width,
            height,
            prev_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn settings(&self) -> &MotionBlurSettings {
        &self.settings
    }

    pub fn set_intensity(&mut self, intensity: f32) {
        self.settings.intensity = intensity.max(0.0).min(2.0);
    }

    pub fn set_max_blur(&mut self, max_blur: f32) {
        self.settings.max_blur = max_blur.max(1.0).min(64.0);
    }

    pub fn set_velocity_scale(&mut self, scale: f32) {
        self.settings.velocity_scale = scale.max(0.1).min(3.0);
    }

    pub fn quality(&self) -> MotionBlurQuality {
        self.quality
    }

    pub fn set_quality(&mut self, quality: MotionBlurQuality) {
        self.quality = quality;
    }

    /// Render motion blur effect.
    ///
    /// `view_proj` is the current frame's view-projection matrix.
    /// `inv_view_proj` is the inverse of the current view-projection matrix.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        view_proj: [[f32; 4]; 4],
        inv_view_proj: [[f32; 4]; 4],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Motion Blur Depth Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Motion Blur Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&depth_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Update uniform buffer
        let uniform = MotionBlurUniform {
            prev_view_proj: self.prev_view_proj,
            inv_view_proj,
            params: [
                self.settings.intensity,
                self.settings.max_blur,
                self.settings.velocity_scale,
                self.quality.sample_count() as f32,
            ],
            resolution: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        // Store current view-proj for next frame
        self.prev_view_proj = view_proj;

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Motion Blur Pass"),
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
