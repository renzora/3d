//! Outline post-processing effect.
//!
//! Renders outlines around objects using depth and normal edge detection.

use wgpu::util::DeviceExt;

/// Outline detection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutlineMode {
    /// Depth-based edge detection only
    Depth,
    /// Normal-based edge detection only
    Normal,
    /// Combined depth and normal detection
    #[default]
    Combined,
}

/// Outline settings.
#[derive(Debug, Clone, Copy)]
pub struct OutlineSettings {
    /// Outline thickness in pixels (1.0 - 5.0)
    pub thickness: f32,
    /// Depth threshold for edge detection (0.0 - 1.0)
    pub depth_threshold: f32,
    /// Normal threshold for edge detection (0.0 - 1.0)
    pub normal_threshold: f32,
    /// Outline color (RGB)
    pub color: [f32; 3],
}

impl Default for OutlineSettings {
    fn default() -> Self {
        Self {
            thickness: 1.0,
            depth_threshold: 0.1,
            normal_threshold: 0.5,
            color: [0.0, 0.0, 0.0], // Black outlines
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OutlineUniform {
    /// x: thickness, y: depth_threshold, z: normal_threshold, w: mode
    params: [f32; 4],
    /// x: r, y: g, z: b, w: unused
    color: [f32; 4],
    /// x: width, y: height, z: 1/width, w: 1/height
    resolution: [f32; 4],
    /// Camera near/far planes for depth linearization
    /// x: near, y: far, z: unused, w: unused
    camera: [f32; 4],
}

/// Outline post-processing pass.
pub struct OutlinePass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    settings: OutlineSettings,
    mode: OutlineMode,
    enabled: bool,
    width: u32,
    height: u32,
    near: f32,
    far: f32,
}

impl OutlinePass {
    /// Create a new outline pass.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Outline Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("outline.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Outline Bind Group Layout"),
            entries: &[
                // Input color texture
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
                // Color sampler
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
            label: Some("Outline Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Outline Pipeline"),
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

        let settings = OutlineSettings::default();
        let mode = OutlineMode::default();

        let uniform = OutlineUniform {
            params: [settings.thickness, settings.depth_threshold, settings.normal_threshold, 2.0], // 2.0 = Combined
            color: [settings.color[0], settings.color[1], settings.color[2], 1.0],
            resolution: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
            camera: [0.1, 100.0, 0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Outline Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Outline Sampler"),
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
            mode,
            enabled: false,
            width,
            height,
            near: 0.1,
            far: 100.0,
        }
    }

    /// Resize the pass.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
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
    pub fn settings(&self) -> &OutlineSettings {
        &self.settings
    }

    /// Set outline thickness.
    pub fn set_thickness(&mut self, thickness: f32) {
        self.settings.thickness = thickness.max(0.5).min(5.0);
    }

    /// Set depth threshold.
    pub fn set_depth_threshold(&mut self, threshold: f32) {
        self.settings.depth_threshold = threshold.max(0.001).min(1.0);
    }

    /// Set normal threshold.
    pub fn set_normal_threshold(&mut self, threshold: f32) {
        self.settings.normal_threshold = threshold.max(0.0).min(1.0);
    }

    /// Set outline color.
    pub fn set_color(&mut self, r: f32, g: f32, b: f32) {
        self.settings.color = [r, g, b];
    }

    /// Get current mode.
    pub fn mode(&self) -> OutlineMode {
        self.mode
    }

    /// Set detection mode.
    pub fn set_mode(&mut self, mode: OutlineMode) {
        self.mode = mode;
    }

    /// Set camera near/far planes for depth linearization.
    pub fn set_camera_planes(&mut self, near: f32, far: f32) {
        self.near = near;
        self.far = far;
    }

    /// Render outline effect.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Outline Depth Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Outline Bind Group"),
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

        // Update uniforms
        let mode_value = match self.mode {
            OutlineMode::Depth => 0.0,
            OutlineMode::Normal => 1.0,
            OutlineMode::Combined => 2.0,
        };

        let uniform = OutlineUniform {
            params: [
                self.settings.thickness,
                self.settings.depth_threshold,
                self.settings.normal_threshold,
                mode_value,
            ],
            color: [self.settings.color[0], self.settings.color[1], self.settings.color[2], 1.0],
            resolution: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            camera: [self.near, self.far, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Outline Pass"),
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
