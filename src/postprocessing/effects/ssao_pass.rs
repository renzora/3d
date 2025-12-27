//! SSAO (Screen-Space Ambient Occlusion) post-processing effect.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// SSAO quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SsaoQuality {
    /// Low quality - 8 samples.
    Low,
    /// Medium quality - 16 samples.
    #[default]
    Medium,
    /// High quality - 32 samples.
    High,
    /// Ultra quality - 64 samples.
    Ultra,
}

/// SSAO settings.
#[derive(Debug, Clone)]
pub struct SsaoSettings {
    /// Quality preset (affects sample count).
    pub quality: SsaoQuality,
    /// Sampling radius in world units.
    pub radius: f32,
    /// Intensity/strength of the occlusion.
    pub intensity: f32,
    /// Bias to prevent self-occlusion artifacts.
    pub bias: f32,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            quality: SsaoQuality::Medium,
            radius: 0.5,
            intensity: 1.0,
            bias: 0.025,
        }
    }
}

/// SSAO uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniform {
    /// Projection matrix for reconstructing positions.
    projection: [[f32; 4]; 4],
    /// Inverse projection matrix.
    inv_projection: [[f32; 4]; 4],
    /// 1/width, 1/height, width, height
    resolution: [f32; 4],
    /// radius, intensity, bias, sample_count
    params: [f32; 4],
    /// near, far, 0, 0
    near_far: [f32; 4],
}

/// SSAO post-processing pass.
pub struct SsaoPass {
    enabled: bool,
    settings: SsaoSettings,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    // Projection matrices (updated each frame)
    projection: [[f32; 4]; 4],
    inv_projection: [[f32; 4]; 4],
    near: f32,
    far: f32,
    // Pipelines
    ssao_pipeline: Option<wgpu::RenderPipeline>,
    blur_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    // Bind group layouts
    ssao_bind_group_layout: Option<wgpu::BindGroupLayout>,
    blur_bind_group_layout: Option<wgpu::BindGroupLayout>,
    composite_bind_group_layout: Option<wgpu::BindGroupLayout>,
    // Textures
    ssao_texture: Option<wgpu::Texture>,
    ssao_view: Option<wgpu::TextureView>,
    blur_texture: Option<wgpu::Texture>,
    blur_view: Option<wgpu::TextureView>,
    noise_texture: Option<wgpu::Texture>,
    noise_view: Option<wgpu::TextureView>,
    // Sample kernel buffer
    kernel_buffer: Option<wgpu::Buffer>,
    // Buffers
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    // Samplers
    linear_sampler: Option<wgpu::Sampler>,
    point_sampler: Option<wgpu::Sampler>,
}

impl SsaoPass {
    /// Create a new SSAO pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: SsaoSettings::default(),
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            projection: [[0.0; 4]; 4],
            inv_projection: [[0.0; 4]; 4],
            near: 0.1,
            far: 100.0,
            ssao_pipeline: None,
            blur_pipeline: None,
            composite_pipeline: None,
            ssao_bind_group_layout: None,
            blur_bind_group_layout: None,
            composite_bind_group_layout: None,
            ssao_texture: None,
            ssao_view: None,
            blur_texture: None,
            blur_view: None,
            noise_texture: None,
            noise_view: None,
            kernel_buffer: None,
            uniform_buffer: None,
            quad_buffer: None,
            linear_sampler: None,
            point_sampler: None,
        }
    }

    /// Get settings.
    pub fn settings(&self) -> &SsaoSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: SsaoSettings) {
        self.settings = settings;
    }

    /// Set quality preset.
    pub fn set_quality(&mut self, quality: SsaoQuality) {
        self.settings.quality = quality;
    }

    /// Set radius.
    pub fn set_radius(&mut self, radius: f32) {
        self.settings.radius = radius.max(0.01);
    }

    /// Set intensity.
    pub fn set_intensity(&mut self, intensity: f32) {
        self.settings.intensity = intensity.max(0.0);
    }

    /// Update projection matrices (call before render).
    pub fn set_projection(&mut self, projection: [[f32; 4]; 4], inv_projection: [[f32; 4]; 4], near: f32, far: f32) {
        self.projection = projection;
        self.inv_projection = inv_projection;
        self.near = near;
        self.far = far;
    }

    fn sample_count(&self) -> u32 {
        match self.settings.quality {
            SsaoQuality::Low => 8,
            SsaoQuality::Medium => 16,
            SsaoQuality::High => 32,
            SsaoQuality::Ultra => 64,
        }
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.format = format;

        // Create samplers
        self.linear_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSAO Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.point_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSAO Point Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        // Generate sample kernel (hemisphere)
        let kernel = self.generate_sample_kernel(64);
        self.kernel_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSAO Kernel Buffer"),
            contents: bytemuck::cast_slice(&kernel),
            usage: wgpu::BufferUsages::STORAGE,
        }));

        // Generate noise texture (4x4 random rotation vectors)
        self.create_noise_texture(device, queue);

        // Create bind group layouts
        self.create_bind_group_layouts(device);

        // Create uniform buffer
        let uniform = SsaoUniform {
            projection: [[0.0; 4]; 4],
            inv_projection: [[0.0; 4]; 4],
            resolution: [1.0 / width as f32, 1.0 / height as f32, width as f32, height as f32],
            params: [self.settings.radius, self.settings.intensity, self.settings.bias, self.sample_count() as f32],
            near_far: [0.1, 100.0, 0.0, 0.0],
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSAO Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSAO Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipelines
        self.create_pipelines(device, format);

        // Create intermediate textures
        self.create_textures(device);
    }

    fn generate_sample_kernel(&self, count: usize) -> Vec<[f32; 4]> {
        let mut kernel = Vec::with_capacity(count);

        for i in 0..count {
            // Generate random point in hemisphere using Halton-like sequence
            let xi1 = Self::halton(i as u32 + 1, 2);
            let xi2 = Self::halton(i as u32 + 1, 3);

            // Convert to hemisphere direction
            let phi = 2.0 * std::f32::consts::PI * xi1;
            let cos_theta = 1.0 - xi2;
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

            let x = sin_theta * phi.cos();
            let y = sin_theta * phi.sin();
            let z = cos_theta; // Hemisphere pointing in +Z

            // Scale samples to be distributed more towards center
            let scale = (i as f32) / (count as f32);
            let scale = 0.1 + scale * scale * 0.9; // lerp(0.1, 1.0, scale^2)

            kernel.push([x * scale, y * scale, z * scale, 0.0]);
        }

        kernel
    }

    fn halton(index: u32, base: u32) -> f32 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f32;
        let mut i = index;

        while i > 0 {
            result += f * (i % base) as f32;
            i /= base;
            f /= base as f32;
        }

        result
    }

    fn create_noise_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // 4x4 noise texture with random rotation vectors
        let mut noise_data = Vec::with_capacity(16 * 4);

        for i in 0..16 {
            let angle = (i as f32 / 16.0) * std::f32::consts::PI * 2.0 +
                        Self::halton(i as u32 + 1, 5) * std::f32::consts::PI;
            let x = angle.cos();
            let y = angle.sin();
            // Store as RGBA8
            noise_data.push(((x * 0.5 + 0.5) * 255.0) as u8);
            noise_data.push(((y * 0.5 + 0.5) * 255.0) as u8);
            noise_data.push(0u8);
            noise_data.push(255u8);
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Noise Texture"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * 4),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );

        self.noise_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.noise_texture = Some(texture);
    }

    fn create_bind_group_layouts(&mut self, device: &wgpu::Device) {
        // SSAO pass: depth + noise + kernel + uniforms
        self.ssao_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Bind Group Layout"),
            entries: &[
                // Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth sampler (non-filtering for depth textures)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Noise texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Noise sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Sample kernel
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        // Blur pass: ssao texture + uniforms
        self.blur_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Blur Bind Group Layout"),
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

        // Composite pass: scene + ssao + uniforms
        self.composite_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Composite Bind Group Layout"),
            entries: &[
                // Scene texture
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
                // SSAO texture
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
        // SSAO pipeline
        let ssao_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAO Pipeline Layout"),
            bind_group_layouts: &[self.ssao_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAO Shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_SHADER.into()),
        });

        self.ssao_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Pipeline"),
            layout: Some(&ssao_layout),
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
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

        // Blur pipeline
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAO Blur Pipeline Layout"),
            bind_group_layouts: &[self.blur_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAO Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(BLUR_SHADER.into()),
        });

        self.blur_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Blur Pipeline"),
            layout: Some(&blur_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
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

        // Composite pipeline
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAO Composite Pipeline Layout"),
            bind_group_layouts: &[self.composite_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAO Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });

        self.composite_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Composite Pipeline"),
            layout: Some(&composite_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
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
        // SSAO output texture (single channel)
        let ssao_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.ssao_view = Some(ssao_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.ssao_texture = Some(ssao_texture);

        // Blur output texture
        let blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Blur Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.blur_view = Some(blur_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.blur_texture = Some(blur_texture);
    }

    fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let uniform = SsaoUniform {
                projection: self.projection,
                inv_projection: self.inv_projection,
                resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
                params: [self.settings.radius, self.settings.intensity, self.settings.bias, self.sample_count() as f32],
                near_far: [self.near, self.far, 0.0, 0.0],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }

    /// Render SSAO effect.
    /// Takes depth texture as input and scene texture, outputs composited result.
    pub fn render_with_depth(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        scene_view: &wgpu::TextureView,
        output: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if !self.enabled {
            // Just copy scene to output
            self.blit(encoder, scene_view, output, device, queue);
            return;
        }

        let Some(ref linear_sampler) = self.linear_sampler else { return };
        let Some(ref point_sampler) = self.point_sampler else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref kernel_buffer) = self.kernel_buffer else { return };
        let Some(ref noise_view) = self.noise_view else { return };
        let Some(ref ssao_view) = self.ssao_view else { return };
        let Some(ref blur_view) = self.blur_view else { return };

        self.update_uniforms(queue);

        // Pass 1: Generate SSAO
        {
            let Some(ref ssao_pipeline) = self.ssao_pipeline else { return };
            let Some(ref ssao_layout) = self.ssao_bind_group_layout else { return };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSAO Bind Group"),
                layout: ssao_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(point_sampler), // Non-filtering for depth
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(noise_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(point_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: kernel_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ssao_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(ssao_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 2: Blur SSAO
        {
            let Some(ref blur_pipeline) = self.blur_pipeline else { return };
            let Some(ref blur_layout) = self.blur_bind_group_layout else { return };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSAO Blur Bind Group"),
                layout: blur_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(ssao_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Blur Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: blur_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(blur_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 3: Composite
        {
            let Some(ref composite_pipeline) = self.composite_pipeline else { return };
            let Some(ref composite_layout) = self.composite_bind_group_layout else { return };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSAO Composite Bind Group"),
                layout: composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(blur_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Composite Pass"),
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

            pass.set_pipeline(composite_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
    }

    fn blit(&self, encoder: &mut wgpu::CommandEncoder, input: &wgpu::TextureView, output: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(ref composite_pipeline) = self.composite_pipeline else { return };
        let Some(ref composite_layout) = self.composite_bind_group_layout else { return };
        let Some(ref linear_sampler) = self.linear_sampler else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref blur_view) = self.blur_view else { return };

        // Write zero intensity uniform so composite shader just passes through
        let uniform = SsaoUniform {
            projection: self.projection,
            inv_projection: self.inv_projection,
            resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
            params: [self.settings.radius, 0.0, self.settings.bias, self.sample_count() as f32], // intensity = 0
            near_far: [self.near, self.far, 0.0, 0.0],
        };
        queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        // Use composite shader but with intensity 0 (no darkening)
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAO Blit Bind Group"),
            layout: composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(blur_view), // Will be ignored with intensity 0
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSAO Blit Pass"),
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

        pass.set_pipeline(composite_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, quad_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }
}

impl Default for SsaoPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for SsaoPass {
    fn name(&self) -> &str {
        "ssao"
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
        _encoder: &mut wgpu::CommandEncoder,
        _input: &wgpu::TextureView,
        _output: &wgpu::TextureView,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        // SSAO requires depth buffer, use render_with_depth instead
    }
}

// SSAO shader - samples depth buffer to compute ambient occlusion
const SSAO_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    resolution: vec4<f32>,  // 1/w, 1/h, w, h
    params: vec4<f32>,      // radius, intensity, bias, sample_count
    near_far: vec4<f32>,    // near, far, 0, 0
}

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var depth_sampler: sampler;
@group(0) @binding(2) var noise_texture: texture_2d<f32>;
@group(0) @binding(3) var noise_sampler: sampler;
@group(0) @binding(4) var<storage, read> kernel: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

// Reconstruct view-space position from depth
fn viewPositionFromDepth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert UV to NDC
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    // Create clip space position
    let clip = vec4<f32>(ndc, depth, 1.0);

    // Transform to view space
    var view = params.inv_projection * clip;
    view = view / view.w;

    return view.xyz;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let radius = params.params.x;
    let intensity = params.params.y;
    let bias = params.params.z;
    let sample_count = i32(params.params.w);
    let texel = params.resolution.xy;

    // Sample depth at center
    let depth = textureSampleLevel(depth_texture, depth_sampler, in.uv, 0);

    // Skip sky/background
    if (depth >= 1.0) {
        return 1.0;
    }

    // Reconstruct view-space position
    let fragPos = viewPositionFromDepth(in.uv, depth);

    // Compute normal from depth derivatives (approximation)
    let depthL = textureSampleLevel(depth_texture, depth_sampler, in.uv - vec2<f32>(texel.x, 0.0), 0);
    let depthR = textureSampleLevel(depth_texture, depth_sampler, in.uv + vec2<f32>(texel.x, 0.0), 0);
    let depthT = textureSampleLevel(depth_texture, depth_sampler, in.uv - vec2<f32>(0.0, texel.y), 0);
    let depthB = textureSampleLevel(depth_texture, depth_sampler, in.uv + vec2<f32>(0.0, texel.y), 0);

    let posL = viewPositionFromDepth(in.uv - vec2<f32>(texel.x, 0.0), depthL);
    let posR = viewPositionFromDepth(in.uv + vec2<f32>(texel.x, 0.0), depthR);
    let posT = viewPositionFromDepth(in.uv - vec2<f32>(0.0, texel.y), depthT);
    let posB = viewPositionFromDepth(in.uv + vec2<f32>(0.0, texel.y), depthB);

    let normal = normalize(cross(posR - posL, posB - posT));

    // Sample noise for random rotation
    let noiseScale = params.resolution.zw / 4.0;
    let noise = textureSampleLevel(noise_texture, noise_sampler, in.uv * noiseScale, 0.0).xy * 2.0 - 1.0;

    // Create TBN matrix for hemisphere orientation
    let tangent = normalize(vec3<f32>(noise.x, noise.y, 0.0) - normal * dot(vec3<f32>(noise.x, noise.y, 0.0), normal));
    let bitangent = cross(normal, tangent);
    let TBN = mat3x3<f32>(tangent, bitangent, normal);

    // Sample hemisphere
    var occlusion = 0.0;
    for (var i = 0; i < sample_count; i++) {
        // Get sample position
        let sampleVec = TBN * kernel[i].xyz;
        let samplePos = fragPos + sampleVec * radius;

        // Project sample to screen space
        var offset = params.projection * vec4<f32>(samplePos, 1.0);
        offset = offset / offset.w;
        let sampleUV = vec2<f32>(offset.x * 0.5 + 0.5, 0.5 - offset.y * 0.5);

        // Sample depth at offset position
        let sampleDepth = textureSampleLevel(depth_texture, depth_sampler, sampleUV, 0);
        let sampleViewPos = viewPositionFromDepth(sampleUV, sampleDepth);

        // Range check and accumulate
        let rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleViewPos.z));

        // Compare depths
        if (sampleViewPos.z >= samplePos.z + bias) {
            occlusion += rangeCheck;
        }
    }

    occlusion = 1.0 - (occlusion / f32(sample_count)) * intensity;

    return occlusion;
}
"#;

// Blur shader - bilateral blur to preserve edges
const BLUR_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    resolution: vec4<f32>,
    params: vec4<f32>,
    near_far: vec4<f32>,
}

@group(0) @binding(0) var ssao_texture: texture_2d<f32>;
@group(0) @binding(1) var ssao_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let texel = params.resolution.xy;

    // 4x4 box blur
    var result = 0.0;
    for (var x = -2; x <= 2; x++) {
        for (var y = -2; y <= 2; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            result += textureSampleLevel(ssao_texture, ssao_sampler, in.uv + offset, 0.0).r;
        }
    }

    return result / 25.0;
}
"#;

// Composite shader - multiply scene by SSAO
const COMPOSITE_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    resolution: vec4<f32>,
    params: vec4<f32>,
    near_far: vec4<f32>,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var ssao_texture: texture_2d<f32>;
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
    let scene = textureSampleLevel(scene_texture, tex_sampler, in.uv, 0.0);
    let intensity = params.params.y;

    // Skip AO when intensity is 0 (used for passthrough blit)
    if (intensity <= 0.0) {
        return scene;
    }

    let ao = textureSampleLevel(ssao_texture, tex_sampler, in.uv, 0.0).r;

    // Apply AO to scene
    return vec4<f32>(scene.rgb * ao, scene.a);
}
"#;
