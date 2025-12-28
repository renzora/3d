//! Lumen-style Screen-Space Global Illumination (SSGI) post-processing effect.
//!
//! Implements indirect diffuse lighting through screen-space ray tracing:
//! 1. SSGI Pass: Trace rays in hemisphere, sample scene radiance at hits
//! 2. Denoise Pass: Edge-aware bilateral blur
//! 3. Temporal Pass: Accumulate with history for stability
//! 4. Composite Pass: Add GI to scene

use crate::postprocessing::pass::{FullscreenVertex, Pass, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// Lumen quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LumenQuality {
    /// Low quality - 4 rays, 8 steps.
    Low,
    /// Medium quality - 8 rays, 12 steps.
    #[default]
    Medium,
    /// High quality - 12 rays, 16 steps.
    High,
    /// Ultra quality - 16 rays, 24 steps.
    Ultra,
}

impl LumenQuality {
    /// Get the number of rays for this quality level.
    pub fn ray_count(&self) -> u32 {
        match self {
            Self::Low => 4,
            Self::Medium => 8,
            Self::High => 12,
            Self::Ultra => 16,
        }
    }

    /// Get the number of steps per ray for this quality level.
    pub fn step_count(&self) -> u32 {
        match self {
            Self::Low => 8,
            Self::Medium => 12,
            Self::High => 16,
            Self::Ultra => 24,
        }
    }
}

/// Lumen settings.
#[derive(Debug, Clone)]
pub struct LumenSettings {
    /// Quality preset (affects ray/step count).
    pub quality: LumenQuality,
    /// Maximum trace distance in world units.
    pub max_distance: f32,
    /// Depth thickness for hit detection.
    pub thickness: f32,
    /// GI intensity multiplier.
    pub intensity: f32,
    /// Enable indirect diffuse.
    pub indirect_diffuse: bool,
}

impl Default for LumenSettings {
    fn default() -> Self {
        Self {
            quality: LumenQuality::Medium,
            max_distance: 5.0,
            thickness: 0.1,
            intensity: 1.0,
            indirect_diffuse: true,
        }
    }
}

/// Lumen uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LumenUniform {
    /// Projection matrix.
    projection: [[f32; 4]; 4],
    /// Inverse projection matrix.
    inv_projection: [[f32; 4]; 4],
    /// View matrix.
    view: [[f32; 4]; 4],
    /// Previous frame view-projection (for reprojection).
    prev_view_proj: [[f32; 4]; 4],
    /// 1/width, 1/height, width, height
    resolution: [f32; 4],
    /// max_distance, thickness, intensity, ray_count
    params: [f32; 4],
    /// step_count, frame_index, near, far
    params2: [f32; 4],
}

/// Lumen GI post-processing pass.
pub struct LumenPass {
    enabled: bool,
    settings: LumenSettings,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    frame_index: u32,
    // Camera data
    projection: [[f32; 4]; 4],
    inv_projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    near: f32,
    far: f32,
    // Pipelines
    ssgi_pipeline: Option<wgpu::RenderPipeline>,
    denoise_pipeline: Option<wgpu::RenderPipeline>,
    temporal_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    // Bind group layouts
    ssgi_bind_group_layout: Option<wgpu::BindGroupLayout>,
    denoise_bind_group_layout: Option<wgpu::BindGroupLayout>,
    temporal_bind_group_layout: Option<wgpu::BindGroupLayout>,
    composite_bind_group_layout: Option<wgpu::BindGroupLayout>,
    // Textures
    gi_texture: Option<wgpu::Texture>,
    gi_view: Option<wgpu::TextureView>,
    denoised_texture: Option<wgpu::Texture>,
    denoised_view: Option<wgpu::TextureView>,
    history_texture: Option<wgpu::Texture>,
    history_view: Option<wgpu::TextureView>,
    temporal_output_texture: Option<wgpu::Texture>,
    temporal_output_view: Option<wgpu::TextureView>,
    prev_depth_texture: Option<wgpu::Texture>,
    prev_depth_view: Option<wgpu::TextureView>,
    // Buffers
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    // Samplers
    linear_sampler: Option<wgpu::Sampler>,
    point_sampler: Option<wgpu::Sampler>,
}

impl LumenPass {
    /// Create a new Lumen pass.
    pub fn new() -> Self {
        Self {
            enabled: false, // Disabled by default (expensive)
            settings: LumenSettings::default(),
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            frame_index: 0,
            projection: [[0.0; 4]; 4],
            inv_projection: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            prev_view_proj: [[0.0; 4]; 4],
            near: 0.1,
            far: 100.0,
            ssgi_pipeline: None,
            denoise_pipeline: None,
            temporal_pipeline: None,
            composite_pipeline: None,
            ssgi_bind_group_layout: None,
            denoise_bind_group_layout: None,
            temporal_bind_group_layout: None,
            composite_bind_group_layout: None,
            gi_texture: None,
            gi_view: None,
            denoised_texture: None,
            denoised_view: None,
            history_texture: None,
            history_view: None,
            temporal_output_texture: None,
            temporal_output_view: None,
            prev_depth_texture: None,
            prev_depth_view: None,
            uniform_buffer: None,
            quad_buffer: None,
            linear_sampler: None,
            point_sampler: None,
        }
    }

    /// Get settings.
    pub fn settings(&self) -> &LumenSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: LumenSettings) {
        self.settings = settings;
    }

    /// Set quality preset.
    pub fn set_quality(&mut self, quality: LumenQuality) {
        self.settings.quality = quality;
    }

    /// Set max trace distance.
    pub fn set_max_distance(&mut self, distance: f32) {
        self.settings.max_distance = distance.max(0.1);
    }

    /// Set GI intensity.
    pub fn set_intensity(&mut self, intensity: f32) {
        self.settings.intensity = intensity.max(0.0);
    }

    /// Set thickness.
    pub fn set_thickness(&mut self, thickness: f32) {
        self.settings.thickness = thickness.clamp(0.01, 1.0);
    }

    /// Update camera matrices (call before render).
    pub fn set_matrices(
        &mut self,
        projection: [[f32; 4]; 4],
        inv_projection: [[f32; 4]; 4],
        view: [[f32; 4]; 4],
        near: f32,
        far: f32,
    ) {
        // Store previous view_proj for reprojection
        self.prev_view_proj = self.compute_view_proj();
        self.projection = projection;
        self.inv_projection = inv_projection;
        self.view = view;
        self.near = near;
        self.far = far;
    }

    fn compute_view_proj(&self) -> [[f32; 4]; 4] {
        // Simple matrix multiply (view * proj)
        let mut result = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += self.projection[i][k] * self.view[k][j];
                }
            }
        }
        result
    }

    /// Initialize GPU resources.
    pub fn init(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) {
        self.width = width;
        self.height = height;
        self.format = format;

        // Create samplers
        self.linear_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Lumen Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.point_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Lumen Point Sampler"),
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
        let uniform = LumenUniform {
            projection: [[0.0; 4]; 4],
            inv_projection: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            prev_view_proj: [[0.0; 4]; 4],
            resolution: [
                1.0 / width as f32,
                1.0 / height as f32,
                width as f32,
                height as f32,
            ],
            params: [
                self.settings.max_distance,
                self.settings.thickness,
                self.settings.intensity,
                self.settings.quality.ray_count() as f32,
            ],
            params2: [
                self.settings.quality.step_count() as f32,
                0.0,
                0.1,
                100.0,
            ],
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lumen Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lumen Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipelines
        self.create_pipelines(device, format);

        // Create textures
        self.create_textures(device);
    }

    fn create_bind_group_layouts(&mut self, device: &wgpu::Device) {
        // SSGI pass: depth + scene + samplers + uniforms
        self.ssgi_bind_group_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lumen SSGI Bind Group Layout"),
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
                    // Scene texture
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
                    // Point sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // Linear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
            }));

        // Denoise pass: GI + depth + samplers + uniforms
        self.denoise_bind_group_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lumen Denoise Bind Group Layout"),
                entries: &[
                    // GI texture
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
                    // Linear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Point sampler
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
            }));

        // Temporal pass: current + history + depth + prev_depth + samplers + uniforms
        self.temporal_bind_group_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lumen Temporal Bind Group Layout"),
                entries: &[
                    // Current GI
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
                    // History GI
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
                    // Current depth
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Previous depth (stored as R16Float)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Linear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Point sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // Uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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

        // Composite pass: scene + GI + sampler + uniforms
        self.composite_bind_group_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lumen Composite Bind Group Layout"),
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
                    // GI texture
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
                    // Linear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Uniforms
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
        // SSGI pipeline
        let ssgi_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lumen SSGI Pipeline Layout"),
            bind_group_layouts: &[self.ssgi_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let ssgi_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lumen SSGI Shader"),
            source: wgpu::ShaderSource::Wgsl(SSGI_SHADER.into()),
        });

        self.ssgi_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lumen SSGI Pipeline"),
            layout: Some(&ssgi_layout),
            vertex: wgpu::VertexState {
                module: &ssgi_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssgi_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
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

        // Denoise pipeline
        let denoise_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lumen Denoise Pipeline Layout"),
            bind_group_layouts: &[self.denoise_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let denoise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lumen Denoise Shader"),
            source: wgpu::ShaderSource::Wgsl(DENOISE_SHADER.into()),
        });

        self.denoise_pipeline =
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Lumen Denoise Pipeline"),
                layout: Some(&denoise_layout),
                vertex: wgpu::VertexState {
                    module: &denoise_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[FullscreenVertex::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &denoise_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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

        // Temporal pipeline
        let temporal_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lumen Temporal Pipeline Layout"),
            bind_group_layouts: &[self.temporal_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lumen Temporal Shader"),
            source: wgpu::ShaderSource::Wgsl(TEMPORAL_SHADER.into()),
        });

        self.temporal_pipeline =
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Lumen Temporal Pipeline"),
                layout: Some(&temporal_layout),
                vertex: wgpu::VertexState {
                    module: &temporal_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[FullscreenVertex::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &temporal_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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
            label: Some("Lumen Composite Pipeline Layout"),
            bind_group_layouts: &[self.composite_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lumen Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });

        self.composite_pipeline =
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Lumen Composite Pipeline"),
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
        // GI output texture (RGBA16F for HDR)
        let gi_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Lumen GI Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.gi_view = Some(gi_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.gi_texture = Some(gi_texture);

        // Denoised texture
        let denoised_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Lumen Denoised Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.denoised_view =
            Some(denoised_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.denoised_texture = Some(denoised_texture);

        // History texture
        let history_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Lumen History Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.history_view =
            Some(history_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.history_texture = Some(history_texture);

        // Temporal output texture
        let temporal_output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Lumen Temporal Output Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.temporal_output_view =
            Some(temporal_output_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.temporal_output_texture = Some(temporal_output_texture);

        // Previous depth texture (R16Float for filterability)
        let prev_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Lumen Previous Depth Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.prev_depth_view =
            Some(prev_depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.prev_depth_texture = Some(prev_depth_texture);
    }

    fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let uniform = LumenUniform {
                projection: self.projection,
                inv_projection: self.inv_projection,
                view: self.view,
                prev_view_proj: self.prev_view_proj,
                resolution: [
                    1.0 / self.width as f32,
                    1.0 / self.height as f32,
                    self.width as f32,
                    self.height as f32,
                ],
                params: [
                    self.settings.max_distance,
                    self.settings.thickness,
                    self.settings.intensity,
                    self.settings.quality.ray_count() as f32,
                ],
                params2: [
                    self.settings.quality.step_count() as f32,
                    self.frame_index as f32,
                    self.near,
                    self.far,
                ],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }

    /// Render Lumen GI effect.
    pub fn render_with_depth(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        scene_view: &wgpu::TextureView,
        output: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.frame_index = self.frame_index.wrapping_add(1);

        if !self.enabled {
            self.blit(encoder, scene_view, output, device, queue);
            return;
        }

        let Some(ref linear_sampler) = self.linear_sampler else {
            return;
        };
        let Some(ref point_sampler) = self.point_sampler else {
            return;
        };
        let Some(ref quad_buffer) = self.quad_buffer else {
            return;
        };
        let Some(ref uniform_buffer) = self.uniform_buffer else {
            return;
        };
        let Some(ref gi_view) = self.gi_view else {
            return;
        };
        let Some(ref denoised_view) = self.denoised_view else {
            return;
        };
        let Some(ref history_view) = self.history_view else {
            return;
        };
        let Some(ref temporal_output_view) = self.temporal_output_view else {
            return;
        };
        let Some(ref prev_depth_view) = self.prev_depth_view else {
            return;
        };

        self.update_uniforms(queue);

        // Pass 1: SSGI ray tracing
        {
            let Some(ref ssgi_pipeline) = self.ssgi_pipeline else {
                return;
            };
            let Some(ref ssgi_layout) = self.ssgi_bind_group_layout else {
                return;
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Lumen SSGI Bind Group"),
                layout: ssgi_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(point_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lumen SSGI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: gi_view,
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

            pass.set_pipeline(ssgi_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 2: Spatial denoise
        {
            let Some(ref denoise_pipeline) = self.denoise_pipeline else {
                return;
            };
            let Some(ref denoise_layout) = self.denoise_bind_group_layout else {
                return;
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Lumen Denoise Bind Group"),
                layout: denoise_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(gi_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(point_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lumen Denoise Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: denoised_view,
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

            pass.set_pipeline(denoise_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 3: Temporal accumulation
        {
            let Some(ref temporal_pipeline) = self.temporal_pipeline else {
                return;
            };
            let Some(ref temporal_layout) = self.temporal_bind_group_layout else {
                return;
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Lumen Temporal Bind Group"),
                layout: temporal_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(denoised_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(history_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(prev_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(point_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lumen Temporal Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: temporal_output_view,
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

            pass.set_pipeline(temporal_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Copy temporal output to history
        if let (Some(ref temporal_output), Some(ref history)) =
            (&self.temporal_output_texture, &self.history_texture)
        {
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: temporal_output,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: history,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Pass 4: Composite
        {
            let Some(ref composite_pipeline) = self.composite_pipeline else {
                return;
            };
            let Some(ref composite_layout) = self.composite_bind_group_layout else {
                return;
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Lumen Composite Bind Group"),
                layout: composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(temporal_output_view),
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
                label: Some("Lumen Composite Pass"),
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

    fn blit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let Some(ref composite_pipeline) = self.composite_pipeline else {
            return;
        };
        let Some(ref composite_layout) = self.composite_bind_group_layout else {
            return;
        };
        let Some(ref linear_sampler) = self.linear_sampler else {
            return;
        };
        let Some(ref quad_buffer) = self.quad_buffer else {
            return;
        };
        let Some(ref uniform_buffer) = self.uniform_buffer else {
            return;
        };
        let Some(ref temporal_output_view) = self.temporal_output_view else {
            return;
        };

        // Write zero intensity so composite just passes through
        let uniform = LumenUniform {
            projection: self.projection,
            inv_projection: self.inv_projection,
            view: self.view,
            prev_view_proj: self.prev_view_proj,
            resolution: [
                1.0 / self.width as f32,
                1.0 / self.height as f32,
                self.width as f32,
                self.height as f32,
            ],
            params: [self.settings.max_distance, self.settings.thickness, 0.0, 0.0], // intensity = 0
            params2: [0.0, self.frame_index as f32, self.near, self.far],
        };
        queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lumen Blit Bind Group"),
            layout: composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(temporal_output_view),
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
            label: Some("Lumen Blit Pass"),
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

impl Default for LumenPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for LumenPass {
    fn name(&self) -> &str {
        "lumen"
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
        // Use render_with_depth instead
    }
}

// ============================================================================
// SHADERS
// ============================================================================

// SSGI shader - traces rays in hemisphere and samples scene radiance
const SSGI_SHADER: &str = r#"
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
    view: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    resolution: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
}

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var point_sampler: sampler;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var<uniform> params: Params;

const PI: f32 = 3.14159265359;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

// Better hash function for less structured noise
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Hash-based 2D random with temporal variation
fn random2(pixel: vec2<f32>, frame: f32, offset: f32) -> vec2<f32> {
    let p1 = pixel + vec2<f32>(frame * 1.618033988749895, offset);
    let p2 = pixel + vec2<f32>(offset * 2.718281828459045, frame * 3.141592653589793);
    return vec2<f32>(hash(p1), hash(p2));
}

// Get view-space position from depth
fn get_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    var view_pos = params.inv_projection * ndc;
    view_pos /= view_pos.w;
    return view_pos.xyz;
}

// Reconstruct normal from depth
fn get_normal(uv: vec2<f32>, center_depth: f32) -> vec3<f32> {
    let texel = params.resolution.xy;

    let depth_r = textureSampleLevel(depth_texture, point_sampler, uv + vec2<f32>(texel.x, 0.0), 0);
    let depth_l = textureSampleLevel(depth_texture, point_sampler, uv - vec2<f32>(texel.x, 0.0), 0);
    let depth_u = textureSampleLevel(depth_texture, point_sampler, uv + vec2<f32>(0.0, texel.y), 0);
    let depth_d = textureSampleLevel(depth_texture, point_sampler, uv - vec2<f32>(0.0, texel.y), 0);

    let pos_c = get_view_pos(uv, center_depth);
    let pos_r = get_view_pos(uv + vec2<f32>(texel.x, 0.0), depth_r);
    let pos_l = get_view_pos(uv - vec2<f32>(texel.x, 0.0), depth_l);
    let pos_u = get_view_pos(uv + vec2<f32>(0.0, texel.y), depth_u);
    let pos_d = get_view_pos(uv - vec2<f32>(0.0, texel.y), depth_d);

    // Use smallest difference for more accurate normals at edges
    let dx = select(pos_r - pos_c, pos_c - pos_l, abs(depth_r - center_depth) > abs(depth_l - center_depth));
    let dy = select(pos_u - pos_c, pos_c - pos_d, abs(depth_u - center_depth) > abs(depth_d - center_depth));

    return normalize(cross(dx, dy));
}

// Project view position to screen UV
fn project_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    var clip = params.projection * vec4<f32>(view_pos, 1.0);
    clip /= clip.w;
    return vec3<f32>(clip.xy * 0.5 + 0.5, clip.z);
}

// Cosine-weighted hemisphere sampling
fn cosine_sample_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let theta = 2.0 * PI * u2;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - u1));
    return vec3<f32>(x, y, z);
}

// Create TBN matrix from normal
fn create_tbn(n: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.y) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return mat3x3<f32>(tangent, bitangent, n);
}

// Trace a single GI ray
fn trace_gi_ray(origin: vec3<f32>, direction: vec3<f32>, step_count: i32) -> vec4<f32> {
    let max_distance = params.params.x;
    let thickness = params.params.y;

    let step_size = max_distance / f32(step_count);
    var ray_pos = origin;

    for (var i = 0; i < step_count; i++) {
        ray_pos += direction * step_size;

        // Project to screen
        let screen = project_to_screen(ray_pos);
        let uv = screen.xy;

        // Check bounds
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }

        // Sample depth
        let sampled_depth = textureSampleLevel(depth_texture, point_sampler, uv, 0);

        // Sky hit - sample sky color for ambient contribution
        if (sampled_depth >= 0.9999) {
            let sky_color = textureSampleLevel(scene_texture, linear_sampler, uv, 0.0).rgb;
            // Reduced intensity for sky contribution (acts as ambient)
            let sky_intensity = 0.3;
            let t = f32(i) / f32(step_count);
            let distance_fade = 1.0 - t;
            return vec4<f32>(sky_color * sky_intensity * distance_fade, 1.0);
        }

        let sampled_pos = get_view_pos(uv, sampled_depth);

        // Check for hit
        let depth_diff = ray_pos.z - sampled_pos.z;
        if (depth_diff > 0.0 && depth_diff < thickness) {
            // Hit! Sample scene color
            let hit_color = textureSampleLevel(scene_texture, linear_sampler, uv, 0.0).rgb;

            // Distance fade
            let t = f32(i) / f32(step_count);
            let distance_fade = 1.0 - t * t;

            // Edge fade (avoid sampling at screen edges)
            let edge_fade = smoothstep(0.0, 0.1, uv.x) * smoothstep(1.0, 0.9, uv.x) *
                           smoothstep(0.0, 0.1, uv.y) * smoothstep(1.0, 0.9, uv.y);

            return vec4<f32>(hit_color * distance_fade * edge_fade, 1.0);
        }
    }

    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_count = i32(params.params.w);
    let step_count = i32(params.params2.x);
    let frame = params.params2.y;

    let depth = textureSampleLevel(depth_texture, point_sampler, in.uv, 0);

    // Skip sky
    if (depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let view_pos = get_view_pos(in.uv, depth);
    let normal = get_normal(in.uv, depth);
    let tbn = create_tbn(normal);

    var accumulated_gi = vec3<f32>(0.0);
    var total_weight = 0.0;

    // Trace multiple rays
    for (var i = 0; i < ray_count; i++) {
        // Generate random direction using better hash
        let rnd = random2(in.position.xy, frame, f32(i) * 17.31);

        // Cosine-weighted hemisphere sample
        let local_dir = cosine_sample_hemisphere(rnd.x, rnd.y);
        let world_dir = tbn * local_dir;

        // Weight by cos(theta) - already built into cosine sampling
        let weight = local_dir.z;

        // Trace ray
        let hit = trace_gi_ray(view_pos + normal * 0.01, world_dir, step_count);

        if (hit.w > 0.0) {
            accumulated_gi += hit.rgb * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0) {
        accumulated_gi /= total_weight;
    }

    return vec4<f32>(accumulated_gi, 1.0);
}
"#;

// Denoise shader - edge-aware bilateral blur
const DENOISE_SHADER: &str = r#"
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
    view: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    resolution: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
}

@group(0) @binding(0) var gi_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var point_sampler: sampler;
@group(0) @binding(4) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fn linearize_depth(d: f32) -> f32 {
    let near = params.params2.z;
    let far = params.params2.w;
    return near * far / (far - d * (far - near));
}

// Reconstruct normal from depth
fn get_normal(uv: vec2<f32>) -> vec3<f32> {
    let texel = params.resolution.xy;
    let center = textureSampleLevel(depth_texture, point_sampler, uv, 0);
    let left = textureSampleLevel(depth_texture, point_sampler, uv - vec2<f32>(texel.x, 0.0), 0);
    let right = textureSampleLevel(depth_texture, point_sampler, uv + vec2<f32>(texel.x, 0.0), 0);
    let up = textureSampleLevel(depth_texture, point_sampler, uv - vec2<f32>(0.0, texel.y), 0);
    let down = textureSampleLevel(depth_texture, point_sampler, uv + vec2<f32>(0.0, texel.y), 0);

    let dx = (right - left) * 0.5;
    let dy = (down - up) * 0.5;

    return normalize(vec3<f32>(-dx, -dy, 1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = params.resolution.xy;
    let center_gi = textureSampleLevel(gi_texture, linear_sampler, in.uv, 0.0);
    let center_depth = textureSampleLevel(depth_texture, point_sampler, in.uv, 0);

    // Skip sky
    if (center_depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let center_linear_depth = linearize_depth(center_depth);
    let center_normal = get_normal(in.uv);

    // Edge-aware bilateral blur (larger kernel for aggressive denoising)
    var total_gi = vec3<f32>(0.0);
    var total_weight = 0.0;

    let blur_radius = 5;
    let sigma_spatial = 4.0;
    let sigma_depth = 0.15;
    let sigma_normal = 0.7;

    for (var y = -blur_radius; y <= blur_radius; y++) {
        for (var x = -blur_radius; x <= blur_radius; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let sample_uv = in.uv + offset;

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }

            let sample_gi = textureSampleLevel(gi_texture, linear_sampler, sample_uv, 0.0);
            let sample_depth = textureSampleLevel(depth_texture, point_sampler, sample_uv, 0);
            let sample_linear_depth = linearize_depth(sample_depth);

            // Spatial weight
            let dist2 = f32(x * x + y * y);
            let spatial_weight = exp(-dist2 / (2.0 * sigma_spatial * sigma_spatial));

            // Depth weight
            let depth_diff = abs(center_linear_depth - sample_linear_depth) / center_linear_depth;
            let depth_weight = exp(-depth_diff * depth_diff / (2.0 * sigma_depth * sigma_depth));

            // Normal weight
            let sample_normal = get_normal(sample_uv);
            let normal_diff = 1.0 - max(dot(center_normal, sample_normal), 0.0);
            let normal_weight = exp(-normal_diff * normal_diff / (2.0 * sigma_normal * sigma_normal));

            let weight = spatial_weight * depth_weight * normal_weight;
            total_gi += sample_gi.rgb * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0) {
        total_gi /= total_weight;
    }

    return vec4<f32>(total_gi, 1.0);
}
"#;

// Temporal shader - accumulate with history
const TEMPORAL_SHADER: &str = r#"
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
    view: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    resolution: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
}

@group(0) @binding(0) var current_gi: texture_2d<f32>;
@group(0) @binding(1) var history_gi: texture_2d<f32>;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(3) var prev_depth_texture: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var point_sampler: sampler;
@group(0) @binding(6) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fn linearize_depth(d: f32) -> f32 {
    let near = params.params2.z;
    let far = params.params2.w;
    return near * far / (far - d * (far - near));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let current = textureSampleLevel(current_gi, linear_sampler, in.uv, 0.0);
    let history = textureSampleLevel(history_gi, linear_sampler, in.uv, 0.0);

    let current_depth = textureSampleLevel(depth_texture, point_sampler, in.uv, 0);
    let prev_depth = textureSampleLevel(prev_depth_texture, linear_sampler, in.uv, 0.0).r;

    // Skip sky
    if (current_depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Depth-based disocclusion detection
    let current_linear = linearize_depth(current_depth);
    let prev_linear = linearize_depth(prev_depth);

    let depth_diff = abs(current_linear - prev_linear) / max(current_linear, 0.001);
    let depth_threshold = 0.05;
    let depth_weight = 1.0 - saturate(depth_diff / depth_threshold);

    // Temporal blend (95% history when stable - more aggressive for noise reduction)
    let base_blend = 0.95;

    // Even during disocclusion, keep some temporal stability to avoid harsh noise
    let min_blend = 0.5; // Never go below 50% history to reduce visible noise
    let temporal_blend = mix(min_blend, base_blend, depth_weight);

    // Check if history is valid (more lenient check)
    let history_valid = select(0.7, 1.0, prev_depth > 0.0001);
    let effective_blend = temporal_blend * history_valid;

    // Neighborhood clamping with larger kernel (5x5)
    let texel = params.resolution.xy;
    var gi_min = current.rgb;
    var gi_max = current.rgb;

    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            if (x == 0 && y == 0) { continue; }
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let neighbor = textureSampleLevel(current_gi, linear_sampler, in.uv + offset, 0.0).rgb;
            gi_min = min(gi_min, neighbor);
            gi_max = max(gi_max, neighbor);
        }
    }

    // Expand bounds more generously to allow temporal accumulation
    let margin = 0.25;
    gi_min = gi_min - margin;
    gi_max = gi_max + margin;

    // Clamp history
    let clamped_history = clamp(history.rgb, gi_min, gi_max);

    // Blend
    let result = mix(current.rgb, clamped_history, effective_blend);

    return vec4<f32>(result, 1.0);
}
"#;

// Composite shader - add GI to scene
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
    view: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    resolution: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var gi_texture: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
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
    let intensity = params.params.z;

    let scene = textureSampleLevel(scene_texture, linear_sampler, in.uv, 0.0);
    let gi = textureSampleLevel(gi_texture, linear_sampler, in.uv, 0.0);

    // Add GI to scene (indirect diffuse)
    // In a full implementation, we'd multiply by albedo here
    let result = scene.rgb + gi.rgb * intensity;

    return vec4<f32>(result, scene.a);
}
"#;
