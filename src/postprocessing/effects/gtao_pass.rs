//! GTAO (Ground Truth Ambient Occlusion) post-processing effect.
//!
//! Based on "Practical Realtime Strategies for Accurate Indirect Occlusion"
//! by Jorge Jimenez et al. GTAO uses horizon-based ray marching for more
//! physically accurate ambient occlusion than traditional SSAO.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// GTAO quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GtaoQuality {
    /// Low quality - 2 directions, 4 steps.
    Low,
    /// Medium quality - 4 directions, 6 steps.
    #[default]
    Medium,
    /// High quality - 6 directions, 8 steps.
    High,
    /// Ultra quality - 8 directions, 12 steps.
    Ultra,
}

impl GtaoQuality {
    /// Get the number of directions for this quality level.
    pub fn direction_count(&self) -> u32 {
        match self {
            Self::Low => 2,
            Self::Medium => 4,
            Self::High => 6,
            Self::Ultra => 8,
        }
    }

    /// Get the number of steps per direction for this quality level.
    pub fn step_count(&self) -> u32 {
        match self {
            Self::Low => 4,
            Self::Medium => 6,
            Self::High => 8,
            Self::Ultra => 12,
        }
    }
}

/// GTAO settings.
#[derive(Debug, Clone)]
pub struct GtaoSettings {
    /// Quality preset (affects direction/step count).
    pub quality: GtaoQuality,
    /// Effect radius in world units.
    pub radius: f32,
    /// Falloff start distance (0-1 of radius).
    pub falloff_start: f32,
    /// Intensity/strength of the occlusion.
    pub intensity: f32,
    /// Power applied to final AO (contrast adjustment).
    pub power: f32,
    /// Thin occluder heuristic strength.
    pub thin_occluder_compensation: f32,
}

impl Default for GtaoSettings {
    fn default() -> Self {
        Self {
            quality: GtaoQuality::Medium,
            radius: 0.5,
            falloff_start: 0.2,
            intensity: 1.5,
            power: 1.5,
            thin_occluder_compensation: 0.5,
        }
    }
}

/// GTAO uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GtaoUniform {
    /// Projection matrix for reconstructing positions.
    projection: [[f32; 4]; 4],
    /// Inverse projection matrix.
    inv_projection: [[f32; 4]; 4],
    /// View matrix (for world-space normals).
    view: [[f32; 4]; 4],
    /// 1/width, 1/height, width, height
    resolution: [f32; 4],
    /// radius, falloff_start, intensity, power
    params: [f32; 4],
    /// direction_count, step_count, thin_occluder_comp, frame_index
    params2: [f32; 4],
    /// near, far, aspect, fov_tan
    camera_params: [f32; 4],
}

/// GTAO post-processing pass.
pub struct GtaoPass {
    enabled: bool,
    settings: GtaoSettings,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    frame_index: u32,
    // Projection matrices (updated each frame)
    projection: [[f32; 4]; 4],
    inv_projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    near: f32,
    far: f32,
    fov: f32,
    // Pipelines
    gtao_pipeline: Option<wgpu::RenderPipeline>,
    spatial_filter_pipeline: Option<wgpu::RenderPipeline>,
    temporal_filter_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    // Bind group layouts
    gtao_bind_group_layout: Option<wgpu::BindGroupLayout>,
    filter_bind_group_layout: Option<wgpu::BindGroupLayout>,
    temporal_bind_group_layout: Option<wgpu::BindGroupLayout>,
    composite_bind_group_layout: Option<wgpu::BindGroupLayout>,
    // Textures
    ao_texture: Option<wgpu::Texture>,
    ao_view: Option<wgpu::TextureView>,
    filtered_texture: Option<wgpu::Texture>,
    filtered_view: Option<wgpu::TextureView>,
    // Temporal history textures (ping-pong)
    history_texture: Option<wgpu::Texture>,
    history_view: Option<wgpu::TextureView>,
    temporal_output_texture: Option<wgpu::Texture>,
    temporal_output_view: Option<wgpu::TextureView>,
    // Previous frame depth for disocclusion detection
    prev_depth_texture: Option<wgpu::Texture>,
    prev_depth_view: Option<wgpu::TextureView>,
    // Buffers
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    // Samplers
    linear_sampler: Option<wgpu::Sampler>,
    point_sampler: Option<wgpu::Sampler>,
}

impl GtaoPass {
    /// Create a new GTAO pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: GtaoSettings::default(),
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            frame_index: 0,
            projection: [[0.0; 4]; 4],
            inv_projection: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            near: 0.1,
            far: 100.0,
            fov: 45.0_f32.to_radians(),
            gtao_pipeline: None,
            spatial_filter_pipeline: None,
            temporal_filter_pipeline: None,
            composite_pipeline: None,
            gtao_bind_group_layout: None,
            filter_bind_group_layout: None,
            temporal_bind_group_layout: None,
            composite_bind_group_layout: None,
            ao_texture: None,
            ao_view: None,
            filtered_texture: None,
            filtered_view: None,
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
    pub fn settings(&self) -> &GtaoSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: GtaoSettings) {
        self.settings = settings;
    }

    /// Set quality preset.
    pub fn set_quality(&mut self, quality: GtaoQuality) {
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

    /// Set power (contrast).
    pub fn set_power(&mut self, power: f32) {
        self.settings.power = power.clamp(0.5, 4.0);
    }

    /// Set falloff start.
    pub fn set_falloff_start(&mut self, falloff: f32) {
        self.settings.falloff_start = falloff.clamp(0.0, 1.0);
    }

    /// Set thin occluder compensation.
    pub fn set_thin_occluder_compensation(&mut self, comp: f32) {
        self.settings.thin_occluder_compensation = comp.clamp(0.0, 1.0);
    }

    /// Update projection matrices (call before render).
    pub fn set_matrices(&mut self, projection: [[f32; 4]; 4], inv_projection: [[f32; 4]; 4], view: [[f32; 4]; 4], near: f32, far: f32, fov: f32) {
        self.projection = projection;
        self.inv_projection = inv_projection;
        self.view = view;
        self.near = near;
        self.far = far;
        self.fov = fov;
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.format = format;

        // Create samplers
        self.linear_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("GTAO Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.point_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("GTAO Point Sampler"),
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
        let uniform = GtaoUniform {
            projection: [[0.0; 4]; 4],
            inv_projection: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            resolution: [1.0 / width as f32, 1.0 / height as f32, width as f32, height as f32],
            params: [self.settings.radius, self.settings.falloff_start, self.settings.intensity, self.settings.power],
            params2: [
                self.settings.quality.direction_count() as f32,
                self.settings.quality.step_count() as f32,
                self.settings.thin_occluder_compensation,
                0.0,
            ],
            camera_params: [0.1, 100.0, width as f32 / height as f32, (45.0_f32.to_radians() / 2.0).tan()],
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GTAO Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GTAO Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipelines
        self.create_pipelines(device, format);

        // Create intermediate textures
        self.create_textures(device);
    }

    fn create_bind_group_layouts(&mut self, device: &wgpu::Device) {
        // GTAO pass: depth + uniforms
        self.gtao_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GTAO Bind Group Layout"),
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
                // Depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
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
        }));

        // Spatial filter pass: AO texture + depth + uniforms
        self.filter_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GTAO Filter Bind Group Layout"),
            entries: &[
                // AO texture
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
                // Point sampler for depth
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

        // Temporal filter pass: current AO + history AO + current depth + prev depth + uniforms
        self.temporal_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GTAO Temporal Bind Group Layout"),
            entries: &[
                // Current AO texture (spatially filtered)
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
                // History AO texture
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
                // Current depth texture
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
                // Previous depth texture
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
                // Point sampler for depth
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

        // Composite pass: scene + AO + uniforms
        self.composite_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GTAO Composite Bind Group Layout"),
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
                // AO texture
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
                // Sampler
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
        // GTAO pipeline
        let gtao_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GTAO Pipeline Layout"),
            bind_group_layouts: &[self.gtao_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let gtao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GTAO Shader"),
            source: wgpu::ShaderSource::Wgsl(GTAO_SHADER.into()),
        });

        self.gtao_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GTAO Pipeline"),
            layout: Some(&gtao_layout),
            vertex: wgpu::VertexState {
                module: &gtao_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gtao_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
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

        // Spatial filter pipeline
        let filter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GTAO Filter Pipeline Layout"),
            bind_group_layouts: &[self.filter_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GTAO Spatial Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(SPATIAL_FILTER_SHADER.into()),
        });

        self.spatial_filter_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GTAO Spatial Filter Pipeline"),
            layout: Some(&filter_layout),
            vertex: wgpu::VertexState {
                module: &filter_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &filter_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
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

        // Temporal filter pipeline
        let temporal_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GTAO Temporal Pipeline Layout"),
            bind_group_layouts: &[self.temporal_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GTAO Temporal Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(TEMPORAL_FILTER_SHADER.into()),
        });

        self.temporal_filter_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GTAO Temporal Filter Pipeline"),
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
                    format: wgpu::TextureFormat::R16Float,
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
            label: Some("GTAO Composite Pipeline Layout"),
            bind_group_layouts: &[self.composite_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GTAO Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });

        self.composite_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GTAO Composite Pipeline"),
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
        // GTAO output texture (R16 float for precision)
        let ao_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GTAO Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.ao_view = Some(ao_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.ao_texture = Some(ao_texture);

        // Filtered output texture
        let filtered_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GTAO Filtered Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.filtered_view = Some(filtered_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.filtered_texture = Some(filtered_texture);

        // History texture for temporal accumulation
        let history_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GTAO History Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.history_view = Some(history_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.history_texture = Some(history_texture);

        // Temporal output texture
        let temporal_output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GTAO Temporal Output Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.temporal_output_view = Some(temporal_output_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.temporal_output_texture = Some(temporal_output_texture);

        // Previous frame depth (stored as R16Float for filterability)
        let prev_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GTAO Previous Depth Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.prev_depth_view = Some(prev_depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.prev_depth_texture = Some(prev_depth_texture);
    }

    fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let aspect = self.width as f32 / self.height as f32;
            let uniform = GtaoUniform {
                projection: self.projection,
                inv_projection: self.inv_projection,
                view: self.view,
                resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
                params: [self.settings.radius, self.settings.falloff_start, self.settings.intensity, self.settings.power],
                params2: [
                    self.settings.quality.direction_count() as f32,
                    self.settings.quality.step_count() as f32,
                    self.settings.thin_occluder_compensation,
                    self.frame_index as f32,
                ],
                camera_params: [self.near, self.far, aspect, (self.fov / 2.0).tan()],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }

    /// Render GTAO effect.
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

        let Some(ref linear_sampler) = self.linear_sampler else { return };
        let Some(ref point_sampler) = self.point_sampler else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref ao_view) = self.ao_view else { return };
        let Some(ref filtered_view) = self.filtered_view else { return };
        let Some(ref history_view) = self.history_view else { return };
        let Some(ref temporal_output_view) = self.temporal_output_view else { return };
        let Some(ref prev_depth_view) = self.prev_depth_view else { return };

        self.update_uniforms(queue);

        // Pass 1: Generate GTAO
        {
            let Some(ref gtao_pipeline) = self.gtao_pipeline else { return };
            let Some(ref gtao_layout) = self.gtao_bind_group_layout else { return };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GTAO Bind Group"),
                layout: gtao_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(point_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GTAO Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ao_view,
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

            pass.set_pipeline(gtao_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 2: Spatial filter (edge-aware blur)
        {
            let Some(ref filter_pipeline) = self.spatial_filter_pipeline else { return };
            let Some(ref filter_layout) = self.filter_bind_group_layout else { return };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GTAO Filter Bind Group"),
                layout: filter_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(ao_view),
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
                label: Some("GTAO Filter Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: filtered_view,
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

            pass.set_pipeline(filter_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 3: Temporal filter (accumulate with history)
        {
            let Some(ref temporal_pipeline) = self.temporal_filter_pipeline else { return };
            let Some(ref temporal_layout) = self.temporal_bind_group_layout else { return };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GTAO Temporal Bind Group"),
                layout: temporal_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(filtered_view),
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
                label: Some("GTAO Temporal Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: temporal_output_view,
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

            pass.set_pipeline(temporal_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Copy temporal output to history for next frame
        if let (Some(ref temporal_output), Some(ref history)) = (&self.temporal_output_texture, &self.history_texture) {
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
            let Some(ref composite_pipeline) = self.composite_pipeline else { return };
            let Some(ref composite_layout) = self.composite_bind_group_layout else { return };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GTAO Composite Bind Group"),
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
                label: Some("GTAO Composite Pass"),
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

    /// Copy current depth to previous depth buffer (call after render).
    pub fn copy_depth_to_history(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth_texture: &wgpu::Texture,
    ) {
        if let Some(ref prev_depth) = self.prev_depth_texture {
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: depth_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::ImageCopyTexture {
                    texture: prev_depth,
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
    }

    fn blit(&self, encoder: &mut wgpu::CommandEncoder, input: &wgpu::TextureView, output: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(ref composite_pipeline) = self.composite_pipeline else { return };
        let Some(ref composite_layout) = self.composite_bind_group_layout else { return };
        let Some(ref linear_sampler) = self.linear_sampler else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref filtered_view) = self.filtered_view else { return };

        // Write zero intensity uniform so composite shader just passes through
        let aspect = self.width as f32 / self.height as f32;
        let uniform = GtaoUniform {
            projection: self.projection,
            inv_projection: self.inv_projection,
            view: self.view,
            resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
            params: [self.settings.radius, self.settings.falloff_start, 0.0, self.settings.power], // intensity = 0
            params2: [
                self.settings.quality.direction_count() as f32,
                self.settings.quality.step_count() as f32,
                self.settings.thin_occluder_compensation,
                self.frame_index as f32,
            ],
            camera_params: [self.near, self.far, aspect, (self.fov / 2.0).tan()],
        };
        queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GTAO Blit Bind Group"),
            layout: composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(filtered_view),
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
            label: Some("GTAO Blit Pass"),
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

impl Default for GtaoPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for GtaoPass {
    fn name(&self) -> &str {
        "gtao"
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
        // GTAO requires depth buffer, use render_with_depth instead
    }
}

// GTAO shader - horizon-based ambient occlusion with ground truth integral
const GTAO_SHADER: &str = r#"
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
    resolution: vec4<f32>,  // 1/w, 1/h, w, h
    params: vec4<f32>,      // radius, falloff_start, intensity, power
    params2: vec4<f32>,     // direction_count, step_count, thin_occluder_comp, frame
    camera_params: vec4<f32>, // near, far, aspect, fov_tan
}

const PI: f32 = 3.14159265359;

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var depth_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

// Linearize depth from [0,1] to view-space Z
fn linearize_depth(d: f32) -> f32 {
    let near = params.camera_params.x;
    let far = params.camera_params.y;
    return near * far / (far - d * (far - near));
}

// Reconstruct view-space position from UV and depth
fn view_position_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let clip = vec4<f32>(ndc, depth, 1.0);
    var view = params.inv_projection * clip;
    view = view / view.w;
    return view.xyz;
}

// Get view-space normal from depth derivatives
fn get_view_normal(uv: vec2<f32>, center_pos: vec3<f32>) -> vec3<f32> {
    let texel = params.resolution.xy;

    let depth_l = textureSampleLevel(depth_texture, depth_sampler, uv - vec2<f32>(texel.x, 0.0), 0);
    let depth_r = textureSampleLevel(depth_texture, depth_sampler, uv + vec2<f32>(texel.x, 0.0), 0);
    let depth_t = textureSampleLevel(depth_texture, depth_sampler, uv - vec2<f32>(0.0, texel.y), 0);
    let depth_b = textureSampleLevel(depth_texture, depth_sampler, uv + vec2<f32>(0.0, texel.y), 0);

    let pos_l = view_position_from_depth(uv - vec2<f32>(texel.x, 0.0), depth_l);
    let pos_r = view_position_from_depth(uv + vec2<f32>(texel.x, 0.0), depth_r);
    let pos_t = view_position_from_depth(uv - vec2<f32>(0.0, texel.y), depth_t);
    let pos_b = view_position_from_depth(uv + vec2<f32>(0.0, texel.y), depth_b);

    // Use smallest derivatives for better edge handling
    let dx_l = center_pos - pos_l;
    let dx_r = pos_r - center_pos;
    let dy_t = center_pos - pos_t;
    let dy_b = pos_b - center_pos;

    var dx = dx_r;
    if (abs(dx_l.z) < abs(dx_r.z)) {
        dx = dx_l;
    }

    var dy = dy_b;
    if (abs(dy_t.z) < abs(dy_b.z)) {
        dy = dy_t;
    }

    return normalize(cross(dx, dy));
}

// Interleaved gradient noise
fn interleaved_gradient_noise(pos: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pos, magic.xy)));
}

// GTAO visibility integral (closed-form solution)
// Based on "Practical Realtime Strategies for Accurate Indirect Occlusion"
fn integrate_arc(h1: f32, h2: f32, n_dot_v: f32) -> f32 {
    // Ground truth visibility integral for a given horizon angle pair
    let cos_h1 = cos(h1);
    let cos_h2 = cos(h2);
    let sin_h1 = sin(h1);
    let sin_h2 = sin(h2);

    // Integral of max(0, cos(theta)) over [h1, h2]
    // This is the actual visibility function weighted by cosine
    let a = -cos(2.0 * h1 - n_dot_v) + n_dot_v + 2.0 * h1 * sin(n_dot_v);
    let b = -cos(2.0 * h2 - n_dot_v) + n_dot_v + 2.0 * h2 * sin(n_dot_v);

    return 0.25 * (a + b);
}

// Simplified GTAO integration
fn gtao_fast_acos(x: f32) -> f32 {
    // Fast acos approximation
    let abs_x = abs(x);
    var res = -0.156583 * abs_x + PI * 0.5;
    res = res * sqrt(1.0 - abs_x);
    if (x >= 0.0) {
        return res;
    } else {
        return PI - res;
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let radius = params.params.x;
    let falloff_start = params.params.y;
    let intensity = params.params.z;
    let power = params.params.w;
    let direction_count = i32(params.params2.x);
    let step_count = i32(params.params2.y);
    let thin_occluder = params.params2.z;
    let frame = params.params2.w;

    let texel = params.resolution.xy;

    // Sample center depth
    let center_depth = textureSampleLevel(depth_texture, depth_sampler, in.uv, 0);

    // Skip sky/background
    if (center_depth >= 1.0) {
        return 1.0;
    }

    // Reconstruct view-space position
    let center_pos = view_position_from_depth(in.uv, center_depth);
    let view_z = -center_pos.z;

    // Get view-space normal
    let normal = get_view_normal(in.uv, center_pos);

    // Scale radius by depth for consistent screen-space footprint
    let scaled_radius = radius / view_z;
    let screen_radius = scaled_radius * params.resolution.z; // In pixels

    // Clamp screen radius for performance
    let max_screen_radius = 128.0;
    let actual_radius = min(screen_radius, max_screen_radius) * texel.x;

    // Jitter starting angle based on position and frame
    let noise = interleaved_gradient_noise(in.position.xy + vec2<f32>(frame * 7.0, frame * 11.0));
    let start_angle = noise * PI;

    var ao = 0.0;
    let angle_step = PI / f32(direction_count);

    // For each direction in the slice
    for (var dir = 0; dir < direction_count; dir++) {
        let angle = start_angle + f32(dir) * angle_step;
        let direction = vec2<f32>(cos(angle), sin(angle));

        // Project normal onto the slice plane (2D in screen space)
        // This gives us the projected normal angle in the slice
        let slice_dir_3d = vec3<f32>(direction.x, direction.y, 0.0);
        let proj_normal = normal - slice_dir_3d * dot(normal, slice_dir_3d);
        let proj_normal_length = length(proj_normal);

        // Compute the tangent angle in the slice (bent normal)
        var n_angle = gtao_fast_acos(clamp(proj_normal.z / max(proj_normal_length, 0.001), -1.0, 1.0));
        if (dot(vec3<f32>(direction, 0.0), proj_normal) < 0.0) {
            n_angle = -n_angle;
        }

        // Initialize horizon angles to the tangent plane
        var h1 = -n_angle;
        var h2 = n_angle;

        // March along the direction to find horizons
        for (var step = 1; step <= step_count; step++) {
            let t = (f32(step) + noise * 0.5) / f32(step_count);
            let offset = direction * actual_radius * t;

            // Sample positive direction
            let sample_uv_pos = in.uv + offset;
            if (sample_uv_pos.x >= 0.0 && sample_uv_pos.x <= 1.0 && sample_uv_pos.y >= 0.0 && sample_uv_pos.y <= 1.0) {
                let sample_depth_pos = textureSampleLevel(depth_texture, depth_sampler, sample_uv_pos, 0);
                if (sample_depth_pos < 1.0) {
                    let sample_pos = view_position_from_depth(sample_uv_pos, sample_depth_pos);
                    let delta = sample_pos - center_pos;
                    let delta_len = length(delta);

                    if (delta_len > 0.001) {
                        // Falloff based on distance
                        let falloff = 1.0 - saturate((delta_len - radius * falloff_start) / (radius * (1.0 - falloff_start)));

                        // Compute horizon angle in the slice
                        let sample_h = atan2(delta.z, length(delta.xy));

                        // Update horizon with thin occluder handling
                        let thickness_check = saturate(delta.z / (delta_len * thin_occluder + 0.001));
                        h2 = max(h2, mix(h2, sample_h, falloff * thickness_check));
                    }
                }
            }

            // Sample negative direction
            let sample_uv_neg = in.uv - offset;
            if (sample_uv_neg.x >= 0.0 && sample_uv_neg.x <= 1.0 && sample_uv_neg.y >= 0.0 && sample_uv_neg.y <= 1.0) {
                let sample_depth_neg = textureSampleLevel(depth_texture, depth_sampler, sample_uv_neg, 0);
                if (sample_depth_neg < 1.0) {
                    let sample_pos = view_position_from_depth(sample_uv_neg, sample_depth_neg);
                    let delta = sample_pos - center_pos;
                    let delta_len = length(delta);

                    if (delta_len > 0.001) {
                        let falloff = 1.0 - saturate((delta_len - radius * falloff_start) / (radius * (1.0 - falloff_start)));

                        let sample_h = atan2(delta.z, length(delta.xy));

                        let thickness_check = saturate(delta.z / (delta_len * thin_occluder + 0.001));
                        h1 = min(h1, mix(h1, -sample_h, falloff * thickness_check));
                    }
                }
            }
        }

        // Clamp horizons to hemisphere
        h1 = max(h1, -PI * 0.5);
        h2 = min(h2, PI * 0.5);

        // Integrate visibility over the slice using the ground truth formula
        // Simplified: use the cosine-weighted hemisphere integral
        let n = clamp(proj_normal.z, 0.0, 1.0);
        let vis = (1.0 - cos(h2 - n_angle)) * 0.5 + (1.0 - cos(-h1 - n_angle)) * 0.5;

        ao += vis;
    }

    ao /= f32(direction_count);

    // Apply intensity and power
    ao = saturate(1.0 - ao * intensity);
    ao = pow(ao, power);

    return ao;
}
"#;

// Spatial filter shader - edge-aware blur using depth
const SPATIAL_FILTER_SHADER: &str = r#"
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
    resolution: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    camera_params: vec4<f32>,
}

@group(0) @binding(0) var ao_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var ao_sampler: sampler;
@group(0) @binding(3) var depth_sampler: sampler;
@group(0) @binding(4) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fn linearize_depth(d: f32) -> f32 {
    let near = params.camera_params.x;
    let far = params.camera_params.y;
    return near * far / (far - d * (far - near));
}

// Reconstruct view-space normal from depth
fn reconstruct_normal_filter(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    let center = textureSampleLevel(depth_texture, depth_sampler, uv, 0);
    let left = textureSampleLevel(depth_texture, depth_sampler, uv - vec2<f32>(texel.x, 0.0), 0);
    let right = textureSampleLevel(depth_texture, depth_sampler, uv + vec2<f32>(texel.x, 0.0), 0);
    let up = textureSampleLevel(depth_texture, depth_sampler, uv - vec2<f32>(0.0, texel.y), 0);
    let down = textureSampleLevel(depth_texture, depth_sampler, uv + vec2<f32>(0.0, texel.y), 0);

    let dx = (right - left) * 0.5;
    let dy = (down - up) * 0.5;

    return normalize(vec3<f32>(-dx, -dy, 1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let texel = params.resolution.xy;
    let center_ao = textureSampleLevel(ao_texture, ao_sampler, in.uv, 0.0).r;
    let center_depth = textureSampleLevel(depth_texture, depth_sampler, in.uv, 0);

    // Skip sky
    if (center_depth >= 1.0) {
        return 1.0;
    }

    let center_linear_depth = linearize_depth(center_depth);
    let center_normal = reconstruct_normal_filter(in.uv, texel);

    // Cross bilateral filter with normal-aware edge preservation
    // Larger kernel for smoother results
    var total_ao = 0.0;
    var total_weight = 0.0;

    let blur_radius = 5;
    let sigma_spatial = 4.0;
    let sigma_depth = 0.12;
    let sigma_normal = 0.5;

    for (var y = -blur_radius; y <= blur_radius; y++) {
        for (var x = -blur_radius; x <= blur_radius; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let sample_uv = in.uv + offset;

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }

            let sample_ao = textureSampleLevel(ao_texture, ao_sampler, sample_uv, 0.0).r;
            let sample_depth = textureSampleLevel(depth_texture, depth_sampler, sample_uv, 0);
            let sample_linear_depth = linearize_depth(sample_depth);

            // Spatial weight (Gaussian)
            let dist2 = f32(x * x + y * y);
            let spatial_weight = exp(-dist2 / (2.0 * sigma_spatial * sigma_spatial));

            // Depth weight (edge-preserving)
            let depth_diff = abs(center_linear_depth - sample_linear_depth) / center_linear_depth;
            let depth_weight = exp(-depth_diff * depth_diff / (2.0 * sigma_depth * sigma_depth));

            // Normal weight (prevent bleeding across surfaces)
            let sample_normal = reconstruct_normal_filter(sample_uv, texel);
            let normal_diff = 1.0 - max(dot(center_normal, sample_normal), 0.0);
            let normal_weight = exp(-normal_diff * normal_diff / (2.0 * sigma_normal * sigma_normal));

            let weight = spatial_weight * depth_weight * normal_weight;
            total_ao += sample_ao * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0) {
        return total_ao / total_weight;
    }
    return center_ao;
}
"#;

// Temporal filter shader - blend current with history based on depth similarity
const TEMPORAL_FILTER_SHADER: &str = r#"
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
    resolution: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    camera_params: vec4<f32>,
}

@group(0) @binding(0) var current_ao: texture_2d<f32>;
@group(0) @binding(1) var history_ao: texture_2d<f32>;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(3) var prev_depth_texture: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var depth_sampler: sampler;
@group(0) @binding(6) var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fn linearize_depth(d: f32) -> f32 {
    let near = params.camera_params.x;
    let far = params.camera_params.y;
    return near * far / (far - d * (far - near));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let current = textureSampleLevel(current_ao, linear_sampler, in.uv, 0.0).r;
    let history = textureSampleLevel(history_ao, linear_sampler, in.uv, 0.0).r;

    let current_depth = textureSampleLevel(depth_texture, depth_sampler, in.uv, 0);
    let prev_depth = textureSampleLevel(prev_depth_texture, linear_sampler, in.uv, 0.0).r;

    // Skip sky pixels
    if (current_depth >= 1.0) {
        return 1.0;
    }

    // Linearize depths for comparison
    let current_linear = linearize_depth(current_depth);
    let prev_linear = linearize_depth(prev_depth);

    // Compute depth-based disocclusion weight
    // If depth changed significantly, the pixel was likely occluded/disoccluded
    let depth_diff = abs(current_linear - prev_linear) / max(current_linear, 0.001);
    let depth_threshold = 0.05; // 5% depth difference threshold
    let depth_weight = 1.0 - saturate(depth_diff / depth_threshold);

    // Temporal blend factor (higher = more history, smoother but more ghosting)
    // Use ~90% history when depth is stable, 0% when disoccluded
    let base_blend = 0.9;
    let temporal_blend = base_blend * depth_weight;

    // Check if history is valid (not first frame - history starts at 1.0)
    let history_valid = select(0.0, 1.0, history < 0.999 || prev_depth > 0.001);
    let effective_blend = temporal_blend * history_valid;

    // Clamp history to neighborhood to prevent ghosting
    let texel = params.resolution.xy;
    var ao_min = current;
    var ao_max = current;

    // Sample 3x3 neighborhood for clamping
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) { continue; }
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let neighbor = textureSampleLevel(current_ao, linear_sampler, in.uv + offset, 0.0).r;
            ao_min = min(ao_min, neighbor);
            ao_max = max(ao_max, neighbor);
        }
    }

    // Expand bounds slightly to allow some temporal smoothing
    let margin = 0.1;
    ao_min = ao_min - margin;
    ao_max = ao_max + margin;

    // Clamp history to neighborhood bounds
    let clamped_history = clamp(history, ao_min, ao_max);

    // Blend current with clamped history
    return mix(current, clamped_history, effective_blend);
}
"#;

// Composite shader - multiply scene by AO
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
    resolution: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    camera_params: vec4<f32>,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var ao_texture: texture_2d<f32>;
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
    let intensity = params.params.z;

    // Skip AO when intensity is 0 (used for passthrough blit)
    if (intensity <= 0.0) {
        return scene;
    }

    let ao = textureSampleLevel(ao_texture, tex_sampler, in.uv, 0.0).r;

    // Apply AO to scene - AO affects ambient/indirect light only
    // In a full implementation, we'd want to separate direct from indirect
    // For now, we apply a multi-bounce approximation
    let multi_bounce_ao = ao + ao * (1.0 - ao) * 0.5;

    return vec4<f32>(scene.rgb * multi_bounce_ao, scene.a);
}
"#;
