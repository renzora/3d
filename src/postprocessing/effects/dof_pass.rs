//! Depth of Field post-processing effect with circular bokeh.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// Depth of Field quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DofQuality {
    /// Low quality - 22 samples (3 rings).
    Low,
    /// Medium quality - 43 samples (5 rings).
    #[default]
    Medium,
    /// High quality - 71 samples (7 rings).
    High,
    /// Ultra quality - 106 samples (9 rings).
    Ultra,
}

impl DofQuality {
    /// Get the number of sample rings for this quality level.
    pub fn ring_count(&self) -> u32 {
        match self {
            DofQuality::Low => 3,
            DofQuality::Medium => 5,
            DofQuality::High => 7,
            DofQuality::Ultra => 9,
        }
    }
}

/// Depth of Field settings.
#[derive(Debug, Clone)]
pub struct DofSettings {
    /// Distance to the focal plane (in world units).
    pub focal_distance: f32,
    /// Range around focal distance that stays sharp.
    pub focal_range: f32,
    /// Maximum blur strength (affects bokeh size).
    pub blur_strength: f32,
    /// Enable near blur (objects closer than focal plane).
    pub near_blur: bool,
    /// Enable far blur (objects further than focal plane).
    pub far_blur: bool,
    /// Bokeh brightness boost (makes highlights pop).
    pub highlight_boost: f32,
    /// Highlight threshold (luminance above this gets boosted).
    pub highlight_threshold: f32,
}

impl Default for DofSettings {
    fn default() -> Self {
        Self {
            focal_distance: 5.0,
            focal_range: 2.0,
            blur_strength: 1.0,
            near_blur: true,
            far_blur: true,
            highlight_boost: 0.5,
            highlight_threshold: 0.5,
        }
    }
}

/// DoF uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DofUniform {
    /// focal_distance, focal_range, blur_strength, ring_count
    params: [f32; 4],
    /// near_blur, far_blur, near, far
    params2: [f32; 4],
    /// 1/width, 1/height, width, height
    resolution: [f32; 4],
    /// highlight_boost, highlight_threshold, aspect_ratio, max_coc_pixels
    params3: [f32; 4],
}

/// Depth of Field post-processing pass with circular bokeh.
pub struct DofPass {
    enabled: bool,
    settings: DofSettings,
    quality: DofQuality,
    width: u32,
    height: u32,
    near: f32,
    far: f32,
    output_format: wgpu::TextureFormat,
    // GPU resources
    coc_pipeline: Option<wgpu::RenderPipeline>,
    bokeh_pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    bokeh_bind_group_layout: Option<wgpu::BindGroupLayout>,
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    sampler: Option<wgpu::Sampler>,
    point_sampler: Option<wgpu::Sampler>,
    // CoC texture (stores CoC in alpha, color in RGB)
    coc_texture: Option<wgpu::Texture>,
    coc_view: Option<wgpu::TextureView>,
}

impl DofPass {
    /// Create a new DoF pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: DofSettings::default(),
            quality: DofQuality::default(),
            width: 1,
            height: 1,
            near: 0.1,
            far: 100.0,
            output_format: wgpu::TextureFormat::Bgra8Unorm,
            coc_pipeline: None,
            bokeh_pipeline: None,
            bind_group_layout: None,
            bokeh_bind_group_layout: None,
            uniform_buffer: None,
            quad_buffer: None,
            sampler: None,
            point_sampler: None,
            coc_texture: None,
            coc_view: None,
        }
    }

    /// Get settings.
    pub fn settings(&self) -> &DofSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: DofSettings) {
        self.settings = settings;
    }

    /// Set focal distance.
    pub fn set_focal_distance(&mut self, distance: f32) {
        self.settings.focal_distance = distance.max(0.1);
    }

    /// Set focal range.
    pub fn set_focal_range(&mut self, range: f32) {
        self.settings.focal_range = range.max(0.1);
    }

    /// Set blur strength.
    pub fn set_blur_strength(&mut self, strength: f32) {
        self.settings.blur_strength = strength.clamp(0.0, 5.0);
    }

    /// Set quality preset.
    pub fn set_quality(&mut self, quality: DofQuality) {
        self.quality = quality;
    }

    /// Set camera near/far planes for depth linearization.
    pub fn set_camera_planes(&mut self, near: f32, far: f32) {
        self.near = near;
        self.far = far;
    }

    /// Set bokeh highlight boost (makes bright spots pop more).
    pub fn set_highlight_boost(&mut self, boost: f32) {
        self.settings.highlight_boost = boost.clamp(0.0, 2.0);
    }

    /// Set bokeh highlight threshold (luminance above this gets boosted).
    pub fn set_highlight_threshold(&mut self, threshold: f32) {
        self.settings.highlight_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Initialize GPU resources.
    /// The format parameter is the output format (typically surface format like Bgra8Unorm).
    pub fn init(&mut self, device: &wgpu::Device, output_format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.output_format = output_format;

        // Create samplers
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DoF Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.point_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DoF Point Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        // Create bind group layout for CoC pass
        self.bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoF CoC Bind Group Layout"),
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
                // Point sampler (for depth)
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

        // Create bind group layout for bokeh pass (uses CoC texture instead of depth)
        self.bokeh_bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoF Bokeh Bind Group Layout"),
            entries: &[
                // CoC texture (RGB = color, A = CoC)
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
                // Linear sampler
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
        }));

        // Create uniform buffer
        let uniform = self.create_uniform();
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DoF Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DoF Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create CoC texture (RGBA16F to store color + CoC)
        self.create_textures(device);

        // Create pipelines (bokeh outputs to output_format)
        self.create_pipelines(device);
    }

    fn create_uniform(&self) -> DofUniform {
        let aspect = self.width as f32 / self.height as f32;
        let max_coc_pixels = self.settings.blur_strength * 40.0; // Max CoC in pixels

        DofUniform {
            params: [
                self.settings.focal_distance,
                self.settings.focal_range,
                self.settings.blur_strength,
                self.quality.ring_count() as f32,
            ],
            params2: [
                if self.settings.near_blur { 1.0 } else { 0.0 },
                if self.settings.far_blur { 1.0 } else { 0.0 },
                self.near,
                self.far,
            ],
            resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
            params3: [
                self.settings.highlight_boost,
                self.settings.highlight_threshold,
                aspect,
                max_coc_pixels,
            ],
        }
    }

    fn create_textures(&mut self, device: &wgpu::Device) {
        // Use RGBA16Float for CoC texture to store HDR color + CoC
        let coc_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DoF CoC Texture"),
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
        self.coc_view = Some(coc_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.coc_texture = Some(coc_texture);
    }

    fn create_pipelines(&mut self, device: &wgpu::Device) {
        // CoC calculation pass
        let coc_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF CoC Pipeline Layout"),
            bind_group_layouts: &[self.bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let coc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF CoC Shader"),
            source: wgpu::ShaderSource::Wgsl(DOF_COC_SHADER.into()),
        });

        self.coc_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF CoC Pipeline"),
            layout: Some(&coc_layout),
            vertex: wgpu::VertexState {
                module: &coc_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &coc_shader,
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

        // Bokeh blur pass
        let bokeh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF Bokeh Pipeline Layout"),
            bind_group_layouts: &[self.bokeh_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let bokeh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Bokeh Shader"),
            source: wgpu::ShaderSource::Wgsl(DOF_BOKEH_SHADER.into()),
        });

        self.bokeh_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Bokeh Pipeline"),
            layout: Some(&bokeh_layout),
            vertex: wgpu::VertexState {
                module: &bokeh_shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bokeh_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.output_format,
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

    fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let uniform = self.create_uniform();
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }

    /// Render DoF effect with depth texture.
    pub fn render_with_depth(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if !self.enabled {
            return;
        }

        let Some(ref coc_pipeline) = self.coc_pipeline else { return };
        let Some(ref bokeh_pipeline) = self.bokeh_pipeline else { return };
        let Some(ref bind_group_layout) = self.bind_group_layout else { return };
        let Some(ref bokeh_bind_group_layout) = self.bokeh_bind_group_layout else { return };
        let Some(ref sampler) = self.sampler else { return };
        let Some(ref point_sampler) = self.point_sampler else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };
        let Some(ref coc_view) = self.coc_view else { return };

        self.update_uniforms(queue);

        // Pass 1: Calculate CoC and store color + CoC
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF CoC Bind Group"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
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
                label: Some("DoF CoC Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: coc_view,
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

            pass.set_pipeline(coc_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 2: Bokeh blur using disc sampling
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF Bokeh Bind Group"),
                layout: bokeh_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(coc_view),
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
                label: Some("DoF Bokeh Pass"),
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

            pass.set_pipeline(bokeh_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
    }

    /// Update the output format and recreate pipelines.
    /// Call this when switching between SDR and HDR output modes.
    pub fn set_output_format(&mut self, format: wgpu::TextureFormat, device: &wgpu::Device) {
        if self.output_format == format {
            return;
        }
        self.output_format = format;

        // Recreate pipelines with new format
        if self.bokeh_pipeline.is_some() {
            self.create_pipelines(device);
        }
    }
}

impl Default for DofPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for DofPass {
    fn name(&self) -> &str {
        "dof"
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

        // Recreate textures
        if self.coc_texture.is_some() {
            self.create_textures(device);
        }
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

// CoC calculation shader - stores color + CoC
const DOF_COC_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    // focal_distance, focal_range, blur_strength, ring_count
    params: vec4<f32>,
    // near_blur, far_blur, near, far
    params2: vec4<f32>,
    // 1/width, 1/height, width, height
    resolution: vec4<f32>,
    // highlight_boost, highlight_threshold, aspect_ratio, max_coc_pixels
    params3: vec4<f32>,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
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

// Linearize depth from [0,1] to view-space distance
fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let focal_dist = params.params.x;
    let focal_range = params.params.y;
    let near_blur = params.params2.x;
    let far_blur = params.params2.y;
    let near = params.params2.z;
    let far = params.params2.w;

    // Sample color and depth
    let color = textureSampleLevel(scene_texture, linear_sampler, in.uv, 0.0).rgb;
    let depth_raw = textureSampleLevel(depth_texture, point_sampler, in.uv, 0);
    let depth = linearize_depth(depth_raw, near, far);

    // Calculate signed CoC (negative = near, positive = far)
    let diff = depth - focal_dist;
    var coc: f32;

    if (diff < 0.0) {
        // Near field
        coc = -diff / focal_range * near_blur;
        coc = -clamp(coc, 0.0, 1.0); // Negative for near
    } else {
        // Far field
        coc = diff / focal_range * far_blur;
        coc = clamp(coc, 0.0, 1.0); // Positive for far
    }

    // Store color in RGB, signed CoC in alpha
    return vec4<f32>(color, coc);
}
"#;

// Bokeh blur shader - disc sampling with brightness weighting
const DOF_BOKEH_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    // focal_distance, focal_range, blur_strength, ring_count
    params: vec4<f32>,
    // near_blur, far_blur, near, far
    params2: vec4<f32>,
    // 1/width, 1/height, width, height
    resolution: vec4<f32>,
    // highlight_boost, highlight_threshold, aspect_ratio, max_coc_pixels
    params3: vec4<f32>,
}

@group(0) @binding(0) var coc_texture: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const GOLDEN_ANGLE: f32 = 2.39996323; // PI * (3.0 - sqrt(5.0))

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

// Calculate luminance for brightness weighting
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let blur_strength = params.params.z;
    let ring_count = i32(params.params.w);
    let texel_size = params.resolution.xy;
    let highlight_boost = params.params3.x;
    let highlight_threshold = params.params3.y;
    let aspect = params.params3.z;
    let max_coc_pixels = params.params3.w;

    // Get center sample
    let center = textureSampleLevel(coc_texture, linear_sampler, in.uv, 0.0);
    let center_color = center.rgb;
    let center_coc = center.a; // Signed CoC
    let abs_coc = abs(center_coc);

    // No blur needed if in focus
    if (abs_coc < 0.01 || blur_strength <= 0.0) {
        return vec4<f32>(center_color, 1.0);
    }

    // Calculate blur radius in UV space
    let blur_radius_pixels = abs_coc * max_coc_pixels;
    let blur_radius = vec2<f32>(
        blur_radius_pixels * texel_size.x,
        blur_radius_pixels * texel_size.y * aspect
    );

    // Accumulate samples using disc pattern
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    // Add center sample
    let center_lum = luminance(center_color);
    var center_weight = 1.0;
    if (center_lum > highlight_threshold) {
        center_weight += (center_lum - highlight_threshold) * highlight_boost;
    }
    color_sum += center_color * center_weight;
    weight_sum += center_weight;

    // Sample in concentric rings using golden angle for even distribution
    var sample_index = 0;
    for (var ring = 1; ring <= ring_count; ring++) {
        let ring_radius = f32(ring) / f32(ring_count);
        let samples_in_ring = ring * 8; // More samples in outer rings

        for (var s = 0; s < samples_in_ring; s++) {
            // Golden angle spiral for uniform disc distribution
            let angle = f32(sample_index) * GOLDEN_ANGLE;
            let r = ring_radius;

            let offset = vec2<f32>(
                cos(angle) * r * blur_radius.x,
                sin(angle) * r * blur_radius.y
            );

            let sample_uv = in.uv + offset;
            let sample_data = textureSampleLevel(coc_texture, linear_sampler, sample_uv, 0.0);
            let sample_color = sample_data.rgb;
            let sample_coc = sample_data.a;

            // Weight by sample's CoC - larger CoC means more contribution
            // Also prevents sharp foreground from bleeding into background
            let sample_abs_coc = abs(sample_coc);
            var weight = 1.0;

            // Foreground/background handling:
            // - If center is background (positive CoC) and sample is foreground (negative CoC),
            //   only include if sample's CoC is large enough to reach here
            // - If center is foreground, include all samples
            if (center_coc > 0.0 && sample_coc < 0.0) {
                // Background pixel looking at foreground sample
                // Only include if foreground blur reaches this pixel
                let reach = sample_abs_coc / abs_coc;
                weight *= smoothstep(0.0, 1.0, reach);
            }

            // Brightness weighting for bokeh highlights
            let lum = luminance(sample_color);
            if (lum > highlight_threshold) {
                weight *= 1.0 + (lum - highlight_threshold) * highlight_boost;
            }

            // CoC-based weighting - larger blur = more contribution
            weight *= max(sample_abs_coc, abs_coc * 0.5);

            color_sum += sample_color * weight;
            weight_sum += weight;
            sample_index++;
        }
    }

    // Normalize
    var final_color = color_sum / max(weight_sum, 0.001);

    return vec4<f32>(final_color, 1.0);
}
"#;
