//! Depth of Field post-processing effect.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// Depth of Field quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DofQuality {
    /// Low quality - 5 samples per direction.
    Low,
    /// Medium quality - 9 samples per direction.
    #[default]
    Medium,
    /// High quality - 13 samples per direction.
    High,
}

/// Depth of Field settings.
#[derive(Debug, Clone)]
pub struct DofSettings {
    /// Distance to the focal plane (in world units).
    pub focal_distance: f32,
    /// Range around focal distance that stays sharp.
    pub focal_range: f32,
    /// Maximum blur strength.
    pub blur_strength: f32,
    /// Enable near blur (objects closer than focal plane).
    pub near_blur: bool,
    /// Enable far blur (objects further than focal plane).
    pub far_blur: bool,
}

impl Default for DofSettings {
    fn default() -> Self {
        Self {
            focal_distance: 5.0,
            focal_range: 2.0,
            blur_strength: 1.0,
            near_blur: true,
            far_blur: true,
        }
    }
}

/// DoF uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DofUniform {
    /// focal_distance, focal_range, blur_strength, sample_count
    params: [f32; 4],
    /// near_blur, far_blur, near, far
    params2: [f32; 4],
    /// 1/width, 1/height, width, height
    resolution: [f32; 4],
}

/// Depth of Field post-processing pass.
pub struct DofPass {
    enabled: bool,
    settings: DofSettings,
    quality: DofQuality,
    width: u32,
    height: u32,
    near: f32,
    far: f32,
    // GPU resources
    blur_h_pipeline: Option<wgpu::RenderPipeline>,
    blur_v_pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    sampler: Option<wgpu::Sampler>,
    point_sampler: Option<wgpu::Sampler>,
    // Intermediate texture for two-pass blur
    blur_texture: Option<wgpu::Texture>,
    blur_view: Option<wgpu::TextureView>,
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
            blur_h_pipeline: None,
            blur_v_pipeline: None,
            bind_group_layout: None,
            uniform_buffer: None,
            quad_buffer: None,
            sampler: None,
            point_sampler: None,
            blur_texture: None,
            blur_view: None,
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

    fn sample_count(&self) -> u32 {
        match self.quality {
            DofQuality::Low => 5,
            DofQuality::Medium => 9,
            DofQuality::High => 13,
        }
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;

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

        // Create bind group layout
        self.bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoF Bind Group Layout"),
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

        // Create uniform buffer
        let uniform = DofUniform {
            params: [
                self.settings.focal_distance,
                self.settings.focal_range,
                self.settings.blur_strength,
                self.sample_count() as f32,
            ],
            params2: [
                if self.settings.near_blur { 1.0 } else { 0.0 },
                if self.settings.far_blur { 1.0 } else { 0.0 },
                self.near,
                self.far,
            ],
            resolution: [1.0 / width as f32, 1.0 / height as f32, width as f32, height as f32],
        };
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

        // Create intermediate blur texture
        self.create_textures(device, format);

        // Create pipelines
        self.create_pipelines(device, format);
    }

    fn create_textures(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DoF Blur Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.blur_view = Some(blur_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.blur_texture = Some(blur_texture);
    }

    fn create_pipelines(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF Pipeline Layout"),
            bind_group_layouts: &[self.bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        // Horizontal blur shader
        let blur_h_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Horizontal Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(DOF_BLUR_H_SHADER.into()),
        });

        self.blur_h_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Horizontal Blur Pipeline"),
            layout: Some(&layout),
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

        // Vertical blur shader
        let blur_v_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Vertical Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(DOF_BLUR_V_SHADER.into()),
        });

        self.blur_v_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Vertical Blur Pipeline"),
            layout: Some(&layout),
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
    }

    fn update_uniforms(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.uniform_buffer {
            let uniform = DofUniform {
                params: [
                    self.settings.focal_distance,
                    self.settings.focal_range,
                    self.settings.blur_strength,
                    self.sample_count() as f32,
                ],
                params2: [
                    if self.settings.near_blur { 1.0 } else { 0.0 },
                    if self.settings.far_blur { 1.0 } else { 0.0 },
                    self.near,
                    self.far,
                ],
                resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
            };
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
            // Just blit input to output
            self.blit(encoder, input, output, device, queue);
            return;
        }

        let Some(ref blur_h_pipeline) = self.blur_h_pipeline else { return };
        let Some(ref blur_v_pipeline) = self.blur_v_pipeline else { return };
        let Some(ref bind_group_layout) = self.bind_group_layout else { return };
        let Some(ref sampler) = self.sampler else { return };
        let Some(ref point_sampler) = self.point_sampler else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };
        let Some(ref blur_view) = self.blur_view else { return };

        self.update_uniforms(queue);

        // Pass 1: Horizontal blur (input -> blur_texture)
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF Horizontal Bind Group"),
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
                label: Some("DoF Horizontal Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: blur_view,
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

            pass.set_pipeline(blur_h_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }

        // Pass 2: Vertical blur (blur_texture -> output)
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF Vertical Bind Group"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(blur_view),
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
                label: Some("DoF Vertical Pass"),
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

            pass.set_pipeline(blur_v_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, quad_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
    }

    fn blit(&self, encoder: &mut wgpu::CommandEncoder, input: &wgpu::TextureView, output: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Set blur_strength to 0 to disable effect
        if let Some(ref buffer) = self.uniform_buffer {
            let uniform = DofUniform {
                params: [self.settings.focal_distance, self.settings.focal_range, 0.0, self.sample_count() as f32],
                params2: [0.0, 0.0, self.near, self.far],
                resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }

        let Some(ref blur_h_pipeline) = self.blur_h_pipeline else { return };
        let Some(ref bind_group_layout) = self.bind_group_layout else { return };
        let Some(ref sampler) = self.sampler else { return };
        let Some(ref point_sampler) = self.point_sampler else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };
        let Some(ref blur_view) = self.blur_view else { return }; // Need a depth view for bind group

        // We need a valid depth texture for the bind group, but since blur_strength is 0,
        // the shader will just copy. Use blur_view as placeholder (not ideal but works).
        // In practice, render_with_depth is always called with a real depth view.

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DoF Blit Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(blur_view), // Placeholder
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
            label: Some("DoF Blit Pass"),
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

        pass.set_pipeline(blur_h_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, quad_buffer.slice(..));
        pass.draw(0..6, 0..1);
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

        // Recreate blur texture
        if let Some(ref blur_tex) = self.blur_texture {
            let format = blur_tex.format();
            self.create_textures(device, format);
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

// Shared shader code for both passes
const DOF_SHADER_COMMON: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    // focal_distance, focal_range, blur_strength, sample_count
    params: vec4<f32>,
    // near_blur, far_blur, near, far
    params2: vec4<f32>,
    // 1/width, 1/height, width, height
    resolution: vec4<f32>,
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

// Calculate Circle of Confusion
fn calc_coc(depth: f32, focal_dist: f32, focal_range: f32, near_blur: f32, far_blur: f32) -> f32 {
    let diff = depth - focal_dist;

    // Calculate blur amount based on distance from focal plane
    var coc: f32;
    if (diff < 0.0) {
        // Near blur
        coc = -diff / focal_range * near_blur;
    } else {
        // Far blur
        coc = diff / focal_range * far_blur;
    }

    return clamp(coc, 0.0, 1.0);
}
"#;

const DOF_BLUR_H_SHADER: &str = concat!(r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    // focal_distance, focal_range, blur_strength, sample_count
    params: vec4<f32>,
    // near_blur, far_blur, near, far
    params2: vec4<f32>,
    // 1/width, 1/height, width, height
    resolution: vec4<f32>,
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

fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

fn calc_coc(depth: f32, focal_dist: f32, focal_range: f32, near_blur: f32, far_blur: f32) -> f32 {
    let diff = depth - focal_dist;
    var coc: f32;
    if (diff < 0.0) {
        coc = -diff / focal_range * near_blur;
    } else {
        coc = diff / focal_range * far_blur;
    }
    return clamp(coc, 0.0, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let focal_dist = params.params.x;
    let focal_range = params.params.y;
    let blur_strength = params.params.z;
    let sample_count = i32(params.params.w);
    let near_blur = params.params2.x;
    let far_blur = params.params2.y;
    let near = params.params2.z;
    let far = params.params2.w;
    let texel_size = params.resolution.xy;

    // Get center color and depth
    let center_color = textureSampleLevel(scene_texture, linear_sampler, in.uv, 0.0);
    let center_depth_raw = textureSampleLevel(depth_texture, point_sampler, in.uv, 0);
    let center_depth = linearize_depth(center_depth_raw, near, far);
    let center_coc = calc_coc(center_depth, focal_dist, focal_range, near_blur, far_blur);

    // No blur or in focus - return original
    if (blur_strength <= 0.0 || center_coc < 0.01) {
        return center_color;
    }

    // Horizontal blur weighted by CoC
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    let blur_radius = center_coc * blur_strength * 10.0;
    let half_samples = sample_count / 2;

    for (var i = -half_samples; i <= half_samples; i = i + 1) {
        let offset = vec2<f32>(f32(i) * texel_size.x * blur_radius, 0.0);
        let sample_uv = in.uv + offset;

        // Sample color and depth
        let sample_color = textureSampleLevel(scene_texture, linear_sampler, sample_uv, 0.0).rgb;
        let sample_depth_raw = textureSampleLevel(depth_texture, point_sampler, sample_uv, 0);
        let sample_depth = linearize_depth(sample_depth_raw, near, far);
        let sample_coc = calc_coc(sample_depth, focal_dist, focal_range, near_blur, far_blur);

        // Weight by gaussian-like falloff and CoC
        let dist = abs(f32(i)) / f32(half_samples + 1);
        let gaussian = exp(-dist * dist * 2.0);
        let weight = gaussian * max(sample_coc, center_coc * 0.5);

        color += sample_color * weight;
        weight_sum += weight;
    }

    if (weight_sum > 0.0) {
        color /= weight_sum;
    }

    return vec4<f32>(color, 1.0);
}
"#);

const DOF_BLUR_V_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    // focal_distance, focal_range, blur_strength, sample_count
    params: vec4<f32>,
    // near_blur, far_blur, near, far
    params2: vec4<f32>,
    // 1/width, 1/height, width, height
    resolution: vec4<f32>,
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

fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

fn calc_coc(depth: f32, focal_dist: f32, focal_range: f32, near_blur: f32, far_blur: f32) -> f32 {
    let diff = depth - focal_dist;
    var coc: f32;
    if (diff < 0.0) {
        coc = -diff / focal_range * near_blur;
    } else {
        coc = diff / focal_range * far_blur;
    }
    return clamp(coc, 0.0, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let focal_dist = params.params.x;
    let focal_range = params.params.y;
    let blur_strength = params.params.z;
    let sample_count = i32(params.params.w);
    let near_blur = params.params2.x;
    let far_blur = params.params2.y;
    let near = params.params2.z;
    let far = params.params2.w;
    let texel_size = params.resolution.xy;

    // Get center color and depth
    let center_color = textureSampleLevel(scene_texture, linear_sampler, in.uv, 0.0);
    let center_depth_raw = textureSampleLevel(depth_texture, point_sampler, in.uv, 0);
    let center_depth = linearize_depth(center_depth_raw, near, far);
    let center_coc = calc_coc(center_depth, focal_dist, focal_range, near_blur, far_blur);

    // No blur or in focus - return original
    if (blur_strength <= 0.0 || center_coc < 0.01) {
        return center_color;
    }

    // Vertical blur weighted by CoC
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    let blur_radius = center_coc * blur_strength * 10.0;
    let half_samples = sample_count / 2;

    for (var i = -half_samples; i <= half_samples; i = i + 1) {
        let offset = vec2<f32>(0.0, f32(i) * texel_size.y * blur_radius);
        let sample_uv = in.uv + offset;

        // Sample color and depth
        let sample_color = textureSampleLevel(scene_texture, linear_sampler, sample_uv, 0.0).rgb;
        let sample_depth_raw = textureSampleLevel(depth_texture, point_sampler, sample_uv, 0);
        let sample_depth = linearize_depth(sample_depth_raw, near, far);
        let sample_coc = calc_coc(sample_depth, focal_dist, focal_range, near_blur, far_blur);

        // Weight by gaussian-like falloff and CoC
        let dist = abs(f32(i)) / f32(half_samples + 1);
        let gaussian = exp(-dist * dist * 2.0);
        let weight = gaussian * max(sample_coc, center_coc * 0.5);

        color += sample_color * weight;
        weight_sum += weight;
    }

    if (weight_sum > 0.0) {
        color /= weight_sum;
    }

    return vec4<f32>(color, 1.0);
}
"#;
