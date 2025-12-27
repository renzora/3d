//! Screen-Space Reflections post-processing effect.

use crate::postprocessing::pass::{Pass, FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// SSR quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SsrQuality {
    /// Low quality - 16 steps.
    Low,
    /// Medium quality - 32 steps.
    #[default]
    Medium,
    /// High quality - 64 steps.
    High,
    /// Ultra quality - 128 steps.
    Ultra,
}

/// SSR settings.
#[derive(Debug, Clone)]
pub struct SsrSettings {
    /// Maximum ray distance in view space.
    pub max_distance: f32,
    /// Depth thickness for hit detection.
    pub thickness: f32,
    /// Reflection intensity (0.0 - 1.0).
    pub intensity: f32,
    /// Roughness threshold - surfaces rougher than this won't reflect.
    pub roughness_threshold: f32,
}

impl Default for SsrSettings {
    fn default() -> Self {
        Self {
            max_distance: 10.0,
            thickness: 0.5,
            intensity: 0.5,
            roughness_threshold: 0.5,
        }
    }
}

/// SSR uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrUniform {
    /// Projection matrix.
    projection: [[f32; 4]; 4],
    /// Inverse projection matrix.
    inv_projection: [[f32; 4]; 4],
    /// View matrix.
    view: [[f32; 4]; 4],
    /// 1/width, 1/height, width, height
    resolution: [f32; 4],
    /// max_distance, thickness, intensity, step_count
    params: [f32; 4],
    /// near, far, roughness_threshold, 0
    params2: [f32; 4],
}

/// Screen-Space Reflections post-processing pass.
pub struct SsrPass {
    enabled: bool,
    settings: SsrSettings,
    quality: SsrQuality,
    width: u32,
    height: u32,
    near: f32,
    far: f32,
    projection: [[f32; 4]; 4],
    inv_projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    // GPU resources
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    linear_sampler: Option<wgpu::Sampler>,
    point_sampler: Option<wgpu::Sampler>,
}

impl SsrPass {
    /// Create a new SSR pass.
    pub fn new() -> Self {
        Self {
            enabled: true,
            settings: SsrSettings::default(),
            quality: SsrQuality::default(),
            width: 1,
            height: 1,
            near: 0.1,
            far: 100.0,
            projection: [[0.0; 4]; 4],
            inv_projection: [[0.0; 4]; 4],
            view: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            pipeline: None,
            bind_group_layout: None,
            uniform_buffer: None,
            quad_buffer: None,
            linear_sampler: None,
            point_sampler: None,
        }
    }

    /// Get settings.
    pub fn settings(&self) -> &SsrSettings {
        &self.settings
    }

    /// Set settings.
    pub fn set_settings(&mut self, settings: SsrSettings) {
        self.settings = settings;
    }

    /// Set maximum ray distance.
    pub fn set_max_distance(&mut self, distance: f32) {
        self.settings.max_distance = distance.max(1.0);
    }

    /// Set depth thickness.
    pub fn set_thickness(&mut self, thickness: f32) {
        self.settings.thickness = thickness.clamp(0.01, 2.0);
    }

    /// Set reflection intensity.
    pub fn set_intensity(&mut self, intensity: f32) {
        self.settings.intensity = intensity.clamp(0.0, 1.0);
    }

    /// Set quality preset.
    pub fn set_quality(&mut self, quality: SsrQuality) {
        self.quality = quality;
    }

    /// Set projection matrices.
    pub fn set_projection(&mut self, proj: [[f32; 4]; 4], inv_proj: [[f32; 4]; 4], near: f32, far: f32) {
        self.projection = proj;
        self.inv_projection = inv_proj;
        self.near = near;
        self.far = far;
    }

    /// Set view matrix.
    pub fn set_view(&mut self, view: [[f32; 4]; 4]) {
        self.view = view;
    }

    fn step_count(&self) -> u32 {
        match self.quality {
            SsrQuality::Low => 16,
            SsrQuality::Medium => 32,
            SsrQuality::High => 64,
            SsrQuality::Ultra => 128,
        }
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        // Create samplers
        self.linear_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSR Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.point_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSR Point Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        // Create bind group layout
        self.bind_group_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Bind Group Layout"),
            entries: &[
                // Scene color texture
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
        let uniform = SsrUniform {
            projection: self.projection,
            inv_projection: self.inv_projection,
            view: self.view,
            resolution: [1.0 / width as f32, 1.0 / height as f32, width as f32, height as f32],
            params: [self.settings.max_distance, self.settings.thickness, self.settings.intensity, self.step_count() as f32],
            params2: [self.near, self.far, self.settings.roughness_threshold, 0.0],
        };
        self.uniform_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSR Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        // Create quad buffer
        self.quad_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSR Quad Buffer"),
            contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        // Create pipeline
        self.create_pipeline(device, format);
    }

    fn create_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Pipeline Layout"),
            bind_group_layouts: &[self.bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSR Shader"),
            source: wgpu::ShaderSource::Wgsl(SSR_SHADER.into()),
        });

        self.pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSR Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
            let uniform = SsrUniform {
                projection: self.projection,
                inv_projection: self.inv_projection,
                view: self.view,
                resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
                params: [self.settings.max_distance, self.settings.thickness, self.settings.intensity, self.step_count() as f32],
                params2: [self.near, self.far, self.settings.roughness_threshold, 0.0],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }

    /// Render SSR effect with depth texture.
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
            self.blit(encoder, input, output, device, queue);
            return;
        }

        let Some(ref pipeline) = self.pipeline else { return };
        let Some(ref bind_group_layout) = self.bind_group_layout else { return };
        let Some(ref linear_sampler) = self.linear_sampler else { return };
        let Some(ref point_sampler) = self.point_sampler else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };

        self.update_uniforms(queue);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR Bind Group"),
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
            label: Some("SSR Pass"),
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

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, quad_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }

    fn blit(&self, encoder: &mut wgpu::CommandEncoder, input: &wgpu::TextureView, output: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Set intensity to 0 for passthrough
        if let Some(ref buffer) = self.uniform_buffer {
            let uniform = SsrUniform {
                projection: self.projection,
                inv_projection: self.inv_projection,
                view: self.view,
                resolution: [1.0 / self.width as f32, 1.0 / self.height as f32, self.width as f32, self.height as f32],
                params: [self.settings.max_distance, self.settings.thickness, 0.0, self.step_count() as f32],
                params2: [self.near, self.far, self.settings.roughness_threshold, 0.0],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }

        let Some(ref pipeline) = self.pipeline else { return };
        let Some(ref bind_group_layout) = self.bind_group_layout else { return };
        let Some(ref linear_sampler) = self.linear_sampler else { return };
        let Some(ref point_sampler) = self.point_sampler else { return };
        let Some(ref uniform_buffer) = self.uniform_buffer else { return };
        let Some(ref quad_buffer) = self.quad_buffer else { return };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR Blit Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input), // Placeholder, not used when intensity=0
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
            label: Some("SSR Blit Pass"),
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

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, quad_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }
}

impl Default for SsrPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for SsrPass {
    fn name(&self) -> &str {
        "ssr"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn resize(&mut self, width: u32, height: u32, _device: &wgpu::Device) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
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

const SSR_SHADER: &str = r#"
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
    // 1/width, 1/height, width, height
    resolution: vec4<f32>,
    // max_distance, thickness, intensity, step_count
    params: vec4<f32>,
    // near, far, roughness_threshold, 0
    params2: vec4<f32>,
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

// Linearize depth
fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

// Reconstruct view-space position from UV and depth
fn get_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let near = params.params2.x;
    let far = params.params2.y;

    // Convert UV to clip space
    let clip = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);

    // Transform to view space
    var view_pos = params.inv_projection * clip;
    view_pos /= view_pos.w;

    return view_pos.xyz;
}

// Reconstruct normal from depth derivatives
fn get_normal(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let texel = params.resolution.xy;

    // Sample neighboring depths
    let depth_r = textureSampleLevel(depth_texture, point_sampler, uv + vec2<f32>(texel.x, 0.0), 0);
    let depth_u = textureSampleLevel(depth_texture, point_sampler, uv + vec2<f32>(0.0, texel.y), 0);

    // Get view positions
    let pos_c = get_view_pos(uv, depth);
    let pos_r = get_view_pos(uv + vec2<f32>(texel.x, 0.0), depth_r);
    let pos_u = get_view_pos(uv + vec2<f32>(0.0, texel.y), depth_u);

    // Calculate normal from cross product
    let normal = normalize(cross(pos_r - pos_c, pos_u - pos_c));

    return normal;
}

// Project view position to screen UV
fn project_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    var clip = params.projection * vec4<f32>(view_pos, 1.0);
    clip /= clip.w;

    let screen_uv = clip.xy * 0.5 + 0.5;
    return vec3<f32>(screen_uv, clip.z);
}

// Ray march in screen space
fn ray_march(origin: vec3<f32>, direction: vec3<f32>, step_count: i32) -> vec4<f32> {
    let max_distance = params.params.x;
    let thickness = params.params.y;
    let near = params.params2.x;
    let far = params.params2.y;

    let step_size = max_distance / f32(step_count);
    var ray_pos = origin;

    for (var i = 0; i < step_count; i = i + 1) {
        ray_pos += direction * step_size;

        // Project to screen
        let screen = project_to_screen(ray_pos);
        let uv = screen.xy;

        // Check bounds
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }

        // Sample depth at this position
        let sampled_depth = textureSampleLevel(depth_texture, point_sampler, uv, 0);
        let sampled_pos = get_view_pos(uv, sampled_depth);

        // Check for hit
        let depth_diff = ray_pos.z - sampled_pos.z;
        if (depth_diff > 0.0 && depth_diff < thickness) {
            // Hit! Calculate fade based on distance and edge
            let edge_fade = 1.0 - pow(max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0, 2.0);
            let distance_fade = 1.0 - f32(i) / f32(step_count);
            let fade = edge_fade * distance_fade;

            return vec4<f32>(uv, fade, 1.0);
        }
    }

    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let intensity = params.params.z;
    let step_count = i32(params.params.w);

    // Sample scene color
    let scene_color = textureSampleLevel(scene_texture, linear_sampler, in.uv, 0.0);

    // Early out if no reflection
    if (intensity <= 0.0) {
        return scene_color;
    }

    // Get depth
    let depth = textureSampleLevel(depth_texture, point_sampler, in.uv, 0);

    // Skip sky/far plane
    if (depth >= 0.9999) {
        return scene_color;
    }

    // Get view position and normal
    let view_pos = get_view_pos(in.uv, depth);
    let normal = get_normal(in.uv, depth);

    // Calculate view direction (from camera to point)
    let view_dir = normalize(view_pos);

    // Calculate reflection direction
    let reflect_dir = reflect(view_dir, normal);

    // Check if reflection is valid (pointing away from camera)
    if (dot(reflect_dir, normal) < 0.0) {
        return scene_color;
    }

    // Fresnel effect - more reflection at grazing angles
    let fresnel = pow(1.0 - max(dot(-view_dir, normal), 0.0), 2.0);

    // Ray march
    let hit = ray_march(view_pos, reflect_dir, step_count);

    if (hit.w > 0.0) {
        // Sample reflected color
        let reflect_color = textureSampleLevel(scene_texture, linear_sampler, hit.xy, 0.0);

        // Blend based on intensity, fresnel, and fade
        let blend = intensity * fresnel * hit.z;
        return vec4<f32>(mix(scene_color.rgb, reflect_color.rgb, blend), scene_color.a);
    }

    return scene_color;
}
"#;
