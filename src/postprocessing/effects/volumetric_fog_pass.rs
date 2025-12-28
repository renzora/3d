//! Volumetric Fog post-processing effect with god rays.
//!
//! Implements atmospheric scattering through ray marching:
//! 1. Ray march through fog volume from camera
//! 2. Accumulate in-scattering from light sources
//! 3. Apply extinction (absorption) along the ray
//! 4. Composite with scene

use crate::postprocessing::pass::{FullscreenVertex, FULLSCREEN_QUAD_VERTICES};
use wgpu::util::DeviceExt;

/// Volumetric fog quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VolumetricFogQuality {
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

impl VolumetricFogQuality {
    /// Get the number of ray march steps for this quality level.
    pub fn step_count(&self) -> u32 {
        match self {
            Self::Low => 16,
            Self::Medium => 32,
            Self::High => 64,
            Self::Ultra => 128,
        }
    }
}

/// Volumetric fog settings.
#[derive(Debug, Clone)]
pub struct VolumetricFogSettings {
    /// Quality preset (affects step count).
    pub quality: VolumetricFogQuality,
    /// Fog density (0.0 - 1.0).
    pub density: f32,
    /// Fog start distance from camera.
    pub start_distance: f32,
    /// Fog end distance (max ray march distance).
    pub end_distance: f32,
    /// Fog height falloff (0 = uniform, higher = ground fog).
    pub height_falloff: f32,
    /// Base fog height (fog is densest below this).
    pub base_height: f32,
    /// Scattering coefficient for in-scattering.
    pub scattering: f32,
    /// Absorption coefficient.
    pub absorption: f32,
    /// God ray intensity (light scattering toward camera).
    pub god_ray_intensity: f32,
    /// God ray decay (how quickly rays fade).
    pub god_ray_decay: f32,
    /// Fog color.
    pub fog_color: [f32; 3],
    /// Enable god rays.
    pub god_rays_enabled: bool,
}

impl Default for VolumetricFogSettings {
    fn default() -> Self {
        Self {
            quality: VolumetricFogQuality::Medium,
            density: 0.02,
            start_distance: 1.0,
            end_distance: 100.0,
            height_falloff: 0.1,
            base_height: 0.0,
            scattering: 0.5,
            absorption: 0.1,
            god_ray_intensity: 1.0,
            god_ray_decay: 0.95,
            fog_color: [0.7, 0.8, 0.9],
            god_rays_enabled: true,
        }
    }
}

/// Volumetric fog uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VolumetricFogUniform {
    /// Inverse view-projection matrix.
    inv_view_proj: [[f32; 4]; 4],
    /// Camera position.
    camera_pos: [f32; 4],
    /// Light direction (xyz) + intensity (w).
    light_dir: [f32; 4],
    /// Light color (rgb) + god_ray_enabled (w).
    light_color: [f32; 4],
    /// Fog color (rgb) + density (w).
    fog_color: [f32; 4],
    /// start_distance, end_distance, height_falloff, base_height.
    fog_params: [f32; 4],
    /// scattering, absorption, god_ray_intensity, god_ray_decay.
    scatter_params: [f32; 4],
    /// step_count, time, 1/width, 1/height.
    render_params: [f32; 4],
    /// near, far, 0, 0.
    near_far: [f32; 4],
}

/// Volumetric fog post-processing pass.
pub struct VolumetricFogPass {
    enabled: bool,
    settings: VolumetricFogSettings,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    // Matrices
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    near: f32,
    far: f32,
    // Light info
    light_dir: [f32; 3],
    light_intensity: f32,
    light_color: [f32; 3],
    // Frame counter for noise
    frame: u32,
    // Pipeline
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    // Textures
    fog_texture: Option<wgpu::Texture>,
    fog_view: Option<wgpu::TextureView>,
    // Buffers
    uniform_buffer: Option<wgpu::Buffer>,
    quad_buffer: Option<wgpu::Buffer>,
    // Samplers
    linear_sampler: Option<wgpu::Sampler>,
    point_sampler: Option<wgpu::Sampler>,
}

impl VolumetricFogPass {
    /// Create a new volumetric fog pass.
    pub fn new() -> Self {
        Self {
            enabled: false,
            settings: VolumetricFogSettings::default(),
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            inv_view_proj: [[0.0; 4]; 4],
            camera_pos: [0.0; 3],
            near: 0.1,
            far: 100.0,
            light_dir: [0.0, -1.0, 0.0],
            light_intensity: 1.0,
            light_color: [1.0, 1.0, 1.0],
            frame: 0,
            pipeline: None,
            bind_group_layout: None,
            fog_texture: None,
            fog_view: None,
            uniform_buffer: None,
            quad_buffer: None,
            linear_sampler: None,
            point_sampler: None,
        }
    }

    /// Check if enabled.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Set enabled state.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get settings.
    pub fn settings(&self) -> &VolumetricFogSettings {
        &self.settings
    }

    /// Set quality preset.
    pub fn set_quality(&mut self, quality: VolumetricFogQuality) {
        self.settings.quality = quality;
    }

    /// Set fog density.
    pub fn set_density(&mut self, density: f32) {
        self.settings.density = density.clamp(0.0, 1.0);
    }

    /// Set start distance.
    pub fn set_start_distance(&mut self, distance: f32) {
        self.settings.start_distance = distance.max(0.0);
    }

    /// Set end distance.
    pub fn set_end_distance(&mut self, distance: f32) {
        self.settings.end_distance = distance.max(self.settings.start_distance + 1.0);
    }

    /// Set height falloff.
    pub fn set_height_falloff(&mut self, falloff: f32) {
        self.settings.height_falloff = falloff.max(0.0);
    }

    /// Set base height.
    pub fn set_base_height(&mut self, height: f32) {
        self.settings.base_height = height;
    }

    /// Set scattering coefficient.
    pub fn set_scattering(&mut self, scattering: f32) {
        self.settings.scattering = scattering.clamp(0.0, 2.0);
    }

    /// Set absorption coefficient.
    pub fn set_absorption(&mut self, absorption: f32) {
        self.settings.absorption = absorption.clamp(0.0, 1.0);
    }

    /// Set god ray intensity.
    pub fn set_god_ray_intensity(&mut self, intensity: f32) {
        self.settings.god_ray_intensity = intensity.max(0.0);
    }

    /// Set god ray decay.
    pub fn set_god_ray_decay(&mut self, decay: f32) {
        self.settings.god_ray_decay = decay.clamp(0.8, 1.0);
    }

    /// Set fog color.
    pub fn set_fog_color(&mut self, r: f32, g: f32, b: f32) {
        self.settings.fog_color = [r, g, b];
    }

    /// Enable or disable god rays.
    pub fn set_god_rays_enabled(&mut self, enabled: bool) {
        self.settings.god_rays_enabled = enabled;
    }

    /// Set camera matrices.
    pub fn set_matrices(
        &mut self,
        inv_view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        near: f32,
        far: f32,
    ) {
        self.inv_view_proj = inv_view_proj;
        self.camera_pos = camera_pos;
        self.near = near;
        self.far = far;
    }

    /// Set light information.
    pub fn set_light(&mut self, direction: [f32; 3], intensity: f32, color: [f32; 3]) {
        self.light_dir = direction;
        self.light_intensity = intensity;
        self.light_color = color;
    }

    /// Initialize the pass.
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
            label: Some("Volumetric Fog Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.point_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Volumetric Fog Point Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        // Create quad buffer
        self.quad_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Volumetric Fog Quad Buffer"),
                contents: bytemuck::cast_slice(&FULLSCREEN_QUAD_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        );

        // Create uniform buffer
        self.uniform_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volumetric Fog Uniform Buffer"),
            size: std::mem::size_of::<VolumetricFogUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volumetric Fog Bind Group Layout"),
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
                // Point sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Uniform buffer
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

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volumetric Fog Shader"),
            source: wgpu::ShaderSource::Wgsl(VOLUMETRIC_FOG_SHADER.into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volumetric Fog Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create pipeline
        self.pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Volumetric Fog Pipeline"),
            layout: Some(&pipeline_layout),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        }));

        self.bind_group_layout = Some(bind_group_layout);

        // Create fog texture
        self.create_textures(device);
    }

    fn create_textures(&mut self, device: &wgpu::Device) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Volumetric Fog Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.fog_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.fog_texture = Some(texture);
    }

    /// Resize the pass.
    pub fn resize(&mut self, width: u32, height: u32, device: &wgpu::Device) {
        if width != self.width || height != self.height {
            self.width = width;
            self.height = height;
            self.create_textures(device);
        }
    }

    /// Render volumetric fog.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        scene_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if !self.enabled {
            return;
        }

        let Some(pipeline) = &self.pipeline else {
            return;
        };
        let Some(bind_group_layout) = &self.bind_group_layout else {
            return;
        };
        let Some(uniform_buffer) = &self.uniform_buffer else {
            return;
        };
        let Some(quad_buffer) = &self.quad_buffer else {
            return;
        };
        let Some(linear_sampler) = &self.linear_sampler else {
            return;
        };
        let Some(point_sampler) = &self.point_sampler else {
            return;
        };

        self.frame = self.frame.wrapping_add(1);

        // Update uniform
        let uniform = VolumetricFogUniform {
            inv_view_proj: self.inv_view_proj,
            camera_pos: [self.camera_pos[0], self.camera_pos[1], self.camera_pos[2], 1.0],
            light_dir: [
                self.light_dir[0],
                self.light_dir[1],
                self.light_dir[2],
                self.light_intensity,
            ],
            light_color: [
                self.light_color[0],
                self.light_color[1],
                self.light_color[2],
                if self.settings.god_rays_enabled { 1.0 } else { 0.0 },
            ],
            fog_color: [
                self.settings.fog_color[0],
                self.settings.fog_color[1],
                self.settings.fog_color[2],
                self.settings.density,
            ],
            fog_params: [
                self.settings.start_distance,
                self.settings.end_distance,
                self.settings.height_falloff,
                self.settings.base_height,
            ],
            scatter_params: [
                self.settings.scattering,
                self.settings.absorption,
                self.settings.god_ray_intensity,
                self.settings.god_ray_decay,
            ],
            render_params: [
                self.settings.quality.step_count() as f32,
                self.frame as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            near_far: [self.near, self.far, 0.0, 0.0],
        };
        queue.write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Volumetric Fog Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_view),
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

        // Render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Volumetric Fog Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
    }
}

impl Default for VolumetricFogPass {
    fn default() -> Self {
        Self::new()
    }
}

// Volumetric fog shader
const VOLUMETRIC_FOG_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,       // xyz = direction, w = intensity
    light_color: vec4<f32>,     // rgb = color, w = god_rays_enabled
    fog_color: vec4<f32>,       // rgb = color, w = density
    fog_params: vec4<f32>,      // start, end, height_falloff, base_height
    scatter_params: vec4<f32>,  // scattering, absorption, god_ray_intensity, god_ray_decay
    render_params: vec4<f32>,   // step_count, frame, 1/width, 1/height
    near_far: vec4<f32>,        // near, far, 0, 0
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
fn linearize_depth(d: f32) -> f32 {
    let near = params.near_far.x;
    let far = params.near_far.y;
    return near * far / (far - d * (far - near));
}

// Reconstruct world position from depth
fn get_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    var world = params.inv_view_proj * ndc;
    world /= world.w;
    return world.xyz;
}

// Henyey-Greenstein phase function for anisotropic scattering
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
}

// Interleaved gradient noise for dithering
fn interleaved_gradient_noise(pixel: vec2<f32>, frame: f32) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    let rotated = pixel + frame * vec2<f32>(5.0, 7.0);
    return fract(magic.z * fract(dot(rotated, magic.xy)));
}

// Calculate fog density at a point
fn get_fog_density(pos: vec3<f32>) -> f32 {
    let base_density = params.fog_color.w;
    let height_falloff = params.fog_params.z;
    let base_height = params.fog_params.w;

    // Height-based falloff (exponential)
    let height_factor = exp(-max(0.0, pos.y - base_height) * height_falloff);

    return base_density * height_factor;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene_color = textureSampleLevel(scene_texture, linear_sampler, in.uv, 0.0).rgb;
    let depth = textureSampleLevel(depth_texture, point_sampler, in.uv, 0);

    // Skip sky
    if (depth >= 0.9999) {
        return vec4<f32>(scene_color, 1.0);
    }

    let world_pos = get_world_pos(in.uv, depth);
    let camera_pos = params.camera_pos.xyz;
    let ray_dir = normalize(world_pos - camera_pos);
    let scene_distance = length(world_pos - camera_pos);

    // Ray march parameters
    let step_count = i32(params.render_params.x);
    let start_dist = params.fog_params.x;
    let end_dist = min(params.fog_params.y, scene_distance);

    // Early out if scene is before fog starts
    if (scene_distance < start_dist) {
        return vec4<f32>(scene_color, 1.0);
    }

    let march_distance = end_dist - start_dist;
    let step_size = march_distance / f32(step_count);

    // Jitter start position for temporal stability
    let noise = interleaved_gradient_noise(in.position.xy, params.render_params.y);
    let jitter = noise * step_size;

    // Light parameters
    let light_dir = normalize(params.light_dir.xyz);
    let light_intensity = params.light_dir.w;
    let light_color = params.light_color.rgb;
    let god_rays_enabled = params.light_color.w > 0.5;
    let god_ray_intensity = params.scatter_params.z;

    // Scattering parameters
    let scattering = params.scatter_params.x;
    let absorption = params.scatter_params.y;
    let extinction = scattering + absorption;

    // Fog color
    let fog_color = params.fog_color.rgb;

    // Accumulate fog
    var transmittance = 1.0;
    var in_scatter = vec3<f32>(0.0);

    for (var i = 0; i < step_count; i++) {
        let t = start_dist + jitter + f32(i) * step_size;
        let sample_pos = camera_pos + ray_dir * t;

        // Get local density
        let density = get_fog_density(sample_pos);

        if (density > 0.001) {
            // Beer-Lambert extinction
            let sample_extinction = extinction * density * step_size;
            let sample_transmittance = exp(-sample_extinction);

            // In-scattering from light
            var light_scatter = vec3<f32>(0.0);

            if (god_rays_enabled) {
                // Phase function for directional scattering
                let cos_theta = dot(ray_dir, -light_dir);
                let phase = henyey_greenstein(cos_theta, 0.5); // g=0.5 for forward scattering

                // Light contribution (simplified - no shadow check for performance)
                light_scatter = light_color * light_intensity * phase * god_ray_intensity;
            }

            // Add ambient fog color
            let ambient_scatter = fog_color * 0.3;

            // Integrate in-scattering
            let scatter_contrib = (light_scatter + ambient_scatter) * scattering * density * step_size;
            in_scatter += scatter_contrib * transmittance;

            // Update transmittance
            transmittance *= sample_transmittance;

            // Early out if fully opaque
            if (transmittance < 0.01) {
                break;
            }
        }
    }

    // Combine with scene
    let final_color = scene_color * transmittance + in_scatter;

    return vec4<f32>(final_color, 1.0);
}
"#;
