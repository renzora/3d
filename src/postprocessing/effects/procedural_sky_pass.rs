//! Procedural sky rendering with atmospheric scattering.
//!
//! Generates a physically-based sky without requiring cubemap textures.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Procedural sky settings.
#[derive(Debug, Clone)]
pub struct ProceduralSkySettings {
    /// Sun direction (normalized).
    pub sun_direction: [f32; 3],
    /// Sun intensity multiplier.
    pub sun_intensity: f32,
    /// Rayleigh scattering coefficient (controls blue sky).
    pub rayleigh_coefficient: f32,
    /// Mie scattering coefficient (controls haze/glow around sun).
    pub mie_coefficient: f32,
    /// Mie scattering direction (-1 to 1, controls sun glow spread).
    pub mie_directional_g: f32,
    /// Atmospheric turbidity (haziness, 1-10).
    pub turbidity: f32,
    /// Sun disk angular size in degrees.
    pub sun_disk_size: f32,
    /// Sun disk brightness.
    pub sun_disk_intensity: f32,
    /// Ground/horizon color.
    pub ground_color: [f32; 3],
    /// Exposure adjustment.
    pub exposure: f32,
    /// Cloud movement speed.
    pub cloud_speed: f32,
}

impl Default for ProceduralSkySettings {
    fn default() -> Self {
        Self {
            sun_direction: [0.5, 0.5, 0.5],
            sun_intensity: 22.0,
            rayleigh_coefficient: 1.0,
            mie_coefficient: 1.0,
            mie_directional_g: 0.8,
            turbidity: 2.0,
            sun_disk_size: 1.0,
            sun_disk_intensity: 100.0,
            ground_color: [0.37, 0.35, 0.33],
            exposure: 1.0,
            cloud_speed: 0.02,
        }
    }
}

/// Procedural sky uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ProceduralSkyUniform {
    inv_view_proj: [[f32; 4]; 4],
    sun_direction: [f32; 3],
    sun_intensity: f32,
    rayleigh_coefficient: f32,
    mie_coefficient: f32,
    mie_directional_g: f32,
    turbidity: f32,
    sun_disk_size: f32,
    sun_disk_intensity: f32,
    _pad1: [f32; 2],
    ground_color: [f32; 3],
    exposure: f32,
    time: f32,
    cloud_speed: f32,
    _pad2: [f32; 2],
}

impl Default for ProceduralSkyUniform {
    fn default() -> Self {
        let settings = ProceduralSkySettings::default();
        Self {
            inv_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            sun_direction: settings.sun_direction,
            sun_intensity: settings.sun_intensity,
            rayleigh_coefficient: settings.rayleigh_coefficient,
            mie_coefficient: settings.mie_coefficient,
            mie_directional_g: settings.mie_directional_g,
            turbidity: settings.turbidity,
            sun_disk_size: settings.sun_disk_size,
            sun_disk_intensity: settings.sun_disk_intensity,
            _pad1: [0.0; 2],
            ground_color: settings.ground_color,
            exposure: settings.exposure,
            time: 0.0,
            cloud_speed: settings.cloud_speed,
            _pad2: [0.0; 2],
        }
    }
}

/// Procedural sky rendering pass.
pub struct ProceduralSkyPass {
    enabled: bool,
    settings: ProceduralSkySettings,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ProceduralSkyPass {
    /// Create a new procedural sky pass.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Procedural Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/procedural_sky.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Procedural Sky Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Procedural Sky Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Procedural Sky Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
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
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let uniform = ProceduralSkyUniform::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Procedural Sky Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Procedural Sky Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            enabled: false, // Disabled by default (use cubemap skybox)
            settings: ProceduralSkySettings::default(),
            pipeline,
            bind_group_layout,
            uniform_buffer,
            bind_group,
        }
    }

    /// Check if procedural sky is enabled.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable procedural sky.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get current settings.
    pub fn settings(&self) -> &ProceduralSkySettings {
        &self.settings
    }

    /// Set sun direction (will be normalized).
    pub fn set_sun_direction(&mut self, x: f32, y: f32, z: f32) {
        let len = (x * x + y * y + z * z).sqrt();
        if len > 0.0 {
            self.settings.sun_direction = [x / len, y / len, z / len];
        }
    }

    /// Set sun direction from azimuth (0-360) and elevation (-90 to 90) in degrees.
    pub fn set_sun_position(&mut self, azimuth: f32, elevation: f32) {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();
        let cos_el = el_rad.cos();
        self.settings.sun_direction = [
            cos_el * az_rad.sin(),
            el_rad.sin(),
            cos_el * az_rad.cos(),
        ];
    }

    /// Set sun intensity.
    pub fn set_sun_intensity(&mut self, intensity: f32) {
        self.settings.sun_intensity = intensity.max(0.0);
    }

    /// Set Rayleigh scattering coefficient (affects blue color).
    pub fn set_rayleigh(&mut self, coefficient: f32) {
        self.settings.rayleigh_coefficient = coefficient.max(0.0);
    }

    /// Set Mie scattering coefficient (affects haze).
    pub fn set_mie(&mut self, coefficient: f32) {
        self.settings.mie_coefficient = coefficient.max(0.0);
    }

    /// Set Mie directional G (-1 to 1).
    pub fn set_mie_directional_g(&mut self, g: f32) {
        self.settings.mie_directional_g = g.clamp(-0.999, 0.999);
    }

    /// Set atmospheric turbidity (1-10).
    pub fn set_turbidity(&mut self, turbidity: f32) {
        self.settings.turbidity = turbidity.clamp(1.0, 10.0);
    }

    /// Set sun disk size in degrees.
    pub fn set_sun_disk_size(&mut self, size: f32) {
        self.settings.sun_disk_size = size.max(0.0);
    }

    /// Set sun disk intensity.
    pub fn set_sun_disk_intensity(&mut self, intensity: f32) {
        self.settings.sun_disk_intensity = intensity.max(0.0);
    }

    /// Set ground color.
    pub fn set_ground_color(&mut self, r: f32, g: f32, b: f32) {
        self.settings.ground_color = [r, g, b];
    }

    /// Set exposure.
    pub fn set_exposure(&mut self, exposure: f32) {
        self.settings.exposure = exposure.max(0.0);
    }

    /// Update uniform buffer with current settings and view matrix.
    pub fn update_uniform(&self, queue: &wgpu::Queue, inv_view_proj: [[f32; 4]; 4], time: f32) {
        let uniform = ProceduralSkyUniform {
            inv_view_proj,
            sun_direction: self.settings.sun_direction,
            sun_intensity: self.settings.sun_intensity,
            rayleigh_coefficient: self.settings.rayleigh_coefficient,
            mie_coefficient: self.settings.mie_coefficient,
            mie_directional_g: self.settings.mie_directional_g,
            turbidity: self.settings.turbidity,
            sun_disk_size: self.settings.sun_disk_size,
            sun_disk_intensity: self.settings.sun_disk_intensity,
            _pad1: [0.0; 2],
            ground_color: self.settings.ground_color,
            exposure: self.settings.exposure,
            time,
            cloud_speed: self.settings.cloud_speed,
            _pad2: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Set cloud movement speed.
    pub fn set_cloud_speed(&mut self, speed: f32) {
        self.settings.cloud_speed = speed.max(0.0);
    }

    /// Render the procedural sky.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if !self.enabled {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    /// Generate cubemap faces for IBL from current procedural sky settings.
    /// Returns 6 faces in order: +X, -X, +Y, -Y, +Z, -Z (RGBA8 data).
    pub fn generate_cubemap_faces(&self, size: u32) -> [Vec<u8>; 6] {
        let mut faces: [Vec<u8>; 6] = Default::default();
        let settings = &self.settings;
        let sun_dir = normalize(settings.sun_direction);

        for face in 0..6 {
            let mut data = vec![0u8; (size * size * 4) as usize];

            for y in 0..size {
                for x in 0..size {
                    let ray_dir = normalize(texel_to_direction(face, x, y, size));

                    // Compute sky color using atmospheric scattering
                    let color = if ray_dir[1] < 0.0 {
                        // Below horizon - ground color
                        let ground_ambient = (sun_dir[1] * 0.5 + 0.5).max(0.1);
                        [
                            settings.ground_color[0] * ground_ambient,
                            settings.ground_color[1] * ground_ambient,
                            settings.ground_color[2] * ground_ambient,
                        ]
                    } else {
                        // Sky with atmospheric scattering
                        let mut sky = atmosphere(ray_dir, sun_dir, settings);

                        // Add sun disk (but not for IBL - too bright)
                        // Skip sun disk for environment map to avoid artifacts

                        // Apply exposure
                        sky[0] *= settings.exposure;
                        sky[1] *= settings.exposure;
                        sky[2] *= settings.exposure;

                        sky
                    };

                    // Tonemap for LDR output (simple Reinhard)
                    let mapped = [
                        color[0] / (1.0 + color[0]),
                        color[1] / (1.0 + color[1]),
                        color[2] / (1.0 + color[2]),
                    ];

                    let idx = ((y * size + x) * 4) as usize;
                    data[idx] = (mapped[0].clamp(0.0, 1.0) * 255.0) as u8;
                    data[idx + 1] = (mapped[1].clamp(0.0, 1.0) * 255.0) as u8;
                    data[idx + 2] = (mapped[2].clamp(0.0, 1.0) * 255.0) as u8;
                    data[idx + 3] = 255;
                }
            }

            faces[face] = data;
        }

        faces
    }
}

// ============ Helper functions for cubemap generation ============

const PI: f32 = std::f32::consts::PI;

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Convert cubemap face texel to world direction.
fn texel_to_direction(face: usize, x: u32, y: u32, size: u32) -> [f32; 3] {
    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

    match face {
        0 => [1.0, -v, -u],   // +X
        1 => [-1.0, -v, u],   // -X
        2 => [u, 1.0, v],     // +Y
        3 => [u, -1.0, -v],   // -Y
        4 => [u, -v, 1.0],    // +Z
        5 => [-u, -v, -1.0],  // -Z
        _ => [0.0, 0.0, 1.0],
    }
}

/// Rayleigh phase function
fn rayleigh_phase(cos_theta: f32) -> f32 {
    (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta)
}

/// Mie phase function (Henyey-Greenstein)
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 1.0 - g2;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
    (1.0 / (4.0 * PI)) * num / denom
}

/// Get Rayleigh scattering coefficients (boosted blue for vibrant sky)
fn get_beta_rayleigh(rayleigh_coefficient: f32) -> [f32; 3] {
    [
        6.5e-6 * rayleigh_coefficient,
        15.0e-6 * rayleigh_coefficient,
        40.0e-6 * rayleigh_coefficient,
    ]
}

/// Get Mie scattering coefficients
fn get_beta_mie(mie_coefficient: f32, turbidity: f32) -> [f32; 3] {
    let base = 21e-6 * mie_coefficient * turbidity;
    [base, base, base]
}

/// Optical depth for Rayleigh scattering
fn optical_depth_rayleigh(y: f32) -> f32 {
    let h_r = 8500.0;
    (-y.max(0.0) * 5.0).exp() * h_r
}

/// Optical depth for Mie scattering
fn optical_depth_mie(y: f32) -> f32 {
    let h_m = 1200.0;
    (-y.max(0.0) * 2.5).exp() * h_m
}

/// Main atmospheric scattering calculation
fn atmosphere(ray_dir: [f32; 3], sun_dir: [f32; 3], settings: &ProceduralSkySettings) -> [f32; 3] {
    let cos_theta = dot(ray_dir, sun_dir);

    let beta_r = get_beta_rayleigh(settings.rayleigh_coefficient);
    let beta_m = get_beta_mie(settings.mie_coefficient, settings.turbidity);

    let y = ray_dir[1];
    let depth_r = optical_depth_rayleigh(y);
    let depth_m = optical_depth_mie(y);

    // In-scattering
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, settings.mie_directional_g);

    // Scattering contributions
    let scatter_r = [
        beta_r[0] * phase_r * depth_r,
        beta_r[1] * phase_r * depth_r,
        beta_r[2] * phase_r * depth_r,
    ];
    let scatter_m = [
        beta_m[0] * phase_m * depth_m,
        beta_m[1] * phase_m * depth_m,
        beta_m[2] * phase_m * depth_m,
    ];

    // Sun color
    let sun_color = [
        1.0 * settings.sun_intensity,
        0.98 * settings.sun_intensity,
        0.92 * settings.sun_intensity,
    ];

    let mut sky_color = [
        (scatter_r[0] + scatter_m[0]) * sun_color[0],
        (scatter_r[1] + scatter_m[1]) * sun_color[1],
        (scatter_r[2] + scatter_m[2]) * sun_color[2],
    ];

    // Add vibrant blue gradient for clear sky look
    let zenith_blue = [
        0.15 * settings.sun_intensity * 0.08,
        0.35 * settings.sun_intensity * 0.08,
        0.85 * settings.sun_intensity * 0.08,
    ];
    let blue_gradient = y.max(0.0).powf(0.6);
    sky_color[0] += zenith_blue[0] * blue_gradient;
    sky_color[1] += zenith_blue[1] * blue_gradient;
    sky_color[2] += zenith_blue[2] * blue_gradient;

    // Improved horizon - warmer, less grey
    let horizon_factor = 1.0 - y;
    let horizon_warmth = horizon_factor.powf(3.0) * 0.15;
    let horizon_tint = [
        0.9 * horizon_warmth * settings.sun_intensity * 0.5,
        0.85 * horizon_warmth * settings.sun_intensity * 0.5,
        0.75 * horizon_warmth * settings.sun_intensity * 0.5,
    ];
    sky_color[0] += horizon_tint[0];
    sky_color[1] += horizon_tint[1];
    sky_color[2] += horizon_tint[2];

    // Sunset/sunrise colors
    let sun_height = sun_dir[1];
    if sun_height < 0.3 {
        let sunset_factor = 1.0 - sun_height / 0.3;
        let horizon_glow = horizon_factor.powf(4.0) * 0.4;
        sky_color[0] += 1.8 * sunset_factor * horizon_glow * sun_color[0] * 0.5;
        sky_color[1] += 0.6 * sunset_factor * horizon_glow * sun_color[1] * 0.5;
        sky_color[2] += 0.25 * sunset_factor * horizon_glow * sun_color[2] * 0.5;
    }

    sky_color
}
