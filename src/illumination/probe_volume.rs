//! Irradiance probe volume for global illumination fallback.
//!
//! Uses L2 Spherical Harmonics to store incoming radiance at probe positions.
//! Provides stable GI when SSGI rays go off-screen.

use wgpu::util::DeviceExt;

/// L2 Spherical Harmonics coefficients (9 * vec3).
/// Used for storing irradiance at a single probe point.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphericalHarmonics {
    /// 9 coefficients for R, G, B channels.
    /// Flattened: [L00, L1-1, L10, L11, L2-2, L2-1, L20, L21, L22]
    /// Stored as vec4s (xyz = color, w = padding) for GPU alignment.
    pub coeffs: [[f32; 4]; 9],
}

impl Default for SphericalHarmonics {
    fn default() -> Self {
        Self {
            coeffs: [[0.0; 4]; 9],
        }
    }
}

/// Probe grid info uniform for GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProbeGridInfo {
    /// Origin xyz, w = spacing
    pub origin: [f32; 4],
    /// Dimensions xyz (as floats for WGSL), w = total_probes
    pub dim: [f32; 4],
}

/// A 3D grid of irradiance probes for global illumination.
pub struct ProbeVolume {
    /// World-space origin of the probe grid.
    pub origin: [f32; 3],
    /// Dimensions of the probe grid (e.g., 8x4x8).
    pub dim: [u32; 3],
    /// Distance between probes in world units.
    pub spacing: f32,
    /// Whether the probe system is enabled.
    pub enabled: bool,

    // GPU resources
    probe_buffer: Option<wgpu::Buffer>,
    info_buffer: Option<wgpu::Buffer>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    bind_group: Option<wgpu::BindGroup>,

    // CPU data
    probes: Vec<SphericalHarmonics>,
}

impl ProbeVolume {
    /// Create a new probe volume.
    ///
    /// # Arguments
    /// * `origin` - World-space origin (corner) of the probe grid
    /// * `dim` - Number of probes in each dimension (x, y, z)
    /// * `spacing` - Distance between probes in world units
    pub fn new(origin: [f32; 3], dim: [u32; 3], spacing: f32) -> Self {
        let count = (dim[0] * dim[1] * dim[2]) as usize;
        Self {
            origin,
            dim,
            spacing,
            enabled: true,
            probe_buffer: None,
            info_buffer: None,
            bind_group_layout: None,
            bind_group: None,
            probes: vec![SphericalHarmonics::default(); count],
        }
    }

    /// Create a default probe volume centered around origin.
    pub fn default_volume() -> Self {
        // 8x4x8 grid, 2 units spacing, centered at origin
        let dim = [8, 4, 8];
        let spacing = 2.0;
        let origin = [
            -(dim[0] as f32 * spacing) / 2.0,
            0.0, // Start at ground level
            -(dim[2] as f32 * spacing) / 2.0,
        ];
        Self::new(origin, dim, spacing)
    }

    /// Get the bind group layout for shader access.
    pub fn bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.bind_group_layout.as_ref()
    }

    /// Get the bind group for shader access.
    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Create storage buffer for probe SH data
        let probe_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Probe Volume SH Buffer"),
            contents: bytemuck::cast_slice(&self.probes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create uniform buffer for grid info
        let info = self.create_grid_info();
        let info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Probe Grid Info Buffer"),
            contents: bytemuck::cast_slice(&[info]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Probe Volume Bind Group Layout"),
            entries: &[
                // Probe SH Data (Storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Grid Info (Uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Probe Volume Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: probe_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: info_buffer.as_entire_binding(),
                },
            ],
        });

        self.probe_buffer = Some(probe_buffer);
        self.info_buffer = Some(info_buffer);
        self.bind_group_layout = Some(layout);
        self.bind_group = Some(bind_group);

        // Bake initial sky lighting
        self.bake_sky(queue, [0.6, 0.75, 0.9], [0.2, 0.18, 0.15]);
    }

    fn create_grid_info(&self) -> ProbeGridInfo {
        ProbeGridInfo {
            origin: [self.origin[0], self.origin[1], self.origin[2], self.spacing],
            dim: [
                self.dim[0] as f32,
                self.dim[1] as f32,
                self.dim[2] as f32,
                self.probes.len() as f32,
            ],
        }
    }

    /// Bake sky/ground gradient into all probes.
    /// This provides immediate ambient GI even without ray tracing.
    pub fn bake_sky(&mut self, queue: &wgpu::Queue, sky_color: [f32; 3], ground_color: [f32; 3]) {
        let sh = self.create_sky_sh(sky_color, ground_color);

        // Fill all probes with sky SH
        for probe in self.probes.iter_mut() {
            *probe = sh;
        }

        // Upload to GPU
        if let Some(buffer) = &self.probe_buffer {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.probes));
        }
    }

    /// Bake sky with a directional sun contribution.
    pub fn bake_sky_with_sun(
        &mut self,
        queue: &wgpu::Queue,
        sky_color: [f32; 3],
        ground_color: [f32; 3],
        sun_dir: [f32; 3],
        sun_color: [f32; 3],
        sun_intensity: f32,
    ) {
        let sky_sh = self.create_sky_sh(sky_color, ground_color);
        let sun_sh = self.create_directional_sh(sun_dir, sun_color, sun_intensity);

        // Combine sky and sun SH
        for probe in self.probes.iter_mut() {
            for i in 0..9 {
                probe.coeffs[i][0] = sky_sh.coeffs[i][0] + sun_sh.coeffs[i][0];
                probe.coeffs[i][1] = sky_sh.coeffs[i][1] + sun_sh.coeffs[i][1];
                probe.coeffs[i][2] = sky_sh.coeffs[i][2] + sun_sh.coeffs[i][2];
            }
        }

        if let Some(buffer) = &self.probe_buffer {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.probes));
        }
    }

    /// Create SH coefficients for a sky/ground gradient.
    fn create_sky_sh(&self, sky: [f32; 3], ground: [f32; 3]) -> SphericalHarmonics {
        let mut sh = SphericalHarmonics::default();

        // Average color (ambient)
        let avg = [
            (sky[0] + ground[0]) * 0.5,
            (sky[1] + ground[1]) * 0.5,
            (sky[2] + ground[2]) * 0.5,
        ];

        // Vertical gradient (sky - ground)
        let diff = [
            (sky[0] - ground[0]) * 0.5,
            (sky[1] - ground[1]) * 0.5,
            (sky[2] - ground[2]) * 0.5,
        ];

        // SH basis function normalization constants
        let c0 = 0.282095; // Y_00
        let c1 = 0.488603; // Y_1m

        // Band 0 (L00) - Ambient/DC term
        sh.coeffs[0] = [avg[0] / c0, avg[1] / c0, avg[2] / c0, 0.0];

        // Band 1 (L1-1, L10, L11) - Directional terms
        // L1-1 corresponds to Y axis (vertical)
        sh.coeffs[1] = [diff[0] / c1, diff[1] / c1, diff[2] / c1, 0.0]; // Y (up)
        // L10 and L11 are 0 for pure vertical gradient
        sh.coeffs[2] = [0.0, 0.0, 0.0, 0.0]; // Z
        sh.coeffs[3] = [0.0, 0.0, 0.0, 0.0]; // X

        sh
    }

    /// Create SH coefficients for a directional light source.
    fn create_directional_sh(
        &self,
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    ) -> SphericalHarmonics {
        let mut sh = SphericalHarmonics::default();

        // Normalize direction
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        let d = if len > 0.0 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        };

        let irradiance = [
            color[0] * intensity,
            color[1] * intensity,
            color[2] * intensity,
        ];

        // SH projection of a directional light
        // These are the ZH (zonal harmonics) coefficients for clamped cosine
        let c0 = 0.282095;
        let c1 = 0.488603;
        let c2 = 1.092548;
        let c3 = 0.315392;
        let c4 = 0.546274;

        // Band 0
        sh.coeffs[0] = [
            irradiance[0] * c0 * 0.886227,
            irradiance[1] * c0 * 0.886227,
            irradiance[2] * c0 * 0.886227,
            0.0,
        ];

        // Band 1
        let b1 = 1.023328;
        sh.coeffs[1] = [
            irradiance[0] * c1 * b1 * d[1],
            irradiance[1] * c1 * b1 * d[1],
            irradiance[2] * c1 * b1 * d[1],
            0.0,
        ];
        sh.coeffs[2] = [
            irradiance[0] * c1 * b1 * d[2],
            irradiance[1] * c1 * b1 * d[2],
            irradiance[2] * c1 * b1 * d[2],
            0.0,
        ];
        sh.coeffs[3] = [
            irradiance[0] * c1 * b1 * d[0],
            irradiance[1] * c1 * b1 * d[0],
            irradiance[2] * c1 * b1 * d[0],
            0.0,
        ];

        // Band 2 (simplified)
        let b2 = 0.495416;
        sh.coeffs[4] = [
            irradiance[0] * c2 * b2 * d[0] * d[1],
            irradiance[1] * c2 * b2 * d[0] * d[1],
            irradiance[2] * c2 * b2 * d[0] * d[1],
            0.0,
        ];
        sh.coeffs[5] = [
            irradiance[0] * c2 * b2 * d[1] * d[2],
            irradiance[1] * c2 * b2 * d[1] * d[2],
            irradiance[2] * c2 * b2 * d[1] * d[2],
            0.0,
        ];
        sh.coeffs[6] = [
            irradiance[0] * c3 * b2 * (3.0 * d[2] * d[2] - 1.0),
            irradiance[1] * c3 * b2 * (3.0 * d[2] * d[2] - 1.0),
            irradiance[2] * c3 * b2 * (3.0 * d[2] * d[2] - 1.0),
            0.0,
        ];
        sh.coeffs[7] = [
            irradiance[0] * c2 * b2 * d[0] * d[2],
            irradiance[1] * c2 * b2 * d[0] * d[2],
            irradiance[2] * c2 * b2 * d[0] * d[2],
            0.0,
        ];
        sh.coeffs[8] = [
            irradiance[0] * c4 * b2 * (d[0] * d[0] - d[1] * d[1]),
            irradiance[1] * c4 * b2 * (d[0] * d[0] - d[1] * d[1]),
            irradiance[2] * c4 * b2 * (d[0] * d[0] - d[1] * d[1]),
            0.0,
        ];

        sh
    }

    /// Update grid info on the GPU (call after changing origin/spacing).
    pub fn update_grid_info(&self, queue: &wgpu::Queue) {
        if let Some(buffer) = &self.info_buffer {
            let info = self.create_grid_info();
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[info]));
        }
    }

    /// Set the volume origin.
    pub fn set_origin(&mut self, origin: [f32; 3]) {
        self.origin = origin;
    }

    /// Set the probe spacing.
    pub fn set_spacing(&mut self, spacing: f32) {
        self.spacing = spacing.max(0.1);
    }

    /// Get probe count.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }
}

impl Default for ProbeVolume {
    fn default() -> Self {
        Self::default_volume()
    }
}
