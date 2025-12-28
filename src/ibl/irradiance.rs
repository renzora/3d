//! Irradiance map generator for diffuse IBL.
//!
//! Generates a low-resolution cubemap where each texel represents
//! the total diffuse irradiance from the hemisphere oriented in that direction.
//! This is used for the diffuse component of image-based lighting.

use std::f32::consts::PI;

/// Irradiance map generator for diffuse IBL.
pub struct IrradianceGenerator {
    /// Number of samples per hemisphere for convolution.
    sample_count: u32,
}

impl Default for IrradianceGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl IrradianceGenerator {
    /// Create a new irradiance generator with default settings.
    pub fn new() -> Self {
        Self { sample_count: 64 }
    }

    /// Set the number of samples for hemisphere integration.
    pub fn with_samples(mut self, count: u32) -> Self {
        self.sample_count = count;
        self
    }

    /// Generate an irradiance cubemap from source environment faces.
    ///
    /// The output is typically much smaller than the source (e.g., 32x32)
    /// since irradiance is a low-frequency signal.
    ///
    /// Returns 6 faces in order: +X, -X, +Y, -Y, +Z, -Z (RGBA8 data).
    pub fn generate_irradiance_map(
        &self,
        source_faces: &[Vec<u8>; 6],
        source_size: u32,
        output_size: u32,
    ) -> [Vec<u8>; 6] {
        let mut result: [Vec<u8>; 6] = Default::default();

        for face in 0..6 {
            result[face] = self.convolve_face(source_faces, source_size, face, output_size);
        }

        result
    }

    /// Convolve a single face to compute irradiance.
    fn convolve_face(
        &self,
        source_faces: &[Vec<u8>; 6],
        source_size: u32,
        face: usize,
        output_size: u32,
    ) -> Vec<u8> {
        let mut output = vec![0u8; (output_size * output_size * 4) as usize];

        for y in 0..output_size {
            for x in 0..output_size {
                // Convert texel to normal direction
                let normal = texel_to_direction(face, x, y, output_size);
                let normal = normalize(normal);

                // Compute irradiance by integrating over hemisphere
                let irradiance = self.compute_irradiance(source_faces, source_size, normal);

                // Write to output (clamp to [0, 1] for LDR)
                let idx = ((y * output_size + x) * 4) as usize;
                output[idx] = (irradiance[0].clamp(0.0, 1.0) * 255.0) as u8;
                output[idx + 1] = (irradiance[1].clamp(0.0, 1.0) * 255.0) as u8;
                output[idx + 2] = (irradiance[2].clamp(0.0, 1.0) * 255.0) as u8;
                output[idx + 3] = 255;
            }
        }

        output
    }

    /// Compute irradiance for a given normal direction.
    ///
    /// Integrates: E(n) = ∫_Ω L(ω) * max(0, n·ω) dω
    fn compute_irradiance(
        &self,
        source_faces: &[Vec<u8>; 6],
        source_size: u32,
        normal: [f32; 3],
    ) -> [f32; 3] {
        let mut irradiance = [0.0f32; 3];

        // Create tangent space basis
        let up = if normal[1].abs() < 0.999 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };

        let tangent = normalize(cross(up, normal));
        let bitangent = cross(normal, tangent);

        // Sample hemisphere using stratified sampling
        let sample_delta = 1.0 / self.sample_count as f32;
        let mut total_weight = 0.0f32;

        for i in 0..self.sample_count {
            for j in 0..self.sample_count {
                // Stratified sample with jitter
                let xi = [(i as f32 + 0.5) * sample_delta, (j as f32 + 0.5) * sample_delta];

                // Convert to spherical coordinates on hemisphere
                // Using cosine-weighted sampling for efficiency
                let phi = 2.0 * PI * xi[0];
                let cos_theta = (1.0 - xi[1]).sqrt(); // Cosine-weighted
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                // Tangent space direction
                let tangent_dir = [
                    phi.cos() * sin_theta,
                    phi.sin() * sin_theta,
                    cos_theta,
                ];

                // Transform to world space
                let sample_dir = [
                    tangent[0] * tangent_dir[0] + bitangent[0] * tangent_dir[1] + normal[0] * tangent_dir[2],
                    tangent[1] * tangent_dir[0] + bitangent[1] * tangent_dir[1] + normal[1] * tangent_dir[2],
                    tangent[2] * tangent_dir[0] + bitangent[2] * tangent_dir[1] + normal[2] * tangent_dir[2],
                ];

                // Sample environment
                let sample_color = sample_cubemap(source_faces, source_size, sample_dir);

                // For cosine-weighted sampling, weight is just 1 (already accounted for)
                // But we still weight by cos(theta) for proper integration
                let n_dot_l = cos_theta;

                irradiance[0] += sample_color[0] * n_dot_l;
                irradiance[1] += sample_color[1] * n_dot_l;
                irradiance[2] += sample_color[2] * n_dot_l;
                total_weight += n_dot_l;
            }
        }

        // Normalize and apply PI factor for Lambert BRDF
        if total_weight > 0.0 {
            let scale = PI / total_weight;
            [
                irradiance[0] * scale,
                irradiance[1] * scale,
                irradiance[2] * scale,
            ]
        } else {
            [0.0, 0.0, 0.0]
        }
    }
}

/// Generate an irradiance cubemap texture from source faces.
pub fn generate_irradiance_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_faces: &[Vec<u8>; 6],
    source_size: u32,
    output_size: u32,
) -> wgpu::Texture {
    let generator = IrradianceGenerator::new().with_samples(32);
    let irradiance_faces = generator.generate_irradiance_map(source_faces, source_size, output_size);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Irradiance Cubemap"),
        size: wgpu::Extent3d {
            width: output_size,
            height: output_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Upload each face
    for (face, data) in irradiance_faces.iter().enumerate() {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(output_size * 4),
                rows_per_image: Some(output_size),
            },
            wgpu::Extent3d {
                width: output_size,
                height: output_size,
                depth_or_array_layers: 1,
            },
        );
    }

    texture
}

/// Generate a procedural sky irradiance map.
pub fn generate_sky_irradiance(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
) -> wgpu::Texture {
    // Generate source sky faces
    let source_faces = generate_procedural_sky_faces(64);
    generate_irradiance_cubemap(device, queue, &source_faces, 64, size)
}

/// Generate procedural sky faces (simple gradient).
fn generate_procedural_sky_faces(size: u32) -> [Vec<u8>; 6] {
    let mut faces: [Vec<u8>; 6] = Default::default();

    // Sky colors
    let sky_top = [0.4f32, 0.6, 1.0];    // Light blue
    let sky_horizon = [0.7, 0.85, 1.0];   // Pale blue
    let ground = [0.4, 0.35, 0.3];        // Brown/grey

    for face in 0..6 {
        let mut data = vec![0u8; (size * size * 4) as usize];

        for y in 0..size {
            for x in 0..size {
                let dir = texel_to_direction(face, x, y, size);
                let dir = normalize(dir);

                // Blend based on Y direction
                let blend = dir[1] * 0.5 + 0.5; // Map -1..1 to 0..1

                let color = if blend > 0.5 {
                    // Upper hemisphere - sky
                    let t = (blend - 0.5) * 2.0;
                    lerp_color(sky_horizon, sky_top, t)
                } else {
                    // Lower hemisphere - ground
                    let t = blend * 2.0;
                    lerp_color(ground, sky_horizon, t)
                };

                let idx = ((y * size + x) * 4) as usize;
                data[idx] = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
                data[idx + 1] = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
                data[idx + 2] = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
                data[idx + 3] = 255;
            }
        }

        faces[face] = data;
    }

    faces
}

// ============ Helper functions ============

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

/// Sample cubemap at a given direction.
fn sample_cubemap(faces: &[Vec<u8>; 6], size: u32, dir: [f32; 3]) -> [f32; 3] {
    let dir = normalize(dir);
    let abs_dir = [dir[0].abs(), dir[1].abs(), dir[2].abs()];

    let (face, u, v) = if abs_dir[0] >= abs_dir[1] && abs_dir[0] >= abs_dir[2] {
        if dir[0] > 0.0 {
            (0, -dir[2] / abs_dir[0], -dir[1] / abs_dir[0])
        } else {
            (1, dir[2] / abs_dir[0], -dir[1] / abs_dir[0])
        }
    } else if abs_dir[1] >= abs_dir[0] && abs_dir[1] >= abs_dir[2] {
        if dir[1] > 0.0 {
            (2, dir[0] / abs_dir[1], dir[2] / abs_dir[1])
        } else {
            (3, dir[0] / abs_dir[1], -dir[2] / abs_dir[1])
        }
    } else {
        if dir[2] > 0.0 {
            (4, dir[0] / abs_dir[2], -dir[1] / abs_dir[2])
        } else {
            (5, -dir[0] / abs_dir[2], -dir[1] / abs_dir[2])
        }
    };

    let x = ((u * 0.5 + 0.5) * size as f32).clamp(0.0, size as f32 - 1.0) as u32;
    let y = ((v * 0.5 + 0.5) * size as f32).clamp(0.0, size as f32 - 1.0) as u32;

    let idx = ((y * size + x) * 4) as usize;
    let data = &faces[face];

    if idx + 2 < data.len() {
        [
            data[idx] as f32 / 255.0,
            data[idx + 1] as f32 / 255.0,
            data[idx + 2] as f32 / 255.0,
        ]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn lerp_color(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}
