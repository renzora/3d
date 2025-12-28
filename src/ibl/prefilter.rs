//! Pre-filtered environment map generator for specular IBL.
//!
//! Generates mip levels of a cubemap where each level represents
//! increasing roughness, using importance-sampled GGX convolution.

use std::f32::consts::PI;

/// Pre-filter generator for environment maps.
pub struct PrefilterGenerator {
    /// Number of samples for convolution.
    sample_count: u32,
}

impl Default for PrefilterGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefilterGenerator {
    /// Create a new prefilter generator with default settings.
    pub fn new() -> Self {
        Self { sample_count: 32 }
    }

    /// Set the number of samples for convolution.
    pub fn with_samples(mut self, count: u32) -> Self {
        self.sample_count = count;
        self
    }

    /// Generate a prefiltered environment cubemap from source faces.
    ///
    /// Returns a vector of mip levels, each containing 6 faces (RGBA8 data).
    /// Mip 0 = roughness 0 (mirror), higher mips = higher roughness.
    pub fn generate_prefiltered_cubemap(
        &self,
        source_faces: &[Vec<u8>; 6],
        source_size: u32,
        mip_levels: u32,
    ) -> Vec<[Vec<u8>; 6]> {
        let mut result = Vec::with_capacity(mip_levels as usize);

        for mip in 0..mip_levels {
            let mip_size = (source_size >> mip).max(1);
            let roughness = mip as f32 / (mip_levels - 1).max(1) as f32;

            let mut mip_faces: [Vec<u8>; 6] = Default::default();

            for face in 0..6 {
                mip_faces[face] = self.convolve_face(
                    source_faces,
                    source_size,
                    face,
                    mip_size,
                    roughness,
                );
            }

            result.push(mip_faces);
        }

        result
    }

    /// Convolve a single face of the cubemap for a given roughness.
    fn convolve_face(
        &self,
        source_faces: &[Vec<u8>; 6],
        source_size: u32,
        face: usize,
        output_size: u32,
        roughness: f32,
    ) -> Vec<u8> {
        let mut output = vec![0u8; (output_size * output_size * 4) as usize];

        for y in 0..output_size {
            for x in 0..output_size {
                // Convert texel to direction
                let dir = texel_to_direction(face, x, y, output_size);

                // Convolve
                let color = if roughness < 0.01 {
                    // For roughness ~0, just sample directly
                    sample_cubemap(source_faces, source_size, dir)
                } else {
                    self.importance_sample_ggx(source_faces, source_size, dir, roughness)
                };

                // Write to output
                let idx = ((y * output_size + x) * 4) as usize;
                output[idx] = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
                output[idx + 1] = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
                output[idx + 2] = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
                output[idx + 3] = 255;
            }
        }

        output
    }

    /// Importance sample the GGX distribution for a given normal direction.
    fn importance_sample_ggx(
        &self,
        source_faces: &[Vec<u8>; 6],
        source_size: u32,
        n: [f32; 3],
        roughness: f32,
    ) -> [f32; 3] {
        let n = normalize(n);
        let r = n; // Reflection direction = normal for env map
        let v = r; // View direction = reflection direction

        let mut total_color = [0.0f32; 3];
        let mut total_weight = 0.0f32;

        let alpha = roughness * roughness;

        for i in 0..self.sample_count {
            // Hammersley sequence
            let xi = hammersley(i, self.sample_count);

            // Importance sample GGX
            let h = importance_sample_ggx_dir(xi, n, alpha);

            // Reflect view around half vector
            let l = reflect_vec(v, h);
            let l = normalize(l);

            let n_dot_l = dot(n, l).max(0.0);

            if n_dot_l > 0.0 {
                // Sample environment
                let sample_color = sample_cubemap(source_faces, source_size, l);

                total_color[0] += sample_color[0] * n_dot_l;
                total_color[1] += sample_color[1] * n_dot_l;
                total_color[2] += sample_color[2] * n_dot_l;
                total_weight += n_dot_l;
            }
        }

        if total_weight > 0.0 {
            [
                total_color[0] / total_weight,
                total_color[1] / total_weight,
                total_color[2] / total_weight,
            ]
        } else {
            sample_cubemap(source_faces, source_size, n)
        }
    }
}

/// Generate a prefiltered procedural sky cubemap.
/// This creates a simple gradient sky with prefiltered mip levels.
pub fn generate_prefiltered_sky(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
    mip_levels: u32,
) -> wgpu::Texture {
    // Generate source sky faces
    let source_faces = generate_procedural_sky_faces(size);

    // Prefilter
    let generator = PrefilterGenerator::new().with_samples(32);
    let mip_data = generator.generate_prefiltered_cubemap(&source_faces, size, mip_levels);

    // Create texture with mip levels
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Prefiltered Sky Cubemap"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Upload each mip level
    for (mip, faces) in mip_data.iter().enumerate() {
        let mip_size = (size >> mip).max(1);

        for (face, data) in faces.iter().enumerate() {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: mip as u32,
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
                    bytes_per_row: Some(mip_size * 4),
                    rows_per_image: Some(mip_size),
                },
                wgpu::Extent3d {
                    width: mip_size,
                    height: mip_size,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    texture
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
    // Map texel to [-1, 1]
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

    // Find dominant axis
    let (face, u, v) = if abs_dir[0] >= abs_dir[1] && abs_dir[0] >= abs_dir[2] {
        if dir[0] > 0.0 {
            (0, -dir[2] / abs_dir[0], -dir[1] / abs_dir[0])  // +X
        } else {
            (1, dir[2] / abs_dir[0], -dir[1] / abs_dir[0])   // -X
        }
    } else if abs_dir[1] >= abs_dir[0] && abs_dir[1] >= abs_dir[2] {
        if dir[1] > 0.0 {
            (2, dir[0] / abs_dir[1], dir[2] / abs_dir[1])    // +Y
        } else {
            (3, dir[0] / abs_dir[1], -dir[2] / abs_dir[1])   // -Y
        }
    } else {
        if dir[2] > 0.0 {
            (4, dir[0] / abs_dir[2], -dir[1] / abs_dir[2])   // +Z
        } else {
            (5, -dir[0] / abs_dir[2], -dir[1] / abs_dir[2])  // -Z
        }
    };

    // Map [-1, 1] to texel coordinates
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

/// Hammersley sequence for quasi-random sampling.
fn hammersley(i: u32, n: u32) -> [f32; 2] {
    [i as f32 / n as f32, radical_inverse_vdc(i)]
}

/// Van der Corput radical inverse.
fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.3283064365386963e-10
}

/// Importance sample GGX distribution to get half vector.
fn importance_sample_ggx_dir(xi: [f32; 2], n: [f32; 3], alpha: f32) -> [f32; 3] {
    let a2 = alpha * alpha;

    let phi = 2.0 * PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a2 - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    // Tangent space half vector
    let h_tangent = [
        phi.cos() * sin_theta,
        phi.sin() * sin_theta,
        cos_theta,
    ];

    // Create tangent space basis
    let up = if n[1].abs() < 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };

    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);

    // Transform to world space
    [
        tangent[0] * h_tangent[0] + bitangent[0] * h_tangent[1] + n[0] * h_tangent[2],
        tangent[1] * h_tangent[0] + bitangent[1] * h_tangent[1] + n[1] * h_tangent[2],
        tangent[2] * h_tangent[0] + bitangent[2] * h_tangent[1] + n[2] * h_tangent[2],
    ]
}

/// Reflect vector v around normal n.
fn reflect_vec(v: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    let d = 2.0 * dot(v, n);
    [
        d * n[0] - v[0],
        d * n[1] - v[1],
        d * n[2] - v[2],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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
