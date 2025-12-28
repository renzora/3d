//! HDR and EXR image loader for environment maps.
//!
//! Supports loading high dynamic range images in:
//! - Radiance HDR (.hdr) format
//! - OpenEXR (.exr) format
//!
//! Provides equirectangular to cubemap conversion for skybox use.

use std::f32::consts::PI;

/// HDR image data with floating-point RGB values.
pub struct HdrImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGB floating-point data (3 floats per pixel).
    pub data: Vec<f32>,
}

impl HdrImage {
    /// Load an HDR or EXR image from bytes.
    ///
    /// Automatically detects format from the data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HdrError> {
        use image::ImageReader;
        use std::io::Cursor;

        let reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| HdrError::IoError(e.to_string()))?;

        let img = reader
            .decode()
            .map_err(|e| HdrError::DecodeError(e.to_string()))?;

        let rgb32f = img.into_rgb32f();
        let (width, height) = rgb32f.dimensions();
        let data: Vec<f32> = rgb32f.into_raw();

        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Sample the HDR image at UV coordinates with bilinear filtering.
    pub fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let u = u.fract();
        let v = v.fract();
        let u = if u < 0.0 { u + 1.0 } else { u };
        let v = if v < 0.0 { v + 1.0 } else { v };

        let x = u * (self.width - 1) as f32;
        let y = v * (self.height - 1) as f32;

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x.fract();
        let fy = y.fract();

        let c00 = self.get_pixel(x0, y0);
        let c10 = self.get_pixel(x1, y0);
        let c01 = self.get_pixel(x0, y1);
        let c11 = self.get_pixel(x1, y1);

        // Bilinear interpolation
        [
            lerp(lerp(c00[0], c10[0], fx), lerp(c01[0], c11[0], fx), fy),
            lerp(lerp(c00[1], c10[1], fx), lerp(c01[1], c11[1], fx), fy),
            lerp(lerp(c00[2], c10[2], fx), lerp(c01[2], c11[2], fx), fy),
        ]
    }

    /// Get a pixel at integer coordinates.
    fn get_pixel(&self, x: u32, y: u32) -> [f32; 3] {
        let idx = ((y * self.width + x) * 3) as usize;
        if idx + 2 < self.data.len() {
            [self.data[idx], self.data[idx + 1], self.data[idx + 2]]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Convert equirectangular HDR panorama to cubemap faces.
    ///
    /// Returns 6 faces in order: +X, -X, +Y, -Y, +Z, -Z
    /// Each face contains RGBA16F data (4 f16 values per pixel).
    pub fn to_cubemap_faces(&self, face_size: u32) -> [Vec<u8>; 6] {
        let mut faces: [Vec<u8>; 6] = Default::default();

        for face in 0..6 {
            let mut data = vec![0u8; (face_size * face_size * 8) as usize]; // RGBA16F = 8 bytes per pixel

            for y in 0..face_size {
                for x in 0..face_size {
                    // Convert texel to direction
                    let dir = texel_to_direction(face, x, y, face_size);
                    let dir = normalize(dir);

                    // Convert direction to equirectangular UV
                    let (u, v) = direction_to_equirect(dir);

                    // Sample HDR image
                    let color = self.sample(u, v);

                    // Convert to f16 and store as RGBA
                    let idx = ((y * face_size + x) * 8) as usize;
                    let r = half::f16::from_f32(color[0]);
                    let g = half::f16::from_f32(color[1]);
                    let b = half::f16::from_f32(color[2]);
                    let a = half::f16::from_f32(1.0);

                    data[idx..idx + 2].copy_from_slice(&r.to_le_bytes());
                    data[idx + 2..idx + 4].copy_from_slice(&g.to_le_bytes());
                    data[idx + 4..idx + 6].copy_from_slice(&b.to_le_bytes());
                    data[idx + 6..idx + 8].copy_from_slice(&a.to_le_bytes());
                }
            }

            faces[face] = data;
        }

        faces
    }

    /// Convert equirectangular HDR panorama to cubemap faces with RGBA8 output.
    ///
    /// Applies tonemapping for LDR output.
    /// Returns 6 faces in order: +X, -X, +Y, -Y, +Z, -Z
    pub fn to_cubemap_faces_ldr(&self, face_size: u32, exposure: f32) -> [Vec<u8>; 6] {
        let mut faces: [Vec<u8>; 6] = Default::default();

        for face in 0..6 {
            let mut data = vec![0u8; (face_size * face_size * 4) as usize];

            for y in 0..face_size {
                for x in 0..face_size {
                    let dir = texel_to_direction(face, x, y, face_size);
                    let dir = normalize(dir);
                    let (u, v) = direction_to_equirect(dir);
                    let color = self.sample(u, v);

                    // Apply exposure and ACES tonemapping
                    let exposed = [
                        color[0] * exposure,
                        color[1] * exposure,
                        color[2] * exposure,
                    ];
                    let tonemapped = aces_tonemap(exposed);

                    let idx = ((y * face_size + x) * 4) as usize;
                    data[idx] = (tonemapped[0].clamp(0.0, 1.0) * 255.0) as u8;
                    data[idx + 1] = (tonemapped[1].clamp(0.0, 1.0) * 255.0) as u8;
                    data[idx + 2] = (tonemapped[2].clamp(0.0, 1.0) * 255.0) as u8;
                    data[idx + 3] = 255;
                }
            }

            faces[face] = data;
        }

        faces
    }
}

/// Errors that can occur when loading HDR images.
#[derive(Debug)]
pub enum HdrError {
    /// IO error during loading.
    IoError(String),
    /// Error decoding the image format.
    DecodeError(String),
}

impl std::fmt::Display for HdrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HdrError::IoError(e) => write!(f, "IO error: {}", e),
            HdrError::DecodeError(e) => write!(f, "Decode error: {}", e),
        }
    }
}

impl std::error::Error for HdrError {}

// ============ Helper functions ============

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Convert cubemap face texel to world direction.
fn texel_to_direction(face: usize, x: u32, y: u32, size: u32) -> [f32; 3] {
    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

    match face {
        0 => [1.0, -v, -u],  // +X
        1 => [-1.0, -v, u],  // -X
        2 => [u, 1.0, v],    // +Y
        3 => [u, -1.0, -v],  // -Y
        4 => [u, -v, 1.0],   // +Z
        5 => [-u, -v, -1.0], // -Z
        _ => [0.0, 0.0, 1.0],
    }
}

/// Convert 3D direction to equirectangular UV coordinates.
fn direction_to_equirect(dir: [f32; 3]) -> (f32, f32) {
    let theta = dir[0].atan2(dir[2]); // Azimuth: -PI to PI
    let phi = dir[1].asin(); // Elevation: -PI/2 to PI/2

    let u = (theta + PI) / (2.0 * PI); // 0 to 1
    let v = (phi + PI / 2.0) / PI; // 0 to 1

    (u, 1.0 - v) // Flip V for typical image orientation
}

/// ACES filmic tonemapping.
fn aces_tonemap(color: [f32; 3]) -> [f32; 3] {
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;

    [
        (color[0] * (A * color[0] + B)) / (color[0] * (C * color[0] + D) + E),
        (color[1] * (A * color[1] + B)) / (color[1] * (C * color[1] + D) + E),
        (color[2] * (A * color[2] + B)) / (color[2] * (C * color[2] + D) + E),
    ]
}

/// Create a wgpu cubemap texture from HDR image data.
pub fn create_hdr_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    hdr: &HdrImage,
    face_size: u32,
    generate_mipmaps: bool,
) -> wgpu::Texture {
    let mip_levels = if generate_mipmaps {
        (face_size as f32).log2().floor() as u32 + 1
    } else {
        1
    };

    let faces = hdr.to_cubemap_faces_ldr(face_size, 1.0);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("HDR Cubemap"),
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Upload base mip level
    for (face, data) in faces.iter().enumerate() {
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
                bytes_per_row: Some(face_size * 4),
                rows_per_image: Some(face_size),
            },
            wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 1,
            },
        );
    }

    texture
}

/// Create a prefiltered HDR cubemap for IBL.
///
/// Generates mip levels with increasing roughness for specular reflections.
pub fn create_prefiltered_hdr_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    hdr: &HdrImage,
    face_size: u32,
    mip_levels: u32,
) -> wgpu::Texture {
    use crate::ibl::PrefilterGenerator;

    // First convert to LDR cubemap faces
    let source_faces = hdr.to_cubemap_faces_ldr(face_size, 1.0);

    // Prefilter for IBL
    let generator = PrefilterGenerator::new().with_samples(64);
    let mip_data = generator.generate_prefiltered_cubemap(&source_faces, face_size, mip_levels);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Prefiltered HDR Cubemap"),
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
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
        let mip_size = (face_size >> mip).max(1);

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
