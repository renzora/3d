//! Cube texture (cubemap) implementation for skyboxes and environment maps.

use crate::core::Id;
use wgpu::util::DeviceExt;

/// Face order for cube maps: +X, -X, +Y, -Y, +Z, -Z
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeFace {
    /// Positive X (+X, right)
    PositiveX = 0,
    /// Negative X (-X, left)
    NegativeX = 1,
    /// Positive Y (+Y, top)
    PositiveY = 2,
    /// Negative Y (-Y, bottom)
    NegativeY = 3,
    /// Positive Z (+Z, front)
    PositiveZ = 4,
    /// Negative Z (-Z, back)
    NegativeZ = 5,
}

/// A cube texture (cubemap) for skyboxes and environment mapping.
pub struct CubeTexture {
    /// Unique ID.
    id: Id,
    /// Texture size (width = height for each face).
    size: u32,
    /// The GPU texture.
    texture: wgpu::Texture,
    /// Cube view for sampling.
    view: wgpu::TextureView,
    /// Texture format.
    format: wgpu::TextureFormat,
}

impl CubeTexture {
    /// Create a cube texture from 6 RGBA8 face images.
    /// Faces must be in order: +X, -X, +Y, -Y, +Z, -Z
    /// Each face must be `size x size` pixels with 4 bytes per pixel (RGBA).
    pub fn from_faces(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        faces: [&[u8]; 6],
        size: u32,
        label: Option<&str>,
    ) -> Self {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;

        // Combine all 6 faces into one contiguous buffer
        let mut all_data = Vec::with_capacity((size * size * 4 * 6) as usize);
        for face in faces {
            all_data.extend_from_slice(face);
        }

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label,
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 6,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &all_data,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Cube Texture View"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        Self {
            id: Id::new(),
            size,
            texture,
            view,
            format,
        }
    }

    /// Create a cube texture from 6 encoded image files (PNG, JPEG, etc.).
    /// Faces must be in order: +X, -X, +Y, -Y, +Z, -Z
    pub fn from_face_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        faces: [&[u8]; 6],
        label: Option<&str>,
    ) -> Result<Self, String> {
        use image::GenericImageView;

        let mut decoded_faces: [Vec<u8>; 6] = Default::default();
        let mut size = 0u32;

        for (i, face_data) in faces.iter().enumerate() {
            let img = image::load_from_memory(face_data)
                .map_err(|e| format!("Failed to decode face {}: {}", i, e))?;

            let (w, h) = img.dimensions();
            if w != h {
                return Err(format!("Face {} is not square: {}x{}", i, w, h));
            }

            if i == 0 {
                size = w;
            } else if w != size {
                return Err(format!(
                    "Face {} has different size than face 0: {} vs {}",
                    i, w, size
                ));
            }

            decoded_faces[i] = img.to_rgba8().into_raw();
        }

        let face_refs: [&[u8]; 6] = [
            &decoded_faces[0],
            &decoded_faces[1],
            &decoded_faces[2],
            &decoded_faces[3],
            &decoded_faces[4],
            &decoded_faces[5],
        ];

        Ok(Self::from_faces(device, queue, face_refs, size, label))
    }

    /// Create a simple procedural sky gradient cubemap.
    /// Creates a sky-to-ground gradient for basic skybox rendering.
    pub fn procedural_sky(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        sky_color: [u8; 3],
        horizon_color: [u8; 3],
        ground_color: [u8; 3],
    ) -> Self {
        let mut faces: [Vec<u8>; 6] = Default::default();
        let face_size = (size * size * 4) as usize;

        for face_idx in 0..6 {
            let mut data = Vec::with_capacity(face_size);

            for y in 0..size {
                for x in 0..size {
                    // Calculate direction vector for this texel
                    let (_dx, dy, _dz) = Self::texel_to_direction(face_idx, x, y, size);

                    // Blend based on Y component (up direction)
                    let t = dy; // -1 (ground) to +1 (sky)

                    let (r, g, b) = if t > 0.0 {
                        // Sky to horizon
                        let blend = t;
                        (
                            Self::lerp_u8(horizon_color[0], sky_color[0], blend),
                            Self::lerp_u8(horizon_color[1], sky_color[1], blend),
                            Self::lerp_u8(horizon_color[2], sky_color[2], blend),
                        )
                    } else {
                        // Horizon to ground
                        let blend = -t;
                        (
                            Self::lerp_u8(horizon_color[0], ground_color[0], blend),
                            Self::lerp_u8(horizon_color[1], ground_color[1], blend),
                            Self::lerp_u8(horizon_color[2], ground_color[2], blend),
                        )
                    };

                    data.extend_from_slice(&[r, g, b, 255]);
                }
            }

            faces[face_idx] = data;
        }

        let face_refs: [&[u8]; 6] = [
            &faces[0],
            &faces[1],
            &faces[2],
            &faces[3],
            &faces[4],
            &faces[5],
        ];

        Self::from_faces(device, queue, face_refs, size, Some("Procedural Sky Cubemap"))
    }

    /// Create a default sky cubemap (blue sky, white horizon, dark ground).
    pub fn default_sky(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::procedural_sky(
            device,
            queue,
            64,
            [100, 150, 230], // Sky blue
            [200, 210, 220], // Light horizon
            [50, 45, 40],    // Dark ground
        )
    }

    /// Convert texel coordinates to a direction vector.
    fn texel_to_direction(face: usize, x: u32, y: u32, size: u32) -> (f32, f32, f32) {
        // Map x, y to [-1, 1]
        let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
        let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

        let (dx, dy, dz) = match face {
            0 => (1.0, -v, -u),  // +X
            1 => (-1.0, -v, u),  // -X
            2 => (u, 1.0, v),    // +Y
            3 => (u, -1.0, -v),  // -Y
            4 => (u, -v, 1.0),   // +Z
            5 => (-u, -v, -1.0), // -Z
            _ => (0.0, 0.0, 0.0),
        };

        // Normalize
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        (dx / len, dy / len, dz / len)
    }

    fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
        let t = t.clamp(0.0, 1.0);
        ((a as f32) * (1.0 - t) + (b as f32) * t) as u8
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get texture size (each face is size x size).
    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Get the texture format.
    #[inline]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Get the underlying wgpu texture.
    #[inline]
    pub fn wgpu_texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Get the cube texture view for sampling.
    #[inline]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}
