//! Detail normal map generator for micro-surface detail.
//!
//! Generates a procedural noise-based normal map that tiles seamlessly.
//! Used for adding fine surface detail visible at close range.

use crate::core::Id;

/// Detail normal map texture for micro-surface imperfections.
pub struct DetailNormalMap {
    /// Unique ID.
    id: Id,
    /// Texture resolution (width and height).
    resolution: u32,
    /// The GPU texture.
    texture: wgpu::Texture,
    /// Texture view.
    view: wgpu::TextureView,
    /// Sampler for the detail map (with repeat addressing).
    sampler: wgpu::Sampler,
}

impl DetailNormalMap {
    /// Default resolution for detail normal map.
    pub const DEFAULT_RESOLUTION: u32 = 256;

    /// Generate a new detail normal map with the specified resolution.
    pub fn generate(device: &wgpu::Device, queue: &wgpu::Queue, resolution: u32) -> Self {
        let data = Self::compute_detail_normal(resolution);
        Self::from_data(device, queue, &data, resolution)
    }

    /// Generate a detail normal map with default resolution.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::generate(device, queue, Self::DEFAULT_RESOLUTION)
    }

    /// Compute the detail normal map data on CPU using multi-octave noise.
    /// Returns RGBA8 data (normal.x in R, normal.y in G, normal.z in B, A=255).
    fn compute_detail_normal(resolution: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((resolution * resolution * 4) as usize);

        // Generate height field first, then compute normals
        let mut heights = vec![0.0f32; (resolution * resolution) as usize];

        // Multi-octave noise for height field
        for y in 0..resolution {
            for x in 0..resolution {
                let idx = (y * resolution + x) as usize;
                let fx = x as f32 / resolution as f32;
                let fy = y as f32 / resolution as f32;

                // Use multiple octaves of noise for natural look
                let mut height = 0.0f32;
                let mut amplitude = 1.0f32;
                let mut frequency = 4.0f32; // Base frequency for tiling

                for _ in 0..4 {
                    // Tileable noise using sine waves
                    height += amplitude * Self::tileable_noise(fx, fy, frequency);
                    amplitude *= 0.5;
                    frequency *= 2.0;
                }

                heights[idx] = height;
            }
        }

        // Normalize heights to [0, 1]
        let min_h = heights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_h = heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_h - min_h).max(0.001);
        for h in &mut heights {
            *h = (*h - min_h) / range;
        }

        // Compute normals from height field using central differences
        let strength = 2.0; // Normal map strength

        for y in 0..resolution {
            for x in 0..resolution {
                // Sample neighboring heights (with wrapping for seamless tiling)
                let x_prev = if x == 0 { resolution - 1 } else { x - 1 };
                let x_next = if x == resolution - 1 { 0 } else { x + 1 };
                let y_prev = if y == 0 { resolution - 1 } else { y - 1 };
                let y_next = if y == resolution - 1 { 0 } else { y + 1 };

                let h_left = heights[(y * resolution + x_prev) as usize];
                let h_right = heights[(y * resolution + x_next) as usize];
                let h_down = heights[(y_prev * resolution + x) as usize];
                let h_up = heights[(y_next * resolution + x) as usize];

                // Central differences for gradient
                let dx = (h_right - h_left) * strength;
                let dy = (h_up - h_down) * strength;

                // Construct normal from gradient
                let normal = glam::Vec3::new(-dx, -dy, 1.0).normalize();

                // Convert from [-1, 1] to [0, 255]
                let r = ((normal.x * 0.5 + 0.5) * 255.0) as u8;
                let g = ((normal.y * 0.5 + 0.5) * 255.0) as u8;
                let b = ((normal.z * 0.5 + 0.5) * 255.0) as u8;

                data.push(r);
                data.push(g);
                data.push(b);
                data.push(255);
            }
        }

        data
    }

    /// Tileable noise function using sine waves
    fn tileable_noise(x: f32, y: f32, frequency: f32) -> f32 {
        use std::f32::consts::PI;

        let fx = x * frequency * 2.0 * PI;
        let fy = y * frequency * 2.0 * PI;

        // Combine multiple sine waves at different angles for varied noise
        let n1 = (fx).sin() * (fy).cos();
        let n2 = (fx * 1.7 + 0.5).sin() * (fy * 1.3 + 0.8).cos();
        let n3 = (fx * 0.7 + fy * 1.1).sin();
        let n4 = (fx * 2.3 - fy * 0.9).cos();

        // Mix with golden ratio offsets for less regular patterns
        let phi = 1.618033988749895;
        let n5 = (fx * phi).sin() * (fy * phi).sin();
        let n6 = ((fx + fy) * phi * 0.5).cos();

        (n1 + n2 * 0.7 + n3 * 0.5 + n4 * 0.3 + n5 * 0.4 + n6 * 0.2) / 3.1
    }

    /// Create a detail normal map from pre-computed data.
    fn from_data(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8], resolution: u32) -> Self {
        use wgpu::util::DeviceExt;

        let format = wgpu::TextureFormat::Rgba8Unorm;
        let size = wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Detail Normal Map"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Use Repeat addressing for seamless tiling
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Detail Normal Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            id: Id::new(),
            resolution,
            texture,
            view,
            sampler,
        }
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the texture resolution.
    #[inline]
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Get the texture view.
    #[inline]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get the sampler.
    #[inline]
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
}
