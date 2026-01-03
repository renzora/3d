//! Detail albedo map generator for micro-surface color variation.
//!
//! Generates a procedural noise-based grayscale texture that tiles seamlessly.
//! Used for adding fine color variation visible at close range via overlay blending.

use crate::core::Id;

/// Detail albedo map texture for micro-surface color variation.
pub struct DetailAlbedoMap {
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

impl DetailAlbedoMap {
    /// Default resolution for detail albedo map.
    pub const DEFAULT_RESOLUTION: u32 = 256;

    /// Generate a new detail albedo map with the specified resolution.
    pub fn generate(device: &wgpu::Device, queue: &wgpu::Queue, resolution: u32) -> Self {
        let data = Self::compute_detail_albedo(resolution);
        Self::from_data(device, queue, &data, resolution)
    }

    /// Generate a detail albedo map with default resolution.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::generate(device, queue, Self::DEFAULT_RESOLUTION)
    }

    /// Compute the detail albedo map data on CPU using multi-octave noise.
    /// Returns RGBA8 data with grayscale values centered around 0.5 (128).
    fn compute_detail_albedo(resolution: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((resolution * resolution * 4) as usize);

        // Generate noise values
        let mut values = vec![0.0f32; (resolution * resolution) as usize];

        // Multi-octave noise for organic variation
        for y in 0..resolution {
            for x in 0..resolution {
                let idx = (y * resolution + x) as usize;
                let fx = x as f32 / resolution as f32;
                let fy = y as f32 / resolution as f32;

                // Use multiple octaves of noise for natural look
                let mut value = 0.0f32;
                let mut amplitude = 1.0f32;
                let mut frequency = 4.0f32; // Base frequency for tiling

                for _ in 0..4 {
                    // Tileable noise using sine waves
                    value += amplitude * Self::tileable_noise(fx, fy, frequency);
                    amplitude *= 0.5;
                    frequency *= 2.0;
                }

                values[idx] = value;
            }
        }

        // Normalize to [-1, 1] range, then shift to [0, 1]
        let min_v = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_v - min_v).max(0.001);

        for v in &mut values {
            // Normalize to [0, 1], with 0.5 being neutral (no change in overlay blend)
            *v = (*v - min_v) / range;
        }

        // Convert to RGBA8 grayscale
        for y in 0..resolution {
            for x in 0..resolution {
                let idx = (y * resolution + x) as usize;
                let value = values[idx];

                // Convert to 8-bit grayscale
                let gray = (value * 255.0).clamp(0.0, 255.0) as u8;

                // Store as grayscale in all channels (R=G=B for grayscale)
                data.push(gray);
                data.push(gray);
                data.push(gray);
                data.push(255);
            }
        }

        data
    }

    /// Tileable noise function using sine waves.
    /// Same algorithm as detail_normal for consistency.
    fn tileable_noise(x: f32, y: f32, frequency: f32) -> f32 {
        use std::f32::consts::PI;

        let fx = x * frequency * 2.0 * PI;
        let fy = y * frequency * 2.0 * PI;

        // Combine multiple sine waves at different angles for varied noise
        // Use slightly different coefficients than detail_normal for visual variety
        let n1 = (fx + 0.3).sin() * (fy + 0.7).cos();
        let n2 = (fx * 1.5 + 0.2).sin() * (fy * 1.4 + 0.6).cos();
        let n3 = (fx * 0.8 + fy * 1.2).sin();
        let n4 = (fx * 2.1 - fy * 0.8).cos();

        // Mix with golden ratio offsets for less regular patterns
        let phi = 1.618033988749895;
        let n5 = (fx * phi + 0.1).sin() * (fy * phi + 0.2).sin();
        let n6 = ((fx + fy) * phi * 0.6).cos();

        (n1 + n2 * 0.7 + n3 * 0.5 + n4 * 0.3 + n5 * 0.4 + n6 * 0.2) / 3.1
    }

    /// Create a detail albedo map from pre-computed data.
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
                label: Some("Detail Albedo Map"),
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
            label: Some("Detail Albedo Sampler"),
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
