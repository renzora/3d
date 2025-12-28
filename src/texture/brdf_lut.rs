//! BRDF Look-Up Table (LUT) for IBL.
//!
//! Pre-integrates the BRDF for environment lighting using the split-sum approximation.
//! The LUT is indexed by NdotV (x) and roughness (y), returning (scale, bias) for F0.

use crate::core::Id;
use std::f32::consts::PI;

/// BRDF Look-Up Table texture for physically based rendering.
pub struct BrdfLut {
    /// Unique ID.
    id: Id,
    /// LUT resolution (width and height).
    resolution: u32,
    /// The GPU texture.
    texture: wgpu::Texture,
    /// Texture view.
    view: wgpu::TextureView,
    /// Sampler for the LUT.
    sampler: wgpu::Sampler,
}

impl BrdfLut {
    /// Default resolution for the BRDF LUT (128x128 for fast startup).
    pub const DEFAULT_RESOLUTION: u32 = 128;

    /// Generate a new BRDF LUT with the specified resolution.
    pub fn generate(device: &wgpu::Device, queue: &wgpu::Queue, resolution: u32) -> Self {
        let data = Self::compute_lut(resolution);
        Self::from_data(device, queue, &data, resolution)
    }

    /// Generate a BRDF LUT with default resolution.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::generate(device, queue, Self::DEFAULT_RESOLUTION)
    }

    /// Compute the BRDF LUT data on CPU.
    /// Returns RGBA8 data (scale in R, bias in G, unused in B and A).
    fn compute_lut(resolution: u32) -> Vec<u8> {
        // 32 samples for fast generation (still gives reasonable quality)
        let sample_count = 32u32;
        let mut data = Vec::with_capacity((resolution * resolution * 4) as usize);

        for y in 0..resolution {
            for x in 0..resolution {
                // Map to [0, 1] range
                let n_dot_v = (x as f32 + 0.5) / resolution as f32;
                let roughness = (y as f32 + 0.5) / resolution as f32;

                // Clamp NdotV to avoid division by zero
                let n_dot_v = n_dot_v.max(0.001);

                let (scale, bias) = Self::integrate_brdf(n_dot_v, roughness, sample_count);

                // Store as 8-bit values (scale and bias are in [0, 1] range)
                let scale_u8 = (scale.clamp(0.0, 1.0) * 255.0) as u8;
                let bias_u8 = (bias.clamp(0.0, 1.0) * 255.0) as u8;

                data.push(scale_u8);
                data.push(bias_u8);
                data.push(0); // Unused
                data.push(255); // Alpha = 1
            }
        }

        data
    }

    /// Integrate the BRDF for a given NdotV and roughness using importance sampling.
    fn integrate_brdf(n_dot_v: f32, roughness: f32, sample_count: u32) -> (f32, f32) {
        let v = glam::Vec3::new(
            (1.0 - n_dot_v * n_dot_v).sqrt(), // sin(theta)
            0.0,
            n_dot_v, // cos(theta)
        );
        let n = glam::Vec3::Z;

        let mut scale = 0.0f32;
        let mut bias = 0.0f32;

        let alpha = roughness * roughness;

        for i in 0..sample_count {
            // Hammersley sequence for quasi-random sampling
            let xi = Self::hammersley(i, sample_count);

            // Importance sample the GGX distribution
            let h = Self::importance_sample_ggx(xi, alpha);

            // Reflect view vector around half vector
            let l = 2.0 * v.dot(h) * h - v;

            let n_dot_l = l.z.max(0.0);
            let n_dot_h = h.z.max(0.0);
            let v_dot_h = v.dot(h).max(0.0);

            if n_dot_l > 0.0 {
                // Geometry term (Smith GGX)
                let g = Self::geometry_smith(n_dot_v, n_dot_l, alpha);

                // Visibility term
                let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v);

                // Fresnel term (Schlick)
                let fc = (1.0 - v_dot_h).powf(5.0);

                scale += (1.0 - fc) * g_vis;
                bias += fc * g_vis;
            }
        }

        let inv_samples = 1.0 / sample_count as f32;
        (scale * inv_samples, bias * inv_samples)
    }

    /// Hammersley sequence for quasi-random sampling.
    fn hammersley(i: u32, n: u32) -> glam::Vec2 {
        glam::Vec2::new(i as f32 / n as f32, Self::radical_inverse_vdc(i))
    }

    /// Van der Corput radical inverse.
    fn radical_inverse_vdc(mut bits: u32) -> f32 {
        bits = (bits << 16) | (bits >> 16);
        bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
        bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
        bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
        bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
        bits as f32 * 2.3283064365386963e-10 // 1.0 / 0x100000000
    }

    /// Importance sample the GGX distribution.
    fn importance_sample_ggx(xi: glam::Vec2, alpha: f32) -> glam::Vec3 {
        let a2 = alpha * alpha;

        let phi = 2.0 * PI * xi.x;
        let cos_theta = ((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y)).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        glam::Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta)
    }

    /// Smith's geometry function for GGX.
    fn geometry_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
        Self::geometry_schlick_ggx(n_dot_v, alpha) * Self::geometry_schlick_ggx(n_dot_l, alpha)
    }

    /// Schlick-GGX geometry function.
    fn geometry_schlick_ggx(n_dot: f32, alpha: f32) -> f32 {
        let k = alpha / 2.0; // For IBL, use alpha/2 instead of (alpha+1)^2/8
        n_dot / (n_dot * (1.0 - k) + k)
    }

    /// Create a BRDF LUT from pre-computed data.
    fn from_data(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8], resolution: u32) -> Self {
        use wgpu::util::DeviceExt;

        // Use RGBA8Unorm for maximum compatibility
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let size = wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("BRDF LUT"),
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("BRDF LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
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

    /// Get the LUT resolution.
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
