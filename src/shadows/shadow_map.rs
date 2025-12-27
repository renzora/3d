//! Shadow map texture management.

use bytemuck::{Pod, Zeroable};

use super::{ShadowConfig, MAX_CASCADES, MAX_SHADOW_LIGHTS};

/// A single shadow map texture.
pub struct ShadowMap {
    /// Depth texture for shadow map.
    texture: wgpu::Texture,
    /// Texture view for rendering to.
    view: wgpu::TextureView,
    /// Resolution.
    resolution: u32,
    /// Light-space view-projection matrix.
    light_matrix: [[f32; 4]; 4],
}

impl ShadowMap {
    /// Create a new shadow map with the given resolution.
    pub fn new(device: &wgpu::Device, resolution: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            resolution,
            light_matrix: [[0.0; 4]; 4],
        }
    }

    /// Get the texture view for rendering to this shadow map.
    #[inline]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get the texture for binding.
    #[inline]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Get the resolution.
    #[inline]
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Set the light-space matrix.
    pub fn set_light_matrix(&mut self, matrix: [[f32; 4]; 4]) {
        self.light_matrix = matrix;
    }

    /// Get the light-space matrix.
    #[inline]
    pub fn light_matrix(&self) -> &[[f32; 4]; 4] {
        &self.light_matrix
    }

    /// Resize the shadow map.
    pub fn resize(&mut self, device: &wgpu::Device, resolution: u32) {
        if self.resolution != resolution {
            *self = Self::new(device, resolution);
        }
    }
}

/// GPU-friendly shadow uniform data.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ShadowUniform {
    /// Light-space matrices for each shadow-casting light.
    pub light_matrices: [[[f32; 4]; 4]; MAX_SHADOW_LIGHTS],
    /// Cascade matrices for directional light CSM.
    pub cascade_matrices: [[[f32; 4]; 4]; MAX_CASCADES],
    /// Cascade split distances.
    pub cascade_splits: [f32; 4],
    /// Per-light shadow config: x=bias, y=normal_bias, z=map_index, w=light_type.
    pub shadow_config: [[f32; 4]; MAX_SHADOW_LIGHTS],
    /// Global config: x=num_shadow_lights, y=pcf_mode, z=map_size, w=num_cascades.
    pub global_config: [f32; 4],
}

impl Default for ShadowUniform {
    fn default() -> Self {
        Self {
            light_matrices: [[[0.0; 4]; 4]; MAX_SHADOW_LIGHTS],
            cascade_matrices: [[[0.0; 4]; 4]; MAX_CASCADES],
            cascade_splits: [0.0; 4],
            shadow_config: [[-1.0, 0.0, -1.0, 0.0]; MAX_SHADOW_LIGHTS],
            global_config: [0.0, 2.0, 2048.0, 4.0], // 0 lights, Soft3x3, 2048 resolution, 4 cascades
        }
    }
}

impl ShadowUniform {
    /// Create a new shadow uniform.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of shadow-casting lights.
    pub fn set_num_shadow_lights(&mut self, count: u32) {
        self.global_config[0] = count as f32;
    }

    /// Set the PCF mode.
    pub fn set_pcf_mode(&mut self, mode: super::PCFMode) {
        self.global_config[1] = mode as u32 as f32;
    }

    /// Set the shadow map size.
    pub fn set_map_size(&mut self, size: u32) {
        self.global_config[2] = size as f32;
    }

    /// Set the number of cascades.
    pub fn set_num_cascades(&mut self, count: u32) {
        self.global_config[3] = count as f32;
    }

    /// Set the light matrix for a shadow-casting light.
    pub fn set_light_matrix(&mut self, index: usize, matrix: [[f32; 4]; 4]) {
        if index < MAX_SHADOW_LIGHTS {
            self.light_matrices[index] = matrix;
        }
    }

    /// Set the cascade matrix for a cascade level.
    pub fn set_cascade_matrix(&mut self, index: usize, matrix: [[f32; 4]; 4]) {
        if index < MAX_CASCADES {
            self.cascade_matrices[index] = matrix;
        }
    }

    /// Set the cascade split distances.
    pub fn set_cascade_splits(&mut self, splits: [f32; 4]) {
        self.cascade_splits = splits;
    }

    /// Set shadow config for a light.
    /// map_index of -1 means no shadow.
    pub fn set_shadow_config(
        &mut self,
        light_index: usize,
        bias: f32,
        normal_bias: f32,
        map_index: i32,
        light_type: u32,
    ) {
        if light_index < MAX_SHADOW_LIGHTS {
            self.shadow_config[light_index] = [bias, normal_bias, map_index as f32, light_type as f32];
        }
    }

    /// Clear all shadow data.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Shadow atlas managing multiple shadow maps and bind groups.
pub struct ShadowAtlas {
    /// Individual shadow maps for each shadow-casting light.
    shadow_maps: Vec<ShadowMap>,
    /// Cascade shadow maps for directional light.
    cascade_maps: Vec<ShadowMap>,
    /// Comparison sampler for shadow mapping.
    sampler: wgpu::Sampler,
    /// Shadow uniform buffer.
    uniform_buffer: wgpu::Buffer,
    /// Shadow uniform data.
    uniform_data: ShadowUniform,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group.
    bind_group: Option<wgpu::BindGroup>,
    /// Current resolution.
    resolution: u32,
}

impl ShadowAtlas {
    /// Create a new shadow atlas.
    pub fn new(device: &wgpu::Device, config: &ShadowConfig) -> Self {
        let resolution = config.resolution;

        // Create shadow maps for point/spot lights
        let shadow_maps: Vec<ShadowMap> = (0..MAX_SHADOW_LIGHTS)
            .map(|_| ShadowMap::new(device, resolution))
            .collect();

        // Create cascade shadow maps for directional light
        let cascade_maps: Vec<ShadowMap> = (0..MAX_CASCADES)
            .map(|_| ShadowMap::new(device, resolution))
            .collect();

        // Comparison sampler for PCF
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Create bind group layout
        let bind_group_layout = Self::create_bind_group_layout(device);

        // Create uniform buffer
        let uniform_data = ShadowUniform::default();
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Uniform Buffer"),
            size: std::mem::size_of::<ShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut atlas = Self {
            shadow_maps,
            cascade_maps,
            sampler,
            uniform_buffer,
            uniform_data,
            bind_group_layout,
            bind_group: None,
            resolution,
        };

        atlas.update_bind_group(device);
        atlas
    }

    /// Create the bind group layout for shadow sampling.
    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Bind Group Layout"),
            entries: &[
                // Shadow uniform data
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Shadow maps (array of 4 for point/spot)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: std::num::NonZeroU32::new(MAX_SHADOW_LIGHTS as u32),
                },
                // Cascade shadow maps (array of 4 for directional CSM)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: std::num::NonZeroU32::new(MAX_CASCADES as u32),
                },
                // Comparison sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        })
    }

    /// Update the bind group.
    pub fn update_bind_group(&mut self, device: &wgpu::Device) {
        let shadow_views: Vec<&wgpu::TextureView> =
            self.shadow_maps.iter().map(|m| m.view()).collect();
        let cascade_views: Vec<&wgpu::TextureView> =
            self.cascade_maps.iter().map(|m| m.view()).collect();

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureViewArray(&shadow_views),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureViewArray(&cascade_views),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
    }

    /// Write uniform data to the GPU buffer.
    pub fn update_uniform(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform_data));
    }

    /// Get the bind group layout.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the bind group.
    #[inline]
    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    /// Get a shadow map by index.
    #[inline]
    pub fn shadow_map(&self, index: usize) -> Option<&ShadowMap> {
        self.shadow_maps.get(index)
    }

    /// Get a mutable shadow map by index.
    #[inline]
    pub fn shadow_map_mut(&mut self, index: usize) -> Option<&mut ShadowMap> {
        self.shadow_maps.get_mut(index)
    }

    /// Get a cascade shadow map by index.
    #[inline]
    pub fn cascade_map(&self, index: usize) -> Option<&ShadowMap> {
        self.cascade_maps.get(index)
    }

    /// Get a mutable cascade shadow map by index.
    #[inline]
    pub fn cascade_map_mut(&mut self, index: usize) -> Option<&mut ShadowMap> {
        self.cascade_maps.get_mut(index)
    }

    /// Get the uniform data for modification.
    #[inline]
    pub fn uniform_data(&self) -> &ShadowUniform {
        &self.uniform_data
    }

    /// Get mutable uniform data.
    #[inline]
    pub fn uniform_data_mut(&mut self) -> &mut ShadowUniform {
        &mut self.uniform_data
    }

    /// Resize all shadow maps.
    pub fn resize(&mut self, device: &wgpu::Device, resolution: u32) {
        if self.resolution != resolution {
            self.resolution = resolution;
            for map in &mut self.shadow_maps {
                map.resize(device, resolution);
            }
            for map in &mut self.cascade_maps {
                map.resize(device, resolution);
            }
            self.uniform_data.set_map_size(resolution);
            self.update_bind_group(device);
        }
    }

    /// Get the current resolution.
    #[inline]
    pub fn resolution(&self) -> u32 {
        self.resolution
    }
}
