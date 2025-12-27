//! Texture sampler configuration.

/// Texture addressing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressMode {
    /// Clamp to edge pixel.
    #[default]
    ClampToEdge,
    /// Repeat the texture.
    Repeat,
    /// Mirror and repeat.
    MirrorRepeat,
}

impl From<AddressMode> for wgpu::AddressMode {
    fn from(mode: AddressMode) -> Self {
        match mode {
            AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            AddressMode::Repeat => wgpu::AddressMode::Repeat,
            AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        }
    }
}

/// Texture filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// Nearest neighbor (pixelated).
    Nearest,
    /// Linear interpolation (smooth).
    #[default]
    Linear,
}

impl From<FilterMode> for wgpu::FilterMode {
    fn from(mode: FilterMode) -> Self {
        match mode {
            FilterMode::Nearest => wgpu::FilterMode::Nearest,
            FilterMode::Linear => wgpu::FilterMode::Linear,
        }
    }
}

/// Sampler configuration descriptor.
#[derive(Debug, Clone, Copy)]
pub struct SamplerDescriptor {
    /// Address mode for U coordinate.
    pub address_mode_u: AddressMode,
    /// Address mode for V coordinate.
    pub address_mode_v: AddressMode,
    /// Address mode for W coordinate (3D textures).
    pub address_mode_w: AddressMode,
    /// Magnification filter.
    pub mag_filter: FilterMode,
    /// Minification filter.
    pub min_filter: FilterMode,
    /// Mipmap filter.
    pub mipmap_filter: FilterMode,
    /// Anisotropic filtering level (1 = disabled).
    pub anisotropy_clamp: u16,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            anisotropy_clamp: 1,
        }
    }
}

impl SamplerDescriptor {
    /// Create a repeating sampler.
    pub fn repeating() -> Self {
        Self {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            ..Default::default()
        }
    }

    /// Create a nearest-neighbor (pixelated) sampler.
    pub fn nearest() -> Self {
        Self {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        }
    }
}

/// A GPU texture sampler.
pub struct Sampler {
    /// The wgpu sampler.
    sampler: wgpu::Sampler,
}

impl Sampler {
    /// Create a new sampler from a descriptor.
    pub fn new(device: &wgpu::Device, desc: &SamplerDescriptor) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: desc.address_mode_u.into(),
            address_mode_v: desc.address_mode_v.into(),
            address_mode_w: desc.address_mode_w.into(),
            mag_filter: desc.mag_filter.into(),
            min_filter: desc.min_filter.into(),
            mipmap_filter: desc.mipmap_filter.into(),
            anisotropy_clamp: desc.anisotropy_clamp,
            ..Default::default()
        });

        Self { sampler }
    }

    /// Create a default linear sampler with repeat wrapping.
    pub fn linear(device: &wgpu::Device) -> Self {
        Self::new(device, &SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            ..SamplerDescriptor::default()
        })
    }

    /// Create a nearest-neighbor sampler.
    pub fn nearest(device: &wgpu::Device) -> Self {
        Self::new(device, &SamplerDescriptor::nearest())
    }

    /// Get the underlying wgpu sampler.
    #[inline]
    pub fn wgpu_sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
}
