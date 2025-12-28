//! Shadow configuration types.

/// Shadow quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShadowQuality {
    /// 512x512 resolution.
    Low,
    /// 1024x1024 resolution.
    Medium,
    /// 2048x2048 resolution (default).
    #[default]
    High,
    /// 4096x4096 resolution.
    Ultra,
}

impl ShadowQuality {
    /// Get the shadow map resolution for this quality level.
    pub fn resolution(&self) -> u32 {
        match self {
            Self::Low => 512,
            Self::Medium => 1024,
            Self::High => 2048,
            Self::Ultra => 4096,
        }
    }

    /// Create from resolution value.
    pub fn from_resolution(resolution: u32) -> Self {
        match resolution {
            0..=512 => Self::Low,
            513..=1024 => Self::Medium,
            1025..=2048 => Self::High,
            _ => Self::Ultra,
        }
    }
}

/// PCF (Percentage Closer Filtering) mode for soft shadows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum PCFMode {
    /// No filtering (hard shadows).
    None = 0,
    /// Hardware 2x2 bilinear PCF.
    Hardware2x2 = 1,
    /// 3x3 software PCF (9 samples).
    #[default]
    Soft3x3 = 2,
    /// 5x5 software PCF (25 samples).
    Soft5x5 = 3,
    /// Poisson disk sampling (16 samples).
    PoissonDisk = 4,
    /// PCSS (Percentage-Closer Soft Shadows) with variable penumbra.
    PCSS = 5,
}

impl PCFMode {
    /// Get the number of samples for this PCF mode.
    pub fn sample_count(&self) -> u32 {
        match self {
            Self::None => 1,
            Self::Hardware2x2 => 4,
            Self::Soft3x3 => 9,
            Self::Soft5x5 => 25,
            Self::PoissonDisk => 16,
            Self::PCSS => 32, // 16 blocker search + 16 PCF
        }
    }
}

/// PCSS (Percentage-Closer Soft Shadows) configuration.
#[derive(Debug, Clone, Copy)]
pub struct PCSSConfig {
    /// Light size in world units. Larger = softer shadows.
    pub light_size: f32,
    /// Number of samples for blocker search (should be power of 2).
    pub blocker_search_samples: u32,
    /// Number of samples for PCF filtering (should be power of 2).
    pub pcf_samples: u32,
    /// Maximum filter radius in texels.
    pub max_filter_radius: f32,
    /// Near plane for penumbra calculation.
    pub near_plane: f32,
}

impl Default for PCSSConfig {
    fn default() -> Self {
        Self {
            light_size: 0.5,
            blocker_search_samples: 16,
            pcf_samples: 16,
            max_filter_radius: 10.0,
            near_plane: 0.1,
        }
    }
}

impl PCSSConfig {
    /// Create a new PCSS configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the light size (larger = softer shadows).
    pub fn light_size(mut self, size: f32) -> Self {
        self.light_size = size.max(0.01);
        self
    }

    /// Set the number of blocker search samples.
    pub fn blocker_search_samples(mut self, samples: u32) -> Self {
        self.blocker_search_samples = samples.clamp(4, 64);
        self
    }

    /// Set the number of PCF samples.
    pub fn pcf_samples(mut self, samples: u32) -> Self {
        self.pcf_samples = samples.clamp(4, 64);
        self
    }

    /// Set the maximum filter radius in texels.
    pub fn max_filter_radius(mut self, radius: f32) -> Self {
        self.max_filter_radius = radius.clamp(1.0, 50.0);
        self
    }

    /// Set the near plane for penumbra calculation.
    pub fn near_plane(mut self, near: f32) -> Self {
        self.near_plane = near.max(0.001);
        self
    }
}

/// Contact shadow configuration for screen-space contact shadows.
#[derive(Debug, Clone, Copy)]
pub struct ContactShadowConfig {
    /// Whether contact shadows are enabled.
    pub enabled: bool,
    /// Maximum ray distance in world units.
    pub max_distance: f32,
    /// Thickness of objects for ray marching.
    pub thickness: f32,
    /// Number of ray march steps (quality vs performance).
    pub steps: u32,
    /// Intensity of contact shadows (0.0 - 1.0).
    pub intensity: f32,
}

impl Default for ContactShadowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // More aggressive defaults to hide shadow map gaps at object bases
            max_distance: 1.0,    // Larger range for better coverage
            thickness: 0.1,       // Slightly thicker for reliable occlusion
            steps: 16,            // More steps for quality (matches shader)
            intensity: 0.8,       // Stronger shadows
        }
    }
}

impl ContactShadowConfig {
    /// Create a new contact shadow configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable contact shadows.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set maximum ray distance in world units.
    pub fn max_distance(mut self, distance: f32) -> Self {
        self.max_distance = distance.clamp(0.1, 5.0);  // Allow larger range
        self
    }

    /// Set object thickness for ray marching.
    pub fn thickness(mut self, thickness: f32) -> Self {
        self.thickness = thickness.clamp(0.01, 1.0);  // Allow thicker for aggressive coverage
        self
    }

    /// Set number of ray march steps.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps.clamp(4, 32);  // Note: shader uses fixed 16 steps
        self
    }

    /// Set shadow intensity.
    pub fn intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }
}

/// Configuration for shadow mapping.
#[derive(Debug, Clone)]
pub struct ShadowConfig {
    /// Shadow map resolution.
    pub resolution: u32,
    /// PCF filtering mode.
    pub pcf_mode: PCFMode,
    /// Depth bias to prevent shadow acne.
    pub bias: f32,
    /// Normal offset bias.
    pub normal_bias: f32,
    /// Maximum shadow render distance.
    pub max_distance: f32,
    /// Distance at which shadows start fading (as fraction of max_distance).
    pub fade_start: f32,
    /// Whether shadows are enabled.
    pub enabled: bool,
    /// PCSS configuration (used when pcf_mode is PCSS).
    pub pcss: PCSSConfig,
    /// Contact shadow configuration.
    pub contact: ContactShadowConfig,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            resolution: 2048,
            pcf_mode: PCFMode::Soft3x3,
            bias: 0.005,
            normal_bias: 0.02,
            max_distance: 100.0,
            fade_start: 0.9,
            enabled: true,
            pcss: PCSSConfig::default(),
            contact: ContactShadowConfig::default(),
        }
    }
}

impl ShadowConfig {
    /// Create a new shadow configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration with a quality preset.
    pub fn with_quality(quality: ShadowQuality) -> Self {
        Self {
            resolution: quality.resolution(),
            ..Default::default()
        }
    }

    /// Set the shadow map resolution.
    pub fn resolution(mut self, resolution: u32) -> Self {
        self.resolution = resolution;
        self
    }

    /// Set the PCF mode.
    pub fn pcf_mode(mut self, mode: PCFMode) -> Self {
        self.pcf_mode = mode;
        self
    }

    /// Set the depth bias.
    pub fn bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }

    /// Set the normal bias.
    pub fn normal_bias(mut self, normal_bias: f32) -> Self {
        self.normal_bias = normal_bias;
        self
    }

    /// Set the maximum shadow distance.
    pub fn max_distance(mut self, max_distance: f32) -> Self {
        self.max_distance = max_distance;
        self
    }

    /// Set whether shadows are enabled.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the PCSS configuration.
    pub fn pcss_config(mut self, config: PCSSConfig) -> Self {
        self.pcss = config;
        self
    }

    /// Enable PCSS with default settings.
    pub fn with_pcss(mut self) -> Self {
        self.pcf_mode = PCFMode::PCSS;
        self
    }

    /// Enable PCSS with custom light size.
    pub fn with_pcss_light_size(mut self, light_size: f32) -> Self {
        self.pcf_mode = PCFMode::PCSS;
        self.pcss.light_size = light_size;
        self
    }

    /// Set contact shadow configuration.
    pub fn contact_config(mut self, config: ContactShadowConfig) -> Self {
        self.contact = config;
        self
    }

    /// Enable contact shadows with default settings.
    pub fn with_contact_shadows(mut self) -> Self {
        self.contact.enabled = true;
        self
    }

    /// Enable contact shadows with custom intensity.
    pub fn with_contact_shadows_intensity(mut self, intensity: f32) -> Self {
        self.contact.enabled = true;
        self.contact.intensity = intensity.clamp(0.0, 1.0);
        self
    }
}

/// Configuration for cascaded shadow maps.
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    /// Number of cascades (1-4).
    pub num_cascades: u32,
    /// Split scheme blend factor (0 = uniform, 1 = logarithmic).
    pub split_lambda: f32,
    /// Maximum shadow distance for cascades.
    pub max_distance: f32,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            num_cascades: 4,
            split_lambda: 0.5,
            max_distance: 100.0,
        }
    }
}

impl CascadeConfig {
    /// Create a new cascade configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of cascades.
    pub fn num_cascades(mut self, num: u32) -> Self {
        self.num_cascades = num.clamp(1, 4);
        self
    }

    /// Set the split lambda (0 = uniform, 1 = logarithmic).
    pub fn split_lambda(mut self, lambda: f32) -> Self {
        self.split_lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum shadow distance.
    pub fn max_distance(mut self, distance: f32) -> Self {
        self.max_distance = distance;
        self
    }
}
