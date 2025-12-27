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
        }
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
