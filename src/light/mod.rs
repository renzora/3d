//! Lighting module for scene illumination.

mod ambient;
mod directional;
mod hemisphere;
mod point;
mod spot;

pub use ambient::AmbientLight;
pub use directional::DirectionalLight;
pub use hemisphere::HemisphereLight;
pub use point::PointLight;
pub use spot::SpotLight;

use crate::math::Color;
use bytemuck::{Pod, Zeroable};

/// Maximum number of lights supported in a single render pass.
pub const MAX_LIGHTS: usize = 16;

/// Light type identifier for GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LightType {
    /// Point light (omni-directional).
    Point = 0,
    /// Directional light (sun-like).
    Directional = 1,
    /// Spot light (cone-shaped).
    Spot = 2,
}

/// GPU-friendly light data structure.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LightUniform {
    /// Light position (for point/spot) or direction (for directional).
    pub position: [f32; 3],
    /// Light type (0=point, 1=directional, 2=spot).
    pub light_type: u32,
    /// Light color.
    pub color: [f32; 3],
    /// Light intensity.
    pub intensity: f32,
    /// Direction (for spot/directional lights).
    pub direction: [f32; 3],
    /// Range/radius (0 = infinite for directional).
    pub range: f32,
    /// Inner cone angle cosine (spot light).
    pub inner_cone_cos: f32,
    /// Outer cone angle cosine (spot light).
    pub outer_cone_cos: f32,
    /// Padding for alignment.
    pub _padding: [f32; 2],
}

impl Default for LightUniform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            light_type: 0,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            direction: [0.0, -1.0, 0.0],
            range: 10.0,
            inner_cone_cos: 0.9,  // ~25 degrees
            outer_cone_cos: 0.8,  // ~36 degrees
            _padding: [0.0; 2],
        }
    }
}

/// GPU-friendly hemisphere light data.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct HemisphereLightUniform {
    /// Sky color (RGB) + enabled flag (W: 1.0 = enabled).
    pub sky: [f32; 4],
    /// Ground color (RGB) + intensity (W).
    pub ground: [f32; 4],
}

impl Default for HemisphereLightUniform {
    fn default() -> Self {
        Self {
            sky: [0.6, 0.75, 1.0, 0.0],    // Light blue, disabled
            ground: [0.4, 0.3, 0.2, 1.0],   // Brown, intensity 1.0
        }
    }
}

/// Lights uniform buffer containing all scene lights.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LightsUniform {
    /// Ambient light color and intensity (RGB + intensity in alpha).
    pub ambient: [f32; 4],
    /// Hemisphere light sky color (RGB) + enabled flag (W).
    pub hemisphere_sky: [f32; 4],
    /// Hemisphere light ground color (RGB) + intensity (W).
    pub hemisphere_ground: [f32; 4],
    /// Number of active lights.
    pub num_lights: u32,
    /// Padding (using u32 to match WGSL alignment).
    pub _padding: [u32; 3],
    /// Array of lights.
    pub lights: [LightUniform; MAX_LIGHTS],
}

impl Default for LightsUniform {
    fn default() -> Self {
        Self {
            ambient: [0.03, 0.03, 0.03, 1.0],
            hemisphere_sky: [0.6, 0.75, 1.0, 0.0],      // Disabled by default
            hemisphere_ground: [0.4, 0.3, 0.2, 1.0],
            num_lights: 0,
            _padding: [0; 3],
            lights: [LightUniform::default(); MAX_LIGHTS],
        }
    }
}

impl LightsUniform {
    /// Create a new lights uniform with ambient light.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set ambient light.
    pub fn set_ambient(&mut self, color: Color, intensity: f32) {
        self.ambient = [color.r * intensity, color.g * intensity, color.b * intensity, intensity];
    }

    /// Set hemisphere light.
    pub fn set_hemisphere(&mut self, sky_color: Color, ground_color: Color, intensity: f32, enabled: bool) {
        self.hemisphere_sky = [
            sky_color.r,
            sky_color.g,
            sky_color.b,
            if enabled { 1.0 } else { 0.0 },
        ];
        self.hemisphere_ground = [
            ground_color.r,
            ground_color.g,
            ground_color.b,
            intensity,
        ];
    }

    /// Enable or disable hemisphere light.
    pub fn set_hemisphere_enabled(&mut self, enabled: bool) {
        self.hemisphere_sky[3] = if enabled { 1.0 } else { 0.0 };
    }

    /// Add a light and return its index.
    pub fn add_light(&mut self, light: LightUniform) -> Option<usize> {
        if (self.num_lights as usize) < MAX_LIGHTS {
            let index = self.num_lights as usize;
            self.lights[index] = light;
            self.num_lights += 1;
            Some(index)
        } else {
            None
        }
    }

    /// Clear all lights (keeps ambient).
    pub fn clear_lights(&mut self) {
        self.num_lights = 0;
    }
}

/// Common light trait.
pub trait Light {
    /// Get the light as a GPU uniform.
    fn to_uniform(&self) -> LightUniform;
}
