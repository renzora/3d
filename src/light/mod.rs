//! Lighting module for scene illumination.

mod ambient;
mod capsule;
mod directional;
mod disk;
mod hemisphere;
mod point;
mod rect;
mod sphere;
mod spot;

pub use ambient::AmbientLight;
pub use capsule::CapsuleLight;
pub use directional::DirectionalLight;
pub use disk::DiskLight;
pub use hemisphere::HemisphereLight;
pub use point::PointLight;
pub use rect::RectLight;
pub use sphere::SphereLight;
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
    /// Rect light (rectangular area light).
    Rect = 3,
    /// Capsule/Tube light (line-based area light).
    Capsule = 4,
    /// Disk light (circular area light).
    Disk = 5,
    /// Sphere light (spherical area light).
    Sphere = 6,
}

/// GPU-friendly light data structure (80 bytes).
///
/// For rect lights (type=3):
/// - `inner_cone_cos` stores the rectangle width
/// - `outer_cone_cos` stores the rectangle height
/// - `tangent` stores the tangent vector (width direction)
/// - `flags` bit 0: two-sided emission
///
/// For capsule/tube lights (type=4):
/// - `position` stores the start point
/// - `direction` stores the end point
/// - `inner_cone_cos` stores the length
/// - `outer_cone_cos` stores the radius
///
/// For disk lights (type=5):
/// - `inner_cone_cos` stores the disk radius
/// - `direction` stores the disk normal
/// - `flags` bit 0: two-sided emission
///
/// For sphere lights (type=6):
/// - `inner_cone_cos` stores the sphere radius
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LightUniform {
    /// Light position (for point/spot/rect) or direction (for directional).
    pub position: [f32; 3],
    /// Light type (0=point, 1=directional, 2=spot, 3=rect).
    pub light_type: u32,
    /// Light color.
    pub color: [f32; 3],
    /// Light intensity.
    pub intensity: f32,
    /// Direction (for spot/directional/rect lights). For rect, this is the normal.
    pub direction: [f32; 3],
    /// Range/radius (0 = infinite for directional).
    pub range: f32,
    /// Inner cone angle cosine (spot light) or width (rect light).
    pub inner_cone_cos: f32,
    /// Outer cone angle cosine (spot light) or height (rect light).
    pub outer_cone_cos: f32,
    /// Tangent vector for rect lights (defines width direction).
    pub tangent: [f32; 3],
    /// Flags: bit 0 = two-sided (rect lights).
    pub flags: u32,
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
            inner_cone_cos: 0.9,  // ~25 degrees for spot, or width for rect
            outer_cone_cos: 0.8,  // ~36 degrees for spot, or height for rect
            tangent: [1.0, 0.0, 0.0],
            flags: 0,
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

/// Render mode for debug visualization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum RenderMode {
    /// Standard PBR lit rendering.
    #[default]
    Lit = 0,
    /// Unlit - base color only, no lighting.
    Unlit = 1,
    /// Visualize world-space normals.
    Normals = 2,
    /// Visualize depth buffer.
    Depth = 3,
    /// Visualize metallic value.
    Metallic = 4,
    /// Visualize roughness value.
    Roughness = 5,
    /// Visualize ambient occlusion.
    AO = 6,
    /// Visualize UV coordinates.
    UVs = 7,
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
    /// Render mode (0=Lit, 1=Unlit, 2=Normals, 3=Depth, 4=Metallic, 5=Roughness, 6=AO, 7=UVs).
    pub render_mode: u32,
    /// Padding (using u32 to match WGSL alignment).
    pub _padding: [u32; 2],
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
            render_mode: 0,
            _padding: [0; 2],
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
