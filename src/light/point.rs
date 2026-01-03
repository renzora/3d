//! Point light (omni-directional).

use crate::core::Id;
use crate::math::{Color, Vector3};
use super::{Light, LightUniform, LightType};

/// Point light emitting in all directions from a position.
pub struct PointLight {
    /// Unique ID.
    id: Id,
    /// Light color.
    pub color: Color,
    /// Light intensity.
    pub intensity: f32,
    /// Light position.
    pub position: Vector3,
    /// Light range (distance at which intensity falls to zero).
    pub range: f32,
    /// Whether this light casts shadows.
    pub cast_shadow: bool,
    /// Shadow depth bias to prevent shadow acne.
    pub shadow_bias: f32,
    /// Shadow normal bias.
    pub shadow_normal_bias: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self::new(Color::WHITE, 1.0, Vector3::ZERO, 10.0)
    }
}

impl PointLight {
    /// Create a new point light.
    pub fn new(color: Color, intensity: f32, position: Vector3, range: f32) -> Self {
        Self {
            id: Id::new(),
            color,
            intensity,
            position,
            range,
            cast_shadow: false,
            shadow_bias: 0.005,
            shadow_normal_bias: 0.02,
        }
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Set the light position.
    pub fn set_position(&mut self, position: Vector3) {
        self.position = position;
    }

    /// Enable shadows for this light with default settings.
    pub fn with_shadows(mut self) -> Self {
        self.cast_shadow = true;
        self
    }

    /// Enable shadows with custom bias values.
    pub fn with_shadow_bias(mut self, bias: f32, normal_bias: f32) -> Self {
        self.cast_shadow = true;
        self.shadow_bias = bias;
        self.shadow_normal_bias = normal_bias;
        self
    }
}

impl Light for PointLight {
    fn to_uniform(&self) -> LightUniform {
        LightUniform {
            position: [self.position.x, self.position.y, self.position.z],
            light_type: LightType::Point as u32,
            color: [self.color.r, self.color.g, self.color.b],
            intensity: self.intensity,
            direction: [0.0, 0.0, 0.0], // Not used for point
            range: self.range,
            inner_cone_cos: 0.0,
            outer_cone_cos: 0.0,
            tangent: [0.0, 0.0, 0.0], // Not used for point
            flags: 0,
        }
    }
}
