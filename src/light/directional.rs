//! Directional light (sun-like parallel rays).

use crate::core::Id;
use crate::math::{Color, Vector3};
use super::{Light, LightUniform, LightType};

/// Directional light emitting parallel rays (like the sun).
pub struct DirectionalLight {
    /// Unique ID.
    id: Id,
    /// Light color.
    pub color: Color,
    /// Light intensity.
    pub intensity: f32,
    /// Light direction (normalized).
    direction: Vector3,
    /// Whether this light casts shadows.
    pub cast_shadow: bool,
    /// Shadow depth bias to prevent shadow acne.
    pub shadow_bias: f32,
    /// Shadow normal bias.
    pub shadow_normal_bias: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self::new(Color::WHITE, 1.0, Vector3::new(0.0, -1.0, 0.0))
    }
}

impl DirectionalLight {
    /// Create a new directional light.
    pub fn new(color: Color, intensity: f32, direction: Vector3) -> Self {
        Self {
            id: Id::new(),
            color,
            intensity,
            direction: direction.normalized(),
            cast_shadow: false,
            shadow_bias: 0.005,
            shadow_normal_bias: 0.02,
        }
    }

    /// Create a sun-like light pointing downward at an angle.
    pub fn sun() -> Self {
        Self::new(
            Color::new(1.0, 0.95, 0.9),
            2.0,
            Vector3::new(-0.5, -1.0, -0.3),
        )
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the light direction.
    #[inline]
    pub fn direction(&self) -> Vector3 {
        self.direction
    }

    /// Set the light direction.
    pub fn set_direction(&mut self, direction: Vector3) {
        self.direction = direction.normalized();
    }

    /// Point the light at a target from a position.
    pub fn look_at(&mut self, from: Vector3, to: Vector3) {
        self.direction = (to - from).normalized();
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

impl Light for DirectionalLight {
    fn to_uniform(&self) -> LightUniform {
        LightUniform {
            position: [0.0, 0.0, 0.0], // Not used for directional
            light_type: LightType::Directional as u32,
            color: [self.color.r, self.color.g, self.color.b],
            intensity: self.intensity,
            direction: [self.direction.x, self.direction.y, self.direction.z],
            range: 0.0, // Infinite
            inner_cone_cos: 0.0,
            outer_cone_cos: 0.0,
            tangent: [0.0, 0.0, 0.0], // Not used for directional
            flags: 0,
        }
    }
}
