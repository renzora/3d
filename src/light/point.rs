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
            _padding: [0.0; 2],
        }
    }
}
