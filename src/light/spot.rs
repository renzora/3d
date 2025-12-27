//! Spot light (cone-shaped).

use crate::core::Id;
use crate::math::{Color, Vector3};
use super::{Light, LightUniform, LightType};

/// Spot light emitting in a cone from a position.
pub struct SpotLight {
    /// Unique ID.
    id: Id,
    /// Light color.
    pub color: Color,
    /// Light intensity.
    pub intensity: f32,
    /// Light position.
    pub position: Vector3,
    /// Light direction.
    direction: Vector3,
    /// Light range.
    pub range: f32,
    /// Inner cone angle in radians (full intensity).
    inner_angle: f32,
    /// Outer cone angle in radians (falloff to zero).
    outer_angle: f32,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self::new(
            Color::WHITE,
            1.0,
            Vector3::new(0.0, 5.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            10.0,
            25.0_f32.to_radians(),
            35.0_f32.to_radians(),
        )
    }
}

impl SpotLight {
    /// Create a new spot light.
    ///
    /// # Arguments
    /// * `color` - Light color
    /// * `intensity` - Light intensity
    /// * `position` - Light position
    /// * `direction` - Light direction
    /// * `range` - Maximum range
    /// * `inner_angle` - Inner cone angle in radians (full intensity)
    /// * `outer_angle` - Outer cone angle in radians (falloff edge)
    pub fn new(
        color: Color,
        intensity: f32,
        position: Vector3,
        direction: Vector3,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self {
            id: Id::new(),
            color,
            intensity,
            position,
            direction: direction.normalized(),
            range,
            inner_angle,
            outer_angle,
        }
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

    /// Set the light position.
    pub fn set_position(&mut self, position: Vector3) {
        self.position = position;
    }

    /// Point the light at a target.
    pub fn look_at(&mut self, target: Vector3) {
        self.direction = (target - self.position).normalized();
    }

    /// Get the inner cone angle in radians.
    #[inline]
    pub fn inner_angle(&self) -> f32 {
        self.inner_angle
    }

    /// Get the outer cone angle in radians.
    #[inline]
    pub fn outer_angle(&self) -> f32 {
        self.outer_angle
    }

    /// Set cone angles in radians.
    pub fn set_angles(&mut self, inner: f32, outer: f32) {
        self.inner_angle = inner;
        self.outer_angle = outer.max(inner);
    }

    /// Set cone angles in degrees.
    pub fn set_angles_degrees(&mut self, inner: f32, outer: f32) {
        self.set_angles(inner.to_radians(), outer.to_radians());
    }
}

impl Light for SpotLight {
    fn to_uniform(&self) -> LightUniform {
        LightUniform {
            position: [self.position.x, self.position.y, self.position.z],
            light_type: LightType::Spot as u32,
            color: [self.color.r, self.color.g, self.color.b],
            intensity: self.intensity,
            direction: [self.direction.x, self.direction.y, self.direction.z],
            range: self.range,
            inner_cone_cos: self.inner_angle.cos(),
            outer_cone_cos: self.outer_angle.cos(),
            _padding: [0.0; 2],
        }
    }
}
