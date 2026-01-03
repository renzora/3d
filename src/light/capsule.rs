//! Capsule/Tube light (line-based area light).
//!
//! Capsule light implementation using line irradiance
//! for accurate soft lighting from tube-shaped sources like fluorescent lights.

use crate::core::Id;
use crate::math::{Color, Vector3};
use super::{Light, LightUniform, LightType};

/// Capsule/Tube area light emitting from a line segment with spherical caps.
///
/// Capsule lights provide physically accurate soft lighting from linear
/// light sources like fluorescent tubes, LED strips, or neon signs.
///
/// # Implementation Notes
/// Based on "Real Shading in Unreal Engine 4" by Brian Karis (SIGGRAPH 2013).
///
/// The light is defined by two endpoints forming a line segment, with
/// an optional radius for the capsule thickness.
pub struct CapsuleLight {
    /// Unique ID.
    id: Id,
    /// Light color.
    pub color: Color,
    /// Light intensity (in lumens for physical units).
    pub intensity: f32,
    /// Start point of the capsule line segment.
    pub start: Vector3,
    /// End point of the capsule line segment.
    pub end: Vector3,
    /// Capsule radius (thickness of the tube).
    pub radius: f32,
    /// Light range (distance at which intensity falls to zero).
    pub range: f32,
    /// Whether this light casts shadows.
    pub cast_shadow: bool,
    /// Soft source radius for shadow penumbra.
    pub source_radius: f32,
    /// Shadow depth bias.
    pub shadow_bias: f32,
    /// Shadow normal bias.
    pub shadow_normal_bias: f32,
}

impl Default for CapsuleLight {
    fn default() -> Self {
        Self::new(
            Color::WHITE,
            1000.0, // Default to 1000 lumens
            Vector3::new(-1.0, 2.0, 0.0),  // Start point
            Vector3::new(1.0, 2.0, 0.0),   // End point (2 units long)
            0.05,                           // 5cm radius
            20.0,                           // 20 unit range
        )
    }
}

impl CapsuleLight {
    /// Create a new capsule light.
    pub fn new(
        color: Color,
        intensity: f32,
        start: Vector3,
        end: Vector3,
        radius: f32,
        range: f32,
    ) -> Self {
        Self {
            id: Id::new(),
            color,
            intensity,
            start,
            end,
            radius,
            range,
            cast_shadow: false,
            source_radius: radius,
            shadow_bias: 0.005,
            shadow_normal_bias: 0.02,
        }
    }

    /// Create a capsule light centered at a position with a given direction and length.
    pub fn from_center(
        color: Color,
        intensity: f32,
        center: Vector3,
        direction: Vector3,
        length: f32,
        radius: f32,
        range: f32,
    ) -> Self {
        let dir = direction.normalized();
        let half_len = length * 0.5;
        let start = center - dir * half_len;
        let end = center + dir * half_len;

        Self::new(color, intensity, start, end, radius, range)
    }

    /// Create a horizontal tube light (common for ceiling fixtures).
    pub fn horizontal(
        color: Color,
        intensity: f32,
        center: Vector3,
        length: f32,
        radius: f32,
        range: f32,
    ) -> Self {
        Self::from_center(
            color,
            intensity,
            center,
            Vector3::new(1.0, 0.0, 0.0), // Along X axis
            length,
            radius,
            range,
        )
    }

    /// Create a vertical tube light (like a standing lamp).
    pub fn vertical(
        color: Color,
        intensity: f32,
        center: Vector3,
        length: f32,
        radius: f32,
        range: f32,
    ) -> Self {
        Self::from_center(
            color,
            intensity,
            center,
            Vector3::new(0.0, 1.0, 0.0), // Along Y axis
            length,
            radius,
            range,
        )
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the center of the capsule.
    pub fn center(&self) -> Vector3 {
        (self.start + self.end) * 0.5
    }

    /// Get the direction of the capsule (from start to end, normalized).
    pub fn direction(&self) -> Vector3 {
        (self.end - self.start).normalized()
    }

    /// Get the length of the capsule line segment.
    pub fn length(&self) -> f32 {
        (self.end - self.start).length()
    }

    /// Set the start point.
    pub fn set_start(&mut self, start: Vector3) {
        self.start = start;
    }

    /// Set the end point.
    pub fn set_end(&mut self, end: Vector3) {
        self.end = end;
    }

    /// Set both endpoints.
    pub fn set_endpoints(&mut self, start: Vector3, end: Vector3) {
        self.start = start;
        self.end = end;
    }

    /// Set the capsule from center, direction, and length.
    pub fn set_from_center(&mut self, center: Vector3, direction: Vector3, length: f32) {
        let dir = direction.normalized();
        let half_len = length * 0.5;
        self.start = center - dir * half_len;
        self.end = center + dir * half_len;
    }

    /// Enable shadows with default settings.
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

    /// Set the source radius for soft shadows.
    pub fn with_source_radius(mut self, radius: f32) -> Self {
        self.source_radius = radius;
        self
    }

    /// Get the surface area of the capsule (for energy normalization).
    pub fn surface_area(&self) -> f32 {
        let length = self.length();
        // Surface area of a capsule = cylinder + two hemisphere caps
        // = 2*PI*r*L + 4*PI*r^2
        let cylinder_area = 2.0 * std::f32::consts::PI * self.radius * length;
        let cap_area = 4.0 * std::f32::consts::PI * self.radius * self.radius;
        cylinder_area + cap_area
    }
}

impl Light for CapsuleLight {
    fn to_uniform(&self) -> LightUniform {
        // Pack capsule data into LightUniform:
        // - position: start point
        // - direction: end point (reusing direction field)
        // - inner_cone_cos: length (computed for validation)
        // - outer_cone_cos: radius
        // - tangent: not used for capsule, but we store end point components here for extra precision
        LightUniform {
            position: [self.start.x, self.start.y, self.start.z],
            light_type: LightType::Capsule as u32,
            color: [self.color.r, self.color.g, self.color.b],
            intensity: self.intensity,
            direction: [self.end.x, self.end.y, self.end.z], // Store end point in direction
            range: self.range,
            inner_cone_cos: self.length(),  // Store length
            outer_cone_cos: self.radius,    // Store radius
            tangent: [0.0, 0.0, 0.0],       // Not used for capsule
            flags: 0,
        }
    }
}
