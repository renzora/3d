//! Disk light (circular area light).
//!
//! Disk area light implementation for circular light sources
//! like ceiling fixtures, spotlights with diffusers, or round windows.

use crate::core::Id;
use crate::math::{Color, Vector3};
use super::{Light, LightUniform, LightType};

/// Disk area light emitting from a flat circular surface.
///
/// Disk lights provide physically accurate soft shadows from circular
/// light sources. They're ideal for recessed ceiling lights, round
/// light fixtures, and spotlight sources.
///
/// # Implementation Notes
/// Based on "Real-Time Area Lighting: a Journey from Research to Production"
/// by Stephen Hill and Eric Heitz (SIGGRAPH 2016).
pub struct DiskLight {
    /// Unique ID.
    id: Id,
    /// Light color.
    pub color: Color,
    /// Light intensity (in lumens for physical units).
    pub intensity: f32,
    /// Light position (center of the disk).
    pub position: Vector3,
    /// Light direction (normal to the disk, pointing outward).
    pub direction: Vector3,
    /// Disk radius.
    pub radius: f32,
    /// Light range (distance at which intensity falls to zero).
    pub range: f32,
    /// Inner cone angle for spot-like falloff (optional, 0 = no falloff).
    pub inner_angle: f32,
    /// Outer cone angle for spot-like falloff (optional, 0 = no falloff).
    pub outer_angle: f32,
    /// Whether this light casts shadows.
    pub cast_shadow: bool,
    /// Whether the light emits from both sides.
    pub two_sided: bool,
    /// Shadow depth bias.
    pub shadow_bias: f32,
    /// Shadow normal bias.
    pub shadow_normal_bias: f32,
}

impl Default for DiskLight {
    fn default() -> Self {
        Self::new(
            Color::WHITE,
            1000.0, // Default to 1000 lumens
            Vector3::new(0.0, 3.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0), // Pointing down
            0.5,                           // 0.5 unit radius
            20.0,                          // 20 unit range
        )
    }
}

impl DiskLight {
    /// Create a new disk light.
    pub fn new(
        color: Color,
        intensity: f32,
        position: Vector3,
        direction: Vector3,
        radius: f32,
        range: f32,
    ) -> Self {
        Self {
            id: Id::new(),
            color,
            intensity,
            position,
            direction: direction.normalized(),
            radius,
            range,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cast_shadow: false,
            two_sided: false,
            shadow_bias: 0.005,
            shadow_normal_bias: 0.02,
        }
    }

    /// Create a disk light facing a target point.
    pub fn looking_at(
        color: Color,
        intensity: f32,
        position: Vector3,
        target: Vector3,
        radius: f32,
        range: f32,
    ) -> Self {
        let direction = (target - position).normalized();
        Self::new(color, intensity, position, direction, radius, range)
    }

    /// Create a downward-facing ceiling light.
    pub fn ceiling(
        color: Color,
        intensity: f32,
        position: Vector3,
        radius: f32,
        range: f32,
    ) -> Self {
        Self::new(
            color,
            intensity,
            position,
            Vector3::new(0.0, -1.0, 0.0), // Pointing down
            radius,
            range,
        )
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

    /// Set the light direction (normal).
    pub fn set_direction(&mut self, direction: Vector3) {
        self.direction = direction.normalized();
    }

    /// Set the disk radius.
    pub fn set_radius(&mut self, radius: f32) {
        self.radius = radius;
    }

    /// Add spot-like angular falloff.
    pub fn with_cone_falloff(mut self, inner_angle: f32, outer_angle: f32) -> Self {
        self.inner_angle = inner_angle;
        self.outer_angle = outer_angle;
        self
    }

    /// Enable shadows with default settings.
    pub fn with_shadows(mut self) -> Self {
        self.cast_shadow = true;
        self
    }

    /// Enable two-sided emission.
    pub fn with_two_sided(mut self) -> Self {
        self.two_sided = true;
        self
    }

    /// Enable shadows with custom bias values.
    pub fn with_shadow_bias(mut self, bias: f32, normal_bias: f32) -> Self {
        self.cast_shadow = true;
        self.shadow_bias = bias;
        self.shadow_normal_bias = normal_bias;
        self
    }

    /// Get the area of the disk.
    pub fn area(&self) -> f32 {
        std::f32::consts::PI * self.radius * self.radius
    }
}

impl Light for DiskLight {
    fn to_uniform(&self) -> LightUniform {
        LightUniform {
            position: [self.position.x, self.position.y, self.position.z],
            light_type: LightType::Disk as u32,
            color: [self.color.r, self.color.g, self.color.b],
            intensity: self.intensity,
            direction: [self.direction.x, self.direction.y, self.direction.z],
            range: self.range,
            inner_cone_cos: self.radius,     // Store radius
            outer_cone_cos: self.inner_angle, // Store inner angle
            tangent: [0.0, 0.0, 0.0],        // Not needed for disk
            flags: if self.two_sided { 1 } else { 0 },
        }
    }
}
