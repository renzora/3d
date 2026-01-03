//! Sphere light (spherical area light).
//!
//! Sphere area light implementation for omnidirectional
//! soft lighting from spherical sources like light bulbs or globes.

use crate::core::Id;
use crate::math::{Color, Vector3};
use super::{Light, LightUniform, LightType};

/// Sphere area light emitting uniformly from a spherical surface.
///
/// Sphere lights provide physically accurate soft shadows from
/// omnidirectional sources like light bulbs, glowing orbs, or
/// any spherical emitter.
///
/// # Implementation Notes
/// Based on "Real Shading in Unreal Engine 4" by Brian Karis (SIGGRAPH 2013).
///
/// Key features:
/// - Sphere solid angle calculation for diffuse
/// - Representative point on sphere surface for specular
/// - Energy normalization based on sphere surface area
pub struct SphereLight {
    /// Unique ID.
    id: Id,
    /// Light color.
    pub color: Color,
    /// Light intensity (in lumens for physical units).
    pub intensity: f32,
    /// Light position (center of the sphere).
    pub position: Vector3,
    /// Sphere radius (source size).
    pub source_radius: f32,
    /// Light range (distance at which intensity falls to zero).
    pub range: f32,
    /// Whether this light casts shadows.
    pub cast_shadow: bool,
    /// Shadow depth bias.
    pub shadow_bias: f32,
    /// Shadow normal bias.
    pub shadow_normal_bias: f32,
}

impl Default for SphereLight {
    fn default() -> Self {
        Self::new(
            Color::WHITE,
            1000.0, // Default to 1000 lumens
            Vector3::ZERO,
            0.1,    // 10cm radius (small light bulb)
            20.0,   // 20 unit range
        )
    }
}

impl SphereLight {
    /// Create a new sphere light.
    pub fn new(
        color: Color,
        intensity: f32,
        position: Vector3,
        source_radius: f32,
        range: f32,
    ) -> Self {
        Self {
            id: Id::new(),
            color,
            intensity,
            position,
            source_radius,
            range,
            cast_shadow: false,
            shadow_bias: 0.005,
            shadow_normal_bias: 0.02,
        }
    }

    /// Create a sphere light representing a light bulb.
    pub fn light_bulb(color: Color, intensity: f32, position: Vector3) -> Self {
        Self::new(
            color,
            intensity,
            position,
            0.03, // 3cm radius (typical bulb)
            15.0,
        )
    }

    /// Create a larger sphere light (like a glowing orb).
    pub fn orb(color: Color, intensity: f32, position: Vector3, radius: f32) -> Self {
        Self::new(color, intensity, position, radius, radius * 30.0)
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

    /// Set the sphere radius.
    pub fn set_radius(&mut self, radius: f32) {
        self.source_radius = radius;
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

    /// Get the surface area of the sphere.
    pub fn surface_area(&self) -> f32 {
        4.0 * std::f32::consts::PI * self.source_radius * self.source_radius
    }

    /// Get the solid angle subtended by the sphere from a given point.
    pub fn solid_angle_from(&self, point: Vector3) -> f32 {
        let d = (self.position - point).length();
        if d <= self.source_radius {
            // Inside the sphere - full hemisphere visible
            return 2.0 * std::f32::consts::PI;
        }
        // Solid angle of a sphere: 2*PI*(1 - cos(theta)) where sin(theta) = r/d
        let sin_theta = self.source_radius / d;
        let cos_theta = (1.0 - sin_theta * sin_theta).sqrt();
        2.0 * std::f32::consts::PI * (1.0 - cos_theta)
    }
}

impl Light for SphereLight {
    fn to_uniform(&self) -> LightUniform {
        LightUniform {
            position: [self.position.x, self.position.y, self.position.z],
            light_type: LightType::Sphere as u32,
            color: [self.color.r, self.color.g, self.color.b],
            intensity: self.intensity,
            direction: [0.0, 0.0, 0.0], // Not used for sphere
            range: self.range,
            inner_cone_cos: self.source_radius, // Store radius
            outer_cone_cos: 0.0,                // Not used
            tangent: [0.0, 0.0, 0.0],           // Not used
            flags: 0,
        }
    }
}
