//! Rect light (area light with rectangular shape).
//!
//! Rectangular area light implementation using spherical rectangle sampling
//! for accurate soft shadows and realistic falloff.

use crate::core::Id;
use crate::math::{Color, Vector3};
use super::{Light, LightUniform, LightType};

/// Rectangular area light emitting from a flat rectangular surface.
///
/// Rect lights provide physically accurate soft shadows and realistic
/// lighting from rectangular light sources like windows, TV screens,
/// or architectural lighting panels.
///
/// # Implementation Notes
/// Based on "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines"
/// by Heitz et al. 2016.
pub struct RectLight {
    /// Unique ID.
    id: Id,
    /// Light color.
    pub color: Color,
    /// Light intensity (in lumens for physical units).
    pub intensity: f32,
    /// Light position (center of the rectangle).
    pub position: Vector3,
    /// Light direction (normal to the rectangle, pointing outward).
    pub direction: Vector3,
    /// Tangent vector (defines the "width" direction of the rectangle).
    pub tangent: Vector3,
    /// Rectangle width (along tangent direction).
    pub width: f32,
    /// Rectangle height (along bitangent direction).
    pub height: f32,
    /// Light range (distance at which intensity falls to zero).
    pub range: f32,
    /// Barn door angle (0 = no barn doors, >0 = restricts emission angle).
    pub barn_door_angle: f32,
    /// Barn door length.
    pub barn_door_length: f32,
    /// Whether this light casts shadows.
    pub cast_shadow: bool,
    /// Whether the light emits from both sides.
    pub two_sided: bool,
    /// Shadow depth bias.
    pub shadow_bias: f32,
    /// Shadow normal bias.
    pub shadow_normal_bias: f32,
}

impl Default for RectLight {
    fn default() -> Self {
        Self::new(
            Color::WHITE,
            1000.0, // Default to 1000 lumens
            Vector3::ZERO,
            Vector3::new(0.0, 0.0, -1.0), // Facing -Z
            Vector3::new(1.0, 0.0, 0.0),  // Width along X
            1.0,                           // 1 unit wide
            1.0,                           // 1 unit tall
            20.0,                          // 20 unit range
        )
    }
}

impl RectLight {
    /// Create a new rect light.
    pub fn new(
        color: Color,
        intensity: f32,
        position: Vector3,
        direction: Vector3,
        tangent: Vector3,
        width: f32,
        height: f32,
        range: f32,
    ) -> Self {
        // Ensure direction and tangent are normalized and orthogonal
        let dir_normalized = direction.normalized();
        let tangent_normalized = tangent.normalized();

        Self {
            id: Id::new(),
            color,
            intensity,
            position,
            direction: dir_normalized,
            tangent: tangent_normalized,
            width,
            height,
            range,
            barn_door_angle: 88.0, // Nearly 90 degrees (fully open)
            barn_door_length: 0.0,
            cast_shadow: false,
            two_sided: false,
            shadow_bias: 0.005,
            shadow_normal_bias: 0.02,
        }
    }

    /// Create a rect light facing a target point.
    pub fn looking_at(
        color: Color,
        intensity: f32,
        position: Vector3,
        target: Vector3,
        width: f32,
        height: f32,
        range: f32,
    ) -> Self {
        let direction = (target - position).normalized();
        // Choose tangent perpendicular to direction
        let up = if direction.y.abs() > 0.99 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let tangent = direction.cross(&up).normalized();

        Self::new(color, intensity, position, direction, tangent, width, height, range)
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

    /// Set the light dimensions.
    pub fn set_size(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
    }

    /// Enable shadows with default settings.
    pub fn with_shadows(mut self) -> Self {
        self.cast_shadow = true;
        self
    }

    /// Enable two-sided emission (light emits from both sides).
    pub fn with_two_sided(mut self) -> Self {
        self.two_sided = true;
        self
    }

    /// Set barn door parameters for directional control.
    pub fn with_barn_doors(mut self, angle: f32, length: f32) -> Self {
        self.barn_door_angle = angle;
        self.barn_door_length = length;
        self
    }

    /// Enable shadows with custom bias values.
    pub fn with_shadow_bias(mut self, bias: f32, normal_bias: f32) -> Self {
        self.cast_shadow = true;
        self.shadow_bias = bias;
        self.shadow_normal_bias = normal_bias;
        self
    }

    /// Get the area of the rectangle in world units squared.
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Get the bitangent vector (perpendicular to both direction and tangent).
    pub fn bitangent(&self) -> Vector3 {
        self.direction.cross(&self.tangent).normalized()
    }

    /// Get the four corners of the rectangle in world space.
    pub fn corners(&self) -> [Vector3; 4] {
        let half_width = self.width * 0.5;
        let half_height = self.height * 0.5;
        let bitangent = self.bitangent();

        [
            self.position - self.tangent * half_width - bitangent * half_height,
            self.position + self.tangent * half_width - bitangent * half_height,
            self.position + self.tangent * half_width + bitangent * half_height,
            self.position - self.tangent * half_width + bitangent * half_height,
        ]
    }
}

impl Light for RectLight {
    fn to_uniform(&self) -> LightUniform {
        LightUniform {
            position: [self.position.x, self.position.y, self.position.z],
            light_type: LightType::Rect as u32,
            color: [self.color.r, self.color.g, self.color.b],
            intensity: self.intensity,
            direction: [self.direction.x, self.direction.y, self.direction.z],
            range: self.range,
            // Repurpose cone angles for rect dimensions
            inner_cone_cos: self.width,
            outer_cone_cos: self.height,
            // Store tangent in extended fields
            tangent: [self.tangent.x, self.tangent.y, self.tangent.z],
            flags: if self.two_sided { 1 } else { 0 },
        }
    }
}
