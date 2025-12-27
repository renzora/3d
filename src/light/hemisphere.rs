//! Hemisphere light for sky/ground gradient illumination.

use crate::core::Id;
use crate::math::Color;

/// Hemisphere light that illuminates with a gradient between sky and ground colors.
///
/// Surfaces facing up receive the sky color, surfaces facing down receive the ground color,
/// and surfaces at an angle receive a blend of both colors.
pub struct HemisphereLight {
    /// Unique ID.
    id: Id,
    /// Sky color (for surfaces facing up, Y+).
    pub sky_color: Color,
    /// Ground color (for surfaces facing down, Y-).
    pub ground_color: Color,
    /// Light intensity.
    pub intensity: f32,
    /// Whether the light is enabled.
    pub enabled: bool,
}

impl Default for HemisphereLight {
    fn default() -> Self {
        Self::new(
            Color::new(0.6, 0.75, 1.0),  // Light blue sky
            Color::new(0.4, 0.3, 0.2),   // Brown ground
            1.0,
        )
    }
}

impl HemisphereLight {
    /// Create a new hemisphere light.
    pub fn new(sky_color: Color, ground_color: Color, intensity: f32) -> Self {
        Self {
            id: Id::new(),
            sky_color,
            ground_color,
            intensity,
            enabled: true,
        }
    }

    /// Create a typical outdoor hemisphere light.
    pub fn outdoor() -> Self {
        Self::new(
            Color::new(0.7, 0.85, 1.0),   // Light blue sky
            Color::new(0.4, 0.35, 0.3),   // Earthy ground
            0.5,
        )
    }

    /// Create an indoor hemisphere light.
    pub fn indoor() -> Self {
        Self::new(
            Color::new(0.9, 0.9, 0.85),   // Warm ceiling
            Color::new(0.5, 0.45, 0.4),   // Floor reflection
            0.3,
        )
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the effective sky color (color * intensity).
    pub fn effective_sky_color(&self) -> Color {
        Color::new(
            self.sky_color.r * self.intensity,
            self.sky_color.g * self.intensity,
            self.sky_color.b * self.intensity,
        )
    }

    /// Get the effective ground color (color * intensity).
    pub fn effective_ground_color(&self) -> Color {
        Color::new(
            self.ground_color.r * self.intensity,
            self.ground_color.g * self.intensity,
            self.ground_color.b * self.intensity,
        )
    }

    /// Set enabled state.
    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}
