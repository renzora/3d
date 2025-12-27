//! Ambient light for global illumination.

use crate::core::Id;
use crate::math::Color;

/// Ambient light that illuminates all objects equally.
pub struct AmbientLight {
    /// Unique ID.
    id: Id,
    /// Light color.
    pub color: Color,
    /// Light intensity.
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self::new(Color::WHITE, 0.1)
    }
}

impl AmbientLight {
    /// Create a new ambient light.
    pub fn new(color: Color, intensity: f32) -> Self {
        Self {
            id: Id::new(),
            color,
            intensity,
        }
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the effective color (color * intensity).
    pub fn effective_color(&self) -> Color {
        Color::new(
            self.color.r * self.intensity,
            self.color.g * self.intensity,
            self.color.b * self.intensity,
        )
    }
}
