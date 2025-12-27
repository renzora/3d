//! Color implementation with RGB and HSL support.

use super::Vector3;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// RGB color with values in 0.0-1.0 range.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Color {
    /// Red component (0.0 to 1.0).
    pub r: f32,
    /// Green component (0.0 to 1.0).
    pub g: f32,
    /// Blue component (0.0 to 1.0).
    pub b: f32,
}

impl Color {
    /// Black (0, 0, 0).
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };
    /// White (1, 1, 1).
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };
    /// Red (1, 0, 0).
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0 };
    /// Green (0, 1, 0).
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0 };
    /// Blue (0, 0, 1).
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0 };
    /// Yellow (1, 1, 0).
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0 };
    /// Cyan (0, 1, 1).
    pub const CYAN: Self = Self { r: 0.0, g: 1.0, b: 1.0 };
    /// Magenta (1, 0, 1).
    pub const MAGENTA: Self = Self { r: 1.0, g: 0.0, b: 1.0 };
    /// Gray (0.5, 0.5, 0.5).
    pub const GRAY: Self = Self { r: 0.5, g: 0.5, b: 0.5 };

    /// Create a new color from RGB values (0.0-1.0).
    #[inline]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Create a color with all components set to the same value.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { r: v, g: v, b: v }
    }

    /// Create from an array.
    #[inline]
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self { r: a[0], g: a[1], b: a[2] }
    }

    /// Convert to an array.
    #[inline]
    pub const fn to_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// Create from a hex integer (0xRRGGBB).
    pub fn from_hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 16) & 0xFF) as f32 / 255.0,
            g: ((hex >> 8) & 0xFF) as f32 / 255.0,
            b: (hex & 0xFF) as f32 / 255.0,
        }
    }

    /// Convert to hex integer.
    pub fn to_hex(&self) -> u32 {
        let r = (self.r.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (self.g.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (self.b.clamp(0.0, 1.0) * 255.0) as u32;
        (r << 16) | (g << 8) | b
    }

    /// Create from RGB bytes (0-255).
    pub fn from_rgb_bytes(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
        }
    }

    /// Convert to RGB bytes.
    pub fn to_rgb_bytes(&self) -> [u8; 3] {
        [
            (self.r.clamp(0.0, 1.0) * 255.0) as u8,
            (self.g.clamp(0.0, 1.0) * 255.0) as u8,
            (self.b.clamp(0.0, 1.0) * 255.0) as u8,
        ]
    }

    /// Create from HSL values.
    /// - h: Hue (0.0-1.0, wraps around)
    /// - s: Saturation (0.0-1.0)
    /// - l: Lightness (0.0-1.0)
    pub fn from_hsl(h: f32, s: f32, l: f32) -> Self {
        if s == 0.0 {
            return Self::splat(l);
        }

        let hue_to_rgb = |p: f32, q: f32, mut t: f32| {
            if t < 0.0 { t += 1.0; }
            if t > 1.0 { t -= 1.0; }
            if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
            if t < 1.0 / 2.0 { return q; }
            if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
            p
        };

        let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
        let p = 2.0 * l - q;

        Self {
            r: hue_to_rgb(p, q, h + 1.0 / 3.0),
            g: hue_to_rgb(p, q, h),
            b: hue_to_rgb(p, q, h - 1.0 / 3.0),
        }
    }

    /// Convert to HSL.
    pub fn to_hsl(&self) -> (f32, f32, f32) {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let l = (max + min) / 2.0;

        if max == min {
            return (0.0, 0.0, l);
        }

        let d = max - min;
        let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };

        let h = if max == self.r {
            ((self.g - self.b) / d + if self.g < self.b { 6.0 } else { 0.0 }) / 6.0
        } else if max == self.g {
            ((self.b - self.r) / d + 2.0) / 6.0
        } else {
            ((self.r - self.g) / d + 4.0) / 6.0
        };

        (h, s, l)
    }

    /// Set components.
    #[inline]
    pub fn set(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.r = r;
        self.g = g;
        self.b = b;
        self
    }

    /// Copy from another color.
    #[inline]
    pub fn copy(&mut self, c: &Color) -> &mut Self {
        self.r = c.r;
        self.g = c.g;
        self.b = c.b;
        self
    }

    /// Convert from sRGB to linear space.
    pub fn srgb_to_linear(&self) -> Self {
        let to_linear = |c: f32| {
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        Self {
            r: to_linear(self.r),
            g: to_linear(self.g),
            b: to_linear(self.b),
        }
    }

    /// Convert from linear to sRGB space.
    pub fn linear_to_srgb(&self) -> Self {
        let to_srgb = |c: f32| {
            if c <= 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        };
        Self {
            r: to_srgb(self.r),
            g: to_srgb(self.g),
            b: to_srgb(self.b),
        }
    }

    /// Add another color.
    #[inline]
    pub fn add(&self, other: &Color) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }

    /// Multiply by another color.
    #[inline]
    pub fn multiply(&self, other: &Color) -> Self {
        Self {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }

    /// Multiply by a scalar.
    #[inline]
    pub fn multiply_scalar(&self, s: f32) -> Self {
        Self {
            r: self.r * s,
            g: self.g * s,
            b: self.b * s,
        }
    }

    /// Linear interpolation.
    #[inline]
    pub fn lerp(&self, other: &Color, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Get luminance (perceived brightness).
    #[inline]
    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Clamp all components to 0.0-1.0.
    #[inline]
    pub fn clamp(&self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }

    /// Convert to Vector3.
    #[inline]
    pub const fn to_vector3(&self) -> Vector3 {
        Vector3 {
            x: self.r,
            y: self.g,
            z: self.b,
        }
    }

    /// Create from Vector3.
    #[inline]
    pub const fn from_vector3(v: &Vector3) -> Self {
        Self {
            r: v.x,
            g: v.y,
            b: v.z,
        }
    }

    /// Check if approximately equal.
    #[inline]
    pub fn approx_eq(&self, other: &Color, epsilon: f32) -> bool {
        (self.r - other.r).abs() < epsilon
            && (self.g - other.g).abs() < epsilon
            && (self.b - other.b).abs() < epsilon
    }
}

impl From<[f32; 3]> for Color {
    fn from(a: [f32; 3]) -> Self {
        Self::from_array(a)
    }
}

impl From<Color> for [f32; 3] {
    fn from(c: Color) -> Self {
        c.to_array()
    }
}

impl From<u32> for Color {
    fn from(hex: u32) -> Self {
        Self::from_hex(hex)
    }
}

impl From<Vector3> for Color {
    fn from(v: Vector3) -> Self {
        Self::from_vector3(&v)
    }
}

impl std::ops::Add for Color {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl std::ops::Mul for Color {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}

impl std::ops::Mul<f32> for Color {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        self.multiply_scalar(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_roundtrip() {
        let c = Color::from_hex(0xFF8040);
        assert_eq!(c.to_hex(), 0xFF8040);
    }

    #[test]
    fn test_hsl_roundtrip() {
        let c = Color::from_hsl(0.5, 0.5, 0.5);
        let (h, s, l) = c.to_hsl();
        assert!((h - 0.5).abs() < 0.01);
        assert!((s - 0.5).abs() < 0.01);
        assert!((l - 0.5).abs() < 0.01);
    }
}
