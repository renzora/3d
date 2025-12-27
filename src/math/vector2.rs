//! 2D Vector implementation.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 2D vector with x and y components.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Vector2 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
}

impl Vector2 {
    /// Zero vector (0, 0).
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    /// One vector (1, 1).
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };
    /// Unit X vector (1, 0).
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0 };
    /// Unit Y vector (0, 1).
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0 };

    /// Create a new Vector2.
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Create a vector with all components set to the same value.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }

    /// Create from an array.
    #[inline]
    pub const fn from_array(a: [f32; 2]) -> Self {
        Self { x: a[0], y: a[1] }
    }

    /// Convert to an array.
    #[inline]
    pub const fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Set the components of this vector.
    #[inline]
    pub fn set(&mut self, x: f32, y: f32) -> &mut Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Copy from another vector.
    #[inline]
    pub fn copy(&mut self, v: &Vector2) -> &mut Self {
        self.x = v.x;
        self.y = v.y;
        self
    }

    /// Get the length (magnitude) of the vector.
    #[inline]
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Get the squared length of the vector.
    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Normalize the vector (make it unit length).
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        let len = self.length();
        if len > 0.0 {
            let inv_len = 1.0 / len;
            self.x *= inv_len;
            self.y *= inv_len;
        }
        self
    }

    /// Return a normalized copy of the vector.
    #[inline]
    pub fn normalized(&self) -> Self {
        let mut v = *self;
        v.normalize();
        v
    }

    /// Dot product with another vector.
    #[inline]
    pub fn dot(&self, other: &Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Cross product (returns scalar z-component).
    #[inline]
    pub fn cross(&self, other: &Vector2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Distance to another vector.
    #[inline]
    pub fn distance_to(&self, other: &Vector2) -> f32 {
        (*self - *other).length()
    }

    /// Squared distance to another vector.
    #[inline]
    pub fn distance_to_squared(&self, other: &Vector2) -> f32 {
        (*self - *other).length_squared()
    }

    /// Linear interpolation to another vector.
    #[inline]
    pub fn lerp(&self, other: &Vector2, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    /// Component-wise minimum.
    #[inline]
    pub fn min(&self, other: &Vector2) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Component-wise maximum.
    #[inline]
    pub fn max(&self, other: &Vector2) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }

    /// Clamp components between min and max vectors.
    #[inline]
    pub fn clamp(&self, min: &Vector2, max: &Vector2) -> Self {
        Self {
            x: self.x.max(min.x).min(max.x),
            y: self.y.max(min.y).min(max.y),
        }
    }

    /// Floor all components.
    #[inline]
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    /// Ceil all components.
    #[inline]
    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
        }
    }

    /// Round all components.
    #[inline]
    pub fn round(&self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
        }
    }

    /// Negate all components.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self
    }

    /// Get the angle of this vector in radians.
    #[inline]
    pub fn angle(&self) -> f32 {
        self.y.atan2(self.x)
    }

    /// Rotate the vector by an angle in radians.
    #[inline]
    pub fn rotate(&self, angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            x: self.x * cos - self.y * sin,
            y: self.x * sin + self.y * cos,
        }
    }

    /// Check if the vector is approximately equal to another.
    #[inline]
    pub fn approx_eq(&self, other: &Vector2, epsilon: f32) -> bool {
        (self.x - other.x).abs() < epsilon && (self.y - other.y).abs() < epsilon
    }
}

// Operator implementations
impl Add for Vector2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl AddAssign for Vector2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vector2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl SubAssign for Vector2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<f32> for Vector2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Vector2> for f32 {
    type Output = Vector2;
    #[inline]
    fn mul(self, rhs: Vector2) -> Vector2 {
        Vector2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl MulAssign<f32> for Vector2 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<f32> for Vector2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self {
            x: self.x * inv,
            y: self.y * inv,
        }
    }
}

impl DivAssign<f32> for Vector2 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        let inv = 1.0 / rhs;
        self.x *= inv;
        self.y *= inv;
    }
}

impl Neg for Vector2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl From<[f32; 2]> for Vector2 {
    fn from(a: [f32; 2]) -> Self {
        Self::from_array(a)
    }
}

impl From<Vector2> for [f32; 2] {
    fn from(v: Vector2) -> Self {
        v.to_array()
    }
}

impl From<(f32, f32)> for Vector2 {
    fn from((x, y): (f32, f32)) -> Self {
        Self { x, y }
    }
}

impl From<glam::Vec2> for Vector2 {
    fn from(v: glam::Vec2) -> Self {
        Self { x: v.x, y: v.y }
    }
}

impl From<Vector2> for glam::Vec2 {
    fn from(v: Vector2) -> Self {
        glam::Vec2::new(v.x, v.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let v = Vector2::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_length() {
        let v = Vector2::new(3.0, 4.0);
        assert_eq!(v.length(), 5.0);
    }

    #[test]
    fn test_normalize() {
        let mut v = Vector2::new(3.0, 4.0);
        v.normalize();
        assert!((v.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot() {
        let a = Vector2::new(1.0, 0.0);
        let b = Vector2::new(0.0, 1.0);
        assert_eq!(a.dot(&b), 0.0);
    }
}
