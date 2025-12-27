//! 4D Vector implementation.

use super::{Matrix4, Vector3};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 4D vector with x, y, z, and w components.
/// Used for homogeneous coordinates and RGBA colors.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Vector4 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
    /// W component.
    pub w: f32,
}

impl Vector4 {
    /// Zero vector (0, 0, 0, 0).
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    /// One vector (1, 1, 1, 1).
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
    /// Unit X vector (1, 0, 0, 0).
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0, z: 0.0, w: 0.0 };
    /// Unit Y vector (0, 1, 0, 0).
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0, z: 0.0, w: 0.0 };
    /// Unit Z vector (0, 0, 1, 0).
    pub const UNIT_Z: Self = Self { x: 0.0, y: 0.0, z: 1.0, w: 0.0 };
    /// Unit W vector (0, 0, 0, 1).
    pub const UNIT_W: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    /// Create a new Vector4.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Create a vector with all components set to the same value.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v, w: v }
    }

    /// Create from an array.
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self { x: a[0], y: a[1], z: a[2], w: a[3] }
    }

    /// Convert to an array.
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Create from Vector3 with w component.
    #[inline]
    pub const fn from_vec3(v: Vector3, w: f32) -> Self {
        Self { x: v.x, y: v.y, z: v.z, w }
    }

    /// Get xyz components as Vector3.
    #[inline]
    pub const fn xyz(&self) -> Vector3 {
        Vector3 { x: self.x, y: self.y, z: self.z }
    }

    /// Set the components of this vector.
    #[inline]
    pub fn set(&mut self, x: f32, y: f32, z: f32, w: f32) -> &mut Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
        self
    }

    /// Copy from another vector.
    #[inline]
    pub fn copy(&mut self, v: &Vector4) -> &mut Self {
        self.x = v.x;
        self.y = v.y;
        self.z = v.z;
        self.w = v.w;
        self
    }

    /// Get the length (magnitude) of the vector.
    #[inline]
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Get the squared length of the vector.
    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Normalize the vector (make it unit length).
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        let len = self.length();
        if len > 0.0 {
            let inv_len = 1.0 / len;
            self.x *= inv_len;
            self.y *= inv_len;
            self.z *= inv_len;
            self.w *= inv_len;
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
    pub fn dot(&self, other: &Vector4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Linear interpolation to another vector.
    #[inline]
    pub fn lerp(&self, other: &Vector4, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
            w: self.w + (other.w - self.w) * t,
        }
    }

    /// Component-wise minimum.
    #[inline]
    pub fn min(&self, other: &Vector4) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
            w: self.w.min(other.w),
        }
    }

    /// Component-wise maximum.
    #[inline]
    pub fn max(&self, other: &Vector4) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
            w: self.w.max(other.w),
        }
    }

    /// Clamp components between min and max vectors.
    #[inline]
    pub fn clamp(&self, min: &Vector4, max: &Vector4) -> Self {
        Self {
            x: self.x.max(min.x).min(max.x),
            y: self.y.max(min.y).min(max.y),
            z: self.z.max(min.z).min(max.z),
            w: self.w.max(min.w).min(max.w),
        }
    }

    /// Floor all components.
    #[inline]
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
            w: self.w.floor(),
        }
    }

    /// Ceil all components.
    #[inline]
    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
            w: self.w.ceil(),
        }
    }

    /// Round all components.
    #[inline]
    pub fn round(&self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
            w: self.w.round(),
        }
    }

    /// Negate all components.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self.w = -self.w;
        self
    }

    /// Apply a Matrix4 transformation.
    #[inline]
    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        let e = &m.elements;
        Self {
            x: e[0] * self.x + e[4] * self.y + e[8] * self.z + e[12] * self.w,
            y: e[1] * self.x + e[5] * self.y + e[9] * self.z + e[13] * self.w,
            z: e[2] * self.x + e[6] * self.y + e[10] * self.z + e[14] * self.w,
            w: e[3] * self.x + e[7] * self.y + e[11] * self.z + e[15] * self.w,
        }
    }

    /// Check if the vector is approximately equal to another.
    #[inline]
    pub fn approx_eq(&self, other: &Vector4, epsilon: f32) -> bool {
        (self.x - other.x).abs() < epsilon
            && (self.y - other.y).abs() < epsilon
            && (self.z - other.z).abs() < epsilon
            && (self.w - other.w).abs() < epsilon
    }
}

// Operator implementations
impl Add for Vector4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl AddAssign for Vector4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl Sub for Vector4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl SubAssign for Vector4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl Mul<f32> for Vector4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}

impl Mul<Vector4> for f32 {
    type Output = Vector4;
    #[inline]
    fn mul(self, rhs: Vector4) -> Vector4 {
        Vector4 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
            w: self * rhs.w,
        }
    }
}

impl MulAssign<f32> for Vector4 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w *= rhs;
    }
}

impl Div<f32> for Vector4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
            w: self.w * inv,
        }
    }
}

impl DivAssign<f32> for Vector4 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        let inv = 1.0 / rhs;
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
        self.w *= inv;
    }
}

impl Neg for Vector4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl From<[f32; 4]> for Vector4 {
    fn from(a: [f32; 4]) -> Self {
        Self::from_array(a)
    }
}

impl From<Vector4> for [f32; 4] {
    fn from(v: Vector4) -> Self {
        v.to_array()
    }
}

impl From<(f32, f32, f32, f32)> for Vector4 {
    fn from((x, y, z, w): (f32, f32, f32, f32)) -> Self {
        Self { x, y, z, w }
    }
}

impl From<glam::Vec4> for Vector4 {
    fn from(v: glam::Vec4) -> Self {
        Self { x: v.x, y: v.y, z: v.z, w: v.w }
    }
}

impl From<Vector4> for glam::Vec4 {
    fn from(v: Vector4) -> Self {
        glam::Vec4::new(v.x, v.y, v.z, v.w)
    }
}
