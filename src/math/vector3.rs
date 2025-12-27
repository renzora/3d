//! 3D Vector implementation.

use super::{Matrix4, Quaternion, Vector2};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 3D vector with x, y, and z components.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Vector3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vector3 {
    /// Zero vector (0, 0, 0).
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    /// One vector (1, 1, 1).
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    /// Unit X vector (1, 0, 0).
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    /// Unit Y vector (0, 1, 0).
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    /// Unit Z vector (0, 0, 1).
    pub const UNIT_Z: Self = Self { x: 0.0, y: 0.0, z: 1.0 };
    /// Up vector (0, 1, 0).
    pub const UP: Self = Self::UNIT_Y;
    /// Down vector (0, -1, 0).
    pub const DOWN: Self = Self { x: 0.0, y: -1.0, z: 0.0 };
    /// Forward vector (0, 0, -1) - looking into the screen in right-handed coords.
    pub const FORWARD: Self = Self { x: 0.0, y: 0.0, z: -1.0 };
    /// Back vector (0, 0, 1).
    pub const BACK: Self = Self { x: 0.0, y: 0.0, z: 1.0 };
    /// Right vector (1, 0, 0).
    pub const RIGHT: Self = Self::UNIT_X;
    /// Left vector (-1, 0, 0).
    pub const LEFT: Self = Self { x: -1.0, y: 0.0, z: 0.0 };

    /// Create a new Vector3.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Create a vector with all components set to the same value.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Create from an array.
    #[inline]
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }

    /// Convert to an array.
    #[inline]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Create from a Vector2 with z = 0.
    #[inline]
    pub const fn from_vec2(v: Vector2, z: f32) -> Self {
        Self { x: v.x, y: v.y, z }
    }

    /// Get xy components as Vector2.
    #[inline]
    pub const fn xy(&self) -> Vector2 {
        Vector2 { x: self.x, y: self.y }
    }

    /// Set the components of this vector.
    #[inline]
    pub fn set(&mut self, x: f32, y: f32, z: f32) -> &mut Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self
    }

    /// Copy from another vector.
    #[inline]
    pub fn copy(&mut self, v: &Vector3) -> &mut Self {
        self.x = v.x;
        self.y = v.y;
        self.z = v.z;
        self
    }

    /// Get the length (magnitude) of the vector.
    #[inline]
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Get the squared length of the vector.
    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Get the Manhattan length of the vector.
    #[inline]
    pub fn manhattan_length(&self) -> f32 {
        self.x.abs() + self.y.abs() + self.z.abs()
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

    /// Set the length of this vector.
    #[inline]
    pub fn set_length(&mut self, length: f32) -> &mut Self {
        self.normalize();
        self.x *= length;
        self.y *= length;
        self.z *= length;
        self
    }

    /// Dot product with another vector.
    #[inline]
    pub fn dot(&self, other: &Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product with another vector.
    #[inline]
    pub fn cross(&self, other: &Vector3) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Distance to another vector.
    #[inline]
    pub fn distance_to(&self, other: &Vector3) -> f32 {
        (*self - *other).length()
    }

    /// Squared distance to another vector.
    #[inline]
    pub fn distance_to_squared(&self, other: &Vector3) -> f32 {
        (*self - *other).length_squared()
    }

    /// Manhattan distance to another vector.
    #[inline]
    pub fn manhattan_distance_to(&self, other: &Vector3) -> f32 {
        (self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()
    }

    /// Linear interpolation to another vector.
    #[inline]
    pub fn lerp(&self, other: &Vector3, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    /// Spherical linear interpolation (for directions).
    pub fn slerp(&self, other: &Vector3, t: f32) -> Self {
        let dot = self.dot(other).clamp(-1.0, 1.0);
        let theta = dot.acos() * t;
        let relative = (*other - *self * dot).normalized();
        *self * theta.cos() + relative * theta.sin()
    }

    /// Component-wise minimum.
    #[inline]
    pub fn min(&self, other: &Vector3) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Component-wise maximum.
    #[inline]
    pub fn max(&self, other: &Vector3) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Clamp components between min and max vectors.
    #[inline]
    pub fn clamp(&self, min: &Vector3, max: &Vector3) -> Self {
        Self {
            x: self.x.max(min.x).min(max.x),
            y: self.y.max(min.y).min(max.y),
            z: self.z.max(min.z).min(max.z),
        }
    }

    /// Clamp the length of this vector.
    #[inline]
    pub fn clamp_length(&self, min: f32, max: f32) -> Self {
        let len = self.length();
        if len < min {
            self.normalized() * min
        } else if len > max {
            self.normalized() * max
        } else {
            *self
        }
    }

    /// Floor all components.
    #[inline]
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    /// Ceil all components.
    #[inline]
    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }

    /// Round all components.
    #[inline]
    pub fn round(&self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
        }
    }

    /// Absolute value of all components.
    #[inline]
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Negate all components.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self
    }

    /// Reflect this vector off a surface with the given normal.
    #[inline]
    pub fn reflect(&self, normal: &Vector3) -> Self {
        *self - *normal * 2.0 * self.dot(normal)
    }

    /// Get the angle to another vector in radians.
    #[inline]
    pub fn angle_to(&self, other: &Vector3) -> f32 {
        let denominator = (self.length_squared() * other.length_squared()).sqrt();
        if denominator == 0.0 {
            std::f32::consts::FRAC_PI_2
        } else {
            (self.dot(other) / denominator).clamp(-1.0, 1.0).acos()
        }
    }

    /// Project this vector onto another vector.
    #[inline]
    pub fn project(&self, onto: &Vector3) -> Self {
        let denom = onto.length_squared();
        if denom == 0.0 {
            Self::ZERO
        } else {
            *onto * (self.dot(onto) / denom)
        }
    }

    /// Project this vector onto a plane defined by its normal.
    #[inline]
    pub fn project_on_plane(&self, plane_normal: &Vector3) -> Self {
        *self - self.project(plane_normal)
    }

    /// Apply a Matrix4 transformation.
    #[inline]
    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        let e = &m.elements;
        let w = 1.0 / (e[3] * self.x + e[7] * self.y + e[11] * self.z + e[15]);
        Self {
            x: (e[0] * self.x + e[4] * self.y + e[8] * self.z + e[12]) * w,
            y: (e[1] * self.x + e[5] * self.y + e[9] * self.z + e[13]) * w,
            z: (e[2] * self.x + e[6] * self.y + e[10] * self.z + e[14]) * w,
        }
    }

    /// Apply a Quaternion rotation.
    #[inline]
    pub fn apply_quaternion(&self, q: &Quaternion) -> Self {
        // q * v * q^-1
        let qx = q.x;
        let qy = q.y;
        let qz = q.z;
        let qw = q.w;

        let ix = qw * self.x + qy * self.z - qz * self.y;
        let iy = qw * self.y + qz * self.x - qx * self.z;
        let iz = qw * self.z + qx * self.y - qy * self.x;
        let iw = -qx * self.x - qy * self.y - qz * self.z;

        Self {
            x: ix * qw + iw * -qx + iy * -qz - iz * -qy,
            y: iy * qw + iw * -qy + iz * -qx - ix * -qz,
            z: iz * qw + iw * -qz + ix * -qy - iy * -qx,
        }
    }

    /// Check if the vector is approximately equal to another.
    #[inline]
    pub fn approx_eq(&self, other: &Vector3, epsilon: f32) -> bool {
        (self.x - other.x).abs() < epsilon
            && (self.y - other.y).abs() < epsilon
            && (self.z - other.z).abs() < epsilon
    }

    /// Component-wise multiplication.
    #[inline]
    pub fn multiply(&self, other: &Vector3) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    /// Component-wise division.
    #[inline]
    pub fn divide(&self, other: &Vector3) -> Self {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}

// Operator implementations
impl Add for Vector3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for Vector3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vector3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign for Vector3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f32> for Vector3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vector3> for f32 {
    type Output = Vector3;
    #[inline]
    fn mul(self, rhs: Vector3) -> Vector3 {
        Vector3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl MulAssign<f32> for Vector3 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div<f32> for Vector3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }
}

impl DivAssign<f32> for Vector3 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        let inv = 1.0 / rhs;
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl Neg for Vector3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl From<[f32; 3]> for Vector3 {
    fn from(a: [f32; 3]) -> Self {
        Self::from_array(a)
    }
}

impl From<Vector3> for [f32; 3] {
    fn from(v: Vector3) -> Self {
        v.to_array()
    }
}

impl From<(f32, f32, f32)> for Vector3 {
    fn from((x, y, z): (f32, f32, f32)) -> Self {
        Self { x, y, z }
    }
}

impl From<glam::Vec3> for Vector3 {
    fn from(v: glam::Vec3) -> Self {
        Self { x: v.x, y: v.y, z: v.z }
    }
}

impl From<Vector3> for glam::Vec3 {
    fn from(v: Vector3) -> Self {
        glam::Vec3::new(v.x, v.y, v.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross() {
        let x = Vector3::UNIT_X;
        let y = Vector3::UNIT_Y;
        let z = x.cross(&y);
        assert!(z.approx_eq(&Vector3::UNIT_Z, 1e-6));
    }

    #[test]
    fn test_reflect() {
        let v = Vector3::new(1.0, -1.0, 0.0);
        let n = Vector3::UNIT_Y;
        let r = v.reflect(&n);
        assert!(r.approx_eq(&Vector3::new(1.0, 1.0, 0.0), 1e-6));
    }
}
