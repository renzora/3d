//! Quaternion implementation for rotations.

use super::{Euler, EulerOrder, Matrix4, Vector3};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::ops::{Mul, MulAssign};

/// A quaternion representing a rotation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Quaternion {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
    /// W component (scalar).
    pub w: f32,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quaternion {
    /// Identity quaternion (no rotation).
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    /// Create a new quaternion.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Create from an array [x, y, z, w].
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self { x: a[0], y: a[1], z: a[2], w: a[3] }
    }

    /// Convert to an array [x, y, z, w].
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Set the components.
    #[inline]
    pub fn set(&mut self, x: f32, y: f32, z: f32, w: f32) -> &mut Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
        self
    }

    /// Copy from another quaternion.
    #[inline]
    pub fn copy(&mut self, q: &Quaternion) -> &mut Self {
        self.x = q.x;
        self.y = q.y;
        self.z = q.z;
        self.w = q.w;
        self
    }

    /// Set to identity.
    #[inline]
    pub fn set_identity(&mut self) -> &mut Self {
        *self = Self::IDENTITY;
        self
    }

    /// Create a quaternion from Euler angles.
    pub fn from_euler(euler: &Euler) -> Self {
        let c1 = (euler.x / 2.0).cos();
        let c2 = (euler.y / 2.0).cos();
        let c3 = (euler.z / 2.0).cos();
        let s1 = (euler.x / 2.0).sin();
        let s2 = (euler.y / 2.0).sin();
        let s3 = (euler.z / 2.0).sin();

        match euler.order {
            EulerOrder::XYZ => Self {
                x: s1 * c2 * c3 + c1 * s2 * s3,
                y: c1 * s2 * c3 - s1 * c2 * s3,
                z: c1 * c2 * s3 + s1 * s2 * c3,
                w: c1 * c2 * c3 - s1 * s2 * s3,
            },
            EulerOrder::YXZ => Self {
                x: s1 * c2 * c3 + c1 * s2 * s3,
                y: c1 * s2 * c3 - s1 * c2 * s3,
                z: c1 * c2 * s3 - s1 * s2 * c3,
                w: c1 * c2 * c3 + s1 * s2 * s3,
            },
            EulerOrder::ZXY => Self {
                x: s1 * c2 * c3 - c1 * s2 * s3,
                y: c1 * s2 * c3 + s1 * c2 * s3,
                z: c1 * c2 * s3 + s1 * s2 * c3,
                w: c1 * c2 * c3 - s1 * s2 * s3,
            },
            EulerOrder::ZYX => Self {
                x: s1 * c2 * c3 - c1 * s2 * s3,
                y: c1 * s2 * c3 + s1 * c2 * s3,
                z: c1 * c2 * s3 - s1 * s2 * c3,
                w: c1 * c2 * c3 + s1 * s2 * s3,
            },
            EulerOrder::YZX => Self {
                x: s1 * c2 * c3 + c1 * s2 * s3,
                y: c1 * s2 * c3 + s1 * c2 * s3,
                z: c1 * c2 * s3 - s1 * s2 * c3,
                w: c1 * c2 * c3 - s1 * s2 * s3,
            },
            EulerOrder::XZY => Self {
                x: s1 * c2 * c3 - c1 * s2 * s3,
                y: c1 * s2 * c3 - s1 * c2 * s3,
                z: c1 * c2 * s3 + s1 * s2 * c3,
                w: c1 * c2 * c3 + s1 * s2 * s3,
            },
        }
    }

    /// Create a quaternion from axis-angle representation.
    pub fn from_axis_angle(axis: &Vector3, angle: f32) -> Self {
        let half_angle = angle / 2.0;
        let s = half_angle.sin();
        Self {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w: half_angle.cos(),
        }
    }

    /// Create from rotation matrix elements.
    #[allow(clippy::too_many_arguments)]
    pub fn from_rotation_matrix_elements(
        m00: f32, m01: f32, m02: f32,
        m10: f32, m11: f32, m12: f32,
        m20: f32, m21: f32, m22: f32,
    ) -> Self {
        let trace = m00 + m11 + m22;

        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            Self {
                w: 0.25 / s,
                x: (m21 - m12) * s,
                y: (m02 - m20) * s,
                z: (m10 - m01) * s,
            }
        } else if m00 > m11 && m00 > m22 {
            let s = 2.0 * (1.0 + m00 - m11 - m22).sqrt();
            Self {
                w: (m21 - m12) / s,
                x: 0.25 * s,
                y: (m01 + m10) / s,
                z: (m02 + m20) / s,
            }
        } else if m11 > m22 {
            let s = 2.0 * (1.0 + m11 - m00 - m22).sqrt();
            Self {
                w: (m02 - m20) / s,
                x: (m01 + m10) / s,
                y: 0.25 * s,
                z: (m12 + m21) / s,
            }
        } else {
            let s = 2.0 * (1.0 + m22 - m00 - m11).sqrt();
            Self {
                w: (m10 - m01) / s,
                x: (m02 + m20) / s,
                y: (m12 + m21) / s,
                z: 0.25 * s,
            }
        }
    }

    /// Create from a Matrix4 (extracts rotation).
    pub fn from_matrix4(m: &Matrix4) -> Self {
        let e = &m.elements;

        // Get scale to normalize
        let sx = Vector3::new(e[0], e[1], e[2]).length();
        let sy = Vector3::new(e[4], e[5], e[6]).length();
        let sz = Vector3::new(e[8], e[9], e[10]).length();

        let inv_sx = if sx > 0.0 { 1.0 / sx } else { 0.0 };
        let inv_sy = if sy > 0.0 { 1.0 / sy } else { 0.0 };
        let inv_sz = if sz > 0.0 { 1.0 / sz } else { 0.0 };

        Self::from_rotation_matrix_elements(
            e[0] * inv_sx, e[4] * inv_sy, e[8] * inv_sz,
            e[1] * inv_sx, e[5] * inv_sy, e[9] * inv_sz,
            e[2] * inv_sx, e[6] * inv_sy, e[10] * inv_sz,
        )
    }

    /// Create a quaternion that rotates from one direction to another.
    pub fn from_unit_vectors(from: &Vector3, to: &Vector3) -> Self {
        let r = from.dot(to) + 1.0;

        if r < 1e-6 {
            // Vectors are opposite
            if from.x.abs() > from.z.abs() {
                Self::new(-from.y, from.x, 0.0, 0.0).normalized()
            } else {
                Self::new(0.0, -from.z, from.y, 0.0).normalized()
            }
        } else {
            let cross = from.cross(to);
            Self::new(cross.x, cross.y, cross.z, r).normalized()
        }
    }

    /// Get the length of the quaternion.
    #[inline]
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Get the squared length.
    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Normalize the quaternion.
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

    /// Return a normalized copy.
    #[inline]
    pub fn normalized(&self) -> Self {
        let mut q = *self;
        q.normalize();
        q
    }

    /// Conjugate (inverse for unit quaternions).
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Invert the quaternion.
    #[inline]
    pub fn inverse(&self) -> Self {
        self.conjugate().normalized()
    }

    /// Dot product.
    #[inline]
    pub fn dot(&self, other: &Quaternion) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Multiply by another quaternion.
    pub fn multiply(&self, other: &Quaternion) -> Self {
        Self {
            x: self.x * other.w + self.w * other.x + self.y * other.z - self.z * other.y,
            y: self.y * other.w + self.w * other.y + self.z * other.x - self.x * other.z,
            z: self.z * other.w + self.w * other.z + self.x * other.y - self.y * other.x,
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        }
    }

    /// Pre-multiply by another quaternion.
    pub fn premultiply(&self, other: &Quaternion) -> Self {
        other.multiply(self)
    }

    /// Spherical linear interpolation.
    pub fn slerp(&self, other: &Quaternion, t: f32) -> Self {
        if t == 0.0 {
            return *self;
        }
        if t == 1.0 {
            return *other;
        }

        let mut cos_half_theta = self.dot(other);
        let mut other = *other;

        // Take shorter path
        if cos_half_theta < 0.0 {
            other = Self::new(-other.x, -other.y, -other.z, -other.w);
            cos_half_theta = -cos_half_theta;
        }

        if cos_half_theta >= 1.0 {
            return *self;
        }

        let half_theta = cos_half_theta.acos();
        let sin_half_theta = (1.0 - cos_half_theta * cos_half_theta).sqrt();

        if sin_half_theta.abs() < 0.001 {
            return Self {
                x: self.x * 0.5 + other.x * 0.5,
                y: self.y * 0.5 + other.y * 0.5,
                z: self.z * 0.5 + other.z * 0.5,
                w: self.w * 0.5 + other.w * 0.5,
            };
        }

        let ratio_a = ((1.0 - t) * half_theta).sin() / sin_half_theta;
        let ratio_b = (t * half_theta).sin() / sin_half_theta;

        Self {
            x: self.x * ratio_a + other.x * ratio_b,
            y: self.y * ratio_a + other.y * ratio_b,
            z: self.z * ratio_a + other.z * ratio_b,
            w: self.w * ratio_a + other.w * ratio_b,
        }
    }

    /// Rotate a vector by this quaternion.
    pub fn rotate_vector(&self, v: &Vector3) -> Vector3 {
        v.apply_quaternion(self)
    }

    /// Get the rotation angle in radians.
    pub fn angle(&self) -> f32 {
        2.0 * self.w.clamp(-1.0, 1.0).acos()
    }

    /// Get the rotation axis.
    pub fn axis(&self) -> Vector3 {
        let sin_half = (1.0 - self.w * self.w).sqrt();
        if sin_half < 0.0001 {
            Vector3::UNIT_X
        } else {
            Vector3::new(self.x / sin_half, self.y / sin_half, self.z / sin_half)
        }
    }

    /// Check if approximately equal.
    #[inline]
    pub fn approx_eq(&self, other: &Quaternion, epsilon: f32) -> bool {
        (self.x - other.x).abs() < epsilon
            && (self.y - other.y).abs() < epsilon
            && (self.z - other.z).abs() < epsilon
            && (self.w - other.w).abs() < epsilon
    }
}

impl Mul for Quaternion {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.multiply(&rhs)
    }
}

impl MulAssign for Quaternion {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.multiply(&rhs);
    }
}

impl Mul<Vector3> for Quaternion {
    type Output = Vector3;
    fn mul(self, rhs: Vector3) -> Vector3 {
        self.rotate_vector(&rhs)
    }
}

impl From<glam::Quat> for Quaternion {
    fn from(q: glam::Quat) -> Self {
        Self {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }
}

impl From<Quaternion> for glam::Quat {
    fn from(q: Quaternion) -> Self {
        glam::Quat::from_xyzw(q.x, q.y, q.z, q.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let q = Quaternion::IDENTITY;
        let v = Vector3::new(1.0, 2.0, 3.0);
        let result = q.rotate_vector(&v);
        assert!(result.approx_eq(&v, 1e-6));
    }

    #[test]
    fn test_axis_angle() {
        let q = Quaternion::from_axis_angle(&Vector3::UNIT_Y, std::f32::consts::PI);
        let v = Vector3::UNIT_X;
        let result = q.rotate_vector(&v);
        assert!(result.approx_eq(&-Vector3::UNIT_X, 1e-5));
    }
}
