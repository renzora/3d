//! Euler angles implementation.

use super::{Matrix4, Quaternion, Vector3};
use serde::{Deserialize, Serialize};

/// Order of Euler angle rotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum EulerOrder {
    /// X, then Y, then Z.
    #[default]
    XYZ,
    /// Y, then X, then Z.
    YXZ,
    /// Z, then X, then Y.
    ZXY,
    /// Z, then Y, then X.
    ZYX,
    /// Y, then Z, then X.
    YZX,
    /// X, then Z, then Y.
    XZY,
}

/// Euler angles representation of rotation.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Euler {
    /// Rotation around X axis in radians.
    pub x: f32,
    /// Rotation around Y axis in radians.
    pub y: f32,
    /// Rotation around Z axis in radians.
    pub z: f32,
    /// Order of rotations.
    pub order: EulerOrder,
}

impl Euler {
    /// Zero rotation.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        order: EulerOrder::XYZ,
    };

    /// Create new Euler angles.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, order: EulerOrder) -> Self {
        Self { x, y, z, order }
    }

    /// Create with default XYZ order.
    #[inline]
    pub const fn xyz(x: f32, y: f32, z: f32) -> Self {
        Self {
            x,
            y,
            z,
            order: EulerOrder::XYZ,
        }
    }

    /// Set the components.
    #[inline]
    pub fn set(&mut self, x: f32, y: f32, z: f32, order: EulerOrder) -> &mut Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self.order = order;
        self
    }

    /// Copy from another Euler.
    #[inline]
    pub fn copy(&mut self, e: &Euler) -> &mut Self {
        self.x = e.x;
        self.y = e.y;
        self.z = e.z;
        self.order = e.order;
        self
    }

    /// Create from a rotation matrix.
    pub fn from_matrix4(m: &Matrix4, order: EulerOrder) -> Self {
        let e = &m.elements;

        // Get scale to handle non-uniform scaling
        let sx = Vector3::new(e[0], e[1], e[2]).length();
        let sy = Vector3::new(e[4], e[5], e[6]).length();
        let sz = Vector3::new(e[8], e[9], e[10]).length();

        let m11 = e[0] / sx;
        let m12 = e[4] / sy;
        let m13 = e[8] / sz;
        let m21 = e[1] / sx;
        let m22 = e[5] / sy;
        let m23 = e[9] / sz;
        let m31 = e[2] / sx;
        let m32 = e[6] / sy;
        let m33 = e[10] / sz;

        Self::from_rotation_matrix_elements(m11, m12, m13, m21, m22, m23, m31, m32, m33, order)
    }

    /// Create from rotation matrix elements.
    #[allow(clippy::too_many_arguments)]
    pub fn from_rotation_matrix_elements(
        m11: f32, m12: f32, m13: f32,
        m21: f32, m22: f32, m23: f32,
        m31: f32, m32: f32, m33: f32,
        order: EulerOrder,
    ) -> Self {
        let clamp = |x: f32| x.clamp(-1.0, 1.0);

        match order {
            EulerOrder::XYZ => {
                let y = clamp(m13).asin();
                let (x, z) = if m13.abs() < 0.9999999 {
                    ((-m23).atan2(m33), (-m12).atan2(m11))
                } else {
                    (m32.atan2(m22), 0.0)
                };
                Self { x, y, z, order }
            }
            EulerOrder::YXZ => {
                let x = clamp(-m23).asin();
                let (y, z) = if m23.abs() < 0.9999999 {
                    (m13.atan2(m33), m21.atan2(m22))
                } else {
                    ((-m31).atan2(m11), 0.0)
                };
                Self { x, y, z, order }
            }
            EulerOrder::ZXY => {
                let x = clamp(m32).asin();
                let (y, z) = if m32.abs() < 0.9999999 {
                    ((-m31).atan2(m33), (-m12).atan2(m22))
                } else {
                    (0.0, m21.atan2(m11))
                };
                Self { x, y, z, order }
            }
            EulerOrder::ZYX => {
                let y = clamp(-m31).asin();
                let (x, z) = if m31.abs() < 0.9999999 {
                    (m32.atan2(m33), m21.atan2(m11))
                } else {
                    (0.0, (-m12).atan2(m22))
                };
                Self { x, y, z, order }
            }
            EulerOrder::YZX => {
                let z = clamp(m21).asin();
                let (x, y) = if m21.abs() < 0.9999999 {
                    ((-m23).atan2(m22), (-m31).atan2(m11))
                } else {
                    (0.0, m13.atan2(m33))
                };
                Self { x, y, z, order }
            }
            EulerOrder::XZY => {
                let z = clamp(-m12).asin();
                let (x, y) = if m12.abs() < 0.9999999 {
                    (m32.atan2(m22), m13.atan2(m11))
                } else {
                    ((-m23).atan2(m33), 0.0)
                };
                Self { x, y, z, order }
            }
        }
    }

    /// Create from a quaternion.
    pub fn from_quaternion(q: &Quaternion, order: EulerOrder) -> Self {
        let m = Matrix4::from_quaternion(q);
        Self::from_matrix4(&m, order)
    }

    /// Convert to a Vector3 (x, y, z angles).
    #[inline]
    pub const fn to_vector3(&self) -> Vector3 {
        Vector3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    /// Reorder to a different rotation order.
    pub fn reorder(&self, new_order: EulerOrder) -> Self {
        let q = Quaternion::from_euler(self);
        Self::from_quaternion(&q, new_order)
    }

    /// Check if approximately equal.
    #[inline]
    pub fn approx_eq(&self, other: &Euler, epsilon: f32) -> bool {
        (self.x - other.x).abs() < epsilon
            && (self.y - other.y).abs() < epsilon
            && (self.z - other.z).abs() < epsilon
            && self.order == other.order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_roundtrip() {
        let euler = Euler::xyz(0.1, 0.2, 0.3);
        let q = Quaternion::from_euler(&euler);
        let euler2 = Euler::from_quaternion(&q, EulerOrder::XYZ);
        assert!(euler.approx_eq(&euler2, 1e-5));
    }
}
