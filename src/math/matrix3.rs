//! 3x3 Matrix implementation.

use super::{Matrix4, Vector3};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// A 3x3 matrix stored in column-major order.
/// Used for normal transformations and 2D transforms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Matrix3 {
    /// Matrix elements in column-major order.
    /// [m00, m10, m20, m01, m11, m21, m02, m12, m22]
    pub elements: [f32; 9],
}

impl Default for Matrix3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Matrix3 {
    /// Identity matrix.
    pub const IDENTITY: Self = Self {
        elements: [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
    };

    /// Zero matrix.
    pub const ZERO: Self = Self {
        elements: [0.0; 9],
    };

    /// Create a new Matrix3 from elements in row-major order.
    #[inline]
    pub const fn new(
        m00: f32, m01: f32, m02: f32,
        m10: f32, m11: f32, m12: f32,
        m20: f32, m21: f32, m22: f32,
    ) -> Self {
        Self {
            elements: [
                m00, m10, m20,
                m01, m11, m21,
                m02, m12, m22,
            ],
        }
    }

    /// Create from column-major array.
    #[inline]
    pub const fn from_cols_array(elements: [f32; 9]) -> Self {
        Self { elements }
    }

    /// Create identity matrix.
    #[inline]
    pub fn identity() -> Self {
        Self::IDENTITY
    }

    /// Set to identity matrix.
    #[inline]
    pub fn set_identity(&mut self) -> &mut Self {
        self.elements = Self::IDENTITY.elements;
        self
    }

    /// Set all elements.
    #[inline]
    pub fn set(
        &mut self,
        m00: f32, m01: f32, m02: f32,
        m10: f32, m11: f32, m12: f32,
        m20: f32, m21: f32, m22: f32,
    ) -> &mut Self {
        let e = &mut self.elements;
        e[0] = m00; e[3] = m01; e[6] = m02;
        e[1] = m10; e[4] = m11; e[7] = m12;
        e[2] = m20; e[5] = m21; e[8] = m22;
        self
    }

    /// Extract the normal matrix from a Matrix4 (inverse transpose of upper-left 3x3).
    pub fn from_matrix4_normal(m: &Matrix4) -> Self {
        let me = &m.elements;

        let a00 = me[0]; let a01 = me[4]; let a02 = me[8];
        let a10 = me[1]; let a11 = me[5]; let a12 = me[9];
        let a20 = me[2]; let a21 = me[6]; let a22 = me[10];

        let b01 = a22 * a11 - a12 * a21;
        let b11 = -a22 * a10 + a12 * a20;
        let b21 = a21 * a10 - a11 * a20;

        let det = a00 * b01 + a01 * b11 + a02 * b21;

        if det == 0.0 {
            return Self::IDENTITY;
        }

        let inv_det = 1.0 / det;

        Self {
            elements: [
                b01 * inv_det,
                (-a22 * a01 + a02 * a21) * inv_det,
                (a12 * a01 - a02 * a11) * inv_det,
                b11 * inv_det,
                (a22 * a00 - a02 * a20) * inv_det,
                (-a12 * a00 + a02 * a10) * inv_det,
                b21 * inv_det,
                (-a21 * a00 + a01 * a20) * inv_det,
                (a11 * a00 - a01 * a10) * inv_det,
            ],
        }
    }

    /// Extract upper-left 3x3 from a Matrix4.
    pub fn from_matrix4(m: &Matrix4) -> Self {
        let me = &m.elements;
        Self {
            elements: [
                me[0], me[1], me[2],
                me[4], me[5], me[6],
                me[8], me[9], me[10],
            ],
        }
    }

    /// Multiply this matrix by another.
    pub fn multiply(&self, other: &Matrix3) -> Self {
        let a = &self.elements;
        let b = &other.elements;

        Self {
            elements: [
                a[0] * b[0] + a[3] * b[1] + a[6] * b[2],
                a[1] * b[0] + a[4] * b[1] + a[7] * b[2],
                a[2] * b[0] + a[5] * b[1] + a[8] * b[2],

                a[0] * b[3] + a[3] * b[4] + a[6] * b[5],
                a[1] * b[3] + a[4] * b[4] + a[7] * b[5],
                a[2] * b[3] + a[5] * b[4] + a[8] * b[5],

                a[0] * b[6] + a[3] * b[7] + a[6] * b[8],
                a[1] * b[6] + a[4] * b[7] + a[7] * b[8],
                a[2] * b[6] + a[5] * b[7] + a[8] * b[8],
            ],
        }
    }

    /// Multiply by a scalar.
    pub fn multiply_scalar(&mut self, s: f32) -> &mut Self {
        for e in &mut self.elements {
            *e *= s;
        }
        self
    }

    /// Calculate the determinant.
    pub fn determinant(&self) -> f32 {
        let e = &self.elements;
        e[0] * (e[4] * e[8] - e[5] * e[7])
            - e[3] * (e[1] * e[8] - e[2] * e[7])
            + e[6] * (e[1] * e[5] - e[2] * e[4])
    }

    /// Invert this matrix.
    pub fn invert(&mut self) -> &mut Self {
        let e = &self.elements;
        let a00 = e[0]; let a01 = e[3]; let a02 = e[6];
        let a10 = e[1]; let a11 = e[4]; let a12 = e[7];
        let a20 = e[2]; let a21 = e[5]; let a22 = e[8];

        let b01 = a22 * a11 - a12 * a21;
        let b11 = -a22 * a10 + a12 * a20;
        let b21 = a21 * a10 - a11 * a20;

        let det = a00 * b01 + a01 * b11 + a02 * b21;

        if det == 0.0 {
            self.set_identity();
            return self;
        }

        let inv_det = 1.0 / det;

        self.elements = [
            b01 * inv_det,
            b11 * inv_det,
            b21 * inv_det,
            (-a22 * a01 + a02 * a21) * inv_det,
            (a22 * a00 - a02 * a20) * inv_det,
            (-a21 * a00 + a01 * a20) * inv_det,
            (a12 * a01 - a02 * a11) * inv_det,
            (-a12 * a00 + a02 * a10) * inv_det,
            (a11 * a00 - a01 * a10) * inv_det,
        ];

        self
    }

    /// Return the inverse of this matrix.
    pub fn inverse(&self) -> Self {
        let mut m = *self;
        m.invert();
        m
    }

    /// Transpose this matrix.
    pub fn transpose(&mut self) -> &mut Self {
        self.elements.swap(1, 3);
        self.elements.swap(2, 6);
        self.elements.swap(5, 7);
        self
    }

    /// Return the transpose of this matrix.
    pub fn transposed(&self) -> Self {
        let mut m = *self;
        m.transpose();
        m
    }

    /// Transform a Vector3 by this matrix.
    pub fn transform_vector(&self, v: &Vector3) -> Vector3 {
        let e = &self.elements;
        Vector3 {
            x: e[0] * v.x + e[3] * v.y + e[6] * v.z,
            y: e[1] * v.x + e[4] * v.y + e[7] * v.z,
            z: e[2] * v.x + e[5] * v.y + e[8] * v.z,
        }
    }

    /// Create a 2D rotation matrix.
    pub fn from_rotation(theta: f32) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        Self {
            elements: [
                c, s, 0.0,
                -s, c, 0.0,
                0.0, 0.0, 1.0,
            ],
        }
    }

    /// Create a 2D scale matrix.
    pub fn from_scale(sx: f32, sy: f32) -> Self {
        Self {
            elements: [
                sx, 0.0, 0.0,
                0.0, sy, 0.0,
                0.0, 0.0, 1.0,
            ],
        }
    }

    /// Create a 2D translation matrix.
    pub fn from_translation(tx: f32, ty: f32) -> Self {
        Self {
            elements: [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                tx, ty, 1.0,
            ],
        }
    }

    /// Check if approximately equal to another matrix.
    pub fn approx_eq(&self, other: &Matrix3, epsilon: f32) -> bool {
        self.elements.iter()
            .zip(other.elements.iter())
            .all(|(a, b)| (a - b).abs() < epsilon)
    }
}

impl std::ops::Mul for Matrix3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.multiply(&rhs)
    }
}

impl std::ops::Mul<Vector3> for Matrix3 {
    type Output = Vector3;
    fn mul(self, rhs: Vector3) -> Vector3 {
        self.transform_vector(&rhs)
    }
}

impl From<glam::Mat3> for Matrix3 {
    fn from(m: glam::Mat3) -> Self {
        Self {
            elements: m.to_cols_array(),
        }
    }
}

impl From<Matrix3> for glam::Mat3 {
    fn from(m: Matrix3) -> Self {
        glam::Mat3::from_cols_array(&m.elements)
    }
}
