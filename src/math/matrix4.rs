//! 4x4 Matrix implementation.

use super::{Euler, Quaternion, Vector3};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// A 4x4 matrix stored in column-major order.
/// Used for 3D transformations (model, view, projection matrices).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Matrix4 {
    /// Matrix elements in column-major order.
    /// [m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32, m03, m13, m23, m33]
    pub elements: [f32; 16],
}

impl Default for Matrix4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Matrix4 {
    /// Identity matrix.
    pub const IDENTITY: Self = Self {
        elements: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    };

    /// Zero matrix.
    pub const ZERO: Self = Self {
        elements: [0.0; 16],
    };

    /// Create a new Matrix4 from elements in row-major order.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        m00: f32, m01: f32, m02: f32, m03: f32,
        m10: f32, m11: f32, m12: f32, m13: f32,
        m20: f32, m21: f32, m22: f32, m23: f32,
        m30: f32, m31: f32, m32: f32, m33: f32,
    ) -> Self {
        Self {
            elements: [
                m00, m10, m20, m30,
                m01, m11, m21, m31,
                m02, m12, m22, m32,
                m03, m13, m23, m33,
            ],
        }
    }

    /// Create from column-major array.
    #[inline]
    pub const fn from_cols_array(elements: [f32; 16]) -> Self {
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

    /// Copy from another matrix.
    #[inline]
    pub fn copy(&mut self, m: &Matrix4) -> &mut Self {
        self.elements = m.elements;
        self
    }

    /// Extract the position (translation) component.
    #[inline]
    pub fn get_position(&self) -> Vector3 {
        Vector3 {
            x: self.elements[12],
            y: self.elements[13],
            z: self.elements[14],
        }
    }

    /// Set the position (translation) component.
    #[inline]
    pub fn set_position(&mut self, v: &Vector3) -> &mut Self {
        self.elements[12] = v.x;
        self.elements[13] = v.y;
        self.elements[14] = v.z;
        self
    }

    /// Extract scale from the matrix.
    pub fn get_scale(&self) -> Vector3 {
        let e = &self.elements;
        let sx = Vector3::new(e[0], e[1], e[2]).length();
        let sy = Vector3::new(e[4], e[5], e[6]).length();
        let sz = Vector3::new(e[8], e[9], e[10]).length();
        Vector3::new(sx, sy, sz)
    }

    /// Get the maximum scale component.
    pub fn get_max_scale(&self) -> f32 {
        let e = &self.elements;
        let sx2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        let sy2 = e[4] * e[4] + e[5] * e[5] + e[6] * e[6];
        let sz2 = e[8] * e[8] + e[9] * e[9] + e[10] * e[10];
        sx2.max(sy2).max(sz2).sqrt()
    }

    /// Compose a transformation matrix from position, quaternion, and scale.
    pub fn compose(position: &Vector3, quaternion: &Quaternion, scale: &Vector3) -> Self {
        let x2 = quaternion.x + quaternion.x;
        let y2 = quaternion.y + quaternion.y;
        let z2 = quaternion.z + quaternion.z;
        let xx = quaternion.x * x2;
        let xy = quaternion.x * y2;
        let xz = quaternion.x * z2;
        let yy = quaternion.y * y2;
        let yz = quaternion.y * z2;
        let zz = quaternion.z * z2;
        let wx = quaternion.w * x2;
        let wy = quaternion.w * y2;
        let wz = quaternion.w * z2;

        Self {
            elements: [
                (1.0 - (yy + zz)) * scale.x,
                (xy + wz) * scale.x,
                (xz - wy) * scale.x,
                0.0,
                (xy - wz) * scale.y,
                (1.0 - (xx + zz)) * scale.y,
                (yz + wx) * scale.y,
                0.0,
                (xz + wy) * scale.z,
                (yz - wx) * scale.z,
                (1.0 - (xx + yy)) * scale.z,
                0.0,
                position.x,
                position.y,
                position.z,
                1.0,
            ],
        }
    }

    /// Decompose the matrix into position, quaternion, and scale.
    pub fn decompose(&self) -> (Vector3, Quaternion, Vector3) {
        let e = &self.elements;

        let mut sx = Vector3::new(e[0], e[1], e[2]).length();
        let sy = Vector3::new(e[4], e[5], e[6]).length();
        let sz = Vector3::new(e[8], e[9], e[10]).length();

        // If determinant is negative, negate one scale
        if self.determinant() < 0.0 {
            sx = -sx;
        }

        let position = Vector3::new(e[12], e[13], e[14]);
        let scale = Vector3::new(sx, sy, sz);

        // Build rotation matrix without scale
        let inv_sx = 1.0 / sx;
        let inv_sy = 1.0 / sy;
        let inv_sz = 1.0 / sz;

        let m00 = e[0] * inv_sx;
        let m01 = e[4] * inv_sy;
        let m02 = e[8] * inv_sz;
        let m10 = e[1] * inv_sx;
        let m11 = e[5] * inv_sy;
        let m12 = e[9] * inv_sz;
        let m20 = e[2] * inv_sx;
        let m21 = e[6] * inv_sy;
        let m22 = e[10] * inv_sz;

        let quaternion = Quaternion::from_rotation_matrix_elements(
            m00, m01, m02,
            m10, m11, m12,
            m20, m21, m22,
        );

        (position, quaternion, scale)
    }

    /// Create a translation matrix.
    pub fn from_translation(v: &Vector3) -> Self {
        Self {
            elements: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                v.x, v.y, v.z, 1.0,
            ],
        }
    }

    /// Create a rotation matrix from a quaternion.
    pub fn from_quaternion(q: &Quaternion) -> Self {
        Self::compose(&Vector3::ZERO, q, &Vector3::ONE)
    }

    /// Create a rotation matrix from euler angles.
    pub fn from_euler(euler: &Euler) -> Self {
        Self::from_quaternion(&Quaternion::from_euler(euler))
    }

    /// Create a scale matrix.
    pub fn from_scale(v: &Vector3) -> Self {
        Self {
            elements: [
                v.x, 0.0, 0.0, 0.0,
                0.0, v.y, 0.0, 0.0,
                0.0, 0.0, v.z, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Create a rotation matrix around the X axis.
    pub fn from_rotation_x(theta: f32) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        Self {
            elements: [
                1.0, 0.0, 0.0, 0.0,
                0.0, c, s, 0.0,
                0.0, -s, c, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Create a rotation matrix around the Y axis.
    pub fn from_rotation_y(theta: f32) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        Self {
            elements: [
                c, 0.0, -s, 0.0,
                0.0, 1.0, 0.0, 0.0,
                s, 0.0, c, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Create a rotation matrix around the Z axis.
    pub fn from_rotation_z(theta: f32) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        Self {
            elements: [
                c, s, 0.0, 0.0,
                -s, c, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Create a rotation matrix around an arbitrary axis.
    pub fn from_axis_angle(axis: &Vector3, angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Self {
            elements: [
                t * x * x + c,
                t * x * y + s * z,
                t * x * z - s * y,
                0.0,
                t * x * y - s * z,
                t * y * y + c,
                t * y * z + s * x,
                0.0,
                t * x * z + s * y,
                t * y * z - s * x,
                t * z * z + c,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        }
    }

    /// Create a view matrix (look-at).
    /// Returns the inverse of the camera transform.
    pub fn look_at(eye: &Vector3, target: &Vector3, up: &Vector3) -> Self {
        let f = (*target - *eye).normalized(); // forward
        let r = f.cross(up).normalized();       // right
        let u = r.cross(&f);                    // up

        // View matrix is inverse of camera matrix
        // For orthonormal basis, inverse = transpose for rotation part
        // Translation is -dot(axis, eye) for each axis
        Self {
            elements: [
                r.x, u.x, -f.x, 0.0,
                r.y, u.y, -f.y, 0.0,
                r.z, u.z, -f.z, 0.0,
                -r.dot(eye), -u.dot(eye), f.dot(eye), 1.0,
            ],
        }
    }

    /// Create a perspective projection matrix.
    /// Uses wgpu/Vulkan depth range (0 to 1).
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let tan_half_fov = (fov_y / 2.0).tan();
        let f = 1.0 / tan_half_fov;

        // wgpu uses 0-1 depth range (not -1 to 1 like OpenGL)
        Self {
            elements: [
                f / aspect, 0.0, 0.0, 0.0,
                0.0, f, 0.0, 0.0,
                0.0, 0.0, far / (near - far), -1.0,
                0.0, 0.0, (near * far) / (near - far), 0.0,
            ],
        }
    }

    /// Create an orthographic projection matrix.
    /// Uses wgpu/Vulkan depth range (0 to 1).
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let w = 1.0 / (right - left);
        let h = 1.0 / (top - bottom);
        let d = 1.0 / (far - near);

        // wgpu uses 0-1 depth range (not -1 to 1 like OpenGL)
        Self {
            elements: [
                2.0 * w, 0.0, 0.0, 0.0,
                0.0, 2.0 * h, 0.0, 0.0,
                0.0, 0.0, -d, 0.0,
                -(right + left) * w, -(top + bottom) * h, -near * d, 1.0,
            ],
        }
    }

    /// Multiply this matrix by another.
    pub fn multiply(&self, other: &Matrix4) -> Self {
        let a = &self.elements;
        let b = &other.elements;

        Self {
            elements: [
                a[0] * b[0] + a[4] * b[1] + a[8] * b[2] + a[12] * b[3],
                a[1] * b[0] + a[5] * b[1] + a[9] * b[2] + a[13] * b[3],
                a[2] * b[0] + a[6] * b[1] + a[10] * b[2] + a[14] * b[3],
                a[3] * b[0] + a[7] * b[1] + a[11] * b[2] + a[15] * b[3],

                a[0] * b[4] + a[4] * b[5] + a[8] * b[6] + a[12] * b[7],
                a[1] * b[4] + a[5] * b[5] + a[9] * b[6] + a[13] * b[7],
                a[2] * b[4] + a[6] * b[5] + a[10] * b[6] + a[14] * b[7],
                a[3] * b[4] + a[7] * b[5] + a[11] * b[6] + a[15] * b[7],

                a[0] * b[8] + a[4] * b[9] + a[8] * b[10] + a[12] * b[11],
                a[1] * b[8] + a[5] * b[9] + a[9] * b[10] + a[13] * b[11],
                a[2] * b[8] + a[6] * b[9] + a[10] * b[10] + a[14] * b[11],
                a[3] * b[8] + a[7] * b[9] + a[11] * b[10] + a[15] * b[11],

                a[0] * b[12] + a[4] * b[13] + a[8] * b[14] + a[12] * b[15],
                a[1] * b[12] + a[5] * b[13] + a[9] * b[14] + a[13] * b[15],
                a[2] * b[12] + a[6] * b[13] + a[10] * b[14] + a[14] * b[15],
                a[3] * b[12] + a[7] * b[13] + a[11] * b[14] + a[15] * b[15],
            ],
        }
    }

    /// Pre-multiply this matrix by another.
    pub fn premultiply(&self, other: &Matrix4) -> Self {
        other.multiply(self)
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

        let n11 = e[0]; let n12 = e[4]; let n13 = e[8]; let n14 = e[12];
        let n21 = e[1]; let n22 = e[5]; let n23 = e[9]; let n24 = e[13];
        let n31 = e[2]; let n32 = e[6]; let n33 = e[10]; let n34 = e[14];
        let n41 = e[3]; let n42 = e[7]; let n43 = e[11]; let n44 = e[15];

        n41 * (n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34)
            + n42 * (n11 * n23 * n34 - n11 * n24 * n33 + n14 * n21 * n33 - n13 * n21 * n34 + n13 * n24 * n31 - n14 * n23 * n31)
            + n43 * (n11 * n24 * n32 - n11 * n22 * n34 - n14 * n21 * n32 + n12 * n21 * n34 + n14 * n22 * n31 - n12 * n24 * n31)
            + n44 * (-n13 * n22 * n31 - n11 * n23 * n32 + n11 * n22 * n33 + n13 * n21 * n32 - n12 * n21 * n33 + n12 * n23 * n31)
    }

    /// Invert this matrix.
    pub fn invert(&mut self) -> &mut Self {
        let e = &self.elements;

        let n11 = e[0]; let n12 = e[4]; let n13 = e[8]; let n14 = e[12];
        let n21 = e[1]; let n22 = e[5]; let n23 = e[9]; let n24 = e[13];
        let n31 = e[2]; let n32 = e[6]; let n33 = e[10]; let n34 = e[14];
        let n41 = e[3]; let n42 = e[7]; let n43 = e[11]; let n44 = e[15];

        let t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
        let t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
        let t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
        let t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

        let det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;

        if det == 0.0 {
            self.set_identity();
            return self;
        }

        let det_inv = 1.0 / det;

        self.elements = [
            t11 * det_inv,
            (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * det_inv,
            (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * det_inv,
            (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * det_inv,
            t12 * det_inv,
            (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * det_inv,
            (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * det_inv,
            (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * det_inv,
            t13 * det_inv,
            (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * det_inv,
            (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * det_inv,
            (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * det_inv,
            t14 * det_inv,
            (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * det_inv,
            (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * det_inv,
            (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * det_inv,
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
        self.elements.swap(1, 4);
        self.elements.swap(2, 8);
        self.elements.swap(3, 12);
        self.elements.swap(6, 9);
        self.elements.swap(7, 13);
        self.elements.swap(11, 14);
        self
    }

    /// Return the transpose of this matrix.
    pub fn transposed(&self) -> Self {
        let mut m = *self;
        m.transpose();
        m
    }

    /// Transform a Vector3 as a point (with translation).
    pub fn transform_point(&self, v: &Vector3) -> Vector3 {
        let e = &self.elements;
        let w = 1.0 / (e[3] * v.x + e[7] * v.y + e[11] * v.z + e[15]);
        Vector3 {
            x: (e[0] * v.x + e[4] * v.y + e[8] * v.z + e[12]) * w,
            y: (e[1] * v.x + e[5] * v.y + e[9] * v.z + e[13]) * w,
            z: (e[2] * v.x + e[6] * v.y + e[10] * v.z + e[14]) * w,
        }
    }

    /// Transform a Vector3 as a direction (without translation).
    pub fn transform_direction(&self, v: &Vector3) -> Vector3 {
        let e = &self.elements;
        Vector3 {
            x: e[0] * v.x + e[4] * v.y + e[8] * v.z,
            y: e[1] * v.x + e[5] * v.y + e[9] * v.z,
            z: e[2] * v.x + e[6] * v.y + e[10] * v.z,
        }
    }

    /// Check if approximately equal to another matrix.
    pub fn approx_eq(&self, other: &Matrix4, epsilon: f32) -> bool {
        self.elements.iter()
            .zip(other.elements.iter())
            .all(|(a, b)| (a - b).abs() < epsilon)
    }

    /// Convert to column-major 2D array (for GPU uniform buffers).
    pub fn to_cols_array_2d(&self) -> [[f32; 4]; 4] {
        let e = &self.elements;
        [
            [e[0], e[1], e[2], e[3]],
            [e[4], e[5], e[6], e[7]],
            [e[8], e[9], e[10], e[11]],
            [e[12], e[13], e[14], e[15]],
        ]
    }
}

impl std::ops::Mul for Matrix4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.multiply(&rhs)
    }
}

impl std::ops::Mul<Vector3> for Matrix4 {
    type Output = Vector3;
    fn mul(self, rhs: Vector3) -> Vector3 {
        self.transform_point(&rhs)
    }
}

impl From<glam::Mat4> for Matrix4 {
    fn from(m: glam::Mat4) -> Self {
        Self {
            elements: m.to_cols_array(),
        }
    }
}

impl From<Matrix4> for glam::Mat4 {
    fn from(m: Matrix4) -> Self {
        glam::Mat4::from_cols_array(&m.elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let m = Matrix4::IDENTITY;
        let v = Vector3::new(1.0, 2.0, 3.0);
        let result = m.transform_point(&v);
        assert!(result.approx_eq(&v, 1e-6));
    }

    #[test]
    fn test_translation() {
        let m = Matrix4::from_translation(&Vector3::new(10.0, 20.0, 30.0));
        let v = Vector3::ZERO;
        let result = m.transform_point(&v);
        assert!(result.approx_eq(&Vector3::new(10.0, 20.0, 30.0), 1e-6));
    }

    #[test]
    fn test_inverse() {
        let m = Matrix4::from_translation(&Vector3::new(1.0, 2.0, 3.0));
        let inv = m.inverse();
        let result = m.multiply(&inv);
        assert!(result.approx_eq(&Matrix4::IDENTITY, 1e-6));
    }
}
