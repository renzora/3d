//! Plane implementation.

use super::{Matrix4, Vector3};
use serde::{Deserialize, Serialize};

/// An infinite plane defined by a normal and constant (distance from origin).
/// The plane equation is: normal · point + constant = 0
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Plane {
    /// Normal vector of the plane (should be normalized).
    pub normal: Vector3,
    /// Distance from origin (negative of d in ax + by + cz + d = 0).
    pub constant: f32,
}

impl Plane {
    /// XY plane (normal pointing +Z).
    pub const XY: Self = Self { normal: Vector3::UNIT_Z, constant: 0.0 };
    /// XZ plane (normal pointing +Y).
    pub const XZ: Self = Self { normal: Vector3::UNIT_Y, constant: 0.0 };
    /// YZ plane (normal pointing +X).
    pub const YZ: Self = Self { normal: Vector3::UNIT_X, constant: 0.0 };

    /// Create a new plane.
    #[inline]
    pub const fn new(normal: Vector3, constant: f32) -> Self {
        Self { normal, constant }
    }

    /// Create a plane from normal and a point on the plane.
    #[inline]
    pub fn from_normal_and_point(normal: Vector3, point: &Vector3) -> Self {
        let n = normal.normalized();
        Self {
            normal: n,
            constant: -point.dot(&n),
        }
    }

    /// Create a plane from three coplanar points.
    pub fn from_coplanar_points(a: &Vector3, b: &Vector3, c: &Vector3) -> Self {
        let normal = (*c - *b).cross(&(*a - *b)).normalized();
        Self::from_normal_and_point(normal, a)
    }

    /// Set the plane components.
    #[inline]
    pub fn set(&mut self, normal: Vector3, constant: f32) -> &mut Self {
        self.normal = normal;
        self.constant = constant;
        self
    }

    /// Set from components (a, b, c, d) where ax + by + cz + d = 0.
    #[inline]
    pub fn set_components(&mut self, a: f32, b: f32, c: f32, d: f32) -> &mut Self {
        self.normal = Vector3::new(a, b, c);
        self.constant = d;
        self
    }

    /// Copy from another plane.
    #[inline]
    pub fn copy(&mut self, p: &Plane) -> &mut Self {
        self.normal = p.normal;
        self.constant = p.constant;
        self
    }

    /// Normalize the plane (ensure normal is unit length).
    pub fn normalize(&mut self) -> &mut Self {
        let inv_len = 1.0 / self.normal.length();
        self.normal.x *= inv_len;
        self.normal.y *= inv_len;
        self.normal.z *= inv_len;
        self.constant *= inv_len;
        self
    }

    /// Return a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut p = *self;
        p.normalize();
        p
    }

    /// Negate the plane (flip normal and constant).
    pub fn negate(&mut self) -> &mut Self {
        self.normal = -self.normal;
        self.constant = -self.constant;
        self
    }

    /// Return the negated plane.
    pub fn negated(&self) -> Self {
        Self {
            normal: -self.normal,
            constant: -self.constant,
        }
    }

    /// Get signed distance from a point to the plane.
    /// Positive = point is on the normal side.
    #[inline]
    pub fn distance_to_point(&self, point: &Vector3) -> f32 {
        self.normal.dot(point) + self.constant
    }

    /// Get the projection of a point onto the plane.
    #[inline]
    pub fn project_point(&self, point: &Vector3) -> Vector3 {
        *point - self.normal * self.distance_to_point(point)
    }

    /// Check if a point is on the positive side of the plane.
    #[inline]
    pub fn is_point_in_front(&self, point: &Vector3) -> bool {
        self.distance_to_point(point) > 0.0
    }

    /// Apply a Matrix4 transformation to the plane.
    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        // Transform a point on the plane
        let point_on_plane = self.normal * (-self.constant);
        let transformed_point = m.transform_point(&point_on_plane);

        // Transform the normal using the inverse transpose
        let normal_matrix = m.inverse().transposed();
        let transformed_normal = normal_matrix.transform_direction(&self.normal).normalized();

        Self::from_normal_and_point(transformed_normal, &transformed_point)
    }

    /// Check if approximately equal.
    #[inline]
    pub fn approx_eq(&self, other: &Plane, epsilon: f32) -> bool {
        self.normal.approx_eq(&other.normal, epsilon)
            && (self.constant - other.constant).abs() < epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_to_point() {
        let plane = Plane::new(Vector3::UNIT_Y, 0.0);
        assert!((plane.distance_to_point(&Vector3::new(0.0, 5.0, 0.0)) - 5.0).abs() < 1e-6);
        assert!((plane.distance_to_point(&Vector3::new(0.0, -3.0, 0.0)) + 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_project_point() {
        let plane = Plane::new(Vector3::UNIT_Y, 0.0);
        let projected = plane.project_point(&Vector3::new(1.0, 5.0, 2.0));
        assert!(projected.approx_eq(&Vector3::new(1.0, 0.0, 2.0), 1e-6));
    }
}
