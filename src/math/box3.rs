//! Axis-aligned bounding box implementation.

use super::{Matrix4, Plane, Sphere, Vector3};
use serde::{Deserialize, Serialize};

/// An axis-aligned bounding box (AABB).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Box3 {
    /// Minimum corner.
    pub min: Vector3,
    /// Maximum corner.
    pub max: Vector3,
}

impl Default for Box3 {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Box3 {
    /// Empty box (inverted, ready to expand).
    pub const EMPTY: Self = Self {
        min: Vector3 { x: f32::INFINITY, y: f32::INFINITY, z: f32::INFINITY },
        max: Vector3 { x: f32::NEG_INFINITY, y: f32::NEG_INFINITY, z: f32::NEG_INFINITY },
    };

    /// Unit box centered at origin.
    pub const UNIT: Self = Self {
        min: Vector3 { x: -0.5, y: -0.5, z: -0.5 },
        max: Vector3 { x: 0.5, y: 0.5, z: 0.5 },
    };

    /// Create a new box.
    #[inline]
    pub const fn new(min: Vector3, max: Vector3) -> Self {
        Self { min, max }
    }

    /// Create a box from center and size.
    pub fn from_center_size(center: Vector3, size: Vector3) -> Self {
        let half = size * 0.5;
        Self {
            min: center - half,
            max: center + half,
        }
    }

    /// Create a box from an array of points.
    pub fn from_points(points: &[Vector3]) -> Self {
        let mut result = Self::EMPTY;
        for p in points {
            result.expand_by_point(p);
        }
        result
    }

    /// Set the box corners.
    #[inline]
    pub fn set(&mut self, min: Vector3, max: Vector3) -> &mut Self {
        self.min = min;
        self.max = max;
        self
    }

    /// Copy from another box.
    #[inline]
    pub fn copy(&mut self, b: &Box3) -> &mut Self {
        self.min = b.min;
        self.max = b.max;
        self
    }

    /// Make the box empty.
    #[inline]
    pub fn make_empty(&mut self) -> &mut Self {
        *self = Self::EMPTY;
        self
    }

    /// Check if the box is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.max.x < self.min.x || self.max.y < self.min.y || self.max.z < self.min.z
    }

    /// Get the center of the box.
    #[inline]
    pub fn center(&self) -> Vector3 {
        if self.is_empty() {
            Vector3::ZERO
        } else {
            (self.min + self.max) * 0.5
        }
    }

    /// Get the size of the box.
    #[inline]
    pub fn size(&self) -> Vector3 {
        if self.is_empty() {
            Vector3::ZERO
        } else {
            self.max - self.min
        }
    }

    /// Expand to include a point.
    #[inline]
    pub fn expand_by_point(&mut self, point: &Vector3) -> &mut Self {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
        self
    }

    /// Expand by a scalar amount in all directions.
    #[inline]
    pub fn expand_by_scalar(&mut self, scalar: f32) -> &mut Self {
        self.min = self.min - Vector3::splat(scalar);
        self.max = self.max + Vector3::splat(scalar);
        self
    }

    /// Expand by a vector amount.
    #[inline]
    pub fn expand_by_vector(&mut self, v: &Vector3) -> &mut Self {
        self.min = self.min - *v;
        self.max = self.max + *v;
        self
    }

    /// Check if a point is inside the box.
    #[inline]
    pub fn contains_point(&self, point: &Vector3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x
            && point.y >= self.min.y && point.y <= self.max.y
            && point.z >= self.min.z && point.z <= self.max.z
    }

    /// Check if this box fully contains another box.
    #[inline]
    pub fn contains_box(&self, other: &Box3) -> bool {
        self.min.x <= other.min.x && other.max.x <= self.max.x
            && self.min.y <= other.min.y && other.max.y <= self.max.y
            && self.min.z <= other.min.z && other.max.z <= self.max.z
    }

    /// Check if this box intersects another box.
    #[inline]
    pub fn intersects_box(&self, other: &Box3) -> bool {
        other.max.x >= self.min.x && other.min.x <= self.max.x
            && other.max.y >= self.min.y && other.min.y <= self.max.y
            && other.max.z >= self.min.z && other.min.z <= self.max.z
    }

    /// Check if this box intersects a sphere.
    #[inline]
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        let closest = self.clamp_point(&sphere.center);
        closest.distance_to_squared(&sphere.center) <= sphere.radius * sphere.radius
    }

    /// Check if this box intersects a plane.
    pub fn intersects_plane(&self, plane: &Plane) -> bool {
        // Compute the extent in the direction of the plane normal
        let center = self.center();
        let half_size = self.size() * 0.5;
        let r = half_size.x * plane.normal.x.abs()
            + half_size.y * plane.normal.y.abs()
            + half_size.z * plane.normal.z.abs();

        let d = plane.distance_to_point(&center);
        d.abs() <= r
    }

    /// Clamp a point to the box.
    #[inline]
    pub fn clamp_point(&self, point: &Vector3) -> Vector3 {
        point.clamp(&self.min, &self.max)
    }

    /// Get the distance from a point to the box.
    #[inline]
    pub fn distance_to_point(&self, point: &Vector3) -> f32 {
        self.clamp_point(point).distance_to(point)
    }

    /// Get the bounding sphere of this box.
    pub fn bounding_sphere(&self) -> Sphere {
        Sphere::from_box3(self)
    }

    /// Get the intersection of this box with another.
    pub fn intersection(&self, other: &Box3) -> Self {
        Self {
            min: self.min.max(&other.min),
            max: self.max.min(&other.max),
        }
    }

    /// Get the union of this box with another.
    pub fn union(&self, other: &Box3) -> Self {
        Self {
            min: self.min.min(&other.min),
            max: self.max.max(&other.max),
        }
    }

    /// Apply a Matrix4 transformation.
    /// Note: This returns the AABB of the transformed box, not an OBB.
    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }

        // Get the 8 corners of the box
        let corners = [
            Vector3::new(self.min.x, self.min.y, self.min.z),
            Vector3::new(self.min.x, self.min.y, self.max.z),
            Vector3::new(self.min.x, self.max.y, self.min.z),
            Vector3::new(self.min.x, self.max.y, self.max.z),
            Vector3::new(self.max.x, self.min.y, self.min.z),
            Vector3::new(self.max.x, self.min.y, self.max.z),
            Vector3::new(self.max.x, self.max.y, self.min.z),
            Vector3::new(self.max.x, self.max.y, self.max.z),
        ];

        let mut result = Self::EMPTY;
        for corner in &corners {
            let transformed = m.transform_point(corner);
            result.expand_by_point(&transformed);
        }
        result
    }

    /// Translate the box.
    #[inline]
    pub fn translate(&self, offset: &Vector3) -> Self {
        Self {
            min: self.min + *offset,
            max: self.max + *offset,
        }
    }

    /// Get the 8 corners of the box.
    pub fn corners(&self) -> [Vector3; 8] {
        [
            Vector3::new(self.min.x, self.min.y, self.min.z),
            Vector3::new(self.min.x, self.min.y, self.max.z),
            Vector3::new(self.min.x, self.max.y, self.min.z),
            Vector3::new(self.min.x, self.max.y, self.max.z),
            Vector3::new(self.max.x, self.min.y, self.min.z),
            Vector3::new(self.max.x, self.min.y, self.max.z),
            Vector3::new(self.max.x, self.max.y, self.min.z),
            Vector3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// Check if approximately equal.
    #[inline]
    pub fn approx_eq(&self, other: &Box3, epsilon: f32) -> bool {
        self.min.approx_eq(&other.min, epsilon) && self.max.approx_eq(&other.max, epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_size() {
        let b = Box3::from_center_size(Vector3::ZERO, Vector3::ONE);
        assert!(b.center().approx_eq(&Vector3::ZERO, 1e-6));
        assert!(b.size().approx_eq(&Vector3::ONE, 1e-6));
    }

    #[test]
    fn test_contains() {
        let b = Box3::new(Vector3::ZERO, Vector3::ONE);
        assert!(b.contains_point(&Vector3::splat(0.5)));
        assert!(!b.contains_point(&Vector3::splat(2.0)));
    }

    #[test]
    fn test_expand() {
        let mut b = Box3::EMPTY;
        b.expand_by_point(&Vector3::ZERO);
        b.expand_by_point(&Vector3::ONE);
        assert!(b.min.approx_eq(&Vector3::ZERO, 1e-6));
        assert!(b.max.approx_eq(&Vector3::ONE, 1e-6));
    }
}
