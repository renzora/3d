//! Bounding sphere implementation.

use super::{Box3, Matrix4, Plane, Vector3};
use serde::{Deserialize, Serialize};

/// A bounding sphere defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Sphere {
    /// Center of the sphere.
    pub center: Vector3,
    /// Radius of the sphere.
    pub radius: f32,
}

impl Sphere {
    /// Unit sphere at origin.
    pub const UNIT: Self = Self { center: Vector3::ZERO, radius: 1.0 };

    /// Create a new sphere.
    #[inline]
    pub const fn new(center: Vector3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Create a sphere that bounds an array of points.
    pub fn from_points(points: &[Vector3]) -> Self {
        if points.is_empty() {
            return Self::default();
        }

        // First find bounding box center
        let mut min = points[0];
        let mut max = points[0];
        for p in points.iter().skip(1) {
            min = min.min(p);
            max = max.max(p);
        }
        let center = (min + max) * 0.5;

        // Then find max distance from center
        let mut max_dist_sq = 0.0_f32;
        for p in points {
            max_dist_sq = max_dist_sq.max(center.distance_to_squared(p));
        }

        Self {
            center,
            radius: max_dist_sq.sqrt(),
        }
    }

    /// Create a sphere from a bounding box.
    pub fn from_box3(box3: &Box3) -> Self {
        let center = box3.center();
        Self {
            center,
            radius: center.distance_to(&box3.max),
        }
    }

    /// Set the sphere components.
    #[inline]
    pub fn set(&mut self, center: Vector3, radius: f32) -> &mut Self {
        self.center = center;
        self.radius = radius;
        self
    }

    /// Copy from another sphere.
    #[inline]
    pub fn copy(&mut self, s: &Sphere) -> &mut Self {
        self.center = s.center;
        self.radius = s.radius;
        self
    }

    /// Check if the sphere is empty (radius <= 0).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.radius <= 0.0
    }

    /// Make the sphere empty.
    #[inline]
    pub fn make_empty(&mut self) -> &mut Self {
        self.center = Vector3::ZERO;
        self.radius = -1.0;
        self
    }

    /// Check if a point is inside the sphere.
    #[inline]
    pub fn contains_point(&self, point: &Vector3) -> bool {
        self.center.distance_to_squared(point) <= self.radius * self.radius
    }

    /// Get the distance from a point to the sphere surface.
    /// Negative if inside.
    #[inline]
    pub fn distance_to_point(&self, point: &Vector3) -> f32 {
        self.center.distance_to(point) - self.radius
    }

    /// Check if this sphere intersects another sphere.
    #[inline]
    pub fn intersects_sphere(&self, other: &Sphere) -> bool {
        let radius_sum = self.radius + other.radius;
        self.center.distance_to_squared(&other.center) <= radius_sum * radius_sum
    }

    /// Check if this sphere intersects a box.
    pub fn intersects_box(&self, box3: &Box3) -> bool {
        let closest = box3.clamp_point(&self.center);
        self.center.distance_to_squared(&closest) <= self.radius * self.radius
    }

    /// Check if this sphere intersects a plane.
    pub fn intersects_plane(&self, plane: &Plane) -> bool {
        plane.distance_to_point(&self.center).abs() <= self.radius
    }

    /// Clamp a point to the sphere surface.
    pub fn clamp_point(&self, point: &Vector3) -> Vector3 {
        let delta = *point - self.center;
        let dist = delta.length();
        if dist <= self.radius {
            *point
        } else {
            self.center + delta * (self.radius / dist)
        }
    }

    /// Get the bounding box of this sphere.
    pub fn bounding_box(&self) -> Box3 {
        Box3::new(
            self.center - Vector3::splat(self.radius),
            self.center + Vector3::splat(self.radius),
        )
    }

    /// Apply a Matrix4 transformation.
    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        Self {
            center: m.transform_point(&self.center),
            radius: self.radius * m.get_max_scale(),
        }
    }

    /// Translate the sphere.
    #[inline]
    pub fn translate(&self, offset: &Vector3) -> Self {
        Self {
            center: self.center + *offset,
            radius: self.radius,
        }
    }

    /// Expand to include a point.
    pub fn expand_by_point(&mut self, point: &Vector3) -> &mut Self {
        if self.is_empty() {
            self.center = *point;
            self.radius = 0.0;
        } else {
            let dist = self.center.distance_to(point);
            if dist > self.radius {
                let half_delta = (dist - self.radius) * 0.5;
                self.center = self.center + (*point - self.center).normalized() * half_delta;
                self.radius += half_delta;
            }
        }
        self
    }

    /// Union with another sphere.
    pub fn union(&self, other: &Sphere) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }

        let dist = self.center.distance_to(&other.center);

        // One sphere contains the other
        if dist + other.radius <= self.radius {
            return *self;
        }
        if dist + self.radius <= other.radius {
            return *other;
        }

        // Calculate new sphere
        let new_radius = (self.radius + other.radius + dist) * 0.5;
        let t = (new_radius - self.radius) / dist;
        let new_center = self.center.lerp(&other.center, t);

        Self {
            center: new_center,
            radius: new_radius,
        }
    }

    /// Check if approximately equal.
    #[inline]
    pub fn approx_eq(&self, other: &Sphere, epsilon: f32) -> bool {
        self.center.approx_eq(&other.center, epsilon)
            && (self.radius - other.radius).abs() < epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_point() {
        let s = Sphere::new(Vector3::ZERO, 1.0);
        assert!(s.contains_point(&Vector3::ZERO));
        assert!(s.contains_point(&Vector3::new(0.5, 0.5, 0.0)));
        assert!(!s.contains_point(&Vector3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_intersects_sphere() {
        let a = Sphere::new(Vector3::ZERO, 1.0);
        let b = Sphere::new(Vector3::new(1.5, 0.0, 0.0), 1.0);
        assert!(a.intersects_sphere(&b));

        let c = Sphere::new(Vector3::new(3.0, 0.0, 0.0), 1.0);
        assert!(!a.intersects_sphere(&c));
    }
}
