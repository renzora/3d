//! Ray implementation for raycasting.

use super::{Box3, Matrix4, Plane, Sphere, Triangle, Vector3};
use serde::{Deserialize, Serialize};

/// A ray with an origin and direction.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Ray {
    /// Origin point of the ray.
    pub origin: Vector3,
    /// Direction of the ray (should be normalized).
    pub direction: Vector3,
}

impl Ray {
    /// Create a new ray.
    #[inline]
    pub const fn new(origin: Vector3, direction: Vector3) -> Self {
        Self { origin, direction }
    }

    /// Set the ray components.
    #[inline]
    pub fn set(&mut self, origin: Vector3, direction: Vector3) -> &mut Self {
        self.origin = origin;
        self.direction = direction;
        self
    }

    /// Copy from another ray.
    #[inline]
    pub fn copy(&mut self, r: &Ray) -> &mut Self {
        self.origin = r.origin;
        self.direction = r.direction;
        self
    }

    /// Get a point at distance t along the ray.
    #[inline]
    pub fn at(&self, t: f32) -> Vector3 {
        self.origin + self.direction * t
    }

    /// Get the point on the ray closest to a given point.
    pub fn closest_point_to_point(&self, point: &Vector3) -> Vector3 {
        let t = (*point - self.origin).dot(&self.direction);
        if t < 0.0 {
            self.origin
        } else {
            self.at(t)
        }
    }

    /// Get distance from ray to a point.
    pub fn distance_to_point(&self, point: &Vector3) -> f32 {
        self.closest_point_to_point(point).distance_to(point)
    }

    /// Get squared distance from ray to a point.
    pub fn distance_sq_to_point(&self, point: &Vector3) -> f32 {
        self.closest_point_to_point(point).distance_to_squared(point)
    }

    /// Get distance from ray origin to a plane.
    /// Returns None if ray is parallel to plane.
    pub fn distance_to_plane(&self, plane: &Plane) -> Option<f32> {
        let denom = plane.normal.dot(&self.direction);
        if denom.abs() < 1e-8 {
            // Ray is parallel to plane
            if plane.distance_to_point(&self.origin).abs() < 1e-8 {
                Some(0.0)
            } else {
                None
            }
        } else {
            let t = -(self.origin.dot(&plane.normal) + plane.constant) / denom;
            if t >= 0.0 { Some(t) } else { None }
        }
    }

    /// Intersect with a plane.
    /// Returns the intersection point, or None if no intersection.
    pub fn intersect_plane(&self, plane: &Plane) -> Option<Vector3> {
        self.distance_to_plane(plane).map(|t| self.at(t))
    }

    /// Check if ray intersects a plane.
    pub fn intersects_plane(&self, plane: &Plane) -> bool {
        self.distance_to_plane(plane).is_some()
    }

    /// Intersect with a sphere.
    /// Returns the distance to intersection, or None if no intersection.
    pub fn intersect_sphere(&self, sphere: &Sphere) -> Option<f32> {
        let oc = self.origin - sphere.center;
        let b = oc.dot(&self.direction);
        let c = oc.length_squared() - sphere.radius * sphere.radius;

        let discriminant = b * b - c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_discriminant = discriminant.sqrt();
        let t1 = -b - sqrt_discriminant;
        let t2 = -b + sqrt_discriminant;

        if t1 >= 0.0 {
            Some(t1)
        } else if t2 >= 0.0 {
            Some(t2)
        } else {
            None
        }
    }

    /// Check if ray intersects a sphere.
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        self.intersect_sphere(sphere).is_some()
    }

    /// Intersect with an axis-aligned bounding box.
    /// Returns (tmin, tmax) if intersection, None otherwise.
    pub fn intersect_box(&self, box3: &Box3) -> Option<(f32, f32)> {
        let inv_dir = Vector3::new(
            1.0 / self.direction.x,
            1.0 / self.direction.y,
            1.0 / self.direction.z,
        );

        let t1 = (box3.min.x - self.origin.x) * inv_dir.x;
        let t2 = (box3.max.x - self.origin.x) * inv_dir.x;
        let t3 = (box3.min.y - self.origin.y) * inv_dir.y;
        let t4 = (box3.max.y - self.origin.y) * inv_dir.y;
        let t5 = (box3.min.z - self.origin.z) * inv_dir.z;
        let t6 = (box3.max.z - self.origin.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if tmax < 0.0 || tmin > tmax {
            None
        } else {
            Some((tmin.max(0.0), tmax))
        }
    }

    /// Check if ray intersects a box.
    pub fn intersects_box(&self, box3: &Box3) -> bool {
        self.intersect_box(box3).is_some()
    }

    /// Intersect with a triangle.
    /// Returns (distance, u, v) where u,v are barycentric coordinates.
    /// Uses Möller–Trumbore algorithm.
    pub fn intersect_triangle(&self, triangle: &Triangle, backface_culling: bool) -> Option<(f32, f32, f32)> {
        let edge1 = triangle.b - triangle.a;
        let edge2 = triangle.c - triangle.a;
        let h = self.direction.cross(&edge2);
        let a = edge1.dot(&h);

        if backface_culling {
            if a < 1e-8 {
                return None;
            }
        } else if a.abs() < 1e-8 {
            return None;
        }

        let f = 1.0 / a;
        let s = self.origin - triangle.a;
        let u = f * s.dot(&h);

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(&edge1);
        let v = f * self.direction.dot(&q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(&q);

        if t > 1e-8 {
            Some((t, u, v))
        } else {
            None
        }
    }

    /// Apply a Matrix4 transformation to this ray.
    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        Self {
            origin: m.transform_point(&self.origin),
            direction: m.transform_direction(&self.direction).normalized(),
        }
    }

    /// Check if approximately equal.
    pub fn approx_eq(&self, other: &Ray, epsilon: f32) -> bool {
        self.origin.approx_eq(&other.origin, epsilon)
            && self.direction.approx_eq(&other.direction, epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at() {
        let ray = Ray::new(Vector3::ZERO, Vector3::UNIT_Z);
        let p = ray.at(5.0);
        assert!(p.approx_eq(&Vector3::new(0.0, 0.0, 5.0), 1e-6));
    }

    #[test]
    fn test_sphere_intersection() {
        let ray = Ray::new(Vector3::new(0.0, 0.0, -5.0), Vector3::UNIT_Z);
        let sphere = Sphere::new(Vector3::ZERO, 1.0);
        let t = ray.intersect_sphere(&sphere);
        assert!(t.is_some());
        assert!((t.unwrap() - 4.0).abs() < 1e-6);
    }
}
