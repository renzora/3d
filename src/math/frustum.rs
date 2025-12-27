//! View frustum implementation for culling.

use super::{Box3, Matrix4, Plane, Sphere, Vector3};
use serde::{Deserialize, Serialize};

/// A view frustum defined by 6 planes.
/// Used for frustum culling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Frustum {
    /// The six planes of the frustum.
    /// Order: left, right, bottom, top, near, far
    pub planes: [Plane; 6],
}

impl Default for Frustum {
    fn default() -> Self {
        Self {
            planes: [Plane::default(); 6],
        }
    }
}

impl Frustum {
    /// Create a new frustum from 6 planes.
    #[inline]
    pub const fn new(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// Create a frustum from a projection-view matrix.
    pub fn from_matrix(m: &Matrix4) -> Self {
        let e = &m.elements;

        // Left plane
        let left = Plane {
            normal: Vector3::new(e[3] + e[0], e[7] + e[4], e[11] + e[8]),
            constant: e[15] + e[12],
        }.normalized();

        // Right plane
        let right = Plane {
            normal: Vector3::new(e[3] - e[0], e[7] - e[4], e[11] - e[8]),
            constant: e[15] - e[12],
        }.normalized();

        // Bottom plane
        let bottom = Plane {
            normal: Vector3::new(e[3] + e[1], e[7] + e[5], e[11] + e[9]),
            constant: e[15] + e[13],
        }.normalized();

        // Top plane
        let top = Plane {
            normal: Vector3::new(e[3] - e[1], e[7] - e[5], e[11] - e[9]),
            constant: e[15] - e[13],
        }.normalized();

        // Near plane
        let near = Plane {
            normal: Vector3::new(e[3] + e[2], e[7] + e[6], e[11] + e[10]),
            constant: e[15] + e[14],
        }.normalized();

        // Far plane
        let far = Plane {
            normal: Vector3::new(e[3] - e[2], e[7] - e[6], e[11] - e[10]),
            constant: e[15] - e[14],
        }.normalized();

        Self {
            planes: [left, right, bottom, top, near, far],
        }
    }

    /// Set the frustum from a projection-view matrix.
    pub fn set_from_matrix(&mut self, m: &Matrix4) -> &mut Self {
        *self = Self::from_matrix(m);
        self
    }

    /// Copy from another frustum.
    pub fn copy(&mut self, f: &Frustum) -> &mut Self {
        self.planes = f.planes;
        self
    }

    /// Check if a point is inside the frustum.
    pub fn contains_point(&self, point: &Vector3) -> bool {
        for plane in &self.planes {
            if plane.distance_to_point(point) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Check if a sphere intersects the frustum.
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        for plane in &self.planes {
            let distance = plane.distance_to_point(&sphere.center);
            if distance < -sphere.radius {
                return false;
            }
        }
        true
    }

    /// Check if a box intersects the frustum.
    pub fn intersects_box(&self, box3: &Box3) -> bool {
        for plane in &self.planes {
            // Get the corner that is most in the direction of the plane normal
            let p = Vector3::new(
                if plane.normal.x > 0.0 { box3.max.x } else { box3.min.x },
                if plane.normal.y > 0.0 { box3.max.y } else { box3.min.y },
                if plane.normal.z > 0.0 { box3.max.z } else { box3.min.z },
            );

            if plane.distance_to_point(&p) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Get the left plane.
    #[inline]
    pub fn left(&self) -> &Plane { &self.planes[0] }
    /// Get the right plane.
    #[inline]
    pub fn right(&self) -> &Plane { &self.planes[1] }
    /// Get the bottom plane.
    #[inline]
    pub fn bottom(&self) -> &Plane { &self.planes[2] }
    /// Get the top plane.
    #[inline]
    pub fn top(&self) -> &Plane { &self.planes[3] }
    /// Get the near plane.
    #[inline]
    pub fn near(&self) -> &Plane { &self.planes[4] }
    /// Get the far plane.
    #[inline]
    pub fn far(&self) -> &Plane { &self.planes[5] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_projection() {
        let proj = Matrix4::perspective(
            std::f32::consts::FRAC_PI_4,
            1.0,
            0.1,
            100.0,
        );
        let frustum = Frustum::from_matrix(&proj);

        // Point at origin should be inside
        assert!(frustum.contains_point(&Vector3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn test_sphere_intersection() {
        let proj = Matrix4::perspective(
            std::f32::consts::FRAC_PI_4,
            1.0,
            0.1,
            100.0,
        );
        let frustum = Frustum::from_matrix(&proj);

        // Sphere in front should intersect
        let sphere = Sphere::new(Vector3::new(0.0, 0.0, -5.0), 1.0);
        assert!(frustum.intersects_sphere(&sphere));

        // Sphere behind should not intersect
        let sphere_behind = Sphere::new(Vector3::new(0.0, 0.0, 5.0), 1.0);
        assert!(!frustum.intersects_sphere(&sphere_behind));
    }
}
