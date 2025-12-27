//! Triangle implementation.

use super::{Box3, Plane, Vector3};
use serde::{Deserialize, Serialize};

/// A triangle defined by three vertices.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Triangle {
    /// First vertex.
    pub a: Vector3,
    /// Second vertex.
    pub b: Vector3,
    /// Third vertex.
    pub c: Vector3,
}

impl Triangle {
    /// Create a new triangle.
    #[inline]
    pub const fn new(a: Vector3, b: Vector3, c: Vector3) -> Self {
        Self { a, b, c }
    }

    /// Set the triangle vertices.
    #[inline]
    pub fn set(&mut self, a: Vector3, b: Vector3, c: Vector3) -> &mut Self {
        self.a = a;
        self.b = b;
        self.c = c;
        self
    }

    /// Copy from another triangle.
    #[inline]
    pub fn copy(&mut self, t: &Triangle) -> &mut Self {
        self.a = t.a;
        self.b = t.b;
        self.c = t.c;
        self
    }

    /// Get the area of the triangle.
    pub fn area(&self) -> f32 {
        let ab = self.b - self.a;
        let ac = self.c - self.a;
        ab.cross(&ac).length() * 0.5
    }

    /// Get the centroid (center) of the triangle.
    #[inline]
    pub fn centroid(&self) -> Vector3 {
        (self.a + self.b + self.c) / 3.0
    }

    /// Get the normal of the triangle.
    pub fn normal(&self) -> Vector3 {
        let ab = self.b - self.a;
        let ac = self.c - self.a;
        ab.cross(&ac).normalized()
    }

    /// Get the plane containing this triangle.
    pub fn plane(&self) -> Plane {
        Plane::from_coplanar_points(&self.a, &self.b, &self.c)
    }

    /// Get the bounding box of this triangle.
    pub fn bounding_box(&self) -> Box3 {
        Box3::from_points(&[self.a, self.b, self.c])
    }

    /// Get the barycentric coordinates of a point on the triangle.
    /// Returns (u, v, w) where point = a*u + b*v + c*w.
    pub fn barycentric(&self, point: &Vector3) -> (f32, f32, f32) {
        let v0 = self.c - self.a;
        let v1 = self.b - self.a;
        let v2 = *point - self.a;

        let dot00 = v0.dot(&v0);
        let dot01 = v0.dot(&v1);
        let dot02 = v0.dot(&v2);
        let dot11 = v1.dot(&v1);
        let dot12 = v1.dot(&v2);

        let denom = dot00 * dot11 - dot01 * dot01;
        if denom.abs() < 1e-10 {
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        }

        let inv_denom = 1.0 / denom;
        let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
        let w = 1.0 - u - v;

        (w, v, u)
    }

    /// Check if a point is inside the triangle (2D check on triangle plane).
    pub fn contains_point(&self, point: &Vector3) -> bool {
        let (u, v, w) = self.barycentric(point);
        u >= 0.0 && v >= 0.0 && w >= 0.0
    }

    /// Get the UV coordinates at a point given vertex UVs.
    pub fn uv_at_point(&self, point: &Vector3, uv_a: (f32, f32), uv_b: (f32, f32), uv_c: (f32, f32)) -> (f32, f32) {
        let (u, v, w) = self.barycentric(point);
        (
            u * uv_a.0 + v * uv_b.0 + w * uv_c.0,
            u * uv_a.1 + v * uv_b.1 + w * uv_c.1,
        )
    }

    /// Get the closest point on the triangle to a given point.
    pub fn closest_point(&self, point: &Vector3) -> Vector3 {
        let ab = self.b - self.a;
        let ac = self.c - self.a;
        let ap = *point - self.a;

        let d1 = ab.dot(&ap);
        let d2 = ac.dot(&ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return self.a;
        }

        let bp = *point - self.b;
        let d3 = ab.dot(&bp);
        let d4 = ac.dot(&bp);
        if d3 >= 0.0 && d4 <= d3 {
            return self.b;
        }

        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return self.a + ab * v;
        }

        let cp = *point - self.c;
        let d5 = ab.dot(&cp);
        let d6 = ac.dot(&cp);
        if d6 >= 0.0 && d5 <= d6 {
            return self.c;
        }

        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return self.a + ac * w;
        }

        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return self.b + (self.c - self.b) * w;
        }

        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        self.a + ab * v + ac * w
    }

    /// Check if approximately equal.
    #[inline]
    pub fn approx_eq(&self, other: &Triangle, epsilon: f32) -> bool {
        self.a.approx_eq(&other.a, epsilon)
            && self.b.approx_eq(&other.b, epsilon)
            && self.c.approx_eq(&other.c, epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_area() {
        let t = Triangle::new(
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        assert!((t.area() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normal() {
        let t = Triangle::new(
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let n = t.normal();
        assert!(n.approx_eq(&Vector3::UNIT_Z, 1e-6));
    }

    #[test]
    fn test_barycentric() {
        let t = Triangle::new(
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let centroid = t.centroid();
        let (u, v, w) = t.barycentric(&centroid);
        assert!((u - 1.0/3.0).abs() < 1e-5);
        assert!((v - 1.0/3.0).abs() < 1e-5);
        assert!((w - 1.0/3.0).abs() < 1e-5);
    }
}
