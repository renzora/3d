//! 3D Line segment implementation.

use super::{Matrix4, Vector3};
use serde::{Deserialize, Serialize};

/// A 3D line segment defined by start and end points.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Line3 {
    /// Start point of the line.
    pub start: Vector3,
    /// End point of the line.
    pub end: Vector3,
}

impl Line3 {
    /// Create a new line segment.
    #[inline]
    pub const fn new(start: Vector3, end: Vector3) -> Self {
        Self { start, end }
    }

    /// Set the line segment points.
    #[inline]
    pub fn set(&mut self, start: Vector3, end: Vector3) -> &mut Self {
        self.start = start;
        self.end = end;
        self
    }

    /// Copy from another line.
    #[inline]
    pub fn copy(&mut self, line: &Line3) -> &mut Self {
        self.start = line.start;
        self.end = line.end;
        self
    }

    /// Get the center (midpoint) of the line.
    #[inline]
    pub fn center(&self) -> Vector3 {
        (self.start + self.end) * 0.5
    }

    /// Get the direction vector from start to end.
    #[inline]
    pub fn delta(&self) -> Vector3 {
        self.end - self.start
    }

    /// Get the length of the line segment.
    #[inline]
    pub fn length(&self) -> f32 {
        self.start.distance_to(&self.end)
    }

    /// Get the squared length of the line segment.
    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.start.distance_to_squared(&self.end)
    }

    /// Get a point at parameter t along the line.
    /// t=0 returns start, t=1 returns end.
    #[inline]
    pub fn at(&self, t: f32) -> Vector3 {
        self.start.lerp(&self.end, t)
    }

    /// Get the parameter t for the closest point on the line to a given point.
    /// The parameter is clamped to [0, 1].
    pub fn closest_point_parameter(&self, point: &Vector3, clamp_to_line: bool) -> f32 {
        let delta = self.delta();
        let length_sq = delta.length_squared();

        if length_sq < 1e-10 {
            return 0.0;
        }

        let t = (*point - self.start).dot(&delta) / length_sq;

        if clamp_to_line {
            t.clamp(0.0, 1.0)
        } else {
            t
        }
    }

    /// Get the closest point on the line to a given point.
    pub fn closest_point(&self, point: &Vector3, clamp_to_line: bool) -> Vector3 {
        let t = self.closest_point_parameter(point, clamp_to_line);
        self.at(t)
    }

    /// Get the distance from a point to this line segment.
    pub fn distance_to_point(&self, point: &Vector3, clamp_to_line: bool) -> f32 {
        self.closest_point(point, clamp_to_line).distance_to(point)
    }

    /// Get the squared distance from a point to this line segment.
    pub fn distance_sq_to_point(&self, point: &Vector3, clamp_to_line: bool) -> f32 {
        self.closest_point(point, clamp_to_line).distance_to_squared(point)
    }

    /// Get the closest points between this line and another line.
    /// Returns (t1, t2) where the closest point on this line is at(t1)
    /// and the closest point on the other line is at(t2).
    pub fn closest_points_to_line(&self, other: &Line3, clamp_to_lines: bool) -> (f32, f32) {
        let d1 = self.delta();
        let d2 = other.delta();
        let r = self.start - other.start;

        let a = d1.length_squared();
        let b = d1.dot(&d2);
        let c = d2.length_squared();
        let d = d1.dot(&r);
        let e = d2.dot(&r);

        let denom = a * c - b * b;

        let (mut s, mut t);

        if denom.abs() < 1e-10 {
            // Lines are parallel
            s = 0.0;
            t = if b > c { d / b } else { e / c };
        } else {
            s = (b * e - c * d) / denom;
            t = (a * e - b * d) / denom;
        }

        if clamp_to_lines {
            s = s.clamp(0.0, 1.0);
            t = t.clamp(0.0, 1.0);
        }

        (s, t)
    }

    /// Get the minimum distance between this line segment and another.
    pub fn distance_to_line(&self, other: &Line3, clamp_to_lines: bool) -> f32 {
        let (s, t) = self.closest_points_to_line(other, clamp_to_lines);
        let p1 = self.at(s);
        let p2 = other.at(t);
        p1.distance_to(&p2)
    }

    /// Apply a Matrix4 transformation.
    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        Self {
            start: m.transform_point(&self.start),
            end: m.transform_point(&self.end),
        }
    }

    /// Check if approximately equal.
    #[inline]
    pub fn approx_eq(&self, other: &Line3, epsilon: f32) -> bool {
        self.start.approx_eq(&other.start, epsilon) && self.end.approx_eq(&other.end, epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length() {
        let line = Line3::new(Vector3::ZERO, Vector3::new(3.0, 4.0, 0.0));
        assert!((line.length() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_at() {
        let line = Line3::new(Vector3::ZERO, Vector3::new(10.0, 0.0, 0.0));
        let mid = line.at(0.5);
        assert!(mid.approx_eq(&Vector3::new(5.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_closest_point() {
        let line = Line3::new(Vector3::ZERO, Vector3::new(10.0, 0.0, 0.0));
        let point = Vector3::new(5.0, 5.0, 0.0);
        let closest = line.closest_point(&point, true);
        assert!(closest.approx_eq(&Vector3::new(5.0, 0.0, 0.0), 1e-6));
    }
}
