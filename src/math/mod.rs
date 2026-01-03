//! # Math Module
//!
//! Complete 3D mathematics library for the Ren engine.
//! Provides vectors, matrices, quaternions, and geometric primitives.
//!
//! This module wraps `glam` for performance while providing a Three.js/Babylon.js-like API.

mod vector2;
mod vector3;
mod vector4;
mod matrix3;
mod matrix4;
mod quaternion;
mod euler;
mod color;
mod ray;
mod plane;
mod sphere;
mod box3;
mod frustum;
mod triangle;
mod line3;
mod raycaster;

pub use vector2::Vector2;
pub use vector3::Vector3;
pub use vector4::Vector4;
pub use matrix3::Matrix3;
pub use matrix4::Matrix4;
pub use quaternion::Quaternion;
pub use euler::{Euler, EulerOrder};
pub use color::Color;
pub use ray::Ray;
pub use plane::Plane;
pub use sphere::Sphere;
pub use box3::Box3;
pub use frustum::Frustum;
pub use triangle::Triangle;
pub use line3::Line3;
pub use raycaster::Raycaster;

/// Common math constants and utilities.
pub mod consts {
    /// Pi constant.
    pub const PI: f32 = std::f32::consts::PI;
    /// Two times Pi.
    pub const TWO_PI: f32 = PI * 2.0;
    /// Half of Pi.
    pub const HALF_PI: f32 = PI / 2.0;
    /// Degrees to radians conversion factor.
    pub const DEG2RAD: f32 = PI / 180.0;
    /// Radians to degrees conversion factor.
    pub const RAD2DEG: f32 = 180.0 / PI;
    /// Small epsilon for floating point comparisons.
    pub const EPSILON: f32 = 1e-6;
}

/// Convert degrees to radians.
#[inline]
pub fn deg_to_rad(degrees: f32) -> f32 {
    degrees * consts::DEG2RAD
}

/// Convert radians to degrees.
#[inline]
pub fn rad_to_deg(radians: f32) -> f32 {
    radians * consts::RAD2DEG
}

/// Clamp a value between min and max.
#[inline]
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

/// Linear interpolation between two values.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smooth step interpolation.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
