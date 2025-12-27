//! Interpolation functions for animation.

/// Interpolation mode for keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationMode {
    /// Linear interpolation between keyframes.
    #[default]
    Linear,
    /// Step/discrete - no interpolation, jump to next value.
    Step,
    /// Cubic spline interpolation for smooth curves.
    CubicSpline,
}

/// Linear interpolation between two values.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Linear interpolation for arrays.
#[inline]
pub fn lerp_array<const N: usize>(a: &[f32; N], b: &[f32; N], t: f32) -> [f32; N] {
    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = lerp(a[i], b[i], t);
    }
    result
}

/// Spherical linear interpolation for quaternions.
pub fn slerp(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    let mut b = *b;

    // Compute the cosine of the angle between the two vectors.
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

    // If the dot product is negative, negate one quaternion to take the shorter path.
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }

    // If the inputs are too close, linearly interpolate.
    if dot > 0.9995 {
        let result = [
            lerp(a[0], b[0], t),
            lerp(a[1], b[1], t),
            lerp(a[2], b[2], t),
            lerp(a[3], b[3], t),
        ];
        // Normalize
        let len = (result[0] * result[0] + result[1] * result[1] +
                   result[2] * result[2] + result[3] * result[3]).sqrt();
        return [result[0] / len, result[1] / len, result[2] / len, result[3] / len];
    }

    let theta_0 = dot.acos();
    let theta = theta_0 * t;
    let sin_theta = theta.sin();
    let sin_theta_0 = theta_0.sin();

    let s0 = (theta_0 - theta).cos() - dot * sin_theta / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;

    [
        a[0] * s0 + b[0] * s1,
        a[1] * s0 + b[1] * s1,
        a[2] * s0 + b[2] * s1,
        a[3] * s0 + b[3] * s1,
    ]
}

/// Cubic Hermite spline interpolation.
/// p0, p1 are the values, m0, m1 are the tangents.
pub fn cubic_hermite(p0: f32, m0: f32, p1: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

/// Cubic Hermite spline interpolation for arrays.
pub fn cubic_hermite_array<const N: usize>(
    p0: &[f32; N],
    m0: &[f32; N],
    p1: &[f32; N],
    m1: &[f32; N],
    t: f32,
) -> [f32; N] {
    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = cubic_hermite(p0[i], m0[i], p1[i], m1[i], t);
    }
    result
}

/// Easing functions for animation timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Easing {
    /// No easing, linear progression.
    #[default]
    Linear,
    /// Slow start.
    EaseIn,
    /// Slow end.
    EaseOut,
    /// Slow start and end.
    EaseInOut,
    /// Quadratic ease in.
    QuadIn,
    /// Quadratic ease out.
    QuadOut,
    /// Quadratic ease in/out.
    QuadInOut,
    /// Cubic ease in.
    CubicIn,
    /// Cubic ease out.
    CubicOut,
    /// Cubic ease in/out.
    CubicInOut,
}

impl Easing {
    /// Apply the easing function to a normalized time value (0-1).
    pub fn apply(&self, t: f32) -> f32 {
        match self {
            Easing::Linear => t,
            Easing::EaseIn => t * t,
            Easing::EaseOut => t * (2.0 - t),
            Easing::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            Easing::QuadIn => t * t,
            Easing::QuadOut => 1.0 - (1.0 - t) * (1.0 - t),
            Easing::QuadInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Easing::CubicIn => t * t * t,
            Easing::CubicOut => 1.0 - (1.0 - t).powi(3),
            Easing::CubicInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            }
        }
    }
}
