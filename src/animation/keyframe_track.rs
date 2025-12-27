//! Keyframe tracks for animating different property types.

use super::interpolant::{InterpolationMode, lerp, lerp_array, slerp};

/// A keyframe with time and value.
#[derive(Debug, Clone)]
pub struct Keyframe<T: Clone> {
    /// Time in seconds.
    pub time: f32,
    /// Value at this keyframe.
    pub value: T,
}

impl<T: Clone> Keyframe<T> {
    /// Create a new keyframe.
    pub fn new(time: f32, value: T) -> Self {
        Self { time, value }
    }
}

/// Track for animating a single f32 value.
#[derive(Debug, Clone)]
pub struct NumberTrack {
    /// Property name/path being animated.
    pub name: String,
    /// Keyframes sorted by time.
    keyframes: Vec<Keyframe<f32>>,
    /// Interpolation mode.
    pub interpolation: InterpolationMode,
}

impl NumberTrack {
    /// Create a new number track.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            keyframes: Vec::new(),
            interpolation: InterpolationMode::Linear,
        }
    }

    /// Create from times and values.
    pub fn from_arrays(name: impl Into<String>, times: &[f32], values: &[f32]) -> Self {
        let keyframes = times.iter().zip(values.iter())
            .map(|(&t, &v)| Keyframe::new(t, v))
            .collect();
        Self {
            name: name.into(),
            keyframes,
            interpolation: InterpolationMode::Linear,
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, time: f32, value: f32) {
        self.keyframes.push(Keyframe::new(time, value));
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Get the duration of this track.
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    /// Sample the track at a given time.
    pub fn sample(&self, time: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }

        if self.keyframes.len() == 1 || time <= self.keyframes[0].time {
            return self.keyframes[0].value;
        }

        let last = self.keyframes.len() - 1;
        if time >= self.keyframes[last].time {
            return self.keyframes[last].value;
        }

        // Find the two keyframes to interpolate between
        let idx = self.keyframes.iter()
            .position(|k| k.time > time)
            .unwrap_or(last);

        let k0 = &self.keyframes[idx - 1];
        let k1 = &self.keyframes[idx];

        let t = (time - k0.time) / (k1.time - k0.time);

        match self.interpolation {
            InterpolationMode::Step => k0.value,
            InterpolationMode::Linear => lerp(k0.value, k1.value, t),
            InterpolationMode::CubicSpline => lerp(k0.value, k1.value, t), // Simplified
        }
    }
}

/// Track for animating Vector3 (position, scale).
#[derive(Debug, Clone)]
pub struct VectorTrack {
    /// Property name/path being animated.
    pub name: String,
    /// Keyframes sorted by time.
    keyframes: Vec<Keyframe<[f32; 3]>>,
    /// Interpolation mode.
    pub interpolation: InterpolationMode,
}

impl VectorTrack {
    /// Create a new vector track.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            keyframes: Vec::new(),
            interpolation: InterpolationMode::Linear,
        }
    }

    /// Create from times and values.
    pub fn from_arrays(name: impl Into<String>, times: &[f32], values: &[[f32; 3]]) -> Self {
        let keyframes = times.iter().zip(values.iter())
            .map(|(&t, v)| Keyframe::new(t, *v))
            .collect();
        Self {
            name: name.into(),
            keyframes,
            interpolation: InterpolationMode::Linear,
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, time: f32, value: [f32; 3]) {
        self.keyframes.push(Keyframe::new(time, value));
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Get the duration of this track.
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    /// Sample the track at a given time.
    pub fn sample(&self, time: f32) -> [f32; 3] {
        if self.keyframes.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        if self.keyframes.len() == 1 || time <= self.keyframes[0].time {
            return self.keyframes[0].value;
        }

        let last = self.keyframes.len() - 1;
        if time >= self.keyframes[last].time {
            return self.keyframes[last].value;
        }

        let idx = self.keyframes.iter()
            .position(|k| k.time > time)
            .unwrap_or(last);

        let k0 = &self.keyframes[idx - 1];
        let k1 = &self.keyframes[idx];

        let t = (time - k0.time) / (k1.time - k0.time);

        match self.interpolation {
            InterpolationMode::Step => k0.value,
            InterpolationMode::Linear => lerp_array(&k0.value, &k1.value, t),
            InterpolationMode::CubicSpline => lerp_array(&k0.value, &k1.value, t),
        }
    }
}

/// Track for animating quaternion rotations.
#[derive(Debug, Clone)]
pub struct QuaternionTrack {
    /// Property name/path being animated.
    pub name: String,
    /// Keyframes sorted by time.
    keyframes: Vec<Keyframe<[f32; 4]>>,
    /// Interpolation mode (slerp used for Linear).
    pub interpolation: InterpolationMode,
}

impl QuaternionTrack {
    /// Create a new quaternion track.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            keyframes: Vec::new(),
            interpolation: InterpolationMode::Linear,
        }
    }

    /// Create from times and values.
    pub fn from_arrays(name: impl Into<String>, times: &[f32], values: &[[f32; 4]]) -> Self {
        let keyframes = times.iter().zip(values.iter())
            .map(|(&t, v)| Keyframe::new(t, *v))
            .collect();
        Self {
            name: name.into(),
            keyframes,
            interpolation: InterpolationMode::Linear,
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, time: f32, value: [f32; 4]) {
        self.keyframes.push(Keyframe::new(time, value));
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Get the duration of this track.
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    /// Sample the track at a given time.
    pub fn sample(&self, time: f32) -> [f32; 4] {
        if self.keyframes.is_empty() {
            return [0.0, 0.0, 0.0, 1.0]; // Identity quaternion
        }

        if self.keyframes.len() == 1 || time <= self.keyframes[0].time {
            return self.keyframes[0].value;
        }

        let last = self.keyframes.len() - 1;
        if time >= self.keyframes[last].time {
            return self.keyframes[last].value;
        }

        let idx = self.keyframes.iter()
            .position(|k| k.time > time)
            .unwrap_or(last);

        let k0 = &self.keyframes[idx - 1];
        let k1 = &self.keyframes[idx];

        let t = (time - k0.time) / (k1.time - k0.time);

        match self.interpolation {
            InterpolationMode::Step => k0.value,
            InterpolationMode::Linear | InterpolationMode::CubicSpline => {
                slerp(&k0.value, &k1.value, t)
            }
        }
    }
}

/// Track for animating RGBA colors.
#[derive(Debug, Clone)]
pub struct ColorTrack {
    /// Property name/path being animated.
    pub name: String,
    /// Keyframes sorted by time.
    keyframes: Vec<Keyframe<[f32; 4]>>,
    /// Interpolation mode.
    pub interpolation: InterpolationMode,
}

impl ColorTrack {
    /// Create a new color track.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            keyframes: Vec::new(),
            interpolation: InterpolationMode::Linear,
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, time: f32, value: [f32; 4]) {
        self.keyframes.push(Keyframe::new(time, value));
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Get the duration of this track.
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    /// Sample the track at a given time.
    pub fn sample(&self, time: f32) -> [f32; 4] {
        if self.keyframes.is_empty() {
            return [1.0, 1.0, 1.0, 1.0];
        }

        if self.keyframes.len() == 1 || time <= self.keyframes[0].time {
            return self.keyframes[0].value;
        }

        let last = self.keyframes.len() - 1;
        if time >= self.keyframes[last].time {
            return self.keyframes[last].value;
        }

        let idx = self.keyframes.iter()
            .position(|k| k.time > time)
            .unwrap_or(last);

        let k0 = &self.keyframes[idx - 1];
        let k1 = &self.keyframes[idx];

        let t = (time - k0.time) / (k1.time - k0.time);

        match self.interpolation {
            InterpolationMode::Step => k0.value,
            InterpolationMode::Linear | InterpolationMode::CubicSpline => {
                lerp_array(&k0.value, &k1.value, t)
            }
        }
    }
}

/// Union type for different track kinds.
#[derive(Debug, Clone)]
pub enum Track {
    /// Single number track.
    Number(NumberTrack),
    /// Vector3 track (position, scale).
    Vector(VectorTrack),
    /// Quaternion track (rotation).
    Quaternion(QuaternionTrack),
    /// Color track (RGBA).
    Color(ColorTrack),
}

impl Track {
    /// Get the property name.
    pub fn name(&self) -> &str {
        match self {
            Track::Number(t) => &t.name,
            Track::Vector(t) => &t.name,
            Track::Quaternion(t) => &t.name,
            Track::Color(t) => &t.name,
        }
    }

    /// Get the duration of this track.
    pub fn duration(&self) -> f32 {
        match self {
            Track::Number(t) => t.duration(),
            Track::Vector(t) => t.duration(),
            Track::Quaternion(t) => t.duration(),
            Track::Color(t) => t.duration(),
        }
    }
}
