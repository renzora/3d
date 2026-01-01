//! Particle data structures for GPU storage.

use bytemuck::{Pod, Zeroable};

/// Particle state stored in GPU storage buffer.
/// Layout optimized for GPU cache coherence (96 bytes per particle).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ParticleGpu {
    /// Position in world space (xyz) + age in seconds (w).
    pub position_age: [f32; 4],
    /// Velocity (xyz) + lifetime in seconds (w).
    pub velocity_lifetime: [f32; 4],
    /// Current color (rgba).
    pub color: [f32; 4],
    /// Size (x=current, y=start, z=end) + rotation angle (w).
    pub size_rotation: [f32; 4],
    /// Flags: x=alive (0 or 1), y=texture_index, z=random_seed, w=rotation_speed.
    pub flags: [f32; 4],
    /// Start color for interpolation.
    pub start_color: [f32; 4],
    /// End color for interpolation.
    pub end_color: [f32; 4],
}

impl Default for ParticleGpu {
    fn default() -> Self {
        Self {
            position_age: [0.0, 0.0, 0.0, 0.0],
            velocity_lifetime: [0.0, 0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            size_rotation: [1.0, 1.0, 1.0, 0.0],
            flags: [0.0, 0.0, 0.0, 0.0], // Dead by default
            start_color: [1.0, 1.0, 1.0, 1.0],
            end_color: [1.0, 1.0, 1.0, 0.0],
        }
    }
}

/// Emission parameters uniform for spawning new particles.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct EmissionParams {
    /// Emitter position (xyz) + emission rate (w).
    pub emitter_pos_rate: [f32; 4],
    /// Emitter direction (xyz) + spread angle in radians (w).
    pub emitter_dir_spread: [f32; 4],
    /// Min velocity (xyz) + min lifetime (w).
    pub velocity_min_lifetime: [f32; 4],
    /// Max velocity (xyz) + max lifetime (w).
    pub velocity_max_lifetime: [f32; 4],
    /// Start color (rgba).
    pub start_color: [f32; 4],
    /// End color (rgba).
    pub end_color: [f32; 4],
    /// Size: x=start_min, y=start_max, z=end_min, w=end_max.
    pub size_params: [f32; 4],
    /// Emitter shape: x=type (0=point, 1=sphere, 2=cone, 3=box), y=radius/angle, z=height, w=unused.
    pub emitter_shape: [f32; 4],
    /// Rotation: x=min_speed, y=max_speed, z=unused, w=unused.
    pub rotation_params: [f32; 4],
    /// Color variance: x=start_r, y=start_g, z=start_b, w=start_a.
    pub start_color_variance: [f32; 4],
    /// Color variance: x=end_r, y=end_g, z=end_b, w=end_a.
    pub end_color_variance: [f32; 4],
    /// Box half extents (xyz) + unused (w). Only used when shape is Box.
    pub box_half_extents: [f32; 4],
}

impl Default for EmissionParams {
    fn default() -> Self {
        Self {
            emitter_pos_rate: [0.0, 0.0, 0.0, 10.0],
            emitter_dir_spread: [0.0, 1.0, 0.0, 0.5],
            velocity_min_lifetime: [0.0, 1.0, 0.0, 1.0],
            velocity_max_lifetime: [0.0, 2.0, 0.0, 2.0],
            start_color: [1.0, 1.0, 1.0, 1.0],
            end_color: [1.0, 1.0, 1.0, 0.0],
            size_params: [0.1, 0.2, 0.2, 0.4],
            emitter_shape: [0.0, 0.0, 0.0, 0.0], // Point emitter
            rotation_params: [0.0, 0.0, 0.0, 0.0],
            start_color_variance: [0.0, 0.0, 0.0, 0.0],
            end_color_variance: [0.0, 0.0, 0.0, 0.0],
            box_half_extents: [1.0, 1.0, 1.0, 0.0],
        }
    }
}
