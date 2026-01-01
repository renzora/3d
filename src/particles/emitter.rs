//! Emitter configuration and shape types.

use super::particle::EmissionParams;

/// Emitter shape type.
#[derive(Debug, Clone, Copy, Default)]
pub enum EmitterShape {
    /// Emit from a single point.
    #[default]
    Point,
    /// Emit from random positions on a sphere surface.
    Sphere {
        /// Sphere radius.
        radius: f32,
    },
    /// Emit from random positions in a sphere volume.
    SphereVolume {
        /// Sphere radius.
        radius: f32,
    },
    /// Emit in a cone direction.
    Cone {
        /// Cone half-angle in radians.
        angle: f32,
        /// Cone height.
        height: f32,
    },
    /// Emit from random positions in a box.
    Box {
        /// Box half-extents (x, y, z).
        half_extents: [f32; 3],
    },
}

impl EmitterShape {
    /// Convert to GPU format: [type, param1, param2, param3].
    pub fn to_gpu_params(&self) -> ([f32; 4], [f32; 4]) {
        match self {
            EmitterShape::Point => ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
            EmitterShape::Sphere { radius } => ([1.0, *radius, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
            EmitterShape::SphereVolume { radius } => {
                ([2.0, *radius, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
            }
            EmitterShape::Cone { angle, height } => {
                ([3.0, *angle, *height, 0.0], [0.0, 0.0, 0.0, 0.0])
            }
            EmitterShape::Box { half_extents } => (
                [4.0, 0.0, 0.0, 0.0],
                [half_extents[0], half_extents[1], half_extents[2], 0.0],
            ),
        }
    }
}

/// Particle blend mode determines render pass ordering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParticleBlendMode {
    /// Additive blending (fire, sparks, magic). Order-independent.
    #[default]
    Additive,
    /// Alpha blending (smoke, dust, debris). May require depth sorting.
    AlphaBlend,
}

/// Particle effect preset for common effects.
#[derive(Debug, Clone, Copy)]
pub enum ParticlePreset {
    /// Fire effect with rising flames and embers.
    Fire,
    /// Smoke effect with billowing clouds.
    Smoke,
    /// Spark effect with fast-moving bright particles.
    Sparks,
    /// Debris effect with physics-affected chunks.
    Debris,
    /// Magic energy effect with glowing orbs.
    MagicEnergy,
    /// Custom configuration.
    Custom,
}

/// Complete emitter configuration.
#[derive(Debug, Clone)]
pub struct EmitterConfig {
    /// Maximum number of particles this emitter can have alive.
    pub max_particles: u32,
    /// Particles to emit per second.
    pub emission_rate: f32,
    /// Burst count (0 for continuous emission).
    pub burst_count: u32,
    /// Emitter shape.
    pub shape: EmitterShape,
    /// Emission direction (normalized).
    pub direction: [f32; 3],
    /// Spread angle in radians (0 = focused, PI = omnidirectional).
    pub spread: f32,
    /// Minimum initial velocity.
    pub velocity_min: [f32; 3],
    /// Maximum initial velocity.
    pub velocity_max: [f32; 3],
    /// Minimum lifetime in seconds.
    pub lifetime_min: f32,
    /// Maximum lifetime in seconds.
    pub lifetime_max: f32,
    /// Start color.
    pub start_color: [f32; 4],
    /// Start color variance (random offset).
    pub start_color_variance: [f32; 4],
    /// End color.
    pub end_color: [f32; 4],
    /// End color variance (random offset).
    pub end_color_variance: [f32; 4],
    /// Size at start of life.
    pub size_start: f32,
    /// Size start variance.
    pub size_start_variance: f32,
    /// Size at end of life.
    pub size_end: f32,
    /// Size end variance.
    pub size_end_variance: f32,
    /// Minimum rotation speed (radians per second).
    pub rotation_speed_min: f32,
    /// Maximum rotation speed (radians per second).
    pub rotation_speed_max: f32,
    /// Blend mode (determines render pass).
    pub blend_mode: ParticleBlendMode,
    /// Texture atlas index (0 = default texture).
    pub texture_index: u32,
    /// Whether to animate through texture atlas.
    pub animate_texture: bool,
    /// Soft particle fade distance (0 = disabled).
    pub soft_fade_distance: f32,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            max_particles: 1000,
            emission_rate: 50.0,
            burst_count: 0,
            shape: EmitterShape::Point,
            direction: [0.0, 1.0, 0.0],
            spread: 0.5,
            velocity_min: [0.0, 1.0, 0.0],
            velocity_max: [0.0, 2.0, 0.0],
            lifetime_min: 1.0,
            lifetime_max: 2.0,
            start_color: [1.0, 1.0, 1.0, 1.0],
            start_color_variance: [0.0, 0.0, 0.0, 0.0],
            end_color: [1.0, 1.0, 1.0, 0.0],
            end_color_variance: [0.0, 0.0, 0.0, 0.0],
            size_start: 0.1,
            size_start_variance: 0.02,
            size_end: 0.3,
            size_end_variance: 0.05,
            rotation_speed_min: 0.0,
            rotation_speed_max: 0.0,
            blend_mode: ParticleBlendMode::Additive,
            texture_index: 0,
            animate_texture: false,
            soft_fade_distance: 0.5,
        }
    }
}

impl EmitterConfig {
    /// Create a fire effect preset.
    pub fn fire_preset() -> Self {
        Self {
            max_particles: 800,
            emission_rate: 120.0,
            burst_count: 0,
            // Emit from a flat disc at the base
            shape: EmitterShape::Box {
                half_extents: [0.15, 0.02, 0.15],
            },
            direction: [0.0, 1.0, 0.0],
            spread: 0.25, // Fairly focused upward
            // Faster rise with some horizontal turbulence
            velocity_min: [-0.3, 1.5, -0.3],
            velocity_max: [0.3, 3.5, 0.3],
            // Short-lived for snappy flames
            lifetime_min: 0.4,
            lifetime_max: 1.0,
            // Bright yellow-white core (HDR values > 1.0 for bloom)
            start_color: [3.0, 2.0, 0.5, 1.0],
            start_color_variance: [0.5, 0.5, 0.2, 0.0],
            // Fade to deep red/orange then transparent
            end_color: [1.0, 0.15, 0.0, 0.0],
            end_color_variance: [0.2, 0.1, 0.0, 0.0],
            // Start small at base, grow as they rise
            size_start: 0.12,
            size_start_variance: 0.04,
            size_end: 0.5,
            size_end_variance: 0.15,
            // Gentle rotation for organic look
            rotation_speed_min: -2.0,
            rotation_speed_max: 2.0,
            blend_mode: ParticleBlendMode::Additive,
            texture_index: 0,
            animate_texture: false,
            soft_fade_distance: 0.3,
        }
    }

    /// Create a smoke effect preset.
    pub fn smoke_preset() -> Self {
        Self {
            max_particles: 200,
            emission_rate: 15.0,
            burst_count: 0,
            shape: EmitterShape::Point,
            direction: [0.0, 1.0, 0.0],
            spread: 0.8,
            velocity_min: [-0.3, 0.5, -0.3],
            velocity_max: [0.3, 1.5, 0.3],
            lifetime_min: 2.0,
            lifetime_max: 4.0,
            start_color: [0.3, 0.3, 0.3, 0.6],
            start_color_variance: [0.1, 0.1, 0.1, 0.1],
            end_color: [0.5, 0.5, 0.5, 0.0],
            end_color_variance: [0.1, 0.1, 0.1, 0.0],
            size_start: 0.2,
            size_start_variance: 0.05,
            size_end: 1.5,
            size_end_variance: 0.3,
            rotation_speed_min: -0.5,
            rotation_speed_max: 0.5,
            blend_mode: ParticleBlendMode::AlphaBlend,
            texture_index: 0,
            animate_texture: false,
            soft_fade_distance: 1.0,
        }
    }

    /// Create a sparks effect preset.
    pub fn sparks_preset() -> Self {
        Self {
            max_particles: 1000,
            emission_rate: 100.0,
            burst_count: 0,
            shape: EmitterShape::Point,
            direction: [0.0, 1.0, 0.0],
            spread: 1.2,
            velocity_min: [-3.0, 2.0, -3.0],
            velocity_max: [3.0, 6.0, 3.0],
            lifetime_min: 0.3,
            lifetime_max: 1.0,
            start_color: [1.0, 0.8, 0.3, 1.0],
            start_color_variance: [0.1, 0.2, 0.1, 0.0],
            end_color: [1.0, 0.3, 0.0, 0.0],
            end_color_variance: [0.1, 0.1, 0.0, 0.0],
            size_start: 0.02,
            size_start_variance: 0.01,
            size_end: 0.01,
            size_end_variance: 0.005,
            rotation_speed_min: 0.0,
            rotation_speed_max: 0.0,
            blend_mode: ParticleBlendMode::Additive,
            texture_index: 0,
            animate_texture: false,
            soft_fade_distance: 0.1,
        }
    }

    /// Create a debris effect preset.
    pub fn debris_preset() -> Self {
        Self {
            max_particles: 100,
            emission_rate: 0.0, // Burst only
            burst_count: 50,
            shape: EmitterShape::SphereVolume { radius: 0.5 },
            direction: [0.0, 1.0, 0.0],
            spread: std::f32::consts::PI,
            velocity_min: [-5.0, 2.0, -5.0],
            velocity_max: [5.0, 8.0, 5.0],
            lifetime_min: 1.0,
            lifetime_max: 3.0,
            start_color: [0.6, 0.5, 0.4, 1.0],
            start_color_variance: [0.1, 0.1, 0.1, 0.0],
            end_color: [0.4, 0.35, 0.3, 0.5],
            end_color_variance: [0.05, 0.05, 0.05, 0.0],
            size_start: 0.1,
            size_start_variance: 0.05,
            size_end: 0.1,
            size_end_variance: 0.03,
            rotation_speed_min: -5.0,
            rotation_speed_max: 5.0,
            blend_mode: ParticleBlendMode::AlphaBlend,
            texture_index: 0,
            animate_texture: false,
            soft_fade_distance: 0.3,
        }
    }

    /// Create a magic energy effect preset.
    pub fn magic_preset() -> Self {
        Self {
            max_particles: 300,
            emission_rate: 30.0,
            burst_count: 0,
            shape: EmitterShape::Sphere { radius: 0.5 },
            direction: [0.0, 0.0, 0.0],
            spread: 0.0,
            velocity_min: [-0.5, -0.5, -0.5],
            velocity_max: [0.5, 0.5, 0.5],
            lifetime_min: 0.8,
            lifetime_max: 1.5,
            start_color: [0.3, 0.5, 1.0, 1.0],
            start_color_variance: [0.2, 0.2, 0.1, 0.0],
            end_color: [0.8, 0.3, 1.0, 0.0],
            end_color_variance: [0.1, 0.1, 0.1, 0.0],
            size_start: 0.15,
            size_start_variance: 0.05,
            size_end: 0.05,
            size_end_variance: 0.02,
            rotation_speed_min: -2.0,
            rotation_speed_max: 2.0,
            blend_mode: ParticleBlendMode::Additive,
            texture_index: 0,
            animate_texture: false,
            soft_fade_distance: 0.2,
        }
    }

    /// Convert to emission parameters for GPU uniform.
    pub fn to_emission_params(&self, position: [f32; 3]) -> EmissionParams {
        let (shape_params, box_extents) = self.shape.to_gpu_params();

        EmissionParams {
            emitter_pos_rate: [position[0], position[1], position[2], self.emission_rate],
            emitter_dir_spread: [
                self.direction[0],
                self.direction[1],
                self.direction[2],
                self.spread,
            ],
            velocity_min_lifetime: [
                self.velocity_min[0],
                self.velocity_min[1],
                self.velocity_min[2],
                self.lifetime_min,
            ],
            velocity_max_lifetime: [
                self.velocity_max[0],
                self.velocity_max[1],
                self.velocity_max[2],
                self.lifetime_max,
            ],
            start_color: self.start_color,
            end_color: self.end_color,
            size_params: [
                self.size_start - self.size_start_variance,
                self.size_start + self.size_start_variance,
                self.size_end - self.size_end_variance,
                self.size_end + self.size_end_variance,
            ],
            emitter_shape: shape_params,
            rotation_params: [
                self.rotation_speed_min,
                self.rotation_speed_max,
                0.0,
                0.0,
            ],
            start_color_variance: self.start_color_variance,
            end_color_variance: self.end_color_variance,
            box_half_extents: box_extents,
        }
    }
}
