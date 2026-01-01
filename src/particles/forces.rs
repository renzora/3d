//! Force field definitions for particle physics.

use bytemuck::{Pod, Zeroable};

/// Maximum number of force fields per particle system.
pub const MAX_FORCE_FIELDS: usize = 8;

/// Force field type for CPU configuration.
#[derive(Debug, Clone, Copy)]
pub enum ForceType {
    /// Constant directional force (gravity, wind).
    Directional {
        /// Force direction (normalized).
        direction: [f32; 3],
        /// Force strength.
        strength: f32,
    },
    /// Point attractor (positive strength) or repulsor (negative strength).
    Point {
        /// Center position in world space.
        position: [f32; 3],
        /// Force strength (positive = attract, negative = repel).
        strength: f32,
        /// Effect radius.
        radius: f32,
    },
    /// Turbulence/noise force for organic movement.
    Turbulence {
        /// Noise frequency (higher = more detail).
        frequency: f32,
        /// Force amplitude.
        amplitude: f32,
        /// Number of octaves for fractal noise.
        octaves: u32,
    },
    /// Vortex force around an axis.
    Vortex {
        /// Vortex axis (normalized).
        axis: [f32; 3],
        /// Vortex center position.
        position: [f32; 3],
        /// Rotational strength.
        strength: f32,
    },
    /// Drag/air resistance (slows particles down).
    Drag {
        /// Drag coefficient (0-1, higher = more drag).
        coefficient: f32,
    },
}

impl ForceType {
    /// Convert to GPU format.
    pub fn to_gpu(&self) -> ForceFieldGpu {
        match self {
            ForceType::Directional {
                direction,
                strength,
            } => ForceFieldGpu {
                type_enabled: [0.0, 0.0, 0.0, 1.0],
                position_strength: [direction[0], direction[1], direction[2], *strength],
                params: [0.0, 0.0, 0.0, 0.0],
            },
            ForceType::Point {
                position,
                strength,
                radius,
            } => ForceFieldGpu {
                type_enabled: [1.0, 0.0, 0.0, 1.0],
                position_strength: [position[0], position[1], position[2], *strength],
                params: [*radius, 0.0, 0.0, 0.0],
            },
            ForceType::Turbulence {
                frequency,
                amplitude,
                octaves,
            } => ForceFieldGpu {
                type_enabled: [2.0, 0.0, 0.0, 1.0],
                position_strength: [0.0, 0.0, 0.0, 0.0],
                params: [*frequency, *amplitude, *octaves as f32, 0.0],
            },
            ForceType::Vortex {
                axis,
                position,
                strength,
            } => ForceFieldGpu {
                type_enabled: [3.0, 0.0, 0.0, 1.0],
                position_strength: [axis[0], axis[1], axis[2], *strength],
                params: [position[0], position[1], position[2], 0.0],
            },
            ForceType::Drag { coefficient } => ForceFieldGpu {
                type_enabled: [4.0, 0.0, 0.0, 1.0],
                position_strength: [0.0, 0.0, 0.0, 0.0],
                params: [*coefficient, 0.0, 0.0, 0.0],
            },
        }
    }
}

/// Force field data for GPU uniform/storage buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ForceFieldGpu {
    /// Type (0=directional, 1=point, 2=turbulence, 3=vortex, 4=drag) and enabled flag.
    /// x=type, y=unused, z=unused, w=enabled (0 or 1).
    pub type_enabled: [f32; 4],
    /// Position/direction (xyz) + strength (w).
    pub position_strength: [f32; 4],
    /// Additional parameters depending on force type.
    /// Directional: unused
    /// Point: x=radius
    /// Turbulence: x=frequency, y=amplitude, z=octaves
    /// Vortex: xyz=center position
    /// Drag: x=coefficient
    pub params: [f32; 4],
}

impl Default for ForceFieldGpu {
    fn default() -> Self {
        Self {
            type_enabled: [0.0, 0.0, 0.0, 0.0], // Disabled
            position_strength: [0.0, 0.0, 0.0, 0.0],
            params: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Preset force configurations.
impl ForceType {
    /// Standard gravity (9.81 m/s^2 downward).
    pub fn gravity() -> Self {
        ForceType::Directional {
            direction: [0.0, -1.0, 0.0],
            strength: 9.81,
        }
    }

    /// Light gravity for floaty particles.
    pub fn light_gravity() -> Self {
        ForceType::Directional {
            direction: [0.0, -1.0, 0.0],
            strength: 2.0,
        }
    }

    /// Upward buoyancy for smoke/fire.
    pub fn buoyancy(strength: f32) -> Self {
        ForceType::Directional {
            direction: [0.0, 1.0, 0.0],
            strength,
        }
    }

    /// Horizontal wind.
    pub fn wind(direction: [f32; 3], strength: f32) -> Self {
        let len = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        let normalized = if len > 0.0 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [1.0, 0.0, 0.0]
        };
        ForceType::Directional {
            direction: normalized,
            strength,
        }
    }

    /// Attractor that pulls particles toward a point.
    pub fn attractor(position: [f32; 3], strength: f32, radius: f32) -> Self {
        ForceType::Point {
            position,
            strength,
            radius,
        }
    }

    /// Repulsor that pushes particles away from a point.
    pub fn repulsor(position: [f32; 3], strength: f32, radius: f32) -> Self {
        ForceType::Point {
            position,
            strength: -strength,
            radius,
        }
    }

    /// Light turbulence for organic movement.
    pub fn light_turbulence() -> Self {
        ForceType::Turbulence {
            frequency: 1.0,
            amplitude: 0.5,
            octaves: 2,
        }
    }

    /// Strong turbulence for chaotic movement.
    pub fn strong_turbulence() -> Self {
        ForceType::Turbulence {
            frequency: 2.0,
            amplitude: 2.0,
            octaves: 4,
        }
    }

    /// Air resistance.
    pub fn air_resistance(coefficient: f32) -> Self {
        ForceType::Drag { coefficient }
    }
}
