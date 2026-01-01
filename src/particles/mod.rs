//! GPU-accelerated particle system module.
//!
//! This module provides a high-performance particle system using GPU compute shaders
//! for simulation and instanced rendering for display.

mod emitter;
mod forces;
mod gpu_resources;
mod particle;
mod particle_system;

pub use emitter::{EmitterConfig, EmitterShape, ParticleBlendMode, ParticlePreset};
pub use forces::{ForceFieldGpu, ForceType, MAX_FORCE_FIELDS};
pub use gpu_resources::{IndirectDrawArgs, ParticleGpuResources, SimulationUniform};
pub use particle::{EmissionParams, ParticleGpu};
pub use particle_system::ParticleSystem;
