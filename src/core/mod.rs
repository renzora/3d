//! # Core Module
//!
//! Core engine functionality including wgpu context management,
//! rendering pipeline, and timing utilities.

mod engine;
mod context;
mod renderer;
mod clock;
mod id;

pub use engine::{Engine, EngineBuilder};
pub use context::{Context, ContextError};
pub use renderer::{Renderer, RenderInfo};
pub use clock::Clock;
pub use id::{Id, IdGenerator};

/// Render configuration options.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Enable anti-aliasing.
    pub antialias: bool,
    /// Enable alpha blending.
    pub alpha: bool,
    /// Enable depth testing.
    pub depth: bool,
    /// Enable stencil buffer.
    pub stencil: bool,
    /// Power preference for GPU selection.
    pub power_preference: wgpu::PowerPreference,
    /// Present mode (vsync).
    pub present_mode: wgpu::PresentMode,
    /// Clear color.
    pub clear_color: wgpu::Color,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            antialias: true,
            alpha: false,
            depth: true,
            stencil: false,
            power_preference: wgpu::PowerPreference::HighPerformance,
            present_mode: wgpu::PresentMode::AutoVsync,
            clear_color: wgpu::Color {
                r: 0.1,
                g: 0.1,
                b: 0.1,
                a: 1.0,
            },
        }
    }
}
