//! Material module for shaders and rendering properties.

mod basic;
mod line;
mod pbr;
mod standard;

pub use basic::BasicMaterial;
pub use line::{LineMaterial, LineModelUniform};
pub use pbr::PbrCameraUniform;
pub use standard::{CameraUniform, ModelUniform, StandardMaterial};
