//! Material module for shaders and rendering properties.

mod basic;
mod line;
mod pbr;
mod pbr_lit;
mod pbr_textured;
mod standard;

pub use basic::BasicMaterial;
pub use line::{LineMaterial, LineModelUniform};
pub use pbr::{PbrCameraUniform, PbrMaterial, PbrMaterialUniform, PbrModelUniform};
pub use pbr_lit::LitPbrMaterial;
pub use pbr_textured::{TexturedPbrMaterial, TexturedPbrMaterialUniform};
pub use standard::{CameraUniform, ModelUniform, StandardMaterial};
