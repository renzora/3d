//! # Ren - High-Performance WASM/wgpu 3D Engine
//!
//! Ren is a 3D rendering engine built with Rust, targeting WebGPU through wgpu.
//! It features Nanite-style virtualized geometry and Lumen-style global illumination.
//!
//! ## Features
//!
//! - **Math**: Complete 3D math library (vectors, matrices, quaternions)
//! - **Core**: wgpu context management, rendering pipeline
//! - **Scene**: Scene graph with Object3D hierarchy
//! - **Nanite**: Virtualized geometry with GPU-driven rendering
//! - **Lumen**: Global illumination with SDF tracing
//!
//! ## Example
//!
//! ```ignore
//! use ren::prelude::*;
//!
//! let engine = Engine::new().await?;
//! let scene = Scene::new();
//! let camera = PerspectiveCamera::new(75.0, 16.0 / 9.0, 0.1, 1000.0);
//!
//! let geometry = BoxGeometry::new(1.0, 1.0, 1.0);
//! let material = StandardMaterial::new();
//! let mesh = Mesh::new(geometry, material);
//! scene.add(mesh);
//!
//! engine.render(&scene, &camera);
//! ```

#![warn(missing_docs)]
#![allow(dead_code)]

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "web")]
use console_error_panic_hook;

pub mod math;
pub mod core;
pub mod scene;
pub mod geometry;
pub mod material;
pub mod camera;
pub mod texture;
pub mod light;
pub mod controls;
pub mod objects;
pub mod helpers;
pub mod animation;
pub mod loaders;
pub mod postprocessing;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub mod web;

// Re-export commonly used types
pub mod prelude {
    //! Convenient re-exports of commonly used types.

    pub use crate::math::*;
    pub use crate::core::*;
    pub use crate::scene::*;
    pub use crate::geometry::*;
    pub use crate::material::*;
    pub use crate::camera::*;
    pub use crate::texture::*;
    pub use crate::light::*;
    pub use crate::controls::*;
    pub use crate::objects::*;
    pub use crate::helpers::*;
    pub use crate::animation::*;
    pub use crate::loaders::*;
    pub use crate::postprocessing::*;
}

/// Initialize the engine for WASM environments.
/// Sets up panic hooks for better error messages in the browser console.
#[cfg(feature = "web")]
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Engine version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Engine name.
pub const NAME: &str = "Ren";
