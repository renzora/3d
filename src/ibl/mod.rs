//! Image-Based Lighting (IBL) module.
//!
//! Provides tools for realistic environment lighting including:
//! - Pre-filtered environment maps for specular IBL
//! - Irradiance maps for diffuse IBL
//! - Spherical harmonics projection

pub mod prefilter;
pub mod irradiance;

pub use prefilter::PrefilterGenerator;
pub use irradiance::{IrradianceGenerator, generate_irradiance_cubemap, generate_sky_irradiance};
