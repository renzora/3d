//! Shadow mapping module for real-time shadows.
//!
//! This module provides comprehensive shadow mapping support including:
//! - Cascaded Shadow Maps (CSM) for directional lights
//! - Perspective shadow maps for spot lights
//! - Cube shadow maps for point lights
//! - PCF (Percentage Closer Filtering) for soft shadows
//!
//! # Example
//!
//! ```ignore
//! use ren::shadows::{ShadowConfig, ShadowQuality, PCFMode};
//!
//! // Create shadow configuration
//! let config = ShadowConfig::with_quality(ShadowQuality::High)
//!     .pcf_mode(PCFMode::Soft3x3)
//!     .bias(0.005);
//!
//! // Enable shadows on a directional light
//! let mut sun = DirectionalLight::sun();
//! sun.cast_shadow = true;
//! ```

mod cascade;
mod point_shadow;
mod shadow_config;
mod shadow_map;
mod shadow_pass;

pub use cascade::CascadedShadowMap;
pub use point_shadow::{CubeFace, PointShadowMap};
pub use shadow_config::{CascadeConfig, ContactShadowConfig, PCFMode, PCSSConfig, ShadowConfig, ShadowQuality};
pub use shadow_map::{ShadowAtlas, ShadowMap, ShadowUniform};
pub use shadow_pass::ShadowPass;

/// Maximum number of shadow-casting lights (excluding cascades).
pub const MAX_SHADOW_LIGHTS: usize = 4;

/// Maximum number of cascades for directional light CSM.
pub const MAX_CASCADES: usize = 4;
