//! Animation system for keyframe-based animations.
//!
//! Supports animating positions, rotations, scales, and other properties
//! with various interpolation modes.

mod keyframe_track;
mod animation_clip;
mod animation_action;
mod animation_mixer;
mod interpolant;

pub use keyframe_track::*;
pub use animation_clip::*;
pub use animation_action::*;
pub use animation_mixer::*;
pub use interpolant::{Easing, InterpolationMode};
