//! Texture module for image and texture management.

mod sampler;
mod texture2d;

pub use sampler::{Sampler, SamplerDescriptor, AddressMode, FilterMode};
pub use texture2d::Texture2D;
