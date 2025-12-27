//! # Scene Module
//!
//! Scene graph implementation with hierarchical transformations.
//! Provides Object3D as the base for all scene objects.

mod object3d;
mod scene;
mod transform;
mod visibility;

pub use object3d::Object3D;
pub use scene::Scene;
pub use transform::Transform;
pub use visibility::Visibility;

/// Object type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectType {
    /// Generic Object3D.
    Object3D,
    /// Scene root.
    Scene,
    /// Group node.
    Group,
    /// Mesh object.
    Mesh,
    /// Line object.
    Line,
    /// Points object.
    Points,
    /// Sprite object.
    Sprite,
    /// Camera.
    Camera,
    /// Light.
    Light,
    /// Bone for skeletal animation.
    Bone,
    /// Skeleton.
    Skeleton,
    /// Helper object.
    Helper,
}

/// Layer mask for object visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Layers {
    mask: u32,
}

impl Layers {
    /// Create layers with only layer 0 enabled.
    pub const DEFAULT: Self = Self { mask: 1 };

    /// Create layers with no layers enabled.
    pub const NONE: Self = Self { mask: 0 };

    /// Create layers with all layers enabled.
    pub const ALL: Self = Self { mask: u32::MAX };

    /// Create a new Layers.
    #[inline]
    pub const fn new() -> Self {
        Self::DEFAULT
    }

    /// Set a specific layer.
    #[inline]
    pub fn set(&mut self, layer: u8) {
        if layer < 32 {
            self.mask |= 1 << layer;
        }
    }

    /// Enable a specific layer.
    #[inline]
    pub fn enable(&mut self, layer: u8) {
        self.set(layer);
    }

    /// Disable a specific layer.
    #[inline]
    pub fn disable(&mut self, layer: u8) {
        if layer < 32 {
            self.mask &= !(1 << layer);
        }
    }

    /// Toggle a specific layer.
    #[inline]
    pub fn toggle(&mut self, layer: u8) {
        if layer < 32 {
            self.mask ^= 1 << layer;
        }
    }

    /// Enable all layers.
    #[inline]
    pub fn enable_all(&mut self) {
        self.mask = u32::MAX;
    }

    /// Disable all layers.
    #[inline]
    pub fn disable_all(&mut self) {
        self.mask = 0;
    }

    /// Check if a specific layer is enabled.
    #[inline]
    pub fn test(&self, layer: u8) -> bool {
        if layer < 32 {
            (self.mask & (1 << layer)) != 0
        } else {
            false
        }
    }

    /// Check if this layers mask intersects with another.
    #[inline]
    pub fn intersects(&self, other: &Layers) -> bool {
        (self.mask & other.mask) != 0
    }

    /// Get the raw mask value.
    #[inline]
    pub fn mask(&self) -> u32 {
        self.mask
    }
}
