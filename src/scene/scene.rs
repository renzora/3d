//! Scene container - the root of the scene graph.

use super::{Object3D, ObjectType};
use crate::core::Id;
use crate::math::Color;
use std::sync::{Arc, RwLock};

/// Background type for the scene.
#[derive(Debug, Clone)]
pub enum Background {
    /// Solid color background.
    Color(Color),
    /// Skybox texture (to be implemented).
    // Skybox(TextureId),
    /// No background (transparent).
    None,
}

impl Default for Background {
    fn default() -> Self {
        Self::None
    }
}

/// Fog type for the scene.
#[derive(Debug, Clone)]
pub enum Fog {
    /// No fog.
    None,
    /// Linear fog with near and far distances.
    Linear {
        /// Fog color.
        color: Color,
        /// Distance where fog starts.
        near: f32,
        /// Distance where fog is fully opaque.
        far: f32,
    },
    /// Exponential fog.
    Exponential {
        /// Fog color.
        color: Color,
        /// Fog density.
        density: f32,
    },
    /// Exponential squared fog.
    ExponentialSquared {
        /// Fog color.
        color: Color,
        /// Fog density.
        density: f32,
    },
}

impl Default for Fog {
    fn default() -> Self {
        Self::None
    }
}

/// The scene - root container for all objects.
pub struct Scene {
    /// The root object.
    root: Object3D,
    /// Scene background.
    background: Background,
    /// Scene fog.
    fog: Fog,
    /// Environment map (to be implemented).
    // environment: Option<TextureId>,
    /// Override material for all objects (debug).
    // override_material: Option<MaterialId>,
    /// Auto-update world matrices.
    auto_update: bool,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    /// Create a new empty scene.
    pub fn new() -> Self {
        let mut root = Object3D::with_type(ObjectType::Scene);
        root.set_name("Scene");

        Self {
            root,
            background: Background::None,
            fog: Fog::None,
            auto_update: true,
        }
    }

    /// Get the scene ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.root.id()
    }

    /// Get the scene name.
    #[inline]
    pub fn name(&self) -> &str {
        self.root.name()
    }

    /// Set the scene name.
    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.root.set_name(name);
    }

    /// Get the background.
    #[inline]
    pub fn background(&self) -> &Background {
        &self.background
    }

    /// Set the background.
    #[inline]
    pub fn set_background(&mut self, background: Background) {
        self.background = background;
    }

    /// Set background color.
    #[inline]
    pub fn set_background_color(&mut self, color: Color) {
        self.background = Background::Color(color);
    }

    /// Get the fog.
    #[inline]
    pub fn fog(&self) -> &Fog {
        &self.fog
    }

    /// Set the fog.
    #[inline]
    pub fn set_fog(&mut self, fog: Fog) {
        self.fog = fog;
    }

    /// Set linear fog.
    pub fn set_linear_fog(&mut self, color: Color, near: f32, far: f32) {
        self.fog = Fog::Linear { color, near, far };
    }

    /// Set exponential fog.
    pub fn set_exponential_fog(&mut self, color: Color, density: f32) {
        self.fog = Fog::Exponential { color, density };
    }

    /// Add an object to the scene.
    pub fn add(&mut self, object: Arc<RwLock<Object3D>>) {
        self.root.add(object);
    }

    /// Remove an object from the scene by ID.
    pub fn remove(&mut self, id: Id) -> Option<Arc<RwLock<Object3D>>> {
        self.root.remove_by_id(id)
    }

    /// Clear all objects from the scene.
    pub fn clear(&mut self) {
        self.root.clear();
    }

    /// Get the children (top-level objects).
    #[inline]
    pub fn children(&self) -> &[Arc<RwLock<Object3D>>] {
        self.root.children()
    }

    /// Get the number of top-level objects.
    #[inline]
    pub fn children_count(&self) -> usize {
        self.root.children_count()
    }

    /// Get the root object.
    #[inline]
    pub fn root(&self) -> &Object3D {
        &self.root
    }

    /// Get mutable root object.
    #[inline]
    pub fn root_mut(&mut self) -> &mut Object3D {
        &mut self.root
    }

    /// Update all world matrices in the scene.
    pub fn update_world_matrices(&mut self) {
        self.root.update_world_matrix(false, true);
    }

    /// Traverse all objects in the scene.
    pub fn traverse<F>(&self, mut callback: F)
    where
        F: FnMut(&Object3D),
    {
        self.traverse_recursive(&self.root, &mut callback);
    }

    fn traverse_recursive<F>(&self, object: &Object3D, callback: &mut F)
    where
        F: FnMut(&Object3D),
    {
        callback(object);
        for child in object.children() {
            if let Ok(child_guard) = child.read() {
                self.traverse_recursive(&child_guard, callback);
            }
        }
    }

    /// Traverse all visible objects in the scene.
    pub fn traverse_visible<F>(&self, mut callback: F)
    where
        F: FnMut(&Object3D),
    {
        self.traverse_visible_recursive(&self.root, &mut callback);
    }

    fn traverse_visible_recursive<F>(&self, object: &Object3D, callback: &mut F)
    where
        F: FnMut(&Object3D),
    {
        if !object.is_visible() {
            return;
        }
        callback(object);
        for child in object.children() {
            if let Ok(child_guard) = child.read() {
                self.traverse_visible_recursive(&child_guard, callback);
            }
        }
    }

    /// Find an object by name.
    pub fn find_by_name(&self, name: &str) -> Option<Arc<RwLock<Object3D>>> {
        self.find_by_name_recursive(self.root.children(), name)
    }

    fn find_by_name_recursive(
        &self,
        children: &[Arc<RwLock<Object3D>>],
        name: &str,
    ) -> Option<Arc<RwLock<Object3D>>> {
        for child in children {
            if let Ok(child_guard) = child.read() {
                if child_guard.name() == name {
                    return Some(Arc::clone(child));
                }
                if let Some(found) = self.find_by_name_recursive(child_guard.children(), name) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Find an object by ID.
    pub fn find_by_id(&self, id: Id) -> Option<Arc<RwLock<Object3D>>> {
        self.find_by_id_recursive(self.root.children(), id)
    }

    fn find_by_id_recursive(
        &self,
        children: &[Arc<RwLock<Object3D>>],
        id: Id,
    ) -> Option<Arc<RwLock<Object3D>>> {
        for child in children {
            if let Ok(child_guard) = child.read() {
                if child_guard.id() == id {
                    return Some(Arc::clone(child));
                }
                if let Some(found) = self.find_by_id_recursive(child_guard.children(), id) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Count total objects in the scene.
    pub fn count_objects(&self) -> usize {
        let mut count = 0;
        self.traverse(|_| count += 1);
        count
    }
}

impl std::fmt::Debug for Scene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scene")
            .field("id", &self.id())
            .field("name", &self.name())
            .field("children", &self.children_count())
            .field("background", &self.background)
            .field("fog", &self.fog)
            .finish()
    }
}
