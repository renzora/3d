//! Base class for all scene objects.

use super::{Layers, ObjectType, Transform, Visibility};
use crate::core::Id;
use crate::math::{Box3, Matrix4, Quaternion, Sphere, Vector3};
use std::sync::{Arc, RwLock, Weak};

/// Base class for all objects in the scene graph.
pub struct Object3D {
    /// Unique identifier.
    id: Id,
    /// Object name.
    name: String,
    /// Object type.
    object_type: ObjectType,
    /// Transform component.
    transform: Transform,
    /// Visibility settings.
    visibility: Visibility,
    /// Parent object.
    parent: Option<Weak<RwLock<Object3D>>>,
    /// Child objects.
    children: Vec<Arc<RwLock<Object3D>>>,
    /// User data.
    user_data: Option<Box<dyn std::any::Any + Send + Sync>>,
    /// World matrix needs auto-update.
    matrix_auto_update: bool,
    /// World matrix is up to date.
    matrix_world_needs_update: bool,
}

impl Default for Object3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Object3D {
    /// Create a new Object3D.
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            name: String::new(),
            object_type: ObjectType::Object3D,
            transform: Transform::new(),
            visibility: Visibility::new(),
            parent: None,
            children: Vec::new(),
            user_data: None,
            matrix_auto_update: true,
            matrix_world_needs_update: true,
        }
    }

    /// Create with a specific type.
    pub fn with_type(object_type: ObjectType) -> Self {
        let mut obj = Self::new();
        obj.object_type = object_type;
        obj
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the object name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the object name.
    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Get the object type.
    #[inline]
    pub fn object_type(&self) -> ObjectType {
        self.object_type
    }

    /// Get the transform.
    #[inline]
    pub fn transform(&self) -> &Transform {
        &self.transform
    }

    /// Get mutable transform.
    #[inline]
    pub fn transform_mut(&mut self) -> &mut Transform {
        self.matrix_world_needs_update = true;
        &mut self.transform
    }

    /// Get visibility settings.
    #[inline]
    pub fn visibility(&self) -> &Visibility {
        &self.visibility
    }

    /// Get mutable visibility settings.
    #[inline]
    pub fn visibility_mut(&mut self) -> &mut Visibility {
        &mut self.visibility
    }

    /// Check if visible.
    #[inline]
    pub fn is_visible(&self) -> bool {
        self.visibility.is_visible()
    }

    /// Set visibility.
    #[inline]
    pub fn set_visible(&mut self, visible: bool) {
        self.visibility.set_visible(visible);
    }

    /// Get the layers.
    #[inline]
    pub fn layers(&self) -> &Layers {
        self.visibility.layers()
    }

    /// Get mutable layers.
    #[inline]
    pub fn layers_mut(&mut self) -> &mut Layers {
        self.visibility.layers_mut()
    }

    // === Position shortcuts ===

    /// Get position.
    #[inline]
    pub fn position(&self) -> &Vector3 {
        &self.transform.position
    }

    /// Set position.
    #[inline]
    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.transform.set_position(x, y, z);
        self.matrix_world_needs_update = true;
    }

    /// Set position from vector.
    #[inline]
    pub fn set_position_vec(&mut self, position: Vector3) {
        self.transform.set_position_vec(position);
        self.matrix_world_needs_update = true;
    }

    // === Rotation shortcuts ===

    /// Get rotation (Euler angles).
    #[inline]
    pub fn rotation(&self) -> &crate::math::Euler {
        &self.transform.rotation
    }

    /// Get quaternion.
    #[inline]
    pub fn quaternion(&self) -> &Quaternion {
        &self.transform.quaternion
    }

    /// Set rotation from Euler angles.
    #[inline]
    pub fn set_rotation(&mut self, x: f32, y: f32, z: f32) {
        self.transform.set_rotation(x, y, z);
        self.matrix_world_needs_update = true;
    }

    /// Set rotation from quaternion.
    #[inline]
    pub fn set_quaternion(&mut self, quaternion: Quaternion) {
        self.transform.set_rotation_quaternion(quaternion);
        self.matrix_world_needs_update = true;
    }

    // === Scale shortcuts ===

    /// Get scale.
    #[inline]
    pub fn scale(&self) -> &Vector3 {
        &self.transform.scale
    }

    /// Set scale.
    #[inline]
    pub fn set_scale(&mut self, x: f32, y: f32, z: f32) {
        self.transform.set_scale(x, y, z);
        self.matrix_world_needs_update = true;
    }

    /// Set uniform scale.
    #[inline]
    pub fn set_scale_uniform(&mut self, s: f32) {
        self.transform.set_scale_uniform(s);
        self.matrix_world_needs_update = true;
    }

    // === Transform operations ===

    /// Translate by a vector.
    #[inline]
    pub fn translate(&mut self, v: &Vector3) {
        self.transform.translate(v);
        self.matrix_world_needs_update = true;
    }

    /// Rotate around X axis.
    #[inline]
    pub fn rotate_x(&mut self, angle: f32) {
        self.transform.rotate_x(angle);
        self.matrix_world_needs_update = true;
    }

    /// Rotate around Y axis.
    #[inline]
    pub fn rotate_y(&mut self, angle: f32) {
        self.transform.rotate_y(angle);
        self.matrix_world_needs_update = true;
    }

    /// Rotate around Z axis.
    #[inline]
    pub fn rotate_z(&mut self, angle: f32) {
        self.transform.rotate_z(angle);
        self.matrix_world_needs_update = true;
    }

    /// Look at a target.
    pub fn look_at(&mut self, target: &Vector3) {
        self.transform.look_at(target, &Vector3::UP);
        self.matrix_world_needs_update = true;
    }

    // === Matrix access ===

    /// Get the local matrix.
    pub fn local_matrix(&mut self) -> &Matrix4 {
        self.transform.local_matrix()
    }

    /// Get the world matrix.
    pub fn world_matrix(&self) -> &Matrix4 {
        self.transform.world_matrix()
    }

    /// Update the world matrix.
    pub fn update_world_matrix(&mut self, update_parents: bool, update_children: bool) {
        // Update parent first if requested
        if update_parents {
            if let Some(ref parent_weak) = self.parent {
                if let Some(parent) = parent_weak.upgrade() {
                    if let Ok(mut parent_guard) = parent.write() {
                        parent_guard.update_world_matrix(true, false);
                    }
                }
            }
        }

        // Update local matrix if needed
        if self.matrix_auto_update {
            self.transform.update_local_matrix();
        }

        // Update world matrix
        if let Some(ref parent_weak) = self.parent {
            if let Some(parent) = parent_weak.upgrade() {
                if let Ok(parent_guard) = parent.read() {
                    self.transform.update_world_matrix(Some(parent_guard.transform.world_matrix()));
                }
            }
        } else {
            self.transform.update_world_matrix(None);
        }

        self.matrix_world_needs_update = false;

        // Update children if requested
        if update_children {
            for child in &self.children {
                if let Ok(mut child_guard) = child.write() {
                    child_guard.update_world_matrix(false, true);
                }
            }
        }
    }

    // === Hierarchy ===

    /// Get the parent.
    pub fn parent(&self) -> Option<Arc<RwLock<Object3D>>> {
        self.parent.as_ref().and_then(|w| w.upgrade())
    }

    /// Check if has parent.
    #[inline]
    pub fn has_parent(&self) -> bool {
        self.parent.is_some()
    }

    /// Get children.
    #[inline]
    pub fn children(&self) -> &[Arc<RwLock<Object3D>>] {
        &self.children
    }

    /// Get number of children.
    #[inline]
    pub fn children_count(&self) -> usize {
        self.children.len()
    }

    /// Add a child.
    pub fn add(&mut self, child: Arc<RwLock<Object3D>>) {
        // Set parent reference on child
        // Note: This requires the parent to also be wrapped in Arc<RwLock<>>
        // For now we just add to children list
        self.children.push(child);
    }

    /// Remove a child by ID.
    pub fn remove_by_id(&mut self, id: Id) -> Option<Arc<RwLock<Object3D>>> {
        if let Some(pos) = self.children.iter().position(|c| {
            c.read().map(|guard| guard.id() == id).unwrap_or(false)
        }) {
            Some(self.children.remove(pos))
        } else {
            None
        }
    }

    /// Clear all children.
    pub fn clear(&mut self) {
        self.children.clear();
    }

    // === User data ===

    /// Set user data.
    pub fn set_user_data<T: Send + Sync + 'static>(&mut self, data: T) {
        self.user_data = Some(Box::new(data));
    }

    /// Get user data.
    pub fn user_data<T: 'static>(&self) -> Option<&T> {
        self.user_data.as_ref().and_then(|d| d.downcast_ref())
    }

    /// Get mutable user data.
    pub fn user_data_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.user_data.as_mut().and_then(|d| d.downcast_mut())
    }

    // === Bounds (to be overridden by subclasses) ===

    /// Get the local bounding box.
    pub fn bounding_box(&self) -> Box3 {
        Box3::EMPTY
    }

    /// Get the local bounding sphere.
    pub fn bounding_sphere(&self) -> Sphere {
        Sphere::new(Vector3::ZERO, 0.0)
    }

    /// Get the world bounding box.
    pub fn world_bounding_box(&self) -> Box3 {
        self.bounding_box().apply_matrix4(self.transform.world_matrix())
    }
}

impl std::fmt::Debug for Object3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Object3D")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("type", &self.object_type)
            .field("visible", &self.visibility.is_visible())
            .field("children", &self.children.len())
            .finish()
    }
}
