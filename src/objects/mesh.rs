//! Mesh object combining geometry and material.

use crate::core::Id;
use crate::geometry::BufferGeometry;
use crate::math::{Box3, Matrix4, Quaternion, Sphere, Vector3};
use std::sync::Arc;

/// A mesh is a renderable object with geometry and material.
pub struct Mesh {
    /// Unique identifier.
    id: Id,
    /// Object name.
    name: String,
    /// Geometry data.
    geometry: Arc<BufferGeometry>,
    /// Material index (references material in renderer).
    material_index: usize,
    /// Local position.
    pub position: Vector3,
    /// Local rotation.
    pub rotation: Quaternion,
    /// Local scale.
    pub scale: Vector3,
    /// Local matrix (cached).
    local_matrix: Matrix4,
    /// World matrix.
    world_matrix: Matrix4,
    /// Whether matrices need update.
    matrix_needs_update: bool,
    /// Visibility flag.
    pub visible: bool,
    /// Cast shadows.
    pub cast_shadow: bool,
    /// Receive shadows.
    pub receive_shadow: bool,
    /// Frustum culling enabled.
    pub frustum_culled: bool,
    /// Render order (for transparency sorting).
    pub render_order: i32,
}

impl Mesh {
    /// Create a new mesh with geometry.
    pub fn new(geometry: Arc<BufferGeometry>) -> Self {
        Self {
            id: Id::new(),
            name: String::new(),
            geometry,
            material_index: 0,
            position: Vector3::ZERO,
            rotation: Quaternion::IDENTITY,
            scale: Vector3::ONE,
            local_matrix: Matrix4::IDENTITY,
            world_matrix: Matrix4::IDENTITY,
            matrix_needs_update: true,
            visible: true,
            cast_shadow: true,
            receive_shadow: true,
            frustum_culled: true,
            render_order: 0,
        }
    }

    /// Create a mesh with geometry and material index.
    pub fn with_material(geometry: Arc<BufferGeometry>, material_index: usize) -> Self {
        let mut mesh = Self::new(geometry);
        mesh.material_index = material_index;
        mesh
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
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Get the geometry.
    #[inline]
    pub fn geometry(&self) -> &BufferGeometry {
        &self.geometry
    }

    /// Get the material index.
    #[inline]
    pub fn material_index(&self) -> usize {
        self.material_index
    }

    /// Set the material index.
    pub fn set_material_index(&mut self, index: usize) {
        self.material_index = index;
    }

    /// Set position.
    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.position = Vector3::new(x, y, z);
        self.matrix_needs_update = true;
    }

    /// Set position from vector.
    pub fn set_position_vec(&mut self, position: Vector3) {
        self.position = position;
        self.matrix_needs_update = true;
    }

    /// Set rotation from quaternion.
    pub fn set_rotation(&mut self, rotation: Quaternion) {
        self.rotation = rotation;
        self.matrix_needs_update = true;
    }

    /// Set rotation from Euler angles (radians).
    pub fn set_rotation_euler(&mut self, x: f32, y: f32, z: f32) {
        use crate::math::{Euler, EulerOrder};
        self.rotation = Quaternion::from_euler(&Euler::new(x, y, z, EulerOrder::XYZ));
        self.matrix_needs_update = true;
    }

    /// Set scale.
    pub fn set_scale(&mut self, x: f32, y: f32, z: f32) {
        self.scale = Vector3::new(x, y, z);
        self.matrix_needs_update = true;
    }

    /// Set uniform scale.
    pub fn set_scale_uniform(&mut self, s: f32) {
        self.scale = Vector3::new(s, s, s);
        self.matrix_needs_update = true;
    }

    /// Translate by a vector.
    pub fn translate(&mut self, v: &Vector3) {
        self.position = self.position + *v;
        self.matrix_needs_update = true;
    }

    /// Rotate around X axis (radians).
    pub fn rotate_x(&mut self, angle: f32) {
        self.rotation = self.rotation * Quaternion::from_axis_angle(&Vector3::UNIT_X, angle);
        self.matrix_needs_update = true;
    }

    /// Rotate around Y axis (radians).
    pub fn rotate_y(&mut self, angle: f32) {
        self.rotation = self.rotation * Quaternion::from_axis_angle(&Vector3::UNIT_Y, angle);
        self.matrix_needs_update = true;
    }

    /// Rotate around Z axis (radians).
    pub fn rotate_z(&mut self, angle: f32) {
        self.rotation = self.rotation * Quaternion::from_axis_angle(&Vector3::UNIT_Z, angle);
        self.matrix_needs_update = true;
    }

    /// Update the local matrix from position, rotation, scale.
    pub fn update_matrix(&mut self) {
        if self.matrix_needs_update {
            self.local_matrix = Matrix4::compose(&self.position, &self.rotation, &self.scale);
            self.world_matrix = self.local_matrix;
            self.matrix_needs_update = false;
        }
    }

    /// Update world matrix with parent matrix.
    pub fn update_world_matrix(&mut self, parent_world: Option<&Matrix4>) {
        self.update_matrix();
        if let Some(parent) = parent_world {
            self.world_matrix = *parent * self.local_matrix;
        } else {
            self.world_matrix = self.local_matrix;
        }
    }

    /// Get the local matrix.
    pub fn local_matrix(&mut self) -> &Matrix4 {
        self.update_matrix();
        &self.local_matrix
    }

    /// Get the world matrix.
    pub fn world_matrix(&self) -> &Matrix4 {
        &self.world_matrix
    }

    /// Get the world matrix data as array for GPU.
    pub fn world_matrix_array(&self) -> [[f32; 4]; 4] {
        self.world_matrix.to_cols_array_2d()
    }

    /// Get the local bounding box.
    pub fn bounding_box(&self) -> Box3 {
        self.geometry.bounding_box().cloned().unwrap_or(Box3::EMPTY)
    }

    /// Get the local bounding sphere.
    pub fn bounding_sphere(&self) -> Sphere {
        self.geometry.bounding_sphere().cloned().unwrap_or(Sphere::new(Vector3::ZERO, 0.0))
    }

    /// Get the world bounding box.
    pub fn world_bounding_box(&self) -> Box3 {
        self.bounding_box().apply_matrix4(&self.world_matrix)
    }

    /// Look at a target point.
    pub fn look_at(&mut self, target: &Vector3) {
        // Create a look-at matrix and extract rotation from it
        let look_matrix = Matrix4::look_at(&self.position, target, &Vector3::UP);
        let (_, rotation, _) = look_matrix.decompose();
        self.rotation = rotation;
        self.matrix_needs_update = true;
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new(Arc::new(BufferGeometry::new()))
    }
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("visible", &self.visible)
            .field("position", &self.position)
            .finish()
    }
}
