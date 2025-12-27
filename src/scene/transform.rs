//! Transform component for scene objects.

use crate::math::{Euler, Matrix4, Quaternion, Vector3};

/// Transform component containing position, rotation, and scale.
#[derive(Debug, Clone)]
pub struct Transform {
    /// Local position.
    pub position: Vector3,
    /// Local rotation as Euler angles.
    pub rotation: Euler,
    /// Local rotation as quaternion.
    pub quaternion: Quaternion,
    /// Local scale.
    pub scale: Vector3,
    /// Local transformation matrix.
    local_matrix: Matrix4,
    /// World transformation matrix.
    world_matrix: Matrix4,
    /// Whether the local matrix needs updating.
    local_matrix_dirty: bool,
    /// Whether the world matrix needs updating.
    world_matrix_dirty: bool,
    /// Use quaternion for rotation (vs euler).
    use_quaternion: bool,
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform {
    /// Create a new identity transform.
    pub fn new() -> Self {
        Self {
            position: Vector3::ZERO,
            rotation: Euler::ZERO,
            quaternion: Quaternion::IDENTITY,
            scale: Vector3::ONE,
            local_matrix: Matrix4::IDENTITY,
            world_matrix: Matrix4::IDENTITY,
            local_matrix_dirty: false,
            world_matrix_dirty: false,
            use_quaternion: false,
        }
    }

    /// Create a transform from position.
    pub fn from_position(position: Vector3) -> Self {
        let mut t = Self::new();
        t.position = position;
        t.local_matrix_dirty = true;
        t
    }

    /// Create a transform from position, rotation, and scale.
    pub fn from_components(position: Vector3, rotation: Euler, scale: Vector3) -> Self {
        let mut t = Self::new();
        t.position = position;
        t.rotation = rotation;
        t.quaternion = Quaternion::from_euler(&rotation);
        t.scale = scale;
        t.local_matrix_dirty = true;
        t
    }

    /// Set position.
    #[inline]
    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.position.set(x, y, z);
        self.local_matrix_dirty = true;
    }

    /// Set position from vector.
    #[inline]
    pub fn set_position_vec(&mut self, position: Vector3) {
        self.position = position;
        self.local_matrix_dirty = true;
    }

    /// Set rotation from Euler angles.
    #[inline]
    pub fn set_rotation(&mut self, x: f32, y: f32, z: f32) {
        self.rotation.x = x;
        self.rotation.y = y;
        self.rotation.z = z;
        self.quaternion = Quaternion::from_euler(&self.rotation);
        self.local_matrix_dirty = true;
    }

    /// Set rotation from Euler.
    #[inline]
    pub fn set_rotation_euler(&mut self, euler: Euler) {
        self.rotation = euler;
        self.quaternion = Quaternion::from_euler(&euler);
        self.local_matrix_dirty = true;
    }

    /// Set rotation from quaternion.
    #[inline]
    pub fn set_rotation_quaternion(&mut self, quaternion: Quaternion) {
        self.quaternion = quaternion;
        self.rotation = Euler::from_quaternion(&quaternion, self.rotation.order);
        self.use_quaternion = true;
        self.local_matrix_dirty = true;
    }

    /// Set scale.
    #[inline]
    pub fn set_scale(&mut self, x: f32, y: f32, z: f32) {
        self.scale.set(x, y, z);
        self.local_matrix_dirty = true;
    }

    /// Set uniform scale.
    #[inline]
    pub fn set_scale_uniform(&mut self, s: f32) {
        self.scale = Vector3::splat(s);
        self.local_matrix_dirty = true;
    }

    /// Translate by a vector.
    #[inline]
    pub fn translate(&mut self, v: &Vector3) {
        self.position += *v;
        self.local_matrix_dirty = true;
    }

    /// Translate along X axis.
    #[inline]
    pub fn translate_x(&mut self, distance: f32) {
        self.position.x += distance;
        self.local_matrix_dirty = true;
    }

    /// Translate along Y axis.
    #[inline]
    pub fn translate_y(&mut self, distance: f32) {
        self.position.y += distance;
        self.local_matrix_dirty = true;
    }

    /// Translate along Z axis.
    #[inline]
    pub fn translate_z(&mut self, distance: f32) {
        self.position.z += distance;
        self.local_matrix_dirty = true;
    }

    /// Rotate around X axis.
    #[inline]
    pub fn rotate_x(&mut self, angle: f32) {
        let q = Quaternion::from_axis_angle(&Vector3::UNIT_X, angle);
        self.quaternion = self.quaternion.multiply(&q);
        self.rotation = Euler::from_quaternion(&self.quaternion, self.rotation.order);
        self.local_matrix_dirty = true;
    }

    /// Rotate around Y axis.
    #[inline]
    pub fn rotate_y(&mut self, angle: f32) {
        let q = Quaternion::from_axis_angle(&Vector3::UNIT_Y, angle);
        self.quaternion = self.quaternion.multiply(&q);
        self.rotation = Euler::from_quaternion(&self.quaternion, self.rotation.order);
        self.local_matrix_dirty = true;
    }

    /// Rotate around Z axis.
    #[inline]
    pub fn rotate_z(&mut self, angle: f32) {
        let q = Quaternion::from_axis_angle(&Vector3::UNIT_Z, angle);
        self.quaternion = self.quaternion.multiply(&q);
        self.rotation = Euler::from_quaternion(&self.quaternion, self.rotation.order);
        self.local_matrix_dirty = true;
    }

    /// Rotate around an arbitrary axis.
    #[inline]
    pub fn rotate_on_axis(&mut self, axis: &Vector3, angle: f32) {
        let q = Quaternion::from_axis_angle(axis, angle);
        self.quaternion = self.quaternion.multiply(&q);
        self.rotation = Euler::from_quaternion(&self.quaternion, self.rotation.order);
        self.local_matrix_dirty = true;
    }

    /// Look at a target position.
    pub fn look_at(&mut self, target: &Vector3, up: &Vector3) {
        let m = Matrix4::look_at(&self.position, target, up);
        self.quaternion = Quaternion::from_matrix4(&m);
        self.rotation = Euler::from_quaternion(&self.quaternion, self.rotation.order);
        self.local_matrix_dirty = true;
    }

    /// Get the local transformation matrix.
    pub fn local_matrix(&mut self) -> &Matrix4 {
        if self.local_matrix_dirty {
            self.update_local_matrix();
        }
        &self.local_matrix
    }

    /// Get the world transformation matrix.
    pub fn world_matrix(&self) -> &Matrix4 {
        &self.world_matrix
    }

    /// Update the local matrix from position, rotation, and scale.
    pub fn update_local_matrix(&mut self) {
        self.local_matrix = Matrix4::compose(&self.position, &self.quaternion, &self.scale);
        self.local_matrix_dirty = false;
        self.world_matrix_dirty = true;
    }

    /// Update the world matrix given a parent's world matrix.
    pub fn update_world_matrix(&mut self, parent_world_matrix: Option<&Matrix4>) {
        if self.local_matrix_dirty {
            self.update_local_matrix();
        }

        match parent_world_matrix {
            Some(parent) => {
                self.world_matrix = parent.multiply(&self.local_matrix);
            }
            None => {
                self.world_matrix = self.local_matrix;
            }
        }
        self.world_matrix_dirty = false;
    }

    /// Check if the local matrix needs updating.
    #[inline]
    pub fn is_local_dirty(&self) -> bool {
        self.local_matrix_dirty
    }

    /// Check if the world matrix needs updating.
    #[inline]
    pub fn is_world_dirty(&self) -> bool {
        self.world_matrix_dirty
    }

    /// Mark transform as dirty.
    #[inline]
    pub fn set_dirty(&mut self) {
        self.local_matrix_dirty = true;
        self.world_matrix_dirty = true;
    }

    /// Get the world position.
    pub fn world_position(&self) -> Vector3 {
        self.world_matrix.get_position()
    }

    /// Get the world scale.
    pub fn world_scale(&self) -> Vector3 {
        self.world_matrix.get_scale()
    }

    /// Get the world quaternion.
    pub fn world_quaternion(&self) -> Quaternion {
        let (_, q, _) = self.world_matrix.decompose();
        q
    }

    /// Get the forward direction (local -Z in world space).
    pub fn forward(&self) -> Vector3 {
        self.world_matrix.transform_direction(&Vector3::FORWARD)
    }

    /// Get the up direction (local +Y in world space).
    pub fn up(&self) -> Vector3 {
        self.world_matrix.transform_direction(&Vector3::UP)
    }

    /// Get the right direction (local +X in world space).
    pub fn right(&self) -> Vector3 {
        self.world_matrix.transform_direction(&Vector3::RIGHT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_transform() {
        let t = Transform::new();
        assert!(t.position.approx_eq(&Vector3::ZERO, 1e-6));
        assert!(t.scale.approx_eq(&Vector3::ONE, 1e-6));
    }

    #[test]
    fn test_translation() {
        let mut t = Transform::new();
        t.set_position(1.0, 2.0, 3.0);
        let m = t.local_matrix();
        let pos = m.get_position();
        assert!(pos.approx_eq(&Vector3::new(1.0, 2.0, 3.0), 1e-6));
    }
}
