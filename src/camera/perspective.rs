//! Perspective camera.

use crate::core::Id;
use crate::math::{Matrix4, Vector3};

/// A perspective projection camera.
pub struct PerspectiveCamera {
    /// Unique ID.
    id: Id,
    /// Field of view in degrees.
    pub fov: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
    /// Camera position.
    pub position: Vector3,
    /// Camera target (look-at point).
    pub target: Vector3,
    /// Up vector.
    pub up: Vector3,
    /// View matrix.
    view_matrix: Matrix4,
    /// Projection matrix.
    projection_matrix: Matrix4,
    /// Combined view-projection matrix.
    view_projection_matrix: Matrix4,
    /// Whether matrices need updating.
    needs_update: bool,
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self::new(60.0, 16.0 / 9.0, 0.1, 1000.0)
    }
}

impl PerspectiveCamera {
    /// Create a new perspective camera.
    pub fn new(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let mut camera = Self {
            id: Id::new(),
            fov,
            aspect,
            near,
            far,
            position: Vector3::new(0.0, 0.0, 5.0),
            target: Vector3::ZERO,
            up: Vector3::UP,
            view_matrix: Matrix4::IDENTITY,
            projection_matrix: Matrix4::IDENTITY,
            view_projection_matrix: Matrix4::IDENTITY,
            needs_update: true,
        };
        camera.update_matrices();
        camera
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Set the camera position.
    pub fn set_position(&mut self, position: Vector3) {
        self.position = position;
        self.needs_update = true;
    }

    /// Set the camera target.
    pub fn set_target(&mut self, target: Vector3) {
        self.target = target;
        self.needs_update = true;
    }

    /// Look at a target from the current position.
    pub fn look_at(&mut self, target: Vector3) {
        self.target = target;
        self.needs_update = true;
    }

    /// Set the field of view.
    pub fn set_fov(&mut self, fov: f32) {
        self.fov = fov;
        self.needs_update = true;
    }

    /// Set the aspect ratio.
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
        self.needs_update = true;
    }

    /// Set near and far planes.
    pub fn set_clip_planes(&mut self, near: f32, far: f32) {
        self.near = near;
        self.far = far;
        self.needs_update = true;
    }

    /// Get the view matrix.
    pub fn view_matrix(&mut self) -> &Matrix4 {
        if self.needs_update {
            self.update_matrices();
        }
        &self.view_matrix
    }

    /// Get the projection matrix.
    pub fn projection_matrix(&mut self) -> &Matrix4 {
        if self.needs_update {
            self.update_matrices();
        }
        &self.projection_matrix
    }

    /// Get the combined view-projection matrix.
    pub fn view_projection_matrix(&mut self) -> &Matrix4 {
        if self.needs_update {
            self.update_matrices();
        }
        &self.view_projection_matrix
    }

    /// Update all matrices.
    pub fn update_matrices(&mut self) {
        self.view_matrix = Matrix4::look_at(&self.position, &self.target, &self.up);
        self.projection_matrix = Matrix4::perspective(
            self.fov.to_radians(),
            self.aspect,
            self.near,
            self.far,
        );
        self.view_projection_matrix = self.projection_matrix.multiply(&self.view_matrix);
        self.needs_update = false;
    }

    /// Get the forward direction.
    pub fn forward(&self) -> Vector3 {
        (self.target - self.position).normalized()
    }

    /// Get the right direction.
    pub fn right(&self) -> Vector3 {
        self.forward().cross(&self.up).normalized()
    }

    /// Orbit around the target.
    pub fn orbit(&mut self, delta_phi: f32, delta_theta: f32) {
        let offset = self.position - self.target;
        let radius = offset.length();

        // Convert to spherical coordinates
        let mut theta = offset.z.atan2(offset.x);
        let mut phi = (offset.y / radius).acos();

        // Apply delta
        theta += delta_phi;
        phi = (phi + delta_theta).clamp(0.01, std::f32::consts::PI - 0.01);

        // Convert back to Cartesian
        self.position = self.target + Vector3::new(
            radius * phi.sin() * theta.cos(),
            radius * phi.cos(),
            radius * phi.sin() * theta.sin(),
        );

        self.needs_update = true;
    }

    /// Dolly (move forward/backward).
    pub fn dolly(&mut self, distance: f32) {
        let direction = self.forward();
        self.position = self.position + direction * distance;
        self.needs_update = true;
    }

    /// Pan (move parallel to view plane).
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let right = self.right();
        let up = self.up;
        let offset = right * delta_x + up * delta_y;
        self.position = self.position + offset;
        self.target = self.target + offset;
        self.needs_update = true;
    }
}
