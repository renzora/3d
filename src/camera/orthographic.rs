//! Orthographic camera.

use crate::core::Id;
use crate::math::{Matrix4, Vector3};

/// An orthographic projection camera.
pub struct OrthographicCamera {
    /// Unique ID.
    id: Id,
    /// Left plane.
    pub left: f32,
    /// Right plane.
    pub right: f32,
    /// Top plane.
    pub top: f32,
    /// Bottom plane.
    pub bottom: f32,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
    /// Zoom level.
    pub zoom: f32,
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

impl OrthographicCamera {
    /// Create a new orthographic camera.
    pub fn new(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) -> Self {
        let mut camera = Self {
            id: Id::new(),
            left,
            right,
            top,
            bottom,
            near,
            far,
            zoom: 1.0,
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

    /// Create from width and height (centered).
    pub fn from_size(width: f32, height: f32, near: f32, far: f32) -> Self {
        let half_width = width / 2.0;
        let half_height = height / 2.0;
        Self::new(-half_width, half_width, half_height, -half_height, near, far)
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

    /// Set zoom level.
    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.max(0.001);
        self.needs_update = true;
    }

    /// Set the frustum planes.
    pub fn set_frustum(&mut self, left: f32, right: f32, top: f32, bottom: f32) {
        self.left = left;
        self.right = right;
        self.top = top;
        self.bottom = bottom;
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

        let dx = (self.right - self.left) / (2.0 * self.zoom);
        let dy = (self.top - self.bottom) / (2.0 * self.zoom);
        let cx = (self.right + self.left) / 2.0;
        let cy = (self.top + self.bottom) / 2.0;

        self.projection_matrix = Matrix4::orthographic(
            cx - dx,
            cx + dx,
            cy + dy,
            cy - dy,
            self.near,
            self.far,
        );

        self.view_projection_matrix = self.projection_matrix.multiply(&self.view_matrix);
        self.needs_update = false;
    }
}
