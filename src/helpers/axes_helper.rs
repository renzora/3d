//! Axes helper for visualizing coordinate system.

use crate::math::Vector3;
use crate::objects::{Line, LineVertex};

/// Helper to visualize the XYZ axes.
pub struct AxesHelper {
    /// The line object.
    line: Line,
}

impl AxesHelper {
    /// Create a new axes helper with specified size.
    pub fn new(size: f32) -> Self {
        let mut vertices = Vec::with_capacity(6);

        // X axis (red)
        vertices.push(LineVertex::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]));
        vertices.push(LineVertex::new([size, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]));

        // Y axis (green)
        vertices.push(LineVertex::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]));
        vertices.push(LineVertex::new([0.0, size, 0.0], [0.0, 1.0, 0.0, 1.0]));

        // Z axis (blue)
        vertices.push(LineVertex::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]));
        vertices.push(LineVertex::new([0.0, 0.0, size], [0.0, 0.0, 1.0, 1.0]));

        let mut line = Line::new();
        line.set_vertices(vertices);

        Self { line }
    }

    /// Get the underlying line.
    #[inline]
    pub fn line(&self) -> &Line {
        &self.line
    }

    /// Get mutable line.
    #[inline]
    pub fn line_mut(&mut self) -> &mut Line {
        &mut self.line
    }

    /// Set position.
    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.line.position = Vector3::new(x, y, z);
    }

    /// Update GPU buffer.
    pub fn update_buffer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.line.update_buffer(device, queue);
    }
}

impl Default for AxesHelper {
    fn default() -> Self {
        Self::new(1.0)
    }
}
