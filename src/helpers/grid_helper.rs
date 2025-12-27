//! Grid helper for visualizing a ground plane.

use crate::math::Vector3;
use crate::objects::{Line, LineVertex};

/// Helper to visualize a grid on the XZ plane.
pub struct GridHelper {
    /// The line object.
    line: Line,
}

impl GridHelper {
    /// Create a new grid helper.
    ///
    /// # Arguments
    /// * `size` - Total size of the grid (side length)
    /// * `divisions` - Number of divisions per side
    /// * `color1` - Color for main lines
    /// * `color2` - Color for secondary lines
    pub fn new(size: f32, divisions: usize, color1: [f32; 4], color2: [f32; 4]) -> Self {
        let half_size = size / 2.0;
        let step = size / divisions as f32;
        let center = divisions / 2;

        let mut vertices = Vec::new();

        for i in 0..=divisions {
            let pos = -half_size + (i as f32 * step);
            let color = if i == center { color1 } else { color2 };

            // Line parallel to Z axis
            vertices.push(LineVertex::new([pos, 0.0, -half_size], color));
            vertices.push(LineVertex::new([pos, 0.0, half_size], color));

            // Line parallel to X axis
            vertices.push(LineVertex::new([-half_size, 0.0, pos], color));
            vertices.push(LineVertex::new([half_size, 0.0, pos], color));
        }

        let mut line = Line::new();
        line.set_vertices(vertices);

        Self { line }
    }

    /// Create a simple grid with default colors.
    pub fn simple(size: f32, divisions: usize) -> Self {
        Self::new(
            size,
            divisions,
            [0.6, 0.6, 0.6, 1.0], // Light gray for center
            [0.3, 0.3, 0.3, 1.0], // Dark gray for others
        )
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

impl Default for GridHelper {
    fn default() -> Self {
        Self::simple(10.0, 10)
    }
}
