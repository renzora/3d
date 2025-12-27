//! Box helper for visualizing bounding boxes.

use crate::math::{Box3, Vector3};
use crate::objects::{Line, LineVertex};

/// Helper to visualize a bounding box wireframe.
pub struct BoxHelper {
    /// The line object.
    line: Line,
    /// Color.
    color: [f32; 4],
}

impl BoxHelper {
    /// Create a new box helper from a Box3.
    pub fn new(bbox: &Box3, color: [f32; 4]) -> Self {
        let mut helper = Self {
            line: Line::new(),
            color,
        };
        helper.set_from_box(bbox);
        helper
    }

    /// Create a unit box helper centered at origin.
    pub fn unit(color: [f32; 4]) -> Self {
        let bbox = Box3::new(
            Vector3::new(-0.5, -0.5, -0.5),
            Vector3::new(0.5, 0.5, 0.5),
        );
        Self::new(&bbox, color)
    }

    /// Set from a Box3.
    pub fn set_from_box(&mut self, bbox: &Box3) {
        let min = &bbox.min;
        let max = &bbox.max;

        // 8 corners
        let c000 = Vector3::new(min.x, min.y, min.z);
        let c001 = Vector3::new(min.x, min.y, max.z);
        let c010 = Vector3::new(min.x, max.y, min.z);
        let c011 = Vector3::new(min.x, max.y, max.z);
        let c100 = Vector3::new(max.x, min.y, min.z);
        let c101 = Vector3::new(max.x, min.y, max.z);
        let c110 = Vector3::new(max.x, max.y, min.z);
        let c111 = Vector3::new(max.x, max.y, max.z);

        let mut vertices = Vec::with_capacity(24);
        let color = self.color;

        // Bottom face
        vertices.push(LineVertex::from_vec3(c000, color));
        vertices.push(LineVertex::from_vec3(c100, color));
        vertices.push(LineVertex::from_vec3(c100, color));
        vertices.push(LineVertex::from_vec3(c101, color));
        vertices.push(LineVertex::from_vec3(c101, color));
        vertices.push(LineVertex::from_vec3(c001, color));
        vertices.push(LineVertex::from_vec3(c001, color));
        vertices.push(LineVertex::from_vec3(c000, color));

        // Top face
        vertices.push(LineVertex::from_vec3(c010, color));
        vertices.push(LineVertex::from_vec3(c110, color));
        vertices.push(LineVertex::from_vec3(c110, color));
        vertices.push(LineVertex::from_vec3(c111, color));
        vertices.push(LineVertex::from_vec3(c111, color));
        vertices.push(LineVertex::from_vec3(c011, color));
        vertices.push(LineVertex::from_vec3(c011, color));
        vertices.push(LineVertex::from_vec3(c010, color));

        // Vertical edges
        vertices.push(LineVertex::from_vec3(c000, color));
        vertices.push(LineVertex::from_vec3(c010, color));
        vertices.push(LineVertex::from_vec3(c100, color));
        vertices.push(LineVertex::from_vec3(c110, color));
        vertices.push(LineVertex::from_vec3(c101, color));
        vertices.push(LineVertex::from_vec3(c111, color));
        vertices.push(LineVertex::from_vec3(c001, color));
        vertices.push(LineVertex::from_vec3(c011, color));

        self.line.set_vertices(vertices);
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

    /// Set color.
    pub fn set_color(&mut self, color: [f32; 4]) {
        self.color = color;
        // Update vertex colors
        for vertex in self.line.vertices_mut() {
            vertex.color = color;
        }
    }

    /// Update GPU buffer.
    pub fn update_buffer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.line.update_buffer(device, queue);
    }
}

impl Default for BoxHelper {
    fn default() -> Self {
        Self::unit([1.0, 1.0, 0.0, 1.0]) // Yellow
    }
}
