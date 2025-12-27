//! Plane geometry.

use super::Vertex;
use bytemuck;
use wgpu::util::DeviceExt;

/// A plane geometry (flat rectangular surface).
pub struct PlaneGeometry {
    /// Width (X axis).
    pub width: f32,
    /// Height (Y axis, but lies in Z when unrotated).
    pub height: f32,
    /// Width segments.
    pub width_segments: u32,
    /// Height segments.
    pub height_segments: u32,
}

impl Default for PlaneGeometry {
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl PlaneGeometry {
    /// Create a new plane geometry.
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            width_segments: 1,
            height_segments: 1,
        }
    }

    /// Create with segments.
    pub fn with_segments(width: f32, height: f32, width_segments: u32, height_segments: u32) -> Self {
        Self {
            width,
            height,
            width_segments: width_segments.max(1),
            height_segments: height_segments.max(1),
        }
    }

    /// Build the geometry and return vertex/index buffers.
    pub fn build(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let half_width = self.width / 2.0;
        let half_height = self.height / 2.0;

        let segment_width = self.width / self.width_segments as f32;
        let segment_height = self.height / self.height_segments as f32;

        let normal = [0.0, 1.0, 0.0];

        // Generate vertices
        for iy in 0..=self.height_segments {
            let y = iy as f32 * segment_height - half_height;

            for ix in 0..=self.width_segments {
                let x = ix as f32 * segment_width - half_width;

                let position = [x, 0.0, -y]; // Plane lies in XZ plane, facing +Y
                let uv = [
                    ix as f32 / self.width_segments as f32,
                    1.0 - iy as f32 / self.height_segments as f32,
                ];

                vertices.push(Vertex::new(position, normal, uv));
            }
        }

        // Generate indices
        for iy in 0..self.height_segments {
            for ix in 0..self.width_segments {
                let a = ix + (self.width_segments + 1) * iy;
                let b = ix + (self.width_segments + 1) * (iy + 1);
                let c = (ix + 1) + (self.width_segments + 1) * (iy + 1);
                let d = (ix + 1) + (self.width_segments + 1) * iy;

                indices.push(a);
                indices.push(b);
                indices.push(d);

                indices.push(b);
                indices.push(c);
                indices.push(d);
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plane Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plane Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer, indices.len() as u32)
    }
}
