//! Torus geometry.

use super::Vertex;
use bytemuck;
use std::f32::consts::PI;
use wgpu::util::DeviceExt;

/// A torus (donut) geometry.
pub struct TorusGeometry {
    /// Major radius (distance from center to tube center).
    pub radius: f32,
    /// Tube radius.
    pub tube: f32,
    /// Radial segments.
    pub radial_segments: u32,
    /// Tubular segments.
    pub tubular_segments: u32,
    /// Arc angle.
    pub arc: f32,
}

impl Default for TorusGeometry {
    fn default() -> Self {
        Self::new(1.0, 0.4, 16, 48)
    }
}

impl TorusGeometry {
    /// Create a new torus geometry.
    pub fn new(radius: f32, tube: f32, radial_segments: u32, tubular_segments: u32) -> Self {
        Self {
            radius,
            tube,
            radial_segments: radial_segments.max(3),
            tubular_segments: tubular_segments.max(3),
            arc: PI * 2.0,
        }
    }

    /// Create a partial torus.
    pub fn partial(
        radius: f32,
        tube: f32,
        radial_segments: u32,
        tubular_segments: u32,
        arc: f32,
    ) -> Self {
        Self {
            radius,
            tube,
            radial_segments: radial_segments.max(3),
            tubular_segments: tubular_segments.max(3),
            arc,
        }
    }

    /// Build the geometry and return vertex/index buffers.
    pub fn build(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Generate vertices
        for j in 0..=self.radial_segments {
            for i in 0..=self.tubular_segments {
                let u = i as f32 / self.tubular_segments as f32 * self.arc;
                let v = j as f32 / self.radial_segments as f32 * PI * 2.0;

                // Position
                let x = (self.radius + self.tube * v.cos()) * u.cos();
                let y = self.tube * v.sin();
                let z = (self.radius + self.tube * v.cos()) * u.sin();

                // Normal
                let center_x = self.radius * u.cos();
                let center_z = self.radius * u.sin();
                let nx = x - center_x;
                let ny = y;
                let nz = z - center_z;
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                let normal = [nx / len, ny / len, nz / len];

                let uv = [
                    i as f32 / self.tubular_segments as f32,
                    j as f32 / self.radial_segments as f32,
                ];

                vertices.push(Vertex::new([x, y, z], normal, uv));
            }
        }

        // Generate indices
        for j in 1..=self.radial_segments {
            for i in 1..=self.tubular_segments {
                let a = (self.tubular_segments + 1) * j + i - 1;
                let b = (self.tubular_segments + 1) * (j - 1) + i - 1;
                let c = (self.tubular_segments + 1) * (j - 1) + i;
                let d = (self.tubular_segments + 1) * j + i;

                indices.push(a);
                indices.push(b);
                indices.push(d);

                indices.push(b);
                indices.push(c);
                indices.push(d);
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Torus Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Torus Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer, indices.len() as u32)
    }
}
