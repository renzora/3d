//! Sphere geometry.

use super::Vertex;
use bytemuck;
use std::f32::consts::PI;
use wgpu::util::DeviceExt;

/// A sphere geometry.
pub struct SphereGeometry {
    /// Radius.
    pub radius: f32,
    /// Width segments (longitude).
    pub width_segments: u32,
    /// Height segments (latitude).
    pub height_segments: u32,
    /// Phi start angle (horizontal).
    pub phi_start: f32,
    /// Phi length.
    pub phi_length: f32,
    /// Theta start angle (vertical).
    pub theta_start: f32,
    /// Theta length.
    pub theta_length: f32,
}

impl Default for SphereGeometry {
    fn default() -> Self {
        Self::new(1.0, 32, 16)
    }
}

impl SphereGeometry {
    /// Create a new sphere geometry.
    pub fn new(radius: f32, width_segments: u32, height_segments: u32) -> Self {
        Self {
            radius,
            width_segments: width_segments.max(3),
            height_segments: height_segments.max(2),
            phi_start: 0.0,
            phi_length: PI * 2.0,
            theta_start: 0.0,
            theta_length: PI,
        }
    }

    /// Create a partial sphere.
    pub fn partial(
        radius: f32,
        width_segments: u32,
        height_segments: u32,
        phi_start: f32,
        phi_length: f32,
        theta_start: f32,
        theta_length: f32,
    ) -> Self {
        Self {
            radius,
            width_segments: width_segments.max(3),
            height_segments: height_segments.max(2),
            phi_start,
            phi_length,
            theta_start,
            theta_length,
        }
    }

    /// Build the geometry and return vertex/index buffers.
    pub fn build(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut grid: Vec<Vec<u32>> = Vec::new();

        let mut index = 0u32;

        // Generate vertices
        for iy in 0..=self.height_segments {
            let mut row = Vec::new();
            let v = iy as f32 / self.height_segments as f32;
            let theta = self.theta_start + v * self.theta_length;

            for ix in 0..=self.width_segments {
                let u = ix as f32 / self.width_segments as f32;
                let phi = self.phi_start + u * self.phi_length;

                // Spherical coordinates to Cartesian
                let x = -self.radius * theta.sin() * phi.cos();
                let y = self.radius * theta.cos();
                let z = self.radius * theta.sin() * phi.sin();

                let position = [x, y, z];

                // Normal is just the normalized position for a sphere centered at origin
                let len = (x * x + y * y + z * z).sqrt();
                let normal = if len > 0.0 {
                    [x / len, y / len, z / len]
                } else {
                    [0.0, 1.0, 0.0]
                };

                let uv = [u, 1.0 - v];

                vertices.push(Vertex::new(position, normal, uv));
                row.push(index);
                index += 1;
            }

            grid.push(row);
        }

        // Generate indices
        for iy in 0..self.height_segments {
            for ix in 0..self.width_segments {
                let a = grid[iy as usize][(ix + 1) as usize];
                let b = grid[iy as usize][ix as usize];
                let c = grid[(iy + 1) as usize][ix as usize];
                let d = grid[(iy + 1) as usize][(ix + 1) as usize];

                // Skip degenerate triangles at poles
                if iy != 0 {
                    indices.push(a);
                    indices.push(b);
                    indices.push(d);
                }

                if iy != self.height_segments - 1 {
                    indices.push(b);
                    indices.push(c);
                    indices.push(d);
                }
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer, indices.len() as u32)
    }
}
