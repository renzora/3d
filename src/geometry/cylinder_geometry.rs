//! Cylinder geometry.

use super::Vertex;
use bytemuck;
use std::f32::consts::PI;
use wgpu::util::DeviceExt;

/// A cylinder geometry.
pub struct CylinderGeometry {
    /// Top radius.
    pub radius_top: f32,
    /// Bottom radius.
    pub radius_bottom: f32,
    /// Height.
    pub height: f32,
    /// Radial segments.
    pub radial_segments: u32,
    /// Height segments.
    pub height_segments: u32,
    /// Open ended (no caps).
    pub open_ended: bool,
    /// Theta start angle.
    pub theta_start: f32,
    /// Theta length.
    pub theta_length: f32,
}

impl Default for CylinderGeometry {
    fn default() -> Self {
        Self::new(1.0, 1.0, 1.0, 32, 1)
    }
}

impl CylinderGeometry {
    /// Create a new cylinder geometry.
    pub fn new(
        radius_top: f32,
        radius_bottom: f32,
        height: f32,
        radial_segments: u32,
        height_segments: u32,
    ) -> Self {
        Self {
            radius_top,
            radius_bottom,
            height,
            radial_segments: radial_segments.max(3),
            height_segments: height_segments.max(1),
            open_ended: false,
            theta_start: 0.0,
            theta_length: PI * 2.0,
        }
    }

    /// Create a cone (cylinder with top radius = 0).
    pub fn cone(radius: f32, height: f32, radial_segments: u32, height_segments: u32) -> Self {
        Self::new(0.0, radius, height, radial_segments, height_segments)
    }

    /// Set open ended.
    pub fn with_open_ended(mut self, open_ended: bool) -> Self {
        self.open_ended = open_ended;
        self
    }

    /// Build the geometry and return vertex/index buffers.
    pub fn build(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let half_height = self.height / 2.0;

        // Generate torso
        self.build_torso(&mut vertices, &mut indices, half_height);

        // Generate caps
        if !self.open_ended {
            if self.radius_top > 0.0 {
                self.build_cap(&mut vertices, &mut indices, true, half_height);
            }
            if self.radius_bottom > 0.0 {
                self.build_cap(&mut vertices, &mut indices, false, half_height);
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cylinder Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cylinder Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer, indices.len() as u32)
    }

    fn build_torso(&self, vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, half_height: f32) {
        let mut index_array: Vec<Vec<u32>> = Vec::new();
        let slope = (self.radius_bottom - self.radius_top) / self.height;

        let mut index = vertices.len() as u32;

        // Generate vertices
        for y in 0..=self.height_segments {
            let v = y as f32 / self.height_segments as f32;
            let radius = v * (self.radius_bottom - self.radius_top) + self.radius_top;
            let py = v * self.height - half_height;

            let mut row = Vec::new();

            for x in 0..=self.radial_segments {
                let u = x as f32 / self.radial_segments as f32;
                let theta = u * self.theta_length + self.theta_start;

                let sin_theta = theta.sin();
                let cos_theta = theta.cos();

                let px = radius * sin_theta;
                let pz = radius * cos_theta;

                // Normal
                let nx = sin_theta;
                let ny = slope;
                let nz = cos_theta;
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                let normal = [nx / len, ny / len, nz / len];

                vertices.push(Vertex::new([px, py, pz], normal, [u, 1.0 - v]));
                row.push(index);
                index += 1;
            }

            index_array.push(row);
        }

        // Generate indices
        for y in 0..self.height_segments {
            for x in 0..self.radial_segments {
                let a = index_array[y as usize][(x + 1) as usize];
                let b = index_array[y as usize][x as usize];
                let c = index_array[(y + 1) as usize][x as usize];
                let d = index_array[(y + 1) as usize][(x + 1) as usize];

                indices.push(a);
                indices.push(b);
                indices.push(d);

                indices.push(b);
                indices.push(c);
                indices.push(d);
            }
        }
    }

    fn build_cap(&self, vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, top: bool, half_height: f32) {
        let center_index = vertices.len() as u32;
        let radius = if top { self.radius_top } else { self.radius_bottom };
        let sign = if top { 1.0 } else { -1.0 };
        let py = sign * half_height;

        // Center vertex
        let normal = [0.0, sign, 0.0];
        vertices.push(Vertex::new([0.0, py, 0.0], normal, [0.5, 0.5]));

        // Generate ring vertices
        let first_vertex = vertices.len() as u32;

        for x in 0..=self.radial_segments {
            let u = x as f32 / self.radial_segments as f32;
            let theta = u * self.theta_length + self.theta_start;

            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let px = radius * sin_theta;
            let pz = radius * cos_theta;

            let uv = [cos_theta * 0.5 + 0.5, sin_theta * 0.5 * sign + 0.5];
            vertices.push(Vertex::new([px, py, pz], normal, uv));
        }

        // Generate indices
        for x in 0..self.radial_segments {
            let i = first_vertex + x;
            if top {
                indices.push(center_index);
                indices.push(i + 1);
                indices.push(i);
            } else {
                indices.push(center_index);
                indices.push(i);
                indices.push(i + 1);
            }
        }
    }
}
