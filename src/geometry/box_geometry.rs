//! Box geometry (rectangular cuboid).

use super::Vertex;
use bytemuck;
use wgpu::util::DeviceExt;

/// A box (rectangular cuboid) geometry.
pub struct BoxGeometry {
    /// Width (X axis).
    pub width: f32,
    /// Height (Y axis).
    pub height: f32,
    /// Depth (Z axis).
    pub depth: f32,
    /// Width segments.
    pub width_segments: u32,
    /// Height segments.
    pub height_segments: u32,
    /// Depth segments.
    pub depth_segments: u32,
}

impl Default for BoxGeometry {
    fn default() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
}

impl BoxGeometry {
    /// Create a new box geometry.
    pub fn new(width: f32, height: f32, depth: f32) -> Self {
        Self {
            width,
            height,
            depth,
            width_segments: 1,
            height_segments: 1,
            depth_segments: 1,
        }
    }

    /// Create with segments.
    pub fn with_segments(
        width: f32,
        height: f32,
        depth: f32,
        width_segments: u32,
        height_segments: u32,
        depth_segments: u32,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            width_segments: width_segments.max(1),
            height_segments: height_segments.max(1),
            depth_segments: depth_segments.max(1),
        }
    }

    /// Build the geometry and return vertex/index buffers.
    pub fn build(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let half_width = self.width / 2.0;
        let half_height = self.height / 2.0;
        let half_depth = self.depth / 2.0;

        // Build each face
        // +X face
        self.build_plane(
            &mut vertices,
            &mut indices,
            2, 1, 0, // u, v, w axes
            -1.0, -1.0, // u_dir, v_dir
            self.depth, self.height, self.width,
            self.depth_segments, self.height_segments,
            half_width, // offset
            [1.0, 0.0, 0.0], // normal
        );

        // -X face
        self.build_plane(
            &mut vertices,
            &mut indices,
            2, 1, 0,
            1.0, -1.0,
            self.depth, self.height, self.width,
            self.depth_segments, self.height_segments,
            -half_width,
            [-1.0, 0.0, 0.0],
        );

        // +Y face
        self.build_plane(
            &mut vertices,
            &mut indices,
            0, 2, 1,
            1.0, 1.0,
            self.width, self.depth, self.height,
            self.width_segments, self.depth_segments,
            half_height,
            [0.0, 1.0, 0.0],
        );

        // -Y face
        self.build_plane(
            &mut vertices,
            &mut indices,
            0, 2, 1,
            1.0, -1.0,
            self.width, self.depth, self.height,
            self.width_segments, self.depth_segments,
            -half_height,
            [0.0, -1.0, 0.0],
        );

        // +Z face
        self.build_plane(
            &mut vertices,
            &mut indices,
            0, 1, 2,
            1.0, -1.0,
            self.width, self.height, self.depth,
            self.width_segments, self.height_segments,
            half_depth,
            [0.0, 0.0, 1.0],
        );

        // -Z face
        self.build_plane(
            &mut vertices,
            &mut indices,
            0, 1, 2,
            -1.0, -1.0,
            self.width, self.height, self.depth,
            self.width_segments, self.height_segments,
            -half_depth,
            [0.0, 0.0, -1.0],
        );

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Box Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Box Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer, indices.len() as u32)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_plane(
        &self,
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
        u: usize,
        v: usize,
        w: usize,
        u_dir: f32,
        v_dir: f32,
        width: f32,
        height: f32,
        _depth: f32,
        grid_x: u32,
        grid_y: u32,
        offset: f32,
        normal: [f32; 3],
    ) {
        let segment_width = width / grid_x as f32;
        let segment_height = height / grid_y as f32;
        let half_width = width / 2.0;
        let half_height = height / 2.0;

        let vertex_offset = vertices.len() as u32;

        for iy in 0..=grid_y {
            let y = (iy as f32 * segment_height - half_height) * v_dir;

            for ix in 0..=grid_x {
                let x = (ix as f32 * segment_width - half_width) * u_dir;

                let mut position = [0.0f32; 3];
                position[u] = x;
                position[v] = y;
                position[w] = offset;

                let uv = [
                    ix as f32 / grid_x as f32,
                    1.0 - iy as f32 / grid_y as f32,
                ];

                vertices.push(Vertex::new(position, normal, uv));
            }
        }

        // Generate indices
        for iy in 0..grid_y {
            for ix in 0..grid_x {
                let a = vertex_offset + ix + (grid_x + 1) * iy;
                let b = vertex_offset + ix + (grid_x + 1) * (iy + 1);
                let c = vertex_offset + (ix + 1) + (grid_x + 1) * (iy + 1);
                let d = vertex_offset + (ix + 1) + (grid_x + 1) * iy;

                indices.push(a);
                indices.push(b);
                indices.push(d);

                indices.push(b);
                indices.push(c);
                indices.push(d);
            }
        }
    }
}
