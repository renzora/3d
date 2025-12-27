//! Line rendering objects.

use crate::core::Id;
use crate::math::{Matrix4, Quaternion, Vector3};
use bytemuck::{Pod, Zeroable};

/// Line vertex with position and color.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LineVertex {
    /// Position.
    pub position: [f32; 3],
    /// Color (RGBA).
    pub color: [f32; 4],
}

impl LineVertex {
    /// Create a new line vertex.
    pub const fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }

    /// Create from Vector3 and color.
    pub fn from_vec3(position: Vector3, color: [f32; 4]) -> Self {
        Self {
            position: [position.x, position.y, position.z],
            color,
        }
    }

    /// Get the vertex buffer layout.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }

    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];
}

/// A continuous line through a series of points.
pub struct Line {
    /// Unique identifier.
    id: Id,
    /// Object name.
    name: String,
    /// Line vertices.
    vertices: Vec<LineVertex>,
    /// Vertex buffer.
    vertex_buffer: Option<wgpu::Buffer>,
    /// Whether buffer needs update.
    needs_update: bool,
    /// Local position.
    pub position: Vector3,
    /// Local rotation.
    pub rotation: Quaternion,
    /// Local scale.
    pub scale: Vector3,
    /// World matrix.
    world_matrix: Matrix4,
    /// Whether matrices need update.
    matrix_needs_update: bool,
    /// Visibility flag.
    pub visible: bool,
    /// Line width (if supported).
    pub line_width: f32,
}

impl Line {
    /// Create a new empty line.
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            name: String::new(),
            vertices: Vec::new(),
            vertex_buffer: None,
            needs_update: true,
            position: Vector3::ZERO,
            rotation: Quaternion::IDENTITY,
            scale: Vector3::ONE,
            world_matrix: Matrix4::IDENTITY,
            matrix_needs_update: true,
            visible: true,
            line_width: 1.0,
        }
    }

    /// Create a line from points with uniform color.
    pub fn from_points(points: &[Vector3], color: [f32; 4]) -> Self {
        let vertices = points
            .iter()
            .map(|p| LineVertex::from_vec3(*p, color))
            .collect();
        let mut line = Self::new();
        line.vertices = vertices;
        line
    }

    /// Create a line from point pairs (for line segments).
    pub fn from_pairs(pairs: &[(Vector3, Vector3)], color: [f32; 4]) -> Self {
        let mut vertices = Vec::with_capacity(pairs.len() * 2);
        for (start, end) in pairs {
            vertices.push(LineVertex::from_vec3(*start, color));
            vertices.push(LineVertex::from_vec3(*end, color));
        }
        let mut line = Self::new();
        line.vertices = vertices;
        line
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Get the vertices.
    #[inline]
    pub fn vertices(&self) -> &[LineVertex] {
        &self.vertices
    }

    /// Get mutable vertices.
    pub fn vertices_mut(&mut self) -> &mut Vec<LineVertex> {
        self.needs_update = true;
        &mut self.vertices
    }

    /// Set vertices.
    pub fn set_vertices(&mut self, vertices: Vec<LineVertex>) {
        self.vertices = vertices;
        self.needs_update = true;
    }

    /// Add a point.
    pub fn add_point(&mut self, point: Vector3, color: [f32; 4]) {
        self.vertices.push(LineVertex::from_vec3(point, color));
        self.needs_update = true;
    }

    /// Clear all points.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.needs_update = true;
    }

    /// Get vertex count.
    #[inline]
    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Check if buffer needs update.
    #[inline]
    pub fn needs_update(&self) -> bool {
        self.needs_update
    }

    /// Get the vertex buffer.
    #[inline]
    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    /// Update the matrix.
    pub fn update_matrix(&mut self) {
        if self.matrix_needs_update {
            self.world_matrix = Matrix4::compose(&self.position, &self.rotation, &self.scale);
            self.matrix_needs_update = false;
        }
    }

    /// Get the world matrix.
    pub fn world_matrix(&self) -> &Matrix4 {
        &self.world_matrix
    }

    /// Update GPU buffer.
    pub fn update_buffer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.needs_update || self.vertices.is_empty() {
            return;
        }

        let data = bytemuck::cast_slice(&self.vertices);

        if let Some(ref buffer) = self.vertex_buffer {
            if buffer.size() >= data.len() as u64 {
                queue.write_buffer(buffer, 0, data);
            } else {
                use wgpu::util::DeviceExt;
                self.vertex_buffer = Some(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Line Vertex Buffer"),
                        contents: data,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        } else {
            use wgpu::util::DeviceExt;
            self.vertex_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Line Vertex Buffer"),
                    contents: data,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            ));
        }

        self.needs_update = false;
    }
}

impl Default for Line {
    fn default() -> Self {
        Self::new()
    }
}

/// Line segments (pairs of vertices).
pub struct LineSegments {
    /// The underlying line.
    line: Line,
}

impl LineSegments {
    /// Create new empty line segments.
    pub fn new() -> Self {
        Self { line: Line::new() }
    }

    /// Create from point pairs.
    pub fn from_pairs(pairs: &[(Vector3, Vector3)], color: [f32; 4]) -> Self {
        Self {
            line: Line::from_pairs(pairs, color),
        }
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.line.id()
    }

    /// Get the underlying line.
    #[inline]
    pub fn line(&self) -> &Line {
        &self.line
    }

    /// Get mutable underlying line.
    #[inline]
    pub fn line_mut(&mut self) -> &mut Line {
        &mut self.line
    }

    /// Add a line segment.
    pub fn add_segment(&mut self, start: Vector3, end: Vector3, color: [f32; 4]) {
        self.line.add_point(start, color);
        self.line.add_point(end, color);
    }

    /// Get segment count.
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.line.vertices.len() / 2
    }

    /// Update GPU buffer.
    pub fn update_buffer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.line.update_buffer(device, queue);
    }
}

impl Default for LineSegments {
    fn default() -> Self {
        Self::new()
    }
}
