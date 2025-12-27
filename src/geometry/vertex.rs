//! Vertex types and layouts.

use bytemuck::{Pod, Zeroable};

/// Standard vertex with position, normal, and UV coordinates.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    /// Position in local space.
    pub position: [f32; 3],
    /// Normal vector.
    pub normal: [f32; 3],
    /// Texture coordinates.
    pub uv: [f32; 2],
}

impl Vertex {
    /// Create a new vertex.
    pub const fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self { position, normal, uv }
    }

    /// Get the vertex buffer layout for this vertex type.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }

    /// Vertex attributes.
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
        // position
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        // normal
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x3,
        },
        // uv
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x2,
        },
    ];
}

/// Simple position-only vertex.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PositionVertex {
    /// Position in local space.
    pub position: [f32; 3],
}

impl PositionVertex {
    /// Create a new position vertex.
    pub const fn new(position: [f32; 3]) -> Self {
        Self { position }
    }

    /// Get the vertex buffer layout.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Position and color vertex (useful for simple colored shapes).
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ColorVertex {
    /// Position in local space.
    pub position: [f32; 3],
    /// RGBA color.
    pub color: [f32; 4],
}

impl ColorVertex {
    /// Create a new color vertex.
    pub const fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }

    /// Get the vertex buffer layout.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }

    /// Vertex attributes.
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
