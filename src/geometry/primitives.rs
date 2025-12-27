//! Primitive geometry builders.

use super::vertex::ColorVertex;
use bytemuck;
use wgpu::util::DeviceExt;

/// Create a colored triangle geometry.
/// Returns vertex buffer and vertex count.
pub fn create_triangle(device: &wgpu::Device) -> (wgpu::Buffer, u32) {
    let vertices = [
        ColorVertex::new([0.0, 0.5, 0.0], [1.0, 0.0, 0.0, 1.0]),   // Top (red)
        ColorVertex::new([-0.5, -0.5, 0.0], [0.0, 1.0, 0.0, 1.0]), // Bottom-left (green)
        ColorVertex::new([0.5, -0.5, 0.0], [0.0, 0.0, 1.0, 1.0]),  // Bottom-right (blue)
    ];

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Triangle Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    (buffer, 3)
}

/// Create a colored quad geometry.
/// Returns vertex buffer, index buffer, and index count.
pub fn create_quad(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let vertices = [
        ColorVertex::new([-0.5, 0.5, 0.0], [1.0, 0.0, 0.0, 1.0]),  // Top-left
        ColorVertex::new([0.5, 0.5, 0.0], [0.0, 1.0, 0.0, 1.0]),   // Top-right
        ColorVertex::new([0.5, -0.5, 0.0], [0.0, 0.0, 1.0, 1.0]),  // Bottom-right
        ColorVertex::new([-0.5, -0.5, 0.0], [1.0, 1.0, 0.0, 1.0]), // Bottom-left
    ];

    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Quad Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Quad Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    (vertex_buffer, index_buffer, 6)
}
