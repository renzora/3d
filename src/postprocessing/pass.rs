//! Base render pass trait for post-processing.

use wgpu::{CommandEncoder, TextureView};

/// A render pass in the post-processing pipeline.
pub trait Pass {
    /// Get the name of this pass.
    fn name(&self) -> &str;

    /// Check if this pass is enabled.
    fn enabled(&self) -> bool {
        true
    }

    /// Set whether this pass is enabled.
    fn set_enabled(&mut self, enabled: bool);

    /// Render this pass.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder to record commands
    /// * `input` - Input texture from previous pass
    /// * `output` - Output texture to render to
    /// * `device` - wgpu device for resource creation
    /// * `queue` - wgpu queue for buffer updates
    fn render(
        &self,
        encoder: &mut CommandEncoder,
        input: &TextureView,
        output: &TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );

    /// Called when the render target size changes.
    fn resize(&mut self, width: u32, height: u32, device: &wgpu::Device);
}

/// Uniform data for full-screen quad rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FullscreenUniform {
    /// Resolution (width, height, 1/width, 1/height).
    pub resolution: [f32; 4],
    /// Time in seconds.
    pub time: f32,
    /// Frame number.
    pub frame: u32,
    /// Padding.
    pub _padding: [f32; 2],
}

impl Default for FullscreenUniform {
    fn default() -> Self {
        Self {
            resolution: [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0],
            time: 0.0,
            frame: 0,
            _padding: [0.0; 2],
        }
    }
}

/// Vertex for fullscreen quad (position + uv).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FullscreenVertex {
    /// Position (x, y).
    pub position: [f32; 2],
    /// UV coordinates.
    pub uv: [f32; 2],
}

impl FullscreenVertex {
    /// Vertex buffer layout.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
            ],
        }
    }
}

/// Fullscreen quad vertices (two triangles).
pub const FULLSCREEN_QUAD_VERTICES: [FullscreenVertex; 6] = [
    // First triangle
    FullscreenVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    FullscreenVertex { position: [1.0, -1.0], uv: [1.0, 1.0] },
    FullscreenVertex { position: [1.0, 1.0], uv: [1.0, 0.0] },
    // Second triangle
    FullscreenVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    FullscreenVertex { position: [1.0, 1.0], uv: [1.0, 0.0] },
    FullscreenVertex { position: [-1.0, 1.0], uv: [0.0, 0.0] },
];
