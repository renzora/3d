//! Main renderer implementation.

use super::{Context, RenderConfig};
use crate::math::Color;

/// Render statistics for the current frame.
#[derive(Debug, Clone, Default)]
pub struct RenderInfo {
    /// Number of draw calls.
    pub draw_calls: u32,
    /// Number of triangles rendered.
    pub triangles: u32,
    /// Number of points rendered.
    pub points: u32,
    /// Number of lines rendered.
    pub lines: u32,
    /// Frame number.
    pub frame: u64,
}

impl RenderInfo {
    /// Reset the statistics.
    pub fn reset(&mut self) {
        self.draw_calls = 0;
        self.triangles = 0;
        self.points = 0;
        self.lines = 0;
    }
}

/// The main renderer.
pub struct Renderer {
    /// Render configuration.
    #[allow(dead_code)]
    config: RenderConfig,
    /// Depth texture.
    #[allow(dead_code)]
    depth_texture: Option<wgpu::Texture>,
    /// Depth texture view.
    depth_view: Option<wgpu::TextureView>,
    /// Render statistics.
    info: RenderInfo,
    /// Clear color.
    clear_color: Color,
    /// Auto clear color buffer.
    auto_clear: bool,
    /// Auto clear depth buffer.
    auto_clear_depth: bool,
}

impl Renderer {
    /// Create a new renderer.
    pub fn new(ctx: &Context, config: RenderConfig) -> Self {
        let depth_texture = ctx.create_depth_texture();
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            config,
            depth_texture: Some(depth_texture),
            depth_view: Some(depth_view),
            info: RenderInfo::default(),
            clear_color: Color::new(0.1, 0.1, 0.1),
            auto_clear: true,
            auto_clear_depth: true,
        }
    }

    /// Get render info.
    #[inline]
    pub fn info(&self) -> &RenderInfo {
        &self.info
    }

    /// Get mutable render info.
    #[inline]
    pub fn info_mut(&mut self) -> &mut RenderInfo {
        &mut self.info
    }

    /// Set the clear color.
    #[inline]
    pub fn set_clear_color(&mut self, color: Color) {
        self.clear_color = color;
    }

    /// Get the clear color.
    #[inline]
    pub fn clear_color(&self) -> Color {
        self.clear_color
    }

    /// Set auto clear.
    #[inline]
    pub fn set_auto_clear(&mut self, auto_clear: bool) {
        self.auto_clear = auto_clear;
    }

    /// Set auto clear depth.
    #[inline]
    pub fn set_auto_clear_depth(&mut self, auto_clear_depth: bool) {
        self.auto_clear_depth = auto_clear_depth;
    }

    /// Handle resize.
    pub fn resize(&mut self, ctx: &Context) {
        let depth_texture = ctx.create_depth_texture();
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_texture = Some(depth_texture);
        self.depth_view = Some(depth_view);
    }

    /// Get the depth texture view.
    #[inline]
    pub fn depth_view(&self) -> Option<&wgpu::TextureView> {
        self.depth_view.as_ref()
    }

    /// Get the wgpu clear color.
    fn wgpu_clear_color(&self) -> wgpu::Color {
        wgpu::Color {
            r: self.clear_color.r as f64,
            g: self.clear_color.g as f64,
            b: self.clear_color.b as f64,
            a: 1.0,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.info.reset();
        self.info.frame += 1;
    }

    /// Render a frame.
    /// This is a basic implementation that just clears the screen.
    pub fn render(&mut self, ctx: &Context) -> Result<(), wgpu::SurfaceError> {
        self.begin_frame();

        let output = ctx.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = ctx.create_command_encoder();

        // Build color attachment
        let color_attachment = wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: if self.auto_clear {
                    wgpu::LoadOp::Clear(self.wgpu_clear_color())
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
        };

        // Build depth attachment
        let depth_attachment = self.depth_view.as_ref().map(|depth_view| {
            wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: if self.auto_clear_depth {
                        wgpu::LoadOp::Clear(1.0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }
        });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // Render pass scope - drawing commands go here
        }

        ctx.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
