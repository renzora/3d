//! Effect composer for managing post-processing pipeline.

use super::pass::Pass;
use wgpu::util::DeviceExt;

/// Manages a chain of post-processing passes.
pub struct EffectComposer {
    /// Render passes in order.
    passes: Vec<Box<dyn Pass>>,
    /// Ping-pong render targets.
    render_targets: [Option<RenderTarget>; 2],
    /// Current render target index.
    current_target: usize,
    /// Width of render targets.
    width: u32,
    /// Height of render targets.
    height: u32,
    /// Surface format.
    format: wgpu::TextureFormat,
    /// Fullscreen quad vertex buffer.
    quad_buffer: Option<wgpu::Buffer>,
}

/// A render target texture with view.
pub struct RenderTarget {
    /// The texture.
    pub texture: wgpu::Texture,
    /// Texture view.
    pub view: wgpu::TextureView,
}

impl RenderTarget {
    /// Create a new render target.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Post-Process Render Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self { texture, view }
    }
}

impl EffectComposer {
    /// Create a new effect composer.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fullscreen Quad Buffer"),
            contents: bytemuck::cast_slice(&super::pass::FULLSCREEN_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let mut composer = Self {
            passes: Vec::new(),
            render_targets: [None, None],
            current_target: 0,
            width,
            height,
            format,
            quad_buffer: Some(quad_buffer),
        };

        composer.create_render_targets(device);
        composer
    }

    /// Create or recreate render targets.
    fn create_render_targets(&mut self, device: &wgpu::Device) {
        self.render_targets[0] = Some(RenderTarget::new(device, self.width, self.height, self.format));
        self.render_targets[1] = Some(RenderTarget::new(device, self.width, self.height, self.format));
    }

    /// Add a pass to the chain.
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Remove a pass by name.
    pub fn remove_pass(&mut self, name: &str) -> Option<Box<dyn Pass>> {
        if let Some(idx) = self.passes.iter().position(|p| p.name() == name) {
            Some(self.passes.remove(idx))
        } else {
            None
        }
    }

    /// Get a pass by name.
    pub fn get_pass(&self, name: &str) -> Option<&dyn Pass> {
        self.passes.iter().find(|p| p.name() == name).map(|p| p.as_ref())
    }

    /// Get a mutable pass by name.
    pub fn get_pass_mut(&mut self, name: &str) -> Option<&mut Box<dyn Pass>> {
        self.passes.iter_mut().find(|p| p.name() == name)
    }

    /// Handle resize.
    pub fn resize(&mut self, width: u32, height: u32, device: &wgpu::Device) {
        if width == 0 || height == 0 {
            return;
        }

        self.width = width;
        self.height = height;
        self.create_render_targets(device);

        for pass in &mut self.passes {
            pass.resize(width, height, device);
        }
    }

    /// Render all passes.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder
    /// * `scene_texture` - The rendered scene texture
    /// * `output` - Final output texture view
    /// * `device` - wgpu device
    /// * `queue` - wgpu queue
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene_texture: &wgpu::TextureView,
        output: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let enabled_passes: Vec<usize> = self.passes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.enabled())
            .map(|(i, _)| i)
            .collect();

        if enabled_passes.is_empty() {
            // No passes enabled, just copy scene to output
            self.copy_texture(encoder, scene_texture, output, device);
            return;
        }

        // First pass reads from scene texture
        let mut input = scene_texture;
        self.current_target = 0;

        for (idx, &pass_idx) in enabled_passes.iter().enumerate() {
            let is_last = idx == enabled_passes.len() - 1;

            // Output is either a ping-pong target or the final output
            let output_view = if is_last {
                output
            } else {
                &self.render_targets[self.current_target].as_ref().unwrap().view
            };

            self.passes[pass_idx].render(encoder, input, output_view, device, queue);

            if !is_last {
                // Next pass reads from current target
                input = &self.render_targets[self.current_target].as_ref().unwrap().view;
                // Swap to other target for next pass
                self.current_target = 1 - self.current_target;
            }
        }
    }

    /// Copy one texture to another (simple blit).
    fn copy_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        _input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        _device: &wgpu::Device,
    ) {
        // Simple clear for now - a proper implementation would use a copy shader
        let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Copy Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }

    /// Get the fullscreen quad vertex buffer.
    pub fn quad_buffer(&self) -> Option<&wgpu::Buffer> {
        self.quad_buffer.as_ref()
    }

    /// Get render target dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
