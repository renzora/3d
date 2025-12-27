//! Main engine entry point.

use super::{Clock, Context, ContextError, RenderConfig, Renderer};

/// The main Ren engine.
/// Manages the rendering context, renderer, and timing.
pub struct Engine {
    /// The wgpu context.
    pub context: Context,
    /// The renderer.
    pub renderer: Renderer,
    /// The clock for timing.
    pub clock: Clock,
    /// Current width.
    width: u32,
    /// Current height.
    height: u32,
}

impl Engine {
    /// Create a new engine from a window handle.
    ///
    /// # Arguments
    /// * `window` - A window handle (e.g., from winit or web_sys::HtmlCanvasElement)
    /// * `width` - Initial width in pixels
    /// * `height` - Initial height in pixels
    ///
    /// # Safety
    /// The window must outlive the engine.
    pub async fn new<W>(
        window: W,
        width: u32,
        height: u32,
    ) -> Result<Self, ContextError>
    where
        W: Into<wgpu::SurfaceTarget<'static>>,
    {
        Self::with_config(window, width, height, RenderConfig::default()).await
    }

    /// Create a new engine with custom configuration.
    pub async fn with_config<W>(
        window: W,
        width: u32,
        height: u32,
        config: RenderConfig,
    ) -> Result<Self, ContextError>
    where
        W: Into<wgpu::SurfaceTarget<'static>>,
    {
        let context = Context::new(window, width, height, &config).await?;
        let renderer = Renderer::new(&context, config);
        let clock = Clock::start_new();

        Ok(Self {
            context,
            renderer,
            clock,
            width,
            height,
        })
    }

    /// Handle resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 && (width != self.width || height != self.height) {
            self.width = width;
            self.height = height;
            self.context.resize(width, height);
            self.renderer.resize(&self.context);
        }
    }

    /// Get current width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get current height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get aspect ratio.
    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    /// Get delta time since last frame.
    pub fn delta_time(&mut self) -> f32 {
        self.clock.get_delta() as f32
    }

    /// Get elapsed time since engine start.
    pub fn elapsed_time(&mut self) -> f32 {
        self.clock.get_elapsed_time() as f32
    }

    /// Get the device.
    #[inline]
    pub fn device(&self) -> &wgpu::Device {
        &self.context.device
    }

    /// Get the queue.
    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.context.queue
    }

    /// Render a frame (basic clear screen).
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.renderer.render(&self.context)
    }
}

/// Builder for configuring the engine.
pub struct EngineBuilder {
    config: RenderConfig,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineBuilder {
    /// Create a new engine builder.
    pub fn new() -> Self {
        Self {
            config: RenderConfig::default(),
        }
    }

    /// Set anti-aliasing.
    pub fn antialias(mut self, enabled: bool) -> Self {
        self.config.antialias = enabled;
        self
    }

    /// Set alpha blending.
    pub fn alpha(mut self, enabled: bool) -> Self {
        self.config.alpha = enabled;
        self
    }

    /// Set depth testing.
    pub fn depth(mut self, enabled: bool) -> Self {
        self.config.depth = enabled;
        self
    }

    /// Set power preference.
    pub fn power_preference(mut self, preference: wgpu::PowerPreference) -> Self {
        self.config.power_preference = preference;
        self
    }

    /// Set present mode.
    pub fn present_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.config.present_mode = mode;
        self
    }

    /// Set clear color.
    pub fn clear_color(mut self, r: f64, g: f64, b: f64) -> Self {
        self.config.clear_color = wgpu::Color { r, g, b, a: 1.0 };
        self
    }

    /// Build the engine.
    pub async fn build<W>(
        self,
        window: W,
        width: u32,
        height: u32,
    ) -> Result<Engine, ContextError>
    where
        W: Into<wgpu::SurfaceTarget<'static>>,
    {
        Engine::with_config(window, width, height, self.config).await
    }
}
