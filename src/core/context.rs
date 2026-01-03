//! wgpu context management.

use super::RenderConfig;
use thiserror::Error;

/// Errors that can occur during context creation.
#[derive(Error, Debug)]
pub enum ContextError {
    /// Failed to create wgpu instance.
    #[error("Failed to create wgpu instance")]
    InstanceCreation,

    /// Failed to request adapter.
    #[error("Failed to request adapter: no suitable GPU found")]
    AdapterRequest,

    /// Failed to request device.
    #[error("Failed to request device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    /// Failed to create surface.
    #[error("Failed to create surface: {0}")]
    SurfaceCreation(#[from] wgpu::CreateSurfaceError),

    /// Surface is not supported by adapter.
    #[error("Surface not supported by adapter")]
    SurfaceNotSupported,
}

/// The wgpu rendering context.
/// Manages the device, queue, and surface configuration.
pub struct Context {
    /// The wgpu instance.
    pub instance: wgpu::Instance,
    /// The rendering surface.
    pub surface: wgpu::Surface<'static>,
    /// The GPU adapter.
    pub adapter: wgpu::Adapter,
    /// The GPU device.
    pub device: wgpu::Device,
    /// The command queue.
    pub queue: wgpu::Queue,
    /// Surface configuration.
    pub surface_config: wgpu::SurfaceConfiguration,
    /// Current surface texture format.
    pub surface_format: wgpu::TextureFormat,
    /// Depth texture format.
    pub depth_format: wgpu::TextureFormat,
    /// Current width.
    pub width: u32,
    /// Current height.
    pub height: u32,
}

impl Context {
    /// Create a new context from a window handle.
    ///
    /// # Safety
    /// The window must outlive the context.
    pub async fn new<W>(window: W, width: u32, height: u32, config: &RenderConfig) -> Result<Self, ContextError>
    where
        W: Into<wgpu::SurfaceTarget<'static>>,
    {
        // Create instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface
        let surface = instance.create_surface(window)?;

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(ContextError::AdapterRequest)?;

        // Get surface capabilities
        let surface_caps = surface.get_capabilities(&adapter);

        // Choose surface format
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        // Request device with higher storage buffer limits for Nanite
        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::downlevel_webgl2_defaults()
            .using_resolution(adapter_limits.clone());

        // Request higher storage buffer limit for Nanite (default is 128MB, need 256MB+)
        required_limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .max(256 * 1024 * 1024);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Ren Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        // Configure surface
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: config.present_mode,
            alpha_mode: if config.alpha {
                wgpu::CompositeAlphaMode::PreMultiplied
            } else {
                surface_caps.alpha_modes[0]
            },
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        let depth_format = wgpu::TextureFormat::Depth32Float;

        Ok(Self {
            instance,
            surface,
            adapter,
            device,
            queue,
            surface_config,
            surface_format,
            depth_format,
            width,
            height,
        })
    }

    /// Resize the surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.width = width;
            self.height = height;
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    /// Get the current aspect ratio.
    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    /// Get the current surface texture.
    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }

    /// Create a command encoder.
    pub fn create_command_encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Ren Command Encoder"),
        })
    }

    /// Submit commands to the queue.
    pub fn submit(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.queue.submit(commands);
    }

    /// Create a depth texture.
    pub fn create_depth_texture(&self) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    /// Create a buffer.
    pub fn create_buffer(&self, descriptor: &wgpu::BufferDescriptor) -> wgpu::Buffer {
        self.device.create_buffer(descriptor)
    }

    /// Create a buffer with data.
    pub fn create_buffer_init(&self, descriptor: &wgpu::util::BufferInitDescriptor) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(descriptor)
    }
}
