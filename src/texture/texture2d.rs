//! 2D texture implementation.

use crate::core::Id;
use wgpu::util::DeviceExt;

/// A 2D texture.
pub struct Texture2D {
    /// Unique ID.
    id: Id,
    /// Texture width.
    width: u32,
    /// Texture height.
    height: u32,
    /// The GPU texture.
    texture: wgpu::Texture,
    /// Texture view.
    view: wgpu::TextureView,
    /// Texture format.
    format: wgpu::TextureFormat,
}

impl Texture2D {
    /// Create a new texture from RGBA8 data.
    /// Uses wgpu's create_texture_with_data which handles row alignment automatically.
    pub fn from_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        width: u32,
        height: u32,
        label: Option<&str>,
    ) -> Self {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Use create_texture_with_data which handles alignment automatically
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            id: Id::new(),
            width,
            height,
            texture,
            view,
            format,
        }
    }

    /// Create a solid color texture (1x1).
    pub fn from_color(device: &wgpu::Device, queue: &wgpu::Queue, r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::from_rgba8(device, queue, &[r, g, b, a], 1, 1, Some("Solid Color Texture"))
    }

    /// Create a white texture (1x1).
    pub fn white(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, 255, 255, 255, 255)
    }

    /// Create a black texture (1x1).
    pub fn black(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, 0, 0, 0, 255)
    }

    /// Create a default normal map texture (1x1, flat normal pointing up).
    pub fn default_normal(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Normal pointing straight out: (0, 0, 1) encoded as (128, 128, 255)
        Self::from_rgba8(
            device,
            queue,
            &[128, 128, 255, 255],
            1,
            1,
            Some("Default Normal Map"),
        )
    }

    /// Create a texture from encoded image bytes (PNG, JPEG, etc.).
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        label: Option<&str>,
    ) -> Result<Self, String> {
        use image::GenericImageView;

        let img = image::load_from_memory(data)
            .map_err(|e| format!("Failed to decode image: {}", e))?;

        let rgba = img.to_rgba8();
        let (width, height) = img.dimensions();

        // Get raw bytes explicitly
        let raw_data = rgba.as_raw();

        Ok(Self::from_rgba8(
            device,
            queue,
            raw_data,
            width,
            height,
            label,
        ))
    }

    /// Create a checkerboard pattern texture.
    pub fn checkerboard(device: &wgpu::Device, queue: &wgpu::Queue, size: u32, tile_size: u32) -> Self {
        let mut data = Vec::with_capacity((size * size * 4) as usize);

        for y in 0..size {
            for x in 0..size {
                let is_white = ((x / tile_size) + (y / tile_size)) % 2 == 0;
                let color = if is_white { 255 } else { 64 };
                data.extend_from_slice(&[color, color, color, 255]);
            }
        }

        Self::from_rgba8(device, queue, &data, size, size, Some("Checkerboard Texture"))
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get texture width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get texture height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the texture format.
    #[inline]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Get the underlying wgpu texture.
    #[inline]
    pub fn wgpu_texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Get the texture view.
    #[inline]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}
