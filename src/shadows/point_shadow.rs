//! Point light shadow maps using cube textures.

use crate::math::{Matrix4, Vector3};

/// Face directions for cube shadow map rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CubeFace {
    /// Positive X (+X).
    PositiveX = 0,
    /// Negative X (-X).
    NegativeX = 1,
    /// Positive Y (+Y).
    PositiveY = 2,
    /// Negative Y (-Y).
    NegativeY = 3,
    /// Positive Z (+Z).
    PositiveZ = 4,
    /// Negative Z (-Z).
    NegativeZ = 5,
}

impl CubeFace {
    /// Get all cube faces in order.
    pub const ALL: [CubeFace; 6] = [
        CubeFace::PositiveX,
        CubeFace::NegativeX,
        CubeFace::PositiveY,
        CubeFace::NegativeY,
        CubeFace::PositiveZ,
        CubeFace::NegativeZ,
    ];

    /// Get the direction vector for this face.
    pub fn direction(&self) -> Vector3 {
        match self {
            CubeFace::PositiveX => Vector3::new(1.0, 0.0, 0.0),
            CubeFace::NegativeX => Vector3::new(-1.0, 0.0, 0.0),
            CubeFace::PositiveY => Vector3::new(0.0, 1.0, 0.0),
            CubeFace::NegativeY => Vector3::new(0.0, -1.0, 0.0),
            CubeFace::PositiveZ => Vector3::new(0.0, 0.0, 1.0),
            CubeFace::NegativeZ => Vector3::new(0.0, 0.0, -1.0),
        }
    }

    /// Get the up vector for this face.
    pub fn up(&self) -> Vector3 {
        match self {
            CubeFace::PositiveX => Vector3::new(0.0, -1.0, 0.0),
            CubeFace::NegativeX => Vector3::new(0.0, -1.0, 0.0),
            CubeFace::PositiveY => Vector3::new(0.0, 0.0, 1.0),
            CubeFace::NegativeY => Vector3::new(0.0, 0.0, -1.0),
            CubeFace::PositiveZ => Vector3::new(0.0, -1.0, 0.0),
            CubeFace::NegativeZ => Vector3::new(0.0, -1.0, 0.0),
        }
    }
}

/// Point light shadow map using a cube texture.
pub struct PointShadowMap {
    /// Cube depth texture.
    texture: wgpu::Texture,
    /// Views for each face (for rendering).
    face_views: [wgpu::TextureView; 6],
    /// Full cube view (for sampling).
    cube_view: wgpu::TextureView,
    /// Resolution of each face.
    resolution: u32,
    /// Light position.
    position: Vector3,
    /// Light range.
    range: f32,
    /// View-projection matrices for each face.
    matrices: [[[f32; 4]; 4]; 6],
}

impl PointShadowMap {
    /// Create a new point shadow map.
    pub fn new(device: &wgpu::Device, resolution: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Point Shadow Cube Map"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 6, // 6 faces
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create individual face views for rendering
        let face_views: [wgpu::TextureView; 6] = std::array::from_fn(|i| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Point Shadow Face {}", i)),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
            })
        });

        // Create cube view for sampling
        let cube_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Point Shadow Cube View"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        Self {
            texture,
            face_views,
            cube_view,
            resolution,
            position: Vector3::ZERO,
            range: 10.0,
            matrices: [[[0.0; 4]; 4]; 6],
        }
    }

    /// Update the shadow map for a point light.
    pub fn update(&mut self, position: Vector3, range: f32) {
        self.position = position;
        self.range = range;
        self.calculate_matrices();
    }

    /// Calculate view-projection matrices for all 6 faces.
    fn calculate_matrices(&mut self) {
        // 90 degree FOV for cube faces
        let projection = Matrix4::perspective(
            std::f32::consts::FRAC_PI_2, // 90 degrees
            1.0,                          // Square faces
            0.1,
            self.range,
        );

        for (i, face) in CubeFace::ALL.iter().enumerate() {
            let target = self.position + face.direction();
            let view = Matrix4::look_at(&self.position, &target, &face.up());
            let view_proj = projection.multiply(&view);
            self.matrices[i] = view_proj.to_cols_array_2d();
        }
    }

    /// Get the view for a specific face (for rendering).
    #[inline]
    pub fn face_view(&self, face: CubeFace) -> &wgpu::TextureView {
        &self.face_views[face as usize]
    }

    /// Get the cube view (for sampling).
    #[inline]
    pub fn cube_view(&self) -> &wgpu::TextureView {
        &self.cube_view
    }

    /// Get the texture.
    #[inline]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Get the resolution.
    #[inline]
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Get the matrix for a specific face.
    #[inline]
    pub fn matrix(&self, face: CubeFace) -> &[[f32; 4]; 4] {
        &self.matrices[face as usize]
    }

    /// Get all face matrices.
    #[inline]
    pub fn matrices(&self) -> &[[[f32; 4]; 4]; 6] {
        &self.matrices
    }

    /// Get the light position.
    #[inline]
    pub fn position(&self) -> Vector3 {
        self.position
    }

    /// Get the light range.
    #[inline]
    pub fn range(&self) -> f32 {
        self.range
    }

    /// Resize the shadow map.
    pub fn resize(&mut self, device: &wgpu::Device, resolution: u32) {
        if self.resolution != resolution {
            *self = Self::new(device, resolution);
            self.calculate_matrices();
        }
    }
}
