//! Base loader trait and common types.

use std::collections::HashMap;

/// Loading state for async operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadState {
    /// Not started.
    Idle,
    /// Currently loading.
    Loading,
    /// Successfully loaded.
    Loaded,
    /// Failed to load.
    Failed,
}

/// Progress information for loading operations.
#[derive(Debug, Clone)]
pub struct LoadProgress {
    /// Number of items loaded.
    pub loaded: usize,
    /// Total number of items.
    pub total: usize,
    /// Current item being loaded.
    pub current_item: Option<String>,
    /// Error message if failed.
    pub error: Option<String>,
}

impl LoadProgress {
    /// Create new progress tracker.
    pub fn new(total: usize) -> Self {
        Self {
            loaded: 0,
            total,
            current_item: None,
            error: None,
        }
    }

    /// Get progress as a fraction (0.0 - 1.0).
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            1.0
        } else {
            self.loaded as f32 / self.total as f32
        }
    }

    /// Get progress as a percentage (0 - 100).
    pub fn percentage(&self) -> u32 {
        (self.fraction() * 100.0) as u32
    }

    /// Check if loading is complete.
    pub fn is_complete(&self) -> bool {
        self.loaded >= self.total
    }

    /// Check if loading failed.
    pub fn is_failed(&self) -> bool {
        self.error.is_some()
    }
}

impl Default for LoadProgress {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Result type for loaded geometry data.
#[derive(Debug, Clone)]
pub struct LoadedGeometry {
    /// Vertex positions (vec3).
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals (vec3).
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates (vec2).
    pub uvs: Vec<[f32; 2]>,
    /// Vertex indices.
    pub indices: Vec<u32>,
    /// Vertex colors if present (vec4).
    pub colors: Option<Vec<[f32; 4]>>,
    /// Tangents if present (vec4).
    pub tangents: Option<Vec<[f32; 4]>>,
}

impl LoadedGeometry {
    /// Create empty geometry.
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            colors: None,
            tangents: None,
        }
    }

    /// Get vertex count.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Get triangle count.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Check if geometry has normals.
    pub fn has_normals(&self) -> bool {
        !self.normals.is_empty()
    }

    /// Check if geometry has UVs.
    pub fn has_uvs(&self) -> bool {
        !self.uvs.is_empty()
    }

    /// Generate flat normals from positions and indices.
    pub fn compute_flat_normals(&mut self) {
        self.normals.clear();
        self.normals.resize(self.positions.len(), [0.0, 0.0, 0.0]);

        for i in (0..self.indices.len()).step_by(3) {
            let i0 = self.indices[i] as usize;
            let i1 = self.indices[i + 1] as usize;
            let i2 = self.indices[i + 2] as usize;

            let p0 = self.positions[i0];
            let p1 = self.positions[i1];
            let p2 = self.positions[i2];

            // Edge vectors
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            // Cross product
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];

            // Accumulate (for smooth normals)
            for &idx in &[i0, i1, i2] {
                self.normals[idx][0] += n[0];
                self.normals[idx][1] += n[1];
                self.normals[idx][2] += n[2];
            }
        }

        // Normalize
        for normal in &mut self.normals {
            let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if len > 0.0 {
                normal[0] /= len;
                normal[1] /= len;
                normal[2] /= len;
            }
        }
    }
}

impl Default for LoadedGeometry {
    fn default() -> Self {
        Self::new()
    }
}

/// Material data loaded from a file.
#[derive(Debug, Clone)]
pub struct LoadedMaterial {
    /// Material name.
    pub name: String,
    /// Base color (RGBA).
    pub base_color: [f32; 4],
    /// Metallic factor (0-1).
    pub metallic: f32,
    /// Roughness factor (0-1).
    pub roughness: f32,
    /// Emissive color (RGB).
    pub emissive: [f32; 3],
    /// Base color texture path.
    pub base_color_texture: Option<String>,
    /// Normal map texture path.
    pub normal_texture: Option<String>,
    /// Metallic-roughness texture path.
    pub metallic_roughness_texture: Option<String>,
    /// Emissive texture path.
    pub emissive_texture: Option<String>,
    /// Occlusion texture path.
    pub occlusion_texture: Option<String>,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    /// Alpha cutoff for mask mode.
    pub alpha_cutoff: f32,
    /// Double-sided rendering.
    pub double_sided: bool,
}

/// Alpha blending mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaMode {
    /// Fully opaque.
    #[default]
    Opaque,
    /// Alpha mask with cutoff.
    Mask,
    /// Alpha blending.
    Blend,
}

impl LoadedMaterial {
    /// Create a default white material.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0, 0.0, 0.0],
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
            occlusion_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

impl Default for LoadedMaterial {
    fn default() -> Self {
        Self::new("Default")
    }
}

/// A mesh loaded from a file (geometry + material reference).
#[derive(Debug, Clone)]
pub struct LoadedMesh {
    /// Mesh name.
    pub name: String,
    /// Geometry data.
    pub geometry: LoadedGeometry,
    /// Material index (references LoadedScene.materials).
    pub material_index: Option<usize>,
}

impl LoadedMesh {
    /// Create a new loaded mesh.
    pub fn new(name: impl Into<String>, geometry: LoadedGeometry) -> Self {
        Self {
            name: name.into(),
            geometry,
            material_index: None,
        }
    }
}

/// A node in the scene hierarchy.
#[derive(Debug, Clone)]
pub struct LoadedNode {
    /// Node name.
    pub name: String,
    /// Local translation.
    pub translation: [f32; 3],
    /// Local rotation (quaternion xyzw).
    pub rotation: [f32; 4],
    /// Local scale.
    pub scale: [f32; 3],
    /// Mesh indices (references LoadedScene.meshes).
    pub mesh_indices: Vec<usize>,
    /// Child node indices.
    pub children: Vec<usize>,
}

impl LoadedNode {
    /// Create a new node.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            mesh_indices: Vec::new(),
            children: Vec::new(),
        }
    }
}

impl Default for LoadedNode {
    fn default() -> Self {
        Self::new("Node")
    }
}

/// Decoded texture image data.
#[derive(Debug, Clone)]
pub struct LoadedTexture {
    /// Texture width.
    pub width: u32,
    /// Texture height.
    pub height: u32,
    /// RGBA8 pixel data (4 bytes per pixel, tightly packed).
    pub data: Vec<u8>,
}

impl LoadedTexture {
    /// Create a new loaded texture.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self { width, height, data }
    }
}

/// A complete scene loaded from a file.
#[derive(Debug, Clone)]
pub struct LoadedScene {
    /// Scene name.
    pub name: String,
    /// All meshes in the scene.
    pub meshes: Vec<LoadedMesh>,
    /// All materials in the scene.
    pub materials: Vec<LoadedMaterial>,
    /// All nodes in the scene.
    pub nodes: Vec<LoadedNode>,
    /// Root node indices.
    pub root_nodes: Vec<usize>,
    /// Embedded textures (name -> decoded RGBA data).
    pub textures: HashMap<String, LoadedTexture>,
}

impl LoadedScene {
    /// Create a new empty scene.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            meshes: Vec::new(),
            materials: Vec::new(),
            nodes: Vec::new(),
            root_nodes: Vec::new(),
            textures: HashMap::new(),
        }
    }

    /// Get total vertex count across all meshes.
    pub fn total_vertices(&self) -> usize {
        self.meshes.iter().map(|m| m.geometry.vertex_count()).sum()
    }

    /// Get total triangle count across all meshes.
    pub fn total_triangles(&self) -> usize {
        self.meshes.iter().map(|m| m.geometry.triangle_count()).sum()
    }
}

impl Default for LoadedScene {
    fn default() -> Self {
        Self::new("Untitled")
    }
}

/// Error type for loading operations.
#[derive(Debug, Clone)]
pub struct LoadError {
    /// Error message.
    pub message: String,
    /// Source file if known.
    pub source: Option<String>,
}

impl LoadError {
    /// Create a new load error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    /// Create with source file.
    pub fn with_source(message: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: Some(source.into()),
        }
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref source) = self.source {
            write!(f, "{}: {}", source, self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for LoadError {}
