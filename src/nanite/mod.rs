//! Nanite-style virtualized geometry system.
//!
//! This module implements GPU-driven rendering with mesh clustering,
//! hierarchical LOD, frustum/occlusion culling, and visibility buffer rendering.
//!
//! ## Architecture
//!
//! 1. **Preprocessing**: Meshes are split into clusters of ~128 triangles each
//! 2. **GPU Culling**: Compute shaders cull clusters against frustum and HZB
//! 3. **Visibility Buffer**: Render cluster/triangle IDs to a visibility texture
//! 4. **Material Pass**: Fullscreen pass shades pixels using visibility buffer data

mod cluster;
mod culling;
mod gpu_resources;
mod hierarchy;
mod hzb;
mod optimizer;
mod preprocessing;
mod rasterizer;
mod renderer;
mod simplification;

pub use cluster::{ClusterGpu, ClusterGroupGpu, NaniteVertex, NaniteVertexAttribute};
pub use culling::{FrustumCuller, FrustumCullUniform};
pub use gpu_resources::{NaniteGpuResources, NaniteMaterialGpu};
pub use hierarchy::{LodDecision, LodSelectionUniform, NaniteHierarchy};
pub use hzb::{HzbGenerator, HzbBuildUniform, OcclusionCullUniform};
pub use optimizer::{NaniteOptimizer, NaniteStatistics, MaterialBatch};
pub use preprocessing::{build_clusters, build_clusters_with_lod, ClusterBuildResult, LodBuildConfig};
pub use rasterizer::{SoftwareRasterizer, SwRasterUniform, ClassifyUniform, ClassifyCounters};
pub use renderer::NaniteRenderer;
pub use simplification::simplify_mesh;

/// Configuration for the Nanite rendering system.
#[derive(Debug, Clone)]
pub struct NaniteConfig {
    /// Maximum number of clusters per mesh.
    pub max_clusters: u32,
    /// Maximum number of vertices (independent of cluster count for efficiency).
    pub max_vertices: u32,
    /// Maximum number of indices.
    pub max_indices: u32,
    /// Target triangles per cluster (64-128 recommended).
    pub triangles_per_cluster: u32,
    /// Maximum number of instances that can be rendered.
    pub max_instances: u32,
    /// Enable software rasterization for small triangles.
    pub enable_software_rasterization: bool,
    /// Pixel threshold below which to use software rasterization.
    pub software_rasterization_threshold: f32,
    /// Enable hierarchical Z-buffer occlusion culling.
    pub enable_occlusion_culling: bool,
    /// Enable LOD hierarchy (if false, always render finest level).
    pub enable_lod: bool,
    /// Screen-space error threshold for LOD selection (pixels).
    pub lod_error_threshold: f32,
}

impl Default for NaniteConfig {
    fn default() -> Self {
        Self {
            // Buffer sizes for large scenes (e.g., cyberpunk city with ~10K clusters)
            // 16384 clusters supports models up to ~2M triangles
            max_clusters: 16384,
            // Vertices are typically shared, so ~1.5 verts per triangle is realistic
            // 4M vertices * 16 bytes (position) = 64MB
            // 4M vertices * 64 bytes (attributes) = 256MB
            max_vertices: 4_000_000,
            // Indices: 3 per triangle, ~2M triangles = 6M indices * 4 bytes = 24MB
            max_indices: 8_000_000,
            triangles_per_cluster: 128,
            max_instances: 1024,
            enable_software_rasterization: true,  // Phase 5 - enabled
            software_rasterization_threshold: 2.0,
            enable_occlusion_culling: true,       // Phase 4 - enabled
            enable_lod: true,                     // Phase 3 - enabled
            lod_error_threshold: 1.0,
        }
    }
}

/// Statistics for Nanite rendering (per frame).
#[derive(Debug, Clone, Default)]
pub struct NaniteStats {
    /// Total clusters in scene.
    pub total_clusters: u32,
    /// Clusters passing frustum culling.
    pub clusters_after_frustum: u32,
    /// Clusters passing occlusion culling.
    pub clusters_after_occlusion: u32,
    /// Triangles rendered via hardware rasterization.
    pub hw_triangles: u32,
    /// Triangles rendered via software rasterization.
    pub sw_triangles: u32,
}
