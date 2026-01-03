//! Cluster and cluster group data structures for GPU.

use bytemuck::{Pod, Zeroable};

/// GPU representation of a mesh cluster (64 bytes).
///
/// A cluster is a group of triangles (typically 64-128) that are
/// culled and rendered as a unit. This structure is read by compute
/// shaders for culling and by the visibility buffer vertex shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ClusterGpu {
    /// Bounding sphere: xyz = center (local space), w = radius.
    pub bounding_sphere: [f32; 4],
    /// AABB minimum point (local space), w = unused.
    pub aabb_min: [f32; 4],
    /// AABB maximum point (local space), w = unused.
    pub aabb_max: [f32; 4],
    /// Parent error metric (for LOD selection). If screen-space error
    /// of parent is below threshold, render children instead.
    pub parent_error: f32,
    /// This cluster's error metric. Used to determine if this cluster
    /// should be rendered or if we should descend to finer LOD.
    pub cluster_error: f32,
    /// LOD level (0 = finest, higher = coarser).
    pub lod_level: u32,
    /// Index of the ClusterGroup this cluster belongs to.
    pub group_index: u32,
    /// Offset into the global index buffer.
    pub index_offset: u32,
    /// Number of triangles in this cluster.
    pub triangle_count: u32,
    /// Offset into the global vertex buffer (for position).
    pub vertex_offset: u32,
    /// Material ID for this cluster.
    pub material_id: u32,
}

impl ClusterGpu {
    /// Size of cluster in bytes (3 vec4s = 48 + 8 u32s = 32 = 80 bytes).
    pub const SIZE: usize = 80;
}

impl Default for ClusterGpu {
    fn default() -> Self {
        Self {
            bounding_sphere: [0.0, 0.0, 0.0, 1.0],
            aabb_min: [-1.0, -1.0, -1.0, 0.0],
            aabb_max: [1.0, 1.0, 1.0, 0.0],
            parent_error: f32::MAX,
            cluster_error: 0.0,
            lod_level: 0,
            group_index: 0,
            index_offset: 0,
            triangle_count: 0,
            vertex_offset: 0,
            material_id: 0,
        }
    }
}

/// GPU representation of a cluster group (32 bytes).
///
/// A cluster group contains multiple clusters at the same LOD level
/// that are swapped atomically during LOD transitions. This ensures
/// crack-free rendering at LOD boundaries.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ClusterGroupGpu {
    /// Parent group index (u32::MAX if this is a root group).
    pub parent_group: u32,
    /// First child group index.
    pub first_child: u32,
    /// Number of child groups.
    pub child_count: u32,
    /// First cluster index in this group.
    pub first_cluster: u32,
    /// Number of clusters in this group.
    pub cluster_count: u32,
    /// Maximum parent error of clusters in this group.
    pub max_parent_error: f32,
    /// LOD level of this group.
    pub lod_level: u32,
    /// Padding for 32-byte alignment.
    pub _pad: u32,
}

impl ClusterGroupGpu {
    /// Size of cluster group in bytes.
    pub const SIZE: usize = 32;

    /// Sentinel value for no parent.
    pub const NO_PARENT: u32 = u32::MAX;
}

impl Default for ClusterGroupGpu {
    fn default() -> Self {
        Self {
            parent_group: Self::NO_PARENT,
            first_child: 0,
            child_count: 0,
            first_cluster: 0,
            cluster_count: 0,
            max_parent_error: f32::MAX,
            lod_level: 0,
            _pad: 0,
        }
    }
}

/// Vertex data for Nanite rendering - positions only (16 bytes).
///
/// Positions are stored separately from attributes for better cache
/// efficiency during culling (which only needs positions).
///
/// NOTE: Padded to 16 bytes to match WGSL `array<vec3<f32>>` alignment
/// requirements in storage buffers (vec3 elements have 16-byte stride).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NaniteVertex {
    /// Position in local space.
    pub position: [f32; 3],
    /// Padding to match WGSL vec3 alignment (16 bytes per element).
    pub _pad: f32,
}

impl NaniteVertex {
    /// Create a new vertex.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
            _pad: 0.0,
        }
    }
}

/// Vertex attributes for Nanite rendering (32 bytes).
///
/// Stored separately from positions. Only fetched during the material
/// pass when we need to shade visible pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NaniteVertexAttribute {
    /// Normal vector (xyz), w = unused.
    pub normal: [f32; 4],
    /// Tangent vector (xyz), w = handedness sign.
    pub tangent: [f32; 4],
    /// Texture coordinates (uv), zw = unused.
    pub uv: [f32; 4],
    /// Vertex color (rgba).
    pub color: [f32; 4],
}

impl Default for NaniteVertexAttribute {
    fn default() -> Self {
        Self {
            normal: [0.0, 1.0, 0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv: [0.0, 0.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Instance data for Nanite meshes (80 bytes).
///
/// Each instance has a transform matrix and can reference a different
/// set of clusters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NaniteInstanceGpu {
    /// World transform matrix (row-major for GPU).
    pub transform: [[f32; 4]; 4],
    /// First cluster index for this instance's mesh.
    pub first_cluster: u32,
    /// Number of clusters in this instance's mesh.
    pub cluster_count: u32,
    /// First group index for this instance's mesh.
    pub first_group: u32,
    /// Number of groups in this instance's mesh.
    pub group_count: u32,
}

impl Default for NaniteInstanceGpu {
    fn default() -> Self {
        Self {
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            first_cluster: 0,
            cluster_count: 0,
            first_group: 0,
            group_count: 0,
        }
    }
}

/// Culling uniforms passed to compute shaders (64 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NaniteCullingUniform {
    /// View-projection matrix for frustum culling.
    pub view_proj: [[f32; 4]; 4],
}

impl Default for NaniteCullingUniform {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

/// Visibility buffer ID packing utilities.
///
/// The visibility buffer stores a 32-bit ID per pixel:
/// - Bits 0-11: Triangle ID within cluster (up to 4096 triangles)
/// - Bits 12-27: Cluster ID (up to 65536 clusters)
/// - Bits 28-31: Instance ID (up to 16 instances per draw, or use separate texture)
pub struct VisibilityId;

impl VisibilityId {
    /// Pack triangle, cluster, and instance IDs into a 32-bit value.
    #[inline]
    pub const fn pack(triangle_id: u32, cluster_id: u32, instance_id: u32) -> u32 {
        (triangle_id & 0xFFF) | ((cluster_id & 0xFFFF) << 12) | ((instance_id & 0xF) << 28)
    }

    /// Unpack triangle ID from visibility buffer value.
    #[inline]
    pub const fn triangle_id(packed: u32) -> u32 {
        packed & 0xFFF
    }

    /// Unpack cluster ID from visibility buffer value.
    #[inline]
    pub const fn cluster_id(packed: u32) -> u32 {
        (packed >> 12) & 0xFFFF
    }

    /// Unpack instance ID from visibility buffer value.
    #[inline]
    pub const fn instance_id(packed: u32) -> u32 {
        (packed >> 28) & 0xF
    }

    /// Invalid/empty visibility ID (background pixels).
    pub const INVALID: u32 = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_size() {
        assert_eq!(std::mem::size_of::<ClusterGpu>(), ClusterGpu::SIZE);
    }

    #[test]
    fn test_cluster_group_size() {
        assert_eq!(
            std::mem::size_of::<ClusterGroupGpu>(),
            ClusterGroupGpu::SIZE
        );
    }

    #[test]
    fn test_visibility_id_packing() {
        let triangle = 123;
        let cluster = 456;
        let instance = 7;

        let packed = VisibilityId::pack(triangle, cluster, instance);
        assert_eq!(VisibilityId::triangle_id(packed), triangle);
        assert_eq!(VisibilityId::cluster_id(packed), cluster);
        assert_eq!(VisibilityId::instance_id(packed), instance);
    }
}
