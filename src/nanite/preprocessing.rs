//! Mesh preprocessing for Nanite clustering.
//!
//! This module converts regular meshes into clusters suitable for
//! GPU-driven rendering. Supports both simple clustering (Phase 1)
//! and LOD hierarchy with mesh simplification (Phase 3).

use super::cluster::{ClusterGpu, ClusterGroupGpu, NaniteVertex, NaniteVertexAttribute};
use super::hierarchy::NaniteHierarchy;
use super::simplification::simplify_mesh;
use crate::geometry::Vertex;

/// Result of building clusters from a mesh.
pub struct ClusterBuildResult {
    /// Clusters for this mesh (all LOD levels combined).
    pub clusters: Vec<ClusterGpu>,
    /// Cluster groups (for LOD hierarchy).
    pub groups: Vec<ClusterGroupGpu>,
    /// Vertex positions (all LOD levels combined).
    pub positions: Vec<NaniteVertex>,
    /// Vertex attributes (all LOD levels combined).
    pub attributes: Vec<NaniteVertexAttribute>,
    /// Triangle indices (all LOD levels combined).
    pub indices: Vec<u32>,
    /// LOD hierarchy (if built with LOD support).
    pub hierarchy: Option<NaniteHierarchy>,
    /// Maximum LOD level.
    pub max_lod_level: u32,
}

/// Configuration for LOD building.
#[derive(Debug, Clone)]
pub struct LodBuildConfig {
    /// Triangles per cluster.
    pub triangles_per_cluster: u32,
    /// Maximum number of LOD levels to generate.
    pub max_lod_levels: u32,
    /// Simplification ratio per LOD level (0.5 = half triangles each level).
    pub simplification_ratio: f32,
    /// Minimum triangles to stop LOD generation.
    pub min_triangles: u32,
    /// Error scale factor for LOD selection.
    pub error_scale: f32,
}

impl Default for LodBuildConfig {
    fn default() -> Self {
        Self {
            triangles_per_cluster: 128,
            max_lod_levels: 8,
            simplification_ratio: 0.5,
            min_triangles: 32,
            error_scale: 1.0,
        }
    }
}

/// Build clusters from a mesh.
///
/// For Phase 1, this simply splits the mesh into fixed-size clusters
/// of `triangles_per_cluster` triangles each.
pub fn build_clusters(
    vertices: &[Vertex],
    indices: &[u32],
    triangles_per_cluster: u32,
    material_id: u32,
) -> ClusterBuildResult {
    let tri_count = indices.len() / 3;
    let cluster_count = (tri_count as u32 + triangles_per_cluster - 1) / triangles_per_cluster;

    let mut clusters = Vec::with_capacity(cluster_count as usize);
    let mut groups = Vec::with_capacity(cluster_count as usize);
    let mut positions = Vec::with_capacity(vertices.len());
    let mut attributes = Vec::with_capacity(vertices.len());
    let mut out_indices = Vec::with_capacity(indices.len());

    // Convert vertices to Nanite format
    for v in vertices {
        positions.push(NaniteVertex {
            position: v.position,
            _pad: 0.0,
        });
        attributes.push(NaniteVertexAttribute {
            normal: [v.normal[0], v.normal[1], v.normal[2], 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0], // Default tangent
            uv: [v.uv[0], v.uv[1], 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0], // Default white
        });
    }

    // Copy indices
    out_indices.extend_from_slice(indices);

    // Create clusters
    let mut current_tri = 0;
    let mut cluster_idx = 0;

    while current_tri < tri_count {
        let tris_in_cluster = (tri_count - current_tri).min(triangles_per_cluster as usize);
        let index_offset = (current_tri * 3) as u32;

        // Compute bounding box and sphere for this cluster
        let (aabb_min, aabb_max, sphere) = compute_cluster_bounds(
            vertices,
            indices,
            current_tri,
            tris_in_cluster,
        );

        let cluster = ClusterGpu {
            bounding_sphere: sphere,
            aabb_min,
            aabb_max,
            parent_error: f32::MAX, // No parent in Phase 1
            cluster_error: 0.0,     // Finest level
            lod_level: 0,
            group_index: cluster_idx,
            index_offset,
            triangle_count: tris_in_cluster as u32,
            vertex_offset: 0, // All share same vertex buffer
            material_id,
        };

        // Each cluster gets its own group in Phase 1
        let group = ClusterGroupGpu {
            parent_group: ClusterGroupGpu::NO_PARENT,
            first_child: 0,
            child_count: 0,
            first_cluster: cluster_idx,
            cluster_count: 1,
            max_parent_error: f32::MAX,
            lod_level: 0,
            _pad: 0,
        };

        clusters.push(cluster);
        groups.push(group);

        current_tri += tris_in_cluster;
        cluster_idx += 1;
    }

    ClusterBuildResult {
        clusters,
        groups,
        positions,
        attributes,
        indices: out_indices,
        hierarchy: None,
        max_lod_level: 0,
    }
}

/// Build clusters with LOD hierarchy and mesh simplification.
///
/// This creates multiple LOD levels by progressively simplifying the mesh
/// and building a hierarchy of cluster groups for seamless transitions.
pub fn build_clusters_with_lod(
    vertices: &[Vertex],
    indices: &[u32],
    config: &LodBuildConfig,
    material_id: u32,
) -> ClusterBuildResult {
    let mut all_clusters: Vec<ClusterGpu> = Vec::new();
    let mut all_positions: Vec<NaniteVertex> = Vec::new();
    let mut all_attributes: Vec<NaniteVertexAttribute> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    // Track LOD level data for hierarchy building
    struct LodLevelData {
        first_cluster: u32,
        cluster_count: u32,
        error: f32,
    }
    let mut lod_levels: Vec<LodLevelData> = Vec::new();

    // Convert initial vertices to position array
    let mut current_positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
    let mut current_indices: Vec<u32> = indices.to_vec();

    // Store original vertex attributes (will be interpolated for simplified versions)
    let original_normals: Vec<[f32; 3]> = vertices.iter().map(|v| v.normal).collect();
    let original_uvs: Vec<[f32; 2]> = vertices.iter().map(|v| v.uv).collect();

    let mut lod_level = 0u32;
    let mut accumulated_error = 0.0f32;

    while lod_level <= config.max_lod_levels {
        let tri_count = current_indices.len() / 3;

        if tri_count < config.min_triangles as usize {
            break;
        }

        // Build clusters for this LOD level
        let first_cluster = all_clusters.len() as u32;
        let vertex_offset = all_positions.len() as u32;
        let index_offset = all_indices.len() as u32;

        // Add vertices for this LOD level
        for (i, pos) in current_positions.iter().enumerate() {
            all_positions.push(NaniteVertex { position: *pos, _pad: 0.0 });

            // For LOD 0, use original attributes
            // For higher LODs, use defaults (could interpolate, but expensive)
            if lod_level == 0 && i < original_normals.len() {
                all_attributes.push(NaniteVertexAttribute {
                    normal: [original_normals[i][0], original_normals[i][1], original_normals[i][2], 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    uv: [original_uvs[i][0], original_uvs[i][1], 0.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                });
            } else {
                // Simplified LOD - compute flat normals per cluster
                all_attributes.push(NaniteVertexAttribute {
                    normal: [0.0, 1.0, 0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    uv: [0.0, 0.0, 0.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                });
            }
        }

        // Add indices for this LOD level
        all_indices.extend(current_indices.iter());

        // Create clusters for this LOD level
        let cluster_count_before = all_clusters.len();
        let mut current_tri = 0usize;

        while current_tri < tri_count {
            let tris_in_cluster = (tri_count - current_tri).min(config.triangles_per_cluster as usize);
            let cluster_index_offset = index_offset + (current_tri * 3) as u32;

            // Compute bounding box and sphere for this cluster
            let (aabb_min, aabb_max, sphere) = compute_cluster_bounds_raw(
                &current_positions,
                &current_indices,
                current_tri,
                tris_in_cluster,
            );

            let cluster = ClusterGpu {
                bounding_sphere: sphere,
                aabb_min,
                aabb_max,
                parent_error: if lod_level == 0 { f32::MAX } else { accumulated_error * 2.0 },
                cluster_error: accumulated_error,
                lod_level,
                group_index: all_clusters.len() as u32,
                index_offset: cluster_index_offset,
                triangle_count: tris_in_cluster as u32,
                vertex_offset,
                material_id,
            };

            all_clusters.push(cluster);
            current_tri += tris_in_cluster;
        }

        let cluster_count = (all_clusters.len() - cluster_count_before) as u32;

        lod_levels.push(LodLevelData {
            first_cluster,
            cluster_count,
            error: accumulated_error,
        });

        // Simplify for next LOD level
        if lod_level < config.max_lod_levels {
            let target_ratio = config.simplification_ratio;
            let (simplified_positions, simplified_indices, simplify_error) =
                simplify_mesh(&current_positions, &current_indices, target_ratio);

            if simplified_indices.len() / 3 >= current_indices.len() / 3 {
                // No progress in simplification, stop
                break;
            }

            current_positions = simplified_positions;
            current_indices = simplified_indices;
            accumulated_error += simplify_error * config.error_scale;
        }

        lod_level += 1;
    }

    // Build hierarchy from LOD level clusters
    let hierarchy = NaniteHierarchy::build_from_clusters(
        &all_clusters,
        config.error_scale,
        config.max_lod_levels,
    );

    // Build cluster groups from hierarchy
    let groups = hierarchy.to_cluster_groups(0);

    ClusterBuildResult {
        clusters: all_clusters,
        groups,
        positions: all_positions,
        attributes: all_attributes,
        indices: all_indices,
        hierarchy: Some(hierarchy),
        max_lod_level: lod_level.saturating_sub(1),
    }
}

/// Compute bounding box and sphere for a cluster from raw positions.
fn compute_cluster_bounds_raw(
    positions: &[[f32; 3]],
    indices: &[u32],
    start_tri: usize,
    tri_count: usize,
) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];

    let start_idx = start_tri * 3;
    let end_idx = start_idx + tri_count * 3;

    for i in start_idx..end_idx {
        if i >= indices.len() {
            break;
        }
        let idx = indices[i] as usize;
        if idx >= positions.len() {
            continue;
        }
        let pos = positions[idx];

        min[0] = min[0].min(pos[0]);
        min[1] = min[1].min(pos[1]);
        min[2] = min[2].min(pos[2]);

        max[0] = max[0].max(pos[0]);
        max[1] = max[1].max(pos[1]);
        max[2] = max[2].max(pos[2]);
    }

    // Compute center and radius
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];

    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    let radius = (dx * dx + dy * dy + dz * dz).sqrt() * 0.5;

    (
        [min[0], min[1], min[2], 0.0],
        [max[0], max[1], max[2], 0.0],
        [center[0], center[1], center[2], radius],
    )
}

/// Compute bounding box and sphere for a cluster.
fn compute_cluster_bounds(
    vertices: &[Vertex],
    indices: &[u32],
    start_tri: usize,
    tri_count: usize,
) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];

    let start_idx = start_tri * 3;
    let end_idx = start_idx + tri_count * 3;

    for i in start_idx..end_idx {
        let idx = indices[i] as usize;
        let pos = vertices[idx].position;

        min[0] = min[0].min(pos[0]);
        min[1] = min[1].min(pos[1]);
        min[2] = min[2].min(pos[2]);

        max[0] = max[0].max(pos[0]);
        max[1] = max[1].max(pos[1]);
        max[2] = max[2].max(pos[2]);
    }

    // Compute center and radius
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];

    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    let radius = (dx * dx + dy * dy + dz * dz).sqrt() * 0.5;

    (
        [min[0], min[1], min[2], 0.0],
        [max[0], max[1], max[2], 0.0],
        [center[0], center[1], center[2], radius],
    )
}

/// Build clusters from raw position and index data (for GLTF loader).
pub fn build_clusters_from_raw(
    positions: &[[f32; 3]],
    normals: Option<&[[f32; 3]]>,
    uvs: Option<&[[f32; 2]]>,
    indices: &[u32],
    triangles_per_cluster: u32,
    material_id: u32,
) -> ClusterBuildResult {
    // Convert to Vertex format
    let vertices: Vec<Vertex> = positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            Vertex::new(
                *pos,
                normals.map_or([0.0, 1.0, 0.0], |n| n[i]),
                uvs.map_or([0.0, 0.0], |u| u[i]),
            )
        })
        .collect();

    build_clusters(&vertices, indices, triangles_per_cluster, material_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_building() {
        // Create a simple quad (2 triangles)
        let vertices = vec![
            Vertex::new([-1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
            Vertex::new([1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0]),
            Vertex::new([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0]),
            Vertex::new([-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0]),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];

        let result = build_clusters(&vertices, &indices, 128, 0);

        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].triangle_count, 2);
        assert_eq!(result.positions.len(), 4);
        assert_eq!(result.indices.len(), 6);
    }

    #[test]
    fn test_multiple_clusters() {
        // Create a mesh that will span multiple clusters
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Create 256 triangles (should create 2 clusters with 128 tri/cluster)
        for i in 0..256 {
            let base = (i * 3) as u32;
            let x = (i % 16) as f32;
            let z = (i / 16) as f32;

            vertices.push(Vertex::new([x, 0.0, z], [0.0, 1.0, 0.0], [0.0, 0.0]));
            vertices.push(Vertex::new([x + 1.0, 0.0, z], [0.0, 1.0, 0.0], [1.0, 0.0]));
            vertices.push(Vertex::new([x + 0.5, 0.0, z + 1.0], [0.0, 1.0, 0.0], [0.5, 1.0]));

            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }

        let result = build_clusters(&vertices, &indices, 128, 0);

        assert_eq!(result.clusters.len(), 2);
        assert_eq!(result.clusters[0].triangle_count, 128);
        assert_eq!(result.clusters[1].triangle_count, 128);
    }
}
