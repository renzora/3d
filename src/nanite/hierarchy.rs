//! Hierarchical LOD structure for Nanite.
//!
//! The LOD hierarchy is a Directed Acyclic Graph (DAG) where:
//! - Leaf nodes are the finest LOD level (original geometry clusters)
//! - Parent nodes contain simplified versions of their children
//! - ClusterGroups ensure crack-free transitions between LOD levels

use super::cluster::{ClusterGpu, ClusterGroupGpu};

/// A node in the LOD hierarchy.
#[derive(Debug, Clone)]
pub struct HierarchyNode {
    /// Cluster indices at this node.
    pub clusters: Vec<u32>,
    /// Child node indices (empty for leaf nodes).
    pub children: Vec<u32>,
    /// Parent node index (None for root).
    pub parent: Option<u32>,
    /// LOD level (0 = finest, higher = coarser).
    pub lod_level: u32,
    /// Maximum screen-space error for this node.
    /// If the projected error is below threshold, render this node.
    /// Otherwise, descend to children.
    pub max_error: f32,
    /// Bounding sphere center.
    pub bounds_center: [f32; 3],
    /// Bounding sphere radius.
    pub bounds_radius: f32,
}

impl Default for HierarchyNode {
    fn default() -> Self {
        Self {
            clusters: Vec::new(),
            children: Vec::new(),
            parent: None,
            lod_level: 0,
            max_error: 0.0,
            bounds_center: [0.0; 3],
            bounds_radius: 1.0,
        }
    }
}

/// The complete LOD hierarchy for a Nanite mesh.
#[derive(Debug, Clone)]
pub struct NaniteHierarchy {
    /// All hierarchy nodes (DAG structure).
    pub nodes: Vec<HierarchyNode>,
    /// Root node indices (entry points for traversal).
    pub roots: Vec<u32>,
    /// Maximum LOD level in the hierarchy.
    pub max_lod_level: u32,
    /// Total number of clusters across all LOD levels.
    pub total_clusters: u32,
}

impl NaniteHierarchy {
    /// Create a new empty hierarchy.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
            max_lod_level: 0,
            total_clusters: 0,
        }
    }

    /// Build a simple hierarchy from leaf clusters.
    ///
    /// This creates a binary tree where each parent node contains
    /// simplified geometry from merging two child nodes.
    pub fn build_from_clusters(
        clusters: &[ClusterGpu],
        simplification_ratio: f32,
        max_lod_levels: u32,
    ) -> Self {
        if clusters.is_empty() {
            return Self::new();
        }

        let mut hierarchy = Self::new();

        // Create leaf nodes (LOD 0) - one node per cluster
        let mut current_level_nodes: Vec<u32> = Vec::new();
        for (i, cluster) in clusters.iter().enumerate() {
            let node = HierarchyNode {
                clusters: vec![i as u32],
                children: Vec::new(),
                parent: None,
                lod_level: 0,
                max_error: cluster.cluster_error,
                bounds_center: [
                    cluster.bounding_sphere[0],
                    cluster.bounding_sphere[1],
                    cluster.bounding_sphere[2],
                ],
                bounds_radius: cluster.bounding_sphere[3],
            };
            let node_idx = hierarchy.nodes.len() as u32;
            hierarchy.nodes.push(node);
            current_level_nodes.push(node_idx);
        }

        hierarchy.total_clusters = clusters.len() as u32;

        // Build parent levels by grouping nodes
        let mut lod_level = 1u32;
        while current_level_nodes.len() > 1 && lod_level <= max_lod_levels {
            let mut next_level_nodes: Vec<u32> = Vec::new();

            // Group nodes in pairs (or groups of 4 for better balance)
            let group_size = 2usize;
            let mut i = 0;
            while i < current_level_nodes.len() {
                let end = (i + group_size).min(current_level_nodes.len());
                let children: Vec<u32> = current_level_nodes[i..end].to_vec();

                if children.is_empty() {
                    break;
                }

                // Compute merged bounds
                let (center, radius) = Self::compute_merged_bounds(&hierarchy.nodes, &children);

                // Compute error for this level
                // Error increases with LOD level (coarser = more error)
                let child_max_error = children
                    .iter()
                    .map(|&idx| hierarchy.nodes[idx as usize].max_error)
                    .fold(0.0f32, f32::max);
                let level_error = child_max_error + radius * simplification_ratio;

                // Create parent node
                let parent_idx = hierarchy.nodes.len() as u32;
                let mut parent_node = HierarchyNode {
                    clusters: Vec::new(), // Will be filled with simplified clusters
                    children: children.clone(),
                    parent: None,
                    lod_level,
                    max_error: level_error,
                    bounds_center: center,
                    bounds_radius: radius,
                };

                // Collect all clusters from children for this parent
                // In a full implementation, these would be simplified versions
                for &child_idx in &children {
                    let child = &hierarchy.nodes[child_idx as usize];
                    parent_node.clusters.extend(&child.clusters);
                }

                // Update children to point to parent
                for &child_idx in &children {
                    hierarchy.nodes[child_idx as usize].parent = Some(parent_idx);
                }

                hierarchy.nodes.push(parent_node);
                next_level_nodes.push(parent_idx);

                i += group_size;
            }

            // Handle odd node left over
            if i < current_level_nodes.len() {
                // Promote single node to next level
                let single_idx = current_level_nodes[i];
                next_level_nodes.push(single_idx);
            }

            current_level_nodes = next_level_nodes;
            lod_level += 1;
        }

        // Mark remaining nodes as roots
        hierarchy.roots = current_level_nodes;
        hierarchy.max_lod_level = lod_level.saturating_sub(1);

        hierarchy
    }

    /// Compute merged bounding sphere for a set of nodes.
    fn compute_merged_bounds(nodes: &[HierarchyNode], indices: &[u32]) -> ([f32; 3], f32) {
        if indices.is_empty() {
            return ([0.0; 3], 1.0);
        }

        // Compute center as average of child centers
        let mut center = [0.0f32; 3];
        for &idx in indices {
            let node = &nodes[idx as usize];
            center[0] += node.bounds_center[0];
            center[1] += node.bounds_center[1];
            center[2] += node.bounds_center[2];
        }
        let n = indices.len() as f32;
        center[0] /= n;
        center[1] /= n;
        center[2] /= n;

        // Compute radius as max distance from center to any child sphere edge
        let mut radius = 0.0f32;
        for &idx in indices {
            let node = &nodes[idx as usize];
            let dx = node.bounds_center[0] - center[0];
            let dy = node.bounds_center[1] - center[1];
            let dz = node.bounds_center[2] - center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt() + node.bounds_radius;
            radius = radius.max(dist);
        }

        (center, radius)
    }

    /// Convert hierarchy to GPU cluster groups.
    ///
    /// Each hierarchy node becomes a ClusterGroup that can be
    /// atomically swapped during LOD transitions.
    pub fn to_cluster_groups(&self, base_cluster_offset: u32) -> Vec<ClusterGroupGpu> {
        let mut groups = Vec::with_capacity(self.nodes.len());

        for (_i, node) in self.nodes.iter().enumerate() {
            let first_child = if node.children.is_empty() {
                0
            } else {
                node.children[0]
            };

            let group = ClusterGroupGpu {
                parent_group: node.parent.unwrap_or(ClusterGroupGpu::NO_PARENT),
                first_child,
                child_count: node.children.len() as u32,
                first_cluster: if node.clusters.is_empty() {
                    0
                } else {
                    base_cluster_offset + node.clusters[0]
                },
                cluster_count: node.clusters.len() as u32,
                max_parent_error: node.max_error,
                lod_level: node.lod_level,
                _pad: 0,
            };

            groups.push(group);
        }

        groups
    }

    /// Get clusters for a specific LOD level.
    pub fn get_clusters_at_lod(&self, lod_level: u32) -> Vec<u32> {
        self.nodes
            .iter()
            .filter(|n| n.lod_level == lod_level)
            .flat_map(|n| n.clusters.iter().copied())
            .collect()
    }

    /// Calculate screen-space error for a node given camera distance.
    ///
    /// Returns the projected error in pixels.
    pub fn calculate_screen_error(
        node: &HierarchyNode,
        camera_distance: f32,
        screen_height: f32,
        fov_y: f32,
    ) -> f32 {
        if camera_distance <= 0.0 {
            return f32::MAX;
        }

        // Project the error sphere to screen space
        // error_pixels = (error_world / distance) * (screen_height / (2 * tan(fov/2)))
        let projection_scale = screen_height / (2.0 * (fov_y * 0.5).tan());
        (node.max_error / camera_distance) * projection_scale
    }
}

impl Default for NaniteHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

/// LOD selection result for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodDecision {
    /// Render this node's clusters.
    Render,
    /// Descend to children for finer detail.
    Descend,
    /// Skip this node (culled or too far).
    Skip,
}

/// GPU-friendly LOD selection parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LodSelectionUniform {
    /// Screen height in pixels.
    pub screen_height: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Error threshold in pixels. If projected error < threshold, render.
    pub error_threshold: f32,
    /// Force LOD level (-1 for automatic).
    pub force_lod: i32,
}

impl Default for LodSelectionUniform {
    fn default() -> Self {
        Self {
            screen_height: 1080.0,
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            error_threshold: 1.0,               // 1 pixel error threshold
            force_lod: -1,                      // Automatic
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchy_build() {
        // Create 4 test clusters
        let clusters: Vec<ClusterGpu> = (0..4)
            .map(|i| {
                let mut c = ClusterGpu::default();
                c.bounding_sphere = [i as f32, 0.0, 0.0, 1.0];
                c.cluster_error = 0.01;
                c
            })
            .collect();

        let hierarchy = NaniteHierarchy::build_from_clusters(&clusters, 0.1, 3);

        // Should have leaf nodes + parent nodes
        assert!(hierarchy.nodes.len() >= 4);
        assert!(!hierarchy.roots.is_empty());
        assert!(hierarchy.max_lod_level >= 1);
    }

    #[test]
    fn test_screen_error() {
        let node = HierarchyNode {
            max_error: 0.1, // 10cm world-space error
            ..Default::default()
        };

        // At 10m distance, with 1080p and 45 degree FOV
        let error = NaniteHierarchy::calculate_screen_error(
            &node,
            10.0,
            1080.0,
            std::f32::consts::FRAC_PI_4,
        );

        // Should be a small number of pixels
        assert!(error > 0.0);
        assert!(error < 100.0);
    }
}
