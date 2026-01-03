//! Mesh simplification using Quadric Error Metrics (QEM).
//!
//! This implements the algorithm from "Surface Simplification Using Quadric Error Metrics"
//! by Garland and Heckbert (1997).
//!
//! The key idea is to assign a 4x4 "error quadric" matrix to each vertex that measures
//! how far the vertex is from its original supporting planes. When we collapse an edge,
//! we sum the quadrics of the two endpoints and find the optimal position that minimizes
//! the combined error.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// A 4x4 symmetric matrix representing the error quadric.
/// Stored as upper triangular (10 elements) for efficiency.
#[derive(Debug, Clone, Copy)]
pub struct Quadric {
    /// Elements: a, b, c, d, e, f, g, h, i, j
    /// Represents the symmetric matrix:
    /// | a b c d |
    /// | b e f g |
    /// | c f h i |
    /// | d g i j |
    pub data: [f64; 10],
}

impl Quadric {
    /// Create a zero quadric.
    pub fn zero() -> Self {
        Self { data: [0.0; 10] }
    }

    /// Create a quadric from a plane equation ax + by + cz + d = 0.
    /// The quadric is p * p^T where p = [a, b, c, d].
    pub fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            data: [
                a * a,     // a
                a * b,     // b
                a * c,     // c
                a * d,     // d
                b * b,     // e
                b * c,     // f
                b * d,     // g
                c * c,     // h
                c * d,     // i
                d * d,     // j
            ],
        }
    }

    /// Add two quadrics.
    pub fn add(&self, other: &Quadric) -> Self {
        let mut result = Self::zero();
        for i in 0..10 {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    /// Evaluate the quadric error for a point [x, y, z].
    /// Returns v^T * Q * v where v = [x, y, z, 1].
    pub fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        let [a, b, c, d, e, f, g, h, i, j] = self.data;

        a * x * x + 2.0 * b * x * y + 2.0 * c * x * z + 2.0 * d * x
            + e * y * y + 2.0 * f * y * z + 2.0 * g * y
            + h * z * z + 2.0 * i * z
            + j
    }

    /// Find the optimal position that minimizes the quadric error.
    /// Returns None if the matrix is singular.
    pub fn optimal_position(&self) -> Option<[f64; 3]> {
        let [a, b, c, d, e, f, g, h, i, _j] = self.data;

        // We need to solve:
        // | a b c |   | x |   | -d |
        // | b e f | * | y | = | -g |
        // | c f h |   | z |   | -i |

        // Use Cramer's rule for 3x3 system
        let det = a * (e * h - f * f) - b * (b * h - f * c) + c * (b * f - e * c);

        if det.abs() < 1e-10 {
            return None;
        }

        let inv_det = 1.0 / det;

        let x = inv_det * ((-d) * (e * h - f * f) - b * ((-g) * h - f * (-i)) + c * ((-g) * f - e * (-i)));
        let y = inv_det * (a * ((-g) * h - f * (-i)) - (-d) * (b * h - f * c) + c * (b * (-i) - (-g) * c));
        let z = inv_det * (a * (e * (-i) - (-g) * f) - b * (b * (-i) - (-g) * c) + (-d) * (b * f - e * c));

        Some([x, y, z])
    }
}

impl Default for Quadric {
    fn default() -> Self {
        Self::zero()
    }
}

/// Edge collapse candidate with error cost.
#[derive(Debug, Clone)]
struct EdgeCollapse {
    /// Vertex indices (smaller, larger).
    v0: u32,
    v1: u32,
    /// Optimal collapse position.
    new_position: [f64; 3],
    /// Error cost of this collapse.
    error: f64,
}

impl PartialEq for EdgeCollapse {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error && self.v0 == other.v0 && self.v1 == other.v1
    }
}

impl Eq for EdgeCollapse {}

impl PartialOrd for EdgeCollapse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeCollapse {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (smallest error first)
        other.error.partial_cmp(&self.error).unwrap_or(Ordering::Equal)
    }
}

/// Mesh simplifier using Quadric Error Metrics.
pub struct MeshSimplifier {
    /// Vertex positions.
    positions: Vec<[f64; 3]>,
    /// Vertex quadrics.
    quadrics: Vec<Quadric>,
    /// Triangle indices (groups of 3).
    triangles: Vec<[u32; 3]>,
    /// Map from vertex to triangles containing it.
    vertex_triangles: HashMap<u32, HashSet<usize>>,
    /// Set of removed vertices.
    removed_vertices: HashSet<u32>,
    /// Set of removed triangles.
    removed_triangles: HashSet<usize>,
    /// Edge collapse heap.
    heap: BinaryHeap<EdgeCollapse>,
}

impl MeshSimplifier {
    /// Create a new mesh simplifier.
    pub fn new(positions: &[[f32; 3]], indices: &[u32]) -> Self {
        let positions: Vec<[f64; 3]> = positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();

        let triangles: Vec<[u32; 3]> = indices
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect();

        let mut simplifier = Self {
            positions,
            quadrics: Vec::new(),
            triangles,
            vertex_triangles: HashMap::new(),
            removed_vertices: HashSet::new(),
            removed_triangles: HashSet::new(),
            heap: BinaryHeap::new(),
        };

        simplifier.initialize();
        simplifier
    }

    /// Initialize quadrics and edge heap.
    fn initialize(&mut self) {
        let vertex_count = self.positions.len();

        // Initialize vertex quadrics to zero
        self.quadrics = vec![Quadric::zero(); vertex_count];

        // Build vertex-triangle map and compute quadrics
        for (tri_idx, tri) in self.triangles.iter().enumerate() {
            let v0 = tri[0] as usize;
            let v1 = tri[1] as usize;
            let v2 = tri[2] as usize;

            // Add to vertex-triangle map
            self.vertex_triangles.entry(tri[0]).or_default().insert(tri_idx);
            self.vertex_triangles.entry(tri[1]).or_default().insert(tri_idx);
            self.vertex_triangles.entry(tri[2]).or_default().insert(tri_idx);

            // Compute plane equation for this triangle
            let p0 = self.positions[v0];
            let p1 = self.positions[v1];
            let p2 = self.positions[v2];

            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            // Cross product for normal
            let nx = e1[1] * e2[2] - e1[2] * e2[1];
            let ny = e1[2] * e2[0] - e1[0] * e2[2];
            let nz = e1[0] * e2[1] - e1[1] * e2[0];

            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len < 1e-10 {
                continue; // Degenerate triangle
            }

            let a = nx / len;
            let b = ny / len;
            let c = nz / len;
            let d = -(a * p0[0] + b * p0[1] + c * p0[2]);

            let plane_quadric = Quadric::from_plane(a, b, c, d);

            // Add to each vertex's quadric
            self.quadrics[v0] = self.quadrics[v0].add(&plane_quadric);
            self.quadrics[v1] = self.quadrics[v1].add(&plane_quadric);
            self.quadrics[v2] = self.quadrics[v2].add(&plane_quadric);
        }

        // Build initial edge collapse candidates
        let mut edges_seen = HashSet::new();
        for tri in &self.triangles {
            for i in 0..3 {
                let v0 = tri[i].min(tri[(i + 1) % 3]);
                let v1 = tri[i].max(tri[(i + 1) % 3]);
                let edge = (v0, v1);

                if edges_seen.insert(edge) {
                    if let Some(collapse) = self.compute_edge_collapse(v0, v1) {
                        self.heap.push(collapse);
                    }
                }
            }
        }
    }

    /// Compute the optimal edge collapse for an edge.
    fn compute_edge_collapse(&self, v0: u32, v1: u32) -> Option<EdgeCollapse> {
        if self.removed_vertices.contains(&v0) || self.removed_vertices.contains(&v1) {
            return None;
        }

        let q_sum = self.quadrics[v0 as usize].add(&self.quadrics[v1 as usize]);

        // Try to find optimal position
        let new_position = q_sum.optimal_position().unwrap_or_else(|| {
            // Fallback: use midpoint
            let p0 = self.positions[v0 as usize];
            let p1 = self.positions[v1 as usize];
            [
                (p0[0] + p1[0]) * 0.5,
                (p0[1] + p1[1]) * 0.5,
                (p0[2] + p1[2]) * 0.5,
            ]
        });

        let error = q_sum.evaluate(new_position[0], new_position[1], new_position[2]);

        Some(EdgeCollapse {
            v0,
            v1,
            new_position,
            error: error.max(0.0), // Clamp negative errors
        })
    }

    /// Simplify the mesh to the target triangle count.
    pub fn simplify(&mut self, target_triangles: usize) -> f64 {
        let mut max_error = 0.0;
        let initial_triangles = self.triangles.len();

        while self.active_triangle_count() > target_triangles {
            let collapse = loop {
                match self.heap.pop() {
                    Some(c) => {
                        // Check if this collapse is still valid
                        if !self.removed_vertices.contains(&c.v0)
                            && !self.removed_vertices.contains(&c.v1)
                        {
                            break c;
                        }
                    }
                    None => return max_error,
                }
            };

            max_error = max_error.max(collapse.error);
            self.perform_collapse(&collapse);

            // Progress check
            if self.active_triangle_count() <= target_triangles {
                break;
            }
        }

        log::debug!(
            "Simplified from {} to {} triangles, max error: {}",
            initial_triangles,
            self.active_triangle_count(),
            max_error
        );

        max_error
    }

    /// Perform an edge collapse.
    fn perform_collapse(&mut self, collapse: &EdgeCollapse) {
        let v0 = collapse.v0;
        let v1 = collapse.v1;

        // Move v0 to new position
        self.positions[v0 as usize] = collapse.new_position;

        // Update v0's quadric
        self.quadrics[v0 as usize] = self.quadrics[v0 as usize].add(&self.quadrics[v1 as usize]);

        // Mark v1 as removed
        self.removed_vertices.insert(v1);

        // Get triangles to update/remove
        let v1_triangles: Vec<usize> = self
            .vertex_triangles
            .get(&v1)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        // Update triangles containing v1
        for tri_idx in v1_triangles {
            if self.removed_triangles.contains(&tri_idx) {
                continue;
            }

            let tri = &mut self.triangles[tri_idx];

            // Replace v1 with v0
            for i in 0..3 {
                if tri[i] == v1 {
                    tri[i] = v0;
                }
            }

            // Check if triangle is degenerate
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                self.removed_triangles.insert(tri_idx);
            } else {
                // Update vertex-triangle map
                self.vertex_triangles.entry(v0).or_default().insert(tri_idx);
            }
        }

        // Remove v1 from vertex-triangle map
        self.vertex_triangles.remove(&v1);

        // Add new edge collapse candidates for v0's edges
        if let Some(v0_triangles) = self.vertex_triangles.get(&v0).cloned() {
            let mut new_edges = HashSet::new();
            for tri_idx in v0_triangles {
                if self.removed_triangles.contains(&tri_idx) {
                    continue;
                }
                let tri = &self.triangles[tri_idx];
                for i in 0..3 {
                    if tri[i] == v0 {
                        new_edges.insert(tri[(i + 1) % 3]);
                        new_edges.insert(tri[(i + 2) % 3]);
                    }
                }
            }

            for &other in &new_edges {
                if let Some(collapse) = self.compute_edge_collapse(v0.min(other), v0.max(other)) {
                    self.heap.push(collapse);
                }
            }
        }
    }

    /// Get the current active triangle count.
    pub fn active_triangle_count(&self) -> usize {
        self.triangles.len() - self.removed_triangles.len()
    }

    /// Get the simplified mesh data.
    pub fn get_result(&self) -> (Vec<[f32; 3]>, Vec<u32>) {
        // Build vertex remapping
        let mut vertex_remap: HashMap<u32, u32> = HashMap::new();
        let mut new_positions: Vec<[f32; 3]> = Vec::new();

        for (old_idx, pos) in self.positions.iter().enumerate() {
            if !self.removed_vertices.contains(&(old_idx as u32)) {
                let new_idx = new_positions.len() as u32;
                vertex_remap.insert(old_idx as u32, new_idx);
                new_positions.push([pos[0] as f32, pos[1] as f32, pos[2] as f32]);
            }
        }

        // Build new indices
        let mut new_indices: Vec<u32> = Vec::new();
        for (tri_idx, tri) in self.triangles.iter().enumerate() {
            if !self.removed_triangles.contains(&tri_idx) {
                if let (Some(&i0), Some(&i1), Some(&i2)) = (
                    vertex_remap.get(&tri[0]),
                    vertex_remap.get(&tri[1]),
                    vertex_remap.get(&tri[2]),
                ) {
                    new_indices.push(i0);
                    new_indices.push(i1);
                    new_indices.push(i2);
                }
            }
        }

        (new_positions, new_indices)
    }
}

/// Simplify a mesh to a target triangle count.
///
/// Returns (simplified_positions, simplified_indices, max_error).
pub fn simplify_mesh(
    positions: &[[f32; 3]],
    indices: &[u32],
    target_ratio: f32,
) -> (Vec<[f32; 3]>, Vec<u32>, f32) {
    let current_triangles = indices.len() / 3;
    let target_triangles = ((current_triangles as f32 * target_ratio) as usize).max(1);

    let mut simplifier = MeshSimplifier::new(positions, indices);
    let max_error = simplifier.simplify(target_triangles);
    let (new_positions, new_indices) = simplifier.get_result();

    (new_positions, new_indices, max_error as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadric_plane() {
        // Plane z = 0
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);

        // Point on plane should have zero error
        assert!(q.evaluate(0.0, 0.0, 0.0).abs() < 1e-10);
        assert!(q.evaluate(1.0, 2.0, 0.0).abs() < 1e-10);

        // Point off plane should have non-zero error
        assert!(q.evaluate(0.0, 0.0, 1.0) > 0.9);
    }

    #[test]
    fn test_simplify_cube() {
        // Simple cube (8 vertices, 12 triangles)
        let positions = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];

        let indices = [
            0, 1, 2, 0, 2, 3, // front
            4, 6, 5, 4, 7, 6, // back
            0, 4, 5, 0, 5, 1, // bottom
            2, 6, 7, 2, 7, 3, // top
            0, 3, 7, 0, 7, 4, // left
            1, 5, 6, 1, 6, 2, // right
        ];

        let (new_pos, new_idx, error) = simplify_mesh(&positions, &indices, 0.5);

        assert!(new_idx.len() / 3 <= 12);
        assert!(new_pos.len() <= 8);
        assert!(error >= 0.0);
    }
}
