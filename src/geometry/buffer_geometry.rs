//! Buffer geometry for storing vertex and index data.

use crate::core::Id;
use crate::math::{Box3, Sphere, Vector3};

/// A geometry stored in GPU buffers.
pub struct BufferGeometry {
    /// Unique ID.
    id: Id,
    /// Vertex buffer.
    vertex_buffer: Option<wgpu::Buffer>,
    /// Index buffer.
    index_buffer: Option<wgpu::Buffer>,
    /// Number of vertices.
    vertex_count: u32,
    /// Number of indices.
    index_count: u32,
    /// Bounding box.
    bounding_box: Option<Box3>,
    /// Bounding sphere.
    bounding_sphere: Option<Sphere>,
    /// Whether the geometry needs to update GPU buffers.
    needs_update: bool,
}

impl BufferGeometry {
    /// Create a new empty buffer geometry.
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            vertex_buffer: None,
            index_buffer: None,
            vertex_count: 0,
            index_count: 0,
            bounding_box: None,
            bounding_sphere: None,
            needs_update: true,
        }
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the vertex buffer.
    #[inline]
    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    /// Get the index buffer.
    #[inline]
    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    /// Get the vertex count.
    #[inline]
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Get the index count.
    #[inline]
    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    /// Check if the geometry has indices.
    #[inline]
    pub fn has_indices(&self) -> bool {
        self.index_buffer.is_some() && self.index_count > 0
    }

    /// Get the bounding box.
    #[inline]
    pub fn bounding_box(&self) -> Option<&Box3> {
        self.bounding_box.as_ref()
    }

    /// Get the bounding sphere.
    #[inline]
    pub fn bounding_sphere(&self) -> Option<&Sphere> {
        self.bounding_sphere.as_ref()
    }

    /// Set the vertex buffer.
    pub fn set_vertex_buffer(&mut self, buffer: wgpu::Buffer, count: u32) {
        self.vertex_buffer = Some(buffer);
        self.vertex_count = count;
    }

    /// Set the index buffer.
    pub fn set_index_buffer(&mut self, buffer: wgpu::Buffer, count: u32) {
        self.index_buffer = Some(buffer);
        self.index_count = count;
    }

    /// Set the bounding box.
    pub fn set_bounding_box(&mut self, bbox: Box3) {
        self.bounding_box = Some(bbox);
    }

    /// Set the bounding sphere.
    pub fn set_bounding_sphere(&mut self, sphere: Sphere) {
        self.bounding_sphere = Some(sphere);
    }

    /// Compute bounding box from positions.
    pub fn compute_bounding_box(&mut self, positions: &[[f32; 3]]) {
        if positions.is_empty() {
            self.bounding_box = None;
            return;
        }

        let mut bbox = Box3::EMPTY;
        for pos in positions {
            bbox.expand_by_point(&Vector3::new(pos[0], pos[1], pos[2]));
        }
        self.bounding_box = Some(bbox);
    }

    /// Compute bounding sphere from bounding box.
    pub fn compute_bounding_sphere(&mut self) {
        if let Some(bbox) = &self.bounding_box {
            let center = bbox.center();
            let radius = bbox.size().length() * 0.5;
            self.bounding_sphere = Some(Sphere::new(center, radius));
        }
    }

    /// Mark geometry as needing update.
    pub fn mark_needs_update(&mut self) {
        self.needs_update = true;
    }

    /// Check if geometry needs update.
    #[inline]
    pub fn needs_update(&self) -> bool {
        self.needs_update
    }

    /// Clear the needs update flag.
    pub fn clear_needs_update(&mut self) {
        self.needs_update = false;
    }
}

impl Default for BufferGeometry {
    fn default() -> Self {
        Self::new()
    }
}
