//! Instanced mesh for rendering many copies efficiently.

use crate::core::Id;
use crate::geometry::BufferGeometry;
use crate::math::{Matrix4, Quaternion, Vector3};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Instance data for GPU instancing.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct InstanceData {
    /// Model matrix (column-major).
    pub model_matrix: [[f32; 4]; 4],
    /// Instance color (RGBA).
    pub color: [f32; 4],
}

impl Default for InstanceData {
    fn default() -> Self {
        Self {
            model_matrix: Matrix4::IDENTITY.to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl InstanceData {
    /// Create instance data from transform components.
    pub fn from_transform(position: Vector3, rotation: Quaternion, scale: Vector3) -> Self {
        let matrix = Matrix4::compose(&position, &rotation, &scale);
        Self {
            model_matrix: matrix.to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// Create instance data from transform with color.
    pub fn from_transform_color(
        position: Vector3,
        rotation: Quaternion,
        scale: Vector3,
        color: [f32; 4],
    ) -> Self {
        let matrix = Matrix4::compose(&position, &rotation, &scale);
        Self {
            model_matrix: matrix.to_cols_array_2d(),
            color,
        }
    }

    /// Create instance data from matrix.
    pub fn from_matrix(matrix: Matrix4) -> Self {
        Self {
            model_matrix: matrix.to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// Get the vertex buffer layout for instancing.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }

    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
        // model_matrix column 0
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32x4,
        },
        // model_matrix column 1
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
            shader_location: 4,
            format: wgpu::VertexFormat::Float32x4,
        },
        // model_matrix column 2
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32x4,
        },
        // model_matrix column 3
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
            shader_location: 6,
            format: wgpu::VertexFormat::Float32x4,
        },
        // color
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
            shader_location: 7,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];
}

/// An instanced mesh renders many copies of the same geometry efficiently.
pub struct InstancedMesh {
    /// Unique identifier.
    id: Id,
    /// Object name.
    name: String,
    /// Shared geometry.
    geometry: Arc<BufferGeometry>,
    /// Material index.
    material_index: usize,
    /// Instance data (CPU side).
    instances: Vec<InstanceData>,
    /// Instance buffer (GPU side).
    instance_buffer: Option<wgpu::Buffer>,
    /// Whether instances need GPU update.
    needs_update: bool,
    /// Visibility flag.
    pub visible: bool,
    /// Cast shadows.
    pub cast_shadow: bool,
    /// Receive shadows.
    pub receive_shadow: bool,
    /// Frustum culling (per-instance would be expensive).
    pub frustum_culled: bool,
}

impl InstancedMesh {
    /// Create a new instanced mesh.
    pub fn new(geometry: Arc<BufferGeometry>, count: usize) -> Self {
        let instances = vec![InstanceData::default(); count];
        Self {
            id: Id::new(),
            name: String::new(),
            geometry,
            material_index: 0,
            instances,
            instance_buffer: None,
            needs_update: true,
            visible: true,
            cast_shadow: true,
            receive_shadow: true,
            frustum_culled: false,
        }
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Get the geometry.
    #[inline]
    pub fn geometry(&self) -> &BufferGeometry {
        &self.geometry
    }

    /// Get the material index.
    #[inline]
    pub fn material_index(&self) -> usize {
        self.material_index
    }

    /// Set the material index.
    pub fn set_material_index(&mut self, index: usize) {
        self.material_index = index;
    }

    /// Get the number of instances.
    #[inline]
    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Resize the instance count.
    pub fn set_count(&mut self, count: usize) {
        self.instances.resize(count, InstanceData::default());
        self.needs_update = true;
    }

    /// Get instance data.
    #[inline]
    pub fn get_instance(&self, index: usize) -> Option<&InstanceData> {
        self.instances.get(index)
    }

    /// Set instance data.
    pub fn set_instance(&mut self, index: usize, data: InstanceData) {
        if index < self.instances.len() {
            self.instances[index] = data;
            self.needs_update = true;
        }
    }

    /// Set instance matrix.
    pub fn set_matrix_at(&mut self, index: usize, matrix: Matrix4) {
        if index < self.instances.len() {
            self.instances[index].model_matrix = matrix.to_cols_array_2d();
            self.needs_update = true;
        }
    }

    /// Set instance color.
    pub fn set_color_at(&mut self, index: usize, color: [f32; 4]) {
        if index < self.instances.len() {
            self.instances[index].color = color;
            self.needs_update = true;
        }
    }

    /// Set instance transform.
    pub fn set_transform_at(
        &mut self,
        index: usize,
        position: Vector3,
        rotation: Quaternion,
        scale: Vector3,
    ) {
        if index < self.instances.len() {
            let matrix = Matrix4::compose(&position, &rotation, &scale);
            self.instances[index].model_matrix = matrix.to_cols_array_2d();
            self.needs_update = true;
        }
    }

    /// Get all instances.
    #[inline]
    pub fn instances(&self) -> &[InstanceData] {
        &self.instances
    }

    /// Get mutable access to all instances.
    pub fn instances_mut(&mut self) -> &mut [InstanceData] {
        self.needs_update = true;
        &mut self.instances
    }

    /// Check if GPU buffer needs update.
    #[inline]
    pub fn needs_update(&self) -> bool {
        self.needs_update
    }

    /// Mark as needing update.
    pub fn mark_needs_update(&mut self) {
        self.needs_update = true;
    }

    /// Get the instance buffer.
    #[inline]
    pub fn instance_buffer(&self) -> Option<&wgpu::Buffer> {
        self.instance_buffer.as_ref()
    }

    /// Update GPU buffer.
    pub fn update_buffer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.needs_update {
            return;
        }

        let data = bytemuck::cast_slice(&self.instances);

        if let Some(ref buffer) = self.instance_buffer {
            // Check if buffer is large enough
            if buffer.size() >= data.len() as u64 {
                queue.write_buffer(buffer, 0, data);
            } else {
                // Recreate buffer
                self.instance_buffer = Some(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer"),
                        contents: data,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        } else {
            // Create buffer
            self.instance_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer"),
                    contents: data,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            ));
        }

        self.needs_update = false;
    }
}

impl std::fmt::Debug for InstancedMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstancedMesh")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("count", &self.instances.len())
            .field("visible", &self.visible)
            .finish()
    }
}
