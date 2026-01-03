//! GPU frustum culling for Nanite clusters.
//!
//! Supports both single-phase (frustum only) and two-phase (frustum + occlusion) culling.

use bytemuck::{Pod, Zeroable};
use crate::math::{Frustum, Matrix4};

/// Frustum culling uniform data for GPU (128 bytes).
///
/// Contains the 6 frustum planes in a GPU-friendly format,
/// plus additional culling parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FrustumCullUniform {
    /// Frustum planes (6 planes, each as vec4: xyz=normal, w=distance).
    /// Order: left, right, bottom, top, near, far
    pub planes: [[f32; 4]; 6],
    /// Culling params: x=cluster_count, y=instance_count, z=screen_height, w=fov_y
    pub params: [f32; 4],
    /// View-projection matrix for screen-space calculations.
    pub view_proj: [[f32; 4]; 4],
}

impl Default for FrustumCullUniform {
    fn default() -> Self {
        Self {
            planes: [[0.0, 1.0, 0.0, 1000.0]; 6],
            params: [0.0; 4],
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

impl FrustumCullUniform {
    /// Create culling uniform from a view-projection matrix.
    pub fn from_view_proj(
        view_proj: &Matrix4,
        cluster_count: u32,
        instance_count: u32,
        screen_height: f32,
        fov_y: f32,
    ) -> Self {
        let frustum = Frustum::from_matrix(view_proj);

        let mut planes = [[0.0f32; 4]; 6];
        for (i, plane) in frustum.planes.iter().enumerate() {
            planes[i] = [
                plane.normal.x,
                plane.normal.y,
                plane.normal.z,
                plane.constant,
            ];
        }

        Self {
            planes,
            params: [cluster_count as f32, instance_count as f32, screen_height, fov_y],
            view_proj: view_proj.to_cols_array_2d(),
        }
    }
}

/// Manages GPU frustum culling compute pipeline.
pub struct FrustumCuller {
    /// Compute pipeline for frustum culling.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for culling.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer for culling parameters.
    uniform_buffer: wgpu::Buffer,
    /// Bind group for culling (recreated when resources change).
    bind_group: Option<wgpu::BindGroup>,
}

impl FrustumCuller {
    /// Create a new frustum culler.
    pub fn new(device: &wgpu::Device) -> Self {
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite Frustum Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/nanite_frustum_cull.wgsl").into(),
            ),
        });

        // Bind group layout for culling
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nanite Frustum Cull Bind Group Layout"),
            entries: &[
                // Culling uniform (frustum planes, params)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Clusters (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Instances (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Visible clusters output (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Counters (atomic read/write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Indirect draw args (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Nanite Frustum Cull Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nanite Frustum Cull Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Frustum Cull Uniform Buffer"),
            size: std::mem::size_of::<FrustumCullUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            bind_group: None,
        }
    }

    /// Create or update the bind group with current buffers.
    /// For single-phase culling (no occlusion), outputs directly to visible_clusters_buffer.
    pub fn update_bind_group(
        &mut self,
        device: &wgpu::Device,
        cluster_buffer: &wgpu::Buffer,
        instance_buffer: &wgpu::Buffer,
        visible_clusters_buffer: &wgpu::Buffer,
        counter_buffer: &wgpu::Buffer,
        indirect_buffer: &wgpu::Buffer,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Frustum Cull Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cluster_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visible_clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: indirect_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Create or update the bind group for two-phase culling.
    /// For two-phase culling (with occlusion), outputs to frustum_clusters_buffer.
    pub fn update_bind_group_for_two_phase(
        &mut self,
        device: &wgpu::Device,
        cluster_buffer: &wgpu::Buffer,
        instance_buffer: &wgpu::Buffer,
        frustum_clusters_buffer: &wgpu::Buffer,
        frustum_counter_buffer: &wgpu::Buffer,
        indirect_buffer: &wgpu::Buffer,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Frustum Cull Bind Group (Two-Phase)"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cluster_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: frustum_clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: frustum_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: indirect_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Update culling uniforms.
    pub fn update_uniform(
        &self,
        queue: &wgpu::Queue,
        view_proj: &Matrix4,
        cluster_count: u32,
        instance_count: u32,
        screen_height: f32,
        fov_y: f32,
    ) {
        let uniform = FrustumCullUniform::from_view_proj(
            view_proj,
            cluster_count,
            instance_count,
            screen_height,
            fov_y,
        );
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Dispatch frustum culling compute shader.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, cluster_count: u32) {
        if let Some(bind_group) = &self.bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nanite Frustum Cull Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            // Dispatch one thread per cluster, 64 threads per workgroup
            let workgroups = (cluster_count + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Get the bind group layout for external use.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
