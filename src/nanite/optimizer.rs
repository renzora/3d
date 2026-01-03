//! Nanite rendering optimizations.
//!
//! This module provides various optimization passes for the Nanite pipeline:
//! - Cluster compaction: Removes gaps in visible cluster list for better cache efficiency
//! - Material batching: Sorts clusters by material ID to reduce state changes
//! - Two-pass occlusion: Coarse then refined HZB testing for better culling
//! - Statistics collection: GPU-side performance counters

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Statistics collected during Nanite rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct NaniteStatistics {
    /// Total clusters in scene.
    pub total_clusters: u32,
    /// Clusters after frustum culling.
    pub frustum_passed: u32,
    /// Clusters after occlusion culling.
    pub occlusion_passed: u32,
    /// Clusters after LOD selection.
    pub lod_selected: u32,
    /// Clusters sent to SW rasterization.
    pub sw_clusters: u32,
    /// Clusters sent to HW rasterization.
    pub hw_clusters: u32,
    /// Triangles rendered via SW.
    pub sw_triangles: u32,
    /// Triangles rendered via HW.
    pub hw_triangles: u32,
    /// Number of unique materials rendered.
    pub material_count: u32,
    /// Number of material batches.
    pub batch_count: u32,
    /// Padding.
    pub _pad: [u32; 2],
}

/// Material batch for sorted rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct MaterialBatch {
    /// Material ID.
    pub material_id: u32,
    /// First cluster index in the sorted list.
    pub first_cluster: u32,
    /// Number of clusters with this material.
    pub cluster_count: u32,
    /// First triangle (for indirect draw).
    pub first_triangle: u32,
    /// Triangle count.
    pub triangle_count: u32,
    /// Padding.
    pub _pad: [u32; 3],
}

/// Compaction uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CompactUniform {
    /// Number of input elements.
    pub input_count: u32,
    /// Workgroup size.
    pub workgroup_size: u32,
    /// Padding.
    pub _pad: [u32; 2],
}

impl Default for CompactUniform {
    fn default() -> Self {
        Self {
            input_count: 0,
            workgroup_size: 256,
            _pad: [0; 2],
        }
    }
}

/// Material sort uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialSortUniform {
    /// Number of visible clusters.
    pub cluster_count: u32,
    /// Maximum material ID (for counting sort).
    pub max_material_id: u32,
    /// Padding.
    pub _pad: [u32; 2],
}

impl Default for MaterialSortUniform {
    fn default() -> Self {
        Self {
            cluster_count: 0,
            max_material_id: 256,
            _pad: [0; 2],
        }
    }
}

/// Two-pass occlusion uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TwoPassOcclusionUniform {
    /// View-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// HZB dimensions (width, height, mip_count, unused).
    pub hzb_size: [f32; 4],
    /// Params: x=cluster_count, y=coarse_mip_offset, z=refine_threshold, w=unused.
    pub params: [f32; 4],
}

impl Default for TwoPassOcclusionUniform {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            hzb_size: [512.0, 512.0, 9.0, 0.0],
            params: [0.0, 2.0, 0.5, 0.0],
        }
    }
}

/// Nanite optimizer managing all optimization passes.
pub struct NaniteOptimizer {
    // Compaction pipeline
    compact_pipeline: wgpu::ComputePipeline,
    compact_bind_group_layout: wgpu::BindGroupLayout,
    compact_uniform_buffer: wgpu::Buffer,
    compact_bind_group: Option<wgpu::BindGroup>,

    // Material sort pipeline
    material_count_pipeline: wgpu::ComputePipeline,
    material_scatter_pipeline: wgpu::ComputePipeline,
    material_bind_group_layout: wgpu::BindGroupLayout,
    material_uniform_buffer: wgpu::Buffer,
    material_bind_group: Option<wgpu::BindGroup>,

    // Two-pass occlusion
    coarse_occlusion_pipeline: wgpu::ComputePipeline,
    refine_occlusion_pipeline: wgpu::ComputePipeline,
    two_pass_bind_group_layout: wgpu::BindGroupLayout,
    two_pass_uniform_buffer: wgpu::Buffer,
    two_pass_bind_group: Option<wgpu::BindGroup>,

    // Buffers
    /// Compacted cluster buffer.
    pub compacted_buffer: wgpu::Buffer,
    /// Material count histogram.
    pub material_histogram: wgpu::Buffer,
    /// Material offsets (prefix sum of histogram).
    pub material_offsets: wgpu::Buffer,
    /// Sorted clusters by material.
    pub sorted_clusters: wgpu::Buffer,
    /// Material batches for rendering.
    pub material_batches: wgpu::Buffer,
    /// Coarse-passed clusters (intermediate for two-pass).
    pub coarse_passed_buffer: wgpu::Buffer,
    /// Coarse pass counter.
    pub coarse_counter_buffer: wgpu::Buffer,
    /// Statistics buffer.
    pub stats_buffer: wgpu::Buffer,
    /// Readback buffer for CPU access to stats.
    pub stats_readback_buffer: wgpu::Buffer,

    /// Maximum clusters.
    max_clusters: u32,
    /// Maximum materials.
    max_materials: u32,
}

impl NaniteOptimizer {
    /// Create a new optimizer.
    pub fn new(device: &wgpu::Device, max_clusters: u32, max_materials: u32) -> Self {
        // Create compaction shader and pipeline
        let compact_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite Compact Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/nanite_compact.wgsl").into(),
            ),
        });

        let compact_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite Compact Bind Group Layout"),
                entries: &[
                    // Uniform
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
                    // Input clusters (read)
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
                    // Input counter (read)
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
                    // Output clusters (write)
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
                ],
            });

        let compact_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nanite Compact Pipeline Layout"),
                bind_group_layouts: &[&compact_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nanite Compact Pipeline"),
            layout: Some(&compact_pipeline_layout),
            module: &compact_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let compact_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Compact Uniform"),
            size: std::mem::size_of::<CompactUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create material sort shaders and pipelines
        let material_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite Material Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/nanite_material_sort.wgsl").into(),
            ),
        });

        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite Material Sort Bind Group Layout"),
                entries: &[
                    // Uniform
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
                    // Visible clusters (read)
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
                    // Histogram (atomic)
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
                    // Offsets (read in scatter, write in prefix sum)
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
                    // Sorted output (write)
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
                    // Material batches (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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

        let material_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nanite Material Sort Pipeline Layout"),
                bind_group_layouts: &[&material_bind_group_layout],
                push_constant_ranges: &[],
            });

        let material_count_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Nanite Material Count Pipeline"),
                layout: Some(&material_pipeline_layout),
                module: &material_shader,
                entry_point: Some("count_materials"),
                compilation_options: Default::default(),
                cache: None,
            });

        let material_scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Nanite Material Scatter Pipeline"),
                layout: Some(&material_pipeline_layout),
                module: &material_shader,
                entry_point: Some("scatter_by_material"),
                compilation_options: Default::default(),
                cache: None,
            });

        let material_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Material Sort Uniform"),
            size: std::mem::size_of::<MaterialSortUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create two-pass occlusion shaders and pipelines
        let two_pass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite Two-Pass Occlusion Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/nanite_two_pass_occlusion.wgsl").into(),
            ),
        });

        let two_pass_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite Two-Pass Occlusion Bind Group Layout"),
                entries: &[
                    // Uniform
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
                    // Input clusters (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Input counter (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output clusters (write)
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
                    // Output counter (atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // HZB texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // HZB sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let two_pass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nanite Two-Pass Occlusion Pipeline Layout"),
                bind_group_layouts: &[&two_pass_bind_group_layout],
                push_constant_ranges: &[],
            });

        let coarse_occlusion_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Nanite Coarse Occlusion Pipeline"),
                layout: Some(&two_pass_pipeline_layout),
                module: &two_pass_shader,
                entry_point: Some("coarse_occlusion"),
                compilation_options: Default::default(),
                cache: None,
            });

        let refine_occlusion_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Nanite Refine Occlusion Pipeline"),
                layout: Some(&two_pass_pipeline_layout),
                module: &two_pass_shader,
                entry_point: Some("refine_occlusion"),
                compilation_options: Default::default(),
                cache: None,
            });

        let two_pass_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Two-Pass Occlusion Uniform"),
            size: std::mem::size_of::<TwoPassOcclusionUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create buffers
        let visible_cluster_size = 16u64; // 4 x u32

        let compacted_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Compacted Clusters"),
            size: max_clusters as u64 * visible_cluster_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_histogram = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Material Histogram"),
            size: max_materials as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Material Offsets"),
            size: max_materials as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sorted_clusters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Sorted Clusters"),
            size: max_clusters as u64 * visible_cluster_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_batches = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Material Batches"),
            size: max_materials as u64 * std::mem::size_of::<MaterialBatch>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let coarse_passed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Coarse Passed Clusters"),
            size: max_clusters as u64 * visible_cluster_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let coarse_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Coarse Counter"),
            size: 16, // 4 u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stats_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nanite Statistics"),
            contents: bytemuck::cast_slice(&[NaniteStatistics::default()]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let stats_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Statistics Readback"),
            size: std::mem::size_of::<NaniteStatistics>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            compact_pipeline,
            compact_bind_group_layout,
            compact_uniform_buffer,
            compact_bind_group: None,
            material_count_pipeline,
            material_scatter_pipeline,
            material_bind_group_layout,
            material_uniform_buffer,
            material_bind_group: None,
            coarse_occlusion_pipeline,
            refine_occlusion_pipeline,
            two_pass_bind_group_layout,
            two_pass_uniform_buffer,
            two_pass_bind_group: None,
            compacted_buffer,
            material_histogram,
            material_offsets,
            sorted_clusters,
            material_batches,
            coarse_passed_buffer,
            coarse_counter_buffer,
            stats_buffer,
            stats_readback_buffer,
            max_clusters,
            max_materials,
        }
    }

    /// Reset buffers for new frame.
    pub fn reset(&self, queue: &wgpu::Queue) {
        // Clear histogram
        let zero_histogram = vec![0u32; self.max_materials as usize];
        queue.write_buffer(&self.material_histogram, 0, bytemuck::cast_slice(&zero_histogram));

        // Clear coarse counter
        queue.write_buffer(&self.coarse_counter_buffer, 0, bytemuck::cast_slice(&[0u32; 4]));

        // Reset stats
        queue.write_buffer(
            &self.stats_buffer,
            0,
            bytemuck::cast_slice(&[NaniteStatistics::default()]),
        );
    }

    /// Create compact bind group.
    pub fn create_compact_bind_group(
        &mut self,
        device: &wgpu::Device,
        input_buffer: &wgpu::Buffer,
        counter_buffer: &wgpu::Buffer,
    ) {
        self.compact_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Compact Bind Group"),
            layout: &self.compact_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.compact_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.compacted_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Create material sort bind group.
    pub fn create_material_bind_group(
        &mut self,
        device: &wgpu::Device,
        cluster_buffer: &wgpu::Buffer,
        visible_clusters_buffer: &wgpu::Buffer,
    ) {
        self.material_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Material Sort Bind Group"),
            layout: &self.material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.material_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cluster_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: visible_clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.material_histogram.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.material_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.sorted_clusters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.material_batches.as_entire_binding(),
                },
            ],
        }));
    }

    /// Create two-pass occlusion bind group.
    pub fn create_two_pass_bind_group(
        &mut self,
        device: &wgpu::Device,
        cluster_buffer: &wgpu::Buffer,
        instance_buffer: &wgpu::Buffer,
        input_buffer: &wgpu::Buffer,
        input_counter: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        output_counter: &wgpu::Buffer,
        hzb_view: &wgpu::TextureView,
        hzb_sampler: &wgpu::Sampler,
    ) {
        self.two_pass_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Two-Pass Occlusion Bind Group"),
            layout: &self.two_pass_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.two_pass_uniform_buffer.as_entire_binding(),
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
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: input_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(hzb_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(hzb_sampler),
                },
            ],
        }));
    }

    /// Update compact uniform.
    pub fn update_compact_uniform(&self, queue: &wgpu::Queue, input_count: u32) {
        let uniform = CompactUniform {
            input_count,
            workgroup_size: 256,
            _pad: [0; 2],
        };
        queue.write_buffer(
            &self.compact_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform]),
        );
    }

    /// Update material sort uniform.
    pub fn update_material_uniform(&self, queue: &wgpu::Queue, cluster_count: u32) {
        let uniform = MaterialSortUniform {
            cluster_count,
            max_material_id: self.max_materials,
            _pad: [0; 2],
        };
        queue.write_buffer(
            &self.material_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform]),
        );
    }

    /// Update two-pass occlusion uniform.
    pub fn update_two_pass_uniform(
        &self,
        queue: &wgpu::Queue,
        view_proj: &[[f32; 4]; 4],
        hzb_width: f32,
        hzb_height: f32,
        hzb_mip_count: f32,
        cluster_count: u32,
    ) {
        let uniform = TwoPassOcclusionUniform {
            view_proj: *view_proj,
            hzb_size: [hzb_width, hzb_height, hzb_mip_count, 0.0],
            params: [cluster_count as f32, 2.0, 0.5, 0.0],
        };
        queue.write_buffer(
            &self.two_pass_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform]),
        );
    }

    /// Dispatch compaction pass.
    pub fn dispatch_compact(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        if let Some(bind_group) = &self.compact_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nanite Compact Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.compact_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            let workgroups = (count + 255) / 256;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
    }

    /// Dispatch material counting pass.
    pub fn dispatch_material_count(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        if let Some(bind_group) = &self.material_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nanite Material Count Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.material_count_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            let workgroups = (count + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
    }

    /// Dispatch material scatter pass.
    pub fn dispatch_material_scatter(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        if let Some(bind_group) = &self.material_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nanite Material Scatter Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.material_scatter_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            let workgroups = (count + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
    }

    /// Dispatch coarse occlusion pass.
    pub fn dispatch_coarse_occlusion(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        if let Some(bind_group) = &self.two_pass_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nanite Coarse Occlusion Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.coarse_occlusion_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            let workgroups = (count + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
    }

    /// Dispatch refine occlusion pass.
    pub fn dispatch_refine_occlusion(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        if let Some(bind_group) = &self.two_pass_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nanite Refine Occlusion Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.refine_occlusion_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            let workgroups = (count + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
    }

    /// Copy statistics to readback buffer.
    pub fn copy_stats(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.stats_buffer,
            0,
            &self.stats_readback_buffer,
            0,
            std::mem::size_of::<NaniteStatistics>() as u64,
        );
    }

    /// Get compacted buffer for rendering.
    pub fn compacted_buffer(&self) -> &wgpu::Buffer {
        &self.compacted_buffer
    }

    /// Get sorted clusters buffer.
    pub fn sorted_clusters(&self) -> &wgpu::Buffer {
        &self.sorted_clusters
    }

    /// Get material batches buffer.
    pub fn material_batches(&self) -> &wgpu::Buffer {
        &self.material_batches
    }
}
