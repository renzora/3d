//! Software rasterization for small triangles.
//!
//! When triangles project to less than ~2x2 pixels, hardware rasterization
//! becomes inefficient due to quad overshading. This module provides a
//! compute shader-based software rasterizer for these small triangles.
//!
//! ## Pipeline
//! 1. Classification pass: Sort visible clusters into SW/HW bins based on screen-space size
//! 2. SW rasterization: Compute shader rasterizes small triangles directly to visibility buffer
//! 3. HW rasterization: Standard render pass for larger triangles
//! 4. Results merge in the visibility buffer (using atomic depth test)

use bytemuck::{Pod, Zeroable};

/// Software rasterization uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SwRasterUniform {
    /// View-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Inverse view-projection matrix (for depth reconstruction).
    pub inv_view_proj: [[f32; 4]; 4],
    /// Screen dimensions: x=width, y=height, z=1/width, w=1/height.
    pub screen_size: [f32; 4],
    /// Raster params: x=sw_cluster_count, y=total_triangles, z=depth_bias, w=unused.
    pub params: [f32; 4],
}

impl Default for SwRasterUniform {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            inv_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            screen_size: [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0],
            params: [0.0; 4],
        }
    }
}

/// Classification uniform for sorting clusters into SW/HW bins.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ClassifyUniform {
    /// View-projection matrix for screen-space size calculation.
    pub view_proj: [[f32; 4]; 4],
    /// Screen dimensions: x=width, y=height, z=unused, w=unused.
    pub screen_size: [f32; 4],
    /// Params: x=visible_count, y=size_threshold (pixels), z=unused, w=unused.
    pub params: [f32; 4],
}

impl Default for ClassifyUniform {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            screen_size: [1920.0, 1080.0, 0.0, 0.0],
            params: [0.0, 2.0, 0.0, 0.0], // Default 2 pixel threshold
        }
    }
}

/// Counters for SW/HW cluster classification.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct ClassifyCounters {
    /// Number of clusters routed to software rasterization.
    pub sw_count: u32,
    /// Number of clusters routed to hardware rasterization.
    pub hw_count: u32,
    /// Total triangles for SW rasterization.
    pub sw_triangles: u32,
    /// Total triangles for HW rasterization.
    pub hw_triangles: u32,
}

/// Software rasterizer.
pub struct SoftwareRasterizer {
    // Classification pipeline
    classify_pipeline: wgpu::ComputePipeline,
    classify_bind_group_layout: wgpu::BindGroupLayout,
    classify_uniform_buffer: wgpu::Buffer,
    classify_bind_group: Option<wgpu::BindGroup>,

    // SW rasterization pipeline
    sw_raster_pipeline: wgpu::ComputePipeline,
    sw_raster_bind_group_layout: wgpu::BindGroupLayout,
    sw_raster_uniform_buffer: wgpu::Buffer,
    sw_raster_bind_group: Option<wgpu::BindGroup>,

    // Classification output buffers
    /// Clusters routed to software rasterization.
    pub sw_clusters_buffer: wgpu::Buffer,
    /// Clusters routed to hardware rasterization.
    pub hw_clusters_buffer: wgpu::Buffer,
    /// Classification counters.
    pub classify_counters_buffer: wgpu::Buffer,
    /// HW indirect draw buffer.
    pub hw_indirect_buffer: wgpu::Buffer,

    // Atomic depth buffer for SW rasterization
    /// Atomic depth buffer (stores depth as u32 for atomic compare-and-swap).
    pub atomic_depth_buffer: wgpu::Buffer,

    // Configuration
    /// Screen width.
    pub width: u32,
    /// Screen height.
    pub height: u32,
    /// Pixel size threshold for SW rasterization.
    pub size_threshold: f32,
    /// Maximum clusters.
    max_clusters: u32,
}

impl SoftwareRasterizer {
    /// Create a new software rasterizer.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, max_clusters: u32) -> Self {
        // Create classification pipeline
        let classify_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite Classify Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/nanite_classify.wgsl").into(),
            ),
        });

        let classify_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite Classify Bind Group Layout"),
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
                    // Visible clusters input (read)
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
                    // SW clusters output (write)
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
                    // HW clusters output (write)
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
                    // Counters (atomic)
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
                    // HW indirect args (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
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

        let classify_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nanite Classify Pipeline Layout"),
                bind_group_layouts: &[&classify_bind_group_layout],
                push_constant_ranges: &[],
            });

        let classify_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nanite Classify Pipeline"),
            layout: Some(&classify_pipeline_layout),
            module: &classify_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let classify_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Classify Uniform Buffer"),
            size: std::mem::size_of::<ClassifyUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create SW rasterization pipeline
        let sw_raster_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite SW Rasterize Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/nanite_sw_rasterize.wgsl").into(),
            ),
        });

        let sw_raster_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite SW Rasterize Bind Group Layout"),
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
                    // Positions (read)
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
                    // Indices (read)
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
                    // SW clusters (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Classify counters (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Visibility buffer (storage texture, atomic write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadWrite,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Atomic depth buffer (storage buffer)
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
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

        let sw_raster_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nanite SW Rasterize Pipeline Layout"),
                bind_group_layouts: &[&sw_raster_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sw_raster_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nanite SW Rasterize Pipeline"),
            layout: Some(&sw_raster_pipeline_layout),
            module: &sw_raster_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sw_raster_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite SW Rasterize Uniform Buffer"),
            size: std::mem::size_of::<SwRasterUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create classification output buffers
        let visible_cluster_size = 16u64; // 4 x u32
        let sw_clusters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite SW Clusters Buffer"),
            size: max_clusters as u64 * visible_cluster_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hw_clusters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite HW Clusters Buffer"),
            size: max_clusters as u64 * visible_cluster_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let classify_counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Classify Counters Buffer"),
            size: std::mem::size_of::<ClassifyCounters>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hw_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite HW Indirect Buffer"),
            size: 16, // 4 x u32 for indirect draw
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create atomic depth buffer (one u32 per pixel)
        let atomic_depth_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Atomic Depth Buffer"),
            size: (width * height * 4) as u64, // 4 bytes per pixel
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            classify_pipeline,
            classify_bind_group_layout,
            classify_uniform_buffer,
            classify_bind_group: None,
            sw_raster_pipeline,
            sw_raster_bind_group_layout,
            sw_raster_uniform_buffer,
            sw_raster_bind_group: None,
            sw_clusters_buffer,
            hw_clusters_buffer,
            classify_counters_buffer,
            hw_indirect_buffer,
            atomic_depth_buffer,
            width,
            height,
            size_threshold: 2.0,
            max_clusters,
        }
    }

    /// Resize buffers for new screen dimensions.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }

        self.width = width;
        self.height = height;

        // Recreate atomic depth buffer
        self.atomic_depth_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Atomic Depth Buffer"),
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind groups need to be recreated
        self.classify_bind_group = None;
        self.sw_raster_bind_group = None;
    }

    /// Create classify bind group.
    pub fn create_classify_bind_group(
        &mut self,
        device: &wgpu::Device,
        cluster_buffer: &wgpu::Buffer,
        instance_buffer: &wgpu::Buffer,
        visible_clusters_buffer: &wgpu::Buffer,
    ) {
        self.classify_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Classify Bind Group"),
            layout: &self.classify_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.classify_uniform_buffer.as_entire_binding(),
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
                    resource: self.sw_clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.hw_clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.classify_counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.hw_indirect_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Create SW rasterize bind group.
    pub fn create_sw_raster_bind_group(
        &mut self,
        device: &wgpu::Device,
        cluster_buffer: &wgpu::Buffer,
        instance_buffer: &wgpu::Buffer,
        position_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        visibility_view: &wgpu::TextureView,
    ) {
        self.sw_raster_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite SW Rasterize Bind Group"),
            layout: &self.sw_raster_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sw_raster_uniform_buffer.as_entire_binding(),
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
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.sw_clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.classify_counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(visibility_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.atomic_depth_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Reset counters for new frame.
    pub fn reset_counters(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.classify_counters_buffer,
            0,
            bytemuck::cast_slice(&[ClassifyCounters::default()]),
        );
        // Reset HW indirect args
        queue.write_buffer(
            &self.hw_indirect_buffer,
            0,
            bytemuck::cast_slice(&[0u32, 0u32, 0u32, 0u32]),
        );
    }

    /// Clear atomic depth buffer to minimum depth (0 = farthest in our scheme).
    pub fn clear_atomic_depth(&self, queue: &wgpu::Queue) {
        // Clear to 0 (farthest depth in our representation where higher = closer)
        let clear_data = vec![0u32; (self.width * self.height) as usize];
        queue.write_buffer(&self.atomic_depth_buffer, 0, bytemuck::cast_slice(&clear_data));
    }

    /// Update classify uniform.
    pub fn update_classify_uniform(
        &self,
        queue: &wgpu::Queue,
        view_proj: &[[f32; 4]; 4],
        visible_count: u32,
    ) {
        let uniform = ClassifyUniform {
            view_proj: *view_proj,
            screen_size: [self.width as f32, self.height as f32, 0.0, 0.0],
            params: [visible_count as f32, self.size_threshold, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.classify_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform]),
        );
    }

    /// Update SW rasterize uniform.
    pub fn update_sw_raster_uniform(
        &self,
        queue: &wgpu::Queue,
        view_proj: &[[f32; 4]; 4],
        inv_view_proj: &[[f32; 4]; 4],
        sw_cluster_count: u32,
        total_triangles: u32,
    ) {
        let uniform = SwRasterUniform {
            view_proj: *view_proj,
            inv_view_proj: *inv_view_proj,
            screen_size: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            params: [sw_cluster_count as f32, total_triangles as f32, 0.0001, 0.0],
        };
        queue.write_buffer(
            &self.sw_raster_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform]),
        );
    }

    /// Dispatch classification pass.
    pub fn dispatch_classify(&self, encoder: &mut wgpu::CommandEncoder, visible_count: u32) {
        if let Some(bind_group) = &self.classify_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nanite Classify Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.classify_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            let workgroups = (visible_count + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
    }

    /// Dispatch SW rasterization pass.
    pub fn dispatch_sw_rasterize(&self, encoder: &mut wgpu::CommandEncoder, triangle_count: u32) {
        if let Some(bind_group) = &self.sw_raster_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nanite SW Rasterize Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.sw_raster_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            // One thread per triangle
            let workgroups = (triangle_count + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
    }

    /// Get HW clusters buffer for hardware rasterization.
    pub fn hw_clusters_buffer(&self) -> &wgpu::Buffer {
        &self.hw_clusters_buffer
    }

    /// Get HW indirect buffer for hardware rasterization.
    pub fn hw_indirect_buffer(&self) -> &wgpu::Buffer {
        &self.hw_indirect_buffer
    }

    /// Set the pixel size threshold for SW rasterization.
    pub fn set_size_threshold(&mut self, threshold: f32) {
        self.size_threshold = threshold;
    }
}
