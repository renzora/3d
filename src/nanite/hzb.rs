//! Hierarchical Z-Buffer (HZB) for occlusion culling.
//!
//! The HZB is a mip chain of the depth buffer where each texel contains
//! the maximum depth (furthest) of its corresponding region. This allows
//! efficient occlusion testing by sampling a single texel at the appropriate
//! mip level for a cluster's screen-space bounding box.

use bytemuck::{Pod, Zeroable};

/// HZB build uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct HzbBuildUniform {
    /// Source mip dimensions (width, height).
    pub src_size: [u32; 2],
    /// Destination mip dimensions (width, height).
    pub dst_size: [u32; 2],
}

impl Default for HzbBuildUniform {
    fn default() -> Self {
        Self {
            src_size: [1, 1],
            dst_size: [1, 1],
        }
    }
}

/// Occlusion culling uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OcclusionCullUniform {
    /// View-projection matrix for projecting bounding boxes.
    pub view_proj: [[f32; 4]; 4],
    /// Camera position for distance calculations.
    pub camera_pos: [f32; 4],
    /// HZB dimensions at mip 0 (width, height, mip_count, unused).
    pub hzb_size: [f32; 4],
    /// Culling params: x=cluster_count, y=instance_count, z=screen_height, w=fov_y
    pub params: [f32; 4],
}

impl Default for OcclusionCullUniform {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_pos: [0.0, 0.0, 5.0, 1.0],
            hzb_size: [512.0, 512.0, 9.0, 0.0],
            params: [0.0; 4],
        }
    }
}

/// Hierarchical Z-Buffer generator and occlusion culler.
pub struct HzbGenerator {
    /// HZB texture with mip chain.
    pub hzb_texture: wgpu::Texture,
    /// Views for each mip level.
    pub hzb_mip_views: Vec<wgpu::TextureView>,
    /// Full texture view for sampling.
    pub hzb_view: wgpu::TextureView,
    /// Sampler for HZB texture.
    pub hzb_sampler: wgpu::Sampler,
    /// HZB width at mip 0.
    pub width: u32,
    /// HZB height at mip 0.
    pub height: u32,
    /// Number of mip levels.
    pub mip_count: u32,

    // Build pipeline
    build_pipeline: wgpu::ComputePipeline,
    build_bind_group_layout: wgpu::BindGroupLayout,
    build_uniform_buffer: wgpu::Buffer,
    build_bind_groups: Vec<wgpu::BindGroup>,

    // Occlusion cull pipeline
    occlusion_pipeline: wgpu::ComputePipeline,
    occlusion_bind_group_layout: wgpu::BindGroupLayout,
    occlusion_uniform_buffer: wgpu::Buffer,
    occlusion_bind_group: Option<wgpu::BindGroup>,
}

impl HzbGenerator {
    /// Calculate the number of mip levels for given dimensions.
    fn calculate_mip_count(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height);
        (max_dim as f32).log2().floor() as u32 + 1
    }

    /// Create a new HZB generator.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // Round up to power of 2 for clean mip chain
        let hzb_width = width.next_power_of_two().max(64);
        let hzb_height = height.next_power_of_two().max(64);
        let mip_count = Self::calculate_mip_count(hzb_width, hzb_height);

        // Create HZB texture with mip chain
        // Using R32Float to store depth values
        let hzb_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Nanite HZB Texture"),
            size: wgpu::Extent3d {
                width: hzb_width,
                height: hzb_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create view for each mip level
        let mut hzb_mip_views = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            let view = hzb_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Nanite HZB Mip {} View", mip)),
                format: Some(wgpu::TextureFormat::R32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });
            hzb_mip_views.push(view);
        }

        // Full texture view for sampling in occlusion cull
        let hzb_view = hzb_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Nanite HZB Full View"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_count),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        // Sampler for HZB (use nearest to get conservative depth)
        let hzb_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Nanite HZB Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create HZB build pipeline
        let build_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite HZB Build Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/nanite_hzb_build.wgsl").into(),
            ),
        });

        let build_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite HZB Build Bind Group Layout"),
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
                    // Source mip (texture)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Destination mip (storage texture)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let build_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Nanite HZB Build Pipeline Layout"),
            bind_group_layouts: &[&build_bind_group_layout],
            push_constant_ranges: &[],
        });

        let build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nanite HZB Build Pipeline"),
            layout: Some(&build_pipeline_layout),
            module: &build_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let build_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite HZB Build Uniform Buffer"),
            size: std::mem::size_of::<HzbBuildUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create occlusion cull pipeline
        let occlusion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite Occlusion Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/nanite_occlusion_cull.wgsl").into(),
            ),
        });

        let occlusion_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite Occlusion Cull Bind Group Layout"),
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
                    // Frustum-passed clusters (read) - input from frustum cull
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
                    // Frustum counter (read) - number of frustum-passed clusters
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
                    // Visible clusters output (write)
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
                    // Visible counter (atomic)
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
                    // Indirect draw args (write)
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
                    // HZB texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
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
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let occlusion_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nanite Occlusion Cull Pipeline Layout"),
                bind_group_layouts: &[&occlusion_bind_group_layout],
                push_constant_ranges: &[],
            });

        let occlusion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nanite Occlusion Cull Pipeline"),
            layout: Some(&occlusion_pipeline_layout),
            module: &occlusion_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let occlusion_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Occlusion Cull Uniform Buffer"),
            size: std::mem::size_of::<OcclusionCullUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            hzb_texture,
            hzb_mip_views,
            hzb_view,
            hzb_sampler,
            width: hzb_width,
            height: hzb_height,
            mip_count,
            build_pipeline,
            build_bind_group_layout,
            build_uniform_buffer,
            build_bind_groups: Vec::new(),
            occlusion_pipeline,
            occlusion_bind_group_layout,
            occlusion_uniform_buffer,
            occlusion_bind_group: None,
        }
    }

    /// Resize the HZB texture.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let hzb_width = width.next_power_of_two().max(64);
        let hzb_height = height.next_power_of_two().max(64);

        if hzb_width == self.width && hzb_height == self.height {
            return;
        }

        *self = Self::new(device, width, height);
    }

    /// Create bind groups for HZB build passes.
    /// Call this after the depth texture is available.
    pub fn create_build_bind_groups(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
    ) {
        self.build_bind_groups.clear();

        // First pass: depth buffer -> mip 0
        // Subsequent passes: mip N-1 -> mip N
        for mip in 0..self.mip_count {
            let src_view = if mip == 0 {
                depth_view
            } else {
                &self.hzb_mip_views[(mip - 1) as usize]
            };
            let dst_view = &self.hzb_mip_views[mip as usize];

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Nanite HZB Build Bind Group Mip {}", mip)),
                layout: &self.build_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.build_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(dst_view),
                    },
                ],
            });

            self.build_bind_groups.push(bind_group);
        }
    }

    /// Create bind group for occlusion culling.
    pub fn create_occlusion_bind_group(
        &mut self,
        device: &wgpu::Device,
        cluster_buffer: &wgpu::Buffer,
        instance_buffer: &wgpu::Buffer,
        frustum_clusters_buffer: &wgpu::Buffer,
        frustum_counter_buffer: &wgpu::Buffer,
        visible_clusters_buffer: &wgpu::Buffer,
        visible_counter_buffer: &wgpu::Buffer,
        indirect_buffer: &wgpu::Buffer,
    ) {
        self.occlusion_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Occlusion Cull Bind Group"),
            layout: &self.occlusion_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.occlusion_uniform_buffer.as_entire_binding(),
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
                    resource: visible_clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: visible_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: indirect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&self.hzb_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&self.hzb_sampler),
                },
            ],
        }));
    }

    /// Build the HZB mip chain from the depth buffer.
    pub fn build_hzb(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, src_width: u32, src_height: u32) {
        if self.build_bind_groups.is_empty() {
            return;
        }

        let mut src_w = src_width;
        let mut src_h = src_height;

        for mip in 0..self.mip_count {
            let dst_w = (self.width >> mip).max(1);
            let dst_h = (self.height >> mip).max(1);

            // Update uniform
            let uniform = HzbBuildUniform {
                src_size: [src_w, src_h],
                dst_size: [dst_w, dst_h],
            };
            queue.write_buffer(&self.build_uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

            // Dispatch compute
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Nanite HZB Build Mip {}", mip)),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, &self.build_bind_groups[mip as usize], &[]);

            // 8x8 workgroups
            let workgroups_x = (dst_w + 7) / 8;
            let workgroups_y = (dst_h + 7) / 8;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);

            // Next iteration uses this mip as source
            src_w = dst_w;
            src_h = dst_h;
        }
    }

    /// Update occlusion culling uniforms.
    pub fn update_occlusion_uniform(
        &self,
        queue: &wgpu::Queue,
        view_proj: &[[f32; 4]; 4],
        camera_pos: [f32; 3],
        cluster_count: u32,
        instance_count: u32,
        screen_height: f32,
        fov_y: f32,
    ) {
        let uniform = OcclusionCullUniform {
            view_proj: *view_proj,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
            hzb_size: [self.width as f32, self.height as f32, self.mip_count as f32, 0.0],
            params: [cluster_count as f32, instance_count as f32, screen_height, fov_y],
        };
        queue.write_buffer(&self.occlusion_uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Dispatch occlusion culling.
    /// This reads from the frustum-passed cluster list and outputs the final visible list.
    pub fn dispatch_occlusion_cull(&self, encoder: &mut wgpu::CommandEncoder, frustum_passed_count: u32) {
        if let Some(bind_group) = &self.occlusion_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nanite Occlusion Cull Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.occlusion_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            // One thread per frustum-passed cluster
            let workgroups = (frustum_passed_count + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
    }

    /// Get the HZB texture view for external use.
    pub fn hzb_view(&self) -> &wgpu::TextureView {
        &self.hzb_view
    }

    /// Get HZB dimensions.
    pub fn dimensions(&self) -> (u32, u32, u32) {
        (self.width, self.height, self.mip_count)
    }
}
