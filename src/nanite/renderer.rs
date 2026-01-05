//! Nanite renderer implementation.

use super::cluster::{ClusterGpu, NaniteCullingUniform, NaniteInstanceGpu};
use super::culling::FrustumCuller;
use super::gpu_resources::{NaniteBindGroupLayouts, NaniteGpuResources};
use super::hzb::HzbGenerator;
use super::preprocessing::ClusterBuildResult;
use super::{NaniteConfig, NaniteStats};
use crate::math::Matrix4;

/// Nanite renderer for GPU-driven virtualized geometry.
pub struct NaniteRenderer {
    /// Configuration.
    config: NaniteConfig,
    /// Bind group layouts.
    layouts: NaniteBindGroupLayouts,
    /// GPU resources.
    resources: NaniteGpuResources,
    /// Visibility pass render pipeline.
    visibility_pipeline: wgpu::RenderPipeline,
    /// Material pass render pipeline.
    material_pipeline: wgpu::RenderPipeline,
    /// Shadow depth render pipeline.
    shadow_pipeline: wgpu::RenderPipeline,
    /// Light camera bind group layout for shadows.
    light_camera_layout: wgpu::BindGroupLayout,
    /// Frustum culler (compute shader).
    frustum_culler: FrustumCuller,
    /// Whether frustum culling is enabled.
    culling_enabled: bool,
    /// Whether occlusion culling is enabled (requires frustum culling).
    occlusion_culling_enabled: bool,
    /// HZB generator for occlusion culling.
    hzb_generator: HzbGenerator,
    /// Material bind group (visibility + depth textures).
    material_bind_group: Option<wgpu::BindGroup>,
    /// Texture bind group (materials + textures + sampler + shadows).
    texture_bind_group: Option<wgpu::BindGroup>,
    /// Registered meshes.
    meshes: Vec<NaniteMeshData>,
    /// Frame statistics.
    stats: NaniteStats,
}

/// Internal mesh data.
struct NaniteMeshData {
    /// First cluster index in global buffer.
    first_cluster: u32,
    /// Number of clusters.
    cluster_count: u32,
    /// First group index.
    #[allow(dead_code)]
    first_group: u32,
    /// Number of groups.
    #[allow(dead_code)]
    group_count: u32,
}

impl NaniteRenderer {
    /// Create a new Nanite renderer.
    pub fn new(
        device: &wgpu::Device,
        config: NaniteConfig,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let layouts = NaniteBindGroupLayouts::new(device);
        let resources = NaniteGpuResources::new(device, &config, &layouts, width, height);

        // Create frustum culler
        let frustum_culler = FrustumCuller::new(device);

        // Create HZB generator for occlusion culling
        let hzb_generator = HzbGenerator::new(device, width, height);

        // Load shaders
        let visibility_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite Visibility Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/nanite_visibility.wgsl").into()),
        });

        let material_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite Material Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/nanite_material.wgsl").into()),
        });

        // Camera bind group layout (shared with PBR)
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Visibility pass pipeline layout
        let visibility_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nanite Visibility Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &layouts.geometry,
                    &layouts.visibility,
                ],
                push_constant_ranges: &[],
            });

        // Visibility pass pipeline
        let visibility_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Nanite Visibility Pipeline"),
            layout: Some(&visibility_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &visibility_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &visibility_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None, // No blending for visibility IDs
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Material pass pipeline layout (4 bind groups max: camera, geometry, material, textures+shadows)
        let material_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nanite Material Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &layouts.geometry,
                    &layouts.material,
                    &layouts.textures, // Includes materials, textures, AND shadows (bindings 0-7)
                ],
                push_constant_ranges: &[],
            });

        // Material pass pipeline (fullscreen)
        let material_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Nanite Material Pipeline"),
            layout: Some(&material_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &material_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &material_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None, // Opaque, no blending needed
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Fullscreen triangle
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // Reading depth, not writing
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Shadow shader
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nanite Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/nanite_shadow.wgsl").into()),
        });

        // Light camera bind group layout (for shadow pass)
        let light_camera_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite Light Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Shadow pass pipeline layout
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nanite Shadow Pipeline Layout"),
                bind_group_layouts: &[
                    &light_camera_layout,
                    &layouts.geometry,
                    &layouts.visibility,
                ],
                push_constant_ranges: &[],
            });

        // Shadow pass pipeline (depth-only)
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Nanite Shadow Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shadow_shader,
                entry_point: Some("fs_main"),
                targets: &[], // No color output, depth-only
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            config,
            layouts,
            resources,
            visibility_pipeline,
            material_pipeline,
            shadow_pipeline,
            light_camera_layout,
            frustum_culler,
            culling_enabled: true,
            occlusion_culling_enabled: false, // Disabled - needs more debugging
            hzb_generator,
            material_bind_group: None,
            texture_bind_group: None,
            meshes: Vec::new(),
            stats: NaniteStats::default(),
        }
    }

    /// Enable or disable frustum culling.
    pub fn set_culling_enabled(&mut self, enabled: bool) {
        self.culling_enabled = enabled;
    }

    /// Check if culling is enabled.
    pub fn culling_enabled(&self) -> bool {
        self.culling_enabled
    }

    /// Enable or disable occlusion culling.
    pub fn set_occlusion_culling_enabled(&mut self, enabled: bool) {
        self.occlusion_culling_enabled = enabled;
    }

    /// Check if occlusion culling is enabled.
    pub fn occlusion_culling_enabled(&self) -> bool {
        self.occlusion_culling_enabled && self.culling_enabled
    }

    /// Register a mesh for Nanite rendering.
    ///
    /// Returns the mesh index for use with instances.
    pub fn register_mesh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, result: ClusterBuildResult) -> usize {
        let first_cluster = self.resources.cluster_count;
        let first_group = self.resources.cluster_count; // Groups match clusters 1:1 in Phase 1

        // Get current buffer offsets before uploading
        let vertex_offset_base = self.resources.vertex_count;
        let index_offset_base = self.resources.index_count;

        // Adjust cluster offsets to point to global buffer positions
        let adjusted_clusters: Vec<ClusterGpu> = result.clusters.iter().map(|c| {
            let mut cluster = *c;
            cluster.vertex_offset += vertex_offset_base;
            cluster.index_offset += index_offset_base;
            cluster
        }).collect();

        // Upload data (appends to global buffers)
        self.resources.upload_clusters(queue, &adjusted_clusters);
        self.resources.upload_positions(queue, &result.positions);
        self.resources.upload_attributes(queue, &result.attributes);
        self.resources.upload_indices(queue, &result.indices);

        let mesh_data = NaniteMeshData {
            first_cluster,
            cluster_count: result.clusters.len() as u32,
            first_group,
            group_count: result.groups.len() as u32,
        };

        let mesh_index = self.meshes.len();
        self.meshes.push(mesh_data);

        // Update frustum culler bind group for single-phase culling
        // (outputs directly to visible_clusters_buffer)
        self.frustum_culler.update_bind_group(
            device,
            &self.resources.cluster_buffer,
            &self.resources.instance_buffer,
            &self.resources.visible_clusters_buffer,
            &self.resources.visible_counter_buffer,
            &self.resources.indirect_buffer,
        );

        mesh_index
    }

    /// Update instance transforms.
    pub fn update_instances(&mut self, queue: &wgpu::Queue, instances: &[(usize, Matrix4)]) {
        let gpu_instances: Vec<NaniteInstanceGpu> = instances
            .iter()
            .map(|(mesh_idx, transform)| {
                let mesh = &self.meshes[*mesh_idx];
                NaniteInstanceGpu {
                    transform: transform.to_cols_array_2d(),
                    first_cluster: mesh.first_cluster,
                    cluster_count: mesh.cluster_count,
                    first_group: mesh.first_cluster,
                    group_count: mesh.cluster_count,
                }
            })
            .collect();

        self.resources.upload_instances(queue, &gpu_instances);
    }

    /// Create material bind group with depth texture.
    pub fn create_material_bind_group(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
    ) {
        self.material_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Material Bind Group"),
            layout: &self.layouts.material,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.resources.visibility_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        }));
    }

    /// Update culling uniforms.
    pub fn update_culling(
        &self,
        queue: &wgpu::Queue,
        view_proj: &Matrix4,
        camera_pos: [f32; 3],
        screen_height: f32,
        fov_y: f32,
    ) {
        let uniform = NaniteCullingUniform {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(
            &self.resources.culling_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform]),
        );

        // Also update frustum culler uniform
        self.frustum_culler.update_uniform(
            queue,
            view_proj,
            self.resources.cluster_count,
            self.resources.instance_count.max(1),
            screen_height,
            fov_y,
        );

        // Update occlusion culling uniform
        if self.occlusion_culling_enabled {
            self.hzb_generator.update_occlusion_uniform(
                queue,
                &view_proj.to_cols_array_2d(),
                camera_pos,
                self.resources.cluster_count,
                self.resources.instance_count.max(1),
                screen_height,
                fov_y,
            );
        }
    }

    /// Run frustum culling compute pass.
    ///
    /// This should be called before render_visibility or cull_occlusion.
    pub fn cull_frustum(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.resources.cluster_count == 0 || !self.culling_enabled {
            return;
        }

        self.frustum_culler.dispatch(encoder, self.resources.cluster_count);
    }

    /// Run occlusion culling compute pass.
    ///
    /// This should be called after cull_frustum and before render_visibility.
    /// Uses the HZB from the previous frame for 1-frame latency occlusion testing.
    pub fn cull_occlusion(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.resources.cluster_count == 0 || !self.occlusion_culling_enabled() {
            return;
        }

        // Dispatch occlusion culling using frustum-passed clusters
        // We pass max_clusters as the upper bound; the shader reads actual count from buffer
        self.hzb_generator.dispatch_occlusion_cull(encoder, self.resources.cluster_count);
    }

    /// Build the HZB mip chain from the current depth buffer.
    ///
    /// This should be called after render_visibility to prepare for next frame's occlusion culling.
    pub fn build_hzb(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, src_width: u32, src_height: u32) {
        if !self.occlusion_culling_enabled() {
            return;
        }

        self.hzb_generator.build_hzb(encoder, queue, src_width, src_height);
    }

    /// Create HZB build bind groups using the depth view.
    ///
    /// This must be called after the depth texture is created and whenever it's recreated.
    pub fn create_hzb_build_bind_groups(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        self.hzb_generator.create_build_bind_groups(device, depth_view);
    }

    /// Resize visibility buffer and HZB.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.resources.resize(device, width, height);
        self.hzb_generator.resize(device, width, height);
        // Material bind group needs to be recreated after resize
        self.material_bind_group = None;
    }

    /// Render visibility pass.
    ///
    /// This pass writes cluster/triangle IDs to the visibility buffer.
    pub fn render_visibility<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.resources.cluster_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.visibility_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.resources.geometry_bind_group, &[]);
        render_pass.set_bind_group(2, &self.resources.visibility_bind_group, &[]);

        if self.culling_enabled {
            // Use indirect draw with GPU-culled results
            render_pass.draw_indirect(&self.resources.indirect_buffer, 0);
        } else {
            // Fallback: render all clusters directly (Phase 1 behavior)
            let total_triangles: u32 = self.meshes.iter().map(|m| {
                m.cluster_count * self.config.triangles_per_cluster
            }).sum();

            render_pass.draw(0..total_triangles * 3, 0..1);
        }
    }

    /// Render material pass.
    ///
    /// This fullscreen pass reads the visibility buffer and shades visible pixels.
    /// texture_bind_group must include shadow resources (bindings 3-7).
    pub fn render_material<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.resources.cluster_count == 0
            || self.material_bind_group.is_none()
            || self.texture_bind_group.is_none()
        {
            return;
        }

        render_pass.set_pipeline(&self.material_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.resources.geometry_bind_group, &[]);
        render_pass.set_bind_group(2, self.material_bind_group.as_ref().unwrap(), &[]);
        render_pass.set_bind_group(3, self.texture_bind_group.as_ref().unwrap(), &[]); // Includes shadows

        // Fullscreen triangle
        render_pass.draw(0..3, 0..1);
    }

    /// Get visibility texture view for render pass.
    pub fn visibility_view(&self) -> &wgpu::TextureView {
        &self.resources.visibility_view
    }

    /// Get current statistics.
    pub fn stats(&self) -> &NaniteStats {
        &self.stats
    }

    /// Get cluster count.
    pub fn cluster_count(&self) -> u32 {
        self.resources.cluster_count
    }

    /// Get mesh count.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Get total triangle count across all clusters.
    pub fn triangle_count(&self) -> u32 {
        // Each cluster has ~128 triangles by default
        self.resources.cluster_count * self.config.triangles_per_cluster
    }

    /// Clear all registered meshes.
    pub fn clear_meshes(&mut self) {
        self.meshes.clear();
        self.resources.cluster_count = 0;
        self.resources.vertex_count = 0;
        self.resources.index_count = 0;
    }

    /// Get geometry bind group.
    pub fn geometry_bind_group(&self) -> &wgpu::BindGroup {
        &self.resources.geometry_bind_group
    }

    /// Get indirect buffer for reading back stats.
    pub fn indirect_buffer(&self) -> &wgpu::Buffer {
        &self.resources.indirect_buffer
    }

    /// Reset for new frame.
    pub fn begin_frame(&mut self, queue: &wgpu::Queue) {
        // Reset visible counter for single-phase frustum culling
        self.resources.reset_culling(queue);
        self.stats = NaniteStats {
            total_clusters: self.resources.cluster_count,
            ..Default::default()
        };
    }

    /// Upload materials and create texture array.
    /// Note: After calling this, you must call set_texture_bind_group with a bind group
    /// that includes the materials, textures, AND shadow resources.
    pub fn upload_materials_and_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        materials: &[super::gpu_resources::NaniteMaterialGpu],
        textures: &[(u32, u32, &[u8])], // (width, height, rgba_data)
    ) {
        // Upload materials
        self.resources.upload_materials(queue, materials);

        // Create texture array (resize all to 2048x2048 for quality)
        if !textures.is_empty() {
            self.resources.create_texture_array(device, queue, textures, 2048);
        } else {
            // Create fallback checkerboard texture array (Unreal-style)
            self.create_fallback_texture_array(device, queue);
        }
    }

    /// Create a fallback checkerboard texture array (Unreal-style beveled tile pattern).
    fn create_fallback_texture_array(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.resources.texture_array_view().is_some() {
            return;
        }

        // Create Unreal-style beveled floor tile pattern with rounded corners
        // 2x2 grid of tiles, each tile is one checker with rounded beveled edges
        let size = 256u32;
        let tile_size = size / 2;    // 2x2 tiles
        let grout_width = 3u32;      // Dark gap between tiles
        let bevel_width = 6u32;      // Beveled/rounded edge
        let corner_radius = 12u32;   // Rounded corner radius

        let mut pixels = vec![0u8; (size * size * 4) as usize];

        // Warm tan/beige colors like Unreal's floor
        let grout = [90u8, 85, 80, 255];           // Dark grout/gap
        let tile_light = [210u8, 200, 185, 255];  // Light tan tile
        let tile_dark = [180u8, 170, 155, 255];   // Dark tan tile

        // Simple hash-based noise function
        let noise = |x: u32, y: u32| -> f32 {
            let n = x.wrapping_mul(374761393).wrapping_add(y.wrapping_mul(668265263));
            let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
            let n = n ^ (n >> 16);
            (n as f32 / u32::MAX as f32) * 2.0 - 1.0 // Returns -1.0 to 1.0
        };
        let noise_strength = 8.0f32; // Subtle noise amount

        for y in 0..size {
            for x in 0..size {
                let idx = ((y * size + x) * 4) as usize;

                // Which tile are we in (0 or 1 in each axis)
                let tile_x = x / tile_size;
                let tile_y = y / tile_size;
                let is_light_tile = (tile_x + tile_y) % 2 == 0;

                // Position within the tile
                let local_x = x % tile_size;
                let local_y = y % tile_size;

                // Distance from tile edges
                let dist_left = local_x;
                let dist_right = tile_size - 1 - local_x;
                let dist_top = local_y;
                let dist_bottom = tile_size - 1 - local_y;

                // Calculate distance from edge with rounded corners (floating point for smooth AA)
                let corner_r = corner_radius as f32;
                let dl = dist_left as f32;
                let dr = dist_right as f32;
                let dt = dist_top as f32;
                let db = dist_bottom as f32;

                let dist_edge = if dl < corner_r && dt < corner_r {
                    let cx = corner_r - dl;
                    let cy = corner_r - dt;
                    let d = (cx * cx + cy * cy).sqrt();
                    (corner_r - d).max(0.0)
                } else if dr < corner_r && dt < corner_r {
                    let cx = corner_r - dr;
                    let cy = corner_r - dt;
                    let d = (cx * cx + cy * cy).sqrt();
                    (corner_r - d).max(0.0)
                } else if dl < corner_r && db < corner_r {
                    let cx = corner_r - dl;
                    let cy = corner_r - db;
                    let d = (cx * cx + cy * cy).sqrt();
                    (corner_r - d).max(0.0)
                } else if dr < corner_r && db < corner_r {
                    let cx = corner_r - dr;
                    let cy = corner_r - db;
                    let d = (cx * cx + cy * cy).sqrt();
                    (corner_r - d).max(0.0)
                } else {
                    dl.min(dr).min(dt).min(db)
                };

                // Base tile color
                let base_color = if is_light_tile { &tile_light } else { &tile_dark };

                let grout_w = grout_width as f32;
                let bevel_w = bevel_width as f32;
                let aa_width = 1.5f32; // Anti-aliasing width in pixels

                // Smooth blending helper
                let smoothstep = |edge0: f32, edge1: f32, x: f32| -> f32 {
                    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                };

                let lerp_color = |c1: &[u8; 4], c2: &[u8; 4], t: f32| -> [u8; 4] {
                    [
                        (c1[0] as f32 + (c2[0] as f32 - c1[0] as f32) * t) as u8,
                        (c1[1] as f32 + (c2[1] as f32 - c1[1] as f32) * t) as u8,
                        (c1[2] as f32 + (c2[2] as f32 - c1[2] as f32) * t) as u8,
                        255,
                    ]
                };

                // Determine which edges we're near for bevel lighting
                let near_top = dt < db;
                let near_left = dl < dr;
                let light_adjust = if near_top || near_left { 20.0 } else { -25.0 };

                // Calculate beveled tile color at current position
                let bevel_factor = ((dist_edge - grout_w) / bevel_w).clamp(0.0, 1.0);
                let adjust = light_adjust * (1.0 - bevel_factor);
                let bevel_color: [u8; 4] = [
                    (base_color[0] as f32 + adjust).clamp(0.0, 255.0) as u8,
                    (base_color[1] as f32 + adjust).clamp(0.0, 255.0) as u8,
                    (base_color[2] as f32 + adjust).clamp(0.0, 255.0) as u8,
                    255,
                ];

                let color: [u8; 4] = if dist_edge < grout_w - aa_width {
                    // Pure grout
                    grout
                } else if dist_edge < grout_w + aa_width {
                    // Smooth transition from grout to bevel
                    let t = smoothstep(grout_w - aa_width, grout_w + aa_width, dist_edge);
                    lerp_color(&grout, &bevel_color, t)
                } else if dist_edge < grout_w + bevel_w {
                    // Bevel region (already smooth via bevel_factor)
                    bevel_color
                } else {
                    // Tile interior
                    *base_color
                };

                // Apply subtle noise to the color
                let n = noise(x, y) * noise_strength;
                let final_color = [
                    (color[0] as f32 + n).clamp(0.0, 255.0) as u8,
                    (color[1] as f32 + n).clamp(0.0, 255.0) as u8,
                    (color[2] as f32 + n).clamp(0.0, 255.0) as u8,
                    255,
                ];

                pixels[idx..idx + 4].copy_from_slice(&final_color);
            }
        }

        let fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Nanite Floor Tile Texture"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );

        let view = fallback_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Store in resources
        self.resources.texture_array = Some(fallback_texture);
        self.resources.texture_array_view = Some(view);
        self.resources.texture_count = 1;
    }

    /// Check if textures are loaded.
    pub fn has_textures(&self) -> bool {
        self.resources.texture_array_view().is_some()
    }

    /// Get material buffer for bind group creation.
    pub fn material_buffer(&self) -> &wgpu::Buffer {
        self.resources.material_buffer()
    }

    /// Get texture array view for bind group creation.
    pub fn texture_array_view(&self) -> Option<&wgpu::TextureView> {
        self.resources.texture_array_view()
    }

    /// Get texture sampler for bind group creation.
    pub fn texture_sampler(&self) -> &wgpu::Sampler {
        self.resources.texture_sampler()
    }

    /// Create a light camera bind group for shadow rendering.
    pub fn create_light_camera_bind_group(
        &self,
        device: &wgpu::Device,
        light_matrix_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Light Camera Bind Group"),
            layout: &self.light_camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_matrix_buffer.as_entire_binding(),
            }],
        })
    }

    /// Render Nanite geometry into a shadow map.
    ///
    /// This should be called during the shadow pass for each shadow-casting light.
    /// The render pass should already be configured with the shadow map as depth attachment.
    pub fn render_shadow<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        light_camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.resources.cluster_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.shadow_pipeline);
        render_pass.set_bind_group(0, light_camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.resources.geometry_bind_group, &[]);
        render_pass.set_bind_group(2, &self.resources.visibility_bind_group, &[]);

        if self.culling_enabled {
            // Use indirect draw with GPU-culled results
            render_pass.draw_indirect(&self.resources.indirect_buffer, 0);
        } else {
            // Fallback: render all clusters directly
            let total_triangles: u32 = self.meshes.iter().map(|m| {
                m.cluster_count * self.config.triangles_per_cluster
            }).sum();

            render_pass.draw(0..total_triangles * 3, 0..1);
        }
    }

    /// Set texture bind group (includes materials, textures, and shadows).
    /// This should be called by the web renderer after creating the combined bind group.
    pub fn set_texture_bind_group(&mut self, texture_bind_group: wgpu::BindGroup) {
        self.texture_bind_group = Some(texture_bind_group);
    }

    /// Get the light camera bind group layout.
    pub fn light_camera_layout(&self) -> &wgpu::BindGroupLayout {
        &self.light_camera_layout
    }

    /// Get visibility bind group for shadow pass.
    pub fn visibility_bind_group(&self) -> &wgpu::BindGroup {
        &self.resources.visibility_bind_group
    }

    /// Get the textures bind group layout (includes shadow bindings).
    pub fn textures_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layouts.textures
    }
}
