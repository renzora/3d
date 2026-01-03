//! GPU buffer management for Nanite rendering.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::cluster::{ClusterGpu, ClusterGroupGpu, NaniteInstanceGpu, NaniteVertex, NaniteVertexAttribute};
use super::NaniteConfig;

/// Visible cluster entry output from culling (16 bytes).
/// Matches the shader's VisibleCluster struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct VisibleClusterGpu {
    /// Cluster index in the global cluster buffer.
    pub cluster_id: u32,
    /// Instance index this cluster belongs to.
    pub instance_id: u32,
    /// Running sum of triangles for vertex indexing.
    pub triangle_offset: u32,
    /// Padding.
    pub _pad: u32,
}

/// Material data for GPU (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NaniteMaterialGpu {
    /// Base color RGBA.
    pub base_color: [f32; 4],
    /// Index into texture array (-1 = no texture).
    pub texture_index: i32,
    /// Metallic factor.
    pub metallic: f32,
    /// Roughness factor.
    pub roughness: f32,
    /// Padding.
    pub _pad: f32,
}

impl Default for NaniteMaterialGpu {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            texture_index: -1,
            metallic: 0.0,
            roughness: 0.5,
            _pad: 0.0,
        }
    }
}

/// GPU resources for Nanite rendering.
pub struct NaniteGpuResources {
    // Geometry buffers
    /// Storage buffer containing all clusters.
    pub cluster_buffer: wgpu::Buffer,
    /// Storage buffer containing cluster groups (for LOD hierarchy).
    pub group_buffer: wgpu::Buffer,
    /// Storage buffer containing vertex positions.
    pub position_buffer: wgpu::Buffer,
    /// Storage buffer containing vertex attributes (normals, UVs, etc.).
    pub attribute_buffer: wgpu::Buffer,
    /// Storage buffer containing triangle indices.
    pub index_buffer: wgpu::Buffer,

    // Instance data
    /// Storage buffer containing instance transforms.
    pub instance_buffer: wgpu::Buffer,

    // Culling output buffers
    /// Storage buffer for visible cluster indices (output from culling).
    pub visible_clusters_buffer: wgpu::Buffer,
    /// Atomic counter for visible clusters.
    pub visible_counter_buffer: wgpu::Buffer,
    /// Indirect draw arguments (set by culling compute shader).
    pub indirect_buffer: wgpu::Buffer,

    // Two-phase culling intermediate buffers (for occlusion culling)
    /// Storage buffer for frustum-passed clusters (intermediate from frustum -> occlusion cull).
    pub frustum_clusters_buffer: wgpu::Buffer,
    /// Counter for frustum-passed clusters.
    pub frustum_counter_buffer: wgpu::Buffer,

    // Visibility buffer
    /// Visibility texture (R32Uint) storing packed triangle/cluster/instance IDs.
    pub visibility_texture: wgpu::Texture,
    /// View for the visibility texture.
    pub visibility_view: wgpu::TextureView,

    // Material/Texture resources
    /// Storage buffer containing material data.
    pub material_buffer: wgpu::Buffer,
    /// Texture array for base color textures (one layer per texture).
    pub texture_array: Option<wgpu::Texture>,
    /// View for the texture array.
    pub texture_array_view: Option<wgpu::TextureView>,
    /// Sampler for textures.
    pub texture_sampler: wgpu::Sampler,

    // Uniforms
    /// Culling uniforms (view-proj matrix, etc.).
    pub culling_uniform_buffer: wgpu::Buffer,

    // Bind groups
    /// Bind group for geometry data.
    pub geometry_bind_group: wgpu::BindGroup,
    /// Bind group for culling compute shader.
    pub culling_bind_group: wgpu::BindGroup,
    /// Bind group for visibility pass.
    pub visibility_bind_group: wgpu::BindGroup,

    // Configuration
    /// Maximum number of clusters.
    pub max_clusters: u32,
    /// Maximum number of vertices.
    pub max_vertices: u32,
    /// Maximum number of indices.
    pub max_indices: u32,
    /// Maximum number of instances.
    pub max_instances: u32,

    // Current counts
    /// Current number of clusters.
    pub cluster_count: u32,
    /// Current number of vertices.
    pub vertex_count: u32,
    /// Current number of indices.
    pub index_count: u32,
    /// Current number of instances.
    pub instance_count: u32,
    /// Current number of materials.
    pub material_count: u32,
    /// Current number of textures in the array.
    pub texture_count: u32,
}

/// Bind group layouts for Nanite rendering.
pub struct NaniteBindGroupLayouts {
    /// Layout for geometry data (clusters, vertices, indices).
    pub geometry: wgpu::BindGroupLayout,
    /// Layout for culling compute shader.
    pub culling: wgpu::BindGroupLayout,
    /// Layout for visibility pass.
    pub visibility: wgpu::BindGroupLayout,
    /// Layout for material pass (visibility + depth textures).
    pub material: wgpu::BindGroupLayout,
    /// Layout for textures and shadows (group 3: materials, textures, and shadow data).
    pub textures: wgpu::BindGroupLayout,
}

impl NaniteBindGroupLayouts {
    /// Create bind group layouts.
    pub fn new(device: &wgpu::Device) -> Self {
        // Geometry bind group layout
        // All entries need FRAGMENT visibility for material pass
        let geometry = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nanite Geometry Bind Group Layout"),
            entries: &[
                // Clusters
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cluster groups
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Positions
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Attributes
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Indices
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Instances
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Culling bind group layout
        let culling = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nanite Culling Bind Group Layout"),
            entries: &[
                // Culling uniforms
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
                // Visible clusters output
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Visible counter
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Indirect draw args
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

        // Visibility bind group layout
        let visibility = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nanite Visibility Bind Group Layout"),
            entries: &[
                // Visible clusters (input from culling)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Visible counter (to know how many visible clusters)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Material bind group layout
        let material = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nanite Material Bind Group Layout"),
            entries: &[
                // Visibility texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth texture (for reconstructing position)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Textures + Shadows bind group layout (merged into group 3)
        let textures = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nanite Textures and Shadows Bind Group Layout"),
            entries: &[
                // Binding 0: Materials storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Texture array
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: Texture sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 3: Shadow uniform data
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: 2D shadow map (for directional/spot lights)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 5: Comparison sampler for 2D shadow map
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // Binding 6: Cube shadow map (for point lights)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 7: Comparison sampler for cube shadow map
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        Self {
            geometry,
            culling,
            visibility,
            material,
            textures,
        }
    }
}

/// Indirect draw arguments for Nanite rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NaniteIndirectArgs {
    /// Number of vertices per instance (3 * triangles for non-indexed).
    pub vertex_count: u32,
    /// Number of instances (visible clusters).
    pub instance_count: u32,
    /// First vertex.
    pub first_vertex: u32,
    /// First instance.
    pub first_instance: u32,
}

impl Default for NaniteIndirectArgs {
    fn default() -> Self {
        Self {
            vertex_count: 0,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        }
    }
}

/// Counter data for culling.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NaniteCounterData {
    /// Number of visible clusters.
    pub visible_count: u32,
    /// Padding.
    pub _pad: [u32; 3],
}

impl Default for NaniteCounterData {
    fn default() -> Self {
        Self {
            visible_count: 0,
            _pad: [0; 3],
        }
    }
}

impl NaniteGpuResources {
    /// Create GPU resources for Nanite rendering.
    pub fn new(
        device: &wgpu::Device,
        config: &NaniteConfig,
        layouts: &NaniteBindGroupLayouts,
        width: u32,
        height: u32,
    ) -> Self {
        let max_clusters = config.max_clusters;
        let max_vertices = config.max_vertices;
        let max_indices = config.max_indices;
        let max_instances = config.max_instances;

        // Create geometry buffers
        let cluster_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Cluster Buffer"),
            size: (max_clusters as usize * std::mem::size_of::<ClusterGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let group_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Group Buffer"),
            size: (max_clusters as usize * std::mem::size_of::<ClusterGroupGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Position Buffer"),
            size: (max_vertices as usize * std::mem::size_of::<NaniteVertex>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let attribute_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Attribute Buffer"),
            size: (max_vertices as usize * std::mem::size_of::<NaniteVertexAttribute>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Index Buffer"),
            size: (max_indices as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nanite Instance Buffer"),
            contents: bytemuck::cast_slice(&vec![NaniteInstanceGpu::default(); max_instances as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Culling output buffers
        let visible_clusters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Visible Clusters Buffer"),
            size: (max_clusters as usize * std::mem::size_of::<VisibleClusterGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nanite Visible Counter Buffer"),
            contents: bytemuck::cast_slice(&[NaniteCounterData::default()]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nanite Indirect Buffer"),
            contents: bytemuck::cast_slice(&[NaniteIndirectArgs::default()]),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Two-phase culling intermediate buffers
        let frustum_clusters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nanite Frustum Clusters Buffer"),
            size: (max_clusters as usize * std::mem::size_of::<VisibleClusterGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frustum_counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nanite Frustum Counter Buffer"),
            contents: bytemuck::cast_slice(&[NaniteCounterData::default()]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Visibility texture
        let visibility_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Nanite Visibility Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let visibility_view = visibility_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Material buffer (max 256 materials)
        let max_materials = 256u32;
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nanite Material Buffer"),
            contents: bytemuck::cast_slice(&vec![NaniteMaterialGpu::default(); max_materials as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Texture sampler with anisotropic filtering for quality
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Nanite Texture Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16, // Max anisotropic filtering
            ..Default::default()
        });

        // Culling uniform buffer
        let culling_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nanite Culling Uniform Buffer"),
            contents: bytemuck::cast_slice(&[super::cluster::NaniteCullingUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind groups
        let geometry_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Geometry Bind Group"),
            layout: &layouts.geometry,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cluster_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: group_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: attribute_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });

        let culling_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Culling Bind Group"),
            layout: &layouts.culling,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: culling_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: visible_clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: visible_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: indirect_buffer.as_entire_binding(),
                },
            ],
        });

        let visibility_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nanite Visibility Bind Group"),
            layout: &layouts.visibility,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visible_clusters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: visible_counter_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            cluster_buffer,
            group_buffer,
            position_buffer,
            attribute_buffer,
            index_buffer,
            instance_buffer,
            visible_clusters_buffer,
            visible_counter_buffer,
            indirect_buffer,
            frustum_clusters_buffer,
            frustum_counter_buffer,
            visibility_texture,
            visibility_view,
            material_buffer,
            texture_array: None,
            texture_array_view: None,
            texture_sampler,
            culling_uniform_buffer,
            geometry_bind_group,
            culling_bind_group,
            visibility_bind_group,
            max_clusters,
            max_vertices,
            max_indices,
            max_instances,
            cluster_count: 0,
            vertex_count: 0,
            index_count: 0,
            instance_count: 0,
            material_count: 0,
            texture_count: 0,
        }
    }

    /// Resize visibility texture to match new dimensions.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.visibility_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Nanite Visibility Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        self.visibility_view = self.visibility_texture.create_view(&wgpu::TextureViewDescriptor::default());
    }

    /// Reset visible cluster count for new frame.
    pub fn reset_culling(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.visible_counter_buffer,
            0,
            bytemuck::cast_slice(&[NaniteCounterData::default()]),
        );
        queue.write_buffer(
            &self.indirect_buffer,
            0,
            bytemuck::cast_slice(&[NaniteIndirectArgs::default()]),
        );
    }

    /// Reset frustum culling buffers for two-phase culling.
    pub fn reset_frustum_culling(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.frustum_counter_buffer,
            0,
            bytemuck::cast_slice(&[NaniteCounterData::default()]),
        );
    }

    /// Reset all culling buffers for two-phase culling.
    pub fn reset_two_phase_culling(&self, queue: &wgpu::Queue) {
        self.reset_frustum_culling(queue);
        self.reset_culling(queue);
    }

    /// Upload cluster data (appends to existing data).
    pub fn upload_clusters(&mut self, queue: &wgpu::Queue, clusters: &[ClusterGpu]) {
        if clusters.is_empty() {
            return;
        }
        let offset = self.cluster_count as u64 * std::mem::size_of::<ClusterGpu>() as u64;
        queue.write_buffer(&self.cluster_buffer, offset, bytemuck::cast_slice(clusters));
        self.cluster_count += clusters.len() as u32;
    }

    /// Upload vertex positions (appends to existing data).
    pub fn upload_positions(&mut self, queue: &wgpu::Queue, positions: &[NaniteVertex]) {
        if positions.is_empty() {
            return;
        }
        let offset = self.vertex_count as u64 * std::mem::size_of::<NaniteVertex>() as u64;
        queue.write_buffer(&self.position_buffer, offset, bytemuck::cast_slice(positions));
        self.vertex_count += positions.len() as u32;
    }

    /// Upload vertex attributes (appends to existing data).
    pub fn upload_attributes(&mut self, queue: &wgpu::Queue, attributes: &[NaniteVertexAttribute]) {
        if attributes.is_empty() {
            return;
        }
        // Use vertex_count as the attribute count (they're 1:1 with positions)
        let offset = (self.vertex_count - attributes.len() as u32) as u64
            * std::mem::size_of::<NaniteVertexAttribute>() as u64;
        queue.write_buffer(&self.attribute_buffer, offset, bytemuck::cast_slice(attributes));
    }

    /// Upload indices (appends to existing data).
    pub fn upload_indices(&mut self, queue: &wgpu::Queue, indices: &[u32]) {
        if indices.is_empty() {
            return;
        }
        let offset = self.index_count as u64 * std::mem::size_of::<u32>() as u64;
        queue.write_buffer(&self.index_buffer, offset, bytemuck::cast_slice(indices));
        self.index_count += indices.len() as u32;
    }

    /// Upload instance transforms.
    pub fn upload_instances(&mut self, queue: &wgpu::Queue, instances: &[NaniteInstanceGpu]) {
        if instances.is_empty() {
            return;
        }
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
        self.instance_count = instances.len() as u32;
    }

    /// Upload materials.
    pub fn upload_materials(&mut self, queue: &wgpu::Queue, materials: &[NaniteMaterialGpu]) {
        if materials.is_empty() {
            return;
        }
        queue.write_buffer(&self.material_buffer, 0, bytemuck::cast_slice(materials));
        self.material_count = materials.len() as u32;
    }

    /// Create texture array from a list of RGBA textures.
    /// Uses the maximum texture size found (up to 4096) to preserve quality.
    pub fn create_texture_array(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &[(u32, u32, &[u8])], // (width, height, rgba_data)
        _min_size: u32, // Ignored, we use max of source sizes
    ) {
        if textures.is_empty() {
            return;
        }

        let layer_count = textures.len() as u32;

        // Find maximum texture dimension, capped at 2048 for fast loading
        // (2048 is still high quality and loads 4x faster than 4096)
        let max_size = textures
            .iter()
            .map(|(w, h, _)| (*w).max(*h))
            .max()
            .unwrap_or(1024)
            .min(2048);

        // Round up to power of 2 for better GPU performance
        let target_size = max_size.next_power_of_two().min(2048);

        // Calculate mip levels
        let mip_count = (target_size as f32).log2().floor() as u32 + 1;

        log::info!("Nanite: Creating texture array {}x{} with {} mip levels for {} textures",
            target_size, target_size, mip_count, layer_count);

        // Create texture array with mipmaps
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Nanite Texture Array"),
            size: wgpu::Extent3d {
                width: target_size,
                height: target_size,
                depth_or_array_layers: layer_count,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload each texture layer with mipmaps
        for (layer_idx, (src_width, src_height, src_data)) in textures.iter().enumerate() {
            // Resize to target size (base mip level)
            let base_mip = resize_texture(*src_width, *src_height, src_data, target_size);

            // Upload base mip level
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer_idx as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &base_mip,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(target_size * 4),
                    rows_per_image: Some(target_size),
                },
                wgpu::Extent3d {
                    width: target_size,
                    height: target_size,
                    depth_or_array_layers: 1,
                },
            );

            // Generate and upload mip levels
            let mut current_data = base_mip;
            let mut current_size = target_size;

            for mip_level in 1..mip_count {
                let new_size = (current_size / 2).max(1);
                let mip_data = generate_mip_level(&current_data, current_size, new_size);

                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer_idx as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &mip_data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(new_size * 4),
                        rows_per_image: Some(new_size),
                    },
                    wgpu::Extent3d {
                        width: new_size,
                        height: new_size,
                        depth_or_array_layers: 1,
                    },
                );

                current_data = mip_data;
                current_size = new_size;
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Nanite Texture Array View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        self.texture_array = Some(texture);
        self.texture_array_view = Some(view);
        self.texture_count = layer_count;
    }

    /// Get texture array view (if created).
    pub fn texture_array_view(&self) -> Option<&wgpu::TextureView> {
        self.texture_array_view.as_ref()
    }

    /// Get material buffer.
    pub fn material_buffer(&self) -> &wgpu::Buffer {
        &self.material_buffer
    }

    /// Get texture sampler.
    pub fn texture_sampler(&self) -> &wgpu::Sampler {
        &self.texture_sampler
    }
}

/// Pre-computed sRGB to linear lookup table (256 entries)
/// Generated at runtime on first use for fast conversion
static SRGB_TO_LINEAR_LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut lut = [0.0f32; 256];
    for i in 0..256 {
        let value = i as f32 / 255.0;
        lut[i] = if value <= 0.04045 {
            value / 12.92
        } else {
            // Compute ((value + 0.055) / 1.055)^2.4
            let base = (value + 0.055) / 1.055;
            base.powf(2.4)
        };
    }
    lut
});

/// Fast sRGB to linear using lookup table
#[inline]
fn srgb_to_linear_fast(value: u8) -> f32 {
    SRGB_TO_LINEAR_LUT[value as usize]
}

/// Fast linear to sRGB using approximation
/// Uses a fast approximation that's accurate enough for texture processing
#[inline]
fn linear_to_srgb_fast(value: f32) -> u8 {
    if value <= 0.0031308 {
        (value * 12.92 * 255.0).round().clamp(0.0, 255.0) as u8
    } else {
        // Approximate pow(x, 1/2.4) ≈ sqrt(x) * pow(x, 0.0833)
        // Using faster approximation: 1.055 * x^0.4167 - 0.055
        // x^0.4167 ≈ sqrt(sqrt(x)) * sqrt(sqrt(sqrt(x)))
        let sqrt_x = value.sqrt();
        let x_0416 = sqrt_x * sqrt_x.sqrt().sqrt();
        let srgb = 1.055 * x_0416 - 0.055;
        (srgb * 255.0).round().clamp(0.0, 255.0) as u8
    }
}

/// Generate a mip level by box filtering (2x2 average) with sRGB-correct blending.
fn generate_mip_level(src_data: &[u8], src_size: u32, dst_size: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_size * dst_size * 4) as usize];

    for dst_y in 0..dst_size {
        for dst_x in 0..dst_size {
            let src_x = dst_x * 2;
            let src_y = dst_y * 2;

            // Get 4 source pixels (2x2 block)
            let idx00 = ((src_y * src_size + src_x) * 4) as usize;
            let idx10 = ((src_y * src_size + (src_x + 1).min(src_size - 1)) * 4) as usize;
            let idx01 = (((src_y + 1).min(src_size - 1) * src_size + src_x) * 4) as usize;
            let idx11 = (((src_y + 1).min(src_size - 1) * src_size + (src_x + 1).min(src_size - 1)) * 4) as usize;

            let dst_idx = ((dst_y * dst_size + dst_x) * 4) as usize;

            // Average RGB in linear space using fast LUT
            for c in 0..3 {
                let p00 = srgb_to_linear_fast(src_data.get(idx00 + c).copied().unwrap_or(255));
                let p10 = srgb_to_linear_fast(src_data.get(idx10 + c).copied().unwrap_or(255));
                let p01 = srgb_to_linear_fast(src_data.get(idx01 + c).copied().unwrap_or(255));
                let p11 = srgb_to_linear_fast(src_data.get(idx11 + c).copied().unwrap_or(255));

                let avg = (p00 + p10 + p01 + p11) * 0.25;
                dst[dst_idx + c] = linear_to_srgb_fast(avg);
            }

            // Average alpha directly
            let a00 = src_data.get(idx00 + 3).copied().unwrap_or(255) as u32;
            let a10 = src_data.get(idx10 + 3).copied().unwrap_or(255) as u32;
            let a01 = src_data.get(idx01 + 3).copied().unwrap_or(255) as u32;
            let a11 = src_data.get(idx11 + 3).copied().unwrap_or(255) as u32;
            dst[dst_idx + 3] = ((a00 + a10 + a01 + a11) / 4) as u8;
        }
    }

    dst
}

/// Bilinear resize of RGBA texture with proper sRGB handling.
/// Stretches texture to fill target size (preserves UV mapping).
/// Interpolation is done in linear space to avoid color shifts.
fn resize_texture(src_width: u32, src_height: u32, src_data: &[u8], target_size: u32) -> Vec<u8> {
    if src_width == target_size && src_height == target_size {
        return src_data.to_vec();
    }

    let mut dst = vec![0u8; (target_size * target_size * 4) as usize];

    for dst_y in 0..target_size {
        for dst_x in 0..target_size {
            // Map destination to source coordinates (stretching to fill)
            let src_x_f = ((dst_x as f32 + 0.5) / target_size as f32) * src_width as f32 - 0.5;
            let src_y_f = ((dst_y as f32 + 0.5) / target_size as f32) * src_height as f32 - 0.5;

            // Clamp to valid range
            let src_x_f = src_x_f.max(0.0).min((src_width - 1) as f32);
            let src_y_f = src_y_f.max(0.0).min((src_height - 1) as f32);

            // Get integer and fractional parts
            let x0 = src_x_f.floor() as u32;
            let y0 = src_y_f.floor() as u32;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);
            let fx = src_x_f - x0 as f32;
            let fy = src_y_f - y0 as f32;

            // Get the four neighboring pixels
            let idx00 = ((y0 * src_width + x0) * 4) as usize;
            let idx10 = ((y0 * src_width + x1) * 4) as usize;
            let idx01 = ((y1 * src_width + x0) * 4) as usize;
            let idx11 = ((y1 * src_width + x1) * 4) as usize;

            let dst_idx = ((dst_y * target_size + dst_x) * 4) as usize;

            // Bilinear interpolation for RGB channels (in linear space using fast LUT)
            for c in 0..3 {
                let p00 = srgb_to_linear_fast(src_data.get(idx00 + c).copied().unwrap_or(255));
                let p10 = srgb_to_linear_fast(src_data.get(idx10 + c).copied().unwrap_or(255));
                let p01 = srgb_to_linear_fast(src_data.get(idx01 + c).copied().unwrap_or(255));
                let p11 = srgb_to_linear_fast(src_data.get(idx11 + c).copied().unwrap_or(255));

                let top = p00 * (1.0 - fx) + p10 * fx;
                let bottom = p01 * (1.0 - fx) + p11 * fx;
                let linear_value = top * (1.0 - fy) + bottom * fy;

                dst[dst_idx + c] = linear_to_srgb_fast(linear_value);
            }

            // Alpha channel - interpolate directly (not sRGB encoded)
            let a00 = src_data.get(idx00 + 3).copied().unwrap_or(255) as f32;
            let a10 = src_data.get(idx10 + 3).copied().unwrap_or(255) as f32;
            let a01 = src_data.get(idx01 + 3).copied().unwrap_or(255) as f32;
            let a11 = src_data.get(idx11 + 3).copied().unwrap_or(255) as f32;
            let top_a = a00 * (1.0 - fx) + a10 * fx;
            let bottom_a = a01 * (1.0 - fx) + a11 * fx;
            let alpha = top_a * (1.0 - fy) + bottom_a * fy;
            dst[dst_idx + 3] = alpha.round().clamp(0.0, 255.0) as u8;
        }
    }

    dst
}
