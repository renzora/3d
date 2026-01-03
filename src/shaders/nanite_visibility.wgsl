// Nanite Visibility Buffer Shader
// Writes packed triangle/cluster/instance IDs to visibility texture
//
// With frustum culling enabled:
// - vertex_index encodes which triangle vertex within the visible set
// - We look up the visible cluster from the visible_clusters buffer
// - Each visible cluster entry contains the cluster_id and triangle_offset

// Camera uniform (matches PbrCameraUniform)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    render_mode: u32,
    hemisphere_sky: vec4<f32>,
    hemisphere_ground: vec4<f32>,
    ibl_settings: vec4<f32>,
    light0_pos: vec4<f32>,
    light0_color: vec4<f32>,
    light1_pos: vec4<f32>,
    light1_color: vec4<f32>,
    light2_pos: vec4<f32>,
    light2_color: vec4<f32>,
    light3_pos: vec4<f32>,
    light3_color: vec4<f32>,
    detail_settings: vec4<f32>,
}

// Cluster data (64 bytes)
struct Cluster {
    bounding_sphere: vec4<f32>,  // xyz=center, w=radius
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    parent_error: f32,
    cluster_error: f32,
    lod_level: u32,
    group_index: u32,
    index_offset: u32,
    triangle_count: u32,
    vertex_offset: u32,
    material_id: u32,
}

// Instance data (80 bytes)
struct Instance {
    transform: mat4x4<f32>,
    first_cluster: u32,
    cluster_count: u32,
    first_group: u32,
    group_count: u32,
}

// Visible cluster entry from culling (16 bytes)
struct VisibleCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,  // Cumulative triangle count before this cluster
    _pad: u32,
}

// Bind groups
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> clusters: array<Cluster>;
@group(1) @binding(1) var<storage, read> groups: array<u32>;  // Placeholder
@group(1) @binding(2) var<storage, read> positions: array<vec3<f32>>;
@group(1) @binding(3) var<storage, read> attributes: array<vec4<f32>>;  // Normals, etc.
@group(1) @binding(4) var<storage, read> indices: array<u32>;
@group(1) @binding(5) var<storage, read> instances: array<Instance>;

@group(2) @binding(0) var<storage, read> visible_clusters: array<VisibleCluster>;

// Counter data from culling pass
struct CounterData {
    visible_count: u32,
    _pad: array<u32, 3>,
}
@group(2) @binding(1) var<storage, read> counter: CounterData;

// Vertex output
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) visibility_id: u32,
}

// Pack visibility ID from triangle, cluster, instance
fn pack_visibility_id(triangle_id: u32, cluster_id: u32, instance_id: u32) -> u32 {
    return (triangle_id & 0xFFFu) | ((cluster_id & 0xFFFFu) << 12u) | ((instance_id & 0xFu) << 28u);
}

// Max triangles per cluster (must match NaniteConfig)
const MAX_TRIANGLES_PER_CLUSTER: u32 = 128u;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Instanced rendering: each instance = one visible cluster
    // vertex_index = which vertex within the cluster (0 to MAX_TRIANGLES_PER_CLUSTER * 3 - 1)
    // instance_index = which visible cluster
    let visible_cluster_idx = instance_index;
    let triangle_in_cluster = vertex_index / 3u;
    let vertex_in_triangle = vertex_index % 3u;

    // Check if this instance is valid
    let visible_count = counter.visible_count;
    if (visible_cluster_idx >= visible_count) {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.visibility_id = 0u;
        return out;
    }

    let vis_cluster = visible_clusters[visible_cluster_idx];
    let cluster = clusters[vis_cluster.cluster_id];

    // Check if this triangle exists in this cluster
    if (triangle_in_cluster >= cluster.triangle_count) {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.visibility_id = 0u;
        return out;
    }

    // Get index into global index buffer
    let index_in_cluster = triangle_in_cluster * 3u + vertex_in_triangle;
    let global_index_offset = cluster.index_offset + index_in_cluster;
    let vertex_idx = indices[global_index_offset];

    // Get vertex position
    let local_position = positions[cluster.vertex_offset + vertex_idx];

    // Apply instance transform if available
    var world_position = local_position;
    if (vis_cluster.instance_id < arrayLength(&instances)) {
        let instance = instances[vis_cluster.instance_id];
        let transformed = instance.transform * vec4<f32>(local_position, 1.0);
        world_position = transformed.xyz;
    }

    // Transform to clip space
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);

    // Pack visibility ID
    out.visibility_id = pack_visibility_id(triangle_in_cluster, vis_cluster.cluster_id, vis_cluster.instance_id);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    // Write packed visibility ID
    // Adding 1 so that 0 represents "no geometry" (background)
    return in.visibility_id + 1u;
}
