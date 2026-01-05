// Nanite Shadow Depth Shader
// Renders Nanite geometry into shadow maps from light's perspective
// Uses instanced rendering like the visibility pass

// Light camera uniform (just view-proj matrix)
struct LightCamera {
    view_proj: mat4x4<f32>,
}

// Cluster data (64 bytes)
struct Cluster {
    bounding_sphere: vec4<f32>,
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

// Visible cluster entry from culling
struct VisibleCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,
    _pad: u32,
}

// Counter data
struct CounterData {
    visible_count: u32,
    _pad: array<u32, 3>,
}

// Bind groups
@group(0) @binding(0) var<uniform> light_camera: LightCamera;

@group(1) @binding(0) var<storage, read> clusters: array<Cluster>;
@group(1) @binding(1) var<storage, read> groups: array<u32>;
@group(1) @binding(2) var<storage, read> positions: array<vec3<f32>>;
@group(1) @binding(3) var<storage, read> attributes: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read> indices: array<u32>;
@group(1) @binding(5) var<storage, read> instances: array<Instance>;

@group(2) @binding(0) var<storage, read> visible_clusters: array<VisibleCluster>;
@group(2) @binding(1) var<storage, read> counter: CounterData;

// Max triangles per cluster (must match NaniteConfig)
const MAX_TRIANGLES_PER_CLUSTER: u32 = 128u;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Instanced rendering: each instance = one cluster
    // instance_index = cluster index
    // vertex_index = which vertex within that cluster (0 to MAX_TRIANGLES_PER_CLUSTER * 3 - 1)
    let cluster_id = instance_index;
    let triangle_in_cluster = vertex_index / 3u;
    let vertex_in_triangle = vertex_index % 3u;

    // Check if this cluster is valid
    let cluster_count = arrayLength(&clusters);
    if (cluster_id >= cluster_count) {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let cluster = clusters[cluster_id];

    // Check if this triangle exists in this cluster
    if (triangle_in_cluster >= cluster.triangle_count) {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    // Get index into global index buffer
    let index_in_cluster = triangle_in_cluster * 3u + vertex_in_triangle;
    let global_index_offset = cluster.index_offset + index_in_cluster;
    let vertex_idx = indices[global_index_offset];

    // Get vertex position
    let local_position = positions[cluster.vertex_offset + vertex_idx];

    // Find which instance owns this cluster
    var world_position = local_position;
    let instance_count = arrayLength(&instances);
    for (var i = 0u; i < instance_count; i++) {
        let inst = instances[i];
        // Check if this cluster belongs to this instance
        if (cluster_id >= inst.first_cluster && cluster_id < inst.first_cluster + inst.cluster_count) {
            let transformed = inst.transform * vec4<f32>(local_position, 1.0);
            world_position = transformed.xyz;
            break;
        }
    }

    // Transform to light's clip space
    out.clip_position = light_camera.view_proj * vec4<f32>(world_position, 1.0);

    return out;
}

// Empty fragment shader - we only care about depth
@fragment
fn fs_main(in: VertexOutput) {
    // Depth is automatically written
}
