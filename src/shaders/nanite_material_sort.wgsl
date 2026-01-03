// Nanite Material Sort Compute Shader
// Sorts visible clusters by material ID to reduce state changes during rendering
// Uses counting sort: count → prefix sum → scatter

// Cluster data (matches ClusterGpu)
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

// Visible cluster entry
struct VisibleCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,
    _pad: u32,
}

// Material batch for rendering
struct MaterialBatch {
    material_id: u32,
    first_cluster: u32,
    cluster_count: u32,
    first_triangle: u32,
    triangle_count: u32,
    _pad: array<u32, 3>,
}

// Atomic histogram entry
struct HistogramEntry {
    count: atomic<u32>,
}

// Atomic offset entry
struct OffsetEntry {
    offset: atomic<u32>,
}

// Material sort uniform
struct MaterialSortUniform {
    cluster_count: u32,
    max_material_id: u32,
    _pad: array<u32, 2>,
}

@group(0) @binding(0) var<uniform> uniforms: MaterialSortUniform;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read> visible_clusters: array<VisibleCluster>;
@group(0) @binding(3) var<storage, read_write> histogram: array<HistogramEntry>;
@group(0) @binding(4) var<storage, read_write> offsets: array<OffsetEntry>;
@group(0) @binding(5) var<storage, read_write> sorted_clusters: array<VisibleCluster>;
@group(0) @binding(6) var<storage, read_write> material_batches: array<MaterialBatch>;

// Count materials - first pass
@compute @workgroup_size(64)
fn count_materials(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;

    if (thread_idx >= uniforms.cluster_count) {
        return;
    }

    let vis_cluster = visible_clusters[thread_idx];
    let cluster = clusters[vis_cluster.cluster_id];
    let material_id = cluster.material_id;

    // Clamp to valid range
    let clamped_id = min(material_id, uniforms.max_material_id - 1u);

    // Increment histogram
    atomicAdd(&histogram[clamped_id].count, 1u);
}

// Prefix sum and batch generation
// Note: This is a simple serial prefix sum for small material counts
// For larger counts, use parallel prefix sum
@compute @workgroup_size(1)
fn prefix_sum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var running_offset = 0u;
    var batch_idx = 0u;

    for (var i = 0u; i < uniforms.max_material_id; i = i + 1u) {
        let count = atomicLoad(&histogram[i].count);

        if (count > 0u) {
            // Store offset for scatter pass
            atomicStore(&offsets[i].offset, running_offset);

            // Create material batch
            material_batches[batch_idx] = MaterialBatch(
                i,                  // material_id
                running_offset,     // first_cluster
                count,              // cluster_count
                0u,                 // first_triangle (computed later)
                0u,                 // triangle_count (computed later)
                array<u32, 3>(0u, 0u, 0u)
            );

            batch_idx = batch_idx + 1u;
            running_offset = running_offset + count;
        }
    }
}

// Scatter by material - second pass
@compute @workgroup_size(64)
fn scatter_by_material(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;

    if (thread_idx >= uniforms.cluster_count) {
        return;
    }

    let vis_cluster = visible_clusters[thread_idx];
    let cluster = clusters[vis_cluster.cluster_id];
    let material_id = cluster.material_id;

    // Clamp to valid range
    let clamped_id = min(material_id, uniforms.max_material_id - 1u);

    // Get output position via atomic increment
    let output_idx = atomicAdd(&offsets[clamped_id].offset, 1u);

    // Write to sorted position
    sorted_clusters[output_idx] = vis_cluster;
}
