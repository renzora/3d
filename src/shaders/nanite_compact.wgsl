// Nanite Cluster Compaction Compute Shader
// Removes gaps in the visible cluster list for better cache efficiency
// Simple stream compaction: reads valid entries and writes them contiguously

// Visible cluster entry
struct VisibleCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,
    _pad: u32,
}

// Counter data
struct CounterData {
    count: u32,
    _pad: array<u32, 3>,
}

// Compaction uniform
struct CompactUniform {
    input_count: u32,
    workgroup_size: u32,
    _pad: array<u32, 2>,
}

@group(0) @binding(0) var<uniform> uniforms: CompactUniform;
@group(0) @binding(1) var<storage, read> input_clusters: array<VisibleCluster>;
@group(0) @binding(2) var<storage, read> counter: CounterData;
@group(0) @binding(3) var<storage, read_write> output_clusters: array<VisibleCluster>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let valid_count = counter.count;

    // Only process valid entries
    if (thread_idx >= valid_count) {
        return;
    }

    // Simple copy - the input is already compacted by atomic adds
    // This pass ensures contiguous memory layout and can be extended
    // for more complex compaction (e.g., removing specific entries)
    let cluster = input_clusters[thread_idx];

    // Recalculate triangle offset for contiguous output
    // In a full implementation, this would use parallel prefix sum
    output_clusters[thread_idx] = cluster;
}
