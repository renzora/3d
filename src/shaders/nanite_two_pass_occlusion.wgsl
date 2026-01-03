// Nanite Two-Pass Occlusion Culling Compute Shader
// Pass 1 (Coarse): Quick test at lower HZB mip level, passes uncertain clusters
// Pass 2 (Refine): Precise test at correct mip level for uncertain clusters
// This reduces HZB sampling cost for clearly visible/occluded clusters

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

// Instance data (matches NaniteInstanceGpu)
struct Instance {
    transform: mat4x4<f32>,
    first_cluster: u32,
    cluster_count: u32,
    first_group: u32,
    group_count: u32,
}

// Visible cluster entry
struct VisibleCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,
    _pad: u32,
}

// Counter data
struct CounterData {
    count: atomic<u32>,
    total_triangles: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
}

// Read-only counter
struct CounterDataRead {
    count: u32,
    _pad: array<u32, 3>,
}

// Two-pass occlusion uniform
struct TwoPassUniform {
    view_proj: mat4x4<f32>,
    hzb_size: vec4<f32>,  // x=width, y=height, z=mip_count, w=unused
    params: vec4<f32>,    // x=cluster_count, y=coarse_mip_offset, z=refine_threshold, w=unused
}

@group(0) @binding(0) var<uniform> uniforms: TwoPassUniform;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> input_clusters: array<VisibleCluster>;
@group(0) @binding(4) var<storage, read> input_counter: CounterDataRead;
@group(0) @binding(5) var<storage, read_write> output_clusters: array<VisibleCluster>;
@group(0) @binding(6) var<storage, read_write> output_counter: CounterData;
@group(0) @binding(7) var hzb_texture: texture_2d<f32>;
@group(0) @binding(8) var hzb_sampler: sampler;

// Transform point by matrix
fn transform_point(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
    let transformed = m * vec4<f32>(p, 1.0);
    return transformed.xyz;
}

// Get max scale from transform
fn get_max_scale(m: mat4x4<f32>) -> f32 {
    let sx = length(vec3<f32>(m[0][0], m[0][1], m[0][2]));
    let sy = length(vec3<f32>(m[1][0], m[1][1], m[1][2]));
    let sz = length(vec3<f32>(m[2][0], m[2][1], m[2][2]));
    return max(max(sx, sy), sz);
}

// Project AABB to screen space, returns (uv_min, uv_max, min_depth)
fn project_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec4<f32> {
    var screen_min = vec2<f32>(1.0);
    var screen_max = vec2<f32>(-1.0);
    var all_behind = true;

    // Project all 8 corners
    for (var i = 0u; i < 8u; i = i + 1u) {
        let corner = vec3<f32>(
            select(aabb_min.x, aabb_max.x, (i & 1u) != 0u),
            select(aabb_min.y, aabb_max.y, (i & 2u) != 0u),
            select(aabb_min.z, aabb_max.z, (i & 4u) != 0u),
        );

        let clip = uniforms.view_proj * vec4<f32>(corner, 1.0);

        if (clip.w > 0.0) {
            all_behind = false;
            let ndc = clip.xy / clip.w;
            screen_min = min(screen_min, ndc);
            screen_max = max(screen_max, ndc);
        }
    }

    if (all_behind) {
        return vec4<f32>(-2.0);
    }

    // Clamp and convert to UV
    screen_min = clamp(screen_min, vec2<f32>(-1.0), vec2<f32>(1.0));
    screen_max = clamp(screen_max, vec2<f32>(-1.0), vec2<f32>(1.0));

    let uv_min = (screen_min + 1.0) * 0.5;
    let uv_max = (screen_max + 1.0) * 0.5;

    return vec4<f32>(uv_min.x, uv_min.y, uv_max.x, uv_max.y);
}

// Calculate min depth of AABB
fn calc_min_depth(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> f32 {
    var min_depth = 1.0;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let corner = vec3<f32>(
            select(aabb_min.x, aabb_max.x, (i & 1u) != 0u),
            select(aabb_min.y, aabb_max.y, (i & 2u) != 0u),
            select(aabb_min.z, aabb_max.z, (i & 4u) != 0u),
        );

        let clip = uniforms.view_proj * vec4<f32>(corner, 1.0);
        if (clip.w > 0.0) {
            min_depth = min(min_depth, clip.z / clip.w);
        }
    }

    return min_depth;
}

// Calculate mip level for screen-space size
fn calc_mip_level(uv_min: vec2<f32>, uv_max: vec2<f32>, mip_offset: f32) -> u32 {
    let screen_size = (uv_max - uv_min) * uniforms.hzb_size.xy;
    let max_extent = max(screen_size.x, screen_size.y);
    let mip = max(0.0, log2(max_extent) - 1.0 + mip_offset);
    return min(u32(mip), u32(uniforms.hzb_size.z) - 1u);
}

// Sample HZB at corners
fn sample_hzb_max(uv_min: vec2<f32>, uv_max: vec2<f32>, mip: u32) -> f32 {
    let d00 = textureSampleLevel(hzb_texture, hzb_sampler, uv_min, f32(mip)).r;
    let d10 = textureSampleLevel(hzb_texture, hzb_sampler, vec2<f32>(uv_max.x, uv_min.y), f32(mip)).r;
    let d01 = textureSampleLevel(hzb_texture, hzb_sampler, vec2<f32>(uv_min.x, uv_max.y), f32(mip)).r;
    let d11 = textureSampleLevel(hzb_texture, hzb_sampler, uv_max, f32(mip)).r;

    return max(max(d00, d10), max(d01, d11));
}

// Occlusion result
const OCCLUDED: u32 = 0u;
const VISIBLE: u32 = 1u;
const UNCERTAIN: u32 = 2u;

// Coarse occlusion test - uses higher mip level for quick rejection
fn coarse_test(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> u32 {
    let screen_bounds = project_aabb(aabb_min, aabb_max);

    // Behind camera
    if (screen_bounds.x < -1.0) {
        return OCCLUDED;
    }

    let uv_min = screen_bounds.xy;
    let uv_max = screen_bounds.zw;

    // Degenerate
    if (uv_max.x <= uv_min.x || uv_max.y <= uv_min.y) {
        return VISIBLE; // Conservative
    }

    let min_depth = calc_min_depth(aabb_min, aabb_max);

    // Use coarser mip level (offset from ideal)
    let coarse_mip_offset = uniforms.params.y;
    let mip = calc_mip_level(uv_min, uv_max, coarse_mip_offset);

    let hzb_depth = sample_hzb_max(uv_min, uv_max, mip);

    // Definitely occluded
    if (min_depth > hzb_depth + 0.001) {
        return OCCLUDED;
    }

    // Check if close to boundary (uncertain)
    let threshold = uniforms.params.z;
    let depth_diff = hzb_depth - min_depth;

    if (depth_diff < threshold) {
        return UNCERTAIN;
    }

    return VISIBLE;
}

// Refined occlusion test - uses correct mip level
fn refine_test(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let screen_bounds = project_aabb(aabb_min, aabb_max);

    if (screen_bounds.x < -1.0) {
        return false; // Occluded
    }

    let uv_min = screen_bounds.xy;
    let uv_max = screen_bounds.zw;

    if (uv_max.x <= uv_min.x || uv_max.y <= uv_min.y) {
        return true; // Visible (conservative)
    }

    let min_depth = calc_min_depth(aabb_min, aabb_max);

    // Use correct mip level (no offset)
    let mip = calc_mip_level(uv_min, uv_max, 0.0);

    let hzb_depth = sample_hzb_max(uv_min, uv_max, mip);

    return min_depth <= hzb_depth;
}

// Output visible cluster
fn output_visible(vis_cluster: VisibleCluster, cluster: Cluster) {
    let idx = atomicAdd(&output_counter.count, 1u);
    let tri_offset = atomicAdd(&output_counter.total_triangles, cluster.triangle_count);

    output_clusters[idx] = VisibleCluster(
        vis_cluster.cluster_id,
        vis_cluster.instance_id,
        tri_offset,
        0u
    );
}

// Coarse occlusion pass - quick test, outputs visible and uncertain
@compute @workgroup_size(64)
fn coarse_occlusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let input_count = input_counter.count;

    if (thread_idx >= input_count) {
        return;
    }

    let vis_cluster = input_clusters[thread_idx];
    let cluster = clusters[vis_cluster.cluster_id];

    // Get world-space AABB (identity transform for now)
    let aabb_min = cluster.aabb_min.xyz;
    let aabb_max = cluster.aabb_max.xyz;

    let result = coarse_test(aabb_min, aabb_max);

    // Pass visible and uncertain to next stage
    if (result != OCCLUDED) {
        output_visible(vis_cluster, cluster);
    }
}

// Refined occlusion pass - precise test for uncertain clusters
// Note: In a full implementation, this would read from a separate "uncertain" buffer
// For simplicity, this performs the refined test on all coarse-passed clusters
@compute @workgroup_size(64)
fn refine_occlusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let input_count = input_counter.count;

    if (thread_idx >= input_count) {
        return;
    }

    let vis_cluster = input_clusters[thread_idx];
    let cluster = clusters[vis_cluster.cluster_id];

    // Get world-space AABB
    let aabb_min = cluster.aabb_min.xyz;
    let aabb_max = cluster.aabb_max.xyz;

    // Perform refined test
    if (refine_test(aabb_min, aabb_max)) {
        output_visible(vis_cluster, cluster);
    }
}
