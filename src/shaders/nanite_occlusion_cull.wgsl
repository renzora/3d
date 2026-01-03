// Nanite Occlusion Culling Compute Shader
// Tests frustum-passed clusters against the Hierarchical Z-Buffer (HZB)
// Outputs final list of visible clusters for rendering
//
// Uses conservative testing to avoid false occlusion during camera movement

// Cluster data (matches ClusterGpu)
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

// Instance data (matches NaniteInstanceGpu)
struct Instance {
    transform: mat4x4<f32>,
    first_cluster: u32,
    cluster_count: u32,
    first_group: u32,
    group_count: u32,
}

// Frustum-passed cluster entry (from frustum cull stage)
struct FrustumCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,
    _pad: u32,
}

// Visible cluster output
struct VisibleCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,
    _pad: u32,
}

// Counters
struct Counters {
    visible_count: atomic<u32>,
    total_triangles: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
}

// Indirect draw arguments (using atomics for thread-safe updates)
struct IndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

// Occlusion culling uniform
struct OcclusionUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    hzb_size: vec4<f32>,  // x=width, y=height, z=mip_count, w=unused
    params: vec4<f32>,     // x=cluster_count, y=instance_count, z=screen_height, w=fov_y
}

// Frustum counter (read-only from frustum cull)
struct FrustumCounter {
    count: u32,
    _pad: array<u32, 3>,
}

@group(0) @binding(0) var<uniform> uniforms: OcclusionUniform;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> frustum_clusters: array<FrustumCluster>;
@group(0) @binding(4) var<storage, read> frustum_counter: FrustumCounter;
@group(0) @binding(5) var<storage, read_write> visible_clusters: array<VisibleCluster>;
@group(0) @binding(6) var<storage, read_write> counters: Counters;
@group(0) @binding(7) var<storage, read_write> indirect_args: IndirectArgs;
@group(0) @binding(8) var hzb_texture: texture_2d<f32>;
@group(0) @binding(9) var hzb_sampler: sampler;

// Conservative depth bias to account for 1-frame latency during camera movement
const DEPTH_BIAS: f32 = 0.01;
// Screen-space expansion factor for conservative bounds (in pixels at mip 0)
const SCREEN_EXPANSION_PIXELS: f32 = 1.0;

// optimized firstbithigh equivalent for mip level calculation
// More efficient than log2 and gives exact integer result
fn first_bit_high(x: u32) -> u32 {
    // WGSL's firstLeadingBit returns the bit position of the highest set bit
    // For x=0, it returns 0xFFFFFFFF, so we need to handle that
    if (x == 0u) {
        return 0u;
    }
    return 31u - firstLeadingBit(x);
}

// Project a point to clip space, returns (ndc.xy, linear_depth, clip.w)
fn project_point(view_proj: mat4x4<f32>, world_pos: vec3<f32>) -> vec4<f32> {
    let clip = view_proj * vec4<f32>(world_pos, 1.0);
    return vec4<f32>(clip.xy / clip.w, clip.z / clip.w, clip.w);
}

// Transform AABB by matrix - handles rotation correctly by transforming all 8 corners
fn transform_aabb(m: mat4x4<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> array<vec3<f32>, 2> {
    var new_min = vec3<f32>(1e10, 1e10, 1e10);
    var new_max = vec3<f32>(-1e10, -1e10, -1e10);

    // Transform all 8 corners and find new AABB
    for (var i = 0u; i < 8u; i = i + 1u) {
        let corner = vec3<f32>(
            select(aabb_min.x, aabb_max.x, (i & 1u) != 0u),
            select(aabb_min.y, aabb_max.y, (i & 2u) != 0u),
            select(aabb_min.z, aabb_max.z, (i & 4u) != 0u),
        );
        let transformed = (m * vec4<f32>(corner, 1.0)).xyz;
        new_min = min(new_min, transformed);
        new_max = max(new_max, transformed);
    }

    return array<vec3<f32>, 2>(new_min, new_max);
}

// optimized screen rect calculation with pixel-center coverage
// Projects AABB to screen space using pixel centers for tighter bounds
// Returns: (pixel_min.xy, pixel_max.xy) or (-2,-2,-2,-2) if behind camera
fn project_aabb_to_screen_rect(
    view_proj: mat4x4<f32>,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    hzb_size: vec2<f32>,
    out_min_depth: ptr<function, f32>
) -> vec4<f32> {
    var ndc_min = vec2<f32>(1e10, 1e10);
    var ndc_max = vec2<f32>(-1e10, -1e10);
    var min_depth: f32 = 1.0;
    var any_in_front = false;
    var any_behind = false;

    // Project all 8 corners (optimized: BoxCullFrustumPerspective pattern)
    for (var i = 0u; i < 8u; i = i + 1u) {
        let corner = vec3<f32>(
            select(aabb_min.x, aabb_max.x, (i & 1u) != 0u),
            select(aabb_min.y, aabb_max.y, (i & 2u) != 0u),
            select(aabb_min.z, aabb_max.z, (i & 4u) != 0u),
        );

        let clip = view_proj * vec4<f32>(corner, 1.0);

        // Check if in front of near plane
        if (clip.w > 0.001) {
            any_in_front = true;
            let ndc = clip.xy / clip.w;
            let depth = clip.z / clip.w;

            ndc_min = min(ndc_min, ndc);
            ndc_max = max(ndc_max, ndc);
            min_depth = min(min_depth, depth);
        } else {
            any_behind = true;
        }
    }

    // If all corners behind camera, mark as invisible
    if (!any_in_front) {
        *out_min_depth = 1.0;
        return vec4<f32>(-2.0, -2.0, -2.0, -2.0);
    }

    // If any corner is behind camera, expand to screen edges for that axis
    // (handled via bIntersectsNearPlane flag)
    if (any_behind) {
        ndc_min = max(ndc_min, vec2<f32>(-1.0));
        ndc_max = min(ndc_max, vec2<f32>(1.0));
    }

    // Convert NDC [-1,1] to pixel coordinates [0, hzb_size]
    // Note: Y is flipped (NDC Y+ is up, pixel Y+ is down)
    var pixel_min = vec2<f32>(
        (ndc_min.x * 0.5 + 0.5) * hzb_size.x,
        (0.5 - ndc_max.y * 0.5) * hzb_size.y  // Flip Y
    );
    var pixel_max = vec2<f32>(
        (ndc_max.x * 0.5 + 0.5) * hzb_size.x,
        (0.5 - ndc_min.y * 0.5) * hzb_size.y  // Flip Y
    );

    //'s GetScreenRect: Calculate pixel coverage using pixel centers
    // A pixel is covered if its center (pixel + 0.5) is inside the projected bounds
    // floor(Min - 0.5) + 1.0 gives first pixel whose center is covered
    // floor(Max - 0.5) gives last pixel whose center is covered
    pixel_min = floor(pixel_min - 0.5) + 1.0;
    pixel_max = floor(pixel_max - 0.5);

    // Add conservative expansion (in pixels) to account for camera movement
    pixel_min = pixel_min - vec2<f32>(SCREEN_EXPANSION_PIXELS);
    pixel_max = pixel_max + vec2<f32>(SCREEN_EXPANSION_PIXELS);

    // Clamp to valid range
    pixel_min = max(pixel_min, vec2<f32>(0.0));
    pixel_max = min(pixel_max, hzb_size - vec2<f32>(1.0));

    // Apply conservative depth bias (make object appear closer)
    *out_min_depth = min_depth - DEPTH_BIAS;

    return vec4<f32>(pixel_min.x, pixel_min.y, pixel_max.x, pixel_max.y);
}

// optimized mip level calculation using firstbithigh
// More efficient than log2 and ensures we sample at the correct granularity
fn calculate_mip_level_for_rect(pixel_min: vec2<f32>, pixel_max: vec2<f32>, mip_count: u32) -> u32 {
    // Calculate pixel extent
    let rect_size = pixel_max - pixel_min;
    let max_extent = u32(max(rect_size.x, rect_size.y));

    //'s MipLevelForRect: use firstbithigh to find the mip level
    // This ensures the rect covers at most a few texels at the chosen mip
    let mip = first_bit_high(max(max_extent, 1u));

    return min(mip, mip_count - 1u);
}

// optimized 4x4 HZB sampling for more accurate occlusion testing
// Samples a grid of points across the projected rectangle
fn sample_hzb_4x4(pixel_min: vec2<f32>, pixel_max: vec2<f32>, hzb_size: vec2<f32>, mip: u32) -> f32 {
    // Convert pixel coords at mip 0 to UV coords at the target mip level
    let mip_scale = 1.0 / f32(1u << mip);
    let mip_size = hzb_size * mip_scale;

    // UV coordinates for the rect at the target mip
    let uv_min = pixel_min / hzb_size;
    let uv_max = pixel_max / hzb_size;

    var max_depth: f32 = 0.0;

    // samples a 4x4 grid for large rects
    // For smaller rects, this effectively becomes a 2x2 sample with redundancy
    for (var y = 0u; y < 4u; y = y + 1u) {
        let v = uv_min.y + (uv_max.y - uv_min.y) * (f32(y) + 0.5) / 4.0;
        for (var x = 0u; x < 4u; x = x + 1u) {
            let u = uv_min.x + (uv_max.x - uv_min.x) * (f32(x) + 0.5) / 4.0;
            let uv = vec2<f32>(u, v);
            let d = textureSampleLevel(hzb_texture, hzb_sampler, uv, f32(mip)).r;
            max_depth = max(max_depth, d);
        }
    }

    return max_depth;
}

// Simpler 2x2 sampling for small clusters (faster path)
fn sample_hzb_2x2(pixel_min: vec2<f32>, pixel_max: vec2<f32>, hzb_size: vec2<f32>, mip: u32) -> f32 {
    // UV coordinates at mip 0
    let uv_min = pixel_min / hzb_size;
    let uv_max = pixel_max / hzb_size;

    // Sample 4 corners
    let d00 = textureSampleLevel(hzb_texture, hzb_sampler, uv_min, f32(mip)).r;
    let d10 = textureSampleLevel(hzb_texture, hzb_sampler, vec2<f32>(uv_max.x, uv_min.y), f32(mip)).r;
    let d01 = textureSampleLevel(hzb_texture, hzb_sampler, vec2<f32>(uv_min.x, uv_max.y), f32(mip)).r;
    let d11 = textureSampleLevel(hzb_texture, hzb_sampler, uv_max, f32(mip)).r;

    return max(max(d00, d10), max(d01, d11));
}

// Test if cluster is definitely occluded (optimized IsVisibleHZB)
// Returns true ONLY if we're certain the cluster is hidden
fn is_definitely_occluded(
    view_proj: mat4x4<f32>,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    hzb_size: vec2<f32>,
    mip_count: u32
) -> bool {
    // Project AABB to screen space with pixel-center based bounds
    var min_depth: f32;
    let screen_rect = project_aabb_to_screen_rect(view_proj, aabb_min, aabb_max, hzb_size, &min_depth);

    // If behind camera or invalid, don't cull (be conservative)
    if (screen_rect.x < -1.0) {
        return false;  // Don't cull - might be visible
    }

    let pixel_min = screen_rect.xy;
    let pixel_max = screen_rect.zw;

    // If AABB projects to invalid area, don't cull
    if (pixel_max.x < pixel_min.x || pixel_max.y < pixel_min.y) {
        return false;  // Don't cull - degenerate or sub-pixel projection
    }

    // Calculate mip level using firstbithigh
    let mip = calculate_mip_level_for_rect(pixel_min, pixel_max, mip_count);

    // Calculate screen-space extent for adaptive sampling
    let rect_size = pixel_max - pixel_min;
    let max_extent = max(rect_size.x, rect_size.y);

    // Sample HZB - use 4x4 for large clusters, 2x2 for small ones
    var hzb_max_depth: f32;
    if (max_extent > 16.0) {
        // Large cluster: use 4x4 sampling for better accuracy
        hzb_max_depth = sample_hzb_4x4(pixel_min, pixel_max, hzb_size, mip);
    } else {
        // Small cluster: 2x2 sampling is sufficient
        hzb_max_depth = sample_hzb_2x2(pixel_min, pixel_max, hzb_size, mip);
    }

    // Only cull if the object's nearest point is DEFINITELY behind the HZB
    // Object is occluded if: object_min_depth > hzb_max_depth
    // (object is further than the furthest point in the HZB region)
    return min_depth > hzb_max_depth;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let frustum_count = frustum_counter.count;

    if (thread_idx >= frustum_count) {
        return;
    }

    // Get the frustum-passed cluster
    let frustum_cluster = frustum_clusters[thread_idx];
    let cluster_id = frustum_cluster.cluster_id;
    let instance_id = frustum_cluster.instance_id;

    let cluster = clusters[cluster_id];

    // Get instance transform (or identity if no instances)
    var transform = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    let instance_count = u32(uniforms.params.y);
    if (instance_count > 0u) {
        let instance = instances[instance_id];
        transform = instance.transform;
    }

    // Transform AABB to world space (handles rotation correctly)
    let world_aabb = transform_aabb(transform, cluster.aabb_min.xyz, cluster.aabb_max.xyz);
    let world_min = world_aabb[0];
    let world_max = world_aabb[1];

    // Test occlusion against HZB (optimized conservative test)
    let hzb_size = uniforms.hzb_size.xy;
    let mip_count = u32(uniforms.hzb_size.z);

    // Only skip if DEFINITELY occluded
    if (is_definitely_occluded(uniforms.view_proj, world_min, world_max, hzb_size, mip_count)) {
        return;
    }

    // Cluster is (potentially) visible - add to output list
    let visible_idx = atomicAdd(&counters.visible_count, 1u);
    let tri_offset = atomicAdd(&counters.total_triangles, cluster.triangle_count);

    visible_clusters[visible_idx] = VisibleCluster(
        cluster_id,
        instance_id,
        tri_offset,
        0u
    );

    // Update indirect draw arguments
    atomicMax(&indirect_args.vertex_count, (tri_offset + cluster.triangle_count) * 3u);
    atomicMax(&indirect_args.instance_count, 1u);
}
