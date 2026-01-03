// Nanite Frustum Culling Compute Shader
// Tests each cluster's bounding sphere against frustum planes
// Performs LOD selection based on screen-space error
// Outputs visible cluster indices for rendering

// Cluster data (64 bytes, matches ClusterGpu)
struct Cluster {
    bounding_sphere: vec4<f32>,  // xyz=center, w=radius
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    parent_error: f32,           // Error threshold to use parent LOD
    cluster_error: f32,          // This cluster's error metric
    lod_level: u32,
    group_index: u32,
    index_offset: u32,
    triangle_count: u32,
    vertex_offset: u32,
    material_id: u32,
}

// Instance data (80 bytes, matches NaniteInstanceGpu)
struct Instance {
    transform: mat4x4<f32>,
    first_cluster: u32,
    cluster_count: u32,
    first_group: u32,
    group_count: u32,
}

// Culling uniform with LOD selection parameters
struct CullUniform {
    // Frustum planes: xyz=normal, w=distance (6 planes)
    plane0: vec4<f32>,
    plane1: vec4<f32>,
    plane2: vec4<f32>,
    plane3: vec4<f32>,
    plane4: vec4<f32>,
    plane5: vec4<f32>,
    // Params: x=cluster_count, y=instance_count, z=screen_height, w=fov_y
    params: vec4<f32>,
    // View-projection matrix
    view_proj: mat4x4<f32>,
}

// LOD selection uniform
struct LodUniform {
    // Camera position for distance calculation
    camera_pos: vec4<f32>,
    // LOD params: x=error_threshold (pixels), y=enable_lod (0/1), z=force_lod (-1 for auto), w=unused
    lod_params: vec4<f32>,
}

// Counters for atomic operations
struct Counters {
    visible_count: atomic<u32>,
    total_triangles: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
}

// Indirect draw arguments (with atomic fields for compute shader updates)
struct IndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

// Visible cluster entry: packed cluster_id + instance_id
struct VisibleCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,  // Running sum of triangles for vertex indexing
    _pad: u32,
}

// Bind group 0: All culling resources
@group(0) @binding(0) var<uniform> cull_uniform: CullUniform;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read_write> visible_clusters: array<VisibleCluster>;
@group(0) @binding(4) var<storage, read_write> counters: Counters;
@group(0) @binding(5) var<storage, read_write> indirect_args: IndirectArgs;

// Test sphere against a single plane
// Returns signed distance (negative = outside)
fn sphere_plane_distance(sphere_center: vec3<f32>, sphere_radius: f32, plane: vec4<f32>) -> f32 {
    let normal = plane.xyz;
    let distance = plane.w;
    return dot(normal, sphere_center) + distance;
}

// Test sphere against all 6 frustum planes
fn is_sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    // Test against each plane
    // If sphere is completely behind any plane, it's outside
    if (sphere_plane_distance(center, radius, cull_uniform.plane0) < -radius) { return false; }
    if (sphere_plane_distance(center, radius, cull_uniform.plane1) < -radius) { return false; }
    if (sphere_plane_distance(center, radius, cull_uniform.plane2) < -radius) { return false; }
    if (sphere_plane_distance(center, radius, cull_uniform.plane3) < -radius) { return false; }
    if (sphere_plane_distance(center, radius, cull_uniform.plane4) < -radius) { return false; }
    if (sphere_plane_distance(center, radius, cull_uniform.plane5) < -radius) { return false; }

    return true;
}

// Transform a point by a 4x4 matrix
fn transform_point(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
    let transformed = m * vec4<f32>(p, 1.0);
    return transformed.xyz;
}

// Get maximum scale factor from transform matrix (for radius scaling)
fn get_max_scale(m: mat4x4<f32>) -> f32 {
    // Get the length of each basis vector
    let sx = length(vec3<f32>(m[0][0], m[0][1], m[0][2]));
    let sy = length(vec3<f32>(m[1][0], m[1][1], m[1][2]));
    let sz = length(vec3<f32>(m[2][0], m[2][1], m[2][2]));
    return max(max(sx, sy), sz);
}

// Calculate screen-space error in pixels
fn calculate_screen_error(world_error: f32, distance: f32, screen_height: f32, fov_y: f32) -> f32 {
    if (distance <= 0.0) {
        return 1000000.0; // Very large error for behind camera
    }

    // Project error to screen space
    // error_pixels = (error_world / distance) * (screen_height / (2 * tan(fov/2)))
    let projection_scale = screen_height / (2.0 * tan(fov_y * 0.5));
    return (world_error / distance) * projection_scale;
}

// LOD selection result
const LOD_RENDER: u32 = 0u;   // Render this cluster
const LOD_SKIP: u32 = 1u;     // Skip - use coarser LOD
const LOD_DESCEND: u32 = 2u;  // Use finer LOD

// Max triangles per cluster (must match NaniteConfig and visibility shader)
const MAX_TRIANGLES_PER_CLUSTER: u32 = 128u;

// Determine if this cluster should be rendered based on LOD
fn select_lod(cluster: Cluster, camera_distance: f32, screen_height: f32, fov_y: f32, error_threshold: f32, enable_lod: bool, force_lod: i32) -> u32 {
    // If LOD is disabled, always render finest level (lod_level == 0)
    if (!enable_lod) {
        if (cluster.lod_level == 0u) {
            return LOD_RENDER;
        } else {
            return LOD_SKIP;
        }
    }

    // If forcing a specific LOD level
    if (force_lod >= 0) {
        if (cluster.lod_level == u32(force_lod)) {
            return LOD_RENDER;
        } else {
            return LOD_SKIP;
        }
    }

    // Calculate screen-space error for this cluster
    let screen_error = calculate_screen_error(cluster.cluster_error, camera_distance, screen_height, fov_y);

    // Calculate screen-space error for parent
    let parent_screen_error = calculate_screen_error(cluster.parent_error, camera_distance, screen_height, fov_y);

    // Decision:
    // - If our error is acceptable (< threshold), we're a good candidate
    // - If parent's error is also acceptable, prefer parent (coarser)
    // - If neither works, render this (finest available)

    if (screen_error < error_threshold) {
        // This cluster's error is acceptable
        if (parent_screen_error < error_threshold && cluster.lod_level > 0u) {
            // Parent is also acceptable - use coarser LOD
            return LOD_SKIP;
        }
        // Render this cluster
        return LOD_RENDER;
    } else {
        // Error too high - need finer detail
        if (cluster.lod_level == 0u) {
            // Already at finest level, must render
            return LOD_RENDER;
        }
        // Use finer LOD
        return LOD_DESCEND;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cluster_count = u32(cull_uniform.params.x);
    let instance_count = u32(cull_uniform.params.y);
    let screen_height = cull_uniform.params.z;
    let fov_y = cull_uniform.params.w;

    // Each thread processes one cluster across all instances
    let cluster_idx = global_id.x;

    if (cluster_idx >= cluster_count) {
        return;
    }

    let cluster = clusters[cluster_idx];

    // For Phase 2/3, we only have one instance at identity transform
    // In the future, we'll iterate over instances
    let instance_idx = 0u;

    // Get instance transform (identity for Phase 2)
    var transform = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    if (instance_count > 0u) {
        let instance = instances[instance_idx];
        transform = instance.transform;
    }

    // Transform bounding sphere to world space
    let local_center = cluster.bounding_sphere.xyz;
    let local_radius = cluster.bounding_sphere.w;

    let world_center = transform_point(transform, local_center);
    let scale = get_max_scale(transform);
    let world_radius = local_radius * scale;

    // Frustum test
    if (!is_sphere_in_frustum(world_center, world_radius)) {
        return;
    }

    // LOD selection
    // Extract camera position from view-projection matrix inverse (approximation)
    // For now, assume camera at origin - will be fixed with proper uniform
    let camera_pos = vec3<f32>(0.0, 0.0, 5.0); // Default camera position

    let camera_distance = length(world_center - camera_pos);

    // LOD parameters (hardcoded for now, will come from uniform)
    let error_threshold = 1.0; // 1 pixel error threshold
    let enable_lod = true;
    let force_lod = -1; // Automatic

    let lod_decision = select_lod(cluster, camera_distance, screen_height, fov_y, error_threshold, enable_lod, force_lod);

    if (lod_decision != LOD_RENDER) {
        return;
    }

    // Cluster is visible and at correct LOD! Add to output list
    let visible_idx = atomicAdd(&counters.visible_count, 1u);

    // Calculate triangle offset (for vertex shader indexing)
    let tri_offset = atomicAdd(&counters.total_triangles, cluster.triangle_count);

    visible_clusters[visible_idx] = VisibleCluster(
        cluster_idx,
        instance_idx,
        tri_offset,
        0u
    );

    // Update indirect draw arguments for instanced rendering
    // vertex_count = MAX_TRIANGLES_PER_CLUSTER * 3 (vertices per instance)
    // instance_count = visible_count (one instance per visible cluster)
    atomicMax(&indirect_args.vertex_count, MAX_TRIANGLES_PER_CLUSTER * 3u);
    atomicMax(&indirect_args.instance_count, visible_idx + 1u);
}
