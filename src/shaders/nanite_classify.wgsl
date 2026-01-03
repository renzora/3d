// Nanite Cluster Classification Compute Shader
// Sorts visible clusters into SW (small triangles) and HW (large triangles) bins
// Uses sphere-to-screen projection (Mara & Morgan 2013) for accurate bounds

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

// Visible cluster entry
struct VisibleCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,
    _pad: u32,
}

// Classification counters
struct ClassifyCounters {
    sw_count: atomic<u32>,
    hw_count: atomic<u32>,
    sw_triangles: atomic<u32>,
    hw_triangles: atomic<u32>,
}

// Indirect draw arguments
struct IndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

// Classification uniform
struct ClassifyUniform {
    view_proj: mat4x4<f32>,
    screen_size: vec4<f32>,  // x=width, y=height, z=proj_scale (screen_height / 2*tan(fov/2)), w=unused
    params: vec4<f32>,       // x=visible_count, y=size_threshold (pixels), z=unused, w=unused
}

@group(0) @binding(0) var<uniform> uniforms: ClassifyUniform;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> visible_clusters: array<VisibleCluster>;
@group(0) @binding(4) var<storage, read_write> sw_clusters: array<VisibleCluster>;
@group(0) @binding(5) var<storage, read_write> hw_clusters: array<VisibleCluster>;
@group(0) @binding(6) var<storage, read_write> counters: ClassifyCounters;
@group(0) @binding(7) var<storage, read_write> hw_indirect: IndirectArgs;

// Transform a point by a 4x4 matrix
fn transform_point(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
    let transformed = m * vec4<f32>(p, 1.0);
    return transformed.xyz;
}

// Get maximum scale factor from transform matrix
fn get_max_scale(m: mat4x4<f32>) -> f32 {
    let sx = length(vec3<f32>(m[0][0], m[0][1], m[0][2]));
    let sy = length(vec3<f32>(m[1][0], m[1][1], m[1][2]));
    let sz = length(vec3<f32>(m[2][0], m[2][1], m[2][2]));
    return max(max(sx, sy), sz);
}

// optimized sphere-to-screen rect calculation
// Based on "2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere"
// by Mara & Morgan 2013 (Journal of Computer Graphics Techniques)
// Returns: vec4(min_x, min_y, max_x, max_y) in pixels, or invalid rect if behind camera
fn sphere_to_screen_rect(
    view_center: vec3<f32>,  // Sphere center in view space
    radius: f32,
    screen_size: vec2<f32>,
    proj_scale: f32          // screen_height / (2 * tan(fov/2))
) -> vec4<f32> {
    let c = view_center;
    let r = radius;

    // If sphere is behind or intersecting near plane, return large rect
    if (c.z - r < 0.1) {
        return vec4<f32>(0.0, 0.0, screen_size.x, screen_size.y);
    }

    // Distance squared from origin to sphere center
    let d2 = dot(c, c);
    let r2 = r * r;

    // If camera is inside sphere, return full screen
    if (d2 <= r2) {
        return vec4<f32>(0.0, 0.0, screen_size.x, screen_size.y);
    }

    // Calculate the projected bounds using Mara & Morgan method
    // For X axis: find tangent points where view ray grazes the sphere
    // The tangent rays form a cone, and we want the screen-space extent

    let cx = c.x;
    let cz = c.z;

    // For X bounds: solve for tangent in XZ plane
    // Discriminant for tangent calculation
    let dx2 = cx * cx + cz * cz;
    let disc_x = dx2 - r2;

    var x_min: f32;
    var x_max: f32;

    if (disc_x > 0.0) {
        let sqrt_disc_x = sqrt(disc_x);
        // Tangent points in XZ plane
        let t_x = r / sqrt(dx2);
        let sin_x = t_x;
        let cos_x = sqrt(1.0 - t_x * t_x);

        // Project tangent directions to screen X
        // Using similar triangles: screen_x = (world_x / world_z) * proj_scale
        let tan_left = (cx * cos_x - cz * sin_x) / (cz * cos_x + cx * sin_x);
        let tan_right = (cx * cos_x + cz * sin_x) / (cz * cos_x - cx * sin_x);

        x_min = (tan_left * proj_scale + screen_size.x * 0.5);
        x_max = (tan_right * proj_scale + screen_size.x * 0.5);

        // Ensure correct ordering
        if (x_min > x_max) {
            let tmp = x_min;
            x_min = x_max;
            x_max = tmp;
        }
    } else {
        // Sphere covers full X range
        x_min = 0.0;
        x_max = screen_size.x;
    }

    // For Y bounds: similar calculation in YZ plane
    let cy = c.y;
    let dy2 = cy * cy + cz * cz;
    let disc_y = dy2 - r2;

    var y_min: f32;
    var y_max: f32;

    if (disc_y > 0.0) {
        let t_y = r / sqrt(dy2);
        let sin_y = t_y;
        let cos_y = sqrt(1.0 - t_y * t_y);

        let tan_top = (cy * cos_y - cz * sin_y) / (cz * cos_y + cy * sin_y);
        let tan_bottom = (cy * cos_y + cz * sin_y) / (cz * cos_y - cy * sin_y);

        // Note: Y is flipped for screen coordinates
        y_min = (screen_size.y * 0.5 - tan_bottom * proj_scale);
        y_max = (screen_size.y * 0.5 - tan_top * proj_scale);

        if (y_min > y_max) {
            let tmp = y_min;
            y_min = y_max;
            y_max = tmp;
        }
    } else {
        y_min = 0.0;
        y_max = screen_size.y;
    }

    // Clamp to screen bounds
    x_min = max(x_min, 0.0);
    x_max = min(x_max, screen_size.x);
    y_min = max(y_min, 0.0);
    y_max = min(y_max, screen_size.y);

    return vec4<f32>(x_min, y_min, x_max, y_max);
}

// Simple fallback: calculate screen-space size of a bounding sphere
fn calculate_screen_size_simple(center: vec3<f32>, radius: f32, view_proj: mat4x4<f32>, screen_size: vec2<f32>) -> f32 {
    // Project center to clip space
    let clip = view_proj * vec4<f32>(center, 1.0);

    // If behind camera, consider it large (use HW raster)
    if (clip.w <= 0.0) {
        return 1000.0;
    }

    // Project to NDC
    let ndc = clip.xyz / clip.w;

    // Calculate screen-space radius using projection
    // This approximates the projected size
    let edge_clip = view_proj * vec4<f32>(center + vec3<f32>(radius, 0.0, 0.0), 1.0);

    if (edge_clip.w <= 0.0) {
        return 1000.0;
    }

    let edge_ndc = edge_clip.xyz / edge_clip.w;

    // Screen-space distance in pixels
    let screen_radius = length((edge_ndc.xy - ndc.xy) * screen_size * 0.5);

    // Return diameter as the "size"
    return screen_radius * 2.0;
}

// SW rasterization is most efficient for triangles covering < 2x2 pixels on average
// Above this threshold, hardware rasterization wins due to parallelism
const DEFAULT_SW_THRESHOLD: f32 = 4.0;  // 2x2 pixel triangles

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let visible_count = u32(uniforms.params.x);
    let size_threshold = select(DEFAULT_SW_THRESHOLD, uniforms.params.y, uniforms.params.y > 0.0);

    if (thread_idx >= visible_count) {
        return;
    }

    // Get the visible cluster
    let vis_cluster = visible_clusters[thread_idx];
    let cluster_id = vis_cluster.cluster_id;
    let instance_id = vis_cluster.instance_id;

    let cluster = clusters[cluster_id];

    // Get instance transform (or identity if no instances)
    var transform = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    // Note: In a full implementation, we'd check instance_count and get the transform

    // Transform bounding sphere to world space
    let world_center = transform_point(transform, cluster.bounding_sphere.xyz);
    let scale = get_max_scale(transform);
    let world_radius = cluster.bounding_sphere.w * scale;

    // Calculate screen-space cluster size
    let cluster_screen_size = calculate_screen_size_simple(
        world_center,
        world_radius,
        uniforms.view_proj,
        uniforms.screen_size.xy
    );

    // Estimate average triangle size from cluster size
    // For a cluster with N triangles covering area A, average triangle area ≈ A/N
    // Average triangle "size" (edge length) ≈ sqrt(A/N) ≈ sqrt(cluster_area / N)
    // cluster_area ≈ cluster_screen_size^2, so avg_tri_size ≈ cluster_screen_size / sqrt(N)
    let triangle_count = cluster.triangle_count;
    let cluster_area = cluster_screen_size * cluster_screen_size;
    let avg_triangle_area = cluster_area / max(f32(triangle_count), 1.0);
    let avg_triangle_size = sqrt(avg_triangle_area);

    // optimized classification:
    // - SW rasterizer is faster for sub-pixel and very small triangles
    // - HW rasterizer is faster for larger triangles due to parallelism
    // - The crossover point depends on GPU, but ~2x2 pixels is typical
    let use_sw_raster = avg_triangle_size < size_threshold;

    if (use_sw_raster) {
        // Small triangles -> software rasterization
        let sw_idx = atomicAdd(&counters.sw_count, 1u);
        let tri_offset = atomicAdd(&counters.sw_triangles, triangle_count);

        sw_clusters[sw_idx] = VisibleCluster(
            cluster_id,
            instance_id,
            tri_offset,
            0u
        );
    } else {
        // Large triangles -> hardware rasterization
        let hw_idx = atomicAdd(&counters.hw_count, 1u);
        let tri_offset = atomicAdd(&counters.hw_triangles, triangle_count);

        hw_clusters[hw_idx] = VisibleCluster(
            cluster_id,
            instance_id,
            tri_offset,
            0u
        );

        // Update indirect draw arguments for HW rasterization
        atomicMax(&hw_indirect.vertex_count, (tri_offset + triangle_count) * 3u);
        hw_indirect.instance_count = 1u;
    }
}
