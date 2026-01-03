// Nanite Software Rasterization Compute Shader
// Rasterizes small triangles directly using compute, bypassing hardware rasterizer overhead
// Based on Nanite rasterization with subpixel precision and fill convention
//
// Key techniques:
// - 8.8 fixed-point subpixel precision (256 subpixel samples)
// - Top-left fill convention for correct edge coverage
// - Pre-computed edge constants for incremental rasterization
// - Adaptive scanline/rect rasterization based on triangle shape
// - Depth plane interpolation for efficient Z computation

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

// Vertex position
struct NaniteVertex {
    position: vec4<f32>,
}

// Visible cluster entry
struct VisibleCluster {
    cluster_id: u32,
    instance_id: u32,
    triangle_offset: u32,
    _pad: u32,
}

// Classification counters (read-only in this shader)
struct ClassifyCounters {
    sw_count: u32,
    hw_count: u32,
    sw_triangles: u32,
    hw_triangles: u32,
}

// SW rasterization uniform
struct SwRasterUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    screen_size: vec4<f32>,  // x=width, y=height, z=1/width, w=1/height
    params: vec4<f32>,       // x=sw_cluster_count, y=total_triangles, z=depth_bias, w=unused
}

// Atomic depth buffer entry
struct AtomicDepthEntry {
    depth: atomic<u32>,
}

// optimized rasterization triangle structure
// Pre-computed values for efficient pixel iteration
struct RasterTri {
    // Pixel bounding box (inclusive)
    min_pixel: vec2<i32>,
    max_pixel: vec2<i32>,

    // Edge vectors (in subpixel space, scaled to pixel stepping)
    edge01: vec2<f32>,
    edge12: vec2<f32>,
    edge20: vec2<f32>,

    // Half-edge constants (adjusted for fill convention)
    c0: f32,
    c1: f32,
    c2: f32,

    // Depth plane: z = depth_plane.x + bary.y * depth_plane.y + bary.z * depth_plane.z
    depth_plane: vec3<f32>,

    // 1/W for perspective-correct interpolation
    inv_w: vec3<f32>,

    // Barycentric derivatives for attribute interpolation
    bary_dx: vec3<f32>,
    bary_dy: vec3<f32>,

    // Triangle validity flags
    is_valid: bool,
    is_backface: bool,
}

@group(0) @binding(0) var<uniform> uniforms: SwRasterUniform;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> positions: array<NaniteVertex>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> sw_clusters: array<VisibleCluster>;
@group(0) @binding(6) var<storage, read> counters: ClassifyCounters;
@group(0) @binding(7) var visibility_buffer: texture_storage_2d<r32uint, read_write>;
@group(0) @binding(8) var<storage, read_write> atomic_depth: array<AtomicDepthEntry>;

// Subpixel precision: 8.8 fixed point (256 subpixel samples per pixel)
const SUBPIXEL_SAMPLES: f32 = 256.0;
const SUBPIXEL_SAMPLES_U: u32 = 256u;

// Maximum triangle size in pixels for SW rasterizer
//limits to 64x64 to prevent extremely long rasterization times
const MAX_RASTER_SIZE: i32 = 64;

// Convert depth to u32 for atomic operations
// Using reversed-Z style: higher u32 = closer to camera
fn depth_to_u32(depth: f32) -> u32 {
    let clamped = clamp(depth, 0.0, 1.0);
    return u32((1.0 - clamped) * 4294967295.0);
}

// Project vertex to screen space with subpixel precision
// Returns: xy = subpixel coords, z = NDC depth, w = clip.w
fn project_vertex_subpixel(pos: vec3<f32>, view_proj: mat4x4<f32>, screen_size: vec2<f32>) -> vec4<f32> {
    let clip = view_proj * vec4<f32>(pos, 1.0);

    // Perspective divide
    let ndc = clip.xyz / clip.w;

    // Convert to screen space with subpixel precision
    // Subpixel coords: [0, screen_size * SUBPIXEL_SAMPLES]
    let screen = vec2<f32>(
        (ndc.x * 0.5 + 0.5) * screen_size.x * SUBPIXEL_SAMPLES,
        (1.0 - (ndc.y * 0.5 + 0.5)) * screen_size.y * SUBPIXEL_SAMPLES  // Flip Y
    );

    return vec4<f32>(screen, ndc.z, clip.w);
}

// optimized triangle setup with subpixel precision and fill convention
fn setup_triangle(scissor_min: vec2<i32>, scissor_max: vec2<i32>, v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>) -> RasterTri {
    var tri: RasterTri;
    tri.is_valid = true;
    tri.inv_w = vec3<f32>(v0.w, v1.w, v2.w);

    // Vertex positions in subpixel space (8.8 fixed point conceptually)
    let vert0 = v0.xy;
    let vert1 = v1.xy;
    let vert2 = v2.xy;

    // Edge vectors (in subpixel space)
    tri.edge01 = vert0 - vert1;
    tri.edge12 = vert1 - vert2;
    tri.edge20 = vert2 - vert0;

    // Calculate signed area (2x area via cross product)
    // DetXY = Edge01.y * Edge20.x - Edge01.x * Edge20.y
    let det_xy = tri.edge01.y * tri.edge20.x - tri.edge01.x * tri.edge20.y;
    tri.is_backface = det_xy >= 0.0;

    // For backface culling disabled, swap winding
    if (tri.is_backface) {
        // Swap winding order for consistent rasterization
        tri.edge01 = -tri.edge01;
        tri.edge12 = -tri.edge12;
        tri.edge20 = -tri.edge20;
    }

    // Bounding box in subpixel space
    let min_subpixel = min(min(vert0, vert1), vert2);
    let max_subpixel = max(max(vert0, vert1), vert2);

    // Convert to pixel coordinates (round to nearest pixel center)
    // floor((subpixel + half_sample - 1) / samples) for min
    // floor((subpixel - half_sample - 1) / samples) for max (inclusive)
    let half_sample = SUBPIXEL_SAMPLES * 0.5;
    tri.min_pixel = vec2<i32>(floor((min_subpixel + half_sample - 1.0) / SUBPIXEL_SAMPLES));
    tri.max_pixel = vec2<i32>(floor((max_subpixel - half_sample - 1.0) / SUBPIXEL_SAMPLES));

    // Apply scissor
    tri.min_pixel = max(tri.min_pixel, scissor_min);
    tri.max_pixel = min(tri.max_pixel, scissor_max - vec2<i32>(1));

    // Limit rasterizer bounds to prevent extremely long rasterization
    tri.max_pixel = min(tri.max_pixel, tri.min_pixel + vec2<i32>(MAX_RASTER_SIZE - 1));

    // Cull if no pixels covered
    if (any(tri.min_pixel > tri.max_pixel)) {
        tri.is_valid = false;
        return tri;
    }

    // Rebase vertices to min_pixel with half-pixel offset
    let base_subpixel = vec2<f32>(tri.min_pixel) * SUBPIXEL_SAMPLES + half_sample;
    let rebased0 = vert0 - base_subpixel;
    let rebased1 = vert1 - base_subpixel;
    let rebased2 = vert2 - base_subpixel;

    // Half-edge constants (evaluate edge function at origin after rebasing)
    tri.c0 = tri.edge12.y * rebased1.x - tri.edge12.x * rebased1.y;
    tri.c1 = tri.edge20.y * rebased2.x - tri.edge20.x * rebased2.y;
    tri.c2 = tri.edge01.y * rebased0.x - tri.edge01.x * rebased0.y;

    // Scale factor to normalize barycentric coordinates
    let sum_c = tri.c0 + tri.c1 + tri.c2;
    if (abs(sum_c) < 0.0001) {
        tri.is_valid = false;
        return tri;
    }
    let scale_to_unit = SUBPIXEL_SAMPLES / sum_c;

    //Fill Convention: Top-left rule for CCW triangles
    // Subtract 1 from C if edge is not top-left:
    // - Top edge: horizontal edge with Y going up (edge.y > 0)
    // - Left edge: edge going up (edge.y > 0) or exactly horizontal going left (edge.y == 0 && edge.x < 0)
    // For screen coords with Y+ down, this is inverted
    tri.c0 -= select(0.0, 1.0, tri.edge12.y < 0.0 || (tri.edge12.y == 0.0 && tri.edge12.x > 0.0));
    tri.c1 -= select(0.0, 1.0, tri.edge20.y < 0.0 || (tri.edge20.y == 0.0 && tri.edge20.x > 0.0));
    tri.c2 -= select(0.0, 1.0, tri.edge01.y < 0.0 || (tri.edge01.y == 0.0 && tri.edge01.x > 0.0));

    // Scale C down by SUBPIXEL_SAMPLES for pixel stepping
    tri.c0 = tri.c0 / SUBPIXEL_SAMPLES;
    tri.c1 = tri.c1 / SUBPIXEL_SAMPLES;
    tri.c2 = tri.c2 / SUBPIXEL_SAMPLES;

    // Barycentric derivatives for interpolation
    tri.bary_dx = vec3<f32>(-tri.edge12.y, -tri.edge20.y, -tri.edge01.y) * scale_to_unit;
    tri.bary_dy = vec3<f32>(tri.edge12.x, tri.edge20.x, tri.edge01.x) * scale_to_unit;

    // Depth plane for linear interpolation
    // z = z0 + (z1 - z0) * bary1 + (z2 - z0) * bary2
    tri.depth_plane.x = v0.z;
    tri.depth_plane.y = v1.z - v0.z;
    tri.depth_plane.z = v2.z - v0.z;
    tri.depth_plane.yz = tri.depth_plane.yz * scale_to_unit;

    return tri;
}

// Write a pixel to the visibility buffer with atomic depth test
fn write_pixel(coord: vec2<i32>, vis_id: u32, depth: f32, screen_width: u32) {
    // Add depth bias
    let biased_depth = depth + uniforms.params.z;

    // Convert to u32 for atomic comparison (higher = closer)
    let depth_u32 = depth_to_u32(biased_depth);

    let pixel_idx = u32(coord.y) * screen_width + u32(coord.x);

    // Atomic depth test
    let old_depth = atomicMax(&atomic_depth[pixel_idx].depth, depth_u32);

    // If we won the depth test
    if (depth_u32 > old_depth) {
        textureStore(visibility_buffer, coord, vec4<u32>(vis_id, 0u, 0u, 0u));
    }
}

// optimized rect rasterization for small triangles
// Uses nested loops with incremental edge function updates
fn rasterize_tri_rect(tri: RasterTri, vis_id: u32, screen_width: u32) {
    var cy0 = tri.c0;
    var cy1 = tri.c1;
    var cy2 = tri.c2;

    var y = tri.min_pixel.y;
    loop {
        var cx0 = cy0;
        var cx1 = cy1;
        var cx2 = cy2;

        var x = tri.min_pixel.x;
        loop {
            // Check if pixel is inside triangle (all edge functions >= 0)
            if (min(min(cx0, cx1), cx2) >= 0.0) {
                // Compute normalized barycentric coordinates
                let sum = cx0 + cx1 + cx2;
                if (sum > 0.0) {
                    let inv_sum = 1.0 / sum;
                    let b0 = cx0 * inv_sum;
                    let b1 = cx1 * inv_sum;
                    let b2 = cx2 * inv_sum;

                    // Interpolate depth
                    let depth = tri.depth_plane.x + b1 * tri.depth_plane.y + b2 * tri.depth_plane.z;

                    write_pixel(vec2<i32>(x, y), vis_id, depth, screen_width);
                }
            }

            if (x >= tri.max_pixel.x) {
                break;
            }

            // Step edge functions in X
            cx0 -= tri.edge12.y;
            cx1 -= tri.edge20.y;
            cx2 -= tri.edge01.y;
            x = x + 1;
        }

        if (y >= tri.max_pixel.y) {
            break;
        }

        // Step edge functions in Y
        cy0 += tri.edge12.x;
        cy1 += tri.edge20.x;
        cy2 += tri.edge01.x;
        y = y + 1;
    }
}

// optimized scanline rasterization for wider triangles
// Computes X span per scanline using edge intersections
fn rasterize_tri_scanline(tri: RasterTri, vis_id: u32, screen_width: u32) {
    var cy0 = tri.c0;
    var cy1 = tri.c1;
    var cy2 = tri.c2;

    // Edge slopes for X intersection calculation
    let edge012 = vec3<f32>(tri.edge12.y, tri.edge20.y, tri.edge01.y);
    let b_open_edge = edge012 < vec3<f32>(0.0);
    let inv_edge012 = select(
        vec3<f32>(1e8),
        1.0 / edge012,
        abs(edge012) > vec3<f32>(0.0001)
    );

    var y = tri.min_pixel.y;
    let width = f32(tri.max_pixel.x - tri.min_pixel.x);

    loop {
        // Calculate X intersections for each edge
        let cross_x = vec3<f32>(cy0, cy1, cy2) * inv_edge012;

        // Opening edges (edge going up) give min X, closing edges give max X
        let min_x_vals = select(vec3<f32>(0.0), cross_x, b_open_edge);
        let max_x_vals = select(cross_x, vec3<f32>(width), b_open_edge);

        // Find actual span
        let x0_f = ceil(max(max(min_x_vals.x, min_x_vals.y), min_x_vals.z));
        let x1_f = min(min(max_x_vals.x, max_x_vals.y), max_x_vals.z);

        // Initialize edge functions for start of span
        var cx0 = cy0 - x0_f * tri.edge12.y;
        var cx1 = cy1 - x0_f * tri.edge20.y;
        var cx2 = cy2 - x0_f * tri.edge01.y;

        let x_start = tri.min_pixel.x + i32(x0_f);
        let x_end = tri.min_pixel.x + i32(x1_f);

        // Rasterize the span
        var x = x_start;
        loop {
            if (x > x_end) {
                break;
            }

            // Check edge functions (might still need check due to rounding)
            if (min(min(cx0, cx1), cx2) >= 0.0) {
                let sum = cx0 + cx1 + cx2;
                if (sum > 0.0) {
                    let inv_sum = 1.0 / sum;
                    let b0 = cx0 * inv_sum;
                    let b1 = cx1 * inv_sum;
                    let b2 = cx2 * inv_sum;

                    let depth = tri.depth_plane.x + b1 * tri.depth_plane.y + b2 * tri.depth_plane.z;

                    write_pixel(vec2<i32>(x, y), vis_id, depth, screen_width);
                }
            }

            cx0 -= tri.edge12.y;
            cx1 -= tri.edge20.y;
            cx2 -= tri.edge01.y;
            x = x + 1;
        }

        if (y >= tri.max_pixel.y) {
            break;
        }

        cy0 += tri.edge12.x;
        cy1 += tri.edge20.x;
        cy2 += tri.edge01.x;
        y = y + 1;
    }
}

// Adaptive rasterization: choose method based on triangle shape
fn rasterize_tri_adaptive(tri: RasterTri, vis_id: u32, screen_width: u32) {
    let width = tri.max_pixel.x - tri.min_pixel.x;

    // Use scanline for wide triangles (>4 pixels wide)
    // Scanline has overhead but is faster for wide spans
    if (width > 4) {
        rasterize_tri_scanline(tri, vis_id, screen_width);
    } else {
        rasterize_tri_rect(tri, vis_id, screen_width);
    }
}

// Find which SW cluster and triangle index this thread should process
fn find_triangle(thread_idx: u32, sw_count: u32) -> vec2<u32> {
    var running_offset = 0u;

    for (var i = 0u; i < sw_count; i = i + 1u) {
        let sw_cluster = sw_clusters[i];
        let cluster = clusters[sw_cluster.cluster_id];
        let tri_count = cluster.triangle_count;

        if (thread_idx < running_offset + tri_count) {
            let local_tri = thread_idx - running_offset;
            return vec2<u32>(i, local_tri);
        }

        running_offset = running_offset + tri_count;
    }

    return vec2<u32>(0xFFFFFFFFu, 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let sw_triangle_count = counters.sw_triangles;
    let sw_count = counters.sw_count;

    if (thread_idx >= sw_triangle_count || sw_count == 0u) {
        return;
    }

    // Find which cluster and triangle this thread processes
    let cluster_tri = find_triangle(thread_idx, sw_count);
    if (cluster_tri.x == 0xFFFFFFFFu) {
        return;
    }

    let sw_cluster_idx = cluster_tri.x;
    let local_tri_idx = cluster_tri.y;

    let sw_cluster = sw_clusters[sw_cluster_idx];
    let cluster_id = sw_cluster.cluster_id;
    let cluster = clusters[cluster_id];

    // Get triangle indices
    let base_idx = cluster.index_offset + local_tri_idx * 3u;
    let i0 = indices[base_idx + 0u];
    let i1 = indices[base_idx + 1u];
    let i2 = indices[base_idx + 2u];

    // Get vertex positions
    let v0_local = positions[cluster.vertex_offset + i0].position.xyz;
    let v1_local = positions[cluster.vertex_offset + i1].position.xyz;
    let v2_local = positions[cluster.vertex_offset + i2].position.xyz;

    // Transform by instance (identity for now - would use instance transform)
    let v0_world = v0_local;
    let v1_world = v1_local;
    let v2_world = v2_local;

    // Project to screen space with subpixel precision
    let screen_size = uniforms.screen_size.xy;
    let p0 = project_vertex_subpixel(v0_world, uniforms.view_proj, screen_size);
    let p1 = project_vertex_subpixel(v1_world, uniforms.view_proj, screen_size);
    let p2 = project_vertex_subpixel(v2_world, uniforms.view_proj, screen_size);

    // Check if any vertex is behind camera
    if (p0.w <= 0.0 || p1.w <= 0.0 || p2.w <= 0.0) {
        return;
    }

    // Setup triangle with optimized precision and fill convention
    let scissor_min = vec2<i32>(0);
    let scissor_max = vec2<i32>(screen_size);
    let tri = setup_triangle(scissor_min, scissor_max, p0, p1, p2);

    if (!tri.is_valid) {
        return;
    }

    // Pack visibility ID: triangle_id (12 bits) | cluster_id (16 bits) | instance_id (4 bits)
    let vis_id = (local_tri_idx & 0xFFFu) | ((cluster_id & 0xFFFFu) << 12u) | ((sw_cluster.instance_id & 0xFu) << 28u);

    // Rasterize using adaptive method
    let screen_width = u32(screen_size.x);
    rasterize_tri_adaptive(tri, vis_id, screen_width);
}
