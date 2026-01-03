// Nanite HZB (Hierarchical Z-Buffer) Build Compute Shader
// Downsamples depth buffer to create mip chain for occlusion culling
// Each texel stores the maximum (furthest) depth of its region

struct HzbBuildUniform {
    src_size: vec2<u32>,  // Source mip dimensions
    dst_size: vec2<u32>,  // Destination mip dimensions
}

@group(0) @binding(0) var<uniform> uniforms: HzbBuildUniform;
@group(0) @binding(1) var src_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture: texture_storage_2d<r32float, write>;

// Sample depth with bounds checking
fn sample_depth(coord: vec2<i32>, size: vec2<u32>) -> f32 {
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    return textureLoad(src_texture, clamped, 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_coord = vec2<u32>(global_id.xy);

    // Check bounds
    if (dst_coord.x >= uniforms.dst_size.x || dst_coord.y >= uniforms.dst_size.y) {
        return;
    }

    // Calculate source coordinates (2x2 block)
    let src_coord = vec2<i32>(dst_coord) * 2;

    // Sample 2x2 block from source
    let d00 = sample_depth(src_coord + vec2<i32>(0, 0), uniforms.src_size);
    let d10 = sample_depth(src_coord + vec2<i32>(1, 0), uniforms.src_size);
    let d01 = sample_depth(src_coord + vec2<i32>(0, 1), uniforms.src_size);
    let d11 = sample_depth(src_coord + vec2<i32>(1, 1), uniforms.src_size);

    // Take maximum depth (furthest from camera in reversed-Z)
    // For reversed-Z depth buffer: 1.0 = near, 0.0 = far
    // We want the minimum value (furthest point) for conservative occlusion
    // But if using standard depth: 0.0 = near, 1.0 = far
    // We want the maximum value (furthest point)
    //
    // For this implementation, we use standard depth convention (0=near, 1=far)
    // so we take the maximum to get the furthest depth in the region
    let max_depth = max(max(d00, d10), max(d01, d11));

    // Write to destination mip
    textureStore(dst_texture, vec2<i32>(dst_coord), vec4<f32>(max_depth, 0.0, 0.0, 0.0));
}
