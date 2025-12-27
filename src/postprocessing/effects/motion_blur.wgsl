// Motion blur shader
// Computes per-pixel velocity from depth and view-projection matrices

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct MotionBlurUniforms {
    prev_view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // x: intensity, y: max_blur, z: velocity_scale, w: sample_count
    params: vec4<f32>,
    // x: width, y: height, z: 1/width, w: 1/height
    resolution: vec4<f32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var depth_sampler: sampler;
@group(0) @binding(4) var<uniform> uniforms: MotionBlurUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Full-screen triangle (covers -1 to 3 range, clipped to screen)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return out;
}

// Reconstruct world position from depth
fn world_position_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert UV to NDC
    let ndc = vec4<f32>(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,
        depth,
        1.0
    );

    // Transform to world space
    let world_pos = uniforms.inv_view_proj * ndc;
    return world_pos.xyz / world_pos.w;
}

// Project world position to screen space using previous frame's matrix
fn project_to_prev_frame(world_pos: vec3<f32>) -> vec2<f32> {
    let prev_clip = uniforms.prev_view_proj * vec4<f32>(world_pos, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;

    // Convert from NDC to UV
    return vec2<f32>(
        (prev_ndc.x + 1.0) * 0.5,
        1.0 - (prev_ndc.y + 1.0) * 0.5
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let intensity = uniforms.params.x;
    let max_blur = uniforms.params.y;
    let velocity_scale = uniforms.params.z;
    let sample_count = i32(uniforms.params.w);

    // Early out if intensity is zero
    if (intensity <= 0.0) {
        return textureSampleLevel(input_texture, tex_sampler, in.uv, 0.0);
    }

    // Sample depth (depth textures require integer mip level)
    let depth = textureSampleLevel(depth_texture, depth_sampler, in.uv, 0i);

    // Skip pixels at far plane (skybox, etc.)
    if (depth >= 1.0) {
        return textureSampleLevel(input_texture, tex_sampler, in.uv, 0.0);
    }

    // Reconstruct world position
    let world_pos = world_position_from_depth(in.uv, depth);

    // Project to previous frame
    let prev_uv = project_to_prev_frame(world_pos);

    // Calculate velocity (in UV space)
    var velocity = (in.uv - prev_uv) * velocity_scale * intensity;

    // Clamp velocity magnitude to max_blur (in pixels, then convert to UV)
    let velocity_pixels = velocity * uniforms.resolution.xy;
    let velocity_length = length(velocity_pixels);

    if (velocity_length > max_blur) {
        velocity = velocity * (max_blur / velocity_length);
    }

    // Skip if velocity is too small
    if (length(velocity * uniforms.resolution.xy) < 0.5) {
        return textureSampleLevel(input_texture, tex_sampler, in.uv, 0.0);
    }

    // Accumulate samples along velocity direction
    var color = vec4<f32>(0.0);
    let step = velocity / f32(sample_count);

    for (var i = 0; i < sample_count; i++) {
        let offset = f32(i) - f32(sample_count - 1) * 0.5;
        let sample_uv = in.uv + step * offset;

        // Clamp to valid UV range
        let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        color += textureSampleLevel(input_texture, tex_sampler, clamped_uv, 0.0);
    }

    color /= f32(sample_count);

    return color;
}
