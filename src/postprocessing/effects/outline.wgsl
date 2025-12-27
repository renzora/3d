// Outline shader using depth-based edge detection
// Uses Sobel operator on depth buffer to detect edges

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct OutlineUniforms {
    // x: thickness, y: depth_threshold, z: normal_threshold, w: mode (0=depth, 1=normal, 2=combined)
    params: vec4<f32>,
    // x: r, y: g, z: b, w: unused
    color: vec4<f32>,
    // x: width, y: height, z: 1/width, w: 1/height
    resolution: vec4<f32>,
    // x: near, y: far, z: unused, w: unused
    camera: vec4<f32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var depth_sampler: sampler;
@group(0) @binding(4) var<uniform> uniforms: OutlineUniforms;

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

// Linearize depth from 0-1 range to view space distance
fn linearize_depth(depth: f32) -> f32 {
    let near = uniforms.camera.x;
    let far = uniforms.camera.y;
    return near * far / (far - depth * (far - near));
}

// Sample depth at UV offset
fn sample_depth(uv: vec2<f32>) -> f32 {
    let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleLevel(depth_texture, depth_sampler, clamped_uv, 0i);
}

// Sobel edge detection on depth
fn sobel_depth(uv: vec2<f32>, thickness: f32) -> f32 {
    let texel = vec2<f32>(uniforms.resolution.z, uniforms.resolution.w) * thickness;

    // Sample 3x3 neighborhood
    let tl = linearize_depth(sample_depth(uv + vec2<f32>(-texel.x, -texel.y)));
    let t  = linearize_depth(sample_depth(uv + vec2<f32>(0.0, -texel.y)));
    let tr = linearize_depth(sample_depth(uv + vec2<f32>(texel.x, -texel.y)));
    let l  = linearize_depth(sample_depth(uv + vec2<f32>(-texel.x, 0.0)));
    let c  = linearize_depth(sample_depth(uv));
    let r  = linearize_depth(sample_depth(uv + vec2<f32>(texel.x, 0.0)));
    let bl = linearize_depth(sample_depth(uv + vec2<f32>(-texel.x, texel.y)));
    let b  = linearize_depth(sample_depth(uv + vec2<f32>(0.0, texel.y)));
    let br = linearize_depth(sample_depth(uv + vec2<f32>(texel.x, texel.y)));

    // Sobel operators
    let gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    let gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);

    // Magnitude normalized by center depth to make it scale-independent
    let magnitude = sqrt(gx * gx + gy * gy);
    return magnitude / max(c, 0.001);
}

// Roberts cross edge detection (simpler, faster alternative)
fn roberts_depth(uv: vec2<f32>, thickness: f32) -> f32 {
    let texel = vec2<f32>(uniforms.resolution.z, uniforms.resolution.w) * thickness;

    let c  = linearize_depth(sample_depth(uv));
    let r  = linearize_depth(sample_depth(uv + vec2<f32>(texel.x, 0.0)));
    let b  = linearize_depth(sample_depth(uv + vec2<f32>(0.0, texel.y)));
    let br = linearize_depth(sample_depth(uv + vec2<f32>(texel.x, texel.y)));

    let gx = c - br;
    let gy = r - b;

    let magnitude = sqrt(gx * gx + gy * gy);
    return magnitude / max(c, 0.001);
}

// Reconstruct normal from depth (for normal-based edge detection)
fn get_normal_from_depth(uv: vec2<f32>) -> vec3<f32> {
    let texel = vec2<f32>(uniforms.resolution.z, uniforms.resolution.w);

    let c = linearize_depth(sample_depth(uv));
    let l = linearize_depth(sample_depth(uv - vec2<f32>(texel.x, 0.0)));
    let r = linearize_depth(sample_depth(uv + vec2<f32>(texel.x, 0.0)));
    let t = linearize_depth(sample_depth(uv - vec2<f32>(0.0, texel.y)));
    let b = linearize_depth(sample_depth(uv + vec2<f32>(0.0, texel.y)));

    let dx = (r - l) * 0.5;
    let dy = (b - t) * 0.5;

    return normalize(vec3<f32>(-dx, -dy, 1.0));
}

// Normal-based edge detection
fn normal_edge(uv: vec2<f32>, thickness: f32) -> f32 {
    let texel = vec2<f32>(uniforms.resolution.z, uniforms.resolution.w) * thickness;

    let c = get_normal_from_depth(uv);
    let l = get_normal_from_depth(uv - vec2<f32>(texel.x, 0.0));
    let r = get_normal_from_depth(uv + vec2<f32>(texel.x, 0.0));
    let t = get_normal_from_depth(uv - vec2<f32>(0.0, texel.y));
    let b = get_normal_from_depth(uv + vec2<f32>(0.0, texel.y));

    // Compare dot products with center normal
    let diff_l = 1.0 - dot(c, l);
    let diff_r = 1.0 - dot(c, r);
    let diff_t = 1.0 - dot(c, t);
    let diff_b = 1.0 - dot(c, b);

    return max(max(diff_l, diff_r), max(diff_t, diff_b));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let thickness = uniforms.params.x;
    let depth_threshold = uniforms.params.y;
    let normal_threshold = uniforms.params.z;
    let mode = uniforms.params.w;

    let color = textureSampleLevel(input_texture, tex_sampler, in.uv, 0.0);
    let outline_color = vec3<f32>(uniforms.color.x, uniforms.color.y, uniforms.color.z);

    // Check if we're at far plane (skybox)
    let depth = sample_depth(in.uv);
    if (depth >= 0.9999) {
        return color;
    }

    var edge = 0.0;

    if (mode < 0.5) {
        // Depth only
        let depth_edge = sobel_depth(in.uv, thickness);
        edge = step(depth_threshold, depth_edge);
    } else if (mode < 1.5) {
        // Normal only
        let normal_edge_val = normal_edge(in.uv, thickness);
        edge = step(normal_threshold, normal_edge_val);
    } else {
        // Combined
        let depth_edge = sobel_depth(in.uv, thickness);
        let normal_edge_val = normal_edge(in.uv, thickness);

        let depth_factor = step(depth_threshold, depth_edge);
        let normal_factor = step(normal_threshold, normal_edge_val);

        edge = max(depth_factor, normal_factor);
    }

    // Blend outline with original color
    let result = mix(color.rgb, outline_color, edge);

    return vec4<f32>(result, color.a);
}
