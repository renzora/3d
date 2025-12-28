// Skybox shader - renders a cubemap as background
// The skybox is rendered at maximum depth (behind everything)

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _padding: f32,
    hemisphere_sky: vec4<f32>,
    hemisphere_ground: vec4<f32>,
}

// Inverse view-projection for reconstructing world direction from clip space
struct SkyboxUniform {
    inv_view_proj: mat4x4<f32>,
    exposure: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> skybox: SkyboxUniform;

@group(1) @binding(0)
var skybox_texture: texture_cube<f32>;
@group(1) @binding(1)
var skybox_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_dir: vec3<f32>,
}

// Full-screen triangle vertices
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate full-screen triangle
    // Vertices: (-1, -1), (3, -1), (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    // Position at far plane (z = 1.0 in NDC, which maps to max depth)
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0);

    // Calculate world direction from clip position
    let clip_pos = vec4<f32>(x, y, 1.0, 1.0);
    let world_pos = skybox.inv_view_proj * clip_pos;
    out.world_dir = normalize(world_pos.xyz / world_pos.w);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample cubemap using world direction
    let color = textureSample(skybox_texture, skybox_sampler, in.world_dir).rgb;

    // Apply exposure
    let exposed = color * skybox.exposure;

    return vec4<f32>(exposed, 1.0);
}
