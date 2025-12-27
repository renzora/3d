// Standard shader with MVP transforms and basic lighting

struct CameraUniform {
    view_proj: mat4x4<f32>,
}

struct ModelUniform {
    model: mat4x4<f32>,
    normal: mat3x3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> model: ModelUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_position: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_pos = model.model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = model.normal * in.normal;
    out.uv = in.uv;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple directional lighting
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let normal = normalize(in.world_normal);

    let ambient = 0.2;
    let diffuse = max(dot(normal, light_dir), 0.0);
    let lighting = ambient + diffuse * 0.8;

    // Base color (can be replaced with texture sampling later)
    let base_color = vec3<f32>(0.8, 0.8, 0.8);

    return vec4<f32>(base_color * lighting, 1.0);
}
