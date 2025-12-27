// PBR (Physically Based Rendering) shader
// Uses metallic-roughness workflow with GGX/Schlick BRDF

const PI: f32 = 3.14159265359;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
}

struct ModelUniform {
    model: mat4x4<f32>,
    normal: mat3x3<f32>,
}

struct MaterialUniform {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    _padding: f32,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
    intensity: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> model: ModelUniform;

@group(2) @binding(0)
var<uniform> material: MaterialUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model.model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = normalize(model.normal * in.normal);
    out.uv = in.uv;
    return out;
}

// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// GGX/Trowbridge-Reitz normal distribution function
fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;

    let num = a2;
    var denom = (n_dot_h2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

// Smith's Schlick-GGX geometry function
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let num = n_dot_v;
    let denom = n_dot_v * (1.0 - k) + k;
    return num / denom;
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx1 * ggx2;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = material.base_color.rgb;
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04); // Prevent divide by zero
    let ao = material.ao;

    let n = normalize(in.world_normal);
    let v = normalize(camera.position - in.world_position);

    // Calculate reflectance at normal incidence (F0)
    // Dielectrics use 0.04, metals use albedo color
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Hardcoded lights for now (will be dynamic later)
    var lo = vec3<f32>(0.0);

    // Light 1 - Key light
    let light1_pos = vec3<f32>(2.0, 4.0, 2.0);
    let light1_color = vec3<f32>(1.0, 0.95, 0.9);
    let light1_intensity = 5.0;
    lo += calculate_light(in.world_position, n, v, f0, albedo, metallic, roughness,
                          light1_pos, light1_color, light1_intensity);

    // Light 2 - Fill light
    let light2_pos = vec3<f32>(-3.0, 2.0, -1.0);
    let light2_color = vec3<f32>(0.6, 0.7, 1.0);
    let light2_intensity = 2.0;
    lo += calculate_light(in.world_position, n, v, f0, albedo, metallic, roughness,
                          light2_pos, light2_color, light2_intensity);

    // Ambient lighting (simple approximation)
    let ambient = vec3<f32>(0.03) * albedo * ao;

    var color = ambient + lo;

    // HDR tonemapping (Reinhard)
    color = color / (color + vec3<f32>(1.0));

    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}

fn calculate_light(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    f0: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    light_pos: vec3<f32>,
    light_color: vec3<f32>,
    light_intensity: f32,
) -> vec3<f32> {
    let l = normalize(light_pos - world_pos);
    let h = normalize(v + l);
    let distance = length(light_pos - world_pos);
    let attenuation = 1.0 / (distance * distance);
    let radiance = light_color * light_intensity * attenuation;

    // Cook-Torrance BRDF
    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let numerator = ndf * g * f;
    let denominator = 4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0) + 0.0001;
    let specular = numerator / denominator;

    // Energy conservation
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic; // Metals have no diffuse

    let n_dot_l = max(dot(n, l), 0.0);

    return (kd * albedo / PI + specular) * radiance * n_dot_l;
}
