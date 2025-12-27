// PBR shader with texture support and hemisphere lighting
// Supports albedo, metallic-roughness, normal maps, and hemisphere light

const PI: f32 = 3.14159265359;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _padding: f32,
    // Hemisphere light: sky.rgb = sky color, sky.w = enabled (1.0 = on)
    hemisphere_sky: vec4<f32>,
    // Hemisphere light: ground.rgb = ground color, ground.w = intensity
    hemisphere_ground: vec4<f32>,
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
    use_albedo_map: f32,
    use_normal_map: f32,
    use_metallic_roughness_map: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> model: ModelUniform;

@group(2) @binding(0)
var<uniform> material: MaterialUniform;

@group(3) @binding(0)
var albedo_texture: texture_2d<f32>;
@group(3) @binding(1)
var albedo_sampler: sampler;
@group(3) @binding(2)
var normal_texture: texture_2d<f32>;
@group(3) @binding(3)
var normal_sampler: sampler;
@group(3) @binding(4)
var metallic_roughness_texture: texture_2d<f32>;
@group(3) @binding(5)
var metallic_roughness_sampler: sampler;

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
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model.model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = normalize(model.normal * in.normal);
    out.uv = in.uv;

    // Calculate tangent and bitangent for normal mapping
    // This is a simplified version - proper tangent should come from vertex data
    let n = out.world_normal;
    var t = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(dot(n, t)) > 0.9) {
        t = vec3<f32>(0.0, 1.0, 0.0);
    }
    out.tangent = normalize(t - n * dot(n, t));
    out.bitangent = cross(n, out.tangent);

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

// Calculate hemisphere light contribution based on world normal
fn calculate_hemisphere_light(normal: vec3<f32>) -> vec3<f32> {
    // Check if hemisphere light is enabled
    if (camera.hemisphere_sky.w < 0.5) {
        return vec3<f32>(0.0);
    }

    let sky_color = camera.hemisphere_sky.rgb;
    let ground_color = camera.hemisphere_ground.rgb;
    let intensity = camera.hemisphere_ground.w;

    // Map normal.y from [-1, 1] to [0, 1] for blending
    let blend = normal.y * 0.5 + 0.5;

    // Interpolate between ground and sky color
    let hemisphere_color = mix(ground_color, sky_color, blend);

    return hemisphere_color * intensity;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample textures or use material values
    var albedo = material.base_color.rgb;
    if (material.use_albedo_map > 0.5) {
        albedo = textureSample(albedo_texture, albedo_sampler, in.uv).rgb;
    }

    var metallic = material.metallic;
    var roughness = material.roughness;
    if (material.use_metallic_roughness_map > 0.5) {
        let mr = textureSample(metallic_roughness_texture, metallic_roughness_sampler, in.uv);
        // glTF convention: G = roughness, B = metallic
        roughness = mr.g;
        metallic = mr.b;
    }
    roughness = max(roughness, 0.04);

    var n = normalize(in.world_normal);
    if (material.use_normal_map > 0.5) {
        let tangent_normal = textureSample(normal_texture, normal_sampler, in.uv).rgb * 2.0 - 1.0;
        let tbn = mat3x3<f32>(
            normalize(in.tangent),
            normalize(in.bitangent),
            n
        );
        n = normalize(tbn * tangent_normal);
    }

    let ao = material.ao;
    let v = normalize(camera.position - in.world_position);

    // Calculate reflectance at normal incidence (F0)
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);

    // Light 1 - Key light (stronger for HDR output)
    let light1_pos = vec3<f32>(2.0, 4.0, 2.0);
    let light1_color = vec3<f32>(1.0, 0.95, 0.9);
    let light1_intensity = 8.0;
    lo += calculate_light(in.world_position, n, v, f0, albedo, metallic, roughness,
                          light1_pos, light1_color, light1_intensity);

    // Light 2 - Fill light
    let light2_pos = vec3<f32>(-3.0, 2.0, -1.0);
    let light2_color = vec3<f32>(0.6, 0.7, 1.0);
    let light2_intensity = 4.0;
    lo += calculate_light(in.world_position, n, v, f0, albedo, metallic, roughness,
                          light2_pos, light2_color, light2_intensity);

    // Light 3 - Back/rim light
    let light3_pos = vec3<f32>(0.0, 2.0, -3.0);
    let light3_color = vec3<f32>(0.9, 0.9, 1.0);
    let light3_intensity = 3.0;
    lo += calculate_light(in.world_position, n, v, f0, albedo, metallic, roughness,
                          light3_pos, light3_color, light3_intensity);

    // Ambient lighting (flat ambient + hemisphere gradient)
    let flat_ambient = vec3<f32>(0.08);
    let hemisphere_ambient = calculate_hemisphere_light(n);
    let ambient = (flat_ambient + hemisphere_ambient) * albedo * ao;

    var color = ambient + lo;

    // Output HDR values directly - tonemapping is done in post-processing
    // This allows values > 1.0 to trigger bloom effect
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
    kd *= 1.0 - metallic;

    let n_dot_l = max(dot(n, l), 0.0);

    return (kd * albedo / PI + specular) * radiance * n_dot_l;
}
