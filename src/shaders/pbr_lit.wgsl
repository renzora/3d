// PBR shader with dynamic lighting support
// Supports up to 16 dynamic lights (point, directional, spot)
// Also supports hemisphere lighting for sky/ground gradient ambient

const PI: f32 = 3.14159265359;
const MAX_LIGHTS: u32 = 16u;

// Light types
const LIGHT_POINT: u32 = 0u;
const LIGHT_DIRECTIONAL: u32 = 1u;
const LIGHT_SPOT: u32 = 2u;

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
    light_type: u32,
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    range: f32,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    _padding: vec2<f32>,
}

struct LightsUniform {
    ambient: vec4<f32>,
    // Hemisphere light: sky.rgb = sky color, sky.w = enabled (1.0 = on, 0.0 = off)
    hemisphere_sky: vec4<f32>,
    // Hemisphere light: ground.rgb = ground color, ground.w = intensity
    hemisphere_ground: vec4<f32>,
    num_lights: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    lights: array<Light, 16>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> model: ModelUniform;

@group(2) @binding(0)
var<uniform> material: MaterialUniform;

@group(3) @binding(0)
var<uniform> lights_data: LightsUniform;

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

// Calculate attenuation for point/spot lights
fn calculate_attenuation(distance: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 1.0; // No attenuation for directional
    }
    // Smooth falloff
    let attenuation = clamp(1.0 - pow(distance / range, 4.0), 0.0, 1.0);
    return attenuation * attenuation / (distance * distance + 1.0);
}

// Calculate spot light cone attenuation
fn calculate_spot_attenuation(light_dir: vec3<f32>, spot_dir: vec3<f32>, inner_cos: f32, outer_cos: f32) -> f32 {
    let cos_angle = dot(light_dir, spot_dir);
    return clamp((cos_angle - outer_cos) / (inner_cos - outer_cos), 0.0, 1.0);
}

// Calculate hemisphere light contribution based on world normal
// Surfaces facing up (normal.y = 1) get sky color
// Surfaces facing down (normal.y = -1) get ground color
// Horizontal surfaces get a blend
fn calculate_hemisphere_light(normal: vec3<f32>) -> vec3<f32> {
    // Check if hemisphere light is enabled
    if (lights_data.hemisphere_sky.w < 0.5) {
        return vec3<f32>(0.0);
    }

    let sky_color = lights_data.hemisphere_sky.rgb;
    let ground_color = lights_data.hemisphere_ground.rgb;
    let intensity = lights_data.hemisphere_ground.w;

    // Map normal.y from [-1, 1] to [0, 1] for blending
    // normal.y = 1 (facing up) -> blend = 1 (full sky)
    // normal.y = -1 (facing down) -> blend = 0 (full ground)
    let blend = normal.y * 0.5 + 0.5;

    // Interpolate between ground and sky color
    let hemisphere_color = mix(ground_color, sky_color, blend);

    return hemisphere_color * intensity;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = material.base_color.rgb;
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04);
    let ao = material.ao;

    let n = normalize(in.world_normal);
    let v = normalize(camera.position - in.world_position);

    // Calculate F0 (reflectance at normal incidence)
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Accumulate light contribution
    var lo = vec3<f32>(0.0);

    for (var i = 0u; i < min(lights_data.num_lights, MAX_LIGHTS); i++) {
        let light = lights_data.lights[i];

        var l: vec3<f32>;
        var attenuation: f32 = 1.0;

        if (light.light_type == LIGHT_DIRECTIONAL) {
            // Directional light
            l = normalize(-light.direction);
        } else if (light.light_type == LIGHT_POINT) {
            // Point light
            let light_vec = light.position - in.world_position;
            let distance = length(light_vec);
            l = normalize(light_vec);
            attenuation = calculate_attenuation(distance, light.range);
        } else {
            // Spot light
            let light_vec = light.position - in.world_position;
            let distance = length(light_vec);
            l = normalize(light_vec);
            attenuation = calculate_attenuation(distance, light.range);
            attenuation *= calculate_spot_attenuation(-l, normalize(light.direction),
                                                       light.inner_cone_cos, light.outer_cone_cos);
        }

        let h = normalize(v + l);
        let radiance = light.color * light.intensity * attenuation;

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
        lo += (kd * albedo / PI + specular) * radiance * n_dot_l;
    }

    // Ambient lighting (flat ambient + hemisphere gradient)
    let flat_ambient = lights_data.ambient.rgb;
    let hemisphere_ambient = calculate_hemisphere_light(n);
    let ambient = (flat_ambient + hemisphere_ambient) * albedo * ao;

    var color = ambient + lo;

    // HDR tonemapping (Reinhard)
    color = color / (color + vec3<f32>(1.0));

    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
