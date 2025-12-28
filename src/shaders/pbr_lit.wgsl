// PBR shader with dynamic lighting and shadow support
// Supports up to 16 dynamic lights (point, directional, spot)
// Supports shadow mapping with PCF soft shadows
// Also supports hemisphere lighting for sky/ground gradient ambient

const PI: f32 = 3.14159265359;
const MAX_LIGHTS: u32 = 16u;
const MAX_SHADOW_LIGHTS: u32 = 4u;
const MAX_CASCADES: u32 = 4u;

// Light types
const LIGHT_POINT: u32 = 0u;
const LIGHT_DIRECTIONAL: u32 = 1u;
const LIGHT_SPOT: u32 = 2u;

// PCF modes
const PCF_NONE: u32 = 0u;
const PCF_HARDWARE_2X2: u32 = 1u;
const PCF_SOFT_3X3: u32 = 2u;
const PCF_SOFT_5X5: u32 = 3u;
const PCF_POISSON: u32 = 4u;

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
    // Render mode: 0=Lit, 1=Unlit, 2=Normals, 3=Depth, 4=Metallic, 5=Roughness, 6=AO, 7=UVs
    render_mode: u32,
    _pad0: u32,
    _pad1: u32,
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

// Shadow uniform data
struct ShadowUniform {
    // Light-space matrices for each shadow-casting light
    light_matrices: array<mat4x4<f32>, 4>,
    // Cascade matrices for directional light CSM
    cascade_matrices: array<mat4x4<f32>, 4>,
    // Cascade split distances
    cascade_splits: vec4<f32>,
    // Per-light shadow config: x=bias, y=normal_bias, z=map_index, w=light_type
    shadow_config: array<vec4<f32>, 4>,
    // Global config: x=num_shadow_lights, y=pcf_mode, z=map_size, w=num_cascades
    global_config: vec4<f32>,
}

@group(4) @binding(0)
var<uniform> shadow_data: ShadowUniform;

@group(4) @binding(1)
var shadow_maps: binding_array<texture_depth_2d, 4>;

@group(4) @binding(2)
var cascade_maps: binding_array<texture_depth_2d, 4>;

@group(4) @binding(3)
var shadow_sampler: sampler_comparison;

// Poisson disk samples for PCF
const POISSON_DISK: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>(0.94558609, -0.76890725),
    vec2<f32>(-0.094184101, -0.92938870),
    vec2<f32>(0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543, 0.27676845),
    vec2<f32>(0.97484398, 0.75648379),
    vec2<f32>(0.44323325, -0.97511554),
    vec2<f32>(0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>(0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507),
    vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>(0.19984126, 0.78641367),
    vec2<f32>(0.14383161, -0.14100790)
);

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

// Get cascade index based on view-space depth
fn get_cascade_index(view_z: f32) -> u32 {
    let num_cascades = u32(shadow_data.global_config.w);
    for (var i = 0u; i < num_cascades; i++) {
        if (view_z < shadow_data.cascade_splits[i]) {
            return i;
        }
    }
    return num_cascades - 1u;
}

// Sample shadow map with PCF filtering
fn sample_shadow_pcf(shadow_coord: vec3<f32>, map_index: u32, is_cascade: bool, bias: f32) -> f32 {
    let pcf_mode = u32(shadow_data.global_config.y);
    let map_size = shadow_data.global_config.z;
    let texel_size = 1.0 / map_size;
    let compare_depth = shadow_coord.z - bias;

    // Clamp to valid shadow map region
    if (shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
        shadow_coord.y < 0.0 || shadow_coord.y > 1.0 ||
        shadow_coord.z < 0.0 || shadow_coord.z > 1.0) {
        return 1.0; // Outside shadow frustum - fully lit
    }

    var shadow = 0.0;

    if (pcf_mode == PCF_NONE || pcf_mode == PCF_HARDWARE_2X2) {
        // Single sample or hardware 2x2 PCF
        if (is_cascade) {
            shadow = textureSampleCompare(cascade_maps[map_index], shadow_sampler, shadow_coord.xy, compare_depth);
        } else {
            shadow = textureSampleCompare(shadow_maps[map_index], shadow_sampler, shadow_coord.xy, compare_depth);
        }
    } else if (pcf_mode == PCF_SOFT_3X3) {
        // 3x3 PCF kernel (9 samples)
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
                if (is_cascade) {
                    shadow += textureSampleCompare(cascade_maps[map_index], shadow_sampler, shadow_coord.xy + offset, compare_depth);
                } else {
                    shadow += textureSampleCompare(shadow_maps[map_index], shadow_sampler, shadow_coord.xy + offset, compare_depth);
                }
            }
        }
        shadow /= 9.0;
    } else if (pcf_mode == PCF_SOFT_5X5) {
        // 5x5 PCF kernel (25 samples)
        for (var y = -2; y <= 2; y++) {
            for (var x = -2; x <= 2; x++) {
                let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
                if (is_cascade) {
                    shadow += textureSampleCompare(cascade_maps[map_index], shadow_sampler, shadow_coord.xy + offset, compare_depth);
                } else {
                    shadow += textureSampleCompare(shadow_maps[map_index], shadow_sampler, shadow_coord.xy + offset, compare_depth);
                }
            }
        }
        shadow /= 25.0;
    } else {
        // Poisson disk sampling (16 samples)
        for (var i = 0u; i < 16u; i++) {
            let offset = POISSON_DISK[i] * texel_size * 2.0;
            if (is_cascade) {
                shadow += textureSampleCompare(cascade_maps[map_index], shadow_sampler, shadow_coord.xy + offset, compare_depth);
            } else {
                shadow += textureSampleCompare(shadow_maps[map_index], shadow_sampler, shadow_coord.xy + offset, compare_depth);
            }
        }
        shadow /= 16.0;
    }

    return shadow;
}

// Calculate shadow factor for a light
fn calculate_shadow(
    light_index: u32,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    view_z: f32,
) -> f32 {
    // Check if this light has shadows
    let num_shadow_lights = u32(shadow_data.global_config.x);
    if (light_index >= num_shadow_lights) {
        return 1.0; // No shadow for this light
    }

    let config = shadow_data.shadow_config[light_index];
    let bias = config.x;
    let normal_bias = config.y;
    let map_index = i32(config.z);
    let light_type = u32(config.w);

    if (map_index < 0) {
        return 1.0; // Shadow disabled for this light
    }

    // Apply normal bias (offset position along normal)
    let biased_position = world_position + world_normal * normal_bias;

    var light_matrix: mat4x4<f32>;
    var is_cascade = false;
    var cascade_index = 0u;

    // For directional lights, use cascade selection
    if (light_type == LIGHT_DIRECTIONAL) {
        cascade_index = get_cascade_index(view_z);
        light_matrix = shadow_data.cascade_matrices[cascade_index];
        is_cascade = true;
    } else {
        light_matrix = shadow_data.light_matrices[light_index];
    }

    // Transform to light space
    let light_space_pos = light_matrix * vec4<f32>(biased_position, 1.0);

    // Perspective divide and convert to texture coordinates
    var shadow_coord = light_space_pos.xyz / light_space_pos.w;
    shadow_coord.x = shadow_coord.x * 0.5 + 0.5;
    shadow_coord.y = shadow_coord.y * -0.5 + 0.5; // Flip Y for texture coords

    if (is_cascade) {
        return sample_shadow_pcf(shadow_coord, cascade_index, true, bias);
    } else {
        return sample_shadow_pcf(shadow_coord, u32(map_index), false, bias);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = material.base_color.rgb;
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04);
    let ao = material.ao;

    let n = normalize(in.world_normal);
    let v = normalize(camera.position - in.world_position);

    // Calculate view-space depth for cascade selection
    let view_z = length(camera.position - in.world_position);

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

        // Calculate shadow factor (1.0 = fully lit, 0.0 = fully shadowed)
        var shadow_factor = 1.0;
        if (i < MAX_SHADOW_LIGHTS) {
            shadow_factor = calculate_shadow(i, in.world_position, n, view_z);
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
        // Apply shadow factor to direct lighting
        lo += (kd * albedo / PI + specular) * radiance * n_dot_l * shadow_factor;
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
