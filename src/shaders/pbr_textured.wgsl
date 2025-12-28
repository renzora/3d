// PBR shader with texture support, hemisphere lighting, and shadow mapping
// Supports albedo, metallic-roughness, normal maps, hemisphere light, and directional shadows

const PI: f32 = 3.14159265359;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _padding: f32,
    // Hemisphere light: sky.rgb = sky color, sky.w = enabled (1.0 = on)
    hemisphere_sky: vec4<f32>,
    // Hemisphere light: ground.rgb = ground color, ground.w = intensity
    hemisphere_ground: vec4<f32>,
    // IBL settings: x=diffuse intensity, y=specular intensity, z=unused, w=unused
    ibl_settings: vec4<f32>,
    // Scene lights (4 lights, each with position.xyz + intensity.w, color.rgb + enabled.w)
    light0_pos: vec4<f32>,  // xyz=position, w=intensity
    light0_color: vec4<f32>, // rgb=color, w=enabled
    light1_pos: vec4<f32>,
    light1_color: vec4<f32>,
    light2_pos: vec4<f32>,
    light2_color: vec4<f32>,
    light3_pos: vec4<f32>,
    light3_color: vec4<f32>,
}

struct ShadowUniform {
    // Light-space matrix for shadow mapping
    light_view_proj: mat4x4<f32>,
    // Shadow settings: x=bias, y=normal_bias, z=enabled, w=light_type (0=dir, 1=spot, 2=point)
    shadow_params: vec4<f32>,
    // Light direction (xyz) for directional, or light position (xyz) for spot/point
    light_dir_or_pos: vec4<f32>,
    // Shadow map size (x=width, y=height, z=1/width, w=1/height)
    shadow_map_size: vec4<f32>,
    // Spot light direction (xyz) and range (w)
    spot_direction: vec4<f32>,
    // Spot light params: x=outer_cos, y=inner_cos, z=intensity, w=unused
    spot_params: vec4<f32>,
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

// Shadow mapping (merged into group 3)
@group(3) @binding(6)
var<uniform> shadow: ShadowUniform;
@group(3) @binding(7)
var shadow_map: texture_depth_2d;
@group(3) @binding(8)
var shadow_sampler: sampler_comparison;
@group(3) @binding(9)
var shadow_cube_map: texture_depth_cube;
@group(3) @binding(10)
var shadow_cube_sampler: sampler_comparison;

// Environment map for IBL (Image-Based Lighting)
@group(3) @binding(11)
var env_map: texture_cube<f32>;
@group(3) @binding(12)
var env_sampler: sampler;
// BRDF Look-Up Table for split-sum approximation
@group(3) @binding(13)
var brdf_lut: texture_2d<f32>;
@group(3) @binding(14)
var brdf_sampler: sampler;

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

// Fresnel-Schlick with roughness for IBL
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let smooth_factor = max(vec3<f32>(1.0 - roughness), f0);
    return f0 + (smooth_factor - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Sample environment map for IBL specular reflections
fn sample_env_specular(reflect_dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    // Use mip level based on roughness (crude approximation without prefiltered maps)
    // For a proper implementation, we'd need a prefiltered environment map
    let mip_level = roughness * 4.0; // Adjust based on cubemap mip levels
    let env_color = textureSampleLevel(env_map, env_sampler, reflect_dir, mip_level).rgb;
    return env_color;
}

// Sample environment map for IBL diffuse irradiance
fn sample_env_diffuse(normal: vec3<f32>) -> vec3<f32> {
    // For diffuse, sample at highest mip level (most blurred)
    // This is a crude approximation - proper IBL uses a precomputed irradiance map
    let mip_level = 4.0; // Use highest mip for maximum blur
    let env_color = textureSampleLevel(env_map, env_sampler, normal, mip_level).rgb;
    return env_color;
}

// Calculate IBL contribution (diffuse + specular) using split-sum approximation
fn calculate_ibl(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    // Get IBL intensity from camera uniform
    let ibl_diffuse_intensity = camera.ibl_settings.x;
    let ibl_specular_intensity = camera.ibl_settings.y;

    let n_dot_v = max(dot(normal, view_dir), 0.001);

    // Fresnel term for IBL - used for energy balance
    let f = fresnel_schlick_roughness(n_dot_v, f0, roughness);

    // Specular and diffuse weights
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic; // Metallic surfaces have no diffuse

    // IBL Diffuse - controlled by ibl_diffuse_intensity
    let irradiance = sample_env_diffuse(normal);
    let diffuse = irradiance * albedo * ibl_diffuse_intensity;

    // IBL Specular using BRDF LUT (split-sum approximation)
    let reflect_dir = reflect(-view_dir, normal);
    let prefiltered_color = sample_env_specular(reflect_dir, roughness);

    // Sample BRDF LUT: x = NdotV, y = roughness
    // Returns vec2(scale, bias) for Fresnel term
    let brdf_uv = vec2<f32>(n_dot_v, roughness);
    let brdf = textureSample(brdf_lut, brdf_sampler, brdf_uv).rg;

    // Split-sum approximation: Lo = prefilteredColor * (F0 * scale + bias)
    // This properly integrates the BRDF over the hemisphere
    let specular = prefiltered_color * (f0 * brdf.x + brdf.y) * ibl_specular_intensity;

    // Combine diffuse and specular with AO
    let ambient = (kd * diffuse + specular) * ao;

    return ambient;
}

// Calculate shadow factor for directional/spot lights (2D shadow map)
fn calculate_shadow_2d(world_pos: vec3<f32>, normal: vec3<f32>, to_light: vec3<f32>) -> f32 {
    // Transform to light space
    let light_space_pos = shadow.light_view_proj * vec4<f32>(world_pos, 1.0);

    // Perspective divide
    let proj_coords_raw = light_space_pos.xyz / light_space_pos.w;

    // Transform from [-1, 1] to [0, 1] for texture sampling
    let proj_x = proj_coords_raw.x * 0.5 + 0.5;
    let proj_y = proj_coords_raw.y * -0.5 + 0.5; // Flip Y for texture coordinates
    let proj_z = proj_coords_raw.z;

    let bias = shadow.shadow_params.x;
    let normal_bias = shadow.shadow_params.y;

    // Apply normal bias based on angle to light
    let cos_angle = max(dot(normal, to_light), 0.0);
    let slope_bias = bias + normal_bias * (1.0 - cos_angle);
    let compare_depth = proj_z - slope_bias;

    // Clamp UV to valid range for sampling
    let uv = clamp(vec2<f32>(proj_x, proj_y), vec2<f32>(0.001), vec2<f32>(0.999));

    // Sample shadow map with 3x3 PCF
    let texel_size = shadow.shadow_map_size.zw;
    var shadow_factor = 0.0;

    // Unrolled 3x3 PCF for uniform control flow
    shadow_factor += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(-1.0, -1.0) * texel_size, compare_depth);
    shadow_factor += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(0.0, -1.0) * texel_size, compare_depth);
    shadow_factor += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(1.0, -1.0) * texel_size, compare_depth);
    shadow_factor += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(-1.0, 0.0) * texel_size, compare_depth);
    shadow_factor += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(0.0, 0.0) * texel_size, compare_depth);
    shadow_factor += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(1.0, 0.0) * texel_size, compare_depth);
    shadow_factor += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(-1.0, 1.0) * texel_size, compare_depth);
    shadow_factor += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(0.0, 1.0) * texel_size, compare_depth);
    shadow_factor += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(1.0, 1.0) * texel_size, compare_depth);
    shadow_factor /= 9.0;

    // Check bounds and shadows enabled - multiply result instead of early return
    let in_bounds = step(0.0, proj_x) * step(proj_x, 1.0) * step(0.0, proj_y) * step(proj_y, 1.0) * step(0.0, proj_z) * step(proj_z, 1.0);

    return mix(1.0, shadow_factor, in_bounds);
}

// Calculate shadow factor for point lights (cube shadow map)
fn calculate_shadow_cube(world_pos: vec3<f32>, normal: vec3<f32>, light_pos: vec3<f32>) -> f32 {
    // Direction from light to fragment
    let light_to_frag = world_pos - light_pos;
    let dist = length(light_to_frag);
    let dir = normalize(light_to_frag);

    // Get range from spot_direction.w
    let range = shadow.spot_direction.w;

    // Normalize distance to [0, 1] range for comparison
    let normalized_depth = dist / range;

    let bias = shadow.shadow_params.x * 2.0; // Cube maps need more bias
    let compare_depth = normalized_depth - bias;

    // Sample cube shadow map - use the direction to sample
    // The cube map stores normalized depth values
    let shadow_factor = textureSampleCompare(shadow_cube_map, shadow_cube_sampler, dir, compare_depth);

    // Check if within range
    let in_range = step(normalized_depth, 1.0);

    return mix(1.0, shadow_factor, in_range);
}

// Calculate shadow factor based on light type
fn calculate_shadow(world_pos: vec3<f32>, normal: vec3<f32>, to_light: vec3<f32>, light_type: f32) -> f32 {
    let shadows_on = step(0.5, shadow.shadow_params.z);

    if (shadows_on < 0.5) {
        return 1.0;
    }

    // Use cube shadow for point lights, 2D shadow for directional/spot
    if (light_type > 1.5) {
        // Point light - use cube shadow map
        let light_pos = shadow.light_dir_or_pos.xyz;
        return calculate_shadow_cube(world_pos, normal, light_pos);
    } else {
        // Directional or spot - use 2D shadow map
        return calculate_shadow_2d(world_pos, normal, to_light);
    }
}

// Calculate shadow-casting light contribution (directional or spot)
fn calculate_shadow_light(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    f0: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    // Check if shadow light is enabled
    let shadows_enabled = shadow.shadow_params.z;
    if (shadows_enabled < 0.5) {
        return vec3<f32>(0.0);
    }

    let light_type = shadow.shadow_params.w; // 0=directional, 1=spot, 2=point

    var light_dir: vec3<f32>;
    var attenuation: f32 = 1.0;
    var spot_effect: f32 = 1.0;

    if (light_type < 0.5) {
        // Directional light: direction is constant
        light_dir = normalize(-shadow.light_dir_or_pos.xyz);
    } else if (light_type < 1.5) {
        // Spot light: direction from light position to fragment
        let light_pos = shadow.light_dir_or_pos.xyz;
        let to_frag = world_pos - light_pos;
        let dist = length(to_frag);
        light_dir = normalize(-to_frag);

        // Distance attenuation
        let range = shadow.spot_direction.w;
        attenuation = clamp(1.0 - dist / range, 0.0, 1.0);
        attenuation *= attenuation; // Quadratic falloff

        // Spot cone falloff
        let spot_dir = normalize(shadow.spot_direction.xyz);
        let outer_cos = shadow.spot_params.x;
        let inner_cos = shadow.spot_params.y;
        let cos_angle = dot(-light_dir, spot_dir);
        spot_effect = clamp((cos_angle - outer_cos) / (inner_cos - outer_cos), 0.0, 1.0);
    } else {
        // Point light: direction from light position to fragment (no shadow support yet)
        let light_pos = shadow.light_dir_or_pos.xyz;
        let to_frag = world_pos - light_pos;
        let dist = length(to_frag);
        light_dir = normalize(-to_frag);

        // Distance attenuation
        let range = shadow.spot_direction.w;
        attenuation = clamp(1.0 - dist / range, 0.0, 1.0);
        attenuation *= attenuation;
    }

    let h = normalize(v + light_dir);

    // Light color and intensity
    let light_color = vec3<f32>(1.0, 0.98, 0.95);
    let light_intensity = 3.0;
    let radiance = light_color * light_intensity * attenuation * spot_effect;

    // Cook-Torrance BRDF
    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, light_dir, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let numerator = ndf * g * f;
    let denominator = 4.0 * max(dot(n, v), 0.0) * max(dot(n, light_dir), 0.0) + 0.0001;
    let specular = numerator / denominator;

    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;

    let n_dot_l = max(dot(n, light_dir), 0.0);

    // Get shadow factor
    let shadow_factor = calculate_shadow(world_pos, n, light_dir, light_type);

    return (kd * albedo / PI + specular) * radiance * n_dot_l * shadow_factor;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample textures or use material values
    var albedo = material.base_color.rgb;
    var alpha = material.base_color.a;
    if (material.use_albedo_map > 0.5) {
        let albedo_sample = textureSample(albedo_texture, albedo_sampler, in.uv);
        albedo = albedo_sample.rgb;
        alpha = albedo_sample.a * material.base_color.a;
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

    // Shadow-casting light (directional, spot, or point)
    lo += calculate_shadow_light(in.world_position, n, v, f0, albedo, metallic, roughness);

    // Light 0 - Key light (configurable)
    if (camera.light0_color.w > 0.5) {
        lo += calculate_light(in.world_position, n, v, f0, albedo, metallic, roughness,
                              camera.light0_pos.xyz, camera.light0_color.rgb, camera.light0_pos.w);
    }

    // Light 1 - Fill light (configurable)
    if (camera.light1_color.w > 0.5) {
        lo += calculate_light(in.world_position, n, v, f0, albedo, metallic, roughness,
                              camera.light1_pos.xyz, camera.light1_color.rgb, camera.light1_pos.w);
    }

    // Light 2 - Rim light (configurable)
    if (camera.light2_color.w > 0.5) {
        lo += calculate_light(in.world_position, n, v, f0, albedo, metallic, roughness,
                              camera.light2_pos.xyz, camera.light2_color.rgb, camera.light2_pos.w);
    }

    // Light 3 - Extra light (configurable)
    if (camera.light3_color.w > 0.5) {
        lo += calculate_light(in.world_position, n, v, f0, albedo, metallic, roughness,
                              camera.light3_pos.xyz, camera.light3_color.rgb, camera.light3_pos.w);
    }

    // IBL (Image-Based Lighting) from environment map
    let ibl_ambient = calculate_ibl(n, v, albedo, metallic, roughness, ao, f0);

    // Hemisphere light (additive, for additional ambient gradient)
    let hemisphere_ambient = calculate_hemisphere_light(n) * albedo * ao * 0.3;

    // Flat ambient to prevent dark areas (especially interiors)
    // Higher value helps when normals face away from main lights
    let flat_ambient = vec3<f32>(0.15) * albedo * ao;

    var color = flat_ambient + hemisphere_ambient + ibl_ambient + lo;

    // Alpha cutoff - discard fully transparent pixels (for decals, stickers, etc.)
    if (alpha < 0.1) {
        discard;
    }

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
