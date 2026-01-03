// PBR shader with texture support, hemisphere lighting, and shadow mapping
// Supports albedo, metallic-roughness, normal maps, hemisphere light, and directional shadows

const PI: f32 = 3.14159265359;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    // Render mode: 0=Lit, 1=Unlit, 2=Normals, 3=Depth, 4=Metallic, 5=Roughness, 6=AO, 7=UVs
    render_mode: u32,
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
    // Detail mapping: x=enabled (0/1), y=scale (UV tiling), z=intensity, w=max_distance
    detail_settings: vec4<f32>,
    // Detail albedo: x=enabled (0/1), y=scale, z=intensity, w=blend_mode (0=overlay, 1=multiply, 2=soft_light)
    detail_albedo_settings: vec4<f32>,
    // Rect light 0 (rectangular area light)
    // xyz=position, w=enabled (1.0 = on)
    rectlight0_pos: vec4<f32>,
    // xyz=direction (normal), w=width
    rectlight0_dir_width: vec4<f32>,
    // xyz=tangent, w=height
    rectlight0_tan_height: vec4<f32>,
    // rgb=color, w=intensity
    rectlight0_color: vec4<f32>,
    // Rect light 1
    rectlight1_pos: vec4<f32>,
    rectlight1_dir_width: vec4<f32>,
    rectlight1_tan_height: vec4<f32>,
    rectlight1_color: vec4<f32>,
    // Capsule light 0 (tube/line light)
    // xyz=start, w=enabled (1.0 = on)
    capsule0_start: vec4<f32>,
    // xyz=end, w=radius
    capsule0_end_radius: vec4<f32>,
    // rgb=color, w=intensity
    capsule0_color: vec4<f32>,
    // Capsule light 1
    capsule1_start: vec4<f32>,
    capsule1_end_radius: vec4<f32>,
    capsule1_color: vec4<f32>,
    // Disk light 0 (circular area light)
    // xyz=position, w=enabled (1.0 = on)
    disk0_pos: vec4<f32>,
    // xyz=direction (normal), w=radius
    disk0_dir_radius: vec4<f32>,
    // rgb=color, w=intensity
    disk0_color: vec4<f32>,
    // Disk light 1
    disk1_pos: vec4<f32>,
    disk1_dir_radius: vec4<f32>,
    disk1_color: vec4<f32>,
    // Sphere light 0 (spherical area light)
    // xyz=position, w=enabled (1.0 = on)
    sphere0_pos: vec4<f32>,
    // x=radius, y=range
    sphere0_radius_range: vec4<f32>,
    // rgb=color, w=intensity
    sphere0_color: vec4<f32>,
    // Sphere light 1
    sphere1_pos: vec4<f32>,
    sphere1_radius_range: vec4<f32>,
    sphere1_color: vec4<f32>,
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
    // Spot light params: x=outer_cos, y=inner_cos, z=intensity, w=pcf_mode
    spot_params: vec4<f32>,
    // PCSS params: x=light_size, y=near_plane, z=blocker_search_radius, w=max_filter_radius
    pcss_params: vec4<f32>,
    // Contact shadow params: x=enabled, y=max_distance, z=thickness, w=intensity
    contact_params: vec4<f32>,
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
    clear_coat: f32,
    clear_coat_roughness: f32,
    sheen: f32,
    use_albedo_map: f32,
    use_normal_map: f32,
    use_metallic_roughness_map: f32,
    sheen_color: vec3<f32>,
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

// Irradiance map for diffuse IBL (pre-convolved for hemisphere integration)
@group(3) @binding(15)
var irradiance_map: texture_cube<f32>;
@group(3) @binding(16)
var irradiance_sampler: sampler;

// Detail normal map for micro-surface detail (tiling noise texture)
@group(3) @binding(17)
var detail_normal_map: texture_2d<f32>;
@group(3) @binding(18)
var detail_normal_sampler: sampler;

// Detail albedo map for micro-surface color variation (grayscale noise texture)
@group(3) @binding(19)
var detail_albedo_map: texture_2d<f32>;
@group(3) @binding(20)
var detail_albedo_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) barycentric: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
    @location(5) barycentric: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model.model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = normalize(model.normal * in.normal);
    out.uv = in.uv;
    out.barycentric = in.barycentric;

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

// Fresnel-Schlick approximation with Optimized shadow compensation
// Anything less than 2% reflectance is physically impossible and is instead considered shadowing
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let fc = pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    // The saturate(50.0 * f0.g) term compensates for very dark F0 values (optimized technique)
    return saturate(50.0 * f0.g) * fc + (1.0 - fc) * f0;
}

// Disney Burley Diffuse BRDF (optimized technique)
// Provides roughness-dependent diffuse that looks more realistic than Lambert
fn diffuse_burley(diffuse_color: vec3<f32>, roughness: f32, n_dot_v: f32, n_dot_l: f32, v_dot_h: f32) -> vec3<f32> {
    let fd90 = 0.5 + 2.0 * v_dot_h * v_dot_h * roughness;
    let fd_v = 1.0 + (fd90 - 1.0) * pow(1.0 - n_dot_v, 5.0);
    let fd_l = 1.0 + (fd90 - 1.0) * pow(1.0 - n_dot_l, 5.0);
    return diffuse_color * (1.0 / PI) * fd_v * fd_l;
}

// ============================================================================
//Rough Diffuse BRDF Models
// Energy-preserving diffuse that properly handles microfacet interactions
// ============================================================================

// [Portsmouth et al. 2025, "EON: A Practical Energy-Preserving Rough Diffuse BRDF"]
// Energy-preserving rough diffuse with multi-scattering
fn diffuse_eon(diffuse_color: vec3<f32>, roughness: f32, n_dot_v: f32, n_dot_l: f32, v_dot_l: f32) -> vec3<f32> {
    // Albedo inversion for EON model to maintain consistent color with lambert
    let rho = diffuse_color * (1.0 + (0.189468 - 0.189468 * diffuse_color) * roughness);

    // Main shaping term from Oren-Nayar model (with tweaks by Fujii)
    let s = v_dot_l - n_dot_v * n_dot_l;
    let s_over_t = max(s / max(0.000001, max(n_dot_v, n_dot_l)), s);

    // AF approximation (nearly a straight line)
    let constant1_fon = 0.5 - 2.0 / (3.0 * PI);
    let af = 1.0 - roughness * (1.0 - 1.0 / (1.0 + constant1_fon));
    let f_ss = af * (1.0 + roughness * s_over_t);

    // First order approximation for multi-scattering
    let g1 = 0.262048;
    let g_over_pi_v = g1 - g1 * n_dot_v;
    // Use (1 - Eo) as non-reciprocal approach to energy conservation
    let f_ms = 1.0 - af * (1.0 + roughness * g_over_pi_v);

    // The Rho_ms term can be approximated as just Rho^2
    return rho * (f_ss + rho * f_ms) * (1.0 / PI);
}

// [Chan 2024, "Multiscattering Diffuse and Specular BRDFs"]
// GGX-based rough diffuse with multiscattering approximation
// retro_weight: 1.0 for point/directional lights, reduced for area lights to avoid artifacts
fn diffuse_ggx_rough(
    diffuse_color: vec3<f32>,
    roughness: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    v_dot_h: f32,
    n_dot_h: f32,
    retro_weight: f32
) -> vec3<f32> {
    // Saturate inputs to avoid negative values from interpolation
    let nov = saturate(n_dot_v);
    let nol = saturate(n_dot_l);
    let voh = saturate(v_dot_h);
    let noh = saturate(n_dot_h);

    // Apply retro-reflectivity weight to roughness
    let r = roughness * retro_weight;
    let alpha = r * r;

    // Chan 2024 multiscattering model
    // FSmooth = 1 (energy balance handled externally)
    let f_smooth = 1.0;
    let scale = max(0.55 - 0.2 * r, 1.25 - 1.6 * r);
    let bias = saturate(4.0 * alpha);
    let f_rough = scale * (noh + bias) / (noh + 0.025) * voh * voh;
    let diffuse_ss = mix(f_smooth, f_rough, r);
    let diffuse_ms = alpha * 0.38;

    return (1.0 / PI) * diffuse_color * (diffuse_ss + diffuse_ms);
}

// [Chan 2018, "Material Advances in Call of Duty: WWII"]
// Alternative rough diffuse with retro-reflectivity
// retro_weight: reduces retro-reflection for area lights to avoid artifacts
fn diffuse_cod_wwii(
    diffuse_color: vec3<f32>,
    roughness: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    v_dot_h: f32,
    n_dot_h: f32,
    retro_weight: f32
) -> vec3<f32> {
    let nov = saturate(n_dot_v);
    let nol = saturate(n_dot_l);
    let voh = saturate(v_dot_h);
    let noh = saturate(n_dot_h);

    let a2 = roughness * roughness * roughness * roughness; // Pow4
    // g = saturate((1/18) * log2(2 / a2 - 1))
    let g = saturate((1.0 / 18.0) * log2(2.0 / max(a2, 0.0001) - 1.0));

    // Fresnel-like term
    let f0 = voh + pow(1.0 - voh, 5.0);
    let fd_v = 1.0 - 0.75 * pow(1.0 - nov, 5.0);
    let fd_l = 1.0 - 0.75 * pow(1.0 - nol, 5.0);

    // Rough to smooth response interpolation
    let fd = mix(f0, fd_v * fd_l, saturate(2.2 * g - 0.5));

    // Retro-reflectivity contribution (fades out for area lights)
    var fb = ((34.5 * g - 59.0) * g + 24.5) * voh * exp2(-max(73.2 * g - 21.2, 8.9) * sqrt(noh));
    fb *= retro_weight;

    var lobe = (1.0 / PI) * (fd + fb);
    // Clamp to avoid too bright edges with normal maps at high roughness
    lobe = min(1.0, lobe);

    return diffuse_color * lobe;
}

// Blend detail normal with base normal using UDN (Unity Derivative Normal) blending
// This produces higher quality results than simple linear blending
fn blend_normals_udn(base: vec3<f32>, detail: vec3<f32>) -> vec3<f32> {
    // UDN blending: base.xy + detail.xy, base.z * detail.z
    let blended = vec3<f32>(
        base.xy + detail.xy,
        base.z * detail.z
    );
    return normalize(blended);
}

// Sample and apply detail normal map
// Returns the modified normal in tangent space
fn apply_detail_normal(
    base_normal: vec3<f32>,
    world_pos: vec3<f32>,
    uv: vec2<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
) -> vec3<f32> {
    // Get settings
    let enabled = camera.detail_settings.x;
    let scale = camera.detail_settings.y;      // UV tiling scale
    let intensity = camera.detail_settings.z;  // Blend intensity
    let max_distance = camera.detail_settings.w; // Fade distance

    // Always sample the texture first (required for uniform control flow)
    let detail_uv = uv * scale;
    let detail_sample = textureSample(detail_normal_map, detail_normal_sampler, detail_uv).rgb;

    // Calculate distance from camera for LOD fade
    let view_distance = length(camera.position - world_pos);
    let distance_fade = saturate(1.0 - view_distance / max_distance);

    // Compute effective blend factor (0 if disabled or too far)
    let blend_factor = enabled * intensity * distance_fade;

    // If no blending needed, return base normal
    if (blend_factor <= 0.001) {
        return base_normal;
    }

    // Convert from [0,1] to [-1,1] range
    let detail_normal_raw = detail_sample * 2.0 - 1.0;

    // Blend detail normal with flat normal based on blend factor
    let detail_normal = mix(vec3<f32>(0.0, 0.0, 1.0), detail_normal_raw, blend_factor);

    // Build TBN matrix for the base normal
    let tbn = mat3x3<f32>(
        normalize(tangent),
        normalize(bitangent),
        base_normal
    );

    // Transform detail normal from tangent space and blend with base
    // The detail normal perturbs the base normal in tangent space
    let blended = blend_normals_udn(
        vec3<f32>(0.0, 0.0, 1.0), // Base as "up" in tangent space
        detail_normal
    );

    // Transform blended result back to world space
    return normalize(tbn * blended);
}

// Overlay blend - standard Photoshop overlay (adds contrast/variation)
fn blend_overlay(base: f32, detail: f32) -> f32 {
    if (base < 0.5) {
        return 2.0 * base * detail;
    } else {
        return 1.0 - 2.0 * (1.0 - base) * (1.0 - detail);
    }
}

// Soft light blend - gentler effect than overlay
fn blend_soft_light(base: f32, detail: f32) -> f32 {
    if (detail < 0.5) {
        return base - (1.0 - 2.0 * detail) * base * (1.0 - base);
    } else {
        var d: f32;
        if (base < 0.25) {
            d = ((16.0 * base - 12.0) * base + 4.0) * base;
        } else {
            d = sqrt(base);
        }
        return base + (2.0 * detail - 1.0) * (d - base);
    }
}

// Apply detail albedo map for micro-surface color variation
fn apply_detail_albedo(
    base_color: vec3<f32>,
    world_pos: vec3<f32>,
    uv: vec2<f32>,
) -> vec3<f32> {
    // Get settings
    let enabled = camera.detail_albedo_settings.x;
    let scale = camera.detail_albedo_settings.y;
    let intensity = camera.detail_albedo_settings.z;
    let blend_mode = u32(camera.detail_albedo_settings.w);

    // Always sample the texture first (required for uniform control flow)
    let detail_uv = uv * scale;
    let detail = textureSample(detail_albedo_map, detail_albedo_sampler, detail_uv).r;

    // Calculate distance from camera for LOD fade
    let max_distance = camera.detail_settings.w; // Share max_distance with detail normal
    let view_distance = length(camera.position - world_pos);
    let distance_fade = saturate(1.0 - view_distance / max_distance);

    // Compute effective blend factor (0 if disabled or too far)
    let blend_factor = enabled * intensity * distance_fade;

    // If no blending needed, return base color
    if (blend_factor <= 0.001) {
        return base_color;
    }

    // Apply blend mode
    var result: vec3<f32>;
    switch (blend_mode) {
        case 1u: { // Multiply
            // Multiply blend: detail centered at 0.5, so scale to [0, 2] range
            result = base_color * mix(vec3<f32>(1.0), vec3<f32>(detail * 2.0), blend_factor);
        }
        case 2u: { // Soft Light
            result = vec3<f32>(
                mix(base_color.r, blend_soft_light(base_color.r, detail), blend_factor),
                mix(base_color.g, blend_soft_light(base_color.g, detail), blend_factor),
                mix(base_color.b, blend_soft_light(base_color.b, detail), blend_factor),
            );
        }
        default: { // Overlay (0)
            result = vec3<f32>(
                mix(base_color.r, blend_overlay(base_color.r, detail), blend_factor),
                mix(base_color.g, blend_overlay(base_color.g, detail), blend_factor),
                mix(base_color.b, blend_overlay(base_color.b, detail), blend_factor),
            );
        }
    }
    return result;
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

// Optimized optimized Smith Joint Approximate visibility function
// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
// This is faster and produces very similar results to the full Smith function
fn vis_smith_joint_approx(a2: f32, n_dot_v: f32, n_dot_l: f32) -> f32 {
    let a = sqrt(a2);
    let vis_smith_v = n_dot_l * (n_dot_v * (1.0 - a) + a);
    let vis_smith_l = n_dot_v * (n_dot_l * (1.0 - a) + a);
    return 0.5 / (vis_smith_v + vis_smith_l + 0.0001);
}

// ============================================================================
//Area Light Energy Normalization
// From ShadingModels.ush - de Carpentier 2017 "Decima Engine" approach
// Prevents blown-out specular highlights from large area light sources
// ============================================================================

// Compute modified roughness squared for area lights
// sin_alpha: sin of half-angle subtended by the light source
// v_dot_h: dot product of view and half vector
fn new_a2(a2: f32, sin_alpha: f32, v_dot_h: f32) -> f32 {
    //formulation from ShadingModels.ush
    return a2 + 0.25 * sin_alpha * (3.0 * sqrt(a2) + sin_alpha) / (v_dot_h + 0.001);
}

//Energy normalization for area lights
// Adjusts roughness and returns energy multiplier to prevent over-brightening
// sphere_sin_alpha: sin of half-angle subtended by sphere (radius/distance)
// sphere_sin_alpha_soft: soft version for roughness modification (can be 0)
// line_cos_subtended: cos(2*alpha) for line lights, 1.0 for non-line lights
fn energy_normalization(
    a2: f32,
    v_dot_h: f32,
    sphere_sin_alpha: f32,
    sphere_sin_alpha_soft: f32,
    line_cos_subtended: f32,
) -> vec2<f32> {
    // Start with input roughness
    var modified_a2 = a2;

    // Apply soft sphere roughness modification
    if (sphere_sin_alpha_soft > 0.0) {
        modified_a2 = saturate(modified_a2 + sphere_sin_alpha_soft * sphere_sin_alpha_soft / (v_dot_h * 3.6 + 0.4));
    }

    // Compute sphere energy normalization
    var sphere_a2 = modified_a2;
    var energy = 1.0;

    if (sphere_sin_alpha > 0.0) {
        sphere_a2 = new_a2(modified_a2, sphere_sin_alpha, v_dot_h);
        energy = modified_a2 / max(sphere_a2, 0.00001);
    }

    // Compute line energy normalization (for capsule/tube lights)
    if (line_cos_subtended < 1.0) {
        let line_cos_two_alpha = line_cos_subtended;
        let line_tan_alpha = sqrt((1.0001 - line_cos_two_alpha) / (1.0 + line_cos_two_alpha));
        let line_a2 = new_a2(sphere_a2, line_tan_alpha, v_dot_h);
        energy *= sqrt(sphere_a2 / max(line_a2, 0.00001));
    }

    // Return modified a2 and energy multiplier
    return vec2<f32>(sphere_a2, energy);
}

// Simplified energy normalization for sphere/disk lights (no line component)
fn energy_normalization_sphere(a2: f32, v_dot_h: f32, sin_alpha: f32) -> vec2<f32> {
    return energy_normalization(a2, v_dot_h, sin_alpha, sin_alpha * 0.5, 1.0);
}

// Energy normalization for capsule/tube lights (sphere + line components)
fn energy_normalization_capsule(
    a2: f32,
    v_dot_h: f32,
    sphere_sin_alpha: f32,
    line_cos_subtended: f32
) -> vec2<f32> {
    return energy_normalization(a2, v_dot_h, sphere_sin_alpha, sphere_sin_alpha * 0.5, line_cos_subtended);
}

// Energy normalization for rect lights (use smaller dimension as sphere)
fn energy_normalization_rect(a2: f32, v_dot_h: f32, sin_alpha: f32) -> vec2<f32> {
    return energy_normalization(a2, v_dot_h, sin_alpha, sin_alpha * 0.5, 1.0);
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

// ============================================================================
//Clear Coat Shading Functions
// From ClearCoatCommon.ush and ShadingModels.ush
// Two-layer model: clear coat (polyurethane, IOR=1.5) over base material
// ============================================================================

// Clear coat constants
const CLEAR_COAT_IOR: f32 = 1.5;
const CLEAR_COAT_F0: f32 = 0.04;  // ((1.5 - 1) / (1.5 + 1))^2 ≈ 0.04

// Fresnel for clear coat layer (scalar, since F0 is constant 0.04)
fn fresnel_schlick_clear_coat(cos_theta: f32) -> f32 {
    let fc = pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    return CLEAR_COAT_F0 + (1.0 - CLEAR_COAT_F0) * fc;
}

// Simple clear coat transmittance (absorption through the clear coat layer)
//SimpleClearCoatTransmittance from ClearCoatCommon.ush
// color_weight: controls absorption color (0 = no absorption)
fn clear_coat_transmittance(n_dot_v: f32, clear_coat: f32, color_weight: vec3<f32>) -> vec3<f32> {
    // Path length through the clear coat layer
    let path_length = 1.0 / max(n_dot_v, 0.001);

    // Simple absorption model (Beer-Lambert law)
    // More path length = more absorption
    let absorption = mix(vec3<f32>(1.0), color_weight, clear_coat * (1.0 - exp(-path_length * 0.5)));

    return absorption;
}

// Calculate refracted dot products for clear coat
//RefractClearCoatContext from ClearCoatCommon.ush
// Returns the cosine of the refracted angle using Snell's law
fn refract_cos_theta(cos_theta: f32) -> f32 {
    // Snell's law: sin(θ1)/sin(θ2) = n2/n1
    // For IOR = 1.5, n2/n1 = 1/1.5 = 0.666...
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let sin_theta_refracted = sin_theta / CLEAR_COAT_IOR;
    let cos_theta_refracted = sqrt(max(0.0, 1.0 - sin_theta_refracted * sin_theta_refracted));
    return cos_theta_refracted;
}

// Clear coat specular BRDF (simplified GGX for the top layer)
fn clear_coat_specular(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    h: vec3<f32>,
    clear_coat_roughness: f32,
) -> f32 {
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // Use minimum roughness to avoid zero division
    let roughness = max(clear_coat_roughness, 0.01);
    let a = roughness * roughness;
    let a2 = a * a;

    // GGX NDF for clear coat
    let d = distribution_ggx(n, h, roughness);

    // Visibility term (simplified for clear coat)
    let vis = vis_smith_joint_approx(a2, n_dot_v, n_dot_l);

    // Fresnel for clear coat
    let f = fresnel_schlick_clear_coat(v_dot_h);

    return d * vis * f;
}

// Calculate complete clear coat contribution
// Returns: (clear_coat_specular, base_attenuation)
// - clear_coat_specular: the specular reflection from the clear coat layer
// - base_attenuation: how much light reaches the base layer (1 - Fresnel)
fn calculate_clear_coat(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    clear_coat: f32,
    clear_coat_roughness: f32,
) -> vec2<f32> {
    // Early out if no clear coat
    if (clear_coat < 0.001) {
        return vec2<f32>(0.0, 1.0);
    }

    let h = normalize(v + l);
    let v_dot_h = max(dot(v, h), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);

    // Clear coat specular contribution
    let cc_spec = clear_coat_specular(n, v, l, h, clear_coat_roughness);

    // Fresnel determines how much is reflected vs transmitted
    let f = fresnel_schlick_clear_coat(v_dot_h);

    // Base layer receives attenuated light (what wasn't reflected by clear coat)
    // Account for both incoming and outgoing Fresnel (squared)
    let base_attenuation = (1.0 - f * clear_coat) * (1.0 - f * clear_coat);

    return vec2<f32>(cc_spec * clear_coat * n_dot_l, base_attenuation);
}

// IBL contribution for clear coat layer
fn calculate_clear_coat_ibl(
    n: vec3<f32>,
    v: vec3<f32>,
    clear_coat: f32,
    clear_coat_roughness: f32,
) -> vec2<f32> {
    // Early out if no clear coat
    if (clear_coat < 0.001) {
        return vec2<f32>(0.0, 1.0);
    }

    let n_dot_v = max(dot(n, v), 0.0);

    // Fresnel for IBL (using NoV instead of VoH)
    let f = fresnel_schlick_clear_coat(n_dot_v);

    // Reflection direction for IBL sampling
    let r = reflect(-v, n);

    // Sample environment map with clear coat roughness mip level
    let mip_level = clear_coat_roughness * 4.0; // Adjust based on your mip count
    let prefilterd_color = textureSampleLevel(env_map, env_sampler, r, mip_level).rgb;

    // Get intensity from IBL settings
    let ibl_specular_intensity = camera.ibl_settings.y;

    // Clear coat IBL specular
    let cc_ibl = prefilterd_color * f * clear_coat * ibl_specular_intensity;

    // Luminance of clear coat IBL for energy conservation
    let cc_luminance = dot(cc_ibl, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Base attenuation (simplified for IBL)
    let base_attenuation = 1.0 - f * clear_coat;

    return vec2<f32>(cc_luminance, base_attenuation);
}

// ============================================================================
//Cloth/Sheen BRDF Functions
// From BRDF.ush - Two-lobe cloth model with asperity scattering
// ============================================================================

// Inverse GGX distribution for cloth (asperity scattering layer)
//D_InvGGX from BRDF.ush
fn d_inv_ggx(a2: f32, n_dot_h: f32) -> f32 {
    let a = 4.0;
    let d = (n_dot_h - a2 * n_dot_h) * n_dot_h + a2;
    return (1.0 / PI) * (1.0 / (1.0 + a * a2)) * (1.0 + 4.0 * a2 * a2 / (d * d));
}

// Cloth/Ashikhmin visibility function
//Vis_Cloth from BRDF.ush
fn vis_cloth(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return 1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v) + 0.0001);
}

// Charlie sheen distribution (Estevez-Kulla 2017)
//D_Charlie from BRDF.ush - softer falloff than GGX for fabric
fn d_charlie(roughness: f32, n_dot_h: f32) -> f32 {
    let inv_r = 1.0 / max(roughness, 0.001);
    let cos2h = n_dot_h * n_dot_h;
    let sin2h = 1.0 - cos2h;
    return (2.0 + inv_r) * pow(sin2h, inv_r * 0.5) / (2.0 * PI);
}

// Charlie visibility helper function
//Vis_Charlie_L from BRDF.ush
fn vis_charlie_l(x: f32, r: f32) -> f32 {
    let r_sat = saturate(r);
    let r2 = 1.0 - (1.0 - r_sat) * (1.0 - r_sat);

    let a = mix(25.3245, 21.5473, r2);
    let b = mix(3.32435, 3.82987, r2);
    let c = mix(0.16801, 0.19823, r2);
    let d = mix(-1.27393, -1.97760, r2);
    let e = mix(-4.85967, -4.32054, r2);

    return a / ((1.0 + b * pow(x, c)) + d * x + e);
}

// Charlie visibility function (Estevez-Kulla 2017)
//Vis_Charlie from BRDF.ush
fn vis_charlie(roughness: f32, n_dot_v: f32, n_dot_l: f32) -> f32 {
    let vis_v = select(
        exp(2.0 * vis_charlie_l(0.5, roughness) - vis_charlie_l(1.0 - n_dot_v, roughness)),
        exp(vis_charlie_l(n_dot_v, roughness)),
        n_dot_v < 0.5
    );
    let vis_l = select(
        exp(2.0 * vis_charlie_l(0.5, roughness) - vis_charlie_l(1.0 - n_dot_l, roughness)),
        exp(vis_charlie_l(n_dot_l, roughness)),
        n_dot_l < 0.5
    );

    return 1.0 / ((1.0 + vis_v + vis_l) * (4.0 * n_dot_v * n_dot_l) + 0.0001);
}

// Calculate cloth specular using Optimized two-lobe model
// Lerps between standard GGX and cloth sheen based on cloth amount
fn calculate_cloth_specular(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    h: vec3<f32>,
    roughness: f32,
    sheen: f32,
    sheen_color: vec3<f32>,
    f0: vec3<f32>,
) -> vec3<f32> {
    // Early out if no sheen
    if (sheen < 0.001) {
        return vec3<f32>(0.0);
    }

    let n_dot_v = max(dot(n, v), 0.001);
    let n_dot_l = max(dot(n, l), 0.001);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // Standard GGX specular (base layer)
    let a = roughness * roughness;
    let a2 = a * a;
    let d1 = distribution_ggx(n, h, roughness);
    let vis1 = vis_smith_joint_approx(a2, n_dot_v, n_dot_l);
    let f1 = fresnel_schlick(v_dot_h, f0);
    let spec1 = d1 * vis1 * f1;

    // Cloth sheen specular (asperity scattering layer)
    // Uses inverse GGX distribution and cloth visibility
    let cloth_roughness = pow(roughness, 4.0); //uses Pow4(roughness)
    let d2 = d_inv_ggx(cloth_roughness, n_dot_h);
    let vis2 = vis_cloth(n_dot_v, n_dot_l);
    let f2 = fresnel_schlick(v_dot_h, sheen_color);
    let spec2 = d2 * vis2 * f2;

    // Lerp between standard specular and cloth sheen
    return mix(spec1, spec2, sheen);
}

// Calculate cloth diffuse with energy conservation
// Returns attenuated diffuse based on sheen absorption
fn calculate_cloth_diffuse(
    diffuse_color: vec3<f32>,
    n_dot_l: f32,
    sheen: f32,
) -> vec3<f32> {
    // Cloth absorbs some light in the sheen layer
    // Simple approximation: reduce diffuse by sheen amount
    let diffuse_attenuation = 1.0 - sheen * 0.5;
    return diffuse_color * (1.0 / PI) * diffuse_attenuation;
}

// ============ Procedural Sky for IBL ============

// Hash functions for noise
fn sky_hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn sky_hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn sky_hash22(p: vec2<f32>) -> vec2<f32> {
    let n = sin(dot(p, vec2<f32>(41.0, 289.0)));
    return fract(vec2<f32>(262144.0, 32768.0) * n);
}

// 3D noise for clouds
fn sky_noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(
            mix(sky_hash31(i + vec3<f32>(0.0, 0.0, 0.0)), sky_hash31(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(sky_hash31(i + vec3<f32>(0.0, 1.0, 0.0)), sky_hash31(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(sky_hash31(i + vec3<f32>(0.0, 0.0, 1.0)), sky_hash31(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(sky_hash31(i + vec3<f32>(0.0, 1.0, 1.0)), sky_hash31(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

// FBM for clouds
fn sky_fbm(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = p;
    for (var i = 0; i < 5; i++) {
        value += amplitude * sky_noise3d(pos);
        amplitude *= 0.5;
        pos *= 2.0;
    }
    return value;
}

// Rayleigh phase function
fn sky_rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

// Mie phase function
fn sky_mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = (1.0 - g2);
    let denom = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return (1.0 / (4.0 * PI)) * num / denom;
}

// Atmospheric scattering
fn sky_atmosphere(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sun_intensity: f32) -> vec3<f32> {
    let cos_theta = dot(ray_dir, sun_dir);
    let y = ray_dir.y;

    // Rayleigh scattering coefficients (boosted blue)
    let beta_r = vec3<f32>(6.5e-6, 15.0e-6, 40.0e-6);
    // Mie scattering
    let beta_m = vec3<f32>(21e-6 * 2.0);

    // Optical depths
    let depth_r = exp(-max(y, 0.0) * 5.0) * 8500.0;
    let depth_m = exp(-max(y, 0.0) * 2.5) * 1200.0;

    let phase_r = sky_rayleigh_phase(cos_theta);
    let phase_m = sky_mie_phase(cos_theta, 0.8);

    let scatter_r = beta_r * phase_r * depth_r;
    let scatter_m = beta_m * phase_m * depth_m;

    let sun_color = vec3<f32>(1.0, 0.98, 0.92) * sun_intensity;
    var sky_color = (scatter_r + scatter_m) * sun_color;

    // Vibrant blue gradient
    let zenith_blue = vec3<f32>(0.15, 0.35, 0.85) * sun_intensity * 0.08;
    let blue_gradient = pow(max(0.0, y), 0.6);
    sky_color += zenith_blue * blue_gradient;

    // Warm horizon
    let horizon_factor = 1.0 - y;
    let horizon_warmth = pow(horizon_factor, 3.0) * 0.15;
    sky_color += vec3<f32>(0.9, 0.85, 0.75) * horizon_warmth * sun_intensity * 0.5;

    // Sunset colors
    let sun_height = sun_dir.y;
    if (sun_height < 0.3) {
        let sunset_factor = 1.0 - sun_height / 0.3;
        let horizon_glow = pow(horizon_factor, 4.0) * 0.4;
        sky_color += vec3<f32>(1.8, 0.6, 0.25) * sunset_factor * horizon_glow * sun_color * 0.5;
    }

    return sky_color;
}

// Stars
fn sky_stars(ray_dir: vec3<f32>, sun_height: f32) -> vec3<f32> {
    if (ray_dir.y < 0.0) { return vec3<f32>(0.0); }

    let star_visibility = smoothstep(0.1, -0.2, sun_height);
    if (star_visibility <= 0.0) { return vec3<f32>(0.0); }

    let theta = atan2(ray_dir.z, ray_dir.x);
    let phi = asin(ray_dir.y);
    let uv = vec2<f32>(theta, phi) * 80.0;
    let grid_id = floor(uv);
    let grid_uv = fract(uv) - 0.5;

    var star_color = vec3<f32>(0.0);
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let offset = vec2<f32>(f32(i), f32(j));
            let cell_id = grid_id + offset;
            let rand = sky_hash22(cell_id);
            let star_pos = offset + rand - 0.5;
            let d = length(grid_uv - star_pos);
            let brightness_rand = sky_hash21(cell_id * 1.31);

            if (brightness_rand > 0.92) {
                let star_size = 0.02 + brightness_rand * 0.03;
                var star_brightness = smoothstep(star_size, 0.0, d);
                star_brightness += smoothstep(star_size * 0.3, 0.0, d) * 2.0;
                star_color += star_brightness * vec3<f32>(1.0) * (0.5 + brightness_rand * 1.5);
            }
        }
    }
    return star_color * star_visibility;
}

// Moon
fn sky_moon(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let moon_dir = -sun_dir;
    if (sun_dir.y > 0.1) { return vec3<f32>(0.0); }

    let moon_visibility = smoothstep(0.1, -0.1, sun_dir.y);
    let cos_theta = dot(ray_dir, moon_dir);
    let moon_angular_radius = 0.009;

    var moon_color = vec3<f32>(0.0);
    let moon_cos = cos(moon_angular_radius);
    if (cos_theta > moon_cos) {
        let edge_factor = (cos_theta - moon_cos) / (1.0 - moon_cos);
        moon_color = vec3<f32>(0.95, 0.93, 0.88) * smoothstep(0.0, 1.0, edge_factor) * 2.0;
    }

    // Moon glow
    let glow_cos = cos(moon_angular_radius * 4.0);
    if (cos_theta > glow_cos) {
        let glow_factor = (cos_theta - glow_cos) / (moon_cos - glow_cos);
        moon_color += vec3<f32>(0.8, 0.85, 0.95) * pow(max(0.0, glow_factor), 2.0) * 0.3;
    }

    return moon_color * moon_visibility;
}

// Sun disk with glare
fn sky_sun(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sun_intensity: f32) -> vec3<f32> {
    let cos_theta = dot(ray_dir, sun_dir);
    let sun_angular_radius = 0.5 * 0.00873;

    var sun_color = vec3<f32>(0.0);

    // Core sun disk
    let core_cos = cos(sun_angular_radius);
    if (cos_theta > core_cos) {
        let edge_factor = (cos_theta - core_cos) / (1.0 - core_cos);
        sun_color += vec3<f32>(1.0, 0.99, 0.95) * smoothstep(0.0, 0.3, edge_factor) * sun_intensity * 0.5;
    }

    // Outer ring glare
    let ring_inner = sun_angular_radius;
    let ring_outer = sun_angular_radius * 2.5;
    if (cos_theta > cos(ring_outer) && cos_theta < cos(ring_inner)) {
        let ring_mid = (ring_inner + ring_outer) * 0.5;
        let angle_from_sun = acos(cos_theta);
        let ring_width = ring_outer - ring_inner;
        let dist_from_mid = abs(angle_from_sun - ring_mid) / (ring_width * 0.5);
        let ring_intensity = (1.0 - dist_from_mid * dist_from_mid) * sun_intensity * 0.2;
        sun_color += vec3<f32>(1.0, 0.97, 0.9) * max(0.0, ring_intensity);
    }

    // Soft glow
    let glow_cos = cos(sun_angular_radius * 6.0);
    if (cos_theta > glow_cos) {
        let glow_factor = (cos_theta - glow_cos) / (core_cos - glow_cos);
        sun_color += vec3<f32>(1.0, 0.95, 0.85) * pow(max(0.0, glow_factor), 3.0) * sun_intensity * 0.08;
    }

    return sun_color;
}

// Clouds
fn sky_clouds(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sun_intensity: f32) -> vec4<f32> {
    if (ray_dir.y < 0.05) { return vec4<f32>(0.0); }

    let cloud_height = 0.15;
    let t = cloud_height / max(ray_dir.y, 0.001);
    let cloud_pos = ray_dir * t;
    let cloud_uv = cloud_pos.xz * 3.0;

    var density = sky_fbm(vec3<f32>(cloud_uv.x, cloud_uv.y, 0.0));
    density = smoothstep(0.45, 0.75, density);
    density *= smoothstep(0.05, 0.2, ray_dir.y);
    density *= 1.0 - smoothstep(0.6, 1.0, ray_dir.y) * 0.3;

    if (density <= 0.0) { return vec4<f32>(0.0); }

    // Cloud lighting
    var cloud_color = vec3<f32>(1.0);
    let sun_dot = dot(ray_dir, sun_dir);
    let light_factor = sun_dot * 0.3 + 0.7;

    if (sun_dir.y < 0.3) {
        let sunset_t = 1.0 - sun_dir.y / 0.3;
        cloud_color = mix(cloud_color, vec3<f32>(1.0, 0.7, 0.4), sunset_t * 0.6);
    }

    cloud_color *= (light_factor - smoothstep(0.0, 0.5, density) * 0.3 + 0.3);
    cloud_color *= max(0.4, sun_dir.y * 0.6 + 0.6) * sun_intensity * 0.1;

    if (sun_dir.y < 0.0) {
        cloud_color *= smoothstep(-0.3, 0.0, sun_dir.y);
    }

    return vec4<f32>(cloud_color, density);
}

// Full procedural sky
fn procedural_sky_color(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_intensity = 22.0;
    var color: vec3<f32>;

    if (ray_dir.y < 0.0) {
        // Below horizon - ground reflection
        let ground_color = vec3<f32>(0.45, 0.40, 0.32);
        let ground_ambient = max(0.15, sun_dir.y * 0.4 + 0.5);
        let horizon_sky = sky_atmosphere(vec3<f32>(ray_dir.x, 0.01, ray_dir.z), sun_dir, sun_intensity);
        let horizon_blend = smoothstep(-0.15, 0.0, ray_dir.y);
        color = mix(ground_color * ground_ambient, horizon_sky, horizon_blend);
    } else {
        // Sky
        color = sky_atmosphere(ray_dir, sun_dir, sun_intensity);
        color += sky_stars(ray_dir, sun_dir.y);
        color += sky_moon(ray_dir, sun_dir);

        // Clouds
        let cloud = sky_clouds(ray_dir, sun_dir, sun_intensity);
        if (cloud.a > 0.0) {
            color = mix(color, cloud.rgb / max(cloud.a, 0.001), cloud.a * 0.9);
        }

        // Sun
        if (sun_dir.y > -0.1) {
            color += sky_sun(ray_dir, sun_dir, sun_intensity);
        }
    }

    return color;
}

// Sample prefiltered environment map for IBL specular reflections
// Each mip level is convolved with increasing roughness (GGX importance sampling)
fn sample_env_specular(reflect_dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    // Check if procedural sky is enabled (ibl_settings.z > 0)
    if (camera.ibl_settings.z > 0.5) {
        // Use full procedural sky - sun direction from shadow uniform
        let sun_dir = normalize(-shadow.light_dir_or_pos.xyz);
        let sky = procedural_sky_color(reflect_dir, sun_dir);
        // Scale by light intensity (spot_params.z) so reflections dim at night
        let light_intensity = shadow.spot_params.z;
        // Reduce reflection based on roughness - rough surfaces have very weak reflections
        let roughness_fade = (1.0 - roughness) * (1.0 - roughness); // Quadratic falloff
        return sky * 0.04 * light_intensity * roughness_fade;
    }

    // Mip 0 = mirror reflection (roughness 0)
    // Mip 4 = fully rough (roughness 1)
    let max_mip = 4.0;
    let mip_level = roughness * max_mip;
    let env_color = textureSampleLevel(env_map, env_sampler, reflect_dir, mip_level).rgb;
    return env_color;
}

// Sample irradiance map for IBL diffuse lighting
// The irradiance map is pre-convolved to represent the integral of incoming light
// over the hemisphere for each direction, giving proper diffuse IBL
fn sample_env_diffuse(normal: vec3<f32>) -> vec3<f32> {
    // Check if procedural sky is enabled (ibl_settings.z > 0)
    if (camera.ibl_settings.z > 0.5) {
        // Use procedural sky for diffuse - sun direction from shadow uniform
        let sun_dir = normalize(-shadow.light_dir_or_pos.xyz);
        let sky = procedural_sky_color(normal, sun_dir);
        // Scale by light intensity so ambient dims at night
        let light_intensity = shadow.spot_params.z;
        return sky * 0.03 * light_intensity;
    }

    // Sample the pre-convolved irradiance cubemap
    // Each texel contains the integral: E(n) = ∫_Ω L(ω) * max(0, n·ω) dω
    let irradiance = textureSample(irradiance_map, irradiance_sampler, normal).rgb;
    return irradiance;
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

// ============ PCSS (Percentage-Closer Soft Shadows) ============

// Poisson disk samples for randomized shadow sampling (16 samples)
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

// Interleaved gradient noise for sample rotation (avoids banding)
fn interleaved_gradient_noise_shadow(pos: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pos, magic.xy)));
}

// Rotate a 2D vector by angle
fn rotate_2d(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
}

// PCSS Step 1: Blocker search - find average blocker depth
fn pcss_blocker_search(uv: vec2<f32>, receiver_depth: f32, search_radius: f32, screen_pos: vec2<f32>) -> vec2<f32> {
    var blocker_sum = 0.0;
    var blocker_count = 0.0;

    let texel_size = shadow.shadow_map_size.zw;
    let rotation = interleaved_gradient_noise_shadow(screen_pos) * 6.28318530718; // 2*PI

    // Sample shadow map to find blockers (objects between light and receiver)
    for (var i = 0u; i < 16u; i++) {
        let offset = rotate_2d(POISSON_DISK[i], rotation) * search_radius * texel_size;
        let sample_uv = uv + offset;

        // Use textureLoad to get raw depth (need level 0)
        let sample_coord = vec2<i32>(sample_uv * shadow.shadow_map_size.xy);
        let clamped_coord = clamp(sample_coord, vec2<i32>(0), vec2<i32>(shadow.shadow_map_size.xy) - vec2<i32>(1));
        let blocker_depth = textureLoad(shadow_map, clamped_coord, 0);

        // If blocker is closer to light than receiver, it's a blocker
        if (blocker_depth < receiver_depth) {
            blocker_sum += blocker_depth;
            blocker_count += 1.0;
        }
    }

    // Return (average blocker depth, blocker count)
    if (blocker_count > 0.0) {
        return vec2<f32>(blocker_sum / blocker_count, blocker_count);
    }
    return vec2<f32>(-1.0, 0.0); // No blockers found
}

// PCSS Step 2: Estimate penumbra size based on blocker distance
fn pcss_estimate_penumbra(receiver_depth: f32, blocker_depth: f32) -> f32 {
    let light_size = shadow.pcss_params.x;
    let near_plane = shadow.pcss_params.y;

    // Penumbra width = (receiver - blocker) * lightSize / blocker
    // This is derived from similar triangles in the shadow geometry
    let penumbra = (receiver_depth - blocker_depth) * light_size / blocker_depth;

    return penumbra;
}

// PCSS Step 3: PCF with variable filter radius
fn pcss_pcf(uv: vec2<f32>, receiver_depth: f32, filter_radius: f32, screen_pos: vec2<f32>) -> f32 {
    var shadow_sum = 0.0;
    let texel_size = shadow.shadow_map_size.zw;
    let rotation = interleaved_gradient_noise_shadow(screen_pos) * 6.28318530718;

    for (var i = 0u; i < 16u; i++) {
        let offset = rotate_2d(POISSON_DISK[i], rotation) * filter_radius * texel_size;
        let sample_uv = uv + offset;
        shadow_sum += textureSampleCompare(shadow_map, shadow_sampler, sample_uv, receiver_depth);
    }

    return shadow_sum / 16.0;
}

// Full PCSS algorithm
// Note: Restructured to avoid non-uniform control flow before textureSampleCompare
fn calculate_shadow_pcss(world_pos: vec3<f32>, normal: vec3<f32>, to_light: vec3<f32>, screen_pos: vec2<f32>) -> f32 {
    // Transform to light space
    let light_space_pos = shadow.light_view_proj * vec4<f32>(world_pos, 1.0);
    let proj_coords_raw = light_space_pos.xyz / light_space_pos.w;

    let proj_x = proj_coords_raw.x * 0.5 + 0.5;
    let proj_y = proj_coords_raw.y * -0.5 + 0.5;
    let proj_z = proj_coords_raw.z;

    // Apply bias
    let bias = shadow.shadow_params.x;
    let normal_bias = shadow.shadow_params.y;
    let cos_angle = max(dot(normal, to_light), 0.0);
    let slope_bias = bias + normal_bias * (1.0 - cos_angle);
    let receiver_depth = proj_z - slope_bias;

    let uv = clamp(vec2<f32>(proj_x, proj_y), vec2<f32>(0.001), vec2<f32>(0.999));

    // Check bounds (computed but not used for early return to maintain uniform control flow)
    let in_bounds = step(0.0, proj_x) * step(proj_x, 1.0) * step(0.0, proj_y) * step(proj_y, 1.0) * step(0.0, proj_z) * step(proj_z, 1.0);

    // Step 1: Blocker search
    let search_radius = shadow.pcss_params.z;
    let blocker_result = pcss_blocker_search(uv, proj_z, search_radius, screen_pos);

    // Step 2: Estimate penumbra size (use safe default if no blockers)
    let has_blockers = step(0.5, blocker_result.y);
    let safe_blocker_depth = select(proj_z * 0.5, blocker_result.x, has_blockers > 0.5);
    let penumbra = pcss_estimate_penumbra(proj_z, safe_blocker_depth);

    // Clamp filter radius
    let max_radius = shadow.pcss_params.w;
    let filter_radius = clamp(penumbra * shadow.shadow_map_size.x, 1.0, max_radius);

    // Step 3: PCF with variable radius (always execute to maintain uniform control flow)
    let shadow_factor = pcss_pcf(uv, receiver_depth, filter_radius, screen_pos);

    // Apply bounds check and blocker check via blending instead of branching
    // If out of bounds or no blockers, return 1.0 (fully lit)
    let final_shadow = mix(1.0, shadow_factor, in_bounds * has_blockers);

    return final_shadow;
}

// ============ Contact Shadows (Screen-Space Ray Marching) ============

// Interleaved gradient noise for contact shadow dithering
fn interleaved_gradient_noise_contact(pos: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pos, magic.xy)));
}

// Ray march in screen space to find contact shadows
// Contact shadows add fine shadow detail at object contact points that shadow maps miss
// Enhanced version with:
// - More steps (16) for better coverage
// - Dithered starting offset to reduce banding
// - Exponential step distribution (smaller steps near contact point)
// - Smooth occlusion falloff
fn calculate_contact_shadow(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    screen_pos: vec2<f32>,
    depth: f32,
) -> f32 {
    // Check if contact shadows are enabled
    let enabled = shadow.contact_params.x;
    if (enabled < 0.5) {
        return 1.0;
    }

    let max_distance = shadow.contact_params.y;
    let thickness = shadow.contact_params.z;
    let intensity = shadow.contact_params.w;

    // Use 16 steps for better quality
    let num_steps = 16u;

    // Dithering offset to reduce banding artifacts
    let dither = interleaved_gradient_noise_contact(screen_pos) * 0.5;

    // Start position slightly offset along normal to prevent self-shadowing
    var ray_pos = world_pos + normal * 0.005;

    // March along the light direction with exponential step distribution
    // Smaller steps near the surface, larger steps farther away
    var occlusion = 0.0;
    var total_distance = 0.0;

    for (var i = 0u; i < num_steps; i++) {
        // Exponential step distribution: t goes from 0 to 1
        let t = (f32(i) + dither) / f32(num_steps);
        // Use quadratic distribution for more samples near contact
        let target_dist = t * t * max_distance;
        let step_dist = target_dist - total_distance;
        total_distance = target_dist;

        // Move ray toward light
        ray_pos += light_dir * step_dist;

        // Transform ray position to clip space
        let clip_pos = camera.view_proj * vec4<f32>(ray_pos, 1.0);
        let ndc = clip_pos.xyz / clip_pos.w;

        // Convert to screen UV
        let ray_uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);

        // Check if UV is in bounds
        if (ray_uv.x < 0.0 || ray_uv.x > 1.0 || ray_uv.y < 0.0 || ray_uv.y > 1.0) {
            break;
        }

        // The ray depth in clip space (0 = near, 1 = far for WebGPU)
        let ray_depth = ndc.z;

        // Check against the starting depth with thickness tolerance
        let depth_diff = ray_depth - depth;

        // If ray is behind geometry (positive diff) but within thickness, it's occluded
        if (depth_diff > 0.0 && depth_diff < thickness) {
            // Smooth occlusion falloff based on:
            // 1. Distance traveled (farther = less occlusion)
            // 2. Depth penetration (deeper = more occlusion)
            let distance_fade = 1.0 - (total_distance / max_distance);
            let depth_fade = 1.0 - (depth_diff / thickness);
            let combined_fade = distance_fade * depth_fade;

            // Use soft maximum for smooth accumulation
            occlusion = max(occlusion, combined_fade);
        }
    }

    // Apply smoothstep for softer shadow edges
    occlusion = smoothstep(0.0, 1.0, occlusion);

    // Apply intensity and return shadow factor
    return 1.0 - occlusion * intensity;
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

// Calculate shadow factor based on light type and PCF mode
fn calculate_shadow(world_pos: vec3<f32>, normal: vec3<f32>, to_light: vec3<f32>, light_type: f32, screen_pos: vec2<f32>, depth: f32) -> f32 {
    let shadows_on = step(0.5, shadow.shadow_params.z);

    if (shadows_on < 0.5) {
        return 1.0;
    }

    var shadow_factor = 1.0;

    // Use cube shadow for point lights, 2D shadow for directional/spot
    if (light_type > 1.5) {
        // Point light - use cube shadow map (PCSS not supported for cube maps)
        let light_pos = shadow.light_dir_or_pos.xyz;
        shadow_factor = calculate_shadow_cube(world_pos, normal, light_pos);
    } else {
        // Check PCF mode (stored in spot_params.w)
        let pcf_mode = shadow.spot_params.w;

        // PCSS mode = 5
        if (pcf_mode > 4.5) {
            shadow_factor = calculate_shadow_pcss(world_pos, normal, to_light, screen_pos);
        } else {
            // Standard PCF modes (0-4)
            shadow_factor = calculate_shadow_2d(world_pos, normal, to_light);
        }
    }

    // Apply contact shadows (screen-space ray marching)
    let contact_shadow = calculate_contact_shadow(world_pos, normal, to_light, screen_pos, depth);

    // Combine shadow map and contact shadows (use minimum = more shadow)
    return min(shadow_factor, contact_shadow);
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
    screen_pos: vec2<f32>,
    depth: f32,
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

    // Light color and intensity (from shadow uniform)
    let light_color = vec3<f32>(1.0, 0.98, 0.95);
    let light_intensity = shadow.spot_params.z * 3.0; // spot_params.z is the intensity multiplier
    let radiance = light_color * light_intensity * attenuation * spot_effect;

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, light_dir), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // Cook-Torrance BRDF with optimizations
    let a = roughness * roughness;
    let a2 = a * a;

    // GGX NDF
    let ndf = distribution_ggx(n, h, roughness);

    // Use Optimized optimized visibility function
    let vis = vis_smith_joint_approx(a2, n_dot_v, n_dot_l);

    // Fresnel with shadow compensation
    let f = fresnel_schlick(v_dot_h, f0);

    // Specular = D * Vis * F
    let specular = ndf * vis * f;

    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;

    // Use GGX rough diffuse with full retro-reflectivity for point/directional lights
    let diffuse = diffuse_ggx_rough(albedo, roughness, n_dot_v, n_dot_l, v_dot_h, n_dot_h, 1.0);

    // Get shadow factor (includes contact shadows)
    let shadow_factor = calculate_shadow(world_pos, n, light_dir, light_type, screen_pos, depth);

    return (kd * diffuse + specular) * radiance * n_dot_l * shadow_factor;
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

    // Apply detail albedo for micro-surface color variation
    albedo = apply_detail_albedo(albedo, in.world_position, in.uv);

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

    // Apply detail normal mapping for micro-surface detail at close range
    n = apply_detail_normal(n, in.world_position, in.uv, in.tangent, in.bitangent);

    let ao = material.ao;
    let clear_coat = material.clear_coat;
    let clear_coat_roughness = max(material.clear_coat_roughness, 0.01);
    let sheen = material.sheen;
    let sheen_color = material.sheen_color;
    let v = normalize(camera.position - in.world_position);

    // Calculate reflectance at normal incidence (F0)
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);

    // Shadow-casting light (directional, spot, or point)
    lo += calculate_shadow_light(in.world_position, n, v, f0, albedo, metallic, roughness, in.clip_position.xy, in.clip_position.z);

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

    // Rect light 0 - Area light rectangular area light
    if (camera.rectlight0_pos.w > 0.5) {
        lo += calculate_rect_light(
            in.world_position, n, v, f0, albedo, metallic, roughness,
            camera.rectlight0_pos.xyz,
            camera.rectlight0_dir_width.xyz,
            camera.rectlight0_tan_height.xyz,
            camera.rectlight0_dir_width.w,   // width
            camera.rectlight0_tan_height.w,  // height
            camera.rectlight0_color.rgb,
            camera.rectlight0_color.w,       // intensity
            20.0                              // range (default)
        );
    }

    // Rect light 1
    if (camera.rectlight1_pos.w > 0.5) {
        lo += calculate_rect_light(
            in.world_position, n, v, f0, albedo, metallic, roughness,
            camera.rectlight1_pos.xyz,
            camera.rectlight1_dir_width.xyz,
            camera.rectlight1_tan_height.xyz,
            camera.rectlight1_dir_width.w,   // width
            camera.rectlight1_tan_height.w,  // height
            camera.rectlight1_color.rgb,
            camera.rectlight1_color.w,       // intensity
            20.0                              // range (default)
        );
    }

    // Capsule light 0 - Area light tube/line area light
    if (camera.capsule0_start.w > 0.5) {
        lo += calculate_capsule_light(
            in.world_position, n, v, f0, albedo, metallic, roughness,
            camera.capsule0_start.xyz,
            camera.capsule0_end_radius.xyz,
            camera.capsule0_end_radius.w,    // radius
            camera.capsule0_color.rgb,
            camera.capsule0_color.w,         // intensity
            20.0                              // range (default)
        );
    }

    // Capsule light 1
    if (camera.capsule1_start.w > 0.5) {
        lo += calculate_capsule_light(
            in.world_position, n, v, f0, albedo, metallic, roughness,
            camera.capsule1_start.xyz,
            camera.capsule1_end_radius.xyz,
            camera.capsule1_end_radius.w,    // radius
            camera.capsule1_color.rgb,
            camera.capsule1_color.w,         // intensity
            20.0                              // range (default)
        );
    }

    // Disk light 0 - Area light circular area light
    if (camera.disk0_pos.w > 0.5) {
        lo += calculate_disk_light(
            in.world_position, n, v, f0, albedo, metallic, roughness,
            camera.disk0_pos.xyz,
            camera.disk0_dir_radius.xyz,
            camera.disk0_dir_radius.w,       // radius
            camera.disk0_color.rgb,
            camera.disk0_color.w,            // intensity
            20.0                              // range (default)
        );
    }

    // Disk light 1
    if (camera.disk1_pos.w > 0.5) {
        lo += calculate_disk_light(
            in.world_position, n, v, f0, albedo, metallic, roughness,
            camera.disk1_pos.xyz,
            camera.disk1_dir_radius.xyz,
            camera.disk1_dir_radius.w,       // radius
            camera.disk1_color.rgb,
            camera.disk1_color.w,            // intensity
            20.0                              // range (default)
        );
    }

    // Sphere light 0 - Area light spherical area light
    if (camera.sphere0_pos.w > 0.5) {
        lo += calculate_sphere_light(
            in.world_position, n, v, f0, albedo, metallic, roughness,
            camera.sphere0_pos.xyz,
            camera.sphere0_radius_range.x,   // radius
            camera.sphere0_color.rgb,
            camera.sphere0_color.w,          // intensity
            camera.sphere0_radius_range.y    // range
        );
    }

    // Sphere light 1
    if (camera.sphere1_pos.w > 0.5) {
        lo += calculate_sphere_light(
            in.world_position, n, v, f0, albedo, metallic, roughness,
            camera.sphere1_pos.xyz,
            camera.sphere1_radius_range.x,   // radius
            camera.sphere1_color.rgb,
            camera.sphere1_color.w,          // intensity
            camera.sphere1_radius_range.y    // range
        );
    }

    // IBL (Image-Based Lighting) from environment map
    let ibl_ambient = calculate_ibl(n, v, albedo, metallic, roughness, ao, f0);

    // Hemisphere light (additive, for additional ambient gradient)
    let hemisphere_ambient = calculate_hemisphere_light(n) * albedo * ao * 0.3;

    // Flat ambient to prevent dark areas (especially interiors)
    // Higher value helps when normals face away from main lights
    let flat_ambient = vec3<f32>(0.15) * albedo * ao;

    // Calculate clear coat contribution
    // Clear coat attenuates the base layer and adds its own specular reflection
    var clear_coat_specular = vec3<f32>(0.0);
    var base_attenuation = 1.0;

    if (clear_coat > 0.001) {
        // Calculate clear coat IBL
        let cc_ibl_result = calculate_clear_coat_ibl(n, v, clear_coat, clear_coat_roughness);
        let cc_ibl_luminance = cc_ibl_result.x;
        base_attenuation = cc_ibl_result.y;

        // Sample environment for clear coat reflection
        let r = reflect(-v, n);
        let mip_level = clear_coat_roughness * 4.0;
        let cc_env = textureSampleLevel(env_map, env_sampler, r, mip_level).rgb;

        // Clear coat fresnel at view angle
        let n_dot_v = max(dot(n, v), 0.0);
        let cc_fresnel = fresnel_schlick_clear_coat(n_dot_v);

        // Clear coat specular from IBL
        let ibl_specular_intensity = camera.ibl_settings.y;
        clear_coat_specular = cc_env * cc_fresnel * clear_coat * ibl_specular_intensity;
    }

    // Calculate cloth/sheen IBL contribution
    // Cloth adds a soft, fuzzy highlight from the sheen layer
    var sheen_specular = vec3<f32>(0.0);

    if (sheen > 0.001) {
        let n_dot_v = max(dot(n, v), 0.0);
        let r = reflect(-v, n);

        // Sample environment at higher mip for soft cloth highlight
        // Cloth roughness is higher to create softer reflections
        let sheen_mip = roughness * 6.0 + 2.0;
        let sheen_env = textureSampleLevel(env_map, env_sampler, r, sheen_mip).rgb;

        // Charlie-style fresnel using sheen color
        let sheen_fresnel = fresnel_schlick(n_dot_v, sheen_color);

        // Apply sheen with inverse GGX-like falloff (brighter at grazing angles)
        let grazing_boost = 1.0 + (1.0 - n_dot_v) * 2.0;

        let ibl_specular_intensity = camera.ibl_settings.y;
        sheen_specular = sheen_env * sheen_fresnel * sheen * grazing_boost * ibl_specular_intensity * 0.5;
    }

    // Apply clear coat attenuation to base layer and add clear coat + sheen specular
    var color = (flat_ambient + hemisphere_ambient + ibl_ambient + lo) * base_attenuation + clear_coat_specular + sheen_specular;

    // Alpha cutoff - discard fully transparent pixels (for decals, stickers, etc.)
    if (alpha < 0.1) {
        discard;
    }

    // Handle render modes for debug visualization
    // 0=Lit, 1=Unlit, 2=Normals, 3=Depth, 4=Metallic, 5=Roughness, 6=AO, 7=UVs, 8=Flat, 9=Wireframe
    let mode = camera.render_mode;

    if (mode == 1u) {
        // Unlit - base color with simple directional lighting (no PBR, no textures effects)
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
        let ndotl = max(dot(n, light_dir), 0.0);
        let ambient = 0.3;
        let diffuse = ndotl * 0.7;
        color = albedo * (ambient + diffuse);
    } else if (mode == 2u) {
        // Normals - visualize world-space normals (map from [-1,1] to [0,1])
        color = n * 0.5 + 0.5;
    } else if (mode == 3u) {
        // Depth - visualize depth (using clip_position.z)
        // Adjust for better visualization
        let depth = pow(in.clip_position.z, 0.5); // Non-linear for better visualization
        color = vec3<f32>(depth, depth, depth);
    } else if (mode == 4u) {
        // Metallic - visualize metallic value
        color = vec3<f32>(metallic, metallic, metallic);
    } else if (mode == 5u) {
        // Roughness - visualize roughness value
        color = vec3<f32>(roughness, roughness, roughness);
    } else if (mode == 6u) {
        // AO - visualize ambient occlusion
        color = vec3<f32>(ao, ao, ao);
    } else if (mode == 7u) {
        // UVs - visualize UV coordinates
        color = vec3<f32>(fract(in.uv.x), fract(in.uv.y), 0.0);
    } else if (mode == 8u) {
        // Flat - clay/matte look with simple lighting (no textures, neutral gray)
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
        let ndotl = max(dot(n, light_dir), 0.0);
        let ambient = 0.35;
        let diffuse = ndotl * 0.65;
        let clay_color = vec3<f32>(0.7, 0.7, 0.7);
        color = clay_color * (ambient + diffuse);
    } else if (mode == 9u) {
        // True wireframe using barycentric coordinates
        // Each vertex has a unique barycentric: (1,0,0), (0,1,0), (0,0,1)
        // Near triangle edges, one component approaches 0
        let bary = in.barycentric;

        // Calculate edge proximity using screen-space derivatives for anti-aliasing
        let d = fwidth(bary);

        // Find the minimum barycentric coordinate (closest to an edge)
        let min_bary = min(min(bary.x, bary.y), bary.z);

        // Use the derivative to determine edge width (adaptive to screen space)
        let min_d = min(min(d.x, d.y), d.z);
        let edge_width = 1.0; // Line thickness in pixels

        // Discard interior pixels - only keep edge pixels
        if (min_bary > edge_width * min_d) {
            discard;
        }

        // White lines only
        color = vec3<f32>(1.0, 1.0, 1.0);
    }
    // mode == 0u is normal Lit rendering, already computed above

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

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // Cook-Torrance BRDF with optimizations
    let a = roughness * roughness;
    let a2 = a * a;

    // GGX NDF
    let ndf = distribution_ggx(n, h, roughness);

    // Use Optimized optimized visibility function (combines G term with denominator)
    let vis = vis_smith_joint_approx(a2, n_dot_v, n_dot_l);

    // Fresnel with shadow compensation
    let f = fresnel_schlick(v_dot_h, f0);

    // Specular = D * Vis * F (Vis already includes 1/(4*NoV*NoL))
    let specular = ndf * vis * f;

    // Energy conservation
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;

    // Use GGX rough diffuse
    let diffuse = diffuse_ggx_rough(albedo, roughness, n_dot_v, n_dot_l, v_dot_h, n_dot_h, 1.0);

    return (kd * diffuse + specular) * radiance * n_dot_l;
}

// ============================================================================
// Area Light Rect Light Implementation
// Based on "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines"
// by Heitz et al. 2016.
// ============================================================================

// Rect light structure for convenience
struct RectLight {
    position: vec3<f32>,
    direction: vec3<f32>,   // Normal to rect (pointing outward)
    tangent: vec3<f32>,     // Width direction
    bitangent: vec3<f32>,   // Height direction (computed)
    width: f32,
    height: f32,
    color: vec3<f32>,
    intensity: f32,
}

// Calculate solid angle of a spherical rectangle
// Based on "The Solid Angle of a Plane Triangle" by Van Oosterom & Strackee
fn spherical_rect_solid_angle(
    p: vec3<f32>,           // Shading point
    rect_center: vec3<f32>,
    rect_x: vec3<f32>,      // Half-width vector
    rect_y: vec3<f32>,      // Half-height vector
) -> f32 {
    // Four corners of the rectangle relative to shading point
    let p0 = rect_center - rect_x - rect_y - p;
    let p1 = rect_center + rect_x - rect_y - p;
    let p2 = rect_center + rect_x + rect_y - p;
    let p3 = rect_center - rect_x + rect_y - p;

    // Normalized directions to corners
    let d0 = normalize(p0);
    let d1 = normalize(p1);
    let d2 = normalize(p2);
    let d3 = normalize(p3);

    // Solid angle as sum of spherical excess of two triangles
    // Using the formula: tan(Omega/2) = (a·(b×c)) / (1 + a·b + b·c + c·a)
    let n01 = cross(d0, d1);
    let n12 = cross(d1, d2);
    let n23 = cross(d2, d3);
    let n30 = cross(d3, d0);

    // Sum angles at each edge
    let g0 = acos(clamp(-dot(n01, n30), -1.0, 1.0));
    let g1 = acos(clamp(-dot(n12, n01), -1.0, 1.0));
    let g2 = acos(clamp(-dot(n23, n12), -1.0, 1.0));
    let g3 = acos(clamp(-dot(n30, n23), -1.0, 1.0));

    // Spherical excess = sum of angles - 2*PI
    let solid_angle = max(0.0, g0 + g1 + g2 + g3 - 2.0 * PI);

    return solid_angle;
}

// Find closest point on rectangle to a given point
fn closest_point_on_rect(
    p: vec3<f32>,
    rect_center: vec3<f32>,
    rect_tangent: vec3<f32>,
    rect_bitangent: vec3<f32>,
    half_width: f32,
    half_height: f32,
) -> vec3<f32> {
    let d = p - rect_center;

    // Project onto rect plane
    let u = clamp(dot(d, rect_tangent), -half_width, half_width);
    let v = clamp(dot(d, rect_bitangent), -half_height, half_height);

    return rect_center + rect_tangent * u + rect_bitangent * v;
}

// Most Representative Point (MRP) for specular reflection
// Finds the point on the rect that best represents the specular highlight
fn rect_most_representative_point(
    world_pos: vec3<f32>,
    reflect_dir: vec3<f32>,
    rect_center: vec3<f32>,
    rect_normal: vec3<f32>,
    rect_tangent: vec3<f32>,
    rect_bitangent: vec3<f32>,
    half_width: f32,
    half_height: f32,
) -> vec3<f32> {
    // Intersect reflection ray with rect plane
    let n_dot_r = dot(rect_normal, reflect_dir);

    // Handle grazing angles
    if (abs(n_dot_r) < 0.0001) {
        return closest_point_on_rect(world_pos, rect_center, rect_tangent, rect_bitangent, half_width, half_height);
    }

    let d = dot(rect_center - world_pos, rect_normal);
    let t = d / n_dot_r;

    // If intersection is behind the surface, use closest point
    if (t < 0.0) {
        return closest_point_on_rect(world_pos, rect_center, rect_tangent, rect_bitangent, half_width, half_height);
    }

    let hit_point = world_pos + reflect_dir * t;

    // Clamp to rect bounds
    return closest_point_on_rect(hit_point, rect_center, rect_tangent, rect_bitangent, half_width, half_height);
}

// Rect light diffuse irradiance using solid angle approximation
fn rect_diffuse_irradiance(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    rect: RectLight,
) -> f32 {
    let half_width = rect.width * 0.5;
    let half_height = rect.height * 0.5;

    let rect_x = rect.tangent * half_width;
    let rect_y = rect.bitangent * half_height;

    // Calculate solid angle
    let solid_angle = spherical_rect_solid_angle(world_pos, rect.position, rect_x, rect_y);

    // Direction to rect center
    let to_light = rect.position - world_pos;
    let dist = length(to_light);
    let l = to_light / max(dist, 0.0001);

    // Lambertian factor
    let n_dot_l = max(dot(n, l), 0.0);

    // Light facing factor (rect emits from front face)
    let light_facing = max(-dot(rect.direction, l), 0.0);

    // Irradiance = solid_angle * cos(theta) * light_facing / PI
    return solid_angle * n_dot_l * light_facing / PI;
}

// Rect light distance attenuation with Area light falloff
fn rect_attenuation(dist_sq: f32, range: f32) -> f32 {
    // Smooth falloff to zero at range
    let range_sq = range * range;
    let dist_ratio = dist_sq / range_sq;

    //inverse square falloff with window
    let falloff = saturate(1.0 - dist_ratio * dist_ratio);
    let falloff_sq = falloff * falloff;

    // Combine with distance attenuation
    let attenuation = falloff_sq / max(dist_sq, 0.01);

    return attenuation;
}

// Main rect light calculation using optimized techniques
fn calculate_rect_light(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    f0: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    rect_pos: vec3<f32>,
    rect_dir: vec3<f32>,
    rect_tan: vec3<f32>,
    rect_width: f32,
    rect_height: f32,
    rect_color: vec3<f32>,
    rect_intensity: f32,
    rect_range: f32,
) -> vec3<f32> {
    // Build rect light structure
    var rect: RectLight;
    rect.position = rect_pos;
    rect.direction = normalize(rect_dir);
    rect.tangent = normalize(rect_tan);
    rect.bitangent = cross(rect.direction, rect.tangent);
    rect.width = rect_width;
    rect.height = rect_height;
    rect.color = rect_color;
    rect.intensity = rect_intensity;

    let half_width = rect_width * 0.5;
    let half_height = rect_height * 0.5;

    // Distance check
    let to_center = rect_pos - world_pos;
    let dist_sq = dot(to_center, to_center);

    // Range-based attenuation
    let attenuation = rect_attenuation(dist_sq, rect_range);
    if (attenuation < 0.0001) {
        return vec3<f32>(0.0);
    }

    // Check if point is behind the rect light
    let facing = dot(to_center, rect.direction);
    if (facing > 0.0) {
        return vec3<f32>(0.0);
    }

    // === DIFFUSE ===
    let diffuse_irradiance = rect_diffuse_irradiance(world_pos, n, rect);

    // === SPECULAR ===
    // Use Most Representative Point for specular
    let reflect_dir = reflect(-v, n);
    let mrp = rect_most_representative_point(
        world_pos,
        reflect_dir,
        rect.position,
        rect.direction,
        rect.tangent,
        rect.bitangent,
        half_width,
        half_height
    );

    // Calculate specular from MRP
    let l_spec = normalize(mrp - world_pos);
    let h = normalize(v + l_spec);

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l_spec), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // Cook-Torrance specular with energy normalization
    let a = roughness * roughness;
    let a2 = a * a;

    // Calculate sin_alpha from light size and distance (optimized technique)
    let mrp_dist = length(mrp - world_pos);
    let light_size = min(rect_width, rect_height); // Use smaller dimension for sin_alpha
    let sin_alpha = saturate(light_size * 0.5 / max(mrp_dist, 0.001));

    //energy normalization - prevents blown-out highlights
    let energy_result = energy_normalization_rect(a2, v_dot_h, sin_alpha);
    let a2_normalized = energy_result.x;
    let energy = energy_result.y;

    // Modified NDF with energy-normalized roughness
    let denom = n_dot_h * n_dot_h * (a2_normalized - 1.0) + 1.0;
    let ndf = a2_normalized / (PI * denom * denom);

    // Visibility term (use original a2 for geometric term)
    let vis = vis_smith_joint_approx(a2, n_dot_v, n_dot_l);

    // Fresnel
    let f = fresnel_schlick(v_dot_h, f0);

    // Specular = D * Vis * F * Energy
    let specular = ndf * vis * f * energy;

    // Energy conservation
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;

    // Use GGX rough diffuse with reduced retro-reflectivity for area lights
    // Larger lights (higher sin_alpha) get less retro-reflectivity to avoid artifacts
    let retro_weight = 1.0 - sin_alpha;
    let diffuse = diffuse_ggx_rough(albedo, roughness, n_dot_v, n_dot_l, v_dot_h, n_dot_h, retro_weight);

    // Combine diffuse and specular with attenuation
    let radiance = rect_color * rect_intensity * attenuation;

    let diffuse_contrib = kd * diffuse * diffuse_irradiance;
    let specular_contrib = specular * n_dot_l;

    return (diffuse_contrib + specular_contrib) * radiance;
}

// ============================================================================
// Area Light Capsule/Tube Light Implementation
// Based on "Real Shading in Unreal Engine 4" by Brian Karis. (SIGGRAPH 2013).
// ============================================================================

// Capsule light structure for convenience
struct CapsuleLight {
    start: vec3<f32>,
    end: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
}

// Find closest point on line segment to a point
// Returns the closest point and the parametric t value [0, 1]
fn closest_point_on_segment(
    p: vec3<f32>,
    a: vec3<f32>,
    b: vec3<f32>,
) -> vec3<f32> {
    let ab = b - a;
    let ab_len_sq = dot(ab, ab);

    if (ab_len_sq < 0.0001) {
        // Degenerate case: segment is a point
        return a;
    }

    let t = saturate(dot(p - a, ab) / ab_len_sq);
    return a + ab * t;
}

// Line irradiance calculation (optimized method)
// Integrates the irradiance from a line segment
fn line_irradiance(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    line_start: vec3<f32>,
    line_end: vec3<f32>,
) -> f32 {
    // Vector from surface point to line endpoints
    let l0 = line_start - world_pos;
    let l1 = line_end - world_pos;

    // Length of vectors
    let l0_len = length(l0);
    let l1_len = length(l1);

    // Avoid division by zero
    if (l0_len < 0.0001 || l1_len < 0.0001) {
        return 0.0;
    }

    // Normalized directions
    let l0_dir = l0 / l0_len;
    let l1_dir = l1 / l1_len;

    // Angle between the two directions (for integration)
    let cos_angle = dot(l0_dir, l1_dir);

    //line irradiance formula (from paper)
    // This integrates cos(theta) / r^2 along the line
    let n_dot_l0 = dot(n, l0_dir);
    let n_dot_l1 = dot(n, l1_dir);

    // Approximate irradiance using endpoint average weighted by solid angle
    let irradiance = (n_dot_l0 / l0_len + n_dot_l1 / l1_len) * 0.5;

    // Scale by line length factor (approximates integral over line)
    let line_length = length(line_end - line_start);
    let avg_dist = (l0_len + l1_len) * 0.5;

    return max(irradiance, 0.0) * line_length / max(avg_dist, 0.01);
}

// Find the representative point on a capsule for specular reflection
fn capsule_representative_point(
    world_pos: vec3<f32>,
    reflect_dir: vec3<f32>,
    capsule_start: vec3<f32>,
    capsule_end: vec3<f32>,
    capsule_radius: f32,
) -> vec3<f32> {
    // Ray-line closest approach for representative point
    let line_dir = normalize(capsule_end - capsule_start);
    let line_length = length(capsule_end - capsule_start);

    // Find closest point on reflection ray to the line
    let w0 = world_pos - capsule_start;
    let a = dot(reflect_dir, reflect_dir);  // Always 1 for normalized
    let b = dot(reflect_dir, line_dir);
    let c = dot(line_dir, line_dir);        // Always 1 for normalized
    let d = dot(reflect_dir, w0);
    let e = dot(line_dir, w0);

    let denom = a * c - b * b;

    var t_line: f32;
    if (abs(denom) < 0.0001) {
        // Lines are parallel, use closest point approach
        t_line = e / c;
    } else {
        // General case
        let t_ray = (b * e - c * d) / denom;
        t_line = (a * e - b * d) / denom;

        // Only use positive ray distances
        if (t_ray < 0.0) {
            // Reflection ray points away, find closest point on segment
            t_line = saturate(dot(world_pos - capsule_start, line_dir) / max(line_length, 0.001));
        }
    }

    // Clamp to segment bounds
    t_line = saturate(t_line * line_length / max(line_length, 0.001));

    // Get point on line
    let point_on_line = capsule_start + line_dir * t_line * line_length;

    // For specular, we want the point on the capsule surface
    // Move toward the reflection ray by the capsule radius
    let to_point = point_on_line - world_pos;
    let to_point_dir = normalize(to_point);

    // Blend between line point and offset by radius based on angle
    let blend = saturate(dot(reflect_dir, to_point_dir));

    return point_on_line;
}

// Capsule light distance attenuation with Area light falloff
fn capsule_attenuation(closest_dist_sq: f32, range: f32) -> f32 {
    // Smooth falloff to zero at range
    let range_sq = range * range;
    let dist_ratio = closest_dist_sq / range_sq;

    //inverse square falloff with window
    let falloff = saturate(1.0 - dist_ratio * dist_ratio);
    let falloff_sq = falloff * falloff;

    // Combine with distance attenuation
    let attenuation = falloff_sq / max(closest_dist_sq, 0.01);

    return attenuation;
}

// Main capsule light calculation using optimized techniques
fn calculate_capsule_light(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    f0: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    capsule_start: vec3<f32>,
    capsule_end: vec3<f32>,
    capsule_radius: f32,
    capsule_color: vec3<f32>,
    capsule_intensity: f32,
    capsule_range: f32,
) -> vec3<f32> {
    // Find closest point on capsule axis to the surface point
    let closest_on_line = closest_point_on_segment(world_pos, capsule_start, capsule_end);

    // Distance check (from closest point on line)
    let to_closest = closest_on_line - world_pos;
    let closest_dist_sq = dot(to_closest, to_closest);

    // Range-based attenuation
    let attenuation = capsule_attenuation(closest_dist_sq, capsule_range);
    if (attenuation < 0.0001) {
        return vec3<f32>(0.0);
    }

    // === DIFFUSE ===
    // Use line irradiance for accurate diffuse from the tube
    let diffuse_irradiance = line_irradiance(world_pos, n, capsule_start, capsule_end);

    // === SPECULAR ===
    // Use representative point method for specular
    let reflect_dir = reflect(-v, n);
    let repr_point = capsule_representative_point(
        world_pos,
        reflect_dir,
        capsule_start,
        capsule_end,
        capsule_radius
    );

    // Calculate specular from representative point
    let l_spec = normalize(repr_point - world_pos);
    let h = normalize(v + l_spec);

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l_spec), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // Cook-Torrance specular with energy normalization
    let a = roughness * roughness;
    let a2 = a * a;

    // Calculate energy normalization parameters for capsule/tube lights
    let repr_dist = length(repr_point - world_pos);
    let tube_length = length(capsule_end - capsule_start);

    // Sphere sin_alpha from capsule radius
    let sphere_sin_alpha = saturate(capsule_radius / max(repr_dist, 0.001));

    // Line cos_subtended: cos(2*alpha) where tan(alpha) = (length/2) / distance
    // Using identity: cos(2*alpha) = 1 / (1 + tan²(alpha)) * (1 - tan²(alpha))
    //                              = (1 - tan²(alpha)) / (1 + tan²(alpha))
    let half_length = tube_length * 0.5;
    let tan_alpha = half_length / max(repr_dist, 0.001);
    let tan_alpha_sq = tan_alpha * tan_alpha;
    let line_cos_subtended = (1.0 - tan_alpha_sq) / (1.0 + tan_alpha_sq);

    //energy normalization for capsule (sphere + line)
    let energy_result = energy_normalization_capsule(a2, v_dot_h, sphere_sin_alpha, line_cos_subtended);
    let a2_normalized = energy_result.x;
    let energy = energy_result.y;

    // Modified NDF with energy-normalized roughness
    let denom = n_dot_h * n_dot_h * (a2_normalized - 1.0) + 1.0;
    let ndf = a2_normalized / (PI * denom * denom);

    // Visibility term (use original a2)
    let vis = vis_smith_joint_approx(a2, n_dot_v, n_dot_l);

    // Fresnel
    let f = fresnel_schlick(v_dot_h, f0);

    // Specular = D * Vis * F * Energy
    let specular = ndf * vis * f * energy;

    // Energy conservation
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;

    // Use GGX rough diffuse with reduced retro-reflectivity for area lights
    // Capsule lights: use sphere sin_alpha for retro-weight reduction
    let retro_weight = 1.0 - sphere_sin_alpha;
    let diffuse = diffuse_ggx_rough(albedo, roughness, n_dot_v, n_dot_l, v_dot_h, n_dot_h, retro_weight);

    // Combine diffuse and specular with attenuation
    let radiance = capsule_color * capsule_intensity * attenuation;

    // Normalize by tube length for consistent brightness
    let length_normalization = 1.0 / max(tube_length, 0.1);

    let diffuse_contrib = kd * diffuse * diffuse_irradiance * length_normalization;
    let specular_contrib = specular * n_dot_l * length_normalization;

    return (diffuse_contrib + specular_contrib) * radiance;
}

// ============================================================================
// Area Light Disk Light Implementation
// Based on "Real-Time Area Lighting" by Stephen Hill and Eric Heitz
// ============================================================================

// Disk light structure
struct DiskLight {
    position: vec3<f32>,
    direction: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
}

// Calculate solid angle of a disk from a point
fn disk_solid_angle(
    world_pos: vec3<f32>,
    disk_center: vec3<f32>,
    disk_normal: vec3<f32>,
    disk_radius: f32,
) -> f32 {
    let to_disk = disk_center - world_pos;
    let dist = length(to_disk);

    if (dist < 0.0001) {
        return PI; // At disk center, hemisphere visible
    }

    let to_disk_dir = to_disk / dist;

    // Angle between viewing direction and disk normal
    let cos_angle = abs(dot(to_disk_dir, disk_normal));

    // Projected radius based on viewing angle
    let projected_radius = disk_radius * cos_angle;

    // Approximate solid angle using projected disk
    let sin_half_angle = min(projected_radius / dist, 1.0);
    let solid_angle = PI * sin_half_angle * sin_half_angle;

    return solid_angle;
}

// Find the representative point on a disk for specular
fn disk_representative_point(
    world_pos: vec3<f32>,
    reflect_dir: vec3<f32>,
    disk_center: vec3<f32>,
    disk_normal: vec3<f32>,
    disk_radius: f32,
) -> vec3<f32> {
    // Intersect reflection ray with disk plane
    let n_dot_r = dot(disk_normal, reflect_dir);

    // Handle grazing angles
    if (abs(n_dot_r) < 0.0001) {
        // Return closest point on disk edge
        let to_center = disk_center - world_pos;
        let proj = to_center - disk_normal * dot(to_center, disk_normal);
        let proj_len = length(proj);
        if (proj_len > 0.0001) {
            return disk_center - normalize(proj) * min(disk_radius, proj_len);
        }
        return disk_center;
    }

    let d = dot(disk_center - world_pos, disk_normal);
    let t = d / n_dot_r;

    // If intersection is behind, find closest point
    if (t < 0.0) {
        let to_center = disk_center - world_pos;
        let closest_dir = normalize(to_center);
        return disk_center - closest_dir * min(disk_radius, length(to_center));
    }

    let hit_point = world_pos + reflect_dir * t;

    // Check if hit point is within disk radius
    let offset = hit_point - disk_center;
    let offset_len = length(offset);

    if (offset_len <= disk_radius) {
        return hit_point;
    }

    // Clamp to disk edge
    return disk_center + normalize(offset) * disk_radius;
}

// Main disk light calculation
fn calculate_disk_light(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    f0: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    disk_pos: vec3<f32>,
    disk_dir: vec3<f32>,
    disk_radius: f32,
    disk_color: vec3<f32>,
    disk_intensity: f32,
    disk_range: f32,
) -> vec3<f32> {
    let disk_normal = normalize(disk_dir);

    // Distance check
    let to_center = disk_pos - world_pos;
    let dist_sq = dot(to_center, to_center);

    // Range-based attenuation
    let range_sq = disk_range * disk_range;
    let dist_ratio = dist_sq / range_sq;
    let falloff = saturate(1.0 - dist_ratio * dist_ratio);
    let falloff_sq = falloff * falloff;
    let attenuation = falloff_sq / max(dist_sq, 0.01);

    if (attenuation < 0.0001) {
        return vec3<f32>(0.0);
    }

    // Check if point is behind the disk
    let facing = dot(to_center, disk_normal);
    if (facing > 0.0) {
        return vec3<f32>(0.0);
    }

    // === DIFFUSE ===
    let solid_angle = disk_solid_angle(world_pos, disk_pos, disk_normal, disk_radius);
    let to_center_dir = normalize(to_center);
    let n_dot_l = max(dot(n, to_center_dir), 0.0);
    let light_facing = max(-dot(disk_normal, to_center_dir), 0.0);
    let diffuse_irradiance = solid_angle * n_dot_l * light_facing / PI;

    // === SPECULAR ===
    let reflect_dir = reflect(-v, n);
    let repr_point = disk_representative_point(
        world_pos, reflect_dir, disk_pos, disk_normal, disk_radius
    );

    let l_spec = normalize(repr_point - world_pos);
    let h = normalize(v + l_spec);

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l_spec = max(dot(n, l_spec), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    //energy normalization for disk lights
    let a = roughness * roughness;
    let a2 = a * a;
    let repr_dist = length(repr_point - world_pos);

    // Calculate sin_alpha from disk radius and distance
    let sin_alpha = saturate(disk_radius / max(repr_dist, 0.001));

    //energy normalization
    let energy_result = energy_normalization_sphere(a2, v_dot_h, sin_alpha);
    let a2_normalized = energy_result.x;
    let energy = energy_result.y;

    // Modified NDF with energy-normalized roughness
    let denom = n_dot_h * n_dot_h * (a2_normalized - 1.0) + 1.0;
    let ndf = a2_normalized / (PI * denom * denom);

    // Visibility term (use original a2)
    let vis = vis_smith_joint_approx(a2, n_dot_v, n_dot_l_spec);
    let f = fresnel_schlick(v_dot_h, f0);

    // Specular = D * Vis * F * Energy
    let specular = ndf * vis * f * energy;

    // Energy conservation
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;

    // Use GGX rough diffuse with reduced retro-reflectivity for area lights
    let retro_weight = 1.0 - sin_alpha;
    let diffuse = diffuse_ggx_rough(albedo, roughness, n_dot_v, n_dot_l_spec, v_dot_h, n_dot_h, retro_weight);

    let radiance = disk_color * disk_intensity * attenuation;
    let diffuse_contrib = kd * diffuse * diffuse_irradiance;
    let specular_contrib = specular * n_dot_l_spec;

    return (diffuse_contrib + specular_contrib) * radiance;
}

// ============================================================================
// Area Light Sphere Light Implementation
// Based on "Real Shading in Unreal Engine 4" by Brian Karis.
// ============================================================================

// Sphere light structure
struct SphereLightData {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
}

// Calculate solid angle of a sphere from a point
fn sphere_solid_angle(dist: f32, radius: f32) -> f32 {
    if (dist <= radius) {
        // Inside the sphere
        return 2.0 * PI;
    }
    // Solid angle: 2*PI*(1 - cos(theta)) where sin(theta) = r/d
    let sin_theta = min(radius / dist, 1.0);
    let cos_theta = sqrt(1.0 - sin_theta * sin_theta);
    return 2.0 * PI * (1.0 - cos_theta);
}

// Find the representative point on sphere surface for specular
fn sphere_representative_point(
    world_pos: vec3<f32>,
    reflect_dir: vec3<f32>,
    sphere_center: vec3<f32>,
    sphere_radius: f32,
) -> vec3<f32> {
    // Find closest point on ray to sphere center
    let to_center = sphere_center - world_pos;
    let t = max(dot(to_center, reflect_dir), 0.0);
    let closest_on_ray = world_pos + reflect_dir * t;

    // Vector from sphere center to closest point on ray
    let to_closest = closest_on_ray - sphere_center;
    let dist_to_closest = length(to_closest);

    if (dist_to_closest < 0.0001) {
        // Ray goes through center, return point on sphere toward viewer
        return sphere_center - reflect_dir * sphere_radius;
    }

    // Point on sphere surface closest to the ray
    let dir_to_closest = to_closest / dist_to_closest;
    return sphere_center + dir_to_closest * sphere_radius;
}

// Main sphere light calculation
fn calculate_sphere_light(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    f0: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    sphere_pos: vec3<f32>,
    sphere_radius: f32,
    sphere_color: vec3<f32>,
    sphere_intensity: f32,
    sphere_range: f32,
) -> vec3<f32> {
    // Distance check
    let to_center = sphere_pos - world_pos;
    let dist = length(to_center);
    let dist_sq = dist * dist;

    // Range-based attenuation
    let range_sq = sphere_range * sphere_range;
    let dist_ratio = dist_sq / range_sq;
    let falloff = saturate(1.0 - dist_ratio * dist_ratio);
    let falloff_sq = falloff * falloff;
    let attenuation = falloff_sq / max(dist_sq, 0.01);

    if (attenuation < 0.0001) {
        return vec3<f32>(0.0);
    }

    let l = to_center / max(dist, 0.0001);

    // === DIFFUSE ===
    let solid_angle = sphere_solid_angle(dist, sphere_radius);
    let n_dot_l = max(dot(n, l), 0.0);
    let diffuse_irradiance = solid_angle * n_dot_l / PI;

    // === SPECULAR ===
    let reflect_dir = reflect(-v, n);
    let repr_point = sphere_representative_point(
        world_pos, reflect_dir, sphere_pos, sphere_radius
    );

    let l_spec = normalize(repr_point - world_pos);
    let h = normalize(v + l_spec);

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l_spec = max(dot(n, l_spec), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    //energy normalization for sphere lights
    let a = roughness * roughness;
    let a2 = a * a;
    let repr_dist = length(repr_point - world_pos);

    // Calculate sin_alpha from sphere radius and distance
    let sin_alpha = saturate(sphere_radius / max(repr_dist, 0.001));

    //energy normalization
    let energy_result = energy_normalization_sphere(a2, v_dot_h, sin_alpha);
    let a2_normalized = energy_result.x;
    let energy = energy_result.y;

    // Modified NDF with energy-normalized roughness
    let denom = n_dot_h * n_dot_h * (a2_normalized - 1.0) + 1.0;
    let ndf = a2_normalized / (PI * denom * denom);

    // Visibility term (use original a2)
    let vis = vis_smith_joint_approx(a2, n_dot_v, n_dot_l_spec);
    let f = fresnel_schlick(v_dot_h, f0);

    // Specular = D * Vis * F * Energy
    let specular = ndf * vis * f * energy;

    // Energy conservation
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;

    // Use GGX rough diffuse with reduced retro-reflectivity for area lights
    let retro_weight = 1.0 - sin_alpha;
    let diffuse = diffuse_ggx_rough(albedo, roughness, n_dot_v, n_dot_l_spec, v_dot_h, n_dot_h, retro_weight);

    // Normalize by sphere surface area for consistent brightness
    let area_normalization = 1.0 / (4.0 * PI * sphere_radius * sphere_radius + 0.0001);
    let radiance = sphere_color * sphere_intensity * attenuation * area_normalization;

    let diffuse_contrib = kd * diffuse * diffuse_irradiance;
    let specular_contrib = specular * n_dot_l_spec;

    return (diffuse_contrib + specular_contrib) * radiance;
}
