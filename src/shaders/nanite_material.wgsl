// Nanite Material Pass Shader
// Fullscreen pass that reads visibility buffer and shades visible pixels
// Uses proper PBR (Cook-Torrance BRDF) for high quality rendering

const PI: f32 = 3.14159265359;

// Camera uniform (matches PbrCameraUniform)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    render_mode: u32,
    hemisphere_sky: vec4<f32>,
    hemisphere_ground: vec4<f32>,
    ibl_settings: vec4<f32>,
    light0_pos: vec4<f32>,
    light0_color: vec4<f32>,
    light1_pos: vec4<f32>,
    light1_color: vec4<f32>,
    light2_pos: vec4<f32>,
    light2_color: vec4<f32>,
    light3_pos: vec4<f32>,
    light3_color: vec4<f32>,
    detail_settings: vec4<f32>,
}

// Cluster data (64 bytes)
struct Cluster {
    bounding_sphere: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    parent_error: f32,
    cluster_error: f32,
    lod_level: u32,
    group_index: u32,
    index_offset: u32,
    triangle_count: u32,
    vertex_offset: u32,
    material_id: u32,
}

// Vertex attribute (32 bytes)
struct VertexAttribute {
    normal: vec4<f32>,    // xyz = normal, w = unused
    tangent: vec4<f32>,   // xyz = tangent, w = handedness
    uv: vec4<f32>,        // xy = uv, zw = unused
    color: vec4<f32>,     // rgba
}

// Material data (32 bytes)
struct Material {
    base_color: vec4<f32>,  // RGBA
    texture_index: i32,     // -1 = no texture
    metallic: f32,
    roughness: f32,
    _pad: f32,
}

// Bind groups
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> clusters: array<Cluster>;
@group(1) @binding(1) var<storage, read> groups: array<u32>;  // Placeholder
@group(1) @binding(2) var<storage, read> positions: array<vec3<f32>>;
@group(1) @binding(3) var<storage, read> attributes: array<VertexAttribute>;
@group(1) @binding(4) var<storage, read> indices: array<u32>;
@group(1) @binding(5) var<storage, read> instances: array<u32>;  // Placeholder

@group(2) @binding(0) var visibility_texture: texture_2d<u32>;
@group(2) @binding(1) var depth_texture: texture_depth_2d;

// Shadow uniform struct (matches WebShadowUniform)
struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
    shadow_params: vec4<f32>,         // x=bias, y=normal_bias, z=enabled, w=light_type
    light_dir_or_pos: vec4<f32>,
    shadow_map_size: vec4<f32>,
    spot_direction: vec4<f32>,
    spot_params: vec4<f32>,
    pcss_params: vec4<f32>,
    contact_params: vec4<f32>,
}

// Light types
const LIGHT_DIRECTIONAL: u32 = 0u;
const LIGHT_SPOT: u32 = 1u;
const LIGHT_POINT: u32 = 2u;

// Group 3: Materials and shadows
@group(3) @binding(0) var<storage, read> materials: array<Material>;
@group(3) @binding(1) var texture_array: texture_2d_array<f32>;
@group(3) @binding(2) var texture_sampler: sampler;
@group(3) @binding(3) var<uniform> shadow_data: ShadowUniform;
@group(3) @binding(4) var shadow_map: texture_depth_2d;
@group(3) @binding(5) var shadow_sampler: sampler_comparison;
@group(3) @binding(6) var shadow_cube_map: texture_depth_cube;
@group(3) @binding(7) var shadow_cube_sampler: sampler_comparison;

// Vertex output for fullscreen triangle
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// ============== PBR Functions (optimized techniques) ==============

// Fresnel-Schlick approximation with Optimized shadow compensation
// Anything less than 2% reflectance is physically impossible and is instead considered shadowing
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let fc = pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    // The saturate(50.0 * f0.g) term compensates for very dark F0 values (optimized technique)
    return saturate(50.0 * f0.g) * fc + (1.0 - fc) * f0;
}

// Fresnel-Schlick with roughness for IBL
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Disney Burley Diffuse BRDF (optimized technique)
// Provides roughness-dependent diffuse that looks more realistic than Lambert
fn diffuse_burley(diffuse_color: vec3<f32>, roughness: f32, n_dot_v: f32, n_dot_l: f32, v_dot_h: f32) -> vec3<f32> {
    let fd90 = 0.5 + 2.0 * v_dot_h * v_dot_h * roughness;
    let fd_v = 1.0 + (fd90 - 1.0) * pow(1.0 - n_dot_v, 5.0);
    let fd_l = 1.0 + (fd90 - 1.0) * pow(1.0 - n_dot_l, 5.0);
    return diffuse_color * (1.0 / PI) * fd_v * fd_l;
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

// Optimized optimized Smith Joint Approximate visibility function
// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
fn vis_smith_joint_approx(a2: f32, n_dot_v: f32, n_dot_l: f32) -> f32 {
    let a = sqrt(a2);
    let vis_smith_v = n_dot_l * (n_dot_v * (1.0 - a) + a);
    let vis_smith_l = n_dot_v * (n_dot_l * (1.0 - a) + a);
    return 0.5 / (vis_smith_v + vis_smith_l + 0.0001);
}

// Calculate PBR lighting for a single point light (with optimized techniques)
fn calculate_pbr_light(
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
    shadow_factor: f32,
) -> vec3<f32> {
    // Calculate direction FROM world position TO light (correct direction for lighting)
    let to_light = light_pos - world_pos;
    let distance = length(to_light);
    let l = to_light / distance;  // normalized light direction
    let h = normalize(v + l);

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }

    // Distance attenuation (inverse square falloff)
    let attenuation = 1.0 / (distance * distance);
    let radiance = light_color * light_intensity * attenuation;

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

    // Energy conservation
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;

    // Use Disney Burley diffuse
    let diffuse = diffuse_burley(albedo, roughness, n_dot_v, n_dot_l, v_dot_h);

    return (kd * diffuse + specular) * radiance * n_dot_l * shadow_factor;
}

// ============== Utility Functions ==============

// Fullscreen triangle vertices
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);

    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);

    return out;
}

// Unpack triangle ID from visibility ID
fn unpack_triangle_id(packed: u32) -> u32 {
    return (packed - 1u) & 0xFFFu;
}

// Unpack cluster ID from visibility ID
fn unpack_cluster_id(packed: u32) -> u32 {
    return ((packed - 1u) >> 12u) & 0xFFFFu;
}

// Unpack instance ID from visibility ID
fn unpack_instance_id(packed: u32) -> u32 {
    return ((packed - 1u) >> 28u) & 0xFu;
}

// Compute barycentric coordinates for a point in a triangle
fn compute_barycentric(p: vec2<f32>, v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>) -> vec3<f32> {
    let v0v1 = v1 - v0;
    let v0v2 = v2 - v0;
    let v0p = p - v0;

    let d00 = dot(v0v1, v0v1);
    let d01 = dot(v0v1, v0v2);
    let d11 = dot(v0v2, v0v2);
    let d20 = dot(v0p, v0v1);
    let d21 = dot(v0p, v0v2);

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    return vec3<f32>(u, v, w);
}

// Compute signed area of a 2D triangle
fn triangle_area_2d(v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>) -> f32 {
    return abs((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)) * 0.5;
}

// Compute mip level based on screen-space and UV-space triangle areas
fn compute_mip_level(
    screen0: vec2<f32>, screen1: vec2<f32>, screen2: vec2<f32>,
    uv0: vec2<f32>, uv1: vec2<f32>, uv2: vec2<f32>,
    tex_size: vec2<f32>,
    mip_count: f32,
) -> f32 {
    let screen_area = triangle_area_2d(screen0, screen1, screen2);
    let uv_area = triangle_area_2d(uv0 * tex_size, uv1 * tex_size, uv2 * tex_size);

    if (screen_area < 0.0001 || uv_area < 0.0001) {
        return 0.0;
    }

    let texels_per_pixel = uv_area / screen_area;
    let mip = max(0.0, log2(texels_per_pixel) * 0.5 - 0.5);

    return min(mip, mip_count - 1.0);
}

// Calculate PBR lighting for a directional light (no attenuation, with optimized techniques)
fn calculate_pbr_directional_light(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    f0: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    light_intensity: f32,
    shadow_factor: f32,
) -> vec3<f32> {
    let l = normalize(light_dir);
    let h = normalize(v + l);

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }

    // No distance attenuation for directional lights
    let radiance = light_color * light_intensity;

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

    // Energy conservation
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;

    // Use Disney Burley diffuse
    let diffuse = diffuse_burley(albedo, roughness, n_dot_v, n_dot_l, v_dot_h);

    return (kd * diffuse + specular) * radiance * n_dot_l * shadow_factor;
}

// ============== Shadow Functions ==============

fn sample_shadow_2d(shadow_coord: vec3<f32>, bias: f32) -> f32 {
    let texel_size = shadow_data.shadow_map_size.zw;
    let compare_depth = shadow_coord.z - bias;
    let clamped_uv = clamp(shadow_coord.xy, vec2<f32>(0.001), vec2<f32>(0.999));

    let in_bounds = shadow_coord.x >= 0.0 && shadow_coord.x <= 1.0 &&
                    shadow_coord.y >= 0.0 && shadow_coord.y <= 1.0 &&
                    shadow_coord.z >= 0.0 && shadow_coord.z <= 1.0;

    let shadow_sample = textureSampleCompare(shadow_map, shadow_sampler, clamped_uv, compare_depth);

    return select(1.0, shadow_sample, in_bounds);
}

fn sample_shadow_cube(world_pos: vec3<f32>, light_pos: vec3<f32>, range: f32, bias: f32) -> f32 {
    let light_to_frag = world_pos - light_pos;
    let dist = length(light_to_frag);
    let sample_dir = normalize(light_to_frag);
    let shadow_depth = textureSampleCompare(shadow_cube_map, shadow_cube_sampler, sample_dir, (dist / range) - bias);
    return shadow_depth;
}

fn calculate_shadow(world_position: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    let enabled = shadow_data.shadow_params.z;
    let bias = shadow_data.shadow_params.x;
    let normal_bias = shadow_data.shadow_params.y;
    let light_type = u32(shadow_data.shadow_params.w);

    let biased_position = world_position + world_normal * normal_bias;

    let light_space_pos = shadow_data.light_view_proj * vec4<f32>(biased_position, 1.0);
    var shadow_coord = light_space_pos.xyz / light_space_pos.w;
    shadow_coord.x = shadow_coord.x * 0.5 + 0.5;
    shadow_coord.y = shadow_coord.y * -0.5 + 0.5;
    let shadow_2d = sample_shadow_2d(shadow_coord, bias);

    let light_pos = shadow_data.light_dir_or_pos.xyz;
    let range = shadow_data.spot_direction.w;
    let shadow_cube = sample_shadow_cube(biased_position, light_pos, range, bias);

    let shadow_value = select(shadow_2d, shadow_cube, light_type == LIGHT_POINT);

    return select(shadow_value, 1.0, enabled < 0.5);
}

// ============== Main Fragment Shader ==============

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel_coords = vec2<i32>(in.position.xy);
    let vis_id = textureLoad(visibility_texture, pixel_coords, 0).r;

    if (vis_id == 0u) {
        discard;
    }

    // Unpack IDs
    let triangle_id = unpack_triangle_id(vis_id);
    let cluster_id = unpack_cluster_id(vis_id);
    let cluster = clusters[cluster_id];

    // Get triangle indices
    let idx0 = indices[cluster.index_offset + triangle_id * 3u + 0u];
    let idx1 = indices[cluster.index_offset + triangle_id * 3u + 1u];
    let idx2 = indices[cluster.index_offset + triangle_id * 3u + 2u];

    // Get vertex positions
    let p0 = positions[cluster.vertex_offset + idx0];
    let p1 = positions[cluster.vertex_offset + idx1];
    let p2 = positions[cluster.vertex_offset + idx2];

    // Get vertex attributes
    let attr0 = attributes[cluster.vertex_offset + idx0];
    let attr1 = attributes[cluster.vertex_offset + idx1];
    let attr2 = attributes[cluster.vertex_offset + idx2];

    // Project triangle vertices to screen space
    let clip0 = camera.view_proj * vec4<f32>(p0, 1.0);
    let clip1 = camera.view_proj * vec4<f32>(p1, 1.0);
    let clip2 = camera.view_proj * vec4<f32>(p2, 1.0);

    let ndc0 = clip0.xy / clip0.w;
    let ndc1 = clip1.xy / clip1.w;
    let ndc2 = clip2.xy / clip2.w;

    let dims = textureDimensions(visibility_texture);
    let screen0 = vec2<f32>((ndc0.x * 0.5 + 0.5) * f32(dims.x), (0.5 - ndc0.y * 0.5) * f32(dims.y));
    let screen1 = vec2<f32>((ndc1.x * 0.5 + 0.5) * f32(dims.x), (0.5 - ndc1.y * 0.5) * f32(dims.y));
    let screen2 = vec2<f32>((ndc2.x * 0.5 + 0.5) * f32(dims.x), (0.5 - ndc2.y * 0.5) * f32(dims.y));

    // Compute barycentric coordinates
    let bary = compute_barycentric(in.position.xy, screen0, screen1, screen2);

    // Perspective-correct interpolation weights
    let w0 = bary.x / clip0.w;
    let w1 = bary.y / clip1.w;
    let w2 = bary.z / clip2.w;
    let w_sum = w0 + w1 + w2;

    // Interpolate attributes
    let normal = normalize((attr0.normal.xyz * w0 + attr1.normal.xyz * w1 + attr2.normal.xyz * w2) / w_sum);

    let uv0 = attr0.uv.xy;
    let uv1 = attr1.uv.xy;
    let uv2 = attr2.uv.xy;
    let uv = (uv0 * w0 + uv1 * w1 + uv2 * w2) / w_sum;

    let vertex_color = (attr0.color.rgb * w0 + attr1.color.rgb * w1 + attr2.color.rgb * w2) / w_sum;
    let world_pos = (p0 * w0 + p1 * w1 + p2 * w2) / w_sum;

    // View direction
    let v = normalize(camera.position - world_pos);
    let n = normal;

    // Get material
    let material = materials[cluster.material_id];
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04); // Prevent divide by zero in GGX

    // Sample texture
    let tex_idx = max(0, material.texture_index);
    let tex_dims = textureDimensions(texture_array, 0);
    let tex_size = vec2<f32>(f32(tex_dims.x), f32(tex_dims.y));
    let mip_count = f32(textureNumLevels(texture_array));
    let mip_level = compute_mip_level(screen0, screen1, screen2, uv0, uv1, uv2, tex_size, mip_count);
    let tex_color = textureSampleLevel(texture_array, texture_sampler, uv, tex_idx, mip_level);

    // Base color
    let has_texture = material.texture_index >= 0;
    var albedo = select(material.base_color.rgb, tex_color.rgb, has_texture) * vertex_color;

    // F0 (reflectance at normal incidence)
    // Dielectrics use 0.04, metals use albedo color
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Calculate shadow
    let shadow = calculate_shadow(world_pos, n);

    // ============== Lighting ==============
    var lo = vec3<f32>(0.0);

    // Main shadow-casting light (directional, spot, or point - from shadow_data)
    let shadows_enabled = shadow_data.shadow_params.z > 0.5;
    if (shadows_enabled) {
        let light_type = u32(shadow_data.shadow_params.w);
        let light_intensity = shadow_data.spot_params.z * 3.0;  // Same multiplier as regular PBR
        let light_color = vec3<f32>(1.0, 0.98, 0.95);  // Slightly warm white

        if (light_type == LIGHT_DIRECTIONAL) {
            // Directional light - direction is from shadow_data
            let light_dir = normalize(-shadow_data.light_dir_or_pos.xyz);
            lo += calculate_pbr_directional_light(
                world_pos, n, v, f0, albedo, metallic, roughness,
                light_dir, light_color, light_intensity, shadow
            );
        } else if (light_type == LIGHT_SPOT || light_type == LIGHT_POINT) {
            // Spot/Point light - position is from shadow_data
            let light_pos = shadow_data.light_dir_or_pos.xyz;
            let range = shadow_data.spot_direction.w;
            let to_light = light_pos - world_pos;
            let dist = length(to_light);
            let attenuation = clamp(1.0 - dist / range, 0.0, 1.0);
            let att_squared = attenuation * attenuation;

            // Spot cone (only for spot lights)
            var spot_effect = 1.0;
            if (light_type == LIGHT_SPOT) {
                let spot_dir = normalize(shadow_data.spot_direction.xyz);
                let light_to_frag = normalize(-to_light);
                let cos_angle = dot(light_to_frag, spot_dir);
                let outer_cos = shadow_data.spot_params.x;
                let inner_cos = shadow_data.spot_params.y;
                spot_effect = clamp((cos_angle - outer_cos) / (inner_cos - outer_cos), 0.0, 1.0);
            }

            lo += calculate_pbr_light(
                world_pos, n, v, f0, albedo, metallic, roughness,
                light_pos, light_color, light_intensity * att_squared * spot_effect, shadow
            );
        }
    }

    // Light 0 (additional fill light, no shadow)
    if (camera.light0_color.w > 0.5) {
        lo += calculate_pbr_light(
            world_pos, n, v, f0, albedo, metallic, roughness,
            camera.light0_pos.xyz,
            camera.light0_color.rgb,
            camera.light0_pos.w,
            1.0
        );
    }

    // Light 1
    if (camera.light1_color.w > 0.5) {
        lo += calculate_pbr_light(
            world_pos, n, v, f0, albedo, metallic, roughness,
            camera.light1_pos.xyz,
            camera.light1_color.rgb,
            camera.light1_pos.w,
            1.0
        );
    }

    // Light 2
    if (camera.light2_color.w > 0.5) {
        lo += calculate_pbr_light(
            world_pos, n, v, f0, albedo, metallic, roughness,
            camera.light2_pos.xyz,
            camera.light2_color.rgb,
            camera.light2_pos.w,
            1.0
        );
    }

    // Light 3
    if (camera.light3_color.w > 0.5) {
        lo += calculate_pbr_light(
            world_pos, n, v, f0, albedo, metallic, roughness,
            camera.light3_pos.xyz,
            camera.light3_color.rgb,
            camera.light3_pos.w,
            1.0
        );
    }

    // ============== Ambient ==============

    // Flat ambient to prevent completely dark areas (matching regular PBR shader)
    let flat_ambient = vec3<f32>(0.15) * albedo;

    // Hemisphere lighting (additive)
    var hemisphere_ambient = vec3<f32>(0.0);
    if (camera.hemisphere_sky.w > 0.5) {
        let up = vec3<f32>(0.0, 1.0, 0.0);
        let hemi_factor = dot(n, up) * 0.5 + 0.5;
        let hemi_color = mix(camera.hemisphere_ground.rgb, camera.hemisphere_sky.rgb, hemi_factor);

        // Apply Fresnel to ambient for more realistic look
        let n_dot_v = max(dot(n, v), 0.0);
        let f_ambient = fresnel_schlick_roughness(n_dot_v, f0, roughness);
        let kd_ambient = (1.0 - f_ambient) * (1.0 - metallic);

        hemisphere_ambient = kd_ambient * albedo * hemi_color * 0.3;
    }

    let ambient = flat_ambient + hemisphere_ambient;

    // Final color
    var color = ambient + lo;

    // Debug render modes
    switch (camera.render_mode) {
        case 1u: {  // Depth
            let depth = textureLoad(depth_texture, pixel_coords, 0);
            let linear_depth = (depth - 0.1) / (100.0 - 0.1);
            return vec4<f32>(vec3<f32>(linear_depth), 1.0);
        }
        case 2u: {  // Normals
            return vec4<f32>(n * 0.5 + 0.5, 1.0);
        }
        case 3u: {  // Albedo
            return vec4<f32>(albedo, 1.0);
        }
        case 4u: {  // Metallic
            return vec4<f32>(vec3<f32>(metallic), 1.0);
        }
        case 5u: {  // Roughness
            return vec4<f32>(vec3<f32>(roughness), 1.0);
        }
        case 6u: {  // AO (not available, show white)
            return vec4<f32>(1.0, 1.0, 1.0, 1.0);
        }
        case 7u: {  // UVs
            return vec4<f32>(uv, 0.0, 1.0);
        }
        default: {
            return vec4<f32>(color, 1.0);
        }
    }
}
