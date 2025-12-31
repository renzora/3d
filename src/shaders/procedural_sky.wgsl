// Procedural sky shader with atmospheric scattering, clouds, and sun bloom
// Based on Preetham/Hosek-Wilkie sky model simplified for real-time

struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    rayleigh_coefficient: f32,
    mie_coefficient: f32,
    mie_directional_g: f32,
    turbidity: f32,
    sun_disk_size: f32,
    sun_disk_intensity: f32,
    ground_color: vec3<f32>,
    exposure: f32,
    time: f32,
    cloud_speed: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ray_dir: vec3<f32>,
}

// Generate full-screen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Full-screen triangle vertices
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 1.0, 1.0);

    // Calculate world-space ray direction
    let clip_pos = vec4<f32>(x, y, 1.0, 1.0);
    let world_pos = uniforms.inv_view_proj * clip_pos;
    out.ray_dir = normalize(world_pos.xyz / world_pos.w);

    return out;
}

const PI: f32 = 3.14159265359;

// ============ Noise functions for clouds ============

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// 3D value noise
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(
            mix(hash31(i + vec3<f32>(0.0, 0.0, 0.0)), hash31(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0, 1.0, 0.0)), hash31(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash31(i + vec3<f32>(0.0, 0.0, 1.0)), hash31(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0, 1.0, 1.0)), hash31(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

// Fractal Brownian Motion for cloud detail
fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise3d(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// Cloud density function
fn cloud_density(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    // Only render clouds above horizon
    if (ray_dir.y < 0.05) {
        return 0.0;
    }

    // Project ray onto cloud layer (approximate height)
    let cloud_height = 0.15; // Height in sky dome units
    let t = cloud_height / max(ray_dir.y, 0.001);
    let cloud_pos = ray_dir * t;

    // Cloud UV coordinates with time-based movement
    let cloud_scale = 3.0;
    let wind_direction = vec2<f32>(1.0, 0.3); // Wind blows mostly in X direction
    let wind_offset = wind_direction * uniforms.time * uniforms.cloud_speed;
    let cloud_uv = cloud_pos.xz * cloud_scale + wind_offset;

    // Multi-octave noise for cloud shape
    // Add slight variation at different speeds for more natural movement
    let noise_pos = vec3<f32>(cloud_uv.x, cloud_uv.y, uniforms.time * uniforms.cloud_speed * 0.1);
    var density = fbm(noise_pos, 5);

    // Shape the clouds - create more defined edges
    let coverage = 0.45; // Cloud coverage (0-1)
    density = smoothstep(coverage, coverage + 0.3, density);

    // Fade clouds at horizon
    let horizon_fade = smoothstep(0.05, 0.2, ray_dir.y);
    density *= horizon_fade;

    // Fade clouds towards zenith slightly
    let zenith_fade = 1.0 - smoothstep(0.6, 1.0, ray_dir.y) * 0.3;
    density *= zenith_fade;

    return density;
}

// Cloud lighting and color
fn render_clouds(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sky_color: vec3<f32>) -> vec3<f32> {
    let density = cloud_density(ray_dir, sun_dir);

    if (density <= 0.0) {
        return vec3<f32>(0.0);
    }

    // Base cloud color (bright white)
    var cloud_color = vec3<f32>(1.0, 1.0, 1.0);

    // Sun-lit side vs shadow side
    let sun_dot = dot(ray_dir, sun_dir);
    let light_factor = sun_dot * 0.3 + 0.7; // Soft lighting

    // Add sun color to clouds
    let sun_height = sun_dir.y;
    if (sun_height < 0.3) {
        // Sunset/sunrise - warm cloud colors
        let sunset_t = 1.0 - sun_height / 0.3;
        cloud_color = mix(cloud_color, vec3<f32>(1.0, 0.7, 0.4), sunset_t * 0.6);
    }

    // Shadow on bottom of clouds
    let shadow = smoothstep(0.0, 0.5, density) * 0.3;
    cloud_color *= (light_factor - shadow + 0.3);

    // Cloud brightness based on sun
    let brightness = max(0.4, sun_height * 0.6 + 0.6) * uniforms.sun_intensity * 0.1;
    cloud_color *= brightness;

    // Night clouds are very dim
    if (sun_height < 0.0) {
        cloud_color *= smoothstep(-0.3, 0.0, sun_height);
    }

    return cloud_color * density;
}

// ============ Star functions ============

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let n = sin(dot(p, vec2<f32>(41.0, 289.0)));
    return fract(vec2<f32>(262144.0, 32768.0) * n);
}

fn stars(ray_dir: vec3<f32>, sun_height: f32) -> vec3<f32> {
    if (ray_dir.y < 0.0) {
        return vec3<f32>(0.0);
    }

    let star_visibility = smoothstep(0.1, -0.2, sun_height);
    if (star_visibility <= 0.0) {
        return vec3<f32>(0.0);
    }

    let theta = atan2(ray_dir.z, ray_dir.x);
    let phi = asin(ray_dir.y);

    let star_density = 80.0;
    let uv = vec2<f32>(theta, phi) * star_density;
    let grid_id = floor(uv);
    let grid_uv = fract(uv) - 0.5;

    var star_color = vec3<f32>(0.0);

    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let offset = vec2<f32>(f32(i), f32(j));
            let cell_id = grid_id + offset;

            let rand = hash22(cell_id);
            let star_pos = offset + rand - 0.5;
            let d = length(grid_uv - star_pos);

            let brightness_rand = hash21(cell_id * 1.31);

            if (brightness_rand > 0.92) {
                let star_size = 0.02 + brightness_rand * 0.03;
                var star_brightness = smoothstep(star_size, 0.0, d);
                let core = smoothstep(star_size * 0.3, 0.0, d) * 2.0;
                star_brightness = star_brightness + core;

                var color = vec3<f32>(1.0);
                let color_rand = hash21(cell_id * 2.17);
                if (color_rand > 0.8) {
                    color = vec3<f32>(0.8, 0.9, 1.0);
                } else if (color_rand > 0.6) {
                    color = vec3<f32>(1.0, 0.95, 0.8);
                } else if (color_rand > 0.55) {
                    color = vec3<f32>(1.0, 0.7, 0.6);
                }

                let intensity = 0.5 + brightness_rand * 1.5;
                star_color += star_brightness * color * intensity;
            }

            if (brightness_rand > 0.995) {
                let bright_size = 0.06;
                let glow = smoothstep(bright_size, 0.0, d) * 3.0;
                star_color += glow * vec3<f32>(1.0, 0.98, 0.95);
            }
        }
    }

    return star_color * star_visibility;
}

// ============ Atmospheric scattering ============

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = (1.0 - g2);
    let denom = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return (1.0 / (4.0 * PI)) * num / denom;
}

fn get_beta_rayleigh() -> vec3<f32> {
    // Boosted blue channel for more vibrant sky with ACES tonemapping
    return vec3<f32>(6.5e-6, 15.0e-6, 40.0e-6) * uniforms.rayleigh_coefficient;
}

fn get_beta_mie() -> vec3<f32> {
    return vec3<f32>(21e-6) * uniforms.mie_coefficient * uniforms.turbidity;
}

fn optical_depth_rayleigh(y: f32) -> f32 {
    let H_R = 8500.0;
    return exp(-max(y, 0.0) * 5.0) * H_R;
}

fn optical_depth_mie(y: f32) -> f32 {
    let H_M = 1200.0;
    return exp(-max(y, 0.0) * 2.5) * H_M;
}

fn atmosphere(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let cos_theta = dot(ray_dir, sun_dir);

    let beta_r = get_beta_rayleigh();
    let beta_m = get_beta_mie();

    let y = ray_dir.y;
    let depth_r = optical_depth_rayleigh(y);
    let depth_m = optical_depth_mie(y);

    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, uniforms.mie_directional_g);

    let scatter_r = beta_r * phase_r * depth_r;
    let scatter_m = beta_m * phase_m * depth_m;

    let sun_color = vec3<f32>(1.0, 0.98, 0.92) * uniforms.sun_intensity;
    var sky_color = (scatter_r + scatter_m) * sun_color;

    // Add vibrant blue gradient for clear sky look
    let zenith_blue = vec3<f32>(0.15, 0.35, 0.85) * uniforms.sun_intensity * 0.08;
    let blue_gradient = pow(max(0.0, y), 0.6);
    sky_color += zenith_blue * blue_gradient;

    // Improved horizon - warmer, less grey
    let horizon_factor = 1.0 - y;
    let horizon_warmth = pow(horizon_factor, 3.0) * 0.15;
    let horizon_tint = vec3<f32>(0.9, 0.85, 0.75) * horizon_warmth * uniforms.sun_intensity * 0.5;
    sky_color += horizon_tint;

    let sun_height = sun_dir.y;
    if (sun_height < 0.3) {
        let sunset_factor = 1.0 - sun_height / 0.3;
        let horizon_glow = pow(horizon_factor, 4.0) * 0.4;
        let sunset_color = vec3<f32>(1.8, 0.6, 0.25) * sunset_factor * horizon_glow;
        sky_color += sunset_color * sun_color * 0.5;
    }

    return sky_color;
}

// ============ Sun rendering with outer ring glare ============

fn sun_disk(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let cos_theta = dot(ray_dir, sun_dir);
    let sun_angular_radius = uniforms.sun_disk_size * 0.00873;

    var sun_contribution = vec3<f32>(0.0);

    // Core sun disk - bright solid circle
    let core_cos = cos(sun_angular_radius);
    if (cos_theta > core_cos) {
        let edge_factor = (cos_theta - core_cos) / (1.0 - core_cos);
        // Softer edge falloff
        let sun_intensity = smoothstep(0.0, 0.3, edge_factor) * uniforms.sun_disk_intensity;
        sun_contribution += vec3<f32>(1.0, 0.99, 0.95) * sun_intensity;
    }

    // Outer ring glare - the main visible glare effect
    let ring_inner = sun_angular_radius * 1.0;
    let ring_outer = sun_angular_radius * 2.5;
    let ring_cos_inner = cos(ring_inner);
    let ring_cos_outer = cos(ring_outer);

    if (cos_theta > ring_cos_outer && cos_theta < ring_cos_inner) {
        // Create ring shape - peaks between inner and outer
        let ring_mid = (ring_inner + ring_outer) * 0.5;
        let ring_cos_mid = cos(ring_mid);
        let angle_from_sun = acos(cos_theta);

        // Distance from ring center (normalized)
        let ring_width = ring_outer - ring_inner;
        let dist_from_mid = abs(angle_from_sun - ring_mid) / (ring_width * 0.5);
        let ring_intensity = (1.0 - dist_from_mid * dist_from_mid) * uniforms.sun_disk_intensity * 0.4;

        sun_contribution += vec3<f32>(1.0, 0.97, 0.9) * max(0.0, ring_intensity);
    }

    // Soft outer glow (subtle, not rays)
    let glow_size = sun_angular_radius * 6.0;
    let glow_cos = cos(glow_size);
    if (cos_theta > glow_cos) {
        let glow_factor = (cos_theta - glow_cos) / (core_cos - glow_cos);
        let glow = pow(max(0.0, glow_factor), 3.0) * uniforms.sun_disk_intensity * 0.15;
        sun_contribution += vec3<f32>(1.0, 0.95, 0.85) * glow;
    }

    // Sun color shift at sunset
    let sun_height = sun_dir.y;
    if (sun_height < 0.2) {
        let sunset_t = 1.0 - sun_height / 0.2;
        sun_contribution.r *= 1.0 + sunset_t * 0.3;
        sun_contribution.g *= 1.0 - sunset_t * 0.2;
        sun_contribution.b *= 1.0 - sunset_t * 0.5;
    }

    return sun_contribution;
}

// ============ Moon rendering ============

fn moon(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    // Moon is opposite to sun
    let moon_dir = -sun_dir;

    // Only show moon when sun is below horizon (night time)
    if (sun_dir.y > 0.1) {
        return vec3<f32>(0.0);
    }

    // Moon visibility fades in as sun sets
    let moon_visibility = smoothstep(0.1, -0.1, sun_dir.y);

    let cos_theta = dot(ray_dir, moon_dir);
    let moon_angular_radius = 0.009; // Moon is about 0.5 degrees

    var moon_contribution = vec3<f32>(0.0);

    // Moon disk
    let moon_cos = cos(moon_angular_radius);
    if (cos_theta > moon_cos) {
        let edge_factor = (cos_theta - moon_cos) / (1.0 - moon_cos);

        // Moon surface with subtle shading
        let moon_base = vec3<f32>(0.95, 0.93, 0.88); // Slightly warm white

        // Simple phase shading based on angle
        let phase_shading = smoothstep(0.0, 1.0, edge_factor);
        let moon_intensity = phase_shading * 2.0;

        moon_contribution = moon_base * moon_intensity;
    }

    // Subtle moon glow
    let glow_size = moon_angular_radius * 4.0;
    let glow_cos = cos(glow_size);
    if (cos_theta > glow_cos) {
        let glow_factor = (cos_theta - glow_cos) / (moon_cos - glow_cos);
        let glow = pow(max(0.0, glow_factor), 2.0) * 0.3;
        moon_contribution += vec3<f32>(0.8, 0.85, 0.95) * glow;
    }

    return moon_contribution * moon_visibility;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = normalize(in.ray_dir);
    let sun_dir = normalize(uniforms.sun_direction);

    var color: vec3<f32>;

    // Below horizon - improved ground/horizon rendering
    if (ray_dir.y < 0.0) {
        // Get sky color at horizon for blending
        let horizon_sky = atmosphere(vec3<f32>(ray_dir.x, 0.01, ray_dir.z), sun_dir);

        // Ground with ambient based on sun position
        let ground_ambient = max(0.15, sun_dir.y * 0.4 + 0.5);
        let ground = uniforms.ground_color * ground_ambient;

        // Smooth blend from ground to horizon sky
        let horizon_blend = smoothstep(-0.15, 0.0, ray_dir.y);
        color = mix(ground, horizon_sky, horizon_blend);
    } else {
        // Sky with atmospheric scattering
        color = atmosphere(ray_dir, sun_dir);

        // Add stars at night (before clouds so they're occluded)
        color += stars(ray_dir, sun_dir.y);

        // Add moon at night
        color += moon(ray_dir, sun_dir);

        // Add clouds
        let cloud_color = render_clouds(ray_dir, sun_dir, color);
        // Blend clouds over sky - clouds partially occlude what's behind
        let cloud_dens = cloud_density(ray_dir, sun_dir);
        color = mix(color, cloud_color / max(cloud_dens, 0.001), cloud_dens * 0.9);

        // Add sun disk and glare if sun is above horizon
        if (sun_dir.y > -0.1) {
            color += sun_disk(ray_dir, sun_dir);
        }
    }

    // Apply exposure
    color *= uniforms.exposure;

    return vec4<f32>(color, 1.0);
}
