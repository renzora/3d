// Procedural sky shader with physically-based atmospheric scattering
// Based on Bruneton 2017 atmosphere model
// Features: Rayleigh + Mie + Ozone, multi-scattering, proper sun transmittance

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

// ============ Planet/Atmosphere Constants ============
// Based on Earth parameters (distances in km, then normalized)

const PLANET_RADIUS: f32 = 6371.0;      // Earth radius in km
const ATMOSPHERE_HEIGHT: f32 = 100.0;    // Atmosphere thickness in km
const ATMOSPHERE_RADIUS: f32 = 6471.0;   // Planet + atmosphere

// Height scale factors (in km)
const RAYLEIGH_SCALE_HEIGHT: f32 = 8.5;  // Rayleigh density scale height
const MIE_SCALE_HEIGHT: f32 = 1.2;       // Mie density scale height

// Ozone layer parameters
const OZONE_CENTER_HEIGHT: f32 = 25.0;   // Center of ozone layer (km)
const OZONE_WIDTH: f32 = 15.0;           // Width of ozone layer (km)

// Scattering coefficients at sea level (1/km)
// Rayleigh: wavelength-dependent scattering
fn get_rayleigh_scattering() -> vec3<f32> {
    // Standard Earth Rayleigh coefficients (1/km) - boosted for visual appeal
    return vec3<f32>(5.8e-3, 13.5e-3, 33.1e-3) * uniforms.rayleigh_coefficient;
}

// Mie scattering (mostly wavelength-independent)
fn get_mie_scattering() -> vec3<f32> {
    return vec3<f32>(3.996e-3) * uniforms.mie_coefficient * uniforms.turbidity;
}

fn get_mie_extinction() -> vec3<f32> {
    // Mie extinction = scattering + absorption
    // For typical atmospheric aerosols, absorption is about 10% of extinction
    return get_mie_scattering() * 1.11;
}

// Ozone absorption (no scattering, only absorption)
fn get_ozone_absorption() -> vec3<f32> {
    // Ozone absorbs in Chappuis band (orange-red) and Hartley band (UV)
    // This gives the sky its deep blue at zenith
    return vec3<f32>(0.65e-3, 1.88e-3, 0.085e-3) * uniforms.turbidity;
}

// ============ Density Functions ============

// Rayleigh density at height h (km above sea level)
fn rayleigh_density(h: f32) -> f32 {
    return exp(-h / RAYLEIGH_SCALE_HEIGHT);
}

// Mie density at height h
fn mie_density(h: f32) -> f32 {
    return exp(-h / MIE_SCALE_HEIGHT);
}

// Ozone density - peaks at ~25km altitude with triangular profile
fn ozone_density(h: f32) -> f32 {
    let dist_from_center = abs(h - OZONE_CENTER_HEIGHT);
    return max(0.0, 1.0 - dist_from_center / OZONE_WIDTH);
}

// ============ Phase Functions ============

// Rayleigh phase function
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

// Henyey-Greenstein phase function for Mie scattering
fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Cornette-Shanks phase function (improved Mie, used in some implementations)
fn cornette_shanks_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_theta * cos_theta);
    let denom = (8.0 * PI) * (2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return num / denom;
}

// ============ Ray-Sphere Intersection ============

// Returns distance to intersection with sphere, or -1 if no intersection
// ro: ray origin, rd: ray direction, center: sphere center, radius: sphere radius
fn ray_sphere_intersect(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, radius: f32) -> vec2<f32> {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - c;

    if (discriminant < 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }

    let sqrt_disc = sqrt(discriminant);
    return vec2<f32>(-b - sqrt_disc, -b + sqrt_disc);
}

// ============ Optical Depth Integration ============

// Compute optical depth along a ray from point to atmosphere edge
fn compute_optical_depth(ray_origin: vec3<f32>, ray_dir: vec3<f32>, ray_length: f32, num_samples: i32) -> vec3<f32> {
    let step_size = ray_length / f32(num_samples);
    var optical_depth_r = 0.0;
    var optical_depth_m = 0.0;
    var optical_depth_o = 0.0;

    let planet_center = vec3<f32>(0.0, -PLANET_RADIUS, 0.0);

    for (var i = 0; i < num_samples; i++) {
        let t = (f32(i) + 0.5) * step_size;
        let sample_pos = ray_origin + ray_dir * t;
        let height = length(sample_pos - planet_center) - PLANET_RADIUS;

        optical_depth_r += rayleigh_density(height) * step_size;
        optical_depth_m += mie_density(height) * step_size;
        optical_depth_o += ozone_density(height) * step_size;
    }

    return vec3<f32>(optical_depth_r, optical_depth_m, optical_depth_o);
}

// ============ Main Atmospheric Scattering ============

struct ScatteringResult {
    inscattering: vec3<f32>,
    transmittance: vec3<f32>,
}

fn compute_atmosphere(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sun_dir: vec3<f32>, num_samples: i32, num_light_samples: i32) -> ScatteringResult {
    var result: ScatteringResult;
    result.inscattering = vec3<f32>(0.0);
    result.transmittance = vec3<f32>(1.0);

    let planet_center = vec3<f32>(0.0, -PLANET_RADIUS, 0.0);

    // Find atmosphere intersection
    let atmo_intersection = ray_sphere_intersect(ray_origin, ray_dir, planet_center, ATMOSPHERE_RADIUS);
    if (atmo_intersection.y < 0.0) {
        return result;
    }

    // Check for planet intersection
    let planet_intersection = ray_sphere_intersect(ray_origin, ray_dir, planet_center, PLANET_RADIUS);

    let t_start = max(0.0, atmo_intersection.x);
    var t_end = atmo_intersection.y;

    // If ray hits planet, stop there
    if (planet_intersection.x > 0.0) {
        t_end = planet_intersection.x;
    }

    let ray_length = t_end - t_start;
    if (ray_length <= 0.0) {
        return result;
    }

    let step_size = ray_length / f32(num_samples);

    // Scattering coefficients
    let beta_r = get_rayleigh_scattering();
    let beta_m = get_mie_scattering();
    let beta_m_ext = get_mie_extinction();
    let beta_o = get_ozone_absorption();

    // Phase functions
    let cos_theta = dot(ray_dir, sun_dir);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = henyey_greenstein_phase(cos_theta, uniforms.mie_directional_g);

    // Accumulated optical depth
    var optical_depth_r = 0.0;
    var optical_depth_m = 0.0;
    var optical_depth_o = 0.0;

    // Ray marching
    for (var i = 0; i < num_samples; i++) {
        let t = t_start + (f32(i) + 0.5) * step_size;
        let sample_pos = ray_origin + ray_dir * t;
        let height = length(sample_pos - planet_center) - PLANET_RADIUS;

        // Local density at this point
        let density_r = rayleigh_density(height);
        let density_m = mie_density(height);
        let density_o = ozone_density(height);

        // Accumulate optical depth
        optical_depth_r += density_r * step_size;
        optical_depth_m += density_m * step_size;
        optical_depth_o += density_o * step_size;

        // Compute transmittance from camera to this point
        let extinction = beta_r * optical_depth_r +
                        beta_m_ext * optical_depth_m +
                        beta_o * optical_depth_o;
        let transmittance_view = exp(-extinction);

        // Now compute transmittance from this point to sun
        let sun_ray = ray_sphere_intersect(sample_pos, sun_dir, planet_center, ATMOSPHERE_RADIUS);
        if (sun_ray.y > 0.0) {
            // Check if sun ray hits planet (in shadow)
            let sun_planet = ray_sphere_intersect(sample_pos, sun_dir, planet_center, PLANET_RADIUS);

            if (sun_planet.x < 0.0 || sun_planet.x > sun_ray.y) {
                // Not in planet shadow - compute optical depth to sun
                let sun_optical_depth = compute_optical_depth(sample_pos, sun_dir, sun_ray.y, num_light_samples);

                let extinction_sun = beta_r * sun_optical_depth.x +
                                    beta_m_ext * sun_optical_depth.y +
                                    beta_o * sun_optical_depth.z;
                let transmittance_sun = exp(-extinction_sun);

                // Compute in-scattered light at this point
                let scattering_r = beta_r * density_r;
                let scattering_m = beta_m * density_m;

                let inscattered = (scattering_r * phase_r + scattering_m * phase_m) *
                                 transmittance_view * transmittance_sun * step_size;

                result.inscattering += inscattered;
            }
        }
    }

    // Final transmittance
    let final_extinction = beta_r * optical_depth_r +
                          beta_m_ext * optical_depth_m +
                          beta_o * optical_depth_o;
    result.transmittance = exp(-final_extinction);

    return result;
}

// Multi-scattering approximation (simplified Bruneton approach)
fn multi_scattering_approx(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    // Approximate second-order scattering as isotropic ambient term
    // This brightens the sky, especially at horizons
    let y = max(0.0, ray_dir.y);
    let sun_height = max(0.0, sun_dir.y);

    // Multi-scatter increases towards horizon and with sun height
    let horizon_factor = pow(1.0 - y, 2.0);
    let sun_factor = sun_height * 0.5 + 0.5;

    // Use average scattering color
    let avg_scatter = (get_rayleigh_scattering() + get_mie_scattering()) * 0.5;

    return avg_scatter * horizon_factor * sun_factor * 0.15;
}

// ============ Sun Disk with Transmittance ============

fn sun_disk_with_transmittance(ray_dir: vec3<f32>, sun_dir: vec3<f32>, atmosphere_transmittance: vec3<f32>) -> vec3<f32> {
    let cos_theta = dot(ray_dir, sun_dir);
    let sun_angular_radius = uniforms.sun_disk_size * 0.00873; // Convert to radians
    let cos_sun_radius = cos(sun_angular_radius);

    var sun_contribution = vec3<f32>(0.0);

    // Sun disk luminance (physically, sun is ~1.6e9 cd/m^2, but we use artistic values)
    let sun_luminance = vec3<f32>(1.0, 0.98, 0.92) * uniforms.sun_disk_intensity;

    if (cos_theta > cos_sun_radius) {
        // Inside sun disk
        // Soft edge to prevent bloom flickering
        let soft_edge = saturate(2.0 * (cos_theta - cos_sun_radius) / (1.0 - cos_sun_radius));
        sun_contribution = sun_luminance * soft_edge;
    }

    // Corona/glow around sun
    let glow_radius = sun_angular_radius * 4.0;
    let cos_glow = cos(glow_radius);
    if (cos_theta > cos_glow && cos_theta <= cos_sun_radius) {
        let glow_factor = (cos_theta - cos_glow) / (cos_sun_radius - cos_glow);
        let glow = pow(glow_factor, 2.0) * 0.3;
        sun_contribution += sun_luminance * glow;
    }

    // Apply atmospheric transmittance to sun
    sun_contribution *= atmosphere_transmittance;

    // Sunset/sunrise color shift (increased scattering at low angles)
    let sun_height = sun_dir.y;
    if (sun_height < 0.2 && sun_height > -0.1) {
        let sunset_t = 1.0 - (sun_height + 0.1) / 0.3;
        // Red shift due to longer atmospheric path
        sun_contribution.r *= 1.0 + sunset_t * 0.5;
        sun_contribution.g *= 1.0 - sunset_t * 0.2;
        sun_contribution.b *= 1.0 - sunset_t * 0.6;
    }

    return sun_contribution;
}

// ============ Noise functions for clouds ============

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

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

fn cloud_density(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    if (ray_dir.y < 0.05) {
        return 0.0;
    }

    let cloud_height = 0.15;
    let t = cloud_height / max(ray_dir.y, 0.001);
    let cloud_pos = ray_dir * t;

    let cloud_scale = 3.0;
    let wind_direction = vec2<f32>(1.0, 0.3);
    let wind_offset = wind_direction * uniforms.time * uniforms.cloud_speed;
    let cloud_uv = cloud_pos.xz * cloud_scale + wind_offset;

    let noise_pos = vec3<f32>(cloud_uv.x, cloud_uv.y, uniforms.time * uniforms.cloud_speed * 0.1);
    var density = fbm(noise_pos, 5);

    let coverage = 0.45;
    density = smoothstep(coverage, coverage + 0.3, density);

    let horizon_fade = smoothstep(0.05, 0.2, ray_dir.y);
    density *= horizon_fade;

    let zenith_fade = 1.0 - smoothstep(0.6, 1.0, ray_dir.y) * 0.3;
    density *= zenith_fade;

    return density;
}

fn render_clouds(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sky_color: vec3<f32>) -> vec3<f32> {
    let density = cloud_density(ray_dir, sun_dir);

    if (density <= 0.0) {
        return vec3<f32>(0.0);
    }

    var cloud_color = vec3<f32>(1.0, 1.0, 1.0);

    let sun_dot = dot(ray_dir, sun_dir);
    let light_factor = sun_dot * 0.3 + 0.7;

    let sun_height = sun_dir.y;
    if (sun_height < 0.3) {
        let sunset_t = 1.0 - sun_height / 0.3;
        cloud_color = mix(cloud_color, vec3<f32>(1.0, 0.7, 0.4), sunset_t * 0.6);
    }

    let shadow = smoothstep(0.0, 0.5, density) * 0.3;
    cloud_color *= (light_factor - shadow + 0.3);

    let brightness = max(0.4, sun_height * 0.6 + 0.6) * uniforms.sun_intensity * 0.1;
    cloud_color *= brightness;

    if (sun_height < 0.0) {
        cloud_color *= smoothstep(-0.3, 0.0, sun_height);
    }

    return cloud_color * density;
}

// ============ Stars ============

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

// ============ Moon ============

fn moon(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let moon_dir = -sun_dir;

    if (sun_dir.y > 0.1) {
        return vec3<f32>(0.0);
    }

    let moon_visibility = smoothstep(0.1, -0.1, sun_dir.y);

    let cos_theta = dot(ray_dir, moon_dir);
    let moon_angular_radius = 0.009;

    var moon_contribution = vec3<f32>(0.0);

    let moon_cos = cos(moon_angular_radius);
    if (cos_theta > moon_cos) {
        let edge_factor = (cos_theta - moon_cos) / (1.0 - moon_cos);
        let moon_base = vec3<f32>(0.95, 0.93, 0.88);
        let phase_shading = smoothstep(0.0, 1.0, edge_factor);
        let moon_intensity = phase_shading * 2.0;
        moon_contribution = moon_base * moon_intensity;
    }

    let glow_size = moon_angular_radius * 4.0;
    let glow_cos = cos(glow_size);
    if (cos_theta > glow_cos) {
        let glow_factor = (cos_theta - glow_cos) / (moon_cos - glow_cos);
        let glow = pow(max(0.0, glow_factor), 2.0) * 0.3;
        moon_contribution += vec3<f32>(0.8, 0.85, 0.95) * glow;
    }

    return moon_contribution * moon_visibility;
}

// ============ Main Fragment Shader ============

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = normalize(in.ray_dir);
    let sun_dir = normalize(uniforms.sun_direction);

    // Camera is slightly above ground (1.7m = human eye height, scaled)
    let camera_height = 0.0017; // km above sea level
    let ray_origin = vec3<f32>(0.0, camera_height, 0.0);

    var color: vec3<f32>;

    // Below horizon - ground rendering
    if (ray_dir.y < 0.0) {
        // Get sky color at horizon for blending
        let horizon_result = compute_atmosphere(
            ray_origin,
            vec3<f32>(ray_dir.x, 0.01, ray_dir.z),
            sun_dir,
            16,
            4
        );
        let horizon_sky = horizon_result.inscattering * uniforms.sun_intensity;

        // Ground with ambient based on sun position
        let ground_ambient = max(0.15, sun_dir.y * 0.4 + 0.5);
        let ground = uniforms.ground_color * ground_ambient;

        // Smooth blend from ground to horizon sky
        let horizon_blend = smoothstep(-0.15, 0.0, ray_dir.y);
        color = mix(ground, horizon_sky, horizon_blend);
    } else {
        // Main atmospheric scattering (reduced samples for performance)
        // Use 32 samples for main ray, 8 for light rays
        let atmo_result = compute_atmosphere(ray_origin, ray_dir, sun_dir, 32, 8);

        // Apply sun intensity to inscattering
        color = atmo_result.inscattering * uniforms.sun_intensity;

        // Add multi-scattering approximation
        color += multi_scattering_approx(ray_dir, sun_dir) * uniforms.sun_intensity;

        // Add stars at night (before clouds)
        color += stars(ray_dir, sun_dir.y);

        // Add moon at night
        color += moon(ray_dir, sun_dir);

        // Add clouds
        let cloud_color = render_clouds(ray_dir, sun_dir, color);
        let cloud_dens = cloud_density(ray_dir, sun_dir);
        color = mix(color, cloud_color / max(cloud_dens, 0.001), cloud_dens * 0.9);

        // Add sun disk with proper atmospheric transmittance
        if (sun_dir.y > -0.1) {
            color += sun_disk_with_transmittance(ray_dir, sun_dir, atmo_result.transmittance);
        }
    }

    // Apply exposure
    color *= uniforms.exposure;

    return vec4<f32>(color, 1.0);
}
