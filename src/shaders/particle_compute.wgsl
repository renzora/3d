// Particle Compute Shader
// GPU-based particle simulation with emit, update, and compact kernels

const PI: f32 = 3.14159265359;

// Particle data structure (must match Rust ParticleGpu)
struct Particle {
    position_age: vec4<f32>,      // xyz=position, w=age
    velocity_lifetime: vec4<f32>, // xyz=velocity, w=lifetime
    color: vec4<f32>,             // rgba
    size_rotation: vec4<f32>,     // x=current_size, y=start_size, z=end_size, w=rotation
    flags: vec4<f32>,             // x=alive, y=tex_idx, z=seed, w=rotation_speed
    start_color: vec4<f32>,       // Start color for interpolation
    end_color: vec4<f32>,         // End color for interpolation
}

// Counters for atomic operations
struct Counters {
    alive_count: atomic<u32>,
    dead_count: atomic<u32>,
    emit_count: atomic<u32>,
    _padding: u32,
}

// Simulation parameters
struct SimulationUniform {
    time_params: vec4<f32>,   // x=total_time, y=delta_time, z=frame, w=random_seed
    spawn_params: vec4<f32>,  // x=max_particles, y=emission_rate, z=spawn_count, w=soft_fade
    force_params: vec4<f32>,  // x=force_count, y=unused, z=unused, w=unused
    bounds_min: vec4<f32>,    // xyz=min, w=enabled
    bounds_max: vec4<f32>,    // xyz=max, w=unused
}

// Emission parameters
struct EmissionParams {
    emitter_pos_rate: vec4<f32>,      // xyz=position, w=rate
    emitter_dir_spread: vec4<f32>,    // xyz=direction, w=spread
    velocity_min_lifetime: vec4<f32>, // xyz=vel_min, w=lifetime_min
    velocity_max_lifetime: vec4<f32>, // xyz=vel_max, w=lifetime_max
    start_color: vec4<f32>,
    end_color: vec4<f32>,
    size_params: vec4<f32>,           // x=start_min, y=start_max, z=end_min, w=end_max
    emitter_shape: vec4<f32>,         // x=type, y=param1, z=param2, w=param3
    rotation_params: vec4<f32>,       // x=min_speed, y=max_speed
    start_color_variance: vec4<f32>,
    end_color_variance: vec4<f32>,
    box_half_extents: vec4<f32>,
}

// Force field
struct ForceField {
    type_enabled: vec4<f32>,       // x=type, w=enabled
    position_strength: vec4<f32>,  // xyz=pos/dir, w=strength
    params: vec4<f32>,             // Additional params
}

// Indirect draw arguments
struct IndirectArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

// Storage buffers
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> dead_list: array<u32>;
@group(0) @binding(2) var<storage, read_write> alive_list: array<u32>;
@group(0) @binding(3) var<storage, read_write> counters: Counters;
@group(0) @binding(4) var<storage, read_write> indirect: IndirectArgs;

// Uniform buffers
@group(1) @binding(0) var<uniform> sim: SimulationUniform;
@group(1) @binding(1) var<uniform> emission: EmissionParams;
@group(1) @binding(2) var<storage, read> forces: array<ForceField>;

// ============ Random Number Generation ============

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random_float(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / f32(0xFFFFFFFFu);
}

fn random_range(seed: ptr<function, u32>, min_val: f32, max_val: f32) -> f32 {
    return mix(min_val, max_val, random_float(seed));
}

fn random_vec3(seed: ptr<function, u32>, min_val: vec3<f32>, max_val: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        random_range(seed, min_val.x, max_val.x),
        random_range(seed, min_val.y, max_val.y),
        random_range(seed, min_val.z, max_val.z)
    );
}

// ============ Emitter Shape Sampling ============

fn sample_point_emitter() -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn sample_sphere_emitter(seed: ptr<function, u32>, radius: f32, is_volume: bool) -> vec3<f32> {
    let theta = random_float(seed) * 2.0 * PI;
    let phi = acos(1.0 - 2.0 * random_float(seed));
    var r = radius;
    if (is_volume) {
        r = radius * pow(random_float(seed), 1.0 / 3.0);
    }
    return r * vec3<f32>(
        sin(phi) * cos(theta),
        cos(phi),
        sin(phi) * sin(theta)
    );
}

fn sample_cone_emitter(seed: ptr<function, u32>, angle: f32, height: f32) -> vec3<f32> {
    let theta = random_float(seed) * 2.0 * PI;
    let r = random_float(seed) * tan(angle) * height;
    return vec3<f32>(r * cos(theta), height, r * sin(theta));
}

fn sample_box_emitter(seed: ptr<function, u32>, half_extents: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        (random_float(seed) * 2.0 - 1.0) * half_extents.x,
        (random_float(seed) * 2.0 - 1.0) * half_extents.y,
        (random_float(seed) * 2.0 - 1.0) * half_extents.z
    );
}

fn sample_emitter_position(seed: ptr<function, u32>) -> vec3<f32> {
    let shape_type = u32(emission.emitter_shape.x);

    switch (shape_type) {
        case 0u: { // Point
            return sample_point_emitter();
        }
        case 1u: { // Sphere surface
            return sample_sphere_emitter(seed, emission.emitter_shape.y, false);
        }
        case 2u: { // Sphere volume
            return sample_sphere_emitter(seed, emission.emitter_shape.y, true);
        }
        case 3u: { // Cone
            return sample_cone_emitter(seed, emission.emitter_shape.y, emission.emitter_shape.z);
        }
        case 4u: { // Box
            return sample_box_emitter(seed, emission.box_half_extents.xyz);
        }
        default: {
            return vec3<f32>(0.0);
        }
    }
}

// ============ Direction with Spread ============

fn apply_spread(base_dir: vec3<f32>, spread: f32, seed: ptr<function, u32>) -> vec3<f32> {
    if (spread < 0.001) {
        return base_dir;
    }

    // Random angle within spread cone
    let theta = random_float(seed) * 2.0 * PI;
    let phi = random_float(seed) * spread;

    // Create rotation to align with base direction
    let up = vec3<f32>(0.0, 1.0, 0.0);
    var right: vec3<f32>;
    if (abs(dot(base_dir, up)) > 0.99) {
        right = normalize(cross(base_dir, vec3<f32>(1.0, 0.0, 0.0)));
    } else {
        right = normalize(cross(base_dir, up));
    }
    let forward = normalize(cross(right, base_dir));

    // Generate direction within cone
    let local_dir = vec3<f32>(
        sin(phi) * cos(theta),
        cos(phi),
        sin(phi) * sin(theta)
    );

    // Transform to world space
    return normalize(
        local_dir.x * right +
        local_dir.y * base_dir +
        local_dir.z * forward
    );
}

// ============ Force Application ============

fn apply_forces(position: vec3<f32>, velocity: vec3<f32>, delta_time: f32, seed: ptr<function, u32>) -> vec3<f32> {
    var new_velocity = velocity;
    let force_count = u32(sim.force_params.x);
    let time = sim.time_params.x;

    for (var i = 0u; i < force_count && i < 8u; i++) {
        let force = forces[i];
        let force_type = u32(force.type_enabled.x);
        let enabled = force.type_enabled.w > 0.5;

        if (!enabled) {
            continue;
        }

        switch (force_type) {
            case 0u: { // Directional (gravity, wind)
                let direction = force.position_strength.xyz;
                let strength = force.position_strength.w;
                new_velocity += direction * strength * delta_time;
            }
            case 1u: { // Point attractor/repulsor
                let center = force.position_strength.xyz;
                let strength = force.position_strength.w;
                let radius = force.params.x;

                let to_center = center - position;
                let dist = length(to_center);

                if (dist < radius && dist > 0.001) {
                    let falloff = 1.0 - (dist / radius);
                    new_velocity += normalize(to_center) * strength * falloff * falloff * delta_time;
                }
            }
            case 2u: { // Turbulence
                let frequency = force.params.x;
                let amplitude = force.params.y;

                // Simple 3D noise-based turbulence
                let noise_pos = position * frequency + vec3<f32>(time * 0.5);
                let turbulence = vec3<f32>(
                    sin(noise_pos.x * 1.7 + noise_pos.y * 2.3) * cos(noise_pos.z * 1.9),
                    sin(noise_pos.y * 2.1 + noise_pos.z * 1.5) * cos(noise_pos.x * 2.7),
                    sin(noise_pos.z * 1.3 + noise_pos.x * 2.9) * cos(noise_pos.y * 1.1)
                );
                new_velocity += turbulence * amplitude * delta_time;
            }
            case 3u: { // Vortex
                let axis = normalize(force.position_strength.xyz);
                let strength = force.position_strength.w;
                let center = force.params.xyz;

                let to_center = position - center;
                let tangent = cross(axis, to_center);
                let tangent_len = length(tangent);

                if (tangent_len > 0.001) {
                    new_velocity += normalize(tangent) * strength * delta_time;
                }
            }
            case 4u: { // Drag
                let drag = force.params.x;
                new_velocity *= 1.0 - min(drag * delta_time, 0.99);
            }
            default: {}
        }
    }

    return new_velocity;
}

// ============ Emit Kernel ============

@compute @workgroup_size(64)
fn emit_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let spawn_count = u32(sim.spawn_params.z);
    if (global_id.x >= spawn_count) {
        return;
    }

    // Try to get a dead particle
    let dead_index = atomicSub(&counters.dead_count, 1u);
    if (dead_index == 0u) {
        // No dead particles available, restore counter
        atomicAdd(&counters.dead_count, 1u);
        return;
    }

    let particle_index = dead_list[dead_index - 1u];

    // Initialize random seed
    var seed = global_id.x ^ u32(sim.time_params.w) ^ (particle_index * 1973u);

    // Sample emitter position
    let local_pos = sample_emitter_position(&seed);
    let world_pos = emission.emitter_pos_rate.xyz + local_pos;

    // Random velocity within range
    let base_vel = random_vec3(
        &seed,
        emission.velocity_min_lifetime.xyz,
        emission.velocity_max_lifetime.xyz
    );

    // Apply spread to direction
    let base_dir = normalize(emission.emitter_dir_spread.xyz);
    let spread = emission.emitter_dir_spread.w;
    let spread_dir = apply_spread(base_dir, spread, &seed);
    let vel = spread_dir * length(base_vel);

    // Random lifetime
    let lifetime = random_range(
        &seed,
        emission.velocity_min_lifetime.w,
        emission.velocity_max_lifetime.w
    );

    // Random size
    let size_start = random_range(&seed, emission.size_params.x, emission.size_params.y);
    let size_end = random_range(&seed, emission.size_params.z, emission.size_params.w);

    // Random rotation speed
    let rotation_speed = random_range(&seed, emission.rotation_params.x, emission.rotation_params.y);
    let initial_rotation = random_float(&seed) * 2.0 * PI;

    // Random color with variance
    let start_color = clamp(
        emission.start_color.xyz + (random_vec3(&seed, vec3<f32>(-1.0), vec3<f32>(1.0))) * emission.start_color_variance.xyz,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
    let start_alpha = clamp(
        emission.start_color.w + (random_float(&seed) * 2.0 - 1.0) * emission.start_color_variance.w,
        0.0,
        1.0
    );

    let end_color = clamp(
        emission.end_color.xyz + (random_vec3(&seed, vec3<f32>(-1.0), vec3<f32>(1.0))) * emission.end_color_variance.xyz,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
    let end_alpha = clamp(
        emission.end_color.w + (random_float(&seed) * 2.0 - 1.0) * emission.end_color_variance.w,
        0.0,
        1.0
    );

    // Initialize particle
    var particle: Particle;
    particle.position_age = vec4<f32>(world_pos, 0.0);
    particle.velocity_lifetime = vec4<f32>(vel, lifetime);
    particle.color = vec4<f32>(start_color, start_alpha);
    particle.size_rotation = vec4<f32>(size_start, size_start, size_end, initial_rotation);
    particle.flags = vec4<f32>(1.0, 0.0, f32(seed), rotation_speed);
    particle.start_color = vec4<f32>(start_color, start_alpha);
    particle.end_color = vec4<f32>(end_color, end_alpha);

    particles[particle_index] = particle;
}

// ============ Update Kernel ============

@compute @workgroup_size(64)
fn update_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let max_particles = u32(sim.spawn_params.x);
    if (global_id.x >= max_particles) {
        return;
    }

    var particle = particles[global_id.x];

    // Skip dead particles
    if (particle.flags.x < 0.5) {
        return;
    }

    let delta_time = sim.time_params.y;
    let age = particle.position_age.w + delta_time;
    let lifetime = particle.velocity_lifetime.w;

    // Kill if exceeded lifetime
    if (age >= lifetime) {
        particle.flags.x = 0.0; // Mark as dead
        particles[global_id.x] = particle;

        // Add to dead list
        let dead_index = atomicAdd(&counters.dead_count, 1u);
        dead_list[dead_index] = global_id.x;
        return;
    }

    // Random seed for turbulence
    var seed = u32(particle.flags.z);

    // Apply forces
    var velocity = particle.velocity_lifetime.xyz;
    velocity = apply_forces(particle.position_age.xyz, velocity, delta_time, &seed);

    // Integrate position
    let position = particle.position_age.xyz + velocity * delta_time;

    // Check bounds
    let bounds_enabled = sim.bounds_min.w > 0.5;
    if (bounds_enabled) {
        if (position.x < sim.bounds_min.x || position.x > sim.bounds_max.x ||
            position.y < sim.bounds_min.y || position.y > sim.bounds_max.y ||
            position.z < sim.bounds_min.z || position.z > sim.bounds_max.z) {
            // Kill particle
            particle.flags.x = 0.0;
            particles[global_id.x] = particle;
            let dead_index = atomicAdd(&counters.dead_count, 1u);
            dead_list[dead_index] = global_id.x;
            return;
        }
    }

    // Interpolate properties
    let t = age / lifetime;
    let size = mix(particle.size_rotation.y, particle.size_rotation.z, t);
    let color = mix(particle.start_color, particle.end_color, t);

    // Update rotation
    let rotation_speed = particle.flags.w;
    let rotation = particle.size_rotation.w + rotation_speed * delta_time;

    // Write back
    particle.position_age = vec4<f32>(position, age);
    particle.velocity_lifetime = vec4<f32>(velocity, lifetime);
    particle.size_rotation.x = size;
    particle.size_rotation.w = rotation;
    particle.color = color;

    particles[global_id.x] = particle;

    // Add to alive list for rendering
    let alive_index = atomicAdd(&counters.alive_count, 1u);
    alive_list[alive_index] = global_id.x;
}

// ============ Finalize Kernel ============
// Copies alive count to indirect draw buffer

@compute @workgroup_size(1)
fn finalize_indirect() {
    let alive_count = atomicLoad(&counters.alive_count);
    atomicStore(&indirect.instance_count, alive_count);
}

// ============ Reset Kernel ============
// Resets counters at the start of each frame

@compute @workgroup_size(1)
fn reset_counters() {
    atomicStore(&counters.alive_count, 0u);
    atomicStore(&counters.emit_count, 0u);
}
