//! GPU buffer management for particle systems.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::forces::{ForceFieldGpu, MAX_FORCE_FIELDS};
use super::particle::{EmissionParams, ParticleGpu};

/// GPU buffers for particle simulation.
pub struct ParticleGpuResources {
    /// Storage buffer for particle state (read/write in compute).
    pub particle_buffer: wgpu::Buffer,
    /// Buffer for indirect draw arguments.
    pub indirect_buffer: wgpu::Buffer,
    /// Atomic counter buffer.
    pub counter_buffer: wgpu::Buffer,
    /// Emission parameters uniform.
    pub emission_buffer: wgpu::Buffer,
    /// Force fields storage buffer.
    pub forces_buffer: wgpu::Buffer,
    /// Simulation uniforms (time, delta, etc.).
    pub simulation_buffer: wgpu::Buffer,
    /// Dead particle index list (for recycling).
    pub dead_list_buffer: wgpu::Buffer,
    /// Alive particle index list (for compact rendering).
    pub alive_list_buffer: wgpu::Buffer,
    /// Maximum particle count.
    pub max_particles: u32,
}

impl ParticleGpuResources {
    /// Create new GPU resources for a particle system.
    pub fn new(device: &wgpu::Device, max_particles: u32) -> Self {
        // Particle storage buffer
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Buffer"),
            size: (max_particles as usize * std::mem::size_of::<ParticleGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dead list buffer - initially all particles are dead
        let dead_indices: Vec<u32> = (0..max_particles).collect();
        let dead_list_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dead List Buffer"),
            contents: bytemuck::cast_slice(&dead_indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Alive list buffer
        let alive_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alive List Buffer"),
            size: (max_particles as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Counter buffer: [alive_count, dead_count, emit_count, frame_random]
        let initial_counters = CounterData {
            alive_count: 0,
            dead_count: max_particles,
            emit_count: 0,
            _padding: 0,
        };
        let counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Counter Buffer"),
            contents: bytemuck::cast_slice(&[initial_counters]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Indirect draw buffer
        let indirect_args = IndirectDrawArgs {
            vertex_count: 4,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Indirect Buffer"),
            contents: bytemuck::cast_slice(&[indirect_args]),
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        // Emission uniform buffer
        let emission_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Emission Buffer"),
            contents: bytemuck::cast_slice(&[EmissionParams::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Forces storage buffer
        let forces: [ForceFieldGpu; MAX_FORCE_FIELDS] = Default::default();
        let forces_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Forces Buffer"),
            contents: bytemuck::cast_slice(&forces),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Simulation uniform buffer
        let simulation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Buffer"),
            contents: bytemuck::cast_slice(&[SimulationUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            particle_buffer,
            indirect_buffer,
            counter_buffer,
            emission_buffer,
            forces_buffer,
            simulation_buffer,
            dead_list_buffer,
            alive_list_buffer,
            max_particles,
        }
    }

    /// Reset the particle system (kill all particles).
    pub fn reset(&self, queue: &wgpu::Queue) {
        // Reset counters
        let counters = CounterData {
            alive_count: 0,
            dead_count: self.max_particles,
            emit_count: 0,
            _padding: 0,
        };
        queue.write_buffer(&self.counter_buffer, 0, bytemuck::cast_slice(&[counters]));

        // Reset dead list to all indices
        let dead_indices: Vec<u32> = (0..self.max_particles).collect();
        queue.write_buffer(&self.dead_list_buffer, 0, bytemuck::cast_slice(&dead_indices));

        // Reset indirect draw
        let indirect = IndirectDrawArgs {
            vertex_count: 4,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        queue.write_buffer(&self.indirect_buffer, 0, bytemuck::cast_slice(&[indirect]));
    }
}

/// Counter data for atomic operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CounterData {
    /// Number of alive particles.
    pub alive_count: u32,
    /// Number of dead particles available for spawning.
    pub dead_count: u32,
    /// Number of particles to emit this frame.
    pub emit_count: u32,
    /// Padding.
    pub _padding: u32,
}

/// Indirect draw arguments for GPU-driven rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IndirectDrawArgs {
    /// Vertex count (always 4 for quad).
    pub vertex_count: u32,
    /// Instance count (= alive particles, written by compute).
    pub instance_count: u32,
    /// First vertex.
    pub first_vertex: u32,
    /// First instance.
    pub first_instance: u32,
}

/// Simulation uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SimulationUniform {
    /// Time: x=total_time, y=delta_time, z=frame_count, w=random_seed.
    pub time_params: [f32; 4],
    /// Spawn: x=max_particles, y=emission_rate, z=particles_to_spawn, w=soft_fade_distance.
    pub spawn_params: [f32; 4],
    /// Force: x=force_count, y=unused, z=unused, w=unused.
    pub force_params: [f32; 4],
    /// Bounds min (xyz) + bounds_enabled (w).
    pub bounds_min: [f32; 4],
    /// Bounds max (xyz) + unused (w).
    pub bounds_max: [f32; 4],
}

impl Default for SimulationUniform {
    fn default() -> Self {
        Self {
            time_params: [0.0, 0.016, 0.0, 0.0],
            spawn_params: [1000.0, 50.0, 0.0, 0.5],
            force_params: [0.0, 0.0, 0.0, 0.0],
            bounds_min: [-1000.0, -1000.0, -1000.0, 0.0],
            bounds_max: [1000.0, 1000.0, 1000.0, 0.0],
        }
    }
}

/// Render parameters uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ParticleRenderUniform {
    /// Resolution: x=width, y=height, z=1/width, w=1/height.
    pub resolution: [f32; 4],
    /// Params: x=soft_fade_distance, y=texture_rows, z=texture_cols, w=unused.
    pub params: [f32; 4],
    /// Camera near/far: x=near, y=far, z=unused, w=unused.
    pub camera_params: [f32; 4],
    /// Camera right vector (xyz) for billboarding, w=unused.
    pub camera_right: [f32; 4],
    /// Camera up vector (xyz) for billboarding, w=unused.
    pub camera_up: [f32; 4],
}

impl Default for ParticleRenderUniform {
    fn default() -> Self {
        Self {
            resolution: [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0],
            params: [0.5, 1.0, 1.0, 0.0],
            camera_params: [0.1, 1000.0, 0.0, 0.0],
            camera_right: [1.0, 0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0, 0.0],
        }
    }
}
