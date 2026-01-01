//! Main ParticleSystem struct for GPU-accelerated particle simulation.

use wgpu::util::DeviceExt;

use super::emitter::{EmitterConfig, ParticleBlendMode, ParticlePreset};
use super::forces::{ForceFieldGpu, ForceType, MAX_FORCE_FIELDS};
use super::gpu_resources::{ParticleGpuResources, ParticleRenderUniform, SimulationUniform};
use crate::core::Id;

/// GPU-accelerated particle system.
pub struct ParticleSystem {
    /// Unique identifier.
    id: Id,
    /// System name.
    name: String,
    /// Emitter configuration.
    config: EmitterConfig,
    /// GPU resources.
    gpu: Option<ParticleGpuResources>,

    // Compute pipelines
    reset_pipeline: Option<wgpu::ComputePipeline>,
    emit_pipeline: Option<wgpu::ComputePipeline>,
    update_pipeline: Option<wgpu::ComputePipeline>,
    finalize_pipeline: Option<wgpu::ComputePipeline>,

    // Render pipelines
    additive_pipeline: Option<wgpu::RenderPipeline>,
    alpha_blend_pipeline: Option<wgpu::RenderPipeline>,

    // Bind group layouts
    compute_storage_layout: Option<wgpu::BindGroupLayout>,
    compute_uniform_layout: Option<wgpu::BindGroupLayout>,
    render_particle_layout: Option<wgpu::BindGroupLayout>,
    render_texture_layout: Option<wgpu::BindGroupLayout>,

    // Bind groups
    compute_storage_bind_group: Option<wgpu::BindGroup>,
    compute_uniform_bind_group: Option<wgpu::BindGroup>,
    render_particle_bind_group: Option<wgpu::BindGroup>,
    render_texture_bind_group: Option<wgpu::BindGroup>,

    // Render uniform buffer
    render_uniform_buffer: Option<wgpu::Buffer>,

    /// Active force fields.
    forces: Vec<ForceType>,
    /// World position of emitter.
    position: [f32; 3],
    /// Emission accumulator (for fractional particles per frame).
    emission_accumulator: f32,
    /// Is the system playing.
    playing: bool,
    /// Loop playback.
    looping: bool,
    /// Total elapsed time.
    elapsed_time: f32,
    /// Duration (0 = infinite).
    duration: f32,
    /// Visibility flag.
    pub visible: bool,

    // Cached render params
    width: u32,
    height: u32,
    near: f32,
    far: f32,
}

impl ParticleSystem {
    /// Create a new particle system with the given configuration.
    pub fn new(config: EmitterConfig) -> Self {
        Self {
            id: Id::new(),
            name: String::new(),
            config,
            gpu: None,
            reset_pipeline: None,
            emit_pipeline: None,
            update_pipeline: None,
            finalize_pipeline: None,
            additive_pipeline: None,
            alpha_blend_pipeline: None,
            compute_storage_layout: None,
            compute_uniform_layout: None,
            render_particle_layout: None,
            render_texture_layout: None,
            compute_storage_bind_group: None,
            compute_uniform_bind_group: None,
            render_particle_bind_group: None,
            render_texture_bind_group: None,
            render_uniform_buffer: None,
            forces: Vec::new(),
            position: [0.0, 0.0, 0.0],
            emission_accumulator: 0.0,
            playing: true,
            looping: true,
            elapsed_time: 0.0,
            duration: 0.0,
            visible: true,
            width: 1920,
            height: 1080,
            near: 0.1,
            far: 1000.0,
        }
    }

    /// Create a particle system from a preset.
    pub fn from_preset(preset: ParticlePreset) -> Self {
        let config = match preset {
            ParticlePreset::Fire => EmitterConfig::fire_preset(),
            ParticlePreset::Smoke => EmitterConfig::smoke_preset(),
            ParticlePreset::Sparks => EmitterConfig::sparks_preset(),
            ParticlePreset::Debris => EmitterConfig::debris_preset(),
            ParticlePreset::MagicEnergy => EmitterConfig::magic_preset(),
            ParticlePreset::Custom => EmitterConfig::default(),
        };
        Self::new(config)
    }

    /// Get the particle system ID.
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the configuration.
    pub fn config(&self) -> &EmitterConfig {
        &self.config
    }

    /// Get mutable configuration.
    pub fn config_mut(&mut self) -> &mut EmitterConfig {
        &mut self.config
    }

    /// Set the system name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Initialize GPU resources and pipelines.
    pub fn init(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        hdr_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        default_texture: &wgpu::TextureView,
        depth_texture: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        // Create GPU resources
        self.gpu = Some(ParticleGpuResources::new(device, self.config.max_particles));

        // Create bind group layouts
        self.create_bind_group_layouts(device);

        // Create compute pipelines
        self.create_compute_pipelines(device);

        // Create render pipelines
        self.create_render_pipelines(device, hdr_format, camera_bind_group_layout);

        // Create bind groups
        self.create_bind_groups(device, default_texture, depth_texture, sampler);
    }

    fn create_bind_group_layouts(&mut self, device: &wgpu::Device) {
        // Compute storage layout (group 0)
        self.compute_storage_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Particle Compute Storage Layout"),
                entries: &[
                    // Particles
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Dead list
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Alive list
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Counters
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Indirect
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }));

        // Compute uniform layout (group 1)
        self.compute_uniform_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Particle Compute Uniform Layout"),
                entries: &[
                    // Simulation uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Emission uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Forces storage
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }));

        // Render particle layout (group 1 for render)
        self.render_particle_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Particle Render Particle Layout"),
                entries: &[
                    // Particles (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Alive list (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Render params
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }));

        // Render texture layout (group 2 for render)
        self.render_texture_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Particle Render Texture Layout"),
                entries: &[
                    // Particle texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Depth texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            }));
    }

    fn create_compute_pipelines(&mut self, device: &wgpu::Device) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/particle_compute.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Compute Pipeline Layout"),
            bind_group_layouts: &[
                self.compute_storage_layout.as_ref().unwrap(),
                self.compute_uniform_layout.as_ref().unwrap(),
            ],
            push_constant_ranges: &[],
        });

        // Reset counters pipeline
        self.reset_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Particle Reset Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("reset_counters"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        // Emit pipeline
        self.emit_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Particle Emit Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("emit_particles"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        // Update pipeline
        self.update_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Particle Update Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("update_particles"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        // Finalize indirect pipeline
        self.finalize_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Particle Finalize Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("finalize_indirect"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
    }

    fn create_render_pipelines(
        &mut self,
        device: &wgpu::Device,
        hdr_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/particle_render.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Render Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,
                self.render_particle_layout.as_ref().unwrap(),
                self.render_texture_layout.as_ref().unwrap(),
            ],
            push_constant_ranges: &[],
        });

        // Depth stencil state (read-only)
        let depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        // Additive blend state
        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        // Alpha blend state
        let alpha_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        // Additive pipeline
        self.additive_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Particle Additive Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main_additive"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: hdr_format,
                        blend: Some(additive_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(depth_stencil.clone()),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));

        // Alpha blend pipeline
        self.alpha_blend_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Particle Alpha Blend Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: hdr_format,
                        blend: Some(alpha_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(depth_stencil),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));
    }

    fn create_bind_groups(
        &mut self,
        device: &wgpu::Device,
        default_texture: &wgpu::TextureView,
        depth_texture: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        let gpu = self.gpu.as_ref().unwrap();

        // Compute storage bind group
        self.compute_storage_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Compute Storage Bind Group"),
            layout: self.compute_storage_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu.dead_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu.alive_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu.counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu.indirect_buffer.as_entire_binding(),
                },
            ],
        }));

        // Compute uniform bind group
        self.compute_uniform_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Compute Uniform Bind Group"),
            layout: self.compute_uniform_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.simulation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu.emission_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu.forces_buffer.as_entire_binding(),
                },
            ],
        }));

        // Render uniform buffer
        let render_uniform = ParticleRenderUniform {
            resolution: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            params: [self.config.soft_fade_distance, 1.0, 1.0, 0.0],
            camera_params: [self.near, self.far, 0.0, 0.0],
            camera_right: [1.0, 0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0, 0.0],
        };
        self.render_uniform_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Particle Render Uniform Buffer"),
                contents: bytemuck::cast_slice(&[render_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        ));

        // Render particle bind group
        self.render_particle_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Render Particle Bind Group"),
            layout: self.render_particle_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu.alive_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.render_uniform_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));

        // Render texture bind group
        self.render_texture_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Render Texture Bind Group"),
            layout: self.render_texture_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(default_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_texture),
                },
            ],
        }));
    }

    /// Update the particle system (runs compute shaders).
    pub fn update(&mut self, encoder: &mut wgpu::CommandEncoder, delta_time: f32, queue: &wgpu::Queue) {
        if !self.playing || !self.visible || self.gpu.is_none() {
            return;
        }

        self.elapsed_time += delta_time;

        // Calculate how many particles to spawn
        self.emission_accumulator += self.config.emission_rate * delta_time;
        let spawn_count = self.emission_accumulator as u32;
        self.emission_accumulator -= spawn_count as f32;

        // Update simulation uniform
        let gpu = self.gpu.as_ref().unwrap();
        let sim_uniform = SimulationUniform {
            time_params: [
                self.elapsed_time,
                delta_time,
                0.0,
                (self.elapsed_time * 12345.6789).fract() * 1000000.0,
            ],
            spawn_params: [
                self.config.max_particles as f32,
                self.config.emission_rate,
                spawn_count as f32,
                self.config.soft_fade_distance,
            ],
            force_params: [self.forces.len() as f32, 0.0, 0.0, 0.0],
            bounds_min: [-1000.0, -1000.0, -1000.0, 0.0],
            bounds_max: [1000.0, 1000.0, 1000.0, 0.0],
        };
        queue.write_buffer(
            &gpu.simulation_buffer,
            0,
            bytemuck::cast_slice(&[sim_uniform]),
        );

        // Update emission params
        let emission_params = self.config.to_emission_params(self.position);
        queue.write_buffer(
            &gpu.emission_buffer,
            0,
            bytemuck::cast_slice(&[emission_params]),
        );

        // Update forces
        let mut force_data: [ForceFieldGpu; MAX_FORCE_FIELDS] = Default::default();
        for (i, force) in self.forces.iter().take(MAX_FORCE_FIELDS).enumerate() {
            force_data[i] = force.to_gpu();
        }
        queue.write_buffer(&gpu.forces_buffer, 0, bytemuck::cast_slice(&force_data));

        // Dispatch compute shaders
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Particle Compute Pass"),
                timestamp_writes: None,
            });

            let storage_bg = self.compute_storage_bind_group.as_ref().unwrap();
            let uniform_bg = self.compute_uniform_bind_group.as_ref().unwrap();

            // Reset counters
            if let Some(ref pipeline) = self.reset_pipeline {
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, storage_bg, &[]);
                compute_pass.set_bind_group(1, uniform_bg, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }

            // Emit new particles
            if spawn_count > 0 {
                if let Some(ref pipeline) = self.emit_pipeline {
                    compute_pass.set_pipeline(pipeline);
                    compute_pass.set_bind_group(0, storage_bg, &[]);
                    compute_pass.set_bind_group(1, uniform_bg, &[]);
                    compute_pass.dispatch_workgroups((spawn_count + 63) / 64, 1, 1);
                }
            }

            // Update existing particles
            if let Some(ref pipeline) = self.update_pipeline {
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, storage_bg, &[]);
                compute_pass.set_bind_group(1, uniform_bg, &[]);
                compute_pass.dispatch_workgroups((self.config.max_particles + 63) / 64, 1, 1);
            }

            // Finalize indirect draw args
            if let Some(ref pipeline) = self.finalize_pipeline {
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, storage_bg, &[]);
                compute_pass.set_bind_group(1, uniform_bg, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }
        }
    }

    /// Render particles.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, camera_bind_group: &'a wgpu::BindGroup) {
        if !self.visible || self.gpu.is_none() {
            return;
        }

        let gpu = self.gpu.as_ref().unwrap();

        // Select pipeline based on blend mode
        let pipeline = match self.config.blend_mode {
            ParticleBlendMode::Additive => self.additive_pipeline.as_ref(),
            ParticleBlendMode::AlphaBlend => self.alpha_blend_pipeline.as_ref(),
        };

        if let Some(pipeline) = pipeline {
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, self.render_particle_bind_group.as_ref().unwrap(), &[]);
            render_pass.set_bind_group(2, self.render_texture_bind_group.as_ref().unwrap(), &[]);

            // Draw using indirect buffer
            render_pass.draw_indirect(&gpu.indirect_buffer, 0);
        }
    }

    /// Update render parameters (call when resolution or camera changes).
    pub fn update_render_params(
        &mut self,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        near: f32,
        far: f32,
        camera_right: [f32; 3],
        camera_up: [f32; 3],
    ) {
        self.width = width;
        self.height = height;
        self.near = near;
        self.far = far;

        if let Some(ref buffer) = self.render_uniform_buffer {
            let uniform = ParticleRenderUniform {
                resolution: [
                    width as f32,
                    height as f32,
                    1.0 / width as f32,
                    1.0 / height as f32,
                ],
                params: [self.config.soft_fade_distance, 1.0, 1.0, 0.0],
                camera_params: [near, far, 0.0, 0.0],
                camera_right: [camera_right[0], camera_right[1], camera_right[2], 0.0],
                camera_up: [camera_up[0], camera_up[1], camera_up[2], 0.0],
            };
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
        }
    }

    /// Add a force field.
    pub fn add_force(&mut self, force: ForceType) {
        if self.forces.len() < MAX_FORCE_FIELDS {
            self.forces.push(force);
        }
    }

    /// Clear all force fields.
    pub fn clear_forces(&mut self) {
        self.forces.clear();
    }

    /// Set the emitter position.
    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.position = [x, y, z];
    }

    /// Get the emitter position.
    pub fn position(&self) -> [f32; 3] {
        self.position
    }

    /// Start playing the particle system.
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Pause the particle system.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop and reset the particle system.
    pub fn stop(&mut self, queue: &wgpu::Queue) {
        self.playing = false;
        self.elapsed_time = 0.0;
        self.emission_accumulator = 0.0;

        if let Some(ref gpu) = self.gpu {
            gpu.reset(queue);
        }
    }

    /// Emit a burst of particles.
    pub fn burst(&mut self, count: u32) {
        self.emission_accumulator += count as f32;
    }

    /// Check if the system is playing.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Get the blend mode.
    pub fn blend_mode(&self) -> ParticleBlendMode {
        self.config.blend_mode
    }
}
