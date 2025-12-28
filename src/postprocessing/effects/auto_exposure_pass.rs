//! Auto-Exposure (Eye Adaptation) post-processing effect.
//!
//! Implements automatic exposure adjustment based on scene luminance:
//! 1. Build luminance histogram from HDR scene
//! 2. Calculate target exposure from histogram (ignoring outliers)
//! 3. Temporally smooth exposure changes for natural adaptation
//!
//! This creates the effect of eyes adjusting when moving from dark to bright areas.

use wgpu::util::DeviceExt;

/// Auto-exposure settings.
#[derive(Debug, Clone)]
pub struct AutoExposureSettings {
    /// Minimum exposure value (prevents over-darkening in bright scenes).
    pub min_exposure: f32,
    /// Maximum exposure value (prevents over-brightening in dark scenes).
    pub max_exposure: f32,
    /// Speed of adaptation from dark to bright (higher = faster).
    pub speed_up: f32,
    /// Speed of adaptation from bright to dark (higher = faster).
    pub speed_down: f32,
    /// Low percentile to ignore (cuts dark outliers, 0.0-1.0).
    pub low_percent: f32,
    /// High percentile to ignore (cuts bright outliers, 0.0-1.0).
    pub high_percent: f32,
    /// Exposure compensation (manual adjustment on top of auto).
    pub exposure_compensation: f32,
    /// Target middle-gray value (default 0.18 = 18% gray).
    pub key_value: f32,
}

impl Default for AutoExposureSettings {
    fn default() -> Self {
        Self {
            min_exposure: -4.0,  // Very dark scenes
            max_exposure: 4.0,   // Very bright scenes
            speed_up: 3.0,       // Adapt quickly to brightness
            speed_down: 1.0,     // Adapt slowly to darkness (more realistic)
            low_percent: 0.1,    // Ignore darkest 10%
            high_percent: 0.9,   // Ignore brightest 10%
            exposure_compensation: 0.0,
            key_value: 0.18,
        }
    }
}

/// Histogram uniform data for compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HistogramUniform {
    /// Image dimensions (width, height, 1/width, 1/height).
    dimensions: [f32; 4],
    /// Luminance range (min_log_lum, inv_log_lum_range, 0, 0).
    lum_range: [f32; 4],
}

/// Adaptation uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdaptationUniform {
    /// low_percent, high_percent, delta_time, num_pixels
    params: [f32; 4],
    /// min_exposure, max_exposure, speed_up, speed_down
    adaptation: [f32; 4],
    /// key_value, exposure_compensation, 0, 0
    exposure: [f32; 4],
}

/// Auto-exposure post-processing pass.
pub struct AutoExposurePass {
    enabled: bool,
    settings: AutoExposureSettings,
    width: u32,
    height: u32,

    // Current exposure state
    current_exposure: f32,

    // GPU resources
    histogram_pipeline: Option<wgpu::ComputePipeline>,
    average_pipeline: Option<wgpu::ComputePipeline>,
    histogram_buffer: Option<wgpu::Buffer>,
    exposure_buffer: Option<wgpu::Buffer>,
    histogram_uniform_buffer: Option<wgpu::Buffer>,
    adaptation_uniform_buffer: Option<wgpu::Buffer>,
    histogram_bind_group_layout: Option<wgpu::BindGroupLayout>,
    average_bind_group_layout: Option<wgpu::BindGroupLayout>,
    sampler: Option<wgpu::Sampler>,
}

impl AutoExposurePass {
    /// Number of histogram bins (256 for good granularity).
    const HISTOGRAM_BINS: u32 = 256;

    /// Workgroup size for histogram compute.
    const WORKGROUP_SIZE: u32 = 16;

    /// Minimum log luminance to track.
    const MIN_LOG_LUMINANCE: f32 = -10.0;

    /// Maximum log luminance to track.
    const MAX_LOG_LUMINANCE: f32 = 2.0;

    /// Create a new auto-exposure pass.
    pub fn new() -> Self {
        Self {
            enabled: false,
            settings: AutoExposureSettings::default(),
            width: 0,
            height: 0,
            current_exposure: 0.0,
            histogram_pipeline: None,
            average_pipeline: None,
            histogram_buffer: None,
            exposure_buffer: None,
            histogram_uniform_buffer: None,
            adaptation_uniform_buffer: None,
            histogram_bind_group_layout: None,
            average_bind_group_layout: None,
            sampler: None,
        }
    }

    /// Check if auto-exposure is enabled.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable auto-exposure.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get current settings.
    pub fn settings(&self) -> &AutoExposureSettings {
        &self.settings
    }

    /// Get current calculated exposure value.
    pub fn current_exposure(&self) -> f32 {
        self.current_exposure + self.settings.exposure_compensation
    }

    /// Set minimum exposure.
    pub fn set_min_exposure(&mut self, value: f32) {
        self.settings.min_exposure = value;
    }

    /// Set maximum exposure.
    pub fn set_max_exposure(&mut self, value: f32) {
        self.settings.max_exposure = value;
    }

    /// Set adaptation speed (dark to bright).
    pub fn set_speed_up(&mut self, value: f32) {
        self.settings.speed_up = value.max(0.1);
    }

    /// Set adaptation speed (bright to dark).
    pub fn set_speed_down(&mut self, value: f32) {
        self.settings.speed_down = value.max(0.1);
    }

    /// Set exposure compensation.
    pub fn set_exposure_compensation(&mut self, value: f32) {
        self.settings.exposure_compensation = value;
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        // Create sampler
        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Auto Exposure Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Create histogram buffer (256 bins * 4 bytes)
        self.histogram_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Luminance Histogram Buffer"),
            size: (Self::HISTOGRAM_BINS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create exposure buffer (stores current and target exposure)
        // [current_exposure, target_exposure, average_luminance, padding]
        self.exposure_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Exposure Buffer"),
            contents: bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }));

        // Create uniform buffers
        self.histogram_uniform_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Histogram Uniform Buffer"),
            size: std::mem::size_of::<HistogramUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.adaptation_uniform_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adaptation Uniform Buffer"),
            size: std::mem::size_of::<AdaptationUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create bind group layouts
        self.histogram_bind_group_layout = Some(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Histogram Bind Group Layout"),
                entries: &[
                    // Input HDR texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
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
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Histogram buffer (read-write)
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
                    // Uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        self.average_bind_group_layout = Some(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Average Bind Group Layout"),
                entries: &[
                    // Histogram buffer (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Exposure buffer (read-write)
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
                    // Uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        self.create_pipelines(device);
    }

    fn create_pipelines(&mut self, device: &wgpu::Device) {
        // Histogram compute shader
        let histogram_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Histogram Shader"),
            source: wgpu::ShaderSource::Wgsl(HISTOGRAM_SHADER.into()),
        });

        let histogram_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Histogram Pipeline Layout"),
            bind_group_layouts: &[self.histogram_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        self.histogram_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Histogram Pipeline"),
                layout: Some(&histogram_layout),
                module: &histogram_shader,
                entry_point: Some("build_histogram"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        // Average/adaptation compute shader
        let average_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Average Shader"),
            source: wgpu::ShaderSource::Wgsl(AVERAGE_SHADER.into()),
        });

        let average_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Average Pipeline Layout"),
            bind_group_layouts: &[self.average_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        self.average_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Average Pipeline"),
                layout: Some(&average_layout),
                module: &average_shader,
                entry_point: Some("calculate_exposure"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
    }

    /// Resize the pass.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Update exposure from HDR scene.
    /// Returns the current exposure value after adaptation.
    pub fn update(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
        delta_time: f32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> f32 {
        if !self.enabled {
            return self.settings.exposure_compensation;
        }

        let Some(histogram_pipeline) = &self.histogram_pipeline else {
            return self.settings.exposure_compensation;
        };
        let Some(average_pipeline) = &self.average_pipeline else {
            return self.settings.exposure_compensation;
        };
        let Some(histogram_buffer) = &self.histogram_buffer else {
            return self.settings.exposure_compensation;
        };
        let Some(exposure_buffer) = &self.exposure_buffer else {
            return self.settings.exposure_compensation;
        };

        // Clear histogram buffer
        encoder.clear_buffer(histogram_buffer, 0, None);

        // Update histogram uniforms
        let log_lum_range = Self::MAX_LOG_LUMINANCE - Self::MIN_LOG_LUMINANCE;
        let histogram_uniform = HistogramUniform {
            dimensions: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            lum_range: [
                Self::MIN_LOG_LUMINANCE,
                1.0 / log_lum_range,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(
            self.histogram_uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[histogram_uniform]),
        );

        // Update adaptation uniforms
        let num_pixels = (self.width * self.height) as f32;
        let adaptation_uniform = AdaptationUniform {
            params: [
                self.settings.low_percent,
                self.settings.high_percent,
                delta_time,
                num_pixels,
            ],
            adaptation: [
                self.settings.min_exposure,
                self.settings.max_exposure,
                self.settings.speed_up,
                self.settings.speed_down,
            ],
            exposure: [
                self.settings.key_value,
                self.settings.exposure_compensation,
                Self::MIN_LOG_LUMINANCE,
                log_lum_range,
            ],
        };
        queue.write_buffer(
            self.adaptation_uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[adaptation_uniform]),
        );

        // Create bind groups
        let histogram_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Histogram Bind Group"),
            layout: self.histogram_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.sampler.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.histogram_uniform_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });

        let average_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Average Bind Group"),
            layout: self.average_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: exposure_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.adaptation_uniform_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });

        // Dispatch histogram pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Histogram Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(histogram_pipeline);
            pass.set_bind_group(0, &histogram_bind_group, &[]);

            let workgroups_x = (self.width + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
            let workgroups_y = (self.height + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Dispatch average/adaptation pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Exposure Adaptation Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(average_pipeline);
            pass.set_bind_group(0, &average_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Return current exposure (will be updated on GPU, but we use last frame's value)
        self.current_exposure + self.settings.exposure_compensation
    }

    /// Get the exposure buffer for direct GPU access (e.g., by tonemapping shader).
    pub fn exposure_buffer(&self) -> Option<&wgpu::Buffer> {
        self.exposure_buffer.as_ref()
    }
}

impl Default for AutoExposurePass {
    fn default() -> Self {
        Self::new()
    }
}

// Histogram compute shader
const HISTOGRAM_SHADER: &str = r#"
struct Uniforms {
    dimensions: vec4<f32>,  // width, height, 1/width, 1/height
    lum_range: vec4<f32>,   // min_log_lum, inv_log_lum_range, 0, 0
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(3) var<uniform> params: Uniforms;

// Luminance calculation (Rec. 709)
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16, 1)
fn build_histogram(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<i32>(global_id.xy);
    let dims = vec2<i32>(i32(params.dimensions.x), i32(params.dimensions.y));

    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    // Sample HDR color
    let uv = (vec2<f32>(pixel) + 0.5) * params.dimensions.zw;
    let color = textureSampleLevel(hdr_texture, tex_sampler, uv, 0.0).rgb;

    // Calculate luminance
    let lum = luminance(color);

    // Skip very dark pixels (avoid log(0))
    if (lum < 0.001) {
        return;
    }

    // Convert to log space and map to histogram bin
    let log_lum = log2(lum);
    let normalized = (log_lum - params.lum_range.x) * params.lum_range.y;
    let bin = u32(clamp(normalized * 255.0, 0.0, 255.0));

    // Atomic increment
    atomicAdd(&histogram[bin], 1u);
}
"#;

// Average/adaptation compute shader
const AVERAGE_SHADER: &str = r#"
struct AdaptationParams {
    params: vec4<f32>,      // low_percent, high_percent, delta_time, num_pixels
    adaptation: vec4<f32>,  // min_exposure, max_exposure, speed_up, speed_down
    exposure: vec4<f32>,    // key_value, exposure_compensation, min_log_lum, log_lum_range
}

@group(0) @binding(0) var<storage, read> histogram: array<u32, 256>;
@group(0) @binding(1) var<storage, read_write> exposure_data: array<f32, 4>;
@group(0) @binding(2) var<uniform> params: AdaptationParams;

@compute @workgroup_size(1, 1, 1)
fn calculate_exposure() {
    let low_percent = params.params.x;
    let high_percent = params.params.y;
    let delta_time = params.params.z;
    let num_pixels = params.params.w;

    let min_exposure = params.adaptation.x;
    let max_exposure = params.adaptation.y;
    let speed_up = params.adaptation.z;
    let speed_down = params.adaptation.w;

    let key_value = params.exposure.x;
    let min_log_lum = params.exposure.z;
    let log_lum_range = params.exposure.w;

    // Count total pixels in histogram
    var total_count = 0u;
    for (var i = 0u; i < 256u; i++) {
        total_count += histogram[i];
    }

    if (total_count == 0u) {
        return;
    }

    // Find low and high cutoff counts
    let low_count = u32(f32(total_count) * low_percent);
    let high_count = u32(f32(total_count) * high_percent);

    // Accumulate weighted average, ignoring outliers
    var running_count = 0u;
    var weighted_sum = 0.0;
    var weight_total = 0.0;

    for (var i = 0u; i < 256u; i++) {
        let bin_count = histogram[i];
        let prev_count = running_count;
        running_count += bin_count;

        // Skip if entirely in low outlier region
        if (running_count <= low_count) {
            continue;
        }

        // Stop if we've passed high outlier region
        if (prev_count >= high_count) {
            break;
        }

        // Calculate how much of this bin to include
        var include_count = bin_count;

        // Clip low end
        if (prev_count < low_count) {
            include_count -= (low_count - prev_count);
        }

        // Clip high end
        if (running_count > high_count) {
            include_count -= (running_count - high_count);
        }

        // Convert bin index to log luminance
        let bin_log_lum = min_log_lum + (f32(i) / 255.0) * log_lum_range;

        weighted_sum += bin_log_lum * f32(include_count);
        weight_total += f32(include_count);
    }

    // Calculate average log luminance
    var avg_log_lum = 0.0;
    if (weight_total > 0.0) {
        avg_log_lum = weighted_sum / weight_total;
    }

    // Convert to linear luminance
    let avg_luminance = exp2(avg_log_lum);

    // Calculate target exposure using the key value
    // exposure = log2(key_value / avg_luminance)
    let target_exposure = log2(key_value / max(avg_luminance, 0.001));
    let clamped_target = clamp(target_exposure, min_exposure, max_exposure);

    // Get current exposure
    let current_exposure = exposure_data[0];

    // Adaptive speed based on direction
    let speed = select(speed_down, speed_up, clamped_target > current_exposure);

    // Exponential smoothing
    let adaptation_factor = 1.0 - exp(-delta_time * speed);
    let new_exposure = mix(current_exposure, clamped_target, adaptation_factor);

    // Store results
    exposure_data[0] = new_exposure;           // Current exposure (smoothed)
    exposure_data[1] = clamped_target;         // Target exposure
    exposure_data[2] = avg_luminance;          // Average luminance
    exposure_data[3] = 0.0;                    // Padding
}
"#;
