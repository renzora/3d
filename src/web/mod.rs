//! Web bindings for the Ren engine.
//!
//! This module provides JavaScript-friendly APIs via wasm-bindgen.

use wasm_bindgen::prelude::*;
use web_sys::{window, HtmlCanvasElement};
use std::cell::RefCell;
use wgpu::util::DeviceExt;
use bytemuck;

use crate::core::Clock;
use crate::math::{Color, Matrix3, Matrix4, Vector3};
use crate::material::{TexturedPbrMaterial, TexturedPbrMaterialUniform, LineMaterial, LineModelUniform, PbrCameraUniform, PbrModelUniform};
use crate::texture::{Texture2D, Sampler};
use crate::camera::PerspectiveCamera;
use crate::controls::OrbitControls;
use crate::helpers::{AxesHelper, GridHelper};
use crate::loaders::{GltfLoader, ObjLoader, LoadedScene};
use crate::postprocessing::{TonemappingPass, TonemappingMode, BloomPass, BloomSettings, FxaaPass, FxaaQuality, SmaaPass, SmaaQuality, SsaoPass, SsaoQuality, SsrPass, SsrQuality, TaaPass, VignettePass, DofPass, DofQuality, MotionBlurPass, MotionBlurQuality, OutlinePass, OutlineMode, ColorCorrectionPass, Pass};
use crate::shadows::{PCFMode, ShadowConfig};
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

/// Shadow uniform data for the shader.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct WebShadowUniform {
    /// Light-space view-projection matrix.
    pub light_view_proj: [[f32; 4]; 4],
    /// Shadow params: x=bias, y=normal_bias, z=enabled, w=pcf_mode
    pub shadow_params: [f32; 4],
    /// Light direction (xyz) + padding
    pub light_direction: [f32; 4],
    /// Shadow map size: x=width, y=height, z=1/width, w=1/height
    pub shadow_map_size: [f32; 4],
}

impl Default for WebShadowUniform {
    fn default() -> Self {
        Self {
            light_view_proj: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            shadow_params: [0.005, 0.02, 0.0, 2.0], // bias, normal_bias, disabled, PCF 3x3
            light_direction: [-0.5, -1.0, -0.3, 0.0],
            shadow_map_size: [1024.0, 1024.0, 1.0 / 1024.0, 1.0 / 1024.0],
        }
    }
}

/// Anti-aliasing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AaMode {
    /// No anti-aliasing.
    None,
    /// FXAA (Fast Approximate Anti-Aliasing).
    #[default]
    Fxaa,
    /// SMAA (Subpixel Morphological Anti-Aliasing).
    Smaa,
    /// TAA (Temporal Anti-Aliasing).
    Taa,
}

/// Render statistics exposed to JavaScript.
#[wasm_bindgen]
pub struct RenderStats {
    /// Number of draw calls.
    pub draw_calls: u32,
    /// Number of triangles rendered.
    pub triangles: u32,
    /// Current frame number.
    pub frame: u64,
}

/// A renderable mesh with geometry buffers and transform.
struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    model_buffer: wgpu::Buffer,
    model_bind_group: wgpu::BindGroup,
    material_buffer: wgpu::Buffer,
    material_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    position: Vector3,
    rotation: Vector3,
    scale: Vector3,
}

/// Line renderable object.
struct LineObject {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    model_buffer: wgpu::Buffer,
    model_bind_group: wgpu::BindGroup,
}

/// The main Ren application for web environments.
#[wasm_bindgen]
pub struct RenApp {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    clear_color: Color,
    clock: RefCell<Clock>,
    frame_count: RefCell<u64>,
    width: u32,
    height: u32,
    // Rendering
    material: TexturedPbrMaterial,
    line_material: LineMaterial,
    camera: RefCell<PerspectiveCamera>,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    line_camera_bind_group: wgpu::BindGroup,
    // Controls
    controls: RefCell<OrbitControls>,
    // Meshes
    meshes: Vec<Mesh>,
    // Line objects (grid, axes)
    line_objects: Vec<LineObject>,
    // Visibility flags for helpers
    show_grid: bool,
    show_axes: bool,
    // Textures
    default_sampler: Sampler,
    white_texture: Texture2D,
    normal_texture: Texture2D,
    // Post-processing
    scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    bloom_input_texture: wgpu::Texture,
    bloom_input_view: wgpu::TextureView,
    tonemapped_texture: wgpu::Texture,
    tonemapped_view: wgpu::TextureView,
    vignette_texture: wgpu::Texture,
    vignette_view: wgpu::TextureView,
    bloom: BloomPass,
    tonemapping: TonemappingPass,
    fxaa: FxaaPass,
    smaa: SmaaPass,
    taa: TaaPass,
    ssao: SsaoPass,
    ssr: SsrPass,
    vignette: VignettePass,
    dof: DofPass,
    motion_blur: MotionBlurPass,
    outline: OutlinePass,
    color_correction: ColorCorrectionPass,
    aa_mode: AaMode,
    surface_format: wgpu::TextureFormat,
    // Hemisphere light state
    hemisphere_enabled: bool,
    hemisphere_sky_color: [f32; 3],
    hemisphere_ground_color: [f32; 3],
    hemisphere_intensity: f32,
    // Shadow system
    shadow_config: ShadowConfig,
    shadows_enabled: bool,
    shadow_map_texture: wgpu::Texture,
    shadow_map_view: wgpu::TextureView,
    shadow_sampler: wgpu::Sampler,
    shadow_uniform_buffer: wgpu::Buffer,
    shadow_depth_pipeline: wgpu::RenderPipeline,
    shadow_model_bind_group_layout: wgpu::BindGroupLayout,
    light_direction: [f32; 3],
}

#[wasm_bindgen]
impl RenApp {
    /// Create a new Ren application attached to a canvas element.
    #[wasm_bindgen]
    pub async fn new(canvas_id: &str) -> Result<RenApp, JsValue> {
        let window = window().ok_or_else(|| JsValue::from_str("No window object"))?;
        let document = window.document().ok_or_else(|| JsValue::from_str("No document"))?;

        let canvas: HtmlCanvasElement = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| JsValue::from_str(&format!("Canvas '{}' not found", canvas_id)))?
            .dyn_into()
            .map_err(|_| JsValue::from_str("Element is not a canvas"))?;

        let dpr = window.device_pixel_ratio();
        let width = (canvas.client_width() as f64 * dpr) as u32;
        let height = (canvas.client_height() as f64 * dpr) as u32;

        canvas.set_width(width);
        canvas.set_height(height);

        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {:?}", e)))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| JsValue::from_str("No suitable GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Ren Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to get device: {:?}", e)))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create material and pipeline
        let mut material = TexturedPbrMaterial::new();
        material.build_pipeline(&device, &queue, surface_format, depth_format);

        // Create line material and pipeline
        let mut line_material = LineMaterial::new();
        line_material.build_pipeline(&device, surface_format, depth_format);

        // Create default textures for meshes without textures
        let white_texture = Texture2D::white(&device, &queue);
        let normal_texture = Texture2D::default_normal(&device, &queue);
        let default_sampler = Sampler::linear(&device);

        // Create camera
        let aspect = width as f32 / height as f32;
        let mut camera = PerspectiveCamera::new(60.0, aspect, 0.1, 100.0);
        let camera_pos = Vector3::new(0.0, 3.0, 6.0);
        camera.set_position(camera_pos);
        camera.set_target(Vector3::ZERO);

        let camera_uniform = PbrCameraUniform {
            view_proj: matrix4_to_array(camera.view_projection_matrix()),
            position: [camera_pos.x, camera_pos.y, camera_pos.z],
            _padding: 0.0,
            hemisphere_sky: [0.6, 0.75, 1.0, 0.0],  // Disabled by default
            hemisphere_ground: [0.4, 0.3, 0.2, 1.0],
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = material
            .create_camera_bind_group(&device, &camera_buffer)
            .ok_or_else(|| JsValue::from_str("Failed to create camera bind group"))?;

        let line_camera_bind_group = line_material
            .create_camera_bind_group(&device, &camera_buffer)
            .ok_or_else(|| JsValue::from_str("Failed to create line camera bind group"))?;

        // Empty mesh list - models loaded via load_gltf/load_obj
        let meshes = Vec::new();

        let clock = Clock::new();

        // Create orbit controls
        let controls = OrbitControls::with_target(Vector3::ZERO);

        // Create line objects (grid and axes)
        let mut line_objects = Vec::new();

        // Grid helper - 10x10, gray
        let grid = GridHelper::simple(10.0, 10);
        let grid_vertices = grid.line().vertices();
        let grid_vertex_data: Vec<f32> = grid_vertices.iter()
            .flat_map(|v| [v.position[0], v.position[1], v.position[2], v.color[0], v.color[1], v.color[2], v.color[3]])
            .collect();
        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&grid_vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let grid_model_uniform = LineModelUniform::default();
        let grid_model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Model Buffer"),
            contents: bytemuck::cast_slice(&[grid_model_uniform]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let grid_model_bind_group = line_material
            .create_model_bind_group(&device, &grid_model_buffer)
            .ok_or_else(|| JsValue::from_str("Failed to create grid model bind group"))?;
        line_objects.push(LineObject {
            vertex_buffer: grid_vertex_buffer,
            vertex_count: grid_vertices.len() as u32,
            model_buffer: grid_model_buffer,
            model_bind_group: grid_model_bind_group,
        });

        // Axes helper - size 2
        let axes = AxesHelper::new(2.0);
        let axes_vertices = axes.line().vertices();
        let axes_vertex_data: Vec<f32> = axes_vertices.iter()
            .flat_map(|v| [v.position[0], v.position[1], v.position[2], v.color[0], v.color[1], v.color[2], v.color[3]])
            .collect();
        let axes_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Axes Vertex Buffer"),
            contents: bytemuck::cast_slice(&axes_vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let axes_model_uniform = LineModelUniform::default();
        let axes_model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Axes Model Buffer"),
            contents: bytemuck::cast_slice(&[axes_model_uniform]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let axes_model_bind_group = line_material
            .create_model_bind_group(&device, &axes_model_buffer)
            .ok_or_else(|| JsValue::from_str("Failed to create axes model bind group"))?;
        line_objects.push(LineObject {
            vertex_buffer: axes_vertex_buffer,
            vertex_count: axes_vertices.len() as u32,
            model_buffer: axes_model_buffer,
            model_bind_group: axes_model_bind_group,
        });

        // Create scene render target for post-processing
        let scene_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let scene_view = scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bloom input texture (copy of scene for bloom to read from, also used by DoF)
        let bloom_input_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Input Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let bloom_input_view = bloom_input_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create tonemapped texture (for FXAA to read from)
        let tonemapped_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tonemapped Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tonemapped_view = tonemapped_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create vignette texture (between tonemapping and AA)
        let vignette_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Vignette Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let vignette_view = vignette_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bloom pass
        let mut bloom = BloomPass::new();
        bloom.set_settings(BloomSettings {
            intensity: 0.8,
            threshold: 0.6,
            soft_threshold: 0.5,
            blur_iterations: 4,
        });
        bloom.init(&device, surface_format, width, height);

        // Create tonemapping pass
        let mut tonemapping = TonemappingPass::new();
        tonemapping.set_exposure(1.5);
        tonemapping.set_mode(TonemappingMode::Aces);
        tonemapping.init(&device, surface_format);

        // Create FXAA pass
        let mut fxaa = FxaaPass::new();
        fxaa.set_quality(FxaaQuality::High);
        fxaa.init(&device, surface_format, width, height);

        // Create SMAA pass
        let mut smaa = SmaaPass::new();
        smaa.set_quality(SmaaQuality::High);
        smaa.init(&device, surface_format, width, height);

        // Create TAA pass
        let mut taa = TaaPass::new();
        taa.init(&device, surface_format, width, height);

        // Create SSAO pass
        let mut ssao = SsaoPass::new();
        ssao.set_quality(SsaoQuality::Medium);
        ssao.init(&device, &queue, surface_format, width, height);

        // Create SSR pass
        let mut ssr = SsrPass::new();
        ssr.set_enabled(false); // Disabled by default
        ssr.init(&device, surface_format, width, height);

        // Create vignette pass
        let mut vignette = VignettePass::new();
        vignette.init(&device, surface_format, width, height);

        // Create DoF pass
        let mut dof = DofPass::new();
        dof.set_enabled(false); // Disabled by default
        dof.init(&device, surface_format, width, height);

        // Create motion blur pass
        let motion_blur = MotionBlurPass::new(&device, width, height, surface_format);

        // Create outline pass
        let outline = OutlinePass::new(&device, width, height, surface_format);

        // Create color correction pass
        let color_correction = ColorCorrectionPass::new(&device, surface_format);

        // ========== Shadow System Setup ==========
        let shadow_resolution = 1024u32;
        let shadow_config = ShadowConfig::default();

        // Create shadow map depth texture
        let shadow_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map Texture"),
            size: wgpu::Extent3d {
                width: shadow_resolution,
                height: shadow_resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_view = shadow_map_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create comparison sampler for shadow mapping
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Create shadow uniform buffer
        let shadow_uniform = WebShadowUniform::default();
        let shadow_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shadow Uniform Buffer"),
            contents: bytemuck::cast_slice(&[shadow_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create depth-only pipeline for shadow pass
        let shadow_shader_source = r#"
            struct LightUniform {
                view_proj: mat4x4<f32>,
            }
            struct ModelUniform {
                model: mat4x4<f32>,
                normal: mat3x3<f32>,
            }

            @group(0) @binding(0) var<uniform> light: LightUniform;
            @group(1) @binding(0) var<uniform> model: ModelUniform;

            @vertex
            fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
                return light.view_proj * model.model * vec4<f32>(position, 1.0);
            }

            @fragment
            fn fs_main() {}
        "#;

        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(shadow_shader_source.into()),
        });

        // Light uniform bind group layout (for shadow pass)
        let shadow_light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Light Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Model bind group layout for shadow pass
        let shadow_model_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Model Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&shadow_light_bind_group_layout, &shadow_model_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout matching the existing Vertex struct
        let shadow_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 32, // 3 floats pos + 3 floats normal + 2 floats uv = 8 floats = 32 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        };

        let shadow_depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Depth Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_main"),
                buffers: &[shadow_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shadow_shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create light uniform buffer and bind group for shadow pass
        let light_uniform_data: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let _shadow_light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shadow Light Buffer"),
            contents: bytemuck::cast_slice(&light_uniform_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Ok(RenApp {
            device,
            queue,
            surface,
            surface_config,
            depth_texture,
            depth_view,
            clear_color: Color::new(0.05, 0.05, 0.08),
            clock: RefCell::new(clock),
            frame_count: RefCell::new(0),
            width,
            height,
            material,
            line_material,
            camera: RefCell::new(camera),
            camera_buffer,
            camera_bind_group,
            line_camera_bind_group,
            controls: RefCell::new(controls),
            meshes,
            line_objects,
            show_grid: true,
            show_axes: true,
            default_sampler,
            white_texture,
            normal_texture,
            scene_texture,
            scene_view,
            bloom_input_texture,
            bloom_input_view,
            tonemapped_texture,
            tonemapped_view,
            vignette_texture,
            vignette_view,
            bloom,
            tonemapping,
            fxaa,
            smaa,
            taa,
            ssao,
            ssr,
            vignette,
            dof,
            motion_blur,
            outline,
            color_correction,
            aa_mode: AaMode::Fxaa,
            surface_format,
            // Hemisphere light defaults
            hemisphere_enabled: false,
            hemisphere_sky_color: [0.6, 0.75, 1.0],
            hemisphere_ground_color: [0.4, 0.3, 0.2],
            hemisphere_intensity: 1.0,
            // Shadow system
            shadow_config,
            shadows_enabled: false,
            shadow_map_texture,
            shadow_map_view,
            shadow_sampler,
            shadow_uniform_buffer,
            shadow_depth_pipeline,
            shadow_model_bind_group_layout,
            light_direction: [-0.5, -1.0, -0.3],
        })
    }

    fn create_mesh_with_transform(
        device: &wgpu::Device,
        material: &TexturedPbrMaterial,
        vertex_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        index_count: u32,
        world_transform: &Matrix4,
        mat_uniform: TexturedPbrMaterialUniform,
        texture_bind_group: wgpu::BindGroup,
    ) -> Result<Mesh, JsValue> {
        // Compute normal matrix from world transform
        let normal_matrix = Matrix3::from_matrix4_normal(world_transform);

        let model_uniform = PbrModelUniform {
            model: matrix4_to_array(world_transform),
            normal: matrix3_to_padded_array(&normal_matrix),
        };

        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Model Buffer"),
            contents: bytemuck::cast_slice(&[model_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let model_bind_group = material
            .create_model_bind_group(device, &model_buffer)
            .ok_or_else(|| JsValue::from_str("Failed to create model bind group"))?;

        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Buffer"),
            contents: bytemuck::cast_slice(&[mat_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let material_bind_group = material
            .create_material_bind_group(device, &material_buffer)
            .ok_or_else(|| JsValue::from_str("Failed to create material bind group"))?;

        Ok(Mesh {
            vertex_buffer,
            index_buffer,
            index_count,
            model_buffer,
            model_bind_group,
            material_buffer,
            material_bind_group,
            texture_bind_group,
            position: Vector3::ZERO,
            rotation: Vector3::ZERO,
            scale: Vector3::ONE,
        })
    }

    /// Render a single frame.
    #[wasm_bindgen]
    pub fn frame(&mut self) -> Result<(), JsValue> {
        let mut clock = self.clock.borrow_mut();
        let _delta = clock.get_delta() as f32;
        let _elapsed = clock.get_elapsed_time() as f32;
        drop(clock);

        // Update controls and camera
        {
            let mut controls = self.controls.borrow_mut();
            let mut camera = self.camera.borrow_mut();
            controls.update(&mut camera);

            let pos = camera.position;
            let mut view_proj = matrix4_to_array(camera.view_projection_matrix());

            // Apply TAA jitter to projection matrix (industry standard technique)
            // Jitter is applied to clip space coordinates via projection matrix elements [2][0] and [2][1]
            if self.aa_mode == AaMode::Taa && self.taa.enabled() {
                let (jitter_x, jitter_y) = self.taa.get_jitter_offset();
                // Apply jitter in clip space (NDC range is -1 to 1)
                view_proj[2][0] += jitter_x;
                view_proj[2][1] += jitter_y;
            }

            let camera_uniform = PbrCameraUniform {
                view_proj,
                position: [pos.x, pos.y, pos.z],
                _padding: 0.0,
                hemisphere_sky: [
                    self.hemisphere_sky_color[0],
                    self.hemisphere_sky_color[1],
                    self.hemisphere_sky_color[2],
                    if self.hemisphere_enabled { 1.0 } else { 0.0 },
                ],
                hemisphere_ground: [
                    self.hemisphere_ground_color[0],
                    self.hemisphere_ground_color[1],
                    self.hemisphere_ground_color[2],
                    self.hemisphere_intensity,
                ],
            };
            self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

            // Update SSAO projection matrices
            if self.ssao.enabled() {
                let proj = matrix4_to_array(camera.projection_matrix());
                let inv_proj = matrix4_to_array(&camera.projection_matrix().inverse());
                self.ssao.set_projection(proj, inv_proj, camera.near, camera.far);
            }

            // Update DoF camera planes
            if self.dof.enabled() {
                self.dof.set_camera_planes(camera.near, camera.far);
            }

            // Update SSR projection matrices
            if self.ssr.enabled() {
                let proj = matrix4_to_array(camera.projection_matrix());
                let inv_proj = matrix4_to_array(&camera.projection_matrix().inverse());
                let view = matrix4_to_array(camera.view_matrix());
                self.ssr.set_projection(proj, inv_proj, camera.near, camera.far);
                self.ssr.set_view(view);
            }

            // Update outline camera planes
            if self.outline.enabled() {
                self.outline.set_camera_planes(camera.near, camera.far);
            }
        }

        // Get view-projection matrices for motion blur (outside borrow scope)
        let (view_proj_for_mb, inv_view_proj_for_mb) = {
            let mut camera = self.camera.borrow_mut();
            let vp = camera.view_projection_matrix();
            (matrix4_to_array(&vp), matrix4_to_array(&vp.inverse()))
        };

        // Get surface texture
        let output = self.surface.get_current_texture()
            .map_err(|e| JsValue::from_str(&format!("Surface error: {:?}", e)))?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        let clear_color = wgpu::Color {
            r: self.clear_color.r as f64,
            g: self.clear_color.g as f64,
            b: self.clear_color.b as f64,
            a: 1.0,
        };

        // ========== Shadow Pass ==========
        if self.shadows_enabled {
            // Calculate light-space matrix for directional light
            let light_dir = Vector3::new(self.light_direction[0], self.light_direction[1], self.light_direction[2]).normalized();

            // Create orthographic projection centered on scene
            let light_pos = Vector3::new(-light_dir.x * 15.0, -light_dir.y * 15.0, -light_dir.z * 15.0);
            let light_target = Vector3::ZERO;
            let light_up = if light_dir.y.abs() > 0.99 { Vector3::new(0.0, 0.0, 1.0) } else { Vector3::new(0.0, 1.0, 0.0) };

            let light_view = Matrix4::look_at(&light_pos, &light_target, &light_up);
            let ortho_size = 20.0;
            let light_proj = Matrix4::orthographic(-ortho_size, ortho_size, -ortho_size, ortho_size, 0.1, 50.0);
            let light_view_proj = light_proj.multiply(&light_view);

            // Update shadow uniform
            let resolution = self.shadow_config.resolution as f32;
            let shadow_uniform = WebShadowUniform {
                light_view_proj: matrix4_to_array(&light_view_proj),
                shadow_params: [
                    self.shadow_config.bias,
                    self.shadow_config.normal_bias,
                    1.0, // enabled
                    match self.shadow_config.pcf_mode {
                        PCFMode::None => 0.0,
                        PCFMode::Hardware2x2 => 1.0,
                        PCFMode::Soft3x3 => 2.0,
                        PCFMode::Soft5x5 => 3.0,
                        PCFMode::PoissonDisk => 4.0,
                    },
                ],
                light_direction: [light_dir.x, light_dir.y, light_dir.z, 0.0],
                shadow_map_size: [resolution, resolution, 1.0 / resolution, 1.0 / resolution],
            };
            self.queue.write_buffer(&self.shadow_uniform_buffer, 0, bytemuck::cast_slice(&[shadow_uniform]));

            // Create light bind group for shadow pass
            let light_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shadow Light Buffer"),
                contents: bytemuck::cast_slice(&matrix4_to_array(&light_view_proj)),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let shadow_light_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow Light BG Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

            let light_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shadow Light Bind Group"),
                layout: &shadow_light_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                }],
            });

            // Shadow render pass
            {
                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Shadow Render Pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_map_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                shadow_pass.set_pipeline(&self.shadow_depth_pipeline);
                shadow_pass.set_bind_group(0, &light_bind_group, &[]);

                // Render all meshes to shadow map
                for mesh in &self.meshes {
                    // Create model bind group for shadow pass
                    let shadow_model_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Shadow Model Bind Group"),
                        layout: &self.shadow_model_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: mesh.model_buffer.as_entire_binding(),
                        }],
                    });

                    shadow_pass.set_bind_group(1, &shadow_model_bind_group, &[]);
                    shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    shadow_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        } else {
            // Update shadow uniform with shadows disabled
            let shadow_uniform = WebShadowUniform {
                shadow_params: [0.005, 0.02, 0.0, 2.0], // disabled
                ..Default::default()
            };
            self.queue.write_buffer(&self.shadow_uniform_buffer, 0, bytemuck::cast_slice(&[shadow_uniform]));
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,  // Render to intermediate texture for post-processing
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Render PBR meshes
            if let Some(pipeline) = self.material.pipeline() {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

                for mesh in &self.meshes {
                    render_pass.set_bind_group(1, &mesh.model_bind_group, &[]);
                    render_pass.set_bind_group(2, &mesh.material_bind_group, &[]);
                    render_pass.set_bind_group(3, &mesh.texture_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }

            // Render line objects (grid at index 0, axes at index 1)
            if let Some(pipeline) = self.line_material.pipeline() {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &self.line_camera_bind_group, &[]);

                // Grid (index 0)
                if self.show_grid {
                    if let Some(line_obj) = self.line_objects.get(0) {
                        render_pass.set_bind_group(1, &line_obj.model_bind_group, &[]);
                        render_pass.set_vertex_buffer(0, line_obj.vertex_buffer.slice(..));
                        render_pass.draw(0..line_obj.vertex_count, 0..1);
                    }
                }

                // Axes (index 1)
                if self.show_axes {
                    if let Some(line_obj) = self.line_objects.get(1) {
                        render_pass.set_bind_group(1, &line_obj.model_bind_group, &[]);
                        render_pass.set_vertex_buffer(0, line_obj.vertex_buffer.slice(..));
                        render_pass.draw(0..line_obj.vertex_count, 0..1);
                    }
                }
            }
        }

        // Copy scene to bloom input texture (so we can read from it while writing to scene)
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.scene_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.bloom_input_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        // Apply SSAO (reads depth + bloom_input, outputs to scene_view with AO applied)
        self.ssao.render_with_depth(&mut encoder, &self.depth_view, &self.bloom_input_view, &self.scene_view, &self.device, &self.queue);

        // Apply bloom post-processing (reads from bloom_input, additively blends to scene_view)
        self.bloom.render(&mut encoder, &self.bloom_input_view, &self.scene_view, &self.device, &self.queue);

        // Apply SSR if enabled (scene + depth -> bloom_input)
        let post_ssr_input = if self.ssr.enabled() {
            self.ssr.render_with_depth(&mut encoder, &self.depth_view, &self.scene_view, &self.bloom_input_view, &self.device, &self.queue);
            &self.bloom_input_view
        } else {
            &self.scene_view
        };

        // Apply motion blur if enabled (post_ssr + depth -> scene_view or bloom_input)
        let post_motion_blur_input = if self.motion_blur.enabled() {
            // Render to the texture that wasn't used as input
            let output_view = if self.ssr.enabled() {
                &self.scene_view  // SSR used bloom_input, so output to scene
            } else {
                &self.bloom_input_view  // No SSR, input is scene, output to bloom_input
            };
            self.motion_blur.render(
                &mut encoder,
                post_ssr_input,
                &self.depth_view,
                output_view,
                view_proj_for_mb,
                inv_view_proj_for_mb,
                &self.device,
                &self.queue,
            );
            output_view
        } else {
            post_ssr_input
        };

        // Apply outline if enabled
        let post_outline_input = if self.outline.enabled() {
            // Alternate between textures to avoid read/write conflict
            let output_view = if post_motion_blur_input as *const _ == &self.scene_view as *const _ {
                &self.bloom_input_view
            } else {
                &self.scene_view
            };
            self.outline.render(
                &mut encoder,
                post_motion_blur_input,
                &self.depth_view,
                output_view,
                &self.device,
                &self.queue,
            );
            output_view
        } else {
            post_motion_blur_input
        };

        // Apply DoF if enabled (post_outline -> vignette texture as intermediate)
        let tonemapping_input = if self.dof.enabled() {
            self.dof.render_with_depth(&mut encoder, &self.depth_view, post_outline_input, &self.vignette_view, &self.device, &self.queue);
            &self.vignette_view
        } else {
            post_outline_input
        };

        // Apply tonemapping post-processing
        self.tonemapping.render(&mut encoder, tonemapping_input, &self.tonemapped_view, &self.device, &self.queue);

        // Apply color correction if enabled (tonemapped -> vignette_view as intermediate)
        let post_color_correction = if self.color_correction.enabled() {
            self.color_correction.render(&mut encoder, &self.tonemapped_view, &self.vignette_view, &self.device, &self.queue);
            &self.vignette_view
        } else {
            &self.tonemapped_view
        };

        // Apply vignette if enabled
        let aa_input = if self.vignette.enabled() {
            // Output to the texture not currently being used as input
            let output = if post_color_correction as *const _ == &self.vignette_view as *const _ {
                &self.tonemapped_view
            } else {
                &self.vignette_view
            };
            self.vignette.render(&mut encoder, post_color_correction, output, &self.device, &self.queue);
            output
        } else {
            post_color_correction
        };

        // Apply anti-aliasing based on selected mode
        match self.aa_mode {
            AaMode::None => {
                // No AA - just blit (FXAA handles blit when disabled)
                self.fxaa.render(&mut encoder, aa_input, &view, &self.device, &self.queue);
            }
            AaMode::Fxaa => {
                self.fxaa.render(&mut encoder, aa_input, &view, &self.device, &self.queue);
            }
            AaMode::Smaa => {
                self.smaa.render(&mut encoder, aa_input, &view, &self.device, &self.queue);
            }
            AaMode::Taa => {
                self.taa.render(&mut encoder, aa_input, &view, &self.device, &self.queue);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Advance TAA frame counter
        if self.aa_mode == AaMode::Taa {
            self.taa.next_frame();
        }

        *self.frame_count.borrow_mut() += 1;

        Ok(())
    }

    /// Handle window resize.
    #[wasm_bindgen]
    pub fn resize(&mut self, width: u32, height: u32) {
        let window = window().unwrap();
        let dpr = window.device_pixel_ratio();
        let width = (width as f64 * dpr) as u32;
        let height = (height as f64 * dpr) as u32;

        if width > 0 && height > 0 {
            self.width = width;
            self.height = height;
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);

            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate scene texture for post-processing
            self.scene_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Scene Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.scene_view = self.scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate bloom input texture
            self.bloom_input_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Bloom Input Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.bloom_input_view = self.bloom_input_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate tonemapped texture
            self.tonemapped_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Tonemapped Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.tonemapped_view = self.tonemapped_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate vignette texture
            self.vignette_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Vignette Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.vignette_view = self.vignette_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Resize bloom textures
            self.bloom.resize(width, height, &self.device);

            // Resize AA passes
            self.fxaa.resize(width, height, &self.device);
            self.smaa.resize(width, height, &self.device);
            self.taa.resize(width, height, &self.device);
            self.ssao.resize(width, height, &self.device);
            self.ssr.resize(width, height, &self.device);
            self.vignette.resize(width, height, &self.device);
            self.dof.resize(width, height, &self.device);
            self.motion_blur.resize(width, height);
            self.outline.resize(width, height);

            // Update camera aspect ratio
            let mut camera = self.camera.borrow_mut();
            camera.set_aspect(width as f32 / height as f32);
            let pos = camera.position;
            let camera_uniform = PbrCameraUniform {
                view_proj: matrix4_to_array(camera.view_projection_matrix()),
                position: [pos.x, pos.y, pos.z],
                _padding: 0.0,
                hemisphere_sky: [
                    self.hemisphere_sky_color[0],
                    self.hemisphere_sky_color[1],
                    self.hemisphere_sky_color[2],
                    if self.hemisphere_enabled { 1.0 } else { 0.0 },
                ],
                hemisphere_ground: [
                    self.hemisphere_ground_color[0],
                    self.hemisphere_ground_color[1],
                    self.hemisphere_ground_color[2],
                    self.hemisphere_intensity,
                ],
            };
            self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
        }
    }

    /// Set the clear color.
    #[wasm_bindgen]
    pub fn set_clear_color(&mut self, r: f32, g: f32, b: f32) {
        self.clear_color = Color::new(r, g, b);
    }

    /// Get the current frame number.
    #[wasm_bindgen]
    pub fn frame_number(&self) -> u64 {
        *self.frame_count.borrow()
    }

    /// Get render statistics.
    #[wasm_bindgen]
    pub fn stats(&self) -> RenderStats {
        RenderStats {
            draw_calls: self.meshes.len() as u32,
            triangles: self.meshes.iter().map(|m| m.index_count / 3).sum(),
            frame: *self.frame_count.borrow(),
        }
    }

    /// Handle mouse drag for rotation (left button).
    #[wasm_bindgen]
    pub fn on_mouse_drag(&self, delta_x: f32, delta_y: f32) {
        let mut controls = self.controls.borrow_mut();
        controls.rotate_by_pixels(delta_x, delta_y, self.height as f32);
    }

    /// Handle mouse drag for panning (right button or shift+left).
    #[wasm_bindgen]
    pub fn on_mouse_pan(&self, delta_x: f32, delta_y: f32) {
        let mut controls = self.controls.borrow_mut();
        let camera = self.camera.borrow();
        controls.pan(delta_x, delta_y, &camera);
    }

    /// Handle mouse wheel for zoom.
    #[wasm_bindgen]
    pub fn on_mouse_wheel(&self, delta: f32) {
        let mut controls = self.controls.borrow_mut();
        controls.zoom_by_wheel(delta);
    }

    /// Load a GLTF/GLB model from bytes.
    #[wasm_bindgen]
    pub fn load_gltf(&mut self, data: &[u8]) -> Result<u32, JsValue> {
        let loader = GltfLoader::new();
        let scene = loader.load_from_bytes(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to load GLTF: {}", e)))?;

        self.add_loaded_scene(&scene)
    }

    /// Load a Wavefront OBJ model from string.
    #[wasm_bindgen]
    pub fn load_obj(&mut self, content: &str) -> Result<u32, JsValue> {
        let loader = ObjLoader::new();
        let scene = loader.load_from_str(content)
            .map_err(|e| JsValue::from_str(&format!("Failed to load OBJ: {}", e)))?;

        self.add_loaded_scene(&scene)
    }

    /// Add a loaded scene to the renderer, returns number of meshes added.
    fn add_loaded_scene(&mut self, scene: &LoadedScene) -> Result<u32, JsValue> {
        use crate::math::Quaternion;

        // Compute world transforms for each mesh by walking the node hierarchy
        let mut mesh_transforms: Vec<Option<Matrix4>> = vec![None; scene.meshes.len()];

        // Recursive function to compute world transforms
        fn process_node(
            node_idx: usize,
            parent_transform: &Matrix4,
            scene: &LoadedScene,
            mesh_transforms: &mut [Option<Matrix4>],
        ) {
            if let Some(node) = scene.nodes.get(node_idx) {
                // Compute local transform
                let t = &node.translation;
                let r = &node.rotation;
                let s = &node.scale;

                let local_transform = Matrix4::compose(
                    &Vector3::new(t[0], t[1], t[2]),
                    &Quaternion::new(r[0], r[1], r[2], r[3]),
                    &Vector3::new(s[0], s[1], s[2]),
                );

                // World transform = parent * local
                let world_transform = parent_transform.multiply(&local_transform);

                // Assign world transform to all meshes in this node
                for &mesh_idx in &node.mesh_indices {
                    if mesh_idx < mesh_transforms.len() {
                        mesh_transforms[mesh_idx] = Some(world_transform.clone());
                    }
                }

                // Process children
                for &child_idx in &node.children {
                    process_node(child_idx, &world_transform, scene, mesh_transforms);
                }
            }
        }

        // Start from root nodes with identity transform
        let identity = Matrix4::identity();
        for &root_idx in &scene.root_nodes {
            process_node(root_idx, &identity, scene, &mut mesh_transforms);
        }

        // Load textures from scene into GPU textures (already decoded as RGBA8)
        let mut gpu_textures: HashMap<String, Texture2D> = HashMap::new();
        for (name, loaded_tex) in &scene.textures {
            let texture = Texture2D::from_rgba8(
                &self.device,
                &self.queue,
                &loaded_tex.data,
                loaded_tex.width,
                loaded_tex.height,
                Some(name),
            );
            gpu_textures.insert(name.clone(), texture);
        }

        let mut added = 0u32;

        for (mesh_idx, loaded_mesh) in scene.meshes.iter().enumerate() {
            let geometry = &loaded_mesh.geometry;

            // Build vertex data (position, normal, uv)
            let mut vertex_data = Vec::with_capacity(geometry.positions.len() * 8);
            for i in 0..geometry.positions.len() {
                let pos = geometry.positions[i];
                let normal = geometry.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
                let uv = geometry.uvs.get(i).copied().unwrap_or([0.0, 0.0]);

                vertex_data.extend_from_slice(&pos);
                vertex_data.extend_from_slice(&normal);
                vertex_data.extend_from_slice(&uv);
            }

            let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Mesh {} Vertex Buffer", loaded_mesh.name)),
                contents: bytemuck::cast_slice(&vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Mesh {} Index Buffer", loaded_mesh.name)),
                contents: bytemuck::cast_slice(&geometry.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            // Get material properties
            let material_data = loaded_mesh.material_index
                .and_then(|i| scene.materials.get(i));

            let base_color = material_data
                .map(|m| m.base_color)
                .unwrap_or([0.8, 0.8, 0.8, 1.0]);

            let metallic = material_data
                .map(|m| m.metallic)
                .unwrap_or(0.0);

            let roughness = material_data
                .map(|m| m.roughness)
                .unwrap_or(0.5);

            // Check for textures
            let albedo_texture_name = material_data
                .and_then(|m| m.base_color_texture.as_ref());
            let normal_texture_name = material_data
                .and_then(|m| m.normal_texture.as_ref());
            let mr_texture_name = material_data
                .and_then(|m| m.metallic_roughness_texture.as_ref());

            let has_albedo = albedo_texture_name.is_some() &&
                albedo_texture_name.map(|n| gpu_textures.contains_key(n)).unwrap_or(false);
            let has_normal = normal_texture_name.is_some() &&
                normal_texture_name.map(|n| gpu_textures.contains_key(n)).unwrap_or(false);
            let has_mr = mr_texture_name.is_some() &&
                mr_texture_name.map(|n| gpu_textures.contains_key(n)).unwrap_or(false);

            let mat_uniform = TexturedPbrMaterialUniform {
                base_color,
                metallic,
                roughness,
                ao: 1.0,
                use_albedo_map: if has_albedo { 1.0 } else { 0.0 },
                use_normal_map: if has_normal { 1.0 } else { 0.0 },
                use_metallic_roughness_map: if has_mr { 1.0 } else { 0.0 },
                _padding: [0.0; 2],
            };

            // Get textures or use defaults
            let albedo_tex = albedo_texture_name
                .and_then(|n| gpu_textures.get(n))
                .unwrap_or(&self.white_texture);
            let normal_tex = normal_texture_name
                .and_then(|n| gpu_textures.get(n))
                .unwrap_or(&self.normal_texture);
            let mr_tex = mr_texture_name
                .and_then(|n| gpu_textures.get(n))
                .unwrap_or(&self.white_texture);

            // Create combined texture + shadow bind group
            let texture_bind_group = self.material
                .create_texture_shadow_bind_group(
                    &self.device,
                    albedo_tex,
                    &self.default_sampler,
                    normal_tex,
                    &self.default_sampler,
                    mr_tex,
                    &self.default_sampler,
                    &self.shadow_uniform_buffer,
                    &self.shadow_map_view,
                    &self.shadow_sampler,
                )
                .ok_or_else(|| JsValue::from_str("Failed to create texture bind group"))?;

            // Get the world transform for this mesh, or use identity
            let world_transform = mesh_transforms[mesh_idx].clone().unwrap_or_else(Matrix4::identity);

            let mesh = Self::create_mesh_with_transform(
                &self.device,
                &self.material,
                vertex_buffer,
                index_buffer,
                geometry.indices.len() as u32,
                &world_transform,
                mat_uniform,
                texture_bind_group,
            )?;

            self.meshes.push(mesh);
            added += 1;
        }

        Ok(added)
    }

    /// Clear all loaded meshes.
    #[wasm_bindgen]
    pub fn clear_loaded_meshes(&mut self) {
        self.meshes.clear();
    }

    /// Get total mesh count.
    #[wasm_bindgen]
    pub fn mesh_count(&self) -> u32 {
        self.meshes.len() as u32
    }

    /// Enable or disable bloom effect.
    #[wasm_bindgen]
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.bloom.set_enabled(enabled);
    }

    /// Set bloom intensity (0.0 - 2.0+ recommended).
    #[wasm_bindgen]
    pub fn set_bloom_intensity(&mut self, intensity: f32) {
        let mut settings = self.bloom.settings().clone();
        settings.intensity = intensity;
        self.bloom.set_settings(settings);
    }

    /// Set bloom threshold (0.0 - 1.0, pixels brighter than this glow).
    #[wasm_bindgen]
    pub fn set_bloom_threshold(&mut self, threshold: f32) {
        let mut settings = self.bloom.settings().clone();
        settings.threshold = threshold;
        self.bloom.set_settings(settings);
    }

    /// Set tonemapping exposure.
    #[wasm_bindgen]
    pub fn set_exposure(&mut self, exposure: f32) {
        self.tonemapping.set_exposure(exposure);
    }

    /// Set anti-aliasing mode (0=None, 1=FXAA, 2=SMAA, 3=TAA).
    #[wasm_bindgen]
    pub fn set_aa_mode(&mut self, mode: u32) {
        self.aa_mode = match mode {
            0 => AaMode::None,
            1 => AaMode::Fxaa,
            2 => AaMode::Smaa,
            3 => AaMode::Taa,
            _ => AaMode::Fxaa,
        };
        // Update enabled state based on mode
        self.fxaa.set_enabled(self.aa_mode == AaMode::Fxaa || self.aa_mode == AaMode::None);
        self.smaa.set_enabled(self.aa_mode == AaMode::Smaa);
        self.taa.set_enabled(self.aa_mode == AaMode::Taa);
    }

    /// Get current AA mode (0=None, 1=FXAA, 2=SMAA, 3=TAA).
    #[wasm_bindgen]
    pub fn get_aa_mode(&self) -> u32 {
        match self.aa_mode {
            AaMode::None => 0,
            AaMode::Fxaa => 1,
            AaMode::Smaa => 2,
            AaMode::Taa => 3,
        }
    }

    /// Enable or disable FXAA anti-aliasing (legacy, use set_aa_mode instead).
    #[wasm_bindgen]
    pub fn set_fxaa_enabled(&mut self, enabled: bool) {
        if enabled {
            self.set_aa_mode(1); // FXAA
        } else {
            self.set_aa_mode(0); // None
        }
    }

    /// Set FXAA quality (0=Low, 1=Medium, 2=High).
    #[wasm_bindgen]
    pub fn set_fxaa_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => FxaaQuality::Low,
            1 => FxaaQuality::Medium,
            _ => FxaaQuality::High,
        };
        self.fxaa.set_quality(q);
    }

    /// Set SMAA quality (0=Low, 1=Medium, 2=High, 3=Ultra).
    #[wasm_bindgen]
    pub fn set_smaa_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => SmaaQuality::Low,
            1 => SmaaQuality::Medium,
            2 => SmaaQuality::High,
            _ => SmaaQuality::Ultra,
        };
        self.smaa.set_quality(q);
    }

    /// Set TAA blend factor (0.0-1.0, higher = more temporal stability).
    #[wasm_bindgen]
    pub fn set_taa_blend_factor(&mut self, factor: f32) {
        self.taa.set_blend_factor(factor);
    }

    /// Set TAA sharpness (0.0-1.0).
    #[wasm_bindgen]
    pub fn set_taa_sharpness(&mut self, sharpness: f32) {
        self.taa.set_sharpness(sharpness);
    }

    /// Show or hide the grid helper.
    #[wasm_bindgen]
    pub fn set_grid_visible(&mut self, visible: bool) {
        self.show_grid = visible;
    }

    /// Get grid visibility state.
    #[wasm_bindgen]
    pub fn is_grid_visible(&self) -> bool {
        self.show_grid
    }

    /// Show or hide the axes helper.
    #[wasm_bindgen]
    pub fn set_axes_visible(&mut self, visible: bool) {
        self.show_axes = visible;
    }

    /// Get axes visibility state.
    #[wasm_bindgen]
    pub fn is_axes_visible(&self) -> bool {
        self.show_axes
    }

    /// Enable or disable SSAO.
    #[wasm_bindgen]
    pub fn set_ssao_enabled(&mut self, enabled: bool) {
        self.ssao.set_enabled(enabled);
    }

    /// Check if SSAO is enabled.
    #[wasm_bindgen]
    pub fn is_ssao_enabled(&self) -> bool {
        self.ssao.enabled()
    }

    /// Set SSAO radius (0.1 - 2.0 recommended).
    #[wasm_bindgen]
    pub fn set_ssao_radius(&mut self, radius: f32) {
        self.ssao.set_radius(radius);
    }

    /// Set SSAO intensity (0.0 - 3.0 recommended).
    #[wasm_bindgen]
    pub fn set_ssao_intensity(&mut self, intensity: f32) {
        self.ssao.set_intensity(intensity);
    }

    /// Set SSAO quality (0=Low, 1=Medium, 2=High, 3=Ultra).
    #[wasm_bindgen]
    pub fn set_ssao_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => SsaoQuality::Low,
            1 => SsaoQuality::Medium,
            2 => SsaoQuality::High,
            _ => SsaoQuality::Ultra,
        };
        self.ssao.set_quality(q);
    }

    /// Enable or disable vignette effect.
    #[wasm_bindgen]
    pub fn set_vignette_enabled(&mut self, enabled: bool) {
        self.vignette.set_enabled(enabled);
    }

    /// Check if vignette is enabled.
    #[wasm_bindgen]
    pub fn is_vignette_enabled(&self) -> bool {
        self.vignette.enabled()
    }

    /// Set vignette intensity (0.0 - 1.0).
    #[wasm_bindgen]
    pub fn set_vignette_intensity(&mut self, intensity: f32) {
        self.vignette.set_intensity(intensity);
    }

    /// Set vignette smoothness (0.01 - 2.0).
    #[wasm_bindgen]
    pub fn set_vignette_smoothness(&mut self, smoothness: f32) {
        self.vignette.set_smoothness(smoothness);
    }

    /// Set vignette roundness (0.0 = rectangular, 1.0 = circular).
    #[wasm_bindgen]
    pub fn set_vignette_roundness(&mut self, roundness: f32) {
        self.vignette.set_roundness(roundness);
    }

    /// Enable or disable Depth of Field effect.
    #[wasm_bindgen]
    pub fn set_dof_enabled(&mut self, enabled: bool) {
        self.dof.set_enabled(enabled);
    }

    /// Check if DoF is enabled.
    #[wasm_bindgen]
    pub fn is_dof_enabled(&self) -> bool {
        self.dof.enabled()
    }

    /// Set DoF focal distance (in world units).
    #[wasm_bindgen]
    pub fn set_dof_focal_distance(&mut self, distance: f32) {
        self.dof.set_focal_distance(distance);
    }

    /// Set DoF focal range (sharpness zone around focal distance).
    #[wasm_bindgen]
    pub fn set_dof_focal_range(&mut self, range: f32) {
        self.dof.set_focal_range(range);
    }

    /// Set DoF blur strength (0.0 - 5.0).
    #[wasm_bindgen]
    pub fn set_dof_blur_strength(&mut self, strength: f32) {
        self.dof.set_blur_strength(strength);
    }

    /// Set DoF quality (0=Low, 1=Medium, 2=High).
    #[wasm_bindgen]
    pub fn set_dof_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => DofQuality::Low,
            1 => DofQuality::Medium,
            _ => DofQuality::High,
        };
        self.dof.set_quality(q);
    }

    /// Enable or disable Screen-Space Reflections.
    #[wasm_bindgen]
    pub fn set_ssr_enabled(&mut self, enabled: bool) {
        self.ssr.set_enabled(enabled);
    }

    /// Check if SSR is enabled.
    #[wasm_bindgen]
    pub fn is_ssr_enabled(&self) -> bool {
        self.ssr.enabled()
    }

    /// Set SSR maximum ray distance.
    #[wasm_bindgen]
    pub fn set_ssr_max_distance(&mut self, distance: f32) {
        self.ssr.set_max_distance(distance);
    }

    /// Set SSR depth thickness.
    #[wasm_bindgen]
    pub fn set_ssr_thickness(&mut self, thickness: f32) {
        self.ssr.set_thickness(thickness);
    }

    /// Set SSR reflection intensity (0.0 - 1.0).
    #[wasm_bindgen]
    pub fn set_ssr_intensity(&mut self, intensity: f32) {
        self.ssr.set_intensity(intensity);
    }

    /// Set SSR quality (0=Low, 1=Medium, 2=High, 3=Ultra).
    #[wasm_bindgen]
    pub fn set_ssr_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => SsrQuality::Low,
            1 => SsrQuality::Medium,
            2 => SsrQuality::High,
            _ => SsrQuality::Ultra,
        };
        self.ssr.set_quality(q);
    }

    /// Enable or disable motion blur.
    #[wasm_bindgen]
    pub fn set_motion_blur_enabled(&mut self, enabled: bool) {
        self.motion_blur.set_enabled(enabled);
    }

    /// Check if motion blur is enabled.
    #[wasm_bindgen]
    pub fn is_motion_blur_enabled(&self) -> bool {
        self.motion_blur.enabled()
    }

    /// Set motion blur intensity (0.0 - 2.0).
    #[wasm_bindgen]
    pub fn set_motion_blur_intensity(&mut self, intensity: f32) {
        self.motion_blur.set_intensity(intensity);
    }

    /// Set motion blur maximum blur distance in pixels (1.0 - 64.0).
    #[wasm_bindgen]
    pub fn set_motion_blur_max(&mut self, max_blur: f32) {
        self.motion_blur.set_max_blur(max_blur);
    }

    /// Set motion blur velocity scale (0.1 - 3.0).
    #[wasm_bindgen]
    pub fn set_motion_blur_velocity_scale(&mut self, scale: f32) {
        self.motion_blur.set_velocity_scale(scale);
    }

    /// Set motion blur quality (0=Low, 1=Medium, 2=High).
    #[wasm_bindgen]
    pub fn set_motion_blur_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => MotionBlurQuality::Low,
            1 => MotionBlurQuality::Medium,
            _ => MotionBlurQuality::High,
        };
        self.motion_blur.set_quality(q);
    }

    /// Enable or disable outline effect.
    #[wasm_bindgen]
    pub fn set_outline_enabled(&mut self, enabled: bool) {
        self.outline.set_enabled(enabled);
    }

    /// Check if outline is enabled.
    #[wasm_bindgen]
    pub fn is_outline_enabled(&self) -> bool {
        self.outline.enabled()
    }

    /// Set outline thickness (0.5 - 5.0).
    #[wasm_bindgen]
    pub fn set_outline_thickness(&mut self, thickness: f32) {
        self.outline.set_thickness(thickness);
    }

    /// Set outline depth threshold (0.001 - 1.0).
    #[wasm_bindgen]
    pub fn set_outline_depth_threshold(&mut self, threshold: f32) {
        self.outline.set_depth_threshold(threshold);
    }

    /// Set outline normal threshold (0.0 - 1.0).
    #[wasm_bindgen]
    pub fn set_outline_normal_threshold(&mut self, threshold: f32) {
        self.outline.set_normal_threshold(threshold);
    }

    /// Set outline color (RGB, 0.0 - 1.0).
    #[wasm_bindgen]
    pub fn set_outline_color(&mut self, r: f32, g: f32, b: f32) {
        self.outline.set_color(r, g, b);
    }

    /// Set outline mode (0=Depth, 1=Normal, 2=Combined).
    #[wasm_bindgen]
    pub fn set_outline_mode(&mut self, mode: u32) {
        let m = match mode {
            0 => OutlineMode::Depth,
            1 => OutlineMode::Normal,
            _ => OutlineMode::Combined,
        };
        self.outline.set_mode(m);
    }

    /// Enable or disable color correction.
    #[wasm_bindgen]
    pub fn set_color_correction_enabled(&mut self, enabled: bool) {
        self.color_correction.set_enabled(enabled);
    }

    /// Check if color correction is enabled.
    #[wasm_bindgen]
    pub fn is_color_correction_enabled(&self) -> bool {
        self.color_correction.enabled()
    }

    /// Set color correction brightness (-1.0 to 1.0).
    #[wasm_bindgen]
    pub fn set_cc_brightness(&mut self, brightness: f32) {
        self.color_correction.set_brightness(brightness);
    }

    /// Set color correction contrast (0.0 to 2.0).
    #[wasm_bindgen]
    pub fn set_cc_contrast(&mut self, contrast: f32) {
        self.color_correction.set_contrast(contrast);
    }

    /// Set color correction saturation (0.0 to 2.0).
    #[wasm_bindgen]
    pub fn set_cc_saturation(&mut self, saturation: f32) {
        self.color_correction.set_saturation(saturation);
    }

    /// Set color correction gamma (0.5 to 2.5).
    #[wasm_bindgen]
    pub fn set_cc_gamma(&mut self, gamma: f32) {
        self.color_correction.set_gamma(gamma);
    }

    /// Set color correction temperature (-1.0 cool to 1.0 warm).
    #[wasm_bindgen]
    pub fn set_cc_temperature(&mut self, temperature: f32) {
        self.color_correction.set_temperature(temperature);
    }

    /// Set color correction tint (-1.0 green to 1.0 magenta).
    #[wasm_bindgen]
    pub fn set_cc_tint(&mut self, tint: f32) {
        self.color_correction.set_tint(tint);
    }

    /// Reset color correction to defaults.
    #[wasm_bindgen]
    pub fn reset_color_correction(&mut self) {
        self.color_correction.reset();
    }

    // ========== Hemisphere Light ==========

    /// Enable or disable hemisphere light.
    #[wasm_bindgen]
    pub fn set_hemisphere_light_enabled(&mut self, enabled: bool) {
        self.hemisphere_enabled = enabled;
    }

    /// Check if hemisphere light is enabled.
    #[wasm_bindgen]
    pub fn is_hemisphere_light_enabled(&self) -> bool {
        self.hemisphere_enabled
    }

    /// Set hemisphere light sky color (RGB, 0.0-1.0).
    #[wasm_bindgen]
    pub fn set_hemisphere_sky_color(&mut self, r: f32, g: f32, b: f32) {
        self.hemisphere_sky_color = [r, g, b];
    }

    /// Set hemisphere light ground color (RGB, 0.0-1.0).
    #[wasm_bindgen]
    pub fn set_hemisphere_ground_color(&mut self, r: f32, g: f32, b: f32) {
        self.hemisphere_ground_color = [r, g, b];
    }

    /// Set hemisphere light intensity (0.0 to 2.0).
    #[wasm_bindgen]
    pub fn set_hemisphere_intensity(&mut self, intensity: f32) {
        self.hemisphere_intensity = intensity.clamp(0.0, 2.0);
    }

    /// Get hemisphere light intensity.
    #[wasm_bindgen]
    pub fn get_hemisphere_intensity(&self) -> f32 {
        self.hemisphere_intensity
    }

    // ========== Shadows ==========

    /// Initialize the shadow system (shadows are pre-initialized now).
    #[wasm_bindgen]
    pub fn init_shadows(&mut self) {
        self.shadows_enabled = true;
    }

    /// Initialize the shadow system with a specific resolution.
    #[wasm_bindgen]
    pub fn init_shadows_with_resolution(&mut self, resolution: u32) {
        self.set_shadow_resolution(resolution);
        self.shadows_enabled = true;
    }

    /// Enable or disable shadows.
    #[wasm_bindgen]
    pub fn set_shadows_enabled(&mut self, enabled: bool) {
        self.shadows_enabled = enabled;
    }

    /// Check if shadows are enabled.
    #[wasm_bindgen]
    pub fn is_shadows_enabled(&self) -> bool {
        self.shadows_enabled
    }

    /// Set shadow map resolution (256, 512, 1024, 2048, or 4096).
    #[wasm_bindgen]
    pub fn set_shadow_resolution(&mut self, resolution: u32) {
        let resolution = resolution.clamp(256, 4096);
        self.shadow_config.resolution = resolution;

        // Recreate shadow map texture with new resolution
        self.shadow_map_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map Texture"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.shadow_map_view = self.shadow_map_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Note: Changing shadow resolution requires recreating all mesh texture bind groups
        // since shadow resources are now merged into the texture bind group (group 3).
        // This is a tradeoff for staying within WebGPU's 4 bind group limit.
        // For now, meshes created before resolution change will use the old shadow map.
    }

    /// Get current shadow resolution.
    #[wasm_bindgen]
    pub fn get_shadow_resolution(&self) -> u32 {
        self.shadow_config.resolution
    }

    /// Set PCF mode (0=None, 1=Hardware2x2, 2=Soft3x3, 3=Soft5x5, 4=PoissonDisk).
    #[wasm_bindgen]
    pub fn set_shadow_pcf_mode(&mut self, mode: u32) {
        let pcf_mode = match mode {
            0 => PCFMode::None,
            1 => PCFMode::Hardware2x2,
            2 => PCFMode::Soft3x3,
            3 => PCFMode::Soft5x5,
            _ => PCFMode::PoissonDisk,
        };
        self.shadow_config.pcf_mode = pcf_mode;
    }

    /// Get current PCF mode (0=None, 1=Hardware2x2, 2=Soft3x3, 3=Soft5x5, 4=PoissonDisk).
    #[wasm_bindgen]
    pub fn get_shadow_pcf_mode(&self) -> u32 {
        match self.shadow_config.pcf_mode {
            PCFMode::None => 0,
            PCFMode::Hardware2x2 => 1,
            PCFMode::Soft3x3 => 2,
            PCFMode::Soft5x5 => 3,
            PCFMode::PoissonDisk => 4,
        }
    }

    /// Set shadow bias (0.0 - 0.1, default 0.005).
    #[wasm_bindgen]
    pub fn set_shadow_bias(&mut self, bias: f32) {
        self.shadow_config.bias = bias.clamp(0.0, 0.1);
    }

    /// Get current shadow bias.
    #[wasm_bindgen]
    pub fn get_shadow_bias(&self) -> f32 {
        self.shadow_config.bias
    }

    /// Set shadow normal bias (0.0 - 0.1, default 0.02).
    #[wasm_bindgen]
    pub fn set_shadow_normal_bias(&mut self, normal_bias: f32) {
        self.shadow_config.normal_bias = normal_bias.clamp(0.0, 0.1);
    }

    /// Get current shadow normal bias.
    #[wasm_bindgen]
    pub fn get_shadow_normal_bias(&self) -> f32 {
        self.shadow_config.normal_bias
    }

    /// Set number of shadow cascades for directional lights (1-4).
    /// Note: Currently using single shadow map, cascades not yet implemented in web.
    #[wasm_bindgen]
    pub fn set_shadow_cascades(&mut self, _num_cascades: u32) {
        // Cascades not yet implemented - using single shadow map
    }

    /// Get current number of shadow cascades.
    #[wasm_bindgen]
    pub fn get_shadow_cascades(&self) -> u32 {
        1 // Currently using single shadow map
    }

    /// Set the light direction for shadow casting.
    #[wasm_bindgen]
    pub fn set_shadow_light_direction(&mut self, x: f32, y: f32, z: f32) {
        self.light_direction = [x, y, z];
    }

    /// Add demo shapes to showcase bloom effect.
    #[wasm_bindgen]
    pub fn add_demo_shapes(&mut self) -> Result<(), JsValue> {
        // Create several cubes with different materials
        let shapes = [
            // Bright emissive cubes (will bloom)
            ([-2.0, 0.5, 0.0], [1.5, 0.3, 0.1, 1.0], 0.0, 0.3),   // Bright red/orange
            ([0.0, 0.5, -2.0], [0.2, 1.5, 0.3, 1.0], 0.0, 0.3),   // Bright green
            ([2.0, 0.5, 0.0], [0.2, 0.4, 1.5, 1.0], 0.0, 0.3),    // Bright blue
            ([0.0, 0.5, 2.0], [1.5, 1.2, 0.2, 1.0], 0.0, 0.3),    // Bright yellow
            // Center metallic sphere-ish cube
            ([0.0, 1.0, 0.0], [0.9, 0.9, 0.95, 1.0], 1.0, 0.1),   // Chrome
            // Dark contrast cubes
            ([-1.0, 0.3, -1.0], [0.15, 0.15, 0.15, 1.0], 0.0, 0.8), // Dark gray
            ([1.0, 0.3, 1.0], [0.15, 0.15, 0.15, 1.0], 0.0, 0.8),   // Dark gray
        ];

        for (pos, color, metallic, roughness) in shapes {
            self.add_cube(pos, 0.6, color, metallic, roughness)?;
        }

        // Add a bright white sphere in the center top
        self.add_sphere([0.0, 2.5, 0.0], 0.4, [2.0, 2.0, 2.0, 1.0], 0.0, 0.2)?;

        Ok(())
    }

    /// Add a cube at the given position.
    fn add_cube(
        &mut self,
        position: [f32; 3],
        size: f32,
        color: [f32; 4],
        metallic: f32,
        roughness: f32,
    ) -> Result<(), JsValue> {
        let s = size / 2.0;

        // Cube vertices: position (3) + normal (3) + uv (2) = 8 floats per vertex
        #[rustfmt::skip]
        let vertices: Vec<f32> = vec![
            // Front face
            -s, -s,  s,  0.0, 0.0, 1.0,  0.0, 1.0,
             s, -s,  s,  0.0, 0.0, 1.0,  1.0, 1.0,
             s,  s,  s,  0.0, 0.0, 1.0,  1.0, 0.0,
            -s,  s,  s,  0.0, 0.0, 1.0,  0.0, 0.0,
            // Back face
             s, -s, -s,  0.0, 0.0, -1.0,  0.0, 1.0,
            -s, -s, -s,  0.0, 0.0, -1.0,  1.0, 1.0,
            -s,  s, -s,  0.0, 0.0, -1.0,  1.0, 0.0,
             s,  s, -s,  0.0, 0.0, -1.0,  0.0, 0.0,
            // Top face
            -s,  s,  s,  0.0, 1.0, 0.0,  0.0, 1.0,
             s,  s,  s,  0.0, 1.0, 0.0,  1.0, 1.0,
             s,  s, -s,  0.0, 1.0, 0.0,  1.0, 0.0,
            -s,  s, -s,  0.0, 1.0, 0.0,  0.0, 0.0,
            // Bottom face
            -s, -s, -s,  0.0, -1.0, 0.0,  0.0, 1.0,
             s, -s, -s,  0.0, -1.0, 0.0,  1.0, 1.0,
             s, -s,  s,  0.0, -1.0, 0.0,  1.0, 0.0,
            -s, -s,  s,  0.0, -1.0, 0.0,  0.0, 0.0,
            // Right face
             s, -s,  s,  1.0, 0.0, 0.0,  0.0, 1.0,
             s, -s, -s,  1.0, 0.0, 0.0,  1.0, 1.0,
             s,  s, -s,  1.0, 0.0, 0.0,  1.0, 0.0,
             s,  s,  s,  1.0, 0.0, 0.0,  0.0, 0.0,
            // Left face
            -s, -s, -s,  -1.0, 0.0, 0.0,  0.0, 1.0,
            -s, -s,  s,  -1.0, 0.0, 0.0,  1.0, 1.0,
            -s,  s,  s,  -1.0, 0.0, 0.0,  1.0, 0.0,
            -s,  s, -s,  -1.0, 0.0, 0.0,  0.0, 0.0,
        ];

        #[rustfmt::skip]
        let indices: Vec<u32> = vec![
            0, 1, 2, 0, 2, 3,       // Front
            4, 5, 6, 4, 6, 7,       // Back
            8, 9, 10, 8, 10, 11,    // Top
            12, 13, 14, 12, 14, 15, // Bottom
            16, 17, 18, 16, 18, 19, // Right
            20, 21, 22, 20, 22, 23, // Left
        ];

        self.add_geometry(&vertices, &indices, position, color, metallic, roughness)
    }

    /// Add a UV sphere at the given position.
    fn add_sphere(
        &mut self,
        position: [f32; 3],
        radius: f32,
        color: [f32; 4],
        metallic: f32,
        roughness: f32,
    ) -> Result<(), JsValue> {
        let segments = 16u32;
        let rings = 12u32;

        let mut vertices: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Generate vertices
        for ring in 0..=rings {
            let v = ring as f32 / rings as f32;
            let phi = v * std::f32::consts::PI;

            for seg in 0..=segments {
                let u = seg as f32 / segments as f32;
                let theta = u * std::f32::consts::PI * 2.0;

                let x = (phi.sin()) * (theta.cos());
                let y = phi.cos();
                let z = (phi.sin()) * (theta.sin());

                // Position
                vertices.push(x * radius);
                vertices.push(y * radius);
                vertices.push(z * radius);
                // Normal (same as position for unit sphere)
                vertices.push(x);
                vertices.push(y);
                vertices.push(z);
                // UV
                vertices.push(u);
                vertices.push(v);
            }
        }

        // Generate indices
        for ring in 0..rings {
            for seg in 0..segments {
                let curr_row = ring * (segments + 1);
                let next_row = (ring + 1) * (segments + 1);

                indices.push(curr_row + seg);
                indices.push(next_row + seg);
                indices.push(next_row + seg + 1);

                indices.push(curr_row + seg);
                indices.push(next_row + seg + 1);
                indices.push(curr_row + seg + 1);
            }
        }

        self.add_geometry(&vertices, &indices, position, color, metallic, roughness)
    }

    /// Internal helper to add geometry with material.
    fn add_geometry(
        &mut self,
        vertices: &[f32],
        indices: &[u32],
        position: [f32; 3],
        color: [f32; 4],
        metallic: f32,
        roughness: f32,
    ) -> Result<(), JsValue> {
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shape Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shape Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mat_uniform = TexturedPbrMaterialUniform {
            base_color: color,
            metallic,
            roughness,
            ao: 1.0,
            use_albedo_map: 0.0,
            use_normal_map: 0.0,
            use_metallic_roughness_map: 0.0,
            _padding: [0.0; 2],
        };

        // Create combined texture + shadow bind group with default textures
        let texture_bind_group = self.material
            .create_texture_shadow_bind_group(
                &self.device,
                &self.white_texture,
                &self.default_sampler,
                &self.normal_texture,
                &self.default_sampler,
                &self.white_texture,
                &self.default_sampler,
                &self.shadow_uniform_buffer,
                &self.shadow_map_view,
                &self.shadow_sampler,
            )
            .ok_or_else(|| JsValue::from_str("Failed to create texture bind group"))?;

        // Create transform matrix
        let world_transform = Matrix4::from_translation(&Vector3::new(position[0], position[1], position[2]));

        let mesh = Self::create_mesh_with_transform(
            &self.device,
            &self.material,
            vertex_buffer,
            index_buffer,
            indices.len() as u32,
            &world_transform,
            mat_uniform,
            texture_bind_group,
        )?;

        self.meshes.push(mesh);
        Ok(())
    }
}

fn matrix4_to_array(m: &Matrix4) -> [[f32; 4]; 4] {
    let e = &m.elements;
    [
        [e[0], e[1], e[2], e[3]],
        [e[4], e[5], e[6], e[7]],
        [e[8], e[9], e[10], e[11]],
        [e[12], e[13], e[14], e[15]],
    ]
}

fn matrix3_to_padded_array(m: &Matrix3) -> [[f32; 4]; 3] {
    let e = &m.elements;
    [
        [e[0], e[1], e[2], 0.0],
        [e[3], e[4], e[5], 0.0],
        [e[6], e[7], e[8], 0.0],
    ]
}
