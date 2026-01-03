//! Web bindings for the Ren engine.
//!
//! This module provides JavaScript-friendly APIs via wasm-bindgen.

use wasm_bindgen::prelude::*;
use web_sys::{window, HtmlCanvasElement};
use std::cell::RefCell;
use wgpu::util::DeviceExt;
use bytemuck;

use crate::core::Clock;
use crate::math::{Color, Euler, EulerOrder, Matrix3, Matrix4, Quaternion, Vector3};
use crate::material::{TexturedPbrMaterial, TexturedPbrMaterialUniform, LineMaterial, LineModelUniform, PbrCameraUniform, PbrModelUniform};
use crate::texture::{BrdfLut, Texture2D, Sampler};
use crate::camera::PerspectiveCamera;
use crate::controls::OrbitControls;
use crate::helpers::{AxesHelper, GridHelper, GizmoAxis, GizmoDragResult, GizmoMode, TransformGizmo};
use crate::math::Raycaster;
use crate::loaders::{GltfLoader, ObjLoader, LoadedScene};
use crate::postprocessing::{TonemappingPass, TonemappingMode, BloomPass, BloomSettings, FxaaPass, FxaaQuality, SmaaPass, SmaaQuality, SsaoPass, SsaoQuality, GtaoPass, GtaoQuality, LumenPass, LumenQuality, SsrPass, SsrQuality, TaaPass, VignettePass, DofPass, DofQuality, MotionBlurPass, MotionBlurQuality, OutlinePass, OutlineMode, ColorCorrectionPass, SkyboxPass, ProceduralSkyPass, VolumetricFogPass, VolumetricFogQuality, AutoExposurePass, Pass};
use crate::texture::CubeTexture;
use crate::shadows::{PCFMode, ShadowConfig};
use crate::ibl::prefilter::generate_prefiltered_sky;
use crate::ibl::irradiance::generate_sky_irradiance;
use crate::illumination::ProbeVolume;
use crate::nanite::{NaniteRenderer, NaniteConfig, build_clusters};
use crate::geometry::Vertex;
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

/// Shadow light type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum ShadowLightType {
    /// Directional light (sun).
    #[default]
    Directional = 0,
    /// Spot light.
    Spot = 1,
    /// Point light.
    Point = 2,
}

/// Shadow uniform data for the shader.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct WebShadowUniform {
    /// Light-space view-projection matrix.
    pub light_view_proj: [[f32; 4]; 4],
    /// Shadow params: x=bias, y=normal_bias, z=enabled, w=light_type (0=dir, 1=spot, 2=point)
    pub shadow_params: [f32; 4],
    /// Light direction (xyz) for directional, or position (xyz) for spot/point
    pub light_dir_or_pos: [f32; 4],
    /// Shadow map size: x=width, y=height, z=1/width, w=1/height
    pub shadow_map_size: [f32; 4],
    /// Spot direction (xyz) and range (w)
    pub spot_direction: [f32; 4],
    /// Spot params: x=outer_cos, y=inner_cos, z=intensity, w=pcf_mode
    pub spot_params: [f32; 4],
    /// PCSS params: x=light_size, y=near_plane, z=blocker_search_radius, w=max_filter_radius
    pub pcss_params: [f32; 4],
    /// Contact shadow params: x=enabled, y=max_distance, z=thickness, w=intensity
    pub contact_params: [f32; 4],
}

impl Default for WebShadowUniform {
    fn default() -> Self {
        Self {
            light_view_proj: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            shadow_params: [0.005, 0.02, 0.0, 0.0], // bias, normal_bias, disabled, directional
            light_dir_or_pos: [-0.5, -1.0, -0.3, 0.0],
            shadow_map_size: [2048.0, 2048.0, 1.0 / 2048.0, 1.0 / 2048.0],
            spot_direction: [0.0, -1.0, 0.0, 10.0], // direction (0,-1,0), range 10
            spot_params: [0.9063, 0.9659, 1.0, 2.0], // cos(25°), cos(15°), intensity 1, pcf_mode (Soft3x3)
            pcss_params: [0.5, 0.1, 5.0, 10.0], // light_size, near_plane, blocker_search_radius, max_filter_radius
            contact_params: [0.0, 0.5, 0.05, 0.5], // disabled, max_distance, thickness, intensity
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
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    // Copy of depth texture for particle soft-fade (can't read and write same texture)
    depth_copy_texture: wgpu::Texture,
    depth_copy_view: wgpu::TextureView,
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
    auto_exposure: AutoExposurePass,
    fxaa: FxaaPass,
    smaa: SmaaPass,
    taa: TaaPass,
    ssao: SsaoPass,
    gtao: GtaoPass,
    lumen: LumenPass,
    probe_volume: ProbeVolume,
    ssr: SsrPass,
    vignette: VignettePass,
    dof: DofPass,
    motion_blur: MotionBlurPass,
    outline: OutlinePass,
    color_correction: ColorCorrectionPass,
    volumetric_fog: VolumetricFogPass,
    aa_mode: AaMode,
    surface_format: wgpu::TextureFormat,
    hdr_format: wgpu::TextureFormat,
    // HDR display support
    hdr_display_available: bool,
    hdr_display_format: Option<wgpu::TextureFormat>,
    hdr_output_enabled: bool,
    // Hemisphere light state
    hemisphere_enabled: bool,
    hemisphere_sky_color: [f32; 3],
    hemisphere_ground_color: [f32; 3],
    hemisphere_intensity: f32,
    // IBL intensity
    ibl_diffuse_intensity: f32,
    ibl_specular_intensity: f32,
    // Scene lights (4 configurable lights)
    lights: [[f32; 8]; 4], // Each light: [pos.x, pos.y, pos.z, intensity, color.r, color.g, color.b, enabled]
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
    light_intensity: f32,
    light_color: [f32; 3],
    /// Cached light view-projection matrix for volumetric fog.
    light_view_proj_matrix: [[f32; 4]; 4],
    // Shadow light type and spot light settings
    shadow_light_type: ShadowLightType,
    spot_position: [f32; 3],
    spot_direction: [f32; 3],
    spot_range: f32,
    spot_inner_angle: f32,
    spot_outer_angle: f32,
    // Cube shadow map for point lights
    shadow_cube_map_texture: wgpu::Texture,
    shadow_cube_map_view: wgpu::TextureView,
    shadow_cube_face_views: [wgpu::TextureView; 6],
    shadow_cube_sampler: wgpu::Sampler,
    // Skybox
    skybox_pass: SkyboxPass,
    skybox_texture: CubeTexture,
    skybox_bind_group: wgpu::BindGroup,
    skybox_sampler: wgpu::Sampler,
    skybox_enabled: bool,
    // Procedural sky
    procedural_sky: ProceduralSkyPass,
    // Prefiltered environment map for IBL specular
    prefiltered_env_texture: wgpu::Texture,
    prefiltered_env_view: wgpu::TextureView,
    // Irradiance map for IBL diffuse
    irradiance_texture: wgpu::Texture,
    irradiance_view: wgpu::TextureView,
    // BRDF Look-Up Table for IBL
    brdf_lut: BrdfLut,
    // Detail normal map for micro-surface detail
    detail_normal_map: crate::texture::DetailNormalMap,
    // Detail albedo map for micro-surface color variation
    detail_albedo_map: crate::texture::DetailAlbedoMap,
    // Detail normal settings: enabled, scale, intensity, max_distance
    detail_enabled: bool,
    detail_scale: f32,
    detail_intensity: f32,
    detail_max_distance: f32,
    // Detail albedo settings: enabled, scale, intensity, blend_mode
    detail_albedo_enabled: bool,
    detail_albedo_scale: f32,
    detail_albedo_intensity: f32,
    detail_albedo_blend_mode: u32,
    // Render mode: 0=Lit, 1=Unlit, 2=Normals, 3=Depth, 4=Metallic, 5=Roughness, 6=AO, 7=UVs, 8=Flat
    render_mode: u32,
    // Wireframe rendering
    wireframe_supported: bool,
    wireframe_enabled: bool,
    // Particle systems
    particle_systems: Vec<crate::particles::ParticleSystem>,
    particle_sampler: wgpu::Sampler,
    // Nanite virtualized geometry
    nanite_renderer: Option<NaniteRenderer>,
    nanite_enabled: bool,
    nanite_camera_bind_group: Option<wgpu::BindGroup>,
    // Transform gizmo
    gizmo: TransformGizmo,
    gizmo_line_object: Option<LineObject>,
    gizmo_enabled: bool,
    gizmo_target_mesh: Option<usize>,
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

        // Check if wireframe (polygon line mode) is supported
        let adapter_features = adapter.features();
        let wireframe_supported = adapter_features.contains(wgpu::Features::POLYGON_MODE_LINE);

        let requested_features = if wireframe_supported {
            wgpu::Features::POLYGON_MODE_LINE
        } else {
            wgpu::Features::empty()
        };

        // Get adapter limits and request higher storage buffer size for Nanite
        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::downlevel_webgl2_defaults()
            .using_resolution(adapter_limits.clone());

        // Request higher storage buffer limit for Nanite (default is 128MB, need 256MB+)
        // Use adapter's limit if available, otherwise request 256MB minimum
        required_limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .max(256 * 1024 * 1024);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Ren Device"),
                    required_features: requested_features,
                    required_limits,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to get device: {:?}", e)))?;

        let surface_caps = surface.get_capabilities(&adapter);

        // Check for HDR display formats (10-bit or 16-bit float)
        let hdr_display_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| matches!(f,
                wgpu::TextureFormat::Rgba16Float |
                wgpu::TextureFormat::Rgb10a2Unorm
            ));

        let has_hdr_display = hdr_display_format.is_some();

        // Default to SDR sRGB format (can switch to HDR later via API)
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

        // Log HDR display capability
        if has_hdr_display {
            web_sys::console::log_1(&format!("HDR display supported: {:?}", hdr_display_format.unwrap()).into());
        }

        // HDR format for scene rendering (before tonemapping)
        let hdr_format = wgpu::TextureFormat::Rgba16Float;

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Depth copy texture for particle soft-fade (can't read and write same texture in a pass)
        let depth_copy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Copy Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let depth_copy_view = depth_copy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create material and pipeline (target HDR format for full precision)
        let mut material = TexturedPbrMaterial::new();
        material.build_pipeline(&device, &queue, hdr_format, depth_format, wireframe_supported);

        // Create line material and pipeline (also targets HDR format)
        let mut line_material = LineMaterial::new();
        line_material.build_pipeline(&device, hdr_format, depth_format);

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
            render_mode: 0,
            hemisphere_sky: [0.6, 0.75, 1.0, 0.0],  // Disabled by default
            hemisphere_ground: [0.4, 0.3, 0.2, 1.0],
            ibl_settings: [0.3, 1.0, 0.0, 0.0],
            // Car studio lighting preset
            light0_pos: [5.0, 8.0, 5.0, 15.0],
            light0_color: [1.0, 0.98, 0.95, 1.0],
            light1_pos: [-5.0, 6.0, 3.0, 10.0],
            light1_color: [0.9, 0.95, 1.0, 1.0],
            light2_pos: [0.0, 4.0, -6.0, 8.0],
            light2_color: [1.0, 1.0, 1.0, 1.0],
            light3_pos: [-3.0, 1.0, -3.0, 5.0],
            light3_color: [0.8, 0.85, 0.9, 1.0],
            detail_settings: [0.0, 10.0, 0.3, 5.0],  // Disabled by default
            detail_albedo_settings: [0.0, 10.0, 0.3, 0.0],  // Disabled by default
            // Rect lights: disabled by default
            rectlight0_pos: [0.0, 0.0, 0.0, 0.0],
            rectlight0_dir_width: [0.0, 0.0, -1.0, 1.0],
            rectlight0_tan_height: [1.0, 0.0, 0.0, 1.0],
            rectlight0_color: [1.0, 1.0, 1.0, 10.0],
            rectlight1_pos: [0.0, 0.0, 0.0, 0.0],
            rectlight1_dir_width: [0.0, 0.0, -1.0, 1.0],
            rectlight1_tan_height: [1.0, 0.0, 0.0, 1.0],
            rectlight1_color: [1.0, 1.0, 1.0, 10.0],
            // Capsule lights: disabled by default
            capsule0_start: [0.0, 0.0, 0.0, 0.0],
            capsule0_end_radius: [1.0, 0.0, 0.0, 0.05],
            capsule0_color: [1.0, 1.0, 1.0, 10.0],
            capsule1_start: [0.0, 0.0, 0.0, 0.0],
            capsule1_end_radius: [1.0, 0.0, 0.0, 0.05],
            capsule1_color: [1.0, 1.0, 1.0, 10.0],
            // Disk lights: disabled by default
            disk0_pos: [0.0, 0.0, 0.0, 0.0],
            disk0_dir_radius: [0.0, -1.0, 0.0, 0.5],
            disk0_color: [1.0, 1.0, 1.0, 10.0],
            disk1_pos: [0.0, 0.0, 0.0, 0.0],
            disk1_dir_radius: [0.0, -1.0, 0.0, 0.5],
            disk1_color: [1.0, 1.0, 1.0, 10.0],
            // Sphere lights: disabled by default
            sphere0_pos: [0.0, 0.0, 0.0, 0.0],
            sphere0_radius_range: [0.1, 20.0, 0.0, 0.0],
            sphere0_color: [1.0, 1.0, 1.0, 10.0],
            sphere1_pos: [0.0, 0.0, 0.0, 0.0],
            sphere1_radius_range: [0.1, 20.0, 0.0, 0.0],
            sphere1_color: [1.0, 1.0, 1.0, 10.0],
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

        // Create scene render target for post-processing (HDR format for full precision)
        let scene_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: hdr_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let scene_view = scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bloom input texture (HDR format for bloom to read from, also used by DoF)
        let bloom_input_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Input Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: hdr_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
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
        bloom.init(&device, hdr_format, width, height);

        // Create tonemapping pass
        let mut tonemapping = TonemappingPass::new();
        tonemapping.set_exposure(1.5);
        tonemapping.set_mode(TonemappingMode::AgX);
        tonemapping.init(&device, surface_format);

        // Create auto-exposure pass (disabled by default, user can enable for eye adaptation)
        let mut auto_exposure = AutoExposurePass::new();
        auto_exposure.set_enabled(false);
        auto_exposure.init(&device, width, height);

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
        ssao.init(&device, &queue, hdr_format, width, height);

        // Create GTAO pass (disabled by default, more advanced than SSAO)
        let mut gtao = GtaoPass::new();
        gtao.set_quality(GtaoQuality::Medium);
        gtao.set_enabled(false); // Disabled by default, user can switch from SSAO
        gtao.init(&device, &queue, hdr_format, width, height);

        // Create Lumen GI pass (disabled by default)
        let mut lumen = LumenPass::new();
        lumen.set_quality(LumenQuality::Medium);
        lumen.set_enabled(false);
        lumen.init(&device, &queue, hdr_format, width, height);

        // Create irradiance probe volume for GI fallback
        let mut probe_volume = ProbeVolume::default();
        probe_volume.init(&device, &queue);

        // Create SSR pass (renders to HDR scene)
        let mut ssr = SsrPass::new();
        ssr.set_enabled(false); // Disabled by default
        ssr.init(&device, hdr_format, width, height);

        // Create vignette pass (post-tonemapping, uses surface format)
        let mut vignette = VignettePass::new();
        vignette.init(&device, surface_format, width, height);

        // Create DoF pass (outputs to surface format for vignette texture)
        let mut dof = DofPass::new();
        dof.set_enabled(false); // Disabled by default
        dof.init(&device, surface_format, width, height);

        // Create motion blur pass (renders to HDR scene)
        let motion_blur = MotionBlurPass::new(&device, width, height, hdr_format);

        // Create outline pass (renders to HDR scene)
        let outline = OutlinePass::new(&device, width, height, hdr_format);

        // Create color correction pass
        let color_correction = ColorCorrectionPass::new(&device, surface_format);

        // Create volumetric fog pass (disabled by default)
        let mut volumetric_fog = VolumetricFogPass::new();
        volumetric_fog.set_enabled(false);
        volumetric_fog.init(&device, &queue, hdr_format, width, height);

        // ========== Shadow System Setup ==========
        let shadow_config = ShadowConfig::default();
        let shadow_resolution = shadow_config.resolution;

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

        // Create cube shadow map for point lights (6 faces)
        let shadow_cube_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Cube Map Texture"),
            size: wgpu::Extent3d {
                width: shadow_resolution,
                height: shadow_resolution,
                depth_or_array_layers: 6, // 6 faces for cube map
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create cube view for sampling
        let shadow_cube_map_view = shadow_cube_map_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow Cube Map View"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        // Create individual face views for rendering
        let shadow_cube_face_views: [wgpu::TextureView; 6] = std::array::from_fn(|i| {
            shadow_cube_map_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Shadow Cube Face {} View", i)),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: i as u32,
                array_layer_count: Some(1),
            })
        });

        // Cube shadow sampler (same comparison settings)
        let shadow_cube_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Cube Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // ========== Skybox Setup ==========
        // Create skybox pass (must match scene texture format - HDR)
        let skybox_pass = SkyboxPass::new(&device, hdr_format);

        // Create default procedural sky cubemap
        let skybox_texture = CubeTexture::default_sky(&device, &queue);

        // Create skybox sampler (linear filtering for smooth gradients)
        let skybox_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Skybox Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create skybox texture bind group
        let skybox_bind_group = skybox_pass.create_texture_bind_group(
            &device,
            skybox_texture.view(),
            &skybox_sampler,
        );

        // Create procedural sky pass
        let procedural_sky = ProceduralSkyPass::new(&device, hdr_format);

        // ========== BRDF LUT Setup ==========
        // Generate BRDF Look-Up Table for proper IBL split-sum approximation
        let brdf_lut = BrdfLut::new(&device, &queue);

        // ========== Detail Normal Map ==========
        // Generate procedural detail normal map for micro-surface detail
        let detail_normal_map = crate::texture::DetailNormalMap::new(&device, &queue);

        // ========== Detail Albedo Map ==========
        // Generate procedural detail albedo map for micro-surface color variation
        let detail_albedo_map = crate::texture::DetailAlbedoMap::new(&device, &queue);

        // ========== Particle System Sampler ==========
        let particle_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Particle Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ========== Prefiltered Environment Map ==========
        // Generate prefiltered cubemap for roughness-based IBL specular
        // Each mip level is convolved for increasing roughness
        let prefiltered_env_texture = generate_prefiltered_sky(&device, &queue, 64, 5);
        let prefiltered_env_view = prefiltered_env_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Prefiltered Env View"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(5),
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        // ========== Irradiance Map ==========
        // Generate irradiance cubemap for diffuse IBL
        // Low-res since irradiance is a low-frequency signal
        let irradiance_texture = generate_sky_irradiance(&device, &queue, 32);
        let irradiance_view = irradiance_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Irradiance View"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
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

        // Vertex buffer layout matching the expanded vertex format (with barycentric coords)
        let shadow_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 44, // 3 floats pos + 3 floats normal + 2 floats uv + 3 floats bary = 11 floats = 44 bytes
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
            adapter,
            device,
            queue,
            surface,
            surface_config,
            depth_texture,
            depth_view,
            depth_copy_texture,
            depth_copy_view,
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
            auto_exposure,
            fxaa,
            smaa,
            taa,
            ssao,
            gtao,
            lumen,
            probe_volume,
            ssr,
            vignette,
            dof,
            motion_blur,
            outline,
            color_correction,
            volumetric_fog,
            aa_mode: AaMode::Fxaa,
            surface_format,
            hdr_format,
            // HDR display support
            hdr_display_available: has_hdr_display,
            hdr_display_format,
            hdr_output_enabled: false,
            // Hemisphere light defaults
            hemisphere_enabled: false,
            hemisphere_sky_color: [0.6, 0.75, 1.0],
            hemisphere_ground_color: [0.4, 0.3, 0.2],
            hemisphere_intensity: 1.0,
            // IBL intensity
            ibl_diffuse_intensity: 0.3,
            ibl_specular_intensity: 1.0,
            // Scene lights - car studio preset
            lights: [
                [5.0, 8.0, 5.0, 15.0, 1.0, 0.98, 0.95, 1.0],     // Key light: warm white
                [-5.0, 6.0, 3.0, 10.0, 0.9, 0.95, 1.0, 1.0],     // Fill light: cool white
                [0.0, 4.0, -6.0, 8.0, 1.0, 1.0, 1.0, 1.0],       // Rim light: pure white
                [-3.0, 1.0, -3.0, 5.0, 0.8, 0.85, 0.9, 1.0],     // Ground bounce: slight blue
            ],
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
            light_intensity: 1.0,
            light_color: [1.0, 1.0, 1.0],
            light_view_proj_matrix: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            shadow_light_type: ShadowLightType::Directional,
            spot_position: [0.0, 5.0, 0.0],
            spot_direction: [0.0, -1.0, 0.0],
            spot_range: 15.0,
            spot_inner_angle: 15.0_f32.to_radians(),
            spot_outer_angle: 25.0_f32.to_radians(),
            shadow_cube_map_texture,
            shadow_cube_map_view,
            shadow_cube_face_views,
            shadow_cube_sampler,
            skybox_pass,
            skybox_texture,
            skybox_bind_group,
            skybox_sampler,
            skybox_enabled: true,
            procedural_sky,
            prefiltered_env_texture,
            prefiltered_env_view,
            irradiance_texture,
            irradiance_view,
            brdf_lut,
            detail_normal_map,
            detail_albedo_map,
            detail_enabled: false,
            detail_scale: 10.0,
            detail_intensity: 0.3,
            detail_max_distance: 5.0,
            detail_albedo_enabled: false,
            detail_albedo_scale: 10.0,
            detail_albedo_intensity: 0.3,
            detail_albedo_blend_mode: 0, // 0=overlay, 1=multiply, 2=soft_light
            render_mode: 0,
            wireframe_supported,
            wireframe_enabled: false,
            particle_systems: Vec::new(),
            particle_sampler,
            // Nanite will be lazily initialized when first mesh is added
            nanite_renderer: None,
            nanite_enabled: true,
            nanite_camera_bind_group: None,
            // Transform gizmo
            gizmo: TransformGizmo::new(),
            gizmo_line_object: None,
            gizmo_enabled: false,
            gizmo_target_mesh: None,
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
        let delta = clock.get_delta() as f32;
        let elapsed = clock.get_elapsed_time() as f32;
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
                render_mode: self.render_mode,
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
                ibl_settings: [
                    self.ibl_diffuse_intensity,
                    self.ibl_specular_intensity,
                    if self.procedural_sky.enabled() { 1.0 } else { 0.0 },
                    0.0,
                ],
                light0_pos: [self.lights[0][0], self.lights[0][1], self.lights[0][2], self.lights[0][3]],
                light0_color: [self.lights[0][4], self.lights[0][5], self.lights[0][6], self.lights[0][7]],
                light1_pos: [self.lights[1][0], self.lights[1][1], self.lights[1][2], self.lights[1][3]],
                light1_color: [self.lights[1][4], self.lights[1][5], self.lights[1][6], self.lights[1][7]],
                light2_pos: [self.lights[2][0], self.lights[2][1], self.lights[2][2], self.lights[2][3]],
                light2_color: [self.lights[2][4], self.lights[2][5], self.lights[2][6], self.lights[2][7]],
                light3_pos: [self.lights[3][0], self.lights[3][1], self.lights[3][2], self.lights[3][3]],
                light3_color: [self.lights[3][4], self.lights[3][5], self.lights[3][6], self.lights[3][7]],
                detail_settings: [
                    if self.detail_enabled { 1.0 } else { 0.0 },
                    self.detail_scale,
                    self.detail_intensity,
                    self.detail_max_distance,
                ],
                detail_albedo_settings: [
                    if self.detail_albedo_enabled { 1.0 } else { 0.0 },
                    self.detail_albedo_scale,
                    self.detail_albedo_intensity,
                    self.detail_albedo_blend_mode as f32,
                ],
                // Rect lights: disabled by default (can be enabled via API)
                rectlight0_pos: [0.0, 0.0, 0.0, 0.0],
                rectlight0_dir_width: [0.0, 0.0, -1.0, 1.0],
                rectlight0_tan_height: [1.0, 0.0, 0.0, 1.0],
                rectlight0_color: [1.0, 1.0, 1.0, 10.0],
                rectlight1_pos: [0.0, 0.0, 0.0, 0.0],
                rectlight1_dir_width: [0.0, 0.0, -1.0, 1.0],
                rectlight1_tan_height: [1.0, 0.0, 0.0, 1.0],
                rectlight1_color: [1.0, 1.0, 1.0, 10.0],
                // Capsule lights: disabled by default
                capsule0_start: [0.0, 0.0, 0.0, 0.0],
                capsule0_end_radius: [1.0, 0.0, 0.0, 0.05],
                capsule0_color: [1.0, 1.0, 1.0, 10.0],
                capsule1_start: [0.0, 0.0, 0.0, 0.0],
                capsule1_end_radius: [1.0, 0.0, 0.0, 0.05],
                capsule1_color: [1.0, 1.0, 1.0, 10.0],
                // Disk lights: disabled by default
                disk0_pos: [0.0, 0.0, 0.0, 0.0],
                disk0_dir_radius: [0.0, -1.0, 0.0, 0.5],
                disk0_color: [1.0, 1.0, 1.0, 10.0],
                disk1_pos: [0.0, 0.0, 0.0, 0.0],
                disk1_dir_radius: [0.0, -1.0, 0.0, 0.5],
                disk1_color: [1.0, 1.0, 1.0, 10.0],
                // Sphere lights: disabled by default
                sphere0_pos: [0.0, 0.0, 0.0, 0.0],
                sphere0_radius_range: [0.1, 20.0, 0.0, 0.0],
                sphere0_color: [1.0, 1.0, 1.0, 10.0],
                sphere1_pos: [0.0, 0.0, 0.0, 0.0],
                sphere1_radius_range: [0.1, 20.0, 0.0, 0.0],
                sphere1_color: [1.0, 1.0, 1.0, 10.0],
            };
            self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

            // Update SSAO projection matrices
            if self.ssao.enabled() {
                let proj = matrix4_to_array(camera.projection_matrix());
                let inv_proj = matrix4_to_array(&camera.projection_matrix().inverse());
                self.ssao.set_projection(proj, inv_proj, camera.near, camera.far);
            }

            // Update GTAO projection matrices
            if self.gtao.enabled() {
                let proj = matrix4_to_array(camera.projection_matrix());
                let inv_proj = matrix4_to_array(&camera.projection_matrix().inverse());
                let view = matrix4_to_array(camera.view_matrix());
                self.gtao.set_matrices(proj, inv_proj, view, camera.near, camera.far, camera.fov);
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

            // Update Lumen GI matrices
            if self.lumen.enabled() {
                let proj = matrix4_to_array(camera.projection_matrix());
                let inv_proj = matrix4_to_array(&camera.projection_matrix().inverse());
                let view = matrix4_to_array(camera.view_matrix());
                let inv_view = matrix4_to_array(&camera.view_matrix().inverse());
                self.lumen.set_matrices(proj, inv_proj, view, inv_view, camera.near, camera.far);
            }

            // Update volumetric fog matrices
            if self.volumetric_fog.enabled() {
                let inv_view_proj = matrix4_to_array(&camera.view_projection_matrix().inverse());
                let camera_pos = [camera.position.x, camera.position.y, camera.position.z];
                self.volumetric_fog.set_matrices(inv_view_proj, camera_pos, camera.near, camera.far);
                // Use shadow light direction for god rays
                let light_dir = self.light_direction;
                let light_color = self.lights[0][4..7].try_into().unwrap_or([1.0, 1.0, 1.0]);
                let light_intensity = self.lights[0][3];
                self.volumetric_fog.set_light(light_dir, light_intensity, light_color);
            }

            // Update outline camera planes
            if self.outline.enabled() {
                self.outline.set_camera_planes(camera.near, camera.far);
            }

            // Update skybox uniform (needs inverse view-proj for world direction)
            let inv_view_proj = matrix4_to_array(&camera.view_projection_matrix().inverse());
            if self.skybox_enabled {
                self.skybox_pass.update_uniform(&self.queue, inv_view_proj);
            }
            // Update procedural sky uniform
            if self.procedural_sky.enabled() {
                self.procedural_sky.update_uniform(&self.queue, inv_view_proj, elapsed);
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

        // Use simple backgrounds for debug modes
        let clear_color = match self.render_mode {
            8 => wgpu::Color { r: 0.05, g: 0.05, b: 0.06, a: 1.0 }, // Flat: very dark gray
            9 => wgpu::Color::BLACK, // Wireframe: black
            _ => wgpu::Color {
                r: self.clear_color.r as f64,
                g: self.clear_color.g as f64,
                b: self.clear_color.b as f64,
                a: 1.0,
            }
        };

        // ========== Shadow Pass ==========
        if self.shadows_enabled {
            let resolution = self.shadow_config.resolution as f32;

            // Calculate light-space matrix based on light type
            let (light_view_proj, shadow_uniform) = match self.shadow_light_type {
                ShadowLightType::Directional => {
                    // Directional light: orthographic projection
                    let light_dir = Vector3::new(self.light_direction[0], self.light_direction[1], self.light_direction[2]).normalized();
                    let light_pos = Vector3::new(-light_dir.x * 15.0, -light_dir.y * 15.0, -light_dir.z * 15.0);
                    let light_target = Vector3::ZERO;
                    let light_up = if light_dir.y.abs() > 0.99 { Vector3::new(0.0, 0.0, 1.0) } else { Vector3::new(0.0, 1.0, 0.0) };

                    let light_view = Matrix4::look_at(&light_pos, &light_target, &light_up);
                    let ortho_size = 20.0;
                    let light_proj = Matrix4::orthographic(-ortho_size, ortho_size, -ortho_size, ortho_size, 0.1, 50.0);
                    let view_proj = light_proj.multiply(&light_view);

                    let uniform = WebShadowUniform {
                        light_view_proj: matrix4_to_array(&view_proj),
                        shadow_params: [
                            self.shadow_config.bias,
                            self.shadow_config.normal_bias,
                            1.0, // enabled
                            ShadowLightType::Directional as u32 as f32,
                        ],
                        light_dir_or_pos: [light_dir.x, light_dir.y, light_dir.z, 0.0],
                        shadow_map_size: [resolution, resolution, 1.0 / resolution, 1.0 / resolution],
                        spot_direction: [0.0, -1.0, 0.0, 10.0],
                        spot_params: [0.9063, 0.9659, self.light_intensity, self.shadow_config.pcf_mode as u32 as f32],
                        pcss_params: [
                            self.shadow_config.pcss.light_size,
                            self.shadow_config.pcss.near_plane,
                            5.0, // blocker search radius in texels
                            self.shadow_config.pcss.max_filter_radius,
                        ],
                        contact_params: [
                            if self.shadow_config.contact.enabled { 1.0 } else { 0.0 },
                            self.shadow_config.contact.max_distance,
                            self.shadow_config.contact.thickness,
                            self.shadow_config.contact.intensity,
                        ],
                    };
                    (view_proj, uniform)
                },
                ShadowLightType::Spot => {
                    // Spot light: perspective projection
                    let light_pos = Vector3::new(self.spot_position[0], self.spot_position[1], self.spot_position[2]);
                    let spot_dir = Vector3::new(self.spot_direction[0], self.spot_direction[1], self.spot_direction[2]).normalized();
                    let light_target = light_pos + spot_dir;
                    let light_up = if spot_dir.y.abs() > 0.99 { Vector3::new(0.0, 0.0, 1.0) } else { Vector3::new(0.0, 1.0, 0.0) };

                    let light_view = Matrix4::look_at(&light_pos, &light_target, &light_up);
                    // Use outer angle * 2 for the perspective FOV
                    let fov = self.spot_outer_angle * 2.0;
                    let light_proj = Matrix4::perspective(fov, 1.0, 0.1, self.spot_range);
                    let view_proj = light_proj.multiply(&light_view);

                    let uniform = WebShadowUniform {
                        light_view_proj: matrix4_to_array(&view_proj),
                        shadow_params: [
                            self.shadow_config.bias,
                            self.shadow_config.normal_bias,
                            1.0, // enabled
                            ShadowLightType::Spot as u32 as f32,
                        ],
                        light_dir_or_pos: [light_pos.x, light_pos.y, light_pos.z, 0.0],
                        shadow_map_size: [resolution, resolution, 1.0 / resolution, 1.0 / resolution],
                        spot_direction: [spot_dir.x, spot_dir.y, spot_dir.z, self.spot_range],
                        spot_params: [self.spot_outer_angle.cos(), self.spot_inner_angle.cos(), self.light_intensity, self.shadow_config.pcf_mode as u32 as f32],
                        pcss_params: [
                            self.shadow_config.pcss.light_size,
                            self.shadow_config.pcss.near_plane,
                            5.0,
                            self.shadow_config.pcss.max_filter_radius,
                        ],
                        contact_params: [
                            if self.shadow_config.contact.enabled { 1.0 } else { 0.0 },
                            self.shadow_config.contact.max_distance,
                            self.shadow_config.contact.thickness,
                            self.shadow_config.contact.intensity,
                        ],
                    };
                    (view_proj, uniform)
                },
                ShadowLightType::Point => {
                    // Point light: cube shadow map (6 faces)
                    let light_pos = Vector3::new(self.spot_position[0], self.spot_position[1], self.spot_position[2]);

                    // Use identity matrix as placeholder - actual rendering is done per-face below
                    let view_proj = Matrix4::identity();

                    let uniform = WebShadowUniform {
                        light_view_proj: matrix4_to_array(&view_proj), // Not used for cube sampling
                        shadow_params: [
                            self.shadow_config.bias,
                            self.shadow_config.normal_bias,
                            1.0, // enabled
                            ShadowLightType::Point as u32 as f32,
                        ],
                        light_dir_or_pos: [light_pos.x, light_pos.y, light_pos.z, 0.0],
                        shadow_map_size: [resolution, resolution, 1.0 / resolution, 1.0 / resolution],
                        spot_direction: [0.0, -1.0, 0.0, self.spot_range], // w = range for cube sampling
                        spot_params: [0.0, 0.0, self.light_intensity, self.shadow_config.pcf_mode as u32 as f32],
                        pcss_params: [
                            self.shadow_config.pcss.light_size,
                            self.shadow_config.pcss.near_plane,
                            5.0,
                            self.shadow_config.pcss.max_filter_radius,
                        ],
                        contact_params: [
                            if self.shadow_config.contact.enabled { 1.0 } else { 0.0 },
                            self.shadow_config.contact.max_distance,
                            self.shadow_config.contact.thickness,
                            self.shadow_config.contact.intensity,
                        ],
                    };
                    (view_proj, uniform)
                },
            };

            self.queue.write_buffer(&self.shadow_uniform_buffer, 0, bytemuck::cast_slice(&[shadow_uniform]));

            // Store light view-projection matrix for volumetric fog
            self.light_view_proj_matrix = matrix4_to_array(&light_view_proj);

            // Create bind group layout for shadow pass (reused for all passes)
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

            // For point lights, render 6 cube faces
            if matches!(self.shadow_light_type, ShadowLightType::Point) {
                let light_pos = Vector3::new(self.spot_position[0], self.spot_position[1], self.spot_position[2]);
                let light_proj = Matrix4::perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, self.spot_range);

                // Cube face directions: +X, -X, +Y, -Y, +Z, -Z
                let face_directions: [(Vector3, Vector3); 6] = [
                    (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),  // +X
                    (Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0)), // -X
                    (Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0)),   // +Y
                    (Vector3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 0.0, -1.0)), // -Y
                    (Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, -1.0, 0.0)),  // +Z
                    (Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, -1.0, 0.0)), // -Z
                ];

                for (face_idx, (dir, up)) in face_directions.iter().enumerate() {
                    let light_target = Vector3::new(
                        light_pos.x + dir.x,
                        light_pos.y + dir.y,
                        light_pos.z + dir.z,
                    );
                    let light_view = Matrix4::look_at(&light_pos, &light_target, up);
                    let face_view_proj = light_proj.multiply(&light_view);

                    // Create light buffer for this face
                    let light_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("Shadow Cube Face {} Light Buffer", face_idx)),
                        contents: bytemuck::cast_slice(&matrix4_to_array(&face_view_proj)),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                    let light_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Shadow Cube Face {} Light Bind Group", face_idx)),
                        layout: &shadow_light_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: light_buffer.as_entire_binding(),
                        }],
                    });

                    // Render to this cube face
                    {
                        let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some(&format!("Shadow Cube Face {} Pass", face_idx)),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &self.shadow_cube_face_views[face_idx],
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

                        // Render all meshes to this cube face
                        for mesh in &self.meshes {
                            let shadow_model_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("Shadow Cube Model Bind Group"),
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
                }
            } else {
                // Directional or Spot light: single 2D shadow map
                let light_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Shadow Light Buffer"),
                    contents: bytemuck::cast_slice(&matrix4_to_array(&light_view_proj)),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let light_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Shadow Light Bind Group"),
                    layout: &shadow_light_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: light_buffer.as_entire_binding(),
                    }],
                });

                // Shadow render pass for 2D shadow map
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

                    // Render Nanite geometry to shadow map
                    if self.nanite_enabled {
                        if let Some(ref nanite) = self.nanite_renderer {
                            // Create light camera bind group for Nanite shadow pass
                            let light_matrix_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Nanite Light Matrix Buffer"),
                                contents: bytemuck::cast_slice(&matrix4_to_array(&light_view_proj)),
                                usage: wgpu::BufferUsages::UNIFORM,
                            });
                            let light_camera_bind_group = nanite.create_light_camera_bind_group(
                                &self.device,
                                &light_matrix_buffer,
                            );
                            // render_shadow sets pipeline and draws
                            nanite.render_shadow(&mut shadow_pass, &light_camera_bind_group);
                        }
                    }
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

        // ========== Nanite GPU Culling (Compute Pass) ==========
        if self.nanite_enabled {
            if let Some(ref mut nanite) = self.nanite_renderer {
                // Reset culling buffers for new frame
                nanite.begin_frame(&self.queue);

                // Get camera data for culling
                let mut camera = self.camera.borrow_mut();
                let fov = camera.fov;
                let view_proj = camera.view_projection_matrix().clone();
                let camera_pos = camera.position;
                drop(camera);

                nanite.update_culling(
                    &self.queue,
                    &view_proj,
                    [camera_pos.x, camera_pos.y, camera_pos.z],
                    self.height as f32,
                    fov,
                );

                // Two-phase culling:
                // 1. Frustum cull → frustum_clusters_buffer
                nanite.cull_frustum(&mut encoder);
                // 2. Occlusion cull (using previous frame's HZB) → visible_clusters_buffer
                nanite.cull_occlusion(&mut encoder);
            }
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

            // Render sky first (at far depth, behind everything)
            // Skip sky for flat mode (8) and wireframe mode (9)
            if self.render_mode != 8 && self.render_mode != 9 {
                // Procedural sky takes priority if enabled
                if self.procedural_sky.enabled() {
                    self.procedural_sky.render(&mut render_pass);
                } else if self.skybox_enabled {
                    self.skybox_pass.render(&mut render_pass, &self.skybox_bind_group);
                }
            }

            // Render PBR meshes (use wireframe pipeline if enabled and available)
            let pipeline_to_use = if self.wireframe_enabled {
                self.material.wireframe_pipeline().or_else(|| self.material.pipeline())
            } else {
                self.material.pipeline()
            };

            if let Some(pipeline) = pipeline_to_use {
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

                // Transform gizmo (index stored in gizmo_line_object)
                if self.gizmo_enabled {
                    if let Some(ref line_obj) = self.gizmo_line_object {
                        render_pass.set_bind_group(1, &line_obj.model_bind_group, &[]);
                        render_pass.set_vertex_buffer(0, line_obj.vertex_buffer.slice(..));
                        render_pass.draw(0..line_obj.vertex_count, 0..1);
                    }
                }
            }
        }

        // ========== Nanite Visibility Pass ==========
        // Render cluster/triangle IDs to visibility buffer
        if self.nanite_enabled {
            if let Some(ref nanite) = self.nanite_renderer {
                if nanite.cluster_count() > 0 {
                    let mut visibility_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Nanite Visibility Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: nanite.visibility_view(),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load, // Keep existing depth
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    if let Some(ref camera_bg) = self.nanite_camera_bind_group {
                        nanite.render_visibility(&mut visibility_pass, camera_bg);
                    }
                }
            }
        }

        // ========== Nanite HZB Build (Compute Pass) ==========
        // Build hierarchical Z-buffer from depth for next frame's occlusion culling
        if self.nanite_enabled {
            if let Some(ref nanite) = self.nanite_renderer {
                if nanite.cluster_count() > 0 {
                    nanite.build_hzb(&mut encoder, &self.queue, self.width, self.height);
                }
            }
        }

        // ========== Nanite Material Pass ==========
        // Shade pixels using visibility buffer data
        if self.nanite_enabled {
            if let Some(ref nanite) = self.nanite_renderer {
                if nanite.cluster_count() > 0 {
                    let mut material_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Nanite Material Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.scene_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load, // Keep existing scene content
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None, // Reading depth via texture
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    if let Some(ref camera_bg) = self.nanite_camera_bind_group {
                        nanite.render_material(&mut material_pass, camera_bg);
                    }
                }
            }
        }

        // ========== Particle System Update (Compute Pass) ==========
        // Calculate camera right/up vectors for billboarding from view matrix
        let (camera_right, camera_up, cam_near, cam_far) = {
            let mut camera = self.camera.borrow_mut();
            let view_matrix = camera.view_matrix();
            let view_elems = &view_matrix.elements;
            // Camera right is the first row of the view matrix
            let right = [view_elems[0], view_elems[4], view_elems[8]];
            // Camera up is the second row of the view matrix
            let up = [view_elems[1], view_elems[5], view_elems[9]];
            (right, up, camera.near, camera.far)
        };

        for particle_system in &mut self.particle_systems {
            // Update render params with camera vectors each frame
            particle_system.update_render_params(
                &self.queue,
                self.width,
                self.height,
                cam_near,
                cam_far,
                camera_right,
                camera_up,
            );
            particle_system.update(&mut encoder, delta, &self.queue);
        }

        // ========== Particle System Rendering ==========
        // Render particles after main geometry but before post-processing
        // Separate pass for each blend mode
        if !self.particle_systems.is_empty() {
            // Copy depth texture to depth_copy for soft particle sampling
            // (Can't read and write same texture in a render pass)
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.depth_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.depth_copy_texture,
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
            // Collect systems by blend mode
            let additive_systems: Vec<_> = self.particle_systems.iter()
                .filter(|ps| ps.visible && ps.blend_mode() == crate::particles::ParticleBlendMode::Additive)
                .collect();
            let alpha_systems: Vec<_> = self.particle_systems.iter()
                .filter(|ps| ps.visible && ps.blend_mode() == crate::particles::ParticleBlendMode::AlphaBlend)
                .collect();

            // Render alpha-blended particles first (they need proper depth order)
            if !alpha_systems.is_empty() {
                let mut particle_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Alpha Particle Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                for ps in alpha_systems {
                    ps.render(&mut particle_pass, &self.camera_bind_group);
                }
            }

            // Render additive particles (order-independent)
            if !additive_systems.is_empty() {
                let mut particle_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Additive Particle Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                for ps in additive_systems {
                    ps.render(&mut particle_pass, &self.camera_bind_group);
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

        // Apply AO (SSAO or GTAO - only one should be enabled at a time)
        // GTAO is more physically accurate but more expensive
        // Only apply in Lit mode (render_mode == 0)
        if self.render_mode == 0 {
            if self.gtao.enabled() {
                self.gtao.render_with_depth(&mut encoder, &self.depth_view, &self.bloom_input_view, &self.scene_view, &self.device, &self.queue);
            } else {
                self.ssao.render_with_depth(&mut encoder, &self.depth_view, &self.bloom_input_view, &self.scene_view, &self.device, &self.queue);
            }
        }

        // Apply Lumen GI if enabled (adds global illumination to scene)
        // Only apply in Lit mode (render_mode == 0)
        if self.lumen.enabled() && self.render_mode == 0 {
            // Get probe bind group for SH irradiance fallback when SSGI rays miss
            let probe_bind_group = self.probe_volume.bind_group();
            self.lumen.render_with_depth(&mut encoder, &self.depth_view, &self.scene_view, &self.bloom_input_view, probe_bind_group, &self.device, &self.queue);
            // Copy result back to scene_view for next passes
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.bloom_input_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.scene_texture,
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
        }

        // Apply volumetric fog if enabled (adds fog and god rays to scene)
        // Only apply in Lit mode (render_mode == 0)
        if self.volumetric_fog.enabled() && self.render_mode == 0 {
            // Set shadow matrix for volumetric god rays
            let shadow_resolution = self.shadow_config.resolution as f32;
            self.volumetric_fog.set_shadow_matrix(self.light_view_proj_matrix, shadow_resolution);
            // Render fog: reads from scene_view + shadow_map, writes to bloom_input_view
            self.volumetric_fog.render(&mut encoder, &self.depth_view, &self.shadow_map_view, &self.scene_view, &self.bloom_input_view, &self.device, &self.queue);
            // Copy result back to scene_view for next passes
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.bloom_input_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.scene_texture,
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
        }

        // Apply bloom post-processing (reads from bloom_input, additively blends to scene_view)
        // Only apply in Lit mode (render_mode == 0)
        if self.render_mode == 0 {
            self.bloom.render(&mut encoder, &self.bloom_input_view, &self.scene_view, &self.device, &self.queue);
        }

        // Apply SSR if enabled (scene + depth -> bloom_input)
        // Only apply in Lit mode (render_mode == 0)
        let post_ssr_input = if self.ssr.enabled() && self.render_mode == 0 {
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

        // Update auto-exposure if enabled (analyzes HDR scene, calculates exposure)
        if self.auto_exposure.enabled() {
            let exposure = self.auto_exposure.update(
                &mut encoder,
                tonemapping_input,
                delta,
                &self.device,
                &self.queue,
            );
            // Apply auto-exposure to tonemapping (convert from EV to linear)
            self.tonemapping.set_exposure(2.0_f32.powf(exposure));
        }

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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate depth copy texture for particle soft-fade
            self.depth_copy_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Copy Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.depth_copy_view = self.depth_copy_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate scene texture for post-processing (HDR format)
            self.scene_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Scene Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.hdr_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.scene_view = self.scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate bloom input texture (HDR format)
            self.bloom_input_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Bloom Input Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.hdr_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
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
            self.gtao.resize(width, height, &self.device);
            self.lumen.resize(width, height, &self.device);
            self.ssr.resize(width, height, &self.device);
            self.vignette.resize(width, height, &self.device);
            self.dof.resize(width, height, &self.device);
            self.motion_blur.resize(width, height);
            self.outline.resize(width, height);
            self.volumetric_fog.resize(width, height, &self.device);

            // Resize Nanite visibility buffer and HZB
            if let Some(ref mut nanite) = self.nanite_renderer {
                nanite.resize(&self.device, width, height);
                // Recreate bind groups that depend on depth texture
                nanite.create_material_bind_group(&self.device, &self.depth_view);
                nanite.create_hzb_build_bind_groups(&self.device, &self.depth_view);
            }

            // Update camera aspect ratio
            let mut camera = self.camera.borrow_mut();
            camera.set_aspect(width as f32 / height as f32);
            let pos = camera.position;
            let camera_uniform = PbrCameraUniform {
                view_proj: matrix4_to_array(camera.view_projection_matrix()),
                position: [pos.x, pos.y, pos.z],
                render_mode: self.render_mode,
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
                ibl_settings: [
                    self.ibl_diffuse_intensity,
                    self.ibl_specular_intensity,
                    if self.procedural_sky.enabled() { 1.0 } else { 0.0 },
                    0.0,
                ],
                light0_pos: [self.lights[0][0], self.lights[0][1], self.lights[0][2], self.lights[0][3]],
                light0_color: [self.lights[0][4], self.lights[0][5], self.lights[0][6], self.lights[0][7]],
                light1_pos: [self.lights[1][0], self.lights[1][1], self.lights[1][2], self.lights[1][3]],
                light1_color: [self.lights[1][4], self.lights[1][5], self.lights[1][6], self.lights[1][7]],
                light2_pos: [self.lights[2][0], self.lights[2][1], self.lights[2][2], self.lights[2][3]],
                light2_color: [self.lights[2][4], self.lights[2][5], self.lights[2][6], self.lights[2][7]],
                light3_pos: [self.lights[3][0], self.lights[3][1], self.lights[3][2], self.lights[3][3]],
                light3_color: [self.lights[3][4], self.lights[3][5], self.lights[3][6], self.lights[3][7]],
                detail_settings: [
                    if self.detail_enabled { 1.0 } else { 0.0 },
                    self.detail_scale,
                    self.detail_intensity,
                    self.detail_max_distance,
                ],
                detail_albedo_settings: [
                    if self.detail_albedo_enabled { 1.0 } else { 0.0 },
                    self.detail_albedo_scale,
                    self.detail_albedo_intensity,
                    self.detail_albedo_blend_mode as f32,
                ],
                // Rect lights: disabled by default (can be enabled via API)
                rectlight0_pos: [0.0, 0.0, 0.0, 0.0],
                rectlight0_dir_width: [0.0, 0.0, -1.0, 1.0],
                rectlight0_tan_height: [1.0, 0.0, 0.0, 1.0],
                rectlight0_color: [1.0, 1.0, 1.0, 10.0],
                rectlight1_pos: [0.0, 0.0, 0.0, 0.0],
                rectlight1_dir_width: [0.0, 0.0, -1.0, 1.0],
                rectlight1_tan_height: [1.0, 0.0, 0.0, 1.0],
                rectlight1_color: [1.0, 1.0, 1.0, 10.0],
                // Capsule lights: disabled by default
                capsule0_start: [0.0, 0.0, 0.0, 0.0],
                capsule0_end_radius: [1.0, 0.0, 0.0, 0.05],
                capsule0_color: [1.0, 1.0, 1.0, 10.0],
                capsule1_start: [0.0, 0.0, 0.0, 0.0],
                capsule1_end_radius: [1.0, 0.0, 0.0, 0.05],
                capsule1_color: [1.0, 1.0, 1.0, 10.0],
                // Disk lights: disabled by default
                disk0_pos: [0.0, 0.0, 0.0, 0.0],
                disk0_dir_radius: [0.0, -1.0, 0.0, 0.5],
                disk0_color: [1.0, 1.0, 1.0, 10.0],
                disk1_pos: [0.0, 0.0, 0.0, 0.0],
                disk1_dir_radius: [0.0, -1.0, 0.0, 0.5],
                disk1_color: [1.0, 1.0, 1.0, 10.0],
                // Sphere lights: disabled by default
                sphere0_pos: [0.0, 0.0, 0.0, 0.0],
                sphere0_radius_range: [0.1, 20.0, 0.0, 0.0],
                sphere0_color: [1.0, 1.0, 1.0, 10.0],
                sphere1_pos: [0.0, 0.0, 0.0, 0.0],
                sphere1_radius_range: [0.1, 20.0, 0.0, 0.0],
                sphere1_color: [1.0, 1.0, 1.0, 10.0],
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

    /// Handle mouse drag for orbit rotation (Alt + left button).
    #[wasm_bindgen]
    pub fn on_mouse_orbit(&self, delta_x: f32, delta_y: f32) {
        let mut controls = self.controls.borrow_mut();
        controls.rotate_by_pixels(delta_x, delta_y, self.height as f32);
    }

    /// Handle mouse drag for freelook (right button).
    /// Rotates the camera view direction without moving position.
    #[wasm_bindgen]
    pub fn on_mouse_freelook(&self, delta_x: f32, delta_y: f32) {
        let mut controls = self.controls.borrow_mut();
        let mut camera = self.camera.borrow_mut();

        // Sensitivity for freelook
        let sensitivity = 0.003;

        // Get current forward direction
        let diff = controls.target - camera.position;
        let forward = diff.normalized();

        // Calculate current yaw and pitch
        let mut yaw = forward.z.atan2(forward.x);
        let mut pitch = forward.y.asin();

        // Apply deltas
        yaw += delta_x * sensitivity;
        pitch -= delta_y * sensitivity;

        // Clamp pitch to prevent flipping
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
        pitch = pitch.clamp(-max_pitch, max_pitch);

        // Calculate new forward direction
        let new_forward = Vector3::new(
            pitch.cos() * yaw.cos(),
            pitch.sin(),
            pitch.cos() * yaw.sin(),
        );

        // Update target to be in front of camera
        let distance = diff.length();
        controls.target = camera.position + new_forward * distance;

        // Update camera to look at new target
        camera.set_target(controls.target);
    }

    /// Handle mouse drag for panning (middle button).
    #[wasm_bindgen]
    pub fn on_mouse_pan(&self, delta_x: f32, delta_y: f32) {
        let mut controls = self.controls.borrow_mut();
        let camera = self.camera.borrow();
        controls.pan(delta_x, delta_y, &camera);
    }

    /// Handle mouse drag for horizontal glide (left button).
    /// LMB drag = forward/back movement + yaw turning.
    #[wasm_bindgen]
    pub fn on_mouse_glide(&self, delta_x: f32, delta_y: f32) {
        let mut controls = self.controls.borrow_mut();
        let mut camera = self.camera.borrow_mut();

        // Get current camera yaw from forward direction
        let forward_3d = controls.target - camera.position;
        let distance = forward_3d.length();
        let yaw = forward_3d.z.atan2(forward_3d.x);

        // Yaw turning from horizontal mouse movement
        let mouse_sensitivity = 0.005;
        let yaw_delta = delta_x * mouse_sensitivity;
        let new_yaw = yaw + yaw_delta;

        // Forward movement from vertical mouse movement, rotated by current yaw
        let move_speed = distance * 0.001;
        let forward_amount = -delta_y * move_speed;
        let movement = Vector3::new(
            forward_amount * new_yaw.cos(),
            0.0,
            forward_amount * new_yaw.sin(),
        );

        // Move camera position
        let new_pos = camera.position + movement;
        camera.set_position(new_pos);

        // Update target: maintain distance, apply new yaw
        let new_forward = Vector3::new(
            distance * new_yaw.cos(),
            forward_3d.y, // Keep same pitch
            distance * new_yaw.sin(),
        );
        controls.target = new_pos + new_forward;
        camera.set_target(controls.target);
    }

    /// Handle mouse wheel for zoom.
    #[wasm_bindgen]
    pub fn on_mouse_wheel(&self, delta: f32) {
        let mut controls = self.controls.borrow_mut();
        controls.zoom_by_wheel(delta);
    }

    /// Handle keyboard movement (WASD + QE).
    /// forward: W/S, right: A/D, up: Q/E
    #[wasm_bindgen]
    pub fn on_keyboard_move(&self, forward: f32, right: f32, up: f32) {
        let mut controls = self.controls.borrow_mut();
        let camera = self.camera.borrow();
        controls.translate(forward, right, up, &camera);
    }

    /// Focus camera on origin (F key).
    #[wasm_bindgen]
    pub fn focus_on_origin(&self) {
        let mut controls = self.controls.borrow_mut();
        let mut camera = self.camera.borrow_mut();

        // Reset target to origin
        controls.target = Vector3::ZERO;

        // Position camera at a good viewing distance
        let distance = 5.0;
        camera.set_position(Vector3::new(distance, distance * 0.7, distance));
        camera.set_target(Vector3::ZERO);

        // Reset controls to match camera
        controls.reset_to_camera(&camera);
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

            // Build vertex data with barycentric coordinates for wireframe rendering
            // We need to expand indexed geometry to give each triangle unique barycentric coords
            // Vertex format: position(3) + normal(3) + uv(2) + barycentric(3) = 11 floats
            let bary_coords = [
                [1.0f32, 0.0, 0.0], // First vertex of triangle
                [0.0f32, 1.0, 0.0], // Second vertex
                [0.0f32, 0.0, 1.0], // Third vertex
            ];

            let triangle_count = geometry.indices.len() / 3;
            let mut vertex_data = Vec::with_capacity(triangle_count * 3 * 11);
            let mut new_indices: Vec<u32> = Vec::with_capacity(triangle_count * 3);

            for tri in 0..triangle_count {
                for v in 0..3 {
                    let idx = geometry.indices[tri * 3 + v] as usize;
                    let pos = geometry.positions[idx];
                    let normal = geometry.normals.get(idx).copied().unwrap_or([0.0, 1.0, 0.0]);
                    let uv = geometry.uvs.get(idx).copied().unwrap_or([0.0, 0.0]);
                    let bary = bary_coords[v];

                    vertex_data.extend_from_slice(&pos);
                    vertex_data.extend_from_slice(&normal);
                    vertex_data.extend_from_slice(&uv);
                    vertex_data.extend_from_slice(&bary);

                    new_indices.push((tri * 3 + v) as u32);
                }
            }

            let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Mesh {} Vertex Buffer", loaded_mesh.name)),
                contents: bytemuck::cast_slice(&vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Mesh {} Index Buffer", loaded_mesh.name)),
                contents: bytemuck::cast_slice(&new_indices),
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
                clear_coat: 0.0,
                clear_coat_roughness: 0.03,
                sheen: 0.0,
                use_albedo_map: if has_albedo { 1.0 } else { 0.0 },
                use_normal_map: if has_normal { 1.0 } else { 0.0 },
                use_metallic_roughness_map: if has_mr { 1.0 } else { 0.0 },
                _padding1: [0.0; 3],
                sheen_color: [1.0, 1.0, 1.0],
                _padding2: 0.0,
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

            // Create combined texture + shadow + env + BRDF + irradiance + detail bind group
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
                    &self.shadow_cube_map_view,
                    &self.shadow_cube_sampler,
                    &self.prefiltered_env_view,  // Use prefiltered env map for IBL
                    &self.skybox_sampler,
                    self.brdf_lut.view(),
                    self.brdf_lut.sampler(),
                    &self.irradiance_view,
                    &self.skybox_sampler,
                    self.detail_normal_map.view(),
                    self.detail_normal_map.sampler(),
                    self.detail_albedo_map.view(),
                    self.detail_albedo_map.sampler(),
                )
                .ok_or_else(|| JsValue::from_str("Failed to create texture bind group"))?;

            // Get the world transform for this mesh, or use identity
            let world_transform = mesh_transforms[mesh_idx].clone().unwrap_or_else(Matrix4::identity);

            let mesh = Self::create_mesh_with_transform(
                &self.device,
                &self.material,
                vertex_buffer,
                index_buffer,
                new_indices.len() as u32,
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

    /// Set tonemapping exposure (manual).
    /// Note: When auto-exposure is enabled, this value is overridden.
    #[wasm_bindgen]
    pub fn set_exposure(&mut self, exposure: f32) {
        self.tonemapping.set_exposure(exposure);
    }

    /// Set tonemapping mode.
    /// 0=Linear, 1=Reinhard, 2=ReinhardLum, 3=ACES, 4=Uncharted2, 5=AgX
    #[wasm_bindgen]
    pub fn set_tonemapping_mode(&mut self, mode: u32) {
        use crate::prelude::TonemappingMode;
        self.tonemapping.set_mode(TonemappingMode::from_u32(mode));
    }

    // ========== Auto-Exposure ==========

    /// Enable or disable auto-exposure (eye adaptation).
    #[wasm_bindgen]
    pub fn set_auto_exposure_enabled(&mut self, enabled: bool) {
        self.auto_exposure.set_enabled(enabled);
    }

    /// Check if auto-exposure is enabled.
    #[wasm_bindgen]
    pub fn is_auto_exposure_enabled(&self) -> bool {
        self.auto_exposure.enabled()
    }

    /// Set auto-exposure minimum EV (prevents over-darkening).
    #[wasm_bindgen]
    pub fn set_auto_exposure_min(&mut self, min_ev: f32) {
        self.auto_exposure.set_min_exposure(min_ev);
    }

    /// Set auto-exposure maximum EV (prevents over-brightening).
    #[wasm_bindgen]
    pub fn set_auto_exposure_max(&mut self, max_ev: f32) {
        self.auto_exposure.set_max_exposure(max_ev);
    }

    /// Set auto-exposure adaptation speed (dark to bright).
    #[wasm_bindgen]
    pub fn set_auto_exposure_speed_up(&mut self, speed: f32) {
        self.auto_exposure.set_speed_up(speed);
    }

    /// Set auto-exposure adaptation speed (bright to dark).
    #[wasm_bindgen]
    pub fn set_auto_exposure_speed_down(&mut self, speed: f32) {
        self.auto_exposure.set_speed_down(speed);
    }

    /// Set auto-exposure compensation (manual adjustment on top of auto).
    #[wasm_bindgen]
    pub fn set_auto_exposure_compensation(&mut self, ev: f32) {
        self.auto_exposure.set_exposure_compensation(ev);
    }

    /// Get current auto-exposure value (in EV).
    #[wasm_bindgen]
    pub fn get_auto_exposure_value(&self) -> f32 {
        self.auto_exposure.current_exposure()
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

    /// Set render mode (0=Lit, 1=Unlit, 2=Normals, 3=Depth, 4=Metallic, 5=Roughness, 6=AO, 7=UVs, 8=Flat, 9=Wireframe).
    #[wasm_bindgen]
    pub fn set_render_mode(&mut self, mode: u32) {
        self.render_mode = mode.min(9); // Clamp to valid range
    }

    /// Get current render mode.
    #[wasm_bindgen]
    pub fn get_render_mode(&self) -> u32 {
        self.render_mode
    }

    /// Enable or disable wireframe rendering.
    #[wasm_bindgen]
    pub fn set_wireframe_enabled(&mut self, enabled: bool) {
        self.wireframe_enabled = enabled;
    }

    /// Check if wireframe rendering is enabled.
    #[wasm_bindgen]
    pub fn is_wireframe_enabled(&self) -> bool {
        self.wireframe_enabled
    }

    /// Check if wireframe rendering is supported on this device.
    #[wasm_bindgen]
    pub fn is_wireframe_supported(&self) -> bool {
        self.wireframe_supported
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

    /// Set SSAO bias to prevent self-occlusion artifacts.
    #[wasm_bindgen]
    pub fn set_ssao_bias(&mut self, bias: f32) {
        self.ssao.set_bias(bias);
    }

    // ==================== GTAO (Ground Truth Ambient Occlusion) ====================

    /// Enable or disable GTAO (more accurate than SSAO but more expensive).
    /// Note: GTAO and SSAO are mutually exclusive - enabling one should disable the other.
    #[wasm_bindgen]
    pub fn set_gtao_enabled(&mut self, enabled: bool) {
        self.gtao.set_enabled(enabled);
        // When enabling GTAO, disable SSAO and vice versa
        if enabled {
            self.ssao.set_enabled(false);
        }
    }

    /// Check if GTAO is enabled.
    #[wasm_bindgen]
    pub fn is_gtao_enabled(&self) -> bool {
        self.gtao.enabled()
    }

    /// Set GTAO radius (0.1 - 2.0 recommended).
    #[wasm_bindgen]
    pub fn set_gtao_radius(&mut self, radius: f32) {
        self.gtao.set_radius(radius);
    }

    /// Set GTAO intensity (0.5 - 3.0 recommended).
    #[wasm_bindgen]
    pub fn set_gtao_intensity(&mut self, intensity: f32) {
        self.gtao.set_intensity(intensity);
    }

    /// Set GTAO power/contrast (0.5 - 4.0, default 1.5).
    #[wasm_bindgen]
    pub fn set_gtao_power(&mut self, power: f32) {
        self.gtao.set_power(power);
    }

    /// Set GTAO quality (0=Low, 1=Medium, 2=High, 3=Ultra).
    #[wasm_bindgen]
    pub fn set_gtao_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => GtaoQuality::Low,
            1 => GtaoQuality::Medium,
            2 => GtaoQuality::High,
            _ => GtaoQuality::Ultra,
        };
        self.gtao.set_quality(q);
    }

    /// Set GTAO falloff start (0.0 - 1.0, fraction of radius where falloff begins).
    #[wasm_bindgen]
    pub fn set_gtao_falloff(&mut self, falloff: f32) {
        self.gtao.set_falloff_start(falloff);
    }

    /// Set GTAO thin occluder compensation (0.0 - 1.0).
    #[wasm_bindgen]
    pub fn set_gtao_thin_occluder(&mut self, comp: f32) {
        self.gtao.set_thin_occluder_compensation(comp);
    }

    // ==================== Lumen GI (Global Illumination) ====================

    /// Enable or disable Lumen GI (screen-space global illumination).
    #[wasm_bindgen]
    pub fn set_lumen_enabled(&mut self, enabled: bool) {
        self.lumen.set_enabled(enabled);
    }

    /// Check if Lumen GI is enabled.
    #[wasm_bindgen]
    pub fn is_lumen_enabled(&self) -> bool {
        self.lumen.enabled()
    }

    /// Set Lumen GI quality (0=Low, 1=Medium, 2=High, 3=Ultra).
    #[wasm_bindgen]
    pub fn set_lumen_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => LumenQuality::Low,
            1 => LumenQuality::Medium,
            2 => LumenQuality::High,
            _ => LumenQuality::Ultra,
        };
        self.lumen.set_quality(q);
    }

    /// Set Lumen GI intensity (0.0 - 3.0 recommended, default 1.0).
    #[wasm_bindgen]
    pub fn set_lumen_intensity(&mut self, intensity: f32) {
        self.lumen.set_intensity(intensity);
    }

    /// Set Lumen GI max trace distance (1.0 - 20.0 recommended, default 5.0).
    #[wasm_bindgen]
    pub fn set_lumen_max_distance(&mut self, distance: f32) {
        self.lumen.set_max_distance(distance);
    }

    /// Set Lumen GI thickness for depth testing (0.01 - 0.5 recommended, default 0.1).
    #[wasm_bindgen]
    pub fn set_lumen_thickness(&mut self, thickness: f32) {
        self.lumen.set_thickness(thickness);
    }

    // ==================== Volumetric Fog ====================

    /// Enable or disable volumetric fog.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_enabled(&mut self, enabled: bool) {
        self.volumetric_fog.set_enabled(enabled);
    }

    /// Check if volumetric fog is enabled.
    #[wasm_bindgen]
    pub fn is_volumetric_fog_enabled(&self) -> bool {
        self.volumetric_fog.enabled()
    }

    /// Set volumetric fog quality (0=Low, 1=Medium, 2=High, 3=Ultra).
    #[wasm_bindgen]
    pub fn set_volumetric_fog_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => VolumetricFogQuality::Low,
            1 => VolumetricFogQuality::Medium,
            2 => VolumetricFogQuality::High,
            _ => VolumetricFogQuality::Ultra,
        };
        self.volumetric_fog.set_quality(q);
    }

    /// Set fog density (0.0 - 1.0).
    #[wasm_bindgen]
    pub fn set_volumetric_fog_density(&mut self, density: f32) {
        self.volumetric_fog.set_density(density);
    }

    /// Set fog start distance.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_start(&mut self, distance: f32) {
        self.volumetric_fog.set_start_distance(distance);
    }

    /// Set fog end distance.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_end(&mut self, distance: f32) {
        self.volumetric_fog.set_end_distance(distance);
    }

    /// Set fog height falloff (0 = uniform, higher = ground fog).
    #[wasm_bindgen]
    pub fn set_volumetric_fog_height_falloff(&mut self, falloff: f32) {
        self.volumetric_fog.set_height_falloff(falloff);
    }

    /// Set fog base height.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_base_height(&mut self, height: f32) {
        self.volumetric_fog.set_base_height(height);
    }

    /// Set fog scattering coefficient.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_scattering(&mut self, scattering: f32) {
        self.volumetric_fog.set_scattering(scattering);
    }

    /// Set fog absorption coefficient.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_absorption(&mut self, absorption: f32) {
        self.volumetric_fog.set_absorption(absorption);
    }

    /// Set god ray intensity.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_god_ray_intensity(&mut self, intensity: f32) {
        self.volumetric_fog.set_god_ray_intensity(intensity);
    }

    /// Set god ray decay.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_god_ray_decay(&mut self, decay: f32) {
        self.volumetric_fog.set_god_ray_decay(decay);
    }

    /// Set fog color.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_color(&mut self, r: f32, g: f32, b: f32) {
        self.volumetric_fog.set_fog_color(r, g, b);
    }

    /// Enable or disable god rays.
    #[wasm_bindgen]
    pub fn set_volumetric_fog_god_rays_enabled(&mut self, enabled: bool) {
        self.volumetric_fog.set_god_rays_enabled(enabled);
    }

    /// Set IBL diffuse intensity (environment lighting).
    #[wasm_bindgen]
    pub fn set_ibl_diffuse_intensity(&mut self, intensity: f32) {
        self.ibl_diffuse_intensity = intensity;
        // TODO: Update shader uniform when IBL uniforms are implemented
    }

    /// Set IBL specular intensity (environment reflections).
    #[wasm_bindgen]
    pub fn set_ibl_specular_intensity(&mut self, intensity: f32) {
        self.ibl_specular_intensity = intensity;
    }

    /// Enable or disable detail mapping for micro-surface detail.
    #[wasm_bindgen]
    pub fn set_detail_enabled(&mut self, enabled: bool) {
        self.detail_enabled = enabled;
    }

    /// Set detail mapping UV scale (tiling frequency).
    /// Higher values = more tiling = finer detail.
    #[wasm_bindgen]
    pub fn set_detail_scale(&mut self, scale: f32) {
        self.detail_scale = scale.clamp(1.0, 50.0);
    }

    /// Set detail mapping intensity (blend strength).
    #[wasm_bindgen]
    pub fn set_detail_intensity(&mut self, intensity: f32) {
        self.detail_intensity = intensity.clamp(0.0, 1.0);
    }

    /// Set detail mapping maximum distance (fade out distance).
    #[wasm_bindgen]
    pub fn set_detail_max_distance(&mut self, distance: f32) {
        self.detail_max_distance = distance.clamp(1.0, 50.0);
    }

    /// Enable or disable detail albedo for micro-surface color variation.
    #[wasm_bindgen]
    pub fn set_detail_albedo_enabled(&mut self, enabled: bool) {
        self.detail_albedo_enabled = enabled;
    }

    /// Set detail albedo UV scale (tiling frequency).
    /// Higher values = more tiling = finer color variation.
    #[wasm_bindgen]
    pub fn set_detail_albedo_scale(&mut self, scale: f32) {
        self.detail_albedo_scale = scale.clamp(1.0, 50.0);
    }

    /// Set detail albedo intensity (blend strength).
    #[wasm_bindgen]
    pub fn set_detail_albedo_intensity(&mut self, intensity: f32) {
        self.detail_albedo_intensity = intensity.clamp(0.0, 1.0);
    }

    /// Set detail albedo blend mode.
    /// 0 = Overlay (default), 1 = Multiply, 2 = Soft Light
    #[wasm_bindgen]
    pub fn set_detail_albedo_blend_mode(&mut self, mode: u32) {
        self.detail_albedo_blend_mode = mode.min(2);
    }

    /// Set light position (index 0-3).
    #[wasm_bindgen]
    pub fn set_light_position(&mut self, index: usize, x: f32, y: f32, z: f32) {
        if index < 4 {
            self.lights[index][0] = x;
            self.lights[index][1] = y;
            self.lights[index][2] = z;
        }
    }

    /// Set light intensity (index 0-3).
    #[wasm_bindgen]
    pub fn set_light_intensity(&mut self, index: usize, intensity: f32) {
        if index < 4 {
            self.lights[index][3] = intensity;
        }
    }

    /// Set light color (index 0-3).
    #[wasm_bindgen]
    pub fn set_light_color(&mut self, index: usize, r: f32, g: f32, b: f32) {
        if index < 4 {
            self.lights[index][4] = r;
            self.lights[index][5] = g;
            self.lights[index][6] = b;
        }
    }

    /// Enable or disable a light (index 0-3).
    #[wasm_bindgen]
    pub fn set_light_enabled(&mut self, index: usize, enabled: bool) {
        if index < 4 {
            self.lights[index][7] = if enabled { 1.0 } else { 0.0 };
        }
    }

    /// Apply car studio lighting preset.
    #[wasm_bindgen]
    pub fn apply_car_studio_preset(&mut self) {
        self.lights = [
            [5.0, 8.0, 5.0, 15.0, 1.0, 0.98, 0.95, 1.0],     // Key light
            [-5.0, 6.0, 3.0, 10.0, 0.9, 0.95, 1.0, 1.0],     // Fill light
            [0.0, 4.0, -6.0, 8.0, 1.0, 1.0, 1.0, 1.0],       // Rim light
            [-3.0, 1.0, -3.0, 5.0, 0.8, 0.85, 0.9, 1.0],     // Ground bounce
        ];
    }

    /// Apply outdoor lighting preset.
    #[wasm_bindgen]
    pub fn apply_outdoor_preset(&mut self) {
        self.lights = [
            [10.0, 20.0, 10.0, 25.0, 1.0, 0.95, 0.85, 1.0],  // Sun
            [-5.0, 5.0, 5.0, 5.0, 0.7, 0.8, 1.0, 1.0],       // Sky fill
            [0.0, 2.0, 0.0, 3.0, 0.6, 0.5, 0.4, 1.0],        // Ground reflection
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],        // Disabled
        ];
    }

    /// Apply dramatic lighting preset.
    #[wasm_bindgen]
    pub fn apply_dramatic_preset(&mut self) {
        self.lights = [
            [3.0, 5.0, 0.0, 20.0, 1.0, 0.9, 0.8, 1.0],       // Strong side key
            [-8.0, 3.0, -2.0, 4.0, 0.4, 0.5, 0.7, 1.0],      // Subtle fill
            [-2.0, 6.0, -5.0, 12.0, 1.0, 1.0, 1.0, 1.0],     // Strong rim
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],        // Disabled
        ];
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

    /// Set DoF quality (0=Low, 1=Medium, 2=High, 3=Ultra).
    #[wasm_bindgen]
    pub fn set_dof_quality(&mut self, quality: u32) {
        let q = match quality {
            0 => DofQuality::Low,
            1 => DofQuality::Medium,
            2 => DofQuality::High,
            _ => DofQuality::Ultra,
        };
        self.dof.set_quality(q);
    }

    /// Set DoF bokeh highlight boost (0.0 - 2.0).
    #[wasm_bindgen]
    pub fn set_dof_highlight_boost(&mut self, boost: f32) {
        self.dof.set_highlight_boost(boost);
    }

    /// Set DoF bokeh highlight threshold (0.0 - 1.0).
    #[wasm_bindgen]
    pub fn set_dof_highlight_threshold(&mut self, threshold: f32) {
        self.dof.set_highlight_threshold(threshold);
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

    // ========== HDR Display ==========

    /// Check if the display supports HDR output (10-bit or 16-bit float).
    #[wasm_bindgen]
    pub fn has_hdr_display(&self) -> bool {
        self.hdr_display_available
    }

    /// Get the HDR display format as a string (e.g., "Rgba10a2Unorm", "Rgba16Float").
    #[wasm_bindgen]
    pub fn get_hdr_display_format(&self) -> Option<String> {
        self.hdr_display_format.map(|f| format!("{:?}", f))
    }

    /// Check if HDR output is currently enabled.
    #[wasm_bindgen]
    pub fn is_hdr_output_enabled(&self) -> bool {
        self.hdr_output_enabled
    }

    /// Enable or disable HDR output.
    /// When enabled, uses 10-bit or 16-bit float output with adjusted tonemapping.
    /// Returns false if HDR display is not available.
    #[wasm_bindgen]
    pub fn set_hdr_output_enabled(&mut self, enabled: bool) -> bool {
        if enabled && !self.hdr_display_available {
            return false;
        }

        if enabled == self.hdr_output_enabled {
            return true;
        }

        self.hdr_output_enabled = enabled;

        // Reconfigure surface with appropriate format
        let new_format = if enabled {
            self.hdr_display_format.unwrap_or(self.surface_format)
        } else {
            // Find sRGB format from capabilities
            let caps = self.surface.get_capabilities(&self.adapter);
            caps.formats
                .iter()
                .copied()
                .find(|f| f.is_srgb())
                .unwrap_or(caps.formats[0])
        };

        self.surface_format = new_format;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: new_format,
            width: self.width,
            height: self.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        self.surface.configure(&self.device, &config);

        // Set HDR output mode on tonemapping pass
        self.tonemapping.set_hdr_output(enabled);

        // Reinitialize post-tonemapping passes with new format
        self.tonemapping.init(&self.device, new_format);
        self.fxaa.init(&self.device, new_format, self.width, self.height);
        self.smaa.init(&self.device, new_format, self.width, self.height);
        self.taa.init(&self.device, new_format, self.width, self.height);
        self.vignette.init(&self.device, new_format, self.width, self.height);

        // Recreate tonemapped and vignette textures with new format
        self.tonemapped_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tonemapped Texture"),
            size: wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: new_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.tonemapped_view = self.tonemapped_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.vignette_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Vignette Texture"),
            size: wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: new_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.vignette_view = self.vignette_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Update DoF pipeline to match new vignette texture format
        self.dof.set_output_format(new_format, &self.device);

        // AgX works well for both HDR and SDR - no need to switch modes
        // AgX naturally desaturates bright lights, avoiding the ACES "notorious six" color clipping
        self.tonemapping.set_mode(TonemappingMode::AgX);

        true
    }

    // ========== Skybox ==========

    /// Enable or disable skybox.
    #[wasm_bindgen]
    pub fn set_skybox_enabled(&mut self, enabled: bool) {
        self.skybox_enabled = enabled;
    }

    /// Check if skybox is enabled.
    #[wasm_bindgen]
    pub fn is_skybox_enabled(&self) -> bool {
        self.skybox_enabled
    }

    /// Set skybox exposure (brightness multiplier).
    #[wasm_bindgen]
    pub fn set_skybox_exposure(&mut self, exposure: f32) {
        self.skybox_pass.set_exposure(exposure);
    }

    /// Set procedural sky colors (RGB, 0-255).
    /// Creates a new procedural sky cubemap with the given colors.
    #[wasm_bindgen]
    pub fn set_sky_colors(&mut self, sky_r: u8, sky_g: u8, sky_b: u8, horizon_r: u8, horizon_g: u8, horizon_b: u8, ground_r: u8, ground_g: u8, ground_b: u8) {
        // Create new procedural sky with custom colors
        self.skybox_texture = CubeTexture::procedural_sky(
            &self.device,
            &self.queue,
            64,
            [sky_r, sky_g, sky_b],
            [horizon_r, horizon_g, horizon_b],
            [ground_r, ground_g, ground_b],
        );

        // Update bind group with new texture
        self.skybox_bind_group = self.skybox_pass.create_texture_bind_group(
            &self.device,
            self.skybox_texture.view(),
            &self.skybox_sampler,
        );
    }

    /// Load an HDR or EXR environment map from bytes.
    /// Converts equirectangular panorama to cubemap and generates prefiltered mips for IBL.
    #[wasm_bindgen]
    pub fn set_skybox_hdr(&mut self, data: &[u8]) -> Result<(), JsValue> {
        use crate::loaders::{HdrImage, create_prefiltered_hdr_cubemap};
        use crate::ibl::generate_irradiance_cubemap;

        // Parse HDR/EXR image
        let hdr = HdrImage::from_bytes(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to load HDR: {}", e)))?;

        let face_size = 256u32; // Good balance of quality and performance
        let mip_levels = 6u32;

        // Create prefiltered cubemap for both skybox and IBL specular
        self.prefiltered_env_texture = create_prefiltered_hdr_cubemap(
            &self.device,
            &self.queue,
            &hdr,
            face_size,
            mip_levels,
        );

        self.prefiltered_env_view = self.prefiltered_env_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // Also update the skybox texture for rendering the background
        let skybox_faces = hdr.to_cubemap_faces_ldr(face_size, 1.0);
        self.skybox_texture = CubeTexture::from_faces_owned(&self.device, &self.queue, &skybox_faces, face_size);

        // Generate irradiance map from HDR for diffuse IBL
        self.irradiance_texture = generate_irradiance_cubemap(
            &self.device,
            &self.queue,
            &skybox_faces,
            face_size,
            32, // Irradiance map can be low-res since it's a low-frequency signal
        );
        self.irradiance_view = self.irradiance_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // Update skybox bind group
        self.skybox_bind_group = self.skybox_pass.create_texture_bind_group(
            &self.device,
            self.skybox_texture.view(),
            &self.skybox_sampler,
        );

        Ok(())
    }

    /// Load an HDR or EXR environment map from bytes with custom settings.
    #[wasm_bindgen]
    pub fn set_skybox_hdr_with_options(&mut self, data: &[u8], face_size: u32, exposure: f32) -> Result<(), JsValue> {
        use crate::loaders::{HdrImage, create_prefiltered_hdr_cubemap};
        use crate::ibl::generate_irradiance_cubemap;

        let hdr = HdrImage::from_bytes(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to load HDR: {}", e)))?;

        let mip_levels = (face_size as f32).log2().floor() as u32 + 1;
        let mip_levels = mip_levels.min(8).max(1);

        // Create prefiltered cubemap for IBL specular
        self.prefiltered_env_texture = create_prefiltered_hdr_cubemap(
            &self.device,
            &self.queue,
            &hdr,
            face_size,
            mip_levels,
        );

        self.prefiltered_env_view = self.prefiltered_env_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // Create skybox with custom exposure
        let skybox_faces = hdr.to_cubemap_faces_ldr(face_size, exposure);
        self.skybox_texture = CubeTexture::from_faces_owned(&self.device, &self.queue, &skybox_faces, face_size);

        // Generate irradiance map from HDR for diffuse IBL
        self.irradiance_texture = generate_irradiance_cubemap(
            &self.device,
            &self.queue,
            &skybox_faces,
            face_size,
            32, // Irradiance map can be low-res since it's a low-frequency signal
        );
        self.irradiance_view = self.irradiance_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        self.skybox_bind_group = self.skybox_pass.create_texture_bind_group(
            &self.device,
            self.skybox_texture.view(),
            &self.skybox_sampler,
        );

        Ok(())
    }

    // ========== Procedural Sky ==========

    /// Enable or disable procedural sky (replaces cubemap skybox when enabled).
    #[wasm_bindgen]
    pub fn set_procedural_sky_enabled(&mut self, enabled: bool) {
        self.procedural_sky.set_enabled(enabled);
    }

    /// Check if procedural sky is enabled.
    #[wasm_bindgen]
    pub fn is_procedural_sky_enabled(&self) -> bool {
        self.procedural_sky.enabled()
    }

    /// Set sun position from azimuth and elevation angles (in degrees).
    /// Azimuth: 0-360 (compass direction), Elevation: -90 to 90 (height above horizon).
    #[wasm_bindgen]
    pub fn set_procedural_sky_sun_position(&mut self, azimuth: f32, elevation: f32) {
        self.procedural_sky.set_sun_position(azimuth, elevation);
    }

    /// Set sun direction directly (will be normalized).
    #[wasm_bindgen]
    pub fn set_procedural_sky_sun_direction(&mut self, x: f32, y: f32, z: f32) {
        self.procedural_sky.set_sun_direction(x, y, z);
    }

    /// Set sun intensity.
    #[wasm_bindgen]
    pub fn set_procedural_sky_sun_intensity(&mut self, intensity: f32) {
        self.procedural_sky.set_sun_intensity(intensity);
    }

    /// Set Rayleigh scattering coefficient (controls blue sky intensity).
    #[wasm_bindgen]
    pub fn set_procedural_sky_rayleigh(&mut self, coefficient: f32) {
        self.procedural_sky.set_rayleigh(coefficient);
    }

    /// Set Mie scattering coefficient (controls haze/glow around sun).
    #[wasm_bindgen]
    pub fn set_procedural_sky_mie(&mut self, coefficient: f32) {
        self.procedural_sky.set_mie(coefficient);
    }

    /// Set Mie directional G (-1 to 1, controls sun glow concentration).
    #[wasm_bindgen]
    pub fn set_procedural_sky_mie_directional(&mut self, g: f32) {
        self.procedural_sky.set_mie_directional_g(g);
    }

    /// Set atmospheric turbidity (1-10, affects haziness).
    #[wasm_bindgen]
    pub fn set_procedural_sky_turbidity(&mut self, turbidity: f32) {
        self.procedural_sky.set_turbidity(turbidity);
    }

    /// Set sun disk angular size in degrees.
    #[wasm_bindgen]
    pub fn set_procedural_sky_sun_size(&mut self, size: f32) {
        self.procedural_sky.set_sun_disk_size(size);
    }

    /// Set sun disk brightness.
    #[wasm_bindgen]
    pub fn set_procedural_sky_sun_disk_intensity(&mut self, intensity: f32) {
        self.procedural_sky.set_sun_disk_intensity(intensity);
    }

    /// Set ground/horizon color (RGB 0-1).
    #[wasm_bindgen]
    pub fn set_procedural_sky_ground_color(&mut self, r: f32, g: f32, b: f32) {
        self.procedural_sky.set_ground_color(r, g, b);
    }

    /// Set procedural sky exposure.
    #[wasm_bindgen]
    pub fn set_procedural_sky_exposure(&mut self, exposure: f32) {
        self.procedural_sky.set_exposure(exposure);
    }

    /// Set procedural sky cloud movement speed.
    #[wasm_bindgen]
    pub fn set_procedural_sky_cloud_speed(&mut self, speed: f32) {
        self.procedural_sky.set_cloud_speed(speed);
    }

    /// Update environment maps (skybox and irradiance) from procedural sky.
    /// Call this to make PBR materials reflect the procedural sky.
    #[wasm_bindgen]
    pub fn update_procedural_sky_environment(&mut self) {
        use crate::ibl::generate_irradiance_cubemap;

        // Generate cubemap faces from procedural sky (64x64 is enough for IBL)
        let face_size = 64u32;
        let faces = self.procedural_sky.generate_cubemap_faces(face_size);

        // Update skybox texture for reflections
        self.skybox_texture = CubeTexture::from_faces_owned(&self.device, &self.queue, &faces, face_size);

        // Generate irradiance map for diffuse IBL
        self.irradiance_texture = generate_irradiance_cubemap(
            &self.device,
            &self.queue,
            &faces,
            face_size,
            32, // Irradiance is low frequency, 32x32 is fine
        );
        self.irradiance_view = self.irradiance_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // Update PBR bind groups to use new environment maps
        self.update_pbr_environment_bind_groups();
    }

    /// Internal method to update PBR bind groups with new environment maps.
    fn update_pbr_environment_bind_groups(&mut self) {
        // Recreate the skybox bind group with the new texture
        self.skybox_bind_group = self.skybox_pass.create_texture_bind_group(
            &self.device,
            self.skybox_texture.view(),
            &self.skybox_sampler,
        );
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

    /// Set PCF mode (0=None, 1=Hardware2x2, 2=Soft3x3, 3=Soft5x5, 4=PoissonDisk, 5=PCSS).
    #[wasm_bindgen]
    pub fn set_shadow_pcf_mode(&mut self, mode: u32) {
        let pcf_mode = match mode {
            0 => PCFMode::None,
            1 => PCFMode::Hardware2x2,
            2 => PCFMode::Soft3x3,
            3 => PCFMode::Soft5x5,
            4 => PCFMode::PoissonDisk,
            _ => PCFMode::PCSS,
        };
        self.shadow_config.pcf_mode = pcf_mode;
    }

    /// Get current PCF mode (0=None, 1=Hardware2x2, 2=Soft3x3, 3=Soft5x5, 4=PoissonDisk, 5=PCSS).
    #[wasm_bindgen]
    pub fn get_shadow_pcf_mode(&self) -> u32 {
        match self.shadow_config.pcf_mode {
            PCFMode::None => 0,
            PCFMode::Hardware2x2 => 1,
            PCFMode::Soft3x3 => 2,
            PCFMode::Soft5x5 => 3,
            PCFMode::PoissonDisk => 4,
            PCFMode::PCSS => 5,
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

    // ========== PCSS Settings ==========

    /// Set PCSS light size (0.1 - 10.0, default 0.5). Larger = softer shadows.
    #[wasm_bindgen]
    pub fn set_pcss_light_size(&mut self, light_size: f32) {
        self.shadow_config.pcss.light_size = light_size.clamp(0.1, 10.0);
    }

    /// Get current PCSS light size.
    #[wasm_bindgen]
    pub fn get_pcss_light_size(&self) -> f32 {
        self.shadow_config.pcss.light_size
    }

    /// Set PCSS max filter radius in texels (1.0 - 50.0, default 10.0).
    #[wasm_bindgen]
    pub fn set_pcss_max_filter_radius(&mut self, radius: f32) {
        self.shadow_config.pcss.max_filter_radius = radius.clamp(1.0, 50.0);
    }

    /// Get current PCSS max filter radius.
    #[wasm_bindgen]
    pub fn get_pcss_max_filter_radius(&self) -> f32 {
        self.shadow_config.pcss.max_filter_radius
    }

    // ========== Contact Shadow Settings ==========

    /// Enable or disable contact shadows.
    #[wasm_bindgen]
    pub fn set_contact_shadows_enabled(&mut self, enabled: bool) {
        self.shadow_config.contact.enabled = enabled;
    }

    /// Check if contact shadows are enabled.
    #[wasm_bindgen]
    pub fn get_contact_shadows_enabled(&self) -> bool {
        self.shadow_config.contact.enabled
    }

    /// Set contact shadow max distance (0.1 - 2.0, default 0.5).
    #[wasm_bindgen]
    pub fn set_contact_shadow_distance(&mut self, distance: f32) {
        self.shadow_config.contact.max_distance = distance.clamp(0.1, 2.0);
    }

    /// Get current contact shadow max distance.
    #[wasm_bindgen]
    pub fn get_contact_shadow_distance(&self) -> f32 {
        self.shadow_config.contact.max_distance
    }

    /// Set contact shadow thickness (0.01 - 0.5, default 0.05).
    #[wasm_bindgen]
    pub fn set_contact_shadow_thickness(&mut self, thickness: f32) {
        self.shadow_config.contact.thickness = thickness.clamp(0.01, 0.5);
    }

    /// Get current contact shadow thickness.
    #[wasm_bindgen]
    pub fn get_contact_shadow_thickness(&self) -> f32 {
        self.shadow_config.contact.thickness
    }

    /// Set contact shadow intensity (0.0 - 1.0, default 0.5).
    #[wasm_bindgen]
    pub fn set_contact_shadow_intensity(&mut self, intensity: f32) {
        self.shadow_config.contact.intensity = intensity.clamp(0.0, 1.0);
    }

    /// Get current contact shadow intensity.
    #[wasm_bindgen]
    pub fn get_contact_shadow_intensity(&self) -> f32 {
        self.shadow_config.contact.intensity
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

    /// Set the directional light intensity (0-1 for night, 1+ for day).
    #[wasm_bindgen]
    pub fn set_directional_light_intensity(&mut self, intensity: f32) {
        self.light_intensity = intensity.max(0.0);
    }

    /// Set the directional light color (RGB 0-1).
    #[wasm_bindgen]
    pub fn set_directional_light_color(&mut self, r: f32, g: f32, b: f32) {
        self.light_color = [r, g, b];
    }

    /// Set shadow light type (0=directional, 1=spot, 2=point).
    #[wasm_bindgen]
    pub fn set_shadow_light_type(&mut self, light_type: u32) {
        self.shadow_light_type = match light_type {
            0 => ShadowLightType::Directional,
            1 => ShadowLightType::Spot,
            2 => ShadowLightType::Point,
            _ => ShadowLightType::Directional,
        };
    }

    /// Set spot light position.
    #[wasm_bindgen]
    pub fn set_spot_position(&mut self, x: f32, y: f32, z: f32) {
        self.spot_position = [x, y, z];
    }

    /// Set spot light direction.
    #[wasm_bindgen]
    pub fn set_spot_direction(&mut self, x: f32, y: f32, z: f32) {
        self.spot_direction = [x, y, z];
    }

    /// Set spot light range.
    #[wasm_bindgen]
    pub fn set_spot_range(&mut self, range: f32) {
        self.spot_range = range;
    }

    /// Set spot light angles in degrees.
    #[wasm_bindgen]
    pub fn set_spot_angles(&mut self, inner_deg: f32, outer_deg: f32) {
        self.spot_inner_angle = inner_deg.to_radians();
        self.spot_outer_angle = outer_deg.to_radians();
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

    /// Add a floor plane at the given Y position.
    #[wasm_bindgen]
    pub fn add_floor(
        &mut self,
        y: f32,
        size: f32,
        r: f32,
        g: f32,
        b: f32,
        metallic: f32,
        roughness: f32,
    ) -> Result<(), JsValue> {
        let s = size / 2.0;
        let tile = size / 4.0; // UV tiling

        // Floor plane vertices: position (3) + normal (3) + uv (2)
        #[rustfmt::skip]
        let vertices: Vec<f32> = vec![
            // Floor quad facing up (Y+)
            -s, 0.0,  s,  0.0, 1.0, 0.0,  0.0, tile,
             s, 0.0,  s,  0.0, 1.0, 0.0,  tile, tile,
             s, 0.0, -s,  0.0, 1.0, 0.0,  tile, 0.0,
            -s, 0.0, -s,  0.0, 1.0, 0.0,  0.0, 0.0,
        ];

        #[rustfmt::skip]
        let indices: Vec<u32> = vec![
            0, 1, 2, 0, 2, 3,
        ];

        self.add_geometry(&vertices, &indices, [0.0, y, 0.0], [r, g, b, 1.0], metallic, roughness)
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
        // Expand indexed geometry with barycentric coordinates for wireframe
        // Input vertices: 8 floats per vertex (pos3 + normal3 + uv2)
        // Output vertices: 11 floats per vertex (pos3 + normal3 + uv2 + bary3)
        let bary_coords = [
            [1.0f32, 0.0, 0.0],
            [0.0f32, 1.0, 0.0],
            [0.0f32, 0.0, 1.0],
        ];

        let triangle_count = indices.len() / 3;
        let mut expanded_vertices: Vec<f32> = Vec::with_capacity(triangle_count * 3 * 11);
        let mut new_indices: Vec<u32> = Vec::with_capacity(triangle_count * 3);

        for tri in 0..triangle_count {
            for v in 0..3 {
                let idx = indices[tri * 3 + v] as usize;
                let base = idx * 8; // 8 floats per input vertex

                // Copy position (3), normal (3), uv (2)
                expanded_vertices.extend_from_slice(&vertices[base..base + 8]);
                // Add barycentric
                expanded_vertices.extend_from_slice(&bary_coords[v]);

                new_indices.push((tri * 3 + v) as u32);
            }
        }

        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shape Vertex Buffer"),
            contents: bytemuck::cast_slice(&expanded_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shape Index Buffer"),
            contents: bytemuck::cast_slice(&new_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mat_uniform = TexturedPbrMaterialUniform {
            base_color: color,
            metallic,
            roughness,
            ao: 1.0,
            clear_coat: 0.0,
            clear_coat_roughness: 0.03,
            sheen: 0.0,
            use_albedo_map: 0.0,
            use_normal_map: 0.0,
            use_metallic_roughness_map: 0.0,
            _padding1: [0.0; 3],
            sheen_color: [1.0, 1.0, 1.0],
            _padding2: 0.0,
        };

        // Create combined texture + shadow + env + BRDF + irradiance + detail bind group with default textures
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
                &self.shadow_cube_map_view,
                &self.shadow_cube_sampler,
                &self.prefiltered_env_view,  // Use prefiltered env map for IBL
                &self.skybox_sampler,
                self.brdf_lut.view(),
                self.brdf_lut.sampler(),
                &self.irradiance_view,
                &self.skybox_sampler,
                self.detail_normal_map.view(),
                self.detail_normal_map.sampler(),
                self.detail_albedo_map.view(),
                self.detail_albedo_map.sampler(),
            )
            .ok_or_else(|| JsValue::from_str("Failed to create texture bind group"))?;

        // Create transform matrix
        let world_transform = Matrix4::from_translation(&Vector3::new(position[0], position[1], position[2]));

        let mesh = Self::create_mesh_with_transform(
            &self.device,
            &self.material,
            vertex_buffer,
            index_buffer,
            new_indices.len() as u32,
            &world_transform,
            mat_uniform,
            texture_bind_group,
        )?;

        self.meshes.push(mesh);
        Ok(())
    }

    // ==================== Particle System API ====================

    /// Create a new particle system from a preset.
    /// Returns the particle system ID.
    /// Presets: "fire", "smoke", "sparks", "debris", "magic"
    #[wasm_bindgen]
    pub fn create_particle_system(&mut self, preset: &str) -> u32 {
        use crate::particles::{ParticlePreset, ParticleSystem};

        let preset_type = match preset.to_lowercase().as_str() {
            "fire" => ParticlePreset::Fire,
            "smoke" => ParticlePreset::Smoke,
            "sparks" => ParticlePreset::Sparks,
            "debris" => ParticlePreset::Debris,
            "magic" | "energy" => ParticlePreset::MagicEnergy,
            _ => ParticlePreset::Fire, // Default to fire
        };

        let mut ps = ParticleSystem::from_preset(preset_type);

        // Get camera bind group layout from material
        let camera_layout = self.material.camera_bind_group_layout()
            .expect("Camera bind group layout not initialized");

        // Initialize the particle system
        // Use depth_copy_view (not depth_view) because we can't read and write same texture
        ps.init(
            &self.device,
            &self.queue,
            self.hdr_format,
            camera_layout,
            self.white_texture.view(),
            &self.depth_copy_view,
            &self.particle_sampler,
        );

        // Add preset-specific forces for more realistic behavior
        match preset_type {
            ParticlePreset::Fire => {
                // Add turbulence for flickering flames
                ps.add_force(crate::particles::ForceType::Turbulence {
                    frequency: 3.0,
                    amplitude: 1.2,
                    octaves: 2,
                });
                // Slight upward thermal lift
                ps.add_force(crate::particles::ForceType::Directional {
                    direction: [0.0, 1.0, 0.0],
                    strength: 0.8,
                });
            }
            ParticlePreset::Smoke => {
                // Gentle turbulence for billowing
                ps.add_force(crate::particles::ForceType::Turbulence {
                    frequency: 1.0,
                    amplitude: 0.5,
                    octaves: 2,
                });
            }
            ParticlePreset::Sparks => {
                // Gravity for sparks
                ps.add_force(crate::particles::ForceType::Directional {
                    direction: [0.0, -1.0, 0.0],
                    strength: 4.0,
                });
            }
            ParticlePreset::Debris => {
                // Strong gravity for debris
                ps.add_force(crate::particles::ForceType::Directional {
                    direction: [0.0, -1.0, 0.0],
                    strength: 9.8,
                });
            }
            _ => {}
        }

        let id = self.particle_systems.len() as u32;
        self.particle_systems.push(ps);
        id
    }

    /// Set particle system position.
    #[wasm_bindgen]
    pub fn set_particle_position(&mut self, id: u32, x: f32, y: f32, z: f32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.set_position(x, y, z);
        }
    }

    /// Get number of particle systems.
    #[wasm_bindgen]
    pub fn particle_system_count(&self) -> u32 {
        self.particle_systems.len() as u32
    }

    /// Set particle system visibility.
    #[wasm_bindgen]
    pub fn set_particle_visible(&mut self, id: u32, visible: bool) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.visible = visible;
        }
    }

    /// Play a particle system.
    #[wasm_bindgen]
    pub fn play_particle_system(&mut self, id: u32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.play();
        }
    }

    /// Pause a particle system.
    #[wasm_bindgen]
    pub fn pause_particle_system(&mut self, id: u32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.pause();
        }
    }

    /// Stop and reset a particle system.
    #[wasm_bindgen]
    pub fn stop_particle_system(&mut self, id: u32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.stop(&self.queue);
        }
    }

    /// Emit a burst of particles.
    #[wasm_bindgen]
    pub fn particle_burst(&mut self, id: u32, count: u32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.burst(count);
        }
    }

    /// Add gravity force to a particle system.
    #[wasm_bindgen]
    pub fn add_particle_gravity(&mut self, id: u32, strength: f32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.add_force(crate::particles::ForceType::Directional {
                direction: [0.0, -1.0, 0.0],
                strength,
            });
        }
    }

    /// Add wind force to a particle system.
    #[wasm_bindgen]
    pub fn add_particle_wind(&mut self, id: u32, x: f32, y: f32, z: f32, strength: f32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            let len = (x * x + y * y + z * z).sqrt();
            let dir = if len > 0.0 { [x / len, y / len, z / len] } else { [1.0, 0.0, 0.0] };
            ps.add_force(crate::particles::ForceType::Directional {
                direction: dir,
                strength,
            });
        }
    }

    /// Add turbulence force to a particle system.
    #[wasm_bindgen]
    pub fn add_particle_turbulence(&mut self, id: u32, frequency: f32, amplitude: f32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.add_force(crate::particles::ForceType::Turbulence {
                frequency,
                amplitude,
                octaves: 2,
            });
        }
    }

    /// Add attractor force to a particle system.
    #[wasm_bindgen]
    pub fn add_particle_attractor(&mut self, id: u32, x: f32, y: f32, z: f32, strength: f32, radius: f32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.add_force(crate::particles::ForceType::Point {
                position: [x, y, z],
                strength,
                radius,
            });
        }
    }

    /// Clear all forces from a particle system.
    #[wasm_bindgen]
    pub fn clear_particle_forces(&mut self, id: u32) {
        if let Some(ps) = self.particle_systems.get_mut(id as usize) {
            ps.clear_forces();
        }
    }

    /// Remove a particle system.
    #[wasm_bindgen]
    pub fn remove_particle_system(&mut self, id: u32) {
        let idx = id as usize;
        if idx < self.particle_systems.len() {
            self.particle_systems.remove(idx);
        }
    }

    /// Remove all particle systems.
    #[wasm_bindgen]
    pub fn clear_particle_systems(&mut self) {
        self.particle_systems.clear();
    }

    // ========== Nanite Virtualized Geometry ==========

    /// Initialize the Nanite renderer if not already initialized.
    fn ensure_nanite_initialized(&mut self) {
        if self.nanite_renderer.is_none() {
            let config = NaniteConfig::default();
            let depth_format = wgpu::TextureFormat::Depth32Float;

            let mut nanite = NaniteRenderer::new(
                &self.device,
                config,
                self.hdr_format,
                depth_format,
                self.width,
                self.height,
            );

            // Create material bind group with depth texture
            nanite.create_material_bind_group(&self.device, &self.depth_view);

            // Create HZB build bind groups for occlusion culling
            nanite.create_hzb_build_bind_groups(&self.device, &self.depth_view);

            // Create camera bind group for Nanite (compatible layout)
            let camera_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nanite Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

            self.nanite_camera_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Nanite Camera Bind Group"),
                layout: &camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                }],
            }));

            // Create combined texture+shadow bind group for Nanite material pass
            // Group 3 bindings: 0=materials, 1=texture_array, 2=texture_sampler, 3=shadow_data, 4=shadow_map, 5=shadow_sampler, 6=shadow_cube, 7=shadow_cube_sampler
            if let Some(texture_view) = nanite.texture_array_view() {
                let texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Nanite Textures+Shadows Bind Group"),
                    layout: nanite.textures_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: nanite.material_buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(nanite.texture_sampler()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.shadow_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.shadow_map_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&self.shadow_cube_map_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Sampler(&self.shadow_cube_sampler),
                        },
                    ],
                });
                nanite.set_texture_bind_group(texture_bind_group);
            }

            self.nanite_renderer = Some(nanite);
        }
    }

    /// Enable or disable Nanite rendering.
    #[wasm_bindgen]
    pub fn set_nanite_enabled(&mut self, enabled: bool) {
        self.nanite_enabled = enabled;
    }

    /// Check if Nanite is enabled.
    #[wasm_bindgen]
    pub fn nanite_enabled(&self) -> bool {
        self.nanite_enabled
    }

    /// Load a mesh as Nanite geometry from vertex/index arrays.
    /// Returns the Nanite mesh ID for use with instance transforms.
    #[wasm_bindgen]
    pub fn load_nanite_mesh(&mut self, positions: &[f32], normals: &[f32], uvs: &[f32], indices: &[u32]) -> u32 {
        self.ensure_nanite_initialized();

        // Convert flat arrays to vertices
        let vertex_count = positions.len() / 3;
        let has_normals = normals.len() == vertex_count * 3;
        let has_uvs = uvs.len() == vertex_count * 2;

        let mut vertices: Vec<Vertex> = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let position = [
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2],
            ];
            let normal = if has_normals {
                [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]]
            } else {
                [0.0, 1.0, 0.0]
            };
            let uv = if has_uvs {
                [uvs[i * 2], uvs[i * 2 + 1]]
            } else {
                [0.0, 0.0]
            };
            vertices.push(Vertex {
                position,
                normal,
                uv,
                barycentric: [0.0, 0.0, 0.0], // Not used for Nanite
            });
        }

        // Build clusters
        let config = NaniteConfig::default();
        let result = build_clusters(&vertices, indices, config.triangles_per_cluster, 0);

        // Register with renderer
        if let Some(ref mut nanite) = self.nanite_renderer {
            let mesh_id = nanite.register_mesh(&self.device, &self.queue, result);
            return mesh_id as u32;
        }

        0
    }

    /// Set Nanite instance transform.
    #[wasm_bindgen]
    pub fn set_nanite_instance(&mut self, mesh_id: u32, m00: f32, m01: f32, m02: f32, m03: f32,
                               m10: f32, m11: f32, m12: f32, m13: f32,
                               m20: f32, m21: f32, m22: f32, m23: f32,
                               m30: f32, m31: f32, m32: f32, m33: f32) {
        if let Some(ref mut nanite) = self.nanite_renderer {
            let transform = Matrix4::new(
                m00, m01, m02, m03,
                m10, m11, m12, m13,
                m20, m21, m22, m23,
                m30, m31, m32, m33,
            );
            nanite.update_instances(&self.queue, &[(mesh_id as usize, transform)]);
        }
    }

    /// Get Nanite cluster count (for debugging).
    #[wasm_bindgen]
    pub fn nanite_cluster_count(&self) -> u32 {
        if let Some(ref nanite) = self.nanite_renderer {
            nanite.cluster_count()
        } else {
            0
        }
    }

    /// Add a demo cube rendered via Nanite.
    #[wasm_bindgen]
    pub fn add_nanite_demo_cube(&mut self) {
        self.ensure_nanite_initialized();

        // Simple cube with 8 vertices and 12 triangles (2 per face)
        let s = 0.5f32; // half-size

        // 8 corner vertices
        let positions: Vec<[f32; 3]> = vec![
            [-s, -s, -s], // 0: left-bottom-back
            [ s, -s, -s], // 1: right-bottom-back
            [ s,  s, -s], // 2: right-top-back
            [-s,  s, -s], // 3: left-top-back
            [-s, -s,  s], // 4: left-bottom-front
            [ s, -s,  s], // 5: right-bottom-front
            [ s,  s,  s], // 6: right-top-front
            [-s,  s,  s], // 7: left-top-front
        ];

        // Face normals for each vertex (approximation - use face normal)
        let normals: Vec<[f32; 3]> = vec![
            [-0.577, -0.577, -0.577],
            [ 0.577, -0.577, -0.577],
            [ 0.577,  0.577, -0.577],
            [-0.577,  0.577, -0.577],
            [-0.577, -0.577,  0.577],
            [ 0.577, -0.577,  0.577],
            [ 0.577,  0.577,  0.577],
            [-0.577,  0.577,  0.577],
        ];

        let uvs: Vec<[f32; 2]> = vec![
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ];

        // 12 triangles (2 per face), counter-clockwise winding when viewed from outside
        let indices: Vec<u32> = vec![
            // Front face (+Z)
            4, 5, 6,  4, 6, 7,
            // Back face (-Z)
            1, 0, 3,  1, 3, 2,
            // Right face (+X)
            5, 1, 2,  5, 2, 6,
            // Left face (-X)
            0, 4, 7,  0, 7, 3,
            // Top face (+Y)
            7, 6, 2,  7, 2, 3,
            // Bottom face (-Y)
            0, 1, 5,  0, 5, 4,
        ];

        // Convert to Vertex format
        let vertices: Vec<Vertex> = positions.iter().enumerate().map(|(i, &pos)| {
            Vertex {
                position: pos,
                normal: normals[i],
                uv: uvs[i % uvs.len()],
                barycentric: [0.0, 0.0, 0.0],
            }
        }).collect();

        // Build clusters (12 triangles = 1 cluster)
        let config = NaniteConfig::default();
        let result = build_clusters(&vertices, &indices, config.triangles_per_cluster, 0);

        // Register with Nanite renderer
        if let Some(ref mut nanite) = self.nanite_renderer {
            let mesh_id = nanite.register_mesh(&self.device, &self.queue, result);
            // Set identity transform (centered at origin)
            nanite.update_instances(&self.queue, &[(mesh_id, Matrix4::identity())]);
        }
    }

    /// Load a GLTF/GLB file as Nanite geometry.
    /// Returns the number of meshes loaded.
    #[wasm_bindgen]
    pub fn load_gltf_nanite(&mut self, data: &[u8]) -> Result<u32, JsValue> {
        self.ensure_nanite_initialized();

        // Parse GLTF
        let loader = GltfLoader::new();
        let scene = loader.load_from_bytes(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to load GLTF: {}", e)))?;

        // Helper to compute transform matrix from TRS using Matrix4::compose
        fn compute_node_transform(node: &crate::loaders::LoadedNode) -> Matrix4 {
            let t = node.translation;
            let r = node.rotation; // quaternion xyzw
            let s = node.scale;

            let position = Vector3::new(t[0], t[1], t[2]);
            let quaternion = Quaternion::new(r[0], r[1], r[2], r[3]);
            let scale = Vector3::new(s[0], s[1], s[2]);

            Matrix4::compose(&position, &quaternion, &scale)
        }

        // Compute world transforms for each mesh
        let mut mesh_transforms: Vec<(usize, Matrix4)> = Vec::new();

        fn process_node(
            node_idx: usize,
            parent_transform: &Matrix4,
            scene: &LoadedScene,
            mesh_transforms: &mut Vec<(usize, Matrix4)>,
            compute_transform: fn(&crate::loaders::LoadedNode) -> Matrix4,
        ) {
            if let Some(node) = scene.nodes.get(node_idx) {
                let local_transform = compute_transform(node);
                let world_transform = parent_transform.multiply(&local_transform);

                for &mesh_idx in &node.mesh_indices {
                    mesh_transforms.push((mesh_idx, world_transform.clone()));
                }

                for &child_idx in &node.children {
                    process_node(child_idx, &world_transform, scene, mesh_transforms, compute_transform);
                }
            }
        }

        let identity = Matrix4::identity();
        for &root_idx in &scene.root_nodes {
            process_node(root_idx, &identity, &scene, &mut mesh_transforms, compute_node_transform);
        }

        let config = NaniteConfig::default();
        let mut loaded_count = 0u32;
        let mut all_mesh_ids: Vec<usize> = Vec::new();

        for (mesh_idx, world_transform) in mesh_transforms {
            if let Some(loaded_mesh) = scene.meshes.get(mesh_idx) {
                let geometry = &loaded_mesh.geometry;

                // Convert to Vertex format and bake world transform into positions
                // This is needed because the frustum cull shader currently only supports
                // a single instance transform for all clusters
                let mut vertices: Vec<Vertex> = Vec::with_capacity(geometry.positions.len());
                for i in 0..geometry.positions.len() {
                    let pos = geometry.positions[i];
                    let normal = geometry.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);

                    // Transform position by world matrix
                    let pos_vec = Vector3::new(pos[0], pos[1], pos[2]);
                    let transformed_pos = world_transform.transform_point(&pos_vec);

                    // Transform normal by world matrix (ignoring translation)
                    let normal_vec = Vector3::new(normal[0], normal[1], normal[2]);
                    let transformed_normal = world_transform.transform_direction(&normal_vec);

                    vertices.push(Vertex {
                        position: [transformed_pos.x, transformed_pos.y, transformed_pos.z],
                        normal: [transformed_normal.x, transformed_normal.y, transformed_normal.z],
                        uv: geometry.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                        barycentric: [0.0, 0.0, 0.0],
                    });
                }

                // Build clusters
                let result = build_clusters(
                    &vertices,
                    &geometry.indices,
                    config.triangles_per_cluster,
                    loaded_mesh.material_index.unwrap_or(0) as u32,
                );

                // Register with Nanite renderer
                if let Some(ref mut nanite) = self.nanite_renderer {
                    let nanite_mesh_id = nanite.register_mesh(&self.device, &self.queue, result);
                    all_mesh_ids.push(nanite_mesh_id);
                    loaded_count += 1;
                }
            }
        }

        // Set identity transform for all instances (transforms already baked into vertices)
        if let Some(ref mut nanite) = self.nanite_renderer {
            let instances: Vec<(usize, Matrix4)> = all_mesh_ids
                .iter()
                .map(|&id| (id, Matrix4::identity()))
                .collect();
            nanite.update_instances(&self.queue, &instances);
        }

        // Extract materials and textures
        use crate::nanite::NaniteMaterialGpu;

        // Build texture name -> index map
        let mut texture_name_to_index: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
        let mut texture_data: Vec<(u32, u32, Vec<u8>)> = Vec::new();

        for (name, tex) in &scene.textures {
            let idx = texture_data.len() as i32;
            texture_name_to_index.insert(name.clone(), idx);
            web_sys::console::log_1(&format!(
                "  Texture {}: {} = {}x{} ({} bytes)",
                idx, name, tex.width, tex.height, tex.data.len()
            ).into());
            texture_data.push((tex.width, tex.height, tex.data.clone()));
        }

        web_sys::console::log_1(&format!(
            "Nanite: Found {} textures, {} materials",
            texture_data.len(),
            scene.materials.len()
        ).into());

        // Build materials array
        let mut gpu_materials: Vec<NaniteMaterialGpu> = Vec::new();
        for (i, mat) in scene.materials.iter().enumerate() {
            let texture_index = mat.base_color_texture
                .as_ref()
                .and_then(|name| texture_name_to_index.get(name).copied())
                .unwrap_or(-1);

            web_sys::console::log_1(&format!(
                "  Material {}: base_color={:?}, tex_ref={:?}, tex_idx={}",
                i, mat.base_color, mat.base_color_texture, texture_index
            ).into());

            gpu_materials.push(NaniteMaterialGpu {
                base_color: [
                    mat.base_color[0],
                    mat.base_color[1],
                    mat.base_color[2],
                    mat.base_color[3],
                ],
                texture_index,
                metallic: mat.metallic,
                roughness: mat.roughness,
                _pad: 0.0,
            });
        }

        // Add default material if none
        if gpu_materials.is_empty() {
            gpu_materials.push(NaniteMaterialGpu::default());
        }

        // Upload materials and textures
        if let Some(ref mut nanite) = self.nanite_renderer {
            let texture_refs: Vec<(u32, u32, &[u8])> = texture_data
                .iter()
                .map(|(w, h, d)| (*w, *h, d.as_slice()))
                .collect();

            nanite.upload_materials_and_textures(
                &self.device,
                &self.queue,
                &gpu_materials,
                &texture_refs,
            );

            // Create combined texture+shadow bind group
            if let Some(texture_view) = nanite.texture_array_view() {
                let texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Nanite Textures+Shadows Bind Group"),
                    layout: nanite.textures_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: nanite.material_buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(nanite.texture_sampler()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.shadow_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.shadow_map_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&self.shadow_cube_map_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Sampler(&self.shadow_cube_sampler),
                        },
                    ],
                });
                nanite.set_texture_bind_group(texture_bind_group);
            }
        }

        Ok(loaded_count)
    }

    // ========== Material Properties ==========

    /// Set clear coat intensity for all meshes (0.0 = none, 1.0 = full glossy coat).
    #[wasm_bindgen]
    pub fn set_clear_coat(&mut self, value: f32) {
        for mesh in &self.meshes {
            // clear_coat is at byte offset 28 (after base_color[16] + metallic[4] + roughness[4] + ao[4])
            self.queue.write_buffer(&mesh.material_buffer, 28, bytemuck::cast_slice(&[value]));
        }
    }

    /// Set clear coat roughness for all meshes (typically very low, e.g., 0.03).
    #[wasm_bindgen]
    pub fn set_clear_coat_roughness(&mut self, value: f32) {
        for mesh in &self.meshes {
            // clear_coat_roughness is at byte offset 32
            self.queue.write_buffer(&mesh.material_buffer, 32, bytemuck::cast_slice(&[value]));
        }
    }

    /// Set sheen/cloth intensity for all meshes (0.0 = none, 1.0 = full cloth BRDF).
    #[wasm_bindgen]
    pub fn set_sheen(&mut self, value: f32) {
        for mesh in &self.meshes {
            // sheen is at byte offset 36
            self.queue.write_buffer(&mesh.material_buffer, 36, bytemuck::cast_slice(&[value]));
        }
    }

    /// Set sheen color for all meshes.
    #[wasm_bindgen]
    pub fn set_sheen_color(&mut self, r: f32, g: f32, b: f32) {
        for mesh in &self.meshes {
            // sheen_color is at byte offset 64 (after padding at 52-63)
            self.queue.write_buffer(&mesh.material_buffer, 64, bytemuck::cast_slice(&[r, g, b]));
        }
    }

    /// Set metallic factor for all meshes (0.0 = dielectric, 1.0 = metal).
    #[wasm_bindgen]
    pub fn set_material_metallic(&mut self, value: f32) {
        for mesh in &self.meshes {
            // metallic is at byte offset 16 (after base_color[16])
            self.queue.write_buffer(&mesh.material_buffer, 16, bytemuck::cast_slice(&[value]));
        }
    }

    /// Set roughness factor for all meshes (0.04 = mirror-like, 1.0 = completely rough).
    #[wasm_bindgen]
    pub fn set_material_roughness(&mut self, value: f32) {
        for mesh in &self.meshes {
            // roughness is at byte offset 20
            self.queue.write_buffer(&mesh.material_buffer, 20, bytemuck::cast_slice(&[value]));
        }
    }

    // ========== Transform Gizmo ==========

    /// Enable or disable the transform gizmo.
    #[wasm_bindgen]
    pub fn set_gizmo_enabled(&mut self, enabled: bool) {
        self.gizmo_enabled = enabled;
        if enabled {
            self.update_gizmo_buffers();
        }
    }

    /// Check if gizmo is enabled.
    #[wasm_bindgen]
    pub fn is_gizmo_enabled(&self) -> bool {
        self.gizmo_enabled
    }

    /// Set the gizmo mode (0 = Translate, 1 = Rotate, 2 = Scale).
    #[wasm_bindgen]
    pub fn set_gizmo_mode(&mut self, mode: u32) {
        let gizmo_mode = match mode {
            0 => GizmoMode::Translate,
            1 => GizmoMode::Rotate,
            2 => GizmoMode::Scale,
            _ => GizmoMode::Translate,
        };
        self.gizmo.set_mode(gizmo_mode);
        if self.gizmo_enabled {
            self.update_gizmo_buffers();
        }
    }

    /// Get current gizmo mode (0 = Translate, 1 = Rotate, 2 = Scale).
    #[wasm_bindgen]
    pub fn get_gizmo_mode(&self) -> u32 {
        match self.gizmo.mode() {
            GizmoMode::Translate => 0,
            GizmoMode::Rotate => 1,
            GizmoMode::Scale => 2,
        }
    }

    /// Attach gizmo to a mesh by index.
    #[wasm_bindgen]
    pub fn attach_gizmo_to_mesh(&mut self, mesh_index: usize) {
        if mesh_index < self.meshes.len() {
            self.gizmo_target_mesh = Some(mesh_index);
            let pos = self.meshes[mesh_index].position;
            self.gizmo.set_position(pos);
            if self.gizmo_enabled {
                self.update_gizmo_buffers();
            }
        }
    }

    /// Detach gizmo from any mesh.
    #[wasm_bindgen]
    pub fn detach_gizmo(&mut self) {
        self.gizmo_target_mesh = None;
    }

    /// Get the index of the mesh the gizmo is attached to, or -1 if none.
    #[wasm_bindgen]
    pub fn get_gizmo_target_mesh(&self) -> i32 {
        self.gizmo_target_mesh.map(|i| i as i32).unwrap_or(-1)
    }

    /// Check if gizmo is currently being dragged.
    #[wasm_bindgen]
    pub fn is_gizmo_dragging(&self) -> bool {
        self.gizmo.is_dragging()
    }

    /// Handle gizmo hover - returns the hovered axis (0=None, 1=X, 2=Y, 3=Z, 4=XY, 5=XZ, 6=YZ, 7=Center).
    #[wasm_bindgen]
    pub fn on_gizmo_hover(&mut self, screen_x: f32, screen_y: f32) -> u32 {
        if !self.gizmo_enabled || self.gizmo_target_mesh.is_none() {
            return 0;
        }

        let ray = self.screen_to_ray(screen_x, screen_y);
        let camera = self.camera.borrow();
        self.gizmo.update_screen_scale(camera.position);
        drop(camera);

        let axis = self.gizmo.hit_test(&ray);
        if axis.is_some() {
            self.update_gizmo_buffers();
        }

        match axis {
            Some(GizmoAxis::X) => 1,
            Some(GizmoAxis::Y) => 2,
            Some(GizmoAxis::Z) => 3,
            Some(GizmoAxis::XY) => 4,
            Some(GizmoAxis::XZ) => 5,
            Some(GizmoAxis::YZ) => 6,
            Some(GizmoAxis::Center) => 7,
            _ => 0,
        }
    }

    /// Start a gizmo drag operation. Returns true if drag started successfully.
    #[wasm_bindgen]
    pub fn on_gizmo_drag_start(&mut self, screen_x: f32, screen_y: f32) -> bool {
        if !self.gizmo_enabled {
            return false;
        }

        let mesh_index = match self.gizmo_target_mesh {
            Some(i) => i,
            None => return false,
        };

        let ray = self.screen_to_ray(screen_x, screen_y);
        let camera = self.camera.borrow();
        self.gizmo.update_screen_scale(camera.position);
        drop(camera);

        let axis = match self.gizmo.hit_test(&ray) {
            Some(a) => a,
            None => return false,
        };

        if axis == GizmoAxis::None {
            return false;
        }

        let mesh = &self.meshes[mesh_index];
        let position = mesh.position;
        let rotation = Quaternion::from_euler(&Euler::xyz(mesh.rotation.x, mesh.rotation.y, mesh.rotation.z));
        let scale = mesh.scale;

        let started = self.gizmo.begin_drag(axis, &ray, position, rotation, scale);
        if started {
            self.update_gizmo_buffers();
        }
        started
    }

    /// Update a gizmo drag operation and apply transform to target mesh.
    #[wasm_bindgen]
    pub fn on_gizmo_drag(&mut self, screen_x: f32, screen_y: f32) {
        if !self.gizmo.is_dragging() {
            return;
        }

        let mesh_index = match self.gizmo_target_mesh {
            Some(i) => i,
            None => return,
        };

        let ray = self.screen_to_ray(screen_x, screen_y);
        let result = self.gizmo.update_drag(&ray);

        // Apply the transform result to the mesh
        match result {
            GizmoDragResult::Translate(delta) => {
                let mesh = &mut self.meshes[mesh_index];
                mesh.position = mesh.position + delta;
                self.gizmo.set_position(mesh.position);
                self.update_mesh_transform(mesh_index);
            }
            GizmoDragResult::Rotate(delta_quat) => {
                let mesh = &mut self.meshes[mesh_index];
                let current_quat = Quaternion::from_euler(&Euler::xyz(mesh.rotation.x, mesh.rotation.y, mesh.rotation.z));
                let new_quat = delta_quat.multiply(&current_quat);
                let euler = Euler::from_quaternion(&new_quat, EulerOrder::XYZ);
                mesh.rotation = Vector3::new(euler.x, euler.y, euler.z);
                self.update_mesh_transform(mesh_index);
            }
            GizmoDragResult::Scale(delta_scale) => {
                let mesh = &mut self.meshes[mesh_index];
                mesh.scale = Vector3::new(
                    mesh.scale.x * delta_scale.x,
                    mesh.scale.y * delta_scale.y,
                    mesh.scale.z * delta_scale.z,
                );
                self.update_mesh_transform(mesh_index);
            }
            GizmoDragResult::None => {}
        }

        self.update_gizmo_buffers();
    }

    /// End a gizmo drag operation.
    #[wasm_bindgen]
    pub fn on_gizmo_drag_end(&mut self) {
        self.gizmo.end_drag();
        self.update_gizmo_buffers();
    }

    /// Get number of meshes (for UI to show mesh selector).
    #[wasm_bindgen]
    pub fn get_mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Convert screen coordinates to a world-space ray.
    fn screen_to_ray(&self, screen_x: f32, screen_y: f32) -> crate::math::Ray {
        let mut camera = self.camera.borrow_mut();
        let view = camera.view_matrix().clone();
        let proj = camera.projection_matrix().clone();
        let view_proj = proj.multiply(&view);
        let view_proj_inverse = view_proj.inverse();

        Raycaster::ray_from_screen(
            screen_x,
            screen_y,
            self.width as f32,
            self.height as f32,
            &view_proj_inverse,
        )
    }

    /// Update mesh transform buffer after gizmo manipulation.
    fn update_mesh_transform(&mut self, mesh_index: usize) {
        if mesh_index >= self.meshes.len() {
            return;
        }

        let mesh = &self.meshes[mesh_index];
        let rotation_quat = Quaternion::from_euler(&Euler::xyz(mesh.rotation.x, mesh.rotation.y, mesh.rotation.z));
        let world_transform = Matrix4::compose(&mesh.position, &rotation_quat, &mesh.scale);
        let normal_matrix = Matrix3::from_matrix4_normal(&world_transform);

        let model_uniform = PbrModelUniform {
            model: matrix4_to_array(&world_transform),
            normal: matrix3_to_padded_array(&normal_matrix),
        };

        self.queue.write_buffer(
            &mesh.model_buffer,
            0,
            bytemuck::cast_slice(&[model_uniform]),
        );
    }

    /// Update gizmo line object buffers.
    fn update_gizmo_buffers(&mut self) {
        // Update gizmo scale and rebuild geometry
        let camera = self.camera.borrow();
        self.gizmo.update_screen_scale(camera.position);
        drop(camera);

        self.gizmo.update(&self.device, &self.queue);

        let vertices = self.gizmo.line().vertices();
        if vertices.is_empty() {
            return;
        }

        let vertex_data: Vec<f32> = vertices
            .iter()
            .flat_map(|v| [v.position[0], v.position[1], v.position[2], v.color[0], v.color[1], v.color[2], v.color[3]])
            .collect();

        // Compute model uniform with gizmo position
        let position = self.gizmo.position();
        let model_matrix = Matrix4::from_translation(&position);
        let model_uniform = LineModelUniform {
            model: matrix4_to_array(&model_matrix),
        };

        if let Some(ref mut line_obj) = self.gizmo_line_object {
            // Update existing buffers if size is the same
            if line_obj.vertex_count == vertices.len() as u32 {
                self.queue.write_buffer(&line_obj.vertex_buffer, 0, bytemuck::cast_slice(&vertex_data));
                self.queue.write_buffer(&line_obj.model_buffer, 0, bytemuck::cast_slice(&[model_uniform]));
            } else {
                // Recreate buffers
                let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Gizmo Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

                let model_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Gizmo Model Buffer"),
                    contents: bytemuck::cast_slice(&[model_uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                if let Some(model_bind_group) = self.line_material.create_model_bind_group(&self.device, &model_buffer) {
                    line_obj.vertex_buffer = vertex_buffer;
                    line_obj.vertex_count = vertices.len() as u32;
                    line_obj.model_buffer = model_buffer;
                    line_obj.model_bind_group = model_bind_group;
                }
            }
        } else {
            // Create new line object
            let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gizmo Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertex_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

            let model_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gizmo Model Buffer"),
                contents: bytemuck::cast_slice(&[model_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            if let Some(model_bind_group) = self.line_material.create_model_bind_group(&self.device, &model_buffer) {
                self.gizmo_line_object = Some(LineObject {
                    vertex_buffer,
                    vertex_count: vertices.len() as u32,
                    model_buffer,
                    model_bind_group,
                });
            }
        }
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
