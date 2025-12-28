//! PBR (Physically Based Rendering) material.

use crate::core::Id;
use crate::geometry::Vertex;
use crate::math::Color;
use bytemuck::{Pod, Zeroable};

/// Camera uniform for PBR (includes position for specular and hemisphere light).
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PbrCameraUniform {
    /// Combined view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Camera world position.
    pub position: [f32; 3],
    /// Render mode: 0=Lit, 1=Unlit, 2=Normals, 3=Depth, 4=Metallic, 5=Roughness, 6=AO, 7=UVs
    pub render_mode: u32,
    /// Hemisphere light sky color (RGB) + enabled flag (W: 1.0 = enabled).
    pub hemisphere_sky: [f32; 4],
    /// Hemisphere light ground color (RGB) + intensity (W).
    pub hemisphere_ground: [f32; 4],
    /// IBL settings: x=diffuse intensity, y=specular intensity, z=unused, w=unused.
    pub ibl_settings: [f32; 4],
    /// Light 0: position.xyz + intensity.w
    pub light0_pos: [f32; 4],
    /// Light 0: color.rgb + enabled.w
    pub light0_color: [f32; 4],
    /// Light 1: position.xyz + intensity.w
    pub light1_pos: [f32; 4],
    /// Light 1: color.rgb + enabled.w
    pub light1_color: [f32; 4],
    /// Light 2: position.xyz + intensity.w
    pub light2_pos: [f32; 4],
    /// Light 2: color.rgb + enabled.w
    pub light2_color: [f32; 4],
    /// Light 3: position.xyz + intensity.w
    pub light3_pos: [f32; 4],
    /// Light 3: color.rgb + enabled.w
    pub light3_color: [f32; 4],
    /// Detail mapping: x=enabled (0/1), y=scale (UV tiling), z=intensity, w=max_distance
    pub detail_settings: [f32; 4],
}

impl Default for PbrCameraUniform {
    fn default() -> Self {
        Self {
            view_proj: [[0.0; 4]; 4],
            position: [0.0; 3],
            render_mode: 0,
            hemisphere_sky: [0.6, 0.75, 1.0, 0.0],
            hemisphere_ground: [0.4, 0.3, 0.2, 1.0],
            ibl_settings: [0.3, 1.0, 0.0, 0.0],
            // Car studio lighting preset
            light0_pos: [5.0, 8.0, 5.0, 15.0],      // Key light: front-right, high, intensity 15
            light0_color: [1.0, 0.98, 0.95, 1.0],   // Warm white, enabled
            light1_pos: [-5.0, 6.0, 3.0, 10.0],     // Fill light: front-left, intensity 10
            light1_color: [0.9, 0.95, 1.0, 1.0],    // Cool white, enabled
            light2_pos: [0.0, 4.0, -6.0, 8.0],      // Rim light: behind, intensity 8
            light2_color: [1.0, 1.0, 1.0, 1.0],     // Pure white, enabled
            light3_pos: [-3.0, 1.0, -3.0, 5.0],     // Ground bounce: low, back-left, intensity 5
            light3_color: [0.8, 0.85, 0.9, 1.0],    // Slight blue, enabled
            // Detail mapping: disabled by default, scale=10 (tiles 10x), intensity=0.3, max_distance=5
            detail_settings: [0.0, 10.0, 0.3, 5.0],
        }
    }
}

/// Model uniform data.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PbrModelUniform {
    /// Model matrix.
    pub model: [[f32; 4]; 4],
    /// Normal matrix (inverse transpose of model, padded to 3x4).
    pub normal: [[f32; 4]; 3],
}

/// Material properties for PBR.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PbrMaterialUniform {
    /// Base color (albedo).
    pub base_color: [f32; 4],
    /// Metallic factor (0 = dielectric, 1 = metal).
    pub metallic: f32,
    /// Roughness factor (0 = smooth, 1 = rough).
    pub roughness: f32,
    /// Ambient occlusion.
    pub ao: f32,
    /// Padding.
    pub _padding: f32,
}

impl Default for PbrMaterialUniform {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            _padding: 0.0,
        }
    }
}

/// A PBR material using metallic-roughness workflow.
pub struct PbrMaterial {
    /// Unique ID.
    id: Id,
    /// Base color.
    pub color: Color,
    /// Metallic factor.
    pub metallic: f32,
    /// Roughness factor.
    pub roughness: f32,
    /// Ambient occlusion.
    pub ao: f32,
    /// Render pipeline.
    pipeline: Option<wgpu::RenderPipeline>,
    /// Camera bind group layout.
    camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Model bind group layout.
    model_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Material bind group layout.
    material_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Whether the material needs to rebuild.
    needs_update: bool,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self::new()
    }
}

impl PbrMaterial {
    /// Shader source.
    const SHADER_SOURCE: &'static str = include_str!("../shaders/pbr.wgsl");

    /// Create a new PBR material with default values.
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            color: Color::WHITE,
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            pipeline: None,
            camera_bind_group_layout: None,
            model_bind_group_layout: None,
            material_bind_group_layout: None,
            needs_update: true,
        }
    }

    /// Create a metallic material.
    pub fn metal(color: Color, roughness: f32) -> Self {
        Self {
            color,
            metallic: 1.0,
            roughness,
            ..Self::new()
        }
    }

    /// Create a dielectric (non-metal) material.
    pub fn dielectric(color: Color, roughness: f32) -> Self {
        Self {
            color,
            metallic: 0.0,
            roughness,
            ..Self::new()
        }
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the render pipeline.
    #[inline]
    pub fn pipeline(&self) -> Option<&wgpu::RenderPipeline> {
        self.pipeline.as_ref()
    }

    /// Get camera bind group layout.
    #[inline]
    pub fn camera_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.camera_bind_group_layout.as_ref()
    }

    /// Get model bind group layout.
    #[inline]
    pub fn model_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.model_bind_group_layout.as_ref()
    }

    /// Get material bind group layout.
    #[inline]
    pub fn material_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.material_bind_group_layout.as_ref()
    }

    /// Get the material uniform data.
    pub fn uniform(&self) -> PbrMaterialUniform {
        PbrMaterialUniform {
            base_color: [self.color.r, self.color.g, self.color.b, 1.0],
            metallic: self.metallic,
            roughness: self.roughness,
            ao: self.ao,
            _padding: 0.0,
        }
    }

    /// Build the render pipeline.
    pub fn build_pipeline(
        &mut self,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PBR Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SOURCE.into()),
        });

        // Camera bind group layout (group 0)
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PBR Camera Bind Group Layout"),
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

        // Model bind group layout (group 1)
        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PBR Model Bind Group Layout"),
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

        // Material bind group layout (group 2)
        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PBR Material Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &model_bind_group_layout,
                &material_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PBR Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        self.pipeline = Some(pipeline);
        self.camera_bind_group_layout = Some(camera_bind_group_layout);
        self.model_bind_group_layout = Some(model_bind_group_layout);
        self.material_bind_group_layout = Some(material_bind_group_layout);
        self.needs_update = false;
    }

    /// Create a camera bind group.
    pub fn create_camera_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        self.camera_bind_group_layout.as_ref().map(|layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PBR Camera Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
    }

    /// Create a model bind group.
    pub fn create_model_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        self.model_bind_group_layout.as_ref().map(|layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PBR Model Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
    }

    /// Create a material bind group.
    pub fn create_material_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        self.material_bind_group_layout.as_ref().map(|layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PBR Material Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
    }

    /// Check if needs update.
    #[inline]
    pub fn needs_update(&self) -> bool {
        self.needs_update || self.pipeline.is_none()
    }
}
