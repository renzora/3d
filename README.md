# Ren - WASM/wgpu 3D Engine Roadmap

A high-performance 3D engine built with Rust, wgpu, and WebAssembly featuring Nanite-style virtualized geometry and Lumen-style global illumination.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         JavaScript API Layer                              │
│  (Mirrors Three.js/Babylon.js API - one class per file)                  │
├──────────────────────────────────────────────────────────────────────────┤
│                    wasm-bindgen / SharedArrayBuffer                       │
├──────────────────────────────────────────────────────────────────────────┤
│                      Rust Core Engine                                     │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                      Rendering Pipelines                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │  │
│  │  │   Standard   │  │    Nanite    │  │    Lumen     │             │  │
│  │  │   Pipeline   │  │   Pipeline   │  │   Pipeline   │             │  │
│  │  │ (traditional)│  │ (virtualized)│  │  (global GI) │             │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │  │
│  │         │                 │                 │                      │  │
│  │         └─────────────────┴─────────────────┘                      │  │
│  │                           │                                        │  │
│  │                           ▼                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │              Unified G-Buffer / Lighting Pass               │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                     wgpu / WebGPU Backend                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation & Core Rendering ✅

### 1.1 Project Setup
- [x] `Cargo.toml` - wgpu, wasm-bindgen, web-sys dependencies
- [x] `build-wasm.sh` - WASM build script
- [x] `web/index.html` - Test page with stats overlay
- [ ] `package.json` - JS library bundling (future)
- [ ] `tsconfig.json` - TypeScript definitions (future)

### 1.2 Math Library (`src/math/`) ✅
| File | Description | Status |
|------|-------------|--------|
| `mod.rs` | Module exports | ✅ |
| `vector2.rs` | 2D vector operations | ✅ |
| `vector3.rs` | 3D vector operations | ✅ |
| `vector4.rs` | 4D vector/homogeneous coords | ✅ |
| `matrix3.rs` | 3x3 matrix (normals, 2D transforms) | ✅ |
| `matrix4.rs` | 4x4 matrix (MVP transforms) | ✅ |
| `quaternion.rs` | Rotation representation | ✅ |
| `euler.rs` | Euler angle rotations | ✅ |
| `color.rs` | RGB/HSL color with conversions | ✅ |
| `ray.rs` | Ray for raycasting | ✅ |
| `plane.rs` | Infinite plane | ✅ |
| `sphere.rs` | Bounding sphere | ✅ |
| `box3.rs` | Axis-aligned bounding box | ✅ |
| `frustum.rs` | View frustum (6 planes) | ✅ |
| `triangle.rs` | Triangle primitive | ✅ |
| `line3.rs` | Line segment | ✅ |

### 1.3 Core Engine (`src/core/`) ✅
| File | Description | Status |
|------|-------------|--------|
| `mod.rs` | Module exports | ✅ |
| `engine.rs` | Main engine entry point | ✅ |
| `context.rs` | wgpu device/queue/surface management | ✅ |
| `renderer.rs` | WebGPU render pipeline orchestration | ✅ |
| `clock.rs` | Delta time, elapsed time | ✅ |
| `id.rs` | Unique ID generation | ✅ |
| `render_state.rs` | Current render pass state | |
| `render_target.rs` | Framebuffer abstraction | |
| `event_dispatcher.rs` | Event system base | |
| `layers.rs` | Object layer masking (32 layers) | |
| `bind_group_manager.rs` | wgpu bind group caching | |
| `pipeline_cache.rs` | Compiled pipeline caching | |
| `resource_manager.rs` | GPU resource lifecycle | |

### 1.4 Scene Graph (`src/scene/`) ✅
| File | Description | Status |
|------|-------------|--------|
| `mod.rs` | Module exports | ✅ |
| `object3d.rs` | Base class for all 3D objects | ✅ |
| `scene.rs` | Root scene container | ✅ |
| `transform.rs` | Position/rotation/scale component | ✅ |
| `visibility.rs` | Visibility and culling flags | ✅ |
| `group.rs` | Empty transform node | |
| `fog.rs` | Linear/exponential fog | |
| `background.rs` | Scene background (color/texture/skybox) | |

---

## Phase 2: Geometry System ✅

### 2.1 Buffer Management (`src/geometry/`)
| File | Description | Status |
|------|-------------|--------|
| `vertex.rs` | Vertex types (Vertex, ColorVertex, PositionVertex) | ✅ |
| `buffer_geometry.rs` | Vertex/index buffer container | ✅ |
| `primitives.rs` | Quick triangle/quad creation | ✅ |

### 2.2 Primitive Geometries (`src/geometry/`)
| File | Description | Status |
|------|-------------|--------|
| `box_geometry.rs` | Cube/rectangular prism | ✅ |
| `sphere_geometry.rs` | UV sphere | ✅ |
| `plane_geometry.rs` | Flat plane with subdivisions | ✅ |
| `cylinder_geometry.rs` | Cylinder with caps | ✅ |
| `torus_geometry.rs` | Donut shape | ✅ |
| `icosahedron_geometry.rs` | Icosahedron (basis for geodesic) | |
| `circle_geometry.rs` | Flat circle/disc | |
| `ring_geometry.rs` | Flat ring/annulus | |
| `cone_geometry.rs` | Cone with base | |
| `torus_knot_geometry.rs` | Torus knot | |
| `capsule_geometry.rs` | Capsule/stadium shape | |
| `dodecahedron_geometry.rs` | 12-sided polyhedron | |
| `octahedron_geometry.rs` | 8-sided polyhedron | |
| `tetrahedron_geometry.rs` | 4-sided polyhedron | |
| `polyhedron_geometry.rs` | Base for platonic solids | |
| `lathe_geometry.rs` | Revolution of 2D path | |
| `extrude_geometry.rs` | Extruded 2D shape | |
| `tube_geometry.rs` | Tube along curve | |
| `edges_geometry.rs` | Edge lines from geometry | |
| `wireframe_geometry.rs` | Wireframe from geometry | |
| `text_geometry.rs` | 3D text from font | |

### 2.3 Camera System (`src/camera/`) ✅
| File | Description | Status |
|------|-------------|--------|
| `perspective.rs` | Perspective projection camera | ✅ |
| `orthographic.rs` | Orthographic projection camera | ✅ |

### 2.3 Curve System (`src/curves/`)
| File | Description |
|------|-------------|
| `curve.rs` | Base curve trait |
| `curve_path.rs` | Path made of curves |
| `catmull_rom_curve3.rs` | Catmull-Rom spline |
| `cubic_bezier_curve3.rs` | Cubic Bezier curve |
| `quadratic_bezier_curve3.rs` | Quadratic Bezier |
| `line_curve3.rs` | Straight line segment |
| `ellipse_curve.rs` | Ellipse/arc curve |

---

## Phase 3: Materials & Shaders ✅

### 3.1 Material System (`src/materials/`)
| File | Description | Status |
|------|-------------|--------|
| `material.rs` | Base material trait | |
| `basic_material.rs` | Unlit solid color/texture | ✅ |
| `lambert_material.rs` | Diffuse-only lighting | |
| `phong_material.rs` | Specular highlights | |
| `standard_material.rs` | PBR metallic-roughness | ✅ |
| `physical_material.rs` | Extended PBR (clearcoat, sheen) |
| `toon_material.rs` | Cel-shaded/cartoon |
| `normal_material.rs` | Debug normals visualization |
| `depth_material.rs` | Depth buffer output |
| `points_material.rs` | Point cloud rendering |
| `line_basic_material.rs` | Simple line material |
| `sprite_material.rs` | Billboard sprite |
| `shader_material.rs` | Custom shader material |
| `shadow_material.rs` | Shadow-only material |

### 3.2 Shader System (`src/shaders/`)
| File | Description | Status |
|------|-------------|--------|
| `basic.wgsl` | Basic unlit shader | ✅ |
| `standard.wgsl` | Standard lighting shader | ✅ |
| `pbr.wgsl` | PBR metallic-roughness shader | ✅ |
| `pbr_textured.wgsl` | Full PBR with IBL, shadows, debug modes | ✅ |
| `pbr_lit.wgsl` | PBR with lighting | ✅ |
| `line.wgsl` | Line rendering shader | ✅ |
| `skybox.wgsl` | Skybox shader | ✅ |
| `shader_lib.rs` | Shader chunk library | |
| `shader_chunk.rs` | Reusable shader snippets | |
| `uniform_lib.rs` | Common uniform definitions | |
| `program.rs` | Compiled shader program | |

---

## Phase 4: Textures & Images ✅

### 4.1 Texture System (`src/texture/`)
| File | Description | Status |
|------|-------------|--------|
| `mod.rs` | Module exports | ✅ |
| `texture2d.rs` | Standard 2D texture | ✅ |
| `sampler.rs` | Texture sampling parameters | ✅ |
| `texture3d.rs` | 3D/volume texture | |
| `cube_texture.rs` | Cubemap texture | ✅ |
| `brdf_lut.rs` | BRDF integration LUT for IBL | ✅ |
| `data_texture.rs` | Texture from raw data | |
| `compressed_texture.rs` | GPU compressed formats | |
| `video_texture.rs` | Video as texture | |
| `depth_texture.rs` | Depth buffer texture | |
| `render_target_texture.rs` | Render-to-texture | |

### 4.2 Loaders (`src/loaders/`)
| File | Description | Status |
|------|-------------|--------|
| `image_loader.rs` | PNG/JPG/WebP loading | |
| `texture_loader.rs` | Texture from URL | |
| `cube_texture_loader.rs` | Cubemap loading | |
| `hdr_loader.rs` | HDRI environment maps | ✅ |
| `ktx2_loader.rs` | KTX2 compressed textures | |

---

## Phase 5: Lighting System ✅

### 5.1 Lights (`src/light/`)
| File | Description | Status |
|------|-------------|--------|
| `mod.rs` | Light types, LightsUniform | ✅ |
| `ambient.rs` | Global ambient illumination | ✅ |
| `directional.rs` | Sunlight (parallel rays) | ✅ |
| `point.rs` | Omni-directional light | ✅ |
| `spot.rs` | Cone-shaped light | ✅ |
| `hemisphere_light.rs` | Sky/ground gradient | ✅ |
| `rect_area_light.rs` | Area light (LTC) | |

### 5.2 Shadows (`src/shadows/`) ✅
| File | Description | Status |
|------|-------------|--------|
| `mod.rs` | Shadow config, PCF modes | ✅ |
| `shadow_config.rs` | Shadow quality settings, PCSS config | ✅ |
| `shadow_map.rs` | Shadow map texture management | ✅ |
| `shadow_pass.rs` | Shadow rendering pass | ✅ |
| `point_shadow.rs` | Omnidirectional shadow (cubemap) | ✅ |
| `cascade.rs` | Cascaded shadow maps | ✅ (infrastructure) |
| Contact shadows | Screen-space ray marching | ✅ (in pbr_textured.wgsl) |
| PCSS | Percentage-Closer Soft Shadows | ✅ (in pbr_textured.wgsl) |

---

## Phase 6: Cameras & Controls ✅

### 6.1 Cameras (`src/camera/`)
| File | Description | Status |
|------|-------------|--------|
| `perspective.rs` | Perspective projection | ✅ |
| `orthographic.rs` | Orthographic projection | ✅ |
| `cube_camera.rs` | 6-face cubemap capture | |

### 6.2 Controls (`src/controls/`)
| File | Description | Status |
|------|-------------|--------|
| `orbit.rs` | Orbit around target | ✅ |
| `fly_controls.rs` | 6DOF flight | |
| `first_person_controls.rs` | FPS-style look | |
| `pointer_lock_controls.rs` | Mouse lock controls | |
| `drag_controls.rs` | Object dragging | |
| `transform_controls.rs` | Gizmo manipulation | |

---

## Phase 7: Objects & Renderables ✅

### 7.1 Objects (`src/objects/`)
| File | Description | Status |
|------|-------------|--------|
| `mesh.rs` | Geometry + material renderable | ✅ |
| `skinned_mesh.rs` | Animated mesh with skeleton | |
| `instanced_mesh.rs` | GPU instanced mesh | ✅ |
| `line.rs` | Line rendering | ✅ |
| `line_segments.rs` | Disconnected line segments | |
| `points.rs` | Point cloud | |
| `sprite.rs` | Billboard sprite | |
| `lod.rs` | Level of detail object | |
| `bone.rs` | Skeleton bone | |
| `skeleton.rs` | Bone hierarchy | |

### 7.2 Helpers (`src/helpers/`)
| File | Description | Status |
|------|-------------|--------|
| `axes_helper.rs` | XYZ axes visualization | ✅ |
| `grid_helper.rs` | Ground grid | ✅ |
| `box_helper.rs` | Bounding box wireframe | ✅ |
| `arrow_helper.rs` | Direction arrow | |
| `camera_helper.rs` | Camera frustum vis | |
| `light_helper.rs` | Light visualization | |
| `skeleton_helper.rs` | Bone visualization | |

---

## Phase 8: Animation System ✅

### 8.1 Animation Core (`src/animation/`)
| File | Description | Status |
|------|-------------|--------|
| `animation_clip.rs` | Animation data container | ✅ |
| `animation_mixer.rs` | Animation playback controller | ✅ |
| `animation_action.rs` | Single animation instance | ✅ |
| `keyframe_track.rs` | Base keyframe track | ✅ |
| `vector_keyframe_track.rs` | Vector3 animation | ✅ |
| `quaternion_keyframe_track.rs` | Rotation animation | ✅ |
| `number_keyframe_track.rs` | Scalar animation | ✅ |
| `color_keyframe_track.rs` | Color animation | ✅ |
| `property_binding.rs` | Object property binding | |

### 8.2 Interpolation (`src/animation/interpolants/`)
| File | Description | Status |
|------|-------------|--------|
| `interpolant.rs` | Base interpolant | ✅ |
| `linear_interpolant.rs` | Linear interpolation | ✅ |
| `discrete_interpolant.rs` | Step/discrete | ✅ |
| `cubic_interpolant.rs` | Cubic spline | ✅ |

---

## Phase 9: Model Loaders ✅

### 9.1 Format Loaders (`src/loaders/`)
| File | Description | Status |
|------|-------------|--------|
| `loader.rs` | Base loader trait | ✅ |
| `loading_manager.rs` | Load progress tracking | ✅ |
| `gltf_loader.rs` | GLTF/GLB models with textures | ✅ |
| `hdr_loader.rs` | HDR/EXR environment maps | ✅ |
| `obj_loader.rs` | Wavefront OBJ | ✅ |
| `draco_loader.rs` | Draco mesh compression | |

---

## Phase 10: Post-Processing ✅

### 10.1 Effect System (`src/postprocessing/`)
| File | Description | Status |
|------|-------------|--------|
| `effect_composer.rs` | Post-process pipeline | ✅ |
| `pass.rs` | Base render pass | ✅ |
| `render_pass.rs` | Main scene render | |
| `shader_pass.rs` | Full-screen shader | |
| `copy_pass.rs` | Texture copy | |
| `output_pass.rs` | Final output | |

### 10.2 Effects (`src/postprocessing/effects/`)
| File | Description | Status |
|------|-------------|--------|
| `bloom_pass.rs` | Glow/bloom effect | ✅ |
| `ssao_pass.rs` | Screen-space AO | ✅ |
| `ssr_pass.rs` | Screen-space reflections | ✅ |
| `dof_pass.rs` | Depth of field | ✅ |
| `motion_blur_pass.rs` | Motion blur | ✅ |
| `fxaa_pass.rs` | FXAA anti-aliasing | ✅ |
| `smaa_pass.rs` | SMAA anti-aliasing | ✅ |
| `taa_pass.rs` | Temporal AA | ✅ |
| `outline_pass.rs` | Object outlines | ✅ |
| `tonemapping_pass.rs` | HDR tonemapping | ✅ |
| `color_correction_pass.rs` | Brightness/contrast/gamma | ✅ |
| `vignette_pass.rs` | Vignette effect | ✅ |
| `gtao_pass.rs` | Ground Truth AO | ✅ |
| `lumen_pass.rs` | Screen-space Global Illumination | ✅ |
| `skybox_pass.rs` | Skybox rendering | ✅ |
| `volumetric_fog_pass.rs` | Volumetric fog with shadow-aware god rays | ✅ |
| `auto_exposure_pass.rs` | Auto-exposure with histogram-based eye adaptation | ✅ |

---

## Phase 11: SharedArrayBuffer Multi-Threading

### 11.1 Thread Infrastructure (`src/threading/`)
| File | Description |
|------|-------------|
| `thread_pool.rs` | Rayon-style thread pool for WASM |
| `worker_pool.rs` | Web Worker pool management |
| `shared_memory.rs` | SharedArrayBuffer wrapper |
| `atomic_ops.rs` | Atomics operations wrapper |
| `spinlock.rs` | Spinlock for short critical sections |
| `mutex.rs` | Mutex using Atomics.wait/notify |
| `channel.rs` | Lock-free SPSC/MPSC channels |
| `ring_buffer.rs` | Lock-free ring buffer |

### 11.2 Parallel Data Structures (`src/threading/collections/`)
| File | Description |
|------|-------------|
| `concurrent_vec.rs` | Thread-safe growable array |
| `concurrent_queue.rs` | Lock-free queue |
| `arena_allocator.rs` | Thread-local bump allocator |
| `object_pool.rs` | Thread-safe object pool |

### 11.3 Parallel Jobs (`src/threading/jobs/`)
| File | Description |
|------|-------------|
| `job.rs` | Job/task representation |
| `job_system.rs` | Job scheduling system |
| `parallel_for.rs` | Parallel for loop |

### 11.4 Memory Layout (`src/threading/memory/`)
| File | Description |
|------|-------------|
| `shared_heap.rs` | Shared linear heap |
| `typed_array_view.rs` | View into SharedArrayBuffer |
| `interop_buffer.rs` | JS ↔ WASM shared buffer |
| `double_buffer.rs` | Double-buffering for frames |
| `frame_allocator.rs` | Per-frame bump allocator |

---

## Phase 12: Nanite - Virtualized Geometry

### Architecture

Nanite is NOT a separate renderer. It's a geometry pipeline that feeds into the same unified lighting pass.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Geometry Pipelines                              │
│                                                                         │
│   ┌─────────────────┐              ┌─────────────────┐                 │
│   │  Standard Mesh  │              │   Nanite Mesh   │                 │
│   │    Pipeline     │              │    Pipeline     │                 │
│   │                 │              │                 │                 │
│   │  - Vertex buf   │              │  - Clusters     │                 │
│   │  - Index buf    │              │  - DAG LOD      │                 │
│   │  - Draw calls   │              │  - Vis buffer   │                 │
│   └────────┬────────┘              └────────┬────────┘                 │
│            │                                │                          │
│            │         ┌──────────────────────┘                          │
│            │         │                                                 │
│            ▼         ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │                     G-Buffer (shared)                           │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                │                                       │
│                                ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │                   Lighting Pass (shared)                        │ │
│   │                   (includes Lumen GI)                           │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Asset Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Asset Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   OFFLINE (Build Time):                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  GLB/GLTF ──► ren-cli preprocess ──► .nanite file              │  │
│   │              (Rust CLI tool)                                    │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   RUNTIME (On Demand):                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  GLB ──► Web Worker ──► Process ──► IndexedDB cache            │  │
│   │                        (seconds)    (reused next load)          │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 12.1 Preprocessing Pipeline (`src/nanite/preprocessing/`)
| File | Description |
|------|-------------|
| `mesh_simplifier.rs` | Quadric error mesh decimation |
| `cluster_builder.rs` | Split mesh into 128-triangle clusters |
| `cluster_group.rs` | Group clusters for hierarchical LOD |
| `dag_builder.rs` | Build cluster DAG (directed acyclic graph) |
| `bounding_sphere.rs` | Tight bounding spheres for clusters |
| `edge_lock.rs` | Lock boundary edges during simplification |
| `mesh_partitioner.rs` | METIS-style graph partitioning |
| `hierarchy_builder.rs` | Build LOD hierarchy from clusters |
| `nanite_mesh.rs` | Final nanite mesh container |
| `nanite_file.rs` | .nanite file format read/write |

### 12.2 GPU-Driven Pipeline (`src/nanite/runtime/`)
| File | Description |
|------|-------------|
| `instance_culler.rs` | GPU instance frustum/occlusion cull |
| `cluster_culler.rs` | Per-cluster visibility culling |
| `persistent_culler.rs` | Two-pass occlusion culling |
| `hi_z_buffer.rs` | Hierarchical depth buffer |
| `visibility_buffer.rs` | Visibility buffer (triangle ID + instance ID) |
| `software_rasterizer.rs` | Compute shader rasterization (<32px triangles) |
| `hardware_rasterizer.rs` | Standard rasterization (>32px triangles) |
| `hybrid_rasterizer.rs` | Switches between SW/HW based on size |
| `indirect_dispatch.rs` | GPU-driven indirect dispatch |
| `cluster_lod_selector.rs` | GPU LOD selection per cluster |
| `streaming_manager.rs` | Geometry streaming from disk/network |
| `page_cache.rs` | GPU memory page cache for geometry |

### 12.3 Material Pass (`src/nanite/shading/`)
| File | Description |
|------|-------------|
| `visibility_resolve.rs` | Resolve visibility buffer to G-buffer |
| `deferred_material.rs` | Deferred material evaluation |
| `material_bin.rs` | Bin triangles by material |

### 12.4 Data Structures (`src/nanite/data/`)
| File | Description |
|------|-------------|
| `cluster.rs` | Cluster data structure (128 tris) |
| `cluster_group.rs` | Group of 8-32 clusters |
| `bvh_node.rs` | BVH node for clusters |
| `packed_vertex.rs` | Quantized vertex format |
| `packed_index.rs` | Compressed index buffer |
| `error_metric.rs` | LOD error calculation |
| `page.rs` | Streamable geometry page (64KB) |

---

## Phase 13: Lumen - Global Illumination ✅ (SSGI Implemented)

### Current Implementation

Screen-Space Global Illumination (SSGI) is fully implemented with:
- **SSGI Pass**: Cosine-weighted hemisphere ray tracing (4-16 rays based on quality)
- **Denoise Pass**: Edge-aware bilateral blur (5x5 kernel) with depth/normal weighting
- **Temporal Pass**: 95% history blend with disocclusion detection and neighborhood clamping
- **Composite Pass**: Adds GI to scene with intensity control

Quality presets: Low (4 rays), Medium (8 rays), High (12 rays), Ultra (16 rays)

### Architecture (Full Lumen - Future)

Lumen provides global illumination through a hybrid approach: screen-space for nearby, SDF tracing for far-field.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Lumen GI Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Scene Representation (Preprocessed):                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
│   │  Mesh SDFs  │  │ Global SDF  │  │Surface Cards│                    │
│   │ (per mesh)  │──►  (merged)   │  │  (probes)   │                    │
│   └─────────────┘  └─────────────┘  └─────────────┘                    │
│                                                                         │
│   Per-Frame:                                                            │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │              Screen Probe Placement (adaptive)                  │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│              ┌───────────────┴───────────────┐                         │
│              ▼                               ▼                         │
│   ┌─────────────────────┐         ┌─────────────────────┐              │
│   │   SSGI (near-field) │         │  SDF Trace (far)    │              │
│   │      < 2 meters     │         │    > 2 meters       │              │
│   └──────────┬──────────┘         └──────────┬──────────┘              │
│              │                               │                         │
│              └───────────────┬───────────────┘                         │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    Radiance Cache                               │  │
│   │        (screen probes + world probes + temporal filter)         │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│              ┌───────────────┴───────────────┐                         │
│              ▼                               ▼                         │
│   ┌─────────────────────┐         ┌─────────────────────┐              │
│   │  Indirect Diffuse   │         │ Indirect Specular   │              │
│   │        (GI)         │         │   (Reflections)     │              │
│   └─────────────────────┘         └─────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 13.1 Scene Representation (`src/lumen/scene/`)
| File | Description |
|------|-------------|
| `sdf_generator.rs` | Generate signed distance fields from meshes |
| `mesh_sdf.rs` | Per-mesh SDF representation |
| `global_sdf.rs` | Global merged SDF for scene |
| `sdf_brick.rs` | Brick-based SDF storage |
| `surface_cache.rs` | Cache surface lighting |
| `card.rs` | Surface card (flat light probe) |
| `card_atlas.rs` | Atlas of surface cards |

### 13.2 Ray Tracing (`src/lumen/tracing/`)
| File | Description |
|------|-------------|
| `software_raytracer.rs` | Compute shader ray tracing |
| `sdf_tracer.rs` | Ray march through SDF |
| `screen_trace.rs` | Screen-space ray tracing (SSGI) |
| `ray.rs` | Ray representation |
| `hit_result.rs` | Ray hit information |
| `cone_tracer.rs` | Cone tracing for soft effects |

### 13.3 Radiance Cache (`src/lumen/cache/`)
| File | Description |
|------|-------------|
| `screen_probe.rs` | Screen-space radiance probe |
| `screen_probe_grid.rs` | Adaptive screen probe placement |
| `world_probe.rs` | World-space radiance cache |
| `radiance_cache.rs` | Hierarchical radiance cache |
| `sh_probe.rs` | Spherical harmonics probe |
| `temporal_filter.rs` | Temporal reprojection/filter |
| `spatial_filter.rs` | Spatial denoising |

### 13.4 Lighting (`src/lumen/lighting/`)
| File | Description |
|------|-------------|
| `direct_lighting.rs` | Direct light contribution |
| `indirect_diffuse.rs` | Indirect diffuse GI |
| `indirect_specular.rs` | Indirect specular/reflections |
| `final_gather.rs` | Final gather pass |

### 13.5 Reflections (`src/lumen/reflections/`)
| File | Description |
|------|-------------|
| `ssr.rs` | Screen-space reflections |
| `hierarchical_ssr.rs` | Hi-Z SSR tracing |
| `sdf_reflections.rs` | SDF-traced reflections |
| `reflection_denoiser.rs` | Reflection denoising |

### 13.6 Shadows (`src/lumen/shadows/`)
| File | Description |
|------|-------------|
| `ray_traced_shadows.rs` | RT soft shadows |
| `distance_field_shadows.rs` | SDF soft shadows |
| `contact_shadows.rs` | Screen-space contact shadows |

---

## Phase 14: Advanced Rendering

### 14.0 Debug Visualization / Render Modes ✅
| Mode | Description | Status |
|------|-------------|--------|
| Lit | Full PBR lighting | ✅ |
| Unlit | Base color with simple lighting | ✅ |
| Normals | World-space normal visualization | ✅ |
| Depth | Linear depth visualization | ✅ |
| Metallic | Metallic factor visualization | ✅ |
| Roughness | Roughness factor visualization | ✅ |
| AO | Ambient occlusion visualization | ✅ |
| UVs | UV coordinate visualization | ✅ |
| Flat | Clay/matte shading | ✅ |
| Wireframe | True wireframe (barycentric coords) | ✅ |

### 14.1 Rendering Pipelines (`src/renderers/`)
| File | Description |
|------|-------------|
| `forward_renderer.rs` | Standard forward rendering |
| `deferred_renderer.rs` | Deferred shading |
| `render_info.rs` | Render statistics |
| `render_lists.rs` | Opaque/transparent sorting |
| `g_buffer.rs` | G-Buffer management |

### 14.2 Environment (`src/environment/`)
| File | Description | Status |
|------|-------------|--------|
| `environment_map.rs` | Environment cubemap | |
| `pmrem_generator.rs` | Prefiltered mipmap env | ✅ (in ibl/prefilter.rs) |
| `spherical_harmonics.rs` | SH for irradiance | |
| `skybox.rs` | Skybox rendering | ✅ |
| `procedural_sky.rs` | Procedural sky | |

---

## Phase 14.5: Realistic PBR & IBL Quality

This phase focuses on achieving Sketchfab-level rendering quality through proper IBL implementation and PBR enhancements.

### 14.5.1 Pre-filtered Environment Maps (`src/ibl/`) ✅
| File | Description | Status |
|------|-------------|--------|
| `mod.rs` | IBL module exports | ✅ |
| `prefilter.rs` | Generate pre-convolved env maps with GGX importance sampling | ✅ |
| `irradiance.rs` | Diffuse irradiance convolution | ✅ |
| HDR loader | Load HDR/EXR environment maps | ✅ (in loaders/hdr_loader.rs) |
| Cubemap converter | Convert equirect to cubemap with HDR support | ✅ (in hdr_loader.rs) |

### 14.5.2 BRDF Look-Up Table (`src/texture/`)
| File | Description | Status |
|------|-------------|--------|
| `brdf_lut.rs` | Pre-compute BRDF integration LUT | ✅ |
| Split-sum approximation | Implemented in pbr_textured.wgsl | ✅ |

### 14.5.3 Spherical Harmonics (`src/ibl/`)
| File | Description | Status |
|------|-------------|--------|
| `sh_coefficients.rs` | 9-coefficient SH representation (L2) | |
| `sh_projection.rs` | Project environment to SH | |
| `sh_irradiance.rs` | Evaluate SH for diffuse lighting | |
| `sh_rotation.rs` | Rotate SH coefficients | |

### 14.5.4 Energy Conservation & Multi-Scattering (`src/materials/`)
| File | Description | Status |
|------|-------------|--------|
| `energy_compensation.rs` | Multi-scattering energy compensation | |
| `kulla_conty.rs` | Kulla-Conty multi-scattering BRDF | |
| `fresnel_coat.rs` | Proper Fresnel for clearcoat | |

### 14.5.5 Ground Plane & Grounding (`src/helpers/`)
| File | Description | Status |
|------|-------------|--------|
| `shadow_catcher.rs` | Ground plane that only receives shadows | |
| `reflection_plane.rs` | Ground plane reflections | |
| `contact_shadow.rs` | Screen-space contact shadows | |
| `ao_ground.rs` | Ambient occlusion ground contact | |

### 14.5.6 HDR Pipeline Improvements (`src/postprocessing/`)
| File | Description | Status |
|------|-------------|--------|
| `tonemapping_pass.rs` | ACES/AgX/Uncharted2 filmic tonemapping | ✅ |
| `auto_exposure.rs` | Automatic exposure based on luminance | |
| `histogram_exposure.rs` | Histogram-based exposure | |
| `bloom_pass.rs` | Proper HDR bloom with threshold | ✅ |

### 14.5.7 Advanced Shadows (`src/shadows/`)
| File | Description | Status |
|------|-------------|--------|
| `shadow_config.rs` | PCSS + Contact shadow config | ✅ |
| `pbr_textured.wgsl` | PCSS + Contact shadow shaders | ✅ |
| `contact_shadows.rs` | (implemented in pbr_textured.wgsl) | ✅ |
| `shadow_denoiser.rs` | Temporal shadow denoising | |

### 14.5.8 Advanced Ambient Occlusion (`src/postprocessing/effects/`)
| File | Description | Status |
|------|-------------|--------|
| `gtao_pass.rs` | Ground Truth Ambient Occlusion | ✅ |
| `ssgi_pass.rs` | Screen-Space Global Illumination | |
| `bent_normals.rs` | Bent normals for improved AO | |

### 14.5.9 Clustered Forward Rendering (`src/renderers/`)
| File | Description | Status |
|------|-------------|--------|
| `clustered_forward.rs` | 3D tile/cluster light assignment | |
| `light_culling.rs` | GPU light culling compute shader | |
| `light_buffer.rs` | Storage buffer for dynamic lights | |
| `cluster_builder.rs` | Build frustum clusters | |

### 14.5.10 Shaders (`src/shaders/`)
| File | Description | Status |
|------|-------------|--------|
| `pbr_textured.wgsl` | Full IBL with BRDF LUT + env map | ✅ |
| `sh_lighting.wgsl` | Spherical harmonics lighting | |
| `prefilter_env.wgsl` | Environment prefiltering shader | |
| `pcss.wgsl` | (implemented in pbr_textured.wgsl) | ✅ |
| `contact_shadows.wgsl` | (implemented in pbr_textured.wgsl) | ✅ |
| `gtao.wgsl` | (implemented inline in gtao_pass.rs) | ✅ |
| `clustered_lighting.wgsl` | Clustered forward light loop | |

### Key Implementation Notes

**BRDF LUT ✅ IMPLEMENTED**
```
- 128x128 RGBA8 texture (fast generation)
- Indexed by (NdotV, roughness)
- Returns (scale, bias) for F0 in R,G channels
- Pre-computed at startup in src/texture/brdf_lut.rs
- Used in pbr_textured.wgsl split-sum approximation
```

**Pre-filtered Environment Maps ✅ IMPLEMENTED**
```
- 6 mip levels (64x64 base, configurable)
- Mip 0 = mirror reflection (roughness 0)
- Higher mips = blurrier (higher roughness)
- GGX importance sampling with Hammersley sequence
- Procedural sky or HDR/EXR file input
- src/ibl/prefilter.rs - CPU-side convolution
```

**HDR/EXR Skybox Loading ✅ IMPLEMENTED**
```
- Supports Radiance HDR (.hdr) and OpenEXR (.exr) formats
- Equirectangular to cubemap conversion
- Automatic prefiltering for IBL
- ACES tonemapping for LDR display
- src/loaders/hdr_loader.rs
```

**HDR Display Output ✅ IMPLEMENTED**
```
- Auto-detects HDR display capability
- Supports Rgba16Float (16-bit) and Rgb10a2Unorm (10-bit)
- Runtime toggle between SDR and HDR output
- Automatic tonemapping adjustment (AgX for HDR, ACES for SDR)
- Full 16-bit float render pipeline (no banding)
- Dithering in tonemapping shader for 8-bit output
API:
  app.has_hdr_display()           // Check availability
  app.get_hdr_display_format()    // "Rgba16Float" or "Rgb10a2Unorm"
  app.set_hdr_output_enabled(true) // Enable HDR output
```

**Irradiance Maps ✅ IMPLEMENTED**
```
- Pre-convolved cubemap for diffuse IBL
- Low-res (32x32) since irradiance is low-frequency
- Cosine-weighted hemisphere sampling
- Proper integration: E(n) = ∫_Ω L(ω) * max(0, n·ω) dω
- Auto-regenerated when HDR skybox is loaded
- src/ibl/irradiance.rs - CPU-side convolution
- Replaces crude "sample highest mip" approximation
```

**PCSS (Percentage-Closer Soft Shadows) ✅ IMPLEMENTED**
```
- Variable penumbra based on blocker distance
- Realistic soft shadows that get softer further from occluder
- Three-step algorithm:
  1. Blocker search (16 Poisson samples)
  2. Penumbra estimation from similar triangles
  3. PCF with variable filter radius
- Interleaved gradient noise for sample rotation (no banding)
- Configurable light size (larger = softer shadows)
- Configurable max filter radius
- src/shadows/shadow_config.rs - PCSSConfig struct
- src/shaders/pbr_textured.wgsl - PCSS shader functions
API:
  app.set_shadow_pcf_mode(5)          // Enable PCSS
  app.set_pcss_light_size(0.5)        // Light size (0.1-10.0)
  app.set_pcss_max_filter_radius(10)  // Max blur radius
```

**Contact Shadows (Screen-Space Ray Marching) ✅ IMPLEMENTED**
```
- Fine shadow detail at object contact points
- Screen-space ray marching toward light
- 8-step ray march for performance
- Complements shadow maps for close-range detail
- Configurable distance, thickness, and intensity
- src/shadows/shadow_config.rs - ContactShadowConfig struct
- src/shaders/pbr_textured.wgsl - calculate_contact_shadow function
API:
  app.set_contact_shadows_enabled(true)   // Enable contact shadows
  app.set_contact_shadow_distance(0.5)    // Max ray distance (0.1-2.0)
  app.set_contact_shadow_thickness(0.05)  // Object thickness (0.01-0.5)
  app.set_contact_shadow_intensity(0.5)   // Shadow intensity (0-1)
```

**GTAO (Ground Truth Ambient Occlusion) ✅ IMPLEMENTED**
```
- Horizon-based AO with ground truth visibility integral
- More physically accurate than traditional SSAO
- Multi-direction ray marching (2-8 directions based on quality)
- Variable step count per direction (4-12 steps)
- Edge-aware spatial blur for denoising
- Configurable radius, intensity, power, and falloff
- src/postprocessing/effects/gtao_pass.rs - GtaoPass struct
- Inline WGSL shaders for GTAO, spatial filter, and composite
API:
  app.set_gtao_enabled(true)      // Enable GTAO (disables SSAO)
  app.set_gtao_quality(1)         // 0=Low, 1=Medium, 2=High, 3=Ultra
  app.set_gtao_radius(0.5)        // Effect radius (0.1-2.0)
  app.set_gtao_intensity(1.5)     // Intensity (0.5-3.0)
  app.set_gtao_power(1.5)         // Power/contrast (0.5-4.0)
  app.set_gtao_falloff(0.2)       // Falloff start (0.0-1.0)
```

**Spherical Harmonics**
```
- L2 (9 coefficients) sufficient for diffuse
- Much faster than cubemap sampling
- Smooth, no aliasing
- Easy to blend/interpolate
```

**Ground Plane**
```
- Essential for "grounding" objects
- Shadow-only material (alpha = shadow)
- Optional reflection (SSR or planar)
- Contact shadows at object base
```

### Priority Order for Realism

| Priority | Feature | Impact | Effort | Status |
|----------|---------|--------|--------|--------|
| 1 | Pre-filtered Environment Maps | High | Medium | ✅ Done |
| 2 | HDR Display Output (10-bit/16-bit) | High | Medium | ✅ Done |
| 3 | HDR/EXR Skybox Loading | High | Medium | ✅ Done |
| 4 | Irradiance Map (diffuse IBL) | High | Medium | ✅ Done |
| 5 | PCSS Soft Shadows | Medium | Low | ✅ Done |
| 6 | Contact Shadows | Medium | Low | ✅ Done |
| 7 | GTAO | Medium | Medium | ✅ Done |
| 8 | Spherical Harmonics | Medium | Medium | |
| 9 | Clustered Forward | High (many lights) | High | |
| 10 | Auto Exposure | Low | Low | |

---

## Phase 15: Optimization

### 15.1 Culling (`src/culling/`)
| File | Description |
|------|-------------|
| `frustum_culler.rs` | View frustum culling |
| `occlusion_culler.rs` | Occlusion culling |
| `bvh.rs` | Bounding volume hierarchy |
| `octree.rs` | Spatial octree |

### 15.2 Batching (`src/batching/`)
| File | Description |
|------|-------------|
| `static_batch.rs` | Static geometry batching |
| `instance_batch.rs` | GPU instancing manager |
| `indirect_draw.rs` | Indirect draw calls |

---

## Phase 16: Physics Integration

### 16.1 Physics Bridge (`src/physics/`)
| File | Description |
|------|-------------|
| `physics_world.rs` | Physics simulation wrapper |
| `rigid_body.rs` | Rigid body component |
| `collider.rs` | Collision shape base |
| `box_collider.rs` | Box collision shape |
| `sphere_collider.rs` | Sphere collision |
| `capsule_collider.rs` | Capsule collision |
| `mesh_collider.rs` | Mesh collision |
| `character_controller.rs` | Character physics |

---

## Phase 17: Audio

### 17.1 Audio System (`src/audio/`)
| File | Description |
|------|-------------|
| `audio_context.rs` | Web Audio context wrapper |
| `audio_listener.rs` | 3D audio listener |
| `audio.rs` | Non-positional audio |
| `positional_audio.rs` | 3D positioned sound |
| `audio_loader.rs` | Audio file loading |

---

## Phase 18: Particles

### 18.1 Particle System (`src/particles/`)
| File | Description |
|------|-------------|
| `particle_system.rs` | Particle system manager |
| `particle.rs` | Single particle |
| `emitter.rs` | Base emitter |
| `point_emitter.rs` | Point emission |
| `box_emitter.rs` | Box volume emission |
| `sphere_emitter.rs` | Sphere volume emission |
| `gpu_particle_system.rs` | GPU-computed particles |

---

## Phase 19: VR/AR Support

### 19.1 XR (`src/xr/`)
| File | Description |
|------|-------------|
| `xr_manager.rs` | WebXR session management |
| `xr_controller.rs` | XR input controller |
| `xr_hand.rs` | Hand tracking |

---

## Phase 20: JavaScript API Layer

### 20.1 Core Bindings (`js/src/core/`)
| File | Description |
|------|-------------|
| `Engine.ts` | Main engine class |
| `Scene.ts` | Scene wrapper |
| `Renderer.ts` | Renderer wrapper |
| `Object3D.ts` | Base 3D object |
| `Clock.ts` | Time utilities |

All Rust modules mirrored with TypeScript classes.

---

## Phase 21: CLI Tools

### 21.1 Asset Processing (`ren-cli/`)
| File | Description |
|------|-------------|
| `main.rs` | CLI entry point |
| `commands/preprocess.rs` | Nanite preprocessing |
| `commands/sdf.rs` | SDF generation |
| `commands/compress.rs` | Texture compression |
| `commands/validate.rs` | Asset validation |

---

## File Count Summary

| Category | Files |
|----------|-------|
| Math | 16 |
| Core | 12 |
| Scene | 8 |
| Buffers | 5 |
| Geometries | 21 |
| Curves | 7 |
| Materials | 14 |
| Shaders | 4 |
| Textures | 10 |
| Loaders | 11 |
| Lights | 7 |
| Shadows | 6 |
| Cameras | 4 |
| Controls | 6 |
| Objects | 10 |
| Helpers | 7 |
| Animation | 13 |
| Post-Process | 17 |
| Threading | 21 |
| Nanite | 32 |
| Lumen | 28 |
| Advanced Render | 9 |
| Optimization | 7 |
| Physics | 8 |
| Audio | 5 |
| Particles | 7 |
| XR | 3 |
| CLI | 5 |
| **Total Rust** | **~300 files** |
| **JS Bindings** | **~100 files** |

---

## Priority Order

```
Phase 1   ████████████████████  Foundation ✅
Phase 2   ████████████████████  Geometry ✅
Phase 3   ████████████████████  Materials ✅
Phase 4   ████████████████████  Textures ✅
Phase 5   ████████████████████  Lighting ✅
Phase 6   ████████████████████  Cameras ✅
Phase 7   ████████████████      Objects ✅ (partial)
Phase 8   ████████████████████  Animation ✅
Phase 9   ████████████████████  Loaders ✅
Phase 10  ████████████████████  Post-Process ✅
Phase 11  ████████████████████  SharedArrayBuffer (enables parallelism)
Phase 12  ████████████████      Nanite (core differentiator)
Phase 13  ████████████████████  Lumen ✅ (SSGI implemented)
Phase 14  ████████████████      Advanced Render ✅ (partial - IBL, shadows, debug modes)
Phase 15  ██████████            Optimization
Phase 16  ████████              Physics
Phase 17  ██████                Audio
Phase 18  ██████                Particles
Phase 19  ████                  VR/AR
Phase 20  ████████████████████  JS API (parallel with Rust)
Phase 21  ████████              CLI Tools
```
