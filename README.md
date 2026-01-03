# Ren - WebGPU 3D Engine

A high-performance 3D engine built with Rust, wgpu, and WebAssembly featuring Nanite-style virtualized geometry and Lumen-style global illumination.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         JavaScript API Layer                              │
│  (wasm-bindgen bindings for web integration)                             │
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
- [x] `index.html` - Test page with full UI controls
- [ ] `package.json` - JS library bundling (future)
- [ ] `tsconfig.json` - TypeScript definitions (future)

### 1.2 Math Library (`src/math/`) ✅
| File | Description | Status |
|------|-------------|--------|
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
| `engine.rs` | Main engine entry point | ✅ |
| `context.rs` | wgpu device/queue/surface management | ✅ |
| `renderer.rs` | WebGPU render pipeline orchestration | ✅ |
| `clock.rs` | Delta time, elapsed time | ✅ |
| `id.rs` | Unique ID generation | ✅ |
| `render_state.rs` | Current render pass state | |
| `bind_group_manager.rs` | wgpu bind group caching | |
| `pipeline_cache.rs` | Compiled pipeline caching | |

### 1.4 Scene Graph (`src/scene/`) ✅
| File | Description | Status |
|------|-------------|--------|
| `object3d.rs` | Base class for all 3D objects | ✅ |
| `scene.rs` | Root scene container | ✅ |
| `transform.rs` | Position/rotation/scale component | ✅ |
| `visibility.rs` | Visibility and culling flags | ✅ |
| `group.rs` | Empty transform node | |
| `fog.rs` | Linear/exponential fog | |

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
| `cone_geometry.rs` | Cone with base | |
| `capsule_geometry.rs` | Capsule/stadium shape | |

### 2.3 Camera System (`src/camera/`) ✅
| File | Description | Status |
|------|-------------|--------|
| `perspective.rs` | Perspective projection camera | ✅ |
| `orthographic.rs` | Orthographic projection camera | ✅ |

---

## Phase 3: Materials & Shaders ✅

### 3.1 Material System (`src/material/`)
| File | Description | Status |
|------|-------------|--------|
| `basic.rs` | Unlit solid color/texture | ✅ |
| `standard.rs` | Standard lit material | ✅ |
| `pbr.rs` | PBR metallic-roughness | ✅ |
| `pbr_textured.rs` | Full PBR with textures | ✅ |
| `line.rs` | Line material | ✅ |
| `physical_material.rs` | Extended PBR (clearcoat, sheen) | ✅ |

### 3.2 PBR Features (`src/shaders/pbr_textured.wgsl`)
| Feature | Description | Status |
|---------|-------------|--------|
| Cook-Torrance BRDF | GGX + Schlick-GGX | ✅ |
| Metallic/Roughness | Standard PBR workflow | ✅ |
| Normal Mapping | Tangent-space normals | ✅ |
| Clear Coat | Two-layer model | ✅ |
| Cloth/Sheen BRDF | Inverse GGX + Charlie sheen | ✅ |
| Energy Normalization | Area light compensation | ✅ |
| Rough Diffuse | Oren-Nayar approximation | ✅ |
| Detail Normal Maps | Procedural micro-surface | ✅ |
| Detail Albedo Maps | Color variation with blend modes | ✅ |

### 3.3 Shader System (`src/shaders/`)
| File | Description | Status |
|------|-------------|--------|
| `basic.wgsl` | Basic unlit shader | ✅ |
| `standard.wgsl` | Standard lighting shader | ✅ |
| `pbr.wgsl` | PBR metallic-roughness shader | ✅ |
| `pbr_textured.wgsl` | Full PBR with IBL, shadows | ✅ |
| `pbr_lit.wgsl` | PBR with dynamic lighting | ✅ |
| `line.wgsl` | Line rendering shader | ✅ |
| `skybox.wgsl` | Skybox shader | ✅ |
| `procedural_sky.wgsl` | Atmospheric scattering | ✅ |

---

## Phase 4: Textures & Images ✅

### 4.1 Texture System (`src/texture/`)
| File | Description | Status |
|------|-------------|--------|
| `texture2d.rs` | Standard 2D texture | ✅ |
| `sampler.rs` | Texture sampling parameters | ✅ |
| `cube_texture.rs` | Cubemap texture | ✅ |
| `brdf_lut.rs` | BRDF integration LUT for IBL | ✅ |
| `detail_normal_map.rs` | Procedural detail normals | ✅ |
| `detail_albedo.rs` | Procedural detail albedo | ✅ |
| `texture3d.rs` | 3D/volume texture | |
| `compressed_texture.rs` | GPU compressed formats | |

### 4.2 Loaders (`src/loaders/`)
| File | Description | Status |
|------|-------------|--------|
| `gltf_loader.rs` | GLTF/GLB models with textures | ✅ |
| `obj_loader.rs` | Wavefront OBJ | ✅ |
| `hdr_loader.rs` | HDR/EXR environment maps | ✅ |
| `ktx2_loader.rs` | KTX2 compressed textures | |

---

## Phase 5: Lighting System ✅

### 5.1 Light Types (`src/light/`)
| Type | Description | Status |
|------|-------------|--------|
| Point Light | Omni-directional with falloff | ✅ |
| Directional Light | Parallel rays (sun) | ✅ |
| Spot Light | Cone with inner/outer angles | ✅ |
| Hemisphere Light | Sky/ground gradient ambient | ✅ |
| Rect Light | Rectangular area light (LTC) | ✅ |
| Capsule Light | Tube/neon area light | ✅ |
| Disk Light | Circular area light | ✅ |
| Sphere Light | Spherical area light | ✅ |

### 5.2 Shadows (`src/shadows/`) ✅
| Feature | Description | Status |
|---------|-------------|--------|
| Shadow Maps | Basic shadow mapping | ✅ |
| Cascaded Shadows | CSM for directional lights | ✅ |
| Point Shadows | Omnidirectional (cubemap) | ✅ |
| Spot Shadows | Perspective shadow maps | ✅ |
| PCF Filtering | 2x2, 3x3, 5x5, Poisson | ✅ |
| PCSS | Variable penumbra soft shadows | ✅ |
| Contact Shadows | Screen-space ray marching | ✅ |

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

### 6.3 Camera Controls
| Input | Action |
|-------|--------|
| LMB + drag | Forward/back movement + yaw turning |
| RMB + drag | Freelook (rotate view in place) |
| RMB + WASD | Fly movement (while RMB held) |
| RMB + Q/E | Fly up/down |
| RMB + Shift | Speed boost (3x faster) |
| MMB + drag | Pan |
| Scroll wheel | Dolly forward/back |
| F | Focus on origin |
| 1/2/3 | Transform gizmo modes |

---

## Phase 7: Objects & Renderables ✅

### 7.1 Objects (`src/objects/`)
| File | Description | Status |
|------|-------------|--------|
| `mesh.rs` | Geometry + material renderable | ✅ |
| `instanced_mesh.rs` | GPU instanced mesh | ✅ |
| `line.rs` | Line rendering | ✅ |
| `skinned_mesh.rs` | Animated mesh with skeleton | |
| `sprite.rs` | Billboard sprite | |

### 7.2 Helpers (`src/helpers/`)
| File | Description | Status |
|------|-------------|--------|
| `axes_helper.rs` | XYZ axes visualization | ✅ |
| `grid_helper.rs` | Ground grid | ✅ |
| `box_helper.rs` | Bounding box wireframe | ✅ |
| `transform_gizmo.rs` | Translate/rotate/scale gizmo | ✅ |
| `arrow_helper.rs` | Direction arrow | |
| `light_helper.rs` | Light visualization | |

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

### 8.2 Interpolation (`src/animation/interpolants/`)
| File | Description | Status |
|------|-------------|--------|
| `linear_interpolant.rs` | Linear interpolation | ✅ |
| `discrete_interpolant.rs` | Step/discrete | ✅ |
| `cubic_interpolant.rs` | Cubic spline | ✅ |

---

## Phase 9: Model Loaders ✅

| Format | Description | Status |
|--------|-------------|--------|
| GLTF/GLB | Full 2.0 with materials and textures | ✅ |
| OBJ | Wavefront with MTL | ✅ |
| HDR | Radiance HDR for IBL | ✅ |
| EXR | OpenEXR for HDR environments | ✅ |
| Draco | Draco mesh compression | |

---

## Phase 10: Post-Processing ✅

### 10.1 Effects (`src/postprocessing/effects/`)
| Effect | Description | Status |
|--------|-------------|--------|
| Bloom | HDR glow with threshold | ✅ |
| Tonemapping | Reinhard, ACES, AgX, Uncharted2 | ✅ |
| Auto Exposure | Histogram-based eye adaptation | ✅ |
| Color Correction | Brightness, contrast, saturation, gamma, temperature | ✅ |
| FXAA | Fast approximate AA | ✅ |
| SMAA | Subpixel morphological AA | ✅ |
| TAA | Temporal AA with jitter | ✅ |
| SSAO | Screen-space ambient occlusion | ✅ |
| GTAO | Ground-truth ambient occlusion | ✅ |
| SSR | Screen-space reflections | ✅ |
| Depth of Field | Bokeh with highlight boost | ✅ |
| Motion Blur | Per-pixel velocity blur | ✅ |
| Volumetric Fog | God rays with shadows | ✅ |
| Vignette | Screen edge darkening | ✅ |
| Outline | Edge detection (depth/normal) | ✅ |
| Lumen GI | Screen-space global illumination | ✅ |
| Procedural Sky | Rayleigh/Mie scattering | ✅ |
| Skybox | HDR cubemap environment | ✅ |
| HDR Output | 10-bit/16-bit display support | ✅ |

---

## Phase 11: Image-Based Lighting ✅

### 11.1 IBL System (`src/ibl/`)
| Feature | Description | Status |
|---------|-------------|--------|
| Prefiltered Env Maps | GGX importance sampling | ✅ |
| Irradiance Maps | Diffuse IBL convolution | ✅ |
| BRDF LUT | Split-sum approximation | ✅ |
| HDR Skybox Loading | HDR/EXR to cubemap | ✅ |
| HDR Display Output | 10-bit/16-bit support | ✅ |

---

## Phase 12: Nanite - Virtualized Geometry ✅

GPU-driven rendering pipeline for high-polygon meshes.

### 12.1 Core Pipeline (`src/nanite/`)
| Feature | Description | Status |
|---------|-------------|--------|
| Mesh Clustering | 128-triangle clusters | ✅ |
| Cluster Hierarchy | LOD DAG structure | ✅ |
| GPU Frustum Culling | Compute shader culling | ✅ |
| HZB Occlusion | Hierarchical Z-buffer | ✅ |
| Visibility Buffer | Deferred material eval | ✅ |
| Software Rasterization | Small triangle optimization | ✅ |
| Material Pass | PBR shading from vis buffer | ✅ |
| Shadow Pass | Nanite shadow rendering | ✅ |

### 12.2 Limits
| Resource | Limit |
|----------|-------|
| Max Clusters | 16,384 |
| Max Vertices | 4,000,000 |
| Max Indices | 8,000,000 |
| Max Instances | 1,024 |

### 12.3 Shaders (`src/shaders/nanite_*.wgsl`)
| Shader | Description | Status |
|--------|-------------|--------|
| `nanite_visibility.wgsl` | Visibility buffer write | ✅ |
| `nanite_material.wgsl` | Material shading pass | ✅ |
| `nanite_frustum_cull.wgsl` | Frustum culling compute | ✅ |
| `nanite_occlusion_cull.wgsl` | HZB occlusion compute | ✅ |
| `nanite_hzb_build.wgsl` | Depth pyramid build | ✅ |
| `nanite_sw_raster.wgsl` | Software rasterization | ✅ |
| `nanite_classify.wgsl` | Triangle classification | ✅ |
| `nanite_shadow.wgsl` | Shadow map rendering | ✅ |

---

## Phase 13: Lumen - Global Illumination ✅

### 13.1 SSGI Implementation
| Feature | Description | Status |
|---------|-------------|--------|
| SSGI Pass | Hemisphere ray tracing | ✅ |
| Denoise Pass | Edge-aware bilateral blur | ✅ |
| Temporal Pass | 95% history blend | ✅ |
| Quality Presets | Low (4) to Ultra (16 rays) | ✅ |

### 13.2 Irradiance Probes (`src/gi/`)
| Feature | Description | Status |
|---------|-------------|--------|
| Probe Volume | 3D grid of probes | ✅ |
| SH Storage | L2 spherical harmonics | ✅ |
| Probe Sampling | Trilinear interpolation | ✅ |
| Visibility Weighting | Occlusion-aware blending | ✅ |

### 13.3 Future (SDF-Based)
| Feature | Description | Status |
|---------|-------------|--------|
| Mesh SDFs | Per-mesh signed distance fields | |
| Global SDF | Merged scene SDF | |
| SDF Tracing | Ray march through SDF | |
| Surface Cards | Flat light probes | |

---

## Phase 14: Particles ✅

### 14.1 GPU Particle System (`src/particles/`)
| Feature | Description | Status |
|---------|-------------|--------|
| Compute Simulation | GPU particle physics | ✅ |
| Instanced Rendering | Efficient draw calls | ✅ |
| Soft Particles | Depth-based fade | ✅ |

### 14.2 Emitter Shapes
| Shape | Status |
|-------|--------|
| Point | ✅ |
| Sphere Surface | ✅ |
| Sphere Volume | ✅ |
| Cone | ✅ |
| Box | ✅ |

### 14.3 Forces & Effects
| Feature | Status |
|---------|--------|
| Gravity | ✅ |
| Drag | ✅ |
| Turbulence | ✅ |
| Vortex | ✅ |
| Additive Blend | ✅ |
| Alpha Blend | ✅ |

### 14.4 Presets
| Preset | Status |
|--------|--------|
| Fire | ✅ |
| Smoke | ✅ |
| Sparks | ✅ |
| Magic | ✅ |

---

## Phase 15: Debug Visualization ✅

| Mode | Description | Status |
|------|-------------|--------|
| Lit | Full PBR rendering | ✅ |
| Unlit | Base color only | ✅ |
| Normals | World-space normals | ✅ |
| Depth | Linear depth buffer | ✅ |
| Metallic | Metallic values | ✅ |
| Roughness | Roughness values | ✅ |
| AO | Ambient occlusion | ✅ |
| UVs | UV coordinates | ✅ |
| Flat | Clay/matte shading | ✅ |
| Wireframe | Polygon edges | ✅ |

---

## Phase 16: Multi-Threading (Future)

### 16.1 Thread Infrastructure (`src/threading/`)
| File | Description |
|------|-------------|
| `thread_pool.rs` | Rayon-style thread pool for WASM |
| `worker_pool.rs` | Web Worker pool management |
| `shared_memory.rs` | SharedArrayBuffer wrapper |
| `channel.rs` | Lock-free SPSC/MPSC channels |

---

## Phase 17: Physics Integration (Future)

### 17.1 Physics Bridge (`src/physics/`)
| File | Description |
|------|-------------|
| `physics_world.rs` | Physics simulation wrapper |
| `rigid_body.rs` | Rigid body component |
| `collider.rs` | Collision shapes |
| `character_controller.rs` | Character physics |

---

## Phase 18: Audio (Future)

### 18.1 Audio System (`src/audio/`)
| File | Description |
|------|-------------|
| `audio_context.rs` | Web Audio context wrapper |
| `audio_listener.rs` | 3D audio listener |
| `positional_audio.rs` | 3D positioned sound |

---

## Phase 19: VR/AR Support (Future)

### 19.1 XR (`src/xr/`)
| File | Description |
|------|-------------|
| `xr_manager.rs` | WebXR session management |
| `xr_controller.rs` | XR input controller |
| `xr_hand.rs` | Hand tracking |

---

## Phase 20: JavaScript API Layer (Future)

### 20.1 TypeScript Bindings (`js/src/`)
| File | Description |
|------|-------------|
| `Engine.ts` | Main engine class |
| `Scene.ts` | Scene wrapper |
| `Renderer.ts` | Renderer wrapper |
| `Object3D.ts` | Base 3D object |

---

## Building

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for web
wasm-pack build --target web

# Output in pkg/
```

## Usage

```html
<canvas id="canvas"></canvas>
<script type="module">
  import init, { RenApp } from './pkg/ren.js';

  await init();
  const app = await RenApp.new('canvas');

  // Load a model
  const response = await fetch('model.glb');
  const data = new Uint8Array(await response.arrayBuffer());
  app.load_gltf(data);

  // Configure lighting
  app.set_light_enabled(0, true);
  app.set_light_position(0, 5.0, 8.0, 5.0);
  app.set_light_intensity(0, 15.0);

  // Enable effects
  app.set_bloom_enabled(true);
  app.set_ssao_enabled(true);

  // Render loop
  function frame() {
    app.frame();
    requestAnimationFrame(frame);
  }
  frame();
</script>
```

## API Reference

### Materials
```javascript
app.set_clear_coat(value)           // 0.0 - 1.0
app.set_clear_coat_roughness(value) // 0.0 - 0.5
app.set_sheen(value)                // 0.0 - 1.0
app.set_sheen_color(r, g, b)        // RGB 0.0 - 1.0
app.set_material_metallic(value)    // 0.0 - 1.0
app.set_material_roughness(value)   // 0.04 - 1.0
```

### Detail Textures
```javascript
// Detail Normal (micro-surface variation)
app.set_detail_normal_enabled(bool)
app.set_detail_normal_scale(value)      // UV tiling scale
app.set_detail_normal_intensity(value)  // 0.0 - 1.0

// Detail Albedo (color variation)
app.set_detail_albedo_enabled(bool)
app.set_detail_albedo_scale(value)      // UV tiling scale
app.set_detail_albedo_intensity(value)  // 0.0 - 1.0
app.set_detail_albedo_blend_mode(mode)  // 0=Overlay, 1=Multiply, 2=Soft Light
```

### Lighting
```javascript
app.set_light_enabled(index, bool)
app.set_light_position(index, x, y, z)
app.set_light_color(index, r, g, b)
app.set_light_intensity(index, value)
app.set_hemisphere_light(skyR, skyG, skyB, groundR, groundG, groundB, intensity)
```

### Shadows
```javascript
app.set_shadows_enabled(bool)
app.set_shadow_resolution(512|1024|2048|4096)
app.set_shadow_pcf_mode(0-5)         // 0=Hard, 5=PCSS
app.set_pcss_light_size(value)
app.set_contact_shadows_enabled(bool)
```

### Post-Processing
```javascript
app.set_bloom_enabled(bool)
app.set_tonemapping_mode(0-5)        // 0=Linear, 3=ACES, 5=AgX
app.set_exposure(value)
app.set_auto_exposure_enabled(bool)
app.set_ssao_enabled(bool)
app.set_ssr_enabled(bool)
app.set_dof_enabled(bool)
app.set_motion_blur_enabled(bool)
app.set_volumetric_fog_enabled(bool)
app.set_lumen_enabled(bool)
```

### Environment
```javascript
app.load_hdr_skybox(data)
app.set_procedural_sky_enabled(bool)
app.set_sun_elevation(degrees)
app.set_sun_azimuth(degrees)
app.set_ibl_intensity(diffuse, specular)
```

### Nanite
```javascript
app.set_nanite_enabled(bool)
app.load_gltf_nanite(data)
app.nanite_cluster_count()
```

### Particles
```javascript
app.add_particle_system(preset)      // "fire", "smoke", "sparks"
app.set_particle_emitter_shape(id, shape)
app.set_particle_emission_rate(id, rate)
```

### Transform Gizmo
```javascript
app.set_gizmo_enabled(bool)
app.set_gizmo_mode(mode)            // 0=Translate, 1=Rotate, 2=Scale
app.is_gizmo_enabled()
```

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation & Core | ✅ Complete |
| 2 | Geometry System | ✅ Complete |
| 3 | Materials & Shaders | ✅ Complete |
| 4 | Textures & Images | ✅ Complete |
| 5 | Lighting System | ✅ Complete |
| 6 | Cameras & Controls | ✅ Complete |
| 7 | Objects & Renderables | ✅ Complete |
| 8 | Animation System | ✅ Complete |
| 9 | Model Loaders | ✅ Complete |
| 10 | Post-Processing | ✅ Complete (19 effects) |
| 11 | Image-Based Lighting | ✅ Complete |
| 12 | Nanite Virtualized Geometry | ✅ Complete |
| 13 | Lumen Global Illumination | ✅ SSGI + Probes |
| 14 | Particle System | ✅ Complete |
| 15 | Debug Visualization | ✅ Complete |
| 16 | Multi-Threading | Planned |
| 17 | Physics | Planned |
| 18 | Audio | Planned |
| 19 | VR/AR | Planned |
| 20 | JavaScript API | Planned |

## License

MIT
