//! PBR (Physically Based Rendering) uniform structures.

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
    /// Detail albedo: x=enabled (0/1), y=scale, z=intensity, w=blend_mode (0=overlay, 1=multiply, 2=soft_light)
    pub detail_albedo_settings: [f32; 4],
    /// Rect light 0: position.xyz + enabled.w
    pub rectlight0_pos: [f32; 4],
    /// Rect light 0: direction.xyz + width.w
    pub rectlight0_dir_width: [f32; 4],
    /// Rect light 0: tangent.xyz + height.w
    pub rectlight0_tan_height: [f32; 4],
    /// Rect light 0: color.rgb + intensity.w
    pub rectlight0_color: [f32; 4],
    /// Rect light 1: position.xyz + enabled.w
    pub rectlight1_pos: [f32; 4],
    /// Rect light 1: direction.xyz + width.w
    pub rectlight1_dir_width: [f32; 4],
    /// Rect light 1: tangent.xyz + height.w
    pub rectlight1_tan_height: [f32; 4],
    /// Rect light 1: color.rgb + intensity.w
    pub rectlight1_color: [f32; 4],
    /// Capsule light 0: start.xyz + enabled.w
    pub capsule0_start: [f32; 4],
    /// Capsule light 0: end.xyz + radius.w
    pub capsule0_end_radius: [f32; 4],
    /// Capsule light 0: color.rgb + intensity.w
    pub capsule0_color: [f32; 4],
    /// Capsule light 1: start.xyz + enabled.w
    pub capsule1_start: [f32; 4],
    /// Capsule light 1: end.xyz + radius.w
    pub capsule1_end_radius: [f32; 4],
    /// Capsule light 1: color.rgb + intensity.w
    pub capsule1_color: [f32; 4],
    /// Disk light 0: position.xyz + enabled.w
    pub disk0_pos: [f32; 4],
    /// Disk light 0: direction.xyz + radius.w
    pub disk0_dir_radius: [f32; 4],
    /// Disk light 0: color.rgb + intensity.w
    pub disk0_color: [f32; 4],
    /// Disk light 1: position.xyz + enabled.w
    pub disk1_pos: [f32; 4],
    /// Disk light 1: direction.xyz + radius.w
    pub disk1_dir_radius: [f32; 4],
    /// Disk light 1: color.rgb + intensity.w
    pub disk1_color: [f32; 4],
    /// Sphere light 0: position.xyz + enabled.w
    pub sphere0_pos: [f32; 4],
    /// Sphere light 0: radius.x + range.y (unused zw)
    pub sphere0_radius_range: [f32; 4],
    /// Sphere light 0: color.rgb + intensity.w
    pub sphere0_color: [f32; 4],
    /// Sphere light 1: position.xyz + enabled.w
    pub sphere1_pos: [f32; 4],
    /// Sphere light 1: radius.x + range.y (unused zw)
    pub sphere1_radius_range: [f32; 4],
    /// Sphere light 1: color.rgb + intensity.w
    pub sphere1_color: [f32; 4],
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
            // Detail albedo: disabled by default, scale=10, intensity=0.3, blend_mode=0 (overlay)
            detail_albedo_settings: [0.0, 10.0, 0.3, 0.0],
            // Rect lights: disabled by default
            rectlight0_pos: [0.0, 0.0, 0.0, 0.0],         // Disabled (w=0)
            rectlight0_dir_width: [0.0, 0.0, -1.0, 1.0],  // Facing -Z, width=1
            rectlight0_tan_height: [1.0, 0.0, 0.0, 1.0],  // Tangent along X, height=1
            rectlight0_color: [1.0, 1.0, 1.0, 10.0],      // White, intensity=10
            rectlight1_pos: [0.0, 0.0, 0.0, 0.0],         // Disabled (w=0)
            rectlight1_dir_width: [0.0, 0.0, -1.0, 1.0],
            rectlight1_tan_height: [1.0, 0.0, 0.0, 1.0],
            rectlight1_color: [1.0, 1.0, 1.0, 10.0],
            // Capsule lights: disabled by default
            capsule0_start: [0.0, 0.0, 0.0, 0.0],         // Disabled (w=0)
            capsule0_end_radius: [1.0, 0.0, 0.0, 0.05],   // 1 unit long, 5cm radius
            capsule0_color: [1.0, 1.0, 1.0, 10.0],        // White, intensity=10
            capsule1_start: [0.0, 0.0, 0.0, 0.0],         // Disabled (w=0)
            capsule1_end_radius: [1.0, 0.0, 0.0, 0.05],
            capsule1_color: [1.0, 1.0, 1.0, 10.0],
            // Disk lights: disabled by default
            disk0_pos: [0.0, 0.0, 0.0, 0.0],              // Disabled (w=0)
            disk0_dir_radius: [0.0, -1.0, 0.0, 0.5],      // Facing down, radius=0.5
            disk0_color: [1.0, 1.0, 1.0, 10.0],           // White, intensity=10
            disk1_pos: [0.0, 0.0, 0.0, 0.0],              // Disabled (w=0)
            disk1_dir_radius: [0.0, -1.0, 0.0, 0.5],
            disk1_color: [1.0, 1.0, 1.0, 10.0],
            // Sphere lights: disabled by default
            sphere0_pos: [0.0, 0.0, 0.0, 0.0],            // Disabled (w=0)
            sphere0_radius_range: [0.1, 20.0, 0.0, 0.0],  // radius=0.1, range=20
            sphere0_color: [1.0, 1.0, 1.0, 10.0],         // White, intensity=10
            sphere1_pos: [0.0, 0.0, 0.0, 0.0],            // Disabled (w=0)
            sphere1_radius_range: [0.1, 20.0, 0.0, 0.0],
            sphere1_color: [1.0, 1.0, 1.0, 10.0],
        }
    }
}
