// Particle Render Shader
// Billboard rendering with soft particles and texture support

const PI: f32 = 3.14159265359;

// Particle data structure (must match compute shader)
struct Particle {
    position_age: vec4<f32>,
    velocity_lifetime: vec4<f32>,
    color: vec4<f32>,
    size_rotation: vec4<f32>,
    flags: vec4<f32>,
    start_color: vec4<f32>,
    end_color: vec4<f32>,
}

// Camera uniform (matches PbrCameraUniform - 272 bytes)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    render_mode: u32,
    hemisphere_sky: vec4<f32>,
    hemisphere_ground: vec4<f32>,
    ibl_settings: vec4<f32>,
    light0_pos: vec4<f32>,
    light0_color: vec4<f32>,
    light1_pos: vec4<f32>,
    light1_color: vec4<f32>,
    light2_pos: vec4<f32>,
    light2_color: vec4<f32>,
    light3_pos: vec4<f32>,
    light3_color: vec4<f32>,
    detail_settings: vec4<f32>,
}

// Render parameters
struct RenderParams {
    resolution: vec4<f32>,     // x=width, y=height, z=1/width, w=1/height
    params: vec4<f32>,         // x=soft_fade_distance, y=tex_rows, z=tex_cols, w=unused
    camera_params: vec4<f32>,  // x=near, y=far, z=unused, w=unused
    camera_right: vec4<f32>,   // xyz=camera right vector for billboarding
    camera_up: vec4<f32>,      // xyz=camera up vector for billboarding
}

// Bind groups
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(1) @binding(1) var<storage, read> alive_list: array<u32>;
@group(1) @binding(2) var<uniform> render_params: RenderParams;

@group(2) @binding(0) var particle_texture: texture_2d<f32>;
@group(2) @binding(1) var particle_sampler: sampler;
@group(2) @binding(2) var depth_texture: texture_depth_2d;

// Vertex output
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) particle_depth: f32,
}

// Quad vertices for billboard (triangle strip order)
const QUAD_POSITIONS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-0.5, -0.5),  // bottom-left
    vec2<f32>(0.5, -0.5),   // bottom-right
    vec2<f32>(-0.5, 0.5),   // top-left
    vec2<f32>(0.5, 0.5),    // top-right
);

const QUAD_UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Get particle index from alive list
    let particle_index = alive_list[instance_index];
    let particle = particles[particle_index];

    // Check if particle is alive
    if (particle.flags.x < 0.5) {
        // Dead particle - move off screen
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        out.world_position = vec3<f32>(0.0);
        out.particle_depth = 0.0;
        return out;
    }

    let world_pos = particle.position_age.xyz;
    let size = particle.size_rotation.x;
    let rotation = particle.size_rotation.w;
    let age = particle.position_age.w;
    let lifetime = particle.velocity_lifetime.w;

    // Get quad corner
    let corner = QUAD_POSITIONS[vertex_index];

    // Apply rotation
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);
    let rotated = vec2<f32>(
        corner.x * cos_r - corner.y * sin_r,
        corner.x * sin_r + corner.y * cos_r
    );

    // Billboard: expand in camera's right and up directions
    // Use camera vectors passed via render_params
    let right = render_params.camera_right.xyz;
    let up = render_params.camera_up.xyz;

    let vertex_pos = world_pos + (right * rotated.x + up * rotated.y) * size;

    // Calculate UV with optional texture atlas
    var uv = QUAD_UVS[vertex_index];
    let tex_cols = u32(render_params.params.z);
    let tex_rows = u32(render_params.params.y);

    if (tex_cols > 1u || tex_rows > 1u) {
        // Animated texture atlas - select frame based on age
        let total_frames = tex_cols * tex_rows;
        let t = age / lifetime;
        let frame = u32(t * f32(total_frames)) % total_frames;
        let col = frame % tex_cols;
        let row = frame / tex_cols;

        let frame_size = vec2<f32>(1.0 / f32(tex_cols), 1.0 / f32(tex_rows));
        uv = uv * frame_size + vec2<f32>(f32(col), f32(row)) * frame_size;
    }

    // Age-based fade (smooth fade in at start, fade out at end)
    let t = age / lifetime;
    let fade_in = smoothstep(0.0, 0.05, t);
    let fade_out = 1.0 - smoothstep(0.85, 1.0, t);
    var color = particle.color;
    color.a *= fade_in * fade_out;

    // Calculate clip position
    out.clip_position = camera.view_proj * vec4<f32>(vertex_pos, 1.0);
    out.uv = uv;
    out.color = color;
    out.world_position = vertex_pos;

    // Store linear depth for soft particles
    // For perspective projection, clip.w equals the view-space depth
    out.particle_depth = out.clip_position.w;

    return out;
}

// Linearize depth from depth buffer
fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - depth * (far - near));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample particle texture
    var tex_color = textureSample(particle_texture, particle_sampler, in.uv);

    // Just clip the corners slightly for better blending (keeps density)
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(in.uv - center) * 2.0;
    let corner_clip = 1.0 - smoothstep(0.85, 1.0, dist);

    // Apply particle color
    var final_color = tex_color * in.color;
    final_color.a *= corner_clip;

    // Soft particle: fade based on depth difference with scene
    let soft_fade_dist = render_params.params.x;
    if (soft_fade_dist > 0.0) {
        // Get screen-space coordinates (integer pixel position)
        let screen_coords = vec2<i32>(in.clip_position.xy);

        // Load scene depth (textureLoad for depth textures, not textureSample)
        let scene_depth_raw = textureLoad(depth_texture, screen_coords, 0);

        // Linearize depths
        let near = render_params.camera_params.x;
        let far = render_params.camera_params.y;
        let linear_scene_depth = linearize_depth(scene_depth_raw, near, far);

        // Calculate fade based on depth difference
        let depth_diff = linear_scene_depth - in.particle_depth;
        let soft_fade = saturate(depth_diff / soft_fade_dist);

        final_color.a *= soft_fade;
    }

    // Discard fully transparent pixels
    if (final_color.a < 0.01) {
        discard;
    }

    return final_color;
}

// Alternative fragment shader for additive blending (pre-multiplied alpha)
@fragment
fn fs_main_additive(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample particle texture
    var tex_color = textureSample(particle_texture, particle_sampler, in.uv);

    // Just clip the corners slightly for better blending (keeps density)
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(in.uv - center) * 2.0;
    let corner_clip = 1.0 - smoothstep(0.85, 1.0, dist);

    // Apply particle color
    var final_color = tex_color * in.color;
    final_color.a *= corner_clip;

    // Soft particle fade
    let soft_fade_dist = render_params.params.x;
    if (soft_fade_dist > 0.0) {
        let screen_coords = vec2<i32>(in.clip_position.xy);
        let scene_depth_raw = textureLoad(depth_texture, screen_coords, 0);

        let near = render_params.camera_params.x;
        let far = render_params.camera_params.y;
        let linear_scene_depth = linearize_depth(scene_depth_raw, near, far);

        let depth_diff = linear_scene_depth - in.particle_depth;
        let soft_fade = saturate(depth_diff / soft_fade_dist);

        final_color.a *= soft_fade;
    }

    // For additive blending, pre-multiply RGB by alpha
    final_color = vec4<f32>(final_color.rgb * final_color.a, final_color.a);

    if (final_color.a < 0.01) {
        discard;
    }

    return final_color;
}
