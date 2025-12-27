// Color correction shader
// Applies brightness, contrast, saturation, gamma, temperature, and tint adjustments

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct ColorCorrectionUniforms {
    // x: brightness, y: contrast, z: saturation, w: gamma
    params1: vec4<f32>,
    // x: temperature, y: tint, z: unused, w: unused
    params2: vec4<f32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: ColorCorrectionUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Full-screen triangle (covers -1 to 3 range, clipped to screen)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return out;
}

// Convert RGB to luminance
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Apply saturation adjustment
fn adjust_saturation(color: vec3<f32>, saturation: f32) -> vec3<f32> {
    let lum = luminance(color);
    return mix(vec3<f32>(lum), color, saturation);
}

// Apply contrast adjustment
fn adjust_contrast(color: vec3<f32>, contrast: f32) -> vec3<f32> {
    return (color - 0.5) * contrast + 0.5;
}

// Apply brightness adjustment
fn adjust_brightness(color: vec3<f32>, brightness: f32) -> vec3<f32> {
    return color + brightness;
}

// Apply gamma correction
fn adjust_gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
}

// Apply color temperature (warm/cool shift)
fn adjust_temperature(color: vec3<f32>, temperature: f32) -> vec3<f32> {
    // Warm adds red/yellow, cool adds blue
    var result = color;
    if (temperature > 0.0) {
        // Warm: increase red, slightly decrease blue
        result.r += temperature * 0.1;
        result.g += temperature * 0.05;
        result.b -= temperature * 0.1;
    } else {
        // Cool: increase blue, slightly decrease red
        result.r += temperature * 0.1;
        result.b -= temperature * 0.1;
    }
    return result;
}

// Apply tint (green/magenta shift)
fn adjust_tint(color: vec3<f32>, tint: f32) -> vec3<f32> {
    var result = color;
    if (tint > 0.0) {
        // Magenta: increase red and blue, decrease green
        result.r += tint * 0.05;
        result.g -= tint * 0.1;
        result.b += tint * 0.05;
    } else {
        // Green: increase green, decrease red and blue
        result.r += tint * 0.05;
        result.g -= tint * 0.1;
        result.b += tint * 0.05;
    }
    return result;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let brightness = uniforms.params1.x;
    let contrast = uniforms.params1.y;
    let saturation = uniforms.params1.z;
    let gamma = uniforms.params1.w;
    let temperature = uniforms.params2.x;
    let tint = uniforms.params2.y;

    var color = textureSampleLevel(input_texture, tex_sampler, in.uv, 0.0);

    // Apply adjustments in order
    var rgb = color.rgb;

    // Temperature and tint first (color balance)
    rgb = adjust_temperature(rgb, temperature);
    rgb = adjust_tint(rgb, tint);

    // Brightness
    rgb = adjust_brightness(rgb, brightness);

    // Contrast
    rgb = adjust_contrast(rgb, contrast);

    // Saturation
    rgb = adjust_saturation(rgb, saturation);

    // Gamma last
    rgb = adjust_gamma(rgb, gamma);

    // Clamp to valid range
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(rgb, color.a);
}
