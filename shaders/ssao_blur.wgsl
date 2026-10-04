// Separable, depth-aware blur of the half-resolution SSILVB target (RGB = indirect light,
// A = AO). Every tap is weighted by its view depth relative to the centre texel (both read at
// the texel's full-resolution source pixel, see ssilvb.wgsl) so AO and bounce light stay on
// the surface that produced them instead of bleeding across silhouettes or into the sky.

struct BloomBlurUniforms {
    direction: vec2<f32>,
    radius: f32,
    _padding0: f32,
    texel_size: vec2<f32>,
    // Camera near/far planes for depth linearisation.
    near_far: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> blur: BloomBlurUniforms;
@group(0) @binding(1) var ssao_texture: texture_2d<f32>;
@group(0) @binding(2) var post_sampler: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;

// Relative depth difference that falls to 1/e weight is 1/DEPTH_SHARPNESS.
const DEPTH_SHARPNESS: f32 = 24.0;
const SKY_LINEAR_DEPTH: f32 = 1.0e30;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    let pos = positions[vertex_index];
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    let uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

// Linear view depth of half-resolution texel `texel`, read at its full-resolution source
// pixel (must match source_pixel() in ssilvb.wgsl).
fn texel_linear_depth(texel: vec2<i32>) -> f32 {
    let full_size = vec2<i32>(textureDimensions(depth_tex));
    let pixel = clamp(texel * 2 + vec2<i32>(1), vec2<i32>(0), full_size - vec2<i32>(1));
    let d = textureLoad(depth_tex, pixel, 0);
    if (d >= 1.0) {
        return SKY_LINEAR_DEPTH;
    }
    let n = blur.near_far.x;
    let f = blur.near_far.y;
    return (n * f) / max(f - d * (f - n), 1e-6);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(ssao_texture));
    let center = vec2<i32>(frag_pos.xy);
    let center_value = textureLoad(ssao_texture, center, 0);
    let center_depth = texel_linear_depth(center);
    if (center_depth >= SKY_LINEAR_DEPTH) {
        return center_value;
    }

    // Half Gaussian kernel; tap spacing scales with the configured blur radius.
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    let step = max(blur.radius, 1.0);
    let dir = vec2<i32>(blur.direction);
    var sum = center_value * weights[0];
    var weight_sum = weights[0];
    for (var k = 1; k <= 4; k++) {
        let offset = i32(round(f32(k) * step));
        for (var side = -1; side <= 1; side += 2) {
            let texel = clamp(center + dir * (offset * side), vec2<i32>(0), size - vec2<i32>(1));
            let sample_depth = texel_linear_depth(texel);
            let difference = abs(sample_depth - center_depth) / center_depth;
            let w = weights[k] * exp(-difference * DEPTH_SHARPNESS);
            sum += textureLoad(ssao_texture, texel, 0) * w;
            weight_sum += w;
        }
    }
    return sum / weight_sum;
}
