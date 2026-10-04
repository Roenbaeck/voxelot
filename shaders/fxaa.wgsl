// FXAA 3.11-style edge anti-aliasing on the final composited image (before UI overlays).
// The source may be extended-linear HDR (macOS EDR, Rgba16Float) or SDR, so edge detection
// runs on a compressed perceptual luma (Reinhard then sqrt) instead of raw values: HDR
// highlights neither dominate the contrast test nor saturate it. Blending stays in the
// source's own (linear) space.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;

const EDGE_THRESHOLD: f32 = 0.125;
const EDGE_THRESHOLD_MIN: f32 = 0.0312;
const SUBPIXEL_QUALITY: f32 = 0.75;
const SEARCH_STEPS: i32 = 10;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

fn fxaa_luma(color: vec3<f32>) -> f32 {
    let l = max(dot(color, vec3<f32>(0.299, 0.587, 0.114)), 0.0);
    return sqrt(l / (1.0 + l));
}

fn read_luma(pixel: vec2<i32>, offset: vec2<i32>, size: vec2<i32>) -> f32 {
    let p = clamp(pixel + offset, vec2<i32>(0), size - vec2<i32>(1));
    return fxaa_luma(textureLoad(source, p, 0).rgb);
}

fn sample_luma(uv: vec2<f32>) -> f32 {
    return fxaa_luma(textureSampleLevel(source, linear_sampler, uv, 0.0).rgb);
}

fn search_step(i: i32) -> f32 {
    var steps = array<f32, 10>(1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 4.0, 8.0, 12.0);
    return steps[i];
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(source));
    let pixel = vec2<i32>(frag_pos.xy);
    let texel = 1.0 / vec2<f32>(size);
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) * texel;

    let center_color = textureLoad(source, pixel, 0);
    let luma_center = fxaa_luma(center_color.rgb);
    let luma_down = read_luma(pixel, vec2<i32>(0, 1), size);
    let luma_up = read_luma(pixel, vec2<i32>(0, -1), size);
    let luma_left = read_luma(pixel, vec2<i32>(-1, 0), size);
    let luma_right = read_luma(pixel, vec2<i32>(1, 0), size);
    let luma_min = min(luma_center, min(min(luma_down, luma_up), min(luma_left, luma_right)));
    let luma_max = max(luma_center, max(max(luma_down, luma_up), max(luma_left, luma_right)));
    let luma_range = luma_max - luma_min;
    // Flat regions exit after five reads.
    if (luma_range < max(EDGE_THRESHOLD_MIN, luma_max * EDGE_THRESHOLD)) {
        return center_color;
    }

    let luma_down_left = read_luma(pixel, vec2<i32>(-1, 1), size);
    let luma_up_right = read_luma(pixel, vec2<i32>(1, -1), size);
    let luma_up_left = read_luma(pixel, vec2<i32>(-1, -1), size);
    let luma_down_right = read_luma(pixel, vec2<i32>(1, 1), size);
    let luma_down_up = luma_down + luma_up;
    let luma_left_right = luma_left + luma_right;
    let luma_left_corners = luma_down_left + luma_up_left;
    let luma_down_corners = luma_down_left + luma_down_right;
    let luma_right_corners = luma_down_right + luma_up_right;
    let luma_up_corners = luma_up_right + luma_up_left;
    let edge_horizontal = abs(-2.0 * luma_left + luma_left_corners)
        + abs(-2.0 * luma_center + luma_down_up) * 2.0
        + abs(-2.0 * luma_right + luma_right_corners);
    let edge_vertical = abs(-2.0 * luma_up + luma_up_corners)
        + abs(-2.0 * luma_center + luma_left_right) * 2.0
        + abs(-2.0 * luma_down + luma_down_corners);
    let is_horizontal = edge_horizontal >= edge_vertical;

    // Pick the side of the edge with the steeper gradient.
    let luma1 = select(luma_left, luma_up, is_horizontal);
    let luma2 = select(luma_right, luma_down, is_horizontal);
    let gradient1 = luma1 - luma_center;
    let gradient2 = luma2 - luma_center;
    let is1_steepest = abs(gradient1) >= abs(gradient2);
    let gradient_scaled = 0.25 * max(abs(gradient1), abs(gradient2));
    var step_length = select(texel.x, texel.y, is_horizontal);
    var luma_local_average = 0.5 * (luma2 + luma_center);
    if (is1_steepest) {
        step_length = -step_length;
        luma_local_average = 0.5 * (luma1 + luma_center);
    }
    var edge_uv = uv;
    if (is_horizontal) {
        edge_uv.y += step_length * 0.5;
    } else {
        edge_uv.x += step_length * 0.5;
    }

    // Walk along the edge in both directions until its luma ends.
    let offset = select(vec2<f32>(0.0, texel.y), vec2<f32>(texel.x, 0.0), is_horizontal);
    var uv1 = edge_uv - offset * search_step(0);
    var uv2 = edge_uv + offset * search_step(0);
    var luma_end1 = sample_luma(uv1) - luma_local_average;
    var luma_end2 = sample_luma(uv2) - luma_local_average;
    var reached1 = abs(luma_end1) >= gradient_scaled;
    var reached2 = abs(luma_end2) >= gradient_scaled;
    for (var i = 1; i < SEARCH_STEPS; i++) {
        if (reached1 && reached2) {
            break;
        }
        if (!reached1) {
            uv1 -= offset * search_step(i);
            luma_end1 = sample_luma(uv1) - luma_local_average;
            reached1 = abs(luma_end1) >= gradient_scaled;
        }
        if (!reached2) {
            uv2 += offset * search_step(i);
            luma_end2 = sample_luma(uv2) - luma_local_average;
            reached2 = abs(luma_end2) >= gradient_scaled;
        }
    }

    let distance1 = select(uv.y - uv1.y, uv.x - uv1.x, is_horizontal);
    let distance2 = select(uv2.y - uv.y, uv2.x - uv.x, is_horizontal);
    let is_direction1 = distance1 < distance2;
    let distance_final = min(distance1, distance2);
    let edge_thickness = distance1 + distance2;
    let is_luma_center_smaller = luma_center < luma_local_average;
    let correct_variation = (select(luma_end2, luma_end1, is_direction1) < 0.0) != is_luma_center_smaller;
    var pixel_offset = select(0.0, -distance_final / edge_thickness + 0.5, correct_variation);

    // Sub-pixel aliasing: isolated bright or dark pixels (distant voxel terraces).
    let luma_average = (1.0 / 12.0) * (2.0 * (luma_down_up + luma_left_right)
        + luma_left_corners + luma_right_corners);
    let subpixel1 = clamp(abs(luma_average - luma_center) / luma_range, 0.0, 1.0);
    let subpixel2 = (-2.0 * subpixel1 + 3.0) * subpixel1 * subpixel1;
    pixel_offset = max(pixel_offset, subpixel2 * subpixel2 * SUBPIXEL_QUALITY);

    var final_uv = uv;
    if (is_horizontal) {
        final_uv.y += pixel_offset * step_length;
    } else {
        final_uv.x += pixel_offset * step_length;
    }
    return vec4<f32>(textureSampleLevel(source, linear_sampler, final_uv, 0.0).rgb, 1.0);
}
