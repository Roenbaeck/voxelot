// Separable Gaussian smoothing pass that refines the stochastic DoF blur

struct DoFUniforms {
    focal_distance: f32,
    focal_range: f32,
    blur_strength: f32,
    near_plane: f32,
    far_plane: f32,
    _padding: vec3<f32>,
};

struct DoFSmoothUniforms {
    direction_radius: vec4<f32>,
    texel_coc: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> dof_uniforms: DoFUniforms;

@group(0) @binding(1)
var color_texture: texture_2d<f32>;

@group(0) @binding(2)
var depth_texture: texture_depth_2d;

@group(0) @binding(3)
var color_sampler: sampler;

@group(0) @binding(4)
var<uniform> smooth_uniforms: DoFSmoothUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    let pos = positions[vertex_index];
    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    let uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
    output.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return output;
}

fn linearize_depth(depth: f32) -> f32 {
    let near_plane = dof_uniforms.near_plane;
    let far_plane = dof_uniforms.far_plane;
    let z_ndc = depth * 2.0 - 1.0;
    return (2.0 * near_plane * far_plane) /
        (far_plane + near_plane - z_ndc * (far_plane - near_plane));
}

fn calculate_coc(linear_depth: f32) -> f32 {
    let distance_from_focal = abs(linear_depth - dof_uniforms.focal_distance);
    let focus_band = dof_uniforms.focal_range * 4.0;
    if distance_from_focal < focus_band {
        return 0.0;
    }
    let focal_norm = max(dof_uniforms.focal_distance, 1e-3);
    let softness = max(focus_band * 0.8, focal_norm * 0.25);
    let base = clamp((distance_from_focal - focus_band) / (softness + 1.0), 0.0, 6.0);
    let linear_cap = 1.25;
    let linear_scale = 0.26;
    var base_coc = base * linear_scale;

    if (base > linear_cap) {
        let exp_input = base - linear_cap;
        let exp_curve = 1.0 - exp(-exp_input * 0.8);
        let exp_gain = 0.75 + base * 0.32;
        let offset = linear_cap * linear_scale;
        base_coc = offset + exp_curve * exp_gain;
    }

    let far_distance = max(linear_depth - dof_uniforms.focal_distance, 0.0);
    let far_ratio = clamp(far_distance / (focal_norm * 0.65 + 25.0), 0.0, 4.0);
    let far_boost = 1.0 + far_ratio * 2.4 + far_ratio * far_ratio * 1.2;
    let strength = base_coc * dof_uniforms.blur_strength * far_boost;
    return clamp(strength, 0.0, 5.0);
}

const CENTER_WEIGHT: f32 = 0.227027;
const GAUSS_OFFSETS: array<f32, 2> = array<f32, 2>(1.0, 2.0);
const GAUSS_WEIGHTS: array<f32, 2> = array<f32, 2>(0.316216, 0.070270);

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions_u = textureDimensions(color_texture, 0);
    let dimensions = vec2<f32>(dimensions_u);
    let base_uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0));

    let texel_scale = vec2<f32>(smooth_uniforms.texel_coc.x, smooth_uniforms.texel_coc.y);
    let dir_raw = smooth_uniforms.direction_radius.xy;
    let dir_len = max(length(dir_raw), 1e-5);
    let dir = dir_raw / dir_len;
    let texel_step = vec2<f32>(dir.x * texel_scale.x, dir.y * texel_scale.y);
    let radius_scale = smooth_uniforms.direction_radius.z;
    let coc_scale = smooth_uniforms.direction_radius.w;
    let coc_bias = smooth_uniforms.texel_coc.z;

    if radius_scale <= 1e-4 || coc_scale <= 1e-4 {
        let base_color = textureSample(color_texture, color_sampler, base_uv);
        return vec4<f32>(base_color.rgb, 1.0);
    }

    let base_coords = vec2<i32>(base_uv * (dimensions - vec2<f32>(1.0)));
    let depth_value = textureLoad(depth_texture, base_coords, 0);
    let linear_depth = linearize_depth(depth_value);
    let center_coc = calculate_coc(linear_depth);
    let base_color = textureSample(color_texture, color_sampler, base_uv).rgb;

    let center_blend = clamp(pow(center_coc * coc_scale + coc_bias, 1.2), 0.0, 1.0);

    var accum = base_color * CENTER_WEIGHT;
    var weight_sum = CENTER_WEIGHT;
    var max_blend = center_blend;

    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let offset = GAUSS_OFFSETS[i] * radius_scale;
        let weight = GAUSS_WEIGHTS[i];
        let delta = texel_step * offset;

        let sample_uv_pos = base_uv + delta;
        if all(sample_uv_pos >= vec2<f32>(0.0)) && all(sample_uv_pos <= vec2<f32>(1.0)) {
            let sample_coords = vec2<i32>(sample_uv_pos * (dimensions - vec2<f32>(1.0)));
            let sample_depth = textureLoad(depth_texture, sample_coords, 0);
            let sample_linear = linearize_depth(sample_depth);
            let sample_coc = calculate_coc(sample_linear);
            let intensity = clamp(pow(max(sample_coc, center_coc) * coc_scale + coc_bias, 1.2), 0.0, 1.0);
            let w = weight * max(intensity, 1e-3);
            let sample_color = textureSampleLevel(color_texture, color_sampler, sample_uv_pos, 0.0).rgb;
            accum = accum + sample_color * w;
            weight_sum = weight_sum + w;
            max_blend = max(max_blend, intensity);
        }

        let sample_uv_neg = base_uv - delta;
        if all(sample_uv_neg >= vec2<f32>(0.0)) && all(sample_uv_neg <= vec2<f32>(1.0)) {
            let sample_coords = vec2<i32>(sample_uv_neg * (dimensions - vec2<f32>(1.0)));
            let sample_depth = textureLoad(depth_texture, sample_coords, 0);
            let sample_linear = linearize_depth(sample_depth);
            let sample_coc = calculate_coc(sample_linear);
            let intensity = clamp(pow(max(sample_coc, center_coc) * coc_scale + coc_bias, 1.2), 0.0, 1.0);
            let w = weight * max(intensity, 1e-3);
            let sample_color = textureSampleLevel(color_texture, color_sampler, sample_uv_neg, 0.0).rgb;
            accum = accum + sample_color * w;
            weight_sum = weight_sum + w;
            max_blend = max(max_blend, intensity);
        }
    }

    let smooth_color = accum / max(weight_sum, 1e-4);
    let blend = clamp(pow(max_blend, 1.1), 0.0, 1.0);
    let final_color = mix(base_color, smooth_color, blend);
    return vec4<f32>(final_color, 1.0);
}
