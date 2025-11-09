// Depth of Field post-processing shader implementing a hexagonal bokeh blur

struct DoFUniforms {
    focal_distance: f32,
    focal_range: f32,
    blur_strength: f32,
    near_plane: f32,
    far_plane: f32,
    _padding: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> dof_uniforms: DoFUniforms;

@group(0) @binding(1)
var color_texture: texture_2d<f32>;

@group(0) @binding(2)
var depth_texture: texture_depth_2d;

@group(0) @binding(3)
var color_sampler: sampler;

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

const TAU: f32 = 6.2831853;
const SAMPLE_COUNT: u32 = 12u;
const POISSON_OFFSETS: array<vec2<f32>, SAMPLE_COUNT> = array<vec2<f32>, SAMPLE_COUNT>(
    vec2<f32>(-0.613392, 0.617481),
    vec2<f32>(0.170019, -0.040254),
    vec2<f32>(-0.299417, -0.776534),
    vec2<f32>(0.645680, -0.388154),
    vec2<f32>(-0.651784, -0.266776),
    vec2<f32>(0.421003, 0.706885),
    vec2<f32>(-0.150590, 0.051698),
    vec2<f32>(0.110903, 0.689379),
    vec2<f32>(-0.002864, -0.364402),
    vec2<f32>(0.916106, 0.139173),
    vec2<f32>(-0.406787, 0.903441),
    vec2<f32>(0.482421, -0.857227),
);

fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn rotate2(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions_u = textureDimensions(color_texture, 0);
    let dimensions = vec2<f32>(dimensions_u);
    let pixel_size = 1.0 / dimensions;
    let focal_norm = max(dof_uniforms.focal_distance, 1e-3);

    let base_uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let base_coords = vec2<i32>(base_uv * (dimensions - vec2<f32>(1.0)));
    let depth_value = textureLoad(depth_texture, base_coords, 0);
    let linear_depth = linearize_depth(depth_value);

    if dof_uniforms.blur_strength <= 1e-4 {
        return textureSample(color_texture, color_sampler, base_uv);
    }

    let coc = calculate_coc(linear_depth);
    if coc < 0.01 {
        return textureSample(color_texture, color_sampler, base_uv);
    }

    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    // Always include the base pixel to keep focused regions sharp
    let base_color = textureSample(color_texture, color_sampler, base_uv).rgb;
    color_sum = base_color;
    weight_sum = 1.0;

    let blur_radius = coc * (22.0 + 20.0 * dof_uniforms.blur_strength);
    let rotation = hash12(base_uv) * TAU;

    for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let rotated = rotate2(POISSON_OFFSETS[i], rotation);
        let offset = rotated * pixel_size * blur_radius;
        let sample_uv = base_uv + offset;

        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            continue;
        }

        let sample_color = textureSampleLevel(color_texture, color_sampler, sample_uv, 0.0).rgb;
        let sample_coords = vec2<i32>(sample_uv * (dimensions - vec2<f32>(1.0)));
        let sample_depth = textureLoad(depth_texture, sample_coords, 0);
        let sample_linear_depth = linearize_depth(sample_depth);

        let sample_coc = calculate_coc(sample_linear_depth);
        if sample_coc <= 0.0 {
            continue;
        }

    let depth_diff = abs(linear_depth - sample_linear_depth);
    let depth_avg = (linear_depth + sample_linear_depth) * 0.5;
    let coc_mix = max(coc, sample_coc);
    let far_scale = 1.0 + clamp(depth_avg / focal_norm, 0.0, 8.0) * 1.3;
    let dynamic_tolerance = 24.0 + clamp(coc_mix, 0.0, 1.5) * 260.0;
    let depth_tolerance = (dynamic_tolerance + depth_avg * 0.28) * far_scale;
    let depth_weight = 1.0 - clamp(depth_diff / depth_tolerance, 0.0, 1.0);
        let radius_norm = clamp(length(rotated), 0.0, 1.0);
        let shape_weight = 1.0 - radius_norm * radius_norm;
        let tap_weight = sample_coc * depth_weight * shape_weight;

        if tap_weight <= 0.0 {
            continue;
        }

        color_sum = color_sum + sample_color * tap_weight;
        weight_sum = weight_sum + tap_weight;
    }

    let final_color = color_sum / max(weight_sum, 1e-4);
    return vec4<f32>(final_color, 1.0);
}
