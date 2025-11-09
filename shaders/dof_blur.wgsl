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
    if distance_from_focal < dof_uniforms.focal_range {
        return 0.0;
    }
    let blur = (distance_from_focal - dof_uniforms.focal_range) /
        max(dof_uniforms.focal_distance, 1e-3);
    return min(blur * dof_uniforms.blur_strength, 1.0);
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

    let blur_radius = coc * 28.0;
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

        let depth_weight = 1.0 - min(abs(linear_depth - sample_linear_depth) / 20.0, 1.0);
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
