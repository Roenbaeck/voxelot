// Depth of Field CoC & copy shader (cheap): compute CoC and copy base color without blending/sampling.

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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let base_uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let depth = textureSample(depth_texture, color_sampler, base_uv);
    let linear_depth = linearize_depth(depth);

    let focus_distance = dof_uniforms.focal_distance;
    let focal_range = dof_uniforms.focal_range;
    let distance_from_focus = linear_depth - focus_distance;
    if (abs(distance_from_focus) < focal_range) || (dof_uniforms.blur_strength <= 0.01) {
        let base_color = textureSample(color_texture, color_sampler, base_uv).rgb;
        return vec4<f32>(base_color, 0.0);
    }

    // Compute pixel CoC magnitude (unsigned approx), store normalized alpha.
    let sensor_scale = vec2<f32>(textureDimensions(color_texture, 0)).y * 0.5;
    let blur_start = distance_from_focus - sign(distance_from_focus) * focal_range;
    let coc_norm = blur_start / max(abs(linear_depth), 1e-3);
    let coc_pixels = clamp(coc_norm * sensor_scale * 0.02, -15.0, 15.0);
    let coc_abs = abs(coc_pixels);
    let base_color = textureSample(color_texture, color_sampler, base_uv).rgb;
    return vec4<f32>(base_color, coc_abs / 15.0);
}
