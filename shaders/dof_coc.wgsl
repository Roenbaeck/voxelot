// Depth of Field Circle of Confusion pass
// Inspired by Streets-GL (MIT licensed) coc.frag implementation.

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
var depth_texture: texture_depth_2d;

@group(0) @binding(2)
var depth_sampler: sampler;

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

// Returns CoC radius in pixels (signed: foreground negative, background positive).
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(depth_texture, 0));
    let base_uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let depth = textureSample(depth_texture, depth_sampler, base_uv);
    let linear_depth = linearize_depth(depth);

    // Focus model: distance from focal plane mapped to CoC size.
    let focus_distance = dof_uniforms.focal_distance;
    let distance_from_focus = linear_depth - focus_distance;

    // Scale factor: choose constants to get a good artistic range.
    let sensor_scale = dimensions.y * 0.5;
    let coc_norm = distance_from_focus / max(linear_depth, 1e-3);
    let coc_pixels = clamp(coc_norm * sensor_scale * 0.02, -15.0, 15.0);

    return vec4<f32>(coc_pixels, 0.0, 0.0, 1.0);
}
