// Depth of Field post-processing shader for voxel rendering
// Uses a wider separable Gaussian blur to reduce banding/ghosting.

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

// Depth texture for reconstructing CoC on-the-fly (fused pass).
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

// Calculate blur amount (Circle of Confusion) based on depth
fn calculate_blur(linear_depth: f32) -> f32 {
    let distance_from_focal = abs(linear_depth - dof_uniforms.focal_distance);
    let focal_band = dof_uniforms.focal_range;
    
    // Objects within focal range are sharp
    if distance_from_focal < focal_band {
        return 0.0;
    }
    
    // Gradual blur increase outside focal range
    let blur_start = distance_from_focal - focal_band;
    let blur_scale = blur_start / max(dof_uniforms.focal_distance * 0.5, 50.0);
    
    return clamp(blur_scale * dof_uniforms.blur_strength, 0.0, 1.0);
}
// Simple 2D bokeh kernel inspired by Streets-GL: a small ring of
// samples in NDC space that we scale by the pixel CoC radius.
const BOKEH_SAMPLES: array<vec2<f32>, 12> = array<vec2<f32>, 12>(
    vec2<f32>( 1.0,  0.0),
    vec2<f32>(-1.0,  0.0),
    vec2<f32>( 0.0,  1.0),
    vec2<f32>( 0.0, -1.0),
    vec2<f32>( 0.7071,  0.7071),
    vec2<f32>(-0.7071,  0.7071),
    vec2<f32>( 0.7071, -0.7071),
    vec2<f32>(-0.7071, -0.7071),
    // Slightly inner ring to approximate smoother disc
    vec2<f32>( 0.5,  0.0),
    vec2<f32>(-0.5,  0.0),
    vec2<f32>( 0.0,  0.5),
    vec2<f32>( 0.0, -0.5),
);

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(color_texture, 0));
    let pixel_size = 1.0 / dimensions;
    let base_uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0));

    // Reconstruct linear depth and CoC (signed) directly; focal band zeroed.
    let depth = textureSample(depth_texture, color_sampler, base_uv);
    let near_plane = dof_uniforms.near_plane;
    let far_plane = dof_uniforms.far_plane;
    let z_ndc = depth * 2.0 - 1.0;
    let linear_depth = (2.0 * near_plane * far_plane) /
        (far_plane + near_plane - z_ndc * (far_plane - near_plane));

    let focus_distance = dof_uniforms.focal_distance;
    let focal_range = dof_uniforms.focal_range;
    let distance_from_focus = linear_depth - focus_distance;
    if abs(distance_from_focus) < focal_range || dof_uniforms.blur_strength <= 0.01 {
        let base_color = textureSample(color_texture, color_sampler, base_uv).rgb;
        return vec4<f32>(base_color, 0.0);
    }

    // Compute pixel CoC radius (signed) and derive magnitude.
    let sensor_scale = dimensions.y * 0.5;
    let blur_start = distance_from_focus - sign(distance_from_focus) * focal_range;
    let coc_norm = blur_start / max(abs(linear_depth), 1e-3);
    let coc_pixels = clamp(coc_norm * sensor_scale * 0.02, -15.0, 15.0);
    let coc_abs = abs(coc_pixels);

    // Adaptive tap count based on radius.
    // Use more samples for larger blur radii to ensure smooth results
    var taps: u32;
    if coc_abs < 2.0 { taps = 4u; }
    else if coc_abs < 5.0 { taps = 8u; }
    else if coc_abs < 10.0 { taps = 12u; }
    else { taps = 12u; } // All 12 samples for distant blur

    let blur_radius_pixels = clamp(coc_abs * dof_uniforms.blur_strength, 0.5, 25.0);
    let uv_radius = blur_radius_pixels * pixel_size;
    let base_color = textureSample(color_texture, color_sampler, base_uv).rgb;
    var accum = base_color;
    var weight_sum = 1.0;

    // Gather (no CoC edge gating to keep fused pass cheaper).
    for (var i: u32 = 0u; i < taps; i = i + 1u) {
        let dir = BOKEH_SAMPLES[i];
        let sample_uv = base_uv + dir * uv_radius;
        if all(sample_uv >= vec2<f32>(0.0)) && all(sample_uv <= vec2<f32>(1.0)) {
            let c = textureSample(color_texture, color_sampler, sample_uv).rgb;
            accum += c;
            weight_sum += 1.0;
        }
    }

    let blurred = accum / weight_sum;
    // Store normalized CoC magnitude in alpha for combine pass.
    return vec4<f32>(blurred, coc_abs / 15.0);
}
