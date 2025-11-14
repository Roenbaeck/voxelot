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

// CoC texture produced by dof_coc.wgsl; .r contains CoC in pixels.
@group(0) @binding(2)
var coc_texture: texture_2d<f32>;

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
    let base_coords = vec2<i32>(base_uv * (dimensions - vec2<f32>(1.0)));
    // Sample depth via color texture's dimensions for consistency
    let coc_value = textureLoad(coc_texture, base_coords, 0).r;
    let coc_abs = abs(coc_value);

    // Early exit if blur is disabled
    if dof_uniforms.blur_strength <= 0.01 {
        return textureSample(color_texture, color_sampler, base_uv);
    }

    // Use CoC magnitude as blur driver
    if coc_abs < 0.01 {
        return textureSample(color_texture, color_sampler, base_uv);
    }

    // Convert CoC in pixels to texture-space radius.
    let blur_radius_pixels = clamp(coc_abs * dof_uniforms.blur_strength, 0.5, 20.0);
    let base_color = textureSample(color_texture, color_sampler, base_uv).rgb;

    // Accumulate bokeh samples in a disc scaled by CoC.
    var accum = base_color;
    var weight_sum = 1.0;

    // Convert pixel radius to UV radius.
    let uv_radius = blur_radius_pixels * pixel_size;

    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let dir = BOKEH_SAMPLES[i];
        let sample_uv = base_uv + dir * uv_radius;
        if all(sample_uv >= vec2<f32>(0.0)) && all(sample_uv <= vec2<f32>(1.0)) {
            let coords = vec2<i32>(sample_uv * (dimensions - vec2<f32>(1.0)));
            let sample_coc = abs(textureLoad(coc_texture, coords, 0).r);
            // Preserve edges: only gather from samples that are at least
            // as blurry as our current pixel (or slightly less).
            if sample_coc >= coc_abs * 0.6 {
                let c = textureSample(color_texture, color_sampler, sample_uv).rgb;
                accum += c;
                weight_sum += 1.0;
            }
        }
    }

    let blurred = accum / weight_sum;
    return vec4<f32>(blurred, 1.0);
}
