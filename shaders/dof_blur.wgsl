// Simplified Depth of Field post-processing shader for voxel rendering

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

// Simple 9-tap kernel for smooth blur without bokeh artifacts
const KERNEL_OFFSETS: array<vec2<f32>, 9> = array<vec2<f32>, 9>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0,  0.0), vec2<f32>(0.0,  0.0), vec2<f32>(1.0,  0.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(0.0,  1.0), vec2<f32>(1.0,  1.0),
);

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(color_texture, 0));
    let pixel_size = 1.0 / dimensions;

    let base_uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let base_coords = vec2<i32>(base_uv * (dimensions - vec2<f32>(1.0)));
    let depth_value = textureLoad(depth_texture, base_coords, 0);
    let linear_depth = linearize_depth(depth_value);

    // Early exit if blur is disabled
    if dof_uniforms.blur_strength <= 0.01 {
        return textureSample(color_texture, color_sampler, base_uv);
    }

    let blur_amount = calculate_blur(linear_depth);
    
    // If in focus, return sharp image
    if blur_amount < 0.01 {
        return textureSample(color_texture, color_sampler, base_uv);
    }

    // Apply depth-aware blur
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    
    // Blur radius scales with blur amount
    let blur_radius = blur_amount * 3.0 * dof_uniforms.blur_strength;
    
    for (var i = 0; i < 9; i++) {
        let offset = KERNEL_OFFSETS[i] * pixel_size * blur_radius;
        let sample_uv = base_uv + offset;
        
        // Skip out-of-bounds samples
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            continue;
        }
        
        let sample_coords = vec2<i32>(sample_uv * (dimensions - vec2<f32>(1.0)));
        let sample_depth_value = textureLoad(depth_texture, sample_coords, 0);
        let sample_linear_depth = linearize_depth(sample_depth_value);
        
        // Strong depth discontinuity check to prevent halos
        let depth_diff = abs(linear_depth - sample_linear_depth);
        let depth_threshold = max(linear_depth * 0.05, 5.0); // 5% of depth or 5 units minimum
        
        if depth_diff > depth_threshold {
            continue; // Skip samples across depth boundaries
        }
        
        let sample_color = textureSample(color_texture, color_sampler, sample_uv).rgb;
        
        // Gaussian-like weight based on distance from center
        let dist = length(KERNEL_OFFSETS[i]);
        let weight = exp(-dist * dist * 0.5);
        
        color_sum += sample_color * weight;
        weight_sum += weight;
    }

    if weight_sum > 0.0 {
        return vec4<f32>(color_sum / weight_sum, 1.0);
    } else {
        // Fallback to center pixel if all samples rejected
        return textureSample(color_texture, color_sampler, base_uv);
    }
}
