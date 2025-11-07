// Depth of Field post-processing shader
// Creates "tilt-shift" toy world effect

struct Uniforms {
    focal_distance: f32,      // Distance in focus
    focal_range: f32,         // Range of sharp focus
    blur_strength: f32,       // Max blur amount
    _padding: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var color_texture: texture_2d<f32>;

@group(0) @binding(2)
var depth_texture: texture_depth_2d;

@group(0) @binding(3)
var texture_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle
    var output: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);
    return output;
}

// Linearize depth for DoF calculation
fn linearize_depth(depth: f32) -> f32 {
    let near = 0.1;
    let far = 1000.0;
    return (2.0 * near) / (far + near - depth * (far - near));
}

// Calculate circle of confusion based on depth
fn calculate_coc(depth: f32) -> f32 {
    let linear_depth = linearize_depth(depth) * 1000.0; // Convert to world units
    let distance_from_focal = abs(linear_depth - uniforms.focal_distance);
    
    if (distance_from_focal < uniforms.focal_range) {
        return 0.0; // In focus
    }
    
    let blur = (distance_from_focal - uniforms.focal_range) / uniforms.focal_distance;
    return min(blur * uniforms.blur_strength, 1.0);
}

// Hexagonal bokeh blur pattern (better quality than box blur)
const SAMPLE_OFFSETS: array<vec2<f32>, 19> = array<vec2<f32>, 19>(
    vec2<f32>(0.0, 0.0),
    // Inner ring
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.866, 0.5),
    vec2<f32>(0.866, -0.5),
    vec2<f32>(0.0, -1.0),
    vec2<f32>(-0.866, -0.5),
    vec2<f32>(-0.866, 0.5),
    // Outer ring
    vec2<f32>(0.0, 2.0),
    vec2<f32>(1.732, 1.0),
    vec2<f32>(1.732, -1.0),
    vec2<f32>(0.0, -2.0),
    vec2<f32>(-1.732, -1.0),
    vec2<f32>(-1.732, 1.0),
    vec2<f32>(1.0, 1.732),
    vec2<f32>(1.0, -1.732),
    vec2<f32>(-1.0, -1.732),
    vec2<f32>(-1.0, 1.732),
    vec2<f32>(2.0, 0.0),
    vec2<f32>(-2.0, 0.0),
);

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(color_texture);
    let pixel_size = 1.0 / vec2<f32>(dimensions);
    
    // Sample depth at current pixel
    let depth = textureSample(depth_texture, texture_sampler, input.uv);
    let coc = calculate_coc(depth);
    
    // If in focus, return original color
    if (coc < 0.01) {
        return textureSample(color_texture, texture_sampler, input.uv);
    }
    
    // Apply bokeh blur
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    
    let blur_radius = coc * 20.0; // Max blur radius in pixels
    
    for (var i = 0; i < 19; i++) {
        let offset = SAMPLE_OFFSETS[i] * pixel_size * blur_radius;
        let sample_uv = input.uv + offset;
        
        // Sample color
        let sample_color = textureSample(color_texture, texture_sampler, sample_uv).rgb;
        
        // Sample depth and calculate weight (prevents background blur bleeding)
        let sample_depth = textureSample(depth_texture, texture_sampler, sample_uv);
        let sample_coc = calculate_coc(sample_depth);
        let weight = 1.0 - abs(coc - sample_coc); // Prefer similar depths
        
        color_sum += sample_color * weight;
        weight_sum += weight;
    }
    
    let final_color = color_sum / weight_sum;
    return vec4<f32>(final_color, 1.0);
}
