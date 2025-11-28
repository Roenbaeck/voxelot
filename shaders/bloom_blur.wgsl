struct BloomBlurUniforms {
    direction: vec2<f32>,
    radius: f32,
    _padding0: f32,
    texel_size: vec2<f32>,
    _padding1: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> blur: BloomBlurUniforms;
@group(0) @binding(1) var bloom_texture: texture_2d<f32>;
@group(0) @binding(2) var post_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    let pos = positions[vertex_index];
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    let uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 9-tap Gaussian blur for smoother gradients
    // Offsets and weights calculated for sigma=2.0
    let offsets = array<f32, 5>(0.0, 1.0, 2.0, 3.0, 4.0);
    let weights = array<f32, 5>(
        0.227027,   // Center
        0.1945946,  // ±1
        0.1216216,  // ±2
        0.0540541,  // ±3
        0.0162162   // ±4
    );

    var color = textureSample(bloom_texture, post_sampler, in.uv).rgb * weights[0];

    for (var i = 1; i < 5; i++) {
        let offset = blur.direction * blur.texel_size * (offsets[i] * blur.radius);
        color += textureSample(bloom_texture, post_sampler, in.uv + offset).rgb * weights[i];
        color += textureSample(bloom_texture, post_sampler, in.uv - offset).rgb * weights[i];
    }

    // Add subtle dithering to eliminate final banding
    let dither = interleaved_gradient_noise(in.position.xy);
    let dither_amount = 1.0 / 1024.0; // Very subtle for bloom
    
    return vec4<f32>(color + dither * dither_amount, 1.0);
}

// Interleaved gradient noise for dithering
fn interleaved_gradient_noise(screen_pos: vec2<f32>) -> vec3<f32> {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    let noise = fract(magic.z * fract(dot(screen_pos, magic.xy)));
    let tri = noise - 0.5;
    return vec3<f32>(tri);
}


