struct BloomExtractUniforms {
    threshold: f32,
    knee: f32,
    intensity: f32,
    _padding0: f32,
    source_texel_size: vec2<f32>,
    _padding1: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> bloom: BloomExtractUniforms;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
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
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5),
        vec2<f32>(-0.5, 0.5),
        vec2<f32>(0.5, 0.5),
    );

    var sample_sum = vec3<f32>(0.0);
    sample_sum = sample_sum
        + textureSample(scene_color, post_sampler, uv + offsets[0] * bloom.source_texel_size).rgb;
    sample_sum = sample_sum
        + textureSample(scene_color, post_sampler, uv + offsets[1] * bloom.source_texel_size).rgb;
    sample_sum = sample_sum
        + textureSample(scene_color, post_sampler, uv + offsets[2] * bloom.source_texel_size).rgb;
    sample_sum = sample_sum
        + textureSample(scene_color, post_sampler, uv + offsets[3] * bloom.source_texel_size).rgb;
    let average_color = sample_sum * 0.25;

    let brightness = max(max(average_color.r, average_color.g), average_color.b);
    let highlight = smoothstep(bloom.threshold, bloom.threshold + bloom.knee, brightness);
    let weight = highlight * highlight * bloom.intensity;
    let bloom_color = average_color * weight;
    return vec4<f32>(bloom_color, 1.0);
}
