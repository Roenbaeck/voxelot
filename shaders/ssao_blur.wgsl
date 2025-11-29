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
@group(0) @binding(1) var ssao_texture: texture_2d<f32>;
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
    let offsets = array<f32, 3>(0.0, 1.38461538, 3.23076923);
    let weights = array<f32, 3>(0.227027, 0.3162162, 0.070270);

    var v = textureSample(ssao_texture, post_sampler, uv) * weights[0];

    let offset1 = blur.direction * blur.texel_size * (offsets[1] * blur.radius);
    v = v
        + (textureSample(ssao_texture, post_sampler, uv + offset1)
            + textureSample(ssao_texture, post_sampler, uv - offset1))
        * weights[1];

    let offset2 = blur.direction * blur.texel_size * (offsets[2] * blur.radius);
    v = v
        + (textureSample(ssao_texture, post_sampler, uv + offset2)
            + textureSample(ssao_texture, post_sampler, uv - offset2))
        * weights[2];

    return v;
}
