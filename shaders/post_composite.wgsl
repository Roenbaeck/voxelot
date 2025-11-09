struct CompositeUniforms {
    bloom_strength: f32,
    saturation_boost: f32,
    exposure: f32,
    _padding0: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> composite: CompositeUniforms;
@group(0) @binding(1) var post_color: texture_2d<f32>;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(3) var post_sampler: sampler;

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
    let base = textureSample(post_color, post_sampler, uv).rgb;
    let bloom = textureSample(bloom_texture, post_sampler, uv).rgb;

    let luma = dot(base, vec3<f32>(0.299, 0.587, 0.114));
    let balance = base - vec3<f32>(luma, luma, luma);
    let saturated = vec3<f32>(luma, luma, luma) + balance * composite.saturation_boost;

    var color = saturated + bloom * composite.bloom_strength;
    color = color * composite.exposure;
    color = max(color, vec3<f32>(0.0));

    return vec4<f32>(color, 1.0);
}
