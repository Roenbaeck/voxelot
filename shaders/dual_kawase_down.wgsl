// Dual Kawase Down sample (WGSL)

struct KawaseUniforms {
    texel_size: vec2<f32>,
    offset: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> kawase: KawaseUniforms;

@group(0) @binding(1)
var input_texture: texture_2d<f32>;
@group(0) @binding(2)
var input_sampler: sampler;

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
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let off = kawase.offset;
    let ts = kawase.texel_size;

    // sample center and 4 offsets (kawase kernel)
    let c = textureSample(input_texture, input_sampler, uv).rgb;
    let s1 = textureSample(input_texture, input_sampler, uv + vec2<f32>( ts.x * off, 0.0)).rgb;
    let s2 = textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x * off, 0.0)).rgb;
    let s3 = textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0, ts.y * off)).rgb;
    let s4 = textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0, -ts.y * off)).rgb;

    // Simple average
    let avg = (c + s1 + s2 + s3 + s4) / 5.0;

    // Keep alpha channel (CoC) untouched: sample alpha from input
    let alpha = textureSample(input_texture, input_sampler, uv).a;

    return vec4<f32>(avg, alpha);
}
