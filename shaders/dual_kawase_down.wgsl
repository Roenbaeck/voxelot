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

    // 9-tap kernel: center + 4 neighbors + 4 diagonals with Gaussian-like weights
    // Keep center sampled with textureLoad (non-filtered) so alpha/CoC is exact
    let tex_w = i32(round(1.0 / ts.x));
    let tex_h = i32(round(1.0 / ts.y));
    let coord_x = clamp(i32(floor(uv.x * f32(tex_w))), 0, tex_w - 1);
    let coord_y = clamp(i32(floor(uv.y * f32(tex_h))), 0, tex_h - 1);
    let center_load = textureLoad(input_texture, vec2<i32>(coord_x, coord_y), 0);
    let c = center_load.rgb;
    let s1 = textureSample(input_texture, input_sampler, uv + vec2<f32>( ts.x * off, 0.0)).rgb;
    let s2 = textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x * off, 0.0)).rgb;
    let s3 = textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0, ts.y * off)).rgb;
    let s4 = textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0, -ts.y * off)).rgb;
    let s5 = textureSample(input_texture, input_sampler, uv + vec2<f32>( ts.x * off,  ts.y * off)).rgb;
    let s6 = textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x * off,  ts.y * off)).rgb;
    let s7 = textureSample(input_texture, input_sampler, uv + vec2<f32>( ts.x * off, -ts.y * off)).rgb;
    let s8 = textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x * off, -ts.y * off)).rgb;

    // Weighted Gaussian-like kernel (center 4, edges 2, corners 1) normalized by 16
    let avg = (4.0 * c + 2.0 * (s1 + s2 + s3 + s4) + (s5 + s6 + s7 + s8)) / 16.0;

    // Keep alpha channel (CoC) untouched: reuse sampled alpha
    let alpha = center_load.a;

    return vec4<f32>(avg, alpha);
}
