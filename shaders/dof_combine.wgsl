// Depth of Field combine pass
// Inspired by Streets-GL (MIT licensed) dofCombine.frag implementation.

@group(0) @binding(0)
var dof_texture: texture_2d<f32>;

@group(0) @binding(1)
var coc_texture: texture_2d<f32>;

@group(0) @binding(2)
var source_texture: texture_2d<f32>;

@group(0) @binding(3)
var linear_sampler: sampler;

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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let source = textureSample(source_texture, linear_sampler, uv);
    let dof = textureSample(dof_texture, linear_sampler, uv);
    let coc = textureSample(coc_texture, linear_sampler, uv).r;

    let strength = smoothstep(0.1, 1.0, abs(coc));
    // Simple lerp between sharp and blurred based on CoC.
    let color = mix(source.rgb, dof.rgb, strength);
    return vec4<f32>(color, 1.0);
}
