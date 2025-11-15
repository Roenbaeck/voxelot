// dof_blur.wgsl
// Deprecated: fused DoF implementation removed in favor of Dual Kawase blur.
// Kept as a minimal placeholder. Delete this file when you no longer need the backup.

struct DummyUniform {}

@group(0) @binding(0)
var<uniform> _u: DummyUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>;
    @location(0) uv: vec2<f32>;
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOutput;
    let p = positions[vertex_index];
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_main(_in: VertexOutput) -> @location(0) vec4<f32> {
    // Transparent no-op fragment to keep file syntactically valid.
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
