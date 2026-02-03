struct Instance {
    pos: vec2<f32>,
    size: vec2<f32>,
    color: vec4<f32>,
    shape: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) shape: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) pos: vec2<f32>,
    @location(1) size: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) shape: f32,
) -> VertexOutput {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );
    let uv = quad[vertex_index];
    let ndc = pos + uv * size;
    var out: VertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.color = color;
    out.uv = uv;
    out.shape = shape;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let s = i32(in.shape + 0.5);
    if (s == 1) {
        // Up-pointing arrow (triangle)
        let dx = abs(in.uv.x - 0.5) * 2.0;
        let keep = select(0.0, 1.0, in.uv.y >= dx);
        return vec4<f32>(in.color.rgb, in.color.a * keep);
    }
    if (s == 2) {
        // Down-pointing arrow (triangle)
        let dx = abs(in.uv.x - 0.5) * 2.0;
        let keep = select(0.0, 1.0, in.uv.y <= (1.0 - dx));
        return vec4<f32>(in.color.rgb, in.color.a * keep);
    }
    return in.color;
}
