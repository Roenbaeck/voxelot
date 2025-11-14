struct VsOut {
    @builtin(position) pos: vec4<f32>;
    @location(0) uv: vec2<f32>;
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uv_coords = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(verts[vi], 0.0, 1.0);
    out.uv = uv_coords[vi];
    return out;
}

@group(0) @binding(0)
var coc_tex: texture_2d<f32>;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let dims = textureDimensions(coc_tex);
    let px = vec2<i32>(in.uv * vec2<f32>(dims));
    let coc = textureLoad(coc_tex, px, 0).x;
    // Map CoC (in pixels, signed) to 0..1 for visualization. Adjust max radius if needed.
    let max_radius: f32 = 24.0;
    let v = clamp(abs(coc) / max_radius, 0.0, 1.0);
    return vec4<f32>(v, v, v, 1.0);
}
