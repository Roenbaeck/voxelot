@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // full-screen triangle positions
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let p = pos[vi];
    return vec4<f32>(p.x, p.y, 0.0, 1.0);
}

@group(0) @binding(0) var my_tex: texture_2d<f32>;
@group(0) @binding(1) var my_sampler: sampler;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy / vec2<f32>(f32(textureDimensions(my_tex).x), f32(textureDimensions(my_tex).y));
    return textureSample(my_tex, my_sampler, uv);
}
