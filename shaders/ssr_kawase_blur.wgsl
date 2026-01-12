// Kawase blur variant for SSR (matches the viewer's UV convention: Y flipped like post_composite).

struct KawaseUniforms {
    texel_size: vec2<f32>,
    offset: f32,
    _pad: f32,
};

var<immediate> kawase: KawaseUniforms;

@group(0) @binding(0)
var input_texture: texture_2d<f32>;
@group(0) @binding(1)
var input_sampler: sampler;
@group(0) @binding(2)
var scene_depth: texture_depth_2d;
@group(0) @binding(3)
var normal_gbuffer: texture_2d<f32>;

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
    let uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    // Start on octahedron surface
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));

    // Fold back the parts where z is negative (lower hemisphere)
    if (v.z < 0.0) {
        let ox = (1.0 - abs(v.y)) * select(-1.0, 1.0, v.x >= 0.0);
        let oy = (1.0 - abs(v.x)) * select(-1.0, 1.0, v.y >= 0.0);
        v.x = ox;
        v.y = oy;
    }

    return normalize(v);
}

fn uv_to_pixel(uv: vec2<f32>, dim: vec2<u32>) -> vec2<i32> {
    let px = clamp(i32(uv.x * f32(dim.x)), 0, i32(dim.x) - 1);
    let py = clamp(i32(uv.y * f32(dim.y)), 0, i32(dim.y) - 1);
    return vec2<i32>(px, py);
}

fn load_depth_at_uv(uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(scene_depth);
    let px = uv_to_pixel(uv, dim);
    return textureLoad(scene_depth, px, 0);
}

fn load_normal_at_uv(uv: vec2<f32>) -> vec3<f32> {
    let dim = textureDimensions(normal_gbuffer);
    let px = uv_to_pixel(uv, dim);
    return oct_decode(textureLoad(normal_gbuffer, px, 0).xy);
}

fn edge_weight(depth0: f32, normal0: vec3<f32>, uv: vec2<f32>) -> f32 {
    let depth1 = load_depth_at_uv(uv);
    let normal1 = load_normal_at_uv(uv);
    let depth_diff = abs(depth1 - depth0);
    let ndot = clamp(dot(normal0, normal1), 0.0, 1.0);
    let normal_diff = 1.0 - ndot;

    // Tune: raw depth is non-linear; this is still a good edge-stop for silhouettes.
    let w_depth = exp(-depth_diff * 200.0);
    let w_normal = exp(-normal_diff * 12.0);
    return w_depth * w_normal;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let ts = kawase.texel_size;

    // 9-tap kernel: center + 4 neighbors + 4 diagonals with Gaussian-like weights.
    // Important: make it alpha-aware so reflection color doesn't bleed past reflective geometry edges.
    // We weight each tap by its own SSR alpha and normalize by total alpha weight.
    let center_sample = textureSample(input_texture, input_sampler, uv);
    let depth0 = load_depth_at_uv(uv);
    let normal0 = load_normal_at_uv(uv);

    // Distance-based blur scaling: reduce blur for distant surfaces.
    // Reversed-Z depth: close surfaces have depth ~1, far surfaces have depth ~0.
    // We want more blur when close and less when far.
    let distance_factor = smoothstep(0.001, 0.5, depth0);
    let off = kawase.offset * (0.3 + 0.7 * distance_factor);

    let uv1 = uv + vec2<f32>( ts.x * off, 0.0);
    let uv2 = uv + vec2<f32>(-ts.x * off, 0.0);
    let uv3 = uv + vec2<f32>(0.0, ts.y * off);
    let uv4 = uv + vec2<f32>(0.0, -ts.y * off);
    let uv5 = uv + vec2<f32>( ts.x * off,  ts.y * off);
    let uv6 = uv + vec2<f32>(-ts.x * off,  ts.y * off);
    let uv7 = uv + vec2<f32>( ts.x * off, -ts.y * off);
    let uv8 = uv + vec2<f32>(-ts.x * off, -ts.y * off);

    let t1 = textureSample(input_texture, input_sampler, uv1);
    let t2 = textureSample(input_texture, input_sampler, uv2);
    let t3 = textureSample(input_texture, input_sampler, uv3);
    let t4 = textureSample(input_texture, input_sampler, uv4);
    let t5 = textureSample(input_texture, input_sampler, uv5);
    let t6 = textureSample(input_texture, input_sampler, uv6);
    let t7 = textureSample(input_texture, input_sampler, uv7);
    let t8 = textureSample(input_texture, input_sampler, uv8);

    // Edge-stop weights (preserve internal edges where the reflective surface changes normal/depth).
    let e1 = edge_weight(depth0, normal0, uv1);
    let e2 = edge_weight(depth0, normal0, uv2);
    let e3 = edge_weight(depth0, normal0, uv3);
    let e4 = edge_weight(depth0, normal0, uv4);
    let e5 = edge_weight(depth0, normal0, uv5);
    let e6 = edge_weight(depth0, normal0, uv6);
    let e7 = edge_weight(depth0, normal0, uv7);
    let e8 = edge_weight(depth0, normal0, uv8);

    let w0 = 4.0 * center_sample.a;
    let w1 = 2.0 * t1.a * e1;
    let w2 = 2.0 * t2.a * e2;
    let w3 = 2.0 * t3.a * e3;
    let w4 = 2.0 * t4.a * e4;
    let w5 = 1.0 * t5.a * e5;
    let w6 = 1.0 * t6.a * e6;
    let w7 = 1.0 * t7.a * e7;
    let w8 = 1.0 * t8.a * e8;

    let rgb_sum =
        w0 * center_sample.rgb +
        w1 * t1.rgb +
        w2 * t2.rgb +
        w3 * t3.rgb +
        w4 * t4.rgb +
        w5 * t5.rgb +
        w6 * t6.rgb +
        w7 * t7.rgb +
        w8 * t8.rgb;

    let w_sum = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8;
    let avg = select(vec3<f32>(0.0), rgb_sum / max(w_sum, 1e-5), w_sum > 1e-5);

    // Preserve alpha (SSR strength) from the center sample (avoid expanding reflection coverage).
    return vec4<f32>(avg, center_sample.a);
}
