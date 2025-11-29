struct CompositeUniforms {
    bloom_strength: f32,
    saturation_boost: f32,
    exposure: f32,
    ssao_enabled: f32,
    // Reserve a full vec4 for other per-pass state (debug and padding)
    ssao_debug: f32,
    ssao_strength: f32,
    ssr_debug: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> composite: CompositeUniforms;
@group(0) @binding(1) var post_color: texture_2d<f32>;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var ssao_texture: texture_2d<f32>;
@group(0) @binding(5) var ssr_debug_texture: texture_2d<f32>;
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
    
    // Sample SSILVB: RGB = accumulated emissive light, A = ambient occlusion
    let ssilvb_sample = textureSample(ssao_texture, post_sampler, uv);
    let indirect_light = ssilvb_sample.rgb;
    let raw_ao = ssilvb_sample.a;
    
    var ao: f32 = 1.0;
    if (composite.ssao_enabled > 0.5) {
        // Blend between no occlusion (1.0) and raw AO by strength.
        ao = 1.0 - composite.ssao_strength * (1.0 - raw_ao);
    }

    // Optional debug overlay: show SSAO in greyscale when ssao_debug is set
    if (composite.ssao_debug > 0.5) {
        // Show raw AO as greyscale (white = occluded) for easier debugging.
        return vec4<f32>(vec3<f32>(raw_ao), 1.0);
    }

    // SSR Debug overlay: show SSR texture directly when enabled
    if (composite.ssr_debug > 0.5) {
        let ssr_col = textureSample(ssr_debug_texture, post_sampler, uv);
        return vec4<f32>(ssr_col.rgb, 1.0);
    }

    let luma = dot(base, vec3<f32>(0.299, 0.587, 0.114));
    let balance = base - vec3<f32>(luma, luma, luma);
    let saturated = vec3<f32>(luma, luma, luma) + balance * composite.saturation_boost;

    var color = saturated + bloom * composite.bloom_strength;
    color = color + indirect_light; // Add screen-space emissive indirect lighting
    color = color * ao; // apply SSAO visibility (0..1) to darken color
    color = color * composite.exposure;
    color = max(color, vec3<f32>(0.0));

    return vec4<f32>(color, 1.0);
}
