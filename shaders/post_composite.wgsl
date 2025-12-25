struct CompositeUniforms {
    bloom_strength: f32,
    saturation_boost: f32,
    exposure: f32,
    ssao_enabled: f32,
    // Reserve a full vec4 for other per-pass state (debug and padding)
    ssao_debug: f32,
    ssao_strength: f32,
    ssr_debug: f32,
    indirect_light_scale: f32, // Modulates emissive bounce light by ambient darkness (0=day, 1=night)
    hdr_highlight_compression: f32,
    _pad2: f32,
    uv_scale: vec2<f32>,
    uv_offset: vec2<f32>,
};

fn compress_highlights_hdr(color: vec3<f32>) -> vec3<f32> {
    // Preserve values in [0..1] and apply a soft shoulder above 1.0.
    // This keeps HDR headroom (values can still exceed 1.0) while preventing
    // extreme highlights from turning into a veiling-glare look.
    let x = max(color, vec3<f32>(0.0));
    let base = min(x, vec3<f32>(1.0));
    let hi = max(x - vec3<f32>(1.0), vec3<f32>(0.0));
    // Soft-shoulder highlights without crushing them.
    // With max_hi = 16, output can still reach ~17 in extreme cases (base + max_hi).
    let max_hi = vec3<f32>(16.0);
    let hi_comp = (hi * max_hi) / (max_hi + hi);
    return base + hi_comp;
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> composite: CompositeUniforms;
@group(0) @binding(1) var post_color: texture_2d<f32>;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var ssao_texture: texture_2d<f32>;
@group(0) @binding(5) var ssr_debug_texture: texture_2d<f32>;
@group(0) @binding(6) var rc_texture: texture_2d<f32>;
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
    let sample_uv = composite.uv_offset + uv * composite.uv_scale;
    let base = textureSample(post_color, post_sampler, sample_uv).rgb;
    let bloom = textureSample(bloom_texture, post_sampler, sample_uv).rgb;
    
    // Sample SSILVB: RGB = accumulated emissive light, A = ambient occlusion
    let ssilvb_sample = textureSample(ssao_texture, post_sampler, sample_uv);
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
        let ssr_col = textureSample(ssr_debug_texture, post_sampler, sample_uv);
        return vec4<f32>(ssr_col.rgb, 1.0);
    }

    let luma = dot(base, vec3<f32>(0.299, 0.587, 0.114));
    let balance = base - vec3<f32>(luma, luma, luma);
    let saturated = vec3<f32>(luma, luma, luma) + balance * composite.saturation_boost;

    // Sample Radiance Cascades (RC) high-frequency GI
    let rc_light = textureSample(rc_texture, post_sampler, sample_uv).rgb;

    // Note: direct emissive is already included in 'base' (added in DoF CoC pass)
    // Apply AO to all lighting (direct + bloom + GI)
    // AO represents how much ambient light reaches a surface, affecting both direct and indirect
    var color = (saturated + bloom * composite.bloom_strength + (indirect_light + rc_light) * composite.indirect_light_scale) * ao;
    color = color * composite.exposure;
    color = max(color, vec3<f32>(0.0));

    if (composite.hdr_highlight_compression > 0.5) {
        color = compress_highlights_hdr(color);
    }

    return vec4<f32>(color, 1.0);
}
