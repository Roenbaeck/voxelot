struct CompositeUniforms {
    bloom_strength: f32,
    saturation_boost: f32,
    exposure: f32,
    ssao_enabled: f32,
    ssao_debug: f32,
    ssao_strength: f32,
    ssr_debug: f32,
    indirect_light_scale: f32,
    hdr_highlight_compression: f32,
    hzb_debug: f32,
    hzb_mips: f32,
    near: f32,
    far: f32,
    ssr_enabled: f32,  // Whether to apply SSR reflections
    _pad: vec2<f32>,
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
@group(0) @binding(7) var hzb_texture: texture_2d<f32>;
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

    // Optional debug overlay: show SSAO/SSGI when ssao_debug is set
    if (composite.ssao_debug > 0.5) {
        // Map sample_uv (0..1 over whole texture) to viewport_uv (0..1 over viewport)
        let viewport_uv = (sample_uv - composite.uv_offset) / composite.uv_scale;
        
        if (viewport_uv.x < 0.5) {
            // Show accumulated indirect light (SSGI) on the left
            return vec4<f32>(indirect_light * composite.indirect_light_scale, 1.0);
        } else {
            // Show raw AO as greyscale on the right
            return vec4<f32>(vec3<f32>(raw_ao), 1.0);
        }
    }

    // SSR Debug overlay: show SSR texture directly when enabled
    if (composite.ssr_debug > 0.5) {
        let ssr_col = textureSample(ssr_debug_texture, post_sampler, sample_uv);
        return vec4<f32>(ssr_col.rgb, 1.0);
    }

    // HZB debug grid: splits screen into 4x4 tiles showing mip 0..15
    if (composite.hzb_debug > 0.5) {
        let tiles: f32 = 4.0;
        let tiles_i: u32 = 4u;

        // Map sample_uv (0..1 over whole texture) to viewport_uv (0..1 over viewport)
        let viewport_uv = (sample_uv - composite.uv_offset) / composite.uv_scale;

        if (viewport_uv.x >= 0.0 && viewport_uv.x <= 1.0 && viewport_uv.y >= 0.0 && viewport_uv.y <= 1.0) {
            let cell = vec2<u32>(u32(floor(viewport_uv.x * tiles)), u32(floor(viewport_uv.y * tiles)));
            let mip_idx = cell.x + cell.y * tiles_i;
            let mip_f = f32(mip_idx);

            // local uv inside cell
            let local_uv = fract(viewport_uv * tiles);

            // clamp mip to available mips
            if (mip_f < composite.hzb_mips) {
                // Sample HZB at specified mip
                // We sample the viewport part of the HZB mip for each cell
                let hzb_uv = local_uv * composite.uv_scale + composite.uv_offset;
                let depth_sample = textureSampleLevel(hzb_texture, post_sampler, hzb_uv, mip_f).r;

                // Draw thin borders for cells
                let border = 0.02;
                let edge = step(local_uv.x, border) + step(local_uv.y, border) + step(1.0 - local_uv.x, border) + step(1.0 - local_uv.y, border);
                let border_col = vec3<f32>(0.0, 0.0, 0.0);

                // Linearize depth for better visualization
                // Standard depth: d = f/(f-n) - (f*n)/((f-n)*z)
                // z = (f*n) / (f - d*(f-n))
                let n = composite.near;
                let f = composite.far;
                let z_linear = (f * n) / (f - depth_sample * (f - n));

                // Logarithmic mapping to see detail across the whole range [near, far]
                let fill_val = (log2(z_linear) - log2(n)) / (log2(f) - log2(n));
                let fill = vec3<f32>(fill_val);

                let col = mix(fill, border_col, clamp(edge, 0.0, 1.0));
                return vec4<f32>(col, 1.0);
            }
        }
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

    // Apply SSR reflections if enabled
    // The SSR texture contains (reflection_color.rgb, reflectivity * edge_fade)
    if (composite.ssr_enabled > 0.5) {
        let ssr_sample = textureSample(ssr_debug_texture, post_sampler, sample_uv);
        let ssr_reflection = ssr_sample.rgb;
        let ssr_strength = ssr_sample.a;
        // Add reflections on top of the base color, scaled by strength
        // This gives partial reflections for lower reflectivity materials
        color = color + ssr_reflection * ssr_strength;
    }

    color = color * composite.exposure;
    color = max(color, vec3<f32>(0.0));

    if (composite.hdr_highlight_compression > 0.5) {
        color = compress_highlights_hdr(color);
    }

    return vec4<f32>(color, 1.0);
}
