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
    sdr_tonemap_mode: f32, // 0 = Reinhard, 1 = ACES-like (default)
    hzb_debug: f32,
    hzb_mips: f32,
    near: f32,
    far: f32,
    ssr_enabled: f32,  // Whether to apply SSR reflections
    gi_combined_debug: f32,
    radiance_cascades_debug: f32,
    gi_probes_debug: f32,
    gi_ssgi_debug: f32,
    rc_active: f32,  // 1 when the radiance-cascade texture can be non-zero
    indirect_albedo: f32,  // 1 = screen-space indirect light is multiplied by albedo * haze transmittance
    ambient_only_ao: f32,  // 1 = contact AO only darkens the occludable share of the surface light
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

// Simple global Reinhard tone-mapper for SDR/sRGB presentation.
// This is used when HDR presentation is not active (i.e., we are rendering
// to an sRGB swapchain and need to compress HDR-range colors to [0,1].)
fn tone_map_sdr(color: vec3<f32>) -> vec3<f32> {
    // Luma-based Reinhard to preserve chroma:
    // - map luminance with Reinhard
    // - scale RGB by the luminance ratio
    // This avoids the common “gray filter” look of per-channel Reinhard.
    let luma_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
    let y = max(dot(color, luma_weights), 1e-6);
    let y_mapped = y / (1.0 + y);
    let scale = y_mapped / y;
    return color * scale;
}

// ACES-like filmic curve (more punchy, preserves saturation and highlights).
// Uses a simple rational polynomial approximation that's inexpensive and
// produces similar results to common ACES approximations.
fn tone_map_aces(color: vec3<f32>) -> vec3<f32> {
    // Uncharted2 / ACES-like approximation constants
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;
    let x = color;
    let num = x * (a * x + vec3<f32>(b));
    let den = x * (c * x + vec3<f32>(d)) + vec3<f32>(e);
    return clamp(num / den, vec3<f32>(0.0), vec3<f32>(1.0));
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
// Full-resolution scene depth (render-target size) for joint-bilateral upsampling.
@group(0) @binding(8) var scene_depth: texture_depth_2d;
// Surface G-buffer (render-target size): rgb = albedo * haze transmittance, a = share of the
// pixel's radiance that is occludable surface light (sky ambient, moon, probe GI).
@group(0) @binding(9) var surface_gbuffer: texture_2d<f32>;

// Relative view-depth difference at which an upsample tap's weight falls to 1/e is
// 1/UPSAMPLE_DEPTH_SHARPNESS.
const UPSAMPLE_DEPTH_SHARPNESS: f32 = 32.0;
const SKY_LINEAR_DEPTH: f32 = 1.0e30;

fn linear_depth_at_pixel(pixel: vec2<i32>) -> f32 {
    let d = textureLoad(scene_depth, pixel, 0);
    if (d >= 1.0) {
        return SKY_LINEAR_DEPTH;
    }
    let n = composite.near;
    let f = composite.far;
    return (n * f) / max(f - d * (f - n), 1e-6);
}

// Joint-bilateral upsample of a lighting buffer rendered at 1/scale of the scene depth size,
// whose texel t was shaded at full-resolution pixel t*scale + scale/2 (see ssilvb.wgsl).
// Bilinear weights keep surfaces smooth; the relative-depth term rejects texels from another
// surface, so AO/GI does not smear across silhouettes or into the sky.
fn upsample_bilateral(tex: texture_2d<f32>, scale: i32, uv: vec2<f32>, center_depth: f32) -> vec4<f32> {
    let full_size = vec2<i32>(textureDimensions(scene_depth));
    let low_size = vec2<i32>(textureDimensions(tex));
    let pixel = uv * vec2<f32>(full_size) - vec2<f32>(0.5);
    let position = (pixel - vec2<f32>(f32(scale / 2))) / f32(scale);
    let base = vec2<i32>(floor(position));
    let fraction = position - floor(position);
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    var nearest = vec4<f32>(0.0);
    var nearest_difference = 1.0e38;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let texel = clamp(base + vec2<i32>(x, y), vec2<i32>(0), low_size - vec2<i32>(1));
            let source = clamp(
                texel * scale + vec2<i32>(scale / 2),
                vec2<i32>(0),
                full_size - vec2<i32>(1)
            );
            let difference = abs(linear_depth_at_pixel(source) - center_depth) / center_depth;
            let bilinear = select(fraction.x, 1.0 - fraction.x, x == 0)
                * select(fraction.y, 1.0 - fraction.y, y == 0);
            let weight = (bilinear + 1.0e-3) * exp(-difference * UPSAMPLE_DEPTH_SHARPNESS);
            let value = textureLoad(tex, texel, 0);
            sum += value * weight;
            weight_sum += weight;
            if (difference < nearest_difference) {
                nearest_difference = difference;
                nearest = value;
            }
        }
    }
    return select(nearest, sum / weight_sum, weight_sum > 1.0e-4);
}

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

    // Depth of the scene pixel this output pixel shows (sky: no AO/GI).
    let depth_size = vec2<i32>(textureDimensions(scene_depth));
    let center_pixel = clamp(
        vec2<i32>(sample_uv * vec2<f32>(depth_size)),
        vec2<i32>(0),
        depth_size - vec2<i32>(1)
    );
    let center_depth = linear_depth_at_pixel(center_pixel);
    let is_sky = center_depth >= SKY_LINEAR_DEPTH;

    // Sample SSILVB (half resolution): RGB = accumulated emissive light, A = ambient occlusion
    var ssilvb_sample = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (!is_sky) {
        ssilvb_sample = upsample_bilateral(ssao_texture, 2, sample_uv, center_depth);
    }
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
            // Show accumulated indirect light (SSGI) on the left (unscaled for debug visibility)
            return vec4<f32>(indirect_light, 1.0);
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

    let luma_weights = vec3<f32>(0.2126, 0.7152, 0.0722);

    // Sample Radiance Cascades (RC) GI (render-target resolution), depth-aware as well since
    // the presented image is rescaled from the render target.
    var rc_light = vec3<f32>(0.0);
    if (!is_sky && composite.rc_active > 0.5) {
        rc_light = upsample_bilateral(rc_texture, 1, sample_uv, center_depth).rgb;
    }

    // Sum everything before saturation and tonemapping to match old "punchy" look.
    // Note: direct emissive is already included in 'base' (added in DoF CoC pass).
    // The screen-space indirect light is irradiance: it reaches the eye after reflecting off
    // the surface (albedo) and through the same haze as direct light. Sky pixels have none.
    let surface = textureLoad(surface_gbuffer, center_pixel, 0);
    var indirect_albedo = vec3<f32>(1.0);
    if (composite.indirect_albedo > 0.5) {
        indirect_albedo = surface.rgb;
    }
    let indirect_sum = (indirect_light + rc_light) * composite.indirect_light_scale * indirect_albedo;

    // SSILVB AO is a near-field screen-space effect. Applying it to the entire
    // HDR radiance sum turns its finite radius into a camera-centered shadow
    // volume, especially once WTS has added stable sunlight at distance. Keep
    // real contact occlusion, but suppress broad low-grade haze.
    let occlusion = clamp(1.0 - ao, 0.0, 1.0);
    let contact_occlusion = smoothstep(0.02, 0.18, occlusion) * min(occlusion * 1.15, 0.9);
    let contact_ao = 1.0 - contact_occlusion;
    var color = (base + indirect_sum) * contact_ao + bloom * composite.bloom_strength;
    if (composite.ambient_only_ao > 0.5) {
        // Contact AO only occludes the sky/ambient (non-sunlight) share of the surface light:
        // shadow-mapped sun is already shadowed, and emission and haze are not light that a
        // nearby surface can block. The share (a channel) is the fraction of this pixel's
        // radiance that is occludable surface light, written by the scene shaders. The
        // screen-space indirect light is ambient-like and stays fully occludable.
        color = base - base * (surface.a * contact_occlusion) + indirect_sum * contact_ao
            + bloom * composite.bloom_strength;
    }

    // Sample Radiance Cascades (RC) high-frequency GI for debug overlays (redundant but kept for structure)
    // let rc_light is already sampled above

    // GI Combined Debug overlay
    if (composite.gi_combined_debug > 0.5) {
        // Show the combined indirect light with the same AO used by the final
        // composite so the debug view preserves visible shadowing.
        return vec4<f32>(indirect_light * contact_ao, 1.0);
    }

    // GI Probes Debug overlay
    if (composite.gi_probes_debug > 0.5) {
        return vec4<f32>(indirect_light, 1.0);
    }

    // GI SSGI Debug overlay
    if (composite.gi_ssgi_debug > 0.5) {
        return vec4<f32>(indirect_light, 1.0);
    }

    // RC Debug overlay
    if (composite.radiance_cascades_debug > 0.5) {
        return vec4<f32>(rc_light, 1.0);
    }

    // Apply SSR reflections if enabled
    if (composite.ssr_enabled > 0.5) {
        let ssr_sample = textureSample(ssr_debug_texture, post_sampler, sample_uv);
        let ssr_reflection = ssr_sample.rgb;
        let ssr_strength = ssr_sample.a;
        color = color + ssr_reflection * ssr_strength;
    }

    // SATURATION BOOST (Applied to the final light sum)
    // IMPORTANT: preserve luminance even when saturation_boost > 1.0.
    // Using mix(gray, color, t) with t>1 brightens the image by extrapolation.
    let luma = dot(color, luma_weights);
    let gray = vec3<f32>(luma);
    color = gray + (color - gray) * composite.saturation_boost;

    color = color * composite.exposure;
    color = max(color, vec3<f32>(0.0));

    if (composite.hdr_highlight_compression > 0.5) {
        // Preserve HDR headroom and apply a soft shoulder for EDR/HDR presentation.
        color = compress_highlights_hdr(color);
    } else {
        // SDR path: choose the tonemapper per uniform.
        // 0.0 = None (Clamp), 1.0 = Reinhard, 2.0 = ACES-like
        if (composite.sdr_tonemap_mode > 1.5) {
            color = tone_map_aces(color);
        } else if (composite.sdr_tonemap_mode > 0.5) {
            color = tone_map_sdr(color);
        } else {
            // No tonemapping (clamping only), matches very early project look
            color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }

    return vec4<f32>(color, 1.0);
}
