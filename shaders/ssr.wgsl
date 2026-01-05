// Cubemap-only reflections pass.
// This pass outputs a reflected color (RGB) and reflectivity in A.
// Screen-space ray marching / temporal accumulation are intentionally disabled.

struct CameraUniforms {
    inverse_view: mat4x4<f32>,
    inverse_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    skybox_rotation: f32,
    skybox_brightness: f32,
    skybox_saturation: f32,
    _pad1: vec2<f32>,
    skybox_tint: vec3<f32>,
    skybox_tint_strength: f32,
    // xyz = probe center in world space, w = proxy half-extent for box-projected cubemap sampling
    probe_pos_extent: vec4<f32>,
}

struct SSRParams {
    max_steps: u32,
    max_binary_steps: u32,
    step_size: f32,
    thickness: f32,
    overscan: f32,
    bloom_strength: f32,
    frame_index: f32,
    history_valid: f32,
    probe_valid: f32,
    probe_strength: f32,
    // Debug controls packed here:
    //   x = probe-only debug view (1 = show probe, ignore skybox)
    //   y = flip-Y when sampling probe (1 = invert direction.y)
    _pad2: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: SSRParams;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var hzb_texture: texture_2d<f32>;
@group(0) @binding(6) var hzb_sampler: sampler;
@group(0) @binding(7) var normal_gbuffer: texture_2d<f32>;
@group(0) @binding(8) var material_gbuffer: texture_2d<f32>;
@group(0) @binding(9) var ssao_texture: texture_2d<f32>;
@group(0) @binding(10) var bloom_texture: texture_2d<f32>;
@group(0) @binding(11) var skybox_texture: texture_2d<f32>;
@group(0) @binding(12) var skybox_sampler: sampler;
@group(0) @binding(13) var ssr_history: texture_2d<f32>;
@group(0) @binding(14) var reflection_probe: texture_cube<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    // Fullscreen triangle
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);
    return output;
}

fn decode_world_normal(encoded: vec3<f32>) -> vec3<f32> {
    let n = encoded * 2.0 - 1.0;
    return normalize(select(n, vec3<f32>(0.0, 1.0, 0.0), dot(n, n) < 1e-8));
}

fn uv_to_pixel(uv: vec2<f32>, dim: vec2<u32>) -> vec2<i32> {
    let x = clamp(i32(uv.x * f32(dim.x)), 0, i32(dim.x) - 1);
    let y = clamp(i32(uv.y * f32(dim.y)), 0, i32(dim.y) - 1);
    return vec2<i32>(x, y);
}

fn load_depth_at_uv(uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(scene_depth);
    let px = uv_to_pixel(uv, dim);
    return textureLoad(scene_depth, px, 0);
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let z_ndc = depth * 2.0 - 1.0;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, z_ndc, 1.0);
    let view_pos = camera.inverse_proj * ndc;
    let view_pos_3d = view_pos.xyz / view_pos.w;
    let world_pos = camera.inverse_view * vec4<f32>(view_pos_3d, 1.0);
    return world_pos.xyz;
}

fn intersect_ray_aabb(origin: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> f32 {
    // Returns the first positive intersection distance along the ray, or -1 if no hit.
    let eps = 1e-6;
    let safe_dir = vec3<f32>(
        select(dir.x, eps, abs(dir.x) < eps),
        select(dir.y, eps, abs(dir.y) < eps),
        select(dir.z, eps, abs(dir.z) < eps),
    );
    let inv_dir = 1.0 / safe_dir;
    let t0 = (bmin - origin) * inv_dir;
    let t1 = (bmax - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
    let tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (tmax < max(tmin, 0.0)) {
        return -1.0;
    }
    return select(tmin, tmax, tmin < 0.0);
}

fn parallax_correct_probe_dir(ray_origin: vec3<f32>, reflect_dir: vec3<f32>) -> vec3<f32> {
    let probe_center = camera.probe_pos_extent.xyz;
    let half_extent = camera.probe_pos_extent.w;
    if (half_extent <= 0.0) {
        return reflect_dir;
    }
    let half = vec3<f32>(half_extent);
    let bmin = probe_center - half;
    let bmax = probe_center + half;
    let t = intersect_ray_aabb(ray_origin, reflect_dir, bmin, bmax);
    if (t <= 0.0) {
        return reflect_dir;
    }
    let hit_pos = ray_origin + reflect_dir * t;
    return normalize(hit_pos - probe_center);
}

fn sample_sky_equirect(reflect_dir: vec3<f32>) -> vec3<f32> {
    let angle = camera.skybox_rotation;
    let c = cos(angle);
    let s = sin(angle);
    let rotated_dir = vec3<f32>(
        reflect_dir.x * c + reflect_dir.z * s,
        reflect_dir.y,
        reflect_dir.x * -s + reflect_dir.z * c
    );
    let PI = 3.14159265359;
    let TWO_PI = 6.28318530718;
    let u = 0.5 + atan2(rotated_dir.z, rotated_dir.x) / TWO_PI;
    let v = 0.5 - asin(clamp(rotated_dir.y, -1.0, 1.0)) / PI;
    let sky_sample = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v)).rgb;

    let min_sat = camera.skybox_saturation;
    let brightness = camera.skybox_brightness;
    let sat = clamp(min_sat + (1.0 - min_sat) * brightness, 0.0, 1.0);
    let lum = dot(sky_sample, vec3<f32>(0.299, 0.587, 0.114));
    let desaturated = mix(vec3<f32>(lum), sky_sample, sat);
    let tint = camera.skybox_tint;
    let tint_strength = camera.skybox_tint_strength;
    let effect_strength = (1.0 - brightness) * tint_strength;
    let tinted = mix(desaturated, desaturated * tint, effect_strength);
    return tinted * brightness;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let depth = load_depth_at_uv(input.uv);
    if (depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let material = textureSample(material_gbuffer, linear_sampler, input.uv);
    let reflectivity = material.r;
    if (reflectivity < 0.01) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let world_pos = reconstruct_world_pos(input.uv, depth);
    let gbuf = textureSample(normal_gbuffer, linear_sampler, input.uv);
    let normal = decode_world_normal(gbuf.rgb);

    let view_dir = normalize(world_pos - camera.camera_pos);
    let reflect_dir = reflect(view_dir, normal);

    let sky_color = sample_sky_equirect(reflect_dir);

    var probe_dir = parallax_correct_probe_dir(world_pos, reflect_dir);
    if (params._pad2.y > 0.5) {
        probe_dir.y = -probe_dir.y;
    }
    let probe_color = textureSample(reflection_probe, linear_sampler, probe_dir).rgb;

    if (params._pad2.x > 0.5) {
        return vec4<f32>(probe_color, 1.0);
    }

    let probe_on = (params.probe_valid * params.probe_strength) > 0.5;
    let out_color = select(sky_color, probe_color, probe_on);
    return vec4<f32>(out_color, reflectivity);
}
/*
// Screen Space Reflections shader with HZB acceleration

struct CameraUniforms {
    inverse_view: mat4x4<f32>,
    inverse_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    skybox_rotation: f32,  // Skybox rotation angle in radians
    skybox_brightness: f32,
    skybox_saturation: f32,
    _pad1: vec2<f32>,
    skybox_tint: vec3<f32>,
    skybox_tint_strength: f32,
    // xyz = probe center in world space, w = proxy half-extent for box-projected cubemap sampling
    probe_pos_extent: vec4<f32>,
}

struct SSRParams {
    max_steps: u32,
    max_binary_steps: u32,
    step_size: f32,
    thickness: f32,
    overscan: f32,  // How much extra UV space we can sample (e.g., 0.2 = 20% overscan)
    bloom_strength: f32,  // How much bloom to add to reflected colors
    frame_index: f32,
    history_valid: f32,
    probe_valid: f32,
    probe_strength: f32,
    _pad2: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: SSRParams;
@group(0) @binding(2) var scene_color: texture_2d<f32>;

    return vec4<f32>(fallback_color, reflectivity);
    
    // Screen edges (x = +/-w, y = +/-w)
    if (abs(dir.x - dir.w) > 1e-6) {
        let t_x_pos = (start.w - start.x) / (dir.x - dir.w);
        if (t_x_pos > 0.0 && t_x_pos < 1.0) { t = min(t, t_x_pos); }
    }
    if (abs(dir.x + dir.w) > 1e-6) {
        let t_x_neg = (-start.w - start.x) / (dir.x + dir.w);
        if (t_x_neg > 0.0 && t_x_neg < 1.0) { t = min(t, t_x_neg); }
    }
    if (abs(dir.y - dir.w) > 1e-6) {
        let t_y_pos = (start.w - start.y) / (dir.y - dir.w);
        if (t_y_pos > 0.0 && t_y_pos < 1.0) { t = min(t, t_y_pos); }
    }
    if (abs(dir.y + dir.w) > 1e-6) {
        let t_y_neg = (-start.w - start.y) / (dir.y + dir.w);
        if (t_y_neg > 0.0 && t_y_neg < 1.0) { t = min(t, t_y_neg); }
    }
    
    // Far plane (z = w)
    if (abs(dir.z - dir.w) > 1e-6) {
        let t_z_pos = (start.w - start.z) / (dir.z - dir.w);
        if (t_z_pos > 0.0 && t_z_pos < 1.0) { t = min(t, t_z_pos); }
    }
    // WGPU depth range [0, w]
    if (abs(dir.z) > 1e-6) {
        let t_z_neg = -start.z / dir.z;
        if (t_z_neg > 0.0 && t_z_neg < 1.0) { t = min(t, t_z_neg); }
    }
    
    return start + dir * t;
}

// New SSR implementation based on zznewclear13's article
// Returns vec4: xy = hit UV, z = hit confidence (0 or 1), w = unused
fn ssr_ray_march(start_world: vec3<f32>, reflect_dir: vec3<f32>, start_uv: vec2<f32>) -> vec4<f32> {
    let max_dist = 200.0;
    let end_world = start_world + reflect_dir * max_dist;

    let start_clip = camera.view_proj * vec4<f32>(start_world, 1.0);
    var end_clip = camera.view_proj * vec4<f32>(end_world, 1.0);

    // Clip the ray to the frustum
    end_clip = frustum_clip(start_clip, end_clip);

    let q0 = start_clip.xyz / start_clip.w;
    let q1 = end_clip.xyz / end_clip.w;

    let start_screen = vec3<f32>(q0.x * 0.5 + 0.5, 0.5 - q0.y * 0.5, q0.z);
    let end_screen = vec3<f32>(q1.x * 0.5 + 0.5, 0.5 - q1.y * 0.5, q1.z);

    let num_steps = max(1u, params.max_steps);
    let delta = (end_screen - start_screen) / f32(num_steps);

    // Screen-space step magnitude for adaptive thickness
    let dim_u = textureDimensions(scene_color);
    let dim = vec2<f32>(f32(dim_u.x), f32(dim_u.y));
    let screen_delta_length = length(delta.xy * dim);
    let screen_space_factor = max(1.0, screen_delta_length * 0.5);

    // Jitter the start along the ray in screen-space to avoid visible marching bands.
    // Keep this stable per-pixel to avoid extra temporal noise on rigid surfaces.
    let jitter = fract(sin(dot(start_uv * dim, vec2<f32>(127.1, 311.7))) * 43758.5453123);
    var current_screen = start_screen + delta * jitter;

    let hzb_levels = max(1, textureNumLevels(hzb_texture));
    let max_mip = i32(hzb_levels) - 1;
    var current_mip = min(3, max_mip);

    for (var i = 0u; i < num_steps; i++) {
        current_screen += delta;
        let uv = current_screen.xy;

        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }

        let ray_depth = current_screen.z;
        let hzb_depth = load_hzb_depth_at_uv(uv, current_mip);

        // Potential intersection when ray is behind the min-depth in this block.
        if (ray_depth > hzb_depth) {
            if (current_mip > 0) {
                // Descend to finer mip and backtrack one step so we re-check at higher detail.
                current_mip -= 1;
                current_screen -= delta;
                continue;
            }

            // Full-res depth hit-test (unfiltered)
            let sampled_depth = load_depth_at_uv(uv);
            if (ray_depth > sampled_depth) {
                // Thickness test in linear depth
                let ray_linear_depth = get_linear_depth(ray_depth);
                let sampled_linear_depth = get_linear_depth(sampled_depth);

                // Adaptive thickness: scale by distance (via depth) and screen-space step size.
                let thickness = (params.thickness + 0.02 * sampled_linear_depth) * screen_space_factor;
                if (ray_linear_depth - sampled_linear_depth < thickness) {
                    // Binary search around the last step interval.
                    var low = current_screen - delta;
                    var high = current_screen;
                    for (var j = 0u; j < params.max_binary_steps; j++) {
                        let mid = (low + high) * 0.5;
                        let mid_uv = mid.xy;
                        if (mid_uv.x < 0.0 || mid_uv.x > 1.0 || mid_uv.y < 0.0 || mid_uv.y > 1.0) {
                            low = mid;
                            continue;
                        }
                        let mid_depth = load_depth_at_uv(mid_uv);
                        if (mid.z > mid_depth) {
                            high = mid;
                        } else {
                            low = mid;
                        }
                    }
                    return vec4<f32>(high.xy, 1.0, 0.0);
                }
            }
        } else {
            // No intersection at this block; optionally ascend to a coarser mip for speed.
            if (current_mip < min(3, max_mip)) {
                current_mip += 1;
            }
        }
    }

    return vec4<f32>(-1.0, -1.0, -1.0, 0.0);
}


@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel_size = vec2<f32>(1.0) / vec2<f32>(textureDimensions(scene_color));
    let depth = load_depth_at_uv(input.uv);
    
    // Early out for sky (depth = 1.0)
    if (depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Sample material reflectivity from G-buffer
    let material = textureSample(material_gbuffer, linear_sampler, input.uv);
    let reflectivity = material.r;
    
    // Early out for non-reflective materials
    if (reflectivity < 0.01) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let world_pos = reconstruct_world_pos(input.uv, depth);
    // Use the G-buffer world normal instead of estimating from depth.
    // This is cheaper and much more stable near edges.
    let gbuf = textureSample(normal_gbuffer, linear_sampler, input.uv);
    let normal = decode_world_normal(gbuf.rgb);
    let view_dir = normalize(world_pos - camera.camera_pos);
    
    // Reflect view direction around normal
    let reflect_dir = reflect(view_dir, normal);
    
    // Calculate grazing angle factor - reflections at steep angles are less stable
    // View dir and normal are normalized, dot gives cosine of angle
    let grazing = 1.0 - abs(dot(view_dir, normal));  // 0 = head-on, 1 = grazing
    let grazing_fade = smoothstep(0.85, 0.95, grazing);  // Fade out at very grazing angles
    
    // Combined fade factor
    let angle_fade = grazing_fade;
    
    // NOTE: Cubemap-only reflections (probe fallback). Screen-space ray marching is disabled.

    // Calculate Skybox Reflection (always needed for fallback/mixing)
    // Apply skybox rotation to match the scene
    let angle = camera.skybox_rotation;
    let c = cos(angle);
    let s = sin(angle);
    let rotated_dir = vec3<f32>(
        reflect_dir.x * c + reflect_dir.z * s,
        reflect_dir.y,
        reflect_dir.x * -s + reflect_dir.z * c
    );
    
    // Convert reflection direction to equirectangular UV (matching skybox and water shaders)
    let PI = 3.14159265359;
    let TWO_PI = 6.28318530718;
    let u = 0.5 + atan2(rotated_dir.z, rotated_dir.x) / TWO_PI;
    let v = 0.5 - asin(clamp(rotated_dir.y, -1.0, 1.0)) / PI;
    
    // Use skybox_sampler which has Repeat on U for proper equirectangular wrapping
    let sky_sample = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v)).rgb;
    
    // Apply brightness and desaturation (match skybox pass)
    let min_sat = camera.skybox_saturation;
    let brightness = camera.skybox_brightness;
    let sat = clamp(min_sat + (1.0 - min_sat) * brightness, 0.0, 1.0);
    let luminance = dot(sky_sample, vec3<f32>(0.299, 0.587, 0.114));
    let desaturated = mix(vec3<f32>(luminance), sky_sample, sat);
    let tint = camera.skybox_tint;
    let tint_strength = camera.skybox_tint_strength;
    let effect_strength = (1.0 - brightness) * tint_strength;
    let tinted = mix(desaturated, desaturated * tint, effect_strength);
    let sky_color = tinted * brightness;

    // Reflection probe fallback (contains skybox + cached geometry from probe views).
    // Debug controls are packed into params._pad2:
    //   _pad2.x = probe-only debug view (1 = show probe, ignore SSR)
    //   _pad2.y = flip-Y when sampling the probe (1 = invert direction.y)
    var probe_dir = parallax_correct_probe_dir(world_pos, reflect_dir);
    if (params._pad2.y > 0.5) {
        probe_dir.y = -probe_dir.y;
    }
    let probe_color = textureSample(reflection_probe, linear_sampler, probe_dir).rgb;
    // The probe already includes the skybox, so when valid we prefer it directly.
    let probe_on = (params.probe_valid * params.probe_strength) > 0.5;
    let fallback_color = select(sky_color, probe_color, probe_on);

    if (params._pad2.x > 0.5) {
        return vec4<f32>(probe_color, 1.0);
    }

    if (hit_result.z > 0.0) {
        let hit_uv = hit_result.xy;
        let dim = vec2<f32>(textureDimensions(scene_color));
        let texel = 1.0 / dim;
        
        // Sample G-buffer at hit point for stable distance and normal check
        let hit_dim = textureDimensions(normal_gbuffer);
        let hit_px = uv_to_pixel(hit_uv, hit_dim);
        let hit_gbuf = textureLoad(normal_gbuffer, hit_px, 0);
        let hit_normal = decode_world_normal(hit_gbuf.rgb);
        let reflection_distance = hit_gbuf.a; // view_z stored in alpha
        
        // Facing check: reject hits on surfaces facing away from the ray
        let facing = dot(hit_normal, -reflect_dir);
        let facing_factor = smoothstep(0.0, 0.2, facing);
        
        // Distance-based blur: further reflections get more blur to hide instability
        let distance_blur_start = 5.0;   // Start blurring after this distance
        let distance_blur_end = 80.0;    // Full blur at this distance
        let distance_blur_factor = smoothstep(distance_blur_start, distance_blur_end, reflection_distance);
        
        // Combine with roughness (less reflective = more blur)
        let roughness = 1.0 - reflectivity;
        
        // Final blur spread: distance-based + roughness-based
        var blur_factor = max(distance_blur_factor, roughness);
        let blur_spread = 1.0 + blur_factor * 3.0;
        
        // Start with sharp sample
        var reflection_color = textureSample(scene_color, linear_sampler, hit_uv).rgb;
        var bloom_accum = textureSample(bloom_texture, linear_sampler, hit_uv).rgb;
        var ao_accum = textureSample(ssao_texture, linear_sampler, hit_uv).a;

        // Edge-aware filtering: silhouettes and depth/normal discontinuities are where SSR is
        // most unstable (sparkle/noise). Water looks better partly because it always applies
        // a small gather in these cases.
        // We detect an edge via depth/normal deltas in a 4-neighborhood and enforce a small
        // bilateral gather even when the material is glossy.
        let center_depth = load_depth_at_uv(hit_uv);
        let center_lin = get_linear_depth(center_depth);
        let n_dim = textureDimensions(normal_gbuffer);

        var max_depth_delta = 0.0;
        var max_norm_delta = 0.0;

        for (var k = 0; k < 4; k = k + 1) {
            let o = select(
                vec2<f32>(texel.x, 0.0),
                select(vec2<f32>(-texel.x, 0.0), select(vec2<f32>(0.0, texel.y), vec2<f32>(0.0, -texel.y), k == 2), k == 1),
                k == 0
            );
            let uv_n = hit_uv + o;
            if (uv_n.x < 0.0 || uv_n.x > 1.0 || uv_n.y < 0.0 || uv_n.y > 1.0) {
                max_depth_delta = max(max_depth_delta, 5.0);
                max_norm_delta = max(max_norm_delta, 1.0);
                continue;
            }
            let d_n = load_depth_at_uv(uv_n);
            if (d_n >= 0.9999) {
                // Neighbor is sky -> strong edge
                max_depth_delta = max(max_depth_delta, 5.0);
                max_norm_delta = max(max_norm_delta, 1.0);
                continue;
            }
            let lin_n = get_linear_depth(d_n);
            max_depth_delta = max(max_depth_delta, abs(lin_n - center_lin));

            let px_n = uv_to_pixel(uv_n, n_dim);
            let gbuf_n = textureLoad(normal_gbuffer, px_n, 0);
            let n_n = decode_world_normal(gbuf_n.rgb);
            max_norm_delta = max(max_norm_delta, 1.0 - clamp(dot(hit_normal, n_n), 0.0, 1.0));
        }

        let depth_edge = smoothstep(0.20, 1.00, min(max_depth_delta, 5.0));
        let norm_edge = smoothstep(0.15, 0.45, max_norm_delta);
        let edge_instability = max(depth_edge, norm_edge);
        blur_factor = max(blur_factor, edge_instability * 0.35);
        
        if (blur_factor > 0.01) {
            var accum_color = vec3<f32>(0.0);
            var accum_bloom = vec3<f32>(0.0);
            var accum_ao = 0.0;
            var weight_sum = 0.0;

            // Apply 3x3 gather blur with depth rejection and improved weighting
            for (var oy = -1; oy <= 1; oy = oy + 1) {
                for (var ox = -1; ox <= 1; ox = ox + 1) {
                    let offset_uv = hit_uv + vec2<f32>(f32(ox), f32(oy)) * texel * blur_spread;
                    
                    // UV bounds check
                    if (offset_uv.x < 0.0 || offset_uv.x > 1.0 || offset_uv.y < 0.0 || offset_uv.y > 1.0) {
                        continue;
                    }

                    // Depth rejection: don't blur with sky
                    let s_dim = textureDimensions(scene_depth);
                    let s_coords = uv_to_pixel(offset_uv, s_dim);
                    let s_depth = textureLoad(scene_depth, s_coords, 0);
                    if (s_depth >= 0.9999) { continue; }

                    // Bilateral rejection: avoid bleeding across silhouettes
                    let s_lin = get_linear_depth(s_depth);
                    let depth_gate = 0.25 + 0.02 * center_lin;
                    if (abs(s_lin - center_lin) > depth_gate) { continue; }

                    let s_n_px = uv_to_pixel(offset_uv, n_dim);
                    let s_n_gbuf = textureLoad(normal_gbuffer, s_n_px, 0);
                    let s_n = decode_world_normal(s_n_gbuf.rgb);
                    if (dot(hit_normal, s_n) < 0.6) { continue; }

                    let sample_color = textureSample(scene_color, linear_sampler, offset_uv).rgb;
                    let sample_bloom = textureSample(bloom_texture, linear_sampler, offset_uv).rgb;
                    let sample_ssao = textureSample(ssao_texture, linear_sampler, offset_uv);
                    
                    // Improved weighting: suppress HDR fireflies but don't emphasize black holes
                    let lum = luminance(sample_color + sample_bloom * params.bloom_strength);
                    // Only suppress if lum > 1.0 (HDR)
                    var w = 1.0 / (1.0 + max(0.0, lum - 1.0));

                    // Additional bilateral weighting inside the accepted neighborhood
                    // (soft falloff rather than hard reject only).
                    let depth_w = exp(-abs(s_lin - center_lin) * 2.0);
                    let norm_w = pow(clamp(dot(hit_normal, s_n), 0.0, 1.0), 4.0);
                    w *= depth_w * norm_w;
                    
                    accum_color += sample_color * w;
                    accum_bloom += sample_bloom * w;
                    accum_ao += sample_ssao.a * w;
                    weight_sum += w;
                }
            }
            
            if (weight_sum > 0.0) {
                let gathered_color = accum_color / weight_sum;
                let gathered_bloom = accum_bloom / weight_sum;
                let gathered_ao = accum_ao / weight_sum;
                
                reflection_color = mix(reflection_color, gathered_color, blur_factor);
                bloom_accum = mix(bloom_accum, gathered_bloom, blur_factor);
                ao_accum = mix(ao_accum, gathered_ao, blur_factor);
            }
        }
        
        // Apply bloom and SSAO
        let bloomed_reflection = reflection_color + bloom_accum * params.bloom_strength;
        return vec4<f32>(fallback_color, reflectivity);
    */