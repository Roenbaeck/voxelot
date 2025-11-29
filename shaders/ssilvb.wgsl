struct SsaoUniforms {
    sample_count: u32,
    slice_count: u32,
    sample_radius: f32,
    hit_thickness: f32,
    screen_width: f32,
    screen_height: f32,
    inverse_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> ssao: SsaoUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var post_sampler: sampler;
@group(0) @binding(3) var emissive_tex: texture_2d<f32>;

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

const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

// Reconstruct view-space position from depth
fn reconstruct_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_z = depth;
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ndc = vec4<f32>(ndc_xy, ndc_z, 1.0);
    let view_pos = ssao.inverse_projection * ndc;
    return view_pos.xyz / view_pos.w;
}

// Fetch depth directly from texture
fn fetch_depth(coord: vec2<i32>) -> f32 {
    return textureLoad(depth_tex, coord, 0);
}

// Reconstruct normal from depth buffer using central differences
fn compute_normal_from_depth(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(ssao.screen_width, ssao.screen_height);
    let p = vec2<i32>(uv * size);
    
    let c0 = fetch_depth(p);
    
    // Check for edge cases
    if (c0 >= 1.0) { return vec3<f32>(0.0, 0.0, 1.0); }

    let l1 = fetch_depth(p - vec2<i32>(1, 0));
    let r1 = fetch_depth(p + vec2<i32>(1, 0));
    let b1 = fetch_depth(p - vec2<i32>(0, 1));
    let t1 = fetch_depth(p + vec2<i32>(0, 1));
    
    let l2 = fetch_depth(p - vec2<i32>(2, 0));
    let r2 = fetch_depth(p + vec2<i32>(2, 0));
    let b2 = fetch_depth(p - vec2<i32>(0, 2));
    let t2 = fetch_depth(p + vec2<i32>(0, 2));
    
    let dl = abs((2.0 * l1 - l2) - c0);
    let dr = abs((2.0 * r1 - r2) - c0);
    let db = abs((2.0 * b1 - b2) - c0);
    let dt = abs((2.0 * t1 - t2) - c0);
    
    let ce = reconstruct_position(uv, c0);
    
    let dpdx = select(
        -ce + reconstruct_position(uv + vec2<f32>(1.0 / size.x, 0.0), r1),
        ce - reconstruct_position(uv - vec2<f32>(1.0 / size.x, 0.0), l1),
        dl < dr
    );
    
    let dpdy = select(
        -ce + reconstruct_position(uv + vec2<f32>(0.0, 1.0 / size.y), t1),
        ce - reconstruct_position(uv - vec2<f32>(0.0, 1.0 / size.y), b1),
        db < dt
    );

    return normalize(cross(dpdx, dpdy));
}

// Interleaved Gradient Noise
fn ign(uv: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(uv, vec2<f32>(0.06711056, 0.00583715))));
}

fn fast_acos(x: f32) -> f32 {
    let out_val = -0.156583 * abs(x) + HALF_PI;
    let res = out_val * sqrt(1.0 - abs(x));
    return select(PI - res, res, x >= 0.0);
}

fn count_bits(value: u32) -> u32 {
    var v = value;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    return (((v + (v >> 4u)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24u;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, post_sampler, uv);
    if (depth >= 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let view_pos = reconstruct_position(uv, depth);
    let normal = compute_normal_from_depth(uv);
    let view_vec = normalize(-view_pos); // View vector pointing to camera (0,0,0) in view space

    let screen_size = vec2<f32>(ssao.screen_width, ssao.screen_height);
    let frag_coord = uv * screen_size;
    
    // Random rotation
    let noise = ign(frag_coord);
    
    let sample_count = f32(ssao.sample_count);
    let slice_count = f32(ssao.slice_count);
    let radius = ssao.sample_radius;
    
    var visibility = 0.0;
    var accumulated_light = vec3<f32>(0.0);
    
    for (var slice = 0u; slice < ssao.slice_count; slice = slice + 1u) {
        let phi = (PI / slice_count) * (f32(slice) + noise);
        let slice_dir = vec2<f32>(cos(phi), sin(phi));
        
        // Project normal onto slice plane
        // Actually, let's follow the reference logic more closely for the slice construction
        
        // Reference uses search in 2D screen space along the slice direction
        // UV Y is down, View Space Y is up. So we must flip Y for the search direction in UV space.
        let search_dir = vec2<f32>(slice_dir.x, -slice_dir.y);
        
        // Calculate tangent angle of the surface in the slice plane
        // We need to project the view-space normal onto the slice plane defined by view_vec and search_dir
        // But simpler GTAO/SSILVB often just marches in screen space.
        
        // Let's use the reference's horizon search
        
        // Construct slice plane basis
        // We are working in View Space mostly.
        // Slice direction in view space (approximate)
        let slice_dir_vs = vec3<f32>(slice_dir, 0.0);
        
        // Compute projected normal on the slice plane
        let plane_n = normalize(cross(slice_dir_vs, view_vec));
        let proj_n = normal - plane_n * dot(normal, plane_n);
        let proj_n_len = length(proj_n);
        
        var cos_n = 0.0;
        var sin_n = 1.0;
        var n_angle = 0.0;

        if (proj_n_len > 0.001) {
            let pn = proj_n / proj_n_len;
            cos_n = dot(pn, view_vec); // cos of angle between normal and view vector
            // Calculate sign of angle
            let t = cross(plane_n, pn);
            let sgn = select(1.0, -1.0, dot(view_vec, t) < 0.0);
            n_angle = sgn * fast_acos(cos_n);
        }
        
        // Horizon search
        var occlusion_bits = 0u;
        
        // Two directions: -1 and 1
        for (var side = 0u; side < 2u; side = side + 1u) {
            let direction = select(-1.0, 1.0, side == 0u);
            let ray_dir = search_dir * direction;
            
            // Logarithmic stepping
            let step_factor = pow(radius, 1.0 / sample_count);
            // Reference uses view space radius but steps in screen space.
            // Let's map radius to screen space approximately.
            // Screen space radius ~= radius / view_z * projection_scale
            // For simplicity, let's treat radius as world/view units and project.
            
            // Approximate screen radius at this depth
            // This is a simplification.
            let proj_scale = ssao.inverse_projection[1][1] * ssao.screen_height; // Approx
            let screen_radius = (radius * proj_scale) / -view_pos.z;
            
            // If screen radius is too small, skip
            // if (screen_radius < 2.0) { continue; } // Removed culling to allow distant lights
            
            let step_ratio = pow(screen_radius, 1.0 / sample_count);
            var current_step = 1.0; // Start at 1 pixel offset
            
            // Jitter starting position
            // current_step *= pow(step_ratio, noise); 
            // Actually reference does: t = pow(s, rnd); where s is step factor.
            
            current_step = pow(step_ratio, select(noise, 1.0 - noise, side == 1u));
            
            var last_horizon_angle = n_angle;

            // Hardcoded loop limit increase for better long-range sampling
            for (var s = 0u; s < 64u; s = s + 1u) {
                if (s >= ssao.sample_count) { break; } 

                let sample_uv = uv + (ray_dir * current_step) / screen_size;
                current_step *= step_ratio;
                
                if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) { break; }
                
                let sample_depth = textureSampleLevel(depth_tex, post_sampler, sample_uv, 0);
                let sample_pos = reconstruct_position(sample_uv, sample_depth);
                
                let delta = sample_pos - view_pos;
                let dist_sq = dot(delta, delta);
                let dist = sqrt(dist_sq);
                let dist_vec = delta / dist;
                
                // Horizon angle
                let horizon_cos = dot(dist_vec, view_vec);
                let horizon_angle = fast_acos(horizon_cos) * direction;
                
                // Thickness heuristic
                let back_horizon_cos = dot(normalize(delta - view_vec * ssao.hit_thickness), view_vec);
                let back_horizon_angle = fast_acos(back_horizon_cos) * direction;
                
                // Convert to [0, 1] relative to normal
                let h1 = clamp((horizon_angle + n_angle) / PI + 0.5, 0.0, 1.0);
                let h2 = clamp((back_horizon_angle + n_angle) / PI + 0.5, 0.0, 1.0);
                
                let min_h = min(h1, h2);
                let max_h = max(h1, h2);
                
                // Bitmask
                // 32 bits represent [0, 1] range
                let start_bit = u32(min_h * 32.0);
                let end_bit = u32(max_h * 32.0);
                
                // Create mask for range [start_bit, end_bit]
                // Be careful with shifts > 31
                if (start_bit < 32u) {
                    let count = min(end_bit - start_bit, 32u - start_bit);
                    if (count > 0u) {
                        let mask = (0xFFFFFFFFu >> (32u - count)) << start_bit;
                        occlusion_bits = occlusion_bits | mask;
                    }
                }

                // Indirect Lighting Accumulation
                // If this sample is "above" the previous horizon, it contributes light
                if (horizon_angle > last_horizon_angle) {
                    let visible_angle = horizon_angle - last_horizon_angle;
                    
                    // Sample emissive texture
                    let emissive_sample = textureSampleLevel(emissive_tex, post_sampler, sample_uv, 0);
                    let emissive_color = emissive_sample.rgb;
                    let emissive_strength = emissive_sample.a; // Assuming alpha contains strength or similar

                    if (length(emissive_color) > 0.0) {
                        // Inverse square falloff + solid angle approximation
                        // visible_angle is the angular size of the visible segment
                        // We also attenuate by distance to avoid over-contribution from far sources
                        let attenuation = 1.0 / (1.0 + dist_sq * 0.02); 
                        
                        accumulated_light += emissive_color * visible_angle * attenuation * 1.0; // No boost
                    }
                    
                    last_horizon_angle = horizon_angle;
                }
            }
        }
        
        let occluded_fraction = f32(count_bits(occlusion_bits)) / 32.0;
        visibility += 1.0 - occluded_fraction;
    }
    
    visibility /= slice_count;
    accumulated_light /= slice_count;
    
    // Chromatic noise reduction:
    // Use the center pixel's emissive color as a base to avoid dark "holes" in the lighting
    // This makes noise appear as brightness variations within the same color instead of color vs black
    let center_emissive = textureSampleLevel(emissive_tex, post_sampler, uv, 0).rgb;
    
    // If we have very little accumulated light, blend towards the center emissive color
    // This makes sparse sampling look like "dimmer version of the right color" instead of "black"
    let light_strength = min(length(accumulated_light), 1.0);
    let final_light = mix(center_emissive * 0.3, accumulated_light, light_strength);
    
    // Apply strength/contrast
    visibility = pow(visibility, 2.0); // Ad-hoc contrast
    
    return vec4<f32>(final_light, visibility);
}
