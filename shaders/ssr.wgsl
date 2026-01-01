// Screen Space Reflections shader with HZB acceleration

struct CameraUniforms {
    inverse_view: mat4x4<f32>,
    inverse_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    skybox_rotation: f32,  // Skybox rotation angle in radians
    skybox_brightness: f32,
    skybox_saturation: f32,
    _pad1: vec2<f32>,
    skybox_tint: vec3<f32>,
    skybox_tint_strength: f32,
}

struct SSRParams {
    max_steps: u32,
    max_binary_steps: u32,
    step_size: f32,
    thickness: f32,
    overscan: f32,  // How much extra UV space we can sample (e.g., 0.2 = 20% overscan)
    bloom_strength: f32,  // How much bloom to add to reflected colors
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: SSRParams;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var hzb_texture: texture_2d<f32>;
@group(0) @binding(6) var hzb_sampler: sampler;
@group(0) @binding(7) var normal_gbuffer: texture_2d<f32>;
@group(0) @binding(8) var material_gbuffer: texture_2d<f32>;  // R=reflectivity
@group(0) @binding(9) var ssao_texture: texture_2d<f32>;  // SSAO for reflected surfaces
@group(0) @binding(10) var bloom_texture: texture_2d<f32>;  // Bloom for reflected surfaces
@group(0) @binding(11) var skybox_texture: texture_2d<f32>;  // Skybox for missed rays
@group(0) @binding(12) var skybox_sampler: sampler;  // Skybox sampler with Repeat on U

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

// Reconstruct world position from depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert UV and depth to NDC (depth is [0,1], convert to NDC z)
    let z_ndc = depth * 2.0 - 1.0;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, z_ndc, 1.0);
    
    // Unproject to view space
    let view_pos = camera.inverse_proj * ndc;
    let view_pos_3d = view_pos.xyz / view_pos.w;
    
    // Transform to world space
    let world_pos = camera.inverse_view * vec4<f32>(view_pos_3d, 1.0);
    return world_pos.xyz;
}

// Project world position to screen space
// Returns vec4: xyz = screen pos (xy in UV, z in NDC depth), w = 1 if valid (in front of camera), 0 if behind
fn world_to_screen(world_pos: vec3<f32>) -> vec4<f32> {
    let clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    // Check if behind camera (w <= 0 means behind near plane)
    if (clip_pos.w <= 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);  // Invalid
    }
    
    let ndc = clip_pos.xyz / clip_pos.w;
    // Convert NDC to UV space (flip Y for texture coords)
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    
    // Use linear depth (clip.w) normalized, for more uniform stepping
    return vec4<f32>(uv, ndc.z, 1.0);  // Valid
}

fn decode_world_normal(encoded: vec3<f32>) -> vec3<f32> {
    // Stored as (n * 0.5 + 0.5) in voxel.wgsl
    let n = encoded * 2.0 - 1.0;
    // Guard against zero/denormals
    return normalize(select(n, vec3<f32>(0.0, 1.0, 0.0), dot(n, n) < 1e-8));
}

// Calculate max mip level for HZB
fn get_max_mip_level() -> f32 {
    let dims = vec2<f32>(textureDimensions(hzb_texture, 0));
    let max_dim = max(dims.x, dims.y);
    return floor(log2(max_dim));
}

fn luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// HZB-Accelerated Ray Marching (Screen Space)
// Returns vec4: xy = hit UV, z = hit confidence (0 or 1), w = world distance traveled
fn hzb_ray_march(start_pos: vec3<f32>, ray_dir: vec3<f32>, start_uv: vec2<f32>, start_depth: f32) -> vec4<f32> {
    var hit_result = vec4<f32>(-1.0, -1.0, -1.0, 0.0);
    
    // Maximum world distance for SSR
    let max_dist = 200.0;
    let thickness_base = 0.01;  // Slightly increased base thickness to fill holes
    
    // Project start and end to screen space
    let start_screen_raw = world_to_screen(start_pos);
    if (start_screen_raw.w < 0.5) {
        return hit_result;  // Start is behind camera
    }
    
    let end_pos = start_pos + ray_dir * max_dist;
    let end_screen_raw = world_to_screen(end_pos);
    
    let start_screen = start_screen_raw.xyz;
    var end_screen = end_screen_raw.xyz;
    
    // If end is behind camera, we need to find where ray crosses near plane
    if (end_screen_raw.w < 0.5) {
        // Binary search to find the point where ray crosses behind camera
        var t_min = 0.0;
        var t_max = 1.0;
        for (var i = 0; i < 8; i++) {
            let t_mid = (t_min + t_max) * 0.5;
            let mid_pos = start_pos + ray_dir * max_dist * t_mid;
            let mid_screen = world_to_screen(mid_pos);
            if (mid_screen.w > 0.5) {
                t_min = t_mid;
            } else {
                t_max = t_mid;
            }
        }
        // Use the last valid point
        let valid_pos = start_pos + ray_dir * max_dist * t_min;
        let valid_screen = world_to_screen(valid_pos);
        end_screen = valid_screen.xyz;
    }
    
    // Check if screen-space movement is too small (ray going toward camera)
    let dim = vec2<f32>(textureDimensions(scene_color));
    let screen_dist = length((end_screen.xy - start_screen.xy) * dim);
    if (screen_dist < 2.0) {
        // Ray has minimal screen-space movement, skip geometry trace but allow skybox
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Compute screen-space delta - divide total distance by max_steps for uniform stepping
    let delta = (end_screen - start_screen) / f32(params.max_steps);
    
    // Jitter start position to reduce banding artifacts
    let jitter_seed = start_uv * dim;
    let jitter = fract(sin(dot(jitter_seed, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    var current_screen = start_screen + delta * jitter;
    
    let max_mip = get_max_mip_level();
    // Cap the effective maximum mip used by SSR to avoid relying on highly-coarsened mips
    // that can contain noisy/unstable reductions. Mips above ~4 are too coarse for accurate
    // binary refinement and tend to blink as HZB generation races or produces holes.
    let effective_max_mip = min(max_mip, 3.0);
    var current_mip = min(1.0, effective_max_mip);  // Start at a fine mip to avoid skipping thin geometry
    
    // Calculate adaptive thickness based on screen-space step size and distance
    let screen_delta_length = length(delta.xy * dim);
    
    // Minimum distance before accepting hits (in terms of steps)
    let min_steps = 3u;
    
    // Calculate UV bounds with overscan
    let os = params.overscan;
    let uv_min = -os / (1.0 + os);
    let uv_max = 1.0 + os / (1.0 + os);
    
    // Track if ray is going toward or away from camera (in depth)
    let ray_goes_further = delta.z > 0.0;
    
    // Hierarchical ray march
    for (var i = 0u; i < params.max_steps; i++) {
        current_screen += delta;
        
        let uv = current_screen.xy;
        
        // Check if out of screen bounds (with overscan)
        if (uv.x < uv_min || uv.x > uv_max || uv.y < uv_min || uv.y > uv_max) {
            break;
        }
        
        // Skip first few steps to avoid self-intersection
        if (i < min_steps) {
            continue;
        }
        
        // For rays going further from camera: reject hits at depth less than start (behind us)
        let ray_depth = current_screen.z;
        if (ray_goes_further && ray_depth < start_depth - 0.001) {
            continue;
        }
        
        // Sample HZB at current mip level
        let hzb_depth = textureSampleLevel(hzb_texture, hzb_sampler, uv, current_mip).r;
        
        // Skip if we're hitting sky (depth >= 0.9999) - let the skybox path handle it
        if (hzb_depth >= 0.9999) {
            continue;
        }
        
        // Adaptive thickness: increase for distance and screen-space step size
        // Using ray_depth as a proxy for distance (0 to 1)
        let adaptive_thickness = (thickness_base + ray_depth * 0.02) * max(1.0, screen_delta_length * 0.05);
        
        // Check if ray is behind surface (potential intersection)
        if (ray_depth > hzb_depth && ray_depth - hzb_depth < adaptive_thickness) {
            if (current_mip <= 0.5) {
                // At finest detail, perform binary refinement
                var refined_screen = current_screen - delta;
                var refined_delta = delta;
                
                for (var j = 0u; j < params.max_binary_steps; j++) {
                    refined_delta *= 0.5;
                    refined_screen += refined_delta;
                    
                    let refined_uv = refined_screen.xy;
                    let refined_depth_sample = textureSampleLevel(hzb_texture, hzb_sampler, refined_uv, 0.0).r;
                    
                    if (refined_screen.z > refined_depth_sample) {
                        refined_screen -= refined_delta;
                    }
                }
                
                hit_result = vec4<f32>(refined_screen.xy, 1.0, 0.0); // Distance will be filled from G-buffer
                break;
            } else {
                // Descend to finer mip level
                current_mip = max(0.0, current_mip - 1.0);
            }
        } else {
            // No intersection at this point
            // Ascend to coarser mip if we're far from surfaces
            let depth_diff = abs(ray_depth - hzb_depth);
            if (depth_diff > adaptive_thickness * 4.0 && current_mip < effective_max_mip) {
                current_mip = min(effective_max_mip, current_mip + 1.0);
            }
        }
    }
    
    return hit_result;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel_size = vec2<f32>(1.0) / vec2<f32>(textureDimensions(scene_color));
    let depth = textureSample(scene_depth, linear_sampler, input.uv);
    
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
    
    // Offset start position along normal to avoid self-intersection
    let ray_start = world_pos + normal * 1.5;
    
    // Check if ray passes too close to camera - this causes projection singularities
    // that appear as "black holes" in cardinal directions
    // Calculate closest distance from ray to camera position using point-to-line distance
    let to_camera = camera.camera_pos - ray_start;
    let ray_proj = dot(to_camera, reflect_dir);
    let closest_point_on_ray = ray_start + reflect_dir * max(0.0, ray_proj);
    let dist_to_camera = length(camera.camera_pos - closest_point_on_ray);
    
    // If ray passes within 2 units of camera, skip geometry tracing (use skybox only)
    var hit_result = vec4<f32>(-1.0, -1.0, -1.0, 0.0);
    if (dist_to_camera >= 2.0 || ray_proj <= 0.0) {
        // Ray doesn't pass close to camera, trace normally
        hit_result = hzb_ray_march(ray_start, reflect_dir, input.uv, depth);
    }
    // Otherwise hit_result stays as "no hit" and we fall through to skybox

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

    if (hit_result.z > 0.0) {
        let hit_uv = hit_result.xy;
        let dim = vec2<f32>(textureDimensions(scene_color));
        let texel = 1.0 / dim;
        
        // Sample G-buffer at hit point for stable distance and normal check
        let hit_gbuf = textureSample(normal_gbuffer, linear_sampler, hit_uv);
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
        let blur_factor = max(distance_blur_factor, roughness);
        let blur_spread = 1.0 + blur_factor * 3.0;
        
        // Start with sharp sample
        var reflection_color = textureSample(scene_color, linear_sampler, hit_uv).rgb;
        var bloom_accum = textureSample(bloom_texture, linear_sampler, hit_uv).rgb;
        var ao_accum = textureSample(ssao_texture, linear_sampler, hit_uv).a;
        
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
                    let s_coords = vec2<i32>(dim * offset_uv);
                    if (s_coords.x < 0 || s_coords.x >= i32(dim.x) || s_coords.y < 0 || s_coords.y >= i32(dim.y)) {
                        continue;
                    }
                    let s_depth = textureLoad(scene_depth, s_coords, 0);
                    if (s_depth >= 0.9999) { continue; }

                    let sample_color = textureSample(scene_color, linear_sampler, offset_uv).rgb;
                    let sample_bloom = textureSample(bloom_texture, linear_sampler, offset_uv).rgb;
                    let sample_ssao = textureSample(ssao_texture, linear_sampler, offset_uv);
                    
                    // Improved weighting: suppress HDR fireflies but don't emphasize black holes
                    let lum = luminance(sample_color + sample_bloom * params.bloom_strength);
                    // Only suppress if lum > 1.0 (HDR)
                    let w = 1.0 / (1.0 + max(0.0, lum - 1.0));
                    
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
        let reflection_with_ao = bloomed_reflection * ao_accum;
        
        // Fade based on distance from screen edges
        let edge_fade = min(
            min(hit_result.x, 1.0 - hit_result.x),
            min(hit_result.y, 1.0 - hit_result.y)
        );
        let edge_factor = smoothstep(0.0, 0.1, edge_fade);
        
        // Fade reflections at very long distances (less accurate)
        let distance_fade = 1.0 - smoothstep(50.0, 100.0, reflection_distance);
        
        // Calculate confidence in the screen-space reflection
        // If confidence is low (edge, distance, angle, facing), we mix in the skybox
        let confidence = edge_factor * distance_fade * (1.0 - angle_fade) * facing_factor;
        
        // Mix geometry reflection with skybox based on confidence
        let final_color = mix(sky_color, reflection_with_ao, confidence);
        
        return vec4<f32>(final_color, reflectivity);
    }
    
    // No geometry hit - return skybox
    return vec4<f32>(sky_color, reflectivity);
}