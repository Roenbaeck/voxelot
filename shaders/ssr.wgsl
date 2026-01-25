// Volumetric Grid-Based Specular Reflections
// Optimized: reduced samples, single skybox sample, capped iterations

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
    gi_grid_origin: vec3<i32>,
    _pad_gi1: i32,
    gi_grid_dims: vec3<i32>,
    _pad_gi2: i32,
    sun_direction_intensity: vec4<f32>,
    sun_color_water_level: vec4<f32>,
    water_vis_fog_density: vec4<f32>,
    water_color: vec4<f32>,
    ambient_color: vec4<f32>,
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
    gi_scale: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
    _pad6: f32,
    _pad7: f32,
    _pad8: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: SSRParams;
@group(0) @binding(2) var scene_depth: texture_depth_2d;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var hzb_texture: texture_2d<f32>;
@group(0) @binding(5) var hzb_sampler: sampler;
@group(0) @binding(6) var normal_gbuffer: texture_2d<f32>;
@group(0) @binding(7) var material_gbuffer: texture_2d<f32>;
@group(0) @binding(8) var skybox_texture: texture_2d<f32>;
@group(0) @binding(9) var skybox_sampler: sampler;
@group(0) @binding(10) var gi_probe_px: texture_3d<f32>;
@group(0) @binding(11) var gi_probe_nx: texture_3d<f32>;
@group(0) @binding(12) var gi_probe_py: texture_3d<f32>;
@group(0) @binding(13) var gi_probe_ny: texture_3d<f32>;
@group(0) @binding(14) var gi_probe_pz: texture_3d<f32>;
@group(0) @binding(15) var gi_probe_nz: texture_3d<f32>;
@group(0) @binding(16) var gi_probe_color: texture_3d<f32>;
@group(0) @binding(17) var gi_probe_bbox: texture_3d<u32>;
@group(0) @binding(18) var scene_color: texture_2d<f32>;


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);
    return output;
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

fn load_depth_at_uv(uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(scene_depth);
    let px = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0), vec2<i32>(dim) - 1);
    return textureLoad(scene_depth, px, 0);
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, depth * 2.0 - 1.0, 1.0);
    let view_pos = camera.inverse_proj * ndc;
    let world_pos = camera.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
    return world_pos.xyz;
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
    let u = 0.5 + atan2(rotated_dir.z, rotated_dir.x) * 0.15915494;
    let v = 0.5 - asin(clamp(rotated_dir.y, -1.0, 1.0)) * 0.31830989;
    let sky_sample = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v)).rgb;

    let brightness = camera.skybox_brightness;
    let sat = clamp(camera.skybox_saturation + (1.0 - camera.skybox_saturation) * brightness, 0.0, 1.0);
    let lum = dot(sky_sample, vec3<f32>(0.299, 0.587, 0.114));
    let desaturated = mix(vec3<f32>(lum), sky_sample, sat);
    let effect_strength = (1.0 - brightness) * camera.skybox_tint_strength;
    let tinted = mix(desaturated, desaturated * camera.skybox_tint, effect_strength);
    return tinted * brightness;
}

fn trace_local_ssr(start_pos: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    let step_size = 0.8;
    let max_steps = 180; // ~144 units
    var current_pos = start_pos + dir * 0.2; 

    let dim = textureDimensions(scene_depth);
    let fdim = vec2<f32>(dim);

    for (var i = 0; i < max_steps; i++) {
        current_pos += dir * step_size;

        let clip_pos = camera.view_proj * vec4<f32>(current_pos, 1.0);
        if (clip_pos.w <= 0.0) { continue; }
        
        let ndc = clip_pos.xyz / clip_pos.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            continue;
        }

        let px = vec2<i32>(uv * fdim);
        let d_raw = textureLoad(scene_depth, px, 0);

        if (d_raw >= 0.999999 || d_raw <= 0.0) { continue; } 

        let sample_pos = reconstruct_world_pos(uv, d_raw);
        let dist_sample = distance(camera.camera_pos, sample_pos);
        let dist_ray = distance(camera.camera_pos, current_pos);

        // Ray is "behind" surface from camera view
        if (dist_ray > dist_sample + 0.05) {
            let thickness = 0.5 + f32(i) * 0.015;
            if (dist_ray - dist_sample < thickness) {
                // Binary refinement for better precision
                var refine_pos = current_pos;
                var prev_pos = current_pos - dir * step_size;
                for (var j = 0; j < 3; j++) {
                    let mid = mix(prev_pos, refine_pos, 0.5);
                    let mid_clip = camera.view_proj * vec4<f32>(mid, 1.0);
                    let mid_ndc = mid_clip.xyz / mid_clip.w;
                    let mid_uv = vec2<f32>(mid_ndc.x * 0.5 + 0.5, 0.5 - mid_ndc.y * 0.5);
                    let mid_d = textureLoad(scene_depth, vec2<i32>(mid_uv * fdim), 0);
                    if (distance(camera.camera_pos, mid) > distance(camera.camera_pos, reconstruct_world_pos(mid_uv, mid_d)) + 0.05) {
                        refine_pos = mid;
                    } else {
                        prev_pos = mid;
                    }
                }

                let final_clip = camera.view_proj * vec4<f32>(refine_pos, 1.0);
                let final_ndc = final_clip.xyz / final_clip.w;
                let final_uv = vec2<f32>(final_ndc.x * 0.5 + 0.5, 0.5 - final_ndc.y * 0.5);
                let final_px = vec2<i32>(final_uv * fdim);

                let color = textureLoad(scene_color, final_px, 0).rgb;
                // Tight edge fade
                let edge_fade = clamp(10.0 * min(min(final_uv.x, 1.0 - final_uv.x), min(final_uv.y, 1.0 - final_uv.y)), 0.0, 1.0);
                return vec4<f32>(color, edge_fade);
            }
        }
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// Optimized radiance: 3 samples with branchless selection (Integer/Discrete version)
fn sample_radiance_int(pos: vec3<i32>, dir: vec3<f32>) -> vec3<f32> {
    let w = dir * dir;
    let color_x = select(textureLoad(gi_probe_nx, pos, 0).rgb,
                         textureLoad(gi_probe_px, pos, 0).rgb, dir.x > 0.0);
    let color_y = select(textureLoad(gi_probe_ny, pos, 0).rgb,
                         textureLoad(gi_probe_py, pos, 0).rgb, dir.y > 0.0);
    let color_z = select(textureLoad(gi_probe_nz, pos, 0).rgb,
                         textureLoad(gi_probe_pz, pos, 0).rgb, dir.z > 0.0);
    return (color_x * w.x + color_y * w.y + color_z * w.z) * params.gi_scale;
}

// Optimized radiance: 3 samples with branchless selection
fn sample_radiance(uvw: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let w = dir * dir;
    let color_x = select(textureSampleLevel(gi_probe_nx, linear_sampler, uvw, 0.0).rgb,
                         textureSampleLevel(gi_probe_px, linear_sampler, uvw, 0.0).rgb, dir.x > 0.0);
    let color_y = select(textureSampleLevel(gi_probe_ny, linear_sampler, uvw, 0.0).rgb,
                         textureSampleLevel(gi_probe_py, linear_sampler, uvw, 0.0).rgb, dir.y > 0.0);
    let color_z = select(textureSampleLevel(gi_probe_nz, linear_sampler, uvw, 0.0).rgb,
                         textureSampleLevel(gi_probe_pz, linear_sampler, uvw, 0.0).rgb, dir.z > 0.0);
    return (color_x * w.x + color_y * w.y + color_z * w.z) * params.gi_scale;
}

struct RayHit {
    t_near: f32,
    t_far: f32,
    normal: vec3<f32>,
}

fn ray_aabb_intersection(ro: vec3<f32>, rd: vec3<f32>, min_p: vec3<f32>, max_p: vec3<f32>) -> RayHit {
    let inv_dir = 1.0 / rd;
    let t_bot = inv_dir * (min_p - ro);
    let t_top = inv_dir * (max_p - ro);
    let t_min = min(t_bot, t_top);
    let t_max = max(t_bot, t_top);
    let t_near = max(max(t_min.x, t_min.y), t_min.z);
    let t_far = min(min(t_max.x, t_max.y), t_max.z);
    
    let mask = vec3<f32>(
        f32(t_near == t_min.x),
        f32(t_near == t_min.y),
        f32(t_near == t_min.z)
    );
    let normal = -sign(rd) * mask;
    
    return RayHit(t_near, t_far, normal);
}

fn is_water_visible_at_chunk(chunk_i: vec3<i32>, world_grid_origin: vec3<f32>, water_pos: vec3<f32>) -> bool {
    let occ = textureLoad(gi_probe_color, chunk_i, 0).a;
    if (occ <= 0.05) {
        return true;
    }
    let packed = textureLoad(gi_probe_bbox, chunk_i, 0).r;
    let bbox_valid = (packed >> 24u) & 1u;
    if (bbox_valid == 1u) {
        let xmin = f32(packed & 0xFu);
        let ymin = f32((packed >> 4u) & 0xFu);
        let zmin = f32((packed >> 8u) & 0xFu);
        let xmax = f32((packed >> 12u) & 0xFu);
        let ymax = f32((packed >> 16u) & 0xFu);
        let zmax = f32((packed >> 20u) & 0xFu);
        
        let chunk_origin_y = f32(chunk_i.y) * 16.0 + world_grid_origin.y;
        let chunk_origin_x = f32(chunk_i.x) * 16.0 + world_grid_origin.x;
        let chunk_origin_z = f32(chunk_i.z) * 16.0 + world_grid_origin.z;

        // If the chunk is completely submerged, water is visible above it
        if (chunk_origin_y + ymax + 1.0) < camera.sun_color_water_level.w {
            return true;
        }

        // Check if the water intersection point is within the solid column of this chunk
        let lx = water_pos.x - chunk_origin_x;
        let lz = water_pos.z - chunk_origin_z;
        
        // Expand bounds slightly for robustness
        if (lx >= xmin - 0.1 && lx <= xmax + 1.1 && lz >= zmin - 0.1 && lz <= zmax + 1.1) {
             // We are inside the horizontal bounds of the solid part.
             // If the solid part goes below water, then it blocks water.
             if (chunk_origin_y + ymin <= camera.sun_color_water_level.w) {
                 return false;
             }
        }
        return true;
    }
    return false;
}

fn sample_gi_grid(world_pos: vec3<f32>, reflect_dir: vec3<f32>, sky_color: vec3<f32>) -> vec4<f32> {
    let dims = vec3<f32>(camera.gi_grid_dims);
    let world_grid_origin = vec3<f32>(camera.gi_grid_origin) * 16.0;
    let sun_dir = normalize(camera.sun_direction_intensity.xyz);
    let brightness = camera.skybox_brightness;

    // Water plane intersection
    var t_water = -1.0;
    if (abs(reflect_dir.y) > 0.001) {
        t_water = (camera.sun_color_water_level.w - world_pos.y) / reflect_dir.y;
    }

    // Water color - blend skybox reflection for visual structure
    let shallow_color = vec3<f32>(0.15, 0.45, 0.50) * brightness;
    let deep_color = vec3<f32>(0.02, 0.12, 0.20) * brightness;
    let base_water = mix(shallow_color, deep_color, 0.5);
    // Reflect the incoming ray off the water surface (approximate normal = up)
    let water_reflect_dir = reflect(reflect_dir, vec3<f32>(0.0, 1.0, 0.0));
    let water_sky = sample_sky_equirect(water_reflect_dir);
    // Fresnel-ish blend: more sky reflection at grazing angles
    let water_fresnel = pow(1.0 - abs(reflect_dir.y), 2.0);
    let water_color = mix(base_water, water_sky * 0.8, 0.4 + water_fresnel * 0.5) * 0.85;

    // DDA Setup
    let bias = 0.05;
    let start_pos = world_pos + reflect_dir * bias;
    let safe_dir = reflect_dir + select(vec3<f32>(0.0), vec3<f32>(1e-8), abs(reflect_dir) < vec3<f32>(1e-8));
    let inv_dir = 1.0 / safe_dir;
    let step_vec = vec3<i32>(sign(safe_dir));
    
    let start_chunk_f = (start_pos - world_grid_origin) / 16.0;
    var current_i = vec3<i32>(floor(start_chunk_f));
    
    // Identify the chunk containing the ray origin to prevent self-intersection,
    // while allowing immediate hits in neighboring chunks.
    let origin_chunk_f = (world_pos - world_grid_origin) / 16.0;
    let origin_chunk_i = vec3<i32>(floor(origin_chunk_f));
    
    let t_delta = abs(inv_dir) * 16.0;
    var t_max = (vec3<f32>(current_i) + max(vec3<f32>(step_vec), vec3<f32>(0.0)) - start_chunk_f) * 16.0 * inv_dir;

    var accumulated_color = vec3<f32>(0.0);
    var remaining_alpha = 1.0;
    var t_current = 0.0;

    // Cap iterations for performance
    let max_iter = min(params.max_steps, 20u);

    for (var i = 0u; i < max_iter; i++) {
        // Bounds check
        if (any(current_i < vec3<i32>(0)) || any(current_i >= camera.gi_grid_dims)) {
            break;
        }

        var water_opportunity = false;

        // Water intersection check
        if (t_water > 0.0 && (t_current + bias) >= t_water) {
            let water_hit = world_pos + reflect_dir * t_water;
            let water_chunk_f = (water_hit - world_grid_origin) / 16.0;
            let water_chunk_i = vec3<i32>(floor(water_chunk_f));

            if (all(water_chunk_i >= vec3<i32>(0)) && all(water_chunk_i < camera.gi_grid_dims)) {
                let occ = textureLoad(gi_probe_color, water_chunk_i, 0).a;
                if (occ <= 0.05) {
                    accumulated_color += water_color * remaining_alpha;
                    remaining_alpha = 0.0;
                    break;
                }
                water_opportunity = true;
            } else {
                 // Water outside grid? Render it.
                 accumulated_color += water_color * remaining_alpha;
                 remaining_alpha = 0.0;
                 break;
            }
            // Consume t_water but remember the opportunity to draw it
            t_water = -1.0;
        }

        // Quick occupancy check
        let chunk_data = textureLoad(gi_probe_color, current_i, 0);
        
        if (chunk_data.a > 0.05) {
            let packed_bbox = textureLoad(gi_probe_bbox, current_i, 0).r;
            let valid = (packed_bbox >> 24u) & 1u;
            
            if (valid == 1u) {
                let xmin = f32(packed_bbox & 0xFu);
                let ymin = f32((packed_bbox >> 4u) & 0xFu);
                let zmin = f32((packed_bbox >> 8u) & 0xFu);
                let xmax = f32((packed_bbox >> 12u) & 0xFu);
                let ymax = f32((packed_bbox >> 16u) & 0xFu);
                let zmax = f32((packed_bbox >> 20u) & 0xFu);
                
                let chunk_origin = vec3<f32>(current_i) * 16.0 + world_grid_origin;
                let aabb_min = chunk_origin + vec3<f32>(xmin, ymin, zmin);
                let aabb_max = chunk_origin + vec3<f32>(xmax + 1.0, ymax + 1.0, zmax + 1.0);
                
                // Skip submerged chunks
                if (aabb_max.y >= camera.sun_color_water_level.w) {
                    let hit = ray_aabb_intersection(world_pos, reflect_dir, aabb_min, aabb_max);
                    
                    // Only enforce the large bias (0.1) if we are in the same chunk as the origin.
                    // For neighbor chunks, even if we are very close (e.g. at the boundary),
                    // we should accept the hit to avoid gaps (the "H shape" artifact).
                    let is_self = all(current_i == origin_chunk_i);
                    let min_dist = select(0.001, 0.1, is_self);

                    if (hit.t_near <= hit.t_far && hit.t_far > 0.0 && hit.t_near > min_dist) {
                        let hit_point_raw = world_pos + reflect_dir * hit.t_near;
                        var hit_point = hit_point_raw;
                        var mesh_color = vec3<f32>(0.0);
                        var mesh_blend = 0.0;
                        var hit_normal = hit.normal;

                        // Hybrid Mesh Snapping: Check if we hit visible screen geometry
                        // This uses the depth buffer to resolve the exact mesh surface instead of the bounding box
                        let clip_pos = camera.view_proj * vec4<f32>(hit_point, 1.0);
                        if (clip_pos.w > 0.0) {
                            let ndc = clip_pos.xyz / clip_pos.w;
                            if (abs(ndc.x) < 1.0 && abs(ndc.y) < 1.0) {
                                let screen_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                                let dim = textureDimensions(scene_depth);
                                let px = vec2<i32>(screen_uv * vec2<f32>(dim));
                                let d_buf = textureLoad(scene_depth, px, 0);
                                // Skip background or invalid depth
                                if (d_buf < 0.999999) {
                                    let mesh_pos = reconstruct_world_pos(screen_uv, d_buf);
                                    let dist = distance(mesh_pos, hit_point);
                                    // Require closer match and roughly similar surface normal to avoid snapping to unrelated geometry
                                    if (dist < 4.0) {
                                        let mesh_normal = oct_decode(textureSample(normal_gbuffer, linear_sampler, screen_uv).xy);
                                        let normal_align = dot(mesh_normal, hit_normal);
                                        if (normal_align > 0.4) {
                                            mesh_color = textureSampleLevel(scene_color, linear_sampler, screen_uv, 0.0).rgb;
                                            let edge_x = min(screen_uv.x, 1.0 - screen_uv.x);
                                            let edge_y = min(screen_uv.y, 1.0 - screen_uv.y);
                                            let edge_factor = saturate(min(edge_x, edge_y) * 10.0);
                                            let dist_factor = 1.0 - saturate((dist - 1.0) / 3.0);
                                            mesh_blend = edge_factor * dist_factor;
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Hit below water
                        if (hit_point.y < camera.sun_color_water_level.w) {
                            accumulated_color += water_color * remaining_alpha;
                            remaining_alpha = 0.0;
                            break;
                        }

                        // Decide whether to accept a coarse AABB/probe hit, or skip it and continue DDA.
                        let dim = textureDimensions(scene_depth);
                        var screen_disproved = false;
                        var is_on_screen = false;

                        {
                            let clip_pos = camera.view_proj * vec4<f32>(hit_point, 1.0);
                            if (clip_pos.w > 0.0) {
                                let ndc = clip_pos.xyz / clip_pos.w;
                                if (abs(ndc.x) < 0.99 && abs(ndc.y) < 0.99) {
                                    is_on_screen = true;
                                    let screen_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                                    let px = vec2<i32>(screen_uv * vec2<f32>(dim));
                                    let d_raw = textureLoad(scene_depth, px, 0);
                                    
                                    // DISPROVE: If the depth buffer shows something clearly behind our hit,
                                    // then the AABB is hitting empty air at that location.
                                    let mesh_pos = reconstruct_world_pos(screen_uv, d_raw);
                                    let mesh_dist_to_cam = distance(camera.camera_pos, mesh_pos);
                                    let hit_dist_to_cam = distance(camera.camera_pos, hit_point);
                                    
                                    // Disprove if the screen depth is significantly behind the hit point
                                    if (d_raw >= 0.999999 || mesh_dist_to_cam > hit_dist_to_cam + 1.0) {
                                        screen_disproved = true;
                                    }
                                }
                            }
                        }

                        let probe_alpha = textureLoad(gi_probe_color, current_i, 0).a;
                        let hit_dot = abs(dot(reflect_dir, hit_normal));
                        
                        // Grazing rejection: kills "H-shapes" on side-faces of AABBs
                        let is_grazing = (hit_dot < 0.04); 

                        // Accept if: 
                        // 1. Strong mesh snap (always reliable)
                        // 2. Off-screen (cannot disprove, so trust probe)
                        // 3. On-screen and not disproved by depth
                        // AND it's not a grazing artifact.
                        let accept_aabb = (mesh_blend >= 0.25) || (!is_grazing && (!is_on_screen || !screen_disproved) && probe_alpha > 0.1);

                        if (accept_aabb) {
                            // Use exact chunk color to avoid bleeding dark edges from empty neighbors
                            let hit_data = textureLoad(gi_probe_color, current_i, 0);
                            let rad = sample_radiance_int(current_i, reflect_dir);

                            let dot_sun = saturate(dot(hit_normal, sun_dir));
                            let sun_lit = camera.sun_color_water_level.xyz * camera.sun_direction_intensity.w * dot_sun * 1.5;
                            let base_color = hit_data.rgb * (rad * 2.5 + sun_lit);
                            
                            // Secondary reflection (simplified Fresnel, reuse sky_color)
                            let surface_brightness = dot(hit_data.rgb, vec3<f32>(0.299, 0.587, 0.114));
                            let view_to_cam = normalize(camera.camera_pos - hit_point);
                            let cos_theta = saturate(dot(hit_normal, view_to_cam));
                            let f0 = 0.04 + surface_brightness * 0.5;
                            let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
                            
                            var lit_color = mix(base_color, sky_color, fresnel * 0.7);

                            // Attenuate local probe-only hits slightly to blend better with mesh detail
                            if (mesh_blend < 0.25) {
                                lit_color = lit_color * 0.75;
                            }

                            let final_color = mix(lit_color, mesh_color, mesh_blend);

                            accumulated_color += final_color * remaining_alpha;
                            remaining_alpha = 0.0;
                            break;
                        }
                    }
                    else if (water_opportunity) {
                        // Bizarre case: Dense chunk, solid bbox, but ray missed bbox face or started inside.
                        // Implies empty space inside bbox (or precision issue).
                        // Since we are "inside" the bounding volume, assume we are seeing water.
                        accumulated_color += water_color * remaining_alpha;
                        remaining_alpha = 0.0;
                        break;
                    }
                } else if (water_opportunity) {
                    // Chunk submerged fully or partially -> but we had an opportunity?
                    // If AABB is submerged, geometry check is skipped.
                    // Water wins.
                    accumulated_color += water_color * remaining_alpha;
                    remaining_alpha = 0.0;
                    break;
                }
            } else {
                // Fallback: semi-transparent volume
                let chunk_min_y = f32(current_i.y) * 16.0 + world_grid_origin.y;
                if (chunk_min_y + 16.0 >= camera.sun_color_water_level.w) {
                    // Use discrete probe lookup to avoid trilinear bleeding at chunk boundaries
                    let entry_uvw = (start_pos + safe_dir * t_current - world_grid_origin) / (dims * 16.0);
                    let entry_chunk_f = entry_uvw * vec3<f32>(camera.gi_grid_dims);
                    let entry_chunk_i = vec3<i32>(floor(entry_chunk_f));

                    if (all(entry_chunk_i >= vec3<i32>(0)) && all(entry_chunk_i < camera.gi_grid_dims)) {
                        let hit_data = textureLoad(gi_probe_color, entry_chunk_i, 0);
                        let rad = sample_radiance_int(entry_chunk_i, reflect_dir);
                        let sun_lit = camera.sun_color_water_level.xyz * camera.sun_direction_intensity.w * 0.5;
                        let lit_color = hit_data.rgb * (rad * 2.5 + sun_lit);

                        let alpha = saturate(hit_data.a * 2.0) * remaining_alpha;
                        accumulated_color += lit_color * alpha;
                        remaining_alpha -= alpha;

                        if (remaining_alpha < 0.1) {
                            remaining_alpha = 0.0;
                            break;
                        }
                    }
                }
            }
        } else if (water_opportunity) {
             // Chunk was empty (or occupancy < 0.05), but we entered here? 
             // Note: occupancy check `if (chunk_data.a > 0.05)` controls this block.
             // If we are here in `else`, chunk is empty.
             // If trunk is empty, water is not blocked.
             accumulated_color += water_color * remaining_alpha;
             remaining_alpha = 0.0;
             break;
        }

        // DDA advance
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                t_current = t_max.x;
                t_max.x += t_delta.x;
                current_i.x += step_vec.x;
            } else {
                t_current = t_max.z;
                t_max.z += t_delta.z;
                current_i.z += step_vec.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                t_current = t_max.y;
                t_max.y += t_delta.y;
                current_i.y += step_vec.y;
            } else {
                t_current = t_max.z;
                t_max.z += t_delta.z;
                current_i.z += step_vec.z;
            }
        }
    }

    // Final water check
    if (remaining_alpha > 0.0 && t_water > 0.0) {
        let water_hit = world_pos + reflect_dir * t_water;
        let water_chunk_f = (water_hit - world_grid_origin) / 16.0;
        let water_chunk_i = vec3<i32>(floor(water_chunk_f));
        if (all(water_chunk_i >= vec3<i32>(0)) && all(water_chunk_i < camera.gi_grid_dims)) {
            if (is_water_visible_at_chunk(water_chunk_i, world_grid_origin, water_hit)) {
                accumulated_color += water_color * remaining_alpha;
                remaining_alpha = 0.0;
            }
        }
    }
    
    return vec4<f32>(accumulated_color, 1.0 - remaining_alpha);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let depth = load_depth_at_uv(input.uv);
    if (depth >= 0.999999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let material = textureSample(material_gbuffer, linear_sampler, input.uv);
    let reflectivity = material.r;
    if (reflectivity < 0.01) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let world_pos = reconstruct_world_pos(input.uv, depth);

    // Distance-based reflection fade: keeps far reflections from looking overly punchy/noisy.
    // This scales the returned alpha (reflectivity), so downstream passes (DoF/composite)
    // automatically blend less SSR with distance.
    let dist_to_cam = distance(camera.camera_pos, world_pos);
    let dist_fade = 1.0 - smoothstep(400.0, 2000.0, dist_to_cam);
    let reflectivity_faded = reflectivity * clamp(dist_fade, 0.0, 1.0);
    if (reflectivity_faded < 0.005) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Skip submerged surfaces
    if (camera.water_vis_fog_density.x > 0.0 && world_pos.y < camera.sun_color_water_level.w) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let gbuf = textureSample(normal_gbuffer, linear_sampler, input.uv).xy;
    let normal = oct_decode(gbuf);

    let view_dir = normalize(world_pos - camera.camera_pos);
    let reflect_dir = reflect(view_dir, normal);

    // Sample sky once
    let sky_color = sample_sky_equirect(reflect_dir);
    
    // Hybrid: Local Screen-Space Raymarch + Distant GI Grid Trace
    let ssr_res = trace_local_ssr(world_pos, reflect_dir);
    
    // Hybrid Strategy: 
    // - Use High-Res SSR for everything on-screen up to ~140 units.
    // - Start GI Trace further out to avoid "Phantom AABB" artifacts from local geometry.
    // - If SSR misses, GI starts closer to pick up low-frequency details.
    let gi_start_offset = mix(2.0, 64.0, ssr_res.a);
    let gi_res_raw = sample_gi_grid(world_pos + reflect_dir * gi_start_offset, reflect_dir, sky_color);
    
    // If we have a clear SSR hit (no fade), ignore GI entirely for this ray.
    // Otherwise, blend GI into the gaps (edges/range limit).
    let gi_res = vec4<f32>(
        mix(gi_res_raw.rgb, ssr_res.rgb, ssr_res.a),
        max(ssr_res.a, gi_res_raw.a)
    );
    
    let out_color = mix(sky_color, gi_res.rgb, gi_res.a);

    return vec4<f32>(out_color, reflectivity_faded);
}

