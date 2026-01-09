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
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    water_level: f32,
    water_visibility: f32,
    water_color: vec4<f32>,
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

fn decode_world_normal(encoded: vec3<f32>) -> vec3<f32> {
    let n = encoded * 2.0 - 1.0;
    return normalize(select(n, vec3<f32>(0.0, 1.0, 0.0), dot(n, n) < 1e-8));
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

fn is_water_visible_at_chunk(chunk_i: vec3<i32>, world_grid_origin: vec3<f32>) -> bool {
    let occ = textureLoad(gi_probe_color, chunk_i, 0).a;
    if (occ <= 0.05) {
        return true;
    }
    let packed = textureLoad(gi_probe_bbox, chunk_i, 0).r;
    let bbox_valid = (packed >> 24u) & 1u;
    if (bbox_valid == 1u) {
        let ymax = f32((packed >> 16u) & 0xFu);
        let chunk_origin_y = f32(chunk_i.y) * 16.0 + world_grid_origin.y;
        return (chunk_origin_y + ymax + 1.0) < camera.water_level;
    }
    return false;
}

fn sample_gi_grid(world_pos: vec3<f32>, reflect_dir: vec3<f32>, sky_color: vec3<f32>) -> vec4<f32> {
    let dims = vec3<f32>(camera.gi_grid_dims);
    let world_grid_origin = vec3<f32>(camera.gi_grid_origin) * 16.0;
    let sun_dir = normalize(camera.sun_direction);
    let brightness = camera.skybox_brightness;

    // Water plane intersection
    var t_water = -1.0;
    if (abs(reflect_dir.y) > 0.001) {
        t_water = (camera.water_level - world_pos.y) / reflect_dir.y;
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

        // Water intersection check
        if (t_water > 0.0 && (t_current + bias) >= t_water) {
            let water_hit = world_pos + reflect_dir * t_water;
            let water_chunk_f = (water_hit - world_grid_origin) / 16.0;
            let water_chunk_i = vec3<i32>(floor(water_chunk_f));

            if (all(water_chunk_i >= vec3<i32>(0)) && all(water_chunk_i < camera.gi_grid_dims)) {
                if (is_water_visible_at_chunk(water_chunk_i, world_grid_origin)) {
                    accumulated_color += water_color * remaining_alpha;
                    remaining_alpha = 0.0;
                    break;
                }
            }
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
                if (aabb_max.y >= camera.water_level) {
                    let hit = ray_aabb_intersection(world_pos, reflect_dir, aabb_min, aabb_max);
                    
                    if (hit.t_near <= hit.t_far && hit.t_far > 0.0 && hit.t_near > 0.1) {
                        let hit_point_raw = world_pos + reflect_dir * hit.t_near;
                        var hit_point = hit_point_raw;
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
                                if (d_buf < 0.9999) {
                                    let mesh_pos = reconstruct_world_pos(screen_uv, d_buf);
                                    // If the mesh is reasonably close to our voxel hit, assume we hit the mesh
                                    // 3.0 is a heuristic tolerance (voxel size is 16, but bbox is smaller)
                                    if (distance(mesh_pos, hit_point) < 3.0) {
                                        // Precise mesh hit: Sample screen color directly
                                        // This gives us full detail (texture, lighting, shadows) instead of coarse GI
                                        let screen_color = textureSampleLevel(scene_color, linear_sampler, screen_uv, 0.0).rgb;
                                        
                                        // Check for water below mesh (e.g. pier piling)
                                        if (mesh_pos.y < camera.water_level) {
                                            // Submerged part might be tinted or hidden, but for now just use screen entry
                                            // Or maybe we shouldn't have hit it if it's underwater?
                                            // The screen color includes water surface if water was drawn? 
                                            // No, water pass is AFTER ssr. So 'scene_color' is opaque geometry only.
                                            // If mesh_pos.y < water, it's underwater geometry.
                                            // We should tint it blue?
                                            // Let's keep it simple: just reflect what we see.
                                        }

                                        accumulated_color += screen_color * remaining_alpha;
                                        remaining_alpha = 0.0;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // Hit below water
                        if (hit_point.y < camera.water_level) {
                            accumulated_color += water_color * remaining_alpha;
                            remaining_alpha = 0.0;
                            break;
                        }

                        // Trilinear color lookup
                        let hit_uvw = (hit_point - world_grid_origin) / (dims * 16.0);
                        let hit_data = textureSampleLevel(gi_probe_color, linear_sampler, hit_uvw, 0.0);
                        let rad = sample_radiance(hit_uvw, reflect_dir);

                        let dot_sun = saturate(dot(hit_normal, sun_dir));
                        let sun_lit = camera.sun_color * camera.sun_intensity * dot_sun * 1.5;
                        let base_color = hit_data.rgb * (rad * 2.5 + sun_lit);
                        
                        // Secondary reflection (simplified Fresnel, reuse sky_color)
                        let surface_brightness = dot(hit_data.rgb, vec3<f32>(0.299, 0.587, 0.114));
                        let view_to_cam = normalize(camera.camera_pos - hit_point);
                        let cos_theta = saturate(dot(hit_normal, view_to_cam));
                        let f0 = 0.04 + surface_brightness * 0.5;
                        let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
                        
                        let lit_color = mix(base_color, sky_color, fresnel * 0.7);
                        
                        accumulated_color += lit_color * remaining_alpha;
                        remaining_alpha = 0.0;
                        break;
                    }
                }
            } else {
                // Fallback: semi-transparent volume
                let chunk_min_y = f32(current_i.y) * 16.0 + world_grid_origin.y;
                if (chunk_min_y + 16.0 >= camera.water_level) {
                    let entry_uvw = (start_pos + safe_dir * t_current - world_grid_origin) / (dims * 16.0);
                    let hit_data = textureSampleLevel(gi_probe_color, linear_sampler, entry_uvw, 0.0);
                    let rad = sample_radiance(entry_uvw, reflect_dir);
                    let sun_lit = camera.sun_color * camera.sun_intensity * 0.5;
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
            if (is_water_visible_at_chunk(water_chunk_i, world_grid_origin)) {
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
    if (depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let material = textureSample(material_gbuffer, linear_sampler, input.uv);
    let reflectivity = material.r;
    if (reflectivity < 0.01) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let world_pos = reconstruct_world_pos(input.uv, depth);

    // Skip submerged surfaces
    if (camera.water_visibility > 0.0 && world_pos.y < camera.water_level) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let gbuf = textureSample(normal_gbuffer, linear_sampler, input.uv);
    let normal = decode_world_normal(gbuf.rgb);

    let view_dir = normalize(world_pos - camera.camera_pos);
    let reflect_dir = reflect(view_dir, normal);

    // Sample sky once
    let sky_color = sample_sky_equirect(reflect_dir);
    
    // GI grid trace
    let gi_res = sample_gi_grid(world_pos, reflect_dir, sky_color);
    
    let out_color = mix(sky_color, gi_res.rgb, gi_res.a);

    return vec4<f32>(out_color, reflectivity);
}

