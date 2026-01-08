// Volumetric Grid-Based Specular Reflections
// Replaces anchor-based cubemaps with a stable 3D GI Radiance Grid

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
    let px = clamp(i32(uv.x * f32(dim.x)), 0, i32(dim.x) - 1);
    let py = clamp(i32(uv.y * f32(dim.y)), 0, i32(dim.y) - 1);
    return vec2<i32>(px, py);
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

fn sample_radiance(uvw: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let w = dir * dir;
    let color_x = select(textureSampleLevel(gi_probe_nx, linear_sampler, uvw, 0.0).rgb, textureSampleLevel(gi_probe_px, linear_sampler, uvw, 0.0).rgb, dir.x > 0.0);
    let color_y = select(textureSampleLevel(gi_probe_ny, linear_sampler, uvw, 0.0).rgb, textureSampleLevel(gi_probe_py, linear_sampler, uvw, 0.0).rgb, dir.y > 0.0);
    let color_z = select(textureSampleLevel(gi_probe_nz, linear_sampler, uvw, 0.0).rgb, textureSampleLevel(gi_probe_pz, linear_sampler, uvw, 0.0).rgb, dir.z > 0.0);
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
    
    // Improved robust normal calculation using masks
    var normal = vec3<f32>(0.0);
    let mask = vec3<f32>(
        select(0.0, 1.0, t_near == t_min.x),
        select(0.0, 1.0, t_near == t_min.y),
        select(0.0, 1.0, t_near == t_min.z)
    );
    normal = -sign(rd) * mask;
    
    return RayHit(t_near, t_far, normal);
}

fn sample_gi_grid(world_pos: vec3<f32>, reflect_dir: vec3<f32>) -> vec4<f32> {
    let dims = vec3<f32>(camera.gi_grid_dims);
    let world_grid_origin = vec3<f32>(camera.gi_grid_origin) * 16.0;
    
    // Coarse Volume Ray Marching - Using DDA for stability at distance
    let sun_dir = normalize(camera.sun_direction);
    let brightness = camera.skybox_brightness;

    // Water plane intersection: t = (water_y - pos_y) / dir_y
    // IMPORTANT: we do NOT treat water as an infinite guaranteed hit.
    // We only accept a water hit if the intersection point is inside the GI grid
    // AND that local chunk has no geometry at/above water level.
    var t_water = -1.0;
    if (abs(reflect_dir.y) > 0.001) {
        t_water = (camera.water_level - world_pos.y) / reflect_dir.y;
    }

    // Water base color synthesized from sun and ambient
    let shallow_color = vec3<f32>(0.15, 0.45, 0.50) * brightness;
    let deep_color = vec3<f32>(0.02, 0.12, 0.20) * brightness;
    let sky_on_water = sample_sky_equirect(reflect(reflect_dir, vec3<f32>(0.0, 1.0, 0.0)));

    // Default to a simple tint if we don't compute a water intersection depth
    var water_color = camera.water_color.rgb * (brightness + 0.1);

    // If we have a valid water intersection distance, synthesize a closer match to `water.wgsl`
    if (t_water > 0.0) {
        let hit_y = world_pos.y + reflect_dir.y * t_water;
        let depth_diff = camera.water_level - hit_y;
        let max_depth = max(camera.water_visibility, 1.0);
        let depth_factor = clamp(depth_diff / max_depth, 0.0, 1.0);
        let water_tint = mix(shallow_color, deep_color, depth_factor);
        let underwater_visibility = exp(-depth_diff * 0.3);

        // Approximate underwater blend: prefer the water tint while allowing some sky influence
        water_color = mix(water_tint, sky_on_water * 0.3 + water_tint * 0.7, underwater_visibility);
    }

    // Slightly darken synthesized water reflections so they don't appear brighter than the surface
    water_color *= 0.8;

    // DDA Setup
    // Bias start point slightly to avoid self-shadowing and boundary precision issues
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
    var t_current = 0.0; // Distance from start_pos

    for (var i = 0u; i < params.max_steps; i++) {
        // Bounds check
        if (any(current_i < vec3<i32>(0)) || any(current_i >= camera.gi_grid_dims)) {
            break;
        }

        // Potential water intersection (relative to world_pos)
        if (t_water > 0.0 && (t_current + bias) >= t_water) {
            let water_hit_point = world_pos + reflect_dir * t_water;
            let water_chunk_f = (water_hit_point - world_grid_origin) / 16.0;
            let water_chunk_i = vec3<i32>(floor(water_chunk_f));

            // Only allow water if the intersection is inside the GI grid.
            var water_visible = false;
            if (all(water_chunk_i >= vec3<i32>(0)) && all(water_chunk_i < camera.gi_grid_dims)) {
                let occ = textureLoad(gi_probe_color, water_chunk_i, 0).a;
                if (occ <= 0.05) {
                    water_visible = true;
                } else {
                    let packed = textureLoad(gi_probe_bbox, water_chunk_i, 0).r;
                    let bbox_valid = (packed >> 24u) & 1u;
                    if (bbox_valid == 1u) {
                        let ymax = f32((packed >> 16u) & 0xFu);
                        let chunk_origin = vec3<f32>(water_chunk_i) * 16.0 + world_grid_origin;
                        let aabb_max_y = chunk_origin.y + (ymax + 1.0);
                        // If the chunk's highest voxel is below the water plane, water surface can exist here.
                        // Otherwise (land/buildings above water), don't force a water hit.
                        water_visible = aabb_max_y < camera.water_level;
                    }
                }
            }

            if (water_visible) {
                accumulated_color += water_color * remaining_alpha;
                remaining_alpha = 0.0;
                break;
            }

            // Water not visible here (likely land above water) or outside grid: disable water fallback.
            t_water = -1.0;
        }

        // Discrete occupancy check for quick culling
        let occupancy = textureLoad(gi_probe_color, current_i, 0).a;
        
        if (occupancy > 0.05) {
            let packed_bbox = textureLoad(gi_probe_bbox, current_i, 0).r;
            let valid = (packed_bbox >> 24u) & 1u;
            
            if (valid == 1u) {
                let xmin = f32(packed_bbox & 0xFu);
                let ymin = f32((packed_bbox >> 4u) & 0xFu);
                let zmin = f32((packed_bbox >> 8u) & 0xFu);
                let xmax = f32((packed_bbox >> 12u) & 0xFu);
                let ymax = f32((packed_bbox >> 16u) & 0xFu);
                let zmax = f32((packed_bbox >> 20u) & 0xFu);
                
                let chunk_world_origin = vec3<f32>(current_i) * 16.0 + world_grid_origin;
                let aabb_min = chunk_world_origin + vec3<f32>(xmin, ymin, zmin);
                let aabb_max = chunk_world_origin + vec3<f32>(xmax + 1.0, ymax + 1.0, zmax + 1.0);
                
                // IGNORE SUBMERGED GEOMETRY: if the entire chunk is below water, skip it
                if (aabb_max.y >= camera.water_level) {
                    let hit = ray_aabb_intersection(world_pos, reflect_dir, aabb_min, aabb_max);
                    
                    // SURFACE CLIPPING: 
                    if (hit.t_near <= hit.t_far && hit.t_far > 0.0 && hit.t_near > 0.1) {
                        // If the specific hit point is below water level, treat it as hitting water
                        let hit_point = world_pos + reflect_dir * hit.t_near;
                        if (hit_point.y < camera.water_level) {
                            accumulated_color += water_color * remaining_alpha;
                            remaining_alpha = 0.0;
                            break;
                        }

                        // Use a smooth trilinear UVW for color and radiance lookup to eliminate banding
                        let hit_uvw = (hit_point - world_grid_origin) / (dims * 16.0);
                        let hit_data = textureSampleLevel(gi_probe_color, linear_sampler, hit_uvw, 0.0);
                        let rad = sample_radiance(hit_uvw, reflect_dir);

                        let dot_sun = saturate(dot(hit.normal, sun_dir));
                        let sun_lit = camera.sun_color * camera.sun_intensity * dot_sun * 1.5;
                        
                        let base_lit_color = hit_data.rgb * (rad * 2.5 + sun_lit);

                        // Add skybox reflections for reflected surfaces.
                        // Key change: blend toward the skybox directly (not tinted by albedo)
                        // and weight it using a Fresnel term so it reads as "reflective".
                        let hit_view_dir = normalize(hit_point - camera.camera_pos);
                        let hit_reflect_dir = reflect(hit_view_dir, hit.normal);
                        let hit_sky_color = sample_sky_equirect(hit_reflect_dir);

                        // Heuristic material reflectivity proxy (we don't store reflectivity in GI probes).
                        // Use brightness to bias metallic-looking chunks to higher reflectance.
                        let surface_brightness = dot(hit_data.rgb, vec3<f32>(0.299, 0.587, 0.114));
                        let metallic_bias = saturate(surface_brightness * 1.25);

                        // Fresnel Schlick approximation.
                        // cos_theta uses the view vector from surface -> camera.
                        let view_to_cam = normalize(camera.camera_pos - hit_point);
                        let cos_theta = saturate(dot(hit.normal, view_to_cam));
                        let fresnel = pow(1.0 - cos_theta, 5.0);

                        // Base reflectance: dielectric ~0.04, increase toward ~0.6 for bright/metallic chunks.
                        let f0 = mix(0.04, 0.6, metallic_bias);
                        let reflect_w = saturate(f0 + (1.0 - f0) * fresnel);

                        // Final blend. Extra boost to make "reflection of reflective" visible.
                        let lit_color = mix(base_lit_color, hit_sky_color, reflect_w);
                        
                        accumulated_color += lit_color * remaining_alpha;
                        remaining_alpha = 0.0;
                        break; 
                    }
                }
            } else {
                // Fallback for chunks without bbox: treat as semi-transparent volume
                let entry_point = start_pos + safe_dir * t_current;
                let entry_uvw = (entry_point - world_grid_origin) / (dims * 16.0);
                
                let chunk_world_min_y = f32(current_i.y) * 16.0 + world_grid_origin.y;
                if (chunk_world_min_y + 16.0 >= camera.water_level) {
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

        // Advance DDA to next boundary
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

    // FINAL FLOOR: only allow water if we can validate it inside the GI grid.
    if (remaining_alpha > 0.0 && t_water > 0.0) {
        let water_hit_point = world_pos + reflect_dir * t_water;
        let water_chunk_f = (water_hit_point - world_grid_origin) / 16.0;
        let water_chunk_i = vec3<i32>(floor(water_chunk_f));
        if (all(water_chunk_i >= vec3<i32>(0)) && all(water_chunk_i < camera.gi_grid_dims)) {
            let occ = textureLoad(gi_probe_color, water_chunk_i, 0).a;
            var water_visible = occ <= 0.05;
            if (!water_visible) {
                let packed = textureLoad(gi_probe_bbox, water_chunk_i, 0).r;
                let bbox_valid = (packed >> 24u) & 1u;
                if (bbox_valid == 1u) {
                    let ymax = f32((packed >> 16u) & 0xFu);
                    let chunk_origin = vec3<f32>(water_chunk_i) * 16.0 + world_grid_origin;
                    let aabb_max_y = chunk_origin.y + (ymax + 1.0);
                    water_visible = aabb_max_y < camera.water_level;
                }
            }
            if (water_visible) {
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

    // Disable SSR for submerged materials to avoid underwater reflection artifacts.
    // (Water surface reflections are handled separately in `water.wgsl`.)
    if (camera.water_visibility > 0.0 && world_pos.y < camera.water_level) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let gbuf = textureSample(normal_gbuffer, linear_sampler, input.uv);
    let normal = decode_world_normal(gbuf.rgb);

    let view_dir = normalize(world_pos - camera.camera_pos);
    let reflect_dir = reflect(view_dir, normal);

    let sky_color = sample_sky_equirect(reflect_dir);
    let gi_res = sample_gi_grid(world_pos, reflect_dir);
    
    let probe_color = gi_res.rgb;
    let probe_valid = gi_res.a;

    // Blend GI grid reflection with skybox
    let out_color = mix(sky_color, probe_color, probe_valid);

    return vec4<f32>(out_color, reflectivity);
}

