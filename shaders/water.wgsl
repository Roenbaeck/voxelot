struct CameraUniforms {
    mvp: mat4x4<f32>,
    sun_view_proj: mat4x4<f32>,
    camera_shadow_strength: vec4<f32>,
    sun_direction_shadow_bias: vec4<f32>,
    fog_time_pad: vec4<f32>,
    sun_color_pad: vec4<f32>,
    ambient_color_pad: vec4<f32>,
    shadow_texel_size_pad: vec4<f32>,
    shadow_darkness_pad: vec4<f32>,
    moon_direction_intensity: vec4<f32>,
    moon_color_pad: vec4<f32>,
    skybox_saturation_pad: vec4<f32>,
    skybox_tint_pad: vec4<f32>,
    light_probe_count: u32,
    lod_distance: f32,
    envelope_distance: f32,
    envelope_fade_range: f32,
    water_level: f32,
    water_visibility: f32,
    _water_pad: vec2<f32>,
    inverse_view: mat4x4<f32>,
    inverse_proj: mat4x4<f32>,
};

struct WaterUniforms {
    water_level: f32,
    wave_strength: f32,
    speed: f32,
    _pad0: f32,
    water_color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<uniform> water: WaterUniforms;

@group(1) @binding(1)
var skybox_texture: texture_2d<f32>;
@group(1) @binding(2)
var skybox_sampler: sampler;

@group(1) @binding(3)
var depth_texture: texture_depth_2d;
// We don't need a sampler for depth texture load

@group(1) @binding(4)
var scene_color_texture: texture_2d<f32>;
@group(1) @binding(5)
var scene_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Project world position to screen UV coordinates
fn world_to_screen_uv(world_pos: vec3<f32>) -> vec3<f32> {
    let clip_pos = camera.mvp * vec4<f32>(world_pos, 1.0);
    let ndc = clip_pos.xyz / clip_pos.w;
    // Convert NDC to UV: x: [-1,1] -> [0,1], y: [-1,1] -> [1,0] (flip Y)
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3<f32>(uv, ndc.z);
}

// Reconstruct world position from UV and depth
fn reconstruct_world_pos_uv(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert depth from [0,1] to NDC [-1,1] for inverse projection
    let z_ndc = depth * 2.0 - 1.0;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, z_ndc, 1.0);
    let view_pos_unnorm = camera.inverse_proj * ndc;
    let view_pos = view_pos_unnorm.xyz / view_pos_unnorm.w;
    return (camera.inverse_view * vec4<f32>(view_pos, 1.0)).xyz;
}

// Simple ray march to find where reflection ray hits scene geometry
// Returns: vec3(hit_uv.x, hit_uv.y, hit_valid) where hit_valid > 0 means valid hit
// Simple ray march to find where reflection ray hits scene geometry
// Returns: vec3(hit_uv.x, hit_uv.y, hit_valid) where hit_valid > 0 means valid hit
fn trace_water_reflection(start_pos: vec3<f32>, ray_dir: vec3<f32>, cam_pos: vec3<f32>) -> vec3<f32> {
    let max_steps = 32u;
    let step_size = 2.0; // World units per step
    let thickness = 1.5; // Tolerance for hit detection
    
    // Calculate end position
    let max_dist = f32(max_steps) * step_size;
    let end_pos = start_pos + ray_dir * max_dist;
    
    // Project to screen space
    let start_screen = world_to_screen_uv(start_pos);
    let end_screen = world_to_screen_uv(end_pos);
    
    let delta = (end_screen - start_screen) / f32(max_steps);
    var current_screen = start_screen;
    
    // Get depth texture dimensions for bounds checking
    let dim = textureDimensions(depth_texture);
    
    for (var i = 0u; i < max_steps; i++) {
        current_screen += delta;
        
        let uv = current_screen.xy;
        
        // Check screen bounds
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }
        
        // Sample scene depth at this UV
        let coords = vec2<i32>(vec2<f32>(dim) * uv);
        if (coords.x < 0 || coords.x >= i32(dim.x) || coords.y < 0 || coords.y >= i32(dim.y)) {
            break;
        }
        let scene_depth = textureLoad(depth_texture, coords, 0);
        
        // Skip sky pixels
        if (scene_depth >= 0.9999) {
            continue;
        }
        
        // Compare ray depth vs scene depth
        // Note: scene_depth is raw depth buffer value (0..1)
        // current_screen.z is also projected depth (0..1)
        
        let ray_depth = current_screen.z;
        
        // Check if ray is behind surface
        // In standard depth (0=near, 1=far), larger value means further away
        if (ray_depth > scene_depth) {
            // We need to check thickness.
            // Converting thickness to depth units is non-linear and tricky.
            // For now, let's reconstruct world position for the thickness check to be safe,
            // OR approximate thickness in depth units.
            // Reconstructing world pos is expensive, which we want to avoid.
            // But we only do it IF we hit something.
            
            let surface_pos = reconstruct_world_pos_uv(uv, scene_depth);
            
            // Reconstruct ray pos from screen pos? Or just use interpolated depth?
            // We can reconstruct ray world pos from current_screen
            let ray_world_pos = reconstruct_world_pos_uv(uv, ray_depth);
            
            let dist_diff = distance(ray_world_pos, surface_pos);
            
            if (dist_diff < thickness) {
                // Edge fade for smoother blending near screen edges
                let edge_fade = min(
                    min(uv.x, 1.0 - uv.x),
                    min(uv.y, 1.0 - uv.y)
                );
                let edge_factor = smoothstep(0.0, 0.1, edge_fade);
                return vec3<f32>(uv, edge_factor);
            }
        }
    }
    
    return vec3<f32>(0.0, 0.0, 0.0); // No hit
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Full screen triangle
    let uv = vec2<f32>(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u));
    out.clip_position = vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0); // z = 1.0 (far plane)
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1. Reconstruct ray direction
    let ndc = vec4<f32>(in.uv * 2.0 - 1.0, 1.0, 1.0);
    let view_space_pos = camera.inverse_proj * ndc;
    let view_space_dir = view_space_pos.xyz / view_space_pos.w;
    let world_dir = normalize((camera.inverse_view * vec4<f32>(view_space_dir, 0.0)).xyz);
    let cam_pos = camera.camera_shadow_strength.xyz;

    // 2. Intersect with water plane (y = water_level)
    // Ray: P = O + t * D
    // Plane: P.y = water_level
    // O.y + t * D.y = water_level
    // t = (water_level - O.y) / D.y

    let water_level = water.water_level;
    
    // Check if looking away from plane
    // If camera is above water (cam_pos.y > water_level), we need D.y < 0 to hit it.
    // If camera is below water (cam_pos.y < water_level), we need D.y > 0 to hit it (surface from below).
    // For now, let's assume we only render top surface if above water.
    
    if (cam_pos.y > water_level && world_dir.y >= -0.0001) {
        discard;
    }
    
    let t = (water_level - cam_pos.y) / world_dir.y;
    
    if (t <= 0.0) {
        discard;
    }

    // 3. Check against scene depth
    // We need to project the hit position back to screen space to get depth
    let hit_pos = cam_pos + t * world_dir;
    let clip_pos = camera.mvp * vec4<f32>(hit_pos, 1.0);
    let ndc_pos = clip_pos.xyz / clip_pos.w;
    let screen_uv = vec2<f32>(ndc_pos.x * 0.5 + 0.5, 0.5 - ndc_pos.y * 0.5);
    
    // Sample scene depth
    // textureLoad uses integer coordinates
    let dim = textureDimensions(depth_texture);
    let coords = vec2<i32>(vec2<f32>(dim) * screen_uv);
    
    // Bounds check
    if (coords.x < 0 || coords.x >= i32(dim.x) || coords.y < 0 || coords.y >= i32(dim.y)) {
        // Should not happen if math is correct, but safe guard
        discard;
    }

    let scene_depth_raw = textureLoad(depth_texture, coords, 0);
    
    // We need to compare linear depths or just check if scene is closer.
    // In standard depth buffer (0..1), closer objects have smaller depth (if clear=1.0 and func=less).
    // Wait, wgpu default is 0.0 = near, 1.0 = far.
    // Let's check the clip_pos.z / clip_pos.w against scene_depth_raw.
    // Actually, clip_pos.z / clip_pos.w is in NDC [-1, 1] for OpenGL, but [0, 1] for WebGPU/Vulkan/Metal/DX12?
    // wgpu uses [0, 1] for z in NDC.
    
    let water_depth = ndc_pos.z;
    
    // If water is behind scene geometry, discard
    // Note: using a small epsilon to avoid z-fighting at shore
    if (water_depth >= scene_depth_raw) {
        discard;
    }

    // 4. Render Water
    
    // Fresnel
    // Normal is up (0, 1, 0)
    let normal = vec3<f32>(0.0, 1.0, 0.0);
    let view_dir = -world_dir;
    let ndotv = max(dot(normal, view_dir), 0.0);
    let fresnel = pow(1.0 - ndotv, 4.0);
    let reflection_strength = mix(0.1, 0.8, fresnel); // Base reflectivity + fresnel
    
    // Reflection (Skybox)
    // Reflect view vector around normal
    let reflect_dir_raw = reflect(world_dir, normal);

    // Apply skybox rotation to reflection vector
    let angle = camera.fog_time_pad.z;
    let c = cos(angle);
    let s = sin(angle);
    let reflect_dir = vec3<f32>(
        reflect_dir_raw.x * c + reflect_dir_raw.z * s,
        reflect_dir_raw.y,
        reflect_dir_raw.x * -s + reflect_dir_raw.z * c
    );
    
    // Sample skybox (equirectangular)
    let u = 0.5 + atan2(reflect_dir.z, reflect_dir.x) / (2.0 * 3.14159265);
    let v = 0.5 - asin(reflect_dir.y) / 3.14159265;
    
    // Sample and apply brightness + desaturation + tint (to match skybox pass)
    let sky_sample = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v)).rgb;
    let brightness = camera.fog_time_pad.w;
    let min_sat = camera.skybox_saturation_pad.x;
    let sat = clamp(min_sat + (1.0 - min_sat) * brightness, 0.0, 1.0);
    let luminance = dot(sky_sample, vec3<f32>(0.299, 0.587, 0.114));
    let desaturated = mix(vec3<f32>(luminance), sky_sample, sat);
    let tint = camera.skybox_tint_pad.xyz;
    let tint_strength = camera.skybox_tint_pad.w;
    let effect_strength = (1.0 - brightness) * tint_strength;
    let tinted = mix(desaturated, desaturated * tint, effect_strength);
    let reflection_color = tinted * brightness;
    
    // Base water color (darken at night)
    let base_color = water.water_color.rgb * brightness;
    
    // Distance from camera to water hit point (for SSR distance fade)
    let dist = distance(cam_pos, hit_pos);
    
    // Trace reflection ray to find where it hits scene geometry
    // Use the unrotated reflect_dir_raw since we need world-space coordinates
    let ssr_hit = trace_water_reflection(hit_pos, reflect_dir_raw, cam_pos);
    let ssr_hit_valid = ssr_hit.z;
    
    var combined_reflection = reflection_color; // Default to skybox
    
    if (ssr_hit_valid > 0.0) {
        // Sample scene color at the hit UV
        let scene_sample = textureSample(scene_color_texture, scene_sampler, ssr_hit.xy);
        let ssr_color = scene_sample.rgb;
        
        // Distance-based fade: weaken SSR at long distances
        let ssr_max_dist = 150.0;
        let ssr_dist_fade = clamp((ssr_max_dist - dist) / ssr_max_dist, 0.0, 1.0);
        
        // Combine edge fade from ray march and distance fade
        let ssr_effect = ssr_hit_valid * ssr_dist_fade;
        
        // Blend SSR with skybox reflection
        combined_reflection = mix(reflection_color, ssr_color, ssr_effect);
    }

    // Mix based on reflection strength (Fresnel)
    var final_rgb = mix(base_color, combined_reflection, reflection_strength);
    
    // Alpha
    // More opaque at grazing angles, more transparent looking down
    var alpha = mix(water.water_color.a, 1.0, fresnel * 0.5);
    
    // Soft shore fade
    // Reconstruct scene world position to get actual distance
    // scene_depth_raw is 0..1 (wgpu default)
    // inverse_proj expects -1..1 (OpenGL convention used in voxelot.rs)
    let z_ndc = scene_depth_raw * 2.0 - 1.0;
    
    let scene_ndc = vec4<f32>(
        screen_uv.x * 2.0 - 1.0,
        1.0 - 2.0 * screen_uv.y,
        z_ndc,
        1.0
    );
    
    let scene_view_pos_unnorm = camera.inverse_proj * scene_ndc;
    let scene_view_pos = scene_view_pos_unnorm.xyz / scene_view_pos_unnorm.w;
    let scene_world_pos = (camera.inverse_view * vec4<f32>(scene_view_pos, 1.0)).xyz;
    
    // Calculate vertical depth difference
    // Water is at y = water_level
    // If scene is underwater, scene_world_pos.y < water_level
    let depth_diff = water_level - scene_world_pos.y;
    
    var shore_fade = smoothstep(0.0, 1.0, depth_diff); // Fade over 1.0 unit
    
    // Don't fade against the skybox (far plane)
    if (scene_depth_raw >= 0.9999) {
        shore_fade = 1.0;
    }
    
    // Depth-based visibility: deeper underwater = more obscured
    // depth_diff is how far below water level (in world units/voxels)
    // Use water_visibility from config to control falloff
    let depth_visibility_falloff = camera.water_visibility;
    let depth_factor = clamp(depth_diff / depth_visibility_falloff, 0.0, 1.0);
    
    // Increase opacity with depth (objects deeper are harder to see through water)
    alpha = mix(alpha, 1.0, depth_factor * 0.85);
    
    // Also tint deeper water more blue/dark
    let deep_water_tint = vec3<f32>(0.1, 0.2, 0.35) * brightness;
    final_rgb = mix(final_rgb, deep_water_tint, depth_factor * 0.7);
    
    return vec4<f32>(final_rgb, alpha * shore_fade);
}
