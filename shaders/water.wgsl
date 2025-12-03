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
    water_elapsed_pad: vec2<f32>, // x = elapsed time for animation
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

@group(1) @binding(4)
var scene_color_texture: texture_2d<f32>;
@group(1) @binding(5)
var scene_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Constants
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// ============================================================================
// NOISE FUNCTIONS
// ============================================================================

// Hash function for procedural noise
fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// Smooth noise
fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(hash2(i + vec2<f32>(0.0, 0.0)), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

// Fractal Brownian Motion noise
fn fbm(p: vec2<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;
    
    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise2(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// Perlin-style gradient noise for smoother results
fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // Smooth interpolation
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

// ============================================================================
// WATER WAVE FUNCTIONS  
// ============================================================================

// Generate animated water normal using layered sine waves
fn get_water_normal(world_pos: vec3<f32>, time: f32) -> vec3<f32> {
    let pos = world_pos.xz;
    let wave_speed = water.speed * 0.5;
    let wave_scale = 0.08; // Base wave frequency
    
    // Multiple wave layers with different frequencies and directions
    var height_dx = 0.0;
    var height_dz = 0.0;
    
    // Wave 1: Primary large waves
    let w1_freq = wave_scale;
    let w1_dir = vec2<f32>(1.0, 0.3);
    let w1_phase = dot(pos, w1_dir) * w1_freq + time * wave_speed;
    let w1_amp = water.wave_strength * 0.4;
    height_dx += w1_amp * w1_dir.x * cos(w1_phase);
    height_dz += w1_amp * w1_dir.y * cos(w1_phase);
    
    // Wave 2: Secondary waves at angle
    let w2_freq = wave_scale * 1.8;
    let w2_dir = vec2<f32>(0.7, 0.7);
    let w2_phase = dot(pos, w2_dir) * w2_freq + time * wave_speed * 1.1;
    let w2_amp = water.wave_strength * 0.25;
    height_dx += w2_amp * w2_dir.x * cos(w2_phase);
    height_dz += w2_amp * w2_dir.y * cos(w2_phase);
    
    // Wave 3: Smaller detail waves
    let w3_freq = wave_scale * 3.2;
    let w3_dir = vec2<f32>(-0.4, 0.9);
    let w3_phase = dot(pos, w3_dir) * w3_freq + time * wave_speed * 0.9;
    let w3_amp = water.wave_strength * 0.15;
    height_dx += w3_amp * w3_dir.x * cos(w3_phase);
    height_dz += w3_amp * w3_dir.y * cos(w3_phase);
    
    // Wave 4: High frequency ripples
    let w4_freq = wave_scale * 5.5;
    let w4_dir = vec2<f32>(0.3, -0.95);
    let w4_phase = dot(pos, w4_dir) * w4_freq + time * wave_speed * 1.3;
    let w4_amp = water.wave_strength * 0.08;
    height_dx += w4_amp * w4_dir.x * cos(w4_phase);
    height_dz += w4_amp * w4_dir.y * cos(w4_phase);
    
    // Add noise-based perturbation for organic look
    let noise_scale = 0.02;
    let noise_time = time * wave_speed * 0.3;
    let n1 = gradient_noise(pos * noise_scale + vec2<f32>(noise_time, 0.0));
    let n2 = gradient_noise(pos * noise_scale + vec2<f32>(0.0, noise_time));
    height_dx += n1 * water.wave_strength * 0.1;
    height_dz += n2 * water.wave_strength * 0.1;
    
    // Construct normal from height gradients
    let normal = normalize(vec3<f32>(-height_dx, 1.0, -height_dz));
    return normal;
}

// Distorted UV coordinates for refraction effect
fn get_refraction_offset(world_pos: vec3<f32>, time: f32) -> vec2<f32> {
    let pos = world_pos.xz;
    let distort_freq = 0.05;
    let distort_speed = water.speed * 0.3;
    
    // Layer multiple noise patterns for organic distortion
    var offset = vec2<f32>(0.0);
    
    offset.x += gradient_noise(pos * distort_freq + vec2<f32>(time * distort_speed, 0.0)) * 0.5;
    offset.y += gradient_noise(pos * distort_freq + vec2<f32>(0.0, time * distort_speed)) * 0.5;
    
    offset.x += gradient_noise(pos * distort_freq * 2.0 + vec2<f32>(time * distort_speed * 0.7, 0.5)) * 0.25;
    offset.y += gradient_noise(pos * distort_freq * 2.0 + vec2<f32>(0.3, time * distort_speed * 0.7)) * 0.25;
    
    return offset * water.wave_strength * 0.01;
}

// ============================================================================
// FOAM FUNCTIONS
// ============================================================================

// Shore/intersection foam based on depth
fn get_shore_foam(depth_diff: f32, world_pos: vec3<f32>, time: f32) -> f32 {
    let foam_depth = 1.5; // Maximum depth for foam visibility
    
    if (depth_diff > foam_depth) {
        return 0.0;
    }
    
    // Base foam mask from depth
    let foam_mask = 1.0 - smoothstep(0.0, foam_depth, depth_diff);
    
    // Animated foam texture using noise
    let pos = world_pos.xz;
    let foam_speed = water.speed * 0.2;
    let foam_scale = 0.15;
    
    // Multiple foam noise layers
    let foam_uv1 = pos * foam_scale + vec2<f32>(time * foam_speed, time * foam_speed * 0.7);
    let foam_uv2 = pos * foam_scale * 1.5 - vec2<f32>(time * foam_speed * 0.8, time * foam_speed * 0.5);
    
    let foam_noise1 = fbm(foam_uv1, 3);
    let foam_noise2 = fbm(foam_uv2, 3);
    let foam_pattern = (foam_noise1 + foam_noise2) * 0.5;
    
    // Cutoff threshold that varies with depth - more foam near shore
    let cutoff = mix(0.3, 0.7, depth_diff / foam_depth);
    let foam = smoothstep(cutoff - 0.1, cutoff + 0.1, foam_pattern);
    
    return foam * foam_mask * foam_mask; // Square for sharper falloff
}

// Surface foam for open water
fn get_surface_foam(world_pos: vec3<f32>, time: f32) -> f32 {
    let pos = world_pos.xz;
    let foam_speed = water.speed * 0.1;
    let foam_scale = 0.03;
    
    // Distort UVs with sine waves for organic movement
    var foam_uv = pos * foam_scale;
    foam_uv.y += 0.01 * (sin(foam_uv.x * 3.5 + time * 0.35) + sin(foam_uv.x * 4.8 + time * 1.05)) / 2.0;
    foam_uv.x += 0.012 * (sin(foam_uv.y * 4.0 + time * 0.50) + sin(foam_uv.y * 6.8 + time * 0.75)) / 2.0;
    
    let foam_noise = fbm(foam_uv + vec2<f32>(time * foam_speed), 4);
    
    // Very sparse surface foam
    let foam = smoothstep(0.7, 0.85, foam_noise) * 0.3;
    
    return foam;
}

// ============================================================================
// CAUSTICS EFFECT
// ============================================================================

fn get_caustics(world_pos: vec3<f32>, time: f32) -> f32 {
    let pos = world_pos.xz;
    let caustic_scale = 0.12;
    let caustic_speed = water.speed * 0.4;
    
    // Two overlapping caustic patterns
    let c1 = fbm(pos * caustic_scale + vec2<f32>(time * caustic_speed, time * caustic_speed * 0.7), 3);
    let c2 = fbm(pos * caustic_scale * 1.2 - vec2<f32>(time * caustic_speed * 0.8, time * caustic_speed), 3);
    
    // Create caustic pattern from interference
    let caustics = pow(c1 * c2, 0.5) * 2.0;
    
    return clamp(caustics, 0.0, 1.0);
}

// ============================================================================
// SPECULAR HIGHLIGHTS
// ============================================================================

fn get_specular(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, smoothness: f32) -> f32 {
    let half_vec = normalize(light_dir + view_dir);
    let ndoth = max(dot(normal, half_vec), 0.0);
    let spec_power = exp2(10.0 * smoothness + 1.0);
    return pow(ndoth, spec_power);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Project world position to screen UV coordinates
fn world_to_screen_uv(world_pos: vec3<f32>) -> vec3<f32> {
    let clip_pos = camera.mvp * vec4<f32>(world_pos, 1.0);
    let ndc = clip_pos.xyz / clip_pos.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3<f32>(uv, ndc.z);
}

// Reconstruct world position from UV and depth
fn reconstruct_world_pos_uv(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let z_ndc = depth * 2.0 - 1.0;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, z_ndc, 1.0);
    let view_pos_unnorm = camera.inverse_proj * ndc;
    let view_pos = view_pos_unnorm.xyz / view_pos_unnorm.w;
    return (camera.inverse_view * vec4<f32>(view_pos, 1.0)).xyz;
}

// Screen-space reflection ray marching
fn trace_water_reflection(start_pos: vec3<f32>, ray_dir: vec3<f32>, cam_pos: vec3<f32>) -> vec3<f32> {
    let max_steps = 32u;
    let step_size = 2.0;
    let thickness = 1.5;
    
    let max_dist = f32(max_steps) * step_size;
    let end_pos = start_pos + ray_dir * max_dist;
    
    let start_screen = world_to_screen_uv(start_pos);
    let end_screen = world_to_screen_uv(end_pos);
    
    let delta = (end_screen - start_screen) / f32(max_steps);
    var current_screen = start_screen;
    
    let dim = textureDimensions(depth_texture);
    
    for (var i = 0u; i < max_steps; i++) {
        current_screen += delta;
        
        let uv = current_screen.xy;
        
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }
        
        let coords = vec2<i32>(vec2<f32>(dim) * uv);
        if (coords.x < 0 || coords.x >= i32(dim.x) || coords.y < 0 || coords.y >= i32(dim.y)) {
            break;
        }
        let scene_depth = textureLoad(depth_texture, coords, 0);
        
        if (scene_depth >= 0.9999) {
            continue;
        }
        
        let ray_depth = current_screen.z;
        
        if (ray_depth > scene_depth) {
            let surface_pos = reconstruct_world_pos_uv(uv, scene_depth);
            let ray_world_pos = reconstruct_world_pos_uv(uv, ray_depth);
            let dist_diff = distance(ray_world_pos, surface_pos);
            
            if (dist_diff < thickness) {
                let edge_fade = min(
                    min(uv.x, 1.0 - uv.x),
                    min(uv.y, 1.0 - uv.y)
                );
                let edge_factor = smoothstep(0.0, 0.1, edge_fade);
                return vec3<f32>(uv, edge_factor);
            }
        }
    }
    
    return vec3<f32>(0.0, 0.0, 0.0);
}

// ============================================================================
// VERTEX SHADER
// ============================================================================

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2<f32>(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u));
    out.clip_position = vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0);
    out.uv = uv;
    return out;
}

// ============================================================================
// FRAGMENT SHADER
// ============================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Get time from camera uniforms
    let time = camera.water_elapsed_pad.x; // Elapsed time for animation
    let brightness = camera.fog_time_pad.w;
    
    // Reconstruct ray direction
    let ndc = vec4<f32>(in.uv * 2.0 - 1.0, 1.0, 1.0);
    let view_space_pos = camera.inverse_proj * ndc;
    let view_space_dir = view_space_pos.xyz / view_space_pos.w;
    let world_dir = normalize((camera.inverse_view * vec4<f32>(view_space_dir, 0.0)).xyz);
    let cam_pos = camera.camera_shadow_strength.xyz;

    let water_level = water.water_level;
    
    // Only render water surface from above
    if (cam_pos.y > water_level && world_dir.y >= -0.0001) {
        discard;
    }
    
    // Intersect with water plane
    let t = (water_level - cam_pos.y) / world_dir.y;
    
    if (t <= 0.0) {
        discard;
    }

    let hit_pos = cam_pos + t * world_dir;
    let clip_pos = camera.mvp * vec4<f32>(hit_pos, 1.0);
    let ndc_pos = clip_pos.xyz / clip_pos.w;
    let screen_uv = vec2<f32>(ndc_pos.x * 0.5 + 0.5, 0.5 - ndc_pos.y * 0.5);
    
    let dim = textureDimensions(depth_texture);
    let coords = vec2<i32>(vec2<f32>(dim) * screen_uv);
    
    if (coords.x < 0 || coords.x >= i32(dim.x) || coords.y < 0 || coords.y >= i32(dim.y)) {
        discard;
    }

    let scene_depth_raw = textureLoad(depth_texture, coords, 0);
    let water_depth = ndc_pos.z;
    
    // Discard if water is behind scene geometry
    if (water_depth >= scene_depth_raw) {
        discard;
    }

    // ========================================================================
    // WATER SURFACE PROPERTIES
    // ========================================================================
    
    // Animated water normal
    let water_normal = get_water_normal(hit_pos, time);
    let view_dir = -world_dir;
    
    // Distance from camera to water
    let dist = distance(cam_pos, hit_pos);
    
    // ========================================================================
    // DEPTH CALCULATION (World-space vertical depth)
    // ========================================================================
    
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
    
    // Vertical depth: how far below water surface
    let depth_diff = water_level - scene_world_pos.y;
    
    // ========================================================================
    // FRESNEL EFFECT
    // ========================================================================
    
    let ndotv = max(dot(water_normal, view_dir), 0.0);
    // Schlick's approximation with water IOR (~1.33)
    let f0 = 0.02; // Fresnel at normal incidence for water
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, 5.0);
    
    // Boost reflection at grazing angles
    let reflection_strength = mix(0.15, 0.95, fresnel);
    
    // ========================================================================
    // REFRACTION - Sample underwater scene with distortion
    // ========================================================================
    
    let refract_offset = get_refraction_offset(hit_pos, time);
    let refracted_uv = clamp(screen_uv + refract_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    
    // Sample underwater scene with refraction
    let underwater_color = textureSample(scene_color_texture, scene_sampler, refracted_uv).rgb;
    
    // ========================================================================
    // DEPTH-BASED WATER COLOR
    // ========================================================================
    
    // Shallow and deep water colors
    let shallow_color = vec3<f32>(0.15, 0.45, 0.50) * brightness;
    let deep_color = vec3<f32>(0.02, 0.12, 0.20) * brightness;
    
    // Depth fade factor (world-space)
    let max_depth = max(camera.water_visibility, 1.0);
    let depth_factor = clamp(depth_diff / max_depth, 0.0, 1.0);
    
    // Blend between shallow and deep colors
    let water_tint = mix(shallow_color, deep_color, depth_factor);
    
    // Blend underwater scene with water tint based on depth
    // Shallow water: more transparent, show underwater scene
    // Deep water: more opaque, dominated by water color
    let underwater_visibility = exp(-depth_diff * 0.3);
    var water_color = mix(water_tint, underwater_color * 0.7 + water_tint * 0.3, underwater_visibility);
    
    // ========================================================================
    // CAUSTICS (on underwater surfaces)
    // ========================================================================
    
    if (scene_depth_raw < 0.9999 && depth_diff > 0.0 && depth_diff < max_depth) {
        let caustic_strength = get_caustics(scene_world_pos, time);
        let caustic_fade = (1.0 - depth_factor) * 0.3 * brightness;
        water_color += vec3<f32>(caustic_strength * caustic_fade);
    }
    
    // ========================================================================
    // REFLECTION - Skybox + SSR
    // ========================================================================
    
    // Reflect view vector around water normal
    let reflect_dir_raw = reflect(world_dir, water_normal);
    
    // Apply skybox rotation
    let angle = camera.fog_time_pad.z;
    let c = cos(angle);
    let s = sin(angle);
    let reflect_dir = vec3<f32>(
        reflect_dir_raw.x * c + reflect_dir_raw.z * s,
        reflect_dir_raw.y,
        reflect_dir_raw.x * -s + reflect_dir_raw.z * c
    );
    
    // Sample skybox (equirectangular)
    let u = 0.5 + atan2(reflect_dir.z, reflect_dir.x) / TWO_PI;
    let v = 0.5 - asin(clamp(reflect_dir.y, -1.0, 1.0)) / PI;
    
    let sky_sample = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v)).rgb;
    
    // Apply brightness and desaturation (match skybox pass)
    let min_sat = camera.skybox_saturation_pad.x;
    let sat = clamp(min_sat + (1.0 - min_sat) * brightness, 0.0, 1.0);
    let luminance = dot(sky_sample, vec3<f32>(0.299, 0.587, 0.114));
    let desaturated = mix(vec3<f32>(luminance), sky_sample, sat);
    let tint = camera.skybox_tint_pad.xyz;
    let tint_strength = camera.skybox_tint_pad.w;
    let effect_strength = (1.0 - brightness) * tint_strength;
    let tinted = mix(desaturated, desaturated * tint, effect_strength);
    var reflection_color = tinted * brightness;
    
    // Screen-space reflections
    let ssr_hit = trace_water_reflection(hit_pos, reflect_dir_raw, cam_pos);
    let ssr_hit_valid = ssr_hit.z;
    
    if (ssr_hit_valid > 0.0) {
        let scene_sample = textureSample(scene_color_texture, scene_sampler, ssr_hit.xy);
        let ssr_color = scene_sample.rgb;
        
        let ssr_max_dist = 150.0;
        let ssr_dist_fade = clamp((ssr_max_dist - dist) / ssr_max_dist, 0.0, 1.0);
        let ssr_effect = ssr_hit_valid * ssr_dist_fade;
        
        reflection_color = mix(reflection_color, ssr_color, ssr_effect);
    }
    
    // ========================================================================
    // SPECULAR HIGHLIGHTS
    // ========================================================================
    
    let sun_dir = normalize(camera.sun_direction_shadow_bias.xyz);
    let sun_color = camera.sun_color_pad.xyz;
    
    // Only add specular when sun is above horizon
    var specular = 0.0;
    if (sun_dir.y > 0.0) {
        let sun_spec = get_specular(water_normal, view_dir, sun_dir, 0.9);
        // Stylized hard specular highlights
        specular = smoothstep(0.5, 0.55, sun_spec) * brightness;
    }
    
    // ========================================================================
    // FOAM
    // ========================================================================
    
    let shore_foam = get_shore_foam(depth_diff, hit_pos, time);
    let surface_foam = get_surface_foam(hit_pos, time);
    let total_foam = clamp(shore_foam + surface_foam, 0.0, 1.0);
    
    let foam_color = vec3<f32>(0.9, 0.95, 1.0) * brightness;
    
    // ========================================================================
    // HORIZON COLOR (Fresnel-based)
    // ========================================================================
    
    // Add horizon color for distant water
    let horizon_color = vec3<f32>(0.4, 0.5, 0.55) * brightness;
    let horizon_blend = pow(1.0 - ndotv, 3.0) * 0.5;
    
    // ========================================================================
    // FINAL COMPOSITION
    // ========================================================================
    
    // Mix water color with reflection based on Fresnel
    var final_color = mix(water_color, reflection_color, reflection_strength);
    
    // Add horizon tint
    final_color = mix(final_color, horizon_color, horizon_blend * 0.3);
    
    // Add specular highlights
    final_color += sun_color * specular * 0.8;
    
    // Overlay foam
    final_color = mix(final_color, foam_color, total_foam);
    
    // ========================================================================
    // ALPHA / TRANSPARENCY
    // ========================================================================
    
    // Base alpha from water color
    var alpha = water.water_color.a;
    
    // More opaque at grazing angles
    alpha = mix(alpha, 1.0, fresnel * 0.6);
    
    // Increase opacity with depth
    let absorption_coeff = 3.0 / max_depth;
    let depth_opacity = 1.0 - exp(-absorption_coeff * max(depth_diff, 0.0));
    alpha = mix(alpha, 1.0, depth_opacity);
    
    // Foam is fully opaque
    alpha = mix(alpha, 1.0, total_foam);
    
    // Shore fade
    var shore_fade = smoothstep(0.0, 0.8, depth_diff);
    if (scene_depth_raw >= 0.9999) {
        shore_fade = 1.0;
    }
    
    return vec4<f32>(final_color, alpha * shore_fade);
}
