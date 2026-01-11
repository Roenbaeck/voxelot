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

    // Boat/wake parameters (optional)
    // boat_pos_wake.xyz = boat position, boat_pos_wake.w = wake strength (0 disables)
    boat_pos_wake: vec4<f32>,
    // boat_dir_speed.xyz = forward direction, boat_dir_speed.w = horizontal speed
    boat_dir_speed: vec4<f32>,

    // DoF parameters: focal distance and focal range (for CoC-consistent reflection blur)
    dof_focal_distance: f32,
    dof_focal_range: f32,
    _pad_dof: f32,
};

// ============================================================================
// BOAT WAKE
// ============================================================================

fn get_boat_wake(world_pos: vec3<f32>, time: f32) -> f32 {
    let strength = water.boat_pos_wake.w;
    if (strength <= 0.0001) {
        return 0.0;
    }

    let boat_pos = water.boat_pos_wake.xyz;
    let d = world_pos.xz - boat_pos.xz;

    var fwd = water.boat_dir_speed.xz;
    let fwd_len = length(fwd);
    if (fwd_len < 0.0001) {
        return 0.0;
    }
    fwd = fwd / fwd_len;
    let right = vec2<f32>(-fwd.y, fwd.x);

    let along = dot(d, fwd);
    let side = dot(d, right);

    // Only behind the boat (negative along)
    if (along > 0.0) {
        return 0.0;
    }

    let t = -along;
    let speed = water.boat_dir_speed.w;
    let speed_factor = clamp(speed * 0.05, 0.0, 1.0);

    // Wake starts narrow behind the stern and spreads as it goes back.
    // Keep the near-boat wake tighter so the bow doesn't look too wide.
    // Start narrower right behind the boat, then widen gradually.
    let spread = 0.25 + t * 0.10;
    let core = exp(-(side * side) / (spread * spread));
    let fade = exp(-t / 35.0);

    // Add gentle bands so it reads as waves
    let bands = 0.65 + 0.35 * sin(t * 1.8 - time * 4.0);

    // A small disturbance ring *behind* the stern (not around the bow)
    let ring_center = boat_pos.xz - fwd * 1.2;
    let rd = world_pos.xz - ring_center;
    let r = length(rd);
    let ring = exp(-((r - 1.1) * (r - 1.1)) / 0.25) * 0.35;

    return strength * speed_factor * (core * fade * bands + ring);
}

fn get_boat_wake_normal(world_pos: vec3<f32>, time: f32, base_normal: vec3<f32>) -> vec3<f32> {
    let strength = water.boat_pos_wake.w;
    if (strength <= 0.0001) {
        return base_normal;
    }

    // Approximate a height-field gradient from the wake scalar (screen-space friendly).
    // This gives us visible "ridges" in reflection/specular without changing geometry.
    // Smaller sampling radius = tighter, more localized ripples.
    let eps = 0.40;
    let w0 = get_boat_wake(world_pos, time);
    if (w0 <= 0.0001) {
        return base_normal;
    }
    let wx = get_boat_wake(world_pos + vec3<f32>(eps, 0.0, 0.0), time);
    let wz = get_boat_wake(world_pos + vec3<f32>(0.0, 0.0, eps), time);

    let ddx = (wx - w0) / eps;
    let ddz = (wz - w0) / eps;

    // Scale is intentionally subtle; wake scalar already includes speed/strength.
    let wake_normal_strength = 1.35;
    let n = normalize(base_normal + vec3<f32>(-ddx * wake_normal_strength, 0.0, -ddz * wake_normal_strength));
    return n;
}

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

// Normal G-buffer: RGB = world normal encoded as n*0.5+0.5, A = view_z (positive view-space depth)
@group(1) @binding(6)
var normal_gbuffer: texture_2d<f32>;

// HZB texture for accelerated SSR
@group(1) @binding(7)
var hzb_texture: texture_2d<f32>;
@group(1) @binding(8)
var hzb_sampler: sampler;

@group(1) @binding(9)
var ssr_texture: texture_2d<f32>;

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
    
    // Large Waves (Reduced complexity)
    let w1_dir = vec2<f32>(1.0, 0.3);
    let w1_phase = dot(pos, w1_dir) * wave_scale + time * wave_speed;
    let w1_amp = water.wave_strength * 0.4;
    height_dx += w1_amp * w1_dir.x * cos(w1_phase);
    height_dz += w1_amp * w1_dir.y * cos(w1_phase);
    
    let w2_dir = vec2<f32>(0.7, 0.7);
    let w2_phase = dot(pos, w2_dir) * (wave_scale * 1.8) + time * (wave_speed * 1.1);
    let w2_amp = water.wave_strength * 0.25;
    height_dx += w2_amp * w2_dir.x * cos(w2_phase);
    height_dz += w2_amp * w2_dir.y * cos(w2_phase);
    
    // Single noise layer for organic detail (instead of 2 + 4 wave layers)
    let noise_scale = 0.02;
    let noise_time = time * wave_speed * 0.3;
    let n1 = gradient_noise(pos * noise_scale + vec2<f32>(noise_time, noise_time * 0.5));
    height_dx += n1 * water.wave_strength * 0.15;
    height_dz += n1 * water.wave_strength * 0.15;
    
    // Construct normal from height gradients
    return normalize(vec3<f32>(-height_dx, 1.0, -height_dz));
}

// Distorted UV coordinates for refraction effect
fn get_refraction_offset(world_pos: vec3<f32>, time: f32) -> vec2<f32> {
    let pos = world_pos.xz;
    let distort_freq = 0.05;
    let distort_speed = water.speed * 0.3;
    
    // Reduced to single noise sample for refraction
    let n = gradient_noise(pos * distort_freq + vec2<f32>(time * distort_speed, time * distort_speed * 0.7));
    return vec2<f32>(n, n * 0.5) * water.wave_strength * 0.01;
}

// ============================================================================
// FOAM FUNCTIONS
// ============================================================================

// Shore/intersection foam based on depth
fn get_shore_foam(depth_diff: f32, world_pos: vec3<f32>, time: f32) -> f32 {
    let foam_depth = 4.0; // Maximum depth for foam visibility (increased)
    
    if (depth_diff > foam_depth || depth_diff < 0.0) {
        return 0.0;
    }
    
    // Base foam mask from depth - stronger near shore
    let depth_normalized = depth_diff / foam_depth;
    let foam_mask = pow(1.0 - depth_normalized, 1.5);
    
    // Animated foam texture using noise
    let pos = world_pos.xz;
    let foam_speed = water.speed * 0.15;
    let foam_scale = 0.25; // Larger scale for more visible foam patches
    
    // Multiple foam noise layers
    let foam_uv1 = pos * foam_scale + vec2<f32>(time * foam_speed, time * foam_speed * 0.7);
    let foam_uv2 = pos * foam_scale * 0.7 - vec2<f32>(time * foam_speed * 0.6, time * foam_speed * 0.4);
    
    let foam_noise1 = fbm(foam_uv1, 3);
    let foam_noise2 = fbm(foam_uv2, 3);
    let foam_pattern = (foam_noise1 + foam_noise2) * 0.5;
    
    // Lower cutoff threshold for more visible foam
    let cutoff = mix(0.25, 0.55, depth_normalized);
    let foam = smoothstep(cutoff - 0.15, cutoff + 0.05, foam_pattern);
    
    // Add a subtle foam line right at the water's edge (reduced intensity)
    let edge_foam = smoothstep(0.5, 0.0, depth_diff) * 0.4;
    
    return max(foam * foam_mask, edge_foam);
}

// Surface foam for open water - disabled for smoother look
fn get_surface_foam(world_pos: vec3<f32>, time: f32) -> f32 {
    // Disabled - was creating too much noise on water surface
    return 0.0;
}

// ============================================================================
// CAUSTICS EFFECT
// ============================================================================

fn get_caustics(world_pos: vec3<f32>, time: f32) -> f32 {
    let pos = world_pos.xz;
    let caustic_scale = 0.06; // Larger scale = smoother pattern
    let caustic_speed = water.speed * 0.25;
    
    // Use smooth sine-based pattern instead of noise
    let p1 = pos * caustic_scale + vec2<f32>(time * caustic_speed, time * caustic_speed * 0.7);
    let p2 = pos * caustic_scale * 0.8 - vec2<f32>(time * caustic_speed * 0.6, time * caustic_speed * 0.9);
    
    // Smooth sine-based caustic pattern
    let c1 = sin(p1.x * 2.0 + p1.y * 1.5) * 0.5 + 0.5;
    let c2 = sin(p2.x * 1.8 - p2.y * 2.2) * 0.5 + 0.5;
    let c3 = sin((p1.x + p2.y) * 1.2) * 0.5 + 0.5;
    
    // Combine for caustic-like interference
    let caustics = c1 * c2 * c3 * 1.5;
    
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

fn decode_world_normal(encoded: vec3<f32>) -> vec3<f32> {
    let n = encoded * 2.0 - 1.0;
    return normalize(select(n, vec3<f32>(0.0, 1.0, 0.0), dot(n, n) < 1e-8));
}

// Project world position to screen UV coordinates
fn world_to_screen_uv(world_pos: vec3<f32>) -> vec3<f32> {
    let clip_pos = camera.mvp * vec4<f32>(world_pos, 1.0);
    // Use rcp to avoid two divisions
    let inv_w = 1.0 / clip_pos.w;
    let ndc = clip_pos.xyz * inv_w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

// Faster bitwise hash for M1
fn fast_hash(p: vec2<f32>) -> f32 {
    let v = bitcast<vec2<u32>>(p);
    let q = (v.x * 1597334677u) ^ (v.y * 3812015487u);
    let h = (q ^ (q >> 16u)) * 1597334677u;
    return f32(h) * (1.0 / 4294967296.0);
}

// Perlin-style gradient noise with optimized hash
fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // Smooth interpolation (Hermite)
    let u = f * f * (3.0 - 2.0 * f);
    
    // Use bitwise hash instead of sin
    let a = fast_hash(i);
    let b = fast_hash(i + vec2<f32>(1.0, 0.0));
    let c = fast_hash(i + vec2<f32>(0.0, 1.0));
    let d = fast_hash(i + vec2<f32>(1.0, 1.0));
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

// Calculate max mip level for HZB
fn get_max_mip_level() -> f32 {
    let dims = vec2<f32>(textureDimensions(hzb_texture, 0));
    let max_dim = max(dims.x, dims.y);
    return floor(log2(max_dim));
}

// Screen-space reflection ray marching with HZB acceleration
fn trace_water_reflection(start_pos: vec3<f32>, ray_dir: vec3<f32>, cam_pos: vec3<f32>, pixel_uv: vec2<f32>) -> vec3<f32> {
    let max_steps = 40u;
    let thickness_base = 5.0;
    
    let max_dist = 400.0;
    let end_pos = start_pos + ray_dir * max_dist;
    
    let start_screen = world_to_screen_uv(start_pos);
    let end_screen = world_to_screen_uv(end_pos);
    
    let delta = (end_screen - start_screen) / f32(max_steps);
    let delta_uv = delta.xy;
    let delta_z = delta.z;
    
    let dim = textureDimensions(depth_texture);
    let screen_delta_length = length(delta_uv * vec2<f32>(f32(dim.x), f32(dim.y)));
    let screen_factor = max(1.0, screen_delta_length * 0.5);

    let jitter = fast_hash(pixel_uv * vec2<f32>(f32(dim.x), f32(dim.y)));
    var uv = start_screen.xy + delta_uv * jitter;
    var ray_z = start_screen.z + delta_z * jitter;
    
    let max_mip = get_max_mip_level();
    var current_mip = min(3.0, max_mip);

    let thickness_scale = 0.00005 * screen_factor; 
    let thickness_offset = (thickness_base * 0.001) * screen_factor;
    let start_cam_dist = distance(cam_pos, start_pos);
    let world_step = max_dist / f32(max_steps);
    
    for (var i = 0u; i < max_steps; i++) {
        uv += delta_uv;
        ray_z += delta_z;
        
        let outside = uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
        if (outside) { break; }
        
        let hzb_depth = textureSampleLevel(hzb_texture, hzb_sampler, uv, current_mip).r;
        
        if (ray_z > hzb_depth) {
            if (current_mip <= 0.5) {
                let dist_to_ray = start_cam_dist + f32(i) * world_step;
                if (ray_z - hzb_depth < (thickness_offset + dist_to_ray * thickness_scale)) { 
                    // Binary search (inlined and simplified)
                    var r_uv = uv - delta_uv;
                    var r_z = ray_z - delta_z;
                    var r_duv = delta_uv;
                    var r_dz = delta_z;
                    for (var j = 0u; j < 4u; j++) {
                        r_duv *= 0.5; r_dz *= 0.5;
                        r_uv += r_duv; r_z += r_dz;
                        if (r_z > textureSampleLevel(hzb_texture, hzb_sampler, r_uv, 0.0).r) {
                            r_uv -= r_duv; r_z -= r_dz;
                        }
                    }
                    let edge_fade = min(min(r_uv.x, 1.0 - r_uv.x), min(r_uv.y, 1.0 - r_uv.y));
                    return vec3<f32>(r_uv, smoothstep(0.0, 0.1, edge_fade));
                }
            } else {
                current_mip = max(0.0, current_mip - 1.0);
                uv -= delta_uv;
                ray_z -= delta_z;
            }
        } else {
            current_mip = min(3.0, current_mip + 1.0);
        }
    }
    return vec3<f32>(0.0);
}

fn screen_space_factor_calc(screen_delta_length: f32) -> f32 {
    return max(1.0, screen_delta_length * 0.5);
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
    
    // Animated water normal + optional boat wake perturbation
    let base_water_normal = get_water_normal(hit_pos, time);
    var water_normal_full = base_water_normal;
    if (water.boat_pos_wake.w > 0.0001) {
        water_normal_full = get_boat_wake_normal(hit_pos, time, base_water_normal);
    }
    let view_dir = -world_dir;
    
    // Distance from camera to water
    let dist = distance(cam_pos, hit_pos);
    
    // Reduce water normal perturbation for distant water to prevent elongated reflections from becoming noisy
    let normal_fade_start = 100.0;
    let normal_fade_end = 400.0;
    let normal_fade = 1.0 - smoothstep(normal_fade_start, normal_fade_end, dist);
    let water_normal = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), water_normal_full, normal_fade));
    
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
    let ssr_hit = trace_water_reflection(hit_pos, reflect_dir_raw, cam_pos, in.uv);
    let ssr_hit_valid = ssr_hit.z;
    var ssr_color = vec3<f32>(0.0);
    var ssr_effect = 0.0;
    var reflection_distance = 0.0;
    var sample_center = vec2<f32>(0.0);

    if (ssr_hit_valid > 0.0) {
        let scene_sample = textureSample(scene_color_texture, scene_sampler, ssr_hit.xy);
        let ssr_add = textureSample(ssr_texture, scene_sampler, ssr_hit.xy).rgb;
        ssr_color = scene_sample.rgb + ssr_add;
        sample_center = ssr_hit.xy;

        // Use the G-buffer at the hit point:
        // - A provides a stable view-space distance (view_z)
        // - RGB provides world normal for a cheap hit validity check (reduces silhouette/false hits)
        let hit_gbuf = textureSample(normal_gbuffer, scene_sampler, ssr_hit.xy);
        reflection_distance = max(hit_gbuf.a, 0.0);
        let hit_normal = decode_world_normal(hit_gbuf.rgb);
        // Ray marches from water toward the scene along reflect_dir_raw.
        // For a plausible hit, the surface should face the incoming ray somewhat.
        let facing = dot(hit_normal, -reflect_dir_raw);
        let facing_factor = smoothstep(0.0, 0.2, facing);

        let ssr_max_dist = 1000.0;
        let ssr_dist_fade = clamp((ssr_max_dist - dist) / ssr_max_dist, 0.0, 1.0);
        ssr_effect = ssr_hit_valid * ssr_dist_fade * facing_factor;
    } else {
        // Cheap fallback for missing SSR hits: project a far point along the reflection ray,
        // sample the scene color there and use it as an approximate fill for distant reflections.
        let fallback_dist = 800.0;
        let fallback_point = hit_pos + reflect_dir_raw * fallback_dist;
        let fallback_scr = world_to_screen_uv(fallback_point);
        if (fallback_scr.x >= 0.0 && fallback_scr.x <= 1.0 && fallback_scr.y >= 0.0 && fallback_scr.y <= 1.0) {
            let fallback_coords = vec2<i32>(vec2<f32>(dim) * fallback_scr.xy);
            if (fallback_coords.x >= 0 && fallback_coords.x < i32(dim.x) && fallback_coords.y >= 0 && fallback_coords.y < i32(dim.y)) {
                let fallback_depth = textureLoad(depth_texture, fallback_coords, 0);
                if (fallback_depth < 0.9999) {
                    let fallback_sample = textureSample(scene_color_texture, scene_sampler, fallback_scr.xy);
                    let fallback_ssr = textureSample(ssr_texture, scene_sampler, fallback_scr.xy).rgb;
                    ssr_color = fallback_sample.rgb + fallback_ssr;
                    sample_center = fallback_scr.xy;
                    // Approximate distance to the reflected point using the fallback projection
                    reflection_distance = distance(cam_pos, fallback_point);
                    // Give it a stronger but still modest effect so it doesn't overpower skybox when inaccurate
                    ssr_effect = 0.8 * clamp((fallback_dist - dist) / fallback_dist, 0.0, 1.0);
                }
            }
        }
    }

    // Distance-based 3x3 gather to simulate out-of-focus blur for distant reflections
    // Relaxed trigger and more aggressive defaults so the effect is visible when needed.
    if (ssr_effect > 0.01 && (sample_center.x != 0.0 || sample_center.y != 0.0)) {
        // Use DoF parameters (focal_range/2 -> focal_distance*2) to determine gather range
        let gather_start = max(1.0, water.dof_focal_range * 0.5);
        let gather_end = max(gather_start + 1.0, water.dof_focal_distance * 2.0);
        // Linear ramp between start and end (clamped)
        let gather_strength = clamp((reflection_distance - gather_start) / (gather_end - gather_start), 0.0, 1.0);
        if (gather_strength > 0.0001) {
            let texel = vec2<f32>(1.0 / f32(dim.x), 1.0 / f32(dim.y));
            // Increase spread multiplier for a softer blur on distant reflections
            let spread = 1.0 + gather_strength * 4.0;
            var accum = vec3<f32>(0.0);
            var valid_count = 0.0;
            for (var oy = -1; oy <= 1; oy = oy + 1) {
                for (var ox = -1; ox <= 1; ox = ox + 1) {
                    let offset_uv = sample_center + vec2<f32>(f32(ox), f32(oy)) * texel * spread;
                    if (offset_uv.x < 0.0 || offset_uv.x > 1.0 || offset_uv.y < 0.0 || offset_uv.y > 1.0) {
                        continue;
                    }
                    let s_coords = vec2<i32>(vec2<f32>(dim) * offset_uv);
                    if (s_coords.x < 0 || s_coords.x >= i32(dim.x) || s_coords.y < 0 || s_coords.y >= i32(dim.y)) {
                        continue;
                    }
                    let s_depth = textureLoad(depth_texture, s_coords, 0);
                    if (s_depth >= 0.9999) { continue; }
                    let s_sample = textureSample(scene_color_texture, scene_sampler, offset_uv);
                    let s_ssr = textureSample(ssr_texture, scene_sampler, offset_uv).rgb;
                    accum += s_sample.rgb + s_ssr;
                    valid_count += 1.0;
                }
            }
            if (valid_count > 0.0) {
                let gathered = accum / valid_count;
                ssr_color = mix(ssr_color, gathered, gather_strength);
            }
        }
    }

    // LUMINANCE-DRIVEN BOOST FOR EMISSIVE SAMPLES
    let ssr_lum = dot(ssr_color, vec3<f32>(0.299, 0.587, 0.114));
    let lum_thresh = 0.3; // threshold for considering a sample emissive
    let lum_ramp = smoothstep(lum_thresh, lum_thresh * 2.0, ssr_lum);
    let ssr_boost = 1.5;
    ssr_color = mix(ssr_color, ssr_color * ssr_boost, lum_ramp);

    reflection_color = mix(reflection_color, ssr_color, ssr_effect);

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
    var boat_wake = 0.0;
    if (water.boat_pos_wake.w > 0.0001) {
        boat_wake = get_boat_wake(hit_pos, time);
    }
    let total_foam = clamp(shore_foam + surface_foam + boat_wake, 0.0, 1.0);
    
    // Foam color darkens with scene brightness
    let foam_color = vec3<f32>(0.9, 0.92, 0.95) * brightness;
    
    // ========================================================================
    // FINAL COMPOSITION
    // ========================================================================
    
    // Mix water color with reflection based on Fresnel
    var final_color = mix(water_color, reflection_color, reflection_strength);
    
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
