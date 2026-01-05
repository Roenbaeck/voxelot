// Voxel rendering shader for hierarchical chunks with global lighting

struct Uniforms {
    mvp: mat4x4<f32>,
    sun_view_proj: mat4x4<f32>,
    camera_shadow_strength: vec4<f32>,
    sun_direction_shadow_bias: vec4<f32>,
    fog_time_pad: vec4<f32>,
    sun_color_pad: vec4<f32>,
    ambient_color_pad: vec4<f32>,
    shadow_texel_size_pad: vec4<f32>,
    shadow_darkness: vec4<f32>,
    // New dual-light additions (moon has no shadow map; intensity in w)
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
}

struct LightProbe {
    position: vec3<f32>,
    _pad0: f32,
    color_power: vec4<f32>,  // RGB color from emissive_sum, A = emissive_power
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<storage, read> light_probes: array<LightProbe>;
@group(0) @binding(4) var<storage, read> palette: array<vec4<f32>>;
// Material properties buffer: [reflectivity, reserved, reserved, reserved] per voxel type
@group(0) @binding(5) var<storage, read> material_props: array<vec4<f32>>;


struct VertexOutputInstanced {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) emissive: vec4<f32>,
    @location(3) light_space_pos: vec4<f32>,
    @location(4) world_pos: vec3<f32>,
    @location(5) ao: f32,
    @location(6) view_z: f32,  // Linear depth for G-buffer
    @location(7) @interpolate(flat) voxel_type: u32,  // For material property lookup
}

struct VertexOutputMesh {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) emissive: vec4<f32>,
    @location(3) light_space_pos: vec4<f32>,
    @location(4) world_pos: vec3<f32>,
    @location(5) view_z: f32,  // Linear depth for G-buffer
    @location(6) material: vec4<f32>,  // R=reflectivity, GBA=reserved
}

struct ShadowVertexOutput {
    @builtin(position) position: vec4<f32>,
}

// Cube vertices (36 vertices for 6 faces)
const CUBE_VERTICES: array<vec3<f32>, 36> = array<vec3<f32>, 36>(
    // Front face
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
    
    // Back face
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    
    // Top face
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0),
    
    // Bottom face
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0),
    
    // Right face
    vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0),
    
    // Left face
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
);

const CUBE_NORMALS: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 0.0, 1.0),   // Front
    vec3<f32>(0.0, 0.0, -1.0),  // Back
    vec3<f32>(0.0, 1.0, 0.0),   // Top
    vec3<f32>(0.0, -1.0, 0.0),  // Bottom
    vec3<f32>(1.0, 0.0, 0.0),   // Right
    vec3<f32>(-1.0, 0.0, 0.0),  // Left
);

fn get_voxel_color(voxel_type: u32) -> vec3<f32> {
    // Look up color from palette buffer (clamped to 0-255 range)
    let idx = min(voxel_type, 255u);
    return palette[idx].rgb;
}

@vertex
fn vs_main(
    @location(0) instance_position: vec3<f32>,
    @location(1) instance_voxel_type: u32,
    @location(2) instance_scale: vec3<f32>,
    @location(3) instance_ao: f32,
    @location(7) instance_custom_color: vec4<f32>,
    @location(4) vertex_position: vec3<f32>,
    @location(5) vertex_normal: vec3<f32>,
    @location(6) instance_emissive: vec4<f32>,
) -> VertexOutputInstanced {
    var output: VertexOutputInstanced;
    
    // Scale the vertex position, then add to instance position
    let scaled_vertex_pos = vertex_position * instance_scale;
    let world_pos = vec4<f32>(instance_position + scaled_vertex_pos, 1.0);
    output.position = uniforms.mvp * world_pos;
    output.light_space_pos = uniforms.sun_view_proj * world_pos;
    output.world_pos = world_pos.xyz;
    
    // Use the per-vertex normal from the buffer
    output.normal = vertex_normal;
    
        // Use custom color if alpha > 0, otherwise use voxel type color
        if (instance_custom_color.a > 0.0) {
            output.color = instance_custom_color;
        } else {
            output.color = vec4<f32>(get_voxel_color(instance_voxel_type), 1.0);
        }
        // AO is passed separately using instance_ao (not embedded in color alpha anymore)
        output.ao = instance_ao;

    output.emissive = instance_emissive;
    // For perspective projection, clip.w = -view_z, so clip.w is positive view distance
    // NOTE: This value (`output.view_z`) is the *view-space distance* (clip.w) and is NOT the
    //       normalized depth in [0..1] that the depth buffer provides. Do NOT pass this value
    //       into functions that expect a normalized depth (e.g., `reconstruct_world_pos(uv, depth)`).
    //       To reconstruct world-space positions use the sampled depth texture (scene_depth)
    //       and then unproject to world space — see other shaders for examples.
    output.view_z = output.position.w;
    // Pass voxel type to fragment shader for material property lookup
    output.voxel_type = instance_voxel_type;
    
    return output;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) material: vec4<f32>,  // R=reflectivity, GBA=reserved
}

@fragment
fn fs_main(input: VertexOutputInstanced) -> FragmentOutput {
    let sun_dir = normalize(uniforms.sun_direction_shadow_bias.xyz);
    let ndotl_raw = dot(input.normal, sun_dir);
    let sun_diffuse = max(ndotl_raw, 0.0);
    let base_shadow = compute_shadow(input.light_space_pos, input.normal, sun_dir);
    let shadow_strength = uniforms.camera_shadow_strength.w;
    let raw_visibility = mix(1.0, base_shadow, shadow_strength);
    let shadow_visibility = clamp(1.0 - (1.0 - raw_visibility) * uniforms.shadow_darkness.x, 0.0, 1.0);
    let emissive_strength = input.emissive.a;
    // Attenuate directional light on emissive surfaces to prevent over-brightness
    // We keep ambient full to avoid "sharpness" artifacts in shadow.
    let dir_light_attenuation = 1.0 - (emissive_strength * 1.0);

    let sun_contribution = sun_diffuse * uniforms.sun_color_pad.xyz * shadow_visibility * dir_light_attenuation;
    var ambient = uniforms.ambient_color_pad.xyz;
    // If normal faces away from sun, reduce ambient for clear back-face darkening.
    // back_strength == 0 for faces facing sun, >0 for faces facing away
    let back_strength = clamp(-ndotl_raw, 0.0, 1.0);
    let back_scale = clamp(uniforms.shadow_darkness.y, 0.0, 1.0); // clamp for safety
    ambient = ambient * mix(vec3<f32>(1.0), vec3<f32>(back_scale), back_strength);

    // Moon light (no shadows yet) -------------------------------------------------
    let moon_dir = normalize(uniforms.moon_direction_intensity.xyz);
    let moon_diffuse = max(dot(input.normal, moon_dir), 0.0);
    let moon_light = moon_diffuse * uniforms.moon_color_pad.xyz * uniforms.moon_direction_intensity.w * dir_light_attenuation;
    
    // Sample light probes for indirect emissive lighting
    var indirect_light = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < uniforms.light_probe_count; i++) {
        let probe = light_probes[i];
        let to_light = probe.position - input.world_pos;
        let dist_sq = dot(to_light, to_light);
        // Very localized lighting: cubic falloff for rapid distance dropoff
        let dist = sqrt(dist_sq);
        let attenuation = (probe.color_power.a * 0.01) / max(dist_sq * dist, 128.0);
        
        // Normalize the probe color to prevent oversaturated color bleeding
        let probe_brightness = max(probe.color_power.r, max(probe.color_power.g, probe.color_power.b));
        let normalized_color = probe.color_power.rgb / max(probe_brightness, 1.0);
        
        indirect_light += normalized_color * attenuation;
    }
    // Keep it subtle but visible for testing (increased from 0.03)
    indirect_light = min(indirect_light, vec3<f32>(0.5, 0.5, 0.5));
    
    let lighting = ambient + sun_contribution + moon_light + indirect_light;
    
    // emissive_strength is already defined above
    let ao = input.ao; // AO passed separately from instance AO attribute
    let color = input.color.rgb * lighting * ao;

    // Fog color modulated by ambient and sky brightness (darker at night)
    let base_fog_color = vec3<f32>(0.7, 0.8, 0.9);
    let skybox_brightness = uniforms.fog_time_pad.w;
    // Mix between a very dark night fog and the ambient color scaled by `skybox_brightness`.
    // This prevents the fog from becoming brighter than the scene when the sky is bright
    // near the horizon during dawn/dusk.
    let fog_base = mix(vec3<f32>(0.02, 0.02, 0.03), uniforms.ambient_color_pad.xyz, skybox_brightness);
    let fog_color = base_fog_color * fog_base * 2.0;
    // Use world-space distance from camera (input.world_pos contains world-space position)
    // uniforms.camera_shadow_strength.xyz stores camera world position (see Rust binding comment)
    let relative_pos = input.world_pos - uniforms.camera_shadow_strength.xyz;
    let distance = length(relative_pos);
    let transmittance = exp(-uniforms.fog_time_pad.x * distance);
    let fog_factor = 1.0 - transmittance;
    // Add directional volumetric scattering from sun so the brightening only occurs
    // when looking toward the sun, and not globally. This prevents distant objects on
    // the horizon from being unnaturally lit when the sun is near the horizon.
    let view_dir = normalize(uniforms.camera_shadow_strength.xyz - input.world_pos);
    let sun_dir_local = normalize(uniforms.sun_direction_shadow_bias.xyz);
    let sun_view_dot = max(dot(view_dir, -sun_dir_local), 0.0);
    let inscatter = uniforms.sun_color_pad.xyz * 0.15 * fog_factor * sun_view_dot;
    let fogged_color = mix(color, fog_color + inscatter, fog_factor);
    
    // Add emissive after fog so it stays bright
    let emissive_rgb = input.emissive.rgb * emissive_strength;
    let final_color = fogged_color + emissive_rgb;

    // Distance-based alpha fade to hide pop-in at far distances
    // Start fading at 80% of LOD distance, fully transparent at 95%
    let fade_start = uniforms.lod_distance * 0.80;
    let fade_end = uniforms.lod_distance * 0.95;
    let fade_factor = smoothstep(fade_start, fade_end, distance);
    
    // NOTE: Dither-based discard removed. The stipple pattern was visible in SSR and
    // water reflections as "black noise". We now rely solely on alpha-based fading
    // which is smoother but may show slight pop-in at LOD boundaries.
    
    // Brighten colors as they approach fade region for fog-like appearance; reduce
    // the brightening factor to avoid extreme brightening near the horizon.
    var brightened = mix(final_color, fog_color, fade_factor * 0.18);

    // Probe-only cheap sky reflection for reflective materials.
    // This is used for the reflection cubemap capture so reflective buildings look
    // closer to the main-view shading (which adds SSR/sky reflections later).
    let mat_idx = min(input.voxel_type, 255u);
    let reflectivity = material_props[mat_idx].r;
    if (uniforms._water_pad.y > 0.5 && reflectivity > 0.001) {
        let rdir = reflect(-view_dir, normalize(input.normal));
        // Simple procedural sky (varies with elevation + azimuth + horizon glow).
        let t = clamp(rdir.y * 0.5 + 0.5, 0.0, 1.0);
        let PI = 3.14159265359;
        let TWO_PI = 6.28318530718;
        let u = fract(0.5 + atan2(rdir.z, rdir.x) / TWO_PI);
        let night_a = vec3<f32>(0.02, 0.02, 0.03);
        let night_b = vec3<f32>(0.04, 0.05, 0.08);
        let day_a = vec3<f32>(0.45, 0.62, 0.92);
        let day_b = vec3<f32>(0.62, 0.74, 0.96);
        let az_blend = 0.5 + 0.5 * sin(u * TWO_PI);
        let night = mix(night_a, night_b, az_blend);
        let day = mix(day_a, day_b, az_blend);
        var env = mix(night, day, t);
        // Horizon glow makes vertical-face reflections less uniform.
        let horizon = pow(1.0 - abs(rdir.y), 3.0);
        env += vec3<f32>(0.35, 0.30, 0.25) * horizon * uniforms.fog_time_pad.w;
        // Add a small sun highlight in the reflection direction.
        let sun_lobe = pow(max(dot(rdir, -sun_dir_local), 0.0), 64.0);
        env += uniforms.sun_color_pad.xyz * (0.35 * sun_lobe) * uniforms.fog_time_pad.w;
        // Apply night tint similarly to skybox shading.
        let tint = uniforms.skybox_tint_pad.xyz;
        let tint_strength = uniforms.skybox_tint_pad.w;
        let brightness = uniforms.fog_time_pad.w;
        let effect_strength = (1.0 - brightness) * tint_strength;
        env = mix(env, env * tint, effect_strength);
        let env_strength = clamp(reflectivity * 0.75, 0.0, 0.9);
        brightened = mix(brightened, env, env_strength);
    }

    // Envelope fade: if we are approaching the envelope distance, fade towards the envelope color (Type 0)
    // This helps blend the detailed mesh into the simplified envelope mesh.
    let env_dist = uniforms.envelope_distance;
    let env_fade_range = uniforms.envelope_fade_range;
    // Start fading before the envelope distance
    let env_fade_start = env_dist - env_fade_range;
    let env_fade_factor = smoothstep(env_fade_start, env_dist, distance);
    
    if (env_fade_factor > 0.0) {
        // Get color of Type 0 (envelope color)
        // We assume Type 0 is used for envelopes as per design
        let env_color_base = get_voxel_color(0u);
        // Apply lighting to envelope color so it matches the scene
        let env_lit = env_color_base * lighting;
        let env_fogged = mix(env_lit, fog_color + inscatter, fog_factor);
        
        brightened = mix(brightened, env_fogged, env_fade_factor);
    }
    
    // Underwater depth fade: fade geometry based on depth below water surface
    // Similar to distance fade, but based on y-coordinate depth from water_level (Y is up)
    let water_level = uniforms.water_level;
    let water_vis = uniforms.water_visibility;
    let underwater_depth = water_level - input.world_pos.y;
    var underwater_alpha = 1.0;
    
    // Only apply underwater fade if the geometry is actually underwater
    if (underwater_depth > 0.0 && water_vis > 0.0) {
        // Calculate fade factor: 0 at surface, 1 at max visibility depth
        let underwater_fade = smoothstep(0.0, water_vis, underwater_depth);
        
        // Alpha fade using Beer-Lambert exponential decay (how light behaves in water)
        // exp(-k*d) where k controls absorption rate, d is depth
        // Very aggressive absorption to limit visibility depth
        let absorption_coefficient = 12.0 / max(water_vis, 1.0);
        underwater_alpha = exp(-absorption_coefficient * underwater_depth);
        
        // Force objects beyond half visibility to be nearly invisible
        if (underwater_depth > water_vis * 0.5) {
            underwater_alpha = underwater_alpha * 0.1;
        }
        
        // Darken based on depth to simulate light absorption
        let depth_darkness = exp(-3.5 * underwater_depth / max(water_vis, 1.0));
        brightened = brightened * depth_darkness;
        
        // Fade to water color (deep water color) as objects get deeper
        // This works even without alpha blending by making objects blend into water
        let deep_water_color = vec3<f32>(0.02, 0.12, 0.20) * ambient;
        // More aggressive fade curve - objects become water color quickly
        let color_fade = pow(underwater_fade, 0.6); // Power < 1 makes fade faster
        brightened = mix(brightened, deep_water_color, color_fade);
    }
    
    // Add gradual alpha fading for smoother transitions
    // Alpha fades 60-95% while dithering operates 80-95% on semi-transparent fragments
    let alpha_fade_start = uniforms.lod_distance * 0.60;
    let alpha_fade_end = uniforms.lod_distance * 0.95;
    let lod_alpha = 1.0 - smoothstep(alpha_fade_start, alpha_fade_end, distance);
    
    // Apply both LOD fade and underwater fade (multiplicative)
    let alpha = lod_alpha * underwater_alpha * input.color.a;

    var out: FragmentOutput;
    out.color = vec4<f32>(brightened, alpha);
    
    // Scale emissive by strength and apply fades so it doesn't pop in/out
    let final_emissive = input.emissive.rgb * input.emissive.a * (1.0 - env_fade_factor) * alpha;
    out.emissive = vec4<f32>(final_emissive, input.emissive.a);
    // Encode world-space normal: map [-1,1] to [0,1] for storage
    // Store linear depth in W channel for SSILVB/GTAO
    out.normal = vec4<f32>(input.normal * 0.5 + 0.5, input.view_z);
    // Look up material reflectivity from material_props buffer
    out.material = vec4<f32>(reflectivity, 0.0, 0.0, 0.0);
    return out;
}

// Mesh pipeline entry points -------------------------------------------------
@vertex
fn vs_mesh(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) emissive: vec4<f32>,
    @location(4) material: vec4<f32>,
) -> VertexOutputMesh {
    var out: VertexOutputMesh;
    let world_pos = vec4<f32>(position, 1.0);
    out.position = uniforms.mvp * world_pos;
    out.light_space_pos = uniforms.sun_view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.normal = normal;
    out.color = color;
    out.emissive = emissive;
    out.material = material;
    // For perspective projection, clip.w = -view_z, so clip.w is positive view distance
    out.view_z = out.position.w;
    return out;
}

@fragment
fn fs_mesh(input: VertexOutputMesh) -> FragmentOutput {
    let sun_dir = normalize(uniforms.sun_direction_shadow_bias.xyz);
    let sun_diffuse = max(dot(input.normal, sun_dir), 0.0);
    let base_shadow = compute_shadow(input.light_space_pos, input.normal, sun_dir);
    let shadow_strength = uniforms.camera_shadow_strength.w;
    let raw_visibility = mix(1.0, base_shadow, shadow_strength);
    let shadow_visibility = clamp(1.0 - (1.0 - raw_visibility) * uniforms.shadow_darkness.x, 0.0, 1.0);
    let emissive_strength = input.emissive.a;
    // Attenuate directional light on emissive surfaces to prevent over-brightness
    let dir_light_attenuation = 1.0 - (emissive_strength * 1.0);

    let sun_contribution = sun_diffuse * uniforms.sun_color_pad.xyz * shadow_visibility * dir_light_attenuation;
    var ambient = uniforms.ambient_color_pad.xyz;
    // If normal faces away from sun, reduce ambient for back-face darkening
    let ndotl_raw = dot(input.normal, sun_dir);
    let back_strength = clamp(-ndotl_raw, 0.0, 1.0);
    let back_scale = clamp(uniforms.shadow_darkness.y, 0.0, 1.0);
    ambient = ambient * mix(vec3<f32>(1.0), vec3<f32>(back_scale), back_strength);

    // Moon light (no shadows yet)
    let moon_dir = normalize(uniforms.moon_direction_intensity.xyz);
    let moon_diffuse = max(dot(input.normal, moon_dir), 0.0);
    let moon_light = moon_diffuse * uniforms.moon_color_pad.xyz * uniforms.moon_direction_intensity.w * dir_light_attenuation;
    
    // Sample light probes for indirect emissive lighting
    var indirect_light = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < uniforms.light_probe_count; i++) {
        let probe = light_probes[i];
        let to_light = probe.position - input.world_pos;
        let dist_sq = dot(to_light, to_light);
        // Very localized lighting: cubic falloff for rapid distance dropoff
        let dist = sqrt(dist_sq);
        let attenuation = (probe.color_power.a * 0.01) / max(dist_sq * dist, 128.0);
        
        // Normalize the probe color to prevent oversaturated color bleeding
        let probe_brightness = max(probe.color_power.r, max(probe.color_power.g, probe.color_power.b));
        let normalized_color = probe.color_power.rgb / max(probe_brightness, 1.0);
        
        indirect_light += normalized_color * attenuation;
    }
    // Keep it subtle but visible for testing (increased from 0.03)
    indirect_light = min(indirect_light, vec3<f32>(0.5, 0.5, 0.5));
    
    let lighting = ambient + sun_contribution + moon_light + indirect_light;
    
    // emissive_strength is already defined above
    let color = input.color.rgb * lighting * input.color.a;
    
    // Fog color modulated by ambient and sky brightness (darker at night)
    let base_fog_color = vec3<f32>(0.7, 0.8, 0.9);
    let skybox_brightness = uniforms.fog_time_pad.w;
    let fog_base = mix(vec3<f32>(0.02, 0.02, 0.03), uniforms.ambient_color_pad.xyz, skybox_brightness);
    let fog_color = base_fog_color * fog_base * 2.0;
    // Use world-space distance from camera for mesh pipeline as well
    let relative_pos = input.world_pos - uniforms.camera_shadow_strength.xyz;
    let distance = length(relative_pos);
    let transmittance = exp(-uniforms.fog_time_pad.x * distance);
    let fog_factor = 1.0 - transmittance;
    // Add directional volumetric scattering from sun (towards sun only)
    let view_dir = normalize(uniforms.camera_shadow_strength.xyz - input.world_pos);
    let sun_dir_local = normalize(uniforms.sun_direction_shadow_bias.xyz);
    let sun_view_dot = max(dot(view_dir, -sun_dir_local), 0.0);
    let inscatter = uniforms.sun_color_pad.xyz * 0.15 * fog_factor * sun_view_dot;
    let fogged_color = mix(color, fog_color + inscatter, fog_factor);
    
    // Add emissive after fog so it stays bright
    let emissive_rgb = input.emissive.rgb * emissive_strength;
    let final_color = fogged_color + emissive_rgb;
    
    // Distance-based alpha fade to hide pop-in at far distances
    // Start fading at 80% of LOD distance, fully transparent at 95%
    let fade_start = uniforms.lod_distance * 0.80;
    let fade_end = uniforms.lod_distance * 0.95;
    let fade_factor = smoothstep(fade_start, fade_end, distance);
    
    // NOTE: Dither-based discard removed. The stipple pattern was visible in SSR and
    // water reflections as "black noise". We now rely solely on alpha-based fading
    // which is smoother but may show slight pop-in at LOD boundaries.
    
    // Brighten colors as they approach fade region for fog-like appearance
    var brightened = mix(final_color, fog_color, fade_factor * 0.18);

    // Probe-only cheap sky reflection for reflective materials.
    let reflectivity = input.material.r;
    if (uniforms._water_pad.y > 0.5 && reflectivity > 0.001) {
        let rdir = reflect(-view_dir, normalize(input.normal));
        let t = clamp(rdir.y * 0.5 + 0.5, 0.0, 1.0);
        let PI = 3.14159265359;
        let TWO_PI = 6.28318530718;
        let u = fract(0.5 + atan2(rdir.z, rdir.x) / TWO_PI);
        let night_a = vec3<f32>(0.02, 0.02, 0.03);
        let night_b = vec3<f32>(0.04, 0.05, 0.08);
        let day_a = vec3<f32>(0.45, 0.62, 0.92);
        let day_b = vec3<f32>(0.62, 0.74, 0.96);
        let az_blend = 0.5 + 0.5 * sin(u * TWO_PI);
        let night = mix(night_a, night_b, az_blend);
        let day = mix(day_a, day_b, az_blend);
        var env = mix(night, day, t);
        let horizon = pow(1.0 - abs(rdir.y), 3.0);
        env += vec3<f32>(0.35, 0.30, 0.25) * horizon * uniforms.fog_time_pad.w;
        let sun_lobe = pow(max(dot(rdir, -sun_dir_local), 0.0), 64.0);
        env += uniforms.sun_color_pad.xyz * (0.35 * sun_lobe) * uniforms.fog_time_pad.w;
        let tint = uniforms.skybox_tint_pad.xyz;
        let tint_strength = uniforms.skybox_tint_pad.w;
        let brightness = uniforms.fog_time_pad.w;
        let effect_strength = (1.0 - brightness) * tint_strength;
        env = mix(env, env * tint, effect_strength);
        let env_strength = clamp(reflectivity * 0.75, 0.0, 0.9);
        brightened = mix(brightened, env, env_strength);
    }

    // Envelope fade: if we are approaching the envelope distance, fade towards the envelope color (Type 0)
    // This helps blend the detailed mesh into the simplified envelope mesh.
    let env_dist = uniforms.envelope_distance;
    let env_fade_range = uniforms.envelope_fade_range;
    // Start fading before the envelope distance
    let env_fade_start = env_dist - env_fade_range;
    let env_fade_factor = smoothstep(env_fade_start, env_dist, distance);
    
    if (env_fade_factor > 0.0) {
        // Get color of Type 0 (envelope color)
        // We assume Type 0 is used for envelopes as per design
        let env_color_base = get_voxel_color(0u);
        // Apply lighting to envelope color so it matches the scene
        let env_lit = env_color_base * lighting;
        let env_fogged = mix(env_lit, fog_color + inscatter, fog_factor);
        
        brightened = mix(brightened, env_fogged, env_fade_factor);
    }
    
    // Underwater depth fade: fade geometry based on depth below water surface
    // Similar to distance fade, but based on y-coordinate depth from water_level (Y is up)
    let water_level_mesh = uniforms.water_level;
    let water_vis_mesh = uniforms.water_visibility;
    let underwater_depth_mesh = water_level_mesh - input.world_pos.y;
    var underwater_alpha_mesh = 1.0;
    
    // Only apply underwater fade if the geometry is actually underwater
    if (underwater_depth_mesh > 0.0 && water_vis_mesh > 0.0) {
        // Calculate fade factor: 0 at surface, 1 at max visibility depth
        let underwater_fade_mesh = smoothstep(0.0, water_vis_mesh, underwater_depth_mesh);
        
        // Alpha fade using Beer-Lambert exponential decay (how light behaves in water)
        // exp(-k*d) where k controls absorption rate, d is depth
        // Very aggressive absorption to limit visibility depth
        let absorption_coefficient_mesh = 12.0 / max(water_vis_mesh, 1.0);
        underwater_alpha_mesh = exp(-absorption_coefficient_mesh * underwater_depth_mesh);
        
        // Force objects beyond half visibility to be nearly invisible
        if (underwater_depth_mesh > water_vis_mesh * 0.5) {
            underwater_alpha_mesh = underwater_alpha_mesh * 0.1;
        }
        
        // Darken based on depth to simulate light absorption
        let depth_darkness_mesh = exp(-3.5 * underwater_depth_mesh / max(water_vis_mesh, 1.0));
        brightened = brightened * depth_darkness_mesh;
        
        // Fade to water color (deep water color) as objects get deeper
        // This works even without alpha blending by making objects blend into water
        let deep_water_color_mesh = vec3<f32>(0.02, 0.12, 0.20) * ambient;
        // More aggressive fade curve - objects become water color quickly
        let color_fade_mesh = pow(underwater_fade_mesh, 0.6); // Power < 1 makes fade faster
        brightened = mix(brightened, deep_water_color_mesh, color_fade_mesh);
    }
    
    // Add gradual alpha fading for smoother transitions
    // Alpha fades 60-95% while dithering operates 80-95% on semi-transparent fragments
    let alpha_fade_start = uniforms.lod_distance * 0.60;
    let alpha_fade_end = uniforms.lod_distance * 0.95;
    let lod_alpha_mesh = 1.0 - smoothstep(alpha_fade_start, alpha_fade_end, distance);
    
    // Apply both LOD fade and underwater fade (multiplicative)
    let alpha = lod_alpha_mesh * underwater_alpha_mesh * input.color.a;
    
    var out: FragmentOutput;
    out.color = vec4<f32>(brightened, alpha);
    
    // Scale emissive by strength and apply fades so it doesn't pop in/out
    let final_emissive = input.emissive.rgb * input.emissive.a * (1.0 - env_fade_factor) * alpha;
    out.emissive = vec4<f32>(final_emissive, input.emissive.a);
    // Encode world-space normal: map [-1,1] to [0,1] for storage
    // Store linear depth in W channel for SSILVB/GTAO
    out.normal = vec4<f32>(input.normal * 0.5 + 0.5, input.view_z);
    // Use material reflectivity from vertex attribute
    out.material = input.material;
    return out;
}

fn compute_shadow(light_space_pos: vec4<f32>, normal: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    if (light_space_pos.w <= 0.0) {
        return 1.0;
    }

    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let uv = vec2<f32>(proj_coords.x * 0.5 + 0.5, 0.5 - proj_coords.y * 0.5);
    // If the projection falls outside the shadow map we still want a sensible
    // fallback rather than returning fully unshadowed (1.0) which creates a
    // visible horizontal 'cutoff' as the light or camera moves. Instead, clamp
    // the UV to the shadow map edge and continue sampling; this produces a
    // smoother transition at the shadow map boundary and avoids the hard
    // brightness line seen at certain times of day.
    let uv_clamped = clamp(uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let outside = (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0);

    let depth = clamp(proj_coords.z, 0.0, 1.0);
    let base_bias = uniforms.sun_direction_shadow_bias.w;
    let ndotl = max(dot(normal, sun_dir), 0.0);
    // Slope bias: more bias when surface is at grazing angle to light
    let slope_bias = (1.0 - ndotl) * 0.002;
    // Texel bias: prevent self-shadowing artifacts
    let texel_bias = max(uniforms.shadow_texel_size_pad.x, uniforms.shadow_texel_size_pad.y) * 0.5;
    let depth_ref = clamp(depth - (base_bias + slope_bias + texel_bias), 0.0, 1.0);
    
    // Improved PCF: 3x3 Gaussian kernel for smoother shadow edges.
    // radius in texels is provided in shadow_texel_size_pad.z
    let texel_size = vec2<f32>(uniforms.shadow_texel_size_pad.x, uniforms.shadow_texel_size_pad.y);
    let radius = uniforms.shadow_texel_size_pad.z;

    // If Poisson is enabled (pcf_poisson_samples > 0), use a Poisson disk sampling
    // pattern with a stable rotation per fragment to reduce banding/hard halo.
    // Otherwise, use the Gaussian 3x3 as before.
    let poisson_samples = i32(uniforms.shadow_texel_size_pad.w);
    if (poisson_samples > 0) {
        // Poisson disk offsets (8 samples)
        let poisson: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
            vec2<f32>(-0.94201624, -0.39906216),
            vec2<f32>(0.94558609, -0.76890725),
            vec2<f32>(-0.094184101, -0.92938870),
            vec2<f32>(0.34495938, 0.29387760),
            vec2<f32>(-0.91588581, 0.45771432),
            vec2<f32>(-0.81544232, -0.87912464),
            vec2<f32>(-0.38277543, 0.27676845),
            vec2<f32>(0.97484398, 0.75648379),
        );

        // Stable rotation derived from UV to decorrelate samples between fragments.
        let rnd = fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        let angle = rnd * 6.28318530718; // 2*pi
        let c = cos(angle);
        let s = sin(angle);
        let rot = mat2x2<f32>(vec2<f32>(c, -s), vec2<f32>(s, c));

        var shadow_val = 0.0;
        var count = 0;
        let max_samples = min(poisson_samples, 8);
        for (var i: i32 = 0; i < max_samples; i = i + 1) {
            let base_off = poisson[i];
            let roff = rot * base_off;
            let off = roff * texel_size * radius;
            // Clamp sampling coordinates to avoid reading outside the shadow map
            // when the fragment projects outside the shadow map. This helps
            // remove the harsh transition between clamped/unclamped sampling.
            shadow_val += textureSampleCompare(shadow_map, shadow_sampler, uv_clamped + off, depth_ref);
            count = count + 1;
        }
        return shadow_val / f32(count);
    } else {
        // 3x3 Gaussian weights (sum = 16)
        let weights = array<f32, 9>(1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0);
        var shadow = 0.0;
        var idx: i32 = 0;
        for (var y: i32 = -1; y <= 1; y = y + 1) {
            for (var x: i32 = -1; x <= 1; x = x + 1) {
                let off = vec2<f32>(f32(x), f32(y)) * texel_size * radius;
                shadow += weights[idx] * textureSampleCompare(shadow_map, shadow_sampler, uv_clamped + off, depth_ref);
                idx = idx + 1;
            }
        }
        return shadow / 16.0;
    }
}

@vertex
fn vs_shadow_instanced(
    @location(0) instance_position: vec3<f32>,
    @location(1) _instance_voxel_type: u32,
    @location(2) instance_scale: vec3<f32>,
    @location(3) _instance_ao: f32,
    @location(7) _instance_custom_color: vec4<f32>,
    @location(4) vertex_position: vec3<f32>,
    @location(5) _vertex_normal: vec3<f32>,
    @location(6) _instance_emissive: vec4<f32>,
) -> ShadowVertexOutput {
    var output: ShadowVertexOutput;
    let scaled_vertex_pos = vertex_position * instance_scale;
    let world_pos = vec4<f32>(instance_position + scaled_vertex_pos, 1.0);
    output.position = uniforms.sun_view_proj * world_pos;
    return output;
}

@vertex
fn vs_shadow_mesh(
    @location(0) position: vec3<f32>,
    @location(1) _normal: vec3<f32>,
    @location(2) _color: vec4<f32>,
    @location(3) _emissive: vec4<f32>,
) -> ShadowVertexOutput {
    var output: ShadowVertexOutput;
    output.position = uniforms.sun_view_proj * vec4<f32>(position, 1.0);
    return output;
}
