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
    // New dual-light additions (moon has no shadow map; intensity in w)
    moon_direction_intensity: vec4<f32>,
    moon_color_pad: vec4<f32>,
    light_probe_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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


struct VertexOutputInstanced {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) emissive: vec4<f32>,
    @location(3) light_space_pos: vec4<f32>,
    @location(4) world_pos: vec3<f32>,
}

struct VertexOutputMesh {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) emissive: vec4<f32>,
    @location(3) light_space_pos: vec4<f32>,
    @location(4) world_pos: vec3<f32>,
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
    switch (voxel_type) {
        case 1u: {
            return vec3<f32>(0.1, 0.9, 0.3); // Neon grass highlights
        }
        case 2u: {
            return vec3<f32>(1.0, 0.35, 0.35); // Sunlit red concrete
        }
        case 3u: {
            return vec3<f32>(0.35, 0.5, 1.0); // Electric blue panels
        }
        case 4u: {
            return vec3<f32>(0.95, 0.9, 0.35); // Warm accent lighting
        }
        case 5u: {
            return vec3<f32>(0.95, 0.4, 1.0); // Vibrant magenta glass
        }
        case 6u: {
            return vec3<f32>(0.3, 0.95, 1.0); // Cyan signage glow
        }
        case 7u: {
            return vec3<f32>(0.85, 0.85, 0.85); // Bright concrete walls
        }
        default: {
            return vec3<f32>(1.0, 1.0, 1.0); // White default
        }
    }
}

@vertex
fn vs_main(
    @location(0) instance_position: vec3<f32>,
    @location(1) instance_voxel_type: u32,
    @location(2) instance_scale: f32,
    @location(3) instance_custom_color: vec4<f32>,
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
        output.color = instance_custom_color.rgb;
    } else {
        output.color = get_voxel_color(instance_voxel_type);
    }

    output.emissive = instance_emissive;
    
    return output;
}

@fragment
fn fs_main(input: VertexOutputInstanced) -> @location(0) vec4<f32> {
    let sun_dir = normalize(uniforms.sun_direction_shadow_bias.xyz);
    let sun_diffuse = max(dot(input.normal, sun_dir), 0.0);
    let base_shadow = compute_shadow(input.light_space_pos, input.normal, sun_dir);
    let shadow_strength = uniforms.camera_shadow_strength.w;
    let shadow_visibility = mix(1.0, base_shadow, shadow_strength);
    let sun_contribution = sun_diffuse * uniforms.sun_color_pad.xyz * shadow_visibility;
    let ambient = uniforms.ambient_color_pad.xyz;

    // Moon light (no shadows yet) -------------------------------------------------
    let moon_dir = normalize(uniforms.moon_direction_intensity.xyz);
    let moon_diffuse = max(dot(input.normal, moon_dir), 0.0);
    let moon_light = moon_diffuse * uniforms.moon_color_pad.xyz * uniforms.moon_direction_intensity.w;
    
    // Sample light probes for indirect emissive lighting
    var indirect_light = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < uniforms.light_probe_count; i++) {
        let probe = light_probes[i];
        let to_light = probe.position - input.world_pos;
        let dist_sq = dot(to_light, to_light);
        // Very localized lighting: cubic falloff for rapid distance dropoff
        let dist = sqrt(dist_sq);
        let attenuation = (probe.color_power.a * 0.02) / max(dist_sq * dist, 64.0);
        indirect_light += probe.color_power.rgb * attenuation;
    }
    // Keep it subtle - max 10% brightness from emissive lights
    indirect_light = min(indirect_light, vec3<f32>(0.1, 0.1, 0.1));
    
    let lighting = ambient + sun_contribution + moon_light + indirect_light;
    
    // For emissive surfaces, reduce lighting influence to prevent over-brightening
    // Emissive surfaces should show their base color + emission, not lit base color + emission
    let emissive_strength = input.emissive.a;
    let lighting_multiplier = mix(1.0, 0.3, emissive_strength);
    let color = input.color * (lighting * lighting_multiplier);

    let fog_color = vec3<f32>(0.7, 0.8, 0.9);
    let distance = length(input.position.xyz);
    let fog_factor = 1.0 - exp(-uniforms.fog_time_pad.x * distance);
    let fogged_color = mix(color, fog_color, fog_factor * 0.5);
    
    // Add emissive after fog so it stays bright
    let emissive = input.emissive.rgb * emissive_strength;
    let final_color = fogged_color + emissive;

    return vec4<f32>(final_color, 1.0);
}

// Mesh pipeline entry points -------------------------------------------------
@vertex
fn vs_mesh(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) emissive: vec4<f32>,
) -> VertexOutputMesh {
    var out: VertexOutputMesh;
    let world_pos = vec4<f32>(position, 1.0);
    out.position = uniforms.mvp * world_pos;
    out.light_space_pos = uniforms.sun_view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.normal = normal;
    out.color = color.rgb;
    out.emissive = emissive;
    return out;
}

@fragment
fn fs_mesh(input: VertexOutputMesh) -> @location(0) vec4<f32> {
    let sun_dir = normalize(uniforms.sun_direction_shadow_bias.xyz);
    let sun_diffuse = max(dot(input.normal, sun_dir), 0.0);
    let base_shadow = compute_shadow(input.light_space_pos, input.normal, sun_dir);
    let shadow_strength = uniforms.camera_shadow_strength.w;
    let shadow_visibility = mix(1.0, base_shadow, shadow_strength);
    let sun_contribution = sun_diffuse * uniforms.sun_color_pad.xyz * shadow_visibility;
    let ambient = uniforms.ambient_color_pad.xyz;

    // Moon light (no shadows yet)
    let moon_dir = normalize(uniforms.moon_direction_intensity.xyz);
    let moon_diffuse = max(dot(input.normal, moon_dir), 0.0);
    let moon_light = moon_diffuse * uniforms.moon_color_pad.xyz * uniforms.moon_direction_intensity.w;
    
    // Sample light probes for indirect emissive lighting
    var indirect_light = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < uniforms.light_probe_count; i++) {
        let probe = light_probes[i];
        let to_light = probe.position - input.world_pos;
        let dist_sq = dot(to_light, to_light);
        // Very localized lighting: cubic falloff for rapid distance dropoff
        let dist = sqrt(dist_sq);
        let attenuation = (probe.color_power.a * 0.02) / max(dist_sq * dist, 64.0);
        indirect_light += probe.color_power.rgb * attenuation;
    }
    // Keep it subtle - max 10% brightness from emissive lights
    indirect_light = min(indirect_light, vec3<f32>(0.1, 0.1, 0.1));
    
    let lighting = ambient + sun_contribution + moon_light + indirect_light;
    
    // For emissive surfaces, reduce lighting influence to prevent over-brightening
    // Emissive surfaces should show their base color + emission, not lit base color + emission
    let emissive_strength = input.emissive.a;
    let lighting_multiplier = mix(1.0, 0.3, emissive_strength);
    let color = input.color * (lighting * lighting_multiplier);
    
    let fog_color = vec3<f32>(0.7, 0.8, 0.9);
    let distance = length(input.position.xyz);
    let fog_factor = 1.0 - exp(-uniforms.fog_time_pad.x * distance);
    let fogged_color = mix(color, fog_color, fog_factor * 0.5);
    
    // Add emissive after fog so it stays bright
    let emissive = input.emissive.rgb * emissive_strength;
    let final_color = fogged_color + emissive;
    return vec4<f32>(final_color, 1.0);
}

fn compute_shadow(light_space_pos: vec4<f32>, normal: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    if (light_space_pos.w <= 0.0) {
        return 1.0;
    }

    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let uv = vec2<f32>(proj_coords.x * 0.5 + 0.5, 0.5 - proj_coords.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }

    let depth = clamp(proj_coords.z, 0.0, 1.0);
    let base_bias = uniforms.sun_direction_shadow_bias.w;
    let ndotl = max(dot(normal, sun_dir), 0.0);
    // Slope bias: more bias when surface is at grazing angle to light
    let slope_bias = (1.0 - ndotl) * 0.002;
    // Texel bias: prevent self-shadowing artifacts
    let texel_bias = max(uniforms.shadow_texel_size_pad.x, uniforms.shadow_texel_size_pad.y) * 0.5;
    let depth_ref = clamp(depth - (base_bias + slope_bias + texel_bias), 0.0, 1.0);
    
    // 2x2 PCF (Percentage Closer Filtering) for smoother shadow edges
    let texel_size = vec2<f32>(uniforms.shadow_texel_size_pad.x, uniforms.shadow_texel_size_pad.y);
    var shadow = 0.0;
    let offset = texel_size * 0.5;
    shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2(-offset.x, -offset.y), depth_ref);
    shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2(offset.x, -offset.y), depth_ref);
    shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2(-offset.x, offset.y), depth_ref);
    shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2(offset.x, offset.y), depth_ref);
    return shadow * 0.25;
}

@vertex
fn vs_shadow_instanced(
    @location(0) instance_position: vec3<f32>,
    @location(1) _instance_voxel_type: u32,
    @location(2) instance_scale: f32,
    @location(3) _instance_custom_color: vec4<f32>,
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
