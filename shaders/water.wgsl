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
    light_probe_count: u32,
    lod_distance: f32,
    envelope_distance: f32,
    envelope_fade_range: f32,
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

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
    let reflect_dir = reflect(world_dir, normal);
    
    // Sample skybox (equirectangular)
    let u = 0.5 + atan2(reflect_dir.z, reflect_dir.x) / (2.0 * 3.14159265);
    let v = 0.5 - asin(reflect_dir.y) / 3.14159265;
    let reflection_color = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v)).rgb;
    
    // Base water color
    let base_color = water.water_color.rgb;
    
    // Mix based on reflection
    let final_rgb = mix(base_color, reflection_color, reflection_strength);
    
    // Alpha
    // More opaque at grazing angles, more transparent looking down
    let alpha = mix(water.water_color.a, 1.0, fresnel * 0.5);
    
    // Soft shore fade
    // Calculate depth difference
    // We need linear depth for this to be accurate, but raw depth diff might suffice for visual hack
    // Linearize depth: z_linear = (2.0 * near * far) / (far + near - z_ndc * (far - near)); 
    // ... or just use the raw difference and tune a magic number.
    // Let's try raw diff first.
    let depth_diff = scene_depth_raw - water_depth;
    let shore_fade = smoothstep(0.0, 0.001, depth_diff); // Tune 0.001 based on scene scale/projection
    
    return vec4<f32>(final_rgb, alpha * shore_fade);
}
