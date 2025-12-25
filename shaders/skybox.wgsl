// This matches the Uniforms struct in voxelot.rs
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

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var skybox_texture: texture_2d<f32>;
@group(1) @binding(1)
var skybox_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) rotated_dir: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Full screen triangle
    let uv = vec2<f32>(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u));
    out.clip_position = vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0); // z = 1.0 (far plane)
    out.uv = uv;

    // Calculate ray direction from camera
    // We want the direction corresponding to the pixel on the far plane
    
    // Convert UV to NDC
    let ndc = vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0);
    
    // Unproject to world space
    // We only care about direction, so we can ignore translation part of view matrix
    // But since we have inverse_view and inverse_proj, we can use them directly.
    
    // Note: camera.inverse_proj transforms from NDC to View space
    // camera.inverse_view transforms from View space to World space
    
    let view_space_pos = camera.inverse_proj * ndc;
    let view_space_dir = view_space_pos.xyz / view_space_pos.w;
    
    // We want direction, so set w=0 for view matrix transform (ignore translation)
    let world_dir = (camera.inverse_view * vec4<f32>(view_space_dir, 0.0)).xyz;
    
    // Apply skybox rotation (around Y axis)
    let angle = camera.fog_time_pad.z;
    let c = cos(angle);
    let s = sin(angle);
    // Rotation matrix around Y:
    // [ c  0  s ]
    // [ 0  1  0 ]
    // [-s  0  c ]
    out.rotated_dir = vec3<f32>(
        world_dir.x * c + world_dir.z * s,
        world_dir.y,
        world_dir.x * -s + world_dir.z * c
    );

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let dir = normalize(in.rotated_dir);
    
    // Convert direction to equirectangular UV
    // atan2(z, x) gives angle in [-PI, PI]. We want [0, 1].
    // acos(y) gives angle in [0, PI]. We want [0, 1].
    
    let u = 0.5 + atan2(dir.z, dir.x) / (2.0 * 3.14159265);
    let v = 0.5 - asin(dir.y) / 3.14159265; // y is up
    
    let color = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v));
    
    // Apply brightness (dim at night)
    let brightness = camera.fog_time_pad.w;
    // Compute saturation: as brightness drops toward 0, saturation approaches min sat.
    let min_sat = camera.skybox_saturation_pad.x;
    let sat = min_sat + (1.0 - min_sat) * brightness;
    // Convert color to grayscale using luminance coefficients and mix with original
    let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let desaturated = mix(vec3<f32>(luminance), color.rgb, sat);
    // Apply tint towards `skybox_night_tint` with intensity scaled by how dark it is
    let tint = camera.skybox_tint_pad.xyz;
    let tint_strength = camera.skybox_tint_pad.w;
    let effect_strength = (1.0 - brightness) * tint_strength; // stronger at night
    let tinted = mix(desaturated, desaturated * tint, effect_strength);
    
    var out: FragmentOutput;
    out.color = vec4<f32>(tinted * brightness, color.a);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, 0.0); // Skybox is not emissive in the G-Buffer sense
    out.normal = vec4<f32>(0.0, 0.0, 0.0, 0.0); // Sky has no valid normal (detected by depth >= 1.0)
    return out;
}
