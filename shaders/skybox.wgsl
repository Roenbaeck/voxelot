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
    gi_scale: f32,
    _pad_gi0: f32,
    _pad_gi1: f32,
    _pad_gi2: f32,
    // rgb = shared horizon haze colour, w = atmosphere.horizon_haze_strength (density scale; 0 = legacy fog).
    haze_color: vec4<f32>,
};

// Shared sky/haze model constants (identical in voxel.wgsl, water.wgsl, impostor.wgsl).
const HAZE_SUN_GLOW: f32 = 0.15;
const HAZE_SCALE_HEIGHT: f32 = 140.0;
// Elevation (sine, ~4 degrees) over which the sky fades from the haze colour at the horizon
// to the skybox image.
const HAZE_HORIZON_BAND: f32 = 0.07;
// Distance at which the horizon band's opacity is evaluated: the haze the far sea reaches
// (water level, seen from the camera height), so sea and sky meet at the same value.
const HAZE_HORIZON_DISTANCE: f32 = 8000.0;

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
    // Unrotated world-space view direction (for the haze sun glow).
    @location(2) world_dir: vec3<f32>,
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
    
    out.world_dir = world_dir;

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
    @location(2) normal: vec2<f32>,
    @location(3) material: f32,
    @location(4) surface: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let dir = normalize(in.rotated_dir);
    
    // Convert direction to equirectangular UV
    // Pre-computed reciprocals avoid per-fragment divisions
    let INV_TWO_PI = 0.15915494;  // 1.0 / (2.0 * PI)
    let INV_PI     = 0.31830989;  // 1.0 / PI
    
    let u = 0.5 + atan2(dir.z, dir.x) * INV_TWO_PI;
    let v = 0.5 - asin(dir.y) * INV_PI; // y is up
    
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
    
    var sky = tinted * brightness;
    if (camera.haze_color.w > 0.0) {
        // Shared sky/haze model: near the horizon the sky fades to the same haze colour that
        // distant land and water fade to (voxel.wgsl compute_fog, water.wgsl), so there is
        // no seam where fully fogged geometry meets the sky. The haze colour comes from this
        // skybox's own horizon after the same day/night transform, so night stays dark.
        let view_dir = normalize(in.world_dir);
        let sun_dir = camera.sun_direction_shadow_bias.xyz;
        let haze = camera.haze_color.rgb
            + camera.sun_color_pad.xyz * HAZE_SUN_GLOW * max(dot(view_dir, sun_dir), 0.0);
        let band = 1.0 - smoothstep(0.0, HAZE_HORIZON_BAND, max(dir.y, 0.0));
        // Haze of a sea-level point HAZE_HORIZON_DISTANCE away (haze_amount in voxel.wgsl).
        let camera_height = max(camera.camera_shadow_strength.y - camera.water_level, 0.0);
        let camera_density = exp(-camera_height / HAZE_SCALE_HEIGHT);
        var height_factor = camera_density;
        if (camera_height > 1.0) {
            height_factor = HAZE_SCALE_HEIGHT * (1.0 - camera_density) / camera_height;
        }
        let opacity = 1.0 - exp(-camera.fog_time_pad.x * camera.haze_color.w * HAZE_HORIZON_DISTANCE * height_factor);
        sky = mix(sky, haze, band * opacity);
    }

    var out: FragmentOutput;
    out.color = vec4<f32>(sky, color.a);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, 1.0); // Skybox is not emissive in the G-Buffer sense
    out.normal = vec2<f32>(0.0, 0.0); // Sky has no valid normal (detected by depth >= 1.0)
    out.material = 0.0; // Sky has no reflectivity
    out.surface = vec4<f32>(0.0); // Sky: no albedo, nothing for contact AO to occlude
    return out;
}
