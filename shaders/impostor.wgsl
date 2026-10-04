struct CullParams {
    camera_position : vec3<f32>,
    candidate_count : u32,
    camera_forward : vec3<f32>,
    _pad0 : u32,
    near_plane : f32,
    far_plane : f32,
    camera_right : vec3<f32>,
    _pad_r0 : u32,
    camera_up : vec3<f32>,
    _pad_u0 : u32,
    fov_tan : f32,
    aspect : f32,
    screen_width : f32,
    screen_height : f32,
    fog_density : f32,
    skybox_brightness : f32,
    impostor_pixel_threshold : f32,
    impostor_pixel_size : f32,
    lod_render_distance : f32,
    detail_cull_distance : f32,
    envelope_distance : f32,
    envelope_fade_range : f32,
    hzb_enabled : u32,
    max_hzb_mip : u32,
    hzb_valid : u32,
    _pad3 : u32,
    view_proj : mat4x4<f32>,
    hzb_view_proj : mat4x4<f32>,
    hzb_camera_position : vec3<f32>,
    hzb_margin_px : f32,
    // rgb = shared horizon haze colour, w = atmosphere.horizon_haze_strength (density scale; 0 = legacy fog).
    haze_color : vec4<f32>,
    // x = haze base height (water level)
    haze_params : vec4<f32>,
};

// Shared sky/haze model constant (identical in voxel.wgsl, water.wgsl, skybox.wgsl).
const HAZE_SCALE_HEIGHT: f32 = 140.0;

// Fraction of light replaced by haze between `camera` and `point` (see voxel.wgsl).
fn haze_amount(point: vec3<f32>, camera_pos: vec3<f32>, dist: f32, density: f32, base_height: f32) -> f32 {
    let camera_height = max(camera_pos.y - base_height, 0.0);
    let point_height = max(point.y - base_height, 0.0);
    let camera_density = exp(-camera_height / HAZE_SCALE_HEIGHT);
    let height_delta = point_height - camera_height;
    var height_factor = camera_density;
    if (abs(height_delta) > 1.0) {
        height_factor = HAZE_SCALE_HEIGHT
            * (camera_density - exp(-point_height / HAZE_SCALE_HEIGHT)) / height_delta;
    }
    return 1.0 - exp(-dist * density * height_factor);
}

struct ImpostorInstance {
    position : vec3<f32>,
    _pad0 : f32,
    color : vec4<f32>,
    emissive : vec4<f32>,
    material : vec4<f32>,
};

@group(0) @binding(0)
var<uniform> params : CullParams;

struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) emissive : vec4<f32>,
    @location(2) world_pos : vec3<f32>,
    @location(3) @interpolate(flat) material : vec4<f32>,
};

const QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-0.5, -0.5),
    vec2<f32>(0.5, -0.5),
    vec2<f32>(0.5, 0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>(0.5, 0.5),
    vec2<f32>(-0.5, 0.5)
);

@vertex
fn vs_main(
    @builtin(vertex_index) vid : u32,
    @location(0) inst_pos : vec3<f32>,
    @location(1) inst_color : vec4<f32>,
    @location(2) inst_emissive : vec4<f32>,
    @location(3) inst_material : vec4<f32>,
) -> VertexOut {
    var out : VertexOut;
    let clip = params.view_proj * vec4<f32>(inst_pos, 1.0);
    let size_px = max(params.impostor_pixel_size, 1.0);
    let offset_px = QUAD[vid] * size_px;
    let screen_w = max(params.screen_width, 1.0);
    let screen_h = max(params.screen_height, 1.0);
    let ndc_offset = vec2<f32>(
        offset_px.x * 2.0 / screen_w,
        offset_px.y * 2.0 / screen_h
    );
    out.position = clip + vec4<f32>(ndc_offset * clip.w, 0.0, 0.0);
    out.color = inst_color;
    out.emissive = inst_emissive;
    out.world_pos = inst_pos;
    out.material = inst_material;
    return out;
}

struct FragmentOut {
    @location(0) color : vec4<f32>,
    @location(1) emissive : vec4<f32>,
    @location(2) normal : vec2<f32>,
    @location(3) material : f32,
    @location(4) surface : vec4<f32>,
};

@fragment
fn fs_main(input : VertexOut) -> FragmentOut {
    var out : FragmentOut;
    let alpha = select(1.0, input.color.a, input.color.a > 0.0);
    let reflectivity = clamp(input.material.r, 0.0, 1.0);
    let sky_reflection = mix(vec3<f32>(0.035, 0.04, 0.06), vec3<f32>(0.78, 0.86, 0.95), params.skybox_brightness);
    let base_color = input.color.rgb * params.skybox_brightness + sky_reflection * reflectivity;
    let fog_density = max(params.fog_density, 0.0);
    let dist = length(params.camera_position - input.world_pos);
    let transmittance = exp(-fog_density * dist);
    var fog_color = mix(vec3<f32>(0.02, 0.02, 0.03), vec3<f32>(0.7, 0.8, 0.9), params.skybox_brightness);
    var fog_amount = 1.0 - transmittance;
    if (params.haze_color.w > 0.0) {
        // Shared sky/haze model (voxel.wgsl compute_fog), without the sun glow term.
        fog_color = params.haze_color.rgb;
        fog_amount = haze_amount(
            input.world_pos,
            params.camera_position,
            dist,
            fog_density * params.haze_color.w,
            params.haze_params.x,
        );
    }
    let fogged = mix(base_color, fog_color, fog_amount);
    out.color = vec4<f32>(fogged, alpha);
    let emitted = input.emissive.rgb * input.emissive.a;
    out.emissive = vec4<f32>(emitted, 1.0);
    // Surface G-buffer (see voxel.wgsl pack_surface): the impostor's whole diffuse term is
    // ambient-like, so its share is the diffuse luminance over the final luminance.
    let impostor_luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let diffuse_luma = dot(input.color.rgb * params.skybox_brightness, impostor_luma) * (1.0 - fog_amount);
    let total_luma = dot(fogged + emitted, impostor_luma);
    out.surface = vec4<f32>(
        input.color.rgb * (1.0 - fog_amount),
        clamp(diffuse_luma / max(total_luma, 1.0e-4), 0.0, 1.0),
    );
    out.normal = vec2<f32>(0.0, 0.0);
    out.material = reflectivity;
    return out;
}
