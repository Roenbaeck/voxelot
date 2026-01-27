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
    _pad3 : f32,
    view_proj : mat4x4<f32>,
};

struct ImpostorInstance {
    position : vec3<f32>,
    _pad0 : f32,
    color : vec4<f32>,
    emissive : vec4<f32>,
};

@group(0) @binding(0)
var<uniform> params : CullParams;

struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) emissive : vec4<f32>,
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
    return out;
}

struct FragmentOut {
    @location(0) color : vec4<f32>,
    @location(1) emissive : vec4<f32>,
    @location(2) normal : vec2<f32>,
    @location(3) material : f32,
};

@fragment
fn fs_main(input : VertexOut) -> FragmentOut {
    var out : FragmentOut;
    let alpha = select(1.0, input.color.a, input.color.a > 0.0);
    let base_color = input.color.rgb * params.skybox_brightness;
    let fog_density = max(params.fog_density, 0.0);
    let dist = length(params.camera_position - input.position.xyz);
    let transmittance = exp(-fog_density * dist);
    let fog_base = mix(vec3<f32>(0.02, 0.02, 0.03), vec3<f32>(0.7, 0.8, 0.9), params.skybox_brightness);
    let fog_color = fog_base * vec3<f32>(1.0, 1.0, 1.0);
    let fogged = mix(base_color, fog_color, 1.0 - transmittance);
    out.color = vec4<f32>(fogged, alpha);
    out.emissive = vec4<f32>(input.emissive.rgb * input.emissive.a, 1.0);
    out.normal = vec2<f32>(0.0, 0.0);
    out.material = 0.0;
    return out;
}
