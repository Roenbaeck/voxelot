// Cubemap-only reflections pass.
// This pass outputs a reflected color (RGB) and reflectivity in A.
// Screen-space ray marching / temporal accumulation are intentionally disabled.

struct CameraUniforms {
    inverse_view: mat4x4<f32>,
    inverse_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    skybox_rotation: f32,
    skybox_brightness: f32,
    skybox_saturation: f32,
    _pad1: vec2<f32>,
    skybox_tint: vec3<f32>,
    skybox_tint_strength: f32,
    // xyz = probe center in world space, w = proxy half-extent for box-projected cubemap sampling
    probe_pos_extent: vec4<f32>,
}

struct SSRParams {
    max_steps: u32,
    max_binary_steps: u32,
    step_size: f32,
    thickness: f32,
    overscan: f32,
    bloom_strength: f32,
    frame_index: f32,
    history_valid: f32,
    probe_valid: f32,
    probe_strength: f32,
    // Debug controls packed here:
    //   x = probe-only debug view (1 = show probe, ignore skybox)
    //   y = flip-Y when sampling probe (1 = invert direction.y)
    _pad2: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: SSRParams;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var hzb_texture: texture_2d<f32>;
@group(0) @binding(6) var hzb_sampler: sampler;
@group(0) @binding(7) var normal_gbuffer: texture_2d<f32>;
@group(0) @binding(8) var material_gbuffer: texture_2d<f32>;
@group(0) @binding(9) var ssao_texture: texture_2d<f32>;
@group(0) @binding(10) var bloom_texture: texture_2d<f32>;
@group(0) @binding(11) var skybox_texture: texture_2d<f32>;
@group(0) @binding(12) var skybox_sampler: sampler;
@group(0) @binding(13) var ssr_history: texture_2d<f32>;
@group(0) @binding(14) var reflection_probe: texture_cube<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    // Fullscreen triangle
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);
    return output;
}

fn decode_world_normal(encoded: vec3<f32>) -> vec3<f32> {
    let n = encoded * 2.0 - 1.0;
    return normalize(select(n, vec3<f32>(0.0, 1.0, 0.0), dot(n, n) < 1e-8));
}

fn uv_to_pixel(uv: vec2<f32>, dim: vec2<u32>) -> vec2<i32> {
    let x = clamp(i32(uv.x * f32(dim.x)), 0, i32(dim.x) - 1);
    let y = clamp(i32(uv.y * f32(dim.y)), 0, i32(dim.y) - 1);
    return vec2<i32>(x, y);
}

fn load_depth_at_uv(uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(scene_depth);
    let px = uv_to_pixel(uv, dim);
    return textureLoad(scene_depth, px, 0);
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let z_ndc = depth * 2.0 - 1.0;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, z_ndc, 1.0);
    let view_pos = camera.inverse_proj * ndc;
    let view_pos_3d = view_pos.xyz / view_pos.w;
    let world_pos = camera.inverse_view * vec4<f32>(view_pos_3d, 1.0);
    return world_pos.xyz;
}

fn intersect_ray_aabb(origin: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> f32 {
    // Returns the first positive intersection distance along the ray, or -1 if no hit.
    let eps = 1e-6;
    let safe_dir = vec3<f32>(
        select(dir.x, eps, abs(dir.x) < eps),
        select(dir.y, eps, abs(dir.y) < eps),
        select(dir.z, eps, abs(dir.z) < eps),
    );
    let inv_dir = 1.0 / safe_dir;
    let t0 = (bmin - origin) * inv_dir;
    let t1 = (bmax - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
    let tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (tmax < max(tmin, 0.0)) {
        return -1.0;
    }
    return select(tmin, tmax, tmin < 0.0);
}

fn parallax_correct_probe_dir(ray_origin: vec3<f32>, reflect_dir: vec3<f32>) -> vec3<f32> {
    let probe_center = camera.probe_pos_extent.xyz;
    let half_extent = camera.probe_pos_extent.w;
    if (half_extent <= 0.0) {
        return reflect_dir;
    }
    let half = vec3<f32>(half_extent);
    let bmin = probe_center - half;
    let bmax = probe_center + half;
    let t = intersect_ray_aabb(ray_origin, reflect_dir, bmin, bmax);
    if (t <= 0.0) {
        return reflect_dir;
    }
    let hit_pos = ray_origin + reflect_dir * t;
    return normalize(hit_pos - probe_center);
}

fn sample_sky_equirect(reflect_dir: vec3<f32>) -> vec3<f32> {
    let angle = camera.skybox_rotation;
    let c = cos(angle);
    let s = sin(angle);
    let rotated_dir = vec3<f32>(
        reflect_dir.x * c + reflect_dir.z * s,
        reflect_dir.y,
        reflect_dir.x * -s + reflect_dir.z * c
    );
    let PI = 3.14159265359;
    let TWO_PI = 6.28318530718;
    let u = 0.5 + atan2(rotated_dir.z, rotated_dir.x) / TWO_PI;
    let v = 0.5 - asin(clamp(rotated_dir.y, -1.0, 1.0)) / PI;
    let sky_sample = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v)).rgb;

    let min_sat = camera.skybox_saturation;
    let brightness = camera.skybox_brightness;
    let sat = clamp(min_sat + (1.0 - min_sat) * brightness, 0.0, 1.0);
    let lum = dot(sky_sample, vec3<f32>(0.299, 0.587, 0.114));
    let desaturated = mix(vec3<f32>(lum), sky_sample, sat);
    let tint = camera.skybox_tint;
    let tint_strength = camera.skybox_tint_strength;
    let effect_strength = (1.0 - brightness) * tint_strength;
    let tinted = mix(desaturated, desaturated * tint, effect_strength);
    return tinted * brightness;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let depth = load_depth_at_uv(input.uv);
    if (depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let material = textureSample(material_gbuffer, linear_sampler, input.uv);
    let reflectivity = material.r;
    if (reflectivity < 0.01) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let world_pos = reconstruct_world_pos(input.uv, depth);
    let gbuf = textureSample(normal_gbuffer, linear_sampler, input.uv);
    let normal = decode_world_normal(gbuf.rgb);

    let view_dir = normalize(world_pos - camera.camera_pos);
    let reflect_dir = reflect(view_dir, normal);

    let sky_color = sample_sky_equirect(reflect_dir);

    var probe_dir = parallax_correct_probe_dir(world_pos, reflect_dir);
    if (params._pad2.y > 0.5) {
        probe_dir.y = -probe_dir.y;
    }
    let probe_color = textureSample(reflection_probe, linear_sampler, probe_dir).rgb;

    if (params._pad2.x > 0.5) {
        return vec4<f32>(probe_color, 1.0);
    }

    let probe_on = (params.probe_valid * params.probe_strength) > 0.5;
    let out_color = select(sky_color, probe_color, probe_on);
    return vec4<f32>(out_color, reflectivity);
}
