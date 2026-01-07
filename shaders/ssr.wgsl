// Volumetric Grid-Based Specular Reflections
// Replaces anchor-based cubemaps with a stable 3D GI Radiance Grid

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
    gi_grid_origin: vec3<i32>,
    _pad_gi1: i32,
    gi_grid_dims: vec3<i32>,
    _pad_gi2: i32,
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
    gi_scale: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
    _pad6: f32,
    _pad7: f32,
    _pad8: f32,
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
// GI probe 3D volumes (Rgba16Float) with hardware trilinear filtering.
// Faces are stored separately: +X, -X, +Y, -Y, +Z, -Z
@group(0) @binding(14) var gi_probe_px: texture_3d<f32>;
@group(0) @binding(15) var gi_probe_nx: texture_3d<f32>;
@group(0) @binding(16) var gi_probe_py: texture_3d<f32>;
@group(0) @binding(17) var gi_probe_ny: texture_3d<f32>;
@group(0) @binding(18) var gi_probe_pz: texture_3d<f32>;
@group(0) @binding(19) var gi_probe_nz: texture_3d<f32>;
// Average chunk color and occupancy volume for coarse reflections
@group(0) @binding(20) var gi_probe_color: texture_3d<f32>;

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
    let px = clamp(i32(uv.x * f32(dim.x)), 0, i32(dim.x) - 1);
    let py = clamp(i32(uv.y * f32(dim.y)), 0, i32(dim.y) - 1);
    return vec2<i32>(px, py);
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

fn sample_radiance(uvw: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let w = dir * dir;
    let color_x = select(textureSampleLevel(gi_probe_nx, linear_sampler, uvw, 0.0).rgb, textureSampleLevel(gi_probe_px, linear_sampler, uvw, 0.0).rgb, dir.x > 0.0);
    let color_y = select(textureSampleLevel(gi_probe_ny, linear_sampler, uvw, 0.0).rgb, textureSampleLevel(gi_probe_py, linear_sampler, uvw, 0.0).rgb, dir.y > 0.0);
    let color_z = select(textureSampleLevel(gi_probe_nz, linear_sampler, uvw, 0.0).rgb, textureSampleLevel(gi_probe_pz, linear_sampler, uvw, 0.0).rgb, dir.z > 0.0);
    return (color_x * w.x + color_y * w.y + color_z * w.z) * params.gi_scale;
}

fn sample_gi_grid(world_pos: vec3<f32>, reflect_dir: vec3<f32>) -> vec4<f32> {
    let dims = vec3<f32>(camera.gi_grid_dims);
    let grid_origin = vec3<f32>(camera.gi_grid_origin) * 16.0;
    
    // Coarse Volume Ray Marching
    // We step through the GI grid looking for opaque chunks (reflected geometry)
    var current_pos = world_pos;
    let step_size = 16.0; // One chunk per step
    
    var accumulated_color = vec3<f32>(0.0);
    var remaining_alpha = 1.0;
    var hit_count = 0u;

    for (var i = 1u; i < 16u; i++) {
        current_pos += reflect_dir * step_size;
        
        let grid_coord = (current_pos - grid_origin) / 16.0;
        let uvw = grid_coord / dims;
        
        if (any(grid_coord < vec3<f32>(0.0)) || any(grid_coord >= dims)) {
            break;
        }
        
        let chunk_data = textureSampleLevel(gi_probe_color, linear_sampler, uvw, 0.0);
        let occupancy = chunk_data.a;
        
        if (occupancy > 0.05) {
            let rad = sample_radiance(uvw, reflect_dir);
            let lit_color = chunk_data.rgb * rad * 2.5; // Coarse factor for lit surface radiance
            
            let alpha = occupancy * remaining_alpha;
            accumulated_color += lit_color * alpha;
            remaining_alpha -= alpha;
            hit_count += 1u;
            
            if (remaining_alpha < 0.1) {
                remaining_alpha = 0.0;
                break;
            }
        }
    }
    
    if (hit_count == 0u) {
        let distant_uvw = clamp((((world_pos + reflect_dir * 128.0) - grid_origin) / 16.0) / dims, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(sample_radiance(distant_uvw, reflect_dir), 0.0);
    }
    
    return vec4<f32>(accumulated_color, 1.0 - remaining_alpha);
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
    let gi_res = sample_gi_grid(world_pos, reflect_dir);
    
    let probe_color = gi_res.rgb;
    let probe_valid = gi_res.a;

    // Blend GI grid reflection with skybox
    let out_color = mix(sky_color, probe_color, probe_valid);

    return vec4<f32>(out_color, reflectivity);
}

