// Radiance Cascades 3D - GPU Implementation for Voxelot
// This shader implements the merging and sampling logic for Radiance Cascades.

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
};

struct RCParams {
    screen_width: f32,
    screen_height: f32,
    cascade_count: u32,
    ray_count_base: u32,
    step_size: f32,
    max_dist: f32,
    light_probe_count: u32,
    frame_count: u32,
};

struct LightProbe {
    position: vec3<f32>,
    _pad0: f32,
    color_power: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: RCParams;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var hzb_texture: texture_2d<f32>;
@group(0) @binding(5) var normal_gbuffer: texture_2d<f32>;
@group(0) @binding(6) var post_sampler: sampler;

// Output texture for the radiance
@group(0) @binding(7) var output_tex: texture_storage_2d<rgba16float, write>;

@group(0) @binding(8) var<storage, read> light_probes: array<LightProbe>;

const PI: f32 = 3.14159265359;
const GOLDEN_RATIO: f32 = 1.61803398875;

// Average value of: smoothstep(0.7, 1.0, dot(ray_dir, dir_to_light))
// when ray_dir is uniformly distributed on the sphere.
// This replaces the previous low-ray-count Monte Carlo integration (which produced visible speckle).
const AVG_SMOOTH_ALIGNMENT: f32 = 0.075;

// Match CPU-side probe culling radius (see voxelot.rs LIGHT_RADIUS_SQ).
const LIGHT_INFLUENCE_RADIUS: f32 = 48.0;
// Smoothly fade RC on/off near the influence edge to avoid popping.
const LIGHT_FADE_RANGE: f32 = 10.0;

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let world_pos = camera.inverse_view_proj * vec4<f32>(ndc, 1.0);
    return world_pos.xyz / world_pos.w;
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    // Start on octahedron surface
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));

    // Fold back the parts where z is negative (lower hemisphere)
    if (v.z < 0.0) {
        let ox = (1.0 - abs(v.y)) * select(-1.0, 1.0, v.x >= 0.0);
        let oy = (1.0 - abs(v.x)) * select(-1.0, 1.0, v.y >= 0.0);
        v.x = ox;
        v.y = oy;
    }

    return normalize(v);
}

fn sample_dynamic_lights_avg(world_pos: vec3<f32>, normal: vec3<f32>, max_dist: f32) -> vec3<f32> {
    var light_acc = vec3<f32>(0.0);
    for (var i = 0u; i < params.light_probe_count; i++) {
        let probe = light_probes[i];
        let to_light = probe.position - world_pos;
        let dist_sq = dot(to_light, to_light);
        
        if (dist_sq > 0.001 && dist_sq < max_dist * max_dist) {
            let dist = sqrt(dist_sq);
            let dir_to_light = to_light / dist;
            
            // Lambertian term: dot(normal, dir_to_light)
            // We use max(0, dot) to only light surfaces facing the light.
            let lambert = max(0.0, dot(normal, dir_to_light));
            
            if (lambert > 0.0) {
                let attenuation = (probe.color_power.a * 0.1) / (dist_sq + 1.0);
                // We still use AVG_SMOOTH_ALIGNMENT as a base factor for the RC integration,
                // but now weighted by the actual surface normal.
                light_acc += probe.color_power.rgb * attenuation * lambert * 0.5;
            }
        }
    }
    return light_acc;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = vec2<u32>(u32(params.screen_width), u32(params.screen_height));
    if (id.x >= screen_size.x || id.y >= screen_size.y) {
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(screen_size);
    let depth = textureLoad(scene_depth, vec2<i32>(id.xy), 0);
    
    if (depth >= 1.0) {
        textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(0.0));
        return;
    }

    let world_pos = reconstruct_world_pos(uv, depth);
    
    // Sample Octahedral-encoded world-space normal from G-buffer
    let gbuf = textureLoad(normal_gbuffer, vec2<i32>(id.xy), 0).xy;
    let world_normal = oct_decode(gbuf);
    
    // Distance culling + smooth fade for dynamic lights.
    // We fade in near the probe influence edge to avoid abrupt popping.
    var nearest_dist_sq = 1e30;
    for (var i = 0u; i < params.light_probe_count; i++) {
        let probe_pos = light_probes[i].position;
        let d = probe_pos - world_pos;
        let dist_sq = dot(d, d);
        nearest_dist_sq = min(nearest_dist_sq, dist_sq);
    }

    let nearest_dist = sqrt(max(0.0, nearest_dist_sq));
    let fade = 1.0 - smoothstep(LIGHT_INFLUENCE_RADIUS - LIGHT_FADE_RANGE, LIGHT_INFLUENCE_RADIUS, nearest_dist);
    if (fade <= 0.0) {
        textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(0.0));
        return;
    }

    // Radiance Cascades Merging Logic
    // In a full implementation, this would be multiple passes.
    // Here we implement a single-pass version that leverages the probe grid.
    
    var total_radiance = vec3<f32>(0.0);
    let cascade_count = params.cascade_count;
    
    for (var c = 0u; c < cascade_count; c++) {
        // Each cascade covers a larger area. (Ray count no longer applies: we use a deterministic integral.)
        let cascade_dist = params.max_dist * f32(c + 1u) / f32(cascade_count);

        let cascade_radiance = sample_dynamic_lights_avg(world_pos, world_normal, cascade_dist);
        total_radiance += cascade_radiance / f32(cascade_count);
    }
    
    textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(total_radiance * fade, 1.0));
}

