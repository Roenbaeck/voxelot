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
@group(0) @binding(6) var post_sampler: sampler;

// Output texture for the radiance
@group(0) @binding(7) var output_tex: texture_storage_2d<rgba16float, write>;

@group(0) @binding(8) var<storage, read> light_probes: array<LightProbe>;

const PI: f32 = 3.14159265359;
const GOLDEN_RATIO: f32 = 1.61803398875;

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let world_pos = camera.inverse_view_proj * vec4<f32>(ndc, 1.0);
    return world_pos.xyz / world_pos.w;
}

fn sample_dynamic_lights(world_pos: vec3<f32>, ray_dir: vec3<f32>, max_dist: f32) -> vec3<f32> {
    var light_acc = vec3<f32>(0.0);
    for (var i = 0u; i < params.light_probe_count; i++) {
        let probe = light_probes[i];
        let to_light = probe.position - world_pos;
        let dist_sq = dot(to_light, to_light);
        
        if (dist_sq > 0.001 && dist_sq < max_dist * max_dist) {
            let dist = sqrt(dist_sq);
            let dir_to_light = to_light / dist;
            let alignment = dot(ray_dir, dir_to_light);
            
            // Smooth falloff to fix "bokeh" circles and handle low ray counts
            let smooth_alignment = smoothstep(0.7, 1.0, alignment);
            if (smooth_alignment > 0.0) {
                let attenuation = (probe.color_power.a * 0.1) / (dist_sq + 1.0);
                light_acc += probe.color_power.rgb * attenuation * smooth_alignment;
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
    
    // Early exit: Distance culling for dynamic lights
    // If no dynamic lights are within range, we don't need to run the expensive ray loops.
    var near_light = false;
    for (var i = 0u; i < params.light_probe_count; i++) {
        let probe_pos = light_probes[i].position;
        let dist_sq = dot(probe_pos - world_pos, probe_pos - world_pos);
        // Use a reasonable influence radius (e.g., 48 units)
        if (dist_sq < 2304.0) { // 48 * 48
            near_light = true;
            break;
        }
    }

    if (!near_light) {
        textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(0.0));
        return;
    }

    // Per-pixel jitter to break up aliasing patterns
    let jitter = hash22(uv * 1000.0 + f32(params.frame_count % 1000u));
    let rotation_offset = jitter.x * 2.0 * PI;

    // Radiance Cascades Merging Logic
    // In a full implementation, this would be multiple passes.
    // Here we implement a single-pass version that leverages the probe grid.
    
    var total_radiance = vec3<f32>(0.0);
    let cascade_count = params.cascade_count;
    
    for (var c = 0u; c < cascade_count; c++) {
        // Each cascade has more rays but covers a larger area
        let ray_count = params.ray_count_base * (1u << c);
        let cascade_dist = params.max_dist * f32(c + 1u) / f32(cascade_count);
        
        var cascade_radiance = vec3<f32>(0.0);
        
        for (var r = 0u; r < ray_count; r++) {
            // Optimized Fibonacci sphere distribution with per-pixel rotation jitter
            let z = 1.0 - 2.0 * (f32(r) + 0.5) / f32(ray_count);
            let radius = sqrt(max(0.0, 1.0 - z * z));
            let theta = 2.0 * PI * GOLDEN_RATIO * f32(r) + rotation_offset;
            let ray_dir = vec3<f32>(radius * cos(theta), radius * sin(theta), z);
            
            // Trace ray against dynamic lights
            let dynamic_radiance = sample_dynamic_lights(world_pos, ray_dir, cascade_dist);
            
            cascade_radiance += dynamic_radiance;
        }
        
        total_radiance += (cascade_radiance / f32(ray_count)) / f32(cascade_count);
    }
    
    textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(total_radiance, 1.0));
}

