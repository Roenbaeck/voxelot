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
    _pad1: vec2<f32>,
};

struct GiProbe {
    position: vec4<f32>,
    light_data: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: RCParams;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var hzb_texture: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> gi_probes: array<GiProbe>;
@group(0) @binding(6) var post_sampler: sampler;

// Output texture for the radiance
@group(0) @binding(7) var output_tex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let world_pos = camera.inverse_view_proj * vec4<f32>(ndc, 1.0);
    return world_pos.xyz / world_pos.w;
}

// Samples the cubemap-like probe data
fn sample_probe_radiance(probe_idx: u32, dir: vec3<f32>) -> vec3<f32> {
    let probe = gi_probes[probe_idx];
    let normal = normalize(dir);
    
    let w_x = normal.x * normal.x;
    let w_y = normal.y * normal.y;
    let w_z = normal.z * normal.z;
    
    let idx_x = select(0u, 1u, normal.x > 0.0);
    let idx_y = select(2u, 3u, normal.y > 0.0);
    let idx_z = select(4u, 5u, normal.z > 0.0);
    
    return probe.light_data[idx_x].rgb * w_x + 
           probe.light_data[idx_y].rgb * w_y + 
           probe.light_data[idx_z].rgb * w_z;
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
            // Fibonacci sphere or similar for 3D ray distribution
            let phi = acos(1.0 - 2.0 * (f32(r) + 0.5) / f32(ray_count));
            let theta = PI * (1.0 + sqrt(5.0)) * (f32(r) + 0.5);
            let ray_dir = vec3<f32>(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
            
            // Trace ray against HZB for local occlusion
            // (Simplified for this example)
            let hit_radiance = sample_probe_radiance(0u, ray_dir); // Sample nearest probe for now
            cascade_radiance += hit_radiance;
        }
        
        total_radiance += (cascade_radiance / f32(ray_count)) / f32(cascade_count);
    }
    
    textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(total_radiance, 1.0));
}

