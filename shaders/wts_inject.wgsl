// WTS-RT: Light Injection Shader
// Seeds the WTS simulation by injecting direct sunlight into the GI probe grid.

struct InjectParams {
    _unused0: u32,
    gi_grid_origin_x: i32,
    gi_grid_origin_y: i32,
    gi_grid_origin_z: i32,
    gi_grid_dim_x: i32,
    gi_grid_dim_y: i32,
    gi_grid_dim_z: i32,
    _pad0: u32,
    sun_direction_intensity: vec4<f32>, // xyz: direction, w: intensity
    sun_color_pad: vec4<f32>,           // rgb: color, w: pad
    shadow_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var seed_texture : texture_3d<f32>;
@group(0) @binding(1) var<uniform> params : InjectParams;
@group(0) @binding(2) var injection_texture : texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var shadow_texture : texture_depth_2d;
@group(0) @binding(4) var shadow_sampler : sampler_comparison;

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let dims = vec3<i32>(params.gi_grid_dim_x, params.gi_grid_dim_y, params.gi_grid_dim_z);

    let grid_coord = vec3<i32>(global_id);
    if (any(grid_coord < vec3<i32>(0)) || any(grid_coord >= dims)) {
        return;
    }

    let seed = textureLoad(seed_texture, grid_coord, 0);
    let occupancy = clamp(seed.a, 0.0, 1.0);
    if (occupancy <= 0.001) {
        return;
    }

    let grid_origin = vec3<i32>(params.gi_grid_origin_x, params.gi_grid_origin_y, params.gi_grid_origin_z);
    let world_coord_i = grid_coord + grid_origin;
    let center = (vec3<f32>(world_coord_i) + vec3<f32>(0.5)) * 16.0;
    let sun_dir = normalize(params.sun_direction_intensity.xyz);
    let sun_side = vec3<f32>(
        select(-1.0, 1.0, sun_dir.x >= 0.0),
        select(-1.0, 1.0, sun_dir.y >= 0.0),
        select(-1.0, 1.0, sun_dir.z >= 0.0),
    );
    let shadow_sample_pos = center + sun_side * 8.25;

    var light = vec3<f32>(0.0);

    // Sunlight Injection (Shadow Map)
    // Project center to shadow space
    let shadow_pos = params.shadow_matrix * vec4<f32>(shadow_sample_pos, 1.0);
    // Convert NDC to UV (0..1)
    // NDC: -1..1 -> 0..1: x*0.5+0.5, y*-0.5+0.5 (flip Y)
    // Wait, wgpu shadow texture coords: 0,0 is top-left? 
    // Usually shadow matrix does the crop/transform.
    // Standard perspective divide:
    let proj_coords = shadow_pos.xyz / shadow_pos.w;
    let u = proj_coords.x * 0.5 + 0.5;
    let v = 1.0 - (proj_coords.y * 0.5 + 0.5); // Flip Y for sampling
    let current_depth = proj_coords.z;
    
    if (u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0 && current_depth >= 0.0 && current_depth <= 1.0) {
        let shadow = textureSampleCompareLevel(shadow_texture, shadow_sampler, vec2<f32>(u, v), current_depth - 0.005);
        if (shadow > 0.0) {
            // Phi stores incident radiance; surface shaders apply albedo later.
            // Use occupancy only as a coverage term so sparse-but-visible chunks
            // still receive enough stable sunlight to diffuse.
            let coverage = sqrt(occupancy);
            light += params.sun_color_pad.rgb * params.sun_direction_intensity.w * shadow * coverage;
        }
    }
    
    // Write to Injection Texture
    // Alpha 1.0 is not strictly needed here but kept for consistency
    textureStore(injection_texture, grid_coord, vec4<f32>(light, 1.0));
}
