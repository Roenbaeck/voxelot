// WTS-RT: Light Injection Shader
// Seeds the WTS simulation by injecting direct light and emissives into the grid.

struct InstanceInput {
    position : vec3<f32>,
    _pad0: f32, // explicit padding to match alignment if needed, but usually vec3 is 16-byte aligned in array? 
                // Wait, Rust side uses [f32; 3] usually. 
                // wgpu standard std140/430 rules: vec3 is 16 bytes.
                // Rust struct usually has padding. 
                // Let's assume the previous shader definition was correct.
                // gpu_cull.wgsl didn't have explicit padding in struct, but implicit.
    scale : vec3<f32>,
    _pad1: f32,
    custom_color : vec4<f32>,
    emissive : vec4<f32>,
    voxel_type : u32,
    flags : u32, 
    mesh_index : u32,
    envelope_index : u32,
};

struct InjectParams {
    candidate_count: u32,
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

@group(0) @binding(0) var<storage, read> candidates : array<InstanceInput>;
@group(0) @binding(1) var<uniform> params : InjectParams;
@group(0) @binding(2) var injection_texture : texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var shadow_texture : texture_depth_2d;
@group(0) @binding(4) var shadow_sampler : sampler_comparison;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;
    if (index >= params.candidate_count) {
        return;
    }

    let instance = candidates[index];
    
    // Chunk Center
    // Note: instance.position is min corner. instance.scale is size (usually 16.0).
    let center = instance.position + instance.scale * 0.5;
    
    // GI Grid Coordinate
    // Grid voxel size is 16.0
    let grid_origin = vec3<i32>(params.gi_grid_origin_x, params.gi_grid_origin_y, params.gi_grid_origin_z);
    let world_coord_i = vec3<i32>(floor(center / 16.0));
    let grid_coord = world_coord_i - grid_origin;
    
    let dims = vec3<i32>(params.gi_grid_dim_x, params.gi_grid_dim_y, params.gi_grid_dim_z);
    
    if (any(grid_coord < vec3<i32>(0)) || any(grid_coord >= dims)) {
        return;
    }
    
    // Lighting Calculation
    var light = instance.emissive.rgb;
    
    // Sunlight Injection (Shadow Map)
    // Project center to shadow space
    let shadow_pos = params.shadow_matrix * vec4<f32>(center, 1.0);
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
            // Apply sunlight scaled by chunk color (albedo)
            light += params.sun_color_pad.rgb * params.sun_direction_intensity.w * shadow * instance.custom_color.rgb;
        }
    }
    
    // Write to Injection Texture
    // Alpha 1.0 is not strictly needed here but kept for consistency
    textureStore(injection_texture, grid_coord, vec4<f32>(light, 1.0));
}
