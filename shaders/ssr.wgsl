// Screen Space Reflections shader with HZB acceleration

struct CameraUniforms {
    inverse_view: mat4x4<f32>,
    inverse_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
}

struct SSRParams {
    max_steps: u32,
    max_binary_steps: u32,
    step_size: f32,
    thickness: f32,
    overscan: f32,  // How much extra UV space we can sample (e.g., 0.2 = 20% overscan)
    _pad: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: SSRParams;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var hzb_texture: texture_2d<f32>;
@group(0) @binding(6) var hzb_sampler: sampler;
@group(0) @binding(7) var normal_gbuffer: texture_2d<f32>;
@group(0) @binding(8) var material_gbuffer: texture_2d<f32>;  // R=reflectivity

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

// Reconstruct world position from depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert UV and depth to NDC (depth is [0,1], convert to NDC z)
    let z_ndc = depth * 2.0 - 1.0;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, z_ndc, 1.0);
    
    // Unproject to view space
    let view_pos = camera.inverse_proj * ndc;
    let view_pos_3d = view_pos.xyz / view_pos.w;
    
    // Transform to world space
    let world_pos = camera.inverse_view * vec4<f32>(view_pos_3d, 1.0);
    return world_pos.xyz;
}

// Project world position to screen space
fn world_to_screen(world_pos: vec3<f32>) -> vec3<f32> {
    let clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = clip_pos.xyz / clip_pos.w;
    // Convert to UV coordinates
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3<f32>(uv, ndc.z);
}

fn decode_world_normal(encoded: vec3<f32>) -> vec3<f32> {
    // Stored as (n * 0.5 + 0.5) in voxel.wgsl
    let n = encoded * 2.0 - 1.0;
    // Guard against zero/denormals
    return normalize(select(n, vec3<f32>(0.0, 1.0, 0.0), dot(n, n) < 1e-8));
}

// Calculate max mip level for HZB
fn get_max_mip_level() -> f32 {
    let dims = vec2<f32>(textureDimensions(hzb_texture, 0));
    let max_dim = max(dims.x, dims.y);
    return floor(log2(max_dim));
}

// HZB-Accelerated Ray Marching
// HZB-Accelerated Ray Marching (Screen Space)
// Returns vec4: xy = hit UV, z = hit confidence (0 or 1), w = world distance traveled
fn hzb_ray_march(start_pos: vec3<f32>, ray_dir: vec3<f32>, start_uv: vec2<f32>, start_depth: f32) -> vec4<f32> {
    var hit_result = vec4<f32>(-1.0, -1.0, -1.0, 0.0);
    
    // Calculate end position in world space
    let max_dist = f32(params.max_steps) * params.step_size;
    let end_pos = start_pos + ray_dir * max_dist;
    
    // Project start and end to screen space
    let start_screen = world_to_screen(start_pos);
    let end_screen = world_to_screen(end_pos);
    
    // Calculate delta in screen space
    let delta = (end_screen - start_screen) / f32(params.max_steps);
    // World space step size
    let world_step = max_dist / f32(params.max_steps);
    
    var current_screen = start_screen;
    let max_mip = get_max_mip_level();
    var current_mip = min(4.0, max_mip);
    
    // Minimum world distance before accepting any hit (in world units/voxels)
    // Building walls are often adjacent, so we need a decent distance
    let min_world_distance = 10.0;
    
    // Track if ray is going toward or away from camera (in depth)
    // Positive delta.z means ray goes further from camera, negative means closer
    let ray_goes_further = delta.z > 0.0;
    
    // Calculate UV bounds with overscan
    // Overscan allows sampling off-screen geometry that was rendered
    let os = params.overscan;
    let uv_min = -os / (1.0 + os);  // Maps to start of overscan region
    let uv_max = 1.0 + os / (1.0 + os);  // Maps to end of overscan region
    
    // Hierarchical ray march
    for (var i = 0u; i < params.max_steps; i++) {
        current_screen += delta;
        
        // Current world distance traveled
        let world_dist = f32(i + 1u) * world_step;
        
        let uv = current_screen.xy;
        
        // Check if out of screen bounds (with overscan)
        if (uv.x < uv_min || uv.x > uv_max || uv.y < uv_min || uv.y > uv_max) {
            break;
        }
        
        // Skip hits that are too close in world space (avoid hitting adjacent building surfaces)
        if (world_dist < min_world_distance) {
            continue;
        }
        
        // For rays going further from camera: reject hits at depth less than start (behind us)
        // For rays going toward camera: reject hits at depth greater than start
        let ray_depth = current_screen.z;
        if (ray_goes_further && ray_depth < start_depth - 0.001) {
            continue;  // Hit is behind the starting surface
        }
        
        // Sample HZB at current mip level
        let hzb_depth = textureSampleLevel(hzb_texture, hzb_sampler, uv, current_mip).r;
        
        // Check if ray is behind surface (potential intersection)
        if (ray_depth > hzb_depth && ray_depth - hzb_depth < params.thickness) {
            if (current_mip <= 0.5) {
                // At finest detail, perform binary refinement
                // We can refine in screen space too!
                var refined_screen = current_screen - delta;
                var refined_delta = delta;
                
                for (var j = 0u; j < params.max_binary_steps; j++) {
                    refined_delta *= 0.5;
                    refined_screen += refined_delta;
                    
                    let refined_uv = refined_screen.xy;
                    let refined_depth_sample = textureSampleLevel(hzb_texture, hzb_sampler, refined_uv, 0.0).r;
                    
                    if (refined_screen.z > refined_depth_sample) {
                        refined_screen -= refined_delta;
                    }
                }
                
                hit_result = vec4<f32>(refined_screen.xy, 1.0, world_dist);
                break;
            } else {
                // Descend to finer mip level
                current_mip = max(0.0, current_mip - 1.0);
            }
        } else {
            // No intersection at this point
            // Optionally ascend to coarser mip if we're far from surfaces
            let depth_diff = abs(ray_depth - hzb_depth);
            if (depth_diff > params.thickness * 4.0 && current_mip < max_mip) {
                current_mip = min(max_mip, current_mip + 1.0);
            }
        }
    }
    
    return hit_result;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel_size = vec2<f32>(1.0) / vec2<f32>(textureDimensions(scene_color));
    let depth = textureSample(scene_depth, linear_sampler, input.uv);
    
    // Early out for sky (depth = 1.0)
    if (depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Sample material reflectivity from G-buffer
    let material = textureSample(material_gbuffer, linear_sampler, input.uv);
    let reflectivity = material.r;
    
    // Early out for non-reflective materials
    if (reflectivity < 0.01) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let world_pos = reconstruct_world_pos(input.uv, depth);
    // Use the G-buffer world normal instead of estimating from depth.
    // This is cheaper and much more stable near edges.
    let gbuf = textureSample(normal_gbuffer, linear_sampler, input.uv);
    let normal = decode_world_normal(gbuf.rgb);
    let view_dir = normalize(world_pos - camera.camera_pos);
    
    // Reflect view direction around normal
    let reflect_dir = reflect(view_dir, normal);
    
    // Offset start position along normal to avoid self-intersection
    // Use a larger offset (voxels are 1 unit, so offset by more than 1 to clear)
    let ray_start = world_pos + normal * 1.5;
    
    // Trace ray using HZB acceleration, pass start UV and depth for self-intersection rejection
    let hit_result = hzb_ray_march(ray_start, reflect_dir, input.uv, depth);
    
    if (hit_result.z > 0.0) {
        // Sample color at hit point
        let reflection_color = textureSample(scene_color, linear_sampler, hit_result.xy);
        
        // Fade based on distance from screen edges
        let edge_fade = min(
            min(hit_result.x, 1.0 - hit_result.x),
            min(hit_result.y, 1.0 - hit_result.y)
        );
        let edge_factor = smoothstep(0.0, 0.1, edge_fade);
        
        // Fade reflections at very long distances (less accurate)
        let distance_fade = 1.0 - smoothstep(50.0, 100.0, hit_result.w);
        
        // Modulate reflection strength by material reflectivity
        let final_alpha = edge_factor * distance_fade * reflectivity;
        
        return vec4<f32>(reflection_color.rgb, final_alpha);
    }
    
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
