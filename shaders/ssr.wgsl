// Screen Space Reflections shader

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
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: SSRParams;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var linear_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

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
    // Convert UV and depth to NDC
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    
    // Unproject to view space
    let view_pos = camera.inverse_proj * vec4<f32>(ndc, 1.0);
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

// Estimate normal from depth buffer using screen-space derivatives
fn estimate_normal(uv: vec2<f32>, depth: f32, texel_size: vec2<f32>) -> vec3<f32> {
    let center_pos = reconstruct_world_pos(uv, depth);
    
    // Sample neighboring depths
    let depth_right = textureSample(scene_depth, linear_sampler, uv + vec2<f32>(texel_size.x, 0.0));
    let depth_up = textureSample(scene_depth, linear_sampler, uv + vec2<f32>(0.0, -texel_size.y));
    
    let pos_right = reconstruct_world_pos(uv + vec2<f32>(texel_size.x, 0.0), depth_right);
    let pos_up = reconstruct_world_pos(uv + vec2<f32>(0.0, -texel_size.y), depth_up);
    
    let dx = pos_right - center_pos;
    let dy = pos_up - center_pos;
    
    return normalize(cross(dy, dx));
}

// Ray marching SSR
fn trace_ray(start_pos: vec3<f32>, ray_dir: vec3<f32>) -> vec3<f32> {
    var hit_uv = vec3<f32>(-1.0);
    
    var current_pos = start_pos;
    let step = ray_dir * params.step_size;
    
    // Coarse ray march
    for (var i = 0u; i < params.max_steps; i++) {
        current_pos += step;
        
        let screen_pos = world_to_screen(current_pos);
        let uv = screen_pos.xy;
        
        // Check if out of screen bounds
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }
        
        // Sample depth at this screen position
        let depth_at_uv = textureSample(scene_depth, linear_sampler, uv);
        let surface_pos = reconstruct_world_pos(uv, depth_at_uv);
        
        // Check if ray is behind surface
        let ray_depth = distance(camera.camera_pos, current_pos);
        let surface_depth = distance(camera.camera_pos, surface_pos);
        
        if (ray_depth > surface_depth && ray_depth - surface_depth < params.thickness) {
            hit_uv = vec3<f32>(uv, 1.0);
            
            // Binary refinement
            var refined_pos = current_pos - step;
            var refined_step = step;
            
            for (var j = 0u; j < params.max_binary_steps; j++) {
                refined_step *= 0.5;
                refined_pos += refined_step;
                
                let refined_screen = world_to_screen(refined_pos);
                let refined_uv = refined_screen.xy;
                let refined_depth = textureSample(scene_depth, linear_sampler, refined_uv);
                let refined_surface = reconstruct_world_pos(refined_uv, refined_depth);
                
                let refined_ray_depth = distance(camera.camera_pos, refined_pos);
                let refined_surface_depth = distance(camera.camera_pos, refined_surface);
                
                if (refined_ray_depth > refined_surface_depth) {
                    refined_pos -= refined_step;
                }
            }
            
            hit_uv = vec3<f32>(world_to_screen(refined_pos).xy, 1.0);
            break;
        }
    }
    
    return hit_uv;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel_size = vec2<f32>(1.0) / vec2<f32>(textureDimensions(scene_color));
    let depth = textureSample(scene_depth, linear_sampler, input.uv);
    
    // Early out for sky (depth = 1.0)
    if (depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let world_pos = reconstruct_world_pos(input.uv, depth);
    let normal = estimate_normal(input.uv, depth, texel_size);
    let view_dir = normalize(world_pos - camera.camera_pos);
    
    // Reflect view direction around normal
    let reflect_dir = reflect(view_dir, normal);
    
    // Trace ray
    let hit_uv = trace_ray(world_pos, reflect_dir);
    
    if (hit_uv.z > 0.0) {
        // Sample color at hit point
        let reflection_color = textureSample(scene_color, linear_sampler, hit_uv.xy);
        
        // Fade based on distance from screen edges
        let edge_fade = min(
            min(hit_uv.x, 1.0 - hit_uv.x),
            min(hit_uv.y, 1.0 - hit_uv.y)
        );
        let edge_factor = smoothstep(0.0, 0.1, edge_fade);
        
        return vec4<f32>(reflection_color.rgb, edge_factor);
    }
    
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
