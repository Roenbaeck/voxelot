// Voxel rendering shader for hierarchical chunks with global lighting

struct Uniforms {
    mvp: mat4x4<f32>,           // 64 bytes
    sun_direction: vec3<f32>,   // 12 bytes - Direction TO the sun (normalized)
    fog_density: f32,           // 4 bytes - Fog density (was _padding1)
    sun_color: vec3<f32>,       // 12 bytes - Sun color (e.g., warm yellow)
    _padding2: f32,             // 4 bytes
    ambient_color: vec3<f32>,   // 12 bytes - Ambient/sky color
    time_of_day: f32,           // 4 bytes - 0.0 = midnight, 0.5 = noon, 1.0 = midnight
    // _padding3: vec4<f32> in Rust (16 bytes) - shader doesn't need to declare trailing padding
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;


struct VertexOutputInstanced {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutputMesh {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

// Cube vertices (36 vertices for 6 faces)
const CUBE_VERTICES: array<vec3<f32>, 36> = array<vec3<f32>, 36>(
    // Front face
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
    
    // Back face
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    
    // Top face
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0),
    
    // Bottom face
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0),
    
    // Right face
    vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0),
    
    // Left face
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
);

const CUBE_NORMALS: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 0.0, 1.0),   // Front
    vec3<f32>(0.0, 0.0, -1.0),  // Back
    vec3<f32>(0.0, 1.0, 0.0),   // Top
    vec3<f32>(0.0, -1.0, 0.0),  // Bottom
    vec3<f32>(1.0, 0.0, 0.0),   // Right
    vec3<f32>(-1.0, 0.0, 0.0),  // Left
);

fn get_voxel_color(voxel_type: u32) -> vec3<f32> {
    switch (voxel_type) {
        case 1u: {
            return vec3<f32>(0.3, 0.7, 0.3); // Green ground
        }
        case 2u: {
            return vec3<f32>(0.7, 0.3, 0.3); // Red tower
        }
        case 3u: {
            return vec3<f32>(0.3, 0.3, 0.7); // Blue
        }
        case 4u: {
            return vec3<f32>(0.7, 0.7, 0.3); // Yellow
        }
        case 5u: {
            return vec3<f32>(0.7, 0.3, 0.7); // Magenta
        }
        case 6u: {
            return vec3<f32>(0.3, 0.7, 0.7); // Cyan
        }
        case 7u: {
            return vec3<f32>(0.5, 0.5, 0.5); // Gray wall
        }
        default: {
            return vec3<f32>(1.0, 1.0, 1.0); // White default
        }
    }
}

@vertex
fn vs_main(
    @location(0) instance_position: vec3<f32>,
    @location(1) instance_voxel_type: u32,
    @location(2) instance_scale: f32,
    @location(3) instance_custom_color: vec4<f32>,
    @location(4) vertex_position: vec3<f32>,
    @location(5) vertex_normal: vec3<f32>,
) -> VertexOutputInstanced {
    var output: VertexOutputInstanced;
    
    // Scale the vertex position, then add to instance position
    let scaled_vertex_pos = vertex_position * instance_scale;
    let world_pos = vec4<f32>(instance_position + scaled_vertex_pos, 1.0);
    output.position = uniforms.mvp * world_pos;
    
    // Use the per-vertex normal from the buffer
    output.normal = vertex_normal;
    
    // Use custom color if alpha > 0, otherwise use voxel type color
    if (instance_custom_color.a > 0.0) {
        output.color = instance_custom_color.rgb;
    } else {
        output.color = get_voxel_color(instance_voxel_type);
    }
    
    return output;
}

@fragment
fn fs_main(input: VertexOutputInstanced) -> @location(0) vec4<f32> {
    // Global sun/moon lighting from uniforms
    let sun_dir = normalize(uniforms.sun_direction);
    
    // Diffuse lighting from sun
    let diffuse = max(dot(input.normal, sun_dir), 0.0);
    let sun_contribution = diffuse * uniforms.sun_color;
    
    // Ambient lighting (sky/moonlight)
    let ambient = uniforms.ambient_color;
    
    // Combine lighting
    let lighting = ambient + sun_contribution;
    
    // Apply lighting to voxel color
    let color = input.color * lighting;
    
    // Atmospheric fog with dynamic density from uniforms
    let fog_color = vec3<f32>(0.7, 0.8, 0.9); // Light blue sky
    let distance = length(input.position.xyz);
    let fog_factor = 1.0 - exp(-uniforms.fog_density * distance);
    let final_color = mix(color, fog_color, fog_factor * 0.5);
    
    return vec4<f32>(final_color, 1.0);
}

// Mesh pipeline entry points -------------------------------------------------
@vertex
fn vs_mesh(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> VertexOutputMesh {
    var out: VertexOutputMesh;
    out.position = uniforms.mvp * vec4<f32>(position, 1.0);
    out.normal = normal;
    out.color = color.rgb;
    return out;
}

@fragment
fn fs_mesh(input: VertexOutputMesh) -> @location(0) vec4<f32> {
    let sun_dir = normalize(uniforms.sun_direction);
    let diffuse = max(dot(input.normal, sun_dir), 0.0);
    let sun_contribution = diffuse * uniforms.sun_color;
    let ambient = uniforms.ambient_color;
    let lighting = ambient + sun_contribution;
    let color = input.color * lighting;
    let fog_color = vec3<f32>(0.7, 0.8, 0.9);
    let distance = length(input.position.xyz);
    let fog_factor = 1.0 - exp(-uniforms.fog_density * distance);
    let final_color = mix(color, fog_color, fog_factor * 0.5);
    return vec4<f32>(final_color, 1.0);
}
