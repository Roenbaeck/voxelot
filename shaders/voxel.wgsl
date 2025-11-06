// Voxel rendering shader for hierarchical chunks

struct Uniforms {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) voxel_type: u32,
};

struct VertexOutput {
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
    @builtin(vertex_index) vertex_index: u32,
    input: VertexInput,
) -> VertexOutput {
    var output: VertexOutput;
    
    // Get cube vertex for this vertex index
    let vertex = CUBE_VERTICES[vertex_index];
    
    // Position the cube at the voxel's world position
    let world_pos = vec4<f32>(input.position + vertex, 1.0);
    output.position = uniforms.mvp * world_pos;
    
    // Get normal for this face
    let face_index = vertex_index / 6u;
    output.normal = CUBE_NORMALS[face_index];
    
    // Get color based on voxel type
    output.color = get_voxel_color(input.voxel_type);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ambient = 0.3;
    let diffuse = max(dot(input.normal, light_dir), 0.0) * 0.7;
    let lighting = ambient + diffuse;
    
    let color = input.color * lighting;
    return vec4<f32>(color, 1.0);
}
