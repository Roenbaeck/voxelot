struct Uniforms {
    view_proj: mat4x4<f32>,
    pos0: vec4<f32>, // xyz = world pos, w = scale
    color0: vec4<f32>,
    pos1: vec4<f32>,
    color1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) instance_id: u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    // 12 lines for a cube = 24 vertices
    var pos = array<vec3<f32>, 24>(
        // Bottom face
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0),
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0),
        // Top face
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
        vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
        // Vertical lines
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0)
    );

    let use_outer = instance_index == 1u;
    let pos_data = select(uniforms.pos0, uniforms.pos1, use_outer);
    let world_pos = (pos[vertex_index] * pos_data.w) + pos_data.xyz;
    
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.instance_id = instance_index;
    return out;
}

@fragment
fn fs_main(@location(0) instance_id: u32) -> @location(0) vec4<f32> {
    let use_outer = instance_id == 1u;
    return select(uniforms.color0, uniforms.color1, use_outer);
}
