struct InstanceInput {
    position : vec3<f32>,
    scale : f32,
    custom_color : vec4<f32>,
    voxel_type : u32,
    flags : u32,
    _padding : vec2<u32>,
};

struct CullParams {
    camera_position : vec3<f32>,
    candidate_count : u32,
    camera_forward : vec3<f32>,
    _pad0 : u32,
    near_plane : f32,
    far_plane : f32,
    lod_render_distance : f32,
    _pad1 : f32,
};

@group(0) @binding(0)
var<storage, read_write> candidates : array<InstanceInput>;

@group(0) @binding(1)
var<uniform> params : CullParams;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;
    if (index >= params.candidate_count) {
        return;
    }

    let instance = candidates[index];
    let to_instance = instance.position - params.camera_position;
    // Avoid expensive length() by comparing squared distances
    let dist_sq = dot(to_instance, to_instance);
    let near_sq = params.near_plane * params.near_plane;
    let far_sq = params.far_plane * params.far_plane;
    let lod_sq = params.lod_render_distance * params.lod_render_distance;

    let within_depth = dist_sq >= near_sq && dist_sq <= far_sq;
    let within_lod = dist_sq <= lod_sq;

    // Assume camera_forward is normalized by CPU-side camera; avoid normalize() in shader
    let in_front = dot(params.camera_forward, to_instance) > -instance.scale;

    let visible = within_depth && within_lod && in_front;

    if (visible) {
        candidates[index].flags = 1u;
    } else {
        candidates[index].flags = 0u;
    }
}
