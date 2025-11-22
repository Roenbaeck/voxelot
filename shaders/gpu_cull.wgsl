struct InstanceInput {
    position : vec3<f32>,
    scale : vec3<f32>,
    custom_color : vec4<f32>,
    emissive : vec4<f32>,
    voxel_type : u32,
    flags : u32, // bit 0: has_mesh, bit 1: has_envelope
    mesh_index : u32,
    envelope_index : u32,
};

struct DrawIndexedIndirectArgs {
    vertex_count : u32,
    instance_count : u32,
    first_index : u32,
    base_vertex : i32,
    first_instance : u32,
};

struct DrawIndirectArgs {
    vertex_count : u32,
    instance_count : atomic<u32>,
    first_vertex : u32,
    first_instance : u32,
};

struct CullParams {
    camera_position : vec3<f32>,
    candidate_count : u32,
    camera_forward : vec3<f32>,
    _pad0 : u32,
    near_plane : f32,
    far_plane : f32,
    lod_render_distance : f32,
    detail_cull_distance : f32,
    envelope_distance : f32,
    _pad1 : f32,
    _pad2 : f32,
    _pad3 : f32,
};

@group(0) @binding(0)
var<storage, read_write> candidates : array<InstanceInput>;

@group(0) @binding(1)
var<uniform> params : CullParams;

@group(0) @binding(2)
var<storage, read_write> mesh_indirect : array<DrawIndexedIndirectArgs>;

@group(0) @binding(3)
var<storage, read_write> fallback_indirect : DrawIndirectArgs;

struct VoxelInstanceRaw {
    position : vec3<f32>,
    voxel_type : u32,
    scale : vec3<f32>,
    ao_factor : f32,
    custom_color : vec4<f32>,
    emissive : vec4<f32>,
};

@group(0) @binding(4)
var<storage, read_write> fallback_instances : array<VoxelInstanceRaw>;

@group(0) @binding(5)
var<storage, read_write> envelope_indirect : array<DrawIndexedIndirectArgs>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;
    if (index >= params.candidate_count) {
        return;
    }

    let instance = candidates[index];
    let half_scale = instance.scale * 0.5;
    let instance_center = instance.position + half_scale;
    let to_instance = instance_center - params.camera_position;
    // Avoid expensive length() by comparing squared distances
    let dist_sq = dot(to_instance, to_instance);
    let near_sq = params.near_plane * params.near_plane;
    let far_sq = params.far_plane * params.far_plane;
    let lod_sq = params.lod_render_distance * params.lod_render_distance;

    let within_depth = dist_sq >= near_sq && dist_sq <= far_sq;
    let within_lod = dist_sq <= lod_sq;

    // Assume camera_forward is normalized by CPU-side camera; avoid normalize() in shader
    // Project half_scale onto forward vector to get effective radius along view direction
    let radius = dot(abs(params.camera_forward), half_scale);
    let in_front = dot(params.camera_forward, to_instance) > -radius;

    let visible = within_depth && within_lod && in_front;

    if (visible) {
        candidates[index].flags = candidates[index].flags | 4u; // Mark visible for debug
        
        let detail_sq = params.detail_cull_distance * params.detail_cull_distance;
        let envelope_sq = params.envelope_distance * params.envelope_distance;
        
        let has_mesh = (instance.flags & 1u) != 0u;
        let has_envelope = (instance.flags & 2u) != 0u;
        let cpu_prepopulated = (instance.flags & 4u) != 0u;
        
        // Priority:
        // 1. Detail Mesh (if near enough AND exists)
        // 2. Envelope Mesh (if exists AND (far enough OR detail missing))
        // 3. Fallback (if neither exists)
        
        // Note: Original logic was "use envelope if dist > envelope_dist".
        // But if envelope is missing, we might want detail mesh even if far?
        // Or if detail is missing, we want envelope even if near?
        
        var use_detail = has_mesh && (dist_sq <= envelope_sq);
        var use_envelope = has_envelope && (!use_detail);
        
        // If we want to use detail but don't have it, try envelope
        if (!has_mesh && has_envelope) {
            use_envelope = true;
            use_detail = false;
        }
        
        // If we want to use envelope but don't have it, try detail (if within render distance)
        if (!has_envelope && has_mesh && dist_sq <= lod_sq) {
            use_detail = true;
            use_envelope = false;
        }

        if (use_detail) {
            // Enable mesh draw
            mesh_indirect[instance.mesh_index].instance_count = 1u;
            if (has_envelope) {
                envelope_indirect[instance.envelope_index].instance_count = 0u;
            }
        } else if (use_envelope) {
            // Enable envelope draw
            envelope_indirect[instance.envelope_index].instance_count = 1u;
            if (has_mesh) {
                mesh_indirect[instance.mesh_index].instance_count = 0u;
            }
        } else {
            // Add to fallback instances
            if (!cpu_prepopulated) {
                let idx = atomicAdd(&fallback_indirect.instance_count, 1u);
                var raw : VoxelInstanceRaw;
                raw.position = instance.position;
                raw.voxel_type = instance.voxel_type;
                raw.scale = instance.scale;
                raw.ao_factor = 1.0;
                raw.custom_color = instance.custom_color;
                raw.emissive = instance.emissive;
                fallback_instances[idx] = raw;
            }
            
            
            if (has_mesh) {
                mesh_indirect[instance.mesh_index].instance_count = 0u;
            }
            if (has_envelope) {
                envelope_indirect[instance.envelope_index].instance_count = 0u;
            }
        }
    } else {
        candidates[index].flags = 0u;
        let has_mesh = (instance.flags & 1u) != 0u;
        let has_envelope = (instance.flags & 2u) != 0u;

        if (has_mesh) {
            mesh_indirect[instance.mesh_index].instance_count = 0u;
        }
        if (has_envelope) {
            envelope_indirect[instance.envelope_index].instance_count = 0u;
        }
    }
}
