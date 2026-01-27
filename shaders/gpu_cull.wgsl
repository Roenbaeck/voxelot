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
    camera_right : vec3<f32>,
    _pad_r0 : u32,
    camera_up : vec3<f32>,
    _pad_u0 : u32,
    fov_tan : f32,
    aspect : f32,
    screen_width : f32,
    screen_height : f32,
    fog_density : f32,
    skybox_brightness : f32,
    impostor_pixel_threshold : f32,
    impostor_pixel_size : f32,
    lod_render_distance : f32,
    detail_cull_distance : f32,
    envelope_distance : f32,
    envelope_fade_range : f32,
    hzb_enabled : u32,
    max_hzb_mip : u32,
    _pad3 : f32,
    // View-projection matrix (column-major)
    view_proj : mat4x4<f32>,
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

struct ImpostorInstance {
    position : vec3<f32>,
    _pad0 : f32,
    color : vec4<f32>,
    emissive : vec4<f32>,
};

@group(0) @binding(6)
var<storage, read_write> impostor_indirect : DrawIndirectArgs;

@group(0) @binding(7)
var<storage, read_write> impostor_instances : array<ImpostorInstance>;

@group(0) @binding(8)
var hzb_tex : texture_2d<f32>;

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

    // Assume camera_forward is normalized by CPU-side camera; avoid normalize() in shader
    // Project half_scale onto forward vector to get effective radius along view direction
    let radius = dot(abs(params.camera_forward), half_scale);
    let in_front = dot(params.camera_forward, to_instance) > -radius;

    var visible = within_depth && in_front;

    if (visible) {
        candidates[index].flags = candidates[index].flags | 4u; // Mark visible for debug
        
        let detail_sq = params.detail_cull_distance * params.detail_cull_distance;
        // Use envelope_fade_range to bias the threshold so detail remains active through the fade band
        let env = params.envelope_distance;
        let fade = params.envelope_fade_range;
        let env_plus = env + fade;
        let env_plus_sq = env_plus * env_plus;

        let has_mesh = (instance.flags & 1u) != 0u;
        let has_envelope = (instance.flags & 2u) != 0u;
        let cpu_prepopulated = (instance.flags & 4u) != 0u;
        
        // Priority:
        // 1. Detail Mesh (if near enough AND exists)
        // 2. Envelope Mesh (if exists AND (far enough OR detail missing))
        // 3. Impostor (if projected size is tiny)
        // 4. Fallback (if neither exists)
        
        // Revised logic: extend detail usage out to (env + fade) so the detail mesh
        // can smoothly fade towards envelope shading over the configured fade range.
        var use_detail = has_mesh && (dist_sq <= env_plus_sq);
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

        // CRITICAL FIX: If CPU has already prepopulated individual voxels for this chunk (fallback),
        // we MUST NOT render the mesh or envelope, otherwise we get z-fighting artifacts.
        if (cpu_prepopulated) {
            use_detail = false;
            use_envelope = false;
        }

        // Approximate projected size in pixels for impostor decision.
        let z_cam = dot(to_instance, params.camera_forward);
        let approx_radius = max(half_scale.x, max(half_scale.y, half_scale.z));
        let denom = max(z_cam * params.fov_tan, 1e-4);
        let diameter_px = (approx_radius * params.screen_height) / denom;
        let use_impostor = (diameter_px <= params.impostor_pixel_threshold) && !use_detail && !use_envelope && !cpu_prepopulated;

        // Always cull chunks behind the camera (regardless of HZB)
        if (z_cam <= 0.0) {
            visible = false;
        }

        // HZB occlusion test (conditional on hzb_enabled && visible)
        if (params.hzb_enabled != 0u && visible) {
            // Calculate screen-space AABB by projecting 8 corners of chunk
            let half_scale = instance.scale * 0.5;
            var min_screen = vec2<f32>(1e10);
            var max_screen = vec2<f32>(-1e10);
            var min_depth = 1e10;
            
            // Project all 8 corners
            for (var i = 0; i < 8; i++) {
                let corner_offset = vec3<f32>(
                    select(-half_scale.x, half_scale.x, (i & 1) != 0),
                    select(-half_scale.y, half_scale.y, (i & 2) != 0),
                    select(-half_scale.z, half_scale.z, (i & 4) != 0)
                );
                let world_pos = instance.position + corner_offset;
                
                // Project to clip space
                let clip_pos = params.view_proj * vec4<f32>(world_pos, 1.0);
                
                // Perspective divide to NDC
                if (clip_pos.w > 0.0) {
                    let ndc = clip_pos.xyz / clip_pos.w;
                    
                    // Convert to screen coords (0 to width/height)
                    let screen_x = (ndc.x * 0.5 + 0.5) * params.screen_width;
                    let screen_y = (ndc.y * 0.5 + 0.5) * params.screen_height;
                    
                    min_screen = min(min_screen, vec2<f32>(screen_x, screen_y));
                    max_screen = max(max_screen, vec2<f32>(screen_x, screen_y));
                    
                    // Track minimum (nearest) depth
                    min_depth = min(min_depth, ndc.z);
                }
            }

            // If the projected bounds extend outside the screen, HZB cannot conservatively
            // determine occlusion for the missing region. In that case, skip HZB culling.
            let offscreen = (min_screen.x < 0.0) || (min_screen.y < 0.0) ||
                (max_screen.x > (params.screen_width - 1.0)) || (max_screen.y > (params.screen_height - 1.0));
            if (offscreen) {
                // Keep visible without HZB test
            } else {
            
            // Clamp AABB to screen bounds
            min_screen = clamp(min_screen, vec2<f32>(0.0), vec2<f32>(params.screen_width - 1.0, params.screen_height - 1.0));
            max_screen = clamp(max_screen, vec2<f32>(0.0), vec2<f32>(params.screen_width - 1.0, params.screen_height - 1.0));
            
            // Calculate AABB size
            let aabb_size = max_screen - min_screen;
            let max_size = max(aabb_size.x, aabb_size.y);
            
            // Select mip level where AABB covers ~2-4 pixels
            let mip_level = clamp(
                i32(log2(max_size / 2.0)),
                0,
                i32(params.max_hzb_mip)
            );
            
            // Sample HZB at selected mip level (center of AABB)
            let sample_pos = (min_screen + max_screen) * 0.5;
            let mip_width = max(1.0, params.screen_width / f32(1 << u32(mip_level)));
            let mip_height = max(1.0, params.screen_height / f32(1 << u32(mip_level)));
            let mip_u = i32(clamp(sample_pos.x / f32(1 << u32(mip_level)), 0.0, mip_width - 1.0));
            let mip_v = i32(clamp(sample_pos.y / f32(1 << u32(mip_level)), 0.0, mip_height - 1.0));
            
            let hzb_depth = textureLoad(hzb_tex, vec2<i32>(mip_u, mip_v), mip_level).x;
            
            // Conservative test: cull if chunk's NEAREST point is farther than HZB's FURTHEST point
            if (min_depth > hzb_depth) {
                // Fully occluded
                visible = false;
            }
            }
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
        } else if (use_impostor) {
            if (!cpu_prepopulated) {
                let idx = atomicAdd(&impostor_indirect.instance_count, 1u);
                var imp : ImpostorInstance;
                let has_color = instance.custom_color.a > 0.0;
                imp.position = instance.position + half_scale;
                imp._pad0 = 0.0;
                imp.color = select(vec4<f32>(0.4, 0.4, 0.45, 0.8), instance.custom_color, has_color);
                imp.emissive = instance.emissive;
                impostor_instances[idx] = imp;
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
