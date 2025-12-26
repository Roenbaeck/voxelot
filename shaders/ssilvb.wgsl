struct SsaoUniforms {
    sample_count: u32,
    slice_count: u32,
    sample_radius: f32,
    hit_thickness: f32,
    screen_width: f32,
    screen_height: f32,
    gi_indirect_scale: f32,
    gi_fade_distance: f32,
    gi_fade_range: f32,
    water_level: f32,
    water_visibility: f32,
    _pad2: f32,
    inverse_projection: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    grid_origin: vec3<i32>,
    _pad3: i32,
    grid_dims: vec3<i32>,
    _pad4: i32,
};

@group(0) @binding(0) var<uniform> ssao: SsaoUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var post_sampler: sampler;
// GI probe 3D volumes (Rgba16Float) with hardware trilinear filtering.
// Faces are stored separately: +X, -X, +Y, -Y, +Z, -Z
@group(0) @binding(3) var gi_probe_px: texture_3d<f32>;
@group(0) @binding(4) var gi_probe_nx: texture_3d<f32>;
@group(0) @binding(5) var gi_probe_py: texture_3d<f32>;
@group(0) @binding(6) var gi_probe_ny: texture_3d<f32>;
@group(0) @binding(7) var gi_probe_pz: texture_3d<f32>;
@group(0) @binding(8) var gi_probe_nz: texture_3d<f32>;
@group(0) @binding(9) var normal_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let pos = positions[vertex_index];
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    let uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

// Reconstruct view-space position from depth
fn reconstruct_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_z = depth;
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ndc = vec4<f32>(ndc_xy, ndc_z, 1.0);
    let view_pos = ssao.inverse_projection * ndc;
    return view_pos.xyz / view_pos.w;
}

// Reconstruct view-space position from linear depth
fn reconstruct_position_linear(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let x = ndc_xy.x * linear_depth * ssao.inverse_projection[0][0];
    let y = ndc_xy.y * linear_depth * ssao.inverse_projection[1][1];
    return vec3<f32>(x, y, -linear_depth);
}

// Fetch depth directly from texture
fn fetch_depth(coord: vec2<i32>) -> f32 {
    return textureLoad(depth_tex, coord, 0);
}

// Interleaved Gradient Noise
fn ign(uv: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(uv, vec2<f32>(0.06711056, 0.00583715))));
}

fn fast_acos(x: f32) -> f32 {
    let out_val = -0.156583 * abs(x) + HALF_PI;
    let res = out_val * sqrt(1.0 - abs(x));
    return select(PI - res, res, x >= 0.0);
}

fn count_bits(value: u32) -> u32 {
    var v = value;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    return (((v + (v >> 4u)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24u;
}

fn sample_grid_irradiance(world_pos: vec3<f32>, normal: vec3<f32>, camera_pos: vec3<f32>) -> vec3<f32> {
    // Bias position against normal to sample from same probe but opposite bin
    let biased_pos = world_pos - normal * 0.5;

    // Convert world pos to grid coords
    let grid_coord = biased_pos / 16.0 - vec3<f32>(ssao.grid_origin) - 0.5;
    
    let dims_i = ssao.grid_dims;
    let dims = vec3<f32>(f32(dims_i.x), f32(dims_i.y), f32(dims_i.z));

    // Outside the probe volume: no GI.
    // We check against (dims - 1) because probes are defined at integer texel coordinates.
    if (grid_coord.x < 0.0 || grid_coord.y < 0.0 || grid_coord.z < 0.0 ||
        grid_coord.x > (dims.x - 1.0) || grid_coord.y > (dims.y - 1.0) || grid_coord.z > (dims.z - 1.0)) {
        return vec3<f32>(0.0);
    }

    // Map grid_coord (0..dims-1) to normalized UVW for texel-center sampling.
    let uvw = clamp((grid_coord + vec3<f32>(0.5)) / dims, vec3<f32>(0.0), vec3<f32>(1.0));

    let w_x = normal.x * normal.x;
    let w_y = normal.y * normal.y;
    let w_z = normal.z * normal.z;

    let ir_x = select(
        textureSample(gi_probe_nx, post_sampler, uvw).rgb,
        textureSample(gi_probe_px, post_sampler, uvw).rgb,
        normal.x > 0.0
    );
    let ir_y = select(
        textureSample(gi_probe_ny, post_sampler, uvw).rgb,
        textureSample(gi_probe_py, post_sampler, uvw).rgb,
        normal.y > 0.0
    );
    let ir_z = select(
        textureSample(gi_probe_nz, post_sampler, uvw).rgb,
        textureSample(gi_probe_pz, post_sampler, uvw).rgb,
        normal.z > 0.0
    );

    let total_irradiance = ir_x * w_x + ir_y * w_y + ir_z * w_z;
    
    // Calculate distance to edge of grid for smooth fading
    // grid_coord is in chunk units (0..dims)
    // We fade out over the last 2 chunks (32 units) to avoid popping
    let dist_to_edge = min(
        min(grid_coord.x, f32(dims.x) - grid_coord.x),
        min(min(grid_coord.y, f32(dims.y) - grid_coord.y),
            min(grid_coord.z, f32(dims.z) - grid_coord.z))
    );
    
    let fade = smoothstep(0.0, 2.0, dist_to_edge);
    
    // Distance-based fade to prevent popping when entering lit areas
    // Fade GI to zero as we approach the fade distance
    let dist_to_camera = length(world_pos - camera_pos);
    let fade_start = ssao.gi_fade_distance - ssao.gi_fade_range;
    let distance_fade = 1.0 - smoothstep(fade_start, ssao.gi_fade_distance, dist_to_camera);
    
    // Combine both fades
    let total_fade = fade * distance_fade;

    return total_irradiance * total_fade;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Sample G-buffer normal and linear depth
    let normal_sample = textureSample(normal_tex, post_sampler, uv);
    let linear_depth = normal_sample.w;
    
    // If linear depth is 0, it's likely the skybox or background
    if (linear_depth <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let inv_proj_00 = ssao.inverse_projection[0][0];
    let inv_proj_11 = ssao.inverse_projection[1][1];
    let ndc_xy_pixel = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let view_pos = vec3<f32>(ndc_xy_pixel.x * linear_depth * inv_proj_00, ndc_xy_pixel.y * linear_depth * inv_proj_11, -linear_depth);

    // Sample world-space normal from G-buffer and decode from [0,1] to [-1,1]
    let world_normal = normalize(normal_sample.rgb * 2.0 - 1.0);
    // Transform world normal to view-space for GTAO calculations
    let view_mat_rot = transpose(mat3x3<f32>(
        ssao.inverse_view[0].xyz,
        ssao.inverse_view[1].xyz,
        ssao.inverse_view[2].xyz
    ));
    let normal = normalize(view_mat_rot * world_normal);
    let view_vec = normalize(-view_pos); // View vector pointing to camera (0,0,0) in view space

    // Reconstruct world position
    let world_pos_4 = ssao.inverse_view * vec4<f32>(view_pos, 1.0);
    let world_pos = world_pos_4.xyz / world_pos_4.w;
    
    let screen_size = vec2<f32>(ssao.screen_width, ssao.screen_height);
    let inv_screen_size = 1.0 / screen_size;
    let frag_coord = uv * screen_size;
    
    // Random rotation
    let noise = ign(frag_coord);
    
    let sample_count = f32(ssao.sample_count);
    let slice_count = f32(ssao.slice_count);
    let radius = ssao.sample_radius;
    
    var visibility = 0.0;
    
    let proj_scale = inv_proj_11 * ssao.screen_height;
    let screen_radius = (radius * proj_scale) / -view_pos.z;
    let step_ratio = pow(screen_radius, 1.0 / sample_count);
    let hit_thickness_sq = ssao.hit_thickness * ssao.hit_thickness;

    for (var slice = 0u; slice < ssao.slice_count; slice = slice + 1u) {
        let phi = (2.0 * PI / slice_count) * (f32(slice) + noise);
        let slice_dir = vec2<f32>(cos(phi), sin(phi));
        
        let search_dir = vec2<f32>(slice_dir.x, -slice_dir.y);
        let slice_dir_vs = vec3<f32>(slice_dir, 0.0);
        let plane_n = normalize(cross(slice_dir_vs, view_vec));
        let proj_n = normal - plane_n * dot(normal, plane_n);
        let proj_n_len = length(proj_n);
        
        var cos_n = 0.0;
        var n_angle = 0.0;

        if (proj_n_len > 0.001) {
            let pn = proj_n / proj_n_len;
            cos_n = dot(pn, view_vec);
            let t = cross(plane_n, pn);
            let sgn = select(1.0, -1.0, dot(view_vec, t) < 0.0);
            n_angle = sgn * fast_acos(cos_n);
        }
        
        var occlusion_bits = 0u;
        
        for (var side = 0u; side < 2u; side = side + 1u) {
            if (occlusion_bits == 0xFFFFFFFFu) { break; }

            let direction = select(-1.0, 1.0, side == 0u);
            let ray_dir = search_dir * direction;
            var current_step = pow(step_ratio, select(noise, 1.0 - noise, side == 1u));
            
            for (var s = 0u; s < 64u; s = s + 1u) {
                if (s >= ssao.sample_count) { break; } 

                let sample_uv = uv + (ray_dir * current_step) * inv_screen_size;
                current_step *= step_ratio;
                
                if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) { break; }
                
                let sample_normal = textureSampleLevel(normal_tex, post_sampler, sample_uv, 0);
                let sample_linear_depth = sample_normal.w;
                
                if (sample_linear_depth <= 0.0) { continue; }
                
                let sample_ndc_xy = vec2<f32>(sample_uv.x * 2.0 - 1.0, 1.0 - sample_uv.y * 2.0);
                let sample_pos = vec3<f32>(sample_ndc_xy.x * sample_linear_depth * inv_proj_00, sample_ndc_xy.y * sample_linear_depth * inv_proj_11, -sample_linear_depth);
                
                let delta = sample_pos - view_pos;
                let dist_sq = dot(delta, delta);
                
                let d_dot_v = dot(delta, view_vec);
                let inv_dist = inverseSqrt(max(dist_sq, 1e-8));
                let horizon_cos = d_dot_v * inv_dist;
                let horizon_angle_rel = fast_acos(horizon_cos) * direction;
                
                let back_horizon_num = d_dot_v - ssao.hit_thickness;
                let back_horizon_den_sq = dist_sq - 2.0 * d_dot_v * ssao.hit_thickness + hit_thickness_sq;
                let back_horizon_cos = back_horizon_num * inverseSqrt(back_horizon_den_sq);
                let back_horizon_angle = fast_acos(back_horizon_cos) * direction;
                
                let h1 = clamp((horizon_angle_rel + n_angle) / PI + 0.5, 0.0, 1.0);
                let h2 = clamp((back_horizon_angle + n_angle) / PI + 0.5, 0.0, 1.0);
                
                let min_h = min(h1, h2);
                let max_h = max(h1, h2);
                
                let start_bit = u32(min_h * 32.0);
                let end_bit = u32(max_h * 32.0);
                
                if (start_bit < 32u) {
                    let count = min(end_bit - start_bit, 32u - start_bit);
                    if (count > 0u) {
                        let mask = (0xFFFFFFFFu >> (32u - count)) << start_bit;
                        occlusion_bits = occlusion_bits | mask;
                    }
                }

                if (occlusion_bits == 0xFFFFFFFFu) { break; }
            }

            if (occlusion_bits == 0xFFFFFFFFu) { break; }
        }
        
        let occluded_fraction = f32(count_bits(occlusion_bits)) / 32.0;
        visibility += 1.0 - occluded_fraction;
    }
    
    visibility /= slice_count;
    visibility = pow(visibility, 2.0);
    
    // Extract camera position from inverse view matrix (translation component)
    let camera_pos = ssao.inverse_view[3].xyz;
    
    // Sample GI from probes
    // Scale factor is configurable (gi_indirect_scale in config.toml)
    var indirect_light = sample_grid_irradiance(world_pos, world_normal, camera_pos) * ssao.gi_indirect_scale;
    
    // Fade AO and GI for underwater surfaces to prevent them being visible through water
    let underwater_depth = ssao.water_level - world_pos.y;
    if (underwater_depth > 0.0 && ssao.water_visibility > 0.0) {
        // Aggressive fade: at max visibility depth, AO becomes 1.0 (no occlusion) and GI becomes 0
        let fade_factor = clamp(underwater_depth / ssao.water_visibility, 0.0, 1.0);
        let fade_curve = pow(fade_factor, 0.5); // Faster fade
        
        // Fade AO to 1.0 (no darkening) underwater
        visibility = mix(visibility, 1.0, fade_curve);
        
        // Fade GI to zero underwater
        indirect_light = indirect_light * (1.0 - fade_curve);
    }
    
    return vec4<f32>(indirect_light, visibility);
}
