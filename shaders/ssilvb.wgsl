struct SsaoUniforms {
    sample_count: u32,
    slice_count: u32,
    sample_radius: f32,
    hit_thickness: f32,
    screen_width: f32,
    screen_height: f32,
    _pad0: f32,
    _pad1: f32,
    inverse_projection: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    grid_origin: vec3<i32>,
    _pad2: i32,
    grid_dims: vec3<i32>,
    _pad3: i32,
};

struct GiProbe {
    position: vec4<f32>,
    light_data: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var<uniform> ssao: SsaoUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var post_sampler: sampler;
@group(0) @binding(3) var<storage, read> gi_probes: array<GiProbe>;

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

// Fetch depth directly from texture
fn fetch_depth(coord: vec2<i32>) -> f32 {
    return textureLoad(depth_tex, coord, 0);
}

// Reconstruct normal from depth buffer using central differences
fn compute_normal_from_depth(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(ssao.screen_width, ssao.screen_height);
    let p = vec2<i32>(uv * size);
    
    let c0 = fetch_depth(p);
    
    // Check for edge cases
    if (c0 >= 1.0) { return vec3<f32>(0.0, 0.0, 1.0); }

    let l1 = fetch_depth(p - vec2<i32>(1, 0));
    let r1 = fetch_depth(p + vec2<i32>(1, 0));
    let b1 = fetch_depth(p - vec2<i32>(0, 1));
    let t1 = fetch_depth(p + vec2<i32>(0, 1));
    
    let l2 = fetch_depth(p - vec2<i32>(2, 0));
    let r2 = fetch_depth(p + vec2<i32>(2, 0));
    let b2 = fetch_depth(p - vec2<i32>(0, 2));
    let t2 = fetch_depth(p + vec2<i32>(0, 2));
    
    let dl = abs((2.0 * l1 - l2) - c0);
    let dr = abs((2.0 * r1 - r2) - c0);
    let db = abs((2.0 * b1 - b2) - c0);
    let dt = abs((2.0 * t1 - t2) - c0);
    
    let ce = reconstruct_position(uv, c0);
    
    let dpdx = select(
        -ce + reconstruct_position(uv + vec2<f32>(1.0 / size.x, 0.0), r1),
        ce - reconstruct_position(uv - vec2<f32>(1.0 / size.x, 0.0), l1),
        dl < dr
    );
    
    let dpdy = select(
        -ce + reconstruct_position(uv + vec2<f32>(0.0, 1.0 / size.y), t1),
        ce - reconstruct_position(uv - vec2<f32>(0.0, 1.0 / size.y), b1),
        db < dt
    );

    return normalize(cross(dpdx, dpdy));
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

fn get_probe_irradiance(probe_idx: u32, normal: vec3<f32>) -> vec3<f32> {
    let probe = gi_probes[probe_idx];
    
    let w_x = normal.x * normal.x;
    let w_y = normal.y * normal.y;
    let w_z = normal.z * normal.z;
    
    let idx_x = select(1u, 0u, normal.x > 0.0);
    let idx_y = select(3u, 2u, normal.y > 0.0);
    let idx_z = select(5u, 4u, normal.z > 0.0);
    
    return probe.light_data[idx_x].rgb * w_x + 
           probe.light_data[idx_y].rgb * w_y + 
           probe.light_data[idx_z].rgb * w_z;
}

fn sample_grid_irradiance(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // Apply normal bias to avoid sampling inside walls
    // Push the sample point 0.5 units along the normal (half a block)
    let biased_pos = world_pos + normal * 0.5;

    // Convert world pos to grid coords
    // Grid origin is in chunks (16 units)
    // Probe is at center of chunk (+8.0)
    // grid_coord = (biased_pos - (grid_origin * 16.0 + 8.0)) / 16.0
    //            = biased_pos / 16.0 - grid_origin - 0.5
    
    let grid_coord = biased_pos / 16.0 - vec3<f32>(ssao.grid_origin) - 0.5;
    
    let base = floor(grid_coord);
    let frac = grid_coord - base;
    
    let dims = ssao.grid_dims;
    
    var total_irradiance = vec3<f32>(0.0);
    var total_weight = 0.0;
    
    // Trilinear interpolation
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let offset = vec3<f32>(f32(x), f32(y), f32(z));
                let coord = vec3<i32>(base + offset);
                
                // Check bounds
                if (coord.x >= 0 && coord.x < dims.x &&
                    coord.y >= 0 && coord.y < dims.y &&
                    coord.z >= 0 && coord.z < dims.z) {
                    
                    let idx = u32(coord.x + coord.y * dims.x + coord.z * dims.x * dims.y);
                    let weight = (select(1.0 - frac.x, frac.x, x == 1) *
                                  select(1.0 - frac.y, frac.y, y == 1) *
                                  select(1.0 - frac.z, frac.z, z == 1));
                                  
                    total_irradiance += get_probe_irradiance(idx, normal) * weight;
                    total_weight += weight;
                }
            }
        }
    }
    
    // Calculate distance to edge of grid for smooth fading
    // grid_coord is in chunk units (0..dims)
    // We fade out over the last 2 chunks (32 units) to avoid popping
    let dist_to_edge = min(
        min(grid_coord.x, f32(dims.x) - grid_coord.x),
        min(min(grid_coord.y, f32(dims.y) - grid_coord.y),
            min(grid_coord.z, f32(dims.z) - grid_coord.z))
    );
    
    let fade = smoothstep(0.0, 2.0, dist_to_edge);

    if (total_weight > 0.0) {
        return (total_irradiance / total_weight) * fade;
    } else {
        return vec3<f32>(0.0);
    }
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, post_sampler, uv);
    if (depth >= 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let view_pos = reconstruct_position(uv, depth);
    let normal = compute_normal_from_depth(uv);
    let view_vec = normalize(-view_pos); // View vector pointing to camera (0,0,0) in view space

    // Reconstruct world position
    let world_pos_4 = ssao.inverse_view * vec4<f32>(view_pos, 1.0);
    let world_pos = world_pos_4.xyz / world_pos_4.w;
    
    // Reconstruct world normal
    // inverse_view is view-to-world. Normal is direction, so w=0.
    let world_normal = normalize((ssao.inverse_view * vec4<f32>(normal, 0.0)).xyz);

    let screen_size = vec2<f32>(ssao.screen_width, ssao.screen_height);
    let frag_coord = uv * screen_size;
    
    // Random rotation
    let noise = ign(frag_coord);
    
    let sample_count = f32(ssao.sample_count);
    let slice_count = f32(ssao.slice_count);
    let radius = ssao.sample_radius;
    
    var visibility = 0.0;
    
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
            let direction = select(-1.0, 1.0, side == 0u);
            let ray_dir = search_dir * direction;
            let step_factor = pow(radius, 1.0 / sample_count);
            let proj_scale = ssao.inverse_projection[1][1] * ssao.screen_height;
            let screen_radius = (radius * proj_scale) / -view_pos.z;
            let step_ratio = pow(screen_radius, 1.0 / sample_count);
            var current_step = 1.0;
            current_step = pow(step_ratio, select(noise, 1.0 - noise, side == 1u));
            
            for (var s = 0u; s < 64u; s = s + 1u) {
                if (s >= ssao.sample_count) { break; } 

                let sample_uv = uv + (ray_dir * current_step) / screen_size;
                current_step *= step_ratio;
                
                if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) { break; }
                
                let sample_depth = textureSampleLevel(depth_tex, post_sampler, sample_uv, 0);
                let sample_pos = reconstruct_position(sample_uv, sample_depth);
                
                let delta = sample_pos - view_pos;
                let dist_sq = dot(delta, delta);
                let dist = sqrt(dist_sq);
                let dist_vec = delta / dist;
                
                let horizon_cos = dot(dist_vec, view_vec);
                let horizon_angle_rel = fast_acos(horizon_cos) * direction;
                
                let back_horizon_cos = dot(normalize(delta - view_vec * ssao.hit_thickness), view_vec);
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
            }
        }
        
        let occluded_fraction = f32(count_bits(occlusion_bits)) / 32.0;
        visibility += 1.0 - occluded_fraction;
    }
    
    visibility /= slice_count;
    visibility = pow(visibility, 2.0);
    
    // Sample GI from probes (TODO: fix hard coded 0.01)
    let indirect_light = sample_grid_irradiance(world_pos, world_normal) * 0.01;
    
    return vec4<f32>(indirect_light, visibility);
}
