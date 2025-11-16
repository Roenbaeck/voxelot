struct SsaoUniforms {
    sample_count: u32,
    slice_count: u32,
    sample_radius: f32,
    hit_thickness: f32,
    screen_width: f32,
    screen_height: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> ssao: SsaoUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var post_sampler: sampler;

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

// Reconstruct view-space position from depth
fn reconstruct_position(uv: vec2<f32>) -> vec3<f32> {
    // depth is returned as [0..1], we need clip Z in [-1,1]
    let d: f32 = textureSample(depth_tex, post_sampler, uv);
    // convert from [0,1] to NDC
    let ndc = vec3<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), d * 2.0 - 1.0);
    // Without a full inverse projection, approximate view-space z by linearizing depth
    // We can use a short approximation here; for better results pass inverse proj: TODO
    let z = d;
    return vec3<f32>(ndc.xy, z);
}

// bitcount helper using integer bit operations
// `mut` is a reserved keyword in WGSL; avoid using reserved keywords in parameter names.
fn popcount(value: u32) -> u32 {
    var v: u32 = value;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    return (((v + (v >> 4u)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24u;
}

// update bitmask sectors between min and max horizon
fn update_sectors(minH: f32, maxH: f32, out: u32) -> u32 {
    let sector_count = 32u;
    let start_bit = u32(minH * f32(sector_count));
    let horizon_angle = u32(ceil((maxH - minH) * f32(sector_count)));
    var angle_mask: u32 = 0u;
    if (horizon_angle > 0u) {
        angle_mask = 0xFFFFFFFFu >> (sector_count - horizon_angle);
    }
    let current = angle_mask << start_bit;
    return out | current;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Reconstruct a rough position and normal from depth
    let pos = reconstruct_position(uv);
    let camera = normalize(-pos);

    var occlusion_mask: u32 = 0u;
    var visibility = 0.0;

    let sample_radius = ssao.sample_radius;
    let sample_count = f32(ssao.sample_count);
    let slice_count = f32(ssao.slice_count);

    // jitter per-fragment derived from uv
    let jitter = fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453123) - 0.5;

    for (var slice = 0u; slice < ssao.slice_count; slice = slice + 1u) {
        let phi = (2.0 * 3.14159) * (f32(slice) + jitter) / (slice_count);
        let omega = vec2<f32>(cos(phi), sin(phi));

        var occlusion_bits: u32 = 0u;

        for (var s = 0u; s < ssao.sample_count; s = s + 1u) {
            let step = (f32(s) + jitter) / sample_count + 0.01;
            let aspect = vec2<f32>(ssao.screen_width / ssao.screen_height, 1.0);
            let sampleUV = uv - step * sample_radius * omega * aspect;

            // clamp to screen
            if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) { continue; }

            let sd = reconstruct_position(sampleUV);
            let sampDirection = normalize(sd - pos);
            // horizon angles measured along view direction; approximate
            let frontH = max(-dot(sampDirection, camera), 0.0);
            let backH = max(dot(pos - sd, camera) - ssao.hit_thickness, 0.0);
            let clamped_front = clamp(frontH, 0.0, 1.0);
            let clamped_back = clamp(backH, 0.0, 1.0);

            occlusion_bits = update_sectors(clamped_front, clamped_back, occlusion_bits);
        }

        // compute occlusion for this slice
        let occluded = f32(popcount(occlusion_bits)) / 32.0;
        visibility += 1.0 - occluded;
        occlusion_mask = occlusion_mask | occlusion_bits;
    }

    visibility = visibility / f32(ssao.slice_count);

    // output AO in alpha channel (1.0 -> lit, 0.0 -> occluded)
    return vec4<f32>(0.0, 0.0, 0.0, visibility);
}
