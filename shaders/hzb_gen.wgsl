// HZB Mip Chain Generator
// Generates a hierarchical depth pyramid with two channels:
//   R = MIN depth (nearest)  -> conservative ray-march skipping (water reflections, SSR)
//   G = MAX depth (farthest) -> conservative occlusion culling (gpu_cull.wgsl)
// Mip 0: copy from the depth buffer. Mip 1-N: 2x2 reduction (3 taps along an edge whose
// source size is odd, so every source texel is represented in the coarser mip).

struct HzbParams {
    width: u32,
    height: u32,
    src_mip: u32,
    dst_mip: u32,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var hzb_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: HzbParams;
@group(0) @binding(3) var hzb_src: texture_2d<f32>;

// Mip 0: Copy and convert depth to float
@compute @workgroup_size(8, 8)
fn copy_depth(@builtin(global_invocation_id) id: vec3<u32>) {
    let coords = vec2<i32>(id.xy);
    let dims = textureDimensions(hzb_out);
    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Real depth sampling
    let depth = textureLoad(depth_tex, coords, 0);
    textureStore(hzb_out, coords, vec4<f32>(depth, depth, 0.0, 1.0));
}

// Mip 1-N: Downsample with MIN (R) and MAX (G) reduction
@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let coords = vec2<i32>(id.xy);
    let dims = textureDimensions(hzb_out);
    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    let src_coords = coords * 2;
    // We read from the previous mip level (src_mip) of the source texture
    let src_mip = params.src_mip;
    let src_dims = vec2<i32>(textureDimensions(hzb_src, i32(src_mip)));
    // The last destination column/row also covers the leftover source texel of an odd size.
    let extent_x = select(2, 3, coords.x == i32(dims.x) - 1 && (src_dims.x & 1) == 1);
    let extent_y = select(2, 3, coords.y == i32(dims.y) - 1 && (src_dims.y & 1) == 1);

    var min_d = 1.0;
    var max_d = 0.0;
    for (var y = 0; y < extent_y; y++) {
        for (var x = 0; x < extent_x; x++) {
            let s = src_coords + vec2<i32>(x, y);
            if (s.x < src_dims.x && s.y < src_dims.y) {
                // textureLoad with explicit mip level
                let d = textureLoad(hzb_src, s, i32(src_mip)).rg;
                min_d = min(min_d, d.r);
                max_d = max(max_d, d.g);
            }
        }
    }

    textureStore(hzb_out, coords, vec4<f32>(min_d, max_d, 0.0, 1.0));
}
