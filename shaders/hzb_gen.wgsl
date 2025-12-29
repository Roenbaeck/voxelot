// HZB Mip Chain Generator
// Generates a hierarchical depth pyramid for occlusion culling
// Mip 0: Copy from depth buffer
// Mip 1-N: Downsample with MAX reduction (furthest depth)

struct HzbParams {
    width: u32,
    height: u32,
    src_mip: u32,
    dst_mip: u32,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var hzb_out: texture_storage_2d<r32float, write>;
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
    textureStore(hzb_out, coords, vec4<f32>(depth, 0.0, 0.0, 1.0));
}

// Mip 1-N: Downsample with MAX reduction
@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let coords = vec2<i32>(id.xy);
    let dims = textureDimensions(hzb_out);
    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }
    
    let src_coords = coords * 2;
    let src_dims = textureDimensions(hzb_src);
    
    // MAX reduction (furthest depth) to conservatively detect occlusion
    var max_d = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let s = src_coords + vec2<i32>(x, y);
            if (s.x < i32(src_dims.x) && s.y < i32(src_dims.y)) {
                let d = textureLoad(hzb_src, s, 0).r;
                max_d = max(max_d, d);
            }
        }
    }
    
    textureStore(hzb_out, coords, vec4<f32>(max_d, 0.0, 0.0, 1.0));
}
