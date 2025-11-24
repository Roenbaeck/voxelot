// HZB Mip Chain Generator
// Generates a hierarchical depth pyramid for occlusion culling
// Mip 0: Copy from depth buffer
// Mip 1-N: Downsample with MAX reduction (furthest depth)

struct HzbParams {
    width: u32,
    height: u32,
    src_mip: u32,
    dst_mip: u32,
};

// Mip 0: Copy depth buffer to HZB
@group(0) @binding(0)
var depth_tex: texture_depth_2d;

@group(0) @binding(1)
var hzb_out: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var<uniform> params: HzbParams;

@compute @workgroup_size(8, 8, 1)
fn copy_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    
    if (x >= i32(params.width) || y >= i32(params.height)) {
        return;
    }
    
    // Read depth as scalar
    let d = textureLoad(depth_tex, vec2<i32>(x, y), 0);
    textureStore(hzb_out, vec2<i32>(x, y), vec4<f32>(d, 0.0, 0.0, 0.0));
}

// Mip 1-N: Downsample with MAX reduction
@group(0) @binding(3)
var hzb_src: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    
    if (x >= i32(params.width) || y >= i32(params.height)) {
        return;
    }
    
    // Sample 2x2 region from previous mip
    let base = vec2<i32>(x * 2, y * 2);
    let d00 = textureLoad(hzb_src, base + vec2<i32>(0, 0), i32(params.src_mip)).r;
    let d01 = textureLoad(hzb_src, base + vec2<i32>(0, 1), i32(params.src_mip)).r;
    let d10 = textureLoad(hzb_src, base + vec2<i32>(1, 0), i32(params.src_mip)).r;
    let d11 = textureLoad(hzb_src, base + vec2<i32>(1, 1), i32(params.src_mip)).r;
    
    // MAX reduction (furthest depth) for conservative occlusion
    let max_depth = max(max(d00, d01), max(d10, d11));
    
    textureStore(hzb_out, vec2<i32>(x, y), vec4<f32>(max_depth, 0.0, 0.0, 0.0));
}
