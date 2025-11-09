# Rendering Performance & Visual Enhancements

## Current Status

### ✅ Already Implemented
- **Frustum Culling**: Camera frustum culling working in `culling.rs`
- **Hierarchical LOD**: Distance-based subdivision/merging (`LOD_SUBDIVIDE_DISTANCE = 50.0`)
- **Global Sun/Moon Lighting**: Dynamic time-based lighting with configurable sun direction, colors, and ambient lighting
- **Atmospheric Fog**: Exponential fog with runtime-adjustable density (F/G keys, range 0.0-0.01)
- **Parallel Culling**: Rayon-based parallel chunk processing
- **Occlusion Culling**: Front-to-back traversal with visibility caching
- **Backface Culling**: GPU-level culling enabled (`cull_mode: Some(wgpu::Face::Back)`)
- **Greedy Meshing**: Leaf-chunk quads merged and cached (feature-gated debug logging available)
- **Proper Shader Uniforms**: 128-byte aligned uniform buffer with lighting parameters
### 5. 🎨 **Depth of Field (DoF) Blur - "Tilt-Shift" Effect**
**Status**: ✅ Implemented (2025-11)  
**Implementation**: Primary gather pass in `shaders/dof_blur.wgsl`, followed by separable Gaussian smoothing in `shaders/dof_smooth.wgsl`, integrated in `viewer_hierarchical.rs` with dynamic uniforms and render targets shared with the bloom chain. Runtime controls (`/`, `,`, `.`) adjust focus distance and strength; smoothing parameters respond to frame resolution.

**Current Behaviour**:
- Wide focus band with piecewise linear/exponential CoC curve for gentle transitions.
- Far-field amplification keeps distant geometry soft while near focus stays crisp.
- Bloom operates after DoF to accentuate highlights without re-blurring.

**Next Enhancements (optional):**
- Feed TAA history/tap jitter into DoF to reduce temporal noise.
- Add bokeh shape controls or highlight preservation tweaks for bright emissive voxels.
- Expose advanced tuning (focus band multiplier, smoothing radius) via config UI or hotkeys.

Performance remains ~2 ms @ 1080p thanks to separable smoothing and early-out paths for in-focus pixels.
fn calculate_coc(linear_depth: f32) -> f32 {
    let distance_from_focal = abs(linear_depth - dof_uniforms.focal_distance);
    
    // Sharp within focal range
    if distance_from_focal < dof_uniforms.focal_range {
        return 0.0;
    }
    
    // Blur increases with distance from focal plane
    let blur = (distance_from_focal - dof_uniforms.focal_range) / dof_uniforms.focal_distance;
    return min(blur * dof_uniforms.blur_strength, 1.0);
}

// Hexagonal bokeh blur pattern (better quality than box blur)
const SAMPLE_OFFSETS: array<vec2<f32>, 19> = array<vec2<f32>, 19>(
    vec2<f32>(0.0, 0.0),   // Center
    // Inner ring
    vec2<f32>(0.0, 1.0), vec2<f32>(0.866, 0.5), vec2<f32>(0.866, -0.5),
    vec2<f32>(0.0, -1.0), vec2<f32>(-0.866, -0.5), vec2<f32>(-0.866, 0.5),
    // Outer ring
    vec2<f32>(0.0, 2.0), vec2<f32>(1.732, 1.0), vec2<f32>(1.732, -1.0),
    vec2<f32>(0.0, -2.0), vec2<f32>(-1.732, -1.0), vec2<f32>(-1.732, 1.0),
    vec2<f32>(1.0, 1.732), vec2<f32>(1.0, -1.732),
    vec2<f32>(-1.0, -1.732), vec2<f32>(-1.0, 1.732),
    vec2<f32>(2.0, 0.0), vec2<f32>(-2.0, 0.0),
);

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(color_texture);
    let pixel_size = 1.0 / vec2<f32>(dimensions);
    
    // Sample depth and linearize
    let depth = textureSample(depth_texture, texture_sampler, input.uv);
    let linear_depth = linearize_depth(depth);
    let coc = calculate_coc(linear_depth);
    
    // If in focus, return original color (fast path)
    if coc < 0.01 {
        return textureSample(color_texture, texture_sampler, input.uv);
    }
    
    // Apply hexagonal bokeh blur
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    let blur_radius = coc * 15.0; // Max blur radius in pixels
    
    for (var i = 0; i < 19; i++) {
        let offset = SAMPLE_OFFSETS[i] * pixel_size * blur_radius;
        let sample_uv = input.uv + offset;
        
        // Sample color
        let sample_color = textureSample(color_texture, texture_sampler, sample_uv).rgb;
        
        // Sample depth to prevent background bleeding
        let sample_depth = textureSample(depth_texture, texture_sampler, sample_uv);
        let sample_linear_depth = linearize_depth(sample_depth);
        let sample_coc = calculate_coc(sample_linear_depth);
        
        // Weight by depth similarity (prevents halos)
        let depth_weight = 1.0 - min(abs(linear_depth - sample_linear_depth) / 10.0, 1.0);
        let weight = max(depth_weight, 0.1); // Minimum weight
        
        color_sum += sample_color * weight;
        weight_sum += weight;
    }
    
    let final_color = color_sum / weight_sum;
    return vec4<f32>(final_color, 1.0);
}
```

#### Rust Integration Code

```rust
// Add to Rust structs:
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DoFUniforms {
    focal_distance: f32,  // e.g., 50.0 (where camera is looking)
    focal_range: f32,     // e.g., 15.0 (sharp range)
    blur_strength: f32,   // e.g., 0.8 (0.0 = no blur, 1.0 = max)
    near_plane: f32,      // Camera near (e.g., 0.1)
    far_plane: f32,       // Camera far (e.g., 1000.0)
    _padding: [f32; 3],
}

// In your App struct, add:
struct App {
    // ... existing fields ...
    
    // DoF resources
    dof_pipeline: Option<wgpu::RenderPipeline>,
    dof_bind_group: Option<wgpu::BindGroup>,
    dof_uniform_buffer: Option<wgpu::Buffer>,
    offscreen_color_texture: Option<wgpu::Texture>,
    offscreen_depth_texture: Option<wgpu::Texture>,
    dof_uniforms: DoFUniforms,
}

// In render():
fn render(&mut self) {
    // 1. Render voxels to offscreen texture (instead of screen)
    let mut encoder = self.device.create_command_encoder(...);
    
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.offscreen_color_texture.view,  // Offscreen!
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                ..Default::default()
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.offscreen_depth_texture.view,  // Save depth!
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,  // Keep depth for DoF
                }),
                stencil_ops: None,
            }),
        });
        
        // ... draw voxels as normal ...
    }
    
    // 2. Apply DoF post-process to screen
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &screen_view,  // Now render to screen
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                ..Default::default()
            })],
            depth_stencil_attachment: None,  // No depth needed
        });
        
        render_pass.set_pipeline(&self.dof_pipeline);
        render_pass.set_bind_group(0, &self.dof_bind_group, &[]);
        render_pass.draw(0..3, 0..1);  // Full-screen triangle
    }
    
    self.queue.submit(std::iter::once(encoder.finish()));
}
```

#### Tuning for "Tilt-Shift" Toy World Effect

```rust
// For strong toy/miniature effect:
dof_uniforms.focal_distance = camera_target_distance; // Where you're looking
dof_uniforms.focal_range = 15.0;      // Small = strong effect
dof_uniforms.blur_strength = 0.8;     // High blur

// For subtle cinematic effect:
dof_uniforms.focal_range = 50.0;      // Large = gentle
dof_uniforms.blur_strength = 0.4;     // Subtle

// Runtime controls (add to keyboard input):
if key == KeyCode::BracketLeft {
    dof_uniforms.focal_range -= 2.0;  // Stronger effect
}
if key == KeyCode::BracketRight {
    dof_uniforms.focal_range += 2.0;  // Weaker effect
}
```

**Performance Notes**:
- 19-tap blur is good balance (quality vs speed)
- Consider reducing to 7-tap for 60fps target
- Can optimize with separable blur (horizontal then vertical)
- ~2-3ms per frame at 1080p with 19 samples

---

## Quick Wins (Easy to Implement)

### A. **Ambient Occlusion (Cheap Fake)**
Add darkening to voxel corners:
```wgsl
// In vertex shader, add vertex_index parameter
let corner_ao = array<f32, 8>(1.0, 0.7, 0.7, 0.5, 0.7, 0.5, 0.5, 0.3);
output.ao = corner_ao[vertex_index % 8];

// In fragment shader:
let color = input.color * lighting * input.ao;
```

### B. **Edge Highlighting**
Subtle outlines for toy aesthetic:
```wgsl
// In fragment shader:
let edge_factor = 1.0 - abs(dot(input.normal, normalize(camera_pos - world_pos)));
if (edge_factor > 0.95) {
    color = mix(color, vec3<f32>(0.0, 0.0, 0.0), 0.3); // Darken edges
}
```

### C. **Color Palette Improvements**
Use more saturated, toy-like colors:
```wgsl
// Vibrant toy colors with slight gradients per face
fn get_voxel_color(voxel_type: u32, normal: vec3<f32>) -> vec3<f32> {
    let base_color = get_base_color(voxel_type);
    // Vary brightness by face direction
    let brightness = 0.8 + 0.2 * max(normal.y, 0.0);
    return base_color * brightness;
}
```

## Implementation Priority

**Phase 1 - Immediate Wins** ✅ **COMPLETED**:
1. ✅ Enable backface culling - Changed `cull_mode: None` to `Some(wgpu::Face::Back)` (~50% reduction!)
2. ✅ Fix shader uniform visibility - Added FRAGMENT visibility to bind group
3. ✅ Create proper Uniforms struct - 128-byte aligned with lighting parameters
4. ✅ Implement dynamic lighting - Time-based sun/moon with keyboard control (T key)
5. ✅ Add adjustable fog - Runtime fog density control (F/G keys, 0.0-0.01 range)
6. ✅ Keyboard remapping - International layout support (arrow keys, alpha keys only)

**Phase 2 - Greedy Meshing** (6-8 hours) - **DO THIS FIRST**:
1. Implement greedy meshing algorithm (includes neighbor culling)
2. Add mesh caching system per chunk
3. Handle chunk updates (regenerate mesh on change)
4. Update rendering to use quads instead of per-voxel cubes

**Phase 3 - Tilt-Shift DoF** ✅ Completed:
- Added offscreen pipeline, primary DoF gather pass, separable smoothing, and bloom ordering.
- Keyboard controls wired for focal distance/range; defaults tuned for tilt-shift look.
- Follow-up ideas: optional TAA integration, UI exposure for smoothing radius.

**Phase 4 - Advanced Optimizations** (optional):
1. Inter-chunk face culling (for greedy meshed boundaries)
2. Complete LOD large-chunk rendering
3. GPU-driven rendering with compute shaders
4. Advanced shadows

---

## Expected Results

**Performance** (with all optimizations):
- ✅ Backface culling: ~50% fewer fragments (COMPLETED!)
- Greedy meshing with neighbor culling: ~95% fewer vertices (only external faces, merged into quads)
- Complete LOD: ~70% fewer draw calls
- **Combined**: 100-1000x improvement for large scenes!

**Visuals**:
- ✅ Global lighting with day/night cycle, adjustable atmospheric fog, backface culling
- ✅ Tilt-shift DoF and bloom deliver the miniature aesthetic now in viewer builds
- 🌅 Dynamic lighting + fog combos keep scenes atmospheric across times of day
- **Together**: Polished, distinctive visual style

## Summary

**Current State**: ✅ Solid foundation with dynamic lighting, adjustable fog, and backface culling working. Main bottleneck is now drawing all faces of all voxels (36 vertices each).

**Biggest Impact Changes** (Next Steps):
1. ✅ ~~Enable backface culling~~ - **COMPLETED**
2. ✅ ~~Add dynamic lighting~~ - **COMPLETED**
3. ✅ ~~Add adjustable fog~~ - **COMPLETED**
4. **Implement greedy meshing** (95% vertex reduction, includes neighbor culling) - **DO THIS NEXT**
5. Polish DoF/bloom stack as needed (expose advanced controls, consider TAA integration)
6. Add inter-chunk face culling (optional optimization after greedy meshing)

**Key Insight**: Greedy meshing should be done BEFORE standalone neighbor culling, as it naturally produces only external faces while merging them into larger quads. Doing per-voxel neighbor checks first would be redundant work.

The shader infrastructure is now solid with proper uniforms and dynamic lighting - performance gains are mostly about reducing geometry submission!
