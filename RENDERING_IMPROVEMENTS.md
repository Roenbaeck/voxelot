# Rendering Performance & Visual Enhancements

## Current Status

### 🔴 **Critical Bugs Found** (Must Fix!)

#### 1. **Shader/Buffer Size Mismatch** - Will cause crashes or visual glitches!
**Location**: `viewer_hierarchical.rs` line ~430, ~563

**Problem**: 
- Shader expects 128-byte uniform (MVP + lighting + fog)
- Rust only allocates 128 bytes BUT only writes 64 bytes (MVP matrix)
- Rest of buffer is uninitialized garbage → shader reads random data for lighting!

**Fix**:
```rust
// Current (WRONG):
let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    size: 128, // MVP matrix  <-- LIES! Has more fields!
    ...
});
queue.write_buffer(..., 0, bytemuck::cast_slice(&mvp_cols)); // Only 64 bytes!

// Fixed:
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],           // 64 bytes
    sun_direction: [f32; 3],      // 12 bytes
    _padding1: f32,                // 4 bytes (alignment)
    sun_color: [f32; 3],          // 12 bytes  
    _padding2: f32,                // 4 bytes
    ambient_color: [f32; 3],      // 12 bytes
    time_of_day: f32,             // 4 bytes
    _padding3: [f32; 3],          // 12 bytes (alignment)
}
// Total: 128 bytes ✓

let uniforms = Uniforms {
    mvp: mvp_cols,
    sun_direction: [0.5, 1.0, 0.3],  // Default sun
    _padding1: 0.0,
    sun_color: [1.0, 0.95, 0.8],
    _padding2: 0.0,
    ambient_color: [0.3, 0.35, 0.45],
    time_of_day: 0.5,
    _padding3: [0.0; 3],
};

let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("Uniform Buffer"),
    contents: bytemuck::cast_slice(&[uniforms]),
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
});
```

#### 2. **Shader Visibility Wrong** - Fragment shader can't read uniforms!
**Location**: `viewer_hierarchical.rs` line ~351

**Problem**: Uniforms only visible to vertex shader, but fragment shader needs them for lighting!

**Fix**:
```rust
// Current (WRONG):
wgpu::BindGroupLayoutEntry {
    visibility: wgpu::ShaderStages::VERTEX,  // Fragment can't see it!
    ...
}

// Fixed:
wgpu::BindGroupLayoutEntry {
    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
    ...
}
```

#### 3. **Backface Culling Disabled** - Wastes 50% of fragment shader work!

**Fix**: Already documented in main section below.

---

### ✅ Already Implemented
- **Frustum Culling**: Camera frustum culling is working in `culling.rs`
- **Hierarchical LOD**: Distance-based subdivision/merging (`LOD_SUBDIVIDE_DISTANCE = 50.0`)
- **Global Sun/Moon Lighting**: Shader has `sun_direction`, `sun_color`, `ambient_color` uniforms
- **Atmospheric Fog**: Exponential fog with configurable density in fragment shader
- **Parallel Culling**: Rayon-based parallel chunk processing
- **Occlusion Culling**: Front-to-back traversal with visibility caching

### ⚠️ Partially Implemented
- **Backface Culling**: Currently DISABLED (`cull_mode: None`) - Easy win by enabling!

### ❌ Not Yet Implemented
- **Greedy Meshing**: Still drawing 36 vertices per voxel
- **Neighbor-Based Face Culling**: Drawing all 6 faces even when occluded
- **Depth of Field Post-Process**: No DoF shader exists yet
- **Mesh Caching**: Regenerating geometry every frame
- **GPU-Driven Rendering**: CPU-based culling only

---

## Priority Performance Improvements (Highest Impact First)

### 1. 🚀 **Greedy Meshing** (10-100x vertex reduction)
**Impact**: HUGE - reduces vertices by 90%+ for solid regions  
**Complexity**: Medium

Merge adjacent voxel faces into larger quads. Instead of 6 faces per voxel, create one merged face for entire runs.

```rust
// Algorithm outline:
fn greedy_mesh(chunk: &Chunk) -> Vec<Quad> {
    let mut quads = Vec::new();
    let mut visited = [[[false; 16]; 16]; 16];
    
    // For each direction (±X, ±Y, ±Z)
    for dir in [Dir::PosX, Dir::NegX, ...] {
        // Sweep through slices perpendicular to direction
        for slice in 0..16 {
            // Find rectangular regions of same type
            for y in 0..16 {
                for x in 0..16 {
                    if should_draw_face(x, y, slice, dir) && !visited[x][y][slice] {
                        // Expand horizontally
                        let mut width = 1;
                        while can_expand_x(x + width, y, slice, dir) { width += 1; }
                        
                        // Expand vertically
                        let mut height = 1;
                        while can_expand_y(x, y + height, slice, dir, width) { height += 1; }
                        
                        // Create merged quad
                        quads.push(Quad { x, y, slice, width, height, dir });
                        mark_visited(x, y, slice, width, height);
                    }
                }
            }
        }
    }
    quads
}
```

**Integration**:
- Generate mesh per chunk instead of per voxel
- Cache meshes - only regenerate when chunk changes
- Update instance buffer with quads instead of cubes

### 2. ⚡ **Face Culling** (60-80% vertex reduction)
**Impact**: High - only draw external faces  
**Complexity**: Low

Face culling eliminates faces that won't be visible, dramatically reducing vertex count. There are several types of culling that can be combined:

#### **2a. Neighbor-Based Culling** (Most Important)
Don't draw faces adjacent to solid voxels:

```rust
fn should_draw_face(pos: Pos, direction: Dir, chunk: &Chunk) -> bool {
    let neighbor_pos = pos + direction.offset();
    
    // Draw face if:
    // 1. At chunk boundary (exposed to air/other chunks)
    // 2. Neighbor voxel is empty/air
    // 3. Neighbor voxel is transparent (glass, water, etc.)
    
    if !chunk.in_bounds(neighbor_pos) {
        return true; // Chunk edge - might be visible
    }
    
    let neighbor = chunk.get(neighbor_pos);
    match neighbor {
        None => true,                              // Empty - draw face
        Some(voxel) if voxel.is_transparent() => true,  // Transparent - draw face
        Some(_) => false,                          // Solid neighbor - cull face
    }
}
```

#### **2b. Backface Culling** (GPU-Level, Enable by Default)
GPU automatically culls faces pointing away from camera. This is "free" when enabled:

```rust
// In wgpu pipeline setup:
wgpu::PrimitiveState {
    cull_mode: Some(wgpu::Face::Back),  // Cull back-facing triangles
    front_face: wgpu::FrontFace::Ccw,   // Counter-clockwise = front
    ..Default::default()
}
```

**How it works**: For opaque cubes, you can only see 3 faces maximum from any viewpoint. The GPU automatically skips the other 3 faces pointing away from camera. This is handled in hardware and costs almost nothing.

**Important**: This requires proper winding order in your cube vertices (counter-clockwise when viewed from outside).

#### **2c. View-Frustum Culling** (Per-Voxel, Advanced)
For individual voxels within view frustum, you could theoretically skip faces based on camera direction, but this is usually not worth it because:
- The GPU already does backface culling efficiently
- Per-voxel CPU culling adds overhead
- Only beneficial for very large voxels or special cases

**Recommendation**: Use 2a (neighbor-based) + 2b (GPU backface culling). This combination gives you 80%+ reduction with minimal cost.

```rust
// Combined approach:
fn build_mesh(chunk: &Chunk) -> Vec<Face> {
    let mut faces = Vec::new();
    
    for (pos, voxel) in chunk.iter_voxels() {
        // Only for opaque voxels
        if !voxel.is_opaque() {
            continue; // Transparent voxels need all faces (for now)
        }
        
        // Check each of 6 possible faces
        for direction in [Dir::PosX, Dir::NegX, Dir::PosY, Dir::NegY, Dir::PosZ, Dir::NegZ] {
            // Neighbor-based culling
            if should_draw_face(pos, direction, chunk) {
                faces.push(Face::new(pos, direction, voxel.voxel_type));
            }
            // GPU will handle backface culling automatically
        }
    }
    
    faces
}
```

**Expected Results**:
- Neighbor culling: ~80% reduction (only external faces remain)
- GPU backface culling: ~50% additional reduction on remaining faces
- **Combined**: Only 10% of original faces are drawn (~90% reduction!)

### 3. 📏 **LOD Using Hierarchy** (Distance-based simplification)
**Status**: ✅ **Partially Implemented** - subdivision works, but no large-chunk rendering  
**Impact**: Medium-High for large scenes  
**Complexity**: Low (you already have the hierarchy!)

Currently implemented: Voxels subdivide when camera gets closer than 50 units. What's missing: drawing distant sub-chunks as single large cubes instead of recursing.

```rust
// In culling phase (ADD THIS):
fn collect_with_lod(chunk: &Chunk, world_pos: Vec3, distance: f32, scale: u32) {
    // NEW: If far away, draw entire chunk as one large cube
    if distance > LOD_DISTANCE * scale as f32 {
        instances.push(VoxelInstance {
            position: world_pos,
            scale: scale * 16, // Entire 16³ chunk as one voxel
            voxel_type: chunk.most_common_type(), // Or average color
        });
        return; // Don't recurse
    }
    
    // EXISTING: Close up - recurse into sub-chunks or draw individual voxels
    match voxel {
        Voxel::Solid(_) => { /* draw as normal */ }
        Voxel::Chunk(sub) => {
            if distance < LOD_SUBDIVIDE_DISTANCE {
                collect_with_lod(sub, ...); // Recurse
            }
        }
    }
}

const LOD_DISTANCE: f32 = 200.0; // Tune this
```

### 4. 🎮 **GPU-Driven Rendering** (Reduce CPU overhead)
**Status**: ❌ Not Implemented  
**Impact**: Medium for scenes with many chunks  
**Complexity**: High

Use indirect drawing to let GPU decide what to draw:
```rust
// Store all instances in GPU buffer
// Use compute shader to:
// 1. Frustum cull
// 2. Occlusion cull (using HiZ buffer)
// 3. Write visible indices to indirect draw buffer

wgpu::RenderPass::draw_indirect(indirect_buffer)
```

---

## Visual Enhancements

### 5. 🎨 **Depth of Field (DoF) Blur - "Tilt-Shift" Effect**
**Status**: ❌ Not Implemented (shader needs to be created)  
**Effect**: Miniature/toy world look with selective focus  
**Performance Cost**: Medium (post-process, ~2-3ms at 1080p)

The "tilt-shift" effect makes your voxel world look like a miniature diorama by:
- Keeping a focal plane sharp (e.g., where camera is looking)
- Blurring everything closer and farther away
- Creating shallow depth-of-field like macro photography

**Implementation Steps**:
1. **Render to offscreen textures** (color + depth)
2. **Apply DoF post-process shader** 
3. **Display to screen**

#### DoF Shader (Create: `shaders/dof_blur.wgsl`)

```wgsl
// Depth of Field post-processing shader for "tilt-shift" toy world effect

struct DoFUniforms {
    focal_distance: f32,      // Distance that's in focus (world units)
    focal_range: f32,         // Range of sharp focus (world units)
    blur_strength: f32,       // Maximum blur amount (0.0-1.0)
    near_plane: f32,          // Camera near plane
    far_plane: f32,           // Camera far plane
    _padding: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> dof_uniforms: DoFUniforms;

@group(0) @binding(1)
var color_texture: texture_2d<f32>;

@group(0) @binding(2)
var depth_texture: texture_depth_2d;

@group(0) @binding(3)
var texture_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle (no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);
    return output;
}

// Convert depth buffer value to linear world distance
fn linearize_depth(depth: f32) -> f32 {
    let near = dof_uniforms.near_plane;
    let far = dof_uniforms.far_plane;
    let z_ndc = depth * 2.0 - 1.0; // To NDC
    return (2.0 * near * far) / (far + near - z_ndc * (far - near));
}

// Calculate circle of confusion (blur amount) based on depth
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



### 6. ☀️ **Global Sun/Moon Lighting**
**Status**: ✅ **Shader Ready** - just needs Rust uniform updates  
**Effect**: Dynamic time of day, atmospheric lighting  
**Performance Cost**: Negligible (already in shader!)

The shader already has all the lighting infrastructure! Just need to update the Rust-side uniforms.

**What's Already Working**:
- Sun direction, color, and ambient color in shader
- Diffuse lighting calculation
- Atmospheric fog integration

**Implementation** (update uniforms struct in Rust):
```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    sun_direction: [f32; 3],
    _padding1: f32,
    sun_color: [f32; 3],
    _padding2: f32,
    ambient_color: [f32; 3],
    time_of_day: f32,
}

// Update each frame:
fn update_lighting(&mut self, time: f32) {
    // Animate sun position (0.0 = midnight, 0.5 = noon, 1.0 = midnight)
    let time_of_day = (time * 0.1) % 1.0; // Slow rotation
    let angle = time_of_day * std::f32::consts::TAU;
    
    // Sun moves in arc across sky
    self.uniforms.sun_direction = [
        angle.cos() * 0.5,
        angle.sin(),  // High at noon, low at sunrise/sunset
        0.3,
    ];
    
    // Color changes with time
    if time_of_day < 0.25 || time_of_day > 0.75 {
        // Night - moon (cool blue)
        self.uniforms.sun_color = [0.3, 0.3, 0.5];
        self.uniforms.ambient_color = [0.05, 0.05, 0.15];
    } else if time_of_day < 0.3 || time_of_day > 0.7 {
        // Sunrise/sunset - warm orange
        self.uniforms.sun_color = [1.0, 0.6, 0.3];
        self.uniforms.ambient_color = [0.3, 0.2, 0.2];
    } else {
        // Day - bright yellow sun
        self.uniforms.sun_color = [1.0, 0.95, 0.8];
        self.uniforms.ambient_color = [0.3, 0.35, 0.45]; // Blue sky ambient
    }
}
```

### 7. 🌫️ **Atmospheric Fog**
**Status**: ✅ **Fully Implemented** in shader!  
**Effect**: Depth cues and atmospheric perspective

Already working! Exponential fog in fragment shader. Currently tuned as:
```wgsl
let fog_density = 0.0015;  // Adjustable
let fog_color = vec3<f32>(0.7, 0.8, 0.9);  // Light blue sky
```

To adjust, simply edit these values in `shaders/voxel.wgsl`.

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

**Phase 1 - Immediate Wins** (< 1 hour, huge impact!):
1. ✅ **Enable backface culling** - Change `cull_mode: None` to `Some(wgpu::Face::Back)` (~50% reduction!)
2. **Verify vertex winding** - Ensure counter-clockwise from outside

**Phase 2 - Face Culling** (2-3 hours):
1. Implement neighbor-based face culling (~80% reduction)
2. Add voxel transparency support

**Phase 3 - Tilt-Shift DoF** (3-4 hours):
1. Create offscreen render targets
2. Implement DoF shader (provided above)
3. Add Rust integration
4. Add keyboard controls for tuning

**Phase 4 - Greedy Meshing** (6-8 hours):
1. Implement meshing algorithm
2. Add mesh caching system
3. Handle chunk updates

**Phase 5 - Advanced** (optional):
1. Complete LOD large-chunk rendering
2. GPU-driven rendering
3. Advanced shadows

---

## Expected Results

**Performance** (with all optimizations):
- Backface culling: ~50% fewer fragments (EASY WIN!)
- Neighbor face culling: ~80% fewer vertices  
- Greedy meshing: ~95% fewer vertices
- Complete LOD: ~70% fewer draw calls
- **Combined**: 100-1000x improvement for large scenes!

**Visuals**:
- ✅ **Already have**: Global lighting, atmospheric fog
- 🎨 **With DoF**: Professional miniature/toy world aesthetic
- 🌅 **With dynamic lighting**: Day/night cycles, atmospheric scenes
- **Together**: Polished, distinctive visual style

## Summary

**Current State**: Good foundation with lighting and fog working. Main bottleneck is drawing all faces of all voxels (36 vertices each).

**Biggest Impact Changes**:
1. Enable backface culling (1 line change, 50% gain!)
2. Add DoF for tilt-shift toy aesthetic
3. Implement face culling (80% reduction)
4. Add greedy meshing (10-100x improvement)

The shader infrastructure is solid - performance gains are mostly about reducing geometry submission!
