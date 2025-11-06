# 4-RBM Voxel Engine - Quick Reference

## Running the Benchmark

```bash
# Run in release mode for accurate performance measurements
cargo run --bin bench_4rbm --release

# Run library tests
cargo test --lib --release
```

## API Usage

### Creating a World

```rust
use voxelot::VoxelWorld;

let mut world = VoxelWorld::new();
```

### Adding Voxels

```rust
// Set voxel at (x, y, z) with type
world.set_voxel(10, 20, 30, 42);  // Returns previous type if any
```

### Querying Voxels

```rust
// Get voxel type at position
if let Some(voxel_type) = world.get_voxel(10, 20, 30) {
    println!("Found voxel of type {}", voxel_type);
}

// Check if voxel exists
if world.contains(10, 20, 30) {
    println!("Voxel exists!");
}
```

### Removing Voxels

```rust
// Remove voxel and get its previous type
if let Some(old_type) = world.remove_voxel(10, 20, 30) {
    println!("Removed voxel of type {}", old_type);
}
```

### Culling Visible Voxels

```rust
// Plane propagation (recommended)
let visible = world.cull_visible_plane(
    camera_x,     // i64
    camera_y,     // i64  
    camera_z,     // i64
    view_width,   // u32 - in voxels
    view_height,  // u32 - in voxels
    max_depth,    // u32 - how far to search
);

// Returns Vec<(i64, i64, i64, u16)> - (x, y, z, voxel_type)
for (x, y, z, voxel_type) in visible {
    // Render this voxel
}
```

```rust
// Orthographic iteration (for comparison)
let visible = world.cull_visible_orthographic(
    camera_x, camera_y, camera_z,
    view_width, view_height, max_depth
);
```

### Memory Statistics

```rust
let stats = world.memory_usage();
println!("Total: {} bytes", stats.total());
println!("px: {} bytes", stats.px_bytes);
println!("py: {} bytes", stats.py_bytes);
println!("pz: {} bytes", stats.pz_bytes);
println!("ph: {} bytes", stats.ph_bytes);
println!("types: {} bytes", stats.types_bytes);
```

### World Info

```rust
// Count voxels
let count = world.len();

// Check if empty
if world.is_empty() {
    println!("No voxels!");
}
```

## Architecture Details

### The 4 Bitmaps

1. **px** - X-axis marginal presence: "is there any voxel at x=N?"
2. **py** - Y-axis marginal presence: "is there any voxel at y=N?"
3. **pz** - Z-axis marginal presence: "is there any voxel at z=N?"
4. **ph** - Hash verification: "hash(x,y,z) to eliminate false positives"

### How Queries Work

When you call `world.get_voxel(x, y, z)`:

1. **Fast reject**: Check `px.contains(x)`, `py.contains(y)`, `pz.contains(z)`
   - If any is false, voxel definitely doesn't exist → return None
   
2. **Hash verification**: Compute `h = fnv1a_hash(x, y, z)`
   - Check `ph.contains(h)`
   - If false, this was a false positive → return None
   
3. **Type lookup**: Use `ph.rank(h)` to find index in types array
   - Return `types[rank - 1]`

### Why This Is Fast

- **Marginal checks**: O(1) bitmap lookups
- **Hash verification**: O(1) bitmap lookup
- **Rank operation**: O(1) in Roaring Bitmaps
- **No HashMap overhead**: Direct array indexing via rank

### Memory Efficiency

For sparse worlds (1-10% density):
- **Marginals**: Very small (one bit per unique x/y/z value)
- **Hash bitmap**: ~4 bytes per voxel (compressed)
- **Types array**: 2 bytes per voxel (uncompressed)
- **Total**: ~6-10 bytes per voxel vs 1 byte minimum in dense arrays

## Performance Characteristics

### Query Time
- `contains()`: ~50ns
- `get_voxel()`: ~100ns
- `set_voxel()`: ~500ns (includes rank calculation and array insert)

### Culling Time (128×128×16 world, release mode)
- Plane propagation: 115ms
- Orthographic iteration: 397ms
- **Speedup: 3.43x**

### Memory Usage (128×128×16 world)
- 93,648 voxels
- 782 KB total
- **8.3 bytes/voxel average**

### Scaling
Performance scales roughly linearly with world size:
- 64×64×16: 21ms
- 128×128×16: 115ms  
- 256×256×16: 524ms

## Common Patterns

### Generate Heightmap Terrain

```rust
fn generate_terrain(world: &mut VoxelWorld, width: i64, height: i64) {
    for x in 0..width {
        for y in 0..height {
            let h = compute_height(x, y);  // Your height function
            for z in 0..=h {
                let voxel_type = match z {
                    z if z == h => 0,      // Grass on top
                    z if z > h - 3 => 1,   // Dirt
                    _ => 2,                // Stone
                };
                world.set_voxel(x, y, z, voxel_type);
            }
        }
    }
}
```

### Render Visible Voxels

```rust
let visible = world.cull_visible_plane(cam_x, cam_y, cam_z, 320, 240, 100);

for (x, y, z, voxel_type) in visible {
    // Isometric projection
    let screen_x = (x - cam_x) - (y - cam_y);
    let screen_y = ((x - cam_x) + (y - cam_y)) / 2 - (z - cam_z);
    
    // Draw voxel at screen position with color based on type
    draw_voxel(screen_x, screen_y, get_color(voxel_type));
}
```

### Benchmark Comparison

```rust
use std::time::Instant;

let start = Instant::now();
let visible_plane = world.cull_visible_plane(x, y, z, w, h, d);
let plane_time = start.elapsed();

let start = Instant::now();
let visible_ortho = world.cull_visible_orthographic(x, y, z, w, h, d);
let ortho_time = start.elapsed();

println!("Plane: {:.2}ms, Ortho: {:.2}ms", 
    plane_time.as_secs_f64() * 1000.0,
    ortho_time.as_secs_f64() * 1000.0);
```

## Limitations & Future Work

### Current Limitations
- Coordinate range: 0 to u32::MAX (4.2 billion)
- Type range: 0 to u16::MAX (65,535 types)
- Single-threaded culling
- No spatial acceleration for physics queries

### Planned Optimizations
1. **Streaming intersection**: Process hits in-place without allocating
2. **Parallel culling**: Rayon-based parallelization
3. **Columnar Z-index**: For fast vertical queries (physics, lighting)
4. **Cached projection**: For fixed-camera scenarios

### When to Use This vs Alternatives
- ✓ Use 4-RBM for: Sparse worlds (<10% density), isometric views, memory constraints
- ✗ Use octrees for: Dense worlds (>50% density), complex CSG operations
- ✗ Use voxel grids for: Tiny worlds that fit in cache, frequent modifications

## Troubleshooting

### "Why is my world using so much memory?"
- Check `world.len()` - how many voxels do you actually have?
- Use `world.memory_usage()` to see breakdown
- Hash bitmap (`ph`) will be the largest component (4-8 bytes per voxel)

### "Why is culling slow in debug mode?"
- Always benchmark in release mode: `cargo run --release`
- Debug mode has no SIMD optimizations and lots of bounds checking

### "I'm seeing false positives!"
- The FNV-1a hash should have <1 collision per 4 billion voxels
- If you're seeing many, file a bug report with your test case

### "Can I use negative coordinates?"
- Yes! The API accepts i64, though internally we cast to u32
- Negative values will wrap around or be rejected (currently rejects <0)
- For full signed support, modify the hash function and bitmap key type

## Advanced Topics

### Custom Hash Functions
You can modify `fnv1a_hash()` for different properties:
- More mixing → fewer collisions, slower
- Spatial locality → better compression
- Cryptographic → paranoid collision resistance

### Memory vs Speed Tradeoffs
- More bitmaps = less iteration, more memory
- Current 4-bitmap design is the sweet spot for most uses

### Integrating with ECS
```rust
struct VoxelWorldComponent {
    world: VoxelWorld,
}

// In your game loop
fn culling_system(world_query: Query<&VoxelWorldComponent>, camera_query: Query<&Camera>) {
    for world in world_query.iter() {
        for camera in camera_query.iter() {
            let visible = world.world.cull_visible_plane(
                camera.x, camera.y, camera.z,
                camera.viewport_w, camera.viewport_h,
                camera.far_plane
            );
            // Process visible voxels...
        }
    }
}
```

## Resources

- **OUTLINE.md**: Full design discussion and alternatives
- **4RBM_RESULTS.md**: Performance analysis and benchmarks
- **src/lib.rs**: Complete implementation with inline documentation
- **src/bin/bench_4rbm.rs**: Comprehensive benchmarking tool

## License

This implementation is part of the voxelot project.
