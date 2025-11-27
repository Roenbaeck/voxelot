- ✅ Is it possible to rotate the skybox very slowly to make it look less static?
- ✅ Can we make the skybox gradually darker during night, almost black at midnight?
- ✅ The sky reflected in the water should also be dark at night.
- ✅ Can we reduce the amount of light as we get closer to midnight, so there’s no “moonlight” at midnight? The whole scene is too bright at night.
- ✅ Can we add keys to raise and lower water level?
- ✅ Can we implement screen space reflections (SSR) for the water? It would be good if we later can set a material in the palette.txt to be reflective as well, so the SSR isn't limited to water.
- ✅ Can we make the water level configurable in config.toml (with save on exit)?
- ✅ Can we make a toggle for the GUI overlay? Perhaps the F5-key?
- ✅ Can you update README with any missing keybindings?
- ✅ There's a non-smooth color transition just before time = 0.274 (I paused there).
- ✅ I would like the sun to shine longer, so we see shadows climbing higher up buildings at dusk before light fades away.
- ✅ Skybox should get colors washed out the darker it gets. Now it has a saturated yellow unnatural tint at midnight. 
- ✅ Skybox needs a blueish tint. 
- ✅ Distant objects look like they again get brighter closer to the horizon at early night and early morning. It’s making them look unnaturally lit up. 
- ✅ Reflections from SSR are only visible if the camera is close to a reflective surface.

- 🛑 Reflections should be stronger at night than in the day, since nothing is drowning out the light.
- 🛑 Why is the fullscreen render so much slower than the windowed render? It drops from 120FPS to 20FPS when I go fullscreen.
- 🛑 Sliders instead of buttons to change settings.
- 🛑 Can all of our code be compiled as wasm?

- Would it be a good idea to have a cache for unmeshed chunks as well where we "expand" the dense voxels array into an array with all 4096 positions? We could then reuse this whenever we need to iterate over the individual voxels in the chunk, instead of doing chunk.iter() which is likely more costly.

## Idea to cache calculated occlusion for the "all-voxel" fallback

If we ignore complex self-occlusion (voxels blocking other voxels *within* the same chunk) and only look at **"Which faces of a cube are visible from this angle?"**, there are exactly **8** distinct states (The 8 Octants).

1.  Top-Right-Front
2.  Top-Right-Back
3.  Top-Left-Front
4.  Top-Left-Back
5.  Bottom-Right-Front
6.  ...and so on.

### Why Caching 8 Lists is Dangerous
While caching the results for these 8 states sounds ideal, it has a memory pitfall.
*   A naive implementation would store `8 * Vec<Instance>` per chunk.
*   For a surface-heavy chunk, that might be ~4,000 instances per list.
*   `8 lists * 4,000 items * 16 bytes` ≈ **512 KB per chunk**.
*   If you have 1,000 chunks loaded, that is **500 MB of RAM** just for fallback data that you rarely use.

### The Superior Solution: "The Visibility Mask"
Instead of caching 8 separate lists, you calculate **one** list of "Surface Voxels" when the chunk loads (on a worker thread), but you annotate each voxel with a **bitmask** of which faces are exposed.

This reduces your main-thread loop from **32,768** iterations (all voxels) to roughly **2,000 - 4,000** (only surface voxels), and the check becomes a single CPU instruction.

#### 1. The Structure (Computed Once on Worker Thread)
When `chunk.generate_mesh()` runs, also generate this lightweight fallback list:

```rust
struct FallbackVoxel {
    // Packed local coordinate (e.g. 5 bits for X, 5 for Y, 5 for Z)
    // u16 is enough for 32x32x32 (15 bits needed)
    packed_pos: u16, 
    
    // Bitmask of exposed faces:
    // bit 0: +X (Right), bit 1: -X (Left)
    // bit 2: +Y (Top),   bit 3: -Y (Bottom)
    // bit 4: +Z (Front), bit 5: -Z (Back)
    visible_faces: u8, 
}

// Store this in your Chunk struct
fallback_cache: Vec<FallbackVoxel>,
```

#### 2. The Loop (Runs on Main Thread)
Now, inside `render()`, you determine which "faces" the camera wants to see based on where the camera is relative to the chunk.

```rust
// 1. Determine the "Demand Mask" based on Camera Position vs Chunk Center
// If camera is to the Right (+X), we want to see Right faces (+X).
// If camera is to the Left (-X), we want to see Left faces (-X).
let dx = camera_pos.x - chunk_center.x;
let dy = camera_pos.y - chunk_center.y;
let dz = camera_pos.z - chunk_center.z;

let mut demand_mask = 0u8;
if dx > 0.0 { demand_mask |= 1 << 0; } // Camera is Right, show Right faces
else        { demand_mask |= 1 << 1; } // Camera is Left, show Left faces

if dy > 0.0 { demand_mask |= 1 << 2; } // Camera is Top
else        { demand_mask |= 1 << 3; } // Camera is Bottom

if dz > 0.0 { demand_mask |= 1 << 4; } // Camera is Front
else        { demand_mask |= 1 << 5; } // Camera is Back

// 2. Iterate ONLY the surface voxels (Pre-computed list)
for entry in &chunk.fallback_cache {
    // The Magic: Bitwise AND.
    // If the voxel has ANY exposed face that matches our demand, render it.
    if (entry.visible_faces & demand_mask) != 0 {
        cpu_prepopulated_instances.push(unpack(entry.packed_pos));
    }
}
```

### Why this solves your problem
1.  **Massive Iteration Reduction:** You stop iterating 32k voxels. You only iterate the ~10% that are actually on the surface (the "shell").
2.  **No Logic:** The loop body contains no `if/else` logic, no memory lookups for neighbors, and no coordinate math. It's just `&` and `push`.
3.  **Correctness:** It handles your "Camera Position" requirement perfectly. If you are above the chunk, `demand_mask` includes the "Top" bit. If a voxel is on the bottom of the chunk and only its *bottom* face is exposed, `visible_faces & demand_mask` will be 0, and it will be skipped.

This is the standard technique for software rasterizers and voxel engines to handle occlusion quickly without full raytracing.

---

You say "When chunk.generate_mesh() runs, also generate this lightweight fallback list", but I want to display the fallback while the mesh is being generated, since it will be replaced after and the mesh used in its place for the remainder of the program execution. Would it not be better to calculate this separately and keep it in a cache, rather than a chunk, and remove it from the cache once the meshed version becomes available.

---


You have hit the nail on the head. That is exactly the correct architectural approach.

Since `generate_mesh()` (greedy meshing, ambient occlusion, vertex formatting) is slow, you should race a **"Shell Extraction"** job against it. The shell extraction is $O(N)$ and extremely simple, so it will finish long before the mesh, bridging the gap.

Here is the blueprint for the **Zero-Stutter / Zero-Hole** pipeline:

### 1. The Data Structure
Don't store full `VoxelInstanceRaw` structs (which are large) in your cache. Store a compact "Surface Voxel" that allows for the camera-facing optimization we discussed.

```rust
// Store this in a separate resource: FallbackCache
// HashMap<ChunkPos, Vec<SurfaceVoxel>>
struct SurfaceVoxel {
    // Packed local position (5 bits X, 5 bits Y, 5 bits Z)
    // 0..32 fits in 5 bits. Total 15 bits.
    packed_pos: u16, 
    
    // The "Camera-Facing" Mask
    // bit 0: +X exposed, bit 1: -X exposed
    // bit 2: +Y exposed, bit 3: -Y exposed
    // bit 4: +Z exposed, bit 5: -Z exposed
    visible_faces: u8, 
}
```

### 2. The Lifecycle
This pipeline ensures something is always on screen without stalling the main thread.

1.  **Chunk Data Arrives:**
    *   **Action:** Immediately spawn a background task: `CalculateShellJob`.
    *   *(Optional for absolute 0-hole):* On the very first frame, if the shell isn't ready, run a "super-low-res" pass on the Main Thread (step_by 4) just so the space isn't empty.

2.  **Shell Job (Worker Thread):**
    *   Iterate all 32k voxels.
    *   Check neighbors.
    *   If a voxel is solid and has an Air neighbor, create a `SurfaceVoxel`, calculate its `visible_faces` mask, and add to a `Vec`.
    *   **Result:** A lightweight list of ~2,000 items (vs 32,000).

3.  **Shell Ready (Main Thread):**
    *   Insert into `FallbackCache`.
    *   **Render Loop:** If `!mesh_ready` and `fallback_ready`:
        *   Retrieve `Vec<SurfaceVoxel>`.
        *   Calculate camera `demand_mask` (e.g., "I am above and to the left").
        *   Iterate the Vec, `bitwise AND` the masks.
        *   If match: unpack position -> `cpu_prepopulated_instances`.

4.  **Mesh Ready (Main Thread):**
    *   Upload Mesh to GPU.
    *   **Delete** the entry from `FallbackCache` to free RAM.

### 3. The Algorithm: Shell Extraction with Masks
This is the logic you run on the worker thread. It is fast enough that the "fallback" will appear almost instantly.

```rust
fn generate_shell(chunk: &Chunk) -> Vec<SurfaceVoxel> {
    let mut shell = Vec::with_capacity(2048); // Heuristic start size

    for x in 0..32 {
        for y in 0..32 {
            for z in 0..32 {
                if !chunk.get(x, y, z).is_solid() { continue; }

                let mut mask = 0u8;
                // Check 6 neighbors (careful with boundaries)
                // If neighbor is AIR (or boundary), set the bit.
                if is_air(chunk, x+1, y, z) { mask |= 1 << 0; } // +X
                if is_air(chunk, x-1, y, z) { mask |= 1 << 1; } // -X
                if is_air(chunk, x, y+1, z) { mask |= 1 << 2; } // +Y
                if is_air(chunk, x, y-1, z) { mask |= 1 << 3; } // -Y
                if is_air(chunk, x, y, z+1) { mask |= 1 << 4; } // +Z
                if is_air(chunk, x, y, z-1) { mask |= 1 << 5; } // -Z

                if mask != 0 {
                    let packed = (x as u16) | ((y as u16) << 5) | ((z as u16) << 10);
                    shell.push(SurfaceVoxel { 
                        packed_pos: packed, 
                        visible_faces: mask 
                    });
                }
            }
        }
    }
    shell
}
```

### 4. The Render Loop (Main Thread)
Now your render loop is blazing fast because you are iterating a cached, small list (the shell) and doing a single integer check per item.

```rust
// Inside render() for a chunk without a mesh
if let Some(shell) = self.fallback_cache.get(&chunk_pos) {
    // 1. Calculate Demand Mask (Where is the camera?)
    let d = camera_pos - chunk_world_center;
    let mut demand = 0u8;
    if d.x > 0.0 { demand |= 1 << 0; } else { demand |= 1 << 1; }
    if d.y > 0.0 { demand |= 1 << 2; } else { demand |= 1 << 3; }
    if d.z > 0.0 { demand |= 1 << 4; } else { demand |= 1 << 5; }

    // 2. Fast Filter
    for v in shell {
        // "Do I want to see any face that this voxel is showing?"
        if (v.visible_faces & demand) != 0 {
            let (x, y, z) = unpack(v.packed_pos);
            cpu_prepopulated_instances.push(...);
        }
    }
}
```

### Summary of Benefits
1.  **Main Thread Safety:** You moved the heavy O(N) voxel iteration and memory access to a worker thread.
2.  **Reduced Iterations:** The Main Thread loops over ~2,000 surface voxels instead of 32,768 raw voxels.
3.  **Culling:** The `demand_mask` automatically culls 50% of the surface voxels that are facing away from you (Backface Culling for voxels).
4.  **Memory:** Temporary. The moment the real mesh arrives, you drop this data.
