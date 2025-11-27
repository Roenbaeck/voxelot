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

You are currently avoiding holes by **freezing the entire game loop** until the hole is filled.

Here is exactly what happens in your current `render()` function:

1.  **CPU Starts Frame:** You begin the render loop.
2.  **The "Stutter" Loop:** You hit an unmeshed chunk. The code enters that `for ((x, y, z), voxel) in chunk.iter()` loop.
3.  **Blocking:** While inside this loop, **nothing else happens**. Input handling stops, physics stops, and importantly, **no commands are sent to the GPU**.
4.  **GPU Waiting:** The GPU is sitting idle (or finishing the previous frame), waiting for the CPU to tell it what to do.
5.  **Completion:** Only after you have iterated every single voxel and pushed them into `cpu_prepopulated_instances` does the loop finish.
6.  **Submission:** Finally, you call `queue.write_buffer` and `encoder.submit`.

**Result:** The frame is "complete" (no holes), but it arrives late (stutter).

---

### The Problem with the "Background Shell" Optimization
If you strictly follow the "Background Shell" optimization I proposed previously, **you will introduce holes.**

If the `ShellJob` takes 10ms to finish on a background thread, but your frame renders in 5ms, you will render 2 frames where the chunk is invisible (a hole) before the shell data is ready.

### How to get "No Holes" AND "High FPS"

To maintain your strict "no holes" requirement while eliminating the stutter, you need a **Hybrid Approach**:

**Logic:** "If the optimized cached shell is ready, use it. If it is NOT ready (first frame), panic and do a 'Fast Fallback' right here on the main thread."

Here is the revised `render()` logic:

```rust
// Inside render() loop
if !mesh_ready {
    // 1. Check if we have the clean "Shell" from the worker thread
    if let Some(shell) = self.fallback_cache.get(&chunk_pos) {
        // FAST PATH: Use the cached shell (runs in microseconds)
        self.render_from_shell(shell, camera_pos);
    } 
    else {
        // 2. PANIC MODE (First frame only)
        // The shell isn't ready yet, but we FORBID holes.
        // We must generate data NOW, on the Main Thread.
        
        // OPTIMIZATION: Do not use the full loop. 
        // Use the "Step-by-2" method. It is 8x faster than your current loop.
        // It guarantees coverage (no holes), just slightly blocky for 1 frame.
        self.generate_immediate_fallback_step_by_2(chunk);
        
        // Triggers the background job so next frame hits the Fast Path
        self.trigger_shell_generation_job(chunk_pos); 
    }
}
```

### Why this is the winner:
1.  **Frame 0 (The Pop-in):** You teleport. The chunk appears instantly. You pay a tiny CPU cost (using the `step_by(2)` optimization), so the FPS might dip from 144 to 120, but not to 10.
2.  **Frame 1+:** The background thread finishes the clean "Shell". You switch to that. It looks perfect (1:1 resolution) and is extremely fast to render.
3.  **Frame N:** The Mesh finishes. You switch to the Mesh.

This strategy respects your **Zero Holes** rule but removes the massive loop that is currently killing your performance.

I have a feeling the immediate fallback path (panic mode) will be the hot path here, so why even bother to have a ShellJob. Can't we rely on the panic mode alone? The only reasonable way for ShellJob to have a purpose would be if we send it chunks that are far in the distance from where we are. Is that what you were thinking?

You are asking the right questions. You are distinguishing between **Latency** (time to first pixel) and **Throughput** (cost per frame).

The answer depends on **how long your Meshing takes**.

### The Scenario
1.  **Frame 0:** You look at a new chunk.
2.  **Panic Mode:** You iterate the chunk (on Main Thread) to avoid a hole.
3.  **Frame 1:** Is the mesh ready?
    *   **If YES:** Then `ShellJob` is useless. You used the fallback for 1 frame, then replaced it.
    *   **If NO:** You have to run **Panic Mode** again. And again. And again.

### Why ShellJob exists
Greedy meshing (combining faces) is computationally expensive. It might take **10 to 50 frames** (depending on queue depth) for the mesh to actually arrive.

If you rely **only** on Panic Mode, you are forcing the Main Thread to iterate voxels **every single frame** for those 50 frames.
*   If you have 1 chunk waiting: It's fine.
*   If you teleport and have 20 chunks waiting: You are re-iterating **160,000+ voxels per frame** on the main thread. This will drag your FPS down while you wait for the meshes.

**The `ShellJob` is the bridge.** It is much faster than Meshing. It finishes in ~1 frame. It allows the Main Thread to stop doing work ("Panic Mode") and switch to a cheap, cached lookup ("Shell Mode") while waiting for the slow "Mesh Mode."

---

### However... The "Good Enough" Solution (Simplicity)

If you want to keep your code simple and avoid managing a third state (`ShellCache`), you can stick to **Panic Mode Only**, provided you make it dirt cheap.

If you optimize the Panic Loop enough, running it every frame for 50 frames might not hurt.

**The "Just Panic" Strategy (Recommended for now):**

Forget the background shell job. Optimize the Main Thread fallback loop so aggressively that running it every frame doesn't matter.

**1. Use the `step_by(2)` strided iteration.**
This reduces the workload by **8x** (87.5% reduction).

**2. Optimize the Loop (Remove Vector Allocations).**
Do not allocate `VoxelInstanceRaw` structs. Write directly to a pre-allocated GPU staging buffer if possible, or a persistent `Vec` that you clear every frame.

**3. Frustum Cull.**
Strictly ensure you only run this loop for chunks the camera actually sees.

**The Code:**
```rust
// In render() - The "Just Panic" Approach
if !mesh_ready && self.camera_sees(chunk_aabb) {
    // Run this EVERY frame until mesh arrives.
    // It must be extremely fast.
    
    // Step 2: Skip every other voxel (8x speedup)
    for x in (0..32).step_by(2) {
        for y in (0..32).step_by(2) {
            for z in (0..32).step_by(2) {
                // Direct array access (unsafe for speed if you trust bounds)
                let voxel = chunk.get_fast(x, y, z); 
                
                if voxel.is_solid() {
                    // Push 2x sized instance
                    // Reuse a persistent vector, do NOT `Vec::new()`
                    self.instance_cache.push(VoxelInstanceRaw {
                        pos: [x_w, y_w, z_w],
                        scale: 2.0, // Fills the gaps
                        ..Default::default()
                    });
                }
            }
        }
    }
}
```

### Conclusion
Start with the **"Just Panic" (Optimized)** approach.

1.  Implement `step_by(2)`.
2.  See if the stutter disappears.
3.  If you still see FPS drops when loading many chunks, **then** implement the `ShellJob` caching.

Most likely, the `step_by(2)` optimization alone will make the loop fast enough that you don't need the complexity of a Shell Cache system.
