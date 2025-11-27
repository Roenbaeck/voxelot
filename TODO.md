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

## Idea to cache calculated occlusion for the "all-voxel" fallback

If we ignore complex self-occlusion (voxels blocking other voxels *within* the same chunk) and only look at **"Which faces of a cube are visible from this angle?"**, there are exactly **8** distinct states (The 8 Octants).

1.  Top-Right-Front
2.  Top-Right-Back
3.  Top-Left-Front
4.  Top-Left-Back
5.  Bottom-Right-Front
6.  ...and so on.

### The Superior Solution: "The Visibility Mask"
Instead of caching 8 separate lists, you calculate **one** list of "Surface Voxels" when the chunk loads, and store these in a "shell" cache, and annotate each voxel with a **bitmask** of which faces are exposed.

This reduces your main-thread loop from **32,768** iterations (all voxels) to perhaps **2,000 - 4,000** (only surface voxels), and the check becomes a single CPU instruction.

#### The Loop (Runs on Main Thread)
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
for entry in &chunk.shell_cache {
    // The Magic: Bitwise AND.
    // If the voxel has ANY exposed face that matches our demand, render it.
    if (entry.visible_faces & demand_mask) != 0 {
        cpu_prepopulated_instances.push(entry);
    }
}
```

I want to display the shell fallback while the mesh is being generated, since it will be replaced after and the mesh used in its place for the remainder of the program execution. We should remove it from the cache once the meshed version becomes available.

### The Algorithm: Shell Extraction with Masks
This is hopefully fast enough that the "shell fallback" will appear almost instantly.

```rust
fn generate_shell(chunk: &Chunk) -> Vec<ShellVoxel> {
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
                    shell.push(ShellVoxel { 
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

Here is the revised `render()` logic:

```rust
// Inside render() loop (psuedocode)
if !mesh_ready {
    // 1. Check if we have the clean "Shell" from the worker thread
    if let Some(shell) = self.shell_cache.get(&chunk_pos) {
        // FAST PATH: Use the cached shell (runs in microseconds)
        self.render_from_shell(shell, camera_pos);
    } 
    else {
        let shell = self.generate_shell(chunk);
        // add to shell_cache
        self.shell_cache.insert(chunk_pos, shell);
        self.render_from_shell(shell, camera_pos);
        
    }
}
```

**Optimize the Loop (Remove Vector Allocations).**
Do not allocate `VoxelInstanceRaw` structs. Write directly to a pre-allocated GPU staging buffer if possible, or a persistent `Vec` that you clear every frame.

