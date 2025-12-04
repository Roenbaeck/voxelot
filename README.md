# voxelot

A high-performance sparse voxel engine using hierarchical chunks with Roaring Bitmaps. Theoretical framework is of human origin, but almost all code is AI-generated. The exception being when AI got completely stuck, usually on some trivial thing like an unclosed bracket. It is an experiment to see if a whole game can be created from scratch using AI-assisted coding, where the AI is assisted by experienced programmers. 

Initial code by Claude Sonnet 4.5, GPT-5.1 and Gemini 2.5 Pro, then the bulk by Raptor Mini (occasionally unstuck by Grok Code Fast), and lately optimizations by Gemini 3.0 Pro and Claude Opus 4.5. For currently being a "free" model within GitHub Copilot, Raptor Mini is surprisingly capable, and I've only had to turn to what we believe are more capable models occasionally. 

AI can still run wild and mess up large portions of the code base. If you experiment like this, git commit often. Reverting is your friend.

![Voxelot viewer screenshot](voxelot.png)

## Key Features

### Recursive "Chunks All The Way"
Unlike traditional engines that separate "World", "Region", and "Chunk" classes, Voxelot uses a **uniform recursive structure**. The entire World is a Chunk, which contains sub-Chunks, down to the leaf level. Each chunk, regardless of level, has a side length of 16.
- **Unified Logic**: Culling, raycasting, and storage work identically at all scales.
- **Massive Scale**: A hierarchy depth of 4 gives you a **(16³)⁴** voxel world (281 trillion potential voxels) addressable with simple integer coordinates.

### Bitwise Performance Pipeline
Voxelot leverages **Roaring Bitmaps** and bitwise arithmetic for extreme performance:
- **Multi-stage Culling**:
    1. **Frustum AABB**: Standard intersection check.
    2. **Marginal Bitmaps**: `u16` bitmasks (px/py/pz) instantly reject empty slices.
    3. **Roaring Intersection**: Exact boolean operations isolate *only* the visible voxels within the frustum.
    4. **Visible shell cache**: Non-leaf chunks calculate a shell with a visibility mask, quickly discarding non-visible chunks from any angle.
    5. **Submerged chunks**: Chunks so deep below water level that they would not be visible are culled.
- **Binary Greedy Meshing**: Meshes are generated using bitwise face detection (`col & !(col << 1)`), merging faces into optimal quads faster than standard iteration.
- **Mesh Streaming**: Greedy-meshed leaf chunks stream in gradually with fallback previews.
- **Parallel Processing**: Multi-threaded chunk updates and culling using `rayon`.
- **Visibility Caching**: Spatial hashing reuses visibility results to minimize per-frame work.

### Cinematic Rendering
A modern rendering pipeline built on `wgpu` that goes beyond standard voxel looks:
- **SSAO**: Horizon-based ambient occlusion for depth perception.
- **SSR**: Screen-space reflections and lighting.
- **Dual-Kawase Bloom**: High-quality, multi-pass bloom for glowing voxels.
- **Cinematic Depth of Field**: Bokeh-style blur with focus control.
- **Dynamic Atmosphere**: Day/night cycle with scattering-approximated fog and rotating HDR skybox.
- **Realistic water**: Animated water with depth fading, reflections, and shoreline foam.

### Optimal Sparsity
Storage is proportional to *entropy*, not volume.
- **Roaring Bitmaps**: Compressed bitmap indices mean empty space takes zero memory.
- **Compact Storage**: A 500 million voxel world can be stored in just **~50MB** (`.oct` format).
- **Exact Queries**: No false positives; if the bitmap says a voxel is there, it's there.

### Multiple levels of detail
Chunks have different rendering modes depending on state and distance.
- **Shell**: Unmeshed chunks near the camera render as individual voxels from the shell of the chunk.
- **Meshed**: Once a chunk is meshed, the meshed surfaces are rendered for chunks near the camera.
- **Envelope**: At medium distance uniformly colored, and thereby simpler meshes are used.
- **Bounding box**: At large distance, or when envelopes are missing, chunks are rendered as a tight bounding box.

## Quick Start

```bash
# Run the interactive viewer (release recommended). Use the default `config.toml` in the
# current working directory, or pass a custom config via `--config` or as a single positional
# argument (the positional argument is identical to providing `--config` explicitly):
cargo run --release --bin voxelot
cargo run --release --bin voxelot -- --config worlds/flat_city_test.toml
cargo run --release --bin voxelot -- worlds/flat_city_test.toml

# Show generator help (Rust)
cargo run --bin generate_world -- --help
```

## Interactive Viewer Controls

**Movement:**
- `WASD` - Move forward/left/backward/right
- `Q` / `E` - Move down/up
- `Right Mouse + Drag` - Free-look

**Rotation:**
- `Arrow Left/Right` - Yaw left/right
- `Arrow Up/Down` - Pitch up/down

**Speed Tuning:**
- `-` / `+` - Decrease/increase camera speed multiplier
- `0` - Reset camera speed multiplier

- **Runtime Configuration:**
- `PageDown` / `PageUp` - Decrease/increase camera LOD subdivide distance
- `Z` / `C` - Decrease/increase draw distance (far plane)
- `K` / `L` - Decrease/increase chunk LOD render distance (100-5000 units)
- `F` / `G` - Decrease/increase fog density
- `ESC` - Save config and quit

- **Time & Environment:**
- `T` - Cycle time of day (day → dusk → night → dawn)
- `Y` / `M` - Lower/raise water level

**Effects Controls:**
- `B` - Toggle Bloom
- `N` - Toggle SSAO
- `H` - Toggle SSAO debug view
- `/` - Toggle Depth of Field (DoF)
- `X` - Toggle Kawase DoF blur
- `,` / `.` - Decrease/increase DoF focal distance
- `[` / `]` - Decrease/increase DoF focal range
- `;` / `'` - Decrease/increase DoF blur strength
- `U` / `I` - Decrease/increase Kawase offset
- `O` / `P` - Decrease/increase Kawase iterations
- `F1` / `F2` - Decrease/increase SSAO sample count
- `F3` / `F4` - Decrease/increase SSAO radius
- `F5` - Toggle GUI overlay (FPS counter & debug stats)
- `R` - Toggle SSR (Screen Space Reflections)
- `J` - Toggle HZB (Hierarchical Z-Buffer) culling

## Configuration

The viewer reads and writes settings in a unified TOML configuration file.

- Default behavior: If a `config.toml` file is present in the current working directory, the viewer will use that by default.
- Override with a file path: Use the `--config <path>` flag or pass the path as the first positional argument (e.g. `cargo run --bin voxelot -- worlds/flat_city_test.toml`). The viewer will load from and save to that file.

This file controls the world file path, camera position, rendering options, and visual effects. Edit it to customize your experience.

## Worlds & Per-World Configuration

- Place per-world TOML config files in `worlds/` (some example files are already present). If you prefer to keep per-world settings under version control, pass them with `--config worlds/<your_world>.toml`.
- The viewer reads `.oct` files listed inside your TOML configuration and launches with those settings.

**Structure:**

```toml
[world]        # world file path, palette, camera position
[rendering]    # LOD distances, camera planes, FOV, speed multiplier
[atmosphere]   # fog density, time_of_day
[effects.bloom]          # bloom parameters
[effects.depth_of_field] # DoF parameters
[effects.ssao]           # SSAO parameters
[shadows]      # shadow map size, darkness, PCF settings
[performance]  # mesh cache budget, worker count, upload limits
[debug]        # debug overlay toggles
```

**Example Configuration:**

```toml
[shadows]
map_size = 8192
darkness = 2.0
backface_ambient_scale = 0.25
pcf_radius = 3.0
pcf_poisson_samples = 8

[effects.ssao]
enabled = true
sample_count = 16
radius = 100.0
strength = 1.0
```

## Palette Configuration

The engine uses a simple text-based palette format (`palette.txt`) to define voxel materials.

**Format:**
Each line defines a voxel type:
`index R G B A [emitR emitG emitB emitStrength]`

- `index`: Voxel type ID (0-255)
- `R G B A`: Albedo color (0-255)
- `emitR G B`: Emissive color (0-255, optional)
- `emitStrength`: Emissive intensity (0-255, optional)

**Example:**
```text
# Index R   G   B   A   EmitR EmitG EmitB Strength
1       30  200 50  255                    # Grass (Non-emissive)
2       255 50  50  255 255   100   100   255  # Red Neon (Glowing)
```

## Data Generation

Two generators are provided:

1. `generate_world` – Rust generator, faster for large areas, writes compressed `.oct` + metadata.
2. `voxel_generator_tiles.py` – Legacy Python prototype for synthesizing tile footprints (not maintained).

**File Format:** The preferred runtime format is the compact binary octree (`.oct`).

**Example: Generating a World (Rust)**
```bash
cargo run --bin generate_world -- \
    --center-lon=-74.0060 \
    --center-lat=40.7128 \
    --zoom=15 \
    --radius=2 \
    --voxels-per-tile=128 \
    --meters-per-voxel=1.25 \
    --max-height-voxels=192 \
    --seed=1337 \
    --output-name=worlds/world_1 \
    --format=oct
```

Note: The `generate_world` binary does not currently accept a `--config` path — the command-line flags listed above control generation parameters. To run with a world-config TOML file, launch the viewer with `--config` and adjust parameters to re-run the generator yourself if needed.

## Architecture

```rust
enum Voxel {
    Solid(VoxelType),           // A solid voxel with a type
    Chunk(Arc<Chunk>),          // A sub-chunk (for subdivision)
}

struct Chunk {
    px: u16,                    // Marginal X bitmap (fast rejection)
    py: u16,                    // Marginal Y bitmap
    pz: u16,                    // Marginal Z bitmap
    presence: Bitmap,           // Exact presence (Roaring)
    voxels: Vec<Voxel>,         // Indexed by rank - uniform at all levels!
}

struct World {
    root: Arc<Chunk>,           // The world IS a chunk!
    hierarchy_depth: u8,        // Determines world size: 16^depth
    chunk_size: u32,            // Base chunk size (always 16)
}
```

**"Chunks all the way" philosophy:**
- Uniform Chunk structure at every level of hierarchy
- Each position can be either Voxel::Solid or Voxel::Chunk
- Rank-based indexing works identically for leaves and branches
- Projection bits (px/py/pz) propagate up the hierarchy for efficient culling
- No special-casing - same code handles all levels

## GUI Overlay — Metrics & Explanations

The viewer displays a lightweight real-time overlay (top-left) showing runtime and profiling information. Below is an explanation of each metric so you can interpret the numbers correctly.

- FPS: Frames-per-second measured by the viewer. Lower FPS indicates heavier workload or GPU/CPU bottlenecks.
- Visible: Number of visible chunk instances that reached the draw pass this frame (includes lod/displaced/instanced entries).
- Meshed: The number of meshed chunks currently present in the mesh cache.
- Pending: Number of chunk meshing jobs pending inside the pipeline/queue.
- Jobs/s: Mesh-worker throughput — how many mesh jobs are completed per second.
- Process: Process memory in MiB (Resident Set Size / RSS); this is what `sysinfo` reports for the voxelot process and represents main-process physical memory.
    - Mesh cache: App-level mesh cache size (MiB) and the configured mesh cache budget. This is memory used by GPU mesh data that the app tracks (vertex/index buffers staged by the app), not the OS-size.
    - Envelopes: Memory used by envelope meshes in the mesh cache (MiB) — these are the lower-detail meshes used as fallbacks.

- GPU tracked: The in-app tracked estimate of GPU-resident allocations, shown in MiB.
    - This equals `gpu_buffer_bytes + gpu_texture_bytes` (buffers + textures) and covers all resources we explicitly track in the app (mega vertex/index buffers, indirect buffers, uniform buffers, offscreen textures, ping/pong buffers, etc.).
    - It is an estimate that includes any buffers and textures where we call the tracking helper (buffer+texture accounting). It does not always match the full OS-level GPU memory usage because the GPU driver and platform (e.g., Metal on macOS) often allocate additional driver-side memory (IOSurface/IOAccelerator) which is not accessible via `wgpu`.

    The per-resource breakdown includes:
    - Uniforms: Sum of UBO sizes (MiB) tracked by the app. This includes UBOs like `Uniforms`, Bloom/DoF/SSAO UBOs and Kawase level UBOs.
    - Mega VB/IB: Size in MiB of the “mega” vertex and index buffers (slab-allocated storage used to hold combined chunk meshes).
    - GPU Input: Size of the GPU-visible input buffer for culling / instance candidate information (MiB).
    - Indirects (mesh/env): Size of draw-indirect buffers used by the rendering system to perform multi-draw calls for meshes and envelopes.
    - Offscreen/Depth/Post: Sizes of the offscreen rendering targets (color), depth, and post-processing textures. These are commonly large contributors.
    - Kawase ping/pong: The ping/pong texture chain used for Kawase DoF blur (per level). Each entry denotes a ping/pong pair's total sizes.
    - Bloom ping/pong: The bloom texture pair used for multi-pass blur.
    - SSAO ping/pong: SSAO intermediate textures.
    - Shadow/Skybox: Shadow map bytes and skybox texture bytes. Shadow maps (especially large sizes like 8k or 16k) are one of the largest GPU allocations — reduce `shadow.map_size` if you're near VRAM limits.

- Cull / Group / Mesh / Instancing (ms): Per-frame timings (in milliseconds) for major pipeline steps measured on the CPU side:
    - Cull: Time it took to perform frustum + occlusion + GPU compute culling for this frame.
    - Group: CPU grouping / batching work before issuing draws.
    - Mesh: Total CPU time spent creating/generating meshes for this frame.
    - Instancing: Time spent doing the per-instance drawing path (binding, instanced draws, etc.).
- Draws: Number of actual draw calls issued in the frame.

## Credits
The skybox image is CC0, downloaded from [Polyhaven](https://polyhaven.com).
