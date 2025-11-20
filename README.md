# voxelot

A high-performance sparse voxel engine using hierarchical chunks with Roaring Bitmaps.

![Voxelot viewer screenshot](voxelot.png)

## Key Features

### 🧠 Recursive "Chunks All The Way"
Unlike traditional engines that separate "World", "Region", and "Chunk" classes, Voxelot uses a **uniform recursive structure**. The entire World is a Chunk, which contains sub-Chunks, down to the leaf level.
- **Unified Logic**: Culling, raycasting, and storage work identically at all scales.
- **Massive Scale**: A hierarchy depth of 4 gives you a **65,536³** voxel world (281 trillion potential voxels) addressable with simple integer coordinates.

### 🚀 Bitwise Performance Pipeline
Voxelot leverages **Roaring Bitmaps** and bitwise arithmetic for extreme performance:
- **3-Stage Culling**:
    1. **Frustum AABB**: Standard intersection check.
    2. **Marginal Bitmaps**: `u16` bitmasks (px/py/pz) instantly reject empty slices.
    3. **Roaring Intersection**: Exact boolean operations isolate *only* the visible voxels within the frustum.
- **Binary Greedy Meshing**: Meshes are generated using bitwise face detection (`col & !(col << 1)`), merging faces into optimal quads faster than standard iteration.
- **Mesh Streaming**: Greedy-meshed leaf chunks stream in gradually with fallback previews.
- **Parallel Processing**: Multi-threaded chunk updates and culling using `rayon`.
- **Visibility Caching**: Spatial hashing reuses visibility results to minimize per-frame work.

### 🎨 Cinematic Rendering
A modern rendering pipeline built on `wgpu` that goes beyond standard voxel looks:
- **SSAO**: Horizon-based ambient occlusion for depth perception.
- **Dual-Kawase Bloom**: High-quality, multi-pass bloom for glowing voxels.
- **Cinematic Depth of Field**: Bokeh-style blur with focus control.
- **Dynamic Atmosphere**: Day/night cycle with scattering-approximated fog and HDR skybox.

### 💾 Optimal Sparsity
Storage is proportional to *entropy*, not volume.
- **Roaring Bitmaps**: Compressed bitmap indices mean empty space takes zero memory.
- **Compact Storage**: An 85 million voxel world can be stored in just **~10MB** (`.oct` format).
- **Exact Queries**: No false positives; if the bitmap says a voxel is there, it's there.

## Quick Start

```bash
# Run the interactive viewer (release recommended)
cargo run --release --bin voxelot

# Show generator help (Rust)
cargo run --bin generate_world -- --help
```

## Interactive Viewer Controls

**Movement:**
- `WASD` - Move forward/left/backward/right
- `Space` / `Shift` - Move up/down
- `Right Mouse + Drag` - Free-look

**Rotation:**
- `Arrow Left/Right` - Yaw left/right
- `Arrow Up/Down` - Pitch up/down

**Speed Tuning:**
- `-` / `+` - Decrease/increase camera speed multiplier
- `0` - Reset camera speed multiplier

**Runtime Configuration:**
- `Q` / `E` - Decrease/increase camera LOD subdivide distance
- `Z` / `C` - Decrease/increase draw distance (far plane)
- `K` / `L` - Decrease/increase chunk LOD render distance (100-5000 units)
- `T` - Toggle time of day cycle (pause/resume)
- `F` / `G` - Decrease/increase fog density
- `ESC` - Save config and quit

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

**Configuration:** The viewer uses `config.toml` for all settings including world file path, camera position, rendering options, and visual effects. Edit this file to customize your experience.


## Configuration

Configuration is saved in structured TOML `config.toml`.

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

1. `voxel_generator_tiles.py` – Python prototype for synthesizing tile footprints.
2. `generate_world` – Rust generator, faster for large areas, writes compressed `.oct` + metadata.

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
## Credits
The skybox image is CC0, downloaded from [Polyhaven](https://polyhaven.com).

