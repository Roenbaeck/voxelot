# voxelot

A high-performance sparse voxel engine using hierarchical chunks with LOD and Roaring Bitmaps.

## Features

- **Hierarchical LOD**: Automatic subdivision and merging based on camera distance
- **Fast Culling**: 3-layer performance (marginal → frustum → presence)
- **Memory Efficient**: ~1 byte per voxel with Roaring Bitmap compression
- **Bounded-but-Huge Worlds**: 16^n units (e.g., 16⁴ = 65,536³ voxels)
- **Exact Queries**: No false positives via presence bitmaps
- **True "Chunks All the Way"**: World IS a Chunk - uniform structure at every level!

### Quick Start

```bash
# Test hierarchical LOD system
cargo run --bin test_hierarchical_lod

# Test projection bit propagation
cargo run --bin test_projection_bits

# Run the interactive viewer
cargo run --release --bin viewer_hierarchical

# Run benchmarks
cargo run --release --bin bench_culling
```

### Data Generation (Voxelizing Real World Tiles)

Two generators are now provided:

1. `osm_voxel_generator.py` – legacy one-shot Overpass query producing a coarse static dump `osm_voxels.txt`.
2. `voxel_generator_tiles.py` – new dry-coded tile-based prototype (no external deps yet) that rasterizes synthetic building footprints per Web Mercator tile and writes `osm_voxels_new.txt` plus a meta file.

Planned evolution of the tile generator:
- Replace synthetic footprints with real vector tile (MVT) decoding.
- Add height inference heuristics (already stubbed) and roof shape variants.
- Introduce per-tile material palettes & compressed binary output.
- Hook into streaming residency so tiles load/unload around the camera.

Output compatibility: Both generators currently emit ASCII lines `x y z voxel_type` so the existing Rust ingestion path can remain unchanged while the pipeline is upgraded.

See `VOXEL_GENERATOR_REVAMP.md` for detailed architecture and roadmap.

### Interactive Viewer Controls

**Movement:**
- `WASD` - Move forward/left/backward/right
- `Space` - Move up
- `Shift` - Move down
- `Mouse` - Look around

**Runtime Configuration:**
- `[` / `]` - Decrease/increase LOD subdivide distance
- `-` / `=` - Decrease/increase draw distance (far plane)
- `ESC` - Save config and quit

Configuration is saved to `render_config.txt` and automatically loaded on startup. You can edit this file directly to change render settings without recompiling.

### Architecture

```rust
enum Voxel {
    Solid(VoxelType),          // A solid voxel with a type
    Chunk(Box<Chunk>),          // A sub-chunk (for subdivision)
}

struct Chunk {
    px: u16,                    // Marginal X bitmap (fast rejection)
    py: u16,                    // Marginal Y bitmap
    pz: u16,                    // Marginal Z bitmap
    presence: Bitmap,           // Exact presence (Roaring)
    voxels: Vec<Voxel>,         // Indexed by rank - uniform at all levels!
}

struct World {
    root: Chunk,                // The world IS a chunk!
    hierarchy_depth: u8,        // Determines world size: 16^depth
    chunk_size: u32,            // Base chunk size (always 16)
}
```

**World Sizes by Hierarchy Depth:**
- Depth 1: 16³ = 4,096 voxels
- Depth 2: 256³ = 16,777,216 voxels
- Depth 3: 4,096³ = 68,719,476,736 voxels
- Depth 4: 65,536³ = 281,474,976,710,656 voxels

**"Chunks all the way" philosophy:**
- Uniform Chunk structure at every level of hierarchy
- Each position can be either Voxel::Solid or Voxel::Chunk
- Rank-based indexing works identically for leaves and branches
- Projection bits (px/py/pz) propagate up the hierarchy for efficient culling
- No special-casing - same code handles all levels

**Hierarchical LOD Features:**
- Automatic subdivision when camera approaches (< 50 units)
- Automatic merging when camera recedes (> 100 units)
- Recursive frustum culling descends into sub-chunks
- Marginal culling works at all hierarchy levels
- Distance-based rendering decisions

**Implemented Systems:**
- 6-plane frustum culling (fixed cross-product bug)
- Marginal projection bit culling
- Occlusion culling with front-to-back traversal
- Parallel chunk processing with rayon
- Spatial visibility caching

See [OUTLINE.md](OUTLINE.md) for the full design evolution.

## Project Structure

