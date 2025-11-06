# voxelot

A high-performance sparse voxel engine using hierarchical chunks with LOD and Roaring Bitmaps.

## Features

- **Hierarchical LOD**: Automatic subdivision and merging based on camera distance
- **Fast Culling**: 3-layer performance (marginal → frustum → presence)
- **Memory Efficient**: ~1 byte per voxel with Roaring Bitmap compression
- **Infinite Worlds**: Sparse HashMap-based chunk storage
- **Exact Queries**: No false positives via presence bitmaps
- **Recursive Structure**: "Chunks all the way" - uniform at every level

### Quick Start

```bash
# Test hierarchical LOD system
cargo run --bin test_hierarchical_lod

# Test projection bit propagation
cargo run --bin test_projection_bits

# Run benchmarks
cargo run --release --bin bench_culling
```

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
```

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

