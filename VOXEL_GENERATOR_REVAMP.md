# Voxel Generator Revamp

This doc summarizes the plan and progress for the tile-based voxel generator.

Current state
-------------
- `voxel_generator_tiles.py` is a pure Python prototype that synthesizes tile footprints and voxelizes them to ASCII text.
- `generate_world` is a Rust binary that implements the same procedural rules faster and writes `.oct` + metadata directly.

Goals
-----
- Preserve the rich procedural rules (roads, parks, water, buildings, heights) while achieving dramatically faster generation.
- Eliminate seams at tile boundaries.
- Support deterministic seeds so generator outputs are reproducible.
- Provide both a Python path for experimentation and a Rust path for production.

Seam Fixes and Findings
-----------------------
- The seam seen earlier was caused by tile boundary columns that were left empty by the original Python generator (rare cases of rounding/clamp/edge misalignment in `TileSpace.mercator_to_voxel`).
- Two parallel fixes were implemented:
  - Rust generator: computes tile-space mapping with centered coverage and fills ground by default; produced gap-free `world_1`, `world_2`, `world_3`.
  - Python generator: now pre-fills all columns with the base ground layer before placing roads, parks, and buildings (keeps existing procedural semantics and prevents gaps when tiles are stitched).

Future Work
-----------
- Rework polygon rasterization to use a consistent shared implementation (move to a small library crate) and ensure parity between Python and Rust.
- Add validation doctests or unit tests to check tile-edge coverage for any set of parameters.
- Integrate MVT/Mapbox Vector Tiles for real-world footprints in `generate_world`.
- Optionally add a `verify_tiles` small CLI that compares raw columns across adjacent tiles.

How to regenerate
-----------------
- For fast generation and octree output, use the Rust generator:

```bash
cargo run --bin generate_world -- --center-lon -74.006 --center-lat 40.7128 --zoom 15 --radius 2 --voxels-per-tile 128 --meters-per-voxel 1.25 --max-height-voxels 192 --seed 1337 --output-name worlds/world_1 --format oct
```

- For experimental runs with quick iteration, use the Python generator and then (optionally) convert to `.oct` with the converter tool.

Validation
----------
Use `check_world_gaps` to verify there are no missing columns between tiles:

```bash
cargo run --bin check_world_gaps worlds/world_1.oct worlds/world_1_meta.json
```

If `columns N/N` is reported for every plane, seam coverage is complete.
