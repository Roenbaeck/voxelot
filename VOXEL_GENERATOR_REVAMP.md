# Voxel Generator Revamp

Date: 2025-11-07

Goal: Replace the simplistic Overpass-based one-shot generator (`osm_voxel_generator.py`) with a modular, tile-oriented voxelization pipeline that can approximate the fidelity achievable by engines like streets.gl while staying voxel-friendly.

## Problems With Current Generator
- One large Overpass query → latency + fragile (timeouts, rate limits)
- No Web Mercator / tile addressing; ad-hoc equirectangular projection with constant scale
- Footprints approximated poorly (bounding boxes and min sizes) → distorted shapes
- Heights guessed crudely from `height` or `building:levels`; no fallback heuristics by building type
- Terrain: flat slab over building bounds (optional commented-out elevation code is heavy and unfocused)
- No separation of stages (fetch → parse → classify → voxelize → serialize)
- Output is a monolithic text file; no streaming / paging concept

## Target Architecture
Stage | Responsibility | Notes
----- | -------------- | -----
Tile Fetch | Acquire vector tiles (MVT / protobuf or GeoJSON fallback) for z/x/y range | Pluggable source (local cache, HTTP, mock)
Decode | Parse geometry layers: buildings, landuse, water, roads | Keep raw polygons + tags
Semantic Enrichment | Classify building type, infer height if missing, assign material archetypes | Use rule table
Voxelization | Convert polygons to column spans with materials; handle extrusion + optional roof forms | Works in tile-local coords
Compression | Store per-tile presence + material palette (≤256) + RLE columns | Buffer ready for ingest into world
Assembly | Emit global coordinate-shifted voxels or tile block artifacts for engine | Later integrated with streaming

## Coordinate System
- Use Web Mercator tile scheme.
- For zoom level Z, tile spans: `lon: 360 / 2^Z`, `lat (approx): 170.102258 / 2^Z` degrees
- Convert lat/lon → world meters using Web Mercator formulas, then → voxel coords via scale S (voxels per meter)
- Each tile uses a fixed voxel resolution (e.g., 256×H×256) where H covers expected vertical range (e.g., 512). Adaptive future: quadtree vertical slices.

## Height Inference Rules (Initial Heuristics)
Priority order:
1. Explicit `height` tag (parse meters)
2. `building:levels` * level_height (default 3.1m)
3. `building:levels:underground` reduces visible height
4. Building archetype defaults (house=8m, garage=3m, commercial=12m, industrial=14m, retail=10m, tower=50m)
5. Fallback: 9m (≈ 3 levels)
Voxel Height = ceil(meters / meters_per_voxel). Start with meters_per_voxel = 1.0 or 0.5 for better vertical detail.

## Polygon to Voxel Strategy
1. Project polygon vertices to tile-local voxel space.
2. Build edge table; rasterize to per-scanline spans (even-odd fill).
3. For each filled (x,z) cell, fill y = [0, height_voxels) with base material + top layer (roof).
4. Optional roof variants (flat vs pitched) determined by tags (future): pitched → add triangular profile for small residential footprints.

## Materials
Material Field | Meaning
-------------- | -------
AlbedoIndex | Index into per-tile 8-bit color table
Flags | Bitfield: emissive/window, roof, foliage, water, road
Roughness | Quantized 0–255
Metalness | Quantized 0–255 (likely 0 for most buildings)

## Outputs (Prototype Phase)
- JSON meta + binary blob pair per tile:
  - meta: tile_id, bbox, voxel_res, palette, stats
  - blob: header (magic, version), palette RGBA, column directory, RLE spans or Morton-coded bitmask + material index stream
- For integration with existing engine before streaming: a flat `.txt` fallback writer matching current `osm_voxels.txt` schema (x y z type).

## Dry-Code Constraints
No Python runtime installed presently; code is written to be syntactically plausible and modular but unexecuted. Placeholders for actual network and protobuf decode.

## Next Steps
1. Implement `voxel_generator_tiles.py` skeleton with: TileId, projection helpers, height inference, polygon rasterizer (scanline), and placeholder tile fetcher.
2. Add CLI arguments (area center, zoom radius, output directory, format).
3. Provide stub vector tile decode returning synthetic building footprints for tests.
4. Later: integrate real MVT decode (e.g., `mapbox_vector_tile` or `mvt` library) once environment allows installs.

## Open Questions
- Chosen base vertical resolution vs memory (H=512 may be high; consider dynamic max height per tile).
- Whether to pre-merge adjacent tiles into larger voxel blocks to reduce per-tile overhead.
- Add road extrusions or just surface classification initially?

---
This file will evolve as the prototype solidifies.
