# Tile & Post-Processing Integration Plan

Date: 2025-11-07

This document outlines how to evolve `voxelot` by borrowing architectural ideas (not code) from projects like `streets-gl` that stream and shade large-scale city data with a modern post-processing stack (TAA, SSAO, depth of field, SSR, bloom).

## 1. Current State Summary

Component | Current Approach | Gaps
--------- | ---------------- | ----
OSM ingestion | Python script `osm_voxel_generator.py` pulls a fixed Overpass bounding box and voxelizes naive footprints + flat terrain. | No streaming, no vector tile protocol, blocking API calls, unbounded memory growth if scaled.
World representation | Hierarchical sparse chunk tree (16^n) with roaring bitmaps and LOD recursion by distance. | Great base; lacks paging, streaming hooks, prioritization queues, edit journaling.
Culling | CPU frustum culling + distance-based LOD; optional parallel top-level traversal. | No hierarchical occlusion, no screen-space error, no temporal coherence beyond simple cache.
Rendering | Single pass voxel instancing with per-instance position/type; flat Lambert-ish lighting in `voxel.wgsl`. | No G‑Buffer, no deferred/compute stages, no normal/albedo buffers, no temporal history, no depth pyramid.
Post FX | None. | Missing TAA, SSAO, SSR, Bloom, DoF chain.
Asset/Material | Hard-coded voxel colors by type. | Needs material atlas + physical-ish params (albedo, roughness, emissive).

## 2. Guiding Principles
- Keep hierarchical sparse voxel core; layer streaming + material + framegraph around it.
- Separate data acquisition (OSM vector tiles) from voxelization & residency management.
- Introduce a FrameGraph / RenderGraph to schedule passes and manage intermediate textures.
- Use GPU-friendly compressed per-tile buffers (quantized coords, Morton order) feeding a compute expansion or meshlet path.
- Favor incremental adoption: G-Buffer first; add TAA; then enable SSAO/SSR/Bloom/DoF.

## 3. Tile / Streaming Architecture
### 3.1 Sources
1. OSM Vector Tiles (e.g. openvectortile / tile server). Fallback: On-the-fly Overpass (slow) for small test areas.
2. Elevation tiles (SRTM or ASTER) for terrain layering.

### 3.2 Coordinate System & Indexing
- Adopt Web Mercator (EPSG:3857) tile scheme (z/x/y) internally; map to world integer voxel coordinates via scale S(z) = base_resolution * 2^(Z0 - z).
- Maintain `TileId { zoom: u8, x: u32, y: u32 }` + world-space AABB.

### 3.3 Residency Manager
State machine per tile: Unrequested → QueuedDownload → Downloading → DecodeQueued → Decoding → Voxelizing → GPUUpload → Resident → Evicting.

Priority Score = w_dist * camera_distance^-1 + w_screen * projected_screen_area + w_motion * camera_velocity_alignment + w_feedback * last_frame_importance.

Use a binary heap or radix priority queue; budget N downloads + M decodes + K voxelizations per frame.

### 3.4 Voxelization Strategy
- Building footprints extruded to height (levels or tag heuristics). Sub-voxel precision retained via per-voxel material flags.
- Terrain: generate heightfield per elevation tile, then fill columns (optionally compress using run-length inside a column before expansion on GPU).
- Store per tile:
  - Compressed presence bitmap (retain roaring? or use 64x64xH layering).
  - Material palette up to 256 entries; per-voxel 8-bit index.

### 3.5 Integration with Hierarchical World
Option A: Maintain world root as sparse occupancy; each resident tile writes into appropriate coordinates. Option B: Original hierarchical chunk becomes a *virtual page table* referencing tile blocks (faster eviction). Choose B for scaling: each leaf points to a TileVoxelBlock, reference-counted.

Eviction: detach leaf references; decrement block refcount; free GPU buffers when zero.

## 4. Rendering Pipeline Roadmap
Order of adoption (each stage validated before next):
1. Depth + Albedo + Normal + Material G-Buffer (replace direct color in vertex/fragment).
2. TAA (history buffer, velocity buffer, jittered projection, clamp heuristics).
3. SSAO (HBAO-like or GTAO-lite) using depth + normal pyramids.
4. Screen-Space Reflections (hierarchical ray march on depth pyramid) + fallback roughness cone trace (approx) or reflection probe.
5. Bloom (dual-filter mip chain + energy-conserving combine).
6. Depth of Field (physical circle of confusion, gather pass, optional bokeh shape). Order may shift Bloom after DoF depending on artistic preference.
7. Lighting improvements: directional sun + ambient probe + voxel-based emissive.

### 4.1 Required Buffers
Name | Format | Notes
---- | ------ | -----
Depth | D32 | Also build min/max mip pyramid.
Normals | RG16Snorm | View-space or world-space.
Albedo | SRGB8 | Base color.
Material | RGBA8 | Roughness (R), Metalness (G), Emissive (B), VoxelFlags (A).
Velocity | RG16F | Screen-space motion for TAA.
HistoryColor | SRGB8 | Previous frame resolved color.
Resolved | HDR (RGBA16F) | Lighting accumulation before tone map.
LuminanceAvg | R16F | Temporal exposure adaptation.

### 4.2 FrameGraph Passes (Initial)
1. GeometryPass: writes Depth + G-Buffer.
2. LightingPass: reads G-Buffer; outputs HDR color.
3. TAA: takes current HDR + history + velocity; outputs smoothed HDR + updates history.
4. SSAO: optional; modulates indirect diffuse.
5. SSR: composites reflections into specular channel.
6. BloomExtract + BloomBlur (mip chain) + BloomComposite.
7. DoF (gather or scatter) operating on TAA-resolved color.
8. ToneMap + Gamma + UI overlay.

### 4.3 Temporal Anti-Aliasing Details
- Halton or 2x2 rotated jitter sequence applied to projection matrix (offset / render_target_size).
- Velocity buffer: store previous vs current clip positions per instance; for static voxels supply zero velocity.
- Neighborhood clamping: clip history color toward current min/max luminance neighborhood.
- Reactive mask: mark high-emissive or disocclusion areas to reduce blending.

### 4.4 SSAO (Simplified GTAO Lite)
- Generate normal+depth pyramid.
- For each pixel sample limited horizon directions (e.g. 6) with adaptive step length.
- Ambient term = integrate unoccluded fraction; temporal filter & bilateral blur.

### 4.5 SSR
- Ray direction from view vector reflected about normal.
- March along depth pyramid with binary or mip stepping (max ~64 steps). Early terminate on intersection.
- Roughness-based cone widening: relaxed hit acceptance threshold.
- Fallback: environment probe / sky if miss.

### 4.6 Bloom
- Threshold in HDR space (e.g. luminance > 1.0) with soft knee.
- Downsample chain with prefilter; upsample additive with energy scaling.

### 4.7 DoF
- Compute circle of confusion (CoC) from focal distance & aperture.
- Separate near/far blur; use tile-based gather to limit over-blur on foreground edges.
- Optionally integrate with TAA reprojection.

## 5. GPU Data Layout for Voxels
Option to replace per-vertex duplication with:
- Instance buffer: position (i32x3), material index (u16), scale (u16), LOD flags.
- Indirect draw / multi-draw dispatch built from culling compute pass.
- Future: Mesh shader (when wgpu supports) or cluster culling compute producing compact draw list.

Visibility Compute Pass:
1. Frustum test + LOD selection per top-level cell.
2. Append visible voxel instances into a RW storage buffer (atomic counter) sorted (optional) by material.

## 6. Enhanced Culling & LOD
- Maintain depth-aware screen-space error metric: error = voxel_screen_diameter; subdivide if > pixel_threshold.
- Introduce Hi-Z occlusion: build depth mip pyramid after geometry depth prepass; test voxel AABBs before adding to instance list.
- Temporal coherence: carry forward last frame visible set and test movement deltas first.

## 7. Materials & Lighting
- Introduce `materials.rs` with PBR-ish parameters (albedo, roughness, metalness, emissive, flags).
- Simple directional sun + ambient spherical harmonics (3 bands) or single probe.
- Voxel emission added (e.g., windows at night) feeding Bloom.

## 8. Phased Implementation Plan
Phase | Goal | Key Deliverables | Risk Level
----- | ---- | ---------------- | ----------
P1 | G-Buffer + FrameGraph Skeleton | New render module, geometry pass rewrite, basic lighting | Low
P2 | TAA + Velocity | Jitter, velocity calc, history mgmt | Medium
P3 | Streaming Tile Skeleton | TileId, request queue, async downloader stub, integration points | Medium
P4 | SSAO + Depth Pyramid | Compute passes, bilateral blur | Medium
P5 | SSR + Reflection Fallback | Depth pyramid march, roughness integration | High
P6 | Bloom | Threshold + mip chain | Low
P7 | DoF | CoC calc + blur + composite | Medium
P8 | Full OSM Vector Tile Ingestion | Vector tile decode (protobuf), voxelization, eviction | High
P9 | Hi-Z Occlusion + Screen Error LOD | Depth pyramid reuse, error heuristic | Medium

## 9. Minimal Initial Refactors
- Abstract current draw path: separate data extraction (culling) from submission; allow swap to compute-generated indirect buffer later.
- Introduce `render/graph.rs` with pass registration & resource handles.
- Prepare `render/resources.rs` for texture & buffer descriptors (G-Buffer creation).

## 10. Licensing & Attribution
Both `voxelot` and `streets-gl` are MIT (assumed for external; verify). We must not copy large code blocks; re-implement algorithms using public domain formulations and credited concepts. Cite inspiration in README enhancements.

## 11. Metrics & Success Criteria
Metric | Target After Stage
------ | ------------------
Visual Stability (TAA shimmer) | <5% noticeable shimmer on slow pan.
Frame Time Overhead of TAA | <1.5 ms @1080p.
SSAO Cost | <2.0 ms @1080p (configurable radius).
Streaming Latency (tile request → render) | <1 s on warm network; <150 ms once cached.
Memory per Resident Tile | <256 KB (voxel + materials) for typical urban block.

## 12. Next Immediate Actions
1. Implement G-Buffer pass (replace color-only shader) while keeping existing voxel instancing.
2. Add camera jitter + velocity channel placeholders to start TAA groundwork.
3. Draft Rust modules skeleton: `src/render/{graph.rs, passes/{geometry.rs, lighting.rs, taa.rs}}`.
4. Prototype tile struct & priority queue without networking.

## 13. Open Questions
- Do we need per-voxel ambient occlusion baked or rely purely on SSAO + global lighting?
- Should we compress voxel presence with morton-coded bitplane vs roaring for GPU locality?
- How to batch decode vector tiles (Rust protobuf crate) without stalling main thread? (Tokio + channel?)

## 14. Glossary
- TAA: Temporal Anti-Aliasing.
- SSAO: Screen Space Ambient Occlusion.
- SSR: Screen Space Reflections.
- DoF: Depth of Field.
- Hi-Z: Hierarchical Z (depth pyramid) for occlusion & ray marching acceleration.

---
Prepared as a living document; iterate as implementation uncovers constraints.
