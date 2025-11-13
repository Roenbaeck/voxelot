# Voxelot Architecture & Roadmap 

Date: 2025-11-13

---
## 1. High-Level Overview

Voxelot is a hierarchical sparse voxel engine (uniform 16³ chunks at all levels) targeting large, streamable worlds with modern rendering (G-Buffer, TAA, SSAO, SSR, Bloom, DoF) and data-driven tile ingestion based on OpenStreetMap (OSM) vector tiles + elevation sources.

Core pillars:
1. Hierarchical Sparse Representation ("chunks all the way")
2. Adaptive LOD via subdivision / merge
3. Streaming Tile Residency & Voxelization
4. Modern Framegraph-Based Rendering & Post FX
5. Material & Semantic Layers (future multi-system integration)

---
## 2. Current Engine Capabilities

Implemented:
- Frustum culling (6-plane) & parallel top-level traversal using Rayon
- Hierarchical chunk structure with Roaring bitmaps and marginal axis masks (px/py/pz)
- Distance-based LOD recursion; manual subdivision / merge API
- Greedy meshing for leaf 16³ chunks with quad merging & emissive capture (`meshing.rs`) and multi-threaded mesh workers in `viewer_hierarchical.rs`
- Bloom pass (threshold + mip blur + composite)
- Tilt-shift Depth of Field (DoF) gather + smoothing (WGSL)
- Dual directional lighting: sun (shadowed) + moon (no shadows yet) with continuous time-of-day color & intensity transitions
- Fog controls (exponential distance falloff)
- Basic voxel instancing & meshed chunk rendering (palette-driven materials)

Not Yet Implemented (Planned – still absent in code at 2025-11-13):
- Framegraph abstraction (passes/resources orchestrator) – current viewer uses ad-hoc ordered passes.
- G-Buffer (separate albedo/normal/material/velocity render targets) – rendering is forward with per-vertex color.
- Temporal Anti-Aliasing (TAA) – no history buffer, jitter, or velocity buffer for shading (only camera movement velocity logic present).
- Screen Space Ambient Occlusion (SSAO) – no depth/normal pyramid sampling shaders.
- Screen Space Reflections (SSR) – no ray-march pass or reflection integration.
- Hi-Z occlusion (depth mip pyramid) – depth mips not generated; occlusion relies on CPU logic only.
- Compute-generated indirect draw list for meshed chunks – GPU culling exists but chunk draw submission still manual.
- Material system with roughness/metalness/emissive parameters in shading (palette has emissive but shader is not PBR).

In progress / partially prototyped:
- Design for expanded post-processing (TAA, SSAO, SSR, tone mapping)

---
## 3. Voxel Generation

Add a voxel editing mode. 
The crate `iced` will be used for GUI components.

---
## 4. Rendering & Framegraph Plan

### 4.1 Buffer & Resource Set
Name | Format | Purpose
---- | ------ | -------
Depth | D32 | Geometry depth + pyramid (Hi-Z)
Normals | RG16Snorm | View/world-space normals
Albedo | SRGB8 | Base color
Material | RGBA8 | Roughness, Metalness, Emissive, Flags
Velocity | RG16F | Clip-space motion (TAA)
HistoryColor | SRGB8 | Previous frame resolved color
HDRResolved | RGBA16F | Lighting accumulation
LuminanceAvg | R16F | Auto-exposure temporal adaptation

### 4.2 Pass Ordering (Framegraph)
1. Geometry (populate G-Buffer + depth)
2. Lighting (direct + ambient + emissive → HDRResolved)
3. TAA (jittered projection, velocity, neighborhood clamp)
4. SSAO (optional; depth+normal pyramid; bilateral blur)
5. SSR (ray march depth pyramid; roughness-aware cone)
6. BloomExtract + BloomMipBlur + BloomComposite (existing code rehosted)
7. DoF (tilt-shift gather + smoothing; integrate with jitter later)
8. ToneMap + Gamma + UI

### 4.3 Temporal AA Essentials
- Jitter: Halton(2,3) or 2×2 rotated pattern
- Velocity: previous vs current clip-space or world position per instance (static = 0)
- History clamp: min/max neighborhood luminance & reactive mask for disocclusion/emissive

### 4.4 Hi-Z & Advanced Culling
- Build depth mip pyramid after Geometry pass
- Use for: SSAO horizon steps, SSR step reduction, occlusion tests pre-instance emission
- Future: screen-space error metric (voxel projected diameter vs pixel threshold) → triggers subdivision

### 4.5 Materials & Lighting Extension
- Introduce `materials.rs` with PBR-ish params & flags
- Directional sun + ambient probe / SH (3-band) + per-voxel emissive (windows at night → Bloom)

---
## 5. Performance & Optimization Strategy

### 5.1 Geometry Reduction
- Replace per-cube 36 vertices with greedy meshing (merge quads, remove internal faces)
- Consider instanced box with face culling replaced by mesh generation

### 5.2 Culling Improvements
- Hierarchical occlusion using Hi-Z
- Temporal reuse of visible sets → only validate movement boundaries
- Compute pass for instance list + indirect draw buffer (atomic append)

### 5.3 Chunk Metadata Cache
Add per-chunk:
- `voxel_count`
- `material_span (min,max)`
- `flags: has_transparent, has_emissive, fully_solid`
- Local occupied AABB within 16³
→ Accelerated traversal (skip uniform solids, treat as macro voxel for GI / AO)

---
## 6. Adaptive LOD & Editing Roadmap

Trigger subdivision when:
- Camera distance < threshold OR projected size > N pixels
- High material variance in macro voxel
- Recent player edits in region

Trigger merge when:
- Uniform material & no sub-voxels with edits
- Projected size < collapse pixel threshold

Future: copy-on-write procedural base + sparse edit overlays stored as Roaring deltas for history / branching timelines.

---
## 7. Extended Use Cases (From Suggestions)

Category | Idea
-------- | ----
Global Illumination | Treat macro voxels as radiance volumes; selective descent for detail
Physics Broadphase | Hierarchy as BVH; coarse collision per chunk, refine local contacts
Pathfinding | Coarse path over macro voxels → refine near agent using sub-chunks
Semantic Layers | Material + gameplay flags unify nav, AI, occlusion, audio reverb zones
Procedural Agents | Multi-scale generators (macro terrain, meso buildings, micro details)
Temporal Layers | Chunk deltas enabling time travel / replay / undo

---
## 8. Phased Implementation Checklist

Phase | Focus | Result
----- | ----- | ------
P1 | Greedy Meshing (Completed) | Quad merged leaf chunk meshes, emissive extraction
P2 | G-Buffer + Framegraph skeleton | Baseline for later passes
P3 | TAA + Velocity + Jitter | Temporal stability
P4 | Tile Residency Skeleton | Streaming-ready world ingestion
P5 | SSAO + Depth Pyramid | Ambient contact & occlusion basis
P6 | SSR + Reflection Fallback | Specular detail
P7 | Bloom & DoF Integration (existing) | Unified post chain in framegraph
P8 | Vector Tile Decode + Elevation | Real geographic fidelity
P9 | Hi-Z Occlusion + Screen Error LOD | Efficient scalability
P10 | Material System & Emissive | PBR-ish shading
P11 | Compute Culling + GPU Drawlist | Reduced CPU involvement

---
## 9. Metrics & Success Targets

Rendering Metric | Target
---------------- | ------
TAA cost @1080p | <1.5 ms
SSAO pass cost | <2.0 ms (tunable radius)
SSR max ray steps | ≤64 w/ early exit
Bloom chain | <1.0 ms @1080p
DoF gather+smoothing | ≤2.0 ms @1080p

Streaming Metric | Target
---------------- | ------
Tile decode latency (local cache) | <30 ms
Tile voxelization latency | <20 ms
Total cold tile (net+decode+voxel) | <250 ms

Memory Metric | Target
------------- | ------
Per resident tile | <256 KB
Hierarchy overhead per macro voxel | <64 bytes metadata

---
## 10. Open Questions
- Morton vs roaring for GPU presence: hybrid threshold?
- Depth pyramid resolution: full vs half scale trade-offs?
- Single global material atlas vs per-tile palettes (merging strategy)?
- Jitter sequence length (8 vs 16) for balance of stability vs pattern repetition?
- Procedural elevation vs on-demand API? (offline bake?)

---
## 11. Immediate Next Steps (Actionable)
1. Implement greedy meshing (P6 early) for instant perf gain before full framegraph.
2. Skeleton `render/graph.rs` + `passes/geometry.rs` (empty G-Buffer path)
3. Refactor chunk storage toward fixed array + presence bitmap (measure write performance)
4. Begin tile residency struct definitions in Rust mirroring Python prototype.

---
Consolidated by roadmap automation – evolve here going forward.
