# Roadmap & Architecture

## Prioritized TODOs

### 1. GPU-Resident Chunk Management (High Priority)
**Goal:** Reduce CPU-GPU data transfer by keeping chunk instances on the GPU and only updating their mesh data.
- **Concept:** Treat chunks as persistent "mesh groups" on the GPU.
- **Benefit:** Moving the camera would only require updating a few chunks (added/removed) rather than re-uploading many.
- **Implementation:**
    - Maintain a persistent buffer of active chunk metadata on the GPU.
    - Use a compute shader to manage the "slot" assignment for chunks.
    - Only upload "diffs" (new chunks, removed chunks) to the GPU.
    - See `ROADMAP_IMPLEMENTATION_PLAN.md` for detailed design.

### 2. GPU Occlusion Culling
**Goal:** Improve performance by culling occluded geometry.
- **Idea:** Implement a GPU occlusion culling shader that runs *before* the main render pass.
- **Scope:** Only run this up to `envelope_dist_sq` to cull individual voxels and detailed meshes.
- **Challenge:** Previous attempts hurt performance. Need to find a "rough" culling approach that is cheap enough to be a net win. Maybe hierarchical Z-buffer (HZB) or a simple coverage buffer?
- **Implementation:** See `ROADMAP_IMPLEMENTATION_PLAN.md` for detailed design.

### 3. General Improvements
- **Deterministic Skipping:** Add a per-chunk offset approach so the compute shader can skip prepopulated buffers deterministically.

---

# Architecture Reference

## The "Holy Grail" Architecture (2025 Best Practices)

For a high-performance voxel engine in wgpu (Rust), we aim for a **GPU-driven, indirect rendering** pipeline.

### Core Components

1.  **Mega-Buffers (Implemented)**
    - **Vertex & Index Buffers:** One giant GPU buffer (e.g., 512MB+) for all geometry.
    - **Allocator:** A slab allocator (or free-list) manages regions within this buffer.
    - **Zero-Copy (Ideal):** In wgpu, we use a Staging Buffer -> `copy_buffer_to_buffer` to upload mesh data, avoiding per-mesh buffer creation overhead.

2.  **GPU Culling & Indirect Rendering (Implemented)**
    - **CPU:** Sends a single "Compute Culling" command.
    - **GPU (Compute):**
        - Checks all loaded chunks against the camera frustum.
        - Writes valid draw commands into an `IndirectBuffer`.
    - **CPU:** Sends a single `draw_indexed_indirect` command.
    - **Result:** The CPU handles 1 million chunks as easily as 10.

3.  **Bindless-Style Textures (Implemented via Array)**
    - **Texture Array:** All block textures are loaded into layers of a single `Texture2DArray`.
    - **Shader:** Samples `texture(sampler, uv, layer_index)`.
    - **Benefit:** No bind group switching per chunk.

### Memory Management Strategy

| Component | Technique | Why It's Fast |
| :--- | :--- | :--- |
| **Vertex Data** | Mega-Buffer (Slab Allocation) | No buffer creation overhead |
| **Visibility** | GPU Compute Culling | Zero CPU culling cost |
| **Draw Calls** | Multi-Draw Indirect | 1 draw call for the whole world |
| **Textures** | Texture Array | No bind group swaps |
