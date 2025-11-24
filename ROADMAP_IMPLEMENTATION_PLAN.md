# Roadmap Implementation Plan

This document details the implementation strategy for the top two priorities in `ARCHITECTURE_AND_ROADMAP.md`.

## 1. GPU-Resident Chunk Management

**Objective:** Minimize CPU-GPU bandwidth usage by keeping the state of all loaded chunks on the GPU and only uploading "diffs" (additions/removals) each frame.

### Current State
Currently, we rebuild the `InstanceInput` buffer (candidates for culling) *every frame* on the CPU and upload it to the GPU. This involves iterating over all ~50k loaded chunks, checking their state, and writing to a buffer.

### Proposed Architecture

#### Data Structures

1.  **GPU Chunk Metadata Buffer (`StorageBuffer`)**
    - A persistent buffer storing metadata for *all* loaded chunks (not just visible ones).
    - Struct `GpuChunkMetadata`:
        ```rust
        struct GpuChunkMetadata {
            position: vec3<i32>, // World position
            scale: f32,
            flags: u32,          // HasMesh, HasEnvelope, IsLeaf, etc.
            mesh_index: u32,     // Index into MegaBuffer (if meshed)
            envelope_index: u32, // Index into EnvelopeBuffer (if has envelope)
            // ... other properties
        }
        ```
    - Capacity: Max loaded chunks (e.g., 100k).

2.  **Free List / Slot Manager (CPU)**
    - A simple allocator (e.g., `Vec<u32>` of free indices) to manage slots in the GPU Metadata Buffer.
    - Map `ChunkKey -> SlotIndex`.

#### Workflow

1.  **Chunk Load/Mesh Update:**
    - When a chunk is loaded or meshed, allocate a slot (if new).
    - Write *only* that chunk's metadata to the GPU buffer via `queue.write_buffer` (at the specific offset).

2.  **Chunk Unload:**
    - Mark the slot as free in the CPU allocator.
    - (Optional) Zero out the GPU metadata or mark as "invalid" to prevent rendering.

3.  **Frame Update (Culling Pass):**
    - **CPU:** Does *not* upload the full candidate list.
    - **CPU:** Uploads a small "Frame Global" uniform (Camera pos, frustum, etc.).
    - **GPU (Compute):**
        - Dispatch a compute shader with `GlobalInvocationID` mapping to `SlotIndex`.
        - Each thread reads `GpuChunkMetadata[index]`.
        - If invalid/empty, return.
        - Perform Frustum Culling & LOD selection.
        - Write to `IndirectDrawBuffer` if visible.

### Implementation Steps

1.  **Refactor Chunk Management:**
    - Create `ChunkSlotAllocator` struct.
    - Add `slot_index` to `Chunk` or a separate `ChunkSlotMap`.
2.  **Create GPU Buffers:**
    - `chunk_metadata_buffer`: Large storage buffer.
3.  **Update Compute Shader (`gpu_cull.wgsl`):**
    - Change input from `array<InstanceInput>` (per-frame) to `array<GpuChunkMetadata>` (persistent).
    - Logic remains similar, but iterates over *all slots* instead of *candidates*.
4.  **Modify `voxelot.rs`:**
    - Remove `prepare_gpu_cull_inputs` loop.
    - Implement "dirty" tracking for chunks to trigger metadata updates.

---

## 2. GPU Occlusion Culling

**Objective:** Cull chunks that are occluded by other chunks (e.g., caves, behind mountains) to reduce the number of drawn triangles and fragment shader overdraw.

### Challenge
Traditional occlusion queries are slow (CPU readback). We need a GPU-driven approach.

### Proposed Approach: Hierarchical Z-Buffer (HZB)

#### Concept
1.  **Depth Pyramid:** Generate a mip-mapped chain of the depth buffer from the *previous frame*.
    - Level 0: Full resolution depth.
    - Level 1: Half resolution (min-depth of 2x2 block).
    - ...
    - Level N: 1x1 pixel.
2.  **Culling Shader:**
    - For each chunk, calculate its Screen-Space Bounding Box (AABB).
    - Select the appropriate mip-level where the AABB covers ~4 pixels.
    - Sample the HZB at that level.
    - If `ChunkMinDepth > HZB_Depth`, the chunk is occluded -> Skip.

#### Implementation Steps

1.  **Depth Reprojection (Optional but recommended):**
    - Since we use the *previous* frame's depth, there will be gaps when the camera moves.
    - Reprojecting helps, or just accept 1-frame latency (objects might pop in if moving fast).
    - *Simpler start:* Just use previous frame depth. If camera moves fast, disable occlusion culling for that frame.

2.  **Downsample Pass:**
    - Create a compute shader `generate_hzb.wgsl`.
    - Takes `depth_texture` (read-only) -> writes to `hzb_texture` (mips).
    - Dispatch for each mip level (or single pass with atomics/shared memory).

3.  **Integrate into Culling Shader (`gpu_cull.wgsl`):**
    - Add `hzb_texture` binding.
    - After Frustum Culling, perform Occlusion Check.
    - `if (is_occluded(aabb, hzb)) return;`

4.  **Two-Pass Culling (Advanced):**
    - To avoid "holes" from using previous frame depth:
    - **Pass 1:** Render occluders (large chunks) or use previous depth.
    - **Pass 2:** Render everything else.
    - *Recommendation:* Start with 1-frame latency. It's usually fine for terrain.

### Alternative: "Rough" Grid Culling (Simpler)
- Divide the world into large macro-cells (e.g., 32x32 chunks).
- If a macro-cell is fully occluded (e.g., by a wall of chunks), cull all chunks inside.
- Harder to implement dynamically. HZB is more standard.
