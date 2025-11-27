Based on the analysis of `voxelot.rs`, here are the specific performance bottlenecks and recommended improvements for your Rust code.

### 1. Main Thread Stutter: Fallback Logic
**The Bottleneck:**
Inside `render()`, you iterate over visible chunks. If a chunk is close but not yet meshed, you iterate over **individual voxels** on the main thread to populate `cpu_prepopulated_instances`.
```rust
// In render()
if v.is_leaf_chunk && !has_mesh && dist_sq < fallback_dist_sq {
    if let Some(chunk) = self.world.get_leaf_chunk_at_origin(...) {
        // CRITICAL: This loop runs on the Main Thread!
        for ((x, y, z), voxel) in chunk.iter() {
             cpu_prepopulated_instances.push(...);
        }
    }
}
```
If you teleport or move quickly into a dense area, this loop can run for thousands of voxels across multiple chunks, causing a massive CPU spike and frame stutter.

**The Fix:**
*   **Worker Thread:** Offload the generation of "fallback instances" to your thread pool, just like you do for meshing.
*   **Time Slicing:** Limit the number of chunks you generate fallbacks for per frame (e.g., max 2 chunks per frame). It is better to have a hole for 1 frame than to drop FPS to 10.

### 2. Allocation Churn in Render Loop
**The Bottleneck:**
You create new `Vec`s every frame inside `render()` for GPU inputs.
```rust
// Inside render()
let mut cpu_prepopulated_instances: Vec<VoxelInstanceRaw> = Vec::new();
let mut gpu_inputs: Vec<GpuInstanceInput> = Vec::new();
```
As the world grows, `gpu_inputs` can contain thousands of items. Reallocating this heap memory every frame puts pressure on the allocator.

**The Fix:**
*   Promote these vectors to fields in your `App` struct.
*   Call `.clear()` at the start of the frame. This reuses the underlying capacity and prevents heap thrashing.

### 3. Excessive `queue.write_buffer` Calls (Mesh Uploads)
**The Bottleneck:**
When meshes are ready, you process them in a loop:
```rust
// Inside the loop processing ready_chunk_meshes
queue.write_buffer(mega_vertex_buffer, ...);
queue.write_buffer(mega_index_buffer, ...);
```
If 20 chunks finish meshing in one frame, you trigger 40 separate `write_buffer` calls. Each call involves internal `wgpu` tracking, synchronization, and potentially a staging buffer allocation.

**The Fix:**
*   **Staging Belt:** Allocate one large staging buffer (mapped `COPY_SRC`) at the start of the frame (or persistent).
*   Write all new vertices/indices into this staging buffer via CPU `memcpy`.
*   Issue a series of `encoder.copy_buffer_to_buffer` commands. This is significantly faster for the driver than many small `write_buffer` calls.

### 4. Shadow Pass Overhead (Non-Indirect Fallback)
**The Bottleneck:**
If `multi_draw_indexed_indirect_count` is not supported (or for shadows where you force CPU path), you iterate the entire mesh cache:
```rust
// Shadow Pass
for (_, entry) in self.mesh_cache.iter() {
    shadow_pass.draw_indexed(...);
}
```
If you have 5,000 chunks loaded, this is 5,000 individual draw calls submitted to the driver every frame. This will be CPU-bound.

**The Fix:**
*   **Unified Indirect Buffer:** Even for shadows, try to use `draw_indexed_indirect`. You already have `mesh_indirect_buffer`.
*   If you need to render *all* chunks (not just culling candidates) for shadows, maintain a separate "All Chunks" indirect buffer that you update only when chunks are added/removed, rather than rebuilding a command list every frame.

### 5. Render Pass Thrashing (Kawase Blur)
**The Bottleneck:**
Your Dual Kawase implementation starts a new Render Pass for every dispatch (Down and Up levels):
```rust
// Loop for levels
let mut pass = encoder.begin_render_pass(...);
pass.draw(...);
// End pass
```
Starting a render pass is expensive (pipeline barriers, load/store ops). Doing this 12+ times (6 down, 6 up) per frame adds significant overhead.

**The Fix:**
*   **Texture Views:** Check if you can use **Compute Shaders** for the blur. Compute shaders can write to different textures without the overhead of `begin_render_pass` (using `storage` textures).
*   *Alternatively:* If you stick to Fragment shaders, there is little you can do other than reducing the iteration count or accepting the overhead, as the render target changes every step.

### Summary of Priority Fixes

1.  **High:** Move `cpu_prepopulated_instances` generation off the main thread or time-slice it.
2.  **High:** Reuse `gpu_inputs` and `cpu_prepopulated_instances` vectors (don't `Vec::new()` in loop).
3.  **Medium:** Batch mesh uploads (reduce `queue.write_buffer` count).
4.  **Medium:** Ensure `multi_draw_indirect` is working; fallback to standard `draw_indirect` loop instead of `draw_indexed` loop if possible.
