# Implementation Plan for WGPU 28

First switch to WGPU 28 in Cargo.toml and then fix all breaking changes. 

# Potential breaking changes affecting us

To get `voxelot` compiling on v28, you will need to address these changes:

*   **`enumerate_adapters` is now `async`:**
    If your initialization code enumerates adapters to pick the best high-performance GPU, you now need to `await` this call. This standardizes behavior across Web and Native.
    ```rust
    // Old
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    // New
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    ```

*   **Push Constants are now "Immediates":**
    If you use push constants for passing small data (like chunk indices or base offsets) to your shaders, the API has changed to match the spec.
    *   **Rust:** `render_pass.set_push_constants(...)` $\to$ `render_pass.set_immediates(...)`.
    *   **WGSL:** `var<push_constant>` $\to$ `var<immediate>`.
    *   **Note:** You must now specify a unified `immediate_size` in your `PipelineLayoutDescriptor`.

*   **Subgroup Limits Location:**
    Since you use **GPU compute culling**, you might be checking subgroup sizes to optimize your workgroup reductions. These limits have moved from `Limits` to `AdapterInfo`.

## Breaking Changes

1. **MipmapFilterMode split** - Change `mipmap_filter: FilterMode::X` to `mipmap_filter: MipmapFilterMode::X` in sampler descriptors
2. **Error scopes** - Now return guards, need to await `.pop()` on the guard
3. **Immediates API** (mentioned above)
4. **RenderPassDescriptor** - Add `multiview_mask: None` if you're not using multiview
5. **Subgroup size** - Move from `limits` to `info` if you use it

### Breaking Changes to Consider Before Upgrading
Upgrading from 0.27 to 0.28 will require code changes:
- Push constants → immediates (rename fields, update shader address space to `<immediate>`, change set calls).
- If you use multiview already (unlikely), update `@builtin(view_index)` to `u32`.
- `enumerate_adapters` is now async (minor if you don't use it directly).
- SamplerDescriptor: `mipmap_filter` now uses separate `MipmapFilterMode`.
- Subgroup sizes queried from `AdapterInfo` instead of `Limits`.


# Possible Optimizations

When breaking changes are fixed, we can focus on optimizations. 

#### Mesh shaders
This is the most significant feature in v28 for a voxel engine. Since `voxelot` currently relies on **CPU-side Binary Greedy Meshing** and uploads to a "Mega VB/IB," you are likely bottlenecked by the PCI-E bus during rapid movement or massive world changes.

*   **How it helps you:** instead of generating quads on the CPU and uploading vertices, you could upload the **Roaring Bitmaps (compressed chunk data)** directly to a storage buffer. A **Task Shader** can determine which chunks/sub-chunks are visible, and **Mesh Shaders** can generate the actual triangle geometry on-the-fly from the bitmap data.
*   **Benefit:** This moves the "greedy meshing" logic entirely to the GPU, effectively eliminating the bandwidth cost of vertex uploads. Given your data structure is already bitwise-heavy, it maps extremely well to GPU compute operations.
*   **Status:** In v28, this is available via the `MESH_SHADER` feature (native only currently, excellent on Vulkan/DX12/Metal).

#### Multiview & Texture Arrays
*   **Cascaded Shadow Maps:** The new **Multiview** support (and the `multiview_mask` in `RenderPassDescriptor`) allows you to render to multiple layers of a texture array in a single pass.
*   **Use Case:** If `voxelot` renders Cascaded Shadow Maps (CSM) for its sun light, you can currently likely render all cascades in one draw call (using geometry shaders or instancing), but Multiview offers a cleaner, more hardware-accelerated path for this, reducing draw call overhead for shadows.

#### Subgroup Operations (Shader)
*   **Culling Efficiency:** Your "Compute Culling" pass likely compacts a list of visible chunks into an indirect buffer. `wgpu` v28 improves support and standardization for subgroup operations (`subgroupAdd`, `subgroupBallot`).
*   **Benefit:** ensuring your culling shader uses these intrinsics (checked via the new `AdapterInfo` capabilities) can significantly speed up the generation of your `draw_indirect` buffers by keeping the reduction logic in registers rather than shared memory.

#### Bindless / Texture Arrays
*   The release includes fixes for **Texture Binding Arrays** (especially on Metal).
*   **Use Case:** If your palette system currently uses a texture atlas to avoid binding limits, you might consider switching to a `texture_2d_array` with dynamic indexing. v28 makes this much more stable across backends, preventing the "bleeding" artifacts common with atlases at mipmap levels.

#### LoadOp::DontCare (new variant in render pass color/depth attachments)  
*  This allows leaving render target contents undefined at the start of a pass (instead of mandatory Load or Clear), reducing unnecessary memory loads when you overwrite every pixel.  
   Your engine has multiple offscreen render passes (e.g., for SSAO, SSR, bloom, depth/prepass for HZB). If these passes write to every pixel (common in full-screen post-processing or forward voxel rendering without preserved prior content), this can improve performance by avoiding bandwidth waste. It requires an unsafe token and falls back gracefully on unsupported backends.

#### **Immediates (renamed and simplified push constants)**  
*  Simpler API: single `immediate_size` in pipeline layout, `set_immediates` on encoders (no per-stage). Shader uses `var<immediate>`.  
   If your pipelines use push constants (common for per-draw uniforms like chunk transforms or camera data in voxel engines), the new API is cleaner and more aligned with WebGPU. Migration required if you already use them.

#### **Transient Textures** (Vulkan/Metal support added)  
*  Textures that live only for a frame/submission, allocated from transient memory heaps.  
   Potentially useful for temporary offscreen targets in your post-processing chain (ping-pong textures for bloom/DoF/SSR), reducing memory fragmentation or allocation overhead in long-running scenes.

#### **Other Minor Wins**  
*  Barycentric coordinates builtin in fragment shaders → useful if you add wireframe overlays or advanced shading.  
*  Improved validation and error scopes (now thread-local guards) → better debugging in multithreaded setup (you use rayon for CPU meshing/culling).  
*  Subgroup info moved to AdapterInfo → minor, but useful for tuning compute dispatches if you query subgroup sizes.
