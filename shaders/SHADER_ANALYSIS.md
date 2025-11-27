Your shaders are generally well-structured and readable, but there are several "low-hanging fruit" optimizations—particularly regarding **texture sampling bandwidth**, **ray-marching arithmetic**, and **compute shader memory access**—that could significantly improve performance.

Here is a breakdown of specific improvements.

### 1. Optimize Gaussian Blurs (`bloom_blur.wgsl`, `ssao_blur.wgsl`)
**Issue:** You are performing a discrete 9-tap convolution (center + 4 pairs) by manually sampling every single texel center.
**Improvement:** Use **Linear Sampling Hardware**.
By sampling *between* two texels using a `linear` sampler, the GPU performs a weighted average for free. This allows you to approximate a 9-tap Gaussian blur with just **5 texture fetches** (or a 5-tap with 3 fetches).

*   **Action:**
    1.  Calculate "bilinear offsets" and "bilinear weights" on the CPU.
    2.  Update the shader to sample at `uv + offset` instead of `uv + offset * texel_size`.
    3.  This effectively halves the texture bandwidth for your blur passes.

### 2. Screen-Space Ray Marching (`ssr.wgsl`, `water.wgsl`)
**Issue:** **Heavy Arithmetic inside Loops.**
Both shaders perform ray marching in **World Space**, which forces you to run `world_to_screen` (a Matrix Multiply and Perspective Division) inside the loop for *every step*.
*   `current_pos += ray_dir * step;`
*   `world_to_screen(current_pos);` // <--- Matrix Mul + Div

**Improvement:** **March in Screen Space.**
Convert the ray's start point and direction into Screen Space (pixel coordinates and depth) *before* the loop.
*   **Action:**
    1.  Calculate `start_screen_pos` and `end_screen_pos`.
    2.  Interpolate linearly in 2D (UV space) and hyperbolic/linear depth (depending on depth buffer).
    3.  The loop becomes a simple `current_uv += delta_uv;` and `current_depth += delta_z;`.
    4.  This removes the matrix multiplication from the inner loop entirely.

**Bonus for `water.wgsl`:** It currently uses a linear search (32 steps). Since you already have an HZB texture (generated for SSR/Culling), you should use **HZB tracing** for water reflections too, exactly like you do in `ssr.wgsl`.

### 3. Skybox Optimization (`skybox.wgsl`)
**Issue 1: Expensive Math.**
You are calculating `atan2` and `asin` for every pixel to sample the equirectangular texture. Inverse trigonometry is computationally expensive.
**Issue 2: Per-Pixel Matrix Math.**
You are calculating the view ray direction using `inverse_view` and `inverse_proj` in the **Fragment Shader**.

**Improvement:**
1.  **Move Ray Calculation to Vertex Shader:** Calculate the view-space ray direction for the 4 screen corners in the Vertex Shader and let the rasterizer interpolate it. In the Fragment shader, just `normalize(interpolated_ray)`. This removes 2 matrix multiplications per pixel.
2.  **Use a Cubemap:** If possible, convert your equirectangular skybox to a **Cube Map**. Sampling a `texture_cube` via a vector is hardware-accelerated and requires no `atan2` math.

### 4. HZB Generation (`hzb_gen.wgsl`)
**Issue:** **Bandwidth Overhead.**
You are dispatching a separate compute pass for every mip level. This requires reading and writing the entire texture to global VRAM multiple times (`copy_depth` -> `downsample` -> `downsample`...).

**Improvement:** **Single Pass Downsampler (SPD).**
Use a single compute shader dispatch that uses **workgroup shared memory (LDS)** to generate all mip levels at once.
1.  Load a tile (e.g., 64x64) of the depth buffer.
2.  Store to shared memory.
3.  Synchronize threads, downsample in shared memory, and write out to the respective mip levels in one go.
*   *Note: This is more complex to implement but drastically reduces memory bandwidth.*

### 5. GPU Culling (`gpu_cull.wgsl`)
**Issue:** **Scattered Writes.**
The shader writes to `mesh_indirect[instance.mesh_index]`. Since `mesh_index` is likely random/unsorted for the linear list of candidates, this causes **non-coalesced memory writes**, which is bad for the GPU memory controller.

**Improvement:**
1.  **Group by Mesh:** If possible, sort your `candidates` buffer on the CPU so that instances using the same mesh are contiguous. This makes the writes to `mesh_indirect` sequential.
2.  **Subgroup Operations (Advanced):** For the `fallback_indirect` atomic counter, massive contention will occur if many instances are failing. Use subgroup extensions (if available/supported in your target WGSL version) to aggregate the count within a wave (warp) and perform a single `atomicAdd` per wave.

### 6. Redundant Normal Reconstruction (`ssilvb.wgsl`, `ssr.wgsl`)
**Issue:**
`ssilvb` (SSAO) and `ssr` both contain functions like `compute_normal_from_depth` or `estimate_normal` which sample the depth texture 3-9 times per pixel to reconstruct normals.

**Improvement:**
Since you are writing a voxel engine, you likely know the normals (faces of cubes).
1.  **G-Buffer:** Write your normals to a texture (G-Buffer) during the main `voxel.wgsl` pass.
2.  **Read-Only:** Have SSAO and SSR sample this Normal Texture instead of reconstructing it. This trades a small amount of write bandwidth (in the main pass) for significantly reduced read bandwidth and ALU cost in post-processing.

### 7. Voxel Shader (`voxel.wgsl`)
**Issue:** **Shadow Map Bandwidth.**
You are using a 9-tap PCF (or Poisson) filter.
`textureSampleCompare` is hardware optimized, but 9 fetches per pixel is still heavy.

**Improvement:**
1.  **Dithering/Jitter:** Use 4 taps rotated randomly per pixel (blue noise).
2.  **TAA:** If you have Temporal Anti-Aliasing, the noise will be smoothed out over time, allowing you to get high-quality soft shadows with far fewer samples per frame.

### Summary Checklist

| Priority | Shader | Optimization | Estimated Gain |
| :--- | :--- | :--- | :--- |
| **High** | `ssr.wgsl`, `water.wgsl` | Switch Ray Marching to Screen Space (remove Matrix Mul from loop). | High |
| **High** | `bloom_blur`, `ssao_blur` | Use Linear Sampling (HW filtering) to halve texture fetches. | Medium |
| **High** | `skybox.wgsl` | Move ray direction calculation to Vertex Shader. | Low/Medium |
| **Medium** | `ssilvb`, `ssr` | Use a G-Buffer Normal texture instead of reconstructing from Depth. | Medium |
| **Medium** | `hzb_gen.wgsl` | Implement Single Pass Downsampling (SPD). | Medium |
| **Medium** | `gpu_cull.wgsl` | Sort instances by mesh index on CPU to coalesce indirect buffer writes. | Low |
