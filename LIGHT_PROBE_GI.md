# CPU-Based Light Probe GI Renewal

## Goal Description

Replace the current screen-space Global Illumination (GI) in [ssilvb.wgsl](file:///Users/lars/Documents/GitHub/voxelot/shaders/ssilvb.wgsl) with a robust, CPU-calculated Light Probe system. This leverages available CPU cycles to compute stable, long-range indirect lighting, decoupling it from the GPU frame budget and resolving screen-space limitations (like disappearing off-screen lights).

The new system will:

1.  **Calculate Probes on CPU**: Place a light probe at the center of each 16x16x16 leaf chunk.
2.  **Ray-Cast Irradiance**: Trace rays from "Emissive Chunks" to nearby probes to calculate light transfer, accounting for occlusion by other chunks.
3.  **Store in Ambient Cube**: Accumulate light into 6 directional faces per probe (Up, Down, North, South, East, West) to preserve directionality.
4.  **Upload to GPU**: Maintain a "Clipmap" or "Grid" of probes around the camera in a Storage Buffer.
5.  **Render**: Sample this buffer in [ssilvb.wgsl](file:///Users/lars/Documents/GitHub/voxelot/shaders/ssilvb.wgsl) to apply GI, blending it with existing AO.

## User Review Required

> [!IMPORTANT]
> **Performance Trade-off**: This moves GI calculation to the CPU. While this frees up the GPU, the CPU update rate might be lower (e.g., 10-20Hz) compared to per-frame GPU execution. This effectively makes the GI "low frequency" in time, which is generally acceptable for diffuse indirect light but might lag slightly for fast-moving dynamic lights.

> [!NOTE]
> **Resolution**: Probes are spaced 16 voxels apart (one per leaf chunk). This is "coarse" GI. It will provide excellent general ambience but won't capture detailed small-scale light leaks relative to single voxels. This is intended to be balanced by the existing high-frequency AO.

## Proposed Changes

### Rust Logic (`src/`)

#### [NEW] `src/gi.rs`

-   Create a `GiSystem` struct.
-   Implement `update_probes` method:
    -   Identify active leaf chunks around the camera (Grid/Clipmap).
    -   Identify emissive chunks (source).
    -   For each probe, gather light from emissive sources using ray transformation.
    -   Perform voxel-level (or chunk-level) occlusion checks using the [World](file:///Users/lars/Documents/GitHub/voxelot/src/lib_hierarchical.rs#907-918) hierarchy.
-   Implement `GiProbe` struct (serialized for GPU).

#### [MODIFY] `src/voxelot.rs` (or [lib.rs](file:///Users/lars/Documents/GitHub/voxelot/src/lib.rs)/`main.rs`)

-   Integrate `GiSystem`.
-   Call `update_probes` in a separate thread or per frame.
-   Upload the probe buffer to the GPU.

#### [MODIFY] `src/resources.rs` (or where BindGroups are created)

-   Add the new GI Storage Buffer to the standard bind group (likely `bind_group_0`).

### Shaders (`shaders/`)

#### [MODIFY] [shaders/ssilvb.wgsl](file:///Users/lars/Documents/GitHub/voxelot/shaders/ssilvb.wgsl)

-   Remove existing Screen-Space GI accumulation logic.
-   Add binding for `GiProbes` storage buffer & uniforms (grid origin, size).
-   Implement `sample_gi_probe(world_pos)`:
    -   Calculate probe index from world position.
    -   Fetch the 6 faces.
    -   (Optional) Trilinear interpolation between 8 nearest probes for smoothness.
    -   Return the directional irradiance.
-   Combine the sampled GI with the existing AO visibility.

## Verification Plan

### Automated Tests

-   Verify Rust GI calculation: Write a unit test in `gi.rs` with a simple scene (1 light, 1 wall, 1 probe) and assert the probe receives light (and shadow if occluded).

### Manual Verification

-   **Visual Check**: Place a bright emissive block (e.g., Sunlit Red Concrete). Move a non-emissive block near it. Verify the non-emissive block receives red light on the facing side.
-   **Occlusion**: Build a wall between the light and the block. Verify the light disappears.
-   **Range**: Move far away. Verify the light is still visible (unlike Screen-Space GI).
-   **Performance**: Check frame rate and CPU usage.