# TODO 
- ✅ Is it possible to rotate the skybox very slowly to make it look less static?
- ✅ Can we make the skybox gradually darker during night, almost black at midnight?
- ✅ The sky reflected in the water should also be dark at night.
- ✅ Can we reduce the amount of light as we get closer to midnight, so there’s no “moonlight” at midnight? The whole scene is too bright at night.
- ✅ Can we add keys to raise and lower water level?
- ✅ Can we implement screen space reflections (SSR) for the water? It would be good if we later can set a material in the palette.txt to be reflective as well, so the SSR isn't limited to water.
- ✅ Can we make the water level configurable in config.toml (with save on exit)?
- ✅ Can we make a toggle for the GUI overlay? Perhaps the F5-key?
- ✅ Can you update README with any missing keybindings?
- ✅ There's a non-smooth color transition just before time = 0.274 (I paused there).
- ✅ I would like the sun to shine longer, so we see shadows climbing higher up buildings at dusk before light fades away.
- ✅ Skybox should get colors washed out the darker it gets. Now it has a saturated yellow unnatural tint at midnight. 
- ✅ Skybox needs a blueish tint. 
- ✅ Distant objects look like they again get brighter closer to the horizon at early night and early morning. It’s making them look unnaturally lit up. 
- ✅ Reflections from SSR are only visible if the camera is close to a reflective surface.
- ✅ Implement a cache of shells that can be quickly masked depending on camera position, for simple culling and fewer iterations.
- ✅ The shader dof_blur.wgsl was removed (deprecated placeholder).
- ✅ Why is the fullscreen render so much slower than the windowed render? It drops from 120FPS to 20FPS when I go fullscreen.
- ✅ The underwater geometry looks a little bit sharp. See the encircled area in the screenshot for an example. What do you suggest? Can we make it even more water-like?
- ✅ Print how many chunks are culled by the depth culler.
- ✅ It would be very neat if there could be a shoreline somewhere.
- ✅ Add a beach biome.
- ✅ Reflections should be stronger at night than in the day, since nothing is drowning out the light.

- 🛑 Sliders instead of buttons to change settings.
- 🛑 Can all of our code be compiled as wasm?

## Ideas from the Eye the Isles comparison (Oct 2026)
`../eyetheisles` began as a Swift/Metal port of this renderer (July 2026) and has evolved since. The fixes that came out of comparing the two were applied directly; what follows was deliberately left for later. Paths like `Renderer.swift` refer to that repo. Items marked *(unmeasured)* are hypotheses, not profiled results.

Larger architectural changes:
- Replace the camera-driven box/envelope/fallback-cube LOD ladder (`culling.rs` ~895-913, shell-cube instances in `voxelot.rs`) with a persistent proxy underlay: exact meshes inside a radius, prebaked 4³/8³ greedy proxies beneath, impostors only below ~1.5 px with an exit band, and a per-16³ GPU mask that hides the proxy only where exact geometry was drawn this frame. `4_TIER_LOD.md` in eyetheisles explains why the ladder was abandoned. The seam took three follow-up commits there (83faddf, f04ba86, 4eb3fd3); expect ~200k hidden proxy triangles drawn under exact geometry.
- Real geometric LOD: a 4³ coarse grid per leaf using the dominant exposed material (`downsampleToCoarseGrid`, `CityMeshBuilder.swift`). Our "envelope" mesh only recolours full-resolution geometry. A stepping stone to the proxy underlay.
- Compact mesh vertex: `MeshVertex` is 72 bytes, eyetheisles uses 32 (snorm8 normal, unorm8 colour/emissive, reflectivity in colour alpha). About 2.25x more meshes in the same budget; needs an emissive intensity scale and changes to the pawn, shadow and mesh pipelines.
- Camera-relative projection and depth reconstruction. Absolute `mvp * world_pos` loses depth precision far from the origin; eyetheisles measured 996 of 34,560 beach samples misclassified at ~3,000 units before the fix and none after. Touches `voxel.wgsl`, `water.wgsl`, `ssr.wgsl`, `impostor.wgsl`, `gpu_cull.wgsl`, `radiance_cascades.wgsl` and the host uniforms together. A prerequisite for very large worlds.
- Memory layout: `Voxel` is a 16-byte enum, and 33-72% of leaf chunks in the shipped worlds are uniform but stored as 4096 entries (60,934 of 182,821 in `large_world_test`). Collapse uniform chunks to `Solid` (middle level first), consider dense u8 leaf storage, and consider a fixed 4096-bit presence set instead of Roaring (34 vs 1.5 ns contains+rank at 5% density, micro-benchmark only).
- Region index and streaming for worlds larger than RAM: a `.idx` of region resources, a splitter at top-level cell boundaries (eyetheisles' `VHCRegionSplitter` already depends on this crate and could move into `src/bin`), a 2x2 resident window with velocity lead and hysteresis, and per-region baked proxy and static-GI sidecars with a source fingerprint.
- Extract a headless `voxelot-core` crate (`lib_hierarchical`, `palette`, `file_format`; they only need croaring, rustc-hash, zstd, bytemuck). eyetheisles' world generator compiles those files from this checkout unpinned, so any change here silently changes its output.

Smaller items not done in this round:
- Hysteresis on stateless distance switches (impostor 1.5 -> 2.25 px exit band, detail/envelope thresholds in `gpu_cull.wgsl`).
- Keep mesh residency independent of the view direction: the queue is pruned to frustum/shell-culled chunks (`voxelot.rs` ~13697), so turning shows late meshes *(unmeasured)*.
- The shadow pass draws every entry of both `mesh_cache` and `envelope_mesh_cache` (`voxelot.rs` ~15031); cache the caster set per generation.
- `VisibilityCache` in `culling.rs` is unused by the viewer; per-frame `cull_visible_voxels_parallel` could be threshold-gated *(unmeasured)*.
- Eviction tiers (edited > visible > recently leased > LRU) and a CPU mesh-result cache instead of pure LRU remeshing.
- A GI line-of-sight result cache (eyetheisles `cachedLineOfSight`: 32k entries keyed by start and target) *(unmeasured)*.
- Near sun-shadow cascade (0.17 units/texel over 220 units). The single 4096² map gives ~0.2 on the default config but ~0.45 on `large_world_test`.
- Energy-conserving reflection composite: `mix(color, reflection, a) + emission` instead of additive SSR (`post_composite.wgsl`); sky reflection may also be counted twice with `voxel.wgsl` *(unmeasured)*.
- Bloom: always feed emission into bloom and raise the lit-surface threshold from 0.7 toward 1.6 (aesthetic).
- `get_water_normal` adds one scalar noise value to both slope axes, which biases the normal diagonally *(unmeasured)*.
- Generator: soft ceiling instead of the hard height clamp (`generate_world.rs` ~1485, 1747, 2286) to avoid flat-topped mesas; stratified rock and tanh relief; parallel per-tile voxelization; trees and boulders are clamped at tile edges; `--timings` stage output; a byte-digest regression check; write `.vhc` atomically (it currently writes straight to the final path).
- eyetheisles removed foam, caustics (4d72b88) and WTS (0a0c272) for art and phone-cost reasons. Decide whether we want the same; at least check night views for the "nighttime mountain glow" that motivated removing WTS.

---
Generate a large world for stress testing:
```
cargo run --release --bin generate_world -- --water-level 25.0 --height-range 50.0 --tile-width 32 --tile-height 32 --output-name worlds/large_world_test
```
