# Voxelot Configuration Reference

This document lists all configuration options used by the application, grouped by sections to match the `config.toml` file structure.

Each entry shows the default value (as found in `src/config.rs`), a short description of what the setting controls, where the setting is used in the code, and the expected effects of changing it.

---

## World

- `world.file` (string)
  - Default: `"world_1.vhc"`
  - Description: The path to the hierarchical chunk (`.vhc`) representation of the world loaded by the viewer.
  - Used: `src/bin/voxelot.rs` (loading world)
  - Effect of change: Changing this selects a different `.vhc` file to load; the viewer will display the selected world.

- `world.palette` (string)
  - Default: `"worlds/palette.txt"`
  - Description: Path to the palette definition which maps voxel indices to color/emissive values.
  - Used: `src/bin/voxelot.rs` (palette loading), `src/palette.rs`
  - Effect of change: Changing this alters how voxel indices translate to visible colors and emissive values in the viewer.

- `world.camera_position` (array [f32; 3])
  - Default: `[320.0, 100.0, 320.0]`
  - Description: Initial camera world position when the viewer starts.
  - Used: `src/bin/voxelot.rs` (initial camera setup)
  - Effect of change: Modifying this moves the camera to a different starting point in the loaded world.

- `world.water_level` (float)
  - Default: `16.0`
  - Description: Water plane height (in voxel units); used for generating and displaying water volume and for some generator logic.
  - Used: `src/bin/generate_world.rs` (world generation), `src/bin/voxelot.rs` (viewer initialization & culling logic)
  - Effect of change: Higher values raise the water plane, potentially submerging terrain and causing some generator biome selections (e.g., Beach) to change.

- `world.water_visibility` (float)
  - Default: `20.0`
  - Description: Camera-visible depth from the water surface (how many voxels below the water surface remain visible) — used for underwater culling and fog effects.
  - Used: `src/bin/voxelot.rs` (water visibility), `src/shaders/` (uniforms)
  - Effect of change: Increase to make deeper underwater geometry visible; decreasing reduces underwater draw and simulation cost.

---

## Rendering

- Note: `rendering.lod_subdivide_distance` and `rendering.lod_merge_distance` were removed — use `rendering.chunk_lod_distance` for LOD culling thresholds and to control how far detailed chunks are rendered. Use PageUp/PageDown to tune this value at runtime.

- `rendering.chunk_lod_distance` (float)
  - Default: `800.0`
  - Description: LOD render distance for chunk LOD selection (worker/renderer uses this to choose mesh LOD for chunks). This is the canonical LOD distance and used by the culling system to toggle between a chunk's averaged LOD mesh and subdividing further.
  - Used: `src/culling.rs` (hierarchical chunk culling), `src/bin/voxelot.rs` (initial LOD settings and runtime adjustment via PageUp/PageDown)
  - Effect of change: Increasing this keeps chunk averaged LODs rendered further out (higher detail at a performance cost). Decreasing reduces visible detail further away, improving performance.

- `rendering.fov_degrees` (float)
  - Default: `70.0`
  - Description: Field-of-view of the camera in degrees.
  - Used: `src/bin/voxelot.rs` (camera creation), affects projection matrix
  - Effect of change: Higher FOV increases peripheral coverage but distorts the scene and affects culling/LOD thresholds.

- `rendering.near_plane` (float)
  - Default: `0.1`
  - Description: Camera near plane (distance) used in projection matrix; affects depth precision and near culling.
  - Used: `src/bin/voxelot.rs` (camera creation and updates)
  - Effect of change: Increasing near plane improves depth precision but may clip nearby geometry.

- `rendering.far_plane` (float)
  - Default: `5000.0`
  - Description: Camera far plane (distance) for rendering frustum.
  - Used: `src/bin/voxelot.rs` (camera), modifies culling and LOD updates.
  - Effect of change: Increasing allows further geometry to be visible but affects depth precision and performance.

- `rendering.camera_speed_multiplier` (float)
  - Default: `1.0`
  - Description: A multiplier applied to the base camera movement speed.
  - Used: `src/bin/voxelot.rs` (CameraController speed bootstrap and runtime updates)
  - Effect of change: Higher values make the camera move faster.

- `rendering.window_width` (int)
  - Default: `1280`
  - Description: Initial window width used when creating the viewer.
  - Used: `src/bin/voxelot.rs` (Window creation / resume), saves and updates runtime.
  - Effect of change: Sets the initial window size.

- `rendering.window_height` (int)
  - Default: `720`
  - Description: Initial window height for the viewer; used to initialize render targets.
  - Used: `src/bin/voxelot.rs` (Window creation / resume), saves and updates runtime.
  - Effect of change: Sets the initial window size.

---

## Atmosphere

- `atmosphere.fog_density` (float)
  - Default: `0.003`
  - Description: The density of fog/atmospheric scattering used in scene compositing.
  - Used: `src/bin/voxelot.rs` (composite shader uniforms, fog calculations)
  - Effect of change: Increase to make the world foggier (less visibility), decrease for clearer views.

- `atmosphere.time_of_day` (float 0.0..1.0)
  - Default: `0.5`
  - Description: Time of day normalized as a float (0.0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset, etc.)
  - Used: `src/bin/voxelot.rs` (lighting calculation & day-night cycle)
  - Effect of change: Alters sun angle, color, and baked lighting across the scene.

- `atmosphere.night_skybox_brightness` (float)
  - Default: `0.02`
  - Description: Brightness multiplier used for the night skybox.
  - Used: `src/bin/voxelot.rs` (skybox compositing)
  - Effect of change: Set a higher value to increase the sky brightness at night.

- `atmosphere.horizon_fade_up` (float)
  - Default: `0.05`
  - Description: Fade start of horizon light when the sun rises.
  - Used: `src/bin/voxelot.rs` (skybox & atmosphere rendering)
  - Effect of change: Adjust horizon fade sharpness/timing.

- `atmosphere.horizon_fade_down` (float)
  - Default: `0.25`
  - Description: Fade end for horizon brightness transitions.
  - Used: `src/bin/voxelot.rs` (skybox & atmosphere rendering)
  - Effect of change: Affects gradient of horizon lighting at dusk/dawn.

- `atmosphere.skybox_fade_up` / `skybox_fade_down` / `skybox_min_saturation`
  - Defaults: `0.05`, `0.6`, `0.16`
  - Description: Parameters controlling how the skybox and its saturation fade across the day/night cycle.
  - Used: `src/bin/voxelot.rs` & shaders (skybox composition)
  - Effect of change: Alters how colors desaturate during nights/dawn/dusk.

- `atmosphere.skybox_night_tint` (array [f32; 3])
  - Default: `[0.09, 0.12, 0.24]`
  - Description: RGB color tint applied to the skybox at night.
  - Used: `src/bin/voxelot.rs` and `shaders/skybox.wgsl`
  - Effect of change: Shift the hue of the sky during night.

- `atmosphere.skybox_tint_strength` (float)
  - Default: `0.5`
  - Description: Strength of the night tint applied to the skybox.
  - Used: `src/bin/voxelot.rs` and `shaders/skybox.wgsl`
  - Effect of change: Higher values make the night color more pronounced.

---

## Effects (Depth of Field, Bloom, SSAO)

### Depth of Field (`effects.depth_of_field`)

- `enabled` (bool)
  - Default: `true`
  - Description: Enable or disable the depth of field post-processing pass.
  - Used: `src/bin/voxelot.rs` (DoF pipeline toggling & uniforms)
  - Effect of change: Disabling removes DoF.

- `focal_distance` (float), `focal_range` (float)
  - Defaults: `120.0`, `16.0`
  - Description: The focal point distance and the width of the focused band.
  - Used: `src/bin/voxelot.rs` and DoF shaders (`shaders/*`)
  - Effect of change: Adjust what the camera focuses on and the softness of focus falloff.

- `blur_strength` (float)
  - Default: `1.6`
  - Description: Strength of the blur applied by DoF; lower = softer blur.
  - Used: `src/bin/voxelot.rs` and shaders
  - Effect of change: Larger values increase the intensity of blur; set to `0.0` to effectively disable.

- `kawase_iterations` / `kawase_offset` / `kawase_enabled`
  - Defaults: `3`, `1.0`, `false`
  - Description: Parameters for Kawase blur used by the DoF pass (multi-pass Gaussian-like blur).
  - Used: `src/bin/voxelot.rs` (DoF pipeline creation and Kawase settings)
  - Effect of change: More iterations and larger offset increase the blur spread (costs more GPU time).

---

### Bloom (`effects.bloom`)

- `enabled` (bool)
  - Default: `true`
  - Description: Toggle bloom post-processing effect.
  - Used: `src/bin/voxelot.rs` and bloom shaders
  - Effect of change: Disabling will remove glowing highlights.

- `threshold`, `knee`, `intensity`, `bloom_strength`, `saturation_boost`, `exposure`, `blur_radius`
  - Defaults: `0.7`, `0.6`, `1.8`, `1.6`, `1.5`, `1.18`, `3.8`
  - Description: Controls the bloom intensity, threshold for highlight extraction and tonemapping.
  - Used: `src/bin/voxelot.rs` (bloom uniforms) and bloom shaders
  - Effect of change: Raise `threshold` to limit bloom to brighter areas; raise `intensity`/`bloom_strength` to increase bloom.

- `kawase_enabled` / `kawase_iterations` / `kawase_offset`
  - Defaults: `true`, `4`, `0.5`
  - Description: The bloom blur pipeline uses Kawase blur; these parameters enable and tune its iterations.
  - Used: `src/bin/voxelot.rs` (Kawase pipeline) and bloom shaders
  - Effect of change: More iterations increase blur and cost more GPU time.

---
### Screen-Space Reflections (`effects.ssr`)

- `enabled` (bool)
    - Default: `true`
    - Description: Toggle screen-space reflections rendering (SSR) — used for water and reflective materials.
    - Used: `src/bin/voxelot.rs` (SSR pipeline creation, toggled by runtime `R`), `shaders/ssr.wgsl` (shader implementation)
    - Effect of change: When enabled, SSR will be calculated and composited (can increase GPU usage depending on complexity). When disabled, reflections won't render from SSR.

    - `max_steps` (int)
      - Default: `32`
      - Description: Maximum raymarch steps for SSR sampling; increase for improved coverage and fewer misses at the cost of GPU time.
      - Used: `src/bin/voxelot.rs` (ssr uniforms and shader parameters), `shaders/ssr.wgsl` (raymarch loop)
      - Effect of change: Higher values increase accuracy of reflections but cost more performance.

    - `max_binary_steps` (int)
      - Default: `4`
      - Description: Number of binary search refinement steps used after raymarch finds a hit to localize the reflection point.
      - Used: `src/bin/voxelot.rs`, `shaders/ssr.wgsl`
      - Effect of change: More refinement steps improve precision but increase GPU workload.

    - `step_size` (float)
      - Default: `0.5`
      - Description: The base step size multiplier for the SSR raymarch (in world units * step_size).
      - Effect of change: Smaller values yield finer raymarch and higher accuracy, larger values speed the pass at the risk of missed intersections.

    - `thickness` (float)
      - Default: `0.5`
      - Description: Thickness threshold used when testing ray hits against geometry; larger thickness tolerates thin surfaces and prevents leaks.
      - Effect of change: Increasing thickness reduces holes but can produce incorrect reflections if set too large.

---

### Screen-Space Ambient Occlusion (`effects.ssao`)

- `enabled` (bool)
  - Default: `true`
  - Description: Toggle screen-space ambient occlusion.
  - Used: `src/bin/voxelot.rs` and SSILVB/SSAO shaders
  - Effect of change: Disabling removes ambient occlusion calculations for surfaces.

- `sample_count`, `slice_count`, `radius`, `thickness`, `strength`, `blur_enabled`, `blur_radius`
  - Defaults: `8`, `4`, `4.0`, `0.5`, `1.0`, `true`, `2.0`
  - Description: Parameters used for SSAO sampling, occlusion strength and blur pass settings.
  - Used: `src/bin/voxelot.rs` and SSAO shaders
  - Effect of change: Increasing `sample_count` and `radius` yields more accurate occlusion at performance cost.

---

## Shadows

- `shadows.map_size` (int)
  - Default: `4096`
  - Description: Shadow map resolution; this is a large consumer of GPU memory.
  - Used: `src/bin/voxelot.rs` and shadow shader pipeline configuration
  - Effect of change: Larger size increases shadow fidelity at higher GPU memory cost.

- `shadows.darkness` (float)
  - Default: `1.0`
  - Description: Multiplier controlling shadow darkness intensity.
  - Used: `src/bin/voxelot.rs` and shadow compositing shader
  - Effect of change: Increase to make shadows darker, decrease to soften shadows.

- `shadows.backface_ambient_scale` (float)
  - Default: `0.7`
  - Description: Scale applied to ambient shadow on backfaces (avoid full darkness on backfaces).
  - Used: `src/bin/voxelot.rs` and some shading logic
  - Effect of change: Lowering leads to darker backface ambient; raising keeps them brighter.

- `shadows.pcf_radius` (float) and `pcf_poisson_samples` (int)
  - Defaults: `1.0`, `0`
  - Description: Radius and Poisson sample count for PCF shadow filtering; `0` disables Poisson sampling.
  - Used: `src/bin/voxelot.rs` and shaders
  - Effect of change: Increasing PCF radius smooths shadows at cost of sampling.

---

## Performance

- `performance.mesh_cache_budget_mb` (u64)
  - Default: `256`
  - Description: Memory budget (MB) for mesh cache; stashes meshed chunk vertex/index buffers.
  - Used: `src/bin/voxelot.rs` (mesh cache logic)
  - Effect of change: Increasing gives more memory to GPU caches and reduces re-meshing.

- `performance.mesh_worker_count` (Option<usize>)
  - Default: `None` (auto)
  - Description: Number of worker threads for mesh generation; if `None`, auto-detected.
  - Used: `src/bin/voxelot.rs` (worker pool creation)
  - Effect of change: Increasing thread count may increase meshing throughput if CPU cores allow.

- `performance.mesh_upload_baseline` (usize)
  - Default: `4`
  - Description: Number of jobs to batch before uploading meshes to GPU.
  - Used: `src/bin/voxelot.rs` (mesh upload logic)
  - Effect of change: Higher value batches more work for fewer uploads but may increase latency.

- `performance.envelope_distance`, `envelope_fade_range`, `max_envelope_distance`
  - Defaults: `256.0`, `32.0`, `1000.0`
  - Description: Envelope mesh distances for fallback meshes and their fade ranges.
  - Used: `src/bin/voxelot.rs` (envelopes & LOD)
  - Effect of change: Larger values increase how far distance envelope meshes are used.

- `performance.mesh_priority_sort_interval_frames` (int)
  - Default: `30`
  - Description: How often (in frames) to sort the mesh priority queue.
  - Used: `src/bin/voxelot.rs` (mesh scheduling)
  - Effect: Higher interval reduces CPU overhead but increases delayed reordering of job priorities.

- `performance.mesh_buffer_pool_entries` (usize)
  - Default: `256`
  - Description: Pool entry count for pooled mesh buffers.
  - Used: `src/bin/voxelot.rs` (buffer pool)
  - Effect of change: More entries help caching; less reduces memory.

- `performance.mega_vertex_buffer_mb` / `mega_index_buffer_mb`
  - Defaults: `512`, `256`
  - Description: Memory slab sizes for big vertex/index buffers used to store many chunk meshes.
  - Used: `src/bin/voxelot.rs` (GPU buffer allocations)
  - Effect: Increasing reduces out-of-memory risk but consumes GPU memory.

- `performance.max_draw_capacity` (usize)
  - Default: `20000`
  - Description: Limit to the number of draw calls / instance draws allowed.
  - Used: `src/bin/voxelot.rs` (draw scheduling)
  - Effect of change: Increasing may render more draws but hit performance/limits.

- `performance.render_scale` (float)
  - Default: `1.0`
  - Description: Internal render resolution scaling (1.0 = native resolution).
  - Used: `src/bin/voxelot.rs` (render target sizes)
  - Effect of change: Lower values reduce render resolution and improve GPU performance.

- `performance.max_gpu_instances` (usize)
  - Default: `6000000`
  - Description: Safeguard limit controlling how many per-instance draw entries the GPU will be allocated for.
  - Used: `src/bin/voxelot.rs` (allocations / memory guard)
  - Effect of change: Risk of GPU memory overflow if set too high; insufficient instances can prematurely limit draw capacity.

---

