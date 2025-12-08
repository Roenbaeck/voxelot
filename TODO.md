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

- 🛑 Reflections should be stronger at night than in the day, since nothing is drowning out the light.
- 🛑 Sliders instead of buttons to change settings.
- 🛑 Can all of our code be compiled as wasm?


I want to redo GI. It's not working well. Can we instead leverage it using some CPU cycles? The idea is to add a light probe in the center of each leaf chunk (that contains 16x16x16 voxels). For the proble we calculate an average light contribution per face direction, by tracing rays from all emissive voxels nearby enough to have a meaningful contribution, and that aren't fully occluded by other leaf chunks. This way we should be able to quickly calculate the GI for each light probe, and store it in a buffer that can be sampled by the shader. 

Since voxels in the world aren't likely to change rapidly, we could keep a persistent "probe map" of the world, and only update it when the world changes. This would be much faster than calculating GI for every chunk every frame.

Do you think this could work? I'm not sure whether we'd need to calculate occlusion more often than we do now, or if we could get away with only doing it for the light probes. It may be too "rough" to be useful, but perhaps it could be balanced with the suff we're already calculating for AO and shadow mapping somehow? Note that AO looks great, so I don't want to touch that part of the SSILVB shader.

What do you think?




---
Generate a large world for stress testing:
```
cargo run --release --bin generate_world -- --water-level 25.0 --height-range 50.0 --tile-width 32 --tile-height 32 --output-name worlds/large_world_test
```
