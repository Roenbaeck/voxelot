✅ Is it possible to rotate the skybox very slowly to make it look less static?
✅ Can we make the skybox gradually darker during night, almost black at midnight?
✅ The sky reflected in the water should also be dark at night.
✅ Can we reduce the amount of light as we get closer to midnight, so there’s no “moonlight” at midnight? The whole scene is too bright at night.
✅ Can we add keys to raise and lower water level?
- Can we implement screen space reflections (SSR) for the water? It would be good if we later can set a material in the palette.txt to be reflective as well, so the SSR isn't limited to water.

1. Add SSR texture to water shader's bind group
2. Sample SSR texture in water fragment shader
3. Blend SSR with skybox reflection based on Fresnel/distance

