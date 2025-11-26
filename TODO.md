✅ Is it possible to rotate the skybox very slowly to make it look less static?
✅ Can we make the skybox gradually darker during night, almost black at midnight?
✅ The sky reflected in the water should also be dark at night.
✅ Can we reduce the amount of light as we get closer to midnight, so there’s no “moonlight” at midnight? The whole scene is too bright at night.
✅ Can we add keys to raise and lower water level?
✅ Can we implement screen space reflections (SSR) for the water? It would be good if we later can set a material in the palette.txt to be reflective as well, so the SSR isn't limited to water.
✅ Can we make the water level configurable in config.toml (with save on exit)?

✅ Can we make a toggle for the GUI overlay? Perhaps the F5-key?
✅ Can you update README with any missing keybindings?

- I can see an issue during night. Some surfaces remain bright, even though I don't think they should. The building in the middle ought to be consistently dark green, but it is brighter up to a certain y-level, then distincly turns dark. I see this phenomenon on several buildings at night, and am not sure when this started to appear. Can you try to figure out what is causing this?

I checked closer and at night, I can see the division line start from the bottom of the building, reach it's peak height at midnight, then decline back down and disappear completely in day time. The line moves consistently like this regardless of camera angle, and doesn't jump around or disappear. 

- Reflections sometimes look weird from afar, no gradient, but great up close.
- Reflections change when I turn the camera.
