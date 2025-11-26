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

It looks like it's some form of sun-bleed casuing this. The following two changes makes the issue go away:
```
        // Base moon intensity is low when sun up, increases at night.
        // NOTE: The previous logic inadvertently made the moon brightest at
        // midnight (zenith). Prefer moonlight stronger near the horizon and
        // dimmer near zenith to avoid a sudden brightening at midnight.
        let moon_intensity = if sun_height > 0.0 {
            // Daytime: faint moon, almost invisible
            (0.05 * (1.0 - sun_height.clamp(0.0, 1.0))).clamp(0.0, 0.05)
        } else {
            // Night: stronger when moon is near horizon, weaker at zenith
            // moon_height goes from 0.0 (horizon) to 1.0 (zenith/midnight)
            // fade = 1 at horizon, 0 at zenith
            let fade = (1.0 - moon_height).clamp(0.0, 1.0);
            (0.4 * fade).clamp(0.0, 0.4)
        };
```
followed by:
```
        // If the sun is below the horizon, don't give a pseudo 'sun' contribution
        // through `sun_color` – use the dedicated moon light instead for
        // nighttime directional lighting. This prevents double-counting where
        // both the sun (repr. as 'midnight_moon' color) and the moon add
        // directional light during night.
        let sun_color_fixed = if sun_height > 0.0 {
            sun_color
        } else {
            [0.0, 0.0, 0.0]
        };
```

However, this caused a a non-smooth transition in lighting. It's quite jarring visually, and I loved sunrise and sunset the way they looked before. Is it possible to have the previous smooth sunrise and sunset, while still keeping the issue at bay?

- Reflections sometimes look weird from afar, no gradient, but great up close.
- Reflections change when I turn the camera.
