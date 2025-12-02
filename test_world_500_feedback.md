* Water is just a reflective surface with a tint right now, which means you can see everything under water as clearly regardless of depth. More realistically, the deeper something is under water the harder it is to see. 
---

I would like to have fading and / or dithering for underwater geometry. In config.toml we will set a water_visibility and I would also like underwater geometry to become "invisible" at that point. So if water_level is 500 and water_visibility, anything with y-coordinate 450 or below should no longer be visible. 

My idea was to use the same technique as we have for the distant geometry based on distance from the camera. It seems reasonable that we should be able to use the same type of system, but using distance from water surface level instead (difference in y-coordinate from the water_level set in config.toml) with max transparancy at depth = water_visibility.

This also means we can cull any chunk whose max y-value is below water_level - water_visibility on the CPU so it never reaches the GPU, improving performance.

---

* There's a 1-voxel raised ridge between every tile. Three of the ridges can be clearly seen in the screenshot. 

* It would be very neat if there could be a shoreline somewhere.
