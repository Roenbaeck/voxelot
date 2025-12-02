* Water is just a reflective surface with a tint right now, which means you can see everything under water as clearly regardless of depth. More realistically, the deeper something is under water the harder it is to see. 
---

I am trying to get fading and / or dithering to work for underwater geometry. In config.toml we set a water_visibility and water color correccly changes based on depth, but I would also like underwater geometry to become "invisible" at that point. 

My idea was to use the same technique as we have for the distant geometry based on distance from the camera. It seems reasonable that we should be able to use the same type of system, but using distance from water surface level instead (difference in z-coordinate from the water_level set in config.toml). 

There's an attempted implementation in place, but it's not working. Feel free to replace it if you think that's quicker than trying to fix what is already there.
---

* There's a 1-voxel raised ridge between every tile. Three of the ridges can be clearly seen in the screenshot. 

* It would be very neat if there could be a shoreline somewhere.
