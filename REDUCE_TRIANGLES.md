Our greedy mesher in meshing_optimized is very fast. I am trying to figure out a way to reduce the number of triangles drawn on screen, so what if we were to use it twice, once for the actual voxels in the chunk, and once for an envelope of those voxels and where we treat all of those voxels as having the same type. I believe type 0 (palette.txt) is unusued, so we can use that for tuning the color. 

The envelope version should have considrably less triangles, and we can use this at a certain range away from the camera, since colors tend to be washed out for distant objects anyway. Do you think this would help increase FPS?

---

Can we request envelopes for everything beyond a certain distance, instead of the non-envelope version, and it's the envelope version that will be drawn. Once we get closer, we use the individual voxel version, as we do today, and resubmit chunks for non-envelope versions (unless that is already in the cache). We will likely need a second cache, for enveloped meshes, mirroring the behavior of the existing cache. The cache we have now is 256MB, so we can have a similarly sized enevelope mesh cache. What do you think?

---

That works beautifully, and gave a 30-40% FPS boost. I would like two things. Expose the `envelope_distance` in config.toml. I think 256 is a bit near for the type of scene I have (city), but it could be sufficient for other types of scenes. 

I'd also like for the colors (in the non-envelope region) to approach the color used for the envelopes as we get closer to the non-envelope / envelope boundary. That way the transition will become sort of "seamless". Do you understand what I would like to achieve?

---

Just one more thing, the non-meshed chunks are rendered using their dominant color right now, and it looks like they either aren't faded to the type 0 color or they are beyond the distance to begin with and therefore render with their dominant color. If they are already affected by fading when near the camera, the fix would be to set type 0 color if they're outside `envelope_distance`. Can you check and fix?

