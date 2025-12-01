Some observations, now that we run without overflowing buffers. 
1. I see a lot of mesh jobs start even though I see no new geometry on screen as I turn the camera. Are we scheduling meshing jobs for culled geometry? We should not. 

2. The generated world looks weird with respect to the water level, with submerged buildings on the "city" tiles. 

3. Some hills are cut, so if I look at them from one side they climb smoothly, but from the other they look sliced by a knife. 

4. There's still a horizontal plateau of blue voxels, with some islands (all submerged), which I believe is a "lake" biome. Note that there should be no lake biome. 

Note: Lakes and hills should not be separate biomes on top of the general terrain. The general terrain result in these naturally, thanks to the set water level. 

5. The camera_position puts the camera inside the world. Would it be possible to have a failsafe that moves the camera up above the world if it detects on startup that the camera is inside something?

--------
I am testing world/world_1.oct after the recent changes to generate biomes. The palette.txt was extended with more colors, but palette index 0-7 remain unchanged. I would have expected world_3 to render as before, but I can see that unmeshed chunks (fallback to individual voxels) render correctly with respect to the palette, but that when these get replaced by meshed geometry, some meshes turn green, and it's not the green color from the legacy palette (0 to 7). See screenshot showing both the unmeshed and meshed chunks next to each other and differing in color in some places. 

I don't understand why this is happening, but it is a bug. Can you check why the meshing is picking up the wrong color?

If remove the entries 8 to 46, just keeping the original 0 to 7 in palette.txt the bug is gone. If I add back all 46 entries, the bug comes back. 

Note that since individual voxels render with a different color than the meshed variant of the same geometry, there is a bug here somewhere. 
