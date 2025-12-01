Some observations, now that we run without overflowing buffers. 
1. I see a lot of mesh jobs start even though I see no new geometry on screen as I turn the camera. Are we scheduling meshing jobs for culled geometry? We should not. 

2. The generated world looks weird with respect to the water level, with submerged buildings on the "city" tiles. 

3. Some hills are cut, so if I look at them from one side they climb smoothly, but from the other they look sliced by a knife. 

4. There's still a horizontal plateau of blue voxels, with some islands (all submerged), which I believe is a "lake" biome. Note that there should be no lake biome. 

Note: Lakes and hills should not be separate biomes on top of the general terrain. The general terrain result in these naturally, thanks to the set water level. 

5. The camera_position puts the camera inside the world. Would it be possible to have a failsafe that moves the camera up above the world if it detects on startup that the camera is inside something?


