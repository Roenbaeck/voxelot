
1. The generated world looks weird with respect to the water level, with submerged buildings on the "city" tiles. 

2. Some hills are cut, so if I look at them from one side they climb smoothly, but from the other they look sliced by a knife. 

3. There's still a horizontal plateau of blue voxels, with some islands (all submerged), which I believe is a "lake" biome. Note that there should be no lake biome. 

Note: Lakes and hills should not be separate biomes on top of the general terrain. The general terrain result in these naturally, thanks to the set water level. 

4. I think the range for changing water level using the keyboard M and Y is too limited. Our new default of 500 is outside it.

5. It looks like we are generating voxels all the way down to 0, even if none of those will ever be visible. I realize that if we were to allow "digging" we should have ground all the way down. Perhaps the solution is to instead of 500 dynamically start at the bottom of the lowest valley?
