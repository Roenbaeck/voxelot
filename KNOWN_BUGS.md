# KNOWN BUGS
- ✅ Water disappeared.
- ✅ Reflected skybox doesn't respect skybox_min_saturation and skybox_night_tint.
- ✅ DoF is not working.
- ✅ Draw calls increase as I move around the map, and do not decrease again.
- ✅ Stale code and warnings remaain.
- ✅ Render scale is only half implemented.
- ✅ Fade out between normal and envelopes gone?
- ✅ Fade out in the far distance gone?
- ✅  It still looks like some (or all?) chunks are present when the viewer starts. I can see underwater chunks as I move the camera towards water, that quickly disappear afterwards (culled). Why is that? Can we perform an initial cull first, so not everything is added? It's either that or we see chunks that are not yet meshed, and they behave differently under water than their meshed counterparts.
- ✅ Geometry below water surface level dissolves into white pixels. 
- ✅ When moving the camera there's now some intermittent brightening of the horizon line. It's stable as long as the camera is still, but disappears and appears when I move the camera around. This is related to the new water shader, since it was not there before. 
- ✅ There's an issue around the shoreline, where the coloring remains bright even at night, which looks strange. It should darken like everything else. It may be the foam algorith. This is also related to the new water shader, since it was not there before.
- ✅ There's a 1-voxel raised ridge in many places (between every tile?). 

- 🛑 Trees are generated under water level.
- 🛑 The jungle is too dense, so it looks almost flat green from the top. I think you could make trees even taller, and have even wider canopies. Something like 10-15 on a tile, then add some ground vegetation instead, so we get different levels of vegetation in the jungle. Also, the border blending doesn't seem to be in place, so the jungle ends in straight lines.

