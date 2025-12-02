- ✅ Water disappeared.
- ✅ Reflected skybox doesn't respect skybox_min_saturation and skybox_night_tint.
- ✅ DoF is not working.
- ✅ Draw calls increase as I move around the map, and do not decrease again.
- ✅ Stale code and warnings remaain.
- ✅ Render scale is only half implemented.
- ✅ Fade out between normal and envelopes gone?
- ✅ Fade out in the far distance gone?

- 🛑 There's a 1-voxel raised ridge between every tile. Three of the ridges can be clearly seen in the screenshot. Let's generate 2x2 tile worlds, so we can quickly test any potential fixes while we are working on this.

- 🛑 It still looks like some (or all?) chunks are present when the viewer starts. I can see underwater chunks as I move the camera towards water, that quickly disappear afterwards (culled). Why is that? Can we perform an initial cull first, so not everything is added? It's either that or we see chunks that are not yet meshed, and they behave differently under water than their meshed counterparts.

- 🛑 Trees are generated under water level.
