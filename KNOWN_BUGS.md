- ✅ Water disappeared.
- ✅ Reflected skybox doesn't respect skybox_min_saturation and skybox_night_tint.
- ✅ DoF is not working.
- ✅ Draw calls increase as I move around the map, and do not decrease again.
- ✅ Stale code and warnings remaain.
- ✅ Render scale is only half implemented.
- ✅ Fade out between normal and envelopes gone?
- ✅ Fade out in the far distance gone?
- ✅  It still looks like some (or all?) chunks are present when the viewer starts. I can see underwater chunks as I move the camera towards water, that quickly disappear afterwards (culled). Why is that? Can we perform an initial cull first, so not everything is added? It's either that or we see chunks that are not yet meshed, and they behave differently under water than their meshed counterparts.

- 🛑 Geometry below water surface level dissolves into white pixels. This causes water to look harsh. It would be better if the discarded pixels took on the water color at its depth, so they would progressively get darker the deeper they are, and match the darkening of the water. The screenshot is taken under water to show the white dissolve. We need to change color rather than discard, if that's possible with the hash? 

- 🛑 There's a 1-voxel raised ridge in many places (between every tile?). The issue is clearly visible in the screenshot. Can you possibly fix this?
- 🛑 Trees are generated under water level.
