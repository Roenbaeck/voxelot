# Screen-Space Lighting for Emissive Voxels - Technical Analysis

**Date:** 2025-11-29  
**Topic:** Screen-space techniques for indirect lighting from emissive voxels

---

## TL;DR - What You Already Have

✅ **You're already using Light Probes** - a 3D world-space solution  
✅ **SSILVB.wgsl exists** - Screen-Space Indirect Lighting with Virtual Bounce (currently for AO)  
⚠️ **Consider**: Extending SSILVB or creating SSGI pass for emissive contribution

---

## Current Implementation: Light Probe System

### What's In Place ([voxel.wgsl:183-200](file:///Users/lars/Documents/GitHub/voxelot/shaders/voxel.wgsl#L183-L200))

```wgsl
// Sample light probes for indirect emissive lighting
var indirect_light = vec3<f32>(0.0, 0.0, 0.0);
for (var i = 0u; i < uniforms.light_probe_count; i++) {
    let probe = light_probes[i];
    let to_light = probe.position - input.world_pos;
    let dist_sq = dot(to_light, to_light);
    let dist = sqrt(dist_sq);
    let attenuation = (probe.color_power.a * 0.01) / max(dist_sq * dist, 128.0);
    
    // Normalize color to prevent oversaturated bleeding
    let probe_brightness = max(probe.color_power.r, max(probe.color_power.g, probe.color_power.b));
    let normalized_color = probe.color_power.rgb / max(probe_brightness, 1.0);
    
    indirect_light += normalized_color * attenuation;
}
// Keep it very subtle - max 3% brightness
indirect_light = min(indirect_light, vec3<f32>(0.03, 0.03, 0.03));
```

### Strengths:
- ✅ Works in 3D world space (emissive voxels off-screen still contribute)
- ✅ Physically-based distance attenuation (cubic falloff)
- ✅ Color normalization prevents over-saturation
- ✅ Already integrated into both instanced and mesh pipelines

### Limitations:
- Limited to number of light probes (discrete sampling)
- Doesn't account for occlusion between light source and receiver
- May miss small emissive details

---

## Screen-Space Techniques: SSGI for Emissive Lighting

### What is SSGI?

**Screen-Space Global Illumination** analyzes visible pixels to estimate indirect light bounces:

1. Read G-Buffer (depth, normals, color, emissive)
2. Sample nearby screen-space pixels in hemisphere around surface normal
3. For each sample:
   - If sample is emissive → add its light contribution
   - Distance falloff based on depth difference
   - Angle-based diffuse term (N·L)
4. Accumulate indirect lighting

### How It Works for Emissive Voxels:

```wgsl
// Pseudo-code for SSGI with emissive
@fragment
fn ssgi_pass(uv: vec2<f32>) -> vec4<f32> {
    let depth = sample_depth(uv);
    let normal = sample_normal(uv);  // From G-Buffer or reconstruct
    let pos = reconstruct_position(uv, depth);
    
    var indirect = vec3(0.0);
    
    // Sample in hemisphere around normal
    for (sample in hemisphere_samples) {
        let sample_uv = uv + sample_offset_in_screen_space;
        let sample_depth = sample_depth(sample_uv);
        let sample_pos = reconstruct_position(sample_uv, sample_depth);
        let sample_color = sample_albedo(sample_uv);
        let sample_emissive = sample_emissive_buffer(sample_uv);  // ← KEY
        
        let to_sample = sample_pos - pos;
        let dist = length(to_sample);
        let L = normalize(to_sample);
        
        // Diffuse term
        let NdotL = max(dot(normal, L), 0.0);
        
        // Distance attenuation
        let atten = 1.0 / (1.0 + dist * dist);
        
        // Add emissive contribution
        if (sample_emissive.a > 0.0) {  // Emissive strength check
            indirect += sample_emissive.rgb * NdotL * atten;
        }
        
        // Also add reflected diffuse from lit surfaces
        indirect += sample_color * sample_lighting * NdotL * atten * 0.1;
    }
    
    return vec4(indirect, 1.0);
}
```

---

## Your Options: 3 Approaches

### Option 1: Extend SSILVB for Emissive (Hybrid Approach) ⭐ RECOMMENDED

**Idea:** Your [`ssilvb.wgsl`](file:///Users/lars/Documents/GitHub/voxelot/shaders/ssilvb.wgsl) already does horizon-based sampling. Add emissive contribution.

**How:**
1. Bind emissive G-Buffer texture to SSILVB shader
2. During horizon search (line 209-251), sample emissive at each sample point
3. When sample is emissive, accumulate light instead of/in addition to occlusion
4. Output: `vec4(indirect_emissive, ao)` or separate pass

**Pros:**
- ✅ Reuses existing infrastructure
- ✅ Same sampling pattern as AO (coherent)
- ✅ Screen-space benefits for visible emissive sources
- ✅ Can run at half-res or quarter-res for performance

**Cons:**
- ❌ Screen-space limitation (off-screen emissives don't contribute)
- ❌ Requires G-Buffer emissive output

**Code Change:** Add to existing SSILVB shader around line 216:
```wgsl
// Inside horizon search loop
let sample_emissive = textureSampleLevel(emissive_tex, sampler, sample_uv, 0);
if (sample_emissive.a > 0.0) {
    let emissive_contrib = sample_emissive.rgb * sample_emissive.a;
    let light_dir = normalize(sample_pos - view_pos);
    let ndotl = max(dot(normal, light_dir), 0.0);
    let dist_atten = 1.0 / max(dist_sq, 1.0);
    indirect_light += emissive_contrib * ndotl * dist_atten;
}
```

---

### Option 2: Dedicated SSGI Pass (Full Implementation)

**Idea:** Create a new `ssgi.wgsl` shader specifically for indirect lighting

**Structure:**
- Input: Depth, Normal, Albedo, Emissive G-Buffers
- Output: Indirect lighting texture
- Composite in post-processing

**Pros:**
- ✅ Clean separation of concerns
- ✅ Can use different sampling patterns (Poisson disk, spiral, etc.)
- ✅ Full control over quality/performance tradeoff
- ✅ Industry-standard approach (UE5, Unity HDRP use SSGI)

**Cons:**
- ❌ More complex implementation
- ❌ Additional render pass overhead
- ❌ Screen-space limitations still apply

---

### Option 3: Hybrid Light Probe + Screen-Space

**Idea:** Combine your existing light probes with screen-space detail

**How:**
1. Keep light probes for global/persistent emissive sources
2. Add screen-space pass for additional detail/accuracy
3. Blend: `final_indirect = light_probes * 0.7 + ssgi * 0.3`

**Pros:**
- ✅ Best of both worlds
- ✅ Light probes handle off-screen sources
- ✅ SSGI adds detailed contact lighting
- ✅ Graceful degradation

**Cons:**
- ❌ Most complex to implement
- ❌ Tuning required for balance

---

## Novel Approach: Voxel-Aware Screen-Space Scattering

### Out-of-the-Box Thinking 🚀

Since you have a **voxel engine** with **GPU culling infrastructure**, here's a creative approach:

#### Emissive Voxel Light Scattering (Compute Shader)

**Concept:** Use compute shader to scatter light from emissive voxels to screen-space

```wgsl
// Compute shader: emissive_scatter.wgsl
@compute @workgroup_size(8, 8, 1)
fn scatter_emissive_light(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // For each screen pixel
    let uv = vec2<f32>(id.xy) / screen_size;
    let depth = sample_depth(uv);
    let pos = reconstruct_position(uv, depth);
    let normal = reconstruct_normal(uv, depth);
    
    var indirect = vec3(0.0);
    
    // Iterate visible emissive voxel instances (from culling pass)
    for (var i = 0u; i < emissive_instance_count; i++) {
        let emissive_voxel = emissive_instances[i];
        let to_light = emissive_voxel.position - pos;
        let dist_sq = dot(to_light, to_light);
        
        // Screen-space radius culling
        let screen_radius = project_radius(dist_sq);
        if (screen_radius < 2.0) { continue; }
        
        let L = normalize(to_light);
        let NdotL = max(dot(normal, L), 0.0);
        let atten = emissive_voxel.power / max(dist_sq * sqrt(dist_sq), 1.0);
        
        indirect += emissive_voxel.color * NdotL * atten;
    }
    
    // Atomic add or write to UAV buffer
    output_buffer[id.xy] = vec4(indirect, 1.0);
}
```

**Advantages:**
- ✅ Uses your existing culling infrastructure
- ✅ Only processes visible emissive voxels
- ✅ Can leverage GPU instancing data
- ✅ Naturally handles voxel granularity
- ✅ No screen-space sampling needed

**Implementation:**
1. Tag emissive voxels during culling pass
2. Create separate instance buffer for emissive voxels
3. Run compute shader to scatter their light to screen
4. Blur/denoise if needed
5. Composite in post-processing

---

## Recommendation: Implementation Priority

### Phase 1: Enhance Light Probes (Low Hanging Fruit) ⭐
- Increase light probe budget (currently capped at max 3%)
- Add per-chunk emissive metadata tracking
- Update probes more frequently for dynamic scenes

### Phase 2: Screen-Space Detail via SSILVB Extension ⭐⭐
- Modify existing SSILVB to sample emissive
- Run at half-resolution
- Blend with light probe result

### Phase 3: Full SSGI or Compute Scatter (Polish) ⭐⭐⭐
- If screen-space detail isn't enough, implement dedicated SSGI
- OR try the novel voxel-aware compute scatter approach

---

## Technical Requirements

### For Screen-Space Approach:
1. ✅ **G-Buffer with emissive** - Already have via `@location(2) emissive` output
2. ⚠️ **Need to output emissive to render target** - Currently only in shader interpolator
3. ✅ **Depth buffer** - Already have
4. ✅ **Normal reconstruction** - SSILVB already does this (line 52-92)

### Minimal Changes for SSILVB Extension:
- Add emissive texture binding
- Modify fragment output to include indirect light
- 50-100 lines of shader code

---

## References

- **SSGI Resources**: UE5 Lumen (screen-space component), Unity HDRP SSGI
- **SSILVB Background**: Based on GTAO/HBAO horizon-based techniques
- **Voxel Cone Tracing**: Alternative 3D approach (more expensive, better quality)

