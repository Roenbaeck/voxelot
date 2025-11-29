# Dead Code Analysis - Voxelot Engine

**Analysis Date:** 2025-11-29  
**Status:** Initial analysis complete

> [!NOTE]
> This document identifies code that is either unused, unlikely to be used, or marked for potential removal based on comments and usage patterns. Items are categorized by confidence level.

---

## High Confidence - Safe to Remove

### 1. Config Option: `fallback_bbox_shrink`
**Location:** [`src/config.rs`](file:///Users/lars/Documents/GitHub/voxelot/src/config.rs#L197-L198)

**Evidence:**
```rust
pub fallback_bbox_shrink: f32,  // Line 198
```

**Comment in code:**
```rust
// No shrinkage - prevents position mismatch between fallback voxels and meshes
```

**Usage:** Configuration option is defined and has a default value (1.0), but based on the comment this was intentionally set to no-op value. From conversation history (conversation ca73a79f), this option was confirmed as causing visual mismatches.

**Recommendation:** ✅ Remove config option entirely:
- Remove from `PerformanceConfig` struct
- Remove `default_fallback_bbox_shrink()` function
- Remove from `impl Default for PerformanceConfig`
- Verify no actual usage in voxelot.rs (likely unused)

---

### 3. Unused Utility Binaries - Verification Tools

#### `verify_glam.rs`
**Purpose:** Simple verification of glam matrix perspective calculations  
**Usage Likelihood:** One-time verification, no longer needed  
**Recommendation:** ⚠️ Consider archiving or removing

#### `verify_octree.rs`
**Purpose:** Basic octree file verification by sampling voxels  
**Usage Likelihood:** Development/debugging tool, may still be useful  
**Recommendation:** ⚠️ Keep for now (debugging utility)

#### `reproduce_bbox.rs`
**Purpose:** Tests bounding box calculation with specific test cases  
**Usage Likelihood:** Unit test alternative, development tool  
**Recommendation:** ⚠️ Consider moving to proper unit tests, then remove

---

### 4. Commented-Out Code

#### SSAO Readback Buffer
**Location:** `src/bin/voxelot.rs:925`

```rust
// ssao_readback_buffer: Option<wgpu::Buffer>, // removed: readback temporarily disabled
```

**Recommendation:** ✅ Clean up comment and related commented initialization code (line 1444)

---

## Medium Confidence - Likely Unused

### 5. Light Probe System - 🚧 Planned Feature

**Structures:**
- `LightProbe` struct (line 192-196 in voxelot.rs)
- `ActiveLight` struct (line 166-170 in voxelot.rs)
- `light_probes: Vec<LightProbe>` field in App struct (line 820)

**Purpose:** Emissive voxels shining light on surrounding voxels (indirect lighting)

**Status:** ✅ **INFRASTRUCTURE IN PLACE** - Keep for future implementation
- Light probe buffer is created and bound to shaders
- Data structures exist for emissive indirect lighting
- `active_emitters` populated from chunk emitters (line 7301)
- Feature may not be fully implemented yet but is planned

**Recommendation:** ✅ Keep - this is a planned/in-progress feature, not dead code

---

### 6. SSAO System - ⚠️ Config Default Incorrect

**Config:** `default_ssao_enabled() -> bool { false }` (line 374-376)

**Status:** ✅ **ACTIVELY IN USE** - enabled in `config.toml`

**Issue:**
- Complete SSAO implementation exists and is functional
- Currently enabled in user's `config.toml`
- However, the code default is `false`, which is misleading

**Recommendation:** ✅ Update default to `true`:
```rust
fn default_ssao_enabled() -> bool {
    true  // SSAO is actively used
}
```

---

### 7. HZB (Hierarchical Z-Buffer) System - ⚠️ Config Default Incorrect

**Config:** `default_hzb_enabled() -> bool { false }` (line 448-450)

**Status:** ✅ **ACTIVELY IN USE** - enabled in `config.toml`

**Issue:**
- HZB occlusion culling is implemented and functional
- Currently enabled in user's `config.toml`
- However, the code default is `false`, which is misleading

**Recommendation:** ✅ Update default to `true`:
```rust
fn default_hzb_enabled() -> bool {
    true  // HZB occlusion culling is actively used
}
```

---

## Low Confidence - Diagnostic/Utility Tools

### 8. Development Binaries (Keep for Now)

These tools are likely still useful for world generation and debugging:

- ✅ `check_world_gaps.rs` - Validates tile boundaries in generated worlds
- ✅ `compress_worlds.rs` - World compression utility
- ✅ `convert_osm_format.rs` - OSM data conversion (if using OpenStreetMap data)
- ✅ `estimate_mesh_sizes.rs` - Performance profiling
- ✅ `generate_world.rs` - World generation tool
- ✅ `inspect_world.rs` - World inspection/debugging
- ✅ `test_render_coords.rs` - Coordinate system verification
- ✅ `world_stats.rs` - World statistics

**Recommendation:** Keep these utilities, they serve specific development purposes

---

### 9. DoF (Depth of Field) System

**Config:** Enabled by default but has Kawase implementation disabled

**Evidence:**
- `default_dof_kawase_enabled() -> bool { false }` (line 326-328)
- Main DoF is enabled by default (line 536)
- User is satisfied with graphical parts, so likely in use

**Recommendation:** ✅ Keep DoF system, potentially review Kawase variant

---

## Orphaned Files Check

### Files in shaders/ directory:
- ✅ `bloom_extract.wgsl` - USED (line 5249)
- ✅ `dof_coc_copy.wgsl` - USED (line 6003)
- ✅ `dof_combine.wgsl` - USED (line 5001)
- ✅ `dual_kawase_down.wgsl` - USED (line 5123)
- ✅ `dual_kawase_up.wgsl` - USED (line 5129)
- ✅ `gpu_cull.wgsl` - USED (line 5686)
- ✅ `hzb_gen.wgsl` - USED (line 5785) - actively used
- ✅ `post_composite.wgsl` - USED (line 5267)
- ✅ `skybox.wgsl` - USED (line 3866)
- ✅ `ssao_blur.wgsl` - USED (line 5261) - actively used
- ✅ `ssilvb.wgsl` - USED (line 5255)
- ✅ `ssr.wgsl` - USED (line 4116)
- ✅ `voxel.wgsl` - USED (line 4674)
- ✅ `water.wgsl` - USED (line 4012)

> [!NOTE]
> Bird and boid shaders previously removed. All remaining shaders are in active use.

---

## Summary

### Immediate Actions Required

#### ~~1. Remove Dead Code (High Priority)~~ ✅ COMPLETED
- ✅ ~~`fallback_bbox_shrink` config option~~ - **REMOVED**

#### ~~2. Fix Config Defaults (High Priority)~~ ✅ COMPLETED
~~Update these defaults to match actual usage:~~
- ✅ **FIXED:** `default_ssao_enabled()` now returns `true`
- ✅ **FIXED:** `default_hzb_enabled()` now returns `true`

#### 3. Clean Up Minor Items (Optional)
- Remove commented SSAO readback buffer code (line 925, 1444)

### Keep - Active Features
1. ✅ Light probe system - Planned for emissive voxel lighting
2. ✅ SSAO system - Actively used (just needs default config fix)
3. ✅ HZB system - Actively used (just needs default config fix)
4. ✅ All development/utility binaries in `src/bin/`
5. ✅ All shader files - All in active use

### Already Removed
- ✅ `shaders/bird.wgsl` - Previously removed
- ✅ `shaders/boid_simulation.wgsl` - Previously removed

---

## Estimated Impact

- **Lines of code removable:** ~100-200 (config option + commented code)
- **Config fixes needed:** 2 default values
- **No major system removals needed** - All systems are actively used

