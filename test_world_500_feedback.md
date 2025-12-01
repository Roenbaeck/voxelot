Some observations, now that we run without overflowing buffers. 
1. I see a lot of mesh jobs start even though I see no new geometry on screen as I turn the camera. Are we scheduling meshing jobs for culled geometry? We should not. 

2. The generated world looks weird with respect to the water level, with submerged buildings on the "city" tiles. 

3. Some hills are cut, so if I look at them from one side they climb smoothly, but from the other they look sliced by a knife. 

4. There's still a horizontal plateau of blue voxels, with some islands (all submerged), which I believe is a "lake" biome. Note that there should be no lake biome. 

Note: Lakes and hills should not be separate biomes on top of the general terrain. The general terrain result in these naturally, thanks to the set water level. 

5. The camera_position puts the camera inside the world. Would it be possible to have a failsafe that moves the camera up above the world if it detects on startup that the camera is inside something?



Hierarchical Shell Culling - Implementation Plan
Executive Summary
Goal: Extend the existing shell voxel culling system from leaf chunks (16³) to all hierarchy levels, enabling rapid occlusion culling of entire super-chunks.

Current State: Shell culling only works for leaf chunks with unmeshed geometry

Proposed: Pre-compute shell voxels for every non-leaf chunk during world loading, enabling O(1) visibility tests at any hierarchy level

Expected Impact:

10-100x reduction in visibility testing cost for distant geometry
Eliminates meshing jobs for fully-obscured super-chunks
Minimal memory overhead (~100-500 bytes per non-leaf chunk)
Background: Current Shell Culling
Existing Implementation
Located in 
generate_shell()
:

pub struct ShellVoxel {
    packed_pos: u16,     // x | (y << 4) | (z << 8)
    visible_faces: u8,   // 6-bit face visibility mask
}
Face Visibility Bits:

Bit 0: +X (Right)
Bit 1: -X (Left)
Bit 2: +Y (Top)
Bit 3: -Y (Bottom)
Bit 4: +Z (Front)
Bit 5: -Z (Back)
Current Logic (leaf chunks only):

For each solid voxel, check 6 neighbors
If neighbor is air/empty OR at chunk boundary → set corresponding face bit
Store only voxels with visible_faces != 0 (surface voxels)
Runtime Culling (
voxelot.rs:6460-6520
):

Compute demand_mask based on camera direction (which faces to show)
Test each shell voxel: 
(visible_faces & demand_mask) != 0
Only render voxels with at least one visible face toward camera
Proposed: Hierarchical Shell Extension
Key Insight
A chunk at any hierarchy level is a 16×16×16 grid of sub-chunks. We can apply the exact same logic:

Each "cell" is a sub-chunk instead of a voxel
"Air" means empty sub-chunk (voxel_count == 0)
"Solid" means occupied sub-chunk (voxel_count > 0)
Data Structure Extension
pub struct Chunk {
    // ... existing fields ...
    
    /// Shell for this chunk level (None for leaf chunks)
    /// Each entry represents a sub-chunk with at least one exposed face
    pub hierarchy_shell: Option<Vec<ShellVoxel>>,
}
Memory Cost:

Leaf chunks: None (0 bytes)
Non-leaf chunks: ~10-200 entries × 3 bytes = 30-600 bytes average
For 1M non-leaf chunks: ~100-300 MB (acceptable)
Generation Algorithm
Timing: After 
update_all_lod_metadata()
 in world loading (one-time cost)

Pseudocode:

impl Chunk {
    pub fn generate_hierarchy_shell(&mut self) {
        if self.is_leaf_chunk() {
            self.hierarchy_shell = None; // Use existing shell logic
            return;
        }
        
        let mut shell = Vec::with_capacity(256);
        
        for ((x, y, z), voxel) in self.iter() {
            // Only consider non-empty sub-chunks
            let is_occupied = match voxel {
                Voxel::Solid(_) => true,
                Voxel::Chunk(c) => c.voxel_count > 0,
            };
            
            if !is_occupied {
                continue;
            }
            
            let mut mask = 0u8;
            
            // Check 6 neighbor sub-chunks
            if x == 15 || !self.is_occupied_at(x + 1, y, z) {
                mask |= 1 << 0; // +X exposed
            }
            if x == 0 || !self.is_occupied_at(x - 1, y, z) {
                mask |= 1 << 1; // -X exposed
            }
            // ... repeat for Y and Z ...
            
            if mask != 0 {
                shell.push(ShellVoxel {
                    packed_pos: (x as u16) | ((y as u16) << 4) | ((z as u16) << 8),
                    visible_faces: mask,
                });
            }
        }
        
        self.hierarchy_shell = Some(shell);
    }
    
    fn is_occupied_at(&self, x: u8, y: u8, z: u8) -> bool {
        match self.get(x, y, z) {
            Some(Voxel::Solid(_)) => true,
            Some(Voxel::Chunk(c)) => c.voxel_count > 0,
            None => false,
        }
    }
    
    fn is_leaf_chunk(&self) -> bool {
        // A chunk is a leaf if it contains no sub-chunks
        self.voxels.iter().all(|v| matches!(v, Voxel::Solid(_)))
    }
}
Integration into Culling System
Modification Point: 
process_voxels

Current Logic:

match voxel {
    Voxel::Solid(voxel_type) => {
        // Render individual voxel
    }
    Voxel::Chunk(sub_chunk) => {
        if distance >= lod_render_distance && sub_chunk.voxel_count > 0 {
            // Render entire sub-chunk as single LOD voxel
        } else {
            // Recurse into sub-chunk
        }
    }
}
Proposed Enhancement:

match voxel {
    Voxel::Solid(voxel_type) => {
        // Render individual voxel (unchanged)
    }
    Voxel::Chunk(sub_chunk) => {
        if distance >= lod_render_distance && sub_chunk.voxel_count > 0 {
            // --- NEW: Shell culling for LOD sub-chunks ---
            if let Some(shell) = &sub_chunk.hierarchy_shell {
                // Compute demand mask based on camera direction
                let dx = voxel_center[0] - camera.position[0];
                let dy = voxel_center[1] - camera.position[1];
                let dz = voxel_center[2] - camera.position[2];
                
                let mut demand_mask = 0u8;
                if dx > 0.0 { demand_mask |= 1 << 0; } else { demand_mask |= 1 << 1; }
                if dy > 0.0 { demand_mask |= 1 << 2; } else { demand_mask |= 1 << 3; }
                if dz > 0.0 { demand_mask |= 1 << 4; } else { demand_mask |= 1 << 5; }
                
                // Check if ANY shell voxel has a visible face
                let has_visible_faces = shell.iter()
                    .any(|sv| (sv.visible_faces & demand_mask) != 0);
                
                if !has_visible_faces {
                    return; // FULLY OBSCURED - skip this sub-chunk entirely!
                }
            }
            
            // Render entire sub-chunk as single LOD voxel (if visible)
        } else {
            // Recurse into sub-chunk for detail
        }
    }
}
Implementation Steps
Phase 1: Data Structure (1-2 hours)
✅ Add hierarchy_shell: Option<Vec<ShellVoxel>> to 
Chunk
 struct
✅ Add is_leaf_chunk() and is_occupied_at() helper methods
✅ Add generate_hierarchy_shell() method
Phase 2: Generation (1-2 hours)
✅ Implement shell generation algorithm for non-leaf chunks
✅ Add generate_all_hierarchy_shells() to 
World
✅ Call after 
update_all_lod_metadata()
 in world loading
Phase 3: Integration (2-3 hours)
✅ Modify 
process_voxels
 in 
culling.rs
 to use hierarchy shell
✅ Add demand mask computation for sub-chunks
✅ Add visibility test against shell before rendering LOD voxels
Phase 4: Meshing Optimization (1 hour)
✅ Apply same shell culling logic to meshing loop in 
voxelot.rs
✅ Skip meshing jobs for fully-obscured chunks at any level
Phase 5: Testing & Tuning (2-3 hours)
✅ Verify correctness with debug visualization
✅ Measure performance improvement (FPS, meshing job count)
✅ Tune shell generation parameters if needed
Performance Analysis
Expected Gains
Scenario: Camera facing a dense city with mountains behind

Current Behavior:

Process all visible chunks in frustum
Generate mesh jobs for all near chunks
Many chunks fully obscured but still processed
With Hierarchical Shell:

Test super-chunks first (16×16×16 sub-chunks at once)
Skip entire regions if shell shows no visible faces
Only descend into hierarchy for potentially visible areas
Estimated Speedup:

Visibility testing: 10-50x faster for distant geometry
Meshing: 50-90% reduction in unnecessary jobs
Overall FPS: 20-40% improvement in complex scenes
Memory Overhead
Per Non-Leaf Chunk:

Shell entries: ~10-200 (depends on occlusion)
Entry size: 3 bytes (packed_pos + visible_faces)
Total: ~30-600 bytes average, ~150 bytes median
For Large World (1M non-leaf chunks):

150 MB median
300 MB worst case
Acceptable given modern GPU/RAM (4-16 GB typical)
Computation Cost
One-time Cost (world loading):

Per chunk: ~100-500 iterations (16³ loop)
Simple neighbor checks (6 per position)
Parallel-friendly (independent per chunk)
Estimate: +2-5 seconds for 100K non-leaf chunks
Runtime Cost: O(1) per chunk (simple bitmask test)

Risks & Mitigations
Risk	Impact	Mitigation
High memory use for dense worlds	Medium	Lazy generation on-demand, eviction policy
Shell invalidation on world changes	Low	Mark dirty, regenerate on next frame
False positives (render hidden)	Low	Better than false negatives, acceptable
Implementation complexity	Medium	Reuse existing shell logic, well-tested pattern
Alternative Approaches Considered
1. Occlusion Queries (GPU-based)
❌ Rejected: High latency, GPU roundtrip overhead, complex state management

2. Portal/Cell Visibility (PVS)
❌ Rejected: Requires preprocessing, not suitable for procedural/dynamic worlds

3. Octree-based Culling
✅ Already have this: Hierarchy IS an octree (16-ary), shells enhance it

4. HZB (Hierarchical Z-Buffer)
✅ Complementary: Already implemented, shells work at higher level

Success Criteria
✅ Shell generation completes within 5 seconds for typical worlds
✅ Memory overhead < 500 MB for large worlds
✅ No visual artifacts (false negatives)
✅ FPS improvement > 15% in complex scenes
✅ Meshing job reduction > 40%
Open Questions
When to regenerate shells?

World load: always
Runtime edits: mark dirty, lazy regen
Streaming: generate on chunk load
Shell granularity?

Always 16×16×16 grid? ✅
Adaptive? (too complex)
Should we cache shells in .oct file?

Probably not: fast to compute, world changes
Consider if generation time > 10 seconds
Next Step
Create implementation in phases, starting with Phase 1 data structure changes.