use voxelot::bbox_local_to_world;

fn main() {
    println!("Testing Render Coordinate Calculations...");

    // Scenario: A chunk at Depth 1 (scale 16).
    // It contains a sub-chunk at (0,0,0) which is a leaf chunk (Depth 0, scale 1).
    // The leaf chunk has voxels at x=15.
    // So the leaf chunk's bbox should be [15, 0, 0, 15, 0, 0] (in its local 0-15 coords).

    // If we render the leaf chunk as an LOD instance (because it's far away):
    // scale = 1.
    // unit = scale / 16 = 1 / 16 = 0 (integer division!).
    // x_offset = bbox[0] * unit = 15 * 0 = 0.
    // The instance will be drawn at x=0 instead of x=15.

    // Let's verify this logic.

    let scale = 1;
    let bbox = [15u8, 0, 0, 15, 0, 0];
    let world_x = 100;
    let (pos_i64, _size) = bbox_local_to_world([world_x as i64, 0, 0], scale, bbox);
    let x = pos_i64[0] as f32;

    println!("Scale: {}", scale);
    println!("BBox: {:?}", bbox);
    println!("Calculated X: {}", x);
    println!("Expected X: {}", world_x as f32 + 15.0);

    if (x - (world_x as f32 + 15.0)).abs() > 0.001 {
        println!("FAIL: Coordinate calculation is wrong! ({} vs {})", x, world_x as f32 + 15.0);
    } else {
        println!("PASS: Coordinate calculation is correct.");
    }

    // Also check Depth 1 (scale 16)
    let scale_d1 = 16;
    let unit_d1 = scale_d1 / 16;
    let x_d1 = world_x + (bbox[0] as i64 * unit_d1);
    println!("\nScale: {}", scale_d1);
    println!("Unit (scale/16): {}", unit_d1);
    println!("Calculated X: {}", x_d1);
    println!("Expected X: {}", world_x + 15 * 1); // 15 blocks * 1 unit per block?
                                                  // Wait. If scale is 16, it means the chunk covers 16x16x16 world units.
                                                  // Inside it, there are 16 slots. Each slot is 1x1x1 world unit?
                                                  // No.
                                                  // Depth 0: Scale 1. Chunk size 16. Each voxel is 1x1x1? NO.
                                                  // Depth 0 is leaf. Voxel::Solid is a single point.
                                                  // A Chunk at Depth 0 covers 16x16x16 voxels.
                                                  // So scale passed to process_voxels for a Depth 0 chunk is... 1?

    // Let's trace scales:
    // Root (Depth 3): Scale 4096.
    // Child (Depth 2): Scale 256.
    // Child (Depth 1): Scale 16.
    // Child (Depth 0): Scale 1.

    // If we are at Depth 1 (Scale 16). We iterate slots.
    // Slot 0 is a Voxel::Chunk (Depth 0).
    // We recurse with scale = 16 / 16 = 1.

    // Inside collect_voxels_recursive for Depth 0 (scale 1):
    // We find a Voxel::Chunk? No, Depth 0 contains Voxel::Solid.
    // Wait, Voxel::Chunk can exist at any level?
    // "A hierarchical chunk ('chunks all the way')... At leaf level: Voxel::Solid"
    // So Depth 0 chunk contains Solids.

    // But wait, the code in culling.rs handles Voxel::Chunk.
    // If we are at Depth 1 (Scale 16). We see a Voxel::Chunk (the leaf chunk).
    // We calculate distance. If far, we render IT as an LOD.
    // We use ITS bounding box.
    // ITS bounding box is in range 0..15 (voxels).
    // We are at Scale 16.
    // The Voxel::Chunk represents a 16x16x16 volume.
    // But we are rendering it as a SINGLE box?
    // "if distance >= camera.config.lod_render_distance ... let (pos, size) = ..."

    // If we render a Depth 0 chunk as a single box.
    // The chunk is at `world_x`.
    // It has size 16.
    // `scale` passed to `process_voxels` is 16.
    // `unit` = 16 / 16 = 1.
    // `bbox` = [0, 0, 0, 15, 15, 15].
    // `x` = world_x + 0 * 1 = world_x.
    // `size_x` = (15 - 0 + 1) * 1 = 16.
    // This seems correct.

    // What if we are at Depth 2 (Scale 256).
    // We see a Voxel::Chunk (Depth 1).
    // `scale` is 256.
    // `unit` = 256 / 16 = 16.
    // `bbox` = [0..15].
    // `x` = world_x + 0 * 16.
    // `size` = 16 * 16 = 256.
    // Correct.

    // So when does integer division fail?
    // Only if `scale` < 16.
    // `scale` is powers of 16. 1, 16, 256...
    // If `scale` is 1.
    // This happens if we are processing a Depth 0 chunk (Leaf).
    // And we see a Voxel::Chunk inside it?
    // No, Depth 0 chunks contain Solids.
    // `process_voxels` handles `Voxel::Solid` separately.

    // Wait, does `process_voxels` handle `Voxel::Chunk` inside a Depth 0 chunk?
    // The definition says: "At leaf level: Voxel::Solid(type)".
    // So Depth 0 chunks do NOT contain sub-chunks.

    // So `scale` is always >= 16 when we encounter a `Voxel::Chunk`.
    // So `unit` is always >= 1.

    // So my hypothesis about integer division might be wrong IF scale is always >= 16.
    // Let's verify if scale can ever be 1 for a Voxel::Chunk.
    // Only if the hierarchy is deeper or mixed?
    // "Structure is uniform at all levels... At branch level: Voxel::Chunk".

    // Let's look at the "E-shape" again.
    // It happens at distance.
    // This means we are rendering Depth 1 or Depth 2 chunks as LODs.
    // Scale 16 or 256.

    // If Scale 16 (Depth 1).
    // We have a chunk of size 16x16x16 voxels.
    // It is composed of 16x16x16 Leaf Chunks?
    // No.
    // Depth 0 = Leaf Chunk (16^3 voxels). Size 16 units.
    // Depth 1 = Branch Chunk (16^3 sub-chunks). Size 256 units.
    // Depth 2 = Branch Chunk (16^3 sub-chunks). Size 4096 units.

    // If `scale` is 256 (Depth 1 processing).
    // We are looking at a child which is Depth 0 (Leaf Chunk).
    // Child size is 16.
    // `unit` = 256 / 16 = 16.
    // `bbox` of child (Depth 0) is in range 0..15 (voxels).
    // `x` = world_x + bbox[0] * 16.
    // `size` = (bbox[3] - bbox[0] + 1) * 16.

    // Example: Leaf chunk has voxels at x=0..15.
    // bbox = [0, 15].
    // size = 16 * 16 = 256.
    // This covers the whole leaf chunk. Correct.

    // Example: Leaf chunk has voxels ONLY at x=0.
    // bbox = [0, 0].
    // size = 1 * 16 = 16.
    // It draws a box of size 16 at offset 0.
    // This represents the x=0 slice of the leaf chunk.
    // But wait.
    // The leaf chunk has voxels at x=0.
    // These voxels are size 1.
    // The LOD box is size 16.
    // It covers 16x16x16 area.
    // But the voxels are only at x=0 inside that 16x16x16 area.
    // So the LOD box is 16x wider than the actual voxels!
    // It draws a huge block where there should be a thin slice.

    // THIS IS IT.
    // The bounding box of a Leaf Chunk is in *voxel* coordinates (0..15).
    // When we render it as an LOD instance, we scale it by `unit` (16).
    // So a single voxel (size 1) becomes a box of size 16.
    // This explains why it looks "blocky" or "fat".

    // But the user sees "E-shapes" (holes).
    // If we draw *too much*, we shouldn't see holes.
    // Unless...
    // If the bbox is [0, 0]. Size 16.
    // If the bbox is [15, 15]. Size 16.
    // If we have voxels at 0 and 15.
    // bbox is [0, 15]. Size 256.
    // It draws ONE big box covering everything.

    // So `Chunk::update_lod_metadata` merges everything into one bbox.
    // If we have a gap in the middle, the bbox covers it.
    // So we should see a solid block.

    // Why would we see an E-shape?
    // An E-shape implies we have:
    // [Block]
    // [Empty]
    // [Block]
    // [Empty]
    // [Block]

    // If these are separate chunks (e.g. separate Leaf Chunks in a vertical column).
    // And the "Empty" ones are NOT rendered.
    // Then we see gaps.

    // I fixed the `voxel_count` bug, so "Empty" ones (that actually have voxels) should now be rendered.
    // But the user says it's NOT fixed.

    // Maybe `unit` calculation is wrong for floating point?
    // `let unit = scale / 16;` is integer.
    // `let x = world_x + (bbox[0] as i64 * unit);`

    // Let's verify the "fat block" theory.
    // If I have a wall of thickness 1 voxel.
    // LOD renders it as thickness 16 voxels.
    // This is a known limitation of this LOD system (coarse approximation).
    // But it shouldn't cause *missing* geometry.

    // Let's write the test to confirm the integer math is doing what we expect.
}
