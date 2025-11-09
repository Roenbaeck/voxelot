// Test that projection bits propagate correctly in hierarchical chunks

use voxelot::{World, WorldPos};

fn main() {
    println!("=== Testing Projection Bit Propagation ===\n");

    // Create a depth-2 world (256 units per side)
    let mut world = World::new(2);

    // Create a single voxel at origin
    world.set(WorldPos::new(0, 0, 0), 1);

    println!("1. Initial state - single solid voxel at (0,0,0)");
    let root = world.root();
    println!("   px: {:016b} (bit 0 should be set)", root.px);
    println!("   py: {:016b} (bit 0 should be set)", root.py);
    println!("   pz: {:016b} (bit 0 should be set)", root.pz);
    println!("   Expected: only bit 0 set in each");

    // Subdivide it
    world
        .subdivide_at(WorldPos::new(0, 0, 0))
        .expect("Should subdivide");

    println!("\n2. After subdivision - voxel contains 16³ sub-voxels");
    let root = world.root();
    println!("   px: {:016b} (all 16 bits should be set)", root.px);
    println!("   py: {:016b} (all 16 bits should be set)", root.py);
    println!("   pz: {:016b} (all 16 bits should be set)", root.pz);
    println!("   Expected: 0xFFFF (1111111111111111) in each");

    if root.px == 0xFFFF && root.py == 0xFFFF && root.pz == 0xFFFF {
        println!("   ✓ Projection bits correctly propagated!");
    } else {
        println!("   ✗ ERROR: Projection bits not propagated correctly");
    }

    // Merge it back
    world
        .merge_at(WorldPos::new(0, 0, 0))
        .expect("Should merge");

    println!("\n3. After merging back to solid voxel");
    let root = world.root();
    println!("   px: {:016b} (bit 0 should be set)", root.px);
    println!("   py: {:016b} (bit 0 should be set)", root.py);
    println!("   pz: {:016b} (bit 0 should be set)", root.pz);
    println!("   Expected: only bit 0 set in each (after merge)");

    println!("\n=== Test Complete ===");
}
