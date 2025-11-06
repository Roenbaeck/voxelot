// Test that projection bits propagate correctly in hierarchical chunks

use voxelot::{World, WorldPos};

fn main() {
    println!("=== Testing Projection Bit Propagation ===\n");
    
    let mut world = World::new();
    
    // Create a single voxel at origin
    world.set(WorldPos::new(0, 0, 0), 1);
    
    println!("1. Initial state - single solid voxel at (0,0,0)");
    if let Some(chunk) = world.get_chunk((0, 0, 0)) {
        println!("   px: {:016b} (bit 0 should be set)", chunk.px);
        println!("   py: {:016b} (bit 0 should be set)", chunk.py);
        println!("   pz: {:016b} (bit 0 should be set)", chunk.pz);
        println!("   Expected: only bit 0 set in each");
    }
    
    // Subdivide it
    world.subdivide_at(WorldPos::new(0, 0, 0)).expect("Should subdivide");
    
    println!("\n2. After subdivision - voxel contains 16³ sub-voxels");
    if let Some(chunk) = world.get_chunk((0, 0, 0)) {
        println!("   px: {:016b} (all 16 bits should be set)", chunk.px);
        println!("   py: {:016b} (all 16 bits should be set)", chunk.py);
        println!("   pz: {:016b} (all 16 bits should be set)", chunk.pz);
        println!("   Expected: 0xFFFF (1111111111111111) in each");
        
        if chunk.px == 0xFFFF && chunk.py == 0xFFFF && chunk.pz == 0xFFFF {
            println!("   ✓ Projection bits correctly propagated!");
        } else {
            println!("   ✗ ERROR: Projection bits not propagated correctly");
        }
    }
    
    // Merge it back
    world.merge_at(WorldPos::new(0, 0, 0)).expect("Should merge");
    
    println!("\n3. After merging back to solid voxel");
    if let Some(chunk) = world.get_chunk((0, 0, 0)) {
        println!("   px: {:016b} (bit 0 should be set)", chunk.px);
        println!("   py: {:016b} (bit 0 should be set)", chunk.py);
        println!("   pz: {:016b} (bit 0 should be set)", chunk.pz);
        println!("   Expected: only bit 0 set in each (after merge)");
    }
    
    println!("\n=== Test Complete ===");
}
