//! Simple test program for the hierarchical voxel engine

use voxelot::{World, WorldPos, Camera, cull_visible_voxels, cull_visible_voxels_with_occlusion, get_visible_chunks};

fn main() {
    println!("Hierarchical Voxel Engine Test");
    println!("================================\n");
    
    // Create a world
    let mut world = World::new();
    
    // Add some test voxels
    println!("Creating test world...");
    
    // Create a simple structure - a platform
    for x in -5..=5 {
        for z in -5..=5 {
            world.set(WorldPos::new(x, 0, z), 1); // Ground
        }
    }
    
    // Add a tower
    for y in 1..=5 {
        world.set(WorldPos::new(0, y, 0), 2);
    }
    
    // Add some scattered blocks
    world.set(WorldPos::new(3, 1, 3), 3);
    world.set(WorldPos::new(-3, 1, -3), 4);
    world.set(WorldPos::new(3, 1, -3), 5);
    world.set(WorldPos::new(-3, 1, 3), 6);
    
    // Add a wall for occlusion testing
    for y in 0..=3 {
        for x in -2..=2 {
            world.set(WorldPos::new(x, y, -2), 7);
        }
    }
    
    // Count voxels
    let total_voxels: usize = world.chunks()
        .map(|(_, chunk)| chunk.count() as usize)
        .sum();
    
    println!("World created with {} voxels", total_voxels);
    println!("Number of chunks: {}\n", world.chunks().count());
    
    // Test retrieval
    println!("Testing voxel retrieval:");
    println!("  Voxel at (0, 0, 0): {:?}", world.get(WorldPos::new(0, 0, 0)));
    println!("  Voxel at (0, 3, 0): {:?}", world.get(WorldPos::new(0, 3, 0)));
    println!("  Voxel at (100, 100, 100): {:?}", world.get(WorldPos::new(100, 100, 100)));
    println!();
    
    // Test camera and culling
    println!("Testing camera culling:");
    let mut camera = Camera::new(
        [10.0, 5.0, 10.0],  // position
        [-1.0, -0.5, -1.0], // looking toward origin
        [0.0, 1.0, 0.0],    // up vector
    );
    
    println!("  Camera position: {:?}", camera.position);
    println!("  Camera forward: {:?}", camera.forward);
    println!();
    
    // Test basic culling
    let visible = cull_visible_voxels(&world, &camera);
    println!("  Basic culling - Visible voxels: {}", visible.len());
    
    if !visible.is_empty() {
        println!("    First 5 visible voxels:");
        for (i, instance) in visible.iter().take(5).enumerate() {
            println!("      {}. pos={:?}, type={}, dist={:.1}", 
                i + 1, instance.position, instance.voxel_type, instance.distance);
        }
    }
    println!();
    
    // Test occlusion culling
    let visible_occluded = cull_visible_voxels_with_occlusion(&world, &camera);
    println!("  With occlusion - Visible voxels: {}", visible_occluded.len());
    println!("    Occlusion reduced by: {} voxels ({:.1}%)", 
        visible.len() - visible_occluded.len(),
        (visible.len() - visible_occluded.len()) as f32 / visible.len() as f32 * 100.0);
    println!();
    
    // Test chunk-based culling with LOD
    let chunks = get_visible_chunks(&world, &camera);
    println!("  Visible chunks: {}", chunks.len());
    if !chunks.is_empty() {
        println!("    Chunks by LOD:");
        for info in chunks.iter().take(5) {
            println!("      Chunk {:?}: distance={:.1}, LOD={}", 
                info.chunk_pos, info.distance, info.lod_level);
        }
    }
    println!();
    
    // Test frustum culling by moving camera
    println!("Testing frustum culling:");
    camera.update(
        [0.0, 10.0, 0.0],   // above origin
        [0.0, -1.0, 0.0],   // looking down
        [0.0, 0.0, -1.0],   // up is -Z
    );
    
    let visible_from_above = cull_visible_voxels(&world, &camera);
    println!("  Looking down from above: {} voxels visible", visible_from_above.len());
    
    camera.update(
        [0.0, 0.0, 20.0],   // far on +Z
        [0.0, 0.0, -1.0],   // looking toward origin
        [0.0, 1.0, 0.0],    // up is Y
    );
    
    let visible_from_side = cull_visible_voxels(&world, &camera);
    println!("  Looking from +Z side: {} voxels visible", visible_from_side.len());
    println!();
    
    // Test chunk marginal culling
    println!("Testing chunk structure:");
    for ((cx, cy, cz), chunk) in world.chunks().take(3) {
        println!("  Chunk ({}, {}, {}):", cx, cy, cz);
        println!("    Voxels: {}", chunk.count());
        println!("    px: {:016b}", chunk.px);
        println!("    py: {:016b}", chunk.py);
        println!("    pz: {:016b}", chunk.pz);
    }
    
    println!("\n✓ All tests complete!");
}
