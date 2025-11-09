//! Simple test program for the hierarchical voxel engine

use voxelot::{
    cull_visible_voxels, cull_visible_voxels_with_occlusion, get_visible_chunks, Camera, World,
    WorldPos,
};

fn main() {
    println!("Hierarchical Voxel Engine Test");
    println!("================================\n");

    // Create a world (depth 3 = 4,096 units per side)
    let mut world = World::new(3);
    println!("World size: {} units per side\n", world.world_size());

    // Add some test voxels
    println!("Creating test world...");

    // Create a simple structure - a platform
    for x in 0..=10 {
        for z in 0..=10 {
            world.set(WorldPos::new(x, 0, z), 1); // Ground
        }
    }

    // Add a tower
    for y in 1..=5 {
        world.set(WorldPos::new(5, y, 5), 2);
    }

    // Add some scattered blocks
    world.set(WorldPos::new(8, 1, 8), 3);
    world.set(WorldPos::new(2, 1, 2), 4);
    world.set(WorldPos::new(8, 1, 2), 5);
    world.set(WorldPos::new(2, 1, 8), 6);

    // Add a wall for occlusion testing
    for y in 0..=3 {
        for x in 3..=7 {
            world.set(WorldPos::new(x, y, 3), 7);
        }
    }

    // Count voxels
    let total_voxels = world.root().count();

    println!("World created with {} voxels in root chunk\n", total_voxels);

    // Test retrieval
    println!("Testing voxel retrieval:");
    println!(
        "  Voxel at (0, 0, 0): {:?}",
        world.get(WorldPos::new(0, 0, 0))
    );
    println!(
        "  Voxel at (0, 3, 0): {:?}",
        world.get(WorldPos::new(0, 3, 0))
    );
    println!(
        "  Voxel at (100, 100, 100): {:?}",
        world.get(WorldPos::new(100, 100, 100))
    );
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
            println!(
                "      {}. pos={:?}, type={}, dist={:.1}",
                i + 1,
                instance.position,
                instance.voxel_type,
                instance.distance
            );
        }
    }
    println!();

    // Test occlusion culling
    let visible_occluded = cull_visible_voxels_with_occlusion(&world, &camera);
    println!(
        "  With occlusion - Visible voxels: {}",
        visible_occluded.len()
    );
    println!(
        "    Occlusion reduced by: {} voxels ({:.1}%)",
        visible.len() - visible_occluded.len(),
        (visible.len() - visible_occluded.len()) as f32 / visible.len() as f32 * 100.0
    );
    println!();

    // Test chunk-based culling with LOD
    let chunks = get_visible_chunks(&world, &camera);
    println!("  Visible chunks: {}", chunks.len());
    if !chunks.is_empty() {
        println!("    Chunks by LOD:");
        for info in chunks.iter().take(5) {
            println!(
                "      Chunk {:?}: distance={:.1}, LOD={}",
                info.chunk_pos, info.distance, info.lod_level
            );
        }
    }
    println!();

    // Test frustum culling by moving camera
    println!("Testing frustum culling:");
    camera.update(
        [0.0, 10.0, 0.0], // above origin
        [0.0, -1.0, 0.0], // looking down
        [0.0, 0.0, -1.0], // up is -Z
    );

    let visible_from_above = cull_visible_voxels(&world, &camera);
    println!(
        "  Looking down from above: {} voxels visible",
        visible_from_above.len()
    );

    camera.update(
        [0.0, 0.0, 20.0], // far on +Z
        [0.0, 0.0, -1.0], // looking toward origin
        [0.0, 1.0, 0.0],  // up is Y
    );

    let visible_from_side = cull_visible_voxels(&world, &camera);
    println!(
        "  Looking from +Z side: {} voxels visible",
        visible_from_side.len()
    );
    println!();

    // Test root chunk structure
    println!("Testing root chunk structure:");
    let root = world.root();
    println!("  Root chunk:");
    println!("    Voxels: {}", root.count());
    println!("    px: {:016b}", root.px);
    println!("    py: {:016b}", root.py);
    println!("    pz: {:016b}", root.pz);

    println!("\n✓ All tests complete!");
}
