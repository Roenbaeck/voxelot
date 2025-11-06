// Profile where time is spent
use std::time::Instant;
use voxelot::{World, WorldPos, Camera};

fn main() {
    let mut world = World::new(2); // 256 units per side
    
    // Create a simple world
    for i in 0..10 {
        for j in 0..10 {
            world.set(WorldPos::new(i, 0, j), 1);
        }
    }
    
    let root = world.root();
    println!("World root chunk has {} voxels", root.count());
    
    let camera = Camera::new([5.0, 10.0, 20.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
    
    // Time just accessing the root
    let start = Instant::now();
    let _root = world.root();
    println!("Root access: {:?}", start.elapsed());
    
    // Time the full culling
    let start = Instant::now();
    let visible = voxelot::cull_visible_voxels(&world, &camera);
    let duration = start.elapsed();
    
    println!("Full culling: {:?} ({} visible voxels)", duration, visible.len());
    println!("Per-voxel cost: {:?}", duration / visible.len() as u32);
}
