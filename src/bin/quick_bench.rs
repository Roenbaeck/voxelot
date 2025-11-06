// Quick performance test
use std::time::Instant;
use voxelot::{World, WorldPos, Camera, cull_visible_voxels};

fn main() {
    let mut world = World::new(2); // 256 units per side
    
    // Create a simple world with 100 voxels
    for i in 0..10 {
        for j in 0..10 {
            world.set(WorldPos::new(i, 0, j), 1);
        }
    }
    
    let camera = Camera::new([5.0, 10.0, 20.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
    
    // Warmup
    for _ in 0..10 {
        let _ = cull_visible_voxels(&world, &camera);
    }
    
    // Benchmark
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cull_visible_voxels(&world, &camera);
    }
    let elapsed = start.elapsed();
    
    println!("100 voxels, {} iterations:", iterations);
    println!("Total: {:?}", elapsed);
    println!("Per iteration: {:?}", elapsed / iterations);
    println!("FPS equivalent: {:.1}", 1.0 / (elapsed.as_secs_f64() / iterations as f64));
}
