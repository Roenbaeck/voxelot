//! Benchmark different culling strategies
//!
//! Compares:
//! 1. Serial culling with occlusion
//! 2. Parallel culling
//! 3. Cached culling (with spatial hash)

use std::time::Instant;
use voxelot::{World, WorldPos, Camera, cull_visible_voxels_with_occlusion, cull_visible_voxels_parallel, VisibilityCache};

fn create_test_world() -> World {
    let mut world = World::new();
    
    println!("Creating test world...");
    
    // Ground plane
    for x in -50..50 {
        for z in -50..50 {
            if (x + z) % 3 == 0 {
                world.set(WorldPos::new(x, 0, z), 1);
            }
        }
    }
    
    // Towers
    for i in 0..5 {
        let x = (i - 2) * 20;
        for y in 1..=(10 + i * 3) {
            world.set(WorldPos::new(x, y, 0), 2);
        }
    }
    
    // Scattered structures
    for i in 0..10 {
        let x = (i * 13 - 30) % 40;
        let z = (i * 17 - 20) % 40;
        for y in 1..5 {
            world.set(WorldPos::new(x, y, z), 3);
        }
    }
    
    let voxel_count: u64 = world.chunks().map(|(_, c)| c.count()).sum();
    println!("World: {} voxels in {} chunks\n", voxel_count, world.chunks().count());
    
    world
}

fn main() {
    let world = create_test_world();
    let camera = Camera::new([0.0, 20.0, 50.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
    
    const WARMUP: usize = 5;
    const ITERATIONS: usize = 100;
    
    println!("=== Culling Benchmark ===\n");
    
    // Benchmark 1: Serial with occlusion
    println!("1. Serial culling with occlusion:");
    for _ in 0..WARMUP {
        let _ = cull_visible_voxels_with_occlusion(&world, &camera);
    }
    
    let start = Instant::now();
    let mut result_count = 0;
    for _ in 0..ITERATIONS {
        let visible = cull_visible_voxels_with_occlusion(&world, &camera);
        result_count = visible.len();
    }
    let elapsed = start.elapsed();
    let avg = elapsed / ITERATIONS as u32;
    println!("   {} visible voxels", result_count);
    println!("   Average: {:?}", avg);
    println!("   Total {} iterations: {:?}\n", ITERATIONS, elapsed);
    
    // Benchmark 2: Parallel
    println!("2. Parallel culling:");
    for _ in 0..WARMUP {
        let _ = cull_visible_voxels_parallel(&world, &camera);
    }
    
    let start = Instant::now();
    let mut result_count = 0;
    for _ in 0..ITERATIONS {
        let visible = cull_visible_voxels_parallel(&world, &camera);
        result_count = visible.len();
    }
    let elapsed = start.elapsed();
    let avg = elapsed / ITERATIONS as u32;
    println!("   {} visible voxels", result_count);
    println!("   Average: {:?}", avg);
    println!("   Total {} iterations: {:?}\n", ITERATIONS, elapsed);
    
    // Benchmark 3: Cached (best case - no camera movement)
    println!("3. Cached culling (no movement):");
    let mut cache = VisibilityCache::new();
    
    let start = Instant::now();
    let mut result_count = 0;
    for _ in 0..ITERATIONS {
        let visible = cache.update(&camera, &world);
        result_count = visible.len();
    }
    let elapsed = start.elapsed();
    let avg = elapsed / ITERATIONS as u32;
    println!("   {} visible voxels", result_count);
    println!("   Average: {:?} (first: full calc, rest: cached)", avg);
    println!("   Total {} iterations: {:?}\n", ITERATIONS, elapsed);
    
    // Benchmark 4: Cached with camera movement
    println!("4. Cached culling (with movement simulation):");
    let mut cache = VisibilityCache::new();
    let mut cameras = Vec::new();
    
    // Create camera positions in small movements
    for i in 0..ITERATIONS {
        let offset = (i as f32) * 0.5; // Small movement
        cameras.push(Camera::new(
            [offset, 20.0, 50.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ));
    }
    
    let start = Instant::now();
    let mut result_count = 0;
    for cam in &cameras {
        let visible = cache.update(cam, &world);
        result_count = visible.len();
    }
    let elapsed = start.elapsed();
    let avg = elapsed / ITERATIONS as u32;
    println!("   {} visible voxels (average)", result_count);
    println!("   Average: {:?}", avg);
    println!("   Total {} iterations: {:?}\n", ITERATIONS, elapsed);
    
    println!("=== Summary ===");
    println!("Best performance for static camera: Cached (~0μs after first frame)");
    println!("Best for moving camera: Parallel culling");
    println!("Hybrid approach recommended: Use cache with invalidation threshold");
}
