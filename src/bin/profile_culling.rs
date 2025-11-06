// Profile where time is spent
use std::time::Instant;
use voxelot::{World, WorldPos, Camera};

fn main() {
    let mut world = World::new();
    
    // Create a simple world
    for i in 0..10 {
        for j in 0..10 {
            world.set(WorldPos::new(i, 0, j), 1);
        }
    }
    
    println!("World has {} chunks", world.chunks().count());
    for (pos, chunk) in world.chunks() {
        println!("  Chunk {:?}: {} voxels", pos, chunk.count());
    }
    
    let camera = Camera::new([5.0, 10.0, 20.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
    
    // Time just the chunk finding
    let start = Instant::now();
    let cam_chunk_x = (camera.position[0] / 16.0).floor() as i64;
    let cam_chunk_y = (camera.position[1] / 16.0).floor() as i64;
    let cam_chunk_z = (camera.position[2] / 16.0).floor() as i64;
    let radius = (camera.far / 16.0).ceil() as i64 + 1;
    
    println!("Camera chunk: ({}, {}, {}), radius: {}", cam_chunk_x, cam_chunk_y, cam_chunk_z, radius);
    println!("Checking ({})^3 = {} chunk positions", radius*2+1, (radius*2+1).pow(3));
    
    let mut chunk_count = 0;
    let mut check_count = 0;
    for cx in (cam_chunk_x - radius)..=(cam_chunk_x + radius) {
        for cy in (cam_chunk_y - radius)..=(cam_chunk_y + radius) {
            for cz in (cam_chunk_z - radius)..=(cam_chunk_z + radius) {
                check_count += 1;
                if let Some(_chunk) = world.get_chunk((cx, cy, cz)) {
                    chunk_count += 1;
                }
            }
        }
    }
    println!("Chunk search: {:?} ({} chunks found, {} positions checked)", start.elapsed(), chunk_count, check_count);
    
    // Time the full culling
    let start = Instant::now();
    let visible = voxelot::cull_visible_voxels(&world, &camera);
    let duration = start.elapsed();
    
    println!("Full culling: {:?} ({} visible voxels)", duration, visible.len());
    println!("Per-voxel cost: {:?}", duration / visible.len() as u32);
}
