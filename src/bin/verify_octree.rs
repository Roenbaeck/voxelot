//! Verify octree file by sampling voxels

use std::env;
use std::fs::File;
use std::io::BufReader;
use voxelot::{load_world, WorldPos};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).map(String::as_str).unwrap_or("world_1.oct");
    
    println!("Verifying {}...\n", filename);
    
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);
    let world = load_world(&mut reader)?;
    
    println!("Loaded world with depth {}", world.hierarchy_depth());
    println!("World size: {} units per side", world.world_size());
    
    // Sample some positions from the original bounds
    println!("\nSampling voxels at known positions:");
    let test_positions = vec![
        (0, 0, 0),
        (100, 0, 100),
        (200, 10, 200),
        (300, 20, 300),
        (400, 30, 400),
        (500, 50, 500),
        (600, 80, 600),
    ];
    
    let mut found = 0;
    for (x, y, z) in test_positions {
        let pos = WorldPos::new(x, y, z);
        if let Some(vtype) = world.get(pos) {
            println!("  Position ({}, {}, {}): voxel type {}", x, y, z, vtype);
            found += 1;
        }
    }
    
    println!("\nFound {} voxels at test positions", found);
    
    // Try to count voxels in a specific region
    println!("\nScanning region [0..100, 0..50, 0..100]...");
    let mut region_count = 0;
    for x in 0..100 {
        for y in 0..50 {
            for z in 0..100 {
                if world.get(WorldPos::new(x, y, z)).is_some() {
                    region_count += 1;
                }
            }
        }
    }
    println!("Found {} voxels in scanned region", region_count);
    
    Ok(())
}
