//! Convert osm_voxels.txt to compact octree format
//! Uses the same loading logic as the viewer to ensure compatibility

use std::path::Path;
use voxelot::{save_world_file, load_world_file, World, WorldPos};

fn main() -> std::io::Result<()> {
    println!("OSM Voxel Format Converter");
    println!("==========================\n");

    // Load using the same logic as the viewer (guaranteed to work!)
    println!("Loading osm_voxels.txt...");
    
    let content = std::fs::read_to_string("osm_voxels.txt")?;
    
    // First pass: determine bounds
    let mut min_x = i64::MAX;
    let mut max_x = i64::MIN;
    let mut min_y = i64::MAX;
    let mut max_y = i64::MIN;
    let mut min_z = i64::MAX;
    let mut max_z = i64::MIN;
    let mut voxel_count = 0;

    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 4 {
            if let (Ok(x), Ok(y), Ok(z), Ok(_voxel_type)) = (
                parts[0].parse::<i64>(),
                parts[1].parse::<i64>(),
                parts[2].parse::<i64>(),
                parts[3].parse::<u8>(),
            ) {
                voxel_count += 1;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                min_z = min_z.min(z);
                max_z = max_z.max(z);
            }
        }
    }

    println!("Found {} voxels", voxel_count);
    println!("Bounds: X[{}, {}], Y[{}, {}], Z[{}, {}]", min_x, max_x, min_y, max_y, min_z, max_z);

    // Calculate required depth
    let coords = vec![min_x, max_x, min_y, max_y, min_z, max_z];
    let max_coord = coords.iter().map(|&c| c.abs()).max().unwrap();
    let depth = ((max_coord as f64).log2() / 4.0).ceil() as u8 + 1;

    println!("Using depth {} (world size: {}³)", depth, 16i64.pow(depth as u32));

    // Create world - exactly as viewer does
    let mut world = World::new(depth);
    
    // Second pass: load voxels - exactly as viewer does
    println!("Loading voxels into world...");
    let mut loaded = 0;
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 4 {
            if let (Ok(x), Ok(y), Ok(z), Ok(voxel_type)) = (
                parts[0].parse::<i64>(),
                parts[1].parse::<i64>(),
                parts[2].parse::<i64>(),
                parts[3].parse::<u8>(),
            ) {
                world.set(WorldPos::new(x, y, z), voxel_type);
                loaded += 1;
            }
        }
    }
    
    println!("Loaded {} voxels into world", loaded);
    println!("World.count() reports {} voxels", world.count());

    // Save in octree format
    let output_file = "world_1.oct";
    println!("\nSaving to {}...", output_file);
    // Save compressed .oct with gzip
    save_world_file(&world, Path::new(output_file), true)?;

    // Get file sizes
    let old_size = std::fs::metadata("osm_voxels.txt")?.len();
    let new_size = std::fs::metadata(output_file)?.len();
    let ratio = old_size as f64 / new_size as f64;

    println!("\nConversion complete!");
    println!("Old size: {} bytes ({:.1} MB)", old_size, old_size as f64 / 1_000_000.0);
    println!("New size: {} bytes ({:.1} KB)", new_size, new_size as f64 / 1_000.0);
    println!("Compression ratio: {:.2}x", ratio);
    println!("Space saved: {:.1}%", (1.0 - 1.0 / ratio) * 100.0);

    // Verify by loading using the auto-detecting loader
    println!("\nVerifying by loading...");
    let loaded_world = load_world_file(Path::new(output_file))?;
    println!("Successfully loaded world with depth {}", loaded_world.hierarchy_depth());
    println!("Loaded world has {} voxels", loaded_world.count());
    
    if loaded_world.count() == world.count() {
        println!("✓ Voxel count matches!");
    } else {
        println!("✗ WARNING: Voxel count mismatch! Original: {}, Loaded: {}", 
                 world.count(), loaded_world.count());
    }

    Ok(())
}
