use std::path::PathBuf;
// Removed unused imports: File, BufWriter were unused

use voxelot::{file_format::save_world_file, load_world_file, Palette, World, WorldPos};

fn calculate_required_depth(max_coord: i64) -> u8 {
    let mut depth = 0u8;
    let mut size = 1u64;
    while size <= (max_coord as u64) {
        size *= 16;
        depth += 1;
    }
    depth.max(1)
}

fn main() {
    // Parameters
    let grid = 1024usize;
    let max_height_voxels = 200i64;
    let depth = calculate_required_depth((grid - 1) as i64);
    println!(
        "Generating linear hill test world: {}x{} height={} depth={}",
        grid, grid, max_height_voxels, depth
    );

    let mut world = World::new(depth);
    // simple linear hill: height increases from left to right (x axis)
    for x in 0..grid {
        for z in 0..grid {
            let t = x as f64 / (grid as f64 - 1.0);
            let h = (t * (max_height_voxels as f64)).round() as i64;
            for y in 0..=h {
                world.set(WorldPos::new(x as i64, y, z as i64), 1u8);
            }
        }
    }

    // Save palette (optional) and world
    let palette = Palette::load("worlds/palette.txt");
    let oct_path = PathBuf::from("test_linear_hill.vhc");
    world.update_all_lod_metadata(&palette);
    println!("Saving world to {}...", oct_path.display());
    match save_world_file(&world, &oct_path, true) {
        Ok(_) => println!("Saved {}", oct_path.display()),
        Err(e) => eprintln!("Failed to save world: {}", e),
    }

    // Load back the saved world and perform a basic continuity check across X boundaries
    println!("Loading world back for verification...");
    match load_world_file(&oct_path) {
        Ok(world2) => {
            let mut discontinuities = 0usize;
            let mut bigjumps = 0usize;
            for z in 0..grid {
                for x in 0..(grid - 1) {
                    let mut h_a = -1i64;
                    let mut h_b = -1i64;
                    for y in (0..=max_height_voxels).rev() {
                        if h_a < 0 {
                            if world2.get(WorldPos::new(x as i64, y, z as i64)).is_some() {
                                h_a = y;
                            }
                        }
                        if h_b < 0 {
                            if world2
                                .get(WorldPos::new((x + 1) as i64, y, z as i64))
                                .is_some()
                            {
                                h_b = y;
                            }
                        }
                        if h_a >= 0 && h_b >= 0 {
                            break;
                        }
                    }
                    let diff = (h_a - h_b).abs();
                    if diff > 1 {
                        discontinuities += 1;
                        if diff > 10 {
                            bigjumps += 1;
                            if bigjumps < 10 {
                                println!("Big gap at x={}, z={} diff={}", x, z, diff);
                            }
                        }
                    }
                }
            }
            println!(
                "Checked {} columns; discontinuities >1: {} (big jumps >10: {})",
                grid * (grid - 1),
                discontinuities,
                bigjumps
            );
        }
        Err(e) => eprintln!("Failed to load world for verification: {}", e),
    }
}
