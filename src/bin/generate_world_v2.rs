//! Simplified world generator - generates terrain as a single continuous heightmap,
//! then populates with features (buildings, trees, etc.)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use noise::{NoiseFn, Perlin};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

use voxelot::{octree_format::save_world_file, Palette, World, WorldPos};

#[derive(Parser, Debug)]
#[command(author, version, about = "Simplified voxel world generator")]
struct Args {
    /// World size in voxels (width and depth)
    #[arg(long, default_value_t = 512)]
    size: u32,

    /// Random seed
    #[arg(long, default_value_t = 1337)]
    seed: u64,

    /// Water level in meters (should match config.toml water_level)
    #[arg(long, default_value_t = 500.0)]
    water_level: f64,

    /// Meters per voxel (vertical scale)
    #[arg(long, default_value_t = 1.25)]
    meters_per_voxel: f64,

    /// Terrain scale (higher = more spread out hills)
    #[arg(long, default_value_t = 200.0)]
    terrain_scale: f64,

    /// Terrain amplitude in meters (height variation)
    #[arg(long, default_value_t = 300.0)]
    terrain_amplitude: f64,

    /// Output file name (without extension)
    #[arg(long = "output-name", default_value = "world_1")]
    output_name: String,

    /// Output format
    #[arg(long, value_enum, default_value_t = OutputFormat::Oct)]
    format: OutputFormat,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum OutputFormat {
    Txt,
    Oct,
    Both,
}

// Material indices (0-based, will be +1 when written)
const MAT_STONE: u8 = 0;
const MAT_DIRT: u8 = 1;
const MAT_GRASS: u8 = 2;
const MAT_SAND: u8 = 3;
// MAT_WATER = 4 - NOT USED, water is rendered by the viewer
const MAT_WOOD: u8 = 5;
const MAT_LEAVES: u8 = 6;
const MAT_CONCRETE: u8 = 7;
const MAT_BRICK: u8 = 8;
const MAT_GLASS: u8 = 9;
const MAT_ROOF: u8 = 10;
const MAT_GRAVEL: u8 = 11;

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simplified World Generator v2 ===");
    println!("Size: {}x{} voxels", args.size, args.size);
    println!("Water level: {} meters", args.water_level);
    println!("Meters per voxel: {}", args.meters_per_voxel);
    println!("Terrain amplitude: {} meters", args.terrain_amplitude);
    
    let perlin = Perlin::new(args.seed as u32);
    let mut rng = StdRng::seed_from_u64(args.seed);
    
    let size = args.size as usize;
    let water_level_vox = (args.water_level / args.meters_per_voxel) as i64;
    
    println!("Water level in voxels: {}", water_level_vox);
    
    // Step 1: Generate continuous heightmap for entire world
    // Heights are in VOXELS, centered around water_level_vox
    println!("Generating heightmap...");
    let heightmap = generate_heightmap(
        &perlin, 
        size, 
        water_level_vox,
        args.terrain_scale, 
        args.terrain_amplitude / args.meters_per_voxel,
    );
    
    // Find height statistics
    let min_h = heightmap.iter().copied().min().unwrap_or(0);
    let max_h = heightmap.iter().copied().max().unwrap_or(0);
    let avg_h: i64 = heightmap.iter().copied().sum::<i64>() / heightmap.len() as i64;
    println!("Height range: {} to {} voxels (avg: {})", min_h, max_h, avg_h);
    println!("Heights above water: {} to {} voxels", 
             min_h - water_level_vox, max_h - water_level_vox);
    
    // Step 2: Convert heightmap to voxels (TERRAIN ONLY - no water voxels!)
    println!("Generating terrain voxels...");
    let mut voxels: HashMap<(i64, i64, i64), u8> = HashMap::new();
    
    // We only need to fill from some minimum Y up to the terrain surface
    // The minimum Y should be below the lowest terrain point
    let base_y = (min_h - 10).max(0);
    
    for x in 0..size {
        for z in 0..size {
            let ground_height = heightmap[x + z * size];
            
            // Determine surface material based on height relative to water
            let height_above_water = ground_height - water_level_vox;
            
            // Fill terrain column from base to surface
            for y in base_y..=ground_height {
                let material = if y == ground_height {
                    // Surface layer
                    if height_above_water < -10 {
                        // Deep underwater - sand/gravel seabed
                        MAT_SAND
                    } else if height_above_water < 5 {
                        // Near/below water level - sand (beach/shallows)
                        MAT_SAND
                    } else if height_above_water < 80 {
                        // Normal elevation - grass
                        MAT_GRASS
                    } else {
                        // High elevation - rocky
                        MAT_STONE
                    }
                } else if y > ground_height - 4 {
                    // Top layers under surface - dirt
                    MAT_DIRT
                } else {
                    // Deep underground - stone
                    MAT_STONE
                };
                
                voxels.insert((x as i64, y, z as i64), material);
            }
        }
    }
    
    println!("Base terrain: {} voxels", voxels.len());
    
    // Step 3: Add features (trees, buildings)
    println!("Adding features...");
    
    // Find suitable locations for features
    let mut tree_positions: Vec<(usize, usize)> = Vec::new();
    let mut building_positions: Vec<(usize, usize, usize, usize)> = Vec::new(); // x, z, width, depth
    
    // Place trees on grass areas
    for x in (5..size-5).step_by(8) {
        for z in (5..size-5).step_by(8) {
            let ground_height = heightmap[x + z * size];
            let height_above_water = ground_height - water_level_vox;
            
            // Trees on grass (above water, not too high)
            if height_above_water > 5 && height_above_water < 60 {
                if rng.gen_bool(0.4) {
                    // Jitter position slightly
                    let jx = x as i64 + rng.gen_range(-2..=2);
                    let jz = z as i64 + rng.gen_range(-2..=2);
                    if jx >= 0 && jx < size as i64 && jz >= 0 && jz < size as i64 {
                        tree_positions.push((jx as usize, jz as usize));
                    }
                }
            }
            
            // Buildings on flat-ish elevated terrain
            if height_above_water > 10 && height_above_water < 50 {
                // Check if area is relatively flat
                let mut is_flat = true;
                let check_size = 6;
                let center_h = ground_height;
                for dx in 0..check_size {
                    for dz in 0..check_size {
                        let cx = x + dx;
                        let cz = z + dz;
                        if cx < size && cz < size {
                            let h = heightmap[cx + cz * size];
                            if (h - center_h).abs() > 2 {
                                is_flat = false;
                                break;
                            }
                        }
                    }
                    if !is_flat { break; }
                }
                
                if is_flat && rng.gen_bool(0.15) {
                    let w = rng.gen_range(4..=8);
                    let d = rng.gen_range(4..=8);
                    building_positions.push((x, z, w, d));
                }
            }
        }
    }
    
    // Add trees
    for (tx, tz) in &tree_positions {
        let ground_y = heightmap[*tx + *tz * size];
        add_tree(&mut voxels, *tx as i64, ground_y, *tz as i64, &mut rng);
    }
    println!("Added {} trees", tree_positions.len());
    
    // Add buildings
    for (bx, bz, bw, bd) in &building_positions {
        let ground_y = heightmap[*bx + *bz * size];
        let height = rng.gen_range(8..=20);
        add_building(&mut voxels, *bx as i64, ground_y, *bz as i64, *bw, *bd, height, &mut rng);
    }
    println!("Added {} buildings", building_positions.len());
    
    println!("Total voxels: {}", voxels.len());
    
    // Step 4: Write output
    let mut max_y = 0i64;
    for ((_x, y, _z), _mat) in &voxels {
        max_y = max_y.max(*y);
    }
    
    println!("World bounds: X[0, {}], Y[0, {}], Z[0, {}]", size - 1, max_y, size - 1);
    
    if matches!(args.format, OutputFormat::Txt | OutputFormat::Both) {
        let path = PathBuf::from(format!("{}.txt", args.output_name));
        println!("Writing text format to {}...", path.display());
        let mut file = BufWriter::new(File::create(&path)?);
        for ((x, y, z), mat) in &voxels {
            writeln!(&mut file, "{} {} {} {}", x, y, z, mat + 1)?;
        }
        println!("Wrote text format");
    }
    
    if matches!(args.format, OutputFormat::Oct | OutputFormat::Both) {
        let path = PathBuf::from(format!("{}.oct", args.output_name));
        println!("Writing octree format to {}...", path.display());
        
        let max_coord = max_y.max(size as i64 - 1);
        let depth = calculate_required_depth(max_coord);
        println!("Octree depth: {} (world size: {}³)", depth, 16u64.pow(depth as u32));
        
        let mut world = World::new(depth);
        let palette = Palette::load("worlds/palette.txt");
        
        for ((x, y, z), mat) in &voxels {
            world.set(WorldPos::new(*x, *y, *z), mat + 1);
        }
        
        world.update_all_lod_metadata(&palette);
        save_world_file(&world, &path, true)?;
        
        let file_size = std::fs::metadata(&path)?.len() as f64 / 1024.0 / 1024.0;
        println!("Octree file size: {:.1} MB", file_size);
    }
    
    // Write metadata
    let meta_path = PathBuf::from(format!("{}_meta.json", args.output_name));
    let metadata = serde_json::json!({
        "size": size,
        "water_level_meters": args.water_level,
        "water_level_voxels": water_level_vox,
        "meters_per_voxel": args.meters_per_voxel,
        "seed": args.seed,
        "terrain_scale": args.terrain_scale,
        "terrain_amplitude_meters": args.terrain_amplitude,
        "voxel_count": voxels.len(),
        "tree_count": tree_positions.len(),
        "building_count": building_positions.len(),
        "bounds": {
            "min_x": 0,
            "max_x": size - 1,
            "min_y": base_y,
            "max_y": max_y,
            "min_z": 0,
            "max_z": size - 1,
        },
        "camera_position": [size / 2, water_level_vox as usize + 50, size / 2],
    });
    
    let meta_file = File::create(&meta_path)?;
    serde_json::to_writer_pretty(meta_file, &metadata)?;
    println!("Metadata saved to {}", meta_path.display());
    
    Ok(())
}

/// Generate a continuous heightmap using layered Perlin noise
/// Returns heights in VOXELS, centered around water_level_vox
fn generate_heightmap(
    perlin: &Perlin, 
    size: usize, 
    water_level_vox: i64, 
    scale: f64, 
    amplitude_vox: f64,
) -> Vec<i64> {
    let mut heights = vec![0i64; size * size];
    
    for x in 0..size {
        for z in 0..size {
            let nx = x as f64 / scale;
            let nz = z as f64 / scale;
            
            // Layer multiple octaves of noise for natural-looking terrain
            let mut height = 0.0;
            
            // Large-scale terrain features (mountains, valleys)
            height += perlin.get([nx * 0.5, nz * 0.5]) * 1.0;
            
            // Medium-scale hills
            height += perlin.get([nx * 1.0, nz * 1.0]) * 0.5;
            
            // Small-scale detail
            height += perlin.get([nx * 2.0, nz * 2.0]) * 0.25;
            
            // Fine detail
            height += perlin.get([nx * 4.0, nz * 4.0]) * 0.125;
            
            // Normalize to roughly -1 to 1 range
            height /= 1.875;
            
            // Convert to voxel height centered on water level
            let h = water_level_vox as f64 + height * amplitude_vox;
            let h = h.max(1.0) as i64;
            
            heights[x + z * size] = h;
        }
    }
    
    heights
}

/// Add a simple tree at the given position
fn add_tree(voxels: &mut HashMap<(i64, i64, i64), u8>, x: i64, ground_y: i64, z: i64, rng: &mut StdRng) {
    let trunk_height = rng.gen_range(4..=7);
    let canopy_radius = rng.gen_range(2..=3);
    
    // Trunk
    for y in 1..=trunk_height {
        voxels.insert((x, ground_y + y, z), MAT_WOOD);
    }
    
    // Canopy (sphere-ish)
    let canopy_center_y = ground_y + trunk_height;
    for dx in -canopy_radius..=canopy_radius {
        for dy in 0..=canopy_radius {
            for dz in -canopy_radius..=canopy_radius {
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq <= canopy_radius * canopy_radius + 1 {
                    let lx = x + dx;
                    let ly = canopy_center_y + dy;
                    let lz = z + dz;
                    // Don't overwrite trunk
                    if !(dx == 0 && dz == 0 && dy <= 0) {
                        voxels.insert((lx, ly, lz), MAT_LEAVES);
                    }
                }
            }
        }
    }
}

/// Add a simple building at the given position
fn add_building(
    voxels: &mut HashMap<(i64, i64, i64), u8>,
    x: i64, ground_y: i64, z: i64,
    width: usize, depth: usize, height: usize,
    rng: &mut StdRng
) {
    let wall_mat = if rng.gen_bool(0.5) { MAT_CONCRETE } else { MAT_BRICK };
    
    for dx in 0..width as i64 {
        for dz in 0..depth as i64 {
            for dy in 1..=height as i64 {
                let bx = x + dx;
                let by = ground_y + dy;
                let bz = z + dz;
                
                // Walls (hollow building)
                let is_edge = dx == 0 || dx == width as i64 - 1 || dz == 0 || dz == depth as i64 - 1;
                let is_roof = dy == height as i64;
                
                if is_roof {
                    voxels.insert((bx, by, bz), MAT_ROOF);
                } else if is_edge {
                    // Windows every few blocks
                    let is_window = dy > 1 && dy < height as i64 - 1 
                        && ((dx + dz + dy) % 3 == 0);
                    if is_window {
                        voxels.insert((bx, by, bz), MAT_GLASS);
                    } else {
                        voxels.insert((bx, by, bz), wall_mat);
                    }
                }
                // Interior is hollow
            }
        }
    }
}

fn calculate_required_depth(max_coord: i64) -> u8 {
    let mut depth = 1u8;
    while (16i64.pow(depth as u32)) <= max_coord {
        depth += 1;
    }
    depth
}
