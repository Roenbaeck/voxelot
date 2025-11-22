use std::collections::HashMap;
use std::env;
use std::path::Path;
use voxelot::{load_world_file, Chunk, Voxel, World, WorldPos};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let filename = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("worlds/world.oct"); // Default path

    println!("Loading {}...", filename);
    let mut world = load_world_file(Path::new(filename))?;
    println!("World depth: {}", world.hierarchy_depth());

    // Update metadata so we have bounding boxes
    use voxelot::Palette;
    let palette = Palette::from_string("1 255 255 255 255").unwrap();
    world.update_all_lod_metadata(&palette);

    // We want to find vertical columns of chunks.
    // We'll traverse the tree and collect chunks by (x, z) at the leaf-parent level (depth 1 chunks).
    // Or maybe depth 2 chunks?
    // Let's look at leaf chunks (depth 1).

    let mut columns: HashMap<(i64, i64), Vec<(i64, [u8; 6])>> = HashMap::new();

    traverse(
        &world,
        world.root(),
        world.hierarchy_depth() as i32 - 1,
        0,
        0,
        0,
        &mut columns,
    );

    println!("Found {} columns.", columns.len());

    // Analyze columns
    for ((x, z), chunks) in columns {
        if chunks.len() < 3 {
            continue;
        } // Need a stack

        // Sort by Y
        let mut sorted = chunks.clone();
        sorted.sort_by_key(|c| c.0);

        // Check for width inconsistencies
        // We expect width (xmax - xmin) to be roughly constant for a "cuboid skyscraper"

        let first_width = sorted[0].1[3] as i16 - sorted[0].1[0] as i16;
        let first_depth = sorted[0].1[5] as i16 - sorted[0].1[2] as i16;

        for i in 1..sorted.len() {
            let width = sorted[i].1[3] as i16 - sorted[i].1[0] as i16;
            let depth = sorted[i].1[5] as i16 - sorted[i].1[2] as i16;

            if width != first_width || depth != first_depth {
                println!("Inconsistency in column ({}, {}):", x, z);
                for (y, bbox) in &sorted {
                    println!(
                        "  Y={}: Bbox={:?} Width={} Depth={}",
                        y,
                        bbox,
                        bbox[3] - bbox[0],
                        bbox[5] - bbox[2]
                    );
                }
                println!("--------------------------------");
                break;
            }
        }
    }

    Ok(())
}

fn traverse(
    world: &World,
    chunk: &Chunk,
    level: i32,
    origin_x: i64,
    origin_y: i64,
    origin_z: i64,
    columns: &mut HashMap<(i64, i64), Vec<(i64, [u8; 6])>>,
) {
    let scale = 16i64.pow(level as u32);

    // If we are at the chunk level (e.g. scale 16, depth 0 in traversal terms, but depth 1 in world terms)
    // Actually, Chunk is 16^3.
    // If level == 0, scale is 1. This is leaf voxels.
    // We want to look at Chunks.
    // If level == 0, we are inside a leaf chunk.
    // We want to record the bbox of the *leaf chunk itself*.
    // But we are traversing *inside* it.

    // Wait, traverse is called on root.
    // If root is depth 3. level starts at 2. scale 256.
    // We iterate children.
    // If child is Chunk, we recurse.

    // We want to stop at some level and record the chunk's bbox.
    // Let's stop at level 0 (scale 1).
    // No, we want the bbox of the chunk containing the voxels.
    // That is the chunk at level 0.

    // If level == 0, 'chunk' IS the leaf chunk.
    // We should record its bbox.
    if level == 0 {
        if let Some(bbox) = chunk.bounding_box {
            columns
                .entry((origin_x, origin_z))
                .or_default()
                .push((origin_y, bbox));
        }
        return;
    }

    for (pos, voxel) in chunk.iter() {
        let (lx, ly, lz) = pos;
        let base_x = origin_x + (lx as i64) * scale;
        let base_y = origin_y + (ly as i64) * scale;
        let base_z = origin_z + (lz as i64) * scale;

        match voxel {
            Voxel::Chunk(sub) => {
                traverse(world, sub, level - 1, base_x, base_y, base_z, columns);
            }
            Voxel::Solid(_) => {
                // Should not happen if we stop at level 0 (leaf chunk)
                // But if we have mixed hierarchy, it might.
            }
        }
    }
}
