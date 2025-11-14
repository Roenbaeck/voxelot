use std::env;
use std::path::Path;
use voxelot::{load_world_file, Chunk, Voxel};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: world_stats <path_to_world.oct>");
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    println!("Loading {}", path.display());
    let world = load_world_file(path)?;
    let depth = world.hierarchy_depth();
    let world_size = 16u64.pow(depth as u32);
    println!("Hierarchy depth: {}", depth);
    println!("World size: {} units per side", world_size);

    // Traverse the world to compute a true leaf-equivalent voxel count and bounding box
    let root = world.root();

    let mut leaf_voxel_count: u128 = 0; // use wide accumulator
    let mut min_x: Option<i64> = None;
    let mut min_y: Option<i64> = None;
    let mut min_z: Option<i64> = None;
    let mut max_x: Option<i64> = None;
    let mut max_y: Option<i64> = None;
    let mut max_z: Option<i64> = None;

    fn update_bbox(
        min_x: &mut Option<i64>,
        min_y: &mut Option<i64>,
        min_z: &mut Option<i64>,
        max_x: &mut Option<i64>,
        max_y: &mut Option<i64>,
        max_z: &mut Option<i64>,
        bx: i64,
        by: i64,
        bz: i64,
        ex: i64,
        ey: i64,
        ez: i64,
    ) {
        *min_x = Some(min_x.map_or(bx, |v| v.min(bx)));
        *min_y = Some(min_y.map_or(by, |v| v.min(by)));
        *min_z = Some(min_z.map_or(bz, |v| v.min(bz)));
        *max_x = Some(max_x.map_or(ex, |v| v.max(ex)));
        *max_y = Some(max_y.map_or(ey, |v| v.max(ey)));
        *max_z = Some(max_z.map_or(ez, |v| v.max(ez)));
    }

    // Recursive traversal: level is exponent such that scale = 16^level
    fn traverse(
        chunk: &Chunk,
        level: i32,
        origin_x: i64,
        origin_y: i64,
        origin_z: i64,
        leaf_voxel_count: &mut u128,
        min_x: &mut Option<i64>,
        min_y: &mut Option<i64>,
        min_z: &mut Option<i64>,
        max_x: &mut Option<i64>,
        max_y: &mut Option<i64>,
        max_z: &mut Option<i64>,
    ) {
        let scale = 16i64.pow(level as u32);

        for (pos, voxel) in chunk.iter() {
            let (lx, ly, lz) = pos;
            let base_x = origin_x + (lx as i64) * scale;
            let base_y = origin_y + (ly as i64) * scale;
            let base_z = origin_z + (lz as i64) * scale;

            match voxel {
                Voxel::Solid(_) => {
                    // This solid represents an entire sub-region of size scale^3
                    let ex = base_x + scale - 1;
                    let ey = base_y + scale - 1;
                    let ez = base_z + scale - 1;
                    update_bbox(
                        min_x, min_y, min_z, max_x, max_y, max_z, base_x, base_y, base_z,
                        ex, ey, ez,
                    );

                    // Add leaf-equivalent voxel count
                    let voxels_in_region: u128 = (scale as u128) * (scale as u128) * (scale as u128);
                    *leaf_voxel_count += voxels_in_region;
                }
                Voxel::Chunk(sub) => {
                    // Recurse into the sub-chunk
                    traverse(
                        sub,
                        level - 1,
                        base_x,
                        base_y,
                        base_z,
                        leaf_voxel_count,
                        min_x,
                        min_y,
                        min_z,
                        max_x,
                        max_y,
                        max_z,
                    );
                }
            }
        }
    }

    traverse(
        root,
        depth as i32 - 1,
        0,
        0,
        0,
        &mut leaf_voxel_count,
        &mut min_x,
        &mut min_y,
        &mut min_z,
        &mut max_x,
        &mut max_y,
        &mut max_z,
    );

    println!("Top-level presence count (root slots): {}", world.count());
    println!("Leaf-equivalent voxel count: {}", leaf_voxel_count);
    if let (Some(minx), Some(miny), Some(minz), Some(maxx), Some(maxy), Some(maxz)) = (
        min_x, min_y, min_z, max_x, max_y, max_z,
    ) {
        println!("Voxel bounding box: min=({}, {}, {}), max=({}, {}, {})", minx, miny, minz, maxx, maxy, maxz);
    } else {
        println!("Voxel bounding box: (empty)");
    }

    Ok(())
}
