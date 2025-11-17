use serde::Deserialize;
use std::{env, fs, io, path::Path};
use voxelot::{
    cull_visible_voxels_parallel, load_world_file, Camera, Chunk, Palette, RenderConfig, Voxel,
    VoxelInstance, World, WorldPos,
};

#[derive(Deserialize)]
struct Meta {
    world: MetaWorld,
    tiles: Vec<MetaTile>,
}

#[derive(Deserialize)]
struct MetaWorld {
    voxel_resolution: i64,
}

#[derive(Deserialize)]
struct MetaTile {
    x: i64,
    y: i64,
}

struct Bounds {
    min_x: i64,
    min_y: i64,
    min_z: i64,
    max_x: i64,
    max_y: i64,
    max_z: i64,
}

fn compute_bounds(world: &World) -> Option<Bounds> {
    let depth = world.hierarchy_depth();
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

    fn traverse(
        chunk: &Chunk,
        level: i32,
        origin_x: i64,
        origin_y: i64,
        origin_z: i64,
        min_x: &mut Option<i64>,
        min_y: &mut Option<i64>,
        min_z: &mut Option<i64>,
        max_x: &mut Option<i64>,
        max_y: &mut Option<i64>,
        max_z: &mut Option<i64>,
    ) {
        let scale = 16i64.pow(level as u32);

        for ((lx, ly, lz), voxel) in chunk.iter() {
            let base_x = origin_x + (lx as i64) * scale;
            let base_y = origin_y + (ly as i64) * scale;
            let base_z = origin_z + (lz as i64) * scale;

            match voxel {
                Voxel::Solid(_) => {
                    update_bbox(
                        min_x,
                        min_y,
                        min_z,
                        max_x,
                        max_y,
                        max_z,
                        base_x,
                        base_y,
                        base_z,
                        base_x + scale - 1,
                        base_y + scale - 1,
                        base_z + scale - 1,
                    );
                }
                Voxel::Chunk(sub) => {
                    traverse(
                        sub,
                        level - 1,
                        base_x,
                        base_y,
                        base_z,
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
        world.root(),
        depth as i32 - 1,
        0,
        0,
        0,
        &mut min_x,
        &mut min_y,
        &mut min_z,
        &mut max_x,
        &mut max_y,
        &mut max_z,
    );

    Some(Bounds {
        min_x: min_x.unwrap_or(0),
        min_y: min_y.unwrap_or(0),
        min_z: min_z.unwrap_or(0),
        max_x: max_x.unwrap_or(0),
        max_y: max_y.unwrap_or(0),
        max_z: max_z.unwrap_or(0),
    })
}

enum Axis {
    X,
    Z,
}

fn plane_has_voxel(world: &World, axis: Axis, fixed: i64, bounds: &Bounds) -> bool {
    let (var_min, var_max) = match axis {
        Axis::X => (bounds.min_z, bounds.max_z),
        Axis::Z => (bounds.min_x, bounds.max_x),
    };

    for y in bounds.min_y..=bounds.max_y {
        for var in var_min..=var_max {
            let pos = match axis {
                Axis::X => WorldPos::new(fixed, y, var),
                Axis::Z => WorldPos::new(var, y, fixed),
            };
            if world.get(pos).is_some() {
                return true;
            }
        }
    }

    false
}

fn column_has_voxel(world: &World, axis: Axis, fixed: i64, var: i64, bounds: &Bounds) -> bool {
    for y in bounds.min_y..=bounds.max_y {
        let pos = match axis {
            Axis::X => WorldPos::new(fixed, y, var),
            Axis::Z => WorldPos::new(var, y, fixed),
        };
        if world.get(pos).is_some() {
            return true;
        }
    }
    false
}

fn instance_covers_axis(inst: &VoxelInstance, axis: Axis, coord: i64) -> bool {
    let pos = match axis {
        Axis::X => inst.position[0],
        Axis::Z => inst.position[2],
    };
    let max = pos + inst.scale - 1;
    pos <= coord && coord <= max
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: check_world_gaps <world.oct> <meta.json> [palette.txt]");
        std::process::exit(1);
    }

    let world_path = Path::new(&args[1]);
    let meta_path = Path::new(&args[2]);

    let mut world = load_world_file(world_path)?;
    let palette_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("worlds/palette.txt");
    let palette = Palette::load(palette_path);
    world.update_all_lod_metadata(&palette);

    let meta_text = fs::read_to_string(meta_path)?;
    let meta: Meta = serde_json::from_str(&meta_text)?;

    let tile_coords_x: Vec<i64> = meta.tiles.iter().map(|t| t.x).collect();
    let tile_coords_y: Vec<i64> = meta.tiles.iter().map(|t| t.y).collect();
    let min_tile_x = *tile_coords_x.iter().min().unwrap_or(&0);
    let max_tile_x = *tile_coords_x.iter().max().unwrap_or(&0);
    let min_tile_y = *tile_coords_y.iter().min().unwrap_or(&0);
    let max_tile_y = *tile_coords_y.iter().max().unwrap_or(&0);

    let tile_count_x = (max_tile_x - min_tile_x + 1).max(1);
    let tile_count_z = (max_tile_y - min_tile_y + 1).max(1);
    let tile_size = meta.world.voxel_resolution;

    let bounds = compute_bounds(&world).ok_or_else(|| {
        Box::<dyn std::error::Error>::from(io::Error::new(
            io::ErrorKind::Other,
            "World appears empty",
        ))
    })?;

    println!(
        "World bounds: x={}..{}, y={}..{}, z={}..{}",
        bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y, bounds.min_z, bounds.max_z
    );

    let base_x = bounds.min_x;
    let base_z = bounds.min_z;

    let x_boundaries: Vec<i64> = (1..tile_count_x).map(|i| base_x + i * tile_size).collect();
    let z_boundaries: Vec<i64> = (1..tile_count_z).map(|i| base_z + i * tile_size).collect();

    println!(
        "Tile grid: {}x{}, tile_size={}",
        tile_count_x, tile_count_z, tile_size
    );

    let mut render_config = RenderConfig::default();
    render_config.far_plane = 10000.0;
    render_config.lod_render_distance = 10000.0;
    let camera = Camera::with_config(
        [
            (bounds.min_x + bounds.max_x) as f32 / 2.0,
            bounds.max_y as f32 + 500.0,
            (bounds.min_z + bounds.max_z) as f32 / 2.0,
        ],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        render_config,
    );

    let world_size = world.world_size() as f32;
    let aabb_min = [0.0, 0.0, 0.0];
    let aabb_max = [world_size, world_size, world_size];
    println!(
        "Frustum culls world AABB? {}",
        camera.frustum_cull_aabb(aabb_min, aabb_max)
    );

    let scale = 16i64.pow(world.hierarchy_depth() as u32 - 1);
    for (x, y, z) in world.root().positions() {
        let cell_min = [
            (x as i64 * scale) as f32,
            (y as i64 * scale) as f32,
            (z as i64 * scale) as f32,
        ];
        let cell_max = [
            ((x as i64 + 1) * scale) as f32,
            ((y as i64 + 1) * scale) as f32,
            ((z as i64 + 1) * scale) as f32,
        ];
        println!(
            "Root cell ({},{},{}) frustum? {}",
            x,
            y,
            z,
            camera.frustum_cull_aabb(cell_min, cell_max)
        );
    }

    let visible_instances = cull_visible_voxels_parallel(&world, &camera);
    println!(
        "Visible instances: {} (scales {:?})",
        visible_instances.len(),
        visible_instances
            .iter()
            .take(5)
            .map(|inst| inst.scale)
            .collect::<Vec<i64>>()
    );

    for &coord in &x_boundaries {
        let raw = plane_has_voxel(&world, Axis::X, coord, &bounds);
        let rendered = visible_instances
            .iter()
            .any(|inst| instance_covers_axis(inst, Axis::X, coord));
        let var_min = bounds.min_z;
        let var_max = bounds.max_z;
        let total_columns = (var_max - var_min + 1).max(0);
        let columns_with_voxel = (var_min..=var_max)
            .filter(|&var| column_has_voxel(&world, Axis::X, coord, var, &bounds))
            .count() as i64;
        println!(
            "X-plane at {}: raw {}, rendered {}, columns {}/{}",
            coord,
            if raw { "voxels" } else { "empty" },
            if rendered { "covered" } else { "missing" },
            columns_with_voxel,
            total_columns
        );
    }

    for &coord in &z_boundaries {
        let raw = plane_has_voxel(&world, Axis::Z, coord, &bounds);
        let rendered = visible_instances
            .iter()
            .any(|inst| instance_covers_axis(inst, Axis::Z, coord));
        let var_min = bounds.min_x;
        let var_max = bounds.max_x;
        let total_columns = (var_max - var_min + 1).max(0);
        let columns_with_voxel = (var_min..=var_max)
            .filter(|&var| column_has_voxel(&world, Axis::Z, coord, var, &bounds))
            .count() as i64;
        println!(
            "Z-plane at {}: raw {}, rendered {}, columns {}/{}",
            coord,
            if raw { "voxels" } else { "empty" },
            if rendered { "covered" } else { "missing" },
            columns_with_voxel,
            total_columns
        );
    }

    Ok(())
}
