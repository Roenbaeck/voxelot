use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

use voxelot::{octree_format::save_world_file, Palette, World, WorldPos};

const EARTH_RADIUS_METERS: f64 = 6_378_137.0;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Rust-based generator that replaces voxel_generator_tiles.py"
)]
struct Args {
    #[arg(long, default_value = "-74.006")]
    center_lon: f64,

    #[arg(long, default_value = "40.7128")]
    center_lat: f64,

    #[arg(long, default_value_t = 15)]
    zoom: u32,

    #[arg(long, default_value_t = 2)]
    radius: u32,

    #[arg(long = "voxels-per-tile", default_value_t = 128)]
    voxel_resolution: u32,

    #[arg(long = "meters-per-voxel", default_value = "1.25")]
    meters_per_voxel: f64,

    #[arg(long = "max-height-voxels", default_value_t = 192)]
    max_height_voxels: u32,

    #[arg(long, default_value_t = 1337)]
    seed: u64,

    #[arg(long = "output-name", default_value = "world_1")]
    output_name: String,

    #[arg(long, value_enum, default_value_t = OutputFormat::Oct)]
    format: OutputFormat,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum OutputFormat {
    Txt,
    Oct,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TileId {
    z: u32,
    x: i32,
    y: i32,
}

impl TileId {
    fn lon_lat_bounds(&self) -> ((f64, f64), (f64, f64)) {
        let n = 2f64.powi(self.z as i32);
        let min_lon = self.x as f64 / n * 360.0 - 180.0;
        let max_lon = (self.x as f64 + 1.0) / n * 360.0 - 180.0;

        let merc_y_to_lat = |y: f64| ((y).sinh()).atan().to_degrees();

        let north_lat = merc_y_to_lat(std::f64::consts::PI * (1.0 - 2.0 * self.y as f64 / n));
        let south_lat =
            merc_y_to_lat(std::f64::consts::PI * (1.0 - 2.0 * (self.y as f64 + 1.0) / n));
        ((min_lon, max_lon), (north_lat, south_lat))
    }
}

#[derive(Clone, Debug)]
struct TileData {
    roads: Vec<Polygon>,
    parks: Vec<Polygon>,
    water: Vec<Polygon>,
    buildings: Vec<BuildingEntry>,
    stats: TileStats,
}

type Polygon = Vec<[f64; 2]>;

#[derive(Clone, Debug)]
struct BuildingEntry {
    footprint: Polygon,
    tags: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
struct TileStats {
    district: String,
    building_count: usize,
    park_count: usize,
    water_features: usize,
    road_segments: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    average_height_m: Option<f64>,
    voxel_count: usize,
}

#[derive(Clone, Debug)]
struct TileFetcher {
    seed: u64,
}

impl TileFetcher {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn fetch(&self, tile: TileId) -> TileData {
        let mut rng = StdRng::seed_from_u64(stable_mix(&[
            tile.x as u64,
            tile.y as u64,
            tile.z as u64,
            self.seed,
        ]));
        let ((min_lon, max_lon), (north_lat, south_lat)) = tile.lon_lat_bounds();
        let lon_span = max_lon - min_lon;
        let lat_span = north_lat - south_lat;

        let rect_from_frac = |fx0: f64, fz0: f64, fx1: f64, fz1: f64| {
            vec![
                [min_lon + lon_span * fx0, south_lat + lat_span * fz0],
                [min_lon + lon_span * fx1, south_lat + lat_span * fz0],
                [min_lon + lon_span * fx1, south_lat + lat_span * fz1],
                [min_lon + lon_span * fx0, south_lat + lat_span * fz1],
            ]
        };

        const GRID: usize = 4;
        const ROAD_FRAC: f64 = 0.05;
        let road_positions: Vec<f64> = (0..=GRID).map(|i| i as f64 / GRID as f64).collect();
        let half_road = ROAD_FRAC / 2.0;

        let mut roads = Vec::new();
        for fx in &road_positions {
            let start = (*fx - half_road).max(0.0);
            let end = (*fx + half_road).min(1.0);
            roads.push(rect_from_frac(start, 0.0, end, 1.0));
        }
        for fz in &road_positions {
            let start = (*fz - half_road).max(0.0);
            let end = (*fz + half_road).min(1.0);
            roads.push(rect_from_frac(0.0, start, 1.0, end));
        }

        let mut blocks_frac = Vec::new();
        for bx in 0..GRID {
            let fx0 = road_positions[bx] + half_road;
            let fx1 = road_positions[bx + 1] - half_road;
            if fx1 - fx0 <= 0.02 {
                continue;
            }
            for bz in 0..GRID {
                let fz0 = road_positions[bz] + half_road;
                let fz1 = road_positions[bz + 1] - half_road;
                if fz1 - fz0 <= 0.02 {
                    continue;
                }
                blocks_frac.push((fx0, fz0, fx1, fz1));
            }
        }

        let mut available_indices: Vec<usize> = (0..blocks_frac.len()).collect();
        available_indices.shuffle(&mut rng);
        let slot_weights = WeightedIndex::new([0.4, 0.4, 0.2]).unwrap();
        let park_slots = slot_weights.sample(&mut rng);
        let park_indices: HashSet<usize> =
            available_indices.iter().cloned().take(park_slots).collect();

        let mut parks = Vec::new();
        for idx in &park_indices {
            let (fx0, fz0, fx1, fz1) = blocks_frac[*idx];
            let inset = (fx1 - fx0).min(fz1 - fz0) * 0.18;
            if inset <= 0.0 {
                continue;
            }
            parks.push(rect_from_frac(
                fx0 + inset,
                fz0 + inset,
                fx1 - inset,
                fz1 - inset,
            ));
        }

        let mut water_polys = Vec::new();
        let mut water_indices = HashSet::new();
        if rng.gen_bool(0.3) && !blocks_frac.is_empty() {
            let candidates: Vec<usize> = available_indices
                .iter()
                .copied()
                .filter(|idx| !park_indices.contains(idx))
                .collect();
            if !candidates.is_empty() {
                let idx = candidates[rng.gen_range(0..candidates.len())];
                water_indices.insert(idx);
                let (fx0, fz0, fx1, fz1) = blocks_frac[idx];
                let inset = (fx1 - fx0).min(fz1 - fz0) * 0.3;
                water_polys.push(rect_from_frac(
                    fx0 + inset,
                    fz0 + inset,
                    fx1 - inset,
                    fz1 - inset,
                ));
            }
        }
        if rng.gen_bool(0.25) {
            let width = 0.12 + rng.gen::<f64>() * 0.18;
            if rng.gen_bool(0.5) {
                let fx0 = rng.gen_range(0.0..0.2);
                water_polys.push(rect_from_frac(fx0, 0.0, (fx0 + width).min(1.0), 1.0));
            } else {
                let fz0 = rng.gen_range(0.0..0.2);
                water_polys.push(rect_from_frac(0.0, fz0, 1.0, (fz0 + width).min(1.0)));
            }
        }

        let districts = ["residential", "mixed_use", "industrial", "downtown"];
        let district_weights = WeightedIndex::new([0.35, 0.30, 0.15, 0.20]).unwrap();
        let district = districts[district_weights.sample(&mut rng)].to_string();

        let mut building_entries = Vec::new();
        let mut height_total = 0.0;
        for (idx, frac) in blocks_frac.iter().enumerate() {
            if park_indices.contains(&idx) || water_indices.contains(&idx) {
                continue;
            }
            let (mut fx0, mut fz0, mut fx1, mut fz1) = *frac;
            let margin = (fx1 - fx0).min(fz1 - fz0) * 0.07;
            fx0 += margin;
            fz0 += margin;
            fx1 -= margin;
            fz1 -= margin;
            if fx1 <= fx0 || fz1 <= fz0 {
                continue;
            }

            let lot_w = fx1 - fx0;
            let lot_d = fz1 - fz0;
            if lot_w <= 0.0 || lot_d <= 0.0 {
                continue;
            }

            let (slots, archetypes): (usize, Vec<&str>) = match district.as_str() {
                "residential" => (
                    rng.gen_range(1..=4),
                    vec!["house", "residential", "apartments"],
                ),
                "industrial" => (
                    rng.gen_range(1..=3),
                    vec!["industrial", "warehouse", "commercial"],
                ),
                "downtown" => (rng.gen_range(2..=6), vec!["office", "commercial", "tower"]),
                _ => (
                    rng.gen_range(1..=5),
                    vec!["residential", "commercial", "office"],
                ),
            };

            for slot in 0..slots {
                let bw = lot_w * rng.gen_range(0.3..0.9);
                let bd = lot_d * rng.gen_range(0.3..0.9);
                let bx0 = rng.gen_range(fx0..=(fx1 - bw).max(fx0));
                let bz0 = rng.gen_range(fz0..=(fz1 - bd).max(fz0));
                let bx1 = bx0 + bw;
                let bz1 = bz0 + bd;
                if bx1 - bx0 <= 0.005 || bz1 - bz0 <= 0.005 {
                    continue;
                }

                let archetype = archetypes[rng.gen_range(0..archetypes.len())];
                let mut tags = HashMap::new();
                tags.insert("building".to_string(), archetype.to_string());
                tags.insert("district".to_string(), district.clone());

                match archetype {
                    "house" | "residential" => {
                        let levels = rng.gen_range(2..=4);
                        tags.insert("building:levels".to_string(), levels.to_string());
                        tags.insert(
                            "roof:style".to_string(),
                            ["terracotta", "pitched"]
                                .choose(&mut rng)
                                .unwrap()
                                .to_string(),
                        );
                        tags.insert(
                            "facade".to_string(),
                            ["brick", "plaster"].choose(&mut rng).unwrap().to_string(),
                        );
                    }
                    "apartments" => {
                        let levels = rng.gen_range(4..=8);
                        tags.insert("building:levels".to_string(), levels.to_string());
                        tags.insert(
                            "roof:style".to_string(),
                            ["flat", "terracotta"].choose(&mut rng).unwrap().to_string(),
                        );
                        tags.insert(
                            "facade".to_string(),
                            ["brick", "concrete"].choose(&mut rng).unwrap().to_string(),
                        );
                    }
                    "commercial" | "office" => {
                        let levels = rng.gen_range(6..=16);
                        tags.insert("building:levels".to_string(), levels.to_string());
                        tags.insert("roof:style".to_string(), "flat".to_string());
                        tags.insert(
                            "facade".to_string(),
                            ["glass", "concrete"].choose(&mut rng).unwrap().to_string(),
                        );
                        if rng.gen_bool(0.2) {
                            tags.insert(
                                "name".to_string(),
                                format!("Plaza {}_{}_{}_{}", tile.x, tile.y, idx, slot),
                            );
                        }
                    }
                    "tower" => {
                        let height = rng.gen_range(60.0..=140.0);
                        tags.insert("height".to_string(), format!("{:.1}", height));
                        tags.insert("roof:style".to_string(), "dark".to_string());
                        tags.insert("facade".to_string(), "glass".to_string());
                        if rng.gen_bool(0.5) {
                            tags.insert("spire".to_string(), "true".to_string());
                        }
                    }
                    "industrial" | "warehouse" => {
                        let height = rng.gen_range(10.0..=20.0);
                        tags.insert("height".to_string(), format!("{:.1}", height));
                        tags.insert(
                            "roof:style".to_string(),
                            ["dark", "flat"].choose(&mut rng).unwrap().to_string(),
                        );
                        tags.insert(
                            "facade".to_string(),
                            ["concrete", "brick"].choose(&mut rng).unwrap().to_string(),
                        );
                    }
                    _ => {}
                }

                building_entries.push(BuildingEntry {
                    footprint: rect_from_frac(bx0, bz0, bx1, bz1),
                    tags,
                });
                height_total += infer_height_meters(&building_entries.last().unwrap().tags);
            }
        }

        let building_count = building_entries.len();
        let park_count = parks.len();
        let water_features = water_polys.len();
        let road_segments = roads.len();
        let average_height_m = if building_count == 0 {
            None
        } else {
            Some(height_total / building_count as f64)
        };

        TileData {
            roads,
            parks,
            water: water_polys,
            buildings: building_entries,
            stats: TileStats {
                district,
                building_count,
                park_count,
                water_features,
                road_segments,
                average_height_m,
                voxel_count: 0,
            },
        }
    }
}

fn infer_height_meters(tags: &HashMap<String, String>) -> f64 {
    if let Some(val) = tags.get("height") {
        if let Ok(parsed) = val.parse::<f64>() {
            return parsed;
        }
    }
    if let Some(val) = tags.get("building:levels") {
        if let Ok(levels) = val.parse::<i32>() {
            return (levels.max(1) as f64) * 3.1;
        }
    }
    match tags.get("building").map(|s| s.as_str()) {
        Some("house") => 8.0,
        Some("residential") => 9.0,
        Some("commercial") => 12.0,
        Some("industrial") => 14.0,
        Some("retail") => 10.0,
        Some("tower") => 50.0,
        _ => 9.0,
    }
}

fn stable_mix(values: &[u64]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &value in values {
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0001_0000_01B3);
    }
    hash
}

struct TileSpace {
    voxel_resolution: u32,
    meters_per_voxel: f64,
    min_x_m: f64,
    max_x_m: f64,
    min_y_m: f64,
    max_y_m: f64,
}

impl TileSpace {
    fn new(tile: TileId, voxel_resolution: u32, meters_per_voxel: f64) -> Self {
        let ((min_lon, max_lon), (north_lat, south_lat)) = tile.lon_lat_bounds();
        let (min_x_m, min_y_m) = lon_lat_to_mercator_meters(min_lon, south_lat);
        let (max_x_m, _) = lon_lat_to_mercator_meters(max_lon, south_lat);
        let (_, max_y_m) = lon_lat_to_mercator_meters(min_lon, north_lat);
        let max_x_m = if (max_x_m - min_x_m).abs() < 1e-6 {
            min_x_m + 1.0
        } else {
            max_x_m
        };
        let max_y_m = if (max_y_m - min_y_m).abs() < 1e-6 {
            min_y_m + 1.0
        } else {
            max_y_m
        };
        Self {
            voxel_resolution,
            meters_per_voxel,
            min_x_m,
            max_x_m,
            min_y_m,
            max_y_m,
        }
    }

    fn mercator_to_voxel(&self, x_m: f64, y_m: f64) -> (i32, i32) {
        let rel_x = (x_m - self.min_x_m) / (self.max_x_m - self.min_x_m);
        let rel_y = (y_m - self.min_y_m) / (self.max_y_m - self.min_y_m);
        let mut vx = (rel_x * self.voxel_resolution as f64).floor() as i32;
        let mut vz = (rel_y * self.voxel_resolution as f64).floor() as i32;
        vx = vx.clamp(0, self.voxel_resolution as i32 - 1);
        vz = vz.clamp(0, self.voxel_resolution as i32 - 1);
        (vx, vz)
    }
}

fn lon_lat_to_mercator_meters(lon_deg: f64, lat_deg: f64) -> (f64, f64) {
    let x = lon_deg.to_radians() * EARTH_RADIUS_METERS;
    let y = EARTH_RADIUS_METERS
        * ((lat_deg.to_radians() / 2.0 + std::f64::consts::FRAC_PI_4).tan()).ln();
    (x, y)
}

#[derive(Clone, Debug)]
struct VoxelRecord {
    x: i64,
    y: i64,
    z: i64,
    material_index: usize,
}

#[allow(dead_code)]
struct TileVoxelResult {
    tile: TileId,
    voxels: Vec<VoxelRecord>,
    stats: TileStats,
    voxel_resolution: u32,
}

fn rasterize_polygon(points: &[(i32, i32)], size: i32) -> Vec<(i32, i32)> {
    if points.is_empty() {
        return Vec::new();
    }
    let min_y = *points.iter().map(|(_, y)| y).min().unwrap_or(&0);
    let max_y = *points.iter().map(|(_, y)| y).max().unwrap_or(&0);
    let min_y = min_y.clamp(0, size);
    let max_y = max_y.clamp(0, size);

    let mut filled = Vec::new();
    for z in min_y..=max_y {
        let mut intersections = Vec::new();
        for i in 0..points.len() {
            let (x1, y1) = points[i];
            let (x2, y2) = points[(i + 1) % points.len()];
            if y1 == y2 {
                continue;
            }
            if (y1 <= z && z < y2) || (y2 <= z && z < y1) {
                let t = (z - y1) as f64 / (y2 - y1) as f64;
                let x_int = x1 as f64 + t * (x2 - x1) as f64;
                intersections.push(x_int);
            }
        }
        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for chunk in intersections.chunks(2) {
            if chunk.len() < 2 {
                break;
            }
            let x_start = chunk[0].floor() as i32;
            let x_end = chunk[1].ceil() as i32;
            for x in x_start..x_end {
                if (0..size).contains(&x) && (0..size).contains(&z) {
                    filled.push((x, z));
                }
            }
        }
    }
    filled
}

fn project_polygon(space: &TileSpace, footprint: &[[f64; 2]]) -> Vec<(i32, i32)> {
    footprint
        .iter()
        .map(|coord| {
            let (x_m, y_m) = lon_lat_to_mercator_meters(coord[0], coord[1]);
            space.mercator_to_voxel(x_m, y_m)
        })
        .collect()
}

fn polygon_cells(space: &TileSpace, polygons: &[Polygon]) -> Vec<Vec<(i32, i32)>> {
    polygons
        .iter()
        .filter_map(|footprint| {
            let projected = project_polygon(space, footprint);
            if projected.len() < 3 {
                return None;
            }
            let cells = rasterize_polygon(&projected, space.voxel_resolution as i32);
            if cells.is_empty() {
                return None;
            }
            Some(cells)
        })
        .collect()
}

fn voxelize_tile(
    tile: TileId,
    data: TileData,
    space: &TileSpace,
    max_height_voxels: u32,
) -> TileVoxelResult {
    let mut voxels = Vec::new();
    let materials = vec![
        "ground",
        "road",
        "park",
        "water",
        "wall",
        "wallbrick",
        "glazing",
        "roof",
        "roof_dark",
        "trunk",
        "canopy",
    ];
    let material_index: HashMap<&str, usize> = materials
        .iter()
        .enumerate()
        .map(|(i, &name)| (name, i))
        .collect();

    for x in 0..space.voxel_resolution as i32 {
        for z in 0..space.voxel_resolution as i32 {
            voxels.push(VoxelRecord {
                x: x as i64,
                y: 0,
                z: z as i64,
                material_index: *material_index.get("ground").unwrap_or(&0),
            });
        }
    }

    let apply_polygons = |polys: &[Polygon], voxel_type: &str, height: i64| {
        let mut overrides = Vec::new();
        for cells in polygon_cells(space, polys) {
            for (x, z) in cells {
                overrides.push(VoxelRecord {
                    x: x as i64,
                    y: height,
                    z: z as i64,
                    material_index: *material_index.get(voxel_type).unwrap_or(&0),
                });
            }
        }
        overrides
    };

    voxels.extend(apply_polygons(&data.water, "water", 0));
    if max_height_voxels > 1 {
        voxels.extend(apply_polygons(&data.water, "water", 1));
    }
    voxels.extend(apply_polygons(&data.roads, "road", 0));
    if max_height_voxels > 2 {
        voxels.extend(apply_polygons(&data.roads, "road", 1));
    }
    voxels.extend(apply_polygons(&data.parks, "park", 0));

    let mut park_cells = HashSet::new();
    for cells in polygon_cells(space, &data.parks) {
        for (x, z) in cells {
            park_cells.insert((x, z));
        }
    }

    if !park_cells.is_empty() {
        let mut tree_rng = StdRng::seed_from_u64(stable_mix(&[
            tile.x as u64,
            tile.y as u64,
            tile.z as u64,
            0x51F1F,
        ]));
        let canopy_limit = (max_height_voxels as i32 - 1).min(4);
        if canopy_limit >= 2 {
            for &(x, z) in park_cells.iter() {
                if tree_rng.gen_bool(0.06) {
                    voxels.push(VoxelRecord {
                        x: x as i64,
                        y: 1,
                        z: z as i64,
                        material_index: *material_index.get("trunk").unwrap(),
                    });
                    for h in 2..=canopy_limit {
                        voxels.push(VoxelRecord {
                            x: x as i64,
                            y: h as i64,
                            z: z as i64,
                            material_index: *material_index.get("canopy").unwrap(),
                        });
                    }
                }
            }
        }
    }

    let base_seed = stable_mix(&[tile.x as u64, tile.y as u64, tile.z as u64, 0xDEADBEEF]);
    let max_roof_y = (max_height_voxels as i64 - 1).max(2);

    for (b_idx, entry) in data.buildings.iter().enumerate() {
        let cells_list = polygon_cells(space, &[entry.footprint.clone()]);
        if cells_list.is_empty() {
            continue;
        }
        let cells = &cells_list[0];
        if cells.is_empty() {
            continue;
        }
        let height_m = infer_height_meters(&entry.tags);
        let height_vox = (height_m / space.meters_per_voxel).ceil() as i64;
        let height_vox = height_vox.clamp(3, max_height_voxels as i64);
        let roof_y = height_vox.min(max_roof_y);

        let facade = entry
            .tags
            .get("facade")
            .map(|s| s.as_str())
            .unwrap_or("concrete");
        let wall_name = match facade {
            "brick" => "wallbrick",
            "glass" => "glazing",
            "plaster" => "wall",
            "concrete" => "wall",
            "steel" => "glazing",
            _ => "wall",
        };
        let wall_idx = *material_index.get(wall_name).unwrap_or(&4);
        let glass_idx = *material_index.get("glazing").unwrap();

        let roof_style = entry
            .tags
            .get("roof:style")
            .map(|s| s.as_str())
            .unwrap_or("terracotta");
        let roof_idx = if matches!(roof_style, "dark" | "flat") {
            *material_index.get("roof_dark").unwrap()
        } else {
            *material_index.get("roof").unwrap()
        };

        let band_seed = base_seed ^ ((b_idx as u64) * 0x45D9F3B);
        let mut facade_rng = StdRng::seed_from_u64(band_seed & 0xFFFFFFFFFFFF);
        let window_period = facade_rng.gen_range(3..=5);

        for &(x, z) in cells {
            for y in 1..roof_y {
                let mut material_idx = wall_idx;
                if wall_name == "glazing" {
                    if y % 6 != 0 && y % 6 != 1 {
                        material_idx = glass_idx;
                    }
                } else if y > 2 && y % window_period == 0 {
                    material_idx = glass_idx;
                }
                voxels.push(VoxelRecord {
                    x: x as i64,
                    y,
                    z: z as i64,
                    material_index: material_idx,
                });
            }
            voxels.push(VoxelRecord {
                x: x as i64,
                y: roof_y,
                z: z as i64,
                material_index: roof_idx,
            });
        }

        if entry
            .tags
            .get("spire")
            .map(|s| s == "true")
            .unwrap_or(false)
            && roof_y + 3 < max_height_voxels as i64
        {
            let center = cells[cells.len() / 2];
            for extra in 1..=3 {
                voxels.push(VoxelRecord {
                    x: center.0 as i64,
                    y: roof_y + extra,
                    z: center.1 as i64,
                    material_index: roof_idx,
                });
            }
        }
    }

    let mut stats = data.stats.clone();
    stats.voxel_count = voxels.len();

    TileVoxelResult {
        tile,
        voxels,
        stats,
        voxel_resolution: space.voxel_resolution,
    }
}

fn generate_area(args: &Args) -> Vec<TileVoxelResult> {
    let center_tile = lon_lat_to_tile(args.center_lon, args.center_lat, args.zoom);
    let fetcher = TileFetcher::new(args.seed);
    let mut results = Vec::new();

    for dy in -(args.radius as i32)..=(args.radius as i32) {
        for dx in -(args.radius as i32)..=(args.radius as i32) {
            let tile = TileId {
                z: center_tile.z,
                x: center_tile.x + dx,
                y: center_tile.y + dy,
            };
            let data = fetcher.fetch(tile);
            let space = TileSpace::new(tile, args.voxel_resolution, args.meters_per_voxel);
            let result = voxelize_tile(tile, data, &space, args.max_height_voxels);
            results.push(result);
        }
    }

    results.sort_by_key(|r| (r.tile.y, r.tile.x));
    results
}

fn lon_lat_to_tile(lon: f64, lat: f64, z: u32) -> TileId {
    let n = 2.0f64.powi(z as i32);
    let x = ((lon + 180.0) / 360.0 * n).floor() as i32;
    let lat_rad = lat.to_radians();
    let y = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0 * n)
        .floor() as i32;
    TileId { z, x, y }
}

#[derive(Serialize)]
struct Metadata {
    world: WorldMetadata,
    tiles: Vec<TileMetadata>,
}

#[derive(Serialize)]
struct WorldMetadata {
    center_lon: f64,
    center_lat: f64,
    zoom: u32,
    seed: u64,
    radius: u32,
    voxel_resolution: u32,
    meters_per_voxel: f64,
    max_height_voxels: u32,
    bounds: Bounds,
    camera_position: [i64; 3],
}

#[derive(Serialize)]
struct Bounds {
    min_x: i64,
    max_x: i64,
    min_y: i64,
    max_y: i64,
    min_z: i64,
    max_z: i64,
}

#[derive(Serialize)]
struct TileMetadata {
    z: u32,
    x: i32,
    y: i32,
    voxel_count: usize,
    stats: TileStats,
}

fn calculate_required_depth(max_coord: i64) -> u8 {
    ((max_coord + 1).max(1) as f64).log(16.0).ceil().max(1.0) as u8
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let results = generate_area(&args);
    let first_tile = results.first().ok_or("No tiles generated")?;
    let mut min_x = i64::MAX;
    let mut min_z = i64::MAX;
    let mut max_x = i64::MIN;
    let mut max_z = i64::MIN;
    let mut max_y = i64::MIN;

    for result in &results {
        let tile_offset_x =
            (result.tile.x - first_tile.tile.x) as i64 * result.voxel_resolution as i64;
        let tile_offset_z =
            (result.tile.y - first_tile.tile.y) as i64 * result.voxel_resolution as i64;
        for voxel in &result.voxels {
            let x = tile_offset_x + voxel.x;
            let z = tile_offset_z + voxel.z;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
            max_y = max_y.max(voxel.y);
        }
    }

    println!("World bounds: X[{min_x}, {max_x}], Y[0, {max_y}], Z[{min_z}, {max_z}]");
    let total_voxels: usize = results.iter().map(|r| r.voxels.len()).sum();
    println!("Total voxels: {total_voxels}");

    if matches!(args.format, OutputFormat::Txt | OutputFormat::Both) {
        let ascii_path = PathBuf::from(format!(
            "{}{}.txt",
            args.output_name,
            if args.output_name.ends_with(".txt") {
                ""
            } else {
                ""
            }
        ));
        println!("Writing text format to {}...", ascii_path.display());
        let mut ascii = BufWriter::new(File::create(&ascii_path)?);
        for result in &results {
            let tile_offset_x =
                (result.tile.x - first_tile.tile.x) as i64 * result.voxel_resolution as i64;
            let tile_offset_z =
                (result.tile.y - first_tile.tile.y) as i64 * result.voxel_resolution as i64;
            for voxel in &result.voxels {
                let voxel_type = voxel.material_index + 1;
                writeln!(
                    &mut ascii,
                    "{} {} {} {}",
                    tile_offset_x + voxel.x,
                    voxel.y,
                    tile_offset_z + voxel.z,
                    voxel_type
                )?;
            }
        }
        println!("Wrote text format: {}", ascii_path.display());
    }

    if matches!(args.format, OutputFormat::Oct | OutputFormat::Both) {
        let oct_path = PathBuf::from(format!("{}.oct", args.output_name));
        println!("Writing octree format to {}...", oct_path.display());
        let max_coord = *[max_x, max_y, max_z].iter().max().unwrap_or(&0);
        let depth = calculate_required_depth(max_coord);
        println!(
            "Calculated required depth: {} (world size: {}³)",
            depth,
            16u64.pow(depth as u32)
        );

        let mut world = World::new(depth);
        let palette = Palette::load("worlds/palette.txt");

        for result in &results {
            let tile_offset_x =
                (result.tile.x - first_tile.tile.x) as i64 * result.voxel_resolution as i64;
            let tile_offset_z =
                (result.tile.y - first_tile.tile.y) as i64 * result.voxel_resolution as i64;
            for voxel in &result.voxels {
                let x = tile_offset_x + voxel.x;
                let z = tile_offset_z + voxel.z;
                world.set(
                    WorldPos::new(x, voxel.y, z),
                    (voxel.material_index + 1) as u8,
                );
            }
        }

        world.update_all_lod_metadata(&palette);
        save_world_file(&world, &oct_path, true)?;
        println!(
            "Octree file size: {:.1} MB",
            std::fs::metadata(&oct_path)?.len() as f64 / 1024.0 / 1024.0
        );
    }

    let meta_path = PathBuf::from(format!("{}_meta.json", args.output_name));
    let metadata = Metadata {
        world: WorldMetadata {
            center_lon: args.center_lon,
            center_lat: args.center_lat,
            zoom: args.zoom,
            seed: args.seed,
            radius: args.radius,
            voxel_resolution: args.voxel_resolution,
            meters_per_voxel: args.meters_per_voxel,
            max_height_voxels: args.max_height_voxels,
            bounds: Bounds {
                min_x,
                max_x,
                min_y: 0,
                max_y,
                min_z,
                max_z,
            },
            camera_position: [(min_x + max_x) / 2, max_y + 50, (min_z + max_z) / 2],
        },
        tiles: results
            .iter()
            .map(|result| TileMetadata {
                z: result.tile.z,
                x: result.tile.x,
                y: result.tile.y,
                voxel_count: result.stats.voxel_count,
                stats: result.stats.clone(),
            })
            .collect(),
    };
    let meta_file = File::create(&meta_path)?;
    serde_json::to_writer_pretty(meta_file, &metadata)?;
    println!("Metadata saved to {}", meta_path.display());

    Ok(())
}
