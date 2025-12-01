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
use noise::{Perlin, NoiseFn};

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

    #[arg(long, default_value_t = 2.0)]
    water_level: f64,

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
    biome: Biome,
    stats: TileStats,
    // Optional, precomputed height field for the tile (meters)
    heights_m: Option<Vec<f64>>,
    // Optional per-cell base ground heights (voxel Y levels), used to ensure pad continuity.
    base_ground_vox: Option<Vec<i64>>,
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
    biome: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Biome {
    City,
    Hill,
    Lake,
}

// Material Indices (matching palette.txt)
const MAT_GRASS_DARK: usize = 8;
const MAT_GRASS_LIGHT: usize = 9;
const MAT_DIRT: usize = 10;
const MAT_STONE: usize = 11;
const MAT_SAND: usize = 12;
const MAT_SNOW: usize = 13;
const MAT_WATER_DEEP: usize = 14;
const MAT_WATER_SHALLOW: usize = 15;

const MAT_ASPHALT: usize = 16;
const MAT_CONCRETE: usize = 17;
const MAT_PAVEMENT: usize = 18;
const MAT_COBBLE: usize = 19;

const MAT_BRICK_RED: usize = 20;
const MAT_PLASTER: usize = 21;
const MAT_GLASS: usize = 22;
const MAT_STEEL: usize = 23;
const MAT_ROOF_TERRACOTTA: usize = 24;
const MAT_ROOF_SLATE: usize = 25;
const MAT_MARBLE: usize = 26;

const MAT_TRUNK_DARK: usize = 32;
const MAT_TRUNK_LIGHT: usize = 33;
const MAT_LEAVES_DARK: usize = 34;
const MAT_LEAVES_MED: usize = 35;
const MAT_LEAVES_LIGHT: usize = 36;
const MAT_LEAVES_AUTUMN: usize = 37;

const MAT_LIGHT_WARM: usize = 40;
const MAT_LIGHT_COOL: usize = 41;
const MAT_WINDOW_WARM: usize = 42;
const MAT_WINDOW_COOL: usize = 43;
const MAT_NEON_RED: usize = 44;
const MAT_NEON_GREEN: usize = 45;
const MAT_NEON_BLUE: usize = 46;

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

        // Determine Biome
        // Simple noise-based biome selection
        let biome_noise = rng.gen::<f64>();
        let biome = if biome_noise < 0.2 {
            Biome::Lake
        } else if biome_noise < 0.5 {
            Biome::Hill
        } else {
            Biome::City
        };

        // If Lake or Hill, we might skip city generation logic or adapt it.
        // For now, we'll keep the city generation but maybe clear it if it's a pure nature biome,
        // or we can mix them. Let's make Lake and Hill distinct for now.

        if biome != Biome::City {
            return TileData {
                roads: Vec::new(),
                parks: Vec::new(),
                water: Vec::new(),
                buildings: Vec::new(),
                biome,
                stats: TileStats {
                    district: "nature".to_string(),
                    building_count: 0,
                    park_count: 0,
                    water_features: 0,
                    road_segments: 0,
                    average_height_m: None,
                    voxel_count: 0,
                    biome: format!("{:?}", biome),
                },
                heights_m: None,
                base_ground_vox: None,
            };
        }

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
                biome: format!("{:?}", biome),
            },
            heights_m: None,
            base_ground_vox: None,
            biome,
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

fn fbm(perlin: &Perlin, x: f64, y: f64, octaves: usize, lacunarity: f64, gain: f64) -> f64 {
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut total = 0.0;
    let mut max = 0.0;
    for _ in 0..octaves {
        total += perlin.get([x * frequency, y * frequency]) * amplitude;
        max += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    if max.abs() < 1e-12 { 0.0 } else { total / max }
}

fn smooth_tiles_pass(smoothed_map: &mut HashMap<TileId, Vec<f64>>, size: usize) {
    let keys: Vec<TileId> = smoothed_map.keys().copied().collect();
    for tile_id in keys {
        if let Some(base) = smoothed_map.get(&tile_id) {
            if base.len() != size * size { continue; }
            let mut new_h = base.clone();
            for xi in 0..size {
                for zi in 0..size {
                    let idx = xi + zi * size;
                    let mut sum = base[idx];
                    let mut count = 1.0;
                    // neighbor tiles for edge smoothing
                    if xi == 0 {
                        let neighbor = TileId { z: tile_id.z, x: tile_id.x - 1, y: tile_id.y };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[(size - 1) + zi * size]; count += 1.0;
                        }
                    }
                    if xi == size - 1 {
                        let neighbor = TileId { z: tile_id.z, x: tile_id.x + 1, y: tile_id.y };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[0 + zi * size]; count += 1.0;
                        }
                    }
                    if zi == 0 {
                        let neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y - 1 };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[xi + (size - 1) * size]; count += 1.0;
                        }
                    }
                    if zi == size - 1 {
                        let neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y + 1 };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[xi + 0 * size]; count += 1.0;
                        }
                    }
                    new_h[idx] = sum / count;
                }
            }
            smoothed_map.insert(tile_id, new_h);
        }
    }
}

// Blend across tile edges to create gradual ramps over a width
fn blend_tile_edges(smoothed_map: &mut HashMap<TileId, Vec<f64>>, size: usize, blend_width: usize) {
    let keys: Vec<TileId> = smoothed_map.keys().copied().collect();
    for tile_id in keys {
        if let Some(base) = smoothed_map.get(&tile_id) {
            if base.len() != size * size { continue; }
            let mut new_h = base.clone();
            // blend each side with neighbor
            // left side x=0 blends with neighbor's x=size-1
            let left_neighbor = TileId { z: tile_id.z, x: tile_id.x - 1, y: tile_id.y };
            if let Some(nei) = smoothed_map.get(&left_neighbor) {
                for w in 0..blend_width {
                    let left_x = w;
                    let right_x = size - 1 - w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64; // 0..1
                    for zi in 0..size {
                        let idx = left_x + zi * size;
                        let idx_nei = right_x + zi * size;
                        let a = base[idx];
                        let b = nei[idx_nei];
                        new_h[idx] = a * (1.0 - t) + b * t;
                    }
                }
            }
            // right side x=size-1 blends with neighbor's x=0
            let right_neighbor = TileId { z: tile_id.z, x: tile_id.x + 1, y: tile_id.y };
            if let Some(nei) = smoothed_map.get(&right_neighbor) {
                for w in 0..blend_width {
                    let x = size - 1 - w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for zi in 0..size { let idx = x + zi * size; let idx_nei = w + zi * size; let a = base[idx]; let b = nei[idx_nei]; new_h[idx] = a * (1.0 - t) + b * t; }
                }
            }
            // top/bottom blend similar
            let top_neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y - 1 };
            if let Some(nei) = smoothed_map.get(&top_neighbor) {
                for w in 0..blend_width {
                    let z = w;
                    let z_nei = size - 1 - w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for xi in 0..size { let idx = xi + z * size; let idx_nei = xi + z_nei * size; let a = base[idx]; let b = nei[idx_nei]; new_h[idx] = a * (1.0 - t) + b * t; }
                }
            }
            let bottom_neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y + 1 };
            if let Some(nei) = smoothed_map.get(&bottom_neighbor) {
                for w in 0..blend_width {
                    let z = size - 1 - w;
                    let z_nei = w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for xi in 0..size { let idx = xi + z * size; let idx_nei = xi + z_nei * size; let a = base[idx]; let b = nei[idx_nei]; new_h[idx] = a * (1.0 - t) + b * t; }
                }
            }
            smoothed_map.insert(tile_id, new_h);
        }
    }
}

fn smooth_base_pass(smoothed_map: &mut HashMap<TileId, Vec<i64>>, size: usize) {
    let keys: Vec<TileId> = smoothed_map.keys().copied().collect();
    for tile_id in keys {
        if let Some(base) = smoothed_map.get(&tile_id) {
            if base.len() != size * size { continue; }
            let mut new_b = base.clone();
            for xi in 0..size {
                for zi in 0..size {
                    let idx = xi + zi * size;
                    let mut sum = base[idx] as f64;
                    let mut count = 1.0;
                    if xi == 0 {
                        let neighbor = TileId { z: tile_id.z, x: tile_id.x - 1, y: tile_id.y };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[(size - 1) + zi * size] as f64; count += 1.0;
                        }
                    }
                    if xi == size - 1 {
                        let neighbor = TileId { z: tile_id.z, x: tile_id.x + 1, y: tile_id.y };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[0 + zi * size] as f64; count += 1.0;
                        }
                    }
                    if zi == 0 {
                        let neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y - 1 };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[xi + (size - 1) * size] as f64; count += 1.0;
                        }
                    }
                    if zi == size - 1 {
                        let neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y + 1 };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[xi + 0 * size] as f64; count += 1.0;
                        }
                    }
                    new_b[idx] = (sum / count).round() as i64;
                }
            }
            smoothed_map.insert(tile_id, new_b);
        }
    }
}

fn blend_base_edges(smoothed_map: &mut HashMap<TileId, Vec<i64>>, size: usize, blend_width: usize) {
    let keys: Vec<TileId> = smoothed_map.keys().copied().collect();
    for tile_id in keys {
        if let Some(base) = smoothed_map.get(&tile_id) {
            if base.len() != size * size { continue; }
            let mut new_b = base.clone();
            let left_neighbor = TileId { z: tile_id.z, x: tile_id.x - 1, y: tile_id.y };
            if let Some(nei) = smoothed_map.get(&left_neighbor) {
                for w in 0..blend_width {
                    let left_x = w;
                    let right_x = size - 1 - w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for zi in 0..size {
                        let idx = left_x + zi * size;
                        let idx_nei = right_x + zi * size;
                        let a = base[idx] as f64;
                        let b = nei[idx_nei] as f64;
                        new_b[idx] = (a * (1.0 - t) + b * t).round() as i64;
                    }
                }
            }
            let right_neighbor = TileId { z: tile_id.z, x: tile_id.x + 1, y: tile_id.y };
            if let Some(nei) = smoothed_map.get(&right_neighbor) {
                for w in 0..blend_width {
                    let x = size - 1 - w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for zi in 0..size { let idx = x + zi * size; let idx_nei = w + zi * size; let a = base[idx] as f64; let b = nei[idx_nei] as f64; new_b[idx] = (a * (1.0 - t) + b * t).round() as i64; }
                }
            }
            let top_neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y - 1 };
            if let Some(nei) = smoothed_map.get(&top_neighbor) {
                for w in 0..blend_width {
                    let z = w;
                    let z_nei = size - 1 - w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for xi in 0..size { let idx = xi + z * size; let idx_nei = xi + z_nei * size; let a = base[idx] as f64; let b = nei[idx_nei] as f64; new_b[idx] = (a * (1.0 - t) + b * t).round() as i64; }
                }
            }
            let bottom_neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y + 1 };
            if let Some(nei) = smoothed_map.get(&bottom_neighbor) {
                for w in 0..blend_width {
                    let z = size - 1 - w;
                    let z_nei = w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for xi in 0..size { let idx = xi + z * size; let idx_nei = xi + z_nei * size; let a = base[idx] as f64; let b = nei[idx_nei] as f64; new_b[idx] = (a * (1.0 - t) + b * t).round() as i64; }
                }
            }
            smoothed_map.insert(tile_id, new_b);
        }
    }
}

fn sample_tile_heights(perlin: &Perlin, space: &TileSpace) -> Vec<f64> {
    let size = space.voxel_resolution as usize;
    let mut heights = vec![0.0f64; size * size];
    for xi in 0..size {
        for zi in 0..size {
            let wx = space.min_x_m + (xi as f64 / size as f64) * (space.max_x_m - space.min_x_m);
            let wz = space.min_y_m + (zi as f64 / size as f64) * (space.max_y_m - space.min_y_m);
            let large = fbm(perlin, wx * 0.001, wz * 0.001, 6, 2.0, 0.5) * 80.0;
            let mid = fbm(perlin, wx * 0.005, wz * 0.005, 4, 2.0, 0.5) * 30.0;
            let detail = fbm(perlin, wx * 0.02, wz * 0.02, 3, 2.0, 0.5) * 6.0;
            let base = 5.0;
            heights[xi + zi * size] = (base + large + mid.abs() * 0.6 + detail).max(1.0);
        }
    }
    heights
}

fn compute_river_mask(space: &TileSpace, perlin: &Perlin, heights_m: &[f64], river_scale: f64, threshold: f64) -> Vec<bool> {
    let size = space.voxel_resolution as usize;
    let mut mask = vec![false; size * size];
    let max_river_elevation = space.water_level_m + 40.0;
    for xi in 0..size {
        for zi in 0..size {
            let wx = space.min_x_m + (xi as f64 / size as f64) * (space.max_x_m - space.min_x_m);
            let wz = space.min_y_m + (zi as f64 / size as f64) * (space.max_y_m - space.min_y_m);
            let v = fbm(perlin, wx * river_scale, wz * river_scale, 3, 2.0, 0.5);
            // Rivers follow low values in this global low-frequency noise
            let elev = heights_m[xi + zi * size];
            if v < threshold && elev <= max_river_elevation {
                mask[xi + zi * size] = true;
            }
        }
    }
    // Dilate to give rivers some width
    let mut out = mask.clone();
    let radius = 1;
    for xi in 0..size {
        for zi in 0..size {
            if mask[xi + zi * size] {
                for dx in -(radius as isize)..=(radius as isize) {
                    for dz in -(radius as isize)..=(radius as isize) {
                        let nx = xi as isize + dx;
                        let nz = zi as isize + dz;
                        if nx >= 0 && nx < size as isize && nz >= 0 && nz < size as isize {
                            out[nx as usize + nz as usize * size] = true;
                        }
                    }
                }
            }
        }
    }
    out
}

struct TileSpace {
    voxel_resolution: u32,
    meters_per_voxel: f64,
    min_x_m: f64,
    max_x_m: f64,
    min_y_m: f64,
    max_y_m: f64,
    seed: u64,
    water_level_m: f64,
}

impl TileSpace {
    fn new(tile: TileId, voxel_resolution: u32, meters_per_voxel: f64, seed: u64, water_level_m: f64) -> Self {
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
            seed,
            water_level_m,
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
    match data.biome {
        Biome::City => voxelize_city(tile, data, space, max_height_voxels),
        Biome::Hill => voxelize_hill(tile, data, space, max_height_voxels),
        Biome::Lake => voxelize_lake(tile, data, space, max_height_voxels),
    }
}

// deprecated: previously used simple sin/cos fallback; refactored to a FBM-perlin approach

fn voxelize_hill(
    tile: TileId,
    data: TileData,
    space: &TileSpace,
    max_height_voxels: u32,
) -> TileVoxelResult {
    // Refactored hill generation using multi-octave FBM, slope-based materials, and clustered vegetation.
    let mut voxels = Vec::new();
    // Use precomputed tile heights if provided, otherwise compute locally
    let size = space.voxel_resolution as usize;
    let mut heights_m: Vec<f64> = if let Some(ref pre) = data.heights_m { pre.clone() } else { sample_tile_heights(&Perlin::new(space.seed as u32), space) };
    let perlin = Perlin::new(space.seed as u32);

    // Smooth edges near tile border to make transitions more plausible: simple kernel
    for xi in 0..size {
        for zi in 0..size {
            let mut sum = 0.0;
            let mut cnt = 0;
            for dx in -1..=1 {
                for dz in -1..=1 {
                    let nx = xi as isize + dx as isize;
                    let nz = zi as isize + dz as isize;
                    if nx >= 0 && nx < size as isize && nz >= 0 && nz < size as isize {
                        sum += heights_m[nx as usize + nz as usize * size];
                        cnt += 1;
                    }
                }
            }
            if cnt > 0 {
                heights_m[xi + zi * size] = (sum / cnt as f64) * 0.95 + heights_m[xi + zi * size] * 0.05;
            }
        }
    }

    // Compute river mask and apply small carving before voxelization
    let river_mask = compute_river_mask(space, &perlin, &heights_m, 1.0 / 3000.0, -0.35);
    for xi in 0..size {
        for zi in 0..size {
            let idx = xi + zi * size;
            if river_mask[idx] {
                heights_m[idx] = (heights_m[idx] - 2.0).max(0.5);
            }
        }
    }

    // Convert to voxels, set layers based on slope and height
    for xi in 0..size {
        for zi in 0..size {
            let h_m = heights_m[xi + zi * size];
            let h_vox = (h_m / space.meters_per_voxel).ceil() as i64;
            let h_vox = h_vox.clamp(1, max_height_voxels as i64);

            // approximate slope: sample neighbors
            let center = heights_m[xi + zi * size];
            let neighbor_x = if xi + 1 < size { heights_m[(xi + 1) + zi * size] } else { center };
            let neighbor_z = if zi + 1 < size { heights_m[xi + (zi + 1) * size] } else { center };
            let slope_x = (neighbor_x - center).abs();
            let slope_z = (neighbor_z - center).abs();
            let slope = (slope_x + slope_z) * 0.5 / space.meters_per_voxel;

            for y in 0..h_vox {
                let mat = if slope > 3.0 {
                    // steep cliff - expose stone
                    if y < h_vox - 1 { MAT_STONE } else { MAT_STONE }
                } else if y < h_vox - 4 {
                    MAT_STONE
                } else if y < h_vox - 1 {
                    MAT_DIRT
                } else {
                    // topmost
                    if h_m > 140.0 { MAT_SNOW } else { if (xi + zi) % 3 == 0 { MAT_GRASS_LIGHT } else { MAT_GRASS_DARK } }
                };
                voxels.push(VoxelRecord { x: xi as i64, y, z: zi as i64, material_index: mat });
            }
            if river_mask[xi + zi * size] {
                let water_to = (h_vox + 2).min(max_height_voxels as i64);
                for y in (h_vox + 1)..=water_to {
                    voxels.push(VoxelRecord { x: xi as i64, y, z: zi as i64, material_index: MAT_WATER_SHALLOW });
                }
            }
        }
    }

    // Vegetation clusters: we pick seeds based on noise and place small clusters where slope is gentle
    let mut rng = StdRng::seed_from_u64(stable_mix(&[tile.x as u64, tile.y as u64, tile.z as u64, 0xDEADBEEF]));
    let veg_density = 0.02 + (rng.gen::<f64>() * 0.04);
    for xi in 0..size {
        for zi in 0..size {
                if rng.gen_bool(veg_density) {
                let h_m = heights_m[xi + zi * size];
                    // Skip vegetation if cell is below (or very close to) global water level
                    if h_m <= space.water_level_m + 0.5 {
                        continue;
                    }
                let h_vox = (h_m / space.meters_per_voxel).ceil() as i64;
                // Only place vegetation on gentle slopes and not too high
                let center = heights_m[xi + zi * size];
                let neighbor_x = if xi + 1 < size { heights_m[(xi + 1) + zi * size] } else { center };
                let neighbor_z = if zi + 1 < size { heights_m[xi + (zi + 1) * size] } else { center };
                let slope_x = (neighbor_x - center).abs();
                let slope_z = (neighbor_z - center).abs();
                let slope = (slope_x + slope_z) * 0.5 / space.meters_per_voxel;
                if slope < 1.1 && h_vox + 6 < max_height_voxels as i64 {
                    // Tree probability decreases with altitude
                    let p = 0.5 - (h_m / 500.0);
                    if rng.gen_bool(p.max(0.05)) {
                        let trunk_h = rng.gen_range(3..6);
                        for ty in 0..trunk_h {
                            voxels.push(VoxelRecord { x: xi as i64, y: h_vox + ty, z: zi as i64, material_index: MAT_TRUNK_DARK });
                        }
                        let canopy_base = h_vox + trunk_h - 1;
                        for cx in -2..=2 {
                            for cz in -2..=2 {
                                for cy_offset in 0..=3 {
                                    let cy = canopy_base + cy_offset;
                                    if cy >= h_vox && cy < max_height_voxels as i64 {
                                        let dist = (cx as i32).abs() + (cz as i32).abs() + (cy_offset as i32).abs();
                                        if dist <= 3 {
                                            // Some trees are evergreen, some are autumnal or with lighter canopy
                                            let leaf_choice = if rng.gen_bool(0.12) {
                                                MAT_LEAVES_AUTUMN
                                            } else if rng.gen_bool(0.25) {
                                                MAT_LEAVES_LIGHT
                                            } else {
                                                MAT_LEAVES_DARK
                                            };
                                            voxels.push(VoxelRecord { x: (xi as i64 + cx).clamp(0, size as i64 - 1), y: cy, z: (zi as i64 + cz).clamp(0, size as i64 - 1), material_index: leaf_choice });
                                        }
                                                        // Occasionally place a small rock boulder near trees
                                                        if rng.gen_bool(0.02) {
                                                            let bx = (xi as i64 + rng.gen_range(-1..=1)).clamp(0, size as i64 - 1) as i64;
                                                            let bz = (zi as i64 + rng.gen_range(-1..=1)).clamp(0, size as i64 - 1) as i64;
                                                            let by = h_vox - 1;
                                                            for rx in -1..=0 {
                                                                for rz in -1..=0 {
                                                                    let xrx = (bx + rx).clamp(0, size as i64 - 1);
                                                                    let zrz = (bz + rz).clamp(0, size as i64 - 1);
                                                                    voxels.push(VoxelRecord { x: xrx, y: by, z: zrz, material_index: MAT_STONE });
                                                                }
                                                            }
                                                        }
                                    }
                                }
                            }
                        }
                    }
                }
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

fn voxelize_lake(
    tile: TileId,
    data: TileData,
    space: &TileSpace,
    max_height_voxels: u32,
) -> TileVoxelResult {
    // Improved lake generation using a smooth heightmap + dynamic water fill, shoreline, and vegetation.
    let mut voxels = Vec::new();
    let perlin = Perlin::new(space.seed as u32);
    let size = space.voxel_resolution as usize;
    let heights_m: Vec<f64> = if let Some(ref pre) = data.heights_m { pre.clone() } else { sample_tile_heights(&perlin, space) };
    let mut min_h = f64::INFINITY;
    let mut max_h = f64::NEG_INFINITY;
    for xi in 0..size { for zi in 0..size { let elev = heights_m[xi + zi * size]; min_h = min_h.min(elev); max_h = max_h.max(elev); }}

    // Global water level from world config
    let water_level_m = space.water_level_m;
    let mut rng = StdRng::seed_from_u64(space.seed.wrapping_add(stable_mix(&[tile.x as u64, tile.y as u64, tile.z as u64])));
    let water_level_vox = (water_level_m / space.meters_per_voxel).ceil() as i64;

    // Identify lake mask within tile: cells below water_level_m expanded to make shorelines
    let mut lake_mask = vec![false; size * size];
    for xi in 0..size {
        for zi in 0..size {
            let elev = heights_m[xi + zi * size];
            if elev <= water_level_m + 1.5 {
                lake_mask[xi + zi * size] = true;
            }
        }
    }

    // Rivers based on global low frequency noise
    let river_mask = compute_river_mask(space, &perlin, &heights_m, 1.0 / 3000.0, -0.35);
    for i in 0..lake_mask.len() {
        if river_mask[i] {
            lake_mask[i] = true;
        }
    }

    // Fill ground and water
    for xi in 0..size {
        for zi in 0..size {
            let idx = xi + zi * size;
            let h_m = heights_m[idx];
            let h_vox = (h_m / space.meters_per_voxel).floor() as i64;
            let h_vox = h_vox.clamp(0, max_height_voxels as i64);

            // Ground layering: stone beneath, sand near shoreline, otherwise dirt/grass
            for y in 0..=h_vox {
                let mat = if y < h_vox - 2 { MAT_STONE } else { MAT_DIRT };
                voxels.push(VoxelRecord { x: xi as i64, y, z: zi as i64, material_index: mat });
            }

            if lake_mask[idx] {
                // Shallow gradient for shore
                let water_top = water_level_vox.clamp(0, max_height_voxels as i64);
                let water_depth = water_top - h_vox;
                if water_depth > 0 {
                    // Shallow near edges
                    for y in (h_vox + 1)..=water_top {
                        let mat = if water_depth >= 6 { MAT_WATER_DEEP } else { MAT_WATER_SHALLOW };
                        voxels.push(VoxelRecord { x: xi as i64, y, z: zi as i64, material_index: mat });
                    }
                }
                // Add shoreline sand layer just above water
                if h_vox + 1 < max_height_voxels as i64 {
                    voxels.push(VoxelRecord { x: xi as i64, y: h_vox + 1, z: zi as i64, material_index: MAT_SAND });
                }
                // Wet vegetation around lakes (place only in the shoreline band)
                let water_top = water_level_vox.clamp(0, max_height_voxels as i64);
                if h_vox >= water_top - 2 && h_vox <= water_top && rng.gen_bool(0.02) {
                    voxels.push(VoxelRecord { x: xi as i64, y: h_vox + 1, z: zi as i64, material_index: MAT_TRUNK_LIGHT });
                    voxels.push(VoxelRecord { x: xi as i64, y: h_vox + 2, z: zi as i64, material_index: MAT_LEAVES_MED });
                }
                // Occasional shoreline boulder
                if rng.gen_bool(0.015) {
                    let bx = (xi as i64 + rng.gen_range(-1..=1)).clamp(0, size as i64 - 1);
                    let bz = (zi as i64 + rng.gen_range(-1..=1)).clamp(0, size as i64 - 1);
                    let by = h_vox;
                    voxels.push(VoxelRecord { x: bx, y: by, z: bz, material_index: MAT_STONE });
                }
            } else {
                // Not water - maybe place some trees if gentle slope
                let neighbor_x = if xi + 1 < size { heights_m[(xi + 1) + zi * size] } else { heights_m[idx] };
                let neighbor_z = if zi + 1 < size { heights_m[xi + (zi + 1) * size] } else { heights_m[idx] };
                let slope = ((neighbor_x - heights_m[idx]).abs() + (neighbor_z - heights_m[idx]).abs()) * 0.5 / space.meters_per_voxel;
                if slope < 1.1 && rng.gen_bool(0.03) {
                    let trunk_h = rng.gen_range(3..6);
                    for ty in 0..trunk_h { voxels.push(VoxelRecord { x: xi as i64, y: h_vox + ty, z: zi as i64, material_index: MAT_TRUNK_DARK }); }
                    for cx in -2..=2 {
                        for cz in -2..=2 {
                            for cy_offset in 0..=3 {
                                let cy = h_vox + trunk_h - 2 + cy_offset;
                                if cy >= h_vox && cy < max_height_voxels as i64 {
                                    let dist = (cx as i32).abs() + (cz as i32).abs() + (cy_offset as i32).abs();
                                    if dist <= 3 { voxels.push(VoxelRecord { x: (xi as i64 + cx).clamp(0, size as i64 - 1), y: cy, z: (zi as i64 + cz).clamp(0, size as i64 - 1), material_index: MAT_LEAVES_MED }); }
                                }
                            }
                        }
                    }
                }
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

fn voxelize_city(
    tile: TileId,
    data: TileData,
    space: &TileSpace,
    max_height_voxels: u32,
) -> TileVoxelResult {
    let mut voxels = Vec::new();
    let perlin = Perlin::new(space.seed as u32);
    let heights_m: Vec<f64> = if let Some(ref pre) = data.heights_m { pre.clone() } else { sample_tile_heights(&perlin, space) };

    // Build per-cell base ground that is the maximum of terrain & minimal water-safe base
    let water_level_vox = (space.water_level_m / space.meters_per_voxel).ceil() as i64;
    let mut base_ground_vox = vec![0i64; (space.voxel_resolution * space.voxel_resolution) as usize];
    let size = space.voxel_resolution as usize;
    for xi in 0..size {
        for zi in 0..size {
            let elev_m = heights_m[xi + zi * size];
            let mut ground_vox = (elev_m / space.meters_per_voxel).ceil() as i64;
            // Ensure the city ground is at least water level + margin
            let min_ground = water_level_vox + 3;
            if ground_vox < min_ground { ground_vox = min_ground; }
            base_ground_vox[xi + zi * size] = ground_vox;
        }
    }

    // Building pad pass: expand raised base around building footprints to create smooth pads
    for entry in data.buildings.iter() {
        let cells_list = polygon_cells(space, &[entry.footprint.clone()]);
        if cells_list.is_empty() { continue; }
        let cells = &cells_list[0];
        if cells.is_empty() { continue; }
        // Compute building foundation base as the maximum base_ground across footprint
        let mut b_base = 0i64;
        for &(x, z) in cells {
            let idx_b = x as usize + z as usize * size;
            if base_ground_vox[idx_b] > b_base { b_base = base_ground_vox[idx_b]; }
        }
        // Expand neighboring ground to create a small pad (2 cells) around footprint for smoother transition
        let pad_radius = 2;
        for &(x, z) in cells {
            for dx in -(pad_radius as i32)..=(pad_radius as i32) {
                for dz in -(pad_radius as i32)..=(pad_radius as i32) {
                    let nx = x + dx;
                    let nz = z + dz;
                    if nx >= 0 && nz >= 0 && (nx as usize) < size && (nz as usize) < size {
                        let idxn = nx as usize + nz as usize * size;
                        // Raise base_ground_vox but preserve reasonable slope (decrease with dist)
                        let dist = dx.abs().max(dz.abs()) as i64;
                        let raise_to = (b_base - dist).max(base_ground_vox[idxn]);
                        if raise_to > base_ground_vox[idxn] {
                            base_ground_vox[idxn] = raise_to;
                        }
                    }
                }
            }
        }
    }

    // Fill the ground using the adjusted per-cell base heights
    for xi in 0..size {
        for zi in 0..size {
            let ground_vox = base_ground_vox[xi + zi * size];
            for y in 0..=ground_vox {
                let mat = if y < ground_vox - 2 { MAT_STONE } else { MAT_DIRT };
                voxels.push(VoxelRecord { x: xi as i64, y, z: zi as i64, material_index: mat });
            }
        }
    }

    // Per-cell polygon placement that respects per-cell base ground
    let apply_polygons_per_cell = |polys: &[Polygon], voxel_type: usize, height_offset: i64| {
        let mut overrides = Vec::new();
        for cells in polygon_cells(space, polys) {
            for (x, z) in cells {
                let idx = x as usize + z as usize * size;
                let base_y = base_ground_vox[idx];
                overrides.push(VoxelRecord { x: x as i64, y: base_y + height_offset, z: z as i64, material_index: voxel_type });
            }
        }
        overrides
    };

    // Apply polygons now that terrain fill is created
    voxels.extend(apply_polygons_per_cell(&data.water, MAT_WATER_DEEP, 0));
    if max_height_voxels > 1 { voxels.extend(apply_polygons_per_cell(&data.water, MAT_WATER_DEEP, 1)); }
    voxels.extend(apply_polygons_per_cell(&data.roads, MAT_ASPHALT, 0));
    if max_height_voxels > 2 { voxels.extend(apply_polygons_per_cell(&data.roads, MAT_ASPHALT, 1)); }
    voxels.extend(apply_polygons_per_cell(&data.parks, MAT_GRASS_DARK, 0));
    if max_height_voxels > 1 { voxels.extend(apply_polygons_per_cell(&data.water, MAT_WATER_DEEP, 1)); }
    voxels.extend(apply_polygons_per_cell(&data.roads, MAT_ASPHALT, 0));
    if max_height_voxels > 2 { voxels.extend(apply_polygons_per_cell(&data.roads, MAT_ASPHALT, 1)); }
    // parks will be applied later

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
            // Add park path network based on grid and perlin
            let _path_rng = StdRng::seed_from_u64(space.seed.wrapping_add(stable_mix(&[tile.x as u64, tile.y as u64])));
            for &(x, z) in park_cells.iter() {
                    if tree_rng.gen_bool(0.08) {
                        // only place trees if the cell is not below the water level
                        let idx = x as usize + z as usize * size;
                        if heights_m[idx] > space.water_level_m + 0.5 {
                            let base_y = base_ground_vox[idx];
                            voxels.push(VoxelRecord { x: x as i64, y: base_y, z: z as i64, material_index: MAT_TRUNK_LIGHT });
                            let leaf_choice = if tree_rng.gen_bool(0.25) { MAT_LEAVES_AUTUMN } else { MAT_LEAVES_MED };
                            for h in 1..=canopy_limit { voxels.push(VoxelRecord { x: x as i64, y: (base_y + h as i64), z: z as i64, material_index: leaf_choice }); }
                        }
                    // Small shrub cluster near tree
                    if tree_rng.gen_bool(0.15) {
                        let idx_sh1 = (x + 1) as usize + z as usize * size;
                        let idx_sh2 = (x - 1) as usize + z as usize * size;
                        let base_sh1 = base_ground_vox[idx_sh1];
                        let base_sh2 = base_ground_vox[idx_sh2];
                        voxels.push(VoxelRecord { x: x as i64 + 1, y: base_sh1, z: z as i64, material_index: MAT_LEAVES_LIGHT });
                        voxels.push(VoxelRecord { x: x as i64 - 1, y: base_sh2, z: z as i64, material_index: MAT_LEAVES_LIGHT });
                    }
                    // Occasional statue/bench
                    if tree_rng.gen_bool(0.01) {
                        let idxb = (x + 1) as usize + (z + 1) as usize * size;
                        let base_b = base_ground_vox[idxb];
                        voxels.push(VoxelRecord { x: x as i64 + 1, y: base_b, z: z as i64 + 1, material_index: MAT_MARBLE });
                    }
                }
            }
        }
            // Construct simple paths across the park cells
            let path_threshold = 8;
            for &(x, z) in park_cells.iter() {
                if (x + z) % path_threshold == 0 || (x - z) % (path_threshold + 2) == 0 {
                    // place pavement at base level
                    let idxp = x as usize + z as usize * size;
                    let y = base_ground_vox[idxp] + 1;
                    voxels.push(VoxelRecord { x: x as i64, y, z: z as i64, material_index: MAT_PAVEMENT });
                }
            }
            // Random outhouse placements (small 2x2 structures)
            for &(x, z) in park_cells.iter() {
                if tree_rng.gen_bool(0.015) {
                    let ox = x as i64;
                    let oz = z as i64;
                    let idxo = x as usize + z as usize * size;
                    let oy = base_ground_vox[idxo] + 1;
                    for dx in 0..2 {
                        for dz in 0..2 {
                            let idxf = (x + dx as i32) as usize + (z + dz as i32) as usize * size;
                            let oyf = base_ground_vox[idxf] + 1;
                            voxels.push(VoxelRecord { x: ox + dx, y: oyf, z: oz + dz, material_index: MAT_PLASTER });
                        }
                    }
                    voxels.push(VoxelRecord { x: ox, y: oy + 2, z: oz, material_index: MAT_ROOF_TERRACOTTA });
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

        let wall_idx = match facade {
            "brick" => MAT_BRICK_RED,
            "glass" => MAT_GLASS,
            "plaster" => MAT_PLASTER,
            "concrete" => MAT_CONCRETE,
            "steel" => MAT_STEEL,
            _ => MAT_CONCRETE,
        };
        let glass_idx = MAT_GLASS;

        let roof_style = entry
            .tags
            .get("roof:style")
            .map(|s| s.as_str())
            .unwrap_or("terracotta");
        let roof_idx = if matches!(roof_style, "dark" | "flat") {
            MAT_ROOF_SLATE
        } else {
            MAT_ROOF_TERRACOTTA
        };

        let band_seed = base_seed ^ ((b_idx as u64) * 0x45D9F3B);
        let mut facade_rng = StdRng::seed_from_u64(band_seed & 0xFFFFFFFFFFFF);
        let window_period = facade_rng.gen_range(3..=5);

        // Compute building foundation base as the maximum base_ground across footprint
        let mut b_base = 0i64;
        for &(x, z) in cells {
            let idx_b = x as usize + z as usize * size;
            if base_ground_vox[idx_b] > b_base { b_base = base_ground_vox[idx_b]; }
        }
        // Fill foundation under the building footprint
        for &(x, z) in cells {
            let idx_b = x as usize + z as usize * size;
            let ground_vox = base_ground_vox[idx_b];
            for gy in ground_vox..=b_base {
                let mat = if gy < b_base - 1 { MAT_STONE } else { MAT_DIRT };
                voxels.push(VoxelRecord { x: x as i64, y: gy, z: z as i64, material_index: mat });
            }
        }
        for &(x, z) in cells {
            for y in 1..roof_y {
                let mut material_idx = wall_idx;
                if facade == "glass" {
                    if y % 6 != 0 && y % 6 != 1 {
                        material_idx = glass_idx;
                    }
                } else if y > 2 && y % window_period == 0 {
                    // Window logic
                    if facade_rng.gen_bool(0.1) {
                        material_idx = MAT_WINDOW_WARM;
                    } else {
                        material_idx = glass_idx;
                    }
                }
                voxels.push(VoxelRecord { x: x as i64, y: b_base + y, z: z as i64, material_index: material_idx });
            }
            voxels.push(VoxelRecord { x: x as i64, y: b_base + roof_y, z: z as i64, material_index: roof_idx });
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
                voxels.push(VoxelRecord { x: center.0 as i64, y: b_base + roof_y + extra, z: center.1 as i64, material_index: roof_idx });
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
    let perlin = Perlin::new(args.seed as u32);
    // Collect tile metadata and precompute heights
    let mut tile_map: HashMap<TileId, TileData> = HashMap::new();
    let mut tile_spaces: HashMap<TileId, TileSpace> = HashMap::new();
    let mut tiles: Vec<TileId> = Vec::new();
    for dy in -(args.radius as i32)..=(args.radius as i32) {
        for dx in -(args.radius as i32)..=(args.radius as i32) {
            let tile = TileId { z: center_tile.z, x: center_tile.x + dx, y: center_tile.y + dy };
            let mut data = fetcher.fetch(tile);
            let space = TileSpace::new(tile, args.voxel_resolution, args.meters_per_voxel, args.seed, args.water_level);
            let heights = sample_tile_heights(&perlin, &space);
            data.heights_m = Some(heights);
            tile_map.insert(tile, data);
            tile_spaces.insert(tile, space);
            tiles.push(tile);
        }
    }

    // Smooth across tile boundaries by averaging edge cells with neighbor's edge cells
    let size = args.voxel_resolution as usize;
    let mut smoothed_map: HashMap<TileId, Vec<f64>> = HashMap::new();
    // Gather pad updates without mutating the tile_map during iteration
    let mut pad_updates: Vec<(TileId, usize, i64)> = Vec::new();
    for (tile_id, data) in tile_map.iter() {
        if let Some(h) = data.heights_m.as_ref() { smoothed_map.insert(*tile_id, h.clone()); }
    }
    // Multiple smoothing passes
    let smoothing_passes = 3usize;
    for _ in 0..smoothing_passes { smooth_tiles_pass(&mut smoothed_map, size); }
    // Blend tile edges to produce ramps across tile boundaries
    let blend_width = 6usize;
    blend_tile_edges(&mut smoothed_map, size, blend_width);
    // Additional smoothing passes to settle blend
    for _ in 0..1 { smooth_tiles_pass(&mut smoothed_map, size); }
    // Previously this loop performed a single pass; we've replaced it with helper calls
    for tile_id in tiles.iter() {
        let mut new_h = smoothed_map.get(tile_id).unwrap().clone();
        if new_h.len() != size * size { continue; }
        for xi in 0..size {
            for zi in 0..size {
                let idx = xi + zi * size;
                let mut sum = new_h[idx];
                let mut count = 1.0;
                // Left edge neighbor
                if xi == 0 {
                    let neighbor = TileId { z: tile_id.z, x: tile_id.x - 1, y: tile_id.y };
                    if let Some(nei_h) = smoothed_map.get(&neighbor) {
                        let idx_nei = (size - 1) + zi * size;
                        sum += nei_h[idx_nei]; count += 1.0;
                    }
                }
                // Right edge neighbor
                if xi == size - 1 {
                    let neighbor = TileId { z: tile_id.z, x: tile_id.x + 1, y: tile_id.y };
                    if let Some(nei_h) = smoothed_map.get(&neighbor) {
                        let idx_nei = 0 + zi * size;
                        sum += nei_h[idx_nei]; count += 1.0;
                    }
                }
                // Top edge neighbor
                if zi == 0 {
                    let neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y - 1 };
                    if let Some(nei_h) = smoothed_map.get(&neighbor) {
                        let idx_nei = xi + (size - 1) * size;
                        sum += nei_h[idx_nei]; count += 1.0;
                    }
                }
                // Bottom edge neighbor
                if zi == size - 1 {
                    let neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y + 1 };
                    if let Some(nei_h) = smoothed_map.get(&neighbor) {
                        let idx_nei = xi + 0 * size;
                        sum += nei_h[idx_nei]; count += 1.0;
                    }
                }
                new_h[idx] = sum / count;
            }
        }
        smoothed_map.insert(*tile_id, new_h);
    }
    // Write smoothed heights back to tile data and compute base_ground_vox
    for (tile_id, heights) in smoothed_map.iter() {
        if let Some(data) = tile_map.get_mut(tile_id) {
            data.heights_m = Some(heights.clone());
            // compute base ground vox per cell
            let mut base_vox = vec![0i64; size * size];
            let water_level_vox = (args.water_level / args.meters_per_voxel).ceil() as i64;
            for xi in 0..size {
                for zi in 0..size {
                    let elev_m = heights[xi + zi * size];
                    let mut g = (elev_m / args.meters_per_voxel).ceil() as i64;
                    if g < water_level_vox + 3 { g = water_level_vox + 3; }
                    base_vox[xi + zi * size] = g;
                }
            }
            data.base_ground_vox = Some(base_vox);
        }
    }

    // Diagnostic: compute border differences to quantify smoothing
    let mut max_border_delta = 0.0f64;
    let mut total_border_delta = 0.0f64;
    let mut border_count = 0usize;
    for tile_id in tiles.iter() {
        if let Some(h) = smoothed_map.get(tile_id) {
            for xi in 0..size {
                // compare top edge to neighbor
                let idx_top = xi + 0 * size;
                let top_neighbor = TileId { z: tile_id.z, x: tile_id.x, y: tile_id.y - 1 };
                if let Some(nei) = smoothed_map.get(&top_neighbor) {
                    let nei_idx = xi + (size - 1) * size;
                    let delta = (h[idx_top] - nei[nei_idx]).abs();
                    max_border_delta = max_border_delta.max(delta);
                    total_border_delta += delta;
                    border_count += 1;
                }
                // compare left edge
                let idx_left = 0 + xi * size;
                let left_neighbor = TileId { z: tile_id.z, x: tile_id.x - 1, y: tile_id.y };
                if let Some(nei) = smoothed_map.get(&left_neighbor) {
                    let nei_idx = (size - 1) + xi * size;
                    let delta = (h[idx_left] - nei[nei_idx]).abs();
                    max_border_delta = max_border_delta.max(delta);
                    total_border_delta += delta;
                    border_count += 1;
                }
            }
        }
    }
    if border_count > 0 {
        println!("Height border diffs after smoothing: max = {:.3}m, avg = {:.3}m", max_border_delta, total_border_delta / border_count as f64);
    }

    // Building pad pass across tiles: expand elevated bases around building footprints
    // Increase pad radius for smoother, more gradual ramps across neighbors
    let pad_radius = 3i32;
    for (tile_id, data) in tile_map.iter() {
        if let Some(cells) = Some(&data.buildings) {
            for entry in cells.iter() {
                let footprint_cells = polygon_cells(tile_spaces.get(tile_id).unwrap(), &[entry.footprint.clone()]);
                if footprint_cells.is_empty() { continue; }
                let cells = &footprint_cells[0];
                // compute max base for building footprint
                let mut b_base = 0i64;
                for &(x, z) in cells.iter() {
                    let idx = x as usize + z as usize * size;
                    if let Some(bv) = data.base_ground_vox.as_ref() { if bv[idx] > b_base { b_base = bv[idx]; } }
                }
                // expand pad across neighborhood across tiles
                for &(bx, bz) in cells.iter() {
                    for dx in -(pad_radius)..=(pad_radius) {
                        for dz in -(pad_radius)..=(pad_radius) {
                            let nx = bx + dx;
                            let nz = bz + dz;
                            // if outside current tile bounds, compute neighbor tile and cell
                            if nx < 0 || nx >= size as i32 || nz < 0 || nz >= size as i32 {
                                let mut tx = tile_id.x as i32;
                                let mut ty = tile_id.y as i32;
                                let mut cx = nx;
                                let mut cz = nz;
                                while cx < 0 { cx += size as i32; tx -= 1; }
                                while cx >= size as i32 { cx -= size as i32; tx += 1; }
                                while cz < 0 { cz += size as i32; ty -= 1; }
                                while cz >= size as i32 { cz -= size as i32; ty += 1; }
                                let neighbor_tile = TileId { z: tile_id.z, x: tx, y: ty };
                                let idxn = cx as usize + cz as usize * size;
                                    let dist = dx.abs().max(dz.abs()) as i64;
                                    // slope-aware taper: blend neighbor existing base to building base
                                    let neighbor_base = tile_map.get(&neighbor_tile).and_then(|d| d.base_ground_vox.as_ref()).map(|bv| bv[idxn]).unwrap_or(0i64);
                                    let distf = dist as f64;
                                    let pad_span = (pad_radius + 1) as f64;
                                    let w = (distf / pad_span).clamp(0.0, 1.0);
                                    // ease curve for smoother ramp (quadratic)
                                    let t = 1.0 - w * w;
                                    let raise_to_f = (neighbor_base as f64) * (1.0 - t) + (b_base as f64) * t;
                                    let raise_to = raise_to_f.round() as i64;
                                    pad_updates.push((neighbor_tile, idxn, raise_to));
                            } else {
                                let idx = nx as usize + nz as usize * size;
                                let dist = dx.abs().max(dz.abs()) as i64;
                                let neighbor_base = data.base_ground_vox.as_ref().map(|bv| bv[idx]).unwrap_or(0i64);
                                let distf = dist as f64;
                                let pad_span = (pad_radius + 1) as f64;
                                let w = (distf / pad_span).clamp(0.0, 1.0);
                                let t = 1.0 - w * w;
                                let raise_to_f = (neighbor_base as f64) * (1.0 - t) + (b_base as f64) * t;
                                let raise_to = raise_to_f.round() as i64;
                                pad_updates.push((*tile_id, idx, raise_to));
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply pad updates we collected
    for (tile_id, idx, raise_to) in pad_updates.iter() {
        if let Some(nei_data) = tile_map.get_mut(tile_id) {
            if let Some(nei_bv) = nei_data.base_ground_vox.as_mut() {
                if *idx < nei_bv.len() {
                    if *raise_to > nei_bv[*idx] { nei_bv[*idx] = *raise_to; }
                }
            }
        }
    }

    // Base smoothing after pad updates: smooth base elevations to remove step artifacts
    let mut base_map: HashMap<TileId, Vec<i64>> = HashMap::new();
    for (tile_id, data) in tile_map.iter() { if let Some(bv) = data.base_ground_vox.as_ref() { base_map.insert(*tile_id, bv.clone()); } }
    // More aggressive smoothing to reduce pad-induced plateaus
    for _ in 0..6 { smooth_base_pass(&mut base_map, size); }
    blend_base_edges(&mut base_map, size, 12);
    for _ in 0..3 { smooth_base_pass(&mut base_map, size); }
    for (tile_id, base) in base_map.iter() { if let Some(data) = tile_map.get_mut(tile_id) { data.base_ground_vox = Some(base.clone()); }}

    // Voxelize each tile using the smoothed heights
    for dy in -(args.radius as i32)..=(args.radius as i32) {
        for dx in -(args.radius as i32)..=(args.radius as i32) {
            let tile = TileId { z: center_tile.z, x: center_tile.x + dx, y: center_tile.y + dy };
            let data = tile_map.remove(&tile).unwrap();
            let space = tile_spaces.remove(&tile).unwrap();
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
