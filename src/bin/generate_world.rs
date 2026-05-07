use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

use noise::{NoiseFn, Perlin};
use voxelot::{file_format::save_world_file, Palette, World, WorldPos};

const EARTH_RADIUS_METERS: f64 = 6_378_137.0;
// Total amplitude of noise layers (350 + 100 + 20)
const BASE_HEIGHT_RANGE: f64 = 470.0;

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

    /// Blend width in voxels across tiles
    #[arg(long, default_value_t = 6)]
    tile_blend_width: u32,
    /// Building pad radius in voxels for raised foundations
    #[arg(long, default_value_t = 3)]
    building_pad_radius: i32,
    /// Disable building pads (no raised foundations)
    #[arg(long, default_value_t = false)]
    no_building_pad: bool,
    /// Generate near-flat tile heights (useful for debugging seams)
    #[arg(long, default_value_t = false)]
    flat: bool,
    /// Number of tiles in X direction (overrides radius if > 0)
    #[arg(long, default_value_t = 0)]
    tile_width: u32,
    /// Number of tiles in Y direction (overrides radius if > 0)
    #[arg(long, default_value_t = 0)]
    tile_height: u32,

    #[arg(long = "voxels-per-tile", default_value_t = 128)]
    voxel_resolution: u32,

    #[arg(long = "meters-per-voxel", default_value = "1.25")]
    meters_per_voxel: f64,

    #[arg(long = "max-height-voxels", default_value_t = 1024)]
    max_height_voxels: u32,

    #[arg(long, default_value_t = 1337)]
    seed: u64,

    #[arg(long, default_value_t = 500.0)]
    water_level: f64,

    /// Target height difference from lowest to highest point (approximate)
    #[arg(long, default_value_t = 470.0)]
    height_range: f64,

    /// Sample heights across the whole world grid instead of per-tile sampling
    #[arg(long, default_value_t = false)]
    sample_global: bool,

    #[arg(long = "output-name", default_value = "world_file")]
    output_name: String,

    #[arg(long, value_enum, default_value_t = OutputFormat::Vhc)]
    format: OutputFormat,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum OutputFormat {
    Txt,
    Vhc,
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
    #[allow(dead_code)]
    water: Vec<Polygon>,
    buildings: Vec<BuildingEntry>,
    biome: Biome,
    stats: TileStats,
    // Optional, precomputed height field for the tile (meters)
    heights_m: Option<Vec<f64>>,
    // Optional per-cell base ground heights (voxel Y levels), used to ensure pad continuity.
    base_ground_vox: Option<Vec<i64>>,
    // Neighbor biomes for organic transitions
    neighbor_biomes: Option<NeighborBiomes>,
}

type Polygon = Vec<[f64; 2]>;

#[derive(Clone, Debug)]
struct NeighborBiomes {
    north: Option<Biome>,
    south: Option<Biome>,
    east: Option<Biome>,
    west: Option<Biome>,
}

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
    Beach,
    Jungle,
}

// Material Indices (matching palette.txt)
const MAT_GRASS_DARK: usize = 8;
const MAT_GRASS_LIGHT: usize = 9;
const MAT_DIRT: usize = 10;
const MAT_STONE: usize = 11;
const MAT_SAND: usize = 12;
const MAT_SNOW: usize = 13;
#[allow(dead_code)]
const MAT_WATER_DEEP: usize = 14;
#[allow(dead_code)]
const MAT_WATER_SHALLOW: usize = 15;

const MAT_ASPHALT: usize = 16;
const MAT_CONCRETE: usize = 17;
const MAT_PAVEMENT: usize = 18;
#[allow(dead_code)]
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
const MAT_PALM_FROND_DARK: usize = 38;
const MAT_PALM_FROND_MID: usize = 39;

// Some palette constants are intentionally present but not always used by the
// generator; mark them to suppress "dead_code" warnings while keeping them
// available for palette completeness.
#[allow(dead_code)]
const MAT_LIGHT_WARM: usize = 40; // Becomes 41 after +1 offset
                                  // Some palette constants are intentionally present but not always used by the
                                  // generator; mark them to suppress "dead_code" warnings while keeping them
                                  // available for palette completeness.
#[allow(dead_code)]
const MAT_LIGHT_COOL: usize = 41; // Becomes 42 after +1 offset
const MAT_WINDOW_WARM: usize = 42; // Becomes 43 after +1 offset
#[allow(dead_code)]
const MAT_WINDOW_COOL: usize = 43; // Becomes 44 after +1 offset
                                   // Neon materials are present in palette but not always used by the generator.
#[allow(dead_code)]
const MAT_NEON_RED: usize = 44; // Becomes 45 after +1 offset
#[allow(dead_code)]
const MAT_NEON_GREEN: usize = 45; // Becomes 46 after +1 offset
                                  // MAT_NEON_BLUE is currently defined for palette completeness but not used in generator
#[allow(dead_code)]
const MAT_NEON_BLUE: usize = 46; // Becomes 47 after +1 offset
const MAT_DUNE_GRASS: usize = 47; // Becomes 48 after +1 offset
const MAT_GRASS_COASTAL: usize = 48; // Becomes 49 after +1 offset

// Jungle parameters (tweakable)
const JUNGLE_MIN_TREES: usize = 10;
const JUNGLE_MAX_TREES: usize = 15;
const JUNGLE_FROND_LENGTH: i64 = 8; // length of palm fronds in voxels
const JUNGLE_FROND_COUNT: usize = 6; // number of fronds per palm top
const JUNGLE_SPLIT_SEPARATION: usize = 12; // min separation between large jungle trees

#[derive(Clone, Debug)]
struct TileFetcher {
    seed: u64,
    water_level: f64,
    height_scale: f64,
}

impl TileFetcher {
    fn new(seed: u64, water_level: f64, height_scale: f64) -> Self {
        Self {
            seed,
            water_level,
            height_scale,
        }
    }

    fn fetch(&self, tile: TileId) -> TileData {
        let mut rng = StdRng::seed_from_u64(stable_mix(&[
            tile.x as u64,
            tile.y as u64,
            tile.z as u64,
            self.seed,
        ]));

        // Determine Biome based on terrain height across the tile
        let ((min_lon, max_lon), (north_lat, south_lat)) = tile.lon_lat_bounds();

        // Use a temporary Perlin for biome selection (must match sample_tile_heights seed)
        let perlin = Perlin::new(self.seed as u32);

        // Sample heights at multiple points across the tile to get min/avg height
        let sample_grid = 4;
        let mut min_h = f64::INFINITY;
        let mut sum_h = 0.0;
        for si in 0..sample_grid {
            for sj in 0..sample_grid {
                let frac_x = (si as f64 + 0.5) / sample_grid as f64;
                let frac_y = (sj as f64 + 0.5) / sample_grid as f64;
                let sample_lon = min_lon + (max_lon - min_lon) * frac_x;
                let sample_lat = south_lat + (north_lat - south_lat) * frac_y;
                let (_sx_m, _sy_m) = lon_lat_to_mercator_meters(sample_lon, sample_lat);
                let (sx_m, sy_m) = lon_lat_to_mercator_meters(sample_lon, sample_lat);
                // Create a temporary TileSpace to pass parameters
                let temp_space = TileSpace::new(
                    tile,
                    1,   // dummy resolution
                    1.0, // dummy meters_per_voxel
                    self.seed,
                    self.water_level,
                    self.height_scale,
                );
                let h = get_global_height(&perlin, sx_m, sy_m, &temp_space);
                min_h = min_h.min(h);
                sum_h += h;
            }
        }
        let avg_h = sum_h / (sample_grid * sample_grid) as f64;

        // Use water level from the fetcher (passed from Args)
        let water_level = self.water_level;

        // Use random noise for biome selection (City vs Hill for above-water areas)
        let biome_noise = rng.gen::<f64>();

        // Only allow City biome if the tile's minimum height is well above water level
        // This prevents cities from being placed on terrain that would be partially underwater
        // Require min_h to be at least 50m * scale above water to ensure no artificial ground raising
        let threshold_min = 50.0 * self.height_scale;
        let threshold_avg = 80.0 * self.height_scale;
        let biome = if min_h < water_level + threshold_min {
            // Tile has areas very close to water level - Beach biome
            Biome::Beach
        } else if avg_h < water_level + threshold_avg {
            // Tile is low-lying but above beach threshold - prefer Hill/Jungle biome for more natural look
            if rng.gen_bool(0.3) {
                Biome::Jungle
            } else {
                Biome::Hill
            }
        } else if biome_noise < 0.4 {
            // Random selection for higher terrain
            if rng.gen_bool(0.4) {
                Biome::Jungle
            } else {
                Biome::Hill
            }
        } else {
            Biome::City
        };

        eprintln!(
            "Tile ({},{},{}): min_h={:.1}, avg_h={:.1}, water={:.1}, biome={:?}",
            tile.x, tile.y, tile.z, min_h, avg_h, water_level, biome
        );

        // If Lake or Hill, we might skip city generation logic or adapt it.
        // For now, we'll keep the city generation but maybe clear it if it's a pure nature biome,
        // or we can mix them. Let's make Lake and Hill distinct for now.

        if biome != Biome::City && biome != Biome::Beach {
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
                neighbor_biomes: None,
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
            neighbor_biomes: None,
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
    if max.abs() < 1e-12 {
        0.0
    } else {
        total / max
    }
}

fn smooth_tiles_pass(smoothed_map: &mut HashMap<TileId, Vec<f64>>, size: usize) {
    let keys: Vec<TileId> = smoothed_map.keys().copied().collect();
    for tile_id in keys {
        if let Some(base) = smoothed_map.get(&tile_id) {
            if base.len() != size * size {
                continue;
            }
            let mut new_h = base.clone();
            for xi in 0..size {
                for zi in 0..size {
                    let idx = xi + zi * size;
                    let mut sum = base[idx];
                    let mut count = 1.0;
                    // neighbor tiles for edge smoothing
                    if xi == 0 {
                        let neighbor = TileId {
                            z: tile_id.z,
                            x: tile_id.x - 1,
                            y: tile_id.y,
                        };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[(size - 1) + zi * size];
                            count += 1.0;
                        }
                    }
                    if xi == size - 1 {
                        let neighbor = TileId {
                            z: tile_id.z,
                            x: tile_id.x + 1,
                            y: tile_id.y,
                        };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[0 + zi * size];
                            count += 1.0;
                        }
                    }
                    if zi == 0 {
                        let neighbor = TileId {
                            z: tile_id.z,
                            x: tile_id.x,
                            y: tile_id.y + 1, // South neighbor (tile.y increases southward)
                        };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[xi + (size - 1) * size];
                            count += 1.0;
                        }
                    }
                    if zi == size - 1 {
                        let neighbor = TileId {
                            z: tile_id.z,
                            x: tile_id.x,
                            y: tile_id.y - 1, // North neighbor (tile.y decreases northward)
                        };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[xi + 0 * size];
                            count += 1.0;
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
            if base.len() != size * size {
                continue;
            }
            let mut new_h = base.clone();
            // blend each side with neighbor
            // left side x=0 blends with neighbor's x=size-1
            let left_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x - 1,
                y: tile_id.y,
            };
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
            let right_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x + 1,
                y: tile_id.y,
            };
            if let Some(nei) = smoothed_map.get(&right_neighbor) {
                for w in 0..blend_width {
                    let x = size - 1 - w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for zi in 0..size {
                        let idx = x + zi * size;
                        let idx_nei = w + zi * size;
                        let a = base[idx];
                        let b = nei[idx_nei];
                        new_h[idx] = a * (1.0 - t) + b * t;
                    }
                }
            }
            // north/south blend (tile.y increases southward in slippy tile convention)
            // South neighbor (tile.y + 1) blends with our south edge (zi = 0)
            let south_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y + 1,
            };
            if let Some(nei) = smoothed_map.get(&south_neighbor) {
                for w in 0..blend_width {
                    let z = w; // Our south edge (low z)
                    let z_nei = size - 1 - w; // Their north edge (high z)
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for xi in 0..size {
                        let idx = xi + z * size;
                        let idx_nei = xi + z_nei * size;
                        let a = base[idx];
                        let b = nei[idx_nei];
                        new_h[idx] = a * (1.0 - t) + b * t;
                    }
                }
            }
            // North neighbor (tile.y - 1) blends with our north edge (zi = size-1)
            let north_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y - 1,
            };
            if let Some(nei) = smoothed_map.get(&north_neighbor) {
                for w in 0..blend_width {
                    let z = size - 1 - w; // Our north edge (high z)
                    let z_nei = w; // Their south edge (low z)
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for xi in 0..size {
                        let idx = xi + z * size;
                        let idx_nei = xi + z_nei * size;
                        let a = base[idx];
                        let b = nei[idx_nei];
                        new_h[idx] = a * (1.0 - t) + b * t;
                    }
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
            if base.len() != size * size {
                continue;
            }
            let mut new_b = base.clone();
            for xi in 0..size {
                for zi in 0..size {
                    let idx = xi + zi * size;
                    let mut sum = base[idx] as f64;
                    let mut count = 1.0;
                    if xi == 0 {
                        let neighbor = TileId {
                            z: tile_id.z,
                            x: tile_id.x - 1,
                            y: tile_id.y,
                        };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[(size - 1) + zi * size] as f64;
                            count += 1.0;
                        }
                    }
                    if xi == size - 1 {
                        let neighbor = TileId {
                            z: tile_id.z,
                            x: tile_id.x + 1,
                            y: tile_id.y,
                        };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[0 + zi * size] as f64;
                            count += 1.0;
                        }
                    }
                    if zi == 0 {
                        let neighbor = TileId {
                            z: tile_id.z,
                            x: tile_id.x,
                            y: tile_id.y + 1, // South neighbor
                        };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[xi + (size - 1) * size] as f64;
                            count += 1.0;
                        }
                    }
                    if zi == size - 1 {
                        let neighbor = TileId {
                            z: tile_id.z,
                            x: tile_id.x,
                            y: tile_id.y - 1, // North neighbor
                        };
                        if let Some(nei) = smoothed_map.get(&neighbor) {
                            sum += nei[xi + 0 * size] as f64;
                            count += 1.0;
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
            if base.len() != size * size {
                continue;
            }
            let mut new_b = base.clone();
            let left_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x - 1,
                y: tile_id.y,
            };
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
            let right_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x + 1,
                y: tile_id.y,
            };
            if let Some(nei) = smoothed_map.get(&right_neighbor) {
                for w in 0..blend_width {
                    let x = size - 1 - w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for zi in 0..size {
                        let idx = x + zi * size;
                        let idx_nei = w + zi * size;
                        let a = base[idx] as f64;
                        let b = nei[idx_nei] as f64;
                        new_b[idx] = (a * (1.0 - t) + b * t).round() as i64;
                    }
                }
            }
            // Slippy tiles: tile.y increases southward, so z==0 is south edge
            let top_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y + 1, // South neighbor (z==0 edge)
            };
            if let Some(nei) = smoothed_map.get(&top_neighbor) {
                for w in 0..blend_width {
                    let z = w;
                    let z_nei = size - 1 - w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for xi in 0..size {
                        let idx = xi + z * size;
                        let idx_nei = xi + z_nei * size;
                        let a = base[idx] as f64;
                        let b = nei[idx_nei] as f64;
                        new_b[idx] = (a * (1.0 - t) + b * t).round() as i64;
                    }
                }
            }
            let bottom_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y - 1, // North neighbor (z==size-1 edge)
            };
            if let Some(nei) = smoothed_map.get(&bottom_neighbor) {
                for w in 0..blend_width {
                    let z = size - 1 - w;
                    let z_nei = w;
                    let t = (w + 1) as f64 / (blend_width + 1) as f64;
                    for xi in 0..size {
                        let idx = xi + z * size;
                        let idx_nei = xi + z_nei * size;
                        let a = base[idx] as f64;
                        let b = nei[idx_nei] as f64;
                        new_b[idx] = (a * (1.0 - t) + b * t).round() as i64;
                    }
                }
            }
            smoothed_map.insert(tile_id, new_b);
        }
    }
}

// Enforce identical values across shared tile edges by averaging the integer values
// This avoids 1-voxel seams due to rounding inconsistencies.
// Enforce identical values across shared tile edges by averaging integer values
// across an adjustable width in case pad updates or smoothing cause larger transitions.
// This avoids narrow 1-voxel seams by equalizing a wider band along edges.
fn equalize_tile_edges(
    smoothed_map: &mut HashMap<TileId, Vec<i64>>,
    size: usize,
    width: usize,
) -> usize {
    let keys: Vec<TileId> = smoothed_map.keys().copied().collect();
    let mut updates: Vec<(TileId, usize, i64)> = Vec::new();
    for tile_id in keys.iter() {
        if let Some(base) = smoothed_map.get(tile_id) {
            // Left neighbor: our x=0..width-1 matches neighbor x=size-1 - (0..width-1)
            let left_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x - 1,
                y: tile_id.y,
            };
            if let Some(nei) = smoothed_map.get(&left_neighbor) {
                for w in 0..width {
                    let idx_x = w;
                    let idx_nei_x = size - 1 - w;
                    for zi in 0..size {
                        let idx = idx_x + zi * size;
                        let idx_nei = idx_nei_x + zi * size;
                        let a = base[idx];
                        let b = nei[idx_nei];
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx, avg));
                        updates.push((left_neighbor, idx_nei, avg));
                    }
                }
            }
            // Top neighbor (south): our z=0 matches neighbor z=size-1
            let top_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y + 1,
            };
            if let Some(nei) = smoothed_map.get(&top_neighbor) {
                for w in 0..width {
                    let idx_z = w;
                    let idx_nei_z = size - 1 - w;
                    for xi in 0..size {
                        let idx = xi + idx_z * size;
                        let idx_nei = xi + idx_nei_z * size;
                        let a = base[idx];
                        let b = nei[idx_nei];
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx, avg));
                        updates.push((top_neighbor, idx_nei, avg));
                    }
                }
            }
        }
    }
    let mut applied = 0usize;
    // apply updates
    for (tile, idx, val) in updates.iter() {
        if let Some(v) = smoothed_map.get_mut(tile) {
            if *idx < v.len() {
                if v[*idx] != *val {
                    v[*idx] = *val;
                    applied += 1;
                }
            }
        }
    }
    applied
}

// Clamp small per-edge deltas (e.g., 1-voxel) down to exact equality by averaging only
// where the gap is small. This prevents one-voxel ridges due to rounding noise.
fn clamp_small_edge_deltas(smoothed_map: &mut HashMap<TileId, Vec<i64>>, size: usize) -> usize {
    let keys: Vec<TileId> = smoothed_map.keys().copied().collect();
    let mut updates: Vec<(TileId, usize, i64)> = Vec::new();
    for tile_id in keys.iter() {
        if let Some(base) = smoothed_map.get(tile_id) {
            // left neighbor (our x==0 vs neighbor x==size-1)
            let left_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x - 1,
                y: tile_id.y,
            };
            if let Some(nei) = smoothed_map.get(&left_neighbor) {
                for zi in 0..size {
                    let idx = 0 + zi * size;
                    let idx_nei = (size - 1) + zi * size;
                    let a = base[idx];
                    let b = nei[idx_nei];
                    let delta = (a - b).abs();
                    if delta <= 1 {
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx, avg));
                        updates.push((left_neighbor, idx_nei, avg));
                    }
                }
            }
            // north neighbor (tile.y -1) (our z==size-1 vs neighbor z==0)
            let north_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y - 1,
            };
            if let Some(nei) = smoothed_map.get(&north_neighbor) {
                for xi in 0..size {
                    let idx = xi + (size - 1) * size;
                    let idx_nei = xi + 0 * size;
                    let a = base[idx];
                    let b = nei[idx_nei];
                    let delta = (a - b).abs();
                    if delta <= 1 {
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx, avg));
                        updates.push((north_neighbor, idx_nei, avg));
                    }
                }
            }

            // south neighbor (our z==0 vs neighbor z==size-1)
            let south_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y + 1,
            };
            if let Some(nei) = smoothed_map.get(&south_neighbor) {
                for xi in 0..size {
                    let idx = xi + 0 * size;
                    let idx_nei = xi + (size - 1) * size;
                    let a = base[idx];
                    let b = nei[idx_nei];
                    let delta = (a - b).abs();
                    if delta <= 1 {
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx, avg));
                        updates.push((south_neighbor, idx_nei, avg));
                    }
                }
            }

            // right neighbor (our x==size-1 vs neighbor x==0)
            let right_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x + 1,
                y: tile_id.y,
            };
            if let Some(nei) = smoothed_map.get(&right_neighbor) {
                for zi in 0..size {
                    let idx = (size - 1) + zi * size;
                    let idx_nei = 0 + zi * size;
                    let a = base[idx];
                    let b = nei[idx_nei];
                    let delta = (a - b).abs();
                    if delta <= 1 {
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx, avg));
                        updates.push((right_neighbor, idx_nei, avg));
                    }
                }
            }
        }
    }
    let mut applied = 0usize;
    for (tile, idx, val) in updates.iter() {
        if let Some(v) = smoothed_map.get_mut(tile) {
            if *idx < v.len() {
                if v[*idx] != *val {
                    v[*idx] = *val;
                    applied += 1;
                }
            }
        }
    }
    applied
}

fn get_global_height(perlin: &Perlin, x: f64, y: f64, space: &TileSpace) -> f64 {
    // Large scale terrain for mountains/lakes
    // Base is above water level so more terrain is above water.
    // We scale the base offset by height_scale as well to keep proportions similar,
    // or we can keep it fixed relative to water.
    // Let's make it relative to water level + some base offset scaled.
    let base = space.water_level_m + 80.0 * space.height_scale;

    // Very large scale features (continents/ranges)
    let large = fbm(perlin, x * 0.0002, y * 0.0002, 4, 2.0, 0.5) * 350.0;

    // Medium scale features (hills/valleys)
    let mid = fbm(perlin, x * 0.001, y * 0.001, 4, 2.0, 0.5) * 100.0;

    // Detail
    let detail = fbm(perlin, x * 0.005, y * 0.005, 3, 2.0, 0.5) * 20.0;

    base + (large + mid + detail) * space.height_scale
}

fn sample_tile_heights(perlin: &Perlin, space: &TileSpace) -> Vec<f64> {
    let size = space.voxel_resolution as usize;
    let mut heights = vec![0.0f64; size * size];
    for xi in 0..size {
        for zi in 0..size {
            // Sample at cell centers (xi + 0.5) / size -> consistent with mercator->voxel mapping
            let wx =
                space.min_x_m + ((xi as f64 + 0.5) / size as f64) * (space.max_x_m - space.min_x_m);
            let wz =
                space.min_y_m + ((zi as f64 + 0.5) / size as f64) * (space.max_y_m - space.min_y_m);
            heights[xi + zi * size] = get_global_height(perlin, wx, wz, space).max(1.0);
        }
    }
    heights
}

fn compute_river_mask(
    space: &TileSpace,
    perlin: &Perlin,
    heights_m: &[f64],
    river_scale: f64,
    threshold: f64,
) -> Vec<bool> {
    let size = space.voxel_resolution as usize;
    let mut mask = vec![false; size * size];
    let max_river_elevation = space.water_level_m + 40.0;
    for xi in 0..size {
        for zi in 0..size {
            // Sample at cell centers to be consistent with polygon cell mapping
            let wx =
                space.min_x_m + ((xi as f64 + 0.5) / size as f64) * (space.max_x_m - space.min_x_m);
            let wz =
                space.min_y_m + ((zi as f64 + 0.5) / size as f64) * (space.max_y_m - space.min_y_m);
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
    height_scale: f64,
    /// Global minimum Y voxel level - voxels below this are not generated
    base_y_vox: i64,
}

impl TileSpace {
    fn new(
        tile: TileId,
        voxel_resolution: u32,
        meters_per_voxel: f64,
        seed: u64,
        water_level_m: f64,
        height_scale: f64,
    ) -> Self {
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
            height_scale,
            base_y_vox: 0, // Will be set later after global min is computed
        }
    }

    fn mercator_to_voxel(&self, x_m: f64, y_m: f64) -> (i32, i32) {
        let rel_x = (x_m - self.min_x_m) / (self.max_x_m - self.min_x_m);
        let rel_y = (y_m - self.min_y_m) / (self.max_y_m - self.min_y_m);
        let mut vx = (rel_x * self.voxel_resolution as f64).floor() as i32;
        let mut vz = (rel_y * self.voxel_resolution as f64).floor() as i32;
        vx = vx.clamp(0, self.voxel_resolution as i32 - 1);
        vz = vz.clamp(0, self.voxel_resolution as i32 - 1);
        // Flip Z coordinate so that adjacent tiles have matching edges:
        // zi=0 (south) → high voxel z, zi=size-1 (north) → low voxel z
        let vz_flipped = (self.voxel_resolution as i32 - 1) - vz;
        (vx, vz_flipped)
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
    pad_radius: i32,
    no_building_pad: bool,
) -> TileVoxelResult {
    match data.biome {
        Biome::City => voxelize_city(
            tile,
            data,
            space,
            max_height_voxels,
            pad_radius,
            no_building_pad,
        ),
        Biome::Hill => voxelize_hill(tile, data, space, max_height_voxels),
        Biome::Beach => voxelize_beach(tile, data, space, max_height_voxels),
        Biome::Jungle => voxelize_jungle(tile, data, space, max_height_voxels),
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
    // Use precomputed tile heights if provided (already smoothed by generate_area), otherwise compute locally
    let size = space.voxel_resolution as usize;
    let mut heights_m: Vec<f64> = if let Some(ref pre) = data.heights_m {
        pre.clone()
    } else {
        sample_tile_heights(&Perlin::new(space.seed as u32), space)
    };
    let perlin = Perlin::new(space.seed as u32);

    // Note: Heights are already smoothed across tile boundaries in generate_area,
    // so we skip the per-tile smoothing that was previously here.

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
    // voxel_z is flipped: zi=0 (south in sampling) → high voxel.z, zi=size-1 (north) → low voxel.z
    // This ensures that when tiles are stacked (tile.y increases southward), adjacent tile edges match:
    // - Our tile's zi=0 (south) → voxel.z=size-1 → world Z = tile_offset + size-1
    // - South neighbor's zi=size-1 (north) → voxel.z=0 → world Z = tile_offset + size + 0
    let perlin_transition = Perlin::new((space.seed + 999) as u32);
    for xi in 0..size {
        for zi in 0..size {
            // Determine actual biome at this position
            let actual_biome = if let Some(ref neighbors) = data.neighbor_biomes {
                get_biome_at_position(
                    data.biome,
                    neighbors,
                    xi,
                    0,
                    zi,
                    size,
                    &perlin_transition,
                    (tile.x, tile.y),
                )
            } else {
                data.biome
            };
            let voxel_z = (size - 1 - zi) as i64; // Flip Z coordinate
            let h_m = heights_m[xi + zi * size];
            let h_vox = (h_m / space.meters_per_voxel).round() as i64;
            let h_vox = h_vox.clamp(1, max_height_voxels as i64);

            // approximate slope: sample neighbors
            let center = heights_m[xi + zi * size];
            let neighbor_x = if xi + 1 < size {
                heights_m[(xi + 1) + zi * size]
            } else {
                center
            };
            let neighbor_z = if zi + 1 < size {
                heights_m[xi + (zi + 1) * size]
            } else {
                center
            };
            let slope_x = (neighbor_x - center).abs();
            let slope_z = (neighbor_z - center).abs();
            let slope = (slope_x + slope_z) * 0.5 / space.meters_per_voxel;

            // Start from base_y_vox instead of 0 to avoid generating deep underground voxels
            for y in space.base_y_vox..h_vox {
                let mat = if slope > 3.0 {
                    // steep cliff - expose stone
                    if y < h_vox - 1 {
                        MAT_STONE
                    } else {
                        MAT_STONE
                    }
                } else if y < h_vox - 4 {
                    MAT_STONE
                } else if y < h_vox - 1 {
                    MAT_DIRT
                } else {
                    // topmost
                    // topmost
                    match actual_biome {
                        Biome::Hill => {
                            if h_m > 140.0 {
                                MAT_SNOW
                            } else {
                                // Use muted coastal grass to harmonize with beach
                                MAT_GRASS_COASTAL
                            }
                        }
                        Biome::Beach => MAT_SAND,
                        Biome::City => MAT_CONCRETE,
                        Biome::Jungle => MAT_GRASS_COASTAL,
                    }
                };
                voxels.push(VoxelRecord {
                    x: xi as i64,
                    y,
                    z: voxel_z,
                    material_index: mat,
                });
            }
            // Water is rendered by the viewer as a translucent plane - do NOT generate water voxels!
            // Just mark shoreline/underwater terrain with sand
            // Use round() to match other voxel conversions and reduce differences
            let water_level_vox = (space.water_level_m / space.meters_per_voxel).round() as i64;
            if h_vox <= water_level_vox + 2 {
                // Near or below water - use sand for beaches and seabed
                if let Some(last) = voxels.last_mut() {
                    if last.x == xi as i64 && last.z == voxel_z && last.y == h_vox {
                        last.material_index = MAT_SAND;
                    }
                }
            }

            if river_mask[xi + zi * size] {
                // River areas - just mark with sand, don't add water voxels
                // (water is rendered by the viewer)
                if let Some(last) = voxels.last_mut() {
                    if last.x == xi as i64 && last.z == voxel_z && last.y == h_vox {
                        last.material_index = MAT_SAND;
                    }
                }
            }
        }
    }

    // Vegetation clusters: we pick seeds based on noise and place small clusters where slope is gentle
    let mut rng = StdRng::seed_from_u64(stable_mix(&[
        tile.x as u64,
        tile.y as u64,
        tile.z as u64,
        0xDEADBEEF,
    ]));
    let veg_density = 0.02 + (rng.gen::<f64>() * 0.04);
    for xi in 0..size {
        for zi in 0..size {
            // Recalculate actual_biome for vegetation placement
            let actual_biome = if let Some(ref neighbors) = data.neighbor_biomes {
                get_biome_at_position(
                    data.biome,
                    neighbors,
                    xi,
                    0,
                    zi,
                    size,
                    &perlin_transition,
                    (tile.x, tile.y),
                )
            } else {
                data.biome
            };

            // Only place Hill vegetation if the local biome is actually Hill
            if actual_biome != Biome::Hill {
                continue;
            }

            let voxel_z = (size - 1 - zi) as i64; // Flip Z coordinate
            if rng.gen_bool(veg_density) {
                let h_m = heights_m[xi + zi * size];
                // Skip vegetation if cell is below (or very close to) global water level
                if h_m <= space.water_level_m + 0.5 {
                    continue;
                }
                let h_vox = (h_m / space.meters_per_voxel).round() as i64;
                // Only place vegetation on gentle slopes and not too high
                let center = heights_m[xi + zi * size];
                let neighbor_x = if xi + 1 < size {
                    heights_m[(xi + 1) + zi * size]
                } else {
                    center
                };
                let neighbor_z = if zi + 1 < size {
                    heights_m[xi + (zi + 1) * size]
                } else {
                    center
                };
                let slope_x = (neighbor_x - center).abs();
                let slope_z = (neighbor_z - center).abs();
                let slope = (slope_x + slope_z) * 0.5 / space.meters_per_voxel;
                if slope < 1.1 && h_vox + 6 < max_height_voxels as i64 {
                    // Tree probability decreases with altitude
                    let p = 0.5 - (h_m / 500.0);
                    if rng.gen_bool(p.max(0.05)) {
                        let trunk_h = rng.gen_range(3..6);
                        for ty in 0..trunk_h {
                            voxels.push(VoxelRecord {
                                x: xi as i64,
                                y: h_vox + ty,
                                z: voxel_z,
                                material_index: MAT_TRUNK_DARK,
                            });
                        }
                        let canopy_base = h_vox + trunk_h - 1;
                        for cx in -2..=2 {
                            for cz in -2..=2 {
                                for cy_offset in 0..=3 {
                                    let cy = canopy_base + cy_offset;
                                    if cy >= h_vox && cy < max_height_voxels as i64 {
                                        let dist = (cx as i32).abs()
                                            + (cz as i32).abs()
                                            + (cy_offset as i32).abs();
                                        if dist <= 3 {
                                            // Some trees are evergreen, some are autumnal or with lighter canopy
                                            let leaf_choice = if rng.gen_bool(0.12) {
                                                MAT_LEAVES_AUTUMN
                                            } else if rng.gen_bool(0.25) {
                                                MAT_LEAVES_LIGHT
                                            } else {
                                                MAT_LEAVES_DARK
                                            };
                                            voxels.push(VoxelRecord {
                                                x: (xi as i64 + cx).clamp(0, size as i64 - 1),
                                                y: cy,
                                                z: (voxel_z - cz).clamp(0, size as i64 - 1),
                                                material_index: leaf_choice,
                                            });
                                        }
                                        // Occasionally place a small rock boulder near trees
                                        if rng.gen_bool(0.02) {
                                            let bx = (xi as i64 + rng.gen_range(-1..=1))
                                                .clamp(0, size as i64 - 1)
                                                as i64;
                                            let bz = (voxel_z - rng.gen_range(-1..=1))
                                                .clamp(0, size as i64 - 1)
                                                as i64;
                                            let by = h_vox - 1;
                                            for rx in -1..=0 {
                                                for rz in -1..=0 {
                                                    let xrx = (bx + rx).clamp(0, size as i64 - 1);
                                                    let zrz = (bz - rz).clamp(0, size as i64 - 1);
                                                    voxels.push(VoxelRecord {
                                                        x: xrx,
                                                        y: by,
                                                        z: zrz,
                                                        material_index: MAT_STONE,
                                                    });
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

fn voxelize_jungle(
    tile: TileId,
    data: TileData,
    space: &TileSpace,
    max_height_voxels: u32,
) -> TileVoxelResult {
    let mut voxels = Vec::new();
    let size = space.voxel_resolution as usize;
    let mut heights_m: Vec<f64> = if let Some(ref pre) = data.heights_m {
        pre.clone()
    } else {
        sample_tile_heights(&Perlin::new(space.seed as u32), space)
    };
    let perlin = Perlin::new(space.seed as u32);

    // River mask
    let river_mask = compute_river_mask(space, &perlin, &heights_m, 1.0 / 3000.0, -0.35);
    for xi in 0..size {
        for zi in 0..size {
            let idx = xi + zi * size;
            if river_mask[idx] {
                heights_m[idx] = (heights_m[idx] - 2.0).max(0.5);
            }
        }
    }

    let perlin_transition = Perlin::new((space.seed + 999) as u32);
    for xi in 0..size {
        for zi in 0..size {
            // Determine actual biome
            let actual_biome = if let Some(ref neighbors) = data.neighbor_biomes {
                get_biome_at_position(
                    data.biome,
                    neighbors,
                    xi,
                    0,
                    zi,
                    size,
                    &perlin_transition,
                    (tile.x, tile.y),
                )
            } else {
                data.biome
            };

            let voxel_z = (size - 1 - zi) as i64;
            let h_m = heights_m[xi + zi * size];
            let h_vox = (h_m / space.meters_per_voxel).round() as i64;
            let h_vox = h_vox.clamp(1, max_height_voxels as i64);

            let center = heights_m[xi + zi * size];
            let neighbor_x = if xi + 1 < size {
                heights_m[(xi + 1) + zi * size]
            } else {
                center
            };
            let neighbor_z = if zi + 1 < size {
                heights_m[xi + (zi + 1) * size]
            } else {
                center
            };
            let slope_x = (neighbor_x - center).abs();
            let slope_z = (neighbor_z - center).abs();
            let slope = (slope_x + slope_z) * 0.5 / space.meters_per_voxel;

            for y in space.base_y_vox..h_vox {
                let mat = if slope > 3.0 {
                    MAT_STONE
                } else if y < h_vox - 4 {
                    MAT_STONE
                } else if y < h_vox - 1 {
                    MAT_DIRT
                } else {
                    match actual_biome {
                        Biome::Hill => {
                            // Use perlin noise for natural variation rather than strict grid bands
                            let noise_scale = 0.07;
                            let world_x = tile.x as f64 * size as f64 + xi as f64;
                            let world_z = tile.y as f64 * size as f64 + zi as f64;
                            let n = perlin.get([world_x * noise_scale, world_z * noise_scale]);
                            if n > 0.15 {
                                MAT_GRASS_LIGHT
                            } else {
                                MAT_GRASS_DARK
                            }
                        }
                        Biome::Beach => MAT_SAND,
                        Biome::City => MAT_CONCRETE,
                        Biome::Jungle => MAT_GRASS_COASTAL,
                    }
                };
                voxels.push(VoxelRecord {
                    x: xi as i64,
                    y,
                    z: voxel_z,
                    material_index: mat,
                });
            }

            let water_level_vox = (space.water_level_m / space.meters_per_voxel).round() as i64;
            if h_vox <= water_level_vox + 2 || river_mask[xi + zi * size] {
                if let Some(last) = voxels.last_mut() {
                    if last.x == xi as i64 && last.z == voxel_z && last.y == h_vox {
                        last.material_index = MAT_SAND;
                    }
                }
            }
        }
    }

    // Dense vegetation for Jungle
    // We'll select a fixed-ish number of canopy trees per tile (10-15) with larger combined canopies
    let mut rng = StdRng::seed_from_u64(stable_mix(&[
        tile.x as u64,
        tile.y as u64,
        tile.z as u64,
        0x504E474C, // JUNGLE
    ]));

    // Collect candidate positions for trees (gentle slope, above water)
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for xi in 0..size {
        for zi in 0..size {
            let h_m = heights_m[xi + zi * size];
            if h_m <= space.water_level_m + 0.5 {
                continue;
            }
            let h_vox = (h_m / space.meters_per_voxel).round() as i64;
            if h_vox + 24 >= max_height_voxels as i64 {
                // avoid extremely tall or off-grid trees
                continue;
            }
            // Gentle slope only
            let center = heights_m[xi + zi * size];
            let neighbor_x = if xi + 1 < size {
                heights_m[(xi + 1) + zi * size]
            } else {
                center
            };
            let neighbor_z = if zi + 1 < size {
                heights_m[xi + (zi + 1) * size]
            } else {
                center
            };
            let slope_x = (neighbor_x - center).abs();
            let slope_z = (neighbor_z - center).abs();
            let slope = (slope_x + slope_z) * 0.5 / space.meters_per_voxel;
            if slope >= 1.5 {
                // Too steep
                continue;
            }
            candidates.push((xi, zi));
        }
    }

    // Select a limited number of trees per tile: target 10 - 15
    let desired_trees = rng.gen_range(JUNGLE_MIN_TREES..=JUNGLE_MAX_TREES);
    candidates.shuffle(&mut rng);
    let mut selected_trees: Vec<(usize, usize)> = Vec::new();
    // Keep trees separated to avoid overlapping huge canopies
    for (cx, cz) in candidates.into_iter() {
        if selected_trees.len() >= desired_trees {
            break;
        }
        let mut too_close = false;
        for &(sx, sz) in &selected_trees {
            let dx = (sx as i64 - cx as i64).abs() as usize;
            let dz = (sz as i64 - cz as i64).abs() as usize;
            if dx <= JUNGLE_SPLIT_SEPARATION && dz <= JUNGLE_SPLIT_SEPARATION {
                too_close = true;
                break;
            }
        }
        if !too_close {
            selected_trees.push((cx, cz));
        }
    }

    // Helper: compute distance to nearest edge and whether neighbor biome is not jungle at that edge
    let compute_edge_info = |x: usize, z: usize| -> (usize, bool) {
        let dist_west = x;
        let dist_east = size - 1 - x;
        let dist_south = z;
        let dist_north = size - 1 - z;
        let min_dist = *[dist_west, dist_east, dist_south, dist_north]
            .iter()
            .min()
            .unwrap();
        // Determine if any neighbor biome is NOT Jungle on the closest edge
        let mut edge_different = false;
        if let Some(ref nb) = data.neighbor_biomes {
            if dist_west == min_dist {
                if nb.west != Some(Biome::Jungle) {
                    edge_different = true;
                }
            }
            if dist_east == min_dist {
                if nb.east != Some(Biome::Jungle) {
                    edge_different = true;
                }
            }
            if dist_south == min_dist {
                if nb.south != Some(Biome::Jungle) {
                    edge_different = true;
                }
            }
            if dist_north == min_dist {
                if nb.north != Some(Biome::Jungle) {
                    edge_different = true;
                }
            }
        }
        (min_dist, edge_different)
    };

    for &(xi, zi) in &selected_trees {
        // Calculate actual biome and check
        let actual_biome = if let Some(ref neighbors) = data.neighbor_biomes {
            get_biome_at_position(
                data.biome,
                neighbors,
                xi,
                0,
                zi,
                size,
                &perlin_transition,
                (tile.x, tile.y),
            )
        } else {
            data.biome
        };
        if actual_biome != Biome::Jungle {
            continue;
        }

        let voxel_z = (size - 1 - zi) as i64;
        let h_m = heights_m[xi + zi * size];
        let h_vox = (h_m / space.meters_per_voxel).round() as i64;

        // Determine proximity to tile edge to scale down height/canopy near borders
        let (min_edge_dist, edge_diff) = compute_edge_info(xi, zi);
        // Normalize distance in [0..TRANSITION_WIDTH]; reuse get_biome_at_position TRANSITION_WIDTH
        let transition_width = 16usize;
        let edge_factor = if min_edge_dist >= transition_width {
            1.0
        } else {
            (min_edge_dist as f64) / (transition_width as f64)
        };
        // If neighbor is different biome, reduce height even more
        let edge_reduction = if edge_diff { 0.6 } else { 1.0 };

        // Tall trees (base 12-24m), but apply edge scaling
        let base_trunk_h = rng.gen_range(12..24) as f64;
        let trunk_h_f = (base_trunk_h * edge_factor * edge_reduction).max(6.0);
        let trunk_h = trunk_h_f.round() as i64;

        // Plant main trunk up to split point
        // Choose a split: 1..=3 branches (1 = no split), make splits common
        let split_count = if rng.gen_bool(0.6) {
            2
        } else if rng.gen_bool(0.2) {
            3
        } else {
            1
        };
        let split_height = (trunk_h as f64 * rng.gen_range(0.35..0.65)).round() as i64;

        // Place main trunk blocks up to split_height
        for ty in 0..split_height {
            voxels.push(VoxelRecord {
                x: xi as i64,
                y: h_vox + ty,
                z: voxel_z,
                material_index: MAT_TRUNK_DARK,
            });
        }

        // Create split trunks
        let mut trunk_heads: Vec<(i64, i64, i64)> = Vec::new(); // (x, z, base_y)
        if split_count == 1 {
            trunk_heads.push((xi as i64, voxel_z, h_vox + split_height));
        } else {
            for si in 0..split_count {
                // Angle & offset
                let angle = (si as f64) * (2.0 * std::f64::consts::PI / split_count as f64)
                    + rng.gen_range(-0.4..0.4);
                let ox = (angle.cos() * rng.gen_range(0.0..2.2)).round() as i64;
                let oz = (angle.sin() * rng.gen_range(0.0..2.2)).round() as i64;
                let bx = (xi as i64 + ox).clamp(0, size as i64 - 1);
                let bz = (voxel_z + oz).clamp(0, size as i64 - 1);
                let add_h = rng.gen_range(6..12) as f64;
                let sub_trunk_h = ((trunk_h - split_height) as f64 * rng.gen_range(0.6..1.0)
                    + add_h)
                    .round() as i64;
                // Build the sub-trunk: straight or slightly tilted upward
                let mut tx = bx;
                let mut tz = bz;
                for ty in 0..sub_trunk_h {
                    let py = h_vox + split_height + ty;
                    voxels.push(VoxelRecord {
                        x: tx.clamp(0, size as i64 - 1),
                        y: py,
                        z: tz.clamp(0, size as i64 - 1),
                        material_index: MAT_TRUNK_DARK,
                    });
                    // Slight drift: shift every few steps
                    if rng.gen_bool(0.15) {
                        // drift horizontally by 0 or +-1 on x/z
                        let dx = rng.gen_range(-1..=1);
                        let dz = rng.gen_range(-1..=1);
                        tx = (tx + dx).clamp(0, size as i64 - 1);
                        tz = (tz + dz).clamp(0, size as i64 - 1);
                    }
                }
                trunk_heads.push((tx, tz, h_vox + split_height + sub_trunk_h));
            }
        }

        // Palm-like fronds: long drooping leaves radiating from the top of each trunk
        // Unlike a spherical canopy, palm fronds extend outward and droop downward
        for &(hx, hz, hy_top) in &trunk_heads {
            // Add a small crown at the very top (where fronds emerge)
            for crown_dx in -1i64..=1 {
                for crown_dz in -1i64..=1 {
                    if crown_dx.abs() + crown_dz.abs() <= 1 {
                        let px = (hx + crown_dx).clamp(0, size as i64 - 1);
                        let pz = (hz + crown_dz).clamp(0, size as i64 - 1);
                        voxels.push(VoxelRecord {
                            x: px,
                            y: hy_top,
                            z: pz,
                            material_index: MAT_PALM_FROND_DARK,
                        });
                    }
                }
            }

            // Generate fronds radiating outward and drooping down
            let frond_count = rng.gen_range(JUNGLE_FROND_COUNT..=JUNGLE_FROND_COUNT + 3);
            let base_frond_length =
                rng.gen_range(JUNGLE_FROND_LENGTH - 2..=JUNGLE_FROND_LENGTH + 2);

            for frond_idx in 0..frond_count {
                // Angle around the trunk (evenly distributed with some randomness)
                let base_angle =
                    (frond_idx as f64) * (2.0 * std::f64::consts::PI / frond_count as f64);
                let angle = base_angle + rng.gen_range(-0.3..0.3);

                // Frond length varies slightly
                let frond_length = base_frond_length + rng.gen_range(-1..=1);

                // Trace the frond from trunk top outward and downward
                // Palm fronds curve: start going out and slightly up, then droop down
                for step in 0..frond_length {
                    let t = step as f64 / frond_length as f64; // 0 to 1 along frond

                    // Horizontal distance increases linearly
                    let horiz_dist = (step as f64 + 1.0) * 1.0;

                    // Vertical offset: slight rise at start, then droop (parabolic curve)
                    // y = -a*(t-0.2)^2 + peak at t=0.2, then droop
                    let rise_peak = 1.5;
                    let droop_rate = 4.0;
                    let y_offset =
                        rise_peak * (1.0 - (t - 0.15).powi(2) * droop_rate) - (t * t * 3.0);
                    let y_offset = y_offset.round() as i64;

                    let fx = hx + (angle.cos() * horiz_dist).round() as i64;
                    let fz = hz + (angle.sin() * horiz_dist).round() as i64;
                    let fy = hy_top + y_offset;

                    // Clamp to valid range
                    let px = fx.clamp(0, size as i64 - 1);
                    let pz = fz.clamp(0, size as i64 - 1);

                    if fy >= h_vox && fy < max_height_voxels as i64 {
                        // Check edge proximity for biome transition blending
                        let (dist_to_edge, edge_diff) = compute_edge_info(px as usize, pz as usize);
                        let edge_scale = if dist_to_edge >= transition_width {
                            1.0
                        } else {
                            (dist_to_edge as f64) / (transition_width as f64)
                        };
                        let place_prob = if edge_diff { edge_scale } else { 1.0 };

                        // Check local biome for smoother transitions
                        let local_biome = if let Some(ref neighbors) = data.neighbor_biomes {
                            get_biome_at_position(
                                data.biome,
                                neighbors,
                                px as usize,
                                0,
                                pz as usize,
                                size,
                                &perlin_transition,
                                (tile.x, tile.y),
                            )
                        } else {
                            data.biome
                        };
                        let local_place_prob = if local_biome == Biome::Jungle {
                            place_prob
                        } else {
                            (place_prob * 0.35).max(0.15)
                        };

                        if rng.gen_bool(local_place_prob.max(0.15)) {
                            // Use darker green at base, lighter toward tips
                            let frond_mat = if t < 0.4 {
                                MAT_PALM_FROND_DARK
                            } else if t < 0.7 {
                                MAT_PALM_FROND_MID
                            } else {
                                // Slight variation at tips
                                if rng.gen_bool(0.3) {
                                    MAT_LEAVES_DARK
                                } else {
                                    MAT_PALM_FROND_MID
                                }
                            };

                            voxels.push(VoxelRecord {
                                x: px,
                                y: fy,
                                z: pz,
                                material_index: frond_mat,
                            });

                            // Add some width to the frond (palm leaves have width)
                            if step > 0 && step < frond_length - 1 && rng.gen_bool(0.6) {
                                // Perpendicular direction for frond width
                                let perp_angle = angle + std::f64::consts::PI / 2.0;
                                let side = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                                let wx = fx + (perp_angle.cos() * side).round() as i64;
                                let wz = fz + (perp_angle.sin() * side).round() as i64;
                                let wpx = wx.clamp(0, size as i64 - 1);
                                let wpz = wz.clamp(0, size as i64 - 1);

                                voxels.push(VoxelRecord {
                                    x: wpx,
                                    y: fy,
                                    z: wpz,
                                    material_index: frond_mat,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Add undergrowth: keep undergrowth across tile, but allow smooth transition on edges.
    for xi in 0..size {
        for zi in 0..size {
            let actual_biome = if let Some(ref neighbors) = data.neighbor_biomes {
                get_biome_at_position(
                    data.biome,
                    neighbors,
                    xi,
                    0,
                    zi,
                    size,
                    &perlin_transition,
                    (tile.x, tile.y),
                )
            } else {
                data.biome
            };
            // Allow undergrowth both in jungle and in nearby border cells (blend area)
            let allow_undergrowth = if actual_biome == Biome::Jungle {
                true
            } else {
                // check whether adjacent neighbor is Jungle and we're within blend width
                let blend_width = 8usize;
                let mut near_jungle = false;
                if let Some(ref nb) = data.neighbor_biomes {
                    // If any neighbor is Jungle and this cell is within blend width of that border, allow
                    let dist_west = xi;
                    let dist_east = size - 1 - xi;
                    let dist_south = zi;
                    let dist_north = size - 1 - zi;
                    if nb.west == Some(Biome::Jungle) && dist_west < blend_width {
                        near_jungle = true;
                    }
                    if nb.east == Some(Biome::Jungle) && dist_east < blend_width {
                        near_jungle = true;
                    }
                    if nb.south == Some(Biome::Jungle) && dist_south < blend_width {
                        near_jungle = true;
                    }
                    if nb.north == Some(Biome::Jungle) && dist_north < blend_width {
                        near_jungle = true;
                    }
                }
                near_jungle
            };
            if !allow_undergrowth {
                continue;
            }
            let voxel_z = (size - 1 - zi) as i64;
            let h_m = heights_m[xi + zi * size];
            if h_m <= space.water_level_m + 0.5 {
                continue;
            }
            let h_vox = (h_m / space.meters_per_voxel).round() as i64;
            // Undergrowth density varied; blend with border influence
            let base_prob = 0.18; // more undergrowth in jungle
            let (min_edge_dist, _) = compute_edge_info(xi, zi);
            let edge_prob = if min_edge_dist >= 12 {
                1.0
            } else {
                (min_edge_dist as f64) / 12.0
            };
            let probability = (base_prob as f64) * edge_prob as f64;
            if rng.gen_bool(probability.max(0.06)) {
                let bush_h = rng.gen_range(1..=4);
                for by in 0..bush_h {
                    voxels.push(VoxelRecord {
                        x: xi as i64,
                        y: h_vox + by,
                        z: voxel_z,
                        material_index: MAT_LEAVES_MED,
                    });
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

fn voxelize_beach(
    tile: TileId,
    data: TileData,
    space: &TileSpace,
    max_height_voxels: u32,
) -> TileVoxelResult {
    let mut voxels = Vec::new();
    let size = space.voxel_resolution as usize;

    // Use precomputed heights or sample
    let heights_m: Vec<f64> = if let Some(ref pre) = data.heights_m {
        pre.clone()
    } else {
        sample_tile_heights(&Perlin::new(space.seed as u32), space)
    };

    let perlin_transition = Perlin::new((space.seed + 999) as u32);

    // Vegetation RNG
    let mut rng = StdRng::seed_from_u64(stable_mix(&[
        tile.x as u64,
        tile.y as u64,
        tile.z as u64,
        0xBEAC401,
    ]));

    // Voxelize terrain
    for xi in 0..size {
        for zi in 0..size {
            // Determine actual biome at this position
            let actual_biome = if let Some(ref neighbors) = data.neighbor_biomes {
                get_biome_at_position(
                    data.biome,
                    neighbors,
                    xi,
                    0,
                    zi,
                    size,
                    &perlin_transition,
                    (tile.x, tile.y),
                )
            } else {
                data.biome
            };

            let voxel_z = (size - 1 - zi) as i64;
            let h_m = heights_m[xi + zi * size];
            let h_vox = (h_m / space.meters_per_voxel).round() as i64;
            let h_vox = h_vox.clamp(1, max_height_voxels as i64);

            let dist_from_water = h_m - space.water_level_m;

            // Beach terrain: mostly sand with some stone underneath
            for y in space.base_y_vox..h_vox {
                let mat = if y < h_vox - 3 {
                    MAT_STONE // Deep layer
                } else {
                    // Shore zone (0-5m from water) is always sand
                    if dist_from_water < 5.0 {
                        MAT_SAND
                    } else {
                        match actual_biome {
                            Biome::Beach => MAT_SAND,
                            Biome::Hill => MAT_GRASS_COASTAL,
                            Biome::City => MAT_CONCRETE,
                            Biome::Jungle => MAT_GRASS_COASTAL,
                        }
                    }
                };
                voxels.push(VoxelRecord {
                    x: xi as i64,
                    y,
                    z: voxel_z,
                    material_index: mat,
                });
            }

            // Vegetation placement
            // Only place vegetation if we are in the Beach biome (or blending into it)
            if actual_biome == Biome::Beach && h_m > space.water_level_m {
                // Zone 1: Shore (0-5m) - No vegetation, pure sand

                // Zone 2: Dunes (5-15m) - Coastal grass
                if dist_from_water >= 5.0 && dist_from_water < 15.0 {
                    // Clumpy grass distribution
                    // Use a simple noise function or just RNG with clustering
                    if rng.gen_bool(0.15) {
                        let grass_h = rng.gen_range(1..=2);
                        for gy in 0..grass_h {
                            voxels.push(VoxelRecord {
                                x: xi as i64,
                                y: h_vox + gy,
                                z: voxel_z,
                                material_index: MAT_DUNE_GRASS,
                            });
                        }
                    }
                }

                // Zone 3: Inland (>8m) - Sparse Palm Trees
                // Overlaps slightly with dunes, but mostly further back
                if dist_from_water >= 8.0 {
                    // Much lower density (0.3%)
                    if rng.gen_bool(0.003) {
                        // Palm trunk (taller and thinner than regular trees)
                        let trunk_h = rng.gen_range(6..10); // Slightly taller
                        for ty in 0..trunk_h {
                            voxels.push(VoxelRecord {
                                x: xi as i64,
                                y: h_vox + ty,
                                z: voxel_z,
                                material_index: MAT_TRUNK_LIGHT,
                            });
                        }

                        // Palm fronds (drooping style using new colors)
                        let canopy_y = h_vox + trunk_h;

                        // Small crown
                        voxels.push(VoxelRecord {
                            x: xi as i64,
                            y: canopy_y,
                            z: voxel_z,
                            material_index: MAT_PALM_FROND_DARK,
                        });

                        // Generate fronds
                        let frond_count = rng.gen_range(5..=7);
                        for frond_idx in 0..frond_count {
                            let angle = (frond_idx as f64)
                                * (2.0 * std::f64::consts::PI / frond_count as f64);
                            let length = rng.gen_range(5..=7); // Slightly shorter than jungle palms

                            for r in 1..=length {
                                let drop = (r as f64 * 0.5).powf(1.2) as i64;
                                let fx = (xi as f64 + angle.cos() * r as f64).round() as i64;
                                let fz = (voxel_z as f64 + angle.sin() * r as f64).round() as i64;
                                let fy = canopy_y - drop;

                                if fx >= 0 && fx < size as i64 && fz >= 0 && fz < size as i64 {
                                    let mat = if r < 3 {
                                        MAT_PALM_FROND_DARK
                                    } else {
                                        MAT_PALM_FROND_MID
                                    };

                                    voxels.push(VoxelRecord {
                                        x: fx,
                                        y: fy,
                                        z: fz,
                                        material_index: mat,
                                    });
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

// Determine which biome to use at a specific voxel position, using noise for organic transitions
// Determine which biome to use at a specific voxel position, using noise for organic transitions
fn get_biome_at_position(
    base_biome: Biome,
    neighbors: &NeighborBiomes,
    x: usize,
    _y: usize,
    z: usize,
    size: usize,
    perlin: &Perlin,
    tile_pos: (i32, i32),
) -> Biome {
    const TRANSITION_WIDTH: usize = 16; // Voxels from edge where transitions occur

    // Calculate distance from each edge
    let dist_west = x;
    let dist_east = size - 1 - x;
    let dist_south = z;
    let dist_north = size - 1 - z;

    // Sample noise for organic boundary (world coordinates for seamless transitions)
    let noise_scale = 0.05; // Controls transition roughness
    let world_x = tile_pos.0 as f64 * size as f64 + x as f64;
    let world_z = tile_pos.1 as f64 * size as f64 + z as f64;
    let noise = perlin.get([world_x * noise_scale, world_z * noise_scale]);
    // Map noise from [-1, 1] to [0, TRANSITION_WIDTH]
    let noise_offset = ((noise + 1.0) * 0.5 * TRANSITION_WIDTH as f64) as usize;

    // Helper for deterministic edge hashing
    let edge_hash = |u: i32, v: i32| -> u64 { stable_mix(&[u as u64, v as u64, 0xED6E4A54]) };

    // Check each edge for transitions
    // West Edge (neighbor x-1)
    if dist_west < TRANSITION_WIDTH {
        // Rule: hash(tile_x, tile_y) % 2 == 0 => West invades East (Us)
        if edge_hash(tile_pos.0, tile_pos.1) % 2 == 0 {
            if let Some(west_biome) = neighbors.west {
                if dist_west < noise_offset {
                    return west_biome;
                }
            }
        }
    }

    // East Edge (neighbor x+1)
    if dist_east < TRANSITION_WIDTH {
        // Rule: hash(tile_x + 1, tile_y) % 2 != 0 => East invades West (Us)
        if edge_hash(tile_pos.0 + 1, tile_pos.1) % 2 != 0 {
            if let Some(east_biome) = neighbors.east {
                if dist_east < noise_offset {
                    return east_biome;
                }
            }
        }
    }

    // South Edge (neighbor y+1)
    if dist_south < TRANSITION_WIDTH {
        // Rule: hash(tile_x, tile_y + 1) % 2 != 0 => South invades North (Us)
        if edge_hash(tile_pos.0, tile_pos.1 + 1) % 2 != 0 {
            if let Some(south_biome) = neighbors.south {
                if dist_south < noise_offset {
                    return south_biome;
                }
            }
        }
    }

    // North Edge (neighbor y-1)
    if dist_north < TRANSITION_WIDTH {
        // Rule: hash(tile_x, tile_y) % 2 == 0 => North invades South (Us)
        if edge_hash(tile_pos.0, tile_pos.1) % 2 == 0 {
            if let Some(north_biome) = neighbors.north {
                if dist_north < noise_offset {
                    return north_biome;
                }
            }
        }
    }

    base_biome
}

fn voxelize_city(
    tile: TileId,
    data: TileData,
    space: &TileSpace,
    max_height_voxels: u32,
    pad_radius: i32,
    no_building_pad: bool,
) -> TileVoxelResult {
    let mut voxels = Vec::new();
    let perlin = Perlin::new(space.seed as u32);
    let heights_m: Vec<f64> = if let Some(ref pre) = data.heights_m {
        pre.clone()
    } else {
        sample_tile_heights(&perlin, space)
    };

    // Build per-cell base ground from terrain heights
    // Cities should only be placed on high ground, so no artificial raising is needed
    // NOTE: If a precomputed `base_ground_vox` exists in `data`, prefer that so
    // tile rounding is consistent. Otherwise compute it using round().
    let size = space.voxel_resolution as usize;
    let mut base_ground_vox: Vec<i64> = if let Some(ref precomputed) = data.base_ground_vox {
        precomputed.clone()
    } else {
        let mut base_ground_vox = vec![0i64; size * size];
        for xi in 0..size {
            for zi in 0..size {
                let elev_m = heights_m[xi + zi * size];
                // Use round() to match other conversions and to reduce 1-voxel seams
                let ground_vox = (elev_m / space.meters_per_voxel).round() as i64;
                // Store using flipped z coordinate to match polygon_cells coordinate system
                let flipped_z = size - 1 - zi;
                base_ground_vox[xi + flipped_z * size] = ground_vox;
            }
        }
        base_ground_vox
    };

    // Building pad pass: expand raised base around building footprints to create smooth pads
    // Allow pad disabled via no_building_pad flag
    let pad_radius_effective = if no_building_pad { 0 } else { pad_radius };
    for entry in data.buildings.iter() {
        let cells_list = polygon_cells(space, &[entry.footprint.clone()]);
        if cells_list.is_empty() {
            continue;
        }
        let cells = &cells_list[0];
        if cells.is_empty() {
            continue;
        }
        // Compute building foundation base as the maximum base_ground across footprint
        let mut b_base = 0i64;
        for &(x, z) in cells {
            let idx_b = x as usize + z as usize * size;
            if base_ground_vox[idx_b] > b_base {
                b_base = base_ground_vox[idx_b];
            }
        }
        // Expand neighboring ground to create a small pad (2 cells) around footprint for smoother transition
        let pad_radius = pad_radius_effective;
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
    // Start from base_y_vox to avoid generating deep underground voxels
    for xi in 0..size {
        for zi in 0..size {
            let voxel_z = (size - 1 - zi) as i64; // Flip Z coordinate
                                                  // base_ground_vox uses flipped z indexing to match polygon_cells
            let ground_vox = base_ground_vox[xi + voxel_z as usize * size];
            for y in space.base_y_vox..=ground_vox {
                let mat = if y < ground_vox - 2 {
                    MAT_STONE
                } else {
                    MAT_DIRT
                };
                voxels.push(VoxelRecord {
                    x: xi as i64,
                    y,
                    z: voxel_z,
                    material_index: mat,
                });
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
                overrides.push(VoxelRecord {
                    x: x as i64,
                    y: base_y + height_offset,
                    z: z as i64,
                    material_index: voxel_type,
                });
            }
        }
        overrides
    };

    // Apply polygons now that terrain fill is created
    // NOTE: Do NOT add water voxels - water is rendered by the viewer as a translucent plane
    voxels.extend(apply_polygons_per_cell(&data.roads, MAT_ASPHALT, 0));
    if max_height_voxels > 2 {
        voxels.extend(apply_polygons_per_cell(&data.roads, MAT_ASPHALT, 1));
    }
    voxels.extend(apply_polygons_per_cell(&data.parks, MAT_GRASS_DARK, 0));
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
            let _path_rng = StdRng::seed_from_u64(
                space
                    .seed
                    .wrapping_add(stable_mix(&[tile.x as u64, tile.y as u64])),
            );
            for &(x, z) in park_cells.iter() {
                if tree_rng.gen_bool(0.08) {
                    // only place trees if the cell is not below the water level
                    let idx = x as usize + z as usize * size;
                    if heights_m[idx] > space.water_level_m + 0.5 {
                        let base_y = base_ground_vox[idx];
                        voxels.push(VoxelRecord {
                            x: x as i64,
                            y: base_y,
                            z: z as i64,
                            material_index: MAT_TRUNK_LIGHT,
                        });
                        let leaf_choice = if tree_rng.gen_bool(0.25) {
                            MAT_LEAVES_AUTUMN
                        } else {
                            MAT_LEAVES_MED
                        };
                        for h in 1..=canopy_limit {
                            voxels.push(VoxelRecord {
                                x: x as i64,
                                y: (base_y + h as i64),
                                z: z as i64,
                                material_index: leaf_choice,
                            });
                        }
                    }
                    // Small shrub cluster near tree
                    if tree_rng.gen_bool(0.15) {
                        let idx_sh1 = (x + 1) as usize + z as usize * size;
                        let idx_sh2 = (x - 1) as usize + z as usize * size;
                        let base_sh1 = base_ground_vox[idx_sh1];
                        let base_sh2 = base_ground_vox[idx_sh2];
                        voxels.push(VoxelRecord {
                            x: x as i64 + 1,
                            y: base_sh1,
                            z: z as i64,
                            material_index: MAT_LEAVES_LIGHT,
                        });
                        voxels.push(VoxelRecord {
                            x: x as i64 - 1,
                            y: base_sh2,
                            z: z as i64,
                            material_index: MAT_LEAVES_LIGHT,
                        });
                    }
                    // Occasional statue/bench
                    if tree_rng.gen_bool(0.01) {
                        let idxb = (x + 1) as usize + (z + 1) as usize * size;
                        let base_b = base_ground_vox[idxb];
                        voxels.push(VoxelRecord {
                            x: x as i64 + 1,
                            y: base_b,
                            z: z as i64 + 1,
                            material_index: MAT_MARBLE,
                        });
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
                voxels.push(VoxelRecord {
                    x: x as i64,
                    y,
                    z: z as i64,
                    material_index: MAT_PAVEMENT,
                });
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
                        voxels.push(VoxelRecord {
                            x: ox + dx,
                            y: oyf,
                            z: oz + dz,
                            material_index: MAT_PLASTER,
                        });
                    }
                }
                voxels.push(VoxelRecord {
                    x: ox,
                    y: oy + 2,
                    z: oz,
                    material_index: MAT_ROOF_TERRACOTTA,
                });
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
        let height_vox = (height_m / space.meters_per_voxel).round() as i64;
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
            if base_ground_vox[idx_b] > b_base {
                b_base = base_ground_vox[idx_b];
            }
        }
        // Fill foundation under the building footprint
        for &(x, z) in cells {
            let idx_b = x as usize + z as usize * size;
            let ground_vox = base_ground_vox[idx_b];
            for gy in ground_vox..=b_base {
                let mat = if gy < b_base - 1 { MAT_STONE } else { MAT_DIRT };
                voxels.push(VoxelRecord {
                    x: x as i64,
                    y: gy,
                    z: z as i64,
                    material_index: mat,
                });
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
                voxels.push(VoxelRecord {
                    x: x as i64,
                    y: b_base + y,
                    z: z as i64,
                    material_index: material_idx,
                });
            }
            voxels.push(VoxelRecord {
                x: x as i64,
                y: b_base + roof_y,
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
                    y: b_base + roof_y + extra,
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
    eprintln!("=== GENERATE_AREA v3 - DEBUG HEIGHTS ===");
    let center_tile = lon_lat_to_tile(args.center_lon, args.center_lat, args.zoom);
    eprintln!(
        "Center tile: ({}, {}, {})",
        center_tile.x, center_tile.y, center_tile.z
    );
    let height_scale = args.height_range / BASE_HEIGHT_RANGE;
    let fetcher = TileFetcher::new(args.seed, args.water_level, height_scale);
    let mut results = Vec::new();
    let perlin = Perlin::new(args.seed as u32);
    // Collect tile metadata and precompute heights
    let mut tile_map: HashMap<TileId, TileData> = HashMap::new();
    let mut tile_spaces: HashMap<TileId, TileSpace> = HashMap::new();
    let mut tiles: Vec<TileId> = Vec::new();
    // Determine tile grid size (tile_count = 2*radius + 1 by default). Allow explicit overrides
    let tile_w = if args.tile_width > 0 {
        args.tile_width as i32
    } else {
        (args.radius as i32) * 2 + 1
    };
    let tile_h = if args.tile_height > 0 {
        args.tile_height as i32
    } else {
        (args.radius as i32) * 2 + 1
    };
    let start_x = center_tile.x - (tile_w / 2);
    let start_y = center_tile.y - (tile_h / 2);
    let size = args.voxel_resolution as usize;
    for ty in 0..tile_h {
        for tx in 0..tile_w {
            let tile = TileId {
                z: center_tile.z,
                x: start_x + tx,
                y: start_y + ty,
            };
            let mut data = fetcher.fetch(tile);
            let space = TileSpace::new(
                tile,
                args.voxel_resolution,
                args.meters_per_voxel,
                args.seed,
                args.water_level,
                height_scale,
            );
            let heights = sample_tile_heights(&perlin, &space);
            data.heights_m = Some(heights);
            tile_map.insert(tile, data);
            tile_spaces.insert(tile, space);
            tiles.push(tile);
        }
    }

    // Populate neighbor biomes for organic transitions
    for tile_id in &tiles {
        let north_biome = tile_map
            .get(&TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y - 1,
            })
            .map(|t| t.biome);
        let south_biome = tile_map
            .get(&TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y + 1,
            })
            .map(|t| t.biome);
        let east_biome = tile_map
            .get(&TileId {
                z: tile_id.z,
                x: tile_id.x + 1,
                y: tile_id.y,
            })
            .map(|t| t.biome);
        let west_biome = tile_map
            .get(&TileId {
                z: tile_id.z,
                x: tile_id.x - 1,
                y: tile_id.y,
            })
            .map(|t| t.biome);

        if let Some(data) = tile_map.get_mut(tile_id) {
            data.neighbor_biomes = Some(NeighborBiomes {
                north: north_biome,
                south: south_biome,
                east: east_biome,
                west: west_biome,
            });
        }
    }

    // If sampling globally, create a big grid covering full world extents and fill
    // per-tile heights from it. This ensures consistent sampling across tile boundaries.
    if args.sample_global {
        eprintln!("Sampling heights globally across tile grid...");
        // Determine world bounds in meters from tile_spaces
        let mut global_min_x = f64::INFINITY;
        let mut global_max_x = f64::NEG_INFINITY;
        let mut global_min_y = f64::INFINITY;
        let mut global_max_y = f64::NEG_INFINITY;
        for (_tid, space) in tile_spaces.iter() {
            if space.min_x_m < global_min_x {
                global_min_x = space.min_x_m;
            }
            if space.max_x_m > global_max_x {
                global_max_x = space.max_x_m;
            }
            if space.min_y_m < global_min_y {
                global_min_y = space.min_y_m;
            }
            if space.max_y_m > global_max_y {
                global_max_y = space.max_y_m;
            }
        }
        let grid_w = tile_w as usize * size;
        let grid_h = tile_h as usize * size;
        let mut global_heights = vec![0.0f64; grid_w * grid_h];
        for gz in 0..grid_h {
            for gx in 0..grid_w {
                let wx = global_min_x
                    + ((gx as f64 + 0.5) / grid_w as f64) * (global_max_x - global_min_x);
                let wz = global_min_y
                    + ((gz as f64 + 0.5) / grid_h as f64) * (global_max_y - global_min_y);
                // Create a temporary space for global sampling
                let temp_space = TileSpace::new(
                    center_tile, // Use center tile as reference
                    args.voxel_resolution,
                    args.meters_per_voxel,
                    args.seed,
                    args.water_level,
                    height_scale,
                );
                global_heights[gx + gz * grid_w] =
                    get_global_height(&perlin, wx, wz, &temp_space).max(1.0);
            }
        }
        // Copy slices from global_heights into each tile's `heights_m`
        for ty in 0..tile_h {
            for tx in 0..tile_w {
                let tile = TileId {
                    z: center_tile.z,
                    x: start_x + tx,
                    y: start_y + ty,
                };
                let gx0 = (tx as usize) * size;
                let gz0 = (ty as usize) * size;
                if let Some(data) = tile_map.get_mut(&tile) {
                    let mut h = vec![0.0f64; size * size];
                    for zi in 0..size {
                        for xi in 0..size {
                            let gidx = (gx0 + xi) + (gz0 + zi) * grid_w;
                            h[xi + zi * size] = global_heights[gidx];
                        }
                    }
                    data.heights_m = Some(h);
                }
            }
        }
    }

    // Smooth across tile boundaries by averaging edge cells with neighbor's edge cells
    let size = args.voxel_resolution as usize;
    let mut smoothed_map: HashMap<TileId, Vec<f64>> = HashMap::new();
    // Gather pad updates without mutating the tile_map during iteration
    let mut pad_updates: Vec<(TileId, usize, i64)> = Vec::new();
    for (tile_id, data) in tile_map.iter() {
        if let Some(h) = data.heights_m.as_ref() {
            smoothed_map.insert(*tile_id, h.clone());
        }
    }
    // Multiple smoothing passes
    let smoothing_passes = 3usize;
    for _ in 0..smoothing_passes {
        smooth_tiles_pass(&mut smoothed_map, size);
    }
    // Blend tile edges to produce ramps across tile boundaries
    let blend_width = args.tile_blend_width as usize;
    blend_tile_edges(&mut smoothed_map, size, blend_width);
    // Additional smoothing passes to settle blend
    for _ in 0..1 {
        smooth_tiles_pass(&mut smoothed_map, size);
    }
    // Previously this loop performed a single pass; we've replaced it with helper calls

    // Write smoothed heights back to tile data and compute base_ground_vox
    for (tile_id, heights) in smoothed_map.iter() {
        if let Some(data) = tile_map.get_mut(tile_id) {
            data.heights_m = Some(heights.clone());
            // compute base ground vox per cell - use natural terrain height
            let mut base_vox = vec![0i64; size * size];
            for xi in 0..size {
                for zi in 0..size {
                    let elev_m = heights[xi + zi * size];
                    // Use rounding instead of ceil to reduce 1-voxel seams between tiles
                    let g = (elev_m / args.meters_per_voxel).round() as i64;
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
                // compare top edge (North, z=size-1) to neighbor (y-1, South edge z=0)
                let idx_top = xi + (size - 1) * size;
                let top_neighbor = TileId {
                    z: tile_id.z,
                    x: tile_id.x,
                    y: tile_id.y - 1,
                };
                if let Some(nei) = smoothed_map.get(&top_neighbor) {
                    let nei_idx = xi + 0 * size;
                    let delta = (h[idx_top] - nei[nei_idx]).abs();
                    max_border_delta = max_border_delta.max(delta);
                    total_border_delta += delta;
                    border_count += 1;
                }
                // compare left edge
                let idx_left = 0 + xi * size;
                let left_neighbor = TileId {
                    z: tile_id.z,
                    x: tile_id.x - 1,
                    y: tile_id.y,
                };
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
        println!(
            "Height border diffs after smoothing: max = {:.3}m, avg = {:.3}m",
            max_border_delta,
            total_border_delta / border_count as f64
        );
    }

    // Base border diffs diagnostic will be computed after pad smoothing & equalization

    // Building pad pass across tiles: expand elevated bases around building footprints
    // Increase pad radius for smoother, more gradual ramps across neighbors
    let pad_radius = 3i32;
    for (tile_id, data) in tile_map.iter() {
        if let Some(cells) = Some(&data.buildings) {
            for entry in cells.iter() {
                let footprint_cells = polygon_cells(
                    tile_spaces.get(tile_id).unwrap(),
                    &[entry.footprint.clone()],
                );
                if footprint_cells.is_empty() {
                    continue;
                }
                let cells = &footprint_cells[0];
                // compute max base for building footprint
                let mut b_base = 0i64;
                for &(x, z) in cells.iter() {
                    let idx = x as usize + z as usize * size;
                    if let Some(bv) = data.base_ground_vox.as_ref() {
                        if bv[idx] > b_base {
                            b_base = bv[idx];
                        }
                    }
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
                                while cx < 0 {
                                    cx += size as i32;
                                    tx -= 1;
                                }
                                while cx >= size as i32 {
                                    cx -= size as i32;
                                    tx += 1;
                                }
                                while cz < 0 {
                                    cz += size as i32;
                                    ty -= 1;
                                }
                                while cz >= size as i32 {
                                    cz -= size as i32;
                                    ty += 1;
                                }
                                let neighbor_tile = TileId {
                                    z: tile_id.z,
                                    x: tx,
                                    y: ty,
                                };
                                let idxn = cx as usize + cz as usize * size;
                                let dist = dx.abs().max(dz.abs()) as i64;
                                // slope-aware taper: blend neighbor existing base to building base
                                let neighbor_base = tile_map
                                    .get(&neighbor_tile)
                                    .and_then(|d| d.base_ground_vox.as_ref())
                                    .map(|bv| bv[idxn])
                                    .unwrap_or(0i64);
                                let distf = dist as f64;
                                let pad_span = (pad_radius + 1) as f64;
                                let w = (distf / pad_span).clamp(0.0, 1.0);
                                // ease curve for smoother ramp (quadratic)
                                let t = 1.0 - w * w;
                                let raise_to_f =
                                    (neighbor_base as f64) * (1.0 - t) + (b_base as f64) * t;
                                let raise_to = raise_to_f.round() as i64;
                                pad_updates.push((neighbor_tile, idxn, raise_to));
                            } else {
                                let idx = nx as usize + nz as usize * size;
                                let dist = dx.abs().max(dz.abs()) as i64;
                                let neighbor_base = data
                                    .base_ground_vox
                                    .as_ref()
                                    .map(|bv| bv[idx])
                                    .unwrap_or(0i64);
                                let distf = dist as f64;
                                let pad_span = (pad_radius + 1) as f64;
                                let w = (distf / pad_span).clamp(0.0, 1.0);
                                let t = 1.0 - w * w;
                                let raise_to_f =
                                    (neighbor_base as f64) * (1.0 - t) + (b_base as f64) * t;
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
                    if *raise_to > nei_bv[*idx] {
                        nei_bv[*idx] = *raise_to;
                    }
                }
            }
        }
    }

    // Base smoothing after pad updates: smooth base elevations to remove step artifacts
    let mut base_map: HashMap<TileId, Vec<i64>> = HashMap::new();
    for (tile_id, data) in tile_map.iter() {
        if let Some(bv) = data.base_ground_vox.as_ref() {
            base_map.insert(*tile_id, bv.clone());
        }
    }
    // More aggressive smoothing to reduce pad-induced plateaus
    for _ in 0..6 {
        smooth_base_pass(&mut base_map, size);
    }
    blend_base_edges(&mut base_map, size, args.tile_blend_width as usize * 2);
    // Force identical integer values across shared edges to eliminate 1-voxel seams
    // Use an equalization width proportional to the configured tile blend width
    let equal_width = (args.tile_blend_width as usize).max(1);
    // Iteratively equalize and clamp until stable (or hit max iterations)
    let mut iter = 0usize;
    loop {
        let mut changed = 0usize;
        changed += equalize_tile_edges(&mut base_map, size, equal_width);
        changed += clamp_small_edge_deltas(&mut base_map, size);
        changed += equalize_tile_edges(&mut base_map, size, equal_width);
        iter += 1;
        if changed == 0 || iter > 8 {
            break;
        }
    }
    for _ in 0..3 {
        smooth_base_pass(&mut base_map, size);
    }
    // Re-apply equalization after final smoothing to ensure exact edge equality
    equalize_tile_edges(&mut base_map, size, (args.tile_blend_width as usize).max(1));
    for (tile_id, base) in base_map.iter() {
        if let Some(data) = tile_map.get_mut(tile_id) {
            data.base_ground_vox = Some(base.clone());
        }
    }

    // Diagnostic: base integer border deltas after equalization
    let mut large_deltas: Vec<(TileId, TileId, usize, i64, i64)> = Vec::new();
    let mut max_base_delta = 0i64;
    let mut total_base_delta = 0i64;
    let mut base_count = 0usize;
    for tile_id in tiles.iter() {
        if let Some(base) = base_map.get(tile_id) {
            for xi in 0..size {
                // top edge (North, z=size-1) vs neighbor (y-1, South edge z=0)
                let idx_top = xi + (size - 1) * size;
                let top_neighbor = TileId {
                    z: tile_id.z,
                    x: tile_id.x,
                    y: tile_id.y - 1,
                };
                if let Some(nei) = base_map.get(&top_neighbor) {
                    let nei_idx = xi + 0 * size;
                    let delta = (base[idx_top] - nei[nei_idx]).abs();
                    max_base_delta = max_base_delta.max(delta);
                    total_base_delta += delta as i64;
                    base_count += 1;
                    if delta > 0 {
                        large_deltas.push((
                            *tile_id,
                            top_neighbor,
                            nei_idx,
                            base[idx_top],
                            nei[nei_idx],
                        ));
                    }
                }
                // left edge
                let idx_left = 0 + xi * size;
                let left_neighbor = TileId {
                    z: tile_id.z,
                    x: tile_id.x - 1,
                    y: tile_id.y,
                };
                if let Some(nei) = base_map.get(&left_neighbor) {
                    let nei_idx = (size - 1) + xi * size;
                    let delta = (base[idx_left] - nei[nei_idx]).abs();
                    max_base_delta = max_base_delta.max(delta);
                    total_base_delta += delta as i64;
                    base_count += 1;
                    if delta > 0 {
                        large_deltas.push((
                            *tile_id,
                            left_neighbor,
                            nei_idx,
                            base[idx_left],
                            nei[nei_idx],
                        ));
                    }
                }
            }
        }
    }
    if base_count > 0 {
        println!(
            "Base border diffs after equalization: max = {}, avg = {:.3}",
            max_base_delta,
            total_base_delta as f64 / base_count as f64
        );
    }
    // Print some of the largest border deltas for debugging
    large_deltas.sort_by(|a, b| (b.3 - b.4).abs().cmp(&(a.3 - a.4).abs()));
    println!("Largest base border deltas (tileA, tileB, idx, baseA, baseB):");
    for (a, b, idx, va, vb) in large_deltas.iter().take(10) {
        println!("  {:?} - {:?} idx={} baseA={} baseB={}", a, b, idx, va, vb);
    }
    // More thorough listing of 1-voxel diffs by checking each adjacent tile pair explicitly
    let mut one_diffs: Vec<(TileId, TileId, usize, usize, i64, i64)> = Vec::new();
    for tile_id in tiles.iter() {
        if let Some(base) = base_map.get(tile_id) {
            // left neighbor (our x==0 vs neighbor x==size-1)
            let left_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x - 1,
                y: tile_id.y,
            };
            if let Some(nei) = base_map.get(&left_neighbor) {
                for zi in 0..size {
                    let idx_a = 0 + zi * size;
                    let idx_b = (size - 1) + zi * size;
                    let a = base[idx_a];
                    let b = nei[idx_b];
                    if (a - b).abs() == 1 {
                        one_diffs.push((*tile_id, left_neighbor, idx_a, idx_b, a, b));
                    }
                }
            }
            // right neighbor (our x==size-1 vs neighbor x==0)
            let right_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x + 1,
                y: tile_id.y,
            };
            if let Some(nei) = base_map.get(&right_neighbor) {
                for zi in 0..size {
                    let idx_a = (size - 1) + zi * size;
                    let idx_b = 0 + zi * size;
                    let a = base[idx_a];
                    let b = nei[idx_b];
                    if (a - b).abs() == 1 {
                        one_diffs.push((*tile_id, right_neighbor, idx_a, idx_b, a, b));
                    }
                }
            }
            // north neighbor (our z==size-1 vs neighbor z==0)
            let north_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y - 1,
            };
            if let Some(nei) = base_map.get(&north_neighbor) {
                for xi in 0..size {
                    let idx_a = xi + (size - 1) * size;
                    let idx_b = xi + 0 * size;
                    let a = base[idx_a];
                    let b = nei[idx_b];
                    if (a - b).abs() == 1 {
                        one_diffs.push((*tile_id, north_neighbor, idx_a, idx_b, a, b));
                    }
                }
            }
            // south neighbor (our z==0 vs neighbor z==size-1)
            let south_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y + 1,
            };
            if let Some(nei) = base_map.get(&south_neighbor) {
                for xi in 0..size {
                    let idx_a = xi + 0 * size;
                    let idx_b = xi + (size - 1) * size;
                    let a = base[idx_a];
                    let b = nei[idx_b];
                    if (a - b).abs() == 1 {
                        one_diffs.push((*tile_id, south_neighbor, idx_a, idx_b, a, b));
                    }
                }
            }
        }
    }
    println!("Count of 1-voxel border diffs: {}", one_diffs.len());
    for (a, b, idx_a, idx_b, va, vb) in one_diffs.iter().take(50) {
        println!(
            "  {:?} - {:?} idx_a={} idx_b={} baseA={} baseB={}",
            a, b, idx_a, idx_b, va, vb
        );
    }

    // Compute global minimum height across all tiles to avoid generating deep underground voxels
    let mut global_min_height_m = f64::INFINITY;
    for (_, data) in tile_map.iter() {
        if let Some(heights) = data.heights_m.as_ref() {
            for &h in heights.iter() {
                if h < global_min_height_m {
                    global_min_height_m = h;
                }
            }
        }
    }
    // Convert to voxel level with some margin below the lowest valley
    // Use a margin of 10 voxels below the lowest point for visual consistency
    let global_base_y_vox =
        ((global_min_height_m / args.meters_per_voxel).floor() as i64 - 10).max(0);
    eprintln!(
        "Global terrain: min height = {:.1}m, base_y_vox = {} (saving ~{} voxels per column)",
        global_min_height_m, global_base_y_vox, global_base_y_vox
    );

    // Update all TileSpaces with the global base Y
    for (_, space) in tile_spaces.iter_mut() {
        space.base_y_vox = global_base_y_vox;
    }

    // Voxelize each tile using the smoothed heights. Use the previously computed tile grid
    // (start_x/start_y and tile_w/tile_h) rather than args.radius so we operate on the
    // same set of tiles that were fetched and smoothed above.
    for ty in 0..tile_h {
        for tx in 0..tile_w {
            let tile = TileId {
                z: center_tile.z,
                x: start_x + tx,
                y: start_y + ty,
            };
            let data = tile_map.remove(&tile).unwrap();
            let space = tile_spaces.remove(&tile).unwrap();
            let pad_radius = if args.no_building_pad {
                0
            } else {
                args.building_pad_radius
            };
            let result = voxelize_tile(
                tile,
                data,
                &space,
                args.max_height_voxels,
                pad_radius,
                args.no_building_pad,
            );
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

    if matches!(args.format, OutputFormat::Vhc | OutputFormat::Both) {
        let vhc_path = PathBuf::from(format!("{}.vhc", args.output_name));
        println!(
            "Writing compressed VHC (hierarchical chunk) format to {}...",
            vhc_path.display()
        );
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
        save_world_file(&world, &vhc_path, true)?;
        println!(
            "VHC file size: {:.1} MB",
            std::fs::metadata(&vhc_path)?.len() as f64 / 1024.0 / 1024.0
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
