#!/usr/bin/env python3
"""Procedural tile-based voxel generator producing richer demo worlds."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------- Coordinate & Tile Helpers ----------

EARTH_RADIUS_METERS = 6_378_137.0


def _stable_mix(*values: int) -> int:
    """Return a deterministic 64-bit mix of the provided integers."""
    hash_value = 1469598103934665603  # FNV offset basis
    for value in values:
        hash_value ^= value & 0xFFFFFFFFFFFFFFFF
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return hash_value

@dataclass(frozen=True)
class TileId:
    z: int
    x: int
    y: int

    def parent(self) -> Optional['TileId']:
        if self.z == 0:
            return None
        return TileId(self.z - 1, self.x // 2, self.y // 2)

    def children(self) -> List['TileId']:
        return [
            TileId(self.z + 1, self.x * 2 + dx, self.y * 2 + dy)
            for dy in range(2) for dx in range(2)
        ]

    def lon_lat_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((min_lon, max_lon), (north_lat, south_lat)) in degrees."""
        n = 2 ** self.z
        min_lon = self.x / n * 360.0 - 180.0
        max_lon = (self.x + 1) / n * 360.0 - 180.0
        # Mercator latitude inverse
        def merc_y_to_lat(y: float) -> float:
            return math.degrees(math.atan(math.sinh(y)))
        north_lat = merc_y_to_lat(math.pi * (1 - 2 * self.y / n))
        south_lat = merc_y_to_lat(math.pi * (1 - 2 * (self.y + 1) / n))
        return (min_lon, max_lon), (north_lat, south_lat)

# ---------- Synthetic Data / Fetch Layer ----------

class TileFetcher:
    """Procedural surrogate for a real vector-tile fetcher."""

    def __init__(self, seed: int = 1337) -> None:
        self.seed = seed

    def fetch(self, tile: TileId) -> Dict[str, Any]:
        rng = random.Random(_stable_mix(tile.x, tile.y, tile.z, self.seed))
        (min_lon, max_lon), (north_lat, south_lat) = tile.lon_lat_bounds()
        lon_span = max_lon - min_lon
        lat_span = north_lat - south_lat

        def rect_from_frac(fx0: float, fz0: float, fx1: float, fz1: float) -> List[Tuple[float, float]]:
            lon0 = min_lon + lon_span * fx0
            lon1 = min_lon + lon_span * fx1
            lat0 = south_lat + lat_span * fz0
            lat1 = south_lat + lat_span * fz1
            return [(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)]

        grid = 4
        road_frac = 0.05
        road_positions = [i / grid for i in range(grid + 1)]
        half_road = road_frac / 2

        roads: List[List[Tuple[float, float]]] = []
        for fx in road_positions:
            roads.append(rect_from_frac(max(0.0, fx - half_road), 0.0, min(1.0, fx + half_road), 1.0))
        for fz in road_positions:
            roads.append(rect_from_frac(0.0, max(0.0, fz - half_road), 1.0, min(1.0, fz + half_road)))

        blocks_frac: List[Tuple[float, float, float, float]] = []
        for bx in range(grid):
            fx0 = road_positions[bx] + half_road
            fx1 = road_positions[bx + 1] - half_road
            if fx1 - fx0 <= 0.02:
                continue
            for bz in range(grid):
                fz0 = road_positions[bz] + half_road
                fz1 = road_positions[bz + 1] - half_road
                if fz1 - fz0 <= 0.02:
                    continue
                blocks_frac.append((fx0, fz0, fx1, fz1))

        available_indices = list(range(len(blocks_frac)))
        rng.shuffle(available_indices)
        park_slots = rng.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
        park_indices = set(available_indices[:park_slots])

        parks: List[List[Tuple[float, float]]] = []
        for idx in park_indices:
            fx0, fz0, fx1, fz1 = blocks_frac[idx]
            inset = min(fx1 - fx0, fz1 - fz0) * 0.18
            if inset <= 0:
                continue
            parks.append(rect_from_frac(fx0 + inset, fz0 + inset, fx1 - inset, fz1 - inset))

        water_polys: List[List[Tuple[float, float]]] = []
        water_indices: set[int] = set()
        if rng.random() < 0.3 and blocks_frac:
            candidates = [idx for idx in available_indices if idx not in park_indices]
            if candidates:
                idx = rng.choice(candidates)
                water_indices.add(idx)
                fx0, fz0, fx1, fz1 = blocks_frac[idx]
                inset = min(fx1 - fx0, fz1 - fz0) * 0.3
                water_polys.append(rect_from_frac(fx0 + inset, fz0 + inset, fx1 - inset, fz1 - inset))
        if rng.random() < 0.25:
            width = 0.12 + rng.random() * 0.18
            if rng.random() < 0.5:
                fx0 = rng.uniform(0.0, 0.2)
                water_polys.append(rect_from_frac(fx0, 0.0, min(1.0, fx0 + width), 1.0))
            else:
                fz0 = rng.uniform(0.0, 0.2)
                water_polys.append(rect_from_frac(0.0, fz0, 1.0, min(1.0, fz0 + width)))

        district = rng.choices(
            ["residential", "mixed_use", "industrial", "downtown"],
            weights=[0.35, 0.30, 0.15, 0.20],
        )[0]

        building_entries: List[Dict[str, Any]] = []
        height_total = 0.0
        for idx, frac in enumerate(blocks_frac):
            if idx in park_indices or idx in water_indices:
                continue
            fx0, fz0, fx1, fz1 = frac
            margin = min(fx1 - fx0, fz1 - fz0) * 0.07
            fx0 += margin
            fz0 += margin
            fx1 -= margin
            fz1 -= margin
            if fx1 <= fx0 or fz1 <= fz0:
                continue

            lot_w = fx1 - fx0
            lot_d = fz1 - fz0
            if lot_w <= 0 or lot_d <= 0:
                continue

            if district == "residential":
                slots = rng.randint(1, 4)
                archetypes = ["house", "residential", "apartments"]
            elif district == "industrial":
                slots = rng.randint(1, 3)
                archetypes = ["industrial", "warehouse", "commercial"]
            elif district == "downtown":
                slots = rng.randint(2, 6)
                archetypes = ["office", "commercial", "tower"]
            else:
                slots = rng.randint(1, 5)
                archetypes = ["residential", "commercial", "office"]

            for slot in range(slots):
                bw = lot_w * rng.uniform(0.3, 0.9)
                bd = lot_d * rng.uniform(0.3, 0.9)
                bx0 = rng.uniform(fx0, max(fx0, fx1 - bw))
                bz0 = rng.uniform(fz0, max(fz0, fz1 - bd))
                bx1 = bx0 + bw
                bz1 = bz0 + bd
                if bx1 - bx0 <= 0.005 or bz1 - bz0 <= 0.005:
                    continue

                archetype = rng.choice(archetypes)
                tags: Dict[str, str] = {
                    "building": archetype if archetype != "warehouse" else "industrial",
                    "district": district,
                }

                if archetype in {"house", "residential"}:
                    levels = rng.randint(2, 4)
                    tags["building:levels"] = str(levels)
                    tags["roof:style"] = rng.choice(["terracotta", "pitched"])
                    tags["facade"] = rng.choice(["brick", "plaster"])
                elif archetype == "apartments":
                    levels = rng.randint(4, 8)
                    tags["building:levels"] = str(levels)
                    tags["roof:style"] = rng.choice(["flat", "terracotta"])
                    tags["facade"] = rng.choice(["brick", "concrete"])
                elif archetype in {"commercial", "office"}:
                    levels = rng.randint(6, 16)
                    tags["building:levels"] = str(levels)
                    tags["roof:style"] = "flat"
                    tags["facade"] = rng.choice(["glass", "concrete"])
                    if rng.random() < 0.2:
                        tags["name"] = f"Plaza {tile.x}_{tile.y}_{idx}_{slot}"
                elif archetype == "tower":
                    height = rng.uniform(60.0, 140.0)
                    tags["height"] = f"{height:.1f}"
                    tags["roof:style"] = "dark"
                    tags["facade"] = "glass"
                    if rng.random() < 0.5:
                        tags["spire"] = "true"
                elif archetype in {"industrial", "warehouse"}:
                    height = rng.uniform(10.0, 20.0)
                    tags["height"] = f"{height:.1f}"
                    tags["roof:style"] = rng.choice(["dark", "flat"])
                    tags["facade"] = rng.choice(["concrete", "brick"])

                footprint = rect_from_frac(bx0, bz0, bx1, bz1)
                building_entries.append({
                    "footprint": footprint,
                    "tags": tags,
                })
                height_total += infer_height_meters(tags)

        stats = {
            "district": district,
            "building_count": len(building_entries),
            "park_count": len(parks),
            "water_features": len(water_polys),
            "road_segments": len(roads),
        }
        if building_entries:
            stats["average_height_m"] = height_total / len(building_entries)

        return {
            "roads": roads,
            "parks": parks,
            "water": water_polys,
            "blocks": [rect_from_frac(*frac) for frac in blocks_frac],
            "buildings": building_entries,
            "stats": stats,
        }

# ---------- Height Inference ----------

LEVEL_HEIGHT_METERS = 3.1
DEFAULT_HEIGHT_METERS = 9.0

def infer_height_meters(tags: Dict[str, str]) -> float:
    if "height" in tags:
        try:
            return float(tags["height"])
        except ValueError:
            pass
    if "building:levels" in tags:
        try:
            levels = int(tags["building:levels"]) or 1
            return levels * LEVEL_HEIGHT_METERS
        except ValueError:
            pass
    # Archetype fallback
    archetype_defaults = {
        "house": 8.0,
        "residential": 9.0,
        "commercial": 12.0,
        "industrial": 14.0,
        "retail": 10.0,
        "tower": 50.0,
    }
    btype = tags.get("building")
    if btype in archetype_defaults:
        return archetype_defaults[btype]
    return DEFAULT_HEIGHT_METERS

# ---------- Projection (Lon/Lat -> Tile-Local Voxel Space) ----------

def lon_lat_to_mercator_meters(lon_deg: float, lat_deg: float) -> Tuple[float, float]:
    x = math.radians(lon_deg) * EARTH_RADIUS_METERS
    y = EARTH_RADIUS_METERS * math.log(math.tan(math.pi / 4 + math.radians(lat_deg) / 2))
    return x, y

@dataclass
class TileSpace:
    tile: TileId
    voxel_resolution: int = 256  # width/height in X/Z
    meters_per_voxel: float = 1.0
    # Precomputed bounds in mercator meters
    min_x_m: float = field(init=False)
    max_x_m: float = field(init=False)
    min_y_m: float = field(init=False)
    max_y_m: float = field(init=False)

    def __post_init__(self):
        (min_lon, max_lon), (north_lat, south_lat) = self.tile.lon_lat_bounds()
        self.min_x_m, self.min_y_m = lon_lat_to_mercator_meters(min_lon, south_lat)
        self.max_x_m, _ = lon_lat_to_mercator_meters(max_lon, south_lat)
        _, self.max_y_m = lon_lat_to_mercator_meters(min_lon, north_lat)

        # Small guard against zero span
        if abs(self.max_x_m - self.min_x_m) < 1e-6:
            self.max_x_m = self.min_x_m + 1
        if abs(self.max_y_m - self.min_y_m) < 1e-6:
            self.max_y_m = self.min_y_m + 1

    def mercator_to_voxel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        rel_x = (x_m - self.min_x_m) / (self.max_x_m - self.min_x_m)
        rel_y = (y_m - self.min_y_m) / (self.max_y_m - self.min_y_m)
        vx = int(rel_x * (self.voxel_resolution - 1))
        vz = int(rel_y * (self.voxel_resolution - 1))
        return vx, vz

# ---------- Polygon Rasterization (Scanline Fill) ----------

def rasterize_polygon(points: List[Tuple[int, int]], size: int) -> List[Tuple[int, int]]:
    """Return filled voxel (x,z) positions for a simple, convex polygon.
    This is a naive implementation; robust polygon fill would handle concavity & holes.
    """
    if not points:
        return []
    # Bounding box
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    filled = []
    # Even-odd rule per scanline (z as y axis here)
    for z in range(max(min_y, 0), min(max_y + 1, size)):
        # Collect intersections of polygon edges with this scanline
        intersections: List[float] = []
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            # Ignore horizontal edges
            if y1 == y2:
                continue
            # Check if scanline intersects edge
            if (y1 <= z < y2) or (y2 <= z < y1):
                # Linear interpolation of x at intersection
                t = (z - y1) / (y2 - y1)
                x_int = x1 + t * (x2 - x1)
                intersections.append(x_int)
        intersections.sort()
        # Pair up intersections
        for j in range(0, len(intersections), 2):
            if j + 1 >= len(intersections):
                break
            x_start = math.floor(intersections[j])
            x_end = math.ceil(intersections[j + 1])
            for x in range(x_start, x_end):
                if 0 <= x < size:
                    filled.append((x, z))
    return filled

# ---------- Material & Voxel Output ----------

@dataclass
class Material:
    name: str
    albedo: Tuple[int, int, int]
    roughness: int = 128
    metalness: int = 0
    flags: int = 0  # bitfield future use

@dataclass
class VoxelRecord:
    x: int
    y: int
    z: int
    material_index: int

@dataclass
class TileVoxelResult:
    tile: TileId
    voxels: List[VoxelRecord]
    materials: List[Material]
    voxel_resolution: int
    meters_per_voxel: float
    stats: Dict[str, Any]

# ---------- Core Voxelization ----------

def voxelize_tile(tile: TileId, data: Dict[str, Any], space: TileSpace, max_height_voxels: int = 512) -> TileVoxelResult:
    voxels: List[VoxelRecord] = []
    materials: List[Material] = [
        Material("ground_grass", (102, 160, 102), roughness=220),
        Material("road_asphalt", (72, 72, 78), roughness=180),
        Material("park_grass", (90, 180, 90), roughness=220),
        Material("water", (68, 120, 200), roughness=60),
        Material("building_concrete", (210, 210, 220), roughness=160),
        Material("building_brick", (184, 120, 96), roughness=180),
        Material("building_glass", (160, 195, 220), roughness=90, metalness=10),
        Material("roof_terracotta", (180, 110, 85), roughness=140),
        Material("roof_dark", (90, 90, 94), roughness=150),
        Material("tree_trunk", (128, 92, 52), roughness=170),
        Material("tree_canopy", (60, 140, 70), roughness=230),
    ]
    mat_index = {m.name: i for i, m in enumerate(materials)}

    def project_polygon(footprint: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        projected: List[Tuple[int, int]] = []
        for lon, lat in footprint:
            x_m, y_m = lon_lat_to_mercator_meters(lon, lat)
            projected.append(space.mercator_to_voxel(x_m, y_m))
        return projected

    def polygon_cells(polygons: Iterable[List[Tuple[float, float]]]) -> Iterable[List[Tuple[int, int]]]:
        for footprint in polygons:
            projected = project_polygon(footprint)
            if len(projected) < 3:
                continue
            yield rasterize_polygon(projected, space.voxel_resolution)

    # Fill block interiors with base ground
    for cells in polygon_cells(data.get("blocks", [])):
        for vx, vz in cells:
            voxels.append(VoxelRecord(vx, 0, vz, mat_index["ground_grass"]))

    # Water overrides ground (give it some depth)
    for cells in polygon_cells(data.get("water", [])):
        for vx, vz in cells:
            voxels.append(VoxelRecord(vx, 0, vz, mat_index["water"]))
            if max_height_voxels > 1:
                voxels.append(VoxelRecord(vx, 1, vz, mat_index["water"]))

    # Roads sit slightly above grade for contrast
    for cells in polygon_cells(data.get("roads", [])):
        for vx, vz in cells:
            voxels.append(VoxelRecord(vx, 0, vz, mat_index["road_asphalt"]))
            if max_height_voxels > 2:
                voxels.append(VoxelRecord(vx, 1, vz, mat_index["road_asphalt"]))

    # Parks provide greenery and a source for trees later
    park_cells: List[Tuple[int, int]] = []
    for cells in polygon_cells(data.get("parks", [])):
        for vx, vz in cells:
            voxels.append(VoxelRecord(vx, 0, vz, mat_index["park_grass"]))
        park_cells.extend(cells)

    if park_cells:
        tree_rng = random.Random(_stable_mix(tile.x, tile.y, tile.z, 0x51F1F))
        canopy_limit = min(max_height_voxels - 1, 4)
        if canopy_limit >= 2:
            for vx, vz in sorted(set(park_cells)):
                if tree_rng.random() < 0.06:
                    voxels.append(VoxelRecord(vx, 1, vz, mat_index["tree_trunk"]))
                    for h in range(2, canopy_limit + 1):
                        voxels.append(VoxelRecord(vx, h, vz, mat_index["tree_canopy"]))

    # Buildings
    building_entries = data.get("buildings", [])
    base_seed = _stable_mix(tile.x, tile.y, tile.z, 0xDEADBEEF)
    max_roof_y = max(2, max_height_voxels - 1)

    for b_idx, entry in enumerate(building_entries):
        cells_list = list(polygon_cells([entry["footprint"]]))
        if not cells_list:
            continue
        cells = cells_list[0]
        if not cells:
            continue

        tags: Dict[str, str] = entry.get("tags", {})
        height_m = infer_height_meters(tags)
        height_vox = max(3, min(int(math.ceil(height_m / space.meters_per_voxel)), max_height_voxels))
        roof_y = min(height_vox - 1, max_roof_y)

        facade_style = tags.get("facade", "concrete")
        wall_mat_name = {
            "brick": "building_brick",
            "glass": "building_glass",
            "plaster": "building_concrete",
            "concrete": "building_concrete",
            "steel": "building_glass",
        }.get(facade_style, "building_concrete")
        wall_idx = mat_index[wall_mat_name]
        glass_idx = mat_index["building_glass"]

        roof_style = tags.get("roof:style", "terracotta")
        if roof_style in {"dark", "flat"}:
            roof_idx = mat_index["roof_dark"]
        else:
            roof_idx = mat_index["roof_terracotta"]

        band_seed = base_seed ^ (b_idx * 0x45D9F3B)
        facade_rng = random.Random(band_seed & 0xFFFFFFFFFFFF)
        window_period = facade_rng.randint(3, 5)

        for vx, vz in cells:
            for y in range(1, roof_y):
                material_idx = wall_idx
                if wall_mat_name == "building_glass":
                    if y % 6 not in {0, 1}:
                        material_idx = glass_idx
                elif y > 2 and (y % window_period == 0):
                    material_idx = glass_idx
                voxels.append(VoxelRecord(vx, y, vz, material_idx))
            voxels.append(VoxelRecord(vx, roof_y, vz, roof_idx))

        if tags.get("spire") == "true" and roof_y + 3 < max_height_voxels:
            center = cells[len(cells) // 2]
            for extra in range(1, 4):
                voxels.append(VoxelRecord(center[0], roof_y + extra, center[1], roof_idx))

    stats = data.get("stats", {}).copy()
    stats.setdefault("voxel_count", len(voxels))

    return TileVoxelResult(
        tile=tile,
        voxels=voxels,
        materials=materials,
        voxel_resolution=space.voxel_resolution,
        meters_per_voxel=space.meters_per_voxel,
        stats=stats,
    )

# ---------- Serialization ----------

def serialize_tile_ascii(result: TileVoxelResult, path: str) -> None:
    with open(path, 'w') as f:
        for v in result.voxels:
            # Map material index to a simple voxel type for compatibility (1-ground,2-wall,3-roof)
            voxel_type = v.material_index + 1
            f.write(f"{v.x} {v.y} {v.z} {voxel_type}\n")

# ---------- Orchestration ----------

def generate_area(
    center_lon: float,
    center_lat: float,
    zoom: int,
    *,
    radius: int,
    voxel_resolution: int,
    meters_per_voxel: float,
    max_height_voxels: int,
    seed: int,
) -> List[TileVoxelResult]:
    """Generate voxels for tiles around the given center tile."""

    if radius < 0:
        raise ValueError("radius must be >= 0")

    def lon_lat_to_tile(lon: float, lat: float, z: int) -> TileId:
        n = 2 ** z
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
        return TileId(z, x, y)

    center_tile = lon_lat_to_tile(center_lon, center_lat, zoom)
    fetcher = TileFetcher(seed=seed)
    results: List[TileVoxelResult] = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            tile = TileId(center_tile.z, center_tile.x + dx, center_tile.y + dy)
            data = fetcher.fetch(tile)
            space = TileSpace(tile, voxel_resolution=voxel_resolution, meters_per_voxel=meters_per_voxel)
            result = voxelize_tile(tile, data, space, max_height_voxels=max_height_voxels)
            results.append(result)

    results.sort(key=lambda r: (r.tile.y, r.tile.x))
    return results

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-lon", type=float, default=-73.9855, help="Center longitude in degrees")
    parser.add_argument("--center-lat", type=float, default=40.7580, help="Center latitude in degrees")
    parser.add_argument("--zoom", type=int, default=15, help="Web Mercator zoom level")
    parser.add_argument("--radius", type=int, default=1, help="Tile radius around center (Manhattan distance)")
    parser.add_argument("--voxels-per-tile", type=int, default=256, help="Voxel resolution along X/Z per tile")
    parser.add_argument("--meters-per-voxel", type=float, default=1.5, help="World scale mapping for voxel height")
    parser.add_argument("--max-height-voxels", type=int, default=256, help="Clamp voxelized height to this many voxels")
    parser.add_argument("--seed", type=int, default=2024, help="Deterministic seed controlling procedural content")
    parser.add_argument("--output-prefix", type=str, default="osm_voxels", help="Base filename for outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = generate_area(
        args.center_lon,
        args.center_lat,
        args.zoom,
        radius=args.radius,
        voxel_resolution=args.voxels_per_tile,
        meters_per_voxel=args.meters_per_voxel,
        max_height_voxels=args.max_height_voxels,
        seed=args.seed,
    )

    ascii_path = Path(f"{args.output_prefix}.txt")
    with ascii_path.open("w") as f:
        for r in results:
            tile_offset_x = (r.tile.x - results[0].tile.x) * r.voxel_resolution
            tile_offset_z = (r.tile.y - results[0].tile.y) * r.voxel_resolution
            for v in r.voxels:
                voxel_type = v.material_index + 1
                f.write(f"{tile_offset_x + v.x} {v.y} {tile_offset_z + v.z} {voxel_type}\n")

    meta = {
        "world": {
            "center_lon": args.center_lon,
            "center_lat": args.center_lat,
            "zoom": args.zoom,
            "seed": args.seed,
            "radius": args.radius,
            "voxel_resolution": args.voxels_per_tile,
            "meters_per_voxel": args.meters_per_voxel,
            "max_height_voxels": args.max_height_voxels,
        },
        "tiles": [
            {
                "z": r.tile.z,
                "x": r.tile.x,
                "y": r.tile.y,
                "voxel_count": len(r.voxels),
                "stats": r.stats,
            }
            for r in results
        ],
    }

    meta_path = Path(f"{args.output_prefix}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    total_voxels = sum(len(r.voxels) for r in results)
    print(f"Wrote {total_voxels} voxels across {len(results)} tiles to {ascii_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
