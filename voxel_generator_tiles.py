#!/usr/bin/env python3
"""
Next-Gen OSM Voxel Generator (Tile-Based)

Dry-coded prototype (no external libs executed). This module outlines a higher-fidelity
voxelization pipeline inspired by tile-streaming approaches.

Pipeline Stages:
1. Tile selection
2. Fetch (placeholder synthetic data)
3. Decode (stub returning polygons + tags)
4. Height inference
5. Polygon rasterization
6. Voxel column fill & material assignment
7. Output serialization

Future: replace synthetic fetch/decode with real MVT / GeoJSON ingestion.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable, Optional
import math
import json

# ---------- Coordinate & Tile Helpers ----------

WEB_MERCATOR_WORLD_SIZE = 40075016.68557849  # meters at equator

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
        """Return ((min_lon, max_lon), (min_lat, max_lat)) in degrees."""
        n = 2 ** self.z
        min_lon = self.x / n * 360.0 - 180.0
        max_lon = (self.x + 1) / n * 360.0 - 180.0
        # Mercator latitude inverse
        def merc_y_to_lat(y: float) -> float:
            return math.degrees(math.atan(math.sinh(y)))
        min_lat = merc_y_to_lat(math.pi * (1 - 2 * self.y / n))
        max_lat = merc_y_to_lat(math.pi * (1 - 2 * (self.y + 1) / n))
        return (min_lon, max_lon), (max_lat, min_lat)  # note order for lat (north,south)

# ---------- Synthetic Data / Fetch Layer ----------

class TileFetcher:
    """Placeholder fetcher that returns stubbed building footprints within tile bounds."""
    def fetch(self, tile: TileId) -> Dict:
        # In a real implementation, this would issue HTTP requests for vector tiles.
        # Here we fabricate a few rectangles with plausible tags.
        (min_lon, max_lon), (north_lat, south_lat) = tile.lon_lat_bounds()
        lon_span = max_lon - min_lon
        lat_span = north_lat - south_lat
        center_lon = (min_lon + max_lon) / 2
        center_lat = (north_lat + south_lat) / 2

        buildings = []
        for i in range(5):
            frac_x = (i + 1) / 6
            # Simple rectangular footprint
            b_min_lon = min_lon + lon_span * (frac_x * 0.6)
            b_max_lon = b_min_lon + lon_span * 0.02
            b_min_lat = south_lat + lat_span * 0.3
            b_max_lat = b_min_lat + lat_span * 0.05
            poly = [
                (b_min_lon, b_min_lat),
                (b_max_lon, b_min_lat),
                (b_max_lon, b_max_lat),
                (b_min_lon, b_max_lat),
            ]
            tags = {
                "building": "yes",
                "name": f"Stub {i}",
                "building:levels": str(2 + i),
            }
            buildings.append({"footprint": poly, "tags": tags})

        return {"buildings": buildings, "center": (center_lon, center_lat)}

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
    x = math.radians(lon_deg) * 6378137.0
    y = math.log(math.tan(math.pi / 4 + math.radians(lat_deg) / 2)) * 6378137.0
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
        self.min_x_m, _ = lon_lat_to_mercator_meters(min_lon, south_lat)
        self.max_x_m, _ = lon_lat_to_mercator_meters(max_lon, south_lat)
        _, self.min_y_m = lon_lat_to_mercator_meters(min_lon, south_lat)
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

DEFAULT_BUILDING_MATERIAL = Material("building_wall", (180, 180, 190), roughness=200)
DEFAULT_ROOF_MATERIAL = Material("roof_flat", (190, 170, 120), roughness=180)
GROUND_MATERIAL = Material("ground", (80, 140, 80), roughness=220)

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

# ---------- Core Voxelization ----------

def voxelize_buildings(tile: TileId, data: Dict, space: TileSpace, max_height_voxels: int = 512) -> TileVoxelResult:
    voxels: List[VoxelRecord] = []
    materials: List[Material] = [GROUND_MATERIAL, DEFAULT_BUILDING_MATERIAL, DEFAULT_ROOF_MATERIAL]
    mat_index = {m.name: i for i, m in enumerate(materials)}

    # Ground plane fill (optional: only where buildings exist; here we keep sparse)
    # For each building footprint, place ground voxels at y=0

    for entry in data.get("buildings", []):
        footprint_lon_lat: List[Tuple[float, float]] = entry["footprint"]
        tags = entry.get("tags", {})
        height_m = infer_height_meters(tags)
        height_vox = max(1, min(int(height_m / space.meters_per_voxel), max_height_voxels))

        # Project footprint to voxel space
        projected: List[Tuple[int, int]] = []
        for lon, lat in footprint_lon_lat:
            x_m, y_m = lon_lat_to_mercator_meters(lon, lat)
            vx, vz = space.mercator_to_voxel(x_m, y_m)
            projected.append((vx, vz))

        filled_cells = rasterize_polygon(projected, space.voxel_resolution)

        # Ground fill & column extrusion
        for (vx, vz) in filled_cells:
            # Ground voxel
            voxels.append(VoxelRecord(vx, 0, vz, mat_index[GROUND_MATERIAL.name]))
            for y in range(1, height_vox - 1):
                voxels.append(VoxelRecord(vx, y, vz, mat_index[DEFAULT_BUILDING_MATERIAL.name]))
            if height_vox > 1:
                voxels.append(VoxelRecord(vx, height_vox - 1, vz, mat_index[DEFAULT_ROOF_MATERIAL.name]))

    return TileVoxelResult(tile=tile, voxels=voxels, materials=materials)

# ---------- Serialization ----------

def serialize_tile_ascii(result: TileVoxelResult, path: str) -> None:
    with open(path, 'w') as f:
        for v in result.voxels:
            # Map material index to a simple voxel type for compatibility (1-ground,2-wall,3-roof)
            voxel_type = v.material_index + 1
            f.write(f"{v.x} {v.y} {v.z} {voxel_type}\n")

# ---------- Orchestration ----------

def generate_area(center_lon: float, center_lat: float, zoom: int, radius: int = 0) -> List[TileVoxelResult]:
    """Generate voxels for tiles around a center tile (radius in tile space)."""
    # Determine center tile
    def lon_lat_to_tile(lon: float, lat: float, z: int) -> TileId:
        n = 2 ** z
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return TileId(z, x, y)

    center_tile = lon_lat_to_tile(center_lon, center_lat, zoom)
    fetcher = TileFetcher()
    results: List[TileVoxelResult] = []

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            tile = TileId(center_tile.z, center_tile.x + dx, center_tile.y + dy)
            data = fetcher.fetch(tile)
            space = TileSpace(tile)
            result = voxelize_buildings(tile, data, space)
            results.append(result)
    return results

# ---------- CLI (Dry) ----------

def main():
    center_lon = -73.9855  # Times Square approx
    center_lat = 40.7580
    zoom = 15
    radius = 0
    results = generate_area(center_lon, center_lat, zoom, radius)
    # Combine all tiles into one ascii for compatibility
    with open('osm_voxels_new.txt', 'w') as f:
        for r in results:
            for v in r.voxels:
                voxel_type = v.material_index + 1
                # Shift by tile origin (simple tiling in X/Z): tile.x * 256, tile.y * 256
                base_x = (r.tile.x % 1024) * 256  # naive wrap
                base_z = (r.tile.y % 1024) * 256
                f.write(f"{base_x + v.x} {v.y} {base_z + v.z} {voxel_type}\n")
    meta = {
        "tiles": [
            {"z": r.tile.z, "x": r.tile.x, "y": r.tile.y, "voxel_count": len(r.voxels)}
            for r in results
        ]
    }
    with open('osm_voxels_new_meta.json', 'w') as mf:
        json.dump(meta, mf, indent=2)
    print("Generated", sum(len(r.voxels) for r in results), "voxels across", len(results), "tiles")

if __name__ == '__main__':
    print("Tile-based voxel generator (dry run).")
    main()
