#!/usr/bin/env python3
"""
OSM Voxel Generator - Pull real-world geographic data and convert to voxel format
"""

import warnings
# Suppress the urllib3 SSL warning
warnings.filterwarnings('ignore', message='.*urllib3 v2 only supports OpenSSL.*')

import requests
import json
import math
from typing import List, Tuple, Dict, Any

def lat_lon_to_voxel(lat: float, lon: float, center_lat: float, center_lon: float, scale: float = 10000.0) -> Tuple[int, int]:
    """Convert lat/lon to voxel coordinates relative to center
    
    Scale of 10000 means 0.0001 degrees ≈ 11 meters ≈ 1 voxel
    """
    # Simple equirectangular projection
    x = (lon - center_lon) * scale * math.cos(math.radians(center_lat))
    z = (lat - center_lat) * scale
    return int(x), int(z)

def get_osm_buildings(area_name: str, max_buildings: int = 50) -> List[Dict[str, Any]]:
    """Pull building data from OpenStreetMap"""
    # Try a simpler query first
    overpass_query = f"""
    [out:json];
    area[name="{area_name}"]->.a;
    (
      node(area.a)[building];
      way(area.a)[building];
      relation(area.a)[building];
    );
    out meta;
    """

    try:
        response = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query, timeout=30)
        data = response.json()

        buildings = []
        for element in data['elements'][:max_buildings]:
            if 'tags' in element and 'building' in element['tags']:
                buildings.append(element)

        return buildings
    except Exception as e:
        print(f"Error querying OSM: {e}")
        return []

def get_elevation_batch(locations: List[Tuple[float, float]]) -> List[float]:
    """Get elevation for multiple points using Open Topography API"""
    if not locations:
        return []
    
    # Format as lat,lng|lat,lng|...
    location_str = "|".join(f"{lat},{lon}" for lat, lon in locations)
    url = f"https://api.opentopodata.org/v1/aster30m?locations={location_str}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return [result['elevation'] for result in data['results']]
    except Exception as e:
        print(f"Elevation API error: {e}")
    
    # Fallback: return zeros
    return [0.0] * len(locations)

def generate_simple_terrain(center_lat: float, center_lon: float, building_bounds: Tuple[Tuple[int, int], Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    """Generate simple flat terrain covering the building area"""
    voxels = []
    
    (min_x, max_x), (min_z, max_z) = building_bounds
    
    # Expand slightly beyond buildings
    min_x -= 50
    max_x += 50
    min_z -= 50
    max_z += 50
    
    print(f"Generating flat terrain: X[{min_x}, {max_x}] Z[{min_z}, {max_z}]")
    
    # Create flat terrain at height 0
    base_height = 0
    for x in range(min_x, max_x + 1):
        for z in range(min_z, max_z + 1):
            voxels.append((x + 1000, base_height, z + 1000, 1))  # Grass surface
    
    print(f"Generated {len(voxels)} terrain voxels")
    return voxels
    """Generate terrain voxels using elevation data"""
    voxels = []
    
    # Convert km to degrees (rough approximation)
    km_to_deg = 0.009  # ~1km in degrees latitude
    half_size_deg = (size_km * km_to_deg) / 2
    
    print(f"Generating {resolution}x{resolution} terrain grid...")
    
    # Generate grid of elevation points
    elevations = []
    locations = []
    
    for i in range(resolution):
        for j in range(resolution):
            # Convert grid coordinates to lat/lon
            lat = center_lat + (i / resolution - 0.5) * 2 * half_size_deg
            lon = center_lon + (j / resolution - 0.5) * 2 * half_size_deg
            locations.append((lat, lon))
    
    # Get elevations in batches (API has limits)
    batch_size = 100
    all_elevations = []
    
    for i in range(0, len(locations), batch_size):
        batch = locations[i:i+batch_size]
        batch_elevations = get_elevation_batch(batch)
        all_elevations.extend(batch_elevations)
        print(f"Processed elevation batch {i//batch_size + 1}/{(len(locations)+batch_size-1)//batch_size}")
    
    # Convert elevations to voxel coordinates
    min_elev = min(all_elevations) if all_elevations else 0
    max_elev = max(all_elevations) if all_elevations else 100
    
    # Scale elevations to reasonable voxel heights (0-50 voxels)
    elev_scale = 50.0 / max(1.0, max_elev - min_elev)
    
    # Create a 2D grid to store heights with VOXEL coordinates (not grid indices)
    height_grid = {}
    for i in range(resolution):
        for j in range(resolution):
            idx = i * resolution + j
            if idx < len(all_elevations):
                elevation = all_elevations[idx]
                voxel_height = int((elevation - min_elev) * elev_scale)
                # Convert lat/lon to voxel coordinates (same as buildings!)
                lat = center_lat + (i / resolution - 0.5) * 2 * half_size_deg
                lon = center_lon + (j / resolution - 0.5) * 2 * half_size_deg
                x, z = lat_lon_to_voxel(lat, lon, center_lat, center_lon)
                height_grid[(x, z)] = voxel_height
    
    # Now fill in ALL positions to create solid terrain
    min_x = min(x for x, z in height_grid.keys())
    max_x = max(x for x, z in height_grid.keys())
    min_z = min(z for x, z in height_grid.keys())
    max_z = max(z for x, z in height_grid.keys())
    
    print(f"Terrain coordinate range: X[{min_x}, {max_x}] Z[{min_z}, {max_z}]")
    
    for x in range(min_x, max_x + 1):
        for z in range(min_z, max_z + 1):
            # Use nearest grid point height
            if (x, z) in height_grid:
                voxel_height = height_grid[(x, z)]
            else:
                # Find nearest grid point
                min_dist = float('inf')
                nearest_height = 0
                for (gx, gz), h in height_grid.items():
                    dist = abs(x - gx) + abs(z - gz)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_height = h
                voxel_height = nearest_height
            
            # Create terrain column
            for y in range(voxel_height + 1):
                # Different materials based on height
                if y == voxel_height:
                    voxel_type = 1  # Grass on top
                elif y > voxel_height - 3:
                    voxel_type = 7  # Dirt/rock layers
                else:
                    voxel_type = 7  # Rock base
                
                voxels.append((x + 1000, y, z + 1000, voxel_type))
    
    print(f"Generated {len(voxels)} terrain voxels")
    return voxels

def buildings_to_voxels(buildings: List[Dict[str, Any]], center_lat: float, center_lon: float) -> List[Tuple[int, int, int, int]]:
    """Convert OSM buildings to voxel format (x, y, z, voxel_type)"""
    voxels = []

    for building in buildings[:50]:  # Process up to 50 buildings
        print(f"Processing building type: {building.get('type')}")
        
        # Get building footprint coordinates
        footprint_coords = []
        
        if building['type'] == 'node' and 'lat' in building and 'lon' in building:
            # Single point building - create a larger footprint around it
            lat, lon = building['lat'], building['lon']
            # Create a ~15x15 meter footprint around the point (0.000135 degrees ≈ 15m)
            offset = 0.000135
            footprint_coords = [
                (lat - offset, lon - offset),
                (lat + offset, lon - offset),
                (lat + offset, lon + offset),
                (lat - offset, lon + offset),
            ]
            
        elif building['type'] == 'way' and 'geometry' in building:
            # Way with geometry - extract all node coordinates
            geometry = building['geometry']
            if geometry:
                footprint_coords = [(node['lat'], node['lon']) for node in geometry]
        
        if footprint_coords:
            # Convert all footprint coordinates to voxel space
            voxel_coords = []
            for lat, lon in footprint_coords:
                x, z = lat_lon_to_voxel(lat, lon, center_lat, center_lon)
                voxel_coords.append((x, z))
            
            if voxel_coords:
                # Calculate bounding box of the footprint
                min_x = min(x for x, z in voxel_coords)
                max_x = max(x for x, z in voxel_coords)
                min_z = min(z for x, z in voxel_coords)
                max_z = max(z for x, z in voxel_coords)
                
                # Ensure minimum size (at least 5x5 voxels for visibility)
                width = max(5, max_x - min_x + 1)
                depth = max(5, max_z - min_z + 1)
                
                # Use centroid for positioning
                center_x = (min_x + max_x) // 2
                center_z = (min_z + max_z) // 2
                
                print(f"Building footprint: {width}x{depth} voxels at ({center_x}, {center_z})")

                # Get height from tags or default
                # With scale=10000, voxel spacing is ~11m, so scale down building heights
                height = 3  # default height in voxels (~33m / 10 stories)
                if 'tags' in building:
                    tags = building['tags']
                    if 'height' in tags:
                        try:
                            # Convert meters to voxel units with proper scale
                            # Each voxel unit ≈ 11m at our lat/lon scale
                            meters = float(tags['height'])
                            height = max(1, int(meters / 11.0))  # Scale to voxel units
                            print(f"Height from tags: {meters}m -> {height} voxels")
                        except:
                            pass
                    elif 'building:levels' in tags:
                        try:
                            # Estimate 3 meters per level, then scale
                            meters = int(tags['building:levels']) * 3
                            height = max(1, int(meters / 11.0))
                            print(f"Height from levels: {meters}m -> {height} voxels")
                        except:
                            pass

                # Create building voxels - fill the entire footprint
                base_x, base_z = center_x + 1000, center_z + 1000  # Shift to positive coords
                
                # Fill the footprint area
                for dx in range(width):
                    for dz in range(depth):
                        x_pos = base_x + dx - width // 2
                        z_pos = base_z + dz - depth // 2
                        
                        for y in range(height):
                            voxel_type = 4 if y == height - 1 else 7  # Yellow roof, gray walls
                            voxels.append((x_pos, y, z_pos, voxel_type))

    return voxels

def main():
    print("Starting OSM voxel generator...")

    # Example: Get buildings from a larger area of Manhattan
    bbox = "40.7400,-74.0100,40.7700,-73.9800"  # Larger Times Square area

    print("Pulling building data for larger Times Square area...")

    # Use bounding box query with geometry
    overpass_query = f"""
    [out:json];
    (
      node["building"]({bbox});
      way["building"]({bbox});
      relation["building"]({bbox});
    );
    out geom;
    """

    try:
        response = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query, timeout=30)
        print(f"API Response status: {response.status_code}")
        data = response.json()
        print(f"Total elements returned: {len(data.get('elements', []))}")

        buildings = []
        for element in data['elements'][:100]:  # Increased to 100 buildings
            if 'tags' in element and 'building' in element['tags']:
                buildings.append(element)
                name = element.get('tags', {}).get('name', 'unnamed')
                print(f"Found building: {name}")

        print(f"Filtered to {len(buildings)} buildings with building tags")

        # Use center of the bounding box
        center_lat = (40.7500 + 40.7600) / 2
        center_lon = (-74.0000 + -73.9900) / 2

        print(f"Found {len(buildings)} buildings")

        # Use center of Manhattan
        center_lat, center_lon = 40.7831, -73.9712

        # Convert to voxels
        voxels = buildings_to_voxels(buildings, center_lat, center_lon)

        print(f"Generated {len(voxels)} building voxels")

        # Calculate building bounds
        if voxels:
            building_x_coords = [x for x, y, z, vtype in voxels]
            building_z_coords = [z for x, y, z, vtype in voxels]
            building_bounds = (
                (min(building_x_coords) - 1000, max(building_x_coords) - 1000),
                (min(building_z_coords) - 1000, max(building_z_coords) - 1000)
            )
        else:
            building_bounds = ((-100, 100), (-100, 100))
        
        # Generate simple flat terrain
        terrain_voxels = generate_simple_terrain(center_lat, center_lon, building_bounds)
        
        # Combine building and terrain voxels
        all_voxels = terrain_voxels + voxels

        print(f"Total voxels: {len(all_voxels)} (terrain: {len(terrain_voxels)}, buildings: {len(voxels)})")

        # Save to a simple format that Rust can read
        with open('osm_voxels.txt', 'w') as f:
            for x, y, z, voxel_type in all_voxels:
                f.write(f"{x} {y} {z} {voxel_type}\n")

        print(f"Saved {len(voxels)} voxels to osm_voxels.txt")

        # Also save debug info
        with open('debug.log', 'w') as f:
            f.write(f"Total buildings: {len(buildings)}\n")
            f.write(f"Generated voxels: {len(voxels)}\n")
            if buildings:
                f.write(f"First building: {buildings[0]}\n")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Script starting...")
    main()
    print("Script finished.")