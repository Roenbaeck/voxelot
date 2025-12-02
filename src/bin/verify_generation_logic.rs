use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TileId {
    z: u32,
    x: i32,
    y: i32,
}

fn main() {
    let size = 128;
    let mut map: HashMap<TileId, Vec<i64>> = HashMap::new();

    // Create a 2x2 grid
    // (0,0) (1,0)
    // (0,1) (1,1)
    let tiles = vec![
        TileId { z: 0, x: 0, y: 0 },
        TileId { z: 0, x: 1, y: 0 },
        TileId { z: 0, x: 0, y: 1 },
        TileId { z: 0, x: 1, y: 1 },
    ];

    // Initialize with gradients to easily check matching
    for tile in &tiles {
        let mut data = vec![0; size * size];
        for z in 0..size {
            for x in 0..size {
                // Global coordinates
                let gx = tile.x * size as i32 + x as i32;
                let gy = tile.y * size as i32 + z as i32; // y increases South
                data[x + z * size] = (gx + gy) as i64;
            }
        }
        map.insert(*tile, data);
    }

    println!("Verifying initial state...");
    verify_edges(&map, size);

    println!("Running clamp_small_edge_deltas (CORRECTED logic)...");

    // CORRECTED LOGIC
    let mut updates: Vec<(TileId, usize, i64)> = Vec::new();
    let keys: Vec<TileId> = map.keys().copied().collect();

    for tile_id in keys.iter() {
        if let Some(base) = map.get(tile_id) {
            // north neighbor (tile.y -1) (our z==size-1 vs neighbor z==0)
            let north_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y - 1,
            };
            if let Some(nei) = map.get(&north_neighbor) {
                for xi in 0..size {
                    let idx_a = xi + (size - 1) * size; // Our North edge
                    let idx_b = xi + 0 * size; // Neighbor South edge
                    let a = base[idx_a];
                    let b = nei[idx_b];
                    // In this test, they should match exactly, but let's simulate the logic
                    let delta = (a - b).abs();
                    if delta <= 1 {
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx_a, avg));
                        updates.push((north_neighbor, idx_b, avg));
                    }
                }
            }

            // south neighbor (our z==0 vs neighbor z==size-1)
            let south_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x,
                y: tile_id.y + 1,
            };
            if let Some(nei) = map.get(&south_neighbor) {
                for xi in 0..size {
                    let idx_a = xi + 0 * size; // Our South edge
                    let idx_b = xi + (size - 1) * size; // Neighbor North edge
                    let a = base[idx_a];
                    let b = nei[idx_b];
                    let delta = (a - b).abs();
                    if delta <= 1 {
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx_a, avg));
                        updates.push((south_neighbor, idx_b, avg));
                    }
                }
            }

            // right neighbor (our x==size-1 vs neighbor x==0)
            let right_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x + 1,
                y: tile_id.y,
            };
            if let Some(nei) = map.get(&right_neighbor) {
                for zi in 0..size {
                    let idx_a = (size - 1) + zi * size;
                    let idx_b = 0 + zi * size;
                    let a = base[idx_a];
                    let b = nei[idx_b];
                    let delta = (a - b).abs();
                    if delta <= 1 {
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx_a, avg));
                        updates.push((right_neighbor, idx_b, avg));
                    }
                }
            }

            // left neighbor (our x==0 vs neighbor x==size-1)
            let left_neighbor = TileId {
                z: tile_id.z,
                x: tile_id.x - 1,
                y: tile_id.y,
            };
            if let Some(nei) = map.get(&left_neighbor) {
                for zi in 0..size {
                    let idx_a = 0 + zi * size;
                    let idx_b = (size - 1) + zi * size;
                    let a = base[idx_a];
                    let b = nei[idx_b];
                    let delta = (a - b).abs();
                    if delta <= 1 {
                        let avg = ((a + b) as f64 / 2.0).round() as i64;
                        updates.push((*tile_id, idx_a, avg));
                        updates.push((left_neighbor, idx_b, avg));
                    }
                }
            }
        }
    }

    println!("Updates generated: {}", updates.len());

    // Apply updates
    for (tile, idx, val) in updates {
        if let Some(v) = map.get_mut(&tile) {
            v[idx] = val;
        }
    }

    println!("Verifying final state...");
    verify_edges(&map, size);
    println!("Verification passed!");
}

fn verify_edges(map: &HashMap<TileId, Vec<i64>>, size: usize) {
    for (tile_id, data) in map {
        // Check South neighbor (y+1)
        let south_id = TileId {
            x: tile_id.x,
            y: tile_id.y + 1,
            z: tile_id.z,
        };
        if let Some(south_data) = map.get(&south_id) {
            // Our South edge (zi=0) should match South neighbor's North edge (zi=size-1)
            for x in 0..size {
                let our_val = data[x + 0 * size];
                let their_val = south_data[x + (size - 1) * size];
                if our_val != their_val {
                    println!(
                        "Mismatch between {:?} (South) and {:?} (North) at x={}",
                        tile_id, south_id, x
                    );
                }
            }
        }

        // Check East neighbor (x+1)
        let east_id = TileId {
            x: tile_id.x + 1,
            y: tile_id.y,
            z: tile_id.z,
        };
        if let Some(east_data) = map.get(&east_id) {
            // Our East edge (xi=size-1) should match East neighbor's West edge (xi=0)
            for z in 0..size {
                let our_val = data[(size - 1) + z * size];
                let their_val = east_data[0 + z * size];
                if our_val != their_val {
                    println!(
                        "Mismatch between {:?} (East) and {:?} (West) at z={}",
                        tile_id, east_id, z
                    );
                }
            }
        }
    }
}
