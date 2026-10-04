//! Optimized greedy meshing using bitwise operations
//! Based on https://github.com/TanTanDev/binary_greedy_mesher_demo

use crate::lib_hierarchical::{Chunk, Voxel, World, WorldPos};
use crate::palette::Palette;
use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;

#[derive(Copy, Clone, Debug)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub emissive: [f32; 4],
    pub material: [f32; 4], // R=reflectivity, GBA=reserved
}

#[derive(Copy, Clone, Debug)]
pub struct ChunkEmitter {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

/// Mesh output for a chunk
#[derive(Clone, Debug, Default)]
pub struct ChunkMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub emitters: Vec<ChunkEmitter>,
}

#[derive(Debug, Clone, Copy)]
struct GreedyQuad {
    x: u8,
    y: u8,
    w: u8,
    h: u8,
}

/// Generate quads for a 16x16 binary plane (u16 bitmasks)
fn greedy_mesh_binary_plane(mut data: [u16; 16]) -> Vec<GreedyQuad> {
    let mut quads = Vec::with_capacity(16); // Heuristic

    for row in 0..16 {
        let mut y = 0;
        while y < 16 {
            // Find first solid bit
            // data[row] >> y shifts the row so bit 'y' is at position 0
            let remaining = data[row] >> y;
            if remaining == 0 {
                break; // No more set bits in this row
            }

            // Number of trailing zeros gives us the distance to the next set bit
            let skip = remaining.trailing_zeros();
            y += skip;

            if y >= 16 {
                continue;
            }

            // Now y points to a set bit. Find height of this run of 1s (vertical run in the mask)
            // Note: In the original algorithm "height" refers to the run of 1s in the u16 (which corresponds to one dimension)
            // and "width" refers to how many rows have this same run.
            // Let's stick to the original nomenclature:
            // h = length of run in the integer (y-axis in local 2D coords)
            // w = number of matching integers (x-axis in local 2D coords)

            let current_bits = data[row] >> y;
            let h = current_bits.trailing_ones();

            // Create a mask for this run of 'h' bits
            // e.g. h=2 -> 0b11
            let h_as_mask = if h >= 16 { 0xFFFF } else { (1u16 << h) - 1 };
            let mask = h_as_mask << y;

            // Grow horizontally (check subsequent rows)
            let mut w = 1;
            while (row + w) < 16 {
                // Check if the next row has the exact same bits set in this range
                let next_row_bits = (data[row + w] >> y) & h_as_mask;
                if next_row_bits != h_as_mask {
                    break; // Can't expand
                }

                // Clear the bits we've just claimed so they aren't meshed again
                data[row + w] &= !mask;
                w += 1;
            }

            quads.push(GreedyQuad {
                x: row as u8,
                y: y as u8,
                w: w as u8,
                h: h as u8,
            });

            // Advance y by h (since we consumed these bits)
            y += h;
        }
    }
    quads
}

/// Voxel-grid scratch that `representative_surface_reflectivity` reads. Rows are indexed
/// `[z][y]` with bit `x`, matching `Chunk::flat_index` order (x fastest, then y, then z).
struct SurfaceGrid<'a> {
    /// Bit set where a voxel of any kind (solid or sub-chunk) is present.
    occupied: &'a [[u16; 16]; 16],
    /// Bit set where the voxel is `Voxel::Solid`.
    solid: &'a [[u16; 16]; 16],
    /// Voxel type per `flat_index` (valid where `solid` is set).
    types: &'a [u8; 4096],
}

/// Reflectivity that best represents the visible surface of `chunk` (used by envelope meshes).
///
/// A voxel is "visible" if it touches the chunk boundary or has an empty in-chunk face neighbour.
/// Voxels are visited in flat-index order so the f32 accumulation is identical to a walk over
/// `Chunk::iter()`; this version just works on row bitmasks instead of 6 Roaring `contains` per
/// voxel (which made envelope meshing ~10x slower than detail meshing).
fn representative_surface_reflectivity(
    chunk: &Chunk,
    palette: &Palette,
    grid: &SurfaceGrid,
) -> f32 {
    let mut visible_total = 0u32;
    let mut reflective_total = 0u32;
    let mut reflectivity_sum = 0.0f32;
    let mut reflectivity_max = 0.0f32;

    for z in 0..16usize {
        for y in 0..16usize {
            let row = grid.occupied[z][y];
            if row == 0 {
                continue;
            }
            let surface = if z == 0 || z == 15 || y == 0 || y == 15 {
                row
            } else {
                // Interior row: a voxel is hidden only if all six face neighbours are present
                // (x == 0 / x == 15 voxels always count as surface).
                let enclosed = (row << 1)
                    & (row >> 1)
                    & grid.occupied[z][y - 1]
                    & grid.occupied[z][y + 1]
                    & grid.occupied[z - 1][y]
                    & grid.occupied[z + 1][y]
                    & 0x7FFE;
                row & !enclosed
            };

            let mut bits = surface & grid.solid[z][y];
            while bits != 0 {
                let x = bits.trailing_zeros() as usize;
                bits &= bits - 1;

                let voxel_type = grid.types[(z << 8) | (y << 4) | x];
                let reflectivity = palette.reflectivity(voxel_type as u32);
                visible_total = visible_total.saturating_add(1);
                reflectivity_sum += reflectivity;
                reflectivity_max = reflectivity_max.max(reflectivity);
                if reflectivity > 0.01 {
                    reflective_total = reflective_total.saturating_add(1);
                }
            }
        }
    }

    if visible_total == 0 {
        return palette.reflectivity(chunk.dominant_type as u32);
    }

    let coverage = reflective_total as f32 / visible_total as f32;
    let average = reflectivity_sum / visible_total as f32;
    let coverage_boost = ((coverage - 0.03) / 0.22).clamp(0.0, 1.0);
    average.max(reflectivity_max * coverage_boost * 0.85)
}

/// The six face-neighbour offsets (in chunks) that `generate_chunk_mesh_optimized` reads from
/// its `neighbors` map: boundary-face culling looks one chunk away along a single axis, and
/// `calculate_ao` is still a placeholder that ignores neighbours. If real AO lands and starts
/// sampling edge/corner neighbours, widen this to the 26-neighbourhood again.
pub const MESH_NEIGHBOR_OFFSETS: [(i8, i8, i8); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Snapshot the face-neighbour leaf chunks of the chunk at `origin` (a 16-aligned world origin)
/// in the form `generate_chunk_mesh_optimized` expects. Missing neighbours (outside the world,
/// or not a subdivided chunk) are simply absent from the map.
///
/// `arc_cache` maps neighbour origin -> `Arc<Chunk>` so repeated jobs keep sharing one `Arc` per
/// chunk: a cached entry is reused, otherwise the chunk found in `world` is inserted. (Callers
/// drop entries when a chunk is edited so the next snapshot re-reads the world.)
///
/// This replaces a 27-lookup (3x3x3) snapshot of which 21 entries were never read; it does
/// 6 non-allocating hierarchy walks and a single map allocation.
pub fn snapshot_mesh_neighbors(
    world: &World,
    origin: (i64, i64, i64),
    arc_cache: &mut HashMap<(i64, i64, i64), Arc<Chunk>>,
) -> HashMap<(i8, i8, i8), Arc<Chunk>> {
    let mut neighbors: HashMap<(i8, i8, i8), Arc<Chunk>> =
        HashMap::with_capacity_and_hasher(MESH_NEIGHBOR_OFFSETS.len(), Default::default());
    for &(dx, dy, dz) in &MESH_NEIGHBOR_OFFSETS {
        let nk = (
            origin.0 + ((dx as i64) << 4),
            origin.1 + ((dy as i64) << 4),
            origin.2 + ((dz as i64) << 4),
        );
        let Some(found) = world.leaf_chunk_arc_ref_at_origin(WorldPos::new(nk.0, nk.1, nk.2))
        else {
            continue;
        };
        let arc = match arc_cache.get(&nk) {
            Some(existing) => existing.clone(),
            None => {
                arc_cache.insert(nk, found.clone());
                found.clone()
            }
        };
        neighbors.insert((dx, dy, dz), arc);
    }
    neighbors
}

/// Generate mesh for a chunk using optimized bitwise operations
pub fn generate_chunk_mesh_optimized(
    chunk: &Chunk,
    palette: &Palette,
    neighbors: Option<&HashMap<(i8, i8, i8), Arc<Chunk>>>,
    envelope: bool,
) -> ChunkMesh {
    let mut mesh = ChunkMesh::default();

    // 1. Extract columns into bitmasks
    // axis_cols[axis][z][x] where axis 0=y-cols (x,z plane), 1=x-cols (y,z plane), 2=z-cols (x,y plane)
    // Wait, let's align with the reference implementation logic but adapted for our axes.
    // We want 3 arrays of 16x16 u16s.
    // axis_cols[0] stores Y-columns indexed by [z][x] (so we can check faces along Y) - Wait, no.

    // Let's stick to the reference:
    // axis_cols[0]: x,z plane, bits along y. Indexed [z][x].
    // axis_cols[1]: z,y plane, bits along x. Indexed [y][z].
    // axis_cols[2]: x,y plane, bits along z. Indexed [y][x].

    let mut axis_cols = [[[0u16; 16]; 16]; 3];
    // Dense material grid (1 byte per voxel) so the per-face material lookup below is a plain
    // array read instead of a presence-bitmap `contains` + `rank` + Vec index.
    let mut dense = [0u8; 4096];
    // Envelope meshes also need row occupancy for the surface-reflectivity estimate.
    let mut occupied_rows = [[0u16; 16]; 16];
    let mut solid_rows = [[0u16; 16]; 16];

    // Helper to add a voxel to the bitmasks
    // We can optimize this by using the chunk's existing marginals to skip empty areas?
    // For now, let's just iterate the chunk's sparse storage which is already efficient.
    for ((x, y, z), voxel) in chunk.iter() {
        if envelope {
            occupied_rows[z as usize][y as usize] |= 1 << x;
        }
        if let Voxel::Solid(t) = voxel {
            dense[Chunk::flat_index(x, y, z) as usize] = *t;
            let x = x as usize;
            let y = y as usize;
            let z = z as usize;
            if envelope {
                solid_rows[z][y] |= 1 << x;
            }

            // axis 0 (Y-axis bits): plane XZ
            axis_cols[0][z][x] |= 1 << y;
            // axis 1 (X-axis bits): plane ZY
            axis_cols[1][y][z] |= 1 << x;
            // axis 2 (Z-axis bits): plane XY
            axis_cols[2][y][x] |= 1 << z;
        }
    }

    // Also need to check neighbors for the boundary faces?
    // The original algorithm loads neighbor voxels into a padded array.
    // We don't have a padded array, we have a hashmap of chunks.
    // We can simulate the "padded" check during face culling.
    // Actually, the bitwise culling `col & !(col << 1)` works great for internal faces.
    // For boundary faces (bit 0 and bit 15), we need to check neighbors.

    // Let's build the "face masks" - these are the faces that need to be meshed.
    // 6 sets of planes (2 per axis).
    // col_face_masks[axis*2 + 0]: faces pointing towards negative (e.g. Down)
    // col_face_masks[axis*2 + 1]: faces pointing towards positive (e.g. Up)
    let mut col_face_masks = [[[0u16; 16]; 16]; 6];

    for axis in 0..3 {
        // Optimization: Pre-calculate masks for i and j loops based on axis
        // Axis 0 (Y-cols, XZ plane): i=z, j=x. Skip if pz bit i is 0 or px bit j is 0.
        // Axis 1 (X-cols, ZY plane): i=y, j=z. Skip if py bit i is 0 or pz bit j is 0.
        // Axis 2 (Z-cols, XY plane): i=y, j=x. Skip if py bit i is 0 or px bit j is 0.
        let (i_mask, j_mask) = match axis {
            0 => (chunk.pz, chunk.px),
            1 => (chunk.py, chunk.pz),
            _ => (chunk.py, chunk.px),
        };

        // Hoist neighbor lookups
        let (neg_neighbor, pos_neighbor) = if let Some(neighs) = neighbors {
            let (nx, ny, nz) = match axis {
                0 => (0, -1, 0), // Y-axis neighbor (down)
                1 => (-1, 0, 0), // X-axis neighbor (left)
                _ => (0, 0, -1), // Z-axis neighbor (back)
            };
            (neighs.get(&(nx, ny, nz)), neighs.get(&(-nx, -ny, -nz)))
        } else {
            (None, None)
        };

        for i in 0..16 {
            // Skip if the entire row/plane at 'i' is empty
            if (i_mask & (1 << i)) == 0 {
                continue;
            }

            for j in 0..16 {
                // Skip if the column at 'j' is empty
                if (j_mask & (1 << j)) == 0 {
                    continue;
                }

                let col = axis_cols[axis][i][j];

                // Negative faces: `col & !(col << 1)`
                // Checks if bit k is 1 and k-1 is 0. Face is at k, pointing negative.
                let internal_neg = col & !(col << 1);

                // Positive faces: `col & !(col >> 1)`
                // Checks if bit k is 1 and k+1 is 0. Face is at k+1 (or k's positive side), pointing positive.
                let internal_pos = col & !(col >> 1);

                col_face_masks[2 * axis + 0][i][j] = internal_neg;
                col_face_masks[2 * axis + 1][i][j] = internal_pos;

                // Negative boundary (bit 0): if bit 0 is set, check neighbor at -1.
                if (col & 1) != 0 {
                    // Check neighbor. If neighbor has solid at 15, mask it out.
                    let has_neighbor_solid = if let Some(n_chunk) = neg_neighbor {
                        match axis {
                            0 => n_chunk.contains(j as u8, 15, i as u8), // Y-axis: i=z, j=x
                            1 => n_chunk.contains(15, i as u8, j as u8), // X-axis: i=y, j=z
                            _ => n_chunk.contains(j as u8, i as u8, 15), // Z-axis: i=y, j=x
                        }
                    } else {
                        false
                    };

                    if has_neighbor_solid {
                        col_face_masks[2 * axis + 0][i][j] &= !1;
                    }
                }

                // Positive boundary (bit 15): if bit 15 is set, check neighbor at 16.
                if (col & (1 << 15)) != 0 {
                    let has_neighbor_solid = if let Some(n_chunk) = pos_neighbor {
                        match axis {
                            0 => n_chunk.contains(j as u8, 0, i as u8),
                            1 => n_chunk.contains(0, i as u8, j as u8),
                            _ => n_chunk.contains(j as u8, i as u8, 0),
                        }
                    } else {
                        false
                    };

                    if has_neighbor_solid {
                        col_face_masks[2 * axis + 1][i][j] &= !(1 << 15);
                    }
                }
            }
        }
    }

    // Group exposed faces into 16x16 bit planes keyed by (face direction, depth, voxel type, AO).
    // One flat map to an index into a dense Vec of planes (no nested maps, no plane copies).
    // Planes are emitted in first-seen order, which is fully determined by the scan order below.
    let mut plane_index: HashMap<u32, usize> =
        HashMap::with_capacity_and_hasher(128, Default::default());
    let mut plane_keys: Vec<u32> = Vec::with_capacity(128);
    let mut planes: Vec<[u16; 16]> = Vec::with_capacity(128);

    for face_axis in 0..6usize {
        let axis = face_axis / 2;

        for i in 0..16 {
            for j in 0..16 {
                let mut mask = col_face_masks[face_axis][i][j];
                while mask != 0 {
                    let k = mask.trailing_zeros(); // Coordinate along the main axis (depth)
                    mask &= !(1 << k);

                    let (x, y, z) = match axis {
                        0 => (j, k as usize, i),
                        1 => (k as usize, i, j),
                        _ => (j, i, k as usize),
                    };

                    let voxel_type = if envelope {
                        chunk.dominant_type
                    } else {
                        dense[(z << 8) | (y << 4) | x]
                    };
                    let ao =
                        calculate_ao(chunk, neighbors, x as i32, y as i32, z as i32, face_axis);
                    // face_axis (3 bits) | depth (4 bits) | voxel type (8 bits) | AO (8 bits)
                    let key = ((face_axis as u32) << 20)
                        | (k << 16)
                        | ((voxel_type as u32) << 8)
                        | (ao as u32);

                    let pi = *plane_index.entry(key).or_insert_with(|| {
                        plane_keys.push(key);
                        planes.push([0u16; 16]);
                        planes.len() - 1
                    });

                    // row=j, bit=i
                    planes[pi][j] |= 1 << i;
                }
            }
        }
    }

    // Envelope meshes share one chunk-wide surface reflectivity; compute it once (and only when
    // there is geometry to emit), not once per plane.
    let envelope_reflectivity = if envelope && !planes.is_empty() {
        Some(representative_surface_reflectivity(
            chunk,
            palette,
            &SurfaceGrid {
                occupied: &occupied_rows,
                solid: &solid_rows,
                types: &dense,
            },
        ))
    } else {
        None
    };

    // Now generate mesh
    for (plane, &key) in planes.iter().zip(plane_keys.iter()) {
        let ao_mask = (key & 0xFF) as u8;
        let voxel_type = ((key >> 8) & 0xFF) as u8;
        let depth = (key >> 16) & 15;
        let face_axis = (key >> 20) as usize;
        let axis = face_axis / 2;
        let is_pos = (face_axis % 2) == 1;

        // Map our internal axis (0=Y, 1=X, 2=Z) to spatial axis (0=X, 1=Y, 2=Z)
        let spatial_axis = match axis {
            0 => 1, // Y
            1 => 0, // X
            _ => 2, // Z
        };

        let mut normal = [0.0, 0.0, 0.0];
        normal[spatial_axis] = if is_pos { 1.0 } else { -1.0 };

        let material = palette.material(voxel_type as u32);

        let base_color = if envelope {
            [
                chunk.average_color[0] as f32 / 255.0,
                chunk.average_color[1] as f32 / 255.0,
                chunk.average_color[2] as f32 / 255.0,
                1.0, // Force opaque for meshes
            ]
        } else {
            material.albedo
        };
        let emissive = if envelope {
            [0.0, 0.0, 0.0, 0.0]
        } else {
            [
                material.emissive[0],
                material.emissive[1],
                material.emissive[2],
                material.emissive_intensity,
            ]
        };
        let reflectivity = envelope_reflectivity.unwrap_or(material.reflectivity);
        let material_props = [reflectivity, 0.0, 0.0, 0.0];

        let quads = greedy_mesh_binary_plane(*plane);

        for q in quads {
            // Construct quad vertices
            // q.x = j (u-axis), q.y = i (v-axis)
            // q.w = width along u, q.h = height along v

            // Map u,v,depth back to x,y,z
            // axis 0 (Y): u=x, v=z, d=y. spatial: d=1, u=0, v=2
            // axis 1 (X): u=z, v=y, d=x. spatial: d=0, u=2, v=1
            // axis 2 (Z): u=x, v=y, d=z. spatial: d=2, u=0, v=1

            let (u_axis, v_axis) = match axis {
                0 => (0, 2), // x, z
                1 => (2, 1), // z, y
                _ => (0, 1), // x, y
            };

            // Coordinates
            let u0 = q.x as f32;
            let v0 = q.y as f32;
            let u1 = (q.x + q.w) as f32;
            let v1 = (q.y + q.h) as f32;
            let d = depth as f32 + if is_pos { 1.0 } else { 0.0 }; // Face offset

            // Create 4 vertices
            let mut p0 = [0.0; 3];
            let mut p1 = [0.0; 3];
            let mut p2 = [0.0; 3];
            let mut p3 = [0.0; 3];

            // Helper to set coords
            let set_coords = |p: &mut [f32; 3], u, v| {
                p[spatial_axis] = d;
                p[u_axis] = u;
                p[v_axis] = v;
            };

            set_coords(&mut p0, u0, v0);
            set_coords(&mut p1, u1, v0); // u+
            set_coords(&mut p2, u1, v1); // u+, v+
            set_coords(&mut p3, u0, v1); // v+

            // AO values from mask (packed 2 bits per corner? No, usually 1 value per vertex)
            // We stored AO as a mask of neighbors. We need to convert that to vertex AO.
            // This is complex. The original code packs AO into the key.
            // "we can only greedy mesh same block types + same ambient occlusion"
            // So all vertices in this quad share the same AO configuration?
            // No, AO is vertex-based.
            // If we group by AO, we force quads to have uniform AO on all vertices?
            // That would result in very small quads (no merging across AO changes).
            // But that's what the reference implementation does:
            // `let block_hash = ao_index | ((current_voxel.block_type as u32) << 9);`
            // It groups by the AO of the *face*.
            // Wait, AO is usually calculated per vertex by checking corner neighbors.
            // If we group by "face AO", we might be simplifying.

            // Let's look at the reference `append_vertices`:
            // `let v1ao = ((ao >> 0) & 1) + ...`
            // It unpacks the AO bits from the key to calculate AO for each vertex.
            // So yes, the key contains the neighborhood info (8 bits?), and we compute vertex AO from that.
            // This means we only merge faces that have identical neighbors.
            // This is correct for high-quality AO.

            let ao_bits = ao_mask;
            // Unpack AO for 4 corners.
            // We need a standard mapping of neighbors to bits.
            // Let's define the bits in `calculate_ao`.

            let (ao0, ao1, ao2, ao3) = calculate_vertex_ao(ao_bits);

            let base_idx = mesh.vertices.len() as u32;

            // Add vertices
            mesh.vertices.push(MeshVertex {
                position: p0,
                normal,
                color: apply_ao(base_color, ao0),
                emissive,
                material: material_props,
            });
            mesh.vertices.push(MeshVertex {
                position: p1,
                normal,
                color: apply_ao(base_color, ao1),
                emissive,
                material: material_props,
            });
            mesh.vertices.push(MeshVertex {
                position: p2,
                normal,
                color: apply_ao(base_color, ao2),
                emissive,
                material: material_props,
            });
            mesh.vertices.push(MeshVertex {
                position: p3,
                normal,
                color: apply_ao(base_color, ao3),
                emissive,
                material: material_props,
            });

            // Add indices
            // Determine winding order based on axis and direction
            // Axis 0 (Y): XZ plane. Cross(X, Z) = -Y. So +Y face needs flipped winding.
            // Axis 1 (X): ZY plane. Cross(Z, Y) = -X. So +X face needs flipped winding.
            // Axis 2 (Z): XY plane. Cross(X, Y) = +Z. So +Z face needs standard winding.

            let flip_winding = if axis == 2 { !is_pos } else { is_pos };

            if !flip_winding {
                // Standard winding: 0, 1, 2, 0, 2, 3
                mesh.indices.extend_from_slice(&[
                    base_idx,
                    base_idx + 1,
                    base_idx + 2,
                    base_idx,
                    base_idx + 2,
                    base_idx + 3,
                ]);
            } else {
                // Flipped winding: 0, 2, 1, 0, 3, 2
                mesh.indices.extend_from_slice(&[
                    base_idx,
                    base_idx + 2,
                    base_idx + 1,
                    base_idx,
                    base_idx + 3,
                    base_idx + 2,
                ]);
            }
        }
    }

    // Add emitters separately (simple iteration)
    for ((x, y, z), voxel) in chunk.iter() {
        if let Voxel::Solid(vtype) = voxel {
            let (_, strength) = palette.emissive(*vtype as u32);
            if strength > 0.0 {
                let (color, _) = palette.emissive(*vtype as u32);
                mesh.emitters.push(ChunkEmitter {
                    position: [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5],
                    color,
                    intensity: strength,
                });
            }
        }
    }

    mesh
}

fn calculate_ao(
    _chunk: &Chunk,
    _neighbors: Option<&HashMap<(i8, i8, i8), Arc<Chunk>>>,
    _x: i32,
    _y: i32,
    _z: i32,
    _face_axis: usize,
) -> u8 {
    // Calculate 8-bit mask of neighbors around the face
    // This is a simplified placeholder.
    // For full AO we need to check 8 neighbors in the plane of the face.
    // Let's return 0 (no occlusion) for now to get the meshing working,
    // then implement full AO logic if needed.
    // The original code uses `ADJACENT_AO_DIRS` to sample.
    0
}

fn calculate_vertex_ao(_mask: u8) -> (f32, f32, f32, f32) {
    // Placeholder: return 1.0 (white)
    (1.0, 1.0, 1.0, 1.0)
}

fn apply_ao(color: [f32; 4], ao: f32) -> [f32; 4] {
    [color[0] * ao, color[1] * ao, color[2] * ao, color[3]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib_hierarchical::Chunk;
    use crate::palette::Palette;

    #[test]
    fn test_generate_chunk_mesh_optimized_basic() {
        let palette = Palette::from_string("0 255 255 255 255\n1 255 255 255 255\n").unwrap();
        let mut chunk = Chunk::new();
        chunk.set(8, 8, 8, 1);

        let mesh = generate_chunk_mesh_optimized(&chunk, &palette, None, false);

        assert!(mesh.vertices.len() > 0);
        assert!(mesh.indices.len() > 0);
        // Should have 6 faces * 4 vertices = 24 vertices
        assert_eq!(mesh.vertices.len(), 24);
    }

    #[test]
    fn test_envelope_mesh_preserves_surface_reflectivity() {
        let palette = Palette::from_string(
            "\
0 255 255 255 255 0 0 0 0 0
1 50 50 50 255 0 0 0 0 0
2 100 120 140 255 0 0 0 0 200
",
        )
        .unwrap();
        let mut chunk = Chunk::new();
        for x in 0..8 {
            chunk.set(x, 0, 0, 1);
        }
        for x in 8..12 {
            chunk.set(x, 0, 0, 2);
        }

        let mesh = generate_chunk_mesh_optimized(&chunk, &palette, None, true);

        assert!(mesh.vertices.iter().any(|v| v.material[0] > 0.2));
    }

    fn mesh_signature(mesh: &ChunkMesh) -> (usize, usize, u64) {
        // order-independent checksum over quads (4 vertices each, in emission order)
        let mut sum = 0u64;
        for q in mesh.vertices.chunks(4) {
            let mut h = 0xcbf29ce484222325u64;
            for v in q {
                for c in v
                    .position
                    .iter()
                    .chain(v.normal.iter())
                    .chain(v.color.iter())
                {
                    h ^= c.to_bits() as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            sum = sum.wrapping_add(h);
        }
        (mesh.vertices.len(), mesh.indices.len(), sum)
    }

    #[test]
    #[ignore]
    fn bench_mesh_throughput() {
        let palette = Palette::from_string(
            "0 255 255 255 255\n1 200 50 50 255\n2 50 200 50 255\n3 50 50 200 255\n",
        )
        .unwrap();
        let mut chunk = Chunk::new();
        for z in 0..16u8 {
            for y in 0..12u8 {
                for x in 0..16u8 {
                    if (x as u32 + y as u32 + z as u32) % 5 != 0 {
                        chunk.set(x, y, z, 1 + (x / 4) % 3);
                    }
                }
            }
        }
        let mut neighbors: HashMap<(i8, i8, i8), Arc<Chunk>> = HashMap::default();
        neighbors.insert((1, 0, 0), Arc::new(chunk.clone()));
        let n = 2000;
        type MeshFn = fn(&Chunk, &Palette, Option<&Neighbors>, bool) -> ChunkMesh;
        let legacy: MeshFn = legacy_generate_chunk_mesh;
        let new: MeshFn = generate_chunk_mesh_optimized;
        for (name, nb) in [("no-neighbors", None), ("with-neighbor", Some(&neighbors))] {
            let mut sigs = Vec::new();
            for (label, f) in [("legacy", legacy), ("new", new)] {
                let t = std::time::Instant::now();
                let mut sig = (0, 0, 0);
                for _ in 0..n {
                    let m = f(&chunk, &palette, nb, false);
                    sig = mesh_signature(&m);
                }
                let el = t.elapsed();
                println!(
                    "MESHBENCH synthetic {name} {label}: {:.1} us/chunk, verts {}, idx {}, sig {:016x}",
                    el.as_secs_f64() * 1e6 / n as f64,
                    sig.0,
                    sig.1,
                    sig.2
                );
                sigs.push(sig);
            }
            assert_eq!(sigs[0], sigs[1], "legacy and new meshes differ ({name})");
        }
    }

    // Ported from eyetheisles MeshLayoutTests (bitmask greedy mesher parity checks).
    fn plain_palette() -> Palette {
        Palette::from_string("0 255 255 255 255\n1 255 255 255 255\n2 200 200 200 255\n").unwrap()
    }

    #[test]
    fn test_adjacent_voxels_merge_into_one_box() {
        let mut chunk = Chunk::new();
        chunk.set(2, 3, 4, 1);
        chunk.set(3, 3, 4, 1);
        let mesh = generate_chunk_mesh_optimized(&chunk, &plain_palette(), None, false);
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
        let xs: Vec<f32> = mesh.vertices.iter().map(|v| v.position[0]).collect();
        assert_eq!(xs.iter().cloned().fold(f32::MAX, f32::min), 2.0);
        assert_eq!(xs.iter().cloned().fold(f32::MIN, f32::max), 4.0);
    }

    #[test]
    fn test_material_boundaries_are_preserved() {
        let mut chunk = Chunk::new();
        chunk.set(2, 3, 4, 1);
        chunk.set(3, 3, 4, 2);
        let mesh = generate_chunk_mesh_optimized(&chunk, &plain_palette(), None, false);
        assert_eq!(mesh.vertices.len(), 40);
        assert_eq!(mesh.indices.len(), 60);
    }

    #[test]
    fn test_faces_culled_across_chunk_boundaries() {
        let mut chunk = Chunk::new();
        chunk.set(15, 3, 4, 1);
        let mut neighbor = Chunk::new();
        neighbor.set(0, 3, 4, 1);
        let mut neighbors: HashMap<(i8, i8, i8), Arc<Chunk>> = HashMap::default();
        neighbors.insert((1, 0, 0), Arc::new(neighbor));
        let mesh = generate_chunk_mesh_optimized(&chunk, &plain_palette(), Some(&neighbors), false);
        assert_eq!(mesh.vertices.len(), 20);
        assert_eq!(mesh.indices.len(), 30);

        // and the negative side
        let mut chunk = Chunk::new();
        chunk.set(0, 3, 4, 1);
        let mut neighbor = Chunk::new();
        neighbor.set(15, 3, 4, 1);
        let mut neighbors: HashMap<(i8, i8, i8), Arc<Chunk>> = HashMap::default();
        neighbors.insert((-1, 0, 0), Arc::new(neighbor));
        let mesh = generate_chunk_mesh_optimized(&chunk, &plain_palette(), Some(&neighbors), false);
        assert_eq!(mesh.vertices.len(), 20);
    }

    #[test]
    fn mesh_snapshot_sees_coarse_solid_neighbours() {
        // A coarse uniform `Voxel::Solid` next to a chunk must be snapshotted (as a full chunk of
        // its type) so the shared boundary face is culled, exactly like a subdivided neighbour.
        let mut world = World::new(3);
        world.set_uniform_solid([32, 0, 0], 16, 1);
        world.set(WorldPos::new(31, 5, 5), 1); // local (15, 5, 5) of the leaf chunk at (16, 0, 0)
        let key = (16, 0, 0);
        let chunk = world
            .get_leaf_chunk_arc_at_origin(WorldPos::new(key.0, key.1, key.2))
            .unwrap();

        let mut cache: HashMap<(i64, i64, i64), Arc<Chunk>> = HashMap::default();
        let neighbors = snapshot_mesh_neighbors(&world, key, &mut cache);
        assert!(
            neighbors.contains_key(&(1, 0, 0)),
            "+X coarse solid missing"
        );
        assert_eq!(neighbors.len(), 1, "only the coarse solid is a neighbour");

        let open = generate_chunk_mesh_optimized(&chunk, &plain_palette(), None, false);
        let covered =
            generate_chunk_mesh_optimized(&chunk, &plain_palette(), Some(&neighbors), false);
        assert_eq!(open.vertices.len(), 24);
        assert_eq!(
            covered.vertices.len(),
            20,
            "+X face is hidden by the coarse solid"
        );
    }

    #[test]
    fn test_greedy_meshing_plane() {
        // Test a simple 2x2 block in the middle
        let mut data = [0u16; 16];
        // Rows 4 and 5 have bits 4 and 5 set (0b00110000 = 48)
        data[4] = 48;
        data[5] = 48;

        let quads = greedy_mesh_binary_plane(data);

        assert_eq!(quads.len(), 1);
        let q = quads[0];
        assert_eq!(q.x, 4);
        assert_eq!(q.y, 4);
        assert_eq!(q.w, 2);
        assert_eq!(q.h, 2);
    }

    // ------------------------------------------------------------------------------------
    // Equivalence + benchmark harness against the pre-optimisation mesher and snapshot code
    // ------------------------------------------------------------------------------------

    use std::sync::OnceLock;

    type ArcCache = HashMap<(i64, i64, i64), Arc<Chunk>>;
    type Neighbors = HashMap<(i8, i8, i8), Arc<Chunk>>;

    /// The 27-lookup (3x3x3) neighbour snapshot the viewer used before `snapshot_mesh_neighbors`.
    fn legacy_snapshot_27(world: &World, key: (i64, i64, i64), cache: &mut ArcCache) -> Neighbors {
        let mut neighbors: Neighbors = HashMap::default();
        for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                for dz in -1i64..=1 {
                    let nx = key.0 + (dx << 4);
                    let ny = key.1 + (dy << 4);
                    let nz = key.2 + (dz << 4);
                    if let Some(nc) = world.get_leaf_chunk_arc_at_origin(WorldPos::new(nx, ny, nz))
                    {
                        let nk = (nx, ny, nz);
                        let arc_neigh = if let Some(existing) = cache.get(&nk) {
                            existing.clone()
                        } else {
                            cache.insert(nk, nc.clone());
                            nc
                        };
                        neighbors.insert((dx as i8, dy as i8, dz as i8), arc_neigh);
                    }
                }
            }
        }
        neighbors
    }

    /// Reflectivity estimate as originally written (Roaring `contains` per voxel face), for the
    /// equivalence tests.
    fn legacy_is_surface_voxel(chunk: &Chunk, x: u8, y: u8, z: u8) -> bool {
        x == 0
            || x == 15
            || y == 0
            || y == 15
            || z == 0
            || z == 15
            || !chunk.contains(x + 1, y, z)
            || !chunk.contains(x - 1, y, z)
            || !chunk.contains(x, y + 1, z)
            || !chunk.contains(x, y - 1, z)
            || !chunk.contains(x, y, z + 1)
            || !chunk.contains(x, y, z - 1)
    }

    fn legacy_representative_surface_reflectivity(chunk: &Chunk, palette: &Palette) -> f32 {
        let mut visible_total = 0u32;
        let mut reflective_total = 0u32;
        let mut reflectivity_sum = 0.0f32;
        let mut reflectivity_max = 0.0f32;

        for ((x, y, z), voxel) in chunk.iter() {
            if !legacy_is_surface_voxel(chunk, x, y, z) {
                continue;
            }

            let Voxel::Solid(voxel_type) = voxel else {
                continue;
            };

            let reflectivity = palette.reflectivity(*voxel_type as u32);
            visible_total = visible_total.saturating_add(1);
            reflectivity_sum += reflectivity;
            reflectivity_max = reflectivity_max.max(reflectivity);
            if reflectivity > 0.01 {
                reflective_total = reflective_total.saturating_add(1);
            }
        }

        if visible_total == 0 {
            return palette.reflectivity(chunk.dominant_type as u32);
        }

        let coverage = reflective_total as f32 / visible_total as f32;
        let average = reflectivity_sum / visible_total as f32;
        let coverage_boost = ((coverage - 0.03) / 0.22).clamp(0.0, 1.0);
        average.max(reflectivity_max * coverage_boost * 0.85)
    }

    /// Verbatim logic of the mesher before the dense-grid / flat-plane-map rewrite (only the
    /// explanatory comments were stripped). Kept as the behavioural reference for the tests below.
    fn legacy_generate_chunk_mesh(
        chunk: &Chunk,
        palette: &Palette,
        neighbors: Option<&HashMap<(i8, i8, i8), Arc<Chunk>>>,
        envelope: bool,
    ) -> ChunkMesh {
        let mut mesh = ChunkMesh::default();

        let mut axis_cols = [[[0u16; 16]; 16]; 3];

        for ((x, y, z), voxel) in chunk.iter() {
            if let Voxel::Solid(_) = voxel {
                let x = x as usize;
                let y = y as usize;
                let z = z as usize;

                axis_cols[0][z][x] |= 1 << y;
                axis_cols[1][y][z] |= 1 << x;
                axis_cols[2][y][x] |= 1 << z;
            }
        }

        let mut col_face_masks = [[[0u16; 16]; 16]; 6];

        for axis in 0..3 {
            let (i_mask, j_mask) = match axis {
                0 => (chunk.pz, chunk.px),
                1 => (chunk.py, chunk.pz),
                _ => (chunk.py, chunk.px),
            };

            let (neg_neighbor, pos_neighbor) = if let Some(neighs) = neighbors {
                let (nx, ny, nz) = match axis {
                    0 => (0, -1, 0), // Y-axis neighbor (down)
                    1 => (-1, 0, 0), // X-axis neighbor (left)
                    _ => (0, 0, -1), // Z-axis neighbor (back)
                };
                (neighs.get(&(nx, ny, nz)), neighs.get(&(-nx, -ny, -nz)))
            } else {
                (None, None)
            };

            for i in 0..16 {
                if (i_mask & (1 << i)) == 0 {
                    continue;
                }

                for j in 0..16 {
                    if (j_mask & (1 << j)) == 0 {
                        continue;
                    }

                    let col = axis_cols[axis][i][j];

                    let internal_neg = col & !(col << 1);

                    let internal_pos = col & !(col >> 1);

                    col_face_masks[2 * axis + 0][i][j] = internal_neg;
                    col_face_masks[2 * axis + 1][i][j] = internal_pos;

                    if (col & 1) != 0 {
                        let has_neighbor_solid = if let Some(n_chunk) = neg_neighbor {
                            match axis {
                                0 => n_chunk.contains(j as u8, 15, i as u8), // Y-axis: i=z, j=x
                                1 => n_chunk.contains(15, i as u8, j as u8), // X-axis: i=y, j=z
                                _ => n_chunk.contains(j as u8, i as u8, 15), // Z-axis: i=y, j=x
                            }
                        } else {
                            false
                        };

                        if has_neighbor_solid {
                            col_face_masks[2 * axis + 0][i][j] &= !1;
                        }
                    }

                    if (col & (1 << 15)) != 0 {
                        let has_neighbor_solid = if let Some(n_chunk) = pos_neighbor {
                            match axis {
                                0 => n_chunk.contains(j as u8, 0, i as u8),
                                1 => n_chunk.contains(0, i as u8, j as u8),
                                _ => n_chunk.contains(j as u8, i as u8, 0),
                            }
                        } else {
                            false
                        };

                        if has_neighbor_solid {
                            col_face_masks[2 * axis + 1][i][j] &= !(1 << 15);
                        }
                    }
                }
            }
        }

        let mut planes_by_depth: [HashMap<u32, HashMap<u8, [u16; 16]>>; 6] = [
            HashMap::default(),
            HashMap::default(),
            HashMap::default(),
            HashMap::default(),
            HashMap::default(),
            HashMap::default(),
        ];

        for face_axis in 0..6 {
            let axis = face_axis / 2;

            for i in 0..16 {
                for j in 0..16 {
                    let mut mask = col_face_masks[face_axis][i][j];
                    while mask != 0 {
                        let k = mask.trailing_zeros(); // Coordinate along the main axis (depth)
                        mask &= !(1 << k);

                        let (x, y, z) = match axis {
                            0 => (j, k as usize, i),
                            1 => (k as usize, i, j),
                            _ => (j, i, k as usize),
                        };

                        let voxel_type = if envelope {
                            chunk.dominant_type
                        } else {
                            chunk.get_type(x as u8, y as u8, z as u8).unwrap_or(0)
                        };
                        let ao =
                            calculate_ao(chunk, neighbors, x as i32, y as i32, z as i32, face_axis);
                        let key = ((voxel_type as u32) << 8) | (ao as u32);

                        let depth_map = planes_by_depth[face_axis].entry(key).or_default();
                        let plane = depth_map.entry(k as u8).or_insert([0u16; 16]);

                        plane[j] |= 1 << i;
                    }
                }
            }
        }

        for (face_axis, type_map) in planes_by_depth.iter().enumerate() {
            let axis = face_axis / 2;
            let is_pos = (face_axis % 2) == 1;

            let spatial_axis = match axis {
                0 => 1, // Y
                1 => 0, // X
                _ => 2, // Z
            };

            let mut normal = [0.0, 0.0, 0.0];
            normal[spatial_axis] = if is_pos { 1.0 } else { -1.0 };

            for (key, depth_map) in type_map {
                let voxel_type = (key >> 8) as u8;
                let ao_mask = (key & 0xFF) as u8;
                let material = palette.material(voxel_type as u32);

                let base_color = if envelope {
                    [
                        chunk.average_color[0] as f32 / 255.0,
                        chunk.average_color[1] as f32 / 255.0,
                        chunk.average_color[2] as f32 / 255.0,
                        1.0, // Force opaque for meshes
                    ]
                } else {
                    material.albedo
                };
                let emissive = if envelope {
                    [0.0, 0.0, 0.0, 0.0]
                } else {
                    [
                        material.emissive[0],
                        material.emissive[1],
                        material.emissive[2],
                        material.emissive_intensity,
                    ]
                };
                let reflectivity = if envelope {
                    legacy_representative_surface_reflectivity(chunk, palette)
                } else {
                    material.reflectivity
                };
                let material_props = [reflectivity, 0.0, 0.0, 0.0];

                for (depth, plane) in depth_map {
                    let quads = greedy_mesh_binary_plane(*plane);

                    for q in quads {
                        let (u_axis, v_axis) = match axis {
                            0 => (0, 2), // x, z
                            1 => (2, 1), // z, y
                            _ => (0, 1), // x, y
                        };

                        let u0 = q.x as f32;
                        let v0 = q.y as f32;
                        let u1 = (q.x + q.w) as f32;
                        let v1 = (q.y + q.h) as f32;
                        let d = *depth as f32 + if is_pos { 1.0 } else { 0.0 }; // Face offset

                        let mut p0 = [0.0; 3];
                        let mut p1 = [0.0; 3];
                        let mut p2 = [0.0; 3];
                        let mut p3 = [0.0; 3];

                        let set_coords = |p: &mut [f32; 3], u, v| {
                            p[spatial_axis] = d;
                            p[u_axis] = u;
                            p[v_axis] = v;
                        };

                        set_coords(&mut p0, u0, v0);
                        set_coords(&mut p1, u1, v0); // u+
                        set_coords(&mut p2, u1, v1); // u+, v+
                        set_coords(&mut p3, u0, v1); // v+

                        let ao_bits = ao_mask;

                        let (ao0, ao1, ao2, ao3) = calculate_vertex_ao(ao_bits);

                        let base_idx = mesh.vertices.len() as u32;

                        mesh.vertices.push(MeshVertex {
                            position: p0,
                            normal,
                            color: apply_ao(base_color, ao0),
                            emissive,
                            material: material_props,
                        });
                        mesh.vertices.push(MeshVertex {
                            position: p1,
                            normal,
                            color: apply_ao(base_color, ao1),
                            emissive,
                            material: material_props,
                        });
                        mesh.vertices.push(MeshVertex {
                            position: p2,
                            normal,
                            color: apply_ao(base_color, ao2),
                            emissive,
                            material: material_props,
                        });
                        mesh.vertices.push(MeshVertex {
                            position: p3,
                            normal,
                            color: apply_ao(base_color, ao3),
                            emissive,
                            material: material_props,
                        });

                        let flip_winding = if axis == 2 { !is_pos } else { is_pos };

                        if !flip_winding {
                            mesh.indices.extend_from_slice(&[
                                base_idx,
                                base_idx + 1,
                                base_idx + 2,
                                base_idx,
                                base_idx + 2,
                                base_idx + 3,
                            ]);
                        } else {
                            mesh.indices.extend_from_slice(&[
                                base_idx,
                                base_idx + 2,
                                base_idx + 1,
                                base_idx,
                                base_idx + 3,
                                base_idx + 2,
                            ]);
                        }
                    }
                }
            }
        }

        for ((x, y, z), voxel) in chunk.iter() {
            if let Voxel::Solid(vtype) = voxel {
                let (_, strength) = palette.emissive(*vtype as u32);
                if strength > 0.0 {
                    let (color, _) = palette.emissive(*vtype as u32);
                    mesh.emitters.push(ChunkEmitter {
                        position: [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5],
                        color,
                        intensity: strength,
                    });
                }
            }
        }

        mesh
    }

    struct CityFixture {
        world: World,
        palette: Palette,
        /// Every leaf chunk with its 16-aligned world origin, in traversal order.
        leaves: Vec<((i64, i64, i64), Arc<Chunk>)>,
        /// Smallest / largest leaf origin per axis.
        leaf_lo: [i64; 3],
        leaf_hi: [i64; 3],
    }

    fn collect_leaves(
        chunk: &Chunk,
        cell: i64,
        origin: (i64, i64, i64),
        out: &mut Vec<((i64, i64, i64), Arc<Chunk>)>,
    ) {
        for ((x, y, z), voxel) in chunk.iter() {
            let o = (
                origin.0 + x as i64 * cell,
                origin.1 + y as i64 * cell,
                origin.2 + z as i64 * cell,
            );
            if let Voxel::Chunk(child) = voxel {
                if cell == 16 {
                    out.push((o, child.clone()));
                } else {
                    collect_leaves(child, cell / 16, o, out);
                }
            }
        }
    }

    /// `worlds/flat_city_test.vhc` + its palette, loaded once and shared by the tests below.
    fn city() -> &'static CityFixture {
        static CITY: OnceLock<CityFixture> = OnceLock::new();
        CITY.get_or_init(|| {
            let world = crate::file_format::load_world_file(std::path::Path::new(
                "worlds/flat_city_test.vhc",
            ))
            .expect("load worlds/flat_city_test.vhc");
            let palette = Palette::load("worlds/palette.txt");
            let depth = world.hierarchy_depth();
            assert!(depth >= 2, "fixture needs a hierarchical world");
            let mut leaves = Vec::new();
            collect_leaves(
                world.root(),
                16i64.pow(depth as u32 - 1),
                (0, 0, 0),
                &mut leaves,
            );
            let mut leaf_lo = [i64::MAX; 3];
            let mut leaf_hi = [i64::MIN; 3];
            for (o, _) in &leaves {
                for (axis, c) in [o.0, o.1, o.2].into_iter().enumerate() {
                    leaf_lo[axis] = leaf_lo[axis].min(c);
                    leaf_hi[axis] = leaf_hi[axis].max(c);
                }
            }
            CityFixture {
                world,
                palette,
                leaves,
                leaf_lo,
                leaf_hi,
            }
        })
    }

    /// Deterministic sample of non-empty leaves: up to `per_group` evenly spread picks from each
    /// of (all leaves), (leaves on the world / occupied-region border) and (leaves missing a face
    /// neighbour), plus the eight densest chunks.
    fn sample_leaves(
        fx: &'static CityFixture,
        per_group: usize,
    ) -> Vec<&'static ((i64, i64, i64), Arc<Chunk>)> {
        fn spread(group: &[usize], cap: usize, out: &mut Vec<usize>) {
            let stride = (group.len() / cap.max(1)).max(1);
            out.extend(group.iter().step_by(stride));
        }
        let world_size = fx.world.world_size() as i64;
        let (lo, hi) = (fx.leaf_lo, fx.leaf_hi);
        let on_border =
            |c: i64, lo: i64, hi: i64| c == 0 || c + 16 == world_size || c == lo || c == hi;
        let origins: rustc_hash::FxHashSet<(i64, i64, i64)> =
            fx.leaves.iter().map(|(o, _)| *o).collect();
        let (mut all, mut border, mut missing) = (Vec::new(), Vec::new(), Vec::new());
        let mut densest: Vec<(u64, usize)> = Vec::new();
        for (i, ((x, y, z), chunk)) in fx.leaves.iter().enumerate() {
            if chunk.is_empty() {
                continue;
            }
            all.push(i);
            densest.push((chunk.count(), i));
            if on_border(*x, lo[0], hi[0])
                || on_border(*y, lo[1], hi[1])
                || on_border(*z, lo[2], hi[2])
            {
                border.push(i);
            }
            let missing_neighbor = MESH_NEIGHBOR_OFFSETS.iter().any(|&(dx, dy, dz)| {
                !origins.contains(&(
                    x + ((dx as i64) << 4),
                    y + ((dy as i64) << 4),
                    z + ((dz as i64) << 4),
                ))
            });
            if missing_neighbor {
                missing.push(i);
            }
        }
        assert!(!border.is_empty() && !missing.is_empty());
        let mut picked = Vec::new();
        spread(&all, per_group, &mut picked);
        spread(&border, per_group, &mut picked);
        spread(&missing, per_group, &mut picked);
        densest.sort_unstable_by(|a, b| b.cmp(a));
        picked.extend(densest.iter().take(8).map(|&(_, i)| i));
        picked.sort_unstable();
        picked.dedup();
        picked.into_iter().map(|i| &fx.leaves[i]).collect()
    }

    fn vertex_bits(v: &MeshVertex) -> [u32; 18] {
        let mut out = [0u32; 18];
        let floats = v
            .position
            .iter()
            .chain(v.normal.iter())
            .chain(v.color.iter())
            .chain(v.emissive.iter())
            .chain(v.material.iter());
        for (o, f) in out.iter_mut().zip(floats) {
            *o = f.to_bits();
        }
        out
    }

    /// Order-independent view of a mesh: each quad (4 vertices + its 6 indices rebased to the quad)
    /// as a flat bit pattern, sorted. Two meshes with the same set of quads compare equal even if
    /// the quads were emitted in a different order.
    fn canonical_quads(mesh: &ChunkMesh) -> Vec<Vec<u32>> {
        assert_eq!(mesh.vertices.len() % 4, 0);
        assert_eq!(mesh.indices.len(), mesh.vertices.len() / 4 * 6);
        let mut quads: Vec<Vec<u32>> = mesh
            .vertices
            .chunks_exact(4)
            .enumerate()
            .map(|(qi, vs)| {
                let mut q: Vec<u32> = Vec::with_capacity(78);
                for v in vs {
                    q.extend_from_slice(&vertex_bits(v));
                }
                let base = (qi * 4) as u32;
                for &ix in &mesh.indices[qi * 6..qi * 6 + 6] {
                    assert!(ix >= base && ix < base + 4, "index escapes its quad");
                    q.push(ix - base);
                }
                q
            })
            .collect();
        quads.sort_unstable();
        quads
    }

    fn emitter_bits(mesh: &ChunkMesh) -> Vec<[u32; 7]> {
        mesh.emitters
            .iter()
            .map(|e| {
                [
                    e.position[0].to_bits(),
                    e.position[1].to_bits(),
                    e.position[2].to_bits(),
                    e.color[0].to_bits(),
                    e.color[1].to_bits(),
                    e.color[2].to_bits(),
                    e.intensity.to_bits(),
                ]
            })
            .collect()
    }

    /// Same quads (order-independent), same vertex/index counts, same emitters (in order).
    fn assert_equivalent(new: &ChunkMesh, old: &ChunkMesh, what: &str) {
        assert_eq!(
            new.vertices.len(),
            old.vertices.len(),
            "{what}: vertex count"
        );
        assert_eq!(new.indices.len(), old.indices.len(), "{what}: index count");
        assert!(
            canonical_quads(new) == canonical_quads(old),
            "{what}: quads differ"
        );
        assert_eq!(
            emitter_bits(new),
            emitter_bits(old),
            "{what}: emitters differ"
        );
    }

    /// Exact equality including emission order (used when both sides run the same mesher).
    fn assert_identical(a: &ChunkMesh, b: &ChunkMesh, what: &str) {
        assert_eq!(a.indices, b.indices, "{what}: indices");
        assert_eq!(a.vertices.len(), b.vertices.len(), "{what}: vertex count");
        for (n, (va, vb)) in a.vertices.iter().zip(&b.vertices).enumerate() {
            assert_eq!(vertex_bits(va), vertex_bits(vb), "{what}: vertex {n}");
        }
        assert_eq!(emitter_bits(a), emitter_bits(b), "{what}: emitters");
    }

    #[test]
    fn test_mesher_matches_legacy_on_synthetic_chunks() {
        let palette = Palette::from_string(
            "\
0 255 255 255 255 0 0 0 0 0
1 200 50 50 255 0 0 0 0 0
2 50 200 50 255 255 200 100 4 120
3 50 50 200 255 0 0 0 0 200
",
        )
        .unwrap();

        let empty = Chunk::new();
        let full = Chunk::full(2);
        let mut mixed = Chunk::new();
        let mut state = 0x2545F4914F6CDD1Du64;
        for z in 0..16u8 {
            for y in 0..16u8 {
                for x in 0..16u8 {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    if state % 5 != 0 {
                        mixed.set(x, y, z, 1 + (state >> 8) as u8 % 3);
                    }
                }
            }
        }
        mixed.update_lod_metadata(&palette);
        let mut full_lod = full.clone();
        full_lod.update_lod_metadata(&palette);

        // A chunk that contains sub-chunk voxels (not a leaf): only Solid voxels are meshed.
        let mut with_sub = Chunk::new();
        with_sub.set(1, 1, 1, 1);
        with_sub.set_chunk(2, 1, 1, Chunk::full(3));
        with_sub.set(3, 1, 1, 2);

        let mut neighbors: Neighbors = HashMap::default();
        for &(dx, dy, dz) in &MESH_NEIGHBOR_OFFSETS {
            let nb = if dx + dy + dz > 0 {
                full.clone()
            } else {
                mixed.clone()
            };
            neighbors.insert((dx, dy, dz), Arc::new(nb));
        }
        let mut partial: Neighbors = HashMap::default();
        partial.insert((1, 0, 0), Arc::new(mixed.clone()));

        for (name, chunk) in [
            ("empty", &empty),
            ("full", &full_lod),
            ("mixed", &mixed),
            ("with_sub", &with_sub),
        ] {
            for nb in [None, Some(&partial), Some(&neighbors)] {
                for envelope in [false, true] {
                    let new = generate_chunk_mesh_optimized(chunk, &palette, nb, envelope);
                    let old = legacy_generate_chunk_mesh(chunk, &palette, nb, envelope);
                    let what = format!(
                        "{name} neighbors={} envelope={envelope}",
                        nb.map_or(0, |n| n.len())
                    );
                    assert_equivalent(&new, &old, &what);
                }
            }
        }

        // Output is deterministic: no hash-order dependence between runs or between clones.
        let a = generate_chunk_mesh_optimized(&mixed, &palette, Some(&neighbors), false);
        let b = generate_chunk_mesh_optimized(&mixed.clone(), &palette, Some(&neighbors), false);
        assert_identical(&a, &b, "mixed run-to-run");

        // Sanity on the extremes so the comparison above is not vacuous.
        let m = generate_chunk_mesh_optimized(&empty, &palette, None, false);
        assert!(m.vertices.is_empty() && m.indices.is_empty() && m.emitters.is_empty());
        let m = generate_chunk_mesh_optimized(&full, &palette, None, false);
        assert_eq!(m.vertices.len(), 24, "a full chunk merges into one box");
        let mut full_neighbors: Neighbors = HashMap::default();
        for &off in &MESH_NEIGHBOR_OFFSETS {
            full_neighbors.insert(off, Arc::new(full.clone()));
        }
        let m = generate_chunk_mesh_optimized(&full, &palette, Some(&full_neighbors), false);
        assert!(
            m.vertices.is_empty(),
            "every face is culled by full neighbours"
        );
        let mut one_side: Neighbors = HashMap::default();
        one_side.insert((0, 1, 0), Arc::new(full.clone()));
        let m = generate_chunk_mesh_optimized(&full, &palette, Some(&one_side), false);
        assert_eq!(m.vertices.len(), 20, "only the +Y face is culled");
    }

    #[test]
    fn test_mesher_matches_legacy_on_real_chunks() {
        let fx = city();
        let sample = sample_leaves(fx, 200);
        assert!(sample.len() >= 100, "sample too small: {}", sample.len());
        let mut cache = ArcCache::default();
        let mut quads_seen = 0usize;
        let mut reflective_envelopes = 0usize;
        for ((origin, chunk), n) in sample.iter().map(|l| (&l.0, &l.1)).zip(0..) {
            let neighbors = snapshot_mesh_neighbors(&fx.world, *origin, &mut cache);
            let mut lod_chunk = (**chunk).clone();
            lod_chunk.update_lod_metadata(&fx.palette);
            for (c, envelope) in [(&**chunk, false), (&lod_chunk, true)] {
                let new = generate_chunk_mesh_optimized(c, &fx.palette, Some(&neighbors), envelope);
                let old = legacy_generate_chunk_mesh(c, &fx.palette, Some(&neighbors), envelope);
                assert_equivalent(
                    &new,
                    &old,
                    &format!("chunk #{n} at {origin:?} envelope={envelope}"),
                );
                quads_seen += new.indices.len() / 6;
                if envelope && new.vertices.iter().any(|v| v.material[0] > 0.01) {
                    reflective_envelopes += 1;
                }
            }
        }
        assert!(
            quads_seen > 1000,
            "comparison was nearly vacuous ({quads_seen} quads)"
        );
        assert!(
            reflective_envelopes > 0,
            "no envelope mesh exercised a non-zero surface reflectivity"
        );
    }

    /// Every leaf chunk, exhaustively (slow in debug builds; run with `--release -- --ignored`).
    #[test]
    #[ignore]
    fn test_mesher_matches_legacy_on_all_real_chunks() {
        let fx = city();
        let mut cache = ArcCache::default();
        let mut checked = 0usize;
        for (origin, chunk) in &fx.leaves {
            let neighbors = snapshot_mesh_neighbors(&fx.world, *origin, &mut cache);
            let mut lod_chunk = (**chunk).clone();
            lod_chunk.update_lod_metadata(&fx.palette);
            for (c, envelope) in [(&**chunk, false), (&lod_chunk, true)] {
                let new = generate_chunk_mesh_optimized(c, &fx.palette, Some(&neighbors), envelope);
                let old = legacy_generate_chunk_mesh(c, &fx.palette, Some(&neighbors), envelope);
                assert_equivalent(&new, &old, &format!("{origin:?} envelope={envelope}"));
            }
            checked += 1;
        }
        println!("checked {checked} leaf chunks x 2 modes");
    }

    #[test]
    fn test_leaf_chunk_ref_lookup_matches_arc_lookup() {
        let fx = city();
        let size = fx.world.world_size() as i64;
        // Every leaf resolves to exactly its own chunk, through both lookups.
        for (origin, chunk) in &fx.leaves {
            let pos = WorldPos::new(origin.0, origin.1, origin.2);
            let new = fx.world.leaf_chunk_arc_ref_at_origin(pos).expect("leaf");
            assert!(Arc::ptr_eq(new, chunk), "different chunk at {pos:?}");
            let old = fx.world.get_leaf_chunk_arc_at_origin(pos).expect("leaf");
            assert!(Arc::ptr_eq(&old, chunk), "different chunk at {pos:?}");
        }
        // A strided scan of the occupied region plus a two-chunk halo (empty cells, negative and
        // out-of-world origins) agrees on presence/absence too.
        let (lo, hi) = (fx.leaf_lo, fx.leaf_hi);
        for gz in ((lo[2] >> 4) - 2..=(hi[2] >> 4) + 2).step_by(3) {
            for gy in ((lo[1] >> 4) - 2..=(hi[1] >> 4) + 2).step_by(2) {
                for gx in ((lo[0] >> 4) - 2..=(hi[0] >> 4) + 2).step_by(3) {
                    let pos = WorldPos::new(gx * 16, gy * 16, gz * 16);
                    let old = fx.world.get_leaf_chunk_arc_at_origin(pos);
                    let new = fx.world.leaf_chunk_arc_ref_at_origin(pos);
                    assert_eq!(old.is_some(), new.is_some(), "{pos:?}");
                }
            }
        }
        for pos in [
            WorldPos::new(1, 0, 0),
            WorldPos::new(0, 7, 0),
            WorldPos::new(0, 0, 15),
            WorldPos::new(-16, 0, 0),
            WorldPos::new(size, 0, 0),
            WorldPos::new(0, size + 16, 0),
        ] {
            assert!(
                fx.world.leaf_chunk_arc_ref_at_origin(pos).is_none(),
                "{pos:?}"
            );
            assert!(
                fx.world.get_leaf_chunk_arc_at_origin(pos).is_none(),
                "{pos:?}"
            );
        }
    }

    #[test]
    fn test_leaf_chunk_ref_lookup_small_worlds() {
        // depth 2: root children are leaf chunks
        let mut world = World::new(2);
        world.set(WorldPos::new(17, 3, 5), 1);
        world.set(WorldPos::new(255, 255, 255), 2);
        for gz in -1..=16 {
            for gy in -1..=16 {
                for gx in -1..=16 {
                    let pos = WorldPos::new(gx * 16, gy * 16, gz * 16);
                    let old = world.get_leaf_chunk_arc_at_origin(pos);
                    let new = world.leaf_chunk_arc_ref_at_origin(pos);
                    assert_eq!(old.is_some(), new.is_some(), "{pos:?}");
                    if let (Some(a), Some(b)) = (old, new) {
                        assert!(Arc::ptr_eq(&a, b));
                    }
                }
            }
        }

        // depth 3: a Solid region above leaf level resolves to the shared full chunk of its type
        let mut world = World::new(3);
        world.set(WorldPos::new(300, 20, 40), 1);
        world.set(WorldPos::new(2000, 20, 40), 1);
        // a Solid voxel at level 2 (a 256^3 cell of the depth-3 world), not a chunk
        world.root_mut().set(5, 0, 0, 3);
        fn leaf_at(w: &World, x: i64, y: i64, z: i64) -> Option<&Arc<Chunk>> {
            w.leaf_chunk_arc_ref_at_origin(WorldPos::new(x, y, z))
        }
        assert!(leaf_at(&world, 288, 16, 32).is_some());
        assert!(leaf_at(&world, 2000, 16, 32).is_some());
        assert!(leaf_at(&world, 304, 16, 32).is_none());
        let coarse = leaf_at(&world, 1296, 0, 0).expect("coarse solid must resolve");
        assert_eq!(coarse.count(), 4096);
        assert_eq!(coarse.get_type(7, 7, 7), Some(3));
        for (x, y, z) in [
            (288, 16, 32),
            (2000, 16, 32),
            (304, 16, 32),
            (1296, 0, 0),
            (0, 0, 0),
            (4080, 0, 0),
        ] {
            let pos = WorldPos::new(x, y, z);
            assert_eq!(
                world
                    .get_leaf_chunk_arc_at_origin(pos)
                    .map(|a| Arc::as_ptr(&a)),
                world.leaf_chunk_arc_ref_at_origin(pos).map(Arc::as_ptr),
                "{pos:?}"
            );
        }

        // depth 1: only the origin resolves (the root is the single chunk)
        let world = World::new(1);
        assert!(world
            .leaf_chunk_arc_ref_at_origin(WorldPos::new(0, 0, 0))
            .is_some());
        assert!(world
            .leaf_chunk_arc_ref_at_origin(WorldPos::new(16, 0, 0))
            .is_none());
        assert!(world
            .leaf_chunk_arc_ref_at_origin(WorldPos::new(0, -16, 0))
            .is_none());
    }

    /// The mesher only ever reads the six face neighbours; assert that the new snapshot holds
    /// exactly those, shares the `Arc`s with the old 27-lookup snapshot, and (below) meshes the
    /// same.
    #[test]
    fn test_snapshot_6_matches_27_for_every_leaf_chunk() {
        let fx = city();
        let mut cache27 = ArcCache::default();
        let mut cache6 = ArcCache::default();
        let mut border_chunks = 0usize;
        for (origin, _) in &fx.leaves {
            let old = legacy_snapshot_27(&fx.world, *origin, &mut cache27);
            let new = snapshot_mesh_neighbors(&fx.world, *origin, &mut cache6);
            for off in MESH_NEIGHBOR_OFFSETS {
                match (old.get(&off), new.get(&off)) {
                    (Some(a), Some(b)) => assert!(Arc::ptr_eq(a, b), "{origin:?} {off:?}"),
                    (None, None) => {}
                    _ => panic!("snapshot mismatch at {origin:?} {off:?}"),
                }
            }
            assert!(new.len() <= 6);
            assert!(new.keys().all(|k| MESH_NEIGHBOR_OFFSETS.contains(k)));
            if new.len() < 6 {
                border_chunks += 1;
            }
        }
        assert!(
            border_chunks > 0,
            "fixture has no chunks at the world/region border"
        );
    }

    #[test]
    fn test_mesh_from_6_neighbor_snapshot_equals_mesh_from_27() {
        let fx = city();
        let mut cache27 = ArcCache::default();
        let mut cache6 = ArcCache::default();
        let mut checked = 0usize;
        // Stride sample plus chunks on the world / region border and chunks missing a neighbour.
        let sample = sample_leaves(fx, 200);
        for (origin, chunk) in sample.iter().map(|l| (&l.0, &l.1)) {
            let old = legacy_snapshot_27(&fx.world, *origin, &mut cache27);
            let new = snapshot_mesh_neighbors(&fx.world, *origin, &mut cache6);
            let mut lod_chunk = (**chunk).clone();
            lod_chunk.update_lod_metadata(&fx.palette);
            for (c, envelope) in [(&**chunk, false), (&lod_chunk, true)] {
                let a = generate_chunk_mesh_optimized(c, &fx.palette, Some(&old), envelope);
                let b = generate_chunk_mesh_optimized(c, &fx.palette, Some(&new), envelope);
                assert_identical(&a, &b, &format!("{origin:?} envelope={envelope}"));
            }
            checked += 1;
        }
        assert!(checked >= 100);
    }

    /// Meshes from the two snapshots for *every* leaf chunk (slow in debug; use --release).
    #[test]
    #[ignore]
    fn test_mesh_from_6_neighbor_snapshot_equals_mesh_from_27_all_chunks() {
        let fx = city();
        let mut cache27 = ArcCache::default();
        let mut cache6 = ArcCache::default();
        for (origin, chunk) in &fx.leaves {
            let old = legacy_snapshot_27(&fx.world, *origin, &mut cache27);
            let new = snapshot_mesh_neighbors(&fx.world, *origin, &mut cache6);
            let a = generate_chunk_mesh_optimized(chunk, &fx.palette, Some(&old), false);
            let b = generate_chunk_mesh_optimized(chunk, &fx.palette, Some(&new), false);
            assert_identical(&a, &b, &format!("{origin:?}"));
        }
        println!("checked {} leaf chunks", fx.leaves.len());
    }

    #[test]
    fn test_snapshot_reuses_cached_arc_and_skips_missing() {
        let mut world = World::new(2);
        world.set(WorldPos::new(16, 0, 0), 1); // chunk (16,0,0)
        world.set(WorldPos::new(0, 0, 0), 1); // chunk (0,0,0)
        let mut cache = ArcCache::default();
        let snap = snapshot_mesh_neighbors(&world, (0, 0, 0), &mut cache);
        assert_eq!(
            snap.len(),
            1,
            "only +X exists; -X/±Y/±Z are outside the world or empty"
        );
        let pos_x = snap.get(&(1, 0, 0)).expect("+X neighbour");
        assert!(Arc::ptr_eq(pos_x, cache.get(&(16, 0, 0)).unwrap()));
        assert_eq!(cache.len(), 1, "only chunks that exist are cached");

        // A cached Arc wins over the one in the world (callers drop entries on edits).
        let stale = Arc::new(Chunk::new());
        cache.insert((16, 0, 0), stale.clone());
        let snap = snapshot_mesh_neighbors(&world, (0, 0, 0), &mut cache);
        assert!(Arc::ptr_eq(snap.get(&(1, 0, 0)).unwrap(), &stale));
    }

    #[test]
    #[ignore]
    fn bench_mesher_vs_legacy_real_chunks() {
        let fx = city();
        let mut cache = ArcCache::default();
        let work: Vec<(Arc<Chunk>, Chunk, Neighbors)> = fx
            .leaves
            .iter()
            .filter(|(_, c)| !c.is_empty())
            .map(|(origin, c)| {
                let mut lod = (**c).clone();
                lod.update_lod_metadata(&fx.palette);
                (
                    c.clone(),
                    lod,
                    snapshot_mesh_neighbors(&fx.world, *origin, &mut cache),
                )
            })
            .collect();
        let n = work.len();
        let rounds = 5usize;
        println!("MESHBENCH real chunks: {n} non-empty leaves, {rounds} rounds");
        for envelope in [false, true] {
            let mut best_old = f64::MAX;
            let mut best_new = f64::MAX;
            let mut quads = (0usize, 0usize);
            for _ in 0..rounds {
                let t = std::time::Instant::now();
                let mut q = 0usize;
                for (c, lod, nb) in &work {
                    let c = if envelope { lod } else { &**c };
                    q += legacy_generate_chunk_mesh(c, &fx.palette, Some(nb), envelope)
                        .indices
                        .len();
                }
                best_old = best_old.min(t.elapsed().as_secs_f64());
                quads.0 = q / 6;

                let t = std::time::Instant::now();
                let mut q = 0usize;
                for (c, lod, nb) in &work {
                    let c = if envelope { lod } else { &**c };
                    q += generate_chunk_mesh_optimized(c, &fx.palette, Some(nb), envelope)
                        .indices
                        .len();
                }
                best_new = best_new.min(t.elapsed().as_secs_f64());
                quads.1 = q / 6;
            }
            println!(
                "MESHBENCH envelope={envelope}: legacy {:.1} us/chunk, new {:.1} us/chunk, speedup {:.2}x (quads {} vs {})",
                best_old * 1e6 / n as f64,
                best_new * 1e6 / n as f64,
                best_old / best_new,
                quads.0,
                quads.1
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_neighbor_snapshot_27_vs_6() {
        let fx = city();
        let origins: Vec<(i64, i64, i64)> = fx.leaves.iter().map(|l| l.0).collect();
        let n = origins.len();
        let rounds = 20usize;
        let mut best27 = f64::MAX;
        let mut best6 = f64::MAX;
        for _ in 0..rounds {
            let mut cache = ArcCache::default();
            let t = std::time::Instant::now();
            let mut total = 0usize;
            for o in &origins {
                total += legacy_snapshot_27(&fx.world, *o, &mut cache).len();
            }
            best27 = best27.min(t.elapsed().as_secs_f64());
            std::hint::black_box(total);

            let mut cache = ArcCache::default();
            let t = std::time::Instant::now();
            let mut total = 0usize;
            for o in &origins {
                total += snapshot_mesh_neighbors(&fx.world, *o, &mut cache).len();
            }
            best6 = best6.min(t.elapsed().as_secs_f64());
            std::hint::black_box(total);
        }
        println!(
            "SNAPBENCH {n} jobs: 27-lookup {:.2} us/job, 6-neighbour {:.2} us/job, speedup {:.1}x",
            best27 * 1e6 / n as f64,
            best6 * 1e6 / n as f64,
            best27 / best6
        );
    }
}
