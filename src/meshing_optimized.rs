//! Optimized greedy meshing using bitwise operations
//! Based on https://github.com/TanTanDev/binary_greedy_mesher_demo

use crate::lib_hierarchical::{Chunk, Voxel};
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

    // Helper to add a voxel to the bitmasks
    // We can optimize this by using the chunk's existing marginals to skip empty areas?
    // For now, let's just iterate the chunk's sparse storage which is already efficient.
    for ((x, y, z), voxel) in chunk.iter() {
        if let Voxel::Solid(_) = voxel {
            let x = x as usize;
            let y = y as usize;
            let z = z as usize;

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

    // RESTARTING the grouping logic to include depth
    // Map key: (VoxelType << 8) | AO_mask
    // Value: HashMap<depth, [u16; 16]>
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
                        0
                    } else {
                        chunk.get_type(x as u8, y as u8, z as u8).unwrap_or(0)
                    };
                    let ao =
                        calculate_ao(chunk, neighbors, x as i32, y as i32, z as i32, face_axis);
                    let key = ((voxel_type as u32) << 8) | (ao as u32);

                    let depth_map = planes_by_depth[face_axis].entry(key).or_default();
                    let plane = depth_map.entry(k as u8).or_insert([0u16; 16]);

                    // row=j, bit=i
                    plane[j] |= 1 << i;
                }
            }
        }
    }

    // Now generate mesh
    for (face_axis, type_map) in planes_by_depth.iter().enumerate() {
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
            let material_props = if envelope {
                [0.0, 0.0, 0.0, 0.0]
            } else {
                [material.reflectivity, 0.0, 0.0, 0.0]
            };

            for (depth, plane) in depth_map {
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
                    let d = *depth as f32 + if is_pos { 1.0 } else { 0.0 }; // Face offset

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
}
