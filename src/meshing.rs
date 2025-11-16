//! Greedy meshing for bottom-level chunks (16x16x16)

use crate::lib_hierarchical::{Chunk, Voxel};
use crate::palette::Palette;
use std::collections::{HashMap, HashSet};

macro_rules! mesh_debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "mesh-debug") {
            eprintln!($($arg)*);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib_hierarchical::Chunk;
    use crate::palette::Palette;

    #[test]
    fn test_generate_chunk_mesh_ao_present() {
        // Construct a small palette with indices 0..=1
        let palette = Palette::from_string("0 255 255 255 255\n1 255 255 255 255\n").unwrap();

        // Create a chunk and set a small cluster of voxels to cause AO
        let mut chunk = Chunk::new();
        // Central voxel
        chunk.set(8, 8, 8, 1);
        // Place some neighbors to reduce AO at one corner
        chunk.set(7, 8, 8, 1);
        chunk.set(8, 7, 8, 1);
        chunk.set(8, 8, 7, 1);

        // Generate mesh with full AO strength
        let mesh = generate_chunk_mesh(&chunk, &palette, 1.0);

        // There should be at least one vertex where the alpha channel is less than 1.0
        // (meaning AO was applied), and all alpha values should be in [0.0, 1.0]
        assert!(mesh.vertices.len() > 0);
        let mut found_less_than_one = false;
        for v in &mesh.vertices {
            let a = v.color[3];
            assert!(a >= 0.0 && a <= 1.0);
            if a < 1.0 {
                found_less_than_one = true;
            }
        }
        assert!(found_less_than_one, "No vertex had AO < 1.0 (expected some occlusion)");
    }

    #[test]
    fn test_generate_chunk_mesh_isolated_no_ao() {
        let palette = Palette::from_string("0 255 255 255 255\n1 255 255 255 255\n").unwrap();
        let mut chunk = Chunk::new();
        // Single isolated voxel
        chunk.set(8, 8, 8, 1);
        let mesh = generate_chunk_mesh(&chunk, &palette, 1.0);
        // All vertex alphas should be 1.0
        assert!(mesh.vertices.len() > 0);
        for v in &mesh.vertices {
            let a = v.color[3];
            assert!(a >= 0.9999 && a <= 1.0, "Expected AO alpha near 1.0 for isolated voxel, got {}", a);
        }
    }

    #[test]
    fn test_generate_chunk_mesh_ground_isolated_no_ao() {
        let palette = Palette::from_string("0 255 255 255 255\n1 255 255 255 255\n").unwrap();
        let mut chunk = Chunk::new();
        // Single isolated voxel at the ground (y = 0)
        chunk.set(8, 0, 8, 1);
        let mesh = generate_chunk_mesh(&chunk, &palette, 1.0);
        // All vertex alphas should be near 1.0
        assert!(mesh.vertices.len() > 0);
        for v in &mesh.vertices {
            let a = v.color[3];
            assert!(a >= 0.9999 && a <= 1.0, "Expected AO alpha near 1.0 for ground isolated voxel, got {}", a);
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub emissive: [f32; 4],
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

// Representation of a merged quad (rectangle) created during greedy meshing.
#[derive(Clone, Copy, Debug)]
struct Quad {
    axis: usize,
    d: i32,
    u_axis: usize,
    v_axis: usize,
    u0: i32,
    v0: i32,
    du: i32,
    dv: i32,
    face_type: i32,
}

/// Generate a greedy mesh for a 16x16x16 chunk.
/// Merges coplanar faces with identical voxel types into larger quads.
pub fn generate_chunk_mesh(chunk: &Chunk, palette: &Palette, ao_strength: f32) -> ChunkMesh {
    // 3 axes: 0=x, 1=y, 2=z
    let mut mesh = ChunkMesh::default();

    // Debug: count voxels
    let voxel_count = chunk.iter().count();
    let first_voxel = chunk.iter().next();
    mesh_debug!(
        "DEBUG generate_chunk_mesh: {} voxels, first={:?}",
        voxel_count,
        first_voxel
    );

    // Capture all emissive voxels (center positions within the chunk)
    for ((x, y, z), voxel) in chunk.iter() {
        if let Voxel::Solid(voxel_type) = voxel {
            let (emissive_color, emissive_intensity) = palette.emissive(*voxel_type as u32);
            let has_emission = emissive_intensity > 0.0
                && (emissive_color[0] > 0.0 || emissive_color[1] > 0.0 || emissive_color[2] > 0.0);
            if has_emission {
                mesh.emitters.push(ChunkEmitter {
                    position: [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5],
                    color: emissive_color,
                    intensity: emissive_intensity,
                });
            }
        }
    }

    // Helper to get voxel type at (x,y,z)
    let get_type = |x: i32, y: i32, z: i32| -> Option<u8> {
        if x < 0 || x >= 16 || y < 0 || y >= 16 || z < 0 || z >= 16 {
            return None;
        }
        chunk.get_type(x as u8, y as u8, z as u8)
    };

    // For each axis, create faces between differing neighbor voxels
    let mut quads: Vec<Quad> = Vec::new();

    for axis in 0..3 {
        // The other two axes form the 2D mask
        let (u_axis, v_axis) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };

        let axis_name = ["X", "Y", "Z"][axis];
        let mut faces_this_axis = 0;

        // Iterate slices along the main axis
        for d in 0..=16 {
            // note: d == 16 handles the outer boundary
            // Mask width/height along u and v axes
            let mut mask: [i32; 16 * 16] = [0; 16 * 16];

            // Build mask: positive faces at boundary between solid and empty/different type
            for u in 0..16 {
                for v in 0..16 {
                    // Convert (axis,u,v) to (x,y,z)
                    let mut a = [0i32; 3];
                    a[axis] = d;
                    a[u_axis] = u;
                    a[v_axis] = v;

                    let mut b = a;
                    b[axis] = d - 1; // neighbor on negative side

                    let t_a = get_type(a[0], a[1], a[2]);
                    let t_b = get_type(b[0], b[1], b[2]);

                    // We create a face when one side is solid and the other isn't, or materials differ
                    let face_type = match (t_a, t_b) {
                        (Some(ta), None) => ta as i32, // Solid on 'a' side, empty on 'b' side: positive face at d
                        (None, Some(tb)) => -(tb as i32), // Empty on 'a' side, solid on 'b' side: negative face at d-1
                        (Some(ta), Some(tb)) if ta != tb => ta as i32, // boundary between different types
                        _ => 0,
                    };

                    mask[(u * 16 + v) as usize] = face_type;
                }
            }

            // Greedy merge rectangles in mask
            let mut v_start = 0;
            while v_start < 16 {
                let mut u_start = 0;
                while u_start < 16 {
                    let t = mask[(u_start * 16 + v_start) as usize];
                    if t == 0 {
                        u_start += 1;
                        continue;
                    }

                    // Compute width
                    let mut width = 1;
                    while u_start + width < 16 {
                        let idx = ((u_start + width) * 16 + v_start) as usize;
                        if mask[idx] != t {
                            break;
                        }
                        width += 1;
                    }

                    // Compute height
                    let mut height = 1;
                    'outer: while v_start + height < 16 {
                        for du in 0..width {
                            let idx = ((u_start + du) * 16 + (v_start + height)) as usize;
                            if mask[idx] != t {
                                break 'outer;
                            }
                        }
                        height += 1;
                    }

                    // Collect this rectangle for second pass: we need to compute
                    // corner bitmasks and owners for AO caching before final emission.
                    quads.push(Quad {
                        axis,
                        d,
                        u_axis,
                        v_axis,
                        u0: u_start,
                        v0: v_start,
                        du: width,
                        dv: height,
                        face_type: t,
                    });
                    faces_this_axis += 1;

                    // Mark used
                    for du in 0..width {
                        for dv in 0..height {
                            mask[((u_start + du) * 16 + (v_start + dv)) as usize] = 0;
                        }
                    }

                    u_start += width;
                }
                v_start += 1;
            }
        }
        mesh_debug!("  {} axis: {} faces", axis_name, faces_this_axis);
    }

    // Build corner masks and owners from collected quads
    let mut corner_mask: HashMap<(i32, i32, i32), u8> = HashMap::new();
    let mut corner_owners: HashMap<(i32, i32, i32), HashSet<(i32, i32, i32)>> = HashMap::new();

    for quad in &quads {
        let Quad { axis, d, u_axis, v_axis, u0, v0, du, dv, face_type } = *quad;
        let positive = face_type > 0;

        // compute owner coords for this quad; these should be excluded from AO samples
        let mut owner_coords: Vec<[i32; 3]> = Vec::new();
        for ou in u0..(u0 + du) {
            for ov in v0..(v0 + dv) {
                let mut owner = [0i32; 3];
                owner[axis] = if positive { d } else { d - 1 };
                owner[u_axis] = ou;
                owner[v_axis] = ov;
                owner_coords.push(owner);
            }
        }

        // compute the 4 voxel-space corner coordinates for this rectangle
        let mut du_vec = [0i32; 3];
        du_vec[u_axis] = du;
        let mut dv_vec = [0i32; 3];
        dv_vec[v_axis] = dv;
        let mut base = [0i32; 3];
        base[axis] = d;
        base[u_axis] = u0;
        base[v_axis] = v0;

        let bases = [
            base,
            [base[0] + du_vec[0], base[1] + du_vec[1], base[2] + du_vec[2]],
            [base[0] + du_vec[0] + dv_vec[0], base[1] + du_vec[1] + dv_vec[1], base[2] + du_vec[2] + dv_vec[2]],
            [base[0] + dv_vec[0], base[1] + dv_vec[1], base[2] + dv_vec[2]],
        ];

        // axis bit: we only keep axis identity (bit 0 = X, bit 1 = Y, bit 2 = Z)
        let axis_bit = 1u8 << (axis as u8);
        for bases_corner in &bases {
            let key = (bases_corner[0], bases_corner[1], bases_corner[2]);
            let entry = corner_mask.entry(key).or_insert(0u8);
            *entry |= axis_bit;

            let owners = corner_owners.entry(key).or_insert_with(HashSet::new);
            for o in &owner_coords {
                owners.insert((o[0], o[1], o[2]));
            }
        }
    }

    // AO cache: compute for corners with >= 2 different axis bits set (orthogonal)
    let mut ao_cache: HashMap<(i32, i32, i32), f32> = HashMap::new();
    for (key, mask) in &corner_mask {
        // count orthogonal axis bits
        let mut axis_count = 0;
        for a in 0..3 {
            if (mask & (1u8 << a)) != 0 {
                axis_count += 1;
            }
        }
        if axis_count >= 2 {
            // Quick hack: skip AO for corners on the chunk boundary. Neighbor chunk
            // voxels are not visible to this per-chunk mesher, which causes sharp
            // seams; by skipping AO computation for boundary corners we avoid
            // the abrupt darkening at chunk borders. This will lose AO at some
            // seams but removes the visible artifact.
            if key.0 == 0 || key.1 == 0 || key.2 == 0 || key.0 == 16 || key.1 == 16 || key.2 == 16 {
                continue;
            }
            // compute 3-sample AO for this corner using the owner set to avoid self-occlusion
            let owners_set = corner_owners.get(&key).unwrap();
            let mut count = 0u32;
            let mut samples = 0u32;
            for dx in -1..=0 {
                for dy in -1..=0 {
                    for dz in -1..=0 {
                        let sx = key.0 + dx;
                        let sy = key.1 + dy;
                        let sz = key.2 + dz;
                        // skip owner voxels
                        if owners_set.contains(&(sx, sy, sz)) {
                            continue;
                        }
                        samples += 1;
                        if get_type(sx, sy, sz).is_some() {
                            count += 1;
                        }
                    }
                }
            }
            let occ = if samples > 0 { count as f32 / samples as f32 } else { 0.0 };
            let ao = 1.0 - (occ * ao_strength);
            ao_cache.insert(*key, ao.clamp(0.0, 1.0));
        }
    }

    // Now emit quads using AO cache for candidate corners
    for quad in &quads {
        emit_quad_with_ao_cache(&mut mesh, palette, *quad, &get_type, &ao_cache);
    }

    mesh
}

fn emit_quad<F>(
    mesh: &mut ChunkMesh,
    palette: &Palette,
    axis: usize,
    d: i32,
    u_axis: usize,
    v_axis: usize,
    u0: i32,
    v0: i32,
    du: i32,
    dv: i32,
    face_type: i32,
    get_type: &F,
    ao_strength: f32,
) where F: Fn(i32, i32, i32) -> Option<u8> {
    if face_type == 0 {
        return;
    }

    // Determine face orientation and material
    let positive = face_type > 0;
    let mat = face_type.abs() as u8;

    // Base corner in voxel space
    // Face is always at position 'd' along the axis (the boundary slice)
    let mut base = [0i32; 3];
    base[axis] = d;
    base[u_axis] = u0;
    base[v_axis] = v0;

    // Size vectors along u and v
    let mut du_vec = [0i32; 3];
    du_vec[u_axis] = du;
    let mut dv_vec = [0i32; 3];
    dv_vec[v_axis] = dv;

    // Normal (pointing away from solid, toward empty)
    let mut normal = [0.0f32; 3];
    normal[axis] = if positive { -1.0 } else { 1.0 };

    // Color and emissive data from palette
    let material = palette.material(mat as u32);
    let color = material.albedo;
    let emissive = [
        material.emissive[0],
        material.emissive[1],
        material.emissive[2],
        material.emissive_intensity,
    ];

    // Helper to compute simple AO at a corner by sampling the 8 voxels touching the corner
    // To avoid self-shadowing, exclude owner voxels (the voxels that own this face)
    let mut owner_coords: Vec<[i32; 3]> = Vec::new();
    for ou in u0..(u0 + du) {

    // End of owner loop

    // Now emit quads using AO cache for candidate corners
        for ov in v0..(v0 + dv) {
            let mut owner = [0i32; 3];
            owner[axis] = if positive { d } else { d - 1 };
            owner[u_axis] = ou;
            owner[v_axis] = ov;
            owner_coords.push(owner);
        }
    }

    let compute_ao = |x: i32, y: i32, z: i32| -> f32 {
        let mut count = 0u32;
        let mut samples = 0u32;
        for dx in -1..=0 {
            for dy in -1..=0 {
                for dz in -1..=0 {
                    let sx = x + dx;
                    let sy = y + dy;
                    let sz = z + dz;
                    // Skip owner's own voxels so the block doesn't occlude itself
                    if owner_coords.iter().any(|o| o[0] == sx && o[1] == sy && o[2] == sz) {
                        continue;
                    }
                    samples += 1;
                    if get_type(sx, sy, sz).is_some() {
                        count += 1;
                    }
                }
            }
        }
        let occ = if samples > 0 { count as f32 / samples as f32 } else { 0.0 };
        let ao = 1.0 - (occ * ao_strength);
        ao.clamp(0.0, 1.0)
    };

    // Four corners (in voxel space), convert to f32
    let p0 = [base[0] as f32, base[1] as f32, base[2] as f32];
    let p1 = [
        (base[0] + du_vec[0]) as f32,
        (base[1] + du_vec[1]) as f32,
        (base[2] + du_vec[2]) as f32,
    ];
    let p2 = [
        (base[0] + du_vec[0] + dv_vec[0]) as f32,
        (base[1] + du_vec[1] + dv_vec[1]) as f32,
        (base[2] + du_vec[2] + dv_vec[2]) as f32,
    ];
    let p3 = [
        (base[0] + dv_vec[0]) as f32,
        (base[1] + dv_vec[1]) as f32,
        (base[2] + dv_vec[2]) as f32,
    ];

    let base_index = mesh.vertices.len() as u32;
    // Compute AO for each corner and set it into color alpha
    let c0_ao = compute_ao(base[0], base[1], base[2]);
    let c1_ao = compute_ao(base[0] + du_vec[0], base[1] + du_vec[1], base[2] + du_vec[2]);
    let c2_ao = compute_ao(base[0] + du_vec[0] + dv_vec[0], base[1] + du_vec[1] + dv_vec[1], base[2] + du_vec[2] + dv_vec[2]);
    let c3_ao = compute_ao(base[0] + dv_vec[0], base[1] + dv_vec[1], base[2] + dv_vec[2]);

    let mut color0 = color;
    let mut color1 = color;
    let mut color2 = color;
    let mut color3 = color;
    color0[3] = c0_ao;
    color1[3] = c1_ao;
    color2[3] = c2_ao;
    color3[3] = c3_ao;

    mesh.vertices.extend_from_slice(&[
        MeshVertex {
            position: p0,
            normal,
            color: color0,
            emissive,
        },
        MeshVertex {
            position: p1,
            normal,
            color: color1,
            emissive,
        },
        MeshVertex {
            position: p2,
            normal,
            color: color2,
            emissive,
        },
        MeshVertex {
            position: p3,
            normal,
            color: color3,
            emissive,
        },
    ]);

    // Two triangles with winding order dependent on face orientation
    // Positive faces (normal points toward -axis): use p0→p3→p2 and p0→p2→p1
    // Negative faces (normal points toward +axis): reverse winding to p0→p1→p2 and p0→p2→p3
    if axis == 1 {
        if positive {
            mesh.indices.extend_from_slice(&[
                base_index,
                base_index + 1,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 3,
            ]);
        } else {
            mesh.indices.extend_from_slice(&[
                base_index,
                base_index + 3,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 1,
            ]);
        }
    } else {
        if positive {
            mesh.indices.extend_from_slice(&[
                base_index,
                base_index + 3,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 1,
            ]);
        } else {
            mesh.indices.extend_from_slice(&[
                base_index,
                base_index + 1,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 3,
            ]);
        }
    }
}

// Emit a quad using AO values from a precomputed cache; if a corner isn't in the
// cache it's considered unoccluded (AO = 1.0).
fn emit_quad_with_ao_cache<F>(
    mesh: &mut ChunkMesh,
    palette: &Palette,
    quad: Quad,
    get_type: &F,
    ao_cache: &HashMap<(i32, i32, i32), f32>,
) where
    F: Fn(i32, i32, i32) -> Option<u8>,
{
    let Quad { axis, d, u_axis, v_axis, u0, v0, du, dv, face_type } = quad;
    if face_type == 0 {
        return;
    }

    let positive = face_type > 0;
    let mat = face_type.abs() as u8;

    let mut base = [0i32; 3];
    base[axis] = d;
    base[u_axis] = u0;
    base[v_axis] = v0;

    let mut du_vec = [0i32; 3];
    du_vec[u_axis] = du;
    let mut dv_vec = [0i32; 3];
    dv_vec[v_axis] = dv;

    let mut normal = [0.0f32; 3];
    normal[axis] = if positive { -1.0 } else { 1.0 };

    let material = palette.material(mat as u32);
    let color = material.albedo;
    let emissive = [
        material.emissive[0],
        material.emissive[1],
        material.emissive[2],
        material.emissive_intensity,
    ];

    let key0 = (base[0], base[1], base[2]);
    let key1 = (base[0] + du_vec[0], base[1] + du_vec[1], base[2] + du_vec[2]);
    let key2 = (
        base[0] + du_vec[0] + dv_vec[0],
        base[1] + du_vec[1] + dv_vec[1],
        base[2] + du_vec[2] + dv_vec[2],
    );
    let key3 = (base[0] + dv_vec[0], base[1] + dv_vec[1], base[2] + dv_vec[2]);

    let c0_ao = ao_cache.get(&key0).copied().unwrap_or(1.0);
    let c1_ao = ao_cache.get(&key1).copied().unwrap_or(1.0);
    let c2_ao = ao_cache.get(&key2).copied().unwrap_or(1.0);
    let c3_ao = ao_cache.get(&key3).copied().unwrap_or(1.0);

    let p0 = [base[0] as f32, base[1] as f32, base[2] as f32];
    let p1 = [
        (base[0] + du_vec[0]) as f32,
        (base[1] + du_vec[1]) as f32,
        (base[2] + du_vec[2]) as f32,
    ];
    let p2 = [
        (base[0] + du_vec[0] + dv_vec[0]) as f32,
        (base[1] + du_vec[1] + dv_vec[1]) as f32,
        (base[2] + du_vec[2] + dv_vec[2]) as f32,
    ];
    let p3 = [
        (base[0] + dv_vec[0]) as f32,
        (base[1] + dv_vec[1]) as f32,
        (base[2] + dv_vec[2]) as f32,
    ];

    let mut color0 = color;
    let mut color1 = color;
    let mut color2 = color;
    let mut color3 = color;
    color0[3] = c0_ao;
    color1[3] = c1_ao;
    color2[3] = c2_ao;
    color3[3] = c3_ao;

    let base_index = mesh.vertices.len() as u32;
    mesh.vertices.extend_from_slice(&[
        MeshVertex { position: p0, normal, color: color0, emissive },
        MeshVertex { position: p1, normal, color: color1, emissive },
        MeshVertex { position: p2, normal, color: color2, emissive },
        MeshVertex { position: p3, normal, color: color3, emissive },
    ]);

    // Indices like `emit_quad`
    if axis == 1 {
        if positive {
            mesh.indices.extend_from_slice(&[
                base_index,
                base_index + 1,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 3,
            ]);
        } else {
            mesh.indices.extend_from_slice(&[
                base_index,
                base_index + 3,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 1,
            ]);
        }
    } else {
        if positive {
            mesh.indices.extend_from_slice(&[
                base_index,
                base_index + 3,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 1,
            ]);
        } else {
            mesh.indices.extend_from_slice(&[
                base_index,
                base_index + 1,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 3,
            ]);
        }
    }
}
