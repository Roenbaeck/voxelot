//! Greedy meshing for bottom-level chunks (16x16x16)

use crate::lib_hierarchical::{Chunk, Voxel};
use crate::palette::Palette;

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
    fn test_generate_chunk_mesh_basic() {
        // Construct a small palette with indices 0..=1
        let palette = Palette::from_string("0 255 255 255 255\n1 255 255 255 255\n").unwrap();

        // Create a chunk with a single voxel
        let mut chunk = Chunk::new();
        chunk.set(8, 8, 8, 1);

        // Generate mesh
        let mesh = generate_chunk_mesh(&chunk, &palette, None);

        // Should have vertices and indices
        assert!(mesh.vertices.len() > 0);
        assert!(mesh.indices.len() > 0);
        // All vertex alphas should be 1.0 (no AO)
        for v in &mesh.vertices {
            let a = v.color[3];
            assert!(a >= 0.9999 && a <= 1.0);
        }
    }

    #[test]
    fn test_generate_chunk_mesh_isolated_no_ao() {
        let palette = Palette::from_string("0 255 255 255 255\n1 255 255 255 255\n").unwrap();
        let mut chunk = Chunk::new();
        // Single isolated voxel
        chunk.set(8, 8, 8, 1);
        let mesh = generate_chunk_mesh(&chunk, &palette, None);
        // All vertex alphas should be 1.0
        assert!(mesh.vertices.len() > 0);
        for v in &mesh.vertices {
            let a = v.color[3];
            assert!(
                a >= 0.9999 && a <= 1.0,
                "Expected AO alpha near 1.0 for isolated voxel, got {}",
                a
            );
        }
    }

    #[test]
    fn test_generate_chunk_mesh_ground_isolated_no_ao() {
        let palette = Palette::from_string("0 255 255 255 255\n1 255 255 255 255\n").unwrap();
        let mut chunk = Chunk::new();
        // Single isolated voxel at the ground (y = 0)
        chunk.set(8, 0, 8, 1);
        let mesh = generate_chunk_mesh(&chunk, &palette, None);
        // All vertex alphas should be near 1.0
        assert!(mesh.vertices.len() > 0);
        for v in &mesh.vertices {
            let a = v.color[3];
            assert!(
                a >= 0.9999 && a <= 1.0,
                "Expected AO alpha near 1.0 for ground isolated voxel, got {}",
                a
            );
        }
    }

    #[test]
    fn test_generate_chunk_mesh_chunk_boundary_no_ao_from_neighbors() {
        let palette = Palette::from_string("0 255 255 255 255\n1 255 255 255 255\n").unwrap();

        // Main chunk has a single voxel at the corner touching the +X +Y +Z boundary
        let mut main = Chunk::new();
        main.set(15, 15, 15, 1);

        // Neighbor at +X +Y +Z has a voxel diagonally at (0,0,0)
        let mut neigh = Chunk::new();
        neigh.set(0, 0, 0, 1);

        let mut neighbors = std::collections::HashMap::new();
        neighbors.insert((1i8, 1i8, 1i8), neigh);

        // Generate mesh for main with neighbors; AO should NOT count neighbor voxels
        let mesh = generate_chunk_mesh(&main, &palette, Some(&neighbors));

        // All alphas should be near 1.0 — the neighbor should not cause darkening
        assert!(mesh.vertices.len() > 0);
        for v in &mesh.vertices {
            let a = v.color[3];
            assert!(
                a >= 0.9999 && a <= 1.0,
                "Expected AO alpha near 1.0 for seam corner, got {}",
                a
            );
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
pub fn generate_chunk_mesh(
    chunk: &Chunk,
    palette: &Palette,
    neighbors: Option<&std::collections::HashMap<(i8, i8, i8), Chunk>>,
) -> ChunkMesh {
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
        // Inside chunk
        if x >= 0 && x < 16 && y >= 0 && y < 16 && z >= 0 && z < 16 {
            return chunk.get_type(x as u8, y as u8, z as u8);
        }
        // Out-of-chunk - check neighbor snapshot if present
        if let Some(neigh) = &neighbors {
            let nx = if x < 0 {
                -1
            } else if x >= 16 {
                1
            } else {
                0
            };
            let ny = if y < 0 {
                -1
            } else if y >= 16 {
                1
            } else {
                0
            };
            let nz = if z < 0 {
                -1
            } else if z >= 16 {
                1
            } else {
                0
            };
            let entry = (nx as i8, ny as i8, nz as i8);
            if let Some(ne) = neigh.get(&entry) {
                let lx = if nx < 0 {
                    (x + 16) as u8
                } else if nx > 0 {
                    (x - 16) as u8
                } else {
                    x as u8
                };
                let ly = if ny < 0 {
                    (y + 16) as u8
                } else if ny > 0 {
                    (y - 16) as u8
                } else {
                    y as u8
                };
                let lz = if nz < 0 {
                    (z + 16) as u8
                } else if nz > 0 {
                    (z - 16) as u8
                } else {
                    z as u8
                };
                return ne.get_type(lx, ly, lz);
            }
        }
        None
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

    // Now emit quads
    for quad in &quads {
        emit_quad(&mut mesh, palette, *quad);
    }

    mesh
}

// `emit_quad` is intentionally removed - the AO-aware and cached emission
// is handled by `emit_quad_with_ao_cache`. Keeping this function produced
// a dead-code warning; it was also partially implemented and not used.

// Emit a quad with AO always set to 1.0 (no occlusion).
fn emit_quad(mesh: &mut ChunkMesh, palette: &Palette, quad: Quad) {
    let Quad {
        axis,
        d,
        u_axis,
        v_axis,
        u0,
        v0,
        du,
        dv,
        face_type,
    } = quad;
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

    // AO is always 1.0 (no occlusion)
    let ao = 1.0;

    let mut color0 = color;
    let mut color1 = color;
    let mut color2 = color;
    let mut color3 = color;
    color0[3] = ao;
    color1[3] = ao;
    color2[3] = ao;
    color3[3] = ao;

    let base_index = mesh.vertices.len() as u32;
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
