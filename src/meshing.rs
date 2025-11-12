//! Greedy meshing for bottom-level chunks (16x16x16)

use crate::lib_hierarchical::Chunk;
use crate::palette::Palette;

macro_rules! mesh_debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "mesh-debug") {
            eprintln!($($arg)*);
        }
    };
}

#[derive(Copy, Clone, Debug)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

/// Mesh output for a chunk
#[derive(Clone, Debug, Default)]
pub struct ChunkMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

/// Generate a greedy mesh for a 16x16x16 chunk.
/// Merges coplanar faces with identical voxel types into larger quads.
pub fn generate_chunk_mesh(chunk: &Chunk, palette: &Palette) -> ChunkMesh {
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

    // Helper to get voxel type at (x,y,z)
    let get_type = |x: i32, y: i32, z: i32| -> Option<u8> {
        if x < 0 || x >= 16 || y < 0 || y >= 16 || z < 0 || z >= 16 {
            return None;
        }
        chunk.get_type(x as u8, y as u8, z as u8)
    };

    // For each axis, create faces between differing neighbor voxels
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

                    // Emit this rectangle
                    emit_quad(
                        &mut mesh,
                        palette,
                        axis,
                        d,
                        u_axis,
                        v_axis,
                        u_start,
                        v_start,
                        width,
                        height,
                        t,
                    );
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

    mesh
}

fn emit_quad(
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
) {
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

    // Color from palette (per-voxel RGBA)
    let color = palette.color(mat as u32);

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
    mesh.vertices.extend_from_slice(&[
        MeshVertex {
            position: p0,
            normal,
            color,
        },
        MeshVertex {
            position: p1,
            normal,
            color,
        },
        MeshVertex {
            position: p2,
            normal,
            color,
        },
        MeshVertex {
            position: p3,
            normal,
            color,
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
