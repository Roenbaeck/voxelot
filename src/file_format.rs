//! Compact hierarchical voxel file format (vhc)
//!
//! Format mirrors the internal Chunk structure exactly:
//! - Header: depth (u8)
//! - Root chunk recursively encoded
//!
//! Chunk encoding:
//! - Position count (u16, max 4096 for 16³ chunk)
//! - For each occupied position:
//!   - Position encoded as u16 (z * 256 + y * 16 + x)
//!   - If type == 0: sub-chunk follows (recursively encoded)
//!   - Otherwise: solid voxel type (1-254)

use crate::lib_hierarchical::{Chunk, Voxel};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;
use zstd::stream::read::Decoder as ZstdDecoder;
use zstd::stream::write::Encoder as ZstdEncoder;

/// Save world to compact format (vhc)
pub fn save_world(
    world: &crate::lib_hierarchical::World,
    writer: &mut impl Write,
) -> io::Result<()> {
    // Write depth
    writer.write_all(&[world.hierarchy_depth()])?;

    // Write root chunk
    save_chunk(world.root(), writer)?;

    Ok(())
}

/// Save world to a file path. The function writes compressed `.vhc` using zstd.
pub fn save_world_file(
    world: &crate::lib_hierarchical::World,
    path: &Path,
    _compress: bool,
) -> io::Result<()> {
    // Use zstd compression; caller's compress flag is ignored (always compress)
    let mut payload: Vec<u8> = Vec::new();
    save_world(world, &mut payload)?;

    let file = File::create(path)?;
    let mut encoder =
        ZstdEncoder::new(file, 0).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    encoder.write_all(&payload)?;
    encoder
        .finish()
        .map(|_| ())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

/// Save a chunk recursively
fn save_chunk(chunk: &Chunk, writer: &mut impl Write) -> io::Result<()> {
    // Get all occupied positions
    let positions: Vec<(u8, u8, u8)> = chunk.positions().collect();

    // Write count (u16)
    let count = positions.len() as u16;
    writer.write_all(&count.to_le_bytes())?;

    // Write each position and its data
    for (x, y, z) in positions {
        // Encode position as u16 (z * 256 + y * 16 + x)
        let pos_encoded = ((z as u16) << 8) | ((y as u16) << 4) | (x as u16);
        writer.write_all(&pos_encoded.to_le_bytes())?;

        match chunk.get(x, y, z) {
            Some(Voxel::Solid(vtype)) => {
                writer.write_all(&[*vtype])?;
            }
            Some(Voxel::Chunk(sub_chunk)) => {
                writer.write_all(&[0])?; // 0 means sub-chunk follows
                save_chunk(sub_chunk, writer)?;
            }
            None => {
                // Shouldn't happen since we got this from positions()
                writer.write_all(&[255])?;
            }
        }
    }

    Ok(())
}

/// Load world from compact format (vhc)
pub fn load_world(reader: &mut impl Read) -> io::Result<crate::lib_hierarchical::World> {
    // Read depth
    let mut depth_byte = [0u8; 1];
    reader.read_exact(&mut depth_byte)?;
    let depth = depth_byte[0];

    // Create empty world
    let mut world = crate::lib_hierarchical::World::new(depth);

    // Load root chunk
    load_chunk(world.root_mut(), reader)?;

    Ok(world)
}

/// Load a world from a file.
pub fn load_world_file(path: &Path) -> io::Result<crate::lib_hierarchical::World> {
    // We only support zstd-compressed `.vhc` files; legacy raw `.oct` files have been removed.
    let file = File::open(path)?;
    let mut decoder =
        ZstdDecoder::new(file).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    load_world(&mut decoder)
}

/// Load a chunk recursively
fn load_chunk(chunk: &mut Chunk, reader: &mut impl Read) -> io::Result<()> {
    // Read count of occupied positions (u16)
    let mut count_bytes = [0u8; 2];
    reader.read_exact(&mut count_bytes)?;
    let count = u16::from_le_bytes(count_bytes);

    // Read each occupied position into a temporary list, then commit in sorted order.
    // This avoids O(n^2) vector insert costs in `Chunk::set` when positions are read out of rank order.
    let mut entries: Vec<(u16, Voxel)> = Vec::with_capacity(count as usize);

    for _ in 0..count {
        // Read position (u16: z * 256 + y * 16 + x)
        let mut pos_bytes = [0u8; 2];
        reader.read_exact(&mut pos_bytes)?;
        let pos_encoded = u16::from_le_bytes(pos_bytes);

        // Read voxel type
        let mut type_byte = [0u8; 1];
        reader.read_exact(&mut type_byte)?;
        let vtype = type_byte[0];

        if vtype == 0 {
            // Sub-chunk follows; load recursively and store
            let mut sub_chunk = Chunk::new();
            load_chunk(&mut sub_chunk, reader)?;
            entries.push((pos_encoded, Voxel::Chunk(Arc::new(sub_chunk))));
        } else if vtype != 255 {
            // Solid voxel
            entries.push((pos_encoded, Voxel::Solid(vtype)));
        }
        // 255 means empty (shouldn't happen but handle gracefully)
    }

    // Sort entries by encoded position (flat index) so we can push into `voxels` in rank order
    entries.sort_by_key(|(pos, _)| *pos);

    // Reserve capacity and then commit entries into presence/voxels in order
    chunk.voxels.reserve(entries.len());
    let mut solid_count: u32 = 0;
    let mut xmin: u8 = 16;
    let mut ymin: u8 = 16;
    let mut zmin: u8 = 16;
    let mut xmax: u8 = 0;
    let mut ymax: u8 = 0;
    let mut zmax: u8 = 0;
    let mut bbox_found = false;
    for (pos_encoded, voxel) in entries {
        let x = (pos_encoded & 0xF) as u8;
        let y = ((pos_encoded >> 4) & 0xF) as u8;
        let z = ((pos_encoded >> 8) & 0xF) as u8;

        bbox_found = true;
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        zmin = zmin.min(z);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
        zmax = zmax.max(z);

        match &voxel {
            Voxel::Solid(_) => {
                solid_count = solid_count.saturating_add(1);
            }
            Voxel::Chunk(sub_chunk) => {
                solid_count = solid_count.saturating_add(sub_chunk.voxel_count);
            }
        }

        // Append presence and voxel in rank order
        chunk.presence.add(pos_encoded as u32);
        chunk.voxels.push(voxel);

        // Update marginals for this slot
        chunk.px |= 1 << x;
        chunk.py |= 1 << y;
        chunk.pz |= 1 << z;

        // If this was a sub-chunk, also OR-in its projection bits for quick coarse culling
        if let Voxel::Chunk(ref sub_chunk) = chunk.voxels.last().unwrap() {
            chunk.px |= sub_chunk.px;
            chunk.py |= sub_chunk.py;
            chunk.pz |= sub_chunk.pz;
        }
    }

    // Cache aggregate counts and bbox for faster LOD init
    chunk.voxel_count = solid_count;
    chunk.solid_ratio = solid_count as f32 / (16.0 * 16.0 * 16.0);
    if bbox_found {
        chunk.bounding_box = Some([xmin, ymin, zmin, xmax, ymax, zmax]);
    } else {
        chunk.bounding_box = None;
    }

    Ok(())
}
