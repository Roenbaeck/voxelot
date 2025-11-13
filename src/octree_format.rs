//! Compact hierarchical voxel file format
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
use std::io::{self, Read, Write};

/// Save world to compact format
pub fn save_world(world: &crate::lib_hierarchical::World, writer: &mut impl Write) -> io::Result<()> {
    // Write depth
    writer.write_all(&[world.hierarchy_depth()])?;
    
    // Write root chunk
    save_chunk(world.root(), writer)?;
    
    Ok(())
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
        let pos_encoded = (z as u16) * 256 + (y as u16) * 16 + (x as u16);
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

/// Load world from compact format
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

/// Load a chunk recursively
fn load_chunk(chunk: &mut Chunk, reader: &mut impl Read) -> io::Result<()> {
    // Read count of occupied positions (u16)
    let mut count_bytes = [0u8; 2];
    reader.read_exact(&mut count_bytes)?;
    let count = u16::from_le_bytes(count_bytes);
    
    // Read each occupied position
    for _ in 0..count {
        // Read position (u16: z * 256 + y * 16 + x)
        let mut pos_bytes = [0u8; 2];
        reader.read_exact(&mut pos_bytes)?;
        let pos_encoded = u16::from_le_bytes(pos_bytes);
        
        let x = (pos_encoded % 16) as u8;
        let y = ((pos_encoded / 16) % 16) as u8;
        let z = (pos_encoded / 256) as u8;
        
        // Read voxel type
        let mut type_byte = [0u8; 1];
        reader.read_exact(&mut type_byte)?;
        let vtype = type_byte[0];
        
        if vtype == 0 {
            // Sub-chunk follows
            let mut sub_chunk = Chunk::new();
            load_chunk(&mut sub_chunk, reader)?;
            chunk.set_chunk(x, y, z, sub_chunk);
        } else if vtype != 255 {
            // Solid voxel
            chunk.set(x, y, z, vtype);
        }
        // 255 means empty (shouldn't happen but handle gracefully)
    }
    
    Ok(())
}
