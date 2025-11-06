//! Hierarchical Sparse Voxel Engine using Roaring Bitmaps
//!
//! "Chunks all the way" philosophy:
//! - Uniform Chunk structure at every level
//! - Each position in a chunk is a Voxel (enum: Solid or Chunk)
//! - Marginal bitmaps (px/py/pz) for fast rejection
//! - Roaring bitmap for exact presence
//! - Rank-based indexing into voxel array
//! - Infinite worlds via sparse HashMap

use croaring::Bitmap;
use std::collections::HashMap;

/// Voxel type identifier
pub type VoxelType = u8;

/// A voxel is either solid or contains a sub-chunk
#[derive(Clone, Debug)]
pub enum Voxel {
    /// A solid voxel with a type
    Solid(VoxelType),
    /// A chunk containing 16³ more voxels  
    Chunk(Box<Chunk>),
}

/// A hierarchical chunk ("chunks all the way")
///
/// Structure is uniform at all levels:
/// - px, py, pz: Marginal bitmaps for fast rejection
/// - presence: Exact bitmap of which positions have voxels
/// - voxels: Array indexed by rank(position)
///   - At leaf level: Voxel::Solid(type)
///   - At branch level: Voxel::Chunk(sub_chunk)
#[derive(Clone, Debug)]
pub struct Chunk {
    /// Marginal X bitmap: bit i set if any voxel exists at x=i
    pub px: u16,
    
    /// Marginal Y bitmap: bit i set if any voxel exists at y=i
    pub py: u16,
    
    /// Marginal Z bitmap: bit i set if any voxel exists at z=i
    pub pz: u16,
    
    /// Exact presence bitmap: bit at flat_index(x,y,z) set if voxel exists
    pub presence: Bitmap,
    
    /// Voxel array indexed by rank
    /// Can be Voxel::Solid (leaf) or Voxel::Chunk (branch)
    pub voxels: Vec<Voxel>,
}

impl Chunk {
    /// Create an empty chunk
    pub fn new() -> Self {
        Self {
            px: 0,
            py: 0,
            pz: 0,
            presence: Bitmap::new(),
            voxels: Vec::new(),
        }
    }
    
    /// Convert (x, y, z) coordinates to flat index
    /// x, y, z must be in range [0, 15]
    #[inline]
    pub fn flat_index(x: u8, y: u8, z: u8) -> u32 {
        debug_assert!(x < 16 && y < 16 && z < 16);
        (x as u32) + (y as u32) * 16 + (z as u32) * 256
    }
    
    /// Convert flat index back to (x, y, z)
    #[inline]
    pub fn unflatten(idx: u32) -> (u8, u8, u8) {
        let x = (idx % 16) as u8;
        let y = ((idx / 16) % 16) as u8;
        let z = (idx / 256) as u8;
        (x, y, z)
    }
    
    /// Check if a voxel exists at (x, y, z)
    pub fn contains(&self, x: u8, y: u8, z: u8) -> bool {
        // Fast marginal rejection
        if (self.px & (1 << x)) == 0 { return false; }
        if (self.py & (1 << y)) == 0 { return false; }
        if (self.pz & (1 << z)) == 0 { return false; }
        
        // Exact check
        let idx = Self::flat_index(x, y, z);
        self.presence.contains(idx)
    }
    
    /// Get the voxel at (x, y, z)
    pub fn get(&self, x: u8, y: u8, z: u8) -> Option<&Voxel> {
        if !self.contains(x, y, z) {
            return None;
        }
        
        let idx = Self::flat_index(x, y, z);
        let rank = self.presence.rank(idx) as usize;
        
        // rank-1 gives us the index in the voxels array
        self.voxels.get(rank - 1)
    }
    
    /// Get the voxel type at (x, y, z) if it's a Solid voxel
    pub fn get_type(&self, x: u8, y: u8, z: u8) -> Option<VoxelType> {
        match self.get(x, y, z)? {
            Voxel::Solid(t) => Some(*t),
            Voxel::Chunk(_) => None,
        }
    }
    
    /// Set a solid voxel at (x, y, z)
    pub fn set(&mut self, x: u8, y: u8, z: u8, voxel_type: VoxelType) {
        debug_assert!(x < 16 && y < 16 && z < 16);
        
        let idx = Self::flat_index(x, y, z);
        
        if self.presence.contains(idx) {
            // Update existing voxel
            let rank = self.presence.rank(idx) as usize;
            self.voxels[rank - 1] = Voxel::Solid(voxel_type);
        } else {
            // Insert new voxel
            let rank = self.presence.rank(idx) as usize;
            self.presence.add(idx);
            self.voxels.insert(rank, Voxel::Solid(voxel_type));
            
            // Update marginals
            self.px |= 1 << x;
            self.py |= 1 << y;
            self.pz |= 1 << z;
        }
    }
    
    /// Set a chunk at (x, y, z) - for hierarchical subdivision
    pub fn set_chunk(&mut self, x: u8, y: u8, z: u8, chunk: Chunk) {
        debug_assert!(x < 16 && y < 16 && z < 16);
        
        let idx = Self::flat_index(x, y, z);
        
        // For hierarchical chunks, inherit the sub-chunk's projection bits
        // This allows marginal culling to work at any level
        let sub_px = chunk.px;
        let sub_py = chunk.py;
        let sub_pz = chunk.pz;
        
        if self.presence.contains(idx) {
            // Update existing
            let rank = self.presence.rank(idx) as usize;
            self.voxels[rank - 1] = Voxel::Chunk(Box::new(chunk));
        } else {
            // Insert new
            let rank = self.presence.rank(idx) as usize;
            self.presence.add(idx);
            self.voxels.insert(rank, Voxel::Chunk(Box::new(chunk)));
            
            // Update marginals - set bit for this position
            self.px |= 1 << x;
            self.py |= 1 << y;
            self.pz |= 1 << z;
        }
        
        // Additionally, OR in the sub-chunk's projection bits
        // This propagates occupancy information up the hierarchy
        self.px |= sub_px;
        self.py |= sub_py;
        self.pz |= sub_pz;
    }
    
    /// Remove a voxel at (x, y, z)
    pub fn remove(&mut self, x: u8, y: u8, z: u8) {
        let idx = Self::flat_index(x, y, z);
        
        if !self.presence.contains(idx) {
            return;
        }
        
        let rank = self.presence.rank(idx) as usize;
        self.presence.remove(idx);
        self.voxels.remove(rank - 1);
        
        // Update marginals if needed - check if this was the last voxel on this axis
        if (0..16).all(|i| i == x || !self.contains(i, y, z)) {
            self.px &= !(1 << x);
        }
        if (0..16).all(|i| i == y || !self.contains(x, i, z)) {
            self.py &= !(1 << y);
        }
        if (0..16).all(|i| i == z || !self.contains(x, y, i)) {
            self.pz &= !(1 << z);
        }
    }
    
    /// Get the number of voxels in this chunk
    pub fn count(&self) -> u64 {
        self.presence.cardinality()
    }
    
    /// Check if this chunk is empty
    pub fn is_empty(&self) -> bool {
        self.presence.is_empty()
    }
    
    /// Iterator over all voxel positions
    pub fn positions(&self) -> impl Iterator<Item = (u8, u8, u8)> + '_ {
        self.presence.iter().map(Self::unflatten)
    }
    
    /// Iterator over all (position, voxel) pairs
    pub fn iter(&self) -> impl Iterator<Item = ((u8, u8, u8), &Voxel)> + '_ {
        self.presence.iter().enumerate().map(move |(i, idx)| {
            let pos = Self::unflatten(idx);
            let voxel = &self.voxels[i];
            (pos, voxel)
        })
    }
    
    /// Subdivide a solid voxel into a chunk
    /// Converts Voxel::Solid at (x,y,z) into Voxel::Chunk containing 16³ voxels of the same type
    pub fn subdivide(&mut self, x: u8, y: u8, z: u8) -> Result<(), &'static str> {
        let idx = Self::flat_index(x, y, z);
        
        if !self.presence.contains(idx) {
            return Err("No voxel at this position");
        }
        
        let rank = self.presence.rank(idx) as usize;
        let voxel = &self.voxels[rank - 1];
        
        // Can only subdivide solid voxels
        let voxel_type = match voxel {
            Voxel::Solid(t) => *t,
            Voxel::Chunk(_) => return Err("Already subdivided"),
        };
        
        // Create a new chunk filled with voxels of the same type
        let mut sub_chunk = Chunk::new();
        for sx in 0..16 {
            for sy in 0..16 {
                for sz in 0..16 {
                    sub_chunk.set(sx, sy, sz, voxel_type);
                }
            }
        }
        
        // Replace the solid voxel with the chunk
        // This also updates parent's projection bits to reflect sub-chunk contents
        self.remove(x, y, z);
        self.set_chunk(x, y, z, sub_chunk);
        
        Ok(())
    }
    
    /// Check if a chunk can be merged (all voxels are solid with the same type)
    pub fn can_merge(chunk: &Chunk) -> Option<VoxelType> {
        if chunk.is_empty() {
            return None;
        }
        
        let mut voxel_type = None;
        
        for (_pos, voxel) in chunk.iter() {
            match voxel {
                Voxel::Solid(t) => {
                    if let Some(expected) = voxel_type {
                        if *t != expected {
                            return None; // Different types
                        }
                    } else {
                        voxel_type = Some(*t);
                    }
                }
                Voxel::Chunk(_) => return None, // Contains sub-chunks
            }
        }
        
        voxel_type
    }
    
    /// Merge a sub-chunk back to a solid voxel if all voxels are uniform
    pub fn try_merge(&mut self, x: u8, y: u8, z: u8) -> Result<bool, &'static str> {
        let idx = Self::flat_index(x, y, z);
        
        if !self.presence.contains(idx) {
            return Err("No voxel at this position");
        }
        
        let rank = self.presence.rank(idx) as usize;
        let voxel = &self.voxels[rank - 1];
        
        // Can only merge chunks
        let sub_chunk = match voxel {
            Voxel::Chunk(chunk) => chunk,
            Voxel::Solid(_) => return Ok(false), // Already solid
        };
        
        // Get the sub-chunk's projection bits before merging
        let old_px = sub_chunk.px;
        let old_py = sub_chunk.py;
        let old_pz = sub_chunk.pz;
        
        // Check if the chunk can be merged
        if let Some(uniform_type) = Self::can_merge(sub_chunk) {
            // Replace chunk with solid voxel
            self.voxels[rank - 1] = Voxel::Solid(uniform_type);
            
            // Clear the sub-chunk's projection bits from parent
            // After merge, only the position bit should remain
            self.px &= !old_px | (1 << x);  // Clear old bits, keep position bit
            self.py &= !old_py | (1 << y);
            self.pz &= !old_pz | (1 << z);
            
            Ok(true)
        } else {
            Ok(false) // Cannot merge (not uniform)
        }
    }
    
    /// Get the depth of the hierarchy at a given position (0 = solid, 1+ = subdivided)
    pub fn depth_at(&self, x: u8, y: u8, z: u8) -> Option<usize> {
        match self.get(x, y, z)? {
            Voxel::Solid(_) => Some(0),
            Voxel::Chunk(chunk) => {
                // Find max depth in sub-chunk
                let mut max_depth = 0;
                for ((sx, sy, sz), _) in chunk.iter() {
                    if let Some(depth) = chunk.depth_at(sx, sy, sz) {
                        max_depth = max_depth.max(depth);
                    }
                }
                Some(1 + max_depth)
            }
        }
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

/// World coordinate in 3D space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldPos {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

impl WorldPos {
    pub fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }
    
    /// Convert world position to chunk coordinate and local position
    pub fn to_chunk_local(&self) -> (WorldPos, (u8, u8, u8)) {
        let chunk_x = self.x.div_euclid(16);
        let chunk_y = self.y.div_euclid(16);
        let chunk_z = self.z.div_euclid(16);
        
        let local_x = self.x.rem_euclid(16) as u8;
        let local_y = self.y.rem_euclid(16) as u8;
        let local_z = self.z.rem_euclid(16) as u8;
        
        (WorldPos::new(chunk_x, chunk_y, chunk_z), (local_x, local_y, local_z))
    }
}

/// The voxel world - infinite sparse storage
pub struct World {
    /// Top-level chunks (currently flat, can be extended to hierarchy)
    chunks: HashMap<(i64, i64, i64), Chunk>,
}

impl World {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }
    
    /// Get voxel type at world position (only works for Solid voxels)
    pub fn get(&self, pos: WorldPos) -> Option<VoxelType> {
        let (chunk_pos, local) = pos.to_chunk_local();
        let chunk = self.chunks.get(&(chunk_pos.x, chunk_pos.y, chunk_pos.z))?;
        chunk.get_type(local.0, local.1, local.2)
    }
    
    /// Set a solid voxel at world position
    pub fn set(&mut self, pos: WorldPos, voxel_type: VoxelType) {
        let (chunk_pos, local) = pos.to_chunk_local();
        let chunk = self.chunks
            .entry((chunk_pos.x, chunk_pos.y, chunk_pos.z))
            .or_insert_with(Chunk::new);
        chunk.set(local.0, local.1, local.2, voxel_type);
    }
    
    /// Remove a voxel at world position
    pub fn remove(&mut self, pos: WorldPos) {
        let (chunk_pos, local) = pos.to_chunk_local();
        if let Some(chunk) = self.chunks.get_mut(&(chunk_pos.x, chunk_pos.y, chunk_pos.z)) {
            chunk.remove(local.0, local.1, local.2);
            if chunk.is_empty() {
                self.chunks.remove(&(chunk_pos.x, chunk_pos.y, chunk_pos.z));
            }
        }
    }
    
    /// Get all chunks
    pub fn chunks(&self) -> impl Iterator<Item = ((i64, i64, i64), &Chunk)> {
        self.chunks.iter().map(|(k, v)| (*k, v))
    }
    
    /// Get mutable reference to a chunk
    pub fn get_chunk_mut(&mut self, chunk_pos: (i64, i64, i64)) -> &mut Chunk {
        self.chunks.entry(chunk_pos).or_insert_with(Chunk::new)
    }
    
    /// Get reference to a chunk
    pub fn get_chunk(&self, chunk_pos: (i64, i64, i64)) -> Option<&Chunk> {
        self.chunks.get(&chunk_pos)
    }
    
    /// Subdivide a voxel at world position
    pub fn subdivide_at(&mut self, pos: WorldPos) -> Result<(), &'static str> {
        let (chunk_pos, local) = pos.to_chunk_local();
        let chunk = self.chunks
            .get_mut(&(chunk_pos.x, chunk_pos.y, chunk_pos.z))
            .ok_or("No chunk at this position")?;
        chunk.subdivide(local.0, local.1, local.2)
    }
    
    /// Try to merge a subdivided voxel back to solid
    pub fn merge_at(&mut self, pos: WorldPos) -> Result<bool, &'static str> {
        let (chunk_pos, local) = pos.to_chunk_local();
        let chunk = self.chunks
            .get_mut(&(chunk_pos.x, chunk_pos.y, chunk_pos.z))
            .ok_or("No chunk at this position")?;
        chunk.try_merge(local.0, local.1, local.2)
    }
    
    /// Get the hierarchy depth at a world position
    pub fn depth_at(&self, pos: WorldPos) -> Option<usize> {
        let (chunk_pos, local) = pos.to_chunk_local();
        let chunk = self.chunks.get(&(chunk_pos.x, chunk_pos.y, chunk_pos.z))?;
        chunk.depth_at(local.0, local.1, local.2)
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flat_index() {
        assert_eq!(Chunk::flat_index(0, 0, 0), 0);
        assert_eq!(Chunk::flat_index(1, 0, 0), 1);
        assert_eq!(Chunk::flat_index(0, 1, 0), 16);
        assert_eq!(Chunk::flat_index(0, 0, 1), 256);
        assert_eq!(Chunk::flat_index(15, 15, 15), 4095);
    }
    
    #[test]
    fn test_unflatten() {
        assert_eq!(Chunk::unflatten(0), (0, 0, 0));
        assert_eq!(Chunk::unflatten(1), (1, 0, 0));
        assert_eq!(Chunk::unflatten(16), (0, 1, 0));
        assert_eq!(Chunk::unflatten(256), (0, 0, 1));
        assert_eq!(Chunk::unflatten(4095), (15, 15, 15));
    }
    
    #[test]
    fn test_chunk_set_get() {
        let mut chunk = Chunk::new();
        
        chunk.set(5, 7, 3, 42);
        assert_eq!(chunk.get_type(5, 7, 3), Some(42));
        assert_eq!(chunk.get_type(5, 7, 4), None);
        
        chunk.set(5, 7, 3, 100);
        assert_eq!(chunk.get_type(5, 7, 3), Some(100));
    }
    
    #[test]
    fn test_voxel_enum() {
        let mut chunk = Chunk::new();
        
        // Set a solid voxel
        chunk.set(0, 0, 0, 1);
        assert!(matches!(chunk.get(0, 0, 0), Some(Voxel::Solid(1))));
        
        // Set a sub-chunk
        let sub_chunk = Chunk::new();
        chunk.set_chunk(1, 1, 1, sub_chunk);
        assert!(matches!(chunk.get(1, 1, 1), Some(Voxel::Chunk(_))));
    }
    
    #[test]
    fn test_world() {
        let mut world = World::new();
        
        world.set(WorldPos::new(0, 0, 0), 1);
        world.set(WorldPos::new(100, 200, 300), 2);
        
        assert_eq!(world.get(WorldPos::new(0, 0, 0)), Some(1));
        assert_eq!(world.get(WorldPos::new(100, 200, 300)), Some(2));
        assert_eq!(world.get(WorldPos::new(1, 1, 1)), None);
    }
}
