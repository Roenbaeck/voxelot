//! Hierarchical Sparse Voxel Engine using Roaring Bitmaps
//!
//! "Chunks all the way" philosophy:
//! - Uniform Chunk structure at every level (including the World root!)
//! - Each position in a chunk is a Voxel (enum: Solid or Chunk)
//! - Marginal bitmaps (px/py/pz) for fast rejection
//! - Roaring bitmap for exact presence
//! - Rank-based indexing into voxel array
//! - Bounded but huge worlds: 16^n units (e.g., 16^4 = 65,536³)

use croaring::Bitmap;

use crate::palette::Palette;

/// Voxel type identifier
pub type VoxelType = u8;

/// Convert a VoxelType to RGBA color (matches shader's get_voxel_color)
pub fn voxel_type_to_rgba(voxel_type: VoxelType) -> [u8; 4] {
    let (r, g, b) = match voxel_type {
        1 => (0.1, 0.9, 0.3),    // Neon grass highlights
        2 => (1.0, 0.35, 0.35),  // Sunlit red concrete
        3 => (0.35, 0.5, 1.0),   // Electric blue panels
        4 => (0.95, 0.9, 0.35),  // Warm accent lighting
        5 => (0.95, 0.4, 1.0),   // Vibrant magenta glass
        6 => (0.3, 0.95, 1.0),   // Cyan signage glow
        7 => (0.85, 0.85, 0.85), // Bright concrete walls
        _ => (1.0, 1.0, 1.0),    // White default
    };

    [
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
        255, // Fully opaque
    ]
}

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
/// - LOD metadata: voxel_count and average_color for distance rendering
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

    /// LOD metadata: Number of solid voxels in this chunk
    pub voxel_count: u32,

    /// LOD metadata: Average RGBA color for distance rendering
    /// Alpha represents occupancy (0 = empty, 255 = fully dense)
    pub average_color: [u8; 4],

    /// Sum of emissive RGB (intensity-weighted) for voxels in this chunk
    pub emissive_sum: [f32; 3],

    /// Total emissive intensity across voxels in this chunk
    pub emissive_power: f32,

    /// Count of voxels contributing emissive light
    pub emissive_voxels: u32,

    /// Ratio of solid voxels to total slots (0.0..=1.0)
    pub solid_ratio: f32,
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
            voxel_count: 0,
            average_color: [0, 0, 0, 0], // Empty chunk = transparent
            emissive_sum: [0.0, 0.0, 0.0],
            emissive_power: 0.0,
            emissive_voxels: 0,
            solid_ratio: 0.0,
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
        if (self.px & (1 << x)) == 0 {
            return false;
        }
        if (self.py & (1 << y)) == 0 {
            return false;
        }
        if (self.pz & (1 << z)) == 0 {
            return false;
        }

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

    /// Update LOD metadata using palette material properties for this chunk.
    /// Should be called after modifying chunk contents.
    pub fn update_lod_metadata(&mut self, palette: &Palette) {
        const TOTAL_SLOTS: f32 = (16 * 16 * 16) as f32; // 4096

        let mut albedo_sum = [0.0f32; 4];
        let mut emissive_sum = [0.0f32; 3];
        let mut emissive_power = 0.0f32;
        let mut emissive_voxels = 0u32;
        let mut solid_count = 0u32;

        // Use palette color cache directly to avoid allocating `Material` per voxel.
        let color_cache = palette.colors();
        for voxel in &self.voxels {
            if let Voxel::Solid(voxel_type) = voxel {
                let idx = *voxel_type as usize;
                let material_color = if idx < color_cache.len() {
                    color_cache[idx]
                } else {
                    [1.0, 1.0, 1.0, 1.0]
                };
                albedo_sum[0] += material_color[0];
                albedo_sum[1] += material_color[1];
                albedo_sum[2] += material_color[2];
                albedo_sum[3] += material_color[3];
                solid_count += 1;

                let (emissive_color, emissive_strength) = palette.emissive(*voxel_type as u32);
                if emissive_strength > 0.0 {
                    emissive_sum[0] += emissive_color[0] * emissive_strength;
                    emissive_sum[1] += emissive_color[1] * emissive_strength;
                    emissive_sum[2] += emissive_color[2] * emissive_strength;
                    emissive_power += emissive_strength;
                    emissive_voxels += 1;
                }
            }
        }

        self.voxel_count = solid_count;
        self.solid_ratio = solid_count as f32 / TOTAL_SLOTS;

        if solid_count > 0 {
            let occupancy_scale = 255.0 / TOTAL_SLOTS;
            self.average_color = [
                (albedo_sum[0] * occupancy_scale).clamp(0.0, 255.0) as u8,
                (albedo_sum[1] * occupancy_scale).clamp(0.0, 255.0) as u8,
                (albedo_sum[2] * occupancy_scale).clamp(0.0, 255.0) as u8,
                (albedo_sum[3] * occupancy_scale).clamp(0.0, 255.0) as u8,
            ];
        } else {
            self.average_color = [0, 0, 0, 0];
        }

        self.emissive_sum = emissive_sum;
        self.emissive_power = emissive_power;
        self.emissive_voxels = emissive_voxels;
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
            self.px &= !old_px | (1 << x); // Clear old bits, keep position bit
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
}

/// The voxel world - "chunks all the way" means the World IS a Chunk!
///
/// The hierarchy depth determines world size: 16^depth units per side
/// - depth 1: 16³ = 4,096 voxels
/// - depth 2: 256³ = 16,777,216 voxels  
/// - depth 3: 4,096³ = 68,719,476,736 voxels
/// - depth 4: 65,536³ = 281,474,976,710,656 voxels
pub struct World {
    /// The root chunk - everything is a chunk!
    root: Chunk,

    /// Hierarchy depth (1 = single chunk, 2+ = nested)
    hierarchy_depth: u8,

    /// Base chunk size (always 16)
    #[allow(dead_code)]
    chunk_size: u32,
}

impl World {
    /// Create a new world with the specified hierarchy depth
    ///
    /// World size will be 16^depth units per side:
    /// - depth 1: 16 units (single chunk)
    /// - depth 2: 256 units
    /// - depth 3: 4,096 units
    /// - depth 4: 65,536 units (recommended for large worlds)
    pub fn new(hierarchy_depth: u8) -> Self {
        assert!(hierarchy_depth > 0, "Hierarchy depth must be at least 1");
        let world_size = 16u64.pow(hierarchy_depth as u32);
        println!(
            "Creating world: {} units per side ({} levels deep)",
            world_size, hierarchy_depth
        );

        Self {
            root: Chunk::new(),
            hierarchy_depth,
            chunk_size: 16,
        }
    }

    /// Get a reference to the 16x16x16 leaf chunk located at the given world origin (must be aligned to 16).
    /// Returns None if that position isn't a subdivided chunk.
    pub fn get_leaf_chunk_at_origin(&self, origin: WorldPos) -> Option<&Chunk> {
        // Ensure alignment (optional safety)
        if (origin.x & 15) != 0 || (origin.y & 15) != 0 || (origin.z & 15) != 0 {
            return None;
        }

        if self.hierarchy_depth == 1 {
            // Special case: root is the leaf chunk
            return Some(&self.root);
        }

        let path = self.position_to_path(origin).ok()?;
        // Navigate to the parent of the leaf chunk level (depth-2)
        let parent = self.navigate_to(&path, self.hierarchy_depth as usize - 2)?;
        // The leaf chunk is at position path[depth-2] in that parent
        let &(x, y, z) = &path[self.hierarchy_depth as usize - 2];
        match parent.get(x, y, z)? {
            Voxel::Chunk(c) => Some(c),
            _ => None,
        }
    }

    /// Subdivide a 16×16×16 region into a chunk structure
    /// This collects all existing voxels in the region and organizes them into a chunk
    /// If the parent position doesn't exist yet, it will be created
    pub fn subdivide_region(&mut self, origin: WorldPos) -> Result<(), &'static str> {
        // Ensure alignment
        let aligned_x = origin.x & !15;
        let aligned_y = origin.y & !15;
        let aligned_z = origin.z & !15;

        // Collect all voxels in this 16×16×16 region
        let mut voxel_data = Vec::new();
        for dx in 0..16 {
            for dy in 0..16 {
                for dz in 0..16 {
                    let pos = WorldPos::new(aligned_x + dx, aligned_y + dy, aligned_z + dz);
                    if let Some(vtype) = self.get(pos) {
                        voxel_data.push((dx as u8, dy as u8, dz as u8, vtype));
                    }
                }
            }
        }

        if voxel_data.is_empty() {
            return Err("No voxels in region");
        }

        // Navigate to parent that should contain this region
        let path = self.position_to_path(WorldPos::new(aligned_x, aligned_y, aligned_z))?;

        // Remove all individual voxels in this region
        for dx in 0..16 {
            for dy in 0..16 {
                for dz in 0..16 {
                    let pos = WorldPos::new(aligned_x + dx, aligned_y + dy, aligned_z + dz);
                    let _ = self.remove(pos); // Ignore errors
                }
            }
        }

        // Create a new chunk with the collected voxels
        let mut chunk = Chunk::new();
        for (x, y, z, vtype) in voxel_data {
            chunk.set(x, y, z, vtype);
        }

        // Set the chunk at the parent position
        let parent = self.navigate_to_mut(&path, self.hierarchy_depth as usize - 1);
        let &(x, y, z) = path.last().ok_or("Invalid path")?;
        parent.set_chunk(x, y, z, chunk);

        Ok(())
    }

    /// Get the world size (units per side)
    pub fn world_size(&self) -> u64 {
        16u64.pow(self.hierarchy_depth as u32)
    }

    /// Get the hierarchy depth
    pub fn hierarchy_depth(&self) -> u8 {
        self.hierarchy_depth
    }

    /// Count all voxels in the world
    pub fn count(&self) -> u64 {
        self.root.count()
    }

    /// Convert world position to a path through the hierarchy
    /// Returns a Vec of (x, y, z) tuples, one for each level from root to leaf
    fn position_to_path(&self, pos: WorldPos) -> Result<Vec<(u8, u8, u8)>, &'static str> {
        let world_size = self.world_size() as i64;

        // Check bounds
        if pos.x < 0
            || pos.y < 0
            || pos.z < 0
            || pos.x >= world_size
            || pos.y >= world_size
            || pos.z >= world_size
        {
            return Err("Position out of world bounds");
        }

        let mut path = Vec::with_capacity(self.hierarchy_depth as usize);
        let mut x = pos.x;
        let mut y = pos.y;
        let mut z = pos.z;

        // Walk down the hierarchy from root to leaf
        // At each level, extract the 4-bit index for that level
        for level in (0..self.hierarchy_depth).rev() {
            let divisor = 16i64.pow(level as u32);
            let local_x = (x / divisor) as u8 & 0xF;
            let local_y = (y / divisor) as u8 & 0xF;
            let local_z = (z / divisor) as u8 & 0xF;
            path.push((local_x, local_y, local_z));

            x %= divisor;
            y %= divisor;
            z %= divisor;
        }

        Ok(path)
    }

    /// Navigate to a chunk at the given path depth (0 = root, depth-1 = leaf parent)
    fn navigate_to<'a>(&'a self, path: &[(u8, u8, u8)], depth: usize) -> Option<&'a Chunk> {
        let mut current = &self.root;

        for &(x, y, z) in &path[..depth] {
            match current.get(x, y, z)? {
                Voxel::Chunk(chunk) => current = chunk,
                Voxel::Solid(_) => return None, // Hit a solid before reaching target depth
            }
        }

        Some(current)
    }

    /// Navigate to a mutable chunk at the given path depth, creating sub-chunks as needed
    fn navigate_to_mut<'a>(&'a mut self, path: &[(u8, u8, u8)], depth: usize) -> &'a mut Chunk {
        let mut current = &mut self.root;

        for &(x, y, z) in &path[..depth] {
            let idx = Chunk::flat_index(x, y, z);

            // Check if voxel exists and what type it is
            let needs_chunk = if current.presence.contains(idx) {
                let rank = current.presence.rank(idx) as usize;
                !matches!(current.voxels[rank - 1], Voxel::Chunk(_))
            } else {
                true
            };

            // Create or ensure it's a chunk
            if needs_chunk {
                current.set_chunk(x, y, z, Chunk::new());
            }

            // Navigate into the chunk
            let rank = current.presence.rank(idx) as usize;
            match &mut current.voxels[rank - 1] {
                Voxel::Chunk(chunk) => current = chunk,
                _ => unreachable!(),
            }
        }

        current
    }

    /// Get voxel type at world position (only works for Solid voxels)
    pub fn get(&self, pos: WorldPos) -> Option<VoxelType> {
        let path = self.position_to_path(pos).ok()?;

        // Navigate to the parent chunk
        let parent = self.navigate_to(&path, self.hierarchy_depth as usize - 1)?;

        // Get the leaf position
        let &(x, y, z) = path.last()?;
        parent.get_type(x, y, z)
    }

    /// Set a solid voxel at world position
    pub fn set(&mut self, pos: WorldPos, voxel_type: VoxelType) {
        let path = match self.position_to_path(pos) {
            Ok(p) => p,
            Err(_) => return, // Out of bounds, silently ignore
        };

        let depth = self.hierarchy_depth as usize;

        if depth == 1 {
            // Special case: single-level world, root IS the leaf chunk
            let &(x, y, z) = path.last().unwrap();
            self.root.set(x, y, z, voxel_type);
            return;
        }

        // Navigate to the "grandparent" level (one above the leaf chunk level)
        let grandparent = self.navigate_to_mut(&path, depth - 2);

        // Ensure the leaf chunk exists at path[depth-2]
        let &(lx, ly, lz) = &path[depth - 2];
        let idx = Chunk::flat_index(lx, ly, lz);

        // Check if we need to create or replace with a chunk
        let needs_chunk = if grandparent.presence.contains(idx) {
            let rank = grandparent.presence.rank(idx) as usize;
            !matches!(&grandparent.voxels[rank - 1], Voxel::Chunk(_))
        } else {
            true
        };

        if needs_chunk {
            // Create the leaf chunk (this will replace any existing Solid voxel)
            grandparent.set_chunk(lx, ly, lz, Chunk::new());
        }

        // Now get the leaf chunk and set the voxel in it
        let rank = grandparent.presence.rank(idx) as usize;
        if let Voxel::Chunk(leaf_chunk) = &mut grandparent.voxels[rank - 1] {
            let &(x, y, z) = path.last().unwrap();
            leaf_chunk.set(x, y, z, voxel_type);
        }
    }

    /// Remove a voxel at world position
    pub fn remove(&mut self, pos: WorldPos) {
        let path = match self.position_to_path(pos) {
            Ok(p) => p,
            Err(_) => return, // Out of bounds
        };

        // Navigate to the parent chunk
        let parent = self.navigate_to_mut(&path, self.hierarchy_depth as usize - 1);

        // Remove the leaf voxel
        let &(x, y, z) = path.last().unwrap();
        parent.remove(x, y, z);
    }

    /// Get the root chunk
    pub fn root(&self) -> &Chunk {
        &self.root
    }

    /// Get mutable root chunk
    pub fn root_mut(&mut self) -> &mut Chunk {
        &mut self.root
    }

    /// Subdivide a voxel at world position
    pub fn subdivide_at(&mut self, pos: WorldPos) -> Result<(), &'static str> {
        let path = self.position_to_path(pos)?;
        let parent = self.navigate_to_mut(&path, self.hierarchy_depth as usize - 1);
        let &(x, y, z) = path.last().ok_or("Invalid path")?;
        parent.subdivide(x, y, z)
    }

    /// Try to merge a subdivided voxel back to solid
    pub fn merge_at(&mut self, pos: WorldPos) -> Result<bool, &'static str> {
        let path = self.position_to_path(pos)?;
        let parent = self.navigate_to_mut(&path, self.hierarchy_depth as usize - 1);
        let &(x, y, z) = path.last().ok_or("Invalid path")?;
        parent.try_merge(x, y, z)
    }

    /// Get the hierarchy depth at a world position (beyond the base depth)
    pub fn depth_at(&self, pos: WorldPos) -> Option<usize> {
        let path = self.position_to_path(pos).ok()?;
        let parent = self.navigate_to(&path, self.hierarchy_depth as usize - 1)?;
        let &(x, y, z) = path.last()?;
        parent.depth_at(x, y, z)
    }

    /// Update LOD metadata for all chunks recursively (call after world generation)
    /// This walks through the entire hierarchy and updates voxel_count, average_color,
    /// and emissive aggregates using the provided palette.
    pub fn update_all_lod_metadata(&mut self, palette: &Palette) {
        Self::update_chunk_lod_recursive(&mut self.root, palette);
    }

    /// Recursive helper to update LOD metadata bottom-up
    fn update_chunk_lod_recursive(chunk: &mut Chunk, palette: &Palette) {
        // First, recursively update all sub-chunks. Use Rayon to parallelize recursion across
        // different sub-chunks where possible - this gives a large speedup for deep/large worlds.
        use rayon::prelude::*;
        chunk
            .voxels
            .par_iter_mut()
            .for_each(|voxel| if let Voxel::Chunk(sub_chunk) = voxel {
                Self::update_chunk_lod_recursive(sub_chunk, palette);
            });

        // Then update this chunk's metadata
        chunk.update_lod_metadata(palette);
    }
}

impl Default for World {
    fn default() -> Self {
        // Default to depth 3 (4,096 units per side)
        Self::new(3)
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
        let mut world = World::new(3); // 4,096 units per side

        world.set(WorldPos::new(0, 0, 0), 1);
        world.set(WorldPos::new(100, 200, 300), 2);

        assert_eq!(world.get(WorldPos::new(0, 0, 0)), Some(1));
        assert_eq!(world.get(WorldPos::new(100, 200, 300)), Some(2));
        assert_eq!(world.get(WorldPos::new(1, 1, 1)), None);
    }

    #[test]
    fn test_world_sizes() {
        assert_eq!(World::new(1).world_size(), 16);
        assert_eq!(World::new(2).world_size(), 256);
        assert_eq!(World::new(3).world_size(), 4096);
        assert_eq!(World::new(4).world_size(), 65536);
    }

    #[test]
    fn test_world_bounds() {
        let mut world = World::new(2); // 256 units

        // In bounds
        world.set(WorldPos::new(0, 0, 0), 1);
        world.set(WorldPos::new(255, 255, 255), 2);
        assert_eq!(world.get(WorldPos::new(0, 0, 0)), Some(1));
        assert_eq!(world.get(WorldPos::new(255, 255, 255)), Some(2));

        // Out of bounds
        world.set(WorldPos::new(256, 0, 0), 3);
        world.set(WorldPos::new(-1, 0, 0), 4);
        assert_eq!(world.get(WorldPos::new(256, 0, 0)), None);
        assert_eq!(world.get(WorldPos::new(-1, 0, 0)), None);
    }
}
