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
use std::cell::RefCell;
use std::sync::Arc;

use crate::palette::Palette;

// Thread-local bitmap for ray marching to avoid allocations
thread_local! {
    static RAY_BITMAP: RefCell<Bitmap> = RefCell::new(Bitmap::new());
}

/// Convert a local chunk bounding box (bbox in 0..15 coordinates) to world-space position and
/// world-space size using the supplied scale for the current voxel.
///
/// - bbox: [xmin, ymin, zmin, xmax, ymax, zmax]
/// - origin: world-space origin of the chunk (i64 coordinates) at which local pos 0,0,0 maps to
/// - scale: size of a voxel at this level (e.g., 4096, 256, 16, 1). Note: the helper computes
///   sub-element unit as `unit = scale / 16`; for converting bounding boxes of a leaf chunk
///   to world coordinates pass `scale = 16` (so that `unit = 1.0`). For converting bounding boxes
///   of a sub-chunk inside a chunk, pass the parent chunk's `scale` (e.g., 256 when iterating
///   a parent chunk with `scale=256`).
///
/// Returns: (position_i64, size_f32)
pub fn bbox_local_to_world(origin: [i64; 3], scale: i64, bbox: [u8; 6]) -> ([i64; 3], [f32; 3]) {
    // Each sub-element has size unit = scale / 16.
    let unit = scale / 16;
    let x = origin[0] + (bbox[0] as i64 * unit);
    let y = origin[1] + (bbox[1] as i64 * unit);
    let z = origin[2] + (bbox[2] as i64 * unit);
    let sx = (bbox[3] - bbox[0] + 1) as f32 * unit as f32;
    let sy = (bbox[4] - bbox[1] + 1) as f32 * unit as f32;
    let sz = (bbox[5] - bbox[2] + 1) as f32 * unit as f32;
    ([x, y, z], [sx, sy, sz])
}

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
    Chunk(Arc<Chunk>),
}

/// A voxel on the surface of a chunk, with a bitmask of exposed faces
#[derive(Copy, Clone, Debug)]
pub struct ShellVoxel {
    /// Packed position: x | (y << 4) | (z << 8)
    pub packed_pos: u16,
    /// Bitmask of exposed faces:
    /// bit 0: +X (Right)
    /// bit 1: -X (Left)
    /// bit 2: +Y (Top)
    /// bit 3: -Y (Bottom)
    /// bit 4: +Z (Front)
    /// bit 5: -Z (Back)
    pub visible_faces: u8,
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

    /// Compact axis-aligned bounding box covering all voxels in this chunk.
    /// Stored as local coordinates (xmin, ymin, zmin, xmax, ymax, zmax) within 0..=15.
    /// None if the chunk is empty. Useful for coarse fallback geometry when we don't want to
    /// generate the full mesh for distant chunks.
    pub bounding_box: Option<[u8; 6]>,

    /// Sum of emissive RGB (intensity-weighted) for voxels in this chunk
    pub emissive_sum: [f32; 3],

    /// Total emissive intensity across voxels in this chunk
    pub emissive_power: f32,

    /// Count of voxels contributing emissive light
    pub emissive_voxels: u32,

    /// Ratio of solid voxels to total slots (0.0..=1.0)
    pub solid_ratio: f32,

    /// LOD metadata: The most prominent voxel type on the VISIBLE shell of this chunk.
    /// This is used to determine the average color (mode) for better visual fidelity in reflections.
    pub dominant_type: VoxelType,

    /// Shell of surface sub-chunks for this hierarchy level (None for leaf chunks)
    /// Each entry represents a sub-chunk with at least one exposed face
    /// Enables fast occlusion culling at any hierarchy level
    pub hierarchy_shell: Option<Vec<ShellVoxel>>,
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
            dominant_type: 0,
            bounding_box: None,
            hierarchy_shell: None,
        }
    }

    /// Convert (x, y, z) coordinates to flat index
    /// x, y, z must be in range [0, 15]
    #[inline]
    pub fn flat_index(x: u8, y: u8, z: u8) -> u32 {
        debug_assert!(x < 16 && y < 16 && z < 16);
        (x as u32) | ((y as u32) << 4) | ((z as u32) << 8)
    }

    /// Convert flat index back to (x, y, z)
    #[inline]
    pub fn unflatten(idx: u32) -> (u8, u8, u8) {
        let x = (idx & 0xF) as u8;
        let y = ((idx >> 4) & 0xF) as u8;
        let z = ((idx >> 8) & 0xF) as u8;
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

    /// Get mutable reference to the voxel at (x, y, z)
    pub fn get_mut(&mut self, x: u8, y: u8, z: u8) -> Option<&mut Voxel> {
        if !self.contains(x, y, z) {
            return None;
        }

        let idx = Self::flat_index(x, y, z);
        let rank = self.presence.rank(idx) as usize;

        // rank-1 gives us the index in the voxels array
        self.voxels.get_mut(rank - 1)
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
            self.voxels[rank - 1] = Voxel::Chunk(Arc::new(chunk));
        } else {
            // Insert new
            let rank = self.presence.rank(idx) as usize;
            self.presence.add(idx);
            self.voxels.insert(rank, Voxel::Chunk(Arc::new(chunk)));

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
    /// Update LOD metadata using palette material properties for this chunk.
    /// Should be called after modifying chunk contents.
    /// Update LOD metadata using palette material properties for this chunk.
    /// Should be called after modifying chunk contents.
    pub fn update_lod_metadata(&mut self, palette: &Palette) {
        self.update_lod_metadata_with_mask(palette, 0) // Default: consider all shell voxels
    }

    /// Update LOD metadata with a specific visibility mask (bits 0-5 for +X, -X, +Y, -Y, +Z, -Z).
    /// If mask is 0, falls back to the internal "any surface" shell heuristic.
    /// Uses the MODE (most common type) for color to better represent prominent facades.
    pub fn update_lod_metadata_with_mask(&mut self, palette: &Palette, mask: u8) {
        const TOTAL_SLOTS: f32 = (16 * 16 * 16) as f32; // 4096

        let mut emissive_sum = [0.0f32; 3];
        let mut emissive_power = 0.0f32;
        let mut emissive_voxels = 0u32;
        let mut solid_count = 0u32;

        // Histogram for dominant type calculation (256 types)
        let mut type_counts = [0u32; 256];

        // 1. Accumulate totals for voxel_count and emissive stats across ALL voxels
        for voxel in &self.voxels {
            match voxel {
                Voxel::Solid(voxel_type) => {
                    solid_count += 1;
                    let (em_color, em_strength) = palette.emissive(*voxel_type as u32);
                    if em_strength > 0.0 {
                        emissive_sum[0] += em_color[0] * em_strength;
                        emissive_sum[1] += em_color[1] * em_strength;
                        emissive_sum[2] += em_color[2] * em_strength;
                        emissive_power += em_strength;
                        emissive_voxels += 1;
                    }
                }
                Voxel::Chunk(sub_chunk) => {
                    solid_count = solid_count.saturating_add(sub_chunk.voxel_count);
                    // Propagate recursive emissive stats
                    emissive_sum[0] += sub_chunk.emissive_sum[0];
                    emissive_sum[1] += sub_chunk.emissive_sum[1];
                    emissive_sum[2] += sub_chunk.emissive_sum[2];
                    emissive_power += sub_chunk.emissive_power;
                    emissive_voxels += sub_chunk.emissive_voxels;
                }
            }
        }

        self.voxel_count = solid_count;
        self.solid_ratio = solid_count as f32 / TOTAL_SLOTS;

        // 2. Compute dominant type based on the visual shell only.
        let use_any_shell = mask == 0;

        for ((x, y, z), voxel) in self.iter() {
            let mut is_visible = false;

            if use_any_shell {
                is_visible = x == 0
                    || x == 15
                    || y == 0
                    || y == 15
                    || z == 0
                    || z == 15
                    || !self.contains(x + 1, y, z)
                    || !self.contains(x - 1, y, z)
                    || !self.contains(x, y + 1, z)
                    || !self.contains(x, y - 1, z)
                    || !self.contains(x, y, z + 1)
                    || !self.contains(x, y, z - 1);
            } else {
                if (mask & (1 << 0)) != 0 && x == 15 {
                    is_visible = true;
                }
                if (mask & (1 << 1)) != 0 && x == 0 {
                    is_visible = true;
                }
                if (mask & (1 << 2)) != 0 && y == 15 {
                    is_visible = true;
                }
                if (mask & (1 << 3)) != 0 && y == 0 {
                    is_visible = true;
                }
                if (mask & (1 << 4)) != 0 && z == 15 {
                    is_visible = true;
                }
                if (mask & (1 << 5)) != 0 && z == 0 {
                    is_visible = true;
                }
            }

            if is_visible {
                match voxel {
                    Voxel::Solid(t) => {
                        type_counts[*t as usize] += 1;
                    }
                    Voxel::Chunk(c) => {
                        type_counts[c.dominant_type as usize] += 1;
                    }
                }
            }
        }

        // Find the most frequent type (mode)
        let mut best_type = 0u8;
        let mut max_count = 0u32;
        // Skip type 0 if possible, unless it's the only one (actually Solid(0) shouldn't be empty, but usually 0 is air or similar)
        // In our case Voxel::Solid(0) is a valid material.
        for t in 0..256 {
            if type_counts[t] > max_count {
                max_count = type_counts[t];
                best_type = t as u8;
            }
        }

        self.dominant_type = best_type;

        if max_count > 0 {
            // Set average color to the dominant type's color
            // Use alpha for occupancy
            let color = palette.color_u8(best_type as u32);
            self.average_color = [
                color[0],
                color[1],
                color[2],
                (self.solid_ratio * 255.0).clamp(0.0, 255.0) as u8,
            ];
        } else if solid_count > 0 {
            // Fallback for solid chunks that somehow weren't caught by the shell
            let voxel_type = if let Voxel::Solid(t) = self.voxels[0] {
                t
            } else {
                0
            };
            self.dominant_type = voxel_type;
            let color = palette.color_u8(voxel_type as u32);
            self.average_color = [
                color[0],
                color[1],
                color[2],
                (self.solid_ratio * 255.0) as u8,
            ];
        } else {
            self.average_color = [0, 0, 0, 0];
            self.dominant_type = 0;
        }

        // 3. Compute per-chunk bounding box (covers all solid content)
        let mut xmin: u8 = 16;
        let mut ymin: u8 = 16;
        let mut zmin: u8 = 16;
        let mut xmax: u8 = 0;
        let mut ymax: u8 = 0;
        let mut zmax: u8 = 0;
        let mut bbox_found = false;

        for ((x, y, z), voxel) in self.iter() {
            let is_occupied = match voxel {
                Voxel::Solid(_) => true,
                Voxel::Chunk(c) => !c.is_empty(),
            };

            if is_occupied {
                bbox_found = true;
                xmin = xmin.min(x);
                ymin = ymin.min(y);
                zmin = zmin.min(z);
                xmax = xmax.max(x);
                ymax = ymax.max(y);
                zmax = zmax.max(z);
            }
        }

        if bbox_found {
            self.bounding_box = Some([xmin, ymin, zmin, xmax, ymax, zmax]);
        } else {
            self.bounding_box = None;
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

    /// Check if this chunk is a leaf chunk (contains only solid voxels, no sub-chunks)
    pub fn is_leaf_chunk(&self) -> bool {
        self.voxels.iter().all(|v| matches!(v, Voxel::Solid(_)))
    }

    /// Check if a position is occupied (contains a solid voxel or non-empty sub-chunk)
    #[allow(dead_code)]
    fn is_occupied_at(&self, x: u8, y: u8, z: u8) -> bool {
        match self.get(x, y, z) {
            Some(Voxel::Solid(_)) => true,
            Some(Voxel::Chunk(c)) => c.voxel_count > 0,
            None => false,
        }
    }

    /// Generate a hierarchical shell for non-leaf chunks
    /// Returns a list of sub-chunks that have at least one face exposed
    pub fn generate_hierarchy_shell(&mut self) {
        if self.is_leaf_chunk() || self.is_empty() {
            self.hierarchy_shell = None;
            return;
        }

        let mut shell = Vec::with_capacity(256);

        for ((x, y, z), voxel) in self.iter() {
            // Get the child chunk (skip solids - they're always fully visible)
            let child = match voxel {
                Voxel::Solid(_) => {
                    // Solid voxels: check if neighbors exist
                    let mut mask = 0u8;
                    if x == 15 || self.get(x + 1, y, z).is_none() {
                        mask |= 1 << 0;
                    }
                    if x == 0 || self.get(x - 1, y, z).is_none() {
                        mask |= 1 << 1;
                    }
                    if y == 15 || self.get(x, y + 1, z).is_none() {
                        mask |= 1 << 2;
                    }
                    if y == 0 || self.get(x, y - 1, z).is_none() {
                        mask |= 1 << 3;
                    }
                    if z == 15 || self.get(x, y, z + 1).is_none() {
                        mask |= 1 << 4;
                    }
                    if z == 0 || self.get(x, y, z - 1).is_none() {
                        mask |= 1 << 5;
                    }
                    if mask != 0 {
                        let packed = (x as u16) | ((y as u16) << 4) | ((z as u16) << 8);
                        shell.push(ShellVoxel {
                            packed_pos: packed,
                            visible_faces: mask,
                        });
                    }
                    continue;
                }
                Voxel::Chunk(c) => {
                    if c.voxel_count == 0 {
                        continue;
                    }
                    c
                }
            };

            // Get neighbor chunks for overlap checking
            let get_neighbor = |dx: i8, dy: i8, dz: i8| -> Option<&Arc<Chunk>> {
                let (nx, ny, nz) = (x as i8 + dx, y as i8 + dy, z as i8 + dz);
                if nx < 0 || nx > 15 || ny < 0 || ny > 15 || nz < 0 || nz > 15 {
                    return None;
                }
                match self.get(nx as u8, ny as u8, nz as u8) {
                    Some(Voxel::Chunk(c)) if c.voxel_count > 0 => Some(c),
                    _ => None,
                }
            };

            let neighbor_px = get_neighbor(1, 0, 0);
            let neighbor_nx = get_neighbor(-1, 0, 0);
            let neighbor_py = get_neighbor(0, 1, 0);
            let neighbor_ny = get_neighbor(0, -1, 0);
            let neighbor_pz = get_neighbor(0, 0, 1);
            let neighbor_nz = get_neighbor(0, 0, -1);

            // Compute visibility mask with neighbor overlap
            let mask = child.compute_visibility_mask_with_neighbors(
                neighbor_px,
                neighbor_nx,
                neighbor_py,
                neighbor_ny,
                neighbor_pz,
                neighbor_nz,
            );

            if mask != 0 {
                let packed = (x as u16) | ((y as u16) << 4) | ((z as u16) << 8);
                shell.push(ShellVoxel {
                    packed_pos: packed,
                    visible_faces: mask,
                });
            }
        }

        self.hierarchy_shell = Some(shell);
    }

    /// Compute visibility mask with neighbor overlap.
    /// For boundary voxels, checks if the neighbor chunk has a voxel blocking that face.
    pub fn compute_visibility_mask_with_neighbors(
        &self,
        neighbor_px: Option<&Arc<Chunk>>, // +X neighbor
        neighbor_nx: Option<&Arc<Chunk>>, // -X neighbor
        neighbor_py: Option<&Arc<Chunk>>, // +Y neighbor
        neighbor_ny: Option<&Arc<Chunk>>, // -Y neighbor
        neighbor_pz: Option<&Arc<Chunk>>, // +Z neighbor
        neighbor_nz: Option<&Arc<Chunk>>, // -Z neighbor
    ) -> u8 {
        if self.is_empty() {
            return 0;
        }

        let mut mask = 0u8;

        // Helper: check if neighbor blocks a face at given local position
        // For +X face at (15, y, z), check neighbor_px at (0, y, z)
        let is_blocked_by_neighbor =
            |neighbor: Option<&Arc<Chunk>>, lx: u8, ly: u8, lz: u8| -> bool {
                match neighbor {
                    Some(n) => n.contains(lx, ly, lz),
                    None => false, // No neighbor = exposed to air
                }
            };

        // For leaf chunks, scan voxels
        if self.is_leaf_chunk() {
            for ((x, y, z), voxel) in self.iter() {
                if !matches!(voxel, Voxel::Solid(_)) {
                    continue;
                }

                // +X: voxel has +X exposed if no neighbor at x+1
                if (mask & (1 << 0)) == 0 {
                    let has_internal_neighbor = x < 15 && self.contains(x + 1, y, z);
                    let blocked_by_external =
                        x == 15 && is_blocked_by_neighbor(neighbor_px, 0, y, z);
                    if !has_internal_neighbor && !blocked_by_external {
                        mask |= 1 << 0;
                    }
                }
                // -X
                if (mask & (1 << 1)) == 0 {
                    let has_internal_neighbor = x > 0 && self.contains(x - 1, y, z);
                    let blocked_by_external =
                        x == 0 && is_blocked_by_neighbor(neighbor_nx, 15, y, z);
                    if !has_internal_neighbor && !blocked_by_external {
                        mask |= 1 << 1;
                    }
                }
                // +Y
                if (mask & (1 << 2)) == 0 {
                    let has_internal_neighbor = y < 15 && self.contains(x, y + 1, z);
                    let blocked_by_external =
                        y == 15 && is_blocked_by_neighbor(neighbor_py, x, 0, z);
                    if !has_internal_neighbor && !blocked_by_external {
                        mask |= 1 << 2;
                    }
                }
                // -Y
                if (mask & (1 << 3)) == 0 {
                    let has_internal_neighbor = y > 0 && self.contains(x, y - 1, z);
                    let blocked_by_external =
                        y == 0 && is_blocked_by_neighbor(neighbor_ny, x, 15, z);
                    if !has_internal_neighbor && !blocked_by_external {
                        mask |= 1 << 3;
                    }
                }
                // +Z
                if (mask & (1 << 4)) == 0 {
                    let has_internal_neighbor = z < 15 && self.contains(x, y, z + 1);
                    let blocked_by_external =
                        z == 15 && is_blocked_by_neighbor(neighbor_pz, x, y, 0);
                    if !has_internal_neighbor && !blocked_by_external {
                        mask |= 1 << 4;
                    }
                }
                // -Z
                if (mask & (1 << 5)) == 0 {
                    let has_internal_neighbor = z > 0 && self.contains(x, y, z - 1);
                    let blocked_by_external =
                        z == 0 && is_blocked_by_neighbor(neighbor_nz, x, y, 15);
                    if !has_internal_neighbor && !blocked_by_external {
                        mask |= 1 << 5;
                    }
                }

                if mask == 0b111111 {
                    break;
                }
            }
        } else {
            // For non-leaf chunks, use the hierarchy shell
            if let Some(ref shell) = self.hierarchy_shell {
                for sv in shell.iter() {
                    mask |= sv.visible_faces;
                    if mask == 0b111111 {
                        break;
                    }
                }
            }
        }

        mask
    }

    /// Compute a visibility mask for this chunk using a greedy algorithm.
    /// For each of the 6 directions, we only need to find ONE voxel with that face exposed.
    /// This is much faster than computing the full shell for large chunks.
    ///
    /// Returns a bitmask where each bit indicates if that face direction has any visible geometry:
    /// bit 0: +X, bit 1: -X, bit 2: +Y, bit 3: -Y, bit 4: +Z, bit 5: -Z
    pub fn compute_visibility_mask(&self) -> u8 {
        if self.is_empty() {
            return 0;
        }

        let mut mask = 0u8;

        if self.is_leaf_chunk() {
            // For leaf chunks, scan face voxels until we find one exposed
            // We iterate through the presence bitmap which is sparse

            for ((x, y, z), voxel) in self.iter() {
                if !matches!(voxel, Voxel::Solid(_)) {
                    continue;
                }

                // Check each direction we haven't found yet
                // +X: voxel at x=15 or with no +X neighbor
                if (mask & (1 << 0)) == 0 && (x == 15 || !self.contains(x + 1, y, z)) {
                    mask |= 1 << 0;
                }
                // -X: voxel at x=0 or with no -X neighbor
                if (mask & (1 << 1)) == 0 && (x == 0 || !self.contains(x - 1, y, z)) {
                    mask |= 1 << 1;
                }
                // +Y: voxel at y=15 or with no +Y neighbor
                if (mask & (1 << 2)) == 0 && (y == 15 || !self.contains(x, y + 1, z)) {
                    mask |= 1 << 2;
                }
                // -Y: voxel at y=0 or with no -Y neighbor
                if (mask & (1 << 3)) == 0 && (y == 0 || !self.contains(x, y - 1, z)) {
                    mask |= 1 << 3;
                }
                // +Z: voxel at z=15 or with no +Z neighbor
                if (mask & (1 << 4)) == 0 && (z == 15 || !self.contains(x, y, z + 1)) {
                    mask |= 1 << 4;
                }
                // -Z: voxel at z=0 or with no -Z neighbor
                if (mask & (1 << 5)) == 0 && (z == 0 || !self.contains(x, y, z - 1)) {
                    mask |= 1 << 5;
                }

                // Early exit if all 6 faces found
                if mask == 0b111111 {
                    break;
                }
            }
        } else {
            // For non-leaf chunks, aggregate from children's visibility masks
            // A face is visible if ANY child on that face has it visible
            if let Some(ref shell) = self.hierarchy_shell {
                for sv in shell.iter() {
                    mask |= sv.visible_faces;
                    if mask == 0b111111 {
                        break;
                    }
                }
            }
        }

        mask
    }

    /// Generate a shell of surface voxels for this chunk (leaf level only)
    /// Returns a list of voxels that have at least one face exposed to air (or chunk boundary)
    pub fn generate_shell(&self) -> Vec<ShellVoxel> {
        let mut shell = Vec::with_capacity(512); // Heuristic start size

        for ((x, y, z), voxel) in self.iter() {
            // Only consider solid voxels for the shell
            if let Voxel::Solid(_) = voxel {
                let mut mask = 0u8;

                // Check 6 neighbors
                // If neighbor is AIR (not contained) or boundary, set the bit.

                // +X (Right)
                if x == 15 || !self.contains(x + 1, y, z) {
                    mask |= 1 << 0;
                }
                // -X (Left)
                if x == 0 || !self.contains(x - 1, y, z) {
                    mask |= 1 << 1;
                }
                // +Y (Top)
                if y == 15 || !self.contains(x, y + 1, z) {
                    mask |= 1 << 2;
                }
                // -Y (Bottom)
                if y == 0 || !self.contains(x, y - 1, z) {
                    mask |= 1 << 3;
                }
                // +Z (Front)
                if z == 15 || !self.contains(x, y, z + 1) {
                    mask |= 1 << 4;
                }
                // -Z (Back)
                if z == 0 || !self.contains(x, y, z - 1) {
                    mask |= 1 << 5;
                }

                if mask != 0 {
                    let packed = (x as u16) | ((y as u16) << 4) | ((z as u16) << 8);
                    shell.push(ShellVoxel {
                        packed_pos: packed,
                        visible_faces: mask,
                    });
                }
            }
        }
        shell
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
    root: Arc<Chunk>,

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
            root: Arc::new(Chunk::new()),
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

    /// Get the Arc<Chunk> at the given origin position (avoids cloning the chunk)
    pub fn get_leaf_chunk_arc_at_origin(&self, origin: WorldPos) -> Option<Arc<Chunk>> {
        // Ensure alignment (optional safety)
        if (origin.x & 15) != 0 || (origin.y & 15) != 0 || (origin.z & 15) != 0 {
            return None;
        }

        if self.hierarchy_depth == 1 {
            // Special case: root is the leaf chunk - clone the Arc
            return Some(self.root.clone());
        }

        let path = self.position_to_path(origin).ok()?;
        // Navigate to the parent of the leaf chunk level (depth-2)
        let parent = self.navigate_to(&path, self.hierarchy_depth as usize - 2)?;
        // The leaf chunk is at position path[depth-2] in that parent
        let &(x, y, z) = &path[self.hierarchy_depth as usize - 2];
        match parent.get(x, y, z)? {
            Voxel::Chunk(c) => Some(c.clone()), // Clone the Arc, not the Chunk!
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
        let mut current = Arc::make_mut(&mut self.root);

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

            // Navigate into the chunk - need to use Arc::make_mut
            let rank = current.presence.rank(idx) as usize;
            match &mut current.voxels[rank - 1] {
                Voxel::Chunk(chunk_arc) => {
                    current = Arc::make_mut(chunk_arc);
                }
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

    /// Check line of sight between two world positions using hierarchical bitmap intersection
    /// Returns true if there's a clear line of sight (no voxels blocking)
    pub fn line_of_sight(&self, start: WorldPos, end: WorldPos) -> bool {
        // Early check: if start == end, we have line of sight
        if start == end {
            return true;
        }

        // Use thread-local bitmap to avoid allocations
        RAY_BITMAP.with(|bitmap_cell| {
            let mut bitmap = bitmap_cell.borrow_mut();
            bitmap.clear();

            // Start hierarchical traversal from root
            self.line_of_sight_recursive(
                &self.root,
                start,
                end,
                WorldPos::new(0, 0, 0), // Root origin
                self.hierarchy_depth,
                &mut bitmap,
            )
        })
    }

    /// Recursive helper for line_of_sight using hierarchical bitmap intersection
    fn line_of_sight_recursive(
        &self,
        chunk: &Chunk,
        start: WorldPos,
        end: WorldPos,
        chunk_origin: WorldPos,
        depth: u8,
        bitmap: &mut Bitmap,
    ) -> bool {
        // Calculate the size of voxels at this level
        let voxel_size = 16i64.pow((depth - 1) as u32);

        // Compute which voxels in this chunk the ray passes through
        bitmap.clear();
        self.rasterize_ray_in_chunk(start, end, chunk_origin, voxel_size, bitmap);

        // Fast check: if ray doesn't pass through any voxels in chunk's presence bitmap
        if !bitmap.intersect(&chunk.presence) {
            return true; // Clear line of sight through this chunk
        }

        // Ray intersects with occupied voxels - need to check deeper
        // If we're at leaf level, we have an obstruction
        if depth == 1 {
            return false; // Hit a solid voxel
        }

        // Not at leaf level - descend into sub-chunks that the ray intersects
        // Only check the voxels where bitmap AND presence overlap
        let intersection = bitmap.and(&chunk.presence);

        for idx in intersection.iter() {
            let (x, y, z) = Chunk::unflatten(idx);

            // Get the sub-chunk at this position
            let rank = chunk.presence.rank(idx) as usize;
            if let Some(Voxel::Chunk(sub_chunk)) = chunk.voxels.get(rank - 1) {
                // Calculate origin of this sub-chunk
                let sub_origin = WorldPos::new(
                    chunk_origin.x + (x as i64 * voxel_size),
                    chunk_origin.y + (y as i64 * voxel_size),
                    chunk_origin.z + (z as i64 * voxel_size),
                );

                // Recursively check this sub-chunk
                if !self.line_of_sight_recursive(
                    sub_chunk,
                    start,
                    end,
                    sub_origin,
                    depth - 1,
                    bitmap,
                ) {
                    return false; // Found obstruction
                }
            } else {
                // It's a solid voxel at a non-leaf level - obstruction
                return false;
            }
        }

        // No obstructions found
        true
    }

    /// Rasterize a ray into a bitmap of which voxels (0-15 in each axis) it passes through
    /// Uses a 3D DDA algorithm
    fn rasterize_ray_in_chunk(
        &self,
        start: WorldPos,
        end: WorldPos,
        chunk_origin: WorldPos,
        voxel_size: i64,
        bitmap: &mut Bitmap,
    ) {
        // Convert world positions to chunk-local coordinates (in voxel units 0-15)
        let start_local = [
            ((start.x - chunk_origin.x) as f64 / voxel_size as f64),
            ((start.y - chunk_origin.y) as f64 / voxel_size as f64),
            ((start.z - chunk_origin.z) as f64 / voxel_size as f64),
        ];

        let end_local = [
            ((end.x - chunk_origin.x) as f64 / voxel_size as f64),
            ((end.y - chunk_origin.y) as f64 / voxel_size as f64),
            ((end.z - chunk_origin.z) as f64 / voxel_size as f64),
        ];

        // DDA ray traversal
        let delta = [
            end_local[0] - start_local[0],
            end_local[1] - start_local[1],
            end_local[2] - start_local[2],
        ];

        let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if length < 0.001 {
            // Ray is too short, just add start voxel if in bounds
            let x = start_local[0].floor() as i32;
            let y = start_local[1].floor() as i32;
            let z = start_local[2].floor() as i32;
            if x >= 0 && x < 16 && y >= 0 && y < 16 && z >= 0 && z < 16 {
                bitmap.add(Chunk::flat_index(x as u8, y as u8, z as u8));
            }
            return;
        }

        // Normalized direction
        let dir = [delta[0] / length, delta[1] / length, delta[2] / length];

        // Step sizes for each axis
        let step_x = if dir[0].abs() > 0.0001 {
            1.0 / dir[0].abs()
        } else {
            f64::MAX
        };
        let step_y = if dir[1].abs() > 0.0001 {
            1.0 / dir[1].abs()
        } else {
            f64::MAX
        };
        let step_z = if dir[2].abs() > 0.0001 {
            1.0 / dir[2].abs()
        } else {
            f64::MAX
        };

        // Current voxel
        let mut vx = start_local[0].floor() as i32;
        let mut vy = start_local[1].floor() as i32;
        let mut vz = start_local[2].floor() as i32;

        // Initial t-values to next voxel boundaries
        let mut t_max_x = if dir[0] > 0.0 {
            ((vx + 1) as f64 - start_local[0]) / dir[0]
        } else if dir[0] < 0.0 {
            (vx as f64 - start_local[0]) / dir[0]
        } else {
            f64::MAX
        };

        let mut t_max_y = if dir[1] > 0.0 {
            ((vy + 1) as f64 - start_local[1]) / dir[1]
        } else if dir[1] < 0.0 {
            (vy as f64 - start_local[1]) / dir[1]
        } else {
            f64::MAX
        };

        let mut t_max_z = if dir[2] > 0.0 {
            ((vz + 1) as f64 - start_local[2]) / dir[2]
        } else if dir[2] < 0.0 {
            (vz as f64 - start_local[2]) / dir[2]
        } else {
            f64::MAX
        };

        // Step directions
        let step_dir_x = if dir[0] > 0.0 { 1 } else { -1 };
        let step_dir_y = if dir[1] > 0.0 { 1 } else { -1 };
        let step_dir_z = if dir[2] > 0.0 { 1 } else { -1 };

        // Traverse the ray
        let max_steps = 48; // Max voxels to check (covers diagonal + some margin)
        for _ in 0..max_steps {
            // Add current voxel if in bounds
            if vx >= 0 && vx < 16 && vy >= 0 && vy < 16 && vz >= 0 && vz < 16 {
                bitmap.add(Chunk::flat_index(vx as u8, vy as u8, vz as u8));
            }

            // Check if we've passed the end point
            let current = [vx as f64 + 0.5, vy as f64 + 0.5, vz as f64 + 0.5];
            let to_end = [
                end_local[0] - current[0],
                end_local[1] - current[1],
                end_local[2] - current[2],
            ];
            let dist_sq = to_end[0] * to_end[0] + to_end[1] * to_end[1] + to_end[2] * to_end[2];
            if dist_sq < 0.5 {
                break; // Reached end
            }

            // Step to next voxel boundary
            if t_max_x < t_max_y {
                if t_max_x < t_max_z {
                    vx += step_dir_x;
                    t_max_x += step_x;
                } else {
                    vz += step_dir_z;
                    t_max_z += step_z;
                }
            } else {
                if t_max_y < t_max_z {
                    vy += step_dir_y;
                    t_max_y += step_y;
                } else {
                    vz += step_dir_z;
                    t_max_z += step_z;
                }
            }

            // Safety: exit if we've gone too far outside the chunk
            if vx < -2 || vx > 17 || vy < -2 || vy > 17 || vz < -2 || vz > 17 {
                break;
            }
        }
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
            Arc::make_mut(&mut self.root).set(x, y, z, voxel_type);
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
        if let Voxel::Chunk(leaf_chunk_arc) = &mut grandparent.voxels[rank - 1] {
            let &(x, y, z) = path.last().unwrap();
            Arc::make_mut(leaf_chunk_arc).set(x, y, z, voxel_type);
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
        Arc::make_mut(&mut self.root)
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
        let root_mut = Arc::make_mut(&mut self.root);
        Self::update_chunk_lod_recursive(root_mut, palette);
    }

    /// Recursive helper to update LOD metadata bottom-up
    fn update_chunk_lod_recursive(chunk: &mut Chunk, palette: &Palette) {
        // First, recursively update all sub-chunks. Use Rayon to parallelize recursion across
        // different sub-chunks where possible - this gives a large speedup for deep/large worlds.
        use rayon::prelude::*;
        chunk.voxels.par_iter_mut().for_each(|voxel| {
            if let Voxel::Chunk(sub_chunk_arc) = voxel {
                // Use Arc::make_mut to get exclusive access for mutation
                let sub_chunk = Arc::make_mut(sub_chunk_arc);
                Self::update_chunk_lod_recursive(sub_chunk, palette);
            }
        });

        // Then update this chunk's metadata
        chunk.update_lod_metadata(palette);
    }

    /// Generate hierarchical shells for all non-leaf chunks
    /// Call after update_all_lod_metadata() to enable efficient occlusion culling
    pub fn generate_all_hierarchy_shells(&mut self) {
        let root_mut = Arc::make_mut(&mut self.root);
        let stats = Self::generate_hierarchy_shells_recursive(root_mut, 0);
        println!("Shell generation stats: {:?}", stats);
    }

    /// Recursive helper to generate hierarchy shells bottom-up
    /// Returns (total_shells, total_entries, masks_by_depth)
    fn generate_hierarchy_shells_recursive(
        chunk: &mut Chunk,
        depth: usize,
    ) -> (usize, usize, Vec<u32>) {
        use rayon::prelude::*;

        // First, recursively generate shells for all sub-chunks
        let child_stats: Vec<_> = chunk
            .voxels
            .par_iter_mut()
            .filter_map(|voxel| {
                if let Voxel::Chunk(sub_chunk_arc) = voxel {
                    let sub_chunk = Arc::make_mut(sub_chunk_arc);
                    Some(Self::generate_hierarchy_shells_recursive(
                        sub_chunk,
                        depth + 1,
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Aggregate child stats
        let mut total_shells = 0usize;
        let mut total_entries = 0usize;
        let mut mask_counts = vec![0u32; 64]; // Count of each possible mask value
        for (s, e, m) in child_stats {
            total_shells += s;
            total_entries += e;
            for (i, &c) in m.iter().enumerate() {
                if i < mask_counts.len() {
                    mask_counts[i] += c;
                }
            }
        }

        // Then generate shell for this chunk
        chunk.generate_hierarchy_shell();

        if let Some(ref shell) = chunk.hierarchy_shell {
            total_shells += 1;
            total_entries += shell.len();
            for sv in shell.iter() {
                mask_counts[sv.visible_faces as usize] += 1;
            }
        }

        (total_shells, total_entries, mask_counts)
    }

    /// Second pass of LOD updates: Propagate visibility masks top-down to refine average colors.
    /// This ensures chunks on building facades don't have their average color diluted by buried faces.
    pub fn update_all_visual_lod_metadata(&mut self, palette: &Palette) {
        let root_mut = Arc::make_mut(&mut self.root);
        // Root is visible from all sides in the world view
        Self::update_visual_lod_recursive(root_mut, palette, 0b111111);
    }

    fn update_visual_lod_recursive(chunk: &mut Chunk, palette: &Palette, mask: u8) {
        // Update this chunk with the mask from its parent
        chunk.update_lod_metadata_with_mask(palette, mask);

        // If it's a branch, propagate masks to children in the shell
        let shell_entries = chunk.hierarchy_shell.clone();

        if let Some(entries) = shell_entries {
            for entry in entries {
                let x = (entry.packed_pos & 0xF) as u8;
                let y = ((entry.packed_pos >> 4) & 0xF) as u8;
                let z = ((entry.packed_pos >> 8) & 0xF) as u8;

                if let Some(voxel) = chunk.get_mut(x, y, z) {
                    if let Voxel::Chunk(ref mut child_arc) = voxel {
                        let child = Arc::make_mut(child_arc);
                        Self::update_visual_lod_recursive(child, palette, entry.visible_faces);
                    }
                }
            }
        }
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

    #[test]
    fn test_bbox_local_to_world_leaf() {
        // Leaf chunk: scale=16 -> unit=1
        let bbox = [7u8, 7, 7, 7, 7, 7];
        let (pos, size) = crate::lib_hierarchical::bbox_local_to_world([0, 0, 0], 16, bbox);
        assert_eq!(pos, [7, 7, 7]);
        assert_eq!(size, [1.0, 1.0, 1.0]);
    }
}
