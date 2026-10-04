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
use std::sync::{Arc, LazyLock, OnceLock};

use crate::palette::Palette;

// Thread-local bitmap for ray marching to avoid allocations
thread_local! {
    static RAY_BITMAP: RefCell<Bitmap> = RefCell::new(Bitmap::new());
}

/// Full 16^3 chunk of a single voxel type, shared by every lookup that has to answer for a coarse
/// uniform `Voxel::Solid` as if it were a subdivided leaf chunk. Built lazily per type.
fn uniform_leaf_chunk(voxel_type: VoxelType) -> &'static Arc<Chunk> {
    static CACHE: LazyLock<Vec<OnceLock<Arc<Chunk>>>> = LazyLock::new(|| {
        (0..=VoxelType::MAX as usize)
            .map(|_| OnceLock::new())
            .collect()
    });
    CACHE[voxel_type as usize].get_or_init(|| Arc::new(Chunk::full(voxel_type)))
}

/// What occupies the 16^3 cell of a leaf chunk (see `World::leaf_cell_at_origin`).
enum LeafCell<'a> {
    /// A subdivided leaf chunk.
    Chunk(&'a Arc<Chunk>),
    /// A uniform `Voxel::Solid` stored at this level or above, covering the whole cell.
    Uniform(VoxelType),
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

    /// Create a chunk filled with a single voxel type
    pub fn full(voxel_type: VoxelType) -> Self {
        let mut chunk = Self::new();
        if voxel_type == 0 {
            return chunk;
        }

        chunk.px = 0xFFFF;
        chunk.py = 0xFFFF;
        chunk.pz = 0xFFFF;
        chunk.presence = Bitmap::from_range(0..4096);
        chunk.voxels = vec![Voxel::Solid(voxel_type); 4096];
        chunk.voxel_count = 4096;
        chunk.solid_ratio = 1.0;
        chunk.dominant_type = voxel_type;
        chunk
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

        if voxel_type == 0 {
            if self.presence.contains(idx) {
                let rank = self.presence.rank(idx) as usize;
                self.presence.remove(idx);
                self.voxels.remove(rank - 1);

                // Note: updating px, py, pz marginals is hard when removing,
                // so we leave them set (over-conservative but correct).
            }
            return;
        }

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

    /// Mutable access to the sub-chunk at (x, y, z), creating an empty one if the slot is free or
    /// splitting a `Voxel::Solid` into a full chunk of the same type.
    ///
    /// The slot is classified without cloning the voxel: a cloned `Arc` would still be alive when
    /// `Arc::make_mut` runs, which then always deep-copies the sub-chunk even if it is not shared.
    fn child_chunk_mut(&mut self, x: u8, y: u8, z: u8) -> &mut Chunk {
        let idx = Self::flat_index(x, y, z);

        // `Some(rank)` if the slot already holds a chunk, otherwise what to create.
        let mut existing_rank = None;
        let mut split_type = None;
        if self.presence.contains(idx) {
            let rank = self.presence.rank(idx) as usize;
            match &self.voxels[rank - 1] {
                Voxel::Chunk(_) => existing_rank = Some(rank),
                Voxel::Solid(t) => split_type = Some(*t),
            }
        }

        let rank = match existing_rank {
            Some(rank) => rank,
            None => {
                let new_chunk = match split_type {
                    Some(t) => Chunk::full(t),
                    None => Chunk::new(),
                };
                self.set_chunk(x, y, z, new_chunk);
                self.presence.rank(idx) as usize
            }
        };

        match &mut self.voxels[rank - 1] {
            Voxel::Chunk(chunk_arc) => Arc::make_mut(chunk_arc),
            Voxel::Solid(_) => unreachable!("slot was just made a chunk"),
        }
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

        // Update marginals: a bit may only be cleared when its whole slice is now empty. (Checking
        // just the row through the removed voxel hid other voxels in the same slice.)
        if (0..256u32).all(|yz| !self.presence.contains(x as u32 | (yz << 4))) {
            self.px &= !(1 << x);
        }
        let y_empty = (0..16u32).all(|zi| {
            let start = (zi << 8) | ((y as u32) << 4);
            self.presence.range_cardinality(start..start + 16) == 0
        });
        if y_empty {
            self.py &= !(1 << y);
        }
        let z_start = (z as u32) << 8;
        if self.presence.range_cardinality(z_start..z_start + 256) == 0 {
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
        self.update_lod_metadata_with_mask_internal(palette, mask, false);
    }

    /// Update LOD metadata but preserve precomputed voxel_count/bounding_box if available.
    /// Useful when counts/bounds were computed during file load.
    pub fn update_lod_metadata_with_mask_preserve_bounds(&mut self, palette: &Palette, mask: u8) {
        self.update_lod_metadata_with_mask_internal(palette, mask, true);
    }

    fn update_lod_metadata_with_mask_internal(
        &mut self,
        palette: &Palette,
        mask: u8,
        preserve_bounds: bool,
    ) {
        const TOTAL_SLOTS: f32 = (16 * 16 * 16) as f32; // 4096

        let mut emissive_sum = [0.0f32; 3];
        let mut emissive_power = 0.0f32;
        let mut emissive_voxels = 0u32;

        let reuse_counts = preserve_bounds && self.voxel_count > 0;
        let mut solid_count = if reuse_counts { self.voxel_count } else { 0u32 };

        // 1. Accumulate emissive stats across ALL voxels
        for voxel in &self.voxels {
            match voxel {
                Voxel::Solid(voxel_type) => {
                    if !reuse_counts {
                        solid_count = solid_count.saturating_add(1);
                    }
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
                    if !reuse_counts {
                        solid_count = solid_count.saturating_add(sub_chunk.voxel_count);
                    }
                    // Propagate recursive emissive stats
                    emissive_sum[0] += sub_chunk.emissive_sum[0];
                    emissive_sum[1] += sub_chunk.emissive_sum[1];
                    emissive_sum[2] += sub_chunk.emissive_sum[2];
                    emissive_power += sub_chunk.emissive_power;
                    emissive_voxels += sub_chunk.emissive_voxels;
                }
            }
        }

        if !reuse_counts {
            self.voxel_count = solid_count;
        }
        self.solid_ratio = self.voxel_count as f32 / TOTAL_SLOTS;
        // 2. Compute dominant type and average color based on the visual shell only.
        let use_any_shell = mask == 0;
        let mut color_sum = [0.0f32; 3];
        let mut visible_total = 0u32;
        let mut type_scores = [0.0f32; 256];

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
                let (v_type, v_color) = match voxel {
                    Voxel::Solid(t) => (*t, palette.color(*t as u32)),
                    Voxel::Chunk(c) => (c.dominant_type, Palette::normalize_rgba(c.average_color)),
                };

                let reflectivity = palette.reflectivity(v_type as u32);
                let brightness = (v_color[0] + v_color[1] + v_color[2]) / 3.0;
                let reflective_weight = 1.0 + reflectivity * (2.0 + brightness);
                type_scores[v_type as usize] += reflective_weight;

                color_sum[0] += v_color[0];
                color_sum[1] += v_color[1];
                color_sum[2] += v_color[2];
                visible_total += 1;
            }
        }

        // Find the best type based on scores (favors reflective materials)
        let mut best_type = 0u8;
        let mut max_score = 0.0f32;
        for t in 0..256 {
            if type_scores[t] > max_score {
                max_score = type_scores[t];
                best_type = t as u8;
            }
        }
        self.dominant_type = best_type;

        if visible_total > 0 {
            // Compute real average color of the shell
            let avg_r = (color_sum[0] / visible_total as f32).clamp(0.0, 1.0);
            let avg_g = (color_sum[1] / visible_total as f32).clamp(0.0, 1.0);
            let avg_b = (color_sum[2] / visible_total as f32).clamp(0.0, 1.0);

            self.average_color = [
                (avg_r * 255.0) as u8,
                (avg_g * 255.0) as u8,
                (avg_b * 255.0) as u8,
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
        // Use marginal bitmasks for O(16) bounds instead of scanning all voxels.
        if !preserve_bounds || self.bounding_box.is_none() {
            if self.voxel_count == 0 || self.px == 0 || self.py == 0 || self.pz == 0 {
                self.bounding_box = None;
            } else {
                let xmin = self.px.trailing_zeros() as u8;
                let xmax = 15u8.saturating_sub(self.px.leading_zeros() as u8);
                let ymin = self.py.trailing_zeros() as u8;
                let ymax = 15u8.saturating_sub(self.py.leading_zeros() as u8);
                let zmin = self.pz.trailing_zeros() as u8;
                let zmax = 15u8.saturating_sub(self.pz.leading_zeros() as u8);
                self.bounding_box = Some([xmin, ymin, zmin, xmax, ymax, zmax]);
            }
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

        // Create a new chunk filled with voxels of the same type using Chunk::full
        let sub_chunk = Chunk::full(voxel_type);

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
#[derive(Clone, Debug)]
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

    /// Return the occupied world-space bounds as `[min, max)`, or `None` for an empty world.
    pub fn occupied_bounds(&self) -> Option<([i64; 3], [i64; 3])> {
        let mut bounds = None;
        Self::accumulate_occupied_bounds(
            self.root(),
            [0, 0, 0],
            self.world_size() as i64,
            &mut bounds,
        );
        bounds
    }

    fn include_occupied_bounds(
        bounds: &mut Option<([i64; 3], [i64; 3])>,
        min: [i64; 3],
        max: [i64; 3],
    ) {
        match bounds {
            Some((bounds_min, bounds_max)) => {
                for axis in 0..3 {
                    bounds_min[axis] = bounds_min[axis].min(min[axis]);
                    bounds_max[axis] = bounds_max[axis].max(max[axis]);
                }
            }
            None => {
                *bounds = Some((min, max));
            }
        }
    }

    fn accumulate_occupied_bounds(
        chunk: &Chunk,
        origin: [i64; 3],
        scale: i64,
        bounds: &mut Option<([i64; 3], [i64; 3])>,
    ) {
        if chunk.is_empty() {
            return;
        }

        let unit = (scale / 16).max(1);
        for (x, y, z) in chunk.positions() {
            let child_min = [
                origin[0] + x as i64 * unit,
                origin[1] + y as i64 * unit,
                origin[2] + z as i64 * unit,
            ];
            match chunk.get(x, y, z) {
                Some(Voxel::Solid(_)) => {
                    Self::include_occupied_bounds(
                        bounds,
                        child_min,
                        [
                            child_min[0] + unit,
                            child_min[1] + unit,
                            child_min[2] + unit,
                        ],
                    );
                }
                Some(Voxel::Chunk(sub_chunk)) => {
                    Self::accumulate_occupied_bounds(sub_chunk, child_min, unit, bounds);
                }
                None => {}
            }
        }
    }

    /// Find what occupies the 16^3 leaf-chunk cell at `origin` (aligned to 16): a subdivided chunk,
    /// or a uniform `Voxel::Solid` stored at the leaf-chunk level or at any level above it.
    /// `None` for an empty or out-of-bounds cell and for unaligned origins.
    fn leaf_cell_at_origin(&self, origin: WorldPos) -> Option<LeafCell<'_>> {
        // Ensure alignment (optional safety)
        if (origin.x & 15) != 0 || (origin.y & 15) != 0 || (origin.z & 15) != 0 {
            return None;
        }

        if self.hierarchy_depth == 1 {
            // Special case: root is the leaf chunk
            return Some(LeafCell::Chunk(&self.root));
        }

        let path = self.position_to_path(origin).ok()?;
        // Walk to the leaf chunk's cell, which sits at path[depth-2] in a chunk one level up
        let last_level = self.hierarchy_depth as usize - 2;
        let mut current: &Chunk = &self.root;
        for (level, &(x, y, z)) in path[..=last_level].iter().enumerate() {
            match current.get(x, y, z)? {
                Voxel::Solid(t) => return Some(LeafCell::Uniform(*t)),
                Voxel::Chunk(c) if level == last_level => return Some(LeafCell::Chunk(c)),
                Voxel::Chunk(c) => current = c,
            }
        }
        None
    }

    /// Get a reference to the 16x16x16 leaf chunk located at the given world origin (must be aligned to 16).
    /// Returns None if that position is empty. A uniform `Voxel::Solid` covering the cell (stored at
    /// the leaf-chunk level or above) is returned as a full chunk of its type; that chunk is shared
    /// and only carries the type and counts, not LOD metadata such as `average_color`.
    pub fn get_leaf_chunk_at_origin(&self, origin: WorldPos) -> Option<&Chunk> {
        match self.leaf_cell_at_origin(origin)? {
            LeafCell::Chunk(c) => Some(c),
            LeafCell::Uniform(t) => Some(uniform_leaf_chunk(t)),
        }
    }

    /// Get the Arc<Chunk> at the given origin position (avoids cloning the chunk).
    /// Coarse uniform solids are expanded as in `get_leaf_chunk_at_origin`.
    pub fn get_leaf_chunk_arc_at_origin(&self, origin: WorldPos) -> Option<Arc<Chunk>> {
        match self.leaf_cell_at_origin(origin)? {
            LeafCell::Chunk(c) => Some(c.clone()), // Clone the Arc, not the Chunk!
            LeafCell::Uniform(t) => Some(uniform_leaf_chunk(t).clone()),
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

    /// Borrow the leaf chunk (`Arc`) whose 16-aligned origin is `origin`, without allocating.
    ///
    /// Same lookup as `get_leaf_chunk_arc_at_origin`, but it walks the hierarchy directly with
    /// shifts instead of materialising a `Vec` path via `position_to_path`, and it hands out a
    /// borrow so the caller only pays for an `Arc` clone if it keeps the chunk. Returns `None`
    /// for misaligned or out-of-world origins and for empty cells. A uniform `Voxel::Solid` covering
    /// the cell is answered with the shared full chunk of its type, exactly as
    /// `get_leaf_chunk_arc_at_origin` does.
    ///
    /// Unlike `get_leaf_chunk_arc_at_origin`, a depth-1 world only resolves origin (0, 0, 0)
    /// (the old code returned the root chunk for *any* origin, including positions outside it).
    pub fn leaf_chunk_arc_ref_at_origin(&self, origin: WorldPos) -> Option<&Arc<Chunk>> {
        if (origin.x | origin.y | origin.z) & 15 != 0 {
            return None;
        }
        let world_size = self.world_size() as i64;
        if origin.x < 0
            || origin.y < 0
            || origin.z < 0
            || origin.x >= world_size
            || origin.y >= world_size
            || origin.z >= world_size
        {
            return None;
        }
        if self.hierarchy_depth == 1 {
            return Some(&self.root);
        }

        // Levels `depth-1 ..= 1` each select one child; the child picked at level 1 is the leaf.
        let mut current: &Chunk = &self.root;
        for level in (1..self.hierarchy_depth as u32).rev() {
            let shift = 4 * level;
            let lx = ((origin.x >> shift) & 15) as u8;
            let ly = ((origin.y >> shift) & 15) as u8;
            let lz = ((origin.z >> shift) & 15) as u8;
            match current.get(lx, ly, lz)? {
                Voxel::Chunk(child) if level == 1 => return Some(child),
                Voxel::Chunk(child) => current = child,
                Voxel::Solid(t) => return Some(uniform_leaf_chunk(*t)),
            }
        }
        None
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
            current = current.child_chunk_mut(x, y, z);
        }

        current
    }

    /// Get voxel type at world position (only works for Solid voxels).
    ///
    /// A uniform `Voxel::Solid` stored above the leaf level (covering a whole sub-chunk) answers
    /// with its type for every voxel inside it, as if it were subdivided.
    pub fn get(&self, pos: WorldPos) -> Option<VoxelType> {
        let path = self.position_to_path(pos).ok()?;
        let last_level = path.len() - 1;

        let mut current: &Chunk = &self.root;
        for (level, &(x, y, z)) in path.iter().enumerate() {
            match current.get(x, y, z)? {
                Voxel::Solid(t) => return Some(*t),
                Voxel::Chunk(c) if level < last_level => current = c,
                Voxel::Chunk(_) => return None, // a chunk where a voxel should be
            }
        }
        None
    }

    /// Check line of sight between two world positions using hierarchical bitmap intersection
    /// Returns true if there's a clear line of sight (no voxels blocking).
    ///
    /// The segment runs between the centres of the `start` and `end` voxels. The `end` voxel
    /// itself is not tested (it is the target, typically an emissive voxel); every other voxel
    /// the segment passes through, including `start`, can block. A voxel only counts if the
    /// segment has positive length inside it, so rays grazing an edge or corner are not blocked.
    /// Coarse uniform solids block like any other solid (even one containing `end`), either
    /// endpoint may lie outside the world, and `start == end` is always clear.
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

        // The voxel containing `end` is the target (e.g. the emissive voxel being lit from) and
        // must not occlude its own ray. Only a leaf-level voxel can be excluded; a coarse
        // uniform solid that happens to contain `end` still blocks.
        if depth == 1 {
            let (lx, ly, lz) = (
                end.x - chunk_origin.x,
                end.y - chunk_origin.y,
                end.z - chunk_origin.z,
            );
            if (0..16).contains(&lx) && (0..16).contains(&ly) && (0..16).contains(&lz) {
                bitmap.remove(Chunk::flat_index(lx as u8, ly as u8, lz as u8));
            }
        }

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

    /// Rasterize the segment between the *centres* of the `start` and `end` voxels into a
    /// bitmap of the cells (0-15 per axis, each `voxel_size` wide) of the chunk at `chunk_origin`
    /// that it passes through.
    ///
    /// The segment is first clipped to the chunk's box, so the traversal starts where the ray
    /// enters the chunk no matter how far away `start` is, and stops where it leaves the chunk
    /// or reaches `end`. When several axes cross a cell boundary at the same parameter (an
    /// edge/corner crossing) all of them advance together, so cells the ray only touches along an
    /// edge are not visited and the result is the same for a ray and its reverse.
    fn rasterize_ray_in_chunk(
        &self,
        start: WorldPos,
        end: WorldPos,
        chunk_origin: WorldPos,
        voxel_size: i64,
        bitmap: &mut Bitmap,
    ) {
        const EPS: f64 = 1e-10;
        let scale = 1.0 / voxel_size as f64;

        // Voxel-centre coordinates in this chunk's cell units; the chunk spans [0, 16)^3.
        let s = [
            ((start.x - chunk_origin.x) as f64 + 0.5) * scale,
            ((start.y - chunk_origin.y) as f64 + 0.5) * scale,
            ((start.z - chunk_origin.z) as f64 + 0.5) * scale,
        ];
        let e = [
            ((end.x - chunk_origin.x) as f64 + 0.5) * scale,
            ((end.y - chunk_origin.y) as f64 + 0.5) * scale,
            ((end.z - chunk_origin.z) as f64 + 0.5) * scale,
        ];
        let d = [e[0] - s[0], e[1] - s[1], e[2] - s[2]];

        // Clip the segment (t in [0, 1]) to the chunk box with the slab method.
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for a in 0..3 {
            if d[a].abs() < 1e-12 {
                if s[a] < 0.0 || s[a] >= 16.0 {
                    return;
                }
                continue;
            }
            let inv = 1.0 / d[a];
            let mut ta = (0.0 - s[a]) * inv;
            let mut tb = (16.0 - s[a]) * inv;
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return;
            }
        }
        if t1 - t0 <= EPS {
            return; // only touches the chunk at an edge or corner: no cell is entered
        }

        // Cell at the entry point (nudged into the interval to pick the cell being entered).
        let t_sample = t1.min(t0 + 1e-9);
        let mut cell = [0i32; 3];
        let mut step = [1i32; 3];
        let mut next = [f64::INFINITY; 3];
        let mut delta = [f64::INFINITY; 3];
        for a in 0..3 {
            cell[a] = ((s[a] + d[a] * t_sample).floor() as i32).clamp(0, 15);
            if d[a].abs() >= 1e-12 {
                step[a] = if d[a] < 0.0 { -1 } else { 1 };
                delta[a] = 1.0 / d[a].abs();
                let boundary = if step[a] > 0 { cell[a] + 1 } else { cell[a] } as f64;
                next[a] = (boundary - s[a]) / d[a];
            }
        }

        loop {
            bitmap.add(Chunk::flat_index(
                cell[0] as u8,
                cell[1] as u8,
                cell[2] as u8,
            ));

            let exit_t = t1.min(next[0].min(next[1]).min(next[2]));
            if exit_t >= t1 - EPS {
                break; // segment ends (or leaves the chunk) inside this cell
            }

            // Cross every axis tied at this boundary together.
            for a in 0..3 {
                if next[a] <= exit_t + EPS {
                    cell[a] += step[a];
                    next[a] += delta[a];
                }
            }
            if cell.iter().any(|&c| !(0..16).contains(&c)) {
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

        // Navigate to the leaf chunk (the root itself for a single-level world), creating it or
        // splitting a coarse Solid voxel on the way, then set the voxel in it.
        let &(x, y, z) = path.last().unwrap();
        self.navigate_to_mut(&path, self.hierarchy_depth as usize - 1)
            .set(x, y, z, voxel_type);
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
        Self::update_chunk_lod_recursive(root_mut, palette, false);
    }

    /// Update LOD metadata while preserving precomputed voxel_count/bounding_box.
    /// Useful after loading from file when counts/bounds are already cached.
    pub fn update_all_lod_metadata_preserve_bounds(&mut self, palette: &Palette) {
        let root_mut = Arc::make_mut(&mut self.root);
        Self::update_chunk_lod_recursive(root_mut, palette, true);
    }

    /// Recursive helper to update LOD metadata bottom-up
    fn update_chunk_lod_recursive(chunk: &mut Chunk, palette: &Palette, preserve_bounds: bool) {
        // First, recursively update all sub-chunks. Use Rayon to parallelize recursion across
        // different sub-chunks where possible - this gives a large speedup for deep/large worlds.
        use rayon::prelude::*;
        chunk.voxels.par_iter_mut().for_each(|voxel| {
            if let Voxel::Chunk(sub_chunk_arc) = voxel {
                // Use Arc::make_mut to get exclusive access for mutation
                let sub_chunk = Arc::make_mut(sub_chunk_arc);
                Self::update_chunk_lod_recursive(sub_chunk, palette, preserve_bounds);
            }
        });

        // Then update this chunk's metadata
        if preserve_bounds {
            chunk.update_lod_metadata_with_mask_preserve_bounds(palette, 0);
        } else {
            chunk.update_lod_metadata(palette);
        }
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
    pub fn update_metadata_at(
        &mut self,
        world_x: i64,
        world_y: i64,
        world_z: i64,
        palette: &Palette,
    ) {
        let mut path = Vec::new();
        let mut cur_x = world_x;
        let mut cur_y = world_y;
        let mut cur_z = world_z;

        // Trace the path from leaf up to root, then reverse to go root -> leaf
        for _ in 0..self.hierarchy_depth {
            path.push(((cur_x & 15) as u8, (cur_y & 15) as u8, (cur_z & 15) as u8));
            cur_x >>= 4;
            cur_y >>= 4;
            cur_z >>= 4;
        }
        path.reverse();

        let root_mut = Arc::make_mut(&mut self.root);
        Self::update_metadata_recursive(root_mut, &path, 0, palette);
    }

    fn update_metadata_recursive(
        chunk: &mut Chunk,
        path: &[(u8, u8, u8)],
        depth_idx: usize,
        palette: &Palette,
    ) {
        // If we are not at the leaf level, descend
        if depth_idx < path.len() - 1 {
            let (x, y, z) = path[depth_idx];
            let pos = (x as u16) | ((y as u16) << 4) | ((z as u16) << 8);

            if chunk.presence.contains(pos as u32) {
                let rank = chunk.presence.rank(pos as u32) as usize;
                if rank > 0 {
                    if let Voxel::Chunk(sub_chunk_arc) = &mut chunk.voxels[rank - 1] {
                        Self::update_metadata_recursive(
                            Arc::make_mut(sub_chunk_arc),
                            path,
                            depth_idx + 1,
                            palette,
                        );
                    }
                }
            }
        }

        // After updating children (or if we are the leaf), update this chunk's metadata
        chunk.update_lod_metadata(palette);
        // Refresh hierarchy shell for culling after edits
        chunk.generate_hierarchy_shell();
    }

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
pub(crate) mod test_util {
    //! Helpers shared by the tests of this crate.
    use super::*;

    /// Panic unless the two chunks are structurally identical (recursively): marginals,
    /// presence, voxels in rank order and the cached counts/bounds.
    pub(crate) fn assert_same_chunk(a: &Chunk, b: &Chunk, at: &str) {
        assert_eq!((a.px, a.py, a.pz), (b.px, b.py, b.pz), "marginals at {at}");
        assert!(a.presence == b.presence, "presence at {at}");
        assert_eq!(a.voxel_count, b.voxel_count, "voxel_count at {at}");
        assert_eq!(
            a.solid_ratio.to_bits(),
            b.solid_ratio.to_bits(),
            "solid_ratio at {at}"
        );
        assert_eq!(a.bounding_box, b.bounding_box, "bounding_box at {at}");
        assert_eq!(a.dominant_type, b.dominant_type, "dominant_type at {at}");
        assert_eq!(a.average_color, b.average_color, "average_color at {at}");
        assert_eq!(a.voxels.len(), b.voxels.len(), "voxel count at {at}");
        for (i, (va, vb)) in a.voxels.iter().zip(&b.voxels).enumerate() {
            let at = format!("{at}/{i}");
            match (va, vb) {
                (Voxel::Solid(ta), Voxel::Solid(tb)) => assert_eq!(ta, tb, "solid at {at}"),
                (Voxel::Chunk(ca), Voxel::Chunk(cb)) => assert_same_chunk(ca, cb, &at),
                _ => panic!("voxel kind differs at {at}"),
            }
        }
    }

    pub(crate) fn assert_same_world(a: &World, b: &World, what: &str) {
        assert_eq!(a.hierarchy_depth(), b.hierarchy_depth(), "{what}");
        assert_same_chunk(a.root(), b.root(), what);
    }
}

#[cfg(test)]
impl World {
    /// Test helper: store a uniform solid covering the `size`^3 cell at `origin` (`size` = 16^k
    /// with k >= 1, `origin` aligned to it) as a single coarse `Voxel::Solid` at the matching
    /// hierarchy level, the way a generator or a loaded .vhc file can.
    pub(crate) fn set_uniform_solid(&mut self, origin: [i64; 3], size: i64, t: VoxelType) {
        let k = size.trailing_zeros() as usize / 4;
        assert!(k >= 1 && size == 16i64.pow(k as u32) && k < self.hierarchy_depth as usize);
        let path = self
            .position_to_path(WorldPos::new(origin[0], origin[1], origin[2]))
            .unwrap();
        let level = self.hierarchy_depth as usize - 1 - k;
        let (x, y, z) = path[level];
        self.navigate_to_mut(&path, level).set(x, y, z, t);
    }
}

#[cfg(test)]
mod tests {
    use crate::palette::Palette;

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
    fn test_chunk_remove_keeps_marginals_of_other_voxels() {
        let mut chunk = Chunk::new();
        chunk.set(1, 1, 1, 1);
        chunk.set(2, 1, 1, 3); // shares the y and z slices with (1,1,1)
        chunk.set(1, 5, 5, 2); // shares the x slice with (1,1,1)
        chunk.remove(1, 1, 1);
        assert_eq!(chunk.get_type(2, 1, 1), Some(3));
        assert_eq!(chunk.get_type(1, 5, 5), Some(2));

        chunk.remove(2, 1, 1);
        assert_eq!(chunk.get_type(1, 5, 5), Some(2));
        assert_eq!((chunk.px, chunk.py, chunk.pz), (1 << 1, 1 << 5, 1 << 5));
    }

    #[test]
    fn test_chunk_remove_marginals_stay_exact() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(99);
        let mut chunk = Chunk::new();
        for step in 0..3000 {
            let (x, y, z) = (
                rng.gen_range(0..16),
                rng.gen_range(0..16),
                rng.gen_range(0..16),
            );
            if rng.gen_bool(0.6) {
                chunk.set(x, y, z, 1);
            } else {
                chunk.remove(x, y, z);
            }
            let (mut px, mut py, mut pz) = (0u16, 0u16, 0u16);
            for (vx, vy, vz) in chunk.positions() {
                px |= 1 << vx;
                py |= 1 << vy;
                pz |= 1 << vz;
            }
            assert_eq!((chunk.px, chunk.py, chunk.pz), (px, py, pz), "step {step}");
        }
    }

    #[test]
    fn test_occupied_bounds() {
        let mut world = World::new(2);
        assert_eq!(world.occupied_bounds(), None);

        world.set(WorldPos::new(10, 20, 30), 1);
        world.set(WorldPos::new(200, 5, 40), 2);

        assert_eq!(world.occupied_bounds(), Some(([10, 5, 30], [201, 21, 41])));
    }

    #[test]
    fn test_lod_metadata_prefers_reflective_surface_type() {
        let palette = Palette::from_string(
            "\
0 255 255 255 255 0 0 0 0 0
1 50 50 50 255 0 0 0 0 0
2 100 120 140 255 0 0 0 0 220
",
        )
        .unwrap();
        let mut chunk = Chunk::new();

        chunk.set(0, 0, 0, 1);
        chunk.set(1, 0, 0, 1);
        chunk.set(2, 0, 0, 1);
        chunk.set(3, 0, 0, 2);
        chunk.set(4, 0, 0, 2);

        chunk.update_lod_metadata(&palette);

        assert_eq!(chunk.dominant_type, 2);
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

#[cfg(test)]
mod line_of_sight_tests {
    //! Regression tests for `World::line_of_sight`, ported from the Swift
    //! `VHCHierarchicalLineOfSightTests` in the eyetheisles project. The API takes integer voxel
    //! coordinates and the segment runs between the voxel centres.
    use super::*;

    fn p(x: i64, y: i64, z: i64) -> WorldPos {
        WorldPos::new(x, y, z)
    }

    fn world_with(depth: u8, solids: &[(i64, i64, i64)]) -> World {
        let mut w = World::new(depth);
        for &(x, y, z) in solids {
            w.set(p(x, y, z), 1);
        }
        w
    }

    #[test]
    fn readme_repro_blocker_in_second_child() {
        // depth 2: root cell (1,0,0) is the child beginning at (16,0,0).
        let w = world_with(2, &[(21, 5, 5)]);
        assert_eq!(w.get(p(21, 5, 5)), Some(1));
        assert!(
            !w.line_of_sight(p(0, 5, 5), p(40, 5, 5)),
            "blocker at (21,5,5) ignored"
        );
    }

    #[test]
    fn empty_segment_visible() {
        let w = world_with(1, &[]);
        assert!(w.line_of_sight(p(0, 5, 5), p(12, 5, 5)));
    }

    #[test]
    fn intervening_voxel_blocks_depth1() {
        let w = world_with(1, &[(5, 5, 5)]);
        assert!(!w.line_of_sight(p(0, 5, 5), p(12, 5, 5)));
    }

    #[test]
    fn occupied_endpoint_is_excluded() {
        // The emitter voxel at `end` must not occlude its own ray (swift: excludingEndVoxel).
        let w = world_with(1, &[(5, 5, 5)]);
        assert!(w.line_of_sight(p(0, 5, 5), p(5, 5, 5)));
    }

    #[test]
    fn occupied_endpoint_in_other_child_is_excluded() {
        let w = world_with(2, &[(40, 5, 5)]);
        assert!(w.line_of_sight(p(0, 5, 5), p(40, 5, 5)));
        // ...but a wall in between still blocks.
        let w = world_with(2, &[(40, 5, 5), (30, 5, 5)]);
        assert!(!w.line_of_sight(p(0, 5, 5), p(40, 5, 5)));
    }

    /// Exact reference: segment between voxel centres vs. every solid box `(lo, size)`, excluding
    /// a unit box equal to the end voxel. A box blocks if the segment spends positive length
    /// inside it.
    fn reference_visible_boxes(boxes: &[([i64; 3], i64)], a: WorldPos, b: WorldPos) -> bool {
        let s = [a.x as f64 + 0.5, a.y as f64 + 0.5, a.z as f64 + 0.5];
        let d = [(b.x - a.x) as f64, (b.y - a.y) as f64, (b.z - a.z) as f64];
        for &(lo, size) in boxes {
            if size == 1 && lo == [b.x, b.y, b.z] {
                continue;
            }
            let (mut t0, mut t1) = (0.0f64, 1.0f64);
            let mut hit = true;
            for ax in 0..3 {
                let lo_a = lo[ax] as f64;
                let hi_a = (lo[ax] + size) as f64;
                if d[ax] == 0.0 {
                    if s[ax] <= lo_a || s[ax] >= hi_a {
                        hit = false;
                        break;
                    }
                    continue;
                }
                let mut ta = (lo_a - s[ax]) / d[ax];
                let mut tb = (hi_a - s[ax]) / d[ax];
                if ta > tb {
                    std::mem::swap(&mut ta, &mut tb);
                }
                t0 = t0.max(ta);
                t1 = t1.min(tb);
            }
            if hit && t1 - t0 > 1e-7 {
                return false;
            }
        }
        true
    }

    fn reference_visible(solids: &[(i64, i64, i64)], a: WorldPos, b: WorldPos) -> bool {
        let boxes: Vec<([i64; 3], i64)> = solids.iter().map(|&(x, y, z)| ([x, y, z], 1)).collect();
        reference_visible_boxes(&boxes, a, b)
    }

    #[test]
    fn randomized_matches_exact_reference() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut mismatches = 0;
        let mut missed_blockers = 0; // reported visible, reference says blocked
        let mut false_blocks = 0; // reported blocked, reference says visible
        let mut blocked = 0;
        let mut total = 0;
        for trial in 0..30 {
            let depth = if trial % 2 == 0 { 2 } else { 3 };
            // solids scattered over a 70^3 region that straddles several 16^3 children
            let n = 800 + (trial % 5) * 800;
            let solids: Vec<(i64, i64, i64)> = (0..n)
                .map(|_| {
                    (
                        rng.gen_range(0..70),
                        rng.gen_range(0..70),
                        rng.gen_range(0..70),
                    )
                })
                .collect();
            let w = world_with(depth, &solids);
            for _ in 0..300 {
                let a = p(
                    rng.gen_range(0..70),
                    rng.gen_range(0..70),
                    rng.gen_range(0..70),
                );
                let b = p(
                    rng.gen_range(0..70),
                    rng.gen_range(0..70),
                    rng.gen_range(0..70),
                );
                if a == b {
                    continue;
                }
                let got = w.line_of_sight(a, b);
                let want = reference_visible(&solids, a, b);
                total += 1;
                if !want {
                    blocked += 1;
                }
                if got != want {
                    mismatches += 1;
                    if got {
                        missed_blockers += 1;
                    } else {
                        false_blocks += 1;
                    }
                    if mismatches <= 10 {
                        println!("mismatch depth {depth} {a:?}->{b:?}: got {got}, want {want}");
                    }
                }
            }
        }
        println!(
            "rays {total}, reference-blocked {blocked}, mismatches {mismatches} \
             (missed blockers {missed_blockers}, false blocks {false_blocks})"
        );
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn wall_before_emitter_still_blocks() {
        let w = world_with(1, &[(4, 5, 5), (5, 5, 5)]);
        assert!(!w.line_of_sight(p(0, 5, 5), p(5, 5, 5)));
    }

    #[test]
    fn neighbouring_child_start_distance_sweep() {
        // blocker in child (1,0,0) at x=21; vary how far the start is from the child.
        let w = world_with(2, &[(21, 5, 5)]);
        for sx in [0i64, 8, 12, 13, 14, 15, 16, 17, 20] {
            let vis = w.line_of_sight(p(sx, 5, 5), p(40, 5, 5));
            println!("start x={sx:>2} -> end x=40, blocker x=21 : visible={vis}");
        }
        // All of these must be blocked.
        for sx in [0i64, 8, 12, 13, 14, 15, 16, 17, 20] {
            assert!(!w.line_of_sight(p(sx, 5, 5), p(40, 5, 5)), "start x={sx}");
        }
    }

    #[test]
    fn blocker_in_far_child_negative_direction() {
        let w = world_with(2, &[(21, 5, 5)]);
        assert!(!w.line_of_sight(p(40, 5, 5), p(0, 5, 5)));
    }

    #[test]
    fn blocker_in_middle_child() {
        let w = world_with(2, &[(40, 5, 5)]);
        assert!(!w.line_of_sight(p(0, 5, 5), p(60, 5, 5)));
    }

    #[test]
    fn blocker_in_end_child() {
        let w = world_with(2, &[(50, 5, 5)]);
        assert!(!w.line_of_sight(p(0, 5, 5), p(60, 5, 5)));
    }

    #[test]
    fn blocker_in_start_child_works() {
        // control: blocker in the same leaf chunk as the start
        let w = world_with(2, &[(5, 5, 5)]);
        assert!(!w.line_of_sight(p(0, 5, 5), p(40, 5, 5)));
    }

    #[test]
    fn depth3_long_ray() {
        let w = world_with(3, &[(1000, 7, 7)]);
        assert!(!w.line_of_sight(p(4, 7, 7), p(2000, 7, 7)));
    }

    #[test]
    fn enter_from_outside_world_positive() {
        let w = world_with(1, &[(5, 5, 5)]);
        assert!(!w.line_of_sight(p(-32, 5, 5), p(12, 5, 5)));
    }

    #[test]
    fn enter_from_outside_world_negative() {
        let w = world_with(1, &[(5, 5, 5)]);
        assert!(!w.line_of_sight(p(32, 5, 5), p(0, 5, 5)));
    }

    #[test]
    fn diagonal_blocker_on_ray_forward() {
        let w = world_with(1, &[(5, 5, 5)]);
        assert!(!w.line_of_sight(p(0, 0, 0), p(12, 12, 12)));
    }

    #[test]
    fn diagonal_blocker_on_ray_reverse() {
        let w = world_with(1, &[(5, 5, 5)]);
        assert!(!w.line_of_sight(p(12, 12, 12), p(0, 0, 0)));
    }

    #[test]
    fn diagonal_beside_ray_forward() {
        let w = world_with(1, &[(5, 4, 4)]);
        assert!(w.line_of_sight(p(0, 0, 0), p(12, 12, 12)));
    }

    #[test]
    fn diagonal_beside_ray_reverse() {
        let w = world_with(1, &[(5, 4, 4)]);
        assert!(w.line_of_sight(p(12, 12, 12), p(0, 0, 0)));
    }

    #[test]
    fn uniform_coarse_solid_blocks() {
        let mut w = World::new(2);
        Arc::make_mut(&mut w.root).set(1, 0, 0, 1); // 16^3 uniform solid at coarse level
        assert!(!w.line_of_sight(p(0, 5, 5), p(40, 5, 5)));
    }

    #[test]
    fn wall_behind_endpoint_does_not_block() {
        // Overshoot check: wall lies BEYOND the end point along the ray.
        let w = world_with(1, &[(14, 5, 5)]);
        assert!(
            w.line_of_sight(p(0, 5, 5), p(10, 5, 5)),
            "wall behind the end point blocked the ray"
        );
    }

    #[test]
    #[ignore]
    fn los_perf_probe_like_rays() {
        // gi.rs-like workload: rays up to 64 voxels long in a sparse 256^3 region.
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(7);
        let mut w = World::new(3);
        for _ in 0..60_000 {
            // clustered "city": dense columns on a plane + scattered blocks
            let x = rng.gen_range(0..256);
            let z = rng.gen_range(0..256);
            let h = rng.gen_range(1..20);
            for y in 0..h {
                w.set(p(x, y, z), 1);
            }
        }
        let n = 200_000;
        let rays: Vec<(WorldPos, WorldPos)> = (0..n)
            .map(|_| {
                let a = p(
                    rng.gen_range(32..224),
                    rng.gen_range(2..24),
                    rng.gen_range(32..224),
                );
                let b = p(
                    a.x + rng.gen_range(-40..40),
                    (a.y + rng.gen_range(-12..12)).max(0),
                    a.z + rng.gen_range(-40..40),
                );
                (a, b)
            })
            .collect();
        let t = std::time::Instant::now();
        let mut vis = 0usize;
        for (a, b) in &rays {
            if w.line_of_sight(*a, *b) {
                vis += 1;
            }
        }
        let el = t.elapsed();
        println!(
            "PERF {n} rays: {:?} ({:.0} ns/ray), visible {:.1}%",
            el,
            el.as_nanos() as f64 / n as f64,
            100.0 * vis as f64 / n as f64
        );
    }

    #[test]
    fn solid_emitter_self_block_survey() {
        // gi.rs passes end = emitter voxel index (floor(light centre)); how often is an
        // otherwise-clear ray to a lone solid emitter reported visible?
        let w = world_with(1, &[(10, 8, 8)]);
        let mut vis = 0;
        let mut tot = 0;
        for dz in -8i64..=0 {
            for dy in -8i64..=0 {
                for dx in -8i64..=-1 {
                    let s = p(10 + dx, 8 + dy, 8 + dz);
                    tot += 1;
                    if w.line_of_sight(s, p(10, 8, 8)) {
                        vis += 1;
                    }
                }
            }
        }
        println!("lone emitter, clear rays: visible {vis}/{tot}");
        assert_eq!(vis, tot, "a lone emitter must not block its own rays");
    }
    #[test]
    fn start_voxel_blocks() {
        // gi.rs rejects solid start voxels itself, but the function must not silently ignore one.
        let w = world_with(1, &[(0, 5, 5)]);
        assert!(!w.line_of_sight(p(0, 5, 5), p(8, 5, 5)));
    }

    #[test]
    fn adjacent_voxels_across_a_chunk_boundary() {
        let w = world_with(2, &[(15, 5, 5), (16, 5, 5)]);
        // start itself is solid -> blocked
        assert!(!w.line_of_sight(p(15, 5, 5), p(16, 5, 5)));
        // end voxel (solid) is excluded, nothing else in between
        assert!(w.line_of_sight(p(14, 5, 5), p(15, 5, 5)));
        assert!(w.line_of_sight(p(17, 5, 5), p(16, 5, 5)));
        // but the solid at x=15 is crossed on the way to x=16
        assert!(!w.line_of_sight(p(14, 5, 5), p(16, 5, 5)));
    }

    #[test]
    fn rays_outside_the_world() {
        let w = world_with(1, &[(5, 5, 5)]);
        // entirely outside, never touching the world
        assert!(w.line_of_sight(p(-40, 5, 5), p(-10, 5, 5)));
        assert!(w.line_of_sight(p(-40, 40, 5), p(40, -40, 5)));
        // passes through the blocker from one side of the world to the other
        assert!(!w.line_of_sight(p(-20, 5, 5), p(30, 5, 5)));
        // far-away endpoints must not hang or overflow
        assert!(w.line_of_sight(p(-1_000_000, 5, 5), p(-999_000, 5, 5)));
        assert!(!w.line_of_sight(p(-1_000_000, 5, 5), p(1_000_000, 5, 5)));
    }

    #[test]
    fn edge_touching_cells_do_not_block() {
        // z=5 plane, ray (0,0)->(4,4) passes exactly through the cell corners (1,1), (2,2), (3,3).
        let beside: [&[(i64, i64, i64)]; 2] = [
            &[
                (1, 0, 5),
                (0, 1, 5),
                (2, 1, 5),
                (1, 2, 5),
                (3, 2, 5),
                (2, 3, 5),
            ],
            &[(1, 0, 5), (2, 0, 5), (3, 0, 5), (0, 3, 5), (0, 2, 5)],
        ];
        for blockers in beside {
            let w = world_with(1, blockers);
            assert!(w.line_of_sight(p(0, 0, 5), p(4, 4, 5)), "{blockers:?} fwd");
            assert!(w.line_of_sight(p(4, 4, 5), p(0, 0, 5)), "{blockers:?} rev");
        }
        for on_ray in [(1, 1, 5), (2, 2, 5), (3, 3, 5)] {
            let w = world_with(1, &[on_ray]);
            assert!(!w.line_of_sight(p(0, 0, 5), p(4, 4, 5)), "{on_ray:?} fwd");
            assert!(!w.line_of_sight(p(4, 4, 5), p(0, 0, 5)), "{on_ray:?} rev");
        }
    }

    #[test]
    fn corner_touching_a_chunk_does_not_block() {
        // The segment through (14,17) and (17,14) in the xy plane passes through the shared
        // corner of four leaf chunks at (16,16). Cells that only touch the corner must not block.
        let w = world_with(2, &[(15, 15, 5), (17, 17, 5), (16, 16, 5), (15, 17, 5)]);
        assert!(w.line_of_sight(p(14, 17, 5), p(17, 14, 5)));
        assert!(w.line_of_sight(p(17, 14, 5), p(14, 17, 5)));
        // A solid right on the ray does block.
        let w = world_with(2, &[(15, 16, 5)]);
        assert!(!w.line_of_sight(p(14, 17, 5), p(17, 14, 5)));
    }

    #[test]
    fn coarse_solid_blocks_only_when_crossed() {
        let mut w = World::new(3);
        w.set_uniform_solid([32, 0, 0], 16, 1); // middle-level 16^3 solid at x 32..48
        assert!(!w.line_of_sight(p(0, 5, 5), p(100, 5, 5)));
        assert!(!w.line_of_sight(p(100, 5, 5), p(0, 5, 5)));
        assert!(w.line_of_sight(p(0, 5, 5), p(31, 5, 5)));
        assert!(w.line_of_sight(p(48, 5, 5), p(100, 5, 5)));
        assert!(w.line_of_sight(p(0, 20, 5), p(100, 20, 5)));
        // an end voxel inside a coarse solid is not excluded
        assert!(!w.line_of_sight(p(0, 5, 5), p(40, 5, 5)));
        // root-level coarse solid (256^3) at depth 3
        let mut w = World::new(3);
        w.set_uniform_solid([256, 0, 0], 256, 2);
        assert!(!w.line_of_sight(p(0, 5, 5), p(600, 5, 5)));
        assert!(w.line_of_sight(p(0, 300, 5), p(600, 300, 5)));
        assert!(w.line_of_sight(p(0, 5, 5), p(255, 5, 5)));
    }

    #[test]
    fn randomized_dense_chunk_matches_exact_reference() {
        // Small coordinate range, high density: lots of ties and edge/corner crossings.
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0xD15EA5E);
        for density in [0.05f64, 0.2, 0.5] {
            let mut solids = Vec::new();
            for z in 0..16 {
                for y in 0..16 {
                    for x in 0..16 {
                        if rng.gen_bool(density) {
                            solids.push((x, y, z));
                        }
                    }
                }
            }
            let w = world_with(1, &solids);
            for _ in 0..4000 {
                let a = p(
                    rng.gen_range(-4..20),
                    rng.gen_range(-4..20),
                    rng.gen_range(-4..20),
                );
                let b = p(
                    rng.gen_range(-4..20),
                    rng.gen_range(-4..20),
                    rng.gen_range(-4..20),
                );
                if a == b {
                    continue;
                }
                assert_eq!(
                    w.line_of_sight(a, b),
                    reference_visible(&solids, a, b),
                    "density {density} {a:?}->{b:?}"
                );
            }
        }
    }

    #[test]
    fn randomized_is_symmetric_for_empty_endpoints() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0x5EED);
        let solids: Vec<(i64, i64, i64)> = (0..1500)
            .map(|_| {
                (
                    rng.gen_range(0..40),
                    rng.gen_range(0..40),
                    rng.gen_range(0..40),
                )
            })
            .collect();
        let w = world_with(2, &solids);
        let mut checked = 0;
        for _ in 0..3000 {
            let a = p(
                rng.gen_range(0..40),
                rng.gen_range(0..40),
                rng.gen_range(0..40),
            );
            let b = p(
                rng.gen_range(0..40),
                rng.gen_range(0..40),
                rng.gen_range(0..40),
            );
            if a == b || w.get(a).is_some() || w.get(b).is_some() {
                continue;
            }
            assert_eq!(
                w.line_of_sight(a, b),
                w.line_of_sight(b, a),
                "{a:?} <-> {b:?}"
            );
            checked += 1;
        }
        assert!(checked > 1000);
    }

    #[test]
    fn randomized_with_coarse_solids_matches_exact_reference() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0xC0A25E);
        for trial in 0..12 {
            let mut w = World::new(3);
            let mut boxes: Vec<([i64; 3], i64)> = Vec::new();
            // a few middle-level (16^3) and root-level (256^3) uniform solids
            for _ in 0..3 {
                let o = [
                    rng.gen_range(0..40) * 16,
                    rng.gen_range(0..4) * 16,
                    rng.gen_range(0..40) * 16,
                ];
                w.set_uniform_solid(o, 16, 1);
                boxes.push((o, 16));
            }
            if trial % 4 == 0 {
                let o = [rng.gen_range(1..4) * 256, 0, rng.gen_range(1..4) * 256];
                w.set_uniform_solid(o, 256, 2);
                boxes.push((o, 256));
            }
            // plus scattered unit voxels outside the coarse boxes
            for _ in 0..1500 {
                let v = [
                    rng.gen_range(0..700i64),
                    rng.gen_range(0..64i64),
                    rng.gen_range(0..700i64),
                ];
                let inside = boxes
                    .iter()
                    .any(|&(lo, sz)| (0..3).all(|a| v[a] >= lo[a] && v[a] < lo[a] + sz));
                if !inside {
                    w.set(p(v[0], v[1], v[2]), 3);
                    boxes.push((v, 1));
                }
            }
            for _ in 0..400 {
                let a = p(
                    rng.gen_range(0..700),
                    rng.gen_range(0..64),
                    rng.gen_range(0..700),
                );
                let b = p(
                    rng.gen_range(0..700),
                    rng.gen_range(0..64),
                    rng.gen_range(0..700),
                );
                if a == b {
                    continue;
                }
                assert_eq!(
                    w.line_of_sight(a, b),
                    reference_visible_boxes(&boxes, a, b),
                    "trial {trial} {a:?}->{b:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod cow_tests {
    //! `World::set`/`remove`/`subdivide_at`/`merge_at` must give the same structure as the
    //! original implementation, which cloned the child `Voxel` (and thereby its `Arc`) before
    //! `Arc::make_mut` and so deep-copied the leaf and middle chunk on every edit.
    use super::test_util::assert_same_world;
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn p(x: i64, y: i64, z: i64) -> WorldPos {
        WorldPos::new(x, y, z)
    }

    // ---- Reference: the original algorithm, kept verbatim (modulo `self` -> `world`). ----

    fn old_navigate_to_mut<'a>(
        world: &'a mut World,
        path: &[(u8, u8, u8)],
        depth: usize,
    ) -> &'a mut Chunk {
        let mut current = Arc::make_mut(&mut world.root);

        for &(x, y, z) in &path[..depth] {
            let idx = Chunk::flat_index(x, y, z);

            // Check if voxel exists and what type it is
            let existing_voxel = if current.presence.contains(idx) {
                let rank = current.presence.rank(idx) as usize;
                Some(current.voxels[rank - 1].clone())
            } else {
                None
            };

            let needs_chunk = match &existing_voxel {
                Some(Voxel::Chunk(_)) => false,
                _ => true,
            };

            // Create or ensure it's a chunk
            if needs_chunk {
                let new_chunk = match existing_voxel {
                    Some(Voxel::Solid(t)) => Chunk::full(t),
                    _ => Chunk::new(),
                };
                current.set_chunk(x, y, z, new_chunk);
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

    fn old_set(world: &mut World, pos: WorldPos, voxel_type: VoxelType) {
        let path = match world.position_to_path(pos) {
            Ok(p) => p,
            Err(_) => return, // Out of bounds, silently ignore
        };

        let depth = world.hierarchy_depth as usize;

        if depth == 1 {
            // Special case: single-level world, root IS the leaf chunk
            let &(x, y, z) = path.last().unwrap();
            Arc::make_mut(&mut world.root).set(x, y, z, voxel_type);
            return;
        }

        // Navigate to the "grandparent" level (one above the leaf chunk level)
        let grandparent = old_navigate_to_mut(world, &path, depth - 2);

        // Ensure the leaf chunk exists at path[depth-2]
        let &(lx, ly, lz) = &path[depth - 2];
        let idx = Chunk::flat_index(lx, ly, lz);

        // Check if we need to create or replace with a chunk
        let existing_voxel = if grandparent.presence.contains(idx) {
            let rank = grandparent.presence.rank(idx) as usize;
            Some(grandparent.voxels[rank - 1].clone())
        } else {
            None
        };

        let needs_chunk = match &existing_voxel {
            Some(Voxel::Chunk(_)) => false,
            _ => true,
        };

        if needs_chunk {
            // Create the leaf chunk (splitting a Solid voxel if it exists)
            let new_chunk = match existing_voxel {
                Some(Voxel::Solid(t)) => Chunk::full(t),
                _ => Chunk::new(),
            };
            grandparent.set_chunk(lx, ly, lz, new_chunk);
        }

        // Now get the leaf chunk and set the voxel in it
        let rank = grandparent.presence.rank(idx) as usize;
        if let Voxel::Chunk(leaf_chunk_arc) = &mut grandparent.voxels[rank - 1] {
            let &(x, y, z) = path.last().unwrap();
            Arc::make_mut(leaf_chunk_arc).set(x, y, z, voxel_type);
        }
    }

    fn old_remove(world: &mut World, pos: WorldPos) {
        let path = match world.position_to_path(pos) {
            Ok(p) => p,
            Err(_) => return, // Out of bounds
        };
        let depth = world.hierarchy_depth as usize;
        let parent = old_navigate_to_mut(world, &path, depth - 1);
        let &(x, y, z) = path.last().unwrap();
        parent.remove(x, y, z);
    }

    fn old_subdivide_at(world: &mut World, pos: WorldPos) -> Result<(), &'static str> {
        let path = world.position_to_path(pos)?;
        let depth = world.hierarchy_depth as usize;
        let parent = old_navigate_to_mut(world, &path, depth - 1);
        let &(x, y, z) = path.last().ok_or("Invalid path")?;
        parent.subdivide(x, y, z)
    }

    fn old_merge_at(world: &mut World, pos: WorldPos) -> Result<bool, &'static str> {
        let path = world.position_to_path(pos)?;
        let depth = world.hierarchy_depth as usize;
        let parent = old_navigate_to_mut(world, &path, depth - 1);
        let &(x, y, z) = path.last().ok_or("Invalid path")?;
        parent.try_merge(x, y, z)
    }

    /// Run the same pseudo-random edit sequence through the current implementation and the
    /// original one and compare the resulting worlds.
    fn run_equivalence(depth: u8, seed: u64, ops: usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let size = 16i64.pow(depth as u32);
        // Edits concentrate in a small region so leaf chunks fill up, with some far-flung ones.
        let near = size.min(40);
        let pick = |rng: &mut StdRng| {
            if rng.gen_ratio(1, 8) {
                p(
                    rng.gen_range(0..size),
                    rng.gen_range(0..size),
                    rng.gen_range(0..size),
                )
            } else {
                p(
                    rng.gen_range(0..near),
                    rng.gen_range(0..near),
                    rng.gen_range(0..near),
                )
            }
        };

        // Base world with coarse uniform solids at the middle and root levels, so that edits
        // also split Solid voxels at several depths.
        let mut base = World::new(depth);
        if depth >= 2 {
            for k in 1..depth as u32 {
                let sz = 16i64.pow(k);
                for _ in 0..3 {
                    let o = [
                        rng.gen_range(0..(near / sz).max(1)) * sz,
                        rng.gen_range(0..(near / sz).max(1)) * sz,
                        rng.gen_range(0..(near / sz).max(1)) * sz,
                    ];
                    base.set_uniform_solid(o, sz, rng.gen_range(1..5));
                }
            }
        }
        let mut new_world = base.clone();
        let mut old_world = base;

        for i in 0..ops {
            let pos = pick(&mut rng);
            match rng.gen_range(0..100) {
                0..=54 => {
                    let t = rng.gen_range(1..7);
                    new_world.set(pos, t);
                    old_set(&mut old_world, pos, t);
                }
                55..=59 => {
                    new_world.set(pos, 0);
                    old_set(&mut old_world, pos, 0);
                }
                60..=84 => {
                    new_world.remove(pos);
                    old_remove(&mut old_world, pos);
                }
                85..=91 => {
                    let r_new = new_world.subdivide_at(pos);
                    let r_old = old_subdivide_at(&mut old_world, pos);
                    assert_eq!(r_new, r_old, "subdivide_at result, op {i}");
                }
                92..=96 => {
                    let r_new = new_world.merge_at(pos);
                    let r_old = old_merge_at(&mut old_world, pos);
                    assert_eq!(r_new, r_old, "merge_at result, op {i}");
                }
                _ => {
                    // out of bounds: silently ignored
                    let oob = p(-1 - rng.gen_range(0..5), pos.y, pos.z);
                    new_world.set(oob, 1);
                    old_set(&mut old_world, oob, 1);
                    new_world.remove(p(size, 0, 0));
                    old_remove(&mut old_world, p(size, 0, 0));
                }
            }
            if i % 500 == 499 {
                assert_same_world(&new_world, &old_world, &format!("depth {depth} op {i}"));
            }
        }
        assert_same_world(&new_world, &old_world, &format!("depth {depth} final"));
        assert_eq!(new_world.count(), old_world.count());
    }

    #[test]
    fn edits_match_original_algorithm_depth1() {
        run_equivalence(1, 11, 3000);
    }

    #[test]
    fn edits_match_original_algorithm_depth2() {
        run_equivalence(2, 22, 4000);
    }

    #[test]
    fn edits_match_original_algorithm_depth3() {
        run_equivalence(3, 33, 3000);
    }

    #[test]
    fn edits_match_original_algorithm_depth4() {
        run_equivalence(4, 44, 2000);
    }

    #[test]
    fn splitting_coarse_solids_keeps_their_type() {
        let mut w = World::new(3);
        w.set_uniform_solid([0, 0, 0], 256, 4); // root-level solid
        w.set(p(5, 6, 7), 9);
        assert_eq!(w.get(p(5, 6, 7)), Some(9));
        assert_eq!(w.get(p(5, 6, 8)), Some(4));
        assert_eq!(w.get(p(255, 255, 255)), Some(4));
        w.remove(p(100, 100, 100));
        assert_eq!(w.get(p(100, 100, 100)), None);
        assert_eq!(w.get(p(100, 100, 101)), Some(4));
    }

    fn leaf_ptr(w: &World, origin: WorldPos) -> *const Chunk {
        w.get_leaf_chunk_at_origin(origin).unwrap() as *const Chunk
    }

    fn middle_ptr(w: &World, pos: WorldPos) -> *const Chunk {
        let path = w.position_to_path(pos).unwrap();
        w.navigate_to(&path, 1).unwrap() as *const Chunk
    }

    #[test]
    fn unshared_chunks_are_edited_in_place() {
        let mut w = World::new(3);
        w.set(p(1, 1, 1), 1);
        let (leaf, mid) = (leaf_ptr(&w, p(0, 0, 0)), middle_ptr(&w, p(1, 1, 1)));
        for i in 2..10 {
            w.set(p(i, 1, 1), 2);
            w.remove(p(i - 1, 1, 1));
        }
        assert_eq!(leaf_ptr(&w, p(0, 0, 0)), leaf, "leaf chunk was reallocated");
        assert_eq!(
            middle_ptr(&w, p(1, 1, 1)),
            mid,
            "middle chunk was reallocated"
        );

        // The original algorithm reallocated both on every edit; make sure this check can tell.
        let mut old = World::new(3);
        old_set(&mut old, p(1, 1, 1), 1);
        let (leaf, mid) = (leaf_ptr(&old, p(0, 0, 0)), middle_ptr(&old, p(1, 1, 1)));
        old_set(&mut old, p(2, 1, 1), 2);
        assert!(
            leaf_ptr(&old, p(0, 0, 0)) != leaf || middle_ptr(&old, p(1, 1, 1)) != mid,
            "reference implementation no longer copies; test is vacuous"
        );
    }

    #[test]
    fn shared_chunks_are_copied_on_write() {
        let mut w = World::new(3);
        w.set(p(1, 1, 1), 1);
        w.set(p(40, 1, 1), 2); // a second leaf chunk
        let snapshot = w.clone();
        let (leaf_a, leaf_b) = (
            leaf_ptr(&snapshot, p(0, 0, 0)),
            leaf_ptr(&snapshot, p(32, 0, 0)),
        );

        w.set(p(2, 1, 1), 3);
        w.remove(p(1, 1, 1));

        // Snapshot untouched, edited world diverged
        assert_eq!(snapshot.get(p(1, 1, 1)), Some(1));
        assert_eq!(snapshot.get(p(2, 1, 1)), None);
        assert_eq!(w.get(p(1, 1, 1)), None);
        assert_eq!(w.get(p(2, 1, 1)), Some(3));
        assert_eq!(leaf_ptr(&snapshot, p(0, 0, 0)), leaf_a);
        assert!(leaf_ptr(&w, p(0, 0, 0)) != leaf_a);
        // The sibling leaf chunk that was not edited is still shared with the snapshot
        assert_eq!(leaf_ptr(&w, p(32, 0, 0)), leaf_b);
        assert_eq!(leaf_ptr(&snapshot, p(32, 0, 0)), leaf_b);
    }

    #[test]
    #[ignore]
    fn bench_set_old_vs_new() {
        // Run with: cargo test --release --lib bench_set_old_vs_new -- --ignored --nocapture
        fn fill(w: &mut World, nx: i64, ny: i64, nz: i64, f: fn(&mut World, WorldPos, VoxelType)) {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        f(w, p(x, y, z), 1 + ((x + y + z) % 5) as u8);
                    }
                }
            }
        }
        let t = std::time::Instant::now();
        let mut old = World::new(3);
        fill(&mut old, 128, 128, 64, old_set); // 1,048,576 sets
        let old_t = t.elapsed();

        let t = std::time::Instant::now();
        let mut new = World::new(3);
        fill(&mut new, 128, 128, 64, |w, pos, v| w.set(pos, v));
        let new_t = t.elapsed();
        assert_same_world(&new, &old, "bench world");
        println!(
            "BENCH 1.05M sets: old {:?} ({:.0} ns/set), new {:?} ({:.0} ns/set), {:.1}x",
            old_t,
            old_t.as_nanos() as f64 / 1_048_576.0,
            new_t,
            new_t.as_nanos() as f64 / 1_048_576.0,
            old_t.as_secs_f64() / new_t.as_secs_f64()
        );

        let t = std::time::Instant::now();
        let mut new = World::new(3);
        fill(&mut new, 256, 256, 64, |w, pos, v| w.set(pos, v)); // 4,194,304 sets
        println!("BENCH 4.19M sets: new {:?}", t.elapsed());
    }
}

#[cfg(test)]
mod coarse_solid_tests {
    //! A uniform `Voxel::Solid` stored above the leaf level (a "coarse" solid covering a whole
    //! sub-chunk) must answer lookups with its type at every voxel inside it, like a subdivided
    //! chunk of the same type would.
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::collections::HashMap;

    fn p(x: i64, y: i64, z: i64) -> WorldPos {
        WorldPos::new(x, y, z)
    }

    #[test]
    fn get_inside_middle_level_solid() {
        let mut w = World::new(3);
        w.set_uniform_solid([32, 16, 0], 16, 7);
        for pos in [
            p(32, 16, 0),
            p(47, 31, 15),
            p(40, 20, 5),
            p(32, 31, 0),
            p(47, 16, 15),
        ] {
            assert_eq!(w.get(pos), Some(7), "{pos:?}");
        }
        for pos in [
            p(31, 16, 0),
            p(48, 16, 0),
            p(32, 15, 0),
            p(32, 32, 0),
            p(32, 16, 16),
            p(32, 16, -1),
        ] {
            assert_eq!(w.get(pos), None, "{pos:?}");
        }
    }

    #[test]
    fn get_inside_root_level_solid() {
        let mut w = World::new(3);
        w.set_uniform_solid([256, 0, 512], 256, 3);
        for pos in [
            p(256, 0, 512),
            p(511, 255, 767),
            p(300, 17, 600),
            p(256, 255, 767),
        ] {
            assert_eq!(w.get(pos), Some(3), "{pos:?}");
        }
        for pos in [
            p(255, 0, 512),
            p(512, 0, 512),
            p(256, 256, 512),
            p(256, 0, 511),
        ] {
            assert_eq!(w.get(pos), None, "{pos:?}");
        }

        // depth 2: a root-level solid is a 16^3 block
        let mut w = World::new(2);
        w.set_uniform_solid([16, 32, 48], 16, 5);
        assert_eq!(w.get(p(31, 47, 63)), Some(5));
        assert_eq!(w.get(p(32, 47, 63)), None);
    }

    #[test]
    fn leaf_chunk_lookup_expands_uniform_solids() {
        let mut w = World::new(3);
        w.set_uniform_solid([32, 16, 0], 16, 7); // one leaf-sized cell
        w.set_uniform_solid([256, 0, 512], 256, 3); // 16x16x16 leaf chunks
        w.set(p(1, 1, 1), 1); // an ordinary subdivided leaf chunk

        let origins = [
            (p(32, 16, 0), 7),
            (p(256, 0, 512), 3),
            (p(256 + 16 * 5, 16 * 3, 512 + 16 * 15), 3),
            (p(256 + 240, 240, 512 + 240), 3),
        ];
        for (origin, t) in origins {
            let chunk = w
                .get_leaf_chunk_at_origin(origin)
                .unwrap_or_else(|| panic!("no chunk at {origin:?}"));
            assert_eq!(chunk.count(), 4096, "{origin:?}");
            assert!(
                chunk
                    .iter()
                    .all(|(_, v)| matches!(v, Voxel::Solid(x) if *x == t)),
                "{origin:?}"
            );
            assert_eq!(chunk.get_type(0, 0, 0), Some(t));
            assert_eq!(chunk.get_type(15, 15, 15), Some(t));
            let arc = w.get_leaf_chunk_arc_at_origin(origin).unwrap();
            assert_eq!(arc.count(), 4096);
            assert_eq!(arc.get_type(7, 8, 9), Some(t));
        }

        // ordinary and empty cells behave as before
        assert_eq!(w.get_leaf_chunk_at_origin(p(0, 0, 0)).unwrap().count(), 1);
        assert!(w.get_leaf_chunk_at_origin(p(48, 16, 0)).is_none());
        assert!(w.get_leaf_chunk_at_origin(p(256 + 256, 0, 512)).is_none());
        assert!(w.get_leaf_chunk_arc_at_origin(p(48, 16, 0)).is_none());
        // unaligned and out-of-bounds origins are still rejected
        assert!(w.get_leaf_chunk_at_origin(p(33, 16, 0)).is_none());
        assert!(w.get_leaf_chunk_at_origin(p(-16, 0, 0)).is_none());
        assert!(w.get_leaf_chunk_at_origin(p(4096, 0, 0)).is_none());
    }

    #[test]
    fn leaf_chunk_lookup_expands_uniform_solids_depth2() {
        let mut w = World::new(2);
        w.set_uniform_solid([16, 0, 0], 16, 9);
        let chunk = w.get_leaf_chunk_at_origin(p(16, 0, 0)).unwrap();
        assert_eq!(chunk.count(), 4096);
        assert_eq!(chunk.get_type(3, 4, 5), Some(9));
        assert!(w.get_leaf_chunk_at_origin(p(0, 0, 0)).is_none());
    }

    #[test]
    fn faces_next_to_a_coarse_solid_are_culled_by_the_neighbour_lookup() {
        // The mesher / shell code asks the world for the neighbouring leaf chunks; a coarse solid
        // neighbour must count as blocking the shared boundary.
        let mut w = World::new(3);
        w.set_uniform_solid([32, 0, 0], 16, 7);
        w.set(p(31, 5, 5), 1); // local (15,5,5) of leaf chunk (16,0,0), touching the solid's -X face
        let leaf = w.get_leaf_chunk_at_origin(p(16, 0, 0)).unwrap();
        let px = w.get_leaf_chunk_arc_at_origin(p(32, 0, 0));
        assert!(px.is_some());

        let open = leaf.compute_visibility_mask_with_neighbors(None, None, None, None, None, None);
        assert_eq!(open, 0b111111);
        let covered =
            leaf.compute_visibility_mask_with_neighbors(px.as_ref(), None, None, None, None, None);
        assert_eq!(
            covered, 0b111110,
            "+X face against a coarse solid must not be exposed"
        );
    }

    #[test]
    fn non_allocating_leaf_lookup_agrees_with_the_arc_lookup() {
        let mut w = World::new(3);
        w.set_uniform_solid([32, 16, 0], 16, 7);
        w.set_uniform_solid([256, 0, 512], 256, 3);
        w.set(p(1, 1, 1), 1);
        for (x, y, z) in [
            (0, 0, 0),       // ordinary subdivided leaf
            (32, 16, 0),     // leaf-sized coarse solid
            (48, 16, 0),     // empty
            (256, 0, 512),   // root-level coarse solid (first leaf)
            (496, 240, 752), // root-level coarse solid (last leaf)
            (512, 0, 512),   // just past it
            (33, 16, 0),     // unaligned
            (-16, 0, 0),     // outside
            (4096, 0, 0),    // outside
        ] {
            let owned = w.get_leaf_chunk_arc_at_origin(p(x, y, z));
            let borrowed = w.leaf_chunk_arc_ref_at_origin(p(x, y, z));
            match (owned, borrowed) {
                (None, None) => {}
                (Some(a), Some(b)) => assert!(Arc::ptr_eq(&a, b), "({x},{y},{z})"),
                (a, b) => panic!("({x},{y},{z}): {:?} vs {:?}", a.is_some(), b.is_some()),
            }
        }
    }

    #[test]
    fn uniform_chunks_are_shared_and_typed() {
        let mut w = World::new(3);
        w.set_uniform_solid([32, 0, 0], 16, 7);
        w.set_uniform_solid([48, 0, 0], 16, 7);
        w.set_uniform_solid([64, 0, 0], 16, 8);
        let a = w.get_leaf_chunk_arc_at_origin(p(32, 0, 0)).unwrap();
        let b = w.get_leaf_chunk_arc_at_origin(p(48, 0, 0)).unwrap();
        let c = w.get_leaf_chunk_arc_at_origin(p(64, 0, 0)).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "same-type uniform chunks should be shared"
        );
        assert!(!Arc::ptr_eq(&a, &c));
        assert_eq!(a.dominant_type, 7);
        assert_eq!(c.dominant_type, 8);
    }

    #[test]
    fn editing_inside_a_coarse_solid_keeps_the_rest_of_it() {
        let mut w = World::new(3);
        w.set_uniform_solid([256, 0, 0], 256, 4);
        assert_eq!(w.get(p(300, 20, 30)), Some(4));

        w.remove(p(300, 20, 30));
        assert_eq!(w.get(p(300, 20, 30)), None);
        assert_eq!(w.get(p(301, 20, 30)), Some(4));
        assert_eq!(w.get(p(300, 21, 30)), Some(4));
        assert_eq!(w.get(p(511, 255, 255)), Some(4));
        assert_eq!(w.get(p(256, 0, 0)), Some(4));

        w.set(p(300, 20, 30), 9);
        assert_eq!(w.get(p(300, 20, 30)), Some(9));
        assert_eq!(w.get(p(300, 20, 31)), Some(4));
        // the edited leaf chunk reports the edit, a neighbouring one is still uniform
        let edited = w.get_leaf_chunk_at_origin(p(288, 16, 16)).unwrap();
        assert_eq!(edited.get_type(12, 4, 14), Some(9));
        assert_eq!(edited.get_type(11, 4, 14), Some(4));
        assert_eq!(
            w.get_leaf_chunk_at_origin(p(304, 16, 16)).unwrap().count(),
            4096
        );
    }

    #[test]
    fn randomized_edits_match_a_shadow_model() {
        let mut rng = StdRng::seed_from_u64(0xC0A25E5);
        for depth in [2u8, 3, 4] {
            let mut w = World::new(depth);
            // coarse solids (lo, size, type), placed coarse -> fine so the finer one replaces part
            // of the coarser one (the model lets the smallest box win)
            let mut boxes: Vec<([i64; 3], i64, VoxelType)> = Vec::new();
            let origins: [[i64; 3]; 3] = [[48, 16, 80], [256, 0, 256], [0, 0, 0]]; // by level k-1
            for k in (1..depth as u32).rev() {
                let sz = 16i64.pow(k);
                let o = if depth == 2 {
                    [32, 16, 48]
                } else {
                    origins[k as usize - 1]
                };
                w.set_uniform_solid(o, sz, 1 + k as u8);
                boxes.push((o, sz, 1 + k as u8));
            }
            let base = |pos: WorldPos| -> Option<VoxelType> {
                boxes
                    .iter()
                    .filter(|&&(lo, sz, _)| {
                        (0..3).all(|a| {
                            let c = [pos.x, pos.y, pos.z][a];
                            c >= lo[a] && c < lo[a] + sz
                        })
                    })
                    .min_by_key(|&&(_, sz, _)| sz)
                    .map(|&(_, _, t)| t)
            };
            let mut overrides: HashMap<(i64, i64, i64), Option<VoxelType>> = HashMap::new();
            let span = 16i64.pow(depth as u32).min(300);
            for step in 0..3000 {
                let pos = p(
                    rng.gen_range(0..span),
                    rng.gen_range(0..span.min(80)),
                    rng.gen_range(0..span),
                );
                if rng.gen_bool(0.5) {
                    let t = rng.gen_range(1..9);
                    w.set(pos, t);
                    overrides.insert((pos.x, pos.y, pos.z), Some(t));
                } else {
                    w.remove(pos);
                    overrides.insert((pos.x, pos.y, pos.z), None);
                }
                // probe a few random positions
                for _ in 0..4 {
                    let q = p(
                        rng.gen_range(0..span),
                        rng.gen_range(0..span.min(80)),
                        rng.gen_range(0..span),
                    );
                    let want = match overrides.get(&(q.x, q.y, q.z)) {
                        Some(v) => *v,
                        None => base(q),
                    };
                    assert_eq!(w.get(q), want, "depth {depth} step {step} {q:?}");
                }
            }
            for (&(x, y, z), &v) in &overrides {
                assert_eq!(w.get(p(x, y, z)), v, "depth {depth} final {x},{y},{z}");
            }
        }
    }
}
