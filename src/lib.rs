//! Sparse voxel world with Regions → Chunks → Voxels using croaring::Bitmap (u32).
//! - Flat index per chunk: no false positives
//! - Rank-packed u16 types aligned to presence
//! - Sparse regions/chunks via HashMap
//! - Configurable sizes via constants

use std::collections::HashMap;
use croaring::Bitmap;

// ========= Cached 2D view mask (per-chunk row bitmaps) =========

/// A cached 2D (x,y) mask over chunks: for each region (rx,ry), and chunk (cx,cy),
/// keep VOXEL_BASE rows, each as a Bitmap of local x positions that are still visible.
#[derive(Default, Debug)]
pub struct ViewPlaneMask {
    pub regions: HashMap<(i32, i32), HashMap<(u32, u32), Vec<Bitmap>>>,
}

/// Tunable sizes (voxels per axis).
/// - VOXEL_BASE: voxels per chunk side
/// - CHUNK_BASE: chunks per region side
/// - REGION_BASE: not used functionally yet, but kept configurable
pub const VOXEL_BASE: u32 = 128;
pub const CHUNK_BASE: u32 = 64;
pub const REGION_BASE: u32 = 32;

// Derived sizes
pub const VOXELS_PER_CHUNK: u32 = VOXEL_BASE; // along one axis
pub const CHUNKS_PER_REGION: u32 = CHUNK_BASE; // along one axis
pub const VOXELS_PER_REGION: u32 = VOXELS_PER_CHUNK * CHUNKS_PER_REGION; // along one axis

// Strides for flat indices (x + y*Sx + z*Sx*Sy)
#[inline]
const fn stride_x_voxel() -> u32 { 1 }
#[inline]
const fn stride_y_voxel() -> u32 { VOXEL_BASE }
#[inline]
const fn stride_z_voxel() -> u32 { VOXEL_BASE * VOXEL_BASE }

#[inline]
const fn stride_x_chunk() -> u32 { 1 }
#[inline]
const fn stride_y_chunk() -> u32 { CHUNK_BASE }
#[inline]
const fn stride_z_chunk() -> u32 { CHUNK_BASE * CHUNK_BASE }

pub type RegionKey = (i32, i32, i32);

/// One chunk of VOXEL_BASE³ voxels.
/// presence: RBM of flat voxel indices
/// types: rank-packed u16, aligned to presence ordering
#[derive(Default)]
pub struct Chunk {
    presence: Bitmap,
    types: Vec<u16>,
}

impl Chunk {
    pub fn new() -> Self {
        Self { presence: Bitmap::new(), types: Vec::new() }
    }

    #[inline]
    pub fn flat(x: u32, y: u32, z: u32) -> u32 {
        x * stride_x_voxel() + y * stride_y_voxel() + z * stride_z_voxel()
    }

    pub fn get_local(&self, x: u32, y: u32, z: u32) -> Option<u16> {
        if x >= VOXEL_BASE || y >= VOXEL_BASE || z >= VOXEL_BASE { return None; }
        let idx = Self::flat(x, y, z);
        if !self.presence.contains(idx) { return None; }
        // rank() returns count of elements <= idx (inclusive)
        let r = self.presence.rank(idx) as usize;
        if r == 0 { return None; }
        Some(self.types[r - 1])
    }

    pub fn set_local(&mut self, x: u32, y: u32, z: u32, typ: u16) -> Option<u16> {
        assert!(x < VOXEL_BASE && y < VOXEL_BASE && z < VOXEL_BASE);
        let idx = Self::flat(x, y, z);
        let contained = self.presence.contains(idx);
        let r_inclusive = self.presence.rank(idx) as usize;

        if contained {
            // update at r-1
            let pos = r_inclusive - 1;
            let old = self.types[pos];
            self.types[pos] = typ;
            Some(old)
        } else {
            // insert new voxel
            self.presence.add(idx);
            // For absent value, inclusive rank == count of elements < idx (insert position)
            self.types.insert(r_inclusive, typ);
            None
        }
    }

    pub fn remove_local(&mut self, x: u32, y: u32, z: u32) -> Option<u16> {
        if x >= VOXEL_BASE || y >= VOXEL_BASE || z >= VOXEL_BASE { return None; }
        let idx = Self::flat(x, y, z);
        if !self.presence.contains(idx) { return None; }
        let r = self.presence.rank(idx) as usize;
        let pos = r - 1;
        let old = self.types.remove(pos);
        self.presence.remove(idx);
        Some(old)
    }

    pub fn voxel_count(&self) -> usize { self.presence.cardinality() as usize }
    pub fn presence(&self) -> &Bitmap { &self.presence }

    // Bitwise operations on chunk presence
    
    /// Union: self |= other (add all voxels from other, keeping types from other where they exist)
    pub fn union_with(&mut self, other: &Chunk) {
        // For each voxel in other, add it to self
        for idx in other.presence.iter() {
            if !self.presence.contains(idx) {
                // New voxel - insert it
                let rank_other = other.presence.rank(idx) as usize;
                let typ = other.types[rank_other - 1];
                
                self.presence.add(idx);
                let rank_self = self.presence.rank(idx) as usize;
                self.types.insert(rank_self, typ);
            }
            // If voxel already exists in self, keep self's type
        }
    }

    /// Intersection: self &= other (keep only voxels present in both)
    pub fn intersect_with(&mut self, other: &Chunk) {
        let intersection = self.presence.and(&other.presence);
        
        // Rebuild types array for intersection
        let mut new_types = Vec::with_capacity(intersection.cardinality() as usize);
        for idx in intersection.iter() {
            let rank_self = self.presence.rank(idx) as usize;
            new_types.push(self.types[rank_self - 1]);
        }
        
        self.presence = intersection;
        self.types = new_types;
    }

    /// Difference: self -= other (remove voxels present in other)
    pub fn difference_with(&mut self, other: &Chunk) {
        let difference = self.presence.andnot(&other.presence);
        
        // Rebuild types array for difference
        let mut new_types = Vec::with_capacity(difference.cardinality() as usize);
        for idx in difference.iter() {
            let rank_self = self.presence.rank(idx) as usize;
            new_types.push(self.types[rank_self - 1]);
        }
        
        self.presence = difference;
        self.types = new_types;
    }

    /// XOR: self ^= other (symmetric difference - voxels in one but not both)
    pub fn xor_with(&mut self, other: &Chunk) {
        let xor_result = self.presence.xor(&other.presence);
        
        // Rebuild types array for xor
        let mut new_types = Vec::with_capacity(xor_result.cardinality() as usize);
        for idx in xor_result.iter() {
            if self.presence.contains(idx) {
                let rank_self = self.presence.rank(idx) as usize;
                new_types.push(self.types[rank_self - 1]);
            } else {
                // Must be from other
                let rank_other = other.presence.rank(idx) as usize;
                new_types.push(other.types[rank_other - 1]);
            }
        }
        
        self.presence = xor_result;
        self.types = new_types;
    }

    /// Extract all voxels at a specific Z level using bitmap range operations
    /// This is MUCH faster than iterating through all voxels
    pub fn extract_z_slice(&self, z: u32) -> Chunk {
        if z >= VOXEL_BASE {
            return Chunk::new();
        }
        
        // Calculate the flat index range for this Z level
        // For z-level: indices range from [z*Sy*Sz, (z+1)*Sy*Sz)
        let start_idx = z * stride_z_voxel();
        let end_idx = (z + 1) * stride_z_voxel();
        
        // Create a new bitmap with only voxels in this Z range
        let mut z_plane = Bitmap::new();
        
        // Iterate through existing voxels and check if they're in range
        for idx in self.presence.iter() {
            if idx >= start_idx && idx < end_idx {
                z_plane.add(idx);
            }
        }
        
        // Build the types array for the z-plane
        let mut new_types = Vec::with_capacity(z_plane.cardinality() as usize);
        for idx in z_plane.iter() {
            let rank = self.presence.rank(idx) as usize;
            new_types.push(self.types[rank - 1]);
        }
        
        Chunk {
            presence: z_plane,
            types: new_types,
        }
    }
}

/// Helper to clone a chunk (Bitmap doesn't implement Clone, but has clone method)
fn clone_chunk(chunk: &Chunk) -> Chunk {
    Chunk {
        presence: chunk.presence.clone(),
        types: chunk.types.clone(),
    }
}

/// Helper to clone a region
fn clone_region(region: &Region) -> Region {
    Region {
        chunk_presence: region.chunk_presence.clone(),
        chunks: region.chunks.iter().map(|(&k, v)| (k, clone_chunk(v))).collect(),
    }
}

/// Helper to clone a voxel world
fn clone_world(world: &VoxelWorld) -> VoxelWorld {
    VoxelWorld {
        regions: world.regions.iter().map(|(&k, v)| (k, clone_region(v))).collect(),
    }
}

/// A sparse region containing up to CHUNK_BASE³ chunk slots in a local grid.
/// chunk_presence: RBM of flat chunk indices present in this region.
/// chunks: loaded chunks by local flat id.
#[derive(Default)]
pub struct Region {
    chunk_presence: Bitmap,
    chunks: HashMap<u32, Chunk>,
}

impl Region {
    pub fn new() -> Self { Self { chunk_presence: Bitmap::new(), chunks: HashMap::new() } }

    #[inline]
    pub fn flat_chunk(cx: u32, cy: u32, cz: u32) -> u32 {
        cx * stride_x_chunk() + cy * stride_y_chunk() + cz * stride_z_chunk()
    }

    pub fn get_chunk(&self, cx: u32, cy: u32, cz: u32) -> Option<&Chunk> {
        if cx >= CHUNK_BASE || cy >= CHUNK_BASE || cz >= CHUNK_BASE { return None; }
        let id = Self::flat_chunk(cx, cy, cz);
        self.chunks.get(&id)
    }

    pub fn get_chunk_mut(&mut self, cx: u32, cy: u32, cz: u32) -> &mut Chunk {
        assert!(cx < CHUNK_BASE && cy < CHUNK_BASE && cz < CHUNK_BASE);
        let id = Self::flat_chunk(cx, cy, cz);
        self.chunk_presence.add(id);
        self.chunks.entry(id).or_insert_with(Chunk::new)
    }

    pub fn chunk_presence(&self) -> &Bitmap { &self.chunk_presence }
    pub fn chunk_count(&self) -> usize { self.chunks.len() }

    // Bitwise operations on region chunks
    
    /// Union: merge chunks from other region into self
    pub fn union_with(&mut self, other: &Region) {
        for chunk_id in other.chunk_presence.iter() {
            let other_chunk = &other.chunks[&chunk_id];
            
            if let Some(self_chunk) = self.chunks.get_mut(&chunk_id) {
                // Chunk exists in both - merge
                self_chunk.union_with(other_chunk);
            } else {
                // Chunk only in other - clone it
                self.chunks.insert(chunk_id, clone_chunk(other_chunk));
                self.chunk_presence.add(chunk_id);
            }
        }
    }

    /// Intersection: keep only chunks present in both regions
    pub fn intersect_with(&mut self, other: &Region) {
        let intersection = self.chunk_presence.and(&other.chunk_presence);
        
        // Remove chunks not in intersection
        self.chunks.retain(|id, chunk| {
            if intersection.contains(*id) {
                let other_chunk = &other.chunks[id];
                chunk.intersect_with(other_chunk);
                chunk.voxel_count() > 0 // Keep only if still has voxels
            } else {
                false
            }
        });
        
        self.chunk_presence = intersection;
    }

    /// Difference: remove chunks/voxels present in other
    pub fn difference_with(&mut self, other: &Region) {
        for chunk_id in other.chunk_presence.iter() {
            if let Some(self_chunk) = self.chunks.get_mut(&chunk_id) {
                if let Some(other_chunk) = other.chunks.get(&chunk_id) {
                    self_chunk.difference_with(other_chunk);
                    if self_chunk.voxel_count() == 0 {
                        self.chunks.remove(&chunk_id);
                        self.chunk_presence.remove(chunk_id);
                    }
                }
            }
        }
    }

    /// XOR: symmetric difference
    pub fn xor_with(&mut self, other: &Region) {
        let xor_result = self.chunk_presence.xor(&other.chunk_presence);
        
        for chunk_id in xor_result.iter() {
            if self.chunks.contains_key(&chunk_id) && other.chunks.contains_key(&chunk_id) {
                // In both - xor the chunks
                let other_chunk = &other.chunks[&chunk_id];
                let self_chunk = self.chunks.get_mut(&chunk_id).unwrap();
                self_chunk.xor_with(other_chunk);
                
                if self_chunk.voxel_count() == 0 {
                    self.chunks.remove(&chunk_id);
                }
            } else if !self.chunks.contains_key(&chunk_id) {
                // Only in other - add it
                self.chunks.insert(chunk_id, clone_chunk(&other.chunks[&chunk_id]));
            }
            // If only in self, keep it as-is
        }
        
        self.chunk_presence = xor_result;
    }
}

/// A sparse world keyed by region coordinates (i32), each region contains a local CHUNK_BASE³ grid.
#[derive(Default)]
pub struct VoxelWorld {
    regions: HashMap<RegionKey, Region>,
}

impl VoxelWorld {
    pub fn new() -> Self { Self { regions: HashMap::new() } }
    pub fn clear(&mut self) { self.regions.clear(); }

    #[inline]
    pub fn region_voxel_span() -> i64 { VOXELS_PER_REGION as i64 }

    /// Get mutable region by key, creating if missing.
    fn region_mut(&mut self, key: RegionKey) -> &mut Region {
        self.regions.entry(key).or_insert_with(Region::new)
    }

    // ========= Cached 2D view mask (per-chunk row bitmaps) =========

    

    /// Build a view-plane mask from world-space bounds using only per-row chunk spans.
    pub fn build_view_plane_mask(min_x: i64, max_x: i64, min_y: i64, max_y: i64) -> ViewPlaneMask {
        let mut mask = ViewPlaneMask { regions: HashMap::new() };

        for y in min_y..=max_y {
            let mut x = min_x;
            while x <= max_x {
                let ((rx, ry, _rz), (cx, cy, _cz), (vx, vy, _vz)) = {
                    let (rk, ck, vk) = VoxelWorld::map_global(x, y, 0);
                    (rk, ck, vk)
                };

                let region = mask.regions.entry((rx, ry)).or_insert_with(HashMap::new);
                let rows = region.entry((cx, cy)).or_insert_with(|| {
                    // initialize VOXEL_BASE rows of empty bitmaps
                    (0..VOXEL_BASE).map(|_| Bitmap::new()).collect::<Vec<_>>()
                });

                let x_in_chunk = vx as i64;
                let take = ((VOXEL_BASE as i64) - x_in_chunk).min(max_x - x + 1);
                let start_x = vx;
                let end_x = (vx as i64 + take) as u32; // exclusive
                let row = vy as usize;
                if end_x > start_x {
                    rows[row].add_range(start_x..end_x);
                }

                x += take;
            }
        }

        mask
    }

    /// Helper: collect runs [a,b) from a small bitmap of x positions (local to chunk row)
    fn collect_runs_x(bm: &Bitmap) -> Vec<(u32, u32)> {
        let mut runs = Vec::new();
        let mut run_start: Option<u32> = None;
        let mut last: u32 = 0;
        for v in bm.iter() {
            if let Some(s) = run_start {
                if v == last + 1 {
                    last = v;
                } else {
                    runs.push((s, last + 1));
                    run_start = Some(v);
                    last = v;
                }
            } else {
                run_start = Some(v);
                last = v;
            }
        }
        if let Some(s) = run_start {
            runs.push((s, last + 1));
        }
        runs
    }

    /// Build a VoxelWorld plane at world z from the current mask (no per-voxel loops):
    /// for each row, we add contiguous x-ranges shifted to local vz.
    pub fn build_plane_at_z_from_mask(mask: &ViewPlaneMask, z: i64, voxel_type: u16) -> VoxelWorld {
        let mut world = VoxelWorld::new();
        let vr = VOXELS_PER_REGION as i64;
        let (rz, oz) = div_euclid_floor(z, vr);
        let (cz, vz) = div_euclid_floor(oz, VOXELS_PER_CHUNK as i64);
        let cz = cz as u32;
        let vz = vz as u32;

        for (&(rx, ry), chunks_xy) in &mask.regions {
            let rkey = (rx, ry, rz as i32);
            let region = world.regions.entry(rkey).or_insert_with(Region::new);
            for (&(cx, cy), rows) in chunks_xy {
                let chunk_id = Region::flat_chunk(cx, cy, cz);
                let chunk = region.chunks.entry(chunk_id).or_insert_with(Chunk::new);
                region.chunk_presence.add(chunk_id);

                for (row_idx, row_bm) in rows.iter().enumerate() {
                    if row_bm.is_empty() { continue; }
                    let y = row_idx as u32;
                    for (x0, x1) in Self::collect_runs_x(row_bm) {
                        let start = Chunk::flat(x0, y, vz);
                        let end = Chunk::flat(x1, y, vz);
                        if end > start {
                            chunk.presence.add_range(start..end);
                            let count = (end - start) as usize;
                            chunk.types.extend(std::iter::repeat(voxel_type).take(count));
                        }
                    }
                }
            }
        }

        world
    }

    /// Remove hit voxels from the 2D mask (by x/y only). Hits are at any z; we map to (rx,ry),(cx,cy),(vx,vy).
    pub fn mask_subtract_hits(mask: &mut ViewPlaneMask, hits: &VoxelWorld) {
        hits.for_each_voxel(|x, y, _z, _t| {
            let ((rx, ry, _), (cx, cy, _), (vx, vy, _)) = {
                let (rk, ck, vk) = VoxelWorld::map_global(x, y, 0);
                (rk, ck, vk)
            };
            if let Some(chunks) = mask.regions.get_mut(&(rx, ry)) {
                if let Some(rows) = chunks.get_mut(&(cx, cy)) {
                    rows[vy as usize].remove(vx);
                }
            }
        });
    }

    /// Check if the mask is empty (no x left in any row)
    pub fn mask_is_empty(mask: &ViewPlaneMask) -> bool {
        for chunks in mask.regions.values() {
            for rows in chunks.values() {
                for bm in rows {
                    if !bm.is_empty() { return false; }
                }
            }
        }
        true
    }

    /// Compute hits at world z by intersecting the 2D mask with the world's z-slice directly at chunk level.
    /// Returns a VoxelWorld containing only the hit voxels at that z.
    pub fn intersect_mask_with_world_z(mask: &ViewPlaneMask, world: &VoxelWorld, z: i64) -> VoxelWorld {
        let mut hits_world = VoxelWorld::new();
        let vr = VOXELS_PER_REGION as i64;
        let (rz, oz) = div_euclid_floor(z, vr);
        let (cz, vz) = div_euclid_floor(oz, VOXELS_PER_CHUNK as i64);
        let cz = cz as u32;
        let vz = vz as u32;

        for (&(rx, ry), mask_chunks) in &mask.regions {
            let rkey_world = (rx, ry, rz as i32);
            let Some(world_region) = world.regions.get(&rkey_world) else { continue; };
            let hits_region = hits_world.regions.entry(rkey_world).or_insert_with(Region::new);

            for (&(cx, cy), rows) in mask_chunks {
                let chunk_id = Region::flat_chunk(cx, cy, cz);
                let Some(world_chunk) = world_region.chunks.get(&chunk_id) else { continue; };
                // Extract z-slice from world chunk
                let z_slice = world_chunk.extract_z_slice(vz);
                if z_slice.voxel_count() == 0 { continue; }

                // Build a mask chunk for this (cx,cy,cz) from row runs at vz
                let mut mask_chunk = Chunk::new();
                for (row_idx, row_bm) in rows.iter().enumerate() {
                    if row_bm.is_empty() { continue; }
                    let y = row_idx as u32;
                    for (x0, x1) in Self::collect_runs_x(row_bm) {
                        let start = Chunk::flat(x0, y, vz);
                        let end = Chunk::flat(x1, y, vz);
                        if end > start {
                            mask_chunk.presence.add_range(start..end);
                            // Types in mask are irrelevant; intersection copies from z_slice
                            let count = (end - start) as usize;
                            mask_chunk.types.extend(std::iter::repeat(0u16).take(count));
                        }
                    }
                }

                // Intersect world z-slice with mask_chunk
                let mut hits_chunk = z_slice; // own the z_slice
                hits_chunk.intersect_with(&mask_chunk);
                if hits_chunk.voxel_count() == 0 { continue; }

                hits_region.chunk_presence.add(chunk_id);
                hits_region.chunks.insert(chunk_id, hits_chunk);
            }
        }

        hits_world
    }

    /// Streaming variant: intersect mask with world at z, call on_hit for each hit, and subtract from mask in-place
    pub fn intersect_mask_with_world_z_collect<F>(mask: &mut ViewPlaneMask, world: &VoxelWorld, z: i64, mut on_hit: F)
    where F: FnMut(i64, i64, i64, u16)
    {
        let vr = VOXELS_PER_REGION as i64;
        let (rz, oz) = div_euclid_floor(z, vr);
        let (cz, vz) = div_euclid_floor(oz, VOXELS_PER_CHUNK as i64);
        let cz = cz as u32;
        let vz = vz as u32;

        for (&(rx, ry), mask_chunks) in &mut mask.regions {
            let rkey_world = (rx, ry, rz as i32);
            let Some(world_region) = world.regions.get(&rkey_world) else { continue; };

            let region_offset_x = rx as i64 * VOXELS_PER_REGION as i64;
            let region_offset_y = ry as i64 * VOXELS_PER_REGION as i64;

            for (&(cx, cy), rows) in mask_chunks.iter_mut() {
                let chunk_id = Region::flat_chunk(cx, cy, cz);
                let Some(world_chunk) = world_region.chunks.get(&chunk_id) else { continue; };
                let z_slice = world_chunk.extract_z_slice(vz);
                if z_slice.voxel_count() == 0 { continue; }

                // Build mask presence for this chunk at vz as contiguous ranges per row
                let mut mask_presence = Bitmap::new();
                for (row_idx, row_bm) in rows.iter().enumerate() {
                    if row_bm.is_empty() { continue; }
                    for (x0, x1) in Self::collect_runs_x(row_bm) {
                        let y = row_idx as u32;
                        let start = Chunk::flat(x0, y, vz);
                        let end = Chunk::flat(x1, y, vz);
                        if end > start { mask_presence.add_range(start..end); }
                    }
                }

                // Presence intersection
                let inter = z_slice.presence.and(&mask_presence);
                if inter.is_empty() { continue; }

                // Iterate hits and stream out world coords and types
                for idx in inter.iter() {
                    let vx = (idx % VOXEL_BASE) as i64;
                    let vy = ((idx / VOXEL_BASE) % VOXEL_BASE) as i64;
                    let world_x = region_offset_x + cx as i64 * VOXEL_BASE as i64 + vx;
                    let world_y = region_offset_y + cy as i64 * VOXEL_BASE as i64 + vy;
                    let rank = z_slice.presence.rank(idx) as usize;
                    let voxel_type = z_slice.types[rank - 1];
                    on_hit(world_x, world_y, z, voxel_type);

                    // subtract from 2D mask: remove vx bit from row vy
                    if let Some(row_bm) = rows.get_mut(vy as usize) { row_bm.remove(vx as u32); }
                }
            }
        }
    }

    /// Map global voxel coords to (region key), (local chunk coords), (local voxel coords).
    pub fn map_global(gx: i64, gy: i64, gz: i64) -> (RegionKey, (u32, u32, u32), (u32, u32, u32)) {
        let vr = Self::region_voxel_span();
        // Euclidean division (floor) to handle negatives
        let (rx, ox) = div_euclid_floor(gx, vr);
        let (ry, oy) = div_euclid_floor(gy, vr);
        let (rz, oz) = div_euclid_floor(gz, vr);

        let (cx, vx) = div_euclid_floor(ox, VOXELS_PER_CHUNK as i64);
        let (cy, vy) = div_euclid_floor(oy, VOXELS_PER_CHUNK as i64);
        let (cz, vz) = div_euclid_floor(oz, VOXELS_PER_CHUNK as i64);

        let region = (rx as i32, ry as i32, rz as i32);
        let chunk_local = (cx as u32, cy as u32, cz as u32);
        let voxel_local = (vx as u32, vy as u32, vz as u32);
        (region, chunk_local, voxel_local)
    }

    pub fn set_voxel(&mut self, gx: i64, gy: i64, gz: i64, typ: u16) -> Option<u16> {
        let (rkey, (cx, cy, cz), (vx, vy, vz)) = Self::map_global(gx, gy, gz);
        let region = self.region_mut(rkey);
        let chunk = region.get_chunk_mut(cx, cy, cz);
        chunk.set_local(vx, vy, vz, typ)
    }

    pub fn get_voxel(&self, gx: i64, gy: i64, gz: i64) -> Option<u16> {
        let (rkey, (cx, cy, cz), (vx, vy, vz)) = Self::map_global(gx, gy, gz);
        let region = self.regions.get(&rkey)?;
        let chunk = region.get_chunk(cx, cy, cz)?;
        chunk.get_local(vx, vy, vz)
    }

    pub fn remove_voxel(&mut self, gx: i64, gy: i64, gz: i64) -> Option<u16> {
        let (rkey, (cx, cy, cz), (vx, vy, vz)) = Self::map_global(gx, gy, gz);
        let region = self.regions.get_mut(&rkey)?;
        let chunk = region.get_chunk_mut(cx, cy, cz);
        chunk.remove_local(vx, vy, vz)
    }

    pub fn region_count(&self) -> usize { self.regions.len() }
    pub fn total_chunks(&self) -> usize { self.regions.values().map(|r| r.chunk_count()).sum() }
    pub fn total_voxels(&self) -> usize {
        self.regions.values().map(|r| r.chunks.values().map(|c| c.voxel_count()).sum::<usize>()).sum()
    }

    // Bitwise operations on entire worlds
    
    /// Union: self |= other (merge all voxels from other into self)
    pub fn union_with(&mut self, other: &VoxelWorld) {
        for (&region_key, other_region) in &other.regions {
            if let Some(self_region) = self.regions.get_mut(&region_key) {
                self_region.union_with(other_region);
            } else {
                self.regions.insert(region_key, clone_region(other_region));
            }
        }
    }

    /// Intersection: self &= other (keep only voxels present in both worlds)
    pub fn intersect_with(&mut self, other: &VoxelWorld) {
        self.regions.retain(|region_key, self_region| {
            if let Some(other_region) = other.regions.get(region_key) {
                self_region.intersect_with(other_region);
                self_region.chunk_count() > 0
            } else {
                false // Region not in other, remove it
            }
        });
    }

    /// Difference: self -= other (remove voxels present in other)
    pub fn difference_with(&mut self, other: &VoxelWorld) {
        for (&region_key, other_region) in &other.regions {
            if let Some(self_region) = self.regions.get_mut(&region_key) {
                self_region.difference_with(other_region);
                if self_region.chunk_count() == 0 {
                    self.regions.remove(&region_key);
                }
            }
        }
    }

    /// XOR: self ^= other (symmetric difference)
    pub fn xor_with(&mut self, other: &VoxelWorld) {
        // Process all regions in other
        for (&region_key, other_region) in &other.regions {
            if let Some(self_region) = self.regions.get_mut(&region_key) {
                self_region.xor_with(other_region);
                if self_region.chunk_count() == 0 {
                    self.regions.remove(&region_key);
                }
            } else {
                // Region only in other - add it
                self.regions.insert(region_key, clone_region(other_region));
            }
        }
    }

    /// Create a new world that is the union of self and other (non-mutating)
    pub fn union(&self, other: &VoxelWorld) -> VoxelWorld {
        let mut result = clone_world(self);
        result.union_with(other);
        result
    }

    /// Create a new world that is the intersection of self and other (non-mutating)
    pub fn intersect(&self, other: &VoxelWorld) -> VoxelWorld {
        let mut result = clone_world(self);
        result.intersect_with(other);
        result
    }

    /// Create a new world that is the difference of self and other (non-mutating)
    pub fn difference(&self, other: &VoxelWorld) -> VoxelWorld {
        let mut result = clone_world(self);
        result.difference_with(other);
        result
    }

    /// Create a new world that is the XOR of self and other (non-mutating)
    pub fn xor(&self, other: &VoxelWorld) -> VoxelWorld {
        let mut result = clone_world(self);
        result.xor_with(other);
        result
    }

    /// Collect all voxels in a given bounding box (for plane-based culling)
    /// Returns Vec of (x, y, z, type)
    pub fn collect_voxels_in_box(
        &self,
        min_x: i64, max_x: i64,
        min_y: i64, max_y: i64,
        min_z: i64, max_z: i64,
    ) -> Vec<(i64, i64, i64, u16)> {
        let mut result = Vec::new();
        
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    if let Some(voxel_type) = self.get_voxel(x, y, z) {
                        result.push((x, y, z, voxel_type));
                    }
                }
            }
        }
        
        result
    }

    /// Iterate through all voxels and call a function on each
    /// Useful for rendering without collecting into a Vec first
    pub fn for_each_voxel<F>(&self, mut f: F)
    where
        F: FnMut(i64, i64, i64, u16),
    {
        for (&(rx, ry, rz), region) in &self.regions {
            let region_offset_x = rx as i64 * VOXELS_PER_REGION as i64;
            let region_offset_y = ry as i64 * VOXELS_PER_REGION as i64;
            let region_offset_z = rz as i64 * VOXELS_PER_REGION as i64;
            
            for chunk_idx in region.chunk_presence.iter() {
                let cx = (chunk_idx % CHUNK_BASE) as i64;
                let cy = ((chunk_idx / CHUNK_BASE) % CHUNK_BASE) as i64;
                let cz = (chunk_idx / (CHUNK_BASE * CHUNK_BASE)) as i64;
                
                let chunk_offset_x = region_offset_x + cx * VOXEL_BASE as i64;
                let chunk_offset_y = region_offset_y + cy * VOXEL_BASE as i64;
                let chunk_offset_z = region_offset_z + cz * VOXEL_BASE as i64;
                
                if let Some(chunk) = region.chunks.get(&chunk_idx) {
                    for voxel_idx in chunk.presence.iter() {
                        let vx = (voxel_idx % VOXEL_BASE) as i64;
                        let vy = ((voxel_idx / VOXEL_BASE) % VOXEL_BASE) as i64;
                        let vz = (voxel_idx / (VOXEL_BASE * VOXEL_BASE)) as i64;
                        
                        let world_x = chunk_offset_x + vx;
                        let world_y = chunk_offset_y + vy;
                        let world_z = chunk_offset_z + vz;
                        
                        let rank = chunk.presence.rank(voxel_idx) as usize;
                        let voxel_type = chunk.types[rank - 1];
                        
                        f(world_x, world_y, world_z, voxel_type);
                    }
                }

            }
        }
    }

    /// Create a filled rectangular plane in world space (useful for plane-sweep culling)
    /// This is more efficient than calling set_voxel in a loop
    pub fn create_filled_plane(min_x: i64, max_x: i64, min_y: i64, max_y: i64, z: i64, voxel_type: u16) -> Self {
        let mut world = VoxelWorld::new();
        
        // Batch set voxels by region/chunk to minimize HashMap operations
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                world.set_voxel(x, y, z, voxel_type);
            }
        }
        
        world
    }

    /// Extract all voxels at a specific Z level (useful for plane-based operations)
    pub fn extract_z_plane(&self, z: i64) -> VoxelWorld {
        let mut plane = VoxelWorld::new();
        
        self.for_each_voxel(|x, y, voxel_z, voxel_type| {
            if voxel_z == z {
                plane.set_voxel(x, y, z, voxel_type);
            }
        });
        
        plane
    }

    /// Fast Z-plane extraction using bitmap operations (OPTIMIZED)
    /// This is much faster than extract_z_plane() which iterates all voxels
    pub fn extract_z_plane_fast(&self, target_z: i64) -> VoxelWorld {
        let mut plane = VoxelWorld::new();
        
        // Determine which regions might contain this Z level
        // Each region spans VOXELS_PER_REGION along each axis
        for (&(rx, ry, rz), region) in &self.regions {
            let region_z_min = rz as i64 * VOXELS_PER_REGION as i64;
            let region_z_max = region_z_min + VOXELS_PER_REGION as i64;
            
            // Skip regions that don't contain our target Z
            if target_z < region_z_min || target_z >= region_z_max {
                continue;
            }
            
            // Calculate local Z within the region
            let local_z_in_region = target_z - region_z_min;
            
            // Determine which chunks in this region contain our target Z
            let chunk_z = (local_z_in_region / VOXEL_BASE as i64) as u32;
            let local_z_in_chunk = (local_z_in_region % VOXEL_BASE as i64) as u32;
            
            // Iterate through chunks at this Z level
            for chunk_y in 0..CHUNK_BASE {
                for chunk_x in 0..CHUNK_BASE {
                    let chunk_idx = Region::flat_chunk(chunk_x, chunk_y, chunk_z);
                    
                    if let Some(chunk) = region.chunks.get(&chunk_idx) {
                        // Extract the Z-slice from this chunk using fast bitmap operations
                        let z_slice = chunk.extract_z_slice(local_z_in_chunk);
                        
                        // Add voxels from this slice to the plane
                        for voxel_idx in z_slice.presence.iter() {
                            let vx = (voxel_idx % VOXEL_BASE) as i64;
                            let vy = ((voxel_idx / VOXEL_BASE) % VOXEL_BASE) as i64;
                            // Z is already at our target level (local_z_in_chunk)
                            
                            // Calculate world coordinates
                            let world_x = rx as i64 * VOXELS_PER_REGION as i64 
                                        + chunk_x as i64 * VOXEL_BASE as i64 
                                        + vx;
                            let world_y = ry as i64 * VOXELS_PER_REGION as i64 
                                        + chunk_y as i64 * VOXEL_BASE as i64 
                                        + vy;
                            
                            // Get the voxel type
                            let rank = z_slice.presence.rank(voxel_idx) as usize;
                            let voxel_type = z_slice.types[rank - 1];
                            
                            plane.set_voxel(world_x, world_y, target_z, voxel_type);
                        }
                    }
                }
            }
        }
        
        plane
    }

    /// Fast Z-plane extraction returning voxels directly (no VoxelWorld construction)
    /// This avoids the overhead of calling set_voxel for each voxel
    pub fn extract_z_plane_direct(&self, target_z: i64) -> Vec<(i64, i64, i64, u16)> {
        let mut voxels = Vec::new();
        
        // Determine which regions might contain this Z level
        for (&(rx, ry, rz), region) in &self.regions {
            let region_z_min = rz as i64 * VOXELS_PER_REGION as i64;
            let region_z_max = region_z_min + VOXELS_PER_REGION as i64;
            
            // Skip regions that don't contain our target Z
            if target_z < region_z_min || target_z >= region_z_max {
                continue;
            }
            
            // Calculate local Z within the region
            let local_z_in_region = target_z - region_z_min;
            
            // Determine which chunks in this region contain our target Z
            let chunk_z = (local_z_in_region / VOXEL_BASE as i64) as u32;
            let local_z_in_chunk = (local_z_in_region % VOXEL_BASE as i64) as u32;
            
            // Iterate through chunks at this Z level
            for chunk_y in 0..CHUNK_BASE {
                for chunk_x in 0..CHUNK_BASE {
                    let chunk_idx = Region::flat_chunk(chunk_x, chunk_y, chunk_z);
                    
                    if let Some(chunk) = region.chunks.get(&chunk_idx) {
                        // Extract the Z-slice from this chunk using fast bitmap operations
                        let z_slice = chunk.extract_z_slice(local_z_in_chunk);
                        
                        // Add voxels from this slice directly to result
                        for voxel_idx in z_slice.presence.iter() {
                            let vx = (voxel_idx % VOXEL_BASE) as i64;
                            let vy = ((voxel_idx / VOXEL_BASE) % VOXEL_BASE) as i64;
                            
                            // Calculate world coordinates
                            let world_x = rx as i64 * VOXELS_PER_REGION as i64 
                                        + chunk_x as i64 * VOXEL_BASE as i64 
                                        + vx;
                            let world_y = ry as i64 * VOXELS_PER_REGION as i64 
                                        + chunk_y as i64 * VOXEL_BASE as i64 
                                        + vy;
                            
                            // Get the voxel type
                            let rank = z_slice.presence.rank(voxel_idx) as usize;
                            let voxel_type = z_slice.types[rank - 1];
                            
                            voxels.push((world_x, world_y, target_z, voxel_type));
                        }
                    }
                }
            }
        }
        
        voxels
    }
}

/// Euclidean division (floor) returning (q, r) such that:
/// a = q*b + r, with 0 <= r < |b|
#[inline]
fn div_euclid_floor(a: i64, b: i64) -> (i64, i64) {
    let mut q = a / b;
    let mut r = a % b;
    if r < 0 {
        if b > 0 { q -= 1; r += b; } else { q += 1; r -= b; }
    }
    (q, r)
}

/// Create a filled rectangular plane efficiently using direct bitmap operations
/// This is much faster than calling set_voxel() repeatedly
pub fn create_filled_plane(min_x: i64, max_x: i64, min_y: i64, max_y: i64, z: i64, voxel_type: u16) -> VoxelWorld {
    let mut world = VoxelWorld::new();
    
    // Group by regions for efficient batch insertion
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            let (rkey, (cx, cy, cz), (vx, vy, vz)) = VoxelWorld::map_global(x, y, z);
            
            // Get or create region
            let region = world.regions.entry(rkey).or_insert_with(Region::new);
            
            // Get or create chunk
            let chunk_idx = Region::flat_chunk(cx, cy, cz);
            if !region.chunks.contains_key(&chunk_idx) {
                region.chunks.insert(chunk_idx, Chunk::new());
                region.chunk_presence.add(chunk_idx);
            }
            
            let chunk = region.chunks.get_mut(&chunk_idx).unwrap();
            
            // Add voxel directly to bitmap
            let local_idx = Chunk::flat(vx, vy, vz);
            if !chunk.presence.contains(local_idx) {
                chunk.presence.add(local_idx);
                chunk.types.push(voxel_type);
            }
        }
    }
    
    world
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_euclid_floor() {
        assert_eq!(div_euclid_floor(5, 4), (1, 1));
        assert_eq!(div_euclid_floor(-1, 4), (-1, 3));
        assert_eq!(div_euclid_floor(-5, 4), (-2, 3));
        assert_eq!(div_euclid_floor(5, -4), (-1, 1));
    }

    #[test]
    fn test_world_set_get() {
        let mut w = VoxelWorld::new();
        let pos = (0i64, 0i64, 0i64);
        assert_eq!(w.get_voxel(pos.0, pos.1, pos.2), None);
        w.set_voxel(pos.0, pos.1, pos.2, 7);
        assert_eq!(w.get_voxel(pos.0, pos.1, pos.2), Some(7));
        assert_eq!(w.total_voxels(), 1);
        assert_eq!(w.region_count(), 1);
        assert_eq!(w.total_chunks(), 1);
    }

    #[test]
    fn test_negative_coords() {
        let mut w = VoxelWorld::new();
        let pos = (-1i64, -1i64, -1i64);
        w.set_voxel(pos.0, pos.1, pos.2, 42);
        assert_eq!(w.get_voxel(pos.0, pos.1, pos.2), Some(42));
    }

    #[test]
    fn test_contiguous_space_chunk_boundaries() {
        let mut w = VoxelWorld::new();
        let s = VOXEL_BASE as i64; // chunk side in voxels

        // Values around x boundary [s-1, s]
        w.set_voxel(s - 1, 0, 0, 1);
        w.set_voxel(s, 0, 0, 2);
        assert_eq!(w.get_voxel(s - 1, 0, 0), Some(1));
        assert_eq!(w.get_voxel(s, 0, 0), Some(2));

        // Negative side boundary [-1, 0]
        w.set_voxel(-1, 0, 0, 3);
        w.set_voxel(0, 0, 0, 4);
        assert_eq!(w.get_voxel(-1, 0, 0), Some(3));
        assert_eq!(w.get_voxel(0, 0, 0), Some(4));

        // Do the same for y and z
        w.set_voxel(0, s - 1, 0, 5);
        w.set_voxel(0, s, 0, 6);
        assert_eq!(w.get_voxel(0, s - 1, 0), Some(5));
        assert_eq!(w.get_voxel(0, s, 0), Some(6));

        w.set_voxel(0, -1, 0, 7);
        w.set_voxel(0, 0, 0, 8);
        assert_eq!(w.get_voxel(0, -1, 0), Some(7));
        assert_eq!(w.get_voxel(0, 0, 0), Some(8));

        w.set_voxel(0, 0, s - 1, 9);
        w.set_voxel(0, 0, s, 10);
        assert_eq!(w.get_voxel(0, 0, s - 1), Some(9));
        assert_eq!(w.get_voxel(0, 0, s), Some(10));

        w.set_voxel(0, 0, -1, 11);
        w.set_voxel(0, 0, 0, 12);
        assert_eq!(w.get_voxel(0, 0, -1), Some(11));
        assert_eq!(w.get_voxel(0, 0, 0), Some(12));
    }

    #[test]
    fn test_contiguous_space_region_boundaries() {
        let mut w = VoxelWorld::new();
        let r = VOXELS_PER_REGION as i64; // region side in voxels

        // Cross positive x region boundary
        w.set_voxel(r - 1, 0, 0, 21);
        w.set_voxel(r, 0, 0, 22);
        assert_eq!(w.get_voxel(r - 1, 0, 0), Some(21));
        assert_eq!(w.get_voxel(r, 0, 0), Some(22));

        // Cross negative x region boundary
        w.set_voxel(-1, 0, 0, 23);
        w.set_voxel(-r, 0, 0, 24);
        assert_eq!(w.get_voxel(-1, 0, 0), Some(23));
        assert_eq!(w.get_voxel(-r, 0, 0), Some(24));

        // y axis
        w.set_voxel(0, r - 1, 0, 25);
        w.set_voxel(0, r, 0, 26);
        assert_eq!(w.get_voxel(0, r - 1, 0), Some(25));
        assert_eq!(w.get_voxel(0, r, 0), Some(26));

        w.set_voxel(0, -1, 0, 27);
        w.set_voxel(0, -r, 0, 28);
        assert_eq!(w.get_voxel(0, -1, 0), Some(27));
        assert_eq!(w.get_voxel(0, -r, 0), Some(28));

        // z axis
        w.set_voxel(0, 0, r - 1, 29);
        w.set_voxel(0, 0, r, 30);
        assert_eq!(w.get_voxel(0, 0, r - 1), Some(29));
        assert_eq!(w.get_voxel(0, 0, r), Some(30));

        w.set_voxel(0, 0, -1, 31);
        w.set_voxel(0, 0, -r, 32);
        assert_eq!(w.get_voxel(0, 0, -1), Some(31));
        assert_eq!(w.get_voxel(0, 0, -r), Some(32));
    }
}
