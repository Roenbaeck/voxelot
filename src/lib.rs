//! Sparse voxel world with Regions → Chunks → Voxels using croaring::Bitmap (u32).
//! - Flat index per chunk: no false positives
//! - Rank-packed u16 types aligned to presence
//! - Sparse regions/chunks via HashMap
//! - Configurable sizes via constants

use std::collections::HashMap;
use croaring::Bitmap;

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
