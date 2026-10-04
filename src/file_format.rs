//! Compact hierarchical voxel file format (vhc)
//!
//! Format mirrors the internal Chunk structure exactly:
//! - Header: depth (u8, 1..=7)
//! - Root chunk recursively encoded, nothing after it
//!
//! Chunk encoding:
//! - Position count (u16, max 4096 for 16³ chunk)
//! - For each occupied position:
//!   - Position encoded as u16 (z * 256 + y * 16 + x, below 4096, no duplicates)
//!   - If type == 0: sub-chunk follows (recursively encoded; not allowed in a leaf chunk)
//!   - Otherwise: solid voxel type (1-254)
//!
//! The loader validates these rules and reports violations as `InvalidData` errors.

use crate::lib_hierarchical::{Chunk, Voxel, World};
use croaring::Bitmap;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;
use std::sync::Arc;
use zstd::stream::read::Decoder as ZstdDecoder;
use zstd::stream::write::Encoder as ZstdEncoder;

/// Deepest hierarchy the format accepts (matches the Swift loader).
const MAX_DEPTH: u8 = 7;
/// Slots in a 16³ chunk.
const CHUNK_SLOTS: usize = 16 * 16 * 16;
/// Read-ahead for loading; the format is a stream of 2-3 byte records.
const LOAD_BUFFER_BYTES: usize = 1 << 20;

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Save world to compact format (vhc)
pub fn save_world(world: &World, writer: &mut impl Write) -> io::Result<()> {
    // Write depth
    writer.write_all(&[world.hierarchy_depth()])?;

    // Write root chunk
    save_chunk(world.root(), writer)?;

    Ok(())
}

/// Save world to a file path. The function writes compressed `.vhc` using zstd.
pub fn save_world_file(world: &World, path: &Path, _compress: bool) -> io::Result<()> {
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
    // `voxels` is stored in rank order, so walking the presence bitmap alongside it yields each
    // position with its voxel without a rank lookup per voxel.
    debug_assert_eq!(chunk.presence.cardinality(), chunk.voxels.len() as u64);

    // Write count (u16)
    let count = chunk.voxels.len() as u16;
    writer.write_all(&count.to_le_bytes())?;

    // Records are collected here and written in one go, up to the next sub-chunk (a leaf chunk
    // is a single write).
    let mut records = [0u8; 3 * CHUNK_SLOTS];
    let mut len = 0;
    for (pos_encoded, voxel) in chunk.presence.iter().zip(chunk.voxels.iter()) {
        // Position as u16 (z * 256 + y * 16 + x), which is the flat index, little endian
        let [pos_lo, pos_hi] = (pos_encoded as u16).to_le_bytes();
        match voxel {
            Voxel::Solid(vtype) => {
                records[len..len + 3].copy_from_slice(&[pos_lo, pos_hi, *vtype]);
                len += 3;
            }
            Voxel::Chunk(sub_chunk) => {
                records[len..len + 3].copy_from_slice(&[pos_lo, pos_hi, 0]); // 0 means sub-chunk follows
                writer.write_all(&records[..len + 3])?;
                len = 0;
                save_chunk(sub_chunk, writer)?;
            }
        }
    }
    writer.write_all(&records[..len])?;

    Ok(())
}

/// Load world from compact format (vhc).
///
/// The stream must hold exactly one world: bytes after the root chunk are an error.
pub fn load_world(reader: &mut impl Read) -> io::Result<World> {
    // The records are tiny; buffer the source so a file or zstd decoder is not asked for two or
    // three bytes at a time.
    let mut reader = BufReader::with_capacity(LOAD_BUFFER_BYTES, reader);

    // Read depth
    let mut depth_byte = [0u8; 1];
    reader.read_exact(&mut depth_byte)?;
    let depth = depth_byte[0];
    if !(1..=MAX_DEPTH).contains(&depth) {
        return Err(invalid_data(format!(
            "unsupported hierarchy depth {depth} (expected 1..={MAX_DEPTH})"
        )));
    }

    // Create empty world
    let mut world = World::new(depth);

    // Load root chunk
    load_chunk(world.root_mut(), &mut reader, depth)?;

    if !at_end(&mut reader)? {
        return Err(invalid_data("trailing bytes after root chunk"));
    }

    Ok(world)
}

/// Load a world from a file.
pub fn load_world_file(path: &Path) -> io::Result<World> {
    // We only support zstd-compressed `.vhc` files; legacy raw `.oct` files have been removed.
    let file = File::open(path)?;
    let mut decoder =
        ZstdDecoder::new(file).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    load_world(&mut decoder)
}

/// True if the reader has no more bytes.
fn at_end(reader: &mut impl Read) -> io::Result<bool> {
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(n) => return Ok(n == 0),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}

/// The records of one chunk as they are read: positions with their voxels, in file order.
struct Entries {
    positions: Vec<u32>,
    voxels: Vec<Voxel>,
    /// Positions are strictly ascending (which also rules out duplicates).
    ordered: bool,
}

impl Entries {
    fn with_capacity(count: usize) -> Self {
        Self {
            positions: Vec::with_capacity(count),
            voxels: Vec::with_capacity(count),
            ordered: true,
        }
    }

    fn push(&mut self, pos: u16, voxel: Voxel) {
        let pos = pos as u32;
        self.ordered &= self.positions.last().map_or(true, |&last| pos > last);
        self.positions.push(pos);
        self.voxels.push(voxel);
    }

    /// Positions ascending in rank order, with their voxels. The writer emits them that way, so
    /// only an out-of-order file pays for the sort.
    fn into_sorted(self) -> io::Result<(Vec<u32>, Vec<Voxel>)> {
        if self.ordered {
            return Ok((self.positions, self.voxels));
        }
        let mut pairs: Vec<(u32, Voxel)> = self.positions.into_iter().zip(self.voxels).collect();
        pairs.sort_by_key(|(pos, _)| *pos);
        if let Some(pair) = pairs.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid_data(format!(
                "duplicate chunk position {}",
                pair[0].0
            )));
        }
        Ok(pairs.into_iter().unzip())
    }
}

fn check_position(pos: u16) -> io::Result<()> {
    if pos as usize >= CHUNK_SLOTS {
        return Err(invalid_data(format!(
            "chunk position {pos} is out of range (max {})",
            CHUNK_SLOTS - 1
        )));
    }
    Ok(())
}

/// Load a chunk recursively. `level` is the number of hierarchy levels from this chunk down to
/// the leaves: 1 for a leaf chunk, which can only hold solid voxels.
fn load_chunk(chunk: &mut Chunk, reader: &mut impl Read, level: u8) -> io::Result<()> {
    // Read count of occupied positions (u16)
    let mut count_bytes = [0u8; 2];
    reader.read_exact(&mut count_bytes)?;
    let count = u16::from_le_bytes(count_bytes) as usize;
    if count > CHUNK_SLOTS {
        return Err(invalid_data(format!(
            "chunk declares {count} occupied slots (max {CHUNK_SLOTS})"
        )));
    }

    let mut entries = Entries::with_capacity(count);
    if level == 1 {
        // Leaf: every record is 3 bytes, read them in one go
        let mut records = [0u8; 3 * CHUNK_SLOTS];
        let records = &mut records[..3 * count];
        reader.read_exact(records)?;
        for record in records.chunks_exact(3) {
            let pos = u16::from_le_bytes([record[0], record[1]]);
            check_position(pos)?;
            match record[2] {
                0 => return Err(invalid_data("leaf chunk contains a sub-chunk")),
                255 => {} // empty (shouldn't happen but handle gracefully)
                vtype => entries.push(pos, Voxel::Solid(vtype)),
            }
        }
    } else {
        for _ in 0..count {
            // Read position (u16: z * 256 + y * 16 + x) and voxel type
            let mut record = [0u8; 3];
            reader.read_exact(&mut record)?;
            let pos = u16::from_le_bytes([record[0], record[1]]);
            check_position(pos)?;
            match record[2] {
                0 => {
                    // Sub-chunk follows; load recursively and store
                    let mut sub_chunk = Chunk::new();
                    load_chunk(&mut sub_chunk, reader, level - 1)?;
                    entries.push(pos, Voxel::Chunk(Arc::new(sub_chunk)));
                }
                255 => {} // empty (shouldn't happen but handle gracefully)
                vtype => entries.push(pos, Voxel::Solid(vtype)),
            }
        }
    }

    // Commit in rank order
    let (positions, voxels) = entries.into_sorted()?;

    // Aggregate counts, bbox and marginals for faster LOD init and coarse culling
    let mut solid_count: u32 = 0;
    let (mut xmin, mut ymin, mut zmin) = (16u8, 16u8, 16u8);
    let (mut xmax, mut ymax, mut zmax) = (0u8, 0u8, 0u8);
    let (mut px, mut py, mut pz) = (0u16, 0u16, 0u16);
    for (&pos, voxel) in positions.iter().zip(voxels.iter()) {
        let x = (pos & 0xF) as u8;
        let y = ((pos >> 4) & 0xF) as u8;
        let z = ((pos >> 8) & 0xF) as u8;

        xmin = xmin.min(x);
        ymin = ymin.min(y);
        zmin = zmin.min(z);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
        zmax = zmax.max(z);

        // Marginals for this slot
        px |= 1 << x;
        py |= 1 << y;
        pz |= 1 << z;

        match voxel {
            Voxel::Solid(_) => {
                solid_count = solid_count.saturating_add(1);
            }
            Voxel::Chunk(sub_chunk) => {
                solid_count = solid_count.saturating_add(sub_chunk.voxel_count);
                // Also OR-in the sub-chunk's projection bits for quick coarse culling
                px |= sub_chunk.px;
                py |= sub_chunk.py;
                pz |= sub_chunk.pz;
            }
        }
    }

    chunk.presence = if positions.len() == CHUNK_SLOTS {
        Bitmap::from_range(0..CHUNK_SLOTS as u32)
    } else {
        Bitmap::of(&positions)
    };
    chunk.bounding_box = if positions.is_empty() {
        None
    } else {
        Some([xmin, ymin, zmin, xmax, ymax, zmax])
    };
    chunk.voxels = voxels;
    chunk.px |= px;
    chunk.py |= py;
    chunk.pz |= pz;
    chunk.voxel_count = solid_count;
    chunk.solid_ratio = solid_count as f32 / CHUNK_SLOTS as f32;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib_hierarchical::test_util::assert_same_world;
    use crate::lib_hierarchical::WorldPos;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    // ---- Reference: the original reader and writer, kept verbatim as a baseline. ----

    fn old_save_chunk(chunk: &Chunk, writer: &mut impl Write) -> io::Result<()> {
        let positions: Vec<(u8, u8, u8)> = chunk.positions().collect();
        let count = positions.len() as u16;
        writer.write_all(&count.to_le_bytes())?;
        for (x, y, z) in positions {
            let pos_encoded = ((z as u16) << 8) | ((y as u16) << 4) | (x as u16);
            writer.write_all(&pos_encoded.to_le_bytes())?;
            match chunk.get(x, y, z) {
                Some(Voxel::Solid(vtype)) => {
                    writer.write_all(&[*vtype])?;
                }
                Some(Voxel::Chunk(sub_chunk)) => {
                    writer.write_all(&[0])?;
                    old_save_chunk(sub_chunk, writer)?;
                }
                None => {
                    writer.write_all(&[255])?;
                }
            }
        }
        Ok(())
    }

    fn old_save_world(world: &World) -> Vec<u8> {
        let mut bytes = vec![world.hierarchy_depth()];
        old_save_chunk(world.root(), &mut bytes).unwrap();
        bytes
    }

    fn old_load_chunk(chunk: &mut Chunk, reader: &mut impl Read) -> io::Result<()> {
        let mut count_bytes = [0u8; 2];
        reader.read_exact(&mut count_bytes)?;
        let count = u16::from_le_bytes(count_bytes);
        let mut entries: Vec<(u16, Voxel)> = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mut pos_bytes = [0u8; 2];
            reader.read_exact(&mut pos_bytes)?;
            let pos_encoded = u16::from_le_bytes(pos_bytes);
            let mut type_byte = [0u8; 1];
            reader.read_exact(&mut type_byte)?;
            let vtype = type_byte[0];
            if vtype == 0 {
                let mut sub_chunk = Chunk::new();
                old_load_chunk(&mut sub_chunk, reader)?;
                entries.push((pos_encoded, Voxel::Chunk(Arc::new(sub_chunk))));
            } else if vtype != 255 {
                entries.push((pos_encoded, Voxel::Solid(vtype)));
            }
        }
        entries.sort_by_key(|(pos, _)| *pos);
        chunk.voxels.reserve(entries.len());
        let mut solid_count: u32 = 0;
        let (mut xmin, mut ymin, mut zmin, mut xmax, mut ymax, mut zmax) =
            (16u8, 16, 16, 0u8, 0, 0);
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
                Voxel::Solid(_) => solid_count = solid_count.saturating_add(1),
                Voxel::Chunk(sub_chunk) => {
                    solid_count = solid_count.saturating_add(sub_chunk.voxel_count)
                }
            }
            chunk.presence.add(pos_encoded as u32);
            chunk.voxels.push(voxel);
            chunk.px |= 1 << x;
            chunk.py |= 1 << y;
            chunk.pz |= 1 << z;
            if let Voxel::Chunk(ref sub_chunk) = chunk.voxels.last().unwrap() {
                chunk.px |= sub_chunk.px;
                chunk.py |= sub_chunk.py;
                chunk.pz |= sub_chunk.pz;
            }
        }
        chunk.voxel_count = solid_count;
        chunk.solid_ratio = solid_count as f32 / (16.0 * 16.0 * 16.0);
        chunk.bounding_box = if bbox_found {
            Some([xmin, ymin, zmin, xmax, ymax, zmax])
        } else {
            None
        };
        Ok(())
    }

    fn old_load_world(bytes: &[u8]) -> World {
        let mut reader = bytes;
        let mut depth = [0u8; 1];
        reader.read_exact(&mut depth).unwrap();
        let mut world = World::new(depth[0]);
        old_load_chunk(world.root_mut(), &mut reader).unwrap();
        world
    }

    // ---- Helpers ----

    fn p(x: i64, y: i64, z: i64) -> WorldPos {
        WorldPos::new(x, y, z)
    }

    fn save_bytes(world: &World) -> Vec<u8> {
        let mut bytes = Vec::new();
        save_world(world, &mut bytes).unwrap();
        bytes
    }

    fn load_bytes(bytes: &[u8]) -> io::Result<World> {
        load_world(&mut &bytes[..])
    }

    fn load_error(bytes: &[u8]) -> io::Error {
        match load_bytes(bytes) {
            Ok(_) => panic!("expected an error for {bytes:?}"),
            Err(e) => e,
        }
    }

    fn assert_invalid(bytes: &[u8], expected: &str) {
        let e = load_error(bytes);
        assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{e} for {bytes:?}");
        assert!(e.to_string().contains(expected), "'{e}' lacks '{expected}'");
    }

    /// Chunk record: position (little endian) and type byte.
    fn rec(pos: u16, vtype: u8) -> Vec<u8> {
        let [lo, hi] = pos.to_le_bytes();
        vec![lo, hi, vtype]
    }

    /// Chunk with the given records already encoded (children included), preceded by its count.
    fn chunk_bytes(count: u16, records: &[Vec<u8>]) -> Vec<u8> {
        let mut bytes = count.to_le_bytes().to_vec();
        for r in records {
            bytes.extend_from_slice(r);
        }
        bytes
    }

    fn world_bytes(depth: u8, root: &[u8]) -> Vec<u8> {
        let mut bytes = vec![depth];
        bytes.extend_from_slice(root);
        bytes
    }

    /// Random world with coarse solids, empty chunks, full and sparse leaf chunks.
    fn random_world(rng: &mut StdRng, depth: u8) -> World {
        let mut world = World::new(depth);
        let size = 16i64.pow(depth as u32);
        let near = size.min(48);
        if depth >= 2 {
            for k in 1..depth as u32 {
                let sz = 16i64.pow(k);
                let o = |rng: &mut StdRng| rng.gen_range(0..(near / sz).max(1)) * sz;
                world.set_uniform_solid([o(rng), o(rng), o(rng)], sz, rng.gen_range(1..5));
            }
        }
        for _ in 0..2500 {
            let pos = p(
                rng.gen_range(0..near),
                rng.gen_range(0..near),
                rng.gen_range(0..near),
            );
            match rng.gen_range(0..10) {
                0..=6 => world.set(pos, rng.gen_range(1..6)),
                7 => world.set(pos, 0),
                _ => world.remove(pos),
            }
        }
        // a leaf chunk with all 4096 slots occupied by individually set voxels
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    world.set(p(x, y, z), 1 + ((x + y + z) % 3) as u8);
                }
            }
        }
        world
    }

    // ---- Writer and reader against the original implementation ----

    #[test]
    fn writer_output_is_byte_identical_to_the_original() {
        let mut rng = StdRng::seed_from_u64(0xF11E);
        for depth in 1..=4u8 {
            let world = random_world(&mut rng, depth);
            assert_eq!(save_bytes(&world), old_save_world(&world), "depth {depth}");
        }
    }

    #[test]
    fn reader_builds_the_same_structure_as_the_original() {
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        for depth in 1..=4u8 {
            let world = random_world(&mut rng, depth);
            let bytes = save_bytes(&world);
            let loaded = load_bytes(&bytes).unwrap();
            let reference = old_load_world(&bytes);
            assert_same_world(&loaded, &reference, &format!("depth {depth}"));

            // and it is the same world voxel by voxel
            for _ in 0..3000 {
                let q = p(
                    rng.gen_range(0..64.min(world.world_size() as i64)),
                    rng.gen_range(0..64.min(world.world_size() as i64)),
                    rng.gen_range(0..64.min(world.world_size() as i64)),
                );
                assert_eq!(loaded.get(q), world.get(q), "depth {depth} {q:?}");
            }
            // saving what was loaded gives the same bytes again
            assert_eq!(save_bytes(&loaded), bytes, "depth {depth}");
        }
    }

    #[test]
    fn lod_metadata_is_not_persisted() {
        // The generators used to run update_all_lod_metadata just before saving; the file must not
        // depend on it, because the viewer recomputes it after loading.
        let mut rng = StdRng::seed_from_u64(5);
        let mut world = random_world(&mut rng, 3);
        let before = save_bytes(&world);
        let palette =
            crate::Palette::from_string("1 200 100 50 255\n2 10 20 30 255 255 255 255 200\n")
                .unwrap();
        world.update_all_lod_metadata(&palette);
        assert!(world.root().voxel_count > 0 && world.root().average_color != [0; 4]);
        assert_eq!(save_bytes(&world), before);
    }

    #[test]
    fn unsorted_unique_positions_load_like_sorted_ones() {
        // depth 1, positions 9, 3, 700 in that order
        let shuffled = world_bytes(1, &chunk_bytes(3, &[rec(9, 1), rec(3, 2), rec(700, 3)]));
        let sorted = world_bytes(1, &chunk_bytes(3, &[rec(3, 2), rec(9, 1), rec(700, 3)]));
        let a = load_bytes(&shuffled).unwrap();
        let b = load_bytes(&sorted).unwrap();
        assert_same_world(&a, &b, "shuffled");
        assert_same_world(
            &a,
            &old_load_world(&shuffled),
            "shuffled vs original reader",
        );
        assert_eq!(save_bytes(&a), sorted, "the writer emits rank order");
    }

    #[test]
    fn empty_marker_records_are_skipped() {
        // type 255 means "no voxel" and does not take part in duplicate checks
        let bytes = world_bytes(
            1,
            &chunk_bytes(3, &[rec(4, 7), rec(4, 255), rec(4095, 255)]),
        );
        let world = load_bytes(&bytes).unwrap();
        assert_eq!(world.get(p(4, 0, 0)), Some(7));
        assert_eq!(world.count(), 1);
        assert_same_world(&world, &old_load_world(&bytes), "255 markers");
    }

    #[test]
    fn empty_chunks_and_worlds_round_trip() {
        for depth in 1..=7u8 {
            let bytes = world_bytes(depth, &chunk_bytes(0, &[]));
            let world = load_bytes(&bytes).unwrap();
            assert_eq!(world.hierarchy_depth(), depth);
            assert_eq!(world.count(), 0);
            assert_eq!(save_bytes(&world), bytes);
        }
        // an empty sub-chunk (what editing leaves behind) survives
        let bytes = world_bytes(
            2,
            &chunk_bytes(1, &[[rec(5, 0), chunk_bytes(0, &[])].concat()]),
        );
        let world = load_bytes(&bytes).unwrap();
        assert_eq!(save_bytes(&world), bytes);
    }

    #[test]
    fn file_round_trip() {
        let mut rng = StdRng::seed_from_u64(7);
        let world = random_world(&mut rng, 3);
        let path = std::env::temp_dir().join(format!("voxelot_ff_test_{}.vhc", std::process::id()));
        save_world_file(&world, &path, true).unwrap();
        let loaded = load_world_file(&path);
        let _ = std::fs::remove_file(&path);
        let loaded = loaded.unwrap();
        assert_eq!(save_bytes(&loaded), save_bytes(&world));
    }

    // ---- Malformed input ----

    #[test]
    fn rejects_unsupported_depths() {
        for depth in [0u8, 8, 17, 255] {
            // depth 0 used to panic in World::new (and abort in release builds)
            assert_invalid(&world_bytes(depth, &chunk_bytes(0, &[])), "depth");
        }
        assert_eq!(load_error(&[]).kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn rejects_counts_above_4096() {
        assert_invalid(&world_bytes(1, &4097u16.to_le_bytes()), "occupied slots");
        assert_invalid(&world_bytes(1, &u16::MAX.to_le_bytes()), "occupied slots");
        // 4096 is fine (needs the records to be there, so it is truncated here instead)
        assert_eq!(
            load_error(&world_bytes(1, &4096u16.to_le_bytes())).kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn rejects_positions_out_of_range() {
        for pos in [4096u16, 4097, 0x8000, u16::MAX] {
            assert_invalid(
                &world_bytes(1, &chunk_bytes(1, &[rec(pos, 1)])),
                "out of range",
            );
            assert_invalid(
                &world_bytes(2, &chunk_bytes(1, &[rec(pos, 1)])),
                "out of range",
            );
        }
        // the last valid slot is accepted
        let world = load_bytes(&world_bytes(1, &chunk_bytes(1, &[rec(4095, 1)]))).unwrap();
        assert_eq!(world.get(p(15, 15, 15)), Some(1));
    }

    #[test]
    fn rejects_duplicate_positions() {
        assert_invalid(
            &world_bytes(1, &chunk_bytes(2, &[rec(5, 1), rec(5, 2)])),
            "duplicate",
        );
        assert_invalid(
            &world_bytes(1, &chunk_bytes(3, &[rec(9, 1), rec(3, 1), rec(9, 2)])),
            "duplicate",
        );
        // a duplicate in a non-leaf chunk, between a solid and a sub-chunk
        let sub = chunk_bytes(1, &[rec(1, 1)]);
        assert_invalid(
            &world_bytes(2, &chunk_bytes(2, &[rec(7, 3), [rec(7, 0), sub].concat()])),
            "duplicate",
        );
        // and inside a sub-chunk
        let sub = chunk_bytes(2, &[rec(1, 1), rec(1, 2)]);
        assert_invalid(
            &world_bytes(2, &chunk_bytes(1, &[[rec(7, 0), sub].concat()])),
            "duplicate",
        );
    }

    #[test]
    fn rejects_sub_chunks_in_leaf_chunks() {
        // depth 1: the root is a leaf
        assert_invalid(
            &world_bytes(
                1,
                &chunk_bytes(1, &[[rec(0, 0), chunk_bytes(0, &[])].concat()]),
            ),
            "leaf chunk contains a sub-chunk",
        );
        // depth 2: the sub-chunk is a leaf
        let leaf = chunk_bytes(1, &[[rec(2, 0), chunk_bytes(0, &[])].concat()]);
        assert_invalid(
            &world_bytes(2, &chunk_bytes(1, &[[rec(0, 0), leaf].concat()])),
            "leaf chunk contains a sub-chunk",
        );
    }

    #[test]
    fn rejects_trailing_bytes() {
        let ok = world_bytes(1, &chunk_bytes(1, &[rec(3, 1)]));
        assert!(load_bytes(&ok).is_ok());
        let mut extra = ok.clone();
        extra.push(0);
        assert_invalid(&extra, "trailing bytes");
        let mut extra = ok;
        extra.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_invalid(&extra, "trailing bytes");
    }

    #[test]
    fn truncated_streams_are_errors() {
        let mut rng = StdRng::seed_from_u64(3);
        let world = random_world(&mut rng, 2);
        let bytes = save_bytes(&world);
        assert!(load_bytes(&bytes).is_ok());
        for cut in (0..bytes.len()).step_by(97).chain([bytes.len() - 1]) {
            assert!(
                load_bytes(&bytes[..cut]).is_err(),
                "prefix of {cut} bytes loaded"
            );
        }
    }

    #[test]
    fn garbage_never_panics() {
        let mut rng = StdRng::seed_from_u64(0xBAD);
        for i in 0..3000 {
            let len = rng.gen_range(0..40);
            let mut bytes: Vec<u8> = (0..len).map(|_| rng.gen_range(0..=3)).collect();
            if i % 2 == 0 && !bytes.is_empty() {
                bytes[0] = rng.gen_range(0..10);
            }
            // only the absence of a panic (and of huge allocations) matters here
            let _ = load_bytes(&bytes);
        }
    }

    #[test]
    fn deeply_nested_sub_chunks_stop_at_the_leaf_level() {
        // Each record claims one more nested chunk than the depth allows: an error, not a stack
        // overflow.
        let mut bytes = Vec::new();
        for _ in 0..100_000 {
            bytes.extend_from_slice(&chunk_bytes(1, &[rec(0, 0)]));
        }
        assert_invalid(&world_bytes(7, &bytes), "leaf chunk contains a sub-chunk");
    }

    /// Slow in debug builds; run with `cargo test --release --lib shipped_worlds -- --ignored`.
    /// Every world under worlds/ must load and be re-saved byte for byte.
    #[test]
    #[ignore]
    fn shipped_worlds_load_and_resave_identically() {
        for name in ["flat_city_test", "water_level_test"] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("worlds/{name}.vhc"));
            let world = load_world_file(&path).unwrap();
            let tmp =
                std::env::temp_dir().join(format!("voxelot_{name}_{}.vhc", std::process::id()));
            save_world_file(&world, &tmp, true).unwrap();
            let same = std::fs::read(&path).unwrap() == std::fs::read(&tmp).unwrap();
            let _ = std::fs::remove_file(&tmp);
            assert!(same, "{name} was not re-saved identically");
        }
    }
}
