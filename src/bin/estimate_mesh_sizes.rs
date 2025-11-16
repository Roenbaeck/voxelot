use std::env;
use voxelot::{load_world_file, Palette, generate_chunk_mesh, Chunk};

fn collect_leaf_chunks<'a>(chunk: &'a Chunk, out: &mut Vec<&'a Chunk>) {
    // A leaf chunk is a chunk without any sub-chunks (i.e. contains only Solid voxels)
    let mut has_subchunks = false;
    for (_pos, voxel) in chunk.iter() {
        if let voxelot::Voxel::Chunk(ref sub) = voxel {
            has_subchunks = true;
            collect_leaf_chunks(sub, out);
        }
    }
    if !has_subchunks {
        out.push(chunk);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: estimate_mesh_sizes <world.oct> <palette.txt>");
        std::process::exit(1);
    }
    let world_path = &args[1];
    let palette_path = &args[2];

    let palette = Palette::load(palette_path);
    let world = load_world_file(std::path::Path::new(world_path)).expect("Failed to load world");

    let mut leaves = Vec::new();
    collect_leaf_chunks(world.root(), &mut leaves);
    println!("Found {} leaf chunks", leaves.len());

    let mut nonempty = 0usize;
    let mut total_vertices = 0usize;
    let mut total_indices = 0usize;
    let mut quantized_bytes: usize = 0;
    let mut compressed_bytes_by_level: Vec<usize> = vec![0usize; 5];

    for (i, chunk) in leaves.iter().enumerate() {
        // Count presence directly; `voxel_count` may not be initialized on load
        let presence = chunk.count();
        if presence == 0 { continue; }
        nonempty += 1;
        let mesh = generate_chunk_mesh(chunk, &palette, 1.0, None);
        total_vertices += mesh.vertices.len();
        total_indices += mesh.indices.len();

        // Quantize per-vertex to smaller representation:
        // - position: 3x u16 (6 bytes) with pos * 16 (if positions are 0..16 as floats)
        // - normal: 1x u8 (1 byte) into 6 axis normals
        // - color: 4x u8 (4 bytes)
        // - emissive: 4x u8 (4 bytes) (R,G,B,intensity)
        // We'll store indices as u16 if vertices <= 65535, otherwise u32.

        let mut buf: Vec<u8> = Vec::new();
        for v in &mesh.vertices {
            // position within chunk is in [0..16], convert to u16 with small scale
            let px = (v.position[0] as f32 * 256.0).round() as u16;
            let py = (v.position[1] as f32 * 256.0).round() as u16;
            let pz = (v.position[2] as f32 * 256.0).round() as u16;
            buf.extend(&px.to_le_bytes());
            buf.extend(&py.to_le_bytes());
            buf.extend(&pz.to_le_bytes());

            // normal: choose axis with largest absolute component
            let axis_idx: u8 = if v.normal[0].abs() > v.normal[1].abs() && v.normal[0].abs() > v.normal[2].abs() {
                if v.normal[0] > 0.0 { 0 } else { 1 }
            } else if v.normal[1].abs() > v.normal[2].abs() {
                if v.normal[1] > 0.0 { 2 } else { 3 }
            } else {
                if v.normal[2] > 0.0 { 4 } else { 5 }
            };
            buf.push(axis_idx);

            // color: convert to u8
            for c in &v.color {
                let u = (c.clamp(0.0, 1.0) * 255.0).round() as u8;
                buf.push(u);
            }

            // emissive: also pack as u8s
            for e in &v.emissive {
                let u = (e.clamp(0.0, 1.0) * 255.0).round() as u8;
                buf.push(u);
            }
        }

        // indices
        if mesh.vertices.len() <= std::u16::MAX as usize {
            for idx in &mesh.indices {
                let i16 = *idx as u16;
                buf.extend(&i16.to_le_bytes());
            }
        } else {
            for idx in &mesh.indices {
                let i32b = *idx as u32;
                buf.extend(&i32b.to_le_bytes());
            }
        }

        quantized_bytes += buf.len();

        // Compress per-chunk and measure compressed size to approximate storage using zstd
        for (i, level) in [1, 3, 6, 9, 14].iter().enumerate() {
            let compressed = zstd::stream::encode_all(&buf[..], *level).expect("zstd failure");
            compressed_bytes_by_level[i] += compressed.len();
        }
        if i % 1000 == 0 { println!("Processed {} leaves", i); }
    }

    let mut nonzero_presence = 0usize;
    for chunk in &leaves {
        if chunk.count() > 0 { nonzero_presence += 1; }
    }
    println!("Leaf chunks with presence>0: {}", nonzero_presence);

    let vertex_bytes = total_vertices * std::mem::size_of::<voxelot::meshing::MeshVertex>();
    let index_bytes = total_indices * std::mem::size_of::<u32>();
    println!("Non-empty leaf chunks: {}", nonempty);
    println!("Total mesh vertices: {} ({} bytes)", total_vertices, vertex_bytes);
    println!("Total mesh indices: {} ({} bytes)", total_indices, index_bytes);
    println!("Total raw mesh bytes (vertices + indices): {} bytes ({:.2} MiB)", vertex_bytes + index_bytes, (vertex_bytes + index_bytes) as f64 / (1024.0*1024.0));
    println!("Quantized bytes (before compression): {} bytes ({:.2} MiB)", quantized_bytes, quantized_bytes as f64 / 1024.0 / 1024.0);
    for (i, level) in [1, 3, 6, 9, 14].iter().enumerate() {
        println!("Compressed quantized (zstd level {}) total: {} bytes ({:.2} MiB)", level, compressed_bytes_by_level[i], compressed_bytes_by_level[i] as f64 / 1024.0 / 1024.0);
    }
}
