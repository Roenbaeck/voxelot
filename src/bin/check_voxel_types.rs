use std::collections::HashMap;
use std::env;
use std::path::Path;
use voxelot::{load_world_file, Chunk, Voxel};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let path_str = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("worlds/world_1.vhc");
    let path = Path::new(path_str);
    println!("Loading {}", path.display());
    let world = load_world_file(path)?;

    let mut counts: HashMap<u8, u64> = HashMap::new();
    traverse(world.root(), &mut counts);

    println!("Voxel type counts:");
    let mut keys: Vec<u8> = counts.keys().cloned().collect();
    keys.sort();
    for k in keys {
        println!("Type {}: {}", k, counts[&k]);
    }

    Ok(())
}

fn traverse(chunk: &Chunk, counts: &mut HashMap<u8, u64>) {
    for (_, voxel) in chunk.iter() {
        match voxel {
            Voxel::Solid(t) => {
                *counts.entry(*t).or_insert(0) += 1;
            }
            Voxel::Chunk(sub) => {
                traverse(sub, counts);
            }
        }
    }
}
