use std::path::Path;
use voxelot::{load_world_file, save_world_file};

fn main() -> std::io::Result<()> {
    let worlds = ["worlds/world_1.oct", "worlds/world_2.oct"];
    for p in worlds {
        let path = Path::new(p);
        if !path.exists() {
            eprintln!("Skipping {}, file not found", p);
            continue;
        }
        println!("Loading {} (zstd)", p);
        let world = load_world_file(path)?; // expect zstd compressed files
        println!("Saving compressed {} (zstd)", p);
        save_world_file(&world, path, true)?; // overwrite in-place
        println!("Saved {} (compressed)", p);
    }
    Ok(())
}
