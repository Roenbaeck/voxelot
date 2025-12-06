use std::path::Path;
use voxelot::{file_format::save_world_file, load_world_file};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: resave_world <input> <output>");
        std::process::exit(1);
    }
    let in_path = Path::new(&args[1]);
    let out_path = Path::new(&args[2]);

    println!("Loading {}...", in_path.display());
    let world = load_world_file(in_path)?;
    println!("Saving {}...", out_path.display());
    save_world_file(&world, out_path, true)?;
    println!("Done.");
    Ok(())
}
