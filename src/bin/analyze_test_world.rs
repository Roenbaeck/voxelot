use std::path::PathBuf;
use voxelot::{file_format::load_world_file, WorldPos};

fn main() {
    let path = PathBuf::from("test_linear_hill.vhc");
    println!("Loading {}...", path.display());
    let world = load_world_file(&path).expect("Failed to load world");
    let mut heights = Vec::new();
    let grid = 1024;
    let z = grid / 2; // center row
    for x in 0..grid {
        let mut top = -1i64;
        for y in (0..=200).rev() {
            if world.get(WorldPos::new(x as i64, y, z as i64)).is_some() {
                top = y;
                break;
            }
        }
        heights.push(top);
    }
    // Print first 100 heights and run-length encode
    println!("Heights (first 120):");
    for i in 0..120.min(heights.len()) {
        print!("{:3}", heights[i]);
        if (i + 1) % 20 == 0 { println!(""); }
    }
    println!("\nRun-length encoding of diffs (0 means equal):");
    // compute diffs and runs of equal increments
    let mut runs = Vec::new();
    let mut cur_run = (heights[0], 1usize);
    for i in 1..heights.len() {
        if heights[i] == cur_run.0 {
            cur_run.1 += 1;
        } else {
            runs.push(cur_run);
            cur_run = (heights[i], 1);
        }
    }
    runs.push(cur_run);
    println!("Total runs: {}", runs.len());
    for (i, (val, count)) in runs.iter().enumerate().take(60) {
        println!("{:#04}: val={} count={}", i, val, count);
    }
    // Summarize run lengths
    use std::collections::HashMap;
    let mut hist: HashMap<usize, usize> = HashMap::new();
    for (_, count) in runs.iter() { *hist.entry(*count).or_default() += 1; }
    let mut keys: Vec<_> = hist.keys().cloned().collect();
    keys.sort();
    println!("Run length histogram:");
    for k in keys { println!("{} -> {}", k, hist[&k]); }
}
