use voxelot::VoxelWorld;

fn main() {
    println!("Voxelot demo");
    let mut world = VoxelWorld::new();

    let samples = [
        (0i64, 0i64, 0i64, 1u16),
        (127, 127, 127, 2),
        (128, 0, 0, 3),                // crosses into next chunk on x (if VOXEL_BASE=128)
        ((voxelot::VOXELS_PER_REGION as i64), 0, 0, 4), // crosses into next region on x
        (-1, -1, -1, 5),               // negative coords
    ];

    for (x, y, z, t) in samples {
        world.set_voxel(x, y, z, t);
    }

    println!("Regions: {}", world.region_count());
    println!("Chunks:  {}", world.total_chunks());
    println!("Voxels:  {}", world.total_voxels());

    for (x, y, z, t) in samples {
        let got = world.get_voxel(x, y, z);
        println!("get({x},{y},{z}) = {:?} (expected {t})", got);
    }
}
