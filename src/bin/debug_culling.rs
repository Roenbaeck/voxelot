//! Debug hierarchical culling

use voxelot::{World, WorldPos, Camera, Voxel};

fn main() {
    println!("=== Debug Hierarchical Culling ===\n");
    
    let mut world = World::new(2); // 256 units per side
    
    // Create and subdivide a voxel
    world.set(WorldPos::new(0, 0, 0), 1);
    world.subdivide_at(WorldPos::new(0, 0, 0)).unwrap();
    
    // Verify what we have
    let root = world.root();
    println!("Root chunk has {} voxels", root.count());
    
    if let Some(voxel) = root.get(0, 0, 0) {
        match voxel {
            Voxel::Chunk(sub) => {
                    println!("Voxel at (0,0,0) is a Chunk with {} voxels", sub.count());
                    
                    // Check a few voxels in the sub-chunk
                    println!("\nSample voxels in sub-chunk:");
                    for x in 0..3 {
                        for y in 0..3 {
                            for z in 0..3 {
                                if let Some(v) = sub.get(x, y, z) {
                                    match v {
                                        Voxel::Solid(t) => println!("  ({},{},{}) = Solid({})", x, y, z, t),
                                        Voxel::Chunk(_) => println!("  ({},{},{}) = Chunk", x, y, z),
                                    }
                                }
                            }
                        }
                    }
                }
                _ => println!("Voxel is not a chunk!"),
            }
        }
    }
    
    // Simple camera pointing at origin
    println!("\nTesting simple camera:");
    let camera = Camera::new(
        [25.0, 25.0, 25.0],
        [-1.0, -1.0, -1.0],  // Direction towards origin (will be normalized)
        [0.0, 1.0, 0.0],
    );
    
    println!("Camera position: {:?}", camera.position);
    println!("Camera forward: {:?}", camera.forward);
    
    // Test is_in_front for some positions
    println!("\nTesting is_in_front for positions:");
    for pos in [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]] {
        let in_front = camera.is_in_front(pos);
        let distance = camera.distance_to(pos);
        println!("  {:?}: in_front={}, distance={:.1}", pos, in_front, distance);
    }
    
    // Now test culling
    println!("\nRunning culling:");
    let visible = voxelot::cull_visible_voxels(&world, &camera);
    println!("Found {} visible voxels", visible.len());
    
    if visible.len() > 0 {
        println!("\nFirst 5 visible voxels:");
        for (i, v) in visible.iter().take(5).enumerate() {
            println!("  {}: pos={:?}, type={}, dist={:.1}", i, v.position, v.voxel_type, v.distance);
        }
    } else {
        println!("\nNo voxels found - debugging needed!");
    }
}
