use voxelot::{Chunk, Palette};

fn main() {
    println!("Testing Bounding Box Calculation...");

    // Create a dummy palette
    let palette = Palette::from_string("1 255 255 255 255").unwrap();

    // Test 1: Column in corner (0,0,0) to (0,15,0)
    let mut chunk = Chunk::new();
    for y in 0..16 {
        chunk.set(0, y, 0, 1);
    }
    chunk.update_lod_metadata(&palette);
    let bbox = chunk.bounding_box.expect("Should have bbox");
    println!("Test 1 (Column 0,0,0): {:?}", bbox);
    assert_eq!(bbox, [0, 0, 0, 0, 15, 0], "Column should be tight");

    // Test 2: Two columns (0,0,0) and (15,0,0) - "U shape" or parallel walls
    let mut chunk = Chunk::new();
    for y in 0..16 {
        chunk.set(0, y, 0, 1);
        chunk.set(15, y, 0, 1);
    }
    chunk.update_lod_metadata(&palette);
    let bbox = chunk.bounding_box.expect("Should have bbox");
    println!("Test 2 (Columns 0 and 15): {:?}", bbox);
    assert_eq!(
        bbox,
        [0, 0, 0, 15, 15, 0],
        "Bbox should cover both columns (filling the gap)"
    );

    // Test 3: L-shape (Column at 0,0,0 and Row at 0,0,0..15)
    let mut chunk = Chunk::new();
    for y in 0..16 {
        chunk.set(0, y, 0, 1);
    }
    for x in 0..16 {
        chunk.set(x, 0, 0, 1);
    }
    chunk.update_lod_metadata(&palette);
    let bbox = chunk.bounding_box.expect("Should have bbox");
    println!("Test 3 (L-shape): {:?}", bbox);
    assert_eq!(
        bbox,
        [0, 0, 0, 15, 15, 0],
        "Bbox should cover the L (filling the corner)"
    );

    // Test 4: Single voxel at (7,7,7)
    let mut chunk = Chunk::new();
    chunk.set(7, 7, 7, 1);
    chunk.update_lod_metadata(&palette);
    let bbox = chunk.bounding_box.expect("Should have bbox");
    println!("Test 4 (Single voxel): {:?}", bbox);
    assert_eq!(bbox, [7, 7, 7, 7, 7, 7], "Bbox should be single point");

    // Test 5: Sub-chunk logic
    // Parent chunk with sub-chunk at (0,0,0) that is non-empty
    let mut parent = Chunk::new();
    let mut child = Chunk::new();
    child.set(5, 5, 5, 1); // Child has something
    child.update_lod_metadata(&palette);

    parent.set_chunk(0, 0, 0, child);
    parent.update_lod_metadata(&palette);
    let bbox = parent.bounding_box.expect("Should have bbox");
    println!("Test 5 (Parent with sub-chunk at 0,0,0): {:?}", bbox);
    assert_eq!(
        bbox,
        [0, 0, 0, 0, 0, 0],
        "Parent bbox should be 0,0,0 (one child unit)"
    );
    println!("Parent voxel count: {}", parent.voxel_count);
    assert!(parent.voxel_count > 0, "Parent voxel count should be > 0");
    assert_eq!(
        parent.voxel_count, 1,
        "Parent voxel count should be 1 (from child)"
    );

    println!("All tests passed!");
}
