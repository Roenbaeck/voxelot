#!/usr/bin/env python3
"""
Python implementation of the .oct (octree) voxel format writer.

Format specification:
- Header: depth (u8)
- Root chunk recursively encoded

Chunk encoding:
- Position count (u16, max 4096 for 16³ chunk)
- For each occupied position:
  - Position encoded as u16 (z * 256 + y * 16 + x)
  - If type == 0: sub-chunk follows (recursively encoded)
  - Otherwise: solid voxel type (1-254)
"""

import struct
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Represents a 16³ chunk that can contain solid voxels or sub-chunks."""
    voxels: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    sub_chunks: Dict[Tuple[int, int, int], 'Chunk'] = field(default_factory=dict)
    
    def set_voxel(self, x: int, y: int, z: int, voxel_type: int):
        """Set a solid voxel at position (x, y, z) within this chunk."""
        assert 0 <= x < 16 and 0 <= y < 16 and 0 <= z < 16
        assert 1 <= voxel_type <= 254
        self.voxels[(x, y, z)] = voxel_type
    
    def get_or_create_subchunk(self, x: int, y: int, z: int) -> 'Chunk':
        """Get or create a sub-chunk at position (x, y, z)."""
        assert 0 <= x < 16 and 0 <= y < 16 and 0 <= z < 16
        key = (x, y, z)
        if key not in self.sub_chunks:
            self.sub_chunks[key] = Chunk()
        return self.sub_chunks[key]
    
    def write(self, f) -> None:
        """Write this chunk to a file in octree format."""
        # Combine voxels and sub-chunks into occupied positions
        occupied = {}
        for pos, vtype in self.voxels.items():
            occupied[pos] = ('voxel', vtype)
        for pos, chunk in self.sub_chunks.items():
            occupied[pos] = ('chunk', chunk)
        
        # Write count of occupied positions
        count = len(occupied)
        f.write(struct.pack('<H', count))  # u16 little-endian
        
        # Write each position and its data
        for (x, y, z), data in sorted(occupied.items()):
            # Encode position as u16 (z * 256 + y * 16 + x)
            pos_encoded = z * 256 + y * 16 + x
            f.write(struct.pack('<H', pos_encoded))
            
            if data[0] == 'voxel':
                # Solid voxel - write type byte
                f.write(struct.pack('B', data[1]))
            else:
                # Sub-chunk - write 0 then recursively write the chunk
                f.write(struct.pack('B', 0))
                data[1].write(f)


class OctreeWorld:
    """Represents a hierarchical octree world with configurable depth."""
    
    def __init__(self, depth: int):
        """
        Create a world with the specified hierarchy depth.
        
        World size will be 16^depth units per side:
        - depth 1: 16 units (single chunk)
        - depth 2: 256 units
        - depth 3: 4,096 units
        - depth 4: 65,536 units
        """
        assert 1 <= depth <= 10, "Depth must be between 1 and 10"
        self.depth = depth
        self.root = Chunk()
        self.world_size = 16 ** depth
        print(f"Created octree world: {self.world_size} units per side ({depth} levels deep)")
    
    def set(self, x: int, y: int, z: int, voxel_type: int):
        """Set a voxel at world position (x, y, z)."""
        # Check bounds
        if not (0 <= x < self.world_size and 
                0 <= y < self.world_size and 
                0 <= z < self.world_size):
            return  # Silently ignore out of bounds
        
        # Navigate through hierarchy
        if self.depth == 1:
            # Special case: single-level world, root IS the leaf chunk
            self.root.set_voxel(x, y, z, voxel_type)
            return
        
        # Navigate to the correct position through the hierarchy
        current_chunk = self.root
        divisor = self.world_size // 16  # Start from top level
        
        for level in range(self.depth - 1):
            # Calculate which octant of the current chunk
            local_x = (x // divisor) % 16
            local_y = (y // divisor) % 16
            local_z = (z // divisor) % 16
            
            # Get or create sub-chunk at this position
            current_chunk = current_chunk.get_or_create_subchunk(local_x, local_y, local_z)
            
            # Move to next level
            divisor //= 16
        
        # At leaf level, set the voxel
        leaf_x = x % 16
        leaf_y = y % 16
        leaf_z = z % 16
        current_chunk.set_voxel(leaf_x, leaf_y, leaf_z, voxel_type)
    
    def save(self, filename: str):
        """Save the world to an .oct file."""
        with open(filename, 'wb') as f:
            # Write depth byte
            f.write(struct.pack('B', self.depth))
            
            # Write root chunk
            self.root.write(f)
        
        print(f"Saved world to {filename}")


def calculate_required_depth(max_coord: int) -> int:
    """Calculate the minimum depth needed to contain the given maximum coordinate."""
    import math
    # World size is 16^depth, so depth = ceil(log16(max_coord + 1))
    depth = max(1, math.ceil(math.log(max_coord + 1) / math.log(16)))
    return depth


if __name__ == "__main__":
    # Simple test
    print("Testing octree writer...")
    
    # Create a small test world
    world = OctreeWorld(depth=3)  # 4096³
    
    # Add some test voxels
    for x in range(10):
        for z in range(10):
            world.set(x, 0, z, 1)  # Ground plane
    
    # Add a tower
    for y in range(20):
        world.set(5, y, 5, 2)
    
    # Save
    world.save("test_world.oct")
    
    import os
    size = os.path.getsize("test_world.oct")
    print(f"File size: {size} bytes")
