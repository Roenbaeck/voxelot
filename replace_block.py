
import sys

file_path = '/Users/lars/Documents/GitHub/voxelot/src/bin/voxelot.rs'

with open(file_path, 'r') as f:
    content = f.read()

start_marker = '// Convert remaining instances (exclude those belonging to meshed chunks)'
end_marker = '(out, cpu_mesh_keys.clone())'
end_marker_full = '            (out, cpu_mesh_keys.clone())\n        };'

start_idx = content.find(start_marker)
if start_idx == -1:
    print("Start marker not found")
    sys.exit(1)

end_idx = content.find(end_marker_full, start_idx)
if end_idx == -1:
    print("End marker not found")
    # Try without indentation
    end_marker_full = '(out, cpu_mesh_keys.clone())\n        };'
    end_idx = content.find(end_marker_full, start_idx)
    if end_idx == -1:
         print("End marker not found (2)")
         sys.exit(1)

end_idx += len(end_marker_full)

new_content = """// Convert remaining instances (exclude those belonging to meshed chunks)
        let instance_start = Instant::now();
        
        let mut draw_mesh_keys = HashSet::new();
        for v in &visible {
             if v.is_leaf_chunk {
                let key = (v.position[0], v.position[1], v.position[2]);
                if cpu_mesh_keys.contains(&key) {
                    draw_mesh_keys.insert(key);
                }
             }
        }
        let instances: Vec<VoxelInstanceRaw> = Vec::new();"""

final_content = content[:start_idx] + new_content + content[end_idx:]

with open(file_path, 'w') as f:
    f.write(final_content)

print("Successfully replaced block")
