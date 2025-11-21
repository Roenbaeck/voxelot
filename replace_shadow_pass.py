
import sys

file_path = '/Users/lars/Documents/GitHub/voxelot/src/bin/voxelot.rs'

with open(file_path, 'r') as f:
    content = f.read()

start_marker = 'if has_meshes_to_draw {'
end_marker = 'draw_calls += 1;\n            }'

# Find the first occurrence of start_marker (which is in shadow pass)
start_idx = content.find(start_marker)
if start_idx == -1:
    print("Start marker not found")
    sys.exit(1)

# Find the end of the shadow pass block (it ends with the instances block)
# The instances block ends with `draw_calls += 1;` and `}`
# I need to be careful because `draw_calls += 1` appears multiple times.
# The block I want to replace ends with the `if !instances.is_empty()` block.

# Let's find the `if !instances.is_empty()` block after start_idx
instances_marker = 'if !instances.is_empty() {'
instances_idx = content.find(instances_marker, start_idx)
if instances_idx == -1:
    print("Instances marker not found")
    sys.exit(1)

# Find the end of that block
end_idx = content.find(end_marker, instances_idx)
if end_idx == -1:
    print("End marker not found")
    sys.exit(1)

end_idx += len(end_marker)

new_content = """            if let Some(mesh_indirect) = &self.mesh_indirect_buffer {
                 shadow_pass.set_pipeline(shadow_mesh_pipeline);
                 shadow_pass.set_bind_group(0, shadow_bind_group, &[]);
                 
                 for (i, v) in visible.iter().enumerate() {
                     if !v.is_leaf_chunk { continue; }
                     let key = (v.position[0], v.position[1], v.position[2]);
                     
                     let cam_pos = self.camera_controller.camera.position;
                     let chunk_center = [key.0 as f32 + 8.0, key.1 as f32 + 8.0, key.2 as f32 + 8.0];
                     let dx = chunk_center[0] - cam_pos[0];
                     let dy = chunk_center[1] - cam_pos[1];
                     let dz = chunk_center[2] - cam_pos[2];
                     let dist_sq = dx * dx + dy * dy + dz * dz;
                     let envelope_dist_sq = self.envelope_distance * self.envelope_distance;
                     let use_envelope = dist_sq > envelope_dist_sq;

                     let entry = if use_envelope {
                        self.envelope_mesh_cache.get(&key).or_else(|| self.mesh_cache.get(&key))
                     } else {
                        self.mesh_cache.get(&key).or_else(|| self.envelope_mesh_cache.get(&key))
                     };
                     
                     if let Some(entry) = entry {
                         shadow_pass.set_vertex_buffer(0, entry.vertex_buffer.slice(..));
                         shadow_pass.set_index_buffer(entry.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                         shadow_pass.draw_indexed_indirect(mesh_indirect, (i * 20) as u64);
                         draw_calls += 1;
                     }
                 }
            }
            
            if let Some(fallback_indirect) = &self.fallback_indirect_buffer {
                shadow_pass.set_pipeline(shadow_pipeline);
                shadow_pass.set_bind_group(0, shadow_bind_group, &[]);
                shadow_pass.set_vertex_buffer(0, self.cube_vertex_buffer.as_ref().unwrap().slice(..));
                shadow_pass.set_vertex_buffer(1, self.fallback_instance_buffer.as_ref().unwrap().slice(..));
                shadow_pass.draw_indirect(fallback_indirect, 0);
                draw_calls += 1;
            }"""

final_content = content[:start_idx] + new_content + content[end_idx:]

with open(file_path, 'w') as f:
    f.write(final_content)

print("Successfully replaced shadow pass block")
