
import sys

file_path = '/Users/lars/Documents/GitHub/voxelot/src/bin/voxelot.rs'

with open(file_path, 'r') as f:
    content = f.read()

start_marker = '// Draw meshed chunks first'
end_marker = 'draw_calls += 1;\n            }'

# Find the start marker
start_idx = content.find(start_marker)
if start_idx == -1:
    print("Start marker not found")
    sys.exit(1)

# Find the instances marker after start_idx
instances_marker = 'if !instances.is_empty() {'
instances_idx = content.find(instances_marker, start_idx)
if instances_idx == -1:
    print("Instances marker not found")
    sys.exit(1)

# Find the end marker after instances_idx
end_idx = content.find(end_marker, instances_idx)
if end_idx == -1:
    print("End marker not found")
    sys.exit(1)

end_idx += len(end_marker)

new_content = """            // Draw meshed chunks first
            if let Some(mesh_indirect) = &self.mesh_indirect_buffer {
                render_pass.set_pipeline(self.mesh_pipeline.as_ref().unwrap());
                render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
                let mut drawn_meshes = 0;
                
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
                        render_pass.set_vertex_buffer(0, entry.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            entry.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed_indirect(mesh_indirect, (i * 20) as u64);
                        draw_calls += 1;
                        entry.last_used_frame = self.frame_index;
                        drawn_meshes += 1;
                     }
                }
                if cfg!(feature = "viewer-debug") && self.frame_count == 0 {
                    viewer_debug!(
                        "DEBUG: Drew {} meshes (indirect)",
                        drawn_meshes
                    );
                }
            }

            // Draw remaining instanced cubes
            render_pass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            render_pass.set_vertex_buffer(0, self.cube_vertex_buffer.as_ref().unwrap().slice(..));
            if let Some(fallback_indirect) = &self.fallback_indirect_buffer {
                render_pass.set_vertex_buffer(1, self.fallback_instance_buffer.as_ref().unwrap().slice(..));
                render_pass.draw_indirect(fallback_indirect, 0);
                draw_calls += 1;
            }"""

final_content = content[:start_idx] + new_content + content[end_idx:]

with open(file_path, 'w') as f:
    f.write(final_content)

print("Successfully replaced render pass block")
