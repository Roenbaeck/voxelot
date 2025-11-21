
import sys

file_path = '/Users/lars/Documents/GitHub/voxelot/src/bin/voxelot.rs'

with open(file_path, 'r') as f:
    content = f.read()

old_code = """                     let entry = if use_envelope {
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
                     }"""

new_code = """                     let entry = if use_envelope {
                        self.envelope_mesh_cache.get_mut(&key).or_else(|| self.mesh_cache.get_mut(&key))
                     } else {
                        self.mesh_cache.get_mut(&key).or_else(|| self.envelope_mesh_cache.get_mut(&key))
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
                     }"""

if old_code not in content:
    print("Old code not found")
    sys.exit(1)

new_content = content.replace(old_code, new_code)

with open(file_path, 'w') as f:
    f.write(new_content)

print("Successfully replaced get with get_mut")
