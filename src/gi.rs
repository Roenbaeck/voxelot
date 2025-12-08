use crate::lib_hierarchical::{Chunk, Voxel, World, WorldPos};
use crate::palette::Palette;
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};
use rayon::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GiProbe {
    /// Probe position in world space (w unused/padding)
    pub position: [f32; 4],
    /// Irradiance for 6 faces: +X, -X, +Y, -Y, +Z, -Z
    /// Each is RGBA (A unused/intensity)
    pub light_data: [[f32; 4]; 6],
}

impl Default for GiProbe {
    fn default() -> Self {
        Self {
            position: [0.0; 4],
            light_data: [[0.0; 4]; 6],
        }
    }
}

pub struct GiSystem {
    pub probes: Vec<GiProbe>,
    pub grid_origin: IVec3, // In chunks (chunk_coord)
    pub grid_dims: IVec3,   // Dimensions in chunks
    pub force_update: bool,
    pub light_sources: Vec<(Vec3, Vec3)>,
}

impl GiSystem {
    pub fn new(dims: IVec3) -> Self {
        let count = (dims.x * dims.y * dims.z) as usize;
        Self {
            probes: vec![GiProbe::default(); count],
            grid_origin: IVec3::new(0, 0, 0),
            grid_dims: dims,
            force_update: true,
            light_sources: Vec::new(),
        }
    }

    fn scan_lights(&mut self, world: &World, palette: &Palette, min: IVec3, max: IVec3) {
        for z in min.z..max.z {
            for y in min.y..max.y {
                for x in min.x..max.x {
                    let origin = WorldPos::new(x as i64 * 16, y as i64 * 16, z as i64 * 16);

                    if let Some(chunk) = world.get_leaf_chunk_at_origin(origin) {
                        if chunk.emissive_power > 0.0 {
                            for idx in chunk.presence.iter() {
                                let (lx, ly, lz) = Chunk::unflatten(idx);
                                let rank = chunk.presence.rank(idx) as usize;
                                if let Some(Voxel::Solid(v_type)) = chunk.voxels.get(rank - 1) {
                                    let (e_color, e_strength) = palette.emissive(*v_type as u32);
                                    if e_strength > 0.0 {
                                        let pos = Vec3::new(
                                            (origin.x + lx as i64) as f32 + 0.5,
                                            (origin.y + ly as i64) as f32 + 0.5,
                                            (origin.z + lz as i64) as f32 + 0.5,
                                        );
                                        self.light_sources.push((pos, Vec3::from(e_color) * e_strength));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Update probes based on camera position and world state
    pub fn update(&mut self, world: &World, palette: &Palette, camera_pos: Vec3) {
        // 1. Determine new grid origin (centered on camera, snapped to chunk size)
        let chunk_size = 16.0;
        let cam_chunk = (camera_pos / chunk_size).floor().as_ivec3();
        let half_dims = self.grid_dims / 2;
        let new_origin = cam_chunk - half_dims;

        let shift = new_origin - self.grid_origin;

        if shift == IVec3::ZERO && !self.force_update {
            return;
        }
        
        let force_update = self.force_update;
        self.force_update = false;

        // 2. Update light sources incrementally
        let padding = 4;
        let scan_min = new_origin - IVec3::splat(padding);
        let scan_max = new_origin + self.grid_dims + IVec3::splat(padding);

        // Remove lights outside new range
        let min_pos = scan_min.as_vec3() * 16.0;
        let max_pos = scan_max.as_vec3() * 16.0;
        
        self.light_sources.retain(|(pos, _)| {
            pos.x >= min_pos.x && pos.x < max_pos.x &&
            pos.y >= min_pos.y && pos.y < max_pos.y &&
            pos.z >= min_pos.z && pos.z < max_pos.z
        });

        if force_update {
            self.light_sources.clear();
            self.scan_lights(world, palette, scan_min, scan_max);
        } else {
            let old_origin = self.grid_origin;
            let old_scan_min = old_origin - IVec3::splat(padding);
            let old_scan_max = old_origin + self.grid_dims + IVec3::splat(padding);
            
            let x_range = if shift.x > 0 { old_scan_max.x..scan_max.x } else { scan_min.x..old_scan_min.x };
            let y_range = if shift.y > 0 { old_scan_max.y..scan_max.y } else { scan_min.y..old_scan_min.y };
            let z_range = if shift.z > 0 { old_scan_max.z..scan_max.z } else { scan_min.z..old_scan_min.z };
            
            if !x_range.is_empty() {
                self.scan_lights(world, palette, 
                    IVec3::new(x_range.start, scan_min.y, scan_min.z),
                    IVec3::new(x_range.end, scan_max.y, scan_max.z));
            }
            
            let intersect_x_start = scan_min.x.max(old_scan_min.x);
            let intersect_x_end = scan_max.x.min(old_scan_max.x);
            
            if !y_range.is_empty() && intersect_x_start < intersect_x_end {
                self.scan_lights(world, palette,
                    IVec3::new(intersect_x_start, y_range.start, scan_min.z),
                    IVec3::new(intersect_x_end, y_range.end, scan_max.z));
            }
            
            let intersect_y_start = scan_min.y.max(old_scan_min.y);
            let intersect_y_end = scan_max.y.min(old_scan_max.y);
            
            if !z_range.is_empty() && intersect_x_start < intersect_x_end && intersect_y_start < intersect_y_end {
                self.scan_lights(world, palette,
                    IVec3::new(intersect_x_start, intersect_y_start, z_range.start),
                    IVec3::new(intersect_x_end, intersect_y_end, z_range.end));
            }
        }

        // 3. Update probes with scrolling
        let dims = self.grid_dims;
        let count = (dims.x * dims.y * dims.z) as usize;
        let mut new_probes = vec![GiProbe::default(); count];
        let mut probes_to_compute = Vec::new();

        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let new_idx = (x + y * dims.x + z * dims.x * dims.y) as usize;
                    
                    // Calculate old coordinates
                    let old_x = x + shift.x;
                    let old_y = y + shift.y;
                    let old_z = z + shift.z;

                    let mut reused = false;
                    if !force_update &&
                       old_x >= 0 && old_x < dims.x &&
                       old_y >= 0 && old_y < dims.y &&
                       old_z >= 0 && old_z < dims.z {
                        
                        let old_idx = (old_x + old_y * dims.x + old_z * dims.x * dims.y) as usize;
                        new_probes[new_idx] = self.probes[old_idx];
                        reused = true;
                    }

                    if !reused {
                        probes_to_compute.push((new_idx, x, y, z));
                    }
                }
            }
        }

        // Compute new probes in parallel
        let light_sources = &self.light_sources;

        let computed_data: Vec<(usize, GiProbe)> = probes_to_compute
            .par_iter()
            .map(|&(idx, px, py, pz)| {
                let cx = new_origin.x + px;
                let cy = new_origin.y + py;
                let cz = new_origin.z + pz;

                // Probe is at center of chunk
                let probe_pos = Vec3::new(
                    (cx as f32 * 16.0) + 8.0,
                    (cy as f32 * 16.0) + 8.0,
                    (cz as f32 * 16.0) + 8.0,
                );

                // Check if probe is inside a solid voxel
                let wp = WorldPos::new(probe_pos.x.floor() as i64, probe_pos.y.floor() as i64, probe_pos.z.floor() as i64);
                let chunk_origin = WorldPos::new(wp.x & !15, wp.y & !15, wp.z & !15);
                let mut is_buried = false;
                if let Some(chunk) = world.get_leaf_chunk_at_origin(chunk_origin) {
                     let lx = (wp.x & 15) as u8;
                     let ly = (wp.y & 15) as u8;
                     let lz = (wp.z & 15) as u8;
                     if chunk.contains(lx, ly, lz) {
                         is_buried = true;
                     }
                }

                if is_buried {
                    return (idx, GiProbe::default());
                }

                let mut probe = GiProbe::default();
                probe.position = [probe_pos.x, probe_pos.y, probe_pos.z, 1.0];
                probe.light_data = [[0.0; 4]; 6];

                for (light_pos, light_color) in light_sources {
                    let dir = *light_pos - probe_pos;
                    let dist_sq = dir.length_squared();
                    
                    // Max radius check (e.g., 64 units = 4 chunks)
                    if dist_sq > 0.001 && dist_sq < 64.0 * 64.0 {
                        // Check occlusion for each face individually
                        // Instead of checking from the center of the chunk (which might be occluded),
                        // we check from the center of each face that is facing the light.
                        let faces = [
                            (Vec3::X, 0), (Vec3::NEG_X, 1),
                            (Vec3::Y, 2), (Vec3::NEG_Y, 3),
                            (Vec3::Z, 4), (Vec3::NEG_Z, 5),
                        ];

                        for (face_normal, face_idx) in faces {
                            // Calculate the center point of this face
                            let face_pos = probe_pos + face_normal * 8.0;
                            
                            // Recalculate direction and distance from the face center
                            let face_to_light = *light_pos - face_pos;
                            let face_dist_sq = face_to_light.length_squared();
                            let face_dist = face_dist_sq.sqrt();
                            let face_dir_norm = face_to_light / face_dist;

                            // Check if light is in front of the face
                            let dot = face_normal.dot(face_dir_norm).max(0.0);
                            
                            if dot > 0.0 {
                                // Trace from the face center towards the light
                                // Reduce distance slightly to avoid hitting the light source voxel itself
                                let trace_dist = (face_dist - 1.0).max(0.0);
                                
                                if trace_dist > 0.0 && !trace_ray(world, face_pos, face_dir_norm, trace_dist) {
                                    // Inverse square falloff (using distance from face)
                                    let contribution = *light_color * dot / (1.0 + face_dist_sq);
                                    
                                    probe.light_data[face_idx][0] += contribution.x;
                                    probe.light_data[face_idx][1] += contribution.y;
                                    probe.light_data[face_idx][2] += contribution.z;
                                }
                            }
                        }
                    }
                }
                (idx, probe)
            })
            .collect();

        // Apply computed probes
        for (idx, probe) in computed_data {
            new_probes[idx] = probe;
        }

        self.probes = new_probes;
        self.grid_origin = new_origin;
    }
}

fn trace_ray(world: &World, start: Vec3, dir: Vec3, max_dist: f32) -> bool {
    let step_size = 0.5; // Check every half voxel unit for better accuracy
    let steps = (max_dist / step_size).ceil() as usize;
    let mut pos = start;
    
    // Offset start slightly to avoid self-occlusion
    pos += dir * 0.1;

    for _ in 0..steps {
        pos += dir * step_size;
        if (pos - start).length_squared() > max_dist * max_dist {
            break;
        }

        let wp = WorldPos::new(pos.x.floor() as i64, pos.y.floor() as i64, pos.z.floor() as i64);
        
        let chunk_origin = WorldPos::new(wp.x & !15, wp.y & !15, wp.z & !15);
        
        if let Some(chunk) = world.get_leaf_chunk_at_origin(chunk_origin) {
             let lx = (wp.x & 15) as u8;
             let ly = (wp.y & 15) as u8;
             let lz = (wp.z & 15) as u8;
             
             if chunk.contains(lx, ly, lz) {
                 return true;
             }
        }
    }
    false
}
