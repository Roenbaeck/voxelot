use crate::lib_hierarchical::{World, WorldPos};
use crate::palette::Palette;
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

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

    // Caches
    probe_cache: HashMap<IVec3, GiProbe>,
    // Chunk coordinate -> List of emissive voxels (position, intensity)
    light_cache: HashMap<IVec3, Vec<(Vec3, Vec3)>>,
}

impl GiSystem {
    pub fn new(dims: IVec3) -> Self {
        let count = (dims.x * dims.y * dims.z) as usize;
        Self {
            probes: vec![GiProbe::default(); count],
            grid_origin: IVec3::new(0, 0, 0),
            grid_dims: dims,
            probe_cache: HashMap::new(),
            light_cache: HashMap::new(),
        }
    }

    /// Update probes based on camera position and world state
    pub fn update(&mut self, world: &World, palette: &Palette, camera_pos: Vec3) {
        // 1. Determine new grid origin (centered on camera, snapped to chunk size)
        let chunk_size = 16.0;
        let cam_chunk = (camera_pos / chunk_size).floor().as_ivec3();
        let half_dims = self.grid_dims / 2;
        let new_origin = cam_chunk - half_dims;

        self.grid_origin = new_origin;

        // Define the active grid area
        // let grid_min = new_origin;
        // let grid_max = new_origin + self.grid_dims;

        // 2. Identify missing probes in the active area
        let mut missing_probes = Vec::new();
        for z in 0..self.grid_dims.z {
            for y in 0..self.grid_dims.y {
                for x in 0..self.grid_dims.x {
                    let coord = new_origin + IVec3::new(x, y, z);
                    if !self.probe_cache.contains_key(&coord) {
                        missing_probes.push(coord);
                    }
                }
            }
        }

        // If no probes are missing, we just need to update the flat buffer and return.
        // However, we should also check if we need to load lights for new areas.
        // For simplicity, we drive light loading by probe requirements.
        
        if !missing_probes.is_empty() {
            // 3. Identify required light chunks for the missing probes
            // We need lights from neighbors. Let's say radius is 4 chunks.
            let light_radius = 4;
            let mut required_light_chunks = HashSet::new();
            
            for probe_coord in &missing_probes {
                for z in -light_radius..=light_radius {
                    for y in -light_radius..=light_radius {
                        for x in -light_radius..=light_radius {
                            let light_coord = *probe_coord + IVec3::new(x, y, z);
                            if !self.light_cache.contains_key(&light_coord) {
                                required_light_chunks.insert(light_coord);
                            }
                        }
                    }
                }
            }

            // 4. Compute missing light chunks in parallel
            let new_lights: Vec<(IVec3, Vec<(Vec3, Vec3)>)> = required_light_chunks
                .into_par_iter()
                .map(|chunk_coord| {
                    let mut lights = Vec::new();
                    let origin = WorldPos::new(
                        chunk_coord.x as i64 * 16,
                        chunk_coord.y as i64 * 16,
                        chunk_coord.z as i64 * 16,
                    );

                    if let Some(chunk) = world.get_leaf_chunk_at_origin(origin) {
                        if chunk.emissive_power > 0.0 {
                            for lz in 0..16 {
                                for ly in 0..16 {
                                    for lx in 0..16 {
                                        if chunk.contains(lx, ly, lz) {
                                            if let Some(vtype) = chunk.get_type(lx, ly, lz) {
                                                let (color, intensity) = palette.emissive(vtype as u32);
                                                if intensity > 0.0 {
                                                    let emission = Vec3::from(color) * intensity * 10.0;
                                                    let voxel_pos = Vec3::new(
                                                        (chunk_coord.x as f32 * 16.0) + lx as f32 + 0.5,
                                                        (chunk_coord.y as f32 * 16.0) + ly as f32 + 0.5,
                                                        (chunk_coord.z as f32 * 16.0) + lz as f32 + 0.5,
                                                    );
                                                    lights.push((voxel_pos, emission));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    (chunk_coord, lights)
                })
                .collect();

            // Update light cache
            for (coord, lights) in new_lights {
                self.light_cache.insert(coord, lights);
            }

            // 5. Compute missing probes in parallel
            // We need to pass a read-only view of the light cache or clone relevant parts.
            // Since HashMap isn't Sync for random access during par_iter without RwLock,
            // and we don't want to lock per probe, we can:
            // A) Collect all relevant lights for each probe (lots of copying)
            // B) Use a thread-safe map (e.g. DashMap) - but I can't add dependencies easily.
            // C) Just collect the lights needed for the *batch* of missing probes?
            // D) Since we are in a mutable method, we can't share `self` immutably with par_iter.
            
            // Let's go with A for now, but optimized:
            // We can flatten the relevant lights into a spatial structure or just pass the map if we use a standard iterator?
            // No, standard iterator is single threaded.
            // We want parallel.
            
            // Workaround: Extract the relevant subset of light_cache into a `Arc<HashMap>` or similar?
            // Or just collect all lights in the active area into a flat Vec?
            // If the active area is small, a flat Vec of all lights might be faster than hash lookups anyway.
            
            // Let's collect ALL lights in the grid+radius area into a flat list for the calculation.
            // This is O(TotalLights), but TotalLights in the active area might be manageable.
            // If we have 10k lights, it's fine.
            
            // Optimization: Spatial hashing is better.
            // Let's just use the `light_cache` but we need to access it from threads.
            // We can wrap `light_cache` in a RwLock or just clone the keys/values we need?
            // Cloning `Vec<(Vec3, Vec3)>` is cheap if they are small.
            
            // Let's try this:
            // 1. Collect all `(IVec3, Vec<(Vec3, Vec3)>)` pairs that are relevant for the missing probes.
            //    Actually, we can just collect *all* lights in the extended grid area.
            //    It's a bit wasteful but safe.
            
            // Better: Just iterate missing probes, and for each, gather lights from `light_cache` (single threaded)
            // THEN compute the expensive raycasts in parallel.
            
            // Step 5a: Prepare jobs
            let jobs: Vec<(IVec3, Vec<(Vec3, Vec3)>)> = missing_probes.iter().map(|&probe_coord| {
                let mut lights = Vec::new();
                for z in -light_radius..=light_radius {
                    for y in -light_radius..=light_radius {
                        for x in -light_radius..=light_radius {
                            let light_coord = probe_coord + IVec3::new(x, y, z);
                            if let Some(chunk_lights) = self.light_cache.get(&light_coord) {
                                lights.extend_from_slice(chunk_lights);
                            }
                        }
                    }
                }
                (probe_coord, lights)
            }).collect();

            // Step 5b: Execute jobs in parallel
            let new_probes: Vec<(IVec3, GiProbe)> = jobs.into_par_iter().map(|(probe_coord, lights)| {
                let mut probe = GiProbe::default();
                
                let cx = probe_coord.x;
                let cy = probe_coord.y;
                let cz = probe_coord.z;

                let center_pos = Vec3::new(
                    (cx as f32 * 16.0) + 8.0,
                    (cy as f32 * 16.0) + 8.0,
                    (cz as f32 * 16.0) + 8.0,
                );
                
                probe.position = [center_pos.x, center_pos.y, center_pos.z, 1.0];
                
                let normals = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
                let face_offsets = [
                    Vec3::new(7.0, 0.0, 0.0),  // +X (Reduced from 8.0 to avoid boundary issues)
                    Vec3::new(-7.0, 0.0, 0.0), // -X
                    Vec3::new(0.0, 7.0, 0.0),  // +Y
                    Vec3::new(0.0, -7.0, 0.0), // -Y
                    Vec3::new(0.0, 0.0, 7.0),  // +Z
                    Vec3::new(0.0, 0.0, -7.0), // -Z
                ];

                for f in 0..6 {
                    let face_normal = normals[f];
                    let face_center = center_pos + face_offsets[f];
                    
                    // Check if face center is buried
                    let face_wp = WorldPos::new(
                        face_center.x.floor() as i64,
                        face_center.y.floor() as i64,
                        face_center.z.floor() as i64
                    );
                    // Note: We need `world` access here. `world` is &World, which is Sync.
                    if world.get(face_wp).is_some() {
                        continue;
                    }

                    for (light_pos, light_energy) in &lights {
                        let delta = *light_pos - face_center;
                        if delta.dot(face_normal) <= 0.0 { continue; }

                        let dist_sq = delta.length_squared();
                        if dist_sq > 64.0 * 64.0 || dist_sq < 0.01 { continue; }

                        let dist = dist_sq.sqrt();
                        let dir = delta / dist;
                        let cos_theta = dir.dot(face_normal);
                        
                        if !is_visible_dda(world, face_center, *light_pos) {
                            continue;
                        }

                        let attenuation = 1.0 / (1.0 + dist_sq * 0.1); 
                        let contrib = *light_energy * attenuation * cos_theta;
                        
                        probe.light_data[f][0] += contrib.x;
                        probe.light_data[f][1] += contrib.y;
                        probe.light_data[f][2] += contrib.z;
                    }
                }
                (probe_coord, probe)
            }).collect();

            // Update probe cache
            for (coord, probe) in new_probes {
                self.probe_cache.insert(coord, probe);
            }
        }

        // 6. Fill the flat buffer for GPU
        // We iterate the grid dimensions and fetch from cache
        // This is fast enough to do on main thread
        let dims = self.grid_dims;
        let origin = self.grid_origin;
        
        // Resize if needed (shouldn't be if dims constant)
        let total_probes = (dims.x * dims.y * dims.z) as usize;
        if self.probes.len() != total_probes {
            self.probes.resize(total_probes, GiProbe::default());
        }

        // Sequential copy
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let coord = origin + IVec3::new(x, y, z);
                    let idx = (x + y * dims.x + z * dims.x * dims.y) as usize;
                    if let Some(p) = self.probe_cache.get(&coord) {
                        self.probes[idx] = *p;
                    } else {
                        // Should not happen if logic above is correct
                        self.probes[idx] = GiProbe::default();
                    }
                }
            }
        }
        
        // 7. Prune caches (Optional, to keep memory usage bounded)
        // Remove chunks that are far away
        // let prune_dist = 10; // chunks
        // let center = new_origin + half_dims;
        
        // This might be slow if map is huge. Do it occasionally?
        // For now, let's skip or do a simple check.
        // self.probe_cache.retain(|k, _| (*k - center).abs().max_element() < prune_dist);
        // self.light_cache.retain(|k, _| (*k - center).abs().max_element() < prune_dist + 4);
    }
}

/// Robust Voxel Traversal (DDA)
/// Returns true if the ray from p0 to p1 is clear of obstacles.
fn is_visible_dda(world: &World, p0: Vec3, p1: Vec3) -> bool {
    let d = p1 - p0;
    let len = d.length();
    if len < 0.001 {
        return true;
    }
    let dir = d / len;

    // Ray start (nudge slightly to avoid self-intersection)
    let start = p0 + dir * 0.01;
    
    // Ray end: Stop 0.6 units before the light center to avoid hitting the light voxel itself
    // (Light voxel is 1x1x1, center at 0.5, so 0.6 ensures we are outside)
    let end = p1 - dir * 0.6;
    
    let dist = (end - start).length();
    // If dist is negative (start is past end), it means we are inside the light voxel or very close.
    // In that case, we are visible.
    if (end - start).dot(dir) <= 0.0 {
        return true;
    }

    // Simple stepping for now (DDA is tricky to get perfect with floats/hierarchical)
    // Step size = 0.5 units ensures we don't miss 1x1x1 voxels easily
    let step_size = 0.5;
    let steps = (dist / step_size).ceil() as u32;

    for i in 0..steps {
        let t = i as f32 * step_size;
        let p = start + dir * t;
        let wp = WorldPos::new(p.x.floor() as i64, p.y.floor() as i64, p.z.floor() as i64);

        if world.get(wp).is_some() {
            return false;
        }
    }
    
    true
}
