use crate::lib_hierarchical::{World, WorldPos};
use crate::palette::Palette;
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use crossbeam_channel::{Sender, Receiver};

/// Request to update GI probes
pub struct GiUpdateRequest {
    pub camera_pos: Vec3,
    pub visible_chunks: Vec<IVec3>,
}

/// Result from async GI update
pub enum GiUpdateResult {
    /// Full probe volume for the current `grid_origin` (sent on grid shifts and initial update).
    Full {
        probes: Vec<GiProbe>,
        grid_origin: IVec3,
        probes_calculated: usize,
    },
    /// Sparse probe updates (sent when grid origin is unchanged).
    Partial {
        updates: Vec<GiProbeUpdate>,
        grid_origin: IVec3,
        probes_calculated: usize,
    },
}

#[derive(Copy, Clone, Debug)]
pub struct GiProbeUpdate {
    /// Flat index into the local probe grid (x + y*dims.x + z*dims.x*dims.y)
    pub index: u32,
    pub probe: GiProbe,
}

/// Compact representation of an emissive voxel within a chunk
#[derive(Copy, Clone, Debug)]
struct EmissiveVoxel {
    /// Local position within chunk (0..15)
    local_pos: [u8; 3],
    /// Pre-multiplied emission (color * intensity * 10.0)
    emission: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GiProbe {
    /// Probe position in world space (w unused/padding)
    pub position: [f32; 4],
    /// Irradiance for 6 faces: +X, -X, +Y, -Y, +Z, -Z
    /// Each is RGBA (A unused/intensity)
    pub light_data: [[f32; 4]; 6],
    /// Average chunk color (LOD metadata) - used for coarse reflections
    /// Alpha is occupancy (0..1)
    pub color: [f32; 4],
}

impl Default for GiProbe {
    fn default() -> Self {
        Self {
            position: [0.0; 4],
            light_data: [[0.0; 4]; 6],
            color: [0.0; 4],
        }
    }
}

pub struct GiSystem {
    pub grid_origin: IVec3, // In chunks (chunk_coord)
    pub grid_dims: IVec3,   // Dimensions in chunks

    // Caches
    probe_cache: HashMap<IVec3, GiProbe>,
    // Chunk coordinate -> List of emissive voxels (position, intensity)
    light_cache: HashMap<IVec3, Vec<(Vec3, Vec3)>>,
    // Separate index: chunk coordinate -> emissive voxels in that chunk
    // This avoids O(16³) scans when building light_cache
    emissive_index: HashMap<IVec3, Vec<EmissiveVoxel>>,
    // Track missing probes incrementally to avoid full grid scan every update
    missing_probes: Vec<IVec3>,
    last_grid_origin: IVec3,
}

impl GiSystem {
    pub fn new(dims: IVec3) -> Self {
        Self {
            grid_origin: IVec3::new(0, 0, 0),
            grid_dims: dims,
            probe_cache: HashMap::new(),
            light_cache: HashMap::new(),
            emissive_index: HashMap::new(),
            missing_probes: Vec::new(),
            last_grid_origin: IVec3::new(i32::MAX, i32::MAX, i32::MAX), // Force initial scan
        }
    }

    fn local_index_for_coord(&self, chunk_coord: IVec3) -> Option<u32> {
        let rel = chunk_coord - self.grid_origin;
        if rel.x < 0
            || rel.y < 0
            || rel.z < 0
            || rel.x >= self.grid_dims.x
            || rel.y >= self.grid_dims.y
            || rel.z >= self.grid_dims.z
        {
            return None;
        }
        let x = rel.x as u32;
        let y = rel.y as u32;
        let z = rel.z as u32;
        let dx = self.grid_dims.x as u32;
        let dy = self.grid_dims.y as u32;
        Some(x + y * dx + z * dx * dy)
    }

    pub fn build_flat_probes(&self) -> Vec<GiProbe> {
        let dims = self.grid_dims;
        let origin = self.grid_origin;
        let total_probes = (dims.x * dims.y * dims.z).max(0) as usize;
        let mut probes = vec![GiProbe::default(); total_probes];

        if dims.x <= 0 || dims.y <= 0 || dims.z <= 0 {
            return probes;
        }

        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let coord = origin + IVec3::new(x, y, z);
                    let idx = (x + y * dims.x + z * dims.x * dims.y) as usize;
                    if let Some(p) = self.probe_cache.get(&coord) {
                        probes[idx] = *p;
                    }
                }
            }
        }

        probes
    }

    /// Update probes based on camera position and visible chunks from culling
    pub fn update(
        &mut self,
        world: &World,
        palette: &Palette,
        camera_pos: Vec3,
        visible_chunks: &[IVec3],
    ) -> (bool, Vec<GiProbeUpdate>, usize) {
        // 1. Determine new grid origin (centered on camera, snapped to chunk size)
        let chunk_size = 16.0;
        let cam_chunk = (camera_pos / chunk_size).floor().as_ivec3();
        let half_dims = self.grid_dims / 2;
        let new_origin = cam_chunk - half_dims;

        self.grid_origin = new_origin;

        // 2. Identify missing probes from the VISIBLE chunks (frustum culled)
        // Check for missing probes whenever grid moves OR when we have new visible chunks
        let grid_moved = new_origin != self.last_grid_origin;
        
        if grid_moved {
            self.missing_probes.clear();
        }
        
        // Always check visible chunks for missing probes (handles rotation case)
        for &chunk_coord in visible_chunks {
            // Check if chunk is within our grid bounds
            let relative = chunk_coord - new_origin;
            if relative.x >= 0 && relative.x < self.grid_dims.x
                && relative.y >= 0 && relative.y < self.grid_dims.y
                && relative.z >= 0 && relative.z < self.grid_dims.z
            {
                if !self.probe_cache.contains_key(&chunk_coord) {
                    // Only add if not already in the list
                    if !self.missing_probes.contains(&chunk_coord) {
                        self.missing_probes.push(chunk_coord);
                    }
                }
            }
        }
        
        if grid_moved {
            // Sort missing probes by distance to camera (prioritize nearest)
            // This matches the mesh worker priority system
            self.missing_probes.sort_by(|a, b| {
                // Calculate distance from camera chunk to probe chunk
                let da = (*a - cam_chunk).as_vec3().length_squared();
                let db = (*b - cam_chunk).as_vec3().length_squared();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            self.last_grid_origin = new_origin;
        }

        // If no probes are missing, we just need to update the flat buffer and return.
        // However, we should also check if we need to load lights for new areas.
        // For simplicity, we drive light loading by probe requirements.
        
        let mut probes_calculated = 0;
        let mut updates: Vec<GiProbeUpdate> = Vec::new();
        if !self.missing_probes.is_empty() {
            // Throttle: only process up to 64 probes per update to prevent frame drops
            // Since GI runs async on background thread, this won't impact frame rate
            let probes_to_process: Vec<IVec3> = self.missing_probes.iter().take(64).cloned().collect();
            probes_calculated = probes_to_process.len();
            
            // Remove processed probes from missing list
            self.missing_probes.drain(0..probes_to_process.len().min(self.missing_probes.len()));
            
            // 3. Identify required light chunks for the missing probes
            // We need lights from neighbors. Let's say radius is 4 chunks.
            let light_radius = 4;
            let mut required_light_chunks = HashSet::new();
            
            for probe_coord in &probes_to_process {
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

            // 4. Build emissive index for new chunks (in parallel)
            // This replaces the O(16³) voxel scan with O(E) where E = # emissive voxels
            let new_emissive_data: Vec<(IVec3, Vec<EmissiveVoxel>)> = required_light_chunks
                .par_iter()
                .filter_map(|&chunk_coord| {
                    let origin = WorldPos::new(
                        chunk_coord.x as i64 * 16,
                        chunk_coord.y as i64 * 16,
                        chunk_coord.z as i64 * 16,
                    );

                    let chunk = world.get_leaf_chunk_at_origin(origin)?;
                    if chunk.emissive_power <= 0.0 {
                        return None;
                    }

                    let mut emissives = Vec::new();
                    for lz in 0..16u8 {
                        for ly in 0..16u8 {
                            for lx in 0..16u8 {
                                if chunk.contains(lx, ly, lz) {
                                    if let Some(vtype) = chunk.get_type(lx, ly, lz) {
                                        let (color, intensity) = palette.emissive(vtype as u32);
                                        if intensity > 0.0 {
                                            let emission = Vec3::from(color) * intensity * 10.0;
                                            emissives.push(EmissiveVoxel {
                                                local_pos: [lx, ly, lz],
                                                emission,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if emissives.is_empty() {
                        None
                    } else {
                        Some((chunk_coord, emissives))
                    }
                })
                .collect();

            // Update emissive index
            for (coord, emissives) in new_emissive_data {
                self.emissive_index.insert(coord, emissives);
            }

            // 4b. Build light_cache from emissive_index (fast, no voxel scanning)
            let new_lights: Vec<(IVec3, Vec<(Vec3, Vec3)>)> = required_light_chunks
                .into_iter()
                .filter_map(|chunk_coord| {
                    let emissives = self.emissive_index.get(&chunk_coord)?;
                    if emissives.is_empty() {
                        return None;
                    }

                    let lights: Vec<(Vec3, Vec3)> = emissives
                        .iter()
                        .map(|ev| {
                            let voxel_pos = Vec3::new(
                                (chunk_coord.x as f32 * 16.0) + ev.local_pos[0] as f32 + 0.5,
                                (chunk_coord.y as f32 * 16.0) + ev.local_pos[1] as f32 + 0.5,
                                (chunk_coord.z as f32 * 16.0) + ev.local_pos[2] as f32 + 0.5,
                            );
                            (voxel_pos, ev.emission)
                        })
                        .collect();
                    Some((chunk_coord, lights))
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
            
            // Step 5a: Build flat light list once for all probes (optimization)
            // Instead of doing 729 HashMap lookups per probe, collect all lights in radius once
            let mut all_lights = Vec::new();
            let mut light_chunks_processed = HashSet::new();
            for probe_coord in &probes_to_process {
                for z in -light_radius..=light_radius {
                    for y in -light_radius..=light_radius {
                        for x in -light_radius..=light_radius {
                            let light_coord = *probe_coord + IVec3::new(x, y, z);
                            if light_chunks_processed.insert(light_coord) {
                                if let Some(chunk_lights) = self.light_cache.get(&light_coord) {
                                    all_lights.extend_from_slice(chunk_lights);
                                }
                            }
                        }
                    }
                }
            }
            
            // Step 5b: Prepare jobs with shared light list
            let jobs: Vec<IVec3> = probes_to_process;

            // Step 5c: Execute jobs in parallel with shared light list
            let new_probes: Vec<(IVec3, GiProbe)> = jobs.into_par_iter().map(|probe_coord| {
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

                if let Some(chunk) = world.get_leaf_chunk_at_origin(WorldPos::new(cx as i64 * 16, cy as i64 * 16, cz as i64 * 16)) {
                    probe.color = [
                        chunk.average_color[0] as f32 / 255.0,
                        chunk.average_color[1] as f32 / 255.0,
                        chunk.average_color[2] as f32 / 255.0,
                        chunk.average_color[3] as f32 / 255.0,
                    ];
                }
                
                let normals = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
                // Invert offsets to look "inward" from the opposite face.
                // This ensures that for the +X bin (surfaces facing +X), we sample at the -X boundary (where those surfaces are),
                // and look towards +X (seeing internal lights and far neighbors).
                let face_offsets = [
                    Vec3::new(-7.0, 0.0, 0.0), // For +X bin, sample at -X
                    Vec3::new(7.0, 0.0, 0.0),  // For -X bin, sample at +X
                    Vec3::new(0.0, -7.0, 0.0), // For +Y bin, sample at -Y
                    Vec3::new(0.0, 7.0, 0.0),  // For -Y bin, sample at +Y
                    Vec3::new(0.0, 0.0, -7.0), // For +Z bin, sample at -Z
                    Vec3::new(0.0, 0.0, 7.0),  // For -Z bin, sample at +Z
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

                    for (light_pos, light_energy) in &all_lights {
                        let delta = *light_pos - face_center;
                        if delta.dot(face_normal) <= 0.0 { continue; }

                        let dist_sq = delta.length_squared();
                        if dist_sq > 64.0 * 64.0 || dist_sq < 0.01 { continue; }

                        let dist = dist_sq.sqrt();
                        let dir = delta / dist;
                        let cos_theta = dir.dot(face_normal);
                        
                        // Use hierarchical line_of_sight instead of DDA
                        let start_pos = WorldPos::new(
                            face_center.x.floor() as i64,
                            face_center.y.floor() as i64,
                            face_center.z.floor() as i64,
                        );
                        let end_pos = WorldPos::new(
                            light_pos.x.floor() as i64,
                            light_pos.y.floor() as i64,
                            light_pos.z.floor() as i64,
                        );
                        
                        if !world.line_of_sight(start_pos, end_pos) {
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
                if !grid_moved {
                    if let Some(index) = self.local_index_for_coord(coord) {
                        updates.push(GiProbeUpdate { index, probe });
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

        (grid_moved, updates, probes_calculated)
    }
}

/// Spawn a background worker thread for async GI probe updates
/// Returns (request_sender, result_receiver)
pub fn spawn_gi_worker(
    world: Arc<World>,
    palette: Arc<Palette>,
    grid_dims: IVec3,
) -> (Sender<GiUpdateRequest>, Receiver<GiUpdateResult>) {
    let (request_tx, request_rx) = crossbeam_channel::unbounded::<GiUpdateRequest>();
    let (result_tx, result_rx) = crossbeam_channel::unbounded::<GiUpdateResult>();

    std::thread::Builder::new()
        .name("gi-worker".to_string())
        .spawn(move || {
        let mut gi_system = GiSystem::new(grid_dims);
        
        while let Ok(request) = request_rx.recv() {
            // Update GI system - world is already Arc, no lock needed (World is Sync)
            let (grid_moved, updates, probes_calculated) =
                gi_system.update(&world, &palette, request.camera_pos, &request.visible_chunks);

            let grid_origin = gi_system.grid_origin;
            let result = if grid_moved {
                GiUpdateResult::Full {
                    probes: gi_system.build_flat_probes(),
                    grid_origin,
                    probes_calculated,
                }
            } else {
                GiUpdateResult::Partial {
                    updates,
                    grid_origin,
                    probes_calculated,
                }
            };
            
            // If send fails, main thread has dropped the receiver (shutdown)
            if result_tx.send(result).is_err() {
                break;
            }
        }
    })
    .expect("failed to spawn GI worker");

    (request_tx, result_rx)
}
