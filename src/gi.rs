use crate::lib_hierarchical::{Chunk, Voxel, World, WorldPos};
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
}

impl GiSystem {
    pub fn new(dims: IVec3) -> Self {
        let count = (dims.x * dims.y * dims.z) as usize;
        Self {
            probes: vec![GiProbe::default(); count],
            grid_origin: IVec3::new(0, 0, 0),
            grid_dims: dims,
        }
    }

    /// Update probes based on camera position and world state
    pub fn update(&mut self, world: &World, camera_pos: Vec3) {
        // 1. Determine new grid origin (centered on camera, snapped to chunk size)
        let chunk_size = 16.0;
        let cam_chunk = (camera_pos / chunk_size).floor().as_ivec3();
        let half_dims = self.grid_dims / 2;
        let new_origin = cam_chunk - half_dims;

        self.grid_origin = new_origin;

        // 2. Identify emissive chunks in the vicinity
        // For performance, we'll scan the world hierarchy or just iterate expected positions
        // Since Random Access is valid, let's just iterate our grid for now to find "Local" emissives.
        // A better approach would be to maintain a list of emissive chunks in World, but that requires World changes.
        // Let's assume we can scan the relevant area.

        // Collect potential light sources (mid-points of emissive chunks)
        // Format: (Position, Color, Intensity)
        let mut light_sources = Vec::new();

        let sub_scan_dims = self.grid_dims; // Scan same area as probes for now

        for z in 0..sub_scan_dims.z {
            for y in 0..sub_scan_dims.y {
                for x in 0..sub_scan_dims.x {
                    let cx = new_origin.x + x;
                    let cy = new_origin.y + y;
                    let cz = new_origin.z + z;

                    let origin = WorldPos::new(cx as i64 * 16, cy as i64 * 16, cz as i64 * 16);

                    if let Some(chunk) = world.get_leaf_chunk_at_origin(origin) {
                        if chunk.emissive_power > 0.0 {
                            // Use center of chunk as light source for "rough" GI
                            let center = Vec3::new(
                                (cx as f32 * 16.0) + 8.0,
                                (cy as f32 * 16.0) + 8.0,
                                (cz as f32 * 16.0) + 8.0,
                            );
                            let color = Vec3::from(chunk.emissive_sum); // This is Sum(Color * Strength)
                                                                        // Normalize by voxel count to get average radiance?
                                                                        // emissive_sum in Chunk is accumulated (color * strength).
                                                                        // If we treat the whole chunk as a point light, Total Power ~ emissive_sum.
                                                                        // We shouldn't divide by count if we want total energy.

                            light_sources.push((center, color));
                        }
                    }
                }
            }
        }

        // 3. Update probes in parallel
        // For shift/scroll we could preserve data, but for now simple rebuild
        let dims = self.grid_dims;
        let origin = self.grid_origin;

        self.probes
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, probe)| {
                // Reconstruct 3D index
                let tmp_yz = i as i32 / dims.x;
                let lx = i as i32 % dims.x;
                let ly = tmp_yz % dims.y;
                let lz = tmp_yz / dims.y;

                let cx = origin.x + lx;
                let cy = origin.y + ly;
                let cz = origin.z + lz;

                let center_pos = Vec3::new(
                    (cx as f32 * 16.0) + 8.0,
                    (cy as f32 * 16.0) + 8.0,
                    (cz as f32 * 16.0) + 8.0,
                );

                // Update position
                probe.position = [center_pos.x, center_pos.y, center_pos.z, 1.0];

                // Clear light data
                probe.light_data = [[0.0; 4]; 6];

                // Accumulate light
                for (light_pos, light_energy) in &light_sources {
                    let delta = *light_pos - center_pos;
                    let dist_sq = delta.length_squared();

                    // Max range check (e.g. 100 meters = ~6 chunks)
                    if dist_sq > 200.0 * 200.0 || dist_sq < 1.0 {
                        continue;
                    }

                    let dist = dist_sq.sqrt();
                    let dir = delta / dist;

                    // Occlusion Raycast
                    // Simple heuristic: check midpoint or a few steps
                    // Full voxel traversal is too slow here (probes * lights).
                    // Let's check 3 steps: 25%, 50%, 75%
                    if !is_visible_heuristic(world, center_pos, *light_pos, dist) {
                        continue;
                    }

                    // Attenuation (Inverse Square)
                    let attenuation = 1.0 / (1.0 + dist_sq * 0.05); // Tweak factor

                    let incoming = *light_energy * attenuation * 0.5; // Scale down a bit

                    // Project onto 6 faces (Ambient Cube / Valve basis)
                    // Normals: +X, -X, +Y, -Y, +Z, -Z
                    let normals = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];

                    for f in 0..6 {
                        let cos_theta = dir.dot(normals[f]).max(0.0);
                        if cos_theta > 0.0 {
                            let contrib = incoming * cos_theta;
                            probe.light_data[f][0] += contrib.x;
                            probe.light_data[f][1] += contrib.y;
                            probe.light_data[f][2] += contrib.z;
                        }
                    }
                }
            });
    }
}

/// Very coarse visibility check
fn is_visible_heuristic(world: &World, p0: Vec3, p1: Vec3, dist: f32) -> bool {
    let steps = (dist / 16.0).ceil() as u32; // check every chunk
    if steps <= 1 {
        return true;
    }

    let dir = (p1 - p0) / dist;

    // Check occupancy at intervals
    for i in 1..steps {
        let t = i as f32 * 16.0;
        let p = p0 + dir * t;
        let wp = WorldPos::new(p.x as i64, p.y as i64, p.z as i64);

        // Fast check: get leaf chunk, see if solid ratio is high?
        // Or check specific voxel?
        // Let's just check if there is a chunk there that isn't empty.
        // But we want to allow light through empty space.

        // Check if point is inside a solid voxel?
        // world.get(wp) returns Some(type) if solid.
        if world.get(wp).is_some() {
            return false;
        }
    }
    true
}
