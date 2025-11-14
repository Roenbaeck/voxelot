//! Culling and visibility determination for hierarchical chunks

use crate::lib_hierarchical::{Chunk, Voxel, VoxelType, World};
use std::collections::HashMap;

/// Runtime configuration for rendering and LOD
#[derive(Debug, Clone)]
/// Legacy render config replaced by unified TOML config (`Config.rendering`).
/// Keep a minimal struct so existing code compiles; construct only from `RenderingConfig`.
pub struct RenderConfig {
    pub lod_subdivide_distance: f32,
    pub lod_merge_distance: f32,
    pub lod_render_distance: f32,
    pub far_plane: f32,
    pub fov_degrees: f32,
    pub near_plane: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        // Fallback values if TOML not yet loaded; should be replaced by `RenderingConfig` data.
        Self {
            lod_subdivide_distance: 500.0,
            lod_merge_distance: 1000.0,
            lod_render_distance: 800.0,
            far_plane: 5000.0,
            fov_degrees: 70.0,
            near_plane: 0.1,
        }
    }
}

impl RenderConfig {
    /// Construct from unified TOML `RenderingConfig` (serde-loaded).
    pub fn from_rendering(cfg: &crate::config::RenderingConfig) -> Self {
        Self {
            lod_subdivide_distance: cfg.lod_subdivide_distance,
            lod_merge_distance: cfg.lod_merge_distance,
            lod_render_distance: cfg.chunk_lod_distance,
            far_plane: cfg.far_plane,
            fov_degrees: cfg.fov_degrees,
            near_plane: cfg.near_plane,
        }
    }
}

/// Spatial hash cache for visible voxels
/// Caches visible voxels between frames to avoid recalculation
pub struct VisibilityCache {
    cache: HashMap<(i64, i64, i64), Vec<VoxelInstance>>,
    last_camera_pos: [f32; 3],
    last_camera_forward: [f32; 3],
    dirty: bool,
}

impl VisibilityCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            last_camera_pos: [0.0, 0.0, 0.0],
            last_camera_forward: [0.0, 0.0, -1.0],
            dirty: true,
        }
    }

    /// Check if camera has moved significantly
    fn camera_moved(&self, camera: &Camera) -> bool {
        let pos_delta = [
            camera.position[0] - self.last_camera_pos[0],
            camera.position[1] - self.last_camera_pos[1],
            camera.position[2] - self.last_camera_pos[2],
        ];
        let pos_dist = (pos_delta[0] * pos_delta[0]
            + pos_delta[1] * pos_delta[1]
            + pos_delta[2] * pos_delta[2])
            .sqrt();

        let forward_delta = [
            camera.forward[0] - self.last_camera_forward[0],
            camera.forward[1] - self.last_camera_forward[1],
            camera.forward[2] - self.last_camera_forward[2],
        ];
        let forward_dist = (forward_delta[0] * forward_delta[0]
            + forward_delta[1] * forward_delta[1]
            + forward_delta[2] * forward_delta[2])
            .sqrt();

        // Threshold: moved > 2 units or rotated > 5 degrees
        pos_dist > 2.0 || forward_dist > 0.087 // ~5 degrees
    }

    /// Update cache with new camera position
    pub fn update(&mut self, camera: &Camera, world: &World) -> Vec<VoxelInstance> {
        // Check if we need to recalculate
        if !self.dirty && !self.camera_moved(camera) {
            // Return cached results
            return self.cache.values().flatten().copied().collect();
        }

        // Recalculate visible voxels using parallel culling
        let instances = cull_visible_voxels_parallel(world, camera);

        // Update cache - organize by chunk
        self.cache.clear();
        for instance in &instances {
            let chunk_x = instance.position[0] / 16;
            let chunk_y = instance.position[1] / 16;
            let chunk_z = instance.position[2] / 16;

            self.cache
                .entry((chunk_x, chunk_y, chunk_z))
                .or_insert_with(Vec::new)
                .push(*instance);
        }

        // Update camera state
        self.last_camera_pos = camera.position;
        self.last_camera_forward = camera.forward;
        self.dirty = false;

        instances
    }

    /// Mark cache as dirty (e.g., when world changes)
    pub fn invalidate(&mut self) {
        self.dirty = true;
    }
}

/// Frustum plane for culling tests
#[derive(Debug, Clone, Copy)]
struct Plane {
    normal: [f32; 3],
    distance: f32,
}

impl Plane {
    fn new(normal: [f32; 3], distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Test if a point is in front of (outside) the plane
    fn test_point(&self, point: [f32; 3]) -> bool {
        dot(&self.normal, &point) + self.distance >= 0.0
    }

    /// Test if an AABB is completely behind the plane
    fn test_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        // Get the positive vertex (the one furthest along the normal)
        let p = [
            if self.normal[0] >= 0.0 {
                max[0]
            } else {
                min[0]
            },
            if self.normal[1] >= 0.0 {
                max[1]
            } else {
                min[1]
            },
            if self.normal[2] >= 0.0 {
                max[2]
            } else {
                min[2]
            },
        ];

        // If the positive vertex is behind the plane, the whole box is behind
        self.test_point(p)
    }
}

/// 6-plane frustum for view culling
#[derive(Debug, Clone)]
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    /// Create frustum from camera parameters
    pub fn from_camera(
        position: [f32; 3],
        forward: [f32; 3],
        up: [f32; 3],
        fov: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let forward = normalize(forward);
        let up = normalize(up);
        let right = normalize(cross(&forward, &up));

        // Recompute up to ensure orthogonality
        let up = cross(&right, &forward);

        // Half angles
        let half_v = (fov * 0.5).tan();
        let half_h = half_v * aspect;

        // Compute frustum planes (normal points inward)
        let planes = [
            // Near plane
            Plane::new(forward, -dot(&forward, &position) - near),
            // Far plane
            Plane::new(negate(forward), dot(&forward, &position) + far),
            // Left plane
            {
                let v = add(&forward, &mul_scalar(&right, -half_h));
                let normal = normalize(cross(&v, &up)); // FIXED: flipped order
                Plane::new(normal, -dot(&normal, &position))
            },
            // Right plane
            {
                let v = add(&forward, &mul_scalar(&right, half_h));
                let normal = normalize(cross(&up, &v)); // FIXED: flipped order
                Plane::new(normal, -dot(&normal, &position))
            },
            // Top plane
            {
                let v = add(&forward, &mul_scalar(&up, half_v));
                let normal = normalize(cross(&v, &right)); // FIXED: flipped order
                Plane::new(normal, -dot(&normal, &position))
            },
            // Bottom plane
            {
                let v = add(&forward, &mul_scalar(&up, -half_v));
                let normal = normalize(cross(&right, &v)); // FIXED: flipped order
                Plane::new(normal, -dot(&normal, &position))
            },
        ];

        Self { planes }
    }

    /// Test if an AABB is visible (not completely outside any plane)
    pub fn test_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            if !plane.test_aabb(min, max) {
                return false; // Completely outside this plane
            }
        }
        true // Inside or intersecting all planes
    }
}

/// Camera for view frustum culling
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub config: RenderConfig,
    frustum: Frustum,
}

impl Camera {
    pub fn new(position: [f32; 3], forward: [f32; 3], up: [f32; 3]) -> Self {
    Self::with_config(position, forward, up, RenderConfig::default())
    }

    pub fn with_config(
        position: [f32; 3],
        forward: [f32; 3],
        up: [f32; 3],
        config: RenderConfig,
    ) -> Self {
        let fov = config.fov_degrees.to_radians();
        let aspect = 16.0 / 9.0;
        let near = config.near_plane;
        let far = config.far_plane;

        // Normalize forward vector
        let forward = normalize(forward);

        let frustum = Frustum::from_camera(position, forward, up, fov, aspect, near, far);

        Self {
            position,
            forward,
            up,
            fov,
            aspect,
            near,
            far,
            config,
            frustum,
        }
    }

    /// Update camera parameters and rebuild frustum
    pub fn update(&mut self, position: [f32; 3], forward: [f32; 3], up: [f32; 3]) {
        self.position = position;
        self.forward = forward;
        self.up = up;
        self.frustum = Frustum::from_camera(
            position,
            forward,
            up,
            self.fov,
            self.aspect,
            self.near,
            self.far,
        );
    }

    /// Get the right vector
    pub fn right(&self) -> [f32; 3] {
        normalize(cross(&self.forward, &self.up))
    }

    /// Check if a point is roughly in front of the camera
    pub fn is_in_front(&self, point: [f32; 3]) -> bool {
        let to_point = [
            point[0] - self.position[0],
            point[1] - self.position[1],
            point[2] - self.position[2],
        ];
        dot(&to_point, &self.forward) > self.near
    }

    /// Frustum cull an AABB (proper 6-plane test)
    pub fn frustum_cull_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        self.frustum.test_aabb(min, max)
    }

    /// Get distance from camera to a point
    pub fn distance_to(&self, point: [f32; 3]) -> f32 {
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// A visible voxel instance for rendering
#[derive(Debug, Clone, Copy)]
pub struct VoxelInstance {
    pub position: [i64; 3],
    pub voxel_type: VoxelType,
    pub distance: f32,
    /// Optional: Custom RGBA color for LOD rendering (overrides voxel_type color if Some)
    pub custom_color: Option<[u8; 4]>,
    /// Scale factor for this voxel (1 = normal voxel, 16 = chunk-sized block, etc.)
    pub scale: i64,
    /// Whether this instance represents a leaf chunk that should be rendered via a cached mesh.
    pub is_leaf_chunk: bool,
}

/// Chunk rendering info with LOD
#[derive(Debug, Clone)]
pub struct ChunkRenderInfo {
    pub chunk_pos: (i64, i64, i64),
    pub distance: f32,
    pub lod_level: u8,
}

impl ChunkRenderInfo {
    fn new(chunk_pos: (i64, i64, i64), camera_pos: [f32; 3]) -> Self {
        let chunk_center = [
            (chunk_pos.0 * 16 + 8) as f32,
            (chunk_pos.1 * 16 + 8) as f32,
            (chunk_pos.2 * 16 + 8) as f32,
        ];

        let dx = chunk_center[0] - camera_pos[0];
        let dy = chunk_center[1] - camera_pos[1];
        let dz = chunk_center[2] - camera_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Simple LOD: level 0 for <50 units, level 1 for <150, level 2 beyond
        let lod_level = if distance < 50.0 {
            0
        } else if distance < 150.0 {
            1
        } else {
            2
        };

        Self {
            chunk_pos,
            distance,
            lod_level,
        }
    }
}

/// Recursively collect visible voxels from a chunk, handling hierarchical subdivision
fn collect_voxels_recursive(
    chunk: &Chunk,
    chunk_offset: [i64; 3],
    scale: i64,
    camera: &Camera,
    result: &mut Vec<VoxelInstance>,
) {
    for ((lx, ly, lz), voxel) in chunk.iter() {
        let world_x = chunk_offset[0] + lx as i64 * scale;
        let world_y = chunk_offset[1] + ly as i64 * scale;
        let world_z = chunk_offset[2] + lz as i64 * scale;

        match voxel {
            Voxel::Solid(voxel_type) => {
                // For solid voxels at this hierarchy level, render as a single solid block of the current scale
                // This avoids descending to individual leaf voxels for culling/rendering.
                let voxel_center = [
                    world_x as f32 + (scale as f32 / 2.0),
                    world_y as f32 + (scale as f32 / 2.0),
                    world_z as f32 + (scale as f32 / 2.0),
                ];

                if camera.is_in_front(voxel_center) {
                    let distance = camera.distance_to(voxel_center);
                    result.push(VoxelInstance {
                        position: [world_x, world_y, world_z],
                        voxel_type: *voxel_type,
                        distance,
                        custom_color: None, // Use voxel_type's default color
                        scale,              // Render this solid as a block of size `scale`
                        is_leaf_chunk: false,
                    });
                }
            }
            Voxel::Chunk(sub_chunk) => {
                // For sub-chunks, decide whether to recurse deeper or stop at the final chunk level.
                let voxel_center = [
                    world_x as f32 + (scale as f32 / 2.0),
                    world_y as f32 + (scale as f32 / 2.0),
                    world_z as f32 + (scale as f32 / 2.0),
                ];

                let distance = camera.distance_to(voxel_center);

                // If far away and chunk has voxels, render entire chunk as a single averaged-color block (LOD)
                if distance >= camera.config.lod_render_distance && sub_chunk.voxel_count > 0 {
                    result.push(VoxelInstance {
                        position: [world_x, world_y, world_z],
                        voxel_type: 0, // Ignored when custom_color is Some
                        distance,
                        custom_color: Some(sub_chunk.average_color),
                        scale, // Render the sub-chunk as a single block of this scale
                        is_leaf_chunk: false,
                    });
                    continue;
                }

                // Determine next scale if we were to descend into this sub-chunk
                let next_scale = scale / 16;

                if next_scale > 1 {
                    // We are not yet at the final chunk level; recurse to reach the bottom-level chunk
                    collect_voxels_recursive(
                        sub_chunk,
                        [world_x, world_y, world_z],
                        next_scale,
                        camera,
                        result,
                    );
                } else {
                    // We've reached the final chunk level (leaf chunk of 16x16x16 voxels).
                    if distance >= camera.config.lod_render_distance {
                        // Far: render as a single averaged-color block
                        if sub_chunk.voxel_count > 0 {
                            result.push(VoxelInstance {
                                position: [world_x, world_y, world_z],
                                voxel_type: 0,
                                distance,
                                custom_color: Some(sub_chunk.average_color),
                                scale, // at leaf chunk level, `scale` should be 16
                                is_leaf_chunk: false,
                            });
                        }
                    } else {
                        // Near: request the leaf chunk mesh instead of descending to individual voxels.
                        if sub_chunk.voxel_count > 0 {
                            result.push(VoxelInstance {
                                position: [world_x, world_y, world_z],
                                voxel_type: 0,
                                distance,
                                custom_color: None,
                                scale, // chunk size (16)
                                is_leaf_chunk: true,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Cull voxels for rendering - works with hierarchical World (chunks all the way!)
pub fn cull_visible_voxels(world: &World, camera: &Camera) -> Vec<VoxelInstance> {
    let mut instances = Vec::new();

    // World is now a single root chunk - calculate its size and position
    let world_size = world.world_size() as i64;

    // Get world bounding box (assuming world is centered at origin for now)
    let min = [0.0, 0.0, 0.0];
    let max = [world_size as f32, world_size as f32, world_size as f32];

    // Frustum cull the entire world first
    if !camera.frustum_cull_aabb(min, max) {
        return instances; // Entire world is outside frustum
    }

    // Recursively collect voxels from the root chunk
    // The scale factor depends on hierarchy depth
    let scale = 16i64.pow(world.hierarchy_depth() as u32 - 1);

    collect_voxels_recursive(
        world.root(),
        [0, 0, 0], // World starts at origin
        scale,     // Scale of root voxels
        camera,
        &mut instances,
    );

    instances
}

/// Cull voxels with occlusion testing - hierarchical version
/// Uses recursive traversal of World root chunk
pub fn cull_visible_voxels_with_occlusion(world: &World, camera: &Camera) -> Vec<VoxelInstance> {
    let mut instances = Vec::new();

    // World is now a single root chunk
    let world_size = world.world_size() as i64;
    let min = [0.0, 0.0, 0.0];
    let max = [world_size as f32, world_size as f32, world_size as f32];

    // Frustum cull the entire world first
    if !camera.frustum_cull_aabb(min, max) {
        return instances;
    }

    // Recursively collect voxels with occlusion
    let scale = 16i64.pow(world.hierarchy_depth() as u32 - 1);

    collect_voxels_recursive(world.root(), [0, 0, 0], scale, camera, &mut instances);

    instances
}

/// Parallel culling - for hierarchical world, parallelize at top level of root chunk
pub fn cull_visible_voxels_parallel(world: &World, camera: &Camera) -> Vec<VoxelInstance> {
    use rayon::prelude::*;

    // For hierarchical world, we can parallelize by processing top-level cells
    let world_size = world.world_size() as i64;
    let min = [0.0, 0.0, 0.0];
    let max = [world_size as f32, world_size as f32, world_size as f32];

    // Frustum cull the entire world first
    if !camera.frustum_cull_aabb(min, max) {
        return Vec::new();
    }

    // Collect top-level positions that have voxels
    let scale = 16i64.pow(world.hierarchy_depth() as u32 - 1);
    let root = world.root();

    let top_level_cells: Vec<_> = root.positions().map(|(x, y, z)| (x, y, z)).collect();

    // Process each top-level cell in parallel
    let visible_voxels: Vec<Vec<VoxelInstance>> = top_level_cells
        .par_iter()
        .filter_map(|&(x, y, z)| {
            // Get the voxel at this position
            let voxel = root.get(x, y, z)?;

            // Calculate world position
            let world_x = x as i64 * scale;
            let world_y = y as i64 * scale;
            let world_z = z as i64 * scale;

            // Quick frustum check for this cell
            let cell_min = [world_x as f32, world_y as f32, world_z as f32];
            let cell_max = [
                (world_x + scale) as f32,
                (world_y + scale) as f32,
                (world_z + scale) as f32,
            ];

            if !camera.frustum_cull_aabb(cell_min, cell_max) {
                return None;
            }

            let mut cell_instances = Vec::new();

            match voxel {
                Voxel::Solid(vtype) => {
                    // Add this solid voxel
                    let distance = ((world_x as f32 - camera.position[0]).powi(2)
                        + (world_y as f32 - camera.position[1]).powi(2)
                        + (world_z as f32 - camera.position[2]).powi(2))
                    .sqrt();

                    cell_instances.push(VoxelInstance {
                        position: [world_x, world_y, world_z],
                        voxel_type: *vtype,
                        distance,
                        custom_color: None,
                        scale, // render as a block of current scale (no per-voxel descent)
                        is_leaf_chunk: false,
                    });
                }
                Voxel::Chunk(chunk) => {
                    // Recursively collect from sub-chunk until bottom chunk level.
                    let next_scale = scale / 16;
                    if next_scale > 1 {
                        collect_voxels_recursive(
                            chunk,
                            [world_x, world_y, world_z],
                            next_scale,
                            camera,
                            &mut cell_instances,
                        );
                    } else {
                        // Bottom-level chunk: near = individual voxels, far = averaged block
                        let center = [
                            world_x as f32 + (scale as f32 / 2.0),
                            world_y as f32 + (scale as f32 / 2.0),
                            world_z as f32 + (scale as f32 / 2.0),
                        ];
                        let distance = camera.distance_to(center);
                        if distance >= camera.config.lod_render_distance {
                            if chunk.voxel_count > 0 {
                                cell_instances.push(VoxelInstance {
                                    position: [world_x, world_y, world_z],
                                    voxel_type: 0,
                                    distance,
                                    custom_color: Some(chunk.average_color),
                                    scale, // scale should be 16 here
                                    is_leaf_chunk: false,
                                });
                            }
                        } else {
                            if chunk.voxel_count > 0 {
                                cell_instances.push(VoxelInstance {
                                    position: [world_x, world_y, world_z],
                                    voxel_type: 0,
                                    distance,
                                    custom_color: None,
                                    scale,
                                    is_leaf_chunk: true,
                                });
                            }
                        }
                    }
                }
            }

            Some(cell_instances)
        })
        .collect();

    // Flatten results
    visible_voxels.into_iter().flatten().collect()
}

/// Get visible top-level cells as chunk render info
/// Note: With hierarchical world, this returns top-level cells of the root chunk
pub fn get_visible_chunks(world: &World, camera: &Camera) -> Vec<ChunkRenderInfo> {
    let far_sq = camera.far * camera.far;
    let scale = 16i64.pow(world.hierarchy_depth() as u32 - 1);
    let root = world.root();

    let chunk_infos: Vec<_> = root
        .positions()
        .filter_map(|(x, y, z)| {
            let world_x = x as i64 * scale;
            let world_y = y as i64 * scale;
            let world_z = z as i64 * scale;

            // Check if cell is within far plane distance
            let cell_center = [
                world_x as f32 + scale as f32 / 2.0,
                world_y as f32 + scale as f32 / 2.0,
                world_z as f32 + scale as f32 / 2.0,
            ];
            let dx = cell_center[0] - camera.position[0];
            let dy = cell_center[1] - camera.position[1];
            let dz = cell_center[2] - camera.position[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            // Skip cells beyond far plane
            if dist_sq > far_sq + (scale * scale * 3) as f32 {
                return None;
            }

            let min = [world_x as f32, world_y as f32, world_z as f32];
            let max = [
                (world_x + scale) as f32,
                (world_y + scale) as f32,
                (world_z + scale) as f32,
            ];

            if camera.frustum_cull_aabb(min, max) {
                Some(ChunkRenderInfo::new(
                    (world_x, world_y, world_z),
                    camera.position,
                ))
            } else {
                None
            }
        })
        .collect();

    // Sort by distance for efficient rendering
    let mut sorted_infos = chunk_infos;
    sorted_infos.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    sorted_infos
}

// Vector math helpers
fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

fn negate(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

fn add(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul_scalar(v: &[f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}
