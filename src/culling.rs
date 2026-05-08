//! Culling and visibility determination for hierarchical chunks

use crate::lib_hierarchical::{bbox_local_to_world, Chunk, Voxel, VoxelType, World};
use rustc_hash::FxHashMap as HashMap;

/// Runtime configuration for rendering and LOD
#[derive(Debug, Clone)]
/// Legacy render config replaced by unified TOML config (`Config.rendering`).
/// Keep a minimal struct so existing code compiles; construct only from `RenderingConfig`.
pub struct RenderConfig {
    pub lod_render_distance: f32,
    pub far_plane: f32,
    pub fov_degrees: f32,
    pub near_plane: f32,
    pub culling_overscan: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        // Fallback values if TOML not yet loaded; should be replaced by `RenderingConfig` data.
        Self {
            lod_render_distance: 800.0,
            far_plane: 5000.0,
            fov_degrees: 70.0,
            near_plane: 0.1,
            culling_overscan: 0.0,
        }
    }
}

impl RenderConfig {
    /// Construct from unified TOML `RenderingConfig` (serde-loaded).
    pub fn from_rendering(cfg: &crate::config::RenderingConfig) -> Self {
        Self {
            lod_render_distance: cfg.chunk_lod_distance,
            far_plane: cfg.far_plane,
            fov_degrees: cfg.fov_degrees,
            near_plane: cfg.near_plane,
            culling_overscan: cfg.culling_overscan.max(cfg.render_overscan),
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
            cache: HashMap::default(),
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
        let (instances, _stats, _visible_chunks) = cull_visible_voxels_parallel(world, camera);

        // Update cache - organize by chunk
        self.cache.clear();
        for instance in &instances {
            let chunk_x = instance.position[0] >> 4;
            let chunk_y = instance.position[1] >> 4;
            let chunk_z = instance.position[2] >> 4;

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

use std::sync::OnceLock;

/// Precomputed masks for fast culling
/// Stores bitmaps for all possible contiguous ranges [start, end) in X and Y dimensions (0..16)
struct CullingMasks {
    /// x_masks[start][end] -> Bitmap with bits set where x in start..end
    x_masks: [[croaring::Bitmap; 17]; 16],
    /// y_masks[start][end] -> Bitmap with bits set where y in start..end
    y_masks: [[croaring::Bitmap; 17]; 16],
}

static CULLING_MASKS: OnceLock<CullingMasks> = OnceLock::new();

impl CullingMasks {
    fn get() -> &'static Self {
        CULLING_MASKS.get_or_init(|| {
            let mut x_masks: [[croaring::Bitmap; 17]; 16] = Default::default();
            let mut y_masks: [[croaring::Bitmap; 17]; 16] = Default::default();

            // Precompute X masks
            // Index = x + y*16 + z*256
            for start in 0..16 {
                for end in (start + 1)..=16 {
                    let mut bitmap = croaring::Bitmap::new();
                    // For X range [start, end), we want all indices where x is in this range
                    // Iterate all y and z
                    for z in 0..16 {
                        for y in 0..16 {
                            // Add range [start + y*16 + z*256, end + y*16 + z*256)
                            // Since x is the LSB, a contiguous range in x is contiguous in index
                            let range_start = (start | (y << 4) | (z << 8)) as u32;
                            let range_end = (end + (y << 4) + (z << 8)) as u32;
                            bitmap.add_range(range_start..range_end);
                        }
                    }
                    x_masks[start][end] = bitmap;
                }
            }

            // Precompute Y masks
            for start in 0..16 {
                for end in (start + 1)..=16 {
                    let mut bitmap = croaring::Bitmap::new();
                    // For Y range [start, end), we want all indices where y is in this range
                    // Iterate all z. Within each z, y ranges are contiguous blocks of 16 indices (x=0..16)
                    for z in 0..16 {
                        // The Y range corresponds to indices:
                        // From: 0 (x=0) + start*16 + z*256
                        // To:   16 (x=16) + (end-1)*16 + z*256
                        // But wait, Y is not contiguous in index space because X is inner loop.
                        // Index = x + y*16 + z*256.
                        // A range of Y means: for a fixed Z, we have y in [start, end).
                        // For a fixed Z and fixed Y, we have X in [0, 16).
                        // So for each Z, we have a block of Ys.
                        // Range of indices for a fixed Z and Y range [start, end):
                        // Start index: 0 + start*16 + z*256
                        // End index:   16 + (end-1)*16 + z*256 = 0 + end*16 + z*256
                        // Yes, since X is 0..16, one full Y row is 16 indices.
                        // So Y range [start, end) covers contiguous indices [start*16, end*16) within each Z plane.

                        let range_start = ((start << 4) | (z << 8)) as u32;
                        let range_end = ((end << 4) + (z << 8)) as u32;
                        bitmap.add_range(range_start..range_end);
                    }
                    y_masks[start][end] = bitmap;
                }
            }

            Self { x_masks, y_masks }
        })
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
    pub bounds: ([f32; 3], [f32; 3]), // (min, max) AABB of the frustum
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
        culling_overscan: f32,
    ) -> Self {
        let forward = normalize(forward);
        let up = normalize(up);
        let right = normalize(cross(&forward, &up));

        // Recompute up to ensure orthogonality
        let up = cross(&right, &forward);

        // Half angles (tangents), widened by overscan.
        // overscan=0 keeps original frustum; overscan>0 widens culling without changing rendering FOV.
        let overscan = culling_overscan.max(0.0);
        let half_v = (fov * 0.5).tan() * (1.0 + overscan);
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
                let normal = normalize(cross(&v, &up));
                Plane::new(normal, -dot(&normal, &position))
            },
            // Right plane
            {
                let v = add(&forward, &mul_scalar(&right, half_h));
                let normal = normalize(cross(&up, &v));
                Plane::new(normal, -dot(&normal, &position))
            },
            // Top plane
            {
                let v = add(&forward, &mul_scalar(&up, half_v));
                let normal = normalize(cross(&v, &right));
                Plane::new(normal, -dot(&normal, &position))
            },
            // Bottom plane
            {
                let v = add(&forward, &mul_scalar(&up, -half_v));
                let normal = normalize(cross(&right, &v));
                Plane::new(normal, -dot(&normal, &position))
            },
        ];

        // Compute Frustum AABB
        let near_center = add(&position, &mul_scalar(&forward, near));
        let far_center = add(&position, &mul_scalar(&forward, far));

        let near_height = 2.0 * half_v * near;
        let near_width = near_height * aspect;

        let far_height = 2.0 * half_v * far;
        let far_width = far_height * aspect;

        let mut corners = [[0.0; 3]; 8];

        // Near corners
        corners[0] = add(
            &near_center,
            &add(
                &mul_scalar(&up, near_height * 0.5),
                &mul_scalar(&right, -near_width * 0.5),
            ),
        ); // TL
        corners[1] = add(
            &near_center,
            &add(
                &mul_scalar(&up, near_height * 0.5),
                &mul_scalar(&right, near_width * 0.5),
            ),
        ); // TR
        corners[2] = add(
            &near_center,
            &add(
                &mul_scalar(&up, -near_height * 0.5),
                &mul_scalar(&right, -near_width * 0.5),
            ),
        ); // BL
        corners[3] = add(
            &near_center,
            &add(
                &mul_scalar(&up, -near_height * 0.5),
                &mul_scalar(&right, near_width * 0.5),
            ),
        ); // BR

        // Far corners
        corners[4] = add(
            &far_center,
            &add(
                &mul_scalar(&up, far_height * 0.5),
                &mul_scalar(&right, -far_width * 0.5),
            ),
        ); // TL
        corners[5] = add(
            &far_center,
            &add(
                &mul_scalar(&up, far_height * 0.5),
                &mul_scalar(&right, far_width * 0.5),
            ),
        ); // TR
        corners[6] = add(
            &far_center,
            &add(
                &mul_scalar(&up, -far_height * 0.5),
                &mul_scalar(&right, -far_width * 0.5),
            ),
        ); // BL
        corners[7] = add(
            &far_center,
            &add(
                &mul_scalar(&up, -far_height * 0.5),
                &mul_scalar(&right, far_width * 0.5),
            ),
        ); // BR

        let mut min = corners[0];
        let mut max = corners[0];

        for i in 1..8 {
            min[0] = min[0].min(corners[i][0]);
            min[1] = min[1].min(corners[i][1]);
            min[2] = min[2].min(corners[i][2]);

            max[0] = max[0].max(corners[i][0]);
            max[1] = max[1].max(corners[i][1]);
            max[2] = max[2].max(corners[i][2]);
        }

        Self {
            planes,
            bounds: (min, max),
        }
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

    /// Calculate intersection between Frustum AABB and another AABB
    /// Returns None if no intersection, or Some((min, max)) of the intersection box
    pub fn aabb_intersection(
        &self,
        other_min: [f32; 3],
        other_max: [f32; 3],
    ) -> Option<([f32; 3], [f32; 3])> {
        let (f_min, f_max) = self.bounds;

        let min = [
            f_min[0].max(other_min[0]),
            f_min[1].max(other_min[1]),
            f_min[2].max(other_min[2]),
        ];

        let max = [
            f_max[0].min(other_max[0]),
            f_max[1].min(other_max[1]),
            f_max[2].min(other_max[2]),
        ];

        if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
            None
        } else {
            Some((min, max))
        }
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

        let frustum = Frustum::from_camera(
            position,
            forward,
            up,
            fov,
            aspect,
            near,
            far,
            config.culling_overscan,
        );

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
            self.config.culling_overscan,
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
    pub scale: [f32; 3],
    /// Whether this instance represents a leaf chunk that should be rendered via a cached mesh.
    pub is_leaf_chunk: bool,
}

/// Chunk rendering info with LOD
#[derive(Debug, Clone)]
pub struct ChunkRenderInfo {
    pub chunk_pos: (i64, i64, i64),
    pub distance: f32,
}

impl ChunkRenderInfo {
    fn new(chunk_pos: (i64, i64, i64), camera_pos: [f32; 3]) -> Self {
        let chunk_center = [
            ((chunk_pos.0 << 4) + 8) as f32,
            ((chunk_pos.1 << 4) + 8) as f32,
            ((chunk_pos.2 << 4) + 8) as f32,
        ];

        let dx = chunk_center[0] - camera_pos[0];
        let dy = chunk_center[1] - camera_pos[1];
        let dz = chunk_center[2] - camera_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        Self {
            chunk_pos,
            distance,
        }
    }
}

/// Statistics for culling operations, grouped by reason
#[derive(Default, Debug, Clone)]
pub struct CullStats {
    /// Chunks culled because frustum AABB had no intersection
    pub frustum_aabb_culled: usize,
    /// Chunks culled because marginal bitmap showed no voxels in range
    pub marginal_bitmap_culled: usize,
    /// Chunks culled because hierarchy shell showed no visible faces
    pub hierarchy_shell_culled: usize,
    /// Chunks culled because voxel_count was 0
    pub empty_chunk_culled: usize,
    /// Chunks that passed all culling and were added to result
    pub chunks_visible: usize,
    /// Total chunks examined (all levels)
    pub chunks_examined: usize,
    /// Chunks that had no hierarchy shell (couldn't be shell-culled)
    pub no_shell_available: usize,
}

impl CullStats {
    pub fn merge(&mut self, other: &CullStats) {
        self.frustum_aabb_culled += other.frustum_aabb_culled;
        self.marginal_bitmap_culled += other.marginal_bitmap_culled;
        self.hierarchy_shell_culled += other.hierarchy_shell_culled;
        self.empty_chunk_culled += other.empty_chunk_culled;
        self.chunks_visible += other.chunks_visible;
        self.chunks_examined += other.chunks_examined;
        self.no_shell_available += other.no_shell_available;
    }
}

/// Recursively collect visible voxels from a chunk, handling hierarchical subdivision
fn collect_voxels_recursive(
    chunk: &Chunk,
    chunk_offset: [i64; 3],
    scale: i64,
    camera: &Camera,
    result: &mut Vec<VoxelInstance>,
    stats: &mut CullStats,
    visible_chunks: &mut Vec<glam::IVec3>,
) {
    // 1. Compute Intersection AABB between Frustum AABB and Chunk AABB
    let chunk_min = [
        chunk_offset[0] as f32,
        chunk_offset[1] as f32,
        chunk_offset[2] as f32,
    ];
    let chunk_size = (scale << 4) as f32;
    let chunk_max = [
        chunk_min[0] + chunk_size,
        chunk_min[1] + chunk_size,
        chunk_min[2] + chunk_size,
    ];

    // Get the intersection box in world space
    let (inter_min, inter_max) = match camera.frustum.aabb_intersection(chunk_min, chunk_max) {
        Some(bounds) => bounds,
        None => return, // No overlap with Frustum AABB -> definitely not visible
    };

    // 2. Convert Intersection AABB to Local Chunk Coordinates (0..16)
    // We clamp to 0..16 range (exclusive of 16)
    let local_min_x = ((inter_min[0] - chunk_min[0]) / scale as f32)
        .floor()
        .max(0.0) as u32;
    let local_min_y = ((inter_min[1] - chunk_min[1]) / scale as f32)
        .floor()
        .max(0.0) as u32;
    let local_min_z = ((inter_min[2] - chunk_min[2]) / scale as f32)
        .floor()
        .max(0.0) as u32;

    let local_max_x = ((inter_max[0] - chunk_min[0]) / scale as f32)
        .ceil()
        .min(16.0) as u32;
    let local_max_y = ((inter_max[1] - chunk_min[1]) / scale as f32)
        .ceil()
        .min(16.0) as u32;
    let local_max_z = ((inter_max[2] - chunk_min[2]) / scale as f32)
        .ceil()
        .min(16.0) as u32;

    if local_min_x >= local_max_x || local_min_y >= local_max_y || local_min_z >= local_max_z {
        return; // Empty intersection
    }

    // 3. Marginal Bitmap Culling (Coarse)
    // Create masks for the relevant ranges
    let mut mask_x = 0u16;
    for i in local_min_x..local_max_x {
        mask_x |= 1 << i;
    }
    if (chunk.px & mask_x) == 0 {
        return;
    }

    let mut mask_y = 0u16;
    for i in local_min_y..local_max_y {
        mask_y |= 1 << i;
    }
    if (chunk.py & mask_y) == 0 {
        return;
    }

    let mut mask_z = 0u16;
    for i in local_min_z..local_max_z {
        mask_z |= 1 << i;
    }
    if (chunk.pz & mask_z) == 0 {
        return;
    }

    // 4. Roaring Bitmap Masking (Fine)
    // Instead of iterating all voxels, we iterate only those in the intersection volume.

    let full_coverage = local_min_x == 0
        && local_max_x == 16
        && local_min_y == 0
        && local_max_y == 16
        && local_min_z == 0
        && local_max_z == 16;

    if full_coverage {
        // Standard iteration (but we already did AABB check, so we know it's roughly in view)
        process_voxels(
            chunk.presence.iter(),
            chunk,
            chunk_offset,
            scale,
            camera,
            result,
            stats,
            visible_chunks,
        );
    } else {
        // Masked iteration using precomputed masks
        // Start with the chunk's presence
        let mut visible_voxels = chunk.presence.clone();

        // Z-Culling: Remove planes outside [local_min_z, local_max_z)
        // Each Z plane is 256 indices (16x16).
        // Remove [0, min_z * 256)
        if local_min_z > 0 {
            visible_voxels.remove_range(0..((local_min_z as u32) << 8));
        }
        // Remove [max_z * 256, 16 * 256)
        if local_max_z < 16 {
            visible_voxels.remove_range(((local_max_z as u32) << 8)..4096);
        }

        // If empty after Z culling, return
        if visible_voxels.is_empty() {
            return;
        }

        let masks = CullingMasks::get();

        // Y-Culling: AND with precomputed Y mask
        if local_min_y > 0 || local_max_y < 16 {
            visible_voxels.and_inplace(&masks.y_masks[local_min_y as usize][local_max_y as usize]);
            if visible_voxels.is_empty() {
                return;
            }
        }

        // X-Culling: AND with precomputed X mask
        if local_min_x > 0 || local_max_x < 16 {
            visible_voxels.and_inplace(&masks.x_masks[local_min_x as usize][local_max_x as usize]);
            if visible_voxels.is_empty() {
                return;
            }
        }

        process_voxels(
            visible_voxels.iter(),
            chunk,
            chunk_offset,
            scale,
            camera,
            result,
            stats,
            visible_chunks,
        );
    }
}

fn process_voxels<I>(
    indices: I,
    chunk: &Chunk,
    chunk_offset: [i64; 3],
    scale: i64,
    camera: &Camera,
    result: &mut Vec<VoxelInstance>,
    stats: &mut CullStats,
    visible_chunks: &mut Vec<glam::IVec3>,
) where
    I: Iterator<Item = u32>,
{
    // NOTE: bbox_local_to_world is available at module scope for consistent conversions.

    // We need to map index back to x,y,z.
    // Assuming index = x + y*16 + z*256

    // We also need to access the voxel data.
    // Chunk.voxels is indexed by RANK, not position.
    // So we need: chunk.voxels[chunk.presence.rank(index) - 1]

    // Pre-compute the shell lookup: map packed_pos -> visible_faces
    // The parent chunk's shell tells us which children have exposed faces
    let shell_map: Option<rustc_hash::FxHashMap<u16, u8>> =
        chunk.hierarchy_shell.as_ref().map(|shell| {
            shell
                .iter()
                .map(|sv| (sv.packed_pos, sv.visible_faces))
                .collect()
        });

    for index in indices {
        let z = index >> 8;
        let y = (index >> 4) & 0xF;
        let x = index & 0xF;

        let rank = chunk.presence.rank(index);
        let voxel = &chunk.voxels[(rank - 1) as usize];

        let world_x = chunk_offset[0] + x as i64 * scale;
        let world_y = chunk_offset[1] + y as i64 * scale;
        let world_z = chunk_offset[2] + z as i64 * scale;

        match voxel {
            Voxel::Solid(voxel_type) => {
                let voxel_center = [
                    world_x as f32 + (scale as f32 / 2.0),
                    world_y as f32 + (scale as f32 / 2.0),
                    world_z as f32 + (scale as f32 / 2.0),
                ];

                // We still do the exact check because the AABB intersection is loose (AABB vs AABB, not Frustum vs Voxel)
                if camera.frustum.test_aabb(
                    [world_x as f32, world_y as f32, world_z as f32],
                    [
                        (world_x + scale) as f32,
                        (world_y + scale) as f32,
                        (world_z + scale) as f32,
                    ],
                ) {
                    let distance = camera.distance_to(voxel_center);
                    result.push(VoxelInstance {
                        position: [world_x, world_y, world_z],
                        voxel_type: *voxel_type,
                        distance,
                        custom_color: None,
                        scale: [scale as f32, scale as f32, scale as f32],
                        is_leaf_chunk: false,
                    });
                }
            }
            Voxel::Chunk(sub_chunk) => {
                stats.chunks_examined += 1;
                let voxel_center = [
                    world_x as f32 + (scale as f32 / 2.0),
                    world_y as f32 + (scale as f32 / 2.0),
                    world_z as f32 + (scale as f32 / 2.0),
                ];

                let distance = camera.distance_to(voxel_center);

                // --- Hierarchy Shell Culling ---
                // Use the PARENT chunk's shell to check if this child has visible faces.
                // The visibility mask now correctly propagates from leaf voxels up through
                // the hierarchy, so we can use direction-based culling.
                let packed_pos = (x as u16) | ((y as u16) << 4) | ((z as u16) << 8);

                if let Some(ref shell) = shell_map {
                    if let Some(&visible_faces) = shell.get(&packed_pos) {
                        // Compute demand mask based on camera direction
                        // We need to see faces pointing TOWARD the camera
                        let dx = voxel_center[0] - camera.position[0];
                        let dy = voxel_center[1] - camera.position[1];
                        let dz = voxel_center[2] - camera.position[2];

                        let mut demand_mask = 0u8;
                        // Chunk to right of camera (dx > 0) -> we see its left face (-X, bit 1)
                        // Chunk to left of camera (dx < 0) -> we see its right face (+X, bit 0)
                        if dx > 0.0 {
                            demand_mask |= 1 << 1;
                        } else {
                            demand_mask |= 1 << 0;
                        }
                        if dy > 0.0 {
                            demand_mask |= 1 << 3;
                        } else {
                            demand_mask |= 1 << 2;
                        }
                        if dz > 0.0 {
                            demand_mask |= 1 << 5;
                        } else {
                            demand_mask |= 1 << 4;
                        }

                        // Check if ANY demanded face is visible
                        if (visible_faces & demand_mask) == 0 {
                            stats.hierarchy_shell_culled += 1;
                            continue;
                        }
                    } else {
                        // Not in shell = fully interior, no exposed faces
                        stats.hierarchy_shell_culled += 1;
                        continue;
                    }
                } else {
                    stats.no_shell_available += 1;
                }

                if distance >= camera.config.lod_render_distance && sub_chunk.voxel_count > 0 {
                    let (pos, size) = if let Some(bbox) = sub_chunk.bounding_box {
                        // Convert local bbox (0..15) to world coordinates + size in world units.
                        bbox_local_to_world([world_x, world_y, world_z], scale, bbox)
                    } else {
                        // Fallback to full chunk size if bounding_box is None
                        let size = [scale as f32, scale as f32, scale as f32];
                        ([world_x, world_y, world_z], size)
                    };

                    result.push(VoxelInstance {
                        position: pos,
                        voxel_type: sub_chunk.dominant_type,
                        distance,
                        custom_color: Some(sub_chunk.average_color),
                        scale: size,
                        is_leaf_chunk: false,
                    });
                    stats.chunks_visible += 1;
                } else {
                    // Shell culling already done above, proceed with subdivision
                    let next_scale = scale / 16;
                    if next_scale > 1 {
                        collect_voxels_recursive(
                            sub_chunk,
                            [world_x, world_y, world_z],
                            next_scale,
                            camera,
                            result,
                            stats,
                            visible_chunks,
                        );
                    } else if sub_chunk.voxel_count > 0 {
                        // Final check before adding leaf chunk
                        if camera.frustum.test_aabb(
                            [world_x as f32, world_y as f32, world_z as f32],
                            [
                                (world_x + scale) as f32,
                                (world_y + scale) as f32,
                                (world_z + scale) as f32,
                            ],
                        ) {
                            result.push(VoxelInstance {
                                position: [world_x, world_y, world_z],
                                voxel_type: sub_chunk.dominant_type,
                                distance,
                                custom_color: Some(sub_chunk.average_color),
                                scale: [scale as f32, scale as f32, scale as f32],
                                is_leaf_chunk: true,
                            });
                            stats.chunks_visible += 1;

                            // Track visible leaf chunk position for GI system
                            let chunk_coord = glam::IVec3::new(
                                (world_x / 16) as i32,
                                (world_y / 16) as i32,
                                (world_z / 16) as i32,
                            );
                            visible_chunks.push(chunk_coord);
                        } else {
                            stats.frustum_aabb_culled += 1;
                        }
                    } else {
                        stats.empty_chunk_culled += 1;
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

    let mut stats = CullStats::default();
    let mut visible_chunks = Vec::new();
    collect_voxels_recursive(
        world.root(),
        [0, 0, 0], // World starts at origin
        scale,     // Scale of root voxels
        camera,
        &mut instances,
        &mut stats,
        &mut visible_chunks,
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

    let mut stats = CullStats::default();
    let mut visible_chunks = Vec::new();
    collect_voxels_recursive(
        world.root(),
        [0, 0, 0],
        scale,
        camera,
        &mut instances,
        &mut stats,
        &mut visible_chunks,
    );

    instances
}

/// Parallel culling - for hierarchical world, parallelize at top level of root chunk
/// Returns visible voxel instances and culling statistics
pub fn cull_visible_voxels_parallel(
    world: &World,
    camera: &Camera,
) -> (Vec<VoxelInstance>, CullStats, Vec<glam::IVec3>) {
    use rayon::prelude::*;

    // For hierarchical world, we can parallelize by processing top-level cells
    let world_size = world.world_size() as i64;
    let min = [0.0, 0.0, 0.0];
    let max = [world_size as f32, world_size as f32, world_size as f32];

    // Frustum cull the entire world first
    if !camera.frustum_cull_aabb(min, max) {
        return (Vec::new(), CullStats::default(), Vec::new());
    }

    // Collect top-level positions that have voxels
    let scale = 16i64.pow(world.hierarchy_depth() as u32 - 1);
    let root = world.root();

    let top_level_cells: Vec<_> = root.positions().map(|(x, y, z)| (x, y, z)).collect();

    // Process each top-level cell in parallel, collecting instances, stats, and visible chunk positions
    let results: Vec<(Vec<VoxelInstance>, CullStats, Vec<glam::IVec3>)> = top_level_cells
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

            let mut cell_stats = CullStats::default();
            cell_stats.chunks_examined += 1;

            if !camera.frustum_cull_aabb(cell_min, cell_max) {
                cell_stats.frustum_aabb_culled += 1;
                return Some((Vec::new(), cell_stats, Vec::new()));
            }

            let mut cell_instances = Vec::new();
            let mut visible_chunks = Vec::new();

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
                        scale: [scale as f32, scale as f32, scale as f32],
                        is_leaf_chunk: false,
                    });
                    cell_stats.chunks_visible += 1;
                }
                Voxel::Chunk(chunk) => {
                    // Recursively collect from sub-chunk until bottom chunk level.
                    let next_scale = scale >> 4;
                    if next_scale > 1 {
                        collect_voxels_recursive(
                            chunk,
                            [world_x, world_y, world_z],
                            next_scale,
                            camera,
                            &mut cell_instances,
                            &mut cell_stats,
                            &mut visible_chunks,
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
                                let (pos, size) = if let Some(bbox) = chunk.bounding_box {
                                    bbox_local_to_world([world_x, world_y, world_z], scale, bbox)
                                } else {
                                    (
                                        [world_x, world_y, world_z],
                                        [scale as f32, scale as f32, scale as f32],
                                    )
                                };

                                cell_instances.push(VoxelInstance {
                                    position: pos,
                                    voxel_type: chunk.dominant_type,
                                    distance,
                                    custom_color: Some(chunk.average_color),
                                    scale: size,
                                    is_leaf_chunk: false,
                                });
                                cell_stats.chunks_visible += 1;
                            } else {
                                cell_stats.empty_chunk_culled += 1;
                            }
                        } else {
                            if chunk.voxel_count > 0 {
                                cell_instances.push(VoxelInstance {
                                    position: [world_x, world_y, world_z],
                                    voxel_type: chunk.dominant_type,
                                    distance,
                                    custom_color: Some(chunk.average_color),
                                    scale: [scale as f32, scale as f32, scale as f32],
                                    is_leaf_chunk: true,
                                });
                                cell_stats.chunks_visible += 1;

                                // Track visible leaf chunk position for GI system
                                let chunk_coord = glam::IVec3::new(
                                    (world_x / 16) as i32,
                                    (world_y / 16) as i32,
                                    (world_z / 16) as i32,
                                );
                                visible_chunks.push(chunk_coord);
                            } else {
                                cell_stats.empty_chunk_culled += 1;
                            }
                        }
                    }
                }
            }

            Some((cell_instances, cell_stats, visible_chunks))
        })
        .collect();

    // Merge all results
    let mut all_instances = Vec::new();
    let mut total_stats = CullStats::default();
    let mut all_visible_chunks = Vec::new();
    for (instances, stats, chunks) in results {
        all_instances.extend(instances);
        total_stats.merge(&stats);
        all_visible_chunks.extend(chunks);
    }

    (all_instances, total_stats, all_visible_chunks)
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

/// Convert a local chunk bounding box (bbox in 0..15 coordinates) to world-space position and
/// world-space size using the supplied scale for the current voxel.
///
/// - bbox: [xmin, ymin, zmin, xmax, ymax, zmax]
/// - world_x/y/z: the world-space base position of the current voxel/chunk
/// - scale: size of a voxel at this level (e.g., 256, 4096, ...)
// bbox_local_to_world is provided by `lib_hierarchical` for consistent conversions.

#[cfg(test)]
mod tests_culling {
    use crate::lib_hierarchical::{bbox_local_to_world, World, WorldPos};
    use crate::palette::Palette;

    use super::{cull_visible_voxels_parallel, Camera, RenderConfig};

    #[test]
    fn test_bbox_local_to_world() {
        // scale = 16 -> unit = 1
        let bbox = [7u8, 7, 7, 7, 7, 7];
        let (pos, size) = bbox_local_to_world([100, 200, 300], 16, bbox);
        assert_eq!(pos, [107, 207, 307]);
        assert_eq!(size, [1.0, 1.0, 1.0]);

        // scale = 256 -> unit = 16
        let (pos2, size2) = bbox_local_to_world([0, 0, 0], 256, bbox);
        assert_eq!(pos2, [7 * 16, 7 * 16, 7 * 16]);
        assert_eq!(size2, [16.0, 16.0, 16.0]);

        // scale = 4096 -> unit = 256
        let (pos3, size3) = bbox_local_to_world([0, 0, 0], 4096, [0, 0, 0, 15, 15, 15]);
        assert_eq!(pos3, [0, 0, 0]);
        assert_eq!(size3, [4096.0, 4096.0, 4096.0]);
    }

    #[test]
    fn test_parallel_cull_preserves_chunk_dominant_type_for_lod() {
        let palette = Palette::from_string(
            "\
0 255 255 255 255 0 0 0 0 0
2 100 120 140 255 0 0 0 0 220
",
        )
        .unwrap();
        let mut world = World::new(2);
        world.set(WorldPos::new(0, 0, 0), 2);
        world.update_all_lod_metadata(&palette);
        world.generate_all_hierarchy_shells();

        let camera = Camera::with_config(
            [-32.0, 8.0, 8.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            RenderConfig {
                lod_render_distance: 0.0,
                far_plane: 256.0,
                fov_degrees: 90.0,
                near_plane: 0.1,
                culling_overscan: 0.0,
            },
        );

        let (instances, _, _) = cull_visible_voxels_parallel(&world, &camera);

        assert!(instances.iter().any(|instance| instance.voxel_type == 2));
    }

    #[test]
    fn test_recursive_cull_preserves_subchunk_dominant_type_for_lod() {
        let palette = Palette::from_string(
            "\
0 255 255 255 255 0 0 0 0 0
2 100 120 140 255 0 0 0 0 220
",
        )
        .unwrap();
        let mut world = World::new(3);
        world.set(WorldPos::new(0, 0, 0), 2);
        world.update_all_lod_metadata(&palette);
        world.generate_all_hierarchy_shells();

        let camera = Camera::with_config(
            [-32.0, 8.0, 8.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            RenderConfig {
                lod_render_distance: 0.0,
                far_plane: 4096.0,
                fov_degrees: 90.0,
                near_plane: 0.1,
                culling_overscan: 0.0,
            },
        );

        let (instances, _, _) = cull_visible_voxels_parallel(&world, &camera);

        assert!(instances.iter().any(|instance| instance.voxel_type == 2));
    }
}
