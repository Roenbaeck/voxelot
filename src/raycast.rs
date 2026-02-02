use crate::lib_hierarchical::{VoxelType, World, WorldPos};
use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastHit {
    pub position: WorldPos,
    pub normal: [i8; 3],
    pub distance: f32,
    pub voxel_type: VoxelType,
}

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }
}

/// Cast a ray against the hierarchical world
pub fn trace_ray(world: &World, ray: &Ray, max_distance: f32) -> Option<RaycastHit> {
    // Current position in world space
    let mut t = 0.0;

    // Avoid infinite loops
    if max_distance <= 0.0 {
        return None;
    }

    // Grid coordinate of the current voxel
    let mut current_voxel = ray.origin.floor().as_ivec3();

    // Determine step direction
    let step = ray.direction.signum().as_ivec3();

    // Determine t_delta (distance to travel one unit in each direction)
    let t_delta = 1.0 / ray.direction.abs();

    // Determine t_max (distance to first boundary)
    let frac = ray.origin - ray.origin.floor();

    let mut t_max = Vec3::new(
        if step.x > 0 {
            (1.0 - frac.x) / ray.direction.x
        } else if step.x < 0 {
            frac.x / -ray.direction.x
        } else {
            f32::INFINITY
        },
        if step.y > 0 {
            (1.0 - frac.y) / ray.direction.y
        } else if step.y < 0 {
            frac.y / -ray.direction.y
        } else {
            f32::INFINITY
        },
        if step.z > 0 {
            (1.0 - frac.z) / ray.direction.z
        } else if step.z < 0 {
            frac.z / -ray.direction.z
        } else {
            f32::INFINITY
        },
    );

    // Normal of the last face we entered
    let mut last_normal = [0i8; 3];

    // Safety break counter
    let mut steps = 0;
    const MAX_STEPS: usize = 1000;

    while t < max_distance && steps < MAX_STEPS {
        let wp = WorldPos::new(
            current_voxel.x as i64,
            current_voxel.y as i64,
            current_voxel.z as i64,
        );

        if let Some(voxel_type) = world.get(wp) {
            if voxel_type != 0 {
                return Some(RaycastHit {
                    position: wp,
                    normal: last_normal,
                    distance: t,
                    voxel_type,
                });
            }
        }

        // Advance to next voxel
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                // X Axis
                t = t_max.x;
                t_max.x += t_delta.x;
                current_voxel.x += step.x;
                last_normal = [-step.x as i8, 0, 0];
            } else {
                // Z Axis
                t = t_max.z;
                t_max.z += t_delta.z;
                current_voxel.z += step.z;
                last_normal = [0, 0, -step.z as i8];
            }
        } else {
            if t_max.y < t_max.z {
                // Y Axis
                t = t_max.y;
                t_max.y += t_delta.y;
                current_voxel.y += step.y;
                last_normal = [0, -step.y as i8, 0];
            } else {
                // Z Axis
                t = t_max.z;
                t_max.z += t_delta.z;
                current_voxel.z += step.z;
                last_normal = [0, 0, -step.z as i8];
            }
        }

        steps += 1;
    }

    None
}

// Helper to get placement position (previous empty voxel) and strict hit
pub fn get_target_and_placement(
    world: &World,
    ray: &Ray,
    max_distance: f32,
) -> Option<(RaycastHit, WorldPos)> {
    if let Some(hit) = trace_ray(world, ray, max_distance) {
        let placement = WorldPos::new(
            hit.position.x + hit.normal[0] as i64,
            hit.position.y + hit.normal[1] as i64,
            hit.position.z + hit.normal[2] as i64,
        );
        Some((hit, placement))
    } else {
        None
    }
}
