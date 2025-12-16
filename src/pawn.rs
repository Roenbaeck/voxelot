use crate::{Camera, World, WorldPos};



/// Object-safe pawn trait for simple player-controlled entities
use winit::keyboard::KeyCode;

pub trait Pawn {
    /// Handle a keyboard key event
    fn process_key(&mut self, key: KeyCode, pressed: bool);

    /// Per-frame update
    fn update(&mut self, dt: f32, world: &World, water_level: f32);

    /// Attach this pawn's transform to the supplied camera (for follow/cockpit views)
    fn attach_camera(&self, camera: &mut Camera);

    /// Optional: pose for drawing a custom mesh (position + yaw radians).
    ///
    /// This is intentionally minimal so we can later swap to a walker/bird mesh without
    /// touching physics/input code.
    fn debug_mesh_pose(&self) -> Option<([f32; 3], f32)> {
        None
    }

    /// Optional: provide a simple debug visualization as a colored box: (pos, scale, color)
    fn debug_viz(&self) -> Option<([f32;3],[f32;3],[f32;4])> {
        None
    }
}

/// Very small, easily-replaceable Boat implementation used for testing movement & collisions.
pub struct BoatPawn {
    pub pos: [f32; 3],
    pub yaw: f32,
    pub vel: [f32; 3],
    throttle: f32, // -1..1
    steer: f32,    // -1..1 (left/right)
    hull_half: [f32; 3],
    max_speed: f32,
}

impl BoatPawn {
    pub fn new(spawn_pos: [f32; 3], water_level: f32) -> Self {
        // Ensure the boat spawns on water
        let mut p = spawn_pos;
        p[1] = water_level + 0.5;
        Self {
            pos: p,
            yaw: 0.0,
            vel: [0.0, 0.0, 0.0],
            throttle: 0.0,
            steer: 0.0,
            hull_half: [1.0, 0.5, 2.0], // simple hull extents
            max_speed: 20.0,
        }
    }

    fn forward_vector(&self) -> [f32; 3] {
        [self.yaw.cos(), 0.0, self.yaw.sin()]
    }

    fn aabb_for_pos(&self, pos: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let min = [pos[0] - self.hull_half[0], pos[1] - self.hull_half[1], pos[2] - self.hull_half[2]];
        let max = [pos[0] + self.hull_half[0], pos[1] + self.hull_half[1], pos[2] + self.hull_half[2]];
        (min, max)
    }

    fn collides_with_world(&self, pos: [f32; 3], world: &World) -> bool {
        let (minf, maxf) = self.aabb_for_pos(pos);
        let xmin = minf[0].floor() as i64;
        let xmax = maxf[0].floor() as i64;
        let ymin = minf[1].floor() as i64;
        let ymax = maxf[1].floor() as i64;
        let zmin = minf[2].floor() as i64;
        let zmax = maxf[2].floor() as i64;

        for x in xmin..=xmax {
            for y in ymin..=ymax {
                for z in zmin..=zmax {
                    let wp = WorldPos::new(x, y, z);
                    if let Some(v) = world.get(wp) {
                        if v != 0 {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

impl Pawn for BoatPawn {
    fn process_key(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW => self.throttle = if pressed { 1.0 } else { 0.0 },
            KeyCode::KeyS => self.throttle = if pressed { -1.0 } else { 0.0 },
            KeyCode::KeyA => self.steer = if pressed { -1.0 } else { 0.0 },
            KeyCode::KeyD => self.steer = if pressed { 1.0 } else { 0.0 },
            _ => {}
        }
    }

    fn update(&mut self, dt: f32, world: &World, water_level: f32) {
        // Simple kinematic model
        let accel = 40.0 * self.throttle; // units/s^2
        let forward = self.forward_vector();
        self.vel[0] += forward[0] * accel * dt;
        self.vel[2] += forward[2] * accel * dt;

        // Steering rotates yaw proportional to steer and current speed
        let speed = (self.vel[0] * self.vel[0] + self.vel[2] * self.vel[2]).sqrt();
        self.yaw += self.steer * 1.5 * (1.0 + speed * 0.05) * dt;

        // Clamp speed
        if speed > self.max_speed {
            let scale = self.max_speed / speed;
            self.vel[0] *= scale;
            self.vel[2] *= scale;
        }

        // Damping
        self.vel[0] *= 0.995f32.powf(dt * 60.0);
        self.vel[2] *= 0.995f32.powf(dt * 60.0);

        // Predict motion and sweep-check
        let candidate = [self.pos[0] + self.vel[0] * dt, self.pos[1], self.pos[2] + self.vel[2] * dt];

        // Keep at water level (simple buoyancy)
        let mut candidate = candidate;
        candidate[1] = water_level + self.hull_half[1];

        if self.collides_with_world(candidate, world) {
            // Collision: zero out velocity and don't move
            self.vel = [0.0, 0.0, 0.0];
        } else {
            self.pos = candidate;
        }
    }

    fn attach_camera(&self, camera: &mut Camera) {
        // Put camera slightly above and behind the boat, oriented by yaw
        let offset_back = 6.0;
        let offset_up = 2.0;
        let back = [-self.yaw.cos() * offset_back, 0.0, -self.yaw.sin() * offset_back];
        camera.position[0] = self.pos[0] + back[0];
        camera.position[1] = self.pos[1] + offset_up;
        camera.position[2] = self.pos[2] + back[2];

        let forward = [self.yaw.cos(), 0.0, self.yaw.sin()];
        camera.update(camera.position, forward, [0.0, 1.0, 0.0]);
    }

    fn debug_mesh_pose(&self) -> Option<([f32; 3], f32)> {
        Some((self.pos, self.yaw))
    }

    fn debug_viz(&self) -> Option<([f32;3],[f32;3],[f32;4])> {
        Some((
            self.pos,
            [self.hull_half[0] * 2.0, self.hull_half[1] * 2.0, self.hull_half[2] * 2.0],
            [0.8, 0.25, 0.1, 1.0],
        ))
    }
}

