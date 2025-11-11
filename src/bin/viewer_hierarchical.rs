//! Modern wgpu viewer for hierarchical voxel chunks
//!
//! Features:
//! - 6DOF camera with WASD + mouse
//! - Frustum culling
//! - Occlusion culling
//! - LOD support
//! - Instanced rendering

use glam::{Mat4, Vec3};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

use std::collections::{HashMap, HashSet, VecDeque};
use voxelot::generate_chunk_mesh;
use voxelot::{Camera, RenderConfig, VisibilityCache, World, WorldPos};

macro_rules! viewer_debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "viewer-debug") {
            eprintln!($($arg)*);
        }
    };
}

const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;
const CONFIG_FILE: &str = "render_config.txt";
const GPU_CULL_WORKGROUP_SIZE: u32 = 64;

/// Voxel instance data for GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VoxelInstanceRaw {
    position: [f32; 3],
    voxel_type: u32,
    scale: f32,             // Scale factor (1.0 = 1x1x1, 16.0 = 16x16x16 chunk)
    custom_color: [f32; 4], // RGBA custom color (if custom_color.a > 0, use this instead of voxel_type)
}

/// Input layout for GPU culling compute pass (std430-friendly)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstanceInput {
    position: [f32; 3],
    scale: f32,
    custom_color: [f32; 4],
    voxel_type: u32,
    flags: u32,
    _padding: [u32; 2],
}

/// Parameters consumed by the GPU culling compute shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCullParams {
    camera_position: [f32; 3],
    candidate_count: u32,
    camera_forward: [f32; 3],
    _pad0: u32,
    near_plane: f32,
    far_plane: f32,
    lod_render_distance: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshVertexRaw {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// Uniforms for shader (matches shader layout exactly)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],      // 64 bytes
    sun_direction: [f32; 3], // 12 bytes
    fog_density: f32,        // 4 bytes (was _padding1)
    sun_color: [f32; 3],     // 12 bytes
    _padding2: f32,          // 4 bytes
    ambient_color: [f32; 3], // 12 bytes
    time_of_day: f32,        // 4 bytes
    // Total prior to padding: 112 bytes. Add 32 bytes tail padding (2 vec4 slots)
    _padding3: [f32; 8], // 32 bytes padding to satisfy 144-byte uniform expectations
}

/// Depth-of-field runtime settings (CPU-side convenience)
#[derive(Copy, Clone, Debug)]
struct DoFSettings {
    focal_distance: f32,
    focal_range: f32,
    blur_strength: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomExtractUniforms {
    threshold: f32,
    knee: f32,
    intensity: f32,
    _padding0: f32,
    source_texel_size: [f32; 2],
    _padding1: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomBlurUniforms {
    direction: [f32; 2],
    radius: f32,
    _padding0: f32,
    texel_size: [f32; 2],
    _padding1: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    bloom_strength: f32,
    saturation_boost: f32,
    exposure: f32,
    _padding0: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DoFSmoothUniforms {
    direction_radius: [f32; 4],
    texel_coc: [f32; 4],
}

#[derive(Copy, Clone, Debug)]
struct BloomSettings {
    threshold: f32,
    knee: f32,
    intensity: f32,
    bloom_strength: f32,
    saturation_boost: f32,
    exposure: f32,
    blur_radius: f32,
}

const DOF_UNIFORM_FLOATS: usize = 12;

const CUBE_VERTICES: &[CubeVertex] = &[
    // Front face
    CubeVertex {
        position: [0.0, 0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    CubeVertex {
        position: [1.0, 0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    CubeVertex {
        position: [1.0, 1.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    CubeVertex {
        position: [0.0, 0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    CubeVertex {
        position: [1.0, 1.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    CubeVertex {
        position: [0.0, 1.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    // Back face
    CubeVertex {
        position: [1.0, 0.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    CubeVertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    CubeVertex {
        position: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    CubeVertex {
        position: [1.0, 0.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    CubeVertex {
        position: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    CubeVertex {
        position: [1.0, 1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    // Top face
    CubeVertex {
        position: [0.0, 1.0, 0.0],
        normal: [0.0, 1.0, 0.0],
    },
    CubeVertex {
        position: [0.0, 1.0, 1.0],
        normal: [0.0, 1.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 1.0, 1.0],
        normal: [0.0, 1.0, 0.0],
    },
    CubeVertex {
        position: [0.0, 1.0, 0.0],
        normal: [0.0, 1.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 1.0, 1.0],
        normal: [0.0, 1.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 1.0, 0.0],
        normal: [0.0, 1.0, 0.0],
    },
    // Bottom face
    CubeVertex {
        position: [0.0, 0.0, 1.0],
        normal: [0.0, -1.0, 0.0],
    },
    CubeVertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, -1.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 0.0, 0.0],
        normal: [0.0, -1.0, 0.0],
    },
    CubeVertex {
        position: [0.0, 0.0, 1.0],
        normal: [0.0, -1.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 0.0, 0.0],
        normal: [0.0, -1.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 0.0, 1.0],
        normal: [0.0, -1.0, 0.0],
    },
    // Right face
    CubeVertex {
        position: [1.0, 0.0, 1.0],
        normal: [1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 0.0, 0.0],
        normal: [1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 1.0, 0.0],
        normal: [1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 0.0, 1.0],
        normal: [1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 1.0, 0.0],
        normal: [1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [1.0, 1.0, 1.0],
        normal: [1.0, 0.0, 0.0],
    },
    // Left face
    CubeVertex {
        position: [0.0, 0.0, 0.0],
        normal: [-1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [0.0, 0.0, 1.0],
        normal: [-1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [0.0, 1.0, 1.0],
        normal: [-1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [0.0, 0.0, 0.0],
        normal: [-1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [0.0, 1.0, 1.0],
        normal: [-1.0, 0.0, 0.0],
    },
    CubeVertex {
        position: [0.0, 1.0, 0.0],
        normal: [-1.0, 0.0, 0.0],
    },
];

/// Camera controller for 6DOF movement
struct CameraController {
    camera: Camera,
    base_speed: f32,
    speed_multiplier: f32,
    sensitivity: f32,
    yaw: f32,
    pitch: f32,
    // Input state
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    rotate_left: bool,
    rotate_right: bool,
    rotate_up: bool,
    rotate_down: bool,
}

impl CameraController {
    const MIN_SPEED_MULTIPLIER: f32 = 0.05;
    const MAX_SPEED_MULTIPLIER: f32 = 25.0;
    const ROTATION_SPEED: f32 = std::f32::consts::PI / 2.0; // radians per second

    fn new(position: [f32; 3]) -> Self {
        // Load config from file or use defaults
        let config = RenderConfig::load_or_default(CONFIG_FILE);
        println!("Loaded render config:");
        println!(
            "  LOD subdivide distance: {}",
            config.lod_subdivide_distance
        );
        println!("  Far plane: {}", config.far_plane);
        println!("  FOV: {}°", config.fov_degrees);

        let mut this = Self {
            camera: Camera::with_config(position, [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], config),
            base_speed: 10.0,
            speed_multiplier: 1.0,
            sensitivity: 0.002,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.3, // look slightly downward by default
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            rotate_left: false,
            rotate_right: false,
            rotate_up: false,
            rotate_down: false,
        };
        // Initialize forward/up vectors from yaw/pitch immediately
        this.update_camera_vectors();
        this
    }

    fn process_keyboard(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.down = pressed,
            KeyCode::ArrowUp => self.rotate_up = pressed,
            KeyCode::ArrowDown => self.rotate_down = pressed,
            KeyCode::ArrowLeft => self.rotate_left = pressed,
            KeyCode::ArrowRight => self.rotate_right = pressed,
            KeyCode::Minus if pressed => {
                self.speed_multiplier =
                    (self.speed_multiplier * 0.8).max(Self::MIN_SPEED_MULTIPLIER);
                println!("Camera speed multiplier: {:.2}", self.speed_multiplier);
            }
            KeyCode::Equal if pressed => {
                self.speed_multiplier =
                    (self.speed_multiplier * 1.25).min(Self::MAX_SPEED_MULTIPLIER);
                println!("Camera speed multiplier: {:.2}", self.speed_multiplier);
            }
            KeyCode::Digit0 if pressed => {
                self.speed_multiplier = 1.0;
                println!("Camera speed multiplier reset to 1.00");
            }
            // Runtime config adjustments (only on key press, not release)
            KeyCode::KeyQ if pressed => {
                self.camera.config.lod_subdivide_distance =
                    (self.camera.config.lod_subdivide_distance - 50.0).max(50.0);
                println!(
                    "LOD subdivide distance: {:.0}",
                    self.camera.config.lod_subdivide_distance
                );
            }
            KeyCode::KeyE if pressed => {
                self.camera.config.lod_subdivide_distance =
                    (self.camera.config.lod_subdivide_distance + 50.0).min(2000.0);
                println!(
                    "LOD subdivide distance: {:.0}",
                    self.camera.config.lod_subdivide_distance
                );
            }
            KeyCode::KeyZ if pressed => {
                self.camera.config.far_plane = (self.camera.config.far_plane - 500.0).max(1000.0);
                self.camera.far = self.camera.config.far_plane;
                self.update_camera_vectors(); // Recalculate frustum
                println!("Far plane: {:.0}", self.camera.config.far_plane);
            }
            KeyCode::KeyC if pressed => {
                self.camera.config.far_plane = (self.camera.config.far_plane + 500.0).min(20000.0);
                self.camera.far = self.camera.config.far_plane;
                self.update_camera_vectors(); // Recalculate frustum
                println!("Far plane: {:.0}", self.camera.config.far_plane);
            }
            _ => {}
        }
    }

    fn process_mouse(&mut self, delta_x: f64, delta_y: f64) {
        self.yaw += delta_x as f32 * self.sensitivity;
        self.pitch -= delta_y as f32 * self.sensitivity;
        // Clamp pitch
        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.1,
            std::f32::consts::FRAC_PI_2 - 0.1,
        );
        self.update_camera_vectors();
    }

    fn update_camera_vectors(&mut self) {
        let forward = [
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        ];
        let up = [0.0, 1.0, 0.0];
        self.camera.update(self.camera.position, forward, up);
    }

    fn update(&mut self, dt: f32) {
        if self.rotate_left {
            self.yaw -= Self::ROTATION_SPEED * dt;
        }
        if self.rotate_right {
            self.yaw += Self::ROTATION_SPEED * dt;
        }
        if self.rotate_up {
            self.pitch += Self::ROTATION_SPEED * dt;
        }
        if self.rotate_down {
            self.pitch -= Self::ROTATION_SPEED * dt;
        }

        if self.rotate_left || self.rotate_right || self.rotate_up || self.rotate_down {
            self.pitch = self.pitch.clamp(
                -std::f32::consts::FRAC_PI_2 + 0.1,
                std::f32::consts::FRAC_PI_2 - 0.1,
            );
            self.update_camera_vectors();
        }

        let mut velocity = [0.0, 0.0, 0.0];
        let forward = self.camera.forward;
        let right = self.camera.right();

        if self.forward {
            velocity[0] += forward[0];
            velocity[1] += forward[1];
            velocity[2] += forward[2];
        }
        if self.backward {
            velocity[0] -= forward[0];
            velocity[1] -= forward[1];
            velocity[2] -= forward[2];
        }
        if self.right {
            velocity[0] += right[0];
            velocity[1] += right[1];
            velocity[2] += right[2];
        }
        if self.left {
            velocity[0] -= right[0];
            velocity[1] -= right[1];
            velocity[2] -= right[2];
        }
        if self.up {
            velocity[1] += 1.0;
        }
        if self.down {
            velocity[1] -= 1.0;
        }

        // Normalize velocity
        let len =
            (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2])
                .sqrt();
        if len > 0.001 {
            velocity[0] /= len;
            velocity[1] /= len;
            velocity[2] /= len;
        }

        // Apply movement
        let dynamic_speed = self.base_speed * self.speed_multiplier * self.distance_speed_scale();
        let pos = self.camera.position;
        self.camera.position = [
            pos[0] + velocity[0] * dynamic_speed * dt,
            pos[1] + velocity[1] * dynamic_speed * dt,
            pos[2] + velocity[2] * dynamic_speed * dt,
        ];

        self.update_camera_vectors();
    }

    fn distance_speed_scale(&self) -> f32 {
        // Bump movement speed as the camera gets farther from the origin or higher above the terrain.
        // ln_1p keeps growth gentle close to the ground while still allowing high-altitude flyovers.
        let pos = self.camera.position;
        let horizontal = (pos[0] * pos[0] + pos[2] * pos[2]).sqrt();
        let altitude = pos[1].abs();

        let horizontal_scale = (horizontal / 250.0).ln_1p() + 1.0;
        let altitude_scale = (altitude / 50.0).ln_1p() + 1.0;

        horizontal_scale.max(altitude_scale).clamp(1.0, 12.0)
    }
}

/// Main application state
struct App {
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    mesh_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    cube_vertex_buffer: Option<wgpu::Buffer>,
    instance_buffer: Option<wgpu::Buffer>,
    gpu_input_buffer: Option<wgpu::Buffer>,
    gpu_input_capacity: usize,
    gpu_readback_buffer: Option<wgpu::Buffer>,
    gpu_readback_capacity: usize,
    // Mesh cache: per-leaf-chunk mesh GPU buffers
    mesh_cache: HashMap<(i64, i64, i64), (wgpu::Buffer, wgpu::Buffer, u32)>, // (vbuf, ibuf, index_count)
    instance_capacity: usize,
    cull_pipeline: Option<wgpu::ComputePipeline>,
    cull_bind_group_layout: Option<wgpu::BindGroupLayout>,
    cull_bind_group: Option<wgpu::BindGroup>,
    cull_params_buffer: Option<wgpu::Buffer>,

    world: World,
    camera_controller: CameraController,
    visibility_cache: VisibilityCache,
    pending_chunk_meshes: VecDeque<(i64, i64, i64)>,
    pending_chunk_set: HashSet<(i64, i64, i64)>,
    last_frame: Instant,
    frame_count: u64,
    last_fps_print: Instant,

    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,

    // Lighting state
    time_of_day: f32,
    fog_density: f32,

    // LOD state
    lod_distance: f32,

    // Post-processing state
    dof_pipeline: Option<wgpu::RenderPipeline>,
    dof_bind_group_layout: Option<wgpu::BindGroupLayout>,
    dof_bind_group: Option<wgpu::BindGroup>,
    dof_uniform_buffer: Option<wgpu::Buffer>,
    post_sampler: Option<wgpu::Sampler>,
    offscreen_color_texture: Option<wgpu::Texture>,
    offscreen_color_view: Option<wgpu::TextureView>,
    offscreen_depth_texture: Option<wgpu::Texture>,
    offscreen_depth_view: Option<wgpu::TextureView>,
    post_color_texture: Option<wgpu::Texture>,
    post_color_view: Option<wgpu::TextureView>,
    bloom_ping_texture: Option<wgpu::Texture>,
    bloom_ping_view: Option<wgpu::TextureView>,
    bloom_pong_texture: Option<wgpu::Texture>,
    bloom_pong_view: Option<wgpu::TextureView>,
    dof_smooth_ping_texture: Option<wgpu::Texture>,
    dof_smooth_ping_view: Option<wgpu::TextureView>,
    bloom_extract_pipeline: Option<wgpu::RenderPipeline>,
    bloom_blur_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    dof_smooth_pipeline: Option<wgpu::RenderPipeline>,
    bloom_extract_bind_group_layout: Option<wgpu::BindGroupLayout>,
    bloom_blur_bind_group_layout: Option<wgpu::BindGroupLayout>,
    composite_bind_group_layout: Option<wgpu::BindGroupLayout>,
    dof_smooth_bind_group_layout: Option<wgpu::BindGroupLayout>,
    bloom_extract_bind_group: Option<wgpu::BindGroup>,
    bloom_blur_horizontal_bind_group: Option<wgpu::BindGroup>,
    bloom_blur_vertical_bind_group: Option<wgpu::BindGroup>,
    dof_smooth_horizontal_bind_group: Option<wgpu::BindGroup>,
    dof_smooth_vertical_bind_group: Option<wgpu::BindGroup>,
    composite_bind_group: Option<wgpu::BindGroup>,
    bloom_extract_uniform_buffer: Option<wgpu::Buffer>,
    bloom_blur_horizontal_uniform_buffer: Option<wgpu::Buffer>,
    bloom_blur_vertical_uniform_buffer: Option<wgpu::Buffer>,
    composite_uniform_buffer: Option<wgpu::Buffer>,
    dof_smooth_horizontal_uniform_buffer: Option<wgpu::Buffer>,
    dof_smooth_vertical_uniform_buffer: Option<wgpu::Buffer>,
    dof_settings: DoFSettings,
    dof_enabled: bool,
    bloom_settings: BloomSettings,
    bloom_enabled: bool,
}

impl App {
    fn new() -> Self {
        // Create world with test data (depth 3 = 4,096 units)
        let mut world = World::new(3);

        println!("Creating world (size: {} units)...", world.world_size());

        let mut initial_camera = if cfg!(feature = "test-block-world") {
            [50.0, 15.0, 65.0]
        } else {
            [680.0, 50.0, 670.0]
        };

        if cfg!(feature = "test-block-world") {
            viewer_debug!("Creating test block: 3x5x7 voxels at (50,10,50)");
            let mut count = 0;
            for x in 0..3 {
                for y in 0..5 {
                    for z in 0..7 {
                        world.set(WorldPos::new(50 + x, 10 + y, 50 + z), 2);
                        count += 1;
                    }
                }
            }
            viewer_debug!("Test block created: {} voxels", count);

            if cfg!(feature = "viewer-debug") {
                viewer_debug!("Verifying voxels for test block:");
                for (x, y, z) in [(50, 10, 50), (51, 11, 51), (52, 14, 56)] {
                    if let Some(vtype) = world.get(WorldPos::new(x, y, z)) {
                        viewer_debug!("  ({},{},{}) = type {}", x, y, z, vtype);
                    } else {
                        viewer_debug!("  ({},{},{}) = NONE!", x, y, z);
                    }
                }

                let test_pos = WorldPos::new(50, 10, 50);
                viewer_debug!("Checking world structure around test block...");
                if let Some(vtype) = world.get(test_pos) {
                    viewer_debug!(
                        "  Voxel at ({},{},{}) = type {}",
                        test_pos.x,
                        test_pos.y,
                        test_pos.z,
                        vtype
                    );
                }
                if let Some(depth) = world.depth_at(test_pos) {
                    viewer_debug!(
                        "  Depth at this position: {} (0 = Solid, 1+ = Chunk with N levels below)",
                        depth
                    );
                }
                let chunk_origin =
                    WorldPos::new(test_pos.x & !15, test_pos.y & !15, test_pos.z & !15);
                viewer_debug!(
                    "  Expected leaf chunk origin: ({},{},{})",
                    chunk_origin.x,
                    chunk_origin.y,
                    chunk_origin.z
                );
                if let Some(chunk) = world.get_leaf_chunk_at_origin(chunk_origin) {
                    viewer_debug!("  ✓ Found leaf chunk with {} voxels", chunk.iter().count());
                } else {
                    viewer_debug!("  ✗ Leaf chunk not found");
                }
            }
        } else {
            println!("Loading OSM voxel data...");
            match std::fs::read_to_string("osm_voxels.txt") {
                Ok(content) => {
                    let mut voxel_count = 0;
                    let mut min_x = i64::MAX;
                    let mut max_x = i64::MIN;
                    let mut min_y = i64::MAX;
                    let mut max_y = i64::MIN;
                    let mut min_z = i64::MAX;
                    let mut max_z = i64::MIN;
                    for line in content.lines() {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() == 4 {
                            if let (Ok(x), Ok(y), Ok(z), Ok(voxel_type)) = (
                                parts[0].parse::<i64>(),
                                parts[1].parse::<i64>(),
                                parts[2].parse::<i64>(),
                                parts[3].parse::<u8>(),
                            ) {
                                world.set(WorldPos::new(x, y, z), voxel_type);
                                voxel_count += 1;
                                min_x = min_x.min(x);
                                max_x = max_x.max(x);
                                min_y = min_y.min(y);
                                max_y = max_y.max(y);
                                min_z = min_z.min(z);
                                max_z = max_z.max(z);
                            }
                        }
                    }
                    println!("Loaded {} voxels from OSM data", voxel_count);

                    if voxel_count > 0 {
                        let center_x = (min_x + max_x) as f32 * 0.5;
                        let center_z = (min_z + max_z) as f32 * 0.5;
                        // Hover above the tallest structure with a bit of headroom
                        let eye_height = (max_y + 30).max(40) as f32;
                        initial_camera = [center_x, eye_height, center_z];
                    }
                }
                Err(e) => {
                    println!(
                        "Failed to load osm_voxels.txt: {}. Using fallback world generation.",
                        e
                    );

                    for x in 0..100 {
                        for z in 0..100 {
                            if (x + z) % 3 == 0 {
                                world.set(WorldPos::new(x, 0, z), 1);
                            }
                        }
                    }

                    for i in 0..5 {
                        let x = 30 + i * 20;
                        for y in 1..=(10 + i * 3) {
                            world.set(WorldPos::new(x, y, 50), 2);
                        }
                    }

                    initial_camera = [60.0, 40.0, 120.0];
                }
            }
        }

        println!("World created with voxels");

        println!("Updating LOD metadata...");
        world.update_all_lod_metadata();
        println!("LOD metadata updated");

        println!("\n=== Controls ===");
        println!("Movement: WASD + Space/Shift (up/down)");
        println!("Look: Right Mouse + drag");
        println!("Rotate: Arrow Keys (Left/Right yaw, Up/Down pitch)");
        println!("Camera Speed: -/+ (decrease/increase multiplier), 0 reset");
        println!("Camera LOD Distance: Q/E (decrease/increase)");
        println!("Draw Distance: Z/C (decrease/increase)");
        println!("Chunk LOD Distance: K/L (decrease/increase)");
        println!("Time of Day: T (cycle through day/night)");
        println!("Fog Density: F/G (decrease/increase)");
        println!("Depth of Field: / (toggle), , and . adjust focus");
        println!("Bloom: B (toggle)");
        println!("Quit: ESC");
        println!("================\n");

        Self {
            window: None,
            surface: None,
            device: None,
            queue: None,
            config: None,
            render_pipeline: None,
            mesh_pipeline: None,
            uniform_buffer: None,
            bind_group: None,
            cube_vertex_buffer: None,
            instance_buffer: None,
            gpu_input_buffer: None,
            gpu_input_capacity: 0,
            gpu_readback_buffer: None,
            gpu_readback_capacity: 0,
            mesh_cache: HashMap::new(),
            instance_capacity: 0,
            cull_pipeline: None,
            cull_bind_group_layout: None,
            cull_bind_group: None,
            cull_params_buffer: None,
            world,
            camera_controller: CameraController::new(initial_camera),
            visibility_cache: VisibilityCache::new(),
            pending_chunk_meshes: VecDeque::new(),
            pending_chunk_set: HashSet::new(),
            last_frame: Instant::now(),
            frame_count: 0,
            last_fps_print: Instant::now(),
            mouse_pressed: false,
            last_mouse_pos: None,
            time_of_day: 0.5,    // Start at noon
            fog_density: 0.0015, // Default fog density
            lod_distance: 800.0, // Default LOD render distance
            dof_pipeline: None,
            dof_bind_group_layout: None,
            dof_bind_group: None,
            dof_uniform_buffer: None,
            post_sampler: None,
            offscreen_color_texture: None,
            offscreen_color_view: None,
            offscreen_depth_texture: None,
            offscreen_depth_view: None,
            post_color_texture: None,
            post_color_view: None,
            bloom_ping_texture: None,
            bloom_ping_view: None,
            bloom_pong_texture: None,
            bloom_pong_view: None,
            dof_smooth_ping_texture: None,
            dof_smooth_ping_view: None,
            bloom_extract_pipeline: None,
            bloom_blur_pipeline: None,
            composite_pipeline: None,
            dof_smooth_pipeline: None,
            bloom_extract_bind_group_layout: None,
            bloom_blur_bind_group_layout: None,
            composite_bind_group_layout: None,
            dof_smooth_bind_group_layout: None,
            bloom_extract_bind_group: None,
            bloom_blur_horizontal_bind_group: None,
            bloom_blur_vertical_bind_group: None,
            dof_smooth_horizontal_bind_group: None,
            dof_smooth_vertical_bind_group: None,
            composite_bind_group: None,
            bloom_extract_uniform_buffer: None,
            bloom_blur_horizontal_uniform_buffer: None,
            bloom_blur_vertical_uniform_buffer: None,
            composite_uniform_buffer: None,
            dof_smooth_horizontal_uniform_buffer: None,
            dof_smooth_vertical_uniform_buffer: None,
            dof_settings: DoFSettings {
                focal_distance: 120.0,
                focal_range: 16.0,
                blur_strength: 1.6,
            },
            dof_enabled: true,
            bloom_settings: BloomSettings {
                threshold: 0.7,
                knee: 0.6,
                intensity: 1.8,
                bloom_strength: 1.6,
                saturation_boost: 1.5,
                exposure: 1.18,
                blur_radius: 3.8,
            },
            bloom_enabled: true,
        }
    }

    fn save_config(&self) {
        if let Err(e) = self.camera_controller.camera.config.save(CONFIG_FILE) {
            eprintln!("Failed to save config: {}", e);
        } else {
            println!("Saved render config to {}", CONFIG_FILE);
        }
    }

    fn pack_dof_uniforms(&self, blur_strength: f32) -> [f32; DOF_UNIFORM_FLOATS] {
        let mut data = [0.0_f32; DOF_UNIFORM_FLOATS];
        data[0] = self.dof_settings.focal_distance;
        data[1] = self.dof_settings.focal_range;
        data[2] = blur_strength;
        data[3] = self.camera_controller.camera.near;
        data[4] = self.camera_controller.camera.far;
        data
    }

    fn build_bloom_extract_uniforms(
        &self,
        src_width: u32,
        src_height: u32,
    ) -> BloomExtractUniforms {
        BloomExtractUniforms {
            threshold: self.bloom_settings.threshold,
            knee: self.bloom_settings.knee,
            intensity: self.bloom_settings.intensity,
            _padding0: 0.0,
            source_texel_size: [
                1.0 / src_width.max(1) as f32,
                1.0 / src_height.max(1) as f32,
            ],
            _padding1: [0.0; 2],
        }
    }

    fn build_bloom_blur_uniforms(
        &self,
        target_width: u32,
        target_height: u32,
        direction: [f32; 2],
    ) -> BloomBlurUniforms {
        BloomBlurUniforms {
            direction,
            radius: self.bloom_settings.blur_radius,
            _padding0: 0.0,
            texel_size: [
                1.0 / target_width.max(1) as f32,
                1.0 / target_height.max(1) as f32,
            ],
            _padding1: [0.0; 2],
        }
    }

    fn build_composite_uniforms(&self) -> CompositeUniforms {
        CompositeUniforms {
            bloom_strength: if self.bloom_enabled {
                self.bloom_settings.bloom_strength
            } else {
                0.0
            },
            saturation_boost: self.bloom_settings.saturation_boost,
            exposure: self.bloom_settings.exposure,
            _padding0: 0.0,
        }
    }

    fn process_lighting_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::KeyT => {
                // Cycle through time of day: noon -> sunset -> night -> sunrise -> noon
                self.time_of_day = (self.time_of_day + 0.25) % 1.0;
                let time_name = match (self.time_of_day * 4.0) as u32 {
                    0 => "Midnight",
                    1 => "Sunrise",
                    2 => "Noon",
                    3 => "Sunset",
                    _ => "Unknown",
                };
                println!("Time of day: {} ({:.2})", time_name, self.time_of_day);
            }
            KeyCode::KeyF => {
                // Decrease fog density
                self.fog_density = (self.fog_density - 0.0002).max(0.0);
                println!("Fog density: {:.4}", self.fog_density);
            }
            KeyCode::KeyG => {
                // Increase fog density
                self.fog_density = (self.fog_density + 0.0002).min(0.01);
                println!("Fog density: {:.4}", self.fog_density);
            }
            KeyCode::KeyB => {
                self.bloom_enabled = !self.bloom_enabled;
                println!(
                    "Bloom {}",
                    if self.bloom_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
            KeyCode::KeyK => {
                // Decrease LOD distance (more detail at distance)
                self.lod_distance = (self.lod_distance - 100.0).max(100.0);
                self.camera_controller.camera.config.lod_render_distance = self.lod_distance;
                println!("LOD distance: {:.0} units", self.lod_distance);
            }
            KeyCode::KeyL => {
                // Increase LOD distance (less detail at distance)
                self.lod_distance = (self.lod_distance + 100.0).min(5000.0);
                self.camera_controller.camera.config.lod_render_distance = self.lod_distance;
                println!("LOD distance: {:.0} units", self.lod_distance);
            }
            KeyCode::Comma => {
                self.dof_settings.focal_distance =
                    (self.dof_settings.focal_distance - 10.0).max(10.0);
                println!(
                    "DoF focal distance: {:.1}",
                    self.dof_settings.focal_distance
                );
            }
            KeyCode::Period => {
                self.dof_settings.focal_distance =
                    (self.dof_settings.focal_distance + 10.0).min(5000.0);
                println!(
                    "DoF focal distance: {:.1}",
                    self.dof_settings.focal_distance
                );
            }
            KeyCode::BracketLeft => {
                self.dof_settings.focal_range = (self.dof_settings.focal_range - 5.0).max(5.0);
                println!("DoF focal range: {:.1}", self.dof_settings.focal_range);
            }
            KeyCode::BracketRight => {
                self.dof_settings.focal_range = (self.dof_settings.focal_range + 5.0).min(500.0);
                println!("DoF focal range: {:.1}", self.dof_settings.focal_range);
            }
            KeyCode::Semicolon => {
                self.dof_settings.blur_strength = (self.dof_settings.blur_strength - 0.1).max(0.0);
                println!("DoF blur strength: {:.2}", self.dof_settings.blur_strength);
            }
            KeyCode::Quote => {
                self.dof_settings.blur_strength = (self.dof_settings.blur_strength + 0.1).min(2.5);
                println!("DoF blur strength: {:.2}", self.dof_settings.blur_strength);
            }
            KeyCode::Slash => {
                self.dof_enabled = !self.dof_enabled;
                println!(
                    "DoF {}",
                    if self.dof_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
            _ => {}
        }
    }

    fn recreate_offscreen_targets(&mut self) {
        let (Some(device), Some(config)) = (self.device.as_ref(), self.config.as_ref()) else {
            return;
        };

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Color Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let post_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Post DoF Color Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let post_color_view =
            post_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let dof_smooth_ping_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DoF Smooth Ping Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dof_smooth_ping_view =
            dof_smooth_ping_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bloom_extent = wgpu::Extent3d {
            width: (config.width / 2).max(1),
            height: (config.height / 2).max(1),
            depth_or_array_layers: 1,
        };

        let bloom_ping_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Ping Texture"),
            size: bloom_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_ping_view =
            bloom_ping_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bloom_pong_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Pong Texture"),
            size: bloom_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_pong_view =
            bloom_pong_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.offscreen_color_view = Some(color_view);
        self.offscreen_color_texture = Some(color_texture);
        self.offscreen_depth_view = Some(depth_view);
        self.offscreen_depth_texture = Some(depth_texture);
        self.post_color_view = Some(post_color_view);
        self.post_color_texture = Some(post_color_texture);
        self.dof_smooth_ping_view = Some(dof_smooth_ping_view);
        self.dof_smooth_ping_texture = Some(dof_smooth_ping_texture);
        self.bloom_ping_view = Some(bloom_ping_view);
        self.bloom_ping_texture = Some(bloom_ping_texture);
        self.bloom_pong_view = Some(bloom_pong_view);
        self.bloom_pong_texture = Some(bloom_pong_texture);

        self.update_dof_bind_group();
        self.update_dof_smooth_bind_groups();
        self.update_bloom_uniforms();
        self.update_bloom_bind_groups();
    }

    fn update_dof_bind_group(&mut self) {
        let (
            Some(device),
            Some(layout),
            Some(color_view),
            Some(depth_view),
            Some(sampler),
            Some(ubo),
        ) = (
            self.device.as_ref(),
            self.dof_bind_group_layout.as_ref(),
            self.offscreen_color_view.as_ref(),
            self.offscreen_depth_view.as_ref(),
            self.post_sampler.as_ref(),
            self.dof_uniform_buffer.as_ref(),
        )
        else {
            return;
        };

        self.dof_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DoF Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        }));
    }

    fn update_dof_smooth_bind_groups(&mut self) {
        let (
            Some(device),
            Some(layout),
            Some(post_view),
            Some(depth_view),
            Some(sampler),
            Some(dof_buffer),
            Some(horizontal_buffer),
            Some(vertical_buffer),
            Some(ping_view),
        ) = (
            self.device.as_ref(),
            self.dof_smooth_bind_group_layout.as_ref(),
            self.post_color_view.as_ref(),
            self.offscreen_depth_view.as_ref(),
            self.post_sampler.as_ref(),
            self.dof_uniform_buffer.as_ref(),
            self.dof_smooth_horizontal_uniform_buffer.as_ref(),
            self.dof_smooth_vertical_uniform_buffer.as_ref(),
            self.dof_smooth_ping_view.as_ref(),
        )
        else {
            return;
        };

        self.dof_smooth_horizontal_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF Smooth Horizontal Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dof_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(post_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: horizontal_buffer.as_entire_binding(),
                    },
                ],
            }));

        self.dof_smooth_vertical_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF Smooth Vertical Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dof_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(ping_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: vertical_buffer.as_entire_binding(),
                    },
                ],
            }));
    }

    fn build_dof_smooth_uniforms(
        &self,
        target_width: u32,
        target_height: u32,
        direction: [f32; 2],
    ) -> DoFSmoothUniforms {
        let blur_strength = if self.dof_enabled {
            self.dof_settings.blur_strength
        } else {
            0.0
        };
        let width = target_width.max(1) as f32;
        let height = target_height.max(1) as f32;
        let radius = if blur_strength <= 0.01 {
            0.0
        } else {
            0.85 + blur_strength * 3.2
        };
        let coc_scale = if blur_strength <= 0.01 {
            0.0
        } else {
            0.9 + blur_strength * 1.15
        };
        let coc_bias = -0.05;

        DoFSmoothUniforms {
            direction_radius: [direction[0], direction[1], radius, coc_scale],
            texel_coc: [1.0 / width, 1.0 / height, coc_bias, 0.0],
        }
    }

    fn update_dof_smooth_uniforms(&mut self) {
        let (Some(queue), Some(config), Some(h_buffer), Some(v_buffer)) = (
            self.queue.as_ref(),
            self.config.as_ref(),
            self.dof_smooth_horizontal_uniform_buffer.as_ref(),
            self.dof_smooth_vertical_uniform_buffer.as_ref(),
        ) else {
            return;
        };

        let width = config.width.max(1);
        let height = config.height.max(1);
        let horizontal = self.build_dof_smooth_uniforms(width, height, [1.0, 0.0]);
        let vertical = self.build_dof_smooth_uniforms(width, height, [0.0, 1.0]);

        queue.write_buffer(h_buffer, 0, bytemuck::cast_slice(&[horizontal]));
        queue.write_buffer(v_buffer, 0, bytemuck::cast_slice(&[vertical]));
    }

    fn update_bloom_uniforms(&mut self) {
        let (Some(queue), Some(config)) = (self.queue.as_ref(), self.config.as_ref()) else {
            return;
        };

        let width = config.width.max(1);
        let height = config.height.max(1);
        let bloom_width = (config.width / 2).max(1);
        let bloom_height = (config.height / 2).max(1);

        if let Some(buffer) = self.bloom_extract_uniform_buffer.as_ref() {
            let data = self.build_bloom_extract_uniforms(width, height);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[data]));
        }

        if let Some(buffer) = self.bloom_blur_horizontal_uniform_buffer.as_ref() {
            let data = self.build_bloom_blur_uniforms(bloom_width, bloom_height, [1.0, 0.0]);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[data]));
        }

        if let Some(buffer) = self.bloom_blur_vertical_uniform_buffer.as_ref() {
            let data = self.build_bloom_blur_uniforms(bloom_width, bloom_height, [0.0, 1.0]);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[data]));
        }

        if let Some(buffer) = self.composite_uniform_buffer.as_ref() {
            let data = self.build_composite_uniforms();
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[data]));
        }
    }

    fn update_bloom_bind_groups(&mut self) {
        if self.post_color_view.is_none()
            || self.bloom_ping_view.is_none()
            || self.bloom_pong_view.is_none()
        {
            return;
        }

        let (Some(device), Some(extract_layout), Some(blur_layout), Some(composite_layout)) = (
            self.device.as_ref(),
            self.bloom_extract_bind_group_layout.as_ref(),
            self.bloom_blur_bind_group_layout.as_ref(),
            self.composite_bind_group_layout.as_ref(),
        ) else {
            return;
        };

        let Some(post_view) = self.post_color_view.as_ref() else {
            return;
        };
        let Some(bloom_ping_view) = self.bloom_ping_view.as_ref() else {
            return;
        };
        let Some(bloom_pong_view) = self.bloom_pong_view.as_ref() else {
            return;
        };

        if let (Some(ubo), Some(sampler), Some(blur_horizontal_ubo), Some(blur_vertical_ubo)) = (
            self.bloom_extract_uniform_buffer.as_ref(),
            self.post_sampler.as_ref(),
            self.bloom_blur_horizontal_uniform_buffer.as_ref(),
            self.bloom_blur_vertical_uniform_buffer.as_ref(),
        ) {
            self.bloom_extract_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom Extract Bind Group"),
                    layout: extract_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(post_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                }));

            self.bloom_blur_horizontal_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom Blur Horizontal Bind Group"),
                    layout: blur_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: blur_horizontal_ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(bloom_ping_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                }));

            self.bloom_blur_vertical_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom Blur Vertical Bind Group"),
                    layout: blur_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: blur_vertical_ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(bloom_pong_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                }));
        }

        if let (Some(composite_ubo), Some(sampler)) = (
            self.composite_uniform_buffer.as_ref(),
            self.post_sampler.as_ref(),
        ) {
            self.composite_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Composite Bind Group"),
                    layout: composite_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: composite_ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(post_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(bloom_ping_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                }));
        }
    }

    fn ensure_gpu_input_buffer(&mut self, device: &wgpu::Device, required: usize) {
        if required == 0 {
            return;
        }

        let needed_capacity = required.next_power_of_two();
        if self.gpu_input_capacity < needed_capacity || self.gpu_input_buffer.is_none() {
            if let Some(old_buffer) = self.gpu_input_buffer.take() {
                old_buffer.destroy();
            }

            self.gpu_input_capacity = needed_capacity;
            self.gpu_input_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Instance Input Buffer"),
                size: (self.gpu_input_capacity * std::mem::size_of::<GpuInstanceInput>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));

            self.cull_bind_group = None; // Force rebuild with new buffer
        }

        if self.gpu_readback_capacity < needed_capacity || self.gpu_readback_buffer.is_none() {
            if let Some(old_buffer) = self.gpu_readback_buffer.take() {
                old_buffer.destroy();
            }

            self.gpu_readback_capacity = needed_capacity;
            self.gpu_readback_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Cull Readback Buffer"),
                size: (self.gpu_readback_capacity * std::mem::size_of::<GpuInstanceInput>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    fn ensure_cull_bind_group(&mut self, device: &wgpu::Device) {
        if self.cull_bind_group.is_some() {
            return;
        }

        let (Some(layout), Some(input_buffer), Some(params_buffer)) = (
            self.cull_bind_group_layout.as_ref(),
            self.gpu_input_buffer.as_ref(),
            self.cull_params_buffer.as_ref(),
        ) else {
            return;
        };

        self.cull_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Cull Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    fn run_gpu_culling(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        candidate_count: usize,
    ) -> Option<Vec<GpuInstanceInput>> {
        if candidate_count == 0 {
            return Some(Vec::new());
        }

        let (Some(cull_pipeline), Some(cull_bind_group), Some(input_buffer), Some(readback_buffer)) = (
            self.cull_pipeline.as_ref(),
            self.cull_bind_group.as_ref(),
            self.gpu_input_buffer.as_ref(),
            self.gpu_readback_buffer.as_ref(),
        ) else {
            return None;
        };

        let byte_len = (candidate_count * std::mem::size_of::<GpuInstanceInput>()) as u64;
        if byte_len == 0 {
            return Some(Vec::new());
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Cull Encoder"),
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU Cull Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(cull_pipeline);
        compute_pass.set_bind_group(0, cull_bind_group, &[]);
        let dispatch_x =
            ((candidate_count as u32) + GPU_CULL_WORKGROUP_SIZE - 1) / GPU_CULL_WORKGROUP_SIZE;
        compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
        drop(compute_pass);

        encoder.copy_buffer_to_buffer(input_buffer, 0, readback_buffer, 0, byte_len);

        queue.submit(std::iter::once(encoder.finish()));
        if !device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|status| status.wait_finished())
            .unwrap_or(false)
        {
            return None;
        }

        let slice = readback_buffer.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        if !device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|status| status.wait_finished())
            .unwrap_or(false)
        {
            return None;
        }

        match receiver.recv() {
            Ok(Ok(())) => {}
            _ => return None,
        }

        let data = slice.get_mapped_range();
        let inputs: &[GpuInstanceInput] = bytemuck::cast_slice(&data);
        let mut result = Vec::with_capacity(candidate_count);
        result.extend_from_slice(&inputs[..candidate_count]);
        drop(data);
        readback_buffer.unmap();

        Some(result)
    }

    async fn init_wgpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();

        // Create instance
        let instance = wgpu::Instance::default();

        // Create surface
        let surface = instance.create_surface(window.clone()).unwrap();

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Request device
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/voxel.wgsl").into()),
        });

        // Create bind group layout for uniforms
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create instanced-cube render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Slot 0: Per-vertex data (position + normal)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<CubeVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![4 => Float32x3, 5 => Float32x3],
                    },
                    // Slot 1: Per-instance data (position, type, scale, custom_color)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<VoxelInstanceRaw>()
                            as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3,  // position
                            1 => Uint32,     // voxel_type
                            2 => Float32,    // scale
                            3 => Float32x4   // custom_color (RGBA)
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back), // Enable backface culling (easy win)
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create mesh pipeline (non-instanced, per-vertex position/normal/color)
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_mesh"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MeshVertexRaw>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
                    }
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_mesh"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Depth-of-field post-processing resources
        let initial_dof = self.pack_dof_uniforms(self.dof_settings.blur_strength);
        let dof_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DoF Uniform Buffer"),
            contents: bytemuck::cast_slice(&initial_dof),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let dof_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dof_blur.wgsl").into()),
        });

        let dof_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DoF Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let dof_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF Pipeline Layout"),
            bind_group_layouts: &[&dof_bind_group_layout],
            push_constant_ranges: &[],
        });

        let dof_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Pipeline"),
            layout: Some(&dof_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &dof_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &dof_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let dof_smooth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Smooth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dof_smooth.wgsl").into()),
        });

        let dof_smooth_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DoF Smooth Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let dof_smooth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DoF Smooth Pipeline Layout"),
                bind_group_layouts: &[&dof_smooth_bind_group_layout],
                push_constant_ranges: &[],
            });

        let dof_smooth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Smooth Pipeline"),
            layout: Some(&dof_smooth_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &dof_smooth_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &dof_smooth_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let dof_smooth_horizontal_uniforms = DoFSmoothUniforms {
            direction_radius: [1.0, 0.0, 0.0, 0.0],
            texel_coc: [0.0, 0.0, 0.0, 0.0],
        };
        let dof_smooth_horizontal_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DoF Smooth Horizontal Uniform Buffer"),
                contents: bytemuck::cast_slice(&[dof_smooth_horizontal_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let dof_smooth_vertical_uniforms = DoFSmoothUniforms {
            direction_radius: [0.0, 1.0, 0.0, 0.0],
            texel_coc: [0.0, 0.0, 0.0, 0.0],
        };
        let dof_smooth_vertical_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DoF Smooth Vertical Uniform Buffer"),
                contents: bytemuck::cast_slice(&[dof_smooth_vertical_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let post_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DoF Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bloom_extract_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Extract Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/bloom_extract.wgsl").into(),
            ),
        });

        let bloom_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bloom_blur.wgsl").into()),
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/post_composite.wgsl").into(),
            ),
        });

        let bloom_extract_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Extract Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let bloom_blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Blur Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let bloom_extract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Extract Pipeline Layout"),
                bind_group_layouts: &[&bloom_extract_bind_group_layout],
                push_constant_ranges: &[],
            });

        let bloom_blur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Blur Pipeline Layout"),
                bind_group_layouts: &[&bloom_blur_bind_group_layout],
                push_constant_ranges: &[],
            });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        let bloom_extract_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Extract Pipeline"),
                layout: Some(&bloom_extract_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &bloom_extract_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &bloom_extract_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        let bloom_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Blur Pipeline"),
            layout: Some(&bloom_blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &bloom_blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let bloom_extract_uniforms =
            self.build_bloom_extract_uniforms(config.width.max(1), config.height.max(1));
        let bloom_extract_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bloom Extract Uniform Buffer"),
                contents: bytemuck::cast_slice(&[bloom_extract_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bloom_width = (config.width / 2).max(1);
        let bloom_height = (config.height / 2).max(1);

        let bloom_blur_horizontal_uniforms =
            self.build_bloom_blur_uniforms(bloom_width, bloom_height, [1.0, 0.0]);
        let bloom_blur_horizontal_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bloom Blur Horizontal Uniform Buffer"),
                contents: bytemuck::cast_slice(&[bloom_blur_horizontal_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bloom_blur_vertical_uniforms =
            self.build_bloom_blur_uniforms(bloom_width, bloom_height, [0.0, 1.0]);
        let bloom_blur_vertical_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bloom Blur Vertical Uniform Buffer"),
                contents: bytemuck::cast_slice(&[bloom_blur_vertical_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let composite_uniforms = self.build_composite_uniforms();
        let composite_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Composite Uniform Buffer"),
                contents: bytemuck::cast_slice(&[composite_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create uniform buffer with proper size for Uniforms struct
        let uniforms = Uniforms {
            mvp: [[0.0; 4]; 4], // Will be filled in render()
            sun_direction: [0.5, 1.0, 0.3],
            fog_density: 0.0015,
            sun_color: [1.0, 0.95, 0.8],
            _padding2: 0.0,
            ambient_color: [0.3, 0.35, 0.45],
            time_of_day: 0.5,
            _padding3: [0.0; 8],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/gpu_cull.wgsl").into()),
        });

        let cull_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GPU Cull Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU Cull Pipeline Layout"),
            bind_group_layouts: &[&cull_bind_group_layout],
            push_constant_ranges: &[],
        });

        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GPU Cull Pipeline"),
            layout: Some(&cull_pipeline_layout),
            module: &cull_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cull_params_init = GpuCullParams {
            camera_position: [0.0; 3],
            candidate_count: 0,
            camera_forward: [0.0, 0.0, -1.0],
            _pad0: 0,
            near_plane: 0.1,
            far_plane: 1000.0,
            lod_render_distance: 1000.0,
            _pad1: 0.0,
        };

        let cull_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU Cull Params Buffer"),
            contents: bytemuck::bytes_of(&cull_params_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create cube vertex buffer with positions and normals
        let cube_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        self.cull_pipeline = Some(cull_pipeline);
        self.cull_bind_group_layout = Some(cull_bind_group_layout);
        self.cull_params_buffer = Some(cull_params_buffer);
        self.cull_bind_group = None;

        self.dof_pipeline = Some(dof_pipeline);
        self.dof_bind_group_layout = Some(dof_bind_group_layout);
        self.dof_uniform_buffer = Some(dof_uniform_buffer);
        self.post_sampler = Some(post_sampler);
        self.dof_bind_group = None;
        self.dof_smooth_pipeline = Some(dof_smooth_pipeline);
        self.dof_smooth_bind_group_layout = Some(dof_smooth_bind_group_layout);
        self.dof_smooth_horizontal_uniform_buffer = Some(dof_smooth_horizontal_uniform_buffer);
        self.dof_smooth_vertical_uniform_buffer = Some(dof_smooth_vertical_uniform_buffer);
        self.dof_smooth_horizontal_bind_group = None;
        self.dof_smooth_vertical_bind_group = None;
        self.dof_smooth_ping_texture = None;
        self.dof_smooth_ping_view = None;
        self.bloom_extract_pipeline = Some(bloom_extract_pipeline);
        self.bloom_blur_pipeline = Some(bloom_blur_pipeline);
        self.composite_pipeline = Some(composite_pipeline);
        self.bloom_extract_bind_group_layout = Some(bloom_extract_bind_group_layout);
        self.bloom_blur_bind_group_layout = Some(bloom_blur_bind_group_layout);
        self.composite_bind_group_layout = Some(composite_bind_group_layout);
        self.bloom_extract_uniform_buffer = Some(bloom_extract_uniform_buffer);
        self.bloom_blur_horizontal_uniform_buffer = Some(bloom_blur_horizontal_uniform_buffer);
        self.bloom_blur_vertical_uniform_buffer = Some(bloom_blur_vertical_uniform_buffer);
        self.composite_uniform_buffer = Some(composite_uniform_buffer);
        self.bloom_extract_bind_group = None;
        self.bloom_blur_horizontal_bind_group = None;
        self.bloom_blur_vertical_bind_group = None;
        self.composite_bind_group = None;

        self.window = Some(window);
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);
        self.render_pipeline = Some(render_pipeline);
        self.mesh_pipeline = Some(mesh_pipeline);
        self.uniform_buffer = Some(uniform_buffer);
        self.bind_group = Some(bind_group);
        self.cube_vertex_buffer = Some(cube_vertex_buffer);

        self.recreate_offscreen_targets();

        viewer_debug!("DEBUG: mesh_pipeline created successfully");
        println!("wgpu initialized");
    }

    fn render(&mut self) {
        let device = self.device.as_ref().unwrap().clone();
        let queue = self.queue.as_ref().unwrap().clone();
        let config = self.config.as_ref().unwrap().clone();

        // Update camera
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.camera_controller.update(dt);

        // Cull visible voxels using cached visibility
        let cull_start = Instant::now();
        let visible = self
            .visibility_cache
            .update(&self.camera_controller.camera, &self.world);
        let cull_time = cull_start.elapsed();

        let gpu_inputs: Vec<GpuInstanceInput> = visible
            .iter()
            .map(|v| {
                let custom_color_f32 = if let Some(rgba) = v.custom_color {
                    [
                        rgba[0] as f32 / 255.0,
                        rgba[1] as f32 / 255.0,
                        rgba[2] as f32 / 255.0,
                        rgba[3] as f32 / 255.0,
                    ]
                } else if v.is_leaf_chunk {
                    [0.4, 0.4, 0.45, 0.6]
                } else {
                    [0.0, 0.0, 0.0, 0.0]
                };

                GpuInstanceInput {
                    position: [
                        v.position[0] as f32,
                        v.position[1] as f32,
                        v.position[2] as f32,
                    ],
                    scale: v.scale as f32,
                    custom_color: custom_color_f32,
                    voxel_type: v.voxel_type as u32,
                    flags: 0,
                    _padding: [0; 2],
                }
            })
            .collect();

        let gpu_candidate_count = gpu_inputs.len();

        if gpu_candidate_count > 0 {
            self.ensure_gpu_input_buffer(&device, gpu_candidate_count);
            if let Some(buffer) = self.gpu_input_buffer.as_ref() {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&gpu_inputs));
            }
        }

        if let Some(params_buffer) = self.cull_params_buffer.as_ref() {
            let gpu_params = GpuCullParams {
                camera_position: self.camera_controller.camera.position,
                candidate_count: gpu_candidate_count as u32,
                camera_forward: self.camera_controller.camera.forward,
                _pad0: 0,
                near_plane: self.camera_controller.camera.near,
                far_plane: self.camera_controller.camera.far,
                lod_render_distance: self.lod_distance,
                _pad1: 0.0,
            };
            queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&gpu_params));
        }

        if gpu_candidate_count > 0 {
            self.ensure_cull_bind_group(&device);
        }

        let grouping_start = Instant::now();
        // Collect unique leaf chunk origins flagged by the culler
        let mut leaf_chunks: HashSet<(i64, i64, i64)> = HashSet::new();
        for v in &visible {
            if v.is_leaf_chunk {
                leaf_chunks.insert((v.position[0], v.position[1], v.position[2]));
            }
        }
        let grouping_time = grouping_start.elapsed();

        let mesh_start = Instant::now();
        // Build mesh for any chunk present (near leaf chunk), and mark those chunks for drawing
        let mut cpu_mesh_keys: HashSet<(i64, i64, i64)> = HashSet::new();
        let mut new_meshes_created = 0;
        let mut chunks_not_found = 0;
        let mut missing_chunks: HashSet<(i64, i64, i64)> = HashSet::new();

        for &key in &leaf_chunks {
            if self.mesh_cache.contains_key(&key) {
                cpu_mesh_keys.insert(key);
            } else {
                missing_chunks.insert(key);
                if !self.pending_chunk_set.contains(&key) {
                    self.pending_chunk_meshes.push_back(key);
                    self.pending_chunk_set.insert(key);
                }
            }
        }

        const MAX_MESHES_PER_FRAME: usize = 4;
        let mut processed_meshes = 0;
        while processed_meshes < MAX_MESHES_PER_FRAME {
            let Some(key) = self.pending_chunk_meshes.pop_front() else {
                break;
            };
            self.pending_chunk_set.remove(&key);

            match self
                .world
                .get_leaf_chunk_at_origin(WorldPos::new(key.0, key.1, key.2))
            {
                Some(chunk) => {
                    let mesh = generate_chunk_mesh(chunk);
                    if mesh.indices.is_empty() {
                        processed_meshes += 1;
                        continue;
                    }

                    if new_meshes_created == 0 {
                        let num_voxels = chunk.iter().count();
                        viewer_debug!(
                            "DEBUG first mesh at ({},{},{}): {} voxels in chunk, {} vertices, {} triangles",
                            key.0, key.1, key.2, num_voxels, mesh.vertices.len(), mesh.indices.len() / 3
                        );
                        for (i, v) in mesh.vertices.iter().enumerate() {
                            viewer_debug!(
                                "  vertex {}: pos=[{:.1},{:.1},{:.1}] normal=[{:.1},{:.1},{:.1}]",
                                i,
                                v.position[0],
                                v.position[1],
                                v.position[2],
                                v.normal[0],
                                v.normal[1],
                                v.normal[2]
                            );
                        }
                    }

                    let vb_data: Vec<MeshVertexRaw> = mesh
                        .vertices
                        .iter()
                        .map(|v| MeshVertexRaw {
                            position: [
                                v.position[0] + key.0 as f32,
                                v.position[1] + key.1 as f32,
                                v.position[2] + key.2 as f32,
                            ],
                            normal: v.normal,
                            color: v.color,
                        })
                        .collect();
                    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Chunk Mesh Vertex Buffer"),
                        contents: bytemuck::cast_slice(&vb_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Chunk Mesh Index Buffer"),
                        contents: bytemuck::cast_slice(&mesh.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    viewer_debug!(
                        "Created mesh for chunk ({},{},{}): {} vertices, {} triangles",
                        key.0,
                        key.1,
                        key.2,
                        mesh.vertices.len(),
                        mesh.indices.len() / 3
                    );
                    self.mesh_cache
                        .insert(key, (vbuf, ibuf, mesh.indices.len() as u32));
                    new_meshes_created += 1;

                    if leaf_chunks.contains(&key) {
                        cpu_mesh_keys.insert(key);
                        missing_chunks.remove(&key);
                    }
                }
                None => {
                    chunks_not_found += 1;
                }
            }

            processed_meshes += 1;
        }
        let mesh_time = mesh_start.elapsed();

        if chunks_not_found > 0 && self.frame_count == 0 {
            println!(
                "Warning: {} out of {} potential chunks not found (OSM voxels are not in subdivided chunks)",
                chunks_not_found,
                leaf_chunks.len()
            );
        }

        let gpu_results = if gpu_candidate_count > 0 {
            self.run_gpu_culling(&device, &queue, gpu_candidate_count)
        } else {
            Some(Vec::new())
        };

        // Convert remaining instances (exclude those belonging to meshed chunks)
        let instance_start = Instant::now();
        let (instances, draw_mesh_keys) = if let Some(mapped_inputs) = gpu_results {
            let mut mesh_selection: HashSet<(i64, i64, i64)> = HashSet::new();
            let mut out: Vec<VoxelInstanceRaw> = Vec::new();

            for (input, v) in mapped_inputs.iter().zip(visible.iter()) {
                if input.flags == 0 {
                    continue;
                }

                if v.is_leaf_chunk {
                    let key = (v.position[0], v.position[1], v.position[2]);
                    if cpu_mesh_keys.contains(&key) {
                        mesh_selection.insert(key);
                        continue;
                    }
                }

                out.push(VoxelInstanceRaw {
                    position: input.position,
                    voxel_type: input.voxel_type,
                    scale: input.scale,
                    custom_color: input.custom_color,
                });
            }

            (out, mesh_selection)
        } else {
            let fallback_instances: Vec<VoxelInstanceRaw> = visible
                .iter()
                .filter(|v| {
                    if v.is_leaf_chunk {
                        !cpu_mesh_keys.contains(&(v.position[0], v.position[1], v.position[2]))
                    } else {
                        true
                    }
                })
                .map(|v| {
                    let custom_color_f32 = if let Some(rgba) = v.custom_color {
                        [
                            rgba[0] as f32 / 255.0,
                            rgba[1] as f32 / 255.0,
                            rgba[2] as f32 / 255.0,
                            rgba[3] as f32 / 255.0,
                        ]
                    } else if v.is_leaf_chunk {
                        [0.4, 0.4, 0.45, 0.6]
                    } else {
                        [0.0, 0.0, 0.0, 0.0]
                    };

                    VoxelInstanceRaw {
                        position: [
                            v.position[0] as f32,
                            v.position[1] as f32,
                            v.position[2] as f32,
                        ],
                        voxel_type: v.voxel_type as u32,
                        scale: v.scale as f32,
                        custom_color: custom_color_f32,
                    }
                })
                .collect();

            (fallback_instances, cpu_mesh_keys.clone())
        };
        let instance_time = instance_start.elapsed();

        if cfg!(feature = "viewer-debug") && self.frame_count % 60 == 0 {
            viewer_debug!(
                "Mesh stats: {} cached meshes, {} new this frame, {} potential chunks, pending {}, fallback instances {}",
                self.mesh_cache.len(),
                new_meshes_created,
                leaf_chunks.len(),
                self.pending_chunk_meshes.len(),
                instances.len()
            );
            viewer_debug!(
                "GPU cull: candidates {} -> draw meshes {} -> instanced {}",
                gpu_candidate_count,
                draw_mesh_keys.len(),
                instances.len()
            );
        }

        let has_meshes_to_draw = !draw_mesh_keys.is_empty();
        if instances.is_empty() && !has_meshes_to_draw {
            return; // Nothing to render at all
        }

        if self.instance_capacity < instances.len() {
            self.instance_capacity = instances.len().next_power_of_two();

            if let Some(old_buffer) = self.instance_buffer.take() {
                old_buffer.destroy();
            }

            self.instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Instance Buffer"),
                size: (self.instance_capacity * std::mem::size_of::<VoxelInstanceRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // Write data to the buffer (only if we have instances)
        if !instances.is_empty() {
            queue.write_buffer(
                self.instance_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&instances),
            );
        }

        // Get surface texture
        let output_result = {
            let surface = self.surface.as_ref().unwrap();
            surface.get_current_texture()
        };
        let output = match output_result {
            Ok(texture) => texture,
            Err(e) => {
                eprintln!("Surface error: {:?}", e);
                // Handle specific surface errors
                match e {
                    wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                        // Recreate surface
                        if let Some(surface) = self.surface.as_ref() {
                            surface.configure(&device, &config);
                        }
                        self.recreate_offscreen_targets();
                    }
                    wgpu::SurfaceError::OutOfMemory => {
                        eprintln!("Out of memory!");
                        return;
                    }
                    wgpu::SurfaceError::Timeout => {
                        eprintln!("Surface timeout!");
                    }
                    _ => {
                        eprintln!("Other surface error");
                    }
                }
                return;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create MVP matrix using glam (column-major, right-handed)
        let aspect = config.width as f32 / config.height as f32;
        let projection = Mat4::perspective_rh(
            self.camera_controller.camera.fov,
            aspect,
            self.camera_controller.camera.near,
            self.camera_controller.camera.far,
        );
        let eye = Vec3::from(self.camera_controller.camera.position);
        let center = eye + Vec3::from(self.camera_controller.camera.forward) * 100.0; // look far ahead
        let up = Vec3::from(self.camera_controller.camera.up);
        let view_mat = Mat4::look_at_rh(eye, center, up);
        // Convert from OpenGL-style NDC (glam) to wgpu's 0..1 depth range
        const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
        ]);
        let mvp = OPENGL_TO_WGPU_MATRIX * projection * view_mat;
        let mvp_cols: [[f32; 4]; 4] = mvp.to_cols_array_2d();

        // Calculate lighting based on time of day
        let time_angle = self.time_of_day * std::f32::consts::TAU; // 0..2π
        let sun_height = time_angle.sin();
        let sun_direction = [time_angle.cos() * 0.5, sun_height, 0.3];

        let (sun_color, ambient_color) = if self.time_of_day < 0.25 || self.time_of_day > 0.75 {
            // Night - moon (cool blue)
            ([0.3, 0.3, 0.5], [0.05, 0.05, 0.15])
        } else if self.time_of_day < 0.35 || self.time_of_day > 0.65 {
            // Sunrise/sunset - warm orange/red
            ([1.0, 0.6, 0.3], [0.3, 0.2, 0.2])
        } else {
            // Day - bright yellow sun, blue sky ambient
            ([1.0, 0.95, 0.8], [0.3, 0.35, 0.45])
        };

        // Update uniforms with MVP and lighting data
        let uniforms = Uniforms {
            mvp: mvp_cols,
            sun_direction,
            fog_density: self.fog_density,
            sun_color,
            _padding2: 0.0,
            ambient_color,
            time_of_day: self.time_of_day,
            _padding3: [0.0; 8],
        };

        queue.write_buffer(
            self.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        // Create command encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        if gpu_candidate_count > 0 {
            if let (Some(cull_pipeline), Some(cull_bind_group)) =
                (self.cull_pipeline.as_ref(), self.cull_bind_group.as_ref())
            {
                let dispatch_x = ((gpu_candidate_count as u32) + GPU_CULL_WORKGROUP_SIZE - 1)
                    / GPU_CULL_WORKGROUP_SIZE;
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GPU Cull Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(cull_pipeline);
                compute_pass.set_bind_group(0, cull_bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
            }
        }

        let offscreen_color_view = self
            .offscreen_color_view
            .as_ref()
            .expect("offscreen color view missing");
        let offscreen_depth_view = self
            .offscreen_depth_view
            .as_ref()
            .expect("offscreen depth view missing");

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: offscreen_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: offscreen_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw meshed chunks first
            if has_meshes_to_draw {
                render_pass.set_pipeline(self.mesh_pipeline.as_ref().unwrap());
                render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
                let mut drawn_meshes = 0;
                for (&key, (vbuf, ibuf, index_count)) in &self.mesh_cache {
                    // Only draw meshes that are in view this frame
                    if draw_mesh_keys.contains(&key) {
                        render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..*index_count, 0, 0..1);
                        drawn_meshes += 1;
                    }
                }
                if cfg!(feature = "viewer-debug") && self.frame_count == 0 {
                    viewer_debug!(
                        "DEBUG: Drew {} meshes (has_meshes_to_draw={}, mesh_keys.len()={}, mesh_cache.len()={})",
                        drawn_meshes, has_meshes_to_draw, draw_mesh_keys.len(), self.mesh_cache.len()
                    );
                }
            }

            // Draw remaining instanced cubes
            render_pass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            render_pass.set_vertex_buffer(0, self.cube_vertex_buffer.as_ref().unwrap().slice(..));
            if !instances.is_empty() {
                render_pass.set_vertex_buffer(1, self.instance_buffer.as_ref().unwrap().slice(..));
                render_pass.draw(0..36, 0..instances.len() as u32); // 36 vertices for a cube
            }
        }

        if self.dof_bind_group.is_none() {
            self.update_dof_bind_group();
        }

        self.update_bloom_uniforms();

        if self.composite_bind_group.is_none()
            || self.bloom_extract_bind_group.is_none()
            || self.bloom_blur_horizontal_bind_group.is_none()
            || self.bloom_blur_vertical_bind_group.is_none()
        {
            self.update_bloom_bind_groups();
        }

        if let (Some(dof_pipeline), Some(dof_bind_group), Some(dof_buffer), Some(post_color_view)) = (
            self.dof_pipeline.as_ref(),
            self.dof_bind_group.as_ref(),
            self.dof_uniform_buffer.as_ref(),
            self.post_color_view.as_ref(),
        ) {
            let blur_strength = if self.dof_enabled {
                self.dof_settings.blur_strength
            } else {
                0.0
            };
            let gpu_uniforms = self.pack_dof_uniforms(blur_strength);
            queue.write_buffer(dof_buffer, 0, bytemuck::cast_slice(&gpu_uniforms));

            let mut post_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DoF Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: post_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            post_pass.set_pipeline(dof_pipeline);
            post_pass.set_bind_group(0, dof_bind_group, &[]);
            post_pass.draw(0..3, 0..1);
        } else {
            eprintln!("DoF resources unavailable; skipping post-process!");
        }

        let run_smoothing = self.dof_enabled && self.dof_settings.blur_strength > 0.01;
        if run_smoothing {
            self.update_dof_smooth_uniforms();

            if let (
                Some(dof_smooth_pipeline),
                Some(horizontal_bind_group),
                Some(vertical_bind_group),
                Some(dof_smooth_ping_view),
                Some(post_color_view),
            ) = (
                self.dof_smooth_pipeline.as_ref(),
                self.dof_smooth_horizontal_bind_group.as_ref(),
                self.dof_smooth_vertical_bind_group.as_ref(),
                self.dof_smooth_ping_view.as_ref(),
                self.post_color_view.as_ref(),
            ) {
                let mut smooth_pass_h = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DoF Smooth Horizontal Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: dof_smooth_ping_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                smooth_pass_h.set_pipeline(dof_smooth_pipeline);
                smooth_pass_h.set_bind_group(0, horizontal_bind_group, &[]);
                smooth_pass_h.draw(0..3, 0..1);
                drop(smooth_pass_h);

                let mut smooth_pass_v = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DoF Smooth Vertical Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: post_color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                smooth_pass_v.set_pipeline(dof_smooth_pipeline);
                smooth_pass_v.set_bind_group(0, vertical_bind_group, &[]);
                smooth_pass_v.draw(0..3, 0..1);
            }
        }

        if self.bloom_enabled {
            if let (
                Some(bloom_extract_pipeline),
                Some(bloom_extract_bind_group),
                Some(bloom_ping_view),
            ) = (
                self.bloom_extract_pipeline.as_ref(),
                self.bloom_extract_bind_group.as_ref(),
                self.bloom_ping_view.as_ref(),
            ) {
                let mut extract_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Extract Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: bloom_ping_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                extract_pass.set_pipeline(bloom_extract_pipeline);
                extract_pass.set_bind_group(0, bloom_extract_bind_group, &[]);
                extract_pass.draw(0..3, 0..1);
            }

            if let (Some(bloom_blur_pipeline), Some(horizontal_bind_group), Some(bloom_pong_view)) = (
                self.bloom_blur_pipeline.as_ref(),
                self.bloom_blur_horizontal_bind_group.as_ref(),
                self.bloom_pong_view.as_ref(),
            ) {
                let mut blur_pass_h = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Blur Horizontal Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: bloom_pong_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                blur_pass_h.set_pipeline(bloom_blur_pipeline);
                blur_pass_h.set_bind_group(0, horizontal_bind_group, &[]);
                blur_pass_h.draw(0..3, 0..1);
            }

            if let (Some(bloom_blur_pipeline), Some(vertical_bind_group), Some(bloom_ping_view)) = (
                self.bloom_blur_pipeline.as_ref(),
                self.bloom_blur_vertical_bind_group.as_ref(),
                self.bloom_ping_view.as_ref(),
            ) {
                let mut blur_pass_v = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Blur Vertical Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: bloom_ping_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                blur_pass_v.set_pipeline(bloom_blur_pipeline);
                blur_pass_v.set_bind_group(0, vertical_bind_group, &[]);
                blur_pass_v.draw(0..3, 0..1);
            }
        }

        if let (Some(composite_pipeline), Some(composite_bind_group)) = (
            self.composite_pipeline.as_ref(),
            self.composite_bind_group.as_ref(),
        ) {
            let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            composite_pass.set_pipeline(composite_pipeline);
            composite_pass.set_bind_group(0, composite_bind_group, &[]);
            composite_pass.draw(0..3, 0..1);
        } else {
            eprintln!("Composite resources unavailable; skipping final pass!");
        }

        queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Stats
        self.frame_count += 1;
        if now.duration_since(self.last_fps_print).as_secs() >= 1 {
            let total_visible = visible.len();
            println!(
                "FPS: {}, Visible items: {}, Leaf chunks: {}, Meshed chunks: {}, Pending: {}, Fallback: {}, Cull: {:.2}ms, Group: {:.2}ms, Mesh: {:.2}ms, Instances: {:.2}ms",
                self.frame_count,
                total_visible,
                leaf_chunks.len(),
                draw_mesh_keys.len(),
                self.pending_chunk_meshes.len(),
                missing_chunks.len(),
                cull_time.as_secs_f64() * 1000.0,
                grouping_time.as_secs_f64() * 1000.0,
                mesh_time.as_secs_f64() * 1000.0,
                instance_time.as_secs_f64() * 1000.0
            );
            self.frame_count = 0;
            self.last_fps_print = now;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attrs = WindowAttributes::default()
                .with_title("Hierarchical Voxel Viewer")
                .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));

            let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

            pollster::block_on(self.init_wgpu(window));
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Continuously update and render
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Close requested");
                self.save_config();
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                self.camera_controller.process_keyboard(key, pressed);

                // Handle lighting controls on key press only
                if pressed {
                    self.process_lighting_key(key);
                }

                if key == KeyCode::Escape && pressed {
                    self.save_config();
                    event_loop.exit();
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                self.mouse_pressed = state == ElementState::Pressed;
                if !self.mouse_pressed {
                    self.last_mouse_pos = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some(last_pos) = self.last_mouse_pos {
                        let delta_x = position.x - last_pos.0;
                        let delta_y = position.y - last_pos.1;
                        self.camera_controller.process_mouse(delta_x, delta_y);
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    if let Some(config) = self.config.as_mut() {
                        config.width = new_size.width;
                        config.height = new_size.height;
                    }

                    if let (Some(surface), Some(device), Some(config)) = (
                        self.surface.as_ref(),
                        self.device.as_ref(),
                        self.config.as_ref(),
                    ) {
                        surface.configure(device, config);
                    }

                    self.recreate_offscreen_targets();

                    if let Some(config) = self.config.as_ref() {
                        self.camera_controller.camera.aspect =
                            config.width as f32 / config.height as f32;
                    }

                    let cam = &self.camera_controller.camera;
                    self.camera_controller
                        .camera
                        .update(cam.position, cam.forward, cam.up);
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            _ => {}
        }
    }
}

// Matrix math helpers
// (old CPU-side math helpers removed; using glam Mat4/Vec3 instead)

fn main() {
    env_logger::init();

    println!("Hierarchical Voxel Viewer");
    println!("=========================");
    println!("Controls:");
    println!("  WASD - Move");
    println!("  Space/Shift - Up/Down");
    println!("  Arrow Keys - Rotate (Left/Right yaw, Up/Down pitch)");
    println!("  Right Mouse - Look around");
    println!("  ESC - Quit\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
