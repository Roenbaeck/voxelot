//! Modern wgpu viewer for hierarchical voxel chunks
//!
//! Features:
//! - 6DOF camera with WASD + mouse
//! - Frustum culling
//! - Occlusion culling
//! - LOD support
//! - Instanced rendering

use crossbeam_channel::{unbounded, Receiver, Sender};
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
use sysinfo::{Pid, ProcessExt, System, SystemExt};
use voxelot::{
    cull_visible_voxels_parallel, generate_chunk_mesh, Camera, Chunk, ChunkMesh, Palette,
    RenderConfig, Voxel, VoxelInstance, World, WorldPos,
};

macro_rules! viewer_debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "viewer-debug") {
            eprintln!($($arg)*);
        }
    };
}

const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;
const CONFIG_FILE: &str = "config.toml"; // Unified TOML configuration only
const GPU_CULL_WORKGROUP_SIZE: u32 = 64;
const DEFAULT_MESH_CACHE_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
const SHADOW_MAP_SIZE: u32 = 4096;
const SHADOW_FRUSTUM_EXTENT_MIN: f32 = 150.0;
const SHADOW_FRUSTUM_EXTENT_MAX: f32 = 600.0;
const SHADOW_DISTANCE_MULTIPLIER: f32 = 2.5;
const SHADOW_BIAS: f32 = 0.001;
const SHADOW_STRENGTH_MULTIPLIER: f32 = 1.75;
const NEIGHBOR_OFFSETS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Voxel instance data for GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VoxelInstanceRaw {
    position: [f32; 3],
    voxel_type: u32,
    scale: f32,             // Scale factor (1.0 = 1x1x1, 16.0 = 16x16x16 chunk)
    custom_color: [f32; 4], // RGBA custom color (if custom_color.a > 0, use this instead of voxel_type)
    emissive: [f32; 4],
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
    emissive: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// GPU buffers and bookkeeping for a meshed chunk
struct MeshCacheEntry {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    vertex_bytes: u64,
    index_bytes: u64,
    last_used_frame: u64,
}

impl MeshCacheEntry {
    fn total_bytes(&self) -> u64 {
        self.vertex_bytes + self.index_bytes
    }
}

#[derive(Clone, Debug)]
struct ChunkEmitterWorld {
    position: [f32; 3],
    color: [f32; 3],
    intensity: f32,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct ActiveLight {
    position: [f32; 3],
    color: [f32; 3],
    intensity: f32,
}

#[derive(Debug)]
struct MeshJob {
    key: (i64, i64, i64),
    chunk: Chunk,
}

#[derive(Debug)]
struct MeshResult {
    key: (i64, i64, i64),
    mesh: ChunkMesh,
    voxel_count: u32,
}

/// Light probe for emissive indirect lighting
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LightProbe {
    position: [f32; 3],
    _pad0: f32,
    color_power: [f32; 4], // RGB from emissive_sum, A = emissive_power
}

/// Uniforms for shader (matches shader layout exactly)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],                  // 64 bytes
    sun_view_proj: [[f32; 4]; 4],        // 64 bytes
    camera_shadow_strength: [f32; 4],    // xyz = camera position, w = shadow strength
    sun_direction_shadow_bias: [f32; 4], // xyz = sun dir, w = shadow bias
    fog_time_pad: [f32; 4],              // x = fog density, y = time of day
    sun_color_pad: [f32; 4],             // xyz = sun color
    ambient_color_pad: [f32; 4],         // xyz = ambient color
    shadow_texel_size_pad: [f32; 4],     // xy = 1 / shadow map size
    moon_direction_intensity: [f32; 4],  // xyz = moon dir, w = intensity scalar
    moon_color_pad: [f32; 4],            // xyz = moon color
    light_probe_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

    fn new(position: [f32; 3], render_cfg: &voxelot::config::RenderingConfig) -> Self {
        let rc = RenderConfig::from_rendering(render_cfg);
        println!("Loaded rendering config (TOML):");
        println!("  LOD subdivide distance: {}", rc.lod_subdivide_distance);
        println!("  LOD merge distance: {}", rc.lod_merge_distance);
        println!("  Far plane: {}", rc.far_plane);
        println!("  FOV: {}°", rc.fov_degrees);

        let mut this = Self {
            camera: Camera::with_config(position, [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], rc),
            base_speed: 10.0,
            speed_multiplier: render_cfg.camera_speed_multiplier.max(0.01),
            sensitivity: 0.002,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.3,
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
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    shadow_mesh_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    main_bind_group_layout: Option<wgpu::BindGroupLayout>,
    shadow_bind_group_layout: Option<wgpu::BindGroupLayout>,
    cube_vertex_buffer: Option<wgpu::Buffer>,
    instance_buffer: Option<wgpu::Buffer>,
    shadow_texture: Option<wgpu::Texture>,
    shadow_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
    gpu_input_buffer: Option<wgpu::Buffer>,
    gpu_input_capacity: usize,
    gpu_readback_buffer: Option<wgpu::Buffer>,
    gpu_readback_capacity: usize,
    // Mesh cache: per-leaf-chunk mesh GPU buffers and metadata
    mesh_cache: HashMap<(i64, i64, i64), MeshCacheEntry>,
    mesh_cache_bytes: u64,
    fallback_chunk_instances: HashMap<(i64, i64, i64), Vec<VoxelInstanceRaw>>,
    chunk_emitters: HashMap<(i64, i64, i64), Vec<ChunkEmitterWorld>>,
    active_emitters: Vec<ActiveLight>,
    system_info: System,
    process_pid: Pid,
    instance_capacity: usize,
    cull_pipeline: Option<wgpu::ComputePipeline>,
    cull_bind_group_layout: Option<wgpu::BindGroupLayout>,
    cull_bind_group: Option<wgpu::BindGroup>,
    cull_params_buffer: Option<wgpu::Buffer>,

    world: World,
    palette: Palette,
    camera_controller: CameraController,
    pending_chunk_meshes: VecDeque<(i64, i64, i64)>,
    pending_chunk_set: HashSet<(i64, i64, i64)>,
    mesh_job_tx: Sender<MeshJob>,
    mesh_result_rx: Receiver<MeshResult>,
    mesh_worker_count: usize,
    mesh_jobs_in_flight: usize,
    ready_chunk_meshes: VecDeque<MeshResult>,
    mesh_upload_limit: usize,
    mesh_upload_baseline: usize,
    mesh_upload_max: usize,
    mesh_upload_adjust_timer: f32,
    last_frame: Instant,
    frame_count: u64,
    frame_index: u64,
    last_fps_print: Instant,

    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,

    // Lighting state
    time_of_day: f32,
    time_paused: bool,
    fog_density: f32,
    light_probe_buffer: Option<wgpu::Buffer>,
    light_probe_capacity: usize,

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

    // New CoC/DoF combine resources
    dof_coc_texture: Option<wgpu::Texture>,
    dof_coc_view: Option<wgpu::TextureView>,
    dof_coc_pipeline: Option<wgpu::RenderPipeline>,
    dof_coc_bind_group_layout: Option<wgpu::BindGroupLayout>,
    dof_coc_bind_group: Option<wgpu::BindGroup>,

    // DoF color buffer that stores blurred result before combine
    dof_color_texture: Option<wgpu::Texture>,
    dof_color_view: Option<wgpu::TextureView>,

    dof_combine_pipeline: Option<wgpu::RenderPipeline>,
    dof_combine_bind_group_layout: Option<wgpu::BindGroupLayout>,
    dof_combine_bind_group: Option<wgpu::BindGroup>,
    dof_settings: DoFSettings,
    dof_enabled: bool,
    bloom_settings: BloomSettings,
    bloom_enabled: bool,
    shadow_map_size: u32,
}

impl App {
    fn new() -> Self {
        let mut system_info = System::new();
        let process_pid = Pid::from(std::process::id() as usize);
        system_info.refresh_process(process_pid);

        // Load configuration
        let app_config = voxelot::Config::load_or_default(CONFIG_FILE);

        let mut initial_camera;
        let mut world;

        if cfg!(feature = "test-block-world") {
            // Create test world (depth 3 = 4,096 units)
            world = World::new(3);
            initial_camera = [50.0, 15.0, 65.0];
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
            initial_camera = app_config.world.camera_position;
            
            println!("Loading voxel data from {}...", app_config.world.file);
                // Load octree format from configured path — use auto-detecting loader
                match voxelot::load_world_file(std::path::Path::new(&app_config.world.file)) {
                    Ok(loaded_world) => {
                        world = loaded_world;
                        println!("Loaded world from {} (depth {})", 
                                 app_config.world.file, world.hierarchy_depth());
                    }
                    Err(e) => {
                        println!("Failed to load {}: {}", app_config.world.file, e);
                        println!("Using fallback world generation.");

                        world = World::new(3);
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

        println!("Loading palette from {}...", app_config.world.palette);
        let palette = Palette::load(&app_config.world.palette);
        let (mesh_job_tx, mesh_job_rx) = unbounded::<MeshJob>();
        let (mesh_result_tx, mesh_result_rx) = unbounded::<MeshResult>();

        let available_workers = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(2))
            .unwrap_or(1)
            .max(1);
        let mesh_worker_count = available_workers.min(6);

        for worker_index in 0..mesh_worker_count {
            let job_rx = mesh_job_rx.clone();
            let result_tx = mesh_result_tx.clone();
            let palette_clone = palette.clone();
            std::thread::Builder::new()
                .name(format!("mesh-worker-{}", worker_index))
                .spawn(move || {
                    for job in job_rx.iter() {
                        let MeshJob { key, chunk } = job;
                        // Skip meshing completely empty chunks early.
                        if chunk.voxel_count == 0 {
                            continue;
                        }
                        let mesh = generate_chunk_mesh(&chunk, &palette_clone);
                        if result_tx
                            .send(MeshResult {
                                key,
                                mesh,
                                voxel_count: chunk.voxel_count,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                })
                .expect("failed to spawn mesh worker");
        }

        drop(mesh_result_tx);
        drop(mesh_job_rx);

        let mesh_upload_baseline = 4usize;
        let mesh_upload_max = (mesh_worker_count * 4).max(mesh_upload_baseline * 2);

        println!("Updating LOD metadata...");
        world.update_all_lod_metadata(&palette);
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
            shadow_pipeline: None,
            shadow_mesh_pipeline: None,
            uniform_buffer: None,
            bind_group: None,
            shadow_bind_group: None,
            main_bind_group_layout: None,
            shadow_bind_group_layout: None,
            cube_vertex_buffer: None,
            instance_buffer: None,
            shadow_texture: None,
            shadow_view: None,
            shadow_sampler: None,
            gpu_input_buffer: None,
            gpu_input_capacity: 0,
            gpu_readback_buffer: None,
            gpu_readback_capacity: 0,
            mesh_cache: HashMap::new(),
            mesh_cache_bytes: 0,
            fallback_chunk_instances: HashMap::new(),
            chunk_emitters: HashMap::new(),
            active_emitters: Vec::new(),
            system_info,
            process_pid,
            instance_capacity: 0,
            cull_pipeline: None,
            cull_bind_group_layout: None,
            cull_bind_group: None,
            cull_params_buffer: None,
            world,
            palette,
            mesh_job_tx,
            mesh_result_rx,
            mesh_worker_count,
            mesh_jobs_in_flight: 0,
            ready_chunk_meshes: VecDeque::new(),
            mesh_upload_limit: mesh_upload_baseline,
            mesh_upload_baseline,
            mesh_upload_max,
            mesh_upload_adjust_timer: 0.0,
            // Load unified TOML config once and feed rendering section into camera controller
            camera_controller: {
                let full_cfg = voxelot::Config::load_or_default(CONFIG_FILE);
                CameraController::new(initial_camera, &full_cfg.rendering)
            },
            pending_chunk_meshes: VecDeque::new(),
            pending_chunk_set: HashSet::new(),
            last_frame: Instant::now(),
            frame_count: 0,
            frame_index: 0,
            last_fps_print: Instant::now(),
            mouse_pressed: false,
            last_mouse_pos: None,
            time_of_day: 0.5,    // Start at noon
            time_paused: false,  // Time cycle starts running
            fog_density: 0.003000, // Default fog density
            light_probe_buffer: None,
            light_probe_capacity: 0,
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
            dof_coc_texture: None,
            dof_coc_view: None,
            dof_coc_pipeline: None,
            dof_coc_bind_group_layout: None,
            dof_coc_bind_group: None,
            dof_color_texture: None,
            dof_color_view: None,
            dof_combine_pipeline: None,
            dof_combine_bind_group_layout: None,
            dof_combine_bind_group: None,
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
            shadow_map_size: SHADOW_MAP_SIZE,
        }
    }

    fn save_config(&self) {
        // Persist full TOML config using unified Config (camera speed multiplier retained)
        // We read existing file, update rendering subsection relevant fields, then save.
        if let Ok(mut full_cfg) = voxelot::Config::load(CONFIG_FILE) {
            full_cfg.rendering.lod_subdivide_distance = self.camera_controller.camera.config.lod_subdivide_distance;
            full_cfg.rendering.lod_merge_distance = self.camera_controller.camera.config.lod_merge_distance;
            full_cfg.rendering.chunk_lod_distance = self.camera_controller.camera.config.lod_render_distance;
            full_cfg.rendering.fov_degrees = self.camera_controller.camera.config.fov_degrees;
            full_cfg.rendering.near_plane = self.camera_controller.camera.config.near_plane;
            full_cfg.rendering.far_plane = self.camera_controller.camera.config.far_plane;
            full_cfg.rendering.camera_speed_multiplier = self.camera_controller.speed_multiplier;
            if let Err(e) = full_cfg.save(CONFIG_FILE) {
                eprintln!("Failed to save unified config: {}", e);
            } else {
                println!("Saved unified TOML config to {}", CONFIG_FILE);
            }
        } else {
            eprintln!("Warning: could not load existing TOML config for update; creating default.");
            let mut full_cfg = voxelot::Config::default();
            full_cfg.rendering.lod_subdivide_distance = self.camera_controller.camera.config.lod_subdivide_distance;
            if let Err(e) = full_cfg.save(CONFIG_FILE) {
                eprintln!("Failed to write default unified config: {}", e);
            }
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
                // Toggle time pause
                self.time_paused = !self.time_paused;
                let phase = if self.time_of_day < 0.125 {
                    "Midnight→Dawn"
                } else if self.time_of_day < 0.25 {
                    "Dawn→Sunrise"
                } else if self.time_of_day < 0.5 {
                    "Sunrise→Noon"
                } else if self.time_of_day < 0.75 {
                    "Noon→Sunset"
                } else if self.time_of_day < 0.875 {
                    "Sunset→Dusk"
                } else {
                    "Dusk→Midnight"
                };
                println!(
                    "Time {} at {:.3} ({})",
                    if self.time_paused { "paused" } else { "resumed" },
                    self.time_of_day,
                    phase
                );
            }
            KeyCode::KeyF => {
                // Decrease fog density (smaller step)
                self.fog_density = (self.fog_density - 0.00005).max(0.0);
                println!("Fog density: {:.6}", self.fog_density);
            }
            KeyCode::KeyG => {
                // Increase fog density (smaller step)
                self.fog_density = (self.fog_density + 0.00005).min(0.01);
                println!("Fog density: {:.6}", self.fog_density);
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

        // Intermediate color buffer that stores blurred DoF result before final combine
        let dof_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DoF Color Texture"),
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
        let dof_color_view =
            dof_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        // CoC texture at full resolution, single-channel float
        let coc_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DoF CoC Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let coc_view = coc_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.offscreen_color_view = Some(color_view);
        self.offscreen_color_texture = Some(color_texture);
        self.offscreen_depth_view = Some(depth_view);
        self.offscreen_depth_texture = Some(depth_texture);
        self.post_color_view = Some(post_color_view);
        self.post_color_texture = Some(post_color_texture);
        self.dof_color_texture = Some(dof_color_texture);
        self.dof_color_view = Some(dof_color_view);
        self.dof_coc_texture = Some(coc_texture);
        self.dof_coc_view = Some(coc_view);
        self.dof_smooth_ping_view = Some(dof_smooth_ping_view);
        self.dof_smooth_ping_texture = Some(dof_smooth_ping_texture);
        self.bloom_ping_view = Some(bloom_ping_view);
        self.bloom_ping_texture = Some(bloom_ping_texture);
        self.bloom_pong_view = Some(bloom_pong_view);
        self.bloom_pong_texture = Some(bloom_pong_texture);

        self.update_dof_bind_group();
        self.update_dof_smooth_bind_groups();
    // Combine bind group depends on DoF color and CoC buffers
    self.update_dof_combine_bind_group();
        self.update_bloom_uniforms();
        self.update_bloom_bind_groups();
    }

    fn update_dof_coc_bind_group(&mut self) {
        let (Some(device), Some(layout), Some(depth_view), Some(sampler), Some(ubo)) = (
            self.device.as_ref(),
            self.dof_coc_bind_group_layout.as_ref(),
            self.offscreen_depth_view.as_ref(),
            self.post_sampler.as_ref(),
            self.dof_uniform_buffer.as_ref(),
        ) else {
            return;
        };

        self.dof_coc_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DoF CoC Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        }));
    }

    fn recreate_shadow_map(&mut self) {
        let Some(device) = self.device.as_ref() else {
            return;
        };
        if self.shadow_sampler.is_none() {
            return;
        }

        let extent = wgpu::Extent3d {
            width: self.shadow_map_size,
            height: self.shadow_map_size,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.shadow_texture = Some(texture);
        self.shadow_view = Some(view);
        self.update_main_bind_group();
    }

    fn update_shadow_bind_group(&mut self) {
        let (Some(device), Some(layout), Some(uniform_buffer)) = (
            self.device.as_ref(),
            self.shadow_bind_group_layout.as_ref(),
            self.uniform_buffer.as_ref(),
        ) else {
            return;
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Uniform Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        self.shadow_bind_group = Some(bind_group);
    }

    fn update_main_bind_group(&mut self) {
        let (
            Some(device),
            Some(layout),
            Some(uniform_buffer),
            Some(shadow_view),
            Some(shadow_sampler),
            Some(light_probe_buffer),
        ) = (
            self.device.as_ref(),
            self.main_bind_group_layout.as_ref(),
            self.uniform_buffer.as_ref(),
            self.shadow_view.as_ref(),
            self.shadow_sampler.as_ref(),
            self.light_probe_buffer.as_ref(),
        )
        else {
            return;
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main Uniform Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: light_probe_buffer.as_entire_binding(),
                },
            ],
        });

        self.bind_group = Some(bind_group);
    }

    fn update_dof_bind_group(&mut self) {
        let (
            Some(device),
            Some(layout),
            Some(color_view),
            Some(coc_view),
            Some(sampler),
            Some(ubo),
        ) = (
            self.device.as_ref(),
            self.dof_bind_group_layout.as_ref(),
            self.offscreen_color_view.as_ref(),
            self.dof_coc_view.as_ref(),
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
                    resource: wgpu::BindingResource::TextureView(coc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        }));
    }

    fn update_dof_combine_bind_group(&mut self) {
        let (
            Some(device),
            Some(layout),
            Some(dof_color_view),
            Some(coc_view),
            Some(source_view),
            Some(sampler),
        ) = (
            self.device.as_ref(),
            self.dof_combine_bind_group_layout.as_ref(),
            self.dof_color_view.as_ref(),
            self.dof_coc_view.as_ref(),
            self.offscreen_color_view.as_ref(),
            self.post_sampler.as_ref(),
        )
        else {
            return;
        };

        self.dof_combine_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DoF Combine Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(dof_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(coc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(source_view),
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

    fn mesh_cache_byte_budget(&self) -> u64 {
        DEFAULT_MESH_CACHE_BUDGET_BYTES
    }

    fn max_inflight_jobs(&self) -> usize {
        let workers = self.mesh_worker_count.max(1);
        workers * 2 + self.mesh_upload_limit
    }

    fn adjust_mesh_upload_budget(&mut self, dt: f32, fps: f32) {
        const TARGET_FPS: f32 = 60.0;
        const LOWER_MULTIPLIER: f32 = 0.75;
        const UPPER_MULTIPLIER: f32 = 0.95;

        if !fps.is_finite() {
            return;
        }

        self.mesh_upload_adjust_timer += dt;
        if self.mesh_upload_adjust_timer < 0.5 {
            return;
        }
        self.mesh_upload_adjust_timer = 0.0;

        let mut new_limit = self.mesh_upload_limit;
        if fps < TARGET_FPS * LOWER_MULTIPLIER {
            new_limit = (self.mesh_upload_limit + 1).min(self.mesh_upload_max);
        } else if fps > TARGET_FPS * UPPER_MULTIPLIER {
            if self.mesh_upload_limit > self.mesh_upload_baseline {
                new_limit = self.mesh_upload_limit - 1;
            }
        } else if self.mesh_upload_limit > self.mesh_upload_baseline {
            new_limit = self.mesh_upload_limit - 1;
        } else if self.mesh_upload_limit < self.mesh_upload_baseline {
            new_limit = self.mesh_upload_limit + 1;
        }

        if new_limit != self.mesh_upload_limit {
            viewer_debug!(
                "Mesh upload limit adjusted: {} -> {} (fps {:.1})",
                self.mesh_upload_limit,
                new_limit,
                fps
            );
            self.mesh_upload_limit = new_limit;
        }
    }

    fn evict_mesh_cache(&mut self) {
        let budget = self.mesh_cache_byte_budget();
        if self.mesh_cache_bytes <= budget {
            return;
        }

        let mut entries: Vec<_> = self
            .mesh_cache
            .iter()
            .map(|(key, entry)| (*key, entry.last_used_frame))
            .collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1));

        let mut freed_bytes = 0u64;
        let mut evicted = 0usize;

        for (key, _) in entries {
            if self.mesh_cache_bytes <= budget {
                break;
            }

            if let Some(entry) = self.mesh_cache.remove(&key) {
                let entry_bytes = entry.total_bytes();
                entry.vertex_buffer.destroy();
                entry.index_buffer.destroy();
                self.mesh_cache_bytes = self.mesh_cache_bytes.saturating_sub(entry_bytes);
                self.chunk_emitters.remove(&key);
                freed_bytes += entry_bytes;
                evicted += 1;
            }
        }

        if cfg!(feature = "viewer-debug") && evicted > 0 {
            viewer_debug!(
                "Mesh cache eviction: freed {:.2} MiB across {} entries (budget {:.2} MiB, now {:.2} MiB)",
                freed_bytes as f64 / (1024.0 * 1024.0),
                evicted,
                budget as f64 / (1024.0 * 1024.0),
                self.mesh_cache_bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }

    fn voxel_to_raw(v: &VoxelInstance, palette: &Palette) -> VoxelInstanceRaw {
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

        let (emissive_rgb, emissive_intensity) = if v.custom_color.is_some() {
            ([0.0, 0.0, 0.0], 0.0)
        } else {
            palette.emissive(v.voxel_type as u32)
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
            emissive: [
                emissive_rgb[0],
                emissive_rgb[1],
                emissive_rgb[2],
                emissive_intensity,
            ],
        }
    }

    fn fallback_instances_for_chunk(
        &mut self,
        key: (i64, i64, i64),
    ) -> Option<&[VoxelInstanceRaw]> {
        if !self.fallback_chunk_instances.contains_key(&key) {
            let chunk = self
                .world
                .get_leaf_chunk_at_origin(WorldPos::new(key.0, key.1, key.2))?;
            let mut instances = Vec::with_capacity(chunk.voxel_count as usize);
            Self::collect_chunk_surface_voxels(chunk, key, &self.palette, &mut instances);
            self.fallback_chunk_instances.insert(key, instances);
        }
        self.fallback_chunk_instances
            .get(&key)
            .map(|instances| instances.as_slice())
    }

    fn collect_chunk_surface_voxels(
        chunk: &Chunk,
        origin: (i64, i64, i64),
        palette: &Palette,
        out: &mut Vec<VoxelInstanceRaw>,
    ) {
        for ((lx, ly, lz), voxel) in chunk.iter() {
            let world_pos = (
                origin.0 + lx as i64,
                origin.1 + ly as i64,
                origin.2 + lz as i64,
            );

            match voxel {
                Voxel::Solid(voxel_type) => {
                    if Self::voxel_has_exposed_face(chunk, lx, ly, lz) {
                        let material = palette.material(*voxel_type as u32);
                        out.push(VoxelInstanceRaw {
                            position: [world_pos.0 as f32, world_pos.1 as f32, world_pos.2 as f32],
                            voxel_type: *voxel_type as u32,
                            scale: 1.0,
                            custom_color: material.albedo,
                            emissive: [
                                material.emissive[0],
                                material.emissive[1],
                                material.emissive[2],
                                material.emissive_intensity,
                            ],
                        });
                    }
                }
                Voxel::Chunk(sub_chunk) => {
                    Self::collect_chunk_surface_voxels(sub_chunk, world_pos, palette, out);
                }
            }
        }
    }

    fn voxel_has_exposed_face(chunk: &Chunk, x: u8, y: u8, z: u8) -> bool {
        for (dx, dy, dz) in NEIGHBOR_OFFSETS {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;

            if nx < 0 || nx >= 16 || ny < 0 || ny >= 16 || nz < 0 || nz >= 16 {
                return true;
            }

            let nx_u = nx as u8;
            let ny_u = ny as u8;
            let nz_u = nz as u8;
            if !chunk.contains(nx_u, ny_u, nz_u) {
                return true;
            }
        }

        false
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

        // Create bind group layouts
        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let main_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Main Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create pipeline layouts
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&main_bind_group_layout],
            push_constant_ranges: &[],
        });

        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Pipeline Layout"),
                bind_group_layouts: &[&shadow_bind_group_layout],
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
                            3 => Float32x4,  // custom_color (RGBA)
                            6 => Float32x4   // emissive (RGB + intensity)
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
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertexRaw>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32x4,
                        3 => Float32x4
                    ],
                }],
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

        // Create shadow pipelines (depth-only)
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow_instanced"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<CubeVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![4 => Float32x3, 5 => Float32x3],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<VoxelInstanceRaw>()
                            as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3,
                            1 => Uint32,
                            2 => Float32,
                            3 => Float32x4,
                            6 => Float32x4
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let shadow_mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Mesh Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow_mesh"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertexRaw>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32x4,
                        3 => Float32x4
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
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

        let dof_coc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF CoC Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dof_coc.wgsl").into()),
        });

        let dof_combine_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Combine Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dof_combine.wgsl").into()),
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
                        // CoC is stored in an R32Float texture which is not filterable on many backends.
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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

        let dof_coc_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DoF CoC Bind Group Layout"),
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
                            sample_type: wgpu::TextureSampleType::Depth,
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

        let dof_combine_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DoF Combine Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            // CoC texture is R32Float (non-filterable on many backends).
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                        // CoC texture at binding 1 is non-filterable; use a non-filtering sampler
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let dof_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF Pipeline Layout"),
            bind_group_layouts: &[&dof_bind_group_layout],
            push_constant_ranges: &[],
        });

        let dof_coc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF CoC Pipeline Layout"),
            bind_group_layouts: &[&dof_coc_bind_group_layout],
            push_constant_ranges: &[],
        });

        let dof_combine_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF Combine Pipeline Layout"),
            bind_group_layouts: &[&dof_combine_bind_group_layout],
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
                    // Write blurred DoF into intermediate color buffer
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

        let dof_coc_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF CoC Pipeline"),
            layout: Some(&dof_coc_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &dof_coc_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &dof_coc_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let dof_combine_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Combine Pipeline"),
            layout: Some(&dof_combine_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &dof_combine_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &dof_combine_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
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
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
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
        let shadow_texel = 1.0 / self.shadow_map_size as f32;
        let uniforms = Uniforms {
            mvp: [[0.0; 4]; 4],
            sun_view_proj: [[0.0; 4]; 4],
            camera_shadow_strength: [0.0, 0.0, 0.0, 1.0],
            sun_direction_shadow_bias: [0.5, 1.0, 0.3, SHADOW_BIAS],
            fog_time_pad: [0.003000, 0.5, 0.0, 0.0],
            sun_color_pad: [1.0, 0.95, 0.8, 0.0],
            ambient_color_pad: [0.3, 0.35, 0.45, 0.0],
            shadow_texel_size_pad: [shadow_texel, shadow_texel, 0.0, 0.0],
            moon_direction_intensity: [ -0.5, -1.0, -0.3, 0.2], // initial opposite dim moon
            moon_color_pad: [0.2, 0.25, 0.35, 0.0],
            light_probe_count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create light probe buffer (start with capacity for 64 probes)
        let light_probe_capacity = 64;
        let empty_probes = vec![
            LightProbe {
                position: [0.0; 3],
                _pad0: 0.0,
                color_power: [0.0; 4],
            };
            light_probe_capacity
        ];
        let light_probe_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Probe Buffer"),
            contents: bytemuck::cast_slice(&empty_probes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
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
        self.dof_coc_pipeline = Some(dof_coc_pipeline);
        self.dof_coc_bind_group_layout = Some(dof_coc_bind_group_layout);
        self.dof_combine_pipeline = Some(dof_combine_pipeline);
        self.dof_combine_bind_group_layout = Some(dof_combine_bind_group_layout);
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
        self.shadow_pipeline = Some(shadow_pipeline);
        self.shadow_mesh_pipeline = Some(shadow_mesh_pipeline);
        self.uniform_buffer = Some(uniform_buffer);
        self.light_probe_buffer = Some(light_probe_buffer);
        self.light_probe_capacity = light_probe_capacity;
        self.main_bind_group_layout = Some(main_bind_group_layout);
        self.shadow_bind_group_layout = Some(shadow_bind_group_layout);
        self.shadow_sampler = Some(shadow_sampler);
        self.cube_vertex_buffer = Some(cube_vertex_buffer);

        self.update_shadow_bind_group();
        self.recreate_shadow_map();

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
        self.frame_index = self.frame_index.wrapping_add(1);

        let fps = if dt > 0.0 { 1.0 / dt } else { f32::INFINITY };
        self.adjust_mesh_upload_budget(dt, fps);

        // Auto-advance time of day: full cycle in 120 seconds (60s sun, 60s moon)
        if !self.time_paused {
            self.time_of_day = (self.time_of_day + dt / 120.0) % 1.0;
        }

        self.camera_controller.update(dt);

        // Gather candidate voxels for GPU culling using CPU hierarchy traversal
        let cull_start = Instant::now();
        let visible = cull_visible_voxels_parallel(&self.world, &self.camera_controller.camera);
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

        while let Ok(result) = self.mesh_result_rx.try_recv() {
            self.ready_chunk_meshes.push_back(result);
            if self.mesh_jobs_in_flight > 0 {
                self.mesh_jobs_in_flight -= 1;
            }
        }

        let max_inflight = self.max_inflight_jobs();
        while self.mesh_jobs_in_flight < max_inflight {
            let Some(key) = self.pending_chunk_meshes.pop_front() else {
                break;
            };

            match self
                .world
                .get_leaf_chunk_at_origin(WorldPos::new(key.0, key.1, key.2))
            {
                Some(chunk) => {
                    if self
                        .mesh_job_tx
                        .send(MeshJob {
                            key,
                            chunk: chunk.clone(),
                        })
                        .is_ok()
                    {
                        self.mesh_jobs_in_flight += 1;
                    } else {
                        self.pending_chunk_meshes.push_front(key);
                        self.pending_chunk_set.remove(&key);
                        break;
                    }
                }
                None => {
                    self.pending_chunk_set.remove(&key);
                    self.chunk_emitters.remove(&key);
                    chunks_not_found += 1;
                }
            }
        }

        let mut processed_meshes = 0;
        while processed_meshes < self.mesh_upload_limit {
            let Some(result) = self.ready_chunk_meshes.pop_front() else {
                break;
            };

            let MeshResult {
                key,
                mesh,
                voxel_count,
            } = result;
            self.pending_chunk_set.remove(&key);
            processed_meshes += 1;

            if mesh.emitters.is_empty() {
                self.chunk_emitters.remove(&key);
            } else {
                let world_emitters: Vec<ChunkEmitterWorld> = mesh
                    .emitters
                    .iter()
                    .map(|emitter| ChunkEmitterWorld {
                        position: [
                            key.0 as f32 + emitter.position[0],
                            key.1 as f32 + emitter.position[1],
                            key.2 as f32 + emitter.position[2],
                        ],
                        color: emitter.color,
                        intensity: emitter.intensity,
                    })
                    .collect();
                if world_emitters.is_empty() {
                    self.chunk_emitters.remove(&key);
                } else {
                    self.chunk_emitters.insert(key, world_emitters);
                }
            }

            if mesh.indices.is_empty() {
                continue;
            }

            if new_meshes_created == 0 {
                viewer_debug!(
                    "DEBUG first mesh at ({},{},{}): {} voxels in chunk, {} vertices, {} triangles",
                    key.0,
                    key.1,
                    key.2,
                    voxel_count,
                    mesh.vertices.len(),
                    mesh.indices.len() / 3
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
                    emissive: v.emissive,
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
            let vertex_bytes = (vb_data.len() * std::mem::size_of::<MeshVertexRaw>()) as u64;
            let index_bytes = (mesh.indices.len() * std::mem::size_of::<u32>()) as u64;
            viewer_debug!(
                "Created mesh for chunk ({},{},{}): {} vertices, {} triangles",
                key.0,
                key.1,
                key.2,
                mesh.vertices.len(),
                mesh.indices.len() / 3
            );
            let entry = MeshCacheEntry {
                vertex_buffer: vbuf,
                index_buffer: ibuf,
                index_count: mesh.indices.len() as u32,
                vertex_bytes,
                index_bytes,
                last_used_frame: self.frame_index,
            };
            self.mesh_cache.insert(key, entry);
            self.mesh_cache_bytes += vertex_bytes + index_bytes;
            self.fallback_chunk_instances.remove(&key);
            new_meshes_created += 1;

            if leaf_chunks.contains(&key) {
                cpu_mesh_keys.insert(key);
                missing_chunks.remove(&key);
            }
        }
        let mesh_time = mesh_start.elapsed();

        if self.mesh_cache_bytes > self.mesh_cache_byte_budget() {
            self.evict_mesh_cache();
        }

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

                    if let Some(fallback) = self.fallback_instances_for_chunk(key) {
                        out.extend_from_slice(fallback);
                    } else {
                        out.push(Self::voxel_to_raw(v, &self.palette));
                    }
                    continue;
                }

                let (emissive_rgb, emissive_intensity) = if input.custom_color[3] > 0.0 {
                    ([0.0, 0.0, 0.0], 0.0)
                } else {
                    self.palette.emissive(input.voxel_type)
                };

                out.push(VoxelInstanceRaw {
                    position: input.position,
                    voxel_type: input.voxel_type,
                    scale: input.scale,
                    custom_color: input.custom_color,
                    emissive: [
                        emissive_rgb[0],
                        emissive_rgb[1],
                        emissive_rgb[2],
                        emissive_intensity,
                    ],
                });
            }

            (out, mesh_selection)
        } else {
            let mut out: Vec<VoxelInstanceRaw> = Vec::new();
            for v in &visible {
                if v.is_leaf_chunk {
                    let key = (v.position[0], v.position[1], v.position[2]);
                    if cpu_mesh_keys.contains(&key) {
                        continue;
                    }

                    if let Some(fallback) = self.fallback_instances_for_chunk(key) {
                        out.extend_from_slice(fallback);
                    } else {
                        out.push(Self::voxel_to_raw(v, &self.palette));
                    }
                } else {
                    out.push(Self::voxel_to_raw(v, &self.palette));
                }
            }

            (out, cpu_mesh_keys.clone())
        };
        let instance_time = instance_start.elapsed();

        self.active_emitters.clear();
        for key in &draw_mesh_keys {
            if let Some(emitters) = self.chunk_emitters.get(key) {
                self.active_emitters
                    .extend(emitters.iter().map(|emitter| ActiveLight {
                        position: emitter.position,
                        color: emitter.color,
                        intensity: emitter.intensity,
                    }));
            }
        }

        if cfg!(feature = "viewer-debug") && self.frame_count % 60 == 0 {
            let total_emitters: usize = self.chunk_emitters.values().map(|list| list.len()).sum();
            viewer_debug!(
                "Mesh stats: {} cached meshes, {} new this frame, {} potential chunks, pending {}, ready {}, inflight {}, upload_limit {}, fallback instances {}, emitters {} (active {})",
                self.mesh_cache.len(),
                new_meshes_created,
                leaf_chunks.len(),
                self.pending_chunk_meshes.len(),
                self.ready_chunk_meshes.len(),
                self.mesh_jobs_in_flight,
                self.mesh_upload_limit,
                instances.len(),
                total_emitters,
                self.active_emitters.len()
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
        // Offset the solar angle so that time_of_day ≈0.25 aligns with sunrise; trig handles wrapping
        let time_angle = (self.time_of_day - 0.25) * std::f32::consts::TAU;
        let sun_height = time_angle.sin();
        // Sun moves in an arc: horizontal component (cos) and vertical (sin)
        // Use full range for horizontal to get proper shadow directions
        let sun_direction = [time_angle.cos(), sun_height, 0.2];

        // Smooth color transitions based on time of day
        let (sun_color, ambient_color) = {
            // Define key times and colors
            // Midnight is at 0.0, darkest point
            let midnight_moon = [0.15, 0.18, 0.25];
            let midnight_ambient = [0.02, 0.025, 0.05];
            // Dusk/dawn has some light
            let twilight_moon = [0.35, 0.35, 0.5];
            let twilight_ambient = [0.08, 0.08, 0.15];
            let sunrise_sun = [1.0, 0.6, 0.3];
            let sunrise_ambient = [0.3, 0.2, 0.2];
            let day_sun = [1.0, 0.95, 0.8];
            let day_ambient = [0.3, 0.35, 0.45];
            
            // Interpolate between color phases
            let t = self.time_of_day;
            if t < 0.125 {
                // Midnight to twilight (0.0 -> 0.125)
                let factor = t / 0.125;
                let sun = [
                    midnight_moon[0] + (twilight_moon[0] - midnight_moon[0]) * factor,
                    midnight_moon[1] + (twilight_moon[1] - midnight_moon[1]) * factor,
                    midnight_moon[2] + (twilight_moon[2] - midnight_moon[2]) * factor,
                ];
                let ambient = [
                    midnight_ambient[0] + (twilight_ambient[0] - midnight_ambient[0]) * factor,
                    midnight_ambient[1] + (twilight_ambient[1] - midnight_ambient[1]) * factor,
                    midnight_ambient[2] + (twilight_ambient[2] - midnight_ambient[2]) * factor,
                ];
                (sun, ambient)
            } else if t < 0.25 {
                // Twilight to sunrise (0.125 -> 0.25)
                let factor = (t - 0.125) / 0.125;
                let sun = [
                    twilight_moon[0] + (sunrise_sun[0] - twilight_moon[0]) * factor,
                    twilight_moon[1] + (sunrise_sun[1] - twilight_moon[1]) * factor,
                    twilight_moon[2] + (sunrise_sun[2] - twilight_moon[2]) * factor,
                ];
                let ambient = [
                    twilight_ambient[0] + (sunrise_ambient[0] - twilight_ambient[0]) * factor,
                    twilight_ambient[1] + (sunrise_ambient[1] - twilight_ambient[1]) * factor,
                    twilight_ambient[2] + (sunrise_ambient[2] - twilight_ambient[2]) * factor,
                ];
                (sun, ambient)
            } else if t < 0.5 {
                // Sunrise to day (0.25 -> 0.5)
                let factor = (t - 0.25) / 0.25;
                let sun = [
                    sunrise_sun[0] + (day_sun[0] - sunrise_sun[0]) * factor,
                    sunrise_sun[1] + (day_sun[1] - sunrise_sun[1]) * factor,
                    sunrise_sun[2] + (day_sun[2] - sunrise_sun[2]) * factor,
                ];
                let ambient = [
                    sunrise_ambient[0] + (day_ambient[0] - sunrise_ambient[0]) * factor,
                    sunrise_ambient[1] + (day_ambient[1] - sunrise_ambient[1]) * factor,
                    sunrise_ambient[2] + (day_ambient[2] - sunrise_ambient[2]) * factor,
                ];
                (sun, ambient)
            } else if t < 0.75 {
                // Day to sunset (0.5 -> 0.75)
                let factor = (t - 0.5) / 0.25;
                let sun = [
                    day_sun[0] + (sunrise_sun[0] - day_sun[0]) * factor,
                    day_sun[1] + (sunrise_sun[1] - day_sun[1]) * factor,
                    day_sun[2] + (sunrise_sun[2] - day_sun[2]) * factor,
                ];
                let ambient = [
                    day_ambient[0] + (sunrise_ambient[0] - day_ambient[0]) * factor,
                    day_ambient[1] + (sunrise_ambient[1] - day_ambient[1]) * factor,
                    day_ambient[2] + (sunrise_ambient[2] - day_ambient[2]) * factor,
                ];
                (sun, ambient)
            } else if t < 0.875 {
                // Sunset to twilight (0.75 -> 0.875)
                let factor = (t - 0.75) / 0.125;
                let sun = [
                    sunrise_sun[0] + (twilight_moon[0] - sunrise_sun[0]) * factor,
                    sunrise_sun[1] + (twilight_moon[1] - sunrise_sun[1]) * factor,
                    sunrise_sun[2] + (twilight_moon[2] - sunrise_sun[2]) * factor,
                ];
                let ambient = [
                    sunrise_ambient[0] + (twilight_ambient[0] - sunrise_ambient[0]) * factor,
                    sunrise_ambient[1] + (twilight_ambient[1] - sunrise_ambient[1]) * factor,
                    sunrise_ambient[2] + (twilight_ambient[2] - sunrise_ambient[2]) * factor,
                ];
                (sun, ambient)
            } else {
                // Twilight to midnight (0.875 -> 1.0)
                let factor = (t - 0.875) / 0.125;
                let sun = [
                    twilight_moon[0] + (midnight_moon[0] - twilight_moon[0]) * factor,
                    twilight_moon[1] + (midnight_moon[1] - twilight_moon[1]) * factor,
                    twilight_moon[2] + (midnight_moon[2] - twilight_moon[2]) * factor,
                ];
                let ambient = [
                    twilight_ambient[0] + (midnight_ambient[0] - twilight_ambient[0]) * factor,
                    twilight_ambient[1] + (midnight_ambient[1] - twilight_ambient[1]) * factor,
                    twilight_ambient[2] + (midnight_ambient[2] - twilight_ambient[2]) * factor,
                ];
                (sun, ambient)
            }
        };

        let sun_direction_vec_raw = Vec3::from_array(sun_direction);
        let sun_direction_vec = if sun_direction_vec_raw.length_squared() > 0.0001 {
            sun_direction_vec_raw.normalize()
        } else {
            Vec3::Y
        };
        let camera_pos = self.camera_controller.camera.position;
        let camera_pos_vec = Vec3::from(camera_pos);

        let shadow_extent =
            (self.lod_distance * 0.35).clamp(SHADOW_FRUSTUM_EXTENT_MIN, SHADOW_FRUSTUM_EXTENT_MAX);
        let shadow_distance = shadow_extent * SHADOW_DISTANCE_MULTIPLIER;
        let mut light_up = Vec3::Y;
        if sun_direction_vec.dot(light_up).abs() > 0.9 {
            light_up = Vec3::X;
        }
        let light_target = camera_pos_vec;
        let light_position = light_target + sun_direction_vec * shadow_distance.max(1.0);
        let light_view = Mat4::look_at_rh(light_position, light_target, light_up);

        let camera_forward_vec = Vec3::from(self.camera_controller.camera.forward).normalize();
        let mut camera_up_vec = Vec3::from(self.camera_controller.camera.up);
        if camera_up_vec.length_squared() < 1e-4 {
            camera_up_vec = Vec3::Y;
        }
        camera_up_vec = camera_up_vec.normalize();
        let camera_right_vec = camera_forward_vec.cross(camera_up_vec).normalize();
        let camera_up_vec = camera_right_vec.cross(camera_forward_vec).normalize();

        let frustum_near = self.camera_controller.camera.near.max(0.1);
        let frustum_far = shadow_extent.min(self.camera_controller.camera.far);
        let tan_half_fov = (self.camera_controller.camera.fov * 0.5).tan();
        let near_height = 2.0 * tan_half_fov * frustum_near;
        let near_width = near_height * aspect;
        let far_height = 2.0 * tan_half_fov * frustum_far;
        let far_width = far_height * aspect;

        let near_center = camera_pos_vec + camera_forward_vec * frustum_near;
        let far_center = camera_pos_vec + camera_forward_vec * frustum_far;

        let near_up_vec = camera_up_vec * (near_height * 0.5);
        let near_right_vec = camera_right_vec * (near_width * 0.5);
        let far_up_vec = camera_up_vec * (far_height * 0.5);
        let far_right_vec = camera_right_vec * (far_width * 0.5);

        let frustum_corners = [
            near_center - near_right_vec + near_up_vec,
            near_center + near_right_vec + near_up_vec,
            near_center + near_right_vec - near_up_vec,
            near_center - near_right_vec - near_up_vec,
            far_center - far_right_vec + far_up_vec,
            far_center + far_right_vec + far_up_vec,
            far_center + far_right_vec - far_up_vec,
            far_center - far_right_vec - far_up_vec,
        ];

        let mut bounds_min = Vec3::splat(f32::INFINITY);
        let mut bounds_max = Vec3::splat(f32::NEG_INFINITY);
        for corner in frustum_corners.iter() {
            let corner_ls = (light_view * corner.extend(1.0)).truncate();
            bounds_min = bounds_min.min(corner_ls);
            bounds_max = bounds_max.max(corner_ls);
        }

        let xy_padding = 15.0;
        bounds_min.x -= xy_padding;
        bounds_min.y -= xy_padding;
        bounds_max.x += xy_padding;
        bounds_max.y += xy_padding;

        let z_padding = 25.0;
        let mut near_plane = (-bounds_max.z).max(0.1);
        let mut far_plane = (-bounds_min.z).max(near_plane + 10.0);
        near_plane = (near_plane - z_padding).max(0.1);
        far_plane += z_padding;
        if far_plane <= near_plane + 1.0 {
            far_plane = near_plane + 1.0;
        }

        // Stabilize shadow map by snapping to texel-aligned grid
        // This prevents sub-pixel jitter when camera moves
        let width = bounds_max.x - bounds_min.x;
        let height = bounds_max.y - bounds_min.y;
        let texel_size_x = width / self.shadow_map_size as f32;
        let texel_size_y = height / self.shadow_map_size as f32;
        
        bounds_min.x = (bounds_min.x / texel_size_x).floor() * texel_size_x;
        bounds_min.y = (bounds_min.y / texel_size_y).floor() * texel_size_y;
        bounds_max.x = (bounds_max.x / texel_size_x).ceil() * texel_size_x;
        bounds_max.y = (bounds_max.y / texel_size_y).ceil() * texel_size_y;

        let light_proj = Mat4::orthographic_rh(
            bounds_min.x,
            bounds_max.x,
            bounds_min.y,
            bounds_max.y,
            near_plane,
            far_plane,
        );
        let sun_view_proj = OPENGL_TO_WGPU_MATRIX * light_proj * light_view;
        let sun_view_proj_cols: [[f32; 4]; 4] = sun_view_proj.to_cols_array_2d();

        // Shadow strength based on sun/moon position
        // Smooth continuous transition between sun and moon shadows
        // No discontinuities at horizon crossing
        let shadow_strength = if sun_height > 0.0 {
            // Sun is above horizon - full strength sun shadows
            // Fade in from horizon to avoid sudden appearance
            let fade_in = (sun_height / 0.2).clamp(0.0, 1.0);
            (fade_in * SHADOW_STRENGTH_MULTIPLIER).min(1.0)
        } else {
            // Sun is below horizon - use moon shadows
            let moon_height = -sun_height; // Moon is on opposite side
            // Continuous transition: moon shadows gradually appear as moon rises
            if moon_height < 0.2 {
                // Moon just rising - fade in moon shadows gradually
                let fade = (moon_height / 0.2).clamp(0.0, 1.0);
                fade * 0.4
            } else {
                // Moon well above horizon - stronger shadows at midnight
                let moon_factor = ((moon_height - 0.2) / 0.8).clamp(0.0, 1.0);
                // Moon shadows: 40-70% strength, peaking at midnight
                0.4 + moon_factor * 0.3
            }
        };
        let shadow_texel = 1.0 / self.shadow_map_size as f32;

        // Collect light probes from nearby emissive chunks
        let mut light_probes: Vec<LightProbe> = Vec::new();
        const MAX_LIGHT_PROBES: usize = 32;
        const LIGHT_RADIUS_SQ: f32 = 48.0 * 48.0; // Only consider chunks within 48 units (3 chunks)
        const MIN_EMISSIVE_POWER: f32 = 0.5; // Ignore weak emitters

        // Collect emitters from chunks with cached meshes
        for (chunk_key, emitters) in &self.chunk_emitters {
            if light_probes.len() >= MAX_LIGHT_PROBES {
                break;
            }

            // Check if chunk is reasonably close to camera
            let chunk_center = [
                chunk_key.0 as f32 + 8.0,
                chunk_key.1 as f32 + 8.0,
                chunk_key.2 as f32 + 8.0,
            ];
            let dx = chunk_center[0] - camera_pos[0];
            let dy = chunk_center[1] - camera_pos[1];
            let dz = chunk_center[2] - camera_pos[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq > LIGHT_RADIUS_SQ {
                continue;
            }

            // Aggregate all emitters in this chunk into one light probe
            let mut total_color = [0.0f32; 3];
            let mut total_power = 0.0f32;

            for emitter in emitters {
                total_color[0] += emitter.color[0] * emitter.intensity;
                total_color[1] += emitter.color[1] * emitter.intensity;
                total_color[2] += emitter.color[2] * emitter.intensity;
                total_power += emitter.intensity;
            }

            if total_power > MIN_EMISSIVE_POWER {
                light_probes.push(LightProbe {
                    position: chunk_center,
                    _pad0: 0.0,
                    color_power: [total_color[0], total_color[1], total_color[2], total_power],
                });
            }
        }

        // Sort by distance and keep only nearest probes
        light_probes.sort_by(|a, b| {
            let dist_a_sq = (a.position[0] - camera_pos[0]).powi(2)
                + (a.position[1] - camera_pos[1]).powi(2)
                + (a.position[2] - camera_pos[2]).powi(2);
            let dist_b_sq = (b.position[0] - camera_pos[0]).powi(2)
                + (b.position[1] - camera_pos[1]).powi(2)
                + (b.position[2] - camera_pos[2]).powi(2);
            dist_a_sq.partial_cmp(&dist_b_sq).unwrap()
        });
        light_probes.truncate(MAX_LIGHT_PROBES);

        // Upload light probes to GPU
        if !light_probes.is_empty() {
            // Ensure we have enough capacity
            if light_probes.len() > self.light_probe_capacity {
                // Recreate buffer with more capacity
                self.light_probe_capacity = (light_probes.len() * 2).max(64);
                let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Light Probe Buffer"),
                    size: (self.light_probe_capacity * std::mem::size_of::<LightProbe>())
                        as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.light_probe_buffer = Some(new_buffer);
                self.bind_group = None; // Force bind group recreation
            }

            queue.write_buffer(
                self.light_probe_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&light_probes),
            );
        }

        let light_probe_count = light_probes.len() as u32;

        // Update uniforms with MVP and lighting data
        // Dual light parameters ---------------------------------------------------
        // Moon direction is opposite the sun. Intensity ramps when sun below horizon.
        let moon_direction_vec = -sun_direction_vec;
        let moon_height = (-sun_height).max(0.0);
        // Base moon intensity is low when sun up, increases at night.
        let moon_intensity = if sun_height > 0.0 {
            // Daytime: faint moon, almost invisible
            (0.05 * (1.0 - sun_height.clamp(0.0, 1.0))).clamp(0.0, 0.05)
        } else {
            // Night: ramp from horizon to zenith
            let ramp = (moon_height / 1.0).clamp(0.0, 1.0);
            // Slight boost near midnight
            let midnight_boost = (moon_height - 0.3).max(0.0) * 0.15;
            (0.25 * ramp + midnight_boost).clamp(0.02, 0.4)
        };

        // Derive moon color: cooler at night, slight warm tint near twilight
        let moon_color = if sun_height > 0.0 {
            [0.35, 0.38, 0.45]
        } else {
            // Interpolate twilight -> deep night palette
            let cool_night = [0.18, 0.20, 0.30];
            let twilight = [0.30, 0.33, 0.42];
            let f = (moon_height / 1.0).clamp(0.0, 1.0);
            [
                twilight[0] + (cool_night[0] - twilight[0]) * f,
                twilight[1] + (cool_night[1] - twilight[1]) * f,
                twilight[2] + (cool_night[2] - twilight[2]) * f,
            ]
        };

        let uniforms = Uniforms {
            mvp: mvp_cols,
            sun_view_proj: sun_view_proj_cols,
            camera_shadow_strength: [camera_pos[0], camera_pos[1], camera_pos[2], shadow_strength],
            sun_direction_shadow_bias: [
                sun_direction_vec.x,
                sun_direction_vec.y,
                sun_direction_vec.z,
                SHADOW_BIAS,
            ],
            fog_time_pad: [self.fog_density, self.time_of_day, 0.0, 0.0],
            sun_color_pad: [sun_color[0], sun_color[1], sun_color[2], 0.0],
            ambient_color_pad: [ambient_color[0], ambient_color[1], ambient_color[2], 0.0],
            shadow_texel_size_pad: [shadow_texel, shadow_texel, 0.0, 0.0],
            moon_direction_intensity: [
                moon_direction_vec.x,
                moon_direction_vec.y,
                moon_direction_vec.z,
                moon_intensity,
            ],
            moon_color_pad: [moon_color[0], moon_color[1], moon_color[2], 0.0],
            light_probe_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        queue.write_buffer(
            self.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        if self.shadow_view.is_none() {
            self.recreate_shadow_map();
        }
        if self.shadow_bind_group.is_none() {
            self.update_shadow_bind_group();
        }
        if self.bind_group.is_none() {
            self.update_main_bind_group();
        }

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

        if let (
            Some(shadow_view),
            Some(shadow_pipeline),
            Some(shadow_mesh_pipeline),
            Some(shadow_bind_group),
        ) = (
            self.shadow_view.as_ref(),
            self.shadow_pipeline.as_ref(),
            self.shadow_mesh_pipeline.as_ref(),
            self.shadow_bind_group.as_ref(),
        ) {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if has_meshes_to_draw {
                shadow_pass.set_pipeline(shadow_mesh_pipeline);
                shadow_pass.set_bind_group(0, shadow_bind_group, &[]);
                for key in draw_mesh_keys.iter() {
                    if let Some(entry) = self.mesh_cache.get_mut(key) {
                        shadow_pass.set_vertex_buffer(0, entry.vertex_buffer.slice(..));
                        shadow_pass.set_index_buffer(
                            entry.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        shadow_pass.draw_indexed(0..entry.index_count, 0, 0..1);
                    }
                }
            }

            if !instances.is_empty() {
                shadow_pass.set_pipeline(shadow_pipeline);
                shadow_pass.set_bind_group(0, shadow_bind_group, &[]);
                shadow_pass
                    .set_vertex_buffer(0, self.cube_vertex_buffer.as_ref().unwrap().slice(..));
                shadow_pass.set_vertex_buffer(1, self.instance_buffer.as_ref().unwrap().slice(..));
                shadow_pass.draw(0..36, 0..instances.len() as u32);
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
                for key in draw_mesh_keys.iter() {
                    if let Some(entry) = self.mesh_cache.get_mut(key) {
                        render_pass.set_vertex_buffer(0, entry.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            entry.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..entry.index_count, 0, 0..1);
                        entry.last_used_frame = self.frame_index;
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

        if self.dof_coc_bind_group.is_none() {
            self.update_dof_coc_bind_group();
        }

        self.update_bloom_uniforms();

        if self.composite_bind_group.is_none()
            || self.bloom_extract_bind_group.is_none()
            || self.bloom_blur_horizontal_bind_group.is_none()
            || self.bloom_blur_vertical_bind_group.is_none()
        {
            self.update_bloom_bind_groups();
        }

        if let (Some(dof_coc_pipeline), Some(dof_coc_bind_group), Some(dof_coc_view)) = (
            self.dof_coc_pipeline.as_ref(),
            self.dof_coc_bind_group.as_ref(),
            self.dof_coc_view.as_ref(),
        ) {
            let mut coc_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DoF CoC Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dof_coc_view,
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
            coc_pass.set_pipeline(dof_coc_pipeline);
            coc_pass.set_bind_group(0, dof_coc_bind_group, &[]);
            coc_pass.draw(0..3, 0..1);
        }

        if let (Some(dof_pipeline), Some(dof_bind_group), Some(dof_buffer), Some(dof_color_view)) = (
            self.dof_pipeline.as_ref(),
            self.dof_bind_group.as_ref(),
            self.dof_uniform_buffer.as_ref(),
            self.dof_color_view.as_ref(),
        ) {
            let blur_strength = if self.dof_enabled {
                self.dof_settings.blur_strength
            } else {
                0.0
            };
            let gpu_uniforms = self.pack_dof_uniforms(blur_strength);
            queue.write_buffer(dof_buffer, 0, bytemuck::cast_slice(&gpu_uniforms));

            let mut post_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DoF Blur Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dof_color_view,
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

        // Final DoF combine: source color + blurred DoF + CoC => post_color_view
        if let (Some(dof_combine_pipeline), Some(dof_combine_bind_group), Some(post_color_view)) = (
            self.dof_combine_pipeline.as_ref(),
            self.dof_combine_bind_group.as_ref(),
            self.post_color_view.as_ref(),
        ) {
            let mut combine_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DoF Combine Pass"),
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
            combine_pass.set_pipeline(dof_combine_pipeline);
            combine_pass.set_bind_group(0, dof_combine_bind_group, &[]);
            combine_pass.draw(0..3, 0..1);
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
            let mesh_cache_mib = self.mesh_cache_bytes as f64 / (1024.0 * 1024.0);
            let mesh_budget_mib = self.mesh_cache_byte_budget() as f64 / (1024.0 * 1024.0);
            self.system_info.refresh_process(self.process_pid);
            let process_mem_mib = self
                .system_info
                .process(self.process_pid)
                .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0);
            println!(
                "FPS: {}, Visible items: {}, Leaf chunks: {}, Meshed chunks: {}, Pending: {}, Fallback: {}, Mesh cache: {:.1}/{:.1} MiB, Process: {:.1} MiB, Cull: {:.2}ms, Group: {:.2}ms, Mesh: {:.2}ms, Instances: {:.2}ms",
                self.frame_count,
                total_visible,
                leaf_chunks.len(),
                draw_mesh_keys.len(),
                self.pending_chunk_meshes.len(),
                missing_chunks.len(),
                mesh_cache_mib,
                mesh_budget_mib,
                process_mem_mib,
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
