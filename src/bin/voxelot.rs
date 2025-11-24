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
use std::sync::atomic::{AtomicUsize, Ordering};
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

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use sysinfo::{Pid, ProcessExt, System, SystemExt};
use voxelot::SlabAllocator;
use voxelot::{
    bbox_local_to_world, cull_visible_voxels_parallel, Camera, Chunk, ChunkMesh, Palette,
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
const SHADOW_FRUSTUM_EXTENT_MIN: f32 = 150.0;
const SHADOW_FRUSTUM_EXTENT_MAX: f32 = 600.0;
const SHADOW_DISTANCE_MULTIPLIER: f32 = 2.5;
const SHADOW_BIAS: f32 = 0.001;
const SHADOW_STRENGTH_MULTIPLIER: f32 = 1.75;

/// Voxel instance data for GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VoxelInstanceRaw {
    position: [f32; 3],
    voxel_type: u32,
    scale: [f32; 3],        // Scale factor (non-uniform)
    ao_factor: f32,         // Ambient occlusion / occupancy factor (0.0..=1.0)
    custom_color: [f32; 4], // RGBA custom color (if custom_color.a > 0, use this instead of voxel_type)
    emissive: [f32; 4],
}

/// Input layout for GPU culling compute pass (std430-friendly)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstanceInput {
    position: [f32; 3],
    _pad0: u32,
    scale: [f32; 3],
    _pad1: u32,
    custom_color: [f32; 4],
    emissive: [f32; 4],
    voxel_type: u32,
    flags: u32,          // bit 0: has_mesh, bit 1: has_envelope
    mesh_index: u32,     // Index into MeshIndirectBuffer (if has_mesh)
    envelope_index: u32, // Index into EnvelopeIndirectBuffer (if has_envelope)
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
    _pad_align1: [u32; 2], // Align camera_right to 16 bytes
    camera_right: [f32; 3],
    _pad_r0: u32,
    camera_up: [f32; 3],
    _pad_u0: u32,
    fov_tan: f32,
    aspect: f32,
    screen_width: f32,
    screen_height: f32,
    lod_render_distance: f32,
    detail_cull_distance: f32,
    envelope_distance: f32,
    hzb_enabled: u32,
    max_hzb_mip: u32,
    _pad3: f32,
    _pad_align2: [u32; 2],    // Align view_proj to 16 bytes
    view_proj: [[f32; 4]; 4], // 4x4 matrix (column-major)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct HzbParams {
    width: u32,
    height: u32,
    src_mip: u32,
    dst_mip: u32,
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
#[allow(dead_code)]
struct MeshCacheEntry {
    vertex_offset: u64, // Byte offset in mega vertex buffer
    vertex_count: u32,  // Number of vertices
    index_offset: u64,  // Byte offset in mega index buffer
    index_count: u32,
    vertex_bytes: u64,
    index_bytes: u64,
    last_used_frame: u64,
    is_placeholder: bool,
}

impl MeshCacheEntry {
    fn total_bytes(&self) -> u64 {
        if self.is_placeholder {
            0
        } else {
            self.vertex_bytes + self.index_bytes
        }
    }
}
// Ensure impl CameraController is properly closed (fix potential brace mismatch introduced by refactor)

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
    chunk: Arc<Chunk>,
    /// Neighbor chunks snapshot mapped by (-1..=1) offsets from this chunk
    neighbors: FxHashMap<(i8, i8, i8), Arc<Chunk>>,
    envelope: bool,
}

#[derive(Debug)]
struct MeshResult {
    key: (i64, i64, i64),
    mesh: ChunkMesh,
    voxel_count: u32,
    is_envelope: bool,
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
    shadow_darkness_pad: [f32; 4],       // x = shadow darkness multiplier
    moon_direction_intensity: [f32; 4],  // xyz = moon dir, w = intensity scalar
    moon_color_pad: [f32; 4],            // xyz = moon color
    light_probe_count: u32,
    lod_distance: f32, // LOD render distance for fade calculation
    envelope_distance: f32,
    envelope_fade_range: f32,
    inverse_view: [[f32; 4]; 4],
    inverse_proj: [[f32; 4]; 4],
}

/// Depth-of-field runtime settings (CPU-side convenience)
#[derive(Copy, Clone, Debug)]
struct DoFSettings {
    focal_distance: f32,
    focal_range: f32,
    blur_strength: f32,
    kawase_iterations: usize,
    kawase_offset: f32,
    kawase_enabled: bool,
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
    ssao_enabled: f32,
    ssao_debug: f32,
    ssao_strength: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
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

#[derive(Copy, Clone, Debug)]
struct SsaoSettings {
    sample_count: u32,
    slice_count: u32,
    radius: f32,
    thickness: f32,
    strength: f32,
    blur_enabled: bool,
    blur_radius: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniformsRaw {
    sample_count: u32,
    slice_count: u32,
    sample_radius: f32,
    hit_thickness: f32,
    screen_width: f32,
    screen_height: f32,
    _pad0: f32,
    _pad1: f32,
    inverse_projection: [[f32; 4]; 4],
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
            KeyCode::KeyQ => self.down = pressed,
            KeyCode::KeyE => self.up = pressed,
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
            KeyCode::KeyR if pressed => {
                self.camera.config.lod_subdivide_distance =
                    (self.camera.config.lod_subdivide_distance - 50.0).max(50.0);
                println!(
                    "LOD subdivide distance: {:.0}",
                    self.camera.config.lod_subdivide_distance
                );
            }
            KeyCode::KeyT if pressed => {
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
        let forward = self.camera.forward.clone();
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

        // Normalize velocity and move camera
        let mag =
            (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2])
                .sqrt();
        if mag > 0.00001 {
            let inv = 1.0 / mag;
            let dir = [velocity[0] * inv, velocity[1] * inv, velocity[2] * inv];
            let speed = self.base_speed * self.speed_multiplier * self.distance_speed_scale();
            let delta = [
                dir[0] * speed * dt,
                dir[1] * speed * dt,
                dir[2] * speed * dt,
            ];
            self.camera.position[0] += delta[0];
            self.camera.position[1] += delta[1];
            self.camera.position[2] += delta[2];
            self.update_camera_vectors();
        }
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

    shadow_texture: Option<wgpu::Texture>,
    shadow_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
    gpu_input_buffer: Option<wgpu::Buffer>,
    gpu_input_capacity: usize,
    mesh_indirect_buffer: Option<wgpu::Buffer>,
    envelope_indirect_buffer: Option<wgpu::Buffer>,
    fallback_indirect_buffer: Option<wgpu::Buffer>,
    fallback_instance_buffer: Option<wgpu::Buffer>,
    fallback_instance_capacity: usize,
    // Mesh cache: per-leaf-chunk mesh GPU buffers and metadata
    mesh_cache: FxHashMap<(i64, i64, i64), MeshCacheEntry>,
    mesh_cache_bytes: u64,
    // Stats for UI overlay (refreshed each frame)
    visible_count: usize,
    leaf_chunk_count: usize,
    meshed_chunk_count: usize,
    pending_mesh_count: usize,
    pending_mesh_set_count: usize,
    ready_mesh_count: usize,
    jobs_in_flight: usize,
    jobs_per_sec_snapshot: usize,
    process_mem_mib: f64,
    mesh_cache_mib: f64,
    mesh_budget_mib: f64,
    envelope_cache_mib: f64,
    cull_ms: f64,
    grouping_ms: f64,
    mesh_ms: f64,
    instance_ms: f64,
    draw_calls_count: usize,
    envelope_mesh_cache: FxHashMap<(i64, i64, i64), MeshCacheEntry>,
    envelope_mesh_cache_bytes: u64,
    envelope_mesh_cache_budget_bytes: u64,
    envelope_distance: f32,
    envelope_fade_range: f32,
    max_envelope_distance: f32,
    /// Cached Arc<Chunk> snapshots for mesher jobs to avoid repeated deep clones
    mesh_chunk_arc_cache: FxHashMap<(i64, i64, i64), Arc<Chunk>>,
    /// Count of mesh jobs executed per second by worker threads (reset on FPS print)
    mesh_jobs_executed: Arc<AtomicUsize>,

    // Mega-buffer infrastructure
    mega_vertex_buffer: Option<wgpu::Buffer>,
    mega_index_buffer: Option<wgpu::Buffer>,
    vertex_allocator: voxelot::SlabAllocator,
    index_allocator: voxelot::SlabAllocator,

    // Multi-draw indirect buffers
    multi_draw_indirect_buffer: Option<wgpu::Buffer>,
    multi_draw_count_buffer: Option<wgpu::Buffer>,
    max_draw_capacity: usize,

    /// Stat: number of empty meshes processed (non-geometric chunks)
    stat_empty_meshes: u64,
    stat_vertex_buffers_reused: u64,
    stat_index_buffers_reused: u64,

    chunk_emitters: FxHashMap<(i64, i64, i64), Vec<ChunkEmitterWorld>>,
    active_emitters: Vec<ActiveLight>,
    system_info: System,
    process_pid: Pid,

    cull_pipeline: Option<wgpu::ComputePipeline>,
    cull_bind_group_layout: Option<wgpu::BindGroupLayout>,
    cull_bind_group: Option<wgpu::BindGroup>,
    cull_params_buffer: Option<wgpu::Buffer>,
    hzb_params_buffer: Option<wgpu::Buffer>,

    world: World,
    palette: Palette,
    camera_controller: CameraController,
    pending_chunk_meshes: VecDeque<(i64, i64, i64)>,
    pending_chunk_set: FxHashSet<(i64, i64, i64)>,
    mesh_job_tx: Sender<MeshJob>,
    mesh_result_rx: Receiver<MeshResult>,
    mesh_worker_count: usize,
    mesh_jobs_in_flight: usize,
    ready_chunk_meshes: VecDeque<MeshResult>,
    mesh_upload_limit: usize,
    mesh_upload_baseline: usize,
    mesh_upload_max: usize,
    mesh_upload_adjust_timer: f32,
    // Periodic mesh priority sorting (frames)
    pending_mesh_sort_interval_frames: u64,
    last_pending_mesh_sort_frame: u64,
    mesh_cache_budget_bytes: u64,
    fallback_detail_distance: f32,

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
    dof_coc_pipeline: Option<wgpu::RenderPipeline>,
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
    bloom_extract_pipeline: Option<wgpu::RenderPipeline>,
    // SSILVB (GTAO/SSAO with visibility bitmask)
    ssilvb_pipeline: Option<wgpu::RenderPipeline>,
    bloom_blur_pipeline: Option<wgpu::RenderPipeline>,
    ssao_blur_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    // SSILVB bind/group
    ssilvb_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssilvb_bind_group: Option<wgpu::BindGroup>,
    ssilvb_uniform_buffer: Option<wgpu::Buffer>,
    bloom_extract_bind_group_layout: Option<wgpu::BindGroupLayout>,
    bloom_blur_bind_group_layout: Option<wgpu::BindGroupLayout>,
    composite_bind_group_layout: Option<wgpu::BindGroupLayout>,
    bloom_extract_bind_group: Option<wgpu::BindGroup>,
    bloom_blur_horizontal_bind_group: Option<wgpu::BindGroup>,
    bloom_blur_vertical_bind_group: Option<wgpu::BindGroup>,
    composite_bind_group: Option<wgpu::BindGroup>,
    ssao_ping_texture: Option<wgpu::Texture>,
    ssao_ping_view: Option<wgpu::TextureView>,
    ssao_pong_texture: Option<wgpu::Texture>,
    ssao_pong_view: Option<wgpu::TextureView>,
    // ssao_readback_buffer: Option<wgpu::Buffer>, // removed: readback temporarily disabled
    bloom_extract_uniform_buffer: Option<wgpu::Buffer>,
    bloom_blur_horizontal_uniform_buffer: Option<wgpu::Buffer>,
    bloom_blur_vertical_uniform_buffer: Option<wgpu::Buffer>,
    ssao_blur_horizontal_uniform_buffer: Option<wgpu::Buffer>,
    ssao_blur_vertical_uniform_buffer: Option<wgpu::Buffer>,
    ssao_blur_horizontal_bind_group: Option<wgpu::BindGroup>,
    ssao_blur_vertical_bind_group: Option<wgpu::BindGroup>,
    composite_uniform_buffer: Option<wgpu::Buffer>,
    ssao_enabled: bool,
    ssao_debug: bool,

    // DoF color buffer that stores blurred result before combine
    dof_color_texture: Option<wgpu::Texture>,
    dof_color_view: Option<wgpu::TextureView>,
    // HZB resources
    hzb_texture: Option<wgpu::Texture>,
    hzb_view: Option<wgpu::TextureView>,
    hzb_mip_views: Vec<wgpu::TextureView>, // Per-mip views for storage
    hzb_mip_levels: u32,
    hzb_gen_copy_pipeline: Option<wgpu::ComputePipeline>,
    hzb_gen_downsample_pipeline: Option<wgpu::ComputePipeline>,
    hzb_gen_bind_group_layout: Option<wgpu::BindGroupLayout>,
    hzb_copy_bind_group: Option<wgpu::BindGroup>, // For mip 0 copy
    hzb_downsample_bind_groups: Vec<wgpu::BindGroup>, // Per-mip downsample bind groups
    hzb_bind_group_layout: Option<wgpu::BindGroupLayout>,
    hzb_bind_groups: Vec<Option<wgpu::BindGroup>>,
    hzb_enabled: bool,

    dof_combine_pipeline: Option<wgpu::RenderPipeline>,
    dof_combine_bind_group_layout: Option<wgpu::BindGroupLayout>,
    dof_combine_bind_group: Option<wgpu::BindGroup>,
    dof_settings: DoFSettings,
    dof_enabled: bool,
    // Dual Kawase resources for DoF blur
    kawase_down_pipeline: Option<wgpu::RenderPipeline>,
    kawase_up_pipeline: Option<wgpu::RenderPipeline>,
    kawase_bind_group_layout: Option<wgpu::BindGroupLayout>,
    kawase_down_bind_groups: Vec<Option<wgpu::BindGroup>>,
    kawase_up_bind_groups: Vec<Option<wgpu::BindGroup>>,
    kawase_uniform_buffers: Vec<Option<wgpu::Buffer>>,
    kawase_ping_textures: Vec<Option<wgpu::Texture>>,
    kawase_ping_views: Vec<Option<wgpu::TextureView>>,
    kawase_pong_textures: Vec<Option<wgpu::Texture>>,
    kawase_pong_views: Vec<Option<wgpu::TextureView>>,
    // Kawase per-level extents (width, height)
    kawase_level_sizes: Vec<(u32, u32)>,
    kawase_last_ubo: Vec<[f32; 4]>,
    // Performance instrumentation: accumulate kawase timing
    kawase_write_acc: std::time::Duration,
    kawase_pass_acc: std::time::Duration,
    kawase_acc_frames: u64,
    bloom_settings: BloomSettings,
    bloom_enabled: bool,
    ssao_settings: SsaoSettings,
    shadow_map_size: u32,
    shadow_darkness: f32,
    shadow_backface_scale: f32,
    pcf_radius: f32,
    pcf_poisson_samples: u32,
    // Mesh statistics
    _mesh_buffer_pool_max_entries: usize,

    // egui UI state
    egui_ctx: Option<egui::Context>,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    last_fps: u32,

    // Skybox
    skybox_texture: Option<wgpu::Texture>,
    skybox_texture_bytes: u64,
    skybox_view: Option<wgpu::TextureView>,
    skybox_sampler: Option<wgpu::Sampler>,
    skybox_bind_group: Option<wgpu::BindGroup>,
    skybox_bind_group_layout: Option<wgpu::BindGroupLayout>,
    skybox_pipeline: Option<wgpu::RenderPipeline>,
    // GPU accounting (accurate tracked allocations)
    gpu_buffer_bytes: u64,
    gpu_texture_bytes: u64,
    uniform_buffer_bytes: u64,
    mega_vertex_buffer_bytes: u64,
    mega_index_buffer_bytes: u64,
    multi_draw_indirect_bytes: u64,
    multi_draw_count_bytes: u64,
    gpu_input_buffer_bytes: u64,
    fallback_instance_buffer_bytes: u64,
    fallback_indirect_bytes: u64,
    mesh_indirect_bytes: u64,
    envelope_indirect_bytes: u64,
    cull_params_buffer_bytes: u64,
    offscreen_color_texture_bytes: u64,
    depth_texture_bytes: u64,
    post_color_texture_bytes: u64,
    dof_color_texture_bytes: u64,
    bloom_ping_bytes: u64,
    bloom_pong_bytes: u64,
    kawase_ping_bytes: u64,
    kawase_pong_bytes: u64,
    ssao_ping_bytes: u64,
    ssao_pong_bytes: u64,
    shadow_map_bytes: u64,
    hzb_texture_bytes: u64,
    hzb_params_buffer_bytes: u64,
    cube_vertex_buffer_bytes: u64,
    light_probe_buffer_bytes: u64,
}

impl App {
    // Static helpers that avoid borrowing &mut self during the operation, allowing
    // callers to pass distinct fields as &mut u64 without creating overlapping
    // &mut self borrows which the Rust borrow-checker rejects.
    fn replace_buffer_bytes_static(old: &mut u64, new: u64, agg: &mut u64) {
        if *old > 0 {
            *agg = agg.saturating_sub(*old);
        }
        *agg = agg.saturating_add(new);
        *old = new;
    }

    fn replace_texture_bytes_static(old: &mut u64, new: u64, agg: &mut u64) {
        if *old > 0 {
            *agg = agg.saturating_sub(*old);
        }
        *agg = agg.saturating_add(new);
        *old = new;
    }

    // Compute bytes occupied by a texture with mipmaps
    fn compute_texture_bytes(
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
        mip_level_count: u32,
    ) -> u64 {
        let bpp: u64 = match format {
            wgpu::TextureFormat::Rgba32Float => 16,
            wgpu::TextureFormat::Rgba16Float => 8,
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb => 4,
            wgpu::TextureFormat::Depth32Float => 4,
            _ => 4,
        };
        let mut total: u64 = 0;
        for level in 0..mip_level_count {
            let w = (width >> level).max(1) as u64;
            let h = (height >> level).max(1) as u64;
            let layers = depth_or_array_layers as u64;
            total = total.saturating_add(
                w.saturating_mul(h)
                    .saturating_mul(layers)
                    .saturating_mul(bpp),
            );
        }
        total
    }
    fn new() -> Self {
        let mut system_info = System::new();
        let process_pid = Pid::from(std::process::id() as usize);
        system_info.refresh_process(process_pid);

        // Load configuration once for all initialization
        let cfg = voxelot::Config::load_or_default(CONFIG_FILE);

        let initial_camera;
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
            initial_camera = cfg.world.camera_position;

            println!("Loading voxel data from {}...", cfg.world.file);
            // Load octree format from configured path — use auto-detecting loader
            let load_start = Instant::now();
            world = voxelot::load_world_file(std::path::Path::new(&cfg.world.file)).unwrap_or_else(
                |e| {
                    eprintln!(
                        "ERROR: Failed to load world file '{}': {}",
                        cfg.world.file, e
                    );
                    eprintln!("Please check that the file path in config.toml is correct.");
                    std::process::exit(1);
                },
            );
            let load_elapsed = load_start.elapsed();
            println!(
                "Loaded world from {} (depth {}) (took {:.3}s)",
                cfg.world.file,
                world.hierarchy_depth(),
                load_elapsed.as_secs_f32()
            );
        }

        println!("World created with voxels");

        println!("Loading palette from {}...", cfg.world.palette);
        let palette = Palette::load(&cfg.world.palette);

        let (mesh_job_tx, mesh_job_rx) = unbounded::<MeshJob>();
        let (mesh_result_tx, mesh_result_rx) = unbounded::<MeshResult>();
        let mesh_jobs_executed = Arc::new(AtomicUsize::new(0));

        let available_workers = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(2))
            .unwrap_or(1)
            .max(1);
        let mesh_worker_count = cfg
            .performance
            .mesh_worker_count
            .unwrap_or_else(|| available_workers.min(6));

        let worker_palette = Arc::new(palette.clone());

        for worker_index in 0..mesh_worker_count {
            let job_rx = mesh_job_rx.clone();
            let jobs_executed = mesh_jobs_executed.clone();
            let result_tx = mesh_result_tx.clone();
            let palette = worker_palette.clone();

            std::thread::Builder::new()
                .name(format!("mesh-worker-{}", worker_index))
                .spawn(move || {
                    while let Ok(job) = job_rx.recv() {
                        let MeshJob {
                            key,
                            chunk,
                            neighbors,
                            envelope,
                        } = job;

                        let mesh = voxelot::generate_chunk_mesh_optimized(
                            &chunk,
                            &palette,
                            Some(&neighbors),
                            envelope,
                        );

                        if result_tx
                            .send(MeshResult {
                                key,
                                mesh,
                                voxel_count: chunk.voxel_count,
                                is_envelope: envelope,
                            })
                            .is_err()
                        {
                            break;
                        }
                        // Account for this processed mesh job
                        jobs_executed.fetch_add(1, Ordering::Relaxed);
                    }
                })
                .expect("failed to spawn mesh worker");
        }

        drop(mesh_result_tx);
        drop(mesh_job_rx);

        let mesh_upload_baseline = cfg.performance.mesh_upload_baseline;
        let mesh_upload_max = (mesh_worker_count * 4).max(mesh_upload_baseline * 2);

        println!("Updating LOD metadata...");
        let lod_start = Instant::now();
        world.update_all_lod_metadata(&palette);
        let lod_elapsed = lod_start.elapsed();
        println!(
            "LOD metadata updated (took {:.3}s)",
            lod_elapsed.as_secs_f32()
        );

        println!("\n=== Controls ===");
        println!("Movement: WASD + Q/E (down/up)");
        println!("Look: Right Mouse + drag");
        println!("Rotate: Arrow Keys (Left/Right yaw, Up/Down pitch)");
        println!("Camera Speed: -/+ (decrease/increase multiplier), 0 reset");
        println!("Camera LOD Distance: R/T (decrease/increase)");
        println!("Draw Distance: Z/C (decrease/increase)");
        println!("Chunk LOD Distance: K/L (decrease/increase)");
        println!("Time of Day: T (cycle through day/night)");
        println!("Fog Density: F/G (decrease/increase)");
        println!("Depth of Field: / (toggle), , and . adjust focus");
        println!("Kawase DoF: X (toggle), U/I (offset -/+), O/P (iterations -/+)");
        println!("Bloom: B (toggle)");
        println!("HZB: J (toggle)");
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

            shadow_texture: None,
            shadow_view: None,
            shadow_sampler: None,
            gpu_input_buffer: None,
            gpu_input_capacity: 0,
            mesh_indirect_buffer: None,
            envelope_indirect_buffer: None,
            fallback_indirect_buffer: None,
            fallback_instance_buffer: None,
            fallback_instance_capacity: 0,
            mesh_cache: FxHashMap::default(),
            mesh_jobs_executed: mesh_jobs_executed.clone(),
            mesh_cache_bytes: 0,
            envelope_mesh_cache: FxHashMap::default(),
            envelope_mesh_cache_bytes: 0,
            envelope_mesh_cache_budget_bytes: cfg.performance.mesh_cache_budget_mb as u64
                * 1024
                * 1024,
            envelope_distance: cfg.performance.envelope_distance,
            envelope_fade_range: cfg.performance.envelope_fade_range,
            max_envelope_distance: cfg.performance.max_envelope_distance,
            mega_vertex_buffer: None,
            mega_index_buffer: None,
            vertex_allocator: SlabAllocator::new(
                cfg.performance.mega_vertex_buffer_mb * 1024 * 1024,
            ),
            index_allocator: SlabAllocator::new(cfg.performance.mega_index_buffer_mb * 1024 * 1024),
            multi_draw_indirect_buffer: None,
            multi_draw_count_buffer: None,
            max_draw_capacity: cfg.performance.max_draw_capacity,

            egui_ctx: None,
            egui_winit: None,
            egui_renderer: None,
            last_fps: 0,
            mesh_chunk_arc_cache: FxHashMap::default(),
            // empty_mesh buffers removed; placeholders use offsets into mega buffers
            stat_empty_meshes: 0,
            stat_vertex_buffers_reused: 0,
            stat_index_buffers_reused: 0,
            _mesh_buffer_pool_max_entries: cfg.performance.mesh_buffer_pool_entries,

            chunk_emitters: FxHashMap::default(),
            active_emitters: Vec::new(),
            system_info,
            process_pid,

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
            visible_count: 0,
            leaf_chunk_count: 0,
            meshed_chunk_count: 0,
            pending_mesh_count: 0,
            pending_mesh_set_count: 0,
            ready_mesh_count: 0,
            jobs_in_flight: 0,
            jobs_per_sec_snapshot: 0,
            process_mem_mib: 0.0,
            mesh_cache_mib: 0.0,
            mesh_budget_mib: 0.0,
            envelope_cache_mib: 0.0,
            cull_ms: 0.0,
            grouping_ms: 0.0,
            mesh_ms: 0.0,
            instance_ms: 0.0,
            draw_calls_count: 0,
            mesh_upload_limit: mesh_upload_baseline,
            mesh_upload_baseline,
            mesh_upload_max,
            mesh_upload_adjust_timer: 0.0,
            pending_mesh_sort_interval_frames: cfg.performance.mesh_priority_sort_interval_frames,
            last_pending_mesh_sort_frame: 0,
            mesh_cache_budget_bytes: cfg.performance.mesh_cache_budget_mb as u64 * 1024 * 1024,
            fallback_detail_distance: cfg.performance.fallback_detail_distance,

            camera_controller: CameraController::new(initial_camera, &cfg.rendering),
            pending_chunk_meshes: VecDeque::new(),
            pending_chunk_set: FxHashSet::default(),
            last_frame: Instant::now(),
            frame_count: 0,
            frame_index: 0,
            last_fps_print: Instant::now(),
            mouse_pressed: false,
            last_mouse_pos: None,
            time_of_day: cfg.atmosphere.time_of_day,
            time_paused: false,
            fog_density: cfg.atmosphere.fog_density,
            light_probe_buffer: None,
            light_probe_capacity: 0,
            lod_distance: cfg.rendering.chunk_lod_distance,
            dof_coc_pipeline: None,
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
            ssao_ping_texture: None,
            ssao_ping_view: None,
            ssao_pong_texture: None,
            ssao_pong_view: None,
            // ssao_readback_buffer: None,
            bloom_extract_pipeline: None,
            bloom_blur_pipeline: None,
            ssao_blur_pipeline: None,
            composite_pipeline: None,
            bloom_extract_bind_group_layout: None,
            bloom_blur_bind_group_layout: None,
            composite_bind_group_layout: None,
            bloom_extract_bind_group: None,
            bloom_blur_horizontal_bind_group: None,
            bloom_blur_vertical_bind_group: None,
            composite_bind_group: None,
            bloom_extract_uniform_buffer: None,
            bloom_blur_horizontal_uniform_buffer: None,
            bloom_blur_vertical_uniform_buffer: None,
            ssao_blur_horizontal_uniform_buffer: None,
            ssao_blur_vertical_uniform_buffer: None,
            ssao_blur_horizontal_bind_group: None,
            ssao_blur_vertical_bind_group: None,
            composite_uniform_buffer: None,
            ssilvb_pipeline: None,
            ssilvb_bind_group_layout: None,
            ssilvb_bind_group: None,
            ssilvb_uniform_buffer: None,
            dof_color_texture: None,
            dof_color_view: None,
            hzb_texture: None,
            hzb_view: None,
            hzb_mip_views: Vec::new(),
            hzb_mip_levels: 0,
            hzb_gen_copy_pipeline: None,
            hzb_gen_downsample_pipeline: None,
            hzb_gen_bind_group_layout: None,
            hzb_copy_bind_group: None,
            hzb_downsample_bind_groups: Vec::new(),
            hzb_bind_group_layout: None,
            hzb_bind_groups: Vec::new(),
            hzb_params_buffer: None,
            hzb_enabled: cfg.performance.hzb_enabled,
            dof_combine_pipeline: None,
            dof_combine_bind_group_layout: None,
            dof_combine_bind_group: None,
            dof_settings: DoFSettings {
                focal_distance: cfg.effects.depth_of_field.focal_distance,
                focal_range: cfg.effects.depth_of_field.focal_range,
                blur_strength: cfg.effects.depth_of_field.blur_strength,
                kawase_iterations: cfg.effects.depth_of_field.kawase_iterations,
                kawase_offset: cfg.effects.depth_of_field.kawase_offset,
                kawase_enabled: cfg.effects.depth_of_field.kawase_enabled,
            },
            dof_enabled: cfg.effects.depth_of_field.enabled,
            kawase_down_pipeline: None,
            kawase_up_pipeline: None,
            kawase_bind_group_layout: None,
            kawase_down_bind_groups: Vec::new(),
            kawase_up_bind_groups: Vec::new(),
            kawase_uniform_buffers: Vec::new(),
            kawase_ping_textures: Vec::new(),
            kawase_ping_views: Vec::new(),
            kawase_pong_textures: Vec::new(),
            kawase_pong_views: Vec::new(),
            kawase_level_sizes: Vec::new(),
            kawase_last_ubo: Vec::new(),
            // Instrumentation init
            kawase_write_acc: std::time::Duration::from_secs(0),
            kawase_pass_acc: std::time::Duration::from_secs(0),
            kawase_acc_frames: 0,
            bloom_settings: BloomSettings {
                threshold: cfg.effects.bloom.threshold,
                knee: cfg.effects.bloom.knee,
                intensity: cfg.effects.bloom.intensity,
                bloom_strength: cfg.effects.bloom.bloom_strength,
                saturation_boost: cfg.effects.bloom.saturation_boost,
                exposure: cfg.effects.bloom.exposure,
                blur_radius: cfg.effects.bloom.blur_radius,
            },
            bloom_enabled: cfg.effects.bloom.enabled,
            ssao_settings: SsaoSettings {
                sample_count: cfg.effects.ssao.sample_count,
                slice_count: cfg.effects.ssao.slice_count,
                radius: 4.0,
                thickness: 0.5,
                strength: cfg.effects.ssao.strength,
                blur_enabled: cfg.effects.ssao.blur_enabled,
                blur_radius: cfg.effects.ssao.blur_radius,
            },
            ssao_enabled: cfg.effects.ssao.enabled,
            ssao_debug: false,
            shadow_map_size: cfg.shadows.map_size,
            shadow_darkness: cfg.shadows.darkness,
            shadow_backface_scale: cfg.shadows.backface_ambient_scale,
            pcf_radius: cfg.shadows.pcf_radius,
            pcf_poisson_samples: cfg.shadows.pcf_poisson_samples,

            skybox_texture: None,
            skybox_texture_bytes: 0,
            skybox_view: None,
            skybox_sampler: None,
            skybox_bind_group: None,
            skybox_bind_group_layout: None,
            skybox_pipeline: None,
            // GPU accounting init
            gpu_buffer_bytes: 0,
            gpu_texture_bytes: 0,
            uniform_buffer_bytes: 0,
            mega_vertex_buffer_bytes: 0,
            mega_index_buffer_bytes: 0,
            multi_draw_indirect_bytes: 0,
            multi_draw_count_bytes: 0,
            gpu_input_buffer_bytes: 0,
            fallback_instance_buffer_bytes: 0,
            fallback_indirect_bytes: 0,
            mesh_indirect_bytes: 0,
            envelope_indirect_bytes: 0,
            cull_params_buffer_bytes: 0,
            offscreen_color_texture_bytes: 0,
            depth_texture_bytes: 0,
            post_color_texture_bytes: 0,
            dof_color_texture_bytes: 0,
            bloom_ping_bytes: 0,
            bloom_pong_bytes: 0,
            kawase_ping_bytes: 0,
            kawase_pong_bytes: 0,
            ssao_ping_bytes: 0,
            ssao_pong_bytes: 0,
            shadow_map_bytes: 0,
            hzb_texture_bytes: 0,
            hzb_params_buffer_bytes: 0,
            cube_vertex_buffer_bytes: 0,
            light_probe_buffer_bytes: 0,
        }
    }

    fn save_config(&self) {
        // Persist full TOML config using unified Config (camera speed multiplier retained)
        // We read existing file, update rendering subsection relevant fields, then save.
        if let Ok(mut full_cfg) = voxelot::Config::load(CONFIG_FILE) {
            // Rendering settings
            full_cfg.rendering.lod_subdivide_distance =
                self.camera_controller.camera.config.lod_subdivide_distance;
            full_cfg.rendering.lod_merge_distance =
                self.camera_controller.camera.config.lod_merge_distance;
            full_cfg.rendering.chunk_lod_distance =
                self.camera_controller.camera.config.lod_render_distance;
            full_cfg.rendering.fov_degrees = self.camera_controller.camera.config.fov_degrees;
            full_cfg.rendering.near_plane = self.camera_controller.camera.config.near_plane;
            full_cfg.rendering.far_plane = self.camera_controller.camera.config.far_plane;
            full_cfg.rendering.camera_speed_multiplier = self.camera_controller.speed_multiplier;

            // Atmosphere settings
            full_cfg.atmosphere.time_of_day = self.time_of_day;
            full_cfg.atmosphere.fog_density = self.fog_density;

            // DoF settings
            full_cfg.effects.depth_of_field.enabled = self.dof_enabled;
            full_cfg.effects.depth_of_field.focal_distance = self.dof_settings.focal_distance;
            full_cfg.effects.depth_of_field.focal_range = self.dof_settings.focal_range;
            full_cfg.effects.depth_of_field.blur_strength = self.dof_settings.blur_strength;
            full_cfg.effects.depth_of_field.kawase_iterations = self.dof_settings.kawase_iterations;
            full_cfg.effects.depth_of_field.kawase_offset = self.dof_settings.kawase_offset;
            full_cfg.effects.depth_of_field.kawase_enabled = self.dof_settings.kawase_enabled;
            full_cfg.effects.depth_of_field.blur_strength = self.dof_settings.blur_strength;

            // Bloom settings
            full_cfg.effects.bloom.enabled = self.bloom_enabled;
            full_cfg.effects.bloom.threshold = self.bloom_settings.threshold;
            full_cfg.effects.bloom.knee = self.bloom_settings.knee;
            full_cfg.effects.bloom.intensity = self.bloom_settings.intensity;
            full_cfg.effects.bloom.bloom_strength = self.bloom_settings.bloom_strength;
            full_cfg.effects.bloom.saturation_boost = self.bloom_settings.saturation_boost;
            full_cfg.effects.bloom.exposure = self.bloom_settings.exposure;
            full_cfg.effects.bloom.blur_radius = self.bloom_settings.blur_radius;

            // Shadow settings
            full_cfg.shadows.map_size = self.shadow_map_size;
            full_cfg.shadows.backface_ambient_scale = self.shadow_backface_scale;
            full_cfg.shadows.darkness = self.shadow_darkness;

            // Performance settings
            full_cfg.performance.mesh_worker_count = Some(self.mesh_worker_count);
            full_cfg.performance.mesh_upload_baseline = self.mesh_upload_baseline;
            full_cfg.performance.mesh_cache_budget_mb =
                self.mesh_cache_budget_bytes / (1024 * 1024);
            full_cfg.performance.fallback_detail_distance = self.fallback_detail_distance;
            full_cfg.performance.mesh_priority_sort_interval_frames =
                self.pending_mesh_sort_interval_frames;

            if let Err(e) = full_cfg.save(CONFIG_FILE) {
                eprintln!("Failed to save unified config: {}", e);
            } else {
                println!("Saved unified TOML config to {}", CONFIG_FILE);
            }
        } else {
            eprintln!("Warning: could not load existing TOML config for update; creating default.");
            let mut full_cfg = voxelot::Config::default();
            full_cfg.rendering.lod_subdivide_distance =
                self.camera_controller.camera.config.lod_subdivide_distance;
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

    fn build_ssao_blur_uniforms(
        &self,
        target_width: u32,
        target_height: u32,
        direction: [f32; 2],
    ) -> BloomBlurUniforms {
        BloomBlurUniforms {
            direction,
            radius: self.ssao_settings.blur_radius,
            _padding0: 0.0,
            texel_size: [
                1.0 / target_width.max(1) as f32,
                1.0 / target_height.max(1) as f32,
            ],
            _padding1: [0.0; 2],
        }
    }

    fn build_ssilvb_uniforms(&self, src_width: u32, src_height: u32) -> SsaoUniformsRaw {
        let aspect = src_width as f32 / src_height as f32;
        let projection = Mat4::perspective_rh(
            self.camera_controller.camera.fov,
            aspect,
            self.camera_controller.camera.near,
            self.camera_controller.camera.far,
        );
        // Match main renderer's projection transform (even if redundant for glam, it's what's in the depth buffer)
        const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
        ]);
        let corrected_projection = OPENGL_TO_WGPU_MATRIX * projection;
        let inv_proj = corrected_projection.inverse();
        SsaoUniformsRaw {
            sample_count: self.ssao_settings.sample_count as u32,
            slice_count: self.ssao_settings.slice_count as u32,
            sample_radius: self.ssao_settings.radius,
            hit_thickness: self.ssao_settings.thickness,
            screen_width: src_width as f32,
            screen_height: src_height as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            inverse_projection: inv_proj.to_cols_array_2d(),
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
            ssao_enabled: if self.ssao_enabled { 1.0 } else { 0.0 },
            ssao_debug: if self.ssao_debug { 1.0 } else { 0.0 },
            ssao_strength: self.ssao_settings.strength,
            _padding0: 0.0,
            _padding1: 0.0,
            _padding2: 0.0,
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
                    if self.time_paused {
                        "paused"
                    } else {
                        "resumed"
                    },
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
            KeyCode::KeyN => {
                self.ssao_enabled = !self.ssao_enabled;
                println!(
                    "SSAO {}",
                    if self.ssao_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
            KeyCode::F1 => {
                // decrease SSILVB sample count
                if self.ssao_settings.sample_count > 1 {
                    self.ssao_settings.sample_count =
                        self.ssao_settings.sample_count.saturating_sub(1);
                    println!("SSAO sample_count: {}", self.ssao_settings.sample_count);
                }
            }
            KeyCode::F2 => {
                // increase SSILVB sample count
                self.ssao_settings.sample_count = (self.ssao_settings.sample_count + 1).min(32);
                println!("SSAO sample_count: {}", self.ssao_settings.sample_count);
            }
            KeyCode::F3 => {
                // decrease sampling radius
                self.ssao_settings.radius = (self.ssao_settings.radius - 1.0).max(0.0);
                println!("SSAO radius: {}", self.ssao_settings.radius);
            }
            KeyCode::F4 => {
                // increase sampling radius
                self.ssao_settings.radius += 1.0;
                println!("SSAO radius: {}", self.ssao_settings.radius);
            }
            KeyCode::KeyH => {
                self.ssao_debug = !self.ssao_debug;
                println!(
                    "SSAO debug {}",
                    if self.ssao_debug {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                if self.ssao_debug {
                    // schedule immediate readback of SSAO ping texture to print stats
                    // readback currently disabled, visual debug still active
                }
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
            KeyCode::KeyX => {
                self.dof_settings.kawase_enabled = !self.dof_settings.kawase_enabled;
                println!(
                    "Kawase {}",
                    if self.dof_settings.kawase_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                // Recreate kawase UBOs/bind groups if enabled
                if self.dof_settings.kawase_enabled {
                    self.update_kawase_bind_groups();
                }
            }
            KeyCode::KeyJ => {
                self.hzb_enabled = !self.hzb_enabled;
                println!(
                    "HZB {}",
                    if self.hzb_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                // Note: HZB texture recreation happens automatically on next frame via existing logic
            }
            KeyCode::KeyU => {
                self.dof_settings.kawase_offset =
                    (self.dof_settings.kawase_offset - 0.25).max(0.25);
                println!("Kawase offset: {:.2}", self.dof_settings.kawase_offset);
                self.update_kawase_bind_groups();
            }
            KeyCode::KeyI => {
                self.dof_settings.kawase_offset =
                    (self.dof_settings.kawase_offset + 0.25).min(10.0);
                println!("Kawase offset: {:.2}", self.dof_settings.kawase_offset);
                self.update_kawase_bind_groups();
            }
            KeyCode::KeyO => {
                self.dof_settings.kawase_iterations =
                    (self.dof_settings.kawase_iterations.saturating_sub(1)).max(1);
                println!("Kawase iterations: {}", self.dof_settings.kawase_iterations);
                self.update_kawase_bind_groups();
            }
            KeyCode::KeyP => {
                self.dof_settings.kawase_iterations =
                    (self.dof_settings.kawase_iterations + 1).min(6);
                println!("Kawase iterations: {}", self.dof_settings.kawase_iterations);
                self.update_kawase_bind_groups();
            }
            _ => {}
        }
    }

    fn recreate_offscreen_targets(&mut self) {
        let k_iterations: usize = self.dof_settings.kawase_iterations.min(6).max(1);
        // Create textures with device/config in a local scope, store results in locals
        let (
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            post_color_texture,
            post_color_view,
            dof_color_texture,
            dof_color_view,
            bloom_ping_texture,
            bloom_ping_view,
            bloom_pong_texture,
            bloom_pong_view,
            kawase_ping_textures_local,
            kawase_ping_views_local,
            kawase_pong_textures_local,
            kawase_pong_views_local,
            kawase_level_sizes_local,
            ssao_ping_texture,
            ssao_ping_view,
            ssao_pong_texture,
            ssao_pong_view,
            offscreen_color_bytes,
            depth_bytes,
            post_color_bytes,
            dof_color_bytes,
            bloom_ping_bytes,
            bloom_pong_bytes,
            kawase_ping_total_loc,
            kawase_pong_total_loc,
            ssao_bytes,
        ) =
            {
                let (Some(device), Some(config)) = (self.device.as_ref(), self.config.as_ref())
                else {
                    return;
                };
                let color_texture_loc = device.create_texture(&wgpu::TextureDescriptor {
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
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                // Track bloom extract uniform buffer bytes
                let bloom_extract_bytes = std::mem::size_of::<BloomExtractUniforms>() as u64;
                App::replace_buffer_bytes_static(
                    &mut self.uniform_buffer_bytes,
                    bloom_extract_bytes,
                    &mut self.gpu_buffer_bytes,
                );
                let color_view_loc =
                    color_texture_loc.create_view(&wgpu::TextureViewDescriptor::default());
                // Track offscreen color texture bytes
                let offscreen_color_bytes =
                    App::compute_texture_bytes(config.format, config.width, config.height, 1, 1);

                let depth_texture_loc = device.create_texture(&wgpu::TextureDescriptor {
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
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let bloom_blur_bytes = std::mem::size_of::<BloomBlurUniforms>() as u64;
                App::replace_buffer_bytes_static(
                    &mut self.uniform_buffer_bytes,
                    bloom_blur_bytes,
                    &mut self.gpu_buffer_bytes,
                );
                let depth_view_loc =
                    depth_texture_loc.create_view(&wgpu::TextureViewDescriptor::default());
                let depth_bytes = App::compute_texture_bytes(
                    wgpu::TextureFormat::Depth32Float,
                    config.width,
                    config.height,
                    1,
                    1,
                );

                let post_color_texture_loc = device.create_texture(&wgpu::TextureDescriptor {
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
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                App::replace_buffer_bytes_static(
                    &mut self.uniform_buffer_bytes,
                    bloom_blur_bytes,
                    &mut self.gpu_buffer_bytes,
                );
                let post_color_view_loc =
                    post_color_texture_loc.create_view(&wgpu::TextureViewDescriptor::default());
                let post_color_bytes =
                    App::compute_texture_bytes(config.format, config.width, config.height, 1, 1);

                // Fused DoF blurred texture (half resolution) storing color + normalized CoC in alpha.
                let fused_width = (config.width / 2).max(1);
                let fused_height = (config.height / 2).max(1);
                let dof_color_texture_loc = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("DoF Fused HalfRes Texture"),
                    size: wgpu::Extent3d {
                        width: fused_width,
                        height: fused_height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let composite_bytes = std::mem::size_of::<CompositeUniforms>() as u64;
                App::replace_buffer_bytes_static(
                    &mut self.uniform_buffer_bytes,
                    composite_bytes,
                    &mut self.gpu_buffer_bytes,
                );
                let dof_color_view_loc =
                    dof_color_texture_loc.create_view(&wgpu::TextureViewDescriptor::default());
                let dof_color_bytes =
                    App::compute_texture_bytes(config.format, fused_width, fused_height, 1, 1);

                let bloom_extent = wgpu::Extent3d {
                    width: (config.width / 2).max(1),
                    height: (config.height / 2).max(1),
                    depth_or_array_layers: 1,
                };

                let bloom_ping_texture_loc = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Bloom Ping Texture"),
                    size: bloom_extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let uniform_bytes = std::mem::size_of::<Uniforms>() as u64;
                App::replace_buffer_bytes_static(
                    &mut self.uniform_buffer_bytes,
                    uniform_bytes,
                    &mut self.gpu_buffer_bytes,
                );
                let bloom_ping_view_loc =
                    bloom_ping_texture_loc.create_view(&wgpu::TextureViewDescriptor::default());

                let bloom_pong_texture_loc = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Bloom Pong Texture"),
                    size: bloom_extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let bloom_pong_view_loc =
                    bloom_pong_texture_loc.create_view(&wgpu::TextureViewDescriptor::default());
                let bloom_ping_bytes = App::compute_texture_bytes(
                    config.format,
                    bloom_extent.width,
                    bloom_extent.height,
                    1,
                    1,
                );
                let bloom_pong_bytes = bloom_ping_bytes;

                // Kawase chain creation: compute per-level ping/pong textures locally
                let mut kawase_ping_textures_loc: Vec<Option<wgpu::Texture>> = Vec::new();
                let mut kawase_ping_views_loc: Vec<Option<wgpu::TextureView>> = Vec::new();
                let mut kawase_pong_textures_loc: Vec<Option<wgpu::Texture>> = Vec::new();
                let mut kawase_pong_views_loc: Vec<Option<wgpu::TextureView>> = Vec::new();
                let mut kawase_level_sizes_loc: Vec<(u32, u32)> = Vec::new();
                let mut kawase_ping_total: u64 = 0;
                let mut kawase_pong_total: u64 = 0;
                for level in 0..k_iterations {
                    let w = (fused_width >> level).max(1);
                    let h = (fused_height >> level).max(1);
                    let ping_tex_loc = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&format!("DoF Kawase Ping L{}", level)),
                        size: wgpu::Extent3d {
                            width: w,
                            height: h,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: config.format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let ping_view_loc =
                        ping_tex_loc.create_view(&wgpu::TextureViewDescriptor::default());
                    let pong_tex_loc = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&format!("DoF Kawase Pong L{}", level)),
                        size: wgpu::Extent3d {
                            width: w,
                            height: h,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: config.format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let pong_view_loc =
                        pong_tex_loc.create_view(&wgpu::TextureViewDescriptor::default());
                    kawase_ping_textures_loc.push(Some(ping_tex_loc));
                    kawase_ping_views_loc.push(Some(ping_view_loc));
                    kawase_pong_textures_loc.push(Some(pong_tex_loc));
                    kawase_pong_views_loc.push(Some(pong_view_loc));
                    kawase_level_sizes_loc.push((w, h));
                    kawase_ping_total = kawase_ping_total
                        .saturating_add(App::compute_texture_bytes(config.format, w, h, 1, 1));
                    kawase_pong_total = kawase_pong_total
                        .saturating_add(App::compute_texture_bytes(config.format, w, h, 1, 1));
                }
                let ssao_ping_texture_loc = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("SSAO Ping Texture"),
                    size: wgpu::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let ssao_ping_view_loc =
                    ssao_ping_texture_loc.create_view(&wgpu::TextureViewDescriptor::default());
                let ssao_pong_texture_loc = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("SSAO Pong Texture"),
                    size: wgpu::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let ssao_pong_view_loc =
                    ssao_pong_texture_loc.create_view(&wgpu::TextureViewDescriptor::default());
                let ssao_bytes =
                    App::compute_texture_bytes(config.format, config.width, config.height, 1, 1);

                // Collect locals for assignment outside the scope
                (
                    color_texture_loc,
                    color_view_loc,
                    depth_texture_loc,
                    depth_view_loc,
                    post_color_texture_loc,
                    post_color_view_loc,
                    dof_color_texture_loc,
                    dof_color_view_loc,
                    bloom_ping_texture_loc,
                    bloom_ping_view_loc,
                    bloom_pong_texture_loc,
                    bloom_pong_view_loc,
                    kawase_ping_textures_loc,
                    kawase_ping_views_loc,
                    kawase_pong_textures_loc,
                    kawase_pong_views_loc,
                    kawase_level_sizes_loc,
                    ssao_ping_texture_loc,
                    ssao_ping_view_loc,
                    ssao_pong_texture_loc,
                    ssao_pong_view_loc,
                    offscreen_color_bytes,
                    depth_bytes,
                    post_color_bytes,
                    dof_color_bytes,
                    bloom_ping_bytes,
                    bloom_pong_bytes,
                    kawase_ping_total,
                    kawase_pong_total,
                    ssao_bytes,
                )
            };

        // Now that device/config borrows are dropped, update self with textures and tallies
        App::replace_texture_bytes_static(
            &mut self.offscreen_color_texture_bytes,
            offscreen_color_bytes,
            &mut self.gpu_texture_bytes,
        );
        self.offscreen_color_view = Some(color_view);
        self.offscreen_color_texture = Some(color_texture);
        // Assign and track other created textures
        App::replace_texture_bytes_static(
            &mut self.depth_texture_bytes,
            depth_bytes,
            &mut self.gpu_texture_bytes,
        );
        self.offscreen_depth_view = Some(depth_view);
        self.offscreen_depth_texture = Some(depth_texture);
        App::replace_texture_bytes_static(
            &mut self.post_color_texture_bytes,
            post_color_bytes,
            &mut self.gpu_texture_bytes,
        );
        self.post_color_view = Some(post_color_view);
        self.post_color_texture = Some(post_color_texture);
        App::replace_texture_bytes_static(
            &mut self.dof_color_texture_bytes,
            dof_color_bytes,
            &mut self.gpu_texture_bytes,
        );
        self.dof_color_view = Some(dof_color_view);
        self.dof_color_texture = Some(dof_color_texture);
        App::replace_texture_bytes_static(
            &mut self.bloom_ping_bytes,
            bloom_ping_bytes,
            &mut self.gpu_texture_bytes,
        );
        App::replace_texture_bytes_static(
            &mut self.bloom_pong_bytes,
            bloom_pong_bytes,
            &mut self.gpu_texture_bytes,
        );

        // Kawase chain (filled from earlier locals)
        self.kawase_ping_textures = kawase_ping_textures_local;
        self.kawase_ping_views = kawase_ping_views_local;
        self.kawase_pong_textures = kawase_pong_textures_local;
        self.kawase_pong_views = kawase_pong_views_local;
        self.kawase_level_sizes = kawase_level_sizes_local;
        let new_kawase_ping_total: u64 = kawase_ping_total_loc;
        let new_kawase_pong_total: u64 = kawase_pong_total_loc;
        App::replace_texture_bytes_static(
            &mut self.kawase_ping_bytes,
            new_kawase_ping_total,
            &mut self.gpu_texture_bytes,
        );
        App::replace_texture_bytes_static(
            &mut self.kawase_pong_bytes,
            new_kawase_pong_total,
            &mut self.gpu_texture_bytes,
        );
        self.bloom_ping_view = Some(bloom_ping_view);
        self.bloom_ping_texture = Some(bloom_ping_texture);
        self.bloom_pong_view = Some(bloom_pong_view);
        self.bloom_pong_texture = Some(bloom_pong_texture);

        // SSAO ping/pong textures assigned from locals
        App::replace_texture_bytes_static(
            &mut self.ssao_ping_bytes,
            ssao_bytes,
            &mut self.gpu_texture_bytes,
        );
        App::replace_texture_bytes_static(
            &mut self.ssao_pong_bytes,
            ssao_bytes,
            &mut self.gpu_texture_bytes,
        );
        self.ssao_ping_texture = Some(ssao_ping_texture);
        self.ssao_ping_view = Some(ssao_ping_view);
        self.ssao_pong_texture = Some(ssao_pong_texture);
        self.ssao_pong_view = Some(ssao_pong_view);
        // HZB will be created after closure to avoid borrow conflicts

        self.update_dof_bind_group();
        // Combine bind group depends on DoF color and CoC buffers
        self.update_dof_combine_bind_group();
        // Kawase registrations (UBOs/BindGroups per level)
        self.update_kawase_bind_groups();
        self.update_bloom_uniforms();
        self.update_bloom_bind_groups();

        // Create HZB texture (after we dropped the device/config borrow from the local closure)
        if let Some(device) = self.device.as_ref() {
            let hzb_bytes: u64;
            let mut hzb_mips: u32 = 1;
            let mut hzb_texture_opt: Option<wgpu::Texture> = None;
            let mut hzb_view_opt: Option<wgpu::TextureView> = None;
            let mut hzb_mip_views_local: Vec<wgpu::TextureView> = Vec::new();

            if self.hzb_enabled {
                if let Some(cfg) = self.config.as_ref() {
                    let max_dim = cfg.width.max(cfg.height);
                    let mip_levels = 32 - (max_dim.saturating_sub(1)).leading_zeros();
                    let tex = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("HZB Texture"),
                        size: wgpu::Extent3d {
                            width: cfg.width,
                            height: cfg.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: mip_levels,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R32Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::STORAGE_BINDING
                            | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

                    // Create per-mip views for storage binding
                    for mip in 0..mip_levels {
                        let mip_view = tex.create_view(&wgpu::TextureViewDescriptor {
                            label: Some(&format!("HZB Mip {} View", mip)),
                            format: Some(wgpu::TextureFormat::R32Float),
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: mip,
                            mip_level_count: Some(1),
                            base_array_layer: 0,
                            array_layer_count: Some(1),
                            usage: None,
                        });
                        hzb_mip_views_local.push(mip_view);
                    }

                    hzb_texture_opt = Some(tex);
                    hzb_view_opt = Some(view);
                    hzb_mips = mip_levels;
                    hzb_bytes = App::compute_texture_bytes(
                        wgpu::TextureFormat::R32Float,
                        cfg.width,
                        cfg.height,
                        mip_levels,
                        1,
                    );
                } else {
                    // Should not happen; fallback to 1x1 dummy
                    let tex = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("HZB Dummy Texture"),
                        size: wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R32Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::STORAGE_BINDING
                            | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                    hzb_mip_views_local.push(view.clone());
                    hzb_texture_opt = Some(tex);
                    hzb_view_opt = Some(view);
                    hzb_mips = 1;
                    hzb_bytes =
                        App::compute_texture_bytes(wgpu::TextureFormat::R32Float, 1, 1, 1, 1);
                }
            } else {
                // Create a dummy 1x1 R32 texture
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("HZB Dummy Texture"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                hzb_mip_views_local.push(view.clone());
                hzb_texture_opt = Some(tex);
                hzb_view_opt = Some(view);
                hzb_mips = 1;
                hzb_bytes = App::compute_texture_bytes(wgpu::TextureFormat::R32Float, 1, 1, 1, 1);
            }
            App::replace_texture_bytes_static(
                &mut self.hzb_texture_bytes,
                hzb_bytes,
                &mut self.gpu_texture_bytes,
            );
            self.hzb_texture = hzb_texture_opt;
            self.hzb_view = hzb_view_opt;
            self.hzb_mip_views = hzb_mip_views_local;
            self.hzb_mip_levels = hzb_mips;

            // Create HZB Generation bind groups
            if let Some(gen_layout) = self.hzb_gen_bind_group_layout.as_ref() {
                if let Some(cfg) = self.config.as_ref() {
                    let params = HzbParams {
                        width: cfg.width,
                        height: cfg.height,
                        src_mip: 0,
                        dst_mip: 0,
                    };
                    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("HZB Params Buffer"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                    let params_size = std::mem::size_of::<HzbParams>() as u64;
                    App::replace_buffer_bytes_static(
                        &mut self.hzb_params_buffer_bytes,
                        params_size,
                        &mut self.gpu_buffer_bytes,
                    );
                    self.hzb_params_buffer = Some(params_buf.clone());

                    // Create copy bind group (depth -> mip 0)
                    if let (Some(depth_view), Some(mip0_view)) = (
                        self.offscreen_depth_view.as_ref(),
                        self.hzb_mip_views.get(0),
                    ) {
                        let copy_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("HZB Copy Bind Group"),
                            layout: gen_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(depth_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(mip0_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: params_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(depth_view), // Unused in copy
                                },
                            ],
                        });
                        self.hzb_copy_bind_group = Some(copy_bg);
                    }

                    // Create downsample bind groups (mip N-1 -> mip N)
                    let mut downsample_bgs = Vec::new();
                    if let Some(hzb_view) = self.hzb_view.as_ref() {
                        for dst_mip in 1..(self.hzb_mip_levels as usize) {
                            if let Some(dst_view) = self.hzb_mip_views.get(dst_mip) {
                                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some(&format!(
                                        "HZB Downsample Bind Group Mip {}",
                                        dst_mip
                                    )),
                                    layout: gen_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(hzb_view), // Unused in downsample
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::TextureView(dst_view),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: params_buf.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 3,
                                            resource: wgpu::BindingResource::TextureView(hzb_view), // Source texture
                                        },
                                    ],
                                });
                                downsample_bgs.push(bg);
                            }
                        }
                    }
                    self.hzb_downsample_bind_groups = downsample_bgs;
                }
            }

            // Build HZB cull bind group (for GPU culling shader)
            if let Some(layout) = self.hzb_bind_group_layout.as_ref() {
                if let (Some(depth_view), Some(hzb_view), Some(hzb_params_buf)) = (
                    self.offscreen_depth_view.as_ref(),
                    self.hzb_view.as_ref(),
                    self.hzb_params_buffer.as_ref(),
                ) {
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("HZB Cull Bind Group"),
                        layout: layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(hzb_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: hzb_params_buf.as_entire_binding(),
                            },
                        ],
                    });
                    self.hzb_bind_groups = vec![Some(bg)];
                }
            }
        }
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
        // Track shadow map texture bytes
        let shadow_bytes = App::compute_texture_bytes(
            wgpu::TextureFormat::Depth32Float,
            self.shadow_map_size,
            self.shadow_map_size,
            1,
            1,
        );
        App::replace_texture_bytes_static(
            &mut self.shadow_map_bytes,
            shadow_bytes,
            &mut self.gpu_texture_bytes,
        );
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

    fn update_dof_combine_bind_group(&mut self) {
        let (Some(device), Some(layout), Some(dof_color_view), Some(source_view), Some(sampler)) = (
            self.device.as_ref(),
            self.dof_combine_bind_group_layout.as_ref(),
            self.dof_color_view.as_ref(),
            self.offscreen_color_view.as_ref(),
            self.post_sampler.as_ref(),
        ) else {
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
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        }));
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

        if let Some(buffer) = self.ssao_blur_horizontal_uniform_buffer.as_ref() {
            let ssao_width = (config.width / 2).max(1);
            let ssao_height = (config.height / 2).max(1);
            let data = self.build_ssao_blur_uniforms(ssao_width, ssao_height, [1.0, 0.0]);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[data]));
        }

        if let Some(buffer) = self.ssao_blur_vertical_uniform_buffer.as_ref() {
            let ssao_width = (config.width / 2).max(1);
            let ssao_height = (config.height / 2).max(1);
            let data = self.build_ssao_blur_uniforms(ssao_width, ssao_height, [0.0, 1.0]);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[data]));
        }

        if let Some(buffer) = self.composite_uniform_buffer.as_ref() {
            let data = self.build_composite_uniforms();
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[data]));
        }

        // SSILVB uniforms
        if let Some(buffer) = self.ssilvb_uniform_buffer.as_ref() {
            let data = self.build_ssilvb_uniforms(config.width, config.height);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[data]));
        }
    }

    fn update_bloom_bind_groups(&mut self) {
        if self.offscreen_color_view.is_none()
            || self.post_color_view.is_none()
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

        let Some(offscreen_view) = self.offscreen_color_view.as_ref() else {
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
                            resource: wgpu::BindingResource::TextureView(offscreen_view),
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

            // SSAO blur horizontal bind group (use bloom blur pipeline)
            if self.ssao_settings.blur_enabled {
                if let (Some(ssao_h_ubo), Some(ssao_ping_view), Some(psampler)) = (
                    self.ssao_blur_horizontal_uniform_buffer.as_ref(),
                    self.ssao_ping_view.as_ref(),
                    self.post_sampler.as_ref(),
                ) {
                    self.ssao_blur_horizontal_bind_group =
                        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("SSAO Blur Horizontal Bind Group"),
                            layout: blur_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: ssao_h_ubo.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(ssao_ping_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(psampler),
                                },
                            ],
                        }));
                }
            }

            // SSAO bind group uses uniform 0, offscreen depth (1), and post sampler (2)
            if let (Some(ssao_ubo), Some(depth_view), Some(psampler)) = (
                self.ssilvb_uniform_buffer.as_ref(),
                self.offscreen_depth_view.as_ref(),
                self.post_sampler.as_ref(),
            ) {
                self.ssilvb_bind_group =
                    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("SSILVB Bind Group"),
                        layout: self.ssilvb_bind_group_layout.as_ref().unwrap(),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: ssao_ubo.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(psampler),
                            },
                        ],
                    }));
            }

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

            // SSAO blur vertical bind group (reads from SSAO Pong after horizontal)
            if self.ssao_settings.blur_enabled {
                if let (Some(ssao_v_ubo), Some(ssao_pong_view), Some(psampler)) = (
                    self.ssao_blur_vertical_uniform_buffer.as_ref(),
                    self.ssao_pong_view.as_ref(),
                    self.post_sampler.as_ref(),
                ) {
                    self.ssao_blur_vertical_bind_group =
                        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("SSAO Blur Vertical Bind Group"),
                            layout: blur_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: ssao_v_ubo.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(ssao_pong_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(psampler),
                                },
                            ],
                        }));
                }
            }
        }

        if let (Some(composite_ubo), Some(sampler), Some(ssao_ping_view)) = (
            self.composite_uniform_buffer.as_ref(),
            self.post_sampler.as_ref(),
            self.ssao_ping_view.as_ref(),
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
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(ssao_ping_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                }));
        }
    }

    // Readbacks are disabled — use the debug overlay for immediate inspection

    fn update_kawase_bind_groups(&mut self) {
        let Some(device) = self.device.as_ref() else {
            return;
        };

        self.kawase_down_bind_groups.clear();
        self.kawase_up_bind_groups.clear();
        self.kawase_uniform_buffers.clear();

        let iterations = self.dof_settings.kawase_iterations.min(6).max(1);
        for level in 0..iterations {
            // Create uniform buffer for this level (texel size updated every frame)
            // Determine initial texel_size from kawase_level_sizes if available
            let (texel_x, texel_y) = if let Some((w, h)) = self.kawase_level_sizes.get(level) {
                (1.0 / (*w) as f32, 1.0 / (*h) as f32)
            } else {
                (0.0_f32, 0.0_f32)
            };
            let offset = self.dof_settings.kawase_offset
                * (level as f32 + 1.0)
                * self.dof_settings.blur_strength;
            let ubo_data = [texel_x, texel_y, offset, 0.0f32];
            let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Kawase Uniform L{}", level)),
                contents: bytemuck::cast_slice(&ubo_data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            // Track Kawase UBO bytes (4 floats)
            let kawase_ubo_bytes = std::mem::size_of::<[f32; 4]>() as u64;
            App::replace_buffer_bytes_static(
                &mut self.uniform_buffer_bytes,
                kawase_ubo_bytes,
                &mut self.gpu_buffer_bytes,
            );

            self.kawase_uniform_buffers.push(Some(ubo));
            // Create empty placeholders for bind groups; we'll populate them now if we have the required resources
            self.kawase_down_bind_groups.push(None);
            self.kawase_up_bind_groups.push(None);
            // initialize last_ubo data vector
            self.kawase_last_ubo.push(ubo_data);
        }

        // Create/update the bind groups now that we have UBOs and textures created
        let Some(layout) = self.kawase_bind_group_layout.as_ref() else {
            return;
        };
        let Some(sampler) = self.post_sampler.as_ref() else {
            return;
        };
        // DoF color view is the initial input for level 0 down pass; ping views provide subsequent levels
        let dof_input_view = self.dof_color_view.as_ref();

        for level in 0..iterations {
            // For down passes: input is dof_color_view for level 0, otherwise ping_views[level-1]
            let input_view = if level == 0 {
                if let Some(view) = dof_input_view {
                    view
                } else {
                    continue;
                }
            } else {
                if let Some(Some(view)) = self.kawase_ping_views.get(level - 1) {
                    view
                } else {
                    continue;
                }
            };
            let ubo_ref = self.kawase_uniform_buffers[level].as_ref().unwrap();
            let down_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Kawase Down BG L{}", level)),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubo_ref.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });

            // For up passes: input is ping_views[level]
            let up_input_view = if let Some(Some(view)) = self.kawase_ping_views.get(level) {
                view
            } else {
                continue;
            };
            let up_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Kawase Up BG L{}", level)),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubo_ref.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(up_input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });

            self.kawase_down_bind_groups[level] = Some(down_bg);
            self.kawase_up_bind_groups[level] = Some(up_bg);
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
            // Track GPU input buffer bytes
            let gpu_input_bytes =
                (self.gpu_input_capacity * std::mem::size_of::<GpuInstanceInput>()) as u64;
            App::replace_buffer_bytes_static(
                &mut self.gpu_input_buffer_bytes,
                gpu_input_bytes,
                &mut self.gpu_buffer_bytes,
            );

            // Create Mesh Indirect Buffer
            self.mesh_indirect_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mesh Indirect Buffer"),
                size: (needed_capacity * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>())
                    as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            App::replace_buffer_bytes_static(
                &mut self.mesh_indirect_bytes,
                (needed_capacity * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>())
                    as u64,
                &mut self.gpu_buffer_bytes,
            );

            // Create Envelope Indirect Buffer
            self.envelope_indirect_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Envelope Indirect Buffer"),
                size: (needed_capacity * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>())
                    as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            App::replace_buffer_bytes_static(
                &mut self.envelope_indirect_bytes,
                (needed_capacity * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>())
                    as u64,
                &mut self.gpu_buffer_bytes,
            );

            // Create Fallback Instance Buffer
            self.fallback_instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Fallback Instance Buffer"),
                size: (needed_capacity * std::mem::size_of::<VoxelInstanceRaw>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            App::replace_buffer_bytes_static(
                &mut self.fallback_instance_buffer_bytes,
                (needed_capacity * std::mem::size_of::<VoxelInstanceRaw>()) as u64,
                &mut self.gpu_buffer_bytes,
            );
            self.fallback_instance_capacity = needed_capacity;

            self.cull_bind_group = None; // Force rebuild with new buffer
        }

        // Fallback Indirect Args Buffer (fixed size)
        if self.fallback_indirect_buffer.is_none() {
            self.fallback_indirect_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Fallback Indirect Args Buffer"),
                size: std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            App::replace_buffer_bytes_static(
                &mut self.fallback_indirect_bytes,
                std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64,
                &mut self.gpu_buffer_bytes,
            );
        }
    }

    fn ensure_cull_bind_group(&mut self, device: &wgpu::Device) {
        if self.cull_bind_group.is_some() {
            return;
        }

        let (
            Some(layout),
            Some(input_buffer),
            Some(params_buffer),
            Some(mesh_indirect),
            Some(fallback_indirect),
            Some(fallback_instances),
            Some(envelope_indirect),
            Some(hzb_view),
        ) = (
            self.cull_bind_group_layout.as_ref(),
            self.gpu_input_buffer.as_ref(),
            self.cull_params_buffer.as_ref(),
            self.mesh_indirect_buffer.as_ref(),
            self.fallback_indirect_buffer.as_ref(),
            self.fallback_instance_buffer.as_ref(),
            self.envelope_indirect_buffer.as_ref(),
            self.hzb_view.as_ref(),
        )
        else {
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: mesh_indirect,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: fallback_indirect,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: fallback_instances,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: envelope_indirect,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(hzb_view),
                },
            ],
        }));
    }

    fn mesh_cache_byte_budget(&self) -> u64 {
        self.mesh_cache_budget_bytes
    }

    fn evict_envelope_mesh_cache(&mut self) {
        let budget = self.envelope_mesh_cache_budget_bytes;
        if self.envelope_mesh_cache_bytes <= budget {
            return;
        }

        let mut entries: Vec<_> = self
            .envelope_mesh_cache
            .iter()
            .map(|(key, entry)| (*key, entry.last_used_frame))
            .collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1));

        let mut freed_bytes = 0u64;
        let mut evicted = 0usize;

        for (key, _) in entries {
            if self.envelope_mesh_cache_bytes <= budget {
                break;
            }

            if let Some(entry) = self.envelope_mesh_cache.remove(&key) {
                let entry_bytes = entry.total_bytes();
                // If this was a placeholder (shared empty buffers), do not destroy the buffer
                // as it is owned globally. Only destroy buffers for normal entries.
                if !entry.is_placeholder {
                    // Free regions in allocators
                    self.vertex_allocator
                        .free(entry.vertex_offset, entry.vertex_bytes);
                    self.index_allocator
                        .free(entry.index_offset, entry.index_bytes);
                }
                self.envelope_mesh_cache_bytes =
                    self.envelope_mesh_cache_bytes.saturating_sub(entry_bytes);
                freed_bytes += entry_bytes;
                evicted += 1;
            }
        }

        if evicted > 0 {
            println!(
                "Evicted {} envelope meshes, freed {:.1} MB (current usage {:.1} MB)",
                evicted,
                freed_bytes as f64 / 1024.0 / 1024.0,
                self.envelope_mesh_cache_bytes as f64 / 1024.0 / 1024.0
            );
        }
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
                // If this was a placeholder (shared empty buffers), do not destroy the buffer
                // as it is owned globally. Only destroy buffers for normal entries.
                if !entry.is_placeholder {
                    // Free regions in allocators
                    self.vertex_allocator
                        .free(entry.vertex_offset, entry.vertex_bytes);
                    self.index_allocator
                        .free(entry.index_offset, entry.index_bytes);
                    self.mesh_cache_bytes = self.mesh_cache_bytes.saturating_sub(entry_bytes);
                }
                self.chunk_emitters.remove(&key);
                // Also drop any cached Arc<Chunk> snapshot for this chunk to free memory
                self.mesh_chunk_arc_cache.remove(&key);
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

    fn init_skybox(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        main_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        // Load HDR image
        let hdr_path = "worlds/skybox.hdr";
        // Load HDR image using image crate
        let hdr_image = match image::open(hdr_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Failed to load skybox.hdr: {:?}", e);
                return;
            }
        };

        let width = hdr_image.width();
        let height = hdr_image.height();
        let rgb_image = hdr_image.to_rgb32f();

        // Convert RGB F32 to RGBA F32
        let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
        for pixel in rgb_image.pixels() {
            rgba_data.push(pixel.0[0]);
            rgba_data.push(pixel.0[1]);
            rgba_data.push(pixel.0[2]);
            rgba_data.push(1.0); // Alpha
        }

        // Create texture
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Skybox Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Write texture data using queue
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16 * width), // 4 channels * 4 bytes per f32
                rows_per_image: Some(height),
            },
            texture_size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Skybox Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skybox Bind Group Layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skybox Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/skybox.wgsl").into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[main_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.skybox_texture = Some(texture);
        self.skybox_view = Some(view);
        self.skybox_sampler = Some(sampler);
        self.skybox_bind_group = Some(bind_group);
        self.skybox_bind_group_layout = Some(bind_group_layout);
        self.skybox_pipeline = Some(pipeline);
        // Record and track skybox texture size
        let skybox_bytes = (width as u64) * (height as u64) * 16; // RGBA32F = 16 bytes/pixel
        App::replace_texture_bytes_static(
            &mut self.skybox_texture_bytes,
            skybox_bytes,
            &mut self.gpu_texture_bytes,
        );
    }

    // Allocate or reuse a vertex buffer from the pool; returns (buffer, capacity_bytes)
    #[allow(dead_code)]
    fn allocate_vertex_buffer_from_pool(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        vb_data: &[MeshVertexRaw],
    ) -> (wgpu::Buffer, u64) {
        let required_bytes = (vb_data.len() * std::mem::size_of::<MeshVertexRaw>()) as u64;
        // Create a new buffer sized exactly to required bytes.
        let capacity = required_bytes;
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk Mesh Vertex Buffer (temp)"),
            contents: bytemuck::cast_slice(vb_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        (vbuf, capacity)
    }

    // Allocate or reuse an index buffer from the pool; returns (buffer, capacity_bytes)
    #[allow(dead_code)]
    fn allocate_index_buffer_from_pool(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        idx_data: &[u32],
    ) -> (wgpu::Buffer, u64) {
        let required_bytes = (idx_data.len() * std::mem::size_of::<u32>()) as u64;
        // For now, always allocate a new index buffer
        let capacity = required_bytes;
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk Mesh Index Buffer"),
            contents: bytemuck::cast_slice(idx_data),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        (ibuf, capacity)
    }

    fn allocate_mesh_in_megabuffer(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        vb_data: &[MeshVertexRaw],
        idx_data: &[u32],
    ) -> Result<(u64, u64), voxelot::AllocationError> {
        let vb_bytes = (vb_data.len() * std::mem::size_of::<MeshVertexRaw>()) as u64;
        let ib_bytes = (idx_data.len() * std::mem::size_of::<u32>()) as u64;

        if self.mega_vertex_buffer.is_none() || self.mega_index_buffer.is_none() {
            return Err(voxelot::AllocationError::OutOfMemory);
        }

        let vertex_stride_bytes = std::mem::size_of::<MeshVertexRaw>() as u64;
        let vertex_offset = self
            .vertex_allocator
            .allocate_aligned(vb_bytes, vertex_stride_bytes)?;
        let index_offset = self.index_allocator.allocate_aligned(ib_bytes, 4)?; // u32 indices

        queue.write_buffer(
            self.mega_vertex_buffer.as_ref().unwrap(),
            vertex_offset,
            bytemuck::cast_slice(vb_data),
        );
        queue.write_buffer(
            self.mega_index_buffer.as_ref().unwrap(),
            index_offset,
            bytemuck::cast_slice(idx_data),
        );

        Ok((vertex_offset, index_offset))
    }

    fn run_gpu_culling(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        candidate_count: usize,
        initial_fallback_instances: u32,
    ) {
        if candidate_count == 0 {
            return;
        }

        // Reset fallback indirect buffer (seed instance count = initial_fallback_instances)
        if let Some(buffer) = &self.fallback_indirect_buffer {
            let reset_data = [36u32, initial_fallback_instances, 0, 0];
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&reset_data));
        }

        let (Some(cull_pipeline), Some(cull_bind_group)) =
            (self.cull_pipeline.as_ref(), self.cull_bind_group.as_ref())
        else {
            return;
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Cull Encoder"),
        });

        // Generate HZB mip chain if enabled
        if self.hzb_enabled {
            if let Some(cfg) = self.config.as_ref() {
                // Pass 1: Copy depth -> HZB mip 0
                if let (Some(copy_pipeline), Some(copy_bg), Some(params_buf), Some(queue)) = (
                    self.hzb_gen_copy_pipeline.as_ref(),
                    self.hzb_copy_bind_group.as_ref(),
                    self.hzb_params_buffer.as_ref(),
                    self.queue.as_ref(),
                ) {
                    // Update params for mip 0
                    let params = HzbParams {
                        width: cfg.width,
                        height: cfg.height,
                        src_mip: 0,
                        dst_mip: 0,
                    };
                    queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

                    let mut hzb_copy_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("HZB Copy Pass"),
                            timestamp_writes: None,
                        });
                    hzb_copy_pass.set_pipeline(copy_pipeline);
                    hzb_copy_pass.set_bind_group(0, copy_bg, &[]);
                    let dispatch_x = ((cfg.width + 7) / 8) as u32;
                    let dispatch_y = ((cfg.height + 7) / 8) as u32;
                    hzb_copy_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                    drop(hzb_copy_pass);
                }

                // Pass 2-N: Downsample mip chain
                if let (Some(downsample_pipeline), Some(params_buf), Some(queue)) = (
                    self.hzb_gen_downsample_pipeline.as_ref(),
                    self.hzb_params_buffer.as_ref(),
                    self.queue.as_ref(),
                ) {
                    for dst_mip in 1..(self.hzb_mip_levels as u32) {
                        if let Some(downsample_bg) =
                            self.hzb_downsample_bind_groups.get((dst_mip - 1) as usize)
                        {
                            // Calculate mip dimensions
                            let mip_width = (cfg.width >> dst_mip).max(1);
                            let mip_height = (cfg.height >> dst_mip).max(1);

                            // Update params for this mip
                            let params = HzbParams {
                                width: mip_width,
                                height: mip_height,
                                src_mip: dst_mip - 1,
                                dst_mip,
                            };
                            queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

                            let mut hzb_downsample_pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some(&format!("HZB Downsample Pass Mip {}", dst_mip)),
                                    timestamp_writes: None,
                                });
                            hzb_downsample_pass.set_pipeline(downsample_pipeline);
                            hzb_downsample_pass.set_bind_group(0, downsample_bg, &[]);
                            let dispatch_x = ((mip_width + 7) / 8) as u32;
                            let dispatch_y = ((mip_height + 7) / 8) as u32;
                            hzb_downsample_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                            drop(hzb_downsample_pass);
                        }
                    }
                }
            }
        }

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

        queue.submit(std::iter::once(encoder.finish()));
    }

    fn populate_multi_draw_indirects(&mut self, queue: &wgpu::Queue, visible: &Vec<VoxelInstance>) {
        // Fill mesh and envelope indirect buffers from CPU-visible list
        let mut mesh_args: Vec<wgpu::util::DrawIndexedIndirectArgs> = Vec::new();
        let mut env_args: Vec<wgpu::util::DrawIndexedIndirectArgs> = Vec::new();

        if self.mega_index_buffer.is_none() || self.mega_vertex_buffer.is_none() {
            return;
        }

        let vertex_stride = std::mem::size_of::<MeshVertexRaw>();
        let vertex_buf_size = self.vertex_allocator.total_size();
        let index_buf_size = self.index_allocator.total_size();
        let mut warned = false;
        for v in visible.iter() {
            if !v.is_leaf_chunk {
                continue;
            }
            let key = (v.position[0], v.position[1], v.position[2]);
            // If we don't have mesh/envelope and not in fallback range, log an info
            if cfg!(feature = "viewer-debug") {
                let has_mesh = self.mesh_cache.contains_key(&key);
                let has_envelope = self.envelope_mesh_cache.contains_key(&key);
                if !has_mesh && !has_envelope {
                    // Compute distance squared for fallback decision
                    let cam_pos = self.camera_controller.camera.position;
                    let chunk_center = [key.0 as f32 + 8.0, key.1 as f32 + 8.0, key.2 as f32 + 8.0];
                    let dx = chunk_center[0] - cam_pos[0];
                    let dy = chunk_center[1] - cam_pos[1];
                    let dz = chunk_center[2] - cam_pos[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    let fallback_dist_sq =
                        self.fallback_detail_distance * self.fallback_detail_distance;
                    if dist_sq > fallback_dist_sq {
                        viewer_debug!("MISSING DRAW (no mesh/envelope/fallback): key=({},{},{}) dist={} (> fallback {})", key.0, key.1, key.2, dist_sq, fallback_dist_sq);
                    }
                }
            }
            if let Some(entry) = self.mesh_cache.get_mut(&key) {
                if entry.is_placeholder {
                    // Skip placeholder entries (no geometry) silently
                } else {
                    let idx_end = entry.index_offset + entry.index_bytes;
                    let vb_end = entry.vertex_offset + entry.vertex_bytes;
                    if entry.index_bytes == 0
                        || entry.vertex_bytes == 0
                        || idx_end > index_buf_size
                        || vb_end > vertex_buf_size
                    {
                        if !warned {
                            viewer_debug!("Validation warning on mesh cache entry: index_buf_size={}, vertex_buf_size={} entry: index_offset={}, index_bytes={}, vertex_offset={}, vertex_bytes={}", index_buf_size, vertex_buf_size, entry.index_offset, entry.index_bytes, entry.vertex_offset, entry.vertex_bytes);
                            warned = true;
                        }
                    } else {
                        let first_index = (entry.index_offset / 4) as u32; // u32 indices
                        let base_vertex = (entry.vertex_offset / vertex_stride as u64) as i32;
                        mesh_args.push(wgpu::util::DrawIndexedIndirectArgs {
                            index_count: entry.index_count as u32,
                            instance_count: 1,
                            first_index,
                            base_vertex,
                            first_instance: 0,
                        });
                        // Mark used so eviction won't yoink buffers used this frame
                        entry.last_used_frame = self.frame_index;
                    }
                }
            }
            if let Some(entry) = self.envelope_mesh_cache.get_mut(&key) {
                if entry.is_placeholder {
                    // Skip placeholder entries silently
                } else {
                    let idx_end = entry.index_offset + entry.index_bytes;
                    let vb_end = entry.vertex_offset + entry.vertex_bytes;
                    if entry.index_bytes == 0
                        || entry.vertex_bytes == 0
                        || idx_end > index_buf_size
                        || vb_end > vertex_buf_size
                    {
                        if !warned {
                            viewer_debug!("Validation warning on envelope cache entry: index_buf_size={}, vertex_buf_size={} entry: index_offset={}, index_bytes={}, vertex_offset={}, vertex_bytes={}", index_buf_size, vertex_buf_size, entry.index_offset, entry.index_bytes, entry.vertex_offset, entry.vertex_bytes);
                            warned = true;
                        }
                    } else {
                        let first_index = (entry.index_offset / 4) as u32;
                        let base_vertex = (entry.vertex_offset / vertex_stride as u64) as i32;
                        env_args.push(wgpu::util::DrawIndexedIndirectArgs {
                            index_count: entry.index_count as u32,
                            instance_count: 1,
                            first_index,
                            base_vertex,
                            first_instance: 0,
                        });
                        // Mark used so eviction won't yoink buffers used this frame
                        entry.last_used_frame = self.frame_index;
                    }
                }
            }
        }

        if let Some(buffer) = &self.mesh_indirect_buffer {
            if mesh_args.len() as usize > self.max_draw_capacity {
                viewer_debug!(
                    "Warning: mesh indirect args {} exceed max_draw_capacity {} => truncating",
                    mesh_args.len(),
                    self.max_draw_capacity
                );
                mesh_args.truncate(self.max_draw_capacity);
            }
            if !mesh_args.is_empty() {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&mesh_args));
            } else {
                // zero-length doesn't matter, but clear first 4 bytes
                let zero: [u8; 4] = [0; 4];
                queue.write_buffer(buffer, 0, &zero);
            }
            viewer_debug!("Populated mesh indirects: {} entries", mesh_args.len());
        }
        if let Some(buffer) = &self.envelope_indirect_buffer {
            if env_args.len() as usize > self.max_draw_capacity {
                viewer_debug!(
                    "Warning: envelope indirect args {} exceed max_draw_capacity {} => truncating",
                    env_args.len(),
                    self.max_draw_capacity
                );
                env_args.truncate(self.max_draw_capacity);
            }
            if !env_args.is_empty() {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&env_args));
            } else {
                let zero: [u8; 4] = [0; 4];
                queue.write_buffer(buffer, 0, &zero);
            }
            viewer_debug!("Populated envelope indirects: {} entries", env_args.len());
        }

        // Write counts into multi_draw_count_buffer; offsets: 0=mesh_count, 4=envelope_count
        if let Some(count_buf) = &self.multi_draw_count_buffer {
            let counts = [mesh_args.len() as u32, env_args.len() as u32];
            queue.write_buffer(count_buf, 0, bytemuck::cast_slice(&counts));
        }
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

        // Request device with increased limits
        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = 1_073_741_824; // 1 GB (up from 256 MB default)
        limits.max_storage_buffer_binding_size = 536_870_912; // 512 MB (up from 128 MB default)

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: wgpu::Trace::Off,
            })
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

        // Prefer low-latency present modes when available. `Mailbox` is ideal
        // (low latency + no tearing) but not available on all platforms/drivers.
        // Fall back to `Immediate` if available (may allow tearing), otherwise
        // use `Fifo` which is guaranteed to be supported (vsync).
        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            wgpu::PresentMode::Mailbox
        } else if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Immediate)
        {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
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
                            2 => Float32x3,  // scale
                            3 => Float32,    // ao_factor
                            7 => Float32x4,  // custom_color (RGBA) -- moved to avoid vertex attribute conflict
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
                            2 => Float32x3,
                            3 => Float32,
                            7 => Float32x4,
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
        // Track DoF uniform buffer bytes
        let dof_ubo_bytes = std::mem::size_of::<[f32; DOF_UNIFORM_FLOATS]>() as u64;
        App::replace_buffer_bytes_static(
            &mut self.uniform_buffer_bytes,
            dof_ubo_bytes,
            &mut self.gpu_buffer_bytes,
        );

        let dof_combine_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Combine Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dof_combine.wgsl").into()),
        });

        // Fused DoF bind group layout: uniform, source color, depth, sampler.
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

        // Removed separate CoC bind group layout (CoC computed in fused blur pass).

        // Combine now uses fused blurred texture (with CoC in alpha) + source color + sampler.
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

        let dof_combine_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DoF Combine Pipeline Layout"),
                bind_group_layouts: &[&dof_combine_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Removed separate CoC pipeline (fused into blur pass).
        // Fused DoF combine pipeline (uses blurred texture alpha for CoC)
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

        // Kawase (Dual Kawase) pipelines for down/upsample passes
        let kawase_down_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Kawase Down Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/dual_kawase_down.wgsl").into(),
            ),
        });
        let kawase_up_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Kawase Up Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/dual_kawase_up.wgsl").into(),
            ),
        });

        let kawase_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Kawase Bind Group Layout"),
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

        let kawase_down_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Kawase Down Pipeline Layout"),
                bind_group_layouts: &[&kawase_bind_group_layout],
                push_constant_ranges: &[],
            });
        let kawase_up_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Kawase Up Pipeline Layout"),
                bind_group_layouts: &[&kawase_bind_group_layout],
                push_constant_ranges: &[],
            });

        let kawase_down_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Kawase Down Pipeline"),
            layout: Some(&kawase_down_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &kawase_down_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &kawase_down_shader,
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

        let kawase_up_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Kawase Up Pipeline"),
            layout: Some(&kawase_up_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &kawase_up_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &kawase_up_shader,
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

        self.kawase_down_pipeline = Some(kawase_down_pipeline);
        self.kawase_up_pipeline = Some(kawase_up_pipeline);
        self.kawase_bind_group_layout = Some(kawase_bind_group_layout);

        let post_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DoF Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bloom_extract_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Extract Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/bloom_extract.wgsl").into(),
            ),
        });

        let ssilvb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSILVB / SSAO Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ssilvb.wgsl").into()),
        });

        let bloom_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bloom_blur.wgsl").into()),
        });

        let ssao_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAO Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ssao_blur.wgsl").into()),
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

        let ssilvb_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSILVB Bind Group Layout"),
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
                        binding: 4,
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

        let ssao_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Blur Pipeline"),
            layout: Some(&bloom_blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ssao_blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_blur_shader,
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
            shadow_texel_size_pad: [
                shadow_texel,
                shadow_texel,
                self.pcf_radius,
                self.pcf_poisson_samples as f32,
            ],
            shadow_darkness_pad: [self.shadow_darkness, self.shadow_backface_scale, 0.0, 0.0],
            moon_direction_intensity: [-0.5, -1.0, -0.3, 0.2], // initial opposite dim moon
            moon_color_pad: [0.2, 0.25, 0.35, 0.0],
            light_probe_count: 0,
            lod_distance: 800.0,
            envelope_distance: 256.0,
            envelope_fade_range: 32.0,
            inverse_view: [[0.0; 4]; 4],
            inverse_proj: [[0.0; 4]; 4],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // SSAO blur uniforms (half-resolution like SSao textures)
        let ssao_width = (config.width / 2).max(1);
        let ssao_height = (config.height / 2).max(1);
        let ssao_blur_horizontal_uniforms =
            self.build_ssao_blur_uniforms(ssao_width, ssao_height, [1.0, 0.0]);
        let ssao_blur_horizontal_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SSAO Blur Horizontal Uniform Buffer"),
                contents: bytemuck::cast_slice(&[ssao_blur_horizontal_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        // Track SSAO blur horizontal uniform buffer bytes
        let ssao_blur_bytes = std::mem::size_of::<SsaoUniformsRaw>() as u64;
        App::replace_buffer_bytes_static(
            &mut self.uniform_buffer_bytes,
            ssao_blur_bytes,
            &mut self.gpu_buffer_bytes,
        );

        let ssao_blur_vertical_uniforms =
            self.build_ssao_blur_uniforms(ssao_width, ssao_height, [0.0, 1.0]);
        let ssao_blur_vertical_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SSAO Blur Vertical Uniform Buffer"),
                contents: bytemuck::cast_slice(&[ssao_blur_vertical_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        App::replace_buffer_bytes_static(
            &mut self.uniform_buffer_bytes,
            ssao_blur_bytes,
            &mut self.gpu_buffer_bytes,
        );

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
        // Track light probe buffer bytes
        let probes_size = (light_probe_capacity * std::mem::size_of::<LightProbe>()) as u64;
        App::replace_buffer_bytes_static(
            &mut self.light_probe_buffer_bytes,
            probes_size,
            &mut self.gpu_buffer_bytes,
        );

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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Optional HZB texture for GPU cull tests
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
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

        // HZB generation compute pipelines
        let hzb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HZB Generation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/hzb_gen.wgsl").into()),
        });

        // Bind group layout for copy pass (depth -> hzb mip 0)
        let hzb_gen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("HZB Gen Bind Group Layout"),
                entries: &[
                    // Binding 0: depth texture (copy source)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Binding 1: hzb storage texture (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Binding 2: params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: hzb source texture (for downsample)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let hzb_gen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("HZB Gen Pipeline Layout"),
                bind_group_layouts: &[&hzb_gen_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Copy pipeline (depth -> mip 0)
        let hzb_copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HZB Copy Pipeline"),
            layout: Some(&hzb_gen_pipeline_layout),
            module: &hzb_shader,
            entry_point: Some("copy_depth"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Downsample pipeline (mip N-1 -> mip N)
        let hzb_downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("HZB Downsample Pipeline"),
                layout: Some(&hzb_gen_pipeline_layout),
                module: &hzb_shader,
                entry_point: Some("downsample"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.hzb_gen_bind_group_layout = Some(hzb_gen_bind_group_layout);
        self.hzb_gen_copy_pipeline = Some(hzb_copy_pipeline);
        self.hzb_gen_downsample_pipeline = Some(hzb_downsample_pipeline);

        let cull_params_init = GpuCullParams {
            camera_position: [0.0; 3],
            candidate_count: 0,
            camera_forward: [0.0, 0.0, -1.0],
            _pad0: 0,
            near_plane: 0.1,
            far_plane: 1000.0,
            _pad_align1: [0; 2],
            camera_right: [1.0, 0.0, 0.0],
            _pad_r0: 0,
            camera_up: [0.0, 1.0, 0.0],
            _pad_u0: 0,
            fov_tan: 1.0,
            aspect: 1.0,
            screen_width: 1280.0,
            screen_height: 720.0,
            lod_render_distance: 1000.0,
            detail_cull_distance: 100.0,
            envelope_distance: 1000.0,
            hzb_enabled: 0,
            max_hzb_mip: 0,
            _pad3: 0.0,
            _pad_align2: [0; 2],
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };

        let cull_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU Cull Params Buffer"),
            contents: bytemuck::bytes_of(&cull_params_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Track cull params buffer bytes
        let cull_params_size = std::mem::size_of::<GpuCullParams>() as u64;
        App::replace_buffer_bytes_static(
            &mut self.cull_params_buffer_bytes,
            cull_params_size,
            &mut self.gpu_buffer_bytes,
        );

        // Create cube vertex buffer with positions and normals
        let cube_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // Track cube vertex buffer bytes (small)
        let cube_vertex_bytes = (CUBE_VERTICES.len() * std::mem::size_of::<MeshVertexRaw>()) as u64;
        App::replace_buffer_bytes_static(
            &mut self.cube_vertex_buffer_bytes,
            cube_vertex_bytes,
            &mut self.gpu_buffer_bytes,
        );

        // Create Mega Buffers and multi-draw buffers
        let mega_vertex_size = self.vertex_allocator.total_size();
        let mega_index_size = self.index_allocator.total_size();
        let mega_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mega Vertex Buffer"),
            size: mega_vertex_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Track mega buffer sizes
        App::replace_buffer_bytes_static(
            &mut self.mega_vertex_buffer_bytes,
            mega_vertex_size,
            &mut self.gpu_buffer_bytes,
        );
        let mega_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mega Index Buffer"),
            size: mega_index_size,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        App::replace_buffer_bytes_static(
            &mut self.mega_index_buffer_bytes,
            mega_index_size,
            &mut self.gpu_buffer_bytes,
        );

        // Multi-draw indirect buffers
        let indirect_entry_size = std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;
        let multi_draw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Multi Draw Indirect Buffer"),
            size: (self.max_draw_capacity as u64) * indirect_entry_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Track multi-draw indirect buffer
        App::replace_buffer_bytes_static(
            &mut self.multi_draw_indirect_bytes,
            (self.max_draw_capacity as u64)
                * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64,
            &mut self.gpu_buffer_bytes,
        );
        let multi_draw_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Multi Draw Count Buffer"),
            // store two u32 counts (mesh count @ offset 0, envelope count @ offset 4)
            size: (std::mem::size_of::<u32>() * 2) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        App::replace_buffer_bytes_static(
            &mut self.multi_draw_count_bytes,
            (std::mem::size_of::<u32>() * 2) as u64,
            &mut self.gpu_buffer_bytes,
        );

        self.cull_pipeline = Some(cull_pipeline);
        self.cull_bind_group_layout = Some(cull_bind_group_layout);
        self.cull_params_buffer = Some(cull_params_buffer);
        self.cull_bind_group = None;
        self.mega_vertex_buffer = Some(mega_vertex_buffer);
        self.mega_index_buffer = Some(mega_index_buffer);
        self.multi_draw_indirect_buffer = Some(multi_draw_buf);
        self.multi_draw_count_buffer = Some(multi_draw_count_buf);

        // fused DoF pipeline removed; use CoC copy + Kawase instead
        // DoF CoC copy pipeline (if Kawase is enabled, we use this cheap pass to produce CoC alpha + base color as input)
        let dof_coc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF CoC Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/dof_coc_copy.wgsl").into(),
            ),
        });
        let dof_coc_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF CoC Copy Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("DoF CoC Pipeline Layout"),
                    bind_group_layouts: &[&dof_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
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
        self.dof_coc_pipeline = Some(dof_coc_pipeline);
        self.dof_bind_group_layout = Some(dof_bind_group_layout);
        self.dof_uniform_buffer = Some(dof_uniform_buffer);
        // Remove separate CoC pipeline/bind group (fused into blur pass)
        // CoC pipeline removed (fused)
        self.dof_combine_pipeline = Some(dof_combine_pipeline);
        self.dof_combine_bind_group_layout = Some(dof_combine_bind_group_layout);
        self.post_sampler = Some(post_sampler);
        self.dof_bind_group = None;
        self.bloom_extract_pipeline = Some(bloom_extract_pipeline);
        self.bloom_blur_pipeline = Some(bloom_blur_pipeline);
        self.ssao_blur_pipeline = Some(ssao_blur_pipeline);
        self.composite_pipeline = Some(composite_pipeline);
        self.bloom_extract_bind_group_layout = Some(bloom_extract_bind_group_layout);
        self.bloom_blur_bind_group_layout = Some(bloom_blur_bind_group_layout);
        self.composite_bind_group_layout = Some(composite_bind_group_layout);
        self.bloom_extract_uniform_buffer = Some(bloom_extract_uniform_buffer);
        self.bloom_blur_horizontal_uniform_buffer = Some(bloom_blur_horizontal_uniform_buffer);
        self.bloom_blur_vertical_uniform_buffer = Some(bloom_blur_vertical_uniform_buffer);
        self.ssao_blur_horizontal_uniform_buffer = Some(ssao_blur_horizontal_uniform_buffer);
        self.ssao_blur_vertical_uniform_buffer = Some(ssao_blur_vertical_uniform_buffer);
        // Don't create SSAO blur bind groups until ping/pong views exist; update later in update_bloom_bind_groups()
        self.composite_uniform_buffer = Some(composite_uniform_buffer);
        // SSILVB uniforms
        let ssilvb_uniforms = self.build_ssilvb_uniforms(config.width, config.height);
        let ssilvb_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSILVB Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ssilvb_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Track ssilvb uniform buffer bytes
        let ssilvb_bytes = std::mem::size_of::<SsaoUniformsRaw>() as u64;
        App::replace_buffer_bytes_static(
            &mut self.uniform_buffer_bytes,
            ssilvb_bytes,
            &mut self.gpu_buffer_bytes,
        );
        self.ssilvb_uniform_buffer = Some(ssilvb_uniform_buffer);
        self.bloom_extract_bind_group = None;
        self.bloom_blur_horizontal_bind_group = None;
        self.bloom_blur_vertical_bind_group = None;
        self.composite_bind_group = None;

        // SSILVB: SSAO pipeline creation
        let ssilvb_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSILVB Pipeline Layout"),
                bind_group_layouts: &[&ssilvb_bind_group_layout],
                push_constant_ranges: &[],
            });

        let ssilvb_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSILVB Pipeline"),
            layout: Some(&ssilvb_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ssilvb_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssilvb_shader,
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

        self.ssilvb_pipeline = Some(ssilvb_pipeline);
        self.ssilvb_bind_group_layout = Some(ssilvb_bind_group_layout);

        self.window = Some(window);

        // Initialize skybox before moving values into self
        self.init_skybox(&device, &queue, &config, &main_bind_group_layout);

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

        // Initialize egui
        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            self.window.as_ref().unwrap(),
            Some(self.window.as_ref().unwrap().scale_factor() as f32),
            None, // theme
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .limits()
                    .max_texture_dimension_2d as usize,
            ),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            self.device.as_ref().unwrap(),
            self.config.as_ref().unwrap().format,
            egui_wgpu::RendererOptions::default(),
        );

        self.egui_ctx = Some(egui_ctx);
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);
        self.cube_vertex_buffer = Some(cube_vertex_buffer);

        self.update_shadow_bind_group();
        self.recreate_shadow_map();

        self.recreate_offscreen_targets();

        viewer_debug!("DEBUG: mesh_pipeline created successfully");
        println!("wgpu initialized");
        println!(
            "DoF Kawase enabled: {} (kawase_iterations={}, kawase_offset={})",
            self.dof_settings.kawase_enabled,
            self.dof_settings.kawase_iterations,
            self.dof_settings.kawase_offset
        );
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

        let mut _voxel_expansion_count = 0;
        let mut cpu_prepopulated_instances: Vec<VoxelInstanceRaw> = Vec::new();
        let mut gpu_inputs: Vec<GpuInstanceInput> = Vec::new();
        for (i, v) in visible.iter().enumerate() {
            let key = (v.position[0], v.position[1], v.position[2]);
            let has_mesh = v.is_leaf_chunk && self.mesh_cache.contains_key(&key);
            let has_envelope = v.is_leaf_chunk && self.envelope_mesh_cache.contains_key(&key);
            let mut cpu_prepop = false;

            let cam_pos = self.camera_controller.camera.position;
            let chunk_center = [key.0 as f32 + 8.0, key.1 as f32 + 8.0, key.2 as f32 + 8.0];
            let dx = chunk_center[0] - cam_pos[0];
            let dy = chunk_center[1] - cam_pos[1];
            let dz = chunk_center[2] - cam_pos[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let _envelope_dist_sq = self.envelope_distance * self.envelope_distance;

            // If very near and un-meshed, decompose into voxels (regardless of envelope)
            let fallback_dist_sq = self.fallback_detail_distance * self.fallback_detail_distance;
            if v.is_leaf_chunk && !has_mesh && dist_sq < fallback_dist_sq {
                // Counting attempts is no longer used, keep metric in case we want to log it
                if let Some(chunk) = self
                    .world
                    .get_leaf_chunk_at_origin(WorldPos::new(key.0, key.1, key.2))
                {
                    let mut voxels_written = 0usize;
                    for ((x, y, z), voxel) in chunk.iter() {
                        if let Voxel::Solid(vtype) = voxel {
                            let (emissive_rgb, emissive_intensity) =
                                self.palette.emissive(*vtype as u32);
                            cpu_prepopulated_instances.push(VoxelInstanceRaw {
                                position: [
                                    (key.0 + x as i64) as f32,
                                    (key.1 + y as i64) as f32,
                                    (key.2 + z as i64) as f32,
                                ],
                                voxel_type: *vtype as u32,
                                scale: [1.0, 1.0, 1.0],
                                ao_factor: 1.0,
                                custom_color: [0.0, 0.0, 0.0, 0.0],
                                emissive: [
                                    emissive_rgb[0],
                                    emissive_rgb[1],
                                    emissive_rgb[2],
                                    emissive_intensity,
                                ],
                            });
                            voxels_written += 1;
                        }
                    }
                    if voxels_written > 0 {
                        _voxel_expansion_count += 1;
                        // Mark the chunk candidate as CPU prepopulated so shader won't append fallback instances
                        let mut flags = 0u32;
                        if has_mesh {
                            flags |= 1;
                        }
                        if has_envelope {
                            flags |= 2;
                        }
                        flags |= 4; // CPU prepopulated
                        let _custom_color_f32 = if let Some(rgba) = v.custom_color {
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
                            self.palette.emissive(v.voxel_type as u32)
                        };
                        gpu_inputs.push(GpuInstanceInput {
                            position: [
                                v.position[0] as f32,
                                v.position[1] as f32,
                                v.position[2] as f32,
                            ],
                            _pad0: 0,
                            scale: v.scale,
                            _pad1: 0,
                            custom_color: if v.custom_color.is_some() {
                                let rgba = v.custom_color.unwrap();
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
                            },
                            emissive: [
                                emissive_rgb[0],
                                emissive_rgb[1],
                                emissive_rgb[2],
                                emissive_intensity,
                            ],
                            voxel_type: v.voxel_type as u32,
                            flags,
                            mesh_index: i as u32,
                            envelope_index: i as u32,
                        });
                        continue;
                    }
                }
            } else if v.is_leaf_chunk && !has_mesh && !has_envelope {
                // Distant leaf chunk without a mesh or envelope: draw a bounding-box cube to avoid holes
                // Look up the chunk to get its tight bounding box
                if let Some(chunk) = self
                    .world
                    .get_leaf_chunk_at_origin(WorldPos::new(key.0, key.1, key.2))
                {
                    let (pos, scale) = if let Some(bbox) = chunk.bounding_box {
                        // Use the standard bbox->world conversion helper, origin at chunk key, scale = 1
                        // For leaf chunk bounding box, each local voxel maps to 1 world unit so
                        // pass scale=16 (unit = scale/16 = 1).
                        let (pos_i64, size) = bbox_local_to_world([key.0, key.1, key.2], 16, bbox);
                        (
                            [pos_i64[0] as f32, pos_i64[1] as f32, pos_i64[2] as f32],
                            size,
                        )
                    } else {
                        // Fallback to full chunk if no bbox (shouldn't happen if voxel_count > 0)
                        (
                            [chunk_center[0], chunk_center[1], chunk_center[2]],
                            [16.0, 16.0, 16.0],
                        )
                    };

                    let (emissive_rgb, emissive_intensity) = if v.custom_color.is_some() {
                        ([0.0, 0.0, 0.0], 0.0)
                    } else {
                        // Use average color or first voxel type?
                        // v.voxel_type might be generic. Let's use chunk.average_color if available?
                        // But chunk doesn't store average color in a public field easily accessible here?
                        // Actually Chunk has average_color.
                        self.palette.emissive(v.voxel_type as u32)
                    };

                    let custom_color_f32 = if let Some(rgba) = v.custom_color {
                        [
                            rgba[0] as f32 / 255.0,
                            rgba[1] as f32 / 255.0,
                            rgba[2] as f32 / 255.0,
                            rgba[3] as f32 / 255.0,
                        ]
                    } else if v.is_leaf_chunk {
                        // Use chunk average color for better blending?
                        // For now keep existing logic
                        [0.4, 0.4, 0.45, 0.6]
                    } else {
                        [0.0, 0.0, 0.0, 0.0]
                    };

                    cpu_prepopulated_instances.push(VoxelInstanceRaw {
                        position: pos,
                        voxel_type: v.voxel_type as u32,
                        scale: scale,
                        ao_factor: 1.0,
                        custom_color: custom_color_f32,
                        emissive: [
                            emissive_rgb[0],
                            emissive_rgb[1],
                            emissive_rgb[2],
                            emissive_intensity,
                        ],
                    });

                    _voxel_expansion_count += 1;
                    cpu_prepop = true;
                }
            }

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
                self.palette.emissive(v.voxel_type as u32)
            };

            let mut flags = 0u32;
            if has_mesh {
                flags |= 1;
            }
            if has_envelope {
                flags |= 2;
            }
            if cpu_prepop {
                flags |= 4;
            }

            gpu_inputs.push(GpuInstanceInput {
                position: [
                    v.position[0] as f32,
                    v.position[1] as f32,
                    v.position[2] as f32,
                ],
                _pad0: 0,
                scale: v.scale,
                _pad1: 0,
                custom_color: custom_color_f32,
                emissive: [
                    emissive_rgb[0],
                    emissive_rgb[1],
                    emissive_rgb[2],
                    emissive_intensity,
                ],
                voxel_type: v.voxel_type as u32,
                flags,
                mesh_index: i as u32,
                envelope_index: i as u32,
            });
        }
        // Flatten any outputs (we pushed directly to gpu_inputs where needed)

        let gpu_candidate_count = gpu_inputs.len();

        if gpu_candidate_count > 0 {
            self.ensure_gpu_input_buffer(&device, gpu_candidate_count);
            if let Some(buffer) = self.gpu_input_buffer.as_ref() {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&gpu_inputs));
            }

            // Upload Mesh Indirect Args
            let mut mesh_indirect_args: Vec<wgpu::util::DrawIndexedIndirectArgs> =
                Vec::with_capacity(visible.len());
            for v in visible.iter() {
                let key = (v.position[0], v.position[1], v.position[2]);
                if cfg!(feature = "viewer-debug") {
                    let has_mesh = self.mesh_cache.contains_key(&key);
                    let has_envelope = self.envelope_mesh_cache.contains_key(&key);
                    viewer_debug!(
                        "SCENE DRAW: key=({},{},{}) leaf={} has_mesh={} has_envelope={}",
                        key.0,
                        key.1,
                        key.2,
                        v.is_leaf_chunk,
                        has_mesh,
                        has_envelope
                    );
                    viewer_debug!(
                        "SHADOW DRAW: key=({},{},{}) leaf={} has_mesh={} has_envelope={}",
                        key.0,
                        key.1,
                        key.2,
                        v.is_leaf_chunk,
                        has_mesh,
                        has_envelope
                    );
                }
                if v.is_leaf_chunk {
                    if let Some(mesh_entry) = self.mesh_cache.get(&key) {
                        let vertex_buf_size = self.vertex_allocator.total_size();
                        let index_buf_size = self.index_allocator.total_size();
                        let idx_end = mesh_entry.index_offset + mesh_entry.index_bytes;
                        let vb_end = mesh_entry.vertex_offset + mesh_entry.vertex_bytes;
                        if mesh_entry.index_bytes == 0
                            || mesh_entry.vertex_bytes == 0
                            || idx_end > index_buf_size
                            || vb_end > vertex_buf_size
                        {
                            if cfg!(feature = "viewer-debug") {
                                viewer_debug!("Validation warning on mesh cache entry (gpu prefill): index_buf_size={}, vertex_buf_size={} entry: index_offset={}, index_bytes={}, vertex_offset={}, vertex_bytes={}", index_buf_size, vertex_buf_size, mesh_entry.index_offset, mesh_entry.index_bytes, mesh_entry.vertex_offset, mesh_entry.vertex_bytes);
                            }
                            mesh_indirect_args.push(wgpu::util::DrawIndexedIndirectArgs {
                                index_count: 0,
                                instance_count: 0,
                                first_index: 0,
                                base_vertex: 0,
                                first_instance: 0,
                            });
                        } else {
                            let vertex_stride = std::mem::size_of::<MeshVertexRaw>() as u64;
                            let first_index = (mesh_entry.index_offset / 4) as u32;
                            let base_vertex = (mesh_entry.vertex_offset / vertex_stride) as i32;
                            mesh_indirect_args.push(wgpu::util::DrawIndexedIndirectArgs {
                                index_count: mesh_entry.index_count,
                                instance_count: 0, // Shader will set to 1
                                first_index: first_index,
                                base_vertex: base_vertex,
                                first_instance: 0,
                            });
                        }
                    } else {
                        mesh_indirect_args.push(wgpu::util::DrawIndexedIndirectArgs {
                            index_count: 0,
                            instance_count: 0,
                            first_index: 0,
                            base_vertex: 0,
                            first_instance: 0,
                        });
                    }
                } else {
                    mesh_indirect_args.push(wgpu::util::DrawIndexedIndirectArgs {
                        index_count: 0,
                        instance_count: 0,
                        first_index: 0,
                        base_vertex: 0,
                        first_instance: 0,
                    });
                }
            }

            if let Some(buffer) = self.mesh_indirect_buffer.as_ref() {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&mesh_indirect_args));
                // Mark entries as used so eviction won't free them during this frame
                for v in visible.iter() {
                    if !v.is_leaf_chunk {
                        continue;
                    }
                    let key = (v.position[0], v.position[1], v.position[2]);
                    if let Some(entry) = self.mesh_cache.get_mut(&key) {
                        entry.last_used_frame = self.frame_index;
                    }
                }
            }

            // Upload Envelope Indirect Args
            let mut envelope_indirect_args: Vec<wgpu::util::DrawIndexedIndirectArgs> =
                Vec::with_capacity(visible.len());
            for v in visible.iter() {
                let key = (v.position[0], v.position[1], v.position[2]);
                if v.is_leaf_chunk {
                    if let Some(mesh_entry) = self.envelope_mesh_cache.get(&key) {
                        let vertex_buf_size = self.vertex_allocator.total_size();
                        let index_buf_size = self.index_allocator.total_size();
                        let idx_end = mesh_entry.index_offset + mesh_entry.index_bytes;
                        let vb_end = mesh_entry.vertex_offset + mesh_entry.vertex_bytes;
                        if mesh_entry.index_bytes == 0
                            || mesh_entry.vertex_bytes == 0
                            || idx_end > index_buf_size
                            || vb_end > vertex_buf_size
                        {
                            if cfg!(feature = "viewer-debug") {
                                viewer_debug!("Validation warning on envelope cache entry (gpu prefill): index_buf_size={}, vertex_buf_size={} entry: index_offset={}, index_bytes={}, vertex_offset={}, vertex_bytes={}", index_buf_size, vertex_buf_size, mesh_entry.index_offset, mesh_entry.index_bytes, mesh_entry.vertex_offset, mesh_entry.vertex_bytes);
                            }
                            envelope_indirect_args.push(wgpu::util::DrawIndexedIndirectArgs {
                                index_count: 0,
                                instance_count: 0,
                                first_index: 0,
                                base_vertex: 0,
                                first_instance: 0,
                            });
                        } else {
                            let vertex_stride = std::mem::size_of::<MeshVertexRaw>() as u64;
                            let first_index = (mesh_entry.index_offset / 4) as u32;
                            let base_vertex = (mesh_entry.vertex_offset / vertex_stride) as i32;
                            envelope_indirect_args.push(wgpu::util::DrawIndexedIndirectArgs {
                                index_count: mesh_entry.index_count,
                                instance_count: 0, // Shader will set to 1
                                first_index: first_index,
                                base_vertex: base_vertex,
                                first_instance: 0,
                            });
                        }
                    } else {
                        envelope_indirect_args.push(wgpu::util::DrawIndexedIndirectArgs {
                            index_count: 0,
                            instance_count: 0,
                            first_index: 0,
                            base_vertex: 0,
                            first_instance: 0,
                        });
                    }
                } else {
                    envelope_indirect_args.push(wgpu::util::DrawIndexedIndirectArgs {
                        index_count: 0,
                        instance_count: 0,
                        first_index: 0,
                        base_vertex: 0,
                        first_instance: 0,
                    });
                }
            }

            if let Some(buffer) = self.envelope_indirect_buffer.as_ref() {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&envelope_indirect_args));
                // Mark entries as used so eviction won't free them during this frame
                for v in visible.iter() {
                    if !v.is_leaf_chunk {
                        continue;
                    }
                    let key = (v.position[0], v.position[1], v.position[2]);
                    if let Some(entry) = self.envelope_mesh_cache.get_mut(&key) {
                        entry.last_used_frame = self.frame_index;
                    }
                }
            }

            // Write any CPU prepopulated fallback instances and ensure the fallback instance buffer is large enough
            let cpu_prepopulated_count = cpu_prepopulated_instances.len();
            if cpu_prepopulated_count > 0 {
                // Ensure fallback instance buffer has room for prepopulated + new appended instances
                self.ensure_gpu_input_buffer(&device, gpu_candidate_count + cpu_prepopulated_count);
                if let Some(buffer) = self.fallback_instance_buffer.as_ref() {
                    queue.write_buffer(
                        buffer,
                        0,
                        bytemuck::cast_slice(&cpu_prepopulated_instances),
                    );
                }
            }

            // Reset Fallback Indirect Args (seed instance_count with CPU prepopulated count)
            if let Some(buffer) = self.fallback_indirect_buffer.as_ref() {
                let reset_args = wgpu::util::DrawIndirectArgs {
                    vertex_count: 36,
                    instance_count: cpu_prepopulated_count as u32,
                    first_vertex: 0,
                    first_instance: 0,
                };
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[reset_args]));
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
                _pad_align1: [0; 2],
                camera_right: {
                    let f = glam::Vec3::from_array(self.camera_controller.camera.forward);
                    let u = glam::Vec3::from_array(self.camera_controller.camera.up);
                    let r = f.cross(u).normalize_or_zero();
                    [r.x, r.y, r.z]
                },
                _pad_r0: 0,
                camera_up: self.camera_controller.camera.up,
                _pad_u0: 0,
                fov_tan: (self.camera_controller.camera.fov * 0.5).tan(),
                aspect: self.camera_controller.camera.aspect,
                screen_width: self.config.as_ref().map_or(1280.0, |c| c.width as f32),
                screen_height: self.config.as_ref().map_or(720.0, |c| c.height as f32),
                lod_render_distance: self.lod_distance,
                detail_cull_distance: self.fallback_detail_distance,
                envelope_distance: self.envelope_distance,
                hzb_enabled: if self.hzb_enabled { 1 } else { 0 },
                max_hzb_mip: self.hzb_mip_levels.saturating_sub(1),
                _pad3: 0.0,
                _pad_align2: [0; 2],
                view_proj: {
                    // Build view matrix
                    let cam_pos = glam::Vec3::from_array(self.camera_controller.camera.position);
                    let cam_forward = glam::Vec3::from_array(self.camera_controller.camera.forward);
                    let cam_up = glam::Vec3::from_array(self.camera_controller.camera.up);
                    let view = glam::Mat4::look_to_rh(cam_pos, cam_forward, cam_up);

                    // Build projection matrix
                    let proj = glam::Mat4::perspective_rh(
                        self.camera_controller.camera.fov,
                        self.camera_controller.camera.aspect,
                        self.camera_controller.camera.near,
                        self.camera_controller.camera.far,
                    );

                    // Combine
                    let vp = proj * view;
                    [
                        [vp.x_axis.x, vp.x_axis.y, vp.x_axis.z, vp.x_axis.w],
                        [vp.y_axis.x, vp.y_axis.y, vp.y_axis.z, vp.y_axis.w],
                        [vp.z_axis.x, vp.z_axis.y, vp.z_axis.z, vp.z_axis.w],
                        [vp.w_axis.x, vp.w_axis.y, vp.w_axis.z, vp.w_axis.w],
                    ]
                },
            };
            queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&gpu_params));
        }

        if gpu_candidate_count > 0 {
            self.ensure_cull_bind_group(&device);
        }

        let grouping_start = Instant::now();
        // Collect unique leaf chunk origins flagged by the culler
        let mut leaf_chunks: FxHashSet<(i64, i64, i64)> = FxHashSet::default();
        for v in &visible {
            if v.is_leaf_chunk {
                leaf_chunks.insert((v.position[0], v.position[1], v.position[2]));
            }
        }
        // Debug: print missing top-level neighbor cells near camera occasionally
        if cfg!(feature = "viewer-debug") && self.frame_count % 600 == 0 {
            // Top-level scale (root cell size)
            let root_scale = 16i64.pow(self.world.hierarchy_depth() as u32 - 1);
            let mut root_positions: FxHashSet<(i64, i64, i64)> = FxHashSet::default();
            for (x, y, z) in self.world.root().positions() {
                root_positions.insert((x as i64, y as i64, z as i64));
            }

            // Look for missing neighbors at root level around camera
            let cam = self.camera_controller.camera.position;
            let cam_top_x = (cam[0] as i64 / root_scale).clamp(0, 1 << 30);
            let cam_top_y = (cam[1] as i64 / root_scale).clamp(0, 1 << 30);
            let cam_top_z = (cam[2] as i64 / root_scale).clamp(0, 1 << 30);

            for dx in -2i64..=2i64 {
                for dy in -2i64..=2i64 {
                    for dz in -2i64..=2i64 {
                        let nx = cam_top_x + dx;
                        let ny = cam_top_y + dy;
                        let nz = cam_top_z + dz;
                        if !root_positions.contains(&(nx, ny, nz)) {
                            eprintln!(
                                "DEBUG: missing root cell at ({}, {}, {}) around camera top-level",
                                nx, ny, nz
                            );
                        }
                    }
                }
            }
        }
        let grouping_time = grouping_start.elapsed();

        let mesh_start = Instant::now();
        let mut mesh_leaf_proc_time = std::time::Duration::ZERO;
        let mut mesh_result_collect_time = std::time::Duration::ZERO;
        let mut mesh_schedule_time = std::time::Duration::ZERO;
        let mut mesh_pending_sort_time = std::time::Duration::ZERO;
        let mut mesh_job_creation_time = std::time::Duration::ZERO;
        let mut mesh_job_neighbors_time = std::time::Duration::ZERO;
        // Build mesh for any chunk present (near leaf chunk), and mark those chunks for drawing
        let mut cpu_mesh_keys: FxHashSet<(i64, i64, i64)> = FxHashSet::default();
        let mut new_meshes_created = 0;
        let mut chunks_not_found = 0;
        let mut missing_chunks: FxHashSet<(i64, i64, i64)> = FxHashSet::default();

        let leaf_scan_start = std::time::Instant::now();
        for &key in &leaf_chunks {
            let cam_pos = self.camera_controller.camera.position;
            let chunk_center = [key.0 as f32 + 8.0, key.1 as f32 + 8.0, key.2 as f32 + 8.0];
            let dx = chunk_center[0] - cam_pos[0];
            let dy = chunk_center[1] - cam_pos[1];
            let dz = chunk_center[2] - cam_pos[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let envelope_dist_sq = self.envelope_distance * self.envelope_distance;
            let max_envelope_dist_sq = self.max_envelope_distance * self.max_envelope_distance;
            let use_envelope = dist_sq > envelope_dist_sq;

            if use_envelope && dist_sq > max_envelope_dist_sq {
                continue;
            }

            let has_standard = self.mesh_cache.contains_key(&key);
            let has_envelope = self.envelope_mesh_cache.contains_key(&key);

            let can_draw_mesh = if use_envelope {
                has_envelope || has_standard
            } else {
                has_standard
            };

            if can_draw_mesh {
                cpu_mesh_keys.insert(key);
            }

            let needs_request = if use_envelope {
                !has_envelope
            } else {
                !has_standard
            };

            if needs_request {
                missing_chunks.insert(key);
                // If mesh workers are disabled (0), do not enqueue background meshing jobs
                if self.mesh_worker_count == 0 {
                    continue;
                }

                if !self.pending_chunk_set.contains(&key) {
                    // Prioritize meshing for chunks that are within the LOD distance of the camera
                    // Use squared LOD distance for speed
                    let lod_sq = self.lod_distance * self.lod_distance;
                    if dist_sq <= lod_sq {
                        // Push to front so it's processed sooner
                        self.pending_chunk_meshes.push_front(key);
                    } else {
                        // push to back as normal for background meshing
                        self.pending_chunk_meshes.push_back(key);
                    }
                    self.pending_chunk_set.insert(key);
                }
            }
        }

        mesh_leaf_proc_time += leaf_scan_start.elapsed();
        let result_collect_start = std::time::Instant::now();
        while let Ok(result) = self.mesh_result_rx.try_recv() {
            self.ready_chunk_meshes.push_back(result);
            if self.mesh_jobs_in_flight > 0 {
                self.mesh_jobs_in_flight -= 1;
            }
        }
        mesh_result_collect_time += result_collect_start.elapsed();

        let max_inflight = self.max_inflight_jobs();
        let schedule_start = std::time::Instant::now();
        while self.mesh_jobs_in_flight < max_inflight {
            // Backpressure: don't schedule more worker jobs if the ready result queue is already
            // large. This avoids generating more meshes than we can upload and prevents a
            // runaway backlog that keeps workers busy indefinitely.
            let ready_backlog_limit = std::cmp::max(64, self.mesh_upload_limit * 4);
            if self.ready_chunk_meshes.len() >= ready_backlog_limit {
                break;
            }
            // Occasionally re-sort the pending mesh queue to prioritize near-camera chunks.
            // This avoids reordering every frame and keeps scheduling cheap.
            if self.pending_chunk_meshes.len() > 4
                && (self.frame_index == 0
                    || (self.frame_index - self.last_pending_mesh_sort_frame)
                        >= self.pending_mesh_sort_interval_frames)
            {
                let sort_start = std::time::Instant::now();
                let cam_pos = self.camera_controller.camera.position;
                let mut vec: Vec<_> = self.pending_chunk_meshes.iter().cloned().collect();
                vec.sort_by(|a, b| {
                    let ca = [a.0 as f32 + 8.0, a.1 as f32 + 8.0, a.2 as f32 + 8.0];
                    let cb = [b.0 as f32 + 8.0, b.1 as f32 + 8.0, b.2 as f32 + 8.0];
                    let da = (ca[0] - cam_pos[0])
                        .mul_add(
                            ca[0] - cam_pos[0],
                            (ca[1] - cam_pos[1])
                                .mul_add(ca[1] - cam_pos[1], (ca[2] - cam_pos[2]).powi(2)),
                        )
                        .abs();
                    let db = (cb[0] - cam_pos[0])
                        .mul_add(
                            cb[0] - cam_pos[0],
                            (cb[1] - cam_pos[1])
                                .mul_add(cb[1] - cam_pos[1], (cb[2] - cam_pos[2]).powi(2)),
                        )
                        .abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
                self.pending_chunk_meshes.clear();
                for k in vec {
                    self.pending_chunk_meshes.push_back(k);
                }
                self.last_pending_mesh_sort_frame = self.frame_index;
                mesh_pending_sort_time += sort_start.elapsed();
            }
            let Some(key) = self.pending_chunk_meshes.pop_front() else {
                break;
            };
            let job_create_start = std::time::Instant::now();

            // Determine if we need an envelope or standard mesh
            let cam_pos = self.camera_controller.camera.position;
            let chunk_center = [key.0 as f32 + 8.0, key.1 as f32 + 8.0, key.2 as f32 + 8.0];
            let dx = chunk_center[0] - cam_pos[0];
            let dy = chunk_center[1] - cam_pos[1];
            let dz = chunk_center[2] - cam_pos[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let envelope_dist_sq = self.envelope_distance * self.envelope_distance;
            let max_envelope_dist_sq = self.max_envelope_distance * self.max_envelope_distance;
            let use_envelope = dist_sq > envelope_dist_sq;

            // If it's an envelope candidate but too far away, skip it entirely
            if use_envelope && dist_sq > max_envelope_dist_sq {
                self.pending_chunk_set.remove(&key);
                continue;
            }

            // Check if we already have the desired mesh type (it might have been completed since queuing)
            if use_envelope {
                if self.envelope_mesh_cache.contains_key(&key) {
                    self.pending_chunk_set.remove(&key);
                    continue;
                }
            } else {
                if self.mesh_cache.contains_key(&key) {
                    self.pending_chunk_set.remove(&key);
                    continue;
                }
            }

            match self
                .world
                .get_leaf_chunk_at_origin(WorldPos::new(key.0, key.1, key.2))
            {
                Some(chunk) => {
                    // Snapshot neighbor chunks so AO can be computed across chunk bounds.
                    let neighbor_start = std::time::Instant::now();
                    let mut neighbors: FxHashMap<(i8, i8, i8), Arc<Chunk>> = FxHashMap::default();
                    for dx in -1i64..=1 {
                        for dy in -1i64..=1 {
                            for dz in -1i64..=1 {
                                let nx = key.0 + (dx << 4);
                                let ny = key.1 + (dy << 4);
                                let nz = key.2 + (dz << 4);
                                if let Some(nc) = self
                                    .world
                                    .get_leaf_chunk_at_origin(WorldPos::new(nx, ny, nz))
                                {
                                    let nk = (nx, ny, nz);
                                    // Reuse an Arc snapshot if available; otherwise clone and cache
                                    let arc_neigh = if let Some(existing) =
                                        self.mesh_chunk_arc_cache.get(&nk)
                                    {
                                        existing.clone()
                                    } else {
                                        let a = Arc::new(nc.clone());
                                        self.mesh_chunk_arc_cache.insert(nk, a.clone());
                                        a
                                    };
                                    neighbors.insert((dx as i8, dy as i8, dz as i8), arc_neigh);
                                }
                            }
                        }
                    }

                    mesh_job_neighbors_time += neighbor_start.elapsed();
                    // Use cached Arc for the chunk as well
                    let chunk_arc = if let Some(existing) = self.mesh_chunk_arc_cache.get(&key) {
                        existing.clone()
                    } else {
                        let a = Arc::new(chunk.clone());
                        self.mesh_chunk_arc_cache.insert(key, a.clone());
                        a
                    };

                    if self
                        .mesh_job_tx
                        .send(MeshJob {
                            key,
                            chunk: chunk_arc,
                            neighbors,
                            envelope: use_envelope,
                        })
                        .is_ok()
                    {
                        self.mesh_jobs_in_flight += 1;
                    } else {
                        self.pending_chunk_meshes.push_front(key);
                        self.pending_chunk_set.remove(&key);
                        break;
                    }
                    mesh_job_creation_time += job_create_start.elapsed();
                }
                None => {
                    self.pending_chunk_set.remove(&key);
                    self.chunk_emitters.remove(&key);
                    chunks_not_found += 1;
                }
            }
        }
        mesh_schedule_time += schedule_start.elapsed();

        let mut processed_meshes = 0;
        // Decide per-frame upload limit and allow temporary boost when ready backlog accumulates
        let mut frame_mesh_upload_limit = self.mesh_upload_limit;
        if self.ready_chunk_meshes.len() > self.mesh_upload_limit * 4 {
            // boost cap - allow draining faster when there's a large backlog
            frame_mesh_upload_limit = self.mesh_upload_max;
        }
        // Detailed timing for mesh upload parts: vertex buffer creation, index buffer creation and cache insertion.
        let mut mesh_upload_vbuf_time = std::time::Duration::ZERO;
        let mut mesh_upload_ibuf_time = std::time::Duration::ZERO;
        let mut mesh_upload_entry_time = std::time::Duration::ZERO;
        let mut mesh_build_vb_time = std::time::Duration::ZERO;
        let mut mesh_emitters_proc_time = std::time::Duration::ZERO;
        let mesh_upload_total_start = std::time::Instant::now();
        while processed_meshes < frame_mesh_upload_limit {
            let Some(result) = self.ready_chunk_meshes.pop_front() else {
                break;
            };

            let MeshResult {
                key,
                mesh,
                voxel_count,
                is_envelope,
            } = result;
            self.pending_chunk_set.remove(&key);
            processed_meshes += 1;

            let emitters_start = std::time::Instant::now();
            if !is_envelope {
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
            }

            mesh_emitters_proc_time += emitters_start.elapsed();
            if mesh.indices.is_empty() {
                // Insert a placeholder (no-geometry) entry into the mesh cache so the chunk
                // won't be re-scheduled every frame. Reuse a global empty buffer pair.
                let entry = MeshCacheEntry {
                    vertex_offset: 0,
                    vertex_count: 0,
                    index_offset: 0,
                    index_count: 0,
                    vertex_bytes: 0,
                    index_bytes: 0,
                    last_used_frame: self.frame_index,
                    is_placeholder: true,
                };
                if is_envelope {
                    self.envelope_mesh_cache.insert(key, entry);
                } else {
                    self.mesh_cache.insert(key, entry);
                }

                new_meshes_created += 1;
                self.stat_empty_meshes += 1;
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

            let vb_build_start = std::time::Instant::now();
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
            mesh_build_vb_time += vb_build_start.elapsed();
            let vbuf_start = std::time::Instant::now();
            let (vertex_offset, index_offset) = self
                .allocate_mesh_in_megabuffer(&device, &queue, &vb_data, &mesh.indices)
                .expect("Failed to allocate in mega-buffers");
            mesh_upload_vbuf_time += vbuf_start.elapsed();
            let ibuf_start = std::time::Instant::now();
            mesh_upload_ibuf_time += ibuf_start.elapsed();
            mesh_upload_ibuf_time += ibuf_start.elapsed();
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
                vertex_offset,
                vertex_count: vb_data.len() as u32,
                index_offset,
                index_count: mesh.indices.len() as u32,
                vertex_bytes,
                index_bytes,
                last_used_frame: self.frame_index,
                is_placeholder: false,
            };
            let entry_start = std::time::Instant::now();
            if is_envelope {
                self.envelope_mesh_cache.insert(key, entry);
                self.envelope_mesh_cache_bytes += vertex_bytes + index_bytes;
            } else {
                self.mesh_cache.insert(key, entry);
                self.mesh_cache_bytes += vertex_bytes + index_bytes;
            }
            mesh_upload_entry_time += entry_start.elapsed();

            new_meshes_created += 1;

            if leaf_chunks.contains(&key) {
                cpu_mesh_keys.insert(key);
                missing_chunks.remove(&key);
            }
        }
        let mesh_upload_total_time = mesh_upload_total_start.elapsed();
        let mesh_time = mesh_start.elapsed();

        if self.mesh_cache_bytes > self.mesh_cache_byte_budget() {
            self.evict_mesh_cache();
        }
        if self.envelope_mesh_cache_bytes > self.envelope_mesh_cache_budget_bytes {
            self.evict_envelope_mesh_cache();
        }

        if chunks_not_found > 0 && self.frame_count == 0 {
            println!(
                "Warning: {} out of {} potential chunks not found (OSM voxels are not in subdivided chunks)",
                chunks_not_found,
                leaf_chunks.len()
            );
        }

        if gpu_candidate_count > 0 {
            self.run_gpu_culling(
                &device,
                &queue,
                gpu_candidate_count,
                cpu_prepopulated_instances.len() as u32,
            );
        }

        // If GPU culling is not being used, populate the multi-draw buffers from CPU-visible list.
        // Populate multi-draw buffers from CPU-visible list only if GPU culling is not used
        if gpu_candidate_count == 0 || self.cull_pipeline.is_none() {
            self.populate_multi_draw_indirects(&queue, &visible);
        }

        // Convert remaining instances (exclude those belonging to meshed chunks)
        let instance_start = Instant::now();

        let mut draw_mesh_keys = FxHashSet::default();
        for v in &visible {
            if v.is_leaf_chunk {
                let key = (v.position[0], v.position[1], v.position[2]);
                if cpu_mesh_keys.contains(&key) {
                    draw_mesh_keys.insert(key);
                }
            }
        }
        let instances: Vec<VoxelInstanceRaw> = Vec::new();
        let instance_time = instance_start.elapsed();

        self.active_emitters.clear();
        let mut draw_calls: usize = 0;
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

        if gpu_candidate_count == 0 {
            return;
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

        // Calculate inverse matrices for skybox rendering
        let inverse_view = view_mat.inverse();
        let inverse_proj = projection.inverse();
        let inverse_view_cols: [[f32; 4]; 4] = inverse_view.to_cols_array_2d();
        let inverse_proj_cols: [[f32; 4]; 4] = inverse_proj.to_cols_array_2d();

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
            let day_sun = [0.8, 0.76, 0.64]; // Reduced from [1.0, 0.95, 0.8] to 80%
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

        // Transform all frustum corners to light space first
        let mut frustum_corners_ls: [Vec3; 8] = [Vec3::ZERO; 8];
        for (i, corner) in frustum_corners.iter().enumerate() {
            frustum_corners_ls[i] = (light_view * corner.extend(1.0)).truncate();
        }

        // Calculate bounding sphere center in light space
        let mut frustum_center_ls = Vec3::ZERO;
        for corner_ls in &frustum_corners_ls {
            frustum_center_ls += *corner_ls;
        }
        frustum_center_ls /= 8.0;

        // Calculate radius in light space
        let mut radius: f32 = 0.0;
        for corner_ls in &frustum_corners_ls {
            radius = radius.max(frustum_center_ls.distance(*corner_ls));
        }

        // Create bounds in light space
        let mut bounds_min = frustum_center_ls - Vec3::splat(radius);
        let mut bounds_max = frustum_center_ls + Vec3::splat(radius);

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
                // Track reallocated light probe buffer bytes
                let new_probes_bytes =
                    (self.light_probe_capacity * std::mem::size_of::<LightProbe>()) as u64;
                App::replace_buffer_bytes_static(
                    &mut self.light_probe_buffer_bytes,
                    new_probes_bytes,
                    &mut self.gpu_buffer_bytes,
                );
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
            shadow_texel_size_pad: [
                shadow_texel,
                shadow_texel,
                self.pcf_radius,
                self.pcf_poisson_samples as f32,
            ],
            shadow_darkness_pad: [self.shadow_darkness, self.shadow_backface_scale, 0.0, 0.0],
            moon_direction_intensity: [
                moon_direction_vec.x,
                moon_direction_vec.y,
                moon_direction_vec.z,
                moon_intensity,
            ],
            moon_color_pad: [moon_color[0], moon_color[1], moon_color[2], 0.0],
            light_probe_count,
            lod_distance: self.lod_distance,
            envelope_distance: self.envelope_distance,
            envelope_fade_range: self.envelope_fade_range,
            inverse_view: inverse_view_cols,
            inverse_proj: inverse_proj_cols,
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

        // GPU cull invocation is handled via run_gpu_culling

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

            if let Some(_mesh_indirect) = &self.mesh_indirect_buffer {
                shadow_pass.set_pipeline(shadow_mesh_pipeline);
                shadow_pass.set_bind_group(0, shadow_bind_group, &[]);

                // Bind mega buffers once to reduce binding overhead
                shadow_pass
                    .set_vertex_buffer(0, self.mega_vertex_buffer.as_ref().unwrap().slice(..));
                shadow_pass.set_index_buffer(
                    self.mega_index_buffer.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                // For shadows, always use direct draws (not multi-draw indirect)
                // This allows us to render all leaf chunks, not just camera-visible ones
                let use_multi_draw_for_shadows = false;

                if use_multi_draw_for_shadows {
                    if cfg!(feature = "viewer-debug") {
                        viewer_debug!(
                            "Shader path: Using multi-draw indirect for shadow pass (CPU path)"
                        );
                    }
                    // Use the multi-draw indexed indirect count function
                    if let (Some(count_buf), Some(mesh_indirect)) = (
                        self.multi_draw_count_buffer.as_ref(),
                        self.mesh_indirect_buffer.as_ref(),
                    ) {
                        shadow_pass.multi_draw_indexed_indirect_count(
                            mesh_indirect,
                            0,
                            count_buf,
                            0,
                            self.max_draw_capacity as u32,
                        );
                        draw_calls += self.mesh_cache.len(); // approximate
                    }
                    if let (Some(count_buf), Some(envelope_indirect)) = (
                        self.multi_draw_count_buffer.as_ref(),
                        self.envelope_indirect_buffer.as_ref(),
                    ) {
                        // envelope count is at offset 4
                        shadow_pass.multi_draw_indexed_indirect_count(
                            envelope_indirect,
                            0,
                            count_buf,
                            4,
                            self.max_draw_capacity as u32,
                        );
                        draw_calls += self.envelope_mesh_cache.len(); // approximate
                    }
                } else {
                    // For shadows, render ALL cached meshes (not just camera-visible leaf_chunks)
                    // This ensures complete shadow coverage for all loaded geometry

                    // Draw all detail meshes
                    for (_, entry) in self.mesh_cache.iter() {
                        let start_index = (entry.index_offset / 4) as u32;
                        let end_index = start_index + entry.index_count;
                        let base_vertex = (entry.vertex_offset
                            / std::mem::size_of::<MeshVertexRaw>() as u64)
                            as i32;
                        shadow_pass.draw_indexed(start_index..end_index, base_vertex, 0..1);
                        draw_calls += 1;
                    }

                    // Draw all envelope meshes
                    for (_, entry) in self.envelope_mesh_cache.iter() {
                        let start_index = (entry.index_offset / 4) as u32;
                        let end_index = start_index + entry.index_count;
                        let base_vertex = (entry.vertex_offset
                            / std::mem::size_of::<MeshVertexRaw>() as u64)
                            as i32;
                        shadow_pass.draw_indexed(start_index..end_index, base_vertex, 0..1);
                        draw_calls += 1;
                    }
                }
            }

            if let Some(fallback_indirect) = &self.fallback_indirect_buffer {
                shadow_pass.set_pipeline(shadow_pipeline);
                shadow_pass.set_bind_group(0, shadow_bind_group, &[]);
                shadow_pass
                    .set_vertex_buffer(0, self.cube_vertex_buffer.as_ref().unwrap().slice(..));
                shadow_pass.set_vertex_buffer(
                    1,
                    self.fallback_instance_buffer.as_ref().unwrap().slice(..),
                );
                shadow_pass.draw_indirect(fallback_indirect, 0);
                draw_calls += 1;
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

            // Draw skybox first (background)
            if let (Some(pipeline), Some(bind_group)) = (
                self.skybox_pipeline.as_ref(),
                self.skybox_bind_group.as_ref(),
            ) {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
                render_pass.set_bind_group(1, bind_group, &[]);
                render_pass.draw(0..3, 0..1); // Full screen triangle
            }

            // Draw meshed chunks first
            if let Some(mesh_indirect) = &self.mesh_indirect_buffer {
                render_pass.set_pipeline(self.mesh_pipeline.as_ref().unwrap());
                render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);

                // Bind mega buffers once per pass
                render_pass
                    .set_vertex_buffer(0, self.mega_vertex_buffer.as_ref().unwrap().slice(..));
                render_pass.set_index_buffer(
                    self.mega_index_buffer.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                let multi_draw_supported = device
                    .features()
                    .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
                    && self.cull_pipeline.is_none();

                if multi_draw_supported {
                    if cfg!(feature = "viewer-debug") {
                        viewer_debug!(
                            "Shader path: Using multi-draw indirect for scene pass (CPU path)"
                        );
                    }
                    if let (Some(count_buf), Some(mesh_indirect)) = (
                        self.multi_draw_count_buffer.as_ref(),
                        self.mesh_indirect_buffer.as_ref(),
                    ) {
                        render_pass.multi_draw_indexed_indirect_count(
                            mesh_indirect,
                            0,
                            count_buf,
                            0,
                            self.max_draw_capacity as u32,
                        );
                        draw_calls += self.mesh_cache.len(); // approximate
                    }
                    if let (Some(count_buf), Some(envelope_indirect)) = (
                        self.multi_draw_count_buffer.as_ref(),
                        self.envelope_indirect_buffer.as_ref(),
                    ) {
                        render_pass.multi_draw_indexed_indirect_count(
                            envelope_indirect,
                            0,
                            count_buf,
                            4,
                            self.max_draw_capacity as u32,
                        );
                        draw_calls += self.envelope_mesh_cache.len();
                    }
                } else {
                    if cfg!(feature = "viewer-debug") {
                        viewer_debug!("Shader path: Using per-chunk indirect draws for scene pass (maybe GPU cull active)");
                    }
                    let mut drawn_meshes = 0;
                    for (i, v) in visible.iter().enumerate() {
                        if !v.is_leaf_chunk {
                            continue;
                        }
                        let key = (v.position[0], v.position[1], v.position[2]);

                        // Draw Detail Mesh
                        if let Some(entry) = self.mesh_cache.get_mut(&key) {
                            render_pass.draw_indexed_indirect(mesh_indirect, (i * 20) as u64);
                            draw_calls += 1;
                            entry.last_used_frame = self.frame_index;
                            drawn_meshes += 1;
                        }

                        // Draw Envelope Mesh
                        if let Some(envelope_indirect) = &self.envelope_indirect_buffer {
                            if let Some(entry) = self.envelope_mesh_cache.get_mut(&key) {
                                render_pass
                                    .draw_indexed_indirect(envelope_indirect, (i * 20) as u64);
                                draw_calls += 1;
                                entry.last_used_frame = self.frame_index;
                                drawn_meshes += 1;
                            }
                        }
                    }
                    if cfg!(feature = "viewer-debug") && self.frame_count == 0 {
                        viewer_debug!("DEBUG: Drew {} meshes (indirect)", drawn_meshes);
                    }
                }
            }

            // Draw remaining instanced cubes
            render_pass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            render_pass.set_vertex_buffer(0, self.cube_vertex_buffer.as_ref().unwrap().slice(..));
            if let Some(fallback_indirect) = &self.fallback_indirect_buffer {
                render_pass.set_vertex_buffer(
                    1,
                    self.fallback_instance_buffer.as_ref().unwrap().slice(..),
                );
                render_pass.draw_indirect(fallback_indirect, 0);
                draw_calls += 1;
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

        // Early-out conditions: skip DoF when negligible effect
        let skip_dof = !self.dof_enabled
            || self.dof_settings.blur_strength < 0.05
            || self.dof_settings.focal_range > 450.0;

        if !skip_dof {
            // If DoF is enabled we always run the cheap CoC copy pass to produce base color + CoC in alpha.
            let use_kawase =
                self.dof_settings.kawase_enabled && self.dof_settings.kawase_iterations > 0;
            if let (
                Some(dof_bind_group),
                Some(dof_buffer),
                Some(dof_color_view),
                Some(dof_coc_pipeline),
            ) = (
                self.dof_bind_group.as_ref(),
                self.dof_uniform_buffer.as_ref(),
                self.dof_color_view.as_ref(),
                self.dof_coc_pipeline.as_ref(),
            ) {
                let blur_strength = self.dof_settings.blur_strength;
                let gpu_uniforms = self.pack_dof_uniforms(blur_strength);
                queue.write_buffer(dof_buffer, 0, bytemuck::cast_slice(&gpu_uniforms));
                let mut post_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DoF CoC Copy Pass"),
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
                post_pass.set_pipeline(dof_coc_pipeline);
                post_pass.set_bind_group(0, dof_bind_group, &[]);
                post_pass.draw(0..3, 0..1);
            }

            // Only run Kawase blur passes when Kawase is enabled.
            if use_kawase {
                if let (
                    Some(kawase_down_pipeline),
                    Some(kawase_up_pipeline),
                    Some(kawase_ping_views),
                ) = (
                    self.kawase_down_pipeline.as_ref(),
                    self.kawase_up_pipeline.as_ref(),
                    Some(&self.kawase_ping_views),
                ) {
                    let iterations = self.dof_settings.kawase_iterations.min(6).max(1);
                    let inst_start = std::time::Instant::now();
                    // Down passes
                    for level in 0..iterations {
                        let target_view = kawase_ping_views[level]
                            .as_ref()
                            .expect("Kawase ping view missing");
                        // Update UBO for this level with texel size and offset
                        if let Some(Some(ubo)) = self.kawase_uniform_buffers.get(level) {
                            if let Some((w, h)) = self.kawase_level_sizes.get(level) {
                                let texel_size = [1.0 / (*w) as f32, 1.0 / (*h) as f32];
                                let offset = self.dof_settings.kawase_offset
                                    * (level as f32 + 1.0)
                                    * self.dof_settings.blur_strength;
                                let ubo_data = [texel_size[0], texel_size[1], offset, 0.0f32];
                                let changed = match self.kawase_last_ubo.get(level) {
                                    Some(prev) => {
                                        (prev[0] - ubo_data[0]).abs() > 1e-6
                                            || (prev[1] - ubo_data[1]).abs() > 1e-6
                                            || (prev[2] - ubo_data[2]).abs() > 1e-6
                                            || (prev[3] - ubo_data[3]).abs() > 1e-6
                                    }
                                    None => true,
                                };
                                if changed {
                                    let write_start = std::time::Instant::now();
                                    let queue_ref = self.queue.as_ref().unwrap();
                                    queue_ref.write_buffer(ubo, 0, bytemuck::cast_slice(&ubo_data));
                                    if let Some(prev) = self.kawase_last_ubo.get_mut(level) {
                                        *prev = ubo_data;
                                    }
                                    self.kawase_write_acc += write_start.elapsed();
                                }
                            }
                        }

                        if let Some(Some(bind_group)) = self.kawase_down_bind_groups.get(level) {
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some(&format!("Kawase Down L{}", level)),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: target_view,
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
                            pass.set_pipeline(kawase_down_pipeline);
                            pass.set_bind_group(0, bind_group, &[]);
                            pass.draw(0..3, 0..1);
                        }
                    }

                    // Up passes
                    for level_rev in (0..iterations).rev() {
                        let target_view = if level_rev == 0 {
                            self.dof_color_view.as_ref().unwrap()
                        } else {
                            self.kawase_pong_views[level_rev - 1].as_ref().unwrap()
                        };
                        if let Some(Some(ubo)) = self.kawase_uniform_buffers.get(level_rev) {
                            if let Some((w, h)) = self.kawase_level_sizes.get(level_rev) {
                                let texel_size = [1.0 / (*w) as f32, 1.0 / (*h) as f32];
                                let offset = self.dof_settings.kawase_offset
                                    * (level_rev as f32 + 1.0)
                                    * self.dof_settings.blur_strength;
                                let ubo_data = [texel_size[0], texel_size[1], offset, 0.0f32];
                                let changed = match self.kawase_last_ubo.get(level_rev) {
                                    Some(prev) => {
                                        (prev[0] - ubo_data[0]).abs() > 1e-6
                                            || (prev[1] - ubo_data[1]).abs() > 1e-6
                                            || (prev[2] - ubo_data[2]).abs() > 1e-6
                                            || (prev[3] - ubo_data[3]).abs() > 1e-6
                                    }
                                    None => true,
                                };
                                if changed {
                                    let write_start = std::time::Instant::now();
                                    let queue_ref = self.queue.as_ref().unwrap();
                                    queue_ref.write_buffer(ubo, 0, bytemuck::cast_slice(&ubo_data));
                                    if let Some(prev) = self.kawase_last_ubo.get_mut(level_rev) {
                                        *prev = ubo_data;
                                    }
                                    self.kawase_write_acc += write_start.elapsed();
                                }
                            }
                        }

                        if let Some(Some(bind_group)) = self.kawase_up_bind_groups.get(level_rev) {
                            let pass_start = std::time::Instant::now();
                            let mut up_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some(&format!("Kawase Up L{}", level_rev)),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: target_view,
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
                            up_pass.set_pipeline(kawase_up_pipeline);
                            up_pass.set_bind_group(0, bind_group, &[]);
                            up_pass.draw(0..3, 0..1);
                            self.kawase_pass_acc += pass_start.elapsed();
                        }
                        self.kawase_acc_frames += 1;
                        self.kawase_pass_acc += inst_start.elapsed();
                    }
                }
            }
        }

        // Final DoF combine: source color + blurred DoF + CoC => post_color_view (skip if DoF disabled)
        if !skip_dof {
            if let (
                Some(dof_combine_pipeline),
                Some(dof_combine_bind_group),
                Some(post_color_view),
            ) = (
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
        } else {
            // DoF disabled: copy offscreen color directly to post_color_view for bloom/composite
            if let (Some(offscreen_tex), Some(post_tex)) = (
                self.offscreen_color_texture.as_ref(),
                self.post_color_texture.as_ref(),
            ) {
                let size = self.config.as_ref().map(|c| wgpu::Extent3d {
                    width: c.width,
                    height: c.height,
                    depth_or_array_layers: 1,
                });
                if let Some(extent) = size {
                    encoder.copy_texture_to_texture(
                        offscreen_tex.as_image_copy(),
                        post_tex.as_image_copy(),
                        extent,
                    );
                }
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
                // SSILVB/SSAO: run before bloom so AO can affect later passes
                if self.ssao_enabled {
                    if let (Some(ssilvb_pipeline), Some(ssilvb_bind_group), Some(ssao_ping_view)) = (
                        self.ssilvb_pipeline.as_ref(),
                        self.ssilvb_bind_group.as_ref(),
                        self.ssao_ping_view.as_ref(),
                    ) {
                        let mut ssao_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("SSILVB Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: ssao_ping_view,
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
                        ssao_pass.set_pipeline(ssilvb_pipeline);
                        ssao_pass.set_bind_group(0, ssilvb_bind_group, &[]);
                        ssao_pass.draw(0..3, 0..1);
                    }

                    // Optional SSAO blur (reduce speckle): horizontal then vertical
                    if self.ssao_settings.blur_enabled {
                        if let (Some(ssao_blur_pipeline), Some(ssao_blur_h), Some(ssao_pong_view)) = (
                            self.ssao_blur_pipeline.as_ref(),
                            self.ssao_blur_horizontal_bind_group.as_ref(),
                            self.ssao_pong_view.as_ref(),
                        ) {
                            let mut blur_pass_h =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("SSAO Blur Horizontal Pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: ssao_pong_view,
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
                            blur_pass_h.set_pipeline(ssao_blur_pipeline);
                            blur_pass_h.set_bind_group(0, ssao_blur_h, &[]);
                            blur_pass_h.draw(0..3, 0..1);
                        }
                    }

                    if self.ssao_settings.blur_enabled {
                        if let (Some(ssao_blur_pipeline), Some(ssao_blur_v), Some(ssao_ping_view)) = (
                            self.ssao_blur_pipeline.as_ref(),
                            self.ssao_blur_vertical_bind_group.as_ref(),
                            self.ssao_ping_view.as_ref(),
                        ) {
                            let mut blur_pass_v =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("SSAO Blur Vertical Pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: ssao_ping_view,
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
                            blur_pass_v.set_pipeline(ssao_blur_pipeline);
                            blur_pass_v.set_bind_group(0, ssao_blur_v, &[]);
                            blur_pass_v.draw(0..3, 0..1);
                        }
                    }
                }
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

        // Egui rendering
        if let (Some(egui_ctx), Some(egui_winit), Some(window)) =
            (&self.egui_ctx, &mut self.egui_winit, &self.window)
        {
            let raw_input = egui_winit.take_egui_input(window);
            egui_ctx.begin_pass(raw_input);

            egui::Area::new(egui::Id::new("fps_counter"))
                .fixed_pos(egui::pos2(10.0, 10.0))
                .show(egui_ctx, |ui| {
                    egui::Frame::default()
                        .fill(egui::Color32::from_black_alpha(222))
                        .inner_margin(5.0)
                        .corner_radius(5.0)
                        .show(ui, |ui| {
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                            ui.label(
                                egui::RichText::new(format!("FPS: {}", self.last_fps))
                                    .color(egui::Color32::WHITE)
                                    .size(10.0),
                            );
                            // Additional stats: mirror the console output but in overlay
                            ui.label(
                                egui::RichText::new(format!("Visible: {}", self.visible_count))
                                    .color(egui::Color32::WHITE)
                                    .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!("Meshed: {}", self.meshed_chunk_count))
                                    .color(egui::Color32::WHITE)
                                    .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "Pending: {}",
                                    self.pending_mesh_count
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "Jobs/s: {}",
                                    self.jobs_per_sec_snapshot
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            // (Moved) Mesh cache and envelopes will be printed under 'Process' for clarity
                            ui.label(
                                egui::RichText::new(format!(
                                    "Process: {:.0} MiB",
                                    self.process_mem_mib
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Mesh cache: {:.0}/{:.0} MiB",
                                    self.mesh_cache_mib, self.mesh_budget_mib
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Envelopes: {:.0}/{:.0} MiB",
                                    self.envelope_cache_mib, self.mesh_budget_mib
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "GPU tracked: {:.0} MiB",
                                    (self.gpu_buffer_bytes + self.gpu_texture_bytes) as f64
                                        / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Uniforms: {:.1} MiB",
                                    (self.uniform_buffer_bytes as f64) / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Mega VB/IB: {:.1}/{:.1} MiB",
                                    (self.mega_vertex_buffer_bytes as f64) / (1024.0 * 1024.0),
                                    (self.mega_index_buffer_bytes as f64) / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  GPU Input: {:.1} MiB",
                                    (self.gpu_input_buffer_bytes as f64) / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Indirects (mesh/env): {:.1}/{:.1} MiB",
                                    (self.mesh_indirect_bytes as f64) / (1024.0 * 1024.0),
                                    (self.envelope_indirect_bytes as f64) / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Offscreen/Depth/Post: {:.1}/{:.1}/{:.1} MiB",
                                    (self.offscreen_color_texture_bytes as f64) / (1024.0 * 1024.0),
                                    (self.depth_texture_bytes as f64) / (1024.0 * 1024.0),
                                    (self.post_color_texture_bytes as f64) / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Kawase ping/pong: {:.1}/{:.1} MiB",
                                    (self.kawase_ping_bytes as f64) / (1024.0 * 1024.0),
                                    (self.kawase_pong_bytes as f64) / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Bloom ping/pong: {:.1}/{:.1} MiB",
                                    (self.bloom_ping_bytes as f64) / (1024.0 * 1024.0),
                                    (self.bloom_pong_bytes as f64) / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  SSAO ping/pong: {:.1}/{:.1} MiB",
                                    (self.ssao_ping_bytes as f64) / (1024.0 * 1024.0),
                                    (self.ssao_pong_bytes as f64) / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "  Shadow/Skybox: {:.1}/{:.1} MiB",
                                    (self.shadow_map_bytes as f64) / (1024.0 * 1024.0),
                                    (self.skybox_texture_bytes as f64) / (1024.0 * 1024.0)
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!("Cull: {:.2}ms", self.cull_ms))
                                    .color(egui::Color32::WHITE)
                                    .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!("Group: {:.2}ms", self.grouping_ms))
                                    .color(egui::Color32::WHITE)
                                    .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!("Mesh: {:.2}ms", self.mesh_ms))
                                    .color(egui::Color32::WHITE)
                                    .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "Instancing: {:.2}ms",
                                    self.instance_ms
                                ))
                                .color(egui::Color32::WHITE)
                                .size(10.0),
                            );
                            ui.label(
                                egui::RichText::new(format!("Draws: {}", self.draw_calls_count))
                                    .color(egui::Color32::WHITE)
                                    .size(10.0),
                            );
                        });
                });

            let full_output = egui_ctx.end_pass();
            let paint_jobs = egui_ctx.tessellate(full_output.shapes, egui_ctx.pixels_per_point());
            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [
                    self.config.as_ref().unwrap().width,
                    self.config.as_ref().unwrap().height,
                ],
                pixels_per_point: egui_ctx.pixels_per_point(),
            };

            // Take the renderer out to avoid mutable borrow issues
            if let Some(mut egui_renderer) = self.egui_renderer.take() {
                egui_winit.handle_platform_output(window, full_output.platform_output);

                for (id, image_delta) in &full_output.textures_delta.set {
                    egui_renderer.update_texture(
                        self.device.as_ref().unwrap(),
                        self.queue.as_ref().unwrap(),
                        *id,
                        image_delta,
                    );
                }

                egui_renderer.update_buffers(
                    self.device.as_ref().unwrap(),
                    self.queue.as_ref().unwrap(),
                    &mut encoder,
                    &paint_jobs,
                    &screen_descriptor,
                );

                {
                    let egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    // Convert to 'static lifetime for egui-wgpu
                    let mut egui_pass_static = egui_pass.forget_lifetime();

                    egui_renderer.render(&mut egui_pass_static, &paint_jobs, &screen_descriptor);
                }

                for id in &full_output.textures_delta.free {
                    egui_renderer.free_texture(id);
                }

                self.egui_renderer = Some(egui_renderer);
            }
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
            let (process_mem_mib, process_vmem_mib) = self
                .system_info
                .process(self.process_pid)
                .map(|p| {
                    (
                        p.memory() as f64 / (1024.0 * 1024.0),
                        p.virtual_memory() as f64 / (1024.0 * 1024.0),
                    )
                })
                .unwrap_or((0.0, 0.0));

            // Track memory in major in-app categories (mesh caches + megabuffers + GPU input buffers)
            let tracked_bytes: u64 = self
                .mesh_cache_bytes
                .saturating_add(self.envelope_mesh_cache_bytes)
                .saturating_add(self.vertex_allocator.allocated_bytes())
                .saturating_add(self.index_allocator.allocated_bytes())
                .saturating_add(
                    (self.gpu_input_capacity as u64)
                        * std::mem::size_of::<GpuInstanceInput>() as u64,
                )
                .saturating_add(
                    (self.fallback_instance_capacity as u64)
                        * std::mem::size_of::<VoxelInstanceRaw>() as u64,
                );
            let tracked_mem_mib = tracked_bytes as f64 / (1024.0 * 1024.0);
            // Tracked GPU reserved bytes collected from allocations:
            let gpu_reserved_bytes = self.gpu_buffer_bytes.saturating_add(self.gpu_texture_bytes);
            let gpu_reserved_mib = gpu_reserved_bytes as f64 / (1024.0 * 1024.0);
            let ready_count = self.ready_chunk_meshes.len();
            self.last_fps = self.frame_count as u32;
            let jobs_in_flight = self.mesh_jobs_in_flight;
            let pending_set_count = self.pending_chunk_set.len();
            let jobs_per_sec = self.mesh_jobs_executed.swap(0, Ordering::Relaxed);
            let mesh_idle = self.pending_chunk_meshes.is_empty()
                && self.ready_chunk_meshes.is_empty()
                && jobs_in_flight == 0;
            println!(
                "FPS: {}, Visible items: {}, Leaf chunks: {}, Meshed chunks: {}, Pending: {}, PendingSet: {}, Ready: {}, InFlight: {}, Fallback: {}, Mesh cache: {:.1}/{:.1} MiB, Process (RSS/VM): {:.1}/{:.1} MiB, Tracked: {:.1} MiB, GPU reserved: {:.1} MiB, Cull: {:.2}ms, Group: {:.2}ms, Mesh: {:.2}ms, Instances: {:.2}ms, Draws: {}, Jobs/sec: {}, EmptyMeshes: {}, VReuse: {}, IReuse: {}, VPool: {}, IPool: {}, MeshIdle: {}, DoF Kawase: {} (iter={}, off={:.2}), MeshUp: {:.2}ms parts:(leaf:{:.2} sched:{:.2} sort:{:.2} res:{:.2} jobc:{:.2} jobn:{:.2}) vbld:{:.2} v:{:.2} i:{:.2} ins:{:.2} emit:{:.2} processed:{} limit:{}",
                self.frame_count,
                total_visible,
                leaf_chunks.len(),
                draw_mesh_keys.len(),
                self.pending_chunk_meshes.len(),
                pending_set_count,
                ready_count,
                jobs_in_flight,
                missing_chunks.len(),
                mesh_cache_mib,
                mesh_budget_mib,
                process_mem_mib,
                process_vmem_mib,
                tracked_mem_mib,
                gpu_reserved_mib,
                cull_time.as_secs_f64() * 1000.0,
                grouping_time.as_secs_f64() * 1000.0,
                    (if mesh_idle { std::time::Duration::from_secs(0) } else { mesh_time }).as_secs_f64() * 1000.0,
                instance_time.as_secs_f64() * 1000.0,
                draw_calls,
                jobs_per_sec,
                self.stat_empty_meshes,
                self.stat_vertex_buffers_reused,
                self.stat_index_buffers_reused,
                mesh_idle,
                self.vertex_allocator.allocated_count(),
                self.index_allocator.allocated_count(),
                if self.dof_settings.kawase_enabled {"enabled"} else {"disabled"}
                , self.dof_settings.kawase_iterations
                , self.dof_settings.kawase_offset
                , mesh_upload_total_time.as_secs_f64() * 1000.0
                , mesh_leaf_proc_time.as_secs_f64() * 1000.0
                , mesh_schedule_time.as_secs_f64() * 1000.0
                , mesh_pending_sort_time.as_secs_f64() * 1000.0
                , mesh_result_collect_time.as_secs_f64() * 1000.0
                , mesh_job_creation_time.as_secs_f64() * 1000.0
                , mesh_job_neighbors_time.as_secs_f64() * 1000.0
                , mesh_build_vb_time.as_secs_f64() * 1000.0
                , mesh_upload_vbuf_time.as_secs_f64() * 1000.0
                , mesh_upload_ibuf_time.as_secs_f64() * 1000.0
                , mesh_upload_entry_time.as_secs_f64() * 1000.0
                , mesh_emitters_proc_time.as_secs_f64() * 1000.0
                , processed_meshes
                    , frame_mesh_upload_limit
            );

            // Update UI overlay stats
            self.visible_count = total_visible;
            self.leaf_chunk_count = leaf_chunks.len();
            self.meshed_chunk_count = draw_mesh_keys.len();
            self.pending_mesh_count = self.pending_chunk_meshes.len();
            self.pending_mesh_set_count = pending_set_count;
            self.ready_mesh_count = ready_count;
            self.jobs_in_flight = jobs_in_flight;
            self.jobs_per_sec_snapshot = jobs_per_sec;
            self.process_mem_mib = process_mem_mib;
            self.mesh_cache_mib = mesh_cache_mib;
            self.mesh_budget_mib = mesh_budget_mib;
            self.envelope_cache_mib = self.envelope_mesh_cache_bytes as f64 / (1024.0 * 1024.0);
            self.cull_ms = cull_time.as_secs_f64() * 1000.0;
            self.grouping_ms = grouping_time.as_secs_f64() * 1000.0;
            self.mesh_ms = (if mesh_idle {
                std::time::Duration::from_secs(0)
            } else {
                mesh_time
            })
            .as_secs_f64()
                * 1000.0;
            self.instance_ms = instance_time.as_secs_f64() * 1000.0;
            self.draw_calls_count = draw_calls;

            // Update UI overlay stats
            self.visible_count = total_visible;
            self.leaf_chunk_count = leaf_chunks.len();
            self.meshed_chunk_count = draw_mesh_keys.len();
            self.pending_mesh_count = self.pending_chunk_meshes.len();
            self.pending_mesh_set_count = pending_set_count;
            self.ready_mesh_count = ready_count;
            self.jobs_in_flight = jobs_in_flight;
            self.jobs_per_sec_snapshot = jobs_per_sec;
            // Print Kawase timing ones per second to avoid spamming
            if self.kawase_acc_frames > 0 {
                let avg_write =
                    self.kawase_write_acc.as_secs_f64() * 1000.0 / self.kawase_acc_frames as f64;
                let avg_pass =
                    self.kawase_pass_acc.as_secs_f64() * 1000.0 / self.kawase_acc_frames as f64;
                println!(
                    "Kawase write avg: {:.3}ms, pass avg: {:.3}ms (over {} frames)",
                    avg_write, avg_pass, self.kawase_acc_frames
                );
                self.kawase_write_acc = std::time::Duration::from_secs(0);
                self.kawase_pass_acc = std::time::Duration::from_secs(0);
                self.kawase_acc_frames = 0;
            }
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
        // Let egui handle the event first
        if let (Some(egui_winit), Some(window)) = (&mut self.egui_winit, &self.window) {
            let response = egui_winit.on_window_event(window, &event);
            if response.consumed {
                // egui consumed the event, don't pass it to the game
                return;
            }
        }

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
