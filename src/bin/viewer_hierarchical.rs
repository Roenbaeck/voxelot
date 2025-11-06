//! Modern wgpu viewer for hierarchical voxel chunks
//!
//! Features:
//! - 6DOF camera with WASD + mouse
//! - Frustum culling
//! - Occlusion culling
//! - LOD support
//! - Instanced rendering

use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

use voxelot::{World, WorldPos, Camera, VisibilityCache};

const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;

/// Voxel instance data for GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VoxelInstanceRaw {
    position: [f32; 3],
    voxel_type: u32,
}

/// Camera controller for 6DOF movement
struct CameraController {
    camera: Camera,
    speed: f32,
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
}

impl CameraController {
    fn new(position: [f32; 3]) -> Self {
        Self {
            camera: Camera::new(position, [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
            speed: 10.0,
            sensitivity: 0.002,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
        }
    }
    
    fn process_keyboard(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft => self.down = pressed,
            _ => {}
        }
    }
    
    fn process_mouse(&mut self, delta_x: f64, delta_y: f64) {
        self.yaw += delta_x as f32 * self.sensitivity;
        self.pitch -= delta_y as f32 * self.sensitivity;
        
        // Clamp pitch
        self.pitch = self.pitch.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
        
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
        let len = (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]).sqrt();
        if len > 0.001 {
            velocity[0] /= len;
            velocity[1] /= len;
            velocity[2] /= len;
        }
        
        // Apply movement
        let pos = self.camera.position;
        self.camera.position = [
            pos[0] + velocity[0] * self.speed * dt,
            pos[1] + velocity[1] * self.speed * dt,
            pos[2] + velocity[2] * self.speed * dt,
        ];
        
        self.update_camera_vectors();
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
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    
    world: World,
    camera_controller: CameraController,
    visibility_cache: VisibilityCache,
    last_frame: Instant,
    frame_count: u64,
    last_fps_print: Instant,
    
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        // Create world with test data
        let mut world = World::new();
        
        println!("Creating test world...");
        
        // Ground plane
        for x in -50..50 {
            for z in -50..50 {
                if (x + z) % 3 == 0 {
                    world.set(WorldPos::new(x, 0, z), 1);
                }
            }
        }
        
        // Towers
        for i in 0..5 {
            let x = (i - 2) * 20;
            for y in 1..=(10 + i * 3) {
                world.set(WorldPos::new(x, y, 0), 2);
            }
        }
        
        // Scattered structures
        for i in 0..10 {
            let angle = i as f32 * std::f32::consts::PI * 2.0 / 10.0;
            let radius = 30.0;
            let x = (angle.cos() * radius) as i64;
            let z = (angle.sin() * radius) as i64;
            
            for y in 0..5 {
                world.set(WorldPos::new(x, y, z), 3 + (i % 3) as u8);
            }
        }
        
        let voxel_count: u64 = world.chunks().map(|(_, c)| c.count()).sum();
        println!("World created with {} voxels in {} chunks", voxel_count, world.chunks().count());
        
        Self {
            window: None,
            surface: None,
            device: None,
            queue: None,
            config: None,
            render_pipeline: None,
            uniform_buffer: None,
            bind_group: None,
            world,
            camera_controller: CameraController::new([0.0, 20.0, 50.0]),
            visibility_cache: VisibilityCache::new(),
            last_frame: Instant::now(),
            frame_count: 0,
            last_fps_print: Instant::now(),
            mouse_pressed: false,
            last_mouse_pos: None,
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
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<VoxelInstanceRaw>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32],
                    }
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
        
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: 128, // MVP matrix
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        
        self.window = Some(window);
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);
        self.render_pipeline = Some(render_pipeline);
        self.uniform_buffer = Some(uniform_buffer);
        self.bind_group = Some(bind_group);
        
        println!("wgpu initialized");
    }
    
    fn render(&mut self) {
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let config = self.config.as_ref().unwrap();
        
        // Update camera
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        
        self.camera_controller.update(dt);
        
        // Cull visible voxels using cached visibility
        let cull_start = Instant::now();
        let visible = self.visibility_cache.update(&self.camera_controller.camera, &self.world);
        let cull_time = cull_start.elapsed();
        
        // Convert to GPU instances
        let instances: Vec<VoxelInstanceRaw> = visible
            .iter()
            .map(|v| VoxelInstanceRaw {
                position: [v.position[0] as f32, v.position[1] as f32, v.position[2] as f32],
                voxel_type: v.voxel_type as u32,
            })
            .collect();
        
        // Update or create instance buffer
        if instances.is_empty() {
            return; // Nothing to render
        }
        
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (instances.len() * std::mem::size_of::<VoxelInstanceRaw>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instances));
        
        // Get surface texture
        let output = match surface.get_current_texture() {
            Ok(texture) => texture,
            Err(_) => {
                // Recreate surface
                surface.configure(device, config);
                return;
            }
        };
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
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
        
        // Create MVP matrix
        let aspect = config.width as f32 / config.height as f32;
        let projection = perspective(self.camera_controller.camera.fov, aspect, 0.1, 1000.0);
        let view_matrix = look_at(
            self.camera_controller.camera.position,
            add_vec(self.camera_controller.camera.position, self.camera_controller.camera.forward),
            self.camera_controller.camera.up,
        );
        let mvp = mat_mul(&projection, &view_matrix);
        
        queue.write_buffer(self.uniform_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&mvp));
        
        // Create command encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..36, 0..instances.len() as u32); // 36 vertices for a cube
        }
        
        queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        // Stats
        self.frame_count += 1;
        if now.duration_since(self.last_fps_print).as_secs() >= 1 {
            println!(
                "FPS: {}, Visible voxels: {}, Cull time: {:.2}ms",
                self.frame_count,
                instances.len(),
                cull_time.as_secs_f64() * 1000.0
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
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Close requested");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(key),
                    state,
                    ..
                },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                self.camera_controller.process_keyboard(key, pressed);
                
                if key == KeyCode::Escape && pressed {
                    event_loop.exit();
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => {
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
                    if let (Some(config), Some(surface), Some(device)) = 
                        (&mut self.config, &self.surface, &self.device) {
                        config.width = new_size.width;
                        config.height = new_size.height;
                        surface.configure(device, config);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

// Matrix math helpers
fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), -1.0],
        [0.0, 0.0, (2.0 * far * near) / (near - far), 0.0],
    ]
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize_vec(sub_vec(center, eye));
    let s = normalize_vec(cross_vec(f, up));
    let u = cross_vec(s, f);
    
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot_vec(s, eye), -dot_vec(u, eye), dot_vec(f, eye), 1.0],
    ]
}

fn mat_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn add_vec(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub_vec(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross_vec(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot_vec(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize_vec(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

fn main() {
    env_logger::init();
    
    println!("Hierarchical Voxel Viewer");
    println!("=========================");
    println!("Controls:");
    println!("  WASD - Move");
    println!("  Space/Shift - Up/Down");
    println!("  Right Mouse - Look around");
    println!("  ESC - Quit\n");
    
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
