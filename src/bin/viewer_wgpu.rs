use std::iter;
use std::sync::Arc;
use std::collections::HashSet;
use env_logger;
use winit::{application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, window::Window};
use voxelot::VoxelWorld;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// Represents a visible voxel after plane-based culling
#[derive(Debug, Clone)]
struct VisibleVoxel {
    x: i64,
    y: i64,
    z: i64,
    voxel_type: u16,
    depth: f32,  // Distance from camera for depth sorting
}

/// Simple orthographic camera for top-down view
struct Camera {
    position: (f32, f32, f32),
    direction: (f32, f32, f32),  // Normalized view direction
}

impl Camera {
    fn new() -> Self {
        Self {
            position: (64.0, 64.0, 100.0),  // Above the center of our world
            direction: (0.0, 0.0, -1.0),     // Looking down
        }
    }

    fn with_position(x: f32, y: f32, z: f32) -> Self {
        Self { position: (x, y, z), direction: (0.0, 0.0, -1.0) }
    }

    /// Plane-based culling using pure VoxelWorld bitmap operations
    /// 
    /// **DEMONSTRATES:** Using VoxelWorld.extract_z_plane() for plane-based culling
    /// 
    /// **CURRENT BOTTLENECK:** extract_z_plane() iterates all voxels (slow)
    /// 
    /// **YOUR INSIGHTS ARE CORRECT:** To make this fast, we need:
    /// 
    /// 1. **Efficient Plane Creation** - Create planes directly from bitmaps:
    ///    - Don't iterate through coordinates
    ///    - Instead: construct Chunk/Region bitmaps directly for a given Z-slice
    ///    - Use bitmap range operations: "all indices where z_component == target_z"
    ///    
    /// 2. **Plane Movement via Bitmap Operations**:
    ///    - Precreate view frustum as VoxelWorld (one-time cost)
    ///    - For each depth: intersect (frustum ∩ world) at that Z-slice
    ///    - Shift/transform using bitmap index arithmetic
    ///    
    /// 3. **2D Hit Mask as Bitmap**:
    ///    - Use croaring::Bitmap for (x,y) columns (not VoxelWorld)
    ///    - Map (x,y) → flat_index for 2D bitmap
    ///    - Or: add VoxelWorld2D struct with just (x,y) support
    ///    
    /// **NEXT STEPS to optimize:**
    /// - Add VoxelWorld::extract_z_plane_fast() using direct bitmap manipulation
    /// - Add VoxelWorld::create_plane_at_z() using bitmap construction
    /// - Consider VoxelWorld2D for hit masks
    ///    
    /// **CURRENT PERFORMANCE:** ~14.5s (limited by extract_z_plane iteration)
    /// **TARGET PERFORMANCE:** <100ms (with direct bitmap operations)
    #[allow(dead_code)]
    fn cull_visible_with_planes(&self, world: &VoxelWorld, max_depth: i64) -> Vec<VisibleVoxel> {
        let mut visible = Vec::new();
        
        let start_z = self.position.2 as i64;
        
        // Define view bounds
        let view_width = WIDTH as i64;
        let view_height = HEIGHT as i64;
        let min_x = (self.position.0 as i64).saturating_sub(view_width / 2);
        let max_x = (self.position.0 as i64).saturating_add(view_width / 2);
        let min_y = (self.position.1 as i64).saturating_sub(view_height / 2);
        let max_y = (self.position.1 as i64).saturating_add(view_height / 2);
        
        // Track occluded (x,y) columns - this is 2D, so HashSet is appropriate
        // A true pure-bitmap solution would use a 2D Roaring Bitmap here
        let mut hit_columns: HashSet<(i64, i64)> = HashSet::new();
        
        // Sweep planes from camera toward ground
        for z in (0..=start_z).rev() {
            if z < 0 || z > max_depth {
                continue;
            }
            
            let depth = (start_z - z) as f32;
            
            // Extract all voxels at this depth using FAST bitmap operations
            let depth_plane = world.extract_z_plane_fast(z);
            
            // Process visible voxels at this depth
            depth_plane.for_each_voxel(|x, y, z_hit, voxel_type| {
                // Check view bounds and occlusion
                if x >= min_x && x <= max_x && y >= min_y && y <= max_y {
                    let xy_key = (x, y);
                    
                    if !hit_columns.contains(&xy_key) {
                        visible.push(VisibleVoxel {
                            x,
                            y,
                            z: z_hit,
                            voxel_type,
                            depth,
                        });
                        
                        // Mark column as occluded
                        hit_columns.insert(xy_key);
                    }
                }
            });
            
            // Early exit if all view columns are occluded
            let total_columns = ((max_x - min_x + 1) * (max_y - min_y + 1)) as usize;
            if hit_columns.len() >= total_columns {
                break;
            }
        }
        
        visible
    }

    /// Plane-based culling using DIRECT voxel extraction (FASTEST)
    /// Returns Vec<(x,y,z,type)> instead of VoxelWorld - avoids set_voxel overhead
    #[allow(dead_code)]
    fn cull_visible_direct(&self, world: &VoxelWorld, max_depth: i64) -> Vec<VisibleVoxel> {
        let mut visible = Vec::new();
        
        let start_z = self.position.2 as i64;
        
        // Define view bounds
        let view_width = WIDTH as i64;
        let view_height = HEIGHT as i64;
        let min_x = (self.position.0 as i64).saturating_sub(view_width / 2);
        let max_x = (self.position.0 as i64).saturating_add(view_width / 2);
        let min_y = (self.position.1 as i64).saturating_sub(view_height / 2);
        let max_y = (self.position.1 as i64).saturating_add(view_height / 2);
        
        // Track occluded (x,y) columns
        let mut hit_columns: HashSet<(i64, i64)> = HashSet::new();
        
        // Sweep planes from camera toward ground
        for z in (0..=start_z).rev() {
            if z < 0 || z > max_depth {
                continue;
            }
            
            let depth = (start_z - z) as f32;
            
            // Extract voxels directly - no VoxelWorld construction overhead!
            let voxels_at_depth = world.extract_z_plane_direct(z);
            
            // Process visible voxels at this depth
            for (x, y, z_hit, voxel_type) in voxels_at_depth {
                // Check view bounds and occlusion
                if x >= min_x && x <= max_x && y >= min_y && y <= max_y {
                    let xy_key = (x, y);
                    
                    if !hit_columns.contains(&xy_key) {
                        visible.push(VisibleVoxel {
                            x,
                            y,
                            z: z_hit,
                            voxel_type,
                            depth,
                        });
                        
                        // Mark column as occluded
                        hit_columns.insert(xy_key);
                    }
                }
            }
            
            // Early exit if all view columns are occluded
            let total_columns = ((max_x - min_x + 1) * (max_y - min_y + 1)) as usize;
            if hit_columns.len() >= total_columns {
                break;
            }
        }
        
        visible
    }

    /// TRUE PLANE PROPAGATION using VoxelWorld bitmap operations
    /// 
    /// Algorithm:
    /// 1. Create a FULL plane VoxelWorld (all (x,y) positions marked as "visible")
    /// 2. For each Z-level from camera toward ground:
    ///    a. Intersect: hits = plane ∩ world_slice
    ///    b. Record visible voxels from hits
    ///    c. Remove hits from plane: plane = plane - hits (those columns now occluded)
    ///    d. Stop if plane is empty (all columns occluded)
    /// 
    /// This uses a cached 2D (x,y) mask + per-z plane build with pure bitmap ops
    fn cull_visible_propagating(&self, world: &VoxelWorld, max_depth: i64) -> Vec<VisibleVoxel> {
        let mut visible = Vec::new();
        
        let start_z = self.position.2 as i64;
        
        // Define view bounds
        let view_width = WIDTH as i64;
        let view_height = HEIGHT as i64;
        let min_x = (self.position.0 as i64).saturating_sub(view_width / 2);
        let max_x = (self.position.0 as i64).saturating_add(view_width / 2);
        let min_y = (self.position.1 as i64).saturating_sub(view_height / 2);
        let max_y = (self.position.1 as i64).saturating_add(view_height / 2);
        
        // Build cached view mask once
        let mut mask = VoxelWorld::build_view_plane_mask(min_x, max_x, min_y, max_y);
        
        // Propagate from camera toward ground
            for z in (0..=start_z.min(max_depth)).rev() {
                let depth = (start_z - z) as f32;
                VoxelWorld::intersect_mask_with_world_z_collect(&mut mask, world, z, |x, y, z_hit, voxel_type| {
                    visible.push(VisibleVoxel { x, y, z: z_hit, voxel_type, depth });
                });
                if VoxelWorld::mask_is_empty(&mask) { break; }
            }
        
        visible
    }

    /// Plane-based culling using bitmap iteration (optimized for orthographic)
    /// 
    /// This uses the efficient bitmap iteration from for_each_voxel():
    /// - Iterates only actual voxels (sparse, efficient)
    /// - Organizes by Z depth using BTreeMap
    /// - Applies occlusion in sorted order
    /// 
    /// Trade-offs:
    /// - Pro: Very fast for simple orthographic views (~200ms for 16k voxels)
    /// - Pro: Only visits existing voxels (no empty space checks)
    /// - Pro: Uses bitmap iteration at Chunk/Region level internally
    /// - Con: Doesn't demonstrate full VoxelWorld-level bitmap operations
    /// - Con: Requires sorting step (BTreeMap)
    /// 
    /// Best for: Simple orthographic views, top-down rendering, current use case
    fn cull_visible_orthographic(&self, world: &VoxelWorld, max_depth: i64) -> Vec<VisibleVoxel> {
        let mut visible = Vec::new();
        
        let start_z = self.position.2 as i64;
        
        // Define the view bounds
        let view_width = WIDTH as i64;
        let view_height = HEIGHT as i64;
        let min_x = (self.position.0 as i64).saturating_sub(view_width / 2);
        let max_x = (self.position.0 as i64).saturating_add(view_width / 2);
        let min_y = (self.position.1 as i64).saturating_sub(view_height / 2);
        let max_y = (self.position.1 as i64).saturating_add(view_height / 2);
        
        // Track which (x,y) positions have already been hit (for occlusion)
        let mut hit_map: HashSet<(i64, i64)> = HashSet::new();
        
            // Use bitmap iteration to gather visible voxels into Z-buckets (faster than BTreeMap)
            let z_max = start_z.min(max_depth).max(0) as usize;
            let mut buckets: Vec<Vec<(i64, i64, u16)>> = vec![Vec::new(); z_max + 1];

            world.for_each_voxel(|x, y, z, voxel_type| {
                if x >= min_x && x <= max_x && y >= min_y && y <= max_y && z >= 0 && z <= start_z && z <= max_depth {
                    let idx = z as usize;
                    if idx <= z_max { buckets[idx].push((x, y, voxel_type)); }
                }
            });

            // Process from high z to low z
            for z in (0..=z_max).rev() {
                for &(x, y, voxel_type) in &buckets[z] {
                    let key = (x, y);
                    if !hit_map.contains(&key) {
                        let z_i64 = z as i64;
                        visible.push(VisibleVoxel { x, y, z: z_i64, voxel_type, depth: (start_z - z_i64) as f32 });
                        hit_map.insert(key);
                    }
                }
            }
        
        visible
    }
}


struct App {
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    texture: Option<wgpu::Texture>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    texture_bind_group: Option<wgpu::BindGroup>,
    world: VoxelWorld,
    camera: Camera,
    cpu_buffer: Vec<u8>,
    buffer_dirty: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(event_loop.create_window(Window::default_attributes()
            .with_title("voxelot wgpu viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH as f64, HEIGHT as f64)))
            .unwrap());

        // Init wgpu
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { 
            power_preference: wgpu::PowerPreference::HighPerformance, 
            compatible_surface: Some(&surface), 
            force_fallback_adapter: false 
        })).expect("No adapter");
        
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { 
            required_features: wgpu::Features::empty(), 
            required_limits: wgpu::Limits::default(), 
            label: None,
            memory_hints: wgpu::MemoryHints::default(),
            trace: Default::default(),
            experimental_features: Default::default(),
        })).expect("Failed to request device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];

        let config = wgpu::SurfaceConfiguration { 
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, 
            format: surface_format, 
            width: WIDTH, 
            height: HEIGHT, 
            present_mode: wgpu::PresentMode::Fifo, 
            alpha_mode: surface_caps.alpha_modes[0], 
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Simple fullscreen pipeline
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/fullscreen.wgsl"));

        let texture_size = wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor { 
            label: Some("frame_texture"), 
            size: texture_size, 
            mip_level_count: 1, 
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::Rgba8UnormSrgb, 
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, 
            view_formats: &[] 
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            entries: &[
                wgpu::BindGroupLayoutEntry { 
                    binding: 0, 
                    visibility: wgpu::ShaderStages::FRAGMENT, 
                    ty: wgpu::BindingType::Texture { 
                        multisampled: false, 
                        view_dimension: wgpu::TextureViewDimension::D2, 
                        sample_type: wgpu::TextureSampleType::Float { filterable: true } 
                    }, 
                    count: None 
                },
                wgpu::BindGroupLayoutEntry { 
                    binding: 1, 
                    visibility: wgpu::ShaderStages::FRAGMENT, 
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), 
                    count: None 
                },
            ], 
            label: Some("texture_bind_group_layout") 
        });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            layout: &texture_bind_group_layout, 
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ], 
            label: Some("texture_bind_group") 
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: Some("pipeline_layout"), 
            bind_group_layouts: &[&texture_bind_group_layout], 
            push_constant_ranges: &[] 
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"), 
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
                    blend: Some(wgpu::BlendState::REPLACE), 
                    write_mask: wgpu::ColorWrites::ALL 
                })],
                compilation_options: Default::default(),
            }), 
            primitive: wgpu::PrimitiveState::default(), 
            depth_stencil: None, 
            multisample: wgpu::MultisampleState::default(), 
            multiview: None,
            cache: None,
        });

        // Upload the pre-rendered buffer immediately
        let texture_size = wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { 
                texture: &texture, 
                mip_level: 0, 
                origin: wgpu::Origin3d::ZERO, 
                aspect: wgpu::TextureAspect::All 
            }, 
            &self.cpu_buffer, 
            wgpu::TexelCopyBufferLayout { 
                offset: 0, 
                bytes_per_row: Some(WIDTH * 4), 
                rows_per_image: Some(HEIGHT), 
            }, 
            texture_size
        );

        self.window = Some(window.clone());
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.texture = Some(texture);
        self.render_pipeline = Some(render_pipeline);
        self.texture_bind_group = Some(texture_bind_group);
        
        // Request initial draw
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let Some(ref device) = self.device else { return };
                let Some(ref queue) = self.queue else { return };
                let Some(ref texture) = self.texture else { return };
                let Some(ref surface) = self.surface else { return };
                let Some(ref render_pipeline) = self.render_pipeline else { return };
                let Some(ref texture_bind_group) = self.texture_bind_group else { return };

                // Only regenerate buffer if dirty
                if self.buffer_dirty {
                    // Clear buffer
                    self.cpu_buffer.fill(0);
                    
                    // Use plane-based culling to find visible voxels
                    let visible_voxels = self.camera.cull_visible_orthographic(&self.world, 100);
                    
                    // Render the visible voxels to the CPU buffer
                    for voxel in &visible_voxels {
                        // Map voxel world coordinates to screen coordinates
                        let screen_x = voxel.x;
                        let screen_y = voxel.y;
                        
                        // Check if within screen bounds
                        if screen_x >= 0 && screen_x < WIDTH as i64 && screen_y >= 0 && screen_y < HEIGHT as i64 {
                            let color = match voxel.voxel_type % 6 {
                                0 => [0xFF, 0x88, 0x88, 255],
                                1 => [0xFF, 0xAA, 0x66, 255],
                                2 => [0xFF, 0x66, 0xCC, 255],
                                3 => [0xFF, 0xCC, 0x66, 255],
                                4 => [0xFF, 0x88, 0xCC, 255],
                                _ => [0xFF, 0xDD, 0xDD, 255],
                            };
                            
                            let i = ((screen_y as u32 * WIDTH + screen_x as u32) * 4) as usize;
                            self.cpu_buffer[i..i+4].copy_from_slice(&color);
                        }
                    }

                    let texture_size = wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 };

                    // Upload to texture
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo { 
                            texture, 
                            mip_level: 0, 
                            origin: wgpu::Origin3d::ZERO, 
                            aspect: wgpu::TextureAspect::All 
                        }, 
                        &self.cpu_buffer, 
                        wgpu::TexelCopyBufferLayout { 
                            offset: 0, 
                            bytes_per_row: Some(WIDTH * 4), 
                            rows_per_image: Some(HEIGHT), 
                        }, 
                        texture_size
                    );

                    self.buffer_dirty = false;
                }

                // Render
                let frame = match surface.get_current_texture() { 
                    Ok(f) => f, 
                    Err(e) => { 
                        eprintln!("get_current_texture error: {:?}", e); 
                        return; 
                    } 
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { 
                        label: Some("rpass"), 
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { 
                            view: &view, 
                            resolve_target: None, 
                            ops: wgpu::Operations { 
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), 
                                store: wgpu::StoreOp::Store 
                            },
                            depth_slice: None,
                        })], 
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rpass.set_pipeline(render_pipeline);
                    rpass.set_bind_group(0, texture_bind_group, &[]);
                    rpass.draw(0..3, 0..1);
                }
                queue.submit(iter::once(encoder.finish()));
                frame.present();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Only request redraw if buffer needs updating
        if self.buffer_dirty {
            if let Some(ref window) = self.window {
                window.request_redraw();
            }
        }
    }
}

fn main() {
    env_logger::init();

    let start = std::time::Instant::now();
    
    // World and a CPU buffer
    let mut world = VoxelWorld::new();
    
    // Helper to generate a hill world of given size and max height
    fn generate_hill_world(size: i32, height: i32) -> VoxelWorld {
        let mut w = VoxelWorld::new();
        let center = (size as f32) / 2.0;
        for x in 0..size {
            for y in 0..size {
                let dx = (x as f32 - center).abs();
                let dy = (y as f32 - center).abs();
                let dist = (dx * dx + dy * dy).sqrt();
                let max_z = (height as f32 - dist / 8.0).max(0.0) as i32;
                if max_z > 0 {
                    for z in 0..max_z { w.set_voxel(x as i64, y as i64, z as i64, ((z % 4) + 1) as u16); }
                }
            }
        }
        w
    }

    // Scaling benchmark across world sizes (mask propagation vs BTree iteration)
    println!("\n=== Scaling Benchmark (world sizes) ===");
    let configs = [ (128, 16), (256, 24) ];
    for (size, height) in configs {
        let gen_start = std::time::Instant::now();
        let world_b = generate_hill_world(size, height);
        let gen_time = gen_start.elapsed();
        let cam = Camera::with_position((size as f32)/2.0, (size as f32)/2.0, 100.0);

        let t0 = std::time::Instant::now();
        let vis_prop = cam.cull_visible_propagating(&world_b, height as i64 + 100);
        let dt_prop = t0.elapsed();

        let t1 = std::time::Instant::now();
        let vis_bt = cam.cull_visible_orthographic(&world_b, height as i64 + 100);
        let dt_bt = t1.elapsed();

        println!(
            "size {:>3}×{:>3} h={:>2} | gen {:?} | prop {:>8?} ({} vox) | iter {:>8?} ({} vox)",
            size, size, height, gen_time, dt_prop, vis_prop.len(), dt_bt, vis_bt.len()
        );
    }
    println!("========================================\n");

    // Create a 128x128 hill
    for x in 0..128 { 
        let dx = x as f32 - 64.0;
        for y in 0..128 { 
            let dy = y as f32 - 64.0;
            let h = (dx.hypot(dy) / 8.0) as i64;
            let max_z = 16 - h;  // Smaller max height too
            if max_z > 0 {
                for z in 0..max_z { 
                    world.set_voxel(x as i64, y as i64, z as i64, ((z % 4) + 1) as u16); 
                }
            }
        } 
    }
    
    println!("World generation took: {:?}", start.elapsed());
    
    // Create camera
    let camera = Camera::new();
    
    // Pre-render the buffer before creating the window using plane-based culling
    let mut cpu_buffer: Vec<u8> = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    
    // Performance comparison (set to true to benchmark all methods)
    let run_benchmark = true;
    
    if run_benchmark {
        println!("\n=== Performance Comparison ===");
        println!("Testing 4 plane-based culling approaches:\n");
        
        // Test 1: TRUE plane propagation (pure bitmap operations!)
        let cull_start = std::time::Instant::now();
        let visible_voxels_prop = camera.cull_visible_propagating(&world, 100);
        let prop_time = cull_start.elapsed();
        println!("1. TRUE Plane Propagation:     {} voxels in {:?} ⭐ PURE BITMAP OPS", 
                 visible_voxels_prop.len(), prop_time);
        println!("   - Creates FULL plane, propagates with intersect/difference");
        
        // Test 2: Bitmap iteration with BTreeMap sorting
        let cull_start2 = std::time::Instant::now();
        let visible_voxels = camera.cull_visible_orthographic(&world, 100);
        let iter_time = cull_start2.elapsed();
        println!("2. Bitmap iteration (BTreeMap): {} voxels in {:?}", 
                 visible_voxels.len(), iter_time);
        
        // Test 3: DIRECT voxel extraction (no VoxelWorld construction)
        let cull_start3 = std::time::Instant::now();
        let visible_voxels_direct = camera.cull_visible_direct(&world, 100);
        let direct_time = cull_start3.elapsed();
        println!("3. Direct plane extraction:    {} voxels in {:?}", 
                 visible_voxels_direct.len(), direct_time);
        
        // Test 4: VoxelWorld plane extraction (demonstrates bitmap ops)
        let cull_start4 = std::time::Instant::now();
        let visible_voxels_planes = camera.cull_visible_with_planes(&world, 100);
        let planes_time = cull_start4.elapsed();
        println!("4. VoxelWorld plane culling:   {} voxels in {:?}", 
                 visible_voxels_planes.len(), planes_time);
        
        println!("\n🎯 True Plane Propagation:");
        println!("   - Start with FULL plane (all view columns marked visible)");
        println!("   - For each Z: hits = plane ∩ world_slice");
        println!("   - Remove hits from plane (those columns now occluded)");
        println!("   - Stop when plane is empty (all columns hit)");
        println!("   - ✅ PURE VoxelWorld bitmap operations (intersect, difference)");
        println!("");
        println!("   Performance note:");
        println!("   - Creates new VoxelWorld for each intersect/difference (~100 times)");
        println!("   - This overhead makes it slower for simple orthographic case");
        println!("   - Would excel for: perspective frustums, dynamic occlusion, complex queries");
        println!("   - Demonstrates the CONCEPT of pure bitmap propagation!");
        println!("==============================\n");
    }
    
    let cull_start = std::time::Instant::now();
    let visible_voxels = camera.cull_visible_orthographic(&world, 100);
    println!("Plane-based culling found {} visible voxels in {:?}", 
             visible_voxels.len(), cull_start.elapsed());
    
    // Render the visible voxels to the CPU buffer
    let render_start = std::time::Instant::now();
    
    for voxel in &visible_voxels {
        // Map voxel world coordinates to screen coordinates
        // For simple orthographic top-down: x,y map directly to screen x,y
        let screen_x = voxel.x;
        let screen_y = voxel.y;
        
        // Check if within screen bounds
        if screen_x >= 0 && screen_x < WIDTH as i64 && screen_y >= 0 && screen_y < HEIGHT as i64 {
            let color = match voxel.voxel_type % 6 {
                0 => [0xFF, 0x88, 0x88, 255],
                1 => [0xFF, 0xAA, 0x66, 255],
                2 => [0xFF, 0x66, 0xCC, 255],
                3 => [0xFF, 0xCC, 0x66, 255],
                4 => [0xFF, 0x88, 0xCC, 255],
                _ => [0xFF, 0xDD, 0xDD, 255],
            };
            
            let i = ((screen_y as u32 * WIDTH + screen_x as u32) * 4) as usize;
            cpu_buffer[i..i+4].copy_from_slice(&color);
        }
    }
    
    println!("Rendering {} voxels took: {:?}", visible_voxels.len(), render_start.elapsed());
    println!("Total startup time: {:?}", start.elapsed());

    let mut app = App {
        window: None,
        surface: None,
        device: None,
        queue: None,
        texture: None,
        render_pipeline: None,
        texture_bind_group: None,
        world,
        camera,
        cpu_buffer,
        buffer_dirty: false,  // Already rendered, so not dirty
    };

    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
