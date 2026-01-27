//! Configuration management for Voxelot

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MacosHdrColorspace {
    /// Extended linear sRGB (matches typical linear rendering pipelines).
    ExtendedLinearSrgb,
    /// Extended linear Display P3 (wide gamut; requires rendering pipeline to be in P3 to avoid hue shifts).
    ExtendedLinearDisplayP3,
    /// Extended linear ITU-R BT.2020 (very wide gamut HDR; best for HDR content).
    ExtendedLinearItur2020,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub world: WorldConfig,
    #[serde(default)]
    pub rendering: RenderingConfig,
    #[serde(default)]
    pub atmosphere: AtmosphereConfig,
    #[serde(default)]
    pub effects: EffectsConfig,
    #[serde(default)]
    pub shadows: ShadowConfig,
    #[serde(default)]
    pub performance: PerformanceConfig,
    // (debug config removed) debug flags were unused and have been removed to simplify configuration
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    #[serde(default = "default_world_file")]
    pub file: String,
    #[serde(default = "default_palette_file")]
    pub palette: String,
    #[serde(default = "default_skybox_file")]
    pub skybox: String,
    #[serde(default = "default_camera_position")]
    pub camera_position: [f32; 3],
    #[serde(default = "default_water_level")]
    pub water_level: f32,
    #[serde(default = "default_water_visibility")]
    pub water_visibility: f32,
    // Starting mode: "camera" (default) or "boat". In the future this may become more
    // descriptive (e.g., specify vessel name) to support multiple vessel types.
    #[serde(default = "default_start_mode")]
    pub start_mode: String,
    #[serde(default)]
    pub start_vessel: Option<String>,
}

fn default_start_mode() -> String {
    // Default to starting with the free camera for new configs (backwards compatible with "boat").
    "camera".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingConfig {
    #[serde(default = "default_chunk_lod_distance")]
    pub chunk_lod_distance: f32,
    #[serde(default = "default_fov")]
    pub fov_degrees: f32,
    #[serde(default = "default_near_plane")]
    pub near_plane: f32,
    #[serde(default = "default_far_plane")]
    pub far_plane: f32,
    /// Pixel-size threshold below which distant chunks are rendered as impostors.
    #[serde(default = "default_impostor_pixel_threshold")]
    pub impostor_pixel_threshold: f32,
    /// Pixel size used when rendering impostors (single-pixel by default).
    #[serde(default = "default_impostor_pixel_size")]
    pub impostor_pixel_size: f32,
    #[serde(default = "default_camera_speed")]
    pub camera_speed_multiplier: f32,
    #[serde(default = "default_window_width")]
    pub window_width: u32,
    #[serde(default = "default_window_height")]
    pub window_height: u32,

    /// Enlarges the view frustum used for CPU/GPU culling (but does not change the rendered FOV).
    ///
    /// Value is a fractional margin applied to the tangent of the half-FOV. For example, 0.15
    /// makes culling behave as if the camera FOV were ~15% wider in X/Y.
    ///
    /// This is useful for screen-space effects (e.g., water reflections) that may need geometry
    /// slightly outside the visible viewport.
    #[serde(default = "default_culling_overscan")]
    pub culling_overscan: f32,

    /// Renders the scene with a wider effective FOV into a larger offscreen buffer, then crops to
    /// the window for presentation. This provides extra off-screen scene color/depth for
    /// screen-space effects (notably water reflections / SSR) so they don't hard-cut at the
    /// visible screen edges.
    ///
    /// Value is a fractional margin: 0.15 means render at ~15% wider tangent half-FOV in X/Y.
    /// Increasing this has a GPU cost proportional to the increased pixel count.
    #[serde(default = "default_render_overscan")]
    pub render_overscan: f32,

    /// Enable HDR presentation on macOS (EDR + extended linear Display P3) when supported.
    /// If disabled, the viewer will prefer an sRGB swapchain format.
    #[serde(default = "default_macos_hdr")]
    pub macos_hdr: bool,

    /// Multiplier applied to final exposure when macOS HDR presentation is active.
    /// Useful because HDR presentation may look perceptually darker than SDR.
    #[serde(default = "default_macos_hdr_exposure_boost")]
    pub macos_hdr_exposure_boost: f32,

    /// Which colorspace the macOS CAMetalLayer should advertise when HDR is active.
    /// If your renderer outputs linear sRGB values, prefer `extended_linear_srgb` to avoid hue shifts.
    #[serde(default = "default_macos_hdr_colorspace")]
    pub macos_hdr_colorspace: MacosHdrColorspace,
}

fn default_window_width() -> u32 {
    1280
}

fn default_window_height() -> u32 {
    720
}

fn default_culling_overscan() -> f32 {
    0.0
}

fn default_render_overscan() -> f32 {
    0.0
}

fn default_macos_hdr() -> bool {
    true
}

fn default_macos_hdr_exposure_boost() -> f32 {
    1.0
}

fn default_macos_hdr_colorspace() -> MacosHdrColorspace {
    MacosHdrColorspace::ExtendedLinearSrgb
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphereConfig {
    #[serde(default = "default_fog_density")]
    pub fog_density: f32,
    #[serde(default = "default_time_of_day")]
    pub time_of_day: f32,
    #[serde(default = "default_time_paused")]
    pub time_paused: bool,
    #[serde(default = "default_night_skybox_brightness")]
    pub night_skybox_brightness: f32,
    #[serde(default = "default_horizon_fade_up")]
    pub horizon_fade_up: f32,
    #[serde(default = "default_horizon_fade_down")]
    pub horizon_fade_down: f32,
    #[serde(default = "default_skybox_fade_up")]
    pub skybox_fade_up: f32,
    #[serde(default = "default_skybox_fade_down")]
    pub skybox_fade_down: f32,
    #[serde(default = "default_skybox_min_saturation")]
    pub skybox_min_saturation: f32,
    #[serde(default = "default_skybox_tint")]
    pub skybox_night_tint: [f32; 3],
    #[serde(default = "default_skybox_tint_strength")]
    pub skybox_tint_strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsConfig {
    #[serde(default)]
    pub depth_of_field: DepthOfFieldConfig,
    #[serde(default)]
    pub bloom: BloomConfig,
    #[serde(default)]
    pub ssao: SsaoConfig,
    #[serde(default)]
    pub radiance_cascades: RadianceCascadesConfig,
    #[serde(default)]
    pub gi: GiConfig,
    #[serde(default)]
    pub ssr: SsrConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadianceCascadesConfig {
    #[serde(default = "default_radiance_cascades_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsrConfig {
    #[serde(default = "default_ssr_enabled")]
    pub enabled: bool,
    #[serde(default = "default_ssr_max_steps")]
    pub max_steps: u32,
    #[serde(default = "default_ssr_max_binary_steps")]
    pub max_binary_steps: u32,
    #[serde(default = "default_ssr_step_size")]
    pub step_size: f32,
    #[serde(default = "default_ssr_thickness")]
    pub thickness: f32,
}

fn default_ssr_enabled() -> bool {
    true
}

fn default_ssr_max_steps() -> u32 {
    32
}

fn default_ssr_max_binary_steps() -> u32 {
    4
}

fn default_ssr_step_size() -> f32 {
    0.5
}

fn default_ssr_thickness() -> f32 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsaoConfig {
    #[serde(default = "default_ssao_enabled")]
    pub enabled: bool,
    #[serde(default = "default_ssao_sample_count")]
    pub sample_count: u32,
    #[serde(default = "default_ssao_slice_count")]
    pub slice_count: u32,
    #[serde(default = "default_ssao_radius")]
    pub radius: f32,
    #[serde(default = "default_ssao_thickness")]
    pub thickness: f32,
    #[serde(default = "default_ssao_strength")]
    pub strength: f32,
    #[serde(default = "default_ssao_blur_enabled")]
    pub blur_enabled: bool,
    #[serde(default = "default_ssao_blur_radius")]
    pub blur_radius: f32,
    #[serde(default = "default_ssao_max_ao_distance")]
    pub max_ao_distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiConfig {
    #[serde(default = "default_gi_enabled")]
    pub enabled: bool,
    #[serde(default = "default_gi_indirect_scale")]
    pub indirect_scale: f32,
    #[serde(default = "default_gi_fade_distance")]
    pub fade_distance: f32,
    #[serde(default = "default_gi_fade_range")]
    pub fade_range: f32,
    #[serde(default = "default_gi_grid_dims")]
    pub grid_dims: [i32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthOfFieldConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_dof_focal_distance")]
    pub focal_distance: f32,
    #[serde(default = "default_dof_focal_range")]
    pub focal_range: f32,
    #[serde(default = "default_dof_blur_strength")]
    pub blur_strength: f32,
    #[serde(default = "default_dof_kawase_iterations")]
    pub kawase_iterations: usize,
    #[serde(default = "default_dof_kawase_offset")]
    pub kawase_offset: f32,
    #[serde(default = "default_dof_kawase_enabled")]
    pub kawase_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomConfig {
    #[serde(default = "default_bloom_enabled")]
    pub enabled: bool,
    #[serde(default = "default_bloom_threshold")]
    pub threshold: f32,
    #[serde(default = "default_bloom_knee")]
    pub knee: f32,
    #[serde(default = "default_bloom_intensity")]
    pub intensity: f32,
    #[serde(default = "default_bloom_strength")]
    pub bloom_strength: f32,
    #[serde(default = "default_bloom_saturation")]
    pub saturation_boost: f32,
    #[serde(default = "default_bloom_exposure")]
    pub exposure: f32,
    #[serde(default = "default_bloom_blur_radius")]
    pub blur_radius: f32,
    #[serde(default = "default_bloom_kawase_enabled")]
    pub kawase_enabled: bool,
    #[serde(default = "default_bloom_kawase_iterations")]
    pub kawase_iterations: usize,
    #[serde(default = "default_bloom_kawase_offset")]
    pub kawase_offset: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowConfig {
    #[serde(default = "default_shadow_map_size")]
    pub map_size: u32,
    #[serde(default = "default_shadow_darkness")]
    pub darkness: f32,
    #[serde(default = "default_backface_ambient_scale")]
    pub backface_ambient_scale: f32,
    #[serde(default = "default_shadow_pcf_radius")]
    pub pcf_radius: f32,
    #[serde(default = "default_shadow_pcf_poisson_samples")]
    pub pcf_poisson_samples: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    #[serde(default = "default_mesh_cache_mb")]
    pub mesh_cache_budget_mb: u64,
    #[serde(default = "default_mesh_workers")]
    pub mesh_worker_count: Option<usize>,
    #[serde(default = "default_mesh_upload_baseline")]
    pub mesh_upload_baseline: usize,
    #[serde(default = "default_envelope_distance")]
    pub envelope_distance: f32,
    #[serde(default = "default_envelope_fade_range")]
    pub envelope_fade_range: f32,
    #[serde(default = "default_max_envelope_distance")]
    pub max_envelope_distance: f32,
    #[serde(default = "default_hzb_enabled")]
    pub hzb_enabled: bool,
    #[serde(default = "default_fallback_detail_distance")]
    pub fallback_detail_distance: f32,
    #[serde(default = "default_mesh_priority_sort_interval_frames")]
    pub mesh_priority_sort_interval_frames: u64,
    #[serde(default = "default_mesh_buffer_pool_entries")]
    pub mesh_buffer_pool_entries: usize,
    #[serde(default = "default_mega_vertex_buffer_mb")]
    pub mega_vertex_buffer_mb: u64,
    #[serde(default = "default_mega_index_buffer_mb")]
    pub mega_index_buffer_mb: u64,
    #[serde(default = "default_max_draw_capacity")]
    pub max_draw_capacity: usize,
    #[serde(default = "default_render_scale")]
    pub render_scale: f32,
    /// Maximum number of GPU instances (chunks + fallback voxels). Limits memory usage.
    #[serde(default = "default_max_gpu_instances")]
    pub max_gpu_instances: usize,
    // Profiling is controlled at compile-time via Cargo features (`cpu-profiling` & `gpu-profiling`).
}



// Default values
fn default_world_file() -> String {
    "world_1.vhc".to_string()
}

fn default_palette_file() -> String {
    "worlds/palette.txt".to_string()
}

fn default_skybox_file() -> String {
    "worlds/skybox.hdr".to_string()
}

fn default_camera_position() -> [f32; 3] {
    [320.0, 100.0, 320.0]
}

fn default_water_level() -> f32 {
    16.0
}

fn default_water_visibility() -> f32 {
    20.0
}

// Note: subdivide/merge defaults removed; chunk_lod_distance is the canonical LOD cutoff.

fn default_chunk_lod_distance() -> f32 {
    800.0
}
// Note: subdivide/merge thresholds were removed; `chunk_lod_distance` is the canonical LOD cutoff.

fn default_fov() -> f32 {
    70.0
}

fn default_near_plane() -> f32 {
    0.1
}

fn default_far_plane() -> f32 {
    5000.0
}

fn default_impostor_pixel_threshold() -> f32 {
    1.5
}

fn default_impostor_pixel_size() -> f32 {
    1.0
}

fn default_camera_speed() -> f32 {
    1.0
}

fn default_fog_density() -> f32 {
    0.003000
}

fn default_time_of_day() -> f32 {
    0.5
}

fn default_time_paused() -> bool {
    false
}

fn default_night_skybox_brightness() -> f32 {
    0.02
}

fn default_horizon_fade_up() -> f32 {
    0.05
}

fn default_horizon_fade_down() -> f32 {
    0.25
}

fn default_skybox_fade_up() -> f32 {
    0.05
}

fn default_skybox_fade_down() -> f32 {
    0.6
}

fn default_skybox_min_saturation() -> f32 {
    0.16
}

fn default_skybox_tint() -> [f32; 3] {
    [0.09, 0.12, 0.24] // cool navy/indigo tint
}

fn default_skybox_tint_strength() -> f32 {
    0.5 // half strength at full night fade
}

fn default_dof_focal_distance() -> f32 {
    120.0
}

fn default_dof_focal_range() -> f32 {
    16.0
}

fn default_dof_blur_strength() -> f32 {
    1.6
}

fn default_dof_kawase_iterations() -> usize {
    3
}

fn default_dof_kawase_offset() -> f32 {
    1.0
}

fn default_dof_kawase_enabled() -> bool {
    false
}

fn default_bloom_enabled() -> bool {
    true
}

fn default_bloom_threshold() -> f32 {
    0.7
}

fn default_bloom_knee() -> f32 {
    0.6
}

fn default_bloom_intensity() -> f32 {
    1.8
}

fn default_bloom_strength() -> f32 {
    1.6
}

fn default_bloom_saturation() -> f32 {
    1.5
}

fn default_bloom_exposure() -> f32 {
    1.18
}

fn default_bloom_blur_radius() -> f32 {
    3.8
}

fn default_bloom_kawase_enabled() -> bool {
    true
}

fn default_bloom_kawase_iterations() -> usize {
    4
}

fn default_bloom_kawase_offset() -> f32 {
    0.5
}

fn default_ssao_enabled() -> bool {
    true
}
fn default_ssao_sample_count() -> u32 {
    8
}
fn default_ssao_slice_count() -> u32 {
    4
}
fn default_ssao_radius() -> f32 {
    4.0
}
fn default_ssao_thickness() -> f32 {
    0.5
}

fn default_ssao_strength() -> f32 {
    1.0
}

fn default_ssao_blur_enabled() -> bool {
    true
}

fn default_ssao_blur_radius() -> f32 {
    2.0
}

fn default_ssao_max_ao_distance() -> f32 {
    300.0
}

fn default_radiance_cascades_enabled() -> bool {
    true
}

fn default_gi_enabled() -> bool {
    true
}

fn default_gi_indirect_scale() -> f32 {
    0.003
}

fn default_gi_fade_distance() -> f32 {
    400.0
}

fn default_gi_fade_range() -> f32 {
    100.0
}

fn default_gi_grid_dims() -> [i32; 3] {
    [32, 16, 32]
}

fn default_shadow_map_size() -> u32 {
    4096
}

fn default_shadow_pcf_radius() -> f32 {
    // radius in texels for PCF (1.0 == 3x3 sampling with offsets -1/0/1)
    1.0
}

fn default_shadow_pcf_poisson_samples() -> u32 {
    // number of Poisson disk samples to use. 0 means disabled (use Gaussian 3x3).
    0
}

fn default_shadow_darkness() -> f32 {
    1.0
}

fn default_backface_ambient_scale() -> f32 {
    0.7
}

fn default_mesh_cache_mb() -> u64 {
    256
}

fn default_mesh_workers() -> Option<usize> {
    None // Auto-detect
}

fn default_mesh_upload_baseline() -> usize {
    4
}

fn default_envelope_distance() -> f32 {
    256.0
}

fn default_envelope_fade_range() -> f32 {
    32.0
}

fn default_max_envelope_distance() -> f32 {
    1000.0
}

fn default_hzb_enabled() -> bool {
    true
}

fn default_fallback_detail_distance() -> f32 {
    500.0
}

fn default_mesh_priority_sort_interval_frames() -> u64 {
    30
}

fn default_mesh_buffer_pool_entries() -> usize {
    256
}

fn default_mega_vertex_buffer_mb() -> u64 {
    512
}

fn default_mega_index_buffer_mb() -> u64 {
    256
}

fn default_max_draw_capacity() -> usize {
    20_000
}

fn default_render_scale() -> f32 {
    1.0
}

fn default_max_gpu_instances() -> usize {
    6_000_000 // ~384MB for VoxelInstanceRaw (64 bytes each)
}

// Profiling is controlled via cargo features; no runtime defaults.

// default_true was used by DebugConfig, which was removed.

// Implement Default for all configs
impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            file: default_world_file(),
            palette: default_palette_file(),
            skybox: default_skybox_file(),
            camera_position: default_camera_position(),
            water_level: default_water_level(),
            water_visibility: default_water_visibility(),
            start_mode: default_start_mode(),
            start_vessel: None,
        }
    }
}

impl Default for RenderingConfig {
    fn default() -> Self {
        Self {
            chunk_lod_distance: default_chunk_lod_distance(),
            fov_degrees: default_fov(),
            near_plane: default_near_plane(),
            far_plane: default_far_plane(),
            impostor_pixel_threshold: default_impostor_pixel_threshold(),
            impostor_pixel_size: default_impostor_pixel_size(),
            camera_speed_multiplier: default_camera_speed(),
            window_width: default_window_width(),
            window_height: default_window_height(),
            culling_overscan: default_culling_overscan(),
            render_overscan: default_render_overscan(),
            macos_hdr: default_macos_hdr(),
            macos_hdr_exposure_boost: default_macos_hdr_exposure_boost(),
            macos_hdr_colorspace: default_macos_hdr_colorspace(),
        }
    }
}

impl Default for AtmosphereConfig {
    fn default() -> Self {
        Self {
            fog_density: default_fog_density(),
            time_of_day: default_time_of_day(),
            time_paused: default_time_paused(),
            night_skybox_brightness: default_night_skybox_brightness(),
            horizon_fade_up: default_horizon_fade_up(),
            horizon_fade_down: default_horizon_fade_down(),
            skybox_fade_up: default_skybox_fade_up(),
            skybox_fade_down: default_skybox_fade_down(),
            skybox_min_saturation: default_skybox_min_saturation(),
            skybox_night_tint: default_skybox_tint(),
            skybox_tint_strength: default_skybox_tint_strength(),
        }
    }
}

impl Default for SsrConfig {
    fn default() -> Self {
        Self {
            enabled: default_ssr_enabled(),
            max_steps: default_ssr_max_steps(),
            max_binary_steps: default_ssr_max_binary_steps(),
            step_size: default_ssr_step_size(),
            thickness: default_ssr_thickness(),
        }
    }
}

impl Default for DepthOfFieldConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            focal_distance: default_dof_focal_distance(),
            focal_range: default_dof_focal_range(),
            blur_strength: default_dof_blur_strength(),
            kawase_iterations: default_dof_kawase_iterations(),
            kawase_offset: default_dof_kawase_offset(),
            kawase_enabled: default_dof_kawase_enabled(),
        }
    }
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            enabled: default_bloom_enabled(),
            threshold: default_bloom_threshold(),
            knee: default_bloom_knee(),
            intensity: default_bloom_intensity(),
            bloom_strength: default_bloom_strength(),
            saturation_boost: default_bloom_saturation(),
            exposure: default_bloom_exposure(),
            blur_radius: default_bloom_blur_radius(),
            kawase_enabled: default_bloom_kawase_enabled(),
            kawase_iterations: default_bloom_kawase_iterations(),
            kawase_offset: default_bloom_kawase_offset(),
        }
    }
}

impl Default for EffectsConfig {
    fn default() -> Self {
        Self {
            depth_of_field: DepthOfFieldConfig::default(),
            bloom: BloomConfig::default(),
            ssao: SsaoConfig::default(),
            radiance_cascades: RadianceCascadesConfig::default(),
            gi: GiConfig::default(),
            ssr: SsrConfig::default(),
        }
    }
}

impl Default for RadianceCascadesConfig {
    fn default() -> Self {
        Self {
            enabled: default_radiance_cascades_enabled(),
        }
    }
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            enabled: default_ssao_enabled(),
            sample_count: default_ssao_sample_count(),
            slice_count: default_ssao_slice_count(),
            radius: default_ssao_radius(),
            thickness: default_ssao_thickness(),
            strength: default_ssao_strength(),
            blur_enabled: default_ssao_blur_enabled(),
            blur_radius: default_ssao_blur_radius(),
            max_ao_distance: default_ssao_max_ao_distance(),
        }
    }
}

impl Default for GiConfig {
    fn default() -> Self {
        Self {
            enabled: default_gi_enabled(),
            indirect_scale: default_gi_indirect_scale(),
            fade_distance: default_gi_fade_distance(),
            fade_range: default_gi_fade_range(),
            grid_dims: default_gi_grid_dims(),
        }
    }
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            map_size: default_shadow_map_size(),
            darkness: default_shadow_darkness(),
            backface_ambient_scale: default_backface_ambient_scale(),
            pcf_radius: default_shadow_pcf_radius(),
            pcf_poisson_samples: default_shadow_pcf_poisson_samples(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            mesh_cache_budget_mb: default_mesh_cache_mb(),
            mesh_worker_count: default_mesh_workers(),
            mesh_upload_baseline: default_mesh_upload_baseline(),
            envelope_distance: default_envelope_distance(),
            envelope_fade_range: default_envelope_fade_range(),
            max_envelope_distance: default_max_envelope_distance(),
            hzb_enabled: default_hzb_enabled(),
            fallback_detail_distance: default_fallback_detail_distance(),
            mesh_priority_sort_interval_frames: default_mesh_priority_sort_interval_frames(),
            mesh_buffer_pool_entries: default_mesh_buffer_pool_entries(),
            mega_vertex_buffer_mb: default_mega_vertex_buffer_mb(),
            mega_index_buffer_mb: default_mega_index_buffer_mb(),
            max_draw_capacity: default_max_draw_capacity(),
            render_scale: default_render_scale(),
            max_gpu_instances: default_max_gpu_instances(),
        }
    }
}

// DebugConfig removed

impl Default for Config {
    fn default() -> Self {
        Self {
            world: WorldConfig::default(),
            rendering: RenderingConfig::default(),
            atmosphere: AtmosphereConfig::default(),
            effects: EffectsConfig::default(),
            shadows: ShadowConfig::default(),
            performance: PerformanceConfig::default(),
            // debug: no debug section; removed unused flags
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let contents = toml::to_string_pretty(self)?;
        fs::write(path, contents)?;
        Ok(())
    }

    /// Load configuration from default path, or create default if not found
    pub fn load_or_default(path: &str) -> Self {
        match Self::load(path) {
            Ok(config) => {
                println!("Loaded configuration from {}", path);
                config
            }
            Err(e) => {
                println!("Failed to load {}: {}. Using defaults.", path, e);
                let config = Self::default();
                // Try to save default config
                if let Err(e) = config.save(path) {
                    println!("Warning: Could not save default config: {}", e);
                }
                config
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_defaults() {
        let cfg = Config::default();
        assert_eq!(cfg.shadows.darkness, default_shadow_darkness());
        assert_eq!(
            cfg.shadows.backface_ambient_scale,
            default_backface_ambient_scale()
        );
    }
}
