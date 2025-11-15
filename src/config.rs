//! Configuration management for Voxelot

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

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
    #[serde(default)]
    pub debug: DebugConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    #[serde(default = "default_world_file")]
    pub file: String,
    #[serde(default = "default_palette_file")]
    pub palette: String,
    #[serde(default = "default_camera_position")]
    pub camera_position: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingConfig {
    #[serde(default = "default_lod_subdivide_distance")]
    pub lod_subdivide_distance: f32,
    #[serde(default = "default_lod_merge_distance")]
    pub lod_merge_distance: f32,
    #[serde(default = "default_chunk_lod_distance")]
    pub chunk_lod_distance: f32,
    #[serde(default = "default_fov")]
    pub fov_degrees: f32,
    #[serde(default = "default_near_plane")]
    pub near_plane: f32,
    #[serde(default = "default_far_plane")]
    pub far_plane: f32,
    #[serde(default = "default_camera_speed")]
    pub camera_speed_multiplier: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphereConfig {
    #[serde(default = "default_fog_density")]
    pub fog_density: f32,
    #[serde(default = "default_time_of_day")]
    pub time_of_day: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsConfig {
    #[serde(default)]
    pub depth_of_field: DepthOfFieldConfig,
    #[serde(default)]
    pub bloom: BloomConfig,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowConfig {
    #[serde(default = "default_shadow_map_size")]
    pub map_size: u32,
    #[serde(default = "default_shadow_darkness")]
    pub darkness: f32,
    #[serde(default = "default_backface_ambient_scale")]
    pub backface_ambient_scale: f32,
    #[serde(default = "default_ao_strength")]
    pub ao_strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    #[serde(default = "default_mesh_cache_mb")]
    pub mesh_cache_budget_mb: u64,
    #[serde(default = "default_mesh_workers")]
    pub mesh_worker_count: Option<usize>,
    #[serde(default = "default_mesh_upload_baseline")]
    pub mesh_upload_baseline: usize,
    #[serde(default = "default_fallback_detail_distance")]
    pub fallback_detail_distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    #[serde(default = "default_true")]
    pub show_fps: bool,
    #[serde(default = "default_true")]
    pub show_chunk_stats: bool,
}

// Default values
fn default_world_file() -> String {
    "world_1.oct".to_string()
}

fn default_palette_file() -> String {
    "worlds/palette.txt".to_string()
}

fn default_camera_position() -> [f32; 3] {
    [320.0, 100.0, 320.0]
}

fn default_lod_subdivide_distance() -> f32 {
    500.0
}

fn default_lod_merge_distance() -> f32 {
    1000.0
}

fn default_chunk_lod_distance() -> f32 {
    800.0
}

fn default_fov() -> f32 {
    70.0
}

fn default_near_plane() -> f32 {
    0.1
}

fn default_far_plane() -> f32 {
    5000.0
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

fn default_shadow_map_size() -> u32 {
    4096
}

fn default_shadow_darkness() -> f32 {
    1.0
}

fn default_backface_ambient_scale() -> f32 {
    0.7
}

fn default_ao_strength() -> f32 {
    0.9
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

fn default_fallback_detail_distance() -> f32 {
    500.0
}

fn default_true() -> bool {
    true
}

// Implement Default for all configs
impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            file: default_world_file(),
            palette: default_palette_file(),
            camera_position: default_camera_position(),
        }
    }
}

impl Default for RenderingConfig {
    fn default() -> Self {
        Self {
            lod_subdivide_distance: default_lod_subdivide_distance(),
            lod_merge_distance: default_lod_merge_distance(),
            chunk_lod_distance: default_chunk_lod_distance(),
            fov_degrees: default_fov(),
            near_plane: default_near_plane(),
            far_plane: default_far_plane(),
            camera_speed_multiplier: default_camera_speed(),
        }
    }
}

impl Default for AtmosphereConfig {
    fn default() -> Self {
        Self {
            fog_density: default_fog_density(),
            time_of_day: default_time_of_day(),
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
        }
    }
}

impl Default for EffectsConfig {
    fn default() -> Self {
        Self {
            depth_of_field: DepthOfFieldConfig::default(),
            bloom: BloomConfig::default(),
        }
    }
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            map_size: default_shadow_map_size(),
            darkness: default_shadow_darkness(),
            backface_ambient_scale: default_backface_ambient_scale(),
            ao_strength: default_ao_strength(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            mesh_cache_budget_mb: default_mesh_cache_mb(),
            mesh_worker_count: default_mesh_workers(),
            mesh_upload_baseline: default_mesh_upload_baseline(),
            fallback_detail_distance: default_fallback_detail_distance(),
        }
    }
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            show_fps: true,
            show_chunk_stats: true,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            world: WorldConfig::default(),
            rendering: RenderingConfig::default(),
            atmosphere: AtmosphereConfig::default(),
            effects: EffectsConfig::default(),
            shadows: ShadowConfig::default(),
            performance: PerformanceConfig::default(),
            debug: DebugConfig::default(),
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
        assert_eq!(cfg.shadows.backface_ambient_scale, default_backface_ambient_scale());
        assert_eq!(cfg.shadows.ao_strength, default_ao_strength());
    }
}
