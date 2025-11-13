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
    pub depth_of_field: bool,
    #[serde(default = "default_dof_focus")]
    pub dof_focus_distance: f32,
    #[serde(default = "default_dof_blur")]
    pub dof_blur_size: f32,
    #[serde(default = "default_bloom")]
    pub bloom: bool,
    #[serde(default = "default_bloom_threshold")]
    pub bloom_threshold: f32,
    #[serde(default = "default_bloom_intensity")]
    pub bloom_intensity: f32,
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
    0.0005
}

fn default_time_of_day() -> f32 {
    0.5
}

fn default_dof_focus() -> f32 {
    200.0
}

fn default_dof_blur() -> f32 {
    2.0
}

fn default_bloom() -> bool {
    true
}

fn default_bloom_threshold() -> f32 {
    0.8
}

fn default_bloom_intensity() -> f32 {
    0.3
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

impl Default for EffectsConfig {
    fn default() -> Self {
        Self {
            depth_of_field: false,
            dof_focus_distance: default_dof_focus(),
            dof_blur_size: default_dof_blur(),
            bloom: default_bloom(),
            bloom_threshold: default_bloom_threshold(),
            bloom_intensity: default_bloom_intensity(),
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
