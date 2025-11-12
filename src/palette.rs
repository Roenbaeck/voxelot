use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub albedo: [f32; 4],
    pub emissive: [f32; 3],
    pub emissive_intensity: f32,
}

impl Material {
    const fn new(albedo: [f32; 4], emissive: [f32; 3], emissive_intensity: f32) -> Self {
        Self {
            albedo,
            emissive,
            emissive_intensity,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.0)
    }
}

/// Runtime color palette for voxel types.
#[derive(Clone, Debug)]
pub struct Palette {
    materials: Vec<Material>,
    albedo_cache: Vec<[f32; 4]>,
}

impl Palette {
    /// Load palette from text file. Falls back to built-in defaults on error.
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let text = fs::read_to_string(path.as_ref()).ok();
        if let Some(contents) = text {
            if let Some(palette) = Self::from_string(&contents) {
                return palette;
            }
        }
        Self::default()
    }

    /// Parse palette from string. Supported formats per line:
    /// - `index baseR baseG baseB baseA`
    /// - `index baseR baseG baseB baseA emitR emitG emitB emitStrength`
    /// Values are 0-255 integers.
    pub fn from_string(contents: &str) -> Option<Self> {
        let mut map: HashMap<usize, Material> = HashMap::new();
        for (line_idx, line) in contents.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let parts: Vec<_> = trimmed.split_whitespace().collect();
            if parts.len() != 5 && parts.len() != 9 {
                eprintln!(
                    "palette: skipping line {} (expected 5 or 9 columns, got {})",
                    line_idx + 1,
                    parts.len()
                );
                continue;
            }

            let index = parts[0].parse::<usize>().ok()?;

            let base_bytes: Option<[u8; 4]> = parts[1..5]
                .iter()
                .map(|p| p.parse::<u8>().ok())
                .collect::<Option<Vec<_>>>()
                .and_then(|v| v.try_into().ok());

            if base_bytes.is_none() {
                eprintln!(
                    "palette: skipping line {} (invalid base color values)",
                    line_idx + 1
                );
                continue;
            }

            let base = Self::normalize_rgba(base_bytes.unwrap());

            let (emissive, intensity) = if parts.len() == 9 {
                let emissive_bytes: Option<[u8; 3]> = parts[5..8]
                    .iter()
                    .map(|p| p.parse::<u8>().ok())
                    .collect::<Option<Vec<_>>>()
                    .and_then(|v| v.try_into().ok());

                let strength = parts[8].parse::<u8>().ok();

                if let (Some(em_bytes), Some(strength_byte)) = (emissive_bytes, strength) {
                    (
                        Self::normalize_rgb(em_bytes),
                        (strength_byte as f32 / 255.0).clamp(0.0, 1.0),
                    )
                } else {
                    eprintln!(
                        "palette: skipping emissive data on line {} (invalid values)",
                        line_idx + 1
                    );
                    ([0.0, 0.0, 0.0], 0.0)
                }
            } else {
                ([0.0, 0.0, 0.0], 0.0)
            };

            map.insert(index, Material::new(base, emissive, intensity));
        }

        if map.is_empty() {
            return None;
        }

        let max_index = *map.keys().max().unwrap();
        let mut materials = vec![Material::default(); max_index + 1];
        for (idx, material) in map {
            materials[idx] = material;
        }
        let albedo_cache = materials.iter().map(|m| m.albedo).collect();
        Some(Self {
            materials,
            albedo_cache,
        })
    }

    fn normalize_rgba(bytes: [u8; 4]) -> [f32; 4] {
        [
            bytes[0] as f32 / 255.0,
            bytes[1] as f32 / 255.0,
            bytes[2] as f32 / 255.0,
            bytes[3] as f32 / 255.0,
        ]
    }

    fn normalize_rgb(bytes: [u8; 3]) -> [f32; 3] {
        [
            bytes[0] as f32 / 255.0,
            bytes[1] as f32 / 255.0,
            bytes[2] as f32 / 255.0,
        ]
    }

    pub fn colors(&self) -> &[[f32; 4]] {
        &self.albedo_cache
    }

    pub fn color(&self, index: u32) -> [f32; 4] {
        let idx = index as usize;
        if let Some(material) = self.materials.get(idx) {
            material.albedo
        } else {
            [1.0, 1.0, 1.0, 1.0]
        }
    }

    pub fn emissive(&self, index: u32) -> ([f32; 3], f32) {
        let idx = index as usize;
        if let Some(material) = self.materials.get(idx) {
            (material.emissive, material.emissive_intensity)
        } else {
            ([0.0, 0.0, 0.0], 0.0)
        }
    }

    pub fn material(&self, index: u32) -> Material {
        self.materials
            .get(index as usize)
            .copied()
            .unwrap_or_default()
    }

    pub fn color_u8(&self, index: u32) -> [u8; 4] {
        let color = self.color(index);
        [
            (color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            (color[3].clamp(0.0, 1.0) * 255.0).round() as u8,
        ]
    }

    pub fn gpu_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.albedo_cache)
    }
}

impl Default for Palette {
    fn default() -> Self {
        // Mirrors palette.txt defaults
        let mut materials = vec![Material::default(); 8];
        materials[1] = Material::new([0.102, 0.902, 0.302, 1.0], [0.0, 0.0, 0.0], 0.0);
        materials[2] = Material::new([1.0, 0.349, 0.349, 1.0], [0.0, 0.0, 0.0], 0.0);
        materials[3] = Material::new([0.349, 0.502, 1.0, 1.0], [0.0, 0.0, 0.0], 0.0);
        materials[4] = Material::new([0.949, 0.902, 0.349, 1.0], [0.0, 0.0, 0.0], 0.0);
        materials[5] = Material::new([0.949, 0.4, 1.0, 1.0], [0.0, 0.0, 0.0], 0.0);
        materials[6] = Material::new([0.302, 0.949, 1.0, 1.0], [0.0, 0.0, 0.0], 0.0);
        materials[7] = Material::new([0.851, 0.851, 0.851, 1.0], [0.0, 0.0, 0.0], 0.0);
        let albedo_cache = materials.iter().map(|m| m.albedo).collect();
        Self {
            materials,
            albedo_cache,
        }
    }
}
