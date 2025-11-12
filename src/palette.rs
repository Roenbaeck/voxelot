use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Runtime color palette for voxel types.
#[derive(Clone, Debug)]
pub struct Palette {
    colors: Vec<[f32; 4]>,
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

    /// Parse palette from string with lines: `index R G B A` (0-255 integers).
    pub fn from_string(contents: &str) -> Option<Self> {
        let mut map: HashMap<usize, [f32; 4]> = HashMap::new();
        for (line_idx, line) in contents.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let parts: Vec<_> = trimmed.split_whitespace().collect();
            if parts.len() != 5 {
                eprintln!(
                    "palette: skipping line {} (expected 5 columns, got {})",
                    line_idx + 1,
                    parts.len()
                );
                continue;
            }

            let index = parts[0].parse::<usize>().ok()?;
            let rgba_bytes: Option<[u8; 4]> = parts[1..]
                .iter()
                .map(|p| p.parse::<u8>().ok())
                .collect::<Option<Vec<_>>>()
                .and_then(|v| v.try_into().ok());

            if let Some(bytes) = rgba_bytes {
                map.insert(index, Self::normalize(bytes));
            }
        }

        if map.is_empty() {
            return None;
        }

        let max_index = *map.keys().max().unwrap();
        let mut colors = vec![[1.0, 1.0, 1.0, 1.0]; max_index + 1];
        for (idx, color) in map {
            colors[idx] = color;
        }
        Some(Self { colors })
    }

    fn normalize(bytes: [u8; 4]) -> [f32; 4] {
        [
            bytes[0] as f32 / 255.0,
            bytes[1] as f32 / 255.0,
            bytes[2] as f32 / 255.0,
            bytes[3] as f32 / 255.0,
        ]
    }

    pub fn colors(&self) -> &[[f32; 4]] {
        &self.colors
    }

    pub fn color(&self, index: u32) -> [f32; 4] {
        let idx = index as usize;
        if idx < self.colors.len() {
            self.colors[idx]
        } else {
            [1.0, 1.0, 1.0, 1.0]
        }
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
        bytemuck::cast_slice(&self.colors)
    }
}

impl Default for Palette {
    fn default() -> Self {
        // Mirrors palette.txt defaults
        let mut colors = vec![[1.0, 1.0, 1.0, 1.0]; 8];
        colors[1] = [0.102, 0.902, 0.302, 1.0];
        colors[2] = [1.0, 0.349, 0.349, 1.0];
        colors[3] = [0.349, 0.502, 1.0, 1.0];
        colors[4] = [0.949, 0.902, 0.349, 1.0];
        colors[5] = [0.949, 0.4, 1.0, 1.0];
        colors[6] = [0.302, 0.949, 1.0, 1.0];
        colors[7] = [0.851, 0.851, 0.851, 1.0];
        Self { colors }
    }
}
