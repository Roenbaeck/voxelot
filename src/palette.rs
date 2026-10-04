use crate::lib_hierarchical::VoxelType;
use rustc_hash::FxHashMap as HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Highest material index a palette can define: voxel types are `VoxelType` (u8) and the GPU
/// palette buffers hold 256 entries.
pub const MAX_PALETTE_INDEX: usize = VoxelType::MAX as usize;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub albedo: [f32; 4],
    pub emissive: [f32; 3],
    pub emissive_intensity: f32,
    /// Reflectivity coefficient (0.0 = matte, 1.0 = fully reflective)
    pub reflectivity: f32,
}

impl Material {
    const fn new(
        albedo: [f32; 4],
        emissive: [f32; 3],
        emissive_intensity: f32,
        reflectivity: f32,
    ) -> Self {
        Self {
            albedo,
            emissive,
            emissive_intensity,
            reflectivity,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.0, 0.0)
    }
}

/// A problem found on one line of a palette file. The rest of the file is still used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteIssue {
    /// 1-based line number in the file
    pub line: usize,
    pub message: String,
}

impl fmt::Display for PaletteIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

/// Why a palette could not be loaded.
#[derive(Debug)]
pub enum PaletteError {
    /// The file could not be read.
    Io { path: PathBuf, source: io::Error },
    /// The text holds no usable material definition; `issues` says what was wrong with each line
    /// that was skipped.
    NoMaterials {
        path: Option<PathBuf>,
        issues: Vec<PaletteIssue>,
    },
}

impl fmt::Display for PaletteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaletteError::Io { path, source } => {
                write!(
                    f,
                    "failed to read palette file '{}': {}",
                    path.display(),
                    source
                )
            }
            PaletteError::NoMaterials { path, issues } => {
                match path {
                    Some(path) => write!(
                        f,
                        "palette file '{}' has no valid material definition",
                        path.display()
                    )?,
                    None => write!(f, "palette has no valid material definition")?,
                }
                for issue in issues {
                    write!(f, "; {issue}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for PaletteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PaletteError::Io { source, .. } => Some(source),
            PaletteError::NoMaterials { .. } => None,
        }
    }
}

/// Runtime color palette for voxel types.
#[derive(Clone, Debug)]
pub struct Palette {
    materials: Vec<Material>,
    albedo_cache: Vec<[f32; 4]>,
    issues: Vec<PaletteIssue>,
}

impl Palette {
    /// Load palette from text file. Exits the process on error (see `try_load` to handle errors).
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let path_ref = path.as_ref();
        match Self::try_load(path_ref) {
            Ok(palette) => {
                palette.report_issues();
                palette
            }
            Err(PaletteError::Io { source, .. }) => {
                eprintln!(
                    "ERROR: Failed to read palette file '{}': {}",
                    path_ref.display(),
                    source
                );
                eprintln!("Please check that the file path in your configuration is correct.");
                std::process::exit(1);
            }
            Err(PaletteError::NoMaterials { issues, .. }) => {
                for issue in &issues {
                    eprintln!("palette: {issue}");
                }
                eprintln!(
                    "ERROR: Failed to parse palette file '{}'",
                    path_ref.display()
                );
                eprintln!("Palette file must contain at least one valid material definition.");
                std::process::exit(1);
            }
        }
    }

    /// Load palette from text file. Lines that cannot be used are skipped and listed in
    /// `issues()`; an error is returned if the file cannot be read or holds no valid material.
    pub fn try_load<P: AsRef<Path>>(path: P) -> Result<Self, PaletteError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| PaletteError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::try_from_string(&text).map_err(|e| match e {
            PaletteError::NoMaterials { issues, .. } => PaletteError::NoMaterials {
                path: Some(path.to_path_buf()),
                issues,
            },
            other => other,
        })
    }

    /// Parse palette from string. Supported formats per line:
    /// - `index baseR baseG baseB baseA`
    /// - `index baseR baseG baseB baseA emitR emitG emitB emitStrength`
    /// - `index baseR baseG baseB baseA emitR emitG emitB emitStrength reflectivity`
    /// Values are 0-255 integers; `index` is 0 to `MAX_PALETTE_INDEX`. Lines that cannot be used
    /// are skipped with a message on stderr; returns `None` if no valid material remains.
    pub fn from_string(contents: &str) -> Option<Self> {
        let palette = Self::try_from_string(contents).ok()?;
        palette.report_issues();
        Some(palette)
    }

    /// Like `from_string`, but reports why nothing could be used and keeps the skipped lines in
    /// `issues()` instead of printing them.
    pub fn try_from_string(contents: &str) -> Result<Self, PaletteError> {
        let mut issues = Vec::new();
        let mut map: HashMap<usize, (usize, Material)> = HashMap::default();
        // Editors on some platforms put a byte order mark at the start of the file
        let contents = contents.strip_prefix('\u{feff}').unwrap_or(contents);
        for (line_idx, line) in contents.lines().enumerate() {
            let line_no = line_idx + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut issue = |message: String| {
                issues.push(PaletteIssue {
                    line: line_no,
                    message,
                })
            };

            let parts: Vec<_> = trimmed.split_whitespace().collect();
            if parts.len() != 5 && parts.len() != 9 && parts.len() != 10 {
                issue(format!(
                    "skipped, expected 5, 9, or 10 columns, got {}",
                    parts.len()
                ));
                continue;
            }

            let index = match parts[0].parse::<usize>() {
                Ok(index) if index <= MAX_PALETTE_INDEX => index,
                Ok(index) => {
                    issue(format!(
                        "skipped, index {index} is above the maximum of {MAX_PALETTE_INDEX}"
                    ));
                    continue;
                }
                Err(_) => {
                    issue(format!("skipped, invalid index '{}'", parts[0]));
                    continue;
                }
            };

            let base_bytes: Option<[u8; 4]> = parts[1..5]
                .iter()
                .map(|p| p.parse::<u8>().ok())
                .collect::<Option<Vec<_>>>()
                .and_then(|v| v.try_into().ok());

            let Some(base_bytes) = base_bytes else {
                issue("skipped, invalid base color values".to_string());
                continue;
            };

            let base = Self::normalize_rgba(base_bytes);

            let (emissive, intensity, reflectivity) = if parts.len() >= 9 {
                let emissive_bytes: Option<[u8; 3]> = parts[5..8]
                    .iter()
                    .map(|p| p.parse::<u8>().ok())
                    .collect::<Option<Vec<_>>>()
                    .and_then(|v| v.try_into().ok());

                let strength = parts[8].parse::<u8>().ok();

                // Parse optional reflectivity (10th column, index 9)
                let reflect = if parts.len() == 10 {
                    parts[9].parse::<u8>().ok()
                } else {
                    Some(0)
                };

                if let (Some(em_bytes), Some(strength_byte), Some(reflect_byte)) =
                    (emissive_bytes, strength, reflect)
                {
                    (
                        Self::normalize_rgb(em_bytes),
                        (strength_byte as f32 / 255.0).clamp(0.0, 1.0),
                        (reflect_byte as f32 / 255.0).clamp(0.0, 1.0),
                    )
                } else {
                    issue("emissive/reflectivity ignored, invalid values".to_string());
                    ([0.0, 0.0, 0.0], 0.0, 0.0)
                }
            } else {
                ([0.0, 0.0, 0.0], 0.0, 0.0)
            };

            if let Some((first_line, _)) = map.get(&index) {
                issue(format!(
                    "index {index} was already defined on line {first_line}; this definition wins"
                ));
            }
            map.insert(
                index,
                (
                    line_no,
                    Material::new(base, emissive, intensity, reflectivity),
                ),
            );
        }

        let Some(&max_index) = map.keys().max() else {
            return Err(PaletteError::NoMaterials { path: None, issues });
        };
        let mut materials = vec![Material::default(); max_index + 1];
        for (idx, (_, material)) in map {
            materials[idx] = material;
        }
        let albedo_cache = materials.iter().map(|m| m.albedo).collect();
        Ok(Self {
            materials,
            albedo_cache,
            issues,
        })
    }

    /// Lines of the source text that were skipped or only partly used while parsing.
    pub fn issues(&self) -> &[PaletteIssue] {
        &self.issues
    }

    fn report_issues(&self) {
        for issue in &self.issues {
            eprintln!("palette: {issue}");
        }
    }

    pub fn normalize_rgba(bytes: [u8; 4]) -> [f32; 4] {
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

    pub fn reflectivity(&self, index: u32) -> f32 {
        let idx = index as usize;
        if let Some(material) = self.materials.get(idx) {
            material.reflectivity
        } else {
            0.0
        }
    }

    /// Get material properties buffer for GPU (256 entries of vec4: R=reflectivity, G=reserved, B=reserved, A=reserved)
    /// This is separate from the color palette and contains additional material properties.
    pub fn material_properties_gpu(&self) -> Vec<[f32; 4]> {
        let mut props = vec![[0.0f32, 0.0, 0.0, 0.0]; 256];
        for (i, material) in self.materials.iter().enumerate() {
            if i < 256 {
                props[i] = [material.reflectivity, 0.0, 0.0, 0.0];
            }
        }
        props
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

#[cfg(test)]
mod tests {
    use super::*;

    fn issues_of(contents: &str) -> Vec<String> {
        Palette::try_from_string(contents)
            .map(|p| p.issues().iter().map(|i| i.to_string()).collect())
            .unwrap_or_else(|e| match e {
                PaletteError::NoMaterials { issues, .. } => {
                    issues.iter().map(|i| i.to_string()).collect()
                }
                other => panic!("unexpected {other}"),
            })
    }

    #[test]
    fn parses_all_three_line_formats() {
        let palette = Palette::try_from_string(
            "\
# comment
0 255 255 255 255
1 255 0 0 255 0 255 0 255
2 0 0 255 128 10 20 30 51 102
",
        )
        .unwrap();
        assert!(palette.issues().is_empty());
        assert_eq!(palette.colors().len(), 3);
        assert_eq!(palette.color(0), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(palette.emissive(1), ([0.0, 1.0, 0.0], 1.0));
        assert_eq!(palette.reflectivity(1), 0.0);
        assert_eq!(palette.color_u8(2), [0, 0, 255, 128]);
        assert_eq!(palette.emissive(2).1, 51.0 / 255.0);
        assert_eq!(palette.reflectivity(2), 102.0 / 255.0);
    }

    #[test]
    fn a_bad_index_only_skips_its_own_line() {
        // This used to make the whole palette fail (a `?` on the index parse).
        let palette = Palette::from_string(
            "\
1 10 20 30 255
oops 1 2 3 255
-3 1 2 3 255
2.5 1 2 3 255
2 40 50 60 255
",
        )
        .unwrap();
        assert_eq!(palette.colors().len(), 3);
        assert_eq!(palette.color_u8(1), [10, 20, 30, 255]);
        assert_eq!(palette.color_u8(2), [40, 50, 60, 255]);
        assert_eq!(
            palette.issues().iter().map(|i| i.line).collect::<Vec<_>>(),
            [2, 3, 4]
        );
        assert!(palette.issues()[0].message.contains("invalid index 'oops'"));
    }

    #[test]
    fn indices_are_capped_at_the_voxel_type_range() {
        assert_eq!(MAX_PALETTE_INDEX, 255);
        let palette = Palette::try_from_string(
            "\
1 1 1 1 255
255 9 9 9 255
256 2 2 2 255
4000000000 3 3 3 255
18446744073709551615 4 4 4 255
99999999999999999999999 5 5 5 255
",
        )
        .unwrap();
        // no huge allocation: at most one entry per VoxelType
        assert_eq!(palette.colors().len(), 256);
        assert_eq!(palette.color_u8(255), [9, 9, 9, 255]);
        let lines: Vec<usize> = palette.issues().iter().map(|i| i.line).collect();
        assert_eq!(lines, [3, 4, 5, 6]);
        assert!(palette.issues()[0]
            .message
            .contains("above the maximum of 255"));
    }

    #[test]
    fn malformed_lines_are_reported_with_line_numbers() {
        let issues = issues_of(
            "\
# header

1 10 20 30 255
2 1 2 3
3 1 2 300 255
4 1 2 3 255 1 2
5 1 2 3 255 1 2 3 x
6 1 2 3 255 1 2 3 4 5 6
",
        );
        assert_eq!(
            issues,
            [
                "line 4: skipped, expected 5, 9, or 10 columns, got 4",
                "line 5: skipped, invalid base color values",
                "line 6: skipped, expected 5, 9, or 10 columns, got 7",
                "line 7: emissive/reflectivity ignored, invalid values",
                "line 8: skipped, expected 5, 9, or 10 columns, got 11",
            ]
        );
        // the base color of line 7 is kept, its emissive data is dropped
        let palette = Palette::from_string("5 1 2 3 255 1 2 3 x\n").unwrap();
        assert_eq!(palette.color_u8(5), [1, 2, 3, 255]);
        assert_eq!(palette.emissive(5), ([0.0; 3], 0.0));
    }

    #[test]
    fn redefined_index_uses_the_last_definition() {
        let palette = Palette::try_from_string("1 1 1 1 255\n\n1 2 2 2 255\n").unwrap();
        assert_eq!(palette.color_u8(1), [2, 2, 2, 255]);
        assert_eq!(
            palette.issues(),
            [PaletteIssue {
                line: 3,
                message: "index 1 was already defined on line 1; this definition wins".into()
            }]
        );
    }

    #[test]
    fn nothing_usable_is_an_error_naming_the_lines() {
        for text in ["", "\n\n", "# only a comment\n"] {
            assert!(Palette::from_string(text).is_none());
            match Palette::try_from_string(text) {
                Err(PaletteError::NoMaterials { issues, .. }) => assert!(issues.is_empty()),
                other => panic!("{other:?}"),
            }
        }
        let err = Palette::try_from_string("x 1 2 3 255\n1 2 3\n").unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("no valid material definition"),
            "{message}"
        );
        assert!(
            message.contains("line 1: skipped, invalid index 'x'"),
            "{message}"
        );
        assert!(
            message.contains("line 2: skipped, expected 5, 9, or 10 columns"),
            "{message}"
        );
    }

    #[test]
    fn handles_windows_line_endings_and_a_byte_order_mark() {
        let palette = Palette::try_from_string("\u{feff}0 1 2 3 255\r\n1 4 5 6 255\r\n").unwrap();
        assert!(palette.issues().is_empty());
        assert_eq!(palette.color_u8(0), [1, 2, 3, 255]);
        assert_eq!(palette.color_u8(1), [4, 5, 6, 255]);
    }

    #[test]
    fn try_load_reads_files_and_reports_errors() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();

        let missing = dir.join(format!("voxelot_palette_missing_{pid}.txt"));
        match Palette::try_load(&missing) {
            Err(PaletteError::Io { path, .. }) => assert_eq!(path, missing),
            other => panic!("{other:?}"),
        }
        let message = Palette::try_load(&missing).unwrap_err().to_string();
        assert!(
            message.contains(&missing.display().to_string()),
            "{message}"
        );

        let good = dir.join(format!("voxelot_palette_good_{pid}.txt"));
        std::fs::write(&good, "0 1 2 3 255\nbad line\n1 4 5 6 255\n").unwrap();
        let palette = Palette::try_load(&good);
        let loaded = Palette::load(&good); // same file through the exiting wrapper
        let _ = std::fs::remove_file(&good);
        let palette = palette.unwrap();
        assert_eq!(palette.colors(), loaded.colors());
        assert_eq!(palette.colors().len(), 2);
        assert_eq!(palette.issues()[0].line, 2);

        let empty = dir.join(format!("voxelot_palette_empty_{pid}.txt"));
        std::fs::write(&empty, "# nothing here\nnope\n").unwrap();
        let err = Palette::try_load(&empty);
        let _ = std::fs::remove_file(&empty);
        match err {
            Err(PaletteError::NoMaterials { path, issues }) => {
                assert_eq!(path, Some(empty));
                assert_eq!(issues.len(), 1);
                assert_eq!(issues[0].line, 2);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn shipped_palette_loads_without_issues() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("worlds/palette.txt");
        let palette = Palette::try_load(path).unwrap();
        assert!(palette.issues().is_empty(), "{:?}", palette.issues());
        assert_eq!(palette.colors().len(), 50);
    }
}
