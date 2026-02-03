# REPO_MAP.md — Repository file map & quick guide

Purpose
-------
This file is a compact, machine- and human-friendly map of the repository to help maintainers, contributors, and automation agents find important files and understand their purpose quickly. Keep this file small and factual — add new entries when you add or significantly change areas of the codebase.

How to use
----------
- Agents: prefer this file as a first point of entry to locate relevant source files, binaries, scripts, CI, and data.
- Humans: find the right file quickly without scanning the whole tree; follow the `Path` column to open the file.

Table legend: Path | Type | Short description | Notes / dependencies
---------------------------------------------------------------------------

Path | Type | Description | Notes
---|---|---|---
`Cargo.toml` | file | Project manifest (dependencies, features, target-specific overrides). | `wgpu = "28.0"`; `windows = "0.62.2"` pinned for Windows builds.
`Cargo.lock` | file | Locked dependency graph for reproducible CI/Builds. | Commit in app repos for stable CI.
`README.md` | file | High-level project overview, quick-start, controls and architecture notes. | Good first read for contributors.
`SELF.md` | file | The Agent's Living Identity — persistent context across model switches and sessions. | **Read first** to maintain cognitive continuity.
`LICENSE` | file | Project license text.
`CONFIGURATION.md` | doc | Full reference for `config.toml` options and defaults (derived from `src/config.rs`). | Useful when editing runtime configuration.
`WGPU_28.md` | doc | Migration notes and breaking-change checklist for `wgpu` v28. | Important when touching rendering/shader code.
`KNOWN_BUGS.md` | doc | Tracked issues and regressions.
`TODO.md` | doc | Roadmap and feature ideas for contributors.

Top-level data & demo
`demo/` | dir | Files packaged with the demo build (config, README). | `demo/config.toml` tuned for the distributed package.
`demo/config.toml` | file | Tuned demo config (references `worlds/flat_city_test.vhc`, `worlds/palette.txt`, `worlds/skybox.hdr`). |
`demo/README.md` | file | Instructions for running the demo package. |
`worlds/` | dir | Pre-generated demo and test worlds and metadata. | `.vhc` compressed chunk format, `.toml` world configs.
`worlds/flat_city_test.vhc` | data | Demo world (leaf-equivalent voxel count: ~32,691,321). | Used by `demo/config.toml` and tests.
`worlds/flat_city_test.toml` | file | World-specific config (camera, palette path) used in examples. |
`worlds/palette.txt` | data | Palette mapping voxel indices to color/emit properties (required by viewer). |
`worlds/skybox.hdr` | data | HDR environment map used for the demo skybox (optional but packaged). |

CI & Release
`.github/workflows/build-demo.yml` | CI | Builds `voxelot` across OS matrix, packages demo zips, uploads artifacts, and publishes releases. | Uses `actions/cache`, cross-platform zipping fallbacks and `softprops/action-gh-release@v2`.
`.github/copilot-instructions.md` | doc | Short agent guidance — use `REPO_MAP.md` for the canonical path map. |

Scripts & automation
`scripts/package-demo.sh` | script | Cross-platform packaging of demo binary and assets into a zip (zip / PowerShell / 7z / Python fallback). | Ensures `worlds/` and demo `config.toml` included; sets executable bit where appropriate.
`scripts/delete_oct_files.sh` | script | Utility that deletes legacy `.oct` files (legacy compatibility helper). |

Core library (src/)
`src/lib.rs` | file | Library entry point; re-exports major types and modules. |
`src/lib_hierarchical.rs` | core | Hierarchical chunk world implementation — "chunks all the way" model. | Chunk size fixed at 16; uses Roaring bitmaps for presence and rank indexing.
`src/file_format.rs` | core | `.vhc` load/save (zstd compressed). | Mirrors Chunk layout: positions, type/child encoding, recursion.
`src/meshing_optimized.rs` | core | Binary greedy mesher, `generate_chunk_mesh_optimized`. | Produces `ChunkMesh` (vertices, indices, emitters).
`src/culling.rs` | core | Frustum/HZB/occlusion culling; visibility caches and helpers. | Produces visible voxel instances and visible chunk lists.
`src/buffer_allocator.rs` | core | Slab allocator (SlabAllocator) for large GPU buffers (VB/IB pools). |
`src/config.rs` | core | Serde-config structures and defaults for runtime settings. | Adds world.skybox and many effect toggles; used by viewer.
`src/palette.rs` | core | Palette parsing and material properties (albedo & emissive). |
`src/gi.rs` | core | Global illumination probe system (GiProbe, async updates). | Background probe calculation and caching.
`src/input.rs` | core | Input mapping and debug/config setting helpers. |
`src/pawn.rs` | core | Pawn types (BoatPawn) and player/vessel logic. |
`src/profiling.rs` | core | Small profiling helpers and scope markers for overlays. |

Binaries & tools (src/bin)
`src/bin/voxelot.rs` | binary | Main desktop viewer. | Loads config, world, shaders at runtime; runtime WGSL compilation via `wgpu`.
`src/bin/generate_world.rs` | binary | High-performance world generator (writes `.vhc`). | Many flags to control terrain and tile size.
`src/bin/generate_test_world.rs` | tool | Small generator used for tests/demos. |
`src/bin/analyze_test_world.rs` | tool | Analyzer (sanity checks on generated test worlds). |
`src/bin/inspect_world.rs` | tool | Per-chunk inspector; reports column inconsistencies and bounding boxes. |
`src/bin/world_stats.rs` | tool | Computes leaf-equivalent voxel counts and bounding boxes (used to report ~32.7M). |
`src/bin/estimate_mesh_sizes.rs` | tool | Mesh size estimation tool for planning budgets. |
`src/bin/check_world_gaps.rs` | tool | Validates generated worlds for empty gaps. |
`src/bin/check_voxel_types.rs` | tool | Verifies palette indices and voxel type ranges in `.vhc`. |
`src/bin/reproduce_bbox.rs` | tool | Helps reproduce bbox edge cases for debugging. |
`src/bin/verify_generation_logic.rs` | tool | Generation regression tests and checks. |
`src/bin/verify_glam.rs` | tool | Sanity checks for linear algebra operations. |
`src/bin/verify_octree.rs` | tool | Legacy verification helper for `.oct` (kept for reference). |
`src/bin/test_render_coords.rs` | tool | Debug helper for coordinate mapping in render code. |
`src/bin/compress_worlds.rs` | tool | Utility to recompress or batch-compress `.vhc` files. |

Shaders (shaders/)
`shaders/` | dir | WGSL shaders embedded and compiled at runtime by `wgpu`. |
`shaders/bloom_extract.wgsl` | file | Bloom extraction pass. |
`shaders/dof_coc_copy.wgsl` | file | Depth-of-field circle-of-confusion copy pass. |
`shaders/dof_combine.wgsl` | file | Depth-of-field combine pass. |
`shaders/dual_kawase_down.wgsl` | file | Dual Kawase downsample blur. |
`shaders/dual_kawase_up.wgsl` | file | Dual Kawase upsample blur. |
`shaders/editor_preview.wgsl` | file | Editor preview/wireframe rendering. |
`shaders/gpu_cull.wgsl` | file | GPU culling and visibility. |
`shaders/hzb_gen.wgsl` | file | Hierarchical Z-buffer generation. |
`shaders/impostor.wgsl` | file | Impostor rendering. |
`shaders/post_composite.wgsl` | file | Post-processing composite pass. |
`shaders/radiance_cascades.wgsl` | file | Radiance cascade lighting. |
`shaders/skybox.wgsl` | file | Skybox shader implementing HDR sampling/compositing. |
`shaders/ssao_blur.wgsl` | file | SSAO blur pass. |
`shaders/ssilvb.wgsl` | file | SSGI/AO lighting (SSILVB). |
`shaders/ssr.wgsl` | file | Screen-space reflections. |
`shaders/ssr_kawase_blur.wgsl` | file | SSR blur pass. |
`shaders/voxel.wgsl` | file | Main voxel rendering pipeline. |
`shaders/water.wgsl` | file | Water rendering. |
`shaders/wts_inject.wgsl` | file | WTS sunlight injection. |
`shaders/wts_relax.wgsl` | file | WTS relaxation/diffusion. |

Other files of interest
`demo/config.toml` | file | Tuned demo configuration (camera, rendering, world/palette paths). |
`shaders/skybox.wgsl` | file | Skybox shader implementing HDR sampling and compositing. |
`worlds/*.toml` | files | Per-world config templates (camera, palette, spawn mode). |
`.github/*` | dir | CI and workflow rules; see `build-demo.yml` for demo packaging. |

Maintenance notes
-----------------
- Keep `REPO_MAP.md` updated when you add/remove important files or change resource locations for the demo.
- Use `src/bin/*` helpers to validate world formats after changes to `file_format.rs` or generator code.
- For large reorganizations, run a repo scan (example):

  find . -maxdepth 2 -type f -print | sed 's|^./||' | sort

Contact & ownership
-------------------
- Primary maintainers: repository owners and active contributors (see commit history / CODEOWNERS if present).

(Updated & verified: REPO_MAP.md — I opened and scanned each referenced file to confirm the descriptions above.)
