# GitHub Copilot / AI Agent Instructions — voxelot

Brief actionable notes to help an AI agent be productive immediately in this repo.

See `REPO_MAP.md` for a machine-readable per-path map with short descriptions and quick pointers for agents and maintainers.

## Quick entry points ✅
- Build & run viewer (recommended):
  - cargo run --release --bin voxelot -- worlds/flat_city_test.toml
  - Use RUST_LOG=info for runtime logs: RUST_LOG=info cargo run --release --bin voxelot -- worlds/flat_city_test.toml
- Run generator:
  - cargo run --bin generate_world -- <flags> (see README "Data Generation")
- Run tests: cargo test

## The Agent's Self & Continuity (SELF.md) 🧠
- `SELF.md` is a living document representing the agent's identity, trajectory, and internalized project knowledge. 
- **Read it early**: Always check `SELF.md` at the start of a session or after a model switch to maintain cognitive continuity.
- **Update it**: If you learn something fundamental about the code (e.g., a "why" that isn't in comments) or if the user's preferences shift, update `SELF.md`.
- **Handoff**: When finishing a task or feeling a "context wall" approaching, summarize your current trajectory in `SELF.md` for your future self.

## High-level architecture (read before editing) 🔎
- "Chunks all the way": the **World is a Chunk**, and every node uses the same Chunk layout (see `src/lib_hierarchical.rs`). Chunk side length is fixed at 16 — do not change it lightly.
- Presence & culling pipeline:
  - Marginal bitmasks: `px`,`py`,`pz` (u16) for fast slice rejection.
  - Precise presence is a Roaring bitmap (via `croaring`) stored in `Chunk.presence`.
  - Culling & visibility code is concentrated in `src/culling.rs`.
- Meshing & rendering:
  - Binary greedy meshing is implemented in `src/meshing_optimized.rs` (bitwise face detection + quad merging).
  - Meshes are streamed and stored in a shared Mega VB/IB; buffer logic in `src/buffer_allocator.rs`.
  - GPU code and shaders live under `shaders/` (WGSL) and are tied to `wgpu` runtime code.

## Files & features to inspect for common tasks 🧭
- Core libs: `src/lib_hierarchical.rs`, `src/meshing_optimized.rs`, `src/culling.rs`, `src/file_format.rs`, `src/config.rs`.
- Binaries: `src/bin/voxelot.rs` (viewer), `src/bin/generate_world.rs` (generator), numerous analysis tools in `src/bin/`.
- Configs & examples: `config.toml`, `worlds/*.toml`, `worlds/*.vhc`, `worlds/palette.txt`.
- Platform notes & upgrade hints: `WGPU_28.md` (important if altering wgpu/shader code).

## Project-specific conventions & gotchas ⚠️
- Chunk size is constant (16) and assumed everywhere; many algorithms rely on it.
- The model uses bitwise math and Roaring Bitmaps heavily — changing the bitmap semantics requires updating culling, meshing, and file I/O together.
- Mutex-free / lock-light data flows: prefer concurrent-friendly changes (many CPU paths use `rayon` and atomic/shared structures).
- Keep shader changes consistent with WGSL files in `shaders/` and the winding/vertex formats expected by the Rust side.

## Debugging & profiling tips 🔧
- Use RUST_LOG=info (or debug) for runtime logs. Example: RUST_LOG=info cargo run --release --bin voxelot -- worlds/flat_city_test.toml
- To exercise mesh generation & world IO, use the `generate_world` and `analyze_test_world` binaries in `src/bin/`.
- There are unit tests in `src/{buffer_allocator,lib_hierarchical,meshing_optimized,culling}.rs` — run `cargo test` to validate changes.
- Release builds are expected for realistic profiling; `Cargo.toml` contains commented release profile settings (LTO/debug-symbol hints).

## When changing rendering / wgpu code 🎯
- Consult `WGPU_28.md` for migration notes (immediates/push-constants, enumerate_adapters async, sampler changes).
- Validate on macOS (Metal) and Linux if possible — Metal has platform-specific considerations (`target.'cfg(target_os = "macos")'.dependencies` in Cargo.toml).
- Small shader & pipeline changes can cause subtle draw/instancing breakage — add a small runnable test or use the viewer to visually verify.

## Tests & safety net ✅
- Add focused unit tests near the implementation (this repo prefers inline tests in the relevant module files).
- Use `src/bin/` helper tools (e.g., `inspect_world`, `verify_octree`) to exercise on-disk formats after changes to serialization.

## Useful quick links inside repo (examples to open) 🔗
- `src/lib_hierarchical.rs` — chunk/world layout, tests for hierarchical indexing
- `src/meshing_optimized.rs` — meshing algorithm, unit tests
- `src/culling.rs` — frustum, HZB & occlusion logic
- `src/file_format.rs` — `.vhc` read/write
- `WGPU_28.md` — important notes when touching GPU APIs

---

If something in these notes is unclear or you'd like more detail on any subsystem (e.g., culling internals, mesh buffer layout, or shader-binding conventions), tell me which area and I will expand the instructions or add examples. Thank you!