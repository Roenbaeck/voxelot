//! Voxel Engine Library

pub mod buffer_allocator;
pub mod config;
pub mod culling;
pub mod file_format;
pub mod gi;
pub mod lib_hierarchical;
pub mod meshing_optimized;
pub mod palette;
pub mod profiling;
pub mod pawn;

// Re-export main types
pub use buffer_allocator::{AllocationError, SlabAllocator};
pub use config::Config;
pub use culling::{
    cull_visible_voxels, cull_visible_voxels_parallel, cull_visible_voxels_with_occlusion,
    get_visible_chunks, Camera, ChunkRenderInfo, CullStats, RenderConfig, VisibilityCache,
    VoxelInstance,
};
pub use file_format::{load_world, load_world_file, save_world, save_world_file};
pub use lib_hierarchical::bbox_local_to_world;
pub use lib_hierarchical::{Chunk, ShellVoxel, Voxel, VoxelType, World, WorldPos};
pub use meshing_optimized::{generate_chunk_mesh_optimized, ChunkEmitter, ChunkMesh, MeshVertex};
pub use palette::Palette;
pub use pawn::{Pawn, BoatPawn};
