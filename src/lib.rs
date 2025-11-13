//! Voxel Engine Library

pub mod config;
pub mod culling;
pub mod lib_hierarchical;
pub mod meshing;
pub mod octree_format;
pub mod palette;

// Re-export main types
pub use config::Config;
pub use culling::{
    cull_visible_voxels, cull_visible_voxels_parallel, cull_visible_voxels_with_occlusion,
    get_visible_chunks, Camera, ChunkRenderInfo, RenderConfig, VisibilityCache, VoxelInstance,
};
pub use lib_hierarchical::{Chunk, Voxel, VoxelType, World, WorldPos};
pub use meshing::{generate_chunk_mesh, ChunkEmitter, ChunkMesh, MeshVertex};
pub use octree_format::{load_world, save_world, load_world_file, save_world_file};
pub use palette::Palette;
