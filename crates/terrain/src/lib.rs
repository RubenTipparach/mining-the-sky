//! Cube-sphere quadtree LOD terrain for Mining the Sky.
//!
//! - `cubesphere`: cube-face <-> sphere mapping and tangent bases.
//! - `elevation`: continuous procedural height (seamless at any LOD).
//! - `quadtree`: distance-based LOD selection + crack-free patch meshes.
//! - `render`: CPU verification images (LOD map, hillshaded relief).

pub mod cubesphere;
pub mod elevation;
pub mod octree;
pub mod quadtree;
pub mod raster;
pub mod render;
pub mod surfacenets;

pub use elevation::Elevation;
pub use quadtree::{build_mesh, select, Lod, Mesh, Patch, Planet};
