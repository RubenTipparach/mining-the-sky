//! Chunked octree LOD for dual-contouring terrain. Nodes subdivide where they
//! are near the camera AND contain the iso-surface (empty space is skipped), so
//! the leaf chunks are fine near the viewer and coarse far away. Each leaf is
//! then meshed independently with Surface Nets - the practical octree-DC LOD
//! scheme (seam stitching between levels is the follow-up).

use glam::Vec3;

#[derive(Clone, Copy)]
pub struct Leaf {
    pub min: Vec3,
    pub size: f32,
    pub depth: u32,
}

/// True if the field changes sign anywhere on a coarse 3x3x3 lattice of the box
/// (i.e. the surface plausibly passes through it).
fn contains_surface<F: Fn(Vec3) -> f32>(f: &F, min: Vec3, size: f32) -> bool {
    let mut pos = false;
    let mut neg = false;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let p = min + Vec3::new(i as f32, j as f32, k as f32) * (size * 0.5);
                if f(p) < 0.0 {
                    neg = true;
                } else {
                    pos = true;
                }
                if pos && neg {
                    return true;
                }
            }
        }
    }
    false
}

/// Select the LOD leaves for a camera at `cam`. A node splits while its distance
/// to the camera is less than `size * split` (and it holds surface, and depth <
/// max). Returns only surface-bearing leaves.
pub fn select<F: Fn(Vec3) -> f32>(
    f: &F,
    cam: Vec3,
    region_min: Vec3,
    region_size: f32,
    split: f32,
    max_depth: u32,
) -> Vec<Leaf> {
    let mut out = Vec::new();
    descend(f, cam, region_min, region_size, 0, split, max_depth, &mut out);
    out
}

#[allow(clippy::too_many_arguments)]
fn descend<F: Fn(Vec3) -> f32>(
    f: &F,
    cam: Vec3,
    min: Vec3,
    size: f32,
    depth: u32,
    split: f32,
    max_depth: u32,
    out: &mut Vec<Leaf>,
) {
    if !contains_surface(f, min, size) {
        return; // empty space - prune
    }
    let center = min + Vec3::splat(size * 0.5);
    // distance from the camera to the node (approx AABB via its bounding sphere)
    let dist = ((cam - center).length() - size * 0.87).max(0.0);
    if depth < max_depth && dist < size * split {
        let h = size * 0.5;
        for oct in 0..8 {
            let off = Vec3::new(
                (oct & 1) as f32,
                ((oct >> 1) & 1) as f32,
                ((oct >> 2) & 1) as f32,
            ) * h;
            descend(f, cam, min + off, h, depth + 1, split, max_depth, out);
        }
    } else {
        out.push(Leaf { min, size, depth });
    }
}
