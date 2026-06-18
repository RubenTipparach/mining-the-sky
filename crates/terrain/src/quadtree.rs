//! Cube-sphere quadtree LOD: recursively split face patches that are close to
//! the camera, producing a set of crack-free leaf patches whose triangle
//! density tracks distance. This is the dynamic LOD system.

use crate::cubesphere::{face_dir, face_point, cube_to_sphere};
use crate::elevation::Elevation;
use glam::DVec3;

pub struct Planet {
    pub radius: f64,
}

/// A leaf patch of the quadtree: a (u, v) square on a cube face at some depth.
#[derive(Clone, Copy)]
pub struct Patch {
    pub face: u8,
    pub u0: f64,
    pub v0: f64,
    pub u1: f64,
    pub v1: f64,
    pub depth: u32,
    /// Undisplaced sphere-surface centre (world metres).
    pub center: DVec3,
    /// Longest edge length of the undisplaced patch (world metres).
    pub edge: f64,
}

pub struct Lod {
    pub patches: Vec<Patch>,
    pub per_depth: Vec<u32>,
    pub max_depth_reached: u32,
}

fn corner(planet: &Planet, face: u8, u: f64, v: f64) -> DVec3 {
    face_dir(face, u, v) * planet.radius
}

fn patch_edge(planet: &Planet, face: u8, u0: f64, v0: f64, u1: f64, v1: f64) -> f64 {
    let a = corner(planet, face, u0, v0);
    let b = corner(planet, face, u1, v0);
    let c = corner(planet, face, u1, v1);
    let d = corner(planet, face, u0, v1);
    (a - b).length().max((b - c).length()).max((c - d).length()).max((d - a).length())
}

/// Select the active LOD for a camera at world position `cam` (metres).
/// `split_factor`: split while distance < edge * split_factor.
pub fn select(planet: &Planet, cam: DVec3, split_factor: f64, max_depth: u32) -> Lod {
    let mut patches = Vec::new();
    let mut per_depth = vec![0u32; (max_depth + 1) as usize];
    let mut max_reached = 0u32;
    for face in 0..6u8 {
        descend(
            planet, cam, split_factor, max_depth, face, -1.0, -1.0, 1.0, 1.0, 0,
            &mut patches, &mut per_depth, &mut max_reached,
        );
    }
    Lod { patches, per_depth, max_depth_reached: max_reached }
}

#[allow(clippy::too_many_arguments)]
fn descend(
    planet: &Planet,
    cam: DVec3,
    split_factor: f64,
    max_depth: u32,
    face: u8,
    u0: f64,
    v0: f64,
    u1: f64,
    v1: f64,
    depth: u32,
    out: &mut Vec<Patch>,
    per_depth: &mut [u32],
    max_reached: &mut u32,
) {
    let um = 0.5 * (u0 + u1);
    let vm = 0.5 * (v0 + v1);
    let center = face_dir(face, um, vm) * planet.radius;
    let edge = patch_edge(planet, face, u0, v0, u1, v1);
    let dist = (cam - center).length();

    if depth < max_depth && dist < edge * split_factor {
        descend(planet, cam, split_factor, max_depth, face, u0, v0, um, vm, depth + 1, out, per_depth, max_reached);
        descend(planet, cam, split_factor, max_depth, face, um, v0, u1, vm, depth + 1, out, per_depth, max_reached);
        descend(planet, cam, split_factor, max_depth, face, u0, vm, um, v1, depth + 1, out, per_depth, max_reached);
        descend(planet, cam, split_factor, max_depth, face, um, vm, u1, v1, depth + 1, out, per_depth, max_reached);
    } else {
        per_depth[depth as usize] += 1;
        *max_reached = (*max_reached).max(depth);
        out.push(Patch { face, u0, v0, u1, v1, depth, center, edge });
    }
}

/// A triangle mesh for one patch, in world metres, including a downward skirt
/// around the rim that hides cracks against lower-LOD neighbours.
pub struct Mesh {
    pub positions: Vec<DVec3>,
    pub indices: Vec<u32>,
}

/// Build an (n x n)-vertex patch mesh displaced by elevation, plus skirts.
pub fn build_mesh(planet: &Planet, p: &Patch, n: usize, elev: &Elevation, skirt_m: f64) -> Mesh {
    let mut positions = Vec::with_capacity(n * n + 4 * n);
    let mut indices = Vec::new();

    let surf = |u: f64, v: f64| -> DVec3 {
        let dir = cube_to_sphere(face_point(p.face, u, v)).normalize();
        dir * (planet.radius + elev.land_height_m(dir))
    };

    // grid vertices
    for j in 0..n {
        let tv = j as f64 / (n - 1) as f64;
        let v = p.v0 + (p.v1 - p.v0) * tv;
        for i in 0..n {
            let tu = i as f64 / (n - 1) as f64;
            let u = p.u0 + (p.u1 - p.u0) * tu;
            positions.push(surf(u, v));
        }
    }
    // grid triangles
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    // skirt: drop a wall down from each edge vertex so LOD seams never show gaps
    let base = positions.len() as u32;
    let mut rim: Vec<u32> = Vec::new();
    for i in 0..n {
        rim.push(i as u32); // bottom row
    }
    for j in 1..n {
        rim.push((j * n + n - 1) as u32); // right col
    }
    for i in (0..n - 1).rev() {
        rim.push(((n - 1) * n + i) as u32); // top row
    }
    for j in (1..n - 1).rev() {
        rim.push((j * n) as u32); // left col
    }
    for (k, &r) in rim.iter().enumerate() {
        let top = positions[r as usize];
        let down = top - top.normalize() * skirt_m;
        positions.push(down);
        let _ = k;
    }
    for k in 0..rim.len() {
        let kn = (k + 1) % rim.len();
        let t0 = rim[k];
        let t1 = rim[kn];
        let s0 = base + k as u32;
        let s1 = base + kn as u32;
        indices.extend_from_slice(&[t0, s0, t1, t1, s0, s1]);
    }

    Mesh { positions, indices }
}

impl Lod {
    /// Total triangles if every patch were meshed at resolution `n`.
    pub fn triangle_count(&self, n: usize) -> usize {
        let per_patch = (n - 1) * (n - 1) * 2 + 4 * (n - 1) * 2;
        self.patches.len() * per_patch
    }
}
