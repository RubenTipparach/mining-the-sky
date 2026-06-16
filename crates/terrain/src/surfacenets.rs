//! Naive Surface Nets - a simple form of dual contouring. It meshes the
//! iso-surface (density == 0) of an arbitrary 3D scalar field, placing one
//! vertex per surface-straddling cell at the average of its edge crossings
//! (the "mass point") and connecting a quad across every sign-changing grid
//! edge. Because it works on a full 3D field (not a heightmap) it handles
//! overhangs, arches and caves, and it LODs cleanly via an octree later.

use glam::Vec3;

pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<u32>,
}

// Marching-cubes corner offsets and the 12 edges (corner index pairs).
const CORNERS: [[i32; 3]; 8] = [
    [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
    [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1],
];
const EDGES: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
];

/// Extract a mesh from `f` over a grid of `dim` sample points per axis, with
/// spacing `cell` (world units) starting at `origin`.
pub fn surface_nets<F: Fn(Vec3) -> f32>(f: &F, origin: Vec3, cell: f32, dim: usize) -> Mesh {
    let (mesh, _cell_vert) = mesh_core(f, origin, cell, dim);
    mesh
}

/// Like `surface_nets`, but adds downward skirts along the four vertical
/// boundary faces of the chunk so neighbouring octree leaves of a different LOD
/// can never show a crack. The terrain field is monotone in radius (a
/// heightfield in disguise), so vertically-stacked leaves are pruned and only
/// the four side faces ever abut a coarser/finer neighbour. Each side gets a
/// curtain that drops `skirt_len` along `down`, hidden under the surface.
pub fn surface_nets_skirted<F: Fn(Vec3) -> f32>(
    f: &F,
    origin: Vec3,
    cell: f32,
    dim: usize,
    down: Vec3,
    skirt_len: f32,
) -> Mesh {
    let n = dim;
    let (mut mesh, cell_vert) = mesh_core(f, origin, cell, dim);
    if n < 2 {
        return mesh;
    }
    let idx = |x: usize, y: usize, z: usize| (x * n + y) * n + z;
    let drop = down * skirt_len;

    // The surface vertex of the boundary column at the given face cell, scanning
    // the vertical (y) axis. For a heightfield there is exactly one per column.
    let col = |fixed_axis: usize, fixed: usize, t: usize| -> u32 {
        for y in 0..n - 1 {
            let (x, yy, z) = match fixed_axis {
                0 => (fixed, y, t),
                _ => (t, y, fixed),
            };
            let v = cell_vert[idx(x, yy, z)];
            if v != u32::MAX {
                return v;
            }
        }
        u32::MAX
    };

    let mut skirt = |mesh: &mut Mesh, a: u32, b: u32| {
        if a == u32::MAX || b == u32::MAX {
            return;
        }
        let pa = mesh.positions[a as usize];
        let pb = mesh.positions[b as usize];
        let nrm = (pb - pa).cross(drop).normalize_or_zero();
        let base = mesh.positions.len() as u32;
        for p in [pa, pb, pb + drop, pa + drop] {
            mesh.positions.push(p);
            mesh.normals.push(nrm);
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // x=0 and x=n-2 faces, stitched along z; z=0 and z=n-2 faces, along x.
    for &face in &[0usize, n - 2] {
        for t in 0..n - 2 {
            skirt(&mut mesh, col(0, face, t), col(0, face, t + 1));
            skirt(&mut mesh, col(2, face, t), col(2, face, t + 1));
        }
    }
    mesh
}

/// Shared Surface Nets core: returns the mesh plus the per-cell vertex index
/// table (`u32::MAX` where a cell has no surface vertex), which the skirted
/// variant needs to find boundary columns.
fn mesh_core<F: Fn(Vec3) -> f32>(
    f: &F,
    origin: Vec3,
    cell: f32,
    dim: usize,
) -> (Mesh, Vec<u32>) {
    let n = dim;
    let idx = |x: usize, y: usize, z: usize| (x * n + y) * n + z;
    let pos_of = |x: usize, y: usize, z: usize| {
        origin + Vec3::new(x as f32, y as f32, z as f32) * cell
    };

    // sample the field
    let mut d = vec![0.0f32; n * n * n];
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                d[idx(x, y, z)] = f(pos_of(x, y, z));
            }
        }
    }

    let grad = |p: Vec3| -> Vec3 {
        let e = cell * 0.5;
        Vec3::new(
            f(p + Vec3::X * e) - f(p - Vec3::X * e),
            f(p + Vec3::Y * e) - f(p - Vec3::Y * e),
            f(p + Vec3::Z * e) - f(p - Vec3::Z * e),
        )
        .normalize_or_zero()
    };

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut cell_vert = vec![u32::MAX; n * n * n];

    // one vertex per surface-straddling cell (cells run 0..n-1)
    for x in 0..n - 1 {
        for y in 0..n - 1 {
            for z in 0..n - 1 {
                let mut dens = [0.0f32; 8];
                let mut neg = 0;
                for (c, off) in CORNERS.iter().enumerate() {
                    let v = d[idx(x + off[0] as usize, y + off[1] as usize, z + off[2] as usize)];
                    dens[c] = v;
                    if v < 0.0 {
                        neg += 1;
                    }
                }
                if neg == 0 || neg == 8 {
                    continue;
                }
                let mut sum = Vec3::ZERO;
                let mut cnt = 0.0f32;
                for [a, b] in EDGES {
                    let (da, db) = (dens[a], dens[b]);
                    if (da < 0.0) == (db < 0.0) {
                        continue;
                    }
                    let t = da / (da - db);
                    let pa = pos_of(
                        x + CORNERS[a][0] as usize,
                        y + CORNERS[a][1] as usize,
                        z + CORNERS[a][2] as usize,
                    );
                    let pb = pos_of(
                        x + CORNERS[b][0] as usize,
                        y + CORNERS[b][1] as usize,
                        z + CORNERS[b][2] as usize,
                    );
                    sum += pa.lerp(pb, t);
                    cnt += 1.0;
                }
                let p = sum / cnt;
                cell_vert[idx(x, y, z)] = positions.len() as u32;
                positions.push(p);
                normals.push(grad(p));
            }
        }
    }

    // quad per sign-changing grid edge, from the four cells sharing it.
    let mut indices = Vec::new();
    let mut quad = |a: u32, b: u32, c: u32, e: u32, flip: bool, out: &mut Vec<u32>| {
        if a == u32::MAX || b == u32::MAX || c == u32::MAX || e == u32::MAX {
            return;
        }
        if flip {
            out.extend_from_slice(&[a, c, b, a, e, c]);
        } else {
            out.extend_from_slice(&[a, b, c, a, c, e]);
        }
    };
    for x in 1..n - 1 {
        for y in 1..n - 1 {
            for z in 1..n - 1 {
                let d0 = d[idx(x, y, z)];
                // edge +X
                let dx = d[idx(x + 1, y, z)];
                if (d0 < 0.0) != (dx < 0.0) {
                    quad(
                        cell_vert[idx(x, y - 1, z - 1)],
                        cell_vert[idx(x, y, z - 1)],
                        cell_vert[idx(x, y, z)],
                        cell_vert[idx(x, y - 1, z)],
                        d0 < 0.0,
                        &mut indices,
                    );
                }
                // edge +Y
                let dy = d[idx(x, y + 1, z)];
                if (d0 < 0.0) != (dy < 0.0) {
                    quad(
                        cell_vert[idx(x - 1, y, z - 1)],
                        cell_vert[idx(x, y, z - 1)],
                        cell_vert[idx(x, y, z)],
                        cell_vert[idx(x - 1, y, z)],
                        d0 >= 0.0,
                        &mut indices,
                    );
                }
                // edge +Z
                let dz = d[idx(x, y, z + 1)];
                if (d0 < 0.0) != (dz < 0.0) {
                    quad(
                        cell_vert[idx(x - 1, y - 1, z)],
                        cell_vert[idx(x, y - 1, z)],
                        cell_vert[idx(x, y, z)],
                        cell_vert[idx(x - 1, y, z)],
                        d0 < 0.0,
                        &mut indices,
                    );
                }
            }
        }
    }

    (Mesh { positions, normals, indices }, cell_vert)
}

fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Drop disconnected components smaller than `min_tris` triangles - i.e. remove
/// floating bits, keeping only the connected terrain. A standard, robust way to
/// clean a dual-contoured field instead of hoping the density never floats.
pub fn drop_small_components(mesh: &Mesh, min_tris: usize) -> Mesh {
    let n = mesh.positions.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for t in mesh.indices.chunks(3) {
        let (a, b, c) = (t[0], t[1], t[2]);
        let (ra, rb) = (uf_find(&mut parent, a), uf_find(&mut parent, b));
        parent[ra as usize] = rb;
        let (rb, rc) = (uf_find(&mut parent, b), uf_find(&mut parent, c));
        parent[rb as usize] = rc;
    }
    // triangle count per component root
    let mut count = vec![0u32; n];
    for t in mesh.indices.chunks(3) {
        let r = uf_find(&mut parent, t[0]);
        count[r as usize] += 1;
    }
    let mut map = vec![u32::MAX; n];
    let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for t in mesh.indices.chunks(3) {
        let r = uf_find(&mut parent, t[0]);
        if (count[r as usize] as usize) < min_tris {
            continue;
        }
        for &vi in t {
            if map[vi as usize] == u32::MAX {
                map[vi as usize] = positions.len() as u32;
                positions.push(mesh.positions[vi as usize]);
                normals.push(mesh.normals[vi as usize]);
            }
            indices.push(map[vi as usize]);
        }
    }
    Mesh { positions, normals, indices }
}
