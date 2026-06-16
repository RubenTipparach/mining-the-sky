//! Octree LOD dual-contouring terrain: subdivide finer near the camera, mesh
//! each surface-bearing leaf with Surface Nets, and show both the meshed terrain
//! and the LOD structure (chunks coloured + wireframed by depth).
//!
//! Run: cargo run -p terrain --bin octree --release

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use terrain::octree::{select, Leaf};
use terrain::raster::{self, Frame};
use terrain::surfacenets::surface_nets;

fn main() {
    // domain-warped heightfield (terrain with occasional overhangs)
    let hills = Fbm::<Perlin>::new(7).set_octaves(6).set_frequency(1.0).set_persistence(0.5).set_lacunarity(2.1);
    let warp = Fbm::<Perlin>::new(23).set_octaves(4).set_frequency(1.0).set_persistence(0.5).set_lacunarity(2.2);
    let height = move |x: f32, z: f32| -> f32 {
        let h = (hills.get([(x * 0.0009) as f64, (z * 0.0009) as f64]) as f32) * 950.0;
        let d = (warp.get([(x * 0.006) as f64, (z * 0.006) as f64]) as f32) * 120.0;
        h + d
    };
    // solid heightfield (the only gaps in the render are the LOD-boundary seams).
    let density = move |p: Vec3| -> f32 { p.y - height(p.x, p.z) };

    // a big region; the octree refines toward the camera
    let region = 9000.0f32;
    let region_min = Vec3::splat(-region * 0.5);
    let cam = Vec3::new(-3600.0, 900.0, -3600.0);
    let split = 1.4;
    let max_depth = 6;
    let leaf_n = 14usize; // DC grid per leaf

    let t0 = std::time::Instant::now();
    let leaves = select(&density, cam, region_min, region, split, max_depth);
    let maxd = leaves.iter().map(|l| l.depth).max().unwrap_or(1).max(1);

    // mesh each leaf; concatenate, with per-vertex terrain + LOD colours
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut col_terrain = Vec::new();
    let mut col_lod = Vec::new();
    let mut idx = Vec::new();
    let mut per_depth = vec![0u32; (max_depth + 1) as usize];
    for l in &leaves {
        per_depth[l.depth as usize] += 1;
        let cell = l.size / (leaf_n as f32 - 1.0);
        // mesh the whole leaf chunk (don't trim edges - that punches holes)
        let m = surface_nets(&density, l.min, cell, leaf_n);
        let base = pos.len() as u32;
        let lod_c = depth_color(l.depth, maxd);
        for (p, n) in m.positions.iter().zip(m.normals.iter()) {
            pos.push(*p);
            nrm.push(*n);
            col_terrain.push(terrain_color(p.y, *n));
            col_lod.push(lod_c);
        }
        idx.extend(m.indices.iter().map(|i| i + base));
    }
    let dt = t0.elapsed();

    println!("Octree dual-contouring LOD");
    print!("leaves/depth: ");
    for (d, c) in per_depth.iter().enumerate() {
        if *c > 0 {
            print!("L{d}={c} ");
        }
    }
    println!();
    let finest = leaves.iter().map(|l| l.size).fold(f32::INFINITY, f32::min);
    let coarsest = leaves.iter().map(|l| l.size).fold(0.0, f32::max);
    println!("leaves:       {}", leaves.len());
    println!("leaf size:    {finest:.0} m (fine) .. {coarsest:.0} m (coarse)");
    println!("triangles:    {}", idx.len() / 3);
    println!("build+mesh:   {dt:?}");

    let sun = Vec3::new(-0.4, 0.85, 0.3).normalize();
    let (w, h) = (1100u32, 740u32);
    let eye = cam + Vec3::new(60.0, 60.0, 60.0);
    let target = Vec3::new(700.0, -120.0, 700.0);
    let m = raster::mvp(eye, target, 55.0, w, h);

    std::fs::create_dir_all("out").unwrap();
    let mut f = Frame::new(w, h, true);
    raster::raster(&mut f, m, &pos, &nrm, &col_terrain, &idx, sun);
    f.save("out/octree_terrain.png");

    // LOD view: chunk colour by depth + wireframe boxes
    let mut g = Frame::new(w, h, false);
    raster::raster(&mut g, m, &pos, &nrm, &col_lod, &idx, sun);
    for l in &leaves {
        draw_box(&mut g, m, l, depth_color(l.depth, maxd), w, h);
    }
    g.save("out/octree_lod.png");
    println!("wrote out/octree_terrain.png and out/octree_lod.png");
}

fn draw_box(f: &mut Frame, m: glam::Mat4, l: &Leaf, col: [f32; 3], w: u32, h: u32) {
    let c = [
        l.min,
        l.min + Vec3::new(l.size, 0.0, 0.0),
        l.min + Vec3::new(l.size, l.size, 0.0),
        l.min + Vec3::new(0.0, l.size, 0.0),
        l.min + Vec3::new(0.0, 0.0, l.size),
        l.min + Vec3::new(l.size, 0.0, l.size),
        l.min + Vec3::new(l.size, l.size, l.size),
        l.min + Vec3::new(0.0, l.size, l.size),
    ];
    let e = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    for (a, b) in e {
        if let (Some(sa), Some(sb)) = (raster::screen(m, c[a], w, h), raster::screen(m, c[b], w, h)) {
            raster::line(f, sa, sb, col);
        }
    }
}

fn depth_color(d: u32, maxd: u32) -> [f32; 3] {
    let t = d as f32 / maxd as f32;
    hsv(t * 0.8, 0.8, 1.0)
}

fn terrain_color(y: f32, n: Vec3) -> [f32; 3] {
    let t = ((y + 230.0) / 460.0).clamp(0.0, 1.0);
    let grass = [0.22, 0.34, 0.16];
    let rock = [0.40, 0.36, 0.32];
    let snow = [0.85, 0.87, 0.92];
    let slope = 1.0 - n.y.abs();
    let mut c = mix(grass, rock, (slope * 1.6).clamp(0.0, 1.0));
    c = mix(c, snow, ((t - 0.72) / 0.28).clamp(0.0, 1.0));
    c
}

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    match (i as i32).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}
fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}
