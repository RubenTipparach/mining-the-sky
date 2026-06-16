//! Dual-contouring (Surface Nets) terrain experiment.
//!
//! Meshes the iso-surface of a 3D density field that is a rolling heightfield
//! warped by a y-dependent 3D noise, which makes the surface fold into
//! overhangs and arches - the thing a heightmap LOD can't do. Renders the mesh
//! with a tiny z-buffer rasterizer to prove it out.
//!
//! Run: cargo run -p terrain --bin dc --release

use glam::{Mat4, Vec3, Vec4Swizzles};
use image::{Rgb, RgbImage};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use terrain::surfacenets::{drop_small_components, surface_nets, Mesh};

fn main() {
    // --- density field: a 3D domain-warped heightfield ---
    // Base terrain is a solid heightfield (monotonic in y = one connected body,
    // no floaters). We then bend the horizontal lookup by a y-dependent 3D warp,
    // which tilts/folds columns into overhangs and arches while keeping the
    // surface a single connected sheet - a "proper" way to get overhangs.
    let hills = Fbm::<Perlin>::new(7).set_octaves(6).set_frequency(1.0).set_persistence(0.5).set_lacunarity(2.1);
    let ridge = Fbm::<Perlin>::new(101).set_octaves(5).set_frequency(1.0).set_persistence(0.5).set_lacunarity(2.2);
    let warp = Fbm::<Perlin>::new(23).set_octaves(4).set_frequency(1.0).set_persistence(0.5).set_lacunarity(2.2);

    let span = 120.0f32;
    let dim = 96usize;
    let cell = span / (dim as f32 - 1.0);
    let origin = Vec3::new(-span * 0.5, -span * 0.5, -span * 0.5);

    let height = |x: f32, z: f32| -> f32 {
        let hf = 0.020;
        let h = (hills.get([(x * hf) as f64, (z * hf) as f64]) as f32) * 16.0;
        // ridged detail for sharper relief
        let r = 1.0 - (ridge.get([(x * 0.05) as f64, (z * 0.05) as f64]) as f32).abs();
        h + r * r * 10.0
    };
    let density = |p: Vec3| -> f32 {
        // y-dependent horizontal warp -> occasional overhangs, surface stays
        // connected. Gentler amplitude reads as terrain, not melted blobs.
        let wf = 0.03;
        let amt = 14.0;
        let wx = (warp.get([(p.x * wf) as f64, (p.y * wf) as f64, (p.z * wf) as f64]) as f32) * amt;
        let wz = (warp.get([(p.x * wf + 5.1) as f64, (p.y * wf + 2.3) as f64, (p.z * wf - 1.7) as f64]) as f32) * amt;
        p.y - height(p.x + wx, p.z + wz)
    };

    let t0 = std::time::Instant::now();
    let raw = surface_nets(&density, origin, cell, dim);
    let mesh = drop_small_components(&raw, 250);
    let gen = t0.elapsed();

    println!("Surface Nets dual-contouring (domain-warped heightfield)");
    println!("grid:        {dim}^3 ({} cells)", (dim - 1).pow(3));
    println!("cell size:   {cell:.2} m");
    println!("vertices:    {} ({} before floater removal)", mesh.positions.len(), raw.positions.len());
    println!(
        "triangles:   {} ({} dropped as floaters)",
        mesh.indices.len() / 3,
        (raw.indices.len() - mesh.indices.len()) / 3
    );
    println!("gen time:    {gen:?}");

    std::fs::create_dir_all("out").unwrap();
    render(&mesh, "out/dc_terrain.png", 1100, 750);
    println!("wrote out/dc_terrain.png");
}

fn render(mesh: &Mesh, path: &str, w: u32, h: u32) {
    // low side camera so overhangs/undercuts are visible in profile
    let eye = Vec3::new(110.0, 22.0, 60.0);
    let target = Vec3::new(0.0, 2.0, 0.0);
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    let proj = Mat4::perspective_rh(50f32.to_radians(), w as f32 / h as f32, 1.0, 1000.0);
    let mvp = proj * view;
    let sun = Vec3::new(-0.4, 0.8, 0.35).normalize();

    let mut color = vec![[0.0f32; 3]; (w * h) as usize];
    let mut depth = vec![f32::INFINITY; (w * h) as usize];
    // sky gradient background
    for y in 0..h {
        let t = y as f32 / h as f32;
        let c = [0.55 - 0.2 * t, 0.66 - 0.12 * t, 0.82 - 0.05 * t];
        for x in 0..w {
            color[(y * w + x) as usize] = c;
        }
    }

    let to_screen = |p: Vec3| -> Option<(f32, f32, f32)> {
        let c = mvp * p.extend(1.0);
        if c.w <= 0.0 {
            return None;
        }
        let n = c.xyz() / c.w;
        Some(((n.x * 0.5 + 0.5) * w as f32, (1.0 - (n.y * 0.5 + 0.5)) * h as f32, n.z))
    };

    for tri in mesh.indices.chunks(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (a, b, c) = (mesh.positions[i0], mesh.positions[i1], mesh.positions[i2]);
        let (na, nb, nc) = (mesh.normals[i0], mesh.normals[i1], mesh.normals[i2]);
        let (sa, sb, sc) = match (to_screen(a), to_screen(b), to_screen(c)) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => continue,
        };
        let minx = sa.0.min(sb.0).min(sc.0).floor().max(0.0) as u32;
        let maxx = sa.0.max(sb.0).max(sc.0).ceil().min(w as f32 - 1.0) as u32;
        let miny = sa.1.min(sb.1).min(sc.1).floor().max(0.0) as u32;
        let maxy = sa.1.max(sb.1).max(sc.1).ceil().min(h as f32 - 1.0) as u32;
        let area = edge(sa, sb, sc.0, sc.1);
        if area.abs() < 1e-6 {
            continue;
        }
        for py in miny..=maxy {
            for px in minx..=maxx {
                let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
                let w0 = edge(sb, sc, fx, fy) / area;
                let w1 = edge(sc, sa, fx, fy) / area;
                let w2 = edge(sa, sb, fx, fy) / area;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * sa.2 + w1 * sb.2 + w2 * sc.2;
                let di = (py * w + px) as usize;
                if z >= depth[di] {
                    continue;
                }
                depth[di] = z;
                let n = (na * w0 + nb * w1 + nc * w2).normalize_or_zero();
                let wy = a.y * w0 + b.y * w1 + c.y * w2;
                let lit = n.dot(sun).max(0.0) * 0.85 + 0.15;
                // colour by height + slope
                let rock = [0.40, 0.36, 0.32];
                let grass = [0.24, 0.36, 0.18];
                let snow = [0.85, 0.87, 0.92];
                let t = ((wy + 26.0) / 52.0).clamp(0.0, 1.0);
                let slope = 1.0 - n.y.abs();
                let mut base = mix(grass, rock, (slope * 1.6).clamp(0.0, 1.0));
                base = mix(base, snow, ((t - 0.7) / 0.3).clamp(0.0, 1.0));
                color[di] = [base[0] * lit, base[1] * lit, base[2] * lit];
            }
        }
    }

    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let c = color[(y * w + x) as usize];
            img.put_pixel(x, y, Rgb([enc(c[0]), enc(c[1]), enc(c[2])]));
        }
    }
    img.save(path).unwrap();
}

fn edge(a: (f32, f32, f32), b: (f32, f32, f32), px: f32, py: f32) -> f32 {
    (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0)
}
fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}
fn enc(v: f32) -> u8 {
    (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
}
