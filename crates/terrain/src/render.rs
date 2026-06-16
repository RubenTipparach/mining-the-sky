//! CPU verification renders for the LOD system (no GPU needed):
//! - `write_lod_map`: top-down view of the active patches, coloured by LOD
//!   depth, proving the quadtree refines toward the camera.
//! - `write_relief`: hillshaded elevation around the camera, proving the
//!   procedural terrain is continuous and detailed at any zoom.

use crate::cubesphere::tangent_basis;
use crate::elevation::Elevation;
use crate::quadtree::{Lod, Planet};
use glam::DVec3;
use image::{Rgb, RgbImage};

fn depth_color(depth: u32, maxd: u32) -> Rgb<u8> {
    // cycle hue with depth so each LOD level is visually distinct
    let t = if maxd == 0 { 0.0 } else { depth as f32 / maxd as f32 };
    let h = t * 0.85; // 0=red -> through spectrum
    let (r, g, b) = hsv(h, 0.85, 1.0);
    Rgb([r, g, b])
}

fn hsv(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

fn line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, col: Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, col);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

struct Proj {
    center: DVec3,
    east: DVec3,
    north: DVec3,
    scale: f64, // pixels per metre
    cx: f64,
    cy: f64,
}

impl Proj {
    fn new(cam: DVec3, planet: &Planet, extent_m: f64, size: u32) -> Self {
        let dir = cam.normalize();
        let center = dir * planet.radius;
        let (east, north) = tangent_basis(dir);
        Proj {
            center,
            east,
            north,
            scale: (size as f64 * 0.5) / extent_m,
            cx: size as f64 * 0.5,
            cy: size as f64 * 0.5,
        }
    }
    fn px(&self, p: DVec3) -> (i32, i32) {
        let rel = p - self.center;
        let e = rel.dot(self.east);
        let n = rel.dot(self.north);
        ((self.cx + e * self.scale) as i32, (self.cy - n * self.scale) as i32)
    }
}

/// Top-down map of the active LOD patches around the camera, coloured by depth.
pub fn write_lod_map(planet: &Planet, lod: &Lod, cam: DVec3, extent_m: f64, size: u32, path: &str) {
    let mut img = RgbImage::from_pixel(size, size, Rgb([6, 8, 14]));
    let proj = Proj::new(cam, planet, extent_m, size);
    let maxd = lod.max_depth_reached;

    for patch in &lod.patches {
        // skip patches far outside the framed area (cheap cull by centre)
        let (mx, my) = proj.px(patch.center);
        if mx < -64 || mx > size as i32 + 64 || my < -64 || my > size as i32 + 64 {
            continue;
        }
        let col = depth_color(patch.depth, maxd);
        let c = |u, v| {
            let d = crate::cubesphere::face_dir(patch.face, u, v) * planet.radius;
            proj.px(d)
        };
        let a = c(patch.u0, patch.v0);
        let b = c(patch.u1, patch.v0);
        let d = c(patch.u1, patch.v1);
        let e = c(patch.u0, patch.v1);
        line(&mut img, a.0, a.1, b.0, b.1, col);
        line(&mut img, b.0, b.1, d.0, d.1, col);
        line(&mut img, d.0, d.1, e.0, e.1, col);
        line(&mut img, e.0, e.1, a.0, a.1, col);
    }

    // camera ground marker
    let cc = (proj.cx as i32, proj.cy as i32);
    for d in -5..=5 {
        line(&mut img, cc.0 + d, cc.1, cc.0 + d, cc.1, Rgb([255, 255, 255]));
        line(&mut img, cc.0, cc.1 + d, cc.0, cc.1 + d, Rgb([255, 255, 255]));
    }
    img.save(path).unwrap();
}

/// Hillshaded elevation around the camera, sampled continuously from the
/// procedural field (proves seamlessness and detail at any zoom).
pub fn write_relief(
    planet: &Planet,
    elev: &Elevation,
    cam: DVec3,
    extent_m: f64,
    size: u32,
    path: &str,
) {
    let dir = cam.normalize();
    let center = dir * planet.radius;
    let (east, north) = tangent_basis(dir);
    let mpp = (2.0 * extent_m) / size as f64; // metres per pixel
    let sun = DVec3::new(-0.5, 0.8, 0.4).normalize();

    let height_at = |px: f64, py: f64| -> f64 {
        let world = center + east * px + north * py;
        elev.height_m(world.normalize())
    };

    let mut img = RgbImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let ex = (x as f64 - size as f64 * 0.5) * mpp;
            let ny = (size as f64 * 0.5 - y as f64) * mpp;
            let h = height_at(ex, ny);
            // gradient for hillshade
            let hx = height_at(ex + mpp, ny) - height_at(ex - mpp, ny);
            let hy = height_at(ex, ny + mpp) - height_at(ex, ny - mpp);
            let normal = DVec3::new(-hx, 2.0 * mpp, -hy).normalize();
            let shade = (normal.dot(sun).max(0.0) * 0.85 + 0.15) as f32;

            let base = if h <= 0.0 {
                [0.05, 0.18, 0.32]
            } else {
                let t = (h / 4000.0).clamp(0.0, 1.0) as f32;
                let low = [0.20, 0.36, 0.16];
                let mid = [0.40, 0.34, 0.22];
                let hi = [0.85, 0.86, 0.90];
                if t < 0.5 {
                    mix(low, mid, t * 2.0)
                } else {
                    mix(mid, hi, (t - 0.5) * 2.0)
                }
            };
            let c = [base[0] * shade, base[1] * shade, base[2] * shade];
            img.put_pixel(
                x,
                y,
                Rgb([
                    (c[0] * 255.0) as u8,
                    (c[1] * 255.0) as u8,
                    (c[2] * 255.0) as u8,
                ]),
            );
        }
    }
    img.save(path).unwrap();
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}
