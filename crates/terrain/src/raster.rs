//! A tiny CPU rasterizer (z-buffer, per-vertex colour + normal, Lambert) plus a
//! line drawer, shared by the dual-contouring experiments to render meshes and
//! LOD wireframes to PNG without a GPU.

use glam::{Mat4, Vec3, Vec4Swizzles};
use image::{Rgb, RgbImage};

pub struct Frame {
    pub w: u32,
    pub h: u32,
    color: Vec<[f32; 3]>,
    depth: Vec<f32>,
}

impl Frame {
    /// New frame with a sky gradient (or flat dark if `sky` is false).
    pub fn new(w: u32, h: u32, sky: bool) -> Self {
        let mut color = vec![[0.02, 0.03, 0.05]; (w * h) as usize];
        if sky {
            for y in 0..h {
                let t = y as f32 / h as f32;
                let c = [0.55 - 0.2 * t, 0.66 - 0.12 * t, 0.82 - 0.05 * t];
                for x in 0..w {
                    color[(y * w + x) as usize] = c;
                }
            }
        }
        Frame { w, h, color, depth: vec![f32::INFINITY; (w * h) as usize] }
    }

    pub fn save(&self, path: &str) {
        let mut img = RgbImage::new(self.w, self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                let c = self.color[(y * self.w + x) as usize];
                img.put_pixel(x, y, Rgb([enc(c[0]), enc(c[1]), enc(c[2])]));
            }
        }
        img.save(path).unwrap();
    }
}

pub fn mvp(eye: Vec3, target: Vec3, fov_deg: f32, w: u32, h: u32) -> Mat4 {
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    let proj = Mat4::perspective_rh(fov_deg.to_radians(), w as f32 / h as f32, 1.0, 1.0e6);
    proj * view
}

pub fn screen(mvp: Mat4, p: Vec3, w: u32, h: u32) -> Option<(f32, f32, f32)> {
    let c = mvp * p.extend(1.0);
    if c.w <= 0.0 {
        return None;
    }
    let n = c.xyz() / c.w;
    Some(((n.x * 0.5 + 0.5) * w as f32, (1.0 - (n.y * 0.5 + 0.5)) * h as f32, n.z))
}

/// Rasterize an indexed triangle mesh with per-vertex colour + normal.
pub fn raster(
    f: &mut Frame,
    m: Mat4,
    pos: &[Vec3],
    nrm: &[Vec3],
    col: &[[f32; 3]],
    idx: &[u32],
    sun: Vec3,
) {
    let (w, h) = (f.w, f.h);
    for tri in idx.chunks(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (sa, sb, sc) = match (
            screen(m, pos[i0], w, h),
            screen(m, pos[i1], w, h),
            screen(m, pos[i2], w, h),
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => continue,
        };
        let area = edge(sa, sb, sc.0, sc.1);
        if area.abs() < 1e-6 {
            continue;
        }
        let minx = sa.0.min(sb.0).min(sc.0).floor().max(0.0) as u32;
        let maxx = sa.0.max(sb.0).max(sc.0).ceil().min(w as f32 - 1.0) as u32;
        let miny = sa.1.min(sb.1).min(sc.1).floor().max(0.0) as u32;
        let maxy = sa.1.max(sb.1).max(sc.1).ceil().min(h as f32 - 1.0) as u32;
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
                if z >= f.depth[di] {
                    continue;
                }
                f.depth[di] = z;
                let n = (nrm[i0] * w0 + nrm[i1] * w1 + nrm[i2] * w2).normalize_or_zero();
                let lit = n.dot(sun).max(0.0) * 0.85 + 0.18;
                let c = [
                    col[i0][0] * w0 + col[i1][0] * w1 + col[i2][0] * w2,
                    col[i0][1] * w0 + col[i1][1] * w1 + col[i2][1] * w2,
                    col[i0][2] * w0 + col[i1][2] * w1 + col[i2][2] * w2,
                ];
                f.color[di] = [c[0] * lit, c[1] * lit, c[2] * lit];
            }
        }
    }
}

/// Draw a screen-space line (no depth test), for LOD wireframes.
pub fn line(f: &mut Frame, a: (f32, f32, f32), b: (f32, f32, f32), col: [f32; 3]) {
    let (w, h) = (f.w as i32, f.h as i32);
    let (mut x0, mut y0) = (a.0 as i32, a.1 as i32);
    let (x1, y1) = (b.0 as i32, b.1 as i32);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
            f.color[(y0 * w + x0) as usize] = col;
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn edge(a: (f32, f32, f32), b: (f32, f32, f32), px: f32, py: f32) -> f32 {
    (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0)
}
fn enc(v: f32) -> u8 {
    (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
}
