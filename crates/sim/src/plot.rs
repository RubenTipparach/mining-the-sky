//! Simple CPU plot of the launch: planet, atmosphere, ascent track, final orbit.

use crate::ascent::LaunchResult;
use crate::body::CentralBody;
use image::{Rgb, RgbImage};

fn disc(img: &mut RgbImage, cx: i32, cy: i32, r: i32, col: Rgb<u8>) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let x = cx + dx;
                let y = cy + dy;
                if x >= 0 && x < w && y >= 0 && y < h {
                    img.put_pixel(x as u32, y as u32, col);
                }
            }
        }
    }
}

fn ring(img: &mut RgbImage, cx: i32, cy: i32, r: f64, col: Rgb<u8>) {
    let steps = (r * 6.3).max(64.0) as i32;
    for i in 0..steps {
        let a = i as f64 / steps as f64 * std::f64::consts::TAU;
        let x = cx + (r * a.cos()) as i32;
        let y = cy + (r * a.sin()) as i32;
        if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
            img.put_pixel(x as u32, y as u32, col);
        }
    }
}

fn line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, col: Rgb<u8>) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    loop {
        if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
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

pub fn write_launch_plot(body: &CentralBody, res: &LaunchResult, path: &str) {
    let size = 900i32;
    let mut img = RgbImage::from_pixel(size as u32, size as u32, Rgb([3, 4, 9]));
    let cx = size / 2;
    let cy = size / 2;

    // scale so the final orbit fits comfortably
    let view_r = (res.final_orbit.ra).max(body.radius + res.target_apo_alt) * 1.18;
    let scale = (size as f64 * 0.5) / view_r;

    // atmosphere shell + planet
    disc(&mut img, cx, cy, ((body.radius + body.atmo_top) * scale) as i32, Rgb([10, 22, 40]));
    disc(&mut img, cx, cy, (body.radius * scale) as i32, Rgb([26, 60, 44]));

    // final orbit
    if res.reached_orbit {
        ring(&mut img, cx, cy, res.final_orbit.ra * scale, Rgb([90, 200, 255]));
    } else {
        ring(&mut img, cx, cy, res.final_orbit.ra * scale, Rgb([200, 90, 70]));
    }

    // ascent track, projected onto (east0, up0)
    let project = |r: glam::DVec3| -> (i32, i32) {
        let px = r.dot(res.east0) * scale;
        let py = r.dot(res.up0) * scale;
        (cx + px as i32, cy - py as i32)
    };
    let mut prev: Option<(i32, i32)> = None;
    for s in &res.samples {
        let p = project(s.r);
        if let Some(pp) = prev {
            line(&mut img, pp.0, pp.1, p.0, p.1, Rgb([255, 180, 70]));
        }
        prev = Some(p);
    }

    // MECO marker
    if let Some(m) = res.meco {
        let p = project(m.r);
        disc(&mut img, p.0, p.1, 4, Rgb([255, 240, 120]));
    }

    img.save(path).unwrap();
}
