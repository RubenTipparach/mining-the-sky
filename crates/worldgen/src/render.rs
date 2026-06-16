//! CPU preview renderers. These exist so the procedural generation can be
//! verified and shown as images without a GPU. The real-time view will be a
//! wgpu/WebGPU app; the math here (albedo, day/night terminator, city lights,
//! atmosphere limb) is the reference the GPU shaders will match.

use crate::grid::{dir_to_lonlat, lonlat_to_dir, lonlat_to_pixel, pixel_to_lonlat};
use crate::planet::{albedo_at, mix3};
use crate::sites::CityKind;
use crate::World;
use glam::{DMat3, DQuat, DVec3};
use image::{Rgb, Rgba, RgbImage, RgbaImage};
use rayon::prelude::*;

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn tonemap(c: [f32; 3], exposure: f32) -> Rgb<u8> {
    let f = |v: f32| {
        let m = 1.0 - (-v * exposure).exp();
        (m.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
    };
    Rgb([f(c[0]), f(c[1]), f(c[2])])
}

#[inline]
fn star(x: u32, y: u32) -> f32 {
    let mut h = x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663);
    h ^= h >> 13;
    h = h.wrapping_mul(0x9E37_79B1);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32)
}

// ---------------------------------------------------------------------------
// Equirectangular maps
// ---------------------------------------------------------------------------

pub fn write_height_map(world: &World, path: &str) {
    let w = world.cfg.width;
    let h = world.cfg.height;
    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let e = world.elevation.get(x, y);
            let px = if e <= world.sea_level {
                let t = ((world.sea_level - e) / 0.6).clamp(0.0, 1.0);
                Rgb([(40.0 * (1.0 - t)) as u8, (60.0 * (1.0 - t)) as u8, 120])
            } else {
                let t = ((e - world.sea_level) / 0.7).clamp(0.0, 1.0);
                let v = (60.0 + 195.0 * t) as u8;
                Rgb([v, v, v])
            };
            img.put_pixel(x as u32, y as u32, px);
        }
    }
    img.save(path).unwrap();
}

pub fn write_day_map(world: &World, path: &str) {
    let w = world.cfg.width;
    let h = world.cfg.height;
    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let (_, lat) = pixel_to_lonlat(x, y, w, h);
            let alb = albedo_at(
                world.elevation.get(x, y),
                world.sea_level,
                lat,
                world.hydro.flow.get(x, y),
            );
            img.put_pixel(x as u32, y as u32, tonemap(alb, 2.2));
        }
    }
    img.save(path).unwrap();
}

pub fn write_night_map(world: &World, path: &str) {
    let w = world.cfg.width;
    let h = world.cfg.height;
    let mut img = RgbImage::new(w as u32, h as u32);
    let light = [1.0, 0.82, 0.5];
    for y in 0..h {
        for x in 0..w {
            let em = world.lights.get(x, y);
            // faint dark-ocean/landmass backdrop so the shape reads
            let base = if world.elevation.get(x, y) <= world.sea_level {
                [0.01, 0.02, 0.05]
            } else {
                [0.03, 0.035, 0.03]
            };
            let c = [
                base[0] + light[0] * em,
                base[1] + light[1] * em,
                base[2] + light[2] * em,
            ];
            img.put_pixel(x as u32, y as u32, tonemap(c, 1.6));
        }
    }
    img.save(path).unwrap();
}

/// Day map with city dots, roads, and the launch complex highlighted.
pub fn write_cities_roads_map(world: &World, path: &str) {
    let w = world.cfg.width;
    let h = world.cfg.height;
    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let (_, lat) = pixel_to_lonlat(x, y, w, h);
            let alb = albedo_at(
                world.elevation.get(x, y),
                world.sea_level,
                lat,
                world.hydro.flow.get(x, y),
            );
            // darken a touch so overlays pop
            img.put_pixel(x as u32, y as u32, tonemap(mix3(alb, [0.0; 3], 0.25), 2.2));
        }
    }
    // roads
    for r in &world.roads {
        let col = if r.major { Rgb([255, 170, 60]) } else { Rgb([200, 120, 40]) };
        for p in &r.pts {
            let (lon, lat) = dir_to_lonlat(*p);
            let (x, y) = lonlat_to_pixel(lon, lat, w, h);
            img.put_pixel(x as u32, y as u32, col);
        }
    }
    // cities
    for c in &world.cities {
        let (x, y) = lonlat_to_pixel(c.lon, c.lat, w, h);
        let (rad, col) = match c.kind {
            CityKind::Major => (4i64, Rgb([255, 80, 60])),
            CityKind::Minor => (2i64, Rgb([255, 200, 90])),
        };
        disc(&mut img, x as i64, y as i64, rad, col);
    }
    // launch complex
    let (lx, ly) = lonlat_to_pixel(world.launch.lon, world.launch.lat, w, h);
    cross(&mut img, lx as i64, ly as i64, 9, Rgb([90, 220, 255]));
    img.save(path).unwrap();
}

fn disc(img: &mut RgbImage, cx: i64, cy: i64, r: i64, col: Rgb<u8>) {
    let w = img.width() as i64;
    let h = img.height() as i64;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r * r {
                continue;
            }
            let x = (cx + dx).rem_euclid(w);
            let y = cy + dy;
            if y >= 0 && y < h {
                img.put_pixel(x as u32, y as u32, col);
            }
        }
    }
}

fn cross(img: &mut RgbImage, cx: i64, cy: i64, r: i64, col: Rgb<u8>) {
    let w = img.width() as i64;
    let h = img.height() as i64;
    for d in -r..=r {
        for (x, y) in [(cx + d, cy), (cx, cy + d)] {
            let xx = x.rem_euclid(w);
            if y >= 0 && y < h {
                img.put_pixel(xx as u32, y as u32, col);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Globe (orthographic, "from space")
// ---------------------------------------------------------------------------

pub struct GlobeParams {
    pub size: u32,
    /// Sun direction in *view* space (camera looks down -Z, +Z toward viewer).
    pub sun: DVec3,
    /// Direction (planet space) that should face the camera.
    pub center: DVec3,
    pub night_intensity: f32,
    pub exposure: f32,
}

pub fn render_globe(world: &World, p: &GlobeParams) -> RgbImage {
    let size = p.size;
    let sun = p.sun.normalize();
    // Rotation mapping the planet `center` direction to view +Z.
    let r = DMat3::from_quat(DQuat::from_rotation_arc(
        p.center.normalize(),
        DVec3::Z,
    ));
    let rt = r.transpose(); // view-normal -> planet direction
    let margin = 1.25f64;
    let light_col = [1.0f32, 0.82, 0.5];
    let atmo = [0.30f32, 0.5, 1.0];
    let sun_col = [1.05f32, 1.02, 0.95];

    let rows: Vec<u8> = (0..size)
        .into_par_iter()
        .flat_map_iter(|y| {
            let mut row = Vec::with_capacity(size as usize * 3);
            for x in 0..size {
                let u = ((x as f64 + 0.5) / size as f64 * 2.0 - 1.0) * margin;
                let v = ((y as f64 + 0.5) / size as f64 * 2.0 - 1.0) * margin;
                let cx = u;
                let cy = -v;
                let r2 = cx * cx + cy * cy;
                let col = if r2 <= 1.0 {
                    let nz = (1.0 - r2).sqrt();
                    let nv = DVec3::new(cx, cy, nz); // view-space surface normal
                    let pdir = (rt * nv).normalize();
                    let (_, lat) = dir_to_lonlat(pdir);
                    let elev = world.elevation.sample_dir(pdir);
                    let flow = world.hydro.flow.sample_dir(pdir);
                    let em = world.lights.sample_dir(pdir);
                    let alb = albedo_at(elev, world.sea_level, lat, flow);

                    let ndl = nv.dot(sun) as f32;
                    let day = smoothstep(-0.06, 0.16, ndl);
                    let diffuse = day * (0.12 + 0.88 * ndl.max(0.0));

                    let mut c = [
                        alb[0] * sun_col[0] * diffuse,
                        alb[1] * sun_col[1] * diffuse,
                        alb[2] * sun_col[2] * diffuse,
                    ];
                    // ocean specular glint
                    if elev <= world.sea_level {
                        let view = DVec3::Z;
                        let half = (sun + view).normalize();
                        let spec = nv.dot(half).max(0.0).powf(60.0) as f32 * day;
                        c[0] += spec * 0.8;
                        c[1] += spec * 0.8;
                        c[2] += spec * 0.72;
                    }
                    // city lights on the dark side
                    let night = 1.0 - day;
                    let l = em * night * p.night_intensity;
                    c[0] += light_col[0] * l;
                    c[1] += light_col[1] * l;
                    c[2] += light_col[2] * l;
                    // atmosphere limb glow (fresnel-ish)
                    let rim = (1.0 - nz as f32).powf(3.0);
                    let rim_lit = rim * (0.6 * day + 0.04);
                    c[0] += atmo[0] * rim_lit;
                    c[1] += atmo[1] * rim_lit;
                    c[2] += atmo[2] * rim_lit;
                    c
                } else {
                    // outside the disk: atmosphere halo, then stars
                    let r = r2.sqrt();
                    if r < 1.06 {
                        let nv = DVec3::new(cx / r, cy / r, 0.0);
                        let ndl = nv.dot(sun).max(0.0) as f32;
                        let d = ((r - 1.0) / 0.06) as f32;
                        let glow = (1.0 - d).clamp(0.0, 1.0).powf(2.0);
                        let i = glow * (ndl * 0.9 + 0.05);
                        [atmo[0] * i, atmo[1] * i, atmo[2] * i]
                    } else if star(x, y) > 0.9992 {
                        let b = 0.5 + 0.5 * star(x.wrapping_add(7), y.wrapping_add(3));
                        [b, b, b]
                    } else {
                        [0.0, 0.0, 0.0]
                    }
                };
                let px = tonemap(col, p.exposure);
                row.extend_from_slice(&px.0);
            }
            row
        })
        .collect();

    RgbImage::from_raw(size, size, rows).expect("globe buffer size mismatch")
}

/// Bake an equirectangular RGBA texture for the GPU client:
/// RGB = sRGB-encoded surface albedo, A = city-light emission (0..1). The app
/// samples this so the real-time planet shows the same generated world, with
/// city lights on the dark side.
pub fn write_planet_texture(world: &World, path: &str, tw: usize, th: usize) {
    let enc = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
    let mut img = RgbaImage::new(tw as u32, th as u32);
    for y in 0..th {
        for x in 0..tw {
            let (lon, lat) = pixel_to_lonlat(x, y, tw, th);
            let d = lonlat_to_dir(lon, lat);
            let e = world.elevation.sample_dir(d);
            let f = world.hydro.flow.sample_dir(d);
            let alb = albedo_at(e, world.sea_level, lat, f);
            let em = world.lights.sample_dir(d).clamp(0.0, 1.0);
            img.put_pixel(
                x as u32,
                y as u32,
                Rgba([enc(alb[0]), enc(alb[1]), enc(alb[2]), (em * 255.0).round() as u8]),
            );
        }
    }
    img.save(path).unwrap();
}

pub fn write_globe_day(world: &World, path: &str, size: u32) {
    let p = GlobeParams {
        size,
        sun: DVec3::new(0.5, 0.35, 0.78),
        center: world.launch.dir,
        night_intensity: 1.4,
        exposure: 1.1,
    };
    render_globe(world, &p).save(path).unwrap();
}

pub fn write_globe_night(world: &World, path: &str, size: u32) {
    let p = GlobeParams {
        size,
        sun: DVec3::new(-0.35, 0.22, -0.90),
        center: world.launch.dir,
        night_intensity: 1.7,
        exposure: 1.2,
    };
    render_globe(world, &p).save(path).unwrap();
}
