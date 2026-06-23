//! Night-side emission map: city glow plus lit highways. Sampled by the globe
//! renderer to make cities visible from space on the dark side.
//!
//! Cities are splatted as several overlapping lobes (a bright core plus a few
//! smaller offset suburbs) instead of one circular Gaussian, so a metro reads as
//! an irregular, organic sprawl from orbit rather than a perfect dot. Highways
//! are drawn as bright, thickened threads so the road network is clearly lit
//! between cities.

use crate::grid::{dir_to_lonlat, lonlat_to_dir, lonlat_to_pixel, pixel_to_lonlat, Grid};
use crate::roads::Road;
use crate::sites::{City, CityKind};
use glam::DVec3;
use std::f64::consts::PI;

/// Deterministic 0..1 hash from two integer keys (per-city, per-lobe), so the
/// organic city shapes are stable across runs.
fn hash01(a: u64, b: u64) -> f64 {
    let mut h = a
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Splat one Gaussian lobe of light, centred on unit direction `dir`, into the
/// emission grid (added, so overlapping lobes build up a bright connected core).
fn splat_lobe(g: &mut Grid<f32>, w: usize, h: usize, dir: DVec3, sigma: f64, inten: f64) {
    let (lon, lat) = dir_to_lonlat(dir);
    let (cx, cy) = lonlat_to_pixel(lon, lat, w, h);
    let reach = sigma * 3.0;
    let dy_px = ((reach / PI) * h as f64).ceil() as i64 + 1;
    let lat_fac = lat.cos().max(0.15);
    let dx_px = ((reach / (2.0 * PI) / lat_fac) * w as f64).ceil() as i64 + 2;
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);

    for dy in -dy_px..=dy_px {
        let yy = cy as i64 + dy;
        if yy < 0 || yy >= h as i64 {
            continue;
        }
        let yu = yy as usize;
        for dx in -dx_px..=dx_px {
            let xx = (cx as i64 + dx).rem_euclid(w as i64) as usize;
            let (plon, plat) = pixel_to_lonlat(xx, yu, w, h);
            let pd = lonlat_to_dir(plon, plat);
            let ang = pd.dot(dir).clamp(-1.0, 1.0).acos();
            let v = inten * (-(ang * ang) * inv2s2).exp();
            g.data[yu * w + xx] += v as f32;
        }
    }
}

/// A local east/north tangent basis at unit direction `dir` (north pole = +Y), so
/// city sub-lobes can be offset organically across the surface.
fn tangent_basis(dir: DVec3) -> (DVec3, DVec3) {
    let up = if dir.y.abs() > 0.98 { DVec3::X } else { DVec3::Y };
    let east = up.cross(dir).normalize();
    let north = dir.cross(east).normalize();
    (east, north)
}

pub fn build_lights(w: usize, h: usize, cities: &[City], roads: &[Road]) -> Grid<f32> {
    let mut g = Grid::<f32>::new(w, h);

    // Lit highways first so city cores sit on top. Brighter than before and
    // thickened to a 3 px thread so the network reads clearly from orbit; the
    // centre line is brightest with a softer shoulder.
    for r in roads {
        let inten = if r.major { 0.34 } else { 0.20 };
        for p in &r.pts {
            let (lon, lat) = dir_to_lonlat(*p);
            let (x, y) = lonlat_to_pixel(lon, lat, w, h);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let xx = (x as i64 + dx).rem_euclid(w as i64) as usize;
                    let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                    let edge = dx != 0 || dy != 0;
                    let v = if edge { inten * 0.55 } else { inten };
                    let i = yy * w + xx;
                    g.data[i] = g.data[i].max(v as f32);
                }
            }
        }
    }

    // Organic city glow: a bright core lobe plus a few smaller, offset suburb
    // lobes, so the metro footprint is irregular rather than a perfect circle.
    for (ci, c) in cities.iter().enumerate() {
        let pop_m = (c.population / 1.0e6).max(0.05);
        let major = c.kind == CityKind::Major;
        let core = if major { 1.5 } else { 0.8 };
        let sigma = (0.004 + 0.010 * pop_m.sqrt()).min(0.06); // radians
        let (east, north) = tangent_basis(c.dir);
        let nlobes = if major { 5 } else { 2 };

        for li in 0..nlobes {
            if li == 0 {
                // bright central core, slightly squashed for an irregular shape
                splat_lobe(&mut g, w, h, c.dir, sigma, core);
                continue;
            }
            // suburb lobe: offset within ~2.6 sigma in a random direction, smaller
            // and dimmer than the core, so it reads as sprawl/satellite districts.
            let ang = hash01(ci as u64, li as u64 * 7 + 1) * 2.0 * PI;
            let dist = sigma * (0.6 + 1.9 * hash01(ci as u64, li as u64 * 13 + 3));
            let off = east * (ang.cos() * dist) + north * (ang.sin() * dist);
            let lobe_dir = (c.dir + off).normalize();
            let lobe_sigma = sigma * (0.35 + 0.45 * hash01(ci as u64, li as u64 * 17 + 5));
            let lobe_inten = core * (0.28 + 0.30 * hash01(ci as u64, li as u64 * 19 + 9));
            splat_lobe(&mut g, w, h, lobe_dir, lobe_sigma, lobe_inten);
        }
    }

    g
}
