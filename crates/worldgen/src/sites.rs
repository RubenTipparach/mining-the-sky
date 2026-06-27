//! City and launch-complex placement.
//!
//! Major cities sit on coastal river deltas (high flow accumulation + adjacent
//! to ocean, near sea level). Minor cities follow inland river corridors. The
//! launch complex is placed at the most equatorial major city, nudged to a
//! coast with open ocean to the east (efficient prograde launches downrange).

use crate::grid::{haversine, lonlat_to_dir, pixel_to_lonlat, Grid};
use crate::hydrology::Hydrology;
use crate::rng::{city_name, Rng};
use glam::DVec3;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CityKind {
    Major,
    Minor,
}

#[derive(Clone)]
pub struct City {
    pub lon: f64,
    pub lat: f64,
    pub dir: DVec3,
    pub population: f64,
    pub kind: CityKind,
    pub name: String,
}

#[derive(Clone)]
pub struct LaunchSite {
    pub lon: f64,
    pub lat: f64,
    pub dir: DVec3,
    pub name: String,
}

pub struct Sites {
    pub cities: Vec<City>,
    pub launch: LaunchSite,
}

fn coastal(hydro: &Hydrology, x: usize, y: usize) -> bool {
    let w = hydro.is_ocean.w;
    let h = hydro.is_ocean.h;
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let nx = (x as i64 + dx).rem_euclid(w as i64) as usize;
        let ny = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
        if hydro.is_ocean.get(nx, ny) == 1 {
            return true;
        }
    }
    false
}

/// Coastal land cells near sea level, scored by upstream river flow.
fn delta_candidates(elev: &Grid<f32>, hydro: &Hydrology, sea_level: f32) -> Vec<(usize, f32)> {
    let w = elev.w;
    let h = elev.h;
    let mut out = Vec::new();
    for y in 1..h - 1 {
        for x in 0..w {
            let i = y * w + x;
            if hydro.is_ocean.data[i] == 1 {
                continue;
            }
            if elev.data[i] - sea_level > 0.06 {
                continue; // deltas are low and flat
            }
            if !coastal(hydro, x, y) {
                continue;
            }
            out.push((i, hydro.flow.data[i]));
        }
    }
    out
}

pub fn place_sites(
    elev: &Grid<f32>,
    hydro: &Hydrology,
    sea_level: f32,
    n_major: usize,
    n_minor: usize,
    seed: u64,
) -> Sites {
    let w = elev.w;
    let h = elev.h;
    let mut rng = Rng::new(seed ^ 0x51ED_BEEF);

    // --- Major cities: best-flow coastal deltas, spaced apart. ---
    let mut cand = delta_candidates(elev, hydro, sea_level);
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let major_sep = 0.18; // radians of great-circle separation
    let mut majors: Vec<City> = Vec::new();
    for (i, _) in &cand {
        if majors.len() >= n_major {
            break;
        }
        let (lon, lat) = pixel_to_lonlat(i % w, i / w, w, h);
        if majors.iter().all(|c| haversine(c.lon, c.lat, lon, lat) > major_sep) {
            majors.push(City {
                lon,
                lat,
                dir: lonlat_to_dir(lon, lat),
                population: rng.range(3.0e6, 20.0e6),
                kind: CityKind::Major,
                name: city_name(&mut rng),
            });
        }
    }

    // --- Minor cities: inland river corridors, spaced apart. ---
    let mut river: Vec<(usize, f32)> = (0..w * h)
        .filter(|&i| hydro.is_ocean.data[i] == 0 && hydro.flow.data[i] > 2.0)
        .map(|i| (i, hydro.flow.data[i]))
        .collect();
    river.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let minor_sep = 0.085;
    let mut minors: Vec<City> = Vec::new();
    for (i, _) in &river {
        if minors.len() >= n_minor {
            break;
        }
        let (lon, lat) = pixel_to_lonlat(i % w, i / w, w, h);
        let far_major = majors.iter().all(|c| haversine(c.lon, c.lat, lon, lat) > minor_sep);
        let far_minor = minors.iter().all(|c| haversine(c.lon, c.lat, lon, lat) > minor_sep);
        if far_major && far_minor {
            minors.push(City {
                lon,
                lat,
                dir: lonlat_to_dir(lon, lat),
                population: rng.range(0.3e6, 3.0e6),
                kind: CityKind::Minor,
                name: city_name(&mut rng),
            });
        }
    }

    let launch = pick_launch(elev, hydro, sea_level, &majors);

    let mut cities = majors;
    cities.extend(minors);
    Sites { cities, launch }
}

fn ocean_to_east(elev: &Grid<f32>, hydro: &Hydrology, x: usize, y: usize) -> bool {
    let w = elev.w;
    for step in 1..=8i64 {
        let nx = (x as i64 + step).rem_euclid(w as i64) as usize;
        if hydro.is_ocean.get(nx, y) == 1 {
            return true;
        }
    }
    false
}

fn pick_launch(
    elev: &Grid<f32>,
    hydro: &Hydrology,
    sea_level: f32,
    majors: &[City],
) -> LaunchSite {
    let w = elev.w;
    let h = elev.h;

    // Most equatorial major city as the anchor.
    let anchor = majors
        .iter()
        .min_by(|a, b| a.lat.abs().partial_cmp(&b.lat.abs()).unwrap())
        .cloned();

    if let Some(c) = anchor {
        let (cx, cy) = crate::grid::lonlat_to_pixel(c.lon, c.lat, w, h);
        // Search a window around the anchor for a coastal pad with ocean east.
        let mut best: Option<(usize, usize)> = None;
        let mut best_score = f32::NEG_INFINITY;
        let rad = 60i64;
        for dy in -rad..=rad {
            let yy = (cy as i64 + dy).clamp(1, h as i64 - 2) as usize;
            for dx in -rad..=rad {
                let xx = (cx as i64 + dx).rem_euclid(w as i64) as usize;
                if hydro.is_ocean.get(xx, yy) == 1 {
                    continue;
                }
                if elev.get(xx, yy) - sea_level > 0.05 {
                    continue;
                }
                if !coastal(hydro, xx, yy) || !ocean_to_east(elev, hydro, xx, yy) {
                    continue;
                }
                let (lon, lat) = pixel_to_lonlat(xx, yy, w, h);
                // Prefer near the equator and near the anchor city.
                let score = -(lat.abs() as f32) * 2.0
                    - haversine(c.lon, c.lat, lon, lat) as f32;
                if score > best_score {
                    best_score = score;
                    best = Some((xx, yy));
                }
            }
        }
        if let Some((xx, yy)) = best {
            let (lon, lat) = pixel_to_lonlat(xx, yy, w, h);
            return LaunchSite {
                lon,
                lat,
                dir: lonlat_to_dir(lon, lat),
                name: format!("{} Spaceport", c.name),
            };
        }
        return LaunchSite {
            lon: c.lon,
            lat: c.lat,
            dir: c.dir,
            name: format!("{} Spaceport", c.name),
        };
    }

    LaunchSite {
        lon: 0.0,
        lat: 0.0,
        dir: lonlat_to_dir(0.0, 0.0),
        name: "Primary Spaceport".to_string(),
    }
}
