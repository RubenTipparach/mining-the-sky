//! Night-side emission map: city glow (Gaussian in angular distance, sized by
//! population) plus faint road lighting. Sampled by the globe renderer to make
//! cities visible from space on the dark side.

use crate::grid::{dir_to_lonlat, lonlat_to_dir, lonlat_to_pixel, pixel_to_lonlat, Grid};
use crate::roads::Road;
use crate::sites::{City, CityKind};
use std::f64::consts::PI;

pub fn build_lights(w: usize, h: usize, cities: &[City], roads: &[Road]) -> Grid<f32> {
    let mut g = Grid::<f32>::new(w, h);

    // Faint road lighting first so cities sit on top.
    for r in roads {
        let inten = if r.major { 0.10 } else { 0.05 };
        for p in &r.pts {
            let (lon, lat) = dir_to_lonlat(*p);
            let (x, y) = lonlat_to_pixel(lon, lat, w, h);
            let i = y * w + x;
            g.data[i] = g.data[i].max(inten);
        }
    }

    // City glow, splatted within a local window for performance.
    for c in cities {
        let pop_m = (c.population / 1.0e6).max(0.05);
        let core = if c.kind == CityKind::Major { 1.5 } else { 0.8 };
        let sigma = (0.004 + 0.010 * pop_m.sqrt()).min(0.06); // radians
        let reach = sigma * 3.0;

        let (cx, cy) = lonlat_to_pixel(c.lon, c.lat, w, h);
        let dy_px = ((reach / PI) * h as f64).ceil() as i64 + 1;
        let lat_fac = c.lat.cos().max(0.15);
        let dx_px = ((reach / (2.0 * PI) / lat_fac) * w as f64).ceil() as i64 + 2;

        for dy in -dy_px..=dy_px {
            let yy = cy as i64 + dy;
            if yy < 0 || yy >= h as i64 {
                continue;
            }
            let yu = yy as usize;
            for dx in -dx_px..=dx_px {
                let xx = (cx as i64 + dx).rem_euclid(w as i64) as usize;
                let (lon, lat) = pixel_to_lonlat(xx, yu, w, h);
                let d = lonlat_to_dir(lon, lat);
                let ang = d.dot(c.dir).clamp(-1.0, 1.0).acos();
                let v = core * (-(ang * ang) / (2.0 * sigma * sigma)).exp();
                let i = yu * w + xx;
                g.data[i] += v as f32;
            }
        }
    }

    g
}
