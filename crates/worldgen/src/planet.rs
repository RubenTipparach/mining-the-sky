//! Procedural elevation field on the sphere and sea-level selection.

use crate::grid::{lonlat_to_dir, pixel_to_lonlat, Grid};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

/// Generate a seamless elevation field by sampling 3D fractal noise on the
/// unit sphere (no equirectangular seams or pole pinching). Values are in an
/// arbitrary signed range; sea level is chosen afterwards by percentile.
pub fn generate_elevation(w: usize, h: usize, seed: u32) -> Grid<f32> {
    let continents = Fbm::<Perlin>::new(seed)
        .set_octaves(7)
        .set_frequency(1.0)
        .set_persistence(0.5)
        .set_lacunarity(2.1);
    let detail = Fbm::<Perlin>::new(seed.wrapping_add(101))
        .set_octaves(6)
        .set_frequency(3.0)
        .set_persistence(0.5)
        .set_lacunarity(2.2);
    let warp = Fbm::<Perlin>::new(seed.wrapping_add(202))
        .set_octaves(4)
        .set_frequency(1.5)
        .set_persistence(0.5)
        .set_lacunarity(2.0);

    let mut g = Grid::<f32>::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (lon, lat) = pixel_to_lonlat(x, y, w, h);
            let p = lonlat_to_dir(lon, lat);
            // Domain warp the continent lookup for more organic coastlines.
            let wx = warp.get([p.x * 1.8, p.y * 1.8, p.z * 1.8]) * 0.18;
            let wy = warp.get([p.x * 1.8 + 5.3, p.y * 1.8 + 1.7, p.z * 1.8 - 2.1]) * 0.18;
            let wz = warp.get([p.x * 1.8 - 3.1, p.y * 1.8 + 4.2, p.z * 1.8 + 6.6]) * 0.18;
            let c = continents.get([
                (p.x + wx) * 1.6,
                (p.y + wy) * 1.6,
                (p.z + wz) * 1.6,
            ]);
            let d = detail.get([p.x, p.y, p.z]);
            let e = (c + 0.25 * d).clamp(-1.5, 1.5);
            g.set(x, y, e as f32);
        }
    }
    g
}

/// Pick the sea level so that approximately `land_fraction` of the surface is
/// above water. Robust regardless of the noise distribution.
pub fn sea_level_for_land_fraction(elev: &Grid<f32>, land_fraction: f64) -> f32 {
    let mut v = elev.data.clone();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let frac = (1.0 - land_fraction).clamp(0.0, 1.0);
    let idx = ((frac * v.len() as f64).floor() as usize).min(v.len() - 1);
    v[idx]
}

/// Surface albedo (linear-ish RGB 0..1) used by both the equirect day map and
/// the globe renderer. Combines bathymetry, height-banded biomes, river
/// greening, and polar/alpine snow.
pub fn albedo_at(elev: f32, sea_level: f32, lat: f64, flow: f32) -> [f32; 3] {
    if elev <= sea_level {
        let depth = ((sea_level - elev) / 0.6).clamp(0.0, 1.0);
        let shallow = [0.10, 0.34, 0.48];
        let deep = [0.012, 0.045, 0.13];
        return mix3(shallow, deep, depth);
    }
    let land_h = ((elev - sea_level) / 0.7).clamp(0.0, 1.0);
    let low = [0.17, 0.34, 0.13];
    let mid = [0.31, 0.27, 0.14];
    let high = [0.40, 0.34, 0.28];
    let mut c = if land_h < 0.5 {
        mix3(low, mid, land_h * 2.0)
    } else {
        mix3(mid, high, (land_h - 0.5) * 2.0)
    };
    // River corridors read greener / more fertile.
    if flow > 4.0 {
        c = mix3(c, [0.13, 0.37, 0.11], ((flow - 4.0) / 4.0).clamp(0.0, 1.0));
    }
    // Snow: poleward of ~60deg, or on high alpine terrain.
    let snow_lat = ((lat.abs() - 1.05) / 0.4).clamp(0.0, 1.0) as f32;
    let snow_h = ((land_h - 0.82) / 0.18).clamp(0.0, 1.0);
    let snow = snow_lat.max(snow_h);
    mix3(c, [0.93, 0.94, 0.97], snow)
}

#[inline]
pub fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}
