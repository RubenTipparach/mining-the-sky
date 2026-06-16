//! Continuous procedural elevation.
//!
//! The macro field (continents, coastlines) replicates the worldgen planet
//! field exactly (same Fbm params and formula), so the rocket-view terrain
//! matches the baked map planet. On top of it we add ridged mountains and fine
//! detail for close-up relief. Height is a pure function of sphere direction, so
//! it is seamless across LOD transitions and gains detail as you zoom in.

use glam::DVec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

/// A region flattened toward a target height (launch pad, city sites). `inner`
/// and `outer` are angular radii (radians); fully flat within inner, blending
/// out to natural terrain by outer.
struct FlatZone {
    dir: DVec3,
    inner: f64,
    outer: f64,
    target: f64,
}

pub struct Elevation {
    // macro field (matches worldgen::planet::generate_elevation)
    continents: Fbm<Perlin>,
    detail: Fbm<Perlin>,
    warp: Fbm<Perlin>,
    // close-up relief
    mountains: Fbm<Perlin>,
    hills: Fbm<Perlin>,
    fine: Fbm<Perlin>,
    /// Sea level in the raw noise domain (matches the map's land fraction).
    sea: f64,
    /// Vertical scale: raw noise unit -> metres.
    relief_m: f64,
    flat_zones: Vec<FlatZone>,
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[allow(dead_code)]
fn fib_dir(i: usize, n: usize) -> DVec3 {
    // golden-spiral point on the unit sphere
    let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let t = ga * i as f64;
    DVec3::new(t.cos() * r, y, t.sin() * r)
}

impl Elevation {
    pub fn new(seed: u32) -> Self {
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
        let mountains = Fbm::<Perlin>::new(seed.wrapping_add(777))
            .set_octaves(8)
            .set_frequency(5.0)
            .set_persistence(0.5)
            .set_lacunarity(2.3);
        let hills = Fbm::<Perlin>::new(seed.wrapping_add(909))
            .set_octaves(6)
            .set_frequency(11.0)
            .set_persistence(0.5)
            .set_lacunarity(2.2);
        let fine = Fbm::<Perlin>::new(seed.wrapping_add(1313))
            .set_octaves(7)
            .set_frequency(30.0)
            .set_persistence(0.5)
            .set_lacunarity(2.2);

        Elevation {
            continents,
            detail,
            warp,
            mountains,
            hills,
            fine,
            // Match worldgen's seed-47 sea level exactly (its preview prints
            // 0.070) so the coastline aligns with the baked map and the coastal
            // spaceport reads as land, not water.
            sea: 0.070,
            relief_m: 9000.0,
            flat_zones: Vec::new(),
        }
    }

    /// Flatten a circular region (e.g. the launch pad or a city) to its natural
    /// centre height. `inner_m`/`outer_m` are surface radii in metres.
    pub fn add_flat_zone(&mut self, dir: DVec3, inner_m: f64, outer_m: f64, planet_radius: f64) {
        let d = dir.normalize();
        let target = self.base_height(d);
        self.flat_zones.push(FlatZone {
            dir: d,
            inner: inner_m / planet_radius,
            outer: outer_m / planet_radius,
            target,
        });
    }

    /// The raw macro field value (matches worldgen exactly).
    pub fn raw(&self, dir: DVec3) -> f64 {
        let p = dir.normalize();
        let wx = self.warp.get([p.x * 1.8, p.y * 1.8, p.z * 1.8]) * 0.18;
        let wy = self.warp.get([p.x * 1.8 + 5.3, p.y * 1.8 + 1.7, p.z * 1.8 - 2.1]) * 0.18;
        let wz = self.warp.get([p.x * 1.8 - 3.1, p.y * 1.8 + 4.2, p.z * 1.8 + 6.6]) * 0.18;
        let c = self.continents.get([(p.x + wx) * 1.6, (p.y + wy) * 1.6, (p.z + wz) * 1.6]);
        let d = self.detail.get([p.x, p.y, p.z]);
        (c + 0.25 * d).clamp(-1.5, 1.5)
    }

    #[allow(dead_code)]
    fn sea_level(&self, land_fraction: f64, samples: usize) -> f64 {
        let mut v: Vec<f64> = (0..samples).map(|i| self.raw(fib_dir(i, samples))).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (((1.0 - land_fraction) * samples as f64) as usize).min(samples - 1);
        v[idx]
    }

    /// Natural elevation (metres, signed) before any flatten zones.
    fn base_height(&self, dir: DVec3) -> f64 {
        let d = dir.normalize();
        let above = self.raw(d) - self.sea;
        let land = above.max(0.0);
        // ridged mountains, gated to land and much stronger inland
        let m = 1.0 - self.mountains.get([d.x * 5.0, d.y * 5.0, d.z * 5.0]).abs();
        let ridges = m * m * land * 2.6;
        // rolling hills across the land, growing away from the coast
        let hills = self.hills.get([d.x * 11.0, d.y * 11.0, d.z * 11.0]) * land.sqrt() * 0.12;
        // fine roughness everywhere
        let fine = self.fine.get([d.x, d.y, d.z]) * 0.05;
        (above + ridges + hills + fine) * self.relief_m
    }

    /// Signed elevation in metres relative to sea level (negative = sea floor),
    /// with flatten zones (pad / cities) applied.
    pub fn height_m(&self, dir: DVec3) -> f64 {
        let d = dir.normalize();
        let mut h = self.base_height(d);
        for z in &self.flat_zones {
            let ang = d.dot(z.dir).clamp(-1.0, 1.0).acos();
            let t = 1.0 - smoothstep(z.inner, z.outer, ang); // 1 inside, 0 outside
            h += (z.target - h) * t;
        }
        h
    }

    /// Clamped land height (>= 0) for displacing the visible surface.
    pub fn land_height_m(&self, dir: DVec3) -> f64 {
        self.height_m(dir).max(0.0)
    }
}
