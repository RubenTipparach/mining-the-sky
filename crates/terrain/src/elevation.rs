//! Continuous procedural elevation.
//!
//! The macro field (continents, coastlines) replicates the worldgen planet
//! field exactly (same Fbm params and formula), so the rocket-view terrain
//! matches the baked map planet. On top of it we add ridged mountains and fine
//! detail for close-up relief. Height is a pure function of sphere direction, so
//! it is seamless across LOD transitions and gains detail as you zoom in.

use glam::DVec3;
use noise::core::worley::{distance_functions, ReturnType};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, Worley};

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
    /// Lunar mode: replace the earth-like macro field with a cratered regolith
    /// field (no oceans, no mountains; impact basins instead).
    lunar: bool,
    /// Per-scale crater fields: F1 distance to the nearest crater centre and a
    /// per-crater random value (same seed+frequency, so they share cells).
    crater_dist: Vec<Worley>,
    crater_val: Vec<Worley>,
    /// Crater layer parameters: (radius in cell units, depth in metres).
    crater_cfg: Vec<(f64, f64)>,
    /// A constant height offset (metres) applied last (used to align the lunar
    /// landing site with the rocket-view origin).
    offset_m: f64,
    /// Asteroid mode: a small irregular body - low-frequency lobes plus analytic
    /// impact craters, as a positive radial height field of amplitude `ast_amp`.
    asteroid: bool,
    ast_amp: f64,
    /// Impact craters for asteroid mode: (centre dir, angular radius, depth frac).
    ast_craters: Vec<DVec3>,
    ast_crater_cr: Vec<f64>,
    ast_crater_d: Vec<f64>,
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

        // Crater fields, large -> small. Each scale gets a distance Worley (bowl
        // shape) and a value Worley with the SAME seed + frequency so the random
        // value identifies the same crater the distance is measured from.
        let mk_dist = |s: u32, f: f64| {
            Worley::new(s)
                .set_return_type(ReturnType::Distance)
                .set_distance_function(distance_functions::euclidean)
                .set_frequency(f)
        };
        let mk_val = |s: u32, f: f64| {
            Worley::new(s)
                .set_return_type(ReturnType::Value)
                .set_distance_function(distance_functions::euclidean)
                .set_frequency(f)
        };
        let freqs = [900.0_f64, 3000.0, 9000.0];
        let crater_dist = freqs
            .iter()
            .enumerate()
            .map(|(i, &f)| mk_dist(seed.wrapping_add(4000 + i as u32), f))
            .collect();
        let crater_val = freqs
            .iter()
            .enumerate()
            .map(|(i, &f)| mk_val(seed.wrapping_add(4000 + i as u32), f))
            .collect();
        // (radius in cell units, depth in metres) per scale.
        let crater_cfg = vec![(0.55, 1200.0), (0.5, 430.0), (0.5, 110.0)];

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
            lunar: false,
            crater_dist,
            crater_val,
            crater_cfg,
            offset_m: 0.0,
            asteroid: false,
            ast_amp: 0.0,
            ast_craters: Vec::new(),
            ast_crater_cr: Vec::new(),
            ast_crater_d: Vec::new(),
        }
    }

    /// A cratered, airless regolith field (the moon): no oceans or mountains,
    /// just impact basins of several scales over gentle mare undulation.
    pub fn lunar(seed: u32) -> Self {
        let mut e = Elevation::new(seed);
        e.lunar = true;
        e
    }

    /// A small irregular asteroid: low-frequency lobes + `craters` analytic
    /// impact basins, as a positive radial height field of amplitude `amp_m`
    /// (metres). Used with a small `Planet` radius and the LOD quadtree.
    pub fn asteroid(seed: u32, amp_m: f64, craters: usize) -> Self {
        let mut e = Elevation::new(seed);
        e.asteroid = true;
        e.ast_amp = amp_m;
        // scatter craters (power-law size mix) over the unit sphere
        for k in 0..craters {
            let dir = fib_dir(k, craters.max(1));
            // jitter the golden-spiral point a little by a hash of the seed
            let h = ((seed as f64 * 0.013 + k as f64 * 1.7).sin() * 43758.5).fract();
            let h2 = ((seed as f64 * 0.029 + k as f64 * 3.1).sin() * 24634.6).fract();
            let cr = 0.06 + 0.5 * h.abs().powf(2.5);
            let depth = (0.16 + 0.12 * h2.abs()) * (0.5 + cr);
            e.ast_craters.push(dir);
            e.ast_crater_cr.push(cr);
            e.ast_crater_d.push(depth);
        }
        e
    }

    /// Asteroid radial height (metres), in [~0, amp]. Lobes + craters.
    fn asteroid_height(&self, dir: DVec3) -> f64 {
        let d = dir.normalize();
        let lobe = self.continents.get([d.x * 1.5, d.y * 1.5, d.z * 1.5]); // big lobes
        let med = self.hills.get([d.x * 4.0, d.y * 4.0, d.z * 4.0]);
        let fine = self.fine.get([d.x * 13.0, d.y * 13.0, d.z * 13.0]);
        let mut h = 0.5 + 0.30 * lobe + 0.10 * med + 0.04 * fine; // ~0..1
        for i in 0..self.ast_craters.len() {
            let c = self.ast_craters[i];
            let cr = self.ast_crater_cr[i];
            let depth = self.ast_crater_d[i];
            let ang = (1.0 - dir.dot(c).clamp(-1.0, 1.0)).max(0.0);
            let s = ang / cr;
            if s < 1.5 {
                if s < 1.0 {
                    h -= depth * (1.0 - s * s);
                }
                h += 0.45 * depth * (-(((s - 1.0) / 0.18).powi(2))).exp();
            }
        }
        h.clamp(0.04, 1.05) * self.ast_amp
    }

    /// Shift the whole field vertically (metres). Used to seat the lunar landing
    /// site at the rocket-view origin height.
    pub fn set_offset(&mut self, off: f64) {
        self.offset_m = off;
    }

    /// One crater layer's contribution at `dir` (metres, signed).
    fn crater_layer(&self, dir: DVec3, i: usize) -> f64 {
        let p = [dir.x, dir.y, dir.z];
        let rv = self.crater_val[i].get(p); // 0..1 per crater
        if rv < 0.18 {
            return 0.0; // smooth mare patch: no crater in this cell
        }
        let d = self.crater_dist[i].get(p); // F1 distance, cell units
        let (cr, depth) = self.crater_cfg[i];
        let size = cr * (0.45 + 1.1 * rv); // vary crater radius
        let dep = depth * (0.5 + 0.9 * rv); // and depth
        let rn = d / size;
        // depressed bowl: flattish floor, steepening to the rim at rn = 1
        let floor = if rn < 1.0 { -dep * (1.0 - rn * rn * rn) } else { 0.0 };
        // a modest raised rim/lip + ejecta blanket (real craters: rim height is
        // small next to the bowl depth, so the feature reads as a pit)
        let rim = 0.26 * dep * (-(((rn - 1.0) / 0.22).powi(2))).exp();
        floor + rim
    }

    /// Cratered lunar elevation (metres, signed) before flatten zones / offset.
    fn crater_height(&self, dir: DVec3) -> f64 {
        let d = dir.normalize();
        let mut h = 0.0;
        for i in 0..self.crater_cfg.len() {
            h += self.crater_layer(d, i);
        }
        // broad mare undulation + fine regolith roughness (kept low so the
        // craters stay the dominant relief)
        h += self.hills.get([d.x * 3.0, d.y * 3.0, d.z * 3.0]) * 90.0;
        h += self.fine.get([d.x * 40.0, d.y * 40.0, d.z * 40.0]) * 14.0;
        h
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
        if self.asteroid {
            return self.asteroid_height(d);
        }
        if self.lunar {
            return self.crater_height(d);
        }
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
        h + self.offset_m
    }

    /// Clamped land height (>= 0) for displacing the visible surface.
    pub fn land_height_m(&self, dir: DVec3) -> f64 {
        self.height_m(dir).max(0.0)
    }
}
