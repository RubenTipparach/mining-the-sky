//! Continuous procedural elevation. Because height is a pure function of the
//! sphere direction (not a baked grid), it is sampled at any LOD with no seams
//! and arbitrary detail as you zoom in - this is what makes the terrain truly
//! procedural and seamless across LOD transitions.

use glam::DVec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

pub struct Elevation {
    continents: Fbm<Perlin>,
    mountains: Fbm<Perlin>,
    detail: Fbm<Perlin>,
    /// Sea level in the raw noise domain.
    sea: f64,
    /// Vertical scale: raw noise unit -> metres.
    relief_m: f64,
}

impl Elevation {
    pub fn new(seed: u32) -> Self {
        let continents = Fbm::<Perlin>::new(seed)
            .set_octaves(7)
            .set_frequency(1.0)
            .set_persistence(0.5)
            .set_lacunarity(2.1);
        let mountains = Fbm::<Perlin>::new(seed.wrapping_add(53))
            .set_octaves(8)
            .set_frequency(4.0)
            .set_persistence(0.5)
            .set_lacunarity(2.3);
        let detail = Fbm::<Perlin>::new(seed.wrapping_add(131))
            .set_octaves(9)
            .set_frequency(22.0)
            .set_persistence(0.5)
            .set_lacunarity(2.2);
        Elevation { continents, mountains, detail, sea: 0.07, relief_m: 8000.0 }
    }

    /// Signed elevation in metres relative to sea level (negative = sea floor).
    pub fn height_m(&self, dir: DVec3) -> f64 {
        let d = dir.normalize();
        let c = self.continents.get([d.x * 1.6, d.y * 1.6, d.z * 1.6]);
        let land = (c - self.sea).max(0.0); // 0 over ocean, grows inland
        // Ridged mountains, gated to land and stronger on higher ground.
        let m = 1.0 - self.mountains.get([d.x * 4.0, d.y * 4.0, d.z * 4.0]).abs();
        let ridges = m * m * land * 1.4;
        // Fine detail everywhere, scaled down so it reads as roughness.
        let fine = self.detail.get([d.x, d.y, d.z]) * 0.12;
        ((c - self.sea) + ridges + fine) * self.relief_m
    }

    /// Clamped land height (>= 0) for displacing the visible surface.
    pub fn land_height_m(&self, dir: DVec3) -> f64 {
        self.height_m(dir).max(0.0)
    }
}
