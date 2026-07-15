//! Procedural world generation for Mining the Sky.
//!
//! Pipeline: elevation -> sea level -> hydrology (rivers/deltas) -> city &
//! launch-site placement -> roads -> night lights. Everything is seeded and
//! deterministic so every client generates an identical home world.

pub mod grid;
pub mod hydrology;
pub mod lights;
pub mod planet;
pub mod render;
pub mod rng;
pub mod roads;
pub mod sites;

use grid::Grid;
use hydrology::Hydrology;
use roads::Road;
use sites::{City, CityKind, LaunchSite, Sites};

/// Export the world's cities as the unified [`worldcity::CityDesc`] index: the
/// single source of truth shared by the ground renderer (which generates each
/// city's layout from this), the far-LOD footprints, and the orbital lights.
/// Baked to `cities.bin` so the client loads it without re-running worldgen.
pub fn city_descs(world: &World) -> Vec<worldcity::CityDesc> {
    world
        .cities
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let id = i as u32;
            let pop = c.population as f32;
            let pm = (pop / 1.0e6).max(0.1);
            // matches worldcity::generate's grid sizing, so radius_m bounds the
            // built-up footprint (the LOD / load query radius).
            let n = (4.0 + 2.4 * pm.sqrt()).clamp(4.0, 11.0);
            let radius_m = n * 30.0 * 1.45;
            worldcity::CityDesc {
                id,
                kind: match c.kind {
                    CityKind::Major => 1,
                    CityKind::Minor => 0,
                },
                lon: c.lon as f32,
                lat: c.lat as f32,
                pop,
                seed: id.wrapping_add(1).wrapping_mul(2_654_435_761),
                radius_m,
                _pad: 0,
            }
        })
        .collect()
}

pub struct GenConfig {
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    /// Fictionalized home-world radius (km). Roughly Earth-like for the
    /// vertical slice; the to-scale Kepler-47 bodies live in the sim crate.
    pub radius_km: f64,
    pub land_fraction: f64,
    pub n_major: usize,
    pub n_minor: usize,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            seed: 47,
            width: 2048,
            height: 1024,
            radius_km: 6200.0,
            land_fraction: 0.31,
            n_major: 24,
            n_minor: 150,
        }
    }
}

pub struct World {
    pub cfg: GenConfig,
    pub elevation: Grid<f32>,
    pub sea_level: f32,
    pub hydro: Hydrology,
    pub cities: Vec<City>,
    pub launch: LaunchSite,
    pub roads: Vec<Road>,
    pub lights: Grid<f32>,
}

pub fn generate(cfg: GenConfig) -> World {
    let elevation = planet::generate_elevation(cfg.width, cfg.height, cfg.seed as u32);
    let sea_level = planet::sea_level_for_land_fraction(&elevation, cfg.land_fraction);
    let hydro = hydrology::compute(&elevation, sea_level);
    let Sites { cities, launch } =
        sites::place_sites(&elevation, &hydro, sea_level, cfg.n_major, cfg.n_minor, cfg.seed);
    let roads = roads::build_roads(&cities);
    let lights = lights::build_lights(cfg.width, cfg.height, &cities, &roads);
    World {
        cfg,
        elevation,
        sea_level,
        hydro,
        cities,
        launch,
        roads,
        lights,
    }
}
