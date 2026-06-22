//! Vehicle Assembly Building: a small parts catalog (engines, fuel tanks,
//! payloads) and a player-editable vehicle configuration that compiles down to
//! a `sim::vehicle::Vehicle`. The rocket-view geometry is built proportionally
//! from the same config, so different builds look different.

use sim::vehicle::{Stage, Vehicle};

#[derive(Clone, Copy)]
pub struct Engine {
    pub name: &'static str,
    pub thrust: f64, // N (sea level-ish)
    pub isp: f64,    // s
    pub mass: f64,   // dry kg
    /// Vacuum-optimised (bell nozzle, lower TWR) - just affects the label/look.
    pub vac: bool,
}

#[derive(Clone, Copy)]
pub struct Tank {
    pub name: &'static str,
    pub prop: f64, // kg propellant
    pub dry: f64,  // kg structure
}

#[derive(Clone, Copy)]
pub struct Payload {
    pub name: &'static str,
    pub mass: f64, // kg
    pub color: [f32; 3],
    /// Cargo-module geometry shown inside the fairing (index into
    /// `rocket::cargo_module`), or -1 for a plain boxed satellite.
    pub module: i32,
}

// Thrust / Isp / mass are modelled on real engines (sea-level thrust for the
// boosters, vacuum for the upper-stage engines). A single slot is treated as one
// engine, so cluster with radial boosters for more liftoff thrust the way real
// vehicles do.
pub const ENGINES: &[Engine] = &[
    // Merlin 1D class - small sea-level kerolox.
    Engine { name: "Merlin-1D", thrust: 0.845e6, isp: 282.0, mass: 490.0, vac: false },
    // Aerojet LR-87 (Titan core), dual-chamber hypergolic - medium sea-level.
    Engine { name: "LR-87", thrust: 2.31e6, isp: 300.0, mass: 1_520.0, vac: false },
    // Rocketdyne F-1 (Saturn V) - the most powerful single-chamber engine flown.
    Engine { name: "F-1", thrust: 6.77e6, isp: 265.0, mass: 8_440.0, vac: false },
    // RL-10 class - small high-Isp vacuum upper stage (hydrolox).
    Engine { name: "RL-10", thrust: 0.11e6, isp: 450.0, mass: 280.0, vac: true },
    // Merlin Vacuum class - medium vacuum upper stage.
    Engine { name: "Merlin-Vac", thrust: 0.981e6, isp: 348.0, mass: 490.0, vac: true },
];

// Propellant tanks, ~6% dry-mass fraction (realistic for kerolox stages).
pub const TANKS: &[Tank] = &[
    Tank { name: "Small", prop: 18_000.0, dry: 1_200.0 },
    Tank { name: "Medium", prop: 70_000.0, dry: 4_200.0 },
    Tank { name: "Large", prop: 200_000.0, dry: 12_000.0 },
    Tank { name: "X-Large", prop: 400_000.0, dry: 24_000.0 },
];

/// A radial strap-on booster: a self-contained motor + propellant clustered
/// around a stage. Solid motors (SRBs) run at fixed thrust until burnout;
/// liquid strap-ons trade a little thrust for higher Isp. They burn together
/// with the core stage they ring (jettisoned with it), augmenting its thrust,
/// propellant and mass.
#[derive(Clone, Copy)]
pub struct Booster {
    pub name: &'static str,
    pub thrust: f64, // N each
    pub prop: f64,   // kg propellant each
    pub dry: f64,    // kg structure each
    pub isp: f64,    // s
    pub solid: bool,
}

pub const BOOSTERS: &[Booster] = &[
    // GEM-63 class small solid.
    Booster { name: "GEM-63", thrust: 1.9e6, prop: 44_000.0, dry: 5_000.0, isp: 245.0, solid: true },
    // Shuttle-SRB class large solid.
    Booster { name: "SRB", thrust: 6.0e6, prop: 290_000.0, dry: 30_000.0, isp: 268.0, solid: true },
    // Liquid strap-on (Atlas-booster class).
    Booster { name: "Liquid Strap-on", thrust: 1.0e6, prop: 40_000.0, dry: 3_500.0, isp: 300.0, solid: false },
];

/// Max radial boosters per stage (kept even-ish around the core).
pub const MAX_BOOSTERS: u32 = 8;

pub const PAYLOADS: &[Payload] = &[
    Payload { name: "CubeSat", mass: 200.0, color: [0.7, 0.8, 0.9], module: -1 },
    Payload { name: "ComSat", mass: 1_400.0, color: [0.9, 0.8, 0.3], module: -1 },
    Payload { name: "Station Module", mass: 5_000.0, color: [0.6, 0.85, 1.0], module: -1 },
    Payload { name: "Fuel Depot", mass: 9_000.0, color: [1.0, 0.6, 0.3], module: -1 },
    Payload { name: "Lunar Lander", mass: 6_500.0, color: [0.82, 0.66, 0.26], module: -1 },
    // Surface base cargo: compact, fairing-packed modules that unfold and are
    // assembled on site. A refinery needs power, so deliver a reactor or a
    // solar generator alongside it.
    Payload { name: "Refinery Module", mass: 8_000.0, color: [0.80, 0.62, 0.22], module: 0 },
    Payload { name: "Fission Reactor", mass: 7_000.0, color: [0.85, 0.86, 0.90], module: 1 },
    Payload { name: "Solar Generator", mass: 4_500.0, color: [0.30, 0.40, 0.70], module: 2 },
    Payload { name: "Habitat Module", mass: 6_000.0, color: [0.84, 0.86, 0.90], module: 3 },
    Payload { name: "ISRU Drill Rig", mass: 7_500.0, color: [0.78, 0.62, 0.22], module: 4 },
    // Crewed flight: a re-entry capsule and the service module that flies behind
    // it. The capsule is recovered under parachute; the pair test the crew/service
    // stack, powered descent and parachute descent.
    Payload { name: "Crew Capsule", mass: 3_200.0, color: [0.88, 0.90, 0.94], module: 5 },
    Payload { name: "Service Module", mass: 4_600.0, color: [0.86, 0.72, 0.34], module: 6 },
];

#[derive(Clone, Copy)]
pub struct StageCfg {
    pub engine: usize,
    pub tank: usize,
    /// Number of radial strap-on boosters ringing this stage (0 = none).
    pub boosters: u32,
    /// Which booster type (index into `BOOSTERS`).
    pub booster: usize,
}

impl StageCfg {
    pub fn new(engine: usize, tank: usize) -> Self {
        StageCfg { engine, tank, boosters: 0, booster: 0 }
    }
}

/// The player's current vehicle design (stages bottom-first, like `Vehicle`).
#[derive(Clone)]
pub struct Vab {
    pub stages: Vec<StageCfg>,
    pub payload: usize,
}

impl Vab {
    /// A sensible default two-stage launcher with realistic parts: an F-1 first
    /// stage on a big tank (lifts off at ~1.35 g, like a real booster) and a
    /// Merlin-Vac upper stage, carrying a 5 t station module.
    pub fn default_build() -> Vab {
        Vab {
            stages: vec![
                StageCfg::new(2, 3), // F-1 + X-Large tank (Saturn-V-class first stage)
                StageCfg::new(4, 1), // Merlin-Vac + Medium upper
            ],
            payload: 2, // Station Module (5 t)
        }
    }

    pub fn engine(&self, i: usize) -> Engine {
        ENGINES[self.stages[i].engine.min(ENGINES.len() - 1)]
    }
    pub fn tank(&self, i: usize) -> Tank {
        TANKS[self.stages[i].tank.min(TANKS.len() - 1)]
    }
    pub fn payload(&self) -> Payload {
        PAYLOADS[self.payload.min(PAYLOADS.len() - 1)]
    }
    /// The booster type fitted to stage `i`, and how many.
    pub fn booster(&self, i: usize) -> (Booster, u32) {
        let s = self.stages[i];
        (BOOSTERS[s.booster.min(BOOSTERS.len() - 1)], s.boosters.min(MAX_BOOSTERS))
    }

    /// Compile to a `sim` vehicle. Each stage's radial boosters burn together
    /// with it, so they add their thrust + propellant + mass to the stage and
    /// blend their Isp in (thrust-weighted).
    pub fn to_vehicle(&self) -> Vehicle {
        let stages = (0..self.stages.len())
            .map(|i| {
                let e = self.engine(i);
                let t = self.tank(i);
                let (b, nb) = self.booster(i);
                let nb = nb as f64;
                let b_thrust = nb * b.thrust;
                let thrust = e.thrust + b_thrust;
                // thrust-weighted Isp so strap-ons shift the effective Isp.
                let isp = if thrust > 0.0 {
                    (e.thrust * e.isp + b_thrust * b.isp) / thrust
                } else {
                    e.isp
                };
                Stage {
                    name: e.name,
                    dry: e.mass + t.dry + nb * b.dry,
                    prop: t.prop + nb * b.prop,
                    thrust,
                    isp,
                }
            })
            .collect();
        Vehicle { name: "Custom Vehicle", stages, payload: self.payload().mass }
    }
}
