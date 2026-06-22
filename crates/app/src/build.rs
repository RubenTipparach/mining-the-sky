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

pub const ENGINES: &[Engine] = &[
    Engine { name: "Sparrow", thrust: 0.95e6, isp: 315.0, mass: 1200.0, vac: false },
    Engine { name: "Merlin", thrust: 3.8e6, isp: 300.0, mass: 6000.0, vac: false },
    Engine { name: "Titan-9", thrust: 9.2e6, isp: 295.0, mass: 16000.0, vac: false },
    Engine { name: "Vac-1", thrust: 1.1e6, isp: 345.0, mass: 2500.0, vac: true },
    Engine { name: "Vac-3", thrust: 2.6e6, isp: 350.0, mass: 4200.0, vac: true },
];

pub const TANKS: &[Tank] = &[
    Tank { name: "Small", prop: 18_000.0, dry: 1_400.0 },
    Tank { name: "Medium", prop: 70_000.0, dry: 4_800.0 },
    Tank { name: "Large", prop: 200_000.0, dry: 13_000.0 },
    Tank { name: "X-Large", prop: 380_000.0, dry: 24_000.0 },
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
    Booster { name: "SRB-Lite", thrust: 3.6e6, prop: 70_000.0, dry: 6_000.0, isp: 250.0, solid: true },
    Booster { name: "SRB-Heavy", thrust: 8.5e6, prop: 200_000.0, dry: 16_000.0, isp: 262.0, solid: true },
    Booster { name: "LqStrap", thrust: 4.2e6, prop: 120_000.0, dry: 7_000.0, isp: 295.0, solid: false },
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
    /// A sensible default two-stage launcher (~the old Pioneer).
    pub fn default_build() -> Vab {
        Vab {
            stages: vec![
                StageCfg::new(2, 2), // Titan-9 + Large booster
                StageCfg::new(3, 1), // Vac-1 + Medium upper
            ],
            payload: 1, // ComSat
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
