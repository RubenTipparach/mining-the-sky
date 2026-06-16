//! Staged launch vehicle. Bottom stage is index 0 and burns first.

pub const G0: f64 = 9.80665; // standard gravity for Isp (s)

#[derive(Clone, Copy)]
pub struct Stage {
    pub name: &'static str,
    pub dry: f64,
    pub prop: f64,
    /// Vacuum-ish thrust (N).
    pub thrust: f64,
    /// Specific impulse (s).
    pub isp: f64,
}

impl Stage {
    pub fn wet(&self) -> f64 {
        self.dry + self.prop
    }
    /// Propellant mass flow at full thrust (kg/s).
    pub fn mdot(&self) -> f64 {
        self.thrust / (self.isp * G0)
    }
    /// Ideal delta-v of this stage carrying `upper` mass above it.
    pub fn dv(&self, upper: f64) -> f64 {
        let m0 = self.wet() + upper;
        let mf = self.dry + upper;
        self.isp * G0 * (m0 / mf).ln()
    }
}

pub struct Vehicle {
    pub name: &'static str,
    pub stages: Vec<Stage>,
    pub payload: f64,
}

impl Vehicle {
    /// A two-stage medium launcher, sized with margin to reach low orbit.
    pub fn pioneer() -> Self {
        Vehicle {
            name: "Pioneer I",
            stages: vec![
                Stage { name: "Booster", dry: 20_000.0, prop: 300_000.0, thrust: 6.2e6, isp: 295.0 },
                Stage { name: "Upper", dry: 6_000.0, prop: 62_000.0, thrust: 0.95e6, isp: 345.0 },
            ],
            payload: 5_000.0,
        }
    }

    pub fn liftoff_mass(&self) -> f64 {
        self.payload + self.stages.iter().map(|s| s.wet()).sum::<f64>()
    }

    /// Mass above (not including) stage `idx`: payload + every higher stage.
    pub fn mass_above(&self, idx: usize) -> f64 {
        self.payload
            + self.stages[idx + 1..].iter().map(|s| s.wet()).sum::<f64>()
    }
}
