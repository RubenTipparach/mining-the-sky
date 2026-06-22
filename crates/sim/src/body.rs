//! Central body (the fictional Earth-like home world) and its atmosphere.

use std::f64::consts::PI;

#[derive(Clone, Copy)]
pub struct CentralBody {
    /// Standard gravitational parameter GM (m^3/s^2).
    pub mu: f64,
    /// Equatorial radius (m).
    pub radius: f64,
    /// Sidereal day length (s); drives surface rotation velocity.
    pub day_seconds: f64,
    /// Atmosphere scale height (m).
    pub scale_height: f64,
    /// Sea-level air density (kg/m^3).
    pub sea_density: f64,
    /// Altitude above which drag is negligible (m).
    pub atmo_top: f64,
}

impl CentralBody {
    /// Fictionalized home world: ~Earth radius and Earth-standard gravity.
    pub fn home() -> Self {
        let radius = 6.2e6;
        let surface_g = 9.81; // Earth-standard surface gravity
        CentralBody {
            mu: surface_g * radius * radius,
            radius,
            day_seconds: 24.0 * 3600.0,
            scale_height: 8500.0,
            sea_density: 1.225, // Earth sea-level air density (kg/m^3)
            atmo_top: 100_000.0,
        }
    }

    pub fn surface_gravity(&self) -> f64 {
        self.mu / (self.radius * self.radius)
    }

    /// Sidereal rotation rate (rad/s) about +Y.
    pub fn omega(&self) -> f64 {
        2.0 * PI / self.day_seconds
    }

    pub fn density(&self, altitude: f64) -> f64 {
        if altitude >= self.atmo_top || altitude < 0.0 {
            0.0
        } else {
            self.sea_density * (-altitude / self.scale_height).exp()
        }
    }
}
