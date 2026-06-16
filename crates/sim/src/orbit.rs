//! Two-body orbital state. Elements are derived analytically from the state
//! vector (this is the cheap, deterministic "on-rails" backbone described in
//! the design doc -- no numerical integration needed once in orbit).

use glam::DVec3;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub a: f64,
    pub e: f64,
    /// Apoapsis / periapsis radius (m from center).
    pub ra: f64,
    pub rp: f64,
    /// Orbital period (s), None for non-elliptical orbits.
    pub period: Option<f64>,
    /// Angular momentum and eccentricity vectors (orbit plane + orientation).
    pub h_vec: DVec3,
    pub e_vec: DVec3,
}

pub fn orbit_from_state(r: DVec3, v: DVec3, mu: f64) -> Orbit {
    let rmag = r.length();
    let vmag = v.length();
    let energy = vmag * vmag * 0.5 - mu / rmag;
    let a = -mu / (2.0 * energy);
    let h_vec = r.cross(v);
    let e_vec = v.cross(h_vec) / mu - r / rmag;
    let e = e_vec.length();
    let ra = a * (1.0 + e);
    let rp = a * (1.0 - e);
    let period = if a > 0.0 {
        Some(2.0 * PI * (a.powi(3) / mu).sqrt())
    } else {
        None
    };
    Orbit { a, e, ra, rp, period, h_vec, e_vec }
}

/// Speed for a circular orbit at radius `r`.
pub fn circular_speed(mu: f64, r: f64) -> f64 {
    (mu / r).sqrt()
}

/// vis-viva speed at radius `r` on an orbit of semi-major axis `a`.
pub fn vis_viva(mu: f64, r: f64, a: f64) -> f64 {
    (mu * (2.0 / r - 1.0 / a)).max(0.0).sqrt()
}
