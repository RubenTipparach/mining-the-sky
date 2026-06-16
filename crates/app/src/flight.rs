//! Manual free-flight: take direct control of the craft and fly it under
//! powered thrust + gravity + atmospheric drag, change your orbit, deorbit,
//! and land back on the surface (anywhere on the world). This is the live,
//! integrated counterpart to the scripted on-rails ascent: once you have a
//! ship under thrust we integrate it (the design's "integrate only when
//! thrusting, otherwise stay on rails" rule).

use glam::{DVec3, Vec3};
use sim::body::CentralBody;
use sim::orbit::orbit_from_state;
use sim::vehicle::G0;

const CDA: f64 = 6.0; // Cd * frontal area (m^2), matches the ascent model
const LAND_SPEED: f64 = 18.0; // m/s surface-relative: under this you land, over it you crash

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Prograde,
    Retrograde,
    RadialOut,
    RadialIn,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Prograde => "PROGRADE",
            Mode::Retrograde => "RETRO",
            Mode::RadialOut => "RADIAL OUT",
            Mode::RadialIn => "RADIAL IN",
        }
    }
}

pub struct Craft {
    pub r: DVec3,
    pub v: DVec3,
    pub dry: f64,
    pub prop: f64,
    pub prop_full: f64,
    pub thrust: f64,
    pub isp: f64,
    pub throttle: f64, // 0..1
    pub mode: Mode,
    pub landed: bool,
    pub crashed: bool,
    /// Which body the craft last touched ("HOME" / "MOON" / "").
    pub landed_on: &'static str,
}

/// An extra point-mass gravity source (e.g. the moon), positioned in the same
/// metres-from-home-centre frame the craft uses. Treated as non-rotating.
#[derive(Clone, Copy)]
pub struct GravBody {
    pub center: DVec3,
    pub mu: f64,
    pub radius: f64,
    pub name: &'static str,
}

impl Craft {
    /// A modest maneuvering ship: enough delta-v to deorbit and fly a powered
    /// landing once atmospheric drag has bled off most of the orbital speed.
    pub fn maneuvering(r: DVec3, v: DVec3) -> Craft {
        Craft {
            r,
            v,
            dry: 5_000.0,
            prop: 9_000.0,
            prop_full: 9_000.0,
            thrust: 2.4e5,
            isp: 320.0,
            throttle: 0.0,
            mode: Mode::Prograde,
            landed: false,
            crashed: false,
            landed_on: "",
        }
    }

    pub fn mass(&self) -> f64 {
        self.dry + self.prop
    }

    pub fn prop_frac(&self) -> f32 {
        (self.prop / self.prop_full).clamp(0.0, 1.0) as f32
    }

    fn thrust_dir(&self) -> DVec3 {
        match self.mode {
            Mode::Prograde => self.v.normalize_or_zero(),
            Mode::Retrograde => -self.v.normalize_or_zero(),
            Mode::RadialOut => self.r.normalize_or_zero(),
            Mode::RadialIn => -self.r.normalize_or_zero(),
        }
    }

    fn accel(
        &self,
        body: &CentralBody,
        bodies: &[GravBody],
        r: DVec3,
        v: DVec3,
        tdir: DVec3,
        thrust_n: f64,
        mass: f64,
    ) -> DVec3 {
        let rmag = r.length();
        let mut g = -body.mu / (rmag * rmag * rmag) * r;
        // extra point-mass sources (e.g. the moon)
        for b in bodies {
            let d = r - b.center;
            let dl = d.length().max(1.0);
            g += -b.mu / (dl * dl * dl) * d;
        }
        // atmospheric drag from the home world only
        let v_atm = body.omega() * DVec3::Y.cross(r);
        let v_rel = v - v_atm;
        let vr = v_rel.length();
        let rho = body.density(rmag - body.radius);
        let drag = if vr > 1e-3 {
            -0.5 * rho * vr * CDA / mass * v_rel
        } else {
            DVec3::ZERO
        };
        let thr = if thrust_n > 0.0 { tdir * (thrust_n / mass) } else { DVec3::ZERO };
        g + drag + thr
    }

    /// Advance the craft by `dt_sim` seconds of mission time using fixed
    /// substeps, under home gravity + any extra bodies. No-op once
    /// landed-and-idle or crashed.
    pub fn integrate(&mut self, body: &CentralBody, bodies: &[GravBody], dt_sim: f64) {
        if self.crashed || dt_sim <= 0.0 {
            return;
        }
        let h = 0.1f64;
        let steps = ((dt_sim / h).ceil() as i64).clamp(1, 400);
        for _ in 0..steps {
            let mass = self.mass();
            let thrust_n = if self.throttle > 0.0 && self.prop > 0.0 {
                self.thrust * self.throttle
            } else {
                0.0
            };
            let tdir = self.thrust_dir();

            let r = self.r;
            let v = self.v;
            let a1 = self.accel(body, bodies, r, v, tdir, thrust_n, mass);
            let a2 = self.accel(body, bodies, r + v * (h * 0.5), v + a1 * (h * 0.5), tdir, thrust_n, mass);
            let a3 = self.accel(body, bodies, r + (v + a1 * (h * 0.5)) * (h * 0.5), v + a2 * (h * 0.5), tdir, thrust_n, mass);
            let a4 = self.accel(body, bodies, r + (v + a2 * (h * 0.5)) * h, v + a3 * h, tdir, thrust_n, mass);
            self.r = r + (v + (a1 + a2 + a3) * (h / 6.0)) * h;
            self.v = v + (a1 + 2.0 * a2 + 2.0 * a3 + a4) * (h / 6.0);

            if thrust_n > 0.0 {
                self.prop = (self.prop - thrust_n / (self.isp * G0) * h).max(0.0);
            }

            if self.resolve_contact(body, bodies) {
                return; // crashed
            }
        }
    }

    /// Resolve surface contact against the home world and any extra body.
    /// Returns true if this ended in a crash.
    fn resolve_contact(&mut self, body: &CentralBody, bodies: &[GravBody]) -> bool {
        // home world (rotating)
        if self.r.length() - body.radius <= 0.0 {
            let up = self.r.normalize_or_zero();
            let v_surf = body.omega() * DVec3::Y.cross(self.r);
            let rel = (self.v - v_surf).length();
            self.r = up * body.radius;
            self.v = v_surf;
            self.landed_on = "HOME";
            if rel > LAND_SPEED {
                self.crashed = true;
                return true;
            }
            self.landed = true;
            return false;
        }
        // extra bodies (non-rotating)
        for b in bodies {
            let d = self.r - b.center;
            if d.length() - b.radius <= 0.0 {
                let up = d.normalize_or_zero();
                let rel = self.v.length();
                self.r = b.center + up * b.radius;
                self.v = DVec3::ZERO;
                self.landed_on = b.name;
                if rel > LAND_SPEED {
                    self.crashed = true;
                    return true;
                }
                self.landed = true;
                return false;
            }
        }
        self.landed = false;
        false
    }

    pub fn altitude(&self, body: &CentralBody) -> f64 {
        self.r.length() - body.radius
    }

    pub fn speed(&self) -> f64 {
        self.v.length()
    }

    /// Vertical (radial) speed, positive = climbing.
    pub fn vertical_speed(&self) -> f64 {
        self.v.dot(self.r.normalize_or_zero())
    }

    pub fn status(&self) -> &'static str {
        if self.crashed {
            "CRASHED"
        } else if self.landed {
            "LANDED"
        } else {
            "FLYING"
        }
    }

    /// Unit-sphere position for the marker.
    pub fn marker(&self, body: &CentralBody) -> Vec3 {
        let u = self.r / body.radius;
        Vec3::new(u.x as f32, u.y as f32, u.z as f32)
    }

    /// Predicted conic from the current state, as unit-sphere points (empty for
    /// hyperbolic / escape trajectories).
    pub fn predicted_orbit(&self, body: &CentralBody) -> Vec<Vec3> {
        let orb = orbit_from_state(self.r, self.v, body.mu);
        if orb.e >= 1.0 || orb.a <= 0.0 {
            return Vec::new();
        }
        let w_hat = orb.h_vec.normalize_or_zero();
        let p_hat = if orb.e > 1e-4 {
            orb.e_vec.normalize_or_zero()
        } else {
            // circular: any vector in the plane
            let a = if w_hat.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
            w_hat.cross(a).normalize_or_zero()
        };
        let q_hat = w_hat.cross(p_hat).normalize_or_zero();
        let p = orb.a * (1.0 - orb.e * orb.e);
        let radius = body.radius;
        (0..=180)
            .map(|i| {
                let nu = i as f32 / 180.0 * std::f32::consts::TAU;
                let nu64 = nu as f64;
                let rad = p / (1.0 + orb.e * nu64.cos());
                let pos = (rad * (nu64.cos() * p_hat + nu64.sin() * q_hat)) / radius;
                Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circular(body: &CentralBody, alt: f64) -> Craft {
        let r0 = body.radius + alt;
        let vc = (body.mu / r0).sqrt();
        Craft::maneuvering(DVec3::new(r0, 0.0, 0.0), DVec3::new(0.0, 0.0, vc))
    }

    #[test]
    fn coasting_circular_orbit_stays_circular() {
        let body = CentralBody::home();
        let mut c = circular(&body, 300_000.0); // above the atmosphere, no drag
        for _ in 0..60 {
            c.integrate(&body, &[], 10.0); // 600 s total
        }
        let alt = c.altitude(&body);
        assert!(
            (alt - 300_000.0).abs() < 10_000.0,
            "circular orbit drifted to {} m",
            alt
        );
    }

    #[test]
    fn retro_burn_lowers_periapsis() {
        let body = CentralBody::home();
        let mut c = circular(&body, 300_000.0);
        let before = orbit_from_state(c.r, c.v, body.mu).rp;
        c.mode = Mode::Retrograde;
        c.throttle = 1.0;
        for _ in 0..20 {
            c.integrate(&body, &[], 1.0); // 20 s retro burn
        }
        let after = orbit_from_state(c.r, c.v, body.mu).rp;
        assert!(after < before - 1_000.0, "periapsis {} -> {}", before, after);
        assert!(c.prop < c.prop_full, "burn consumed no propellant");
    }

    #[test]
    fn slow_touchdown_lands_not_crashes() {
        let body = CentralBody::home();
        let up = DVec3::new(1.0, 0.0, 0.0);
        let r = up * (body.radius + 1.0);
        let v_surf = body.omega() * DVec3::Y.cross(r);
        let mut c = Craft::maneuvering(r, v_surf - up * 2.0); // 2 m/s down
        c.throttle = 0.0;
        for _ in 0..40 {
            c.integrate(&body, &[], 0.5);
            if c.landed || c.crashed {
                break;
            }
        }
        assert!(c.landed && !c.crashed, "expected LANDED, got {}", c.status());
        assert_eq!(c.landed_on, "HOME");
    }

    #[test]
    fn slow_descent_lands_on_the_moon() {
        let body = CentralBody::home();
        // a moon far from the home world so home gravity is negligible here
        let moon = GravBody {
            center: DVec3::new(88.0e6, 0.0, 8.0e6),
            mu: 4.9e12, // ~1.7 m/s^2 surface gravity at r=1.674e6
            radius: 1.674e6,
            name: "MOON",
        };
        // start just above the moon's surface, drifting gently toward it
        let up = DVec3::new(0.0, 1.0, 0.0);
        let r = moon.center + up * (moon.radius + 2.0);
        let mut c = Craft::maneuvering(r, -up * 1.5); // 1.5 m/s toward the surface
        c.throttle = 0.0;
        for _ in 0..60 {
            c.integrate(&body, &[moon], 0.5);
            if c.landed || c.crashed {
                break;
            }
        }
        assert!(c.landed && !c.crashed, "expected LANDED, got {}", c.status());
        assert_eq!(c.landed_on, "MOON");
        let surf = (c.r - moon.center).length();
        assert!(
            (surf - moon.radius).abs() < 5.0,
            "craft not on moon surface: {} vs {}",
            surf,
            moon.radius
        );
    }
}
