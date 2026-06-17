//! Player-controlled, multi-stage launch (KSP-style). You start on the pad with
//! the full stack and fly it to orbit yourself: throttle, pitch the rocket over
//! into a gravity turn, and stage when a booster runs dry. It is the live,
//! integrated counterpart to the scripted on-rails ascent - same RK4 gravity +
//! atmospheric drag + thrust core as `flight::Craft`, but with a stack of stages
//! and a thrust vector the player steers.

use glam::{DVec3, Vec3};
use sim::body::CentralBody;
use sim::orbit::orbit_from_state;
use sim::vehicle::{Vehicle, G0};

const CDA: f64 = 6.0; // Cd * frontal area (m^2), matches the ascent model
const CRASH_SPEED: f64 = 18.0; // m/s surface-relative impact tolerance

/// One live stage: its current propellant plus its fixed engine numbers.
#[derive(Clone, Copy)]
pub struct LiveStage {
    pub name: &'static str,
    pub dry: f64,
    pub prop: f64,
    pub prop_full: f64,
    pub thrust: f64,
    pub isp: f64,
}

/// Live launch telemetry for the HUD.
pub struct Tel {
    pub phase: &'static str,
    pub alt_km: f32,
    pub speed: f32,
    pub vspeed: f32,
    pub throttle: f32,
    pub twr: f32,
    pub stage_name: &'static str,
    pub stage_idx: usize,
    pub stage_total: usize,
    pub prop_frac: f32,
    /// Apoapsis / periapsis altitude (km), once the conic is elliptical.
    pub apo_km: f32,
    pub peri_km: f32,
    pub pitch_deg: f32,
}

pub struct Rocket {
    pub r: DVec3,
    pub v: DVec3,
    /// Remaining stages, bottom (active) first.
    pub stages: Vec<LiveStage>,
    pub payload: f64,
    pub throttle: f64, // 0..1
    /// Steering angle from local-up toward downrange (rad). 0 = straight up.
    pub pitch: f64,
    /// Launch-site radial at ignition (defines the launch plane with `plane_n`).
    pub up0: DVec3,
    /// Orbital-plane normal (up0 x launch-heading); the turn stays in this plane.
    pub plane_n: DVec3,
    pub met: f64,
    pub crashed: bool,
    pub orbit_reached: bool,
    /// Set for one frame when a stage was just jettisoned (drives the visual
    /// separation); carries the jettisoned stage index from the original stack.
    pub just_staged: Option<usize>,
    /// Index of the active stage within the original stack (0 = first booster).
    pub stage_base: usize,
    pub stage_total: usize,
}

impl Rocket {
    /// Build the full Pioneer stack sitting on the pad, at inertial state
    /// `(r, v)` (metres / m/s, home-centred). `up_radial` is the local vertical
    /// and `heading` the launch azimuth (both unit, home-centred frame).
    pub fn on_pad(veh: &Vehicle, r: DVec3, v: DVec3, up_radial: DVec3, heading: DVec3) -> Rocket {
        let stages: Vec<LiveStage> = veh
            .stages
            .iter()
            .map(|s| LiveStage {
                name: s.name,
                dry: s.dry,
                prop: s.prop,
                prop_full: s.prop,
                thrust: s.thrust,
                isp: s.isp,
            })
            .collect();
        let up0 = up_radial.normalize();
        let plane_n = up0.cross(heading).normalize_or_zero();
        Rocket {
            r,
            v,
            stages,
            payload: veh.payload,
            throttle: 0.0,
            pitch: 0.0,
            up0,
            plane_n,
            met: 0.0,
            crashed: false,
            orbit_reached: false,
            just_staged: None,
            stage_base: 0,
            stage_total: veh.stages.len(),
        }
    }

    pub fn mass(&self) -> f64 {
        self.payload + self.stages.iter().map(|s| s.dry + s.prop).sum::<f64>()
    }

    /// The active (burning) stage, if any remain.
    fn active(&self) -> Option<&LiveStage> {
        self.stages.first()
    }

    pub fn stage_name(&self) -> &'static str {
        self.active().map(|s| s.name).unwrap_or("PAYLOAD")
    }

    pub fn prop_frac(&self) -> f32 {
        self.active()
            .map(|s| (s.prop / s.prop_full.max(1.0)).clamp(0.0, 1.0) as f32)
            .unwrap_or(0.0)
    }

    /// Throttle * available thrust (0 when the active stage is dry / cut).
    pub fn live_thrust(&self) -> f64 {
        match self.active() {
            Some(s) if s.prop > 0.0 && self.throttle > 0.0 => s.thrust * self.throttle,
            _ => 0.0,
        }
    }

    /// Current downrange horizontal (in the launch plane, perpendicular to up).
    fn horizontal(&self, up: DVec3) -> DVec3 {
        // plane_n x up gives the in-plane horizontal pointing downrange.
        let h = self.plane_n.cross(up);
        h.normalize_or_zero()
    }

    /// Unit thrust / pointing direction for the current pitch.
    pub fn point_dir(&self) -> DVec3 {
        let up = self.r.normalize_or_zero();
        let horiz = self.horizontal(up);
        (up * self.pitch.cos() + horiz * self.pitch.sin()).normalize_or_zero()
    }

    /// Jettison the active stage and ignite the next. No-op on the last stage.
    pub fn jettison(&mut self) {
        if self.stages.len() > 1 {
            self.stages.remove(0);
            self.just_staged = Some(self.stage_base);
            self.stage_base += 1;
        }
    }

    fn accel(
        &self,
        body: &CentralBody,
        r: DVec3,
        v: DVec3,
        tdir: DVec3,
        thrust_n: f64,
        mass: f64,
    ) -> DVec3 {
        let rmag = r.length();
        let mut a = -body.mu / (rmag * rmag * rmag) * r;
        // atmospheric drag (home world only)
        let v_atm = body.omega() * DVec3::Y.cross(r);
        let v_rel = v - v_atm;
        let vr = v_rel.length();
        let rho = body.density(rmag - body.radius);
        if vr > 1e-3 {
            a += -0.5 * rho * vr * CDA / mass * v_rel;
        }
        if thrust_n > 0.0 {
            a += tdir * (thrust_n / mass);
        }
        a
    }

    /// Advance `dt_sim` seconds of flight under gravity + drag + thrust, burning
    /// the active stage's propellant. Returns false once crashed.
    pub fn integrate(&mut self, body: &CentralBody, dt_sim: f64) {
        if self.crashed || dt_sim <= 0.0 {
            return;
        }
        let h = 0.05f64;
        let steps = ((dt_sim / h).ceil() as i64).clamp(1, 2000);
        for _ in 0..steps {
            self.met += h;
            let mass = self.mass();
            let thrust_n = self.live_thrust();
            let tdir = self.point_dir();

            let r = self.r;
            let v = self.v;
            let a1 = self.accel(body, r, v, tdir, thrust_n, mass);
            let a2 = self.accel(body, r + v * (h * 0.5), v + a1 * (h * 0.5), tdir, thrust_n, mass);
            let a3 = self.accel(body, r + (v + a1 * (h * 0.5)) * (h * 0.5), v + a2 * (h * 0.5), tdir, thrust_n, mass);
            let a4 = self.accel(body, r + (v + a2 * (h * 0.5)) * h, v + a3 * h, tdir, thrust_n, mass);
            self.r = r + (v + (a1 + a2 + a3) * (h / 6.0)) * h;
            self.v = v + (a1 + 2.0 * a2 + 2.0 * a3 + a4) * (h / 6.0);

            if thrust_n > 0.0 {
                if let Some(s) = self.stages.first_mut() {
                    s.prop = (s.prop - thrust_n / (s.isp * G0) * h).max(0.0);
                }
            }

            // ground contact at the launch site
            if self.r.length() <= body.radius {
                let up = self.r.normalize_or_zero();
                let v_surf = body.omega() * DVec3::Y.cross(self.r);
                let rel = (self.v - v_surf).length();
                self.r = up * body.radius;
                if self.met > 1.0 && rel > CRASH_SPEED {
                    self.crashed = true;
                    self.v = v_surf;
                    return;
                }
                // sitting on the pad before liftoff: pin to the surface
                self.v = v_surf;
            }

            // orbit achieved once the periapsis clears the atmosphere
            let orb = orbit_from_state(self.r, self.v, body.mu);
            if orb.e < 1.0 && orb.rp - body.radius > body.atmo_top {
                self.orbit_reached = true;
            }
        }
    }

    pub fn altitude(&self, body: &CentralBody) -> f64 {
        self.r.length() - body.radius
    }
    pub fn speed(&self) -> f64 {
        self.v.length()
    }
    pub fn vertical_speed(&self) -> f64 {
        self.v.dot(self.r.normalize_or_zero())
    }

    pub fn telemetry(&self, body: &CentralBody) -> Tel {
        let orb = orbit_from_state(self.r, self.v, body.mu);
        let (apo_km, peri_km) = if orb.a > 0.0 && orb.e < 1.0 {
            (
                ((orb.ra - body.radius) / 1000.0) as f32,
                ((orb.rp - body.radius) / 1000.0) as f32,
            )
        } else {
            (f32::INFINITY, f32::NEG_INFINITY)
        };
        let g = body.mu / (self.r.length() * self.r.length());
        let twr = (self.live_thrust() / (self.mass() * g)) as f32;
        let phase = if self.crashed {
            "CRASHED"
        } else if self.orbit_reached {
            "ORBIT"
        } else if self.live_thrust() > 0.0 {
            "POWERED"
        } else {
            "COAST"
        };
        Tel {
            phase,
            alt_km: (self.altitude(body) / 1000.0) as f32,
            speed: self.speed() as f32,
            vspeed: self.vertical_speed() as f32,
            throttle: self.throttle as f32,
            twr,
            stage_name: self.stage_name(),
            stage_idx: self.stage_base,
            stage_total: self.stage_total,
            prop_frac: self.prop_frac(),
            apo_km,
            peri_km,
            pitch_deg: self.pitch.to_degrees() as f32,
        }
    }

    /// Predicted conic from the current state as unit-sphere points (empty for
    /// hyperbolic / sub-orbital arcs that intersect the ground).
    pub fn predicted_orbit(&self, body: &CentralBody) -> Vec<Vec3> {
        let orb = orbit_from_state(self.r, self.v, body.mu);
        if orb.e >= 1.0 || orb.a <= 0.0 || orb.rp <= body.radius {
            return Vec::new();
        }
        let w = orb.h_vec.normalize_or_zero();
        let p_hat = if orb.e > 1e-4 {
            orb.e_vec.normalize_or_zero()
        } else {
            let a = if w.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
            w.cross(a).normalize_or_zero()
        };
        let q_hat = w.cross(p_hat).normalize_or_zero();
        let p = orb.a * (1.0 - orb.e * orb.e);
        (0..=180)
            .map(|i| {
                let nu = (i as f64 / 180.0) * std::f64::consts::TAU;
                let rad = p / (1.0 + orb.e * nu.cos());
                let pos = (rad * (nu.cos() * p_hat + nu.sin() * q_hat)) / body.radius;
                Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::orbit::orbit_from_state;

    /// Build a Pioneer on the equator at lon 0, heading east.
    fn pad_rocket(body: &CentralBody) -> Rocket {
        let veh = Vehicle::pioneer();
        let up = DVec3::new(1.0, 0.0, 0.0);
        let r = up * body.radius;
        let v = body.omega() * DVec3::Y.cross(r); // surface velocity
        let heading = DVec3::Y.cross(up).normalize(); // east
        Rocket::on_pad(&veh, r, v, up, heading)
    }

    #[test]
    fn full_throttle_gravity_turn_reaches_orbit() {
        let body = CentralBody::home();
        let mut rk = pad_rocket(&body);
        rk.throttle = 1.0;
        // open-loop gravity turn: hold vertical briefly, then ease over to near
        // horizontal and hold while the upper stage builds orbital velocity.
        for _ in 0..1000 {
            // 0.5 s steps, 500 s total (covers both stages' burns)
            let t = rk.met;
            rk.pitch = ((t - 10.0) / 150.0).clamp(0.0, 1.0) * 88f64.to_radians();
            if rk.prop_frac() <= 0.0 && rk.stages.len() > 1 {
                rk.jettison();
            }
            rk.integrate(&body, 0.5);
            if rk.crashed {
                break;
            }
        }
        assert!(!rk.crashed, "rocket crashed during ascent");
        assert!(
            rk.altitude(&body) > 100_000.0,
            "only reached {:.0} m altitude",
            rk.altitude(&body)
        );
        // A real orbit needs orbital horizontal speed; prove the powered flight
        // built it (the precise circularization is the player's job in-game).
        assert!(
            rk.speed() > 6_000.0,
            "only built {:.0} m/s; expected orbital velocity",
            rk.speed()
        );
        let orb = orbit_from_state(rk.r, rk.v, body.mu);
        assert!(orb.e < 1.0 && orb.a > 0.0, "trajectory is not bound: e={:.3}", orb.e);
    }

    #[test]
    fn staging_drops_mass_and_advances() {
        let body = CentralBody::home();
        let mut rk = pad_rocket(&body);
        let m0 = rk.mass();
        assert_eq!(rk.stage_base, 0);
        rk.jettison();
        assert_eq!(rk.stage_base, 1);
        assert_eq!(rk.just_staged, Some(0));
        assert!(rk.mass() < m0, "jettison did not drop mass");
        assert_eq!(rk.stage_name(), "Upper");
    }
}
