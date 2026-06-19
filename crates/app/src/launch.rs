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

// Attitude (pitch) response: the airframe steers toward the commanded pitch
// like a thruster-controlled rigid body instead of snapping. `STEER_GAIN` turns
// the angle error into a target rate (so it eases in near the command);
// `MAX_PITCH_RATE` caps how fast it can rotate; `MAX_PITCH_ACC` caps how fast
// that rate can change, which is what gives the rotation its inertia/momentum.
const STEER_GAIN: f64 = 2.5; // 1/s
const MAX_PITCH_RATE: f64 = 0.35; // rad/s (~20 deg/s)
const MAX_PITCH_ACC: f64 = 0.5; // rad/s^2

// Aerodynamic heating. The convective heat flux on a blunt body goes roughly as
// q ~ sqrt(rho) * v^3 (Sutton-Graves form). We normalise that against
// `HEAT_REF` for the FX glow, and the airframe only takes damage once the flux
// climbs past `HEAT_DAMAGE_Q` (well above anything a normal ascent reaches, so
// only a fast reentry burns). Tuned for the home world's atmosphere.
const HEAT_REF: f64 = 2.0e9; // flux at which the plasma shock reads full white-hot
const HEAT_DAMAGE_Q: f64 = 4.0e9; // flux above which the structure starts to char
const HEAT_DAMAGE_K: f64 = 9.0; // health lost per second per unit of over-flux

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
    /// Structural integrity (0..100) and current heating glow (0..1).
    pub health: f32,
    pub heat: f32,
}

pub struct Rocket {
    pub r: DVec3,
    pub v: DVec3,
    /// Remaining stages, bottom (active) first.
    pub stages: Vec<LiveStage>,
    pub payload: f64,
    pub throttle: f64, // 0..1
    /// Commanded steering angle from local-up toward downrange (rad). 0 = up.
    /// This is the target the player / autopilot dials; the rocket's actual
    /// attitude (`pitch_act`) slews toward it at a limited rate so steering
    /// reads as a rigid body pitching over, not an instant snap.
    pub pitch: f64,
    /// Actual attitude angle the airframe currently holds (rad). Thrust and the
    /// drawn rocket both follow this, so it lags the command realistically.
    pub pitch_act: f64,
    /// Current pitch angular rate (rad/s); integrated under a limited angular
    /// acceleration so the airframe has rotational inertia.
    pub pitch_rate: f64,
    /// Launch-site radial at ignition (defines the launch plane with `plane_n`).
    pub up0: DVec3,
    /// Orbital-plane normal (up0 x launch-heading); the turn stays in this plane.
    pub plane_n: DVec3,
    pub met: f64,
    pub crashed: bool,
    pub orbit_reached: bool,
    /// Structural integrity, 0..100. Aerodynamic heating past the airframe's
    /// tolerance burns it down; at 0 the vehicle is destroyed. A hard ground
    /// impact destroys it outright.
    pub health: f64,
    /// Smoothed aerodynamic-heating level, 0..~1.2, driving the reentry plasma
    /// FX glow (1.0 = white-hot shock). Lags the instantaneous flux a little.
    pub heat: f64,
    /// Set once when the vehicle is destroyed (burn-through or crash), so the
    /// render side can spawn the explosion + debris exactly once.
    pub destroyed: bool,
    /// Set for one frame when a stage was just jettisoned (drives the visual
    /// separation); carries the jettisoned stage index from the original stack.
    pub just_staged: Option<usize>,
    /// Index of the active stage within the original stack (0 = first booster).
    pub stage_base: usize,
    pub stage_total: usize,
    /// Flown ascent path (home-centred world metres), sampled as the rocket
    /// climbs. Drawn on the orbital map so the trajectory is visible during the
    /// sub-orbital ascent, before a predicted conic exists.
    pub trail: Vec<DVec3>,
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
            pitch_act: 0.0,
            pitch_rate: 0.0,
            up0,
            plane_n,
            met: 0.0,
            crashed: false,
            orbit_reached: false,
            health: 100.0,
            heat: 0.0,
            destroyed: false,
            just_staged: None,
            stage_base: 0,
            stage_total: veh.stages.len(),
            trail: vec![r],
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

    /// Unit thrust / pointing direction for the airframe's *actual* attitude
    /// (which lags the command, so the rocket rotates instead of snapping).
    pub fn point_dir(&self) -> DVec3 {
        let up = self.r.normalize_or_zero();
        let horiz = self.horizontal(up);
        (up * self.pitch_act.cos() + horiz * self.pitch_act.sin()).normalize_or_zero()
    }

    /// Slew the actual attitude toward the commanded pitch over `dt` seconds with
    /// a limited angular rate and acceleration, so the airframe carries
    /// rotational inertia rather than teleporting to the new angle. A
    /// proportional law sets the target rate (easing off near the command); the
    /// rate itself can only change so fast (the inertia / control authority).
    fn slew_attitude(&mut self, dt: f64) {
        let err = self.pitch - self.pitch_act;
        let target_rate = (err * STEER_GAIN).clamp(-MAX_PITCH_RATE, MAX_PITCH_RATE);
        let dv = (target_rate - self.pitch_rate).clamp(-MAX_PITCH_ACC * dt, MAX_PITCH_ACC * dt);
        self.pitch_rate += dv;
        self.pitch_act += self.pitch_rate * dt;
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
        // Atmospheric drag. The launch is flown in a frame co-moving with the
        // launch site, so the atmosphere is treated as non-rotating (no surface-
        // wind term) - consistent with igniting at rest over the fixed pad.
        let v_rel = v;
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
            // rotate the airframe toward the commanded pitch before reading the
            // thrust direction, so thrust follows the actual (lagging) attitude.
            self.slew_attitude(h);
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

            // ground contact at the launch site (co-moving frame: surface at rest)
            if self.r.length() <= body.radius {
                let up = self.r.normalize_or_zero();
                let rel = self.v.length();
                self.r = up * body.radius;
                if self.met > 1.0 && rel > CRASH_SPEED {
                    self.crashed = true;
                    self.destroyed = true;
                    self.health = 0.0;
                    self.v = DVec3::ZERO;
                    return;
                }
                self.v = DVec3::ZERO; // pinned to the pad before liftoff
            }

            // aerodynamic heating: glow tracks the flux; burn-through past the
            // material limit eats into structural health, and at zero the
            // vehicle breaks up (the render side spawns the explosion + debris).
            let q = self.heat_flux(body);
            let target = (q / HEAT_REF).min(1.3);
            // ease the glow toward the instantaneous flux (hot fast, cool slow)
            let k = if target > self.heat { 6.0 } else { 1.5 };
            self.heat += (target - self.heat) * (k * h).min(1.0);
            if q > HEAT_DAMAGE_Q {
                self.health -= (q - HEAT_DAMAGE_Q) / HEAT_DAMAGE_Q * HEAT_DAMAGE_K * h;
            }
            if self.health <= 0.0 && !self.crashed {
                self.health = 0.0;
                self.crashed = true;
                self.destroyed = true;
                return;
            }

            // orbit achieved once the periapsis clears the atmosphere
            let orb = orbit_from_state(self.r, self.v, body.mu);
            if orb.e < 1.0 && orb.rp - body.radius > body.atmo_top {
                self.orbit_reached = true;
            }
        }

        // Record the flown path for the map, sampling only when the rocket has
        // moved a visible fraction of a body radius so the trail stays cheap.
        let step = body.radius * 0.0015; // ~9 km on the home world
        if self.trail.last().map(|&p| (p - self.r).length() > step).unwrap_or(true) {
            self.trail.push(self.r);
            if self.trail.len() > 2048 {
                self.trail.remove(0);
            }
        }
    }

    pub fn altitude(&self, body: &CentralBody) -> f64 {
        self.r.length() - body.radius
    }

    /// Instantaneous convective heat flux proxy (q ~ sqrt(rho) * v^3) at the
    /// current state. Zero above the atmosphere.
    pub fn heat_flux(&self, body: &CentralBody) -> f64 {
        let rho = body.density(self.altitude(body));
        if rho <= 0.0 {
            return 0.0;
        }
        let v = self.v.length();
        rho.sqrt() * v * v * v
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
        let phase = if self.destroyed {
            "DESTROYED"
        } else if self.crashed {
            "CRASHED"
        } else if self.heat > 0.55 {
            "REENTRY"
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
            health: self.health as f32,
            heat: self.heat as f32,
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

    /// The flown ascent path as home-centred unit-sphere points (magnitude
    /// encodes altitude), ending at the current position. Used to draw the
    /// rocket's trajectory on the orbital map during the sub-orbital climb.
    pub fn trail_points(&self, body: &CentralBody) -> Vec<Vec3> {
        self.trail
            .iter()
            .chain(std::iter::once(&self.r))
            .map(|&p| (p / body.radius).as_vec3())
            .collect()
    }

    /// Forward conic arc from the current state as unit-sphere points: continues
    /// the bound trajectory ahead of the rocket, stopping where a sub-orbital arc
    /// would re-enter the surface. Gives a visible predicted path on the map even
    /// before a parking orbit is reached. Empty for hyperbolic/degenerate states.
    pub fn forward_arc(&self, body: &CentralBody) -> Vec<Vec3> {
        let orb = orbit_from_state(self.r, self.v, body.mu);
        if orb.e >= 1.0 || orb.a <= 0.0 {
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
        // current true anomaly (angle in the orbit plane from periapsis)
        let nu0 = self.r.dot(q_hat).atan2(self.r.dot(p_hat));
        let mut out = Vec::new();
        let steps = 220;
        for i in 0..=steps {
            let nu = nu0 + (i as f64 / steps as f64) * std::f64::consts::TAU;
            let rad = p / (1.0 + orb.e * nu.cos());
            // sub-orbital arcs dip below the surface: stop the prediction there.
            if rad <= body.radius && i > 0 {
                break;
            }
            let pos = (rad * (nu.cos() * p_hat + nu.sin() * q_hat)) / body.radius;
            out.push(pos.as_vec3());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::orbit::orbit_from_state;

    /// Build a Pioneer on the equator at lon 0, heading east, at rest (the
    /// co-moving launch frame).
    fn pad_rocket(body: &CentralBody) -> Rocket {
        let veh = Vehicle::pioneer();
        let up = DVec3::new(1.0, 0.0, 0.0);
        let r = up * body.radius;
        let heading = DVec3::Y.cross(up).normalize(); // east
        Rocket::on_pad(&veh, r, DVec3::ZERO, up, heading)
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

    /// A sudden, large pitch command must not snap the airframe: the actual
    /// attitude should lag (rate-limited) right after the command, then converge.
    #[test]
    fn attitude_slews_instead_of_snapping() {
        let body = CentralBody::home();
        let mut rk = pad_rocket(&body);
        rk.throttle = 1.0;
        rk.pitch = 80f64.to_radians(); // hard-over command

        // one short tick: the airframe has barely begun to rotate, nowhere near
        // the command (proves it does not teleport to the commanded angle).
        rk.integrate(&body, 0.1);
        assert!(
            rk.pitch_act < 10f64.to_radians(),
            "attitude snapped to {:.1} deg in 0.1 s",
            rk.pitch_act.to_degrees()
        );
        // and the rotation rate is capped (rigid-body inertia, not a jump).
        assert!(rk.pitch_rate <= MAX_PITCH_RATE + 1e-6);

        // hold the command: it converges to the target within a few seconds.
        for _ in 0..200 {
            rk.integrate(&body, 0.1);
        }
        assert!(
            (rk.pitch_act - rk.pitch).abs() < 2f64.to_radians(),
            "attitude failed to converge: act={:.1} cmd={:.1}",
            rk.pitch_act.to_degrees(),
            rk.pitch.to_degrees()
        );
    }
}
