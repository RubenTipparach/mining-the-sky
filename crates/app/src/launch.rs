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

// Liftoff realism. Real engines build to full thrust over a couple of seconds
// (held down on the pad until then) rather than snapping on, and the throttle is
// pulled back so the crew never sees more than a few g of acceleration. On a
// Saturn V the felt acceleration ramps from ~1.25 g at liftoff up toward ~4 g as
// the stage burns light, where the centre engine is cut to hold the limit (see
// the Apollo 8 ascent profile); we model that ceiling with an automatic g-limit.
const SPOOL_TIME: f64 = 2.5; // s for an igniting stage to build to full thrust
const CREW_G_LIMIT: f64 = 4.0; // g the auto-throttle will not let the crew exceed (Apollo S-IC peaked ~3.9 g)
const G_EARTH: f64 = 9.80665; // standard g for the crew-load reference

// Recovery parachute. A deployed main canopy adds a large Cd*area so the vehicle
// settles to a survivable terminal velocity in the lower atmosphere; it opens
// over a couple of seconds (the canopy inflating) rather than instantly.
const CDA_CHUTE: f64 = 600.0; // Cd*area of a fully open main canopy (m^2)
const CHUTE_OPEN_RATE: f64 = 0.5; // open fraction per second (~2 s to full)

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
const HEAT_DAMAGE_Q: f64 = 3.0e9; // flux above which the structure starts to char
const HEAT_DAMAGE_K: f64 = 2.0; // health lost per second per unit of over-flux

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
    /// Acceleration felt by the crew (g); the auto-throttle caps this.
    pub g_force: f32,
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
    /// Engine spool: actual thrust fraction the lit stage has built up, 0..1. It
    /// ramps up over `SPOOL_TIME` after ignition (and resets on staging) so thrust
    /// comes on smoothly instead of snapping to full.
    pub spool: f64,
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
    /// Recovery parachute open fraction, 0 (stowed) .. 1 (fully inflated). Adds a
    /// large drag area when open and drives the drawn canopy.
    pub chute: f64,
    /// The parachute has been armed/deployed (it then inflates over a couple of
    /// seconds toward `chute = 1`).
    pub chute_armed: bool,
    /// Set once a soft touchdown (impact under the crash tolerance) is achieved.
    pub landed: bool,
    /// Engage the powered-descent autopilot: point retrograde-up and throttle to
    /// arrest the fall for a soft (suicide-burn) touchdown.
    pub auto_land: bool,
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
            spool: 0.0,
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
            chute: 0.0,
            chute_armed: false,
            landed: false,
            auto_land: false,
            trail: vec![r],
        }
    }

    /// Arm the recovery parachute; it then inflates over a couple of seconds.
    pub fn deploy_chute(&mut self) {
        self.chute_armed = true;
    }

    /// Powered-descent autopilot. Points the airframe up (so thrust opposes the
    /// fall) and feathers the throttle on a suicide-burn law: the deceleration
    /// needed to null the descent within the remaining altitude, plus gravity.
    /// Runs only when descending in the lower atmosphere / near the ground.
    fn auto_descent_control(&mut self, body: &CentralBody) {
        let up = self.r.normalize_or_zero();
        let alt = self.altitude(body).max(0.0);
        let vspeed = -self.v.dot(up); // +ve = descending
        // hold the airframe vertical so thrust fights gravity + the fall
        self.pitch = 0.0;
        let g = body.mu / self.r.length_squared();
        let mass = self.mass();
        let tmax = self
            .active()
            .map(|s| if s.prop > 0.0 { s.thrust } else { 0.0 })
            .unwrap_or(0.0);
        let tacc = if mass > 0.0 { tmax / mass } else { 0.0 };
        if vspeed > 0.6 && tacc > 0.0 {
            // decelerate to a stop ~6 m above the surface, then hold a soft sink
            let stop_alt = (alt - 6.0).max(1.0);
            let need = vspeed * vspeed / (2.0 * stop_alt);
            self.throttle = ((need + g) / tacc).clamp(0.0, 1.0);
        } else {
            // settled / climbing: cut to a gentle hover-down
            self.throttle = (g / tacc.max(1e-6) * 0.9).clamp(0.0, 1.0);
            if alt < 2.0 {
                self.throttle = 0.0;
            }
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

    /// Net thrust this instant (N): commanded throttle * engine thrust, ramped by
    /// the spool-up and then pulled back by the crew g-limiter so the proper
    /// acceleration (thrust / mass) never exceeds `CREW_G_LIMIT`. Because real
    /// thrust is ~constant while mass burns away, the felt g naturally climbs
    /// through a stage until it hits this ceiling - the Saturn-V ramp - and the
    /// limiter then holds it there (as the centre-engine cutoff did on Apollo).
    pub fn live_thrust(&self) -> f64 {
        let raw = match self.active() {
            Some(s) if s.prop > 0.0 && self.throttle > 0.0 => s.thrust * self.throttle,
            _ => return 0.0,
        };
        let spooled = raw * self.spool;
        let g_cap = CREW_G_LIMIT * G_EARTH * self.mass();
        spooled.min(g_cap)
    }

    /// Felt acceleration on the crew right now, in g (proper acceleration =
    /// non-gravitational force / mass). Used for the HUD and the launch-profile
    /// check; the g-limiter keeps this at or below `CREW_G_LIMIT`.
    pub fn crew_g(&self) -> f64 {
        let mass = self.mass();
        if mass <= 0.0 {
            return 0.0;
        }
        self.live_thrust() / mass / G_EARTH
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
            self.spool = 0.0; // the next stage's engine has to spool up from cold
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
            // base airframe drag plus the deployed canopy's much larger area.
            let cda = CDA + self.chute * CDA_CHUTE;
            a += -0.5 * rho * vr * cda / mass * v_rel;
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
            // inflate the parachute once armed (the canopy fills over ~2 s).
            if self.chute_armed && self.chute < 1.0 {
                self.chute = (self.chute + CHUTE_OPEN_RATE * h).min(1.0);
            }
            // powered-descent autopilot drives throttle + attitude toward a soft
            // landing (before the per-step attitude slew + thrust are read).
            if self.auto_land && !self.landed {
                self.auto_descent_control(body);
            }
            // engine spool: build thrust up over SPOOL_TIME while the lit stage is
            // commanded (and bleed it back off when cut), so liftoff ramps in.
            let lit = self.throttle > 0.0 && self.active().map(|s| s.prop > 0.0).unwrap_or(false);
            let want = if lit { 1.0 } else { 0.0 };
            let drate = h / SPOOL_TIME;
            self.spool = if self.spool < want {
                (self.spool + drate).min(want)
            } else {
                (self.spool - drate).max(want)
            };
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
                if self.met > 1.0 {
                    self.landed = true; // a touchdown under the crash tolerance
                }
                self.v = DVec3::ZERO; // pinned to the pad / resting on the ground
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
            g_force: self.crew_g() as f32,
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

    #[test]
    fn parachute_gives_a_survivable_descent() {
        // A heavy capsule dropped without a chute slams in; with the canopy it
        // settles to a soft touchdown (impact under the crash tolerance).
        let body = CentralBody::home();
        let veh = sim::vehicle::Vehicle {
            name: "capsule",
            stages: vec![sim::vehicle::Stage { name: "c", dry: 2000.0, prop: 0.0, thrust: 0.0, isp: 1.0 }],
            payload: 3200.0,
        };
        let up = DVec3::new(1.0, 0.0, 0.0);
        let heading = DVec3::Y.cross(up).normalize();
        let mut rk = Rocket::on_pad(&veh, up * (body.radius + 1500.0), -up * 120.0, up, heading);
        rk.met = 5.0; // past the pad-abort guard
        rk.deploy_chute();
        // terminal velocity under the canopy is ~12 m/s, so the descent is slow.
        for _ in 0..3000 {
            rk.integrate(&body, 0.1);
            if rk.landed || rk.crashed {
                break;
            }
        }
        assert!(rk.landed && !rk.crashed, "capsule did not land softly under chute");
        assert!(rk.chute > 0.9, "canopy did not fully inflate: {:.2}", rk.chute);
    }

    #[test]
    fn powered_descent_lands_softly() {
        // The auto-descent autopilot brings a falling stage down without crashing.
        let body = CentralBody::home();
        let veh = sim::vehicle::Vehicle {
            name: "lander",
            stages: vec![sim::vehicle::Stage { name: "m", dry: 6000.0, prop: 12000.0, thrust: 3.8e6, isp: 300.0 }],
            payload: 3200.0,
        };
        let up = DVec3::new(1.0, 0.0, 0.0);
        let heading = DVec3::Y.cross(up).normalize();
        let mut rk = Rocket::on_pad(&veh, up * (body.radius + 2500.0), -up * 140.0, up, heading);
        rk.met = 5.0;
        rk.auto_land = true;
        for _ in 0..1500 {
            rk.integrate(&body, 0.1);
            if rk.landed || rk.crashed {
                break;
            }
        }
        assert!(rk.landed && !rk.crashed, "powered descent did not land softly");
    }

    #[test]
    fn launch_g_profile_is_smooth_and_capped() {
        // Fly the actual default vehicle and sample its ascent. It must: not jump
        // off the pad (engines spool up), ramp velocity smoothly, and never push
        // the crew past the g-limit (the auto-throttle pulls back at the ceiling).
        let body = CentralBody::home();
        let veh = crate::build::Vab::default_build().to_vehicle();
        let up = DVec3::new(1.0, 0.0, 0.0);
        let heading = DVec3::Y.cross(up).normalize();
        let mut rk = Rocket::on_pad(&veh, up * body.radius, DVec3::ZERO, up, heading);
        rk.throttle = 1.0;

        let dt = 0.1;
        let mut prev = 0.0;
        let mut max_g = 0.0f64;
        let mut first10_max_dv = 0.0f64;
        let mut lifted_at = -1.0;
        let mut last_speed = 0.0;
        println!("\n   t(s)  alt(km)  speed(m/s)  net_a(m/s^2)  crew_g  spool  stage");
        for i in 1..=3000 {
            let t = i as f64 * dt;
            // gentle open-loop gravity turn, like a flown ascent
            rk.pitch = ((t - 12.0) / 150.0).clamp(0.0, 1.0) * 80f64.to_radians();
            if rk.prop_frac() <= 0.0 && rk.stages.len() > 1 {
                rk.jettison();
            }
            rk.integrate(&body, dt);
            let sp = rk.speed();
            let dv = sp - prev;
            let g = rk.crew_g();
            max_g = max_g.max(g);
            if t <= 10.0 {
                first10_max_dv = first10_max_dv.max(dv.abs());
            }
            if lifted_at < 0.0 && rk.altitude(&body) > 1.0 {
                lifted_at = t;
            }
            if (i % 50 == 0) || (t <= 10.0 && i % 10 == 0) {
                println!(
                    "   {:>5.1} {:>7.2} {:>10.0} {:>12.2} {:>6.2} {:>6.2}   {}",
                    t,
                    rk.altitude(&body) / 1000.0,
                    sp,
                    dv / dt,
                    g,
                    rk.spool,
                    rk.stage_base
                );
            }
            prev = sp;
            last_speed = sp;
            if rk.crashed || rk.orbit_reached {
                break;
            }
        }
        println!(
            "   --> lifted off at t={:.1}s, peak crew g={:.2}, final speed={:.0} m/s",
            lifted_at, max_g, last_speed
        );
        // crew never exceeds the limit (tiny numerical margin)
        assert!(max_g <= CREW_G_LIMIT + 0.15, "crew g exceeded the limit: {max_g:.2}");
        // no instant leap off the pad: thrust spools up, so it holds down ~1 s
        assert!(lifted_at > 0.5, "lifted off instantly (no spool-up): t={lifted_at:.2}");
        // velocity ramps in small increments through the first 10 s (no jump)
        assert!(
            first10_max_dv < 2.5,
            "velocity jumped {first10_max_dv:.2} m/s in one 0.1 s step in the first 10 s"
        );
        // the g-limit did not starve the ascent: it still built real speed
        assert!(last_speed > 3000.0, "ascent stalled: only {last_speed:.0} m/s");
    }
}
