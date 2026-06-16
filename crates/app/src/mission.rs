//! Bridges the `sim` launch-to-orbit into render-space geometry.
//!
//! `sim` works in metres about the planet centre with +Y as the spin axis,
//! which is exactly the convention the planet shader samples the baked texture
//! with, so a launch trajectory drops straight onto the globe: we just scale by
//! the planet radius to get unit-sphere coordinates and keep a handful of
//! parameters to drive the rocket marker along the ascent and parking orbit.

use glam::{DVec3, Vec3};
use sim::ascent::simulate;
use sim::body::CentralBody;
use sim::orbit::circular_speed;
use sim::vehicle::Vehicle;

/// Seed-47 spaceport coordinates (from worldgen), matching `sim`'s launch bin.
const SPACEPORT_LAT_DEG: f64 = -1.7;
const SPACEPORT_LON_DEG: f64 = -102.9;
const TARGET_APO_ALT: f64 = 200_000.0;

pub struct Mission {
    /// Ascent samples as (mission time seconds, unit-sphere position).
    pub path: Vec<(f32, Vec3)>,
    /// Parking-orbit ring, unit-sphere positions (empty if orbit not reached).
    pub ring: Vec<Vec3>,
    pub reached: bool,

    pub spaceport_lat: f32,
    pub spaceport_lon: f32,
    /// A small ring on the surface marking the launch pad.
    pub pad_ring: Vec<Vec3>,

    /// Mission time of main-engine cutoff (the ascent/orbit boundary).
    pub meco_t: f32,
    pub vehicle: &'static str,
    pub stage_count: usize,

    // scalar ascent telemetry, one per sample: (t, alt_km, speed, downrange_km, stage)
    tel: Vec<(f32, f32, f32, f32, usize)>,
    // parking orbit readout
    park_alt_km: f32,
    park_speed: f32,
    peri_km: f32,
    apo_km: f32,

    // parameters for coasting the marker along the ring after MECO
    ring_radius: f32,
    ring_t1: Vec3,
    ring_t2: Vec3,
    theta_meco: f32,
    rate: f32, // rad/s

    // f64 state for seeding manual free-flight
    orbit_t1: DVec3,
    orbit_t2: DVec3,
    ra_m: f64,
    v_circ_ms: f64,
    up0_d: DVec3,
    radius_m: f64,
    omega: f64,
}

/// Live readout for the HUD.
pub struct Telemetry {
    pub phase: &'static str,
    pub alt_km: f32,
    pub speed: f32,
    pub downrange_km: f32,
    pub stage: usize,
    /// peri x apo (km) once in the parking orbit.
    pub orbit: Option<(f32, f32)>,
}

fn unit(r: DVec3, radius: f64) -> Vec3 {
    let u = r / radius;
    Vec3::new(u.x as f32, u.y as f32, u.z as f32)
}

impl Mission {
    pub fn pioneer_from_spaceport() -> Mission {
        let body = CentralBody::home();
        let veh = Vehicle::pioneer();
        let res = simulate(
            &body,
            &veh,
            SPACEPORT_LAT_DEG,
            SPACEPORT_LON_DEG,
            TARGET_APO_ALT,
        );
        let radius = body.radius;

        let path: Vec<(f32, Vec3)> = res
            .samples
            .iter()
            .map(|s| (s.t as f32, unit(s.r, radius)))
            .collect();

        let tel: Vec<(f32, f32, f32, f32, usize)> = res
            .samples
            .iter()
            .map(|s| {
                (
                    s.t as f32,
                    (s.alt / 1000.0) as f32,
                    s.speed as f32,
                    (s.downrange / 1000.0) as f32,
                    s.stage,
                )
            })
            .collect();

        let reached = res.reached_orbit;
        let peri_km = ((res.final_orbit.rp - radius) / 1000.0) as f32;
        let apo_km = ((res.final_orbit.ra - radius) / 1000.0) as f32;
        let park_alt_km = ((res.final_orbit.ra - radius) / 1000.0) as f32;
        let park_speed = circular_speed(body.mu, res.final_orbit.ra) as f32;

        // Build the parking-orbit ring in the orbital plane (perpendicular to h).
        let h = res.final_orbit.h_vec.normalize_or_zero();
        let aref = if h.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let t1 = h.cross(aref).normalize_or_zero();
        let t2 = h.cross(t1).normalize_or_zero();
        let ring_radius = (res.final_orbit.ra / radius) as f32;
        let t1f = Vec3::new(t1.x as f32, t1.y as f32, t1.z as f32);
        let t2f = Vec3::new(t2.x as f32, t2.y as f32, t2.z as f32);

        let ring: Vec<Vec3> = if reached {
            (0..=160)
                .map(|i| {
                    let th = i as f32 / 160.0 * std::f32::consts::TAU;
                    ring_radius * (t1f * th.cos() + t2f * th.sin())
                })
                .collect()
        } else {
            Vec::new()
        };

        // Launch-pad marker: a small ring on the surface around the spaceport.
        let lat = SPACEPORT_LAT_DEG.to_radians();
        let lon = SPACEPORT_LON_DEG.to_radians();
        let up0 = Vec3::new(
            (lat.cos() * lon.cos()) as f32,
            lat.sin() as f32,
            (lat.cos() * lon.sin()) as f32,
        );
        let east = Vec3::Y.cross(up0).normalize();
        let north = up0.cross(east).normalize();
        let pad_ring: Vec<Vec3> = (0..=48)
            .map(|i| {
                let a = i as f32 / 48.0 * std::f32::consts::TAU;
                (up0 + 0.02 * (east * a.cos() + north * a.sin())).normalize()
            })
            .collect();

        let v_circ_ms = circular_speed(body.mu, res.final_orbit.ra);
        let up0_d = DVec3::new(
            lat.cos() * lon.cos(),
            lat.sin(),
            lat.cos() * lon.sin(),
        );

        let (theta_meco, rate, meco_t) = if let Some(m) = res.meco {
            let theta = (m.r.dot(t2)).atan2(m.r.dot(t1)) as f32;
            let rate = (v_circ_ms / res.final_orbit.ra) as f32;
            (theta, rate, m.t as f32)
        } else {
            (0.0, 0.0, f32::INFINITY)
        };

        Mission {
            path,
            ring,
            reached,
            spaceport_lat: SPACEPORT_LAT_DEG.to_radians() as f32,
            spaceport_lon: SPACEPORT_LON_DEG.to_radians() as f32,
            pad_ring,
            meco_t,
            vehicle: veh.name,
            stage_count: veh.stages.len(),
            tel,
            park_alt_km,
            park_speed,
            peri_km,
            apo_km,
            ring_radius,
            ring_t1: t1f,
            ring_t2: t2f,
            theta_meco,
            rate,
            orbit_t1: t1,
            orbit_t2: t2,
            ra_m: res.final_orbit.ra,
            v_circ_ms,
            up0_d,
            radius_m: radius,
            omega: body.omega(),
        }
    }

    /// Inertial state (metres, m/s) of the parking orbit at mission time
    /// `clock`, for handing the craft over to manual free-flight.
    pub fn orbit_state_at(&self, clock: f32) -> (DVec3, DVec3) {
        let theta = (self.theta_meco + self.rate * (clock - self.meco_t)) as f64;
        let r = self.ra_m * (self.orbit_t1 * theta.cos() + self.orbit_t2 * theta.sin());
        let v = self.v_circ_ms * (-self.orbit_t1 * theta.sin() + self.orbit_t2 * theta.cos());
        (r, v)
    }

    /// Inertial state of the craft sitting on the launch pad.
    pub fn pad_state(&self) -> (DVec3, DVec3) {
        let r = self.up0_d * self.radius_m;
        let v = self.omega * DVec3::Y.cross(r);
        (r, v)
    }

    /// Live telemetry at mission time `clock` (only meaningful once launched).
    pub fn telemetry(&self, launched: bool, clock: f32) -> Telemetry {
        if !launched {
            let stage = self.tel.first().map(|s| s.4).unwrap_or(0);
            return Telemetry {
                phase: "READY",
                alt_km: 0.0,
                speed: 0.0,
                downrange_km: 0.0,
                stage,
                orbit: None,
            };
        }
        if !self.reached || clock <= self.meco_t {
            let (alt_km, speed, downrange_km, stage) = sample_tel(&self.tel, clock);
            Telemetry {
                phase: "ASCENT",
                alt_km,
                speed,
                downrange_km,
                stage,
                orbit: None,
            }
        } else {
            Telemetry {
                phase: "ORBIT",
                alt_km: self.park_alt_km,
                speed: self.park_speed,
                downrange_km: 0.0,
                stage: self.stage_count.saturating_sub(1),
                orbit: Some((self.peri_km, self.apo_km)),
            }
        }
    }

    fn ring_point(&self, theta: f32) -> Vec3 {
        self.ring_radius * (self.ring_t1 * theta.cos() + self.ring_t2 * theta.sin())
    }

    /// Rocket position at mission time `clock`: along the powered ascent up to
    /// MECO, then coasting around the parking orbit.
    pub fn rocket_pos(&self, clock: f32) -> Vec3 {
        if !self.reached || clock <= self.meco_t {
            sample_path(&self.path, clock)
        } else {
            let theta = self.theta_meco + self.rate * (clock - self.meco_t);
            self.ring_point(theta)
        }
    }
}

fn sample_tel(tel: &[(f32, f32, f32, f32, usize)], t: f32) -> (f32, f32, f32, usize) {
    if tel.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }
    let lerp = |a: f32, b: f32, f: f32| a + (b - a) * f;
    if t <= tel[0].0 {
        let s = tel[0];
        return (s.1, s.2, s.3, s.4);
    }
    for w in tel.windows(2) {
        let a = w[0];
        let b = w[1];
        if t <= b.0 {
            let f = ((t - a.0) / (b.0 - a.0).max(1e-3)).clamp(0.0, 1.0);
            return (lerp(a.1, b.1, f), lerp(a.2, b.2, f), lerp(a.3, b.3, f), a.4);
        }
    }
    let s = *tel.last().unwrap();
    (s.1, s.2, s.3, s.4)
}

fn sample_path(path: &[(f32, Vec3)], t: f32) -> Vec3 {
    if path.is_empty() {
        return Vec3::Y;
    }
    if t <= path[0].0 {
        return path[0].1;
    }
    for w in path.windows(2) {
        let (t0, p0) = w[0];
        let (t1, p1) = w[1];
        if t <= t1 {
            let f = ((t - t0) / (t1 - t0).max(1e-3)).clamp(0.0, 1.0);
            return p0.lerp(p1, f);
        }
    }
    path.last().unwrap().1
}
