//! Powered-ascent simulation: stack a rocket on the pad at the spaceport and
//! fly a programmed gravity turn to a target apoapsis, staging as tanks empty,
//! then circularize analytically at apoapsis.
//!
//! Powered atmospheric ascent is short, so we integrate it with RK4 (drift is
//! irrelevant over ~10 minutes). Once coasting/in orbit we switch to the
//! analytic two-body model -- the design's "integrate only when thrusting,
//! otherwise stay on rails" rule.

use crate::body::CentralBody;
use crate::orbit::{circular_speed, orbit_from_state, vis_viva, Orbit};
use crate::vehicle::{Vehicle, G0};
use glam::DVec3;

const DRAG_CDA: f64 = 6.0; // Cd * frontal area (m^2)
const KICK_ALT: f64 = 1_000.0; // begin pitch program above this altitude (m)
const TURN_TOP: f64 = 70_000.0; // thrust is horizontal by here (m)

#[derive(Clone, Copy)]
pub struct Sample {
    pub t: f64,
    pub alt: f64,
    pub speed: f64,
    pub downrange: f64,
    pub mass: f64,
    pub stage: usize,
    pub r: DVec3,
    pub v: DVec3,
}

pub struct LaunchResult {
    pub vehicle: &'static str,
    pub samples: Vec<Sample>,
    pub events: Vec<(f64, String)>,
    pub meco: Option<Sample>,
    pub ascent_dv: f64, // delta-v the engines produced up to MECO
    pub circ_dv: f64,
    pub circ_prop_used: f64,
    pub prop_left_after_circ: f64,
    pub final_orbit: Orbit,
    pub reached_orbit: bool,
    pub up0: DVec3,
    pub east0: DVec3,
    pub target_apo_alt: f64,
}

struct Guidance {
    radius: f64,
}

impl Guidance {
    /// Thrust direction: vertical near the pad, then an altitude-scheduled
    /// pitch toward the local east, horizontal by `TURN_TOP`.
    fn thrust_dir(&self, r: DVec3) -> DVec3 {
        let up = r.normalize();
        let east = DVec3::Y.cross(up).normalize();
        let alt = r.length() - self.radius;
        if alt < KICK_ALT {
            up
        } else {
            let frac = ((alt - KICK_ALT) / (TURN_TOP - KICK_ALT)).clamp(0.0, 1.0);
            let theta = frac * std::f64::consts::FRAC_PI_2;
            (up * theta.cos() + east * theta.sin()).normalize()
        }
    }
}

fn accel(
    body: &CentralBody,
    r: DVec3,
    v: DVec3,
    thrust_dir: DVec3,
    thrust: f64,
    mass: f64,
) -> DVec3 {
    // gravity
    let rmag = r.length();
    let g = -body.mu / (rmag * rmag * rmag) * r;
    // drag relative to the rotating atmosphere
    let v_atm = body.omega() * DVec3::Y.cross(r);
    let v_rel = v - v_atm;
    let vr = v_rel.length();
    let alt = rmag - body.radius;
    let rho = body.density(alt);
    let drag = if vr > 1e-3 {
        -0.5 * rho * vr * DRAG_CDA / mass * v_rel
    } else {
        DVec3::ZERO
    };
    let thr = if thrust > 0.0 { thrust_dir * (thrust / mass) } else { DVec3::ZERO };
    g + drag + thr
}

pub fn simulate(
    body: &CentralBody,
    veh: &Vehicle,
    lat_deg: f64,
    lon_deg: f64,
    target_apo_alt: f64,
) -> LaunchResult {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let up0 = DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin());
    let east0 = DVec3::Y.cross(up0).normalize();

    let mut r = up0 * body.radius;
    // start on the rotating surface
    let mut v = body.omega() * DVec3::Y.cross(r);

    let guide = Guidance { radius: body.radius };
    let dt = 0.05;

    let mut stage_idx = 0usize;
    let mut prop_rem: Vec<f64> = veh.stages.iter().map(|s| s.prop).collect();
    let mut samples = Vec::new();
    let mut events = Vec::new();
    let mut ascent_dv = 0.0;
    let mut meco: Option<Sample> = None;
    let mut t = 0.0;

    let mass_now = |stage_idx: usize, prop_rem: &[f64]| -> f64 {
        let mut m = veh.payload;
        for i in stage_idx..veh.stages.len() {
            m += veh.stages[i].dry + prop_rem[i];
        }
        m
    };

    events.push((0.0, format!("Liftoff from {:.1}N {:.1}E", lat_deg, lon_deg)));

    let mut step = 0u64;
    loop {
        let mass = mass_now(stage_idx, &prop_rem);
        let alt = r.length() - body.radius;
        let downrange = up0.dot(r.normalize()).clamp(-1.0, 1.0).acos() * body.radius;

        if step % 20 == 0 {
            samples.push(Sample {
                t,
                alt,
                speed: v.length(),
                downrange,
                mass,
                stage: stage_idx,
                r,
                v,
            });
        }

        // apoapsis-targeting cutoff
        let orb = orbit_from_state(r, v, body.mu);
        if orb.ra - body.radius >= target_apo_alt && alt > body.atmo_top * 0.4 {
            meco = Some(Sample { t, alt, speed: v.length(), downrange, mass, stage: stage_idx, r, v });
            events.push((t, format!("MECO at {:.1} km, v={:.0} m/s", alt / 1000.0, v.length())));
            break;
        }

        // engine on?
        let thrusting = stage_idx < veh.stages.len() && prop_rem[stage_idx] > 0.0;
        let thrust = if thrusting { veh.stages[stage_idx].thrust } else { 0.0 };
        let tdir = guide.thrust_dir(r);

        // RK4 step (thrust dir + magnitude held constant across the step)
        let a1 = accel(body, r, v, tdir, thrust, mass);
        let a2 = accel(body, r + v * (dt * 0.5), v + a1 * (dt * 0.5), tdir, thrust, mass);
        let a3 = accel(body, r + (v + a1 * (dt * 0.5)) * (dt * 0.5), v + a2 * (dt * 0.5), tdir, thrust, mass);
        let a4 = accel(body, r + (v + a2 * (dt * 0.5)) * dt, v + a3 * dt, tdir, thrust, mass);
        let r_new = r + (v + (a1 + a2 + a3) * (dt / 6.0)) * dt;
        let v_new = v + (a1 + 2.0 * a2 + 2.0 * a3 + a4) * (dt / 6.0);
        r = r_new;
        v = v_new;

        // burn propellant + accumulate produced delta-v
        if thrusting {
            let st = &veh.stages[stage_idx];
            let burn = st.mdot() * dt;
            prop_rem[stage_idx] -= burn;
            ascent_dv += st.thrust / mass * dt;
            if prop_rem[stage_idx] <= 0.0 {
                prop_rem[stage_idx] = 0.0;
                if stage_idx + 1 < veh.stages.len() {
                    events.push((t, format!("Stage {} sep ({} depleted)", stage_idx + 1, st.name)));
                    stage_idx += 1;
                } else {
                    events.push((t, "Final stage depleted".to_string()));
                }
            }
        }

        t += dt;
        step += 1;
        if t > 1200.0 || r.length() < body.radius - 100.0 {
            break; // safety: timeout or impact
        }
    }

    // --- Analytic circularization at apoapsis ---
    let (final_orbit, circ_dv, circ_prop_used, prop_left, reached) = if let Some(m) = meco {
        let orb = orbit_from_state(m.r, m.v, body.mu);
        let ra = orb.ra;
        let v_apo = vis_viva(body.mu, ra, orb.a); // tangential speed at apoapsis
        let v_circ = circular_speed(body.mu, ra);
        let dv = (v_circ - v_apo).max(0.0);

        // can the active stage provide it?
        let st = &veh.stages[m.stage];
        let m_after = m.mass / (dv / (st.isp * G0)).exp();
        let prop_used = m.mass - m_after;
        let avail = prop_rem[m.stage];
        let ok = prop_used <= avail + 1.0 && (ra - body.radius) > body.atmo_top;

        // resulting orbit: circular at apoapsis radius
        let circ = Orbit {
            a: ra,
            e: 0.0,
            ra,
            rp: ra,
            period: Some(2.0 * std::f64::consts::PI * (ra.powi(3) / body.mu).sqrt()),
            h_vec: orb.h_vec,
            e_vec: orb.e_vec,
        };
        (circ, dv, prop_used, (avail - prop_used).max(0.0), ok)
    } else {
        let orb = orbit_from_state(r, v, body.mu);
        (orb, 0.0, 0.0, 0.0, false)
    };

    if reached {
        events.push((
            meco.map(|m| m.t).unwrap_or(t),
            format!("Circularize: dv={:.0} m/s -> orbit", circ_dv),
        ));
    }

    LaunchResult {
        vehicle: veh.name,
        samples,
        events,
        meco,
        ascent_dv,
        circ_dv,
        circ_prop_used,
        prop_left_after_circ: prop_left,
        final_orbit,
        reached_orbit: reached,
        up0,
        east0,
        target_apo_alt,
    }
}
