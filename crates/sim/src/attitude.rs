//! Rigid-body attitude dynamics and control.
//!
//! This is the rotational counterpart to the translational flight model: a
//! spacecraft has an orientation, an angular velocity, and a moment of inertia,
//! and it turns only when a real torque acts on it (Euler's rigid-body
//! equations). Torque comes from three effectors, mirroring a real ship:
//!
//! - **Reaction wheels / flywheels**: spin up internal rotors to exchange
//!   angular momentum with the hull. Torque- and momentum-limited; they
//!   *saturate* once the rotors hit their speed limit and then need dumping.
//! - **RCS thrusters**: cold-gas/monoprop jets that produce external torque and
//!   burn propellant. They also *desaturate* the wheels (dump stored momentum).
//! - **Gimbaled main engine**: while the engine is firing, deflecting the
//!   nozzle a few degrees produces a large control torque for free.
//!
//! On top of the dynamics sits an attitude autopilot: a quaternion-feedback
//! PID (proportional-derivative on the pointing error, with a small integral
//! term) that drives the ship to hold any of the six orbital reference
//! directions - prograde / retrograde, normal / anti-normal, radial-in /
//! radial-out - for navigation planning. A PID is the right tool here:
//! rigid-body pointing is a smooth, well-conditioned regulation problem, so a
//! tuned quaternion PD with rate damping is globally stable and cheap. MPC
//! would only pay off with hard actuator constraints or path constraints we do
//! not have for free attitude slews; the allocator below already handles the
//! actuator limits.

use glam::{DQuat, DVec3};

/// The body axis the main engine thrusts along and that the autopilot points
/// (the ship's "nose"). +Y matches the rocket/lander meshes, which are built
/// pointing up +Y.
pub const THRUST_AXIS: DVec3 = DVec3::Y;

/// The six orbital reference attitudes used for navigation, plus a free-drift
/// mode. These are the directions a maneuver is planned against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttitudeMode {
    /// No autopilot: the ship coasts at whatever rate it has (rates damped only
    /// if `damp_when_free` is set by the caller).
    Free,
    Prograde,
    Retrograde,
    Normal,
    AntiNormal,
    RadialOut,
    RadialIn,
}

impl AttitudeMode {
    pub fn label(self) -> &'static str {
        match self {
            AttitudeMode::Free => "FREE",
            AttitudeMode::Prograde => "PROGRADE",
            AttitudeMode::Retrograde => "RETROGRADE",
            AttitudeMode::Normal => "NORMAL",
            AttitudeMode::AntiNormal => "ANTI-NORMAL",
            AttitudeMode::RadialOut => "RADIAL OUT",
            AttitudeMode::RadialIn => "RADIAL IN",
        }
    }

    /// The six pointing modes (excludes Free), in HUD order.
    pub fn six() -> [AttitudeMode; 6] {
        [
            AttitudeMode::Prograde,
            AttitudeMode::Retrograde,
            AttitudeMode::Normal,
            AttitudeMode::AntiNormal,
            AttitudeMode::RadialOut,
            AttitudeMode::RadialIn,
        ]
    }
}

/// World-frame unit direction the nose should point for a given mode, from the
/// orbital state `(r, v)` about the body centre. `None` for `Free`.
pub fn target_dir(mode: AttitudeMode, r: DVec3, v: DVec3) -> Option<DVec3> {
    let h = r.cross(v); // specific angular momentum (orbit normal)
    let d = match mode {
        AttitudeMode::Free => return None,
        AttitudeMode::Prograde => v,
        AttitudeMode::Retrograde => -v,
        AttitudeMode::Normal => h,
        AttitudeMode::AntiNormal => -h,
        AttitudeMode::RadialOut => r,
        AttitudeMode::RadialIn => -r,
    };
    let n = d.normalize_or_zero();
    if n == DVec3::ZERO {
        None
    } else {
        Some(n)
    }
}

/// A rigid body with a diagonal (principal-axis) inertia tensor.
#[derive(Clone, Copy)]
pub struct RigidBody {
    /// Body -> world orientation.
    pub orient: DQuat,
    /// Angular velocity in the BODY frame (rad/s).
    pub omega: DVec3,
    /// Principal moments of inertia about the body axes (kg m^2).
    pub inertia: DVec3,
}

impl RigidBody {
    pub fn new(inertia: DVec3) -> Self {
        RigidBody {
            orient: DQuat::IDENTITY,
            omega: DVec3::ZERO,
            inertia,
        }
    }

    /// A solid-ish cylinder of `mass` (kg), `radius` and `half_len` (m), with
    /// its long axis along the body +Y (the thrust axis).
    pub fn cylinder(mass: f64, radius: f64, half_len: f64) -> Self {
        let len = 2.0 * half_len;
        let axial = 0.5 * mass * radius * radius;
        let transverse = mass * (3.0 * radius * radius + len * len) / 12.0;
        RigidBody::new(DVec3::new(transverse, axial, transverse))
    }

    /// Snap the orientation so the nose points along `dir_world`, at zero rate.
    pub fn point_at(&mut self, dir_world: DVec3) {
        let d = dir_world.normalize_or_zero();
        if d != DVec3::ZERO {
            self.orient = DQuat::from_rotation_arc(THRUST_AXIS, d);
            self.omega = DVec3::ZERO;
        }
    }

    /// The nose direction in world coordinates.
    pub fn nose(&self) -> DVec3 {
        (self.orient * THRUST_AXIS).normalize_or_zero()
    }

    /// World angular velocity (for readouts).
    pub fn omega_world(&self) -> DVec3 {
        self.orient * self.omega
    }

    /// Advance the rotational state by `dt` under a BODY-frame torque, via
    /// Euler's equations `I w_dot = tau - w x (I w)` plus quaternion kinematics.
    pub fn integrate(&mut self, torque_body: DVec3, dt: f64) {
        let i = self.inertia;
        let w = self.omega;
        let iw = DVec3::new(i.x * w.x, i.y * w.y, i.z * w.z);
        let gyro = w.cross(iw); // w x (I w)
        let wdot = DVec3::new(
            (torque_body.x - gyro.x) / i.x,
            (torque_body.y - gyro.y) / i.y,
            (torque_body.z - gyro.z) / i.z,
        );
        self.omega += wdot * dt;

        // q_dot = 0.5 * q * (omega as pure quaternion), integrated explicitly.
        let omega_q = DQuat::from_xyzw(self.omega.x, self.omega.y, self.omega.z, 0.0);
        let qd = self.orient.mul_quat(omega_q);
        let s = 0.5 * dt;
        let q = self.orient;
        self.orient = DQuat::from_xyzw(
            q.x + qd.x * s,
            q.y + qd.y * s,
            q.z + qd.z * s,
            q.w + qd.w * s,
        )
        .normalize();
    }

    /// Stored angular momentum of the hull (world frame).
    pub fn momentum_world(&self) -> DVec3 {
        let i = self.inertia;
        let l_body = DVec3::new(i.x * self.omega.x, i.y * self.omega.y, i.z * self.omega.z);
        self.orient * l_body
    }
}

/// Internal momentum-storage wheels. They deliver torque to the hull while
/// absorbing the opposite momentum into the rotors; once a rotor hits its
/// momentum limit it can give no more torque in that direction (saturation).
#[derive(Clone, Copy)]
pub struct ReactionWheels {
    /// Stored rotor momentum, BODY frame (N m s).
    pub h: DVec3,
    /// Per-axis momentum capacity (N m s).
    pub h_max: f64,
    /// Per-axis torque capacity (N m).
    pub torque_max: f64,
}

impl ReactionWheels {
    pub fn new(h_max: f64, torque_max: f64) -> Self {
        ReactionWheels { h: DVec3::ZERO, h_max, torque_max }
    }

    /// Saturation fraction (0 = empty, 1 = fully spun up) on the worst axis.
    pub fn saturation(&self) -> f64 {
        (self.h.abs().max_element() / self.h_max).clamp(0.0, 1.0)
    }

    /// Try to deliver `cmd` (body torque) for `dt`. Returns the torque actually
    /// applied to the hull; updates stored rotor momentum, clamping at `h_max`
    /// so a saturated axis delivers nothing further.
    fn apply(&mut self, cmd: DVec3, dt: f64) -> DVec3 {
        let lim = self.torque_max;
        let want = DVec3::new(
            cmd.x.clamp(-lim, lim),
            cmd.y.clamp(-lim, lim),
            cmd.z.clamp(-lim, lim),
        );
        // Applying torque +tau to the hull spins the rotor the other way:
        // h_new = h - tau*dt. Clamp h to capacity and back out the real torque.
        let h_target = self.h - want * dt;
        let h_clamped = DVec3::new(
            h_target.x.clamp(-self.h_max, self.h_max),
            h_target.y.clamp(-self.h_max, self.h_max),
            h_target.z.clamp(-self.h_max, self.h_max),
        );
        let applied = (self.h - h_clamped) / dt;
        self.h = h_clamped;
        applied
    }
}

/// RCS thruster set: external control torque that burns propellant, and which
/// can also dump (desaturate) wheel momentum.
#[derive(Clone, Copy)]
pub struct Rcs {
    /// Per-axis torque capacity (N m).
    pub torque_max: f64,
    /// Effective specific impulse of the thrusters (s).
    pub isp: f64,
    /// Effective moment arm (m): torque -> equivalent thrust = torque / arm.
    pub arm: f64,
    pub prop: f64,
    pub prop_full: f64,
}

impl Rcs {
    pub fn new(torque_max: f64, isp: f64, arm: f64, prop: f64) -> Self {
        Rcs { torque_max, isp, arm, prop, prop_full: prop, }
    }

    pub fn prop_frac(&self) -> f64 {
        (self.prop / self.prop_full.max(1e-9)).clamp(0.0, 1.0)
    }

    /// Deliver up to `torque_max` per axis of `cmd`, burning propellant for the
    /// magnitude used. Returns the torque actually applied to the hull.
    fn apply(&mut self, cmd: DVec3, dt: f64) -> DVec3 {
        if self.prop <= 0.0 {
            return DVec3::ZERO;
        }
        let lim = self.torque_max;
        let tau = DVec3::new(
            cmd.x.clamp(-lim, lim),
            cmd.y.clamp(-lim, lim),
            cmd.z.clamp(-lim, lim),
        );
        self.burn(tau.abs().element_sum(), dt);
        tau
    }

    /// Burn propellant for a torque magnitude sustained over `dt`. Mass flow =
    /// equivalent thrust / (isp g0), equivalent thrust = torque / arm.
    fn burn(&mut self, torque_mag: f64, dt: f64) {
        if torque_mag <= 0.0 {
            return;
        }
        let thrust_equiv = torque_mag / self.arm;
        let mdot = thrust_equiv / (self.isp * crate::vehicle::G0);
        self.prop = (self.prop - mdot * dt).max(0.0);
    }
}

/// A simple gimbaled-engine torque source, available only while the main engine
/// is thrusting. Deflecting the nozzle by up to `max_deg` produces a torque of
/// `thrust * sin(angle) * arm` in the two axes perpendicular to the nose. It is
/// "free" (no extra propellant beyond the main burn), so the allocator uses it
/// first during powered flight.
#[derive(Clone, Copy)]
pub struct Gimbal {
    pub max_deg: f64,
    /// Distance from the gimbal pivot to the centre of mass (m).
    pub arm: f64,
}

impl Gimbal {
    pub fn new(max_deg: f64, arm: f64) -> Self {
        Gimbal { max_deg, arm }
    }

    /// Max control torque (about each transverse body axis) at the current
    /// engine `thrust_n`. Zero about the roll (nose) axis - a single gimbal
    /// cannot roll the ship.
    pub fn torque_limit(&self, thrust_n: f64) -> f64 {
        thrust_n * self.max_deg.to_radians().sin() * self.arm
    }
}

/// Quaternion-feedback attitude autopilot (PID with rate damping). Operates in
/// the BODY frame: it points the nose (`THRUST_AXIS`) at a target world
/// direction. Roll about the nose is left free (reduced-attitude control).
#[derive(Clone, Copy)]
pub struct AttitudeController {
    pub kp: f64,
    pub kd: f64,
    pub ki: f64,
    /// Integrator state (body frame), with anti-windup clamp.
    pub integ: DVec3,
    pub integ_max: f64,
}

impl AttitudeController {
    /// Gains tuned (critically-ish damped) for the maneuvering craft's inertia.
    pub fn new() -> Self {
        AttitudeController {
            kp: 6.0,
            kd: 14.0,
            ki: 0.05,
            integ: DVec3::ZERO,
            integ_max: 2.0,
        }
    }

    /// The pointing error: the body-frame rotation vector (axis * angle, rad)
    /// that would turn the nose onto `target_world`. Magnitude is the angle.
    pub fn error(rb: &RigidBody, target_world: DVec3) -> DVec3 {
        let nose = rb.nose();
        let tgt = target_world.normalize_or_zero();
        if tgt == DVec3::ZERO {
            return DVec3::ZERO;
        }
        let axis = nose.cross(tgt);
        let s = axis.length();
        let c = nose.dot(tgt).clamp(-1.0, 1.0);
        let angle = s.atan2(c); // 0..pi
        let e_world = if s > 1e-9 {
            axis / s * angle
        } else if c < 0.0 {
            // antiparallel: pick any axis perpendicular to the nose
            let perp = if nose.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
            nose.cross(perp).normalize_or_zero() * angle
        } else {
            DVec3::ZERO
        };
        // into the body frame (omega and torque live there)
        rb.orient.inverse() * e_world
    }

    /// Commanded body torque (per unit inertia is folded into the gains by the
    /// caller scaling, but here we return a raw torque coefficient * inertia in
    /// `command_torque`). This returns the *acceleration* command (rad/s^2).
    fn accel_cmd(&mut self, rb: &RigidBody, target_world: Option<DVec3>, dt: f64) -> DVec3 {
        let e = match target_world {
            Some(t) => Self::error(rb, t),
            None => DVec3::ZERO, // free: damp rates only
        };
        if target_world.is_some() {
            self.integ += e * dt;
            let m = self.integ_max;
            self.integ = DVec3::new(
                self.integ.x.clamp(-m, m),
                self.integ.y.clamp(-m, m),
                self.integ.z.clamp(-m, m),
            );
        }
        self.kp * e + self.ki * self.integ - self.kd * rb.omega
    }

    /// Commanded BODY torque to slew toward `target_world` (or just damp when
    /// `None`). Scales the PID acceleration by the inertia so the response is
    /// consistent across axes.
    pub fn command_torque(
        &mut self,
        rb: &RigidBody,
        target_world: Option<DVec3>,
        dt: f64,
    ) -> DVec3 {
        let a = self.accel_cmd(rb, target_world, dt);
        DVec3::new(a.x * rb.inertia.x, a.y * rb.inertia.y, a.z * rb.inertia.z)
    }
}

impl Default for AttitudeController {
    fn default() -> Self {
        Self::new()
    }
}

/// What each effector contributed on a control step (for telemetry/HUD).
#[derive(Clone, Copy, Default)]
pub struct TorqueReport {
    pub gimbal: DVec3,
    pub wheels: DVec3,
    pub rcs: DVec3,
    /// Commanded torque that no effector could supply (authority shortfall).
    pub unmet: DVec3,
}

/// Allocate a commanded body torque across the effectors, in priority order:
/// gimbal (free, only while thrusting) -> reaction wheels (free, until
/// saturated) -> RCS (burns propellant). Also desaturates the wheels with RCS
/// when they are spun up past `dump_above` (attitude-neutral momentum dump).
/// Returns the net torque applied to the hull and a per-effector report.
#[allow(clippy::too_many_arguments)]
pub fn allocate(
    cmd: DVec3,
    wheels: &mut ReactionWheels,
    rcs: &mut Rcs,
    gimbal: Gimbal,
    thrust_n: f64,
    dt: f64,
) -> (DVec3, TorqueReport) {
    let mut report = TorqueReport::default();
    let mut remaining = cmd;

    // 1) Gimbal: only the two transverse axes, only while the engine fires.
    if thrust_n > 0.0 {
        let lim = gimbal.torque_limit(thrust_n);
        // transverse axes are body X and Z (Y is the nose/roll axis)
        let g = DVec3::new(
            remaining.x.clamp(-lim, lim),
            0.0,
            remaining.z.clamp(-lim, lim),
        );
        report.gimbal = g;
        remaining -= g;
    }

    // 2) Reaction wheels (no propellant).
    let w = wheels.apply(remaining, dt);
    report.wheels = w;
    remaining -= w;

    // 3) RCS (propellant) for whatever is left.
    let r = rcs.apply(remaining, dt);
    report.rcs = r;
    remaining -= r;

    report.unmet = remaining;

    // 4) Momentum dumping: if the wheels are spun up, fire RCS to bleed their
    // stored momentum back toward zero while the wheels react to hold attitude
    // (net hull torque ~0). This is a gentle background process (a fraction of
    // the wheel torque), so a heavy sustained demand still saturates the wheels
    // and forces the RCS handover above. Costs propellant only.
    let sat = wheels.saturation();
    if sat > 0.8 && rcs.prop > 0.0 {
        let dump_rate = 0.3 * wheels.torque_max;
        let dir = -wheels.h.normalize_or_zero();
        let dump = dir * dump_rate;
        // wheels absorb the opposite so attitude is undisturbed
        let dh = dump * dt;
        wheels.h += dh;
        // clamp toward zero (don't overshoot past zero)
        wheels.h = DVec3::new(
            zero_clamp(wheels.h.x),
            zero_clamp(wheels.h.y),
            zero_clamp(wheels.h.z),
        );
        rcs.burn(dump.abs().element_sum(), dt);
    }

    (report.gimbal + report.wheels + report.rcs, report)
}

fn zero_clamp(x: f64) -> f64 {
    // numerical guard so dumping settles exactly at zero rather than chattering
    if x.abs() < 1e-6 {
        0.0
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn maneuvering_rig() -> (RigidBody, ReactionWheels, Rcs, Gimbal, AttitudeController) {
        // ~14 t maneuvering craft, 2 m radius, 4 m half-length.
        let rb = RigidBody::cylinder(14_000.0, 2.0, 4.0);
        let wheels = ReactionWheels::new(800.0, 250.0);
        let rcs = Rcs::new(4_000.0, 230.0, 2.5, 200.0);
        let gimbal = Gimbal::new(5.0, 6.0);
        (rb, wheels, rcs, gimbal, AttitudeController::new())
    }

    fn slew_to(target: DVec3, secs: f64) -> (f64, Rcs, ReactionWheels) {
        let (mut rb, mut wheels, mut rcs, gimbal, mut ctrl) = maneuvering_rig();
        // start pointing 120 deg away from the target
        rb.point_at(DVec3::new(0.0, -0.5, 0.866));
        let dt = 0.05;
        let steps = (secs / dt) as i32;
        for _ in 0..steps {
            let cmd = ctrl.command_torque(&mut rb, Some(target), dt);
            let (tau, _) = allocate(cmd, &mut wheels, &mut rcs, gimbal, 0.0, dt);
            rb.integrate(tau, dt);
        }
        let err = AttitudeController::error(&rb, target).length().to_degrees();
        (err, rcs, wheels)
    }

    #[test]
    fn slews_to_target_and_holds() {
        let (err, _rcs, _wheels) = slew_to(DVec3::X, 90.0);
        assert!(err < 1.0, "pointing error did not converge: {err} deg");
    }

    #[test]
    fn settles_with_low_residual_rate() {
        let (mut rb, mut wheels, mut rcs, gimbal, mut ctrl) = maneuvering_rig();
        rb.point_at(DVec3::new(1.0, 0.2, 0.0));
        let target = DVec3::new(0.0, 0.0, 1.0);
        let dt = 0.05;
        for _ in 0..(120.0 / dt) as i32 {
            let cmd = ctrl.command_torque(&mut rb, Some(target), dt);
            let (tau, _) = allocate(cmd, &mut wheels, &mut rcs, gimbal, 0.0, dt);
            rb.integrate(tau, dt);
        }
        assert!(rb.omega.length() < 1e-3, "residual rate too high: {}", rb.omega.length());
        let err = AttitudeController::error(&rb, target).length().to_degrees();
        assert!(err < 1.0, "did not hold target: {err} deg");
    }

    #[test]
    fn free_drift_is_damped_to_rest() {
        let (mut rb, mut wheels, mut rcs, gimbal, mut ctrl) = maneuvering_rig();
        rb.omega = DVec3::new(0.05, 0.0, -0.03); // tumbling
        let dt = 0.05;
        for _ in 0..(120.0 / dt) as i32 {
            let cmd = ctrl.command_torque(&mut rb, None, dt);
            let (tau, _) = allocate(cmd, &mut wheels, &mut rcs, gimbal, 0.0, dt);
            rb.integrate(tau, dt);
        }
        assert!(rb.omega.length() < 1e-3, "free-drift rates not damped: {}", rb.omega.length());
    }

    #[test]
    fn wheels_saturate_then_rcs_takes_over() {
        // A persistent one-sided demand should spin the wheels to saturation;
        // once saturated, the wheels can deliver no more and RCS must carry it.
        let (rb, mut wheels, mut rcs, gimbal, _ctrl) = maneuvering_rig();
        let dt = 0.05;
        let cmd = DVec3::new(0.0, 0.0, 200.0); // steady torque demand, body Z
        let mut saturated_seen = false;
        let mut rcs_used = false;
        for _ in 0..8000 {
            let before = rcs.prop;
            let (_tau, rep) = allocate(cmd, &mut wheels, &mut rcs, gimbal, 0.0, dt);
            if wheels.saturation() > 0.95 {
                saturated_seen = true;
            }
            if saturated_seen && rep.rcs.length() > 1.0 && rcs.prop < before {
                rcs_used = true;
                break;
            }
        }
        assert!(saturated_seen, "wheels never saturated");
        assert!(rcs_used, "RCS did not take over after saturation");
    }

    #[test]
    fn euler_free_precession_conserves_momentum() {
        // Torque-free asymmetric body: world angular momentum is conserved even
        // though body-frame omega changes (gyroscopic precession).
        let mut rb = RigidBody::new(DVec3::new(2.0, 1.0, 3.0));
        rb.omega = DVec3::new(0.4, 0.3, 0.2);
        let l0 = rb.momentum_world().length();
        for _ in 0..20_000 {
            rb.integrate(DVec3::ZERO, 0.001);
        }
        let l1 = rb.momentum_world().length();
        assert!((l1 - l0).abs() / l0 < 1e-3, "momentum drifted {l0} -> {l1}");
    }

    #[test]
    fn target_dirs_are_orthonormal_pairs() {
        let r = DVec3::new(7.0e6, 0.0, 0.0);
        let v = DVec3::new(0.0, 0.0, 7500.0);
        let pro = target_dir(AttitudeMode::Prograde, r, v).unwrap();
        let retro = target_dir(AttitudeMode::Retrograde, r, v).unwrap();
        let nrm = target_dir(AttitudeMode::Normal, r, v).unwrap();
        let out = target_dir(AttitudeMode::RadialOut, r, v).unwrap();
        assert!((pro + retro).length() < 1e-9, "pro/retro not opposite");
        assert!(pro.dot(nrm).abs() < 1e-9, "prograde not perpendicular to normal");
        assert!(pro.dot(out).abs() < 1e-9, "prograde not perpendicular to radial");
        // normal should be +Y here (r x v = x_hat x z_hat = -y... check sign)
        assert!((nrm.dot(DVec3::Y)).abs() > 0.99, "normal not along the orbit pole");
        assert!(matches!(target_dir(AttitudeMode::Free, r, v), None));
        let _ = PI;
    }
}
