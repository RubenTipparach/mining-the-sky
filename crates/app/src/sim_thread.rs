//! Dedicated physics thread for the player-flown launch.
//!
//! The rocket's flight integration (gravity + drag + thrust + rigid-body
//! attitude) runs on its OWN wall-clock here, completely separate from the
//! render loop, so a slow or fast frame can never stretch or compress a physics
//! step. The thread advances the authoritative `launch::Rocket` at a fixed
//! sub-step and publishes a snapshot; the render thread sends control inputs
//! (throttle, steering, staging) down and reads the latest snapshot back to draw
//! and report telemetry. This is the same off-thread / latest-wins pattern as
//! `terrain_job`, applied to the sim instead of the mesher.
//!
//! Determinism for tests / headless `shot` / `ascentcsv` is preserved by NOT
//! using this thread there: those paths integrate the very same `Rocket` inline
//! and synchronously (see `World::advance`). On wasm (single-threaded, no worker
//! threads without cross-origin isolation) the thread is likewise absent and the
//! sim runs inline. So this module's worker only drives the live native game.

use crate::launch::Rocket;
use sim::body::CentralBody;

/// Player / autopilot control inputs, sampled from the render-side mirror each
/// frame and applied to the authoritative rocket before each integration batch.
#[derive(Clone, Copy)]
pub struct Controls {
    pub throttle: f64,
    pub pitch: f64,
    pub yaw: f64,
    pub roll: f64,
    pub attitude_hold: bool,
    pub auto_land: bool,
    pub chute_armed: bool,
    /// Frozen re-entry test scene: hold the pose and heat instead of integrating.
    pub frozen: bool,
    /// Heat glow level for the frozen test (drives the plasma FX sweep).
    pub test_heat: f64,
    /// Time scale. The sim advances `real_seconds * warp` of flight time.
    pub warp: f32,
}

impl Default for Controls {
    fn default() -> Self {
        Controls {
            throttle: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            roll: 0.0,
            attitude_hold: false,
            auto_land: false,
            chute_armed: false,
            frozen: false,
            test_heat: 0.0,
            warp: 1.0,
        }
    }
}

/// Commands from the render thread to the sim thread.
pub enum Command {
    /// Begin owning and flying this rocket (a clone of the one the render side
    /// built on the pad), on the given central body.
    Start(Box<Rocket>, CentralBody),
    /// Jettison the active stage.
    Stage,
    /// Stop flying / drop the rocket (reset, landed, returned to the VAB).
    Stop,
    /// Latest control inputs (sent every frame).
    Set(Controls),
}

/// Fixed physics sub-step (s). Small and constant so the integration is stable
/// and the mission clock advances at a steady, frame-rate-independent rate.
const FIXED_DT: f64 = 0.004; // 250 Hz
/// Cap on sub-steps integrated per worker iteration, so an extreme time-warp (or
/// a long stall) can't make one iteration spin for ages; excess time is dropped.
const MAX_SUBSTEPS: u32 = 600;

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use super::*;
    use std::sync::mpsc::{Receiver, Sender, TryRecvError};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// Handle to the background physics worker.
    pub struct SimThread {
        tx: Sender<Command>,
        snap: Arc<Mutex<Option<Rocket>>>,
    }

    impl SimThread {
        pub fn spawn() -> Self {
            let (tx, rx) = std::sync::mpsc::channel::<Command>();
            let snap: Arc<Mutex<Option<Rocket>>> = Arc::new(Mutex::new(None));
            let snap_w = snap.clone();
            std::thread::Builder::new()
                .name("sim-physics".into())
                .spawn(move || worker(rx, snap_w))
                .expect("spawn sim-physics thread");
            SimThread { tx, snap }
        }

        /// Begin flying a clone of `rk` on its own thread.
        pub fn start(&self, rk: &Rocket, body: CentralBody) {
            let _ = self.tx.send(Command::Start(Box::new(rk.clone()), body));
            // adopt the starting state immediately so the first frames have it.
            *self.snap.lock().unwrap() = Some(rk.clone());
        }

        pub fn stop(&self) {
            let _ = self.tx.send(Command::Stop);
            *self.snap.lock().unwrap() = None;
        }

        pub fn stage(&self) {
            let _ = self.tx.send(Command::Stage);
        }

        pub fn set_controls(&self, c: Controls) {
            let _ = self.tx.send(Command::Set(c));
        }

        /// Take the latest published snapshot, if the worker has produced a new
        /// one since the last call (latest-wins; older snapshots are discarded).
        pub fn try_snapshot(&self) -> Option<Rocket> {
            self.snap.lock().unwrap().take()
        }
    }

    fn worker(rx: Receiver<Command>, snap: Arc<Mutex<Option<Rocket>>>) {
        let mut rocket: Option<Rocket> = None;
        let mut body = CentralBody::home();
        let mut controls = Controls::default();
        let mut last = Instant::now();
        let mut acc = 0.0f64;
        loop {
            // Drain every pending command; only the latest controls matter.
            loop {
                match rx.try_recv() {
                    Ok(Command::Start(rk, b)) => {
                        rocket = Some(*rk);
                        body = b;
                        acc = 0.0;
                        last = Instant::now();
                    }
                    Ok(Command::Stop) => {
                        rocket = None;
                        *snap.lock().unwrap() = None;
                    }
                    Ok(Command::Stage) => {
                        if let Some(r) = rocket.as_mut() {
                            r.jettison();
                        }
                    }
                    Ok(Command::Set(c)) => controls = c,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return, // render gone
                }
            }

            let now = Instant::now();
            let real = (now - last).as_secs_f64();
            last = now;

            if let Some(rk) = rocket.as_mut() {
                // Apply the player's control inputs to the authoritative rocket.
                rk.throttle = controls.throttle;
                rk.pitch = controls.pitch;
                rk.yaw = controls.yaw;
                rk.roll = controls.roll;
                rk.attitude_hold = controls.attitude_hold;
                rk.auto_land = controls.auto_land;
                if controls.chute_armed {
                    rk.chute_armed = true;
                }
                rk.just_staged = None;

                if controls.frozen {
                    // Frozen re-entry test: pose + heat, no motion / damage.
                    rk.place_attitude();
                    rk.health = 100.0;
                    rk.destroyed = false;
                    rk.crashed = false;
                    rk.heat = controls.test_heat;
                } else {
                    // Advance exactly `real * warp` of flight time in fixed
                    // sub-steps, carrying the remainder so the clock stays true.
                    acc += real * controls.warp.max(0.0) as f64;
                    let mut n = 0u32;
                    while acc >= FIXED_DT && n < MAX_SUBSTEPS {
                        rk.integrate(&body, FIXED_DT);
                        acc -= FIXED_DT;
                        n += 1;
                    }
                    if n >= MAX_SUBSTEPS {
                        acc = 0.0; // shed the backlog at extreme warp / after a stall
                    }
                }

                *snap.lock().unwrap() = Some(rk.clone());
            }

            // Park briefly between iterations: long enough to not busy-spin, short
            // enough that the published state stays fresh for the render thread.
            std::thread::sleep(Duration::from_micros(1500)); // ~650 Hz poll
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::*;

    /// Web stub: no worker threads, so the live game never spawns this (the sim
    /// runs inline in `World::advance`). Present only so the types resolve; its
    /// methods are inert and it never publishes a snapshot.
    pub struct SimThread;

    impl SimThread {
        pub fn spawn() -> Self {
            SimThread
        }
        pub fn start(&self, _rk: &Rocket, _body: CentralBody) {}
        pub fn stop(&self) {}
        pub fn stage(&self) {}
        pub fn set_controls(&self, _c: Controls) {}
        pub fn try_snapshot(&self) -> Option<Rocket> {
            None
        }
    }
}

pub use imp::SimThread;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use glam::DVec3;
    use sim::vehicle::Vehicle;
    use std::time::{Duration, Instant};

    fn pad_rocket(body: &CentralBody) -> Rocket {
        let veh = Vehicle::pioneer();
        let up = DVec3::new(1.0, 0.0, 0.0);
        let r = up * body.radius;
        let heading = DVec3::Y.cross(up).normalize();
        Rocket::on_pad(&veh, r, DVec3::ZERO, up, heading)
    }

    fn wait_snapshot(sim: &SimThread) -> Rocket {
        for _ in 0..400 {
            if let Some(s) = sim.try_snapshot() {
                return s;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("sim thread produced no snapshot");
    }

    /// The dedicated thread must advance the launch on its OWN wall-clock,
    /// independent of any render loop: mission time tracks real seconds at 1x,
    /// and the physics actually runs (the engine spools up).
    #[test]
    fn thread_flies_launch_on_its_own_wall_clock() {
        let body = CentralBody::home();
        let rk = pad_rocket(&body);
        let sim = SimThread::spawn();
        sim.start(&rk, body);
        sim.set_controls(Controls { throttle: 1.0, ..Controls::default() });

        let t0 = Instant::now();
        std::thread::sleep(Duration::from_millis(1000));
        let wall = t0.elapsed().as_secs_f64();

        let snap = wait_snapshot(&sim);
        // MET tracks wall-clock (slack for thread scheduling + the poll period).
        assert!(
            (snap.met - wall).abs() < 0.2,
            "MET {:.3} s drifted from wall-clock {:.3} s",
            snap.met,
            wall
        );
        // physics ran on the thread: the engine has been spooling up.
        assert!(snap.spool > 0.2, "engine did not spool on the thread: {:.3}", snap.spool);
        assert!(snap.met > 0.5, "barely advanced: MET {:.3} s", snap.met);
    }

    /// Time-warp is honoured by the thread's own clock: at 4x, MET advances
    /// roughly four times wall-clock time.
    #[test]
    fn thread_honours_time_warp() {
        let body = CentralBody::home();
        let rk = pad_rocket(&body);
        let sim = SimThread::spawn();
        sim.start(&rk, body);
        sim.set_controls(Controls { throttle: 1.0, warp: 4.0, ..Controls::default() });

        let t0 = Instant::now();
        std::thread::sleep(Duration::from_millis(500));
        let wall = t0.elapsed().as_secs_f64();

        let snap = wait_snapshot(&sim);
        assert!(
            snap.met > wall * 3.0,
            "4x warp only reached MET {:.3} s in {:.3} s wall",
            snap.met,
            wall
        );
    }
}
