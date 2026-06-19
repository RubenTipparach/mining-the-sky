//! Off-thread planet-terrain meshing.
//!
//! The planet LOD mesh is rebuilt whenever the floating-origin reference snaps
//! (the camera moves far enough from the last reference). Building it on the
//! render thread causes the frame spikes that show up during ascent. On native
//! we hand the build to a background worker and double-buffer: the render thread
//! keeps drawing the current mesh until the new one arrives, then swaps. On wasm
//! there are no worker threads without cross-origin isolation, so the build runs
//! inline (same behaviour as before) - hence the platform split below.

use crate::rocket::{self, MeshVertex};
use glam::DVec3;

/// Inputs for one planet-terrain rebuild. Every field is `Copy`, so the whole
/// job is trivially `Send` - nothing borrowed crosses the thread boundary.
#[derive(Clone, Copy)]
pub struct PlanetJob {
    pub cam_world: DVec3,
    pub ref_local: DVec3,
    pub origin: DVec3,
    pub up: DVec3,
    pub east: DVec3,
    pub north: DVec3,
    pub depth: u32,
    pub lunar: bool,
}

impl PlanetJob {
    fn run(&self) -> Vec<MeshVertex> {
        rocket::planet_terrain(
            self.cam_world,
            self.ref_local,
            self.origin,
            self.up,
            self.east,
            self.north,
            self.depth,
            self.lunar,
        )
        .verts
    }
}

/// A finished mesh plus the reference origin it was built around. The render
/// thread adopts that origin at swap time so terrain and camera stay consistent.
pub struct TerrainResult {
    pub ref_local: DVec3,
    pub verts: Vec<MeshVertex>,
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use super::*;
    use std::sync::mpsc::{Receiver, Sender, TryRecvError};

    /// A persistent background mesher. One job is in flight at a time (the caller
    /// gates on [`busy`](TerrainService::busy)); the worker coalesces to the most
    /// recent queued job so a burst of snaps can't back up a queue of stale work.
    pub struct TerrainService {
        tx: Sender<PlanetJob>,
        rx: Receiver<TerrainResult>,
        busy: bool,
    }

    impl TerrainService {
        pub fn new() -> Self {
            let (job_tx, job_rx) = std::sync::mpsc::channel::<PlanetJob>();
            let (res_tx, res_rx) = std::sync::mpsc::channel::<TerrainResult>();
            std::thread::Builder::new()
                .name("terrain-mesher".into())
                .spawn(move || {
                    while let Ok(job) = job_rx.recv() {
                        // Drain any newer requests; only the latest matters.
                        let mut latest = job;
                        while let Ok(j) = job_rx.try_recv() {
                            latest = j;
                        }
                        let verts = latest.run();
                        if res_tx
                            .send(TerrainResult { ref_local: latest.ref_local, verts })
                            .is_err()
                        {
                            break; // render side gone; shut the worker down
                        }
                    }
                })
                .expect("spawn terrain-mesher thread");
            Self { tx: job_tx, rx: res_rx, busy: false }
        }

        pub fn busy(&self) -> bool {
            self.busy
        }

        pub fn request(&mut self, job: PlanetJob) {
            self.busy = true;
            // If the worker has died the send fails; clear busy so we fall back
            // to a synchronous rebuild rather than hanging forever.
            if self.tx.send(job).is_err() {
                self.busy = false;
            }
        }

        pub fn try_recv(&mut self) -> Option<TerrainResult> {
            match self.rx.try_recv() {
                Ok(r) => {
                    self.busy = false;
                    Some(r)
                }
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    self.busy = false;
                    None
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::*;

    /// Web fallback: no worker threads, so a request meshes inline and the
    /// result is handed back on the next poll. The build still happens on the
    /// render thread (unavoidable without cross-origin isolation), so this keeps
    /// the pre-existing behaviour on the web rather than improving it.
    pub struct TerrainService {
        pending: Option<TerrainResult>,
    }

    impl TerrainService {
        pub fn new() -> Self {
            Self { pending: None }
        }

        pub fn busy(&self) -> bool {
            false
        }

        pub fn request(&mut self, job: PlanetJob) {
            self.pending = Some(TerrainResult { ref_local: job.ref_local, verts: job.run() });
        }

        pub fn try_recv(&mut self) -> Option<TerrainResult> {
            self.pending.take()
        }
    }
}

pub use imp::TerrainService;
