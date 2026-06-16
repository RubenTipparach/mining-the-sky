//! Orbital mechanics and launch-to-orbit simulation for Mining the Sky.
//!
//! - `orbit`: analytic two-body state <-> elements (the "on-rails" backbone).
//! - `body`: the central body and its atmosphere.
//! - `vehicle`: staged launch vehicles.
//! - `ascent`: RK4 powered ascent + analytic circularization.
//! - `plot`: CPU launch/orbit diagram.

pub mod ascent;
pub mod body;
pub mod orbit;
pub mod plot;
pub mod vehicle;
