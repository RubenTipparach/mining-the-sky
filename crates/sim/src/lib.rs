//! Orbital mechanics and launch-to-orbit simulation for Mining the Sky.
//!
//! - `orbit`: analytic two-body state <-> elements (the "on-rails" backbone).
//! - `body`: the central body and its atmosphere.
//! - `vehicle`: staged launch vehicles.
//! - `ascent`: RK4 powered ascent + analytic circularization.
//! - `attitude`: rigid-body rotational dynamics + reaction-wheel/RCS/gimbal
//!   effectors + a quaternion-PID autopilot for the six orbital attitudes.
//! - `plot`: CPU launch/orbit diagram.

pub mod ascent;
pub mod attitude;
pub mod body;
pub mod orbit;
pub mod plot;
pub mod vehicle;
