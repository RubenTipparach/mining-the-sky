//! An autonomous moon-landing bot.
//!
//! The bot flies the same `Craft` the player flies - it only sets the controls
//! a human would (attitude target + throttle), then the real rotational +
//! translational physics integrate it. It runs a small guidance state machine:
//!
//! - **Deorbit**: from lunar orbit, point retrograde and burn to drop into a
//!   descent.
//! - **Brake / descent**: hold retrograde to null the moon-relative velocity on
//!   a suicide-burn-style speed-vs-altitude profile, slowing as it nears the
//!   ground.
//! - **Touchdown**: near the surface, point straight up and feather the
//!   throttle to a soft landing.
//!
//! Because it drives a `Craft`, the player can fly an identical craft by hand
//! and race the bot to the surface.

use crate::flight::{Craft, GravBody};
use sim::body::CentralBody;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BotPhase {
    Deorbit,
    Descent,
    Touchdown,
    Landed,
    Crashed,
}

impl BotPhase {
    pub fn label(self) -> &'static str {
        match self {
            BotPhase::Deorbit => "DEORBIT",
            BotPhase::Descent => "DESCENT",
            BotPhase::Touchdown => "TOUCHDOWN",
            BotPhase::Landed => "LANDED",
            BotPhase::Crashed => "CRASHED",
        }
    }
}

pub struct MoonBot {
    pub phase: BotPhase,
}

impl MoonBot {
    pub fn new() -> Self {
        MoonBot { phase: BotPhase::Deorbit }
    }

    /// Moon-relative altitude of a craft (m).
    pub fn altitude(craft: &Craft, moon: &GravBody) -> f64 {
        (craft.r - moon.center).length() - moon.radius
    }

    /// Set the craft's controls for one guidance tick. Call immediately before
    /// `craft.integrate(..)` with the same `dt`.
    pub fn control(&mut self, craft: &mut Craft, moon: &GravBody) {
        if craft.crashed {
            self.phase = BotPhase::Crashed;
            craft.throttle = 0.0;
            return;
        }
        if craft.landed && craft.landed_on == "MOON" {
            self.phase = BotPhase::Landed;
            craft.throttle = 0.0;
            craft.hold_dir = None;
            return;
        }

        let rm = craft.r - moon.center;
        let up = rm.normalize_or_zero();
        let alt = rm.length() - moon.radius;
        let vm = craft.v; // moon-relative (static moon frame)
        let speed = vm.length();
        let g = moon.mu / rm.length().powi(2);
        let acc = craft.thrust / craft.mass(); // max thrust accel
        // throttle that just cancels gravity (hover)
        let hover = (g / acc).clamp(0.0, 1.0);

        // Target speed as a function of height: a braking profile that uses most
        // of the available deceleration to arrive slow near the ground, with a
        // floor so we keep creeping down at the end.
        let a_eff = (acc - g).max(0.5);
        let brake_alt = (alt - 6.0).max(0.0);
        let v_profile = 0.62 * (2.0 * a_eff * brake_alt).sqrt();
        let v_des = v_profile.max(if alt > 15.0 { 5.0 } else { 1.0 });

        // Point opposite the velocity while we have speed to cancel; once nearly
        // stopped, point straight up so thrust holds altitude.
        craft.hold_dir = Some(if speed > 1.2 { -vm.normalize_or_zero() } else { up });

        // Throttle: hover feed-forward + proportional braking on overspeed.
        let err = speed - v_des;
        craft.throttle = (hover + 0.05 * err).clamp(0.0, 1.0);

        // Phase bookkeeping (for the HUD / race readout).
        self.phase = if alt < 60.0 {
            BotPhase::Touchdown
        } else if speed < 30.0 || alt < 0.4 * moon.radius {
            BotPhase::Descent
        } else {
            BotPhase::Deorbit
        };
    }

    /// Convenience: fly a craft to the surface, stepping guidance + physics at
    /// `dt` for up to `max_s` seconds. Returns the elapsed time. Drag from the
    /// home world is included via `home` (negligible at the moon).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn fly_to_surface(
        &mut self,
        craft: &mut Craft,
        home: &CentralBody,
        moon: &GravBody,
        dt: f64,
        max_s: f64,
    ) -> f64 {
        let mut t = 0.0;
        while t < max_s {
            self.control(craft, moon);
            craft.integrate(home, std::slice::from_ref(moon), dt);
            t += dt;
            if craft.landed || craft.crashed {
                break;
            }
        }
        // sync the final phase (the loop breaks right after the landing step)
        if craft.crashed {
            self.phase = BotPhase::Crashed;
        } else if craft.landed && craft.landed_on == "MOON" {
            self.phase = BotPhase::Landed;
        }
        t
    }
}

impl Default for MoonBot {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flight::Craft;
    use glam::DVec3;

    fn moon() -> GravBody {
        GravBody {
            center: DVec3::new(88.0e6, 0.0, 8.0e6),
            mu: 4.9e12,
            radius: 1.674e6,
            name: "MOON",
        }
    }

    /// Circular low-lunar-orbit start state (30 km), orbiting the moon.
    fn low_lunar_orbit(moon: &GravBody) -> Craft {
        let r0 = moon.center + DVec3::X * (moon.radius + 30_000.0);
        let vc = (moon.mu / (moon.radius + 30_000.0)).sqrt();
        Craft::maneuvering(r0, DVec3::Z * vc)
    }

    #[test]
    fn bot_lands_softly_on_the_moon() {
        let home = CentralBody::home();
        let moon = moon();
        let mut craft = low_lunar_orbit(&moon);
        let mut bot = MoonBot::new();
        let t = bot.fly_to_surface(&mut craft, &home, &moon, 0.1, 1200.0);

        assert!(craft.landed && !craft.crashed, "bot did not land softly: {} (phase {:?})", craft.status(), bot.phase);
        assert_eq!(craft.landed_on, "MOON");
        assert_eq!(bot.phase, BotPhase::Landed);
        // landed within a sensible time and with propellant to spare
        assert!(t < 1200.0, "bot ran out of time");
        assert!(craft.prop > 0.0, "bot ran the tank dry");
        // actually on the surface
        let surf = (craft.r - moon.center).length() - moon.radius;
        assert!(surf.abs() < 5.0, "not on the surface: {surf} m");
    }
}
