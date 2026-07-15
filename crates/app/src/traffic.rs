//! City life: NPC cars that drive the street grid and pedestrians that wander,
//! spawned only while the player is near the city and frozen/despawned when far
//! away, so they cost nothing during a launch or out on the road.
//!
//! The cars run on the same lane grid the `rocket::city` mesh is built from
//! (square `NX x NX` blocks at `SPAN` spacing). A car drives along one axis on a
//! lane line and, at each intersection, either continues or turns onto the
//! crossing street, so traffic stays on the roads without any pathfinding. The
//! whole system self-gates on distance to the city centre with hysteresis.

use crate::rocket::{self, Mesh};
use glam::DVec3;
use std::f32::consts::{FRAC_PI_2, PI};

// Must match `rocket::city`: 7x7 blocks, 60 m block-to-block spacing.
const NX: i32 = 7;
const SPAN: f64 = 60.0;
const HALF: f64 = NX as f64 * SPAN * 0.5; // 210 m, half the grid extent

// Spawn when the player comes within ACTIVATE of the centre; despawn past
// DEACTIVATE (hysteresis avoids thrashing at the boundary).
const ACTIVATE: f64 = HALF + 230.0;
const DEACTIVATE: f64 = HALF + 360.0;

const N_CARS: usize = 24;
const N_PEDS: usize = 46;

/// Half-street lane offset: cars drive on the right of the centreline, so the two
/// directions occupy separate lanes within the ~14 m street instead of overlapping.
const LANE_OFF: f64 = 3.4;

// Collision avoidance. A car eases off when something is close ahead in its lane:
// it slows from SLOW_GAP and is stopped by STOP_GAP. Intersections are handled by
// yielding to whichever car is already closer to the crossing (deadlock-free).
const LOOKAHEAD: f64 = 13.0; // how far ahead a leader / the player is noticed
const SLOW_GAP: f64 = 11.0; // start braking
const STOP_GAP: f64 = 4.6; // fully stopped by here
const LANE_HALF: f64 = 2.6; // lateral half-width that counts as "in my lane"
const INT_BOX: f64 = 7.0; // a car this close to an intersection is "in" it
const INT_ENTER: f64 = 17.0; // start checking the intersection from here
const ACCEL: f64 = 7.0; // m/s^2 speeding back up
const BRAKE: f64 = 16.0; // m/s^2 slowing down

const CAR_COLORS: [[f32; 3]; 6] = [
    [0.85, 0.85, 0.88], // white
    [0.12, 0.13, 0.15], // black
    [0.20, 0.30, 0.55], // blue
    [0.55, 0.57, 0.60], // silver
    [0.86, 0.72, 0.20], // taxi yellow
    [0.45, 0.12, 0.12], // dark red
];

const SHIRTS: [[f32; 3]; 6] = [
    [0.75, 0.30, 0.28],
    [0.25, 0.55, 0.40],
    [0.30, 0.40, 0.70],
    [0.80, 0.70, 0.30],
    [0.55, 0.35, 0.65],
    [0.85, 0.85, 0.88],
];

/// One NPC car, constrained to the lane grid. It travels along `axis` (0 = X,
/// 1 = Z) in direction `sign`, on the lane line `line` of the perpendicular axis.
pub struct Car {
    pub axis: u8,
    pub line: i32,
    pub along: f64,
    pub sign: f64,
    pub speed: f64,      // current speed (modulated by traffic ahead)
    pub cruise: f64,     // free-flowing target speed
    pub color: usize,
}

impl Car {
    /// World position (local launch-tangent metres, ground at y=0). Cars keep to
    /// the right of the centreline so the two directions form separate lanes.
    pub fn pos(&self, center: DVec3) -> DVec3 {
        let fixed = line_coord(if self.axis == 0 { center.z } else { center.x }, self.line);
        // right-hand offset: travel +X -> right is -Z; travel +Z -> right is +X.
        if self.axis == 0 {
            DVec3::new(self.along, 0.0, fixed - self.sign * LANE_OFF)
        } else {
            DVec3::new(fixed + self.sign * LANE_OFF, 0.0, self.along)
        }
    }
    /// The next intersection centre ahead and the forward distance to it.
    fn next_intersection(&self, center: DVec3) -> (DVec3, f64) {
        let travel_c = if self.axis == 0 { center.x } else { center.z };
        let base = travel_c - HALF;
        let u = (self.along - base) / SPAN;
        let k = if self.sign > 0.0 { u.floor() + 1.0 } else { u.ceil() - 1.0 };
        let next = base + k * SPAN;
        let di = (next - self.along) * self.sign;
        let perp = line_coord(if self.axis == 0 { center.z } else { center.x }, self.line);
        let p = if self.axis == 0 {
            DVec3::new(next, 0.0, perp)
        } else {
            DVec3::new(perp, 0.0, next)
        };
        (p, di)
    }
    /// Heading angle (0 = +X), matching the mesh's +X forward axis.
    pub fn yaw(&self) -> f32 {
        match (self.axis, self.sign > 0.0) {
            (0, true) => 0.0,
            (0, false) => PI,
            (_, true) => FRAC_PI_2,
            (_, false) => -FRAC_PI_2,
        }
    }
}

/// One wandering pedestrian.
pub struct Ped {
    pub pos: DVec3,
    pub yaw: f32,
    pub speed: f32,
    pub phase: f32,
    pub turn_t: f32,
    pub shirt: usize,
}

pub struct Traffic {
    pub active: bool,
    pub cars: Vec<Car>,
    pub peds: Vec<Ped>,
    /// One prebuilt car mesh per colour (cars don't animate, so reuse them).
    pub car_meshes: Vec<Mesh>,
    center: DVec3,
    rng: u64,
}

fn line_coord(center_c: f64, k: i32) -> f64 {
    center_c - HALF + k as f64 * SPAN
}

impl Traffic {
    pub fn new(center: DVec3) -> Self {
        let car_meshes = CAR_COLORS.iter().map(|&c| rocket::car(c)).collect();
        Traffic { active: false, cars: Vec::new(), peds: Vec::new(), car_meshes, center, rng: 0x9e37_79b9_7f4a_7c15 }
    }

    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn randf(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn randi(&mut self, n: i32) -> i32 {
        (self.next() % n.max(1) as u64) as i32
    }

    /// Update the crowd. `player` is the player's local position (drives the
    /// near/far gating); `player_car` is the player's car position when driving,
    /// so NPCs brake for it. The system activates near the city and despawns far.
    pub fn update(&mut self, player: DVec3, player_car: Option<DVec3>, dt: f32) {
        let dx = player.x - self.center.x;
        let dz = player.z - self.center.z;
        let dist = (dx * dx + dz * dz).sqrt();
        if !self.active {
            if dist < ACTIVATE {
                self.spawn();
                self.active = true;
            } else {
                return;
            }
        } else if dist > DEACTIVATE {
            // far away: despawn entirely (no sim, no render cost)
            self.cars.clear();
            self.peds.clear();
            self.active = false;
            return;
        }
        self.sim(dt, player_car);
    }

    fn spawn(&mut self) {
        self.cars.clear();
        self.peds.clear();
        for _ in 0..N_CARS {
            let axis = self.randi(2) as u8;
            let line = 1 + self.randi(NX - 1); // interior lane line
            let sign = if self.randf() < 0.5 { 1.0 } else { -1.0 };
            let travel_center = if axis == 0 { self.center.x } else { self.center.z };
            let along = travel_center - HALF + self.randf() * (NX as f64 * SPAN);
            let cruise = 9.0 + self.randf() * 9.0;
            let color = self.randi(CAR_COLORS.len() as i32) as usize;
            self.cars.push(Car { axis, line, along, sign, speed: cruise, cruise, color });
        }
        for _ in 0..N_PEDS {
            let px = self.center.x - HALF + self.randf() * (2.0 * HALF);
            let pz = self.center.z - HALF + self.randf() * (2.0 * HALF);
            let yaw = (self.randf() as f32) * std::f32::consts::TAU;
            let speed = 1.1 + self.randf() as f32 * 0.8;
            let turn_t = 1.0 + self.randf() as f32 * 3.0;
            let shirt = self.randi(SHIRTS.len() as i32) as usize;
            let phase = self.randf() as f32 * 6.0;
            self.peds.push(Ped { pos: DVec3::new(px, 0.0, pz), yaw, speed, phase, turn_t, shirt });
        }
    }

    fn sim(&mut self, dt: f32, player_car: Option<DVec3>) {
        let dtf = dt as f64;
        let center = self.center;
        let n = self.cars.len();
        // Snapshot every car's lane state for this frame's obstacle queries.
        let snap: Vec<(DVec3, u8, f64, f64)> =
            self.cars.iter().map(|c| (c.pos(center), c.axis, c.along, c.sign)).collect();

        // cars: pick a speed from the traffic ahead, then advance + turn.
        for i in 0..n {
            let (axis, sign, line) = {
                let c = &self.cars[i];
                (c.axis, c.sign, c.line)
            };
            let alongi = self.cars[i].along;
            let pi = snap[i].0;
            let peri = if axis == 0 { pi.z } else { pi.x }; // my lane's cross coord

            // nearest blocking distance ahead (free road = LOOKAHEAD).
            let mut gap = LOOKAHEAD;
            // same-street, same-lane leader (rear-end avoidance). Opposing traffic
            // sits in the other lane (offset), so its lateral gap excludes it.
            for j in 0..n {
                if j == i || snap[j].1 != axis {
                    continue;
                }
                let pj = snap[j].0;
                let o_along = if axis == 0 { pj.x } else { pj.z };
                let o_perp = if axis == 0 { pj.z } else { pj.x };
                let fwd = (o_along - alongi) * sign;
                if fwd > 0.0 && fwd < gap && (o_perp - peri).abs() < LANE_HALF {
                    gap = fwd;
                }
            }
            // the player's car: brake if it is ahead in my lane, and hard-stop if
            // it is right on top of me from any direction.
            if let Some(pc) = player_car {
                let o_along = if axis == 0 { pc.x } else { pc.z };
                let o_perp = if axis == 0 { pc.z } else { pc.x };
                let fwd = (o_along - alongi) * sign;
                if fwd > 0.0 && fwd < gap && (o_perp - peri).abs() < LANE_HALF + 1.2 {
                    gap = fwd;
                }
                if (pc - pi).length() < 5.5 {
                    gap = gap.min(STOP_GAP * 0.7);
                }
            }
            // intersection yield: approaching a crossing, give way to any cross car
            // already closer to it (closer car has priority -> deadlock-free).
            let (xc, di) = self.cars[i].next_intersection(center);
            if di < INT_ENTER {
                for j in 0..n {
                    if j == i || snap[j].1 == axis {
                        continue;
                    }
                    let dj = (snap[j].0 - xc).length();
                    if dj < INT_BOX && dj < di {
                        gap = gap.min(di);
                    }
                }
            }

            // ease the speed toward what the gap allows.
            let cruise = self.cars[i].cruise;
            let target = if gap <= STOP_GAP {
                0.0
            } else if gap < SLOW_GAP {
                cruise * (gap - STOP_GAP) / (SLOW_GAP - STOP_GAP)
            } else {
                cruise
            };
            let cur = self.cars[i].speed;
            let speed = if target > cur {
                (cur + ACCEL * dtf).min(target)
            } else {
                (cur - BRAKE * dtf).max(target).max(0.0)
            };
            self.cars[i].speed = speed;

            let travel_center = if axis == 0 { center.x } else { center.z };
            let fixed_center = if axis == 0 { center.z } else { center.x };
            let base = travel_center - HALF;
            let prev = alongi;
            let along = prev + sign * speed * dtf;
            let kprev = ((prev - base) / SPAN).floor() as i32;
            let knew = ((along - base) / SPAN).floor() as i32;
            if knew != kprev {
                // index of the lane line we just crossed (the intersection)
                let kc = if sign > 0.0 { kprev + 1 } else { kprev };
                let at_edge = kc <= 0 || kc >= NX;
                let turn = at_edge || self.randf() < 0.45;
                if turn {
                    // snap to the intersection and turn onto the crossing street
                    let new_along = line_coord(fixed_center, line); // old fixed coord
                    let kalong = line; // its index on the (now travel) axis
                    let mut ns = if self.randf() < 0.5 { 1.0 } else { -1.0 };
                    if kalong <= 0 {
                        ns = 1.0;
                    } else if kalong >= NX {
                        ns = -1.0;
                    }
                    let c = &mut self.cars[i];
                    c.axis = 1 - axis;
                    c.line = kc; // turning onto the line we crossed
                    c.along = new_along;
                    c.sign = ns;
                    continue;
                }
            }
            self.cars[i].along = along;
        }
        // pedestrians: wander, steering back inside the city box at the edges.
        let pn = self.peds.len();
        for i in 0..pn {
            let mut turned = false;
            {
                let p = &mut self.peds[i];
                p.turn_t -= dt;
                if p.turn_t <= 0.0 {
                    turned = true;
                }
            }
            if turned {
                let d = (self.randf() as f32 - 0.5) * 1.4;
                let t = 1.5 + self.randf() as f32 * 3.0;
                let p = &mut self.peds[i];
                p.yaw += d;
                p.turn_t = t;
            }
            let p = &mut self.peds[i];
            p.phase += p.speed * 1.7 * dt;
            let heading = DVec3::new(p.yaw.cos() as f64, 0.0, p.yaw.sin() as f64);
            p.pos += heading * (p.speed as f64 * dtf);
            p.pos.y = 0.0;
            // keep inside the city: steer toward the centre past the edge.
            if (p.pos.x - center.x).abs() > HALF + 12.0 || (p.pos.z - center.z).abs() > HALF + 12.0 {
                let to_c = center - p.pos;
                p.yaw = (to_c.z as f32).atan2(to_c.x as f32);
            }
        }
    }

    pub fn center(&self) -> DVec3 {
        self.center
    }
    pub fn ped_shirt(idx: usize) -> [f32; 3] {
        SHIRTS[idx % SHIRTS.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The crowd must only exist while the player is near the city: asleep when
    /// far (no sim/render cost), populated when near, and despawned again once
    /// the player drives away.
    #[test]
    fn spawns_near_and_despawns_far() {
        let center = DVec3::new(1600.0, 0.0, 1800.0);
        let far = DVec3::ZERO; // ~2.4 km from the city centre
        let mut t = Traffic::new(center);

        t.update(far, None, 0.1);
        assert!(!t.active && t.cars.is_empty() && t.peds.is_empty(), "should stay asleep when far");

        t.update(center, None, 0.1);
        assert!(t.active && !t.cars.is_empty() && !t.peds.is_empty(), "should wake near the city");

        // Simulate a while near the city: cars stay on the grid (finite coords).
        for _ in 0..400 {
            t.update(center, None, 0.05);
        }
        assert!(t.cars.iter().all(|c| c.along.is_finite()), "cars must stay finite/on-grid");
        let in_box = t.peds.iter().all(|p| {
            (p.pos.x - center.x).abs() < HALF + 40.0 && (p.pos.z - center.z).abs() < HALF + 40.0
        });
        assert!(in_box, "pedestrians must stay within the city bounds");
        // traffic must keep flowing: collision avoidance must not gridlock the
        // whole grid into a permanent standstill.
        let moving = t.cars.iter().filter(|c| c.speed > 2.0).count();
        assert!(moving >= t.cars.len() / 3, "traffic gridlocked: only {moving} of {} moving", t.cars.len());

        t.update(far, None, 0.1);
        assert!(!t.active && t.cars.is_empty() && t.peds.is_empty(), "should despawn when far again");
    }
}
