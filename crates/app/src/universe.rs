//! Procedural Kepler-47 system: the real binary plus a fictional roster of
//! undiscovered bodies, at full (realistic-estimate) scale. Every body is on
//! analytic Kepler rails - its position is a pure function of sim time - so the
//! map can run at 1x..10000x time deterministically without stability worries.
//!
//! Units: distances in Mm (1000 km); the barycentre is the frame origin. The
//! map renders camera-relative (floating origin) so f32 survives at AU scale.

use glam::{DQuat, DVec3};
use std::f64::consts::{PI, TAU};

/// Mm per astronomical unit.
pub const AU: f64 = 149_597.870_7;
/// Gravitational parameter of the binary (~1.40 Msun) in Mm^3/s^2.
const MU_BINARY: f64 = 185.8;
const DAY: f64 = 86_400.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    StarA,
    StarB,
    Planet,
    Moon,
    AsteroidMajor,
    AsteroidMinor,
    Comet,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::StarA => "star (primary)",
            Kind::StarB => "star (secondary)",
            Kind::Planet => "planet",
            Kind::Moon => "moon",
            Kind::AsteroidMajor => "major asteroid",
            Kind::AsteroidMinor => "minor asteroid",
            Kind::Comet => "comet",
        }
    }
}

#[derive(Clone)]
pub struct Orbit {
    pub a: f64,      // semi-major axis (Mm)
    pub e: f64,      // eccentricity
    pub period: f64, // seconds
    pub m0: f64,     // mean anomaly at epoch (rad)
    pub incl: f64,   // inclination (rad)
    pub node: f64,   // longitude of ascending node (rad)
    pub argp: f64,   // argument of periapsis (rad)
    /// Parent body index; `None` orbits the barycentre (origin).
    pub parent: Option<usize>,
}

#[derive(Clone)]
pub struct Body {
    pub name: String,
    pub kind: Kind,
    pub radius: f64, // display radius (Mm)
    pub color: [f32; 3],
    pub is_home: bool,
    pub orbit: Orbit,
}

pub struct Universe {
    pub bodies: Vec<Body>,
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }
}

fn period_for(a: f64, mu: f64) -> f64 {
    TAU * (a * a * a / mu).sqrt()
}

fn solve_kepler(m: f64, e: f64) -> f64 {
    let m = m.rem_euclid(TAU);
    let mut ea = if e < 0.8 { m } else { PI };
    for _ in 0..10 {
        ea -= (ea - e * ea.sin() - m) / (1.0 - e * ea.cos());
    }
    ea
}

/// Point on the orbit (relative to its centre) at eccentric anomaly `ea`.
fn ellipse_point(o: &Orbit, ea: f64) -> DVec3 {
    let nu = 2.0 * (((1.0 + o.e).sqrt() * (ea * 0.5).sin()).atan2((1.0 - o.e).sqrt() * (ea * 0.5).cos()));
    let r = o.a * (1.0 - o.e * ea.cos());
    let u = o.argp + nu;
    // in the XZ plane, then tilt by inclination about the node axis
    let base = DVec3::new(r * u.cos(), 0.0, r * u.sin());
    let axis = DVec3::new(o.node.cos(), 0.0, o.node.sin());
    DQuat::from_axis_angle(axis, o.incl) * base
}

/// Position of an orbit (relative to its centre) at time `t`, Y-up Mm.
fn orbit_offset(o: &Orbit, t: f64) -> DVec3 {
    let m = o.m0 + TAU * t / o.period;
    ellipse_point(o, solve_kepler(m, o.e))
}

impl Universe {
    /// World position (Mm, barycentre at origin) of body `i` at time `t`.
    pub fn position(&self, i: usize, t: f64) -> DVec3 {
        let o = &self.bodies[i].orbit;
        let center = match o.parent {
            Some(p) => self.position(p, t),
            None => DVec3::ZERO,
        };
        center + orbit_offset(o, t)
    }

    /// World position of the centre body `i` orbits around, at time `t`.
    pub fn orbit_center(&self, i: usize, t: f64) -> DVec3 {
        match self.bodies[i].orbit.parent {
            Some(p) => self.position(p, t),
            None => DVec3::ZERO,
        }
    }

    /// A point on body `i`'s orbit ring (world Mm), `frac` in 0..1.
    pub fn ring_point(&self, i: usize, frac: f64, t: f64) -> DVec3 {
        let o = &self.bodies[i].orbit;
        self.orbit_center(i, t) + ellipse_point(o, frac * TAU)
    }

    pub fn count(&self, k: Kind) -> usize {
        self.bodies.iter().filter(|b| b.kind == k).count()
    }
}

/// Generate the full system. `home_radius_mm` sizes the home world.
pub fn generate(seed: u64, home_radius_mm: f32) -> Universe {
    let mut rng = Rng::new(seed);
    let mut bodies: Vec<Body> = Vec::new();

    let circ = |rng: &mut Rng| Orbit {
        a: 0.0,
        e: 0.0,
        period: 1.0,
        m0: 0.0,
        incl: 0.0,
        node: rng.range(0.0, TAU),
        argp: 0.0,
        parent: None,
    };

    // --- binary stars (Kepler-47 A + B), period ~7.45 d ---
    let a_bin = 0.0836 * AU; // separation giving ~7.45 d
    let p_bin = period_for(a_bin, MU_BINARY);
    bodies.push(Body {
        name: "KEPLER-47 A".into(),
        kind: Kind::StarA,
        radius: 668.0, // ~0.96 Rsun
        color: [1.6, 1.35, 0.85],
        is_home: false,
        orbit: Orbit { a: a_bin * 0.258, e: 0.02, period: p_bin, m0: 0.0, incl: 0.0, node: 0.0, argp: 0.0, parent: None },
    });
    bodies.push(Body {
        name: "KEPLER-47 B".into(),
        kind: Kind::StarB,
        radius: 244.0, // ~0.35 Rsun
        color: [1.4, 0.55, 0.4],
        is_home: false,
        orbit: Orbit { a: a_bin * 0.742, e: 0.02, period: p_bin, m0: PI, incl: 0.0, node: 0.0, argp: 0.0, parent: None },
    });

    // --- 13 planets (3 known: ~0.29, 0.42, 0.96 AU; 10 estimated outward) ---
    let semi_au = [
        0.29, 0.42, 0.96, 1.5, 2.2, 3.1, 4.4, 6.2, 8.8, 12.5, 18.0, 27.0, 40.0,
    ];
    let home_idx_in_planets = 2usize; // the 0.96 AU world (habitable zone)
    let mut planet_body_idx: Vec<usize> = Vec::new();
    for (pi, &au) in semi_au.iter().enumerate() {
        let a = au * AU;
        let is_home = pi == home_idx_in_planets;
        let (radius, color) = if is_home {
            (home_radius_mm as f64, [0.30, 0.55, 0.45])
        } else if au < 0.7 {
            (rng.range(3.0, 7.0), [rng.range(0.45, 0.7) as f32, rng.range(0.4, 0.55) as f32, rng.range(0.3, 0.45) as f32]) // rocky
        } else if au < 6.0 {
            (rng.range(28.0, 72.0), [rng.range(0.7, 0.9) as f32, rng.range(0.6, 0.78) as f32, rng.range(0.45, 0.62) as f32]) // gas giant
        } else {
            (rng.range(18.0, 30.0), [rng.range(0.5, 0.65) as f32, rng.range(0.7, 0.82) as f32, rng.range(0.82, 0.95) as f32]) // ice giant
        };
        let mut o = circ(&mut rng);
        o.a = a;
        o.e = rng.range(0.01, 0.08);
        o.period = period_for(a, MU_BINARY);
        o.m0 = rng.range(0.0, TAU);
        o.incl = rng.range(-0.05, 0.05);
        o.argp = rng.range(0.0, TAU);
        planet_body_idx.push(bodies.len());
        bodies.push(Body {
            name: format!("K47-P{:02}", pi + 1),
            kind: Kind::Planet,
            radius,
            color,
            is_home,
            orbit: o,
        });
    }

    // --- 25 moons distributed by planet size ---
    let mut quota: Vec<u32> = planet_body_idx
        .iter()
        .map(|&pi| {
            let r = bodies[pi].radius;
            if r > 40.0 {
                4
            } else if r > 18.0 {
                3
            } else if r > 7.0 {
                1
            } else {
                0
            }
        })
        .collect();
    let qlen = quota.len();
    let mut total: i32 = quota.iter().map(|q| *q as i32).sum();
    let mut k = 0usize;
    while total != 25 {
        let j = k % qlen;
        if total < 25 {
            quota[j] += 1;
            total += 1;
        } else if quota[j] > 0 {
            quota[j] -= 1;
            total -= 1;
        }
        k += 1;
    }
    for (pn, &pi) in planet_body_idx.iter().enumerate() {
        let prad = bodies[pi].radius;
        let pname = bodies[pi].name.clone();
        for m in 0..quota[pn] {
            let a = prad * rng.range(8.0, 30.0) + m as f64 * prad * 6.0;
            bodies.push(Body {
                name: format!("{}-M{}", pname, m + 1),
                kind: Kind::Moon,
                radius: rng.range(0.4, 2.0),
                color: [0.55, 0.55, 0.58],
                is_home: false,
                orbit: Orbit {
                    a,
                    e: rng.range(0.0, 0.04),
                    period: rng.range(2.0, 40.0) * DAY,
                    m0: rng.range(0.0, TAU),
                    incl: rng.range(-0.1, 0.1),
                    node: rng.range(0.0, TAU),
                    argp: rng.range(0.0, TAU),
                    parent: Some(pi),
                },
            });
        }
    }

    // --- asteroid belt: 15 major + 50 minor near 1.8 AU ---
    let belt = 1.8 * AU;
    let mut add_ast = |rng: &mut Rng, name: String, kind: Kind, rad: f64, spread: f64, bodies: &mut Vec<Body>| {
        let a = belt + rng.range(-spread, spread);
        bodies.push(Body {
            name,
            kind,
            radius: rad,
            color: [0.52, 0.48, 0.43],
            is_home: false,
            orbit: Orbit {
                a,
                e: rng.range(0.0, 0.12),
                period: period_for(a, MU_BINARY),
                m0: rng.range(0.0, TAU),
                incl: rng.range(-0.12, 0.12),
                node: rng.range(0.0, TAU),
                argp: rng.range(0.0, TAU),
                parent: None,
            },
        });
    };
    for j in 0..15 {
        let rad = rng.range(0.2, 0.6);
        add_ast(&mut rng, format!("K47-A{:02}", j + 1), Kind::AsteroidMajor, rad, 0.15 * AU, &mut bodies);
    }
    for j in 0..50 {
        let rad = rng.range(0.05, 0.2);
        add_ast(&mut rng, format!("K47-a{:02}", j + 1), Kind::AsteroidMinor, rad, 0.22 * AU, &mut bodies);
    }

    // --- 20 comets on eccentric, inclined orbits ---
    for j in 0..20 {
        let a = rng.range(30.0, 90.0) * AU;
        bodies.push(Body {
            name: format!("K47-C{:02}", j + 1),
            kind: Kind::Comet,
            radius: rng.range(0.02, 0.06),
            color: [0.7, 0.85, 0.9],
            is_home: false,
            orbit: Orbit {
                a,
                e: rng.range(0.6, 0.92),
                period: period_for(a, MU_BINARY),
                m0: rng.range(0.0, TAU),
                incl: rng.range(-0.7, 0.7),
                node: rng.range(0.0, TAU),
                argp: rng.range(0.0, TAU),
                parent: None,
            },
        });
    }

    Universe { bodies }
}

impl Universe {
    /// Index of the home world.
    pub fn home_index(&self) -> usize {
        self.bodies.iter().position(|b| b.is_home).unwrap_or(0)
    }

    /// A grouped markdown catalog of every body.
    pub fn catalog_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("# Kepler-47 system catalog (provisional designations)\n\n");
        s.push_str(&format!(
            "Generated: {} bodies - {} stars, {} planets, {} moons, {} major + {} minor asteroids, {} comets. Full-scale orbits; positions are on analytic Kepler rails (a pure function of sim time).\n\n",
            self.bodies.len(),
            self.count(Kind::StarA) + self.count(Kind::StarB),
            self.count(Kind::Planet),
            self.count(Kind::Moon),
            self.count(Kind::AsteroidMajor),
            self.count(Kind::AsteroidMinor),
            self.count(Kind::Comet),
        ));

        s.push_str("## Stars\n\n");
        for b in self.bodies.iter().filter(|b| matches!(b.kind, Kind::StarA | Kind::StarB)) {
            s.push_str(&format!("- {} - {}, R {:.0} Mm\n", b.name, b.kind.label(), b.radius));
        }

        s.push_str("\n## Planets (and their moons)\n\n");
        for (i, b) in self.bodies.iter().enumerate() {
            if b.kind != Kind::Planet {
                continue;
            }
            let home = if b.is_home { " - HOME (launch world)" } else { "" };
            s.push_str(&format!(
                "- {} - planet, a {:.2} AU, period {:.0} d, R {:.1} Mm{}\n",
                b.name,
                b.orbit.a / AU,
                b.orbit.period / DAY,
                b.radius,
                home
            ));
            for m in self.bodies.iter().enumerate().filter(|(_, m)| m.orbit.parent == Some(i)) {
                s.push_str(&format!(
                    "    - {} - moon, a {:.0} Mm, period {:.1} d, R {:.2} Mm\n",
                    m.1.name,
                    m.1.orbit.a,
                    m.1.orbit.period / DAY,
                    m.1.radius
                ));
            }
        }

        s.push_str("\n## Asteroids (belt ~1.8 AU)\n\n");
        for b in self.bodies.iter().filter(|b| b.kind == Kind::AsteroidMajor) {
            s.push_str(&format!("- {} - major asteroid, a {:.2} AU\n", b.name, b.orbit.a / AU));
        }
        s.push_str("\nMinor asteroids: ");
        let minors: Vec<&str> = self
            .bodies
            .iter()
            .filter(|b| b.kind == Kind::AsteroidMinor)
            .map(|b| b.name.as_str())
            .collect();
        s.push_str(&minors.join(", "));
        s.push('\n');

        s.push_str("\n## Comets (eccentric, 30-90 AU)\n\n");
        for b in self.bodies.iter().filter(|b| b.kind == Kind::Comet) {
            s.push_str(&format!("- {} - comet, a {:.0} AU, e {:.2}\n", b.name, b.orbit.a / AU, b.orbit.e));
        }
        s
    }
}
