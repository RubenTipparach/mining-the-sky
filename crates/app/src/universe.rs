//! Procedural Kepler-47 system: the real binary plus a fictional full roster of
//! undiscovered bodies. Deterministic from a seed. Distances/sizes are schematic
//! (compressed, in Mm) so the whole system stays f32-precise and navigable in
//! the orbital map. Technical designations now; proper names later.

use glam::Vec3;

/// Binary barycentre (Mm). The home world sits at the local origin (so the
/// launch sim frame is untouched) on the 360 Mm orbit, hence the barycentre is
/// one home-orbit-radius away.
pub const BARY: Vec3 = Vec3::new(-360.0, 0.0, 0.0);

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
pub struct Body {
    pub name: String,
    pub kind: Kind,
    pub pos: Vec3,
    pub radius: f32,
    pub color: [f32; 3],
    /// Orbit ring center + radius (Mm); `orbit_r == 0` means no ring.
    pub orbit_center: Vec3,
    pub orbit_r: f32,
    pub parent: Option<usize>,
    pub is_home: bool,
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
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
}

const TAU: f32 = std::f32::consts::TAU;

/// Generate the full system. `home_radius_mm` sizes the home world; the landable
/// moon (used by the flight sim) is injected as home's first moon so the map and
/// gameplay agree.
pub fn generate(
    seed: u64,
    home_radius_mm: f32,
    landable_moon_pos: Vec3,
    landable_moon_radius: f32,
) -> Universe {
    let mut rng = Rng::new(seed);
    let mut bodies: Vec<Body> = Vec::new();

    // --- binary stars ---
    bodies.push(Body {
        name: "KEPLER-47 A".into(),
        kind: Kind::StarA,
        pos: BARY + Vec3::new(0.0, 0.0, 26.0),
        radius: 16.0,
        color: [1.6, 1.3, 0.8],
        orbit_center: BARY,
        orbit_r: 0.0,
        parent: None,
        is_home: false,
    });
    bodies.push(Body {
        name: "KEPLER-47 B".into(),
        kind: Kind::StarB,
        pos: BARY - Vec3::new(0.0, 0.0, 26.0),
        radius: 9.0,
        color: [1.3, 0.5, 0.35],
        orbit_center: BARY,
        orbit_r: 0.0,
        parent: None,
        is_home: false,
    });

    // --- 13 planets (3 "known" + 10 undiscovered), home at 360 Mm ---
    let orbit_r = [
        110.0f32, 175.0, 250.0, 360.0, 450.0, 560.0, 690.0, 820.0, 980.0, 1150.0, 1330.0, 1520.0,
        1720.0,
    ];
    let home_idx = 3usize; // the 360 Mm orbit
    let mut planet_indices: Vec<usize> = Vec::new();
    for (i, &orad) in orbit_r.iter().enumerate() {
        let is_home = i == home_idx;
        let angle = if is_home { 0.0 } else { rng.range(0.0, TAU) };
        let pos = if is_home {
            Vec3::ZERO
        } else {
            BARY + Vec3::new(angle.cos(), 0.0, angle.sin()) * orad
        };
        // type by orbit: inner rocky, middle gas giants, outer ice giants
        let (radius, color) = if is_home {
            (home_radius_mm, [0.30, 0.55, 0.45])
        } else if orad < 300.0 {
            (rng.range(3.0, 6.0), [rng.range(0.45, 0.7), rng.range(0.4, 0.55), rng.range(0.3, 0.45)])
        } else if orad < 760.0 {
            (rng.range(12.0, 18.0), [rng.range(0.7, 0.9), rng.range(0.6, 0.78), rng.range(0.45, 0.62)])
        } else {
            (rng.range(9.0, 13.0), [rng.range(0.5, 0.65), rng.range(0.7, 0.82), rng.range(0.82, 0.95)])
        };
        let idx = bodies.len();
        planet_indices.push(idx);
        bodies.push(Body {
            name: format!("K47-P{:02}", i + 1),
            kind: Kind::Planet,
            pos,
            radius,
            color,
            orbit_center: BARY,
            orbit_r: orad,
            parent: None,
            is_home,
        });
    }

    // --- 25 moons distributed by planet size ---
    // quota per planet by display radius, then trimmed/topped to exactly 25
    let mut quota: Vec<u32> = planet_indices
        .iter()
        .map(|&pi| {
            let r = bodies[pi].radius;
            if r > 12.0 {
                4
            } else if r > 7.0 {
                2
            } else {
                1
            }
        })
        .collect();
    let mut total: i32 = quota.iter().map(|q| *q as i32).sum();
    let qlen = quota.len();
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
    for (pi_n, &pi) in planet_indices.iter().enumerate() {
        let (ppos, prad) = (bodies[pi].pos, bodies[pi].radius);
        let pname = bodies[pi].name.clone();
        for m in 0..quota[pi_n] {
            let (mpos, mrad, mr) = if pi == bodies.iter().position(|b| b.is_home).unwrap()
                && m == 0
            {
                // home's first moon is the landable gameplay moon
                (landable_moon_pos, landable_moon_radius, landable_moon_pos.length())
            } else {
                let mr = prad * 2.2 + m as f32 * prad * 0.9 + rng.range(0.0, prad);
                let ang = rng.range(0.0, TAU);
                (ppos + Vec3::new(ang.cos(), 0.0, ang.sin()) * mr, rng.range(0.4, 1.3), mr)
            };
            bodies.push(Body {
                name: format!("{}-M{}", pname, m + 1),
                kind: Kind::Moon,
                pos: mpos,
                radius: mrad,
                color: [0.55, 0.55, 0.58],
                orbit_center: ppos,
                orbit_r: mr,
                parent: Some(pi),
                is_home: false,
            });
        }
    }

    // --- asteroid belt: 15 major + 50 minor between the 450 and 560 orbits ---
    let belt = 505.0f32;
    for j in 0..15 {
        let ar = belt + rng.range(-40.0, 40.0);
        let ang = rng.range(0.0, TAU);
        bodies.push(Body {
            name: format!("K47-A{:02}", j + 1),
            kind: Kind::AsteroidMajor,
            pos: BARY + Vec3::new(ang.cos(), 0.0, ang.sin()) * ar,
            radius: rng.range(0.6, 1.6),
            color: [0.55, 0.50, 0.44],
            orbit_center: BARY,
            orbit_r: ar,
            parent: None,
            is_home: false,
        });
    }
    for j in 0..50 {
        let ar = belt + rng.range(-55.0, 55.0);
        let ang = rng.range(0.0, TAU);
        bodies.push(Body {
            name: format!("K47-a{:02}", j + 1),
            kind: Kind::AsteroidMinor,
            pos: BARY + Vec3::new(ang.cos(), 0.0, ang.sin()) * ar,
            radius: rng.range(0.2, 0.5),
            color: [0.5, 0.46, 0.42],
            orbit_center: BARY,
            orbit_r: ar,
            parent: None,
            is_home: false,
        });
    }

    // --- 20 comets on eccentric orbits ---
    for j in 0..20 {
        let a = rng.range(700.0, 1700.0);
        let e = rng.range(0.6, 0.9);
        let nu = rng.range(0.0, TAU); // true anomaly now
        let om = rng.range(0.0, TAU); // orbit orientation
        let r = a * (1.0 - e * e) / (1.0 + e * nu.cos());
        let th = nu + om;
        bodies.push(Body {
            name: format!("K47-C{:02}", j + 1),
            kind: Kind::Comet,
            pos: BARY + Vec3::new(th.cos(), 0.0, th.sin()) * r,
            radius: 0.4,
            color: [0.7, 0.85, 0.9],
            orbit_center: BARY,
            orbit_r: 0.0, // eccentric; no simple ring
            parent: None,
            is_home: false,
        });
    }

    Universe { bodies }
}

impl Universe {
    pub fn count(&self, k: Kind) -> usize {
        self.bodies.iter().filter(|b| b.kind == k).count()
    }

    /// A grouped markdown catalog of every body.
    pub fn catalog_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("# Kepler-47 system catalog (provisional designations)\n\n");
        s.push_str(&format!(
            "Generated: {} bodies - {} stars, {} planets, {} moons, {} major + {} minor asteroids, {} comets.\n\n",
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
            s.push_str(&format!("- {} - {}\n", b.name, b.kind.label()));
        }

        s.push_str("\n## Planets (and their moons)\n\n");
        for (i, b) in self.bodies.iter().enumerate() {
            if b.kind != Kind::Planet {
                continue;
            }
            let home = if b.is_home { " - HOME (launch world)" } else { "" };
            let orbit_au = b.orbit_r / 360.0; // home orbit ~ 1 reference unit
            s.push_str(&format!(
                "- {} - planet, orbit {:.2} (rel), radius {:.1} Mm{}\n",
                b.name, orbit_au, b.radius, home
            ));
            for m in self.bodies.iter().filter(|m| m.parent == Some(i)) {
                s.push_str(&format!("    - {} - moon, radius {:.2} Mm\n", m.name, m.radius));
            }
        }

        s.push_str("\n## Asteroids\n\n");
        for b in self.bodies.iter().filter(|b| b.kind == Kind::AsteroidMajor) {
            s.push_str(&format!("- {} - major asteroid, orbit {:.0} Mm\n", b.name, b.orbit_r));
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

        s.push_str("\n## Comets\n\n");
        for b in self.bodies.iter().filter(|b| b.kind == Kind::Comet) {
            s.push_str(&format!("- {} - comet (eccentric)\n", b.name));
        }
        s
    }
}
