//! Procedural 3D geometry for the rocket view: a staged launch vehicle built
//! from the `sim` vehicle definition, standing on a launch pad over a ground
//! plane. Flat-shaded triangle soup (per-face normals), drawn non-indexed by
//! the mesh pipeline.

use glam::Vec3;
use sim::vehicle::Vehicle;
use std::f32::consts::TAU;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

#[derive(Default)]
pub struct Mesh {
    pub verts: Vec<MeshVertex>,
}

impl Mesh {
    fn tri(&mut self, a: Vec3, b: Vec3, c: Vec3, n: Vec3, col: [f32; 3]) {
        for p in [a, b, c] {
            self.verts.push(MeshVertex {
                pos: [p.x, p.y, p.z],
                normal: [n.x, n.y, n.z],
                color: col,
            });
        }
    }

    /// A triangle with independent per-vertex normals and colours (for smooth
    /// Gouraud-shaded terrain).
    fn tri3(&mut self, p: [Vec3; 3], n: [Vec3; 3], col: [[f32; 3]; 3]) {
        for i in 0..3 {
            self.verts.push(MeshVertex {
                pos: [p[i].x, p[i].y, p[i].z],
                normal: [n[i].x, n[i].y, n[i].z],
                color: col[i],
            });
        }
    }

    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, n: Vec3, col: [f32; 3]) {
        self.tri(a, b, c, n, col);
        self.tri(a, c, d, n, col);
    }

    /// A frustum (cone when r1==0, cylinder when r0==r1) about a vertical axis
    /// at (cx, cz), from y0 (radius r0) to y1 (radius r1).
    #[allow(clippy::too_many_arguments)]
    fn frustum(
        &mut self,
        cx: f32,
        cz: f32,
        y0: f32,
        y1: f32,
        r0: f32,
        r1: f32,
        sides: usize,
        col: [f32; 3],
        cap0: bool,
        cap1: bool,
    ) {
        let drdy = (r1 - r0) / (y1 - y0).abs().max(1e-3);
        for i in 0..sides {
            let a0 = i as f32 / sides as f32 * TAU;
            let a1 = (i + 1) as f32 / sides as f32 * TAU;
            let am = 0.5 * (a0 + a1);
            let p00 = Vec3::new(cx + r0 * a0.cos(), y0, cz + r0 * a0.sin());
            let p10 = Vec3::new(cx + r0 * a1.cos(), y0, cz + r0 * a1.sin());
            let p11 = Vec3::new(cx + r1 * a1.cos(), y1, cz + r1 * a1.sin());
            let p01 = Vec3::new(cx + r1 * a0.cos(), y1, cz + r1 * a0.sin());
            let n = Vec3::new(am.cos(), -drdy, am.sin()).normalize();
            self.quad(p00, p10, p11, p01, n, col);
            if cap0 {
                self.tri(Vec3::new(cx, y0, cz), p10, p00, Vec3::NEG_Y, col);
            }
            if cap1 {
                self.tri(Vec3::new(cx, y1, cz), p01, p11, Vec3::Y, col);
            }
        }
    }

    fn bx(&mut self, center: Vec3, he: Vec3, col: [f32; 3]) {
        let s = [-1.0f32, 1.0];
        // each axis as the face normal
        for (axis, n) in [
            (0, Vec3::X),
            (0, Vec3::NEG_X),
            (1, Vec3::Y),
            (1, Vec3::NEG_Y),
            (2, Vec3::Z),
            (2, Vec3::NEG_Z),
        ] {
            // build the 4 corners of this face
            let mut corners = [Vec3::ZERO; 4];
            let (u_axis, v_axis) = match axis {
                0 => (1usize, 2usize),
                1 => (0, 2),
                _ => (0, 1),
            };
            for (k, (su, sv)) in [(s[0], s[0]), (s[1], s[0]), (s[1], s[1]), (s[0], s[1])]
                .iter()
                .enumerate()
            {
                let mut p = center;
                let sign = if n[axis] > 0.0 { 1.0 } else { -1.0 };
                p[axis] += sign * he[axis];
                p[u_axis] += su * he[u_axis];
                p[v_axis] += sv * he[v_axis];
                corners[k] = p;
            }
            self.quad(corners[0], corners[1], corners[2], corners[3], n, col);
        }
    }

    /// A square-section strut between two points (for lander legs etc.).
    fn strut(&mut self, a: Vec3, b: Vec3, r: f32, col: [f32; 3]) {
        let d = b - a;
        let len = d.length();
        if len < 1e-4 {
            return;
        }
        let dir = d / len;
        let refv = if dir.y.abs() < 0.95 { Vec3::Y } else { Vec3::X };
        let u = dir.cross(refv).normalize() * r;
        let v = dir.cross(u.normalize()).normalize() * r;
        let ca = [a + u + v, a + u - v, a - u - v, a - u + v];
        let cb = [b + u + v, b + u - v, b - u - v, b - u + v];
        for i in 0..4 {
            let j = (i + 1) % 4;
            let n = (ca[j] - ca[i]).cross(cb[i] - ca[i]).normalize_or_zero();
            self.quad(ca[i], ca[j], cb[j], cb[i], n, col);
        }
        self.quad(ca[0], ca[1], ca[2], ca[3], -dir, col);
        self.quad(cb[3], cb[2], cb[1], cb[0], dir, col);
    }

    /// A hemispherical dome of `radius` and `height`, sitting on `(cx, y0, cz)`,
    /// built from latitude rings (outward normals). Good for habitat / tourist
    /// domes.
    #[allow(clippy::too_many_arguments)]
    fn dome(&mut self, cx: f32, cz: f32, y0: f32, radius: f32, height: f32, sides: usize, rings: usize, col: [f32; 3]) {
        use std::f32::consts::FRAC_PI_2;
        let center = Vec3::new(cx, y0, cz);
        for r in 0..rings {
            let t0 = r as f32 / rings as f32;
            let t1 = (r + 1) as f32 / rings as f32;
            let (ya, ra) = (y0 + height * (t0 * FRAC_PI_2).sin(), radius * (t0 * FRAC_PI_2).cos());
            let (yb, rb) = (y0 + height * (t1 * FRAC_PI_2).sin(), radius * (t1 * FRAC_PI_2).cos());
            for i in 0..sides {
                let a0 = i as f32 / sides as f32 * TAU;
                let a1 = (i + 1) as f32 / sides as f32 * TAU;
                let p00 = Vec3::new(cx + ra * a0.cos(), ya, cz + ra * a0.sin());
                let p10 = Vec3::new(cx + ra * a1.cos(), ya, cz + ra * a1.sin());
                let p11 = Vec3::new(cx + rb * a1.cos(), yb, cz + rb * a1.sin());
                let p01 = Vec3::new(cx + rb * a0.cos(), yb, cz + rb * a0.sin());
                if r == rings - 1 {
                    let apex = Vec3::new(cx, y0 + height, cz);
                    let n = (0.5 * (p00 + p10) - center).normalize_or_zero();
                    self.tri(p00, p10, apex, n, col);
                } else {
                    let nm = (0.25 * (p00 + p10 + p11 + p01) - center).normalize_or_zero();
                    self.quad(p00, p10, p11, p01, nm, col);
                }
            }
        }
    }

    /// A downward hemisphere (mirror of `dome`), for the bottom of a spherical
    /// tank. Apex hangs `height` below `(cx, y0, cz)`.
    #[allow(clippy::too_many_arguments)]
    fn dome_down(&mut self, cx: f32, cz: f32, y0: f32, radius: f32, height: f32, sides: usize, rings: usize, col: [f32; 3]) {
        use std::f32::consts::FRAC_PI_2;
        let center = Vec3::new(cx, y0, cz);
        for r in 0..rings {
            let t0 = r as f32 / rings as f32;
            let t1 = (r + 1) as f32 / rings as f32;
            let (ya, ra) = (y0 - height * (t0 * FRAC_PI_2).sin(), radius * (t0 * FRAC_PI_2).cos());
            let (yb, rb) = (y0 - height * (t1 * FRAC_PI_2).sin(), radius * (t1 * FRAC_PI_2).cos());
            for i in 0..sides {
                let a0 = i as f32 / sides as f32 * TAU;
                let a1 = (i + 1) as f32 / sides as f32 * TAU;
                let p00 = Vec3::new(cx + ra * a0.cos(), ya, cz + ra * a0.sin());
                let p10 = Vec3::new(cx + ra * a1.cos(), ya, cz + ra * a1.sin());
                let p11 = Vec3::new(cx + rb * a1.cos(), yb, cz + rb * a1.sin());
                let p01 = Vec3::new(cx + rb * a0.cos(), yb, cz + rb * a0.sin());
                if r == rings - 1 {
                    let apex = Vec3::new(cx, y0 - height, cz);
                    let n = (0.5 * (p00 + p10) - center).normalize_or_zero();
                    self.tri(p10, p00, apex, n, col);
                } else {
                    let nm = (0.25 * (p00 + p10 + p11 + p01) - center).normalize_or_zero();
                    self.quad(p01, p11, p10, p00, nm, col);
                }
            }
        }
    }
}

/// A 3D lunar descent module: a gold-foil descent stage with a big engine bell,
/// four splayed landing legs with footpads, and a small ascent cabin on top.
/// Built about its footpads at y=0 so it stands on a surface.
pub fn lander() -> Mesh {
    let mut m = Mesh::default();
    let gold = [0.82, 0.66, 0.26];
    let gray = [0.68, 0.70, 0.74];
    let dark = [0.13, 0.13, 0.15];
    let br = 2.2; // descent-stage body radius
    let y0 = 2.2; // body bottom
    let y1 = 4.0; // body top

    // descent stage body (octagonal)
    m.frustum(0.0, 0.0, y0, y1, br, br, 8, gold, true, true);
    // a darker equipment band
    m.frustum(0.0, 0.0, y0 + 0.4, y0 + 0.7, br * 1.02, br * 1.02, 8, [0.5, 0.42, 0.18], false, false);
    // descent engine bell, hanging below the body centre
    m.frustum(0.0, 0.0, 0.9, y0, 0.35, 1.0, 14, dark, false, true);

    // four landing legs + footpads + braces
    for k in 0..4 {
        let a = (k as f32 + 0.5) * std::f32::consts::FRAC_PI_2;
        let (cx, cz) = (a.cos(), a.sin());
        let hip = Vec3::new(cx * br * 0.9, y0 + 0.1, cz * br * 0.9);
        let foot = Vec3::new(cx * (br + 2.0), 0.12, cz * (br + 2.0));
        m.strut(hip, foot, 0.13, gray);
        // brace from higher on the body to the leg
        let shoulder = Vec3::new(cx * br * 0.5, y1 - 0.4, cz * br * 0.5);
        m.strut(shoulder, foot + Vec3::new(0.0, 0.6, 0.0), 0.08, gray);
        // footpad
        m.frustum(foot.x, foot.z, 0.0, 0.28, 0.55, 0.42, 10, gray, true, true);
    }

    // ascent cabin on top + hatch
    m.frustum(0.0, 0.0, y1, y1 + 1.5, 1.5, 1.2, 8, gray, false, true);
    m.frustum(0.0, 0.0, y1 + 1.5, y1 + 1.9, 0.7, 0.55, 10, dark, false, true);
    // four RCS thruster pods at the corners of the descent stage (the FX jets
    // fire from here at radius ~br, y ~ 3.6)
    for k in 0..4 {
        let a = (k as f32 + 0.5) * std::f32::consts::FRAC_PI_2;
        let (cx, cz) = (a.cos(), a.sin());
        let bx = Vec3::new(cx * br * 0.96, 3.6, cz * br * 0.96);
        m.bx(bx, Vec3::new(0.28, 0.34, 0.28), dark);
    }
    m
}

// ----------------------------------------------------------------------------
// Moon-base structures. A small catalog of modular surface buildings, each
// built about its own footprint centre at (cx, 0, cz) so they can be placed on
// the lunar surface (the y=0 plane of the rocket-view lunar terrain).
// ----------------------------------------------------------------------------

/// Metadata for a placeable base structure (name + footprint radius + blurb).
pub struct BaseStructure {
    pub name: &'static str,
    pub kind: &'static str,
    /// Approximate footprint radius (m) for layout / collision.
    pub footprint: f32,
    pub desc: &'static str,
}

/// The buildable moon-base parts catalog (order matches `add_base_structure`).
pub const BASE_PARTS: &[BaseStructure] = &[
    BaseStructure { name: "HQ / Admin", kind: "command", footprint: 8.0, desc: "Command and administration tower with comms dish." },
    BaseStructure { name: "Mining Outpost", kind: "industry", footprint: 9.0, desc: "Regolith drill derrick, ore hopper and conveyor." },
    BaseStructure { name: "Power Reactor", kind: "power", footprint: 8.0, desc: "Compact fission reactor with radiator fins." },
    BaseStructure { name: "Lunar VAB", kind: "industry", footprint: 11.0, desc: "Vehicle assembly hangar for surface-built craft." },
    BaseStructure { name: "3D Printing Facility", kind: "industry", footprint: 9.0, desc: "Gantry printer that fabricates parts from regolith." },
    BaseStructure { name: "Tourist Hub", kind: "civic", footprint: 9.0, desc: "Domed visitor centre and observation gallery." },
    BaseStructure { name: "Spaceport", kind: "transport", footprint: 12.0, desc: "Landing pad with control tower and approach lights." },
    BaseStructure { name: "Hotel", kind: "civic", footprint: 8.0, desc: "Tiered habitat hotel with panoramic window decks." },
    BaseStructure { name: "Refueling Station", kind: "transport", footprint: 9.0, desc: "Cryogenic propellant tanks and transfer pipes." },
];

// Shared lunar-base palette.
const BB_WHITE: [f32; 3] = [0.84, 0.86, 0.90];
const BB_TRIM: [f32; 3] = [0.52, 0.55, 0.60];
const BB_DARK: [f32; 3] = [0.12, 0.13, 0.17];
const BB_WIN: [f32; 3] = [0.30, 0.62, 0.85];
const BB_SOLAR: [f32; 3] = [0.10, 0.14, 0.36];
const BB_GOLD: [f32; 3] = [0.82, 0.66, 0.26];
const BB_STEEL: [f32; 3] = [0.58, 0.61, 0.66];
const BB_RED: [f32; 3] = [0.80, 0.22, 0.16];
const BB_CYAN: [f32; 3] = [0.55, 0.82, 0.95];
const BB_INDUST: [f32; 3] = [0.78, 0.62, 0.22];

/// A flat solar array panel: a thin tilted slab on a short post at (cx, cz).
fn solar_array(m: &mut Mesh, cx: f32, cz: f32, w: f32, d: f32, col: [f32; 3]) {
    m.frustum(cx, cz, 0.0, 1.4, 0.16, 0.16, 6, BB_STEEL, false, false); // post
    // a slab, slightly raised, with a frame underneath
    m.bx(Vec3::new(cx, 1.7, cz), Vec3::new(w, 0.08, d), col);
    m.bx(Vec3::new(cx, 1.55, cz), Vec3::new(w * 0.96, 0.05, d * 0.96), BB_TRIM);
}

/// A low connecting corridor tube between two ground points.
fn corridor(m: &mut Mesh, ax: f32, az: f32, bx: f32, bz: f32) {
    let a = Vec3::new(ax, 1.1, az);
    let b = Vec3::new(bx, 1.1, bz);
    m.strut(a, b, 0.9, BB_WHITE);
}

fn build_hq(m: &mut Mesh, cx: f32, cz: f32) {
    // stacked command tower, narrowing upward
    m.bx(Vec3::new(cx, 2.6, cz), Vec3::new(5.0, 2.6, 4.2), BB_WHITE);
    m.bx(Vec3::new(cx, 2.7, cz), Vec3::new(5.06, 0.8, 4.26), BB_WIN); // window band
    m.bx(Vec3::new(cx, 6.4, cz), Vec3::new(3.6, 1.4, 3.0), BB_WHITE);
    m.bx(Vec3::new(cx, 6.5, cz), Vec3::new(3.66, 0.6, 3.06), BB_WIN);
    m.bx(Vec3::new(cx, 8.6, cz), Vec3::new(2.2, 0.9, 1.9), BB_TRIM);
    // comms dish on a mast
    m.frustum(cx, cz, 9.5, 12.0, 0.12, 0.12, 6, BB_STEEL, false, false);
    m.frustum(cx + 0.9, cz, 11.0, 11.9, 0.0, 1.4, 12, BB_WHITE, false, true);
    // entrance + flag mast
    m.bx(Vec3::new(cx, 1.0, cz + 4.4), Vec3::new(1.2, 1.0, 0.5), BB_DARK);
    m.frustum(cx - 4.0, cz + 3.0, 0.0, 7.0, 0.07, 0.07, 5, BB_STEEL, false, false);
    m.bx(Vec3::new(cx - 3.6, 6.4, cz + 3.0), Vec3::new(0.9, 0.5, 0.04), BB_RED);
}

fn build_mining(m: &mut Mesh, cx: f32, cz: f32) {
    // a dark excavation pit lip + derrick over it
    m.frustum(cx, cz, 0.0, 0.3, 4.0, 3.2, 8, BB_TRIM, true, false);
    m.frustum(cx, cz, 0.05, 0.25, 3.0, 2.6, 8, BB_DARK, true, false);
    // four legs of the drill derrick converging to a head
    let head = Vec3::new(cx, 11.0, cz);
    for k in 0..4 {
        let a = (k as f32 + 0.5) * std::f32::consts::FRAC_PI_2;
        let foot = Vec3::new(cx + 3.4 * a.cos(), 0.2, cz + 3.4 * a.sin());
        m.strut(foot, head, 0.18, BB_INDUST);
        let mid = Vec3::new(cx + 1.9 * a.cos(), 5.5, cz + 1.9 * a.sin());
        m.strut(foot + Vec3::new(0.0, 4.0, 0.0), mid, 0.1, BB_STEEL);
    }
    // drill string down into the pit + winch head
    m.frustum(cx, cz, 0.0, 10.5, 0.25, 0.25, 6, BB_STEEL, false, false);
    m.bx(Vec3::new(cx, 11.2, cz), Vec3::new(1.0, 0.8, 1.0), BB_INDUST);
    // ore hopper + sloped conveyor off to the side
    m.frustum(cx + 6.0, cz, 1.6, 4.6, 1.9, 1.2, 6, BB_STEEL, false, true);
    m.bx(Vec3::new(cx + 6.0, 0.8, cz), Vec3::new(2.0, 0.8, 2.0), BB_TRIM);
    m.strut(Vec3::new(cx + 3.0, 1.0, cz), Vec3::new(cx + 6.0, 3.4, cz), 0.5, BB_DARK);
}

fn build_reactor(m: &mut Mesh, cx: f32, cz: f32) {
    // containment cylinder + dome
    m.frustum(cx, cz, 0.0, 5.0, 2.6, 2.4, 16, BB_WHITE, true, false);
    m.bx(Vec3::new(cx, 2.4, cz), Vec3::new(2.66, 0.7, 2.66), BB_RED); // hazard band (square overlay reads as a ring on the round body corners; keep subtle)
    m.dome(cx, cz, 5.0, 2.4, 1.8, 16, 4, BB_STEEL);
    // radiator fins radiating around the base
    for k in 0..6 {
        let a = k as f32 / 6.0 * TAU;
        let (dx, dz) = (a.cos(), a.sin());
        let bx = cx + dx * 4.6;
        let bz = cz + dz * 4.6;
        // a tall thin panel aligned radially
        let along = Vec3::new(dz, 0.0, -dx); // perpendicular for thickness
        let c = Vec3::new(bx, 2.6, bz);
        // build as a thin box: large along radial+vertical, thin across
        let he = Vec3::new(1.9 * dx.abs() + 0.12, 2.2, 1.9 * dz.abs() + 0.12);
        let _ = along;
        m.bx(c, Vec3::new(he.x, he.y, he.z), BB_TRIM);
    }
    // warning beacon
    m.frustum(cx, cz, 6.8, 7.4, 0.18, 0.1, 6, BB_RED, false, true);
}

fn build_vab(m: &mut Mesh, cx: f32, cz: f32) {
    // tall assembly hangar with a big door recess and roof trusses
    m.bx(Vec3::new(cx, 7.0, cz), Vec3::new(6.0, 7.0, 5.0), BB_WHITE);
    // big door (dark recess) on the +Z face
    m.bx(Vec3::new(cx, 5.0, cz + 5.02), Vec3::new(3.6, 5.0, 0.2), BB_DARK);
    m.bx(Vec3::new(cx, 5.0, cz + 5.06), Vec3::new(0.2, 5.0, 0.2), BB_TRIM); // door split
    // roof trusses
    for s in [-1.0f32, 0.0, 1.0] {
        m.strut(Vec3::new(cx - 6.0, 14.0, cz + s * 3.5), Vec3::new(cx + 6.0, 14.0, cz + s * 3.5), 0.18, BB_STEEL);
    }
    m.bx(Vec3::new(cx, 14.3, cz), Vec3::new(6.1, 0.4, 5.1), BB_TRIM); // roof cap
    // "LVAB" stripe
    m.bx(Vec3::new(cx, 11.6, cz + 5.02), Vec3::new(4.2, 0.7, 0.06), BB_GOLD);
}

fn build_printer(m: &mut Mesh, cx: f32, cz: f32) {
    // low open fabrication bay with a gantry printer over a build plate
    m.bx(Vec3::new(cx, 1.4, cz), Vec3::new(5.5, 1.4, 5.5), BB_WHITE);
    m.bx(Vec3::new(cx, 2.9, cz), Vec3::new(4.6, 0.2, 4.6), BB_DARK); // open bay floor
    m.bx(Vec3::new(cx, 3.0, cz), Vec3::new(2.4, 0.25, 2.4), BB_TRIM); // build plate
    // gantry: two rails + a moving crossbeam + a print head
    for s in [-1.0f32, 1.0] {
        m.bx(Vec3::new(cx + s * 4.4, 4.4, cz), Vec3::new(0.25, 1.6, 4.6), BB_STEEL);
    }
    m.bx(Vec3::new(cx, 5.6, cz + 1.2), Vec3::new(4.6, 0.3, 0.3), BB_STEEL); // crossbeam
    m.bx(Vec3::new(cx + 1.0, 5.0, cz + 1.2), Vec3::new(0.4, 0.7, 0.4), BB_INDUST); // head
    m.frustum(cx + 1.0, cz + 1.2, 4.3, 4.9, 0.06, 0.18, 6, BB_DARK, false, true); // nozzle
}

fn build_tourist(m: &mut Mesh, cx: f32, cz: f32) {
    // a glass observation dome on a low ring wall, with an entrance vestibule
    m.frustum(cx, cz, 0.0, 1.4, 5.2, 5.2, 20, BB_WHITE, false, false); // ring wall
    m.frustum(cx, cz, 0.0, 0.3, 5.2, 5.4, 20, BB_TRIM, true, false); // base lip
    m.dome(cx, cz, 1.4, 5.0, 4.4, 20, 5, BB_CYAN); // glass dome
    // meridian ribs on the dome
    for k in 0..6 {
        let a = k as f32 / 6.0 * TAU;
        m.strut(
            Vec3::new(cx + 5.0 * a.cos(), 1.5, cz + 5.0 * a.sin()),
            Vec3::new(cx, 5.7, cz),
            0.08,
            BB_TRIM,
        );
    }
    m.bx(Vec3::new(cx, 1.2, cz + 5.4), Vec3::new(1.6, 1.2, 1.2), BB_WHITE); // entrance
}

fn build_spaceport(m: &mut Mesh, cx: f32, cz: f32) {
    // a circular landing pad with markings + a control tower beside it
    m.frustum(cx, cz, 0.0, 0.35, 7.0, 7.0, 24, BB_TRIM, true, true);
    m.frustum(cx, cz, 0.36, 0.42, 3.0, 3.0, 20, BB_GOLD, true, false); // centre ring
    m.frustum(cx, cz, 0.43, 0.47, 0.9, 0.9, 12, BB_DARK, true, false);
    // four corner pad lights
    for k in 0..4 {
        let a = (k as f32 + 0.5) * std::f32::consts::FRAC_PI_2;
        m.frustum(cx + 6.2 * a.cos(), cz + 6.2 * a.sin(), 0.4, 1.1, 0.18, 0.1, 5, BB_RED, false, true);
    }
    // control tower off to one edge
    let tx = cx + 8.4;
    m.frustum(tx, cz, 0.0, 8.0, 0.9, 0.7, 8, BB_WHITE, true, false);
    m.frustum(tx, cz, 8.0, 9.4, 2.0, 1.6, 8, BB_WHITE, false, true); // flared cab
    m.frustum(tx, cz, 8.2, 9.1, 2.06, 1.66, 8, BB_WIN, false, false); // cab glass
    m.frustum(tx, cz, 9.4, 11.5, 0.06, 0.06, 4, BB_STEEL, false, false); // antenna
}

fn build_hotel(m: &mut Mesh, cx: f32, cz: f32) {
    // tiered "wedding cake" habitat: stacked cylinders with window bands
    let tiers = [(0.0f32, 4.6f32), (3.4, 3.8), (6.6, 3.0), (9.4, 2.2)];
    for (i, &(y, r)) in tiers.iter().enumerate() {
        let h = if i + 1 < tiers.len() { tiers[i + 1].0 } else { y + 2.4 };
        m.frustum(cx, cz, y, h, r, r * 0.92, 18, BB_WHITE, i == 0, false);
        // window band near the top of each tier
        m.frustum(cx, cz, h - 0.8, h - 0.2, r * 0.94, r * 0.9, 18, BB_WIN, false, false);
    }
    m.dome(cx, cz, 11.8, 2.0, 1.6, 16, 4, BB_GOLD); // roof cupola
    // viewing deck ring around the second tier
    m.frustum(cx, cz, 3.3, 3.5, 4.4, 4.4, 18, BB_TRIM, false, false);
}

fn build_refuel(m: &mut Mesh, cx: f32, cz: f32) {
    // two vertical cryo tanks + a spherical tank + connecting pipes
    for &(dx, dz, r, h) in &[(-2.6f32, 0.0f32, 1.6f32, 7.0f32), (2.6, 0.0, 1.6, 7.0)] {
        m.frustum(cx + dx, cz + dz, 0.0, h, r, r, 12, BB_WHITE, true, false);
        m.dome(cx + dx, cz + dz, h, r, r * 0.8, 12, 3, BB_STEEL);
        m.frustum(cx + dx, cz + dz, 0.0, 0.4, r * 1.05, r * 1.05, 12, BB_TRIM, true, false);
    }
    // spherical tank (dome up + dome down) on a cradle
    let sy = 2.6;
    m.dome(cx, cz + 4.2, sy, 1.8, 1.8, 14, 4, BB_GOLD);
    m.dome_down(cx, cz + 4.2, sy, 1.8, 1.8, 14, 4, BB_GOLD);
    m.frustum(cx, cz + 4.2, 0.0, sy - 1.4, 0.4, 0.6, 6, BB_STEEL, false, false); // pedestal
    // transfer pipes between tanks
    m.strut(Vec3::new(cx - 2.6, 1.2, cz), Vec3::new(cx + 2.6, 1.2, cz), 0.22, BB_STEEL);
    m.strut(Vec3::new(cx, 1.2, cz), Vec3::new(cx, 1.2, cz + 4.2), 0.22, BB_STEEL);
    // a fuel-line mast / flag
    m.frustum(cx + 4.0, cz, 0.0, 5.0, 0.08, 0.08, 5, BB_STEEL, false, false);
    m.bx(Vec3::new(cx + 4.4, 4.4, cz), Vec3::new(0.8, 0.45, 0.04), BB_GOLD);
}

/// Draw base structure `idx` (matching `BASE_PARTS`) at ground centre (cx, cz).
fn add_base_structure(m: &mut Mesh, idx: usize, cx: f32, cz: f32) {
    match idx {
        0 => build_hq(m, cx, cz),
        1 => build_mining(m, cx, cz),
        2 => build_reactor(m, cx, cz),
        3 => build_vab(m, cx, cz),
        4 => build_printer(m, cx, cz),
        5 => build_tourist(m, cx, cz),
        6 => build_spaceport(m, cx, cz),
        7 => build_hotel(m, cx, cz),
        _ => build_refuel(m, cx, cz),
    }
}

/// A single base structure, centred at the origin (for a parts preview).
pub fn base_structure(idx: usize) -> Mesh {
    let mut m = Mesh::default();
    add_base_structure(&mut m, idx % BASE_PARTS.len(), 0.0, 0.0);
    m
}

/// The whole moon base, laid out around a central plaza with connecting
/// corridors and solar arrays.
pub fn moon_base() -> Mesh {
    let mut m = Mesh::default();
    // layout: ring of structures around a central habitat dome + plaza
    let positions: [(f32, f32); 9] = [
        (0.0, -30.0),   // HQ (front)
        (34.0, -16.0),  // Mining
        (40.0, 14.0),   // Reactor
        (22.0, 34.0),   // VAB
        (-22.0, 34.0),  // Printer
        (-40.0, 14.0),  // Tourist
        (-34.0, -16.0), // Spaceport
        (-16.0, 18.0),  // Hotel (inner)
        (16.0, 18.0),   // Refuel (inner)
    ];
    // central plaza pad + habitat dome
    m.frustum(0.0, 4.0, 0.0, 0.25, 12.0, 12.0, 28, BB_TRIM, true, false);
    m.dome(0.0, 4.0, 0.25, 6.0, 4.5, 20, 5, BB_WHITE);
    m.frustum(0.0, 4.0, 0.25, 1.0, 6.0, 5.9, 20, BB_WIN, false, false);
    // connecting corridors from the plaza out to each structure
    for &(x, z) in &positions {
        let d = (x * x + z * z).sqrt().max(1.0);
        let (ux, uz) = (x / d, z / d);
        corridor(&mut m, ux * 7.0, 4.0 + uz * 7.0, x - ux * 9.0, z - uz * 9.0);
    }
    for (i, &(x, z)) in positions.iter().enumerate() {
        add_base_structure(&mut m, i, x, z);
    }
    // a solar-array field behind the reactor
    for k in 0..6 {
        let row = (k / 3) as f32;
        let col = (k % 3) as f32;
        solar_array(&mut m, 52.0 + col * 5.0, 26.0 + row * 6.0, 2.2, 2.6, BB_SOLAR);
    }
    m
}

/// All base structures lined up in a row (for a catalog / parts preview shot),
/// spaced along +X centred on the origin.
pub fn base_catalog() -> Mesh {
    let mut m = Mesh::default();
    let n = BASE_PARTS.len();
    let spacing = 22.0f32;
    for i in 0..n {
        let cx = (i as f32 - (n - 1) as f32 * 0.5) * spacing;
        add_base_structure(&mut m, i, cx, 0.0);
    }
    m
}

// ----------------------------------------------------------------------------
// Fairing-packed cargo modules. Compact, "packaged for launch" versions of the
// surface buildings (folded radiators, stowed arrays, etc.) that ride inside
// the fairing and unfold / are assembled on site. Built about y=0, kept within
// a ~0.78 m radius / ~4.8 m tall envelope so they fit a standard fairing.
// ----------------------------------------------------------------------------

/// A docking/berthing collar at the base of a cargo module.
fn mod_collar(m: &mut Mesh) {
    m.frustum(0.0, 0.0, 0.0, 0.45, 0.78, 0.78, 16, BB_STEEL, true, false);
    m.frustum(0.0, 0.0, 0.45, 0.6, 0.62, 0.62, 16, BB_DARK, false, true);
}

fn mod_refinery(m: &mut Mesh) {
    mod_collar(m);
    // main process drum
    m.frustum(0.0, 0.0, 0.6, 3.6, 0.7, 0.7, 16, BB_INDUST, true, false);
    m.dome(0.0, 0.0, 3.6, 0.7, 0.5, 16, 3, BB_STEEL);
    // stowed fractionating column alongside
    m.frustum(0.46, 0.0, 0.6, 4.4, 0.18, 0.16, 10, BB_WHITE, true, true);
    // a couple of process pipes wrapping the drum
    m.strut(Vec3::new(0.0, 1.0, 0.72), Vec3::new(0.0, 3.2, 0.72), 0.07, BB_STEEL);
    m.strut(Vec3::new(0.0, 1.0, -0.72), Vec3::new(0.0, 3.0, -0.72), 0.07, BB_STEEL);
    // intake hopper at the base
    m.frustum(-0.5, 0.0, 0.6, 1.3, 0.28, 0.12, 8, BB_TRIM, false, true);
}

fn mod_reactor(m: &mut Mesh) {
    mod_collar(m);
    // containment cylinder + dome
    m.frustum(0.0, 0.0, 0.6, 3.4, 0.6, 0.58, 16, BB_WHITE, true, false);
    m.frustum(0.0, 0.0, 2.3, 2.7, 0.62, 0.62, 16, BB_RED, false, false); // hazard band
    m.dome(0.0, 0.0, 3.4, 0.58, 0.5, 16, 3, BB_STEEL);
    // four radiator panels folded flat against the body (deploy on site)
    for k in 0..4 {
        let a = (k as f32 + 0.5) * std::f32::consts::FRAC_PI_2;
        let (cx, cz) = (a.cos() * 0.7, a.sin() * 0.7);
        m.bx(Vec3::new(cx, 2.0, cz), Vec3::new(0.06 + 0.5 * a.sin().abs(), 1.3, 0.06 + 0.5 * a.cos().abs()), BB_TRIM);
    }
}

fn mod_generator(m: &mut Mesh) {
    mod_collar(m);
    // power core box
    m.bx(Vec3::new(0.0, 1.6, 0.0), Vec3::new(0.6, 1.0, 0.6), BB_WHITE);
    // stowed (folded) solar array stacks on two sides
    for s in [-1.0f32, 1.0] {
        for k in 0..4 {
            let y = 0.9 + k as f32 * 0.45;
            m.bx(Vec3::new(s * 0.72, y, 0.0), Vec3::new(0.05, 0.18, 0.55), BB_SOLAR);
        }
    }
    // battery / inverter drum + radiator stub on top
    m.frustum(0.0, 0.0, 2.6, 3.4, 0.4, 0.36, 12, BB_TRIM, false, true);
    m.bx(Vec3::new(0.0, 3.7, 0.0), Vec3::new(0.5, 0.4, 0.05), BB_DARK);
}

fn mod_habitat(m: &mut Mesh) {
    mod_collar(m);
    // pressurised can with a window band + end dome
    m.frustum(0.0, 0.0, 0.6, 3.8, 0.74, 0.74, 18, BB_WHITE, true, false);
    m.frustum(0.0, 0.0, 1.8, 2.4, 0.76, 0.76, 18, BB_WIN, false, false);
    m.dome(0.0, 0.0, 3.8, 0.74, 0.6, 18, 3, BB_WHITE);
    // top docking node + a side berthing port
    m.frustum(0.0, 0.0, 4.4, 4.7, 0.28, 0.28, 12, BB_TRIM, false, true);
    m.frustum(0.76, 0.0, 2.0, 2.0, 0.0, 0.0, 4, BB_TRIM, false, false); // (placeholder, see nub below)
    m.bx(Vec3::new(0.82, 2.0, 0.0), Vec3::new(0.12, 0.3, 0.3), BB_TRIM);
}

fn mod_drill(m: &mut Mesh) {
    mod_collar(m);
    // stowed drill mast + auger bit, hopper and folded outrigger legs
    m.frustum(0.0, 0.0, 0.6, 4.6, 0.16, 0.13, 8, BB_STEEL, false, false); // mast
    m.frustum(0.0, 0.0, 0.5, 0.95, 0.0, 0.3, 8, BB_DARK, false, true); // auger bit (point down-ish)
    m.bx(Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.5, 0.9, 0.5), BB_INDUST); // machinery box
    m.frustum(0.0, 0.0, 2.6, 3.4, 0.6, 0.4, 8, BB_TRIM, false, true); // hopper
    for k in 0..3 {
        let a = k as f32 / 3.0 * TAU;
        m.strut(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.62 * a.cos(), 0.7, 0.62 * a.sin()), 0.07, BB_STEEL);
    }
}

/// A fairing-packed cargo module (index matches the `module` field in the
/// payload catalog: 0 refinery, 1 reactor, 2 generator, 3 habitat, 4 drill).
pub fn cargo_module(idx: usize) -> Mesh {
    let mut m = Mesh::default();
    match idx {
        0 => mod_refinery(&mut m),
        1 => mod_reactor(&mut m),
        2 => mod_generator(&mut m),
        3 => mod_habitat(&mut m),
        _ => mod_drill(&mut m),
    }
    m
}

// ----------------------------------------------------------------------------
// Procedural asteroids. An irregular ("potato") body: a cube-sphere displaced
// by value-noise lumps and gouged by a handful of impact craters, then stretched
// into an ellipsoid. Built about the origin in metres, lit as an airless rock.
// ----------------------------------------------------------------------------

fn hash31(p: Vec3) -> f32 {
    let h = (p.x * 127.1 + p.y * 311.7 + p.z * 74.7).sin() * 43758.547;
    h - h.floor()
}

/// Smooth value noise in 3D (trilinear interpolation of a hashed lattice).
fn vnoise(p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f); // smoothstep weights
    let c = |dx: f32, dy: f32, dz: f32| hash31(i + Vec3::new(dx, dy, dz));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(c(0., 0., 0.), c(1., 0., 0.), u.x);
    let x10 = lerp(c(0., 1., 0.), c(1., 1., 0.), u.x);
    let x01 = lerp(c(0., 0., 1.), c(1., 0., 1.), u.x);
    let x11 = lerp(c(0., 1., 1.), c(1., 1., 1.), u.x);
    let y0 = lerp(x00, x10, u.y);
    let y1 = lerp(x01, x11, u.y);
    lerp(y0, y1, u.z)
}

/// Fractal value noise in [-1, 1].
fn fbm3(p: Vec3, octaves: i32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for _ in 0..octaves {
        sum += amp * (vnoise(p * freq) * 2.0 - 1.0);
        freq *= 2.0;
        amp *= 0.5;
    }
    sum
}

fn sphere_rand(seed: f32, k: usize) -> Vec3 {
    let a = hash31(Vec3::new(seed, k as f32, 1.0)) * TAU;
    let z = hash31(Vec3::new(seed, k as f32, 2.0)) * 2.0 - 1.0;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * a.cos(), z, r * a.sin())
}

/// Unit cube-face point (u, v in [-1, 1]) mapped to a sphere direction.
fn cube_dir(face: usize, u: f32, v: f32) -> Vec3 {
    let p = match face {
        0 => Vec3::new(1.0, v, -u),
        1 => Vec3::new(-1.0, v, u),
        2 => Vec3::new(u, 1.0, -v),
        3 => Vec3::new(u, -1.0, v),
        4 => Vec3::new(u, v, 1.0),
        _ => Vec3::new(-u, v, -1.0),
    };
    p.normalize()
}

fn asteroid_color(dir: Vec3, disp: f32, n: Vec3) -> [f32; 3] {
    // height-tinted rock: darker in the gouged lows, lighter on raised faces
    let h = ((disp - 0.82) / 0.5).clamp(0.0, 1.0);
    let base = mix3([0.19, 0.16, 0.13], [0.41, 0.37, 0.31], h);
    // expose slightly cooler rock on steep slopes (fresh fracture faces)
    let slope = (1.0 - n.dot(dir)).clamp(0.0, 1.0);
    let c = mix3(base, [0.30, 0.30, 0.31], slope * 0.55);
    let j = 0.84 + 0.32 * hash31(dir * 41.0);
    [c[0] * j, c[1] * j, c[2] * j]
}

/// A procedural asteroid of mean `radius` (m), stretched by `elong` (per-axis
/// scale) and roughened by `seed`. `craters` impact basins are gouged in.
pub fn asteroid(seed: f32, radius: f32, elong: Vec3, craters: usize) -> Mesh {
    let off = Vec3::splat(seed * 13.7 + 4.0);
    // Craters with a power-law size mix (many small, a few large basins).
    let crat: Vec<(Vec3, f32, f32)> = (0..craters)
        .map(|k| {
            let dir = sphere_rand(seed + 7.0, k);
            let a = hash31(Vec3::new(seed, k as f32, 3.0));
            let b = hash31(Vec3::new(seed, k as f32, 5.0));
            let cr = 0.05 + 0.5 * a.powf(2.5); // angular radius, biased small
            let depth = (0.16 + 0.12 * b) * (0.5 + cr); // bigger craters dig deeper
            (dir, cr, depth)
        })
        .collect();

    let displace = |dir: Vec3| -> f32 {
        // Keep the base body fairly round (modest lumps) so the craters read as
        // the dominant relief rather than getting lost in noise.
        let lump = fbm3(dir * 2.0 + off, 5);
        let fine = fbm3(dir * 7.0 + off, 4);
        let mut h = 1.0 + 0.13 * lump + 0.035 * fine;
        for &(c, cr, depth) in &crat {
            let ang = (1.0 - dir.dot(c).clamp(-1.0, 1.0)).max(0.0); // 0 at centre
            let s = ang / cr; // 0 centre, 1 rim
            if s < 1.5 {
                // depressed parabolic floor that rises to the rim ...
                if s < 1.0 {
                    h -= depth * (1.0 - s * s);
                }
                // ... plus a sharp raised rim ring + a little ejecta outside it
                let rim = 0.45 * depth * (-(((s - 1.0) / 0.18).powi(2))).exp();
                h += rim;
            }
        }
        h.max(0.42)
    };
    let surf = |dir: Vec3| -> Vec3 {
        let d = displace(dir);
        Vec3::new(dir.x * elong.x, dir.y * elong.y, dir.z * elong.z) * (radius * d)
    };

    let n = 60usize; // resolves the small craters (~125k verts, under DYN cap)
    let mut m = Mesh::default();
    for face in 0..6 {
        let mut pos = vec![Vec3::ZERO; n * n];
        let mut dir = vec![Vec3::ZERO; n * n];
        for j in 0..n {
            for i in 0..n {
                let u = (i as f32 / (n - 1) as f32) * 2.0 - 1.0;
                let v = (j as f32 / (n - 1) as f32) * 2.0 - 1.0;
                let dd = cube_dir(face, u, v);
                dir[j * n + i] = dd;
                pos[j * n + i] = surf(dd);
            }
        }
        // smooth normals via central differences within the face grid
        let mut nrm = vec![Vec3::ZERO; n * n];
        for j in 0..n {
            for i in 0..n {
                let im = i.saturating_sub(1);
                let ip = (i + 1).min(n - 1);
                let jm = j.saturating_sub(1);
                let jp = (j + 1).min(n - 1);
                let du = pos[j * n + ip] - pos[j * n + im];
                let dv = pos[jp * n + i] - pos[jm * n + i];
                let mut nn = du.cross(dv).normalize_or_zero();
                // Orient outward using the position vector from the origin. The
                // body is star-convex (radius always > 0), so the outward normal
                // always has a positive dot with `pos`; the sphere direction is
                // an unreliable proxy on elongated bodies and flipped whole faces.
                if nn.dot(pos[j * n + i]) < 0.0 {
                    nn = -nn;
                }
                if nn.length_squared() < 1e-6 {
                    nn = pos[j * n + i].normalize_or_zero();
                }
                nrm[j * n + i] = nn;
            }
        }
        let col: Vec<[f32; 3]> = (0..n * n)
            .map(|k| asteroid_color(pos[k].normalize_or_zero(), displace(dir[k]), nrm[k]))
            .collect();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let a = j * n + i;
                let b = j * n + i + 1;
                let c = (j + 1) * n + i;
                let d = (j + 1) * n + i + 1;
                m.tri3([pos[a], pos[c], pos[b]], [nrm[a], nrm[c], nrm[b]], [col[a], col[c], col[b]]);
                m.tri3([pos[b], pos[c], pos[d]], [nrm[b], nrm[c], nrm[d]], [col[b], col[c], col[d]]);
            }
        }
    }
    m
}

/// A few named large asteroids with distinct silhouettes.
pub fn asteroid_preset(idx: usize) -> Mesh {
    match idx {
        0 => asteroid(2.0, 520.0, Vec3::new(1.04, 0.96, 1.0), 20), // near-spherical
        1 => asteroid(8.0, 470.0, Vec3::new(1.12, 0.84, 1.05), 30), // squat, cratered
        2 => asteroid(15.0, 360.0, Vec3::new(1.7, 0.78, 0.86), 16), // elongated peanut
        _ => asteroid(23.0, 430.0, Vec3::new(1.55, 0.82, 0.95), 22), // long shard
    }
}

pub const ASTEROID_NAMES: &[&str] = &["Hebe", "Pallas", "Itokara", "Eron"];

/// The five cargo modules lined up in a row (for a parts preview), each
/// unpacked/standing on the ground, spaced along +X centred on the origin.
pub fn cargo_catalog() -> Mesh {
    let mut m = Mesh::default();
    let spacing = 4.0f32;
    for i in 0..5usize {
        let cx = (i as f32 - 2.0) * spacing;
        let cm = cargo_module(i);
        for v in &cm.verts {
            let p = Vec3::new(v.pos[0] + cx, v.pos[1], v.pos[2]);
            m.verts.push(MeshVertex { pos: p.into(), normal: v.normal, color: v.color });
        }
    }
    m
}

/// The flyable rocket body, built about its base at y=0 and pointing +Y. The
/// pad and mount are separate (they stay behind on liftoff). `stage_ranges[i]`
/// is the vertex range of stage i (bottom-first), so a spent stage can be split
/// off and tumbled away at separation; `payload_range` is the payload + nose.
pub struct RocketBody {
    pub mesh: Mesh,
    pub stage_ranges: Vec<std::ops::Range<usize>>,
    pub payload_range: std::ops::Range<usize>,
    /// Height (m) the base sits above the pad slab when resting on the mount.
    pub base_y: f32,
    /// Total stack height (m).
    pub height: f32,
    /// Y to aim the camera at (mid-stack), and a good default distance.
    pub focus_y: f32,
    pub cam_dist: f32,
    /// Engine-cluster radius per stage (plume width).
    pub engine_r: Vec<f32>,
    /// Mesh-Y of each stage's engine mount (where its exhaust exits).
    pub nozzle_y: Vec<f32>,
    /// The cargo module inside the fairing (subset of `payload_range`).
    pub module_range: std::ops::Range<usize>,
    /// The two clamshell fairing halves (each swings out along local +/-X).
    pub fairing_l: std::ops::Range<usize>,
    pub fairing_r: std::ops::Range<usize>,
}

pub const PAD_TOP: f32 = 1.2;
pub const MOUNT_H: f32 = 2.2; // rocket base sits this far above the pad slab
const PROP_DENSITY: f32 = 1000.0; // kg/m^3, sizes stage height from propellant

/// Stage body radius (m) from its propellant load (bigger tank -> wider).
fn stage_radius(prop: f64) -> f32 {
    ((prop / 200_000.0).cbrt() as f32 * 1.9).clamp(0.7, 3.2)
}

/// The static launch pad slab + mount legs (the planet terrain is the ground;
/// see `build_terrain`). The rocket itself is the separate `rocket_body`.
pub fn pad_and_mount() -> Mesh {
    let mut m = Mesh::default();
    m.bx(Vec3::new(0.0, PAD_TOP * 0.5, 0.0), Vec3::new(9.0, PAD_TOP * 0.5, 9.0), [0.42, 0.42, 0.45]);
    for (sx, sz) in [(1.0f32, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        m.bx(
            Vec3::new(sx * 2.3, PAD_TOP + MOUNT_H * 0.5, sz * 2.3),
            Vec3::new(0.35, MOUNT_H * 0.5, 0.35),
            [0.28, 0.29, 0.32],
        );
    }
    m
}

/// The Vehicle Assembly Building: a large enclosed hangar (floor, back + side
/// walls, roof, an open front facing the pad, and internal gantry towers) the
/// rocket is assembled inside. Centred at `c` (local metres), big enough that
/// the camera orbits around inside it.
pub fn hangar(c: Vec3, light_offsets: &[Vec3]) -> Mesh {
    let mut m = Mesh::default();
    let wall = [0.33, 0.35, 0.39];
    let inner = [0.27, 0.29, 0.33];
    let frame = [0.21, 0.22, 0.26];
    let w = 62.0f32; // half-width (X); open front at +X toward the pad
    let d = 56.0f32; // half-depth (Z)
    let h = 150.0f32; // height
    let t = 1.4f32; // wall/floor thickness

    // Solid floor + a paved apron extending well past the walls, so no grass
    // shows inside or just outside the open front. Top (y +1.7) is just under
    // the rocket's engines; it pokes below ground to cover the gentle curvature.
    // Paved floor + apron as a tiled grid of moderate slabs (a single huge quad
    // mis-rasterises against the fine terrain at these km-scale coords). Covers
    // the interior and a margin out the front door so no grass shows near the
    // rocket.
    for ix in -1..=2 {
        for iz in -2..=2 {
            let col = if ix <= 0 { [0.32, 0.33, 0.37] } else { [0.30, 0.31, 0.35] };
            m.bx(
                c + Vec3::new(ix as f32 * (w * 0.95), -2.0, iz as f32 * (d * 0.62)),
                Vec3::new(w * 0.55, 3.7, d * 0.36),
                col,
            );
        }
    }
    // back wall (-X)
    m.bx(c + Vec3::new(-w + t, h * 0.5, 0.0), Vec3::new(t, h * 0.5, d), wall);
    // side walls (+/-Z)
    for sz in [-1.0f32, 1.0] {
        m.bx(c + Vec3::new(0.0, h * 0.5, sz * (d - t)), Vec3::new(w, h * 0.5, t), wall);
    }
    // front: a tall doorway - header beam across the top + jambs at the corners,
    // leaving a wide opening so the rocket can roll out toward the pad.
    m.bx(c + Vec3::new(w - t, h - 7.0, 0.0), Vec3::new(t, 7.0, d), inner);
    for sz in [-1.0f32, 1.0] {
        m.bx(c + Vec3::new(w - t, h * 0.5, sz * (d - 7.0)), Vec3::new(t, h * 0.5, 7.0), inner);
    }
    // roof + a few beams visible from inside (kept clear of the rocket)
    m.bx(c + Vec3::new(0.0, h, 0.0), Vec3::new(w, t, d), inner);
    for k in -3..=3 {
        m.bx(c + Vec3::new(k as f32 * 18.0, h - 2.5, 0.0), Vec3::new(0.6, 1.2, d), frame);
    }
    // light fixtures at the work-light positions (bright; they sit at a point
    // light so they read as glowing lamps)
    for &off in light_offsets {
        m.bx(c + off, Vec3::new(1.6, 0.5, 1.6), [1.6, 1.55, 1.4]);
    }
    m
}

#[derive(Clone, Copy, PartialEq)]
pub enum PartKind {
    Engine,
    Tank,
    Payload,
}

/// Append an axis-aligned box (public wrapper, for rack shelves etc.).
pub fn append_box(m: &mut Mesh, center: Vec3, he: Vec3, col: [f32; 3]) {
    m.bx(center, he, col);
}

/// Append a small 3D model of a catalog part centred at `c`, for the parts rack
/// / drag ghost. `col` tints it.
pub fn append_part(m: &mut Mesh, kind: PartKind, c: Vec3, col: [f32; 3]) {
    match kind {
        PartKind::Engine => {
            // a bell nozzle
            m.frustum(c.x, c.z, c.y - 0.8, c.y + 0.5, 0.7, 0.35, 12, col, true, true);
        }
        PartKind::Tank => {
            m.frustum(c.x, c.z, c.y - 1.1, c.y + 1.1, 0.8, 0.8, 14, col, true, true);
        }
        PartKind::Payload => {
            m.frustum(c.x, c.z, c.y - 0.7, c.y - 0.1, 0.6, 0.6, 12, col, true, false);
            m.frustum(c.x, c.z, c.y - 0.1, c.y + 1.2, 0.6, 0.0, 12, col, false, false);
        }
    }
}

/// Build the rocket body for `veh` about its base at y=0, proportional to each
/// stage's tank (radius/height) and engine (cluster). `payload_col` tints the
/// payload section.
pub fn rocket_body(veh: &Vehicle, payload_col: [f32; 3], module_id: i32) -> RocketBody {
    let mut m = Mesh::default();
    let n = veh.stages.len().max(1);
    let radii: Vec<f32> = veh.stages.iter().map(|s| stage_radius(s.prop)).collect();
    let body_cols = [[0.90f32, 0.90, 0.93], [0.72, 0.74, 0.78], [0.66, 0.68, 0.74]];

    let mut stage_ranges = Vec::new();
    let mut nozzle_y = Vec::new();
    let mut engine_r = Vec::new();
    let mut y = 0.0f32;
    for (i, stage) in veh.stages.iter().enumerate() {
        let start = m.verts.len();
        let r = radii[i];
        let col = body_cols[i.min(body_cols.len() - 1)];
        let vol = stage.prop as f32 / PROP_DENSITY;
        let h = (vol / (std::f32::consts::PI * r * r)).max(2.5);
        nozzle_y.push(y);

        // body + a couple of dark bands for scale
        m.frustum(0.0, 0.0, y, y + h, r, r, 24, col, false, false);
        m.frustum(0.0, 0.0, y + h * 0.33, y + h * 0.33 + 0.3, r * 1.01, r * 1.01, 24, [0.15, 0.16, 0.18], false, false);

        // engines: a cluster for high-thrust boosters, a single bell otherwise
        let nz = if stage.thrust > 5.0e6 { 5 } else if stage.thrust > 2.0e6 { 4 } else { 1 };
        let er = if nz > 1 { (r * 0.5).clamp(0.4, 1.7) } else { (r * 0.45).clamp(0.3, 1.2) };
        engine_r.push(er);
        for k in 0..nz {
            let (ex, ez) = if nz > 1 && k < nz - 1 {
                let a = k as f32 / (nz - 1) as f32 * std::f32::consts::TAU;
                (a.cos() * r * 0.5, a.sin() * r * 0.5)
            } else {
                (0.0, 0.0)
            };
            m.frustum(ex, ez, y - 1.7, y, er * 0.5, er * 0.8, 12, [0.13, 0.13, 0.15], false, true);
        }
        // fins on the first stage
        if i == 0 {
            let fy = y + 2.0;
            for (cx, cz, hx, hz) in [
                (r + 0.7, 0.0, 0.9, 0.12),
                (-(r + 0.7), 0.0, 0.9, 0.12),
                (0.0, r + 0.7, 0.12, 0.9),
                (0.0, -(r + 0.7), 0.12, 0.9),
            ] {
                m.bx(Vec3::new(cx, fy, cz), Vec3::new(hx, 1.8, hz), [0.55, 0.10, 0.10]);
            }
        }

        y += h;
        // interstage tapering to the next stage's radius (or toward the payload)
        let next_r = radii.get(i + 1).copied().unwrap_or(r * 0.85);
        m.frustum(0.0, 0.0, y, y + 0.6, r, next_r, 24, [0.18, 0.18, 0.21], false, false);
        y += 0.6;

        stage_ranges.push(start..m.verts.len());
    }

    // payload section: the cargo module inside a clamshell fairing + nose.
    let pstart = m.verts.len();
    let has_module = module_id >= 0;
    // fairing inner radius: wide enough to enclose a cargo module if present.
    let last_r = radii.last().copied().unwrap_or(1.5);
    let pr = if has_module { (last_r * 0.95).max(1.05) } else { last_r * 0.85 };
    let fy0 = y; // fairing base
    let cyl_h = if has_module { 5.0 } else { 4.0 };
    let nose_h = 4.0;
    let fy1 = fy0 + cyl_h; // shoulder
    let ny = fy1 + nose_h; // nose tip

    // 1) the payload itself, sitting on the upper-stage forward dome
    let mstart = m.verts.len();
    if has_module {
        // place a fairing-fit cargo module (scaled to clear the fairing wall)
        let cm = cargo_module(module_id as usize);
        let s = ((pr * 0.82) / 0.78).min(1.25);
        for v in &cm.verts {
            let p = Vec3::new(v.pos[0] * s, fy0 + 0.2 + v.pos[1] * s.min(1.05), v.pos[2] * s);
            m.verts.push(MeshVertex { pos: p.into(), normal: v.normal, color: v.color });
        }
    } else {
        // a plain boxed satellite
        m.frustum(0.0, 0.0, fy0 + 0.4, fy0 + 3.4, pr * 0.6, pr * 0.55, 12, payload_col, true, true);
    }
    let module_range = mstart..m.verts.len();

    // 2) the fairing as two clamshell halves (each a 180-deg arc of the
    // cylinder + ogive nose), so they can be swung apart to reveal the module.
    let half = |m: &mut Mesh, a_start: f32, a_end: f32| {
        let segs = 12usize;
        let apex = Vec3::new(0.0, ny, 0.0);
        for i in 0..segs {
            let a0 = a_start + (a_end - a_start) * i as f32 / segs as f32;
            let a1 = a_start + (a_end - a_start) * (i + 1) as f32 / segs as f32;
            let am = 0.5 * (a0 + a1);
            // cylinder wall
            let c00 = Vec3::new(pr * a0.cos(), fy0, pr * a0.sin());
            let c10 = Vec3::new(pr * a1.cos(), fy0, pr * a1.sin());
            let c11 = Vec3::new(pr * a1.cos(), fy1, pr * a1.sin());
            let c01 = Vec3::new(pr * a0.cos(), fy1, pr * a0.sin());
            let nwall = Vec3::new(am.cos(), 0.0, am.sin());
            m.quad(c00, c10, c11, c01, nwall, payload_col);
            // ogive nose triangle to the apex
            let nnose = Vec3::new(am.cos(), 0.5, am.sin()).normalize();
            m.tri(c01, c11, apex, nnose, [0.93, 0.93, 0.96]);
        }
    };
    use std::f32::consts::{FRAC_PI_2, PI};
    let flstart = m.verts.len();
    half(&mut m, FRAC_PI_2, FRAC_PI_2 + PI); // -X half
    let fairing_l = flstart..m.verts.len();
    let frstart = m.verts.len();
    half(&mut m, -FRAC_PI_2, FRAC_PI_2); // +X half
    let fairing_r = frstart..m.verts.len();

    y = ny;
    let payload_range = pstart..m.verts.len();

    let _ = n;
    RocketBody {
        mesh: m,
        stage_ranges,
        payload_range,
        base_y: PAD_TOP + MOUNT_H,
        height: y,
        focus_y: y * 0.45,
        cam_dist: y * 1.7,
        engine_r,
        nozzle_y,
        module_range,
        fairing_l,
        fairing_r,
    }
}

// ---------------------------------------------------------------------------
// Real planet terrain in the rocket view.
//
// The planet is ~6200 km; the rocket is metres. We render the LOD cube-sphere
// surface in a local tangent frame whose origin is the spaceport surface point
// (floating origin), so every vertex is small and f32-precise. The mesh pipeline
// applies a logarithmic depth buffer so near (rocket) and far (horizon) coexist
// without z-fighting.
// ---------------------------------------------------------------------------

use glam::DVec3;
use terrain::{build_mesh, select, Elevation, Planet};

/// Spaceport (matches sim / worldgen seed 47).
const SPACEPORT_LAT_DEG: f64 = -1.7;
const SPACEPORT_LON_DEG: f64 = -102.9;
pub const PLANET_RADIUS: f64 = 6.2e6;

/// The launch-site direction (unit), honouring the MTS_TERRAIN_LATLON override.
fn spaceport_dir() -> DVec3 {
    let (lat_deg, lon_deg) = std::env::var("MTS_TERRAIN_LATLON")
        .ok()
        .and_then(|s| {
            let mut it = s.split(',');
            Some((it.next()?.trim().parse().ok()?, it.next()?.trim().parse().ok()?))
        })
        .unwrap_or((SPACEPORT_LAT_DEG, SPACEPORT_LON_DEG));
    let lat = (lat_deg as f64).to_radians();
    let lon = (lon_deg as f64).to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

/// The planet elevation field with the launch-pad flat zone applied (unless
/// MTS_TERRAIN_NOFLAT). Shared by the launch frame and the terrain mesh.
fn launch_elevation() -> Elevation {
    let mut elev = Elevation::new(47);
    if std::env::var("MTS_TERRAIN_NOFLAT").is_err() {
        // flat out far enough to hold the pad, the assembly building ~5 km away,
        // and the rollout corridor between them.
        elev.add_flat_zone(spaceport_dir(), 6500.0, 13000.0, PLANET_RADIUS);
    }
    elev
}

/// The cratered lunar elevation, with a small flat landing site at the
/// rocket-view origin so a touched-down lander rests on the surface (and is
/// offset vertically to align that site with the home-frame origin height).
fn lunar_elevation() -> Elevation {
    let dir = spaceport_dir();
    let mut elev = Elevation::lunar(47);
    // signed crater height at the site (no flat zone / offset applied yet)
    let site = elev.height_m(dir);
    let h0 = launch_elevation().land_height_m(dir);
    // flatten a small touchdown pad, then shift the field so the pad sits at the
    // same height the rocket-view origin assumes.
    elev.add_flat_zone(dir, 140.0, 480.0, PLANET_RADIUS);
    elev.set_offset(h0 - site);
    elev
}

/// The rocket-view local tangent frame at the spaceport: the surface origin
/// (home-centred metres) plus the up / east / north basis. The launch physics
/// and the terrain share this so the flying rocket lines up with the ground.
pub fn launch_frame() -> (DVec3, DVec3, DVec3, DVec3) {
    let dir = spaceport_dir();
    let h0 = launch_elevation().land_height_m(dir);
    let origin = dir * (PLANET_RADIUS + h0);
    let up = dir;
    let north = (DVec3::Y - up * up.dot(DVec3::Y)).normalize();
    let east = north.cross(up).normalize();
    (origin, up, east, north)
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn hashf(p: Vec3) -> f32 {
    let mut h = (p.x * 127.1 + p.y * 311.7 + p.z * 74.7).sin() * 43758.547;
    h -= h.floor();
    h
}

fn terrain_color(signed_h: f64, slope: f32, jitter: f32, abs_lat: f64, lunar: bool) -> [f32; 3] {
    if lunar {
        // dark grey regolith (low albedo, so the relief from lighting reads
        // strongly): darker mare in the basins, lighter highland rims, a touch
        // brighter on steep slopes where fresh material is exposed.
        let h = (signed_h as f32 / 4000.0).clamp(-1.0, 1.0);
        let base = mix3([0.18, 0.18, 0.19], [0.40, 0.40, 0.40], (h * 0.5 + 0.5).clamp(0.0, 1.0));
        let bright = mix3(base, [0.50, 0.49, 0.48], (slope * 1.5).clamp(0.0, 1.0));
        let b = 0.88 + 0.22 * jitter;
        return [bright[0] * b, bright[1] * b, bright[2] * b];
    }
    if signed_h <= 0.0 {
        // shallow to deep sea
        let t = ((-signed_h) / 1200.0).clamp(0.0, 1.0) as f32;
        return mix3([0.07, 0.22, 0.34], [0.03, 0.10, 0.20], t);
    }
    let h = signed_h as f32;
    // land colour by elevation (grass -> scrub)
    let t = (h / 4200.0).clamp(0.0, 1.0);
    let grass = [0.20, 0.34, 0.15];
    let scrub = [0.38, 0.33, 0.18];
    let mut base = mix3(grass, scrub, t);
    // steep faces read as bare rock
    let rock = [0.32, 0.28, 0.25];
    let steep = ((slope - 0.30) / 0.35).clamp(0.0, 1.0);
    base = mix3(base, rock, steep);
    // Latitude-aware snow: high snow line at the equator, low near the poles,
    // plus polar ice caps at any elevation.
    let lat_frac = (abs_lat / std::f64::consts::FRAC_PI_2) as f32; // 0 equator .. 1 pole
    let snow_line = 1000.0 + (1.0 - lat_frac) * 5200.0;
    let alpine = ((h - snow_line) / 1400.0).clamp(0.0, 1.0);
    let polar = ((lat_frac - 0.82) / 0.12).clamp(0.0, 1.0);
    let snow = alpine.max(polar);
    base = mix3(base, [0.90, 0.92, 0.96], snow);
    // micro brightness variation so the ground is not flat
    let b = 0.90 + 0.16 * jitter;
    [base[0] * b, base[1] * b, base[2] * b]
}

/// Build the entire procedural planet as a cube-sphere quadtree LOD mesh,
/// refined toward `cam_world` and coarsening to the far limb - the whole world
/// in one mesh. Vertices are emitted in the launch-tangent frame, camera-
/// relative to `ref_local` (floating origin), so f32 keeps precision near the
/// camera even at planet scale; the mesh pipeline's logarithmic depth lets the
/// metre-scale foreground and the 6000 km limb share one depth buffer. This is
/// the seamless ground-to-orbit terrain.
#[allow(clippy::too_many_arguments)]
pub fn planet_terrain(
    cam_world: DVec3,
    ref_local: DVec3,
    origin: DVec3,
    up: DVec3,
    east: DVec3,
    north: DVec3,
    max_depth: u32,
    lunar: bool,
) -> Mesh {
    let planet = Planet { radius: PLANET_RADIUS };
    let elev = if lunar { lunar_elevation() } else { launch_elevation() };
    let to_local = |w: DVec3| -> DVec3 {
        let d = w - origin;
        DVec3::new(d.dot(east), d.dot(up), d.dot(north))
    };
    let dir_local = |d: DVec3| -> Vec3 {
        Vec3::new(d.dot(east) as f32, d.dot(up) as f32, d.dot(north) as f32)
    };

    let lod = select(&planet, cam_world, 1.5, max_depth);
    let mut m = Mesh::default();
    let n = 9;
    let grid = n * n; // first `grid` positions are the surface; the rest are skirts
    for patch in &lod.patches {
        // Skirt depth scales with the patch so coarse far patches still seal.
        let skirt = (patch.edge * 0.3).clamp(80.0, 80_000.0);
        let pm = build_mesh(&planet, patch, n, &elev, skirt);
        let nv = pm.positions.len();

        // Per-vertex local position and outward (radial) direction in local axes.
        let local: Vec<Vec3> = pm
            .positions
            .iter()
            .map(|&w| (to_local(w) - ref_local).as_vec3())
            .collect();
        let radial: Vec<Vec3> = pm
            .positions
            .iter()
            .map(|&w| dir_local(w.normalize()))
            .collect();

        // Smooth normals: accumulate area-weighted face normals from the surface
        // triangles only (skip skirt walls so they don't tilt the rim). Each
        // shared grid vertex then averages its neighbouring faces.
        let mut nrm = vec![Vec3::ZERO; nv];
        for tri in pm.indices.chunks(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= grid || i1 >= grid || i2 >= grid {
                continue; // skirt triangle
            }
            let mut fnv = (local[i1] - local[i0]).cross(local[i2] - local[i0]);
            let rc = radial[i0] + radial[i1] + radial[i2];
            if fnv.dot(rc) < 0.0 {
                fnv = -fnv;
            }
            nrm[i0] += fnv;
            nrm[i1] += fnv;
            nrm[i2] += fnv;
        }
        for i in 0..nv {
            let mut nn = nrm[i].normalize_or_zero();
            if nn.length_squared() < 1e-6 || nn.dot(radial[i]) < 0.0 {
                nn = radial[i]; // skirt verts and degenerate cases face outward
            }
            nrm[i] = nn;
        }

        // Per-vertex colour (height/slope/lat), with a small position-hashed
        // jitter that now interpolates smoothly instead of per-triangle.
        let col: Vec<[f32; 3]> = (0..nv)
            .map(|i| {
                let cdir = pm.positions[i].normalize();
                let slope = (1.0 - nrm[i].dot(radial[i])).clamp(0.0, 1.0);
                let abs_lat = cdir.y.clamp(-1.0, 1.0).asin().abs();
                terrain_color(elev.height_m(cdir), slope, hashf(local[i]), abs_lat, lunar)
            })
            .collect();

        for tri in pm.indices.chunks(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            m.tri3(
                [local[i0], local[i1], local[i2]],
                [nrm[i0], nrm[i1], nrm[i2]],
                [col[i0], col[i1], col[i2]],
            );
        }
    }
    m
}

/// An asteroid rendered through the LOD quadtree (so detail refines as the
/// camera approaches), centred at the local origin. `cam_local` is the camera
/// position in the asteroid frame; `radius` is the base sphere radius and the
/// `elev` asteroid field adds the lobes/craters on top. Returns local-space
/// vertices (the body sits at the origin, so no floating-origin offset).
pub fn asteroid_terrain(cam_local: DVec3, radius: f64, elev: &Elevation, max_depth: u32) -> Mesh {
    let planet = Planet { radius };
    let amp = (radius * 0.34) as f32; // colour scale (matches the field amplitude)
    let lod = select(&planet, cam_local, 1.6, max_depth);
    let mut m = Mesh::default();
    let n = 9;
    // The true surface point at a direction (same field build_mesh displaced by).
    let surf = |d: DVec3| -> DVec3 { d * (radius + elev.land_height_m(d)) };
    // Analytic normal from the height-field gradient, sampled at a step `eps`
    // (radians) that matches the local mesh spacing. Tying the step to the patch
    // size keeps the normal at the resolution actually being drawn: coarse far
    // patches read big features (no sub-triangle speckle), near patches resolve
    // fine relief. It is a pure function of direction, so patches that share an
    // edge get identical normals - seamless blending across patches and LODs.
    let normal_at = |d: DVec3, eps: f64| -> Vec3 {
        let (t, b) = terrain::cubesphere::tangent_basis(d);
        let p0 = surf(d);
        let pt = surf((d + t * eps).normalize());
        let pb = surf((d + b * eps).normalize());
        let mut nn = (pt - p0).cross(pb - p0).normalize_or_zero();
        if nn.dot(d) < 0.0 {
            nn = -nn;
        }
        if nn.length_squared() < 1e-9 {
            nn = d;
        }
        nn.as_vec3()
    };
    for patch in &lod.patches {
        let skirt = (patch.edge * 0.35).clamp(2.0, 5_000.0);
        let pm = build_mesh(&planet, patch, n, elev, skirt);
        // one vertex spacing of this patch, as an angular step on the unit sphere
        let eps = ((patch.edge / (n as f64 - 1.0)) / radius).clamp(2.0e-4, 0.2);
        let nv = pm.positions.len();
        let local: Vec<Vec3> = pm.positions.iter().map(|&w| w.as_vec3()).collect();
        let radial: Vec<Vec3> = pm.positions.iter().map(|&w| w.normalize_or_zero().as_vec3()).collect();
        let nrm: Vec<Vec3> = pm.positions.iter().map(|&w| normal_at(w.normalize(), eps)).collect();
        let col: Vec<[f32; 3]> = (0..nv)
            .map(|i| {
                let h_frac = ((local[i].length() - radius as f32) / amp).clamp(0.0, 1.0);
                let base = mix3([0.17, 0.14, 0.12], [0.40, 0.36, 0.31], h_frac);
                let slope = (1.0 - nrm[i].dot(radial[i])).clamp(0.0, 1.0);
                let c = mix3(base, [0.30, 0.30, 0.31], slope * 0.5);
                let j = 0.84 + 0.32 * hashf(local[i] * 0.5);
                [c[0] * j, c[1] * j, c[2] * j]
            })
            .collect();
        for tri in pm.indices.chunks(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            m.tri3(
                [local[i0], local[i1], local[i2]],
                [nrm[i0], nrm[i1], nrm[i2]],
                [col[i0], col[i1], col[i2]],
            );
        }
    }
    m
}
