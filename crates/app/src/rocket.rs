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
    // an RCS pod / antenna nub
    m.frustum(br * 0.7, 0.0, y1 - 0.2, y1 + 0.3, 0.18, 0.12, 8, dark, true, true);
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
pub fn rocket_body(veh: &Vehicle, payload_col: [f32; 3]) -> RocketBody {
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

    // payload fairing + nose cone
    let pstart = m.verts.len();
    let pr = radii.last().copied().unwrap_or(1.5) * 0.85;
    m.frustum(0.0, 0.0, y, y + 4.0, pr, pr, 24, payload_col, false, false);
    y += 4.0;
    m.frustum(0.0, 0.0, y, y + 4.0, pr, 0.0, 24, [0.93, 0.93, 0.96], false, false);
    y += 4.0;
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
