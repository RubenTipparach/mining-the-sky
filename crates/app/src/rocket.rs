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

/// The Vehicle Assembly Building: a gantry/hangar the rocket is assembled in,
/// centred at `c` (local metres). Four corner towers, cross beams, a back
/// service tower and a floor pad - recognisable as a building, open at the
/// front so the rocket can roll out.
pub fn hangar(c: Vec3) -> Mesh {
    let mut m = Mesh::default();
    let frame = [0.30, 0.32, 0.36];
    let dark = [0.20, 0.21, 0.24];
    let h = 64.0f32; // tower height
    // floor pad
    m.bx(c + Vec3::new(0.0, 0.4, 0.0), Vec3::new(16.0, 0.4, 16.0), [0.34, 0.34, 0.38]);
    // four corner towers
    for (sx, sz) in [(1.0f32, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        m.bx(
            c + Vec3::new(sx * 13.0, h * 0.5, sz * 13.0),
            Vec3::new(0.8, h * 0.5, 0.8),
            frame,
        );
    }
    // cross beams at three heights (X and Z spanning bars on all four sides)
    for lvl in [0.32f32, 0.62, 0.92] {
        let y = h * lvl;
        for sz in [-1.0f32, 1.0] {
            m.bx(c + Vec3::new(0.0, y, sz * 13.0), Vec3::new(13.0, 0.4, 0.4), dark);
        }
        for sx in [-1.0f32, 1.0] {
            m.bx(c + Vec3::new(sx * 13.0, y, 0.0), Vec3::new(0.4, 0.4, 13.0), dark);
        }
    }
    // back service tower wall (the -Z side, away from the pad which is at +X)
    m.bx(c + Vec3::new(-13.5, h * 0.5, 0.0), Vec3::new(0.5, h * 0.5, 13.0), [0.26, 0.27, 0.30]);
    // roof beams
    for sz in [-1.0f32, 1.0] {
        m.bx(c + Vec3::new(0.0, h, sz * 13.0), Vec3::new(13.0, 0.5, 0.5), frame);
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
        elev.add_flat_zone(spaceport_dir(), 2500.0, 8000.0, PLANET_RADIUS);
    }
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

fn terrain_color(signed_h: f64, slope: f32, jitter: f32, abs_lat: f64) -> [f32; 3] {
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
) -> Mesh {
    let planet = Planet { radius: PLANET_RADIUS };
    let elev = launch_elevation();
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
    for patch in &lod.patches {
        // Skirt depth scales with the patch so coarse far patches still seal.
        let skirt = (patch.edge * 0.3).clamp(80.0, 80_000.0);
        let pm = build_mesh(&planet, patch, n, &elev, skirt);
        for tri in pm.indices.chunks(3) {
            let w0 = pm.positions[tri[0] as usize];
            let w1 = pm.positions[tri[1] as usize];
            let w2 = pm.positions[tri[2] as usize];
            let a = (to_local(w0) - ref_local).as_vec3();
            let b = (to_local(w1) - ref_local).as_vec3();
            let c = (to_local(w2) - ref_local).as_vec3();
            let cdir = ((w0 + w1 + w2) / 3.0).normalize();
            let cdir_l = dir_local(cdir); // outward (radial) in local axes
            let mut nrm = (b - a).cross(c - a).normalize_or_zero();
            if nrm.dot(cdir_l) < 0.0 {
                nrm = -nrm; // face away from the planet centre
            }
            let slope = (1.0 - nrm.dot(cdir_l)).clamp(0.0, 1.0);
            let abs_lat = cdir.y.clamp(-1.0, 1.0).asin().abs();
            let col = terrain_color(elev.height_m(cdir), slope, hashf(a), abs_lat);
            m.tri(a, b, c, nrm, col);
        }
    }
    m
}
