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

    fn plane(&mut self, y: f32, half: f32, col: [f32; 3]) {
        let a = Vec3::new(-half, y, -half);
        let b = Vec3::new(half, y, -half);
        let c = Vec3::new(half, y, half);
        let d = Vec3::new(-half, y, half);
        self.quad(a, b, c, d, Vec3::Y, col);
    }
}

pub struct Scene {
    pub mesh: Mesh,
    /// Y to aim the camera at (mid-stack), in metres.
    pub focus_y: f32,
    /// A good default camera distance, in metres.
    pub cam_dist: f32,
}

const PAD_TOP: f32 = 1.2;
const MOUNT_H: f32 = 2.2; // rocket base sits this far above the pad slab
const PROP_DENSITY: f32 = 1000.0; // kg/m^3, sizes stage height from propellant

/// Build the rocket-view scene from the Pioneer vehicle.
pub fn scene() -> Scene {
    let veh = Vehicle::pioneer();
    let mut m = Mesh::default();

    // pad slab (the planet terrain is the ground; see build_terrain)
    m.bx(Vec3::new(0.0, PAD_TOP * 0.5, 0.0), Vec3::new(9.0, PAD_TOP * 0.5, 9.0), [0.42, 0.42, 0.45]);

    let base_y = PAD_TOP + MOUNT_H;
    let radii = [1.85f32, 1.7];
    let body_cols = [[0.90f32, 0.90, 0.93], [0.72, 0.74, 0.78]];

    // launch mount legs
    for (sx, sz) in [(1.0f32, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        m.bx(
            Vec3::new(sx * 2.3, PAD_TOP + MOUNT_H * 0.5, sz * 2.3),
            Vec3::new(0.35, MOUNT_H * 0.5, 0.35),
            [0.28, 0.29, 0.32],
        );
    }

    let mut y = base_y;
    for (i, stage) in veh.stages.iter().enumerate() {
        let r = radii[i.min(radii.len() - 1)];
        let col = body_cols[i.min(body_cols.len() - 1)];
        let vol = stage.prop as f32 / PROP_DENSITY;
        let h = (vol / (std::f32::consts::PI * r * r)).max(3.0);

        // body
        m.frustum(0.0, 0.0, y, y + h, r, r, 24, col, false, false);
        // a couple of dark bands for visual scale
        m.frustum(0.0, 0.0, y + h * 0.33, y + h * 0.33 + 0.3, r * 1.01, r * 1.01, 24, [0.15, 0.16, 0.18], false, false);

        // engines below this stage's base
        if i == 0 {
            for k in 0..5 {
                let (ex, ez) = if k < 4 {
                    let a = k as f32 * std::f32::consts::FRAC_PI_2;
                    (a.cos() * r * 0.5, a.sin() * r * 0.5)
                } else {
                    (0.0, 0.0)
                };
                m.frustum(ex, ez, y - 1.7, y, 0.4, 0.62, 12, [0.13, 0.13, 0.15], false, true);
            }
            // fins at the four cardinal directions (axis-aligned, no rotation)
            let fy = y + 2.0;
            for (cx, cz, hx, hz) in [
                (r + 0.7, 0.0, 0.9, 0.12),
                (-(r + 0.7), 0.0, 0.9, 0.12),
                (0.0, r + 0.7, 0.12, 0.9),
                (0.0, -(r + 0.7), 0.12, 0.9),
            ] {
                m.bx(Vec3::new(cx, fy, cz), Vec3::new(hx, 1.8, hz), [0.55, 0.10, 0.10]);
            }
        } else {
            m.frustum(0.0, 0.0, y - 1.5, y, 0.5, 0.85, 16, [0.13, 0.13, 0.15], false, true);
        }

        y += h;

        // interstage to the next stage radius (or taper toward payload)
        let next_r = radii.get(i + 1).copied().unwrap_or(r * 0.92);
        m.frustum(0.0, 0.0, y, y + 0.6, r, next_r, 24, [0.18, 0.18, 0.21], false, false);
        y += 0.6;
    }

    // payload section + nose cone
    let pr = radii[radii.len() - 1] * 0.92;
    m.frustum(0.0, 0.0, y, y + 4.0, pr, pr, 24, [0.30, 0.33, 0.42], false, false);
    y += 4.0;
    m.frustum(0.0, 0.0, y, y + 4.5, pr, 0.0, 24, [0.93, 0.93, 0.96], false, false);
    y += 4.5;

    let top = y;
    Scene {
        mesh: m,
        focus_y: top * 0.45,
        cam_dist: top * 1.7,
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

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn hashf(p: Vec3) -> f32 {
    let mut h = (p.x * 127.1 + p.y * 311.7 + p.z * 74.7).sin() * 43758.547;
    h -= h.floor();
    h
}

fn terrain_color(signed_h: f64, slope: f32, jitter: f32) -> [f32; 3] {
    if signed_h <= 0.0 {
        // shallow to deep sea
        let t = ((-signed_h) / 1200.0).clamp(0.0, 1.0) as f32;
        return mix3([0.07, 0.22, 0.34], [0.03, 0.10, 0.20], t);
    }
    let t = (signed_h / 4200.0).clamp(0.0, 1.0) as f32;
    let grass = [0.20, 0.34, 0.15];
    let scrub = [0.36, 0.33, 0.18];
    let snow = [0.88, 0.90, 0.95];
    let mut base = if t < 0.55 {
        mix3(grass, scrub, t / 0.55)
    } else {
        mix3(scrub, snow, (t - 0.55) / 0.45)
    };
    // steep faces read as bare rock
    let rock = [0.32, 0.28, 0.25];
    let steep = ((slope - 0.30) / 0.35).clamp(0.0, 1.0);
    base = mix3(base, rock, steep);
    // micro brightness variation so the ground is not flat
    let b = 0.90 + 0.16 * jitter;
    [base[0] * b, base[1] * b, base[2] * b]
}

/// Build the planet LOD terrain around the spaceport, in the rocket view's
/// local tangent frame (metres, +Y up). Returns the mesh; the local origin is
/// the surface point so the rocket (built at y=0) sits on it.
pub fn build_terrain() -> Mesh {
    let planet = Planet { radius: 6.2e6 };
    let mut elev = Elevation::new(47);

    // Location override (lat,lon degrees) via env, else the spaceport. Lets us
    // verify terrain relief at a mountainous spot without recompiling.
    let (lat_deg, lon_deg) = std::env::var("MTS_TERRAIN_LATLON")
        .ok()
        .and_then(|s| {
            let mut it = s.split(',');
            Some((it.next()?.trim().parse().ok()?, it.next()?.trim().parse().ok()?))
        })
        .unwrap_or((SPACEPORT_LAT_DEG, SPACEPORT_LON_DEG));
    let lat = (lat_deg as f64).to_radians();
    let lon = (lon_deg as f64).to_radians();
    let dir = DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize();
    // Keep the launch pad area flat.
    elev.add_flat_zone(dir, 2500.0, 8000.0, planet.radius);
    let h0 = elev.land_height_m(dir);
    let origin = dir * (planet.radius + h0);

    // tangent basis: up = radial, north toward +Y, east = north x up
    let up = dir;
    let north = (DVec3::Y - up * up.dot(DVec3::Y)).normalize();
    let east = north.cross(up).normalize();
    let to_local = |w: DVec3| -> Vec3 {
        let d = w - origin;
        Vec3::new(d.dot(east) as f32, d.dot(up) as f32, d.dot(north) as f32)
    };

    // DEBUG: global highest-elevation point (to prove terrain has height).
    {
        let mut best = (0.0f64, 0.0f64, f64::MIN);
        let mut la = -80.0;
        while la <= 80.0 {
            let mut lo = -180.0;
            while lo <= 180.0 {
                let llat = (la as f64).to_radians();
                let llon = (lo as f64).to_radians();
                let d = DVec3::new(llat.cos() * llon.cos(), llat.sin(), llat.cos() * llon.sin());
                let h = elev.height_m(d.normalize());
                if h > best.2 {
                    best = (la, lo, h);
                }
                lo += 3.0;
            }
            la += 3.0;
        }
        eprintln!("[terrain] global max elevation {:.0} m at lat {:.1} lon {:.1}", best.2, best.0, best.1);
    }

    // Finer LOD: select as if the camera sits ~60 m over the pad.
    let cam = origin + up * 60.0;
    let lod = select(&planet, cam, 1.4, 18);

    let mut m = Mesh::default();
    let n = 9;
    for patch in &lod.patches {
        let pm = build_mesh(&planet, patch, n, &elev, 120.0);
        for tri in pm.indices.chunks(3) {
            let w0 = pm.positions[tri[0] as usize];
            let w1 = pm.positions[tri[1] as usize];
            let w2 = pm.positions[tri[2] as usize];
            let a = to_local(w0);
            let b = to_local(w1);
            let c = to_local(w2);
            let mut nrm = (b - a).cross(c - a).normalize_or_zero();
            if nrm.y < 0.0 {
                nrm = -nrm; // terrain faces up
            }
            let centroid = (w0 + w1 + w2) / 3.0;
            let slope = 1.0 - nrm.y;
            let col = terrain_color(elev.height_m(centroid.normalize()), slope, hashf(a));
            m.tri(a, b, c, nrm, col);
        }
    }
    m
}
