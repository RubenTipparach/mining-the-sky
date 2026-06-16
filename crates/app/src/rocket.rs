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

    // ground + pad
    m.plane(0.0, 600.0, [0.20, 0.27, 0.15]);
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
