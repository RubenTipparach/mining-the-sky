//! Mining the Sky real-time client.
//!
//! A wgpu/WebGPU app (native + browser via wasm) that renders a live
//! procedural planet and flies an interactive launch-to-orbit. The planet is an
//! orthographic raymarch of the baked worldgen texture; on top of it we draw
//! the `sim` crate's staged ascent and a manual free-flight mode: drag to orbit
//! the camera, scroll to zoom, Space to launch Pioneer I from the seed-47
//! spaceport, F to take manual control and land. This is the start of the
//! Caelum-style renderer; the camera/overlay here is the seam the 3D LOD
//! renderer grows from.
//!
//! The render state is split into `World` (simulation + camera, no GPU) and
//! `Gpu` (pipelines + buffers), so the windowed client and a headless
//! `--shot` screenshot path share the same scene-recording code.

use std::sync::Arc;

use glam::{DVec3, Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

mod flight;
mod launch;
mod mission;
mod rocket;
mod ui;
mod universe;
use flight::{Craft, GravBody, Mode};
use mission::Mission;
use sim::body::CentralBody;

/// Drawable size for the surface. On the web the winit window reports a near
/// zero inner size, so derive it from the canvas client rect times the device
/// pixel ratio (this is what fixes the blank/1x1 browser render).
fn surface_size(window: &Window) -> (u32, u32) {
    #[cfg(target_arch = "wasm32")]
    {
        webx::canvas_size(window)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let s = window.inner_size();
        (s.width.max(1), s.height.max(1))
    }
}

/// Browser glue: status reporting, WebGPU detection, and canvas sizing.
#[cfg(target_arch = "wasm32")]
mod webx {
    use wasm_bindgen::JsValue;
    use winit::platform::web::WindowExtWebSys;
    use winit::window::Window;

    /// Replace the on-page `#hud` text so failures are visible instead of a
    /// blank page.
    pub fn set_status(msg: &str) {
        if let Some(el) = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("hud"))
        {
            el.set_text_content(Some(msg));
        }
    }

    /// True if `navigator.gpu` exists (checked via Reflect so we do not need the
    /// web-sys `Gpu` feature).
    pub fn has_webgpu() -> bool {
        web_sys::window()
            .map(|w| {
                let nav = w.navigator();
                js_sys::Reflect::get(&nav, &JsValue::from_str("gpu"))
                    .map(|v| !v.is_undefined() && !v.is_null())
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Canvas client size in physical pixels (CSS pixels times devicePixelRatio).
    pub fn canvas_size(window: &Window) -> (u32, u32) {
        if let Some(canvas) = window.canvas() {
            let dpr = web_sys::window().map(|w| w.device_pixel_ratio()).unwrap_or(1.0);
            let w = (canvas.client_width() as f64 * dpr).round() as u32;
            let h = (canvas.client_height() as f64 * dpr).round() as u32;
            (w.max(1), h.max(1))
        } else {
            (1, 1)
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    cam_pos: [f32; 4],
    cam_x: [f32; 4],
    cam_y: [f32; 4],
    cam_z: [f32; 4],
    sun: [f32; 4],
    home: [f32; 4],
    moon: [f32; 4],
    sunbody: [f32; 4],  // star A: xyz centre, w radius (Mm)
    sunbody2: [f32; 4], // star B (red): xyz centre, w radius (Mm)
    params: [f32; 4],   // x=tan(fov/2), y=aspect, z=time, w=planet count
    res: [f32; 4],      // x,y=resolution, z=moon count
    planets: [[f32; 4]; 16],    // xyz centre, w radius (Mm)
    planet_col: [[f32; 4]; 16], // rgb colour
    moons: [[f32; 4]; 8],       // nearest moons: xyz centre, w radius (Mm)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OverlayVertex {
    pos: [f32; 2],
    color: [f32; 3],
}

/// Thruster-FX billboard vertex (flame + smoke), drawn by the `fx` pipeline.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FxVertex {
    pos: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
    kind: f32, // 0 = flame (additive), 1 = smoke (premultiplied over)
}

const OVERLAY_CAP: u64 = 8192;
const HUD_CAP: u64 = 40000;
/// Thruster-FX billboards (flame + smoke particles).
const FX_CAP: u64 = 60000;
/// Dynamic rocket-view geometry (pad + rocket + spent booster).
const DYN_MESH_CAP: u64 = 40000;
/// Full-planet LOD terrain (rebuilt as the camera moves).
const TERRAIN_CAP: u64 = 500_000;

/// Render-space length unit for the system view: 1000 km.
const MM: f32 = 1.0e6;

/// Depth format for the rocket view's mesh pass.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Clone, Copy, PartialEq)]
enum View {
    /// Orbital map: perspective view of the bodies + launch/orbit trajectories.
    Map,
    /// 3D surface: the rocket on the pad over LOD terrain.
    Rocket,
}


use universe::{Body, Kind, Universe};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshUniforms {
    viewproj: [[f32; 4]; 4],
    sun: [f32; 4],
    /// params.x = logarithmic-depth Fcoef = 1 / log2(far + 1).
    params: [f32; 4],
    /// rgb = horizon haze colour, w = fog density (1/visibility metres).
    fog: [f32; 4],
}

/// A perspective sky for the rocket view (drawn behind the terrain).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniforms {
    /// Camera basis in WORLD (planet-centred) coords for per-pixel view rays.
    right: [f32; 4],
    up: [f32; 4],
    fwd: [f32; 4],
    /// World sun direction (xyz).
    sun: [f32; 4],
    /// Camera position relative to the planet centre (world metres).
    cam: [f32; 4],
    /// x = tan(fov/2), y = aspect, z = planet radius, w = atmosphere top radius.
    params: [f32; 4],
}

/// Far plane for the rocket-view log-depth buffer (m): beyond planet diameter.
const LOG_DEPTH_FAR: f32 = 2.0e7;

/// Horizon haze colour shared by the sky and the terrain aerial-perspective fog.
const HORIZON: [f32; 3] = [0.74, 0.82, 0.93];

/// A perspective camera for the orbital map. The position is f64 (Mm) so the
/// full-scale system stays precise; projection is camera-relative.
struct SystemCamera {
    pos: DVec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
    fovscale: f32,
    aspect: f32,
}

impl SystemCamera {
    /// Project a world point (Mm, f64) to clip space, or `None` if behind the
    /// camera. Camera-relative, so f32 precision holds near the focused body.
    fn project(&self, p: DVec3) -> Option<[f32; 2]> {
        let rel = (p - self.pos).as_vec3();
        let fd = rel.dot(self.forward);
        if fd <= 0.0 {
            return None;
        }
        let x = rel.dot(self.right) / fd;
        let y = rel.dot(self.up) / fd;
        Some([x / (self.aspect * self.fovscale), y / self.fovscale])
    }
}

// ---------------------------------------------------------------------------
// World: simulation, camera, and flight state (no GPU objects). Also owns the
// per-frame geometry/uniform builders so both render paths share them.
// ---------------------------------------------------------------------------

/// A jettisoned booster tumbling away after staging, integrated ballistically
/// in the rocket-view local frame (metres) for a few seconds of spectacle.
struct SepBooster {
    pos: Vec3,
    vel: Vec3,
    rot: Quat,
    spin: Vec3,
    age: f32,
}

/// One exhaust smoke puff, advected in the rocket-view local frame. Emitted at
/// the nozzle and left behind as the rocket flies on, so it trails and fades.
struct Smoke {
    pos: Vec3,
    vel: Vec3,
    age: f32,
    life: f32,
    size0: f32,
    seed: f32,
}

struct World {
    mission: Mission,
    body: CentralBody,
    flight: Option<Craft>,
    launched: bool,
    clock: f32, // mission-elapsed seconds
    warp: f32,

    // player-controlled launch (KSP-style); replaces the on-rails ascent when
    // the player flies it from the pad in the rocket view.
    launch: Option<launch::Rocket>,
    rocket_body: rocket::RocketBody,
    pad_mesh: rocket::Mesh,
    sep: Option<SepBooster>,
    smoke: Vec<Smoke>,
    smoke_accum: f32, // fractional particle spawn carry
    anim: f32,        // FX animation clock (seconds)
    // launch-site tangent frame (home-centred metres): origin + up/east/north.
    launch_origin: DVec3,
    launch_up: DVec3,
    launch_east: DVec3,
    launch_north: DVec3,
    // Floating-origin reference (launch-tangent metres). The whole rocket-view
    // scene is rendered relative to this point, snapped near the camera and
    // updated (with a terrain rebuild) when the camera moves far from it.
    ref_local: DVec3,
    terrain_dirty: bool,
    terrain_verts: Vec<rocket::MeshVertex>,
    terrain_count: u32,

    // orbital map: a perspective camera framing the focused body.
    view: View,
    sys_az: f32,
    sys_el: f32,
    sys_dist: f64,    // camera distance from the focus point (Mm)
    sys_focus: DVec3, // focused body position (Mm), updated each frame
    sys_time: f64,    // simulation seconds (drives on-rails orbits)
    home_radius_mm: f32,
    moon_center_mm: Vec3,
    moon_radius_mm: f32,

    // moon physics (metres, the frame the flight model integrates in)
    moon_center_m: DVec3,
    moon_mu: f64,
    moon_radius_m: f64,

    // rocket view: an orbit camera framing the 3D vehicle on the pad (metres).
    rocket_az: f32,
    rocket_el: f32,
    rocket_dist: f32,
    rocket_focus_y: f32,

    universe: Universe,
    /// Indices into `universe.bodies` of the navigable bodies (stars + planets).
    nav: Vec<usize>,
    /// Index (into `universe.bodies`) of the focused body.
    focus: usize,
    /// egui body-browser search text.
    ui_search: String,
}

impl World {
    fn new() -> World {
        let mission = Mission::pioneer_from_spaceport();
        let body = CentralBody::home();
        let rocket_body = rocket::rocket_body();
        let rocket_frame = (rocket_body.focus_y, rocket_body.cam_dist);
        let (launch_origin, launch_up, launch_east, launch_north) = rocket::launch_frame();
        let home_radius_mm = (body.radius as f32) / MM;
        // A fictional moon: ~0.27 home radii, parked off to one side. Distance is
        // compressed from the real ~60 radii so both bodies frame nicely in one
        // shot; it will move to a real orbit once the system view is to-scale.
        let moon_radius_mm = home_radius_mm * 0.27;
        let moon_center_mm = Vec3::new(88.0, 0.0, 8.0);
        let moon_center_m = DVec3::new(
            moon_center_mm.x as f64 * MM as f64,
            moon_center_mm.y as f64 * MM as f64,
            moon_center_mm.z as f64 * MM as f64,
        );
        let moon_radius_m = moon_radius_mm as f64 * MM as f64;
        let moon_mu = 1.7 * moon_radius_m * moon_radius_m; // ~1.7 m/s^2 surface gravity
        let mut w = World {
            launched: false,
            clock: 0.0,
            warp: 1.0,
            mission,
            body,
            flight: None,
            launch: None,
            rocket_body,
            pad_mesh: rocket::pad_and_mount(),
            sep: None,
            smoke: Vec::new(),
            smoke_accum: 0.0,
            anim: 0.0,
            launch_origin,
            launch_up,
            launch_east,
            launch_north,
            ref_local: DVec3::ZERO,
            terrain_dirty: true,
            terrain_verts: Vec::new(),
            terrain_count: 0,
            view: View::Rocket,
            sys_az: 1.4,
            sys_el: 0.30,
            sys_dist: 120.0,
            sys_focus: DVec3::ZERO,
            sys_time: 0.0,
            home_radius_mm,
            moon_center_mm,
            moon_radius_mm,
            moon_center_m,
            moon_mu,
            moon_radius_m,
            rocket_az: 4.97,
            rocket_el: 0.12,
            rocket_dist: rocket_frame.1,
            rocket_focus_y: rocket_frame.0,
            universe: Universe { bodies: Vec::new() },
            nav: Vec::new(),
            focus: 0,
            ui_search: String::new(),
        };
        // Generate the full Kepler-47 system; the landable moon is injected as
        // home's first moon so the map and the flight sim agree.
        w.universe = universe::generate(47, w.home_radius_mm);
        w.nav = w
            .universe
            .bodies
            .iter()
            .enumerate()
            .filter(|(_, b)| matches!(b.kind, Kind::StarA | Kind::StarB | Kind::Planet))
            .map(|(i, _)| i)
            .collect();
        // default focus: the home world (body index)
        w.focus = w.universe.home_index();
        w.apply_focus();
        w
    }

    /// The currently focused body.
    fn focus_body(&self) -> &Body {
        &self.universe.bodies[self.focus]
    }

    fn focus_label(&self) -> &str {
        self.focus_body().name.as_str()
    }

    /// Pick the nearest body to a screen position (any body, incl. moons /
    /// asteroids / comets). Returns its body index.
    fn pick_body(&self, res: (f32, f32), cx: f32, cy: f32) -> Option<usize> {
        let cam = self.system_camera(res.0 / res.1.max(1.0));
        let ndc = [cx / res.0 * 2.0 - 1.0, 1.0 - cy / res.1 * 2.0];
        let mut best = None;
        let mut best_d = 0.05f32; // clip-space threshold
        for i in 0..self.universe.bodies.len() {
            if let Some(c) = cam.project(self.universe.position(i, self.sys_time)) {
                let d = ((c[0] - ndc[0]).powi(2) + (c[1] - ndc[1]).powi(2)).sqrt();
                if d < best_d {
                    best_d = d;
                    best = Some(i);
                }
            }
        }
        best
    }

    /// Cycle the orbital-map focus through the navigable bodies.
    fn cycle_focus(&mut self) {
        let cur = self.nav.iter().position(|&i| i == self.focus).unwrap_or(0);
        let next = self.nav[(cur + 1) % self.nav.len()];
        self.set_focus(next);
    }

    fn set_focus(&mut self, body_idx: usize) {
        if body_idx < self.universe.bodies.len() {
            self.focus = body_idx;
            self.apply_focus();
        }
    }

    /// World position of the focused body at the current sim time.
    fn focus_pos(&self) -> DVec3 {
        self.universe.position(self.focus, self.sys_time)
    }

    fn apply_focus(&mut self) {
        let radius = self.focus_body().radius;
        self.sys_focus = self.focus_pos();
        self.sys_dist = (radius * 4.0).max(2.0);
    }

    fn grav_bodies(&self) -> Vec<GravBody> {
        vec![GravBody {
            center: self.moon_center_m,
            mu: self.moon_mu,
            radius: self.moon_radius_m,
            name: "MOON",
        }]
    }

    fn toggle_view(&mut self) {
        self.view = match self.view {
            View::Map => View::Rocket,
            View::Rocket => View::Map,
        };
    }

    /// Rocket-view camera, floating-origin: eye + look target are returned
    /// relative to `ref_local` (small f32 near the camera), plus the basis and
    /// tan(fov/2). The whole rocket-view scene is uploaded in this same frame.
    fn rocket_camera(&self, aspect: f32) -> (Vec3, Vec3, Vec3, Vec3, Vec3, f32) {
        let eye = self.rel(self.camera_eye_local());
        let target = self.rel(self.camera_look_local());
        let fwd = (target - eye).normalize_or_zero();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd).normalize();
        let tan = (50f32.to_radians() * 0.5).tan();
        let _ = aspect;
        (eye, target, right, up, fwd, tan)
    }

    /// View-projection + sun + fog for the rocket view (local scene, metres).
    fn mesh_uniforms(&self, res: [f32; 2]) -> MeshUniforms {
        let aspect = res[0] / res[1].max(1.0);
        let (eye, target, _r, _u, _f, _t) = self.rocket_camera(aspect);
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        // wide near/far range; the logarithmic depth buffer keeps precision.
        let proj = Mat4::perspective_rh(50f32.to_radians(), aspect, 0.1, LOG_DEPTH_FAR);
        let vp = proj * view;
        let fcoef = 1.0 / (LOG_DEPTH_FAR + 1.0).log2();
        MeshUniforms {
            viewproj: vp.to_cols_array_2d(),
            sun: [0.40, 0.72, 0.55, 0.0],
            params: [fcoef, self.anim, 0.0, 0.0],
            // Light aerial haze only; the atmosphere shader does the real work so
            // the planet keeps its colour from altitude.
            fog: [HORIZON[0], HORIZON[1], HORIZON[2], 1.0 / 160_000.0],
        }
    }

    /// Sky/atmosphere uniforms for the rocket view: the camera basis and
    /// position in planet-centred world coords so the shader can ray-march the
    /// atmosphere shell.
    fn sky_uniforms(&self, res: [f32; 2]) -> SkyUniforms {
        let aspect = res[0] / res[1].max(1.0);
        let (_eye, _target, right, up, fwd, tan) = self.rocket_camera(aspect);
        // local camera basis -> world (planet-centred) via the launch frame.
        let to_world_dir = |d: Vec3| -> [f32; 4] {
            let w = self.launch_east * d.x as f64
                + self.launch_up * d.y as f64
                + self.launch_north * d.z as f64;
            [w.x as f32, w.y as f32, w.z as f32, 0.0]
        };
        let cam_world = self.cam_world(self.camera_eye_local());
        // Sun in world coords (the mesh shader's local (0.40,0.72,0.55)).
        let sun = (self.launch_east * 0.40 + self.launch_up * 0.72 + self.launch_north * 0.55)
            .normalize();
        let r_atm = self.body.radius + 90_000.0;
        SkyUniforms {
            right: to_world_dir(right),
            up: to_world_dir(up),
            fwd: to_world_dir(fwd),
            sun: [sun.x as f32, sun.y as f32, sun.z as f32, 0.0],
            cam: [cam_world.x as f32, cam_world.y as f32, cam_world.z as f32, 0.0],
            params: [tan, aspect, self.body.radius as f32, r_atm as f32],
        }
    }

    /// System-view perspective camera: position + basis + tan(fov/2), all in Mm.
    fn system_camera(&self, aspect: f32) -> SystemCamera {
        let dir = Vec3::new(
            self.sys_el.cos() * self.sys_az.cos(),
            self.sys_el.sin(),
            self.sys_el.cos() * self.sys_az.sin(),
        );
        let cam_pos = self.sys_focus + dir.as_dvec3() * self.sys_dist;
        let forward = (self.sys_focus - cam_pos).as_vec3().normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let fov: f32 = 42.0_f32.to_radians();
        SystemCamera {
            pos: cam_pos,
            right,
            up,
            forward,
            fovscale: (fov * 0.5).tan(),
            aspect,
        }
    }

    /// Perspective camera + body uniforms for the orbital map. All body centres
    /// are passed camera-relative (floating origin) so f32 holds at AU scale.
    fn scene_uniforms(&self, res: [f32; 2], time: f32) -> SceneUniforms {
        let aspect = res[0] / res[1].max(1.0);
        let cam = self.system_camera(aspect);
        let t = self.sys_time;
        let rel = |p: DVec3| -> Vec3 { (p - cam.pos).as_vec3() };

        let u = &self.universe;
        let home_i = u.home_index();
        let home_r = rel(u.position(home_i, t));
        let bary_r = rel(DVec3::ZERO);

        // find star indices (0,1 by construction)
        let star_a = rel(u.position(0, t));
        let star_b = rel(u.position(1, t));

        // home's first moon as the rendered "moon"
        let moon_r = u
            .bodies
            .iter()
            .position(|b| b.orbit.parent == Some(home_i))
            .map(|mi| (rel(u.position(mi, t)), u.bodies[mi].radius as f32))
            .unwrap_or((Vec3::ZERO, 0.0));

        // circumbinary planets (excluding the textured home world)
        let mut planets = [[0.0f32; 4]; 16];
        let mut planet_col = [[0.0f32; 4]; 16];
        let mut n = 0usize;
        for (i, b) in u.bodies.iter().enumerate() {
            if b.kind != Kind::Planet || b.is_home || n >= 16 {
                continue;
            }
            let p = rel(u.position(i, t));
            planets[n] = [p.x, p.y, p.z, b.radius as f32];
            planet_col[n] = [b.color[0], b.color[1], b.color[2], 1.0];
            n += 1;
        }

        // nearest moons to the camera, ray-marched as lit spheres up close
        let mut moons = [[0.0f32; 4]; 8];
        let mut cand: Vec<(f64, usize)> = u
            .bodies
            .iter()
            .enumerate()
            .filter(|(_, b)| b.kind == Kind::Moon)
            .map(|(i, _)| ((u.position(i, t) - cam.pos).length(), i))
            .collect();
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut mc = 0usize;
        for &(_, i) in cand.iter().take(8) {
            let p = rel(u.position(i, t));
            moons[mc] = [p.x, p.y, p.z, u.bodies[i].radius as f32];
            mc += 1;
        }

        SceneUniforms {
            cam_pos: [0.0, 0.0, 0.0, 0.0], // camera at origin (floating origin)
            cam_x: [cam.right.x, cam.right.y, cam.right.z, 0.0],
            cam_y: [cam.up.x, cam.up.y, cam.up.z, 0.0],
            cam_z: [cam.forward.x, cam.forward.y, cam.forward.z, 0.0],
            sun: [bary_r.x, bary_r.y, bary_r.z, 0.0], // barycentre (light source)
            home: [home_r.x, home_r.y, home_r.z, self.home_radius_mm],
            moon: [moon_r.0.x, moon_r.0.y, moon_r.0.z, moon_r.1],
            sunbody: [star_a.x, star_a.y, star_a.z, u.bodies[0].radius as f32],
            sunbody2: [star_b.x, star_b.y, star_b.z, u.bodies[1].radius as f32],
            params: [cam.fovscale, aspect, time, n as f32],
            res: [res[0], res[1], mc as f32, 0.0],
            planets,
            planet_col,
            moons,
        }
    }

    /// Advance simulation by a real frame dt (seconds).
    fn advance(&mut self, frame_dt: f32) {
        // On-rails orbital clock: warp is the time scale (1x .. 10000x).
        self.sys_time += frame_dt as f64 * self.warp as f64;
        // keep the camera following the (moving) focused body
        self.sys_focus = self.focus_pos();

        let bodies = self.grav_bodies();
        if let Some(craft) = self.flight.as_mut() {
            let dt_sim = frame_dt * self.warp.min(8.0);
            craft.integrate(&self.body, &bodies, dt_sim as f64);
        } else if self.launched {
            self.clock += frame_dt * self.warp;
        }

        // player-controlled launch + any tumbling spent booster
        if let Some(rk) = self.launch.as_mut() {
            rk.just_staged = None;
            let dt = (frame_dt * self.warp).min(2.0);
            rk.integrate(&self.body, dt as f64);
        }
        let fx_dt = (frame_dt * self.warp).min(0.5);
        self.anim += fx_dt;
        self.advance_sep(frame_dt * self.warp.min(8.0));
        self.advance_smoke(fx_dt);

        // Floating-origin reference: snap near the camera and rebuild the planet
        // terrain when the camera has moved far from the last reference (also
        // refreshes the LOD as the rocket climbs). Threshold grows with altitude
        // so we rebuild often near the ground and rarely high up.
        if self.view == View::Rocket {
            let eye = self.camera_eye_local();
            let alt = self.cam_world(eye).length() - self.body.radius;
            let thresh = (alt.abs() * 0.04).clamp(25.0, 50_000.0);
            if self.terrain_verts.is_empty() || (eye - self.ref_local).length() > thresh {
                self.ref_local = eye;
                self.rebuild_terrain();
                self.terrain_dirty = true; // upload pending
            }
        }
    }

    /// World->local point in the launch-tangent frame (f64).
    fn to_local_d(&self, w: DVec3) -> DVec3 {
        let d = w - self.launch_origin;
        DVec3::new(d.dot(self.launch_east), d.dot(self.launch_up), d.dot(self.launch_north))
    }
    /// World->local direction (f64, no translation).
    fn dir_to_local_d(&self, d: DVec3) -> DVec3 {
        DVec3::new(d.dot(self.launch_east), d.dot(self.launch_up), d.dot(self.launch_north))
    }
    /// Local->world point (f64).
    fn cam_world(&self, local: DVec3) -> DVec3 {
        self.launch_origin
            + self.launch_east * local.x
            + self.launch_up * local.y
            + self.launch_north * local.z
    }
    /// `world - ref_local` collapsed to f32 (the floating-origin upload form).
    fn rel(&self, local: DVec3) -> Vec3 {
        (local - self.ref_local).as_vec3()
    }

    /// The rocket the camera frames (its local position, f64).
    fn cam_target_local(&self) -> DVec3 {
        match self.launch.as_ref() {
            Some(rk) => self.to_local_d(rk.r),
            None => DVec3::new(0.0, self.rocket_body.base_y as f64, 0.0),
        }
    }

    /// The point the rocket-view camera looks at (launch-tangent metres, f64).
    fn camera_look_local(&self) -> DVec3 {
        let target = self.cam_target_local();
        match self.launch.as_ref() {
            Some(rk) => {
                // Low and slow: look near the base so the pad + smoke stay framed;
                // ease up the stack as the rocket climbs away.
                let axis = self.dir_to_local_d(rk.point_dir());
                let f = self.rocket_focus_y as f64 * (target.y / 120.0).clamp(0.25, 1.0);
                target + axis * f
            }
            None => DVec3::new(0.0, self.rocket_focus_y as f64, 0.0),
        }
    }

    /// Camera eye in launch-tangent metres (f64).
    fn camera_eye_local(&self) -> DVec3 {
        let tgt = self.camera_look_local();
        let dir = DVec3::new(
            (self.rocket_el.cos() * self.rocket_az.cos()) as f64,
            self.rocket_el.sin() as f64,
            (self.rocket_el.cos() * self.rocket_az.sin()) as f64,
        );
        let mut eye = tgt + dir * self.rocket_dist as f64;
        // keep above the local ground plane
        let ground = self.cam_world(eye).length() - self.body.radius;
        if ground < 2.0 {
            eye.y += 2.0 - ground;
        }
        eye
    }

    /// Emit + advect exhaust smoke particles in the local frame.
    fn advance_smoke(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        // advect + age existing puffs (light buoyancy, gentle drag)
        for s in self.smoke.iter_mut() {
            s.age += dt;
            s.vel *= 1.0 - (1.5 * dt).min(0.9);
            s.vel.y += 1.2 * dt; // buoyant rise
            s.pos += s.vel * dt;
        }
        self.smoke.retain(|s| s.age < s.life);

        // emit at the nozzle while the active stage is burning
        let Some(rk) = self.launch.as_ref() else {
            self.smoke.clear();
            return;
        };
        let thrust_frac = if rk.live_thrust() > 0.0 { rk.throttle as f32 } else { 0.0 };
        if thrust_frac <= 0.0 {
            return;
        }
        let alt = rk.altitude(&self.body);
        // air density fraction: lots of smoke low down, little in the thin upper
        // atmosphere, none in vacuum (the flame remains regardless).
        let dens = (-(alt / 9000.0)).exp().clamp(0.0, 1.0) as f32;
        let on_pad = alt < 30.0;
        let base = self.to_local(rk.r);
        let down = -self.dir_to_local(rk.point_dir()); // exhaust direction
        let nozzle = base + down * 1.5;
        let er = self.rocket_body.engine_r * if rk.stage_base == 0 { 1.0 } else { 0.45 };

        // spawn rate: heavy at the pad (the ground billow), thinning with density.
        let rate = thrust_frac * (8.0 + 120.0 * dens) * if on_pad { 2.5 } else { 1.0 };
        self.smoke_accum += rate * dt;
        let n = self.smoke_accum.floor() as i32;
        self.smoke_accum -= n as f32;
        let mut seed = (self.anim * 977.0).to_bits();
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let ground = Vec3::new(base.x, rocket::PAD_TOP, base.z);
        for _ in 0..n.min(40) {
            let jit = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
            if on_pad {
                // exhaust deflects off the pad: billow outward + up from the ground
                let a = rnd() * std::f32::consts::TAU;
                let out = Vec3::new(a.cos(), 0.12, a.sin());
                self.smoke.push(Smoke {
                    pos: ground + Vec3::new(jit.x, jit.y.abs() * 0.5, jit.z) * 3.0,
                    vel: out * (9.0 + rnd() * 12.0) + Vec3::Y * (2.0 + rnd() * 3.0),
                    age: 0.0,
                    life: 2.4 + rnd() * 1.8,
                    size0: 3.0 + rnd() * 2.5,
                    seed: rnd(),
                });
            } else {
                self.smoke.push(Smoke {
                    pos: nozzle + jit * (er * 0.8),
                    vel: down * (6.0 + rnd() * 6.0) + jit * 3.0,
                    age: 0.0,
                    life: 1.1 + rnd() * 1.2,
                    size0: er * (0.8 + rnd() * 0.7),
                    seed: rnd(),
                });
            }
        }
        // keep the particle count bounded
        if self.smoke.len() > 900 {
            let drop = self.smoke.len() - 900;
            self.smoke.drain(0..drop);
        }
    }

    /// World->local point in the rocket-view tangent frame (metres).
    fn to_local(&self, w: DVec3) -> Vec3 {
        let d = w - self.launch_origin;
        Vec3::new(
            d.dot(self.launch_east) as f32,
            d.dot(self.launch_up) as f32,
            d.dot(self.launch_north) as f32,
        )
    }

    /// World->local direction (no translation).
    fn dir_to_local(&self, d: DVec3) -> Vec3 {
        Vec3::new(
            d.dot(self.launch_east) as f32,
            d.dot(self.launch_up) as f32,
            d.dot(self.launch_north) as f32,
        )
        .normalize_or_zero()
    }

    /// Ignite the full stack on the pad and begin a player-controlled launch.
    fn ignite_launch(&mut self) {
        let veh = sim::vehicle::Vehicle::pioneer();
        // base of the rocket sits `base_y` above the pad surface, on the mount.
        let r = self.launch_origin + self.launch_up * self.rocket_body.base_y as f64;
        // Co-moving launch frame: ignite at rest relative to the (fixed) pad and
        // terrain. We drop the planet's surface-rotation boost so the rocket
        // doesn't drift sideways out of the local scene; the player flies the
        // gravity turn from a standstill.
        let v = DVec3::ZERO;
        let mut rk = launch::Rocket::on_pad(&veh, r, v, self.launch_up, self.launch_east);
        rk.throttle = 1.0;
        self.launch = Some(rk);
        self.sep = None;
        self.smoke.clear();
        log::info!("Ignition: Pioneer I - throttle up, pitch over, stage when dry");
    }

    /// Jettison the active stage; spawn the spent booster tumbling away.
    fn stage_launch(&mut self) {
        // capture the booster pose + velocity (immutable borrow) before removal
        let (r, pd, v, can_stage) = match self.launch.as_ref() {
            Some(rk) => (rk.r, rk.point_dir(), rk.v, rk.stages.len() > 1),
            None => return,
        };
        if !can_stage {
            return;
        }
        let base_local = self.to_local(r);
        let pd_local = self.dir_to_local(pd);
        let rot = Quat::from_rotation_arc(Vec3::Y, pd_local);
        let vel_local = self.dir_to_local_vec(v);
        self.launch.as_mut().unwrap().jettison();
        self.sep = Some(SepBooster {
            pos: base_local,
            // a gentle retro push so it falls back below the climbing upper stage
            vel: vel_local - pd_local * 12.0,
            rot,
            spin: Vec3::new(0.7, 0.15, 1.1),
            age: 0.0,
        });
    }

    /// World->local velocity vector (unnormalised).
    fn dir_to_local_vec(&self, d: DVec3) -> Vec3 {
        Vec3::new(
            d.dot(self.launch_east) as f32,
            d.dot(self.launch_up) as f32,
            d.dot(self.launch_north) as f32,
        )
    }

    fn reset_launch(&mut self) {
        self.launch = None;
        self.sep = None;
        self.smoke.clear();
    }

    /// Transform a vertex range of the rocket body by pose `q`/`base` (local f64)
    /// into `out`, camera-relative to `ref_local` (floating origin).
    fn xform_into(
        &self,
        out: &mut Vec<rocket::MeshVertex>,
        range: std::ops::Range<usize>,
        q: Quat,
        base: DVec3,
    ) {
        for v in &self.rocket_body.mesh.verts[range] {
            let local = base + (q * Vec3::from(v.pos)).as_dvec3();
            let n = q * Vec3::from(v.normal);
            out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: n.into(), color: v.color });
        }
    }

    /// Rebuild the full-planet LOD terrain, camera-relative to the current
    /// `ref_local`, refined toward the camera. Called when the reference moves.
    fn rebuild_terrain(&mut self) {
        let cam_world = self.cam_world(self.camera_eye_local());
        let m = rocket::planet_terrain(
            cam_world,
            self.ref_local,
            self.launch_origin,
            self.launch_up,
            self.launch_east,
            self.launch_north,
            19,
        );
        self.terrain_count = (m.verts.len() as u64).min(TERRAIN_CAP) as u32;
        self.terrain_verts = m.verts;
    }

    /// Per-frame dynamic rocket-view geometry, camera-relative (floating origin):
    /// the pad + mount, the rocket at its current pose, and any tumbling spent
    /// booster. The full planet terrain lives in its own (rebuilt-on-move) buffer.
    fn build_dynamic_mesh(&self) -> Vec<rocket::MeshVertex> {
        let rb = &self.rocket_body;
        let mut out: Vec<rocket::MeshVertex> = Vec::new();

        // static pad + mount (camera-relative)
        for v in &self.pad_mesh.verts {
            let local = Vec3::from(v.pos).as_dvec3();
            out.push(rocket::MeshVertex {
                pos: self.rel(local).into(),
                normal: v.normal,
                color: v.color,
            });
        }

        // current rocket pose (resting on the pad when not launched)
        let (base_local, quat, on_booster) = match self.launch.as_ref() {
            Some(rk) => {
                let base = self.to_local_d(rk.r);
                let q = Quat::from_rotation_arc(Vec3::Y, self.dir_to_local(rk.point_dir()));
                (base, q, rk.stage_base == 0)
            }
            None => (DVec3::new(0.0, rb.base_y as f64, 0.0), Quat::IDENTITY, true),
        };

        self.xform_into(&mut out, rb.upper.clone(), quat, base_local);
        if on_booster {
            self.xform_into(&mut out, rb.booster.clone(), quat, base_local);
        }

        // tumbling spent booster
        if let Some(s) = self.sep.as_ref() {
            for v in &rb.mesh.verts[rb.booster.clone()] {
                let local = s.pos.as_dvec3() + (s.rot * Vec3::from(v.pos)).as_dvec3();
                let n = s.rot * Vec3::from(v.normal);
                out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: n.into(), color: v.color });
            }
        }

        out
    }

    /// Thruster FX billboards for the rocket view: an emissive flame at the
    /// active nozzle (axis-aligned cards facing the camera) and the smoke-
    /// particle trail (camera-facing puffs). `right`/`up` are the camera basis.
    fn build_fx(&self, eye: Vec3, right: Vec3, up: Vec3) -> Vec<FxVertex> {
        let mut out: Vec<FxVertex> = Vec::new();

        // ---- exhaust flame at the active nozzle ----
        if let Some(rk) = self.launch.as_ref() {
            let tf = if rk.live_thrust() > 0.0 { rk.throttle as f32 } else { 0.0 };
            if tf > 0.0 {
                let down = -self.dir_to_local(rk.point_dir()); // flame opposes thrust
                let er = self.rocket_body.engine_r * if rk.stage_base == 0 { 1.0 } else { 0.5 };
                // nozzle, camera-relative (floating origin)
                let nozzle = self.rel(self.to_local_d(rk.r) + down.as_dvec3() * 1.2);
                let len = (if rk.stage_base == 0 { 7.0 } else { 3.6 }) * (0.6 + 0.4 * tf);
                // axis billboard: width axis is perpendicular to the flame and to
                // the view ray, so the card faces the camera.
                let view = (nozzle - eye).normalize_or_zero();
                let mut w_axis = down.cross(view).normalize_or_zero();
                if w_axis.length_squared() < 1e-4 {
                    w_axis = right;
                }
                let mut card = |length: f32, half_w: f32, seed: f32, inten: f32| {
                    let tip = nozzle + down * length;
                    let wn = w_axis * half_w;
                    let wt = w_axis * (half_w * 0.22);
                    let col = [seed, inten, 0.0, 0.0];
                    let q = [
                        (nozzle - wn, [0.0f32, 0.0]),
                        (nozzle + wn, [1.0, 0.0]),
                        (tip + wt, [1.0, 1.0]),
                        (tip - wt, [0.0, 1.0]),
                    ];
                    for &i in &[0usize, 1, 2, 0, 2, 3] {
                        out.push(FxVertex { pos: q[i].0.into(), uv: q[i].1, color: col, kind: 0.0 });
                    }
                };
                card(len, er * 1.35, 0.10, tf); // orange body
                card(len * 0.55, er * 0.7, 0.63, tf * 1.3); // white-hot core
            }
        }

        // ---- smoke-particle trail (camera-facing billboards) ----
        for s in &self.smoke {
            let t = (s.age / s.life).clamp(0.0, 1.0);
            let size = s.size0 * (1.0 + t * 3.0);
            let fade_in = (s.age / 0.15).clamp(0.0, 1.0);
            let alpha = fade_in * (1.0 - t) * 0.5;
            if alpha <= 0.01 {
                continue;
            }
            let g = 0.85 - 0.4 * t; // cools/darkens with age
            let col = [g, g, g * 1.02, alpha];
            let r = right * size;
            let u = up * size;
            let c = self.rel(s.pos.as_dvec3()); // camera-relative (floating origin)
            let q = [
                (c - r - u, [0.0f32, 0.0]),
                (c + r - u, [1.0, 0.0]),
                (c + r + u, [1.0, 1.0]),
                (c - r + u, [0.0, 1.0]),
            ];
            for &i in &[0usize, 1, 2, 0, 2, 3] {
                out.push(FxVertex { pos: q[i].0.into(), uv: q[i].1, color: col, kind: 1.0 });
            }
        }
        out
    }

    /// Integrate the jettisoned booster (local frame, ~9.2 m/s^2 down).
    fn advance_sep(&mut self, dt: f32) {
        if let Some(s) = self.sep.as_mut() {
            s.age += dt;
            s.vel.y -= 9.2 * dt;
            s.pos += s.vel * dt;
            s.rot = (Quat::from_scaled_axis(s.spin * dt) * s.rot).normalize();
            if s.age > 9.0 {
                self.sep = None;
            }
        }
    }

    fn toggle_launch(&mut self) {
        self.launched = !self.launched;
        self.clock = 0.0;
        if self.launched {
            log::info!("Liftoff: Pioneer I");
        }
    }

    fn toggle_flight(&mut self) {
        if self.flight.is_some() {
            self.flight = None;
            return;
        }
        let (r, v) = if self.launched && self.mission.reached && self.clock > self.mission.meco_t {
            self.mission.orbit_state_at(self.clock)
        } else {
            self.mission.pad_state()
        };
        self.flight = Some(Craft::maneuvering(r, v));
        log::info!("Manual flight control engaged");
    }

    /// World position (unit-sphere) + colour of the map craft/rocket marker.
    fn surface_marker(&self) -> (Vec3, [f32; 3]) {
        if let Some(rk) = self.launch.as_ref() {
            let u = rk.r / self.body.radius;
            let pos = Vec3::new(u.x as f32, u.y as f32, u.z as f32);
            let col = if rk.crashed {
                [1.0, 0.25, 0.2]
            } else if rk.orbit_reached {
                [0.5, 0.9, 1.0]
            } else {
                [1.0, 0.55, 0.15]
            };
            return (pos, col);
        }
        if let Some(craft) = self.flight.as_ref() {
            let col = if craft.crashed {
                [1.0, 0.25, 0.2]
            } else if craft.landed {
                [0.4, 1.0, 0.5]
            } else {
                [1.0, 0.85, 0.25]
            };
            (craft.marker(&self.body), col)
        } else {
            let col = if !self.launched {
                [0.4, 1.0, 0.4]
            } else if self.clock <= self.mission.meco_t {
                [1.0, 0.55, 0.15]
            } else {
                [0.5, 0.9, 1.0]
            };
            let rp = self.mission.rocket_pos(if self.launched { self.clock } else { 0.0 });
            (rp, col)
        }
    }

    /// Launch/ascent/orbit trajectories projected into the perspective map.
    /// Mission positions are unit-sphere (magnitude encodes altitude), so we
    /// scale by the home radius (Mm) and project with the system camera.
    fn build_overlay(&self, aspect: f32) -> Vec<OverlayVertex> {
        if self.view != View::Map {
            return Vec::new();
        }
        let cam = self.system_camera(aspect);
        let r = self.home_radius_mm as f64;
        let home_pos = self.universe.position(self.universe.home_index(), self.sys_time);
        let mut out: Vec<OverlayVertex> = Vec::new();

        // trajectory positions are home-centred unit-sphere; place them at the
        // home world's current orbital position.
        let polyline = |pts: &[Vec3], color: [f32; 3], out: &mut Vec<OverlayVertex>| {
            let mut prev: Option<[f32; 2]> = None;
            for &p in pts {
                let cur = cam.project(home_pos + p.as_dvec3() * r);
                if let (Some(a), Some(b)) = (prev, cur) {
                    out.push(OverlayVertex { pos: a, color });
                    out.push(OverlayVertex { pos: b, color });
                }
                prev = cur;
            }
        };

        polyline(&self.mission.pad_ring, [0.9, 0.6, 0.2], &mut out);

        if let Some(rk) = self.launch.as_ref() {
            // the player launch's predicted conic (empty while still suborbital)
            let pred = rk.predicted_orbit(&self.body);
            polyline(&pred, [1.0, 0.7, 0.25], &mut out);
        } else if let Some(craft) = self.flight.as_ref() {
            let pred = craft.predicted_orbit(&self.body);
            polyline(&pred, [0.5, 0.55, 0.25], &mut out);
        } else {
            if self.mission.reached {
                polyline(&self.mission.ring, [0.25, 0.7, 0.45], &mut out);
            }
            let path_pts: Vec<Vec3> = self.mission.path.iter().map(|(_, p)| *p).collect();
            polyline(&path_pts, [0.20, 0.45, 0.55], &mut out);
            let flown: Vec<Vec3> = self
                .mission
                .path
                .iter()
                .filter(|(t, _)| *t <= self.clock)
                .map(|(_, p)| *p)
                .collect();
            polyline(&flown, [0.45, 0.9, 1.0], &mut out);
        }

        // Orbit rings sampled from the actual ellipses. Planet rings appear at
        // system zoom; the focused planet's moon rings appear when zoomed in.
        let t = self.sys_time;
        let ring = |i: usize, color: [f32; 3], out: &mut Vec<OverlayVertex>| {
            let mut prev: Option<[f32; 2]> = None;
            for k in 0..=128 {
                let cur = cam.project(self.universe.ring_point(i, k as f64 / 128.0, t));
                if let (Some(u), Some(v)) = (prev, cur) {
                    out.push(OverlayVertex { pos: u, color });
                    out.push(OverlayVertex { pos: v, color });
                }
                prev = cur;
            }
        };
        let focus_pos = self.focus_pos();
        for (i, b) in self.universe.bodies.iter().enumerate() {
            match b.kind {
                Kind::Planet if self.sys_dist > 8.0e3 => {
                    ring(i, [0.38, 0.38, 0.52], &mut out);
                }
                Kind::Moon => {
                    // only the focused planet's moons, and only when zoomed near
                    let center = self.universe.orbit_center(i, t);
                    if self.sys_dist < 5.0e3 && (center - focus_pos).length() < 1.0 {
                        ring(i, [0.35, 0.5, 0.75], &mut out);
                    }
                }
                _ => {}
            }
        }

        out
    }

    /// In-scene overlay geometry for the map view: the home-world craft marker
    /// and locator dots for every small body. All text panels live in egui now
    /// (see `ui::build`), so this emits only diamonds/markers, never glyphs.
    fn build_hud(&self, res: (f32, f32)) -> Vec<OverlayVertex> {
        let mut out: Vec<OverlayVertex> = Vec::new();
        if self.view != View::Map {
            return out;
        }

        // Rocket/craft marker, placed at the home world's orbital position.
        let aspect = res.0 / res.1.max(1.0);
        let cam = self.system_camera(aspect);
        let home_pos = self.universe.position(self.universe.home_index(), self.sys_time);
        let (mpos, mcol) = self.surface_marker();
        if let Some(c) = cam.project(home_pos + mpos.as_dvec3() * self.home_radius_mm as f64) {
            push_filled_diamond(&mut out, c, 0.030, aspect, [0.0, 0.0, 0.0]);
            push_filled_diamond(&mut out, c, 0.020, aspect, mcol);
        }

        // locator dots for every small body (moons, asteroids, comets) so the
        // full system reads even though only stars/planets are ray-marched.
        for (i, b) in self.universe.bodies.iter().enumerate() {
            let sz = match b.kind {
                Kind::Moon => 0.009,
                Kind::AsteroidMajor | Kind::Comet => 0.007,
                Kind::AsteroidMinor => 0.005,
                _ => continue,
            };
            if let Some(c) = cam.project(self.universe.position(i, self.sys_time)) {
                push_filled_diamond(&mut out, c, sz + 0.004, aspect, [0.0, 0.0, 0.0]);
                push_filled_diamond(&mut out, c, sz, aspect, b.color);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Gpu: pipelines + buffers, independent of any window/surface.
// ---------------------------------------------------------------------------

struct Gpu {
    scene_pipeline: wgpu::RenderPipeline,
    scene_uniform_buf: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_buf: wgpu::Buffer,
    hud_pipeline: wgpu::RenderPipeline,
    hud_buf: wgpu::Buffer,
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_uniform_buf: wgpu::Buffer,
    mesh_bind_group: wgpu::BindGroup,
    /// Full-planet LOD terrain, rebuilt (camera-relative) when the camera moves.
    terrain_vbuf: wgpu::Buffer,
    /// Dynamic rocket-view geometry (pad, rocket pose, spent booster), rebuilt
    /// every frame.
    dyn_vbuf: wgpu::Buffer,
    /// Thruster FX billboards (flame + smoke particles).
    fx_pipeline: wgpu::RenderPipeline,
    fx_vbuf: wgpu::Buffer,
    sky_pipeline: wgpu::RenderPipeline,
    sky_uniform_buf: wgpu::Buffer,
    sky_bind_group: wgpu::BindGroup,
}

impl Gpu {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Gpu {
        let planet_img = image::load_from_memory(include_bytes!("../assets/planet.png"))
            .expect("decode planet.png")
            .to_rgba8();
        let (tw, th) = planet_img.dimensions();
        let planet_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("planet-tex"),
            size: wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &planet_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &planet_img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * tw),
                rows_per_image: Some(th),
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );
        let planet_view = planet_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let planet_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("planet-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Map view: same bind-group shape (uniform + planet texture + sampler).
        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scene.wgsl").into()),
        });
        let scene_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-uniforms"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene-bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&planet_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&planet_sampler),
                },
            ],
        });
        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene-pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
        });
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let vbuf_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 8,
                    shader_location: 1,
                },
            ],
        };
        let make_line_pipeline = |topology: wgpu::PrimitiveTopology, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&overlay_layout),
                vertex: wgpu::VertexState {
                    module: &overlay_shader,
                    entry_point: Some("vs"),
                    buffers: &[vbuf_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let overlay_pipeline =
            make_line_pipeline(wgpu::PrimitiveTopology::LineList, "overlay-pipeline");
        let hud_pipeline =
            make_line_pipeline(wgpu::PrimitiveTopology::TriangleList, "hud-pipeline");

        let overlay_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay-buf"),
            size: OVERLAY_CAP * std::mem::size_of::<OverlayVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hud_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud-buf"),
            size: HUD_CAP * std::mem::size_of::<OverlayVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Rocket view: triangle-mesh pipeline with a depth buffer.
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rocket"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rocket.wgsl").into()),
        });
        let mesh_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-uniforms"),
            size: std::mem::size_of::<MeshUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mesh_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh-bind-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let mesh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh-bind-group"),
            layout: &mesh_bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: mesh_uniform_buf.as_entire_binding(),
            }],
        });
        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh-layout"),
            bind_group_layouts: &[Some(&mesh_bind_layout)],
            immediate_size: 0,
        });
        let mesh_vbuf_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<rocket::MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
            ],
        };
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh-pipeline"),
            layout: Some(&mesh_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("vs"),
                buffers: &[mesh_vbuf_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Thruster-FX pipeline: flame + smoke billboards, premultiplied-alpha
        // blend (additive flame + over smoke in one pass), depth-tested but never
        // written. Reuses the mesh uniform (viewproj + log depth + time).
        let fx_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fx.wgsl").into()),
        });
        let fx_vbuf_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FxVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 20, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 36, shader_location: 3 },
            ],
        };
        let fx_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fx-pipeline"),
            layout: Some(&mesh_layout),
            vertex: wgpu::VertexState {
                module: &fx_shader,
                entry_point: Some("vs"),
                buffers: &[fx_vbuf_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fx_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let fx_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx-vbuf"),
            size: FX_CAP * std::mem::size_of::<FxVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sky pipeline: fullscreen, depth-compatible with the mesh pass but
        // never writes depth, so the terrain draws over it.
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sky.wgsl").into()),
        });
        let sky_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky-uniforms"),
            size: std::mem::size_of::<SkyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky-bind-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky-bind-group"),
            layout: &sky_bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: sky_uniform_buf.as_entire_binding() }],
        });
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky-layout"),
            bind_group_layouts: &[Some(&sky_bind_layout)],
            immediate_size: 0,
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky-pipeline"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Full-planet terrain is a dynamic, camera-relative buffer rebuilt as the
        // camera moves (floating origin); the rocket/pad are in dyn_vbuf.
        let terrain_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-vbuf"),
            size: TERRAIN_CAP * std::mem::size_of::<rocket::MeshVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dyn_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dyn-mesh-vbuf"),
            size: DYN_MESH_CAP * std::mem::size_of::<rocket::MeshVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Gpu {
            scene_pipeline,
            scene_uniform_buf,
            scene_bind_group,
            overlay_pipeline,
            overlay_buf,
            hud_pipeline,
            hud_buf,
            mesh_pipeline,
            mesh_uniform_buf,
            mesh_bind_group,
            terrain_vbuf,
            dyn_vbuf,
            fx_pipeline,
            fx_vbuf,
            sky_pipeline,
            sky_uniform_buf,
            sky_bind_group,
        }
    }

    /// Upload this frame's uniforms + geometry. Returns (overlay verts, hud
    /// verts, dynamic rocket-mesh verts).
    fn prepare(
        &self,
        queue: &wgpu::Queue,
        world: &mut World,
        w: u32,
        h: u32,
        time: f32,
    ) -> (usize, usize, u32, u32) {
        let res = [w as f32, h.max(1) as f32];
        let mut dyn_n = 0u32;
        let mut fx_n = 0u32;
        match world.view {
            View::Map => {
                let su = world.scene_uniforms(res, time);
                queue.write_buffer(&self.scene_uniform_buf, 0, bytemuck::bytes_of(&su));
            }
            View::Rocket => {
                // (Re)build + upload the camera-relative planet terrain on demand.
                if world.terrain_dirty {
                    if world.terrain_verts.is_empty() {
                        world.rebuild_terrain();
                    }
                    let tc = world.terrain_count as usize;
                    if tc > 0 {
                        queue.write_buffer(
                            &self.terrain_vbuf,
                            0,
                            bytemuck::cast_slice(&world.terrain_verts[..tc]),
                        );
                    }
                    world.terrain_dirty = false;
                }
                let mu = world.mesh_uniforms(res);
                queue.write_buffer(&self.mesh_uniform_buf, 0, bytemuck::bytes_of(&mu));
                let sk = world.sky_uniforms(res);
                queue.write_buffer(&self.sky_uniform_buf, 0, bytemuck::bytes_of(&sk));
                let dyn_verts = world.build_dynamic_mesh();
                dyn_n = (dyn_verts.len() as u64).min(DYN_MESH_CAP) as u32;
                if dyn_n > 0 {
                    queue.write_buffer(
                        &self.dyn_vbuf,
                        0,
                        bytemuck::cast_slice(&dyn_verts[..dyn_n as usize]),
                    );
                }
                let (eye, _t, right, up, _f, _tan) = world.rocket_camera(res[0] / res[1].max(1.0));
                let fx_verts = world.build_fx(eye, right, up);
                fx_n = (fx_verts.len() as u64).min(FX_CAP) as u32;
                if fx_n > 0 {
                    queue.write_buffer(&self.fx_vbuf, 0, bytemuck::cast_slice(&fx_verts[..fx_n as usize]));
                }
            }
        }

        let aspect = res[0] / res[1];
        let verts = world.build_overlay(aspect);
        let n = verts.len().min(OVERLAY_CAP as usize);
        if n > 0 {
            queue.write_buffer(&self.overlay_buf, 0, bytemuck::cast_slice(&verts[..n]));
        }
        let hud_verts = world.build_hud((res[0], res[1]));
        let hn = hud_verts.len().min(HUD_CAP as usize);
        if hn > 0 {
            queue.write_buffer(&self.hud_buf, 0, bytemuck::cast_slice(&hud_verts[..hn]));
        }
        (n, hn, dyn_n, fx_n)
    }

    /// Draw the thruster FX billboards (flame + smoke), in the mesh pass.
    fn draw_fx(&self, pass: &mut wgpu::RenderPass, fx_count: u32) {
        if fx_count > 0 {
            pass.set_pipeline(&self.fx_pipeline);
            pass.set_bind_group(0, &self.mesh_bind_group, &[]);
            pass.set_vertex_buffer(0, self.fx_vbuf.slice(..));
            pass.draw(0..fx_count, 0..1);
        }
    }

    /// Map view: fullscreen multi-body raymarch + 2D overlay/HUD. The rocket
    /// view does not use this; it draws meshes then `draw_overlay`.
    fn draw(&self, pass: &mut wgpu::RenderPass, view: View, n_overlay: usize, n_hud: usize) {
        if view == View::Map {
            pass.set_pipeline(&self.scene_pipeline);
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.draw_overlay(pass, n_overlay, n_hud);
    }

    /// The 2D line overlay + HUD triangles (no depth).
    fn draw_overlay(&self, pass: &mut wgpu::RenderPass, n_overlay: usize, n_hud: usize) {
        if n_overlay > 0 {
            pass.set_pipeline(&self.overlay_pipeline);
            pass.set_vertex_buffer(0, self.overlay_buf.slice(..));
            pass.draw(0..n_overlay as u32, 0..1);
        }
        if n_hud > 0 {
            pass.set_pipeline(&self.hud_pipeline);
            pass.set_vertex_buffer(0, self.hud_buf.slice(..));
            pass.draw(0..n_hud as u32, 0..1);
        }
    }

    /// The 3D rocket/pad/terrain mesh (depth-tested). Used in its own pass.
    fn draw_meshes(&self, pass: &mut wgpu::RenderPass, terrain_count: u32, dyn_count: u32) {
        pass.set_pipeline(&self.mesh_pipeline);
        pass.set_bind_group(0, &self.mesh_bind_group, &[]);
        if terrain_count > 0 {
            pass.set_vertex_buffer(0, self.terrain_vbuf.slice(..));
            pass.draw(0..terrain_count, 0..1);
        }
        if dyn_count > 0 {
            pass.set_vertex_buffer(0, self.dyn_vbuf.slice(..));
            pass.draw(0..dyn_count, 0..1);
        }
    }

    /// Fullscreen sky behind the terrain (no depth write).
    fn draw_sky(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.sky_pipeline);
        pass.set_bind_group(0, &self.sky_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Push a filled diamond (two triangles) at clip point `c` with half-height
/// `hy`, for the HUD/triangle pipeline. Square in pixels via the aspect ratio.
fn push_filled_diamond(out: &mut Vec<OverlayVertex>, c: [f32; 2], hy: f32, aspect: f32, color: [f32; 3]) {
    let hx = hy / aspect;
    let top = [c[0], c[1] + hy];
    let right = [c[0] + hx, c[1]];
    let bot = [c[0], c[1] - hy];
    let left = [c[0] - hx, c[1]];
    for p in [top, right, bot, top, bot, left] {
        out.push(OverlayVertex { pos: p, color });
    }
}


fn render_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn create_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// Mesh pass: clear color to sky, clear depth, depth-test enabled.
fn mesh_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    color: &'a wgpu::TextureView,
    depth: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("mesh-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.45, g: 0.62, b: 0.82, a: 1.0 }),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Overlay pass: keep existing color (load), no depth, for 2D HUD on top.
fn overlay_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    color: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("overlay-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

// ---------------------------------------------------------------------------
// State: the windowed client.
// ---------------------------------------------------------------------------

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    world: World,
    start: instant_now::Instant,
    last_t: f32,
    dragging: bool,
    last_cursor: (f64, f64),
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

impl State {
    async fn new(window: Arc<Window>) -> State {
        let (width, height) = surface_size(&window);

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        let surface = instance.create_surface(window.clone()).expect("surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no adapter");

        let limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits::downlevel_defaults()
        } else {
            wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let gpu = Gpu::new(&device, &queue, format);

        // egui: context + winit input glue + wgpu renderer
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            None,
            None,
            None,
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

        State {
            window,
            surface,
            device,
            queue,
            config,
            gpu,
            world: World::new(),
            start: instant_now::Instant::now(),
            last_t: 0.0,
            dragging: false,
            last_cursor: (0.0, 0.0),
            egui_ctx,
            egui_state,
            egui_renderer,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) {
        // On the web the canvas tracks the viewport via CSS; keep the surface
        // (and thus the canvas backing buffer) in sync each frame.
        #[cfg(target_arch = "wasm32")]
        {
            let (w, h) = surface_size(&self.window);
            if w != self.config.width || h != self.config.height {
                self.resize(w, h);
            }
        }

        let t = self.start.elapsed().as_secs_f32();
        let frame_dt = (t - self.last_t).clamp(0.0, 0.1);
        self.last_t = t;
        self.world.advance(frame_dt);

        let (n, hn, dyn_n, fx_n) = self
            .gpu
            .prepare(&self.queue, &mut self.world, self.config.width, self.config.height, t);
        let terrain_n = self.world.terrain_count;

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });

        // --- egui: run the UI and prepare its draw data ---
        let raw_input = self.egui_state.take_egui_input(&self.window);
        self.egui_ctx.begin_pass(raw_input);
        ui::build(&self.egui_ctx, &mut self.world);
        let full = self.egui_ctx.end_pass();
        self.egui_state
            .handle_platform_output(&self.window, full.platform_output);
        let ppp = self.egui_ctx.pixels_per_point();
        let prims = self.egui_ctx.tessellate(full.shapes, ppp);
        for (id, delta) in &full.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ppp,
        };
        self.egui_renderer
            .update_buffers(&self.device, &self.queue, &mut encoder, &prims, &screen);

        if self.world.view == View::Rocket {
            let depth = create_depth(&self.device, self.config.width, self.config.height);
            {
                let mut pass = mesh_pass(&mut encoder, &view, &depth);
                self.gpu.draw_sky(&mut pass);
                self.gpu.draw_meshes(&mut pass, terrain_n, dyn_n);
                self.gpu.draw_fx(&mut pass, fx_n);
            }
            {
                let mut pass = overlay_pass(&mut encoder, &view);
                self.gpu.draw_overlay(&mut pass, n, hn);
            }
        } else {
            let mut pass = render_pass(&mut encoder, &view);
            self.gpu.draw(&mut pass, self.world.view, n, hn);
        }
        {
            let mut pass = overlay_pass(&mut encoder, &view).forget_lifetime();
            self.egui_renderer.render(&mut pass, &prims, &screen);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        for id in &full.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

// ---------------------------------------------------------------------------
// Headless screenshot (native only): render framed shots to PNGs for visual
// validation of every feature.
// ---------------------------------------------------------------------------

/// (scenario, default output path) for `app shot all`.
#[cfg(not(target_arch = "wasm32"))]
const SHOT_SCENARIOS: &[(&str, &str)] = &[
    ("pad", "out/pad.png"),
    ("ascent", "out/ascent.png"),
    ("surface", "out/client.png"),
    ("flight", "out/flight.png"),
    ("system", "out/system.png"),
    ("moon", "out/moon.png"),
    ("rocket", "out/rocket.png"),
];

#[cfg(not(target_arch = "wasm32"))]
fn make_shot_device() -> (wgpu::Device, wgpu::Queue) {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no adapter");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("shot-device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
    }))
    .expect("device")
}

/// Build the world + sun time for a named validation scenario.
#[cfg(not(target_arch = "wasm32"))]
fn setup_world(scenario: &str, width: u32, height: u32) -> (World, f32) {
    let _ = (width, height);
    let mut world = World::new();

    // Fly a scripted player-style launch (open-loop gravity turn, auto-stage)
    // for `dur` seconds, so the rocket-view launch is verifiable headlessly.
    fn fly(w: &mut World, dur: f32) {
        let mut t = 0.0;
        while t < dur {
            if let Some(rk) = w.launch.as_mut() {
                rk.pitch = (((rk.met - 10.0) / 120.0).clamp(0.0, 1.0) * 78f64.to_radians()).min(1.5);
                if rk.prop_frac() <= 0.0 && rk.stages.len() > 1 && rk.stage_base == 0 {
                    w.stage_launch();
                }
            }
            w.advance(0.1);
            t += 0.1;
        }
    }
    // Fly until the first booster is dry, stage it, then coast a moment so the
    // jettisoned booster is visibly tumbling away.
    fn fly_to_staging(w: &mut World) {
        for _ in 0..4000 {
            let dry = w
                .launch
                .as_ref()
                .map(|rk| rk.stage_base == 0 && rk.prop_frac() <= 0.0)
                .unwrap_or(true);
            if dry {
                break;
            }
            if let Some(rk) = w.launch.as_mut() {
                rk.pitch = (((rk.met - 10.0) / 120.0).clamp(0.0, 1.0) * 78f64.to_radians()).min(1.5);
            }
            w.advance(0.2);
        }
        w.stage_launch();
        // throttle the upper stage down a touch so its plume doesn't crowd the
        // jettisoned booster, then coast so the booster drifts clear.
        for _ in 0..30 {
            w.advance(0.1);
        }
    }

    // Frame the map on the home world (where the launch/orbit is drawn).
    let frame_map = |w: &mut World| {
        w.view = View::Map;
        let hi = w.nav.iter().position(|&i| w.universe.bodies[i].is_home).unwrap_or(0);
        w.set_focus(hi);
        w.sys_az = 1.4;
        w.sys_el = 0.32;
    };
    let time = match scenario {
        "rocket" | "pad" => {
            world.view = View::Rocket;
            world.rocket_az = 4.97; // face inland (land), coast to the sides
            world.rocket_el = 0.12;
            0.0
        }
        "liftoff" => {
            world.view = View::Rocket;
            world.rocket_az = 4.97;
            world.rocket_el = 0.07;
            world.rocket_dist = 60.0;
            world.rocket_el = 0.12;
            world.ignite_launch();
            fly(&mut world, 1.4); // ignition: plume + billowing pad smoke
            0.0
        }
        "liftoff2" => {
            world.view = View::Rocket;
            world.rocket_az = 4.6;
            world.rocket_el = 0.16;
            world.rocket_dist = 110.0;
            world.ignite_launch();
            fly(&mut world, 22.0); // pitched into the gravity turn, smoke trailing
            0.0
        }
        "staging" => {
            world.view = View::Rocket;
            world.rocket_az = 3.6;
            world.rocket_el = 0.18;
            world.rocket_dist = 130.0;
            world.ignite_launch();
            fly_to_staging(&mut world); // spent booster tumbling away
            0.0
        }
        "orbit" => {
            // high up: pull the camera back to frame the planet against space.
            world.view = View::Rocket;
            world.rocket_az = 3.4;
            world.rocket_el = 0.30;
            world.ignite_launch();
            fly(&mut world, 130.0);
            world.rocket_dist = 900.0;
            0.0
        }
        "launchmap" => {
            // a player launch shown on the orbital map: live marker + predicted
            // conic as the upper stage builds its parking orbit.
            world.ignite_launch();
            fly(&mut world, 200.0);
            frame_map(&mut world);
            world.sys_dist = 26.0;
            world.sys_el = 0.5;
            6.0
        }
        "system" => {
            // wide system shot: the binary + planet orbits, framed on the barycentre
            world.view = View::Map;
            world.sys_focus = DVec3::ZERO;
            world.sys_dist = 4.0e6; // ~27 AU
            world.sys_az = 1.4;
            world.sys_el = 0.55;
            6.0
        }
        "moon" => {
            // home + its moon
            frame_map(&mut world);
            world.sys_dist = 60.0;
            6.0
        }
        "moons" => {
            // focus a moon up close so it ray-marches as a real sphere, with its
            // gas giant looming behind.
            world.view = View::Map;
            let midx = world
                .universe
                .bodies
                .iter()
                .position(|b| b.kind == Kind::Moon)
                .unwrap_or(0);
            world.set_focus(midx);
            world.sys_dist = world.universe.bodies[midx].radius * 6.0;
            world.sys_az = 1.1;
            world.sys_el = 0.20;
            6.0
        }
        "ascent" => {
            frame_map(&mut world);
            world.launched = true;
            world.clock = world.mission.meco_t * 0.5; // mid powered ascent
            6.0
        }
        "flight" => {
            frame_map(&mut world);
            world.launched = true;
            world.clock = world.mission.meco_t + 10.0;
            let (r, v) = world.mission.orbit_state_at(world.clock);
            let mut craft = Craft::maneuvering(r, v);
            craft.throttle = 0.6;
            craft.mode = Mode::Retrograde;
            world.flight = Some(craft);
            6.0
        }
        _ => {
            // map view: craft coasting in the parking orbit.
            frame_map(&mut world);
            world.launched = true;
            world.clock = world.mission.meco_t + 240.0;
            6.0
        }
    };
    (world, time)
}

#[cfg(not(target_arch = "wasm32"))]
fn screenshot_all(width: u32, height: u32) {
    let (device, queue) = make_shot_device();
    let gpu = Gpu::new(&device, &queue, wgpu::TextureFormat::Rgba8UnormSrgb);
    for (scenario, path) in SHOT_SCENARIOS {
        render_shot(&device, &queue, &gpu, scenario, path, width, height);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn screenshot(path: &str, width: u32, height: u32, scenario: &str) {
    let (device, queue) = make_shot_device();
    let gpu = Gpu::new(&device, &queue, wgpu::TextureFormat::Rgba8UnormSrgb);
    render_shot(&device, &queue, &gpu, scenario, path, width, height);
}

#[cfg(not(target_arch = "wasm32"))]
fn render_shot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &Gpu,
    scenario: &str,
    path: &str,
    width: u32,
    height: u32,
) {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (mut world, time) = setup_world(scenario, width, height);

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let (n, hn, dyn_n, fx_n) = gpu.prepare(queue, &mut world, width, height, time);
    let terrain_n = world.terrain_count;

    // 256-byte aligned row pitch for the readback copy.
    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot-enc") });
    if world.view == View::Rocket {
        let depth = create_depth(device, width, height);
        {
            let mut pass = mesh_pass(&mut encoder, &target_view, &depth);
            gpu.draw_sky(&mut pass);
            gpu.draw_meshes(&mut pass, terrain_n, dyn_n);
            gpu.draw_fx(&mut pass, fx_n);
        }
        {
            let mut pass = overlay_pass(&mut encoder, &target_view);
            gpu.draw_overlay(&mut pass, n, hn);
        }
    } else {
        let mut pass = render_pass(&mut encoder, &target_view);
        gpu.draw(&mut pass, world.view, n, hn);
    }

    // egui overlay (so the panels are verifiable headlessly), in every view.
    {
        let ctx = egui::Context::default();
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(width as f32, height as f32),
            )),
            ..Default::default()
        };
        let mut w = world;
        ctx.set_pixels_per_point(1.0);
        // frame 1 warms up fonts/layout (and builds the font atlas); frame 2
        // emits the real shapes. Upload textures from both.
        ctx.begin_pass(raw.clone());
        ui::build(&ctx, &mut w);
        let warm = ctx.end_pass();
        ctx.begin_pass(raw);
        ui::build(&ctx, &mut w);
        let full = ctx.end_pass();
        let prims = ctx.tessellate(full.shapes, 1.0);
        let mut renderer =
            egui_wgpu::Renderer::new(device, format, egui_wgpu::RendererOptions::default());
        for (id, delta) in warm.textures_delta.set.iter().chain(full.textures_delta.set.iter()) {
            renderer.update_texture(device, queue, *id, delta);
        }
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [width, height], pixels_per_point: 1.0 };
        renderer.update_buffers(device, queue, &mut encoder, &prims, &screen);
        {
            let mut pass = overlay_pass(&mut encoder, &target_view).forget_lifetime();
            renderer.render(&mut pass, &prims, &screen);
        }
    }

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range();

    let mut pixels = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    readback.unmap();

    let img: image::RgbaImage =
        image::ImageBuffer::from_raw(width, height, pixels).expect("image buffer");
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    img.save(path).expect("write png");
    println!("wrote {path} ({width}x{height})");
}

// ---------------------------------------------------------------------------
// winit app + entry point.
// ---------------------------------------------------------------------------

enum UserEvent {
    Ready(State),
}

struct App {
    proxy: EventLoopProxy<UserEvent>,
    state: Option<State>,
}

impl App {
    fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        App { proxy, state: None }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("Mining the Sky");
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.body())
                .and_then(|body| {
                    let canvas = web_sys::Element::from(window.canvas()?);
                    body.append_child(&canvas).ok()
                })
                .expect("append canvas");
        }

        let proxy = self.proxy.clone();
        let win = window.clone();
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            if !webx::has_webgpu() {
                webx::set_status(
                    "WebGPU is not available in this browser. Use Chrome/Edge 113+ \
                     (or Safari 18+) on a machine with a supported GPU.",
                );
                return;
            }
            webx::set_status("Starting renderer...");
            let state = State::new(win).await;
            webx::set_status(
                "Tab: map / rocket view  -  drag: orbit  -  scroll: zoom  -  \
                 Space: launch  -  F: manual flight",
            );
            let _ = proxy.send_event(UserEvent::Ready(state));
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = pollster::block_on(State::new(win));
            let _ = proxy.send_event(UserEvent::Ready(state));
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        let UserEvent::Ready(state) = event;
        log::info!(
            "Controls: drag orbit, scroll zoom, Space launch, F manual flight, V system view, [ ] warp"
        );
        state.window.request_redraw();
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Let egui see the event first; if it consumed a pointer/keyboard input
        // (i.e. the user is interacting with the UI), don't also drive the game.
        let egui_resp = state.egui_state.on_window_event(&state.window, &event);
        if egui_resp.repaint {
            state.window.request_redraw();
        }
        if egui_resp.consumed
            && matches!(
                event,
                WindowEvent::MouseInput { .. }
                    | WindowEvent::MouseWheel { .. }
                    | WindowEvent::KeyboardInput { .. }
            )
        {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
                state.window.request_redraw();
            }
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                if button == MouseButton::Left {
                    let pressed = btn_state == ElementState::Pressed;
                    // In the map, a click on the body menu jumps focus (and does
                    // not start an orbit drag).
                    if pressed && state.world.view == View::Map {
                        // click a body in the scene to focus it (the egui panel
                        // handles its own clicks before we get here).
                        let res = (state.config.width as f32, state.config.height as f32);
                        let (cx, cy) = (state.last_cursor.0 as f32, state.last_cursor.1 as f32);
                        if let Some(b) = state.world.pick_body(res, cx, cy) {
                            state.world.set_focus(b);
                            return;
                        }
                    }
                    state.dragging = pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                if state.dragging {
                    let dx = (x - state.last_cursor.0) as f32;
                    let dy = (y - state.last_cursor.1) as f32;
                    match state.world.view {
                        View::Map => {
                            state.world.sys_az += dx * 0.005;
                            state.world.sys_el = (state.world.sys_el + dy * 0.005).clamp(-1.5, 1.5);
                        }
                        View::Rocket => {
                            state.world.rocket_az += dx * 0.006;
                            state.world.rocket_el =
                                (state.world.rocket_el + dy * 0.006).clamp(-0.2, 1.4);
                        }
                    }
                }
                state.last_cursor = (x, y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 60.0,
                };
                match state.world.view {
                    View::Map => {
                        // zoom spans body-radius to far outer system (~150 AU)
                        state.world.sys_dist =
                            (state.world.sys_dist * (1.0 - dy as f64 * 0.12)).clamp(2.0, 2.2e7);
                    }
                    View::Rocket => {
                        state.world.rocket_dist =
                            (state.world.rocket_dist * (1.0 - dy * 0.12)).clamp(12.0, 400.0);
                    }
                }
            }
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state == ElementState::Pressed {
                    let code = match key_event.physical_key {
                        PhysicalKey::Code(c) => c,
                        _ => return,
                    };
                    if let Some(craft) = state.world.flight.as_mut() {
                        match code {
                            KeyCode::KeyW => craft.throttle = (craft.throttle + 0.08).min(1.0),
                            KeyCode::KeyS => craft.throttle = (craft.throttle - 0.08).max(0.0),
                            _ => {}
                        }
                    }
                    // player-controlled launch: pitch (W/S), throttle (Shift/Ctrl,
                    // Z full / X cut). These repeat while held.
                    if let Some(rk) = state.world.launch.as_mut() {
                        let step = 2f64.to_radians();
                        match code {
                            KeyCode::KeyW | KeyCode::ArrowUp => {
                                rk.pitch = (rk.pitch + step).min(110f64.to_radians())
                            }
                            KeyCode::KeyS | KeyCode::ArrowDown => {
                                rk.pitch = (rk.pitch - step).max(0.0)
                            }
                            KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                                rk.throttle = (rk.throttle + 0.06).min(1.0)
                            }
                            KeyCode::ControlLeft | KeyCode::ControlRight => {
                                rk.throttle = (rk.throttle - 0.06).max(0.0)
                            }
                            KeyCode::KeyZ => rk.throttle = 1.0,
                            KeyCode::KeyX => rk.throttle = 0.0,
                            _ => {}
                        }
                    }
                    if key_event.repeat {
                        return;
                    }
                    match code {
                        KeyCode::Tab | KeyCode::KeyV => state.world.toggle_view(),
                        KeyCode::KeyC if state.world.view == View::Map => {
                            state.world.cycle_focus()
                        }
                        KeyCode::KeyF => state.world.toggle_flight(),
                        KeyCode::Space => {
                            if state.world.view == View::Rocket {
                                // ignite, then Space stages the spent booster.
                                if state.world.launch.is_none() {
                                    state.world.ignite_launch();
                                } else {
                                    state.world.stage_launch();
                                }
                            } else if state.world.flight.is_none() {
                                state.world.toggle_launch();
                            }
                        }
                        KeyCode::KeyR if state.world.view == View::Rocket => {
                            state.world.reset_launch()
                        }
                        KeyCode::BracketRight => {
                            state.world.warp = (state.world.warp * 2.0).min(10000.0);
                        }
                        KeyCode::BracketLeft => {
                            state.world.warp = (state.world.warp * 0.5).max(1.0);
                        }
                        KeyCode::Digit1
                        | KeyCode::Digit2
                        | KeyCode::Digit3
                        | KeyCode::Digit4
                        | KeyCode::Digit5
                        | KeyCode::Digit6
                        | KeyCode::Digit7
                        | KeyCode::Digit8
                        | KeyCode::Digit9 => {
                            let idx = match code {
                                KeyCode::Digit1 => 0,
                                KeyCode::Digit2 => 1,
                                KeyCode::Digit3 => 2,
                                KeyCode::Digit4 => 3,
                                KeyCode::Digit5 => 4,
                                KeyCode::Digit6 => 5,
                                KeyCode::Digit7 => 6,
                                KeyCode::Digit8 => 7,
                                _ => 8,
                            };
                            // In the map (no manual flight), number keys jump to
                            // a body; in flight they select the thrust mode.
                            if state.world.view == View::Map && state.world.flight.is_none() {
                                if let Some(&bi) = state.world.nav.get(idx) {
                                    state.world.set_focus(bi);
                                }
                            } else if let Some(c) = state.world.flight.as_mut() {
                                c.mode = match idx {
                                    0 => Mode::Prograde,
                                    1 => Mode::Retrograde,
                                    2 => Mode::RadialOut,
                                    3 => Mode::RadialIn,
                                    _ => c.mode,
                                };
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    // Native: `app shot [scenario] [out.png]` renders headless frame(s) and
    // exits. `app shot all` validates every feature into ./out.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if args.iter().any(|a| a == "catalog") {
            let w = World::new();
            let md = w.universe.catalog_markdown();
            let path = "docs/system_catalog.md";
            if let Some(p) = std::path::Path::new(path).parent() {
                let _ = std::fs::create_dir_all(p);
            }
            std::fs::write(path, &md).expect("write catalog");
            println!("{md}");
            println!("wrote {path}");
            return;
        }
        if args.iter().any(|a| a == "shot" || a == "--shot") {
            env_logger::init();
            if args.iter().any(|a| a == "all") {
                screenshot_all(1280, 800);
                return;
            }
            let scenario = if args.iter().any(|a| a == "moons") {
                "moons"
            } else if args.iter().any(|a| a == "moon") {
                "moon"
            } else if args.iter().any(|a| a == "rocket") {
                "rocket"
            } else if args.iter().any(|a| a == "system") {
                "system"
            } else if args.iter().any(|a| a == "pad") {
                "pad"
            } else if args.iter().any(|a| a == "liftoff2") {
                "liftoff2"
            } else if args.iter().any(|a| a == "liftoff") {
                "liftoff"
            } else if args.iter().any(|a| a == "staging") {
                "staging"
            } else if args.iter().any(|a| a == "launchmap") {
                "launchmap"
            } else if args.iter().any(|a| a == "orbit") {
                "orbit"
            } else if args.iter().any(|a| a == "ascent") {
                "ascent"
            } else if args.iter().any(|a| a == "flight") {
                "flight"
            } else {
                "surface"
            };
            let default = match scenario {
                "moons" => "out/moons.png",
                "moon" => "out/moon.png",
                "rocket" => "out/rocket.png",
                "system" => "out/system.png",
                "pad" => "out/pad.png",
                "liftoff" => "out/liftoff.png",
                "liftoff2" => "out/liftoff2.png",
                "staging" => "out/staging.png",
                "launchmap" => "out/launchmap.png",
                "ascent" => "out/ascent.png",
                "flight" => "out/flight.png",
                _ => "out/client.png",
            };
            let path = args
                .iter()
                .skip(1)
                .find(|a| a.ends_with(".png"))
                .cloned()
                .unwrap_or_else(|| default.to_string());
            screenshot(&path, 1280, 800, scenario);
            return;
        }
    }

    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("event loop");
    let app = App::new(event_loop.create_proxy());

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let _ = console_log::init_with_level(log::Level::Info);
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let mut app = app;
        event_loop.run_app(&mut app).expect("run");
    }
}

/// Small cross-target clock: `std::time::Instant` panics on wasm, so use the
/// browser performance clock there.
mod instant_now {
    #[derive(Copy, Clone)]
    pub struct Instant {
        #[cfg(not(target_arch = "wasm32"))]
        inner: std::time::Instant,
        #[cfg(target_arch = "wasm32")]
        start_ms: f64,
    }

    impl Instant {
        #[cfg(not(target_arch = "wasm32"))]
        pub fn now() -> Self {
            Instant { inner: std::time::Instant::now() }
        }
        #[cfg(target_arch = "wasm32")]
        pub fn now() -> Self {
            Instant { start_ms: now_ms() }
        }

        #[cfg(not(target_arch = "wasm32"))]
        pub fn elapsed(&self) -> std::time::Duration {
            self.inner.elapsed()
        }
        #[cfg(target_arch = "wasm32")]
        pub fn elapsed(&self) -> std::time::Duration {
            std::time::Duration::from_secs_f64((now_ms() - self.start_ms) / 1000.0)
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn now_ms() -> f64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0)
    }
}
