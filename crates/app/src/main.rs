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

use glam::{DVec3, Mat3, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

mod flight;
mod hud;
mod mission;
use flight::{Craft, GravBody, Mode};
use hud::Hud;
use mission::Mission;
use sim::body::CentralBody;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    scale: f32,
    time: f32,
    sun: [f32; 4],
    cx: [f32; 4],
    cy: [f32; 4],
    cz: [f32; 4],
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
    params: [f32; 4],
    res: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OverlayVertex {
    pos: [f32; 2],
    color: [f32; 3],
}

const OVERLAY_CAP: u64 = 8192;
const HUD_CAP: u64 = 40000;

/// Render-space length unit for the system view: 1000 km.
const MM: f32 = 1.0e6;

#[derive(Clone, Copy, PartialEq)]
enum View {
    Surface,
    System,
}

/// A perspective camera for the system view (all positions in Mm).
struct SystemCamera {
    pos: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
    fovscale: f32,
    aspect: f32,
}

impl SystemCamera {
    /// Project a world point (Mm) to clip space, or `None` if behind the camera.
    fn project(&self, p: Vec3) -> Option<[f32; 2]> {
        let rel = p - self.pos;
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

struct World {
    mission: Mission,
    body: CentralBody,
    flight: Option<Craft>,
    az: f32,
    el: f32,
    scale: f32,
    launched: bool,
    clock: f32, // mission-elapsed seconds
    warp: f32,

    // system view: a perspective camera framing the home world + its moon.
    view: View,
    sys_az: f32,
    sys_el: f32,
    sys_dist: f32, // camera distance from the focus point (Mm)
    sys_focus: Vec3,
    home_radius_mm: f32,
    moon_center_mm: Vec3,
    moon_radius_mm: f32,

    // moon physics (metres, the frame the flight model integrates in)
    moon_center_m: DVec3,
    moon_mu: f64,
    moon_radius_m: f64,
}

impl World {
    fn new() -> World {
        let mission = Mission::pioneer_from_spaceport();
        let body = CentralBody::home();
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
        World {
            az: mission.spaceport_lon,
            el: mission.spaceport_lat,
            scale: 1.25,
            launched: false,
            clock: 0.0,
            warp: 8.0,
            mission,
            body,
            flight: None,
            view: View::Surface,
            sys_az: 1.4,
            sys_el: 0.30,
            sys_dist: 120.0,
            sys_focus: moon_center_mm * 0.5,
            home_radius_mm,
            moon_center_mm,
            moon_radius_mm,
            moon_center_m,
            moon_mu,
            moon_radius_m,
        }
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
            View::Surface => View::System,
            View::System => View::Surface,
        };
    }

    /// System-view perspective camera: position + basis + tan(fov/2), all in Mm.
    fn system_camera(&self, aspect: f32) -> SystemCamera {
        let dir = Vec3::new(
            self.sys_el.cos() * self.sys_az.cos(),
            self.sys_el.sin(),
            self.sys_el.cos() * self.sys_az.sin(),
        );
        let cam_pos = self.sys_focus + dir * self.sys_dist;
        let forward = (self.sys_focus - cam_pos).normalize();
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

    /// Perspective camera + body uniforms for the system view.
    fn scene_uniforms(&self, res: [f32; 2], time: f32) -> SceneUniforms {
        let aspect = res[0] / res[1].max(1.0);
        let cam = self.system_camera(aspect);

        let st = time * 0.05 + 0.8;
        let sun = Vec3::new(st.cos(), 0.25, st.sin()).normalize();

        SceneUniforms {
            cam_pos: [cam.pos.x, cam.pos.y, cam.pos.z, 0.0],
            cam_x: [cam.right.x, cam.right.y, cam.right.z, 0.0],
            cam_y: [cam.up.x, cam.up.y, cam.up.z, 0.0],
            cam_z: [cam.forward.x, cam.forward.y, cam.forward.z, 0.0],
            sun: [sun.x, sun.y, sun.z, 0.0],
            home: [0.0, 0.0, 0.0, self.home_radius_mm],
            moon: [
                self.moon_center_mm.x,
                self.moon_center_mm.y,
                self.moon_center_mm.z,
                self.moon_radius_mm,
            ],
            params: [cam.fovscale, aspect, time, 0.0],
            res: [res[0], res[1], 0.0, 0.0],
        }
    }

    /// Advance simulation by a real frame dt (seconds).
    fn advance(&mut self, frame_dt: f32) {
        let bodies = self.grav_bodies();
        if let Some(craft) = self.flight.as_mut() {
            let dt_sim = frame_dt * self.warp.min(8.0);
            craft.integrate(&self.body, &bodies, dt_sim as f64);
        } else if self.launched {
            self.clock += frame_dt * self.warp;
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

    /// World-from-view rotation: column 2 is the world point facing the camera.
    fn camera_rot(&self) -> Mat3 {
        let d = Vec3::new(
            self.el.cos() * self.az.cos(),
            self.el.sin(),
            self.el.cos() * self.az.sin(),
        );
        let xc = Vec3::Y.cross(d).normalize();
        let yc = d.cross(xc).normalize();
        Mat3::from_cols(xc, yc, d)
    }

    fn uniforms(&self, res: [f32; 2], time: f32) -> Uniforms {
        let rot = self.camera_rot();
        let st = time * 0.03 + self.mission.spaceport_lon;
        let sun = Vec3::new(st.cos() * 0.95, 0.28, st.sin() * 0.95).normalize();
        Uniforms {
            resolution: res,
            scale: self.scale,
            time,
            sun: [sun.x, sun.y, sun.z, 0.0],
            cx: [rot.x_axis.x, rot.x_axis.y, rot.x_axis.z, 0.0],
            cy: [rot.y_axis.x, rot.y_axis.y, rot.y_axis.z, 0.0],
            cz: [rot.z_axis.x, rot.z_axis.y, rot.z_axis.z, 0.0],
        }
    }

    /// Project a world unit-sphere point through the orthographic camera to
    /// clip space. Returns `None` when the point is hidden behind the planet.
    fn project(p: Vec3, rt: Mat3, aspect: f32, scale: f32) -> Option<[f32; 2]> {
        let v = rt * p;
        let occluded = v.z < 0.0 && (v.x * v.x + v.y * v.y) < 1.0;
        if occluded {
            None
        } else {
            Some([v.x / (aspect * scale), v.y / scale])
        }
    }

    fn build_overlay(&self, rot: Mat3, aspect: f32) -> Vec<OverlayVertex> {
        if self.view == View::System {
            // draw only the craft marker, projected through the perspective camera
            let mut out: Vec<OverlayVertex> = Vec::new();
            if let Some(craft) = self.flight.as_ref() {
                let cam = self.system_camera(aspect);
                let p_mm = (craft.r / MM as f64).as_vec3();
                if let Some(c) = cam.project(p_mm) {
                    let col = if craft.crashed {
                        [1.0, 0.25, 0.2]
                    } else if craft.landed {
                        [0.4, 1.0, 0.5]
                    } else {
                        [1.0, 0.85, 0.25]
                    };
                    push_marker(&mut out, c, aspect, col);
                }
            }
            return out;
        }
        let rt = rot.transpose();
        let scale = self.scale;
        let mut out: Vec<OverlayVertex> = Vec::new();

        let polyline = |pts: &[Vec3], color: [f32; 3], out: &mut Vec<OverlayVertex>| {
            let mut prev: Option<[f32; 2]> = None;
            for &p in pts {
                let cur = Self::project(p, rt, aspect, scale);
                if let (Some(a), Some(b)) = (prev, cur) {
                    out.push(OverlayVertex { pos: a, color });
                    out.push(OverlayVertex { pos: b, color });
                }
                prev = cur;
            }
        };

        // launch-pad marker on the surface
        polyline(&self.mission.pad_ring, [0.9, 0.6, 0.2], &mut out);

        let (marker_pos, marker_col) = if let Some(craft) = self.flight.as_ref() {
            let pred = craft.predicted_orbit(&self.body);
            polyline(&pred, [0.5, 0.55, 0.25], &mut out);
            let col = if craft.crashed {
                [1.0, 0.25, 0.2]
            } else if craft.landed {
                [0.4, 1.0, 0.5]
            } else {
                [1.0, 0.85, 0.25]
            };
            (craft.marker(&self.body), col)
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

            let col = if !self.launched {
                [0.4, 1.0, 0.4]
            } else if self.clock <= self.mission.meco_t {
                [1.0, 0.55, 0.15]
            } else {
                [0.5, 0.9, 1.0]
            };
            let rp = self.mission.rocket_pos(if self.launched { self.clock } else { 0.0 });
            (rp, col)
        };

        if let Some(c) = Self::project(marker_pos, rt, aspect, scale) {
            push_marker(&mut out, c, aspect, marker_col);
        }

        out
    }

    fn build_hud(&self, hud: &Hud, res: (f32, f32)) -> Vec<OverlayVertex> {
        if self.view == View::System {
            return self.build_system_hud(hud, res);
        }
        if let Some(craft) = self.flight.as_ref() {
            return self.build_flight_hud(hud, craft, res);
        }
        if !self.launched {
            return self.build_vehicle_hud(hud, res);
        }

        let mut out: Vec<OverlayVertex> = Vec::new();
        let dim = [0.55, 0.75, 0.85];
        let val = [0.92, 0.96, 1.0];
        let amber = [1.0, 0.78, 0.30];
        let x = 16.0;
        let step = hud::LINE_H;
        let row = |out: &mut Vec<OverlayVertex>, label: &str, value: &str, y: f32| {
            let cx = hud.text(out, label, x, y, dim, res);
            hud.text(out, value, cx, y, val, res);
        };

        let tel = self.mission.telemetry(self.launched, self.clock);
        let mut y = 16.0;
        hud.text(&mut out, self.mission.vehicle, x, y, amber, res);
        y += step * 1.5;

        let phase_col = match tel.phase {
            "ASCENT" => [1.0, 0.55, 0.15],
            "ORBIT" => [0.5, 0.9, 1.0],
            _ => [0.5, 1.0, 0.5],
        };
        let cx = hud.text(&mut out, "PHASE    ", x, y, dim, res);
        hud.text(&mut out, tel.phase, cx, y, phase_col, res);
        y += step;

        row(&mut out, "MET      ", &format!("T+{:.0}S", self.clock.max(0.0)), y);
        y += step;
        row(&mut out, "ALT      ", &format!("{:.1} KM", tel.alt_km), y);
        y += step;
        row(&mut out, "VEL      ", &format!("{:.0} M/S", tel.speed), y);
        y += step;
        if let Some((peri, apo)) = tel.orbit {
            row(&mut out, "ORBIT    ", &format!("{:.0} X {:.0} KM", peri, apo), y);
        } else {
            row(&mut out, "RANGE    ", &format!("{:.0} KM", tel.downrange_km), y);
        }
        y += step;
        row(
            &mut out,
            "STAGE    ",
            &format!("{}/{}", tel.stage + 1, self.mission.stage_count),
            y,
        );
        y += step;
        row(&mut out, "WARP     ", &format!("{:.0}X", self.warp), y);

        let mut hy = res.1 - step * 4.0 - 12.0;
        for line in [
            "SPACE LAUNCH/RESET",
            "F  TAKE MANUAL CONTROL",
            "DRAG ORBIT   SCROLL ZOOM",
            "[ ] TIME WARP",
        ] {
            hud.text(&mut out, line, x, hy, dim, res);
            hy += step;
        }
        out
    }

    fn build_flight_hud(&self, hud: &Hud, craft: &Craft, res: (f32, f32)) -> Vec<OverlayVertex> {
        let mut out: Vec<OverlayVertex> = Vec::new();
        let dim = [0.55, 0.75, 0.85];
        let val = [0.92, 0.96, 1.0];
        let amber = [1.0, 0.78, 0.30];
        let x = 16.0;
        let step = hud::LINE_H;
        let row = |out: &mut Vec<OverlayVertex>, label: &str, value: &str, y: f32| {
            let cx = hud.text(out, label, x, y, dim, res);
            hud.text(out, value, cx, y, val, res);
        };

        let mut y = 16.0;
        hud.text(&mut out, "MANUAL FLIGHT", x, y, amber, res);
        y += step * 1.5;

        let status = craft.status();
        let scol = match status {
            "CRASHED" => [1.0, 0.3, 0.25],
            "LANDED" => [0.4, 1.0, 0.5],
            _ => [1.0, 0.8, 0.3],
        };
        let cx = hud.text(&mut out, "STATUS   ", x, y, dim, res);
        hud.text(&mut out, status, cx, y, scol, res);
        y += step;

        row(&mut out, "ALT      ", &format!("{:.1} KM", craft.altitude(&self.body) / 1000.0), y);
        y += step;
        row(&mut out, "VEL      ", &format!("{:.0} M/S", craft.speed()), y);
        y += step;
        row(&mut out, "VSPD     ", &format!("{:.0} M/S", craft.vertical_speed()), y);
        y += step;
        row(&mut out, "THROTTLE ", &format!("{:.0}", craft.throttle * 100.0), y);
        y += step;
        row(&mut out, "PROP     ", &format!("{:.0}", craft.prop_frac() * 100.0), y);
        y += step;
        row(&mut out, "MODE     ", craft.mode.label(), y);

        let mut hy = res.1 - step * 4.0 - 12.0;
        for line in [
            "W / S  THROTTLE",
            "1 PRO  2 RETRO  3 OUT  4 IN",
            "F  RELEASE CONTROL",
            "[ ] TIME WARP",
        ] {
            hud.text(&mut out, line, x, hy, dim, res);
            hy += step;
        }
        out
    }

    fn build_vehicle_hud(&self, hud: &Hud, res: (f32, f32)) -> Vec<OverlayVertex> {
        let mut out: Vec<OverlayVertex> = Vec::new();
        let dim = [0.55, 0.75, 0.85];
        let val = [0.92, 0.96, 1.0];
        let amber = [1.0, 0.78, 0.30];
        let green = [0.5, 1.0, 0.5];
        let x = 16.0;
        let step = hud::LINE_H;
        let row = |out: &mut Vec<OverlayVertex>, label: &str, value: &str, y: f32, c: [f32; 3]| {
            let cx = hud.text(out, label, x, y, dim, res);
            hud.text(out, value, cx, y, c, res);
        };

        let m = &self.mission;
        let mut y = 16.0;
        hud.text(&mut out, "VEHICLE ASSEMBLY", x, y, amber, res);
        y += step * 1.3;
        hud.text(&mut out, m.vehicle, x, y, val, res);
        y += step * 1.3;

        // stages, top of the stack first
        for (i, (name, wet_t, dv)) in m.stack.iter().enumerate().rev() {
            let label = format!("S{} {}", i + 1, name);
            let value = format!("{:.0} T  {:.0} M/S", wet_t, dv);
            // pad the label column to align values
            let padded = format!("{:<11}", label);
            row(&mut out, &padded, &value, y, val);
            y += step;
        }
        y += step * 0.5;
        row(&mut out, "MASS     ", &format!("{:.0} T", m.liftoff_mass_t), y, val);
        y += step;
        row(&mut out, "TWR      ", &format!("{:.2}", m.liftoff_twr), y, val);
        y += step;
        row(&mut out, "DELTA-V  ", &format!("{:.0} M/S", m.total_dv), y, val);
        y += step;
        row(&mut out, "PAYLOAD  ", &format!("{:.0} T", m.payload_t), y, val);
        y += step;
        row(&mut out, "TARGET   ", &format!("{:.0} KM ORBIT", m.target_orbit_km()), y, val);
        y += step * 1.3;
        hud.text(&mut out, "SPACE  LAUNCH", x, y, green, res);

        let mut hy = res.1 - step * 3.0 - 12.0;
        for line in ["DRAG ORBIT   SCROLL ZOOM", "V  SYSTEM VIEW", "[ ] TIME WARP"] {
            hud.text(&mut out, line, x, hy, dim, res);
            hy += step;
        }
        out
    }

    fn build_system_hud(&self, hud: &Hud, res: (f32, f32)) -> Vec<OverlayVertex> {
        let mut out: Vec<OverlayVertex> = Vec::new();
        let dim = [0.55, 0.75, 0.85];
        let val = [0.92, 0.96, 1.0];
        let amber = [1.0, 0.78, 0.30];
        let x = 16.0;
        let step = hud::LINE_H;
        let row = |out: &mut Vec<OverlayVertex>, label: &str, value: &str, y: f32| {
            let cx = hud.text(out, label, x, y, dim, res);
            hud.text(out, value, cx, y, val, res);
        };

        let mut y = 16.0;
        hud.text(&mut out, "SYSTEM MAP", x, y, amber, res);
        y += step * 1.5;

        let moon_dist = self.moon_center_mm.length();
        row(&mut out, "HOME     ", &format!("R {:.0} KM", self.home_radius_mm * 1000.0), y);
        y += step;
        row(&mut out, "MOON     ", &format!("R {:.0} KM", self.moon_radius_mm * 1000.0), y);
        y += step;
        row(&mut out, "RANGE    ", &format!("{:.0} MM", moon_dist), y);
        y += step;
        row(&mut out, "CAM DIST ", &format!("{:.0} MM", self.sys_dist), y);

        if let Some(craft) = self.flight.as_ref() {
            y += step * 1.5;
            let to_moon = (craft.r - self.moon_center_m).length() / MM as f64;
            let scol = match craft.status() {
                "CRASHED" => [1.0, 0.3, 0.25],
                "LANDED" => [0.4, 1.0, 0.5],
                _ => [1.0, 0.8, 0.3],
            };
            let cx = hud.text(&mut out, "CRAFT    ", x, y, dim, res);
            let label = if craft.landed && craft.landed_on == "MOON" {
                "ON MOON"
            } else {
                craft.status()
            };
            hud.text(&mut out, label, cx, y, scol, res);
            y += step;
            row(&mut out, "TO MOON  ", &format!("{:.1} MM", to_moon), y);
        }

        let mut hy = res.1 - step * 3.0 - 12.0;
        for line in ["V  BACK TO SURFACE", "F  MANUAL CONTROL", "DRAG ORBIT   SCROLL ZOOM"] {
            hud.text(&mut out, line, x, hy, dim, res);
            hy += step;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Gpu: pipelines + buffers, independent of any window/surface.
// ---------------------------------------------------------------------------

struct Gpu {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    scene_pipeline: wgpu::RenderPipeline,
    scene_uniform_buf: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_buf: wgpu::Buffer,
    hud_pipeline: wgpu::RenderPipeline,
    hud_buf: wgpu::Buffer,
}

impl Gpu {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Gpu {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet"),
            source: wgpu::ShaderSource::Wgsl(include_str!("planet.wgsl").into()),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        // System view: same bind-group shape (uniform + planet texture +
        // sampler), different uniform struct and shader.
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
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene-pipeline"),
            layout: Some(&layout),
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

        Gpu {
            pipeline,
            uniform_buf,
            bind_group,
            scene_pipeline,
            scene_uniform_buf,
            scene_bind_group,
            overlay_pipeline,
            overlay_buf,
            hud_pipeline,
            hud_buf,
        }
    }

    /// Upload this frame's uniforms + geometry. Returns (overlay verts, hud verts).
    fn prepare(
        &self,
        queue: &wgpu::Queue,
        world: &World,
        hud: &Hud,
        w: u32,
        h: u32,
        time: f32,
    ) -> (usize, usize) {
        let res = [w as f32, h.max(1) as f32];
        match world.view {
            View::Surface => {
                let uniforms = world.uniforms(res, time);
                queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
            }
            View::System => {
                let su = world.scene_uniforms(res, time);
                queue.write_buffer(&self.scene_uniform_buf, 0, bytemuck::bytes_of(&su));
            }
        }

        let aspect = res[0] / res[1];
        let verts = world.build_overlay(world.camera_rot(), aspect);
        let n = verts.len().min(OVERLAY_CAP as usize);
        if n > 0 {
            queue.write_buffer(&self.overlay_buf, 0, bytemuck::cast_slice(&verts[..n]));
        }
        let hud_verts = world.build_hud(hud, (res[0], res[1]));
        let hn = hud_verts.len().min(HUD_CAP as usize);
        if hn > 0 {
            queue.write_buffer(&self.hud_buf, 0, bytemuck::cast_slice(&hud_verts[..hn]));
        }
        (n, hn)
    }

    fn draw(&self, pass: &mut wgpu::RenderPass, view: View, n_overlay: usize, n_hud: usize) {
        match view {
            View::Surface => {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
            }
            View::System => {
                pass.set_pipeline(&self.scene_pipeline);
                pass.set_bind_group(0, &self.scene_bind_group, &[]);
            }
        }
        pass.draw(0..3, 0..1);
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
}

/// Push a small screen-space diamond marker (line list) at clip point `c`.
fn push_marker(out: &mut Vec<OverlayVertex>, c: [f32; 2], aspect: f32, color: [f32; 3]) {
    let off = 0.022f32;
    let ox = off / aspect;
    let top = [c[0], c[1] + off];
    let right = [c[0] + ox, c[1]];
    let bot = [c[0], c[1] - off];
    let left = [c[0] - ox, c[1]];
    for (a, b) in [(top, right), (right, bot), (bot, left), (left, top)] {
        out.push(OverlayVertex { pos: a, color });
        out.push(OverlayVertex { pos: b, color });
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
    hud: Hud,
    world: World,
    start: instant_now::Instant,
    last_t: f32,
    dragging: bool,
    last_cursor: (f64, f64),
}

impl State {
    async fn new(window: Arc<Window>) -> State {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

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

        State {
            window,
            surface,
            device,
            queue,
            config,
            gpu,
            hud: Hud::new(),
            world: World::new(),
            start: instant_now::Instant::now(),
            last_t: 0.0,
            dragging: false,
            last_cursor: (0.0, 0.0),
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
        let t = self.start.elapsed().as_secs_f32();
        let frame_dt = (t - self.last_t).clamp(0.0, 0.1);
        self.last_t = t;
        self.world.advance(frame_dt);

        let (n, hn) = self
            .gpu
            .prepare(&self.queue, &self.world, &self.hud, self.config.width, self.config.height, t);

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
        {
            let mut pass = render_pass(&mut encoder, &view);
            self.gpu.draw(&mut pass, self.world.view, n, hn);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

// ---------------------------------------------------------------------------
// Headless screenshot (native only): render one framed shot to a PNG.
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn screenshot(path: &str, width: u32, height: u32, scenario: &str) {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("shot-device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
    }))
    .expect("device");

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let gpu = Gpu::new(&device, &queue, format);
    let hud = Hud::new();

    let mut world = World::new();
    let time = match scenario {
        "system" => {
            // wide system shot: home world and moon framed side by side.
            world.view = View::System;
            world.sys_az = 1.4;
            world.sys_el = 0.30;
            world.sys_dist = 120.0;
            6.0
        }
        "moon" => {
            // close-up of the moon with a landed craft marker on its near side.
            world.view = View::System;
            world.sys_focus = world.moon_center_mm;
            world.sys_az = 1.2;
            world.sys_el = 0.25;
            world.sys_dist = 4.5;
            let cam = world.system_camera(width as f32 / height as f32);
            let to_cam = (cam.pos * MM).as_dvec3() - world.moon_center_m;
            let up = to_cam.normalize();
            let mut craft = Craft::maneuvering(
                world.moon_center_m + up * world.moon_radius_m,
                DVec3::ZERO,
            );
            craft.landed = true;
            craft.landed_on = "MOON";
            world.flight = Some(craft);
            6.0
        }
        "pad" => {
            // pre-launch: the vehicle assembly panel, planet framed at the limb.
            world.view = View::Surface;
            world.launched = false;
            world.az = world.mission.spaceport_lon + 1.15;
            world.el = world.mission.spaceport_lat + 0.25;
            world.scale = 1.35;
            (world.az + 1.66 - world.mission.spaceport_lon) / 0.03
        }
        _ => {
            // surface / launch view: launched, craft coasting in its parking
            // orbit, the spaceport at the limb so the ascent arc + orbit ring
            // read clearly, sun ~95 deg off-view so the dark side shows.
            world.view = View::Surface;
            world.launched = true;
            world.clock = world.mission.meco_t + 240.0;
            world.az = world.mission.spaceport_lon + 1.15;
            world.el = world.mission.spaceport_lat + 0.25;
            world.scale = 1.35;
            (world.az + 1.66 - world.mission.spaceport_lon) / 0.03
        }
    };

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

    let (n, hn) = gpu.prepare(&queue, &world, &hud, width, height, time);

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
    {
        let mut pass = render_pass(&mut encoder, &target_view);
        gpu.draw(&mut pass, world.view, n, hn);
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
            let state = State::new(win).await;
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
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
                state.window.request_redraw();
            }
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                if button == MouseButton::Left {
                    state.dragging = btn_state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                if state.dragging {
                    let dx = (x - state.last_cursor.0) as f32;
                    let dy = (y - state.last_cursor.1) as f32;
                    match state.world.view {
                        View::Surface => {
                            state.world.az -= dx * 0.005;
                            state.world.el = (state.world.el + dy * 0.005).clamp(-1.5, 1.5);
                        }
                        View::System => {
                            state.world.sys_az -= dx * 0.005;
                            state.world.sys_el = (state.world.sys_el + dy * 0.005).clamp(-1.5, 1.5);
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
                    View::Surface => {
                        state.world.scale =
                            (state.world.scale * (1.0 - dy * 0.12)).clamp(0.12, 3.0);
                    }
                    View::System => {
                        state.world.sys_dist =
                            (state.world.sys_dist * (1.0 - dy * 0.12)).clamp(15.0, 600.0);
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
                    if key_event.repeat {
                        return;
                    }
                    match code {
                        KeyCode::KeyV => state.world.toggle_view(),
                        KeyCode::KeyF => state.world.toggle_flight(),
                        KeyCode::Space if state.world.flight.is_none() => {
                            state.world.toggle_launch()
                        }
                        KeyCode::BracketRight => {
                            state.world.warp = (state.world.warp * 2.0).min(256.0);
                        }
                        KeyCode::BracketLeft => {
                            state.world.warp = (state.world.warp * 0.5).max(1.0);
                        }
                        KeyCode::Digit1 => {
                            if let Some(c) = state.world.flight.as_mut() {
                                c.mode = Mode::Prograde;
                            }
                        }
                        KeyCode::Digit2 => {
                            if let Some(c) = state.world.flight.as_mut() {
                                c.mode = Mode::Retrograde;
                            }
                        }
                        KeyCode::Digit3 => {
                            if let Some(c) = state.world.flight.as_mut() {
                                c.mode = Mode::RadialOut;
                            }
                        }
                        KeyCode::Digit4 => {
                            if let Some(c) = state.world.flight.as_mut() {
                                c.mode = Mode::RadialIn;
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
    // Native: `app shot [out.png]` renders one headless frame and exits.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if args.iter().any(|a| a == "shot" || a == "--shot") {
            env_logger::init();
            let scenario = if args.iter().any(|a| a == "moon") {
                "moon"
            } else if args.iter().any(|a| a == "system") {
                "system"
            } else if args.iter().any(|a| a == "pad") {
                "pad"
            } else {
                "surface"
            };
            let default = match scenario {
                "moon" => "out/moon.png",
                "system" => "out/system.png",
                "pad" => "out/pad.png",
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
