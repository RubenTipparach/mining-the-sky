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

mod bot;
mod build;
mod flight;
mod launch;
mod mission;
mod rocket;
mod sim_thread;
mod terrain_job;
mod traffic;
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
/// Dynamic rocket-view geometry (pad + rocket + spent booster, or a surface
/// mesh: moon base / cargo module / a full procedural asteroid ~66k verts).
const DYN_MESH_CAP: u64 = 560_000;
/// Procedural re-entry plasma glow mesh (prototype mesh approach): an isosurface
/// shell hugging the vehicle SDF (surface nets), so it can run to tens of
/// thousands of verts on a boostered stack.
const PLASMA_MESH_CAP: u64 = 262_144;
// ---- Re-entry glow-mesh look (play with these) ----
/// Number of nested glow shells extracted from the vehicle SDF. More shells read
/// as a softer, more volumetric fireball (the inner one hugs the hull, the outer
/// ones are progressively fainter, offset wisps).
const PLASMA_GLOW_LAYERS: usize = 8;
/// How far the outermost shell stands off the hull, as a multiple of the glow
/// standoff. Bigger = the glow diffuses further out (softer, puffier silhouette).
const PLASMA_GLOW_SPREAD: f32 = 1.9;
/// Full-planet LOD terrain (rebuilt as the rocket moves across the grid). Sized
/// for the high-detail budget (~1-2M triangles = 3-6M non-indexed vertices); the
/// GPU buffer is this many vertices, so it bounds the densest terrain frame.
const TERRAIN_CAP: u64 = 4_500_000;

/// Floating-origin grid for the planet terrain. The reference is snapped to a
/// lattice so the same rocket position always yields byte-identical geometry
/// (no shimmer) and the mesh only rebuilds when the rocket crosses a cell. The
/// cell size grows with altitude (finer near the ground), snapped to a power of
/// two so it changes only at discrete altitude octaves rather than drifting.
/// The min is ~1 km so slow, low movement never thrashes the mesh; the max
/// keeps rebuilds rare from orbit.
const TERRAIN_GRID_MIN_M: f64 = 1_024.0;
const TERRAIN_GRID_MAX_M: f64 = 1_048_576.0;

/// Altitude (m) above which the planet terrain transitions to the stable coarse
/// cube-sphere LOD regime: rebuild cells widen super-linearly so the high
/// altitude globe stops re-meshing on every kilometre of travel. ~50 km, where
/// the surface no longer shows resolvable fine relief.
const HIGH_ALT_STABLE_M: f64 = 50_000.0;

/// Render-space length unit for the system view: 1000 km.
const MM: f32 = 1.0e6;

/// Local-frame position (metres) of the assembly building. It sits ~5 km from
/// the pad (which is at the origin); the rocket rolls out across the flats to it.
const HANGAR_POS: Vec3 = Vec3::new(-5000.0, 0.0, 0.0);
const RACK_POS: Vec3 = Vec3::new(-5000.0, 0.0, 42.0);

/// Centre of the main metro (local metres, east + north of the pad): the middle
/// downtown of cluster 0, reached by the access road, where you spawn the car.
const CITY_CENTER: Vec3 = Vec3::new(1600.0, 0.0, 1800.0);

/// Spacing between neighbouring downtowns inside a metro cluster (metres). A touch
/// wider than a single 420 m city grid, so the grids nearly meet and read as one
/// continuous sprawl while only one or two NPC crowds wake at a time.
const CITY_SPACING: f32 = 460.0;

/// The metro clusters: (centre, cols, rows, variant base). Each is a dense block
/// of `cols * rows` downtowns packed together into one big sprawl; the clusters
/// sit kilometres apart so only the metro you are in renders / simulates. Cluster
/// 0 is the main metro by the launch pad.
const CLUSTERS: [(Vec3, i32, i32, u32); 3] = [
    (CITY_CENTER, 3, 3, 0),
    (Vec3::new(-2900.0, 0.0, 2700.0), 3, 2, 20),
    (Vec3::new(3200.0, 0.0, -2600.0), 2, 3, 40),
];

/// Render a city when the camera focus is within this range (catches a whole
/// metro at once, so a cluster reads as one continuous sprawl with no pop-in).
const CITY_RENDER_R: f64 = 2300.0;

/// The downtown grid of one metro cluster, centred on `center` (each entry is the
/// city centre + its layout variant, so neighbouring downtowns look distinct).
fn cluster_cities(center: Vec3, cols: i32, rows: i32, vbase: u32) -> Vec<(Vec3, u32)> {
    let mut v = Vec::new();
    let ox = (cols as f32 - 1.0) * 0.5;
    let oz = (rows as f32 - 1.0) * 0.5;
    for c in 0..cols {
        for r in 0..rows {
            let p = center + Vec3::new((c as f32 - ox) * CITY_SPACING, 0.0, (r as f32 - oz) * CITY_SPACING);
            v.push((p, vbase + (c * rows + r) as u32));
        }
    }
    v
}

/// Every drivable downtown across all metro clusters (centre + layout variant).
fn all_cities() -> Vec<(Vec3, u32)> {
    let mut v = Vec::new();
    for &(center, cols, rows, vbase) in &CLUSTERS {
        v.extend(cluster_cities(center, cols, rows, vbase));
    }
    v
}

/// The road network: a curbed access road from the pad to the main metro, lean
/// inter-metro highways wiring the cluster centres into a loop, and avenues
/// linking neighbouring downtowns inside each metro. Each entry is (a, b, mesh)
/// so the renderer can distance-cull a link the player is nowhere near.
fn build_city_roads() -> Vec<(Vec3, Vec3, rocket::Mesh)> {
    let pad = Vec3::new(40.0, 0.0, 0.0);
    let mut roads = Vec::new();
    // access road from the launch complex to the main metro (with curbs).
    roads.push((pad, CITY_CENTER, rocket::road_l(pad, CITY_CENTER, 10.0, true)));
    // inter-metro highways: link the cluster centres into a loop.
    let n = CLUSTERS.len();
    for i in 0..n {
        let (a, b) = (CLUSTERS[i].0, CLUSTERS[(i + 1) % n].0);
        roads.push((a, b, rocket::road_l(a, b, 9.0, false)));
    }
    // avenues between neighbouring (axis-adjacent) downtowns within each metro.
    let max = (CITY_SPACING * 1.3) as f64;
    for &(center, cols, rows, vbase) in &CLUSTERS {
        let cs = cluster_cities(center, cols, rows, vbase);
        for i in 0..cs.len() {
            for j in (i + 1)..cs.len() {
                if (cs[i].0 - cs[j].0).length() as f64 <= max {
                    roads.push((cs[i].0, cs[j].0, rocket::road_l(cs[i].0, cs[j].0, 9.0, false)));
                }
            }
        }
    }
    roads
}

/// Interior work lights of the assembly building: (offset from HANGAR_POS,
/// colour*intensity, range metres). Mounted high on the wall corners (cool) and
/// under the roof (warm) - kept out near the structure, not beside the rocket.
const HANGAR_LIGHTS: [(Vec3, [f32; 3], f32); 6] = [
    (Vec3::new(50.0, 92.0, 50.0), [0.85, 0.95, 1.25], 80.0),
    (Vec3::new(-50.0, 92.0, 50.0), [0.85, 0.95, 1.25], 80.0),
    (Vec3::new(50.0, 92.0, -50.0), [0.85, 0.95, 1.25], 80.0),
    (Vec3::new(-50.0, 92.0, -50.0), [0.85, 0.95, 1.25], 80.0),
    (Vec3::new(28.0, 144.0, 0.0), [1.3, 1.05, 0.7], 95.0),
    (Vec3::new(-28.0, 144.0, 0.0), [1.3, 1.05, 0.7], 95.0),
];

/// Depth format for the rocket view's mesh pass.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// GLSL-style smoothstep for f32 (Hermite ease between edges e0..e1).
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Polynomial smooth-minimum (iq): blends two SDF values over a radius `k` so a
/// union of overlapping shapes reads as one creaseless surface (no faceted ridge
/// where the parts meet). Used to fuse the smeared wake copies of the hull SDF.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k.max(1e-6)).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

/// Signed distance to a round cone (capsule with end radii `r1`..`r2`) from `a`
/// to `b`. Rust port of the iq round-cone SDF, for building the re-entry plasma
/// the glow-mesh isosurface on the CPU.
fn sd_round_cone(p: Vec3, a: Vec3, b: Vec3, r1: f32, r2: f32) -> f32 {
    let ba = b - a;
    let l2 = ba.dot(ba).max(1e-6);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;
    let pa = p - a;
    let y = pa.dot(ba);
    let z = y - l2;
    let d2v = pa * l2 - ba * y;
    let x2 = d2v.dot(d2v);
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = rr.signum() * rr * rr * x2;
    if z.signum() * a2 * z2 > k {
        return (x2 + z2).sqrt() * il2 - r2;
    }
    if y.signum() * a2 * y2 < k {
        return (x2 + y2).sqrt() * il2 - r1;
    }
    (x2 * a2 * il2).sqrt() * il2 + y * rr * il2 - r1
}

#[derive(Clone, Copy, PartialEq, Debug)]
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
    /// params.x = log-depth Fcoef, y = time, z = active point-light count.
    params: [f32; 4],
    /// rgb = horizon haze colour, w = fog density (1/visibility metres).
    fog: [f32; 4],
    /// Interior point lights: xyz = position (camera-relative), w = range (m).
    lights: [[f32; 4]; 8],
    /// rgb = colour * intensity for each light.
    light_col: [[f32; 4]; 8],
    /// Procedural surface detail (asteroid/airless bodies): xyz = body centre in
    /// camera-relative space, w = body radius (m). w = 0 disables.
    detail: [f32; 4],
}

const MAX_LIGHTS: usize = 8;

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

/// Max engine nozzles for the volumetric plume pass (main + radial boosters).
const MAX_PLUME_NOZZLES: usize = 12;

/// Uniforms for the volumetric exhaust-plume pass (camera-relative scene space).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PlumeUniforms {
    right: [f32; 4],
    up: [f32; 4],
    fwd: [f32; 4],
    eye: [f32; 4],
    /// xyz = bounding centre (camera-relative), w = bounding radius.
    center: [f32; 4],
    /// xyz = exhaust direction (unit), w = plume length.
    dir: [f32; 4],
    /// x = tan(fov/2), y = aspect, z = time, w = intensity (0..1).
    params: [f32; 4],
    /// x = nozzle count, y = base radius.
    nnoz: [f32; 4],
    /// Per nozzle: xyz position (camera-relative), w = per-nozzle radius scale.
    noz: [[f32; 4]; MAX_PLUME_NOZZLES],
}

/// Far plane for the rocket-view log-depth buffer (m): beyond planet diameter.
const LOG_DEPTH_FAR: f32 = 2.0e7;

/// Horizon haze colour shared by the sky and the terrain aerial-perspective fog.
const HORIZON: [f32; 3] = [0.74, 0.82, 0.93];

/// Sun direction in the launch tangent frame (east, up, north). The home world
/// uses a fairly high sun; the moon uses a low, grazing sun for long shadows.
const SUN_LOCAL: Vec3 = Vec3::new(0.40, 0.72, 0.55);
/// Night sun: well below the horizon, so the atmosphere shader renders a dark
/// sky and the mesh shader's day/night factor collapses the sky-fill ambient.
const SUN_LOCAL_NIGHT: Vec3 = Vec3::new(0.42, -0.40, 0.50);
const SUN_LOCAL_MOON: Vec3 = Vec3::new(0.62, 0.17, 0.77);
/// Deep-space (asteroid) sun: higher than the lunar grazing sun so a body
/// viewed from the sun side reads as a solid lit rock.
const SUN_LOCAL_SPACE: Vec3 = Vec3::new(0.45, 0.58, 0.68);

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

/// A jettisoned stage tumbling away after separation, integrated ballistically
/// in the rocket-view local frame (metres) for a few seconds of spectacle.
struct SepBooster {
    pos: Vec3,
    vel: Vec3,
    rot: Quat,
    spin: Vec3,
    /// Local-frame gravitational acceleration at the separation point (m/s^2),
    /// so the spent stage falls away under the *actual* gravity at altitude
    /// (gentle high up, in the correct down-direction) instead of a fixed 1 g.
    grav: Vec3,
    age: f32,
    /// Vertex range (in the rocket body mesh) of the jettisoned stage.
    range: std::ops::Range<usize>,
}

/// One chunk of a destroyed vehicle, tumbling away from the blast in the
/// rocket-view local frame. Like `SepBooster` but many at once, each a vertex
/// range of the original rocket mesh.
struct Debris {
    pos: Vec3,
    vel: Vec3,
    rot: Quat,
    spin: Vec3,
    grav: Vec3,
    age: f32,
    range: std::ops::Range<usize>,
}

/// One fireball/smoke particle of an explosion. Drawn through the FX smoke
/// pipeline, but colour-ramped from white-hot through orange to dark smoke as it
/// ages, so a burst reads as a fireball collapsing into a smoke cloud.
struct Boom {
    pos: Vec3,
    vel: Vec3,
    age: f32,
    life: f32,
    size0: f32,
    seed: f32,
}

/// A planned maneuver (burn node) on the craft's current orbit: where to burn
/// (true anomaly `nu`) and the prograde / normal / radial delta-v (m/s).
#[derive(Clone, Copy)]
struct ManeuverNode {
    nu: f64,
    pro: f64,
    nrm: f64,
    rad: f64,
}

/// A payload delivered to orbit by a completed mission. Persists and is drawn
/// circling the home world in the map view; missions accumulate these.
struct OrbitObject {
    name: &'static str,
    color: [f32; 3],
    radius_mm: f32, // orbit radius (Mm, home-centred)
    t1: Vec3,       // orbit-plane basis (home-centred world unit vectors)
    t2: Vec3,
    rate: f32,    // rad/s
    phase0: f32,  // angle at epoch
    epoch: f64,   // sys_time at insertion
}

/// A grabbable part on the VAB rack: its local position, kind and catalog index.
#[derive(Clone, Copy)]
struct RackSlot {
    pos: Vec3,
    kind: rocket::PartKind,
    idx: usize,
}

/// Build the parts rack beside the hangar from the catalog: rows of engines,
/// tanks and payloads as 3D models you can grab. Returns the mesh + slot table.
fn build_rack() -> (rocket::Mesh, Vec<RackSlot>) {
    let mut m = rocket::Mesh::default();
    let mut slots = Vec::new();
    let base = RACK_POS;
    let rows: [(rocket::PartKind, usize, [f32; 3], f32); 3] = [
        (rocket::PartKind::Engine, build::ENGINES.len(), [0.55, 0.55, 0.60], 3.0),
        (rocket::PartKind::Tank, build::TANKS.len(), [0.82, 0.82, 0.86], 6.4),
        (rocket::PartKind::Payload, build::PAYLOADS.len(), [0.88, 0.80, 0.42], 9.6),
    ];
    for (kind, n, col, y) in rows {
        let w = n as f32 * 3.2;
        // shelf bar + back panel
        rocket::append_box(&mut m, base + Vec3::new(0.0, y - 1.4, 0.0), Vec3::new(w * 0.5, 0.2, 1.6), [0.28, 0.29, 0.33]);
        for k in 0..n {
            let x = -(w * 0.5) + 1.6 + k as f32 * 3.2;
            let p = base + Vec3::new(x, y, 0.0);
            rocket::append_part(&mut m, kind, p, col);
            slots.push(RackSlot { pos: p, kind, idx: k });
        }
    }
    // back wall of the rack
    rocket::append_box(&mut m, base + Vec3::new(0.0, 6.0, 1.6), Vec3::new(22.0, 7.0, 0.2), [0.24, 0.25, 0.28]);
    (m, slots)
}

/// Nearest ray-sphere hit distance (or None).
fn ray_sphere_near(o: Vec3, d: Vec3, c: Vec3, r: f32) -> Option<f32> {
    let oc = o - c;
    let b = oc.dot(d);
    let cc = oc.dot(oc) - r * r;
    let disc = b * b - cc;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t > 0.0 {
        Some(t)
    } else {
        None
    }
}

impl OrbitObject {
    /// World position (Mm) at `sys_time`, given the home world's position.
    fn pos_mm(&self, home_pos: DVec3, sys_time: f64) -> DVec3 {
        let th = self.phase0 + self.rate * (sys_time - self.epoch) as f32;
        home_pos + (self.t1 * th.cos() + self.t2 * th.sin()).as_dvec3() * self.radius_mm as f64
    }
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

/// One re-entry spark: a tiny bright ember shed off the windward shock and
/// streaming downstream through the wake, in the rocket-view local frame.
struct Spark {
    pos: Vec3,
    vel: Vec3,
    age: f32,
    life: f32,
    seed: f32,
}

struct World {
    mission: Mission,
    body: CentralBody,
    flight: Option<Craft>,
    /// When engaged, the autonomous moon-landing bot flies `flight` for you.
    moonbot: Option<bot::MoonBot>,
    /// Planned burn node for the craft (the maneuver planner).
    node: Option<ManeuverNode>,
    launched: bool,
    clock: f32, // mission-elapsed seconds
    warp: f32,

    // player-controlled launch (KSP-style); replaces the on-rails ascent when
    // the player flies it from the pad in the rocket view.
    launch: Option<launch::Rocket>,
    /// The current vehicle design (Vehicle Assembly Building).
    vab: build::Vab,
    /// In the assembly building (true) vs out on the pad (false).
    vab_mode: bool,
    /// Roll-out progress: 0 = in the hangar, 1 = on the pad. Animates.
    rollout: f32,
    rolling_out: bool,
    /// Crawler speed multiplier while rolling out (1x .. 64x): lets the player
    /// fast-forward the slow transport instead of watching it creep.
    rollout_speed: f32,
    rocket_body: rocket::RocketBody,
    pad_mesh: rocket::Mesh,
    hangar_mesh: rocket::Mesh,
    rack_mesh: rocket::Mesh,
    /// The crawlerway road (static) and the mobile launch platform that carries
    /// the rocket along it (drawn at the rocket's resting base while not flown).
    road_mesh: rocket::Mesh,
    platform_mesh: rocket::Mesh,
    /// The drivable cities (centre + building mesh) and the road network linking
    /// them to each other and the launch complex (each road is (a, b, mesh) so it
    /// can be distance-culled). All static, in launch-tangent local metres.
    city_meshes: Vec<(Vec3, rocket::Mesh)>,
    /// Night-only warm ground lighting per city (amber street wash + lamp pools),
    /// drawn only when `night` so its amber albedo does not tint the day streets.
    city_glow: Vec<(Vec3, rocket::Mesh)>,
    roads: Vec<(Vec3, Vec3, rocket::Mesh)>,
    car_mesh: rocket::Mesh,
    lander_mesh: rocket::Mesh,
    /// Show the lunar lander standing on the ground (instead of the rocket).
    show_lander: bool,
    /// An assembled/previewed moon base mesh to draw on the surface, if any.
    base_mesh: Option<rocket::Mesh>,
    /// Fairing clamshell open fraction (0 = closed, 1 = halves fully swung out
    /// revealing the cargo module).
    fairing_open: f32,
    /// Show the MOON BASE structures-catalog panel (only for the full colony,
    /// not single delivered modules).
    base_panel: bool,
    /// Deep-space scene (asteroid): suppress the planet terrain and render a
    /// pure starfield sky around the body at the origin.
    space: bool,
    /// Name shown for the body being inspected in a deep-space scene.
    space_label: &'static str,
    /// When set, the asteroid is rendered through the LOD quadtree (detail
    /// refines as the camera approaches), centred at the local origin, using
    /// this elevation field and base radius.
    ast_elev: Option<terrain::Elevation>,
    ast_radius: f64,
    /// Render the surface as the moon: grey regolith + black airless sky.
    lunar: bool,
    /// Night on the home world: a below-horizon sun, so the sky goes dark and
    /// the emissive city lights carry the scene.
    night: bool,
    /// Height (m) the lander floats above the surface (0 = landed).
    lander_alt: f32,
    /// Fire the lander's descent engine (plume under the bell).
    lander_firing: bool,
    /// RCS attitude-thruster activity (0 = idle, 1 = full puff). Drives the
    /// blue-white RCS jets around the lander's upper body.
    lander_rcs: f32,
    /// Grabbable parts on the rack (for in-viewport 3D drag-assembly).
    rack_slots: Vec<RackSlot>,
    /// The part currently being dragged from the rack, if any.
    grab: Option<RackSlot>,
    /// The stack slot the dragged part is hovering (kind + stage), if any.
    grab_target: Option<(rocket::PartKind, usize)>,
    /// Drag ghost position (camera-relative) while grabbing.
    grab_ghost: Vec3,
    sep: Option<SepBooster>,
    /// Chunks of a destroyed vehicle + the fireball particles of its explosion,
    /// and a latch so the blast spawns exactly once.
    debris: Vec<Debris>,
    boom: Vec<Boom>,
    exploded: bool,
    /// Payloads delivered to orbit by completed missions (they accumulate).
    orbits: Vec<OrbitObject>,
    /// Whether the active launch's payload has been captured to orbit yet.
    mission_captured: bool,
    smoke: Vec<Smoke>,
    smoke_accum: f32, // fractional particle spawn carry
    /// Re-entry sparks shed off the plasma shock, streaming through the wake.
    sparks: Vec<Spark>,
    spark_accum: f32, // fractional spark spawn carry
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
    /// LOD-debug overlay: colour the terrain by quadtree depth (toggle with `L`
    /// in the rocket view) so the split rings are visible and tunable.
    lod_debug: bool,
    /// Vertex count of the re-entry plasma glow mesh built this frame.
    plasma_mesh_n: u32,
    /// Frozen re-entry test scene: the vehicle is held at full heat in place so
    /// the plasma FX play continuously and you can orbit / re-aim it to inspect
    /// the shock. No physics integration, no heat damage, never destroyed. Set
    /// by `setup_reentry`, cleared on any normal ignite / reset.
    reentry_test: bool,
    /// Friction (heating) level dialed by the test-scene slider, 0..1. In the
    /// frozen test it drives the vehicle's heat glow directly so you can sweep the
    /// re-entry FX from cold to white-hot and confirm it ramps smoothly.
    test_friction: f32,
    /// Background planet-terrain mesher (double-buffered). See [`terrain_job`].
    terrain_svc: terrain_job::TerrainService,

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
    /// Smoothed model-Y the launched camera aims at: it lerps toward the centre
    /// of the still-attached geometry, so after a stage drops the framing
    /// re-centres on the remaining stack instead of jumping. 0 = uninitialised.
    cam_focus_y: f32,

    universe: Universe,
    /// Indices into `universe.bodies` of the navigable bodies (stars + planets).
    nav: Vec<usize>,
    /// Index (into `universe.bodies`) of the focused body.
    focus: usize,
    /// When set, the map camera frames the active vehicle instead of `focus`.
    focus_rocket: bool,
    /// egui body-browser search text.
    ui_search: String,

    /// Dedicated physics thread that owns and integrates the player launch on its
    /// own wall-clock, separate from rendering (live native game only). When set,
    /// `launch` is a render-side snapshot mirror: control inputs are forwarded to
    /// the thread and the authoritative state is read back each frame. `None` in
    /// tests / headless `shot` / `ascentcsv` and on wasm, where the launch is
    /// integrated inline and synchronously so those paths stay deterministic.
    sim: Option<sim_thread::SimThread>,
    /// True while the dedicated thread is actively flying the player launch. The
    /// frozen / hand-placed test scenes (re-entry, parachute, powered descent)
    /// hand-place state and integrate synchronously, so they detach the thread
    /// (clear this) and run inline; a normal ignition re-arms it.
    sim_drives_launch: bool,

    /// A drivable car you can take from the launch complex out to the city. When
    /// `driving`, WASD steers it and the rocket-view camera follows it.
    driving: bool,
    car_pos: DVec3, // local launch-tangent metres (x east, y up, z north)
    car_yaw: f32,   // heading, 0 = +X (east)
    car_speed: f32, // signed m/s along the heading
    /// Held drive keys: [forward, back, left, right].
    drive_in: [bool; 4],

    /// On foot: a third-person character you can walk around (and use to get in
    /// and out of the car). When `walking`, WASD moves it and the camera follows.
    walking: bool,
    ped_pos: DVec3, // local metres
    ped_yaw: f32,   // heading, 0 = +X
    ped_speed: f32, // m/s along the heading
    walk_phase: f32, // stride cycle phase (for the walk animation)
    /// Held walk keys: [forward, back, left, right].
    walk_in: [bool; 4],

    /// NPC cars + pedestrians that bring each city to life. One crowd per city,
    /// each spawned only while the player is near it and despawned when far away.
    traffic: Vec<traffic::Traffic>,
}

impl World {
    fn new() -> World {
        let mission = Mission::pioneer_from_spaceport();
        let body = CentralBody::home();
        let vab = build::Vab::default_build();
        let init_boosters: Vec<rocket::BoosterViz> = (0..vab.stages.len())
            .map(|i| {
                let (b, n) = vab.booster(i);
                rocket::BoosterViz { count: n, prop: b.prop as f32, solid: b.solid }
            })
            .collect();
        let rocket_body = rocket::rocket_body(
            &vab.to_vehicle(),
            vab.payload().color,
            vab.payload().module,
            &init_boosters,
        );
        let rocket_frame = (rocket_body.focus_y, rocket_body.cam_dist);
        let (rack_mesh_init, rack_slots) = build_rack();
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
            moonbot: None,
            node: None,
            vab,
            vab_mode: true,
            rollout: 0.0,
            rolling_out: false,
            rollout_speed: 1.0,
            launch: None,
            rocket_body,
            pad_mesh: rocket::pad_and_mount(),
            hangar_mesh: rocket::hangar(HANGAR_POS, &HANGAR_LIGHTS.map(|l| l.0)),
            rack_mesh: rack_mesh_init,
            // road from the hangar door out past the pad, with the platform that
            // crawls the rocket along it.
            road_mesh: rocket::crawlerway(HANGAR_POS.x, 12.0, 14.0),
            platform_mesh: rocket::crawler_platform(),
            city_meshes: all_cities().iter().map(|&(c, v)| (c, rocket::city(c, v))).collect(),
            city_glow: all_cities().iter().map(|&(c, _)| (c, rocket::city_night_glow(c))).collect(),
            roads: build_city_roads(),
            car_mesh: rocket::car([0.80, 0.18, 0.16]),
            lander_mesh: rocket::lander(),
            show_lander: false,
            base_mesh: None,
            fairing_open: 0.0,
            base_panel: false,
            space: false,
            space_label: "",
            ast_elev: None,
            ast_radius: 0.0,
            lunar: false,
            night: false,
            lander_alt: 0.0,
            lander_firing: false,
            lander_rcs: 0.0,
            rack_slots,
            grab: None,
            grab_target: None,
            grab_ghost: Vec3::ZERO,
            sep: None,
            debris: Vec::new(),
            boom: Vec::new(),
            exploded: false,
            orbits: Vec::new(),
            mission_captured: false,
            smoke: Vec::new(),
            smoke_accum: 0.0,
            sparks: Vec::new(),
            spark_accum: 0.0,
            anim: 0.0,
            launch_origin,
            launch_up,
            launch_east,
            launch_north,
            ref_local: DVec3::ZERO,
            terrain_dirty: true,
            lod_debug: false,
            plasma_mesh_n: 0,
            reentry_test: false,
            test_friction: 0.9,
            terrain_verts: Vec::new(),
            terrain_count: 0,
            terrain_svc: terrain_job::TerrainService::new(),
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
            rocket_az: 0.7, // start inside the assembly building, facing the rocket
            rocket_el: 0.18,
            rocket_dist: 52.0,
            rocket_focus_y: rocket_frame.0,
            cam_focus_y: 0.0,
            universe: Universe { bodies: Vec::new() },
            nav: Vec::new(),
            focus: 0,
            focus_rocket: false,
            ui_search: String::new(),
            sim: None,
            sim_drives_launch: false,
            driving: false,
            car_pos: DVec3::new(30.0, 0.0, 0.0),
            car_yaw: 0.0,
            car_speed: 0.0,
            drive_in: [false; 4],
            walking: false,
            ped_pos: DVec3::new(34.0, 0.0, 4.0),
            ped_yaw: 0.0,
            ped_speed: 0.0,
            walk_phase: 0.0,
            walk_in: [false; 4],
            traffic: all_cities().iter().map(|&(c, _)| traffic::Traffic::new(c.as_dvec3())).collect(),
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
        // default focus: the active vehicle (falls back to the launch site on
        // the home world before launch).
        w.focus = w.universe.home_index();
        w.focus_rocket = true;
        w.apply_focus();
        w
    }

    /// Spin up the dedicated physics thread and route the player launch through
    /// it. Called only for the live native game (the windowed run): tests,
    /// headless `shot` / `ascentcsv` and wasm leave `sim` as `None` and integrate
    /// the launch inline + synchronously so they stay deterministic.
    fn enable_threaded_sim(&mut self) {
        self.sim = Some(sim_thread::SimThread::spawn());
    }

    /// Release the dedicated thread from the current launch and integrate inline
    /// instead. Used by the hand-placed / frozen test scenes, which set state
    /// directly and step synchronously.
    fn detach_sim_launch(&mut self) {
        if let Some(sim) = self.sim.as_ref() {
            sim.stop();
        }
        self.sim_drives_launch = false;
    }

    /// Whether the dedicated thread should be driving the launch this frame.
    fn sim_thread_active(&self) -> bool {
        self.sim.is_some() && self.sim_drives_launch
    }

    /// Snapshot the render-side control inputs to hand to the sim thread.
    fn launch_controls(&self, rk: &launch::Rocket) -> sim_thread::Controls {
        sim_thread::Controls {
            throttle: rk.throttle,
            pitch: rk.pitch,
            yaw: rk.yaw,
            roll: rk.roll,
            attitude_hold: rk.attitude_hold,
            auto_land: rk.auto_land,
            chute_armed: rk.chute_armed,
            frozen: self.reentry_test,
            test_heat: (self.test_friction * 1.4) as f64,
            warp: self.warp,
        }
    }

    /// The currently focused body.
    fn focus_body(&self) -> &Body {
        &self.universe.bodies[self.focus]
    }

    fn focus_label(&self) -> &str {
        if self.focus_rocket {
            "ACTIVE VEHICLE"
        } else {
            self.focus_body().name.as_str()
        }
    }

    /// The active vehicle's position in system (Mm) coords: the home world's
    /// orbital position plus the rocket/craft offset (or the launch site before
    /// launch), converting home-centred metres to Mm.
    fn rocket_focus_pos(&self) -> DVec3 {
        let home = self.universe.position(self.universe.home_index(), self.sys_time);
        let r = self
            .launch
            .as_ref()
            .map(|rk| rk.r)
            .or_else(|| self.flight.as_ref().map(|c| c.r))
            .unwrap_or(self.launch_origin);
        home + r / MM as f64
    }

    /// Frame the active vehicle in the map view.
    fn set_focus_rocket(&mut self) {
        self.focus_rocket = true;
        self.apply_focus();
    }

    /// True when the map camera is pulled back to system scale (well beyond the
    /// home world), where clicking bodies to focus them makes sense. Close in -
    /// framing the vehicle or home world - click-to-focus is disabled.
    fn in_system_view(&self) -> bool {
        self.sys_dist > self.home_radius_mm as f64 * 8.0
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
            self.focus_rocket = false;
            self.focus = body_idx;
            self.apply_focus();
        }
    }

    /// World position of the current focus target at the current sim time.
    fn focus_pos(&self) -> DVec3 {
        if self.focus_rocket {
            self.rocket_focus_pos()
        } else {
            self.universe.position(self.focus, self.sys_time)
        }
    }

    fn apply_focus(&mut self) {
        self.sys_focus = self.focus_pos();
        self.sys_dist = if self.focus_rocket {
            // close enough to frame the home world and the vehicle's near orbit
            (self.home_radius_mm as f64 * 4.0).max(2.0)
        } else {
            (self.focus_body().radius * 4.0).max(2.0)
        };
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

    /// Toggle the LOD-debug terrain colouring and force an immediate rebuild so
    /// the new colours appear without waiting for the next floating-origin snap.
    fn toggle_lod_debug(&mut self) {
        self.lod_debug = !self.lod_debug;
        if self.view == View::Rocket {
            self.rebuild_terrain();
            self.terrain_dirty = true;
        }
    }

    /// Drop the live view straight into a re-entry test: a vehicle at full heating
    /// in the upper atmosphere. `kind` selects the attitude - 0 = axial
    /// (nose-first), 1 = pitched-over (high angle of attack), 2 = broadside. The
    /// shot scenarios `reentry`, `reentry_tilt`, `reentry_side` use the same setup,
    /// and the "Test Scenes" UI menu calls this directly (no hotkey).
    fn setup_reentry(&mut self, kind: u8) {
        self.view = View::Rocket;
        self.ignite_launch();
        // Hand-placed, frozen scene: integrate inline, not on the thread.
        self.detach_sim_launch();
        let radius = self.body.radius;
        let up = self.launch_up;
        let east = self.launch_east;
        if let Some(rk) = self.launch.as_mut() {
            rk.throttle = 0.0;
            match kind {
                0 => {
                    rk.r = up * (radius + 58_000.0);
                    rk.v = -up * 6_000.0 + east * 2_600.0;
                    rk.pitch = 0.0;
                }
                1 => {
                    rk.r = up * (radius + 58_000.0);
                    rk.pitch = 0.8;
                    rk.v = east * 5_400.0 - up * 3_200.0;
                }
                _ => {
                    rk.r = up * (radius + 55_000.0);
                    rk.pitch = 0.0;
                    rk.v = east * 6_600.0;
                }
            }
            // hold the set-up attitude (so a broadside/tilt entry is not weathervaned
            // away during the bloom) and place the nose there immediately.
            rk.attitude_hold = true;
            rk.place_attitude();
        }
        // Integrate a short while so the heating glow blooms to full, then
        // freeze the scene: from here the vehicle holds its pose at full heat
        // (see the `reentry_test` branch in `advance`) so the FX run forever and
        // you can orbit / pitch it instead of watching it fall and burn up.
        for _ in 0..12 {
            self.advance(0.1); // ~1.2 s for the plasma to bloom
        }
        if let Some(rk) = self.launch.as_mut() {
            rk.health = 100.0; // pristine airframe for the test
            rk.destroyed = false;
            rk.crashed = false;
            rk.throttle = 0.0;
        }
        self.exploded = false;
        self.reentry_test = true;
        self.rocket_az = 4.2;
        self.rocket_el = if kind == 0 { -0.05 } else { 0.05 };
        self.rocket_dist = 95.0;
    }

    /// Test scene: a light crew capsule descending under a deployed parachute in
    /// the lower atmosphere. Builds a throwaway light vehicle without disturbing
    /// the player's VAB design.
    fn setup_parachute(&mut self) {
        self.view = View::Rocket;
        let saved = self.vab.clone();
        self.vab.stages = vec![crate::build::StageCfg::new(0, 0)]; // Sparrow + Small
        self.vab.payload = 10; // Crew Capsule
        self.ignite_launch();
        self.vab = saved; // restore the player's design (rocket_body already built)
        // Hand-placed descent scene: integrate inline, not on the thread.
        self.detach_sim_launch();
        let radius = self.body.radius;
        let up = self.launch_up;
        if let Some(rk) = self.launch.as_mut() {
            rk.throttle = 0.0;
            for s in rk.stages.iter_mut() {
                s.prop = 0.0; // spent: only dry structure + capsule (light)
            }
            rk.r = up * (radius + 3500.0);
            rk.v = -up * 130.0; // falling
            rk.pitch = 0.0;
            rk.attitude_hold = true; // hangs nose-up under the canopy
            rk.place_attitude();
            rk.deploy_chute();
        }
        for _ in 0..45 {
            self.advance(0.1); // ~4.5 s: the canopy inflates and the fall slows
        }
        self.rocket_az = 4.2;
        self.rocket_el = 0.16;
        self.rocket_dist = 80.0;
        self.rocket_focus_y = 8.0;
    }

    /// Test scene: a compact stage doing a powered (suicide-burn) descent to a
    /// soft landing under the auto-descent autopilot.
    fn setup_powered_descent(&mut self) {
        self.view = View::Rocket;
        let saved = self.vab.clone();
        self.vab.stages = vec![crate::build::StageCfg::new(1, 0)]; // Merlin + Small
        self.vab.payload = 10; // Crew Capsule
        self.ignite_launch();
        self.vab = saved;
        // Hand-placed descent scene: integrate inline, not on the thread.
        self.detach_sim_launch();
        let radius = self.body.radius;
        let up = self.launch_up;
        if let Some(rk) = self.launch.as_mut() {
            rk.r = up * (radius + 1800.0);
            rk.v = -up * 160.0; // falling
            rk.pitch = 0.0;
            rk.attitude_hold = true; // hold nose-up for the retro-burn
            rk.place_attitude();
            rk.auto_land = true; // engage the powered-descent autopilot
        }
        for _ in 0..30 {
            self.advance(0.1); // ~3 s into the retro-burn
        }
        self.rocket_az = 4.2;
        self.rocket_el = 0.10;
        self.rocket_dist = 90.0;
        self.rocket_focus_y = 12.0;
    }

    /// Test scene: stand a rocket on the pad carrying `payload`, fairing cracked
    /// open so the payload module is visible. Used to inspect crew/service builds.
    fn setup_payload_preview(&mut self, payload: usize) {
        self.view = View::Rocket;
        self.vab.payload = payload;
        self.rebuild_vehicle();
        self.vab_mode = false;
        self.rolling_out = false;
        self.rollout = 1.0;
        self.launch = None; // resting on the pad, not flying
        self.fairing_open = 0.6;
        let top = self.rocket_body.height;
        self.rocket_az = 5.05;
        self.rocket_el = 0.06;
        self.rocket_focus_y = top - 7.0;
        self.rocket_dist = 16.0;
    }

    /// Live LOD-debug stats for the HUD: the active planet LOD (patch counts per
    /// depth) selected from the current camera, plus the camera altitude (m) and
    /// the current rebuild-grid cell size (m). Planet rocket view only.
    pub fn lod_debug_stats(&self) -> (terrain::Lod, f64, f64) {
        let cam_world = self.cam_world(self.ref_local);
        let lod = rocket::planet_lod(cam_world, 19);
        let p = self.cam_target_local();
        let alt = (self.cam_world(p).length() - self.body.radius).max(0.0);
        // mirror terrain_anchor_local's cell computation for display
        let base = 0.5 * alt.max(1.0) / rocket::PLANET_SPLIT_FACTOR;
        let raw = if alt > HIGH_ALT_STABLE_M { base * (alt / HIGH_ALT_STABLE_M) } else { base };
        let cell = raw.clamp(TERRAIN_GRID_MIN_M, TERRAIN_GRID_MAX_M).log2().floor().exp2();
        (lod, alt, cell)
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

        // Realtime point lights (max 8). At night, the nearest streetlights of
        // the city the camera is in carry the scene so moving NPCs (and the road
        // around the camera) are lit dynamically; the static streets/buildings
        // already glow from their baked emissive pools/windows. Otherwise, the
        // interior work lights pool inside the assembly building.
        let mut lights = [[0.0f32; 4]; MAX_LIGHTS];
        let mut light_col = [[0.0f32; 4]; MAX_LIGHTS];
        let scale = (1.0 - self.rollout).clamp(0.0, 1.0);
        let mut nlights = 0usize;

        let focus = self.cam_target_local();
        let mut city_night: Option<Vec3> = None;
        if self.night {
            let mut best = 700.0f64 * 700.0;
            for (c, _) in all_cities() {
                let d = (c.x as f64 - focus.x).powi(2) + (c.z as f64 - focus.z).powi(2);
                if d < best {
                    best = d;
                    city_night = Some(c);
                }
            }
        }

        if let Some(c) = city_night {
            // warm sodium streetlights: the nearest lamps to the camera focus.
            let mut lamps = rocket::city_lamps(c);
            lamps.sort_by(|a, b| {
                let da = (a.x as f64 - focus.x).powi(2) + (a.z as f64 - focus.z).powi(2);
                let db = (b.x as f64 - focus.x).powi(2) + (b.z as f64 - focus.z).powi(2);
                da.partial_cmp(&db).unwrap()
            });
            for lp in lamps.iter().take(MAX_LIGHTS) {
                let p = self.rel(lp.as_dvec3());
                lights[nlights] = [p.x, p.y, p.z, 30.0];
                light_col[nlights] = [0.95, 0.74, 0.44, 0.0];
                nlights += 1;
            }
        } else if scale > 0.01 {
            // a subtle flicker so the lighting reads as live
            let flick = 0.92 + 0.08 * (self.anim * 9.0).sin();
            for (off, col, range) in HANGAR_LIGHTS {
                let p = self.rel((HANGAR_POS + off).as_dvec3());
                lights[nlights] = [p.x, p.y, p.z, range];
                let s = scale * flick * 1.15; // gentle, so the falloff reads
                light_col[nlights] = [col[0] * s, col[1] * s, col[2] * s, 0.0];
                nlights += 1;
            }
        }

        // On the airless moon there is no sky ambient or aerial haze: flag the
        // shader (sun.w = 1) and kill the fog so the surface reads dark with
        // hard, high-contrast crater shadows.
        let lunar = if self.lunar { 1.0 } else { 0.0 };
        // Aerial-perspective haze scales with the air the camera sits in (an
        // exp falloff at the Rayleigh scale height), so distant ground hazes
        // near the surface but the planet stays crisp when viewed from orbit -
        // otherwise the raw view distance fogs the whole disk to white.
        let cam_alt =
            (self.cam_world(self.camera_eye_local()).length() - self.body.radius).max(0.0) as f32;
        let fog_scale = (-cam_alt / 8_000.0).exp();
        let fog = if self.lunar {
            [0.0, 0.0, 0.0, 0.0]
        } else {
            [HORIZON[0], HORIZON[1], HORIZON[2], fog_scale / 160_000.0]
        };
        // A low, grazing sun on the moon throws long shadows off the crater rims;
        // a higher sun for deep-space asteroid portraits.
        let sun_l = if self.space {
            SUN_LOCAL_SPACE
        } else if self.lunar {
            SUN_LOCAL_MOON
        } else if self.night {
            SUN_LOCAL_NIGHT
        } else {
            SUN_LOCAL
        };
        // Procedural surface detail for the asteroid (fragment-level normal
        // mapping + micro self-shadow). Body centre in camera-relative space is
        // rel(origin); w carries the radius (0 = off for other scenes).
        let detail = if self.ast_elev.is_some() {
            let c = self.rel(DVec3::ZERO);
            [c.x, c.y, c.z, self.ast_radius as f32]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        };
        MeshUniforms {
            viewproj: vp.to_cols_array_2d(),
            sun: [sun_l.x, sun_l.y, sun_l.z, lunar],
            params: [fcoef, self.anim, nlights as f32, scale],
            // Light aerial haze only; the atmosphere shader does the real work so
            // the planet keeps its colour from altitude.
            fog,
            lights,
            light_col,
            detail,
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
        // Sun in world coords, matching the mesh shader's local sun direction
        // (a low, grazing sun on the moon).
        let sl = if self.space {
            SUN_LOCAL_SPACE
        } else if self.lunar {
            SUN_LOCAL_MOON
        } else if self.night {
            SUN_LOCAL_NIGHT
        } else {
            SUN_LOCAL
        };
        let sun = (self.launch_east * sl.x as f64
            + self.launch_up * sl.y as f64
            + self.launch_north * sl.z as f64)
            .normalize();
        let r_atm = self.body.radius + 90_000.0;
        SkyUniforms {
            right: to_world_dir(right),
            up: to_world_dir(up),
            fwd: to_world_dir(fwd),
            sun: [sun.x as f32, sun.y as f32, sun.z as f32, 0.0],
            cam: [
                cam_world.x as f32,
                cam_world.y as f32,
                cam_world.z as f32,
                if self.space { 2.0 } else if self.lunar { 1.0 } else { 0.0 },
            ],
            params: [tan, aspect, self.body.radius as f32, r_atm as f32],
        }
    }

    /// Current re-entry heating glow (0 when not launched / cool). Drives whether
    /// the volumetric plasma pass runs and how bright it is.
    fn plasma_heat(&self) -> f32 {
        self.launch
            .as_ref()
            .filter(|rk| !rk.destroyed)
            .map(|rk| rk.heat as f32)
            .unwrap_or(0.0)
    }

    /// PROTOTYPE (mesh approach): build a glow shell that HUGS the vehicle hull
    /// for the re-entry plasma, instead of raymarching it. The shell is the
    /// isosurface (via surface nets) of the vehicle SDF, smeared downstream along
    /// the airflow so it wraps the real geometry (boosters get their own bulge)
    /// and trails into a wake. Each vertex carries a "cool" coordinate (0 at the
    /// windward face .. 1 at the wake tail) in `color.x` for the white -> orange
    /// -> red ramp, and an outward SDF-gradient normal for the fresnel rim. Verts
    /// are camera-relative, so the mesh pass transforms + depth-tests it.
    fn plasma_mesh(&self) -> Vec<rocket::MeshVertex> {
        let rk = match self.launch.as_ref() {
            Some(rk) if !rk.destroyed => rk,
            _ => return Vec::new(),
        };
        let base = self.to_local_d(rk.r);
        let quat = self.rocket_quat(rk); // full attitude incl. roll
        let heat = (rk.heat as f32).clamp(0.0, 1.4); // glow intensity, fades the FX in
        let height = self.rocket_body.height.max(8.0);
        let center = self.rel(base + (quat * Vec3::new(0.0, height * 0.45, 0.0)).as_dvec3());
        let vrad = self.rocket_body.engine_r.first().copied().unwrap_or(2.0).max(2.5);
        let axis = self.dir_to_local(rk.point_dir());
        let vdir = self.dir_to_local(rk.v.normalize_or_zero());
        let flow = if vdir.length_squared() > 1e-6 {
            let axis_signed = if vdir.dot(axis) < 0.0 { -axis } else { axis };
            let aoa = vdir.dot(axis_signed).clamp(-1.0, 1.0).acos();
            let t = ((aoa - 0.21) / (0.96 - 0.21)).clamp(0.0, 1.0);
            let blend = t * t * (3.0 - 2.0 * t);
            axis_signed.lerp(vdir, blend).normalize_or_zero()
        } else {
            axis
        };

        // Attached SDF prims in camera-relative space (the hull to hug).
        let mut prims: Vec<(Vec3, f32, Vec3, f32)> = Vec::new();
        let xform = |m: [f32; 3]| self.rel(base + (quat * Vec3::from(m)).as_dvec3());
        for (si, pr) in &self.rocket_body.sdf_stage {
            if *si >= rk.stage_base {
                prims.push((xform(pr.a), pr.r1, xform(pr.b), pr.r2));
            }
        }
        for pr in &self.rocket_body.sdf_payload {
            prims.push((xform(pr.a), pr.r1, xform(pr.b), pr.r2));
        }
        if prims.is_empty() {
            return Vec::new();
        }

        let shell = vrad * 0.6; // glow standoff off the hull
        let wake = height * 2.2; // downstream wake length (long, fades to nothing)

        // Flow-aligned frame: +s windward, x/y perpendicular.
        let upref = if flow.dot(Vec3::Y).abs() > 0.9 { Vec3::X } else { Vec3::Y };
        let rt = flow.cross(upref).normalize_or_zero();
        let upv = rt.cross(flow).normalize_or_zero();

        // SDF of the hull (union of round cones) and a downstream-smeared field
        // whose isosurface `== shell` hugs the windward hull and trails a wake.
        let sdf = |p: Vec3| -> f32 {
            prims
                .iter()
                .fold(f32::MAX, |d, &(a, r1, b, r2)| d.min(sd_round_cone(p, a, b, r1, r2)))
        };

        // Bounds of the shell in the flow frame, lead/tail along the airflow.
        let (mut smin, mut smax) = (f32::MAX, f32::MIN);
        let (mut xmin, mut xmax) = (f32::MAX, f32::MIN);
        let (mut ymin, mut ymax) = (f32::MAX, f32::MIN);
        let mut lead = f32::MIN;
        let mut tail = f32::MAX;
        for &(a, r1, b, r2) in &prims {
            for (p, r) in [(a, r1), (b, r2)] {
                let d = p - center;
                let (s, x, y) = (d.dot(flow), d.dot(rt), d.dot(upv));
                let rr = r + shell;
                smin = smin.min(s - rr);
                smax = smax.max(s + rr);
                xmin = xmin.min(x - rr);
                xmax = xmax.max(x + rr);
                ymin = ymin.min(y - rr);
                ymax = ymax.max(y + rr);
                lead = lead.max(s + r);
                tail = tail.min(s - r);
            }
        }
        let body = (lead - tail).max(vrad);

        // Downstream-smeared field whose `== shell` isosurface hugs the windward
        // hull and trails a tapering wake. Smear = smooth-union over upstream
        // shifts of the hull SDF: the smooth min fuses the shifted copies into one
        // creaseless wake (no banding/scallops at any attitude), the step is kept
        // within the smoothing radius so they always overlap, and a per-shift
        // taper pinches the far wake to a point instead of a flat tube.
        let smk = shell * 1.6; // wake-union smoothing radius
        let ksmear = ((wake / smk).ceil() as usize + 1).clamp(8, 24);
        let taper = shell * 2.2 / wake;
        let field = |p: Vec3| -> f32 {
            let mut d = sdf(p);
            for m in 1..ksmear {
                let kd = wake * m as f32 / (ksmear - 1) as f32;
                d = smooth_min(d, sdf(p + flow * kd) + kd * taper, smk);
            }
            d - shell
        };

        smin -= wake; // wake trails downstream (toward low s)
        let pad = shell * 1.8; // room for the outer (offset) glow layers
        smin -= pad;
        xmin -= pad;
        ymin -= pad;
        smax += pad;
        xmax += pad;
        ymax += pad;

        // Grid (cell ~ standoff), capped per axis to bound vert count + CPU cost.
        let cell = (vrad * 0.55).max(0.7);
        let dim = |lo: f32, hi: f32| (((hi - lo) / cell).ceil() as usize).clamp(2, 30);
        let (gs, gx, gy) = (dim(smin, smax), dim(xmin, xmax), dim(ymin, ymax));
        let cs = (smax - smin) / gs as f32;
        let cx = (xmax - xmin) / gx as f32;
        let cy = (ymax - ymin) / gy as f32;
        let cpos = |i: usize, j: usize, k: usize| -> Vec3 {
            center
                + flow * (smin + i as f32 * cs)
                + rt * (xmin + j as f32 * cx)
                + upv * (ymin + k as f32 * cy)
        };

        // Sample the field at every grid corner.
        let (nx, ny, nz) = (gs + 1, gx + 1, gy + 1);
        let cidx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
        let mut fval = vec![0.0f32; nx * ny * nz];
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    fval[cidx(i, j, k)] = field(cpos(i, j, k));
                }
            }
        }

        // Surface nets, extracted at a few iso levels so the glow reads as nested
        // shells (volume + flow) rather than one hard skin: the inner shell is
        // bright and tight on the hull, the outer ones are fainter offset wisps
        // that the shader ripples with animated noise. The expensive field
        // sampling is shared - each extra layer is just another cheap extraction.
        const EDGES: [((usize, usize, usize), (usize, usize, usize)); 12] = [
            ((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
            ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
            ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),
        ];
        let lidx = |i: usize, j: usize, k: usize| (i * gx + j) * gy + k;
        let grad_eps = cell * 0.5;
        // Nested iso levels (iso offset into the field, layer coord 0=inner ..
        // 1=outer for the shader), spaced from the hull out to PLASMA_GLOW_SPREAD.
        let nlayers = PLASMA_GLOW_LAYERS.max(1);
        let layers: Vec<(f32, f32)> = (0..nlayers)
            .map(|i| {
                let f = if nlayers > 1 { i as f32 / (nlayers - 1) as f32 } else { 0.0 };
                (shell * PLASMA_GLOW_SPREAD * f, f)
            })
            .collect();
        let mut out: Vec<rocket::MeshVertex> = Vec::new();
        for &(iso, layer) in &layers {
            let mut cellv: Vec<i32> = vec![-1; gs * gx * gy];
            let mut pos: Vec<Vec3> = Vec::new();
            let mut nrm: Vec<Vec3> = Vec::new();
            let mut coolv: Vec<f32> = Vec::new();
            for i in 0..gs {
                for j in 0..gx {
                    for k in 0..gy {
                        let mut acc = Vec3::ZERO;
                        let mut cnt = 0.0f32;
                        for &((ax, ay, az), (bx, by, bz)) in EDGES.iter() {
                            let fa = fval[cidx(i + ax, j + ay, k + az)] - iso;
                            let fb = fval[cidx(i + bx, j + by, k + bz)] - iso;
                            if (fa < 0.0) != (fb < 0.0) {
                                let t = fa / (fa - fb);
                                acc += cpos(i + ax, j + ay, k + az)
                                    .lerp(cpos(i + bx, j + by, k + bz), t);
                                cnt += 1.0;
                            }
                        }
                        if cnt > 0.0 {
                            let p = acc / cnt;
                            // outward normal from the hull SDF gradient (for fresnel).
                            let n = Vec3::new(
                                sdf(p + Vec3::X * grad_eps) - sdf(p - Vec3::X * grad_eps),
                                sdf(p + Vec3::Y * grad_eps) - sdf(p - Vec3::Y * grad_eps),
                                sdf(p + Vec3::Z * grad_eps) - sdf(p - Vec3::Z * grad_eps),
                            )
                            .normalize_or_zero();
                            // cool: downstream distance from the windward face (0 hot),
                            // 0..0.30 over the hull, 0.30..1 through the wake.
                            let ds = (lead - (p - center).dot(flow)).max(0.0);
                            let cool = if ds <= body {
                                0.30 * (ds / body)
                            } else {
                                (0.30 + 0.70 * ((ds - body) / wake)).min(1.0)
                            };
                            cellv[lidx(i, j, k)] = pos.len() as i32;
                            pos.push(p);
                            nrm.push(n);
                            coolv.push(cool);
                        }
                    }
                }
            }

            let mut tri = |a: i32, b: i32, c: i32, out: &mut Vec<rocket::MeshVertex>| {
                if a < 0 || b < 0 || c < 0 {
                    return;
                }
                for &vi in &[a as usize, b as usize, c as usize] {
                    let p = pos[vi];
                    let n = nrm[vi];
                    out.push(rocket::MeshVertex {
                        pos: [p.x, p.y, p.z],
                        normal: [n.x, n.y, n.z],
                        color: [coolv[vi], layer, heat], // z = heat (FX fades in with it)
                    });
                }
            };
            // Quad around each crossed interior edge (double-sided pipeline, so the
            // winding does not matter - only the normals do).
            let mut quad = |c0: i32, c1: i32, c2: i32, c3: i32, out: &mut Vec<rocket::MeshVertex>| {
                tri(c0, c1, c2, out);
                tri(c0, c2, c3, out);
            };
            for i in 0..gs {
                for j in 1..gx {
                    for k in 1..gy {
                        if (fval[cidx(i, j, k)] < iso) != (fval[cidx(i + 1, j, k)] < iso) {
                            quad(
                                cellv[lidx(i, j - 1, k - 1)], cellv[lidx(i, j, k - 1)],
                                cellv[lidx(i, j, k)], cellv[lidx(i, j - 1, k)], &mut out,
                            );
                        }
                    }
                }
            }
            for i in 1..gs {
                for j in 0..gx {
                    for k in 1..gy {
                        if (fval[cidx(i, j, k)] < iso) != (fval[cidx(i, j + 1, k)] < iso) {
                            quad(
                                cellv[lidx(i - 1, j, k - 1)], cellv[lidx(i, j, k - 1)],
                                cellv[lidx(i, j, k)], cellv[lidx(i - 1, j, k)], &mut out,
                            );
                        }
                    }
                }
            }
            for i in 1..gs {
                for j in 1..gx {
                    for k in 0..gy {
                        if (fval[cidx(i, j, k)] < iso) != (fval[cidx(i, j, k + 1)] < iso) {
                            quad(
                                cellv[lidx(i - 1, j - 1, k)], cellv[lidx(i, j - 1, k)],
                                cellv[lidx(i, j, k)], cellv[lidx(i - 1, j, k)], &mut out,
                            );
                        }
                    }
                }
            }
        }
        out
    }

    /// Exhaust intensity of the active engine (0 when not thrusting / destroyed).
    fn plume_intensity(&self) -> f32 {
        self.launch
            .as_ref()
            .filter(|rk| !rk.destroyed && rk.live_thrust() > 0.0)
            .map(|rk| rk.throttle as f32)
            .unwrap_or(0.0)
    }

    /// Uniforms for the volumetric exhaust-plume pass: the active engine nozzle
    /// (plus any radial-booster nozzles), all firing along the exhaust direction.
    fn plume_uniforms(&self, res: [f32; 2]) -> PlumeUniforms {
        let aspect = res[0] / res[1].max(1.0);
        let (eye, _t, right, up, fwd, tan) = self.rocket_camera(aspect);
        let inten = self.plume_intensity();
        let mut noz = [[0.0f32; 4]; MAX_PLUME_NOZZLES];
        let mut nn = 0usize;
        let (center, bound, dir, length, base_r) = match self.launch.as_ref() {
            Some(rk) => {
                let pdir = self.dir_to_local(rk.point_dir());
                let exhaust = -pdir;
                let q = Quat::from_rotation_arc(Vec3::Y, pdir);
                let base = self.to_local_d(rk.r);
                let sb = rk.stage_base;
                let er = self.rocket_body.engine_r.get(sb).copied().unwrap_or(0.9).max(0.5);
                let ny = self.rocket_body.nozzle_y.get(sb).copied().unwrap_or(0.0);
                let booster_stage = sb == 0;
                let length = (if booster_stage { 30.0 } else { 18.0 }) * (0.6 + 0.4 * inten);
                let main = self.rel(base + (q * Vec3::new(0.0, ny - 1.2, 0.0)).as_dvec3());
                noz[nn] = [main.x, main.y, main.z, 1.0];
                nn += 1;
                let mut spread = 0.0f32;
                if let Some(&(bn, rr)) = self.rocket_body.booster_rings.get(sb) {
                    spread = rr;
                    for k in 0..bn {
                        if nn >= MAX_PLUME_NOZZLES {
                            break;
                        }
                        let a = k as f32 / bn.max(1) as f32 * std::f32::consts::TAU;
                        let off = Vec3::new(a.cos() * rr, ny - 1.0, a.sin() * rr);
                        let p = self.rel(base + (q * off).as_dvec3());
                        noz[nn] = [p.x, p.y, p.z, 0.6];
                        nn += 1;
                    }
                }
                let center = main + exhaust * (length * 0.5);
                let bound = length * 0.65 + spread + er * 2.0;
                (center, bound, exhaust, length, er)
            }
            None => (Vec3::ZERO, 10.0, Vec3::NEG_Y, 10.0, 1.0),
        };
        PlumeUniforms {
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            fwd: [fwd.x, fwd.y, fwd.z, 0.0],
            eye: [eye.x, eye.y, eye.z, 0.0],
            center: [center.x, center.y, center.z, bound],
            dir: [dir.x, dir.y, dir.z, length],
            params: [tan, aspect, self.anim, inten],
            // z = log-depth Fcoef (same as the scene mesh) so the plume's frag
            // depth occludes correctly against the rocket / terrain.
            nnoz: [nn as f32, base_r, 1.0 / (LOG_DEPTH_FAR + 1.0).log2(), 0.0],
            noz,
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
        let dt_sim = (frame_dt * self.warp.min(8.0)) as f64;
        match (self.flight.as_mut(), self.moonbot.as_mut()) {
            (Some(craft), maybe_bot) => {
                // when the bot is engaged it sets the controls a human would
                // (attitude target + throttle), flying relative to the moon.
                if let Some(b) = maybe_bot {
                    b.control(craft, &bodies[0]);
                }
                craft.integrate(&self.body, &bodies, dt_sim);
            }
            (None, _) => {
                if self.launched {
                    self.clock += frame_dt * self.warp;
                }
            }
        }

        // roll the assembled rocket out of the hangar across the flats to the
        // pad. ~64 s end to end at 1x: a slow crawler-transporter pace, but the
        // player can crank `rollout_speed` to fast-forward it.
        if self.rolling_out {
            self.rollout = (self.rollout + frame_dt / 64.0 * self.rollout_speed).min(1.0);
            if self.rollout >= 1.0 {
                self.rolling_out = false;
                self.vab_mode = false; // now on the pad, ready to launch
            }
        }

        // player-controlled launch + any tumbling spent booster
        let frozen = self.reentry_test;
        let test_heat = (self.test_friction * 1.4) as f64; // slider -> glow level
        if self.sim_thread_active() {
            // Threaded: the dedicated physics thread owns and integrates the
            // rocket on its own wall-clock. Forward the current control inputs,
            // then adopt the latest authoritative snapshot as our render-side
            // mirror - keeping the player-owned control fields so an input made
            // this frame is not clobbered by a slightly older snapshot.
            let sim = self.sim.as_ref().unwrap();
            if let Some(rk) = self.launch.as_ref() {
                sim.set_controls(self.launch_controls(rk));
            }
            if let Some(snap) = sim.try_snapshot() {
                if let Some(old) = self.launch.as_ref() {
                    let mut s = snap;
                    s.throttle = old.throttle;
                    s.pitch = old.pitch;
                    s.yaw = old.yaw;
                    s.roll = old.roll;
                    s.attitude_hold = old.attitude_hold;
                    s.auto_land = old.auto_land;
                    s.chute_armed = old.chute_armed;
                    self.launch = Some(s);
                }
                // launch == None means we just reset; ignore stale snapshots.
            }
        } else if let Some(rk) = self.launch.as_mut() {
            // Inline (tests / headless / wasm): integrate synchronously so these
            // paths stay deterministic.
            rk.just_staged = None;
            if frozen {
                // Frozen plasma test: hold the vehicle in place so the FX play
                // continuously and you can orbit / re-aim it. Skip integration
                // entirely (no motion, no damage, never breaks up); attitude tracks
                // the command (W/S/A/D/Q/E), and the friction slider drives the heat
                // glow so you can sweep the re-entry FX from cold to white-hot.
                rk.place_attitude();
                rk.health = 100.0;
                rk.destroyed = false;
                rk.crashed = false;
                rk.heat = test_heat;
            } else {
                let dt = (frame_dt * self.warp).min(2.0);
                rk.integrate(&self.body, dt as f64);
            }
        }
        // drive the car / walk the character (real time, independent of warp).
        self.advance_car(frame_dt);
        self.advance_ped(frame_dt);
        // city life: each city's crowd spawns/sims when the player is near it and
        // despawns when far. When driving, hand the NPCs the player's car so they
        // brake for it.
        let pp = self.player_pos();
        let pc = if self.driving { Some(self.car_pos) } else { None };
        for t in self.traffic.iter_mut() {
            t.update(pp, pc, frame_dt);
        }

        // break-up: a destroyed vehicle (burn-through or crash) explodes once.
        if self.launch.as_ref().map(|rk| rk.destroyed).unwrap_or(false) && !self.exploded {
            self.spawn_explosion();
        }
        self.capture_orbit_if_reached();
        let fx_dt = (frame_dt * self.warp).min(0.5);
        self.anim += fx_dt;
        self.advance_sep(frame_dt * self.warp.min(8.0));
        self.advance_debris(frame_dt * self.warp.min(8.0));
        self.advance_smoke(fx_dt);
        self.advance_sparks(fx_dt);

        // Keep the planet terrain in sync with the floating-origin reference.
        if self.view == View::Rocket {
            if !self.space && self.ast_elev.is_none() {
                self.update_planet_terrain();
            } else {
                // Asteroid / deep-space: cheap and genuinely camera-driven (you
                // orbit the body to refine it), so keep the original synchronous,
                // camera-anchored rebuild.
                let eye = self.camera_eye_local();
                let alt = self.cam_world(eye).length() - self.body.radius;
                let thresh = (alt.abs() * 0.04).clamp(25.0, 50_000.0);
                if self.terrain_verts.is_empty() || (eye - self.ref_local).length() > thresh {
                    self.ref_local = eye;
                    self.rebuild_terrain();
                    self.terrain_dirty = true;
                }
            }

            // Smoothly track the centre of the still-attached geometry so the
            // camera re-centres on the remaining stack after a stage drops.
            let target_center = match self.launch.as_ref() {
                Some(rk) => self.remaining_center_y(rk.stage_base),
                None => self.rocket_body.focus_y,
            };
            if self.cam_focus_y <= 0.0 {
                self.cam_focus_y = target_center; // first frame: snap
            } else {
                let k = 1.0 - (-frame_dt / 0.6).exp();
                self.cam_focus_y += (target_center - self.cam_focus_y) * k;
            }
        }
    }

    /// Grid-snapped terrain anchor in local metres. Anchored to the framed
    /// rocket (NOT the orbiting camera), so looking around never moves it; and
    /// quantised to a power-of-two lattice (cell ~ 5% of altitude) so the
    /// floating origin - and therefore the geometry built around it - is
    /// deterministic: identical rocket position => identical mesh, and it only
    /// changes at fixed lattice steps as the rocket actually travels.
    fn terrain_anchor_local(&self) -> DVec3 {
        let p = self.cam_target_local();
        let alt = (self.cam_world(p).length() - self.body.radius).max(1.0);
        // The quadtree stops splitting once a patch is about `altitude /
        // split_factor` across, so that is the finest patch edge near the
        // rocket. Make the rebuild threshold (the grid cell) ~half of it: lower
        // LODs (higher up) therefore get proportionally larger thresholds, and
        // we never rebuild for sub-patch motion.
        let base = 0.5 * alt / rocket::PLANET_SPLIT_FACTOR;
        // Above ~50 km the ground reads as a smooth globe (no fine relief is
        // resolvable), so we transition to a stable coarse cube-sphere LOD
        // planet: widen the rebuild cell super-linearly with altitude so the
        // upper terrain re-meshes only at large steps instead of churning every
        // kilometre of downrange travel. This is the "annoying rebuild" fix.
        let raw = if alt > HIGH_ALT_STABLE_M {
            base * (alt / HIGH_ALT_STABLE_M)
        } else {
            base
        };
        let raw = raw.clamp(TERRAIN_GRID_MIN_M, TERRAIN_GRID_MAX_M);
        // Snap the cell size itself to a power of two so it only steps at
        // altitude octaves instead of drifting continuously with altitude.
        let grid = (raw.log2().floor()).exp2();
        DVec3::new(
            (p.x / grid).round() * grid,
            (p.y / grid).round() * grid,
            (p.z / grid).round() * grid,
        )
    }

    /// Planet terrain update: rebuild only when the rocket crosses into a new
    /// grid cell. The heavy mesh build runs on the worker thread and is double-
    /// buffered (the current mesh keeps drawing until the new one is ready), so
    /// crossing a cell never spikes a frame. The very first build is synchronous
    /// so there is never a blank frame (and headless shots are unaffected).
    fn update_planet_terrain(&mut self) {
        let anchor = self.terrain_anchor_local();
        if self.terrain_verts.is_empty() {
            self.ref_local = anchor;
            self.rebuild_terrain();
            self.terrain_dirty = true;
        } else if anchor != self.ref_local && !self.terrain_svc.busy() {
            self.terrain_svc.request(terrain_job::PlanetJob {
                cam_world: self.cam_world(anchor),
                ref_local: anchor,
                origin: self.launch_origin,
                up: self.launch_up,
                east: self.launch_east,
                north: self.launch_north,
                depth: 19,
                lunar: self.lunar,
                lod_debug: self.lod_debug,
            });
        }

        // Adopt any mesh the worker finished, swapping its reference origin in
        // atomically with its vertices.
        if let Some(res) = self.terrain_svc.try_recv() {
            self.ref_local = res.ref_local;
            self.terrain_count = (res.verts.len() as u64).min(TERRAIN_CAP) as u32;
            self.terrain_verts = res.verts;
            self.terrain_dirty = true; // upload pending
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

    /// The resting rocket base in local metres: in the hangar during assembly,
    /// sliding to the pad (origin) as roll-out animates.
    fn resting_base_local(&self) -> DVec3 {
        let hangar = HANGAR_POS.as_dvec3();
        let pos = hangar.lerp(DVec3::ZERO, self.rollout as f64);
        pos + DVec3::new(0.0, self.rocket_body.base_y as f64, 0.0)
    }

    /// The rocket the camera frames (its local position, f64).
    fn cam_target_local(&self) -> DVec3 {
        match self.launch.as_ref() {
            Some(rk) => self.to_local_d(rk.r),
            None => self.resting_base_local(),
        }
    }

    /// Model-Y centroid of the geometry still attached at `stage_base` (the
    /// active stage and everything above it, including the payload). Used to
    /// re-centre the camera on the remaining stack after a stage separates.
    fn remaining_center_y(&self, stage_base: usize) -> f32 {
        let rb = &self.rocket_body;
        let mut sum = 0.0f64;
        let mut n = 0u32;
        for r in rb.stage_ranges.iter().skip(stage_base) {
            for v in &rb.mesh.verts[r.clone()] {
                sum += v.pos[1] as f64;
                n += 1;
            }
        }
        for v in &rb.mesh.verts[rb.payload_range.clone()] {
            sum += v.pos[1] as f64;
            n += 1;
        }
        if n == 0 {
            rb.focus_y
        } else {
            (sum / n as f64) as f32
        }
    }

    /// The point the rocket-view camera looks at (launch-tangent metres, f64).
    fn camera_look_local(&self) -> DVec3 {
        // Driving / on foot: frame the car or the character (the camera orbits
        // and zooms around it as usual).
        if self.driving {
            return self.car_pos + DVec3::new(0.0, 1.4, 0.0);
        }
        if self.walking {
            return self.ped_pos + DVec3::new(0.0, 1.5, 0.0);
        }
        let target = self.cam_target_local();
        match self.launch.as_ref() {
            Some(rk) => {
                // Aim at the smoothed centre of the remaining geometry (so the
                // framing re-centres after staging), eased down toward the base
                // low and slow so the pad + smoke stay framed at liftoff.
                let axis = self.dir_to_local_d(rk.point_dir());
                let ease = (self.to_local_d(rk.r).y / 120.0).clamp(0.25, 1.0);
                target + axis * (self.cam_focus_y as f64 * ease)
            }
            None => target + DVec3::new(0.0, self.rocket_focus_y as f64, 0.0),
        }
    }

    /// Camera ray (camera-relative origin + direction) through a cursor NDC.
    fn cursor_ray(&self, ndc: [f32; 2], aspect: f32) -> (Vec3, Vec3) {
        let (eye, _t, right, up, fwd, tan) = self.rocket_camera(aspect);
        let dir = (fwd + right * (ndc[0] * tan * aspect) + up * (ndc[1] * tan)).normalize();
        (eye, dir)
    }

    /// The rocket's stack attach slots (local pos, kind, stage) for drag-drop.
    fn stack_slots(&self) -> Vec<(DVec3, rocket::PartKind, usize)> {
        let base = self.resting_base_local();
        let rb = &self.rocket_body;
        let mut v = Vec::new();
        for i in 0..self.vab.stages.len() {
            let ny = rb.nozzle_y.get(i).copied().unwrap_or(0.0) as f64;
            v.push((base + DVec3::new(0.0, ny, 0.0), rocket::PartKind::Engine, i));
            v.push((base + DVec3::new(0.0, ny + 5.0, 0.0), rocket::PartKind::Tank, i));
        }
        v.push((base + DVec3::new(0.0, rb.height as f64 - 4.0, 0.0), rocket::PartKind::Payload, 0));
        v
    }

    /// Pick a rack part under the cursor (start of a drag).
    fn pick_rack(&self, ndc: [f32; 2], aspect: f32) -> Option<RackSlot> {
        let (o, d) = self.cursor_ray(ndc, aspect);
        let mut best = f32::MAX;
        let mut hit = None;
        for s in &self.rack_slots {
            let c = self.rel(s.pos.as_dvec3());
            if let Some(t) = ray_sphere_near(o, d, c, 1.8) {
                if t < best {
                    best = t;
                    hit = Some(*s);
                }
            }
        }
        hit
    }

    /// The stack slot (matching `kind`) the cursor is over, with its local pos.
    fn pick_stack_slot(&self, ndc: [f32; 2], aspect: f32, kind: rocket::PartKind) -> Option<(rocket::PartKind, usize, Vec3)> {
        let (o, d) = self.cursor_ray(ndc, aspect);
        let mut best = f32::MAX;
        let mut hit = None;
        for (pos, k, stage) in self.stack_slots() {
            if k != kind {
                continue;
            }
            let c = self.rel(pos);
            if let Some(t) = ray_sphere_near(o, d, c, 3.2) {
                if t < best {
                    best = t;
                    hit = Some((k, stage, c));
                }
            }
        }
        hit
    }

    /// While dragging, update the ghost position + the hovered target slot.
    fn update_grab(&mut self, ndc: [f32; 2], aspect: f32) {
        let Some(g) = self.grab else { return };
        if let Some((k, stage, c)) = self.pick_stack_slot(ndc, aspect, g.kind) {
            self.grab_target = Some((k, stage));
            // snap the ghost to the slot, nudged toward the camera so it previews
            // in front of the existing part instead of hiding inside it.
            let (eye, _) = self.cursor_ray(ndc, aspect);
            self.grab_ghost = c + (eye - c).normalize_or_zero() * 4.0;
        } else {
            self.grab_target = None;
            let (o, d) = self.cursor_ray(ndc, aspect);
            self.grab_ghost = o + d * 14.0; // float in front of the camera
        }
    }

    /// Drop the grabbed part: fit it to the hovered slot if compatible.
    fn drop_grab(&mut self) {
        if let (Some(g), Some((k, stage))) = (self.grab, self.grab_target) {
            if g.kind == k {
                match k {
                    rocket::PartKind::Engine => {
                        if let Some(s) = self.vab.stages.get_mut(stage) {
                            s.engine = g.idx;
                        }
                    }
                    rocket::PartKind::Tank => {
                        if let Some(s) = self.vab.stages.get_mut(stage) {
                            s.tank = g.idx;
                        }
                    }
                    rocket::PartKind::Payload => self.vab.payload = g.idx,
                }
                self.rebuild_vehicle();
            }
        }
        self.grab = None;
        self.grab_target = None;
    }

    /// Begin rolling the assembled rocket out of the hangar to the pad.
    fn start_rollout(&mut self) {
        if self.vab_mode {
            self.rebuild_vehicle();
            self.rolling_out = true;
        }
    }

    /// Skip the crawler animation and jump straight to the pad, ready to launch.
    fn skip_rollout(&mut self) {
        self.rollout = 1.0;
        self.rolling_out = false;
        self.vab_mode = false;
    }

    /// Speed up (or slow down) the crawler while it rolls out to the pad.
    /// Doubles/halves the rate, clamped to 1x..64x (64x turns the ~64 s creep
    /// into about a second). The chosen pace persists to the next roll-out.
    fn bump_rollout_speed(&mut self, faster: bool) {
        self.rollout_speed = if faster {
            (self.rollout_speed * 2.0).min(64.0)
        } else {
            (self.rollout_speed * 0.5).max(1.0)
        };
    }

    /// Send the rocket back into the assembly building (between missions).
    fn back_to_vab(&mut self) {
        self.reset_launch();
        self.vab_mode = true;
        self.rolling_out = false;
        self.rollout = 0.0;
    }

    /// Hop in the car and start driving (rocket view). Parks it just east of the
    /// pad on the access road, pointing toward the city, with the camera trailing.
    fn enter_drive(&mut self) {
        self.view = View::Rocket;
        self.reset_launch();
        self.vab_mode = false;
        self.show_lander = false;
        self.car_pos = DVec3::new(40.0, 0.0, 0.0);
        let to_city = CITY_CENTER.as_dvec3() - self.car_pos;
        self.car_yaw = (to_city.z as f32).atan2(to_city.x as f32); // face the city
        self.start_driving();
    }

    /// Begin driving the car wherever it currently sits (used both by the direct
    /// "Drive car" entry and by getting in on foot), with the camera trailing.
    fn start_driving(&mut self) {
        self.car_speed = 0.0;
        self.drive_in = [false; 4];
        self.driving = true;
        self.walking = false;
        self.rocket_az = self.car_yaw + std::f32::consts::PI;
        self.rocket_el = 0.22;
        self.rocket_dist = 18.0;
    }

    /// Park the car and leave drive mode.
    fn exit_drive(&mut self) {
        self.driving = false;
        self.car_speed = 0.0;
        self.drive_in = [false; 4];
    }

    /// Get out on foot beside the car: switch from driving to walking, dropping
    /// the character next to the driver's door.
    fn get_out_car(&mut self) {
        self.driving = false;
        self.car_speed = 0.0;
        self.drive_in = [false; 4];
        // step out to the left of the car's heading.
        let side = DVec3::new(-(self.car_yaw.sin() as f64), 0.0, self.car_yaw.cos() as f64);
        self.ped_pos = self.car_pos + side * 3.0;
        self.ped_pos.y = 0.0;
        self.ped_yaw = self.car_yaw;
        self.start_walking();
    }

    /// Set out on foot from the launch complex: spawn the character by the parked
    /// car so it's easy to find and get into.
    fn enter_walk(&mut self) {
        self.view = View::Rocket;
        self.reset_launch();
        self.vab_mode = false;
        self.show_lander = false;
        // park the car near the pad and stand the character beside it.
        self.car_pos = DVec3::new(34.0, 0.0, 0.0);
        let to_city = CITY_CENTER.as_dvec3() - self.car_pos;
        self.car_yaw = (to_city.z as f32).atan2(to_city.x as f32);
        self.ped_pos = DVec3::new(30.0, 0.0, 4.0);
        self.ped_yaw = self.car_yaw;
        self.start_walking();
    }

    fn start_walking(&mut self) {
        self.walking = true;
        self.driving = false;
        self.ped_speed = 0.0;
        self.walk_in = [false; 4];
        self.rocket_az = self.ped_yaw + std::f32::consts::PI;
        self.rocket_el = 0.20;
        self.rocket_dist = 7.0;
    }

    /// Stop walking and return to the standard launch-complex view.
    fn exit_walk(&mut self) {
        self.walking = false;
        self.ped_speed = 0.0;
        self.walk_in = [false; 4];
    }

    /// The player's current position in local metres (for proximity systems like
    /// city traffic). Far from the city unless the player is on foot or driving.
    fn player_pos(&self) -> DVec3 {
        if self.driving {
            self.car_pos
        } else if self.walking {
            self.ped_pos
        } else {
            DVec3::ZERO // at the pad, ~2.5 km from the city: traffic stays asleep
        }
    }

    /// Whether the character is standing close enough to the car to get in.
    fn near_car(&self) -> bool {
        let d = self.ped_pos - self.car_pos;
        (d.x * d.x + d.z * d.z) < 5.0 * 5.0
    }

    /// Get into the car from on foot (only when standing next to it): keeps the
    /// car where it is and starts driving.
    fn get_in_car(&mut self) {
        if self.near_car() {
            self.start_driving();
        }
    }

    /// Walking dynamics: held WASD move the character (forward/back along its
    /// facing, A/D turn), with a quick ramp and a walk-cycle phase that advances
    /// with distance travelled. Real time (independent of warp).
    fn advance_ped(&mut self, dt: f32) {
        if !self.walking || dt <= 0.0 {
            return;
        }
        let walk_speed = 4.2; // m/s
        let target = ((self.walk_in[0] as i32 - self.walk_in[1] as i32) as f32) * walk_speed;
        // brisk accel/decel toward the target speed.
        let k = (1.0 - (-dt / 0.12).exp()).clamp(0.0, 1.0);
        self.ped_speed += (target - self.ped_speed) * k;
        let turn = (self.walk_in[2] as i32 - self.walk_in[3] as i32) as f32;
        self.ped_yaw += turn * 2.4 * dt;
        let heading = DVec3::new(self.ped_yaw.cos() as f64, 0.0, self.ped_yaw.sin() as f64);
        self.ped_pos += heading * (self.ped_speed * dt) as f64;
        self.ped_pos.y = 0.0;
        // advance the stride with distance so the gait matches the pace.
        self.walk_phase += self.ped_speed.abs() * 1.7 * dt;
        if self.walk_phase > std::f32::consts::TAU {
            self.walk_phase -= std::f32::consts::TAU;
        }
    }

    /// Arcade car dynamics: held WASD give throttle/reverse + steering, with
    /// rolling friction and a top speed. Runs in real time (independent of warp).
    fn advance_car(&mut self, dt: f32) {
        if !self.driving || dt <= 0.0 {
            return;
        }
        let accel = 18.0; // m/s^2 under power
        let max_fwd = 75.0;
        let max_rev = 24.0;
        let mut a = 0.0;
        if self.drive_in[0] {
            a += accel;
        }
        if self.drive_in[1] {
            a -= accel;
        }
        // steering: turn rate scales with speed and flips when reversing.
        let steer = (self.drive_in[2] as i32 - self.drive_in[3] as i32) as f32;
        let speed_frac = (self.car_speed / 22.0).clamp(-1.0, 1.0);
        self.car_yaw += steer * 1.7 * speed_frac * dt;
        // engine, then rolling friction always opposing motion.
        self.car_speed += a * dt;
        let f = 13.0 * dt;
        self.car_speed -= self.car_speed.clamp(-f, f) * if a == 0.0 { 1.0 } else { 0.25 };
        self.car_speed = self.car_speed.clamp(-max_rev, max_fwd);
        let heading = DVec3::new(self.car_yaw.cos() as f64, 0.0, self.car_yaw.sin() as f64);
        self.car_pos += heading * (self.car_speed * dt) as f64;
        self.car_pos.y = 0.0;
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
        // Height above the launch site (the pad sits on terrain well above the
        // body's reference radius, so absolute altitude is the wrong gauge here).
        let agl = (rk.r.length() - self.launch_origin.length()) as f32;
        let on_pad = agl < 30.0;
        let base = self.to_local(rk.r);
        let down = -self.dir_to_local(rk.point_dir()); // exhaust direction
        // emit from the ACTIVE stage's nozzle (up the mesh by nozzle_y)
        let q = Quat::from_rotation_arc(Vec3::Y, self.dir_to_local(rk.point_dir()));
        let ny = self.rocket_body.nozzle_y.get(rk.stage_base).copied().unwrap_or(0.0);
        let nozzle = base + q * Vec3::new(0.0, ny - 1.2, 0.0);
        let er = self.rocket_body.engine_r.get(rk.stage_base).copied().unwrap_or(0.9);

        // Ignition billow: the deflector throws an enormous ground cloud in the
        // first seconds. `billow` is 1 at ignition and fades to 0 by ~6 s, scaling
        // up the spawn rate, particle size, spread and lifetime so liftoff kicks
        // off a big mushrooming cloud that then settles into the normal trail.
        let billow = if on_pad {
            (1.0 - rk.met as f32 / 6.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // spawn rate: heavy at the pad (the ground billow), thinning with density.
        let rate = thrust_frac * (8.0 + 120.0 * dens) * if on_pad { 2.5 + 9.0 * billow } else { 1.0 };
        self.smoke_accum += rate * dt;
        let n = self.smoke_accum.floor() as i32;
        self.smoke_accum -= n as f32;
        let mut seed = (self.anim * 977.0).to_bits();
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let ground = Vec3::new(base.x, rocket::PAD_TOP, base.z);
        for _ in 0..n.min(110) {
            let jit = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
            if on_pad {
                // exhaust deflects off the pad: billow outward + up from the ground.
                // During the ignition billow the cloud spreads much wider, bigger
                // and lingers longer (the iconic mushroom at liftoff).
                let a = rnd() * std::f32::consts::TAU;
                let out = Vec3::new(a.cos(), 0.10 + 0.10 * rnd(), a.sin());
                let spread = 3.0 + 8.0 * billow;
                self.smoke.push(Smoke {
                    pos: ground + Vec3::new(jit.x, jit.y.abs() * 0.5, jit.z) * spread,
                    // billow gas rolls out slowly (dense cloud) rather than blasting
                    // away; the big size + lifetime make it read as a launch plume.
                    vel: out * (8.0 + rnd() * 8.0 + 8.0 * billow) + Vec3::Y * (2.0 + rnd() * 3.0 + 4.0 * billow),
                    age: 0.0,
                    life: 2.4 + rnd() * 1.8 + 3.0 * billow,
                    size0: (2.5 + rnd() * 2.0) * (1.0 + 1.8 * billow),
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
        // keep the particle count bounded (headroom for the big launch billow)
        if self.smoke.len() > 1800 {
            let drop = self.smoke.len() - 1800;
            self.smoke.drain(0..drop);
        }
    }

    /// Emit + advect re-entry sparks: tiny bright embers shed off the windward
    /// shock that streak downstream through the wake while the plasma is hot.
    fn advance_sparks(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        // Carry the embers along with the vehicle: in real flight the rocket is
        // travelling at km/s, so a spark must inherit that bulk velocity or it gets
        // left light-years behind in a single frame. `s.vel` holds only the small
        // velocity RELATIVE to the airframe (the downwind drift that forms the
        // wake); the rocket's own velocity is added here. The frozen test pins the
        // vehicle in place, so there is nothing to inherit there.
        let carry = match self.launch.as_ref() {
            Some(rk) if !self.reentry_test => self.dir_to_local_vec(rk.v),
            _ => Vec3::ZERO,
        };
        for s in self.sparks.iter_mut() {
            s.age += dt;
            s.vel *= 1.0 - (0.6 * dt).min(0.5); // drag on the relative drift only
            s.pos += (carry + s.vel) * dt;
        }
        self.sparks.retain(|s| s.age < s.life);

        let heat = self.plasma_heat();
        let Some(rk) = self.launch.as_ref() else {
            self.sparks.clear();
            return;
        };
        if rk.destroyed || heat < 0.18 {
            return; // embers only once it's genuinely hot (after the glow fades in)
        }

        // Airflow + windward point in the local frame (same derivation as the
        // plasma mesh): sparks are born on the windward face and stream downwind.
        let base = self.to_local(rk.r);
        let axis = self.dir_to_local(rk.point_dir());
        let vdir = self.dir_to_local(rk.v.normalize_or_zero());
        let flow = if vdir.length_squared() > 1e-6 {
            let axis_signed = if vdir.dot(axis) < 0.0 { -axis } else { axis };
            let aoa = vdir.dot(axis_signed).clamp(-1.0, 1.0).acos();
            let t = ((aoa - 0.21) / (0.96 - 0.21)).clamp(0.0, 1.0);
            let blend = t * t * (3.0 - 2.0 * t);
            axis_signed.lerp(vdir, blend).normalize_or_zero()
        } else {
            axis
        };
        // Pose the hull prims into the local frame (q = the SAME body orientation
        // the rocket is drawn with, incl. roll), so the spawn rotates with the
        // airframe's pitch/yaw/roll - the embers come off the rocket, aligned to it.
        let q = self.rocket_quat(rk);
        let downwind = -flow; // sparks stream downstream into the wake
        let mut prims: Vec<(Vec3, f32, Vec3, f32)> = Vec::new();
        for (si, pr) in &self.rocket_body.sdf_stage {
            if *si >= rk.stage_base {
                prims.push((base + q * Vec3::from(pr.a), pr.r1, base + q * Vec3::from(pr.b), pr.r2));
            }
        }
        for pr in &self.rocket_body.sdf_payload {
            prims.push((base + q * Vec3::from(pr.a), pr.r1, base + q * Vec3::from(pr.b), pr.r2));
        }
        if prims.is_empty() {
            return;
        }

        let rate = 130.0 * heat;
        self.spark_accum += rate * dt;
        let n = self.spark_accum.floor() as i32;
        self.spark_accum -= n as f32;
        let mut seed = (self.anim * 1597.0).to_bits() ^ 0x9e37_79b9;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..n.min(24) {
            // Shed an ember off a hull prim's WINDWARD surface (the side facing the
            // oncoming wind `vdir`): on the flank when the wind hits the side, or
            // off the leading end when it hits end-on (axial). Spawning on the real
            // hull means the embers track the rocket's orientation.
            let (a, r1, b, r2) = prims[(rnd() * prims.len() as f32) as usize % prims.len()];
            let axv = b - a;
            let axn = axv / axv.length().max(1e-3);
            let vperp = vdir - axn * vdir.dot(axn);
            let flank = vperp.length() > 0.2;
            let tt = if flank {
                rnd()
            } else if vdir.dot(axn) > 0.0 {
                0.7 + 0.3 * rnd() // windward end = b
            } else {
                0.3 * rnd() // windward end = a
            };
            let rr = r1 + (r2 - r1) * tt;
            let wdir = if flank {
                Quat::from_axis_angle(axn, (rnd() - 0.5) * 2.4) * vperp.normalize_or_zero()
            } else {
                let e1raw = if axn.dot(Vec3::Y).abs() > 0.9 { Vec3::X } else { Vec3::Y };
                let e1 = axn.cross(e1raw).normalize_or_zero();
                let e2 = axn.cross(e1).normalize_or_zero();
                let ang = rnd() * std::f32::consts::TAU;
                e1 * ang.cos() + e2 * ang.sin()
            };
            let surf = a + axv * tt + wdir * rr;
            let tang = axn.cross(wdir).normalize_or_zero();
            self.sparks.push(Spark {
                pos: surf,
                vel: downwind * (50.0 + rnd() * 90.0)
                    + wdir * (4.0 + rnd() * 8.0)
                    + tang * ((rnd() - 0.5) * 18.0),
                age: 0.0,
                life: 0.45 + rnd() * 0.95,
                seed: rnd(),
            });
        }
        if self.sparks.len() > 700 {
            let drop = self.sparks.len() - 700;
            self.sparks.drain(0..drop);
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

    /// Body->local orientation of the rocket (f32), including the commanded ROLL.
    /// The nose direction (pitch + yaw) comes from `point_dir()` exactly as before;
    /// the roll about the nose axis is applied on top, so a roll command shows up
    /// without disturbing the proven nose orientation. (We can't convert `orient`
    /// through the launch-tangent basis - that frame is left-handed, so the matrix
    /// route produces a reflected, invalid quaternion: the bug that flung the
    /// rocket under the pad.)
    fn rocket_quat(&self, rk: &launch::Rocket) -> Quat {
        let nose = self.dir_to_local(rk.point_dir());
        Quat::from_rotation_arc(Vec3::Y, nose) * Quat::from_rotation_y(rk.roll as f32)
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

    /// Rebuild the rocket-view geometry from the current VAB design (call after
    /// the player edits the build).
    fn rebuild_vehicle(&mut self) {
        let boosters: Vec<rocket::BoosterViz> = (0..self.vab.stages.len())
            .map(|i| {
                let (b, n) = self.vab.booster(i);
                rocket::BoosterViz { count: n, prop: b.prop as f32, solid: b.solid }
            })
            .collect();
        self.rocket_body = rocket::rocket_body(
            &self.vab.to_vehicle(),
            self.vab.payload().color,
            self.vab.payload().module,
            &boosters,
        );
        self.rocket_focus_y = self.rocket_body.focus_y;
    }

    /// Ignite the assembled vehicle on the pad and begin a player launch.
    fn ignite_launch(&mut self) {
        self.rebuild_vehicle();
        let veh = self.vab.to_vehicle();
        // base of the rocket sits `base_y` above the pad surface, on the mount.
        let r = self.launch_origin + self.launch_up * self.rocket_body.base_y as f64;
        // Co-moving launch frame: ignite at rest relative to the (fixed) pad and
        // terrain. We drop the planet's surface-rotation boost so the rocket
        // doesn't drift sideways out of the local scene; the player flies the
        // gravity turn from a standstill.
        let v = DVec3::ZERO;
        let mut rk = launch::Rocket::on_pad(&veh, r, v, self.launch_up, self.launch_east);
        rk.throttle = 1.0;
        // Hand the authoritative rocket to the dedicated physics thread (live
        // game); the inline path keeps integrating `self.launch` directly.
        if let Some(sim) = self.sim.as_ref() {
            sim.start(&rk, self.body);
            self.sim_drives_launch = true;
        }
        self.launch = Some(rk);
        self.sep = None;
        self.debris.clear();
        self.boom.clear();
        self.exploded = false;
        self.smoke.clear();
        self.mission_captured = false;
        self.reentry_test = false; // a real ignition flies; only the test freezes
        // you launch from the pad: ensure we're rolled out.
        self.vab_mode = false;
        self.rolling_out = false;
        self.rollout = 1.0;
        self.warp = 1.0; // launch in real time so the ascent reads at its true pace
        log::info!("Ignition: {} - throttle up, pitch over, stage when dry", veh.name);
    }

    /// Jettison the active stage; spawn the spent booster tumbling away.
    fn stage_launch(&mut self) {
        // capture the spent stage's pose + velocity (immutable borrow) first
        let (r, pd, v, stage, can_stage) = match self.launch.as_ref() {
            Some(rk) => (rk.r, rk.point_dir(), rk.v, rk.stage_base, rk.stages.len() > 1),
            None => return,
        };
        if !can_stage {
            return;
        }
        let range = self
            .rocket_body
            .stage_ranges
            .get(stage)
            .cloned()
            .unwrap_or(0..0);
        let base_local = self.to_local(r);
        let pd_local = self.dir_to_local(pd);
        let rot = Quat::from_rotation_arc(Vec3::Y, pd_local);
        let vel_local = self.dir_to_local_vec(v);
        // True gravity at the separation point, in the local frame: magnitude
        // mu/r^2 (so it is weak at altitude) pointing toward the planet centre.
        let rmag = r.length().max(1.0);
        let g_mag = (self.body.mu / (rmag * rmag)) as f32;
        let grav = -self.dir_to_local_vec(r.normalize_or_zero()) * g_mag;
        // A small, lateral nudge so the spent stage clears the climbing upper
        // stage's nozzle instead of overlapping it.
        let side = pd_local.cross(Vec3::Y).normalize_or_zero();
        // Jettison on the authoritative side: the sim thread when it is flying
        // the launch (the snapshot brings the staged state back), else inline.
        if self.sim_thread_active() {
            self.sim.as_ref().unwrap().stage();
        } else {
            self.launch.as_mut().unwrap().jettison();
        }
        self.sep = Some(SepBooster {
            pos: base_local,
            // Springs/retro-rockets give just a few m/s of separation: a soft
            // retro push plus a touch of sideways drift so it floats clear and
            // the upper stage thrusts away on its own.
            vel: vel_local - pd_local * 3.0 + side * 1.2,
            rot,
            spin: Vec3::new(0.18, 0.05, 0.28), // slow, majestic tumble
            grav,
            age: 0.0,
            range,
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
        self.detach_sim_launch();
        self.launch = None;
        self.sep = None;
        self.debris.clear();
        self.boom.clear();
        self.exploded = false;
        self.smoke.clear();
        self.sparks.clear();
        self.mission_captured = false;
        self.reentry_test = false;
    }

    /// Whether the active launch has reached a stable parking orbit.
    fn mission_complete(&self) -> bool {
        self.mission_captured
    }

    /// On reaching a bound parking orbit, deposit the payload as a persistent
    /// satellite circling the home world. Missions accumulate these.
    fn capture_orbit_if_reached(&mut self) {
        if self.mission_captured {
            return;
        }
        let reached = self.launch.as_ref().map(|rk| rk.orbit_reached).unwrap_or(false);
        if !reached {
            return;
        }
        let rk = self.launch.as_ref().unwrap();
        let orb = sim::orbit::orbit_from_state(rk.r, rk.v, self.body.mu);
        let radius_m = rk.r.length();
        let h = orb.h_vec.normalize_or_zero();
        let aref = if h.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let t1d = h.cross(aref).normalize_or_zero();
        let t2d = h.cross(t1d).normalize_or_zero();
        let phase0 = (rk.r.dot(t2d)).atan2(rk.r.dot(t1d)) as f32;
        let rate = (sim::orbit::circular_speed(self.body.mu, radius_m) / radius_m) as f32;
        let pay = self.vab.payload();
        self.orbits.push(OrbitObject {
            name: pay.name,
            color: pay.color,
            radius_mm: (radius_m / MM as f64) as f32,
            t1: t1d.as_vec3(),
            t2: t2d.as_vec3(),
            rate,
            phase0,
            epoch: self.sys_time,
        });
        self.mission_captured = true;
        log::info!("Payload to orbit: {} ({} satellites)", pay.name, self.orbits.len());
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
        self.xform_into_off(out, range, q, base, Vec3::ZERO);
    }

    /// `xform_into` with an extra translation applied in the rocket's own local
    /// frame before the pose (used to swing the fairing clamshell halves apart).
    fn xform_into_off(
        &self,
        out: &mut Vec<rocket::MeshVertex>,
        range: std::ops::Range<usize>,
        q: Quat,
        base: DVec3,
        local_off: Vec3,
    ) {
        for v in &self.rocket_body.mesh.verts[range] {
            let local = base + (q * (Vec3::from(v.pos) + local_off)).as_dvec3();
            let n = q * Vec3::from(v.normal);
            out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: n.into(), color: v.color });
        }
    }

    /// Rebuild the full-planet LOD terrain, camera-relative to the current
    /// `ref_local`, refined toward the camera. Called when the reference moves.
    fn rebuild_terrain(&mut self) {
        // Asteroid: render the body through the LOD quadtree, centred at the
        // local origin, refining as the camera (camera_eye_local) approaches.
        if let Some(elev) = self.ast_elev.as_ref() {
            let cam = self.camera_eye_local();
            let m = rocket::asteroid_terrain(cam, self.ast_radius, elev, 15, self.lod_debug);
            self.terrain_count = (m.verts.len() as u64).min(TERRAIN_CAP) as u32;
            self.terrain_verts = m.verts;
            return;
        }
        // Deep-space (asteroid mesh / empty) scenes have no planet underfoot.
        if self.space {
            self.terrain_verts.clear();
            self.terrain_count = 0;
            return;
        }
        // Anchor the LOD to the grid-snapped reference (the rocket), not the
        // orbiting camera, so the selection - and thus the mesh - is determined
        // solely by where the rocket is, identical every rebuild at the same ref.
        let cam_world = self.cam_world(self.ref_local);
        let m = rocket::planet_terrain(
            cam_world,
            self.ref_local,
            self.launch_origin,
            self.launch_up,
            self.launch_east,
            self.launch_north,
            19,
            self.lunar,
            self.lod_debug,
        );
        self.terrain_count = (m.verts.len() as u64).min(TERRAIN_CAP) as u32;
        self.terrain_verts = m.verts;
        if std::env::var("MTS_DEBUG_VAB").is_ok() {
            let rb = self.resting_base_local().as_vec3();
            let r = self.ref_local.as_vec3();
            let near: Vec<f32> = self
                .terrain_verts
                .iter()
                .filter(|v| {
                    let p = Vec3::from(v.pos) + r;
                    (p.x - rb.x).abs() < 40.0 && (p.z - rb.z).abs() < 40.0
                })
                .map(|v| Vec3::from(v.pos).y + r.y)
                .collect();
            let lo = near.iter().cloned().fold(f32::MAX, f32::min);
            let hi = near.iter().cloned().fold(f32::MIN, f32::max);
            eprintln!("VAB resting_base.y={:.2} terrain near VAB y in [{:.2},{:.2}] ({} verts)", rb.y, lo, hi, near.len());
        }
    }

    /// Per-frame dynamic rocket-view geometry, camera-relative (floating origin):
    /// the pad + mount, the rocket at its current pose, and any tumbling spent
    /// booster. The full planet terrain lives in its own (rebuilt-on-move) buffer.
    fn build_dynamic_mesh(&self) -> Vec<rocket::MeshVertex> {
        let rb = &self.rocket_body;
        let mut out: Vec<rocket::MeshVertex> = Vec::new();

        // Asteroid LOD body: the body itself is the terrain buffer; only draw the
        // lander (at the +Y pole, where radial-up = +Y) if one is present.
        if self.ast_elev.is_some() {
            if self.show_lander {
                let base = DVec3::new(0.0, self.lander_alt as f64, 0.0);
                for v in &self.lander_mesh.verts {
                    let local = base + Vec3::from(v.pos).as_dvec3();
                    out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: v.normal, color: v.color });
                }
            }
            return out;
        }

        // static structures (camera-relative): pad + mount, the assembly hangar,
        // and the parts rack beside it.
        let push_static = |out: &mut Vec<rocket::MeshVertex>, mesh: &rocket::Mesh| {
            for v in &mesh.verts {
                out.push(rocket::MeshVertex {
                    pos: self.rel(Vec3::from(v.pos).as_dvec3()).into(),
                    normal: v.normal,
                    color: v.color,
                });
            }
        };
        // the moon base on the surface (preview / overview), instead of the stack
        if let Some(base) = self.base_mesh.as_ref() {
            for v in &base.verts {
                let local = Vec3::from(v.pos).as_dvec3();
                out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: v.normal, color: v.color });
            }
            // optionally also show the lander parked at the base
            if self.show_lander {
                let b = DVec3::new(0.0, self.lander_alt as f64, 0.0);
                for v in &self.lander_mesh.verts {
                    let local = b + Vec3::from(v.pos).as_dvec3();
                    out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: v.normal, color: v.color });
                }
            }
            return out;
        }
        // the lunar lander on the surface (preview / landed), instead of the launch stack
        if self.show_lander {
            let base = DVec3::new(0.0, self.lander_alt as f64, 0.0);
            for v in &self.lander_mesh.verts {
                let local = base + Vec3::from(v.pos).as_dvec3();
                out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: v.normal, color: v.color });
            }
            return out;
        }

        push_static(&mut out, &self.road_mesh);
        push_static(&mut out, &self.pad_mesh);
        push_static(&mut out, &self.hangar_mesh);
        push_static(&mut out, &self.rack_mesh);

        // The cities + their road network are spread across kilometres; only draw
        // the ones near the camera focus (each city is several km from the next,
        // so usually just the one you are in is built into the frame).
        let focus = self.camera_look_local();
        let near = |p: Vec3| {
            let d = p.as_dvec3() - focus;
            d.x * d.x + d.z * d.z < CITY_RENDER_R * CITY_RENDER_R
        };
        for (center, mesh) in &self.city_meshes {
            if near(*center) {
                push_static(&mut out, mesh);
            }
        }
        // The warm street/ground lighting is drawn only at night (its amber albedo
        // would tint the day streets), tracking the same per-city culling.
        if self.night {
            for (center, mesh) in &self.city_glow {
                if near(*center) {
                    push_static(&mut out, mesh);
                }
            }
        }
        for (a, b, mesh) in &self.roads {
            let mid = (*a + *b) * 0.5;
            if near(*a) || near(*b) || near(mid) {
                push_static(&mut out, mesh);
            }
        }

        // the car, parked or being driven: rotate its local mesh by the heading
        // about +Y and place it at the car position (mesh forward is +X).
        {
            let (s, c) = self.car_yaw.sin_cos();
            for v in &self.car_mesh.verts {
                let p = Vec3::from(v.pos);
                let rp = Vec3::new(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
                let n = Vec3::from(v.normal);
                let rn = Vec3::new(c * n.x - s * n.z, n.y, s * n.x + c * n.z);
                let local = self.car_pos + rp.as_dvec3();
                out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: rn.into(), color: v.color });
            }
        }

        // the third-person character (on foot only), posed for the walk cycle and
        // yawed to its heading.
        if self.walking {
            let moving = (self.ped_speed.abs() / 1.5).clamp(0.0, 1.0);
            let ped = rocket::character(self.walk_phase, moving, [0.20, 0.46, 0.78]);
            let (s, c) = self.ped_yaw.sin_cos();
            for v in &ped.verts {
                let p = Vec3::from(v.pos);
                let rp = Vec3::new(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
                let n = Vec3::from(v.normal);
                let rn = Vec3::new(c * n.x - s * n.z, n.y, s * n.x + c * n.z);
                let local = self.ped_pos + rp.as_dvec3();
                out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: rn.into(), color: v.color });
            }
        }

        // city traffic: NPC cars + pedestrians, for each active city crowd (the
        // player is near it). A small helper poses a local mesh by a heading about
        // +Y and drops it at a local position.
        let mut posed = |out: &mut Vec<rocket::MeshVertex>, verts: &[rocket::MeshVertex], yaw: f32, at: DVec3| {
            let (s, c) = yaw.sin_cos();
            for v in verts {
                let p = Vec3::from(v.pos);
                let rp = Vec3::new(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
                let n = Vec3::from(v.normal);
                let rn = Vec3::new(c * n.x - s * n.z, n.y, s * n.x + c * n.z);
                let local = at + rp.as_dvec3();
                out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: rn.into(), color: v.color });
            }
        };
        for t in &self.traffic {
            if !t.active {
                continue;
            }
            let center = t.center();
            for car in &t.cars {
                let mesh = &t.car_meshes[car.color];
                posed(&mut out, &mesh.verts, car.yaw(), car.pos(center));
            }
            for ped in &t.peds {
                let mesh = rocket::character(ped.phase, 1.0, traffic::Traffic::ped_shirt(ped.shirt));
                posed(&mut out, &mesh.verts, ped.yaw, ped.pos);
            }
        }

        // The mobile launch platform rides under the rocket from the hangar to
        // the pad. Drawn at the resting base (its X slides with rollout) only
        // while the rocket has not lifted off; once flying, it has left the deck.
        if self.launch.is_none() {
            let rb = self.resting_base_local();
            let deck = DVec3::new(rb.x, 0.0, rb.z); // platform built with ground at y=0
            for v in &self.platform_mesh.verts {
                let local = deck + Vec3::from(v.pos).as_dvec3();
                out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: v.normal, color: v.color });
            }
        }

        // current rocket pose: in the hangar / rolling out when not launched
        let (base_local, quat, active) = match self.launch.as_ref() {
            Some(rk) => {
                let base = self.to_local_d(rk.r);
                let q = self.rocket_quat(rk); // full attitude incl. roll
                (base, q, rk.stage_base)
            }
            None => (self.resting_base_local(), Quat::IDENTITY, 0),
        };

        // A destroyed vehicle is drawn as scattered debris (see below) instead
        // of the intact stack; while it's flying, draw the intact rocket.
        let destroyed = self.launch.as_ref().map(|rk| rk.destroyed).unwrap_or(false);
        if !destroyed {
            // draw the payload. When the fairing is closed, the whole payload
            // range draws as one; when opening, swing the two halves out.
            if self.fairing_open > 0.01 {
                let off = self.fairing_open * 4.0;
                self.xform_into(&mut out, rb.module_range.clone(), quat, base_local);
                self.xform_into_off(&mut out, rb.fairing_l.clone(), quat, base_local, Vec3::new(-off, 0.0, 0.0));
                self.xform_into_off(&mut out, rb.fairing_r.clone(), quat, base_local, Vec3::new(off, 0.0, 0.0));
            } else {
                self.xform_into(&mut out, rb.payload_range.clone(), quat, base_local);
            }
            for r in rb.stage_ranges.iter().skip(active) {
                self.xform_into(&mut out, r.clone(), quat, base_local);
            }
            // recovery parachute: a separate canopy mesh posed above the nose,
            // inflating with the rocket's chute fraction.
            if let Some(rk) = self.launch.as_ref() {
                if rk.chute > 0.02 {
                    let canopy = rocket::parachute_canopy(rb.height, rk.chute as f32);
                    for v in &canopy.verts {
                        let local = base_local + (quat * Vec3::from(v.pos)).as_dvec3();
                        let n = quat * Vec3::from(v.normal);
                        out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: n.into(), color: v.color });
                    }
                }
            }
        }

        // explosion debris: each chunk at its own tumbling pose, charred dark.
        for d in &self.debris {
            for v in &rb.mesh.verts[d.range.clone()] {
                let local = d.pos.as_dvec3() + (d.rot * Vec3::from(v.pos)).as_dvec3();
                let n = d.rot * Vec3::from(v.normal);
                // scorch the chunk: darken toward black as it ages
                let s = (1.0 - 0.55 * (d.age / 16.0).clamp(0.0, 1.0)).max(0.2);
                let c = [v.color[0] * s, v.color[1] * s, v.color[2] * s];
                out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: n.into(), color: c });
            }
        }

        // tumbling spent stage
        if let Some(s) = self.sep.as_ref() {
            for v in &rb.mesh.verts[s.range.clone()] {
                let local = s.pos.as_dvec3() + (s.rot * Vec3::from(v.pos)).as_dvec3();
                let n = s.rot * Vec3::from(v.normal);
                out.push(rocket::MeshVertex { pos: self.rel(local).into(), normal: n.into(), color: v.color });
            }
        }

        // drag ghost: the part being dragged from the rack (grab_ghost is already
        // camera-relative). Bright when snapped to a valid slot.
        if let Some(g) = self.grab {
            let col = if self.grab_target.is_some() { [0.5, 1.0, 0.7] } else { [0.8, 0.85, 1.0] };
            let mut gm = rocket::Mesh::default();
            rocket::append_part(&mut gm, g.kind, self.grab_ghost, col);
            out.extend(gm.verts);
        }

        out
    }

    /// Thruster FX billboards for the rocket view: an emissive flame at the
    /// active nozzle (axis-aligned cards facing the camera) and the smoke-
    /// particle trail (camera-facing puffs). `right`/`up` are the camera basis.
    fn build_fx(&self, eye: Vec3, right: Vec3, up: Vec3) -> Vec<FxVertex> {
        let mut out: Vec<FxVertex> = Vec::new();
        // Re-entry plasma is a glow-mesh envelope (plasma_mesh.wgsl) and the
        // engine exhaust plume is a volumetric pass (plume.wgsl). This builder
        // emits only the smoke trail and the lander/RCS jets.

        // ---- lunar descent-engine plume (under the lander's bell) ----
        if self.show_lander && self.lander_firing {
            let down = Vec3::new(0.0, -1.0, 0.0);
            // the descent bell exits near y=0.9 in the lander mesh; the lander
            // itself floats at lander_alt.
            let mount = DVec3::new(0.0, self.lander_alt as f64 + 0.9, 0.0);
            let nozzle = self.rel(mount);
            let view = (nozzle - eye).normalize_or_zero();
            let mut w_axis = down.cross(view).normalize_or_zero();
            if w_axis.length_squared() < 1e-4 {
                w_axis = right;
            }
            let mut card = |length: f32, half_w: f32, seed: f32, inten: f32| {
                let tip = nozzle + down * length;
                let wn = w_axis * half_w;
                let wt = w_axis * (half_w * 0.18);
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
            // a short, translucent-blue-ish vacuum plume: orange-ish edge, hot core
            card(7.0, 1.5, 0.18, 0.9);
            card(4.0, 0.7, 0.70, 1.2);
        }

        // ---- RCS attitude jets around the lander's upper body ----
        if self.show_lander && self.lander_rcs > 0.01 {
            // four thruster clusters at the corners of the descent stage, each
            // firing a short blue-white jet outward (lateral attitude control).
            let inten = self.lander_rcs.clamp(0.0, 1.0);
            let y = self.lander_alt as f64 + 3.6; // upper body
            let br = 2.3f64;
            for k in 0..4 {
                let a = (k as f64 + 0.5) * std::f64::consts::FRAC_PI_2;
                let (cx, cz) = (a.cos(), a.sin());
                // pulse opposite pairs so it reads as a control couple, not a flare
                let pulse = if k % 2 == 0 {
                    0.55 + 0.45 * (self.anim * 30.0).sin()
                } else {
                    0.55 + 0.45 * (self.anim * 30.0 + 3.14).sin()
                };
                let amp = (inten * pulse as f32).clamp(0.0, 1.0);
                if amp < 0.05 {
                    continue;
                }
                let nz_local = DVec3::new(cx * br, y, cz * br);
                let nozzle = self.rel(nz_local);
                let dir = Vec3::new(cx as f32, 0.0, cz as f32); // outward
                let view = (nozzle - eye).normalize_or_zero();
                let mut w_axis = dir.cross(view).normalize_or_zero();
                if w_axis.length_squared() < 1e-4 {
                    w_axis = up;
                }
                let len = 2.6f32 * (0.6 + 0.4 * amp);
                let tip = nozzle + dir * len;
                let wn = w_axis * 0.42;
                let wt = w_axis * 0.07;
                let col = [0.2f32, amp, 0.0, 0.0];
                let q = [
                    (nozzle - wn, [0.0f32, 0.0]),
                    (nozzle + wn, [1.0, 0.0]),
                    (tip + wt, [1.0, 1.0]),
                    (tip - wt, [0.0, 1.0]),
                ];
                for &i in &[0usize, 1, 2, 0, 2, 3] {
                    out.push(FxVertex { pos: q[i].0.into(), uv: q[i].1, color: col, kind: 2.0 });
                }
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

        // ---- explosion fireball particles (hot -> sooty, expanding) ----
        for b in &self.boom {
            let t = (b.age / b.life).clamp(0.0, 1.0);
            // grow as it expands; fade out near end of life
            let size = b.size0 * (0.6 + 1.9 * t);
            let alpha = (1.0 - t).powf(0.7) * 0.9;
            if alpha <= 0.01 {
                continue;
            }
            // colour ramp: white-hot -> yellow -> orange -> dark smoke
            let col = if t < 0.18 {
                let k = t / 0.18;
                [1.0, 0.95 - 0.1 * k, 0.7 - 0.4 * k]
            } else if t < 0.5 {
                let k = (t - 0.18) / 0.32;
                [1.0, 0.6 - 0.35 * k, 0.18 - 0.12 * k]
            } else {
                let k = (t - 0.5) / 0.5;
                let g = 0.32 - 0.24 * k;
                [g + 0.06, g, g]
            };
            let r = right * size;
            let u = up * size;
            let c = self.rel(b.pos.as_dvec3());
            let q = [
                (c - r - u, [0.0f32, 0.0]),
                (c + r - u, [1.0, 0.0]),
                (c + r + u, [1.0, 1.0]),
                (c - r + u, [0.0, 1.0]),
            ];
            // premultiplied-over smoke pipeline expects rgb*alpha in rgb.
            let cm = [col[0] * alpha, col[1] * alpha, col[2] * alpha, alpha];
            for &i in &[0usize, 1, 2, 0, 2, 3] {
                out.push(FxVertex { pos: q[i].0.into(), uv: q[i].1, color: cm, kind: 1.0 });
            }
        }

        // ---- re-entry sparks: bright additive embers shed off the shock ----
        for s in &self.sparks {
            let t = (s.age / s.life).clamp(0.0, 1.0);
            let bright = (1.0 - t) * (s.age / 0.03).min(1.0); // quick birth, fade with age
            if bright <= 0.02 {
                continue;
            }
            let size = 0.6 * (0.5 + 0.5 * (1.0 - t));
            let r = right * size;
            let u = up * size;
            let c = self.rel(s.pos.as_dvec3());
            let col = [s.seed, bright, 0.0, 0.0];
            let q = [
                (c - r - u, [0.0f32, 0.0]),
                (c + r - u, [1.0, 0.0]),
                (c + r + u, [1.0, 1.0]),
                (c - r + u, [0.0, 1.0]),
            ];
            for &i in &[0usize, 1, 2, 0, 2, 3] {
                out.push(FxVertex { pos: q[i].0.into(), uv: q[i].1, color: col, kind: 4.0 });
            }
        }
        out
    }

    /// Break the destroyed vehicle into tumbling debris and spawn a fireball.
    /// Each still-attached stage (and the payload) becomes a chunk flying out
    /// from the centre of mass; a burst of hot particles forms the explosion.
    fn spawn_explosion(&mut self) {
        let Some(rk) = self.launch.as_ref() else { return };
        let base = self.to_local(rk.r);
        let pd_local = self.dir_to_local(rk.point_dir());
        let rot = Quat::from_rotation_arc(Vec3::Y, pd_local);
        let vel_local = self.dir_to_local_vec(rk.v);
        let rmag = rk.r.length().max(1.0);
        let g_mag = (self.body.mu / (rmag * rmag)) as f32;
        let grav = -self.dir_to_local_vec(rk.r.normalize_or_zero()) * g_mag;
        let stage_base = rk.stage_base;
        // place the fireball where the camera is actually looking (up the stack),
        // so the blast frames on-screen instead of sitting below centre.
        let center_y = self.cam_focus_y.max(self.rocket_body.focus_y);

        // deterministic-ish RNG seeded by the impact state
        let mut seed = (rk.r.x.abs() * 13.0 + rk.met * 997.0).to_bits() ^ 0x9E3779B9;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };

        // chunks: each remaining stage + the payload, kicked radially outward
        let mut ranges: Vec<std::ops::Range<usize>> = self
            .rocket_body
            .stage_ranges
            .iter()
            .skip(stage_base)
            .cloned()
            .collect();
        ranges.push(self.rocket_body.payload_range.clone());
        self.debris.clear();
        for range in ranges {
            let kick = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize_or_zero()
                * (8.0 + rnd() * 22.0);
            let spin = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 3.5;
            self.debris.push(Debris {
                pos: base,
                vel: vel_local + kick,
                rot,
                spin,
                grav,
                age: 0.0,
                range,
            });
        }

        // fireball: a dense burst of hot particles from the centre of mass, with
        // a few big slow cores for the heart of the blast and many fast embers.
        let origin = base + pd_local * center_y;
        self.boom.clear();
        for k in 0..130 {
            let dir = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize_or_zero();
            let core = k < 30; // big, slow, central fire
            let spd = if core { 2.0 + rnd() * 9.0 } else { 14.0 + rnd() * 52.0 };
            let size0 = if core { 14.0 + rnd() * 14.0 } else { 5.0 + rnd() * 9.0 };
            let life = if core { 2.6 + rnd() * 2.2 } else { 1.0 + rnd() * 2.0 };
            self.boom.push(Boom {
                pos: origin + dir * (rnd() * 6.0),
                vel: vel_local * 0.35 + dir * spd,
                age: 0.0,
                life,
                size0,
                seed: rnd(),
            });
        }
        self.exploded = true;
    }

    /// Integrate explosion debris chunks + fireball particles (local frame).
    fn advance_debris(&mut self, dt: f32) {
        for d in self.debris.iter_mut() {
            d.age += dt;
            d.vel += d.grav * dt;
            d.pos += d.vel * dt;
            d.rot = (Quat::from_scaled_axis(d.spin * dt) * d.rot).normalize();
        }
        self.debris.retain(|d| d.age < 16.0);
        for b in self.boom.iter_mut() {
            b.age += dt;
            b.vel *= 1.0 - (1.6 * dt).min(0.9); // air-brake the blast
            b.vel.y += 3.0 * dt; // hot gas rises
            b.pos += b.vel * dt;
        }
        self.boom.retain(|b| b.age < b.life);
    }

    /// Integrate the jettisoned booster (local frame, ~9.2 m/s^2 down).
    fn advance_sep(&mut self, dt: f32) {
        // Launch-frame basis (Copy) captured before the &mut borrow, so we can
        // find the stage's altitude (hence air density) without borrowing self.
        let radius = self.body.radius;
        let (origin, east, up, north) =
            (self.launch_origin, self.launch_east, self.launch_up, self.launch_north);
        if let Some(s) = self.sep.as_mut() {
            s.age += dt;
            // air density at the stage's current altitude (sea level = 1, ~0 in
            // space): a stage shed in vacuum coasts; one shed low brakes hard.
            let world = origin
                + east * s.pos.x as f64
                + up * s.pos.y as f64
                + north * s.pos.z as f64;
            let alt = (world.length() - radius).max(0.0) as f32;
            let rho = (-alt / 8500.0).exp();
            // gravity (altitude-appropriate, captured at separation)
            s.vel += s.grav * dt;
            // atmospheric drag: F/m ~ rho * v^2 opposing motion, decelerating the
            // stage toward a terminal velocity in dense air. Clamp the per-step
            // impulse so a fast stage hitting thick air stays stable.
            let speed = s.vel.length();
            let drag_f = (0.0004 * rho * speed * dt).min(0.5);
            s.vel -= s.vel * drag_f;
            s.pos += s.vel * dt;
            // tumble: aerodynamic torque spins it faster the higher the dynamic
            // pressure (rho*v^2); in vacuum it keeps its gentle hand-off spin.
            let tumble = 1.0 + (rho * speed * speed * 1.0e-4).min(4.0);
            s.rot = (Quat::from_scaled_axis(s.spin * tumble * dt) * s.rot).normalize();
            if s.age > 25.0 {
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

    /// Hand the active flight craft to the autonomous moon-landing bot (or take
    /// back manual control). Engages a craft first if none is flying.
    fn toggle_moonbot(&mut self) {
        if self.moonbot.is_some() {
            self.moonbot = None;
            log::info!("Moon bot disengaged - you have control");
            return;
        }
        if self.flight.is_none() {
            self.toggle_flight();
        }
        if self.flight.is_some() {
            self.moonbot = Some(bot::MoonBot::new());
            log::info!("Moon bot engaged - flying to the surface");
        }
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
            // The flown ascent path (cyan) so the live trajectory shows on the
            // map during the sub-orbital climb, the forward conic prediction
            // (amber) ahead of it, and the parking-orbit conic once bound.
            polyline(&rk.trail_points(&self.body), [0.45, 0.9, 1.0], &mut out);
            polyline(&rk.forward_arc(&self.body), [1.0, 0.62, 0.20], &mut out);
            let pred = rk.predicted_orbit(&self.body);
            polyline(&pred, [1.0, 0.7, 0.25], &mut out);
        } else if let Some(craft) = self.flight.as_ref() {
            let pred = craft.predicted_orbit(&self.body);
            polyline(&pred, [0.5, 0.55, 0.25], &mut out);
            // planned maneuver: the resulting orbit, in cyan
            if let Some(n) = self.node {
                let after = craft.node_orbit(&self.body, n.nu, n.pro, n.nrm, n.rad);
                polyline(&after, [0.3, 0.85, 1.0], &mut out);
            }
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

        // rings of payloads our missions placed in orbit (visible zoomed to home)
        if self.sys_dist < 200.0 && !self.orbits.is_empty() {
            let home_pos = self.universe.position(self.universe.home_index(), t);
            for o in &self.orbits {
                let col = [o.color[0] * 0.7, o.color[1] * 0.7, o.color[2] * 0.7];
                let mut prev: Option<[f32; 2]> = None;
                for k in 0..=96 {
                    let th = k as f32 / 96.0 * std::f32::consts::TAU;
                    let p = home_pos
                        + (o.t1 * th.cos() + o.t2 * th.sin()).as_dvec3() * o.radius_mm as f64;
                    let cur = cam.project(p);
                    if let (Some(a), Some(b)) = (prev, cur) {
                        out.push(OverlayVertex { pos: a, color: col });
                        out.push(OverlayVertex { pos: b, color: col });
                    }
                    prev = cur;
                }
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

        // maneuver-node marker (cyan) on the craft's orbit
        if let (Some(craft), Some(n)) = (self.flight.as_ref(), self.node) {
            let np = craft.node_marker(&self.body, n.nu);
            if let Some(c) = cam.project(home_pos + np.as_dvec3() * self.home_radius_mm as f64) {
                push_filled_diamond(&mut out, c, 0.024, aspect, [0.0, 0.0, 0.0]);
                push_filled_diamond(&mut out, c, 0.015, aspect, [0.3, 0.85, 1.0]);
            }
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

        // payloads our missions have placed in orbit around the home world
        for o in &self.orbits {
            if let Some(c) = cam.project(o.pos_mm(home_pos, self.sys_time)) {
                push_filled_diamond(&mut out, c, 0.011, aspect, [0.0, 0.0, 0.0]);
                push_filled_diamond(&mut out, c, 0.007, aspect, o.color);
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
    /// Re-entry plasma: the procedural glow-mesh envelope (depth-tested geometry).
    plasma_mesh_pipeline: wgpu::RenderPipeline,
    plasma_mesh_vbuf: wgpu::Buffer,
    /// Volumetric exhaust plume (fullscreen raymarch, additive over the scene).
    plume_pipeline: wgpu::RenderPipeline,
    plume_uniform_buf: wgpu::Buffer,
    plume_bind_group: wgpu::BindGroup,
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

        // Plume pipeline: fullscreen volumetric exhaust, composited additively.
        let plume_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("plume"),
            source: wgpu::ShaderSource::Wgsl(include_str!("plume.wgsl").into()),
        });
        let plume_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plume-uniforms"),
            size: std::mem::size_of::<PlumeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plume_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("plume-bind-layout"),
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
        let plume_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("plume-bind-group"),
            layout: &plume_bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: plume_uniform_buf.as_entire_binding() }],
        });
        let plume_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("plume-layout"),
            bind_group_layouts: &[Some(&plume_bind_layout)],
            immediate_size: 0,
        });
        let plume_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("plume-pipeline"),
            layout: Some(&plume_layout),
            vertex: wgpu::VertexState {
                module: &plume_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &plume_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // additive: the emissive jet adds light to the scene.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                // Occlude the plume behind closer geometry (the rocket body when it
                // is between the camera and the exhaust). The shader writes the
                // plume's near-surface depth; this tests it against the scene.
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // PROTOTYPE: procedural glow-mesh plasma. Depth-tested geometry (so it
        // occludes correctly), shaded with the same cooling ramp + turbulence.
        // Reuses the mesh uniform (viewproj + log depth + time) and vertex layout.
        let plasma_mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("plasma-mesh"),
            source: wgpu::ShaderSource::Wgsl(include_str!("plasma_mesh.wgsl").into()),
        });
        let plasma_mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("plasma-mesh-pipeline"),
            layout: Some(&mesh_layout),
            vertex: wgpu::VertexState {
                module: &plasma_mesh_shader,
                entry_point: Some("vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<rocket::MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &plasma_mesh_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // premultiplied-over: rgb already carries rgb*alpha.
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
            // Depth-test against the scene (so terrain/the vehicle occlude it), but
            // do not write depth (it is translucent and self-overlapping).
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let plasma_mesh_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plasma-mesh-vbuf"),
            size: PLASMA_MESH_CAP * std::mem::size_of::<rocket::MeshVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
            plasma_mesh_pipeline,
            plasma_mesh_vbuf,
            plume_pipeline,
            plume_uniform_buf,
            plume_bind_group,
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
    ) -> (usize, usize, u32, u32, bool, bool) {
        let res = [w as f32, h.max(1) as f32];
        let mut dyn_n = 0u32;
        let mut fx_n = 0u32;
        let mut plasma_on = false;
        let mut plume_on = false;
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
                // re-entry plasma: build the glow-mesh envelope once there is any
                // meaningful heating. The mesh brightness then scales with the heat
                // level (per-vertex), so the glow fades in low->high with friction
                // instead of snapping on at a threshold.
                world.plasma_mesh_n = 0;
                if world.plasma_heat() > 0.04 {
                    plasma_on = true;
                    let mv = world.plasma_mesh();
                    let mn = (mv.len() as u64).min(PLASMA_MESH_CAP) as u32;
                    if mn > 0 {
                        queue.write_buffer(
                            &self.plasma_mesh_vbuf,
                            0,
                            bytemuck::cast_slice(&mv[..mn as usize]),
                        );
                    }
                    world.plasma_mesh_n = mn;
                }
                // volumetric exhaust plume while the active engine is firing.
                if world.plume_intensity() > 0.01 {
                    let pu = world.plume_uniforms(res);
                    queue.write_buffer(&self.plume_uniform_buf, 0, bytemuck::bytes_of(&pu));
                    plume_on = true;
                }
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
        (n, hn, dyn_n, fx_n, plasma_on, plume_on)
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

    /// Re-entry plasma: the procedural glow-mesh envelope (depth-tested geometry).
    fn draw_plasma_mesh(&self, pass: &mut wgpu::RenderPass, count: u32) {
        if count > 0 {
            pass.set_pipeline(&self.plasma_mesh_pipeline);
            pass.set_bind_group(0, &self.mesh_bind_group, &[]);
            pass.set_vertex_buffer(0, self.plasma_mesh_vbuf.slice(..));
            pass.draw(0..count, 0..1);
        }
    }

    /// Volumetric exhaust plume, composited additively over the scene.
    fn draw_plume(&self, pass: &mut wgpu::RenderPass, on: bool) {
        if on {
            pass.set_pipeline(&self.plume_pipeline);
            pass.set_bind_group(0, &self.plume_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// Compact large counts for the profiler readout: 950, 12.3k, 1.20M.
fn fmt_count(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f32 / 1_000_000.0)
    } else if n >= 10_000 {
        format!("{:.1}k", n as f32 / 1_000.0)
    } else {
        n.to_string()
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
    /// Depth attachment for the rocket-view mesh pass, recreated only on resize
    /// (not per frame).
    depth_view: wgpu::TextureView,
    gpu: Gpu,
    world: World,
    start: instant_now::Instant,
    last_t: f32,
    dragging: bool,
    last_cursor: (f64, f64),
    /// Rolling FPS readout shown in the window title: frames since `fps_since`.
    fps_frames: u32,
    fps_since: f32,
    /// (timestamp_s, frame_ms) samples for the on-screen graph, trimmed to the
    /// last 10 s. Uses the true unclamped frame time so spikes show honestly.
    frame_log: std::collections::VecDeque<(f32, f32)>,
    /// Last frame's geometry load, shown in the graph overlay. One frame stale
    /// (egui's primitives are only known after the graph itself is built), which
    /// is imperceptible.
    tri_count: u32,
    draw_calls: u32,
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
        let depth_view = create_depth(&device, width, height);

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
            depth_view,
            gpu,
            world: {
                // Live game: run the launch physics on its own dedicated thread,
                // decoupled from the render frame rate.
                let mut w = World::new();
                w.enable_threaded_sim();
                w
            },
            start: instant_now::Instant::now(),
            last_t: 0.0,
            dragging: false,
            last_cursor: (0.0, 0.0),
            fps_frames: 0,
            fps_since: 0.0,
            frame_log: std::collections::VecDeque::new(),
            tri_count: 0,
            draw_calls: 0,
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
            self.depth_view = create_depth(&self.device, width, height);
        }
    }

    /// Draw the rolling frame-time graph (last ~10 s) as a small non-interactive
    /// overlay in the bottom-left corner. Must run between egui's `begin_pass`
    /// and `end_pass`. The blue trace is per-frame milliseconds; the green and
    /// amber reference lines mark the 60 fps (16.7 ms) and 30 fps (33.3 ms)
    /// budgets, so spikes above them read at a glance.
    fn draw_frame_graph(&self) {
        use egui::{Align2, Color32, FontId, Pos2, Sense, Stroke, Vec2};
        if self.frame_log.len() < 2 {
            return;
        }
        let now = self.last_t;
        let window_s = 10.0_f32;
        let peak = self.frame_log.iter().map(|&(_, ms)| ms).fold(0.0_f32, f32::max);
        // Auto-scale the y axis to the rolling peak, floored so the 60 fps line
        // is always visible and capped so one huge spike doesn't squash detail.
        let max_ms = peak.max(33.4).min(200.0);
        let cur = self.frame_log.back().map(|&(_, ms)| ms).unwrap_or(0.0);

        egui::Area::new(egui::Id::new("frame-graph"))
            .anchor(Align2::LEFT_BOTTOM, Vec2::new(12.0, -12.0))
            .interactable(false)
            .show(&self.egui_ctx, |ui| {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(300.0, 96.0), Sense::hover());
                let p = ui.painter();
                p.rect_filled(rect, egui::CornerRadius::same(4), Color32::from_black_alpha(190));
                for (ms, col) in [
                    (1000.0 / 60.0, Color32::from_rgb(70, 150, 95)),
                    (1000.0 / 30.0, Color32::from_rgb(175, 140, 60)),
                ] {
                    let y = rect.bottom() - (ms / max_ms) * rect.height();
                    p.line_segment(
                        [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                        Stroke::new(1.0, col),
                    );
                }
                let pts: Vec<Pos2> = self
                    .frame_log
                    .iter()
                    .map(|&(ts, ms)| {
                        let x = rect.right() - ((now - ts) / window_s) * rect.width();
                        let y = rect.bottom() - (ms.min(max_ms) / max_ms) * rect.height();
                        Pos2::new(x.max(rect.left()), y)
                    })
                    .collect();
                p.add(egui::Shape::line(pts, Stroke::new(1.0, Color32::from_rgb(120, 200, 255))));
                p.text(
                    rect.left_top() + Vec2::new(6.0, 4.0),
                    Align2::LEFT_TOP,
                    format!("{:?}  {cur:.1} ms  peak {peak:.1} ms / 10s", self.world.view),
                    FontId::monospace(11.0),
                    Color32::from_rgb(210, 228, 245),
                );
                p.text(
                    rect.left_top() + Vec2::new(6.0, 18.0),
                    Align2::LEFT_TOP,
                    format!(
                        "{} tris  {} draws",
                        fmt_count(self.tri_count),
                        self.draw_calls
                    ),
                    FontId::monospace(11.0),
                    Color32::from_rgb(160, 190, 215),
                );
                // Lights up while the worker thread is meshing terrain - the
                // rebuild that used to spike the frame now runs here instead.
                if self.world.terrain_svc.busy() {
                    p.text(
                        rect.right_top() + Vec2::new(-6.0, 4.0),
                        Align2::RIGHT_TOP,
                        "meshing terrain",
                        FontId::monospace(11.0),
                        Color32::from_rgb(120, 230, 140),
                    );
                }
            });
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
        let raw_dt = (t - self.last_t).max(0.0);
        let frame_dt = raw_dt.min(0.1);
        self.last_t = t;
        self.world.advance(frame_dt);

        // Record the true (unclamped) frame time for the on-screen graph and
        // trim to the last 10 s. Skip absurd gaps (startup, tab-out/resume) so a
        // single multi-second stall doesn't flatten the whole graph.
        let frame_ms = raw_dt * 1000.0;
        if frame_ms < 1000.0 {
            self.frame_log.push_back((t, frame_ms));
        }
        while matches!(self.frame_log.front(), Some(&(ts, _)) if t - ts > 10.0) {
            self.frame_log.pop_front();
        }

        // Rolling FPS readout (once per second) so it's clear which view is slow.
        self.fps_frames += 1;
        let span = t - self.fps_since;
        if span >= 1.0 {
            let fps = self.fps_frames as f32 / span;
            self.window.set_title(&format!(
                "Mining the Sky - {:?} view - {:.0} fps ({:.1} ms)",
                self.world.view,
                fps,
                1000.0 / fps.max(1.0),
            ));
            self.fps_frames = 0;
            self.fps_since = t;
        }

        let (n, hn, dyn_n, fx_n, plasma_on, plume_on) = self
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
        self.draw_frame_graph();
        let full = self.egui_ctx.end_pass();
        self.egui_state
            .handle_platform_output(&self.window, full.platform_output);
        let ppp = self.egui_ctx.pixels_per_point();
        let prims = self.egui_ctx.tessellate(full.shapes, ppp);

        // Geometry load for the profiler overlay. egui contributes one draw and
        // indices/3 triangles per clipped primitive; the scene pipelines are
        // triangle lists except the line overlay (counted as draws, not tris).
        let egui_tris: u32 = prims
            .iter()
            .map(|p| match &p.primitive {
                egui::epaint::Primitive::Mesh(m) => (m.indices.len() / 3) as u32,
                _ => 0,
            })
            .sum();
        let egui_draws = prims.len() as u32;
        let (scene_tris, scene_draws) = if self.world.view == View::Rocket {
            let tris = 1 // sky fullscreen triangle
                + terrain_n / 3
                + dyn_n / 3
                + fx_n / 3
                + hn as u32 / 3;
            let draws = 1 // sky
                + (terrain_n > 0) as u32
                + (dyn_n > 0) as u32
                + (fx_n > 0) as u32
                + (n > 0) as u32 // overlay (lines)
                + (hn > 0) as u32; // hud
            (tris, draws)
        } else {
            // Map: one fullscreen raymarch triangle + hud; overlay is lines.
            (1 + hn as u32 / 3, 1 + (n > 0) as u32 + (hn > 0) as u32)
        };
        self.tri_count = scene_tris + egui_tris;
        self.draw_calls = scene_draws + egui_draws;

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
            let plasma_mesh_n = self.world.plasma_mesh_n;
            {
                let mut pass = mesh_pass(&mut encoder, &view, &self.depth_view);
                self.gpu.draw_sky(&mut pass);
                self.gpu.draw_meshes(&mut pass, terrain_n, dyn_n);
                self.gpu.draw_plume(&mut pass, plume_on);
                // re-entry plasma: depth-tested glow-mesh envelope, in the pass.
                if plasma_on {
                    self.gpu.draw_plasma_mesh(&mut pass, plasma_mesh_n);
                }
                // FX last, so additive sparks/embers sit on top of the plasma glow.
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

/// Measure the re-entry plasma cost on the real GPU. Loops the actual render
/// passes K times in a single submit and blocks on the GPU (poll-wait), so wall
/// time is dominated by GPU execution. Reports the isolated plasma pass and the
/// whole rocket frame with plasma on vs off, so the plasma's share of the frame
/// is a measured number rather than a guess.
#[cfg(not(target_arch = "wasm32"))]
fn bench_plasma(scenario: &str, width: u32, height: u32) {
    use std::time::Instant;
    let (device, queue) = make_shot_device();
    let gpu = Gpu::new(&device, &queue, wgpu::TextureFormat::Rgba8UnormSrgb);
    let (mut world, time) = setup_world(scenario, width, height);
    let (n, hn, dyn_n, fx_n, plasma_on, plume_on) =
        gpu.prepare(&queue, &mut world, width, height, time);
    if !plasma_on {
        println!("plasma not active for scenario '{scenario}' - nothing to bench");
        return;
    }
    let terrain_n = world.terrain_count;
    let pmesh_n = world.plasma_mesh_n;
    let tris = pmesh_n / 3;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench-target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = create_depth(&device, width, height);

    // Encode the whole rocket-view frame (the same passes `render_shot` uses),
    // optionally including the plasma glow-mesh draw.
    let encode_frame = |enc: &mut wgpu::CommandEncoder, plasma: bool| {
        {
            let mut pass = mesh_pass(enc, &target_view, &depth);
            gpu.draw_sky(&mut pass);
            gpu.draw_meshes(&mut pass, terrain_n, dyn_n);
            gpu.draw_plume(&mut pass, plume_on);
            if plasma {
                gpu.draw_plasma_mesh(&mut pass, pmesh_n);
            }
            gpu.draw_fx(&mut pass, fx_n);
        }
        {
            let mut pass = overlay_pass(enc, &target_view);
            gpu.draw_overlay(&mut pass, n, hn);
        }
    };

    // Time `k` encodes (per repetition) of `encode`, GPU-synced; return best ms.
    let time_k = |k: u32, reps: u32, encode: &dyn Fn(&mut wgpu::CommandEncoder)| -> f64 {
        {
            // warmup: shader compile, buffer upload, GPU clock ramp
            let mut e = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            for _ in 0..32 {
                encode(&mut e);
            }
            queue.submit([e.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let mut e = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            for _ in 0..k {
                encode(&mut e);
            }
            let cb = e.finish();
            let t0 = Instant::now();
            queue.submit([cb]);
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            best = best.min(t0.elapsed().as_secs_f64() / k as f64 * 1000.0);
        }
        best
    };

    let frame_on = time_k(120, 6, &|e| encode_frame(e, true));
    let frame_off = time_k(120, 6, &|e| encode_frame(e, false));

    println!("re-entry plasma bench, scenario '{scenario}'");
    println!("  {width}x{height}, glow-mesh envelope: {tris} triangles");
    println!("  whole rocket frame, plasma ON:  {frame_on:.3} ms");
    println!("  whole rocket frame, plasma OFF: {frame_off:.3} ms");
    println!("  => plasma adds {:.3} ms to the frame", frame_on - frame_off);
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

    // Autopilot: fly a closed-loop gravity turn + circularization until a stable
    // parking orbit is reached (or fuel runs out). Used to verify the
    // mission-to-orbit -> persistent-satellite loop headlessly.
    fn fly_to_orbit(w: &mut World) {
        let radius = w.body.radius;
        for _ in 0..12000 {
            let done = w
                .launch
                .as_ref()
                .map(|rk| rk.orbit_reached || rk.crashed)
                .unwrap_or(true);
            if done {
                break;
            }
            // read state, then steer
            let (met, apo, dry) = {
                let rk = w.launch.as_ref().unwrap();
                let orb = sim::orbit::orbit_from_state(rk.r, rk.v, w.body.mu);
                (rk.met, orb.ra - radius, rk.prop_frac() <= 0.0)
            };
            if dry {
                if w.launch.as_ref().map(|r| r.stages.len() > 1).unwrap_or(false) {
                    w.stage_launch();
                } else {
                    break; // out of fuel, no more stages
                }
            }
            if let Some(rk) = w.launch.as_mut() {
                // vertical, then gravity-turn to ~80 deg, then horizontal to raise
                // periapsis once the apoapsis is high enough.
                rk.pitch = if met < 12.0 {
                    0.0
                } else if apo < 180_000.0 {
                    (((met - 12.0) / 110.0).clamp(0.0, 1.0) * 80f64.to_radians()).min(1.4)
                } else {
                    90f64.to_radians()
                };
            }
            w.advance(0.25);
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
        "overlook" => {
            // a high vantage west of the pad looking east over the coast to the
            // ocean, to evaluate the landscape and water surface.
            world.view = View::Rocket;
            world.rocket_az = std::f32::consts::PI;
            world.rocket_el = 0.16;
            world.rocket_dist = 6000.0;
            for _ in 0..40 {
                world.advance(0.1);
            }
            0.0
        }
        "cities" => {
            // an elevated look across the main metro: a dense cluster of downtowns
            // packed into one big sprawl, linked by avenues.
            world.view = View::Rocket;
            world.enter_drive();
            world.car_pos = CITY_CENTER.as_dvec3();
            for _ in 0..20 {
                world.advance(0.1);
            }
            world.rocket_az = 2.3;
            world.rocket_el = 0.42;
            world.rocket_dist = 1500.0;
            0.0
        }
        "descent" => {
            // mid descent: a closer pass over the night side, the city-light
            // clusters filling more of the frame on the way down.
            frame_map(&mut world);
            world.sys_dist = 9.5;
            world.sys_az = 4.3;
            world.sys_el = 0.14;
            6.0
        }
        "cityrise" => {
            // Camera rising straight up over the metro, tilting down to keep it
            // framed, for a ground-to-altitude ascent sequence. Height (km) and
            // day/night come from env vars so one scenario drives the whole run.
            world.view = View::Rocket;
            let km: f64 = std::env::var("ASCENT_KM").ok().and_then(|s| s.parse().ok()).unwrap_or(2.0);
            let night = std::env::var("ASCENT_NIGHT").ok().map(|s| s == "1").unwrap_or(false);
            world.night = night;
            world.enter_drive();
            world.car_pos = CITY_CENTER.as_dvec3();
            for _ in 0..30 {
                world.advance(0.1); // settle terrain LOD + wake the crowd
            }
            // Fixed horizontal standoff; the camera climbs to height `h` and tilts
            // down (elevation grows with height), focused on the city centre.
            let h = km * 1000.0;
            let l = 2500.0f64;
            world.rocket_dist = (l * l + h * h).sqrt() as f32;
            world.rocket_el = h.atan2(l) as f32;
            world.rocket_az = std::f32::consts::FRAC_PI_2;
            0.0
        }
        "nightcity" => {
            // street level in the metro at night: the sky is dark and the
            // emissive windows + traffic carry the scene.
            world.view = View::Rocket;
            world.night = true;
            world.enter_drive();
            // sit the car at a street intersection (a lamp + pool right there) on
            // a north-south avenue, facing north up the street.
            let lane_x = (CITY_CENTER.x - 210.0 + 3.0 * 60.0) as f64;
            world.car_pos = DVec3::new(lane_x, 0.0, (CITY_CENTER.z - 210.0 + 2.0 * 60.0) as f64);
            world.car_yaw = std::f32::consts::FRAC_PI_2;
            for _ in 0..45 {
                world.advance(0.1);
            }
            world.car_yaw = std::f32::consts::FRAC_PI_2;
            world.rocket_az = world.car_yaw + std::f32::consts::PI;
            world.rocket_el = 0.13;
            world.rocket_dist = 26.0;
            0.0
        }
        "citylife" => {
            // the city brought to life: drive in, step the sim so NPC cars and
            // pedestrians spawn and disperse, then frame the street.
            world.view = View::Rocket;
            world.enter_drive();
            // sit on a north-south street lane near the south edge, facing north up
            // the avenue, with a slightly raised chase camera so the traffic on
            // the street and at the cross-street intersections reads.
            let lane_x = (CITY_CENTER.x - 210.0 + 3.0 * 60.0) as f64;
            world.car_pos = DVec3::new(lane_x, 0.0, (CITY_CENTER.z - 175.0) as f64);
            world.car_yaw = std::f32::consts::FRAC_PI_2;
            for _ in 0..45 {
                world.advance(0.1); // spawn + move the crowd a few seconds
            }
            world.car_yaw = std::f32::consts::FRAC_PI_2;
            world.rocket_az = world.car_yaw + std::f32::consts::PI;
            world.rocket_el = 0.30;
            world.rocket_dist = 46.0;
            0.0
        }
        "walk" => {
            // the third-person character on foot by the car, mid-stride.
            world.view = View::Rocket;
            world.enter_walk();
            world.ped_pos = DVec3::new(28.0, 0.0, 6.0);
            world.ped_yaw = 0.5;
            world.ped_speed = 3.5; // pose the walk cycle (idle would stand still)
            world.walk_phase = 1.2;
            // 3/4 front view so the face + swept fringe read against the back hair.
            world.rocket_az = world.ped_yaw - 0.7;
            world.rocket_el = 0.10;
            world.rocket_dist = 5.5;
            0.0
        }
        "drive" => {
            // the car on the road just south of the city, third-person chase
            // camera trailing it as it faces into the downtown grid.
            world.view = View::Rocket;
            world.enter_drive();
            world.car_pos = (CITY_CENTER + Vec3::new(0.0, 0.0, -250.0)).as_dvec3();
            world.car_yaw = std::f32::consts::FRAC_PI_2; // face +Z (toward the city)
            world.rocket_az = world.car_yaw + std::f32::consts::PI;
            world.rocket_el = 0.20;
            world.rocket_dist = 26.0;
            0.0
        }
        "cityview" => {
            // a high overview framing the whole downtown
            world.view = View::Rocket;
            world.enter_drive();
            world.car_pos = CITY_CENTER.as_dvec3();
            world.rocket_az = 0.9;
            world.rocket_el = 0.55;
            world.rocket_dist = 620.0;
            0.0
        }
        "rocket" | "pad" => {
            // rolled out, standing on the pad
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.rocket_az = 4.97; // face inland (land), coast to the sides
            world.rocket_el = 0.12;
            0.0
        }
        "lander" => {
            // the 3D lunar descent module standing on the surface
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.show_lander = true;
            world.rocket_az = 5.4;
            world.rocket_el = 0.16;
            world.rocket_dist = 22.0;
            world.rocket_focus_y = 3.0;
            0.0
        }
        "vab" => {
            // inside the assembly building, looking at the rocket (default start)
            world.view = View::Rocket;
            world.vab_mode = true;
            world.rollout = 0.0;
            world.rocket_az = 0.7;
            world.rocket_el = 0.18;
            world.rocket_dist = 52.0;
            0.0
        }
        "grabdemo" => {
            // verify the pick -> drop -> rebuild path headlessly: grab the X-Large
            // tank off the rack, target stage 0's tank slot, and drop it. The
            // rocket should rebuild with a fat first stage.
            world.view = View::Rocket;
            world.vab_mode = true;
            world.rollout = 0.0;
            world.rocket_az = 0.7;
            world.rocket_el = 0.16;
            world.rocket_dist = 70.0;
            world.grab = world
                .rack_slots
                .iter()
                .find(|s| s.kind == rocket::PartKind::Tank && s.idx == 3) // X-Large
                .copied();
            world.grab_target = Some((rocket::PartKind::Tank, 0));
            world.drop_grab();
            0.0
        }
        "rollout" => {
            // mid roll-out: the rocket part-way between hangar and pad, with the
            // assembly panel showing roll-out progress + the crawler-speed control
            world.view = View::Rocket;
            world.vab_mode = true;
            world.rolling_out = true;
            world.rollout = 0.84; // just clear of the building, nearing the pad
            world.rollout_speed = 8.0; // fast-forwarded by the player
            world.rocket_az = 5.2;
            world.rocket_el = 0.16;
            world.rocket_dist = 120.0;
            0.0
        }
        "liftoff" => {
            world.view = View::Rocket;
            world.rocket_az = 4.97;
            world.rocket_el = 0.20;
            world.rocket_dist = 150.0;
            world.rocket_focus_y = 28.0; // frame the rocket rising out of the cloud
            world.ignite_launch();
            fly(&mut world, 2.8); // climbing out of the big ignition billow
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
        "loddebug" => {
            // On the pad with LOD-debug colouring on and the camera pulled back
            // and up, so the cube-sphere quadtree split rings spread out from the
            // launch site, one flat colour per depth. The floating origin is
            // stable here (no in-flight rebuild), so the rings centre cleanly on
            // the camera the way they do interactively.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.rocket_az = 4.97;
            world.rocket_el = 0.42;
            world.rocket_dist = 4200.0;
            world.lod_debug = true;
            world.rebuild_terrain();
            0.0
        }
        "loddebugmap" => {
            // a launch shown on the orbital map, zoomed to the launch site: the
            // cyan flown trail and the amber forward conic make the trajectory
            // readable mid-ascent (before a parking orbit exists).
            world.ignite_launch();
            fly(&mut world, 150.0);
            frame_map(&mut world);
            world.sys_dist = 11.0;
            world.sys_az = 4.7;
            world.sys_el = 0.18;
            6.0
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
        "boosters" => {
            // a core stage ringed with radial solid rocket boosters, on the pad.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.vab.stages[0].boosters = 6;
            world.vab.stages[0].booster = 1; // SRB-Heavy
            world.rebuild_vehicle();
            world.rocket_az = 4.6;
            world.rocket_el = 0.14;
            world.rocket_dist = 95.0;
            0.0
        }
        "boosterlaunch" => {
            // a boostered stack lifting off: core + 6 SRBs all firing.
            world.view = View::Rocket;
            world.vab.stages[0].boosters = 6;
            world.vab.stages[0].booster = 1;
            world.rebuild_vehicle();
            world.rocket_az = 4.6;
            world.rocket_el = 0.12;
            world.rocket_dist = 120.0;
            world.ignite_launch();
            fly(&mut world, 4.0);
            0.0
        }
        "crash" => {
            // a structural failure mid-air: the vehicle bursts into tumbling
            // debris + a fireball (same path a ground crash or burn-through takes).
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.rocket_az = 4.97;
            world.rocket_el = 0.05;
            world.rocket_dist = 30.0;
            world.ignite_launch();
            world.advance(0.05); // let the camera focus settle on the stack
            if let Some(rk) = world.launch.as_mut() {
                rk.health = 0.0; // force the break-up
            }
            for _ in 0..5 {
                world.advance(0.1); // spawn + bloom the fireball (~0.5 s)
            }
            0.0
        }
        "reentry" => {
            // a vehicle screaming back into the upper atmosphere: the reentry
            // plasma bow shock + streaks at full heat (nose-first).
            world.setup_reentry(0);
            0.0
        }
        "plumetest" => {
            // A firing rocket viewed from above the nose, so the BODY is between
            // the camera and the exhaust: the plume must be occluded by the body,
            // not drawn through it. Used to verify plume depth occlusion.
            world.view = View::Rocket;
            world.ignite_launch();
            let (radius, up) = (world.body.radius, world.launch_up);
            if let Some(rk) = world.launch.as_mut() {
                rk.r = up * (radius + 20_000.0); // 20 km up, no ground in frame
                rk.throttle = 1.0;
                rk.spool = 1.0; // full thrust so the plume is lit
            }
            world.advance(0.05); // build the plume
            world.rocket_az = 0.6;
            world.rocket_el = 1.32; // look nearly straight DOWN the body's axis
            world.rocket_dist = 30.0; // ...so the body covers the exhaust behind it
            0.0
        }
        "reentry_lowheat" => {
            // low friction: the FX should be a faint dull-red wisp (verifies the
            // glow fades in low->high rather than snapping on).
            world.setup_reentry(0);
            if let Some(rk) = world.launch.as_mut() {
                rk.heat = 0.22;
            }
            0.0
        }
        "reentry_midheat" => {
            // mid friction: orange body building toward yellow, no white-hot yet.
            world.setup_reentry(0);
            if let Some(rk) = world.launch.as_mut() {
                rk.heat = 0.55;
            }
            0.0
        }
        "reentry_boost" => {
            // a heavily boostered stack (6 strap-ons => ~16 SDF prims) at full
            // heat: the worst case for the plasma SDF cost, used by `bench`.
            world.vab.stages[0].boosters = 6;
            world.vab.stages[0].booster = 1; // SRB-Heavy
            world.setup_reentry(0);
            0.0
        }
        "reentry_roll" => {
            // boostered + pitched + rolled about the nose: verifies a roll command
            // is actually rendered (the strap-ons rotate about the axis), now that
            // the drawn pose carries the full attitude and not just the nose.
            world.vab.stages[0].boosters = 6;
            world.vab.stages[0].booster = 1; // SRB-Heavy
            world.setup_reentry(1); // pitched-over entry
            if let Some(rk) = world.launch.as_mut() {
                rk.roll = 1.1; // ~63 deg roll about the nose axis
                rk.place_attitude();
            }
            0.0
        }
        "reentry_tilt" => {
            // Same fireball but with a steeply pitched-over airframe and a big
            // angle of attack (velocity well off the body axis): verifies the
            // wake hugs the rocket's own axis instead of skewing off downwind.
            world.setup_reentry(1);
            0.0
        }
        "reentry_side" => {
            // belly-first: nose vertical, velocity purely horizontal (~90 deg
            // angle of attack), so the windward side is the flank and the wake
            // trails straight back along the airstream.
            world.setup_reentry(2);
            0.0
        }
        "parachute" => {
            // a crew capsule descending under a deployed recovery parachute.
            world.setup_parachute();
            0.0
        }
        "poweredland" => {
            // a compact stage doing a powered (suicide-burn) descent to landing.
            world.setup_powered_descent();
            0.0
        }
        "sepfloat" => {
            // a couple of seconds after staging, zoomed in, so the spent booster
            // is visibly floating clear just below the climbing upper stage as the
            // gap opens (soft push + slow tumble).
            world.view = View::Rocket;
            world.rocket_az = 3.4;
            world.rocket_el = 0.16;
            world.rocket_dist = 70.0;
            world.rocket_focus_y = -14.0; // look down toward the drifting booster
            world.ignite_launch();
            fly_to_staging(&mut world);
            for _ in 0..25 {
                world.advance(0.1); // ~2.5 s of drift after separation
            }
            0.0
        }
        "upperflame" => {
            // close-up of the upper stage firing, to check the flame sits at its
            // own nozzle (not the dropped booster's).
            world.view = View::Rocket;
            world.rocket_az = 4.2;
            world.rocket_el = 0.08;
            world.ignite_launch();
            fly_to_staging(&mut world);
            fly(&mut world, 4.0);
            world.rocket_dist = 42.0;
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
        "deploy" => {
            // fly a full mission to orbit, then frame the deployed satellite +
            // its orbit ring around the home world on the map.
            world.ignite_launch();
            fly_to_orbit(&mut world);
            frame_map(&mut world);
            world.sys_dist = 22.0;
            world.sys_el = 0.45;
            6.0
        }
        "constellation" => {
            // several back-to-back missions: each successful flight leaves another
            // payload in orbit, so they accumulate around the home world.
            for p in 0..4usize {
                world.vab.payload = p % build::PAYLOADS.len();
                world.ignite_launch();
                fly_to_orbit(&mut world);
                world.reset_launch();
            }
            frame_map(&mut world);
            world.sys_dist = 22.0;
            world.sys_el = 0.45;
            6.0
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
        "citylights" => {
            // the home world's night side, filling the frame, to show the
            // detailed city lights on the dark hemisphere.
            frame_map(&mut world);
            world.sys_dist = 14.0;
            world.sys_az = 4.3;
            world.sys_el = 0.12;
            6.0
        }
        "homeworld" => {
            // the home planet from orbit on its day side: drifting cloud shells,
            // ocean sun-glint, and a terminator sliver showing the city lights.
            frame_map(&mut world);
            world.sys_dist = 17.0;
            world.sys_az = 1.05;
            world.sys_el = 0.26;
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
        "binary" => {
            // frame both stars of the binary so both coronas are visible together
            world.view = View::Map;
            world.sys_focus = DVec3::ZERO; // barycentre
            world.sys_dist = 17000.0;
            world.sys_az = 1.57;
            world.sys_el = 0.16;
            6.0
        }
        "starb" => {
            // close-up of the red companion star to show its ruddy corona.
            world.view = View::Map;
            let bi = world
                .universe
                .bodies
                .iter()
                .position(|b| b.kind == Kind::StarB)
                .unwrap_or(1);
            world.set_focus(bi);
            world.sys_dist = world.universe.bodies[bi].radius * 9.0;
            world.sys_az = 1.0;
            world.sys_el = 0.10;
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
            craft.aim(Mode::Retrograde, &world.body); // already pointed retro
            world.flight = Some(craft);
            6.0
        }
        "node" => {
            // a craft in low orbit with a planned prograde burn that raises the
            // apoapsis - the maneuver planner: green current orbit, cyan result.
            frame_map(&mut world);
            world.launched = true;
            world.sys_dist = 22.0;
            world.sys_el = 0.5;
            world.clock = world.mission.meco_t + 10.0;
            let (r, v) = world.mission.orbit_state_at(world.clock);
            world.flight = Some(Craft::maneuvering(r, v));
            world.node = Some(ManeuverNode { nu: std::f64::consts::PI, pro: 1600.0, nrm: 0.0, rad: 0.0 });
            6.0
        }
        // ---- Lunar-lander mission, stage by stage ----
        "m1_vab" => {
            // the assembled rocket in the VAB, carrying the Lunar Lander payload.
            world.vab.payload = 4; // Lunar Lander
            world.rebuild_vehicle();
            world.view = View::Rocket;
            world.vab_mode = true;
            world.rollout = 0.0;
            world.rocket_az = 0.7;
            world.rocket_el = 0.16;
            world.rocket_dist = 56.0;
            0.0
        }
        "m2_liftoff" => {
            // lift off from the home world with the lander folded inside the fairing.
            world.vab.payload = 4;
            world.rebuild_vehicle();
            world.view = View::Rocket;
            world.rocket_az = 4.97;
            world.rocket_el = 0.10;
            world.rocket_dist = 70.0;
            world.ignite_launch();
            fly(&mut world, 6.0);
            0.0
        }
        "m3_orbit" => {
            // parking orbit around the home world with a trans-lunar injection
            // burn planned at the node (green current orbit, cyan result).
            world.vab.payload = 4;
            frame_map(&mut world);
            world.launched = true;
            world.sys_dist = 30.0;
            world.sys_el = 0.5;
            world.clock = world.mission.meco_t + 10.0;
            let (r, v) = world.mission.orbit_state_at(world.clock);
            world.flight = Some(Craft::maneuvering(r, v));
            world.node = Some(ManeuverNode { nu: std::f64::consts::PI, pro: 3050.0, nrm: 0.0, rad: 0.0 });
            6.0
        }
        "m4_transfer" => {
            // coasting along the trans-lunar transfer ellipse: the conic reaches
            // out to the moon. Framed to show home, the transfer, and the moon.
            world.view = View::Map;
            let hi = world
                .nav
                .iter()
                .position(|&i| world.universe.bodies[i].is_home)
                .unwrap_or(0);
            world.set_focus(hi);
            world.launched = true;
            // a transfer ellipse: periapsis just above home, apoapsis at the moon.
            let mu = world.body.mu;
            let rp = world.body.radius + 250_000.0;
            let moon_dir = world.moon_center_m.normalize();
            let ra = world.moon_center_m.length();
            let a = 0.5 * (rp + ra);
            let vp = (mu * (2.0 / rp - 1.0 / a)).sqrt();
            // periapsis on the opposite side, velocity perpendicular toward the moon
            let r0 = -moon_dir * rp;
            let tangent = DVec3::new(-moon_dir.z, 0.0, moon_dir.x).normalize();
            let v0 = tangent * vp;
            let mut craft = Craft::maneuvering(r0, v0);
            // coast partway out along the transfer so the craft sits mid-flight.
            for _ in 0..4000 {
                craft.integrate(&world.body, &world.grav_bodies(), 5.0);
                if craft.r.length() > ra * 0.55 {
                    break;
                }
            }
            world.flight = Some(craft);
            world.sys_az = 1.4;
            world.sys_el = 0.55;
            world.sys_dist = 240.0;
            6.0
        }
        "m5_approach" => {
            // high on final approach: the lander hangs over a cratered regolith
            // field, descent engine lit, craters running out to the horizon.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.show_lander = true;
            world.lander_alt = 700.0;
            world.lander_firing = true;
            world.rocket_az = 5.5;
            world.rocket_el = 0.62; // look down over the cratered field
            world.rocket_dist = 170.0;
            world.rocket_focus_y = 700.0;
            0.0
        }
        "m5_descent" => {
            // powered descent over the lunar surface: grey regolith, black airless
            // sky, the lander firing its descent engine just above the ground.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.show_lander = true;
            world.lander_alt = 18.0;
            world.lander_firing = true;
            world.lander_rcs = 0.9; // attitude jets trimming the descent
            world.rocket_az = 5.4;
            world.rocket_el = 0.10;
            world.rocket_dist = 38.0;
            world.rocket_focus_y = 10.0;
            0.0
        }
        "rcsdemo" => {
            // close-up of the RCS attitude jets firing around the lander's upper
            // body (the blue-white cold-gas puffs), on the lunar surface.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.show_lander = true;
            world.lander_alt = 0.0;
            world.lander_firing = false;
            world.lander_rcs = 1.0;
            world.rocket_az = 5.5;
            world.rocket_el = 0.22;
            world.rocket_dist = 13.0;
            world.rocket_focus_y = 4.0;
            0.0
        }
        "botland" => {
            // the autonomous moon-landing bot, flown from low lunar orbit down
            // to the surface, shown at a mid-descent instant with its descent
            // engine + RCS attitude jets firing. The lander's height comes from
            // the bot's actual moon-relative altitude.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.show_lander = true;
            // fly the bot until it is on short final, then read its altitude.
            let moon = world.grav_bodies()[0];
            let r0 = moon.center + DVec3::X * (moon.radius + 30_000.0);
            let vc = (moon.mu / (moon.radius + 30_000.0)).sqrt();
            let mut craft = Craft::maneuvering(r0, DVec3::Z * vc);
            let mut moonbot = bot::MoonBot::new();
            // step until the bot is low (short final), capped so the shot is reproducible
            let dt = 0.1;
            for _ in 0..20_000 {
                moonbot.control(&mut craft, &moon);
                craft.integrate(&world.body, std::slice::from_ref(&moon), dt);
                if bot::MoonBot::altitude(&craft, &moon) < 30.0 || craft.landed || craft.crashed {
                    break;
                }
            }
            let alt = bot::MoonBot::altitude(&craft, &moon).max(0.0) as f32;
            world.lander_alt = alt;
            world.lander_firing = craft.throttle > 0.02;
            // RCS activity from the bot's actual attitude-control torque (floored
            // so the short still clearly shows the jets).
            let rcs_act = (craft.torque_report.rcs.length() / 1500.0) as f32;
            world.lander_rcs = rcs_act.clamp(0.4, 1.0);
            world.rocket_az = 5.4;
            world.rocket_el = 0.12;
            world.rocket_dist = 40.0;
            world.rocket_focus_y = (alt * 0.5 + 4.0).min(20.0);
            0.0
        }
        "m6_landed" => {
            // touched down on the moon: the lander standing on grey regolith under
            // a black, star-flecked sky.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.show_lander = true;
            world.lander_alt = 0.0;
            world.lander_firing = false;
            world.rocket_az = 5.4;
            world.rocket_el = 0.13;
            world.rocket_dist = 22.0;
            world.rocket_focus_y = 3.0;
            0.0
        }
        "cargo" => {
            // a rocket on the pad with the fairing clamshell open, revealing the
            // refinery cargo module packed inside.
            world.vab.payload = 5; // Refinery Module
            world.rebuild_vehicle();
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.fairing_open = 0.55; // clamshell cracked open
            let top = world.rocket_body.height;
            world.rocket_az = 5.05;
            world.rocket_el = 0.05;
            world.rocket_focus_y = top - 7.5;
            world.rocket_dist = 15.0;
            0.0
        }
        "crewcap" => {
            // a rocket on the pad with the fairing open, revealing the crew capsule.
            world.vab.payload = 10; // Crew Capsule
            world.rebuild_vehicle();
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.fairing_open = 0.6;
            let top = world.rocket_body.height;
            world.rocket_az = 5.05;
            world.rocket_el = 0.06;
            world.rocket_focus_y = top - 7.0;
            world.rocket_dist = 16.0;
            0.0
        }
        "servicemod" => {
            // a rocket on the pad with the fairing open, revealing the service module.
            world.vab.payload = 11; // Service Module
            world.rebuild_vehicle();
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.fairing_open = 0.6;
            let top = world.rocket_body.height;
            world.rocket_az = 5.05;
            world.rocket_el = 0.06;
            world.rocket_focus_y = top - 7.0;
            world.rocket_dist = 16.0;
            0.0
        }
        "cargoparts" => {
            // the fairing-packed module catalog, unpacked in a row on the moon.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.base_mesh = Some(rocket::cargo_catalog());
            world.rocket_az = 4.71;
            world.rocket_el = 0.14;
            world.rocket_dist = 34.0;
            world.rocket_focus_y = 2.6;
            0.0
        }
        "delivery" => {
            // a delivered cargo module standing on the lunar surface, ready to be
            // unfolded and assembled on site.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.base_mesh = Some(rocket::cargo_module(0)); // refinery
            world.rocket_az = 5.4;
            world.rocket_el = 0.12;
            world.rocket_dist = 15.0;
            world.rocket_focus_y = 2.6;
            0.0
        }
        "ast_orbit" => {
            // a large asteroid seen from orbit, lit against a starfield.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true; // airless lighting
            world.space = true; // no planet, starfield sky
            world.base_mesh = Some(rocket::asteroid_preset(0));
            world.space_label = rocket::ASTEROID_NAMES[0];
            world.rocket_az = 0.55;
            world.rocket_el = 0.33;
            world.rocket_dist = 1350.0;
            world.rocket_focus_y = 0.0;
            0.0
        }
        "ast_orbit2" => {
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.space = true;
            world.base_mesh = Some(rocket::asteroid_preset(2)); // elongated peanut
            world.space_label = rocket::ASTEROID_NAMES[2];
            world.rocket_az = 0.55;
            world.rocket_el = 0.33;
            world.rocket_dist = 1050.0;
            world.rocket_focus_y = 0.0;
            0.0
        }
        "ast_orbit3" => {
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.space = true;
            world.base_mesh = Some(rocket::asteroid_preset(3)); // long shard
            world.space_label = rocket::ASTEROID_NAMES[3];
            world.rocket_az = 0.5;
            world.rocket_el = 0.34;
            world.rocket_dist = 1300.0;
            world.rocket_focus_y = 0.0;
            0.0
        }
        "ast_craters" => {
            // a closer look down over the heavily-cratered surface so the
            // rimmed impact craters of many sizes read clearly.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.space = true;
            world.base_mesh = Some(rocket::asteroid_preset(1)); // most cratered
            world.space_label = rocket::ASTEROID_NAMES[1];
            world.rocket_az = 0.45;
            world.rocket_el = 0.32;
            world.rocket_dist = 1080.0;
            world.rocket_focus_y = 0.0;
            0.0
        }
        "ast_surf" => {
            // down near the surface of a large asteroid: rubble horizon + space.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.space = true;
            world.base_mesh = Some(rocket::asteroid_preset(1)); // squat, cratered
            world.space_label = rocket::ASTEROID_NAMES[1];
            world.rocket_az = 0.9;
            world.rocket_el = 0.07;
            world.rocket_dist = 545.0;
            world.rocket_focus_y = 250.0;
            0.0
        }
        "ast_surf2" => {
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.space = true;
            world.base_mesh = Some(rocket::asteroid_preset(2));
            world.space_label = rocket::ASTEROID_NAMES[2];
            world.rocket_az = 0.6;
            world.rocket_el = 0.08;
            world.rocket_dist = 430.0;
            world.rocket_focus_y = 170.0;
            0.0
        }
        // ---- Asteroid landing sequence on an LOD body (detail refines as you
        // approach): distance -> orbit -> descent -> touchdown. ----
        "ld_far" | "ld_orbit" | "ld_descent" | "ld_land" => {
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true; // airless lighting
            world.space = true; // starfield sky, no planet
            world.ref_local = DVec3::ZERO; // body at the local origin
            let r = 380.0f64;
            let elev = terrain::Elevation::asteroid(8, r * 0.34, 26);
            let pole = (r + elev.land_height_m(DVec3::Y)) as f32; // surface at +Y
            world.ast_elev = Some(elev);
            world.ast_radius = r;
            world.space_label = rocket::ASTEROID_NAMES[1];
            match scenario {
                "ld_far" => {
                    // a distant speck: the whole body in frame, coarse LOD.
                    world.rocket_az = 0.5;
                    world.rocket_el = 0.26;
                    world.rocket_dist = 3800.0;
                    world.rocket_focus_y = 0.0;
                }
                "ld_orbit" => {
                    // close orbit: the body fills the view, mid LOD refining.
                    world.rocket_az = 0.5;
                    world.rocket_el = 0.30;
                    world.rocket_dist = 1150.0;
                    world.rocket_focus_y = 0.0;
                }
                "ld_descent" => {
                    // powered descent toward the +Y pole, looking down; fine LOD
                    // resolves craters under the hovering lander.
                    world.show_lander = true;
                    world.lander_alt = pole + 75.0;
                    world.lander_firing = true;
                    world.lander_rcs = 0.8;
                    world.rocket_az = 0.6;
                    world.rocket_el = 0.85; // looking down over the pole
                    world.rocket_dist = 120.0;
                    world.rocket_focus_y = pole + 60.0;
                }
                _ => {
                    // touchdown: lander on the regolith, finest LOD, craters close.
                    world.show_lander = true;
                    world.lander_alt = pole + 1.5;
                    world.lander_firing = false;
                    world.rocket_az = 0.6;
                    world.rocket_el = 0.6;
                    world.rocket_dist = 34.0;
                    world.rocket_focus_y = pole + 3.0;
                }
            }
            0.0
        }
        "moonbase" => {
            // an assembled moon base on the cratered surface: HQ, mining,
            // reactor, lunar VAB, printer, tourist dome, spaceport, hotel and
            // refueling station around a central habitat plaza.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.base_mesh = Some(rocket::moon_base());
            world.base_panel = true;
            world.rocket_az = 5.5;
            world.rocket_el = 0.42; // look down over the colony
            world.rocket_dist = 165.0;
            world.rocket_focus_y = 6.0;
            0.0
        }
        "basetour" => {
            // a ground-level view across the colony, close enough to read the
            // building detail (plaza dome, hotel, refuel tanks, reactor beyond).
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.base_mesh = Some(rocket::moon_base());
            world.base_panel = true;
            world.rocket_az = 4.71;
            world.rocket_el = 0.12;
            world.rocket_dist = 62.0;
            world.rocket_focus_y = 6.0;
            0.0
        }
        "baseparts" => {
            // the parts catalog: every structure lined up in a row.
            world.view = View::Rocket;
            world.vab_mode = false;
            world.rollout = 1.0;
            world.lunar = true;
            world.base_mesh = Some(rocket::base_catalog());
            world.base_panel = true;
            world.rocket_az = 4.45; // 3/4 view along the row
            world.rocket_el = 0.20;
            world.rocket_dist = 150.0;
            world.rocket_focus_y = 9.0;
            0.0
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

    let (n, hn, dyn_n, fx_n, plasma_on, plume_on) = gpu.prepare(queue, &mut world, width, height, time);
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
        let plasma_mesh_n = world.plasma_mesh_n;
        {
            let mut pass = mesh_pass(&mut encoder, &target_view, &depth);
            gpu.draw_sky(&mut pass);
            gpu.draw_meshes(&mut pass, terrain_n, dyn_n);
            gpu.draw_plume(&mut pass, plume_on);
            if plasma_on {
                gpu.draw_plasma_mesh(&mut pass, plasma_mesh_n);
            }
            // FX last, so additive sparks/embers sit on top of the plasma glow.
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

/// Render a horizontal "filmstrip" animation of a rocket-view scenario: render
/// `frames` frames at `fw`x`fh`, advancing the sim by `dt` between each (so the
/// vehicle falls and the volumetric FX boil), and tile them left-to-right into a
/// single PNG. No GIF dependency needed.
#[cfg(not(target_arch = "wasm32"))]
fn render_anim(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &Gpu,
    scenario: &str,
    path: &str,
    fw: u32,
    fh: u32,
    frames: u32,
    dt: f32,
) {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (mut world, _time) = setup_world(scenario, fw, fh);

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("anim-target"),
        size: wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = create_depth(device, fw, fh);

    let unpadded = fw * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("anim-readback"),
        size: (padded * fh) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut strip = image::RgbaImage::new(fw * frames, fh);
    for f in 0..frames {
        let anim = world.anim;
        let (n, hn, dyn_n, fx_n, plasma_on, plume_on) =
            gpu.prepare(queue, &mut world, fw, fh, anim);
        let terrain_n = world.terrain_count;
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("anim-enc") });
        {
            let mut pass = mesh_pass(&mut enc, &target_view, &depth);
            gpu.draw_sky(&mut pass);
            gpu.draw_meshes(&mut pass, terrain_n, dyn_n);
            gpu.draw_plume(&mut pass, plume_on);
            gpu.draw_fx(&mut pass, fx_n);
            if plasma_on {
                gpu.draw_plasma_mesh(&mut pass, world.plasma_mesh_n);
            }
        }
        {
            let mut pass = overlay_pass(&mut enc, &target_view);
            gpu.draw_overlay(&mut pass, n, hn);
        }
        enc.copy_texture_to_buffer(
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
                    rows_per_image: Some(fh),
                },
            },
            wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        {
            let data = slice.get_mapped_range();
            for row in 0..fh {
                let start = (row * padded) as usize;
                for col in 0..fw {
                    let i = start + (col * 4) as usize;
                    strip.put_pixel(f * fw + col, row, image::Rgba([data[i], data[i + 1], data[i + 2], 255]));
                }
            }
        }
        readback.unmap();
        world.advance(dt); // fall + boil for the next frame
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    strip.save(path).expect("write png");
    println!("wrote {path} ({}x{fh}, {frames} frames)", fw * frames);
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
                WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. }
            )
        {
            return;
        }
        // For keys, only let egui win when it actually wants text input (a
        // focused text field). Otherwise the game keeps its shortcuts - notably
        // Tab, which egui's focus navigation would otherwise swallow before it
        // ever reaches `toggle_view()`.
        if matches!(event, WindowEvent::KeyboardInput { .. })
            && state.egui_ctx.wants_keyboard_input()
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
                    let w = state.config.width as f32;
                    let h = state.config.height.max(1) as f32;
                    let aspect = w / h;
                    let (cx, cy) = (state.last_cursor.0 as f32, state.last_cursor.1 as f32);
                    let ndc = [cx / w * 2.0 - 1.0, 1.0 - cy / h * 2.0];
                    if pressed {
                        // map: click a body to focus it, but only when zoomed out
                        // to system scale - up close (framing the vehicle/home)
                        // a stray click must not snap focus to a random body.
                        if state.world.view == View::Map && state.world.in_system_view() {
                            if let Some(b) = state.world.pick_body((w, h), cx, cy) {
                                state.world.set_focus(b);
                                return;
                            }
                        }
                        // VAB: grab a 3D part off the rack instead of orbiting
                        if state.world.view == View::Rocket
                            && state.world.vab_mode
                            && state.world.launch.is_none()
                        {
                            if let Some(slot) = state.world.pick_rack(ndc, aspect) {
                                state.world.grab = Some(slot);
                                state.world.update_grab(ndc, aspect);
                                state.dragging = false;
                                return;
                            }
                        }
                        state.dragging = true;
                    } else {
                        // release: drop the grabbed part onto the hovered slot
                        if state.world.grab.is_some() {
                            state.world.drop_grab();
                        }
                        state.dragging = false;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                if state.world.grab.is_some() {
                    let w = state.config.width as f32;
                    let h = state.config.height.max(1) as f32;
                    let ndc = [x as f32 / w * 2.0 - 1.0, 1.0 - y as f32 / h * 2.0];
                    state.world.update_grab(ndc, w / h);
                    state.window.request_redraw();
                } else if state.dragging {
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
                // Car driving / on foot: held WASD/arrows steer the car or move the
                // character. Handle press AND release here (so keys can be held)
                // and skip the other controls.
                if state.world.driving || state.world.walking {
                    let kc = match key_event.physical_key {
                        PhysicalKey::Code(c) => c,
                        _ => return,
                    };
                    let down = key_event.state == ElementState::Pressed;
                    let inp = if state.world.driving {
                        &mut state.world.drive_in
                    } else {
                        &mut state.world.walk_in
                    };
                    match kc {
                        KeyCode::KeyW | KeyCode::ArrowUp => inp[0] = down,
                        KeyCode::KeyS | KeyCode::ArrowDown => inp[1] = down,
                        KeyCode::KeyA | KeyCode::ArrowLeft => inp[2] = down,
                        KeyCode::KeyD | KeyCode::ArrowRight => inp[3] = down,
                        _ => {}
                    }
                    state.window.request_redraw();
                    return;
                }
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
                    // Z full / X cut). In the frozen re-entry test, the same WASDQE
                    // freely rotate the airframe in 3 axes (pitch/yaw/roll) so you
                    // can inspect re-entry from any orientation. These repeat held.
                    let test = state.world.reentry_test;
                    if let Some(rk) = state.world.launch.as_mut() {
                        let step = 2f64.to_radians();
                        if test {
                            match code {
                                KeyCode::KeyW | KeyCode::ArrowUp => rk.pitch += step,
                                KeyCode::KeyS | KeyCode::ArrowDown => rk.pitch -= step,
                                KeyCode::KeyA | KeyCode::ArrowLeft => rk.yaw += step,
                                KeyCode::KeyD | KeyCode::ArrowRight => rk.yaw -= step,
                                KeyCode::KeyQ => rk.roll += step,
                                KeyCode::KeyE => rk.roll -= step,
                                _ => {}
                            }
                        } else {
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
                        KeyCode::KeyB => state.world.toggle_moonbot(),
                        KeyCode::Space => {
                            if state.world.view == View::Rocket {
                                if state.world.vab_mode {
                                    state.world.start_rollout(); // roll out to the pad
                                } else if state.world.launch.is_none() {
                                    state.world.ignite_launch();
                                } else {
                                    state.world.stage_launch();
                                }
                            } else if state.world.flight.is_none() {
                                state.world.toggle_launch();
                            }
                        }
                        KeyCode::KeyR if state.world.view == View::Rocket => {
                            state.world.back_to_vab()
                        }
                        KeyCode::KeyL => state.world.toggle_lod_debug(),
                        // [ ] are always time compression (sim time scale).
                        KeyCode::BracketRight => {
                            state.world.warp = (state.world.warp * 2.0).min(10000.0);
                        }
                        KeyCode::BracketLeft => {
                            state.world.warp = (state.world.warp * 0.5).max(1.0);
                        }
                        // , . are the separate crawler-speed control, only while
                        // rolling out to the pad (they do not touch sim time).
                        KeyCode::Period if state.world.rolling_out => {
                            state.world.bump_rollout_speed(true);
                        }
                        KeyCode::Comma if state.world.rolling_out => {
                            state.world.bump_rollout_speed(false);
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
                                // 1..6 select the six orbital attitudes; 7 frees the autopilot.
                                c.mode = match idx {
                                    0 => Mode::Prograde,
                                    1 => Mode::Retrograde,
                                    2 => Mode::Normal,
                                    3 => Mode::AntiNormal,
                                    4 => Mode::RadialOut,
                                    5 => Mode::RadialIn,
                                    6 => Mode::Free,
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
        // `ascentcsv`: fly the default vehicle to orbit (open-loop gravity turn)
        // and print the ascent profile as CSV to stdout, for plotting.
        if args.iter().any(|a| a == "ascentcsv") {
            use sim::orbit::orbit_from_state;
            let body = CentralBody::home();
            let veh = build::Vab::default_build().to_vehicle();
            let up = DVec3::new(1.0, 0.0, 0.0);
            let heading = DVec3::Y.cross(up).normalize();
            let mut rk = launch::Rocket::on_pad(&veh, up * body.radius, DVec3::ZERO, up, heading);
            rk.throttle = 1.0;
            let dt = 0.2f64;
            let mut prev = 0.0f64;
            let pad_r = body.radius;
            println!("t,alt_km,speed,vspeed,net_a,crew_g,twr,stage,apo_km,peri_km");
            let mut t = 0.0f64;
            for _ in 0..4000 {
                // open-loop gravity turn: hold vertical ~12 s then ease over.
                rk.pitch = (((rk.met - 11.0) / 145.0).clamp(0.0, 1.0) * 80f64.to_radians()).min(1.45);
                if rk.prop_frac() <= 0.0 && rk.stages.len() > 1 {
                    rk.jettison();
                }
                rk.integrate(&body, dt);
                t += dt;
                let sp = rk.speed();
                let net_a = (sp - prev) / dt;
                prev = sp;
                let up_r = rk.r.normalize_or_zero();
                let vspeed = rk.v.dot(up_r);
                let g_local = body.mu / (rk.r.length() * rk.r.length());
                let twr = rk.live_thrust() / (rk.mass() * g_local);
                let orb = orbit_from_state(rk.r, rk.v, body.mu);
                let (apo, peri) = if orb.e < 1.0 && orb.a > 0.0 {
                    ((orb.ra - pad_r) / 1000.0, (orb.rp - pad_r) / 1000.0)
                } else {
                    ((orb.ra - pad_r) / 1000.0, f64::NAN)
                };
                println!(
                    "{:.2},{:.4},{:.2},{:.2},{:.3},{:.3},{:.3},{},{:.1},{:.1}",
                    t,
                    rk.altitude(&body) / 1000.0,
                    sp,
                    vspeed,
                    net_a,
                    rk.crew_g(),
                    twr,
                    rk.stage_base,
                    apo,
                    peri
                );
                if rk.crashed {
                    break;
                }
                if rk.orbit_reached && t > rk.met {
                    break;
                }
            }
            return;
        }
        // `bench [scenario] [WxH]`: time the re-entry plasma glow-mesh pass on the
        // real GPU (loop the actual pipeline in one submit, GPU-synced), so its
        // cost is a measured number instead of a guess.
        if args.iter().any(|a| a == "bench") {
            env_logger::init();
            let scenario = args
                .iter()
                .position(|a| a == "bench")
                .and_then(|i| args.get(i + 1))
                .filter(|a| a.starts_with("reentry"))
                .cloned()
                .unwrap_or_else(|| "reentry".to_string());
            let (w, h) = args
                .iter()
                .find_map(|a| {
                    let (ws, hs) = a.split_once('x')?;
                    Some((ws.parse::<u32>().ok()?, hs.parse::<u32>().ok()?))
                })
                .unwrap_or((1280, 800));
            bench_plasma(&scenario, w, h);
            return;
        }
        // `anim <scenario> [out.png]`: a horizontal filmstrip of the scenario
        // playing out (the rocket falling + the volumetric FX boiling).
        if args.iter().any(|a| a == "anim") {
            env_logger::init();
            let scenario = args
                .iter()
                .position(|a| a == "anim")
                .and_then(|i| args.get(i + 1))
                .filter(|a| !a.ends_with(".png"))
                .cloned()
                .unwrap_or_else(|| "reentry".to_string());
            let path = args
                .iter()
                .find(|a| a.ends_with(".png"))
                .cloned()
                .unwrap_or_else(|| format!("out/{scenario}_anim.png"));
            let (device, queue) = make_shot_device();
            let gpu = Gpu::new(&device, &queue, wgpu::TextureFormat::Rgba8UnormSrgb);
            render_anim(&device, &queue, &gpu, &scenario, &path, 420, 320, 6, 0.12);
            return;
        }
        if args.iter().any(|a| a == "shot" || a == "--shot") {
            env_logger::init();
            if args.iter().any(|a| a == "all") {
                screenshot_all(1280, 800);
                return;
            }
            let scenario = if args.iter().any(|a| a == "cities") {
                "cities"
            } else if args.iter().any(|a| a == "citylife") {
                "citylife"
            } else if args.iter().any(|a| a == "cityview") {
                "cityview"
            } else if args.iter().any(|a| a == "walk") {
                "walk"
            } else if args.iter().any(|a| a == "drive") {
                "drive"
            } else if args.iter().any(|a| a == "m1_vab") {
                "m1_vab"
            } else if args.iter().any(|a| a == "m2_liftoff") {
                "m2_liftoff"
            } else if args.iter().any(|a| a == "m3_orbit") {
                "m3_orbit"
            } else if args.iter().any(|a| a == "m4_transfer") {
                "m4_transfer"
            } else if args.iter().any(|a| a == "m5_approach") {
                "m5_approach"
            } else if args.iter().any(|a| a == "m5_descent") {
                "m5_descent"
            } else if args.iter().any(|a| a == "m6_landed") {
                "m6_landed"
            } else if args.iter().any(|a| a == "botland") {
                "botland"
            } else if args.iter().any(|a| a == "rcsdemo") {
                "rcsdemo"
            } else if args.iter().any(|a| a == "ast_orbit2") {
                "ast_orbit2"
            } else if args.iter().any(|a| a == "ast_orbit3") {
                "ast_orbit3"
            } else if args.iter().any(|a| a == "ast_orbit") {
                "ast_orbit"
            } else if args.iter().any(|a| a == "ld_far") {
                "ld_far"
            } else if args.iter().any(|a| a == "ld_orbit") {
                "ld_orbit"
            } else if args.iter().any(|a| a == "ld_descent") {
                "ld_descent"
            } else if args.iter().any(|a| a == "ld_land") {
                "ld_land"
            } else if args.iter().any(|a| a == "ast_craters") {
                "ast_craters"
            } else if args.iter().any(|a| a == "ast_surf2") {
                "ast_surf2"
            } else if args.iter().any(|a| a == "ast_surf") {
                "ast_surf"
            } else if args.iter().any(|a| a == "cargoparts") {
                "cargoparts"
            } else if args.iter().any(|a| a == "crewcap") {
                "crewcap"
            } else if args.iter().any(|a| a == "servicemod") {
                "servicemod"
            } else if args.iter().any(|a| a == "cargo") {
                "cargo"
            } else if args.iter().any(|a| a == "delivery") {
                "delivery"
            } else if args.iter().any(|a| a == "moonbase") {
                "moonbase"
            } else if args.iter().any(|a| a == "basetour") {
                "basetour"
            } else if args.iter().any(|a| a == "baseparts") {
                "baseparts"
            } else if args.iter().any(|a| a == "moons") {
                "moons"
            } else if args.iter().any(|a| a == "citylights") {
                "citylights"
            } else if args.iter().any(|a| a == "homeworld") {
                "homeworld"
            } else if args.iter().any(|a| a == "overlook") {
                "overlook"
            } else if args.iter().any(|a| a == "descent") {
                "descent"
            } else if args.iter().any(|a| a == "cityrise") {
                "cityrise"
            } else if args.iter().any(|a| a == "nightcity") {
                "nightcity"
            } else if args.iter().any(|a| a == "moon") {
                "moon"
            } else if args.iter().any(|a| a == "rocket") {
                "rocket"
            } else if args.iter().any(|a| a == "starb") {
                "starb"
            } else if args.iter().any(|a| a == "binary") {
                "binary"
            } else if args.iter().any(|a| a == "system") {
                "system"
            } else if args.iter().any(|a| a == "grabdemo") {
                "grabdemo"
            } else if args.iter().any(|a| a == "lander") {
                "lander"
            } else if args.iter().any(|a| a == "vab") {
                "vab"
            } else if args.iter().any(|a| a == "rollout") {
                "rollout"
            } else if args.iter().any(|a| a == "pad") {
                "pad"
            } else if args.iter().any(|a| a == "loddebugmap") {
                "loddebugmap"
            } else if args.iter().any(|a| a == "loddebug") {
                "loddebug"
            } else if args.iter().any(|a| a == "liftoff2") {
                "liftoff2"
            } else if args.iter().any(|a| a == "liftoff") {
                "liftoff"
            } else if args.iter().any(|a| a == "boosterlaunch") {
                "boosterlaunch"
            } else if args.iter().any(|a| a == "boosters") {
                "boosters"
            } else if args.iter().any(|a| a == "crash") {
                "crash"
            } else if args.iter().any(|a| a == "reentry_tilt") {
                "reentry_tilt"
            } else if args.iter().any(|a| a == "reentry_side") {
                "reentry_side"
            } else if args.iter().any(|a| a == "reentry_roll") {
                "reentry_roll"
            } else if args.iter().any(|a| a == "reentry_boost") {
                "reentry_boost"
            } else if args.iter().any(|a| a == "plumetest") {
                "plumetest"
            } else if args.iter().any(|a| a == "reentry_lowheat") {
                "reentry_lowheat"
            } else if args.iter().any(|a| a == "reentry_midheat") {
                "reentry_midheat"
            } else if args.iter().any(|a| a == "reentry") {
                "reentry"
            } else if args.iter().any(|a| a == "parachute") {
                "parachute"
            } else if args.iter().any(|a| a == "poweredland") {
                "poweredland"
            } else if args.iter().any(|a| a == "sepfloat") {
                "sepfloat"
            } else if args.iter().any(|a| a == "staging") {
                "staging"
            } else if args.iter().any(|a| a == "launchmap") {
                "launchmap"
            } else if args.iter().any(|a| a == "upperflame") {
                "upperflame"
            } else if args.iter().any(|a| a == "constellation") {
                "constellation"
            } else if args.iter().any(|a| a == "deploy") {
                "deploy"
            } else if args.iter().any(|a| a == "orbit") {
                "orbit"
            } else if args.iter().any(|a| a == "ascent") {
                "ascent"
            } else if args.iter().any(|a| a == "node") {
                "node"
            } else if args.iter().any(|a| a == "flight") {
                "flight"
            } else {
                "surface"
            };
            let default = match scenario {
                "m1_vab" => "out/m1_vab.png",
                "m2_liftoff" => "out/m2_liftoff.png",
                "m3_orbit" => "out/m3_orbit.png",
                "m4_transfer" => "out/m4_transfer.png",
                "m5_approach" => "out/m5_approach.png",
                "m5_descent" => "out/m5_descent.png",
                "m6_landed" => "out/m6_landed.png",
                "botland" => "out/botland.png",
                "rcsdemo" => "out/rcsdemo.png",
                "ast_orbit" => "out/ast_orbit.png",
                "ast_orbit2" => "out/ast_orbit2.png",
                "ast_orbit3" => "out/ast_orbit3.png",
                "ast_surf" => "out/ast_surf.png",
                "ast_surf2" => "out/ast_surf2.png",
                "ast_craters" => "out/ast_craters.png",
                "ld_far" => "out/ld_far.png",
                "ld_orbit" => "out/ld_orbit.png",
                "ld_descent" => "out/ld_descent.png",
                "ld_land" => "out/ld_land.png",
                "cargo" => "out/cargo.png",
                "cargoparts" => "out/cargoparts.png",
                "delivery" => "out/delivery.png",
                "moonbase" => "out/moonbase.png",
                "basetour" => "out/basetour.png",
                "baseparts" => "out/baseparts.png",
                "moons" => "out/moons.png",
                "moon" => "out/moon.png",
                "citylights" => "out/citylights.png",
                "homeworld" => "out/homeworld.png",
                "overlook" => "out/overlook.png",
                "descent" => "out/descent.png",
                "nightcity" => "out/nightcity.png",
                "cityrise" => "out/cityrise.png",
                "rocket" => "out/rocket.png",
                "system" => "out/system.png",
                "pad" => "out/pad.png",
                "liftoff" => "out/liftoff.png",
                "liftoff2" => "out/liftoff2.png",
                "loddebug" => "out/loddebug.png",
                "loddebugmap" => "out/loddebugmap.png",
                "staging" => "out/staging.png",
                "sepfloat" => "out/sepfloat.png",
                "reentry" => "out/reentry.png",
                "reentry_tilt" => "out/reentry_tilt.png",
                "reentry_roll" => "out/reentry_roll.png",
                "reentry_boost" => "out/reentry_boost.png",
                "reentry_lowheat" => "out/reentry_lowheat.png",
                "reentry_midheat" => "out/reentry_midheat.png",
                "plumetest" => "out/plumetest.png",
                "crash" => "out/crash.png",
                "boosters" => "out/boosters.png",
                "boosterlaunch" => "out/boosterlaunch.png",
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
            // Optional `WIDTHxHEIGHT` token (e.g. `1280x1200`) to size the shot;
            // handy for verifying tall panels that overflow the default frame.
            let (w, h) = args
                .iter()
                .skip(1)
                .find_map(|a| {
                    let (ws, hs) = a.split_once('x')?;
                    Some((ws.parse::<u32>().ok()?, hs.parse::<u32>().ok()?))
                })
                .unwrap_or((1280, 800));
            screenshot(&path, w, h, scenario);
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

#[cfg(test)]
mod reentry_test_scene {
    use super::*;

    /// The frozen re-entry test scene must hold the vehicle still at full heat
    /// indefinitely: no motion, no heat damage, never destroyed - so the plasma
    /// FX can be inspected at leisure (this is the whole point of the test menu).
    #[test]
    fn frozen_scene_survives_a_long_run_without_moving() {
        let mut w = World::new();
        w.setup_reentry(0);
        assert!(w.reentry_test, "scene should be frozen after setup");
        let (r0, heat0) = {
            let rk = w.launch.as_ref().expect("vehicle present");
            (rk.r, rk.heat)
        };
        assert!(heat0 > 0.3, "plasma should have bloomed (heat={heat0})");

        // Far longer than the few seconds it used to last before burning up.
        for _ in 0..1200 {
            w.advance(0.1); // 120 s at 1x
        }

        let rk = w.launch.as_ref().expect("still present - never destroyed");
        assert!(!rk.destroyed, "frozen test vehicle must never break up");
        assert!(!rk.crashed, "frozen test vehicle must never crash");
        assert_eq!(rk.health, 100.0, "airframe stays pristine");
        assert!(rk.heat > 0.3, "stays hot so the FX keep playing (heat={})", rk.heat);
        assert_eq!(rk.r, r0, "vehicle must not move while frozen");
    }

    /// Re-aiming with the pitch command rotates the airframe instantly while
    /// frozen (so you can inspect the shock from any attitude), without it ever
    /// starting to move or take damage.
    #[test]
    fn frozen_scene_lets_you_re_aim_the_airframe() {
        let mut w = World::new();
        w.setup_reentry(0);
        let r0 = w.launch.as_ref().unwrap().r;
        if let Some(rk) = w.launch.as_mut() {
            rk.pitch = 0.6; // command a new attitude (what W/S does)
        }
        w.advance(0.1);
        let rk = w.launch.as_ref().unwrap();
        assert!((rk.pitch_act - 0.6).abs() < 1e-9, "attitude tracks the command");
        assert_eq!(rk.r, r0, "re-aiming must not move the vehicle");
    }

    /// In the frozen test you can yaw and roll the airframe (not just pitch), so
    /// re-entry can be inspected from any 3-axis orientation. Yaw swings the nose;
    /// roll spins about the nose (nose direction unchanged) - neither moves it.
    #[test]
    fn frozen_scene_yaw_and_roll_rotate_the_airframe() {
        let mut w = World::new();
        w.setup_reentry(0);
        let r0 = w.launch.as_ref().unwrap().r;
        let nose0 = w.launch.as_ref().unwrap().point_dir();

        // Yaw swings the nose off its current direction.
        if let Some(rk) = w.launch.as_mut() {
            rk.yaw = 0.6;
        }
        w.advance(0.1);
        let (nose1, orient1) = {
            let rk = w.launch.as_ref().unwrap();
            (rk.point_dir(), rk.orient)
        };
        assert!(nose0.dot(nose1) < 0.99, "yaw should rotate the nose off-axis");

        // Roll spins about the nose axis: the nose stays put, the orientation turns.
        if let Some(rk) = w.launch.as_mut() {
            rk.roll = 0.8;
        }
        w.advance(0.1);
        let rk = w.launch.as_ref().unwrap();
        assert!(rk.point_dir().dot(nose1) > 0.999, "roll must keep the nose fixed");
        assert!(rk.orient.angle_between(orient1) > 1e-3, "roll must change the orientation");
        assert_eq!(rk.r, r0, "rotating must not move the vehicle");
    }

    /// A real ignition is never frozen - it flies normally.
    #[test]
    fn normal_ignition_is_not_frozen() {
        let mut w = World::new();
        w.setup_reentry(0);
        assert!(w.reentry_test);
        w.ignite_launch(); // a genuine launch clears the test freeze
        assert!(!w.reentry_test);
    }

    /// The rocket must never sink through the pad during the engine spool-up: it
    /// is held down until thrust exceeds weight, then lifts off and climbs.
    #[test]
    fn launch_does_not_sink_into_the_pad() {
        let mut w = World::new();
        w.ignite_launch();
        let pad_alt = w.launch.as_ref().unwrap().altitude(&w.body);
        let mut min_alt = pad_alt;
        for _ in 0..160 {
            w.advance(0.05); // ~8 s through spool-up + early ascent
            if let Some(rk) = w.launch.as_ref() {
                min_alt = min_alt.min(rk.altitude(&w.body));
            }
        }
        assert!(
            min_alt >= pad_alt - 0.5,
            "rocket sank below the pad during launch: min_alt={min_alt:.2} pad={pad_alt:.2}"
        );
        let end_alt = w.launch.as_ref().unwrap().altitude(&w.body);
        assert!(end_alt > pad_alt + 1.0, "rocket failed to lift off: end={end_alt:.2} pad={pad_alt:.2}");
    }
}
