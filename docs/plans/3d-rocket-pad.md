# Plan: 3D staged rocket + landing pad (rocket view)

Status: APPROVED DIRECTION (chosen 2026-06-16). Build in phases; verify each
phase with a headless `shot rocket` PNG (lavapipe) plus `cargo check` native +
wasm. This is the active next task.

## Goal

A real 3D rocket built procedurally from the `sim` vehicle's stage list,
standing on a launch pad at the spaceport, rendered in a new perspective
**rocket view**. Stage sections drop as fuel depletes on launch.

## Architecture

The current Surface/System views are raymarched (no mesh pipeline), so this adds
the engine's first real triangle-mesh pipeline - reused later by the LOD terrain
and everything 3D.

- New mesh pipeline in `Gpu`: vertex buffers (pos/normal/color), index buffer, a
  depth texture (recreated on resize), a perspective MVP uniform, and a
  `rocket.wgsl` shader with sun + ambient lighting.
- New `View::Rocket`: a local self-contained scene (rocket at local origin,
  ground plane at y=0, +Y up) so no planet curvature/terrain is needed yet -
  those plug in later. Tab cycles to/from it; orbit-drag + scroll-zoom frame it.

## Procedural geometry

- `rocket.rs` (new, in `app`): build the stack from `sim::Vehicle` stages - each
  stage a cylinder sized from its propellant volume (bigger stages look bigger),
  an interstage ring, engine nozzles (truncated cones) at each stage base, a
  payload section + nose cone on top, fins on the booster. Per-section colors.
- `pad.rs` (new, in `app`): a slab + launch mount/legs + simple flame trench,
  plus a ground plane.

## Phases (each ends in a verified `shot rocket` PNG)

1. Static stack on the pad - rocket + pad + ground + lighting in rocket view,
   orbit camera, per-stage HUD (name, dry/prop mass, thrust, TWR, Isp, stage
   dv). Core deliverable.
2. Staging animation - on launch (Space) the rocket lifts and each stage
   detaches and falls away as its propellant depletes, driven by the existing
   ascent sim state.
3. Engine plume - a simple additive exhaust cone while thrusting (polish).

## Files

- Add: `crates/app/src/rocket.rs`, `crates/app/src/pad.rs`,
  `crates/app/src/rocket.wgsl`.
- Edit: `crates/app/src/main.rs` (mesh pipeline, depth buffer, `View::Rocket`,
  Tab cycle, `shot rocket` scenario), `crates/app/src/hud.rs` (per-stage panel).

## Defaults (change if desired)

- Style: clean hard-sci-fi (metallic greys/whites, colored bands per stage), not
  photoreal.
- Rocket view is a local scene for now; placed on real LOD terrain once that
  lands.
- Tab cycles Map (orbital) <-> Rocket; existing keys unchanged.

## Verification

`cargo run -p app -- shot rocket out/rocket.png` per phase (reads the PNG to
confirm geometry/staging), plus `cargo check` native + wasm. Auto-deploys to the
web demo.

## Backlog (after this task)

- GPU rocket-view LOD terrain (wire in `crates/terrain`): place the pad/rocket on
  real displaced terrain with log-depth + camera-relative positions.
- Ground view + astronaut 3rd-person view; full view cycle.
- Map-view polish: pan, center-on-body / center-on-rocket.
- Economy loop: fundraise, launch parts (robonauts, refineries), assemble a
  factory in orbit, fabricate advanced craft.
