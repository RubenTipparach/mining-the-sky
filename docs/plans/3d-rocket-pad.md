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

1. DONE - Static stack on the pad: 3D Pioneer I (booster + upper + payload +
   nose, fins, engine nozzles) on a launch pad with mount legs over ground/sky,
   in the new mesh+depth pipeline rocket view (Tab cycles to it), orbit camera,
   with the vehicle-assembly HUD (per-stage mass/dv, TWR, total dv, payload,
   target orbit). Verified via `shot rocket`.
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

- DONE (first pass) - GPU rocket-view LOD terrain: the rocket view now renders
  the real `crates/terrain` cube-sphere LOD surface in a floating-origin tangent
  frame at the spaceport, with a logarithmic depth buffer, and the rocket sits on
  it.

## Surface-detail stages (do A, B, D now; C = Phase 2, on user go-ahead)

- Stage A - Terrain realism + match the map: make the rocket-view terrain derive
  from the SAME elevation field as the worldgen baked planet (consistent
  continents/coastlines, so the coastal spaceport reads with ocean to one side),
  and increase relief so the vista is dramatic.
- Stage B - Atmosphere + sky: replace the flat clear with a sky gradient + sun
  glow, and add aerial-perspective (distance fog) on the terrain so it fades into
  the horizon haze. Sky as a fullscreen pass; fog in the terrain fragment.
- Stage D - Surface detail: finer near-pad LOD (deeper quadtree, smaller patches)
  and procedural ground detail (slope-based rock/grass colouring + micro colour
  noise) so the ground is not flat green.
- Stage C = Phase 2 (NOT YET - wait for user): launch + staging animation. On
  Space the rocket lifts and stages detach/fall as propellant depletes.
- Ground view + astronaut 3rd-person view; full view cycle.
- Map-view polish: pan, center-on-body / center-on-rocket.
- Economy loop: fundraise, launch parts (robonauts, refineries), assemble a
  factory in orbit, fabricate advanced craft.
