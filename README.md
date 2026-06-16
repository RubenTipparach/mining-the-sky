# Mining the Sky

A realistic, to-scale, multiplayer hard-sci-fi space sim set in a fictionalized
**Kepler-47** circumbinary system. Massive-scale orbital maneuvering, in-situ
resource utilization (ISRU), space industrialization, factory automation, and
advanced spaceship construction - in an async, time-compressed sandbox.

Built in **Rust** with **WebGPU (`wgpu`)**. Rendering tech (LOD, floating origin,
logarithmic depth, atmospheric scattering, scroll-wheel camera scaling, LOD
analyzer) is ported from the [Caelum](https://github.com/RubenTipparach/Caelum)
engine - minus the hex grid.

## Design

See **[docs/DESIGN.md](docs/DESIGN.md)** for the full design document covering:

- Tech stack decisions (`wgpu`, `bevy_ecs`, `aeronet`/WebTransport, `glam`, `rayon`, `tokio`)
- Rendering pipeline (icosphere quadtree LOD, log-depth, double-float floating origin, single-scattering atmosphere)
- Orbital mechanics (analytic Kepler + patched conics + GPU batch propagation - and why *not* plain RK4)
- Coordinate/time systems and async time compression
- Networking & multiplayer (QUIC, element-set replication, causal economy)
- ISRU / factory / shipbuilding gameplay
- The to-scale Kepler-47 system definition
- Roadmap and open questions

## Crates

- `crates/worldgen` - deterministic procedural home world: 3D-noise elevation,
  rivers/deltas via flow accumulation, coastal-delta major cities, river-corridor
  minor cities, a great-circle MST road network, night-light emission, and an
  auto-sited equatorial launch complex. Includes a CPU PNG preview renderer.
- `crates/sim` - orbital mechanics and launch-to-orbit: analytic two-body
  ("on-rails") state/elements, a central body + atmosphere, staged launch
  vehicles, an RK4 powered-ascent integrator with a programmed gravity turn and
  staging, and analytic circularization at apoapsis. Reaches a stable ~200 km
  orbit from the seed-47 spaceport and plots the trajectory.
- `crates/app` - the real-time client (wgpu / WebGPU) that runs natively and in
  the browser via WebAssembly. Renders the baked worldgen planet (real
  coastlines, day/night terminator, atmospheric limb, dark-side city lights) as
  the seed of the Caelum-style renderer.

## Build and run

```sh
# Generate planet/city/road/night-light PNG previews into ./out
cargo run -p worldgen --bin preview --release -- 47

# Bake the world into the texture the client samples
cargo run -p worldgen --bin bake --release -- 47

# Stack a rocket, launch from the spaceport, reach orbit (writes out/launch.png)
cargo run -p sim --bin launch --release

# Run the real-time WebGPU client natively
cargo run -p app --release

# Build the browser (WASM) client locally
cd crates/app && trunk serve     # then open the printed localhost URL
```

A few generated previews (seed 47):

| Cities + roads | Day | Night (city lights) |
| --- | --- | --- |
| ![](docs/images/cities_roads.png) | ![](docs/images/globe_day.png) | ![](docs/images/globe_night.png) |

Launch-to-orbit (Pioneer I from the seed-47 spaceport, 205 km circular orbit):

![](docs/images/launch.png)

## Live demo (GitHub Pages)

`.github/workflows/pages.yml` builds the WASM client with Trunk and deploys it to
GitHub Pages on every push to `main` (same approach as Caelum's web build). To
enable it once: repo Settings -> Pages -> Source: "GitHub Actions". The demo
needs a WebGPU-capable browser.

## Status

Initial vertical slice working (design doc roadmap M0-M2): procedural planet
with coastal-delta cities, roads, and night lights; a live native/browser
WebGPU view of the baked world; and a staged rocket that launches from the
spaceport into a stable orbit. Next: render the rocket + orbit in the real-time
client, then the economy loop (fundraise, launch parts, assemble in orbit).
