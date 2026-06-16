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

## Status

Early design. No code yet - see the roadmap in the design doc (M0-M5).
