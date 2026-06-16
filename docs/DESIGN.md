# Mining the Sky — Master Design Document

> A realistic, to-scale, multiplayer hard-sci-fi space sim set in a fictionalized
> version of the **Kepler-47** circumbinary system. Massive-scale orbital
> maneuvering, in-situ resource utilization (ISRU), space industrialization,
> factory automation, and advanced spaceship construction — in an async,
> time-compressed sandbox.

- **Status:** Draft 0.1 (initial design)
- **Date:** 2026-06-16
- **Language/Runtime:** Rust
- **Inspirations:** [Caelum](https://github.com/RubenTipparach/Caelum) (rendering tech),
  *High Frontier* (board game — orbital/economic model), Kerbal Space Program
  (patched conics), Factorio / Satisfactory (automation), EVE Online (sandbox/economy).

---

## 1. Vision & Design Pillars

We are building a simulation-first game where the *physics of orbits* and the
*economics of industry* are the core toys. The fantasy is: bootstrap an
interplanetary industrial civilization from a single ship, one Hohmann transfer
and one mined asteroid at a time.

**Pillars**

1. **To-scale and physically honest.** Real distances, real planet sizes, real
   orbital periods. Space *feels* enormous because it *is* enormous. The
   rendering tech (from Caelum) exists specifically to make this scale
   renderable and beautiful.
2. **Orbital maneuvering as the central skill.** Movement is not WASD through a
   void — it is delta-v budgeting, transfer windows, gravity assists, and
   maneuver nodes. Borrowed directly from *High Frontier*'s "burns cost fuel,
   trajectories are committed" feel.
3. **ISRU and industrialization.** You don't buy ships from a shop; you mine
   regolith, refine volatiles, smelt metals, fabricate parts, and assemble
   craft in orbital shipyards. The supply chain *is* the game.
4. **Async multiplayer sandbox with time compression.** Every player runs their
   own clock and their own deterministic orbital propagation. The shared world
   is a persistent economy and a set of authoritative facts, not a lockstep
   simulation. Time compression is local and never blocks other players.
5. **Hard sci-fi, no magic.** Reaction drives, radiators, delta-v, life support,
   power budgets. Constraints are the gameplay.

**Explicit non-goals (for v1)**

- No hex grid. (Caelum's `hex_terrain.c` system is *not* ported.)
- No FTL, no artificial gravity hand-waving beyond spin habitats.
- No twitch combat netcode / rollback. The sim is not lockstep.

---

## 2. What We Take From Caelum (and What We Don't)

Caelum is a custom **C** engine (~80% C) built on sokol-gfx with GLSL shaders.
We are reimplementing the *concepts* in Rust + WebGPU (WGSL), not transliterating
the C. Below is the concrete mapping from Caelum source to our ports.

| Caelum source | What it does | Our port |
| --- | --- | --- |
| `src/lod.c` / `lod.h` | Icosahedron-rooted **aperture-4 quadtree** sphere LOD. 20 root faces, max depth 13, 16,384-node pool, split when `distance < patch_arc * split_factor`. Per-frame GPU upload budget (64/frame). | `crates/render/src/lod/` — same icosphere + quadtree, ported to f64 spherical math + wgpu mesh upload budget. |
| `lod.c` per-level `level_stats{patch_count, min/max_distance}` | The **LOD analyzer** — per-depth statistics gathered each frame in `lod_tree_update`. | `crates/render/src/lod/analyzer.rs` + egui overlay panel. |
| `src/camera.c` / `camera.h` | `pos_d[3]` (f64 accumulator) → `position` (f32 for render). `jetpack_speed_mult` = **scroll-wheel speed multiplier**. `space_mode` toggles at >50 km. | `crates/sim/src/camera.rs` — f64 position, scroll-wheel exponential speed scaling, distance-aware mode switch. |
| `shaders/planet.glsl` | **Logarithmic depth** (`gl_Position.z = (log2(max(1e-6, 1.0+w)) * Fcoef + log_depth.z) * w`) and **double-float floating origin** (`cam_rel = (a_pos - cam_hi) - cam_lo`). | `assets/shaders/*.wgsl` — same log-depth + hi/lo camera-relative trick in WGSL. |
| `shaders/atmosphere.glsl` | **Single-scattering Rayleigh+Mie** ray march, 8 view samples / 4 light samples, Henyey-Greenstein, `wavelengthsInv4 = (5.602, 9.473, 19.644)`. | `assets/shaders/atmosphere.wgsl` — same model, optionally upgraded to precomputed Bruneton LUTs later. |
| `shaders/planet.glsl` surface shading | Smooth terminator, atmospheric tinting, AO, ocean specular, rim light, Rayleigh fog. | Port to WGSL PBR-lite surface model. |
| `src/celestial.c` | Solar-system body definitions and hierarchy. | `crates/sim/src/orbits/` body graph (see §6). |
| `src/hex_terrain.c` | Hex grid terrain. | **Dropped** per design goal. |

**Key Caelum techniques restated (so the math survives the port):**

- **Logarithmic depth buffer (GPU Gems 3 style).** With `Fcoef = 2 / log2(far + 1)`:
  ```
  // vertex shader, after computing clip-space gl_Position
  z = (log2(max(1e-6, 1.0 + w)) * Fcoef + offset) * w;   // offset adjusts GL vs WebGPU 0..1 depth
  ```
  This buys planetary-to-orbital draw distance from a single depth buffer with
  no z-fighting. WebGPU clip space is `0..1` (like D3D/Metal), so our `offset`
  term differs from GL's `-1`.
- **Double-float (hi/lo) floating origin on the GPU.** The CPU keeps f64 world
  positions; each frame we pick an origin (the camera) and pass the camera
  position to the GPU as two f32s (`hi` + `lo`, an error-free split). Vertices
  are also camera-relative. `cam_rel = (pos - cam_hi) - cam_lo` cancels the
  large magnitude first, then corrects the residual, giving sub-cm precision
  thousands of km out without an f64 GPU path.

---

## 3. Technology Stack (Decisions & Rationale)

The user goal: *"compile knowledge and technologies that are cutting edge to
Rust."* Each choice below lists the pick, the runner-up, and why.

### 3.1 Graphics: **`wgpu`** (WebGPU)

- **Pick:** [`wgpu`](https://github.com/gfx-rs/wgpu) — the Rust WebGPU
  implementation. One WGSL codebase runs on Vulkan, Metal, DX12, and the browser
  (WASM + WebGPU). Native multithreaded command encoding, compute shaders for
  our GPU orbital propagation (§6.4).
- **Runner-up:** raw `ash` (Vulkan) — more control, far more boilerplate, no
  browser target. Rejected: WebGPU's portability + compute is exactly what a
  to-scale multiplatform sim needs.
- **Shaders:** WGSL authored directly. We keep a thin reflection layer so
  pipelines validate bind groups at load.

### 3.2 Windowing/Input: **`winit`** + **`egui`** (debug UI)

- `winit` for cross-platform windowing/input (and the WASM canvas).
- `egui` for the LOD analyzer, orbital debug overlays, and dev tooling.
  Ships only in dev builds (feature-gated).

### 3.3 Math: **`glam`** (f32 SIMD) + explicit **f64** for world/orbit state

- `glam` for render-side f32 vectors/matrices (SIMD-accelerated).
- World positions, orbital elements, and time are **f64** (`DVec3` / custom).
  The f32↔f64 boundary is the floating-origin rebasing step (§5).

### 3.4 ECS / Engine core: **`bevy_ecs`** (standalone) + custom render core

- **Pick:** Use `bevy_ecs` as a *standalone* crate (it can be used without the
  full Bevy app/renderer) for its mature, automatically-parallel scheduler and
  archetype storage. Build our **own** wgpu render core, LOD, and orbital
  systems as plain systems/resources around it.
- **Runner-up A:** full **Bevy**. Tempting (huge ecosystem) but its renderer and
  coordinate assumptions (f32 transforms, no native floating origin / log-depth)
  fight our core requirements. We'd spend more time subverting Bevy's render
  graph than building. Revisit if Bevy ships first-class large-world support.
- **Runner-up B:** `hecs` / `flecs-rs` — lighter, but we'd reimplement the
  parallel scheduler. `bevy_ecs` gives us that for free.
- **Decision:** custom engine, `bevy_ecs` for the ECS, `wgpu` for rendering. We
  keep full control of LOD, depth, floating origin, and atmosphere — exactly the
  things Bevy doesn't do for us.

### 3.5 Async runtime & jobs: **`tokio`** (net/IO) + **`rayon`** (data parallelism)

- `tokio` for networking, async asset streaming, and the persistence layer.
- `rayon` for CPU data-parallel work: LOD tree split/merge passes, mesh
  generation, batch orbital propagation fallback, factory/economy ticks.
- The ECS scheduler, rayon, and tokio coexist: ECS owns frame systems, rayon
  owns parallel-for inside a system, tokio owns off-thread IO.

### 3.6 Physics (rigid-body, docking, collisions): **`avian3d`** or `rapier3d`

- Local rigid-body physics (ship docking, construction collisions, surface
  rovers) uses [`avian3d`](https://github.com/Jondolf/avian) (ECS-native) or
  `rapier3d`. **Orbital** motion is *not* rigid-body physics — see §6. Rigid
  bodies operate only in the local floating-origin frame near the player.

### 3.7 Networking: **QUIC via `aeronet` / WebTransport** (see §7)

- **Pick:** [`aeronet`](https://github.com/aecsocket/aeronet) transport
  abstraction over **WebTransport (QUIC)** — reliable streams *and* unreliable
  datagrams, one API for native (`wtransport`/`quinn`) and browser (WASM). This
  is the modern successor to WebRTC data channels for our use case.
- **Runner-up:** WebRTC data channels (`matchbox`/`webrtc-rs`). Heavier
  handshake (SDP/ICE/STUN/TURN), designed for P2P/NAT punching we don't need for
  a client–authority topology. We keep WebRTC in our back pocket only if we
  later want serverless P2P clusters.
- **Replication:** `lightyear` or `renet`-style snapshot/delta replication on
  top of `aeronet`. See §7 for why our async time model means this is *state
  replication*, not lockstep.

### 3.8 Serialization & persistence

- Wire/codec: `bincode` + `serde` for compact binary; `bitcode` if we need
  tighter packing. Snapshots delta-compressed.
- World persistence: embedded `redb` or `sqlite` (via `rusqlite`) for the
  authoritative economy/state; bodies and orbits are mostly procedural + element
  sets, so the DB stays small.

### 3.9 Workspace layout

```
mining-the-sky/
├─ Cargo.toml                  # workspace
├─ crates/
│  ├─ app/        # binary: window, main loop, wires everything together
│  ├─ render/     # wgpu core, LOD, atmosphere, log-depth, floating-origin draw
│  ├─ sim/        # orbits, time, camera, ECS components/systems
│  ├─ industry/   # ISRU, factories, automation, ship construction
│  ├─ net/        # aeronet transport, replication, authority
│  ├─ universe/   # Kepler-47 system definition & procedural bodies
│  └─ core/       # math (f64), units, IDs, shared types
├─ assets/
│  ├─ shaders/    # *.wgsl
│  └─ data/       # system definition (RON/TOML), resource tables
└─ docs/
   └─ DESIGN.md   # this file
```

---

## 4. Engine Architecture & Threading

```
                      ┌──────────────────────────────────────────┐
                      │                main loop                  │
                      │  (winit event loop, fixed + variable dt)  │
                      └───────────────┬──────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
 ┌──────▼──────┐              ┌────────▼────────┐            ┌────────▼────────┐
 │  SIM stage  │              │  RENDER stage   │            │   NET / IO      │
 │ (bevy_ecs   │              │  (wgpu)         │            │  (tokio tasks)  │
 │  schedule)  │              │                 │            │                 │
 │ • time/clock│              │ • floating-orig │            │ • aeronet recv  │
 │ • orbit eval│              │   rebase        │            │ • replication   │
 │ • camera    │              │ • LOD update    │            │ • persistence   │
 │ • industry  │  ── snapshot │   (rayon)       │            │                 │
 │   ticks     │  ──────────► │ • record cmd    │            └─────────────────┘
 │ • physics   │   (interp)   │   buffers       │
 │   (avian)   │              │   (parallel)    │
 └─────────────┘              └─────────────────┘
```

- **Fixed-step simulation** for industry/economy and rigid-body physics
  (e.g. 30 Hz), decoupled from render framerate.
- **Orbital state is evaluated, not stepped** (§6) — it's a pure function of
  absolute time, so it doesn't need a fixed step and is trivially time-
  compressible and parallelizable.
- **Multithreaded rendering:** the LOD update produces a list of visible patches;
  command-buffer recording is split across worker threads via wgpu
  `RenderBundle`s / parallel `CommandEncoder`s, then submitted on the main
  queue. `bevy_ecs`'s scheduler runs independent systems in parallel; `rayon`
  parallelizes the inner loops (per-face LOD descent, per-patch mesh gen).

---

## 5. Coordinate, Scale & Time Systems

### 5.1 Floating origin (the foundation of "to-scale")

- **World space is f64**, in meters, with the system barycenter at the origin.
  Kepler-47c orbits at ~1 AU ≈ `1.5e11` m — f32 (24-bit mantissa) loses meter
  precision past ~16,000 km from origin, so f64 is mandatory for world state.
- **Each frame** we rebase the render origin onto the camera. The GPU never sees
  absolute coordinates; it sees **camera-relative f32**, with the camera passed
  as a **hi/lo double-float split** so even near-but-large offsets keep sub-cm
  precision (Caelum's trick, §2).
- **Logarithmic depth buffer** lets a single pass draw a 1 m bolt on a hull and a
  planet 10,000 km away without z-fighting.

### 5.2 Camera movement scaling (scroll wheel)

Ported from Caelum's `jetpack_speed_mult` / `space_mode`:

- Scroll wheel adjusts an **exponential speed multiplier** (`mult *= 1.2^notches`,
  clamped to a floor of 1.0 and a ceiling tied to context).
- Free-fly speed also **auto-scales with altitude / distance to nearest body**,
  so the same input feels right whether you're inspecting a girder (cm/s) or
  crossing interplanetary space (Gm/s). At >50 km from any surface the camera
  enters `space_mode` (orbital framing, body-relative).
- This is the single best "it just feels right" detail from Caelum and is a v1
  must-have.

### 5.3 Time model & compression (and why it's async-safe)

- There is a single **absolute simulation epoch** `T0` and a monotonic
  **universe time** `t` measured in seconds since `T0`.
- Every player has a **local `t` and a local time-compression factor** (1×,
  10×, 1,000×, …, like KSP/High Frontier warp). Compression just advances *that
  player's* `t` faster.
- **Crucially, body and on-rails ship positions are `f(elements, t)`** — a pure
  function of absolute time (§6). So two players at the same `t` compute the
  *same* positions regardless of how fast each got there. Time compression
  therefore needs **no synchronization** and never blocks anyone. This is the
  architectural reason the async-time goal is even possible.
- The shared *economy and discrete events* (a trade completing, a factory
  finishing a batch, a ship arriving) are reconciled through the authority
  server on a **causal timeline**, not a wall clock (§7.3).

---

## 6. Orbital Mechanics (the core sim)

> The user asked about RK4 "or something better." **Recommendation: do not make
> RK4 the backbone.** Use **analytic Keplerian propagation on patched conics** as
> the primary model, with high-order/symplectic integration reserved for the
> cases that actually need it. Rationale below.

### 6.1 Why not "just RK4 everything"

- **RK4 is not symplectic** — it has secular energy drift, so a station left for
  game-weeks at 10,000× compression slowly spirals. Bad for a persistent sandbox.
- **Stepwise integration is non-deterministic across machines/time-compression.**
  Different step counts (because of different warp factors and frame rates)
  produce different floating-point accumulation → players disagree on positions.
  That breaks the "everyone computes their own, but agrees" requirement.
- **It's expensive at scale.** Tens of thousands of debris/ships/asteroids
  stepped every frame is wasteful when most are on unperturbed conics.

### 6.2 Primary model: analytic Kepler + patched conics ("on rails")

Like KSP and conceptually like *High Frontier*'s committed trajectories:

- Each body/ship on rails has a set of **orbital elements** `(a, e, i, Ω, ω, M0, epoch)`
  about a single dominant primary (a star, planet, or the binary barycenter).
- Position at time `t` = solve **Kepler's equation** `M = E − e·sin E`
  (elliptic) / hyperbolic analog, via **Newton–Halley iteration** (3–4 iters,
  f64). This is:
  - **Exact** (no drift — a parked station stays parked forever),
  - **`O(1)` per body, evaluated at absolute `t`** (deterministic & async-safe),
  - **Embarrassingly parallel** (CPU via rayon, or GPU via compute — §6.4).
- **Patched conics / SOI transitions:** each body owns a sphere-of-influence
  radius `r_SOI = a · (m/M)^(2/5)`. A trajectory is a sequence of conic arcs;
  when a ship crosses an SOI boundary, we re-root its elements to the new
  primary at the crossing time. Hyperbolic flybys give free gravity assists.
- **Maneuver nodes:** a burn at time `t_b` instantaneously changes velocity and
  produces a **new element set valid from `t_b`**. Trajectories are thus a list
  of `(elements, valid_interval, primary)` arcs — cheap to store, replicate, and
  re-derive. This is the "committed burn" feel from *High Frontier*.

### 6.3 The circumbinary wrinkle (Kepler-47 specifics)

Kepler-47 is a **binary** (§8). Planets orbit the *barycenter* of the two stars,
not a single mass — pure two-body conics are an approximation here.

- **Cheap model (v1):** treat the binary as a single point mass at the
  barycenter for planet/ship orbits outside the binary's critical radius
  (planets sit comfortably outside it). Good enough visually and for gameplay.
- **Honest model (v2, optional):** the two stars are a known, closed-form
  two-body orbit (period ~7.45 d); their gravity on a circumbinary craft is a
  **time-periodic perturbation**. Apply it via **Encke's method** (integrate
  only the small deviation from the reference conic) — cheap and drift-resistant
  — only for craft we're actively simulating near the stars.

### 6.4 GPU-assisted batch propagation

- The Kepler solve is a perfect compute-shader workload: one thread per
  body/ship/debris, reading an element buffer, writing a position buffer.
- **Use cases:** rendering positions of thousands of asteroids/debris, drawing
  trajectory ribbons (sample `f(t)` along an arc), transfer-window search
  (evaluate many candidate departure times in parallel).
- Run on the same `wgpu` device as rendering; results feed instanced draws
  directly (positions never round-trip to CPU for visuals).

### 6.5 When we DO integrate numerically

For the small set of bodies under continuous thrust or strong perturbation
(active maneuvering ships, station-keeping, aerobraking passes):

- **Adaptive high-order:** **Dormand–Prince RK45** (error-controlled) for
  general thrusting arcs, or **IAS15** (15th-order adaptive, from the REBOUND
  N-body code) when we want research-grade accuracy on chaotic close encounters.
- **Long-term N-body (if ever needed):** a **Wisdom–Holman symplectic
  integrator** preserves energy over millions of steps — the right tool for
  evolving the *whole system* forward, which we mostly avoid by keeping bodies
  on analytic rails.
- These run off the main thread (rayon/tokio) and only for the handful of craft
  that need them; everything else stays on cheap analytic rails.

**Summary recommendation:** analytic Kepler + patched conics as the spine
(deterministic, async-safe, GPU-batchable), Encke for circumbinary perturbation,
RK45/IAS15 only for active-thrust/close-encounter craft. RK4 alone is the wrong
backbone for a persistent, time-compressed, multiplayer sandbox.

---

## 7. Networking & Multiplayer

### 7.1 Topology

- **Authoritative server** owning the canonical economy, ownership, and discrete
  events. Clients are simulation peers for *continuous* orbital state (which they
  can recompute exactly from elements) and request *discrete* actions from the
  server.
- **Transport:** `aeronet` over **WebTransport/QUIC** — reliable ordered streams
  for RPC/economy, unreliable datagrams for high-frequency presence (a player's
  ship near you). Native and browser share one code path.

### 7.2 What is replicated (and why it's small)

Because orbits are `f(elements, t)`, we **do not stream positions** for on-rails
objects. We replicate:

- **Element sets + maneuver-node lists** (tiny, changes only on burns).
- **Discrete world events** (factory batch complete, trade filled, ship docked).
- **Authoritative economy state** (inventories, market orders, ownership).
- **Live presence** for nearby actively-thrusting craft (datagram stream, only
  within interaction range).

A whole fleet's trajectories are a few KB of elements; clients reconstruct the
motion locally. This is what makes async time compression bandwidth-cheap.

### 7.3 Async time, causal consistency, and conflict resolution

- Players live at different local `t` (different warp). The server keeps a
  **causal event log** keyed by universe time, not wall-clock.
- An action is **submitted with the universe time it occurs at**; the server
  validates against canonical state at that `t` and either commits (assigning a
  definitive ordering) or rejects (e.g. the asteroid was already mined out by a
  player who reached that `t` earlier).
- This is closer to a **distributed ledger / eventual-consistency** model than
  to game lockstep. Contention is on *resources and ownership*, resolved by the
  authority, not on physics (physics is deterministic and shared).
- **Anti-cheat:** because clients compute their own orbits, the server
  spot-checks submitted trajectories against the analytic model (cheap to
  re-derive any single arc) and rejects impossible delta-v / fuel claims.

### 7.4 Why not WebRTC

WebRTC data channels were considered (great for P2P/NAT traversal) but bring
SDP/ICE/STUN/TURN complexity we don't need for a client–authority topology.
WebTransport/QUIC gives us streams+datagrams, native+browser, with a simpler
handshake. WebRTC stays as a future option for serverless P2P shards.

---

## 8. The Kepler-47 System (To-Scale, Fictionalized)

Kepler-47 is a real **transiting circumbinary multiplanet system** ~3,400 ly
away. We model it to scale, then *fictionalize* the bodies the gameplay needs
(moons, asteroid belts, resource distributions, named stations) that reality
doesn't hand us.

### 8.1 Stars (real parameters)

| Body | Mass (M☉) | Radius (R☉) | Temp (K) | Notes |
| --- | --- | --- | --- | --- |
| Kepler-47A (primary) | ~1.04 | ~0.96 | ~5,636 | Sun-like G-type |
| Kepler-47B (secondary) | ~0.36 | ~0.35 | cooler | M-dwarf companion |
| Binary orbit | — | — | — | Period **~7.45 d**, the pair orbits a common barycenter |

### 8.2 Planets (real, approximate — circumbinary, orbiting the barycenter)

| Planet | Period (d) | Radius (R⊕) | Mass (M⊕, est.) | Notes |
| --- | --- | --- | --- | --- |
| Kepler-47b | ~49.5 | ~3.0 | ~7–10 | Innermost; hot |
| Kepler-47d | ~187 | ~7 | (larger) | Discovered later; between b and c |
| Kepler-47c | ~303 | ~4.7 | ~16–23 | In/near the habitable zone |

> Values are approximate and will be pinned to a single cited dataset in
> `assets/data/kepler47.ron`. "To scale" = correct relative sizes, distances,
> and periods; absolute numbers may be tuned slightly for playability and then
> documented as the fictionalization delta.

### 8.3 Fictionalized additions (for gameplay)

- **Moons** around 47c/47d (none confirmed in reality) as early colony targets
  with low gravity wells — ideal first ISRU sites.
- **Asteroid belt(s) / Trojan swarms** at resonances and the binary's stable
  Lagrange-like regions, seeded with a procedural resource distribution
  (metals, volatiles, water ice) — the raw feedstock of the industry game.
- **Named orbital infrastructure** (starter station, shipyards) placed on
  defined element sets.
- **Resource model:** each body/asteroid carries a composition vector (Fe, Ni,
  Si, C, H₂O, volatiles, rare metals) driving the ISRU/refining chains (§9).

### 8.4 Data definition

The system lives in `assets/data/kepler47.ron` as a declarative body graph:
`{ id, parent, mass, radius, orbital_elements, atmosphere_params, composition }`.
Procedural fields (asteroid swarms, surface detail seeds) are generated from a
fixed seed so all clients agree.

---

## 9. Gameplay Systems

### 9.1 ISRU & resource pipeline

Mine → refine → fabricate → assemble, each step a placeable machine with power,
heat, and throughput constraints:

```
regolith/ice  ──mine──►  raw ore/volatiles  ──refine──►  metals, propellant, water/O2
        │                                                     │
        └───────────────────────► fabrication ◄───────────────┘
                                       │
                                  parts (structural, drives, tanks, electronics)
                                       │
                                 orbital shipyard ──► assembled spacecraft
```

- Hard-sci-fi constraints: **power budget** (solar falloff with distance from the
  binary — note *two* light sources), **waste heat / radiators**, **mass &
  delta-v**, **life support** for crewed ops.

### 9.2 Factories & automation (Factorio-in-orbit)

- Machines connected by logistics (conveyors on surface/station, tugs/haulers
  between orbits). Automation logic is data-driven recipes; throughput simulated
  on the fixed-step economy tick, not per-frame.
- Belts/Trojans + planetary surfaces are the build canvases; the LOD/floating-
  origin tech is what lets a sprawling surface factory and an orbital shipyard
  coexist in one continuous world.

### 9.3 Spaceship construction

- Ships are assemblies of fabricated parts with real mass distribution →
  feeds the delta-v / maneuver model (§6). Drives, tanks, radiators, reactors,
  habitats. A ship's capability is the *output* of your industrial chain, not a
  purchase.

### 9.4 *High Frontier* influence

- Trajectories are **committed burns** with fuel costs; transfer windows and
  gravity assists matter. The economic loop (refine fuel to go further to mine
  more to build more) mirrors High Frontier's core engine. We adapt its
  "patched-conic-ish, delta-v-budgeted" movement into a continuous real-time
  (time-compressed) sim.

---

## 10. Rendering Pipeline Detail

Frame outline (per camera):

1. **Rebase origin** to camera; compute hi/lo camera split (CPU, f64→2×f32).
2. **LOD update** (rayon): descend the icosphere quadtree per visible root face,
   split/merge by `distance < patch_arc · split_factor`, collect per-level stats
   for the **analyzer**, enqueue up to N patch mesh uploads this frame (budgeted,
   like Caelum's 64/frame).
3. **GPU orbital pass** (compute): propagate batched bodies/debris elements → an
   instance position buffer.
4. **Depth/opaque pass:** planets & meshes with **logarithmic depth** + camera-
   relative vertices. Surface shading ports Caelum's terminator/AO/ocean-spec/
   rim/fog model to a WGSL PBR-lite shader.
5. **Atmosphere pass:** fullscreen single-scattering Rayleigh+Mie ray march
   (8/4 samples, Henyey-Greenstein) reading scene depth; per-body radii &
   scatter coefficients as uniforms.
6. **Post / UI:** tonemap, then egui dev overlays (LOD analyzer, orbit debug).

Command recording for passes 4–5 is split across threads via render bundles.

---

## 11. Milestones / Roadmap

| Milestone | Goal | Key deliverables |
| --- | --- | --- |
| **M0 — Skeleton** | Window + wgpu clears, workspace crates, CI. | `app`/`render`/`core` crates, WGSL pipeline harness. |
| **M1 — One planet, to scale** | Render Kepler-47c to scale with LOD + log-depth + floating origin + atmosphere; scroll-wheel camera. | LOD quadtree, log-depth WGSL, hi/lo origin, atmosphere port, LOD analyzer overlay. |
| **M2 — Orbits on rails** | Full system from `kepler47.ron`; analytic Kepler + patched conics; maneuver nodes; trajectory ribbons; time compression. | `sim/orbits`, GPU batch propagate, time/clock, transfer planner. |
| **M3 — ISRU loop** | Mine→refine→fabricate→build one ship on one body. | `industry` crate, recipes, power/heat, shipyard. |
| **M4 — Multiplayer sandbox** | Authority server, aeronet/WebTransport, element + economy replication, async time. | `net` crate, causal event log, anti-cheat trajectory checks. |
| **M5 — Browser build** | WASM + WebGPU client. | wasm target, asset streaming, WebTransport-in-browser. |

---

## 12. Open Questions / Risks

1. **Determinism across platforms.** Even analytic Kepler relies on f64
   transcendental functions; results can differ in the last ULPs across CPUs.
   Mitigation: evaluate at absolute `t` (no accumulation), define tolerances for
   "agreement," and let the authority be the tiebreaker on contested resources.
   *Open:* do we need a software-defined `sin/cos`/`sqrt` for bit-exactness?
2. **Bevy_ecs vs. full custom.** Standalone `bevy_ecs` is the plan, but if its
   scheduler assumptions bite us we may drop to `hecs` + a hand-rolled scheduler.
3. **Circumbinary fidelity vs. cost.** v1 single-point-mass barycenter is an
   approximation; decide when (if) Encke perturbation is worth it.
4. **Atmosphere model upgrade path.** Start with Caelum's per-frame single-
   scattering ray march; budget for precomputed Bruneton LUTs if it costs too
   much at scale.
5. **Browser memory/perf.** Massive worlds in WASM: validate streaming + memory
   budget early (M5 is late, but spike it during M1).
6. **WebTransport maturity in browsers.** Confirm target-browser support; keep
   WebRTC fallback designed-for, not bolted-on.
7. **Kepler-47 numbers.** Pin one citable dataset; record every fictionalization
   delta in `kepler47.ron` comments so "to-scale" claims stay honest.

---

## Appendix A — Source References

- Caelum engine (C/sokol-gfx): `src/lod.c`, `src/camera.c`, `shaders/planet.glsl`,
  `shaders/atmosphere.glsl`, `src/celestial.c` —
  <https://github.com/RubenTipparach/Caelum>
- Rust graphics: `wgpu` (WebGPU). Networking: `aeronet` (WebTransport/QUIC),
  `lightyear`/`renet` (replication), `quinn`/`wtransport` (QUIC). ECS:
  `bevy_ecs`. Math: `glam`. Parallelism: `rayon`, `tokio`. Physics: `avian3d`.
- Orbital mechanics: Kepler's equation (Newton–Halley), patched conics (KSP-
  style), Encke's method, Dormand–Prince RK45, IAS15 (REBOUND), Wisdom–Holman
  symplectic.
- Kepler-47 science: Orosz et al. 2012, *Kepler-47: A Transiting Circumbinary
  Multiplanet System* (Science) and follow-ups (the third planet, 47d).
- Depth/precision techniques: GPU Gems 3 logarithmic depth buffer; double-float
  (hi/lo) camera-relative rendering.
