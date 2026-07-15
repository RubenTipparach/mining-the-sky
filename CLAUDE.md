# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Don'ts

- Never rebuild an existing feature from scratch. Before writing something that
  sounds like it might already exist (a generator, a renderer, a UI panel, a
  math helper), search the codebase and find it first. Then extend or
  parameterize the existing code - add an argument, a variant, a strategy - so
  there is one implementation, not two parallel ones that can drift. If the
  current code genuinely cannot be adapted, say why before replacing it.

## Writing conventions

- Never use em-dashes (the "—" character) anywhere in this repo: not in docs,
  comments, commit messages, PR text, or any other content. Use a spaced hyphen
  (" - "), a colon, parentheses, or restructure the sentence instead. This also
  covers en-dashes ("–") in prose; use a hyphen.

## Project

Mining the Sky is a realistic, to-scale, multiplayer hard-sci-fi space sim in
Rust (WebGPU / wgpu). See docs/DESIGN.md for the full design.

## Platform priority

The native desktop build is the priority target from now on, and it is the
multi-threaded one. Use real threads freely for performance: heavy work (terrain
meshing, worldgen, future physics) should run off the render thread on native
(see `crates/app/src/terrain_job.rs` for the pattern - a background worker with
double-buffering so a rebuild never stalls a frame).

Web/wasm is a secondary target and must not gate desktop performance work. The
web build runs single-threaded (real threads on the web need cross-origin
isolation we do not rely on), so threaded systems should fall back to an inline
path on `wasm32` rather than being held back to match it. Keep the wasm build
compiling and runnable, but optimize for native first.

## UI

Prefer a proper GUI toolkit over the bitmap-font / ASCII-terminal HUD that the
prototype currently hand-draws as text quads. Rust has good options that
integrate with wgpu + winit:

- `egui` (immediate-mode) via `egui-wgpu` + `egui-winit` is the default choice
  for in-game HUD, menus, body pickers, build/stage panels, and dev tools.
- `iced` (Elm-style, retained) is the alternative if we want a more structured
  app shell.

New UI should be built with egui rather than more hand-rolled text-quad drawing,
and the existing HUD/menus should migrate to egui as they grow. Keep UI code out
of the render hot path (egui composits over the wgpu frame).

### UI-first controls (no hotkeys for menus/tests)

Everything the player or developer can do must be reachable through the egui UI.
Do not add a keyboard shortcut as the only (or primary) way to trigger an
action.

- Keyboard shortcuts are reserved for controlling the active vehicle (throttle,
  pitch, staging, camera) and for complementing an existing on-screen UI control.
  A key may mirror a button, but never replace it.
- Test/debug scenes and dev toggles (e.g. jumping into a re-entry test,
  switching the plasma renderer) belong in a dedicated egui menu - never bound to
  a hotkey. Put them in the "Test Scenes" panel (`test_menu` in `ui.rs`) and add
  new test scenarios there as buttons, not key bindings.
- When in doubt, add the UI control first; only add a complementary hotkey
  afterwards if it genuinely helps flying.

## Verifying rendering headlessly (software GPU)

The cloud/CI environment has no physical GPU or display, but you can still run
the real wgpu pipeline and verify shaded output by using Mesa's software Vulkan
driver (lavapipe / llvmpipe). This works:

1. Install it once per fresh container (it is not persisted across restarts):

   ```sh
   apt-get install -y mesa-vulkan-drivers vulkan-tools
   ```

2. Point the Vulkan loader at the lavapipe ICD. The repo's `.cargo/config.toml`
   already sets `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json` for all
   `cargo` processes, so `cargo run -p app -- shot ...` just works. For a raw
   binary, export it yourself:

   ```sh
   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
   ./target/release/app shot all
   ```

3. The `app` crate's `shot` mode renders frames to PNGs with no window, e.g.
   `cargo run -p app --release -- shot pad out/pad.png`. Read the PNG to inspect
   the actual GPU output.

So: do NOT claim GPU output is unverifiable. CPU-side generation (worldgen, sim,
LOD quadtree/mesh geometry, rocket stack math) is verifiable directly; shaded
GPU output is verifiable via lavapipe + `shot`. The one thing that still needs a
real browser is confirming the wasm/WebGPU path renders in-browser.

### Always screenshot as you go

Treat a rendered PNG as the unit of done for anything visual. At each visual
change: render a `shot`/preview PNG, `Read` it to actually look at the result
(don't assume), and surface it to the user. Keep a representative shot under
`docs/images/` and reference it from the README/PR. New rendering features should
add a `shot` scenario (or a CPU preview bin) so they stay verifiable headlessly.
This screenshot-driven loop is how this project is built - keep doing it.
