# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Writing conventions

- Never use em-dashes (the "—" character) anywhere in this repo: not in docs,
  comments, commit messages, PR text, or any other content. Use a spaced hyphen
  (" - "), a colon, parentheses, or restructure the sentence instead. This also
  covers en-dashes ("–") in prose; use a hyphen.

## Project

Mining the Sky is a realistic, to-scale, multiplayer hard-sci-fi space sim in
Rust (WebGPU / wgpu). See docs/DESIGN.md for the full design.

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
