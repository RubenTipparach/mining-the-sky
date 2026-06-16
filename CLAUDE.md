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
