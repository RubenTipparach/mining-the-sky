//! Bakes the generated world into the texture the real-time client consumes.
//!
//! Run from the workspace root:
//!   cargo run -p worldgen --bin bake --release -- [seed]
//! Writes crates/app/assets/planet.png (RGBA equirect: RGB albedo, A lights).

use worldgen::{generate, GenConfig};

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(GenConfig::default().seed);

    let world = generate(GenConfig { seed, ..Default::default() });
    std::fs::create_dir_all("crates/app/assets").unwrap();
    worldgen::render::write_planet_texture(&world, "crates/app/assets/planet.png", 2048, 1024);
    println!("baked crates/app/assets/planet.png (2048x1024, seed {seed})");

    // The unified city index: the client loads this and generates/caches each
    // city's layout on demand (same data drives ground, far-LOD and orbit lights).
    let descs = worldgen::city_descs(&world);
    let index = worldcity::CityIndex::from_descs(descs);
    std::fs::write("crates/app/assets/cities.bin", index.to_bytes()).unwrap();
    println!("baked crates/app/assets/cities.bin ({} cities)", index.len());
}
