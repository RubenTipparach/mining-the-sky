//! Generates a home world and writes PNG previews to `out/`.
//!
//! Usage: `cargo run -p worldgen --bin preview --release -- [seed]`

use worldgen::sites::CityKind;
use worldgen::{generate, GenConfig};

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(GenConfig::default().seed);

    std::fs::create_dir_all("out").unwrap();

    let t0 = std::time::Instant::now();
    let world = generate(GenConfig { seed, ..Default::default() });
    eprintln!("generated world (seed {seed}) in {:?}", t0.elapsed());

    let majors = world.cities.iter().filter(|c| c.kind == CityKind::Major).count();
    let minors = world.cities.len() - majors;
    let pop: f64 = world.cities.iter().map(|c| c.population).sum();
    println!("seed:            {seed}");
    println!("grid:            {}x{}", world.cfg.width, world.cfg.height);
    println!("sea level:       {:.3}", world.sea_level);
    println!("cities:          {majors} major + {minors} minor");
    println!("total pop:       {:.1} M", pop / 1.0e6);
    println!("roads:           {} segments", world.roads.len());
    println!(
        "launch complex:  {}  (lat {:.1} deg, lon {:.1} deg)",
        world.launch.name,
        world.launch.lat.to_degrees(),
        world.launch.lon.to_degrees(),
    );

    let r = &worldgen::render::render_globe;
    let _ = r; // (silence if unused in some configs)

    worldgen::render::write_height_map(&world, "out/heightmap.png");
    worldgen::render::write_day_map(&world, "out/day_albedo.png");
    worldgen::render::write_night_map(&world, "out/night_lights.png");
    worldgen::render::write_cities_roads_map(&world, "out/cities_roads.png");
    worldgen::render::write_globe_day(&world, "out/globe_day.png", 1100);
    worldgen::render::write_globe_night(&world, "out/globe_night.png", 1100);

    println!("wrote previews to out/*.png in {:?}", t0.elapsed());
}
