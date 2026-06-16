//! Verify the LOD terrain: select patches for a camera near the surface, print
//! the LOD statistics and geometry counts, and render CPU proof images.
//!
//! Run: cargo run -p terrain --bin lod --release

use glam::DVec3;
use terrain::cubesphere::face_dir;
use terrain::{build_mesh, select, Elevation, Planet};

fn cam_at(planet: &Planet, lat_deg: f64, lon_deg: f64, alt_m: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let dir = DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin());
    dir.normalize() * (planet.radius + alt_m)
}

fn report(planet: &Planet, elev: &Elevation, label: &str, cam: DVec3, extent_m: f64, tag: &str) {
    let split_factor = 2.0;
    let max_depth = 18;
    let n = 17; // vertices per patch edge

    let lod = select(planet, cam, split_factor, max_depth);
    let alt = cam.length() - planet.radius;

    println!("\n== {label} (alt {:.0} m) ==", alt);
    println!("active patches:   {}", lod.patches.len());
    println!("max depth:        {}", lod.max_depth_reached);
    print!("patches/depth:    ");
    for (d, c) in lod.per_depth.iter().enumerate() {
        if *c > 0 {
            print!("L{d}={c} ");
        }
    }
    println!();
    let finest = lod
        .patches
        .iter()
        .map(|p| p.edge)
        .fold(f64::INFINITY, f64::min);
    let coarsest = lod.patches.iter().map(|p| p.edge).fold(0.0, f64::max);
    println!("patch edge range: {:.1} m (finest) .. {:.0} m (coarsest)", finest, coarsest);
    println!("triangles @ n={n}: {}", lod.triangle_count(n));

    // verify crack-free intent: build the finest patch mesh and report counts +
    // that the skirt added rim geometry.
    if let Some(p) = lod.patches.iter().min_by(|a, b| a.edge.partial_cmp(&b.edge).unwrap()) {
        let mesh = build_mesh(planet, p, n, elev, 60.0);
        println!(
            "finest patch mesh: {} verts, {} tris (incl. skirt)",
            mesh.positions.len(),
            mesh.indices.len() / 3
        );
    }

    terrain::render::write_lod_map(planet, &lod, cam, extent_m, 900, &format!("out/lod_{tag}_map.png"));
    terrain::render::write_relief(planet, elev, cam, extent_m, 900, &format!("out/lod_{tag}_relief.png"));
    println!("wrote out/lod_{tag}_map.png and out/lod_{tag}_relief.png");
}

fn main() {
    std::fs::create_dir_all("out").unwrap();
    let planet = Planet { radius: 6.2e6 };
    let elev = Elevation::new(47);

    // sanity: elevation is continuous and varied
    let h0 = elev.height_m(face_dir(4, 0.1, 0.1));
    println!("sample elevation: {:.1} m", h0);

    // Near the surface: the quadtree should refine to tens-of-metres patches.
    let ground = cam_at(&planet, -1.7, -102.9, 2_500.0);
    report(&planet, &elev, "Surface / rocket view", ground, 30_000.0, "surface");

    // Higher up: coarser LOD over a wider area.
    let high = cam_at(&planet, -1.7, -102.9, 120_000.0);
    report(&planet, &elev, "High altitude", high, 800_000.0, "high");
}
