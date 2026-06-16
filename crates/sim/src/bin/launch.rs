//! Stack the Pioneer I, launch from the seed-47 spaceport, and reach orbit.
//!
//! Run: cargo run -p sim --bin launch --release

use sim::ascent::simulate;
use sim::body::CentralBody;
use sim::vehicle::Vehicle;

fn main() {
    let body = CentralBody::home();
    let veh = Vehicle::pioneer();

    // Seed-47 spaceport coordinates (from worldgen).
    let lat = -1.7;
    let lon = -102.9;
    let target_apo = 200_000.0; // 200 km

    let twr = veh.stages[0].thrust / (veh.liftoff_mass() * body.surface_gravity());
    println!("== Vehicle: {} ==", veh.name);
    println!("liftoff mass:   {:.1} t", veh.liftoff_mass() / 1000.0);
    println!("liftoff TWR:    {:.2}", twr);
    let mut upper = veh.payload;
    for (i, s) in veh.stages.iter().enumerate().rev() {
        println!("stage {} ({:<7}) dv: {:.0} m/s", i + 1, s.name, s.dv(upper));
        upper += s.wet();
    }
    println!(
        "ideal total dv: {:.0} m/s",
        {
            let mut up = veh.payload;
            let mut tot = 0.0;
            for s in veh.stages.iter().rev() {
                tot += s.dv(up);
                up += s.wet();
            }
            tot
        }
    );

    let res = simulate(&body, &veh, lat, lon, target_apo);

    println!("\n== Ascent ==");
    for (t, e) in &res.events {
        println!("  t+{:>6.1}s  {}", t, e);
    }

    if let Some(m) = res.meco {
        println!("\nMECO:           t+{:.0}s  alt {:.1} km  v {:.0} m/s  downrange {:.0} km",
            m.t, m.alt / 1000.0, m.speed, m.downrange / 1000.0);
    }
    println!("ascent dv used: {:.0} m/s (gravity+drag+steering losses included)", res.ascent_dv);
    println!("circ burn dv:   {:.0} m/s  (prop used {:.1} t, {:.1} t left)",
        res.circ_dv, res.circ_prop_used / 1000.0, res.prop_left_after_circ / 1000.0);

    let o = &res.final_orbit;
    println!("\n== Result ==");
    println!("orbit:          peri {:.0} km  x  apo {:.0} km",
        (o.rp - body.radius) / 1000.0, (o.ra - body.radius) / 1000.0);
    println!("eccentricity:   {:.4}", o.e);
    if let Some(p) = o.period {
        println!("period:         {:.1} min", p / 60.0);
    }
    println!("REACHED ORBIT:  {}", if res.reached_orbit { "YES" } else { "NO" });

    sim::plot::write_launch_plot(&body, &res, "out/launch.png");
    println!("\nwrote out/launch.png");
}
