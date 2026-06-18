//! Headless verification of the attitude autopilot + rotational dynamics.
//!
//! Runs a maneuvering rig through a sequence of large slews (a 180 deg flip,
//! then two 90 deg turns), driving it with the quaternion-PID autopilot and the
//! gimbal/reaction-wheel/RCS allocator, and plots the telemetry so the
//! controller's behaviour is verifiable without a GPU:
//!
//!   top panel:    pointing error (deg, cyan)   and body rate (deg/s, orange)
//!   bottom panel: reaction-wheel saturation (%, yellow) and RCS propellant (%, green)
//!
//! Run: cargo run -p sim --bin attitude  (writes out/attitude.png)

use glam::DVec3;
use image::{Rgb, RgbImage};
use sim::attitude::{
    allocate, AttitudeController, Gimbal, Rcs, ReactionWheels, RigidBody,
};

fn line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, col: Rgb<u8>) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    loop {
        if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
            img.put_pixel(x as u32, y as u32, col);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draw a series (values already normalised to 0..1) into a panel rect.
fn plot_series(
    img: &mut RgbImage,
    rect: (i32, i32, i32, i32),
    vals: &[f64],
    col: Rgb<u8>,
) {
    let (x0, y0, w, h) = rect;
    let n = vals.len();
    if n < 2 {
        return;
    }
    let mut prev: Option<(i32, i32)> = None;
    for (i, &v) in vals.iter().enumerate() {
        let x = x0 + (i as f64 / (n - 1) as f64 * w as f64) as i32;
        let y = y0 + h - (v.clamp(0.0, 1.0) * h as f64) as i32;
        if let Some((px, py)) = prev {
            line(img, px, py, x, y, col);
        }
        prev = Some((x, y));
    }
}

fn panel_frame(img: &mut RgbImage, rect: (i32, i32, i32, i32)) {
    let (x0, y0, w, h) = rect;
    let g = Rgb([40, 48, 64]);
    // gridlines (4 horizontal)
    for k in 0..=4 {
        let y = y0 + h * k / 4;
        line(img, x0, y, x0 + w, y, g);
    }
    let border = Rgb([90, 110, 140]);
    line(img, x0, y0, x0 + w, y0, border);
    line(img, x0, y0 + h, x0 + w, y0 + h, border);
    line(img, x0, y0, x0, y0 + h, border);
    line(img, x0 + w, y0, x0 + w, y0 + h, border);
}

fn main() {
    // Maneuvering rig (matches the flight Craft): ~14 t can, wheels + RCS + gimbal.
    let mut rb = RigidBody::cylinder(14_000.0, 2.0, 4.0);
    rb.point_at(DVec3::X);
    let mut wheels = ReactionWheels::new(800.0, 250.0);
    let mut rcs = Rcs::new(4_000.0, 230.0, 2.5, 200.0);
    let gimbal = Gimbal::new(5.0, 6.0);
    let mut ctrl = AttitudeController::new();

    // Slew schedule: (start_time_s, target world direction).
    let schedule = [
        (0.0_f64, DVec3::X),       // hold
        (3.0, -DVec3::X),          // 180 deg flip (prograde -> retrograde)
        (45.0, DVec3::Y),          // 90 deg (-> normal)
        (85.0, DVec3::Z),          // 90 deg (-> radial)
    ];

    let dt = 0.05;
    let total = 130.0;
    let steps = (total / dt) as usize;
    let mut err_s = Vec::with_capacity(steps);
    let mut rate_s = Vec::with_capacity(steps);
    let mut sat_s = Vec::with_capacity(steps);
    let mut rcs_s = Vec::with_capacity(steps);
    let mut switch_x = Vec::new();

    let mut last_target = schedule[0].1;
    let mut sched_i = 0usize;
    let mut peak_err = 0.0f64;
    let mut settle_err = 0.0f64;

    for i in 0..steps {
        let t = i as f64 * dt;
        // advance the schedule
        while sched_i + 1 < schedule.len() && t >= schedule[sched_i + 1].0 {
            sched_i += 1;
            last_target = schedule[sched_i].1;
            switch_x.push(i);
        }
        let target = Some(last_target);
        let cmd = ctrl.command_torque(&rb, target, dt);
        // no main-engine thrust in this attitude-only demo, so gimbal is idle
        let (tau, _rep) = allocate(cmd, &mut wheels, &mut rcs, gimbal, 0.0, dt);
        rb.integrate(tau, dt);

        let err = AttitudeController::error(&rb, last_target).length().to_degrees();
        let rate = rb.omega.length().to_degrees();
        err_s.push(err);
        rate_s.push(rate);
        sat_s.push(wheels.saturation());
        rcs_s.push(rcs.prop_frac());
        peak_err = peak_err.max(rate); // peak slew rate
        settle_err = err; // last value
    }

    // ---- render the plot ----
    let (w, h) = (1000u32, 640u32);
    let mut img = RgbImage::from_pixel(w, h, Rgb([6, 8, 14]));

    let pad = 40i32;
    let pw = w as i32 - 2 * pad;
    let ph = (h as i32 - 3 * pad) / 2;
    let top = (pad, pad, pw, ph);
    let bot = (pad, 2 * pad + ph, pw, ph);
    panel_frame(&mut img, top);
    panel_frame(&mut img, bot);

    // mode-switch markers (white verticals on both panels)
    for &sx in &switch_x {
        let x = pad + (sx as f64 / (steps - 1) as f64 * pw as f64) as i32;
        line(&mut img, x, top.1, x, top.1 + top.3, Rgb([120, 120, 140]));
        line(&mut img, x, bot.1, x, bot.1 + bot.3, Rgb([120, 120, 140]));
    }

    // top: error (0..180 deg -> 0..1) cyan, rate (0..30 deg/s -> 0..1) orange
    let err_n: Vec<f64> = err_s.iter().map(|e| e / 180.0).collect();
    let rate_n: Vec<f64> = rate_s.iter().map(|r| r / 30.0).collect();
    plot_series(&mut img, top, &err_n, Rgb([90, 210, 255]));
    plot_series(&mut img, top, &rate_n, Rgb([255, 170, 70]));

    // bottom: wheel saturation (%) yellow, RCS prop (%) green
    plot_series(&mut img, bot, &sat_s, Rgb([240, 220, 80]));
    plot_series(&mut img, bot, &rcs_s, Rgb([120, 230, 140]));

    // legend swatches (top-left of each panel)
    let swatch = |img: &mut RgbImage, x: i32, y: i32, col: Rgb<u8>| {
        for dy in 0..6 {
            for dx in 0..18 {
                img.put_pixel((x + dx) as u32, (y + dy) as u32, col);
            }
        }
    };
    swatch(&mut img, pad + 6, pad + 6, Rgb([90, 210, 255])); // error
    swatch(&mut img, pad + 6, pad + 16, Rgb([255, 170, 70])); // rate
    swatch(&mut img, pad + 6, 2 * pad + ph + 6, Rgb([240, 220, 80])); // wheels
    swatch(&mut img, pad + 6, 2 * pad + ph + 16, Rgb([120, 230, 140])); // rcs

    std::fs::create_dir_all("out").ok();
    img.save("out/attitude.png").unwrap();

    println!("attitude autopilot demo:");
    println!("  peak slew rate : {peak_err:.2} deg/s");
    println!("  final pointing error after last slew : {settle_err:.3} deg");
    println!("  reaction-wheel peak saturation : {:.0} %", sat_s.iter().cloned().fold(0.0, f64::max) * 100.0);
    println!("  RCS propellant remaining : {:.0} %", rcs.prop_frac() * 100.0);
    println!("  wrote out/attitude.png (top: error cyan / rate orange; bottom: wheels yellow / RCS green)");
}
