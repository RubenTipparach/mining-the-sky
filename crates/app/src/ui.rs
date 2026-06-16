//! egui UI: the game's panels (vehicle assembly, ascent telemetry, manual
//! flight, and the orbital-map body browser) plus a hard-sci-fi theme. This
//! replaces the hand-drawn bitmap-font HUD text; in-scene markers/dots and
//! trajectory lines still go through the overlay pipeline.

use crate::flight::Mode;
use crate::universe::Kind;
use crate::{View, World};

const AMBER: egui::Color32 = egui::Color32::from_rgb(255, 196, 80);
const GOOD: egui::Color32 = egui::Color32::from_rgb(120, 230, 140);
const WARN: egui::Color32 = egui::Color32::from_rgb(255, 150, 60);
const DIM: egui::Color32 = egui::Color32::from_rgb(150, 175, 200);

pub fn build(ctx: &egui::Context, world: &mut World) {
    apply_theme(ctx);
    match world.view {
        View::Rocket => vehicle_panel(ctx, world),
        View::Map => {
            body_browser(ctx, world);
            status_panel(ctx, world);
        }
    }
}

fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    let mut v = egui::Visuals::dark();
    let panel = egui::Color32::from_rgb(13, 19, 29); // opaque (fixes faintness)
    v.window_fill = panel;
    v.panel_fill = panel;
    v.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 110, 150));
    v.window_shadow = egui::epaint::Shadow::NONE;
    v.override_text_color = Some(egui::Color32::from_rgb(210, 228, 245));
    v.widgets.noninteractive.bg_fill = panel;
    v.widgets.inactive.bg_fill = egui::Color32::from_rgb(28, 40, 56);
    v.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(24, 34, 48);
    v.widgets.hovered.bg_fill = egui::Color32::from_rgb(42, 74, 104);
    v.widgets.active.bg_fill = egui::Color32::from_rgb(52, 98, 138);
    v.selection.bg_fill = egui::Color32::from_rgb(36, 86, 132);
    v.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(130, 190, 235));
    style.visuals = v;
    style.spacing.item_spacing = egui::vec2(8.0, 4.0);
    // No fade-in: windows must be fully opaque immediately. Without this the
    // Area/Window opening animation leaves panels translucent for the first
    // frames (which is exactly the "super faint" UI), and headless shots only
    // ever render those first frames.
    style.animation_time = 0.0;
    ctx.set_style(style);
}

fn kv(ui: &mut egui::Ui, k: &str, v: &str) {
    ui.label(egui::RichText::new(k).color(DIM));
    ui.label(v);
    ui.end_row();
}

fn vehicle_panel(ctx: &egui::Context, world: &mut World) {
    let m = &world.mission;
    let vehicle = m.vehicle;
    let stack: Vec<(usize, String, f32, f32)> = m
        .stack
        .iter()
        .enumerate()
        .map(|(i, (n, w, d))| (i, n.to_string(), *w, *d))
        .collect();
    let (mass, twr, dv, pay, target) =
        (m.liftoff_mass_t, m.liftoff_twr, m.total_dv, m.payload_t, m.target_orbit_km());
    let launched = world.launched;
    let mut do_launch = false;

    egui::Window::new("VEHICLE ASSEMBLY")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(250.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new(vehicle).heading().color(AMBER));
            ui.add_space(2.0);
            egui::Grid::new("stack").num_columns(3).striped(true).show(ui, |ui| {
                for (i, n, w, d) in stack.iter().rev() {
                    ui.label(format!("S{} {}", i + 1, n));
                    ui.label(format!("{w:.0} t"));
                    ui.label(egui::RichText::new(format!("{d:.0} m/s")).color(DIM));
                    ui.end_row();
                }
            });
            ui.separator();
            egui::Grid::new("stats").num_columns(2).show(ui, |ui| {
                kv(ui, "Mass", &format!("{mass:.0} t"));
                kv(ui, "Liftoff TWR", &format!("{twr:.2}"));
                kv(ui, "Total delta-v", &format!("{dv:.0} m/s"));
                kv(ui, "Payload", &format!("{pay:.0} t"));
                kv(ui, "Target orbit", &format!("{target:.0} km"));
            });
            ui.separator();
            if !launched {
                let btn = egui::Button::new(egui::RichText::new("LAUNCH").strong().color(egui::Color32::BLACK))
                    .fill(GOOD)
                    .min_size(egui::vec2(120.0, 26.0));
                if ui.add(btn).clicked() {
                    do_launch = true;
                }
                ui.label(egui::RichText::new("or press Space").color(DIM));
            } else {
                ui.label(egui::RichText::new("Lifted off - Tab to the map to watch the ascent").color(GOOD));
            }
            ui.separator();
            ui.label(egui::RichText::new("Drag: orbit   Scroll: zoom   Tab: map").color(DIM));
        });

    if do_launch {
        world.toggle_launch();
    }
}

fn status_panel(ctx: &egui::Context, world: &mut World) {
    if world.flight.is_some() {
        flight_panel(ctx, world);
    } else if world.launched {
        telemetry_panel(ctx, world);
    }
}

fn telemetry_panel(ctx: &egui::Context, world: &mut World) {
    let tel = world.mission.telemetry(world.launched, world.clock);
    let (clock, warp, stage_count) = (world.clock.max(0.0), world.warp, world.mission.stage_count);
    let phase_col = match tel.phase {
        "ASCENT" => WARN,
        "ORBIT" => egui::Color32::from_rgb(120, 200, 255),
        _ => GOOD,
    };
    let mut warp_mul = 1.0f32;
    let mut reset = false;
    let mut manual = false;

    egui::Window::new("ASCENT")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(230.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("PHASE").color(DIM));
                ui.label(egui::RichText::new(tel.phase).strong().color(phase_col));
            });
            egui::Grid::new("tel").num_columns(2).show(ui, |ui| {
                kv(ui, "MET", &format!("T+{clock:.0} s"));
                kv(ui, "Altitude", &format!("{:.1} km", tel.alt_km));
                kv(ui, "Velocity", &format!("{:.0} m/s", tel.speed));
                if let Some((peri, apo)) = tel.orbit {
                    kv(ui, "Orbit", &format!("{peri:.0} x {apo:.0} km"));
                } else {
                    kv(ui, "Downrange", &format!("{:.0} km", tel.downrange_km));
                }
                kv(ui, "Stage", &format!("{}/{}", tel.stage + 1, stage_count));
            });
            ui.separator();
            time_controls(ui, warp, &mut warp_mul);
            ui.horizontal(|ui| {
                if ui.button("Take control (F)").clicked() {
                    manual = true;
                }
                if ui.button("Reset (Space)").clicked() {
                    reset = true;
                }
            });
        });

    world.warp = (world.warp * warp_mul).clamp(1.0, 10000.0);
    if reset {
        world.toggle_launch();
    }
    if manual {
        world.toggle_flight();
    }
}

fn flight_panel(ctx: &egui::Context, world: &mut World) {
    let craft = world.flight.as_ref().unwrap();
    let status = craft.status();
    let scol = match status {
        "CRASHED" => egui::Color32::from_rgb(255, 80, 70),
        "LANDED" => GOOD,
        _ => AMBER,
    };
    let (alt, spd, vspd, thr, prop, mode) = (
        craft.altitude(&world.body) / 1000.0,
        craft.speed(),
        craft.vertical_speed(),
        craft.throttle * 100.0,
        craft.prop_frac() * 100.0,
        craft.mode,
    );
    let warp = world.warp;

    #[derive(PartialEq)]
    enum Act {
        Thr(f64),
        Mode(Mode),
        Release,
    }
    let mut act: Option<Act> = None;
    let mut warp_mul = 1.0f32;

    egui::Window::new("MANUAL FLIGHT")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(240.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("STATUS").color(DIM));
                ui.label(egui::RichText::new(status).strong().color(scol));
            });
            egui::Grid::new("fl").num_columns(2).show(ui, |ui| {
                kv(ui, "Altitude", &format!("{alt:.1} km"));
                kv(ui, "Velocity", &format!("{spd:.0} m/s"));
                kv(ui, "Vert speed", &format!("{vspd:.0} m/s"));
                kv(ui, "Throttle", &format!("{thr:.0} %"));
                kv(ui, "Propellant", &format!("{prop:.0} %"));
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Throttle").color(DIM));
                if ui.button("-").clicked() {
                    act = Some(Act::Thr(-0.08));
                }
                if ui.button("+").clicked() {
                    act = Some(Act::Thr(0.08));
                }
            });
            ui.horizontal(|ui| {
                for (lbl, m) in [
                    ("Pro", Mode::Prograde),
                    ("Retro", Mode::Retrograde),
                    ("Out", Mode::RadialOut),
                    ("In", Mode::RadialIn),
                ] {
                    if ui.selectable_label(mode == m, lbl).clicked() {
                        act = Some(Act::Mode(m));
                    }
                }
            });
            ui.separator();
            time_controls(ui, warp, &mut warp_mul);
            if ui.button("Release control (F)").clicked() {
                act = Some(Act::Release);
            }
        });

    world.warp = (world.warp * warp_mul).clamp(1.0, 10000.0);
    match act {
        Some(Act::Release) => world.toggle_flight(),
        Some(Act::Thr(d)) => {
            if let Some(c) = world.flight.as_mut() {
                c.throttle = (c.throttle + d).clamp(0.0, 1.0);
            }
        }
        Some(Act::Mode(m)) => {
            if let Some(c) = world.flight.as_mut() {
                c.mode = m;
            }
        }
        None => {}
    }
}

fn time_controls(ui: &mut egui::Ui, warp: f32, warp_mul: &mut f32) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Time").color(DIM));
        if ui.button("<<").clicked() {
            *warp_mul = 0.5;
        }
        ui.label(egui::RichText::new(format!("{warp:.0}x")).strong());
        if ui.button(">>").clicked() {
            *warp_mul = 2.0;
        }
    });
}

fn body_browser(ctx: &egui::Context, world: &mut World) {
    let bodies: Vec<(usize, String, Kind)> = world
        .universe
        .bodies
        .iter()
        .enumerate()
        .map(|(i, b)| (i, b.name.clone(), b.kind))
        .collect();
    let focus = world.focus;
    let focus_name = world.focus_label().to_string();
    let cam_dist = world.sys_dist;
    let days = world.sys_time / 86_400.0;
    let warp = world.warp;

    let mut to_focus: Option<usize> = None;
    let mut warp_mul = 1.0f32;

    {
        let search = &mut world.ui_search;
        egui::Window::new("SYSTEM BODIES")
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 12.0))
            .default_width(220.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Find").color(DIM));
                    ui.text_edit_singleline(search);
                    if ui.button("x").clicked() {
                        search.clear();
                    }
                });
                let q = search.to_uppercase();
                let mut group = |ui: &mut egui::Ui, title: &str, kinds: &[Kind]| {
                    let items: Vec<&(usize, String, Kind)> = bodies
                        .iter()
                        .filter(|(_, name, k)| kinds.contains(k) && name.to_uppercase().contains(&q))
                        .collect();
                    if items.is_empty() {
                        return;
                    }
                    egui::CollapsingHeader::new(format!("{title} ({})", items.len()))
                        .default_open(!q.is_empty() || title == "Planets")
                        .show(ui, |ui| {
                            for (i, name, _) in items {
                                if ui.selectable_label(*i == focus, name).clicked() {
                                    to_focus = Some(*i);
                                }
                            }
                        });
                };

                egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                    group(ui, "Stars", &[Kind::StarA, Kind::StarB]);
                    group(ui, "Planets", &[Kind::Planet]);
                    group(ui, "Moons", &[Kind::Moon]);
                    group(ui, "Asteroids", &[Kind::AsteroidMajor, Kind::AsteroidMinor]);
                    group(ui, "Comets", &[Kind::Comet]);
                });

                ui.separator();
                egui::Grid::new("mapinfo").num_columns(2).show(ui, |ui| {
                    kv(ui, "Focus", &focus_name);
                    kv(ui, "Cam dist", &format!("{:.0} Mm", cam_dist));
                    kv(ui, "Elapsed", &format!("{days:.1} d"));
                });
                time_controls(ui, warp, &mut warp_mul);
                ui.label(egui::RichText::new("Click a body or the scene to focus").color(DIM));
            });
    }

    world.warp = (world.warp * warp_mul).clamp(1.0, 10000.0);
    if let Some(i) = to_focus {
        world.set_focus(i);
    }
}
