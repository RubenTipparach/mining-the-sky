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
        View::Rocket => {
            if world.space {
                asteroid_panel(ctx, world); // inspecting an asteroid
            } else if world.base_mesh.is_some() && world.base_panel {
                moonbase_panel(ctx, world); // surveying the colony
            } else if world.base_mesh.is_some() {
                // a single delivered cargo module on the surface: no panel
            } else if world.show_lander {
                lander_panel(ctx, world); // on the lunar surface
            } else if world.launch.is_some() {
                launch_panel(ctx, world);
            } else if world.rolling_out {
                rollout_panel(ctx, world); // crawling out to the pad
            } else if world.vab_mode {
                vehicle_panel(ctx, world); // assembling in the building
            } else {
                pad_panel(ctx, world); // rolled out, ready to launch
            }
            if world.lod_debug && !world.space && world.ast_elev.is_none() {
                lod_debug_panel(ctx, world);
            }
        }
        View::Map => {
            body_browser(ctx, world);
            status_panel(ctx, world);
            maneuver_node_panel(ctx, world);
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

/// Key/value row with a coloured value (for status that changes colour).
fn kv_c(ui: &mut egui::Ui, k: &str, v: &str, col: egui::Color32) {
    ui.label(egui::RichText::new(k).color(DIM));
    ui.label(egui::RichText::new(v).color(col));
    ui.end_row();
}

fn vehicle_panel(ctx: &egui::Context, world: &mut World) {
    use crate::build;
    // Drag payload: which catalog part is being dragged.
    #[derive(Clone, Copy, PartialEq)]
    enum Drag {
        Engine(usize),
        Tank(usize),
        Payload(usize),
    }

    let mut vab = world.vab.clone();
    let mut changed = false;
    let mut launch = false;
    enum Act {
        Remove(usize),
        Add,
    }
    let mut act: Option<Act> = None;
    let n_orbit = world.orbits.len();
    let g = world.body.surface_gravity();

    egui::Window::new("VEHICLE ASSEMBLY")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(330.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("VEHICLE ASSEMBLY").heading().color(AMBER));
            ui.label(egui::RichText::new("Drag parts onto the stack").color(DIM));
            ui.add_space(2.0);

            // ---- the stack: a drop slot per stage (engine + tank), top first ----
            for i in (0..vab.stages.len()).rev() {
                let (ei, ti) = (vab.stages[i].engine, vab.stages[i].tank);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("S{}", i + 1)).color(DIM).monospace());

                    // engine slot
                    let (_, drop) = ui.dnd_drop_zone::<Drag, ()>(slot_frame(), |ui| {
                        ui.set_min_size(egui::vec2(96.0, 22.0));
                        ui.label(egui::RichText::new(build::ENGINES[ei].name).color(GOOD));
                    });
                    if let Some(p) = drop {
                        if let Drag::Engine(k) = *p {
                            vab.stages[i].engine = k;
                            changed = true;
                        }
                    }
                    // tank slot
                    let (_, drop) = ui.dnd_drop_zone::<Drag, ()>(slot_frame(), |ui| {
                        ui.set_min_size(egui::vec2(86.0, 22.0));
                        ui.label(egui::RichText::new(build::TANKS[ti].name).color(egui::Color32::from_rgb(150, 200, 255)));
                    });
                    if let Some(p) = drop {
                        if let Drag::Tank(k) = *p {
                            vab.stages[i].tank = k;
                            changed = true;
                        }
                    }
                    if vab.stages.len() > 1 && ui.button("x").clicked() {
                        act = Some(Act::Remove(i));
                    }
                });
                // radial-booster controls for this stage: count stepper + type.
                ui.horizontal(|ui| {
                    let bn = vab.stages[i].boosters;
                    let bt = vab.stages[i].booster;
                    ui.label(egui::RichText::new("   radial").color(DIM).monospace().small());
                    if ui.small_button("-").clicked() && bn > 0 {
                        vab.stages[i].boosters = bn - 1;
                        changed = true;
                    }
                    ui.label(egui::RichText::new(format!("{bn}x")).color(if bn > 0 { GOOD } else { DIM }).monospace());
                    if ui.small_button("+").clicked() && bn < build::MAX_BOOSTERS {
                        vab.stages[i].boosters = bn + 1;
                        changed = true;
                    }
                    let b = build::BOOSTERS[bt];
                    let tag = if b.solid { "SRB" } else { "Liq" };
                    if ui
                        .button(egui::RichText::new(format!("{} [{}]", b.name, tag)).small())
                        .on_hover_text("click to cycle booster type")
                        .clicked()
                    {
                        vab.stages[i].booster = (bt + 1) % build::BOOSTERS.len();
                        changed = true;
                    }
                });
            }
            // payload slot
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("PL").color(DIM).monospace());
                let pi = vab.payload;
                let (_, drop) = ui.dnd_drop_zone::<Drag, ()>(slot_frame(), |ui| {
                    ui.set_min_size(egui::vec2(190.0, 22.0));
                    ui.label(egui::RichText::new(build::PAYLOADS[pi].name).color(AMBER));
                });
                if let Some(p) = drop {
                    if let Drag::Payload(k) = *p {
                        vab.payload = k;
                        changed = true;
                    }
                }
            });
            if ui.button("+ stage").clicked() {
                act = Some(Act::Add);
            }

            ui.separator();
            // ---- parts palette: draggable chips ----
            ui.label(egui::RichText::new("ENGINES").color(DIM).small());
            ui.horizontal_wrapped(|ui| {
                for (k, e) in build::ENGINES.iter().enumerate() {
                    drag_chip(ui, egui::Id::new(("eng", k)), Drag::Engine(k), e.name, GOOD);
                }
            });
            ui.label(egui::RichText::new("TANKS").color(DIM).small());
            ui.horizontal_wrapped(|ui| {
                for (k, t) in build::TANKS.iter().enumerate() {
                    drag_chip(ui, egui::Id::new(("tank", k)), Drag::Tank(k), t.name, egui::Color32::from_rgb(150, 200, 255));
                }
            });
            ui.label(egui::RichText::new("PAYLOADS").color(DIM).small());
            ui.horizontal_wrapped(|ui| {
                for (k, p) in build::PAYLOADS.iter().enumerate() {
                    drag_chip(ui, egui::Id::new(("pl", k)), Drag::Payload(k), p.name, AMBER);
                }
            });

            ui.separator();
            let veh = vab.to_vehicle();
            let mass_t = veh.liftoff_mass() / 1000.0;
            let twr = veh.stages[0].thrust / (veh.liftoff_mass() * g);
            let total_dv: f64 =
                (0..veh.stages.len()).map(|i| veh.stages[i].dv(veh.mass_above(i))).sum();
            egui::Grid::new("vab_stats").num_columns(2).show(ui, |ui| {
                kv(ui, "Liftoff mass", &format!("{mass_t:.0} t"));
                ui.label(egui::RichText::new("Liftoff TWR").color(DIM));
                ui.label(egui::RichText::new(format!("{twr:.2}")).color(if twr < 1.0 { WARN } else { GOOD }));
                ui.end_row();
                kv(ui, "Total delta-v", &format!("{total_dv:.0} m/s"));
                kv(ui, "Payload", &format!("{:.0} kg", vab.payload().mass));
            });
            if twr < 1.0 {
                ui.label(egui::RichText::new("TWR < 1: add engines or drop tankage").color(WARN));
            }

            ui.separator();
            let btn = egui::Button::new(
                egui::RichText::new("ROLL OUT TO PAD").strong().color(egui::Color32::BLACK),
            )
            .fill(AMBER)
            .min_size(egui::vec2(150.0, 26.0));
            if ui.add(btn).clicked() {
                launch = true; // (roll-out, handled below)
            }
            ui.label(egui::RichText::new("Drag parts onto the stack  -  drag to rotate").color(DIM));
            let col = if n_orbit > 0 { GOOD } else { DIM };
            ui.label(egui::RichText::new(format!("Satellites in orbit: {n_orbit}")).color(col));
        });

    match act {
        Some(Act::Remove(i)) => {
            if vab.stages.len() > 1 {
                vab.stages.remove(i);
                changed = true;
            }
        }
        Some(Act::Add) => {
            vab.stages.push(build::StageCfg::new(3, 0));
            changed = true;
        }
        None => {}
    }
    if changed {
        world.vab = vab;
        world.rebuild_vehicle();
    }
    if launch {
        world.start_rollout();
    }
}

/// Shown once the vehicle has rolled out to the pad: final stats + LAUNCH.
fn pad_panel(ctx: &egui::Context, world: &mut World) {
    let veh = world.vab.to_vehicle();
    let g = world.body.surface_gravity();
    let mass_t = veh.liftoff_mass() / 1000.0;
    let twr = veh.stages[0].thrust / (veh.liftoff_mass() * g);
    let total_dv: f64 = (0..veh.stages.len()).map(|i| veh.stages[i].dv(veh.mass_above(i))).sum();
    let rolling = world.rolling_out;
    let mut launch = false;
    let mut back = false;

    egui::Window::new("LAUNCH PAD")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(240.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("LAUNCH PAD").heading().color(AMBER));
            egui::Grid::new("pad_stats").num_columns(2).show(ui, |ui| {
                kv(ui, "Vehicle", veh.name);
                kv(ui, "Liftoff mass", &format!("{mass_t:.0} t"));
                kv(ui, "Liftoff TWR", &format!("{twr:.2}"));
                kv(ui, "Total delta-v", &format!("{total_dv:.0} m/s"));
                kv(ui, "Payload", &format!("{:.0} kg", world.vab.payload().mass));
            });
            ui.separator();
            if rolling {
                ui.label(egui::RichText::new("Rolling out...").color(WARN));
            } else {
                let btn = egui::Button::new(
                    egui::RichText::new("LAUNCH").strong().color(egui::Color32::BLACK),
                )
                .fill(GOOD)
                .min_size(egui::vec2(120.0, 26.0));
                if ui.add(btn).clicked() {
                    launch = true;
                }
                ui.label(egui::RichText::new("Space ignite  Shift/Ctrl throttle  W/S pitch").color(DIM));
                if ui.button("Back to VAB").clicked() {
                    back = true;
                }
            }
        });

    if launch {
        world.ignite_launch();
    }
    if back {
        world.back_to_vab();
    }
}

/// Shown while the crawler is hauling the stack out of the assembly building to
/// the pad: roll-out progress plus a speed control so the player can fast-forward
/// the slow transport instead of watching it creep.
fn rollout_panel(ctx: &egui::Context, world: &mut World) {
    let rollout = world.rollout;
    let speed = world.rollout_speed;
    // Some(true) = crank crawler faster, Some(false) = slower (handled below).
    let mut bump: Option<bool> = None;
    let mut skip = false;

    egui::Window::new("ROLL OUT")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(240.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("ROLLING OUT TO PAD").heading().color(AMBER));
            let pct = (rollout * 100.0).round() as i32;
            ui.label(egui::RichText::new(format!("{pct}%  -  crawler on the way")).color(DIM));
            ui.add(egui::ProgressBar::new(rollout).fill(AMBER).desired_height(10.0));
            ui.separator();
            // Speed stepper, mirroring the radial-booster control in the VAB.
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Crawler speed").color(DIM));
                if ui.small_button("-").clicked() {
                    bump = Some(false);
                }
                ui.label(egui::RichText::new(format!("{speed:.0}x")).color(GOOD).monospace());
                if ui.small_button("+").clicked() {
                    bump = Some(true);
                }
            });
            ui.label(egui::RichText::new("Keys , and . also adjust  ([ ] = time warp)").color(DIM).small());
            ui.separator();
            let btn = egui::Button::new(
                egui::RichText::new("SKIP TO PAD").strong().color(egui::Color32::BLACK),
            )
            .fill(AMBER)
            .min_size(egui::vec2(150.0, 24.0));
            if ui.add(btn).clicked() {
                skip = true;
            }
        });

    if let Some(faster) = bump {
        world.bump_rollout_speed(faster);
    }
    if skip {
        world.skip_rollout();
    }
}

/// Shown on the lunar surface: a compact lander status readout. No launch
/// controls (this is the descent/landed view, not the pad).
fn lander_panel(ctx: &egui::Context, world: &mut World) {
    let landed = world.lander_alt <= 0.1;
    egui::Window::new("LUNAR LANDER")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(220.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("LUNAR LANDER").heading().color(AMBER));
            egui::Grid::new("lander_stats").num_columns(2).show(ui, |ui| {
                kv(ui, "Body", "Moon");
                kv(ui, "Altitude", &format!("{:.1} m", world.lander_alt.max(0.0)));
                kv(
                    ui,
                    "Descent engine",
                    if world.lander_firing { "FIRING" } else { "OFF" },
                );
            });
            ui.separator();
            if landed && !world.lander_firing {
                ui.label(egui::RichText::new("TOUCHDOWN").strong().color(GOOD));
            } else {
                ui.label(egui::RichText::new("Powered descent").color(WARN));
            }
        });
}

/// A small readout while inspecting an asteroid in deep space.
fn asteroid_panel(ctx: &egui::Context, world: &mut World) {
    let name = if world.space_label.is_empty() { "ASTEROID" } else { world.space_label };
    egui::Window::new("ASTEROID")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(200.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new(name).heading().color(AMBER));
            ui.label(egui::RichText::new("Minor body - C/S-type rubble").color(DIM));
        });
}

/// Shown when surveying the moon base: the buildable structures catalog.
fn moonbase_panel(ctx: &egui::Context, _world: &mut World) {
    egui::Window::new("MOON BASE")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(250.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("MOON BASE").heading().color(AMBER));
            ui.label(egui::RichText::new("Buildable structures").color(DIM));
            ui.separator();
            egui::Grid::new("base_parts").num_columns(2).show(ui, |ui| {
                for p in crate::rocket::BASE_PARTS {
                    ui.label(egui::RichText::new(p.name).color(GOOD));
                    ui.label(egui::RichText::new(p.kind).color(DIM));
                    ui.end_row();
                }
            });
        });
}

/// A draggable part chip for the VAB palette.
fn drag_chip<P: std::any::Any + Send + Sync + Clone>(
    ui: &mut egui::Ui,
    id: egui::Id,
    payload: P,
    label: &str,
    col: egui::Color32,
) {
    let resp = ui
        .dnd_drag_source(id, payload, |ui| {
            let frame = egui::Frame::new()
                .fill(egui::Color32::from_rgb(28, 40, 56))
                .inner_margin(egui::Margin::symmetric(6, 3))
                .corner_radius(4);
            frame.show(ui, |ui| {
                ui.label(egui::RichText::new(label).color(col));
            });
        })
        .response;
    resp.on_hover_cursor(egui::CursorIcon::Grab);
}

/// Outlined frame used for the stack's drop slots.
fn slot_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(20, 28, 40))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 90, 120)))
        .inner_margin(egui::Margin::symmetric(6, 2))
        .corner_radius(3)
}

/// KSP-style launch telemetry + controls shown while a player-flown launch is
/// in progress (rocket view).
fn launch_panel(ctx: &egui::Context, world: &mut World) {
    let tel = match world.launch.as_ref() {
        Some(rk) => rk.telemetry(&world.body),
        None => return,
    };
    let phase_col = match tel.phase {
        "CRASHED" => egui::Color32::from_rgb(255, 80, 70),
        "ORBIT" => egui::Color32::from_rgb(120, 200, 255),
        "POWERED" => WARN,
        _ => DIM,
    };
    let twr_col = if tel.twr < 1.0 && tel.phase == "POWERED" { WARN } else { GOOD };
    let complete = world.mission_complete();
    let n_orbit = world.orbits.len();

    enum Act {
        Throttle(f64),
        Stage,
        Reset,
        NewMission,
    }
    let mut act: Option<Act> = None;

    egui::Window::new("LAUNCH CONTROL")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
        .default_width(250.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("PHASE").color(DIM));
                ui.label(egui::RichText::new(tel.phase).strong().color(phase_col));
            });
            egui::Grid::new("ltel").num_columns(2).show(ui, |ui| {
                kv(ui, "Altitude", &format!("{:.1} km", tel.alt_km));
                kv(ui, "Speed", &format!("{:.0} m/s", tel.speed));
                kv(ui, "Vert speed", &format!("{:.0} m/s", tel.vspeed));
                ui.label(egui::RichText::new("TWR").color(DIM));
                ui.label(egui::RichText::new(format!("{:.2}", tel.twr)).color(twr_col));
                ui.end_row();
                kv(ui, "Apoapsis", &fmt_alt(tel.apo_km));
                kv(ui, "Periapsis", &fmt_alt(tel.peri_km));
                kv(ui, "Pitch", &format!("{:.0} deg", tel.pitch_deg));
                kv(
                    ui,
                    "Stage",
                    &format!("{} ({}/{})", tel.stage_name, tel.stage_idx + 1, tel.stage_total),
                );
            });
            ui.separator();
            // throttle bar
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Throttle").color(DIM));
                ui.add(
                    egui::ProgressBar::new(tel.throttle)
                        .desired_width(110.0)
                        .text(format!("{:.0}%", tel.throttle * 100.0)),
                );
            });
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Propellant").color(DIM));
                ui.add(
                    egui::ProgressBar::new(tel.prop_frac)
                        .desired_width(110.0)
                        .fill(egui::Color32::from_rgb(90, 150, 90)),
                );
            });
            // structural integrity: drains under aerodynamic heating
            let hp = (tel.health / 100.0).clamp(0.0, 1.0);
            let hp_col = if hp > 0.5 {
                egui::Color32::from_rgb(90, 170, 90)
            } else if hp > 0.2 {
                WARN
            } else {
                egui::Color32::from_rgb(220, 70, 60)
            };
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Integrity").color(DIM));
                ui.add(
                    egui::ProgressBar::new(hp)
                        .desired_width(110.0)
                        .fill(hp_col)
                        .text(format!("{:.0}%", tel.health)),
                );
            });
            // heating gauge, shown when the air starts to bite
            if tel.heat > 0.1 {
                let hcol = egui::Color32::from_rgb(255, (200.0 - 150.0 * tel.heat).clamp(40.0, 200.0) as u8, 40);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Heating").color(DIM));
                    ui.add(
                        egui::ProgressBar::new(tel.heat.min(1.0))
                            .desired_width(110.0)
                            .fill(hcol)
                            .text(if tel.heat > 0.85 { "PLASMA" } else { "" }),
                    );
                });
            }
            ui.separator();
            if complete {
                ui.label(
                    egui::RichText::new("ORBIT ACHIEVED - payload deployed").strong().color(GOOD),
                );
                ui.label(egui::RichText::new(format!("Satellites in orbit: {n_orbit}")).color(DIM));
                let btn = egui::Button::new(
                    egui::RichText::new("NEW MISSION").strong().color(egui::Color32::BLACK),
                )
                .fill(GOOD)
                .min_size(egui::vec2(140.0, 24.0));
                if ui.add(btn).clicked() {
                    act = Some(Act::NewMission);
                }
                ui.label(egui::RichText::new("(R to return to the VAB)").color(DIM));
            } else {
                ui.horizontal(|ui| {
                    if ui.button("- thr").clicked() {
                        act = Some(Act::Throttle(-0.1));
                    }
                    if ui.button("+ thr").clicked() {
                        act = Some(Act::Throttle(0.1));
                    }
                    if ui.button("STAGE").clicked() {
                        act = Some(Act::Stage);
                    }
                    if ui.button("Reset").clicked() {
                        act = Some(Act::Reset);
                    }
                });
                ui.label(
                    egui::RichText::new("Shift/Ctrl throttle  W/S pitch  Space stage").color(DIM),
                );
            }
        });

    match act {
        Some(Act::Throttle(d)) => {
            if let Some(rk) = world.launch.as_mut() {
                rk.throttle = (rk.throttle + d).clamp(0.0, 1.0);
            }
        }
        Some(Act::Stage) => world.stage_launch(),
        Some(Act::Reset) => world.reset_launch(),
        Some(Act::NewMission) => world.back_to_vab(),
        None => {}
    }
}

/// LOD-debug overlay (rocket view, planet only, toggled with `L`): the terrain
/// is recoloured by quadtree depth and this panel reports the live LOD stats and
/// a colour legend so the split rings can be read and tuned.
fn lod_debug_panel(ctx: &egui::Context, world: &World) {
    let (lod, alt, cell) = world.lod_debug_stats();
    let tris = lod.triangle_count(crate::rocket::PLANET_PATCH_N);
    egui::Window::new("LOD DEBUG")
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 12.0))
        .default_width(220.0)
        .resizable(false)
        .show(ctx, |ui| {
            egui::Grid::new("loddbg").num_columns(2).show(ui, |ui| {
                kv(ui, "Altitude", &fmt_dist(alt));
                kv(ui, "Rebuild cell", &fmt_dist(cell));
                kv(ui, "Patches", &format!("{}", lod.patches.len()));
                kv(ui, "Max depth", &format!("{}", lod.max_depth_reached));
                kv(ui, "Triangles", &format!("{}", tris));
            });
            if alt > 50_000.0 {
                ui.label(egui::RichText::new("STABLE GLOBE (>50 km)").color(GOOD));
            }
            ui.separator();
            ui.label(egui::RichText::new("Depth -> colour").color(DIM));
            // one swatch row per depth that currently has patches
            for (d, &count) in lod.per_depth.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let c = crate::rocket::lod_color(d as u32);
                let col = egui::Color32::from_rgb(
                    (c[0] * 255.0) as u8,
                    (c[1] * 255.0) as u8,
                    (c[2] * 255.0) as u8,
                );
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, col);
                    ui.label(egui::RichText::new(format!("depth {d}  ({count})")).color(DIM));
                });
            }
            ui.separator();
            ui.label(egui::RichText::new("L toggles  -  colours track LOD").color(DIM));
        });
}

/// The maneuver-node planner (map view, with a craft in flight): place a burn
/// node on the orbit, dial prograde/normal/radial delta-v, preview the resulting
/// orbit (drawn cyan on the map), and execute.
fn maneuver_node_panel(ctx: &egui::Context, world: &mut World) {
    let Some(craft) = world.flight.as_ref() else { return };
    let mu = world.body.mu;
    let mut node = world.node.unwrap_or(crate::ManeuverNode { nu: 0.0, pro: 0.0, nrm: 0.0, rad: 0.0 });
    let has = world.node.is_some();
    let (apo, peri) = craft.node_apsides(&world.body, node.nu, node.pro, node.nrm, node.rad);
    let dv = (node.pro * node.pro + node.nrm * node.nrm + node.rad * node.rad).sqrt();

    let mut set = false;
    let mut clear = false;
    let mut execute = false;

    egui::Window::new("MANEUVER NODE")
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-12.0, -12.0))
        .default_width(250.0)
        .resizable(false)
        .show(ctx, |ui| {
            if !has {
                ui.label(egui::RichText::new("Plot a burn to change your orbit").color(DIM));
                if ui.button("+ Plan a burn").clicked() {
                    set = true;
                }
                return;
            }
            let mut deg = node.nu.to_degrees();
            if ui.add(egui::Slider::new(&mut deg, 0.0..=360.0).text("node position")).changed() {
                node.nu = deg.to_radians();
                set = true;
            }
            for (label, val, col) in [
                ("Prograde", &mut node.pro, GOOD),
                ("Normal", &mut node.nrm, egui::Color32::from_rgb(150, 200, 255)),
                ("Radial", &mut node.rad, AMBER),
            ] {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(label).color(col));
                    if ui.add(egui::DragValue::new(val).speed(2.0).suffix(" m/s").range(-5000.0..=5000.0)).changed() {
                        set = true;
                    }
                });
            }
            ui.separator();
            ui.label(format!("Result Ap: {}", fmt_alt(apo)));
            ui.label(format!("Result Pe: {}", fmt_alt(peri)));
            ui.label(egui::RichText::new(format!("Burn delta-v: {dv:.0} m/s")).strong());
            ui.separator();
            ui.horizontal(|ui| {
                let exe = egui::Button::new(egui::RichText::new("EXECUTE").strong().color(egui::Color32::BLACK)).fill(GOOD);
                if ui.add(exe).clicked() {
                    execute = true;
                }
                if ui.button("Clear").clicked() {
                    clear = true;
                }
            });
            ui.label(egui::RichText::new("cyan = resulting orbit").color(DIM));
        });

    if set {
        world.node = Some(node);
    }
    if clear {
        world.node = None;
    }
    if execute {
        if let Some(c) = world.flight.as_mut() {
            c.execute_node(mu, node.nu, node.pro, node.nrm, node.rad);
        }
        world.node = None;
    }
}

fn fmt_alt(km: f32) -> String {
    if km.is_finite() {
        format!("{km:.0} km")
    } else {
        "--".to_string()
    }
}

/// Metres rendered as km above 1 km, else metres. For the LOD-debug readouts.
fn fmt_dist(m: f64) -> String {
    if m >= 1000.0 {
        format!("{:.1} km", m / 1000.0)
    } else {
        format!("{m:.0} m")
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
    // attitude / effector telemetry
    let (perr, rate, wsat, rcsf) = (
        craft.pointing_error_deg(),
        craft.rate_deg_s(),
        craft.wheel_saturation() * 100.0,
        craft.rcs_frac() * 100.0,
    );
    let warp = world.warp;
    let bot_phase = world.moonbot.as_ref().map(|b| b.phase.label());

    #[derive(PartialEq)]
    enum Act {
        Thr(f64),
        Mode(Mode),
        Release,
        ToggleBot,
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
            // attitude / rotational-control readout
            ui.label(egui::RichText::new("ATTITUDE").color(AMBER));
            egui::Grid::new("att").num_columns(2).show(ui, |ui| {
                let ecol = if perr < 2.0 { GOOD } else if perr < 15.0 { AMBER } else { WARN };
                kv_c(ui, "Point error", &format!("{perr:.1} deg"), ecol);
                kv(ui, "Rate", &format!("{rate:.2} deg/s"));
                kv_c(
                    ui,
                    "Wheels",
                    &format!("{wsat:.0} % sat"),
                    if wsat > 80.0 { WARN } else { DIM },
                );
                kv(ui, "RCS prop", &format!("{rcsf:.0} %"));
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
            ui.label(egui::RichText::new("Hold attitude (1-6, 7 = free)").color(DIM));
            ui.horizontal(|ui| {
                for (lbl, m) in [
                    ("Pro", Mode::Prograde),
                    ("Retro", Mode::Retrograde),
                    ("Nml", Mode::Normal),
                    ("Anml", Mode::AntiNormal),
                    ("RadO", Mode::RadialOut),
                    ("RadI", Mode::RadialIn),
                ] {
                    if ui.selectable_label(mode == m, lbl).clicked() {
                        act = Some(Act::Mode(m));
                    }
                }
            });
            if ui.selectable_label(mode == Mode::Free, "Free drift").clicked() {
                act = Some(Act::Mode(Mode::Free));
            }
            ui.separator();
            // moon-landing bot status / engage
            match bot_phase {
                Some(p) => {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("MOON BOT").color(GOOD));
                        ui.label(egui::RichText::new(p).strong().color(AMBER));
                    });
                    if ui.button("Take control (B)").clicked() {
                        act = Some(Act::ToggleBot);
                    }
                }
                None => {
                    if ui.button("Engage moon bot (B)").clicked() {
                        act = Some(Act::ToggleBot);
                    }
                }
            }
            ui.separator();
            time_controls(ui, warp, &mut warp_mul);
            if ui.button("Release control (F)").clicked() {
                act = Some(Act::Release);
            }
        });

    world.warp = (world.warp * warp_mul).clamp(1.0, 10000.0);
    match act {
        Some(Act::Release) => world.toggle_flight(),
        Some(Act::ToggleBot) => world.toggle_moonbot(),
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
    let on_vehicle = world.focus_rocket;
    let cam_dist = world.sys_dist;
    let days = world.sys_time / 86_400.0;
    let warp = world.warp;

    let mut to_focus: Option<usize> = None;
    let mut focus_vehicle = false;
    let mut warp_mul = 1.0f32;

    {
        let search = &mut world.ui_search;
        egui::Window::new("SYSTEM BODIES")
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 12.0))
            .default_width(220.0)
            .resizable(true)
            .show(ctx, |ui| {
                // The active vehicle is always at the top of the list (it is not
                // a universe body, so it cannot appear in the groups below).
                if ui.selectable_label(on_vehicle, egui::RichText::new("ACTIVE VEHICLE").color(AMBER)).clicked() {
                    focus_vehicle = true;
                }
                ui.separator();
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
    if focus_vehicle {
        world.set_focus_rocket();
    } else if let Some(i) = to_focus {
        world.set_focus(i);
    }
}
