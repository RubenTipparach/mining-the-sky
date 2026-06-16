//! egui UI: a proper body browser for the orbital map (scrollable, grouped,
//! searchable, click to focus) plus time-scale controls. This is the start of
//! migrating the bitmap-font HUD to a real GUI toolkit.

use crate::universe::Kind;
use crate::{View, World};

/// Build the orbital-map UI for this frame.
pub fn build(ctx: &egui::Context, world: &mut World) {
    if world.view != View::Map {
        return;
    }

    // snapshot the body list (so the closure doesn't borrow `world` while we
    // also need to mutate focus/warp/search).
    let bodies: Vec<(usize, String, Kind)> = world
        .universe
        .bodies
        .iter()
        .enumerate()
        .map(|(i, b)| (i, b.name.clone(), b.kind))
        .collect();
    let focus = world.focus;
    let focus_name = world.focus_label().to_string();

    let mut to_focus: Option<usize> = None;
    let mut warp = world.warp;

    {
        let search = &mut world.ui_search;
        egui::Window::new("SYSTEM BODIES")
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 12.0))
            .default_width(220.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Find:");
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

                egui::ScrollArea::vertical().max_height(430.0).show(ui, |ui| {
                    group(ui, "Stars", &[Kind::StarA, Kind::StarB]);
                    group(ui, "Planets", &[Kind::Planet]);
                    group(ui, "Moons", &[Kind::Moon]);
                    group(ui, "Asteroids", &[Kind::AsteroidMajor, Kind::AsteroidMinor]);
                    group(ui, "Comets", &[Kind::Comet]);
                });

                ui.separator();
                ui.label(format!("Focus: {focus_name}"));
                ui.horizontal(|ui| {
                    ui.label("Time:");
                    if ui.button("<<").clicked() {
                        warp = (warp / 10.0).max(1.0);
                    }
                    ui.label(format!("{warp:.0}x"));
                    if ui.button(">>").clicked() {
                        warp = (warp * 10.0).min(10000.0);
                    }
                });
            });
    }

    world.warp = warp;
    if let Some(i) = to_focus {
        world.set_focus(i);
    }
}
