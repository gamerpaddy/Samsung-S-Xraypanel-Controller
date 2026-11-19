//! Dark theme.
use eframe::egui::{self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Stroke, TextStyle};
use std::sync::Arc;

pub const CANVAS: Color32 = Color32::from_rgb(8, 9, 12);
pub const BG: Color32 = Color32::from_rgb(17, 19, 25);
pub const CARD: Color32 = Color32::from_rgb(27, 30, 39);
pub const CARD_HOVER: Color32 = Color32::from_rgb(38, 42, 54);
pub const INPUT: Color32 = Color32::from_rgb(12, 14, 19);
pub const STROKE: Color32 = Color32::from_rgb(44, 49, 62);
pub const TEXT: Color32 = Color32::from_rgb(222, 226, 235);
pub const MUTED: Color32 = Color32::from_rgb(140, 148, 166);
pub const ACCENT: Color32 = Color32::from_rgb(86, 156, 255);
pub const GOOD: Color32 = Color32::from_rgb(76, 214, 165);
pub const WARN: Color32 = Color32::from_rgb(255, 196, 87);
pub const DANGER: Color32 = Color32::from_rgb(255, 99, 107);

pub fn apply(ctx: &egui::Context) {
    // Prefer the system UI font when available (clean on Windows), fall back to egui's built-ins.
    let mut fonts = FontDefinitions::default();
    if let Ok(bytes) = std::fs::read(r"C:\Windows\Fonts\segoeui.ttf") {
        fonts.font_data.insert("ui".into(), Arc::new(FontData::from_owned(bytes)));
        fonts.families.get_mut(&FontFamily::Proportional).unwrap().insert(0, "ui".into());
    }
    ctx.set_fonts(fonts);

    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = BG;
    v.extreme_bg_color = INPUT;
    v.faint_bg_color = CARD;
    v.window_stroke = Stroke::new(1.0_f32, STROKE);
    v.window_corner_radius = CornerRadius::same(10);
    v.menu_corner_radius = CornerRadius::same(8);
    v.hyperlink_color = ACCENT;
    v.selection.bg_fill = ACCENT.gamma_multiply(0.55);
    v.selection.stroke = Stroke::new(1.0_f32, Color32::WHITE);
    v.slider_trailing_fill = true;

    let r = CornerRadius::same(7);
    v.widgets.noninteractive.bg_fill = BG;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, STROKE);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.noninteractive.corner_radius = r;
    for (w, fill) in [
        (&mut v.widgets.inactive, CARD),
        (&mut v.widgets.hovered, CARD_HOVER),
        (&mut v.widgets.active, Color32::from_rgb(52, 58, 76)),
        (&mut v.widgets.open, CARD_HOVER),
    ] {
        w.bg_fill = fill;
        w.weak_bg_fill = fill;
        w.corner_radius = r;
        w.bg_stroke = Stroke::new(1.0_f32, STROKE);
        w.fg_stroke = Stroke::new(1.0_f32, TEXT);
    }
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, ACCENT.gamma_multiply(0.7));
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    ctx.set_visuals(v);

    ctx.style_mut(|s| {
        s.spacing.item_spacing = egui::vec2(8.0, 6.0);
        s.spacing.button_padding = egui::vec2(10.0, 5.0);
        s.spacing.slider_width = 170.0;
        s.spacing.interact_size.y = 24.0;
        s.spacing.scroll.bar_width = 6.0;
        s.text_styles = [
            (TextStyle::Heading, FontId::new(19.0, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(13.5, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(13.5, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(11.5, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(12.5, FontFamily::Monospace)),
        ]
        .into();
    });
}
