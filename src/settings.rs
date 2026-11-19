use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Persistent settings. Keys match the Python viewer's `settings.json`; the extra keys are new.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Settings {
    pub color_mode: String,
    pub invert_colors: bool,
    pub flat_field_enabled: bool,
    pub offset_field_enabled: bool,
    pub sharpen_strength: f32,
    pub interpolation_mode: String,
    pub clahe_enabled: bool,
    pub clahe_clip_limit: f32,
    pub clahe_tile_grid_size: u32,
    pub capture_time_ms: u32,
    pub offset_var: i32,
    pub range_var: i32,
    pub clip_threshold: i32,
    pub histogram_visible: bool,
    pub crop_enabled: bool,
    pub crop_coords: Option<[f32; 4]>,
    pub save_dir: String,
    pub current_angle: f32,
    pub capture_format: String,
    pub auto_scan_delay_ms: u32,
    pub tif_percentile: f32,
    pub window_size: Option<[f32; 2]>,
    // --- new in the Rust port ---
    pub n2n_enabled: bool,
    pub n2n_strength: f32,
    pub n2n_width: u32,
    pub n2n_steps: u32,
    pub n2n_batch: u32,
    pub com_port: String,
    pub baud_rate: u32,
    pub panel_host: String,
    pub panel_port: u16,
    pub rotation_step: f32,
    /// write the FPGA window register (0x11C) before every capture
    pub window_enabled: bool,
    pub window_ms: f32,
    /// delay between START and the X-ray trigger sent to the Arduino
    pub trigger_delay_ms: u32,
    pub timeline_visible: bool,
    /// where the panel window opens after START (ms). Only a visual model on the timeline; never sent to the panel.
    pub window_open_ms: f32,
    pub window_open_known: bool,
    /// bumps when the timing model changes so old saved values are replaced
    pub open_model: u32,
}

fn default_serial_port() -> &'static str {
    if cfg!(windows) {
        "COM3"
    } else {
        "/dev/ttyUSB0"
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            color_mode: "grayscale".into(),
            invert_colors: false,
            flat_field_enabled: true,
            offset_field_enabled: true,
            sharpen_strength: 0.0,
            interpolation_mode: "nearest".into(),
            clahe_enabled: false,
            clahe_clip_limit: 5.0,
            clahe_tile_grid_size: 8,
            capture_time_ms: 500,
            offset_var: 0,
            range_var: 16383,
            clip_threshold: 16383,
            histogram_visible: false,
            crop_enabled: false,
            crop_coords: None,
            save_dir: String::new(),
            current_angle: 0.0,
            capture_format: "png".into(),
            auto_scan_delay_ms: 1000,
            tif_percentile: 1.0,
            window_size: None,
            n2n_enabled: false,
            n2n_strength: 1.0,
            n2n_width: 48,
            n2n_steps: 2000,
            n2n_batch: 4,
            com_port: default_serial_port().into(),
            baud_rate: 9600,
            panel_host: "192.168.1.100".into(), // change to your panel's IP in Connection
            panel_port: 20000,
            rotation_step: 1.0,
            window_enabled: false,
            window_ms: 500.0,
            trigger_delay_ms: 200,
            timeline_visible: true,
            window_open_ms: 450.0,
            window_open_known: true,
            open_model: 0,
        }
    }
}

impl Settings {
    pub fn load(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn save(&self, path: &Path) {
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }
}

/// Folder that holds default.bin / offsetfield.tif / flatfield.bin / settings.json.
/// Looks in the working dir, the exe dir and a few parents (so `cargo run` inside the crate works).
pub fn find_base_dir() -> PathBuf {
    let mut starts: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::current_dir() {
        starts.push(d);
    }
    if let Ok(e) = std::env::current_exe() {
        if let Some(p) = e.parent() {
            starts.push(p.to_path_buf());
        }
    }
    for s in &starts {
        for d in s.ancestors().take(6) {
            if ["default.bin", "offsetfield.tif", "flatfield.bin", "settings.json"]
                .iter()
                .any(|f| d.join(f).exists())
            {
                return d.to_path_buf();
            }
        }
    }
    starts.into_iter().next().unwrap_or_else(|| PathBuf::from("."))
}

/// Settings as saved by the program (falls back to the defaults); used by the command-line tools.
pub fn load_effective() -> Settings {
    Settings::load(&find_base_dir().join("settings.json")).unwrap_or_default()
}
