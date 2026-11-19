use crate::hw::{self, HwEvent, PanelCfg, SerialHandle};
use crate::n2n::{self, TrainData};
use crate::pipeline::{self, Field, Flat, Img16, CROP_H, CROP_W, EXPECTED_BYTES, FULL_H, FULL_W};
use crate::settings::{self, Settings};
use crate::theme;
use crate::worker::{self, Display, Job, JobResult, Params};
use crate::{colormap, io};
use eframe::egui::{self, pos2, vec2, Align2, Color32, CornerRadius, FontId, Rect, RichText, Sense, Stroke, StrokeKind};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq)]
enum Handle {
    Nw,
    Ne,
    Sw,
    Se,
    Move,
}

#[derive(Clone, Copy)]
enum Timer {
    Capture,
    ScanStep,
}

enum Act {
    Capture,
    LoadBin,
    SavePng,
    SaveTif,
    SaveBin,
    SelectFolder,
    OpenFolder,
    Ccw,
    Cw,
    ResetAngle,
    ToggleScan,
    CaptureOffset,
    LoadOffset,
    SaveFlat,
    LoadFlat,
    AutoRange,
    ResetRange,
    Fit,
    CropToggled,
    N2nAddCurrent,
    N2nAddFiles,
    N2nClear,
    N2nTrainPairs,
    N2nTrainSingle,
    N2nStop,
    N2nLoad,
    N2nSave,
    N2nWidth(usize),
    Reconnect,
    WindowApply,
    WindowRead,
}

struct Hq {
    key: (u64, [u32; 6], bool),
    tex: egui::TextureHandle,
    rect: Rect,
}

/// One row of the per-capture log (also appended to shots.csv in the save folder).
struct ShotRow {
    time: String,
    idle_s: Option<f32>,
    delay: u32,
    exp: u32,
    win: Option<f32>,
    trig_ms: Option<f32>,
    mean: f32,
    top: f32,
    bottom: f32,
    min: u16,
    max: u16,
}

pub struct App {
    s: Settings,
    base: PathBuf,
    settings_path: PathBuf,
    settings_loaded: bool,
    last_size: Option<[f32; 2]>,

    // raw data + calibration
    raw: Option<Arc<Vec<u16>>>,
    raw_id: u64,
    width: usize,
    height: usize,
    offset: Option<(u64, Arc<Field>)>,
    flat: Option<(u64, Arc<Flat>)>,
    id_ctr: u64,

    // processing
    job_tx: Sender<Job>,
    res_rx: Receiver<JobResult>,
    job_seq: u64,
    last_sig: Option<(Params, u64, u64, u64, Option<u64>)>,
    force_job: bool,
    pending_auto: bool,
    cur: Option<Arc<Img16>>,
    s1: Option<(u64, Arc<Img16>)>,
    den: Option<(u64, Arc<Img16>)>,
    den_requested: Option<u64>,
    disp: Option<Arc<Display>>,
    disp_gen: u64,
    tex: Option<egui::TextureHandle>,
    tex_linear: bool,
    hist: Vec<u32>,
    data_min: u16,
    data_max: u16,
    img_info: String,

    // view
    zoom: f32,
    pan: egui::Vec2,
    crop: Option<[f32; 4]>,
    drag_handle: Option<Handle>,
    hq: Option<Hq>,
    pixel_info: String,

    // hardware
    hw_tx: Sender<HwEvent>,
    hw_rx: Receiver<HwEvent>,
    serial: SerialHandle,
    capturing: bool,
    scan_active: bool,
    turn_cb: bool,
    timers: Vec<(Instant, Timer)>,

    // noise2noise
    n2n: n2n::Handle,
    n2n_ready: bool,
    n2n_device: String,
    n2n_training: bool,
    n2n_progress: (usize, usize),
    n2n_loss: Vec<f32>,
    n2n_msg: String,
    train_set: Vec<(String, Arc<Img16>)>,

    // timing timeline
    tl_start: Option<Instant>,
    tl_trig: Option<f32>,
    tl_data: Option<f32>,
    tl_done: Option<f32>,
    tl_scale: f32,
    tl_drag: bool,

    // shot log
    last_capture_end: Option<Instant>,
    pending_meta: Option<(Option<f32>, u32, u32, Option<f32>)>,
    shots: Vec<ShotRow>,

    status: String,
    toasts: Vec<(String, Instant)>,
}

fn pill(ui: &mut egui::Ui, text: &str, color: Color32) {
    let galley = ui.painter().layout_no_wrap(text.to_owned(), FontId::proportional(11.5), color);
    let (rect, _) = ui.allocate_exact_size(galley.size() + vec2(16.0, 6.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(10), color.gamma_multiply(0.16));
    ui.painter().galley(rect.min + vec2(8.0, 3.0), galley, color);
}

fn card(ui: &mut egui::Ui, title: &str, open: bool, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(theme::CARD)
        .stroke(Stroke::new(1.0_f32, theme::STROKE))
        .corner_radius(CornerRadius::same(9))
        .inner_margin(egui::Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::CollapsingHeader::new(RichText::new(title).strong().color(theme::TEXT))
                .default_open(open)
                .show(ui, |ui| {
                    ui.add_space(2.0);
                    add(ui);
                    ui.add_space(4.0);
                });
        });
    ui.add_space(4.0);
}

fn primary(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_owned()).strong().color(Color32::from_rgb(8, 14, 26))).fill(theme::ACCENT)
}

fn dim(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).small().color(theme::MUTED));
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::apply(&cc.egui_ctx);
        let base = settings::find_base_dir();
        let settings_path = base.join("settings.json");
        let loaded = Settings::load(&settings_path);
        let settings_loaded = loaded.is_some();
        let mut s = loaded.unwrap_or_default();
        if s.open_model < 1 {
            // measured on the panel: all rows are sensitive from ~450 ms after START
            s.window_open_ms = 450.0;
            s.window_open_known = true;
            s.open_model = 1;
        }
        if s.save_dir.is_empty() || !Path::new(&s.save_dir).exists() {
            s.save_dir = base.to_string_lossy().into_owned();
        }
        let (job_tx, res_rx) = worker::spawn(cc.egui_ctx.clone());
        let (hw_tx, hw_rx) = channel();
        let n2n = n2n::spawn(cc.egui_ctx.clone(), base.clone(), s.n2n_width as usize);
        let serial = hw::new_serial();
        let mut app = Self {
            crop: s.crop_coords,
            base,
            settings_path,
            settings_loaded,
            last_size: None,
            raw: None,
            raw_id: 0,
            width: FULL_W,
            height: FULL_H,
            offset: None,
            flat: None,
            id_ctr: 0,
            job_tx,
            res_rx,
            job_seq: 0,
            last_sig: None,
            force_job: false,
            pending_auto: false,
            cur: None,
            s1: None,
            den: None,
            den_requested: None,
            disp: None,
            disp_gen: 0,
            tex: None,
            tex_linear: false,
            hist: vec![],
            data_min: 0,
            data_max: 0,
            img_info: String::new(),
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            drag_handle: None,
            hq: None,
            pixel_info: String::new(),
            hw_tx,
            hw_rx,
            serial,
            capturing: false,
            scan_active: false,
            turn_cb: false,
            timers: vec![],
            n2n,
            n2n_ready: false,
            n2n_device: "…".into(),
            n2n_training: false,
            n2n_progress: (0, 0),
            n2n_loss: vec![],
            n2n_msg: String::new(),
            train_set: vec![],
            tl_start: None,
            tl_trig: None,
            tl_data: None,
            tl_done: None,
            tl_scale: 0.0,
            tl_drag: false,
            last_capture_end: None,
            pending_meta: None,
            shots: vec![],
            status: "No file loaded".into(),
            toasts: vec![],
            s,
        };
        match hw::open_serial(&app.serial, &app.s.com_port, app.s.baud_rate) {
            Ok(()) => eprintln!("Serial port {} opened", app.s.com_port),
            Err(e) => eprintln!("Serial Error: Failed to open {}: {e}", app.s.com_port),
        }
        app.startup_load();
        app
    }

    fn startup_load(&mut self) {
        let def = self.base.join("default.bin");
        if def.is_file() {
            self.load_bin_path(&def, false);
        }
        let off = self.base.join("offsetfield.tif");
        if off.is_file() {
            self.load_offset_path(&off);
        }
        let flat = self.base.join("flatfield.bin");
        if flat.is_file() {
            self.load_flat_path(&flat);
        }
        if self.raw.is_some() {
            if self.offset.is_some() || !self.settings_loaded {
                self.pending_auto = true;
            }
            self.force_job = true;
        }
    }

    // ───────────────────────── helpers ─────────────────────────

    fn toast(&mut self, msg: impl Into<String>) {
        self.toasts.push((msg.into(), Instant::now()));
    }

    fn next_id(&mut self) -> u64 {
        self.id_ctr += 1;
        self.id_ctr
    }

    fn params(&self) -> Params {
        Params {
            width: self.width,
            height: self.height,
            use_offset: self.s.offset_field_enabled,
            use_flat: self.s.flat_field_enabled,
            clip: self.s.clip_threshold.max(0) as u32,
            offset: self.s.offset_var,
            range: self.s.range_var.max(1),
            invert: self.s.invert_colors,
            clahe: self.s.clahe_enabled,
            clahe_clip: self.s.clahe_clip_limit,
            clahe_tiles: self.s.clahe_tile_grid_size.max(1) as usize,
            sharpen: self.s.sharpen_strength,
            cmap: self.s.color_mode.clone(),
            n2n_strength: if self.s.n2n_enabled && self.n2n_ready { self.s.n2n_strength } else { 0.0 },
        }
    }

    fn signature(&self) -> (Params, u64, u64, u64, Option<u64>) {
        let p = self.params();
        let den = if p.n2n_strength > 0.0 { self.den.as_ref().map(|d| d.0) } else { None };
        (
            p,
            self.raw_id,
            self.offset.as_ref().map_or(0, |o| o.0),
            self.flat.as_ref().map_or(0, |o| o.0),
            den,
        )
    }

    fn submit_if_changed(&mut self) {
        let Some(raw) = self.raw.clone() else { return };
        let sig = self.signature();
        if !self.force_job && !self.pending_auto && self.last_sig.as_ref() == Some(&sig) {
            return;
        }
        self.job_seq += 1;
        let p = sig.0.clone();
        let den = if p.n2n_strength > 0.0 { self.den.clone() } else { None };
        let _ = self.job_tx.send(Job {
            id: self.job_seq,
            raw,
            raw_id: self.raw_id,
            off: self.offset.clone(),
            flat: self.flat.clone(),
            den,
            p,
            auto: std::mem::take(&mut self.pending_auto),
        });
        self.last_sig = Some(sig);
        self.force_job = false;
    }

    fn reset_view(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
    }

    // ───────────────────────── loading ─────────────────────────

    fn load_bin_path(&mut self, path: &Path, reset: bool) {
        match io::read_bin(path) {
            Ok(px) => {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.status = format!("Loaded {name} ({} pixels)", fmt_int(px.len()));
                self.set_raw(px);
                if reset {
                    self.reset_view();
                    self.pending_auto = true;
                }
            }
            Err(e) => self.status = format!("Error loading file: {e:#}"),
        }
    }

    fn set_raw(&mut self, px: Vec<u16>) {
        self.raw_id = self.next_id();
        self.raw = Some(Arc::new(px));
        self.force_job = true;
    }

    fn load_offset_path(&mut self, path: &Path) {
        match io::load_offset(path) {
            Ok(f) => {
                self.status = format!(
                    "Offset field loaded: {} ({}×{})",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    f.w,
                    f.h
                );
                let id = self.next_id();
                self.offset = Some((id, Arc::new(f)));
                if self.raw.is_some() {
                    self.pending_auto = true;
                }
            }
            Err(e) => {
                self.status = format!("Error loading offset field: {e:#}");
                self.toast(format!("Failed to load offset field: {e:#}"));
            }
        }
    }

    fn load_flat_path(&mut self, path: &Path) {
        match io::load_flat(path, self.width) {
            Ok(f) => {
                self.status = format!(
                    "Flat field loaded: {} ({}×{})",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    f.w,
                    f.h
                );
                let id = self.next_id();
                self.flat = Some((id, Arc::new(f)));
                self.force_job = true;
            }
            Err(e) => {
                self.status = format!("Error loading flat field: {e:#}");
                self.toast(format!("Failed to load flat field: {e:#}"));
            }
        }
    }

    // ───────────────────────── results ─────────────────────────

    fn poll_results(&mut self, ctx: &egui::Context) {
        while let Ok(r) = self.res_rx.try_recv() {
            let latest = r.id == self.job_seq;
            self.s1_apply(ctx, r, latest);
        }
    }

    fn s1_apply(&mut self, ctx: &egui::Context, r: JobResult, latest: bool) {
        let Some(img) = r.img.clone() else {
            self.status = "Invalid dimensions".into();
            self.cur = None;
            self.disp = None;
            return;
        };
        self.s1 = r.s1.clone().map(|s| (r.s1_key, s));
        self.cur = Some(img.clone());
        self.hist = r.hist;
        self.data_min = r.min;
        self.data_max = r.max;
        if let Some((o, rg)) = r.auto {
            self.s.offset_var = o;
            self.s.range_var = rg;
            if latest {
                // sliders now equal what the worker rendered with: don't re-render
                self.last_sig = Some(self.signature());
            }
        }
        if let Some(d) = r.disp {
            let (w, h) = (d.w, d.h);
            let linear = self.s.interpolation_mode != "nearest";
            let opts = if linear { egui::TextureOptions::LINEAR } else { egui::TextureOptions::NEAREST };
            let ci = if d.ch == 1 {
                egui::ColorImage::from_gray([w, h], &d.data)
            } else {
                egui::ColorImage::from_rgb([w, h], &d.data)
            };
            match &mut self.tex {
                Some(t) if t.size() == [w, h] && self.tex_linear == linear => t.set(ci, opts),
                _ => self.tex = Some(ctx.load_texture("image", ci, opts)),
            }
            self.tex_linear = linear;
            self.disp = Some(d);
            self.disp_gen += 1;
            self.hq = None;
        }
        // crop box follows the image size
        if let Some(c) = &mut self.crop {
            c[0] = c[0].clamp(0.0, img.w as f32);
            c[2] = c[2].clamp(0.0, img.w as f32);
            c[1] = c[1].clamp(0.0, img.h as f32);
            c[3] = c[3].clamp(0.0, img.h as f32);
            if c[2] <= c[0] + 10.0 || c[3] <= c[1] + 10.0 {
                self.crop = None;
            }
        }
        if self.s.crop_enabled && self.crop.is_none() {
            self.init_crop();
        }
        self.img_info = format!(
            "{}×{}  ({}/{} px)  data [{}–{}]",
            img.w,
            img.h,
            fmt_int(img.px.len()),
            fmt_int(r.total_pixels),
            self.data_min,
            self.data_max
        );

        // ask the Noise2Noise engine for the denoised version of this frame when needed
        if self.s.n2n_enabled && self.n2n_ready && !self.n2n_training {
            if let Some((key, s1)) = self.s1.clone() {
                let have = self.den.as_ref().is_some_and(|d| d.0 == key);
                if !have && self.den_requested != Some(key) {
                    self.den_requested = Some(key);
                    let _ = self.n2n.tx.send(n2n::Cmd::Infer { key, img: s1 });
                    self.status = "Noise2Noise: denoising…".into();
                }
            }
        }
    }

    fn init_crop(&mut self) {
        let Some(img) = &self.cur else { return };
        let (w, h) = (img.w as f32, img.h as f32);
        let (cw, ch) = (w * 0.5, h * 0.5);
        let (x1, y1) = ((w - cw) / 2.0, (h - ch) / 2.0);
        self.crop = Some([x1, y1, x1 + cw, y1 + ch]);
    }

    // ───────────────────────── hardware ─────────────────────────

    /// Window register value (0.1 ms units) to write before an acquisition, if the user enabled it.
    fn window_tenths(&self) -> Option<u32> {
        self.s.window_enabled.then(|| (self.s.window_ms * 10.0).round().clamp(1.0, 600_000.0) as u32)
    }

    fn panel_cfg(&self) -> PanelCfg {
        PanelCfg { host: self.s.panel_host.clone(), port: self.s.panel_port }
    }

    fn capture_image(&mut self) {
        if self.capturing {
            self.toast("A capture is already in progress.");
            return;
        }
        let ms = self.s.capture_time_ms;
        if !(10..=3000).contains(&ms) {
            self.toast("Capture time must be between 10 and 3000 ms");
            return;
        }
        let _ = hw::open_serial(&self.serial, &self.s.com_port, self.s.baud_rate);
        self.pending_meta = Some((
            self.last_capture_end.map(|t| t.elapsed().as_secs_f32()),
            self.s.trigger_delay_ms,
            self.s.capture_time_ms,
            self.window_tenths().map(|t| t as f32 / 10.0),
        ));
        self.capturing = true;
        let win = self.window_tenths();
        hw::spawn_capture(self.hw_tx.clone(), self.panel_cfg(), self.serial.clone(), ms, self.s.trigger_delay_ms, win);
    }

    fn capture_offset(&mut self) {
        if self.capturing {
            self.toast("A capture is already in progress.");
            return;
        }
        let ok = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Capture Offset Field")
            .set_description("Ensure X-ray source is OFF before capturing offset field.\n\nThis will overwrite the existing offsetfield.tif.\n\nDo you want to proceed?")
            .set_buttons(rfd::MessageButtons::YesNo)
            .show();
        if ok != rfd::MessageDialogResult::Yes {
            return;
        }
        self.capturing = true;
        let win = self.window_tenths();
        hw::spawn_offset_capture(self.hw_tx.clone(), self.panel_cfg(), win);
    }

    fn poll_hw(&mut self) {
        while let Ok(ev) = self.hw_rx.try_recv() {
            match ev {
                HwEvent::Status(s) => self.status = s,
                HwEvent::Frame(b) => {
                    self.capturing = false;
                    self.process_captured(io::bytes_to_u16(&b));
                }
                HwEvent::OffsetFrame(b) => {
                    self.capturing = false;
                    let raw = io::bytes_to_u16(&b);
                    let (w, h) = (FULL_W, raw.len() / FULL_W);
                    let path = self.base.join("offsetfield.tif");
                    if let Err(e) = io::save_tif16(&path, w, h, &raw[..w * h], None) {
                        self.toast(format!("Error saving offset field: {e:#}"));
                    }
                    let f = io::offset_from_raw(&raw);
                    self.status = format!("Offset field saved: offsetfield.tif ({}×{})", f.w, f.h);
                    let id = self.next_id();
                    self.offset = Some((id, Arc::new(f)));
                    if self.raw.is_some() {
                        self.pending_auto = true;
                    }
                    self.toast("Offset field captured and saved");
                }
                HwEvent::Window { ok, tenths, msg } => {
                    if ok {
                        self.s.window_ms = tenths as f32 / 10.0;
                    }
                    self.status = msg.clone();
                    self.toast(msg);
                }
                HwEvent::Phase(p) => match p {
                    hw::Phase::Started => {
                        self.tl_start = Some(Instant::now());
                        self.tl_trig = None;
                        self.tl_data = None;
                        self.tl_done = None;
                    }
                    hw::Phase::Triggered(t) => self.tl_trig = Some(t),
                    hw::Phase::FirstData(t) => self.tl_data = Some(t),
                    hw::Phase::Done(t) => self.tl_done = Some(t),
                },
                HwEvent::Failed(e) => {
                    self.capturing = false;
                    self.status = e;
                    if self.scan_active {
                        self.scan_active = false;
                    }
                }
                HwEvent::TurntableDone { proceed } => {
                    if self.turn_cb {
                        self.turn_cb = false;
                        if proceed {
                            self.scan_step();
                        } else if self.scan_active {
                            self.scan_active = false;
                            self.status = "Auto scan stopped.".into();
                        }
                    }
                }
            }
        }
    }

    fn process_captured(&mut self, px: Vec<u16>) {
        self.log_shot(&px);
        if self.scan_active {
            let n = px.len().div_ceil(10).max(1);
            let mean = px.iter().step_by(10).map(|&v| v as f64).sum::<f64>() / n as f64;
            if mean < 5000.0 {
                self.status = format!("Underexposed (mean={mean:.0}), retrying...");
                let d = self.s.auto_scan_delay_ms as u64;
                self.timers.push((Instant::now() + Duration::from_millis(d), Timer::Capture));
                return;
            }
        }
        self.status = format!("Capture complete! ({} pixels)", fmt_int(px.len()));
        self.set_raw(px);
        self.reset_view();
        self.pending_auto = true;

        if self.scan_active {
            // saving needs the freshly corrected frame: build it synchronously
            if let Some(img) = self.build_now() {
                self.cur = Some(Arc::new(img));
            }
            match self.s.capture_format.as_str() {
                "tif" => self.save_tif(),
                "bin" => self.save_bin(),
                _ => self.save_png(),
            }
            if self.s.current_angle + self.s.rotation_step > 360.01 {
                self.s.current_angle = 360.0;
                self.timers.push((Instant::now() + Duration::from_millis(100), Timer::ScanStep));
            } else {
                self.rotate(true, true);
            }
        }
    }


    /// Record brightness / top-vs-bottom statistics of a freshly captured raw frame.
    fn log_shot(&mut self, px: &[u16]) {
        let (w, h) = if px.len() == CROP_W * CROP_H { (CROP_W, CROP_H) } else { (FULL_W, (px.len() / FULL_W).max(1)) };
        // the image is shown rotated by 180 deg, so the displayed top is the raw bottom
        let region = |r0: usize, r1: usize| -> f32 {
            let (c0, c1) = (w / 4, w * 3 / 4);
            let mut sum = 0f64;
            let mut n = 0f64;
            for y in r0..r1.min(h) {
                for &v in &px[y * w + c0..y * w + c1] {
                    sum += v as f64;
                    n += 1.0;
                }
            }
            (sum / n.max(1.0)) as f32
        };
        let top = region(h * 2 / 3, h);
        let bottom = region(0, h / 3);
        let mean = region(0, h);
        let (min, max) = pipeline::min_max(px);
        let meta = self.pending_meta.take();
        let row = ShotRow {
            time: chrono::Local::now().format("%H:%M:%S").to_string(),
            idle_s: meta.as_ref().and_then(|m| m.0),
            delay: meta.as_ref().map_or(self.s.trigger_delay_ms, |m| m.1),
            exp: meta.as_ref().map_or(self.s.capture_time_ms, |m| m.2),
            win: meta.as_ref().and_then(|m| m.3),
            trig_ms: self.tl_trig.map(|t| t * 1000.0),
            mean,
            top,
            bottom,
            min,
            max,
        };
        // append to shots.csv
        let path = Path::new(&self.s.save_dir).join("shots.csv");
        let new = !path.exists();
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            use std::io::Write;
            if new {
                let _ = writeln!(f, "time,idle_s,trigger_delay_ms,pulse_ms,window_ms,trigger_sent_ms,mean,top_third,bottom_third,top_over_bottom,min,max");
            }
            let _ = writeln!(
                f,
                "{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.3},{},{}",
                row.time,
                row.idle_s.map_or(String::new(), |v| format!("{v:.1}")),
                row.delay,
                row.exp,
                row.win.map_or(String::new(), |v| format!("{v:.0}")),
                row.trig_ms.map_or(String::new(), |v| format!("{v:.0}")),
                row.mean,
                row.top,
                row.bottom,
                row.top / row.bottom.max(1.0),
                row.min,
                row.max
            );
        }
        self.shots.insert(0, row);
        self.shots.truncate(40);
        self.last_capture_end = Some(Instant::now());
    }

    fn build_now(&self) -> Option<Img16> {
        let raw = self.raw.as_ref()?;
        let off = if self.s.offset_field_enabled { self.offset.as_ref().map(|o| o.1.as_ref()) } else { None };
        let flat = if self.s.flat_field_enabled { self.flat.as_ref().map(|o| o.1.as_ref()) } else { None };
        pipeline::build_img16(raw, self.width, self.height, off, flat, self.s.clip_threshold.max(0) as u32)
    }

    fn rotate(&mut self, cw: bool, scan_cb: bool) {
        let a = self.s.rotation_step;
        if !(0.0..=180.0).contains(&a) {
            return;
        }
        let cmd = format!("{}{:?}\n", if cw { "CW" } else { "CCW" }, a);
        self.turn_cb = scan_cb;
        hw::spawn_serial_command(
            self.hw_tx.clone(),
            self.serial.clone(),
            self.s.com_port.clone(),
            self.s.baud_rate,
            cmd,
            scan_cb,
        );
        self.s.current_angle += if cw { a } else { -a };
    }

    fn toggle_scan(&mut self) {
        if self.scan_active {
            self.scan_active = false;
            self.timers.clear();
            self.status = "Auto scan stopped.".into();
            return;
        }
        if self.s.rotation_step <= 0.0 {
            self.toast("Rotation step must be > 0");
            return;
        }
        self.scan_active = true;
        self.s.current_angle = 0.0;
        self.status = "Starting auto scan...".into();
        self.scan_step();
    }

    fn scan_step(&mut self) {
        if !self.scan_active {
            return;
        }
        if self.s.current_angle > 359.9 {
            self.scan_active = false;
            self.status = "Auto scan complete!".into();
            self.toast("Auto scan completed successfully.");
            return;
        }
        let d = self.s.auto_scan_delay_ms;
        if d > 0 {
            self.status = format!("Waiting {d}ms...");
            self.timers.push((Instant::now() + Duration::from_millis(d as u64), Timer::Capture));
        } else {
            self.capture_image();
        }
    }

    fn poll_timers(&mut self) {
        let now = Instant::now();
        let mut due = vec![];
        self.timers.retain(|(t, k)| {
            if *t <= now {
                due.push(*k);
                false
            } else {
                true
            }
        });
        for k in due {
            match k {
                Timer::Capture => self.capture_image(),
                Timer::ScanStep => self.scan_step(),
            }
        }
    }

    // ───────────────────────── saving ─────────────────────────

    fn base_filename(&self) -> PathBuf {
        let dt = chrono::Local::now().format("%Y%m%d_%H%M%S");
        Path::new(&self.s.save_dir).join(format!("capture_{:.1}_{dt}", self.s.current_angle))
    }

    fn crop_box(&self) -> Option<[usize; 4]> {
        if !self.s.crop_enabled {
            return None;
        }
        self.crop.map(|c| [c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize])
    }

    fn save_bin(&mut self) {
        let Some(raw) = self.raw.clone() else { return self.toast("Error: No file loaded") };
        let p = with_ext(&self.base_filename(), ".bin");
        match io::write_bin(&p, &raw) {
            Ok(()) => self.toast(format!("Saved BIN to {}", file_name(&p))),
            Err(e) => self.toast(format!("Error saving BIN: {e:#}")),
        }
    }

    fn save_png(&mut self) {
        let Some(img) = self.cur.clone() else { return self.toast("Error: No file loaded") };
        let d = worker::render(&img, &self.effective_params());
        let tag = if d.ch == 1 { "gray".to_string() } else { self.s.color_mode.clone() };
        let crop = self.crop_box();
        let suffix = if crop.is_some() { "_cropped" } else { "" };
        let p = with_ext(&self.base_filename(), &format!("_{tag}{suffix}.png"));
        match io::save_png(&p, d.w, d.h, &d.data, d.ch, crop) {
            Ok(()) => self.toast(format!("Saved PNG to {}", file_name(&p))),
            Err(e) => self.toast(format!("Error saving PNG: {e:#}")),
        }
    }

    fn save_tif(&mut self) {
        let Some(img) = self.cur.clone() else { return self.toast("Error: No file loaded") };
        let out = pipeline::stretch16(&img.px, self.s.tif_percentile);
        let crop = self.crop_box();
        let suffix = if crop.is_some() { "_raw16_cropped.tif" } else { "_raw16.tif" };
        let p = with_ext(&self.base_filename(), suffix);
        match io::save_tif16(&p, img.w, img.h, &out, crop) {
            Ok(()) => self.toast(format!("Saved TIF to {}", file_name(&p))),
            Err(e) => self.toast(format!("Error saving TIF: {e:#}")),
        }
    }

    /// Display params with the window the worker actually used (after auto range).
    fn effective_params(&self) -> Params {
        let mut p = self.params();
        p.n2n_strength = 0.0;
        p
    }

    fn save_flat(&mut self) {
        if self.raw.is_none() {
            return self.toast("Error: No file loaded");
        }
        let ok = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Save Flat Field")
            .set_description("Ensure X-ray source is ON and there is NO SUBJECT in the beam.\n\nThis will overwrite the existing flatfield.bin.\n\nDo you want to proceed?")
            .set_buttons(rfd::MessageButtons::YesNo)
            .show();
        if ok != rfd::MessageDialogResult::Yes {
            return;
        }
        let Some(raw) = self.raw.clone() else { return };
        let off = if self.s.offset_field_enabled { self.offset.as_ref().map(|o| o.1.as_ref()) } else { None };
        let Some(img) = pipeline::build_img16(&raw, self.width, self.height, off, None, self.s.clip_threshold.max(0) as u32)
        else {
            return self.toast("Error: Invalid dimensions");
        };
        let path = self.base.join("flatfield.bin");
        match io::write_bin(&path, &img.px) {
            Ok(()) => {
                self.toast("Saved Flat Field to flatfield.bin");
                self.load_flat_path(&path);
            }
            Err(e) => self.toast(format!("Error saving Flat Field: {e:#}")),
        }
    }

    // ───────────────────────── noise2noise ─────────────────────────

    fn poll_n2n(&mut self) {
        while let Ok(ev) = self.n2n.rx.try_recv() {
            match ev {
                n2n::Ev::Info { device, ready, width } => {
                    self.n2n_device = device;
                    let changed = self.n2n_ready != ready;
                    self.n2n_ready = ready;
                    self.s.n2n_width = width as u32;
                    if changed {
                        self.den = None;
                        self.den_requested = None;
                        self.force_job = true;
                    }
                }
                n2n::Ev::Status(s) => self.n2n_msg = s,
                n2n::Ev::Progress { step, total, loss } => {
                    self.n2n_progress = (step, total);
                    if self.n2n_loss.len() > 4000 {
                        self.n2n_loss.remove(0);
                    }
                    self.n2n_loss.push(loss);
                    self.n2n_msg = format!("Training {step}/{total}  loss {loss:.5}");
                }
                n2n::Ev::Inferred { key, img } => {
                    self.den = Some((key, img));
                    self.n2n_msg = "Noise2Noise applied".into();
                    self.status = "Noise2Noise applied".into();
                }
                n2n::Ev::TrainDone { loss, rms_before, rms_after } => {
                    self.n2n_training = false;
                    let better = rms_after < rms_before * 0.99;
                    self.n2n_msg = format!(
                        "Done (loss {loss:.5}). Held-out RMS vs 2nd exposure: raw {rms_before:.1} → model {rms_after:.1} counts ({:+.0}%). {}",
                        (rms_after / rms_before.max(1e-9) - 1.0) * 100.0,
                        if better { "Model beats the raw frame." } else { "Model is NOT better than the raw frame - don't use it." }
                    );
                    self.toast(if better { "Noise2Noise trained: better than raw" } else { "Noise2Noise: model is not better than the raw frame" });
                    self.den = None;
                    self.den_requested = None;
                    self.n2n_ready = true;
                    self.force_job = true;
                }
                n2n::Ev::Error(e) => {
                    self.n2n_training = false;
                    self.den_requested = None;
                    self.n2n_msg = e.clone();
                    self.toast(e);
                }
            }
        }
    }

    fn start_train(&mut self, data: TrainData) {
        self.n2n_training = true;
        self.n2n_loss.clear();
        self.n2n_progress = (0, self.s.n2n_steps as usize);
        let _ = self.n2n.tx.send(n2n::Cmd::Train {
            data,
            steps: self.s.n2n_steps as usize,
            batch: self.s.n2n_batch.max(1) as usize,
        });
    }

    fn add_train_frame(&mut self, name: String, img: Arc<Img16>) {
        let mean = img.px.iter().step_by(7).map(|&v| v as f64).sum::<f64>() / (img.px.len() / 7).max(1) as f64;
        if mean < 3000.0 {
            return self.toast(format!("{name}: frame looks blank (mean {mean:.0}), not added"));
        }
        if let Some((_, first)) = self.train_set.first() {
            if first.w != img.w || first.h != img.h {
                return self.toast("Frame size differs from the training set");
            }
        }
        self.train_set.push((name, img));
    }

    // ───────────────────────── actions ─────────────────────────

    fn run(&mut self, act: Act) {
        match act {
            Act::Capture => self.capture_image(),
            Act::LoadBin => {
                if let Some(p) = rfd::FileDialog::new()
                    .set_title("Select .bin file")
                    .add_filter("Binary files", &["bin"])
                    .add_filter("All files", &["*"])
                    .pick_file()
                {
                    self.load_bin_path(&p, true);
                }
            }
            Act::SavePng => self.save_png(),
            Act::SaveTif => self.save_tif(),
            Act::SaveBin => self.save_bin(),
            Act::SelectFolder => {
                if let Some(p) = rfd::FileDialog::new()
                    .set_title("Select Save Folder")
                    .set_directory(&self.s.save_dir)
                    .pick_folder()
                {
                    self.s.save_dir = p.to_string_lossy().into_owned();
                }
            }
            Act::OpenFolder => {
                let d = self.s.save_dir.clone();
                if Path::new(&d).exists() {
                    let opener = if cfg!(windows) {
                        "explorer"
                    } else if cfg!(target_os = "macos") {
                        "open"
                    } else {
                        "xdg-open"
                    };
                    let _ = std::process::Command::new(opener).arg(&d).spawn();
                } else {
                    self.toast(format!("Folder does not exist: {d}"));
                }
            }
            Act::Ccw => self.rotate(false, false),
            Act::Cw => self.rotate(true, false),
            Act::ResetAngle => self.s.current_angle = 0.0,
            Act::ToggleScan => self.toggle_scan(),
            Act::CaptureOffset => self.capture_offset(),
            Act::LoadOffset => {
                if let Some(p) = rfd::FileDialog::new()
                    .set_title("Select Offset Field (Dark Frame) TIFF")
                    .add_filter("TIFF files", &["tif", "tiff"])
                    .pick_file()
                {
                    self.load_offset_path(&p);
                }
            }
            Act::SaveFlat => self.save_flat(),
            Act::LoadFlat => {
                if let Some(p) = rfd::FileDialog::new()
                    .set_title("Select Flat Field BIN")
                    .add_filter("BIN files", &["bin"])
                    .pick_file()
                {
                    self.load_flat_path(&p);
                }
            }
            Act::AutoRange => self.pending_auto = true,
            Act::ResetRange => {
                self.s.offset_var = 0;
                self.s.range_var = 16383;
            }
            Act::Fit => self.reset_view(),
            Act::CropToggled => {
                if self.s.crop_enabled {
                    if self.cur.is_none() {
                        self.s.crop_enabled = false;
                        self.toast("No image loaded");
                    } else if self.crop.is_none() {
                        self.init_crop();
                    }
                }
            }
            Act::N2nAddCurrent => {
                if let Some((_, img)) = self.s1.clone() {
                    let n = self.train_set.len() + 1;
                    self.add_train_frame(format!("frame {n}"), img);
                } else {
                    self.toast("No image loaded");
                }
            }
            Act::N2nAddFiles => {
                if let Some(files) = rfd::FileDialog::new()
                    .set_title("Frames of the SAME view (.bin raw / 16-bit .tif)")
                    .add_filter("Frames", &["bin", "tif", "tiff", "png"])
                    .pick_files()
                {
                    for f in files {
                        let name = file_name(&f);
                        let is_bin = f.extension().is_some_and(|e| e.eq_ignore_ascii_case("bin"));
                        let img = if is_bin {
                            io::read_bin(&f).ok().and_then(|raw| {
                                let off = if self.s.offset_field_enabled { self.offset.as_ref().map(|o| o.1.as_ref()) } else { None };
                                let flat = if self.s.flat_field_enabled { self.flat.as_ref().map(|o| o.1.as_ref()) } else { None };
                                pipeline::build_img16(&raw, self.width, self.height, off, flat, self.s.clip_threshold.max(0) as u32)
                            })
                        } else {
                            io::load_gray16(&f).ok().map(|(w, h, px)| {
                                // 8-bit sources are scaled into the 14-bit range
                                let max = px.iter().copied().max().unwrap_or(1);
                                let px = if max <= 255 { px.into_iter().map(|v| v * 64).collect() } else { px };
                                Img16 { w, h, px }
                            })
                        };
                        match img {
                            Some(i) => self.add_train_frame(name, Arc::new(i)),
                            None => self.toast(format!("Could not read {name}")),
                        }
                    }
                }
            }
            Act::N2nClear => self.train_set.clear(),
            Act::N2nTrainPairs => {
                let frames = self.train_set.iter().map(|(_, i)| i.clone()).collect();
                self.start_train(TrainData::Pairs(frames));
            }
            Act::N2nTrainSingle => {
                if let Some((_, img)) = self.s1.clone() {
                    self.start_train(TrainData::Single(img));
                } else {
                    self.toast("No image loaded");
                }
            }
            Act::N2nStop => self.n2n.cancel.store(true, std::sync::atomic::Ordering::Relaxed),
            Act::N2nLoad => {
                if let Some(p) = rfd::FileDialog::new().add_filter("weights", &["safetensors"]).pick_file() {
                    let _ = self.n2n.tx.send(n2n::Cmd::Load(p));
                }
            }
            Act::N2nSave => {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("weights", &["safetensors"])
                    .set_file_name("noise2noise_xray.safetensors")
                    .save_file()
                {
                    let _ = self.n2n.tx.send(n2n::Cmd::Save(p));
                }
            }
            Act::N2nWidth(w) => {
                self.s.n2n_width = w as u32;
                self.n2n_ready = false;
                self.den = None;
                self.den_requested = None;
                let _ = self.n2n.tx.send(n2n::Cmd::SetWidth(w));
            }
            Act::WindowApply => {
                let t = (self.s.window_ms * 10.0).round().clamp(1.0, 600_000.0) as u32;
                hw::spawn_window_write(self.hw_tx.clone(), self.panel_cfg(), t);
            }
            Act::WindowRead => hw::spawn_window_read(self.hw_tx.clone(), self.panel_cfg()),
            Act::Reconnect => {
                hw::close_serial(&self.serial);
                match hw::open_serial(&self.serial, &self.s.com_port, self.s.baud_rate) {
                    Ok(()) => self.toast(format!("{} connected", self.s.com_port)),
                    Err(e) => self.toast(format!("Serial: {e}")),
                }
            }
        }
    }

    fn save_settings(&mut self) {
        self.s.crop_coords = self.crop;
        self.s.window_size = self.last_size;
        self.s.save(&self.settings_path);
    }

    // ───────────────────────── UI: side panel ─────────────────────────

    fn side_ui(&mut self, ui: &mut egui::Ui, acts: &mut Vec<Act>) {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("S4335W").heading().strong().color(theme::TEXT));
            ui.label(RichText::new("flat-panel controller").color(theme::MUTED));
        });
        ui.horizontal_wrapped(|ui| {
            let serial_ok = self.serial.lock().map(|g| g.is_some()).unwrap_or(false);
            pill(ui, &format!("{} {}", self.s.com_port, if serial_ok { "connected" } else { "offline" }), if serial_ok { theme::GOOD } else { theme::DANGER });
            pill(ui, &format!("{}:{}", self.s.panel_host, self.s.panel_port), theme::MUTED);
            if self.capturing {
                pill(ui, "capturing", theme::WARN);
            }
            if self.scan_active {
                pill(ui, "auto-scan", theme::ACCENT);
            }
        });
        ui.add_space(6.0);

        card(ui, "Acquisition", true, |ui| {
            ui.horizontal(|ui| {
                ui.label("Exposure");
                ui.add(egui::DragValue::new(&mut self.s.capture_time_ms).range(10..=3000).suffix(" ms"));
                let b = ui.add_enabled(!self.capturing, primary("●  Capture").min_size(vec2(ui.available_width(), 28.0)));
                if b.clicked() {
                    acts.push(Act::Capture);
                }
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.s.window_enabled, "Window")
                    .on_hover_text("Use a custom integration window (FPGA register 0x11C). It is written once when the value changes,\nfollowed by one discarded priming frame (the panel returns a blank frame after a write).\nUnticking does NOT restore the stock value: Apply 500 ms for that, or reboot the panel.");
                ui.add(egui::DragValue::new(&mut self.s.window_ms).range(1.0..=60000.0).speed(1.0).suffix(" ms"));
                if ui.small_button("Apply").on_hover_text("Write it now and verify by readback").clicked() {
                    acts.push(Act::WindowApply);
                }
                if ui.small_button("Read").on_hover_text("Read the live value from the panel").clicked() {
                    acts.push(Act::WindowRead);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Trigger delay");
                ui.add(egui::DragValue::new(&mut self.s.trigger_delay_ms).range(0..=5000).speed(5.0).suffix(" ms"))
                    .on_hover_text("Time between START and the X-ray pulse command to the Arduino (was fixed at 200 ms).\nIt must fall inside the integration window.");
            });
            ui.horizontal(|ui| {
                if ui.button("Load .bin…").clicked() {
                    acts.push(Act::LoadBin);
                }
                ui.label("Format");
                egui::ComboBox::from_id_salt("fmt").width(60.0).selected_text(self.s.capture_format.clone()).show_ui(ui, |ui| {
                    for f in ["png", "tif", "bin"] {
                        ui.selectable_value(&mut self.s.capture_format, f.to_string(), f);
                    }
                });
            });
            ui.horizontal(|ui| {
                if ui.button("Save PNG").clicked() {
                    acts.push(Act::SavePng);
                }
                if ui.button("Save TIF 16-bit").clicked() {
                    acts.push(Act::SaveTif);
                }
                if ui.button("Save BIN").clicked() {
                    acts.push(Act::SaveBin);
                }
            });
            ui.horizontal(|ui| {
                ui.label("TIF clip");
                ui.add(egui::DragValue::new(&mut self.s.tif_percentile).range(0.0..=10.0).speed(0.05).suffix(" %"))
                    .on_hover_text("Crop pixel values by this percentile before stretching to 16 bit");
            });
            ui.horizontal(|ui| {
                if ui.button("Folder…").clicked() {
                    acts.push(Act::SelectFolder);
                }
                if ui.button("Open").clicked() {
                    acts.push(Act::OpenFolder);
                }
            });
            dim(ui, &self.s.save_dir.clone());
        });

        card(ui, "Shot log", false, |ui| {
            if self.shots.is_empty() {
                dim(ui, "Every capture is listed here and appended to shots.csv in the save folder.");
            } else {
                dim(ui, "time      idle    mean  top/bot  max");
                for r in self.shots.iter().take(10) {
                    let ratio = r.top / r.bottom.max(1.0);
                    let col = if r.mean < 500.0 {
                        theme::DANGER
                    } else if !(0.9..=1.1).contains(&ratio) {
                        theme::WARN
                    } else {
                        theme::TEXT
                    };
                    let idle = r.idle_s.map_or("   -".to_string(), |v| format!("{v:>4.0}s"));
                    ui.label(RichText::new(format!("{}  {}  {:>6.0}  {:>6.2}  {:>5}", r.time, idle, r.mean, ratio, r.max)).monospace().small().color(col))
                        .on_hover_text(format!(
                            "pulse delay {} ms, pulse {} ms, window {}\nsource command sent {} ms after START\ntop third {:.0}, bottom third {:.0}, min {}",
                            r.delay,
                            r.exp,
                            r.win.map_or("not controlled".to_string(), |v| format!("{v:.0} ms")),
                            r.trig_ms.map_or("?".to_string(), |v| format!("{v:.0}")),
                            r.top,
                            r.bottom,
                            r.min
                        ));
                }
                dim(ui, "orange: top/bottom differ by >10%, red: almost black");
            }
        });

        card(ui, "Turntable", true, |ui| {
            ui.horizontal(|ui| {
                ui.label("Step");
                ui.add(egui::DragValue::new(&mut self.s.rotation_step).range(0.0..=180.0).speed(0.1).suffix("°"));
                if ui.button("⟲ CCW").clicked() {
                    acts.push(Act::Ccw);
                }
                if ui.button("CW ⟳").clicked() {
                    acts.push(Act::Cw);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Angle");
                ui.label(RichText::new(format!("{:.1}°", self.s.current_angle)).strong());
                if ui.small_button("reset").clicked() {
                    acts.push(Act::ResetAngle);
                }
                ui.label("Delay");
                ui.add(egui::DragValue::new(&mut self.s.auto_scan_delay_ms).range(0..=10000).speed(20).suffix(" ms"));
            });
            let (label, fill) = if self.scan_active { ("■  Stop auto scan", theme::DANGER) } else { ("▶  Start auto scan", theme::GOOD) };
            let b = egui::Button::new(RichText::new(label).strong().color(Color32::from_rgb(8, 14, 20)))
                .fill(fill)
                .min_size(vec2(ui.available_width(), 28.0));
            if ui.add(b).clicked() {
                acts.push(Act::ToggleScan);
            }
        });

        card(ui, "Corrections", true, |ui| {
            ui.horizontal(|ui| {
                if ui.checkbox(&mut self.s.offset_field_enabled, "Offset").changed() {
                    acts.push(Act::AutoRange);
                }
                if ui.checkbox(&mut self.s.flat_field_enabled, "Flat").changed() {
                    acts.push(Act::AutoRange);
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Capture offset").clicked() {
                    acts.push(Act::CaptureOffset);
                }
                if ui.button("Load offset…").clicked() {
                    acts.push(Act::LoadOffset);
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Save flat").clicked() {
                    acts.push(Act::SaveFlat);
                }
                if ui.button("Load flat…").clicked() {
                    acts.push(Act::LoadFlat);
                }
            });
            let of = self.offset.as_ref().map_or("offset: none".to_string(), |o| format!("offset: {}×{}", o.1.w, o.1.h));
            let fl = self.flat.as_ref().map_or("flat: none".to_string(), |o| format!("flat: {}×{}", o.1.w, o.1.h));
            dim(ui, &format!("{of}   ·   {fl}"));
        });

        card(ui, "Connection", false, |ui| {
            ui.horizontal(|ui| {
                ui.label("Serial");
                ui.add(egui::TextEdit::singleline(&mut self.s.com_port).desired_width(64.0));
                ui.add(egui::DragValue::new(&mut self.s.baud_rate).range(300..=2_000_000).speed(100));
                if ui.button("Reconnect").clicked() {
                    acts.push(Act::Reconnect);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Panel");
                ui.add(egui::TextEdit::singleline(&mut self.s.panel_host).desired_width(120.0));
                ui.add(egui::DragValue::new(&mut self.s.panel_port).range(1..=65534));
            });
        });
        ui.add_space(10.0);
    }

    fn right_ui(&mut self, ui: &mut egui::Ui, acts: &mut Vec<Act>) {
        ui.add_space(6.0);
        ui.label(RichText::new("Image").heading().strong().color(theme::TEXT));
        ui.add_space(4.0);
        card(ui, "Display", true, |ui| {
            ui.spacing_mut().slider_width = ui.available_width() - 110.0;
            ui.add(egui::Slider::new(&mut self.s.offset_var, 0..=16383).text("Offset"));
            ui.add(egui::Slider::new(&mut self.s.range_var, 1..=16384).text("Range"));
            ui.add(egui::Slider::new(&mut self.s.clip_threshold, 0..=16383).text("Clip"));
            ui.horizontal(|ui| {
                if ui.button("Auto").clicked() {
                    acts.push(Act::AutoRange);
                }
                if ui.button("Reset").clicked() {
                    acts.push(Act::ResetRange);
                }
                if ui.button("Fit view").clicked() {
                    acts.push(Act::Fit);
                }
            });
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("cmap").width(100.0).selected_text(self.s.color_mode.clone()).show_ui(ui, |ui| {
                    for &n in colormap::NAMES {
                        ui.selectable_value(&mut self.s.color_mode, n.to_string(), n);
                    }
                });
                ui.checkbox(&mut self.s.invert_colors, "Invert");
            });
            ui.horizontal(|ui| {
                ui.label("Resample");
                egui::ComboBox::from_id_salt("interp").width(100.0).selected_text(self.s.interpolation_mode.clone()).show_ui(ui, |ui| {
                    for n in ["nearest", "bilinear", "bicubic", "lanczos"] {
                        ui.selectable_value(&mut self.s.interpolation_mode, n.to_string(), n);
                    }
                });
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.s.histogram_visible, "Histogram");
                if ui.checkbox(&mut self.s.crop_enabled, "Crop").changed() {
                    acts.push(Act::CropToggled);
                }
            });
        });

        card(ui, "Enhance", true, |ui| {
            ui.spacing_mut().slider_width = ui.available_width() - 110.0;
            ui.add(egui::Slider::new(&mut self.s.sharpen_strength, 0.0..=2.0).text("Sharpen"));
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.s.clahe_enabled, "CLAHE");
                ui.label("clip");
                ui.add(egui::DragValue::new(&mut self.s.clahe_clip_limit).range(1.0..=40.0).speed(0.1));
                ui.label("tile");
                ui.add(egui::DragValue::new(&mut self.s.clahe_tile_grid_size).range(2..=32).speed(0.2));
            });
        });

        self.n2n_ui(ui, acts);
        ui.add_space(10.0);
    }

    fn n2n_ui(&mut self, ui: &mut egui::Ui, acts: &mut Vec<Act>) {
        card(ui, "Noise2Noise denoise (experimental)", true, |ui| {
            ui.label(
                RichText::new("EXPERIMENTAL: this may not work and can make images worse. Check the held-out result after training and keep the raw frame.")
                    .small()
                    .color(theme::WARN),
            );
            ui.spacing_mut().slider_width = ui.available_width() - 110.0;
            ui.horizontal(|ui| {
                ui.add_enabled(self.n2n_ready, egui::Checkbox::new(&mut self.s.n2n_enabled, "Enable"));
                pill(
                    ui,
                    if self.n2n_ready { "model ready" } else { "no trained model" },
                    if self.n2n_ready { theme::GOOD } else { theme::WARN },
                );
                pill(ui, &self.n2n_device.clone(), theme::MUTED);
            });
            ui.add_enabled(self.n2n_ready, egui::Slider::new(&mut self.s.n2n_strength, 0.0..=1.0).text("Strength"));
            if let Some((k, _)) = &self.s1 {
                if self.s.n2n_enabled && self.den_requested == Some(*k) && self.den.as_ref().map(|d| d.0) != Some(*k) {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        dim(ui, "denoising…");
                    });
                }
            }
            ui.separator();
            dim(ui, "Train on 2+ exposures of the SAME view, or self-supervised on the current frame.");
            ui.horizontal(|ui| {
                ui.label(format!("Training set: {} frame(s)", self.train_set.len()));
                if ui.small_button("clear").clicked() {
                    acts.push(Act::N2nClear);
                }
            });
            ui.horizontal(|ui| {
                if ui.button("+ current").clicked() {
                    acts.push(Act::N2nAddCurrent);
                }
                if ui.button("+ files…").clicked() {
                    acts.push(Act::N2nAddFiles);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Steps");
                ui.add(egui::DragValue::new(&mut self.s.n2n_steps).range(20..=100_000).speed(20));
                ui.label("Batch");
                ui.add(egui::DragValue::new(&mut self.s.n2n_batch).range(1..=64));
            });
            ui.horizontal(|ui| {
                ui.label("Width");
                let cur = self.s.n2n_width;
                for w in [24usize, 48] {
                    if ui.selectable_label(cur as usize == w, format!("{w}")).clicked() && cur as usize != w {
                        acts.push(Act::N2nWidth(w));
                    }
                }
                dim(ui, "(48 = reference)");
            });
            ui.horizontal(|ui| {
                let can = !self.n2n_training;
                if ui.add_enabled(can && self.train_set.len() >= 2, primary("Train on pairs")).clicked() {
                    acts.push(Act::N2nTrainPairs);
                }
                if ui.add_enabled(can && self.s1.is_some(), egui::Button::new("Self-supervised")).clicked() {
                    acts.push(Act::N2nTrainSingle);
                }
                if ui.add_enabled(!can, egui::Button::new("Stop")).clicked() {
                    acts.push(Act::N2nStop);
                }
            });
            if self.n2n_training || !self.n2n_loss.is_empty() {
                let (s, t) = self.n2n_progress;
                ui.add(egui::ProgressBar::new(if t > 0 { s as f32 / t as f32 } else { 0.0 }).desired_height(8.0));
                self.loss_plot(ui);
            }
            if !self.n2n_msg.is_empty() {
                dim(ui, &self.n2n_msg.clone());
            }
            ui.horizontal(|ui| {
                if ui.button("Load weights…").clicked() {
                    acts.push(Act::N2nLoad);
                }
                if ui.button("Save weights…").clicked() {
                    acts.push(Act::N2nSave);
                }
            });
        });
    }

    fn loss_plot(&self, ui: &mut egui::Ui) {
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 46.0), Sense::hover());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, CornerRadius::same(6), theme::INPUT);
        if self.n2n_loss.len() < 2 {
            return;
        }
        let (mn, mx) = self
            .n2n_loss
            .iter()
            .fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
        let span = (mx - mn).max(1e-9);
        let n = self.n2n_loss.len() as f32 - 1.0;
        let pts: Vec<_> = self
            .n2n_loss
            .iter()
            .enumerate()
            .map(|(i, &v)| pos2(rect.left() + 4.0 + (rect.width() - 8.0) * i as f32 / n, rect.bottom() - 4.0 - (rect.height() - 8.0) * (v - mn) / span))
            .collect();
        p.add(egui::Shape::line(pts, Stroke::new(1.5_f32, theme::ACCENT)));
    }


    // ───────────────────────── UI: timing timeline ─────────────────────────

    fn timeline_ui(&mut self, ui: &mut egui::Ui) {
        let delay = self.s.trigger_delay_ms as f32;
        let exp = self.s.capture_time_ms as f32;
        let win = self.s.window_ms;
        let open = self.s.window_open_ms;
        let known = self.s.window_open_known;
        let managed = self.s.window_enabled;
        let inside = {
            let (a, b) = (delay.max(open), (delay + exp).min(open + win));
            ((b - a).max(0.0) / exp.max(1.0)).clamp(0.0, 1.0)
        };

        ui.horizontal(|ui| {
            ui.label(RichText::new("Timing").strong().color(theme::TEXT));
            let (txt, col) = if !known {
                ("where the panel window opens after START is not known: drag its left handle once you have measured it".to_string(), theme::MUTED)
            } else if inside >= 0.999 {
                ("pulse is fully inside the all-rows window: uniform exposure expected".to_string(), theme::GOOD)
            } else if inside > 0.0 {
                (format!("only {:.0}% of the pulse is in the all-rows window: expect a top/bottom gradient", inside * 100.0), theme::WARN)
            } else {
                ("pulse is outside the all-rows window: partial or no exposure".to_string(), theme::DANGER)
            };
            ui.label(RichText::new(txt).small().color(col));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut parts = vec![];
                if let Some(t) = self.tl_trig {
                    parts.push(format!("source command sent {:.0} ms after START", t * 1000.0));
                }
                if !parts.is_empty() {
                    ui.label(RichText::new(format!("last capture: {}", parts.join(" · "))).small().color(theme::MUTED));
                }
            });
        });

        // time scale (frozen while a handle is being dragged so it does not jitter)
        // the download is not shown; the scale never exceeds 2 s
        let want = (((delay + exp).max(open + win).max(1000.0) * 1.15 / 250.0).ceil() * 250.0).min(2000.0);
        if !self.tl_drag || self.tl_scale <= 0.0 {
            self.tl_scale = want;
        }
        let t_ms = self.tl_scale;

        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 100.0), Sense::hover());
        let p = ui.painter_at(rect);
        let (left, right) = (rect.left() + 92.0, rect.right() - 16.0);
        let wpx = (right - left).max(50.0);
        let x_of = |t: f32| left + t.clamp(0.0, t_ms) / t_ms * wpx;
        let (y_win, y_src) = (rect.top() + 46.0, rect.top() + 78.0);
        let bar_h = 20.0;

        // axis
        let steps = [50.0f32, 100.0, 250.0, 500.0, 1000.0, 2000.0];
        let step = steps.iter().copied().find(|s| wpx * s / t_ms >= 70.0).unwrap_or(2000.0);
        let mut t = 0.0;
        while t <= t_ms + 0.1 {
            let x = x_of(t);
            p.line_segment([pos2(x, rect.top() + 18.0), pos2(x, rect.bottom() - 4.0)], Stroke::new(1.0_f32, theme::STROKE.gamma_multiply(0.7)));
            let label = if t >= 1000.0 && (t % 1000.0) == 0.0 { format!("{:.0} s", t / 1000.0) } else { format!("{t:.0}") };
            p.text(pos2(x + 3.0, rect.top() + 4.0), Align2::LEFT_TOP, label, FontId::proportional(10.5), theme::MUTED);
            t += step;
        }
        p.text(pos2(rect.left() + 4.0, y_win), Align2::LEFT_CENTER, "Panel window", FontId::proportional(12.0), theme::TEXT);
        p.text(pos2(rect.left() + 4.0, y_src), Align2::LEFT_CENTER, "X-ray pulse", FontId::proportional(12.0), theme::TEXT);

        // zones where only part of the rows are sensitive (measured): rows are reset one after another
        // before the window opens and read out one after another after it closes
        let zone = Color32::from_white_alpha(16);
        for (a, b, text) in [(0.0, open, "rows resetting"), (open + win, open + win + 150.0, "read-out")] {
            let zr = Rect::from_min_max(pos2(x_of(a), y_win - bar_h / 2.0), pos2(x_of(b), y_win + bar_h / 2.0));
            p.rect_filled(zr, CornerRadius::same(5), zone);
            if zr.width() > 60.0 {
                p.text(zr.center(), Align2::CENTER_CENTER, text, FontId::proportional(10.5), theme::MUTED);
            }
        }

        // panel window bar
        let win_col = if managed { theme::ACCENT } else { theme::MUTED };
        let win_rect = Rect::from_min_max(pos2(x_of(open), y_win - bar_h / 2.0), pos2(x_of(open + win), y_win + bar_h / 2.0));
        p.rect_filled(win_rect, CornerRadius::same(5), win_col.gamma_multiply(0.35));
        p.rect_stroke(win_rect, CornerRadius::same(5), Stroke::new(1.0_f32, win_col), StrokeKind::Inside);
        p.text(win_rect.left_center() + vec2(16.0, 0.0), Align2::LEFT_CENTER, format!("{win:.0} ms"), FontId::proportional(11.5), theme::TEXT);

        // X-ray bar
        let src_rect = Rect::from_min_max(pos2(x_of(delay), y_src - bar_h / 2.0), pos2(x_of(delay + exp), y_src + bar_h / 2.0));
        p.rect_filled(src_rect, CornerRadius::same(5), theme::WARN.gamma_multiply(0.4));
        p.rect_stroke(src_rect, CornerRadius::same(5), Stroke::new(1.0_f32, theme::WARN), StrokeKind::Inside);
        p.text(
            pos2(src_rect.right() + 6.0, y_src),
            Align2::LEFT_CENTER,
            format!("start {delay:.0} ms · {exp:.0} ms long"),
            FontId::proportional(11.5),
            theme::TEXT,
        );

        // handles
        let mut dragging = false;
        let snap = |t: f32| (t / 5.0).round() * 5.0;
        let mut handle = |ui: &mut egui::Ui, id: &str, center: egui::Pos2, col: Color32| -> Option<f32> {
            let r = Rect::from_center_size(center, vec2(18.0, 30.0));
            let resp = ui.interact(r, ui.id().with(id), Sense::drag());
            if resp.hovered() || resp.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
            let hot = resp.hovered() || resp.dragged();
            p.circle_filled(center, if hot { 8.5 } else { 7.0 }, col);
            p.circle_stroke(center, if hot { 8.5 } else { 7.0 }, Stroke::new(1.5_f32, Color32::WHITE));
            if resp.dragged() {
                dragging = true;
                if let Some(pos) = resp.interact_pointer_pos() {
                    return Some(snap(((pos.x - left) / wpx * t_ms).clamp(0.0, t_ms)));
                }
            }
            None
        };
        if let Some(t) = handle(ui, "h_win", pos2(x_of(open + win), y_win), theme::ACCENT) {
            self.s.window_ms = (t - open).max(1.0);
            self.s.window_enabled = true;
        }
        if let Some(t) = handle(ui, "h_win_start", pos2(x_of(open), y_win), theme::ACCENT.gamma_multiply(0.6)) {
            self.s.window_open_ms = t.min(open + win - 1.0).max(0.0);
            self.s.window_open_known = true;
        }
        if let Some(t) = handle(ui, "h_src_start", pos2(x_of(delay), y_src), theme::WARN) {
            self.s.trigger_delay_ms = t.min(5000.0) as u32;
        }
        if let Some(t) = handle(ui, "h_src_end", pos2(x_of(delay + exp), y_src), theme::WARN) {
            self.s.capture_time_ms = (t - delay).clamp(10.0, 3000.0) as u32;
        }
        self.tl_drag = dragging;

        // measured milestones from the last capture + live playhead
        if let Some(t) = self.tl_trig {
            let x = x_of(t * 1000.0);
            p.add(egui::Shape::convex_polygon(
                vec![pos2(x, y_src - 17.0), pos2(x - 5.0, y_src - 25.0), pos2(x + 5.0, y_src - 25.0)],
                Color32::WHITE,
                Stroke::NONE,
            ));
        }
        let el = match (self.capturing, self.tl_start) {
            (true, Some(s)) => Some(s.elapsed().as_secs_f32()),
            (false, Some(_)) => self.tl_done,
            _ => None,
        };
        if let Some(e) = el {
            let x = x_of(e * 1000.0).min(right + 8.0);
            p.line_segment([pos2(x, rect.top() + 14.0), pos2(x, rect.bottom() - 2.0)], Stroke::new(2.0_f32, Color32::from_rgb(120, 230, 255)));
            if self.capturing {
                ui.ctx().request_repaint_after(Duration::from_millis(33));
            }
        }
        p.text(
            pos2(rect.left() + 4.0, rect.top() + 4.0),
            Align2::LEFT_TOP,
            "ms after START",
            FontId::proportional(10.5),
            theme::MUTED,
        );
        ui.label(
            RichText::new("t=0 is START. Measured on this panel: rows are reset before the blue bar and read out after it (bottom first), so light outside the bar gives gradients. Right handle = window length (sent to the panel); left handle = where it opens (~450 ms, model only).")
                .small()
                .color(theme::MUTED),
        );
    }

    // ───────────────────────── UI: canvas ─────────────────────────

    /// returns (ratio, img_left, img_top)
    fn view(&self, canvas: Rect, ow: usize, oh: usize) -> (f32, f32, f32) {
        let base = (canvas.width() / ow as f32).min(canvas.height() / oh as f32) * 0.98;
        let ratio = base * self.zoom;
        let c = canvas.center() + self.pan;
        (ratio, c.x - ow as f32 * ratio / 2.0, c.y - oh as f32 * ratio / 2.0)
    }

    fn canvas_ui(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::ZERO, theme::CANVAS);

        let (Some(tex), Some(disp)) = (self.tex.clone(), self.disp.clone()) else {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Load a .bin file or press Capture",
                FontId::proportional(16.0),
                theme::MUTED,
            );
            return;
        };
        let (ow, oh) = (disp.w, disp.h);
        let mut scrolled = false;

        // zoom towards the cursor
        if resp.hovered() {
            let dy = ui.input(|i| i.raw_scroll_delta.y);
            if dy != 0.0 {
                scrolled = true;
                if let Some(m) = resp.hover_pos() {
                    let old = self.zoom;
                    self.zoom = (self.zoom * 1.1f32.powf((dy / 50.0).clamp(-4.0, 4.0))).clamp(0.1, 50.0);
                    let sc = self.zoom / old;
                    let mv = m - rect.center();
                    self.pan = (self.pan - mv) * sc + mv;
                }
            }
        }

        // drag: crop handles or pan
        if resp.drag_started() {
            if let Some(p) = resp.interact_pointer_pos() {
                if self.s.crop_enabled {
                    let (ratio, l, t) = self.view(rect, ow, oh);
                    if let Some(h) = self.hit_handle(p, ratio, l, t) {
                        self.drag_handle = Some(h);
                    } else if self.crop.is_none() {
                        let (x, y) = (((p.x - l) / ratio).clamp(0.0, ow as f32), ((p.y - t) / ratio).clamp(0.0, oh as f32));
                        self.crop = Some([x, y, (x + 100.0 / ratio).min(ow as f32), (y + 100.0 / ratio).min(oh as f32)]);
                    }
                }
            }
        }
        if resp.dragged() {
            let d = resp.drag_delta();
            if let (true, Some(h)) = (self.s.crop_enabled, self.drag_handle) {
                let (ratio, _, _) = self.view(rect, ow, oh);
                self.drag_crop(h, d / ratio, ow as f32, oh as f32);
            } else {
                self.pan += d;
            }
        }
        if resp.drag_stopped() {
            self.drag_handle = None;
        }
        let interacting = resp.dragged() || scrolled;

        let (ratio, left, top) = self.view(rect, ow, oh);
        let img_rect = Rect::from_min_size(pos2(left, top), vec2(ow as f32 * ratio, oh as f32 * ratio));
        let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

        // high quality resampling (bicubic / lanczos) once the view settles
        let hq_mode = matches!(self.s.interpolation_mode.as_str(), "bicubic" | "lanczos");
        let mut drew_hq = false;
        if hq_mode {
            let key = (
                self.disp_gen,
                [rect.width().to_bits(), rect.height().to_bits(), self.zoom.to_bits(), self.pan.x.to_bits(), self.pan.y.to_bits(), 0],
                self.s.interpolation_mode == "lanczos",
            );
            if self.hq.as_ref().map(|h| h.key) != Some(key) && !interacting {
                self.hq = self.make_hq(ui.ctx(), &disp, rect, ratio, left, top, key);
            }
            if let Some(h) = &self.hq {
                if h.key == key {
                    painter.image(h.tex.id(), h.rect, uv, Color32::WHITE);
                    drew_hq = true;
                }
            }
        }
        if !drew_hq {
            painter.image(tex.id(), img_rect, uv, Color32::WHITE);
        }

        // pixel readout
        self.pixel_info.clear();
        if let (Some(m), Some(img)) = (resp.hover_pos(), &self.cur) {
            let (x, y) = (((m.x - left) / ratio).floor(), ((m.y - top) / ratio).floor());
            if x >= 0.0 && y >= 0.0 && (x as usize) < img.w && (y as usize) < img.h {
                let v = img.px[y as usize * img.w + x as usize];
                self.pixel_info = format!("X:{} Y:{}  Val:{}", x as usize, y as usize, v);
            }
        }

        self.paint_crop(&painter, rect, ratio, left, top);
        if self.s.histogram_visible {
            self.paint_hist(&painter, rect);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_hq(&self, ctx: &egui::Context, d: &Display, canvas: Rect, ratio: f32, left: f32, top: f32, key: (u64, [u32; 6], bool)) -> Option<Hq> {
        let (ow, oh) = (d.w as f32, d.h as f32);
        let vl = ((canvas.left() - left) / ratio).max(0.0);
        let vt = ((canvas.top() - top) / ratio).max(0.0);
        let vr = ((canvas.right() - left) / ratio).min(ow);
        let vb = ((canvas.bottom() - top) / ratio).min(oh);
        if vr <= vl || vb <= vt {
            return None;
        }
        let m = 2usize;
        let cl = (vl as usize).saturating_sub(m);
        let ct = (vt as usize).saturating_sub(m);
        let cr = ((vr as usize) + m + 1).min(d.w);
        let cb = ((vb as usize) + m + 1).min(d.h);
        let (cw, chh) = (cr - cl, cb - ct);
        let (out_w, out_h) = (((cw as f32 * ratio) as u32).max(1), ((chh as f32 * ratio) as u32).max(1));
        if out_w as u64 * out_h as u64 > 24_000_000 {
            return None;
        }
        let mut buf = Vec::with_capacity(cw * chh * d.ch);
        for y in ct..cb {
            buf.extend_from_slice(&d.data[(y * d.w + cl) * d.ch..(y * d.w + cr) * d.ch]);
        }
        let filter = if key.2 { image::imageops::FilterType::Lanczos3 } else { image::imageops::FilterType::CatmullRom };
        let ci = if d.ch == 1 {
            let img = image::ImageBuffer::<image::Luma<u8>, _>::from_raw(cw as u32, chh as u32, buf)?;
            let r = image::imageops::resize(&img, out_w, out_h, filter);
            egui::ColorImage::from_gray([out_w as usize, out_h as usize], r.as_raw())
        } else {
            let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(cw as u32, chh as u32, buf)?;
            let r = image::imageops::resize(&img, out_w, out_h, filter);
            egui::ColorImage::from_rgb([out_w as usize, out_h as usize], r.as_raw())
        };
        let tex = ctx.load_texture("hq", ci, egui::TextureOptions::NEAREST);
        let r = Rect::from_min_size(pos2(left + cl as f32 * ratio, top + ct as f32 * ratio), vec2(out_w as f32, out_h as f32));
        Some(Hq { key, tex, rect: r })
    }

    fn crop_canvas(&self, c: [f32; 4], ratio: f32, l: f32, t: f32) -> Rect {
        Rect::from_min_max(pos2(l + c[0] * ratio, t + c[1] * ratio), pos2(l + c[2] * ratio, t + c[3] * ratio))
    }

    fn hit_handle(&self, p: egui::Pos2, ratio: f32, l: f32, t: f32) -> Option<Handle> {
        let c = self.crop?;
        let r = self.crop_canvas(c, ratio, l, t);
        let list = [
            (Handle::Nw, r.left_top()),
            (Handle::Ne, r.right_top()),
            (Handle::Sw, r.left_bottom()),
            (Handle::Se, r.right_bottom()),
            (Handle::Move, r.center()),
        ];
        list.into_iter().find(|(_, h)| (p.x - h.x).abs() <= 8.0 && (p.y - h.y).abs() <= 8.0).map(|(k, _)| k)
    }

    fn drag_crop(&mut self, h: Handle, d: egui::Vec2, ow: f32, oh: f32) {
        let Some(mut c) = self.crop else { return };
        match h {
            Handle::Nw => {
                c[0] += d.x;
                c[1] += d.y;
            }
            Handle::Ne => {
                c[2] += d.x;
                c[1] += d.y;
            }
            Handle::Sw => {
                c[0] += d.x;
                c[3] += d.y;
            }
            Handle::Se => {
                c[2] += d.x;
                c[3] += d.y;
            }
            Handle::Move => {
                c[0] += d.x;
                c[2] += d.x;
                c[1] += d.y;
                c[3] += d.y;
            }
        }
        c[0] = c[0].clamp(0.0, ow);
        c[2] = c[2].clamp(0.0, ow);
        c[1] = c[1].clamp(0.0, oh);
        c[3] = c[3].clamp(0.0, oh);
        if c[2] > c[0] + 10.0 && c[3] > c[1] + 10.0 {
            self.crop = Some(c);
        }
    }

    fn paint_crop(&self, p: &egui::Painter, canvas: Rect, ratio: f32, l: f32, t: f32) {
        let (true, Some(c)) = (self.s.crop_enabled, self.crop) else { return };
        let r = self.crop_canvas(c, ratio, l, t);
        let red = Color32::from_rgb(255, 80, 80);
        let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
        p.extend(egui::Shape::dashed_line(&pts, Stroke::new(2.0_f32, red), 6.0, 4.0));
        let ctr = r.center();
        p.extend(egui::Shape::dashed_line(&[ctr, pos2(ctr.x, canvas.bottom())], Stroke::new(1.0_f32, red), 4.0, 4.0));
        for (i, pt) in [r.left_top(), r.right_top(), r.left_bottom(), r.right_bottom(), ctr].into_iter().enumerate() {
            let hr = Rect::from_center_size(pt, vec2(8.0, 8.0));
            p.rect_filled(hr, CornerRadius::same(1), if i < 4 { red } else { theme::ACCENT });
            p.rect_stroke(hr, CornerRadius::same(1), Stroke::new(1.0_f32, Color32::WHITE), StrokeKind::Middle);
        }
    }

    fn paint_hist(&self, p: &egui::Painter, canvas: Rect) {
        if self.hist.len() != 256 {
            return;
        }
        let (w, h) = (300.0, 150.0);
        let r = Rect::from_min_size(pos2(canvas.right() - 10.0 - w, canvas.top() + 10.0), vec2(w, h));
        p.rect_filled(r, CornerRadius::same(6), Color32::from_black_alpha(190));
        let mx = (*self.hist.iter().max().unwrap_or(&1)).max(1) as f32;
        let bw = (w - 8.0) / 256.0;
        for (i, &c) in self.hist.iter().enumerate() {
            let bh = c as f32 / mx * (h - 28.0);
            if bh >= 0.5 {
                let x = r.left() + 4.0 + i as f32 * bw;
                p.rect_filled(
                    Rect::from_min_max(pos2(x, r.bottom() - 10.0 - bh), pos2(x + bw.max(1.0), r.bottom() - 10.0)),
                    CornerRadius::ZERO,
                    Color32::from_rgb(100, 200, 255),
                );
            }
        }
        let lo = (self.s.offset_var.max(0) as f32 / 16384.0 * 256.0).clamp(0.0, 255.0);
        let hi = ((self.s.offset_var + self.s.range_var.max(1)).min(16383) as f32 / 16384.0 * 256.0).clamp(0.0, 255.0);
        for (v, col) in [(lo, Color32::from_rgb(255, 70, 70)), (hi, Color32::from_rgb(70, 255, 110))] {
            let x = r.left() + 4.0 + v * bw;
            p.line_segment([pos2(x, r.top() + 4.0), pos2(x, r.bottom() - 4.0)], Stroke::new(1.5_f32, col));
        }
        p.rect_stroke(r, CornerRadius::same(6), Stroke::new(1.0_f32, Color32::from_white_alpha(120)), StrokeKind::Inside);
    }

    fn toasts_ui(&mut self, ctx: &egui::Context) {
        self.toasts.retain(|(_, t)| t.elapsed() < Duration::from_millis(2600));
        if self.toasts.is_empty() {
            return;
        }
        egui::Area::new(egui::Id::new("toasts"))
            .anchor(Align2::CENTER_BOTTOM, vec2(100.0, -48.0))
            .order(egui::Order::Foreground)
            .interactable(false)
            .show(ctx, |ui| {
                for (m, _) in self.toasts.iter().rev().take(3) {
                    egui::Frame::new()
                        .fill(Color32::from_rgb(30, 34, 45))
                        .stroke(Stroke::new(1.0_f32, theme::STROKE))
                        .corner_radius(CornerRadius::same(10))
                        .inner_margin(egui::Margin::symmetric(16, 9))
                        .shadow(egui::Shadow { offset: [0, 4], blur: 14, spread: 0, color: Color32::from_black_alpha(120) })
                        .show(ui, |ui| ui.label(RichText::new(m).color(theme::TEXT)));
                }
            });
        ctx.request_repaint_after(Duration::from_millis(200));
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_results(ctx);
        self.poll_hw();
        self.poll_n2n();
        self.poll_timers();
        self.last_size = Some([ctx.screen_rect().width(), ctx.screen_rect().height()]);

        let mut acts: Vec<Act> = vec![];

        egui::SidePanel::left("side")
            .exact_width(345.0)
            .resizable(false)
            .frame(egui::Frame::new().fill(theme::BG).inner_margin(egui::Margin::symmetric(10, 4)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.side_ui(ui, &mut acts));
            });

        egui::SidePanel::right("right")
            .exact_width(320.0)
            .resizable(false)
            .frame(egui::Frame::new().fill(theme::BG).inner_margin(egui::Margin::symmetric(10, 4)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.right_ui(ui, &mut acts));
            });

        egui::TopBottomPanel::bottom("status")
            .exact_height(28.0)
            .frame(egui::Frame::new().fill(theme::BG).stroke(Stroke::new(1.0_f32, theme::STROKE)).inner_margin(egui::Margin::symmetric(12, 4)))
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(RichText::new(&self.status).color(theme::TEXT));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!("{:.0}%", self.zoom * 100.0)).color(theme::MUTED));
                        ui.checkbox(&mut self.s.timeline_visible, "Timeline");
                        ui.label(RichText::new(&self.pixel_info).monospace().color(theme::ACCENT));
                        ui.label(RichText::new(&self.img_info).color(theme::MUTED));
                    });
                });
            });

        egui::TopBottomPanel::bottom("timeline")
            .resizable(false)
            .frame(egui::Frame::new().fill(theme::BG).stroke(Stroke::new(1.0_f32, theme::STROKE)).inner_margin(egui::Margin::symmetric(12, 6)))
            .show_animated(ctx, self.s.timeline_visible, |ui| self.timeline_ui(ui));

        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::CANVAS)).show(ctx, |ui| self.canvas_ui(ui));

        for a in acts {
            self.run(a);
        }
        self.submit_if_changed();
        self.toasts_ui(ctx);

        if self.capturing || self.scan_active || !self.timers.is_empty() || self.n2n_training {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.save_settings();
        hw::close_serial(&self.serial);
    }
}

fn fmt_int(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn with_ext(base: &Path, suffix: &str) -> PathBuf {
    let mut s = base.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

#[allow(dead_code)]
const _: (usize, usize, usize, usize, usize) = (CROP_W, CROP_H, EXPECTED_BYTES, 0, 0);
