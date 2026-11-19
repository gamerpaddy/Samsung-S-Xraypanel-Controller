#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]
#![allow(dead_code)]
mod app;
mod clahe;
mod colormap;
mod hw;
mod io;
mod n2n;
mod pipeline;
mod settings;
mod theme;
mod worker;

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s == "--n2n-selftest").unwrap_or(false) {
        if let Err(e) = n2n::selftest(&args[2..]) {
            eprintln!("error: {e:#}");
        }
        return Ok(());
    }

    if args.get(1).map(|s| s == "--n2n-eval").unwrap_or(false) {
        if let Err(e) = n2n::eval_cli(&args[2..]) {
            eprintln!("error: {e:#}");
        }
        return Ok(());
    }
    // `s4335w --shoot <pulse_ms> <window_ms> <delays>`: guarded real exposures (see hw::shoot_cli)
    if args.get(1).map(|s| s == "--shoot").unwrap_or(false) {
        if let Err(e) = hw::shoot_cli(&args[2..]) {
            eprintln!("error: {e}");
        }
        return Ok(());
    }
    // `s4335w --capture-test <window_ms>`: one dark acquisition through the window/priming path
    if args.get(1).map(|s| s == "--capture-test").unwrap_or(false) {
        let d = settings::load_effective();
        let cfg = hw::PanelCfg { host: d.panel_host, port: d.panel_port };
        let win = args.get(2).and_then(|a| a.parse::<f32>().ok()).map(|ms| (ms * 10.0).round() as u32);
        match hw::capture_test(&cfg, win) {
            Ok((m, lo, hi)) => println!("frame mean {m:.1}  data [{lo}-{hi}]"),
            Err(e) => eprintln!("error: {e}"),
        }
        return Ok(());
    }
    // `s4335w --reg-read 0x11C`: read one FPGA register from the panel (read-only diagnostic)
    if args.get(1).map(|s| s == "--reg-read").unwrap_or(false) {
        let addr = args.get(2).and_then(|a| u32::from_str_radix(a.trim_start_matches("0x"), 16).ok()).unwrap_or(hw::REG_WINDOW);
        let d = settings::load_effective();
        let cfg = hw::PanelCfg { host: d.panel_host, port: d.panel_port };
        match hw::reg_op(&cfg, false, addr, 0) {
            Ok(v) => println!("reg 0x{addr:03X} = 0x{v:08X} ({v})"),
            Err(e) => eprintln!("error: {e}"),
        }
        return Ok(());
    }

    let base = settings::find_base_dir();
    let size = settings::Settings::load(&base.join("settings.json"))
        .and_then(|s| s.window_size)
        .filter(|[w, h]| (1024.0..=2560.0).contains(w) && (600.0..=1440.0).contains(h))
        .unwrap_or([1500.0, 920.0]);

    let opts = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size(size)
            .with_min_inner_size([960.0, 600.0])
            .with_title("Samsung S4335W Controller"),
        ..Default::default()
    };
    eframe::run_native("Samsung S4335W Controller", opts, Box::new(|cc| Ok(Box::new(app::App::new(cc)))))
}
