//! Hardware: Samsung panel over two TCP sockets, turntable over serial.
use crate::pipeline::EXPECTED_BYTES;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::mpsc::Sender;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const CMD_START: [u8; 4] = [0x02, 0x00, 0x00, 0x00];
const CMD_MODE: [u8; 4] = [0x01, 0xA0, 0xC1, 0x00];

pub type SerialHandle = Arc<Mutex<Option<Box<dyn serialport::SerialPort>>>>;

/// Timing milestones of one capture; the floats are seconds since START was sent.
#[derive(Clone, Copy, Debug)]
pub enum Phase {
    Started,
    Triggered(f32),
    FirstData(f32),
    Done(f32),
}

pub enum HwEvent {
    Phase(Phase),
    Status(String),
    Frame(Vec<u8>),
    OffsetFrame(Vec<u8>),
    Failed(String),
    /// `proceed == false` means the turntable timed out / failed and an auto scan must stop.
    TurntableDone { proceed: bool },
    /// Result of a window-register read/write. `tenths` is the live register value (0.1 ms units).
    Window { ok: bool, tenths: u32, msg: String },
}

/// FPGA window-time register (value in tenths of a millisecond).
pub const REG_WINDOW: u32 = 0x11C;
/// Window value (0.1 ms) last written by this program; 0 = unknown. Writes are skipped when unchanged.
static LAST_WINDOW: AtomicU32 = AtomicU32::new(0);
const PRIME_CYCLES: u32 = 1;
/// diagnostic override: S4335W_PRIME=<n>
fn prime_cycles() -> u32 {
    std::env::var("S4335W_PRIME").ok().and_then(|v| v.parse().ok()).unwrap_or(PRIME_CYCLES)
}
const CMD_FPGA_READ: u32 = 0xC1D0_0000;
const CMD_FPGA_WRITE: u32 = 0xC1D0_0001;

/// Raw FPGA register access through xcom (`CMD_FPGA_TEST_READ/WRITE`). Returns the value word of the reply.
pub fn reg_op(cfg: &PanelCfg, write: bool, addr: u32, val: u32) -> Result<u32, String> {
    let mut s = connect(&cfg.host, cfg.port)?;
    s.set_read_timeout(Some(Duration::from_secs(4))).ok();
    let mut msg = [0u8; 132];
    msg[0..4].copy_from_slice(&(if write { CMD_FPGA_WRITE } else { CMD_FPGA_READ }).to_le_bytes());
    msg[4..8].copy_from_slice(&addr.to_le_bytes());
    msg[8..12].copy_from_slice(&val.to_le_bytes());
    s.write_all(&msg).map_err(|e| e.to_string())?;
    let mut rep = [0u8; 132];
    s.read_exact(&mut rep).map_err(|e| format!("no reply: {e}"))?;
    Ok(u32::from_le_bytes([rep[8], rep[9], rep[10], rep[11]]))
}

/// Write the window register and verify it by reading it back.
fn set_window(cfg: &PanelCfg, tenths: u32) -> Result<(), String> {
    reg_op(cfg, true, REG_WINDOW, tenths)?;
    let back = reg_op(cfg, false, REG_WINDOW, 0)?;
    if back == tenths {
        Ok(())
    } else {
        Err(format!("window readback mismatch: wrote {tenths}, read {back}"))
    }
}

pub fn spawn_window_write(tx: Sender<HwEvent>, cfg: PanelCfg, tenths: u32) {
    std::thread::spawn(move || {
        let _ = tx.send(HwEvent::Status(format!("Setting window to {:.1} ms...", tenths as f32 / 10.0)));
        let r = set_window(&cfg, tenths).and_then(|_| {
            for k in 1..=prime_cycles() {
                let _ = tx.send(HwEvent::Status(format!("Priming panel after window change ({k}/{PRIME_CYCLES}, frame is discarded)...")));
                acquire(&tx, &cfg, None, None, "priming")?;
            }
            Ok(())
        });
        let ev = match r {
            Ok(()) => {
                LAST_WINDOW.store(tenths, Ordering::Relaxed);
                HwEvent::Window { ok: true, tenths, msg: format!("Window set to {:.1} ms (verified, panel primed)", tenths as f32 / 10.0) }
            }
            Err(e) => HwEvent::Window { ok: false, tenths, msg: format!("Window write failed: {e}") },
        };
        let _ = tx.send(ev);
    });
}

pub fn spawn_window_read(tx: Sender<HwEvent>, cfg: PanelCfg) {
    std::thread::spawn(move || {
        let ev = match reg_op(&cfg, false, REG_WINDOW, 0) {
            Ok(v) => HwEvent::Window { ok: true, tenths: v, msg: format!("Panel window is {:.1} ms", v as f32 / 10.0) },
            Err(e) => HwEvent::Window { ok: false, tenths: 0, msg: format!("Window read failed: {e}") },
        };
        let _ = tx.send(ev);
    });
}

#[derive(Clone)]
pub struct PanelCfg {
    pub host: String,
    pub port: u16,
}

pub fn new_serial() -> SerialHandle {
    Arc::new(Mutex::new(None))
}

pub fn open_serial(h: &SerialHandle, port: &str, baud: u32) -> Result<(), String> {
    let mut g = h.lock().unwrap();
    if g.is_some() {
        return Ok(());
    }
    match serialport::new(port, baud).timeout(Duration::from_secs(1)).open() {
        Ok(p) => {
            *g = Some(p);
            Ok(())
        }
        Err(e) => Err(e.to_string()),
    }
}

pub fn close_serial(h: &SerialHandle) {
    *h.lock().unwrap() = None;
}

fn connect(host: &str, port: u16) -> Result<TcpStream, String> {
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|e| e.to_string())?
        .next()
        .ok_or("bad address")?;
    let s = TcpStream::connect_timeout(&addr, Duration::from_secs(5)).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(Duration::from_secs(5))).ok();
    s.set_nodelay(true).ok();
    Ok(s)
}

/// The panel needs a few seconds after a cycle before it answers the next MODE/START. Until then every
/// capture attempt would hang and (worse) fire the X-ray into a panel that is not running.
static BUSY_UNTIL: Mutex<Option<Instant>> = Mutex::new(None);

/// End of the last panel cycle; the first exposure after a long idle needs a throwaway dark cycle first.
static LAST_END: Mutex<Option<Instant>> = Mutex::new(None);

fn mark_busy(secs: u64) {
    *LAST_END.lock().unwrap() = Some(Instant::now());
    *BUSY_UNTIL.lock().unwrap() = Some(Instant::now() + Duration::from_secs(secs));
}

fn wait_settled(tx: &Sender<HwEvent>) {
    let until = *BUSY_UNTIL.lock().unwrap();
    if let Some(t) = until {
        let now = Instant::now();
        if t > now {
            let _ = tx.send(HwEvent::Status(format!("Letting the panel settle ({:.0} s)...", (t - now).as_secs_f32().ceil())));
            std::thread::sleep(t - now);
        }
    }
}

/// Read from `s` until `pat` has appeared in the byte stream or `total` has elapsed.
fn read_until(s: &mut TcpStream, pat: &[u8], total: Duration) -> bool {
    let end = Instant::now() + total;
    let mut acc: Vec<u8> = Vec::new();
    let mut b = [0u8; 256];
    s.set_read_timeout(Some(Duration::from_millis(400))).ok();
    while Instant::now() < end {
        match s.read(&mut b) {
            Ok(0) => break,
            Ok(n) => {
                acc.extend_from_slice(&b[..n]);
                if acc.windows(pat.len()).any(|w| w == pat) {
                    return true;
                }
            }
            Err(_) => {}
        }
    }
    false
}

fn recv_logged(s: &mut TcpStream, what: &str) {
    let mut b = [0u8; 128];
    match s.read(&mut b) {
        Ok(n) => eprintln!("{what}: {:02x?} ({n} bytes)", &b[..n]),
        Err(e) => eprintln!("{what} error: {e}"),
    }
}

/// Runs one full panel acquisition. `serial_ms` => send the `C<ms>` trigger over serial after START.
fn acquire(
    tx: &Sender<HwEvent>,
    cfg: &PanelCfg,
    serial: Option<(&SerialHandle, u32, u32)>,
    window: Option<u32>,
    label: &str,
) -> Result<Vec<u8>, String> {
    wait_settled(tx);
    // Measured: the first exposure after the panel has been idle for a while is read out too early
    // (the lower part of the frame comes out dark). A throwaway dark cycle right before it fixes that.
    let window_change_pending = window.is_some_and(|t| LAST_WINDOW.load(Ordering::Relaxed) != t);
    if label == "capture" && serial.is_some() && !window_change_pending {
        let idle = LAST_END.lock().unwrap().map(|t| t.elapsed());
        if idle.map_or(true, |d| d > Duration::from_secs(60)) {
            let _ = tx.send(HwEvent::Status("Warm-up cycle (panel was idle), no X-ray...".into()));
            let w = acquire_inner(tx, cfg, None, None, "priming");
            mark_busy(if w.is_ok() { 6 } else { 12 });
            w?;
            wait_settled(tx);
        }
    }
    let r = acquire_inner(tx, cfg, serial, window, label);
    mark_busy(if r.is_ok() { 6 } else { 12 });
    r
}

fn acquire_inner(
    tx: &Sender<HwEvent>,
    cfg: &PanelCfg,
    serial: Option<(&SerialHandle, u32, u32)>,
    window: Option<u32>,
    label: &str,
) -> Result<Vec<u8>, String> {
    let st = |s: String| {
        let _ = tx.send(HwEvent::Status(s));
    };
    if let Some(t) = window {
        if LAST_WINDOW.load(Ordering::Relaxed) != t {
            st(format!("Setting window to {:.1} ms...", t as f32 / 10.0));
            set_window(cfg, t)?;
            // The panel returns an all-zero frame for the first acquisition after a register write,
            // so run (and discard) one acquisition now, with no X-ray trigger.
            // (measured: one priming cycle is not enough, the first real exposure after it still misses;
            // the second one after a window change is fine, so run two discarded cycles)
            for k in 1..=prime_cycles() {
                st(format!("Priming panel after window change ({k}/{PRIME_CYCLES}, frame is discarded)..."));
                acquire(tx, cfg, None, None, "priming")?;
            }
            // measured: a START right after the previous cycle is accepted but the exposure is missed,
            // so let the panel settle again before the real capture
            wait_settled(tx);
            LAST_WINDOW.store(t, Ordering::Relaxed);
        }
    }
    let mut cmd = connect(&cfg.host, cfg.port)?;
    let mut data = connect(&cfg.host, cfg.port + 1)?;
    st("Connected. Sending mode command...".into());
    cmd.write_all(&CMD_MODE).map_err(|e| e.to_string())?;
    if !read_until(&mut cmd, &[0x01, 0xA1, 0xC1, 0x00], Duration::from_secs(3)) {
        return Err("panel did not answer MODE (still busy from the previous capture?). Nothing was started and the X-ray was NOT fired. Wait ~15 s and try again.".into());
    }
    st(format!("Starting {label}..."));
    cmd.write_all(&CMD_START).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let phase = |p: Phase| {
        if label == "capture" {
            let _ = tx.send(HwEvent::Phase(p));
        }
    };
    phase(Phase::Started);
    // never fire the source unless the panel confirmed that it is running (reply code 0x12)
    if !read_until(&mut cmd, &[0x12, 0, 0, 0], Duration::from_millis(2500)) {
        return Err("panel did not acknowledge START. The X-ray was NOT fired. Wait ~15 s and try again.".into());
    }

    if let Some((handle, ms, delay_ms)) = serial {
        std::thread::sleep(Duration::from_millis(delay_ms as u64));
        phase(Phase::Triggered(t0.elapsed().as_secs_f32()));
        let line = format!("C{ms}\n");
        match handle.lock().unwrap().as_mut() {
            Some(p) => {
                if let Err(e) = p.write_all(line.as_bytes()) {
                    eprintln!("Serial command error: {e}");
                }
            }
            None => eprintln!("Serial port not available"),
        }
    }

    data.set_read_timeout(Some(Duration::from_secs(10))).ok();
    let mut buf = vec![0u8; EXPECTED_BYTES];
    let mut got = 0usize;
    while got < EXPECTED_BYTES {
        let end = (got + 65536).min(EXPECTED_BYTES);
        match data.read(&mut buf[got..end]) {
            Ok(0) => break,
            Ok(n) => {
                if got == 0 {
                    phase(Phase::FirstData(t0.elapsed().as_secs_f32()));
                }
                let (old, new) = (got >> 20, (got + n) >> 20);
                got += n;
                if new > old {
                    st(format!("Receiving {label}: {}% ({got}/{EXPECTED_BYTES} bytes)", got * 100 / EXPECTED_BYTES));
                }
            }
            Err(e) if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) => break,
            Err(e) => return Err(e.to_string()),
        }
    }
    if got >= EXPECTED_BYTES {
        phase(Phase::Done(t0.elapsed().as_secs_f32()));
        Ok(buf)
    } else {
        Err(format!("{label} incomplete: {got}/{EXPECTED_BYTES} bytes"))
    }
}

pub fn spawn_capture(tx: Sender<HwEvent>, cfg: PanelCfg, serial: SerialHandle, ms: u32, delay_ms: u32, window: Option<u32>) {
    std::thread::spawn(move || {
        let _ = tx.send(HwEvent::Status("Connecting to panel...".into()));
        match acquire(&tx, &cfg, Some((&serial, ms, delay_ms)), window, "capture") {
            Ok(b) => {
                let _ = tx.send(HwEvent::Frame(b));
            }
            Err(e) => {
                let _ = tx.send(HwEvent::Failed(e));
            }
        }
    });
}

pub fn spawn_offset_capture(tx: Sender<HwEvent>, cfg: PanelCfg, window: Option<u32>) {
    std::thread::spawn(move || {
        let _ = tx.send(HwEvent::Status("Capturing offset field (ensure X-ray is OFF)...".into()));
        match acquire(&tx, &cfg, None, window, "offset") {
            Ok(b) => {
                let _ = tx.send(HwEvent::OffsetFrame(b));
            }
            Err(e) => {
                let _ = tx.send(HwEvent::Failed(e));
            }
        }
    });
}

/// Send a turntable command; optionally wait (max 30 s) for "OK".
pub fn spawn_serial_command(tx: Sender<HwEvent>, serial: SerialHandle, port: String, baud: u32, cmd: String, wait_ok: bool) {
    std::thread::spawn(move || {
        if let Err(e) = open_serial(&serial, &port, baud) {
            let _ = tx.send(HwEvent::Status(format!("Serial Error: {e}")));
            let _ = tx.send(HwEvent::TurntableDone { proceed: true });
            return;
        }
        let mut proceed = true;
        {
            let mut g = serial.lock().unwrap();
            let Some(p) = g.as_mut() else { return };
            let _ = p.clear(serialport::ClearBuffer::Input);
            if let Err(e) = p.write_all(cmd.as_bytes()) {
                let _ = tx.send(HwEvent::Status(format!("Serial error: {e}")));
                let _ = tx.send(HwEvent::TurntableDone { proceed: false });
                return;
            }
            let _ = tx.send(HwEvent::Status(format!("Sent: {}", cmd.trim())));
            if wait_ok {
                let _ = tx.send(HwEvent::Status("Waiting for turntable OK...".into()));
                let start = Instant::now();
                let mut resp = String::new();
                let mut ok = false;
                while start.elapsed() < Duration::from_secs(30) {
                    let n = p.bytes_to_read().unwrap_or(0) as usize;
                    if n > 0 {
                        let mut b = vec![0u8; n];
                        if let Ok(k) = p.read(&mut b) {
                            resp.push_str(&String::from_utf8_lossy(&b[..k]));
                        }
                        if resp.to_uppercase().contains("OK") {
                            ok = true;
                            break;
                        }
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                let _ = tx.send(HwEvent::Status(
                    if ok { "Turntable OK received." } else { "Turntable timeout waiting for OK." }.into(),
                ));
                proceed = ok;
            }
        }
        if !wait_ok {
            std::thread::sleep(Duration::from_millis(500));
        }
        let _ = tx.send(HwEvent::TurntableDone { proceed });
    });
}

/// Diagnostic: one dark acquisition through the normal window/priming path. Returns (mean, min, max).
pub fn capture_test(cfg: &PanelCfg, window: Option<u32>) -> Result<(f64, u16, u16), String> {
    let (tx, _rx) = std::sync::mpsc::channel();
    let b = acquire(&tx, cfg, None, window, "capture")?;
    let px = crate::io::bytes_to_u16(&b);
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
    let (mn, mx) = crate::pipeline::min_max(&px);
    Ok((mean, mn, mx))
}

/// `s4335w --shoot <pulse_ms> <window_ms> <delay_ms>[,<delay_ms>...]`
/// Fires the real X-ray source through the same serial/panel path as the controller, with hard safety limits:
/// pulse <= 500 ms, <= 12 shots, <= 2 s total beam time, 12 s cool-down after every frame download.
pub fn shoot_cli(args: &[String]) -> Result<(), String> {
    let pulse: u32 = args.first().and_then(|a| a.parse().ok()).ok_or("usage: --shoot <pulse_ms> <window_ms> <delay,delay,...>")?;
    let window: f32 = args.get(1).and_then(|a| a.parse().ok()).ok_or("missing window_ms")?;
    let delays: Vec<u32> = args.get(2).ok_or("missing delays")?.split(',').filter_map(|d| d.trim().parse().ok()).collect();
    if pulse == 0 || pulse > 500 {
        return Err("pulse must be 1..=500 ms".into());
    }
    if delays.is_empty() || delays.len() > 12 {
        return Err("1..=12 shots per run".into());
    }
    if pulse as usize * delays.len() > 2000 {
        return Err("total beam time per run limited to 2 s".into());
    }
    // optional 4th argument: seconds to wait after each frame download, per shot (default 12)
    let gaps: Vec<u64> = args.get(3).map(|g| g.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let d = crate::settings::load_effective();
    let cfg = PanelCfg { host: d.panel_host, port: d.panel_port };
    let serial = new_serial();
    open_serial(&serial, &d.com_port, d.baud_rate)?;
    std::thread::sleep(Duration::from_millis(2500)); // an Arduino resets when the port opens
    // diagnostic: S4335W_PRE_WAIT=<seconds> idles the serial link before the first shot
    if let Some(w) = std::env::var("S4335W_PRE_WAIT").ok().and_then(|v| v.parse::<u64>().ok()) {
        println!("idling serial link for {w} s");
        std::thread::sleep(Duration::from_secs(w));
    }
    let out_dir = std::env::temp_dir().join("s4335w_shots");
    let _ = std::fs::create_dir_all(&out_dir);
    println!("shots -> {}", out_dir.display());
    for (i, delay) in delays.iter().enumerate() {
        let (tx, rx) = std::sync::mpsc::channel();
        // window_ms = 0 means: leave the panel's window register untouched (no write, no priming)
        let win = if window > 0.0 { Some((window * 10.0).round() as u32) } else { None };
        let frame = acquire(&tx, &cfg, Some((&serial, pulse, *delay)), win, "capture")?;
        let mut trig = f32::NAN;
        while let Ok(ev) = rx.try_recv() {
            if let HwEvent::Phase(Phase::Triggered(t)) = ev {
                trig = t * 1000.0;
            }
        }
        let px = crate::io::bytes_to_u16(&frame);
        let (w, h) = (3072usize, px.len() / 3072);
        let region = |r0: usize, r1: usize| -> f64 {
            let (c0, c1) = (w / 4, w * 3 / 4);
            let (mut s, mut n) = (0f64, 0f64);
            for y in r0..r1.min(h) {
                for &v in &px[y * w + c0..y * w + c1] {
                    s += v as f64;
                    n += 1.0;
                }
            }
            s / n.max(1.0)
        };
        // displayed image is rotated 180 deg: displayed top = raw bottom
        let (top, bottom, mean) = (region(h * 2 / 3, h), region(0, h / 3), region(0, h));
        // 8 horizontal bands of the displayed image, top to bottom (displayed top = raw bottom)
        let bands: Vec<String> = (0..8)
            .map(|b| format!("{:>5.0}", region(h - (b + 1) * h / 8, h - b * h / 8)))
            .collect();
        println!("        bands top->bottom: {}", bands.join(" "));
        let (mn, mx) = crate::pipeline::min_max(&px);
        println!(
            "shot {i}: delay {delay:>4} ms  pulse {pulse} ms  window {window} ms  trigger sent {trig:5.0} ms | mean {mean:7.0}  top {top:7.0}  bottom {bottom:7.0}  top/bot {:.2}  [{mn}-{mx}]",
            top / bottom.max(1.0)
        );
        let _ = crate::io::write_bin(&out_dir.join(format!("shot_{i}_d{delay}.bin")), &px);
        let gap = gaps.get(i).copied().unwrap_or(12).max(1);
        std::thread::sleep(Duration::from_secs(gap));
    }
    Ok(())
}
