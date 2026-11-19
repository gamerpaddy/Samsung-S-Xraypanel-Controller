//! Noise2Noise (Lehtinen et al., NVlabs/noise2noise) in Rust on top of `candle`.
//!
//! The network is the `autoencoder` U-Net from the reference implementation (7 encoder convs,
//! 5 max-pools, nearest-neighbour up-sampling with skip concatenation, leaky-ReLU 0.1, linear last
//! layer) with one input channel instead of three. Training uses Adam (b1 .9, b2 .99, lr 3e-5 - 3e-4 diverged here -,
//! cosine ramp-down) with an L2 loss between two *independent noisy* realisations of the same scene.
//!
//! Two training sources are supported:
//!  * `Pairs`  - several exposures of the same view (true Noise2Noise, best quality)
//!  * `Single` - one exposure, split into two sub-sampled noisy views (Neighbor2Neighbor style)
use crate::pipeline::{Img16, SAT};
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Conv2d, Conv2dConfig, Init, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

const GRAD_CLIP: f32 = 1.0;
use rand::Rng;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

// ───────────────────────────── network ─────────────────────────────

struct Net {
    c: Vec<Conv2d>,
}

fn specs(b: usize) -> Vec<(&'static str, usize, usize)> {
    let (d, a, c) = (2 * b, b * 4 / 3, b * 2 / 3);
    vec![
        ("enc_conv0", 1, b),
        ("enc_conv1", b, b),
        ("enc_conv2", b, b),
        ("enc_conv3", b, b),
        ("enc_conv4", b, b),
        ("enc_conv5", b, b),
        ("enc_conv6", b, b),
        ("dec_conv5", 2 * b, d),
        ("dec_conv5b", d, d),
        ("dec_conv4", d + b, d),
        ("dec_conv4b", d, d),
        ("dec_conv3", d + b, d),
        ("dec_conv3b", d, d),
        ("dec_conv2", d + b, d),
        ("dec_conv2b", d, d),
        ("dec_conv1a", d + 1, a),
        ("dec_conv1b", a, c),
        ("dec_conv1", c, 1),
    ]
}

fn lrelu(x: &Tensor) -> candle_core::Result<Tensor> {
    if std::env::var("N2N_RELU2").is_ok() {
        // relu(x) - 0.1*relu(-x)
        return x.relu()? - (x.neg()?.relu()? * 0.1)?;
    }
    // max(x, 0.1x) == leaky-ReLU(0.1)
    x.maximum(&(x * 0.1)?)
}

fn up(x: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    x.upsample_nearest2d(h * 2, w * 2)
}

impl Net {
    fn new(vb: VarBuilder, base: usize) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let mut c = Vec::new();
        let n = specs(base).len();
        for (idx, (name, i, o)) in specs(base).into_iter().enumerate() {
            // He init (gain sqrt 2) everywhere. The linear output layer starts near zero because the
            // network predicts a residual (see `forward`), so training starts at the identity map.
            let gain: f64 = if idx == n - 1 { 0.05 } else { std::f64::consts::SQRT_2 };
            let vb = vb.pp(name);
            let ws = vb.get_with_hints(
                (o, i, 3, 3),
                "weight",
                Init::Randn { mean: 0.0, stdev: gain / ((i * 9) as f64).sqrt() },
            )?;
            let bs = vb.get_with_hints(o, "bias", Init::Const(0.0))?;
            c.push(Conv2d::new(ws, Some(bs), cfg));
        }
        Ok(Self { c })
    }

    fn act(&self, i: usize, x: &Tensor) -> candle_core::Result<Tensor> {
        lrelu(&self.c[i].forward(x)?)
    }

    /// x: (N, 1, H, W) with H, W multiples of 32, values in [-0.5, 0.5]
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut n = self.act(0, x)?;
        n = self.act(1, &n)?;
        n = n.max_pool2d(2)?;
        let s1 = n.clone();
        n = self.act(2, &n)?.max_pool2d(2)?;
        let s2 = n.clone();
        n = self.act(3, &n)?.max_pool2d(2)?;
        let s3 = n.clone();
        n = self.act(4, &n)?.max_pool2d(2)?;
        let s4 = n.clone();
        n = self.act(5, &n)?.max_pool2d(2)?;
        n = self.act(6, &n)?;

        n = Tensor::cat(&[&up(&n)?, &s4], 1)?;
        n = self.act(7, &n)?;
        n = self.act(8, &n)?;
        n = Tensor::cat(&[&up(&n)?, &s3], 1)?;
        n = self.act(9, &n)?;
        n = self.act(10, &n)?;
        n = Tensor::cat(&[&up(&n)?, &s2], 1)?;
        n = self.act(11, &n)?;
        n = self.act(12, &n)?;
        n = Tensor::cat(&[&up(&n)?, &s1], 1)?;
        n = self.act(13, &n)?;
        n = self.act(14, &n)?;
        n = Tensor::cat(&[&up(&n)?, x], 1)?;
        n = self.act(15, &n)?;
        n = self.act(16, &n)?;
        // residual output: the net only has to learn the noise-free correction to its input
        x + self.c[17].forward(&n)?
    }
}

// ───────────────────────────── engine ─────────────────────────────

pub enum TrainData {
    Pairs(Vec<Arc<Img16>>),
    Single(Arc<Img16>),
}

pub struct Engine {
    pub dev: Device,
    varmap: VarMap,
    net: Net,
    pub width: usize,
    pub ready: bool,
}

#[inline]
fn norm(v: u16) -> f32 {
    v as f32 / SAT as f32 - 0.5
}

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

impl Engine {
    pub fn new(width: usize) -> Result<Self> {
        let dev = pick_device();
        let (varmap, net) = Self::build(width, &dev)?;
        Ok(Self { dev, varmap, net, width, ready: false })
    }

    fn build(width: usize, dev: &Device) -> Result<(VarMap, Net)> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let net = Net::new(vb, width)?;
        Ok((varmap, net))
    }

    pub fn set_width(&mut self, width: usize) -> Result<()> {
        let (v, n) = Self::build(width, &self.dev)?;
        self.varmap = v;
        self.net = n;
        self.width = width;
        self.ready = false;
        Ok(())
    }

    pub fn is_gpu(&self) -> bool {
        !matches!(self.dev, Device::Cpu)
    }

    pub fn device_name(&self) -> String {
        if self.is_gpu() { "CUDA GPU".into() } else { "CPU".into() }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        self.varmap.load(path).map_err(|e| anyhow!("{e} (was this file trained with width {}?)", self.width))?;
        self.ready = true;
        Ok(())
    }

    /// Same weights, but without autograd tracking (much lower memory for inference).
    fn detached(&self) -> Result<Net> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for (k, v) in self.varmap.data().lock().unwrap().iter() {
            map.insert(k.clone(), v.as_tensor().detach());
        }
        let vb = VarBuilder::from_tensors(map, DType::F32, &self.dev);
        Ok(Net::new(vb, self.width)?)
    }

    /// Tiled full-image inference. Edge pixels are replicated, tiles overlap by `margin`.
    pub fn infer(&self, img: &Img16, progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<u16>> {
        let net = self.detached()?;
        let (w, h) = (img.w, img.h);
        let margin = 32usize;
        let core = if self.is_gpu() { 480usize } else { 256 };
        let tile = core + 2 * margin; // multiple of 32 by construction
        let tiles_x = w.div_ceil(core);
        let tiles_y = h.div_ceil(core);
        let total = tiles_x * tiles_y;
        let mut out = vec![0u16; w * h];
        let mut buf = vec![0f32; tile * tile];
        let mut done = 0;
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let (x0, y0) = ((tx * core) as isize - margin as isize, (ty * core) as isize - margin as isize);
                for yy in 0..tile {
                    let sy = (y0 + yy as isize).clamp(0, h as isize - 1) as usize;
                    for xx in 0..tile {
                        let sx = (x0 + xx as isize).clamp(0, w as isize - 1) as usize;
                        buf[yy * tile + xx] = norm(img.px[sy * w + sx]);
                    }
                }
                let x = Tensor::from_slice(&buf, (1, 1, tile, tile), &self.dev)?;
                let y = net.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
                let (cx, cy) = (tx * core, ty * core);
                for yy in 0..core.min(h - cy) {
                    for xx in 0..core.min(w - cx) {
                        let v = (y[(margin + yy) * tile + margin + xx] + 0.5) * SAT as f32;
                        out[(cy + yy) * w + cx + xx] = v.round().clamp(0.0, SAT as f32) as u16;
                    }
                }
                done += 1;
                progress(done, total);
            }
        }
        Ok(out)
    }

    /// Returns the final (smoothed) training loss.
    pub fn train(
        &mut self,
        data: &TrainData,
        steps: usize,
        batch: usize,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(usize, f32),
    ) -> Result<TrainResult> {
        let base_lr = std::env::var("N2N_LR").ok().and_then(|v| v.parse().ok()).unwrap_or(3e-5f64);
        let vars = self.varmap.all_vars();
        let mut opt = AdamW::new(
            vars.clone(),
            ParamsAdamW { lr: base_lr, beta1: 0.9, beta2: 0.99, eps: 1e-8, weight_decay: 0.0 },
        )?;
        let mut rng = rand::thread_rng();
        let gains = match data {
            TrainData::Pairs(f) => frame_gains(f),
            TrainData::Single(_) => vec![1.0],
        };
        let mut smooth = f32::NAN;
        for step in 0..steps {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let t = step as f64 / steps.max(1) as f64;
            let ramp_up = (t / 0.02).min(1.0);
            let ramp_down = if t > 0.7 { 0.5 * (1.0 + (std::f64::consts::PI * (t - 0.7) / 0.3).cos()) } else { 1.0 };
            opt.set_learning_rate((base_lr * ramp_up * ramp_down).max(1e-6));

            let (xv, yv, mv, p) = sample_batch(data, batch, &gains, &mut rng)?;
            let x = Tensor::from_vec(xv, (batch, 1, p, p), &self.dev)?;
            let y = Tensor::from_vec(yv, (batch, 1, p, p), &self.dev)?;
            let m = Tensor::from_vec(mv, (batch, 1, p, p), &self.dev)?;
            let pred = self.net.forward(&x)?;
            // masked L2: saturated (clipped) pixels carry no noise information and are excluded
            let loss = ((pred - &y)?.sqr()? * &m)?.sum_all()?.broadcast_div(&(m.sum_all()? + 1.0)?)?;
            // backward + global gradient-norm clipping (keeps the un-normalised U-Net stable)
            let mut grads = loss.backward()?;
            let mut total: Option<Tensor> = None;
            for v in &vars {
                if let Some(g) = grads.get(v.as_tensor()) {
                    let s = g.sqr()?.sum_all()?;
                    total = Some(match total {
                        Some(t) => (t + s)?,
                        None => s,
                    });
                }
            }
            if let Some(t) = total {
                let norm = t.to_scalar::<f32>()?.sqrt();
                if !norm.is_finite() {
                    continue; // skip a poisoned step
                }
                if norm > GRAD_CLIP {
                    let sc = (GRAD_CLIP / norm) as f64;
                    for v in &vars {
                        if let Some(g) = grads.get(v.as_tensor()) {
                            let g2 = (g * sc)?;
                            grads.insert(v.as_tensor(), g2);
                        }
                    }
                }
            }
            opt.step(&grads)?;
            if step % 5 == 0 || step + 1 == steps {
                let l = loss.to_scalar::<f32>()?;
                smooth = if smooth.is_nan() { l } else { 0.8 * smooth + 0.2 * l };
                progress(step + 1, smooth);
            }
        }
        self.ready = true;

        // held-out check: does the model predict the *other* noisy frame better than the input itself?
        let (mut sb, mut sa, mut sm) = (0f64, 0f64, 0f64);
        for _ in 0..8 {
            let (xv, yv, mv, p) = sample_batch(data, 4, &gains, &mut rng)?;
            let x = Tensor::from_vec(xv, (4, 1, p, p), &self.dev)?;
            let y = Tensor::from_vec(yv, (4, 1, p, p), &self.dev)?;
            let m = Tensor::from_vec(mv, (4, 1, p, p), &self.dev)?;
            let pred = self.net.forward(&x)?.detach();
            sb += ((&x - &y)?.sqr()? * &m)?.sum_all()?.to_scalar::<f32>()? as f64;
            sa += ((&pred - &y)?.sqr()? * &m)?.sum_all()?.to_scalar::<f32>()? as f64;
            sm += m.sum_all()?.to_scalar::<f32>()? as f64;
        }
        let sm = sm.max(1.0);
        let counts = SAT as f64;
        Ok(TrainResult { loss: smooth, rms_before: (sb / sm).sqrt() * counts, rms_after: (sa / sm).sqrt() * counts })
    }
}

pub struct TrainResult {
    pub loss: f32,
    /// RMS error (in counts) of the raw input against the second exposure
    pub rms_before: f64,
    /// RMS error (in counts) of the model output against the second exposure
    pub rms_after: f64,
}

/// Per-frame brightness gains (relative to frame 0), measured on pixels that are unsaturated in every
/// frame. Exposure dose varies a little from shot to shot; without this the network learns to "average"
/// the brightness, which darkens bright regions.
fn frame_gains(frames: &[Arc<Img16>]) -> Vec<f32> {
    let n = frames[0].px.len();
    let mut sums = vec![0f64; frames.len()];
    for i in 0..n {
        if frames.iter().all(|f| f.px[i] < SAT) {
            for (k, f) in frames.iter().enumerate() {
                sums[k] += f.px[i] as f64;
            }
        }
    }
    sums.iter().map(|&s| if s > 0.0 { (sums[0] / s) as f32 } else { 1.0 }).collect()
}

#[inline]
fn normf(v: f32) -> f32 {
    v / SAT as f32 - 0.5
}

/// Builds a batch of (input, target, mask) patches, values in [-0.5, 0.5]; mask is 0 where either pixel is saturated.
fn sample_batch(data: &TrainData, batch: usize, gains: &[f32], rng: &mut impl Rng) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize)> {
    let sat = SAT;
    match data {
        TrainData::Pairs(frames) => {
            let n = frames.len();
            if n < 2 {
                return Err(anyhow!("need at least two frames of the same scene"));
            }
            let (w, h) = (frames[0].w, frames[0].h);
            let p = (128usize.min(w.min(h)) / 32) * 32;
            if p < 32 {
                return Err(anyhow!("image too small"));
            }
            let (mut xs, mut ys, mut ms) = (Vec::with_capacity(batch * p * p), Vec::with_capacity(batch * p * p), Vec::with_capacity(batch * p * p));
            for _ in 0..batch {
                let i = rng.gen_range(0..n);
                let mut j = rng.gen_range(0..n - 1);
                if j >= i {
                    j += 1;
                }
                // prefer patches that are not mostly clipped
                let (mut best, mut best_frac) = ((0usize, 0usize), -1.0f32);
                for _try in 0..24 {
                    let (x0, y0) = (rng.gen_range(0..=w - p), rng.gen_range(0..=h - p));
                    let mut valid = 0usize;
                    for yy in (0..p).step_by(4) {
                        let o = (y0 + yy) * w + x0;
                        for xx in (0..p).step_by(4) {
                            if frames[i].px[o + xx] < sat && frames[j].px[o + xx] < sat {
                                valid += 1;
                            }
                        }
                    }
                    let frac = valid as f32 / ((p / 4) * (p / 4)) as f32;
                    if frac > best_frac {
                        best_frac = frac;
                        best = (x0, y0);
                    }
                    if frac >= 0.6 {
                        break;
                    }
                }
                let (x0, y0) = best;
                for yy in 0..p {
                    let o = (y0 + yy) * w + x0;
                    for xx in 0..p {
                        let (a, b) = (frames[i].px[o + xx], frames[j].px[o + xx]);
                        xs.push(normf((a as f32 * gains[i]).min(sat as f32)));
                        ys.push(normf((b as f32 * gains[j]).min(sat as f32)));
                        ms.push(if a < sat && b < sat { 1.0 } else { 0.0 });
                    }
                }
            }
            Ok((xs, ys, ms, p))
        }
        TrainData::Single(img) => {
            let (w, h) = (img.w, img.h);
            let p = (128usize.min(w.min(h) / 2) / 32) * 32; // output patch; source crop is 2p x 2p
            if p < 32 {
                return Err(anyhow!("image too small"));
            }
            let (mut xs, mut ys, mut ms) = (Vec::with_capacity(batch * p * p), Vec::with_capacity(batch * p * p), Vec::with_capacity(batch * p * p));
            for _ in 0..batch {
                let (x0, y0) = (rng.gen_range(0..=w - 2 * p), rng.gen_range(0..=h - 2 * p));
                for cy in 0..p {
                    for cx in 0..p {
                        let a = rng.gen_range(0..4usize);
                        let mut b = rng.gen_range(0..3usize);
                        if b >= a {
                            b += 1;
                        }
                        let at = |k: usize| img.px[(y0 + 2 * cy + k / 2) * w + x0 + 2 * cx + k % 2];
                        let (va, vb) = (at(a), at(b));
                        xs.push(norm(va));
                        ys.push(norm(vb));
                        ms.push(if va < sat && vb < sat { 1.0 } else { 0.0 });
                    }
                }
            }
            Ok((xs, ys, ms, p))
        }
    }
}

// ───────────────────────────── background thread ─────────────────────────────

pub enum Cmd {
    Infer { key: u64, img: Arc<Img16> },
    Train { data: TrainData, steps: usize, batch: usize },
    Load(PathBuf),
    Save(PathBuf),
    SetWidth(usize),
}

pub enum Ev {
    Info { device: String, ready: bool, width: usize },
    Status(String),
    Progress { step: usize, total: usize, loss: f32 },
    Inferred { key: u64, img: Arc<Img16> },
    TrainDone { loss: f32, rms_before: f64, rms_after: f64 },
    Error(String),
}

pub struct Handle {
    pub tx: Sender<Cmd>,
    pub rx: Receiver<Ev>,
    pub cancel: Arc<AtomicBool>,
}

pub fn weights_path(dir: &Path, width: usize) -> PathBuf {
    dir.join(format!("noise2noise_w{width}.safetensors"))
}

pub fn spawn(ctx: eframe::egui::Context, dir: PathBuf, width: usize) -> Handle {
    let (tx, crx) = channel::<Cmd>();
    let (etx, rx) = channel::<Ev>();
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    std::thread::spawn(move || {
        let send = |e: Ev| {
            let _ = etx.send(e);
            ctx.request_repaint();
        };
        let mut eng = match Engine::new(width) {
            Ok(e) => e,
            Err(e) => {
                send(Ev::Error(format!("Noise2Noise init failed: {e}")));
                return;
            }
        };
        let autoload = |eng: &mut Engine, send: &dyn Fn(Ev)| {
            let p = weights_path(&dir, eng.width);
            if p.exists() {
                match eng.load(&p) {
                    Ok(()) => send(Ev::Status(format!("Loaded {}", p.file_name().unwrap().to_string_lossy()))),
                    Err(e) => send(Ev::Error(format!("Could not load weights: {e}"))),
                }
            }
            send(Ev::Info { device: eng.device_name(), ready: eng.ready, width: eng.width });
        };
        autoload(&mut eng, &send);
        while let Ok(cmd) = crx.recv() {
            match cmd {
                Cmd::SetWidth(w) => {
                    if let Err(e) = eng.set_width(w) {
                        send(Ev::Error(e.to_string()));
                    }
                    autoload(&mut eng, &send);
                }
                Cmd::Load(p) => {
                    match eng.load(&p) {
                        Ok(()) => send(Ev::Status(format!("Loaded {}", p.display()))),
                        Err(e) => send(Ev::Error(e.to_string())),
                    }
                    send(Ev::Info { device: eng.device_name(), ready: eng.ready, width: eng.width });
                }
                Cmd::Save(p) => match eng.save(&p) {
                    Ok(()) => send(Ev::Status(format!("Saved {}", p.display()))),
                    Err(e) => send(Ev::Error(e.to_string())),
                },
                Cmd::Infer { key, img } => {
                    if !eng.ready {
                        send(Ev::Error("No trained Noise2Noise weights yet - train or load a model first".into()));
                        continue;
                    }
                    let t0 = Instant::now();
                    let r = eng.infer(&img, &mut |d, t| {
                        send(Ev::Status(format!("Noise2Noise: denoising tile {d}/{t}")));
                    });
                    match r {
                        Ok(px) => {
                            send(Ev::Status(format!("Noise2Noise: done in {:.1}s", t0.elapsed().as_secs_f32())));
                            send(Ev::Inferred { key, img: Arc::new(Img16 { w: img.w, h: img.h, px }) });
                        }
                        Err(e) => send(Ev::Error(format!("Denoise failed: {e}"))),
                    }
                }
                Cmd::Train { data, steps, batch } => {
                    c2.store(false, Ordering::Relaxed);
                    let r = eng.train(&data, steps, batch, &c2, &mut |s, l| {
                        send(Ev::Progress { step: s, total: steps, loss: l });
                    });
                    match r {
                        Ok(r) => {
                            let p = weights_path(&dir, eng.width);
                            if let Err(e) = eng.save(&p) {
                                send(Ev::Error(format!("Could not save weights: {e}")));
                            }
                            send(Ev::TrainDone { loss: r.loss, rms_before: r.rms_before, rms_after: r.rms_after });
                            send(Ev::Info { device: eng.device_name(), ready: eng.ready, width: eng.width });
                        }
                        Err(e) => send(Ev::Error(format!("Training failed: {e}"))),
                    }
                }
            }
        }
    });
    Handle { tx, rx, cancel }
}

// ───────────────────────────── self test (CLI) ─────────────────────────────

fn gauss(rng: &mut impl Rng) -> f32 {
    let (u1, u2): (f32, f32) = (rng.gen_range(1e-7..1.0), rng.gen());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// `s4335w --n2n-selftest <image.png> [steps] [width]`
/// Corrupts an image with signal-dependent noise twice, trains on the noisy pairs and reports PSNR.
pub fn selftest(args: &[String]) -> Result<()> {
    let path = args.first().ok_or_else(|| anyhow!("usage: --n2n-selftest <image> [steps] [width]"))?;
    let steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let width: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);
    let (w, h, px) = crate::io::load_gray16(Path::new(path))?;
    let mx = *px.iter().max().unwrap_or(&1).max(&1) as f32;
    let (cw, ch) = (w.min(512) / 32 * 32, h.min(512) / 32 * 32);
    let (ox, oy) = ((w - cw) / 2, (h - ch) / 2);
    // bright enough that part of the image clips, like the real detector
    let clean: Vec<f32> = (0..cw * ch)
        .map(|i| 2000.0 + px[(oy + i / cw) * w + ox + i % cw] as f32 / mx * 17000.0)
        .collect();
    let mut rng = rand::thread_rng();
    let mut noisy = || -> Arc<Img16> {
        let gain = 1.0 + rng.gen_range(-0.06f32..0.06); // shot-to-shot dose variation
        let p = clean
            .iter()
            .map(|&c| (c * gain + gauss(&mut rng) * c.sqrt() * 3.0).clamp(0.0, SAT as f32) as u16)
            .collect();
        Arc::new(Img16 { w: cw, h: ch, px: p })
    };
    let frames: Vec<_> = (0..4).map(|_| noisy()).collect();
    // evaluate on unclipped pixels only, against the clean image (gain-matched mean)
    let valid: Vec<usize> = (0..clean.len()).filter(|&i| clean[i] < 14000.0).collect();
    println!("clipped fraction of frame 0: {:.1}%", frames[0].px.iter().filter(|&&v| v >= SAT).count() as f32 * 100.0 / clean.len() as f32);
    let psnr = |a: &[u16]| -> f32 {
        let mse = valid.iter().map(|&i| (a[i] as f32 - clean[i]).powi(2)).sum::<f32>() / valid.len() as f32;
        10.0 * ((SAT as f32).powi(2) / mse).log10()
    };
    let bias = |a: &[u16]| -> f32 { valid.iter().map(|&i| a[i] as f32 - clean[i]).sum::<f32>() / valid.len() as f32 };
    let mut eng = Engine::new(width)?;
    println!("device: {}, image {cw}x{ch}, steps {steps}, width {width}", eng.device_name());
    println!("PSNR noisy input        : {:.2} dB", psnr(&frames[0].px));
    let t0 = Instant::now();
    let cancel = AtomicBool::new(false);
    let mut next = 1usize;
    let res = eng.train(&TrainData::Pairs(frames.clone()), steps, 4, &cancel, &mut |s, l| {
        if s >= next || s == steps {
            next = (s / 25 + 1) * 25;
            println!("  step {s:5}/{steps}  loss {l:.6}  ({:.2}s/step)", t0.elapsed().as_secs_f32() / s as f32);
        }
    })?;
    println!("final loss {:.5}; held-out RMS vs 2nd exposure: raw {:.1} -> model {:.1} counts", res.loss, res.rms_before, res.rms_after);
    let out = eng.infer(&frames[0], &mut |_, _| {})?;
    println!("PSNR after Noise2Noise  : {:.2} dB", psnr(&out));
    println!("mean bias vs clean (input): {:+.1} counts   (model): {:+.1} counts", bias(&frames[0].px), bias(&out));
    Ok(())
}

/// `s4335w --n2n-eval <frame.bin> [weights.safetensors] [width]`
/// Runs the trained model on a 1024x1024 crop of a real frame and reports what it changes.
pub fn eval_cli(args: &[String]) -> Result<()> {
    use crate::{io, pipeline};
    let bin = args.first().ok_or_else(|| anyhow!("usage: --n2n-eval <frame.bin> [weights] [width]"))?;
    let base = crate::settings::find_base_dir();
    let width: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);
    let wpath = args.get(1).map(PathBuf::from).unwrap_or_else(|| weights_path(&base, width));
    let raw = io::read_bin(Path::new(bin))?;
    let flat = io::load_flat(&base.join("flatfield.bin"), 3072).ok();
    let img = pipeline::build_img16(&raw, 3072, 2560, None, flat.as_ref(), 16383).ok_or_else(|| anyhow!("bad frame"))?;
    let (cw, ch, x0, y0) = (1024usize, 1024usize, img.w / 2 - 700, img.h / 2 - 512);
    let mut px = Vec::with_capacity(cw * ch);
    for y in 0..ch {
        px.extend_from_slice(&img.px[(y0 + y) * img.w + x0..(y0 + y) * img.w + x0 + cw]);
    }
    let crop = Img16 { w: cw, h: ch, px };
    let mut eng = Engine::new(width)?;
    eng.load(&wpath)?;
    println!("device {}, weights {}", eng.device_name(), wpath.display());
    let out = eng.infer(&crop, &mut |_, _| {})?;
    let n = out.len() as f64;
    let diff: Vec<f64> = out.iter().zip(&crop.px).map(|(&a, &b)| a as f64 - b as f64).collect();
    let mean = diff.iter().sum::<f64>() / n;
    let sd = (diff.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n).sqrt();
    let hp = |p: &[u16]| -> f64 {
        let mut s = 0f64;
        let mut c = 0f64;
        for y in 1..ch - 1 {
            for x in 1..cw - 1 {
                let i = y * cw + x;
                let l = 4.0 * p[i] as f64 - p[i - 1] as f64 - p[i + 1] as f64 - p[i - cw] as f64 - p[i + cw] as f64;
                s += l * l;
                c += 1.0;
            }
        }
        (s / c).sqrt()
    };
    let (mi, mo) = (crop.px.iter().map(|&v| v as f64).sum::<f64>() / n, out.iter().map(|&v| v as f64).sum::<f64>() / n);
    println!("mean in {mi:.1} out {mo:.1}  (shift {:+.1})", mo - mi);
    println!("change per pixel: std {sd:.1} counts");
    println!("high-pass energy in {:.1} out {:.1}  (ratio {:.2})", hp(&crop.px), hp(&out), hp(&out) / hp(&crop.px));
    let (o, r) = pipeline::auto_range(&crop.px);
    let a = pipeline::window_u8(&crop.px, o as f32, r as f32, false);
    let b = pipeline::window_u8(&out, o as f32, r as f32, false);
    let mut side = Vec::with_capacity(cw * 2 * ch);
    for y in 0..ch {
        side.extend_from_slice(&a[y * cw..(y + 1) * cw]);
        side.extend_from_slice(&b[y * cw..(y + 1) * cw]);
    }
    let p = std::env::temp_dir().join("n2n_eval_side_by_side.png");
    io::save_png(&p, cw * 2, ch, &side, 1, None)?;
    println!("saved {}", p.display());
    Ok(())
}
