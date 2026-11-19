//! Background image pipeline: raw -> corrected 16-bit (cached) -> optional Noise2Noise blend ->
//! window/level -> CLAHE -> sharpen -> colormap. Only the newest job is ever processed.
use crate::pipeline::{self, Field, Flat, Img16, SAT};
use crate::{clahe, colormap};
use eframe::egui;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

#[derive(Clone, PartialEq, Debug)]
pub struct Params {
    pub width: usize,
    pub height: usize,
    pub use_offset: bool,
    pub use_flat: bool,
    pub clip: u32,
    pub offset: i32,
    pub range: i32,
    pub invert: bool,
    pub clahe: bool,
    pub clahe_clip: f32,
    pub clahe_tiles: usize,
    pub sharpen: f32,
    pub cmap: String,
    /// 0 => Noise2Noise off
    pub n2n_strength: f32,
}

pub struct Job {
    pub id: u64,
    pub raw: Arc<Vec<u16>>,
    pub raw_id: u64,
    pub off: Option<(u64, Arc<Field>)>,
    pub flat: Option<(u64, Arc<Flat>)>,
    pub den: Option<(u64, Arc<Img16>)>,
    pub p: Params,
    pub auto: bool,
}

pub struct Display {
    pub w: usize,
    pub h: usize,
    pub ch: usize,
    pub data: Vec<u8>,
}

pub struct JobResult {
    pub id: u64,
    pub img: Option<Arc<Img16>>,
    pub s1: Option<Arc<Img16>>,
    pub s1_key: u64,
    pub disp: Option<Arc<Display>>,
    pub min: u16,
    pub max: u16,
    pub hist: Vec<u32>,
    pub auto: Option<(i32, i32)>,
    pub total_pixels: usize,
}

/// 8-bit pipeline shared by the viewport and the PNG export.
pub fn render(img: &Img16, p: &Params) -> Display {
    let mut g = pipeline::window_u8(&img.px, p.offset as f32, p.range as f32, p.invert);
    if p.clahe {
        g = clahe::apply(&g, img.w, img.h, p.clahe_clip, p.clahe_tiles);
    }
    if p.sharpen > 0.0 {
        g = pipeline::sharpen(&g, img.w, img.h, p.sharpen);
    }
    match colormap::lut(&p.cmap) {
        Some(l) => Display { w: img.w, h: img.h, ch: 3, data: colormap::apply(&g, &l) },
        None => Display { w: img.w, h: img.h, ch: 1, data: g },
    }
}

type Key = (u64, usize, usize, bool, u64, bool, u64, u32);

fn process(cache: &mut Option<(u64, Arc<Img16>)>, job: Job) -> JobResult {
    let p = &job.p;
    let key: Key = (
        job.raw_id,
        p.width,
        p.height,
        p.use_offset && job.off.is_some(),
        job.off.as_ref().map_or(0, |o| o.0),
        p.use_flat && job.flat.is_some(),
        job.flat.as_ref().map_or(0, |o| o.0),
        p.clip,
    );
    let mut hs = DefaultHasher::new();
    key.hash(&mut hs);
    let s1_key = hs.finish();

    let s1 = match cache {
        Some((k, img)) if *k == s1_key => Some(img.clone()),
        _ => {
            let off = if p.use_offset { job.off.as_ref().map(|o| o.1.as_ref()) } else { None };
            let flat = if p.use_flat { job.flat.as_ref().map(|o| o.1.as_ref()) } else { None };
            let built = pipeline::build_img16(&job.raw, p.width, p.height, off, flat, p.clip).map(Arc::new);
            if let Some(b) = &built {
                *cache = Some((s1_key, b.clone()));
            }
            built
        }
    };
    let total_pixels = job.raw.len();
    let Some(s1) = s1 else {
        return JobResult { id: job.id, img: None, s1: None, s1_key, disp: None, min: 0, max: 0, hist: vec![], auto: None, total_pixels };
    };

    // Noise2Noise blend (saturated pixels are left untouched)
    let img = match &job.den {
        Some((k, den)) if *k == s1_key && p.n2n_strength > 0.0 && den.px.len() == s1.px.len() => {
            let s = p.n2n_strength.clamp(0.0, 1.0);
            let px: Vec<u16> = s1
                .px
                .par_iter()
                .zip(den.px.par_iter())
                .map(|(&a, &b)| {
                    if a >= SAT {
                        a
                    } else {
                        (a as f32 + (b as f32 - a as f32) * s).round().clamp(0.0, SAT as f32) as u16
                    }
                })
                .collect();
            Arc::new(Img16 { w: s1.w, h: s1.h, px })
        }
        _ => s1.clone(),
    };

    let mut params = job.p.clone();
    let auto = if job.auto {
        let a = pipeline::auto_range(&img.px);
        params.offset = a.0;
        params.range = a.1;
        Some(a)
    } else {
        None
    };
    let (min, max) = pipeline::min_max(&img.px);
    let hist = pipeline::hist256(&img.px);
    let disp = Arc::new(render(&img, &params));
    JobResult { id: job.id, img: Some(img), s1: Some(s1), s1_key, disp: Some(disp), min, max, hist, auto, total_pixels }
}

pub fn spawn(ctx: egui::Context) -> (Sender<Job>, Receiver<JobResult>) {
    let (jtx, jrx) = channel::<Job>();
    let (rtx, rrx) = channel::<JobResult>();
    std::thread::spawn(move || {
        let mut cache = None;
        while let Ok(mut job) = jrx.recv() {
            while let Ok(newer) = jrx.try_recv() {
                job = newer;
            }
            let res = process(&mut cache, job);
            if rtx.send(res).is_err() {
                break;
            }
            ctx.request_repaint();
        }
    });
    (jtx, rrx)
}
