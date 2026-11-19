//! Raw-frame correction + 16-bit helpers (port of the numba / numpy code in the Python viewer).
use rayon::prelude::*;

pub const SAT: u16 = 16383; // 14-bit max
pub const FULL_W: usize = 3072;
pub const FULL_H: usize = 2560;
pub const CROP_W: usize = 3041;
pub const CROP_H: usize = 2466;
pub const EXPECTED_BYTES: usize = FULL_W * FULL_H * 2;

#[derive(Clone)]
pub struct Img16 {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u16>,
}

/// Dark-frame / offset field, already aligned with the rotated + cropped image.
pub struct Field {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u16>,
}

/// Flat field, already aligned with the rotated + cropped image.
pub struct Flat {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u16>,
    pub mean: f32,
}

impl Flat {
    pub fn new(w: usize, h: usize, px: Vec<u16>) -> Self {
        let (sum, n) = px
            .par_iter()
            .filter(|&&v| v > 0)
            .fold(|| (0f64, 0u64), |a, &v| (a.0 + v as f64, a.1 + 1))
            .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        let mean = if n > 0 { (sum / n as f64) as f32 } else { 1.0 };
        Self { w, h, px, mean }
    }
}

/// Port of `get_current_img16`: reshape, rotate 180 deg, crop padding, offset, flat, clip.
pub fn build_img16(
    raw: &[u16],
    width: usize,
    height: usize,
    off: Option<&Field>,
    flat: Option<&Flat>,
    clip_thr: u32,
) -> Option<Img16> {
    let avail = raw.len();
    let mut w = width.max(1);
    let mut h = height.max(1);
    let need;
    if avail == CROP_W * CROP_H {
        w = CROP_W;
        h = CROP_H;
        need = avail;
    } else {
        let mut n = w * h;
        if n > avail {
            h = avail / w;
            n = h * w;
        }
        need = n;
    }
    if h == 0 || w == 0 {
        return None;
    }
    let src = &raw[..need];

    // rotate 180 deg == reverse the flattened array; then crop the dead border
    let (ow, oh, r0, c0) = if h == FULL_H && w == FULL_W {
        (w - 17 - 14, h - 92 - 2, 92usize, 17usize)
    } else {
        (w, h, 0, 0)
    };
    let use_off = off.filter(|f| f.w == ow && f.h == oh);
    let use_flat = flat.filter(|f| f.w == ow && f.h == oh);
    let mut px = vec![0u16; ow * oh];

    px.par_chunks_mut(ow).enumerate().for_each(|(y, row)| {
        let sy = y + r0;
        for (x, out) in row.iter_mut().enumerate() {
            let sx = x + c0;
            let raw_v = src[need - 1 - (sy * w + sx)];
            let sat = raw_v >= SAT;
            let mut v = raw_v as i32;
            let i = y * ow + x;
            // saturated pixels carry no signal: subtracting the (noisy) dark frame there would only
            // add grain to clean white areas
            if let (Some(o), false) = (use_off, sat) {
                v = (v - o.px[i] as i32).clamp(0, 65535);
            }
            if let Some(f) = use_flat {
                let fv = f.px[i];
                let fv = if fv == 0 { 1.0 } else { fv as f32 };
                let c = if sat { v as f32 } else { (v as f32 / fv) * f.mean };
                v = c.clamp(0.0, 65535.0) as i32;
            }
            if v as u32 > clip_thr {
                v = SAT as i32;
            }
            *out = v as u16;
        }
    });
    Some(Img16 { w: ow, h: oh, px })
}

/// 8-bit window/level (+ optional inversion).
pub fn window_u8(img: &[u16], offset: f32, range: f32, invert: bool) -> Vec<u8> {
    let k = 255.0 / range.max(1.0);
    img.par_iter()
        .map(|&v| {
            let b = ((v as f32 - offset) * k).clamp(0.0, 255.0) as u8;
            if invert {
                255 - b
            } else {
                b
            }
        })
        .collect()
}

/// 256-bin histogram of the 14-bit range.
pub fn hist256(img: &[u16]) -> Vec<u32> {
    img.par_chunks(1 << 16)
        .map(|c| {
            let mut h = [0u32; 256];
            for &v in c {
                h[(v.min(SAT) >> 6) as usize] += 1;
            }
            h
        })
        .reduce(
            || [0u32; 256],
            |mut a, b| {
                for i in 0..256 {
                    a[i] += b[i];
                }
                a
            },
        )
        .to_vec()
}

pub fn hist16k(img: &[u16]) -> Vec<u32> {
    img.par_chunks(1 << 18)
        .map(|c| {
            let mut h = vec![0u32; 16384];
            for &v in c {
                h[v.min(SAT) as usize] += 1;
            }
            h
        })
        .reduce(
            || vec![0u32; 16384],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        )
}

/// 5th-95th percentile auto range -> (offset, range)
pub fn auto_range(img: &[u16]) -> (i32, i32) {
    let hist = hist16k(img);
    let total = img.len() as f64;
    let (p5, p95) = (total * 0.05, total * 0.95);
    let (mut cum, mut mn, mut mx) = (0f64, 0i32, 16383i32);
    for (i, &c) in hist.iter().enumerate() {
        cum += c as f64;
        if cum >= p5 && mn == 0 {
            mn = i as i32;
        }
        if cum >= p95 {
            mx = i as i32;
            break;
        }
    }
    (mn, (mx - mn).max(1))
}

pub fn min_max(img: &[u16]) -> (u16, u16) {
    img.par_iter()
        .fold(|| (u16::MAX, 0u16), |a, &v| (a.0.min(v), a.1.max(v)))
        .reduce(|| (u16::MAX, 0), |a, b| (a.0.min(b.0), a.1.max(b.1)))
}

/// Stretch to the full 16-bit range (ignoring zeros). `percentile <= 0` => plain min/max.
pub fn stretch16(img: &[u16], percentile: f32) -> Vec<u16> {
    let apply = |min_val: f64, scale: f64| -> Vec<u16> {
        img.par_iter()
            .map(|&v| {
                if v > 0 {
                    ((v as f64 - min_val) * scale).clamp(0.0, 65535.0) as u16
                } else {
                    0
                }
            })
            .collect()
    };
    if percentile <= 0.0 {
        let (mn, mx) = img
            .par_iter()
            .filter(|&&v| v > 0)
            .fold(|| (u16::MAX, 0u16), |a, &v| (a.0.min(v), a.1.max(v)))
            .reduce(|| (u16::MAX, 0), |a, b| (a.0.min(b.0), a.1.max(b.1)));
        if mx <= mn {
            return img.to_vec();
        }
        return apply(mn as f64, 65535.0 / (mx - mn) as f64);
    }
    let mut hist = vec![0u32; 65536];
    let mut valid = 0u64;
    for &v in img {
        if v > 0 {
            hist[v as usize] += 1;
            valid += 1;
        }
    }
    if valid == 0 {
        return vec![0; img.len()];
    }
    let cut = valid as f64 * (percentile as f64 / 100.0);
    let (mut min_val, mut max_val) = (0usize, 65535usize);
    let mut cum = 0f64;
    for i in 1..65536 {
        cum += hist[i] as f64;
        if cum >= cut {
            min_val = i;
            break;
        }
    }
    cum = 0.0;
    for i in (1..65536).rev() {
        cum += hist[i] as f64;
        if cum >= cut {
            max_val = i;
            break;
        }
    }
    let scale = if max_val <= min_val { 1.0 } else { 65535.0 / (max_val - min_val) as f64 };
    apply(min_val as f64, scale)
}

/// Unsharp-style edge sharpening, continuous in `strength` (0..2):
/// out = c + strength * (4c - n - s - e - w)   (strength 1 == the classic 5/-1 kernel)
/// Border handling is OpenCV BORDER_REFLECT_101.
pub fn sharpen(src: &[u8], w: usize, h: usize, strength: f32) -> Vec<u8> {
    if strength <= 0.0 || w < 2 || h < 2 {
        return src.to_vec();
    }
    let refl = |i: isize, n: usize| -> usize {
        if i < 0 {
            (-i) as usize
        } else if i as usize >= n {
            2 * n - 2 - i as usize
        } else {
            i as usize
        }
    };
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let ym = refl(y as isize - 1, h) * w;
        let yp = refl(y as isize + 1, h) * w;
        let yc = y * w;
        for x in 0..w {
            let xm = refl(x as isize - 1, w);
            let xp = refl(x as isize + 1, w);
            let c = src[yc + x] as f32;
            let lap = 4.0 * c - (src[ym + x] as f32 + src[yp + x] as f32 + src[yc + xp] as f32 + src[yc + xm] as f32);
            row[x] = (c + strength * lap).round().clamp(0.0, 255.0) as u8;
        }
    });
    out
}
