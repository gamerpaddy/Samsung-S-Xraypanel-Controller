//! CLAHE, following OpenCV's `cv::CLAHE` (8-bit) step by step so results match the Python version.
use rayon::prelude::*;

const HIST: usize = 256;

fn reflect101(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let n = n as isize;
    let mut i = i;
    while i < 0 || i >= n {
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * (n - 1) - i;
        }
    }
    i as usize
}

pub fn apply(src: &[u8], w: usize, h: usize, clip: f32, tiles: usize) -> Vec<u8> {
    let tiles = tiles.max(1);
    let (tx_n, ty_n) = (tiles, tiles);
    if w < tx_n || h < ty_n {
        return src.to_vec();
    }
    // pad to a multiple of the tile grid (reflect-101), like OpenCV
    let pw = w.div_ceil(tx_n) * tx_n;
    let ph = h.div_ceil(ty_n) * ty_n;
    let padded: Vec<u8>;
    let s: &[u8] = if pw != w || ph != h {
        let mut p = vec![0u8; pw * ph];
        p.par_chunks_mut(pw).enumerate().for_each(|(y, row)| {
            let sy = reflect101(y as isize, h);
            for (x, o) in row.iter_mut().enumerate() {
                *o = src[sy * w + reflect101(x as isize, w)];
            }
        });
        padded = p;
        &padded
    } else {
        src
    };
    let (tw, th) = (pw / tx_n, ph / ty_n);
    let total = (tw * th) as f32;
    let clip_limit = ((clip * total / HIST as f32) as i32).max(1) as u32;
    let lut_scale = (HIST as f32 - 1.0) / total;

    // one LUT per tile
    let luts: Vec<[u8; HIST]> = (0..tx_n * ty_n)
        .into_par_iter()
        .map(|t| {
            let (tx, ty) = (t % tx_n, t / tx_n);
            let mut hist = [0u32; HIST];
            for y in 0..th {
                let row = &s[(ty * th + y) * pw + tx * tw..][..tw];
                for &v in row {
                    hist[v as usize] += 1;
                }
            }
            let clipped: u32 = hist.iter().map(|&c| c.saturating_sub(clip_limit)).sum();
            let batch = clipped / HIST as u32;
            let mut residual = clipped - batch * HIST as u32;
            for c in hist.iter_mut() {
                *c = (*c).min(clip_limit) + batch;
            }
            if residual > 0 {
                let step = (HIST as u32 / residual).max(1) as usize;
                let mut i = 0;
                while i < HIST && residual > 0 {
                    hist[i] += 1;
                    residual -= 1;
                    i += step;
                }
            }
            let mut lut = [0u8; HIST];
            let mut sum = 0u32;
            for i in 0..HIST {
                sum += hist[i];
                lut[i] = (sum as f32 * lut_scale).round().clamp(0.0, 255.0) as u8;
            }
            lut
        })
        .collect();

    // bilinear interpolation between neighbouring tile LUTs
    let inv_tw = 1.0 / tw as f32;
    let inv_th = 1.0 / th as f32;
    let xs: Vec<(usize, usize, f32)> = (0..w)
        .map(|x| {
            let f = x as f32 * inv_tw - 0.5;
            let x1 = f.floor() as isize;
            let xa = f - x1 as f32;
            (x1.max(0) as usize, ((x1 + 1) as usize).min(tx_n - 1), xa)
        })
        .collect();
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let f = y as f32 * inv_th - 0.5;
        let y1 = f.floor() as isize;
        let ya = f - y1 as f32;
        let (ty1, ty2) = (y1.max(0) as usize, ((y1 + 1) as usize).min(ty_n - 1));
        for x in 0..w {
            let v = s[y * pw + x] as usize;
            let (tx1, tx2, xa) = xs[x];
            let top = luts[ty1 * tx_n + tx1][v] as f32 * (1.0 - xa) + luts[ty1 * tx_n + tx2][v] as f32 * xa;
            let bot = luts[ty2 * tx_n + tx1][v] as f32 * (1.0 - xa) + luts[ty2 * tx_n + tx2][v] as f32 * xa;
            row[x] = (top * (1.0 - ya) + bot * ya).round().clamp(0.0, 255.0) as u8;
        }
    });
    out
}
