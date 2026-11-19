//! Colormaps (same formulas as the Python viewer), baked into 256-entry LUTs.
use rayon::prelude::*;

pub const NAMES: &[&str] = &[
    "grayscale", "rainbow", "hot", "inferno", "copper", "seismic", "thermal", "viridis", "plasma", "magma", "turbo",
    "cividis", "bone", "ice", "fire", "cool",
];

/// Gradient stops for the table-driven palettes (linearly interpolated).
fn stops(name: &str) -> Option<&'static [[u8; 3]]> {
    Some(match name {
        "viridis" => &[[68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141], [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37]],
        "plasma" => &[[13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150], [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40], [240, 249, 33]],
        "magma" => &[[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]],
        "turbo" => &[[48, 18, 59], [70, 107, 227], [40, 187, 236], [49, 242, 153], [162, 252, 60], [237, 208, 58], [251, 128, 34], [210, 49, 5], [122, 4, 3]],
        "cividis" => &[[0, 34, 78], [53, 68, 108], [97, 100, 112], [138, 134, 119], [184, 171, 108], [233, 212, 75], [254, 232, 56]],
        "bone" => &[[0, 0, 0], [46, 46, 64], [92, 101, 126], [139, 155, 155], [190, 208, 208], [255, 255, 255]],
        "ice" => &[[0, 0, 0], [10, 30, 80], [20, 90, 160], [80, 170, 220], [200, 235, 250], [255, 255, 255]],
        "fire" => &[[0, 0, 0], [32, 0, 96], [130, 0, 130], [210, 40, 60], [250, 130, 0], [255, 210, 60], [255, 255, 255]],
        "cool" => &[[0, 255, 255], [255, 0, 255]],
        _ => return None,
    })
}

fn from_stops(st: &[[u8; 3]], i: usize) -> [u8; 3] {
    let pos = i as f32 / 255.0 * (st.len() - 1) as f32;
    let k = (pos.floor() as usize).min(st.len() - 2);
    let t = pos - k as f32;
    let mut o = [0u8; 3];
    for c in 0..3 {
        o[c] = (st[k][c] as f32 * (1.0 - t) + st[k + 1][c] as f32 * t).round() as u8;
    }
    o
}

pub type Lut = [[u8; 3]; 256];

fn u(x: f64) -> u8 {
    x.clamp(0.0, 255.0) as u8
}

fn color(name: &str, i: usize) -> [u8; 3] {
    if let Some(st) = stops(name) {
        return from_stops(st, i);
    }
    let v = i as f64 / 255.0;
    match name {
        "rainbow" => {
            if v < 0.125 {
                [0, 0, u(128.0 + v * 1024.0)]
            } else if v < 0.375 {
                let t = (v - 0.125) / 0.25;
                [0, u(t * 255.0), 255]
            } else if v < 0.625 {
                let t = (v - 0.375) / 0.25;
                [u(t * 255.0), 255, u((1.0 - t) * 255.0)]
            } else if v < 0.875 {
                let t = (v - 0.625) / 0.25;
                [255, u((1.0 - t) * 255.0), 0]
            } else {
                let t = (v - 0.875) / 0.125;
                [u(255.0 - t * 128.0), 0, 0]
            }
        }
        "hot" => [
            u((v * 3.0).min(1.0) * 255.0),
            u(((v - 0.333) * 3.0).clamp(0.0, 1.0) * 255.0),
            u(((v - 0.666) * 3.0).clamp(0.0, 1.0) * 255.0),
        ],
        "inferno" => [
            u((v * 1.5).min(1.0) * 255.0),
            u(((v - 0.4) * 2.0).clamp(0.0, 1.0) * 255.0),
            u(((0.5 - v) * 2.0).clamp(0.0, 1.0) * 255.0),
        ],
        "copper" => [u((v * 1.25).min(1.0) * 255.0), u(v * 0.78 * 255.0), u(v * 0.5 * 255.0)],
        "seismic" => {
            if v < 0.5 {
                let t = v / 0.5;
                [u(t * 255.0), u(t * 255.0), 255]
            } else {
                let t = (v - 0.5) / 0.5;
                [255, u((1.0 - t) * 255.0), u((1.0 - t) * 255.0)]
            }
        }
        "thermal" => {
            if v < 0.2 {
                let t = v / 0.2;
                [u(t * 128.0), 0, u(t * 255.0)]
            } else if v < 0.4 {
                let t = (v - 0.2) / 0.2;
                [u(128.0 + t * 127.0), 0, u((1.0 - t) * 255.0)]
            } else if v < 0.6 {
                let t = (v - 0.4) / 0.2;
                [255, u(t * 165.0), 0]
            } else if v < 0.8 {
                let t = (v - 0.6) / 0.2;
                [255, u(165.0 + t * 90.0), 0]
            } else {
                let t = (v - 0.8) / 0.2;
                [255, 255, u(t * 255.0)]
            }
        }
        _ => [i as u8; 3],
    }
}

/// `None` for plain grayscale.
pub fn lut(name: &str) -> Option<Lut> {
    if name == "grayscale" || !NAMES.contains(&name) {
        return None;
    }
    let mut l = [[0u8; 3]; 256];
    for (i, c) in l.iter_mut().enumerate() {
        *c = color(name, i);
    }
    Some(l)
}

pub fn apply(gray: &[u8], lut: &Lut) -> Vec<u8> {
    let mut out = vec![0u8; gray.len() * 3];
    out.par_chunks_mut(3 * 4096)
        .zip(gray.par_chunks(4096))
        .for_each(|(o, g)| {
            for (px, &v) in o.chunks_exact_mut(3).zip(g) {
                px.copy_from_slice(&lut[v as usize]);
            }
        });
    out
}
