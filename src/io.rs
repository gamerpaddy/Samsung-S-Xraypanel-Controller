//! File I/O: raw .bin, 16-bit TIFF, PNG, offset/flat field loading.
use crate::pipeline::{Field, Flat, Img16, CROP_H, CROP_W, FULL_H, FULL_W};
use anyhow::{bail, Context, Result};
use image::{codecs::png::{CompressionType, FilterType, PngEncoder}, DynamicImage, ExtendedColorType, ImageEncoder};
use rayon::prelude::*;
use std::path::Path;

pub fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes.par_chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

pub fn read_bin(path: &Path) -> Result<Vec<u16>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(bytes_to_u16(&bytes))
}

pub fn write_bin(path: &Path, px: &[u16]) -> Result<()> {
    let mut out = Vec::with_capacity(px.len() * 2);
    for v in px {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Decode any image to (w, h, u16 samples) without rescaling 16-bit data.
pub fn load_gray16(path: &Path) -> Result<(usize, usize, Vec<u16>)> {
    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(match img {
        DynamicImage::ImageLuma16(b) => (b.width() as usize, b.height() as usize, b.into_raw()),
        DynamicImage::ImageLuma8(b) => (
            b.width() as usize,
            b.height() as usize,
            b.into_raw().into_iter().map(|v| v as u16).collect(),
        ),
        other => {
            let b = other.into_luma16();
            (b.width() as usize, b.height() as usize, b.into_raw())
        }
    })
}

pub fn save_tif16(path: &Path, w: usize, h: usize, px: &[u16], crop: Option<[usize; 4]>) -> Result<()> {
    let (w2, h2, data) = match crop {
        Some(c) => crop_slice(px, w, h, c),
        None => (w, h, px.to_vec()),
    };
    let buf = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(w2 as u32, h2 as u32, data)
        .context("bad buffer size")?;
    buf.save_with_format(path, image::ImageFormat::Tiff)?;
    Ok(())
}

pub fn save_png(path: &Path, w: usize, h: usize, px: &[u8], channels: usize, crop: Option<[usize; 4]>) -> Result<()> {
    let (w2, h2, data) = match crop {
        Some(c) => crop_slice_n(px, w, h, channels, c),
        None => (w, h, px.to_vec()),
    };
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let enc = PngEncoder::new_with_quality(file, CompressionType::Fast, FilterType::Adaptive);
    let ct = if channels == 1 { ExtendedColorType::L8 } else { ExtendedColorType::Rgb8 };
    enc.write_image(&data, w2 as u32, h2 as u32, ct)?;
    Ok(())
}

/// crop = [x1, y1, x2, y2] in pixels (already clamped by the caller)
fn crop_slice(px: &[u16], w: usize, h: usize, c: [usize; 4]) -> (usize, usize, Vec<u16>) {
    let (x1, y1, x2, y2) = (c[0].min(w), c[1].min(h), c[2].min(w), c[3].min(h));
    let mut out = Vec::with_capacity((x2 - x1) * (y2 - y1));
    for y in y1..y2 {
        out.extend_from_slice(&px[y * w + x1..y * w + x2]);
    }
    (x2 - x1, y2 - y1, out)
}

fn crop_slice_n(px: &[u8], w: usize, h: usize, n: usize, c: [usize; 4]) -> (usize, usize, Vec<u8>) {
    let (x1, y1, x2, y2) = (c[0].min(w), c[1].min(h), c[2].min(w), c[3].min(h));
    let mut out = Vec::with_capacity((x2 - x1) * (y2 - y1) * n);
    for y in y1..y2 {
        out.extend_from_slice(&px[(y * w + x1) * n..(y * w + x2) * n]);
    }
    (x2 - x1, y2 - y1, out)
}

/// Offset field (dark frame) from a TIFF. Handles the transposed layout and the padded full size.
pub fn load_offset(path: &Path) -> Result<Field> {
    let (mut w, mut h, mut px) = load_gray16(path)?;
    if h == FULL_W && w == FULL_H {
        let mut t = vec![0u16; px.len()];
        for y in 0..h {
            for x in 0..w {
                t[x * h + y] = px[y * w + x];
            }
        }
        px = t;
        std::mem::swap(&mut w, &mut h);
    }
    if h == FULL_H && w == FULL_W {
        // the dark frame is stored in sensor orientation; the image is rotated 180 deg, so rotate
        // the offset the same way before cropping (otherwise the column pattern is misaligned)
        px.reverse();
        let (ow, oh, d) = crop_slice(&px, w, h, [17, 92, w - 14, h - 2]);
        return Ok(Field { w: ow, h: oh, px: d });
    }
    Ok(Field { w, h, px })
}

/// Same as `load_offset` but from the raw uint16 stream the panel delivers.
pub fn offset_from_raw(raw: &[u16]) -> Field {
    let w = FULL_W;
    let h = raw.len() / w;
    let px = &raw[..w * h];
    if h == FULL_H {
        let rot: Vec<u16> = px.iter().rev().copied().collect();
        let (ow, oh, d) = crop_slice(&rot, w, h, [17, 92, w - 14, h - 2]);
        Field { w: ow, h: oh, px: d }
    } else {
        Field { w, h, px: px.to_vec() }
    }
}

/// Flat field from a .bin file (rotated layout when full-size, see the Python viewer).
pub fn load_flat(path: &Path, width_hint: usize) -> Result<Flat> {
    let data = read_bin(path)?;
    let total = data.len();
    if total == FULL_W * FULL_H {
        let (w, h, d) = crop_slice(&data, FULL_W, FULL_H, [14, 2, FULL_W - 17, FULL_H - 92]);
        return Ok(Flat::new(w, h, d));
    }
    if total == CROP_W * CROP_H {
        return Ok(Flat::new(CROP_W, CROP_H, data));
    }
    let w = if width_hint > 0 { width_hint } else { FULL_W };
    let h = total / w;
    if h == 0 {
        bail!("Invalid flat field file size");
    }
    let px = &data[..w * h];
    if h == FULL_H && w == FULL_W {
        let (w2, h2, d) = crop_slice(px, w, h, [14, 2, w - 17, h - 92]);
        return Ok(Flat::new(w2, h2, d));
    }
    Ok(Flat::new(w, h, px.to_vec()))
}

pub fn from_img16(img: &Img16) -> (usize, usize, &[u16]) {
    (img.w, img.h, &img.px)
}
