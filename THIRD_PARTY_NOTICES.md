# Third-party notices

This project's own code is released under [The Unlicense](LICENSE) (public domain).
It builds on the work of others, listed here with their licenses. None of the third-party **source code** below is
copied into this repository unless stated; Rust libraries are fetched by Cargo at build time.

## Noise2Noise

The experimental denoiser is based on:

- **Noise2Noise: Learning Image Restoration without Clean Data** - Jaakko Lehtinen, Jacob Munkberg, Jon
  Hasselgren, Samuli Laine, Tero Karras, Miika Aittala, Timo Aila. ICML 2018.
  Paper: <https://arxiv.org/abs/1803.04189>
- **NVlabs/noise2noise** - <https://github.com/NVlabs/noise2noise>, Copyright (c) 2018, NVIDIA CORPORATION.
  Licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

This repository contains **no code, trained weights or data from NVlabs/noise2noise**. `src/n2n.rs` is an
independent re-implementation in Rust (using the [candle](https://github.com/huggingface/candle) library) of the
published network structure (the "autoencoder" U-Net: 7 encoder convolutions, 5 max-pools, nearest-neighbour
up-sampling with skip connections, leaky-ReLU 0.1) with these changes: one input channel, a residual output, a
different weight initialisation, and its own training loop (masked L2 loss, Adam). The NVIDIA repository itself
is **non-commercial**; if you use that code or its assets you are bound by its license. This project's code stays
under the Unlicense, but it is offered as experimental and without any claim that it reproduces the results of
the paper.

The single-image training mode (splitting one frame into two sub-sampled noisy views) is inspired by
**Neighbor2Neighbor** (Huang et al., CVPR 2021, <https://arxiv.org/abs/2101.02824>). No code from it is used.

## Image-processing algorithms

- **CLAHE** (contrast-limited adaptive histogram equalisation, K. Zuiderveld, Graphics Gems IV, 1994). The
  implementation in `src/clahe.rs` follows the documented behaviour of OpenCV's CLAHE (Apache-2.0,
  <https://opencv.org>) so results are comparable; no OpenCV code is included.
- **Colormaps**: viridis, plasma, magma and cividis are piecewise-linear approximations of the matplotlib
  colormaps (CC0 / public domain; viridis, magma, plasma by Nathaniel Smith and Stefan van der Walt, cividis by
  Jamie Nunez et al.). turbo is an approximation of Google's Turbo colormap (Apache-2.0). The other maps are
  simple formulas.

## Rust crates

Direct dependencies (the complete list of all 468 crates is in
[`docs/dependency-licenses.txt`](docs/dependency-licenses.txt)):

| crate | license | purpose |
|---|---|---|
| [eframe / egui](https://github.com/emilk/egui) | MIT OR Apache-2.0 | GUI |
| [rfd](https://github.com/PolyMeilex/rfd) | MIT | file dialogs |
| [rayon](https://github.com/rayon-rs/rayon) | MIT OR Apache-2.0 | multi-threading |
| [serialport](https://github.com/serialport/serialport-rs) | **MPL-2.0** | serial port (turntable / trigger) |
| [image](https://github.com/image-rs/image) | MIT OR Apache-2.0 | PNG / TIFF |
| [candle-core, candle-nn](https://github.com/huggingface/candle) | MIT OR Apache-2.0 | neural-network engine |
| [serde, serde_json](https://serde.rs) | MIT OR Apache-2.0 | settings |
| [chrono](https://github.com/chronotope/chrono) | MIT OR Apache-2.0 | timestamps |
| [anyhow](https://github.com/dtolnay/anyhow) | MIT OR Apache-2.0 | error handling |
| [rand](https://github.com/rust-random/rand) | MIT OR Apache-2.0 | random sampling |

All other crates in the dependency tree are under permissive licenses (MIT, Apache-2.0, BSD, Zlib, Unicode-3.0,
BSL-1.0, ...). A few offer a choice that includes a copyleft option (for example `unescaper`: MIT OR GPL-3.0,
`r-efi`: MIT OR Apache-2.0 OR LGPL-2.1+); the permissive option applies here. `serialport` is under the
file-level copyleft MPL-2.0 and is used **unmodified** as a library.

## Fonts

On Windows the program uses the system font *Segoe UI* if it is present; it is read from the system at run time
and not distributed here. egui bundles its own default fonts (Ubuntu-Light under the Ubuntu Font License, Hack
under MIT, Noto Emoji and emoji-icon-font under the SIL Open Font License), see the egui repository.

## Arduino sketch

`arduino/xrayturntable/xrayturntable.ino` (Unlicense) uses the **AccelStepper** library by Mike McCauley
(<https://www.airspayce.com/mikem/arduino/AccelStepper/>), which is licensed under **GPL-3.0 or a commercial
license**. The library is **not** included here; install it yourself. If you distribute a compiled firmware that
links it, the library's license applies to that binary.

## NVIDIA CUDA (optional)

The optional `cuda` build uses the NVIDIA CUDA toolkit, which is proprietary software under NVIDIA's own license
terms. It is not included or redistributed; you install it yourself.

## Trademarks

"Samsung" and product names such as "S4335W" are trademarks of their respective owners and are used only to
describe compatibility. This project is not affiliated with or endorsed by Samsung.
