# Samsung S-Series X-ray Panel Controller

[![Build](https://github.com/gamerpaddy/Samsung-S-Xraypanel-Controller/actions/workflows/build.yml/badge.svg)](https://github.com/gamerpaddy/Samsung-S-Xraypanel-Controller/actions/workflows/build.yml)

A desktop program (Rust + [egui](https://github.com/emilk/egui)) for **Samsung S-series flat-panel X-ray detectors**
(developed and tested with an **S4335W**). It talks to the panel over your network, captures raw 16-bit frames,
corrects and displays them, can drive a motorised turntable for CT-style scans, and can pulse an X-ray source in
sync with the panel.

> **Unofficial.** This project is not affiliated with, endorsed by or supported by Samsung. It was written by
> observing and measuring one panel. Use it at your own risk.
>
> **X-ray safety.** The program can trigger an X-ray source through hardware you build yourself (see
> [Turntable and X-ray trigger](#turntable-and-x-ray-trigger-arduino)). X-rays are dangerous. Use proper shielding and interlocks, and never rely on this software as a safety device. Not for Medical purposes.
>
> **Noise2Noise denoising is experimental** and may not work (it can even make images worse). See
> [Noise2Noise (experimental)](#noise2noise-experimental).

<!-- Screenshots: add them to docs/screenshots/ and reference them here. -->
<img width="2080" height="1244" alt="s4335w_ewJ9m791wu" src="https://github.com/user-attachments/assets/157f2010-8643-49bc-997c-dd20eef12168" />

## What it does

**Acquisition**
- Captures frames from the panel over TCP (3072 x 2560, 16-bit raw), with progress and error handling.
- Dark-frame (**offset**) capture and **flat-field** correction (capture, save, load).
- Controls the panel's **integration window** (an undocumented FPGA register) and the delay and length of the
  X-ray pulse, with a **timeline** that shows the real timing of reset, window and read-out.
- A **shot log** (also written to `shots.csv`) with brightness and a top-versus-bottom check for every capture.
- **Turntable** control and **auto scan**: rotate in steps, capture, save as PNG / 16-bit TIFF / raw BIN, retry
  automatically on an under-exposed frame.

**Viewer / processing** (everything runs multi-threaded in Rust, zoom and pan are done on the GPU)
- Window/level (offset, range, clip), automatic 5-95 % range, invert.
- 16 colormaps (grayscale, rainbow, hot, inferno, copper, seismic, thermal, viridis, plasma, magma, turbo,
  cividis, bone, ice, fire, cool).
- CLAHE, sharpening, nearest / bilinear / bicubic / Lanczos resampling.
- Histogram overlay, crop tool, pixel read-out, zoom and pan.
- Export the processed view as PNG, a stretched 16-bit TIFF, or the raw BIN.

**Experimental**
- Noise2Noise denoising, trained on your own frames (CPU, or GPU with the optional CUDA build).

## Requirements

- A **Samsung S-series X-ray detector panel** (developed with the S4335W), **connected to the same network as
  your PC, using Ethernet or Wireless**. The program connects to the panel's TCP ports 20000 (control) and
  20001 (image data).
  - You need the panel's network settings (its IP address and, for wireless, the network credentials).
    They are stored in the panel's own firmware / configuration files that belong to your unit (for example
    `system_setting.sh` and the `wpa_supplicant` configuration), not in this repository. Look there, or in the
    documentation that came with your panel. Telnet is root : xray2012
  - Enter the panel's IP in **Connection** inside the program (the default `192.168.1.100` is only a
    placeholder).
- Optional, for turntable scans and synchronised exposures: an **Arduino** with the sketch in
  [`arduino/xrayturntable`](arduino/xrayturntable) and a USB serial connection.
- Windows or Linux. Tested by the author on Windows only; Linux builds are produced by CI but are untested on
  real hardware.
<img width="2080" height="1244" alt="s4335w_5HucevzjJF" src="https://github.com/user-attachments/assets/abd9bcc0-6e93-48df-aefe-6537efa6a72a" />

## Build and run

You need the stable [Rust toolchain](https://rustup.rs).

```sh
git clone https://github.com/gamerpaddy/Samsung-S-Xraypanel-Controller
cd Samsung-S-Xraypanel-Controller
cargo run --release
```

**Linux** needs a few development packages first (Debian/Ubuntu):

```sh
sudo apt-get install libudev-dev pkg-config libgtk-3-dev libxkbcommon-dev libwayland-dev \
  libx11-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxi-dev libgl1-mesa-dev
```

and your user must be allowed to open the serial port (`sudo usermod -aG dialout $USER`, then log in again).

Prebuilt Windows and Linux binaries are produced by the GitHub Actions workflow (see the *Actions* tab and
*Releases*).

**Optional CUDA build** (GPU training/denoising for the experimental Noise2Noise feature): install the NVIDIA
CUDA toolkit and run `cargo build --release --features cuda`. On Windows the CUDA build also needs the Visual
Studio C++ build tools and `NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor`; the CUDA runtime DLLs must be on
`PATH` when you start the program.

## First start

1. Start the program and open **Connection**: set the panel's IP (and the serial port of the Arduino, if you
   have one).
2. Files are looked up in the working folder, the folder of the executable and a few parent folders (the first
   folder that contains any of them is used):
   | file | meaning |
   |---|---|
   | `default.bin` | a raw frame to show at start-up (optional) |
   | `offsetfield.tif` | dark frame (offset correction), 16-bit TIFF |
   | `flatfield.bin` | flat field, 16-bit raw |
   | `settings.json` | all settings of the program (created on exit) |
3. **Capture offset** with the X-ray source **off**, then **Save flat** with the source **on** and nothing in the
   beam. Capture again whenever you change the window or the exposure timing.
4. Press **Capture**.

## Timing: what the integration window really does

The panel has a register (FPGA `0x11C`, value in 0.1 ms, default 500 ms) that the program exposes as **Window**.
On the unit this was developed with it is **not a shutter that opens at START**. Measured with the real X-ray
source (details in [`docs/PANEL_NOTES.md`](docs/PANEL_NOTES.md)):

- For about the first 450 ms after START the rows are still being reset (bottom first). A pulse here only exposes
  part of the frame, which shows up as a top-to-bottom gradient.
- Then all rows are sensitive for roughly the window length (stock: about 500-900 ms after START).
- After that the rows are read out one after another and light is lost (bottom rows first, finished by about
  1.1 s).
- Source and panel drift against each other by tens of milliseconds from shot to shot. Keep margin on both sides.

A good starting point with the stock window: **pulse 200 ms, trigger delay 565 ms**. Use the timeline
(drag the handles) and the shot log to tune it for your setup. The panel also needs a few seconds between
captures; the program waits for it automatically and never fires the source if the panel did not acknowledge the
start.

<img width="874" height="738" alt="s4335w_eVAxnj3ttb" src="https://github.com/user-attachments/assets/d91373a6-c5ee-4fb3-b31a-0f2362c10d2c" />


## Turntable and X-ray trigger (Arduino)

[`arduino/xrayturntable/xrayturntable.ino`](arduino/xrayturntable/xrayturntable.ino) is the sketch the program
talks to (9600 baud, line-based):

| command | action |
|---|---|
| `CW<deg>` / `CCW<deg>` | rotate the stepper by `<deg>` degrees, answers `OK` when done |
| `C<ms>` | set pin 12 high for `<ms>` milliseconds (X-ray trigger), then low |

Pins: `12` trigger output, `8` STEP, `9` DIR (step/dir driver, via the AccelStepper library). Adjust
`stepsPerDegree` for your gearing. The trigger pin is a logic output: drive your source through a suitable
isolated relay/optocoupler stage and keep your own safety interlocks in the loop.
Install the *AccelStepper* library in the Arduino IDE (Library Manager) before compiling.

## Noise2Noise (experimental)

> **Experimental. It may not work and can make your images worse.** There are no pretrained weights; you have to
> train a model on your own frames, and the result depends heavily on your data. Always keep the raw frame.

The panel's right-hand side has a *Noise2Noise denoise (experimental)* section. It trains a small U-Net on pairs
of noisy exposures of the **same** scene (Noise2Noise), or self-supervised on one frame. After training it
reports the error of the model against the raw frame on held-out patches and warns you if the model is not
better. Clipped (saturated) pixels are excluded from training and exposure differences between frames are
matched. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the credits and licenses.

## Command-line diagnostics

```text
s4335w --reg-read 0x11C            read one FPGA register
s4335w --capture-test [window_ms]  one dark capture through the normal path
s4335w --shoot <pulse_ms> <window_ms|0> <delay,delay,...> [gap,gap,...]
                                   FIRES THE X-RAY SOURCE. Limits: pulse <= 500 ms, <= 12 shots, <= 2 s total.
s4335w --n2n-selftest <image> [steps] [width]
s4335w --n2n-eval <frame.bin> [weights] [width]
```

They use the connection settings from `settings.json` (or the defaults).

## Limitations

- Developed and tested with one Samsung S4335W. Other S-series models may differ.
- The panel protocol is not documented by the vendor; everything here was learned by observation, so behaviour
  can change with other firmware.
- Noise2Noise is experimental (see above).

## License

This project is released into the public domain under [The Unlicense](LICENSE).
Third-party components keep their own licenses, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
