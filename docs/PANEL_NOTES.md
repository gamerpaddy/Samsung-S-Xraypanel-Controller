# Panel notes (measured behaviour)

These notes describe what this program relies on and what was measured on **one Samsung S4335W**. The protocol is
not documented by the vendor, so treat everything as observations that may differ on other units or firmware.
No vendor firmware or source code is included in this repository.

## Network

| port | use |
|---|---|
| TCP 20000 | control messages |
| TCP 20001 | image data (raw 16-bit little-endian, 3072 x 2560 = 15,728,640 bytes) |

Control messages are 132 bytes: a 4-byte little-endian command followed by a 128-byte payload (short messages
without padding are accepted for MODE and START).

Capture sequence used by the program:

1. Connect to both ports.
2. Send `MODE` = `01 A0 C1 00`; the panel answers `01 A1 C1 00 ...`.
3. Send `START` = `02 00 00 00`; the panel answers within a few ms, including a message that starts with
   `12 00 00 00`. **The program never fires the X-ray source unless this reply arrived.**
4. After the trigger delay, send `C<ms>\n` to the Arduino (this is the X-ray pulse).
5. Read the 15,728,640 bytes from port 20001 (about 8 s).

The panel needs a few seconds after a finished capture before it answers MODE/START again; starting sooner gives
no reply or a frame in which the exposure is missed. The program waits 6 s after a good capture and 12 s after a
failed one.

## Frame geometry

The raw frame is 3072 x 2560. The program rotates it by 180 degrees and crops the dead border
(rows 92..2558, columns 17..3058), giving 3041 x 2466. A dark frame (offset) is stored in raw sensor orientation
and therefore has to be rotated with the image before it is subtracted. Saturated pixels (16383) are left
untouched by the offset and flat correction.

## Raw FPGA register access (diagnostic)

Two undocumented commands read and write FPGA registers:

| command | meaning | message |
|---|---|---|
| `0xC1D00000` | read register | `[cmd:4][address:4][0:4]` + padding, value in the reply at byte offset 8 |
| `0xC1D00001` | write register | `[cmd:4][address:4][value:4]` + padding |

The program uses only **register `0x11C`** ("window time", unit 0.1 ms, default 5000 = 500 ms). Writing other
registers can upset the panel; do not experiment on a panel you cannot afford to lose. The value is volatile and
is lost on a panel reboot.

After writing the register the panel returns an all-zero frame for the next capture. The program runs a
throwaway dark capture after a write (and a settle wait) before the first real exposure. Likewise, a throwaway
dark capture is run before the first exposure after the panel has been idle for a minute or more; without it the
lower part of that first frame was dark.

## Timing (stock window 500 ms, X-ray source with a 50 Hz supply)

Times are measured from the moment START is sent. A 30 ms pulse was fired at delays from 0 to 1100 ms and the
brightness of eight horizontal bands of the frame was compared with the dark level:

| pulse at | what registered |
|---|---|
| 0 to 300 ms | only part of the rows (top bands missing) or nothing: the rows are still being reset |
| 500 to 800 ms | all bands |
| 900 ms | all bands, bottom band weaker |
| 1000 ms | top five bands only; the rows are being read out, bottom first |
| 1100 ms | nothing |

So the integration is effectively: reset sweep (about 0 to 450 ms), all rows sensitive (about 450 ms plus the
window length), read-out sweep (afterwards). The timeline in the program draws this. Pulse timing between source and panel
varies by tens of milliseconds from shot to shot (the source here has a 50 Hz supply and a slow start), so keep
margin on both sides of the pulse.

With the stock window, a 200 ms pulse started with a trigger delay of 565 ms gave repeatable frames (frame means
within 0.2 % in three consecutive shots).
