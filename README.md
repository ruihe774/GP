# gpagent — Component A: gameplay capture & discretization

Captures what a player is doing — gamepad, microphone, screen — and turns it into a
sparse stream of typed events cheap enough to feed a realtime multimodal model.

This is **Component A** of a "friend on the couch" commentary agent. Component B
(the `gpt-realtime-2.1` session and voice playback) is not built yet; the
interfaces here are shaped to feed it directly.

## Why discretization

Streaming raw microphone and video into the Realtime API would cost more than the
game. Everything here exists to emit only the moments that carry information.

| Modality | Rate | Continuous cost | What we do instead |
|---|---|---|---|
| Audio in | $32/1M tok, 1 tok / 100 ms → 600 tok/min | $0.019/min open mic | VAD-gated speech segments only |
| Image in | $5/1M tok, ~765 tok per 1024x576 frame | $0.033/min at 6 frames/min | triggered + deduplicated frames |
| Text in | $4/1M tok | negligible | one short summary per 500 ms window |

Note the ordering: **screen frames cost more than the microphone**, so the frame
trigger policy is the more valuable knob. `detail: "low"` (a flat 85 tokens per
image) is a further large lever available to Component B.

## Requirements

- Linux with PipeWire and a `xdg-desktop-portal` implementation (developed on
  Ubuntu 26.04, GNOME/Wayland, PipeWire 1.6, GStreamer 1.28)
- System packages: `gstreamer1.0-pipewire`, `gstreamer1.0-plugins-good`,
  `gstreamer1.0-plugins-bad`, `python3-gi`, `gir1.2-gst-plugins-base-1.0`
- Python 3.12+

**Run it from the desktop session, not over SSH.** The gamepad `uaccess` ACL and
the ScreenCast portal both bind to the active seat.

## Setup

```bash
uv venv --python 3.14 --system-site-packages   # GStreamer bindings come from the distro
uv pip install -e ".[dev]"
curl -L -o models/silero_vad.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
```

## Usage

```bash
gpagent devices                    # what was detected, and whether it is usable
gpagent record -o sess -d 60       # capture a minute
gpagent inspect sess               # timeline, stats, estimated cost
gpagent inspect sess --wav --contact-sheet   # listen to it, look at it
gpagent replay sess                # re-emit with original timing
```

First `record` raises a portal permission dialog. Approve it once — the
`restore_token` is cached in `~/.config/gpagent/screencast.json` and later runs
are silent.

Tuning without editing files:

```bash
gpagent record --set gamepad.sticks_mode=full --set screen.long_edge=1280
gpagent record --set audio.vad_backend=webrtc      # drop the ONNX model
gpagent record --no-screen                          # sources are independently disableable
```

## What it emits

`events.jsonl` plus a `blobs/` directory (or `--inline` for one self-contained
file). Every event carries `t` (monotonic seconds since session start), `seq`,
and `type`.

```json
{"t":12.48,"seq":41,"type":"gamepad.activity","device":"2dc8:200f:usb-…","window_ms":500,
 "buttons":{"A":{"taps":3,"held_ms":0}},"triggers":{"LT":"idle","RT":"full"},
 "dpad":"idle","intensity":0.72,"apm":84,"summary":"held RT, tapped A x3"}
{"t":30.10,"seq":88,"type":"speech.segment","blob":"blobs/000088-speech.pcm",
 "dur_ms":1840,"sample_rate":24000,"encoding":"pcm_s16le","rms_dbfs":-22.4}
{"t":12.50,"seq":42,"type":"screen.frame","blob":"blobs/000042-frame.jpg",
 "w":1024,"h":576,"trigger":"gamepad","scene_score":0.11}
```

The `summary` string is the thing Component B actually spends tokens on; the
structured fields are there for policy and debugging.

## Design notes

**Everything is discovered, nothing is named.** Gamepads are found by
ioctl capability probe across all `/dev/input/event*` using udev's own heuristic
(`ABS_X`+`ABS_Y` plus a `BTN_JOYSTICK`/`BTN_GAMEPAD`-range key, minus pointers
and touchpads), so a pad nobody has tested works. Audio leaves the PipeWire
target unset so WirePlumber routes it — and re-routes it when you switch
headsets. Screen geometry comes from whatever the portal negotiates.

**Sticks are configurable and quiet by default.** `sticks_mode=intensity` (the
default) lets stick displacement feed the scalar `intensity` that drives frame
triggers, but never describes direction in the summary — it is mostly "the
player is moving around" and it dominates event volume. `full` describes them;
`off` ignores them. A window containing only stick motion emits no event at all.

**The AEC reference is the whole system output**, captured from the default
sink's monitor (`stream.capture.sink=true`), not just the agent's voice. The
game is coming out of the speakers into the microphone too; on a combined
speaker/mic device that echo would otherwise be detected as speech and billed.

**Frame triggers throttle, they do not debounce.** Gameplay activity is
near-continuous, so a debounce would fire late or starve frames. The first frame
of a burst is the informative one.

**VAD is swappable.** `silero` (default) is an ONNX model that rejects noise,
tones and bass rumble at ≤0.003 against a 0.5 threshold. `webrtc` uses
`webrtcdsp`'s built-in detector at zero extra cost — note it is *edge-reported*
(the level meta is attached only when voice activity flips, see
`gstwebrtcdsp.cpp:516`), so the source holds the level between transitions.

## Tests

```bash
pytest              # 90 tests, no hardware, ~0.1 s
pytest -m hardware  # 5 more, against real devices
```

The suite is deliberately hardware-free: synthetic capability bitmaps for device
classification, synthetic `input_event` structs for the discretizer, scripted
probabilities for the segmenter, and a virtual clock for the trigger policy.

## Verification checklist

Automated tests do not cover acoustics or the feel of the discretizer. Run this
by hand:

1. `gpagent devices` — pad detected, defaults resolved, no sudo needed.
2. `gpagent record -o /tmp/s1 -d 60`, then play: press buttons, hold a trigger,
   say something, sit idle.
3. `gpagent inspect /tmp/s1` — summaries match what you did; speech segments only
   where you spoke; frames attributed to sensible triggers.
4. `--wav` — your voice, with no clipped leading syllable (pre-roll working).
5. **AEC check:** play loud game audio while staying silent. This should produce
   *zero* speech segments. Then talk over it: exactly one clean segment.
6. Re-run `record` — no portal dialog (restore token working).
7. Unplug and replug the pad mid-session — disconnect/connect events, capture
   resumes.

## Known gotchas

- The portal tears down the ScreenCast session, and with it the PipeWire node,
  as soon as the client's D-Bus connection is collected. `ScreenCastSession`
  holds that connection deliberately; dropping it manifests as `pipewiresrc`
  failing with `target not found`.
- `webrtcdsp` refuses to start when `echo-cancel` is on and no probe element
  exists; `AudioSource` retries once without AEC rather than failing.
- Pinning only one dimension in the scale caps while forcing
  `pixel-aspect-ratio=1/1` makes `videoscale` fixate the other to 1. Both
  dimensions are computed explicitly from the source size.
- `Gst.DeviceMonitor` reports each device 2–3× across the pipewire/alsa/pulse
  providers, and a sink shares `node.name` with its own monitor — dedup on
  `(media.class, node.name)`.
