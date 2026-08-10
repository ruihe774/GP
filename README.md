# gpagent — a "friend on the couch" commentary agent

Watches someone play a video game and comments on it like a friend sitting next
to them. Two halves:

- **Component A — capture.** Gamepad, microphone and screen, turned into a sparse
  stream of typed events cheap enough to feed a realtime multimodal model.
- **Component B — the agent.** Consumes that stream, decides *when* there is
  something worth saying, says it with `gpt-realtime-2.1`, and plays it aloud.
  See [Component B](#component-b--the-commentary-agent).

```bash
gpagent record -o sess -d 60        # capture
gpagent inspect sess                # look at what was captured
gpagent commentate --replay sess    # let the agent talk about it
gpagent commentate                  # ...or do both live
```

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

These are the numbers for *capture*. Once Component B is in front of them the
ordering changes — it sends at most one frame per response, and what it says
costs more than anything it is sent. See [Cost, measured](#cost-measured) for
figures reconciled against real API usage.

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
gpagent monitor                    # live per-button view, to check the mapping
gpagent record -o sess -d 60       # capture a minute
gpagent inspect sess               # timeline, stats, estimated cost
gpagent inspect sess --wav --contact-sheet   # listen to it, look at it
gpagent replay sess                # re-emit with original timing
gpagent commentate --replay sess   # run the agent over it (Component B)
```

First `record` raises a portal permission dialog. Approve it once — the
`restore_token` is cached in `~/.config/gpagent/screencast.json` and later runs
are silent.

To capture something else — a different monitor, or one window instead of the
whole screen — force the picker back up:

```bash
gpagent record --pick-screen        # ignore the cached choice and ask again
```

Whatever you pick is saved and reused by subsequent runs, so this is only needed
when the target changes. Note the dialog blocks *only* the screen source; the
gamepad and microphone start recording immediately, so answer it promptly or the
first stretch of the session will have no frames.

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
 "dpad":"idle","holding":["RT"],"intensity":0.72,"apm":84,
 "summary":"holding RT, tapped A x3"}
{"t":30.10,"seq":88,"type":"speech.segment","blob":"blobs/000088-speech.pcm",
 "dur_ms":1840,"sample_rate":24000,"encoding":"pcm_s16le","rms_dbfs":-22.4}
{"t":12.50,"seq":42,"type":"screen.frame","blob":"blobs/000042-frame.jpg",
 "w":1024,"h":576,"trigger":"gamepad","scene_score":0.11}
```

The `summary` string is the thing Component B actually spends tokens on; the
structured fields are there for policy and debugging.

## Design notes

**Summaries are edge-triggered.** A hold is announced once when it starts and
once when it ends; the windows between say nothing and are suppressed entirely
if nothing else happened. Repeating `held RT (0.5s)` twice a second for a trigger
the player is leaning on is pure cost and reads as noise:

```
   0.5s  holding RT                                     holding: RT
   1.0s  tapped A                                       holding: RT
   1.5s  tapped A                                       holding: RT
   2.0s  holding LB                                     holding: LB,RT
   2.5s  (suppressed - nothing new)
   3.0s  released RT after 2.5s, released LB after 1.2s
   3.5s  -> gamepad.idle
```

The `holding` field carries current state on any event that *is* emitted, so a
reader always knows what is still down without the summary repeating it. A press
that starts and ends inside one window reads `held A for 0.3s` instead, since it
was never announced as ongoing. `intensity` deliberately stays level-based: it
drives screen-frame triggers, so a held button must keep counting even while the
summary is quiet.

**Everything is discovered, nothing is named.** Gamepads are found by
ioctl capability probe across all `/dev/input/event*` using udev's own heuristic
(`ABS_X`+`ABS_Y` plus a `BTN_JOYSTICK`/`BTN_GAMEPAD`-range key, minus pointers
and touchpads), so a pad nobody has tested works. Audio leaves the PipeWire
target unset so WirePlumber routes it — and re-routes it when you switch
headsets. Screen geometry comes from whatever the portal negotiates.

**Sticks are configurable and silent by default.** `sticks_mode=off` (the
default) ignores them entirely, so simply moving around never drives a screen
capture — which also makes a drifting stick harmless. `intensity` lets stick
displacement feed the activity scalar without ever describing direction, since
that is mostly "the player is moving around" and it dominates event volume.
`full` describes direction and magnitude. Under `intensity`, a window containing
only stick motion still emits no event at all.

**The AEC reference is the whole system output**, captured from the default
sink's monitor (`stream.capture.sink=true`), not just the agent's voice. The
game is coming out of the speakers into the microphone too; on a combined
speaker/mic device that echo would otherwise be detected as speech and billed.

**Frame triggers throttle, they do not debounce.** Gameplay activity is
near-continuous, so a debounce would fire late or starve frames. The first frame
of a burst is the informative one.

**`triggers.min_interval_s` is a freshness knob, not a cost knob.** It was 5.0
when the assumption was that every captured frame gets sent. Component B sends at
most one frame per response, and only if it is newer than the last one sent, so
capturing more frames costs local JPEG encoding and nothing on the bill. What the
floor actually controls is how stale the screenshot is at the moment the agent
comments — at 5.0 that was 4.2 s on average and 8.0 s at worst, measured on
sess5. It is now 2.0. Simulated against sess5's real signal timeline:

| `min_interval_s` | frames/min | mean age at send | worst |
|---|---|---|---|
| 5.0 | 7.1 | 4.17 s | 8.00 s |
| 3.0 | 9.1 | 2.70 s | 4.37 s |
| **2.0** | **9.5** | **2.53 s** | 5.50 s |
| 1.0 | 11.1 | 2.20 s | 4.37 s |

Below about 3 s the floor stops being the binding constraint — the per-trigger
throttles take over — so going lower buys tenths of a second and encodes more.
Two things that look like improvements and are not: lowering `heartbeat_s`
*raised* the mean age (2.49 s → 2.74 s), because heartbeat frames compete for the
same global floor as speech-triggered ones and crowd out the frames that land
near a response; and relaxing `gamepad_throttle_s`/`scene_throttle_s` changed
nothing at all, because on this recording neither trigger ever fires — every
frame comes from `speech` or `heartbeat`.

`gpagent commentate` reports the measured frame age at send time, so this is
checkable on any session rather than inferred.

**VAD is swappable.** `silero` (default) is an ONNX model that rejects noise,
tones and bass rumble at ≤0.003 against a 0.5 threshold. `webrtc` uses
`webrtcdsp`'s built-in detector at zero extra cost — note it is *edge-reported*
(the level meta is attached only when voice activity flips, see
`gstwebrtcdsp.cpp:516`), so the source holds the level between transitions.

## Tests

```bash
pytest              # 250 tests, no hardware, no network, ~1 s
pytest -m hardware  # 8 more, against real devices
pytest -m network   # 2 more, against the real API (spends money)
```

The suite is deliberately hardware-free: synthetic capability bitmaps for device
classification, synthetic `input_event` structs for the discretizer, scripted
probabilities for the segmenter, and a virtual clock for the trigger policy and
the speaking policy.

## Verifying Component A

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

## Checking the button mapping

`gpagent monitor` prints the resolved layout, then one line per input as you
press things — using the same `resolve_layout` the discretizer uses, so what it
shows is what events will say:

```
    BTN_SOUTH                0x0130  ->  A
    BTN_NORTH                0x0133  ->  Y     <- note: NORTH is Y, WEST is X
    ABS_Z                    0x0002  ->  LT (analog)

      3.18  press     LB       BTN_TL 0x0136
      3.35  release   LB       held 176 ms -> tap
```

Each line carries the kernel code next to the label, so a mis-mapped pad is
obvious. On exit it lists any buttons you did not press, and anything the layout
does not know about is flagged `UNMAPPED`. `--raw` additionally dumps every
evdev event.

### Fixing a wrong mapping

Labels follow the Linux gamepad spec, but not every pad does. If a button comes
out wrong, relabel it by kernel code — the code `monitor` prints is exactly what
you write:

```bash
gpagent monitor --map BTN_NORTH=X --map BTN_WEST=Y   # try it
gpagent record --map BTN_NORTH=X --map BTN_WEST=Y    # use it
```

Make it permanent in a config file, either for every pad:

```toml
[gamepad.button_map]
BTN_NORTH = "X"
BTN_WEST = "Y"
```

or for one device only, keyed `vid:pid` in lowercase hex (`gpagent devices`
prints it), which is the better choice if you swap between controllers:

```toml
[gamepad.device_button_map."2dc8:200f"]
BTN_NORTH = "X"
BTN_WEST = "Y"
```

Precedence runs spec → built-in vendor quirks → `button_map` →
`device_button_map`. `devices` and `monitor` mark relabelled entries
`(remapped)`, an unknown button name is rejected before any device is opened,
and remapping one half of a swap warns that two buttons now share a label.

This is also the quickest way to spot **stick drift**: a worn stick resting near
the deadzone shows up as repeated `stick` lines with no hands on the controller.
Drift is one reason `sticks_mode` defaults to `off` — otherwise it would feed the
activity scalar and quietly pull screen captures while the player sits still.

## Listening to captured speech

Speech blobs are **headerless PCM** — signed 16-bit little-endian, mono,
24 kHz — so a player needs to be told the format. The built-in way:

```bash
gpagent inspect sess3 --wav        # writes sess3/wav/*.wav
```

Or with ffmpeg directly:

```bash
ffplay -f s16le -ar 24000 -ac 1 sess3/blobs/000088-speech.pcm   # play in place
ffmpeg -f s16le -ar 24000 -ac 1 -i in.pcm out.wav               # convert one
```

Convert a whole session (fish):

```fish
for f in sess3/blobs/*.pcm
    ffmpeg -y -f s16le -ar 24000 -ac 1 -i $f (string replace .pcm .wav $f)
end
```

Concatenate everything into one file to review a session quickly (bash/fish):

```bash
cat sess3/blobs/*-speech.pcm | ffmpeg -y -f s16le -ar 24000 -ac 1 -i - all.wav
```

## Diagnosing a session that captured nothing

`record` prints a `note:` line when a source produced nothing and says why, and
`manifest.json` carries the same detail under `devices`:

- `audio.input_peak_dbfs` — how loud the microphone actually got. Below about
  −60 dBFS means the default source is not a live microphone.
- `audio.vad_max_probability` vs `vad_threshold` — whether the VAD came close.
  A live mic plus a max probability near 0 means the audio reaching the VAD did
  not sound like speech; suspect the echo canceller.
- `screen.portal_handshake_s` — an unanswered consent dialog blocks *only* the
  screen source, so the other two keep recording and the result looks like
  broken screen capture rather than a pending dialog.
- `screen.stall_events` — the compositor stopped delivering frames.
- `screen.frames_seen` vs `frames_emitted` vs `frames_deduped` — separates "no
  frames arrived" from "frames arrived and the policy declined them".

If speech goes missing while game audio is playing, the first thing to try is
`--set audio.echo_cancel=false`. If speech then appears, the canceller was
eating it and `echo_suppression_level` is the knob.

---

# Component B — the commentary agent

Holds a `gpt-realtime-2.1` session, decides when to speak, and speaks.

```bash
gpagent commentate                                     # live capture, live API
gpagent commentate --replay sess5                      # a recording, live API
gpagent commentate --replay sess5 --speed 0 --dry-run  # deterministic, no network
```

The API key comes from `OPENAI_API_KEY` or from `.env` at the repo root (which is
gitignored). It is never logged; `load_api_key` returns a `Secret` whose `repr`
is blanked, because an ordinary `str` leaks through any traceback that happens to
have it in scope — pytest renders the arguments of every frame it prints, which
is how it first got out.

## The actual problem: when to talk

The API integration is the easy half. The agent has no user turn to respond to
most of the time, so restraint has to come from somewhere, and the choice made
here is that it comes from the **policy, not the prompt**: the model is only ever
asked for a response at a moment already judged worth speaking at. Asking it to
also decide whether to speak would pay twice for one decision and get a worse
answer.

`agent/policy.py::SpeakPolicy` mirrors `capture/triggers.py::TriggerPolicy` —
pure logic, injectable clock, inputs pushed in, `decide()` names a reason or
returns None. Three reasons:

| reason | when | bound by |
|---|---|---|
| `reply` | the player said something | the global cap only |
| `react` | scene change, or a burst of input | quiet floor + `event_cooldown_s` |
| `ambient` | it has been quiet a while | quiet floor + `ambient_after_s` |

and two rules over them:

- **A global cap** (`max_per_min`, sliding 60 s window) binding *every* reason.
  This is Component A's frame-trigger lesson carried over: three rules each
  obeying only their own cooldown take turns and produce a combined rate far
  above what any of them allows. `tests/test_speak_policy.py` asserts the
  combined rate with all three reasons pending at 4 Hz for ten minutes.
- **A quiet floor** (`min_gap_s`) after the agent stops talking, binding only the
  two unprompted reasons. `reply` is exempt: if the player asks a question two
  seconds after the agent finishes a sentence, answering is correct and refusing
  is broken. Nothing is exempt from the cap.

**Cooldowns adapt.** Every unprompted remark the player doesn't answer multiplies
them by `backoff_factor`; talking to the agent multiplies by `engagement_boost`
(<1) and resets the streak. An agent being ignored gets quieter on its own.

**A wanted reply expires** (`reply_ttl_s`). If the agent was busy or capped when
the player spoke, answering nine seconds later is worse than not answering — the
moment has gone and the audio is stale. Conversely, utterances that arrive while
the agent is mid-sentence are *accumulated*, so three sentences of thinking out
loud get answered as one thought rather than as their tail.

**Both hard gates expire.** "The player is talking" and "a response is in flight"
are absolute — while set, nothing can be said. A gate that sticks is a
permanently mute agent, which is the quietest possible bug to notice, so a
`speech.start` whose segment never arrives (VAD false start) and a
`response.create` that never completes both time out.

Tests assert positive behaviour first. Component A's most expensive bug was a VAD
that returned zero for everything and passed every test, because every test
asserted a negative. A policy that never speaks passes "never talks over the
player", "respects the cooldown" and "obeys the rate cap" perfectly.

## What gets sent

Gamepad summaries are **accumulated and flushed at speak time**, not streamed.
91 windows in 151 s would otherwise become 91 conversation turns, and be billed
whether or not the agent ever had a reason to speak. Same rule for frames, which
matters more: **at most one image per response, and only if it is newer than the
last one sent**. Capture rate and send rate are decoupled — sess5 triggers 28
frames, the agent pays for 14.

```
conversation.item.create   summaries + at most one screenshot
input_audio_buffer.append  the player's utterance, PCM16 24 kHz mono, unconverted
input_audio_buffer.commit
response.create            persona + a per-reason instruction
```

`turn_detection` is `null`: capture already ran a VAD, and a commentator must be
able to speak with no user turn at all.

## Cost, measured

Real `response.done` usage from `gpagent commentate --replay sess5`, 151 s of
play, `gpt-realtime-2.1-mini`, priced at that model's rates. Compare with
`gpagent inspect sess5`, whose estimates had never been reconciled against a bill:

| | `image_detail: high` | `image_detail: low` |
|---|---|---|
| input audio | 1676 tok · $0.0072 | 1761 tok · $0.0080 |
| input image | 31008 tok · $0.0089 | 6500 tok · $0.0020 |
| input text | 9976 tok · $0.0022 | 8230 tok · $0.0019 |
| input cached | 27134 tok · $0.0023 | 10110 tok · $0.0009 |
| **output audio** | 1011 tok · **$0.0202** | 1085 tok · **$0.0217** |
| output text | 659 tok · $0.0016 | 669 tok · $0.0016 |
| **total** | **$0.0423** ($0.0168/min) | **$0.0361** ($0.0143/min) |

Three things this changes about the Component A budget:

1. **Output audio is now the largest single line** — about half the bill, and
   entirely controlled by how often and how long the agent talks. That is why the
   speaking policy, not the frame policy, is the centre of this component.
2. **`gpagent inspect` counts each item once; the API re-bills the whole
   conversation as input on every turn.** So 14 images sent can bill 31008 image
   tokens. Estimates from a capture are a lower bound on input, not a forecast.
3. **Images are no longer 88% of input.** One frame per response at most brings
   them to roughly a third of input at high detail, and `detail: "low"` is a 15%
   saving on the *total*, not the 9x it looks like from the input table alone.

At low detail the remarks stay plausible but get vaguer — they react to the
player's tone rather than the screen ("nice stealth mode vibes" vs "let me know
who shows up"). High is the default; `--image-detail low` is there when the bill
matters more than the HUD.

### Trimming the context made it more expensive

The obvious lever, given that the whole conversation is re-billed each turn, is
to delete old items. Measured, it loses:

| | image tokens | input cost |
|---|---|---|
| keep every image | 31008 | **$0.0204** |
| keep the last two | 8721 | $0.0268 |

Deleting an item invalidates the prompt-cache prefix behind it, and cached input
is ~10x cheaper than fresh. Fewer tokens, more money. Both client-side trimming
knobs (`agent.keep_images`, `agent.prune_after_s`) therefore **default to off**;
the server's `truncation.retention_ratio` is the right mechanism, since it drops
history in amortized batches specifically to keep the cache prefix intact. The
knobs remain for hard bounds on very long sessions.

## Playback, and not hearing yourself

Output goes to the **default sink**, because Component A's echo canceller uses
that sink's monitor as its AEC reference. Target a specific device and the agent
hears itself and replies to itself, forever. The requirement is the routing, not
a particular element — `agent.audio_sink` defaults to `autoaudiosink`, which
names no device, so WirePlumber puts it on the default sink and re-routes it when
the default changes. `test_playback_stream_lands_on_the_default_sink` asserts
this against the live PipeWire graph rather than against the element name.

**Barge-in.** `speech.start` fires at VAD detection time, before the segment
exists. On it the agent sends `response.cancel`, flushes the player, and sends
`conversation.item.truncate` with the milliseconds actually heard, so the model's
idea of what it said matches what the player heard. Audio arriving after the
cancel is dropped rather than played, or the player hears a fragment of the
sentence that was just cut off.

## Offline development

`gpagent.replay.ReplaySource` implements the same `CaptureSource` protocol the
real sources do, so `CaptureBus([ReplaySource("sess5")])` is indistinguishable
from live capture. It also reconstructs the signals the live sources publish
(`gamepad.intensity`, `speech.start`), which never appear in `events.jsonl` —
`speech.start` is placed at the *start* of each utterance, not just before the
segment event, or the "player is talking" gate would open and close in the same
instant and barge-in would be impossible to reproduce offline.

Two drivers over one agent:

- `drive_from_bus` — live capture, and wall-clock replay.
- `drive_from_session` — reads the file directly and passes `now=event.t`. No
  sleeping, no wall clock, fully deterministic. This is `--speed 0`.

`--dry-run` never connects. It prints the turn it would have sent and answers
itself with a synthesised `response.done`, so the policy's response lifecycle,
barge-in and the cost meter are all exercised offline. A dry run that dropped
requests would leave the agent waiting forever for a response that never lands,
which is the exact failure it exists to catch.

## Verifying against a real game

Automated tests cover none of the feel. Run this by hand, from the desktop
session:

1. `gpagent commentate` with a game running and a pad connected. Approve the
   portal dialog once.
2. **Ask it something out loud.** It should answer within a couple of seconds,
   in voice, having seen the screen.
3. **Play quietly for a minute or two.** It should remark unprompted
   occasionally — and if you ignore it, notice it getting less frequent.
4. **Interrupt it mid-sentence.** It must stop immediately, and its next reply
   must not act as though it finished the sentence.
5. **The decisive one: let it talk for 30 s and stay silent.** Then check the
   session for `speech.segment` events in that window — there must be zero. If
   there are any, the agent is hearing itself and will eventually reply to
   itself; check that playback is going to the default sink.
6. `gpagent commentate -o out` then read `out/agent.jsonl` — every decision,
   including the ones where it chose to stay quiet and why.

## Component B gotchas

- `session.audio.output.format` requires `rate`, even though the SDK's TypedDict
  marks it optional and the docs example omits it. The server rejects the session
  without it. Caught by `pytest -m network`, which exists for exactly this class
  of bug: the mocked tests structurally cannot find a payload the server rejects.
- `response.create.instructions` **replaces** the session instructions for that
  response; it does not add to them. Sending only a per-reason line threw the
  persona away on every turn — the first live run answered a swearing player with
  four sentences of encouraging life coaching. `instructions_for()` sends persona
  plus reason.
- `AsyncRealtimeConnection.send()` routes a dict through
  `async_maybe_transform(event, RealtimeClientEventParam)`, which **silently
  drops keys the installed SDK's TypedDicts don't know**. Every payload this
  package sends is asserted to survive that transform intact, so an SDK upgrade
  that starts eating a field (a dropped `turn_detection: null` would hand
  turn-taking back to the server) fails a test instead of degrading the agent.
- `pipewiresink` is the obvious sink and the wrong one: rank 0, a plain
  `GstBaseSink` with no ring buffer, and it garbled speech here even when the
  buffers reaching it were sample-for-sample identical to what the model sent.
  `autoaudiosink` selects `pulsesink`, a `GstAudioBaseSink` built for streamed
  audio.
- An appsrc audio pipeline **cannot preroll before it has data**, and it has none
  until someone speaks. Left unprimed the state change never completes: `start()`
  stalls for its full timeout, the clock never starts, and `set_state(NULL)`
  blocks forever. 20 ms of priming silence fixes all three.
- Buffers pushed into a `format=time` appsrc need explicit timestamps.
  Unstamped, the sink renders them back to back against a clock that has been
  running since startup — 3 s of audio came out in 7 ms. `do-timestamp=true` is
  not the fix either: it stamps buffers as they *arrive*, compressing a 10 s
  reply into the second it took to download.
- `response.output_audio.delta` chunks are base64 of an arbitrary byte count and
  nothing promises an even one. A single odd-length chunk pushed as-is shifts
  every following sample by a byte and the rest of the utterance is white noise.
- The player's timestamps and `event.t` are **different clocks**: `event.t`
  counts from the start of capture, the agent runs on `time.monotonic()`. Mixing
  them makes every frame look hours stale and the agent silently blind.

## Component A gotchas

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
- `videorate max-rate=N` does **not** limit a portal screencast: the stream
  negotiates `framerate=0/1` (variable) and a requested 2 fps measured 21 fps,
  JPEG-encoding the whole way. An explicit `video/x-raw,framerate=N/1`
  capsfilter after `videorate` is what actually throttles it.
- `webrtcdsp echo-suppression-level=high` suppresses the *player* during
  double-talk. Talking over game audio is double-talk by definition, so the
  default is `moderate`.
- `silero_vad.onnx` does **not** carry its own context. The reference wrapper
  prepends 64 samples of the previous chunk, so the input tensor is 576 wide at
  16 kHz, not 512. The graph declares its input as `[None, None]` and therefore
  accepts a bare 512 without complaint — it just returns ~0 for everything,
  speech included. `tests/test_vad_model.py` pins both the tensor width and a
  positive detection, because every *negative* test passes against a VAD that
  is permanently asleep.
