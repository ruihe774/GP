# gpagent — Architecture for Agents

A "friend on the couch" commentary agent that watches gameplay and comments naturally. Two independent halves: capture and agent.

## Architecture

```
Gamepad + Microphone + Screen
         ↓
    Component A: Capture
    (discretize into events)
         ↓
    events.jsonl + blobs/
    (sparse, cheap event stream)
         ↓
    Component B: Agent
    (decide when to speak, call gpt-realtime-2.1)
         ↓
    Commentary + recorded conversation
```

The split exists because **raw media to realtime API costs more than the game**. Capture emits only informative moments; the agent decides when to say something.

## Key Modules

### Component A: Capture (`src/gpagent/capture/`)

**Purpose:** Convert gamepad, microphone, and screen into a discretized event stream.

- `gamepad.py` — gamepad input capture via `/dev/input/event*`. Device discovery via udev heuristics. Emits `gamepad.activity` events with discretized summaries (taps, holds, intensity).
- `audio.py` — microphone capture via PipeWire. VAD segmentation (Silero ONNX or WebRTC). Emits `speech.segment` events as headerless PCM16 24 kHz mono. AEC uses system default sink monitor as reference.
- `screen.py` — screen capture via xdg-desktop-portal. Portal consent dialog handled once, token cached. JPEG encoding. Emits `screen.frame` events triggered by policy.
- `segmenter.py` — VAD model interface. Silero loads `.onnx` file; context tensor is 576 wide (64 samples padding + 512 input), not bare 512. Never returns positive for a raw 512-sample input.
- `vad.py` — VAD abstraction (Silero or WebRTC backend, configurable).
- `triggers.py` — `TriggerPolicy` decides when frames are worth emitting. Four triggers:
  - `gamepad` — activity spike
  - `scene` — screen changed (sample-to-sample comparison, not drift-accumulated)
  - `heartbeat` — timeout floor (metronome)
  - `dedup` — has model seen this already?
  - **Dead triggers fixed:** `scene` now compares sample-to-sample (was comparing to last-sent, drifting with capture-suppression windows); `gamepad` intensity now normalizes over enabled input families (was capped at 0.75 with sticks off).
- `evdev_raw.py` — raw `/dev/input/event` reading and parsing.
- `gst_util.py` — GStreamer utility.
- `portal.py` — xdg-desktop-portal ScreenCast session management. Holds D-Bus connection deliberately; dropping it closes the portal session.

**Event types:**
- `gamepad.activity` — button presses, holds, trigger analog values, stick motion, intensity score, summaries.
- `speech.segment` — VAD-gated speech as PCM blob, duration, sample rate, encoding.
- `screen.frame` — JPEG blob, dimensions, trigger reason, scene score.

**Key design patterns:**
- Summaries are **edge-triggered** — "holding RT" fires once when it starts, once when it ends. Windows in between say nothing.
- `holding` field carries state on *any* emitted event, so readers always know what's down without repetition.
- Everything is **discovered, not named** — gamepad capability probes, PipeWire auto-routes, screen geometry from portal.
- Sticks are **silent by default** (`sticks_mode=off`) — moving the stick never triggers frames.
- Frame **triggers throttle, not debounce** — first frame of a burst is informative; later ones are noise.

### Component B: Agent (`src/gpagent/agent/`)

**Purpose:** Consume the event stream, decide when to speak, and call gpt-realtime-2.1.

- `session.py` — main event loop. Holds `AsyncRealtimeConnection` and drives the API. Accumulates gamepad summaries and chooses at most one frame per response. Implements barge-in (cancel response, truncate audio) on player speech. Listens for `response.done`, computes cost and usage.
- `policy.py` — `SpeakPolicy` pure logic (mirrors `TriggerPolicy`). Three reasons to speak:
  - `reply` — player asked something (global cap + `reply_min_gap_s` between responses, exempt from quiet floor)
  - `react` — scene change or input burst (quiet floor + `event_cooldown_s` + backoff)
  - `ambient` — been quiet a while (quiet floor + `ambient_after_s` + backoff)
  - **A global cap** (`max_per_min`) binds *all* reasons.
  - **A quiet floor** (`min_gap_s`) binds `react` and `ambient` only.
  - Cooldowns adapt: `backoff_factor` on ignored remarks, `engagement_boost` (<1) on player engagement.
  - Bursty input cannot become bursty speech: `reply` has `reply_min_gap_s` spacing measured from response end.
- `transport.py` — handles Realtime API message formatting, payload transform validation.
- `playback.py` — audio output via GStreamer. Appsrc pipeline with 20 ms priming silence (needed for preroll). Buffers need explicit timestamps. Uses `autoaudiosink` (selects `pulsesink`, not `pipewiresink`). Output goes to **default sink** because Component A's AEC uses it as reference.
- `context.py` — conversation context held by the session.
- `persona.py` — agent personality and instructions.
- `env.py` — environment utilities (API key loading as `Secret` to prevent accidental logging).
- `config.py` — agent-side configuration (separate from capture config).

**Key design patterns:**
- Speaking policy is decision-making, not the prompt. The model is only asked when a moment is already judged worth speaking at.
- **Gamepad and frames are accumulated and flushed at speak time**, not streamed. 91 windows = 91 turns = 91 bills.
- At most **one frame per response, and only if newer than last-sent**. Capture and send rates are decoupled.
- `response.create.instructions` **replaces** (not adds to) session instructions. Persona must be included in every `instructions_for()` call.
- `response.cancel` on `speech.start` (VAD fires before segment exists). Truncate audio to milliseconds actually heard so model's idea matches player experience.
- **Barge-in gates:** "player is talking" and "response in flight" are absolute. Both gates timeout to prevent permanent silence.
- `AsyncRealtimeConnection.send()` silently drops unknown keys via SDK transform validation. Every payload is asserted to survive the transform.

### Bus & Events (`src/gpagent/bus.py`, `src/gpagent/events.py`)

- `bus.py` — pub/sub event bus. Sources push events (`CaptureSource` interface), sinks consume them.
- `events.py` — event dataclass definitions (all events carry `t`, `seq`, `type`).

### Replay (`src/gpagent/replay.py`)

- `ReplaySource` — reads `events.jsonl` and re-emits with original timing (or deterministic if `--replay-speed 0`). Reconstructs signals that don't appear in the file (`gamepad.intensity`, `speech.start`). Same protocol as live sources.
- **Two drivers:**
  - `drive_from_bus` — wall-clock replay
  - `drive_from_session` — reads file, passes `now=event.t`, fully deterministic (no sleeping)

### CLI (`src/gpagent/cli.py`)

Commands:
- `devices` — what was detected, usability
- `monitor` — live per-button view, check mapping, detect drift
- `record` — capture a session
- `inspect` — timeline, stats, estimated cost, WAV export, contact sheets
- `replay` — re-emit with original timing
- `commentate` — run agent over captured or live events

Config via `--set key=value` (runtime) or TOML files (persistent).

### Configuration

- `src/gpagent/config.py` — capture config loading from `gpagent.toml` or `gpagent.example.toml`
- `src/gpagent/agent/config.py` — agent config
- Global defaults in code, overrideable via:
  - Command-line `--set` flags
  - TOML files in `~/.config/gpagent/`
  - Environment variables (API key)

### Sinks & Output

- `src/gpagent/sinks/` — output writers (session directory, manifest, cost estimation)

## Important Gotchas for Agents

### Component A Gotchas

1. **Portal session lifecycle:** `ScreenCastSession` holds D-Bus connection deliberately. Dropping it closes the portal session. `pipewiresrc` fails with "target not found" if this happens.
2. **Silero VAD context:** The input tensor is **576 wide**, not 512. Bare 512 returns ~0 for everything because ONNX accepts any shape and the graph has no context. Tests pin both the tensor width and a positive detection.
3. **Frame rate capping:** `videorate max-rate=N` does NOT limit portal screencasts. Stream negotiates `framerate=0/1` (variable). Use an explicit `video/x-raw,framerate=N/1` capsfilter *after* videorate.
4. **Scene score bug (fixed):** Was comparing to last-sent frame, so the longer capture-suppression windows let the screen drift and any threshold cleared after enough time. Now compares sample-to-sample, fires on real transitions, and `dedup` (last-sent comparison) is a separate step.
5. **Gamepad intensity normalization (fixed):** Was weighted as buttons 0.45, triggers 0.30, sticks 0.35, but sticks default to `off` (silently zeroed that weight). Renormalized over actually-enabled families, so the scalar means the same thing regardless of `sticks_mode`.
6. **WebRTC DSP on double-talk:** `echo-suppression-level=high` suppresses the *player* during double-talk. Talking over game audio is double-talk by definition. Default is `moderate`.
7. **GStreamer device dedup:** `DeviceMonitor` reports each device 2–3× and a sink shares `node.name` with its monitor. Dedup on `(media.class, node.name)`.
8. **AEC reference:** Must be the *whole* system output (default sink monitor), not just agent output. Game audio comes through speakers into mic; without system-wide AEC it reads as speech.

### Component B Gotchas

1. **Session audio format:** `session.audio.output.format` requires `rate` even though TypedDict marks it optional and docs omit it. Server rejects without it.
2. **Instructions replace, not add:** `response.create.instructions` **replaces** session instructions. Persona got thrown away on every turn in early runs — first live test answered a swearing player with encouraging life coaching.
3. **SDK payload transform silently drops keys:** `AsyncRealtimeConnection.send()` silently drops keys the installed SDK doesn't know about. Every payload is asserted to survive the transform intact. An SDK upgrade eating a field (e.g., `turn_detection: null` handing turn-taking back to server) will fail a test.
4. **Appsrc preroll deadlock:** Appsrc cannot preroll before it has data. Unprimed, state change blocks forever and clock never starts. 20 ms of priming silence fixes all three.
5. **Timestamps on time-format appsrc:** Buffers need explicit timestamps. Unstamped, sink renders them back-to-back against startup clock — 3 s of audio in 7 ms. `do-timestamp=true` doesn't fix it; it stamps on arrival, compressing the download window.
6. **Base64 chunk alignment:** `response.output_audio.delta` chunks are base64 of arbitrary byte count. Odd-length pushed as-is shifts all following samples by a byte, rest is white noise.
7. **Audio sink selection:** Output to default sink (component A's AEC reference). Target a specific device and agent hears itself, replies to itself. Use `autoaudiosink`, which selects `pulsesink` and re-routes with WirePlumber. Do NOT use `pipewiresink` (rank 0, plain GstBaseSink, garbled speech here even with identical buffers).
8. **Clock mismatch:** Player timestamps (`event.t` from capture) and agent clock (`time.monotonic()`) are different. Mixing them makes frames look hours stale and agent silently blind. The offset between them (`_t_offset`) is estimated as the **smallest** `now - event.t` ever seen, not the first: the first event comes out of a backlog built up while `start()` opened the realtime session, and calibrating on it recorded every response 3.5 s early in a real 39-minute session.
9. **Response lifecycle timeouts:** "Player is talking" and "response in flight" are absolute gates. A stuck gate is permanent silence (quietest possible bug). Both timeout (`speech_start_timeout_s`, `response_timeout_s`).
10. **Sessions expire at 60 minutes.** The server closes with "Your session hit the maximum duration of 60 minutes", the event stream ends, and without a reconnect loop the agent is mute for the rest of the run (sess7 lost its last 82 s). `_pump_events` reopens with backoff and reseeds instructions only; history does not come back, and item ids from the dead session must be dropped or every `truncate`/`delete` errors.
11. **Barge-in only if audible.** Cancel on `speech.start` *only when audio has already reached the player*. A response that has not made a sound is not talking over anyone, and killing it burns a paid turn and leaves the question unanswered — a player talking in short bursts cancelled four consecutive replies at 0.0 s in sess7.
12. **TPM is a real ceiling on long sessions.** The API re-bills the whole conversation each turn, so input per request grows with session length (~10.5 k tokens by minute 60). At a 40 k TPM org limit that is ~4 requests/min, below `speak.max_per_min = 6`; sess7 lost two responses to `rate_limit`. Non-completed `response.done` statuses are logged as `note` lines — this is how you see it.

## Cost Model

From `gpagent.toml` and observed on real sessions:

- **Input audio:** $32/1M tokens, ~1 token per 100 ms → $0.019/min open mic. Gated by VAD.
- **Input image:** $5/1M tokens, ~765 tokens per 1024×576 frame at high detail, ~85 at low detail. At most one per response.
- **Input text:** $4/1M tokens (gamepad summaries, negligible).
- **Input cached:** ~10× cheaper than fresh (prompt-caching).
- **Output audio:** ~$2 per minute of speaking. **Largest single cost line.** Controlled entirely by speaking policy.

**Frame policy is less valuable than in capture.** Component A's most expensive tuning lever. Component B sends at most one frame per response, so capture rate and send rate decouple. Capture 100, send 14.

**Context trimming is counterintuitive:** Deleting old items invalidates prompt-cache prefix. Cached input is 10× cheaper; fewer tokens can mean more money. Client-side trimming defaults off; server's `truncation.retention_ratio` is the right mechanism (drops in amortized batches to keep cache prefix intact).

## Testing Strategy

- `tests/` — 250+ tests, no hardware, no network, ~1 s.
- Synthetic capability bitmaps for device classification.
- Synthetic `input_event` structs for discretizer.
- Scripted probabilities for segmenter.
- Virtual clock for trigger and speak policies.
- `pytest -m hardware` — 8 more tests against real devices (optional).
- `pytest -m network` — 2 more tests against real API (spends money, finds SDK payload bugs the mocked tests can't).

**Avoid negative tests:** A VAD that returns zero for everything passes "never says nothing is speech" perfectly. Assert positive behavior first. Test that `SpeakPolicy` never speaks also passes "respects cooldown", "obeys rate cap" perfectly.

## Key Concepts

### Discretization

Only send the moments that carry information. VAD gates speech. Trigger policy gates frames. Gamepad events emit only on state changes (edge-triggered). Eliminates streaming raw media.

### Trigger Policy (Capture)

Four independent reasons to emit a frame, each with its own floor and cooldown:
- `gamepad` — activity intensity crosses threshold
- `scene` — screen changed (sample-to-sample, not accumulated drift)
- `heartbeat` — timeout floor (metronome, keeps agent fresh)
- `dedup` — model already saw this (last-sent comparison)

Each reason can fire independently; a global cap binds all four together so one hyperactive trigger doesn't starve the others.

### Speak Policy (Agent)

Three independent reasons to say something, each with floor and cooldown:
- `reply` — player said something (exempt from quiet floor, has `reply_min_gap_s` spacing)
- `react` — scene or input burst (needs `burst_windows` hot windows, then quiet floor + cooldown + backoff)
- `ambient` — been quiet (needs `ambient_after_s` quiet + cooldown + backoff)

Global cap (`max_per_min`) over all three. Cooldowns adapt on engagement (backoff on silence, reset on player talk).

### Barge-In

Player cuts off agent mid-sentence:
- `speech.start` fires at VAD detection (before segment event).
- Agent sends `response.cancel`, truncating audio to seconds heard.
- `conversation.item.truncate` with milliseconds heard so model's idea matches reality.
- Audio after cancel is dropped, not played.

## Running Tests

```bash
pytest                    # 250 tests, no hardware
pytest -m hardware        # + 8 hardware tests
pytest -m network         # + 2 API tests (spends money)
pytest tests/test_speak_policy.py::test_rate_cap_holds_globally
```

Specific test files:
- `test_speak_policy.py` — asserts combined rate of all three reasons
- `test_triggers.py` — frame policy logic
- `test_gamepad_discretizer.py` — input mapping and summaries
- `test_segmenter.py` — VAD context
- `test_replay_source.py` — replay signal reconstruction

## Configuration Files

Example config layout:

```toml
[gamepad]
sticks_mode = "off"           # or "intensity", "full"
button_map = { BTN_NORTH = "X" }  # override spec

[gamepad.device_button_map."2dc8:200f"]
BTN_NORTH = "X"               # per-device overrides

[audio]
vad_backend = "silero"        # or "webrtc"
echo_cancel = true
echo_suppression_level = "moderate"

[screen]
long_edge = 1280              # max dimension
detail = "high"               # or "low" for API

[capture.triggers]
gamepad_threshold = 0.35
scene_threshold = 0.06
heartbeat_s = 3.0
min_interval_s = 2.0

[agent]
reply_min_gap_s = 3.0         # space out consecutive replies
max_per_min = 8               # global rate cap
```

## Common Workflows for Agents

**Adding a new input type:** Implement `CaptureSource`, emit events to bus. Follow discretization pattern (edge-trigger, accumulate state in `holding` field).

**Tuning speak timing:** Adjust `SpeakPolicy` parameters. Test with `--replay-speed 0 --dry-run` for deterministic offline runs.

**Debugging cost:** Check `inspect --cost` estimates vs. actual API usage in `manifest.json`. Cost meter in `session.py::usage_for_response()`.

**Finding frame stale:** `commentate` reports frame age at send time. Older frames mean `triggers.min_interval_s` is too high.

**Portal issues:** `manifest.json::devices.screen` carries portal diagnostics (handshake time, stalls, portal errors).
