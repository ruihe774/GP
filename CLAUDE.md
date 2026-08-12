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

The split exists because raw media to a realtime API costs more than the game. Capture emits only informative moments; the agent decides when to say something.

## Key Modules

### Component A: Capture (`src/gpagent/capture/`)

Converts gamepad, microphone, and screen into a discretized event stream.

- `gamepad.py` — gamepad input capture via `/dev/input/event*`. Device discovery via udev heuristics. Emits `gamepad.activity` events with discretized summaries (taps, holds, intensity).
- `audio.py` — microphone capture via PipeWire. VAD segmentation (Silero ONNX or WebRTC). Emits `speech.segment` events as headerless PCM16 24 kHz mono. AEC uses the system default sink monitor as reference.
- `screen.py` — screen capture via xdg-desktop-portal. Portal consent dialog handled once, token cached. JPEG encoding. Emits `screen.frame` events triggered by policy.
- `segmenter.py` — VAD model interface. Silero's context tensor is 576 wide (64 samples padding + 512 input), not bare 512.
- `vad.py` — VAD abstraction (Silero or WebRTC backend, configurable).
- `triggers.py` — `TriggerPolicy` decides when frames are worth emitting, via four independent reasons (see below).
- `evdev_raw.py` — raw `/dev/input/event` reading and parsing.
- `gst_util.py` — GStreamer utility.
- `portal.py` — xdg-desktop-portal ScreenCast session management. Holds the D-Bus connection deliberately; dropping it closes the portal session.

**Event types:**
- `gamepad.activity` — button presses, holds, trigger analog values, stick motion, intensity score, summaries.
- `speech.segment` — VAD-gated speech as a PCM blob, duration, sample rate, encoding.
- `screen.frame` — JPEG blob, dimensions, trigger reason, scene score.

**Key design patterns:**
- Summaries are edge-triggered — "holding RT" fires once when it starts, once when it ends. Windows in between say nothing.
- The `holding` field carries state on any emitted event, so readers always know what's down without repetition.
- Everything is discovered, not named — gamepad capability probes, PipeWire auto-routes, screen geometry from the portal.
- Sticks are silent by default (`sticks_mode=off`) — moving the stick never triggers frames.
- Frame triggers throttle, not debounce — the first frame of a burst is informative; later ones are noise.

### Component B: Agent (`src/gpagent/agent/`)

Consumes the event stream, decides when to speak, and calls gpt-realtime-2.1.

- `session.py` — main event loop. Holds `AsyncRealtimeConnection` and drives the API. Accumulates gamepad summaries and chooses at most one frame per response. Prunes the disposable half of the conversation (`_maybe_prune`). Implements barge-in (cancel response, truncate audio) on player speech. Listens for `response.done`, computes cost and usage.
- `policy.py` — `SpeakPolicy` pure logic (mirrors `TriggerPolicy`), via three independent reasons to speak (see below).
- `transport.py` — handles Realtime API message formatting, payload transform validation.
- `playback.py` — audio output via GStreamer. Appsrc pipeline with priming silence for preroll; buffers need explicit timestamps. Uses `autoaudiosink` (selects `pulsesink`, not `pipewiresink`). Output goes to the default sink because Component A's AEC uses it as reference.
- `hud.py` — on-screen output instead of voice: an override-redirect X11 window (XWayland on this Wayland session, because that is mutter's topmost layer), showing a stack of toasts in one monitor's corner. Pure layout/render (`ToastStack`, `render_card`, `place`), `HudWindow` for everything X, `TextHud` owning a thread so `show()` never blocks a caller. The session's output device when `agent.output = "text"`, the way the audio sink is when it is `"voice"`; `make_hud` degrades to `NullHud` with no display, exactly as `make_player` degrades to `NullPlayer`.
- `context.py` — conversation context held by the session.
- `subtitles.py` — the `.srt` for what is being watched, read off disk (not captured) and sent **whole, once**, as one system item right after `session.update`. `parse_srt` is deliberately forgiving (optional index line, `,` or `.` decimals, markup stripped, UTF-8 then cp1252); `load_script` raises rather than let a session run silently without the dialogue it was told to use.
- `persona.py` — agent personality and instructions.
- `env.py` — environment utilities (API key loading as `Secret` to prevent accidental logging).
- `config.py` — agent-side configuration (separate from capture config).

**Key design patterns:**
- Speaking policy is decision-making, not the prompt. The model is only asked when a moment is already judged worth speaking at.
- Gamepad and frames are accumulated and flushed at speak time, not streamed, to bound turns/bills to responses rather than windows.
- At most one high-detail frame per response, and only if newer than the last one sent. Capture and send rates are decoupled.
- Behind it rides a trail: `agent.image_trail` earlier frames at `"low"` detail, oldest first, so the model sees the trajectory and not just the destination. Trail frames are picked by spread first, value second — the candidate window is split into N equal time buckets and each contributes at most one frame, chosen by `scene_score`. `image_trail_min_gap_s` holds a floor between sent images at both ends. Candidates are only the gap: frames newer than the last one sent and older than the current one.
- `response.create.instructions` replaces (not adds to) session instructions. The persona is sent once in `session.update`; the per-turn reason goes in as a system message item (`persona.reason_note`), keeping the request prefix stable for prompt caching.
- A turn's items are split by *lifetime*, not by content: controller text in one item (`gpctx`), every screenshot in a second (`gpimg`, trail note + trail + current frame), the per-turn nudge in a third (`gprsn`). An item is the unit `conversation.item.delete` operates on, so anything that has to outlive a screenshot cannot share an item with one. All ids are client-assigned, so pruning never waits on `conversation.item.created`.
- Pruning (`agent.prune_after_s`, off by default) deletes images and per-turn nudges on **one shared cutoff, in one round**. Both halves are about the prompt cache: one cutoff because things created together should be deleted together, and rounds because a delete truncates the cached prefix at the deleted item — and deletions come from the oldest end, so any round costs essentially the whole prefix. That price is per round, so `prune_interval_s` is the knob that decides what pruning costs. The conversation proper is never touched client-side; `truncation_retention_ratio` is for that.
- `response.cancel` fires on `speech.start` (VAD fires before the segment event exists). Audio is truncated to milliseconds actually heard so the model's idea matches player experience.
- Barge-in gates ("player is talking", "response in flight") are absolute and both timeout to prevent permanent silence.
- `AsyncRealtimeConnection.send()` silently drops unknown keys via SDK transform validation. Every payload is asserted to survive the transform.
- Subtitles are sent whole and up front *because* of the cache, not despite it. One item at the head of the conversation is fresh input exactly once and cached input on every turn after; the same dialogue fed a cue at a time would land at the tail of each turn, where nothing is cached yet, and be re-billed as fresh forever. Measured on a sess_movie2 dry run with a 328-cue track: $0.3889 → $0.4356 over 36 responses. The price is that the model holds the ending, which is answered in the prompt: `persona.SUBTITLE_NOTE` rides above the script and the movies persona forbids spoilers. "Where are they now" is answered by *not* answering it — nothing here knows. The only position available is `offset_s` plus elapsed session time, which is playback position only for a film that never pauses, never gets rewound over a line and never sits paused through a phone call; all three drift the clock *ahead* of the viewer, the one direction that hands over dialogue they have not reached, and a stated position is trusted in a way an absent one is not. So `subtitles.position_note` is off by default and the script is sent telling the model to locate the scene from what is on screen and to assume it is earlier than it thinks — evidence, which degrades gracefully, rather than arithmetic, which does not. Turned on (for untouched start-to-finish playback) it appends `HH:MM:SS` to the per-turn nudge, hedged as a hint, plus a paragraph on the script item saying the screen wins any disagreement; it rides on the nudge because that item is appended at the tail, so a value that changes every turn costs no cache. The script item is never pruned (it does not grow, and nothing else can reconstruct it) and is re-seeded after a reconnect, the one exception to "reseed instructions only".
- Voice or text is one session-wide switch, `agent.output`, because it *is* `output_modalities` on `session.update`. What a turn sends is identical either way; what changes is that the model writes the remark (`response.output_text.*`, shown whole on `.done`, never per delta) instead of speaking it, that it goes to the HUD instead of the sink, and that barge-in stops applying — text on screen is not talking over anyone, so `_speaking` narrows to "still generating" and `_audible` is always false. A written remark still occupies time: `hud.hold_for` (reading speed) is what the quiet floor and the reply beat are measured from, standing in for a sentence's duration.

### Bus, Events & Cost (`src/gpagent/bus.py`, `events.py`, `tokens.py`)

- `bus.py` — pub/sub event bus. Sources push events (`CaptureSource` interface), sinks consume them.
- `events.py` — event dataclass definitions (all events carry `t`, `seq`, `type`).
- `tokens.py` — token/cost estimation shared by both halves: `Estimate` is Component A's forward-looking guess from a capture, `UsageMeter` is Component B's record of what the API actually billed (folded from `response.done`). Used by `inspect --cost`.

### Replay (`src/gpagent/replay.py`)

- `ReplaySource` — reads `events.jsonl` and re-emits with original timing (or deterministic if `--replay-speed 0`). Reconstructs signals that don't appear in the file (`gamepad.intensity`, `speech.start`). Same protocol as live sources.
- Two drivers: `drive_from_bus` (wall-clock replay) and `drive_from_session` (reads file, passes `now=event.t`, fully deterministic — no sleeping).
- A `commentate` recording holds the agent's own speech too; `build_cues` drops it, since that's output, not capture. Both drivers go through `build_cues`.

### CLI (`src/gpagent/cli.py`)

Commands:
- `devices` — what was detected, usability
- `monitor` — live per-button view, check mapping, detect drift
- `record` — capture a session
- `inspect` — timeline, stats, estimated cost, WAV export, contact sheets. `--contact-sheet` montages everything captured; `--sent-sheet` montages only what the agent sent, one row per turn, each image labelled with its `seq` and detail (green border = high-detail current frame, grey = trail). Reads the `frames` field of the agent log's `ask` lines, so `--agent-log` points it at a run whose blobs live elsewhere.
- `replay` — re-emit with original timing
- `commentate` — run agent over captured or live events. `--srt PATH` gives the agent the subtitle file for what is being watched (`--srt-offset S` says how far into the film the run starts, and only matters with `subtitles.position_note`, which is off by default; `--no-subtitles` ignores a configured path). `--text` runs the session in text output (`agent.output = "text"`): no sink is opened, the HUD is turned on, and remarks are read rather than heard. `--no-text` forces voice over a config that asks for text.
- `hud` — drive the on-screen overlay by hand: `TEXT...`, `--demo` (a scripted reel), `--stdin` (a line per toast), or `--render PATH` to draw the card to a PNG with no display involved. `--render-size WxH` picks the screen it lays out against.

Config via `--set key=value` (runtime) or TOML files (persistent).

### Configuration

- `src/gpagent/config.py` (capture) and `src/gpagent/agent/config.py` (agent) hold defaults in code. TOML sections `[gamepad] [audio] [screen] [triggers]` are Component A; `[agent] [speak] [hud] [subtitles]` are Component B. Per-device gamepad overrides go under `[gamepad.device_button_map."<vid:pid>"]`.
- Loaded from `gpagent.toml` or `~/.config/gpagent/`, overridable ad hoc with `--set section.key=value`. `gpagent.example.toml` is the canonical annotated reference — every value in it is the current default, so defaults aren't duplicated here.

### Sinks & Output

- `src/gpagent/sinks/` — output writers (session directory, manifest, cost estimation)

## Key Concepts

### Trigger Policy (Capture)

Four independent reasons to emit a frame, each with its own floor and cooldown:
- `gamepad` — activity intensity crosses threshold, normalized over enabled input families
- `scene` — screen changed (sample-to-sample comparison, not accumulated drift)
- `heartbeat` — timeout floor (metronome, keeps agent fresh)
- `dedup` — model already saw this (last-sent comparison)

A global cap binds all four together so one hyperactive trigger doesn't starve the others.

### Speak Policy (Agent)

Three independent reasons to say something, each with floor and cooldown:
- `reply` — player said something (exempt from the quiet floor, has `reply_min_gap_s` spacing measured from response end)
- `react` — scene or input burst (needs `burst_windows` hot windows, then quiet floor + `event_cooldown_s` + backoff)
- `ambient` — been quiet a while (needs `ambient_after_s` quiet + cooldown + backoff)

A global cap (`max_per_min`) binds all three; a quiet floor (`min_gap_s`) binds `react` and `ambient` only. Cooldowns adapt: `backoff_factor` on ignored remarks, `engagement_boost` (<1) on player engagement.

### Barge-In

Player cuts off the agent mid-sentence:
1. `speech.start` fires at VAD detection (before the segment event).
2. Agent sends `response.cancel`.
3. `conversation.item.truncate` with milliseconds heard so the model's idea matches reality.
4. Audio after cancel is dropped, not played.

## Cost Model

**Cost and TPM are different problems and pull in opposite directions.** Cached tokens are ~10x cheaper but count against the rate limit *in full*, so history that is nearly free is not nearly free of TPM. sess_movie2 died of this: thirteen consecutive `rate_limit_exceeded` failures over its last seven minutes, at ~2.3k image tokens added per turn and nothing ever leaving. Client-side pruning is the answer to the second problem and a loss on the first — it buys a bounded request in exchange for one invalidated cache prefix per round.

Subtitles are the one input that is cheap *because* the cache exists: the whole film's dialogue, sent once ahead of the first turn, is fresh input once and cached thereafter (~12% on a sess_movie2 dry run), where the same text streamed cue-by-cue would be uncached every turn. Output audio (speaking) is the dominant cost line and is controlled entirely by the speak policy's rate cap. `agent.output = "text"` deletes that line outright — 14 remarks over a sess5 replay billed 478 output text tokens for $0.0011, against roughly a cent of speech for the same fourteen — which leaves re-billed input as effectively the whole bill. Input images are the next lever: cost is per-frame, not per-pixel-stream, because Component B sends at most one high-detail frame per response regardless of how many Component A captured — capture rate and send rate are fully decoupled. Cached input tokens are far cheaper than fresh ones (prompt caching), which is why trimming for *cost* loses: deleting old items invalidates the cache prefix ahead of them, so it can raise the bill even though it shrinks the conversation (sess5, keeping every image: 31008 image tokens for $0.0204; keeping the last two: 8721 for $0.0268). The server's `truncation.retention_ratio` is the sanctioned mechanism — it drops in amortized batches that don't disturb the prefix. `agent.prune_after_s` is the client-side one, off by default, and it is a TPM instrument rather than a cost one. Exact per-token rates live in `tokens.py::MODEL_RATES`, not here.

## Testing Strategy

Default `pytest` run needs no hardware and no network (device capability bitmaps, `input_event` structs, VAD probabilities, and trigger/speak-policy clocks are all synthetic/virtual). Two marked tiers opt into real resources:

```bash
pytest               # default: no hardware, no network
pytest -m hardware   # + real input devices
pytest -m network    # + real API calls — spends money, catches SDK payload bugs mocked tests can't
```

Avoid negative tests: assert positive behavior first. A policy that never speaks trivially passes "respects cooldown" and "obeys rate cap".

## Linting & Type Checking

`ruff` (lint) and `mypy` (types), configured in `pyproject.toml` under `[tool.ruff]` / `[tool.mypy]`, installed via the `dev` dependency group (`uv sync`).

```bash
uv run ruff check .   # lint
uv run mypy           # type check (src + tests)
```

Both must be clean before committing. Notes on the current config:

- `SIM115` (bare `open()` outside a `with`) is disabled repo-wide: several long-lived handles (`JsonlSink._events`, `OpenAITransport`'s log file, GStreamer pipeline handles) are opened in a `start()`/`open()` method and closed in a paired `close()`, which isn't expressible as a single `with` block.
- GStreamer/GObject-introspection handles (`gi.repository.Gst` objects) are untyped at the source, so attributes that hold them are explicitly annotated `Any` rather than left to infer as `None`. Real `openai` SDK types (`AsyncOpenAI`, `AsyncRealtimeConnection`) are used instead of `Any` wherever the SDK actually exports them.
- `OpenAITransport.send()` casts its payload to `Any` at the SDK boundary on purpose — that layer speaks in plain dicts validated at runtime by the SDK's own transform, not by mypy.

## Common Workflows for Agents

**Adding a new input type:** Implement `CaptureSource`, emit events to the bus. Follow the discretization pattern (edge-trigger, accumulate state in `holding` field).

**Tuning speak timing:** Adjust `SpeakPolicy` parameters. Test with `--replay-speed 0 --dry-run` for deterministic offline runs.

**Measuring what history costs:** `commentate --dry-run` bills the whole conversation on every response, not just the turn's own items — `RecordingTransport` keeps a ledger of live items, `conversation.item.delete` removes from it, and cached tokens are modelled by the same prefix rule the server uses. So the growth curve, and what a pruning setting does to it, can be read offline: `commentate --replay sess_movie2 --dry-run -c movies.example.toml` against `--set agent.prune_after_s=0`. On sess_movie2 that is a peak request of 15.6k tokens against 46.4k, and a worst 60 s window of 52k against 163k. It is a model of the server, not the server — read the shape, not the fourth digit.

**Debugging cost:** Check `inspect --cost` estimates vs. actual API usage in `manifest.json`. Cost meter in `session.py::usage_for_response()`. Per-response `usage` (and the cached-token split) lives on `agent.response` events in `events.jsonl`, not `agent.jsonl` (console story only). `-o DIR --no-media` writes the event stream without payloads — the shape to use when re-running a recording to measure a change rather than to listen to it.

**Finding frame stale:** `commentate` reports frame age at send time. Older frames mean `triggers.min_interval_s` is too high.

**Seeing what the model actually looked at:** `ask` lines in `agent.jsonl` carry `sent` (console story: `frame 1024x576 high (0.8s old)`, `trail 2 low (back to 6.3s)`) and `frames` (identities: `{"current": 3870, "detail": "high", "trail": [3809, 3837, ...], "trail_detail": "low"}`). `gpagent inspect DIR --sent-sheet [--agent-log RUN/agent.jsonl]` turns the second into one labelled row of images per turn — use it to check that a trail spans the window instead of clustering.

**Portal issues:** `manifest.json::devices.screen` carries portal diagnostics (handshake time, stalls, portal errors).

## Known Gotchas

**Capture:**
- Cap frame rate at the compositor with a `video/x-raw,max-framerate=N/1` capsfilter directly on `pipewiresrc`; dropping downstream is too late since the readback cost is already paid. `framerate=N/1` is rejected (the stream is variable-rate); only `max-framerate` negotiates. Keep `videorate ! video/x-raw,framerate=N/1` downstream as a floor for compositors that ignore `max-framerate`.
- `echo-suppression-level=high` suppresses the player during double-talk (talking over game audio is double-talk by definition); default is `moderate`.
- `DeviceMonitor` reports each device 2–3×, and a sink shares `node.name` with its monitor — dedup on `(media.class, node.name)`.
- AEC reference must be the whole system output (default sink monitor), not just agent output, or game audio through speakers into the mic reads as speech.

**Agent:**
- `session.audio.output.format` requires `rate` even though the TypedDict marks it optional and docs omit it.
- Base64 audio delta chunks are arbitrary byte counts; odd-length chunks pushed as-is shift all following samples by a byte.
- The offset between player clock (`event.t`) and agent clock (`time.monotonic()`) is estimated as the smallest `now - event.t` ever seen, not the first (the first event can come out of a startup backlog).
- Realtime sessions expire at 60 minutes; `_pump_events` reconnects with backoff and reseeds instructions only — history does not come back, and item ids from the dead session must be dropped before any `truncate`/`delete`.
- TPM is a real ceiling on long sessions: the API re-bills the whole conversation each turn, so input tokens grow with session length. Non-completed `response.done` statuses are logged as `note` lines. `agent.prune_after_s` is the client-side bound on that growth; it is off by default because as a *cost* measure it loses to the prompt cache.
- A `manifest.json` is a record of what a past run used, not a config the user is asking for, so `merge_dict(..., strict=False)` ignores keys this version has dropped and says which. A `-c` file or `--set` stays strict: there an unknown key is a typo.
- `subtitles.position_note` is off by default and should stay off for ordinary viewing: it extrapolates from wall time, so a pause or a seek makes it lie, always forward. With it on, `offset_s` is the only thing tying the script's clock to the session's and nothing can detect it — leave it at 0 on a film already playing and every note is wrong by that much. The note is also suppressed until the first capture event arrives, because before that `_t_offset` is unknown and `time.monotonic()` would report the machine's uptime as the film position.
- `tokens.text_tokens` (chars/4) reads a CJK subtitle track low by roughly 3x, so the size printed at startup is a sanity check on the file, not a bill. Measured on sess_movie3: a 351-cue Chinese track printed `~1879 tok` and billed ~6000 on the first request.
- A response can produce more than one text part — sess_movie3 had one of thirty-two come back as a preamble plus the remark, two `response.output_text.done` events 0.01 s apart. Each part is shown and logged as it arrives; `_add_transcript` joins them so the recorded `agent.response` is what the viewer actually read. Nothing here ever carries reasoning: `response.output_text.done` is an assistant *content part*, and this SDK has no reasoning-text event at all.
- Barge-in should only cancel a response once its audio has actually reached the player — a response that hasn't made a sound isn't talking over anyone.
- The playback clock must be cleared between turns for non-realtime players (`player.discard()` before the next response), or the next utterance's audio accumulates on top of the previous one.

**HUD:**
- Text output is a session-wide choice, not a rendering one: it changes `output_modalities` on `session.update` and moves the words onto `response.output_text.*` instead of audio deltas plus a transcript. No recording on disk holds a text response (every capture was made with `["audio"]`), so replaying one exercises the *inputs* against a live text session and nothing offline can stand in for that — verify a change to this path with a real run, e.g. `gpagent commentate --replay sess5 --text -o DIR`.
- `response.output_text.done` is also emitted for a response that was interrupted or cancelled, so the accept-output gate has to cover text as well; without it a cancelled remark still lands on screen.
- A text `agent.response` has `modality: "text"`, no blob and `dur_ms: 0` — the `transcript` is the remark itself, not a transcript of anything.
- Depth-32 X windows are composited under the Render convention, i.e. **premultiplied** alpha. Straight alpha rings every glyph and rounded corner with a bright fringe.
- The X screen is the union of the outputs (8960x2880 across two here, primary at x=3840), and XWayland reports **device** pixels while the compositor scales by `Xft.dpi` (192 = 2x). Placement is against one `Monitor` rect in root coordinates, and every `*_px` is a design pixel at 96 dpi multiplied by `Monitor.scale`.
- `python-xlib` 0.33 has no XFixes region API, so click-through is the SHAPE extension's `Input` kind with an empty rectangle list. `put_image` does not split oversized requests either — a card is sent in row bands under `max_request_length * 4`.
- Pillow does no font fallback: one file renders the whole card, and DejaVu Sans has no CJK. `hud.font_lang = "ja"` routes the lookup through fontconfig, which is the only reason a Japanese remark renders as words.
