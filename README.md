# gpagent — Your AI Gaming Buddy

gpagent watches you play games and comments on the action like a friend sitting next to you on the couch. No setup hassle — just plug in your gamepad, grab your microphone, and let it watch.

```bash
gpagent commentate      # start watching and chatting live
```

It captures what you're doing (button presses, voice, screen) and feeds it to an AI that decides when to chime in. Ask it questions, and it answers based on what's happening. Ignore it, and it gracefully gets quieter. Interrupt it mid-sentence, and it stops talking.

## What You'll Need

- **Linux** with PipeWire audio and Wayland/X11 (tested on Ubuntu 26.04)
- **Python 3.12+**
- A gamepad (any USB controller)
- A microphone
- An OpenAI API key (set `OPENAI_API_KEY` or create `.env` at the repo root)

System packages (Ubuntu/Debian):
```bash
sudo apt install gstreamer1.0-pipewire gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad python3-gi gir1.2-gst-plugins-base-1.0
```

Note: Run from your desktop, not over SSH. The gamepad and screen capture need to be connected to your display session.

## Install

```bash
uv venv --python 3.12 --system-site-packages
uv pip install -e ".[dev]"

# Download the voice activity detection model (~50 MB)
curl -L -o models/silero_vad.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
```

## Quick Start

**Check your setup:**
```bash
gpagent devices          # lists gamepad, mic, screen — shows if they're ready
```

**Capture a gameplay session:**
```bash
gpagent record -o my_session -d 60    # record 60 seconds
```
On first run, a dialog asks for screen permission. Approve once — it's cached for next time.

**Watch what was captured:**
```bash
gpagent inspect my_session            # timeline and stats
gpagent inspect my_session --wav      # listen to your voice
```

**Let the AI comment on it:**
```bash
gpagent commentate --replay my_session   # run the agent over the recording
```

**Or go live:**
```bash
gpagent commentate                    # capture and comment in real time
```

## Commands

| Command | What it does |
|---------|-------------|
| `devices` | Check gamepad, mic, screen — are they detected? |
| `monitor` | Live button press display, test your controller mapping |
| `record -o NAME -d 60` | Capture gamepad, voice, and screen for 60 seconds |
| `inspect NAME` | View the captured session timeline, stats, and estimated cost |
| `commentate --replay NAME` | Run the agent over a recorded session |
| `commentate` | Capture and comment live in one shot |

## Customization

Change things on the fly with `--set`:
```bash
gpagent record --set screen.long_edge=1280 --set audio.vad_backend=webrtc

# Remap a button (if your controller's buttons are backwards)
gpagent record --map BTN_NORTH=X --map BTN_WEST=Y

# Disable screen capture for just audio + gamepad
gpagent record --no-screen

# Pick a different screen or window
gpagent record --pick-screen
```

Or create a `.toml` file in `~/.config/gpagent/gpagent.toml` for permanent settings.

**Language support:**
```bash
gpagent commentate --language ja --voice cedar
```
Speak Japanese (or any ISO 639-1 language code). Voice choices: `alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `marin`, `sage`, `shimmer`, `verse`.

```bash
gpagent commentate --voice-speed 1.3
```
Speed up or slow down the agent's actual voice (clamped to `[0.25, 1.5]`). Not to be confused with `--replay-speed`, which controls how fast a recorded session is re-emitted.

**Control the bill:**
```bash
# Lower detail on images (cheaper, slightly vaguer)
gpagent commentate --image-detail low --replay my_session

# Slow down playback for longer context
gpagent commentate --replay my_session --replay-speed 0.5
```

## Recording Conversations

Save the whole exchange (your voice and the AI's):
```bash
gpagent commentate -o chat_session   # outputs chat_session/ with both sides
gpagent inspect chat_session --wav   # listen to the full conversation
```

Creates `chat_session/conversation.wav` — both voices laid out with silence in the gaps, so it reads like an actual conversation, not a pile of clips.

## Troubleshooting

**Controller not detected?**
```bash
gpagent monitor           # press buttons, see them appear
gpagent monitor --raw     # see the raw kernel codes if mapping looks wrong
```

**Sticks drifting?**
```bash
gpagent record --set gamepad.sticks_mode=off   # ignore stick movement (default anyway)
```

**No voice coming through?**
- Check your microphone is plugged in and selected as the system default.
- If game audio is loud, the AI might be echo-cancelling your voice. Try:
  ```bash
  gpagent record --set audio.echo_cancel=false
  ```

**Agent won't stop talking?**
- It's probably replying to itself. Check that audio output is going to your speakers (default sink), not to a USB device that's also your microphone.

**First run slow?**
- The portal permission dialog might be taking time. Answer it and subsequent runs are faster.

## Cost

Each minute of commentary costs roughly $0.015–$0.02 (at current OpenAI prices). Most of that is the agent's voice output, not the video or audio input. Using `--image-detail low` saves about 15% on the total.

## Keyboard Remapping

If your controller has buttons in weird places, remap them:

```bash
# Try it first
gpagent monitor --map BTN_NORTH=X --map BTN_WEST=Y

# Make it permanent in ~/.config/gpagent/gpagent.toml
[gamepad.button_map]
BTN_NORTH = "X"
BTN_WEST = "Y"

# Or just for one controller (by USB ID)
[gamepad.device_button_map."2dc8:200f"]
BTN_NORTH = "X"
BTN_WEST = "Y"
```

Run `gpagent devices` to find your controller's USB ID.

## Manual Audio Conversion

Speech is stored as raw PCM (16-bit, 24 kHz, mono). To convert to WAV with ffmpeg:

```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i session/blobs/000042-speech.pcm output.wav
```

Or use the built-in:
```bash
gpagent inspect session --wav   # converts everything to session/wav/
```

## Architecture

Two independent pieces:

1. **Capture** — discretizes gamepad, microphone, and screen into a sparse event stream (only informative moments, not raw video/audio). Runs VAD to gate speech, frame triggers to gate images.

2. **Agent** — reads the event stream and uses gpt-realtime-2.1 to generate commentary. Has a "speak policy" that decides when to chime in, based on scene changes, input bursts, and periods of quiet.

The split saves money: raw video/audio to a realtime API would cost more than the game. By emitting only moments that matter, gpagent keeps the bill manageable.

## More Info

- `gpagent --help` — full command reference
- `CLAUDE.md` — architecture docs (for developers)
