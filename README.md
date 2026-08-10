# GP — AI Gaming Buddy

GP watches you play games and comments on the action like a friend sitting next to you on the couch.

```bash
uv run gpagent commentate      # start watching and chatting live
```

It captures what you're doing (button presses, voice, screen) and feeds it to an AI that decides when to chime in.
Ask it questions, and it answers based on what's happening. Ignore it, and it gracefully gets quieter.
Interrupt it mid-sentence, and it stops talking.

## Requirements

- **Linux** desktop environment with PipeWire and Wayland or X11
- [uv](https://docs.astral.sh/uv/)
- A gamepad supporting evdev; this includes Xbox and Xbox-compatible controllers; keyboard and mouse are not supported yet
- A microphone and a speaker
- An OpenAI API key (set `OPENAI_API_KEY` or create `.env` at the repo root; see `.env.example`)

System packages (Ubuntu/Debian):
```bash
sudo apt install gstreamer1.0-pipewire gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad python3-gi gir1.2-gst-plugins-base-1.0
```

## Install

```bash
uv venv --system-site-packages
uv pip install -e .     # append [dev] for development

# Download the voice activity detection model (~50 MB)
curl -L -o models/silero_vad.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
```

## Quick Start

In the project dir:

**Check your setup:**
```bash
uv run gpagent devices          # lists gamepad, mic, screen — shows if they're ready
```

**Start the agent:**
```bash
# ensure you have put OpenAI API key in .env
uv run gpagent commentate                    # capture and comment in real time
```

On first run, a dialog asks for screen permission.
You can pass `--pick-screen` to show the dialog in subsequent runs.

## Commands

| Command | What it does |
|---------|-------------|
| `devices` | Check gamepad, mic, screen — are they detected? |
| `monitor` | Live button press display, test your controller mapping |
| `record -o NAME` | Capture gamepad, voice, and screen for 60 seconds |
| `inspect NAME` | View the captured session timeline, stats, and estimated cost |
| `commentate --replay NAME` | Run the agent over a recorded session |
| `commentate` | Capture and comment live in one shot |

## Customization

See [gpagent.example.toml](gpagent.example.toml). Copy it, customize the settings, and pass the config file:
```bash
uv run gpagent commentate -c /path/to/gpagent.toml
```

E.g.:
```toml
[agent]
model = "gpt-realtime-2.1"  # use a stronger model
voice = "cedar"             # change the voice
```

## Recording Conversations

Save the whole exchange (your voice and the AI's):
```bash
gpagent commentate -o chat_session   # outputs chat_session/ with both sides
gpagent inspect chat_session --wav   # saves the full conversation
```

## Troubleshooting

**Controller not detected or mismapped?**
```bash
gpagent monitor           # press buttons, see them appear
```

If your controller has buttons in weird places, remap them. E.g.:

```bash
# Try it first
gpagent monitor --map BTN_NORTH=X --map BTN_WEST=Y
```

And make it permanent in the config file
```toml
[gamepad.button_map]
BTN_NORTH = "X"
BTN_WEST = "Y"
```

**No voice coming through?**
- Check your microphone is plugged in and selected as the system default.

**Agent won't stop talking?**
- It's probably replying to itself. Check that audio output is going to your speakers (default sink), not to a USB device that's also your microphone.

## Cost

GP uses OpenAI's realtime API endpoint. The models `gpt-realtime-2.1` and `gpt-realtime-2.1-mini` can be used.
Each minute of commentary costs roughly $0.015–$0.02 (at current OpenAI prices). Most of that is the agent's voice output, not the video or audio input.
Inputs are throttled at the client side so frequent requests are avoided.
Check OpenAI's doc and API dashboard for detailed usage.
