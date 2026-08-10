"""Playing the agent's voice.

**Play to the default sink, and name no device.** Component A's echo canceller
uses the default sink's monitor as its AEC reference, which is what keeps the
agent's own voice out of the microphone: a synthetic voice through the speakers
reached the mic at -40.9 dBFS and was held to a VAD score of 0.178 against a 0.5
threshold, producing zero false segments. Target a specific sink and that
guarantee is gone -- the agent hears itself, replies to itself, and does so
forever.

The requirement is the *routing*, not a particular element. `autoaudiosink`
leaves the device unset, so WirePlumber puts the stream on the default sink and
re-routes it when the default changes (verified: the stream links to the default
sink node with `target.object` unset).

`pipewiresink` was the obvious choice and turned out to be the wrong one. It is a
plain `GstBaseSink` pushing into a PipeWire stream, ranked 0 by GStreamer, and it
reproducibly garbled speech here even though the buffers reaching it were
sample-for-sample correct. `autoaudiosink` selects `pulsesink`, a
`GstAudioBaseSink` with a real ring buffer -- built for exactly this: clock
drift, silence on underrun, and backpressure upstream. Same bytes, same
timestamps, clean playback.

Timing is accounted in `PlaybackTimer`, separately from GStreamer and with an
injectable clock, because barge-in depends on it: `conversation.item.truncate`
needs to know how much of a sentence the player actually heard, and that is not
something worth only being able to test with a sound card attached.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "PlaybackTimer",
    "AudioPlayer",
    "NullPlayer",
    "make_player",
    "align_samples",
]

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
#: How far ahead of the pipeline clock a new utterance is scheduled: both a
#: safety margin so the first buffer is not already late when it reaches the
#: sink, and a jitter buffer. The model usually generates faster than real time,
#: but not always; whenever playback catches up to the data, the next buffer has
#: to be re-stamped into the future and the difference is an audible gap in the
#: middle of a word. Cheaper to start a third of a second later and not stutter.
LEAD_IN_MS = 350


def _ms_of(pcm_bytes: int) -> float:
    return pcm_bytes / BYTES_PER_SAMPLE / (SAMPLE_RATE / 1000.0)


def align_samples(carry: bytes, chunk: bytes) -> tuple[bytes, bytes]:
    """Split a stream of arbitrary byte runs into whole 16-bit samples.

    `response.output_audio.delta` chunks are base64 of an arbitrary byte count,
    and nothing promises an even one. A single odd-length chunk pushed as-is
    shifts every following sample by one byte, and the rest of the utterance
    comes out as white noise -- the failure sounds like a broken codec rather
    than an off-by-one, which is why it is worth a named function and a test.

    Returns (whole samples to play, bytes to carry into the next chunk).
    """
    data = carry + chunk
    usable = len(data) - (len(data) % BYTES_PER_SAMPLE)
    return data[:usable], data[usable:]


class PlaybackTimer:
    """How much of what we queued has actually come out of the speakers.

    Wall-clock accounting against the amount pushed, which is accurate to within
    the sink's latency (tens of ms) -- far inside what a truncate needs, and it
    survives flushes cleanly, which querying the pipeline position does not.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._started: float | None = None
        self._pushed_ms = 0.0
        self.total_spoken_ms = 0.0

    def push(self, pcm_bytes: int, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        if self._started is None:
            self._started = now
        self._pushed_ms += _ms_of(pcm_bytes)

    def played_ms(self, now: float | None = None) -> float:
        if self._started is None:
            return 0.0
        now = self.clock() if now is None else now
        return min(self._pushed_ms, max(0.0, (now - self._started) * 1000.0))

    def remaining_ms(self, now: float | None = None) -> float:
        return max(0.0, self._pushed_ms - self.played_ms(now))

    @property
    def pushed_ms(self) -> float:
        return self._pushed_ms

    @property
    def active(self) -> bool:
        return self._started is not None

    def reset(self, now: float | None = None) -> float:
        """Cut the utterance short. Returns how much of it was heard."""
        return self._end(self.played_ms(now))

    def discard(self) -> None:
        """Drop what is still on the clock without counting it as spoken.

        Only the offline path needs this. `_finish_response` completes the
        utterance and then, for a non-realtime player, calls `simulate()` to
        re-arm the timer so the agent can still tell it is mid-sentence if the
        player interrupts. Nothing cleared that before the next response, so
        the next response's audio accumulated on top of it and a barge-in
        truncated with a `heard_ms` spanning two utterances against only the
        latest item id -- the server answers "Audio content of 7650ms is
        already shorter than 12970ms" and the truncate is lost.
        """
        self._started = None
        self._pushed_ms = 0.0

    def complete(self) -> float:
        """End the utterance normally: all of it was heard."""
        return self._end(self._pushed_ms)

    def _end(self, heard: float) -> float:
        self.total_spoken_ms += heard
        self._started = None
        self._pushed_ms = 0.0
        return heard


class NullPlayer:
    """Accounts for audio without making any. Dry runs, CI, `--no-playback`."""

    name = "null"
    #: nothing here happens in real time, so callers must not wait on it
    realtime = False

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.timer = PlaybackTimer(clock)
        self.bytes_played = 0

    async def start(self) -> None:
        return None

    def push(self, pcm: bytes, now: float | None = None) -> None:
        self.bytes_played += len(pcm)
        self.timer.push(len(pcm), now)

    def simulate(self, duration_ms: float, now: float | None = None) -> None:
        """Pretend an utterance of this length started now.

        Offline there are no audio bytes to push, but the agent still has to
        model a sentence taking time -- otherwise a replay could never catch
        itself talking over the player, and barge-in would be untestable
        without a sound card.
        """
        if duration_ms <= 0:
            return
        self.timer.push(int(duration_ms * SAMPLE_RATE / 1000.0) * BYTES_PER_SAMPLE, now)

    def flush(self, now: float | None = None) -> float:
        return self.timer.reset(now)

    def discard(self) -> None:
        self.timer.discard()

    async def drain(self) -> float:
        # No waiting: a dry run at speed 0 must not spend real seconds
        # pretending to speak. Everything queued counts as heard.
        return self.timer.complete()

    async def stop(self) -> None:
        return None


class AudioPlayer:
    """appsrc -> the default PipeWire sink."""

    name = "pipewire"
    #: audio really does take as long as it takes
    realtime = True

    # Timestamping is done by hand in `push`, and neither obvious alternative
    # works: `do-timestamp=true` stamps buffers as they *arrive*, which would
    # compress a 10 s reply into the second it took to download, and leaving
    # them unstamped in a `format=time` source leaves the sink to render them
    # back to back against a clock that has been running since startup, which
    # is heard as chunks and gaps.
    #: `volume` ships in gst-plugins-base next to `audioconvert`/`audioresample`,
    #: which this pipeline already requires unconditionally -- so it applies the
    #: gain in the pipeline itself rather than scaling PCM samples by hand.
    PIPELINE = (
        "appsrc name=src format=time is-live=false do-timestamp=false "
        "block=false max-bytes=16777216 "
        "caps=audio/x-raw,format=S16LE,rate={rate},channels=1,layout=interleaved "
        "! queue max-size-time=0 max-size-bytes=0 max-size-buffers=0 "
        "! audioconvert ! audioresample "
        "! volume name=vol volume={volume} "
        "! {sink}"
    )

    #: No device named: WirePlumber routes this to the *default* sink, which is
    #: the sink Component A's AEC references. Never add a device here.
    SINK = "autoaudiosink"

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        *,
        sink: str | None = None,
        volume: float = 1.0,
    ):
        #: overridable only so tests can render the timeline somewhere
        #: inspectable; production always uses the default sink
        self.sink = sink or self.SINK
        self.volume = volume
        self.timer = PlaybackTimer(clock)
        self._gst: Any = None
        self._pipeline: Any = None
        self._src: Any = None
        self._vol: Any = None
        #: next presentation timestamp, in pipeline running time
        self._pts: int | None = None
        #: trailing bytes of a chunk that did not end on a sample boundary
        self._carry = b""

    #: silence pushed at startup purely to let the sink preroll
    PRIME_MS = 20

    async def start(self) -> None:
        from ..capture.gst_util import drain_bus_errors, ensure_gst

        Gst = ensure_gst()
        self._gst = Gst
        self._pipeline = Gst.parse_launch(
            self.PIPELINE.format(rate=SAMPLE_RATE, sink=self.sink, volume=self.volume)
        )
        self._src = self._pipeline.get_by_name("src")
        self._vol = self._pipeline.get_by_name("vol")
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            drain_bus_errors(self._pipeline, "playback")
            raise RuntimeError("playback pipeline failed to start")

        # A sink cannot preroll with no data, and an appsrc has none until
        # someone speaks. Left unprimed the pipeline sits in an unfinished
        # async state change: `get_state` stalls for its full timeout, the
        # pipewire clock never starts (so buffer timestamps have nothing to
        # anchor to), and tearing the pipeline down blocks forever. Twenty
        # milliseconds of silence resolves all three.
        self._push_buffer(b"\x00" * (self.PRIME_MS * SAMPLE_RATE // 1000 * BYTES_PER_SAMPLE), 0)
        result, _, _ = self._pipeline.get_state(3 * Gst.SECOND)
        if result == Gst.StateChangeReturn.FAILURE:
            drain_bus_errors(self._pipeline, "playback")
            raise RuntimeError("playback pipeline failed to reach PLAYING")
        log.info("playback open: default pipewire sink")

    def _push_buffer(self, data: bytes, pts: int) -> int:
        """Hand one buffer to appsrc. Returns its duration in ns."""
        Gst = self._gst
        duration = Gst.SECOND * (len(data) // BYTES_PER_SAMPLE) // SAMPLE_RATE
        buffer = Gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)
        buffer.pts = pts
        buffer.duration = duration
        self._src.emit("push-buffer", buffer)
        return duration

    def push(self, pcm: bytes, now: float | None = None) -> None:
        if self._src is None or not pcm:
            return
        data, self._carry = align_samples(self._carry, pcm)
        if not data:
            return

        Gst = self._gst
        running = self._running_time()
        if self._pts is None or self._pts < running:
            # First buffer of an utterance, or the model fell behind real time
            # and our timeline is now in the past. Either way, restart just
            # ahead of the clock; buffers stamped in the past are dropped.
            self._pts = running + LEAD_IN_MS * Gst.MSECOND

        self._pts += self._push_buffer(data, self._pts)
        self.timer.push(len(data), now)

    def _running_time(self) -> int:
        """Where the sink's clock is now, in the same units as a buffer PTS."""
        clock = self._pipeline.get_clock() if self._pipeline is not None else None
        if clock is None:
            return 0
        return max(0, clock.get_time() - self._pipeline.get_base_time())

    def flush(self, now: float | None = None) -> float:
        """Stop mid-sentence. Returns the milliseconds the player heard."""
        heard = self.timer.reset(now)
        self._pts = None
        self._carry = b""
        if self._src is not None:
            Gst = self._gst
            self._src.send_event(Gst.Event.new_flush_start())
            self._src.send_event(Gst.Event.new_flush_stop(True))
        return heard

    def discard(self) -> None:
        # A realtime player is already clean here: `drain` completed the
        # utterance and nothing re-armed the timer. Defined so both players
        # answer the same calls.
        self.timer.discard()

    async def drain(self) -> float:
        """Wait for queued audio to finish playing, then close the utterance."""
        while self.timer.remaining_ms() > 0:
            await asyncio.sleep(min(0.25, self.timer.remaining_ms() / 1000.0))
        # A partial sample left over from the last chunk belongs to an utterance
        # that is now over; carrying it into the next one would prepend a byte
        # of garbage and shift the whole thing again.
        self._carry = b""
        self._pts = None
        return self.timer.complete()

    async def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
            self._src = None


def make_player(
    enabled: bool,
    clock: Callable[[], float] = time.monotonic,
    *,
    sink: str | None = None,
    volume: float = 1.0,
):
    """A real player when asked for and available, a null one otherwise."""
    if not enabled:
        return NullPlayer(clock)
    try:
        return AudioPlayer(clock, sink=sink, volume=volume)
    except Exception:
        log.warning("playback unavailable; running silent", exc_info=True)
        return NullPlayer(clock)
