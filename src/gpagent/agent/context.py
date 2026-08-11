"""What the model gets told, and when.

Gamepad summaries are *accumulated here and flushed on demand*, not streamed to
the API as they arrive. Sending each one as its own conversation item would turn
151 seconds of `sess5` into 91 turns, wreck the turn structure, and bill for
every window whether or not the agent ever had a reason to speak. Flushing at
speak time means a summary is only paid for if it was part of the context for
something actually said.

The same rule governs frames, and matters more because frames dominate the bill:
**at most one full-detail image per response, and only if it is newer than the
last one sent.** That decouples the capture rate from the send rate entirely --
Component A can trigger frames as eagerly as it likes and the agent still pays
for one high-detail frame per utterance at most.

Behind that current frame rides a *trail*: up to `image_trail` earlier frames at
`image_trail_detail` ("low", a flat 85 tokens against ~765). One screenshot says
where the player is and nothing about where they came from, which is most of what
there is to react to. The frames between two responses are captured and otherwise
thrown away, so the trail is paid for in tokens only -- see `_pick_trail` for
which ones are worth a slot.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..events import ScreenFrame
from .config import AgentConfig

__all__ = ["Frame", "TurnContext", "ContextBuffer"]


@dataclass(frozen=True)
class Frame:
    seq: int
    t: float
    data: bytes
    w: int
    h: int
    trigger: str = ""
    #: how much the screen changed at capture time. Component A's own measure of
    #: whether a frame carried news; the trail selector spends its slots on it.
    scene_score: float = 0.0

    @classmethod
    def from_event(cls, event: ScreenFrame, t: float) -> Frame:
        """`t` is the *agent's* clock, not `event.t`.

        Live, `event.t` counts from the start of capture while the agent runs on
        `time.monotonic()`; mixing them makes every frame look hours stale and
        the agent silently blind.
        """
        return cls(
            seq=event.seq,
            t=t,
            data=event.data or b"",
            w=event.w,
            h=event.h,
            trigger=event.trigger,
            scene_score=event.scene_score,
        )


@dataclass
class TurnContext:
    """Everything non-audio that goes into one request."""

    text: str | None = None
    frame: Frame | None = None
    #: earlier frames, oldest first, to be sent at `image_trail_detail`
    trail: list[Frame] = field(default_factory=list)
    dropped_summaries: int = 0

    def __bool__(self) -> bool:
        return self.text is not None or self.frame is not None


#: ceiling on retained trail candidates, so a capture burst cannot grow the
#: buffer without bound. JPEGs at ~100 KB each; the age window usually binds first.
HISTORY_CAP = 64


class ContextBuffer:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._summaries: deque[tuple[float, str]] = deque()
        self._frame: Frame | None = None
        self._last_frame_sent: int | None = None
        #: trail candidates: every frame seen and not yet sent, newest last
        self._history: deque[Frame] = deque(maxlen=HISTORY_CAP)
        #: how far the conversation already covers. Everything up to here has
        #: been shown, so the trail only ever draws from the gap between it and
        #: the current frame -- see `_pick_trail`.
        self._sent_through_t = float("-inf")
        #: frames captured vs frames actually paid for
        self.frames_seen = 0
        self.frames_sent = 0
        self.trail_sent = 0
        self.summaries_seen = 0
        self.summaries_sent = 0

    # -- accumulate --------------------------------------------------------

    def add_summary(self, t: float, text: str) -> None:
        if not text:
            return
        self._summaries.append((t, text))
        self.summaries_seen += 1

    def add_frame(self, frame: Frame) -> None:
        # Newest wins *for the current screen*: an older frame is never a better
        # answer to "what is on screen now". It can still be the best answer to
        # "how did we get here", so it stays a trail candidate either way.
        if self._frame is None or frame.t >= self._frame.t:
            self._frame = frame
        if self.cfg.image_trail > 0:
            self._history.append(frame)
        self.frames_seen += 1

    # -- flush -------------------------------------------------------------

    def take(self, now: float, *, with_image: bool = True) -> TurnContext:
        """Consume everything worth sending with the next response."""
        self._expire(now)
        text, dropped = self._render()
        self._summaries.clear()

        frame = None
        trail: list[Frame] = []
        if with_image and self._frame is not None:
            fresh = now - self._frame.t <= self.cfg.max_image_age_s
            unsent = self._frame.seq != self._last_frame_sent
            if fresh and unsent and self._frame.data:
                frame = self._frame
                # Picked before the current frame is marked sent, so the
                # selector can exclude it by the same rule as everything else.
                trail = self._pick_trail(now, frame)
                self._last_frame_sent = frame.seq
                self.frames_sent += 1
                self.trail_sent += len(trail)
                # The current frame is the newest thing sent, so the whole
                # window behind it is now covered; nothing older is a candidate
                # again and nothing older needs keeping.
                self._sent_through_t = frame.t
                while self._history and self._history[0].t <= frame.t:
                    self._history.popleft()

        if text:
            self.summaries_sent += 1
        return TurnContext(text=text, frame=frame, trail=trail, dropped_summaries=dropped)

    def peek_frame(self) -> Frame | None:
        return self._frame

    # -- internals ---------------------------------------------------------

    def _expire(self, now: float) -> None:
        window = self.cfg.context_window_s
        while self._summaries and now - self._summaries[0][0] > window:
            self._summaries.popleft()

        horizon = self.cfg.image_trail_max_age_s
        while self._history and now - self._history[0].t > horizon:
            self._history.popleft()

    def _pick_trail(self, now: float, current: Frame) -> list[Frame]:
        """The `image_trail` most informative frames leading up to `current`.

        Two rules, and the order matters:

        **Spread first.** The four newest frames are the wrong four: capture
        triggers throttle rather than debounce, so a burst puts them inside a
        couple of seconds and the trail describes the last moment of the window
        four times instead of describing the window. The candidate span is split
        into `image_trail` equal time buckets and each bucket contributes at most
        one frame, so the trail is a trajectory by construction. Buckets with no
        candidate leave their slot to be backfilled, so a quiet stretch costs
        coverage rather than a slot.

        **Value within the bucket.** Which frame represents a bucket is decided
        by `scene_score` -- Component A already measured how much changed, and
        the frame that carried news is the one worth 85 tokens. Ties go to the
        newer frame.

        **Only the gap.** Candidates start *after* the newest frame already
        sent, not merely at "frames not sent individually". A frame from between
        two images the model is already looking at is nearly free of news, and
        spending a slot on it costs a slot covering the stretch nobody has seen;
        it would also drop an image into history older than one already there,
        against the order the accompanying note promises. The unseen trajectory
        is exactly `(last frame sent, current frame)`, and that is the window.
        """
        n = self.cfg.image_trail
        if n <= 0:
            return []

        cutoff = max(now - self.cfg.image_trail_max_age_s, self._sent_through_t)
        cands = sorted(
            (f for f in self._history if f.data and cutoff < f.t < current.t),
            key=lambda f: f.t,
        )
        if len(cands) <= n:
            return cands

        span = current.t - cands[0].t
        if span <= 0:  # every candidate shares a timestamp; value is all there is
            best = sorted(cands, key=lambda f: (f.scene_score, f.t))[-n:]
            return sorted(best, key=lambda f: f.t)

        buckets: dict[int, Frame] = {}
        for f in cands:
            i = min(n - 1, int((f.t - cands[0].t) / span * n))
            held = buckets.get(i)
            if held is None or (f.scene_score, f.t) > (held.scene_score, held.t):
                buckets[i] = f

        chosen = list(buckets.values())
        if len(chosen) < n:
            taken = {f.seq for f in chosen}
            spare = sorted(
                (f for f in cands if f.seq not in taken),
                key=lambda f: (f.scene_score, f.t),
                reverse=True,
            )
            chosen += spare[: n - len(chosen)]
        chosen.sort(key=lambda f: f.t)
        return chosen

    def _render(self) -> tuple[str | None, int]:
        if not self._summaries:
            return None, 0

        # Runs of the same summary collapse: "tapped A" five times in a row is
        # one fact, not five, and reads better as one.
        runs: list[list] = []
        for _, text in self._summaries:
            if runs and runs[-1][0] == text:
                runs[-1][1] += 1
            else:
                runs.append([text, 1])
        parts = [text if n == 1 else f"{text} (x{n})" for text, n in runs]

        # Trim from the *oldest* end: the newest thing that happened is the
        # thing worth spending the character budget on.
        dropped = 0
        limit = self.cfg.max_summary_chars
        while len(_join(parts)) > limit and len(parts) > 1:
            parts.pop(0)
            dropped += 1
        rendered = _join(parts)
        if dropped:
            rendered = "... " + rendered
        return f"[controller] {rendered[:limit]}", dropped


def _join(parts: list[str]) -> str:
    return "; ".join(parts)
