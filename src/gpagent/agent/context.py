"""What the model gets told, and when.

Gamepad summaries are *accumulated here and flushed on demand*, not streamed to
the API as they arrive. Sending each one as its own conversation item would turn
151 seconds of `sess5` into 91 turns, wreck the turn structure, and bill for
every window whether or not the agent ever had a reason to speak. Flushing at
speak time means a summary is only paid for if it was part of the context for
something actually said.

The same rule governs frames, and matters more because frames dominate the bill:
**at most one image per response, and only if it is newer than the last one
sent.** That decouples the capture rate from the send rate entirely -- Component
A can trigger frames as eagerly as it likes and the agent still pays for one per
utterance at most.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

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
        )


@dataclass
class TurnContext:
    """Everything non-audio that goes into one request."""

    text: str | None = None
    frame: Frame | None = None
    dropped_summaries: int = 0

    def __bool__(self) -> bool:
        return self.text is not None or self.frame is not None


class ContextBuffer:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._summaries: deque[tuple[float, str]] = deque()
        self._frame: Frame | None = None
        self._last_frame_sent: int | None = None
        #: frames captured vs frames actually paid for
        self.frames_seen = 0
        self.frames_sent = 0
        self.summaries_seen = 0
        self.summaries_sent = 0

    # -- accumulate --------------------------------------------------------

    def add_summary(self, t: float, text: str) -> None:
        if not text:
            return
        self._summaries.append((t, text))
        self.summaries_seen += 1

    def add_frame(self, frame: Frame) -> None:
        # Newest wins: an older frame is never more informative than a newer one.
        if self._frame is None or frame.t >= self._frame.t:
            self._frame = frame
        self.frames_seen += 1

    # -- flush -------------------------------------------------------------

    def take(self, now: float, *, with_image: bool = True) -> TurnContext:
        """Consume everything worth sending with the next response."""
        self._expire(now)
        text, dropped = self._render()
        self._summaries.clear()

        frame = None
        if with_image and self._frame is not None:
            fresh = now - self._frame.t <= self.cfg.max_image_age_s
            unsent = self._frame.seq != self._last_frame_sent
            if fresh and unsent and self._frame.data:
                frame = self._frame
                self._last_frame_sent = frame.seq
                self.frames_sent += 1

        if text:
            self.summaries_sent += 1
        return TurnContext(text=text, frame=frame, dropped_summaries=dropped)

    def peek_frame(self) -> Frame | None:
        return self._frame

    # -- internals ---------------------------------------------------------

    def _expire(self, now: float) -> None:
        window = self.cfg.context_window_s
        while self._summaries and now - self._summaries[0][0] > window:
            self._summaries.popleft()

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
