"""The film's subtitle file, read once and sent once.

This is Component B input that never came through capture: a `.srt` on disk,
handed to the agent at the start of the session and sent whole, as a single
system item, immediately after `session.update`.

Why the whole file rather than a cue at a time. The agent cannot hear the film
-- the microphone is echo-cancelled against the system sink precisely so that
dialogue does not read as the player talking (see `movies.example.toml`), so
without this it is watching a silent film with subtitles it can only sometimes
read off a screenshot. Feeding cues as they land would fix that and cost far
more: a per-turn item is a per-turn *change* at the tail of the conversation,
re-billed as fresh input on every request after it, whereas one item sent before
the first turn sits in the cached prefix for the whole session and is billed at
the cached rate forever after. A two-hour film is ~8-10k tokens of dialogue,
which is one turn's worth of screenshots at movie settings.

The cost of that is the agent knowing how the film ends. That is a prompt
problem rather than a plumbing one: `persona.SUBTITLE_NOTE` rides with the
script and says, in the item itself, that the part they have not reached is not
his to use, and the movies persona already forbids spoilers outright.

Which leaves the question the prompt cannot answer on its own -- *where are
they now* -- and the honest answer is that nothing here knows. The only position
available is `offset_s` plus elapsed session time, and that is playback position
only for a film that never pauses, never gets rewound over a line, and never
sits paused through a phone call. So the note is off by default
(`SubtitleConfig.position_note`), and the script is instead sent with an
instruction to locate the current scene in it from what is on screen and what is
being said. That is evidence, and evidence degrades gracefully; a clock that has
silently drifted forward reads exactly as confident as one that has not, and the
direction it drifts is towards dialogue that has not happened yet.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..tokens import text_tokens
from .config import SubtitleConfig

log = logging.getLogger(__name__)

__all__ = ["Cue", "Script", "parse_srt", "load_script", "render", "clock"]

#: `00:01:02,345` and the `.345` variant some tools write
_TIME = re.compile(r"(\d+):([0-5]\d):([0-5]\d)[,.](\d{1,3})")
_ARROW = "-->"
#: `<i>`, `</font>`, and the `{\an8}` positioning some rips carry over from ASS
_MARKUP = re.compile(r"<[^>]*>|\{\\[^}]*\}")
#: a cue that is entirely `[DOOR SLAMS]`, `(sighs)` or `♪ ... ♪`
_EFFECT = re.compile(r"^\s*(?:[\[(].*[\])]|[♪#].*)\s*$", re.DOTALL)


@dataclass(frozen=True)
class Cue:
    index: int
    #: seconds into the film, as written in the file
    start: float
    end: float
    text: str


@dataclass
class Script:
    """A parsed subtitle file, and the block of text that gets sent."""

    path: str
    cues: list[Cue] = field(default_factory=list)
    #: rendered exactly as it will be sent, so nothing measures a different string
    text: str = ""
    #: cues dropped as sound effects / music, when `skip_effects` is on
    skipped: int = 0

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def tokens(self) -> int:
        """Estimated, by the same rule `inspect` prices everything else with.

        `tokens.text_tokens` is characters over four, which is about right for
        the Latin scripts it was written for and reads low -- by roughly 3x --
        for a CJK subtitle track. It is a size to sanity-check a file against
        before sending it, not a bill; the bill comes back on `response.done`.
        """
        return text_tokens(self.text)

    @property
    def runtime_s(self) -> float:
        return self.cues[-1].end if self.cues else 0.0

    def __bool__(self) -> bool:
        return bool(self.cues)


def clock(seconds: float) -> str:
    """`HH:MM:SS`, the format the script's own timestamps are written in."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def parse_srt(text: str) -> list[Cue]:
    """Cues from SRT text, in film order.

    Deliberately forgiving, because subtitle files in the wild are: the index
    line is optional, the decimal separator is either `,` or `.`, timestamp
    lines can carry trailing coordinates, and blocks that parse as nothing are
    skipped rather than raising. A file that yields no cues at all is the only
    real failure, and the caller reports that.
    """
    cues: list[Cue] = []
    normalized = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n[ \t]*\n", normalized):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        index: int | None = None
        if _ARROW not in lines[0]:
            head = lines.pop(0).strip()
            index = int(head) if head.isdigit() else None
        if not lines or _ARROW not in lines[0]:
            continue
        stamps = _TIME.findall(lines[0])
        if len(stamps) < 2:
            continue
        start, end = _seconds(stamps[0]), _seconds(stamps[1])
        body = _clean(" ".join(lines[1:]))
        if not body:
            continue
        cues.append(
            Cue(
                index=index if index is not None else len(cues) + 1,
                start=start,
                # A negative-length cue is a typo in the file, not a fact.
                end=max(end, start),
                text=body,
            )
        )
    cues.sort(key=lambda c: (c.start, c.index))
    return cues


def render(cues: list[Cue], *, timestamps: bool = True) -> str:
    """The block of text that is actually sent.

    Timestamps are ~4 tokens a line, about a quarter of the script, and they buy
    the only thing that makes reading ahead avoidable: without them "where they
    have got to" cannot be expressed in terms the model can locate in the file.
    """
    if timestamps:
        return "\n".join(f"{clock(c.start)}  {c.text}" for c in cues)
    return "\n".join(c.text for c in cues)


def load_script(cfg: SubtitleConfig) -> Script | None:
    """Read and render the configured file, or None if there is nothing to read.

    Raises `ValueError` for a file that exists but yields no cues -- an SRT that
    parsed as nothing is a wrong path or a broken download, and a session that
    silently proceeds without the dialogue is the failure this feature exists to
    prevent.
    """
    if not cfg.enabled or not cfg.path:
        return None
    path = Path(cfg.path).expanduser()
    raw = path.read_bytes()
    text = _decode(raw, cfg.encoding)
    cues = parse_srt(text)
    if not cues:
        raise ValueError(f"no subtitle cues found in {path}")

    skipped = 0
    if cfg.skip_effects:
        kept = [c for c in cues if not _EFFECT.match(c.text)]
        skipped = len(cues) - len(kept)
        cues = kept or cues  # an all-effects file is more likely a bad rule

    return Script(
        path=str(path),
        cues=cues,
        text=render(cues, timestamps=cfg.timestamps),
        skipped=skipped,
    )


# -- internals -------------------------------------------------------------


def _seconds(stamp: tuple[str, str, str, str]) -> float:
    h, m, s, frac = stamp
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000.0


def _clean(text: str) -> str:
    return " ".join(_MARKUP.sub("", text).split())


def _decode(raw: bytes, encoding: str) -> str:
    """Subtitle files are frequently not UTF-8 and never say so.

    cp1252 is the common one and, being a single-byte encoding, always
    "succeeds" -- so it goes after UTF-8 rather than being tried first.
    """
    candidates = [encoding] if encoding else ["utf-8-sig", "cp1252"]
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    log.warning("could not decode subtitles cleanly; replacing bad bytes")
    return raw.decode("utf-8", "replace")
