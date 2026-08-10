"""Session writer: events.jsonl + blobs/ + manifest.json.

Blobs sit beside a JSONL index so payloads stay inspectable with ordinary tools
(`jq`, `ffplay`, an image viewer). `inline=True` base64s them into the JSONL
instead, producing one self-contained file -- the convenient shape for replaying
a session into Component B.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from ..events import Event, decode

log = logging.getLogger(__name__)

__all__ = ["JsonlSink", "read_session", "session_dir_name"]

EVENTS_FILE = "events.jsonl"
MANIFEST_FILE = "manifest.json"
BLOBS_DIR = "blobs"


def session_dir_name(when: datetime | None = None) -> str:
    when = when or datetime.now()
    return "session-" + when.strftime("%Y-%m-%dT%H-%M-%S")


class JsonlSink:
    def __init__(self, directory: str | Path, *, inline: bool = False):
        self.directory = Path(directory)
        self.inline = inline
        self._events: IO[str] | None = None
        self._blobs = self.directory / BLOBS_DIR
        self.counts: dict[str, int] = {}
        self.bytes_written = 0

    def __enter__(self) -> JsonlSink:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.inline:
            self._blobs.mkdir(exist_ok=True)
        self._events = open(self.directory / EVENTS_FILE, "w", buffering=1)

    def write(self, event: Event) -> None:
        if self._events is None:
            raise RuntimeError("sink is not open")
        if event.data is not None and not self.inline:
            event.blob = self._write_blob(event)
        line = json.dumps(event.to_dict(inline=self.inline), separators=(",", ":"))
        self._events.write(line + "\n")
        self.counts[event.TYPE] = self.counts.get(event.TYPE, 0) + 1

    def _write_blob(self, event: Event) -> str:
        assert event.data is not None, "caller only writes blobs for events carrying data"
        ext = event.BLOB_EXT or "bin"
        name = f"{event.seq:06d}-{event.BLOB_KIND}.{ext}"
        path = self._blobs / name
        with open(path, "wb") as fh:
            fh.write(event.data)
        self.bytes_written += len(event.data)
        return f"{BLOBS_DIR}/{name}"

    def close(self, manifest: dict[str, Any] | None = None) -> None:
        if self._events is not None:
            self._events.close()
            self._events = None
        payload = {
            "written_at": datetime.now(UTC).isoformat(),
            "inline": self.inline,
            "counts": self.counts,
            "blob_bytes": self.bytes_written,
        }
        if manifest:
            payload.update(manifest)
        with open(self.directory / MANIFEST_FILE, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)


def read_session(directory: str | Path, *, load_blobs: bool = False) -> Iterator[Event]:
    """Yield events from a recorded session, oldest first."""
    directory = Path(directory)
    path = directory / EVENTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"no {EVENTS_FILE} in {directory}")

    def loader(rel: str) -> bytes:
        return (directory / rel).read_bytes()

    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield decode(json.loads(line), blob_loader=loader if load_blobs else None)
            except (ValueError, OSError) as exc:
                log.warning("%s:%d: skipping malformed event (%s)", path, lineno, exc)
