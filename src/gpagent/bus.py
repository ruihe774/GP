"""CaptureBus — merges independent sources into one ordered event stream.

Sources are callback-driven (GStreamer streaming threads, evdev reader threads),
so `emit` and `signal` are safe to call from any thread: they marshal onto the
event loop while stamping `t`/`seq` at *capture* time, not at delivery time.

The bus also carries a small signal hub so sources can hint at each other
(gamepad intensity and speech starts drive screen-frame triggers) without any
source importing another.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any, Protocol

from .events import Event, ScreenFrame

log = logging.getLogger(__name__)

#: events safe to drop under backpressure, newest-information-wins
DROPPABLE = (ScreenFrame,)


class CaptureSource(Protocol):
    """A capture source. Owns its own threads/pipelines; the bus only merges."""

    name: str

    async def start(self, ctx: BusContext) -> None: ...
    async def stop(self) -> None: ...
    def describe(self) -> dict[str, Any]: ...


class BusContext:
    """Handed to each source: how to emit events and exchange hints."""

    def __init__(self, bus: CaptureBus) -> None:
        self._bus = bus

    @property
    def t0(self) -> float:
        return self._bus.t0

    def emit(self, event: Event) -> None:
        self._bus.emit(event)

    def signal(self, name: str, payload: Any = None) -> None:
        self._bus.signal(name, payload)

    def on_signal(self, name: str, callback: Callable[[Any], None]) -> None:
        self._bus.on_signal(name, callback)


class CaptureBus:
    def __init__(
        self,
        sources: Iterable[CaptureSource],
        *,
        maxsize: int = 256,
    ) -> None:
        self.sources = list(sources)
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=maxsize)
        self._seq = itertools.count()
        self._subs: dict[str, list[Callable[[Any], None]]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self._started = False
        self._closed = False
        self.t0: float = 0.0
        self.dropped = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> BusContext:
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self.t0 = time.monotonic()
        ctx = BusContext(self)
        started: list[CaptureSource] = []
        try:
            for src in self.sources:
                await src.start(ctx)
                started.append(src)
        except Exception:
            for src in reversed(started):
                await _safe_stop(src)
            raise
        self._started = True
        return ctx

    async def stop(self) -> None:
        for src in reversed(self.sources):
            await _safe_stop(src)
        self._closed = True
        if self._loop is not None:
            self._queue.put_nowait(None)

    async def __aenter__(self) -> CaptureBus:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # -- emission ----------------------------------------------------------

    def emit(self, event: Event) -> None:
        """Stamp and enqueue. Safe from any thread."""
        if self._closed:
            return
        event.t = time.monotonic() - self.t0
        event.seq = next(self._seq)
        if threading.get_ident() == self._loop_thread:
            self._put(event)
        else:
            assert self._loop is not None
            self._loop.call_soon_threadsafe(self._put, event)

    def _put(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Under pressure, prefer losing a stale frame over stalling capture.
            if isinstance(event, DROPPABLE):
                self.dropped += 1
                log.warning("bus full, dropped %s seq=%d", event.TYPE, event.seq)
                return
            evicted = self._evict_droppable()
            if evicted:
                self.dropped += 1
                self._queue.put_nowait(event)
            else:
                self.dropped += 1
                log.error("bus full, dropped %s seq=%d", event.TYPE, event.seq)

    def _evict_droppable(self) -> bool:
        """Remove the oldest droppable event to make room. Loop thread only."""
        kept: list[Event | None] = []
        found = False
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if not found and isinstance(item, DROPPABLE):
                found = True
                continue
            kept.append(item)
        for item in kept:
            self._queue.put_nowait(item)
        return found

    # -- signals -----------------------------------------------------------

    def signal(self, name: str, payload: Any = None) -> None:
        if threading.get_ident() == self._loop_thread:
            self._dispatch(name, payload)
        elif self._loop is not None:
            self._loop.call_soon_threadsafe(self._dispatch, name, payload)

    def _dispatch(self, name: str, payload: Any) -> None:
        for cb in self._subs.get(name, ()):
            try:
                cb(payload)
            except Exception:
                log.exception("signal handler failed: %s", name)

    def on_signal(self, name: str, callback: Callable[[Any], None]) -> None:
        self._subs[name].append(callback)

    # -- consumption -------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[Event]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    async def drain(self) -> list[Event]:
        """Pull everything currently queued without waiting."""
        out: list[Event] = []
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item is not None:
                out.append(item)
        return out

    def describe(self) -> dict[str, Any]:
        return {src.name: src.describe() for src in self.sources}


async def _safe_stop(src: CaptureSource) -> None:
    try:
        await src.stop()
    except Exception:
        log.exception("error stopping source %s", getattr(src, "name", src))
