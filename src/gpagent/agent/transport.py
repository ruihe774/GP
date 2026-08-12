"""Talking to the Realtime API -- and pretending to, for development.

Everything above this layer speaks in plain dicts, exactly as the wire protocol
documents them. That is worth the one `model_dump()` per received event: the
agent's tests need a two-line fake, not a pydantic fixture factory, and a server
field the installed SDK has never heard of still reaches the agent.

Three implementations:

    OpenAITransport     the real socket
    RecordingTransport  writes what would have been sent, and answers itself
                        with a synthetic response so the whole loop still runs
    FakeTransport       tests

One hazard worth naming: `AsyncRealtimeConnection.send()` routes a dict through
`async_maybe_transform(event, RealtimeClientEventParam)`, which silently drops
keys the installed SDK's TypedDicts do not know about. Every payload this
package sends is asserted to survive that transform intact in
`tests/test_transport_payloads.py`, so an SDK upgrade that starts eating a field
fails a test instead of quietly degrading the agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, cast

from ..tokens import audio_tokens, image_tokens, text_tokens

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.resources.realtime.realtime import (
        AsyncRealtimeConnection,
        AsyncRealtimeConnectionManager,
    )

log = logging.getLogger(__name__)

__all__ = ["Transport", "OpenAITransport", "RecordingTransport", "FakeTransport"]


class Transport(Protocol):
    async def connect(self) -> None: ...
    async def send(self, event: dict) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self) -> AsyncIterator[dict]: ...


class OpenAITransport:
    """The real thing, over the official SDK's websocket connection."""

    name = "openai"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None
        self._manager: AsyncRealtimeConnectionManager | None = None
        self._conn: AsyncRealtimeConnection | None = None
        self.sent = 0
        self.received = 0

    def __repr__(self) -> str:
        # Never let the key reach a traceback via this object's locals.
        return f"OpenAITransport(model={self.model!r})"

    async def connect(self) -> None:
        from openai import AsyncOpenAI

        from .env import Secret

        key = self._api_key
        self._client = AsyncOpenAI(
            api_key=key.reveal() if isinstance(key, Secret) else key
        )
        self._manager = self._client.realtime.connect(model=self.model)
        self._conn = await self._manager.__aenter__()
        log.info("realtime session open (%s)", self.model)

    async def send(self, event: dict) -> None:
        if self._conn is None:
            raise RuntimeError("transport is not connected")
        # Plain dicts by design (see module docstring): the SDK's
        # `async_maybe_transform` validates and coerces at runtime, and
        # `tests/test_transport_payloads.py` asserts every payload survives it.
        await self._conn.send(cast(Any, event))
        self.sent += 1

    async def close(self) -> None:
        if self._manager is not None:
            try:
                await self._manager.__aexit__(None, None, None)
            except Exception:  # a socket already gone is not an error worth raising
                log.debug("realtime close failed", exc_info=True)
            self._manager = None
            self._conn = None
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aiter__(self) -> AsyncIterator[dict]:
        if self._conn is None:
            raise RuntimeError("transport is not connected")
        async for event in self._conn:
            self.received += 1
            yield _as_dict(event)


def _as_dict(event: Any) -> dict:
    dump = getattr(event, "model_dump", None)
    return dump(exclude_none=True) if dump is not None else dict(event)


class _QueueTransport:
    """Shared plumbing for the transports that answer from a local queue."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._incoming: asyncio.Queue[dict | None] = asyncio.Queue()
        self._closed = False

    async def connect(self) -> None:
        return None

    async def send(self, event: dict) -> None:
        self.sent.append(event)

    def feed(self, event: dict) -> None:
        """Push a server event to the consumer."""
        self._incoming.put_nowait(event)

    async def close(self) -> None:
        self._closed = True
        self._incoming.put_nowait(None)

    async def __aiter__(self) -> AsyncIterator[dict]:
        while True:
            event = await self._incoming.get()
            if event is None:
                break
            yield event

    def pending(self) -> int:
        """Server events queued but not yet consumed. See `agent.settle()`."""
        return self._incoming.qsize()

    def sent_of_type(self, type_name: str) -> list[dict]:
        return [e for e in self.sent if e.get("type") == type_name]


class FakeTransport(_QueueTransport):
    """A transport for tests: records what was sent, replays what you feed it."""

    name = "fake"


class RecordingTransport(_QueueTransport):
    """Dry run: never connects, but keeps the whole agent loop honest.

    It answers each `response.create` with a synthetic `response.created` /
    transcript / `response.done`, including a usage block estimated from what
    was actually sent. That means the policy's response lifecycle, the barge-in
    path and the cost meter are all exercised offline -- a dry run that just
    dropped requests on the floor would leave the agent permanently waiting for
    a response that never lands, which is the exact failure it is meant to catch.

    The usage block bills the *whole conversation*, not the turn, because that
    is what the API does and it is the only part of the bill anyone is trying to
    control. A ledger of live items is kept, `conversation.item.delete` removes
    from it, and everything still in it is charged again on the next response --
    so a dry run shows what `agent.prune_after_s` does to the growth curve, and
    to the TPM ceiling that curve runs into. Cached tokens are modelled the same
    way, one rule: a delete truncates the cached prefix at the deleted item.

    It is a model of the server, not the server. Sizes come from `tokens.py`,
    the per-item overhead the API adds is not counted, and a real session's
    cache can miss for reasons nothing here knows about. Read the shape of the
    curve off it, not the fourth digit.
    """

    name = "dry-run"

    #: what a short spoken reply costs in output audio tokens (~2.5 s at 50 ms/tok)
    ASSUMED_REPLY_TOKENS = 50
    #: ...and what the same remark costs written down, which is where the two
    #: modes stop being comparable: text output is roughly an order of
    #: magnitude cheaper per response than speaking the same words.
    ASSUMED_TEXT_TOKENS = 20

    DRY_REMARK = "(dry run: no model was asked)"

    def __init__(self, path: str | Path | None = None, *, text: bool = False):
        super().__init__()
        self.path = Path(path) if path else None
        #: answer the way the session asked to be answered -- a text session
        #: that got audio events back would exercise a path it will never take
        self.text = text
        self._fh: IO[str] | None = None
        #: everything currently in the conversation, in order, by item id --
        #: this is what gets re-billed every response
        self._ledger: dict[str, _InputTally] = {}
        #: audio appended but not yet committed into an item
        self._pending_input = _InputTally()
        #: tokens the next response may charge at the cached rate: what was in
        #: the conversation last time, capped by the earliest delete since
        self._cached = 0
        self._cache_ceiling = _NO_CEILING
        self._responses = 0

    async def connect(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", buffering=1)

    async def close(self) -> None:
        await super().close()
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    async def send(self, event: dict) -> None:
        await super().send(event)
        self._record(event)
        self._tally(event)
        if event.get("type") == "response.create":
            self._answer(event)

    # -- internals ---------------------------------------------------------

    def _record(self, event: dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(_redact(event), separators=(",", ":")) + "\n")

    def _tally(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "session.update":
            # The persona is the one thing charged on every single request and
            # never deleted, so it belongs at the head of the ledger.
            text = ((event.get("session") or {}).get("instructions")) or ""
            self._ledger["dry-instructions"] = _InputTally(text=text_tokens(text))
        elif kind == "input_audio_buffer.append":
            # base64 -> bytes -> samples -> ms, at 24 kHz mono s16
            b64 = event.get("audio") or ""
            self._pending_input.audio_ms += len(b64) * 3 / 4 / 2 / 24.0
        elif kind == "input_audio_buffer.commit":
            # Committing turns the buffer into an item of its own. The server
            # names it, so nothing can delete it -- which is exactly true of the
            # real thing too, and part of what the growth curve is made of.
            self._ledger[f"dry-audio-{len(self._ledger)}"] = self._pending_input
            self._pending_input = _InputTally()
        elif kind == "conversation.item.create":
            item = event.get("item") or {}
            tally = _InputTally()
            for part in item.get("content") or []:
                if part.get("type") == "input_text":
                    tally.text += text_tokens(part.get("text") or "")
                elif part.get("type") == "input_image":
                    detail = part.get("detail", "high")
                    tally.image += image_tokens(1024, 576, detail=detail)
            self._ledger[item.get("id") or f"dry-item-{len(self._ledger)}"] = tally
        elif kind == "conversation.item.delete":
            self._delete(event.get("item_id") or "")

    def _delete(self, item_id: str) -> None:
        """Drop an item, and cap what the next request can still claim as cached.

        A prompt cache is a prefix match, so removing an item leaves only what
        was ahead of it cacheable. Deletions come from the oldest end, which is
        why a pruning round is close to a full invalidation and why doing them
        in bulk is the whole trick.
        """
        if item_id not in self._ledger:
            return
        prefix = 0
        for key, tally in self._ledger.items():
            if key == item_id:
                break
            prefix += tally.total()
        self._cache_ceiling = min(self._cache_ceiling, prefix)
        del self._ledger[item_id]

    def _answer(self, event: dict) -> None:
        self._responses += 1
        item_id = f"dry-item-{self._responses:04d}"
        # Everything still in the conversation is charged again, which is the
        # point of the whole ledger; only the part that survived unchanged since
        # the last response is charged at the cached rate.
        billed = sum(tally.total() for tally in self._ledger.values())
        cached = min(self._cached, self._cache_ceiling, billed)
        tally = _InputTally(
            text=sum(t.text for t in self._ledger.values()),
            image=sum(t.image for t in self._ledger.values()),
            audio_ms=sum(t.audio_ms for t in self._ledger.values()),
        )
        self.feed({"type": "response.created", "response": {"id": f"dry-{self._responses}"}})
        self.feed(
            {
                "type": "response.output_item.added",
                "item": {"id": item_id, "type": "message", "role": "assistant"},
            }
        )
        if self.text:
            self.feed(
                {
                    "type": "response.output_text.done",
                    "item_id": item_id,
                    "text": self.DRY_REMARK,
                }
            )
        else:
            self.feed(
                {
                    "type": "response.output_audio_transcript.done",
                    "item_id": item_id,
                    "transcript": self.DRY_REMARK,
                }
            )
        out_audio = 0 if self.text else self.ASSUMED_REPLY_TOKENS
        out_text = self.ASSUMED_TEXT_TOKENS if self.text else 0
        # The answer joins the conversation and is re-billed with the rest of
        # it from here on, exactly like the items that prompted it.
        self._ledger[item_id] = _InputTally(text=out_audio + out_text)
        self._cached = billed + out_audio + out_text
        self._cache_ceiling = _NO_CEILING
        self.feed(
            {
                "type": "response.done",
                "response": {
                    "id": f"dry-{self._responses}",
                    "status": "completed",
                    "usage": {
                        "input_tokens": tally.total(),
                        "input_token_details": {
                            "text_tokens": tally.text,
                            "audio_tokens": audio_tokens(tally.audio_ms),
                            "image_tokens": tally.image,
                            "cached_tokens": cached,
                        },
                        "output_tokens": out_audio + out_text,
                        "output_token_details": {
                            "text_tokens": out_text,
                            "audio_tokens": out_audio,
                        },
                    },
                },
            }
        )


#: stand-in for "no delete has capped the cache yet"
_NO_CEILING = 1 << 62


class _InputTally:
    def __init__(self, text: int = 0, image: int = 0, audio_ms: float = 0.0) -> None:
        self.text = text
        self.image = image
        self.audio_ms = audio_ms

    def total(self) -> int:
        return self.text + self.image + audio_tokens(self.audio_ms)


def _redact(event: dict) -> dict:
    """Keep the log readable: bulk payloads become their size, not their bytes."""
    out = dict(event)
    if isinstance(out.get("audio"), str):
        out["audio"] = f"<{len(out['audio']) * 3 // 4} bytes pcm>"
    item = out.get("item")
    if isinstance(item, dict) and isinstance(item.get("content"), list):
        item = dict(item)
        content = []
        for part in item["content"]:
            part = dict(part)
            url = part.get("image_url")
            if isinstance(url, str):
                part["image_url"] = f"<{len(url) * 3 // 4} bytes jpeg>"
            content.append(part)
        item["content"] = content
        out["item"] = item
    return out
