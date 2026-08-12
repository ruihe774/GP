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
`tests/test_agent_parts.py::TestSdkPayloadDrift`, so an SDK upgrade that starts
eating a field fails a test instead of quietly degrading the agent.
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
        # `TestSdkPayloadDrift` asserts every payload survives it.
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
        # `close()` leaves a sentinel in the queue to end the iteration. A
        # reopen must not inherit it: the pump would read it as the new
        # connection dropping the instant it was made, and reconnect forever.
        self._closed = False
        keep = []
        while not self._incoming.empty():
            event = self._incoming.get_nowait()
            if event is not None:
                keep.append(event)
        for event in keep:
            self._incoming.put_nowait(event)

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
    from it, `previous_item_id` inserts into the middle of it, and everything
    still in it is charged again on the next response -- so a dry run shows what
    `agent.prune_after_s` does to the growth curve, and to the ceiling that curve
    runs into. Cached tokens are modelled the same way, one rule: a change to the
    conversation truncates the cached prefix at the position it happened.

    With `context_limit` set it also models the *end* of that curve: over the
    limit, a `response.create` is refused with an error instead of answered,
    which is what the API does once `truncation` is disabled and is the one
    path in the agent that no recording can stand in for.

    It is a model of the server, not the server. Sizes come from `tokens.py`,
    the per-item overhead the API adds is not counted, server-side truncation is
    not modelled at all (so a `retention_ratio` dry run is not a prediction of
    one), and a real session's cache can miss for reasons nothing here knows
    about. Read the shape of the curve off it, not the fourth digit.
    """

    name = "dry-run"

    #: what a short spoken reply costs in output audio tokens (~2.5 s at 50 ms/tok)
    ASSUMED_REPLY_TOKENS = 50
    #: ...and what the same remark costs written down, which is where the two
    #: modes stop being comparable: text output is roughly an order of
    #: magnitude cheaper per response than speaking the same words.
    ASSUMED_TEXT_TOKENS = 20

    DRY_REMARK = "(dry run: no model was asked)"
    #: what the player is pretended to have said, so the transcript-driven
    #: half of pruning has something to work with offline
    DRY_HEARD = "(dry run: whatever the player said)"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        text: bool = False,
        context_limit: int = 0,
    ):
        super().__init__()
        self.path = Path(path) if path else None
        #: answer the way the session asked to be answered -- a text session
        #: that got audio events back would exercise a path it will never take
        self.text = text
        #: post-instruction input tokens at which a response is refused rather
        #: than answered; 0 never refuses
        self.context_limit = context_limit
        #: whether the session asked for input transcription -- read off
        #: `session.update`, because it decides whether a committed utterance
        #: ever gets the transcript that lets the client replace it
        self._transcribe = False
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
        await super().connect()
        # A reopened session starts with an empty conversation: history does
        # not come back with it, and neither does the bill for history. Without
        # this a dry run that reconnects keeps charging for a conversation the
        # server has forgotten, which would make the last resort in
        # `session._on_context_full` look like it changed nothing.
        self._ledger = {}
        self._pending_input = _InputTally()
        self._cached = 0
        self._cache_ceiling = _NO_CEILING
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
            session = event.get("session") or {}
            text = session.get("instructions") or ""
            self._ledger["dry-instructions"] = _InputTally(text=text_tokens(text))
            audio_input = (session.get("audio") or {}).get("input") or {}
            self._transcribe = bool(audio_input.get("transcription"))
        elif kind == "input_audio_buffer.append":
            # base64 -> bytes -> samples -> ms, at 24 kHz mono s16
            b64 = event.get("audio") or ""
            self._pending_input.audio_ms += len(b64) * 3 / 4 / 2 / 24.0
        elif kind == "input_audio_buffer.commit":
            self._commit_audio()
        elif kind == "conversation.item.create":
            item = event.get("item") or {}
            tally = _InputTally()
            for part in item.get("content") or []:
                if part.get("type") in ("input_text", "output_text"):
                    tally.text += text_tokens(part.get("text") or "")
                elif part.get("type") == "input_image":
                    detail = part.get("detail", "high")
                    tally.image += image_tokens(1024, 576, detail=detail)
            self._insert(
                item.get("id") or f"dry-item-{len(self._ledger)}",
                tally,
                event.get("previous_item_id"),
            )
        elif kind == "conversation.item.delete":
            self._delete(event.get("item_id") or "")

    def _commit_audio(self) -> None:
        """Turn the appended buffer into a user message item, and name it.

        The server names this item, not the client, and it says so in
        `input_audio_buffer.committed` -- which is the only way the client ever
        learns the id, and therefore the only way the player's utterance is
        ever removable. Feeding that event (and, when the session asked for
        transcription, the transcript that lets the client write a replacement)
        is what puts the audio half of pruning within reach of a dry run.
        """
        item_id = f"dry-heard-{len(self._ledger):04d}"
        self._insert(item_id, self._pending_input, None)
        self._pending_input = _InputTally()
        self.feed({"type": "input_audio_buffer.committed", "item_id": item_id})
        if self._transcribe:
            self.feed(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "transcript": self.DRY_HEARD,
                }
            )

    def _insert(self, item_id: str, tally: _InputTally, after: str | None) -> None:
        """Add an item, at the end or after a named one.

        `previous_item_id` is how a pruning round lands a replacement exactly
        where the thing it replaces was, so position has to be modelled: the
        ledger's order is what `_delete` walks to work out how much of the
        prefix survives, and appending an item that the server would have put
        in the middle would quietly model the wrong cache.

        An insert caps the cache the same way a delete does, and for the same
        reason -- the prefix only matches up to the first change. That is what
        makes "replacing an item costs no more cache than deleting it" a thing
        a dry run can be asked rather than a thing to take on trust: the delete
        that follows caps it at the same position or earlier.
        """
        if after is not None and after not in self._ledger:
            # What the API does: an unknown `previous_item_id` is an error and
            # the item is not added. Anything that gets this wrong would
            # otherwise show up as a silently mis-ordered conversation.
            # Code and wording taken from what the server actually sends, so a
            # log from a dry run and a log from a real session read alike.
            self.feed(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "item_create_invalid_previous_item_id",
                        "message": (
                            f"Error adding item: the previous item with id '{after}' "
                            "does not exist."
                        ),
                    },
                }
            )
            return
        if after is None:
            self._ledger[item_id] = tally
            return
        # Everything up to and *including* the named item is unchanged, so the
        # cache still matches that far; the delete that follows in a pruning
        # round is what caps it lower.
        through = self._prefix_before(after) + self._ledger[after].total()
        self._cache_ceiling = min(self._cache_ceiling, through)
        rebuilt: dict[str, _InputTally] = {}
        for key, value in self._ledger.items():
            rebuilt[key] = value
            if key == after:
                rebuilt[item_id] = tally
        self._ledger = rebuilt

    def _prefix_before(self, item_id: str) -> int:
        prefix = 0
        for key, tally in self._ledger.items():
            if key == item_id:
                break
            prefix += tally.total()
        return prefix

    def _delete(self, item_id: str) -> None:
        """Drop an item, and cap what the next request can still claim as cached.

        A prompt cache is a prefix match, so removing an item leaves only what
        was ahead of it cacheable. Deletions come from the oldest end, which is
        why a pruning round is close to a full invalidation and why doing them
        in bulk is the whole trick.
        """
        if item_id not in self._ledger:
            return
        self._cache_ceiling = min(self._cache_ceiling, self._prefix_before(item_id))
        del self._ledger[item_id]

    def _refuse_if_full(self) -> bool:
        """Model the ceiling `truncation = "disabled"` leaves in place.

        The instructions are exempt because the API's own limit is on what
        comes *after* them (`token_limits.post_instructions`), and because the
        client cannot prune them anyway -- a limit that counted them would be
        measuring something nobody can act on.

        Nothing else is fed: no `response.created`, no `response.done`, no
        counters moved. A refused response is a response that never happened,
        and the agent has to survive exactly that -- an optimistically-set gate
        with nothing coming to clear it.
        """
        if self.context_limit <= 0:
            return False
        instructions = self._ledger.get("dry-instructions")
        billed = sum(tally.total() for tally in self._ledger.values())
        if billed - (instructions.total() if instructions else 0) <= self.context_limit:
            return False
        self.feed(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "conversation_too_long",
                    "message": "Conversation is too long to create a Response.",
                },
            }
        )
        return True

    def _answer(self, event: dict) -> None:
        if self._refuse_if_full():
            return
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
        # it from here on, exactly like the items that prompted it -- as
        # *audio* when it was spoken, which is the whole reason replacing it
        # with its transcript is worth doing. Billed as text it would look free
        # already and the saving would not show up in a dry run at all.
        self._ledger[item_id] = (
            _InputTally(text=out_text)
            if self.text
            else _InputTally(audio_ms=out_audio * AUDIO_OUT_MS_PER_TOKEN)
        )
        self._cached = billed + self._ledger[item_id].total()
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


#: stand-in for "nothing has capped the cache yet"
_NO_CEILING = 1 << 62
#: output audio is billed at ~1 token per 50 ms (see `tokens.Rates`); this
#: converts the assumed reply back into a duration so it can be re-billed as
#: input audio, at the input rate, like the real thing
AUDIO_OUT_MS_PER_TOKEN = 50.0


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
