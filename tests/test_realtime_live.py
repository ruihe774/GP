"""Tests that talk to the real Realtime API and spend real money.

    pytest -m network

Deselected by default. These are here to catch the one class of bug the mocked
tests structurally cannot: the server rejecting a payload we believe is valid.
They use the `-mini` model and keep responses to a sentence.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from gpagent.agent.env import ENV_VAR, MissingAPIKey, load_api_key
from gpagent.agent.playback import NullPlayer
from gpagent.agent.session import CommentaryAgent, ReplayClock
from gpagent.agent.transport import OpenAITransport
from gpagent.config import CaptureConfig

pytestmark = pytest.mark.network

MODEL = "gpt-realtime-2.1-mini"
SESS5 = Path(__file__).resolve().parent.parent / "sess5"


@pytest.fixture
def api_key():
    try:
        return load_api_key()
    except MissingAPIKey:
        pytest.skip(f"{ENV_VAR} is not set")


async def _open(api_key, **agent_kwargs):
    cfg = CaptureConfig()
    cfg.agent.model = MODEL
    cfg.agent.playback = False
    for key, value in agent_kwargs.items():
        setattr(cfg.agent, key, value)
    clock = ReplayClock(0.0)
    transport = OpenAITransport(MODEL, api_key)
    agent = CommentaryAgent(cfg, transport, NullPlayer(clock), clock=clock)
    await agent.start()
    return agent, clock


async def _wait_for(agent, kinds, timeout=45.0):
    async def poll():
        while not any(line.kind in kinds for line in agent.lines):
            await asyncio.sleep(0.1)

    await asyncio.wait_for(poll(), timeout=timeout)
    return [line for line in agent.lines if line.kind in kinds]


def test_the_session_configuration_is_accepted(api_key):
    """The whole point: `session.update` as we build it must not error."""

    async def run():
        agent, _ = await _open(api_key)
        try:
            await asyncio.sleep(3.0)
            return [line for line in agent.lines if line.kind == "error"]
        finally:
            await agent.close()

    errors = asyncio.run(run())
    assert not errors, f"server rejected our session: {[e.text for e in errors]}"


def test_it_answers_a_spoken_question_with_a_frame_attached(api_key):
    """End to end against the real model: text + image + audio in, speech out."""
    from gpagent.events import GamepadActivity, ScreenFrame, SpeechSegment
    from gpagent.sinks.jsonl import read_session

    if not SESS5.exists():
        pytest.skip("sess5 recording is not checked in")
    events = list(read_session(SESS5, load_blobs=True))
    speech = next(e for e in events if isinstance(e, SpeechSegment) and e.data)
    frame = next(e for e in events if isinstance(e, ScreenFrame) and e.data)

    async def run():
        agent, clock = await _open(api_key)
        try:
            clock.set(1.0)
            await agent.handle(
                GamepadActivity(summary="tapped A x3", intensity=0.4, apm=90), now=1.0
            )
            await agent.handle(frame, now=1.5)
            await agent.handle(speech, now=2.0)
            said = await _wait_for(agent, {"say", "error"})
            return said, agent.meter
        finally:
            await agent.close()

    said, meter = asyncio.run(run())
    assert said[0].kind == "say", f"expected speech, got {said[0].text}"
    assert said[0].text.strip(), "the model said nothing at all"
    assert meter.out_audio > 0, "no audio tokens billed -- it did not speak"
    assert meter.in_image > 0, "the frame was not counted as image input"


def test_a_text_session_writes_its_remark_instead_of_speaking_it(api_key):
    """The one thing no recording can stand in for.

    Every session on disk was captured with `output_modalities: ["audio"]`, so
    replaying one exercises the inputs of a text session and none of its
    output. Only the server can say whether it accepts `["text"]` and answers
    on `response.output_text.*`.
    """
    from gpagent.events import ScreenFrame, SpeechSegment
    from gpagent.sinks.jsonl import read_session

    if not SESS5.exists():
        pytest.skip("sess5 recording is not checked in")
    events = list(read_session(SESS5, load_blobs=True))
    speech = next(e for e in events if isinstance(e, SpeechSegment) and e.data)
    frame = next(e for e in events if isinstance(e, ScreenFrame) and e.data)

    async def run():
        agent, clock = await _open(api_key, output="text")
        try:
            clock.set(1.0)
            await agent.handle(frame, now=1.0)
            await agent.handle(speech, now=2.0)
            said = await _wait_for(agent, {"say", "error"})
            return said, agent.meter, list(agent.hud.shown)
        finally:
            await agent.close()

    said, meter, shown = asyncio.run(run())
    assert said[0].kind == "say", f"expected a written remark, got {said[0].text}"
    assert said[0].text.strip(), "the model wrote nothing at all"
    assert shown == [said[0].text], "the remark must reach the hud, not just the log"
    assert meter.out_text > 0, "no text tokens billed -- it did not write"
    assert meter.out_audio == 0, "a text session must not be billed for speech"
