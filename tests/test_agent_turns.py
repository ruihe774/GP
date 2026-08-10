"""What the agent actually puts on the wire, against a fake transport.

As in `test_speak_policy.py`, the load-bearing assertions are positive ones: an
agent that sent nothing at all would satisfy "never talks over the player" and
"attaches at most one image" perfectly.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from gpagent.agent.playback import NullPlayer
from gpagent.agent.session import CommentaryAgent, ReplayClock
from gpagent.agent.transport import FakeTransport
from gpagent.config import CaptureConfig
from gpagent.events import AgentResponse, GamepadActivity, ScreenFrame, SpeechSegment

JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
PCM = b"\x01\x02" * 12000  # 1 s at 24 kHz s16 mono


def reason_note_sent(agent) -> str:
    """The text of the last per-turn system nudge put on the wire."""
    items = [
        e
        for e in agent.transport.sent_of_type("conversation.item.create")
        if e["item"]["role"] == "system"
    ]
    return items[-1]["item"]["content"][0]["text"]


def make_config(**agent_kwargs):
    cfg = CaptureConfig()
    for key, value in agent_kwargs.items():
        setattr(cfg.agent, key, value)
    return cfg


@pytest.fixture
def agent():
    clock = ReplayClock(100.0)
    cfg = make_config()
    transport = FakeTransport()
    agent = CommentaryAgent(cfg, transport, NullPlayer(clock), clock=clock)
    agent.transport = transport
    agent.clock = clock
    agent._clock_obj = clock
    return agent


def advance(agent, t):
    agent._clock_obj.set(t)
    return t


def speech(dur_ms=1000, data=PCM):
    return SpeechSegment(dur_ms=dur_ms, data=data, rms_dbfs=-24.0)


def frame(seq=1, score=0.0, data=JPEG):
    """score=0 is a frame that is only context; score=1 also asks for a remark."""
    return ScreenFrame(seq=seq, w=1024, h=576, data=data, trigger="scene", scene_score=score)


def pad(summary="tapped A", intensity=0.1):
    return GamepadActivity(summary=summary, intensity=intensity, apm=60)


async def complete_response(agent, *, audio_tokens=50, item_id="item-1"):
    """Play the server's side of one response through to `response.done`."""
    await agent.on_server_event(
        {"type": "response.output_item.added", "item": {"id": item_id}}
    )
    await agent.on_server_event(
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 100,
                    "input_token_details": {"text_tokens": 100},
                    "output_tokens": audio_tokens,
                    "output_token_details": {"audio_tokens": audio_tokens},
                }
            },
        }
    )


class TestTheTurnItSends:
    async def test_a_reply_sends_context_then_audio_then_response(self, agent):
        await agent.handle(pad("tapped A x3"), now=100.0)
        await agent.handle(frame(), now=100.5)
        await agent.on_speech_start(now=101.0)
        await agent.handle(speech(dur_ms=1000), now=102.0)

        order = [e["type"] for e in agent.transport.sent]
        assert order == [
            "conversation.item.create",  # context: summary + frame
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "conversation.item.create",  # the per-turn system nudge
            "response.create",
        ]

    async def test_the_players_audio_arrives_intact(self, agent):
        await agent.handle(speech(dur_ms=1000, data=PCM), now=102.0)
        appends = agent.transport.sent_of_type("input_audio_buffer.append")
        assert appends, "the player's utterance must actually be sent"
        rebuilt = b"".join(base64.b64decode(e["audio"]) for e in appends)
        assert rebuilt == PCM

    async def test_long_audio_is_chunked(self, agent):
        big = b"\x00\x01" * 100_000  # ~400 KB
        await agent.handle(speech(dur_ms=8000, data=big), now=102.0)
        appends = agent.transport.sent_of_type("input_audio_buffer.append")
        assert len(appends) > 1
        assert b"".join(base64.b64decode(e["audio"]) for e in appends) == big

    async def test_the_summary_rides_as_text_not_as_its_own_turn(self, agent):
        await agent.handle(pad("holding RT"), now=100.0)
        await agent.handle(pad("released RT after 2.5s"), now=100.5)
        await agent.handle(speech(), now=101.0)

        items = [
            e
            for e in agent.transport.sent_of_type("conversation.item.create")
            if e["item"]["role"] == "user"
        ]
        assert len(items) == 1, "summaries must be accumulated, not streamed"
        text = items[0]["item"]["content"][0]["text"]
        assert "holding RT" in text and "released RT after 2.5s" in text

    async def test_the_frame_rides_as_an_image_with_the_configured_detail(self, agent):
        agent.cfg.image_detail = "low"
        await agent.handle(frame(), now=100.0)
        await agent.handle(speech(), now=101.0)
        content = agent.transport.sent_of_type("conversation.item.create")[0]["item"]["content"]
        image = [c for c in content if c["type"] == "input_image"][0]
        assert image["detail"] == "low"
        assert image["image_url"].startswith("data:image/jpeg;base64,")
        assert base64.b64decode(image["image_url"].split(",", 1)[1]) == JPEG

    async def test_each_reason_steers_the_response(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        assert "happened" in reason_note_sent(agent).lower()

    async def test_the_reason_rides_as_a_system_item_before_the_response(self, agent):
        """The nudge is a conversation item, not a per-response instruction.

        Carried on `response.create.instructions` it would replace the session
        persona, so it had to drag the whole persona along -- ~265 tokens that
        change every turn, sitting where they key the prompt cache. Sending it
        as a tail item keeps the prefix identical to the previous request.
        """
        await agent.handle(frame(score=1.0), now=100.0)
        sent = agent.transport.sent
        item = [
            m
            for m in sent
            if m["type"] == "conversation.item.create" and m["item"]["role"] == "system"
        ][0]
        assert item["item"]["content"][0]["text"].startswith("Right now:")
        assert sent.index(item) < sent.index(agent.transport.sent_of_type("response.create")[0])

    async def test_the_response_leaves_the_session_persona_alone(self, agent):
        """The persona is sent once and never overridden.

        The first live run against sess5 shipped the reason line alone on
        `response.create.instructions`, which replaces rather than merges, and
        got four sentences of encouraging life coaching per turn.
        """
        agent.cfg.persona = "you are a lighthouse keeper"
        await agent.handle(frame(score=1.0), now=100.0)
        assert "lighthouse keeper" in agent.session_update()["session"]["instructions"]
        assert "response" not in agent.transport.sent_of_type("response.create")[0]

    async def test_the_response_carries_no_per_turn_overrides(self, agent):
        """Everything on `response.create` was constant, so it all moved.

        Anything left here would be re-sent every turn, and the fields that
        vary are the ones that cost cached prefix.
        """
        await agent.handle(frame(score=1.0), now=100.0)
        assert agent.transport.sent_of_type("response.create")[0] == {"type": "response.create"}

    async def test_output_length_is_capped(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        session = agent.session_update()["session"]
        assert session["max_output_tokens"] == agent.cfg.max_output_tokens

    async def test_unanswered_utterances_are_all_sent(self, agent):
        """Three sentences while the agent is busy become one answered thought."""
        await agent.handle(frame(score=1.0), now=100.0)  # agent starts talking
        for i, t in enumerate((101.0, 102.0, 103.0)):
            await agent.handle(speech(dur_ms=1000, data=bytes([i, i]) * 12000), now=t)
        await complete_response(agent)
        agent._clock_obj.set(106.0)  # past the simulated speech plus the reply beat
        await agent.tick(106.0)

        appends = agent.transport.sent_of_type("input_audio_buffer.append")
        rebuilt = b"".join(base64.b64decode(e["audio"]) for e in appends)
        assert rebuilt == bytes([0, 0]) * 12000 + bytes([1, 1]) * 12000 + bytes([2, 2]) * 12000


class TestImagesAreRationed:
    async def test_the_same_frame_is_never_sent_twice(self, agent):
        await agent.handle(frame(seq=7), now=100.0)
        await agent.handle(speech(), now=101.0)
        await complete_response(agent)
        await agent.handle(speech(), now=140.0)

        images = [
            part
            for e in agent.transport.sent_of_type("conversation.item.create")
            for part in e["item"]["content"]
            if part["type"] == "input_image"
        ]
        assert len(images) == 1

    async def test_a_newer_frame_is_sent(self, agent):
        await agent.handle(frame(seq=7), now=100.0)
        await agent.handle(speech(), now=101.0)
        await complete_response(agent)
        await agent.handle(frame(seq=8, data=b"\xff\xd8second"), now=139.0)
        await agent.handle(speech(), now=140.0)

        images = [
            part
            for e in agent.transport.sent_of_type("conversation.item.create")
            for part in e["item"]["content"]
            if part["type"] == "input_image"
        ]
        assert len(images) == 2

    async def test_a_stale_frame_is_not_attached(self, agent):
        agent.cfg.max_image_age_s = 20.0
        await agent.handle(frame(seq=7), now=100.0)
        await agent.handle(speech(), now=200.0)
        content = agent.transport.sent_of_type("conversation.item.create")
        images = [
            part for e in content for part in e["item"]["content"] if part["type"] == "input_image"
        ]
        assert images == []

    async def test_unprompted_remarks_can_be_kept_blind(self, agent):
        agent.cfg.image_on_unprompted = False
        await agent.handle(frame(score=1.0), now=100.0)
        items = agent.transport.sent_of_type("conversation.item.create")
        images = [
            part for e in items for part in e["item"]["content"] if part["type"] == "input_image"
        ]
        assert images == []
        assert agent.transport.sent_of_type("response.create"), "it must still speak"


class TestBargeIn:
    async def test_being_interrupted_cancels_and_truncates(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "assistant-1"}}
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio.delta",
                "item_id": "assistant-1",
                # 4 s of audio queued...
                "delta": base64.b64encode(b"\x00\x00" * 96000).decode(),
            }
        )
        agent._clock_obj.set(101.5)
        await agent.on_speech_start(now=101.5)  # ...but cut off after 1.5 s

        assert agent.transport.sent_of_type("response.cancel"), "must stop talking"
        truncate = agent.transport.sent_of_type("conversation.item.truncate")
        assert truncate, "must tell the model what the player actually heard"
        assert truncate[0]["item_id"] == "assistant-1"
        assert 1400 <= truncate[0]["audio_end_ms"] <= 1600

    async def test_it_does_not_start_talking_while_the_player_is(self, agent):
        await agent.on_speech_start(now=100.0)
        await agent.handle(frame(score=1.0), now=100.5)
        assert agent.transport.sent_of_type("response.create") == []

    async def test_it_does_not_cancel_a_response_the_server_already_finished(self, agent):
        """Audio plays for seconds after `response.done`.

        Cancelling then earns "no active response found" and achieves nothing;
        stopping the speakers and truncating the item is the part that matters.
        """
        await agent.handle(frame(score=1.0), now=100.0)
        await complete_response(agent, audio_tokens=100)  # 5 s of speech queued
        agent._clock_obj.set(101.0)
        await agent.on_speech_start(now=101.0)

        assert agent.transport.sent_of_type("response.cancel") == []
        truncate = agent.transport.sent_of_type("conversation.item.truncate")
        assert truncate, "it must still stop and truncate"
        assert 900 <= truncate[0]["audio_end_ms"] <= 1100

    async def test_late_deltas_from_a_cancelled_response_are_not_played(self, agent):
        """The server keeps streaming until it sees the cancel.

        Playing those would emit a fragment of the sentence we just cut off, a
        second after cutting it off.
        """
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "assistant-1"}}
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio.delta",
                "item_id": "assistant-1",
                "delta": base64.b64encode(b"\x00\x00" * 24000).decode(),
            }
        )
        agent._clock_obj.set(100.5)
        await agent.on_speech_start(now=100.5)
        played_before = agent.player.bytes_played

        await agent.on_server_event(
            {
                "type": "response.output_audio.delta",
                "item_id": "assistant-1",
                "delta": base64.b64encode(b"\x11\x11" * 24000).decode(),
            }
        )
        assert agent.player.bytes_played == played_before

    async def test_a_new_turn_stops_the_previous_sentence_first(self, agent):
        """A defensive invariant, driven directly.

        The policy will not normally hand out a turn while the previous
        sentence is still audible -- `_last_spoken_end` sits in the future until
        it finishes. But if it ever did, pushing the new utterance into the same
        appsrc would queue it behind the old one and play both, late. Cheap to
        guarantee, so it is guaranteed.
        """
        await agent.handle(frame(score=1.0), now=100.0)
        await complete_response(agent, audio_tokens=200)  # 10 s of speech
        assert agent._speaking(101.0), "precondition: still mid-sentence"

        agent._clock_obj.set(101.0)
        await agent._speak("reply", 101.0)

        truncate = agent.transport.sent_of_type("conversation.item.truncate")
        assert truncate, "the previous sentence must be stopped and truncated"
        assert 900 <= truncate[0]["audio_end_ms"] <= 1100
        assert [line for line in agent.lines if line.kind == "cut"]
        assert len(agent.transport.sent_of_type("response.create")) == 2

    async def test_nothing_is_truncated_if_nothing_was_heard(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_speech_start(now=100.0)
        assert agent.transport.sent_of_type("conversation.item.truncate") == []


class TestSessionSetup:
    def test_turn_detection_is_off(self, agent):
        session = agent.session_update()["session"]
        assert session["audio"]["input"]["turn_detection"] is None

    def test_input_audio_matches_what_capture_produces(self, agent):
        fmt = agent.session_update()["session"]["audio"]["input"]["format"]
        assert fmt == {"type": "audio/pcm", "rate": 24000}

    def test_the_persona_is_configurable(self):
        cfg = make_config(persona="be a lighthouse")
        transport = FakeTransport()
        agent = CommentaryAgent(cfg, transport, NullPlayer())
        assert agent.session_update()["session"]["instructions"] == "be a lighthouse"

    def test_transcription_is_omitted_when_off(self):
        cfg = make_config(transcribe_player=False)
        agent = CommentaryAgent(cfg, FakeTransport(), NullPlayer())
        assert "transcription" not in agent.session_update()["session"]["audio"]["input"]


class TestLanguage:
    """Half an API setting, half a prompt: the API has no output-language field."""

    def test_no_language_leaves_the_model_to_follow_the_player(self, agent):
        session = agent.session_update()["session"]
        assert "language" not in session["audio"]["input"]["transcription"]
        assert "Speak" not in session["instructions"].split("How you talk")[0]

    def test_it_is_passed_to_input_transcription(self):
        cfg = make_config(language="ja", transcribe_player=True)
        agent = CommentaryAgent(cfg, FakeTransport(), NullPlayer())
        transcription = agent.session_update()["session"]["audio"]["input"]["transcription"]
        assert transcription["language"] == "ja"

    def test_it_names_the_language_in_the_instructions(self):
        cfg = make_config(language="ja")
        agent = CommentaryAgent(cfg, FakeTransport(), NullPlayer())
        assert "Japanese" in agent.session_update()["session"]["instructions"]

    async def test_it_is_not_overridden_per_response(self, agent):
        """Same trap the persona fell into: instructions are replaced, not merged."""
        agent.cfg.language = "es"
        await agent.handle(frame(score=1.0), now=100.0)
        assert "Spanish" in agent.session_update()["session"]["instructions"]
        assert "response" not in agent.transport.sent_of_type("response.create")[0]

    def test_an_unknown_code_is_passed_through(self):
        cfg = make_config(language="nds")
        agent = CommentaryAgent(cfg, FakeTransport(), NullPlayer())
        assert "nds" in agent.session_update()["session"]["instructions"]

    def test_the_persona_itself_stays_english(self):
        cfg = make_config(language="ja")
        agent = CommentaryAgent(cfg, FakeTransport(), NullPlayer())
        assert "sitting on the couch" in agent.session_update()["session"]["instructions"]


class TestRecording:
    """A session must hold both halves of the conversation."""

    def recording_agent(self):
        clock = ReplayClock(100.0)
        recorded = []
        agent = CommentaryAgent(
            make_config(), FakeTransport(), NullPlayer(clock),
            clock=clock, recorder=recorded.append,
        )
        agent._clock_obj = clock
        return agent, recorded

    async def test_capture_events_are_recorded(self):
        agent, recorded = self.recording_agent()
        await agent.handle(pad("tapped A"), now=100.0)
        await agent.handle(frame(seq=2), now=100.5)
        assert [e.TYPE for e in recorded] == ["gamepad.activity", "screen.frame"]

    async def test_what_the_agent_said_is_recorded_with_its_audio(self):
        agent, recorded = self.recording_agent()
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "a1"}}
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio.delta",
                "item_id": "a1",
                "delta": base64.b64encode(b"\x01\x02" * 24000).decode(),  # 1 s
            }
        )
        await agent.on_server_event(
            {"type": "response.output_audio_transcript.done", "transcript": "nice one"}
        )
        await complete_response(agent)

        said = [e for e in recorded if isinstance(e, AgentResponse)]
        assert len(said) == 1
        assert said[0].data == b"\x01\x02" * 24000
        assert said[0].transcript == "nice one"
        assert said[0].reason == "react"
        assert said[0].dur_ms == 1000
        assert not said[0].cut

    async def test_an_interrupted_response_is_recorded_as_cut(self):
        agent, recorded = self.recording_agent()
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "a1"}}
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio.delta",
                "item_id": "a1",
                "delta": base64.b64encode(b"\x00\x00" * 96000).decode(),  # 4 s
            }
        )
        agent._clock_obj.set(101.0)
        await agent.on_speech_start(now=101.0)

        said = [e for e in recorded if isinstance(e, AgentResponse)]
        assert len(said) == 1 and said[0].cut
        assert 900 <= said[0].dur_ms <= 1100, "records what was heard, not what was sent"

    async def test_agent_events_land_on_the_capture_timeline(self):
        """Live, the agent clock and event.t are different bases."""
        clock = ReplayClock(5000.0)  # a monotonic-looking clock...
        recorded = []
        agent = CommentaryAgent(
            make_config(), FakeTransport(), NullPlayer(clock),
            clock=clock, recorder=recorded.append,
        )
        await agent.handle(pad("tapped A"), now=5000.0)  # ...for an event at t=0
        clock.set(5002.0)
        await agent.handle(frame(score=1.0), now=5002.0)
        await complete_response(agent)

        said = [e for e in recorded if isinstance(e, AgentResponse)][0]
        assert 1.9 <= said.t <= 2.6, f"t={said.t} is not on the capture timeline"

    async def test_a_backlogged_first_event_does_not_skew_the_timeline(self):
        """Regression: sess6 recorded every response 3.5 s early.

        The first event the agent sees comes out of the queue that built up
        while `start()` was opening the realtime session, so its lag is the
        startup delay and not the offset between the clocks. Calibrating on it
        shifted the agent's whole half of the recording earlier than the
        player's, which `inspect` showed as replies arriving before the
        question they answered.
        """
        clock = ReplayClock(1000.0)
        recorded = []
        agent = CommentaryAgent(
            make_config(), FakeTransport(), NullPlayer(clock),
            clock=clock, recorder=recorded.append,
        )
        backlogged = pad("tapped A")
        backlogged.t = 0.0
        await agent.handle(backlogged, now=1003.5)  # 3.5 s late out of the queue

        prompt = frame(score=1.0)
        prompt.t = 10.0
        clock.set(1010.0)
        await agent.handle(prompt, now=1010.0)  # and everything after is prompt
        await complete_response(agent)

        said = [e for e in recorded if isinstance(e, AgentResponse)][0]
        assert said.t == pytest.approx(10.0, abs=0.5), (
            f"t={said.t}: the startup backlog leaked into the clock offset"
        )

    async def test_recording_is_off_by_default(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await complete_response(agent)  # must not raise without a recorder


class TestBargeInSpares:
    """Barge-in stops the agent talking over the player. Nothing else."""

    async def push(self, agent, seconds):
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "a1"}}
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio.delta",
                "item_id": "a1",
                "delta": base64.b64encode(b"\x00\x00" * int(24000 * seconds)).decode(),
            }
        )

    async def test_a_response_that_has_not_made_a_sound_survives(self, agent):
        """sess7: four consecutive questions cancelled at 0.0 s, none answered.

        The player talking in short bursts starts a reply, then starts talking
        again before the first audio arrives ~1.9 s later. Cancelling then
        stops nothing and costs the answer.
        """
        await agent.handle(speech(), now=100.0)
        assert agent.transport.sent_of_type("response.create"), "a reply was asked for"
        agent.transport.sent.clear()

        advance(agent, 101.0)
        await agent.on_speech_start(now=101.0)  # they carry on talking

        assert not agent.transport.sent_of_type("response.cancel")
        assert agent._response_active, "the answer was thrown away for nothing"

    async def test_a_response_already_speaking_is_still_cut(self, agent):
        await agent.handle(speech(), now=100.0)
        await self.push(agent, 4.0)
        advance(agent, 101.0)  # 1 s of it has been heard
        await agent.on_speech_start(now=101.0)

        assert agent.transport.sent_of_type("response.cancel")
        truncate = agent.transport.sent_of_type("conversation.item.truncate")
        assert truncate and 900 <= truncate[0]["audio_end_ms"] <= 1100


class TestTheSessionComesBack:
    """The API caps a session at 60 minutes, so this is not an edge case."""

    def dropping_transport(self):
        class Dropping(FakeTransport):
            def __init__(self):
                super().__init__()
                self.connects = 0

            async def connect(self):
                self.connects += 1
                self._incoming = asyncio.Queue()  # a new socket, a new stream
                self._closed = False

            def drop(self):
                self._incoming.put_nowait(None)

        return Dropping()

    async def started(self, transport):
        cfg = make_config(reconnect_min_s=0.01, reconnect_max_s=0.01)
        agent = CommentaryAgent(cfg, transport, NullPlayer())
        await agent.start()
        return agent

    async def settle(self):
        for _ in range(50):
            await asyncio.sleep(0)

    async def test_a_dropped_socket_is_reopened_and_reseeded(self):
        transport = self.dropping_transport()
        agent = await self.started(transport)
        try:
            transport.drop()
            await asyncio.sleep(0.05)
            assert transport.connects == 2, "the agent went permanently mute"
            assert len(transport.sent_of_type("session.update")) == 2, "persona not reseeded"
            assert any("reopened" in line.text for line in agent.lines)
        finally:
            await agent.close()

    async def test_closing_does_not_reopen(self):
        transport = self.dropping_transport()
        agent = await self.started(transport)
        await agent.close()
        await asyncio.sleep(0.05)
        assert transport.connects == 1

    async def test_a_response_in_flight_does_not_gate_the_new_session(self):
        """The in-flight gate is absolute, and nothing will ever finish it."""
        transport = self.dropping_transport()
        agent = await self.started(transport)
        try:
            await agent.handle(frame(score=1.0), now=100.0)
            assert agent._response_active
            transport.drop()
            await asyncio.sleep(0.05)
            assert not agent._response_active
            assert not agent._items and not agent._image_items, "stale item ids kept"
        finally:
            await agent.close()


class TestResponsesThatSayNothing:
    """One reply in sess6 came back empty and nothing anywhere said why."""

    async def test_a_non_completed_response_is_logged(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {
                "type": "response.done",
                "response": {
                    "status": "incomplete",
                    "status_details": {"reason": "max_output_tokens"},
                    "usage": {},
                },
            }
        )
        notes = [line for line in agent.lines if line.kind == "note"]
        assert notes, "a response that said nothing must leave a trace"
        assert "incomplete" in notes[0].text and "max_output_tokens" in notes[0].text

    async def test_our_own_barge_in_is_not_noted_twice(self, agent):
        """It is already logged as `cut`; noting it too buried the real ones."""
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {
                "type": "response.done",
                "response": {
                    "status": "cancelled",
                    "status_details": {"reason": "client_cancelled"},
                    "usage": {},
                },
            }
        )
        assert not [line for line in agent.lines if line.kind == "note"]

    async def test_a_completed_response_is_not_annotated(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {"type": "response.done", "response": {"status": "completed", "usage": {}}}
        )
        assert not [line for line in agent.lines if line.kind == "note"]


class TestContextPruning:
    async def test_old_items_are_deleted(self, agent):
        agent.cfg.prune_after_s = 60.0
        for i, t in enumerate((100.0, 110.0)):
            agent._clock_obj.set(t)
            await agent.on_server_event(
                {"type": "conversation.item.created", "item": {"id": f"old-{i}"}}
            )
        agent._clock_obj.set(400.0)
        await agent.tick(400.0)
        deleted = {e["item_id"] for e in agent.transport.sent_of_type("conversation.item.delete")}
        assert deleted == {"old-0", "old-1"}

    async def test_pruning_is_off_by_default(self, agent):
        """Deleting items invalidates the prompt cache; measured, it loses."""
        assert agent.cfg.prune_after_s == 0.0
        agent._clock_obj.set(100.0)
        await agent.on_server_event(
            {"type": "conversation.item.created", "item": {"id": "old"}}
        )
        agent._clock_obj.set(100_000.0)
        await agent.tick(100_000.0)
        assert agent.transport.sent_of_type("conversation.item.delete") == []

    async def test_recent_items_are_kept(self, agent):
        agent.cfg.prune_after_s = 300.0
        agent._clock_obj.set(100.0)
        await agent.on_server_event(
            {"type": "conversation.item.created", "item": {"id": "fresh"}}
        )
        agent._clock_obj.set(200.0)
        await agent.tick(200.0)
        assert agent.transport.sent_of_type("conversation.item.delete") == []


class TestImagesAreRetired:
    """Images left in context are re-billed on every later turn, not once."""

    async def send_frames(self, agent, count, keep):
        agent.cfg.keep_images = keep
        for i in range(count):
            t = 100.0 + i * 40.0
            agent._clock_obj.set(t)
            await agent.handle(frame(seq=i, data=bytes([i]) + b"\xd8jpeg"), now=t)
            await agent.handle(speech(), now=t + 1)
            await complete_response(agent, item_id=f"item-{i}")

    async def test_only_the_newest_images_are_kept(self, agent):
        await self.send_frames(agent, count=4, keep=2)
        sent = [
            e["item"]["id"]
            for e in agent.transport.sent_of_type("conversation.item.create")
            if any(c["type"] == "input_image" for c in e["item"]["content"])
        ]
        deleted = [e["item_id"] for e in agent.transport.sent_of_type("conversation.item.delete")]
        assert len(sent) == 4
        assert deleted == sent[:2], "the two oldest screenshots must be dropped"

    async def test_keeping_more_deletes_fewer(self, agent):
        await self.send_frames(agent, count=4, keep=4)
        assert agent.transport.sent_of_type("conversation.item.delete") == []

    async def test_items_carry_a_client_id_so_they_can_be_pruned_at_once(self, agent):
        await agent.handle(frame(), now=100.0)
        await agent.handle(speech(), now=101.0)
        item = agent.transport.sent_of_type("conversation.item.create")[0]["item"]
        assert item["id"].startswith("gpctx")


class TestUsageIsRecorded:
    async def test_response_done_feeds_the_meter(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await complete_response(agent, audio_tokens=120)
        assert agent.meter.responses == 1
        assert agent.meter.out_audio == 120
        assert agent.report()["usage"]["spoken_s"] == 6.0
