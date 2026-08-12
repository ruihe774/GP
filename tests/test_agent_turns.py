"""What the agent actually puts on the wire, against a fake transport.

As in `test_speak_policy.py`, the load-bearing assertions are positive ones: an
agent that sent nothing at all would satisfy "never talks over the player" and
"attaches at most one image" perfectly.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from gpagent.agent.hud import NullHud
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


def decode(image_part) -> bytes:
    return base64.b64decode(image_part["image_url"].split(",", 1)[1])


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
            "conversation.item.create",  # the controller summary
            "conversation.item.create",  # the screenshots, in an item of their own
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

    async def test_earlier_frames_ride_along_cheaply_and_in_order(self, agent):
        """The trajectory, not just the destination.

        The trail is the frames capture already paid for and the agent used to
        throw away. They go on the wire at `image_trail_detail` (85 tokens flat)
        ahead of the current frame, oldest first, with a line of text saying so
        -- images in one item carry no timestamps and no stated direction.
        """
        agent.cfg.image_trail = 2
        for i, t in enumerate([100.0, 103.0, 106.0, 109.0]):
            await agent.handle(frame(seq=i, data=JPEG + bytes([i])), now=t)
        await agent.handle(speech(), now=110.0)

        content = agent.transport.sent_of_type("conversation.item.create")[0]["item"]["content"]
        images = [c for c in content if c["type"] == "input_image"]
        assert [c["detail"] for c in images] == ["low", "low", "high"]

        notes = [c for c in content if c["type"] == "input_text"]
        assert "oldest first" in notes[-1]["text"]
        assert content.index(notes[-1]) < content.index(images[0]), "the note introduces them"

        # Spread across the window (not the last two frames), oldest first, and
        # ending on the frame the model is actually being asked about.
        assert [decode(c) for c in images] == [
            JPEG + bytes([1]),
            JPEG + bytes([2]),
            JPEG + bytes([3]),
        ]

    async def test_the_ask_line_names_which_frames_it_sent(self, agent):
        """Identities, not counts.

        "trail 2 low" tells a reader how much was sent but not what, so the
        images the model actually saw could not be pulled back out of the
        recording. `gpagent inspect --sent-sheet` reads this field.
        """
        lines = []
        agent._on_line = lines.append
        agent.cfg.image_trail = 2
        for i, t in enumerate([100.0, 103.0, 106.0, 109.0]):
            await agent.handle(frame(seq=i), now=t)
        await agent.handle(speech(), now=110.0)

        ask = [ln for ln in lines if ln.kind == "ask"][-1]
        assert ask.extra["frames"] == {
            "current": 3,
            "detail": "high",
            "trail": [1, 2],
            "trail_detail": "low",
        }

    async def test_the_trail_is_not_re_sent_on_the_next_turn(self, agent):
        agent.cfg.image_trail = 4
        await agent.handle(frame(seq=0), now=100.0)
        await agent.handle(frame(seq=1), now=101.0)
        await agent.handle(speech(), now=102.0)
        await complete_response(agent)

        advance(agent, 130.0)
        await agent.handle(frame(seq=2), now=130.0)
        await agent.handle(speech(), now=131.0)
        items = [
            e
            for e in agent.transport.sent_of_type("conversation.item.create")
            if e["item"]["role"] == "user"
        ]
        assert len(items) == 2
        second = [c for c in items[1]["item"]["content"] if c["type"] == "input_image"]
        assert [c["detail"] for c in second] == ["high"], "no duplicate of a billed frame"

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
        cfg = make_config(persona="be a lighthouse", wait_tool=False)
        transport = FakeTransport()
        agent = CommentaryAgent(cfg, transport, NullPlayer())
        assert agent.session_update()["session"]["instructions"] == "be a lighthouse"

    def test_the_persona_keeps_the_notes_the_session_shape_adds(self):
        """A replaced persona still gets told about the tool it has been given."""
        cfg = make_config(persona="be a lighthouse")
        agent = CommentaryAgent(cfg, FakeTransport(), NullPlayer())
        instructions = agent.session_update()["session"]["instructions"]
        assert instructions.startswith("be a lighthouse")
        assert "wait_for_user" in instructions

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
            assert not agent._disposable, "stale item ids kept"
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


class TestTheModelCanDeclineATurn:
    """`wait_for_user`: the one way the agent gets to stay quiet.

    The policy opens a turn from what capture can see, which cannot tell the
    player's voice from the television's. This is the model saying so.
    """

    async def hold(self, agent, *, call_id="call_1", item_id="fc-1"):
        """The server's side of a turn answered with a tool call."""
        await agent.on_server_event(
            {
                "type": "response.output_item.added",
                "item": {"id": item_id, "type": "function_call", "name": "wait_for_user"},
            }
        )
        await agent.on_server_event(
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": item_id,
                            "name": "wait_for_user",
                            "call_id": call_id,
                            "arguments": "{}",
                        }
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 4},
                },
            }
        )

    def outputs(self, agent):
        return [
            e["item"]
            for e in agent.transport.sent_of_type("conversation.item.create")
            if e["item"].get("type") == "function_call_output"
        ]

    def test_the_session_offers_the_tool(self, agent):
        session = agent.session_update()["session"]
        assert [t["name"] for t in session["tools"]] == ["wait_for_user"]
        assert "wait_for_user" in session["instructions"], "and says it may be used"

    def test_it_can_be_turned_off(self):
        agent = CommentaryAgent(
            make_config(wait_tool=False), FakeTransport(), NullPlayer()
        )
        session = agent.session_update()["session"]
        assert "tools" not in session
        assert "wait_for_user" not in session["instructions"]

    async def test_the_call_is_answered_and_the_turn_just_ends(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await self.hold(agent)

        outputs = self.outputs(agent)
        assert len(outputs) == 1 and outputs[0]["call_id"] == "call_1"
        assert len(agent.transport.sent_of_type("response.create")) == 1, (
            "answering the tool call must not ask the same question again"
        )
        assert agent.held == 1
        assert [line.kind for line in agent.lines].count("hold") == 1

    async def test_nothing_is_recorded_as_having_been_said(self):
        clock = ReplayClock(100.0)
        recorded = []
        agent = CommentaryAgent(
            make_config(), FakeTransport(), NullPlayer(clock),
            clock=clock, recorder=recorded.append,
        )
        agent._clock_obj = clock
        await agent.handle(frame(score=1.0), now=100.0)
        await self.hold(agent)
        assert not [e for e in recorded if isinstance(e, AgentResponse)]

    async def test_the_agent_can_speak_again_afterwards(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await self.hold(agent)
        t = advance(agent, 160.0)
        await agent.handle(frame(seq=2, score=1.0), now=t)
        await agent.tick(now=t)
        assert len(agent.transport.sent_of_type("response.create")) == 2

    async def test_a_hold_leaves_nothing_behind_in_the_conversation(self, agent):
        """Both halves go on the next round: a hold is over the moment it ends."""
        await agent.handle(frame(score=1.0), now=100.0)
        await self.hold(agent)
        agent.cfg.prune_after_s = 30.0
        t = advance(agent, 200.0)
        await agent.tick(now=t)

        gone = set(deleted(agent))
        assert "fc-1" in gone, "the call itself"
        assert self.outputs(agent)[0]["id"] in gone, "and the answer to it"


class TestPreamblesNeverReachThePlayer:
    """`gpt-realtime-2` opens with "let me check that" unless told not to.

    The persona tells it not to. This is the second line of defence: a response
    part the server marks `phase: "commentary"` is a preamble, and none of it is
    played, shown, or folded into what the agent is recorded as having said.
    """

    async def test_a_spoken_preamble_is_not_played(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {
                "type": "response.output_item.added",
                "item": {"id": "pre-1", "phase": "commentary"},
            }
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio.delta",
                "item_id": "pre-1",
                "delta": base64.b64encode(b"\x01\x02" * 24000).decode(),
            }
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio_transcript.done",
                "item_id": "pre-1",
                "transcript": "let me have a look",
            }
        )
        assert not [line for line in agent.lines if line.kind == "say"]
        assert agent._response_transcript == ""

    async def test_the_remark_after_it_still_arrives(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {
                "type": "response.output_item.added",
                "item": {"id": "pre-1", "phase": "commentary"},
            }
        )
        await agent.on_server_event(
            {
                "type": "response.output_item.added",
                "item": {"id": "a1", "phase": "final_answer"},
            }
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio_transcript.done",
                "item_id": "a1",
                "transcript": "that barrel again",
            }
        )
        said = [line.text for line in agent.lines if line.kind == "say"]
        assert said == ["that barrel again"]

    async def test_a_written_preamble_never_reaches_the_hud(self):
        clock = ReplayClock(100.0)
        agent = CommentaryAgent(
            make_config(output="text"), FakeTransport(), NullPlayer(clock), NullHud(),
            clock=clock,
        )
        agent._clock_obj = clock
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {
                "type": "response.output_item.added",
                "item": {"id": "pre-1", "phase": "commentary"},
            }
        )
        await agent.on_server_event(
            {"type": "response.output_text.done", "item_id": "pre-1", "text": "one sec"}
        )
        assert agent.hud.shown == []


def turns(agent, count, *, every=40.0, start=100.0):
    """Play `count` complete turns, one screenshot and one utterance each."""

    async def run():
        for i in range(count):
            t = start + i * every
            agent._clock_obj.set(t)
            await agent.handle(frame(seq=i, data=bytes([i]) + b"\xd8jpeg"), now=t)
            await agent.handle(speech(), now=t + 1)
            await complete_response(agent, item_id=f"item-{i}")

    return run()


def created(agent, prefix):
    return [
        e["item"]["id"]
        for e in agent.transport.sent_of_type("conversation.item.create")
        if e["item"]["id"].startswith(prefix)
    ]


def deleted(agent):
    return [e["item_id"] for e in agent.transport.sent_of_type("conversation.item.delete")]


def created_text(agent, prefix):
    """Every line of text sent in items whose id starts with `prefix`."""
    return [
        part["text"]
        for e in agent.transport.sent_of_type("conversation.item.create")
        if e["item"]["id"].startswith(prefix)
        for part in e["item"]["content"]
        if part["type"] in ("input_text", "output_text")
    ]


class TestContextPruning:
    """What the client removes again, and how rarely it does it.

    The conversation is re-billed as input on every turn, so a screenshot left
    in it is paid for once per response for as long as it stays -- and cached
    or not, it counts against TPM in full, which is what ended sess_movie2.
    """

    async def test_stale_images_are_deleted(self, agent):
        agent.cfg.prune_after_s = 60.0
        await turns(agent, 4)
        assert deleted(agent), "screenshots must not accumulate forever"
        removable = (
            set(created(agent, "gpimg"))
            | set(created(agent, "gprsn"))
            # the agent's own replies, named by the server in `turns`
            | {f"item-{i}" for i in range(4)}
        )
        assert set(deleted(agent)) <= removable

    async def test_the_screen_note_goes_with_the_frames_it_introduces(self, agent):
        """"[screen] 2 earlier frames..." is meaningless once they are gone.

        It shares the images' item rather than the text one precisely so it
        cannot outlive them: a note left behind would be telling the model to
        look at a sequence of screenshots that is no longer in the conversation.
        """
        agent.cfg.prune_after_s = 60.0
        agent.cfg.image_trail = 2
        for i, t in enumerate([100.0, 103.0, 106.0, 109.0]):
            await agent.handle(frame(seq=i, data=JPEG + bytes([i])), now=t)
        await agent.handle(speech(), now=110.0)
        await complete_response(agent)
        agent._clock_obj.set(400.0)
        await agent.tick(400.0)

        surviving = [
            c["text"]
            for e in agent.transport.sent_of_type("conversation.item.create")
            if e["item"]["id"] not in set(deleted(agent))
            for c in e["item"]["content"]
            if c["type"] == "input_text"
        ]
        assert not [t for t in surviving if "[screen]" in t]

    async def test_both_the_current_frame_and_its_trail_go(self, agent):
        """One item holds both, so one delete takes both with it."""
        agent.cfg.prune_after_s = 60.0
        agent.cfg.image_trail = 2
        for i, t in enumerate([100.0, 103.0, 106.0, 109.0]):
            await agent.handle(frame(seq=i, data=JPEG + bytes([i])), now=t)
        await agent.handle(speech(), now=110.0)
        await complete_response(agent)
        agent._clock_obj.set(400.0)
        await agent.tick(400.0)

        item = [
            e["item"]
            for e in agent.transport.sent_of_type("conversation.item.create")
            if e["item"]["id"] in set(deleted(agent))
        ]
        images = [c for c in item[0]["content"] if c["type"] == "input_image"]
        assert [c["detail"] for c in images] == ["low", "low", "high"]

    async def test_old_system_notes_go_in_the_same_round(self, agent):
        """The per-turn nudge is small, identical every turn, and there is one
        per turn -- so it is the second thing worth deleting, and it costs
        nothing extra to delete it alongside the images."""
        agent.cfg.prune_after_s = 60.0
        await turns(agent, 4)
        gone = set(deleted(agent))
        assert gone & set(created(agent, "gpimg"))
        assert gone & set(created(agent, "gprsn"))

    async def test_the_controller_summary_survives_its_screenshots(self, agent):
        """Text is the cheap half; it is in its own item so it can stay."""
        agent.cfg.prune_after_s = 60.0
        for i in range(4):
            t = 100.0 + i * 40.0
            agent._clock_obj.set(t)
            await agent.handle(pad(f"tapped {i}"), now=t)
            await agent.handle(frame(seq=i), now=t + 0.5)
            await agent.handle(speech(), now=t + 1)
            await complete_response(agent, item_id=f"item-{i}")
        assert created(agent, "gpctx"), "summaries must have gone out"
        assert not set(deleted(agent)) & set(created(agent, "gpctx"))

    async def test_deletes_arrive_in_one_bulk_round(self, agent):
        """The prompt cache is invalidated per round, not per item.

        Six turns inside one interval leave six items' worth of deletions, and
        they must go out together: dribbling one out per turn would pay the
        invalidation six times for the same saving.
        """
        agent.cfg.prune_after_s = 30.0
        agent.cfg.prune_interval_s = 500.0
        await turns(agent, 6, every=20.0)  # 100..200, all inside one interval
        agent._clock_obj.set(1000.0)
        await agent.tick(1000.0)
        assert len(deleted(agent)) >= 6
        rounds = [line for line in agent.lines if line.kind == "prune"]
        assert len(rounds) == 1, "one round, however many items it drops"
        assert agent.report()["pruned"]["rounds"] == 1

    async def test_the_interval_is_what_limits_how_often_the_cache_dies(self, agent):
        agent.cfg.prune_after_s = 10.0
        agent.cfg.prune_interval_s = 200.0
        await turns(agent, 6, every=30.0)  # 150 s of turns, every one of them stale
        assert len([line for line in agent.lines if line.kind == "prune"]) == 1

    async def test_a_shorter_interval_prunes_more_often(self, agent):
        agent.cfg.prune_after_s = 10.0
        agent.cfg.prune_interval_s = 30.0
        await turns(agent, 6, every=30.0)
        assert len([line for line in agent.lines if line.kind == "prune"]) > 1

    async def test_recent_items_are_kept(self, agent):
        agent.cfg.prune_after_s = 300.0
        await turns(agent, 3)
        assert deleted(agent) == []

    async def test_pruning_is_off_by_default(self, agent):
        """As a cost measure, deleting items loses to the prompt cache."""
        assert agent.cfg.prune_after_s == 0.0
        await turns(agent, 4)
        agent._clock_obj.set(100_000.0)
        await agent.tick(100_000.0)
        assert deleted(agent) == []

    async def test_images_carry_a_client_id_so_they_can_be_pruned_at_once(self, agent):
        await agent.handle(frame(), now=100.0)
        await agent.handle(speech(), now=101.0)
        item = agent.transport.sent_of_type("conversation.item.create")[0]["item"]
        assert item["id"].startswith("gpimg")

    async def test_a_pruned_image_leaves_a_line_saying_there_were_images(self, agent):
        """The moment is kept; only the pixels go.

        A screenshot deleted outright is a beat of the session the model no
        longer knows happened. A dozen tokens buys the fact of it back.
        """
        agent.cfg.prune_after_s = 60.0
        agent.cfg.image_trail = 2
        await turns(agent, 4)
        stubs = [text for text in created_text(agent, "gpstb")]
        assert [t for t in stubs if t.startswith("[dropped]")], stubs
        assert not set(created(agent, "gpstb")) & set(deleted(agent)), "a stub is not itself pruned"

    async def test_the_stub_lands_where_the_item_it_replaces_was(self, agent):
        """`previous_item_id`, so the conversation keeps its chronology.

        Appended at the tail instead, the note about a screenshot from four
        minutes ago would read as a remark about what is on screen now.
        """
        agent.cfg.prune_after_s = 60.0
        await turns(agent, 4)
        creates = agent.transport.sent_of_type("conversation.item.create")
        stubs = [e for e in creates if e["item"]["id"].startswith("gpstb")]
        assert stubs, "something must have been replaced"
        for event in stubs:
            assert event["previous_item_id"] in set(deleted(agent))

    async def test_the_stub_is_created_before_the_item_it_replaces_is_deleted(self, agent):
        """The order is forced: `previous_item_id` must name a live item.

        Delete first and the create is refused, the replacement is lost, and
        all that is left of it is an error line.
        """
        agent.cfg.prune_after_s = 60.0
        await turns(agent, 4)
        order = [
            (e["type"], e.get("item", {}).get("id") or e.get("item_id"), e.get("previous_item_id"))
            for e in agent.transport.sent
            if e["type"] in ("conversation.item.create", "conversation.item.delete")
        ]
        seen_deleted: set[str] = set()
        for kind, item_id, after in order:
            if kind == "conversation.item.delete":
                seen_deleted.add(item_id)
            elif after is not None:
                assert after not in seen_deleted, f"{item_id} was inserted after a dead item"

    async def test_every_stale_nudge_goes_in_one_round_however_young(self, agent):
        """A nudge is a sentence about its own turn, and only its own turn.

        The cutoff is about what is still worth looking at; a nudge from the
        previous turn was not worth looking at the moment the turn ended.
        """
        agent.cfg.prune_after_s = 200.0
        await turns(agent, 1, start=100.0)
        await turns(agent, 3, start=250.0, every=10.0)
        # One old turn is what triggers the round, at a cutoff of t=110. The
        # three nudges from 251, 261 and 271 are a fraction of that age.
        agent._clock_obj.set(310.0)
        await agent.tick(310.0)

        notes = created(agent, "gprsn")
        gone = set(deleted(agent))
        assert len(notes) == 4
        assert len(gone & set(notes)) == 3, "every nudge but the newest, whatever its age"

    async def test_the_current_nudge_survives_the_round_that_takes_the_rest(self, agent):
        """It is the only one that is still true."""
        agent.cfg.prune_after_s = 200.0
        await turns(agent, 1, start=100.0)
        await turns(agent, 3, start=250.0, every=10.0)
        agent._clock_obj.set(310.0)
        await agent.tick(310.0)
        assert created(agent, "gprsn")[-1] not in set(deleted(agent))

    async def test_a_round_is_no_more_frequent_than_it_was(self, agent):
        """Sweeping every nudge must not become a reason to *start* a round.

        If it did, a round would fire every `prune_interval_s` for the rest of
        the session and throw away the prompt-cache prefix each time, to save
        thirty tokens of nudge.
        """
        agent.cfg.prune_after_s = 300.0
        agent.cfg.prune_interval_s = 10.0
        await turns(agent, 6, every=20.0)
        assert [line for line in agent.lines if line.kind == "prune"] == []


class TestSpokenTurnsBecomeText:
    """Audio is the one thing the client could never remove, and the dearest.

    A committed utterance and a spoken reply are named by the *server*, so
    until the client learned those names they sat in the conversation for the
    life of the session, re-billed as audio at 8x the text rate every turn.
    Both have a text version already: the transcript.
    """

    async def spoke(self, agent, *, t=100.0, transcript="that was close"):
        """One reply turn, transcribed on both sides, through to response.done."""
        await agent.handle(frame(), now=t)
        await agent.handle(speech(), now=t + 1)
        await agent.on_server_event(
            {"type": "input_audio_buffer.committed", "item_id": "heard-1"}
        )
        await agent.on_server_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "heard-1",
                "transcript": "did you see that",
            }
        )
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "said-1"}}
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio_transcript.done",
                "item_id": "said-1",
                "transcript": transcript,
            }
        )
        await agent.on_server_event(
            {"type": "response.done", "response": {"status": "completed", "usage": {}}}
        )

    async def test_a_committed_utterance_is_replaced_by_its_transcription(self, agent):
        agent.cfg.prune_after_s = 60.0
        await self.spoke(agent)
        agent._clock_obj.set(400.0)
        await agent.tick(400.0)

        assert "heard-1" in deleted(agent)
        assert "[they said] did you see that" in created_text(agent, "gpstb")

    async def test_the_agents_own_words_go_back_in_as_an_assistant_message(self, agent):
        """`conversation.item.create` cannot carry assistant audio, so: text.

        Unmarked, too. The agent did say those words, and a bracket around
        them would read as a stage direction for it to imitate.
        """
        agent.cfg.prune_after_s = 60.0
        await self.spoke(agent, transcript="that was close")
        agent._clock_obj.set(400.0)
        await agent.tick(400.0)

        stub = [
            e
            for e in agent.transport.sent_of_type("conversation.item.create")
            if e["item"]["id"].startswith("gpstb") and e["item"]["role"] == "assistant"
        ]
        assert stub, "the reply must come back as the assistant's own words"
        assert stub[0]["item"]["content"] == [{"type": "output_text", "text": "that was close"}]
        assert "said-1" in deleted(agent)

    async def test_audio_is_not_deleted_before_its_transcript_arrives(self, agent):
        """Delete it first and the only record of what was said goes with it."""
        agent.cfg.prune_after_s = 60.0
        agent.cfg.audio_stub_wait_s = 1000.0  # the deadline is the next test
        await agent.handle(frame(), now=100.0)
        await agent.handle(speech(), now=101.0)
        await agent.on_server_event(
            {"type": "input_audio_buffer.committed", "item_id": "heard-1"}
        )
        agent._clock_obj.set(400.0)
        await agent.tick(400.0)
        assert "heard-1" not in deleted(agent)

    async def test_a_transcript_that_never_arrives_stops_pinning_the_item(self, agent):
        """...but not forever: an absolute gate that cannot clear is a leak.

        One transcription lost to a hiccup would otherwise hold its audio in
        the conversation for the rest of the session.
        """
        agent.cfg.prune_after_s = 60.0
        agent.cfg.audio_stub_wait_s = 30.0
        await agent.handle(frame(), now=100.0)
        await agent.handle(speech(), now=101.0)
        await agent.on_server_event(
            {"type": "input_audio_buffer.committed", "item_id": "heard-1"}
        )
        await agent.on_server_event(
            {"type": "response.done", "response": {"status": "completed", "usage": {}}}
        )
        agent._clock_obj.set(400.0)
        await agent.tick(400.0)
        assert "heard-1" in deleted(agent)
        assert "[they said something here.]" in created_text(agent, "gpstb")

    async def test_a_failed_transcription_settles_the_item_at_once(self, agent):
        agent.cfg.prune_after_s = 60.0
        await agent.handle(speech(), now=101.0)
        await agent.on_server_event(
            {"type": "input_audio_buffer.committed", "item_id": "heard-1"}
        )
        await agent.on_server_event(
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": "heard-1",
            }
        )
        await agent.on_server_event(
            {"type": "response.done", "response": {"status": "completed", "usage": {}}}
        )
        agent._clock_obj.set(200.0)
        await agent.tick(200.0)
        assert "heard-1" in deleted(agent)

    async def test_a_cancelled_reply_keeps_the_part_that_was_said(self, agent):
        """No transcript event follows a cut-off response; what was heard does."""
        agent.cfg.prune_after_s = 60.0
        await agent.handle(frame(), now=100.0)
        await agent.handle(speech(), now=101.0)
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "said-1"}}
        )
        await agent.on_server_event(
            {
                "type": "response.output_audio_transcript.done",
                "item_id": "said-1",
                "transcript": "wait, is that",
            }
        )
        await agent._cancel_response(102.0, why="barge-in")
        agent._clock_obj.set(400.0)
        await agent.tick(400.0)
        assert "wait, is that" in created_text(agent, "gpstb")

    async def test_a_text_session_has_nothing_to_replace(self, agent):
        """Its assistant item is already the words, at the text rate."""
        agent.cfg.output = "text"
        agent.cfg.prune_after_s = 60.0
        await agent.handle(frame(), now=100.0)
        await agent.handle(speech(), now=101.0)
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "said-1"}}
        )
        await agent.on_server_event(
            {"type": "response.output_text.done", "item_id": "said-1", "text": "nice"}
        )
        await agent.on_server_event(
            {"type": "response.done", "response": {"status": "completed", "usage": {}}}
        )
        agent._clock_obj.set(400.0)
        await agent.tick(400.0)
        assert deleted(agent), "the screenshots still go"
        assert "said-1" not in deleted(agent)


class TestConversationFull:
    """What happens once nothing is trimming the conversation but this client.

    `truncation = "disabled"` keeps the server's hands off the oldest end of
    the conversation, where the film's dialogue lives. The price is documented
    and paid here: the server refuses a response rather than shrinking one.
    """

    TOO_LONG = {
        "type": "invalid_request_error",
        "code": "conversation_too_long",
        "message": "Conversation is too long to create a Response.",
    }

    async def refuse(self, agent, err=None):
        await agent.on_server_event({"type": "error", "error": err or self.TOO_LONG})

    async def test_the_session_asks_for_no_server_truncation(self, agent):
        assert agent.session_update()["session"]["truncation"] == "disabled"

    async def test_the_ratio_is_still_available_to_anyone_who_wants_it(self, agent):
        agent.cfg.truncation = "retention_ratio"
        agent.cfg.truncation_retention_ratio = 0.6
        agent.cfg.truncation_post_instruction_tokens = 8000
        assert agent.session_update()["session"]["truncation"] == {
            "type": "retention_ratio",
            "retention_ratio": 0.6,
            "token_limits": {"post_instructions": 8000},
        }

    async def test_a_full_conversation_is_pruned_and_the_turn_asked_again(self, agent):
        agent.cfg.prune_after_s = 60.0
        await turns(agent, 3, every=40.0)
        await agent.handle(frame(seq=9), now=400.0)
        await agent.handle(speech(), now=401.0)
        before = len(agent.transport.sent_of_type("response.create"))
        await self.refuse(agent)

        assert deleted(agent), "room has to actually be made"
        assert len(agent.transport.sent_of_type("response.create")) == before + 1

    async def test_a_forced_round_runs_even_with_scheduled_pruning_off(self, agent):
        """Off means "do not pay the cache routinely", not "do not survive"."""
        assert agent.cfg.prune_after_s == 0.0
        await turns(agent, 3, every=40.0)
        await agent.handle(frame(seq=9), now=400.0)
        await agent.handle(speech(), now=401.0)
        await self.refuse(agent)
        assert deleted(agent)

    async def test_a_forced_round_never_touches_the_turn_it_is_making_room_for(self, agent):
        """Freeing room by deleting the question is asking a different question."""
        await turns(agent, 3, every=40.0)
        await agent.handle(frame(seq=9), now=400.0)
        await agent.handle(speech(), now=401.0)
        live = set(agent._live_items)
        await self.refuse(agent)
        assert live, "the turn must have items of its own"
        assert not live & set(deleted(agent))

    async def test_the_same_refusal_reported_twice_only_costs_one_retry(self, agent):
        """It arrives as an `error`, or on `response.done`, or both."""
        agent.cfg.prune_after_s = 60.0
        await turns(agent, 3, every=40.0)
        await agent.handle(frame(seq=9), now=400.0)
        await agent.handle(speech(), now=401.0)
        before = len(agent.transport.sent_of_type("response.create"))
        await self.refuse(agent)
        await agent.on_server_event(
            {
                "type": "response.done",
                "response": {"status": "failed", "status_details": {"error": self.TOO_LONG}},
            }
        )
        assert len(agent.transport.sent_of_type("response.create")) == before + 1

    async def test_a_refusal_that_frees_nothing_gives_up_rather_than_looping(self, agent):
        """The very first turn: everything in the conversation is the turn."""
        await agent.handle(frame(), now=100.0)
        await agent.handle(speech(), now=101.0)
        before = len(agent.transport.sent_of_type("response.create"))
        await self.refuse(agent)
        assert len(agent.transport.sent_of_type("response.create")) == before
        assert not agent._response_active, "the gate must not be left holding"
        assert [line for line in agent.lines if line.kind == "note"]

    async def test_a_session_with_nothing_left_to_drop_reopens(self, agent):
        """The same recovery the API's 60-minute cap already gets."""
        await agent.handle(frame(), now=100.0)
        await agent.handle(speech(), now=101.0)
        await self.refuse(agent)
        assert agent.transport._closed, "the pump reopens what it finds closed"
        assert [line for line in agent.lines if "reopening" in line.text]

    async def test_a_rate_limit_still_reads_as_a_rate_limit(self, agent):
        """The closest miss there is: also about size, but of a *window*.

        Pruning would not help it, and a spurious round throws away context and
        pays for a duplicate response to fix a problem that was not there.
        """
        await turns(agent, 3, every=40.0)
        await agent.handle(frame(seq=9), now=400.0)
        await agent.handle(speech(), now=401.0)
        before = len(deleted(agent))
        await self.refuse(
            agent,
            {
                "type": "invalid_request_error",
                "code": "rate_limit_exceeded",
                "message": "Rate limit reached: request too large for gpt-realtime",
            },
        )
        assert [line for line in agent.lines if line.kind == "error"], "it is still reported"
        assert len(deleted(agent)) == before

    async def test_an_unknown_code_can_be_named_in_the_config(self, agent):
        """The API does not document the one it actually sends."""
        agent.cfg.context_full_codes = ["something_we_have_not_seen"]
        await turns(agent, 3, every=40.0)
        await agent.handle(frame(seq=9), now=400.0)
        await agent.handle(speech(), now=401.0)
        await self.refuse(
            agent, {"type": "invalid_request_error", "code": "something_we_have_not_seen"}
        )
        assert deleted(agent)

    async def test_the_budget_forces_a_round_before_the_server_complains(self, agent):
        """Meeting the ceiling on our own terms costs a cache prefix; meeting
        it on the server's costs a turn the player was waiting for."""
        agent.cfg.context_budget_tokens = 500
        await turns(agent, 2, every=40.0)
        await agent.handle(frame(seq=9), now=400.0)
        await complete_response(agent, item_id="item-9")
        agent.meter.add({"input_tokens": 900})
        await agent.on_server_event(
            {
                "type": "response.done",
                "response": {"status": "completed", "usage": {"input_tokens": 900}},
            }
        )
        agent._clock_obj.set(410.0)
        await agent.tick(410.0)
        assert deleted(agent), "the budget is what noticed"
        assert [line for line in agent.lines if line.kind == "prune"]

    async def test_a_request_inside_the_budget_prunes_nothing(self, agent):
        agent.cfg.context_budget_tokens = 5000
        await turns(agent, 2, every=40.0)
        agent._clock_obj.set(410.0)
        await agent.tick(410.0)
        assert deleted(agent) == []


class TestUsageIsRecorded:
    async def test_response_done_feeds_the_meter(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        await complete_response(agent, audio_tokens=120)
        assert agent.meter.responses == 1
        assert agent.meter.out_audio == 120
        assert agent.report()["usage"]["spoken_s"] == 6.0


class TestTextOutput:
    """`agent.output = "text"`: the model writes the remark instead of saying it.

    Session-wide, not a rendering choice -- what changes is `output_modalities`
    on `session.update`, and everything downstream follows from the API putting
    the words on `response.output_text.*` instead of audio plus a transcript.
    """

    def text_agent(self, recorder=None, **agent_kwargs):
        clock = ReplayClock(100.0)
        cfg = make_config(output="text", **agent_kwargs)
        agent = CommentaryAgent(
            cfg, FakeTransport(), NullPlayer(clock), NullHud(),
            clock=clock, recorder=recorder,
        )
        agent._clock_obj = clock
        return agent

    async def write(self, agent, text, *, item_id="a1", delay=0.0):
        """The server's side of one written response, through to `response.done`."""
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": item_id}}
        )
        agent._clock_obj.set(agent._clock_obj.now + delay)
        await agent.on_server_event(
            {"type": "response.output_text.delta", "item_id": item_id, "delta": text[:4]}
        )
        await agent.on_server_event(
            {"type": "response.output_text.done", "item_id": item_id, "text": text}
        )
        await agent.on_server_event(
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "input_token_details": {"text_tokens": 100},
                        "output_tokens": 12,
                        "output_token_details": {"text_tokens": 12, "audio_tokens": 0},
                    },
                },
            }
        )

    # -- the session -------------------------------------------------------

    def test_the_session_asks_for_text(self):
        session = self.text_agent().session_update()["session"]
        assert session["output_modalities"] == ["text"]

    def test_a_voice_session_still_asks_for_audio(self, agent):
        assert agent.session_update()["session"]["output_modalities"] == ["audio"]

    def test_no_voice_is_configured_for_a_session_that_never_speaks(self):
        session = self.text_agent().session_update()["session"]
        assert "output" not in session["audio"]

    def test_the_player_is_still_listened_to_and_transcribed(self):
        audio_in = self.text_agent().session_update()["session"]["audio"]["input"]
        assert audio_in["format"] == {"type": "audio/pcm", "rate": 24000}
        assert audio_in["transcription"]["model"]

    def test_the_persona_says_it_is_being_read(self):
        instructions = self.text_agent().session_update()["session"]["instructions"]
        assert "not speaking out loud" in instructions
        assert "sitting on the couch" in instructions, "the persona itself is unchanged"

    def test_a_voice_session_is_told_none_of_that(self, agent):
        assert "not speaking out loud" not in agent.session_update()["session"]["instructions"]

    # -- the words ---------------------------------------------------------

    async def test_the_remark_reaches_the_hud(self):
        agent = self.text_agent()
        await agent.handle(frame(score=1.0), now=100.0)
        await self.write(agent, "that barrel again")
        assert agent.hud.shown == ["that barrel again"]

    async def test_it_is_logged_as_something_said(self):
        agent = self.text_agent()
        await agent.handle(frame(score=1.0), now=100.0)
        await self.write(agent, "that barrel again")
        said = [line for line in agent.lines if line.kind == "say"]
        assert [line.text for line in said] == ["that barrel again"]

    async def test_a_turn_is_sent_the_same_way_it_always_was(self):
        agent = self.text_agent()
        await agent.handle(pad("tapped A x3"), now=100.0)
        await agent.handle(frame(), now=100.5)
        await agent.handle(speech(dur_ms=1000), now=102.0)
        assert [e["type"] for e in agent.transport.sent] == [
            "conversation.item.create",
            "conversation.item.create",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "conversation.item.create",
            "response.create",
        ]

    async def test_a_cancelled_remark_is_never_shown(self):
        """`response.output_text.done` is emitted for interrupted responses too."""
        agent = self.text_agent()
        await agent.handle(speech(), now=100.0)
        await agent._cancel_response(100.5, why="test")
        await self.write(agent, "half a thought")
        assert agent.hud.shown == []

    # -- what a written remark costs the session ---------------------------

    async def test_nobody_is_talked_over_so_nothing_is_cancelled(self):
        """Barge-in exists to stop the agent speaking over someone; text cannot."""
        agent = self.text_agent()
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "a1"}}
        )
        agent._clock_obj.set(100.5)
        await agent.on_speech_start(now=100.5)
        assert agent.transport.sent_of_type("response.cancel") == []

        await self.write(agent, "still finishes the thought")
        assert agent.hud.shown == ["still finishes the thought"]

    async def test_the_reply_still_lands_after_it(self):
        agent = self.text_agent()
        await agent.handle(frame(score=1.0), now=100.0)
        await self.write(agent, "hm")
        agent._clock_obj.set(140.0)
        await agent.handle(speech(dur_ms=800), now=140.0)
        assert len(agent.transport.sent_of_type("response.create")) == 2

    async def test_a_remark_is_still_standing_while_it_is_being_read(self):
        """Reading time is what a written remark occupies, as speech occupies air.

        The visible consequence is the reply beat: it is measured from the end
        of the last response, and for text that end is when the player is done
        reading, not when the model finished writing.
        """
        from gpagent.agent.hud import hold_for

        agent = self.text_agent()
        remark = "that is the third time that exact barrel has ended a run " * 3
        await agent.handle(frame(score=1.0), now=100.0)
        await self.write(agent, remark)

        hold = hold_for(remark, agent.hud_cfg)
        assert hold > agent.hud_cfg.hold_min_s, "the fixture must be worth reading"

        # Still on screen: the answer waits rather than landing on top of it.
        asked_at = 100.0 + hold - 2.0
        agent._clock_obj.set(asked_at)
        await agent.handle(speech(dur_ms=800), now=asked_at)
        assert len(agent.transport.sent_of_type("response.create")) == 1
        assert agent.policy.declined.get("reply_spacing") == 1

        # Read, plus the beat: the question that stayed pending gets answered.
        answered_at = 100.0 + hold + agent.speak_cfg.reply_min_gap_s + 0.5
        agent._clock_obj.set(answered_at)
        await agent.tick(answered_at)
        assert len(agent.transport.sent_of_type("response.create")) == 2

    # -- the recording -----------------------------------------------------

    async def test_the_remark_is_recorded_as_text(self):
        recorded = []
        agent = self.text_agent(recorder=recorded.append)
        await agent.handle(frame(score=1.0), now=100.0)
        await self.write(agent, "that barrel again", delay=0.4)

        said = [e for e in recorded if isinstance(e, AgentResponse)]
        assert len(said) == 1
        assert said[0].modality == "text"
        assert said[0].transcript == "that barrel again"
        assert said[0].data is None and said[0].dur_ms == 0
        assert said[0].latency_ms > 0, "measured from the first delta, not the last"

    async def test_a_response_written_in_two_parts_is_recorded_whole(self):
        # One response can produce more than one output_text part: a model that
        # writes a preamble and then the remark sends two, and the player reads
        # both off the HUD. The recording has to say what they read.
        recorded = []
        agent = self.text_agent(recorder=recorded.append)
        await agent.handle(frame(score=1.0), now=100.0)
        await agent.on_server_event(
            {"type": "response.output_item.added", "item": {"id": "a1"}}
        )
        for part in ("Let me think.", "that barrel again"):
            await agent.on_server_event(
                {"type": "response.output_text.done", "item_id": "a1", "text": part}
            )
        await agent.on_server_event({"type": "response.done", "response": {"status": "completed"}})

        said = [e for e in recorded if isinstance(e, AgentResponse)]
        assert said[0].transcript == "Let me think. that barrel again"
        assert agent.hud.shown == ["Let me think.", "that barrel again"]

    async def test_text_output_tokens_are_metered(self):
        agent = self.text_agent()
        await agent.handle(frame(score=1.0), now=100.0)
        await self.write(agent, "that barrel again")
        assert agent.meter.out_text == 12
        assert agent.meter.out_audio == 0
        assert agent.report()["output"] == "text"
