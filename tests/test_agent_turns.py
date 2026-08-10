"""What the agent actually puts on the wire, against a fake transport.

As in `test_speak_policy.py`, the load-bearing assertions are positive ones: an
agent that sent nothing at all would satisfy "never talks over the player" and
"attaches at most one image" perfectly.
"""

from __future__ import annotations

import base64

import pytest

from gpagent.agent.playback import NullPlayer
from gpagent.agent.session import CommentaryAgent, ReplayClock
from gpagent.agent.transport import FakeTransport
from gpagent.config import CaptureConfig
from gpagent.events import GamepadActivity, ScreenFrame, SpeechSegment

JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
PCM = b"\x01\x02" * 12000  # 1 s at 24 kHz s16 mono


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
            "conversation.item.create",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
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

        items = agent.transport.sent_of_type("conversation.item.create")
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
        instructions = agent.transport.sent_of_type("response.create")[0]["response"][
            "instructions"
        ]
        assert "happened" in instructions.lower()

    async def test_the_persona_survives_the_per_response_override(self, agent):
        """`response.create.instructions` replaces the session's, not adds to it.

        The first live run against sess5 shipped the reason line alone and got
        four sentences of encouraging life coaching per turn, because the
        persona had been thrown away on every response.
        """
        agent.cfg.persona = "you are a lighthouse keeper"
        await agent.handle(frame(score=1.0), now=100.0)
        instructions = agent.transport.sent_of_type("response.create")[0]["response"][
            "instructions"
        ]
        assert "lighthouse keeper" in instructions
        assert "happened" in instructions.lower()

    async def test_output_length_is_capped(self, agent):
        await agent.handle(frame(score=1.0), now=100.0)
        response = agent.transport.sent_of_type("response.create")[0]["response"]
        assert response["max_output_tokens"] == agent.cfg.max_output_tokens

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
