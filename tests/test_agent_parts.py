"""Context accumulation, playback accounting, usage, key loading, SDK drift."""

from __future__ import annotations

import asyncio

import pytest

from gpagent.agent.config import AgentConfig
from gpagent.agent.context import ContextBuffer, Frame
from gpagent.agent.env import ENV_VAR, MissingAPIKey, load_api_key, load_dotenv
from gpagent.agent.playback import NullPlayer, PlaybackTimer
from gpagent.agent.transport import RecordingTransport
from gpagent.tokens import Estimate, UsageMeter, estimate_events, rates_for


def frame(seq=1, t=0.0, data=b"\xff\xd8jpeg"):
    return Frame(seq=seq, t=t, data=data, w=1024, h=576)


class TestContextBuffer:
    def test_summaries_are_flushed_as_one_block(self):
        buf = ContextBuffer(AgentConfig())
        buf.add_summary(1.0, "tapped A")
        buf.add_summary(1.5, "holding RT")
        ctx = buf.take(2.0)
        assert ctx.text == "[controller] tapped A; holding RT"

    def test_flushing_empties_the_buffer(self):
        buf = ContextBuffer(AgentConfig())
        buf.add_summary(1.0, "tapped A")
        buf.take(2.0)
        assert buf.take(2.5).text is None

    def test_repeated_summaries_collapse(self):
        buf = ContextBuffer(AgentConfig())
        for _ in range(5):
            buf.add_summary(1.0, "tapped A")
        assert buf.take(2.0).text == "[controller] tapped A (x5)"

    def test_stale_summaries_are_dropped(self):
        buf = ContextBuffer(AgentConfig(context_window_s=10.0))
        buf.add_summary(1.0, "ancient history")
        buf.add_summary(50.0, "just now")
        assert buf.take(51.0).text == "[controller] just now"

    def test_the_newest_summaries_survive_the_character_cap(self):
        buf = ContextBuffer(AgentConfig(max_summary_chars=40))
        for i in range(20):
            buf.add_summary(float(i), f"pressed button {i}")
        ctx = buf.take(20.0)
        assert "pressed button 19" in ctx.text
        assert "pressed button 0" not in ctx.text
        assert ctx.dropped_summaries > 0

    def test_a_frame_is_attached_once(self):
        buf = ContextBuffer(AgentConfig())
        buf.add_frame(frame(seq=3, t=1.0))
        assert buf.take(2.0).frame is not None
        assert buf.take(3.0).frame is None

    def test_a_newer_frame_replaces_an_unsent_one(self):
        buf = ContextBuffer(AgentConfig())
        buf.add_frame(frame(seq=3, t=1.0, data=b"old"))
        buf.add_frame(frame(seq=4, t=2.0, data=b"new"))
        assert buf.take(3.0).frame.data == b"new"
        assert buf.frames_seen == 2 and buf.frames_sent == 1

    def test_a_stale_frame_is_withheld(self):
        buf = ContextBuffer(AgentConfig(max_image_age_s=5.0))
        buf.add_frame(frame(t=1.0))
        assert buf.take(100.0).frame is None

    def test_images_can_be_declined_for_this_turn(self):
        buf = ContextBuffer(AgentConfig())
        buf.add_frame(frame(t=1.0))
        assert buf.take(2.0, with_image=False).frame is None
        assert buf.take(2.0).frame is not None, "and are still available later"


class TestPlaybackTimer:
    def test_it_reports_what_has_been_heard_so_far(self):
        timer = PlaybackTimer(lambda: 0.0)
        timer.push(24000 * 2 * 4, now=100.0)  # 4 s of 24 kHz s16 mono
        assert timer.pushed_ms == pytest.approx(4000.0)
        assert timer.played_ms(now=101.5) == pytest.approx(1500.0)

    def test_it_never_claims_more_was_heard_than_was_queued(self):
        timer = PlaybackTimer(lambda: 0.0)
        timer.push(24000 * 2, now=100.0)  # 1 s
        assert timer.played_ms(now=200.0) == pytest.approx(1000.0)

    def test_cutting_short_returns_what_was_heard(self):
        timer = PlaybackTimer(lambda: 0.0)
        timer.push(24000 * 2 * 4, now=100.0)
        assert timer.reset(now=101.0) == pytest.approx(1000.0)
        assert timer.remaining_ms(now=101.0) == 0.0

    def test_completing_counts_all_of_it(self):
        timer = PlaybackTimer(lambda: 0.0)
        timer.push(24000 * 2 * 4, now=100.0)
        assert timer.complete() == pytest.approx(4000.0)

    async def test_the_null_player_simulates_an_utterance_taking_time(self):
        clock = _Clock(100.0)
        player = NullPlayer(clock)
        player.simulate(3000.0)
        assert player.timer.remaining_ms() == pytest.approx(3000.0)
        clock.now = 101.0
        assert player.timer.remaining_ms() == pytest.approx(2000.0)
        clock.now = 105.0
        assert player.timer.remaining_ms() == 0.0


class TestSampleAlignment:
    """One odd-length chunk turns the rest of an utterance into white noise."""

    def test_an_odd_chunk_carries_its_last_byte_forward(self):
        from gpagent.agent.playback import align_samples

        data, carry = align_samples(b"", b"\x01\x02\x03")
        assert data == b"\x01\x02"
        assert carry == b"\x03"

    def test_the_carried_byte_leads_the_next_chunk(self):
        from gpagent.agent.playback import align_samples

        data, carry = align_samples(b"\x03", b"\x04\x05\x06")
        assert data == b"\x03\x04\x05\x06"
        assert carry == b""

    def test_a_stream_of_odd_chunks_reassembles_exactly(self):
        from gpagent.agent.playback import align_samples

        original = bytes(range(256)) * 4
        chunks = []
        pos = 0
        for size in (7, 13, 1, 99, 3, 5):  # all odd
            chunks.append(original[pos : pos + size])
            pos += size
        chunks.append(original[pos:])

        played, carry = b"", b""
        for chunk in chunks:
            data, carry = align_samples(carry, chunk)
            played += data
        assert played + carry == original
        assert len(played) % 2 == 0


class TestPlaybackTimestamps:
    """Buffers must be stamped, and stamped ahead of the running clock.

    Unstamped buffers in a `format=time` appsrc are rendered back to back
    against a clock that has been running since startup: heard as chunks and
    gaps rather than speech.
    """

    def player(self, clock_ns=5_000_000_000):
        from gpagent.agent.playback import AudioPlayer

        player = AudioPlayer()
        player._gst = _FakeGst()
        player._pipeline = _FakePipeline(clock_ns)
        player._src = _FakeSrc()
        return player

    def test_the_first_buffer_starts_ahead_of_the_clock(self):
        from gpagent.agent.playback import LEAD_IN_MS

        player = self.player(clock_ns=5_000_000_000)
        player.push(b"\x00\x00" * 2400)  # 100 ms
        pts = player._src.pushed[0].pts
        assert pts == 5_000_000_000 + LEAD_IN_MS * 1_000_000

    def test_buffers_within_an_utterance_are_gapless(self):
        player = self.player()
        for _ in range(4):
            player.push(b"\x00\x00" * 2400)  # 100 ms each
        pushed = player._src.pushed
        for a, b in zip(pushed, pushed[1:]):
            assert b.pts == a.pts + a.duration
        assert all(b.duration == 100_000_000 for b in pushed)

    def test_the_lead_in_is_a_usable_jitter_buffer(self):
        """Too small and any stall in generation is a gap mid-word."""
        from gpagent.agent.playback import LEAD_IN_MS

        assert LEAD_IN_MS >= 250

    def test_a_new_utterance_resyncs_to_the_clock(self):
        player = self.player(clock_ns=5_000_000_000)
        player.push(b"\x00\x00" * 2400)
        first = player._src.pushed[0].pts
        player.flush()
        player._pipeline.now = 30_000_000_000  # 25 s of silence later
        player.push(b"\x00\x00" * 2400)
        assert player._src.pushed[-1].pts > first + 20_000_000_000

    def test_odd_chunks_do_not_reach_the_sink_misaligned(self):
        player = self.player()
        player.push(b"\x01" * 4801)
        player.push(b"\x02" * 4801)
        assert all(len(b.data) % 2 == 0 for b in player._src.pushed)
        joined = b"".join(b.data for b in player._src.pushed)
        assert joined == b"\x01" * 4801 + b"\x02" * 4801


class _FakeBuffer:
    def __init__(self, size):
        self.data = b"\x00" * size
        self.pts = None
        self.duration = None

    def fill(self, offset, data):
        self.data = data


class _FakeGst:
    SECOND = 1_000_000_000
    MSECOND = 1_000_000

    class Buffer:
        @staticmethod
        def new_allocate(_allocator, size, _params):
            return _FakeBuffer(size)

    class Event:
        @staticmethod
        def new_flush_start():
            return "flush-start"

        @staticmethod
        def new_flush_stop(reset):
            return "flush-stop"


class _FakeClock:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def get_time(self):
        return self.pipeline.now


class _FakePipeline:
    def __init__(self, now):
        self.now = now

    def get_clock(self):
        return _FakeClock(self)

    def get_base_time(self):
        return 0


class _FakeSrc:
    def __init__(self):
        self.pushed = []

    def emit(self, _signal, buffer):
        self.pushed.append(buffer)

    def send_event(self, _event):
        return True


class _Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class TestUsageMeter:
    def usage(self, **kwargs):
        base = {
            "input_tokens": 900,
            "input_token_details": {
                "text_tokens": 100,
                "audio_tokens": 35,
                "image_tokens": 765,
                "cached_tokens": 0,
            },
            "output_tokens": 60,
            "output_token_details": {"text_tokens": 0, "audio_tokens": 60},
        }
        base.update(kwargs)
        return base

    def test_it_adds_up_what_was_billed(self):
        meter = UsageMeter(model="gpt-realtime-2.1")
        meter.add(self.usage())
        meter.add(self.usage())
        assert meter.responses == 2
        assert meter.in_image == 1530
        assert meter.out_audio == 120
        assert meter.cost()["out_audio_usd"] == pytest.approx(120 * 64 / 1e6)

    def test_cached_tokens_are_a_subset_not_an_extra(self):
        """Charging them twice would overstate exactly the cheapest tokens."""
        meter = UsageMeter(model="gpt-realtime-2.1")
        meter.add(
            self.usage(
                input_token_details={
                    "text_tokens": 1000,
                    "audio_tokens": 0,
                    "image_tokens": 0,
                    "cached_tokens": 800,
                    "cached_tokens_details": {"text_tokens": 800},
                }
            )
        )
        cost = meter.cost()
        assert cost["in_text_usd"] == pytest.approx(200 * 4 / 1e6)
        assert cost["in_cached_usd"] == pytest.approx(800 * 0.4 / 1e6)

    def test_an_input_total_with_no_breakdown_is_still_charged(self):
        meter = UsageMeter()
        meter.add({"input_tokens": 500, "output_tokens": 0})
        assert meter.unattributed_input == 500
        assert meter.cost()["total_usd"] > 0

    def test_spoken_time_comes_from_output_audio_tokens(self):
        meter = UsageMeter()
        meter.add(self.usage(output_token_details={"audio_tokens": 100}))
        assert meter.spoken_ms == pytest.approx(5000.0)

    def test_mini_is_cheaper_than_the_flagship(self):
        assert rates_for("gpt-realtime-2.1-mini").audio_out_per_mtok < rates_for(
            "gpt-realtime-2.1"
        ).audio_out_per_mtok

    def test_an_unknown_model_falls_back_to_full_price(self):
        assert rates_for("gpt-realtime-9") == rates_for("gpt-realtime-2.1")

    def test_empty_usage_is_ignored(self):
        meter = UsageMeter()
        meter.add(None)
        assert meter.responses == 0


class TestEstimateEvents:
    def test_it_matches_what_inspect_reports(self):
        from gpagent.events import GamepadActivity, ScreenFrame, SpeechSegment

        events = [
            GamepadActivity(summary="tapped A"),
            SpeechSegment(dur_ms=1000),
            ScreenFrame(w=1024, h=576),
        ]
        estimate = estimate_events(events)
        assert estimate.audio_tokens == 10
        assert estimate.frames == 1
        assert estimate.image_tokens == 765
        assert isinstance(estimate, Estimate)


class TestApiKey:
    def test_the_environment_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, "from-env")
        assert load_api_key(tmp_path / ".env") == "from-env"

    def test_it_falls_back_to_dotenv(self, monkeypatch, tmp_path):
        monkeypatch.delenv(ENV_VAR, raising=False)
        path = tmp_path / ".env"
        path.write_text(f'# a comment\n{ENV_VAR}="from-file"\n')
        assert load_api_key(path) == "from-file"

    def test_export_prefixes_and_blank_lines_are_tolerated(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text(f"\nexport {ENV_VAR}=abc\nOTHER=1\n")
        assert load_dotenv(path) == {ENV_VAR: "abc", "OTHER": "1"}

    def test_the_key_does_not_print_itself(self, monkeypatch, tmp_path):
        """A key in an ordinary str leaks through any traceback holding it."""
        monkeypatch.setenv(ENV_VAR, "sk-proj-supersecret")
        key = load_api_key(tmp_path / ".env")
        assert "supersecret" not in repr(key)
        assert "supersecret" not in f"{key}"
        assert "supersecret" not in repr({"k": key})
        assert key.reveal() == "sk-proj-supersecret", "it must still be usable"

    def test_the_transport_does_not_print_it_either(self):
        from gpagent.agent.transport import OpenAITransport

        transport = OpenAITransport("gpt-realtime-2.1-mini", "sk-proj-supersecret")
        assert "supersecret" not in repr(transport)

    def test_the_error_names_the_variable_and_nothing_else(self, monkeypatch, tmp_path):
        monkeypatch.delenv(ENV_VAR, raising=False)
        path = tmp_path / ".env"
        path.write_text("SOMETHING_ELSE=secret-value\n")
        with pytest.raises(MissingAPIKey) as exc:
            load_api_key(path)
        assert ENV_VAR in str(exc.value)
        assert "secret-value" not in str(exc.value)


class TestSdkPayloadDrift:
    """`connection.send(dict)` drops keys the installed SDK does not know.

    Every payload this package sends is checked against the SDK's own transform,
    so an upgrade that starts eating a field fails here loudly instead of
    quietly degrading the agent (a dropped `turn_detection: null` would hand
    turn-taking back to the server; a dropped `detail` would 9x the image bill).
    """

    def payloads(self):
        from gpagent.agent.playback import NullPlayer
        from gpagent.agent.session import CommentaryAgent
        from gpagent.agent.transport import FakeTransport
        from gpagent.config import CaptureConfig

        agent = CommentaryAgent(CaptureConfig(), FakeTransport(), NullPlayer())
        ctx = type("Ctx", (), {"text": "hello", "frame": frame()})()
        return [
            agent.session_update(),
            agent._context_item(ctx),
            {"type": "input_audio_buffer.append", "audio": "AAAA"},
            {"type": "input_audio_buffer.commit"},
            {"type": "response.create", "response": {"instructions": "be brief"}},
            {"type": "response.cancel"},
            {
                "type": "conversation.item.truncate",
                "item_id": "i",
                "content_index": 0,
                "audio_end_ms": 1200,
            },
            {"type": "conversation.item.delete", "item_id": "i"},
        ]

    def test_nothing_we_send_is_dropped_by_the_sdk(self):
        from openai._utils import async_maybe_transform
        from openai.types.realtime.realtime_client_event_param import RealtimeClientEventParam

        for payload in self.payloads():
            transformed = asyncio.run(async_maybe_transform(payload, RealtimeClientEventParam))
            assert transformed == payload, f"SDK dropped fields from {payload['type']}"


class TestRecordingTransport:
    async def test_it_answers_every_request(self):
        """A dry run that dropped requests would leave the agent waiting forever."""
        transport = RecordingTransport()
        await transport.connect()
        await transport.send({"type": "response.create", "response": {}})
        received = []

        async def drain():
            async for event in transport:
                received.append(event["type"])
                if event["type"] == "response.done":
                    break

        await asyncio.wait_for(drain(), timeout=5)
        assert "response.done" in received
        await transport.close()

    async def test_it_estimates_usage_from_what_was_sent(self):
        transport = RecordingTransport()
        await transport.connect()
        await transport.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/jpeg;base64,AA", "detail": "low"}
                    ],
                },
            }
        )
        await transport.send({"type": "response.create", "response": {}})
        done = None
        async for event in transport:
            if event["type"] == "response.done":
                done = event
                break
        details = done["response"]["usage"]["input_token_details"]
        assert details["image_tokens"] == 85, "detail=low must be priced as low"
        await transport.close()
