"""Context accumulation, playback accounting, usage, key loading, SDK drift."""

from __future__ import annotations

import asyncio

import pytest

from gpagent.agent.config import AgentConfig
from gpagent.agent.context import ContextBuffer, Frame, TurnContext
from gpagent.agent.env import ENV_VAR, MissingAPIKey, load_api_key, load_dotenv
from gpagent.agent.playback import NullPlayer, PlaybackTimer
from gpagent.agent.transport import RecordingTransport
from gpagent.tokens import Estimate, UsageMeter, estimate_events, rates_for


def frame(seq=1, t=0.0, data=b"\xff\xd8jpeg", score=0.0):
    return Frame(seq=seq, t=t, data=data, w=1024, h=576, scene_score=score)


def trail_times(ctx):
    return [f.t for f in ctx.trail]


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

    def test_earlier_frames_ride_along_as_a_trail(self):
        buf = ContextBuffer(AgentConfig(image_trail=4))
        for i in range(3):
            buf.add_frame(frame(seq=i, t=2.0 * i))
        ctx = buf.take(4.0)
        assert ctx.frame.t == 4.0, "the current screen is still the newest frame"
        assert trail_times(ctx) == [0.0, 2.0], "and the trail is older, oldest first"
        assert buf.trail_sent == 2

    def test_the_trail_spans_the_window_instead_of_the_last_moment(self):
        """The newest N frames are the wrong N.

        Capture triggers throttle rather than debounce, so a burst clusters
        frames into a second or two. Taking the most recent ones describes that
        second four times over and says nothing about where the player came
        from, which is the whole point of sending more than one.
        """
        buf = ContextBuffer(AgentConfig(image_trail=2))
        for i, t in enumerate([0.0, 2.0, 6.0, 8.0]):
            buf.add_frame(frame(seq=i, t=t))
        buf.add_frame(frame(seq=9, t=10.0))
        assert trail_times(buf.take(10.0)) == [2.0, 8.0], "not [6.0, 8.0], the newest two"

    def test_the_frame_that_changed_the_screen_wins_its_slot(self):
        buf = ContextBuffer(AgentConfig(image_trail=2))
        buf.add_frame(frame(seq=0, t=0.0, score=0.9))
        buf.add_frame(frame(seq=1, t=2.0, score=0.1))
        buf.add_frame(frame(seq=2, t=6.0, score=0.1))
        buf.add_frame(frame(seq=3, t=8.0, score=0.7))
        buf.add_frame(frame(seq=9, t=10.0))
        assert trail_times(buf.take(10.0)) == [0.0, 8.0]

    def test_a_frame_is_never_paid_for_twice(self):
        buf = ContextBuffer(AgentConfig(image_trail=4))
        buf.add_frame(frame(seq=0, t=0.0))
        buf.add_frame(frame(seq=1, t=2.0))
        first = buf.take(3.0)
        assert trail_times(first) == [0.0]

        buf.add_frame(frame(seq=2, t=4.0))
        second = buf.take(5.0)
        assert second.frame.seq == 2
        assert second.trail == [], "everything older is already in the conversation"

    def test_the_trail_only_covers_ground_the_model_has_not_seen(self):
        """Candidates start after the newest frame already sent, not merely at
        "frames not yet sent individually".

        A frame from between two images already in the conversation carries
        almost no news, and the slot it takes is one that could have covered the
        stretch nobody has seen.
        """
        buf = ContextBuffer(AgentConfig(image_trail=1))
        buf.add_frame(frame(seq=0, t=0.0, score=0.9))
        buf.add_frame(frame(seq=1, t=2.0, score=0.1))
        buf.add_frame(frame(seq=2, t=4.0))
        assert trail_times(buf.take(4.0)) == [0.0]

        buf.add_frame(frame(seq=3, t=6.0, score=0.9))
        buf.add_frame(frame(seq=4, t=8.0))
        assert trail_times(buf.take(8.0)) == [6.0], "not t=2.0, which is behind the last send"

    def test_frames_survive_a_turn_that_sent_no_image(self):
        buf = ContextBuffer(AgentConfig(image_trail=4))
        buf.add_frame(frame(seq=0, t=0.0))
        buf.add_frame(frame(seq=1, t=2.0))
        buf.take(3.0, with_image=False)

        buf.add_frame(frame(seq=2, t=4.0))
        assert trail_times(buf.take(4.0)) == [0.0, 2.0], "nothing was covered, so nothing expired"

    def test_a_short_window_yields_a_short_trail_not_a_redundant_one(self):
        """Bucketing spreads picks across the window it is handed; it does not
        ask whether the window is worth spreading. Four frames of the same three
        seconds is one moment sent four times, at 85 tokens each."""
        buf = ContextBuffer(AgentConfig(image_trail=4, image_trail_min_gap_s=1.5))
        for i, t in enumerate([0.0, 0.4, 0.8, 1.2, 1.6, 2.0]):
            buf.add_frame(frame(seq=i, t=t))
        buf.add_frame(frame(seq=9, t=2.4))
        assert trail_times(buf.take(2.4)) == [0.0], "one frame, not four of the same moment"

    def test_a_frame_that_all_but_duplicates_the_current_one_is_not_a_candidate(self):
        buf = ContextBuffer(AgentConfig(image_trail=4, image_trail_min_gap_s=1.5))
        buf.add_frame(frame(seq=0, t=0.0))
        buf.add_frame(frame(seq=1, t=9.9))
        buf.add_frame(frame(seq=2, t=10.0))
        assert trail_times(buf.take(10.0)) == [0.0], "t=9.9 is 0.1s before the current frame"

    def test_backfilled_slots_obey_the_spacing_too(self):
        """The gap the backfill fills is the one it is most likely to violate:
        an empty bucket hands its slot to the best leftover anywhere, which is
        often the frame next door to one already picked."""
        buf = ContextBuffer(AgentConfig(image_trail=4, image_trail_min_gap_s=1.5))
        buf.cfg.image_trail = 3
        for i, (t, score) in enumerate([(0.0, 0.5), (7.0, 0.8), (7.2, 0.9), (7.4, 0.1)]):
            buf.add_frame(frame(seq=i, t=t, score=score))
        buf.add_frame(frame(seq=9, t=10.0))
        # The middle bucket is empty, so a slot goes begging; every leftover
        # that could fill it sits beside the pick that won its own bucket.
        assert trail_times(buf.take(10.0)) == [0.0, 7.2]

    def test_the_spacing_floor_can_be_turned_off(self):
        buf = ContextBuffer(AgentConfig(image_trail=2, image_trail_min_gap_s=0.0))
        buf.add_frame(frame(seq=0, t=0.0))
        buf.add_frame(frame(seq=1, t=0.1))
        buf.add_frame(frame(seq=2, t=0.2))
        assert trail_times(buf.take(0.2)) == [0.0, 0.1]

    def test_the_trail_has_its_own_age_horizon(self):
        buf = ContextBuffer(AgentConfig(image_trail=4, image_trail_max_age_s=5.0))
        buf.add_frame(frame(seq=0, t=0.0))
        buf.add_frame(frame(seq=1, t=8.0))
        buf.add_frame(frame(seq=2, t=10.0))
        assert trail_times(buf.take(10.0)) == [8.0], "t=0.0 is past the horizon"

    def test_the_trail_can_be_turned_off(self):
        buf = ContextBuffer(AgentConfig(image_trail=0))
        buf.add_frame(frame(seq=0, t=0.0))
        buf.add_frame(frame(seq=1, t=1.0))
        ctx = buf.take(2.0)
        assert ctx.frame is not None and ctx.trail == []
        assert buf.trail_sent == 0

    def test_no_trail_rides_along_when_no_frame_does(self):
        buf = ContextBuffer(AgentConfig(image_trail=4))
        buf.add_frame(frame(seq=0, t=0.0))
        buf.add_frame(frame(seq=1, t=1.0))
        ctx = buf.take(2.0, with_image=False)
        assert ctx.frame is None and ctx.trail == []

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

    def test_a_new_utterance_is_measured_on_its_own(self):
        """Discard drops the last sentence off the clock without counting it.

        Offline, `_finish_response` re-arms the timer with `simulate()` so the
        agent can tell it is mid-sentence. Without a discard at the next turn,
        the following utterance accumulated on top and a barge-in truncated
        with a `heard_ms` covering both -- which the server rejects, so the
        model never learns it was cut off.
        """
        timer = PlaybackTimer(lambda: 0.0)
        timer.push(24000 * 2 * 4, now=100.0)  # last turn, still on the clock
        timer.discard()
        timer.push(24000 * 2, now=110.0)  # this turn: 1 s
        assert timer.pushed_ms == pytest.approx(1000.0)
        assert timer.reset(now=110.4) == pytest.approx(400.0)

    def test_discarding_does_not_count_as_spoken(self):
        timer = PlaybackTimer(lambda: 0.0)
        timer.push(24000 * 2 * 4, now=100.0)
        timer.complete()
        timer.push(24000 * 2 * 4, now=104.0)  # the simulate() re-arm
        timer.discard()
        assert timer.total_spoken_ms == pytest.approx(4000.0)

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
        for a, b in zip(pushed, pushed[1:], strict=False):
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
        # A real context, trail included: an item carrying several images at
        # mixed detail is the shape the SDK now has to survive.
        ctx = TurnContext(
            text="hello",
            frame=frame(seq=3, t=2.0),
            trail=[frame(seq=1, t=0.0), frame(seq=2, t=1.0)],
        )
        return [
            agent.session_update(),
            agent._text_item(0.0, "hello"),
            agent._images_item(0.0, ctx),
            agent._reason_item(0.0, "reply"),
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

    def test_a_text_session_survives_the_transform(self):
        """`output_modalities` is the whole feature; a dropped one is a mute HUD."""
        from openai._utils import async_maybe_transform
        from openai.types.realtime.realtime_client_event_param import RealtimeClientEventParam

        from gpagent.agent.playback import NullPlayer
        from gpagent.agent.session import CommentaryAgent
        from gpagent.agent.transport import FakeTransport
        from gpagent.config import CaptureConfig

        cfg = CaptureConfig()
        cfg.agent.output = "text"
        agent = CommentaryAgent(cfg, FakeTransport(), NullPlayer())
        payload = agent.session_update()
        assert payload["session"]["output_modalities"] == ["text"]
        transformed = asyncio.run(async_maybe_transform(payload, RealtimeClientEventParam))
        assert transformed == payload, "SDK dropped fields from a text session.update"

    def test_reasoning_effort_survives_the_transform(self):
        from openai._utils import async_maybe_transform
        from openai.types.realtime.realtime_client_event_param import RealtimeClientEventParam

        from gpagent.agent.playback import NullPlayer
        from gpagent.agent.session import CommentaryAgent
        from gpagent.agent.transport import FakeTransport
        from gpagent.config import CaptureConfig

        cfg = CaptureConfig()
        cfg.agent.reasoning_effort = "high"
        agent = CommentaryAgent(cfg, FakeTransport(), NullPlayer())
        payload = agent.session_update()
        assert payload["session"]["reasoning"] == {"effort": "high"}
        transformed = asyncio.run(async_maybe_transform(payload, RealtimeClientEventParam))
        assert transformed == payload, "SDK dropped the reasoning field"


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
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64,AA",
                            "detail": "low",
                        }
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

    async def image_item(self, transport, item_id, detail="high"):
        await transport.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64,AA",
                            "detail": detail,
                        }
                    ],
                },
            }
        )

    async def usage(self, transport):
        """Ask for a response and return the usage block it comes back with."""
        await transport.send({"type": "response.create", "response": {}})
        async for event in transport:
            if event["type"] == "response.done":
                return event["response"]["usage"]
        raise AssertionError("the dry run must answer every request")

    async def test_the_whole_conversation_is_billed_on_every_response(self):
        """The API re-bills history, and history is the thing being tuned.

        A dry run that charged only for the turn's own items would show a flat
        cost per response and hide the growth curve that hits TPM -- the exact
        thing `agent.prune_after_s` exists to bend.
        """
        transport = RecordingTransport()
        await transport.connect()
        await self.image_item(transport, "img-1", detail="low")
        first = await self.usage(transport)
        await self.image_item(transport, "img-2", detail="low")
        second = await self.usage(transport)
        assert second["input_token_details"]["image_tokens"] == 170, "both frames, again"
        assert second["input_tokens"] > first["input_tokens"]
        await transport.close()

    async def test_a_deleted_item_stops_being_billed(self):
        transport = RecordingTransport()
        await transport.connect()
        await self.image_item(transport, "img-1", detail="low")
        await self.image_item(transport, "img-2", detail="low")
        await self.usage(transport)
        await transport.send({"type": "conversation.item.delete", "item_id": "img-1"})
        after = await self.usage(transport)
        assert after["input_token_details"]["image_tokens"] == 85, "one frame left"
        await transport.close()

    async def test_a_delete_truncates_the_cached_prefix(self):
        """Deleting from the oldest end costs most of the cache, once."""
        transport = RecordingTransport()
        await transport.connect()
        await self.image_item(transport, "img-1")
        await self.image_item(transport, "img-2")
        steady = await self.usage(transport)
        assert (await self.usage(transport))["input_token_details"]["cached_tokens"] > 0
        await transport.send({"type": "conversation.item.delete", "item_id": "img-1"})
        after = await self.usage(transport)
        assert after["input_token_details"]["cached_tokens"] == 0
        # ...and the cache refills on the very next request, which is why the
        # cost of pruning is per round rather than per item.
        assert (await self.usage(transport))["input_token_details"]["cached_tokens"] > 0
        assert steady["input_tokens"] > 0
        await transport.close()

    async def test_a_text_dry_run_is_answered_in_text(self):
        """A dry run must exercise the path the real session will take."""
        transport = RecordingTransport(text=True)
        await transport.connect()
        await transport.send({"type": "response.create", "response": {}})
        received = []
        async for event in transport:
            received.append(event)
            if event["type"] == "response.done":
                break
        kinds = [e["type"] for e in received]
        assert "response.output_text.done" in kinds
        assert "response.output_audio_transcript.done" not in kinds
        out = received[-1]["response"]["usage"]["output_token_details"]
        assert out["text_tokens"] > 0 and out["audio_tokens"] == 0
        await transport.close()
