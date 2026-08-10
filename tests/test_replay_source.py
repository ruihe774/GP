"""Replaying a recording must be indistinguishable from live capture."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from gpagent.bus import CaptureBus
from gpagent.events import GamepadActivity, ScreenFrame, SessionStart, SpeechSegment
from gpagent.replay import SIGNAL_INTENSITY, SIGNAL_SPEECH_START, ReplaySource, build_cues
from gpagent.sinks.jsonl import JsonlSink

SESS5 = Path(__file__).resolve().parent.parent / "sess5"


def write_session(directory) -> Path:
    """A tiny recording with one of everything that carries a payload."""
    with JsonlSink(directory) as sink:
        events = [
            SessionStart(t=0.0, seq=0, started_at="2026-08-09T00:00:00"),
            GamepadActivity(t=0.5, seq=1, summary="tapped A", intensity=0.8, apm=120),
            ScreenFrame(t=1.0, seq=2, w=1024, h=576, data=b"\xff\xd8jpeg", trigger="scene"),
            SpeechSegment(t=3.0, seq=3, dur_ms=2000, data=b"\x01\x02" * 100),
        ]
        for event in events:
            sink.write(event)
    return Path(directory)


class TestCueReconstruction:
    def test_speech_start_lands_at_the_start_of_the_utterance(self):
        """Not immediately before the segment.

        The segment event is emitted after the whole utterance plus hangover.
        Signalling at that moment would open and close the "player is talking"
        gate in the same instant and make barge-in impossible to reproduce.
        """
        segment = SpeechSegment(t=14.3, seq=7, dur_ms=3776)
        cues = build_cues([segment])
        onset = [c for c in cues if c[1] == SIGNAL_SPEECH_START][0]
        assert onset[0] == pytest.approx(14.3 - 3.776)
        assert cues.index(onset) < cues.index(
            [c for c in cues if c[1] == "event"][0]
        )

    def test_intensity_is_signalled_before_its_event(self):
        activity = GamepadActivity(t=5.0, seq=2, intensity=0.9, summary="holding RT")
        cues = build_cues([activity])
        assert [c[1] for c in cues] == [SIGNAL_INTENSITY, "event"]
        assert cues[0][2] == 0.9

    def test_signals_can_be_suppressed(self):
        cues = build_cues([SpeechSegment(t=2.0, seq=1, dur_ms=1000)], signals=False)
        assert [c[1] for c in cues] == ["event"]


class TestThroughTheBus:
    async def test_events_and_payloads_round_trip(self, tmp_path):
        directory = write_session(tmp_path / "sess")
        source = ReplaySource(directory, speed=0.0)
        bus = CaptureBus([source])

        seen = []
        signals = []
        await bus.start()
        bus.on_signal(SIGNAL_SPEECH_START, lambda _: signals.append("speech"))
        bus.on_signal(SIGNAL_INTENSITY, lambda v: signals.append(("intensity", v)))

        async def consume():
            async for event in bus:
                seen.append(event)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(source.finished.wait(), timeout=5)
        await bus.stop()
        await asyncio.wait_for(task, timeout=5)

        types = [e.TYPE for e in seen]
        assert types == [
            "session.start",
            "gamepad.activity",
            "screen.frame",
            "speech.segment",
        ]
        frame = [e for e in seen if isinstance(e, ScreenFrame)][0]
        assert frame.data == b"\xff\xd8jpeg", "blobs must be rehydrated"
        speech = [e for e in seen if isinstance(e, SpeechSegment)][0]
        assert speech.data == b"\x01\x02" * 100
        assert "speech" in signals

    async def test_describe_reports_progress(self, tmp_path):
        directory = write_session(tmp_path / "sess")
        source = ReplaySource(directory, speed=0.0)
        bus = CaptureBus([source])
        await bus.start()
        await asyncio.wait_for(source.finished.wait(), timeout=5)
        await bus.stop()
        assert source.describe()["events_emitted"] == 4

    async def test_an_empty_session_is_an_error_not_a_silent_no_op(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        (directory / "events.jsonl").write_text("")
        bus = CaptureBus([ReplaySource(directory)])
        with pytest.raises(RuntimeError):
            await bus.start()


@pytest.mark.skipif(not SESS5.exists(), reason="sess5 recording is not checked in")
class TestAgainstTheRealRecording:
    def test_cues_cover_the_whole_session(self):
        from gpagent.sinks.jsonl import read_session

        events = list(read_session(SESS5))
        cues = build_cues(events)
        assert len(cues) > len(events)
        times = [c[0] for c in cues]
        assert times == sorted(times), "cues must come out in order"
