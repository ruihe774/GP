"""Event codec, config overrides, and the session writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpagent.config import CaptureConfig
from gpagent.events import (
    GamepadActivity,
    ScreenFrame,
    SessionStart,
    SpeechSegment,
    decode,
)
from gpagent.sinks.jsonl import JsonlSink, read_session


class TestCodec:
    def test_round_trip_preserves_fields(self):
        original = GamepadActivity(
            t=1.25, seq=7, device="pad0", window_ms=500,
            buttons={"A": {"taps": 3, "held_ms": 0}},
            triggers={"LT": "idle", "RT": "full"},
            dpad="N", intensity=0.72, apm=84, summary="held RT, tapped A x3",
        )
        restored = decode(json.loads(json.dumps(original.to_dict())))
        assert restored.summary == original.summary
        assert restored.buttons == original.buttons
        assert restored.t == pytest.approx(original.t)
        assert restored.seq == original.seq

    def test_payload_is_not_inlined_by_default(self):
        event = SpeechSegment(t=1.0, seq=2, data=b"\x01\x02", dur_ms=100)
        assert "data_b64" not in event.to_dict()

    def test_inline_round_trips_payload(self):
        event = ScreenFrame(t=1.0, seq=2, data=b"\xff\xd8\xff\x00", w=8, h=6, trigger="speech")
        restored = decode(json.loads(json.dumps(event.to_dict(inline=True))))
        assert restored.data == b"\xff\xd8\xff\x00"
        assert restored.trigger == "speech"

    def test_sticks_omitted_unless_present(self):
        assert "sticks" not in GamepadActivity(summary="x").to_dict()
        event = GamepadActivity(sticks={"left": {"dir": "N", "mag": "full"}})
        assert "sticks" in event.to_dict()

    def test_unknown_type_is_rejected(self):
        with pytest.raises(ValueError):
            decode({"type": "nope.nope", "t": 0, "seq": 0})

    def test_unknown_fields_are_ignored(self):
        # Forward compatibility: a newer writer may add fields.
        restored = decode(
            {"type": "gamepad.activity", "t": 1.0, "seq": 1, "summary": "x", "future": 42}
        )
        assert restored.summary == "x"


class TestSink:
    def test_blobs_are_written_beside_the_index(self, tmp_path):
        with JsonlSink(tmp_path) as sink:
            sink.write(SessionStart(t=0, seq=0, started_at="now"))
            sink.write(SpeechSegment(t=1.0, seq=1, data=b"\x01\x02" * 10, dur_ms=100))
            sink.write(ScreenFrame(t=1.2, seq=2, data=b"\xff\xd8", w=100, h=56, trigger="speech"))

        assert (tmp_path / "events.jsonl").exists()
        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "blobs" / "000001-speech.pcm").read_bytes() == b"\x01\x02" * 10
        assert (tmp_path / "blobs" / "000002-frame.jpg").read_bytes() == b"\xff\xd8"

    def test_read_session_restores_events(self, tmp_path):
        with JsonlSink(tmp_path) as sink:
            sink.write(SpeechSegment(t=1.0, seq=1, data=b"abcd", dur_ms=100))

        events = list(read_session(tmp_path, load_blobs=True))
        assert len(events) == 1
        assert events[0].data == b"abcd"
        assert events[0].blob == "blobs/000001-speech.pcm"

    def test_metadata_only_read_skips_blobs(self, tmp_path):
        with JsonlSink(tmp_path) as sink:
            sink.write(SpeechSegment(t=1.0, seq=1, data=b"abcd", dur_ms=100))
        events = list(read_session(tmp_path, load_blobs=False))
        assert events[0].data is None
        assert events[0].dur_ms == 100

    def test_inline_mode_writes_no_blob_dir(self, tmp_path):
        with JsonlSink(tmp_path, inline=True) as sink:
            sink.write(ScreenFrame(t=1.0, seq=1, data=b"\xff\xd8", w=4, h=4, trigger="scene"))
        assert not (tmp_path / "blobs").exists()
        assert list(read_session(tmp_path))[0].data == b"\xff\xd8"

    def test_malformed_lines_are_skipped(self, tmp_path):
        with JsonlSink(tmp_path) as sink:
            sink.write(SpeechSegment(t=1.0, seq=1, dur_ms=100))
        with open(tmp_path / "events.jsonl", "a") as fh:
            fh.write("{not json\n")
        assert len(list(read_session(tmp_path))) == 1

    def test_manifest_records_counts(self, tmp_path):
        sink = JsonlSink(tmp_path)
        sink.open()
        sink.write(SpeechSegment(t=1.0, seq=1, dur_ms=100))
        sink.write(SpeechSegment(t=2.0, seq=2, dur_ms=100))
        sink.close({"duration_s": 5.0})
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["counts"]["speech.segment"] == 2
        assert manifest["duration_s"] == 5.0


class TestAgentResponseIsARecordableEvent:
    """A recorded conversation is only readable if the agent's half survives."""

    def event(self):
        from gpagent.events import AgentResponse

        return AgentResponse(
            t=12.5,
            seq=1_000_000,
            data=b"\x01\x02" * 100,
            reason="reply",
            transcript="that boss has too much health",
            dur_ms=1200,
            latency_ms=430,
            cut=True,
            usage={"output_tokens": 40},
        )

    def test_it_survives_a_session_round_trip(self, tmp_path):
        from gpagent.events import AgentResponse
        from gpagent.sinks.jsonl import JsonlSink, read_session

        with JsonlSink(tmp_path / "s") as sink:
            sink.write(self.event())
        back = list(read_session(tmp_path / "s", load_blobs=True))
        assert len(back) == 1
        got = back[0]
        assert isinstance(got, AgentResponse)
        assert got.data == b"\x01\x02" * 100
        assert got.transcript == "that boss has too much health"
        assert got.reason == "reply" and got.cut is True
        assert got.usage == {"output_tokens": 40}

    def test_its_audio_goes_to_a_named_blob(self, tmp_path):
        from gpagent.sinks.jsonl import JsonlSink

        with JsonlSink(tmp_path / "s") as sink:
            sink.write(self.event())
        blobs = [p.name for p in (tmp_path / "s" / "blobs").iterdir()]
        assert blobs == ["1000000-response.pcm"]


class TestConversationTrack:
    """`inspect --wav` lays both voices out on one timeline; it has to be the
    timeline they actually happened on.

    Both event types are stamped when they *finish*, so laying a clip down at
    its own `t` puts it a whole utterance late -- enough, in a real session,
    to make the agent look like it was talking over the player.
    """

    def track(self, tmp_path, events):
        import wave

        from gpagent.cli import _write_wavs

        _write_wavs(tmp_path, events)
        with wave.open(str(tmp_path / "wav" / "conversation.wav"), "rb") as fh:
            return fh.readframes(fh.getnframes()), fh.getframerate()

    def loud(self, seconds):
        return b"\x11\x11" * int(24000 * seconds)

    def test_a_clip_starts_where_the_player_started_talking(self, tmp_path):
        # Someone talks from 4.0 s to 6.0 s; the segment is stamped at 6.0.
        pcm, rate = self.track(
            tmp_path, [SpeechSegment(t=6.0, seq=1, data=self.loud(2.0), dur_ms=2000)]
        )
        first = next(i for i in range(0, len(pcm), 2) if pcm[i : i + 2] != b"\x00\x00")
        assert first / 2 / rate == pytest.approx(4.0, abs=0.05)

    def test_the_two_voices_do_not_land_on_top_of_each_other(self, tmp_path):
        from gpagent.events import AgentResponse

        # A question ending at 6.0 s, answered for 3 s from 7.0 s.
        pcm, rate = self.track(
            tmp_path,
            [
                SpeechSegment(t=6.0, seq=1, data=self.loud(2.0), dur_ms=2000),
                AgentResponse(t=10.0, seq=2, data=self.loud(3.0), dur_ms=3000),
            ],
        )
        voiced = [
            i / 2 / rate for i in range(0, len(pcm), 2) if pcm[i : i + 2] != b"\x00\x00"
        ]
        assert min(voiced) == pytest.approx(4.0, abs=0.05)
        assert max(voiced) == pytest.approx(10.0, abs=0.05)
        gap = [t for t in voiced if 6.1 < t < 6.9]
        assert not gap, "the pause between question and answer was filled in"


class TestExampleConfig:
    """`gpagent.example.toml` documents the defaults, so it must match them.

    It went stale without anyone noticing: two whole sections were missing and
    `[triggers]` still advertised numbers that had changed, which is worse than
    no example at all.
    """

    PATH = Path(__file__).resolve().parent.parent / "gpagent.example.toml"

    def loaded(self) -> dict:
        import tomllib

        with open(self.PATH, "rb") as fh:
            return tomllib.load(fh)

    def test_it_parses_and_every_key_is_real(self):
        from gpagent.config import CaptureConfig

        # from_dict raises on an unknown section or key, so this catches a
        # renamed field as well as a typo.
        CaptureConfig.from_dict(self.loaded())

    def test_every_config_section_is_documented(self):
        from dataclasses import fields

        from gpagent.config import CaptureConfig

        assert {f.name for f in fields(CaptureConfig)} <= set(self.loaded())

    def test_every_documented_value_is_the_current_default(self):
        from dataclasses import fields

        from gpagent.config import CaptureConfig

        defaults = CaptureConfig()
        stale = []
        for section, values in self.loaded().items():
            target = getattr(defaults, section)
            known = {f.name for f in fields(target)}
            for key, value in values.items():
                if key in known and getattr(target, key) != value:
                    stale.append(f"{section}.{key}: file={value!r} code={getattr(target, key)!r}")
        assert not stale, "gpagent.example.toml is out of date:\n  " + "\n  ".join(stale)


class TestConfig:
    def test_defaults_are_sane(self):
        cfg = CaptureConfig()
        assert cfg.gamepad.sticks_mode == "off", "moving must not drive captures"
        assert cfg.audio.vad_backend == "silero"
        assert cfg.audio.out_rate == 24000, "the Realtime API's rate"
        assert cfg.triggers.min_interval_s >= cfg.triggers.gamepad_throttle_s, (
            "the global floor must dominate the per-trigger throttles"
        )

    def test_overrides_apply(self):
        cfg = CaptureConfig()
        cfg.apply_overrides({"gamepad.sticks_mode": "full", "screen.long_edge": 1280})
        assert cfg.gamepad.sticks_mode == "full"
        assert cfg.screen.long_edge == 1280

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError):
            CaptureConfig().apply_overrides({"gamepad.nonexistent": 1})

    def test_unknown_section_is_rejected(self):
        with pytest.raises(ValueError):
            CaptureConfig().apply_overrides({"nope.key": 1})

    def test_toml_round_trip(self, tmp_path):
        path = tmp_path / "cfg.toml"
        path.write_text("[gamepad]\nsticks_mode = 'off'\n\n[audio]\nvad_backend = 'webrtc'\n")
        cfg = CaptureConfig.load(path)
        assert cfg.gamepad.sticks_mode == "off"
        assert cfg.audio.vad_backend == "webrtc"
        assert cfg.screen.long_edge == 1024, "unspecified sections keep defaults"

    def test_to_dict_is_json_serializable(self):
        json.dumps(CaptureConfig().to_dict())


class TestTokens:
    def test_audio_is_one_token_per_100ms(self):
        from gpagent.tokens import audio_tokens

        assert audio_tokens(1000) == 10
        assert audio_tokens(60_000) == 600  # one minute of speech

    def test_low_detail_images_are_flat(self):
        from gpagent.tokens import image_tokens

        assert image_tokens(1024, 576, detail="low") == 85

    def test_larger_images_cost_more(self):
        from gpagent.tokens import image_tokens

        assert image_tokens(1280, 720) > image_tokens(640, 360)

    def test_cost_uses_published_rates(self):
        from gpagent.tokens import Estimate

        cost = Estimate(audio_tokens=1_000_000).cost()
        assert cost["audio_usd"] == pytest.approx(32.0)
