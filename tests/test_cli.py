"""Replay should apply the config a session was recorded with, unless overridden."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pytest

from gpagent.cli import _build_config, _write_sent_sheet
from gpagent.sinks.jsonl import JsonlSink


def _args(**overrides) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        replay=None,
        config=None,
        set=[],
        no_gamepad=False,
        no_audio=False,
        no_screen=False,
        pick_screen=False,
        map=[],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_session_with_config(directory, config: dict) -> None:
    sink = JsonlSink(directory)
    sink.open()
    sink.close({"config": config})


class TestBuildConfigReplayLayer:
    def test_replay_applies_recorded_config(self, tmp_path):
        _write_session_with_config(
            tmp_path, {"gamepad": {"sticks_mode": "full"}, "audio": {"vad_backend": "webrtc"}}
        )
        cfg = _build_config(_args(replay=str(tmp_path)))
        assert cfg.gamepad.sticks_mode == "full"
        assert cfg.audio.vad_backend == "webrtc"

    def test_explicit_set_overrides_recorded_config(self, tmp_path):
        _write_session_with_config(tmp_path, {"gamepad": {"sticks_mode": "full"}})
        cfg = _build_config(_args(replay=str(tmp_path), set=["gamepad.sticks_mode=off"]))
        assert cfg.gamepad.sticks_mode == "off"

    def test_explicit_toml_overrides_recorded_config(self, tmp_path):
        _write_session_with_config(tmp_path, {"gamepad": {"sticks_mode": "full"}})
        toml_path = tmp_path / "override.toml"
        toml_path.write_text("[gamepad]\nsticks_mode = 'off'\n")
        cfg = _build_config(_args(replay=str(tmp_path), config=str(toml_path)))
        assert cfg.gamepad.sticks_mode == "off"

    def test_missing_manifest_falls_back_to_defaults(self, tmp_path):
        cfg = _build_config(_args(replay=str(tmp_path)))
        assert cfg.gamepad.sticks_mode == "off", "no manifest.json: defaults, no error"

    def test_manifest_without_config_key_falls_back_to_defaults(self, tmp_path):
        sink = JsonlSink(tmp_path)
        sink.open()
        sink.close({"duration_s": 1.0})
        cfg = _build_config(_args(replay=str(tmp_path)))
        assert cfg.gamepad.sticks_mode == "off"

    def test_a_setting_this_version_dropped_still_replays(self, tmp_path, capsys):
        """A manifest records what a past run used, it does not request anything.

        `agent.keep_images` was removed when pruning moved to one shared cutoff.
        Every session recorded before that has it, and refusing to replay them
        would make the recordings useless for measuring the change that removed
        it -- which is the one thing they are most needed for.
        """
        _write_session_with_config(tmp_path, {"gamepad": {"nonexistent": 1, "enabled": False}})
        cfg = _build_config(_args(replay=str(tmp_path)))
        assert cfg.gamepad.enabled is False, "the keys it does know must still apply"
        assert "gamepad.nonexistent" in capsys.readouterr().out, "and say what it ignored"

    def test_an_unknown_key_in_a_config_file_is_still_an_error(self, tmp_path):
        """There it is a typo, and silently ignoring it is how a setting
        goes missing for an entire session."""
        path = tmp_path / "gpagent.toml"
        path.write_text("[gamepad]\nnonexistent = 1\n")
        with pytest.raises(SystemExit):
            _build_config(_args(replay=None, config=str(path)))

    def test_no_replay_ignores_directory_state(self, tmp_path):
        _write_session_with_config(tmp_path, {"gamepad": {"sticks_mode": "full"}})
        cfg = _build_config(_args(replay=None))
        assert cfg.gamepad.sticks_mode == "off"


class TestSentSheet:
    """`--sent-sheet` shows what a turn was shown, not what capture recorded."""

    def _session(self, seqs=(1, 2, 3)) -> list:
        from io import BytesIO

        from PIL import Image

        from gpagent.events import ScreenFrame

        buf = BytesIO()
        Image.new("RGB", (64, 36), (90, 90, 90)).save(buf, format="JPEG")
        return [
            ScreenFrame(seq=s, t=float(s), w=64, h=36, data=buf.getvalue(), trigger="scene")
            for s in seqs
        ]

    def _log(self, directory, frames: dict) -> Any:
        path = directory / "agent.jsonl"
        path.write_text(
            json.dumps({"t": 1.0, "kind": "player", "text": "speech"})
            + "\n"
            + json.dumps({"t": 2.0, "kind": "ask", "text": "-> reply", "frames": frames})
            + "\n"
        )
        return path

    def test_it_draws_a_row_of_what_one_turn_sent(self, tmp_path, capsys):
        events = self._session()
        self._log(
            tmp_path,
            {"current": 3, "detail": "high", "trail": [1, 2], "trail_detail": "low"},
        )
        _write_sent_sheet(tmp_path, events)
        out = capsys.readouterr().out
        assert "#1:low #2:low #3:high" in out, "detail is named per image, per turn"
        assert (tmp_path / "sent-sheet.png").exists()

    def test_the_log_can_live_apart_from_the_blobs(self, tmp_path, capsys):
        """A `--no-media` replay has the log; the payloads are in the recording."""
        run = tmp_path / "run"
        run.mkdir()
        self._log(run, {"current": 3, "detail": "high", "trail": [1], "trail_detail": "low"})
        _write_sent_sheet(tmp_path, self._session(), run / "agent.jsonl")
        assert "#1:low #3:high" in capsys.readouterr().out
        assert (tmp_path / "sent-sheet.png").exists()

    def test_a_missing_blob_is_drawn_rather_than_fatal(self, tmp_path, capsys):
        self._log(
            tmp_path,
            {"current": 3, "detail": "high", "trail": [99], "trail_detail": "low"},
        )
        _write_sent_sheet(tmp_path, self._session())
        assert "1 blobs not in" in capsys.readouterr().out
        assert (tmp_path / "sent-sheet.png").exists()

    def test_it_says_so_when_there_is_no_agent_log(self, tmp_path, capsys):
        _write_sent_sheet(tmp_path, self._session())
        assert "no agent log" in capsys.readouterr().out
        assert not (tmp_path / "sent-sheet.png").exists()
