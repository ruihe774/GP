"""Replay should apply the config a session was recorded with, unless overridden."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from gpagent.cli import _build_config
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

    def test_unknown_key_in_recorded_config_raises(self, tmp_path):
        _write_session_with_config(tmp_path, {"gamepad": {"nonexistent": 1}})
        with pytest.raises(SystemExit):
            _build_config(_args(replay=str(tmp_path)))

    def test_no_replay_ignores_directory_state(self, tmp_path):
        _write_session_with_config(tmp_path, {"gamepad": {"sticks_mode": "full"}})
        cfg = _build_config(_args(replay=None))
        assert cfg.gamepad.sticks_mode == "off"
