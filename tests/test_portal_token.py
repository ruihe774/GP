"""Restore-token handling, exercised without touching the real portal."""

from __future__ import annotations

import json

import pytest

from gpagent.capture import portal
from gpagent.config import CaptureConfig


@pytest.fixture
def fake_handshake(monkeypatch):
    """Capture the restore_token the handshake would be given."""
    calls: list[dict] = []

    def stub(**kwargs):
        calls.append(kwargs)
        return {
            "fd": -1,
            "node_id": 42,
            "width": 2560,
            "height": 1440,
            "source_type": 1,
            "restore_token": "fresh-token",
            "session_handle": "/session/1",
            "connection": None,
        }

    monkeypatch.setattr(portal, "_handshake", stub)
    return calls


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "screencast.json"
    path.write_text(json.dumps({"restore_token": "saved-token"}))
    return path


def test_saved_token_is_reused_by_default(fake_handshake, token_file):
    portal.open_screencast(token_path=token_file)
    assert fake_handshake[0]["restore_token"] == "saved-token"


def test_reselect_ignores_the_saved_token(fake_handshake, token_file):
    portal.open_screencast(token_path=token_file, use_saved_token=False)
    assert fake_handshake[0]["restore_token"] is None, (
        "the picker only appears when no restore token is sent"
    )


def test_new_token_is_persisted_after_reselect(fake_handshake, token_file):
    portal.open_screencast(token_path=token_file, use_saved_token=False)
    assert json.loads(token_file.read_text())["restore_token"] == "fresh-token", (
        "a fresh choice must be saved so later runs reuse it"
    )


def test_missing_token_file_is_not_fatal(fake_handshake, tmp_path):
    session = portal.open_screencast(token_path=tmp_path / "absent.json")
    assert fake_handshake[0]["restore_token"] is None
    assert session.node_id == 42


def test_persist_mode_is_requested(fake_handshake, token_file):
    portal.open_screencast(token_path=token_file)
    assert fake_handshake[0]["persist_mode"] == 2, "without persist there is no token at all"


class TestConfigWiring:
    def test_default_reuses_permission(self):
        assert CaptureConfig().screen.reselect_source is False

    def test_pick_screen_flag_sets_it(self):
        from gpagent.cli import _build_config, build_parser

        args = build_parser().parse_args(["record", "--pick-screen"])
        assert _build_config(args).screen.reselect_source is True

    def test_absent_flag_leaves_it_off(self):
        from gpagent.cli import _build_config, build_parser

        args = build_parser().parse_args(["record"])
        assert _build_config(args).screen.reselect_source is False
