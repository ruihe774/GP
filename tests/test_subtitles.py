"""Reading a subtitle file, and putting it on the wire once.

The two halves are tested apart: parsing is a pure function over text in the
wild, and seeding is about *where in the conversation* the script lands, which
is the whole reason for sending it whole.
"""

from __future__ import annotations

import pytest

from gpagent.agent.config import SubtitleConfig
from gpagent.agent.hud import NullHud
from gpagent.agent.playback import NullPlayer
from gpagent.agent.session import CommentaryAgent, ReplayClock
from gpagent.agent.subtitles import clock, load_script, parse_srt, render
from gpagent.agent.transport import FakeTransport
from gpagent.config import CaptureConfig
from gpagent.events import ScreenFrame, SpeechSegment

SRT = """\
1
00:00:01,000 --> 00:00:03,500
You should not have come here.

2
00:00:04,000 --> 00:00:06,000
<i>I know.</i>
It's fine.

3
00:01:02,250 --> 00:01:04,000
[DOOR SLAMS]
"""


def write(tmp_path, text=SRT, name="film.srt", encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


class TestParsing:
    def test_cues_carry_their_film_time_and_text(self):
        cues = parse_srt(SRT)
        assert [c.index for c in cues] == [1, 2, 3]
        assert cues[0].start == pytest.approx(1.0)
        assert cues[0].end == pytest.approx(3.5)
        assert cues[0].text == "You should not have come here."

    def test_multi_line_cues_become_one_line_without_markup(self):
        assert parse_srt(SRT)[1].text == "I know. It's fine."

    def test_a_cue_an_hour_in_keeps_its_hours(self):
        cues = parse_srt("1\n01:30:05,000 --> 01:30:07,000\nStill here.\n")
        assert cues[0].start == pytest.approx(5405.0)
        assert clock(cues[0].start) == "01:30:05"

    def test_crlf_bom_dot_decimals_and_missing_indices_all_parse(self):
        text = "﻿00:00:02.500 --> 00:00:04.000\r\nHello.\r\n\r\n"
        cues = parse_srt(text)
        assert len(cues) == 1
        assert cues[0].start == pytest.approx(2.5)
        assert cues[0].text == "Hello."

    def test_cues_come_back_in_film_order(self):
        text = (
            "2\n00:00:09,000 --> 00:00:10,000\nsecond\n\n"
            "1\n00:00:01,000 --> 00:00:02,000\nfirst\n"
        )
        assert [c.text for c in parse_srt(text)] == ["first", "second"]

    def test_rendering_stamps_each_line_with_its_position(self):
        rendered = render(parse_srt(SRT))
        assert rendered.splitlines()[0] == "00:00:01  You should not have come here."
        assert rendered.splitlines()[2].startswith("00:01:02  ")

    def test_rendering_without_timestamps_is_the_dialogue_alone(self):
        rendered = render(parse_srt(SRT), timestamps=False)
        assert rendered.splitlines()[0] == "You should not have come here."


class TestLoading:
    def test_a_file_loads_into_a_script_that_knows_its_size(self, tmp_path):
        script = load_script(SubtitleConfig(path=str(write(tmp_path))))
        assert len(script.cues) == 3
        assert script.tokens > 0
        assert script.runtime_s == pytest.approx(64.0)

    def test_cp1252_files_are_read_as_text(self, tmp_path):
        text = SRT.replace("I know.", "Naïve, café.")
        script = load_script(SubtitleConfig(path=str(write(tmp_path, text, encoding="cp1252"))))
        assert "Naïve, café." in script.text

    def test_effects_are_kept_by_default_and_dropped_on_request(self, tmp_path):
        path = str(write(tmp_path))
        assert "[DOOR SLAMS]" in load_script(SubtitleConfig(path=path)).text

        trimmed = load_script(SubtitleConfig(path=path, skip_effects=True))
        assert "[DOOR SLAMS]" not in trimmed.text
        assert trimmed.skipped == 1

    def test_no_path_means_no_script(self):
        assert load_script(SubtitleConfig()) is None

    def test_a_configured_path_can_be_turned_off_for_one_run(self, tmp_path):
        cfg = SubtitleConfig(path=str(write(tmp_path)), enabled=False)
        assert load_script(cfg) is None

    def test_a_file_with_no_cues_is_an_error_rather_than_a_silent_session(self, tmp_path):
        path = tmp_path / "empty.srt"
        path.write_text("this is not a subtitle file\n")
        with pytest.raises(ValueError):
            load_script(SubtitleConfig(path=str(path)))


def make_agent(tmp_path, **subtitle_kwargs):
    cfg = CaptureConfig()
    cfg.subtitles = SubtitleConfig(path=str(write(tmp_path)), **subtitle_kwargs)
    clock_obj = ReplayClock(100.0)
    agent = CommentaryAgent(cfg, FakeTransport(), NullPlayer(clock_obj), NullHud(), clock=clock_obj)
    agent._clock_obj = clock_obj
    return agent


def items_of(agent, role):
    return [
        e
        for e in agent.transport.sent_of_type("conversation.item.create")
        if e["item"]["role"] == role
    ]


class TestSeeding:
    async def test_the_whole_script_is_sent_once_before_the_first_turn(self, tmp_path):
        agent = make_agent(tmp_path)
        await agent.start()
        try:
            order = [e["type"] for e in agent.transport.sent]
            assert order == ["session.update", "conversation.item.create"]

            text = items_of(agent, "system")[0]["item"]["content"][0]["text"]
            assert "You should not have come here." in text
            assert "[DOOR SLAMS]" in text, "the file goes whole, ending included"
            assert "not yours to use" in text, "the note travels with the script"
        finally:
            await agent.close()

    async def test_the_script_item_is_never_pruned(self, tmp_path):
        agent = make_agent(tmp_path)
        agent.cfg.prune_after_s = 1.0
        agent.cfg.prune_interval_s = 0.0
        await agent.start()
        try:
            await agent.handle(ScreenFrame(seq=1, w=1024, h=576, data=b"\xff\xd8x"), now=100.0)
            await agent.handle(SpeechSegment(dur_ms=800, data=b"\x00\x01" * 9600), now=101.0)
            await agent.tick(now=500.0)

            deleted = {
                e["item_id"] for e in agent.transport.sent_of_type("conversation.item.delete")
            }
            assert deleted, "the disposable half should still be pruned"
            script_id = items_of(agent, "system")[0]["item"]["id"]
            assert script_id.startswith("gpsub")
            assert script_id not in deleted
        finally:
            await agent.close()

    async def test_no_position_is_claimed_by_default(self, tmp_path):
        # Elapsed session time is not playback position for anything that can be
        # paused, so nothing is asserted about where they are unless the config
        # says playback is untouched.
        agent = make_agent(tmp_path, offset_s=1800.0)
        await agent.start()
        try:
            await agent.handle(
                SpeechSegment(t=120.0, dur_ms=800, data=b"\x00\x01" * 9600), now=120.0
            )
            nudge = items_of(agent, "system")[-1]["item"]["content"][0]["text"]
            assert "00:" not in nudge
            script = items_of(agent, "system")[0]["item"]["content"][0]["text"]
            assert "for you to work out" in script
        finally:
            await agent.close()

    async def test_the_position_note_can_be_turned_on(self, tmp_path):
        agent = make_agent(tmp_path, offset_s=1800.0, position_note=True)
        await agent.start()
        try:
            # Film time is the capture timeline (`event.t`) plus the offset, not
            # the agent's own clock: 120 s into capture, having started half an
            # hour into the film, is 00:32:00.
            await agent.handle(
                SpeechSegment(t=120.0, dur_ms=800, data=b"\x00\x01" * 9600), now=120.0
            )
            nudge = items_of(agent, "system")[-1]["item"]["content"][0]["text"]
            assert "00:32:00" in nudge
            # ...and the script says the clock is a hint, not a reading.
            script = items_of(agent, "system")[0]["item"]["content"][0]["text"]
            assert "hint and not a fact" in script
        finally:
            await agent.close()

    async def test_a_session_with_no_subtitles_sends_nothing_extra(self):
        clock_obj = ReplayClock(100.0)
        agent = CommentaryAgent(
            CaptureConfig(), FakeTransport(), NullPlayer(clock_obj), clock=clock_obj
        )
        await agent.start()
        try:
            assert [e["type"] for e in agent.transport.sent] == ["session.update"]
            assert agent.report()["subtitles"] is None
        finally:
            await agent.close()
