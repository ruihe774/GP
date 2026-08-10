"""Configuration for the commentary agent.

These are added as sections of `CaptureConfig`, so `--set speak.min_gap_s=12`
and a `[speak]` TOML table work with no extra machinery.

The split is deliberate: `AgentConfig` is *what we send and how we talk to the
API*, `SpeakConfig` is *when we open our mouth*. Only the second one decides
whether this feels like a friend or a tech demo, and it is the one with a
virtual-clock test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["AgentConfig", "SpeakConfig"]

ImageDetail = Literal["auto", "low", "high"]


@dataclass
class AgentConfig:
    enabled: bool = True
    #: `-mini` is the cheap one; use it for anything that is not a real session
    model: str = "gpt-realtime-2.1-mini"
    voice: str = "marin"

    #: override the built-in persona (see persona.py); `persona` wins over the file
    persona: str | None = None
    persona_file: str | None = None

    # -- vision -----------------------------------------------------------
    #: "high" is ~765 tok for a 1024x576 frame, "low" a flat 85 -- a 9x lever.
    #: Frames are the dominant input cost, so this and `max_image_age_s` are
    #: the two knobs that matter.
    image_detail: ImageDetail = "high"
    #: never attach a frame older than this; a stale screenshot is worse than none
    max_image_age_s: float = 20.0
    #: attach a frame to unprompted remarks too, not just replies
    image_on_unprompted: bool = True

    # -- context ----------------------------------------------------------
    #: gamepad summaries older than this are dropped rather than sent
    context_window_s: float = 45.0
    max_summary_chars: int = 400
    # Client-side context trimming. Both default to *off*, against expectation,
    # because it was measured and it lost. The Realtime API re-bills the whole
    # conversation as input on every turn, so trimming looks like an obvious win
    # -- but deleting an item invalidates the prompt-cache prefix behind it, and
    # cached input is ~10x cheaper than fresh. On sess5, keeping every image
    # billed 31008 image tokens for $0.0204 of input; deleting all but the last
    # two billed 8721 for $0.0268. Fewer tokens, more money.
    #
    # The server's `truncation.retention_ratio` is the mechanism that actually
    # belongs here: it drops history in amortized batches specifically to keep
    # the cache prefix intact. These two exist for hard bounds on very long
    # sessions; 0 disables each.
    #: delete conversation items older than this, 0 to leave it to the server
    prune_after_s: float = 0.0
    #: how many screenshots stay in the conversation, 0 for unlimited
    keep_images: int = 0
    #: Hard ceiling per response; audio counts toward it at ~1 token / 50 ms.
    #: A backstop against a rambling response, not a length control -- brevity
    #: is the persona's job. Hitting this truncates mid-word, so it is set well
    #: above the observed mean (72 tokens over 14 responses on sess5); 220 was
    #: tight enough to cut one of them off.
    max_output_tokens: int = 320
    #: server-side fallback for the same problem
    truncation_retention_ratio: float = 0.8

    #: transcribe the player so the session log is readable. Costs extra per
    #: minute of speech; turn it off for long sessions where nobody will read it.
    transcribe_player: bool = True
    transcribe_model: str = "gpt-4o-mini-transcribe"

    # -- output -----------------------------------------------------------
    #: play to the *default* sink. Component A's echo canceller uses that sink's
    #: monitor as its AEC reference; any other sink and the agent hears itself
    #: and replies to itself.
    playback: bool = True
    #: GStreamer sink element. `autoaudiosink` selects `pulsesink`, whose ring
    #: buffer handles streamed audio properly; `pipewiresink` is rank 0 and
    #: garbled speech in testing. Whatever you put here must not name a device,
    #: or the echo canceller stops cancelling.
    audio_sink: str = "autoaudiosink"
    #: hard stop on a rambling response -- output audio is the expensive half
    max_response_s: float = 15.0

    #: reconnect backoff bounds if the socket drops mid-session
    reconnect_min_s: float = 1.0
    reconnect_max_s: float = 30.0


@dataclass
class SpeakConfig:
    """When the agent is allowed to talk.

    Three reasons stack under one global cap, mirroring `TriggerConfig`:
    per-reason rules alone compose badly and take turns (Component A measured
    36 frames/min from three triggers that each individually allowed far fewer).
    """

    #: quiet floor after the agent stops talking; binds the two *unprompted*
    #: reasons. A direct question is exempt -- see SpeakPolicy.
    min_gap_s: float = 8.0
    #: Small floor between the end of one response and the start of a reply.
    #: `reply` is exempt from `min_gap_s` on purpose -- refusing to answer a
    #: direct question is worse than being chatty -- but "no floor at all" lets
    #: a player who talks continuously (shouting at a boss fight, say) pull
    #: back-to-back answers, bounded only by `max_per_min`. This is deliberately
    #: much smaller than `min_gap_s`: it breaks up bursts without making the
    #: agent feel slow. A reply held back by this stays pending and fires as
    #: soon as the gap clears, subject to `reply_ttl_s`.
    reply_min_gap_s: float = 3.0
    #: how long a wanted reply stays wanted. If the agent was busy or capped
    #: when the player spoke, answering nine seconds later is worse than not
    #: answering: the moment has gone and the audio is stale.
    reply_ttl_s: float = 8.0
    #: cap on how much unanswered speech is carried into one reply
    max_reply_audio_ms: int = 20000
    #: global ceiling over a sliding 60 s window, binding on *every* reason
    #: including replies. This is the rule that makes the combined rate sane.
    max_per_min: float = 6.0

    # -- react: something happened ----------------------------------------
    scene_threshold: float = 0.35
    intensity_threshold: float = 0.6
    #: consecutive hot gamepad windows before that counts as a burst
    burst_windows: int = 2
    event_cooldown_s: float = 30.0

    # -- ambient: nothing happened ----------------------------------------
    ambient_after_s: float = 75.0
    #: don't chat to an empty room: require play activity this recently. Kept
    #: separate from `ambient_after_s` on purpose -- how long a silence has to
    #: be before it is worth filling and how long before we assume the player
    #: got up and left are different questions.
    ambient_requires_activity: bool = True
    ambient_idle_horizon_s: float = 45.0

    # -- adaptation -------------------------------------------------------
    #: cooldowns multiply by backoff_factor ** min(unanswered, backoff_max)
    #: for every unprompted remark the player does not respond to
    backoff_factor: float = 1.6
    backoff_max: int = 3
    #: ...and shrink by this while the player is actively talking to us
    engagement_boost: float = 0.6
    engagement_window_s: float = 60.0

    # -- deadlock guards --------------------------------------------------
    # Both gates below are absolute: while they are set the agent cannot speak
    # at all. A gate that is set and never cleared is therefore a permanently
    # mute agent, which is the quietest possible bug to notice. Both expire.
    #: a speech.start whose segment never arrives (VAD false start, or a
    #: segment dropped as too short) must not gate the agent forever
    speech_gate_timeout_s: float = 31.0
    #: a response.create that never produces a response.done (dropped socket,
    #: server error) must not either
    response_gate_timeout_s: float = 45.0
