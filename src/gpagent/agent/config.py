"""Configuration for the commentary agent.

These are added as sections of `CaptureConfig`, so `--set speak.min_gap_s=12`
and a `[speak]` TOML table work with no extra machinery.

The split is deliberate: `AgentConfig` is *what we send and how we talk to the
API*, `SpeakConfig` is *when we open our mouth*. Only the second one decides
whether this feels like a friend or a tech demo, and it is the one with a
virtual-clock test suite.

`HudConfig` is a third thing again: *where the words land* when they are read
rather than heard. It is deliberately independent of the other two -- the HUD is
a display surface with its own command (`gpagent hud`) and no knowledge of the
session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "AgentConfig",
    "SpeakConfig",
    "HudConfig",
    "SubtitleConfig",
    "HudAnchor",
    "HudStyle",
    "TruncationMode",
]

ImageDetail = Literal["auto", "low", "high"]
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
#: Whether the agent is heard or read. Session-wide, not per remark: it is
#: `output_modalities` on `session.update`, so it decides what the model
#: generates rather than how one response is presented.
OutputMode = Literal["voice", "text"]
#: Who trims a conversation that outgrows the model's context window.
#: "disabled" makes it the client's job exclusively; the other two hand it back
#: to the server. See `AgentConfig.truncation`.
TruncationMode = Literal["disabled", "auto", "retention_ratio"]
#: Which corner (or edge) the toast stack sits in. The vertical half also picks
#: the growth direction, so the anchored edge never moves as entries arrive.
HudAnchor = Literal[
    "top-right", "top-left", "top", "bottom-right", "bottom-left", "bottom"
]
#: How a remark is rendered. "card" is the panel with an accent bar and rules
#: between entries; "stroked" drops the panel entirely and outlines each glyph
#: instead, for a game whose own HUD a translucent box would compete with.
HudStyle = Literal["card", "stroked"]


@dataclass
class AgentConfig:
    enabled: bool = True
    #: `-mini` is the cheap one; use it for anything that is not a real session
    model: str = "gpt-realtime-2.1-mini"
    voice: str = "marin"
    #: Output audio playback rate, a multiple of normal speed. Server clamps
    #: to [0.25, 1.5]; it's a post-processing rate change, not composition,
    #: so it doesn't affect wording or timing of pauses.
    voice_speed: float = 1.0

    #: override the built-in persona (see persona.py); `persona` wins over the file
    persona: str | None = None
    persona_file: str | None = None

    #: Reasoning effort for reasoning-capable Realtime models (the gpt-realtime-2
    #: family, which includes gpt-realtime-2.1 and -2.1-mini). None omits the
    #: `reasoning` block entirely and leaves it at the server's default -- older,
    #: non-reasoning models (gpt-realtime, gpt-realtime-1.5) reject the field,
    #: so those need `--set agent.reasoning_effort=null`.
    #:
    #: "low" because OpenAI's realtime guide says to start there rather than at
    #: the default, and because of what this agent is asked for: one sentence of
    #: reaction, with no tools to orchestrate and no multi-step task to plan.
    #: The cost of thinking longer is not tokens, it is that the remark lands
    #: after the moment it was about.
    reasoning_effort: ReasoningEffort | None = "low"

    #: Offer `wait_for_user` (persona.WAIT_TOOL): a no-op function whose only
    #: purpose is to let the model end a turn without speaking. `SpeakPolicy`
    #: decides *when* a remark is wanted from what capture can see; it cannot
    #: tell a VAD segment that was the player from one that was the television,
    #: or a burst worth a reaction from a burst that turned out to be nothing.
    #: The model can, and this is the only way it has to say so -- without it,
    #: every turn the policy opens must be filled.
    wait_tool: bool = True
    #: override for persona.WAIT_NOTE, appended to the persona when `wait_tool`
    #: is on. None uses the built-in note.
    wait_note: str | None = None

    #: ISO-639-1 code the player speaks and the agent should answer in, e.g.
    #: "ja". None lets the model follow whatever it hears. Half of this is an
    #: API setting and half is not: the code is passed to input transcription,
    #: where it improves accuracy and latency, but the Realtime API has no
    #: parameter for *output* language, so that is a line in the instructions.
    #: The persona itself stays in English -- it is instructions to the model,
    #: not something anyone hears.
    language: str | None = None
    #: template for the output-language directive appended when `language` is
    #: set; `{name}` is substituted with the resolved language name (see
    #: persona.language_name). None uses the built-in sentence.
    language_directive: str | None = None

    #: override for persona.TEXT_OUTPUT_NOTE, appended when `output = "text"`.
    #: None uses the built-in note.
    text_output_note: str | None = None
    #: per-reason overrides for persona.RESPONSE_INSTRUCTIONS ("reply" /
    #: "react" / "ambient"); a reason missing here falls back to the built-in
    #: text. Sent as the per-turn nudge before every `response.create`.
    reason_instructions: dict[str, str] = field(default_factory=dict)
    #: prefix the per-turn nudge is wrapped in, e.g. "Right now: ".
    reason_note_prefix: str = "Right now: "
    #: template for the trail-frame ordering note sent ahead of trail images
    #: (see Session._context_item); `{n}`, `{frame_word}` ("frame"/"frames")
    #: and `{span}` are substituted. None uses the built-in sentence.
    trail_note_template: str | None = None

    # -- vision -----------------------------------------------------------
    #: "high" is ~765 tok for a 1024x576 frame, "low" a flat 85 -- a 9x lever.
    #: Frames are the dominant input cost, so this and `max_image_age_s` are
    #: the two knobs that matter.
    image_detail: ImageDetail = "high"
    #: How many *earlier* frames ride along with the current one, at
    #: `image_trail_detail`. One screenshot says where the player is; it does not
    #: say where they came from, and most of what a friend on the couch reacts to
    #: is the change. Capture triggers frames far faster than the agent speaks, so
    #: the frames between two responses are otherwise thrown away. Sending a few
    #: of them, spread across the window and cheap, buys the trajectory for ~85
    #: tokens each against the ~765 the current frame costs. 0 restores
    #: one-frame behaviour.
    #:
    #: Sized against TPM, not price. Prompt caching makes the money nearly
    #: irrelevant (~$0.5 of trail over a 66-response session, most of it at the
    #: cached rate), but cached tokens still count against the rate limit, and a
    #: trail stays in history and is re-sent every turn after it: at 4 frames it
    #: adds ~20k tokens/request by turn 60 on top of the ~10.5k baseline, which
    #: is close to the sess7 `rate_limit` losses (CLAUDE.md, Component B gotcha
    #: 12). Retiring a spent trail would flatten that and is the obvious next
    #: step; until then this stays single digits.
    image_trail: int = 4
    #: detail for the trail frames -- "low" is the point of the feature
    image_trail_detail: ImageDetail = "low"
    #: Minimum spacing between images sent in one turn, trail and current alike.
    #: Bucketing spreads the picks across the candidate window but never asks
    #: whether that window is worth spreading: back-to-back replies can leave a
    #: two-second gap, and four frames of the same two seconds is four copies of
    #: one moment at 85 tokens each. Below capture's own `triggers.min_interval_s`
    #: (2.0), so it changes nothing at default settings -- it exists so the agent
    #: does not depend on a capture-side knob for its own correctness. The trail
    #: shrinks rather than clusters: a 3 s window sends one frame, not four.
    image_trail_min_gap_s: float = 1.5
    #: trail frames older than this are dropped. Looser than `max_image_age_s`
    #: on purpose: a stale frame is a bad answer to "what is on screen now" but
    #: a fine answer to "how did we get here".
    image_trail_max_age_s: float = 45.0
    #: never attach a frame older than this; a stale screenshot is worse than none
    max_image_age_s: float = 20.0
    #: attach a frame to unprompted remarks too, not just replies
    image_on_unprompted: bool = True

    # -- context ----------------------------------------------------------
    #: gamepad summaries older than this are dropped rather than sent
    context_window_s: float = 45.0
    max_summary_chars: int = 400
    # Client-side context pruning. Off by default, against expectation, because
    # as a *cost* measure it was measured and it lost. The Realtime API re-bills
    # the whole conversation as input on every turn, so trimming looks like an
    # obvious win -- but deleting an item invalidates the prompt-cache prefix
    # behind it, and cached input is ~10x cheaper than fresh. On sess5, keeping
    # every image billed 31008 image tokens for $0.0204 of input; deleting all
    # but the last two billed 8721 for $0.0268. Fewer tokens, more money.
    #
    # What changes the answer is TPM. Cached tokens are cheap but they are not
    # free of the rate limit -- they count against it in full -- so on a long
    # session with frequent frames the conversation grows until every request
    # is rejected. sess_movie2 died that way: thirteen consecutive
    # `rate_limit_exceeded` failures from 10605 s on, at ~2.3k image tokens
    # added per turn. Against that, pruning is a straight win and the
    # invalidated prefix is what it costs.
    #
    # `prune_after_s = 0` turns off the *scheduled* round only. A round forced
    # by `context_budget_tokens` or by the server refusing a response still
    # runs: with `truncation = "disabled"` there is nothing else left to stop a
    # session growing until it cannot make a request, and "do not pay the cache
    # routinely" is a different statement from "do not survive".
    #
    # A round does not delete history so much as re-state it cheaply. Images
    # come back as a line saying images were here; the player's utterance and
    # the agent's own reply come back as their transcripts, which is the same
    # thread of conversation at a twentieth of the tokens. Only the per-turn
    # nudge is deleted outright -- it means nothing once its turn is over.
    #: age at which images and spoken turns are replaced by text, 0 for no
    #: scheduled pruning at all. One cutoff for every kind, deliberately: they
    #: are created together and removing them in the same round costs one cache
    #: invalidation rather than several.
    prune_after_s: float = 0.0
    #: Floor between pruning rounds. This is the knob that makes pruning cheap:
    #: each round invalidates the cached prefix once, so nine items replaced in
    #: one round cost a ninth of what they cost one per turn. Nothing is sent
    #: between rounds however stale it gets, so the effective retention is
    #: `prune_after_s` to `prune_after_s + prune_interval_s`.
    prune_interval_s: float = 120.0
    #: Templates for what stands in for a removed item; None uses the built-in
    #: sentence (see `session.STUBS`). Short on purpose: a stub is permanent,
    #: so every token in one is re-billed on every turn that follows it.
    #: `{n}` and `{frame_word}` for images, `{transcript}` for speech.
    prune_image_stub_template: str | None = None
    prune_speech_stub_template: str | None = None
    prune_reply_stub_template: str | None = None
    #: How long an audio item waits for its transcript before a generic stub
    #: stands in for it. Audio is never removed before its replacement is
    #: known -- deleting it first would take the only record of what was said
    #: with it -- so without a deadline one lost transcription pins an item in
    #: the conversation for the rest of the session.
    audio_stub_wait_s: float = 30.0
    #: Hard ceiling per response; audio counts toward it at ~1 token / 50 ms.
    #: A backstop against a rambling response, not a length control -- brevity
    #: is the persona's job. Hitting this truncates mid-word, so it is set well
    #: above the observed mean (72 tokens over 14 responses on sess5); 220 was
    #: tight enough to cut one of them off.
    max_output_tokens: int = 320

    # -- the context ceiling ----------------------------------------------
    # Who is allowed to drop history. The server's own mechanism drops from the
    # oldest end, and the oldest end of this conversation is the subtitle
    # script: the one item the session cannot reconstruct, deliberately placed
    # first so it keys the cached prefix of every request (see
    # `session._seed_subtitles`). Amortized batches that keep the cache intact
    # are a good trade right up until the batch takes the film's dialogue.
    #
    # So "disabled", and the client owns the ceiling instead: a pruning round
    # forced by `context_budget_tokens` before the limit, another forced by the
    # server's refusal after it, and a reconnect if neither can free anything.
    # "auto" or "retention_ratio" hand the job back to the server.
    truncation: TruncationMode = "disabled"
    #: Read only when `truncation = "retention_ratio"`: the fraction of the
    #: post-instruction window to retain when trimming, where 1.0 drops only
    #: what it must and lower values drop extra to buy headroom.
    truncation_retention_ratio: float = 0.8
    #: Ceiling on input tokens after instructions, for the retention-ratio
    #: form. 0 omits `token_limits` and leaves the model's own window in charge.
    truncation_post_instruction_tokens: int = 0
    #: Force a pruning round once a response bills more input than this; 0
    #: never does. The point of a budget is to meet the ceiling on our own
    #: terms: without one the first sign of trouble is a rejected
    #: `response.create`, which costs a turn the player was waiting on.
    context_budget_tokens: int = 20000
    #: How many times one turn may be asked for again after the server says the
    #: conversation is too long. Per turn, not per session.
    context_full_retries: int = 1
    #: Stubs kept when a forced round has run out of anything else to remove.
    #: The wall case: once every image and every utterance has already become a
    #: stub, the stubs are the only thing left to spend, and spending the
    #: oldest of them beats a session that can never speak again.
    context_full_keep_stubs: int = 8
    #: Error codes to read as "the conversation is too long" on top of the
    #: built-in heuristic. The API documents that disabling truncation makes
    #: the server error on an oversized conversation but does not document the
    #: code, so this is the escape hatch that avoids needing a code change.
    context_full_codes: list[str] = field(default_factory=list)

    #: transcribe the player so the session log is readable. Costs extra per
    #: minute of speech; turn it off for long sessions where nobody will read it.
    transcribe_player: bool = True
    transcribe_model: str = "gpt-4o-mini-transcribe"

    # -- output -----------------------------------------------------------
    #: Voice or text, for the whole session. This is `output_modalities` on
    #: `session.update`: in "text" the model writes its remark instead of
    #: speaking it, and there is no audio to play, no transcript to wait for and
    #: nothing to talk over. The words land on the HUD (see `HudConfig`), which
    #: is why they are worth reading at all -- a remark with nowhere to go is
    #: just a line in the log. Everything on the *input* side is unchanged: the
    #: player still talks, and is still transcribed.
    output: OutputMode = "voice"
    #: play to the *default* sink. Component A's echo canceller uses that sink's
    #: monitor as its AEC reference; any other sink and the agent hears itself
    #: and replies to itself.
    playback: bool = True
    #: GStreamer sink element. `autoaudiosink` selects `pulsesink`, whose ring
    #: buffer handles streamed audio properly; `pipewiresink` is rank 0 and
    #: garbled speech in testing. Whatever you put here must not name a device,
    #: or the echo canceller stops cancelling.
    audio_sink: str = "autoaudiosink"
    #: linear gain applied by a `volume` element in the playback pipeline.
    #: 1.0 is unity, 0.0 is silent; above 1.0 amplifies (and can clip).
    volume: float = 1.0
    #: hard stop on a rambling response -- output audio is the expensive half
    max_response_s: float = 15.0

    #: reconnect backoff bounds if the socket drops mid-session
    reconnect_min_s: float = 1.0
    reconnect_max_s: float = 30.0

    @property
    def text_output(self) -> bool:
        """Read, not heard. The question a dozen places downstream ask."""
        return self.output == "text"


@dataclass
class SubtitleConfig:
    """The subtitle file for what is being watched, sent once, whole.

    Its own section rather than more keys on `AgentConfig` for the same reason
    `[hud]` is: this is an input the session is *given*, from outside capture
    entirely, and the one thing anybody sets is `path`.

    The whole file goes, including everything the viewer has not reached yet.
    That is a decision about cost and about caching -- see `agent/subtitles.py`
    -- and it hands the model the ending, which is why `persona.SUBTITLE_NOTE`
    travels with it and why `position_note` exists.
    """

    enabled: bool = True
    #: path to a `.srt`. Nothing is sent without one; this is the whole feature's
    #: on switch, and `enabled` is only here so a config can keep the path and
    #: turn it off for one run.
    path: str | None = None
    #: Where the film is when capture starts, in seconds. The script is
    #: timestamped from the film's first frame and the session clock starts when
    #: gpagent does, so this is the difference between the two: start gpagent
    #: 40 s into the film and `offset_s = 40`. Only `position_note` reads it --
    #: the script itself is sent unshifted, in the film's own time. Useless on
    #: its own, since it fixes the start of a mapping whose slope is the
    #: assumption `position_note` cannot make.
    offset_s: float = 0.0
    #: Force a text encoding. Empty tries UTF-8 and then cp1252, which is what
    #: subtitle files in the wild actually are.
    encoding: str = ""
    #: Prefix each line with its `HH:MM:SS`. About a quarter of the script's
    #: tokens, and the only thing that makes "where they have got to" mean
    #: anything -- turn it off only if the position note is off too.
    timestamps: bool = True
    # Telling the model where they are in the film sounds like the obvious
    # counterweight to handing it the whole script -- but the only position
    # available is `offset_s` plus elapsed session time, which is playback time
    # only if the film never pauses, never seeks, and never changes speed. It
    # pauses. Someone gets a drink, someone rewinds ten seconds to catch a line,
    # someone leaves it paused through a phone call, and from then on the note
    # says a minute the viewer has not reached yet -- which is precisely the
    # direction that turns dialogue the agent has into dialogue it should not
    # have. A stated position is trusted in a way an absent one is not, so being
    # confidently wrong here is worse than saying nothing: without it the model
    # locates itself from the screen and from what has just been said, which
    # degrades gracefully because it is evidence rather than arithmetic.
    #
    # Turn it on for playback that genuinely runs start to finish untouched, and
    # set `offset_s` with it.
    #: append the elapsed-time position to the per-turn nudge. Off by default:
    #: it is extrapolated from wall time, so a pause or a seek makes it lie.
    #: When on it rides on the nudge, which is appended at the tail of the
    #: conversation, so a value that changes every turn costs no cache.
    position_note: bool = False
    #: template for that note; `{clock}` is `HH:MM:SS` into the film
    #: `{clock}` is `HH:MM:SS` of elapsed playback. Hedged on purpose -- it is
    #: an extrapolation, and it should not read like a reading off the player.
    position_template: str = " Roughly {clock} in, by the clock."
    #: Drop cues that are entirely `[DOOR SLAMS]`, `(sighs)` or `♪ ... ♪`.
    #: Off by default: on a hearing-impaired track those lines are most of what
    #: the agent knows about a scene with no dialogue in it.
    skip_effects: bool = False
    #: replaces persona.SUBTITLE_NOTE, the framing sent above the script
    note: str | None = None


@dataclass
class HudConfig:
    """The on-screen commentary overlay: a stack of toasts in a screen corner.

    Sized for glancing at, not reading: whatever is on screen is the game, and
    the HUD has to be legible in the corner of an eye during play. That is what
    `max_entries`, the reading-speed hold, and the dimming of older entries are
    all for -- old remarks fade out of the way rather than accumulating into
    something that demands attention.
    """

    enabled: bool = False

    # -- placement --------------------------------------------------------
    anchor: HudAnchor = "top-right"
    #: Which monitor to sit on: "primary", a RandR output name ("HDMI-1"), or an
    #: index. The X screen is the *union* of the outputs -- 8960x2880 across two
    #: here -- so "top right of the screen" is off the side of a monitor unless
    #: one is picked deliberately.
    monitor: str = "primary"
    #: gap between the card and the monitor edges it is anchored to
    margin_px: int = 32
    #: card width as a fraction of the monitor, floored at `min_width_px`. A
    #: fraction rather than pixels because the same config has to look sane on a
    #: 1080p laptop and a 4K TV.
    width_frac: float = 0.28
    min_width_px: int = 320
    #: HiDPI. Every `*_px` here is quoted in design pixels at 96 dpi and
    #: multiplied by this. 0 discovers it: `Xft.dpi` from the X resource manager
    #: (what the compositor tells clients -- 192, i.e. 2x, on this machine),
    #: else the chosen monitor's own physical dpi. XWayland reports device
    #: pixels, so getting this wrong is not subtle: a 22 px card on a 5K panel
    #: is unreadable.
    scale: float = 0.0

    # -- type -------------------------------------------------------------
    #: Resolved through fontconfig (`fc-match`), so the family can be a generic
    #: alias like "sans-serif" and still land on something installed.
    font_family: str = "sans-serif"
    #: Language hint passed to fontconfig as `:lang=`, e.g. "ja". Pillow does no
    #: font fallback -- one TTF renders the whole card, and DejaVu Sans has no
    #: CJK glyphs at all -- so a Japanese session that leaves this empty gets a
    #: card full of tofu. This is the knob that fixes it.
    font_lang: str = ""
    #: an explicit .ttf path, which skips fontconfig entirely
    font_path: str | None = None
    font_size_px: int = 22
    #: line height as a multiple of the font size
    line_spacing: float = 1.25
    padding_px: int = 14
    corner_radius_px: int = 14

    # -- timing -----------------------------------------------------------
    #: how many remarks stay on screen at once; the oldest drops off the top
    max_entries: int = 3
    #: Hold time is read speed, not a constant: a nine-word aside and a
    #: three-line explanation are not on screen for the same length of time.
    #: Characters per second, then clamped by the two bounds below.
    read_cps: float = 16.0
    hold_min_s: float = 4.0
    hold_max_s: float = 16.0
    #: fade-out at the end of an entry's hold; 0 disables the animation
    fade_ms: int = 250

    # -- style ------------------------------------------------------------
    #: "card" draws the panel below; "stroked" is borderless -- no panel, no
    #: accent bar, no rules between entries -- and relies on an outline
    #: around each glyph for legibility over whatever the game is showing.
    style: HudStyle = "card"
    #: outline width for "stroked", in design px. Ignored by "card", which
    #: uses a hairline drop shadow instead.
    stroke_width_px: int = 2

    # -- colour -----------------------------------------------------------
    bg_color: str = "#12141a"
    #: card opacity. Real translucency, from a 32-bit ARGB visual composited by
    #: the compositor -- not a screenshot of what is behind. Also the stroke
    #: colour in "stroked", where there is no panel for it to paint.
    bg_opacity: float = 0.82
    fg_color: str = "#f2f4f8"
    #: the bar marking the newest entry, the one worth reading first ("card" only)
    accent_color: str = "#6fd3ff"
    #: alpha multiplier for everything that is not the newest entry
    dim_older: float = 0.55

    # -- window -----------------------------------------------------------
    #: Pass clicks through to whatever is underneath. Off means the HUD eats
    #: every click that lands on it, which during a game is a bug, not a
    #: feature; it exists only for debugging where the window actually is.
    click_through: bool = True
    #: X display to open, e.g. ":0". None follows $DISPLAY. This is XWayland on
    #: a Wayland session, which is the point: an override-redirect X window is
    #: mutter's topmost layer, and Wayland offers a plain client nothing like it.
    display: str | None = None


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
