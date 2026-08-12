"""What the agent is told about itself.

The instructions budget deliberately does *not* go on "be sparing, don't narrate
everything". Restraint is enforced by `SpeakPolicy`: the model is only ever
asked for a response at a moment already judged worth speaking at, so asking it
to also decide whether to speak would be paying twice for one decision and
getting a worse answer. What the prompt is actually for is length and register
-- the things the policy cannot control.

The persona is sent once, in `session.update`, and never resent. The per-turn
part -- which of the three reasons we are speaking for -- rides as a system
message item appended just before `response.create`. See `reason_note`.
"""

from __future__ import annotations

from pathlib import Path

from .config import AgentConfig

__all__ = [
    "DEFAULT_PERSONA",
    "TEXT_OUTPUT_NOTE",
    "RESPONSE_INSTRUCTIONS",
    "LANGUAGE_NAMES",
    "resolve_persona",
    "instructions",
    "reason_note",
    "language_name",
]

#: Names for the codes people actually use, so the instruction reads as a
#: sentence rather than an ISO code. Anything not here is passed through: the
#: model knows far more language names than this table does, and "answer in
#: nds" is still a workable instruction.
LANGUAGE_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "cs": "Czech", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "nl": "Dutch", "no": "Norwegian", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sv": "Swedish",
    "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese",
    "zh": "Chinese",
}


DEFAULT_PERSONA = """\
You are sitting on the couch next to a friend who is playing a video game. You
are watching, not commentating: you see the occasional screenshot, you get a
terse log of what their hands are doing on the controller, and you hear them
when they talk to you.

How you talk:
- One sentence. Two if it earns it. You are a friend, not a narrator.
- Never describe the screenshot. React to it, or ignore it.
- Never recap what just happened; they were there.
- Never say what you can or cannot see, and never mention screenshots, logs,
  controllers, or being an AI.
- Don't announce yourself, don't ask whether they want commentary, and don't
  offer help they didn't ask for.
- When they ask you something directly, answer it straight away. That is the
  one time being useful beats being brief.
- Swearing, sarcasm, and being wrong are all fine. Being boring is not.
- You only get to speak at moments that were already judged worth speaking at,
  so say the thing. Don't hedge and don't fill.
"""


#: Appended when the session outputs text instead of speech. A remark that is
#: read has different constraints from one that is heard: it lands in a corner
#: of a screen the player is busy looking elsewhere on, it does not fade as it
#: is said, and nothing renders markdown -- asterisks arrive as asterisks.
TEXT_OUTPUT_NOTE = """
You are not speaking out loud. What you say is shown as a line of text in the
corner of their screen, read at a glance while they are playing. Keep it to one
short line -- shorter than you would say it. Plain words only: no markdown, no
asterisks, no emoji, no stage directions, and never refer to the text or the
screen it is on."""


RESPONSE_INSTRUCTIONS = {
    "reply": "They just said something to you. Answer it, briefly.",
    "react": (
        "Something just happened. One short reaction -- what a friend would "
        "actually say out loud, not a description of events."
    ),
    "ambient": (
        "It has been quiet for a while. One short, low-stakes remark or a "
        "question about what they are doing. Do not recap, and do not repeat "
        "anything you have already said."
    ),
}


def resolve_persona(cfg: AgentConfig) -> str:
    """Inline persona, else a file, else the built-in one."""
    if cfg.persona:
        return cfg.persona
    if cfg.persona_file:
        return Path(cfg.persona_file).expanduser().read_text()
    return DEFAULT_PERSONA


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.strip().lower(), code.strip())


def instructions(cfg: AgentConfig) -> str:
    """The persona, plus whatever the session's shape adds to it.

    Both additions are session-wide constants, which is why they belong here
    rather than on any one turn: this string is sent once, in `session.update`,
    and keys the cached prefix of every request after it.
    """
    text = resolve_persona(cfg)
    if cfg.text_output:
        text += cfg.text_output_note if cfg.text_output_note is not None else TEXT_OUTPUT_NOTE
    if cfg.language:
        name = language_name(cfg.language)
        template = cfg.language_directive or (
            "\nSpeak {name} at all times. The player speaks {name}; answer in "
            "{name} even if they use another language for a word or two, and "
            "never comment on which language is being spoken."
        )
        text += template.format(name=name)
    return text


def reason_note(reason: str, cfg: AgentConfig | None = None) -> str:
    """The one-line nudge for this turn, to be sent as a system message.

    This used to be `instructions_for(cfg, reason)`, sent on
    `response.create.instructions`, and it had to carry the whole persona and
    language directive along with it: `response.create.instructions`
    **replaces** the session instructions rather than adding to them, so
    sending the reason line alone threw the persona away on every turn -- the
    first live run against sess5 answered a swearing player with four sentences
    of encouraging life coaching.

    Carrying the persona there was the wrong fix. It put ~265 tokens that vary
    per turn at a position that keys the prompt cache, so a turn whose reason
    differed from the previous turn could not reuse the previous turn's prefix
    and fell back to the last turn with a matching reason. Measured on sess7
    (65 responses, 39 min): 533 uncached tokens on a median same-reason turn
    against 1495 when the reason changed, about $0.28 of a $1.62 session, most
    of it re-billed screenshots.

    So the persona stays in `session.update`, where it is constant and cached,
    and only this line varies. As a conversation item it is appended at the
    tail, leaving the whole prefix byte-identical to the previous request.

    The item persists in history, unlike per-response instructions -- a long
    session accumulates one stale line per turn. They are ~15 tokens each and
    sit inside the cached prefix, and deleting them per turn would invalidate
    exactly the prefix this change exists to preserve.
    """
    overrides = cfg.reason_instructions if cfg else {}
    text = overrides.get(reason, RESPONSE_INSTRUCTIONS[reason])
    prefix = cfg.reason_note_prefix if cfg else "Right now: "
    return f"{prefix}{text}"
