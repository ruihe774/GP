"""What the agent is told about itself.

The instructions budget deliberately does *not* go on "be sparing, don't narrate
everything". Restraint is enforced by `SpeakPolicy`: the model is only ever
asked for a response at a moment already judged worth speaking at, so asking it
to also decide whether to speak would be paying twice for one decision and
getting a worse answer. What the prompt is actually for is length and register
-- the things the policy cannot control.

Per-response instructions ride on `response.create` and override the session
ones for that response only, which is how the three reasons get different
behaviour out of one persona.
"""

from __future__ import annotations

from pathlib import Path

from .config import AgentConfig

__all__ = [
    "DEFAULT_PERSONA",
    "RESPONSE_INSTRUCTIONS",
    "LANGUAGE_NAMES",
    "resolve_persona",
    "instructions",
    "instructions_for",
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
    """The persona, plus the language directive if one was asked for."""
    text = resolve_persona(cfg)
    if cfg.language:
        name = language_name(cfg.language)
        text += (
            f"\nSpeak {name} at all times. The player speaks {name}; answer in "
            f"{name} even if they use another language for a word or two, and "
            f"never comment on which language is being spoken."
        )
    return text


def instructions_for(cfg: AgentConfig, reason: str) -> str:
    """Per-response instructions: the persona *plus* the reason.

    `response.create.instructions` **replaces** the session instructions for
    that response, it does not add to them. Sending the reason line alone threw
    the persona away on every single turn -- the first live run against sess5
    answered a swearing player with four sentences of encouraging life coaching,
    which is exactly what a model with no persona and one line of task framing
    sounds like. The language directive has to ride along for the same reason.
    """
    return f"{instructions(cfg)}\nRight now: {RESPONSE_INSTRUCTIONS[reason]}"
