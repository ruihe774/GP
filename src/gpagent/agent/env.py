"""Locating the OpenAI API key without ever printing it.

The key lives in `.env` at the repo root (gitignored). Nothing here logs a
value, and the error raised when the key is missing names the variable and the
file it looked in, and nothing else.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_dotenv", "load_api_key", "MissingAPIKey", "Secret"]

ENV_VAR = "OPENAI_API_KEY"


class MissingAPIKey(RuntimeError):
    pass


class Secret(str):
    """A string that does not print itself.

    An API key held in an ordinary `str` leaks through any traceback that
    happens to have it in scope -- pytest renders the arguments of every frame
    it shows, which is how this one first got out. Comparison, slicing and
    `os.environ` all still work; only the representation is blanked.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<redacted>"

    def __str__(self) -> str:  # noqa: D105 - so f-strings cannot leak it either
        return "<redacted>"

    def reveal(self) -> str:
        return str.__str__(self)


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Parse a `.env` file into a dict. Missing file is not an error."""
    path = Path(path) if path is not None else _find_dotenv()
    values: dict[str, str] = {}
    if path is None or not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, raw = line.partition("=")
        if not sep:
            continue
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            raw = raw[1:-1]
        values[key.strip()] = raw
    return values


def load_api_key(path: str | Path | None = None) -> Secret:
    """The environment wins; `.env` is the fallback."""
    key = os.environ.get(ENV_VAR)
    if key:
        return Secret(key)
    key = load_dotenv(path).get(ENV_VAR)
    if key:
        return Secret(key)
    where = path or _find_dotenv() or Path(".env").resolve()
    raise MissingAPIKey(f"set {ENV_VAR} in the environment or in {where}")


def _find_dotenv() -> Path | None:
    """Nearest `.env` walking up from the working directory."""
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None
