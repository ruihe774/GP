"""Commentary you read instead of hear: a floating card in the corner.

Voice is not always the right channel -- a quiet room, a recording, a game whose
own audio matters -- and a remark that is *read* needs somewhere to land that is
above a fullscreen game and never in its way. That is an override-redirect X11
window: unmanaged, always in mutter's topmost layer, click-through, and
translucent for real (a 32-bit ARGB visual composited by the compositor, not a
screenshot of what is behind).

**X11 on a Wayland session is the deliberate choice, not a fallback.** Wayland
gives an ordinary client no way to place a surface above someone else's
fullscreen window -- that is `wlr-layer-shell`, which mutter does not implement,
and there is no portal for it. XWayland is present, mutter stacks
override-redirect X windows on top, and this is the whole of what the feature
needs. Everything here therefore talks to the X server directly through
`python-xlib`; there is no toolkit and no second main loop to reconcile with
asyncio.

Three layers, and only the middle one needs a display:

* pure layout and drawing (`ToastStack`, `wrap_text`, `render_card`, `place`) --
  Pillow in, Pillow out, testable with no X server at all;
* `HudWindow`, which is the only thing that knows what an X visual is;
* `TextHud`, which owns a thread so that `show()` never blocks its caller and a
  fade animation cannot stall an event loop.

Three things about the target machine that are not edge cases and are not
optional:

*Two monitors.* The X screen is the union of the outputs (8960x2880 here), so
"the top right of the screen" is off the side of a monitor. Placement is
computed against one `Monitor` rectangle in root coordinates, chosen by
`hud.monitor`.

*HiDPI.* XWayland reports device pixels. Every size in `HudConfig` is quoted in
design pixels at 96 dpi and scaled by `Monitor.scale`, discovered from `Xft.dpi`
(the compositor's own statement: 192, i.e. 2x, here) or from the monitor's
physical dpi. Unscaled, a 22 px card on a 5K panel is a smudge.

*Premultiplied alpha.* Depth-32 X windows are composited under the Render
convention. Straight alpha is not merely a little washed out; the rounded
corners and every glyph edge come out ringed with bright fringes.

One more thing worth knowing before editing: `python-xlib` 0.33 ships no XFixes
region API, so click-through is done with the SHAPE extension's `Input` kind and
an empty rectangle list -- which is what `XFixesSetWindowShapeRegion` sets
anyway. And `put_image` does not split requests, so a card wider than
`max_request_length` (256 KiB of pixels, i.e. most of one at 2x) has to be sent
in row bands by hand.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import HudConfig

log = logging.getLogger(__name__)

__all__ = [
    "Entry",
    "ToastStack",
    "Monitor",
    "Metrics",
    "HudWindow",
    "TextHud",
    "NullHud",
    "make_hud",
    "hold_for",
    "wrap_text",
    "render_card",
    "place",
    "load_font",
    "fade_alpha",
    "pack_pixels",
]

#: the dpi `HudConfig`'s `*_px` values are quoted at
DESIGN_DPI = 96.0
#: nobody has a 5x display, and a bad `width_in_millimeters` should not produce
#: a card larger than the screen
MAX_SCALE = 4.0
#: repaint cadence while an entry is fading out. 25 fps of a small card is
#: nothing next to a game, and anything slower is visibly steppy.
FADE_FRAME_S = 0.04
#: fonts to try when fontconfig is unavailable or answers with something Pillow
#: cannot open
FALLBACK_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


# -- entries ---------------------------------------------------------------


@dataclass
class Entry:
    """One remark on screen, with the moment it stops being one."""

    text: str
    shown_at: float
    expires_at: float


def hold_for(text: str, cfg: HudConfig) -> float:
    """How long a remark stays up: reading time, bounded at both ends.

    A nine-word aside and a three-line explanation are not on screen for the
    same length of time, and a constant hold gets one of the two wrong.
    """
    seconds = len(text) / max(cfg.read_cps, 1e-6)
    return min(max(seconds, cfg.hold_min_s), cfg.hold_max_s)


def fade_alpha(entry: Entry, now: float, fade_s: float) -> float:
    """1.0 while an entry is current, ramping to 0 over its last `fade_s`."""
    if fade_s <= 0:
        return 1.0 if now < entry.expires_at else 0.0
    remaining = entry.expires_at - now
    if remaining >= fade_s:
        return 1.0
    return max(0.0, remaining / fade_s)


class ToastStack:
    """What is on screen, and when that next changes.

    The clock is injected for the same reason `PlaybackTimer`'s is: expiry and
    fading are the whole behaviour here, and neither should need a display or a
    real second to test.
    """

    def __init__(self, cfg: HudConfig, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self._entries: list[Entry] = []

    def add(self, text: str, now: float | None = None) -> Entry:
        now = self.clock() if now is None else now
        entry = Entry(text=text, shown_at=now, expires_at=now + hold_for(text, self.cfg))
        self._entries.append(entry)
        # The oldest falls off the top rather than the card growing forever.
        keep = max(1, self.cfg.max_entries)
        del self._entries[:-keep]
        return entry

    def visible(self, now: float | None = None) -> list[Entry]:
        """Live entries, oldest first. Expired ones are dropped here."""
        now = self.clock() if now is None else now
        self._entries = [e for e in self._entries if e.expires_at > now]
        return list(self._entries)

    def next_wake(self, now: float | None = None) -> float | None:
        """Seconds until the card next needs redrawing, or None if it is empty."""
        now = self.clock() if now is None else now
        entries = [e for e in self._entries if e.expires_at > now]
        if not entries:
            return None
        fade_s = self.cfg.fade_ms / 1000.0
        starts_fading = min(e.expires_at - fade_s for e in entries)
        if now >= starts_fading:
            return min(FADE_FRAME_S, max(e.expires_at for e in entries) - now)
        return starts_fading - now

    def clear(self) -> None:
        self._entries.clear()


# -- geometry --------------------------------------------------------------


@dataclass(frozen=True)
class Monitor:
    """One output, in root coordinates, with how dense its pixels are.

    Root coordinates because that is what a window position is: on this machine
    the primary monitor starts at x=3840, so a card placed at "the right edge"
    of the *screen* would land on the other display.
    """

    x: int
    y: int
    width: int
    height: int
    scale: float = 1.0
    name: str = ""
    primary: bool = False


@dataclass(frozen=True)
class Metrics:
    """`HudConfig`'s design pixels, resolved against one monitor."""

    scale: float
    font_px: int
    line_px: int
    pad: int
    radius: int
    margin: int
    width: int
    accent_px: int
    rule_px: int
    shadow_px: int
    stroke_px: int

    @classmethod
    def for_monitor(cls, cfg: HudConfig, monitor: Monitor) -> Metrics:
        scale = max(0.1, monitor.scale)

        def px(value: float, floor: int = 0) -> int:
            return max(floor, int(round(value * scale)))

        font_px = px(cfg.font_size_px, 1)
        # Floors first, then the monitor: fitting on screen is the rule that
        # wins, however wide the minimum or the type says the card should be.
        width = max(
            int(round(monitor.width * cfg.width_frac)), px(cfg.min_width_px, 1), font_px * 4
        )
        width = min(width, max(font_px, monitor.width - 2 * px(cfg.margin_px)))
        return cls(
            scale=scale,
            font_px=font_px,
            line_px=max(font_px, int(round(font_px * cfg.line_spacing))),
            pad=px(cfg.padding_px),
            radius=px(cfg.corner_radius_px),
            margin=px(cfg.margin_px),
            width=width,
            accent_px=px(3, 1),
            rule_px=max(1, px(1)),
            shadow_px=max(1, px(1)),
            stroke_px=px(cfg.stroke_width_px, 1),
        )


def place(
    card: tuple[int, int], monitor: Monitor, metrics: Metrics, cfg: HudConfig
) -> tuple[int, int]:
    """Where the card goes, in root coordinates.

    The anchored edge is the fixed one: a `top-*` card grows downward and a
    `bottom-*` card grows upward, so a new remark never shoves the others across
    the screen.
    """
    width, height = card
    anchor = cfg.anchor
    if anchor.endswith("left"):
        x = monitor.x + metrics.margin
    elif anchor.endswith("right"):
        x = monitor.x + monitor.width - width - metrics.margin
    else:
        x = monitor.x + (monitor.width - width) // 2
    if anchor.startswith("top"):
        y = monitor.y + metrics.margin
    else:
        y = monitor.y + monitor.height - height - metrics.margin
    return x, y


# -- type ------------------------------------------------------------------


def _fontconfig_path(family: str, lang: str) -> str | None:
    """Ask fontconfig for a file, so "sans-serif" means whatever is installed."""
    if not shutil.which("fc-match"):
        return None
    pattern = family + (f":lang={lang}" if lang else "")
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}", pattern],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        log.debug("fc-match failed for %r", pattern, exc_info=True)
        return None
    path = out.stdout.strip()
    return path or None


def load_font(cfg: HudConfig, size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Resolve one font file for the whole card.

    Pillow does no font fallback -- one file renders every glyph, and DejaVu Sans
    has no CJK at all -- so a Japanese session that leaves `hud.font_lang` empty
    gets a card of tofu. Routing the lookup through fontconfig is what makes
    `font_lang = "ja"` enough to fix that.
    """
    candidates = [p for p in (cfg.font_path, _fontconfig_path(cfg.font_family, cfg.font_lang)) if p]
    candidates.extend(FALLBACK_FONTS)
    for path in candidates:
        try:
            return ImageFont.truetype(path, size_px)
        except OSError:
            log.debug("cannot open font %s", path, exc_info=True)
    log.warning("no usable font found; falling back to Pillow's built-in")
    return ImageFont.load_default(size_px)


def _longest_prefix(text: str, font: Any, max_px: float) -> int:
    """How many characters of `text` fit in `max_px`, at least one."""
    low, high = 1, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if font.getlength(text[:mid]) <= max_px:
            low = mid
        else:
            high = mid - 1
    return low


def wrap_text(text: str, font: Any, max_px: float) -> list[str]:
    """Greedy wrap by measured width, breaking inside a run when it must.

    The character-level break is not a corner case: `agent.language = "ja"` is a
    supported setting, and Japanese arrives as one unbroken run with no spaces
    to wrap on.
    """
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in re.findall(r"\S+\s*", paragraph) or [""]:
            candidate = line + word
            if not line or font.getlength(candidate.rstrip()) <= max_px:
                line = candidate
            else:
                lines.append(line.rstrip())
                line = word
            while font.getlength(line.rstrip()) > max_px and len(line.rstrip()) > 1:
                cut = _longest_prefix(line.rstrip(), font, max_px)
                lines.append(line[:cut])
                line = line[cut:]
        lines.append(line.rstrip())
    return lines or [""]


# -- drawing ---------------------------------------------------------------


def _rgb(color: str) -> tuple[int, int, int]:
    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) not in (6, 8):
        raise ValueError(f"colour must be #rgb, #rrggbb or #rrggbbaa: {color!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _scaled_alpha(layer: Image.Image, factor: float) -> Image.Image:
    """Fade a whole layer, keeping its own per-pixel alpha (glyph antialiasing)."""
    if factor >= 1.0:
        return layer
    alpha = layer.getchannel("A").point(lambda v: int(v * factor))
    layer.putalpha(alpha)
    return layer


def render_card(
    entries: Sequence[Entry],
    fades: Sequence[float],
    cfg: HudConfig,
    metrics: Metrics,
    font: Any,
) -> Image.Image:
    """The whole overlay as one RGBA image, oldest entry at the top.

    Each entry is composited as its own layer rather than drawn straight onto
    the card: `ImageDraw` *replaces* alpha instead of blending it, so a
    half-transparent glyph drawn directly would punch a hole through the panel
    behind it.

    `cfg.style == "stroked"` skips the panel, accent bar and rules entirely --
    there is no card, just text -- and outlines each glyph in `bg_color`
    instead of a drop shadow, since a shadow reads as a shadow only against a
    panel and disappears into an arbitrary game frame.
    """
    bordered = cfg.style == "card"
    pad, line_px = metrics.pad, metrics.line_px
    text_left = pad + (metrics.accent_px + pad if bordered else 0)
    text_width = max(1, metrics.width - text_left - pad)
    blocks = [wrap_text(e.text, font, text_width) for e in entries]
    heights = [len(lines) * line_px + 2 * pad for lines in blocks]
    rule_gap = metrics.rule_px if bordered else 0
    height = sum(heights) + rule_gap * max(0, len(entries) - 1)
    card = Image.new("RGBA", (metrics.width, max(1, height)), (0, 0, 0, 0))
    if not entries:
        return card

    bg = _rgb(cfg.bg_color)
    if bordered:
        # The panel is as opaque as the least faded thing on it, so the
        # background goes with the last remark rather than blinking out from
        # under it.
        panel_fade = max(fades) if fades else 1.0
        panel = Image.new("RGBA", card.size, (0, 0, 0, 0))
        ImageDraw.Draw(panel).rounded_rectangle(
            (0, 0, card.width - 1, card.height - 1),
            radius=metrics.radius,
            fill=(*bg, int(255 * min(1.0, max(0.0, cfg.bg_opacity)))),
        )
        card = Image.alpha_composite(card, _scaled_alpha(panel, panel_fade))

    fg = _rgb(cfg.fg_color)
    accent = _rgb(cfg.accent_color)
    newest = len(entries) - 1
    top = 0
    for index, (lines, block_h) in enumerate(zip(blocks, heights, strict=True)):
        layer = Image.new("RGBA", card.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        if bordered and index == newest:
            draw.rounded_rectangle(
                (pad, top + pad, pad + metrics.accent_px, top + block_h - pad),
                radius=metrics.accent_px // 2,
                fill=(*accent, 255),
            )
        for row, line in enumerate(lines):
            y = top + pad + row * line_px
            if bordered:
                # A hairline shadow, not a blur: the card sits over a game
                # frame that may be any colour, and one dark pixel of offset
                # is the cheapest thing that keeps light text on light
                # scenery readable.
                draw.text((text_left + metrics.shadow_px, y + metrics.shadow_px), line,
                          font=font, fill=(0, 0, 0, 160))
                draw.text((text_left, y), line, font=font, fill=(*fg, 255))
            else:
                draw.text(
                    (text_left, y), line, font=font, fill=(*fg, 255),
                    stroke_width=metrics.stroke_px, stroke_fill=(*bg, 255),
                )
        dim = 1.0 if index == newest else max(0.0, min(1.0, cfg.dim_older))
        card = Image.alpha_composite(card, _scaled_alpha(layer, fades[index] * dim))

        top += block_h
        if bordered and index != newest:
            rule = Image.new("RGBA", card.size, (0, 0, 0, 0))
            ImageDraw.Draw(rule).rectangle(
                (pad, top, card.width - pad, top + metrics.rule_px - 1),
                fill=(*fg, 40),
            )
            card = Image.alpha_composite(card, _scaled_alpha(rule, fades[index]))
            top += metrics.rule_px
    return card


def pack_pixels(image: Image.Image, byte_order: int) -> bytes:
    """RGBA to what the X server wants in a depth-32 ZPixmap.

    Two conversions, both mandatory. Alpha is premultiplied because a
    composited depth-32 window is interpreted under the Render convention --
    straight alpha rings every glyph and rounded corner with a bright fringe.
    Then the channels are ordered for the server: an ARGB pixel is stored
    B,G,R,A on an LSBFirst server and A,R,G,B on an MSBFirst one.
    """
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint16)
    alpha = arr[..., 3:4]
    arr[..., :3] = (arr[..., :3] * alpha + 127) // 255
    out = arr.astype(np.uint8)
    order = [3, 0, 1, 2] if byte_order else [2, 1, 0, 3]
    return out[..., order].tobytes()


# -- the window ------------------------------------------------------------


class HudWindow:
    """The X11 surface: override-redirect, ARGB, click-through, always raised."""

    def __init__(self, cfg: HudConfig):
        self.cfg = cfg
        #: python-xlib handles, untyped at the source (as with GStreamer's)
        self._X: Any = None
        self._display: Any = None
        self._screen: Any = None
        self._window: Any = None
        self._pixmap: Any = None
        self._mapped = False

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        if not (self.cfg.display or os.environ.get("DISPLAY")):
            raise RuntimeError("no X display: set DISPLAY, or hud.display")
        from Xlib import X
        from Xlib import display as xdisplay

        self._X = X
        self._display = xdisplay.Display(self.cfg.display or None)
        self._screen = self._display.screen()
        visual = self._argb_visual()
        root = self._screen.root
        colormap = root.create_colormap(visual, X.AllocNone)
        # A non-default visual needs its own colormap *and* an explicit
        # border_pixel; inheriting either from the parent is a BadMatch.
        self._window = root.create_window(
            0,
            0,
            1,
            1,
            0,
            32,
            X.InputOutput,
            visual,
            colormap=colormap,
            background_pixel=0,
            border_pixel=0,
            override_redirect=True,
            event_mask=X.ExposureMask,
        )
        self._window.set_wm_name("gpagent hud")
        self._window.set_wm_class("hud", "gpagent")
        if self.cfg.click_through:
            self._make_click_through()
        log.info("hud window open on %s", self._display.get_display_name())

    def _argb_visual(self) -> int:
        """A depth-32 TrueColor visual, or there is no translucency to be had."""
        for depth in self._screen.allowed_depths:
            if depth.depth != 32:
                continue
            for visual in depth.visuals:
                # TrueColor(4) before DirectColor(5): both work, the first is
                # what every compositor actually composites.
                if visual.visual_class == 4:
                    return visual.visual_id
            if depth.visuals:
                return depth.visuals[0].visual_id
        raise RuntimeError("no 32-bit ARGB visual: the X server cannot do a translucent HUD")

    def _make_click_through(self) -> None:
        """An empty input shape: every click lands on the game underneath.

        python-xlib 0.33 has no XFixes region API, so this is the SHAPE
        extension's `Input` kind -- the same server-side state
        `XFixesSetWindowShapeRegion` would set.
        """
        from Xlib.ext import shape

        if not self._display.has_extension("SHAPE"):
            log.warning("no SHAPE extension: the hud will swallow clicks that land on it")
            return
        self._window.shape_rectangles(shape.SO.Set, shape.SK.Input, 0, 0, 0, [])

    def close(self) -> None:
        try:
            self._free_pixmap()
            if self._window is not None:
                self._window.destroy()
            if self._display is not None:
                self._display.close()
        except Exception:
            log.debug("hud close failed", exc_info=True)
        finally:
            self._window = None
            self._display = None
            self._mapped = False

    # -- geometry -----------------------------------------------------------

    def monitors(self) -> list[Monitor]:
        """Every output in root coordinates, each with its own scale."""
        scale = self.cfg.scale or self._xft_scale()
        found: list[Monitor] = []
        try:
            if self._display.has_extension("RANDR"):
                for mon in self._screen.root.xrandr_get_monitors().monitors:
                    width = mon.width_in_pixels
                    found.append(
                        Monitor(
                            x=mon.x,
                            y=mon.y,
                            width=width,
                            height=mon.height_in_pixels,
                            scale=scale or _dpi_scale(width, mon.width_in_millimeters),
                            name=self._display.get_atom_name(mon.name),
                            primary=bool(mon.primary),
                        )
                    )
        except Exception:
            # RandR is how you find the outputs, not how you draw on them: a
            # server without it still gets one monitor the size of the screen.
            log.debug("xrandr monitor query failed", exc_info=True)
        if not found:
            width = self._screen.width_in_pixels
            found.append(
                Monitor(
                    x=0,
                    y=0,
                    width=width,
                    height=self._screen.height_in_pixels,
                    scale=scale or _dpi_scale(width, self._screen.width_in_mms),
                    name="screen",
                    primary=True,
                )
            )
        return found

    def monitor(self) -> Monitor:
        return pick_monitor(self.monitors(), self.cfg.monitor)

    def _xft_scale(self) -> float:
        """What the compositor tells X clients the scale is, via Xft.dpi."""
        try:
            from Xlib import Xatom

            prop = self._screen.root.get_full_property(Xatom.RESOURCE_MANAGER, Xatom.STRING)
            if prop is None:
                return 0.0
            raw = prop.value
            text = raw.decode("utf8", "replace") if isinstance(raw, bytes) else str(raw)
            match = re.search(r"^Xft\.dpi:\s*(\d+(?:\.\d+)?)\s*$", text, re.MULTILINE)
            if match:
                return _clamp_scale(float(match.group(1)) / DESIGN_DPI)
        except Exception:
            # No resource manager at all is a perfectly ordinary X server.
            log.debug("Xft.dpi lookup failed", exc_info=True)
        return 0.0

    # -- painting -----------------------------------------------------------

    def blit(self, image: Image.Image, x: int, y: int) -> None:
        """Put the card on screen at root coordinates (x, y), and raise it."""
        X = self._X
        width, height = image.size
        data = pack_pixels(image, self._display.display.info.image_byte_order)
        pixmap = self._screen.root.create_pixmap(width, height, 32)
        gc = pixmap.create_gc()
        for top, rows in _bands(height, width * 4, self._max_request_bytes()):
            offset = top * width * 4
            pixmap.put_image(
                gc, 0, top, width, rows, X.ZPixmap, 32, 0, data[offset : offset + rows * width * 4]
            )
        gc.free()
        # Painting into the window's *background* means the server repaints
        # exposures itself, from its own copy -- no expose handling, and no
        # flicker when another window uncovers us.
        self._window.change_attributes(background_pixmap=pixmap.id)
        self._window.configure(
            x=x, y=y, width=width, height=height, stack_mode=X.Above
        )
        if not self._mapped:
            self._window.map()
            self._mapped = True
        self._window.clear_area()
        self._display.flush()
        # The server keeps its own reference to a background pixmap, so the
        # previous one is only safe to free once this one has replaced it.
        self._free_pixmap()
        self._pixmap = pixmap

    def _max_request_bytes(self) -> int:
        """Payload that fits in one PutImage, leaving room for its header.

        `max_request_length` is in four-byte units, and python-xlib does not
        split a request that overruns it -- it just fails.
        """
        words = self._display.display.info.max_request_length
        return max(4096, words * 4 - 256)

    def hide(self) -> None:
        if self._mapped and self._window is not None:
            self._window.unmap()
            self._display.flush()
            self._mapped = False

    def drain_events(self) -> None:
        """Discard what the server sends us. Nothing here needs input."""
        if self._display is None:
            return
        for _ in range(self._display.pending_events()):
            self._display.next_event()

    def _free_pixmap(self) -> None:
        if self._pixmap is not None:
            self._pixmap.free()
            self._pixmap = None


def _bands(height: int, row_bytes: int, budget: int) -> list[tuple[int, int]]:
    """Split an image into (first row, row count) chunks that fit one request."""
    rows = max(1, budget // max(1, row_bytes))
    return [(top, min(rows, height - top)) for top in range(0, height, rows)]


def _clamp_scale(scale: float) -> float:
    return min(MAX_SCALE, max(1.0, scale))


def _dpi_scale(pixels: int, millimetres: int) -> float:
    """Physical density as a scale factor, when nobody has told us one."""
    if not millimetres or pixels <= 0:
        return 1.0
    dpi = pixels / (millimetres / 25.4)
    # Whole steps. A panel's reported millimetres are loose enough that a
    # continuous factor is false precision -- this 5K reads as 217 dpi, i.e.
    # 2.26x, where every toolkit on the machine has settled on plain 2x.
    return _clamp_scale(round(dpi / DESIGN_DPI))


def pick_monitor(monitors: Sequence[Monitor], want: str) -> Monitor:
    """Resolve `hud.monitor`: "primary", an output name, or an index."""
    if not monitors:
        raise RuntimeError("no monitors")
    key = (want or "primary").strip()
    if key.lower() != "primary":
        for monitor in monitors:
            if monitor.name.lower() == key.lower():
                return monitor
        if key.isdigit() and int(key) < len(monitors):
            return monitors[int(key)]
        log.warning(
            "no monitor %r (have %s); using the primary",
            key,
            ", ".join(m.name for m in monitors),
        )
    for monitor in monitors:
        if monitor.primary:
            return monitor
    return monitors[0]


# -- the hud ---------------------------------------------------------------


class _Stop:
    """Sentinel: the thread is done."""


class _Clear:
    """Sentinel: drop everything on screen."""


class TextHud:
    """Shows remarks on screen. Safe to call from anywhere, including a loop.

    The X connection, the rendering and the expiry timers all live on one
    dedicated thread, the way `portal.py` parks its D-Bus connection: `show()`
    is a queue push that returns immediately, so neither a 10 ms render nor a
    fade animation can stall an asyncio event loop that has an agent to run.
    """

    name = "x11"

    def __init__(
        self,
        cfg: HudConfig,
        clock: Callable[[], float] = time.monotonic,
        *,
        window: HudWindow | None = None,
    ):
        self.cfg = cfg
        self.clock = clock
        self.window = window if window is not None else HudWindow(cfg)
        self.stack = ToastStack(cfg, clock)
        self.shown: list[str] = []
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._font: Any = None
        self._font_px = 0
        self._painted: tuple[Any, ...] | None = None

    # -- public api ---------------------------------------------------------

    def start(self) -> None:
        self.window.open()
        self._thread = threading.Thread(target=self._run, name="gpagent-hud", daemon=True)
        self._thread.start()

    def show(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.shown.append(text)
        self._queue.put(text)

    def clear(self) -> None:
        self._queue.put(_Clear)

    def close(self) -> None:
        self._queue.put(_Stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.window.close()

    # -- the thread ---------------------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                timeout = self.stack.next_wake()
                try:
                    command = self._queue.get(timeout=timeout)
                except queue.Empty:
                    command = None
                if command is _Stop:
                    break
                if command is _Clear:
                    self.stack.clear()
                elif isinstance(command, str):
                    self.stack.add(command)
                self.paint()
                self.window.drain_events()
        except Exception:
            # A dead X connection is a dead overlay, not a dead agent: log it
            # once, stop painting, leave everything else running.
            log.warning("hud stopped", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                self.window.hide()

    def paint(self) -> None:
        """Draw whatever is due right now. The thread's whole job, callable.

        Public because it is the seam: everything above it is pure, everything
        below it is X, and a test can drive one turn of the loop without a
        thread, a display or a real second.
        """
        now = self.clock()
        entries = self.stack.visible(now)
        if not entries:
            if self._painted is not None:
                self.window.hide()
                self._painted = None
            return
        monitor = self.window.monitor()
        metrics = Metrics.for_monitor(self.cfg, monitor)
        fades = [fade_alpha(e, now, self.cfg.fade_ms / 1000.0) for e in entries]
        signature = (
            tuple(e.text for e in entries),
            tuple(round(f, 2) for f in fades),
            monitor,
            metrics,
        )
        if signature == self._painted:
            return
        font = self._font_for(metrics.font_px)
        card = render_card(entries, fades, self.cfg, metrics, font)
        x, y = place(card.size, monitor, metrics, self.cfg)
        self.window.blit(card, x, y)
        self._painted = signature

    def _font_for(self, size_px: int) -> Any:
        if self._font is None or size_px != self._font_px:
            self._font = load_font(self.cfg, size_px)
            self._font_px = size_px
        return self._font


class NullHud:
    """Answers the same calls and shows nothing. No display, dry runs, CI."""

    name = "null"

    def __init__(self) -> None:
        self.shown: list[str] = []

    def start(self) -> None:
        return None

    def show(self, text: str) -> None:
        self.shown.append(text)

    def clear(self) -> None:
        return None

    def close(self) -> None:
        return None


def make_hud(
    cfg: HudConfig,
    clock: Callable[[], float] = time.monotonic,
    *,
    window: HudWindow | None = None,
) -> TextHud | NullHud:
    """A real overlay when asked for and available, a null one otherwise."""
    if not cfg.enabled:
        return NullHud()
    hud = TextHud(cfg, clock, window=window)
    try:
        hud.start()
    except Exception:
        log.warning("hud unavailable; running without it", exc_info=True)
        return NullHud()
    return hud
