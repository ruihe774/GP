"""The on-screen commentary overlay: layout, expiry, and what reaches X.

Nothing here needs a display. The X server is faked the way `_FakeGst` fakes
GStreamer in test_agent_parts.py -- real `Xlib` constants, recorded requests --
so the parts that are easy to get wrong (a 32-bit visual, banded `PutImage`,
premultiplied bytes in the server's own order) are checked without one. The one
test that wants a real display is marked `hardware`.
"""

from __future__ import annotations

import os
import time

import pytest
from PIL import Image

from gpagent.agent.config import HudConfig
from gpagent.agent.hud import (
    FADE_FRAME_S,
    Entry,
    HudWindow,
    Metrics,
    Monitor,
    NullHud,
    TextHud,
    ToastStack,
    fade_alpha,
    hold_for,
    load_font,
    make_hud,
    pack_pixels,
    pick_monitor,
    place,
    render_card,
    wrap_text,
)

#: A 5K panel at 2x that does *not* start at x=0, because the monitor to its
#: left occupies the first 3840 columns of the X screen. Anything that confuses
#: "the screen" with "a monitor" places the card on the wrong display, so this
#: is the default fixture rather than a special case.
PRIMARY = Monitor(x=3840, y=0, width=5120, height=2880, scale=2.0, name="DP-1", primary=True)
SIDE = Monitor(x=0, y=428, width=3840, height=2160, scale=2.0, name="HDMI-1")
#: ...and the layouts that must work just as well. One machine's geometry is a
#: fixture, not a specification: every placement rule below is checked against
#: all of these.
LAYOUTS = [
    Monitor(0, 0, 1920, 1080, scale=1.0, name="laptop", primary=True),
    PRIMARY,
    SIDE,
    Monitor(1920, -600, 1440, 2560, scale=1.0, name="portrait-above"),
    Monitor(0, 0, 3840, 2160, scale=1.5, name="fractional"),
]
ANCHORS = ["top-right", "top-left", "top", "bottom-right", "bottom-left", "bottom"]


class Mono:
    """A font of fixed advance, so wrapping is arithmetic rather than a guess."""

    def getlength(self, text: str) -> float:
        return 10.0 * len(text)


def entries(*texts, at=0.0, hold=10.0):
    return [Entry(text=t, shown_at=at, expires_at=at + hold) for t in texts]


# -- timing ----------------------------------------------------------------


class TestHold:
    def test_a_longer_remark_stays_up_longer(self):
        cfg = HudConfig(read_cps=10.0, hold_min_s=0.0, hold_max_s=100.0)
        assert hold_for("x" * 100, cfg) == pytest.approx(10.0)
        assert hold_for("x" * 50, cfg) == pytest.approx(5.0)

    def test_a_glance_still_gets_the_floor(self):
        cfg = HudConfig(read_cps=10.0, hold_min_s=4.0, hold_max_s=100.0)
        assert hold_for("Oh no.", cfg) == 4.0

    def test_an_essay_is_capped(self):
        cfg = HudConfig(read_cps=10.0, hold_min_s=1.0, hold_max_s=8.0)
        assert hold_for("x" * 1000, cfg) == 8.0

    def test_an_entry_is_solid_until_its_fade_begins(self):
        entry = Entry("hi", shown_at=0.0, expires_at=10.0)
        assert fade_alpha(entry, 5.0, 1.0) == 1.0
        assert fade_alpha(entry, 9.5, 1.0) == pytest.approx(0.5)
        assert fade_alpha(entry, 10.0, 1.0) == 0.0

    def test_without_a_fade_an_entry_is_on_or_off(self):
        entry = Entry("hi", shown_at=0.0, expires_at=10.0)
        assert fade_alpha(entry, 9.99, 0.0) == 1.0
        assert fade_alpha(entry, 10.0, 0.0) == 0.0


class TestToastStack:
    def stack(self, **kwargs):
        cfg = HudConfig(hold_min_s=10.0, hold_max_s=10.0, fade_ms=250, **kwargs)
        return ToastStack(cfg, clock=lambda: 0.0)

    def test_a_remark_appears(self):
        stack = self.stack()
        stack.add("Oh no.", 0.0)
        assert [e.text for e in stack.visible(1.0)] == ["Oh no."]

    def test_the_newest_remarks_are_the_ones_kept(self):
        stack = self.stack(max_entries=2)
        for text in ("first", "second", "third"):
            stack.add(text, 0.0)
        assert [e.text for e in stack.visible(0.0)] == ["second", "third"]

    def test_a_remark_leaves_when_its_time_is_up(self):
        stack = self.stack()
        stack.add("Oh no.", 0.0)
        assert stack.visible(9.9)
        assert stack.visible(10.1) == []

    def test_the_next_wake_is_when_the_first_fade_starts(self):
        stack = self.stack()
        stack.add("Oh no.", 0.0)
        assert stack.next_wake(0.0) == pytest.approx(9.75)

    def test_a_fading_card_wakes_at_frame_rate(self):
        stack = self.stack()
        stack.add("Oh no.", 0.0)
        assert stack.next_wake(9.8) == pytest.approx(FADE_FRAME_S)

    def test_an_empty_stack_asks_to_sleep_indefinitely(self):
        stack = self.stack()
        assert stack.next_wake(0.0) is None
        stack.add("Oh no.", 0.0)
        stack.clear()
        assert stack.next_wake(0.0) is None


# -- layout ----------------------------------------------------------------


class TestWrap:
    def test_a_line_that_fits_stays_one_line(self):
        assert wrap_text("four words fit here", Mono(), 1000) == ["four words fit here"]

    def test_wrapped_lines_fit_and_keep_every_word(self):
        text = "That jump was optimistic, you had the speed but not the angle"
        lines = wrap_text(text, Mono(), 200)
        assert all(Mono().getlength(line) <= 200 for line in lines)
        assert " ".join(lines) == text

    def test_a_run_longer_than_the_card_is_broken_up(self):
        lines = wrap_text("x" * 55, Mono(), 200)
        assert lines == ["x" * 20, "x" * 20, "x" * 15]

    def test_japanese_wraps_without_spaces_to_wrap_on(self):
        text = "ナイス、そのショートカットで一周分ぐらい稼いだよ。"
        lines = wrap_text(text, Mono(), 100)
        assert len(lines) == 3
        assert "".join(lines) == text


class TestMetrics:
    def test_sizes_follow_the_monitor_scale(self):
        cfg = HudConfig(font_size_px=22, padding_px=14, margin_px=32)
        one = Metrics.for_monitor(cfg, Monitor(0, 0, 1920, 1080, scale=1.0))
        two = Metrics.for_monitor(cfg, PRIMARY)
        assert (one.font_px, one.pad, one.margin) == (22, 14, 32)
        assert (two.font_px, two.pad, two.margin) == (44, 28, 64)

    def test_a_fractional_scale_is_honoured_too(self):
        cfg = HudConfig(font_size_px=22)
        assert Metrics.for_monitor(cfg, Monitor(0, 0, 3840, 2160, scale=1.5)).font_px == 33

    @pytest.mark.parametrize("monitor", LAYOUTS, ids=lambda m: m.name)
    def test_the_card_is_a_fraction_of_its_own_monitor(self, monitor):
        cfg = HudConfig(width_frac=0.25, min_width_px=1)
        assert Metrics.for_monitor(cfg, monitor).width == round(monitor.width * 0.25)

    @pytest.mark.parametrize("monitor", LAYOUTS, ids=lambda m: m.name)
    def test_a_narrow_fraction_is_floored_at_the_minimum(self, monitor):
        cfg = HudConfig(width_frac=0.01, min_width_px=320)
        assert Metrics.for_monitor(cfg, monitor).width == round(320 * monitor.scale)

    @pytest.mark.parametrize("monitor", LAYOUTS, ids=lambda m: m.name)
    def test_a_greedy_fraction_still_leaves_its_margins(self, monitor):
        cfg = HudConfig(width_frac=2.0, margin_px=32)
        metrics = Metrics.for_monitor(cfg, monitor)
        assert metrics.width == monitor.width - 2 * metrics.margin

    def test_fitting_on_screen_beats_every_other_floor(self):
        cfg = HudConfig(min_width_px=1200, font_size_px=90, margin_px=8)
        metrics = Metrics.for_monitor(cfg, Monitor(0, 0, 640, 480, scale=1.0))
        assert metrics.width <= 640


class TestPlacement:
    def card_on(self, monitor, anchor, size=(1000, 400), margin_px=32):
        cfg = HudConfig(anchor=anchor, margin_px=margin_px)
        metrics = Metrics.for_monitor(cfg, monitor)
        return place(size, monitor, metrics, cfg), metrics

    def test_a_card_lands_on_the_monitor_it_was_measured_against(self):
        (x, y), _ = self.card_on(PRIMARY, "top-right")
        assert (x, y) == (3840 + 5120 - 1000 - 64, 64)

    @pytest.mark.parametrize("monitor", LAYOUTS, ids=lambda m: m.name)
    @pytest.mark.parametrize("anchor", ANCHORS)
    def test_the_whole_card_stays_on_its_own_monitor(self, monitor, anchor):
        width, height = 400, 200
        (x, y), _ = self.card_on(monitor, anchor, size=(width, height))
        assert monitor.x <= x and x + width <= monitor.x + monitor.width
        assert monitor.y <= y and y + height <= monitor.y + monitor.height

    @pytest.mark.parametrize("monitor", LAYOUTS, ids=lambda m: m.name)
    @pytest.mark.parametrize("anchor", ANCHORS)
    def test_the_anchored_edges_keep_exactly_their_margin(self, monitor, anchor):
        width, height = 400, 200
        (x, y), metrics = self.card_on(monitor, anchor, size=(width, height))
        if anchor.endswith("left"):
            assert x - monitor.x == metrics.margin
        elif anchor.endswith("right"):
            assert (monitor.x + monitor.width) - (x + width) == metrics.margin
        else:
            # centred: the same slack either side, give or take an odd pixel
            assert abs((x - monitor.x) - (monitor.x + monitor.width - x - width)) <= 1
        if anchor.startswith("top"):
            assert y - monitor.y == metrics.margin
        else:
            assert (monitor.y + monitor.height) - (y + height) == metrics.margin

    @pytest.mark.parametrize("monitor", LAYOUTS, ids=lambda m: m.name)
    @pytest.mark.parametrize("anchor", ANCHORS)
    def test_the_anchored_edge_holds_still_as_the_card_grows(self, monitor, anchor):
        (sx, sy), _ = self.card_on(monitor, anchor, size=(400, 200))
        (tx, ty), _ = self.card_on(monitor, anchor, size=(400, 600))
        assert sx == tx
        if anchor.startswith("top"):
            assert sy == ty
        else:
            assert sy + 200 == ty + 600

    def test_the_margin_is_a_design_size_so_it_scales_with_the_display(self):
        one_x, _ = self.card_on(Monitor(0, 0, 1920, 1080, scale=1.0), "top-left")[0]
        two_x, _ = self.card_on(Monitor(0, 0, 1920, 1080, scale=2.0), "top-left")[0]
        assert (one_x, two_x) == (32, 64)

    def test_a_second_monitor_is_addressed_by_name(self):
        assert pick_monitor([PRIMARY, SIDE], "HDMI-1") is SIDE
        assert pick_monitor([PRIMARY, SIDE], "hdmi-1") is SIDE

    def test_the_primary_is_the_default_and_the_fallback(self):
        assert pick_monitor([SIDE, PRIMARY], "primary") is PRIMARY
        assert pick_monitor([SIDE, PRIMARY], "DVI-9") is PRIMARY

    def test_a_monitor_can_be_picked_by_index(self):
        assert pick_monitor([PRIMARY, SIDE], "1") is SIDE


# -- drawing ---------------------------------------------------------------


class TestRenderCard:
    def card(self, texts, fades=None, **kwargs):
        cfg = HudConfig(**kwargs)
        metrics = Metrics.for_monitor(cfg, PRIMARY)
        font = load_font(cfg, metrics.font_px)
        items = entries(*texts)
        return cfg, render_card(items, fades or [1.0] * len(items), cfg, metrics, font)

    def test_the_card_is_rgba_at_the_measured_width(self):
        cfg, card = self.card(["Oh no."])
        assert card.mode == "RGBA"
        assert card.width == Metrics.for_monitor(cfg, PRIMARY).width

    def test_more_remarks_make_a_taller_card(self):
        _, one = self.card(["Oh no."])
        _, three = self.card(["Oh no.", "Again.", "And again."])
        assert three.height > one.height

    def test_a_wrapped_remark_is_taller_than_a_short_one(self):
        _, short = self.card(["Oh no."])
        _, long = self.card(["That jump was optimistic. " * 6])
        assert long.height > short.height

    def test_the_panel_is_as_opaque_as_asked(self):
        _, card = self.card(["Oh no."], bg_opacity=0.8)
        assert card.getchannel("A").getextrema()[1] >= int(255 * 0.8)

    def test_a_fully_faded_card_is_invisible(self):
        _, card = self.card(["Oh no."], [0.0])
        assert card.getchannel("A").getextrema() == (0, 0)

    def test_a_half_faded_card_is_on_its_way_out(self):
        _, solid = self.card(["Oh no."])
        _, faded = self.card(["Oh no."], [0.5])
        assert 0 < faded.getchannel("A").getextrema()[1] < solid.getchannel("A").getextrema()[1]

    def test_stroked_has_no_panel_behind_the_text(self):
        """Borderless: nothing painted except the glyphs and their outline."""
        _, card = self.card(["Oh"], style="stroked", bg_opacity=1.0)
        # A solid "card" panel would make every pixel opaque; stroked leaves
        # the gaps between glyphs transparent.
        assert card.getchannel("A").getextrema()[0] == 0

    def test_stroked_is_narrower_than_a_card_with_no_text_change(self):
        # No accent-bar gutter reserved on the left when there is no accent bar.
        _, card = self.card(["Oh no."], style="card")
        cfg = HudConfig(style="stroked")
        metrics = Metrics.for_monitor(cfg, PRIMARY)
        font = load_font(cfg, metrics.font_px)
        stroked = render_card(entries("Oh no."), [1.0], cfg, metrics, font)
        assert stroked.height <= card.height


class TestPixelPacking:
    def image(self, rgba):
        img = Image.new("RGBA", (2, 1))
        img.putpixel((0, 0), rgba)
        img.putpixel((1, 0), rgba)
        return img

    def test_an_lsb_first_server_gets_bgra(self):
        packed = pack_pixels(self.image((10, 20, 30, 255)), 0)
        assert packed[:4] == bytes((30, 20, 10, 255))

    def test_an_msb_first_server_gets_argb(self):
        packed = pack_pixels(self.image((10, 20, 30, 255)), 1)
        assert packed[:4] == bytes((255, 10, 20, 30))

    def test_alpha_is_premultiplied(self):
        packed = pack_pixels(self.image((255, 255, 255, 128)), 0)
        assert packed[:4] == bytes((128, 128, 128, 128))

    def test_every_pixel_is_four_bytes(self):
        card = Image.new("RGBA", (7, 5), (1, 2, 3, 255))
        assert len(pack_pixels(card, 0)) == 7 * 5 * 4


# -- the X server ----------------------------------------------------------


class FakeVisual:
    def __init__(self, visual_id, visual_class):
        self.visual_id = visual_id
        self.visual_class = visual_class


class FakeDepth:
    def __init__(self, depth, visuals):
        self.depth = depth
        self.visuals = visuals


class FakeGC:
    def __init__(self, drawable):
        self.drawable = drawable

    def free(self):
        self.drawable.gc_freed = True


class FakePixmap:
    def __init__(self, width, height, depth):
        self.width, self.height, self.depth = width, height, depth
        self.bands: list[tuple[int, int, bytes]] = []
        self.freed = False
        self.gc_freed = False
        self.id = 7000 + height

    def create_gc(self):
        return FakeGC(self)

    def put_image(self, gc, x, y, width, height, fmt, depth, left_pad, data):
        self.bands.append((y, height, data))

    def free(self):
        self.freed = True


class FakeWindow:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.shapes: list[tuple] = []
        self.configures: list[dict] = []
        self.attributes: list[dict] = []
        self.maps = 0
        self.clears = 0
        self.name = ""

    def set_wm_name(self, name):
        self.name = name

    def set_wm_class(self, *args):
        pass

    def shape_rectangles(self, *args):
        self.shapes.append(args)

    def change_attributes(self, **kwargs):
        self.attributes.append(kwargs)

    def configure(self, **kwargs):
        self.configures.append(kwargs)

    def map(self):
        self.maps += 1

    def clear_area(self, **kwargs):
        self.clears += 1

    def unmap(self):
        pass

    def destroy(self):
        pass


class FakeRoot:
    def __init__(self, monitors):
        self.window: FakeWindow | None = None
        self.pixmaps: list[FakePixmap] = []
        self._monitors = monitors

    def create_colormap(self, visual, alloc):
        self.colormap = visual
        return visual

    def create_window(self, x, y, w, h, border, depth, klass, visual, **kwargs):
        self.window = FakeWindow(dict(kwargs, depth=depth, visual=visual))
        return self.window

    def create_pixmap(self, width, height, depth):
        pixmap = FakePixmap(width, height, depth)
        self.pixmaps.append(pixmap)
        return pixmap

    def xrandr_get_monitors(self):
        return type("Reply", (), {"monitors": self._monitors})()

    def get_full_property(self, *args):
        return type("Prop", (), {"value": b"Xft.dpi:\t192\n"})()


class FakeScreen:
    width_in_pixels = 8960
    height_in_pixels = 2880
    width_in_mms = 2370

    def __init__(self, depths, monitors):
        self.allowed_depths = depths
        self.root = FakeRoot(monitors)


class FakeDisplay:
    def __init__(self, screen, byte_order=0, max_request_length=4096):
        self._screen = screen
        self.flushes = 0
        self.display = type(
            "Info",
            (),
            {
                "info": type(
                    "I", (), {"image_byte_order": byte_order,
                              "max_request_length": max_request_length}
                )()
            },
        )()

    def screen(self):
        return self._screen

    def has_extension(self, name):
        return name in ("SHAPE", "RANDR")

    def get_atom_name(self, atom):
        return {435: "DP-1", 407: "HDMI-1"}[atom]

    def get_display_name(self):
        return ":99"

    def flush(self):
        self.flushes += 1

    def pending_events(self):
        return 0

    def next_event(self):
        raise AssertionError("no events were pending")

    def close(self):
        pass


def fake_monitor(name, primary, x, y, w, h, mm):
    return type(
        "Mon",
        (),
        {
            "name": name,
            "primary": primary,
            "x": x,
            "y": y,
            "width_in_pixels": w,
            "height_in_pixels": h,
            "width_in_millimeters": mm,
        },
    )()


@pytest.fixture
def x11(monkeypatch):
    """A fake X server, installed where `HudWindow.open` will import it."""
    from Xlib import display as xdisplay

    depths = [
        FakeDepth(24, [FakeVisual(64, 4)]),
        FakeDepth(32, [FakeVisual(112, 5), FakeVisual(113, 4)]),
    ]
    monitors = [
        fake_monitor(435, 1, 3840, 0, 5120, 2880, 600),
        fake_monitor(407, 0, 0, 428, 3840, 2160, 480),
    ]
    display = FakeDisplay(FakeScreen(depths, monitors))
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(xdisplay, "Display", lambda name=None: display)
    return display


class TestHudWindow:
    def test_it_asks_for_a_translucent_override_redirect_window(self, x11):
        window = HudWindow(HudConfig())
        window.open()
        created = x11.screen().root.window.kwargs
        assert created["depth"] == 32
        # TrueColor, not the DirectColor visual listed first at that depth.
        assert created["visual"] == 113
        assert created["override_redirect"] is True
        # A non-default visual needs both of these, or CreateWindow is a BadMatch.
        assert created["colormap"] == 113
        assert created["border_pixel"] == 0

    def test_clicks_pass_through_to_the_game(self, x11):
        window = HudWindow(HudConfig(click_through=True))
        window.open()
        from Xlib.ext import shape

        operation, kind, _, _, _, rectangles = x11.screen().root.window.shapes[0]
        assert (operation, kind) == (shape.SO.Set, shape.SK.Input)
        assert rectangles == []

    def test_the_hud_can_be_made_solid_for_debugging(self, x11):
        window = HudWindow(HudConfig(click_through=False))
        window.open()
        assert x11.screen().root.window.shapes == []

    def test_a_server_without_an_argb_visual_says_so(self, x11):
        x11.screen().allowed_depths = [FakeDepth(24, [FakeVisual(64, 4)])]
        with pytest.raises(RuntimeError, match="ARGB"):
            HudWindow(HudConfig()).open()

    def test_every_output_is_found_with_its_place_and_scale(self, x11):
        window = HudWindow(HudConfig())
        window.open()
        found = window.monitors()
        assert [(m.name, m.x, m.width) for m in found] == [
            ("DP-1", 3840, 5120),
            ("HDMI-1", 0, 3840),
        ]
        # Xft.dpi 192 is the compositor saying "2x", and it wins over guessing.
        assert [m.scale for m in found] == [2.0, 2.0]

    def test_the_configured_monitor_is_the_one_used(self, x11):
        window = HudWindow(HudConfig(monitor="HDMI-1"))
        window.open()
        assert window.monitor().name == "HDMI-1"

    def test_physical_size_gives_a_scale_when_nobody_says_otherwise(self, x11):
        x11.screen().root.get_full_property = lambda *a: None
        window = HudWindow(HudConfig())
        window.open()
        # 5120 px across 600 mm is 217 dpi (2x); 3840 across 480 is 203 (2x).
        assert [m.scale for m in window.monitors()] == [2.0, 2.0]

    def test_an_ordinary_display_is_left_at_1x(self, x11):
        x11.screen().root.get_full_property = lambda *a: None
        x11.screen().root._monitors = [fake_monitor(435, 1, 0, 0, 1920, 1080, 510)]
        window = HudWindow(HudConfig())
        window.open()
        assert window.monitors()[0].scale == 1.0

    def test_an_explicit_scale_overrides_discovery(self, x11):
        window = HudWindow(HudConfig(scale=1.5))
        window.open()
        assert window.monitors()[0].scale == 1.5

    def test_a_server_without_randr_still_gets_one_monitor(self, x11):
        x11.has_extension = lambda name: name == "SHAPE"
        window = HudWindow(HudConfig())
        window.open()
        (monitor,) = window.monitors()
        assert (monitor.width, monitor.height) == (8960, 2880)
        assert monitor.primary

    def test_a_broken_randr_reply_does_not_take_the_hud_with_it(self, x11):
        def boom():
            raise RuntimeError("RRGetMonitors failed")

        x11.screen().root.xrandr_get_monitors = boom
        window = HudWindow(HudConfig())
        window.open()
        assert len(window.monitors()) == 1

    def blit(self, x11, image, **cfg):
        window = HudWindow(HudConfig(**cfg))
        window.open()
        window.blit(image, 100, 50)
        return window, x11.screen().root

    def test_a_card_arrives_whole_in_requests_the_server_will_accept(self, x11):
        card = Image.new("RGBA", (300, 200), (20, 30, 40, 255))
        _, root = self.blit(x11, card)
        pixmap = root.pixmaps[0]
        budget = 4096 * 4 - 256
        assert len(pixmap.bands) > 1, "a 300x200 card should not fit one request here"
        assert all(len(data) <= budget for _, _, data in pixmap.bands)
        assert [y for y, _, _ in pixmap.bands] == sorted(y for y, _, _ in pixmap.bands)
        rebuilt = b"".join(data for _, _, data in pixmap.bands)
        assert rebuilt == pack_pixels(card, 0)

    def test_the_card_is_placed_and_raised(self, x11):
        _, root = self.blit(x11, Image.new("RGBA", (300, 200)))
        configure = root.window.configures[-1]
        from Xlib import X

        assert (configure["x"], configure["y"]) == (100, 50)
        assert (configure["width"], configure["height"]) == (300, 200)
        assert configure["stack_mode"] == X.Above

    def test_the_card_is_painted_as_the_window_background(self, x11):
        window, root = self.blit(x11, Image.new("RGBA", (300, 200)))
        assert root.window.attributes[-1]["background_pixmap"] == root.pixmaps[0].id
        assert root.window.maps == 1
        assert root.window.clears == 1

    def test_a_second_card_replaces_the_first_without_remapping(self, x11):
        window, root = self.blit(x11, Image.new("RGBA", (300, 200)))
        window.blit(Image.new("RGBA", (300, 260)), 100, 50)
        assert root.window.maps == 1
        assert root.pixmaps[0].freed, "the old backing pixmap should not leak"
        assert not root.pixmaps[1].freed


# -- the hud ---------------------------------------------------------------


class RecordingWindow:
    """A `HudWindow` that records instead of drawing."""

    def __init__(self, monitor=PRIMARY):
        self._monitor = monitor
        self.blits: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.hides = 0
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def monitor(self):
        return self._monitor

    def blit(self, image, x, y):
        self.blits.append((image.size, (x, y)))

    def hide(self):
        self.hides += 1

    def drain_events(self):
        pass

    def close(self):
        self.closed = True


class TestTextHud:
    def hud(self, window=None, **cfg):
        cfg.setdefault("enabled", True)
        return TextHud(HudConfig(**cfg), clock=lambda: 0.0, window=window or RecordingWindow())

    def test_a_remark_is_drawn_where_the_anchor_says(self):
        window = RecordingWindow()
        hud = self.hud(window, anchor="top-right", margin_px=32)
        hud.stack.add("Oh no.", 0.0)
        hud.paint()
        (size, (x, y)) = window.blits[0]
        assert y == 64
        assert x == PRIMARY.x + PRIMARY.width - size[0] - 64

    def test_the_same_card_is_not_drawn_twice(self):
        window = RecordingWindow()
        hud = self.hud(window)
        hud.stack.add("Oh no.", 0.0)
        hud.paint()
        hud.paint()
        assert len(window.blits) == 1

    def test_a_new_remark_grows_the_card(self):
        window = RecordingWindow()
        hud = self.hud(window)
        hud.stack.add("Oh no.", 0.0)
        hud.paint()
        hud.stack.add("It happened again.", 0.0)
        hud.paint()
        assert window.blits[1][0][1] > window.blits[0][0][1]

    def test_an_empty_stack_takes_the_window_away(self):
        window = RecordingWindow()
        hud = self.hud(window, hold_min_s=1.0, hold_max_s=1.0)
        hud.stack.add("Oh no.", 0.0)
        hud.paint()
        hud.clock = lambda: 2.0
        hud.paint()
        assert window.hides == 1

    def test_show_reaches_the_screen_through_the_thread(self):
        window = RecordingWindow()
        hud = TextHud(HudConfig(enabled=True, hold_min_s=5.0), window=window)
        hud.start()
        try:
            hud.show("Nice, that shortcut saved most of a lap.")
            deadline = time.monotonic() + 2.0
            while not window.blits and time.monotonic() < deadline:
                time.sleep(0.01)
            assert window.blits
            assert hud.shown == ["Nice, that shortcut saved most of a lap."]
        finally:
            hud.close()
        assert window.closed

    def test_clearing_puts_the_screen_back(self):
        window = RecordingWindow()
        hud = TextHud(HudConfig(enabled=True, hold_min_s=5.0), window=window)
        hud.start()
        try:
            hud.show("Oh no.")
            hud.clear()
            deadline = time.monotonic() + 2.0
            while not window.hides and time.monotonic() < deadline:
                time.sleep(0.01)
            assert window.hides >= 1
        finally:
            hud.close()

    def test_blank_remarks_are_not_worth_a_card(self):
        hud = self.hud()
        hud.show("   ")
        assert hud.shown == []


class TestMakeHud:
    def test_a_disabled_hud_is_a_null_one(self):
        hud = make_hud(HudConfig(enabled=False))
        assert isinstance(hud, NullHud)
        hud.show("Oh no.")
        assert hud.shown == ["Oh no."]

    def test_an_enabled_hud_opens_a_window(self):
        window = RecordingWindow()
        hud = make_hud(HudConfig(enabled=True), window=window)
        try:
            assert isinstance(hud, TextHud)
            assert window.opened
        finally:
            hud.close()

    def test_a_hud_that_cannot_open_degrades_instead_of_raising(self):
        class Broken(RecordingWindow):
            def open(self):
                raise RuntimeError("no 32-bit ARGB visual")

        hud = make_hud(HudConfig(enabled=True), window=Broken())
        assert isinstance(hud, NullHud)


# -- a real display --------------------------------------------------------


@pytest.mark.hardware
class TestRealDisplay:
    """Everything above fakes the server. This one asks it."""

    def test_a_card_reaches_a_real_x_server_intact(self):
        if not os.environ.get("DISPLAY"):
            pytest.skip("no X display")
        from Xlib import X

        cfg = HudConfig(enabled=True)
        window = HudWindow(cfg)
        window.open()
        try:
            monitor = window.monitor()
            metrics = Metrics.for_monitor(cfg, monitor)
            font = load_font(cfg, metrics.font_px)
            items = entries("gpagent hud test", "reading this means it works")
            card = render_card(items, [1.0, 1.0], cfg, metrics, font)
            x, y = place(card.size, monitor, metrics, cfg)
            window.blit(card, x, y)

            attrs = window._window.get_attributes()
            assert attrs.map_state == X.IsViewable
            assert attrs.override_redirect == 1
            geometry = window._window.get_geometry()
            assert (geometry.width, geometry.height) == card.size
            # What the server holds is what we packed, byte for byte.
            order = window._display.display.info.image_byte_order
            row = window._window.get_image(0, 0, card.width, 1, X.ZPixmap, 0xFFFFFFFF)
            assert bytes(row.data) == pack_pixels(card.crop((0, 0, card.width, 1)), order)
        finally:
            window.close()
