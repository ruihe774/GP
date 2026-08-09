"""xdg-desktop-portal ScreenCast handshake.

Wayland has no screen grab; capture goes through the portal, which hands back a
PipeWire remote fd plus a node id. Portal ScreenCast v4+ supports `persist_mode`
and returns a `restore_token`, so the consent dialog is a first-run cost only.

The handshake is request/response over D-Bus signals and needs a GLib main loop,
so it runs once in a dedicated thread with its own main context rather than
forcing a GLib loop on the whole process.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["ScreenCastSession", "open_screencast", "default_token_path", "PortalError"]

PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"
REQUEST_IFACE = "org.freedesktop.portal.Request"


class PortalError(RuntimeError):
    pass


@dataclass
class ScreenCastSession:
    fd: int
    node_id: int
    width: int | None
    height: int | None
    source_type: int | None
    restore_token: str | None
    session_handle: str
    #: The portal tears the session down -- and with it the PipeWire node --
    #: as soon as the client's D-Bus connection goes away. Python will collect
    #: the connection the moment the handshake returns unless it is held here,
    #: which manifests as pipewiresrc failing with "target not found".
    connection: Any = None

    def describe(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "size": [self.width, self.height] if self.width else None,
            "source_type": self.source_type,
            "restored": self.restore_token is not None,
        }

    def close(self) -> None:
        """Close the portal session and release the PipeWire fd."""
        if self.connection is not None:
            try:
                from gi.repository import Gio, GLib

                self.connection.call_sync(
                    PORTAL_BUS, self.session_handle,
                    "org.freedesktop.portal.Session", "Close",
                    GLib.Variant("()", ()), None,
                    Gio.DBusCallFlags.NONE, 2000, None,
                )
            except Exception:
                log.debug("portal session already closed", exc_info=True)
            self.connection = None
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1


def default_token_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "gpagent" / "screencast.json"


def _load_token(path: Path) -> str | None:
    try:
        with open(path) as fh:
            return json.load(fh).get("restore_token")
    except (OSError, ValueError):
        return None


def _save_token(path: Path, token: str | None) -> None:
    if not token:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"restore_token": token}, fh)
        path.chmod(0o600)
    except OSError:
        log.warning("could not persist restore token to %s", path)


def open_screencast(
    *,
    source_types: int = 3,
    cursor_mode: int = 2,
    persist_mode: int = 2,
    multiple: bool = False,
    token_path: Path | None = None,
    timeout_s: float = 120.0,
    use_saved_token: bool = True,
) -> ScreenCastSession:
    """Run the full portal handshake. Blocking; call from a worker thread.

    With `use_saved_token=False` the cached restore token is ignored so the
    portal shows its picker again; whatever is chosen is then saved as usual.
    """
    token_path = token_path or default_token_path()
    saved = _load_token(token_path) if use_saved_token else None
    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.update(
                _handshake(
                    source_types=source_types,
                    cursor_mode=cursor_mode,
                    persist_mode=persist_mode,
                    multiple=multiple,
                    restore_token=saved,
                )
            )
        except BaseException as exc:  # surfaced to the caller below
            error.append(exc)

    thread = threading.Thread(target=run, name="portal-handshake", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise PortalError(
            "portal handshake timed out (was the permission dialog dismissed?)"
        )
    if error:
        raise error[0]

    _save_token(token_path, result.get("restore_token"))
    return ScreenCastSession(**result)


def _handshake(
    *,
    source_types: int,
    cursor_mode: int,
    persist_mode: int,
    multiple: bool,
    restore_token: str | None,
) -> dict[str, Any]:
    import gi

    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib

    context = GLib.MainContext.new()
    context.push_thread_default()
    try:
        conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        unique = conn.get_unique_name()
        sender = unique[1:].replace(".", "_")

        def request(method: str, build_args) -> dict[str, Any]:
            """Invoke a portal method and wait for its Request.Response."""
            token = f"gpagent{secrets.token_hex(8)}"
            path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
            loop = GLib.MainLoop.new(context, False)
            box: dict[str, Any] = {}

            def on_response(_c, _s, _p, _i, _sig, params):
                code, results = params.unpack()
                box["code"] = code
                box["results"] = results
                loop.quit()

            sub = conn.signal_subscribe(
                PORTAL_BUS, REQUEST_IFACE, "Response", path, None,
                Gio.DBusSignalFlags.NONE, on_response,
            )
            try:
                conn.call_sync(
                    PORTAL_BUS, PORTAL_PATH, SCREENCAST_IFACE, method,
                    build_args(token), GLib.VariantType("(o)"),
                    Gio.DBusCallFlags.NONE, -1, None,
                )
                # GLib timeouts are the only way out if the portal never answers.
                GLib.timeout_add_seconds(115, lambda: (loop.quit(), False)[1])
                loop.run()
            finally:
                conn.signal_unsubscribe(sub)

            if "code" not in box:
                raise PortalError(f"{method}: no response from portal")
            if box["code"] == 1:
                raise PortalError(f"{method}: cancelled by user")
            if box["code"] != 0:
                raise PortalError(f"{method}: failed with code {box['code']}")
            return box["results"]

        results = request(
            "CreateSession",
            lambda token: GLib.Variant(
                "(a{sv})",
                (
                    {
                        "handle_token": GLib.Variant("s", token),
                        "session_handle_token": GLib.Variant(
                            "s", f"gpagent{secrets.token_hex(8)}"
                        ),
                    },
                ),
            ),
        )
        session_handle = results["session_handle"]

        select: dict[str, Any] = {
            "types": GLib.Variant("u", source_types),
            "multiple": GLib.Variant("b", multiple),
            "cursor_mode": GLib.Variant("u", cursor_mode),
            "persist_mode": GLib.Variant("u", persist_mode),
        }
        if restore_token:
            select["restore_token"] = GLib.Variant("s", restore_token)

        def select_args(token: str):
            options = dict(select)
            options["handle_token"] = GLib.Variant("s", token)
            return GLib.Variant("(oa{sv})", (session_handle, options))

        try:
            request("SelectSources", select_args)
        except PortalError:
            if not restore_token:
                raise
            # A stale token (compositor restart, GNOME update) must not be fatal.
            log.warning("stored restore token rejected; requesting fresh consent")
            select.pop("restore_token", None)
            request("SelectSources", select_args)

        started = request(
            "Start",
            lambda token: GLib.Variant(
                "(osa{sv})",
                (session_handle, "", {"handle_token": GLib.Variant("s", token)}),
            ),
        )

        streams = started.get("streams") or []
        if not streams:
            raise PortalError("portal returned no streams")
        node_id, props = streams[0]
        size = props.get("size")
        width, height = (int(size[0]), int(size[1])) if size else (None, None)

        reply, fd_list = conn.call_with_unix_fd_list_sync(
            PORTAL_BUS, PORTAL_PATH, SCREENCAST_IFACE, "OpenPipeWireRemote",
            GLib.Variant("(oa{sv})", (session_handle, {})),
            GLib.VariantType("(h)"), Gio.DBusCallFlags.NONE, -1, None, None,
        )
        fd = fd_list.get(reply.unpack()[0])

        return {
            "fd": fd,
            "node_id": int(node_id),
            "width": width,
            "height": height,
            "source_type": props.get("source_type"),
            "restore_token": started.get("restore_token"),
            "session_handle": session_handle,
            "connection": conn,
        }
    finally:
        context.pop_thread_default()
