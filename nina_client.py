"""
nina_client.py
==============

Minimal stdlib HTTP client for NINA's Advanced API plugin (ninaAPI >= 2.1,
default port 1888, base http://host:port/v2/api).

Shared by ``tools/spectral_autofocus.py`` (closed-loop spectral focus) and
the explorer's NINA dialog (capture-to-workfolder / sequence mirroring).
Kept free of any science imports so the GUI can import it cheaply.

Protocol behaviour this client relies on (Advanced API 2.2.x):

* captures need ``save=true`` to enter NINA's image history — an unsaved
  snapshot completes but is never fetchable;
* ``/image/{index}?raw_fits=true`` returns the FITS base64-encoded inside
  the JSON envelope; a raw stream is the announced future, and both forms
  are handled;
* the move endpoint acks before motion starts and ``/info`` can report a
  stale position with ``IsMoving`` false, so move completion requires
  position == target (with a was-moving escape hatch for clamped moves);
* the mount-slew query takes ``ra`` / ``dec`` in decimal degrees and offers
  ``waitForResult``.  This client sends degrees and polls ``mount/info``
  itself.  The documentation does not state the coordinate epoch; the
  endpoint takes **J2000**, so catalog coordinates go as-is and NINA or the
  driver handles the precession.  ``Slewing`` is assumed to suffer the same
  ack-before-motion lag as the focuser, so slew completion also requires
  arrival near the target (or having seen the mount slew);
* ``guider/start`` is the opposite of ack-early: it does not respond until
  PHD2 has settled, which outruns any short socket timeout.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.parse
import urllib.request

DEFAULT_PORT = 1888
POLL_S = 0.5                 # focuser completion poll interval
MOVE_TIMEOUT_S = 300         # generous: long approach moves on slow focusers
SLEW_TIMEOUT_S = 180         # normal go-to plus mount settling
CAPTURE_MARGIN_S = 120       # allowed on top of the exposure for download etc.
GUIDE_START_TIMEOUT_S = 120  # the start endpoint answers only once settled
GUIDE_CONFIRM_S = 60         # then trust the guider state, not the dead socket


class NinaError(RuntimeError):
    pass


def is_guiding(info):
    """True when a guider is connected and actively guiding.

    The one state in which a stopped guider must be resumed, and the one
    that confirms a resume actually took.
    """
    return (bool(info.get("Connected"))
            and str(info.get("State", "")).lower() == "guiding")


class NinaClient:
    """Minimal urllib client for the Advanced API v2 (http://host:port/v2/api)."""

    def __init__(self, host, port=DEFAULT_PORT, timeout=15):
        self.base = f"http://{host}:{port}/v2/api"
        self.timeout = timeout

    def _get(self, path, params=None, timeout=None, raw=False):
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: (str(v).lower() if isinstance(v, bool) else v)
                 for k, v in params.items()})
        with urllib.request.urlopen(url, timeout=timeout or self.timeout) as r:
            body = r.read()
        if raw:
            return body
        payload = json.loads(body)
        if not payload.get("Success", False):
            raise NinaError(f"{path}: {payload.get('Error') or payload}")
        return payload["Response"]

    # ── info ──────────────────────────────────────────────────────────
    def version(self, timeout=None):
        return self._get("/version", timeout=timeout)

    def focuser_info(self):
        return self._get("/equipment/focuser/info")

    def focuser_stable_info(self, timeout=MOVE_TIMEOUT_S):
        """Wait for two matching idle readings, then return the latest one."""
        deadline = time.monotonic() + timeout
        previous = None
        while True:
            info = self.focuser_info()
            moving = (info.get("IsMoving", False)
                      or info.get("IsSettling", False))
            position = int(info["Position"])
            if not moving and position == previous:
                return info
            previous = None if moving else position
            if time.monotonic() > deadline:
                raise NinaError("focuser did not become stable within "
                                f"{timeout}s (at {position})")
            time.sleep(POLL_S)

    def camera_info(self):
        return self._get("/equipment/camera/info")

    def mount_info(self):
        return self._get("/equipment/mount/info")

    def guider_info(self):
        return self._get("/equipment/guider/info")

    def guider_stop(self):
        """Stop autoguiding.  Best-effort: True on ack, False when there is no
        guider to stop, so callers (a slew) can proceed regardless."""
        try:
            self._get("/equipment/guider/stop")
            return True
        except NinaError:
            return False

    def guider_start(self, timeout=GUIDE_START_TIMEOUT_S,
                     confirm_s=GUIDE_CONFIRM_S):
        """Start (resume) autoguiding; blocks until it is actually guiding.

        Not fire-and-forget, whatever the endpoint name suggests: NINA
        does not answer until PHD2 has selected a star and settled, which
        routinely outlasts the client's default 15 s socket timeout — a
        resume that has in fact succeeded then reads as a failure purely
        because the client stopped listening.  Hence a generous read
        timeout, and when even that expires, believing the guider's own
        state rather than the dead socket.  A genuine API error (no guider
        connected, PHD2 down) still raises immediately.
        """
        try:
            self._get("/equipment/guider/start", timeout=timeout)
            return True
        except OSError:                 # socket/read timeout, not an API error
            deadline = time.monotonic() + confirm_s
            while True:
                try:
                    if is_guiding(self.guider_info()):
                        return True
                except (OSError, NinaError):
                    pass                # transient while NINA is busy settling
                if time.monotonic() > deadline:
                    raise NinaError(
                        f"guider start did not confirm within {timeout}s + "
                        f"{confirm_s}s of state polling") from None
                time.sleep(POLL_S)

    def mount_slew(self, ra_deg, dec_deg, timeout=SLEW_TIMEOUT_S):
        """Slew to decimal-degree coordinates and block until it stops.

        Like the focuser move, the slew endpoint can ack before motion
        begins, so a first poll may see ``Slewing`` false on a mount
        that has not started yet.  "Not slewing" alone is therefore not
        completion: the reported coordinates must also be near the
        target, or the mount must have been seen slewing and have since
        stopped.  The 1° tolerance is deliberately loose: it only
        guards against "never started", not pointing accuracy (epoch
        conversion and the pointing model can shift the reported
        position by tenths of a degree).
        """
        ra_deg, dec_deg = float(ra_deg), float(dec_deg)
        # Stop autoguiding before moving: you cannot guide through a slew, and
        # a guider left running fights the mount or re-locks onto the wrong
        # star on arrival.  Best-effort — no guider, or an unreachable one,
        # must not block the slew itself.
        try:
            self.guider_stop()
        except Exception:
            pass
        self._get("/equipment/mount/slew", {"ra": ra_deg, "dec": dec_deg})
        deadline = time.monotonic() + timeout
        seen_slewing = False
        while True:
            info = self.mount_info()
            slewing = info.get("Slewing", False)
            seen_slewing = seen_slewing or slewing
            if not slewing:
                coords = info.get("Coordinates") or {}
                try:
                    d_ra = abs((float(coords["RADegrees"]) - ra_deg
                                + 180.0) % 360.0 - 180.0)
                    d_dec = abs(float(coords["Dec"]) - dec_deg)
                    at_target = d_ra <= 1.0 and d_dec <= 1.0
                except (KeyError, TypeError, ValueError):
                    at_target = False
                if at_target or seen_slewing:
                    return info
            if time.monotonic() > deadline:
                raise NinaError(
                    f"mount did not finish slewing to {ra_deg}, {dec_deg} "
                    f"within {timeout}s")
            time.sleep(POLL_S)

    # ── focuser ───────────────────────────────────────────────────────
    def focuser_move(self, position, timeout=MOVE_TIMEOUT_S):
        """Absolute move; blocks until the focuser is AT the position.

        The move endpoint acks before motion begins, and /info can
        report a stale position with IsMoving still false, which lets the
        next capture start mid-move.  So "not moving" alone is not
        completion: the reported position must equal the target, or — as
        an escape hatch for drivers that clamp at a travel limit — the
        focuser must have been seen moving and have since stopped.  A
        clamped move that completes
        entirely between two polls times out loudly instead.
        """
        position = int(position)
        self._get("/equipment/focuser/move", {"position": position})
        deadline = time.monotonic() + timeout
        seen_moving = False
        while True:
            info = self.focuser_info()
            moving = (info.get("IsMoving", False)
                      or info.get("IsSettling", False))
            seen_moving = seen_moving or moving
            at = int(info["Position"])
            if not moving and (at == position or seen_moving):
                return at
            if time.monotonic() > deadline:
                raise NinaError(
                    f"focuser did not finish moving to {position} within "
                    f"{timeout}s (at {at})")
            time.sleep(POLL_S)

    # ── images ────────────────────────────────────────────────────────
    def image_count(self, image_type="SNAPSHOT"):
        """Number of images of one type in NINA's image history."""
        params = {"count": True}
        if image_type:
            params["imageType"] = image_type
        return int(self._get("/image-history", params))

    def image_history(self, image_type=None):
        """All image-history entries (dicts, incl. Filename), oldest first."""
        params = {"all": True}
        if image_type:
            params["imageType"] = image_type
        return self._get("/image-history", params)

    def fetch_fits(self, index, image_type="SNAPSHOT", timeout=None):
        """Image history entry ``index`` (within ``image_type``) as FITS bytes."""
        params = {"raw_fits": True}
        if image_type:
            params["imageType"] = image_type
        body = self._get(f"/image/{index}", params,
                         timeout=timeout or CAPTURE_MARGIN_S, raw=True)
        if body[:6] == b"SIMPLE":
            # Future ninaAPI: the base64 channel is announced as going
            # away; accept a direct FITS stream when it arrives.
            return body
        # 2.2.x reality: raw_fits comes base64-encoded in the JSON envelope.
        payload = json.loads(body)
        if not payload.get("Success", False):
            raise NinaError(f"image fetch: {payload.get('Error') or payload}")
        return base64.b64decode(payload["Response"])

    def _snapshot_count(self):
        return self.image_count("SNAPSHOT")

    def capture_fits(self, duration, gain=None):
        """One snapshot exposure; returns the FITS bytes.

        save=true is required: only SAVED images enter NINA's image
        history, and /image/{index} serves from that history; an unsaved
        snapshot capture completes but never becomes fetchable.  The
        capture response's
        base64 preview is skipped (omitImage); the frame is fetched from
        the history afterwards.  History entries appear only once image
        preparation finishes, slightly after the capture call returns,
        hence the count poll.
        """
        n_before = self._snapshot_count()
        params = {"duration": float(duration), "waitForResult": True,
                  "omitImage": True, "save": True}
        if gain is not None:
            params["gain"] = int(gain)
        self._get("/equipment/camera/capture", params,
                  timeout=duration + CAPTURE_MARGIN_S)
        deadline = time.monotonic() + 30
        while (n_after := self._snapshot_count()) <= n_before:
            if time.monotonic() > deadline:
                raise NinaError("capture finished but no new snapshot "
                                "appeared in the image history")
            time.sleep(POLL_S)
        return self.fetch_fits(n_after - 1, "SNAPSHOT",
                               timeout=duration + CAPTURE_MARGIN_S)
