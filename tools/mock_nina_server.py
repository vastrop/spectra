"""
mock_nina_server.py
===================

Standalone mock of NINA's Advanced API v2 — the subset ``nina_client.py``
speaks: version, focuser info/move, mount info/slew, camera capture and
image history.  Extracted from the spectral-autofocus self-check (which
still imports it) so it can also run as a fake rig for exercising the
explorer's NINA dialog without hardware:
    py -3.13 tools/mock_nina_server.py [--port 1888]

Behaviour mirrors the protocol quirks documented in ``nina_client.py``:
capture demands ``save=true`` +
``omitImage`` (asserted — a violating request fails loudly), raw FITS is
served base64-encoded in the JSON envelope, and moves/slews complete
~0.2 s after the ack.  The returned state dict carries the fault-injection
switches the self-check flips: ``stale`` (focuser position lags with
IsMoving false — the live ZWO EAF race), ``mount_lag`` (Slewing stays
false for a beat after the slew ack) and ``mount_stuck`` (Slewing never
clears).  Frames are 4x4 zero FITS — enough for client plumbing, not for
scoring anything real.
"""

from __future__ import annotations

import base64
import http.server
import io
import json
import threading
import time
import urllib.parse


def default_state():
    # RA 15h59m30s, Dec +25°55′12.613″ — a plausible Corona Borealis
    # pointing, so the dialog's nearest-A-star list can be sanity-checked
    # against a real chart instead of a made-up spot on the equator.
    return {"pos": 5000, "moving_until": 0.0, "snapshots": [], "moves": [],
            "mount_pos": ((15 + 59 / 60 + 30 / 3600) * 15,
                          25 + 55 / 60 + 12.613 / 3600),
            "mount_target": None, "slews": [], "guider_stops": 0,
            "guider_state": "Stopped", "guider_start_delay": 0.0,
            "mount_slewing_from": 0.0, "mount_moving_until": 0.0,
            "mount_lag": False, "mount_stuck": False,
            "stale": False, "stale_target": None, "stale_ready": 0.0}


def _fits_bytes():
    # Imported lazily: the module itself must stay importable/startable
    # without waiting on the science stack.
    import numpy as np
    from astropy.io import fits as afits
    buf = io.BytesIO()
    afits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(buf)
    return buf.getvalue()


def make_handler(state, quiet=True):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            if not quiet:
                super().log_message(*a)

        def _json(self, response, success=True):
            body = json.dumps({"Response": response, "Error": "",
                               "StatusCode": 200, "Success": success,
                               "Type": "API"}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:
                # The client timed out and hung up (the guider-start check
                # does this on purpose) — not a mock failure, don't dump a
                # socketserver traceback over the self-check output.
                pass

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(url.query))
            path = url.path
            if path == "/v2/api/version":
                self._json("2.2.15-mock")
            elif path == "/v2/api/equipment/focuser/info":
                if (state["stale"] and state["stale_target"] is not None
                        and time.monotonic() >= state["stale_ready"]):
                    state["pos"] = state["stale_target"]
                    state["stale_target"] = None
                moving = time.monotonic() < state["moving_until"]
                self._json({"Position": state["pos"], "IsMoving": moving,
                            "IsSettling": False, "Connected": True,
                            "StepSize": 1, "Temperature": 10.0,
                            "Name": "MockFocuser"})
            elif path == "/v2/api/equipment/focuser/move":
                target = int(q["position"])
                state["moves"].append(target)
                if state["stale"]:
                    state["stale_target"] = target
                    state["stale_ready"] = time.monotonic() + 0.4
                else:
                    state["pos"] = target
                    state["moving_until"] = time.monotonic() + 0.2
                self._json("Move started")
            elif path == "/v2/api/equipment/mount/info":
                now = time.monotonic()
                if (state["mount_target"] is not None
                        and now >= state["mount_moving_until"]):
                    state["mount_pos"] = state["mount_target"]
                    state["mount_target"] = None
                slewing = (state["mount_stuck"]
                           or state["mount_slewing_from"] <= now
                           < state["mount_moving_until"])
                self._json({
                    "Connected": True, "AtPark": False, "Slewing": slewing,
                    "Coordinates": {"RADegrees": state["mount_pos"][0],
                                    "Dec": state["mount_pos"][1],
                                    "Epoch": "J2000"},
                })
            elif path == "/v2/api/equipment/guider/info":
                self._json({"Connected": True, "Name": "MockGuider",
                            "State": state["guider_state"]})
            elif path == "/v2/api/equipment/guider/start":
                # Live behaviour: the call does not answer until PHD2 has
                # picked a star and settled — the delay that outran the
                # client's socket timeout.  Threaded server, so guider/info
                # keeps answering meanwhile (that is the whole point).
                time.sleep(state["guider_start_delay"])
                state["guider_state"] = "Guiding"
                self._json("Guiding started")
            elif path == "/v2/api/equipment/guider/stop":
                state["guider_stops"] += 1
                state["guider_state"] = "Stopped"
                self._json("Guiding stopped")
            elif path == "/v2/api/equipment/mount/slew":
                state["slews"].append(q)
                now = time.monotonic()
                lag = 0.2 if state["mount_lag"] else 0.0
                state["mount_slewing_from"] = now + lag
                state["mount_moving_until"] = now + lag + 0.2
                state["mount_target"] = (float(q["ra"]), float(q["dec"]))
                self._json("Started Slew")
            elif path == "/v2/api/equipment/camera/info":
                self._json({"Connected": True, "Name": "MockCamera",
                            "Gain": 100, "IsExposing": False})
            elif path == "/v2/api/equipment/camera/capture":
                # save=true is what makes the image reach the history.
                assert q.get("omitImage") == "true", q
                assert q.get("save") == "true", q
                state["snapshots"].append(_fits_bytes())
                self._json("Capture started")
            elif path == "/v2/api/image-history":
                self._json(len(state["snapshots"]))
            elif path.startswith("/v2/api/image/"):
                assert q.get("raw_fits") == "true", q
                idx = int(path.rsplit("/", 1)[1])
                # 2.2.x serves raw_fits as base64 inside the JSON envelope.
                self._json(base64.b64encode(
                    state["snapshots"][idx]).decode("ascii"))
            else:
                self._json(f"unknown path {path}", success=False)

    return Handler


def make_server(port=0, quiet=True):
    """Threaded mock server on 127.0.0.1; returns (server, state).

    The server thread is a running daemon; call ``server.shutdown()``
    when done.  Mutate the state dict to inject faults (see module
    docstring); ``server.server_address[1]`` is the bound port.
    """
    state = default_state()
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(state, quiet=quiet))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Fake NINA Advanced API rig for testing without hardware")
    ap.add_argument("--port", type=int, default=1888,
                    help="port to listen on (default 1888, NINA's default)")
    args = ap.parse_args()
    srv, _state = make_server(args.port, quiet=False)
    print(f"Mock NINA Advanced API on http://127.0.0.1:{args.port}/v2/api "
          "— Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
