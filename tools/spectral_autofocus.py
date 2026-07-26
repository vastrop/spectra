"""
tools/spectral_autofocus.py
===========================

Closed-loop SPECTRAL autofocus through NINA's Advanced API plugin.

Why this exists: with a diffraction grating in the train, best spectral
focus is NOT best star focus — NINA's native autofocus optimises the
zero-order PSF and lands at a chromatically offset position.  This tool
sweeps the focuser and scores the dispersed spectrum itself on Balmer
absorption-line sharpness (focus_analyzer's validated metrics), then
moves to the optimum.

Point the telescope at a bright A-type (Balmer-strong) star first; this
tool does not slew.  It talks to the Advanced API plugin (ninaAPI >= 2.1,
default port 1888) over plain HTTP, so it runs fine on the desktop with
NINA on a remote mini PC:

    focuser/move + focuser/info      absolute moves, completion polling
    camera/capture (SNAPSHOT)        one exposure per sweep point
    image/{i}?raw_fits=true          the RAW FITS, straight into the
                                     focus_analyzer reduction

Each frame is saved as af_<position>.fits in a timestamped run folder,
so a run is also a normal focus_analyzer folder — re-analyze offline
anytime with:  py -3.13 focus_analyzer.py --folder focus_runs/<run>

Backlash: the sweep always ascends, and the initial and final moves
approach from below (overshoot then up), so every measured position and
the final set position are reached from the same direction.

If best focus lands on a sweep edge the run is inconclusive, so the tool
automatically captures EXTEND_POINTS more points past that edge and
re-scores, up to MAX_EXTENSIONS times (--no-extend opts out).

The whole run is one callable — ``run_autofocus(args, client, emit,
should_stop)``.  The CLI below is a thin wrapper around it, and the
explorer's NINA panel imports the same function and runs it on a worker
thread: same code, no subprocess, and it therefore works in the frozen
build too.  Progress reaches the caller through ``emit`` (log lines, then
one ``afdata`` dict carrying the sweep model — positions, scores,
best/ideal, parabola coefficients, strip path — for the panel's plot).
``should_stop`` is polled between sweep points, so Cancel takes effect
after the current exposure lands rather than orphaning a half-written
frame.  The best frame's raw 2-D aperture cutout is saved as
best_strip.npy in the run folder as visual focus confirmation.

Usage
-----
    py -3.13 tools/spectral_autofocus.py --host 192.168.1.50 --probe
    py -3.13 tools/spectral_autofocus.py --host 192.168.1.50 \
        --exposure 3 --step 25 --points 9 --dry-run
    py -3.13 tools/spectral_autofocus.py --host 192.168.1.50 --exposure 3

Run with -h for the full option list.  --dry-run sweeps, scores and
reports, then restores the starting position — the trust-building mode.

Self-check (no NINA needed):  py -3.13 tools/spectral_autofocus.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Root modules live one level up from tools/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# focus_analyzer's report uses characters (e.g. ── headers) that a cp1252
# Windows console can't encode; degrade them to '?' instead of crashing.
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import focus_analyzer as fa

# The API client lives in its own root module so the explorer's NINA
# dialog can import it without dragging in the science stack.
from nina_client import NinaClient, NinaError, DEFAULT_PORT, is_guiding


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def sweep_positions(center, step, points):
    """Ascending sweep positions centred on ``center``."""
    start = center - step * (points // 2)
    return [start + step * i for i in range(points)]


# When the best focus lands on a sweep edge the true optimum is probably
# outside the sweep — grab a few more points that way instead of asking
# the user to re-run wider.
EXTEND_POINTS = 3
MAX_EXTENSIONS = 2


def extension_positions(measured, best_pos, step, n=EXTEND_POINTS):
    """New ascending sweep points past the edge the optimum is pinned to,
    or None when the optimum is interior (no extension needed).  Clipped
    to the 0..99999 af_XXXXX filename range."""
    lo, hi = min(measured), max(measured)
    if best_pos == lo:
        new = [lo - step * i for i in range(n, 0, -1)]
    elif best_pos == hi:
        new = [hi + step * i for i in range(1, n + 1)]
    else:
        return None
    new = [p for p in new if 0 <= p <= 99999]
    return new or None


class AutofocusError(RuntimeError):
    """A setup problem that stops the run before any exposure is taken.

    Deliberately NOT SystemExit: this runs on a worker thread inside the
    explorer, where a BaseException would die unnoticed.
    """


def _print_emit(kind, payload):
    """Default sink — the CLI.  Progress goes to stdout; the sweep model is
    of no use to a terminal, so it is dropped."""
    if kind == "log":
        print(payload, flush=True)


def run_sweep(client, positions, exposure, gain, settle, run_dir, overshoot,
              emit=_print_emit, should_stop=None, on_frame=None):
    """Move-capture-save across ``positions`` (ascending, from below).

    Returns False if ``should_stop()`` asked us to stop — checked between
    points, so a cancel takes effect after the current exposure lands
    rather than killing a half-written frame.

    ``on_frame(path)`` is called as soon as each frame is on disk.  The
    caller uses it to start reducing that frame while this loop is already
    moving the focuser and exposing the next one — must not block.
    """
    # Approach the first point from below so every position in the
    # ascending sweep is reached against the same backlash direction.
    approach = positions[0] - overshoot
    emit("log", f"Backlash approach: moving to {approach} first")
    client.focuser_move(approach)

    for pos in positions:
        if should_stop is not None and should_stop():
            return False
        at = client.focuser_move(pos)
        if at != pos:
            emit("log", f"  [warn] asked {pos}, focuser reports {at}")
        if settle > 0:
            time.sleep(settle)
        emit("log", f"  pos {pos}: exposing {exposure}s…")
        blob = client.capture_fits(exposure, gain)
        out = os.path.join(run_dir, f"af_{pos:05d}.fits")
        with open(out, "wb") as f:
            f.write(blob)
        emit("log", f"  pos {pos}: saved {os.path.basename(out)} "
                    f"({len(blob) / 1e6:.1f} MB)")
        if on_frame is not None:
            on_frame(out)
    return True


def sweep_summary(frames, best, ideal, metric, strip_path=None):
    """The sweep model as a plain dict — positions, scores, best/ideal, the
    parabola coefficients and the best frame's saved strip.  Handed to the
    explorer's NINA panel (via ``emit``) to draw the sweep-model plot."""
    key, direction = fa.METRIC_KEYS[metric]
    pts = sorted((fr.position, fr.metrics.get(key, float("nan")))
                 for fr in frames)
    positions = [p for p, _ in pts]
    scores = [float(s) if math.isfinite(s) else None for _, s in pts]
    parabola = None
    if ideal is not None:
        r = fa.estimate_continuous_focus(
            positions, [float("nan") if s is None else s for s in scores],
            direction)
        if r:
            parabola = [float(v) for v in r[1]]
    return {
        "metric": metric, "direction": direction,
        "positions": positions, "scores": scores,
        "best": best.position if best is not None else None,
        "ideal": None if ideal is None else float(ideal),
        "parabola": parabola,
        "strip_path": strip_path,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Spectral autofocus via NINA's Advanced API — sweeps "
                    "the focuser and optimises Balmer-line sharpness of "
                    "the dispersed spectrum (not the zero-order PSF).")
    ap.add_argument("--host", default="localhost",
                    help="NINA machine hostname/IP (default: localhost)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--probe", action="store_true",
                    help="Check API/camera/focuser connectivity and exit.")
    ap.add_argument("--selfcheck", action="store_true",
                    help="Run the built-in self-check (no NINA needed).")

    ap.add_argument("--config", default="spectrum_config.json",
                    help="Analysis JSON config (angle, dispersion nodes).")
    ap.add_argument("--exposure", type=float, default=3.0,
                    help="Exposure per sweep point, seconds (default 3).")
    ap.add_argument("--gain", type=int, default=None,
                    help="Camera gain (default: camera's current setting).")
    ap.add_argument("--center", type=int, default=None,
                    help="Sweep centre (default: current focuser position).")
    ap.add_argument("--step", type=int, default=25,
                    help="Focuser steps between sweep points (default 25).")
    ap.add_argument("--points", type=int, default=9,
                    help="Number of sweep points (default 9).")
    ap.add_argument("--overshoot", type=int, default=None,
                    help="Backlash approach distance below the first point "
                         "(default: 4 x step).")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="Extra settle seconds after each move (default 1).")
    ap.add_argument("--run-dir", default=None,
                    help="Folder for this run's frames/plot/CSV "
                         "(default: focus_runs/<timestamp>).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Sweep and report, but do NOT move to the optimum.")
    ap.add_argument("--no-extend", action="store_true",
                    help="Don't auto-extend the sweep when the optimum "
                         "lands on a sweep edge.")
    ap.add_argument("--no-anchor", action="store_true",
                    help="Search the whole frame for the source in EVERY frame "
                         "instead of reusing the first frame's position "
                         "(slower; for when the centroid drifts a lot).")

    # Reduction / scoring options, mirrored from focus_analyzer.
    ap.add_argument("--metric", default="fwhm",
                    choices=list(fa.METRIC_KEYS.keys()))
    ap.add_argument("--include-halpha", action="store_true")
    ap.add_argument("--lines", default="Hbeta,Hgamma,Hdelta")
    ap.add_argument("--fwhm", type=float, default=fa.DEFAULT_FWHM_DETECT)
    ap.add_argument("--aperture-half", type=int,
                    default=fa.DEFAULT_APERTURE_HALF)
    ap.add_argument("--fixed-aperture", type=int, default=None)
    ap.add_argument("--sky-gap", type=int, default=fa.DEFAULT_SKY_BAND_GAP)
    ap.add_argument("--sky-width", type=int, default=fa.DEFAULT_SKY_BAND_WIDTH)
    ap.add_argument("--calibrate", action="store_true")
    return ap.parse_args(argv)


def probe(client):
    v = client.version()
    print(f"Advanced API reachable: version {v}")
    for name, get in (("camera", client.camera_info),
                      ("focuser", client.focuser_info)):
        try:
            info = get()
            connected = info.get("Connected", False)
            extra = (f", position {info['Position']}" if name == "focuser"
                     and connected else "")
            print(f"  {name}: {'connected' if connected else 'NOT connected'}"
                  f" ({info.get('DisplayName') or info.get('Name', '?')}"
                  f"{extra})")
        except Exception as e:
            print(f"  {name}: query failed — {e}")


class _EmitStream:
    """File-like shim that funnels writes into ``emit``."""

    def write(self, text):
        if text.strip():
            self._emit("log", text.rstrip("\n"))
        return len(text)

    def flush(self):
        pass

    def __init__(self, emit):
        self._emit = emit


def _reduce_and_score(path, cfg, args, lines, cache, anchor_holder):
    """Reduce + score one frame into ``cache`` — runs on the reducer thread.

    The first frame is reduced by a full-frame search and leaves its source
    position in ``anchor_holder``; later frames reuse it and derotate only the
    strip (see focus_analyzer._reduce_anchored).  One worker thread, so the
    frames go through in order and the anchor is set before it is read.

    Swallows failures: the frame simply stays out of the cache, and
    analyze_folder reduces it on the main thread where its error is reported
    the way it always was.  Caching a None (reduce_frame's own "unusable
    frame" answer) is deliberate — that is a verdict, not a failure, and it
    must not be recomputed on every extension.
    """
    try:
        anchor = anchor_holder.get("anchor")
        fr = fa.reduce_frame(path, cfg, args, anchor=anchor)
        if fr is not None:
            fa.score_frame(fr, lines)
            if anchor is None and not getattr(args, "no_anchor", False):
                anchor_holder["anchor"] = fa.anchor_from(fr)
        cache[path] = fr
    except Exception:
        pass


def run_autofocus(args, client, emit=None, should_stop=None, cfg=None):
    """Sweep, score and move to best spectral focus.  The whole tool.

    ``args``       an argparse Namespace from parse_args() — the GUI builds
                   one the same way, so the option defaults live in one place.
    ``emit``       progress sink: emit("log", str) for a line of report,
                   emit("afdata", dict) with the sweep model — a partial one
                   (measured points only) as each frame is scored, then the
                   full model (parabola/best/strip) at the end.  Default
                   prints to stdout (the CLI), which drops afdata.
    ``should_stop``polled between sweep points; True stops the run after the
                   current exposure and restores the starting position.
    ``cfg``        a FocusConfig; default reads args.config from disk.  The
                   explorer passes its LIVE config instead of a temp file.

    Returns (rc, message): rc 0 done, 1 nothing measurable, 2 cancelled.
    Raises AutofocusError for setup problems, NinaError for API failures.
    """
    # One reducer thread, alive for the whole run.  Frames are reduced as they
    # land instead of in a blocking pass at the end, and the cache it fills
    # means an edge extension only analyses the frames it just captured.
    # Shut down here, in a finally, because the body has cancel paths that
    # return early.
    emit = emit or _print_emit
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="af-reduce")
    initial = None
    try:
        finfo = client.focuser_stable_info()
        initial = int(finfo["Position"])
        rc, msg = _run_autofocus(
            args, client, emit, should_stop, cfg, pool, finfo)
        if rc != 0 or args.dry_run:
            at = _return_to_start(client, initial, args, emit)
            msg += f"  Focuser returned to starting position {at}."
        return rc, msg
    except Exception as exc:
        if initial is None:
            raise AutofocusError(
                f"{exc}  WARNING: starting position could not be read; "
                "focuser position is unknown.") from exc
        try:
            at = _return_to_start(client, initial, args, emit)
        except Exception as restore_exc:
            raise AutofocusError(
                f"{exc}  WARNING: could not return the focuser to starting "
                f"position {initial}: {restore_exc}") from exc
        raise AutofocusError(
            f"{exc}  Focuser returned to starting position {at}.") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _return_to_start(client, initial, args, emit):
    overshoot = args.overshoot if args.overshoot is not None else 4 * args.step
    emit("log", f"Returning focuser to starting position {initial}…")
    client.focuser_move(max(0, initial - overshoot))
    return client.focuser_move(initial)


def _run_autofocus(args, client, emit, should_stop, cfg, pool, finfo):
    emit = emit or _print_emit
    if cfg is None:
        cfg = fa.load_config(args.config)
    lines = fa.resolve_lines(args)

    if not finfo.get("Connected", False):
        raise AutofocusError("Focuser is not connected in NINA.")
    center = args.center if args.center is not None else int(finfo["Position"])
    positions = sweep_positions(center, args.step, args.points)
    if positions[0] < 0:
        raise AutofocusError(f"Sweep would reach position {positions[0]} < 0 — "
                             f"raise --center or shrink --step/--points.")
    if positions[-1] > 99999:
        # af_XXXXX.fits carries the position as the filename's last 5
        # digits (focus_analyzer convention); widen both together if a
        # focuser with more than 99999 steps ever appears.
        raise AutofocusError("Positions above 99999 don't fit the af_XXXXX "
                             "filename convention.")
    overshoot = args.overshoot if args.overshoot is not None else 4 * args.step

    run_dir = args.run_dir or os.path.join(
        "focus_runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    emit("log", f"Sweep: {positions[0]}..{positions[-1]} step {args.step} "
                f"({args.points} points, centre {center})")
    emit("log", f"Run folder: {run_dir}")

    # focus_analyzer reports frame by frame with bare print()s.  In-process
    # that would go to a stdout the windowed build doesn't have, so borrow it
    # for the duration and feed the caller's log instead.
    # sys.stdout is process-global, so a concurrent thread printing during a
    # sweep lands in the panel log too.  That is harmless while nothing else
    # in the explorer prints; thread emit through focus_analyzer if that
    # ceases to hold.
    # path -> reduced+scored frame (or None for an unusable one).  Filled by
    # the reducer thread during the sweep and read by analyze_folder; the two
    # never touch it at the same time, because scored() drains the queue first.
    cache = {}
    pending = []
    anchor_holder = {}          # the first reduced frame's source position

    # Animate the wait: push the points measured so far to the plot as each
    # frame finishes reducing, not only at the end.  Partial models carry no
    # parabola/best (nothing to fit until the sweep is scored) — _plot_af_data
    # renders bare measured points fine.  afdata is dropped by the CLI sink,
    # so this path only matters to the GUI.  The done-callback runs on the single
    # reducer thread, synchronously after cache[path] is set, so reading cache
    # here never races the writer.
    metric_key, metric_dir = fa.METRIC_KEYS[args.metric]

    def _emit_partial(_fut):
        pts = sorted((fr.position, fr.metrics.get(metric_key, float("nan")))
                     for fr in cache.values() if fr is not None)
        if not pts:
            return
        emit("afdata", {
            "metric": args.metric, "direction": metric_dir,
            "positions": [p for p, _ in pts],
            "scores": [float(s) if math.isfinite(s) else None
                       for _, s in pts],
            "parabola": None, "best": None, "ideal": None, "strip_path": None})

    def on_frame(path):
        fut = pool.submit(_reduce_and_score, path, cfg, args,
                          lines, cache, anchor_holder)
        pending.append(fut)
        fut.add_done_callback(_emit_partial)

    def scored():
        for fut in pending:          # let the in-flight reductions finish
            fut.result()             # _reduce_and_score swallows; never raises
        pending.clear()
        if emit is _print_emit:
            return fa.analyze_folder(run_dir, cfg, args, lines, cache=cache)
        import contextlib
        with contextlib.redirect_stdout(_EmitStream(emit)):
            return fa.analyze_folder(run_dir, cfg, args, lines, cache=cache)

    if not run_sweep(client, positions, args.exposure, args.gain, args.settle,
                     run_dir, overshoot, emit, should_stop, on_frame):
        return 2, "Autofocus cancelled."

    emit("log", "\nSweep complete — scoring…\n")
    frames, best, ideal = scored()

    # Auto-extend: optimum pinned to a sweep edge means the true minimum
    # is likely just outside — capture a few more points that way and
    # re-score, up to MAX_EXTENSIONS times.
    extensions = 0
    inconclusive = False
    while not args.no_extend and best is not None:
        if extensions >= MAX_EXTENSIONS:
            emit("log", f"\n[warn] Optimum still on the sweep edge after "
                        f"{MAX_EXTENSIONS} extensions — re-run centred on "
                        f"{best.position}.")
            inconclusive = True
            break
        extra = extension_positions([fr.position for fr in frames],
                                    best.position, args.step)
        if extra is None:
            break
        extensions += 1
        emit("log", f"\nBest focus {best.position} sits on the sweep edge — "
                    f"extending {extra[0]}..{extra[-1]} "
                    f"({extensions}/{MAX_EXTENSIONS})…")
        if not run_sweep(client, extra, args.exposure, args.gain, args.settle,
                         run_dir, overshoot, emit, should_stop, on_frame):
            return 2, "Autofocus cancelled during the extension."
        # Only the new frames are analysed here — the cache holds the rest.
        emit("log", "\nExtension complete — scoring the new frames…\n")
        frames, best, ideal = scored()

    # The best frame's raw aperture strip, saved for the explorer's NINA
    # panel to display as visual confirmation of the achieved focus.
    strip_path = None
    if best is not None and best.strip is not None and best.strip.size:
        import numpy as np
        strip_path = os.path.abspath(os.path.join(run_dir, "best_strip.npy"))
        np.save(strip_path, best.strip)

    emit("afdata", sweep_summary(frames, best, ideal, args.metric, strip_path))

    csv_path = os.path.join(run_dir, "focus_analysis.csv")
    png_path = os.path.join(run_dir, "focus_analysis.png")
    fa.write_csv(frames, csv_path)
    emit("log", f"Wrote CSV:   {csv_path}")
    fa.make_plot(frames, cfg, lines, args.metric, best, png_path)
    emit("log", f"Wrote plot:  {png_path}")

    if best is None:
        msg = "No usable spectral measurement in any frame."
        emit("log", "\n" + msg)
        return 1, msg

    if inconclusive:
        msg = ("No reliable focus solution — optimum remained on the sweep "
               f"edge at {best.position} after {MAX_EXTENSIONS} extensions.")
        emit("log", "\n" + msg)
        return 1, msg

    target = int(round(ideal)) if ideal is not None else best.position
    if args.dry_run:
        msg = f"Dry run — focuser NOT moved.  Recommended position: {target}"
        emit("log", "\n" + msg)
        return 0, msg

    # Final move, approached from below like every sweep point.
    emit("log", f"\nMoving to best focus {target} (via {target - overshoot})…")
    client.focuser_move(target - overshoot)
    at = client.focuser_move(target)
    msg = f"Focuser at {at}.  Done."
    emit("log", msg)
    return 0, msg


def main(argv=None):
    args = parse_args(argv)
    if args.selfcheck:
        _selfcheck()
        return 0

    client = NinaClient(args.host, args.port)
    if args.probe:
        probe(client)
        return 0

    try:
        rc, _msg = run_autofocus(args, client)
    except AutofocusError as exc:
        raise SystemExit(str(exc))
    return rc


# ---------------------------------------------------------------------------
# Self-check — a fake Advanced API served over real HTTP on localhost
# ---------------------------------------------------------------------------

def _selfcheck():
    """Exercise the client + sweep against a mocked API (thread + http.server).

    Checks: URL/param encoding, move-completion polling, snapshot
    indexing, raw FITS round-trip, ascending-from-below move order, and
    the af_XXXXX filenames a run folder receives.

    The mock rig lives in tools/mock_nina_server.py (also runnable
    standalone against the NINA dialog); its state dict carries the
    fault-injection switches flipped below.
    """
    import tempfile

    from astropy.io import fits as afits

    from tools.mock_nina_server import make_server

    srv, state = make_server()
    try:
        client = NinaClient("127.0.0.1", srv.server_address[1])
        assert client.version() == "2.2.15-mock"
        assert client.focuser_info()["Connected"]
        state["stale"] = True
        state["stale_target"] = 5050
        state["stale_ready"] = time.monotonic() + 0.4
        t0 = time.monotonic()
        assert client.focuser_stable_info()["Position"] == 5050
        assert time.monotonic() - t0 >= 0.5, "accepted a stale sweep centre"
        state["stale"] = False
        state["moves"].clear()
        assert _return_to_start(client, 5050, parse_args([]),
                                lambda *_: None) == 5050
        assert state["moves"] == [4950, 5050], state["moves"]
        assert abs(client.mount_info()["Coordinates"]["RADegrees"]
                   - 239.875) < 1e-9

        t0 = time.monotonic()
        client.mount_slew(359.25, -12.5)
        assert time.monotonic() - t0 >= 0.2, "did not wait for slew"
        assert state["slews"][-1] == {"ra": "359.25", "dec": "-12.5"}
        # Every slew stops guiding first (you cannot guide through a slew).
        assert state["guider_stops"] >= 1, "slew did not stop guiding"

        # Ack-before-motion race: Slewing stays false briefly after the
        # slew command — completion must wait for seen-slewing or arrival,
        # not trust the first quiet poll.
        state["mount_lag"] = True
        t0 = time.monotonic()
        client.mount_slew(200.0, 30.0)
        assert time.monotonic() - t0 >= 0.4, "returned before the slew began"
        state["mount_lag"] = False

        state["mount_stuck"] = True
        try:
            client.mount_slew(10, 20, timeout=0.01)
            raise AssertionError("stuck mount slew did not time out")
        except NinaError:
            pass
        finally:
            state["mount_stuck"] = False

        # A slow-settling guider start: the HTTP call outruns its timeout,
        # but the guider does come up — the state, not the dead socket,
        # decides, or the resume reports a false failure.
        state["guider_start_delay"] = 1.0
        assert client.guider_start(timeout=0.2, confirm_s=5.0) is True
        assert is_guiding(client.guider_info())
        # …and one that never comes up still fails.
        client.guider_stop()
        state["guider_start_delay"] = 5.0
        try:
            client.guider_start(timeout=0.2, confirm_s=0.6)
            raise AssertionError("dead guider start reported success")
        except NinaError:
            pass
        state["guider_start_delay"] = 0.0

        # Move blocks until IsMoving clears, and reports the position.
        t0 = time.monotonic()
        assert client.focuser_move(5100) == 5100
        assert time.monotonic() - t0 >= 0.2, "did not wait for move"

        # Stale-position race (live ZWO EAF behaviour): the API acks the
        # move but Position lags with IsMoving never true — the wait must
        # hold until Position reaches the target, not return stale.
        state["stale"] = True
        t0 = time.monotonic()
        assert client.focuser_move(5200) == 5200
        assert time.monotonic() - t0 >= 0.4, "returned a stale position"
        state["stale"] = False

        blob = client.capture_fits(0.01)
        assert blob[:6] == b"SIMPLE"

        assert sweep_positions(5000, 25, 9) == [4900, 4925, 4950, 4975, 5000,
                                                5025, 5050, 5075, 5100]

        # Edge-extension decisions: interior best -> no extension; edge
        # best -> ascending points past that edge; clipped at 0.
        grid = [4950, 4975, 5000, 5025, 5050]
        assert extension_positions(grid, 5000, 25) is None
        assert extension_positions(grid, 4950, 25) == [4875, 4900, 4925]
        assert extension_positions(grid, 5050, 25) == [5075, 5100, 5125]
        assert extension_positions([0, 25, 50], 0, 25) is None  # nothing < 0
        assert extension_positions([30, 55, 80], 30, 25) == [5]

        # Sweep model: a plain dict of the model parameters, JSON-clean
        # (the panel plots it; nothing parses stdout any more).
        class _F:
            def __init__(self, pos, fwhm):
                self.position = pos
                self.metrics = {"fwhm_score": fwhm}
        fake = [_F(p, 10 + 0.01 * (p - 5008) ** 2)
                for p in range(4900, 5101, 25)]
        d = sweep_summary(fake, fake[4], 5008.0, "fwhm")
        assert d["best"] == 5000 and abs(d["ideal"] - 5008) < 1e-6
        assert len(d["parabola"]) == 3 and len(d["positions"]) == 9
        assert d["strip_path"] is None
        json.dumps(d)          # the panel receives it as-is; keep it plain
        assert sweep_summary(fake, fake[4], 5008.0, "fwhm",
                             r"C:\x\s.npy")["strip_path"] == r"C:\x\s.npy"

        with tempfile.TemporaryDirectory() as td:
            state["moves"].clear()
            positions = sweep_positions(5000, 25, 3)          # 4975..5025
            assert run_sweep(client, positions, exposure=0.01, gain=None,
                             settle=0, run_dir=td, overshoot=100) is True
            names = sorted(os.listdir(td))
            assert names == ["af_04975.fits", "af_05000.fits",
                             "af_05025.fits"], names
            # ascending, first point approached from below
            assert state["moves"] == [4875, 4975, 5000, 5025], state["moves"]
            with afits.open(os.path.join(td, names[0])) as hdul:
                assert hdul[0].data.shape == (4, 4)

        # Cancellation (what replaces killing the subprocess): should_stop is
        # honoured BETWEEN points, so the frame being exposed still lands and
        # no half-written FITS is left behind.
        with tempfile.TemporaryDirectory() as td:
            state["moves"].clear()
            logged = []
            stop_after = [2]           # stop once two frames are on disk

            def emit(kind, payload):
                logged.append((kind, payload))

            def should_stop():
                return len(os.listdir(td)) >= stop_after[0]

            done = run_sweep(client, sweep_positions(5000, 25, 5),
                             exposure=0.01, gain=None, settle=0, run_dir=td,
                             overshoot=100, emit=emit, should_stop=should_stop)
            assert done is False, "cancelled sweep reported completion"
            names = sorted(os.listdir(td))
            assert names == ["af_04950.fits", "af_04975.fits"], names
            assert all(k == "log" for k, _ in logged)
            # and every frame it did write is a complete, readable FITS
            for n in names:
                with afits.open(os.path.join(td, n)) as hdul:
                    assert hdul[0].data.shape == (4, 4)

        # Pipelining: each frame is handed to on_frame the moment it is on
        # disk (that is what lets the reducer thread work while the next
        # exposure runs), in sweep order, once each.
        with tempfile.TemporaryDirectory() as td:
            handed = []
            assert run_sweep(client, sweep_positions(5000, 25, 3),
                             exposure=0.01, gain=None, settle=0, run_dir=td,
                             overshoot=100, on_frame=handed.append) is True
            assert handed == [os.path.join(td, f"af_{p:05d}.fits")
                              for p in (4975, 5000, 5025)], handed
            assert all(os.path.isfile(p) for p in handed), "handed off early"

        # The frame cache: an edge extension must re-analyse only its NEW
        # frames.  A cache hit skips reduce_frame entirely (stub it to prove
        # it is never called), and a cached None — reduce_frame's own verdict
        # that a frame is unusable — stays dropped instead of being retried.
        with tempfile.TemporaryDirectory() as td:
            paths = [os.path.join(td, f"af_{p:05d}.fits")
                     for p in (4975, 5000, 5025)]
            for p in paths:
                open(p, "wb").close()      # never opened: every path is cached

            class _Cached:                 # a reduced+scored frame, minimally
                def __init__(self, pos, fwhm):
                    self.position, self.path = pos, ""
                    self.spatial_fwhm = 3.0
                    self.metrics = {"fwhm_score": fwhm, "depth_score": 1.0,
                                    "gradient_score": 1.0, "spatial_fwhm": 3.0,
                                    "per_line": {}}
                    # what anchor_from() reads off a cached frame
                    self.source_xy, self.shape, self.row_need = (10.0, 20.0), (0, 0), 25

            cache = {paths[0]: _Cached(4975, 6.0),
                     paths[1]: _Cached(5000, 4.0),
                     paths[2]: None}       # unusable frame, already judged

            def _never(*a, **k):
                raise AssertionError("cache hit still called reduce_frame")

            real_reduce, fa.reduce_frame = fa.reduce_frame, _never
            try:
                af_args = parse_args(["--metric", "fwhm"])
                import contextlib
                import io as _io
                with contextlib.redirect_stdout(_io.StringIO()):
                    frames, best, _ideal = fa.analyze_folder(
                        td, fa.FocusConfig(angle=0.0, dispersion=1.0,
                                           dispersion_nodes=[],
                                           calibration_df=None),
                        af_args, ["Hbeta"], cache=cache)
            finally:
                fa.reduce_frame = real_reduce
            assert [f.position for f in frames] == [4975, 5000], frames
            assert best.position == 5000, best.position
    finally:
        srv.shutdown()
    print("spectral_autofocus self-check OK")


if __name__ == "__main__":
    raise SystemExit(main())
