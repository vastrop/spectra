#!/usr/bin/env python3
"""
corner_dispersion.py
====================

Measures the auto-calibrated dispersion of the SAME star imaged at several
sensor positions, to answer one question: does the grating's dispersion
(Å/px) depend on where in the field the spectrum falls?

Each frame goes through the explorer's own pipeline — derotate, detect the
brightest source, extract, then let ``suggest_dispersion_nodes`` anchor the
Balmer pattern — and the Å/px it fits is reported against the star's
position on the sensor.  Everything but the dispersion prior comes from the
folder's ``spectrum_config.json``; the prior is only a search seed.

Usage
-----
    py -3.13 tools/corner_dispersion.py cornercheck
    py -3.13 tools/corner_dispersion.py cornercheck --baseline center
    py -3.13 tools/corner_dispersion.py cornercheck --prior-sweep

Reading the output
------------------
``suggest_dispersion_nodes`` is a predict-then-fit scheme that scans the
scale as well as the zero-order position, over ±25% of the prior, so the
prior is a search seed and not the answer.  ``--prior-sweep`` is the check:
if neighbouring priors return the same Å/px the number is a measurement, if
it tracks the prior it is not.  The result should stay flat across a wide
sweep, not only within a narrow basin.

Two failure modes exist, in opposite directions.  A prior far BELOW the
truth puts the truth outside the search band, and the answer can come back
confidently wrong; only the sweep exposes it.  A prior far ABOVE the truth
is this tool's doing rather than the scan's: the strip is sized by
``compute_spectrum_width(prior)``, so an inflated prior shortens the strip
in wavelength terms until the telluric node, and eventually Hα, falls off
the end.  The Balmer lock stays exact — the fitted Å/px drifts because
nodes are missing, not because the lines were misidentified.

The reported precision floor is the honest limit.  Line centroids good to a
few tenths of a pixel over the Hδ→telluric span put it near 0.01 Å/px, so
a spread below that means "no field dependence detected", NOT "zero".

One asymmetry is structural, not an oversight: the spectrum disperses to
+x, so a star in a RIGHT-side corner throws its spectrum off the sensor and
cannot be measured at all.  Only left-side field positions are reachable in
a given camera orientation, and the sampling is therefore one-sided by
construction.  Rotating the camera 180° (dispersing leftward) is what would
reach the other half of the field, at the cost of a second config.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import rotate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectrum_core import (                                    # noqa: E402
    _dao_xy, compute_spectrum_width, extract_spectrum, read_fits_image,
    spectrum_fully_in_frame, suggest_dispersion_nodes, to_mono,
)

# Not persisted in spectrum_config.json, which holds calibration state
# only, so the explorer's own defaults stand in.
FWHM, APER_HALF, SKY_GAP, SKY_WIDTH = 5.0, 13, 3, 20
SP_MAX = 8000.0


def brightest_source(image, median, std, width, angle, fwhm=FWHM):
    """The target star: brightest source whose spectrum still fits in frame.

    Same DAOStarFinder call and same in-frame geometry rejection the
    explorer uses, so a frame that works there works here.
    """
    h, w = image.shape
    found = DAOStarFinder(fwhm=fwhm, threshold=5.0 * std)(image - median)
    if found is None or not len(found):
        return None
    found.sort("peak", reverse=True)
    for src in found:
        if spectrum_fully_in_frame(*_dao_xy(src), width, angle, w, h):
            return _dao_xy(src), float(src["peak"]), len(found)
    return None


def measure_frame(path, angle, prior, y_offset=0.0, fwhm=FWHM):
    """(sensor_xy, dispersion, info, nodes) for one frame, or (None, …).

    The star's sensor position is measured on the UNROTATED frame — "where
    in the field" is the question, and rotation moves it — while the
    extraction runs on the rotated one, as in the explorer.
    """
    raw, _hdr = read_fits_image(path)
    data = to_mono(raw)
    _, median, std = sigma_clipped_stats(data[::4, ::4], sigma=3.0)

    on_sensor = brightest_source(data, median, std, 10, 0.0, fwhm)
    rotated = rotate(data, angle, reshape=False, cval=median)
    width = compute_spectrum_width(prior, sp_range_max=SP_MAX)
    found = brightest_source(rotated, median, std, width, angle, fwhm)
    if found is None:
        return None, None, {"error": "no source with a full spectrum"}, []
    (sx, sy), _peak, _n = found

    col_sums = extract_spectrum(sx, sy + y_offset, rotated, width,
                               APER_HALF, SKY_GAP, SKY_WIDTH)[5]
    nodes, info = suggest_dispersion_nodes(col_sums, prior)
    if not nodes:
        return None, None, info, []
    xy = on_sensor[0] if on_sensor else (float("nan"), float("nan"))
    return xy, info["dispersion"], info, nodes


def main():
    ap = argparse.ArgumentParser(
        description="Dispersion of one star at several sensor positions.")
    ap.add_argument("folder", help="folder of .fit frames + spectrum_config.json")
    ap.add_argument("--baseline", default=None,
                    help="frame stem to compare against (default: the first)")
    ap.add_argument("--prior", type=float, default=None,
                    help="Å/px search seed (default: the config's dispersion)")
    ap.add_argument("--prior-sweep", action="store_true",
                    help="re-solve each frame across a range of priors; the "
                         "answer is only a measurement where it stops "
                         "tracking the seed")
    args = ap.parse_args()

    cfg_path = os.path.join(args.folder, "spectrum_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    angle = float(cfg["angle"])
    prior = args.prior if args.prior is not None else float(cfg["dispersion"])
    y_off = float(cfg.get("y_offset", 0.0))

    frames = sorted(glob.glob(os.path.join(args.folder, "*.fit"))
                    + glob.glob(os.path.join(args.folder, "*.fits")))
    if not frames:
        sys.exit(f"No .fit/.fits frames in {args.folder}")

    print(f"config: angle={angle}°  prior={prior} Å/px  y_offset={y_off}")
    print(f"{len(frames)} frame(s)\n")

    if args.prior_sweep:
        seeds = [prior * f for f in (0.94, 0.97, 1.0, 1.03, 1.06)]
        for path in frames:
            name = os.path.splitext(os.path.basename(path))[0]
            out = []
            for seed in seeds:
                _xy, disp, _info, _nd = measure_frame(path, angle, seed, y_off)
                out.append(f"{seed:.2f}→"
                           + ("fail" if disp is None else f"{disp:.4f}"))
            print(f"{name:10s} " + "  ".join(out))
        print("\nA flat middle = the answer is prior-independent there.  A "
              "value that tracks its seed is the seed, not a measurement.")
        return

    rows = []
    for path in frames:
        name = os.path.splitext(os.path.basename(path))[0]
        xy, disp, info, nodes = measure_frame(path, angle, prior, y_off)
        if disp is None:
            print(f"{name:10s} FAILED: {info.get('error', 'unknown')}\n")
            continue
        resid = max(abs(r) for r in info["residuals"])
        rows.append((name, xy[0], xy[1], disp))
        print(f"{name:10s} sensor x={xy[0]:7.1f} y={xy[1]:7.1f}")
        print(f"           dispersion = {disp:.4f} Å/px   x0={info['x0']:.0f}"
              f"   Balmer={info['n_balmer']}"
              f"   telluric={'yes' if info['telluric_added'] else 'no'}"
              f"   max|resid|={resid:.1f} Å")
        print("           nodes: "
              + ", ".join(f"{p:.1f}px→{w:.0f}Å" for p, w in nodes) + "\n")

    if len(rows) < 2:
        return

    base_name = args.baseline or rows[0][0]
    base = next((d for n, _x, _y, d in rows if n == base_name), rows[0][3])
    disps = [d for *_r, d in rows]
    print(f"dispersion vs field position (baseline: {base_name})")
    print(f"{'frame':10s}{'x':>8s}{'y':>8s}{'Å/px':>10s}{'Δ':>10s}{'Δ %':>8s}")
    for name, x, y, d in rows:
        print(f"{name:10s}{x:8.0f}{y:8.0f}{d:10.4f}{d - base:+10.4f}"
              f"{100 * (d - base) / base:+8.2f}")
    spread = max(disps) - min(disps)
    print(f"\nspread: {spread:.4f} Å/px ({100 * spread / base:.2f}% of "
          f"baseline)")
    # The floor, not a formality: a spread under it means the method cannot
    # see a difference, which is not the same as there being none.
    print("precision floor is ~0.01 Å/px (sub-pixel centroids over the "
          "Hδ→telluric span);\nread a spread below that as 'no field "
          "dependence detected', not as zero.")


if __name__ == "__main__":
    main()
