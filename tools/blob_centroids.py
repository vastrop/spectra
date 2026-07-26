#!/usr/bin/env python3
"""
blob_centroids.py
=================

Diagnostic: does an emission-line blob behave like a zero-order blob under
the pipeline's centroid corrections?

DAOStarFinder centroids the *zero order*, whose core saturates on bright
targets, so the returned (x, y) is biased: `best_y_shift` moves the aperture
onto the dispersed trace, and `measure_zero_order_x` re-measures the column
against the blob's own half-max profile.  A nova's Halpha peak is detected by
the same DAO pass as if it were a contaminating star.  If the DAO-vs-measured
signature of the two blobs differs, that difference is a candidate
discriminant for line-vs-star (which otherwise needs Gaia -- see
contaminators_from_sources).

Draws, at 1:1 and at an integer zoom:

    red   -- what DAOStarFinder returned
    cyan  -- the corrected position (measured peak column, trace row)

for the target's zero order and for the on-trace line blob.

Usage
-----
    python tools/blob_centroids.py data_V2014AQL/nova.fits --source 2
    python tools/blob_centroids.py --selfcheck

Both axes are measured with the same routine: `measure_zero_order_x` on the
image gives the sub-pixel column, and the same call on the transposed image
gives the sub-pixel row.  Identical treatment is the whole point -- any
difference between the two blobs is then in the data, not in the method.

Angle, dispersion and the non-linear node solution (with its zero-order
anchor) are read from the explorer's spectrum_config.json, so the columns
this scores and the wavelength it prints are the ones the GUI would show.
"""

import argparse
import json
import os
import sys

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from matplotlib.figure import Figure
from photutils.detection import DAOStarFinder
from scipy.ndimage import rotate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectrum_core import (                                    # noqa: E402
    _dao_xy, _zero_order_profile, best_y_shift, compute_spectrum_width,
    contaminators_from_sources, estimate_source_fwhm, extract_spectrum,
    fit_dispersion_poly, measure_zero_order_x, pixels_to_wavelengths,
    read_fits_image, spectrum_fully_in_frame, to_mono,
    validate_dispersion_poly)

# The explorer's own config, so this reproduces what the GUI shows rather
# than a re-tuned approximation: angle, initial linear dispersion, and the
# non-linear node solution with its zero-order anchor.  Extraction geometry
# still comes from the DEFAULTS below -- the explorer does not persist it.
CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "spectrum_config.json")
DPI = 100


def load_config(path):
    """Config dict, or {} if it is missing/unreadable — every consumer
    below already has a sane fallback, so a missing config degrades to the
    linear dispersion instead of failing."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def measure_xy(rot, x, y, fwhm):
    """Sub-pixel (column, row) of the blob near (x, y), both axes measured
    by measure_zero_order_x -- the second call on the transposed image.

    The row call gets an explicit, tight aperture_half.  Its default
    (1.5 x FWHM, i.e. ~17 columns here) is sized for the zero order, where
    every column in the window belongs to the blob; on the transposed axis
    that same window sums 17 columns of the dispersed continuum, so the row
    profile is the *trace's* and the answer comes back on the trace no
    matter where the blob actually is.  0.6 x FWHM keeps the window inside
    the blob core.  The continuum under the line still contributes -- but
    DAOStarFinder sees that too, so the comparison stays fair.
    """
    mx = measure_zero_order_x(rot, x, y, fwhm)
    cols_half = max(2, int(round(0.6 * fwhm))) if np.isfinite(fwhm) else 2
    my = measure_zero_order_x(rot.T, y, x, fwhm, aperture_half=cols_half)
    return mx, my


def halfmax_widths(rot, x, y, fwhm):
    """(width along columns, width along rows) of the blob's above-half-max
    run, in px -- the same profiles measure_xy centroids, so the shape
    number and the position number come from one view of the blob."""
    cols_half = max(2, int(round(0.6 * fwhm))) if np.isfinite(fwhm) else 2
    return (_run_width(_zero_order_profile(rot, x, y, fwhm)),
            _run_width(_zero_order_profile(rot.T, y, x, fwhm,
                                           aperture_half=cols_half)))


def _run_width(info):
    if info is None:
        return float("nan")
    prof, i_peak = info["prof"], info["i_peak"]
    thresh = 0.5 * info["peak"]
    lo = hi = i_peak
    while lo > 0 and prof[lo - 1] >= thresh:
        lo -= 1
    while hi < prof.size - 1 and prof[hi + 1] >= thresh:
        hi += 1
    return float(hi - lo + 1)


def _panel(fig, x0_px, y0_px, w_px, h_px, W, H):
    """Axes placed by pixel geometry, so imshow lands 1 px per data pixel."""
    return fig.add_axes([x0_px / W, y0_px / H, w_px / W, h_px / H])


def _draw(ax, img, x_lo, y_lo, blobs, zoom=1):
    ax.imshow(img, cmap="gray", origin="lower", interpolation="nearest",
              norm=simple_norm(img, "asinh", percent=99.7),
              extent=(x_lo - 0.5, x_lo + img.shape[1] - 0.5,
                      y_lo - 0.5, y_lo + img.shape[0] - 0.5))
    for (dx, dy), (mx, my), label in blobs:
        r = 3.0 if zoom == 1 else 2.5
        ax.add_artist(_circle(dx, dy, r, "#ff4040"))
        ax.add_artist(_circle(mx, my, r, "#40e0ff"))
        ax.plot([dx, mx], [dy, my], "-", color="#ffd040", lw=0.8)
        ax.annotate(label, (dx, dy), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    color="#ffd040", fontsize=7, annotation_clip=False)
    ax.set_xlim(x_lo - 0.5, x_lo + img.shape[1] - 0.5)
    ax.set_ylim(y_lo - 0.5, y_lo + img.shape[0] - 0.5)


def _circle(x, y, r, color):
    from matplotlib.patches import Circle
    return Circle((x, y), r, fill=False, ec=color, lw=1.0)


def run(args):
    cfg = load_config(args.config)
    angle = args.angle if args.angle is not None else float(cfg.get("angle", 0.0))
    disp = (args.dispersion if args.dispersion is not None
            else float(cfg.get("dispersion", 7.7)))
    nodes = cfg.get("dispersion_nodes") or []

    raw, _ = read_fits_image(args.fits)
    data = to_mono(raw)
    _, med, std = sigma_clipped_stats(data[::4, ::4], sigma=3.0)
    rot = rotate(data, angle, reshape=False, cval=med)
    h, w = rot.shape

    dao = DAOStarFinder(fwhm=args.fwhm, threshold=5.0 * std)
    sources = dao(rot - med)
    if sources is None or not len(sources):
        sys.exit("No sources detected.")
    sources.sort("peak", reverse=True)
    all_xy = np.array([_dao_xy(s) for s in sources], dtype=float)

    width = compute_spectrum_width(disp, args.sp_max)
    valid = [s for s in sources
             if spectrum_fully_in_frame(*_dao_xy(s), width, angle, w, h)]
    if len(valid) < args.source:
        sys.exit(f"Only {len(valid)} valid sources; --source {args.source} "
                 "is out of range.")
    sx, sy = _dao_xy(valid[args.source - 1])
    fwhm = estimate_source_fwhm(rot, sx, sy)

    *_, bbox = extract_spectrum(sx, sy, rot, width, args.aper,
                                args.sky_gap, args.sky_width)
    n_cols = bbox["x_end"] - bbox["x_start"]

    # Measured before the dispersion solution, because the zero-order anchor
    # needs this source's anchor residual: Δ = current − calibration, the
    # same shift the explorer's get_dispersion_poly applies.
    zo_meas = measure_xy(rot, sx, sy, fwhm)
    calib_resid = cfg.get("calib_anchor_resid")
    delta = 0.0
    if (cfg.get("zero_anchor") and calib_resid is not None
            and np.isfinite(zo_meas[0]) and np.isfinite(calib_resid)):
        delta = (zo_meas[0] - bbox["x_start"]) - float(calib_resid)

    # The config's node solution, not the initial linear Å/px: a nova's Hα
    # sits ~850 px out, where the linear guess is tens of Å off and the
    # blob-picking window is only as good as the wavelength axis.
    poly, n_bad = validate_dispersion_poly(fit_dispersion_poly(nodes, delta),
                                           n_cols)
    wls = pixels_to_wavelengths(np.arange(n_cols), disp, poly_coeffs=poly)

    # The y correction the explorer actually applies (same lambda window as
    # the calibrated panel, so the zero order stays out of the score).
    cols = (wls >= args.sp_min) & (wls <= args.sp_max)
    shift, _scores = best_y_shift(sx, sy, rot, width, args.aper,
                                  args.sky_gap, args.sky_width, cols=cols)

    # The line blob: an on-trace source DAO found inside the strip and
    # contaminators_from_sources dropped as the target's own light.  Pick the
    # one nearest the requested wavelength; column 0 is the zero order itself.
    keep, dropped = contaminators_from_sources(all_xy, bbox, fwhm, sy)
    dropped = np.atleast_2d(dropped) if len(dropped) else np.empty((0, 2))
    line_col_guess = float(np.argmin(np.abs(wls - args.line_wl)))
    cands = [d for d in dropped if d[0] > 1.0]
    if not cands:
        sys.exit("No on-trace blob apart from the zero order.")
    lc, ldy = min(cands, key=lambda d: abs(d[0] - line_col_guess))

    band = np.flatnonzero(np.abs(all_xy[:, 1] - sy) <= args.aper + fwhm)

    def dao_at(col):
        """Index into the detection table of the in-band source nearest
        strip column ``col``.  Both selectors above return strip-relative
        columns only; going back to the table keeps the printed DAO x/y
        literally what DAOStarFinder returned, not a recomposition."""
        return band[np.argmin(np.abs(all_xy[band, 0]
                                     - (bbox["x_start"] + col)))]

    lx, ly = all_xy[dao_at(lc)]

    # Control: the brightest source the same pass KEPT as a contaminating
    # star.  Without it the two rows below are just two numbers; with it
    # they are a test — a discriminant has to separate the line from a real
    # star, not merely from the zero order.  Brightest, not nearest: the
    # half-max measurement needs a 5-sigma peak to return anything at all.
    peaks = np.asarray(sources["peak"], dtype=float)
    ctrl = max((dao_at(c) for c in keep), key=lambda i: peaks[i],
               default=None) if len(keep) else None

    rows = [("zero order", (sx, sy)), ("line blob", (lx, ly))]
    if ctrl is not None:
        rows.append(("star (ctrl)", tuple(all_xy[ctrl])))

    print(f"source #{args.source}  DAO ({sx:.3f}, {sy:.3f})  "
          f"FWHM {fwhm:.2f} px   strip {n_cols} cols from x={bbox['x_start']}")
    print(f"angle {angle:.4f} deg   linear {disp} A/px   "
          + (f"{len(nodes)}-node poly deg {len(poly) - 1}, anchor "
             f"delta {delta:+.3f} px" if poly is not None
             else f"LINEAR fallback ({len(nodes)} nodes, "
                  f"{n_bad} non-monotonic steps)"))
    print(f"best_y_shift (trace re-centring) = {shift:+d} px")
    print()
    print(f"{'blob':<13}{'DAO x':>10}{'meas x':>10}{'dx':>8}"
          f"{'DAO y':>10}{'meas y':>10}{'dy':>8}"
          f"{'w_col':>8}{'w_row':>8}{'w_col/w_row':>13}")
    meas = {"zero order": zo_meas}
    for name, (dx, dy) in rows:
        mx, my = meas.get(name) or measure_xy(rot, dx, dy, fwhm)
        meas[name] = (mx, my)
        wc, wr = halfmax_widths(rot, dx, dy, fwhm)
        print(f"{name:<13}{dx:>10.3f}{mx:>10.3f}{mx - dx:>8.3f}"
              f"{dy:>10.3f}{my:>10.3f}{my - dy:>8.3f}"
              f"{wc:>8.1f}{wr:>8.1f}{wc / wr:>13.2f}")
    # dy from contaminators_from_sources is in ROWS, not FWHM.
    lam = float(np.interp(lc, np.arange(n_cols), wls))
    print(f"\nline blob sits {ldy:+.3f} px = {ldy / fwhm:+.3f} FWHM off the "
          f"trace (col {lc:.1f} = {lam:.1f} A, asked for {args.line_wl:.1f});  "
          f"{len(keep)} source(s) kept as contaminators")

    band_half = args.aper + args.sky_gap + 4
    y_lo = max(0, int(sy) - band_half)
    y_hi = min(h, int(sy) + band_half + 1)
    strip = rot[y_lo:y_hi, bbox["x_start"]:bbox["x_end"]]

    # The zero order's corrected row is the one the pipeline really applies
    # (best_y_shift onto the trace); the other blobs have no such correction,
    # so they get their own measured row.
    blobs = []
    for name, (dx, dy) in rows:
        mx, my = meas[name]
        blobs.append(((dx, dy),
                      (mx, sy + shift) if name == "zero order" else (mx, my),
                      name))

    # ── 1:1 strip ────────────────────────────────────────────────────────
    padl, padr, padb, padt = 60, 20, 40, 46  # padt clears the blob labels
    W = strip.shape[1] + padl + padr
    H = strip.shape[0] + padb + padt
    fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#101018")
    ax = _panel(fig, padl, padb, strip.shape[1], strip.shape[0], W, H)
    _draw(ax, strip, bbox["x_start"], y_lo, blobs)
    ax.set_title(f"{os.path.basename(args.fits)}  source #{args.source}  "
                 "1:1  (red = DAO, cyan = corrected)",
                 color="#e0e0e0", fontsize=8, pad=16)
    ax.tick_params(colors="#909090", labelsize=7)
    fig.savefig(args.out_strip)
    print(f"\nwrote {args.out_strip}  ({W}x{H} px, strip 1:1)")

    # ── integer-zoom cutouts ─────────────────────────────────────────────
    z = args.zoom
    cw = args.cut
    cuts = []
    for (dx, dy), corr, label in blobs:
        cx, cy = int(round(dx)), int(round(dy))
        xa, xb = max(0, cx - cw), min(w, cx + cw + 1)
        ya, yb = max(0, cy - cw), min(h, cy + cw + 1)
        cuts.append((rot[ya:yb, xa:xb], xa, ya, [((dx, dy), corr, label)]))

    cell_w = max(c[0].shape[1] for c in cuts) * z
    cell_h = max(c[0].shape[0] for c in cuts) * z
    gap = 50
    W = padl + len(cuts) * cell_w + (len(cuts) - 1) * gap + padr
    H = cell_h + padb + padt
    fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#101018")
    for i, (img, xa, ya, bl) in enumerate(cuts):
        ax = _panel(fig, padl + i * (cell_w + gap), padb,
                    img.shape[1] * z, img.shape[0] * z, W, H)
        _draw(ax, img, xa, ya, bl, zoom=z)
        ax.set_title(f"{bl[0][2]}  x{z}", color="#e0e0e0", fontsize=8)
        ax.tick_params(colors="#909090", labelsize=7)
    fig.savefig(args.out_zoom)
    print(f"wrote {args.out_zoom}  ({W}x{H} px, {z}x nearest-neighbour)")


def _blob(tx, ty, sigma=2.0, clip=None):
    yy, xx = np.mgrid[0:60, 0:60]
    img = 100.0 + 900.0 * np.exp(-(((xx - tx) ** 2 + (yy - ty) ** 2)
                                   / (2 * sigma ** 2)))
    return img if clip is None else np.clip(img, None, clip)


def _selfcheck():
    """measure_xy recovers a known sub-pixel blob position on both axes.

    Tolerance is 0.25 px, not 0.05: measure_zero_order_x centroids the
    *discrete* above-half-max run, so its answer carries a sawtooth bias of
    up to ~0.22 px against a Gaussian truth.  That is the production
    algorithm's own behaviour (it is chosen for stability on flat-topped
    cores, not for sub-0.1 px fidelity) -- the check guards the wiring and
    the transposed-axis reuse, so it must not assert accuracy the routine
    does not claim.  Tracking is asserted separately, and that is the
    property this tool actually leans on.
    """
    for tx, ty in ((30.0, 30.0), (30.37, 29.62), (28.8, 31.25)):
        mx, my = measure_xy(_blob(tx, ty), 30.0, 30.0, 4.7)
        assert abs(mx - tx) < 0.25, (mx, tx)
        assert abs(my - ty) < 0.25, (my, ty)

    # Tracking: a 2 px displacement must show up as a 2 px displacement.
    lo = measure_xy(_blob(29.0, 31.0), 30.0, 30.0, 4.7)
    hi = measure_xy(_blob(31.0, 29.0), 30.0, 30.0, 4.7)
    assert abs((hi[0] - lo[0]) - 2.0) < 0.25, (lo[0], hi[0])
    assert abs((hi[1] - lo[1]) + 2.0) < 0.25, (lo[1], hi[1])

    # A flat-topped (saturated) core still measures near the true centre --
    # the case the whole correction exists for.
    mx, my = measure_xy(_blob(28.8, 31.25, clip=400.0), 30.0, 30.0, 4.7)
    assert abs(mx - 28.8) < 0.3 and abs(my - 31.25) < 0.3, (mx, my)
    print("blob_centroids self-check OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fits", nargs="?", help="rotated-frame source FITS")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--source", type=int, default=2,
                    help="1-based index in the peak-sorted valid list (2 = "
                         "the nova in data_V2014AQL/nova.fits)")
    ap.add_argument("--config", default=CONFIG,
                    help="explorer config supplying angle, dispersion and the "
                         "non-linear node solution (default: repo root)")
    ap.add_argument("--angle", type=float, default=None,
                    help="override the config's rotation angle")
    ap.add_argument("--fwhm", type=float, default=5.0, help="DAO detection FWHM")
    ap.add_argument("--dispersion", type=float, default=None,
                    help="override the config's initial linear A/px (the "
                         "node solution still wins where it applies)")
    ap.add_argument("--sp-min", type=float, default=4000)
    ap.add_argument("--sp-max", type=float, default=8000)
    ap.add_argument("--aper", type=int, default=13)
    ap.add_argument("--sky-gap", type=int, default=3)
    ap.add_argument("--sky-width", type=int, default=20)
    ap.add_argument("--line-wl", type=float, default=6563.0,
                    help="wavelength of the on-trace blob to annotate")
    ap.add_argument("--zoom", type=int, default=8)
    ap.add_argument("--cut", type=int, default=18,
                    help="half-size of the zoom cutouts, px")
    ap.add_argument("--out-strip", default="blob_centroids_strip.png")
    ap.add_argument("--out-zoom", default="blob_centroids_zoom.png")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if not args.fits:
        ap.error("a FITS path is required (or --selfcheck)")
    run(args)


if __name__ == "__main__":
    main()
