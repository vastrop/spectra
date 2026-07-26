"""
focus_analyzer.py
=================
Stand-alone spectral autofocus analyser for the SA-100 / ToupTek pipeline.

Given a folder of FITS frames of a bright A-type star taken at a sweep of
focuser positions, this script:

  1. Loads every FITS in a target folder (default: ./Focus_test).
  2. Extracts the focuser position from the LAST 5 DIGITS of each filename.
  3. Detects the single brightest valid source in each frame.
  4. Extracts the 1-D spectrum using the SAME reduction as Spectrum Explorer
     (mono collapse → rotate by config angle → background-subtracted aperture
     sum), reusing spectrum_core.extract_spectrum.
  5. Builds a wavelength solution from the config's dispersion_nodes
     (cubic-or-lower polynomial, with a monotonicity check + linear fallback),
     exactly as the GUI's get_dispersion_poly / _validate_dispersion_poly do.
  6. Scores spectral sharpness on the Balmer absorption cores (Hβ, Hγ, Hδ by
     default — Hα is off by default because at 7.7 Å/px it sits near the
     telluric B-band and the strip edge). Three independent metrics are
     computed per line: notch DEPTH (contrast), wing GRADIENT steepness, and
     a Gaussian-fit core FWHM. The spatial PSF FWHM is also measured as an
     independent cross-check.
  7. Picks "best focus" by the line-FWHM metric (narrowest = sharpest) by
     default, and writes:
        - a PNG with a small-multiples grid (one panel per position, best
          focus highlighted) + a focus-position-vs-score curve, and
        - a CSV of position vs every metric.

All reduction parameters (angle, dispersion, dispersion_nodes, the embedded
calibration array) come from the JSON config so behaviour matches the main
Spectrum Explorer. The instrument-response calibration itself is NOT needed
for a focus metric — line sharpness is measured on the raw background-
subtracted spectrum — so the calibration array is read but only used if you
explicitly enable --calibrate.

Usage
-----
    python focus_analyzer.py                       # defaults
    python focus_analyzer.py --folder ./Focus_test \
                             --config spectrum_config.json
    python focus_analyzer.py --metric depth        # rank by depth instead
    python focus_analyzer.py --include-halpha      # add Hα to the score

Run with -h for the full option list.

This module depends only on spectrum_core (and wavelength, indirectly via the
Balmer constants defined locally). It does not import the tkinter GUI.
"""

from __future__ import annotations

import os
import re
import csv
import json
import glob
import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
# No matplotlib.use() and no pyplot at import: this module is imported INTO
# the explorer's GUI process (the NINA panel's autofocus), where forcing a
# backend would yank TkAgg out from under the live canvases.  make_plot
# renders through an explicit Agg canvas instead, and only --show touches
# pyplot.
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from astropy.modeling import models
from photutils.detection import DAOStarFinder
from scipy.ndimage import rotate

# specutils: line fitting on a Spectrum container.  In specutils 2.x the
# container class is ``Spectrum`` (the older name ``Spectrum1D`` is a
# deprecated alias); import the current name.
from specutils import Spectrum
from specutils.fitting import fit_lines

# Pipeline helpers — identical reduction to spectrum_explorer.py
from spectrum_core import (
    native_byteorder,
    to_mono,
    compute_spectrum_width,
    estimate_source_fwhm,
    spectrum_fully_in_frame,
    extract_spectrum,
    pixels_to_wavelengths,
    apply_calibration,
    rotate_band,
    _dao_xy,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Balmer rest wavelengths in Å.  Hα/Hβ/Hγ/Hδ.  The config's dispersion nodes
# are pinned at Hδ/Hβ/Hα + a red anchor, so the polynomial wavelength solution
# places these lines reliably regardless of the exact focuser position.
BALMER = {
    "Halpha": 6562.8,
    "Hbeta":  4861.3,
    "Hgamma": 4340.5,
    "Hdelta": 4101.7,
}

# Default detection / extraction params (mirror Spectrum Explorer DEFAULTS).
# These are NOT in the JSON config (the GUI keeps them in the left panel), so
# they are defaulted here and can be overridden from the CLI.
DEFAULT_FWHM_DETECT      = 7.0     # DAOStarFinder fwhm for source detection
DEFAULT_NSOURCES         = 20
DEFAULT_SKY_BAND_GAP     = 3
DEFAULT_SKY_BAND_WIDTH   = 20
DEFAULT_APERTURE_HALF    = 13      # fallback if FWHM auto-fit fails
DEFAULT_SP_RANGE_MIN     = 4000
DEFAULT_SP_RANGE_MAX     = 8000

# Auto-fit multipliers, identical to spectrum_explorer.py
APERTURE_FWHM_MULT = 2.5 / 3.0
SKY_GAP_FWHM_MULT  = 0.7

# Half-window (in Å) used to isolate a Balmer line for the sharpness metrics.
# At 7.7 Å/px this is ~±9 px, comfortably wider than a focused A-star core but
# narrow enough to keep a clean local continuum on each side.
LINE_HALF_WINDOW_A = 70.0

# Continuum is estimated from two side-bands just outside the core window.
CONTINUUM_BAND_A = 60.0   # width of each side band in Å


# ---------------------------------------------------------------------------
# Config loading — mirrors spectrum_explorer._load_config semantics
# ---------------------------------------------------------------------------

@dataclass
class FocusConfig:
    angle: float
    dispersion: float
    dispersion_nodes: list           # list of [pixel, wavelength]
    calibration_df: Optional[object] # pandas DataFrame or None
    sp_min: float = DEFAULT_SP_RANGE_MIN
    sp_max: float = DEFAULT_SP_RANGE_MAX


def load_config(path: str) -> FocusConfig:
    """Read the JSON analysis config from disk (the CLI path)."""
    with open(path, "r") as f:
        return config_from_dict(json.load(f))


def config_from_dict(cfg: dict) -> FocusConfig:
    """Pull the fields the focus analyser needs out of an already-parsed
    config dict.  Tolerant of string-typed numerics (the GUI saves
    angle/dispersion/y_offset as strings).

    Split from load_config so the explorer can hand over its LIVE config —
    a visually tweaked calibration counts — without a temp file.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config file does not contain a JSON object.")

    angle = float(cfg.get("angle", 0.0))
    dispersion = float(cfg.get("dispersion", 7.7))

    # Dispersion nodes: list of 2-element [pixel, wavelength] pairs.
    nodes = []
    for entry in cfg.get("dispersion_nodes", []) or []:
        try:
            nodes.append([float(entry[0]), float(entry[1])])
        except (TypeError, ValueError, IndexError):
            continue

    # Embedded calibration array (optional, only used with --calibrate).
    cal_df = None
    arr = cfg.get("calibration_array")
    if arr:
        try:
            import pandas as pd
            wls = [float(p[0]) for p in arr]
            facs = [float(p[1]) for p in arr]
            if len(wls) >= 2:
                cal_df = pd.DataFrame({"wavelength": wls, "factor": facs})
        except (TypeError, ValueError, IndexError, ImportError):
            cal_df = None

    return FocusConfig(
        angle=angle,
        dispersion=dispersion,
        dispersion_nodes=nodes,
        calibration_df=cal_df,
    )


# ---------------------------------------------------------------------------
# Wavelength solution — identical logic to the GUI's poly fit + validation
# ---------------------------------------------------------------------------

def get_dispersion_poly(nodes):
    """Polynomial coefficients (highest degree first) for the non-linear
    dispersion fit if >= 2 nodes, else None.  deg capped at 3."""
    if len(nodes) < 2:
        return None
    nodes = np.asarray(nodes, dtype=float)
    pixels, wls = nodes[:, 0], nodes[:, 1]
    deg = min(len(nodes) - 1, 3)
    try:
        return np.polyfit(pixels, wls, deg=deg)
    except Exception:
        return None


def validate_dispersion_poly(poly, n_pixels):
    """Return poly unchanged if the implied mapping is strictly monotonic
    over [0, n_pixels); else None (force linear fallback)."""
    if poly is None or n_pixels is None or n_pixels < 2:
        return poly
    wls = np.polyval(poly, np.arange(n_pixels, dtype=float))
    return poly if np.all(np.diff(wls) > 0) else None


def build_wavelengths(n_pixels, dispersion, nodes):
    """Pixel index -> wavelength array, polynomial if valid else linear."""
    poly = validate_dispersion_poly(get_dispersion_poly(nodes), n_pixels)
    pixels = np.arange(n_pixels)
    wls = pixels_to_wavelengths(pixels, dispersion, poly_coeffs=poly)
    return wls, (poly is not None)


# ---------------------------------------------------------------------------
# Filename -> focuser position
# ---------------------------------------------------------------------------

def extract_focus_position(filename: str) -> Optional[int]:
    """Return the integer formed by the LAST 5 DIGITS appearing in the
    filename stem (before extension).  Returns None if fewer than 5 digits
    are present.

    Examples
    --------
    'star_focus_12345.fits'        -> 12345
    'A0V_2026-05-23_pos09850.fit'  -> 9850
    'frame98765.fits'              -> 98765
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    digits = re.findall(r"\d", stem)
    if len(digits) < 5:
        return None
    return int("".join(digits[-5:]))


# ---------------------------------------------------------------------------
# Per-frame reduction: load -> rotate -> detect -> extract
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    path: str
    position: int
    col_sums: np.ndarray            # background-subtracted 1-D spectrum (pixels)
    wls: np.ndarray                 # wavelength per pixel (Å)
    spatial_fwhm: float             # PSF FWHM in pixels (NaN if fit failed)
    source_xy: tuple
    strip: Optional[np.ndarray] = None  # raw 2-D aperture cutout (float32)
    metrics: dict = field(default_factory=dict)   # filled by score_frame
    # What the next frame of the same run needs to reuse this frame's position
    # knowledge (see anchor_from): the frame shape it was measured in, and how
    # many rows either side of the source the extraction actually touched.
    shape: tuple = ()
    row_need: int = 0


# ── Anchored reduction — the fast path for a sweep of one field ─────────────
# Finding the star costs far more than measuring it: on a 2160x3840 frame the
# full-frame stats (~800 ms), derotation (~560 ms) and DAO search (~880 ms) all
# exist to locate a source whose position the previous frame already told us,
# while the extraction that produces the actual numbers takes 4 ms.
#
# So once one frame is reduced, later frames of the same run derotate only the
# strip's rows (spectrum_core.rotate_band — exactly the same pixels the full
# rotation would have produced) and re-detect the star in a window around where
# it was.  The star is still MEASURED in every frame: this skips the searching,
# not the measuring, so a centroid that drifts between frames is followed, not
# assumed away.  Anything the fast path is not sure of — a drift past the search
# window, a strip too near an edge, a frame of a different shape, an aperture
# that outgrew the band — declines and the full-frame path below runs instead.
# --no-anchor forces the full search on every frame.
ANCHOR_SEARCH_HALF = 64     # px the centroid may drift between frames
ANCHOR_BAND_PAD = 24        # px of slack on top of the rows extraction needs


def anchor_from(fr: FrameResult) -> dict:
    """The position knowledge a reduced frame passes to the next one."""
    return {"xy": fr.source_xy, "shape": fr.shape, "row_need": fr.row_need}


def _reduce_anchored(path, pos, data, cfg: FocusConfig, args,
                     anchor: dict) -> Optional[FrameResult]:
    """Reduce one frame reusing a known source position.

    Returns None to mean "I am not confident — do the full-frame search",
    never a guessed measurement.
    """
    if tuple(anchor.get("shape") or ()) != data.shape:
        return None                     # different framing: nothing to reuse
    ax, ay = anchor["xy"]
    h, w = data.shape

    # Rows to derotate: what the extraction will need (twice over, since a
    # defocused frame's aperture is wider than the anchor frame's), plus room
    # for the star to have drifted, plus slack.
    band_half = (ANCHOR_SEARCH_HALF + 2 * int(anchor["row_need"])
                 + ANCHOR_BAND_PAD)
    y0 = int(round(ay)) - band_half
    y1 = int(round(ay)) + band_half + 1
    if y0 < 0 or y1 > h:
        return None                     # strip too near an edge to band safely

    # Background from a strided subsample: 1/16th of the pixels, and it feeds
    # only the detection threshold and the derotation's fill value.  The sky
    # that actually enters the spectrum is still measured per column, from
    # every pixel, by extract_spectrum.
    _, median, std = sigma_clipped_stats(data[::4, ::4], sigma=3.0)

    band = rotate_band(data, cfg.angle, y0, y1, cval=median)

    # Re-detect the star in a window around where it was last frame.
    by = float(ay) - y0
    wy0 = int(max(0, by - ANCHOR_SEARCH_HALF))
    wy1 = int(min(band.shape[0], by + ANCHOR_SEARCH_HALF + 1))
    wx0 = int(max(0, ax - ANCHOR_SEARCH_HALF))
    wx1 = int(min(w, ax + ANCHOR_SEARCH_HALF + 1))
    win = band[wy0:wy1, wx0:wx1]
    sources = DAOStarFinder(fwhm=args.fwhm, threshold=5.0 * std)(win - median)
    if sources is None or len(sources) == 0:
        return None                     # drifted out of the window, or gone
    sources.sort("peak", reverse=True)
    wx, wy = _dao_xy(sources[0])
    sx = wx + wx0
    sy_band = wy + wy0
    sy = sy_band + y0                   # back to full-frame rotated coords

    spectrum_width = compute_spectrum_width(cfg.dispersion, cfg.sp_max)
    if not spectrum_fully_in_frame(sx, sy, spectrum_width, cfg.angle, w, h):
        return None

    fwhm = estimate_source_fwhm(band, sx, sy_band)
    if np.isfinite(fwhm):
        aper_half = max(3, int(round(APERTURE_FWHM_MULT * fwhm)))
        sky_gap   = max(2, int(round(SKY_GAP_FWHM_MULT * fwhm)))
    else:
        aper_half = args.aperture_half
        sky_gap   = args.sky_gap
    if args.fixed_aperture is not None:
        aper_half = args.fixed_aperture

    # extract_spectrum clips its aperture and sky bands to the array it is
    # given (max(...,0) / min(...,h)).  Against a band that is too short it
    # would silently truncate a sky band instead of failing — a quietly
    # different measurement.  Decline rather than let that happen.
    row_need = aper_half + sky_gap + args.sky_width + 1
    if sy_band - row_need < 0 or sy_band + row_need >= band.shape[0]:
        return None

    (region, sky_lo, sky_hi, mask_lo, mask_hi,
     col_sums, _variance, bbox) = extract_spectrum(
        sx, sy_band, band, spectrum_width, aper_half, sky_gap, args.sky_width)

    wls, used_poly = build_wavelengths(len(col_sums), cfg.dispersion,
                                       cfg.dispersion_nodes)
    if args.calibrate and cfg.calibration_df is not None:
        col_sums = apply_calibration(wls, col_sums.astype(float),
                                     cfg.calibration_df)

    return FrameResult(
        path=path, position=pos,
        col_sums=np.asarray(col_sums, dtype=float),
        wls=np.asarray(wls, dtype=float),
        spatial_fwhm=float(fwhm),
        source_xy=(float(sx), float(sy)),
        strip=np.asarray(region, dtype=np.float32),
        shape=data.shape, row_need=int(row_need),
    )


def reduce_frame(path, cfg: FocusConfig, args,
                 anchor: Optional[dict] = None) -> Optional[FrameResult]:
    """Run the full reduction on one FITS file.  Returns a FrameResult, or
    None (with a printed reason) if the frame can't be reduced.

    With ``anchor`` (from a previous frame of the same run, via anchor_from)
    the source is looked for near where it was, instead of by a full-frame
    search — see the note above _reduce_anchored.  The fast path declines
    whenever it is not certain, and the full-frame search below runs anyway,
    so an anchor can never turn a good frame into a wrong measurement.
    """
    pos = extract_focus_position(path)
    if pos is None:
        print(f"  [skip] {os.path.basename(path)}: no 5-digit focuser position in name")
        return None

    # ── Load FITS, byteswap if needed, collapse to mono ──
    try:
        with fits.open(path) as hdu:
            raw = hdu[0].data
            if raw.dtype.byteorder not in ("=", "|", native_byteorder()):
                raw = raw.byteswap().view(raw.dtype.newbyteorder())
    except Exception as e:
        print(f"  [skip] {os.path.basename(path)}: read error: {e}")
        return None

    if raw is None:
        print(f"  [skip] {os.path.basename(path)}: empty primary HDU")
        return None

    data = to_mono(raw)

    if anchor is not None:
        fr = _reduce_anchored(path, pos, data, cfg, args, anchor)
        if fr is not None:
            return fr
        print(f"  [full] pos {pos}: anchored path declined — full-frame search")

    # ── Full-frame search: stats, derotation and a DAO sweep of the whole
    #    frame.  Slower, and the only way to find a source with no prior
    #    information — a first frame, or one whose star has moved further than the
    #    anchored path is willing to assume.
    _, median, std = sigma_clipped_stats(data, sigma=3.0)

    # ── Rotate by the config angle (same as the GUI) ──
    rotated = rotate(data, cfg.angle, reshape=False, cval=median)
    h_img, w_img = rotated.shape

    spectrum_width = compute_spectrum_width(cfg.dispersion, cfg.sp_max)

    # ── Detect sources, keep brightest valid one ──
    daofind = DAOStarFinder(fwhm=args.fwhm, threshold=5.0 * std)
    sources = daofind(rotated - median)
    if sources is None or len(sources) == 0:
        print(f"  [skip] pos {pos}: DAOStarFinder found no sources")
        return None
    sources.sort("peak", reverse=True)
    valid = [s for s in sources
             if spectrum_fully_in_frame(*_dao_xy(s), spectrum_width,
                                        cfg.angle, w_img, h_img)]
    if not valid:
        print(f"  [skip] pos {pos}: no source leaves room for a full spectrum")
        return None
    sx, sy = _dao_xy(valid[0])

    # ── Auto-fit aperture & sky gap from the measured spatial FWHM ──
    fwhm = estimate_source_fwhm(rotated, sx, sy)
    if np.isfinite(fwhm):
        aper_half = max(3, int(round(APERTURE_FWHM_MULT * fwhm)))
        sky_gap   = max(2, int(round(SKY_GAP_FWHM_MULT * fwhm)))
    else:
        aper_half = args.aperture_half
        sky_gap   = args.sky_gap

    # NB: when a frame is badly defocused the trace bloats spatially, so a
    # per-frame aperture would shrink/grow with focus and bias the depth
    # metric.  Comparing like-for-like therefore needs a FIXED aperture
    # across all frames (the median of the per-frame auto-fits is computed
    # by the caller and passed back in args.fixed_aperture if set).  The
    # default is the per-frame auto-fit; --fixed-aperture selects the
    # shared one.
    if args.fixed_aperture is not None:
        aper_half = args.fixed_aperture

    # extract_spectrum returns an 8-tuple in the current spectrum_core
    # (per-column ``variance`` was added for the uncertainty-band work).
    # The focus metrics are computed on the flux only, so variance is
    # discarded here.
    (region, sky_lo, sky_hi, mask_lo, mask_hi,
     col_sums, _variance, bbox) = extract_spectrum(
        sx, sy, rotated, spectrum_width, aper_half, sky_gap, args.sky_width)

    wls, used_poly = build_wavelengths(len(col_sums), cfg.dispersion,
                                       cfg.dispersion_nodes)

    # Optional instrument-response calibration (off by default — not needed
    # for a sharpness metric, and it would divide by a fixed curve anyway).
    if args.calibrate and cfg.calibration_df is not None:
        col_sums = apply_calibration(wls, col_sums.astype(float),
                                     cfg.calibration_df)

    return FrameResult(
        path=path, position=pos,
        col_sums=np.asarray(col_sums, dtype=float),
        wls=np.asarray(wls, dtype=float),
        spatial_fwhm=float(fwhm),
        source_xy=(float(sx), float(sy)),
        strip=np.asarray(region, dtype=np.float32),
        shape=data.shape,
        row_need=int(aper_half + sky_gap + args.sky_width + 1),
    )


# ---------------------------------------------------------------------------
# Balmer-line sharpness metrics
# ---------------------------------------------------------------------------

def measure_line(wls, flux, centre, half=LINE_HALF_WINDOW_A,
                 band=CONTINUUM_BAND_A):
    """Measure sharpness of one absorption line centred near ``centre`` (Å).

    Returns a dict with:
      depth     : (continuum - core_trough) / continuum  — relative notch depth
      gradient  : analytic max |dF/dλ| of the fitted Gaussian, continuum-norm.
      fwhm_A    : Gaussian-fit FWHM of the core in Å (NaN if the fit fails)
      ok        : bool, whether a usable continuum + core were found
    All three are oriented so that SHARPER focus -> LARGER depth, LARGER
    gradient, SMALLER fwhm_A.

    Fitting is a stable TWO-STAGE process:

      Stage 1 — fit a linear continuum to the two side-bands just outside
                the core window with ``numpy.polyfit``, then subtract it.
      Stage 2 — fit a single zero-baseline ``Gaussian1D`` (a dip) to the
                detrended core residual via ``specutils.fit_lines``.

    Two stages, rather than one joint fit, to keep the 3-parameter Gaussian
    well conditioned on a flat residual.  Fitting amplitude/mean/width and
    the continuum slope/intercept simultaneously over the raw core
    reintroduces the amplitude-versus-baseline degeneracy that lets the
    width run away — sigma of ~200 Å on the steep Hβ/Hγ continua.
    Detrending first removes that degeneracy.

    Acceptance guards: R² >= 0.80, 0 < sigma < half-window, amp > 0.
    """
    out = {"depth": np.nan, "gradient": np.nan, "fwhm_A": np.nan,
           "fit_r2": np.nan, "sigma_A": np.nan, "rel_amp": np.nan,
           "ok": False}

    core = (wls >= centre - half) & (wls <= centre + half)
    if np.count_nonzero(core) < 5:
        return out

    # -- Stage 1: linear continuum from the two side-bands --
    # Fit a Linear1D to the flux in the bands just outside +/- half and
    # evaluate it across the core.  This is exactly a first-degree polynomial;
    # using the direct linear solver avoids fit_lines' nonlinear-fitter warning.
    left  = (wls >= centre - half - band) & (wls < centre - half)
    right = (wls >  centre + half) & (wls <= centre + half + band)
    sb = left | right
    sb &= np.isfinite(wls) & np.isfinite(flux)
    if np.count_nonzero(sb) < 4:
        return out
    try:
        slope, intercept = np.polyfit(wls[sb], flux[sb], 1)
    except Exception:
        return out

    cont_at_centre = float(slope * centre + intercept)
    if not np.isfinite(cont_at_centre) or cont_at_centre <= 0:
        return out

    w = wls[core]
    f = flux[core]
    good = np.isfinite(w) & np.isfinite(f)
    if np.count_nonzero(good) < 5:
        return out
    w, f = w[good], f[good]

    cont = slope * w + intercept            # continuum across the core

    # -- Depth: deepest dip of the data below the fitted continuum line.
    # Fit-independent (a raw max-residual, not the Gaussian amplitude), so a
    # real but non-Gaussian trough still reports a depth even when the Stage-2
    # fit is rejected below.  Keeps depth an independent cross-check against
    # the fit-derived FWHM/gradient.
    depth_abs = float(np.nanmax(cont - f))
    out["depth"] = depth_abs / cont_at_centre

    # -- Stage 2: zero-baseline Gaussian dip on the DETRENDED core --
    # detrended ~ 0 on the continuum, dips < 0 in the absorption trough.  A
    # negative-amplitude Gaussian with a flat (zero) baseline is the stable
    # 3-parameter fit; the continuum slope is already removed.
    detrended = f - cont
    i_min = int(np.argmin(detrended))
    amp0 = max(depth_abs, 1e-6)
    sigma0 = max(half / 6.0, 2.0)
    core_spec = Spectrum(flux=detrended * u.dimensionless_unscaled,
                         spectral_axis=w * u.AA)
    try:
        g_fit = fit_lines(
            core_spec,
            models.Gaussian1D(amplitude=-amp0 * u.dimensionless_unscaled,
                              mean=w[i_min] * u.AA,
                              stddev=sigma0 * u.AA))
    except Exception:
        out["ok"] = True
        return out

    sigma = abs(float(g_fit.stddev.value))
    amp_fit = -float(g_fit.amplitude.value)   # flip sign: amp_fit > 0 = a dip
    fwhm_fit = 2.3548 * sigma

    # Always record the raw fit characteristics (even if the guard below
    # rejects the fit) so the diagnostics can show WHY a frame was accepted
    # or rejected.
    out["sigma_A"] = float(sigma)
    out["rel_amp"] = float(amp_fit / cont_at_centre)

    # Fit quality (R2) of the Gaussian on the detrended core: how well a
    # single Gaussian explains the trough once the continuum is removed.
    model_vals = g_fit(w * u.AA).value
    ss_res = float(np.nansum((detrended - model_vals) ** 2))
    ss_tot = float(np.nansum((detrended - np.nanmean(detrended)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    out["fit_r2"] = float(r2)

    MIN_R2 = 0.80   # core must be explained well by a single Gaussian
    plausible = (0 < sigma < half) and (amp_fit > 0)
    if plausible and r2 >= MIN_R2:
        out["fwhm_A"] = float(fwhm_fit)
        # Analytic steepest wing slope of the fitted Gaussian:
        #   max|f'| = A / (sigma * sqrt(e)),  at x = mu +/- sigma.
        # Continuum-normalised; derived from the fit so it carries no pixel
        # noise and stays consistent with fwhm_A (both come from the same
        # accepted fit, so they agree on whether a frame is good).
        out["gradient"] = float(
            amp_fit / (sigma * np.sqrt(np.e))) / cont_at_centre

    out["ok"] = True
    return out


def score_frame(fr: FrameResult, lines):
    """Populate fr.metrics with per-line measurements and aggregate scores.

    Aggregates (averaged over the lines that produced a usable measurement):
      depth_score    : mean relative depth          (bigger = sharper)
      gradient_score : mean normalised wing gradient (bigger = sharper)
      fwhm_score     : mean core FWHM in Å           (smaller = sharper)
    Plus the standalone spatial PSF FWHM (smaller = sharper).
    """
    per_line = {}
    for name in lines:
        per_line[name] = measure_line(fr.wls, fr.col_sums, BALMER[name])

    def _agg(key):
        vals = [per_line[n][key] for n in lines
                if per_line[n]["ok"] and np.isfinite(per_line[n][key])]
        return float(np.mean(vals)) if vals else np.nan

    fr.metrics = {
        "per_line":       per_line,
        "depth_score":    _agg("depth"),
        "gradient_score": _agg("gradient"),
        "fwhm_score":     _agg("fwhm_A"),
        "spatial_fwhm":   fr.spatial_fwhm,
    }


# ---------------------------------------------------------------------------
# Best-focus selection
# ---------------------------------------------------------------------------

# metric name -> (frame.metrics key, "min" or "max" is best)
METRIC_KEYS = {
    "fwhm":     ("fwhm_score",     "min"),
    "depth":    ("depth_score",    "max"),
    "gradient": ("gradient_score", "max"),
    "spatial":  ("spatial_fwhm",   "min"),
}


def pick_best(frames, metric):
    """Return the FrameResult that optimises the chosen metric."""
    key, direction = METRIC_KEYS[metric]
    scored = [(fr, fr.metrics.get(key, np.nan)) for fr in frames]
    scored = [(fr, v) for fr, v in scored if np.isfinite(v)]
    if not scored:
        return None
    if direction == "min":
        return min(scored, key=lambda t: t[1])[0]
    return max(scored, key=lambda t: t[1])[0]


def estimate_continuous_focus(positions, scores, direction="min", fit_points=5):
    """
    Fits a parabola to the points nearest the best discrete focus to
    estimate the continuous ideal focus position.
    """
    # 1. Clean and sort the data
    valid = np.isfinite(scores)
    x = np.array(positions, dtype=float)[valid]
    y = np.array(scores, dtype=float)[valid]

    if len(x) < 3:
        return None  # Not enough data to fit a parabola

    # Sort by position
    order = np.argsort(x)
    x, y = x[order], y[order]

    # 2. Find the index of the best discrete point
    best_idx = np.argmin(y) if direction == "min" else np.argmax(y)

    # 3. Select a neighborhood of points around the best point for the fit
    # (Fitting the whole curve can skew the result if the sweep is very wide,
    # as the far out-of-focus wings become linear rather than parabolic).
    half_window = fit_points // 2
    start_idx = max(0, best_idx - half_window)
    end_idx = min(len(x), start_idx + fit_points)

    # Adjust the window when it runs into the edges
    if end_idx - start_idx < fit_points:
        start_idx = max(0, end_idx - fit_points)

    x_fit = x[start_idx:end_idx]
    y_fit = y[start_idx:end_idx]

    if len(x_fit) < 3:
        return None

    # 4. Fit the parabola: y = ax^2 + bx + c
    a, b, c = np.polyfit(x_fit, y_fit, 2)

    # 5. Sanity Check the curvature
    # If looking for a minimum (FWHM), 'a' must be positive (U-shape)
    # If looking for a maximum (Depth), 'a' must be negative (Bell shape)
    if (direction == "min" and a <= 0) or (direction == "max" and a >= 0):
        print(f"  [warn] Parabola opens the wrong way! 'a'={a:.3e}")
        return None

    # 6. Calculate the vertex
    ideal_x = -b / (2.0 * a)

    # 7. Sanity check: the vertex must fall roughly within the sweep
    if ideal_x < np.min(x) or ideal_x > np.max(x):
        print(f"  [warn] Estimated focus {ideal_x:.0f} is outside the sweep limits.")
        return None

    return ideal_x, (a, b, c)
# ---------------------------------------------------------------------------
# Output: small-multiples grid + focus curve, and CSV
# ---------------------------------------------------------------------------

def write_csv(frames, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "focus_position", "depth_score",
                    "gradient_score", "line_fwhm_A", "spatial_fwhm_px"])
        for fr in sorted(frames, key=lambda r: r.position):
            m = fr.metrics
            w.writerow([
                os.path.basename(fr.path), fr.position,
                f"{m['depth_score']:.5f}"    if np.isfinite(m['depth_score'])    else "",
                f"{m['gradient_score']:.5f}" if np.isfinite(m['gradient_score']) else "",
                f"{m['fwhm_score']:.3f}"     if np.isfinite(m['fwhm_score'])     else "",
                f"{m['spatial_fwhm']:.3f}"   if np.isfinite(m['spatial_fwhm'])   else "",
            ])


def make_plot(frames, cfg, lines, metric, best, out_png, show=False):
    frames = sorted(frames, key=lambda r: r.position)
    n = len(frames)

    # Layout: a grid of spectrum panels on top, a focus-curve row at bottom.
    n_cols = min(4, max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    figsize = (4.2 * n_cols, 2.6 * n_rows + 3.2)
    if show:
        # Interactive: pyplot owns the window and the host's backend.
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize)
    else:
        fig = Figure(figsize=figsize)
        FigureCanvasAgg(fig)     # render target for savefig, backend-agnostic
    gs = GridSpec(n_rows + 1, n_cols, figure=fig,
                  height_ratios=[1] * n_rows + [1.4], hspace=0.45, wspace=0.25)

    # Shared y-scale across panels so depth differences are visually honest:
    # normalise each spectrum by its own local continuum near Hβ for display.
    for idx, fr in enumerate(frames):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[r, c])
        sel = (fr.wls >= cfg.sp_min) & (fr.wls <= cfg.sp_max)
        ax.plot(fr.wls[sel], fr.col_sums[sel], lw=0.6, color="#202020")

        # Mark the Balmer cores used in the score.
        for name in lines:
            ax.axvline(BALMER[name], color="#c04040", ls=":", lw=0.8, alpha=0.7)

        is_best = (best is not None and fr.position == best.position)
        title = f"pos {fr.position}"
        if np.isfinite(fr.metrics["fwhm_score"]):
            title += f"  | line FWHM {fr.metrics['fwhm_score']:.1f} Å"
        ax.set_title(title, fontsize=9,
                     color="#0a7d29" if is_best else "#333333",
                     fontweight="bold" if is_best else "normal")
        if is_best:
            for spine in ax.spines.values():
                spine.set_color("#0a7d29")
                spine.set_linewidth(2.0)
            ax.text(0.5, 0.92, "BEST FOCUS", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9, color="#0a7d29",
                    fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Wavelength (Å)", fontsize=6)
        ax.grid(True, lw=0.3, alpha=0.4)

    # ── Focus curve row (spans all columns) ──
    # NOTE: This is now OUTSIDE the for-loop above!
    axc = fig.add_subplot(gs[n_rows, :])
    positions = np.array([fr.position for fr in frames], dtype=float)

    def _series(key):
        return np.array([fr.metrics.get(key, np.nan) for fr in frames])

    # Normalise each metric to [0, 1] for co-plotting, orienting all so that
    # UP = sharper.  fwhm/spatial are inverted.
    def _norm(vals, invert):
        v = np.array(vals, dtype=float)
        finite = np.isfinite(v)
        if np.count_nonzero(finite) < 2:
            return v
        lo, hi = np.nanmin(v[finite]), np.nanmax(v[finite])
        if hi == lo:
            return np.where(finite, 0.5, np.nan)
        nrm = (v - lo) / (hi - lo)
        return (1.0 - nrm) if invert else nrm

    order = np.argsort(positions)
    p_sorted = positions[order]

    axc.plot(p_sorted, _norm(_series("depth_score"), False)[order],
             "-o", ms=4, label="depth (↑ sharp)", color="#1f77b4")
    axc.plot(p_sorted, _norm(_series("gradient_score"), False)[order],
             "-s", ms=4, label="wing gradient (↑ sharp)", color="#ff7f0e")
    axc.plot(p_sorted, _norm(_series("fwhm_score"), True)[order],
             "-^", ms=4, label="line FWHM (inv, ↑ sharp)", color="#2ca02c")
    axc.plot(p_sorted, _norm(_series("spatial_fwhm"), True)[order],
             "-d", ms=4, label="spatial FWHM (inv, ↑ sharp)", color="#9467bd")

    if best is not None:
        # Plot the discrete best focus
        axc.axvline(best.position, color="#0a7d29", lw=1.5, ls="--",
                    label=f"discrete best = {best.position}")

        # Plot the interpolated ideal focus
        key, direction = METRIC_KEYS[metric]
        interp = estimate_continuous_focus(positions, _series(key), direction)

        if interp:
            ideal_pos, _ = interp
            axc.axvline(ideal_pos, color="red", lw=2.0, ls="-", alpha=0.8,
                        label=f"interpolated best = {ideal_pos:.0f}")

    axc.set_xlabel("Focuser position")
    axc.set_ylabel("Normalised sharpness")
    axc.set_title("Focus sweep — all metrics (normalised, higher = sharper)",
                  fontsize=10)
    axc.grid(True, lw=0.4, alpha=0.5)
    axc.legend(fontsize=7, ncol=2, loc="lower center")

    fig.suptitle(
        f"Spectral autofocus — {n} frames — lines: {', '.join(lines)} "
        f"— ranked by '{metric}'",
        fontsize=12, y=0.995)

    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nWrote plot:  {out_png}")

    if show:
        import matplotlib.pyplot as plt
        plt.show()

def parse_args():
    ap = argparse.ArgumentParser(
        description="Spectral autofocus analyser (SA-100 / ToupTek pipeline).")
    ap.add_argument("--folder", default="./Focus_test",
                    help="Folder of FITS frames (default: ./Focus_test)")
    ap.add_argument("--config", default="spectrum_config.json",
                    help="Analysis JSON config (default: spectrum_config.json)")
    ap.add_argument("--metric", default="fwhm",
                    choices=list(METRIC_KEYS.keys()),
                    help="Metric that decides best focus (default: fwhm)")
    ap.add_argument("--include-halpha", action="store_true",
                    help="Include Hα in the Balmer score (off by default).")
    ap.add_argument("--lines", default="Hbeta,Hgamma,Hdelta",
                    help="Comma list of Balmer lines to score "
                         "(names: Halpha,Hbeta,Hgamma,Hdelta).")
    ap.add_argument("--fwhm", type=float, default=DEFAULT_FWHM_DETECT,
                    help="DAOStarFinder detection FWHM (px).")
    ap.add_argument("--aperture-half", type=int, default=DEFAULT_APERTURE_HALF,
                    help="Aperture half-height fallback (px) if FWHM fit fails.")
    ap.add_argument("--fixed-aperture", type=int, default=None,
                    help="Force a single aperture half-height (px) for ALL "
                         "frames (recommended for fair depth comparison; if "
                         "omitted, the median per-frame auto-fit is used).")
    ap.add_argument("--sky-gap", type=int, default=DEFAULT_SKY_BAND_GAP)
    ap.add_argument("--sky-width", type=int, default=DEFAULT_SKY_BAND_WIDTH)
    ap.add_argument("--calibrate", action="store_true",
                    help="Apply the config's instrument-response calibration "
                         "before scoring (not needed for sharpness; off).")
    ap.add_argument("--no-anchor", action="store_true",
                    help="Search the whole frame for the source in EVERY frame "
                         "instead of reusing the previous frame's position. "
                         "Slower (a full derotation + DAO sweep per frame); use "
                         "it when the centroid can move a lot between frames, "
                         "e.g. long exposures with drift.")
    ap.add_argument("--out-png", default="focus_analysis.png")
    ap.add_argument("--out-csv", default="focus_analysis.csv")
    ap.add_argument("--show", action="store_true",
                    help="Open an interactive window in addition to the PNG.")
    return ap.parse_args()


def resolve_lines(args):
    """Validated list of Balmer line names from --lines / --include-halpha."""
    lines = [s.strip() for s in args.lines.split(",") if s.strip()]
    if args.include_halpha and "Halpha" not in lines:
        lines.insert(0, "Halpha")
    bad = [l for l in lines if l not in BALMER]
    if bad:
        raise SystemExit(f"Unknown line name(s): {bad}. Choose from {list(BALMER)}.")
    return lines


def analyze_folder(folder, cfg, args, lines, cache=None):
    """Reduce + score every FITS in ``folder`` and pick best focus.

    The shared core of the offline CLI (main) and the live NINA sweep
    (tools/spectral_autofocus.py).  Prints the same progress/report as
    the CLI always has.  Returns ``(frames, best, ideal_pos)`` where
    ``ideal_pos`` is the parabola-interpolated position (float) or None
    when the fit was not possible.

    ``cache`` (optional) maps path -> reduced+scored FrameResult, or None
    for a frame that could not be reduced.  Hits skip reduce_frame, which
    is the expensive step; misses are reduced, scored and stored.  The live
    sweep fills it from a background thread as frames land, so an edge
    extension re-scores only the frames it just captured instead of
    re-reducing the whole run.  A None cache reduces everything from
    scratch on every call.
    """
    files = sorted(
        glob.glob(os.path.join(folder, "*.fit")) +
        glob.glob(os.path.join(folder, "*.fits")) +
        glob.glob(os.path.join(folder, "*.fts")))
    if not files:
        raise SystemExit(f"No FITS files (*.fit/*.fits/*.fts) in {folder}")

    print(f"Config: angle={cfg.angle}°  dispersion={cfg.dispersion} Å/px  "
          f"{len(cfg.dispersion_nodes)} dispersion node(s)")
    print(f"Scoring lines: {', '.join(lines)}  |  best-focus metric: {args.metric}")
    print(f"Found {len(files)} FITS file(s) in {folder}\n")

    # ── Pass 1: reduce every frame (cache hits skip the work) ──
    # The first reduced frame anchors the rest: they look for the source near
    # where it was instead of searching the whole frame again.  --no-anchor
    # keeps every frame on the full-frame search.
    use_anchor = not getattr(args, "no_anchor", False)
    anchor = None
    frames = []
    fresh = []                      # reduced here, so still unscored
    for path in files:
        if cache is not None and path in cache:
            fr = cache[path]        # None = a frame that failed to reduce
            if fr is None:
                continue
            frames.append(fr)
            if anchor is None and use_anchor:
                anchor = anchor_from(fr)
            print(f"  [done] pos {fr.position:>6}  (already analysed)")
            continue
        fr = reduce_frame(path, cfg, args, anchor=anchor if use_anchor else None)
        if cache is not None:
            cache[path] = fr        # cache failures too — don't retry them
        if fr is not None:
            frames.append(fr)
            fresh.append(fr)
            if anchor is None and use_anchor:
                anchor = anchor_from(fr)
            print(f"  [ok]   pos {fr.position:>6}  "
                  f"src=({fr.source_xy[0]:.0f},{fr.source_xy[1]:.0f})  "
                  f"spatial FWHM={fr.spatial_fwhm:.1f}px")

    if not frames:
        raise SystemExit("No frames could be reduced — check filenames and config.")

    # ── Optional: share one aperture across frames for fair comparison ──
    if args.fixed_aperture is None:
        med_fwhm = np.nanmedian([fr.spatial_fwhm for fr in frames])
        if np.isfinite(med_fwhm):
            shared = max(3, int(round(APERTURE_FWHM_MULT * med_fwhm)))
            print(f"\nNote: using per-frame auto aperture. For a strictly "
                  f"like-for-like depth comparison consider "
                  f"--fixed-aperture {shared} (median auto-fit).")

    # ── Pass 2: score ──
    # Cached frames arrive already scored (the cache's contract), so only the
    # ones reduced just now need it.  With no cache, `fresh` is every frame.
    for fr in (fresh if cache is not None else frames):
        score_frame(fr, lines)

    best = pick_best(frames, args.metric)

    # interpotale intermediate focus
    key, direction = METRIC_KEYS[args.metric]
    positions = [fr.position for fr in frames]
    scores = [fr.metrics.get(key, np.nan) for fr in frames]

    interpolation_result = (
        estimate_continuous_focus(positions, scores, direction)
        if best is not None else None)

    ideal_pos = None
    if best is None:
        # Every frame reduced but none produced a usable line measurement
        # (e.g. no star / no Balmer lines in the field — a daylight test).
        print(f"\nNo usable {args.metric} measurement in any frame — "
              f"cannot pick a best focus.")
    elif interpolation_result:
        ideal_pos, _poly_coeffs = interpolation_result
        print(f"\nBest discrete focus: position {best.position}")
        print(f"Ideal interpolated focus: position {ideal_pos:.0f}  <=== USE THIS")
    else:
        print(f"\nBest focus ({args.metric}): position {best.position}")

    # ── Report ──
    print("\nFocuser   depth     gradient   lineFWHM(Å)  spatialFWHM(px)")
    for fr in sorted(frames, key=lambda r: r.position):
        m = fr.metrics
        flag = "  <== BEST" if (best and fr.position == best.position) else ""
        print(f"{fr.position:>7}  "
              f"{m['depth_score']:>8.4f}  {m['gradient_score']:>8.4f}  "
              f"{m['fwhm_score']:>10.2f}  {m['spatial_fwhm']:>12.2f}{flag}")
    if best is not None:
        print(f"\nBest focus ({args.metric}): position {best.position}  "
              f"[{os.path.basename(best.path)}]")

    # ── Per-line fit diagnostics ──────────────────────────────────────
    # Dumps, for every frame and every scored line, exactly what the
    # Gaussian fit returned: relative amplitude, sigma, FWHM, R², and the
    # accept flag — the diagnostic for a frame whose spurious FWHM survives
    # the guards.  Each measure_line() result
    # carries these under per_line[name]; FWHM is NaN when the guard
    # rejected the fit (rejected fits drop out of the aggregate).
    print("\n── per-line fit diagnostics ──")
    print(f"{'pos':>7} {'line':>7} {'relAmp':>8} {'sigmaA':>8} "
          f"{'FWHM_A':>8} {'R2':>6}  accepted")
    for fr in sorted(frames, key=lambda r: r.position):
        for name in lines:
            pl = fr.metrics["per_line"].get(name, {})
            relamp = pl.get("rel_amp", np.nan)
            sig    = pl.get("sigma_A", np.nan)
            fw     = pl.get("fwhm_A", np.nan)
            r2     = pl.get("fit_r2", np.nan)
            accepted = "yes" if np.isfinite(fw) else "NO"
            def _f(v, w=8, p=3):
                return f"{v:>{w}.{p}f}" if np.isfinite(v) else f"{'--':>{w}}"
            print(f"{fr.position:>7} {name:>7} {_f(relamp,8,4)} {_f(sig)} "
                  f"{_f(fw)} {_f(r2,6,2)}  {accepted}")

    return frames, best, ideal_pos


def main():
    args = parse_args()

    if not os.path.isdir(args.folder):
        raise SystemExit(f"Folder not found: {args.folder}")
    cfg = load_config(args.config)
    lines = resolve_lines(args)

    frames, best, _ideal = analyze_folder(args.folder, cfg, args, lines)

    write_csv(frames, args.out_csv)
    print(f"Wrote CSV:   {args.out_csv}")
    make_plot(frames, cfg, lines, args.metric, best, args.out_png, show=args.show)


if __name__ == "__main__":
    main()
