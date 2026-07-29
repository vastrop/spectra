"""
spectrum_core.py
================
Pure computation helpers for the spectrum extraction pipeline.

This module contains the maths and reduction routines used by
spectrum_explorer.py.  Nothing here touches tkinter or holds any
application state.  Functions take their inputs as arguments and
return data; the GUI layer is responsible for wiring them up.

Matplotlib is imported because two helpers (custom_formatter,
plot_reference_lines) produce or operate on matplotlib artists.
This is fine — matplotlib is a calculation/plotting library, not a
GUI framework, and the module remains importable without a display.

The spectral line catalogue lives in a separate module (wavelength.py)
because it's pure reference data with no computational logic of its own.
plot_reference_lines accepts a {wavelength: label} dict as a parameter,
so it has no compile-time dependency on the catalogue.
"""

import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from astropy.stats import sigma_clip
from astropy.io import fits


# ---------------------------------------------------------------------------
# Byte order
# ---------------------------------------------------------------------------

def native_byteorder():
    """Return the numpy byteorder character for this machine."""
    return '<' if sys.byteorder == 'little' else '>'

# ---------------------------------------------------------------------------
# Try to be compatible with photutils <3.0
# ---------------------------------------------------------------------------


def _dao_xy(source_row):
    """Return (x, y) centroid from a DAOStarFinder row, compatible with
    photutils <3.0 ('xcentroid') and >=3.0 ('x_centroid')."""
    try:
        return source_row["x_centroid"], source_row["y_centroid"]
    except KeyError:
        return source_row["xcentroid"], source_row["ycentroid"]


# ---------------------------------------------------------------------------
# Wavelength <-> pixel mapping
# ---------------------------------------------------------------------------

def compute_spectrum_width(dispersion, sp_range_max=8000, margin_frac=0.10):
    """
    Strip width (pixels) needed to reach ``sp_range_max`` Å at the given
    linear ``dispersion`` (Å/px), plus a margin.

    The margin is proportional (default 10%) rather than a flat +5 px so
    that a non-linear dispersion fit — whose local Å/px can drop toward
    the red, mapping fewer Å per pixel than the linear estimate — still
    reaches ``sp_range_max`` instead of truncating the red end of the
    calibrated panel.  A few extra columns cost almost nothing; a strip
    that stops short silently clips the spectrum.
    """
    base = sp_range_max / dispersion
    return int(base * (1.0 + margin_frac) + 5)


def dispersion_from_geometry(lines_per_mm, distance_mm, pixel_um):
    """
    First-order linear dispersion (Å/px) of a transmission grating.

        Å/px = 10⁴ × pixel_µm / (lines_per_mm × distance_mm)

    From the grating equation at small angles: with groove spacing d, the
    first order leaves at sinθ = λ/d, so a sensor ``L`` behind the grating
    sees it displaced by ``L·λ/d`` — dλ/dx = d/L, and d = 10⁷/G Å.

    ``distance_mm`` is grating-to-SENSOR, not grating-to-filter-thread; that
    confusion is the usual reason a computed value comes out wrong by a few
    percent.  It is also the only one of the three the user typically has to
    estimate, which is why ``suggest_dispersion_nodes`` searches the scale
    rather than trusting this: this only has to land in the right ballpark.

    A wedge prism ahead of the grating (the SA100 grism configuration) does
    not change this number meaningfully.  It applies a constant deviation to
    the whole beam, shifting zero order and first order together, so it
    alters framing rather than scale; its own chromatic dispersion adds
    ~3% across 4000-8000 Å, which the node fit absorbs.

    Returns NaN on non-positive or non-finite input.
    """
    try:
        g = float(lines_per_mm)
        L = float(distance_mm)
        p = float(pixel_um)
    except (TypeError, ValueError):
        return float("nan")
    if not all(np.isfinite(v) and v > 0 for v in (g, L, p)):
        return float("nan")
    return 1.0e4 * p / (g * L)


def custom_formatter(dispersion):
    def _fmt(x, pos):
        wavelength = int(x * dispersion)
        return f"{round(wavelength / 500) * 500}"
    return _fmt


def pixels_to_wavelengths(pixels, dispersion, poly_coeffs=None):
    """
    Map pixel indices to wavelengths.

    If ``poly_coeffs`` is provided (highest degree first, as returned by
    ``np.polyfit``), the polynomial is evaluated.  Otherwise, a linear
    mapping ``wl = pixel * dispersion`` is used.

    Parameters
    ----------
    pixels : array-like
        Pixel indices along the dispersion axis (0 = first column).
    dispersion : float
        Linear Å/pixel value (used only when ``poly_coeffs`` is None).
    poly_coeffs : 1D array or None
        Polynomial coefficients, highest degree first.

    Returns
    -------
    1D ndarray of wavelengths in Å.
    """
    pixels = np.asarray(pixels, dtype=float)
    if poly_coeffs is None:
        return pixels * dispersion
    return np.polyval(poly_coeffs, pixels)


def invert_poly_to_pixel(wl, dispersion, poly_coeffs, n_pixels=None):
    """
    Find the pixel x such that polyval(poly_coeffs, x) ≈ wl.

    The search spans [0, n_pixels] when n_pixels is given (preferred — it
    matches the actual extracted spectrum), otherwise falls back to
    [0, 2 * wl / dispersion] to remain backwards-compatible.

    Returns None when ``wl`` lies outside the polynomial's coverage:
    the argmin would otherwise clamp to a grid endpoint and a reference
    line could be drawn pinned at the array edge at a wrong position.
    "Outside" means the best residual exceeds one pixel's worth of Å.
    """
    if n_pixels is not None and n_pixels > 1:
        upper = n_pixels - 1
    else:
        upper = max(1.0, wl / dispersion * 2.0)
    px_search = np.linspace(0, upper, 4000)
    vals = np.polyval(poly_coeffs, px_search)
    idx = int(np.argmin(np.abs(vals - wl)))
    if abs(vals[idx] - wl) > abs(dispersion):
        return None
    return float(px_search[idx])


def fit_dispersion_poly(nodes, delta=0.0):
    """
    Fit the pixel→wavelength dispersion polynomial to calibration nodes.

    ``nodes`` is a sequence of [pixel, wavelength_Å] pairs; the degree is
    min(len(nodes) − 1, 3).  When ``delta`` is non-zero the polynomial is
    recomposed to expect ``pixel − delta`` in place of ``pixel`` — the
    zero-order anchor shift that cancels the colour-dependent centroid
    offset between sources (Δ derivation in the explorer's
    get_dispersion_poly docstring).  The recomposition is exact — the
    same polynomial with a shifted argument — so it cannot change the
    fit's shape, only where it is evaluated; the node list itself is
    never mutated.

    Returns coefficients highest-degree-first (np.polyval order), or
    None with fewer than 2 nodes or a failed fit.
    """
    if nodes is None or len(nodes) < 2:
        return None
    arr = np.asarray(nodes, dtype=float)
    pixels, wls = arr[:, 0], arr[:, 1]
    deg = min(len(arr) - 1, 3)
    try:
        poly = np.polyfit(pixels, wls, deg=deg)
    except Exception:
        return None
    if delta:
        # numpy's Polynomial uses ascending-order coefficients; build the
        # object, substitute the linear map (x − delta), then convert back
        # to the highest-degree-first order np.polyval expects.
        coeffs_asc = poly[::-1]
        poly_obj = np.polynomial.Polynomial(coeffs_asc)
        shifted_obj = poly_obj(np.polynomial.Polynomial([-delta, 1.0]))
        poly = shifted_obj.coef[::-1]
    return poly


def validate_dispersion_poly(poly, n_pixels):
    """
    Check that ``poly`` maps [0, n_pixels) to strictly increasing
    wavelengths.

    A non-monotonic polynomial — possible with a noisy fit on 4+ nodes —
    breaks the segment-by-segment fill of the calibrated panel and can
    map two pixels to the same wavelength; detecting it once here is much
    cleaner than guarding every plotting loop.

    Returns (poly, 0) when monotonic (or when there is nothing to check),
    else (None, n_bad) where n_bad counts the non-increasing pixel steps
    — callers fall back to the linear dispersion and may report n_bad.
    """
    if poly is None or n_pixels is None or n_pixels < 2:
        return poly, 0
    pix = np.arange(n_pixels, dtype=float)
    diffs = np.diff(np.polyval(poly, pix))
    if np.all(diffs > 0):
        return poly, 0
    return None, int(np.sum(diffs <= 0))


def dispersion_fit_stats(nodes, n_pixels=None):
    """
    Fit-quality summary for the node editor's status label.

    Fits the same polynomial as ``fit_dispersion_poly`` (no anchor shift
    — the stats describe the stored nodes, not a per-source evaluation)
    and reports:

      deg       — polynomial degree used, min(N − 1, 3)
      rms       — RMS residual at the nodes (Å)
      exact     — True when N ≤ deg + 1, i.e. the fit interpolates the
                  nodes perfectly and rms is 0 by construction
      disp_min / disp_max — |dλ/dpx| range across the nodes
      monotonic — strict monotonicity over [0, n_pixels) (node-pixel span
                  when n_pixels is None): whether a calibrated panel
                  would use the fit or fall back to linear dispersion

    Returns None with fewer than 2 nodes.  Lets np.polyfit exceptions
    propagate — the caller owns the error display.
    """
    if nodes is None or len(nodes) < 2:
        return None
    arr = np.asarray(nodes, dtype=float)
    pixels, wls = arr[:, 0], arr[:, 1]
    deg = min(len(arr) - 1, 3)
    coeffs = np.polyfit(pixels, wls, deg=deg)
    fitted = np.polyval(coeffs, pixels)
    deriv = np.polyder(coeffs)
    disp_at_nodes = np.abs(np.polyval(deriv, pixels))
    n_pix = int(n_pixels) if n_pixels is not None else int(pixels.max()) + 1
    validated, _ = validate_dispersion_poly(coeffs, n_pix)
    return {
        "deg": deg,
        "rms": float(np.sqrt(np.mean((wls - fitted) ** 2))),
        "exact": len(arr) <= deg + 1,
        "disp_min": float(disp_at_nodes.min()),
        "disp_max": float(disp_at_nodes.max()),
        "monotonic": validated is not None,
    }


def build_sky_col_flag(mask_lo, mask_hi, reject_frac=0.5):
    """
    Collapse the 2-D sky-band rejection masks to a 1-D per-column flag.

    A column is flagged True when the fraction of rejected sky pixels in
    that column (across both bands) meets or exceeds ``reject_frac`` — i.e.
    the sigma-clip distrusted most of that column's sky, so the per-frame
    background there is unreliable.  Returns None when there are no sky
    rows (nothing to base a flag on) or nothing is flagged, so the sequence
    treats it as "no frozen sky flag" and falls back to per-frame sky.
    """
    rows = 0
    rej = None
    for m in (mask_lo, mask_hi):
        if m is None or m.size == 0:
            continue
        rows += m.shape[0]
        band_rej = m.sum(axis=0).astype(float)
        rej = band_rej if rej is None else rej + band_rej
    if rej is None or rows == 0:
        return None
    flag = (rej / rows) >= reject_frac
    if not flag.any():
        return None
    return flag


# ---------------------------------------------------------------------------
# Flux operations
# ---------------------------------------------------------------------------

def normalize_flux(intensities, min_val=1e-5, max_val=1.0, return_scale=False):
    """
    Normalise so the maximum equals ``max_val``, ignoring NaNs.

    The transform is purely multiplicative (``arr * max_val / hi``) and
    must NOT subtract an offset: flux *ratios* carry the physics
    (continuum colour, black-body temperature), and an affine rescaling
    that removes the spectrum's own minimum steepens the continuum past
    the Rayleigh-Jeans limit.  ``min_val`` is accepted for signature
    compatibility and is not applied to valid input.

    NaN entries in the input are preserved in the output, so callers
    can treat them as "no calibration available here" and matplotlib
    will draw gaps.

    Returns an empty array if the input is empty, so callers that
    guard plotting with ``len(arr)`` continue to work without a
    ValueError from ``np.nanmax`` on a zero-size array.

    Parameters
    ----------
    return_scale : bool
        If True, also return the multiplicative scale applied to the
        data, ``max_val / hi``.  An associated 1σ uncertainty array is
        normalised by multiplying by this same scale.  In degenerate
        cases (empty, non-finite, or non-positive-max input) the scale
        is 0.0, matching the constant output the data takes.
    """
    arr = np.asarray(intensities, dtype=float)
    if arr.size == 0:
        return (arr, 0.0) if return_scale else arr
    hi = np.nanmax(arr)
    if not np.isfinite(hi) or hi <= 0:
        out = np.full_like(arr, min_val)
        # Keep NaN gaps NaN even in the degenerate branch — the docstring
        # promise that NaNs survive normalisation holds unconditionally.
        out[~np.isfinite(arr)] = np.nan
        return (out, 0.0) if return_scale else out
    scale = max_val / hi
    out = arr * scale
    return (out, scale) if return_scale else out


def load_calibration_file(path):
    # UTF-8 first (the canonical case); fall back to latin-1 if the file
    # was written under a Windows locale and contains stray cp1252 bytes
    # in header comments.  latin-1 decodes any byte stream without error,
    # and the numeric data on each line is pure ASCII so values are
    # unaffected by the fallback choice.
    try:
        df = pd.read_csv(path, sep=r'\s+', header=None, comment="#",
                         names=["wavelength", "factor"], encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=r'\s+', header=None, comment="#",
                         names=["wavelength", "factor"], encoding="latin-1")
    df["wavelength"] = df["wavelength"].astype(str).str.replace(",", ".").astype(float)
    df["factor"] = df["factor"].astype(str).str.replace(",", ".").astype(float)
    # The consumers hand these columns to np.interp, whose xp must be
    # finite and strictly increasing — an unsorted (hand-edited or
    # third-party) response file would otherwise yield silently wrong
    # factors everywhere.  Mirrors load_reference_spectrum.  Non-finite
    # rows are dropped like NaNs; duplicates and too-short tables are
    # errors, not repairs — calibration conflicts get fixed at their
    # source, not averaged away here.
    df = df.dropna().sort_values("wavelength", ignore_index=True)
    df = df[np.isfinite(df["wavelength"]) & np.isfinite(df["factor"])]
    if len(df) < 2:
        raise ValueError(
            f"Calibration file {path} has fewer than two usable "
            f"(finite) rows — cannot interpolate a response from it.")
    if np.any(np.diff(df["wavelength"].to_numpy()) <= 0):
        raise ValueError(
            f"Calibration file {path} contains duplicate wavelengths — "
            f"the response factor there is ambiguous; fix the file.")
    return df.reset_index(drop=True)

# Display-only smoothing for the calibrated spectrum panels.  Per-column
# noise (read + sky + Poisson) varies on the same scale as a single
# pixel, while real spectral features at SA100 resolution span multiple
# pixels — so a small rolling median takes the per-pixel jitter out of
# the plotted line without losing real structure.  Applied just before
# drawing; the underlying flux arrays are preserved for measurements
# and saving.  Set to 1 to disable.

def rolling_median_nan(arr, window):
    """NaN-aware centred rolling median.

    Each output column is the median of the finite values within a
    ``window``-wide window centred on it; columns with no finite value
    in their window stay NaN (so genuine gaps in the spectrum are
    preserved, not smeared).  ``window <= 1`` returns the input
    unchanged.

    Used for two purposes in the display pipeline:
      * smoothing the per-column σ before drawing the ±2σ band
        (see ``BAND_SMOOTH_COLS`` in full_spectrum_viewer);

    Median (not mean) so a single hot/cold column doesn't bleed into
    its neighbours.
    """
    a = np.asarray(arr, dtype=float)
    if window is None or window <= 1 or a.size == 0:
        return a
    half = int(window) // 2
    n = a.size
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = a[lo:hi]
        finite = seg[np.isfinite(seg)]
        if finite.size:
            out[i] = np.median(finite)
    return out

# ---------------------------------------------------------------------------
# Two-column .dat I/O (same format as load_calibration_file)
# ---------------------------------------------------------------------------


def save_spectrum_dat(path, wavelengths, intensities, header=None):
    """
    Write a two-column whitespace-separated .dat file.

    The output format is the symmetric counterpart of
    ``load_calibration_file``: ``wavelength_A  value`` with one
    sample per line, sorted by ascending wavelength.  NaN samples are
    skipped (they would round-trip to None and break later interpolation).

    Parameters
    ----------
    path : str
    wavelengths : array-like  in Å
    intensities : array-like  same length as ``wavelengths``
    header : str or None
        Optional one-line comment written at the top, prefixed with ``# ``.
    """
    wl = np.asarray(wavelengths, dtype=float)
    fx = np.asarray(intensities, dtype=float)
    if wl.shape != fx.shape:
        raise ValueError("wavelengths and intensities must have the same shape")

    order = np.argsort(wl)
    wl = wl[order]
    fx = fx[order]
    keep = np.isfinite(wl) & np.isfinite(fx)
    wl = wl[keep]
    fx = fx[keep]

    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("# " + header.strip() + "\n")
        f.write("# wavelength_A  value\n")
        for w, v in zip(wl, fx):
            f.write(f"{w:.4f}  {v:.6e}\n")


def write_spectrum_fits(path, wavelengths, flux, sigma=None, meta=None):
    """
    Write a calibrated spectrum to a FITS file as a binary table.

    Structure (matches what ``validate_throughput.load_spectrum`` and other
    downstream readers expect):

      * Primary HDU: no image data; carries the extraction/provenance
        metadata in its header (see ``meta`` below).
      * Extension HDU named ``SPECTRUM``: a binary table with three
        double-precision columns — ``WAVELENGTH`` (Å), ``FLUX``
        (normalised), and ``SIGMA`` (1σ uncertainty, NaN-filled when no
        uncertainty is available).

    Byte-order hardening
    --------------------
    FITS binary tables are big-endian on disk.  Numpy arrays on a
    little-endian machine are native ``<f8``; handing those to astropy
    without an explicit big-endian cast was the cause of the historical
    "corrupt payload" failure (mis-aligned bytes read back as ~1e+300).
    This writer coerces every column to big-endian ``>f8`` before building
    the table, so the on-disk payload is always correctly aligned
    regardless of host byte order, and verifies the column lengths match
    before writing.

    Parameters
    ----------
    path : str
        Output path (``.fits``).
    wavelengths : 1D array-like
        Wavelengths in Å.
    flux : 1D array-like
        Flux values, same length as ``wavelengths``.
    sigma : 1D array-like or None
        Per-sample 1σ uncertainty.  If None, a NaN-filled column of the
        right length is written so the SIGMA column is always present
        (readers test it by name, not presence).
    meta : dict or None
        Provenance metadata written into the primary header.  Recognised
        keys are mapped to <=8-char FITS keywords; the important ones for
        downstream tools are ``dispersion`` -> ``DISPERSI`` and the spatial
        FWHM -> ``FWHM``.  Other scalar entries are written under
        sanitised keywords on a best-effort basis (long string values are
        truncated; non-scalar values are skipped).  Nothing here is applied
        to the data — it is recorded for traceability only.
    """
    wl = np.asarray(wavelengths, dtype=float)
    fx = np.asarray(flux, dtype=float)
    if wl.shape != fx.shape:
        raise ValueError("wavelengths and flux must have the same shape")
    n = wl.size

    if sigma is None:
        sg = np.full(n, np.nan, dtype=float)
    else:
        sg = np.asarray(sigma, dtype=float)
        if sg.shape != wl.shape:
            raise ValueError("sigma must match wavelengths in shape")

    # Coerce to big-endian double so the on-disk binary payload is always
    # correctly aligned (the historical corruption fix).
    be = np.dtype(">f8")
    wl_be = np.ascontiguousarray(wl, dtype=be)
    fx_be = np.ascontiguousarray(fx, dtype=be)
    sg_be = np.ascontiguousarray(sg, dtype=be)

    # --- Primary header: provenance metadata ---
    prihdr = fits.Header()
    prihdr["ORIGIN"] = ("Spectrum Explorer", "Writing pipeline")
    prihdr["NAXIS"] = 0
    if meta:
        # Explicit, reader-relied-upon mappings first.
        if "dispersion" in meta:
            try:
                prihdr["DISPERSI"] = (float(meta["dispersion"]),
                                      "Dispersion A/px")
            except (TypeError, ValueError):
                pass
        # Spatial FWHM may arrive under a couple of likely key names.
        for k in ("spatial_fwhm", "spatial_fwhm_px", "fwhm", "FWHM"):
            if k in meta:
                try:
                    prihdr["FWHM"] = (float(meta[k]), "Spatial FWHM px")
                    break
                except (TypeError, ValueError):
                    pass
        if "target" in meta and meta["target"]:
            prihdr["TARGET"] = (str(meta["target"])[:68], "Target / source")
        # Best-effort dump of remaining scalar metadata for traceability.
        _reserved = {"dispersion", "spatial_fwhm", "spatial_fwhm_px",
                     "fwhm", "FWHM", "target"}
        for key, val in meta.items():
            if key in _reserved:
                continue
            kw = "".join(c for c in str(key).upper()
                         if c.isalnum())[:8]
            if not kw:
                continue
            if kw in prihdr:
                continue
            if isinstance(val, bool):
                prihdr[kw] = bool(val)
            elif isinstance(val, (int, float)):
                prihdr[kw] = val
            elif isinstance(val, str):
                prihdr[kw] = val[:68]
            # non-scalar (lists, arrays, dicts) skipped on purpose
    primary = fits.PrimaryHDU(header=prihdr)

    # --- SPECTRUM binary table ---
    cols = fits.ColDefs([
        fits.Column(name="WAVELENGTH", format="D", unit="Angstrom",
                    array=wl_be),
        fits.Column(name="FLUX",       format="D", array=fx_be),
        fits.Column(name="SIGMA",      format="D", array=sg_be),
    ])
    table = fits.BinTableHDU.from_columns(cols, name="SPECTRUM")

    hdul = fits.HDUList([primary, table])
    hdul.writeto(path, overwrite=True)


def load_reference_spectrum(path):
    """
    Load a reference template (e.g. a Pickles library spectrum).

    Robust to multi-column files: only the first two whitespace-separated
    columns are read; any extras are ignored.  Comment lines starting
    with ``#`` are skipped.  Both ``.`` and ``,`` decimal separators
    are accepted.

    Pickles UVK files have the layout
        # lamA  f_a0v  s_a0v  fi  fg  fp  fs  fj  fd
    so columns 1 and 2 give wavelength (Å) and the smoothed A0V flux.

    Auto-detects wavelength units: if the maximum wavelength is below
    3000 the file is assumed to be in nm and is converted to Å.

    Returns
    -------
    wavelengths : 1D ndarray in Å (sorted ascending)
    flux        : 1D ndarray matching ``wavelengths``
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#",
                     usecols=[0, 1], names=["wavelength", "flux"],
                     encoding="utf-8")
    df["wavelength"] = df["wavelength"].astype(str).str.replace(",", ".").astype(float)
    df["flux"] = df["flux"].astype(str).str.replace(",", ".").astype(float)
    df = df.dropna().sort_values("wavelength").reset_index(drop=True)

    wl = df["wavelength"].to_numpy()
    fx = df["flux"].to_numpy()
    # Convert nm → Å if needed (Pickles UVK is already Å, so this is a no-op).
    if len(wl) and np.nanmax(wl) < 3000.0:
        wl = wl * 10.0
    return wl, fx


def load_reference_library(folder):
    """Load every ``.dat`` template in ``folder`` via
    load_reference_spectrum.  Returns {stem: (wl, fx)}; unreadable or
    empty files are silently skipped (the library viewer reports them
    interactively; batch callers just want what loads)."""
    library = {}
    if not os.path.isdir(folder):
        return library
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".dat"):
            continue
        try:
            wl, fx = load_reference_spectrum(os.path.join(folder, name))
        except Exception:
            continue
        if len(wl):
            library[os.path.splitext(name)[0]] = (wl, fx)
    return library


def match_reference_templates(wls, flux, library, step_A=20.0,
                              line_mask_half_A=40.0):
    """
    Rank reference templates (e.g. the Pickles library) by how well
    their shape matches a response-calibrated spectrum.

    Unlike the Planck fit — which uses only continuum anchors — this
    compares the *whole* spectrum, so molecular band structure (TiO in
    M stars) is signal, not nuisance.  It is therefore the right tool
    for cool stars whose colour temperature reads several hundred K
    below T_eff.

    Method: both spectra are resampled onto a common ``step_A`` grid
    over the observed band (the observation is Gaussian-smoothed to the
    grid scale first, so single noisy pixels don't get point-sampled).
    The comparison is done in log-flux with a free amplitude — the
    per-template offset has the closed-form solution "subtract the mean
    residual", exactly as in estimate_planck_temperature.  Balmer cores
    (often filled in emission-line objects) and telluric bands (absent
    from templates) are excluded.

    Parameters
    ----------
    wls, flux : 1D arrays — calibrated spectrum (Å, positive flux).
    library   : dict name -> (wl, fx), e.g. from load_reference_library.
    step_A    : comparison grid step, Å.
    line_mask_half_A : half-width of the Balmer exclusion windows.

    Returns
    -------
    dict with:
      'ranked' : list of {'name', 'rms', 'scale'} sorted best-first
                 (rms of mean-subtracted log-flux residuals; 'scale'
                 multiplies the template onto the observed spectrum), and
      'grid', 'obs', 'mask' : the comparison grid (Å), the smoothed
                 observed flux on it, and the used-pixel mask — so a
                 caller can plot exactly what was compared.
    None if fewer than 10 usable grid points or the library is empty.
    """
    wls = np.asarray(wls, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(wls) & np.isfinite(flux) & (flux > 0)
    if ok.sum() < 10 or not library:
        return None
    wls, flux = wls[ok], flux[ok]
    # np.interp needs ascending xp, and a non-monotonic wavelength array (a
    # bad dispersion poly that bypassed the GUI validation) would otherwise
    # silently produce a nonsense ranking rather than None.
    if not np.all(np.diff(wls) > 0):
        order = np.argsort(wls)
        wls, flux = wls[order], flux[order]

    # Smooth the observation to the grid scale, then resample.  The kernel
    # runs in index space after the finite-mask compression, so smoothing
    # bleeds across gap boundaries — acceptable at the 20 Å grid scale.
    sigma_px = max(1.0, step_A / max(float(np.median(np.diff(wls))), 1e-6) / 2.0)
    smooth = gaussian_filter1d(flux, sigma_px)
    grid = np.arange(wls[0], wls[-1], step_A)
    obs = np.interp(grid, wls, smooth)

    mask = (obs > 0) & ~_mask_balmer(grid, line_mask_half_A) \
        & ~_mask_telluric(grid)
    if mask.sum() < 10:
        return None
    log_obs = np.log(obs[mask])

    ranked = []
    for name, (twl, tfx) in library.items():
        if twl[0] > grid[0] or twl[-1] < grid[-1]:
            continue                      # template doesn't cover the band
        tmpl = np.interp(grid[mask], twl, tfx)
        if np.any(tmpl <= 0):
            continue
        resid = log_obs - np.log(tmpl)
        offset = resid.mean()             # free amplitude, closed form
        resid -= offset
        ranked.append({
            "name": name,
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "scale": float(np.exp(offset)),
        })
    if not ranked:
        return None
    ranked.sort(key=lambda r: r["rms"])
    return {"ranked": ranked, "grid": grid, "obs": obs, "mask": mask}


# ---------------------------------------------------------------------------
# Instrument response generation
# ---------------------------------------------------------------------------

# Balmer series cores — masked before division to suppress the deep A0V
# hydrogen absorption that no realistic smoothing can fully erase.
BALMER_LINES_A = (6562.8, 4861.3, 4340.5, 4101.7)   # Hα, Hβ, Hγ, Hδ


def _mask_balmer(wavelengths, half_width_A):
    """Boolean mask: True inside any Balmer core ±half_width_A."""
    wl = np.asarray(wavelengths, dtype=float)
    m = np.zeros_like(wl, dtype=bool)
    if half_width_A <= 0:
        return m
    for line in BALMER_LINES_A:
        m |= np.abs(wl - line) <= half_width_A
    return m


def _mask_telluric(wavelengths):
    """Boolean mask: True inside any telluric band (centre ± half-width
    from TELLURIC_BANDS_A).  Used to *protect* telluric structure from
    smoothing when building a raw response curve."""
    wl = np.asarray(wavelengths, dtype=float)
    m = np.zeros_like(wl, dtype=bool)
    for centre, half in TELLURIC_BANDS_A:
        m |= np.abs(wl - centre) <= half
    return m


def _interp_over_mask(wavelengths, values, mask):
    """Return a copy of ``values`` with the True regions of ``mask``
    replaced by linear interpolation across the gap, using only the
    unmasked, finite samples as anchors.

    Edge gaps (mask True at the array ends) are clamped to the nearest
    valid value rather than extrapolated.  Returns the input unchanged
    if there are fewer than two valid anchors.
    """
    wl = np.asarray(wavelengths, dtype=float)
    out = np.asarray(values, dtype=float).copy()
    anchor = (~mask) & np.isfinite(out)
    if np.count_nonzero(anchor) < 2:
        return out
    fill = ~anchor
    out[fill] = np.interp(wl[fill], wl[anchor], out[anchor])
    return out


def smooth_spectrum(wavelengths, flux, fwhm_A, mask=None):
    """
    Gaussian-smooth a 1D spectrum in wavelength space.

    The kernel width is specified as FWHM in Å and converted to pixels
    using the median sample spacing of ``wavelengths``.  Masked samples
    (``mask=True``) are excluded from the smoothing: the kernel is
    renormalised over the surviving neighbours so the result is well-
    defined across the masked gaps without dragging in their values.

    Parameters
    ----------
    wavelengths : 1D ndarray in Å (assumed monotonically increasing)
    flux        : 1D ndarray same length
    fwhm_A      : float, Gaussian FWHM in Å
    mask        : 1D boolean ndarray or None
        True marks samples to exclude (e.g. Balmer cores).

    Returns
    -------
    smoothed flux, same length as input (NaN where no neighbour is in range)
    """
    wl = np.asarray(wavelengths, dtype=float)
    fx = np.asarray(flux, dtype=float)
    if mask is None:
        mask = np.zeros_like(wl, dtype=bool)
    valid = (~mask) & np.isfinite(fx)

    if fwhm_A <= 0 or not np.any(valid):
        out = fx.copy()
        out[mask] = np.nan
        return out

    # Convert FWHM (Å) → sigma in pixels via the median pixel spacing.
    # Both early exits NaN the masked samples like the fwhm_A <= 0 branch
    # above — the docstring's exclusion promise holds in every branch.
    if len(wl) < 2:
        out = fx.copy()
        out[mask] = np.nan
        return out
    d_wl = np.median(np.diff(wl))
    if d_wl <= 0:
        out = fx.copy()
        out[mask] = np.nan
        return out
    sigma_px = (fwhm_A / 2.3548) / d_wl

    # Mask-aware convolution: convolve (flux*valid) and (valid) with the
    # same Gaussian, divide.  This is the standard renormalised-kernel
    # trick and handles edges + masked gaps in one shot.
    weighted = np.where(valid, fx, 0.0)
    weights = valid.astype(float)
    num = gaussian_filter1d(weighted, sigma=sigma_px, mode="nearest")
    den = gaussian_filter1d(weights,  sigma=sigma_px, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return out


def compute_instrument_response(obs_wls, obs_flux,
                                ref_wls, ref_flux,
                                smoothing_fwhm_A=50.0,
                                balmer_half_width_A=40.0):
    """
    Build an instrument response curve from an observed A0V star and a
    matching reference (e.g. Pickles A0V) template.

    Steps:
      1. Resample the reference onto ``obs_wls`` (linear interpolation;
         out-of-range samples become NaN — they are not extrapolated).
      2. Build a Balmer-core mask of ±balmer_half_width_A on both spectra.
      3. Smooth observed and reference with the same Gaussian kernel,
         excluding masked samples from the convolution.
      4. Divide observed_smoothed / reference_smoothed.

    The output factor is on the same arbitrary scale as ``obs_flux`` —
    callers using ``apply_calibration`` will divide their science
    spectrum by this curve, so only the curve's shape matters.

    Parameters
    ----------
    obs_wls, obs_flux : 1D ndarrays — wavelength-calibrated A0V observation.
    ref_wls, ref_flux : 1D ndarrays — reference template (e.g. Pickles).
    smoothing_fwhm_A  : Gaussian FWHM in Å applied to both spectra.
    balmer_half_width_A : half-width of the Balmer mask in Å.  Set to 0
                          to disable masking and rely on smoothing alone.

    Returns
    -------
    wavelengths : 1D ndarray  — same as obs_wls
    response    : 1D ndarray  — observed_smoothed / reference_smoothed.
                                NaN outside the reference's coverage.
    diagnostics : dict        — intermediate arrays for plotting:
                                'ref_resampled', 'obs_smoothed',
                                'ref_smoothed', 'mask'.
    """
    obs_wls = np.asarray(obs_wls, dtype=float)
    obs_flux = np.asarray(obs_flux, dtype=float)
    ref_wls = np.asarray(ref_wls, dtype=float)
    ref_flux = np.asarray(ref_flux, dtype=float)

    # Resample reference onto observed grid.  No extrapolation: leave NaN
    # where the observed range overruns the template.
    ref_resampled = np.interp(obs_wls, ref_wls, ref_flux,
                              left=np.nan, right=np.nan)

    mask = _mask_balmer(obs_wls, balmer_half_width_A)
    obs_smooth = smooth_spectrum(obs_wls, obs_flux,
                                 smoothing_fwhm_A, mask=mask)
    ref_smooth = smooth_spectrum(obs_wls, ref_resampled,
                                 smoothing_fwhm_A, mask=mask)

    with np.errstate(divide="ignore", invalid="ignore"):
        response = np.where(
            (ref_smooth > 0) & np.isfinite(ref_smooth),
            obs_smooth / ref_smooth,
            np.nan,
        )
        # Extend the NaN tails (where the reference didn't cover the
        # observed range) by linear extrapolation of the trend just
        # inside each edge.  Without this, the saved .dat ends where
        # the reference ends, and apply_calibration returns NaN past
        # that point — a hard cutoff users perceive as a flat edge.
        response_extended = _trend_extend_edges(obs_wls, response)
        edges_extrapolated = not np.array_equal(
            response, response_extended, equal_nan=True)
        response = response_extended

    diagnostics = {
        "ref_resampled": ref_resampled,
        "obs_smoothed":  obs_smooth,
        "ref_smoothed":  ref_smooth,
        "mask":          mask,
        "edges_extrapolated": edges_extrapolated,
    }
    return obs_wls, response, diagnostics


def compute_instrument_response_raw(obs_wls, obs_flux,
                                    ref_wls, ref_flux,
                                    balmer_half_width_A=40.0,
                                    smoothing_fwhm_A=0.0):
    """
    Build an instrument response curve at *full resolution*, preserving
    telluric absorption bands so they divide out of science spectra,
    while interpolating across the Balmer cores so the broad A0V hydrogen
    lines do not imprint on the response.

    This is the telluric-correcting counterpart to
    ``compute_instrument_response`` (which smooths everything, including
    tellurics, and so cannot remove them).

    Steps:
      1. Resample the reference onto ``obs_wls`` (no extrapolation).
      2. Raw ratio obs / ref_resampled — full resolution, tellurics intact.
      3. Mask the Balmer cores (±balmer_half_width_A) and linearly
         interpolate the ratio across each masked gap.
      4. Optionally apply a light Gaussian smoothing **only outside** the
         telluric bands (telluric pixels are pinned to their raw values).
         Defaults to 0.0 = no smoothing.
      5. Trend-extend the NaN tails (same as the smoothed path).

    Parameters
    ----------
    obs_wls, obs_flux : 1D ndarrays — wavelength-calibrated A0V observation.
    ref_wls, ref_flux : 1D ndarrays — reference template (e.g. Pickles A0V).
    balmer_half_width_A : half-width of the Balmer interpolation window (Å).
    smoothing_fwhm_A  : optional light smoothing FWHM (Å) applied only
                        outside telluric bands.  0.0 disables it.

    Returns
    -------
    wavelengths : 1D ndarray  — same as obs_wls
    response    : 1D ndarray  — raw ratio, Balmer-interpolated, telluric-
                                preserved.  NaN outside reference coverage
                                (then trend-extended at the edges).
    diagnostics : dict        — 'ref_resampled', 'ratio_raw',
                                'balmer_mask', 'telluric_mask',
                                'edges_extrapolated'.
    """
    obs_wls = np.asarray(obs_wls, dtype=float)
    obs_flux = np.asarray(obs_flux, dtype=float)
    ref_wls = np.asarray(ref_wls, dtype=float)
    ref_flux = np.asarray(ref_flux, dtype=float)

    ref_resampled = np.interp(obs_wls, ref_wls, ref_flux,
                              left=np.nan, right=np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_raw = np.where(
            (ref_resampled > 0) & np.isfinite(ref_resampled),
            obs_flux / ref_resampled,
            np.nan,
        )

    balmer_mask = _mask_balmer(obs_wls, balmer_half_width_A)
    telluric_mask = _mask_telluric(obs_wls)

    # Interpolate across the Balmer cores only — tellurics are NOT in the
    # mask, so they are left exactly as observed.
    response = _interp_over_mask(obs_wls, ratio_raw, balmer_mask)

    # Optional cosmetic smoothing that spares the telluric bands: smooth a
    # copy, then restore the raw values inside the telluric mask.
    if smoothing_fwhm_A and smoothing_fwhm_A > 0:
        smoothed = smooth_spectrum(obs_wls, response, smoothing_fwhm_A,
                                   mask=None)
        keep_raw = telluric_mask & np.isfinite(response)
        smoothed[keep_raw] = response[keep_raw]
        response = smoothed

    response_extended = _trend_extend_edges(obs_wls, response)
    edges_extrapolated = not np.array_equal(
        response, response_extended, equal_nan=True)
    response = response_extended

    diagnostics = {
        "ref_resampled":      ref_resampled,
        "ratio_raw":          ratio_raw,
        "balmer_mask":        balmer_mask,
        "telluric_mask":      telluric_mask,
        "edges_extrapolated": edges_extrapolated,
    }
    return obs_wls, response, diagnostics


def _trend_extend_edges(wls, curve, fit_window_A=200.0):
    """Replace leading/trailing NaN regions of ``curve`` with a linear
    extrapolation of the trend in the adjacent ``fit_window_A``-wide
    region of valid samples.

    Used by ``compute_instrument_response`` so the saved response
    curve has finite values across the full observed grid, even where
    the reference template doesn't cover the wavelength range.  Pure
    interior NaN gaps (a hole in the middle of valid data) are left
    alone — only the leading and trailing tails are filled.

    Parameters
    ----------
    wls : 1D ndarray  — wavelengths, monotonically increasing.
    curve : 1D ndarray  — same length; may contain NaN at the edges.
    fit_window_A : float
        Width in Å of the region used to fit the trend at each edge.
        Wider windows smooth out local wiggles but follow large-scale
        trend less faithfully.  ~200 Å is a reasonable default for
        instrument-response curves at SA100 resolution.

    Returns
    -------
    1D ndarray, copy of ``curve`` with NaN tails replaced.
    """
    wl = np.asarray(wls, dtype=float)
    cv = np.asarray(curve, dtype=float).copy()
    finite = np.isfinite(cv)
    if not finite.any():
        return cv

    i_first = int(np.argmax(finite))                       # first valid
    i_last  = len(cv) - 1 - int(np.argmax(finite[::-1]))   # last valid

    # Leading tail
    if i_first > 0:
        w_edge = wl[i_first]
        mask = finite & (wl <= w_edge + fit_window_A)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(wl[mask], cv[mask], 1)
            cv[:i_first] = slope * wl[:i_first] + intercept

    # Trailing tail
    if i_last < len(cv) - 1:
        w_edge = wl[i_last]
        mask = finite & (wl >= w_edge - fit_window_A)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(wl[mask], cv[mask], 1)
            cv[i_last + 1:] = slope * wl[i_last + 1:] + intercept

    return cv


def _interp_response_factors(wavelengths, calibration_df):
    """
    Interpolate the response table at ``wavelengths``; NaN outside coverage.

    Non-positive factors are also NaN'd: the trend-extended tails of a
    response curve (``_trend_extend_edges``) can cross zero on a falling
    edge, and dividing by a near-zero or negative factor would spike the
    calibrated edge — ``normalize_flux`` would then scale the whole
    spectrum off that spike, distorting the slope the pipeline is
    required to preserve.  The table is re-sorted here if needed because
    a calibration_df can also arrive from an embedded config array, not
    only from ``load_calibration_file``.
    """
    wl = np.asarray(wavelengths, dtype=float)
    cal_wl = calibration_df["wavelength"].values.astype(float)
    cal_fac = calibration_df["factor"].values.astype(float)
    if not np.all(np.diff(cal_wl) >= 0):
        order = np.argsort(cal_wl)
        cal_wl, cal_fac = cal_wl[order], cal_fac[order]
    factors = np.interp(wl, cal_wl, cal_fac,
                        left=np.nan, right=np.nan)
    factors[~(factors > 0)] = np.nan
    return factors


def apply_calibration(wavelengths, intensities, calibration_df):
    """
    Divide ``intensities`` by the calibration response interpolated at
    ``wavelengths``.

    Wavelengths outside the calibration table's coverage are returned as
    NaN rather than silently extrapolated to the nearest endpoint factor
    (the default ``np.interp`` behaviour), and so are wavelengths where
    the response factor is non-positive.  Plotting routines treat these
    as gaps; ``normalize_flux`` is NaN-safe.
    """
    factors = _interp_response_factors(wavelengths, calibration_df)
    with np.errstate(divide="ignore", invalid="ignore"):
        return intensities / factors


def apply_calibration_to_sigma(wavelengths, sigma, calibration_df):
    """
    Propagate a 1σ uncertainty array through the response calibration.

    Companion to ``apply_calibration``: since the calibrated flux is
    ``flux / factor``, its standard deviation scales by the same factor,
    ``sigma / factor``.  Uses the identical interpolation and NaN
    handling, so the σ array gaps exactly where the calibrated flux
    does.  Where the response factor is small (e.g. the blue end) the
    uncertainty is correctly inflated.
    """
    sig = np.asarray(sigma, dtype=float)
    factors = _interp_response_factors(wavelengths, calibration_df)
    with np.errstate(divide="ignore", invalid="ignore"):
        return sig / factors

def suggest_response_anchors(obs_wls, obs_flux, ref_wls, ref_flux,
                             n_anchors=12, line_mask_half_A=40.0,
                             sigma=3.0, n_iter=3):
    """
    Auto-suggest a starting set of anchor points for the spline-based
    instrument-response curve.

    Strategy
    --------
    1. Resample the reference onto ``obs_wls`` (no extrapolation) and
       form the raw ratio ``obs / ref`` — the noisy response curve the
       spline has to interpolate smoothly.
    2. Build an avoidance mask around the Balmer cores and the known
       telluric bands: residuals there are unreliable indicators of the
       instrument's true throughput.
    3. Split the valid wavelength range into ``n_anchors`` equal-width
       bins.  In each bin, take a sigma-clipped mean of the unmasked
       ratio samples — robust against the remaining wing residuals that
       the mask does not catch.
    4. The anchor wavelength is the centre of the bin; the anchor flux
       is the clipped mean.  Returning the bin centre (rather than a
       specific pixel) keeps the anchors evenly spaced for the spline.

    Parameters
    ----------
    obs_wls, obs_flux : 1D array-like
        Wavelength-calibrated observed A0V (counts).
    ref_wls, ref_flux : 1D array-like
        Reference template (e.g. Pickles A0V).
    n_anchors : int
        Desired number of anchors (≥ 2).  ~10–14 is usually a good
        starting point: enough to follow the curve's shape, sparse
        enough not to chase noise.
    line_mask_half_A : float
        Half-width in Å of the mask around each Balmer line and
        telluric band.
    sigma : float
        Sigma-clipping threshold for the per-bin mean.
    n_iter : int
        Number of clipping iterations.

    Returns
    -------
    list of (wavelength_Å, ratio) tuples, sorted by wavelength.  May be
    shorter than n_anchors if some bins contain no valid samples.
    """
    obs_wls = np.asarray(obs_wls, dtype=float)
    obs_flux = np.asarray(obs_flux, dtype=float)
    ref_wls = np.asarray(ref_wls, dtype=float)
    ref_flux = np.asarray(ref_flux, dtype=float)
    n_anchors = max(2, int(n_anchors))

    # Raw ratio on the observed grid.  No smoothing — the spline will
    # do the smoothing implicitly between anchors.
    ref_resampled = np.interp(obs_wls, ref_wls, ref_flux,
                              left=np.nan, right=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (ref_resampled > 0) & np.isfinite(ref_resampled),
            obs_flux / ref_resampled,
            np.nan,
        )

    # Avoidance mask: Balmer cores + tellurics.
    avoid = _mask_balmer(obs_wls, line_mask_half_A)
    for centre, half in TELLURIC_BANDS_A:
        avoid |= np.abs(obs_wls - centre) <= half

    finite = np.isfinite(ratio)
    good_wl = obs_wls[finite]
    if good_wl.size < 2:
        return []
    w_lo, w_hi = good_wl.min(), good_wl.max()
    edges = np.linspace(w_lo, w_hi, n_anchors + 1)

    anchors = []
    for i in range(n_anchors):
        in_bin = (obs_wls >= edges[i]) & (obs_wls < edges[i + 1])
        if i == n_anchors - 1:
            in_bin |= (obs_wls == edges[i + 1])

        candidates = in_bin & finite & ~avoid
        if not np.any(candidates):
            # Wholly masked bin — fall back to allowing line pixels
            # rather than leaving a hole in the spline.
            candidates = in_bin & finite
        if not np.any(candidates):
            continue

        vals = ratio[candidates]
        # Sigma-clipped mean.  Iterate a few times to converge on
        # the central population; tolerate the small bins gracefully.
        keep = np.ones_like(vals, dtype=bool)
        for _ in range(max(1, int(n_iter))):
            if np.count_nonzero(keep) < 3:
                break
            mu = float(np.mean(vals[keep]))
            sd = float(np.std(vals[keep]))
            if sd <= 0 or not np.isfinite(sd):
                break
            new_keep = np.abs(vals - mu) <= sigma * sd
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep

        if not np.any(keep):
            continue
        mean_val = float(np.mean(vals[keep]))
        centre_wl = 0.5 * (edges[i] + edges[i + 1])
        anchors.append((centre_wl, mean_val))

    return anchors

def edge_anchors_for_response(obs_wls, anchors):
    """Synthesise edge anchors at ``obs_wls[0]`` and ``obs_wls[-1]``
    from the trend of the existing anchor set.

    Each edge anchor's y-value is a 2-point linear extrapolation
    through the two anchors nearest that edge.  If the existing
    anchor set already has an anchor at (or very near) an edge,
    no anchor is synthesised for that edge.

    Parameters
    ----------
    obs_wls : 1D ndarray
        The observed wavelength grid; only its endpoints are used.
    anchors : list of (wavelength, ratio)
        The anchor set to extend.  Not modified.

    Returns
    -------
    list of (wavelength, ratio) for just the synthesised edge anchors
    (zero, one, or two entries).  The caller merges them into its
    working anchor list.
    """
    if obs_wls is None or len(obs_wls) < 2 or len(anchors) < 2:
        return []

    wl_lo, wl_hi = float(obs_wls[0]), float(obs_wls[-1])
    arr = np.array(sorted(anchors, key=lambda a: a[0]), dtype=float)
    aw, af = arr[:, 0], arr[:, 1]

    # "Near the edge" threshold: half the median anchor spacing, or
    # 5 Å, whichever is larger.  Avoids creating a duplicate edge
    # anchor on top of one the user already placed there.
    if len(aw) >= 2:
        near = max(5.0, 0.5 * float(np.median(np.diff(aw))))
    else:
        near = 5.0

    out = []
    if aw[0] - wl_lo > near:
        # Linear extrapolation from the leftmost two anchors.
        slope = (af[1] - af[0]) / (aw[1] - aw[0])
        y_lo = af[0] + slope * (wl_lo - aw[0])
        out.append((wl_lo, float(y_lo)))
    if wl_hi - aw[-1] > near:
        slope = (af[-1] - af[-2]) / (aw[-1] - aw[-2])
        y_hi = af[-1] + slope * (wl_hi - aw[-1])
        out.append((wl_hi, float(y_hi)))
    return out

# ---------------------------------------------------------------------------
# Continuum estimation and correction
# ---------------------------------------------------------------------------

# Common telluric absorption bands — used by suggest_continuum_anchors so
# auto-placed anchors avoid sitting inside atmospheric features.  Half-widths
# are roughly the band's visible extent at SA100 resolution (~80 Å).
TELLURIC_BANDS_A = (
    (6867.0, 30.0),    # O₂ B-band
    (7186.0, 25.0),    # H₂O
    (7594.0, 35.0),    # O₂ A-band
    (8227.0, 30.0),    # H₂O
)


def fit_continuum_spline(anchors_wl, anchors_flux, target_wls,
                         outside="clamp"):
    """
    Fit a cubic spline through (wavelength, flux) anchor points and
    evaluate it on a target wavelength grid.

    Parameters
    ----------
    anchors_wl   : 1D array-like  — anchor wavelengths in Å.
    anchors_flux : 1D array-like  — flux values at those wavelengths.
    target_wls   : 1D array-like  — wavelengths at which to evaluate.
    outside : {"clamp", "restrict", "extrapolate"}
        Behaviour outside the anchor range.
        ``clamp``       : hold the value of the nearest endpoint anchor.
        ``restrict``    : return NaN.
        ``extrapolate`` : let the spline extrapolate naturally.  Cubic
                          extrapolation can swing wildly; use sparingly.

    Returns
    -------
    1D ndarray, same length as target_wls.  NaN-filled if fewer than 2
    anchors are supplied (spline is undefined).
    """
    aw = np.asarray(anchors_wl, dtype=float)
    af = np.asarray(anchors_flux, dtype=float)
    tw = np.asarray(target_wls, dtype=float)

    if aw.size < 2:
        return np.full_like(tw, np.nan, dtype=float)

    # Sort anchors and drop NaN / duplicate-wavelength entries; CubicSpline
    # requires strictly increasing x.
    order = np.argsort(aw)
    aw, af = aw[order], af[order]
    keep = np.isfinite(aw) & np.isfinite(af)
    aw, af = aw[keep], af[keep]
    if aw.size < 2:
        return np.full_like(tw, np.nan, dtype=float)
    uniq = np.concatenate([[True], np.diff(aw) > 0])
    aw, af = aw[uniq], af[uniq]
    if aw.size < 2:
        return np.full_like(tw, np.nan, dtype=float)

    extrapolate = (outside == "extrapolate")
    spline = CubicSpline(aw, af, extrapolate=extrapolate)
    out = spline(tw)

    if outside == "clamp":
        # Replace out-of-range values with the nearest endpoint anchor flux.
        out = np.where(tw < aw[0],  af[0],  out)
        out = np.where(tw > aw[-1], af[-1], out)
    elif outside == "restrict":
        out = np.where((tw < aw[0]) | (tw > aw[-1]), np.nan, out)
    # "extrapolate" was already handled by CubicSpline itself.

    return out


def _contiguous_regions(wls, mask):
    """
    Convert a boolean ``mask`` (True = inside an avoidance region) into a
    list of ``(lo_A, hi_A)`` wavelength intervals covering each contiguous
    run of True values.  ``wls`` must be sorted ascending and the same
    length as ``mask``.  Returns [] if the mask is all-False.
    """
    regions = []
    in_run = False
    lo = None
    for i, m in enumerate(mask):
        if m and not in_run:
            lo = wls[i]
            in_run = True
        elif not m and in_run:
            regions.append((lo, wls[i - 1]))
            in_run = False
    if in_run:
        regions.append((lo, wls[-1]))
    return regions


def fit_continuum_auto(wls, flux, n_anchors=12, degree=3,
                       line_mask_half_A=40.0):
    """
    Fit a smooth continuum to a spectrum with ``specutils.fit_continuum``
    and return a set of evenly-spaced anchor points sampled from that fit.

    This is the *automatic* counterpart to ``suggest_continuum_anchors``:
    where that picks per-bin extreme pixels, this fits a Chebyshev model to
    the whole spectrum (with sigma-clipping and explicit line-region
    exclusion) and reads anchors off the resulting curve.  The returned
    anchors are ordinary ``(wavelength, flux)`` points — they go straight
    into the same ``continuum_anchors`` list and spline pipeline as
    manually-placed or suggested anchors, so the user can then drag or
    delete individual points to refine the standard fit.  Nothing about the
    downstream rendering/commit path changes.

    The Balmer cores and telluric bands are excluded from the fit using the
    SAME masks as ``suggest_continuum_anchors`` (``_mask_balmer`` +
    ``_mask_telluric``), so the two starting-point helpers agree on what
    "continuum" means.

    Parameters
    ----------
    wls   : 1D array-like  — wavelengths in Å (sorted ascending).
    flux  : 1D array-like  — flux at those wavelengths.
    n_anchors : int        — number of anchors to sample from the fit (≥ 2).
    degree : int           — Chebyshev1D degree for the continuum model.
                             3 is a smooth, well-behaved default; raise it
                             only for unusually structured continua.
    line_mask_half_A : float
        Half-width in Å of the avoidance mask around each Balmer line.
        Telluric bands use their own tabulated half-widths.

    Returns
    -------
    list of (wavelength_Å, flux) tuples, sorted by wavelength.  Empty list
    if there are too few finite samples to fit the requested degree.

    Notes
    -----
    ``specutils`` (and its ``astropy``/``Spectrum`` stack) is imported
    lazily here so that ``spectrum_core`` — imported by every module,
    including the headless tools — does not hard-depend on specutils being
    installed.  Only this auto-fit path requires it.
    """
    wls = np.asarray(wls, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n_anchors = max(2, int(n_anchors))
    degree = max(1, int(degree))

    finite = np.isfinite(wls) & np.isfinite(flux)
    # Need at least degree + 2 points for a sensible polynomial fit.
    if np.count_nonzero(finite) < degree + 2:
        return []
    w, f = wls[finite], flux[finite]

    # Lazy import — keeps specutils optional for the rest of spectrum_core.
    try:
        import astropy.units as u
        from astropy.modeling import models
        from astropy.modeling.fitting import LinearLSQFitter
        from specutils import Spectrum
        from specutils.fitting import fit_continuum
        from specutils.spectra import SpectralRegion
    except ImportError:
        raise ImportError(
            "Auto-fit continuum requires the 'specutils' package "
            "(pip install specutils).")

    spec = Spectrum(flux=f * u.dimensionless_unscaled,
                    spectral_axis=w * u.AA)

    # Build exclusion regions from the same Balmer + telluric masks the
    # anchor-suggester uses, then hand them to fit_continuum so the model
    # is fit only to genuine continuum pixels.
    avoid = _mask_balmer(w, line_mask_half_A) | _mask_telluric(w)
    regions = _contiguous_regions(w, avoid)
    exclude = None
    if regions:
        exclude = SpectralRegion(
            [(lo * u.AA, hi * u.AA) for lo, hi in regions])

    model = models.Chebyshev1D(degree)
    try:
        # A Chebyshev is linear in its coefficients, so the exact linear
        # least-squares fitter is the right tool — faster, no iteration
        # round-off, and it silences astropy's "model is linear in
        # parameters" advisory.  Outlier robustness here comes from the
        # excluded line/telluric regions, not from in-fitter sigma-clipping,
        # so a plain linear fit gives an identical continuum.
        cont_fit = fit_continuum(spec, model=model,
                                 fitter=LinearLSQFitter(),
                                 exclude_regions=exclude)
    except Exception:
        return []

    # Sample the fitted continuum at evenly-spaced wavelengths across the
    # finite data extent; these become the editable anchors.
    anchor_wl = np.linspace(w[0], w[-1], n_anchors)
    anchor_fx = np.asarray(cont_fit(anchor_wl * u.AA).value, dtype=float)

    good = np.isfinite(anchor_wl) & np.isfinite(anchor_fx)
    return [(float(a), float(b))
            for a, b in zip(anchor_wl[good], anchor_fx[good])]


def apply_continuum(flux, continuum, mode):
    """
    Apply a fitted continuum to a flux array.

    Parameters
    ----------
    flux      : 1D array-like
    continuum : 1D array-like, same length as flux
    mode : {"subtract", "normalize"}

    Returns
    -------
    1D ndarray, same length as flux.
    ``subtract``  → flux − continuum  (counts preserved, baseline = 0)
    ``normalize`` → flux ÷ continuum  (dimensionless, baseline ≈ 1)

    Division-by-zero and divide-by-NaN yield NaN; callers should be
    prepared for NaN segments where the continuum is undefined.
    """
    f = np.asarray(flux, dtype=float)
    c = np.asarray(continuum, dtype=float)
    if mode == "subtract":
        return f - c
    if mode == "normalize":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((c > 0) & np.isfinite(c), f / c, np.nan)
    raise ValueError(f"Unknown continuum mode: {mode!r}")


def suggest_continuum_anchors(wls, flux, n_anchors=8,
                              mode="absorption", line_mask_half_A=40.0):
    """
    Auto-suggest a starting set of continuum anchor points.

    Strategy: divide the wavelength range into ``n_anchors`` equal-width
    bins.  Within each bin, mask out pixels near known Balmer lines and
    telluric bands, then pick the extreme pixel — high for absorption-
    dominated spectra (where the continuum sits at the local maximum
    between absorption features), low for emission-dominated spectra
    (where the continuum sits at the local minimum between emission
    features).  Return the (wavelength, flux) of the chosen pixel in
    each bin.

    Parameters
    ----------
    wls   : 1D array-like  — wavelengths in Å (sorted ascending).
    flux  : 1D array-like  — flux at those wavelengths.
    n_anchors : int        — desired number of anchors (≥ 2).
    mode : {"absorption", "emission"}
        "absorption" picks the 90th-percentile pixel per bin (high points
        between absorption lines).
        "emission"   picks the 10th-percentile pixel per bin (low points
        between emission lines).
    line_mask_half_A : float
        Half-width in Å of the avoidance mask around each Balmer line
        and telluric band.  Pixels inside are excluded from the
        per-bin candidate pool.  If a bin ends up entirely masked, the
        mask is dropped for that bin (better an anchor on a line than
        no anchor at all).

    Returns
    -------
    list of (wavelength_Å, flux) tuples, sorted by wavelength.  May be
    shorter than n_anchors if some bins contain no valid pixels.
    """
    wls = np.asarray(wls, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n_anchors = max(2, int(n_anchors))

    if mode not in ("absorption", "emission"):
        raise ValueError(f"Unknown mode: {mode!r}")
    pct = 90.0 if mode == "absorption" else 10.0

    # Build the avoidance mask (True = avoid this pixel).
    avoid = _mask_balmer(wls, line_mask_half_A)
    for centre, half in TELLURIC_BANDS_A:
        avoid |= np.abs(wls - centre) <= half

    finite = np.isfinite(flux)

    # Bin edges in wavelength.  Use the actual data extent rather than
    # the nominal [sp_min, sp_max] window — anchors should be inside
    # the region where the spectrum actually exists.
    good_wl = wls[finite]
    if good_wl.size < 2:
        return []
    w_lo, w_hi = good_wl.min(), good_wl.max()
    edges = np.linspace(w_lo, w_hi, n_anchors + 1)

    anchors = []
    for i in range(n_anchors):
        in_bin = (wls >= edges[i]) & (wls < edges[i + 1])
        if i == n_anchors - 1:
            in_bin |= (wls == edges[i + 1])   # include the upper edge

        candidates = in_bin & finite & ~avoid
        if not np.any(candidates):
            # Whole bin is masked — fall back to allowing line pixels.
            candidates = in_bin & finite
        if not np.any(candidates):
            continue   # nothing usable in this bin

        bin_flux = flux[candidates]
        bin_wls = wls[candidates]
        # Pick the pixel whose flux is at the chosen percentile.  Use
        # nearest-value selection rather than np.percentile's interpolated
        # output so the anchor sits on a real pixel.
        target_val = np.percentile(bin_flux, pct)
        idx = int(np.argmin(np.abs(bin_flux - target_val)))
        anchors.append((float(bin_wls[idx]), float(bin_flux[idx])))

    return anchors


# hc/k in Å·K (second radiation constant) and Wien displacement in Å·K
_PLANCK_C2_A_K = 1.4387769e8
_WIEN_B_A_K = 2.8977719e7


def planck_relative(wls_A, temp_K):
    """
    Un-normalised Planck spectral radiance B_λ(T) shape at wavelengths
    ``wls_A`` (Å): λ⁻⁵ / (exp(c₂/λT) − 1).  Absolute scale is
    meaningless here (it is absorbed by a free amplitude in the fit) —
    the values are tiny but well within float range.
    """
    wls = np.asarray(wls_A, dtype=float)
    x = _PLANCK_C2_A_K / (wls * float(temp_K))
    with np.errstate(over="ignore"):
        return wls ** -5.0 / np.expm1(x)


def _planck_fit_one(wl, fx, t_min, t_max, n_grid):
    """Single Planck-temperature fit on clean arrays (see the public
    wrapper estimate_planck_temperature for the method description)."""
    log_f = np.log(fx)

    def scan(lo, hi):
        temps = np.geomspace(lo, hi, int(n_grid))
        chi = np.empty(temps.size)
        for i, T in enumerate(temps):
            with np.errstate(over="ignore", divide="ignore"):
                log_b = np.log(planck_relative(wl, T))
            resid = log_f - log_b
            resid -= resid.mean()      # closed-form amplitude
            chi[i] = float(np.dot(resid, resid))
        return temps, chi

    # Coarse pass over the full range, then a zoomed pass around the
    # minimum so t_best and the interval are not grid-quantised (the
    # Δχ²=1 valley can be narrower than one coarse grid step).
    temps, chi = scan(float(t_min), float(t_max))
    i0 = int(np.argmin(chi))
    if 0 < i0 < temps.size - 1:
        temps_z, chi_z = scan(temps[i0 - 1], temps[i0 + 1])
        # Merge the two grids so the Δχ²=1 contour below still sees the
        # full range (open bounds) at fine resolution near the minimum.
        temps = np.concatenate([temps, temps_z])
        chi = np.concatenate([chi, chi_z])
        order = np.argsort(temps)
        temps, chi = temps[order], chi[order]

    i_best = int(np.argmin(chi))
    t_best = float(temps[i_best])

    # Scale χ² so reduced χ²_min = 1, then take the Δχ²=1 interval.
    dof = max(wl.size - 2, 1)
    sigma2 = chi[i_best] / dof if chi[i_best] > 0 else 0.0
    inside = chi <= chi[i_best] + max(sigma2, 1e-300)
    t_lo = float(temps[inside].min())
    t_hi = float(temps[inside].max())
    lo_open = bool(inside[0])
    hi_open = bool(inside[-1])

    with np.errstate(over="ignore", divide="ignore"):
        log_b = np.log(planck_relative(wl, t_best))
    log_amp = float(np.mean(log_f - log_b))
    rms = float(np.sqrt(chi[i_best] / wl.size))

    peak_A = _WIEN_B_A_K / t_best
    if peak_A < wl.min():
        regime = "uv-side"
    elif peak_A > wl.max():
        regime = "ir-side"
    else:
        regime = "in-band"

    return {
        "t_best": t_best, "t_lo": t_lo, "t_hi": t_hi,
        "lo_open": lo_open, "hi_open": hi_open,
        "log_amp": log_amp, "rms": rms,
        "peak_A": peak_A, "regime": regime,
    }


def estimate_planck_temperature(anchors_wl, anchors_flux,
                                t_min=2000.0, t_max=150000.0, n_grid=400):
    """
    Estimate a black-body (colour) temperature from continuum anchor
    points by fitting a Planck curve with a free amplitude.

    The fit is done in log-flux, where the unknown amplitude (distance,
    radius, response normalisation) becomes an additive offset with a
    closed-form least-squares solution — so only T is searched, on a
    log-spaced grid (coarse pass, then a zoomed pass around the
    minimum).  The confidence interval starts from the Δχ² = 1 contour
    after scaling χ² so that reduced χ²_min = 1 (anchors carry no
    error bars).  On the Rayleigh-Jeans side the χ² valley is flat
    towards high T, so the interval honestly reports an open upper
    bound ("hotter than X") instead of a fake point estimate.

    Because on the RJ side almost all the temperature information sits
    in the extreme blue and red anchors, a single anchor can dominate
    the fit (high leverage).  With ≥ 4 anchors a leave-one-out pass
    refits with each anchor dropped in turn; the reported interval is
    widened to the union of all Δχ²=1 intervals, and the spread of the
    leave-one-out best fits is returned so the caller can show how
    fragile the estimate is.

    Parameters
    ----------
    anchors_wl, anchors_flux : 1D array-like
        Continuum anchor wavelengths (Å) and fluxes.  Need ≥ 3 finite
        anchors with flux > 0.
    t_min, t_max : float — temperature search range, K.
    n_grid : int — number of log-spaced grid temperatures.

    Returns
    -------
    dict with keys, or None if fewer than 3 usable anchors:
      t_best       : float — best-fit temperature (K), all anchors
      t_lo, t_hi   : float — interval bounds (K): union of the all-anchor
                     Δχ²=1 interval and every leave-one-out interval
      lo_open, hi_open : bool — True if the bound hit the grid edge
                     (the data cannot constrain that side)
      loo_lo, loo_hi : float or None — min/max of the leave-one-out
                     best-fit temperatures (None if < 4 anchors)
      log_amp      : float — additive log-flux offset at t_best; the
                     model is exp(log_amp) * planck_relative(wl, t_best)
      rms          : float — rms of log-flux residuals at t_best
      peak_A       : float — Wien peak wavelength of t_best (Å)
      regime       : {"in-band", "uv-side", "ir-side"} — where the
                     Planck peak sits relative to the anchor range
      loglog_slope : float — d logF / d logλ of the anchors
      super_rj     : bool — True if the continuum falls steeper than
                     λ⁻⁴, i.e. bluer than the T→∞ Rayleigh-Jeans limit.
                     No black body can do that: the flux calibration
                     (response correction / reddening) is then suspect
                     and the temperature is meaningless.
    """
    wl = np.asarray(anchors_wl, dtype=float)
    fx = np.asarray(anchors_flux, dtype=float)
    ok = np.isfinite(wl) & np.isfinite(fx) & (fx > 0)
    wl, fx = wl[ok], fx[ok]
    if wl.size < 3:
        return None

    fit = _planck_fit_one(wl, fx, t_min, t_max, n_grid)
    slope = float(np.polyfit(np.log(wl), np.log(fx), 1)[0])
    fit["loglog_slope"] = slope
    fit["super_rj"] = slope < -4.0
    fit["loo_lo"] = fit["loo_hi"] = None

    if wl.size >= 4:
        keep = np.ones(wl.size, dtype=bool)
        loo_best = []
        for i in range(wl.size):
            keep[i] = False
            sub = _planck_fit_one(wl[keep], fx[keep], t_min, t_max, n_grid)
            keep[i] = True
            loo_best.append(sub["t_best"])
            fit["t_lo"] = min(fit["t_lo"], sub["t_lo"])
            fit["t_hi"] = max(fit["t_hi"], sub["t_hi"])
            fit["lo_open"] |= sub["lo_open"]
            fit["hi_open"] |= sub["hi_open"]
        fit["loo_lo"] = float(min(loo_best))
        fit["loo_hi"] = float(max(loo_best))

    return fit


def wavelength_to_rgb(wavelength_angstrom, gamma=0.8):
    wl = float(wavelength_angstrom) / 10.0
    if 380 <= wl <= 440:
        att = 0.3 + 0.7 * (wl - 380) / 60
        R = ((-(wl - 440) / 60) * att) ** gamma
        G = 0.0
        B = (att) ** gamma
    elif 440 < wl <= 490:
        R = 0.0
        G = ((wl - 440) / 50) ** gamma
        B = 1.0
    elif 490 < wl <= 510:
        R = 0.0
        G = 1.0
        B = (-(wl - 510) / 20) ** gamma
    elif 510 < wl <= 580:
        R = ((wl - 510) / 70) ** gamma
        G = 1.0
        B = 0.0
    elif 580 < wl <= 645:
        R = 1.0
        G = (-(wl - 645) / 65) ** gamma
        B = 0.0
    elif 645 < wl <= 800:
        att = 0.3 + 0.7 * (800 - wl) / 155
        R = (att) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0
    return (R, G, B)


def rainbow_fill(ax, x, flux, zorder=2, color_wls=None, flux_alpha=True,
                 smooth=False):
    """
    Draw the per-wavelength rainbow fill under ``flux`` as ONE artist.

    Replaces the historical one-fill_between-per-pixel-pair loop (hundreds
    of artists rebuilt per redraw — the main render cost of the spectrum
    panels) with a single PolyCollection: one quad per segment.

    Parameters
    ----------
    x : abscissa values (Å for the calibrated panels, pixel index for the
        raw panel).
    color_wls : wavelengths (Å) used for the segment colour; defaults to
        ``x`` (i.e. x is already in Å).
    flux_alpha : when True, alpha = clip(flux, 0.05, 1) — the calibrated
        panels' treatment; when False segments are opaque (raw panel).
    smooth : bring back that historical per-pair ``fill_between`` loop —
        antialiased fills blend at segment boundaries, so colour
        transitions render smoother than the hard-edged quads.  Slow
        (one artist per segment); for offline rendering (the DB
        browser's poster export), never for the interactive panels.

    Segments touching a non-finite sample are skipped, matching the old
    loop's NaN-gap behaviour.
    """
    from matplotlib.collections import PolyCollection
    if color_wls is None:
        color_wls = x
    ok = np.isfinite(flux)
    if smooth:
        for j in range(len(x) - 1):
            if not (ok[j] and ok[j + 1]):
                continue
            r, g, b = wavelength_to_rgb(color_wls[j])
            a = float(np.clip(flux[j], 0.05, 1.0)) if flux_alpha else 1.0
            ax.fill_between([x[j], x[j + 1]], [flux[j], flux[j + 1]], 0.0,
                            color=(r, g, b, a), linewidth=0, zorder=zorder)
        return
    verts, colors = [], []
    for j in range(len(x) - 1):
        if not (ok[j] and ok[j + 1]):
            continue
        r, g, b = wavelength_to_rgb(color_wls[j])
        a = float(np.clip(flux[j], 0.05, 1.0)) if flux_alpha else 1.0
        verts.append(((x[j], 0.0), (x[j], flux[j]),
                      (x[j + 1], flux[j + 1]), (x[j + 1], 0.0)))
        colors.append((r, g, b, a))
    if verts:
        ax.add_collection(PolyCollection(
            verts, facecolors=colors, edgecolors="none", zorder=zorder))


# ---------------------------------------------------------------------------
# Peak / trough snap
# ---------------------------------------------------------------------------

def snap_to_peak(spectrum, pixel, half_window=10, sigma=4.0):
    """
    Snap to the nearest local emission peak or absorption trough.

    Within ``[pixel ± half_window]`` (clipped to bounds), peak and trough
    candidates are scored separately:

      * ``score_peak``   = (flux − local_median, clipped at 0) × Gaussian
      * ``score_trough`` = (local_median − flux, clipped at 0) × Gaussian

    The Gaussian is centred on the click pixel with width ``sigma``, so the
    snap stays close to where the user clicked.  Scoring peaks and troughs
    in separate one-sided arrays prevents a steep continuum on one side of
    the click from out-scoring a smaller feature on the other, which a
    symmetric ``|x − median|`` score cannot avoid.

    Whichever side has the larger best score wins.  If neither has any
    excursion, the click pixel is returned unchanged.

    Parameters
    ----------
    spectrum : 1D ndarray
        The 1D pixel-axis spectrum (column sums).
    pixel : float
        Initial click pixel.
    half_window : int
        Snap window half-width in pixels (default 10).
    sigma : float
        Gaussian weight width in pixels (default 4); larger ⇒ more willing
        to jump across to a stronger, more distant feature.
    """
    n = len(spectrum)
    if n == 0:
        return float(pixel)  # every other path returns float
    p0 = int(np.clip(round(pixel), 0, n - 1))
    lo = max(0, p0 - half_window)
    hi = min(n, p0 + half_window + 1)

    window = spectrum[lo:hi].astype(float)
    indices = np.arange(lo, hi, dtype=float)

    # Detrend against a local LINEAR continuum, not the window median.
    # On a steep monotonic shoulder (e.g. Hδ on the blue rise of an A-star)
    # the median leaves a large slope-driven residual that can make the
    # bright upslope out-score the real absorption dip, flipping the snap to
    # the wrong polarity.  A line fit through the window edges removes that
    # slope so only genuine peaks/troughs score.  Edges = outer third each
    # side, so the central feature itself doesn't bias the baseline.
    third = max(1, len(window) // 3)
    edge_i = np.concatenate([indices[:third], indices[-third:]])
    edge_w = np.concatenate([window[:third], window[-third:]])
    try:
        b_slope, b_int = np.polyfit(edge_i, edge_w, 1)
        baseline = b_slope * indices + b_int
    except Exception:
        baseline = np.median(window)
    detrended = window - baseline

    weight = np.exp(-((indices - pixel) ** 2) / (2.0 * sigma * sigma))

    above = np.maximum(detrended, 0.0)  # peaks only
    below = np.maximum(-detrended, 0.0)  # troughs only

    score_peak = above * weight
    score_trough = below * weight

    best_peak = score_peak.max() if score_peak.size else 0.0
    best_trough = score_trough.max() if score_trough.size else 0.0

    if best_peak == 0.0 and best_trough == 0.0:
        return float(p0)  # featureless window — stay at the click

    # Coarse feature pixel and polarity from the windowed score, then refine
    # to a sub-pixel centre with a profile fit (see refine_centroid).  The
    # fit window is wider than this snap window so the line wings anchor μ.
    if best_peak >= best_trough:
        coarse = lo + int(np.argmax(score_peak))
        polarity = +1
    else:
        coarse = lo + int(np.argmax(score_trough))
        polarity = -1
    return refine_centroid(spectrum, coarse, polarity)

def refine_centroid(spectrum, pixel, polarity, fit_half_window=25):
    """
    Refine a coarse feature pixel to a sub-pixel centre by fitting a
    Gaussian-on-a-linear-baseline profile.

    This is the accurate counterpart to taking the raw argmin/argmax of a
    feature: an integer extremum is quantised and, on a broad flat-bottomed
    line (e.g. an A-star Balmer core), dominated by noise over which exact
    pixel is lowest.  A profile fit uses the whole core plus its wings, so
    the returned centre is sub-pixel and noise-robust.

    The model is
        f(x) = baseline(x)  ∓  A · exp(-(x - μ)² / (2σ²))
    with a linear baseline (slope + intercept) so a sloping local continuum
    does not bias μ.  ``polarity`` selects the sign:
        polarity = -1  → absorption trough (minus a Gaussian)
        polarity = +1  → emission peak     (plus a Gaussian)

    The fit window (``±fit_half_window``) is deliberately wider than the
    coarse feature-finding window so the line wings anchor μ.  Keep it narrow
    enough not to cross into an adjacent feature: at 7.7 Å/px the closest
    Balmer pair (Hγ–Hδ) is ~31 px apart, so the default ±25 stays clear while
    still capturing a defocused SA100 core.

    Parameters
    ----------
    spectrum : 1D ndarray
        Pixel-axis spectrum (column sums).
    pixel : float or int
        Coarse feature pixel (e.g. from snap_to_peak's argmin/argmax head).
    polarity : int
        -1 for an absorption trough, +1 for an emission peak.
    fit_half_window : int
        Half-width of the fit window in pixels (default 25).

    Returns
    -------
    float
        Sub-pixel centre μ on a successful, sane fit; otherwise the coarse
        ``pixel`` unchanged (never worse than the integer estimate).
    """
    n = len(spectrum)
    if n == 0:
        return float(pixel)
    p0 = int(np.clip(round(pixel), 0, n - 1))
    lo = max(0, p0 - fit_half_window)
    hi = min(n, p0 + fit_half_window + 1)

    x = np.arange(lo, hi, dtype=float)
    y = np.asarray(spectrum[lo:hi], dtype=float)
    good = np.isfinite(y)
    if np.count_nonzero(good) < 5:
        return float(p0)
    x, y = x[good], y[good]

    # Linear baseline estimate from the window edges (robust to the dip/peak
    # sitting in the middle): median of the outer thirds on each side.
    third = max(1, len(y) // 3)
    edge_x = np.concatenate([x[:third], x[-third:]])
    edge_y = np.concatenate([y[:third], y[-third:]])
    try:
        b_slope, b_int = np.polyfit(edge_x, edge_y, 1)
    except Exception:
        b_slope, b_int = 0.0, float(np.median(y))
    baseline_at = b_slope * x + b_int

    # Amplitude seed from the coarse pixel's excursion against the baseline.
    resid = polarity * (y - baseline_at)          # positive at the feature
    amp0 = float(np.max(resid)) if resid.size else 0.0
    if not np.isfinite(amp0) or amp0 <= 0:
        return float(p0)

    def model(xx, amp, mu, sigma, slope, intercept):
        return (slope * xx + intercept) + polarity * amp * np.exp(
            -((xx - mu) ** 2) / (2.0 * sigma * sigma))

    try:
        popt, _ = curve_fit(
            model, x, y,
            p0=[amp0, float(p0), 4.0, b_slope, b_int],
            maxfev=2000,
        )
    except Exception:
        return float(p0)

    amp_fit, mu_fit, sigma_fit = popt[0], popt[1], abs(popt[2])

    # Sanity guards — any failure returns the coarse pixel.
    if amp_fit <= 0:                                  # feature not present
        return float(p0)
    if not (lo <= mu_fit <= hi - 1):                  # ran away from window
        return float(p0)
    if sigma_fit < 0.5 or sigma_fit > fit_half_window:  # spike or continuum
        return float(p0)
    return float(mu_fit)

# Balmer lines used for auto dispersion suggestion (always present in an
# A-star), blue to red.  All four are required while the scale is being
# searched; with the scale pinned the solver tolerates a missing line, since
# Hδ can fall off the left edge on some framings.
AUTO_BALMER_A = (4101.7, 4340.5, 4861.3, 6562.8)   # Hδ Hγ Hβ Hα
AUTO_TELLURIC_A = 7594.0                            # O2 A-band (optional node)

# Floor under a line's dip strength before it enters the geometric-mean score
# in suggest_dispersion_nodes.  Keeps log() finite on an absent line while
# leaving it heavily penalised (log 1e-6 = -13.8 against a real line's ~-2.5).
_SCORE_FLOOR = 1e-6


def _dip_strength(spectrum, px, half=12):
    """Local absorption depth at integer pixel ``px``, normalised by the
    local linear continuum. 0 if off-array or no dip. Used only to score the
    coarse zero-point scan — the precise centre comes later from
    refine_centroid."""
    n = len(spectrum)
    lo, hi = max(0, int(px) - half), min(n, int(px) + half + 1)
    if hi - lo < 5:
        return 0.0
    x = np.arange(lo, hi, dtype=float)
    y = np.asarray(spectrum[lo:hi], dtype=float)
    good = np.isfinite(y)
    if np.count_nonzero(good) < 5:
        return 0.0
    x, y = x[good], y[good]
    third = max(1, len(y) // 3)
    ex = np.concatenate([x[:third], x[-third:]])
    ey = np.concatenate([y[:third], y[-third:]])
    try:
        s, b = np.polyfit(ex, ey, 1)
        base = s * x + b
    except Exception:
        base = np.median(y)
    depth = np.max(base - y)
    cont = np.median(base)
    return float(depth / cont) if cont > 0 else 0.0


def suggest_dispersion_nodes(spectrum, dispersion,
                             zero_point_range=(-40, 120),
                             include_telluric=True,
                             search_frac=0.25, search_steps=81):
    """
    Auto-suggest dispersion-calibration nodes for an A-type star.

    Strategy (predict, then fit):
      1. Scan the zero-order pixel x0 over ``zero_point_range`` *and* the
         scale over a band around ``dispersion``.
      2. At each (x0, Å/px), predict each Balmer line's pixel as
         ``x0 + λ/disp`` and score the trial by the geometric mean of the
         normalised dip strength there.  The Balmer convergence pattern
         makes the correct trial a sharp maximum.
      3. At the best trial, refine each in-frame Balmer line to a sub-pixel
         centre with ``refine_centroid`` (absorption polarity).
      4. Predict the telluric 7594 pixel from the Balmer-anchored linear fit
         and add it as a node only if a clean dip is actually present.

    The scale is searched, not trusted
    ----------------------------------
    ``dispersion`` is a seed, not a known quantity: the scale is what this
    calibration exists to measure, and a 10% error already puts Hβ ~60 px
    from its dip, so every predicted line misses.  Searching the scale
    demotes ``dispersion`` from "must be correct" to "must be in the right
    ballpark".

    The band is ``dispersion × (1 ± search_frac)``.  ``search_steps`` across
    ±25% keeps the worst-case line (Hα, the longest lever arm) within
    ~5 px of its prediction between grid points — inside ``_dip_strength``'s
    ±12 px window, so no basin falls between the teeth.  ``search_frac=0``
    pins the scale to ``dispersion`` exactly (single-scale zero-point scan).

    All four Balmer lines are required when the scale is free
    ---------------------------------------------------------
    A shifted line assignment — Hδ landing where Hγ really is, and so on —
    reproduces a different but self-consistent scale (7.65 × 4340.5/4861.3
    ≈ 6.83; 7.65 × 4861.3/6562.8 ≈ 5.67).  These aliases are strong
    attractors on real data.  Requiring all four lines (``min_seen`` rises
    to ``len(AUTO_BALMER_A)`` when the scale is free) suppresses them,
    because the fourth predicted line must also land on a feature.  It also
    bounds the search from below at no cost: a scale small enough to push
    Hα off the end of the strip cannot present four in-frame lines.

    Scoring is the geometric mean of the four dip strengths: it collapses
    when any single line is absent, which kills the aliases, while staying
    smooth in between, which keeps the optimum sharply localised.
    ``_SCORE_FLOOR`` keeps the logarithm finite.

    Domain
    ------
    The scan needs a spectrum in which the Balmer pattern is actually
    detectable; a badly defocused frame has no pattern to find.  Outside a
    capture range of roughly −20% to +35% of the true scale it can return a
    confident wrong lock — plausible nodes, small linear residuals, no
    numerical sign of trouble.  No reliable self-diagnosis exists for that
    case, so the caller plots the nodes for user approval rather than
    applying them silently.

    Returns
    -------
    nodes : list of [pixel, wavelength_Å], sorted by pixel.  May be empty
            if too few Balmer lines are found (caller should warn).
    info  : dict with 'x0', 'dispersion' (the fitted Å/px from the refined
            nodes), 'dispersion_scan' (the grid value that won),
            'dispersion_prior' (what the caller passed), 'n_balmer',
            'telluric_added', and 'residuals' (per-node linear-fit residual
            in Å) for diagnostics.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    n = len(spectrum)
    if n < 50 or not np.isfinite(dispersion) or dispersion <= 0:
        return [], {"error": "spectrum too short or bad dispersion"}

    # _dip_strength depends only on int(px), so tabulating it once per pixel
    # is exact and reduces each trial to four array lookups: one table build
    # of n polyfits instead of n_grid × n_x0 of them.
    strength = np.array([_dip_strength(spectrum, i) for i in range(n)],
                        dtype=float)

    # Suppress windows with no continuum to normalise against.  _dip_strength
    # returns depth/continuum, which is unbounded as the continuum tends to
    # zero, and every strip has columns bluer than the atmospheric cutoff
    # where the star has no flux and noise-over-near-zero scores deeper than
    # any real Balmer wing.  With the scale free the scan is otherwise drawn
    # into that dead blue end and pins at the edge of the band.
    _H = 12                                   # _dip_strength's default window
    local = np.convolve(np.nan_to_num(spectrum, nan=0.0),
                        np.ones(2 * _H + 1) / (2 * _H + 1), mode="same")
    ref = np.nanpercentile(spectrum, 90)
    if np.isfinite(ref) and ref > 0:
        strength[local < 0.05 * ref] = 0.0
    # An absorption dip cannot be deeper than the continuum it sits on, so
    # depth/continuum > 1 is a failed local fit, not a strong line.
    np.clip(strength, 0.0, 1.0, out=strength)

    if search_frac and search_frac > 0:
        grid = np.linspace(dispersion * (1.0 - search_frac),
                           dispersion * (1.0 + search_frac),
                           int(search_steps))
        grid = grid[grid > 0]
        min_seen = len(AUTO_BALMER_A)
    else:
        grid = np.array([float(dispersion)])
        min_seen = 2

    # ── Coarse (zero-point × scale) scan ──
    x0_lo, x0_hi = zero_point_range
    x0s = np.arange(x0_lo, x0_hi + 1, 1.0)
    best_x0, best_disp, best_score = None, None, -np.inf
    for disp in grid:
        line_px = np.array([wl / disp for wl in AUTO_BALMER_A])
        for x0 in x0s:
            log_sum = 0.0
            seen = 0
            for px in x0 + line_px:
                if 0 <= px < n:
                    log_sum += math.log(max(strength[int(px)], _SCORE_FLOOR))
                    seen += 1
            if seen >= min_seen and log_sum / seen > best_score:
                best_score, best_x0, best_disp = log_sum / seen, x0, float(disp)
    if best_x0 is None:
        return [], {"error": "no zero-point produced detectable Balmer dips"}

    # ── Refine each in-frame Balmer line to sub-pixel ──
    # Every prediction and tolerance below uses best_disp, NOT the caller's
    # prior, which only seeded the search band and may be far from the scale
    # that won it.
    nodes = []
    for wl in AUTO_BALMER_A:
        pred = best_x0 + wl / best_disp
        if not (0 <= pred < n):
            continue
        mu = refine_centroid(spectrum, pred, polarity=-1)
        # Accept only if refinement stayed near the prediction (didn't run
        # off to a neighbour) — within ~1.5 lines' worth of pixels.
        if abs(mu - pred) <= 1.5 * (238.0 / best_disp):  # Hγ–Hδ gap as scale
            nodes.append([float(mu), float(wl)])

    if len(nodes) < 2:
        return [], {"error": f"only {len(nodes)} Balmer line(s) found"}

    # ── Optional telluric anchor, predicted from the Balmer linear fit ──
    telluric_added = False
    if include_telluric:
        px = np.array([p for p, _ in nodes])
        wls = np.array([w for _, w in nodes])
        a, b = np.polyfit(px, wls, 1)            # wl = a*px + b
        pred_t = (AUTO_TELLURIC_A - b) / a       # invert to pixel
        if 0 <= pred_t < n and _dip_strength(spectrum, pred_t, half=18) > 0.02:
            mu = refine_centroid(spectrum, pred_t, polarity=-1,
                                 fit_half_window=35)  # band is broad
            if 0 <= mu < n:
                nodes.append([float(mu), float(AUTO_TELLURIC_A)])
                telluric_added = True

    nodes.sort(key=lambda nd: nd[0])

    # ── Diagnostics: per-node residual to the linear fit ──
    px = np.array([p for p, _ in nodes])
    wls = np.array([w for _, w in nodes])
    a, b = np.polyfit(px, wls, 1)
    resid = wls - (a * px + b)
    info = {
        "x0": float(best_x0), "dispersion": float(a),
        "dispersion_scan": float(best_disp),
        "dispersion_prior": float(dispersion),
        "n_balmer": int(len(nodes) - (1 if telluric_added else 0)),
        "telluric_added": telluric_added,
        "residuals": [float(r) for r in resid],
    }
    return nodes, info

# ---------------------------------------------------------------------------
# Source geometry, FWHM, contamination detection
# ---------------------------------------------------------------------------

def rotate_band(data, angle_deg, y0, y1, cval=0.0, order=3):
    """Rows ``[y0, y1)`` of ``rotate(data, angle_deg, reshape=False, cval)``,
    without rotating the whole frame.

    Same cubic-spline resampling, evaluated only on the rows asked for — and
    only over the input rows those rows can actually reach.  Once the strip's
    position is known (a focus sweep, a livestack of one target), derotating
    the full frame to read back a few hundred rows is wasted work: a 2160x3840
    frame costs ~560 ms to rotate, a 400-row band ~50.

    Equivalence is exact, not approximate: ``scipy.ndimage.rotate`` is itself
    an ``affine_transform`` about the array centre, so this reuses its matrix
    and shifts the offset to start at row ``y0``.  The input is cropped to the rows
    the band can reach plus a margin, which is what keeps the spline prefilter
    (an IIR filter over the whole input) from costing full-frame time; the
    margin is far past the filter's decay, so the crop is not visible in the
    result.  See tests/test_rotate_band.py.
    """
    from scipy.ndimage import affine_transform

    data = np.asarray(data)
    h, w = data.shape
    y0 = int(y0)
    y1 = int(y1)
    if not (0 <= y0 < y1 <= h):
        raise ValueError(f"band rows [{y0}, {y1}) outside frame height {h}")

    a = np.deg2rad(angle_deg)
    # The matrix scipy.ndimage.rotate builds for axes (0, 1).
    rot = np.array([[np.cos(a), np.sin(a)],
                    [-np.sin(a), np.cos(a)]])
    # rotate(reshape=False) keeps the shape, so in- and out-centre coincide.
    centre = (np.array([h, w], dtype=float) - 1) / 2.0
    # affine_transform maps input = rot @ output + offset.  Full-frame rotate
    # uses offset = centre - rot @ centre; starting the output at row y0 just
    # shifts the output origin.
    offset = rot @ (np.array([float(y0), 0.0]) - centre) + centre

    # Which input rows can this band reach?  The corners of the output box
    # bound it (the map is affine, so the extremes are at the corners).
    bh = y1 - y0
    corners = ((0.0, 0.0), (0.0, w - 1.0), (bh - 1.0, 0.0), (bh - 1.0, w - 1.0))
    reach = [float((rot @ np.array(c) + offset)[0]) for c in corners]
    # Cubic spline support is 2px; the prefilter's IIR pole is ~-0.268, so 16
    # rows of margin puts its influence at 0.268**16 ~ 1e-9 — unmeasurable.
    pad = 16
    r0 = int(max(0, np.floor(min(reach)) - pad))
    r1 = int(min(h, np.ceil(max(reach)) + pad + 1))

    return affine_transform(
        data[r0:r1], rot, offset=offset - np.array([float(r0), 0.0]),
        output_shape=(bh, w), order=order, mode="constant", cval=cval)


def spectrum_fully_in_frame(xc, yc, length, angle_deg, w, h):
    """
    Test whether a horizontal dispersed spectrum starting at (xc, yc)
    with the given length fits inside the rotated working image.

    Returns
    -------
    bool — True iff every pixel of the segment is both
      (a) inside the rotated array's index bounds, and
      (b) inside the rotated valid-data rectangle (i.e. the un-rotated
          endpoint lands inside [0, w-1] × [0, h-1]).

    Sign convention.  The working image is produced by
    ``scipy.ndimage.rotate(data, angle_deg)``.  To test whether a point on
    that rotated image is backed by real data (rather than cval padding),
    the point is mapped back to original-data coordinates with
    ``t = radians(angle_deg)`` — matching scipy's rotation sense.  The sign
    is load-bearing: ``-angle_deg`` tests against the mirror-rotated
    rectangle and wrongly accepts sources whose spectrum runs into a cval
    corner triangle.
    """
    cx, cy = w / 2.0, h / 2.0
    t = np.radians(angle_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)

    def _both_conditions(x, y):
        # (a) inside array bounds
        if not (0.0 <= x <= (w - 1) and 0.0 <= y <= (h - 1)):
            return False
        # (b) inside the rotated valid-data rectangle
        dx, dy = x - cx, y - cy
        xr = cx + dx * cos_t - dy * sin_t
        yr = cy + dx * sin_t + dy * cos_t
        return 0.0 <= xr <= (w - 1) and 0.0 <= yr <= (h - 1)

    return _both_conditions(xc, yc) and _both_conditions(xc + length, yc)


def validity_boundary_line(length, angle_deg, w, h):
    """
    The right-edge validity boundary as a line on the rotated image.

    Companion to ``spectrum_fully_in_frame``: it returns the locus of
    spectrum *start* points ``(xc, yc)`` whose *tail* ``(xc + length, yc)``
    lands exactly on the original frame's right edge after the same inverse
    rotation that ``spectrum_fully_in_frame`` applies.  Start points to the
    left of this line have their tail inside the right edge (the right-edge
    condition is satisfied); points to the right have the tail spilling into
    the rotated cval-padded corner.

    This is the exact boundary of condition (b)'s right-edge term, so a line
    drawn through the two returned endpoints coincides with the accept /
    reject decision rather than approximating it.  It represents only the
    right-edge constraint — the dominant one — not the top/bottom edges
    that additionally clip the valid region near the corners.

    Derivation.  With centre ``(cx, cy)`` and ``t = radians(angle_deg)``
    (matching ``scipy.ndimage.rotate`` and ``spectrum_fully_in_frame``),
    mapping the tail ``(xc + length, yc)`` back to data coordinates and
    setting its x-coordinate ``xr = w - 1`` gives a straight line in
    ``(xc, yc)``::

        xc(yc) = cx - length
                 + ((w - 1 - cx) + (yc - cy) * sin t) / cos t

    with slope ``tan t``.  Evaluated at ``yc = 0`` and ``yc = h - 1``.

    Parameters
    ----------
    length : float
        Spectrum length in pixels (``spectrum_width``).
    angle_deg : float
        Rotation angle applied to the working image (same sign as passed to
        ``spectrum_fully_in_frame`` and ``scipy.ndimage.rotate``).
    w, h : int
        Rotated image width and height.

    Returns
    -------
    (x0, y0), (x1, y1) : tuple of float pairs
        Two endpoints of the boundary line, at ``yc = 0`` and ``yc = h - 1``.
        x-values are NOT clipped to the image, so the line may start or end
        outside [0, w-1]; the drawing layer can clip via axis limits.  If
        ``cos t`` is zero (a 90° rotation, degenerate for this pipeline) the
        line is reported vertical at ``x = w - 1 - length``.
    """
    cx, cy = w / 2.0, h / 2.0
    t = np.radians(angle_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)

    if abs(cos_t) < 1e-12:
        x = (w - 1) - length
        return (x, 0.0), (x, float(h - 1))

    def _xc(yc):
        return cx - length + ((w - 1 - cx) + (yc - cy) * sin_t) / cos_t

    return (_xc(0.0), 0.0), (_xc(h - 1), float(h - 1))


def estimate_source_fwhm(rotated_data, x, y, half_width=2, search_half=20):
    """
    Estimate the spatial FWHM of a source at (x, y) by fitting a Gaussian
    to a median cross-section near the centroid.

    Takes the median of (2*half_width + 1) adjacent columns centred on x
    (suppresses cosmic rays and column-noise), then fits

        f(y) = A * exp(-(y - y0)² / (2σ²)) + C

    within ±search_half rows of the centroid.  Returns FWHM = 2.355 σ.

    Returns NaN if the fit fails, the source is too close to an edge,
    or the resulting σ is physically implausible (too narrow / too wide).

    Parameters
    ----------
    rotated_data : 2D ndarray
        Rotated working image.
    x, y : float
        Source centroid (pixels in the rotated frame).
    half_width : int
        Median is taken across 2*half_width + 1 columns centred on x.
        Default 2 → 5 columns.
    search_half : int
        Rows above/below y included in the fit.  Default 20.

    Returns
    -------
    fwhm : float
        Spatial FWHM in pixels, or NaN on failure.
    """
    h, w = rotated_data.shape
    x0 = int(round(x))
    y0 = int(round(y))

    x_lo = max(0, x0 - half_width)
    x_hi = min(w, x0 + half_width + 1)
    y_lo = max(0, y0 - search_half)
    y_hi = min(h, y0 + search_half + 1)

    if x_hi - x_lo < 3 or y_hi - y_lo < 5:
        return float("nan")

    profile = np.median(rotated_data[y_lo:y_hi, x_lo:x_hi], axis=1).astype(float)
    rows = np.arange(y_lo, y_hi, dtype=float)

    bg0 = np.median(profile)
    amp0 = float(np.max(profile) - bg0)
    if amp0 <= 0:
        return float("nan")

    def gaussian(y, amp, mu, sigma, bg):
        return amp * np.exp(-((y - mu) ** 2) / (2.0 * sigma * sigma)) + bg

    try:
        popt, _ = curve_fit(
            gaussian, rows, profile,
            p0=[amp0, float(y0), 3.0, bg0],
            maxfev=500,
        )
    except Exception:
        return float("nan")

    sigma = abs(popt[2])
    if sigma < 0.5 or sigma > search_half:
        return float("nan")
    return float(2.3548 * sigma)


def _zero_order_profile(rotated_data, x, y, fwhm=None,
                        aperture_half=None, search_half=None):
    """Build the background-subtracted 1-D column profile of the zero-order
    blob, shared by ``measure_zero_order_x`` (position) and
    ``measure_zero_order_resolution`` (width).

    Sums ``2*aperture_half + 1`` rows centred on ``y`` over the column
    window ``[x - search_half, x + search_half]``, subtracts a robust edge
    background, and applies the same 5σ significance / interior-peak gates
    both callers rely on.  Keeping this in one place stops the two
    measurements drifting apart in how they isolate the blob.

    Parameters are identical to ``measure_zero_order_x``.

    Returns
    -------
    dict or None
        None if the window is degenerate, off-edge, or contains no
        significant interior peak.  Otherwise keys ``cols``, ``prof``,
        ``peak``, ``i_peak``, ``edge_sigma``, ``bg``.
    """
    h, w = rotated_data.shape
    if not (np.isfinite(x) and np.isfinite(y)):
        return None

    f = fwhm if (fwhm is not None and np.isfinite(fwhm) and fwhm > 0) else 4.0
    if aperture_half is None:
        aperture_half = max(2, int(round(1.5 * f)))
    if search_half is None:
        search_half = max(6, int(round(3.0 * f)))

    x0 = int(round(x))
    y0 = int(round(y))

    x_lo = max(0, x0 - search_half)
    x_hi = min(w, x0 + search_half + 1)
    y_lo = max(0, y0 - aperture_half)
    y_hi = min(h, y0 + aperture_half + 1)

    if x_hi - x_lo < 5 or y_hi - y_lo < 3:
        return None

    profile = np.sum(rotated_data[y_lo:y_hi, x_lo:x_hi].astype(float), axis=0)
    cols = np.arange(x_lo, x_hi, dtype=float)
    n = profile.size

    third = max(1, n // 4)
    edge = np.concatenate([profile[:third], profile[-third:]])
    bg = np.median(edge)
    edge_mad = np.median(np.abs(edge - bg))
    edge_sigma = 1.4826 * edge_mad if edge_mad > 0 else np.std(edge)
    prof = profile - bg

    peak = float(np.max(prof))
    if not np.isfinite(peak) or peak <= 0:
        return None
    if edge_sigma > 0 and peak < 5.0 * edge_sigma:
        return None

    i_peak = int(np.argmax(prof))
    if i_peak == 0 or i_peak == n - 1:
        return None

    # Raw 2-D core statistics (pre-sum), where per-pixel saturation is
    # actually visible — row-summing into `prof` launders mild clipping,
    # so the summed profile is a poor place to detect a flat top.  Sample
    # the raw pixels in a small box around the blob centre column.
    core_lo = max(0, i_peak - 3)
    core_hi = min(n, i_peak + 4)
    raw_core = rotated_data[y_lo:y_hi, x_lo + core_lo:x_lo + core_hi].astype(float)
    raw_max = float(np.max(raw_core)) if raw_core.size else float("nan")
    # Count raw pixels within a tight tolerance of the max: a true clip
    # produces many identical (or near-identical) pixels at the ceiling.
    if np.isfinite(raw_max) and raw_max > 0:
        tol = max(1.0, 0.002 * raw_max)
        n_at_max = int(np.sum(raw_core >= raw_max - tol))
    else:
        n_at_max = 0

    return {
        "cols": cols, "prof": prof, "peak": peak,
        "i_peak": i_peak, "edge_sigma": float(edge_sigma), "bg": float(bg),
        "raw_max": raw_max, "n_at_max": n_at_max,
    }


def measure_zero_order_x(rotated_data, x, y, fwhm=None,
                         aperture_half=None, search_half=None):
    """
    Measure the sub-pixel *image-column* position of the undispersed
    (zero-order) peak near a source at (x, y).

    This is the wavelength-scale anchor used to make a dispersion
    solution transferable between sources of different colour: the raw
    DAOStarFinder centroid carries a small, colour-dependent offset from
    the true zero-order position (the zero-order blob through the grating
    is not colour-symmetric), and anchoring the wavelength axis to that
    biased centroid shifts the whole scale by a fraction of a pixel
    between stars.  Measuring the zero-order peak the same way for every
    source removes that bias.

    The measurement is deliberately taken on the *full* rotated image —
    NOT on the already-extracted strip, whose left edge truncates the
    zero-order blob and would reintroduce the bias this removes.
    A symmetric column window of ±search_half around ``x`` keeps the whole
    blob in view.

    Algorithm
    ---------
    1. Sum ``2*aperture_half + 1`` rows centred on ``y`` into a 1D column
       profile over ``[x - search_half, x + search_half]``.
    2. Subtract a robust background (median of the profile's outer edges).
    3. Isolate the central blob: keep the contiguous run of samples above
       half the peak excursion that contains the window-centre maximum.
       This drops any dispersed continuum creeping in from the red side,
       which would otherwise pull a plain centroid redward.
    4. Return the intensity-weighted centroid of that isolated run, in
       absolute image-column coordinates.

    A centroid (not a Gaussian fit) is used on purpose: the zero-order
    image of a bright A0V calibrator is frequently flat-topped or
    saturated, where a Gaussian fit is unstable but a windowed centroid
    is well-behaved.

    Parameters
    ----------
    rotated_data : 2D ndarray
        The rotated working image (same array passed to extract_spectrum).
    x, y : float
        Approximate source centroid (the DAO centroid is fine).
    fwhm : float or None
        Spatial FWHM in pixels, used to size the windows when the explicit
        ``aperture_half`` / ``search_half`` are not given.  Falls back to
        4.0 px when None or non-finite.
    aperture_half : int or None
        Rows above/below ``y`` summed into the profile.  Defaults to
        ``round(1.5 * fwhm)`` (clamped ≥ 2).
    search_half : int or None
        Columns each side of ``x`` included in the window.  Defaults to
        ``round(3.0 * fwhm)`` (clamped ≥ 6) — wide enough to contain the
        full blob plus a little background for the edge-median, narrow
        enough not to reach the dispersed continuum.

    Returns
    -------
    float
        Sub-pixel image-column of the zero-order peak, or NaN if no clear
        peak is present (too faint, off-edge, or degenerate window).
    """
    info = _zero_order_profile(rotated_data, x, y, fwhm,
                               aperture_half, search_half)
    if info is None:
        return float("nan")

    cols = info["cols"]
    prof = info["prof"]
    peak = info["peak"]
    i_peak = info["i_peak"]
    n = prof.size

    # Isolate the contiguous above-half-max run containing the peak.
    thresh = 0.5 * peak
    lo_i = i_peak
    while lo_i > 0 and prof[lo_i - 1] >= thresh:
        lo_i -= 1
    hi_i = i_peak
    while hi_i < n - 1 and prof[hi_i + 1] >= thresh:
        hi_i += 1

    seg_w = prof[lo_i:hi_i + 1]
    seg_x = cols[lo_i:hi_i + 1]
    wsum = float(np.sum(seg_w))
    if wsum <= 0:
        return float("nan")
    return float(np.sum(seg_x * seg_w) / wsum)


def measure_zero_order_resolution(rotated_data, x, y, dispersion,
                                  ref_wl=6563.0, fwhm=None,
                                  aperture_half=None, search_half=None,
                                  flat_top_frac=0.90):
    """Measure the achieved spectral resolution from the zero-order blob.
    Not yet wired into the GUI; intended for the 305 mm f/4 characterisation
    work, where the zero-order LSF gives an upper bound on achievable R before
    committing exposure time to faint first-order targets.

    The zero order is the undispersed image of the source, so its width in
    the dispersion direction (image columns, after derotation) is a direct
    measurement of the line-spread function — the kernel that every
    first-order spectral feature is convolved with.  Its FWHM is therefore
    the resolution element Δλ.  This is a *lower bound* on Δλ (an optimistic
    R): wavelength-dependent aberrations such as grating-induced coma can
    broaden the dispersed orders more than the zero order, so on a fast
    (f/4) system the true first-order Δλ may be larger.  Label the result
    "zero-order-limited resolution" accordingly.

    Two FWHM estimators are returned because the zero order of a bright
    calibrator is often saturated (flat-topped):

    * ``fwhm_px_interp`` — half-maximum crossings by linear interpolation.
      Robust and always available, but a flat (clipped) top inflates it.
    * ``fwhm_px_gauss``  — Gaussian fit; when a flat top is detected the
      saturated core is excluded and only the wings are fit, recovering the
      true underlying width even when the core clips.  NaN if the fit fails.

    A ``saturated`` flag reports whether a flat top was detected.

    Wavelength conversion is linear: ``Δλ = fwhm_px * dispersion`` and
    ``R = ref_wl / Δλ``.  ``ref_wl`` names where R is quoted (R rises toward
    the red for roughly constant Δλ in Å); default Hα.

    Parameters
    ----------
    rotated_data : 2D ndarray
        Rotated working image (dispersion along +x).
    x, y : float
        Approximate source centroid (DAO centroid is fine).
    dispersion : float
        Linear dispersion in Å per pixel.
    ref_wl : float
        Wavelength (Å) at which R is reported.  Default 6563 (Hα).
    fwhm, aperture_half, search_half :
        Window sizing, passed through to ``_zero_order_profile``.
    flat_top_frac : float
        Samples at/above this fraction of peak form the (possibly
        saturated) core, excluded from the Gaussian wing fit when a flat
        top spanning >= 3 samples is found.

    Returns
    -------
    dict with keys ``fwhm_px_interp``, ``fwhm_A_interp``, ``R_interp``,
    ``fwhm_px_gauss``, ``fwhm_A_gauss``, ``R_gauss``, ``saturated``,
    ``ref_wl``, ``peak``, ``n_core``.  Width/R fields are NaN if the
    profile is unusable; the dict is always returned (never None).
    """
    nan = float("nan")
    out = {
        "fwhm_px_interp": nan, "fwhm_A_interp": nan, "R_interp": nan,
        "fwhm_px_gauss": nan, "fwhm_A_gauss": nan, "R_gauss": nan,
        "saturated": False, "ref_wl": float(ref_wl), "peak": nan, "n_core": 0,
    }

    info = _zero_order_profile(rotated_data, x, y, fwhm,
                               aperture_half, search_half)
    if info is None:
        return out

    cols = info["cols"]
    prof = info["prof"]
    peak = info["peak"]
    i_peak = info["i_peak"]
    n = prof.size
    out["peak"] = peak

    # ── Estimator 1: half-max crossings by linear interpolation ──
    half = 0.5 * peak
    xl = nan
    for i in range(i_peak, 0, -1):
        if prof[i] >= half > prof[i - 1]:
            frac = (half - prof[i - 1]) / (prof[i] - prof[i - 1])
            xl = cols[i - 1] + frac * (cols[i] - cols[i - 1])
            break
    xr = nan
    for i in range(i_peak, n - 1):
        if prof[i] >= half > prof[i + 1]:
            frac = (prof[i] - half) / (prof[i] - prof[i + 1])
            xr = cols[i] + frac * (cols[i + 1] - cols[i])
            break
    if np.isfinite(xl) and np.isfinite(xr) and xr > xl:
        fwhm_px_i = float(xr - xl)
        out["fwhm_px_interp"] = fwhm_px_i
        out["fwhm_A_interp"] = fwhm_px_i * dispersion
        if fwhm_px_i * dispersion > 0:
            out["R_interp"] = ref_wl / (fwhm_px_i * dispersion)

    # ── Flat-top (saturation) detection ──
    # Detected on the RAW 2-D core pixels (see _zero_order_profile): a clip
    # produces many pixels pinned at the same ceiling value.  Row-summing
    # into `prof` launders mild per-pixel saturation, so the summed profile
    # can't see it — but the wing-only Gaussian fit below still benefits
    # from excluding the (flattened) summed core when a clip is present.
    n_at_max = info.get("n_at_max", 0)
    flat_top = n_at_max >= 4          # several raw pixels at the ceiling
    out["saturated"] = bool(flat_top)
    # n_core: summed-profile samples in the top decile (for the wing fit).
    core_thresh = flat_top_frac * peak
    n_core = int(np.sum(prof >= core_thresh))
    out["n_core"] = n_core

    # ── Estimator 2: Gaussian fit (wings-only if flat-topped) ──
    if flat_top:
        fit_mask = prof < core_thresh
    else:
        fit_mask = np.ones(n, dtype=bool)
    fit_mask &= prof > 0.1 * peak
    if int(np.sum(fit_mask)) >= 4:
        xfit = cols[fit_mask]
        yfit = prof[fit_mask]

        def _g(xx, amp, mu, sigma):
            return amp * np.exp(-((xx - mu) ** 2) / (2.0 * sigma * sigma))

        sigma0 = (out["fwhm_px_interp"] / 2.3548
                  if np.isfinite(out["fwhm_px_interp"]) else 3.0)
        try:
            popt, _ = curve_fit(
                _g, xfit, yfit,
                p0=[peak, cols[i_peak], max(0.8, sigma0)],
                maxfev=2000,
            )
            sigma = abs(popt[2])
            if 0.3 < sigma < (n / 2.0):
                fwhm_px_g = float(2.3548 * sigma)
                out["fwhm_px_gauss"] = fwhm_px_g
                out["fwhm_A_gauss"] = fwhm_px_g * dispersion
                if fwhm_px_g * dispersion > 0:
                    out["R_gauss"] = ref_wl / (fwhm_px_g * dispersion)
        except Exception:
            pass

    return out


def contaminators_from_sources(sources_xy, bbox, fwhm, target_y,
                               trace_exclude_fwhm=0.2, y_pad_fwhm=1.0):
    """
    Find which already-detected sources contaminate an extraction strip.

    Takes the frame-wide DAOStarFinder list — every source, not just the
    ``nsources`` brightest — and keeps those whose PSF reaches into the
    aperture.  Returns their column indices relative to the strip.

    Detection is frame-wide rather than inside the strip because
    DAOStarFinder needs a whole PSF to find a star, and a star only partly
    inside the band does not present one there.  The stars nearest the
    aperture edge — half in, quietly adding flux to the science spectrum —
    are precisely the ones a band-local detection cannot see.  The
    frame-wide pass has already found them with their full PSF and against
    the frame's own noise; this only asks which of them land in the band.
    It also means the target's own continuum and its trailing knee cannot
    register as fake stars, since the detection never looks at the strip.

    Emission lines are the one trap: a sharp WR/nova peak is a compact,
    PSF-shaped blob on the frame and the frame-wide pass detects it as a
    "star".  It is the target's own light, so it sits on the trace, whereas
    a real contaminator is a different star offset in y.  On real frames an
    emission line lands within a fraction of a pixel of the trace (~0.06
    FWHM for a nova Hα peak) while the nearest real contaminator is ~0.6
    FWHM out, so the exclusion zone must stay tight.  The two offsets do
    not scale alike: the line's is a centroiding error that shrinks with the
    PSF, a star's is fixed in pixels, so a zone stated in FWHM widens
    wherever the FWHM is overestimated — which is where DAOStarFinder errs.
    The same test drops the target's own zero order, which sits at the
    strip's left edge on the trace.

    This is the one call that cannot be made from geometry: a star sitting
    on the trace and an emission line are the same picture.  Hence the
    second return value — the caller reports what was dropped rather than
    deciding silently.  Separating the two requires an external catalogue.

    Parameters
    ----------
    sources_xy : (N, 2) array-like
        (x, y) centroids of every source detected on the SAME image the
        strip was cut from (the rotated frame).  Empty is fine.
    bbox : dict
        From ``extract_spectrum``: x_start/x_end/y_start/y_end.
    fwhm : float
        Spatial FWHM in pixels — the optical system's, not the source's.
    target_y : float
        The target's own row, for the on-trace test.  Pass the y given to
        ``extract_spectrum``, NOT the bbox centre: at a frame edge the bbox
        is clipped and its centre is no longer the target.
    trace_exclude_fwhm : float
        Half-width, in FWHM, of the on-trace zone treated as the target's
        own light.  0 disables the rejection.
    y_pad_fwhm : float
        How far outside the aperture a source still counts, in FWHM.  A star
        one FWHM above the band still spills its wings into it, hence the
        default of 1.0.

    Returns
    -------
    cols : 1D ndarray of float
        Strip-relative column indices of the contaminators.  Empty if none.
    on_trace : (M, 2) ndarray
        The sources dropped by the on-trace test, as (strip column, rows off
        the trace).  These are the judgement calls — an emission line and a
        star that happens to sit on the trace look identical from here — so
        the caller can report them instead of silently discarding them.
    """
    empty = (np.array([], dtype=float), np.empty((0, 2), dtype=float))
    x0 = int(bbox["x_start"])
    x1 = int(bbox["x_end"])
    y0 = int(bbox["y_start"])
    y1 = int(bbox["y_end"])
    pts = np.asarray(sources_xy, dtype=float)
    if pts.size == 0 or not np.isfinite(fwhm) or fwhm <= 0:
        return empty
    pts = np.atleast_2d(pts)
    if pts.shape[1] != 2:
        return empty

    pad = float(y_pad_fwhm) * float(fwhm)
    excl = float(trace_exclude_fwhm) * float(fwhm)
    xc, yc = pts[:, 0], pts[:, 1]

    # y_end is exclusive (extract_spectrum slices with it), so the last row
    # inside the aperture is y_end - 1.
    in_cols = (xc >= x0) & (xc < x1)
    in_rows = (yc >= y0 - pad) & (yc <= (y1 - 1) + pad)
    # Strictly less-than, so trace_exclude_fwhm=0 really does disable the
    # rejection.  With <=, a source landing exactly on the trace would
    # still be dropped at zero width, and callers rely on "0 disables".
    dy = yc - float(target_y)
    own_light = np.abs(dy) < excl
    in_band = in_cols & in_rows
    keep = in_band & ~own_light
    dropped = in_band & own_light
    return (xc[keep] - x0,
            np.column_stack([xc[dropped] - x0, dy[dropped]]))


# ---------------------------------------------------------------------------
# Spectrum extraction
# ---------------------------------------------------------------------------

def _reconcile_col_flag(flag, n_col):
    """Reconcile an optional per-column boolean flag to length ``n_col``.

    Returns a bool array of length ``n_col`` (longer input truncated, shorter
    padded with False), or None when ``flag`` is None/empty.  Mirrors the
    truncate/pad handling used for the in-aperture contaminator mask so a ±1 px
    strip-origin shift between tracked frames never raises.
    """
    if flag is None:
        return None
    flag = np.asarray(flag, dtype=bool)
    if flag.size == 0:
        return None
    if flag.size == n_col:
        return flag
    out = np.zeros(n_col, dtype=bool)
    k = min(flag.size, n_col)
    out[:k] = flag[:k]
    return out


def extract_spectrum(x, y, rotated_data, spectrum_width,
                     aperture_half_height, sky_band_gap, sky_band_width,
                     sky_col_flag=None, sky_gap_hi=None, sky_width_hi=None):
    """
    Extract a background-subtracted 1D spectrum for a source at (x, y).

    Parameters
    ----------
    sky_col_flag : 1D bool ndarray, optional
        Per-column "this column's sky band is untrustworthy" flag, frozen from
        a high-SNR reference (the stacked image) and reused across frames.
        When provided, the per-frame sky level is still computed normally from
        each frame's own sky pixels (so the background stays adaptive to that
        frame's conditions), but for flagged columns the frame's own sky
        estimate is *discarded* and ``sky_per_col`` is linearly interpolated
        from the nearest unflagged columns instead.  This avoids re-deciding
        sky-pixel cleanliness on low-SNR individual frames, where the
        sigma-clip is unreliable.  Length must match the strip column count;
        a mismatched length is reconciled by truncate/pad (the same ±1 px
        strip-origin jitter handling used elsewhere) and a None or all-False
        flag is a no-op.

    Returns
    -------
    region       : 2D ndarray  – raw pixel cutout
    sky_lo, sky_hi : 2D ndarray – the two sky-band cutouts
    mask_lo, mask_hi : 2D bool  – sigma-clip rejection masks per sky band
    column_sums  : 1D ndarray  – background-subtracted collapsed spectrum
    variance     : 1D ndarray  – per-column measurement variance (ADU²) of
                                 ``column_sums``, background term only:

                                     var = σ²_sky · (N_ap + N_ap² / N_sky)

                                 where σ²_sky is the robust per-column sky
                                 variance (std of the sigma-clipped sky
                                 pixels), N_ap the number of aperture rows
                                 summed, and N_sky the surviving sky-pixel
                                 count in that column.  The first term is the
                                 background scatter propagated through the
                                 aperture sum; the second is the uncertainty
                                 of the subtracted per-column sky estimate,
                                 amplified by subtracting it from every
                                 aperture row.  The source Poisson term
                                 (S / gain) is deliberately omitted until a
                                 gain (e⁻/ADU) is available — it is purely
                                 additive and slots in here later without
                                 reworking this term.  Columns with no usable
                                 sky get NaN variance.
    bbox         : dict        – all bounding-box coordinates
    """
    h, w = rotated_data.shape

    x_start = int(max(x, 0))
    x_end = int(min(x + spectrum_width, w))
    y_start = int(max(y - aperture_half_height, 0))
    # +1 so the aperture spans 2*aperture_half_height + 1 rows centred on
    # y, matching the symmetric "±N" convention the UI advertises and the
    # convention used by _zero_order_profile / estimate_source_fwhm.
    # Without it the upper +half row is excluded, biasing the aperture
    # half a row low relative to the centroid.
    y_end = int(min(y + aperture_half_height + 1, h))

    # The low band uses sky_band_gap/sky_band_width; the high band reuses them
    # unless sky_gap_hi/sky_width_hi override — so a caller that passes only the
    # first pair gets the symmetric bands it always did, and the GUI can offset
    # the two bands independently to dodge a contaminator in a crowded field.
    gap_hi = sky_band_gap if sky_gap_hi is None else sky_gap_hi
    width_hi = sky_band_width if sky_width_hi is None else sky_width_hi
    sky_lo_end = int(max(y_start - sky_band_gap, 0))
    sky_lo_start = int(max(sky_lo_end - sky_band_width, 0))
    sky_hi_start = int(min(y_end + gap_hi, h))
    sky_hi_end = int(min(sky_hi_start + width_hi, h))

    region = rotated_data[y_start:y_end, x_start:x_end]
    sky_lo = rotated_data[sky_lo_start:sky_lo_end, x_start:x_end]
    sky_hi = rotated_data[sky_hi_start:sky_hi_end, x_start:x_end]

    if sky_lo.shape[0] > 0 and sky_hi.shape[0] > 0:
        sky = np.vstack([sky_lo, sky_hi])
    elif sky_lo.shape[0] > 0:
        sky = sky_lo
    elif sky_hi.shape[0] > 0:
        sky = sky_hi
    else:
        sky = None

    n_lo = sky_lo.shape[0]   # rows in lower sky band
    n_hi = sky_hi.shape[0]   # rows in upper sky band

    n_ap = region.shape[0]   # aperture rows actually summed
    n_col = x_end - x_start

    if sky is not None:
        # Sigma-clipped per-column background estimate.
        # astropy.stats.sigma_clip iterates correctly: rejected pixels stay
        # rejected, and the per-column median/std are computed on the
        # surviving (non-masked) pixels at every iteration.
        # Sky contamination is one-sided: stars, hot pixels and cosmic rays
        # are all *bright*. So clip tight on the high side (kill the junk)
        # and loose on the low side (faint pixels are real background worth
        # keeping). A symmetric clip wastes half its rejection on the wrong tail.
        SKY_SIGMA_HI = 2.0
        SKY_SIGMA_LO = 4.0
        SKY_ITERS = 5
        sky_f = sky.astype(float)
        clipped = sigma_clip(
            sky_f, sigma_upper=SKY_SIGMA_HI, sigma_lower=SKY_SIGMA_LO,
            maxiters=SKY_ITERS,
            cenfunc="median", stdfunc="std", axis=0, masked=True,
        )
        mask = np.ma.getmaskarray(clipped)   # True = rejected
        # Per-column background: median of surviving pixels.
        # If a column has every pixel rejected (rare), fall back to the
        # un-clipped column median, so the result is never NaN.
        sky_per_col = np.ma.median(clipped, axis=0).filled(
            np.median(sky_f, axis=0))
        sky_per_col = np.asarray(sky_per_col, dtype=float)
        # Split combined mask back into per-band masks.
        # sky was built as vstack([sky_lo, sky_hi]) so lo rows come first.
        mask_lo = mask[:n_lo, :] if n_lo > 0 else np.zeros((0, sky_lo.shape[1]), dtype=bool)
        mask_hi = mask[n_lo:, :] if n_hi > 0 else np.zeros((0, sky_hi.shape[1]), dtype=bool)

        # ── Frozen sky-column flag (reused from a high-SNR reference) ──
        # For columns the reference flagged as untrustworthy, discard this
        # frame's own per-column sky estimate and interpolate it from the
        # nearest unflagged columns.  The overall background level still comes
        # from this frame's clean columns (conditions-adaptive); only the
        # flagged columns borrow a neighbour-derived background.  No-op when
        # the flag is None / all-False / leaves no clean anchors.
        sky_flag = _reconcile_col_flag(sky_col_flag, n_col)
        if sky_flag is not None and sky_flag.any() and not sky_flag.all():
            cols = np.arange(n_col)
            good = ~sky_flag
            sky_per_col[sky_flag] = np.interp(
                cols[sky_flag], cols[good], sky_per_col[good])

        # ── Per-column measurement variance (background term) ──────────
        # Robust per-column sky sigma: std of the sigma-clipped (surviving)
        # sky pixels — consistent with the clipped median taken above.  The
        # clipping has already removed outliers, so the clipped std is an
        # appropriate robust scale here.  N_sky is the surviving pixel count
        # per column.
        sky_sigma = np.ma.std(clipped, axis=0)                 # masked array
        n_sky = (~mask).sum(axis=0).astype(float)              # per-column
        sky_var = np.asarray(
            np.ma.filled(sky_sigma, np.nan), dtype=float) ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            # var = σ²_sky · (N_ap + N_ap² / N_sky)
            variance = sky_var * (n_ap + (n_ap ** 2) / n_sky)
        # Columns with no surviving sky pixel → undefined error → NaN.
        variance[n_sky <= 0] = np.nan
    else:
        sky_per_col = np.zeros(n_col)
        mask_lo = np.zeros((0, n_col), dtype=bool)
        mask_hi = np.zeros((0, n_col), dtype=bool)
        # No sky → no defensible background error.
        variance = np.full(n_col, np.nan)

    column_sums = np.sum(region - sky_per_col[np.newaxis, :], axis=0)

    bbox = dict(
        x_start=x_start, x_end=x_end,
        y_start=y_start, y_end=y_end,
        sky_lo_start=sky_lo_start, sky_lo_end=sky_lo_end,
        sky_hi_start=sky_hi_start, sky_hi_end=sky_hi_end,
    )
    return (region, sky_lo, sky_hi, mask_lo, mask_hi,
            column_sums, variance, bbox)


def best_y_shift(x, y, rotated_data, spectrum_width, aperture_half_height,
                 sky_band_gap, sky_band_width, cols=None, max_shift=5,
                 sky_gap_hi=None, sky_width_hi=None, min_snr=5.0):
    """
    Find the vertical shift whose extraction captures the most spectrum.

    Slides the aperture over ``±max_shift`` rows, extracts at each, and
    returns the shift with the largest background-subtracted flux.  A
    correctly placed aperture encloses the whole trace; a misplaced one
    spills part of it into the sky bands and sums to less.  There is no PSF
    model, noise estimate or centroid: the score is the quantity being
    optimised (the extracted spectrum) measured by the code that produces
    it, so it cannot disagree with what Run will do.

    It is needed because DAOStarFinder centroids the zero-order blob, whose
    core saturates on bright targets, and so misplaces the aperture in y.
    The dispersed trace does not saturate and is the feature that must be
    centred.

    Parameters
    ----------
    x, y : float
        Source position; ``y`` is the current (possibly wrong) aperture
        centre that the returned shift corrects.
    cols : mask / slice / index array, optional
        Columns to score over, restricted to the DISPERSED part of the
        strip.  ``extract_spectrum`` starts its cutout AT the source, so
        column 0 is the zero-order blob: scoring it would just re-centre the
        aperture on the saturated thing DAO already mis-centred on.  None
        scores every column and is almost never what you want.
    max_shift : int
        Rows searched either side.  Default 5 → 11 extractions.
    min_snr : float
        The winning shift's flux must exceed this many σ of its own
        background uncertainty.  Without it a sky-only strip still has a
        best shift — the largest noise fluctuation — and the aperture gets
        moved onto nothing.

    Returns
    -------
    shift : int
        Rows to ADD to ``y``.  0 when the current placement is already best,
        when no shift produces usable flux, or when there is no significant
        spectrum to centre on.
    scores : dict
        ``{shift: score}`` for every shift tried — the caller can show the
        curve and see whether the peak is real or the scan hit its edge.
    """
    scores = {}
    sigmas = {}
    for d in range(-int(max_shift), int(max_shift) + 1):
        try:
            *_, column_sums, variance, _bbox = extract_spectrum(
                x, y + d, rotated_data, spectrum_width, aperture_half_height,
                sky_band_gap, sky_band_width,
                sky_gap_hi=sky_gap_hi, sky_width_hi=sky_width_hi)
        except Exception:
            # A shift that runs the aperture or a sky band off the frame is
            # simply not a candidate; keep scanning the rest.
            continue
        sel = column_sums if cols is None else column_sums[cols]
        if sel.size == 0:
            continue
        total = float(np.nansum(sel))
        if not np.isfinite(total):
            continue
        scores[d] = total
        var = variance if cols is None else variance[cols]
        sigmas[d] = float(np.sqrt(np.nansum(var)))
    if not scores:
        return 0, {}
    best = max(scores, key=scores.get)
    # Nothing to centre on → stay put.  extract_spectrum already measures the
    # background uncertainty of these very sums, so the test costs nothing:
    # on sky alone the flux is consistent with zero however the aperture is
    # placed, and the "best" shift is just the luckiest noise draw.
    sigma = sigmas.get(best, 0.0)
    if scores[best] <= 0 or (sigma > 0 and scores[best] < min_snr * sigma):
        return 0, scores
    return best, scores


# ---------------------------------------------------------------------------
# Reference-line drawing
# ---------------------------------------------------------------------------


# Height at which labels are anchored, as a multiple of y_max.
LABEL_LEVEL = 1.01

# How many rows of labels may stack before a line gives up its label and
# keeps only its rule.  Each row costs a full label height of headroom —
# raising by less does not help, since a rotated label is as tall as its
# string is long and neighbours would still cross.
LABEL_MAX_LEVELS = 3

# How much of a label's width a neighbour may cover before the neighbour
# is dropped, as a fraction of the label.  0 refuses any contact and
# silently hides half the lines wherever they cluster; 1 is the old
# free-for-all where crowded labels printed into illegibility.  Partial
# overlap stays readable — the glyph columns interleave rather than
# collide — and keeps far more of the plot labelled.
LABEL_OVERLAP_FRAC = 0.5

# Breathing space either side of a label, in display pixels.  Labels are
# measured, not estimated, so this is only the gap between neighbours.
LABEL_PAD_PX = 2.0

# Fallback width of a rotated label, as a multiple of the font size in
# points, used only when the canvas cannot produce a renderer to measure
# with.  Deliberately generous: over-reserving drops a label, which is
# recoverable, while under-reserving prints two on top of each other.
LABEL_WIDTH_PTS = 1.3


def plot_reference_lines(ax, lines, dispersion, y_max,
                         poly_coeffs=None, n_pixels=None,
                         colour="#ff6060", fontsize=5, linestyle="--",
                         occupied=None):
    """
    Draw vertical reference lines on a pixel-axis spectrum plot.

    Parameters
    ----------
    ax : matplotlib axis
    lines : dict {wavelength_A: label_str}
    dispersion : float
        Linear Å/pixel — used when ``poly_coeffs`` is None.
    y_max : float
        Top of the spectrum y-range; labels are placed just above it.
    poly_coeffs : 1D array or None
        Polynomial coefficients (highest degree first) for non-linear
        dispersion.  When provided, the pixel position is found by
        numerical inversion.
    n_pixels : int or None
        Total number of pixels in the spectrum (sets the inversion
        search range).  Optional; falls back to the linear estimate if
        omitted.
    colour : str
        Colour for both lines and labels.
    linestyle : str
        Rule style.  Dashed for the catalogue groups; callers drawing
        user-picked annotation lines pass a solid style so the two
        remain distinguishable on the same axes.
    occupied : list or None
        x-positions of labels already placed on this axis.  Pass one list
        through every call that draws on the same axes and label spacing
        spans all of them — otherwise each group starts from an empty
        axis and two groups happily print over each other.  Appended to
        in place; None keeps the placement local to this call.

    Read the current axis x-range so out-of-view lines are dropped
    entirely (rather than relying on clip_on, which still leaves the
    label half-rendered at the edge if its pixel position is just
    outside the visible range).  Tolerate a small inset so a line
    sitting exactly on the axis edge — where its label would spill
    off — is also suppressed.

    """
    xlo, xhi = ax.get_xlim()
    if xhi < xlo:    # axis may be inverted; normalise
        xlo, xhi = xhi, xlo
    inset = 0.005 * (xhi - xlo)    # 0.5% of the visible width
    xlo += inset
    xhi -= inset

    # Resolve positions before drawing anything: the stagger needs the
    # lines in x-order, and the headroom depends on how high it climbs.
    placed = []
    for wl, label in lines.items():
        if poly_coeffs is not None:
            xpix = invert_poly_to_pixel(wl, dispersion, poly_coeffs, n_pixels)
        else:
            xpix = wl / dispersion
        if xpix is None or not (xlo <= xpix <= xhi):
            continue
        placed.append((xpix, label))
    if not placed:
        return
    placed.sort()

    # Every line in view gets its rule, whether or not it ends up labelled.
    for xpix, _label in placed:
        # ymax=0.8 stops the rule short of the label text above it.
        ax.axvline(x=xpix, color=colour, linestyle=linestyle,
                   linewidth=0.8, alpha=0.5, ymax=0.8)

    # Interrupt the rule where the label crosses it: an opaque pad in the
    # axes' own colour breaks the line across the text and lets it resume
    # above, so a label is never read through its own rule.
    face = ax.get_facecolor()
    if face[3] == 0:              # transparent axes — the canvas shows through
        face = ax.figure.get_facecolor()
    label_bbox = dict(facecolor=(*face[:3], 1.0), edgecolor="none", pad=1.5)

    # Pack the labels as boxes, measured rather than estimated: an
    # estimate has to be generous to be safe, and every point it
    # over-reserves is a label needlessly dropped in a crowded stretch.
    # A label that collides with one already placed is raised a full row
    # — one label height, since anything less leaves the glyph columns
    # crossing — and only gives up its label when every row is taken.
    #
    # Boxes are display pixels, with y measured from the common anchor so
    # the numbers survive the y-limit changes below and stay comparable
    # across calls.  The list is shared by every call drawing on this
    # axis, which is what keeps two separately drawn groups off each other.
    renderer = getattr(ax.figure.canvas, "get_renderer", None)
    renderer = renderer() if renderer is not None else None
    scale = ax.figure.dpi / 72.0
    ax_h_px = (ax.get_position().height * ax.figure.get_figheight()
               * ax.figure.dpi)
    if occupied is None:
        occupied = []

    kept = []          # (text artist, vertical offset as a fraction of the axes)
    top_frac = 0.0     # highest point any label reaches, same units
    for xpix, label in placed:
        txt = ax.text(xpix, y_max * LABEL_LEVEL, label, rotation=90,
                      va="bottom", ha="center", color=colour,
                      fontsize=fontsize, clip_on=True, bbox=label_bbox)
        if renderer is not None:
            bb = txt.get_window_extent(renderer)
            w_px, h_px = bb.width, bb.height
        else:
            w_px = LABEL_WIDTH_PTS * fontsize * scale
            h_px = 0.62 * fontsize * len(label) * scale
        centre = ax.transData.transform((xpix, 0.0))[0]
        # Shrink the claim by the tolerated overlap, so neighbours may
        # interleave that far before one of them gives up its label.
        half = (w_px / 2.0 + LABEL_PAD_PX) * (1.0 - LABEL_OVERLAP_FRAC)
        lo_x, hi_x = centre - half, centre + half

        row = h_px + LABEL_PAD_PX
        for level in range(LABEL_MAX_LEVELS):
            y0, y1 = level * row, level * row + h_px
            if not any(lo_x < ohi and olo < hi_x and y0 < oy1 and oy0 < y1
                       for olo, ohi, oy0, oy1 in occupied):
                break
        else:
            txt.remove()          # every row taken: the rule stands alone
            continue
        occupied.append((lo_x, hi_x, y0, y1))
        if ax_h_px > 0:
            kept.append((txt, y0 / ax_h_px))
            top_frac = max(top_frac, y1 / ax_h_px)

    # Open enough headroom for the tallest label on the highest row, then
    # move each label onto its row.  Solving anchor + frac*(top - ylo) <=
    # top for top gives the division; the cap keeps the spectrum itself
    # from being squeezed away when a stretch is hopelessly crowded.
    if kept:
        ylo, ycur = ax.get_ylim()
        need = ylo + (LABEL_LEVEL * y_max - ylo) / (1.0 - min(0.7, top_frac))
        if ycur < need:
            ax.set_ylim(ylo, need)
        ylo, ytop = ax.get_ylim()
        for txt, off_frac in kept:
            if off_frac:
                x, _y = txt.get_position()
                txt.set_position(
                    (x, y_max * LABEL_LEVEL + (ytop - ylo) * off_frac))

def read_fits_image(path):
    """
    Open a FITS file and return ``(raw_data, header)`` for the first HDU
    that holds 2-D or 3-D image data, byteswapped to native order.

    Robust to files whose primary HDU is empty: Rice-compressed FITS
    (``.fits.fz``) and some stacker outputs put the image in extension 1,
    so a plain ``hdu[0].data`` would be ``None`` and the caller would die
    with an unhelpful ``'NoneType' has no attribute dtype``.  This scans
    the HDU list and returns the first image-bearing HDU instead.

    Parameters
    ----------
    path : str
        Filesystem path to the FITS file.

    Returns
    -------
    raw : ndarray
        The image data (2-D or 3-D), in native byte order.  NOT collapsed
        to mono — call ``to_mono`` if a single-channel image is needed.
    header : astropy.io.fits.Header
        A copy of the matching HDU's header (survives file-handle close),
        for WCS / plate-solve hints.

    Raises
    ------
    ValueError
        If no HDU in the file contains 2-D or 3-D image data.
    FileNotFoundError
        Propagated from ``fits.open`` when the path does not exist.
    """
    with fits.open(path) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is not None and data.ndim in (2, 3):
                header = hdu.header.copy()
                if data.dtype.byteorder not in ("=", "|", native_byteorder()):
                    data = data.byteswap().view(data.dtype.newbyteorder())
                else:
                    # Own the pixels before the HDUList closes: astropy
                    # memory-maps by default, and a returned memmap keeps
                    # the FITS open past close() — on Windows that locks
                    # the file against the capture/sync software that
                    # wrote it.  (The byteswap branch above already
                    # produced an owned copy.)
                    data = np.array(data)
                return data, header
    raise ValueError(
        f"No 2-D or 3-D image data found in any HDU of {path}")

def to_mono(data):
    """
    Collapse a FITS array to a 2D mono image.

    Handles:
      - 2D arrays              → returned as-is
      - 3D (channels, H, W)   → mean across axis 0
      - 3D (H, W, channels)   → mean across axis 2
    """
    if data.ndim == 2:
        return data.astype(float)
    if data.ndim == 3:
        # Determine which axis is the channel axis (the smallest one)
        if data.shape[0] <= 4:
            return data.mean(axis=0).astype(float)
        if data.shape[2] <= 4:
            return data.mean(axis=2).astype(float)
        # Fall back: collapse axis 0
        return data.mean(axis=0).astype(float)
    raise ValueError(f"Unexpected data shape: {data.shape}")
