"""
rotation.py
===========
Spectrum tilt (derotation) angle detection.

The pipeline rotates the working frame so the dispersed spectra run
horizontally before extraction.  Finding that tilt angle is a two-pass
process:

    Pass 1 — whole-frame estimate.  A coarse tilt of the whole image.
        Several interchangeable methods exist, all sharing the signature
        ``(data) -> (angle_deg, support)`` and collected in the
        ``FIRST_PASS_METHODS`` registry near the top of this module:

            FIRST_PASS_METHODS = [Hough, FFT, Coherence]

        Assign ``DEFAULT_FIRST_PASS`` (or pass ``first_pass=`` to
        ``estimate_angle``) to pick one.  ``Hough`` (statistical Hough
        transform on stretched, Canny-edged data, median ± MAD over line
        slopes) was designed to catch satellite trails in long exposures
        and is sensitive to noise and to frames with few stars / trails.
        ``FFT`` reads the tilt from the orientation of the image's Fourier
        power and is the candidate for noisy / sparse frames.
        ``Coherence`` uses a structure-tensor coherence-weighted orientation
        histogram, designed to respond to the linear dispersed traces while
        ignoring round PSFs and noise.

        ``angle_deg`` is the signed tilt to feed straight into the rotation
        field, following the program convention: a spectrum tilted by +5.3°
        is corrected by returning −5.3°.  ``support`` is a method-specific
        confidence/diagnostic scalar (Hough: number of lines; FFT: peak
        sharpness) reported in the log, not otherwise interpreted.

    Pass 2 — strip refinement.  ``refine_angle_from_strip`` builds a
        magnitude-weighted gradient-orientation histogram from the brightest
        extracted aperture strip (after a pass-1 rotation) and reads off the
        small residual tilt.

``estimate_angle`` ties the two together and is the single entry point the
GUI calls.  Everything here is pure: it takes arrays / a params dict and
returns numbers plus a diagnostics dict.  No tkinter, no logging, no
matplotlib — the GUI layer is responsible for reporting the diagnostics and
applying the result.  This keeps the angle-detection logic testable on a
folder of FITS files with no display.

Dependencies flow one way: ``rotation`` imports the pure reduction helpers
it needs from ``spectrum_core`` (``extract_spectrum``,
``spectrum_fully_in_frame``, ``_dao_xy``).  ``spectrum_core`` does not
import ``rotation``.
"""

import numpy as np
from scipy.ndimage import rotate as _rotate, gaussian_filter1d, gaussian_filter
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from auto_stretch.stretch import Stretch
from skimage import feature
from skimage.transform import probabilistic_hough_line

from spectrum_core import (
    extract_spectrum,
    spectrum_fully_in_frame,
    _dao_xy,
)


# ---------------------------------------------------------------------------
# Pass 1 — whole-frame Hough estimate  (the swappable first pass)
# ---------------------------------------------------------------------------

def detect_angle_hough(data, *, canny_sigma=3, hough_threshold=10,
                       line_length=70, line_gap=3, rng=42, mad_k=1.5):
    """
    Coarse whole-frame tilt estimate via a statistical Hough transform.

    Applies a PixInsight-like stretch, Canny edge detection, then a
    probabilistic Hough transform.  Outlier line slopes are rejected with a
    median ± ``mad_k`` × MAD filter (robust against the PSF halos and
    hot-pixel line detections that contaminate a mean ± σ criterion).

    This is the first pass and the part most sensitive to noisy data or
    frames with few stars / spectral trails.  It is intentionally the first
    function in this module: alternative first-pass detectors should expose
    the same ``(data) -> (angle, n_lines)`` shape so they can be swapped in
    ``estimate_angle`` directly.

    Parameters
    ----------
    data : 2D ndarray
        Mono image data (any numeric dtype); see ``spectrum_core.to_mono``.
    canny_sigma : float
        Gaussian sigma for the Canny edge detector.
    hough_threshold, line_length, line_gap : int
        ``probabilistic_hough_line`` parameters.
    rng : int
        Seed for the Hough transform, for reproducibility.
    mad_k : float
        Half-width of the MAD outlier filter, in MADs.

    Returns
    -------
    angle : float
        Degrees to pass to ``scipy.ndimage.rotate``.
    n_lines : int
        Number of Hough lines detected (for diagnostics).
    """
    stretched = Stretch().stretch(data.astype(float))
    edges = feature.canny(stretched, sigma=canny_sigma)
    lines = probabilistic_hough_line(
        edges, threshold=hough_threshold, line_length=line_length,
        line_gap=line_gap, rng=rng)

    angles = []
    for p0, p1 in lines:
        if p1[0] != p0[0]:
            slope = (p1[1] - p0[1]) / (p1[0] - p0[0])
            angles.append(np.degrees(np.arctan(slope)))   # no rounding

    if not angles:
        return 0.0, 0

    angles = np.array(angles)
    median_a = np.median(angles)
    mad = np.median(np.abs(angles - median_a))
    if mad == 0:
        filtered = angles                              # all identical -- perfect
    else:
        filtered = angles[np.abs(angles - median_a) <= mad_k * mad]
    if len(filtered) == 0:
        filtered = angles
    return float(np.median(filtered)), len(lines)


# ---------------------------------------------------------------------------
# Pass 1 — Fourier power-orientation estimate
# ---------------------------------------------------------------------------

def _hist_peak(hist, n_bins, smooth_bins):
    """Circular-smooth an orientation histogram and return peak and support."""
    if smooth_bins > 0:
        k = int(max(1, smooth_bins * 3))
        xs = np.arange(-k, k + 1)
        g = np.exp(-(xs ** 2) / (2.0 * smooth_bins ** 2))
        g /= g.sum()
        hist = np.real(np.fft.ifft(
            np.fft.fft(hist) * np.fft.fft(g, len(hist))))
        hist = np.roll(hist, -k)

    mean_h = float(hist.mean())
    if mean_h <= 0:
        return None, 0.0

    i = int(np.argmax(hist))
    support = float(hist[i] / mean_h)
    y0, y1, y2 = hist[(i - 1) % n_bins], hist[i], hist[(i + 1) % n_bins]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return (i + delta + 0.5) * (180.0 / n_bins), support


def detect_angle_fft(data, *, assume_near_horizontal=True, max_tilt_deg=45.0,
                     n_bins=720, smooth_bins=2.0):
    """
    Coarse tilt estimate from the orientation of the image's Fourier power.

    A field of parallel dispersed traces concentrates Fourier power along
    the direction *perpendicular* to the traces.  This builds a
    power-weighted histogram of orientation (mod 180°) over the magnitude
    spectrum, locates its peak (with sub-bin parabolic interpolation),
    rotates by 90° to recover the trace axis, and expresses the result as a
    signed tilt away from horizontal.

    Sign convention (matches ``detect_angle_hough`` and the rotation field):
    a spectrum sloping up toward the right is +tilt and the field receives
    +tilt to rotate it level; the mirror case (sloping down toward the
    right, i.e. −tilt / 354.7°) receives −tilt.  The value is returned
    directly with no extra negation.

    Peak-locating is used rather than a doubled-angle vector moment: the
    moment of a broad power distribution is biased toward the global mean
    orientation and systematically *underestimates* a single dominant tilt.
    The histogram peak does not suffer that bias.

    Unlike the Hough method this makes no use of straight-edge detection or
    thresholds: every trace contributes coherently in Fourier space while
    noise spreads diffusely, which is why it is the candidate for sparse /
    noisy frames.  It assumes a single dominant linear orientation in the
    frame (true for SA100/SA200 fields, where all spectra share the grating
    axis).

    Parameters
    ----------
    data : 2D ndarray
        Mono image data (any numeric dtype).
    assume_near_horizontal : bool
        If True (default), the recovered axis is folded to a small signed
        tilt relative to horizontal (0°), matching the SA100/SA200 geometry
        where the dispersion axis runs roughly left-to-right (tilt in
        −90°…+90° about horizontal).  If False, it is folded relative to
        vertical (90°) instead — a documented hook for gratings mounted
        near-vertical; the rest of the pipeline currently assumes
        horizontal.
    max_tilt_deg : float
        Sanity clamp.  If the folded tilt exceeds this, the axis was likely
        mis-identified (dominant structure not the spectrum); 0.0 is
        returned so the refinement pass starts from no rotation rather than
        a wild value.
    n_bins : int
        Orientation-histogram resolution over [0, 180°).  720 → 0.25°/bin.
    smooth_bins : float
        Circular Gaussian smoothing (in bins) applied to the histogram
        before peak-finding, to suppress quantisation noise.

    Returns
    -------
    angle : float
        Signed tilt in degrees, ready for the rotation field.
    support : float
        Peak prominence: histogram peak height divided by its mean.  ~1 when
        no orientation dominates (pure noise), tens-to-hundreds for a clear
        axis.  A diagnostic only — use it to judge whether the estimate is
        trustworthy, not as an angle.
    """
    img = np.asarray(data, dtype=float)
    if img.ndim != 2 or img.size == 0:
        return 0.0, 0.0

    # Remove the mean so the DC term doesn't dominate, then apply a 2D
    # Hann window to suppress the cross-shaped spectral leakage that the
    # image's hard rectangular border would otherwise inject along the
    # 0°/90° axes — exactly where a near-horizontal spectrum's signal sits.
    img = img - np.mean(img)
    ny, nx = img.shape
    img = img * np.outer(np.hanning(ny), np.hanning(nx))

    # Power spectrum, DC at centre.
    power = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2

    # Frequency coordinates (cycles/pixel), DC at centre.
    fy = np.fft.fftshift(np.fft.fftfreq(ny))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(nx))[None, :]

    # Mask the DC neighbourhood: the few lowest-frequency bins carry huge
    # power unrelated to trace orientation (large-scale gradients, residual
    # background) and would swamp the histogram.
    r2 = fx ** 2 + fy ** 2
    dc_radius = 3.0 / max(ny, nx)          # ~3 bins from centre
    keep = r2 > dc_radius ** 2
    if not np.any(keep):
        return 0.0, 0.0

    w = power[keep]
    ang = np.degrees(np.arctan2(
        (fy * np.ones_like(fx))[keep],
        (fx * np.ones_like(fy))[keep])) % 180.0

    hist, _ = np.histogram(ang, bins=n_bins, range=(0.0, 180.0), weights=w)

    power_orientation, support = _hist_peak(hist, n_bins, smooth_bins)
    if power_orientation is None:
        return 0.0, 0.0

    # Power concentrates perpendicular to the traces → trace axis is +90°.
    axis = power_orientation + 90.0

    # Fold the axis to a signed tilt from the assumed orientation.
    ref = 0.0 if assume_near_horizontal else 90.0
    tilt = ((axis - ref + 90.0) % 180.0) - 90.0   # → (−90, 90]

    if not np.isfinite(tilt) or abs(tilt) > max_tilt_deg:
        return 0.0, support

    # Returned directly: same sign as the Hough method and the rotation
    # field (+tilt for a spectrum sloping up to the right).
    return float(tilt), support


# ---------------------------------------------------------------------------
# Pass 1 — structure-tensor coherence estimate
# ---------------------------------------------------------------------------

def detect_angle_coherence(data, *, assume_near_horizontal=True,
                           max_tilt_deg=45.0, tensor_sigma=4.0,
                           n_bins=720, smooth_bins=2.0):
    """
    Coarse tilt estimate from a structure-tensor / coherence-weighted
    orientation histogram.

    At each pixel the structure tensor — the gradient outer product
    [[Jxx, Jxy], [Jxy, Jyy]] smoothed over ``tensor_sigma`` — describes the
    local gradient distribution.  Its larger eigenvector points across the
    dominant local edge; the spectrum *axis* is perpendicular to that.  The
    eigenvalue gap gives a coherence in [0, 1] that is ~1 on locally linear
    structure (the dispersed traces) and ~0 on isotropic features (star
    PSFs, the zero-order blob, flat noise).  Weighting the orientation
    histogram by coherence × energy therefore lets the thin/soft traces
    dominate while round bright features and noise contribute almost
    nothing — the contamination that pulls a plain magnitude histogram off
    the true axis.

    The histogram peak (sub-bin parabolic interpolation) gives the trace
    axis directly — unlike the FFT method there is no perpendicular flip,
    because the tensor already yields the structure orientation rather than
    a Fourier concentration.  The result is folded to a signed tilt from
    horizontal and returned with the same sign convention as the other
    methods (a spectrum sloping up to the right is +tilt; the field receives
    +tilt to level it).

    ``tensor_sigma`` sets the scale of "linear structure": a few pixels
    suits the wide, soft dispersed traces and, as a side effect, blurs
    razor-thin satellite trails below the coherence threshold faster than
    the fatter spectra — so the method leans toward the spectra without any
    explicit trail handling.

    Parameters
    ----------
    data : 2D ndarray
        Mono image data (any numeric dtype).
    assume_near_horizontal : bool
        If True (default), the recovered axis is folded to a small signed
        tilt relative to horizontal (0°), matching the SA100/SA200 geometry.
        If False, relative to vertical (90°).  See ``detect_angle_fft``.
    max_tilt_deg : float
        Sanity clamp; beyond this the dominant structure was likely not the
        spectrum and 0.0 is returned.
    tensor_sigma : float
        Gaussian sigma (pixels) for both the pre-gradient smoothing and the
        tensor-component smoothing.  Larger favours wider structure.
    n_bins : int
        Orientation-histogram resolution over [0, 180°).  720 → 0.25°/bin.
    smooth_bins : float
        Circular Gaussian smoothing (in bins) before peak-finding.

    Returns
    -------
    angle : float
        Signed tilt in degrees, ready for the rotation field.
    support : float
        Peak prominence: histogram peak height divided by its mean.  ~1 when
        no orientation dominates, larger for a clear axis.  A diagnostic
        only — judges trustworthiness, not the angle.
    """
    img = np.asarray(data, dtype=float)
    if img.ndim != 2 or img.size == 0:
        return 0.0, 0.0

    img = img - np.mean(img)

    # Pre-smooth then take gradients — suppresses pixel noise so the tensor
    # responds to genuine extended structure, not single-pixel hot spots.
    sm = gaussian_filter(img, sigma=tensor_sigma)
    gy, gx = np.gradient(sm)

    # Structure-tensor components, smoothed over the same scale.
    Jxx = gaussian_filter(gx * gx, sigma=tensor_sigma)
    Jyy = gaussian_filter(gy * gy, sigma=tensor_sigma)
    Jxy = gaussian_filter(gx * gy, sigma=tensor_sigma)

    # Per-pixel: dominant gradient orientation and coherence.
    # The gradient (edge-normal) direction is 0.5*atan2(2 Jxy, Jxx - Jyy);
    # the structure/axis orientation is perpendicular → + 90°.
    grad_orient = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)   # radians, edge-normal
    axis_orient = np.degrees(grad_orient) + 90.0           # structure axis, deg

    trace = Jxx + Jyy
    diff = np.hypot(Jxx - Jyy, 2.0 * Jxy)
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(trace > 0, diff / trace, 0.0)   # [0, 1]

    # Weight by coherence × energy: coherence rejects round/noisy pixels,
    # energy (trace = λ1 + λ2) favours the bright extended spectrum.
    weight = (coherence * trace).ravel()
    ang = (axis_orient.ravel()) % 180.0

    finite = np.isfinite(weight) & np.isfinite(ang)
    if not np.any(finite) or np.sum(weight[finite]) <= 0:
        return 0.0, 0.0
    weight = weight[finite]
    ang = ang[finite]

    hist, _ = np.histogram(ang, bins=n_bins, range=(0.0, 180.0), weights=weight)

    axis, support = _hist_peak(hist, n_bins, smooth_bins)
    if axis is None:
        return 0.0, 0.0

    # Fold the axis to a signed tilt from the assumed orientation.
    ref = 0.0 if assume_near_horizontal else 90.0
    tilt = ((axis - ref + 90.0) % 180.0) - 90.0   # → (−90, 90]

    if not np.isfinite(tilt) or abs(tilt) > max_tilt_deg:
        return 0.0, support

    # Returned directly: same sign as the other methods and the rotation
    # field (+tilt for a spectrum sloping up to the right).
    return float(tilt), support


# ---------------------------------------------------------------------------
# First-pass method registry
# ---------------------------------------------------------------------------
# Interchangeable whole-frame detectors, each (data) -> (angle_deg, support).
# Assign DEFAULT_FIRST_PASS to one of these (or pass first_pass= to
# estimate_angle) to select the active method.  Order kept stable so an
# index assignment (e.g. FIRST_PASS_METHODS[1]) is meaningful.
Hough     = detect_angle_hough
FFT       = detect_angle_fft
Coherence = detect_angle_coherence

FIRST_PASS_METHODS = [Hough, FFT, Coherence]

# The method estimate_angle uses when no explicit first_pass is given.
DEFAULT_FIRST_PASS = FFT

# ── Pass-2 refinement flag ────────────────────────────────────────────────
# Whether each first-pass method is followed by the strip-based refinement
# (pass 2).  Testing across the available SA100/SA200 frames showed the FFT
# pass-1 angle is consistently more accurate than pass 2 can achieve: the
# refinement's ~0.25° bin resolution is coarser than a clean FFT detection,
# and the strip's bright zero-order / saturated core pulls its gradient peak
# a fraction of a degree off the true trace — so refinement only ever
# degraded the FFT result.  FFT is therefore single-pass.
#
# Hough remains double-pass: its whole-frame estimate is genuinely coarse
# (designed to catch satellite trails) and the refinement reliably improves
# it (e.g. 4.83° → 5.21° on a frame whose true tilt is 5.3°).
detect_angle_hough.refine = True
detect_angle_fft.refine = False
# Coherence, like FFT, proved more accurate at pass 1 than the strip
# refinement could achieve across the available frames — pass 2 only
# degraded it.  Single-pass.
detect_angle_coherence.refine = False

# ── UI selection registry ─────────────────────────────────────────────────
# Display labels for the first-pass method selector, paired with the method
# they choose.  The GUI builds its radio buttons from this list (preserving
# order) and passes the chosen method straight to estimate_angle(first_pass=).
# Kept here, beside the methods, so the module owns both the method identities
# and how they are named to the user.
FIRST_PASS_CHOICES = [
    ("Statistical Hough", Hough),
    ("FFT (single pass)", FFT),
    ("Coherence",         Coherence),
]

# Short forms for the radio buttons, so the three choices fit on one line of
# the explorer's left pane.  Display only: the labels above remain the stored
# value (spectrum_config.json's `first_pass`, validated against
# FIRST_PASS_CHOICES on load), so abbreviating them here cannot invalidate a
# saved config.  A label with no entry here falls back to itself.
FIRST_PASS_SHORT = {
    "Statistical Hough": "Hough",
    "FFT (single pass)": "FFT",
    "Coherence":         "Cohere",
}

# Label of the choice selected by default in the UI (matches
# DEFAULT_FIRST_PASS = FFT).
DEFAULT_FIRST_PASS_LABEL = "FFT (single pass)"


# ---------------------------------------------------------------------------
# Pass 2 — strip-based refinement
# ---------------------------------------------------------------------------

def refine_angle_from_strip(strip, bin_deg=0.25, smooth_sigma=1.5):
    """
    Refine a near-zero residual tilt from a single extracted aperture strip.

    The strip is expected to contain one spectrum running roughly
    horizontally (i.e. after a coarse derotation).  Dominant edges along
    the spectrum produce gradient vectors perpendicular to the dispersion
    axis; the weighted orientation histogram peaks at 90 +/- residual_tilt,
    so the correction angle is peak_orientation - 90.

    Parameters
    ----------
    strip : 2D ndarray
        Aperture rows only (not sky bands) from extract_spectrum.
    bin_deg : float
        Histogram bin width in degrees (default 0.25 deg).
    smooth_sigma : float
        Gaussian smoothing width applied to the histogram before
        peak-finding, in bins (default 1.5 bins ~ 0.37 deg).

    Returns
    -------
    correction : float
        Degrees to add to the current angle field.  Positive means the
        spectrum still slopes upward left to right.
    peak_orientation : float
        Raw peak of the orientation histogram (degrees, 0-180).
    """
    strip_f = strip.astype(float)
    dy = np.gradient(strip_f, axis=0)
    dx = np.gradient(strip_f, axis=1)
    magnitude = np.hypot(dx, dy)

    # Orientation mapped to [0, 180) -- perpendicular to local edges
    orientation = np.degrees(np.arctan2(dy, dx)) % 180.0

    # Magnitude-weighted histogram
    bins = np.arange(0, 180 + bin_deg, bin_deg)
    hist, edges = np.histogram(orientation, bins=bins, weights=magnitude)
    centres = (edges[:-1] + edges[1:]) / 2.0

    # Smooth to suppress quantisation noise before peak-finding
    hist_s = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)

    peak_orientation = float(centres[np.argmax(hist_s)])

    # A perfectly horizontal spectrum has gradient vectors at 90 deg.
    # Residual tilt shifts the peak; correction is the signed offset.
    correction = peak_orientation - 90.0
    return correction, peak_orientation


# ---------------------------------------------------------------------------
# Two-pass entry point
# ---------------------------------------------------------------------------

def estimate_angle(mono, params, *, first_pass=None):
    """
    Estimate the spectrum tilt angle for ``mono`` using both passes.

    Pass 1 (``first_pass``, default ``DEFAULT_FIRST_PASS``) gives a coarse
    whole-frame angle.  Pass 2 rotates by that angle, detects the brightest
    valid source, extracts its aperture strip and refines the residual tilt
    with ``refine_angle_from_strip``.  Pass 2 is skipped — and the pass-1
    angle returned unchanged — when it cannot run (no params, no sources, no
    in-frame source, or any error) OR when the first-pass method is
    single-pass by design (its ``.refine`` flag is False; e.g. the FFT
    method, whose pass-1 angle is consistently better than refinement can
    achieve).  The reason is recorded in the diagnostics either way.

    This function is pure: it neither logs nor touches any GUI state.  The
    caller inspects the returned ``diagnostics`` dict to report progress.

    Parameters
    ----------
    mono : 2D ndarray
        Mono working image (see ``spectrum_core.to_mono``).
    params : dict or None
        The GUI parameter dict (``fwhm``, ``spectrum_width``,
        ``aperture_half``, ``sky_gap``, ``sky_width``).  If None, pass 2 is
        skipped and the pass-1 angle is returned.
    first_pass : callable or None
        Pass-1 detector with signature ``(mono) -> (angle_deg, support)``.
        Defaults to ``DEFAULT_FIRST_PASS`` when None.  Pick one from
        ``FIRST_PASS_METHODS`` to test alternative whole-frame methods.

    Returns
    -------
    angle : float
        Final tilt angle in degrees.
    diagnostics : dict
        Keys:
          ``method``            name of the pass-1 method used
          ``angle1``            pass-1 angle (deg)
          ``support``           pass-1 support scalar (method-specific)
          ``n_lines``           alias of ``support`` (kept for callers that
                                logged the Hough line count)
          ``angle2``            final angle (deg); equals ``angle1`` if
                                pass 2 did not run
          ``correction``        pass-2 correction (deg) or None
          ``peak_orientation``  pass-2 histogram peak (deg) or None
          ``stage2_ran``        bool — whether the refinement was applied
          ``stage2_reason``     str — why pass 2 did/didn't run
    """
    if first_pass is None:
        first_pass = DEFAULT_FIRST_PASS
    angle1, support = first_pass(mono)

    diag = {
        "method": getattr(first_pass, "__name__", str(first_pass)),
        "angle1": angle1,
        "support": support,
        "n_lines": support,
        "angle2": angle1,
        "correction": None,
        "peak_orientation": None,
        "stage2_ran": False,
        "stage2_reason": "",
    }

    if params is None:
        diag["stage2_reason"] = "no parameters available"
        return angle1, diag

    # Some first-pass methods are single-pass by design (see the
    # ``.refine`` flags below the registry): their pass-1 angle is already
    # better than the strip refinement can achieve, so pass 2 is skipped
    # unconditionally.  Methods default to refining if the flag is absent.
    if not getattr(first_pass, "refine", True):
        diag["stage2_reason"] = "skipped — method is single-pass"
        return angle1, diag

    try:
        median_val = float(np.median(mono))
        rotated1 = _rotate(mono, angle1, reshape=False, cval=median_val)
        _, med1, std1 = sigma_clipped_stats(rotated1, sigma=3.0)
        daofind = DAOStarFinder(fwhm=params["fwhm"], threshold=5.0 * std1)
        sources1 = daofind(rotated1 - med1)

        if sources1 is None or len(sources1) == 0:
            diag["stage2_reason"] = "no sources detected"
            return angle1, diag

        sources1.sort("peak", reverse=True)
        h1, w1 = rotated1.shape
        valid1 = [s for s in sources1
                  if spectrum_fully_in_frame(
                      *_dao_xy(s),
                      params["spectrum_width"], angle1, w1, h1)]

        if not valid1:
            diag["stage2_reason"] = "no valid strip found"
            return angle1, diag

        brt = valid1[0]
        sx, sy = _dao_xy(brt)
        region, *_ = extract_spectrum(
            sx, sy, rotated1,
            params["spectrum_width"], params["aperture_half"],
            params["sky_gap"], params["sky_width"],
        )
        correction, peak_ori = refine_angle_from_strip(region)
        angle2 = angle1 + correction

        diag.update(
            angle2=angle2,
            correction=correction,
            peak_orientation=peak_ori,
            stage2_ran=True,
            stage2_reason="refined",
        )
        return angle2, diag

    except Exception as e:
        diag["stage2_reason"] = f"failed ({e})"
        return angle1, diag
