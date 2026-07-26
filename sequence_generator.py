"""
sequence_generator.py
======================
Spectrum *sequence* generator — concept stage.

Given a folder of tracked FITS frames and the analysis parameters currently
set up in the main ``spectrum_explorer`` window, this module:

  1. globs the FITS files in the folder (sorted by DATE-OBS when available,
     else by filename),
  2. for each frame: loads → mono → rotates (same angle as the live params),
     re-detects sources with the same DAOStarFinder settings, and matches the
     live active source by nearest centroid within a snap radius (a coordinate
     sanity check — frames are tracked, so the same star should reappear at
     essentially the same pixel; a frame whose detection falls outside the
     radius is skipped with a warning),
  3. extracts + calibrates each spectrum through the *same* calibrated-spectrum
     helper the live path uses (passed in as ``calibrate_fn``), so the sequence
     can never silently drift from the interactive result,
  4. shows a plain matplotlib animation of the spectra over time, with a
     per-frame title (timestamp + optional source name) and an optional
     per-frame normalisation toggle to neutralise transparency variation.

Design notes
------------
* Deliberately minimal visuals (a single dark-theme axis + FuncAnimation).
  None of the explorer's reference-line / inset machinery is used yet.
* The per-source zero-order anchor residual is *not* re-measured per frame in
  this concept version — the live dispersion polynomial is reused as-is for
  every frame.  For a tracked star of one colour the per-frame residual differs
  only sub-pixel, so this is an acceptable first approximation; wiring a
  per-frame ``measure_zero_order_x`` is a documented follow-up.
* This module never mutates explorer state.  It reads params/poly snapshots and
  calls a calibration callback; all heavy state stays in the explorer.
"""

import os
import glob

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import rotate
from scipy.signal import savgol_filter

from spectrum_core import (
    read_fits_image,
    to_mono,
    spectrum_fully_in_frame,
    _dao_xy,
)
from source_identification import solve_wcs
import tooltip_help as tt

# Theme constants mirror the explorer's dark matplotlib palette.
BG = "#0e1014"
PANEL_BG = "#0f0f1a"
ACC = "#e0c46c"
CURVE = "#4ec9b0"
FG = "#aab2c0"

FITS_GLOBS = ("*.fits", "*.fit", "*.fits.fz", "*.fts")

# Header keywords tried in order for the per-frame timestamp.
TIME_KEYS = ("DATE-OBS", "DATE_OBS", "DATE", "MJD-OBS", "JD")

# Savitzky-Golay smoothing applied to each frame's calibrated flux before it
# reaches the animation/normalisation.  SavGol suppresses noise while
# preserving Balmer line shape far better than a moving average.  Concept-stage
# fixed defaults; these become dialog controls when the generator moves to its
# own window.
SAVGOL_WINDOW = 11     # odd; clamped down if longer than the spectrum
SAVGOL_POLYORDER = 3   # must be < window

# Percentile-based normalisation bounds (replace raw min/max) so a residual
# spike that SavGol does not fully reject cannot set the display scale.
NORM_PLO = 1.0
NORM_PHI = 99.0


# ---------------------------------------------------------------------------
# Frame discovery + per-frame extraction
# ---------------------------------------------------------------------------

def _find_fits_files(folder):
    """Return a de-duplicated list of FITS paths in ``folder`` (no
    recursion), in capture (mtime) order with path as tiebreak — glob
    order is a filesystem accident, and processing/progress should walk
    the frames chronologically.  (The *animation* order does not rest on
    this: extract_sequence re-sorts results by header timestamp.)"""
    def mtime(p):
        try:
            return os.path.getmtime(p)
        except OSError:
            return float("inf")
    found = []
    seen = set()
    for pat in FITS_GLOBS:
        for path in glob.glob(os.path.join(folder, pat)):
            key = os.path.normcase(os.path.abspath(path))
            if key not in seen:
                seen.add(key)
                found.append(path)
    return sorted(found, key=lambda p: (mtime(p), p))


def _frame_timestamp(header):
    """Best-effort observation timestamp string from a FITS header."""
    for key in TIME_KEYS:
        val = header.get(key)
        if val:
            return str(val)
    return None


def _position_dao(rotated, median, std, params, lx, ly, snap_radius):
    """DAOStarFinder nearest-match positioning.

    Returns (sx, sy) for the detection nearest the live source within
    ``snap_radius``, or raises ``_PositionSkip`` with a reason when no usable
    detection is found.  Behaviour is identical to the original inline block.
    """
    daofind = DAOStarFinder(fwhm=params["fwhm"], threshold=5.0 * std)
    sources = daofind(rotated - median)
    if sources is None or len(sources) == 0:
        raise _PositionSkip("no sources detected")

    best_d, best_xy = None, None
    for s in sources:
        sx, sy = _dao_xy(s)
        d = float(np.hypot(sx - lx, sy - ly))
        if best_d is None or d < best_d:
            best_d, best_xy = d, (sx, sy)

    if best_d > snap_radius:
        raise _PositionSkip(
            f"nearest source {best_d:.1f} px from active position "
            f"(> {snap_radius:.0f} px snap radius)")
    return best_xy


def _position_wcs(rotated, header, target_coord, astap_path):
    """WCS plate-solve positioning.

    Plate-solves the *rotated* frame with ASTAP (``solve_wcs`` returns a WCS in
    rotated-frame pixel coordinates, the same system extraction uses), then maps
    the target sky position to a pixel anchor with ``world_to_pixel``.  The
    result is trusted as-is — no local centroid refinement — because the WCS
    solution defines the position.  Raises ``_PositionSkip`` on solve failure.
    """
    wcs = solve_wcs(rotated, header, astap_path=astap_path)
    if wcs is None:
        raise _PositionSkip("WCS solve failed")
    sx, sy = wcs.world_to_pixel(target_coord)
    sx, sy = float(sx), float(sy)
    if not (np.isfinite(sx) and np.isfinite(sy)):
        raise _PositionSkip("WCS projected a non-finite pixel position")
    return sx, sy


class _PositionSkip(Exception):
    """Internal: a frame could not be positioned and should be skipped."""


def extract_sequence(folder, params, angle, dispersion_poly, calibrate_fn,
                     active_xy, snap_radius, log_fn=None,
                     position_mode="dao", target_coord=None, astap_path=None,
                     savgol_window=SAVGOL_WINDOW,
                     savgol_polyorder=SAVGOL_POLYORDER,
                     progress_fn=None):
    """
    Build a time-ordered list of calibrated spectra from a folder of frames.

    Parameters
    ----------
    folder : str
        Directory containing the tracked FITS frames.
    params : dict
        The live analysis-parameter dict (from ``SpectrumExplorer._params``).
    angle : float
        Rotation angle (deg) to apply to every frame before positioning.
    dispersion_poly : ndarray or None
        Dispersion polynomial coefficients (highest degree first), reused for
        all frames.  ``None`` falls back to linear dispersion inside
        ``calibrate_fn``.
    calibrate_fn : callable
        ``calibrate_fn(rotated, sx, sy, params, poly) -> (wls, norm_flux)``.
        Supplied by the explorer so the sequence uses identical calibration
        logic to the interactive path.
    active_xy : (float, float)
        The live active source centroid in rotated-image coordinates; used by
        DAOStarFinder positioning as the nearest-match reference.
    snap_radius : float
        Maximum centroid distance (px) for a DAOStarFinder detection to count
        as the same source.
    log_fn : callable, optional
        ``log_fn(message, level="info")`` for progress / warnings.
    position_mode : {"dao", "wcs"}
        Per-frame source positioning strategy.  ``"dao"`` uses DAOStarFinder
        nearest-match (default, unchanged).  ``"wcs"`` plate-solves each frame
        with ASTAP and projects ``target_coord`` to a pixel anchor — robust on
        low-SNR frames where DAOStarFinder's detection list is unstable.
    target_coord : astropy SkyCoord, optional
        Required for ``position_mode="wcs"``: the target sky position (derived
        by the caller from the main frame's WCS at the active source).
    astap_path : str, optional
        Required for ``position_mode="wcs"``: path to the ASTAP executable.

    Returns
    -------
    list of dict
        One entry per *kept* frame, each with keys:
            ``path``, ``timestamp`` (str or None), ``wls``, ``flux``.
        Sorted by timestamp when all frames carry one, else by filename.

    Notes
    -----
    Both positioning modes converge on the same (sx, sy) anchor and feed the
    identical edge-check + calibration tail, so the two are directly
    comparable.  In WCS mode the anchor is the geometric zero-order position;
    the zero-order colour-centroid bias is *not* applied here — it rides in the
    dispersion polynomial (composed by the caller), exactly as in the live path.
    Frames that cannot be positioned (no detection / failed solve) are skipped
    with a per-frame warning.
    """
    def log(msg, level="info"):
        if log_fn:
            log_fn(msg, level=level)

    use_wcs = (position_mode == "wcs")
    if use_wcs and target_coord is None:
        raise ValueError("position_mode='wcs' requires target_coord")

    files = _find_fits_files(folder)
    if not files:
        raise FileNotFoundError(f"No FITS files found in {folder}")

    n_total = len(files)
    mode_label = "WCS plate solve" if use_wcs else "DAOStarFinder"
    log(f"Sequence: {n_total} frame(s) found; positioning via {mode_label}.")

    lx, ly = active_xy
    spectrum_width = params["spectrum_width"]

    results = []
    for i, path in enumerate(files, start=1):
        name = os.path.basename(path)
        # Per-frame progress (kept or skipped — every frame advances the bar).
        if progress_fn is not None:
            progress_fn(i, n_total)
        # Heartbeat every 10th frame so a long run shows it is progressing
        # (skips are always logged below; this covers the silent kept frames).
        if i % 10 == 0:
            log(f"…processing frame {i}/{n_total} "
                f"({len(results)} kept so far).")
        try:
            raw, header = read_fits_image(path)
        except Exception as e:
            log(f"{name}: load failed ({e}); skipped.", level="warn")
            continue

        data = to_mono(raw)
        _, median, std = sigma_clipped_stats(data, sigma=3.0)
        rotated = rotate(data, angle, reshape=False, cval=median)
        h_img, w_img = rotated.shape

        # ── Positioning (mode-dependent) ────────────────────────────
        try:
            if use_wcs:
                sx, sy = _position_wcs(rotated, header, target_coord,
                                       astap_path)
            else:
                sx, sy = _position_dao(rotated, median, std, params,
                                       lx, ly, snap_radius)
        except _PositionSkip as skip:
            log(f"{name}: {skip}; skipped.", level="warn")
            continue

        # Reject if the spectrum would run off the frame at this position.
        if not spectrum_fully_in_frame(
                sx, sy, spectrum_width, angle, w_img, h_img):
            log(f"{name}: source too close to edge for a full "
                f"spectrum; skipped.", level="warn")
            continue

        try:
            wls, flux = calibrate_fn(rotated, sx, sy, params, dispersion_poly)
        except Exception as e:
            log(f"{name}: extraction failed ({e}); skipped.", level="warn")
            continue

        if wls is None or len(wls) == 0 or not np.any(np.isfinite(flux)):
            log(f"{name}: empty/invalid calibrated spectrum; skipped.",
                level="warn")
            continue

        flux = np.asarray(flux, dtype=float)
        # SavGol-smooth before the data reaches normalisation/animation.  Keep
        # the raw calibrated flux too, so a future dialog toggle can switch
        # smoothing on/off without re-running the sequence.
        flux_smoothed = _smooth_flux(flux, window=savgol_window,
                                     polyorder=savgol_polyorder)

        results.append({
            "path": path,
            "timestamp": _frame_timestamp(header),
            "wls": np.asarray(wls, dtype=float),
            "flux": flux_smoothed,
            "flux_raw": flux,
        })

    if not results:
        return results

    # Sort by timestamp if every kept frame has one; else by filename.
    # Numeric headers (MJD-OBS/JD) must compare as numbers — lexicographic
    # order puts "999.5" after "1000.2"; ISO DATE-OBS strings fall back
    # to string order, which is chronological for them.
    if all(r["timestamp"] for r in results):
        def _ts_key(r):
            try:
                return (0, float(r["timestamp"]), "")
            except ValueError:
                return (1, 0.0, r["timestamp"])
        results.sort(key=_ts_key)
    else:
        results.sort(key=lambda r: os.path.basename(r["path"]))
    return results


def _smooth_flux(flux, window=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER):
    """Savitzky-Golay smooth a 1-D flux array, robust to short arrays and NaNs.

    SavGol preserves line shape (peak height/width) far better than a moving
    average, which matters because the sequence trace carries Balmer features.

    Guards:
      * window is forced odd and clamped to the array length; if the array is
        too short for even a minimal (polyorder+1, rounded up to odd) window,
        the flux is returned unchanged.
      * NaNs (e.g. masked contaminator columns) are linearly interpolated for
        the filter computation, then restored afterwards so the mask survives.
    """
    flux = np.asarray(flux, dtype=float)
    n = flux.size
    if n == 0:
        return flux

    # Smallest valid odd window for this polyorder.
    min_win = polyorder + 1 + ((polyorder + 1) % 2 == 0)  # round up to odd
    if n < min_win:
        return flux

    win = min(window, n)
    if win % 2 == 0:
        win -= 1
    if win < min_win:
        return flux

    nan_mask = ~np.isfinite(flux)
    if nan_mask.any():
        if nan_mask.all():
            return flux
        idx = np.arange(n)
        work = flux.copy()
        work[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask],
                                   flux[~nan_mask])
    else:
        work = flux

    try:
        smoothed = savgol_filter(work, window_length=win, polyorder=polyorder)
    except Exception:
        return flux

    if nan_mask.any():
        smoothed[nan_mask] = np.nan
    return smoothed


def _normalize_unit(flux, plo=NORM_PLO, phi=NORM_PHI):
    """Rescale a flux array to [0, 1] using robust percentile bounds.

    Uses the ``plo``/``phi`` percentiles of the finite values instead of raw
    min/max, so a single residual spike cannot set the scale.  Values outside
    the percentile range are clipped into [0, 1].  Identity if the range is
    degenerate or there are no finite values.
    """
    finite = flux[np.isfinite(flux)]
    if finite.size == 0:
        return flux
    lo, hi = np.percentile(finite, [plo, phi])
    lo, hi = float(lo), float(hi)
    if hi - lo <= 0:
        return flux
    out = (flux - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Generator + animation dialog
# ---------------------------------------------------------------------------

class SequenceGeneratorDialog(tk.Toplevel):
    """Self-contained spectrum-sequence window: setup → run → animate → save.

    This single modeless ``Toplevel`` owns the whole sequence workflow, the way
    ``FullSpectrumDialog`` owns the full-spectrum view:

      * the tunable controls live here (folder + Browse, positioning mode,
        SavGol window/polyorder, normalisation percentiles + on/off),
      * the **Generate** button runs the extraction in-window, logging progress
        to the parent's log pane,
      * the animation (figure + playback) and **Save GIF** (Save-As) are
        embedded — there is no second window.

    All *analysis* state (params, dispersion poly, main WCS, contaminator mask,
    matched source name, ASTAP path) is read from the parent ``SpectrumExplorer``
    at Generate time — the parent stays the single source of truth.  Nothing is
    returned to the parent.

    Single-instance: the parent holds the live reference in
    ``parent._sequence_dialog`` and re-focuses rather than spawning a duplicate.
    """

    INTERVAL_MS = 600  # per-frame animation dwell

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Spectrum sequence generator")
        self.configure(background=BG)
        self.geometry("900x680")
        self.minsize(640, 520)
        self.transient(parent)
        # No grab_set() — modeless, like the other dialogs.

        self.frames = []
        self.source_name = None
        self.anim = None
        self._playing = False

        # ── Control variables (seeded from module defaults) ──────────
        self.v_folder = tk.StringVar(value="")
        self.v_mode = tk.StringVar(value="dao")  # "dao" | "wcs"
        self.v_savgol_window = tk.IntVar(value=SAVGOL_WINDOW)
        self.v_savgol_poly = tk.IntVar(value=SAVGOL_POLYORDER)
        self.v_norm_lo = tk.DoubleVar(value=NORM_PLO)
        self.v_norm_hi = tk.DoubleVar(value=NORM_PHI)
        self.v_normalize = tk.BooleanVar(value=True)
        self._norm_pcts = dict(plo=NORM_PLO, phi=NORM_PHI)

        self._build_controls()
        self._build_figure()
        tt.attach_tree(self, "SequenceGeneratorDialog")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_controls(self):
        bar = tk.Frame(self, background=BG)
        bar.pack(side="top", fill="x", padx=8, pady=(8, 0))

        # Row 1 — folder + browse + generate
        r1 = tk.Frame(bar, background=BG)
        r1.pack(side="top", fill="x", pady=2)
        tk.Label(r1, text="Frames folder:", background=BG, foreground=FG).pack(
            side="left")
        tk.Entry(r1, textvariable=self.v_folder).pack(
            side="left", fill="x", expand=True, padx=(6, 4))
        tk.Button(r1, text="Browse…", command=self._browse_folder,
                  background="#1e232c", foreground="#e6e9ef", relief="flat",
                  activebackground="#262c37", activeforeground="#e6e9ef").pack(
            side="left")
        self.btn_generate = tk.Button(
            r1, text="▶ Generate", command=self._generate,
            background="#1e232c", foreground="#e6e9ef", relief="flat",
            activebackground="#262c37", activeforeground="#e6e9ef")
        self.btn_generate.pack(side="left", padx=(6, 0))

        # Row 2 — positioning mode
        r2 = tk.Frame(bar, background=BG)
        r2.pack(side="top", fill="x", pady=2)
        tk.Label(r2, text="Positioning:", background=BG, foreground=FG).pack(
            side="left")
        self.rb_dao = tk.Radiobutton(
            r2, text="DAOStarFinder (nearest match)", value="dao",
            variable=self.v_mode, background=BG, foreground=FG,
            selectcolor=PANEL_BG, activebackground=BG, activeforeground=FG,
            highlightthickness=0)
        self.rb_dao.pack(side="left", padx=(6, 0))
        self.rb_wcs = tk.Radiobutton(
            r2, text="WCS plate solve (ASTAP)", value="wcs",
            variable=self.v_mode, background=BG, foreground=FG,
            selectcolor=PANEL_BG, activebackground=BG, activeforeground=FG,
            highlightthickness=0)
        self.rb_wcs.pack(side="left", padx=(6, 0))
        # WCS only available when the parent has a solved main WCS.  Show a
        # short reason when it is not, so the greyed-out control is not a
        # mystery (the usual cause is simply not having pressed "Solve to WCS").
        if getattr(self.parent, "_main_wcs", None) is None:
            self.rb_wcs.configure(state="disabled")
            self.v_mode.set("dao")
            tk.Label(r2, text="(solve the main frame to enable)",
                     background=BG, foreground=ACC).pack(
                side="left", padx=(6, 0))

        # Row 3 — smoothing + normalisation tunables
        r3 = tk.Frame(bar, background=BG)
        r3.pack(side="top", fill="x", pady=2)
        tk.Label(r3, text="SavGol window:", background=BG,
                 foreground=FG).pack(side="left")
        tk.Spinbox(r3, from_=3, to=199, increment=2, width=5,
                   textvariable=self.v_savgol_window).pack(
            side="left", padx=(4, 10))
        tk.Label(r3, text="polyorder:", background=BG, foreground=FG).pack(
            side="left")
        tk.Spinbox(r3, from_=1, to=9, width=4,
                   textvariable=self.v_savgol_poly).pack(
            side="left", padx=(4, 10))
        tk.Label(r3, text="Norm pct lo/hi:", background=BG,
                 foreground=FG).pack(side="left")
        tk.Spinbox(r3, from_=0.0, to=49.0, increment=0.5, width=5,
                   textvariable=self.v_norm_lo).pack(side="left", padx=(4, 2))
        tk.Spinbox(r3, from_=51.0, to=100.0, increment=0.5, width=5,
                   textvariable=self.v_norm_hi).pack(side="left", padx=(2, 0))

        # Row 4 — playback controls (disabled until a sequence exists)
        r4 = tk.Frame(bar, background=BG)
        r4.pack(side="top", fill="x", pady=2)
        tk.Checkbutton(
            r4, text="Normalise each frame (neutralise transparency)",
            variable=self.v_normalize, command=self._on_normalize_toggle,
            background=BG, foreground=FG, selectcolor=PANEL_BG,
            activebackground=BG, activeforeground=FG,
            highlightthickness=0, borderwidth=0).pack(side="left")
        self.btn_save = tk.Button(
            r4, text="Save GIF…", command=self._save_gif, state="disabled",
            background="#1e232c", foreground="#e6e9ef", relief="flat",
            activebackground="#262c37", activeforeground="#e6e9ef")
        self.btn_save.pack(side="right")
        self.btn_play = tk.Button(
            r4, text="Pause", command=self._toggle_play, state="disabled",
            background="#1e232c", foreground="#e6e9ef", relief="flat",
            activebackground="#262c37", activeforeground="#e6e9ef")
        self.btn_play.pack(side="right", padx=(0, 6))

        # Row 5 — progress bar (determinate; advances per frame during a run).
        r5 = tk.Frame(bar, background=BG)
        r5.pack(side="top", fill="x", pady=(2, 0))
        self.progress = ttk.Progressbar(r5, mode="determinate", maximum=1)
        self.progress.pack(side="left", fill="x", expand=True)
        self.lbl_progress = tk.Label(r5, text="", background=BG,
                                     foreground=FG, width=12, anchor="e")
        self.lbl_progress.pack(side="left", padx=(6, 0))

    def _build_figure(self):
        self.fig = plt.Figure(figsize=(7, 4), facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL_BG)
        (self.line,) = self.ax.plot([], [], color=CURVE, lw=1.2)
        self._style_axes()
        self.ax.set_title("Generate a sequence to begin", color=FG,
                          fontsize=9)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True,
                                         padx=8, pady=8)
        toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas.draw()

    def _style_axes(self):
        self.ax.tick_params(colors=FG, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(FG)
        self.ax.set_xlabel("Wavelength (Å)", color=FG, fontsize=9)
        self.ax.set_ylabel("Normalised flux", color=FG, fontsize=9)

    # ------------------------------------------------------------------
    # Folder browse
    # ------------------------------------------------------------------
    def _browse_folder(self):
        # Ensure this window has focus first: on Windows a native picker
        # launched from a modeless transient Toplevel can otherwise open
        # behind the main window and appear to do nothing.
        self.lift()
        self.focus_force()
        folder = filedialog.askdirectory(
            title="Select folder of FITS frames", parent=self,
            mustexist=True)
        if folder:
            self.v_folder.set(folder)

    # ------------------------------------------------------------------
    # Generate — read parent state, run extract_sequence, start animation
    # ------------------------------------------------------------------
    def _log(self, msg, level="info"):
        """Log through the parent's pane and pump the UI so progress shows
        live rather than batching at the end of the run.

        Uses update() rather than update_idletasks(): idle-task processing only
        flushes already-queued redraws and is unreliable for forcing a Text or
        ttk widget to repaint in the middle of a synchronous, CPU/subprocess-
        blocking loop.  update() processes the full event queue including the
        pending repaint.  Safe here because the Generate button is disabled for
        the duration of the run, so re-entrant interaction is not a concern.
        """
        self.parent._log(msg, level=level)
        self._pump()

    def _pump(self):
        """Force a real repaint mid-run.  Tolerates the window being closed
        during a long run (update() then raises TclError)."""
        try:
            self.update()
        except tk.TclError:
            pass

    def _reset_progress(self):
        """Zero the bar before a run.  TclError-tolerant like _pump: the
        window can be closed mid-run via the event pumping."""
        try:
            self.progress.configure(maximum=1, value=0)
            self.lbl_progress.configure(text="")
        except tk.TclError:
            return
        self._pump()

    def _on_progress(self, done, total):
        """Per-frame progress callback from extract_sequence.

        Sets the bar maximum from the total on the first call, advances the
        value, and pumps the event loop so the bar repaints between frames
        (the run is synchronous on the main thread).  TclError-tolerant:
        _pump's update() lets the user close the window mid-run, after
        which these widgets are gone.
        """
        if total <= 0:
            return
        try:
            if int(self.progress["maximum"]) != total:
                self.progress.configure(maximum=total)
            self.progress.configure(value=done)
            self.lbl_progress.configure(text=f"{done}/{total}")
        except tk.TclError:
            return
        self._pump()

    def _generate(self):
        parent = self.parent

        # Preconditions: a live extraction must exist for params + a source.
        if parent.rotated_data is None or parent._last_source_xy is None:
            messagebox.showwarning(
                "No active extraction",
                "Run an extraction and select a source in the main window "
                "first, so the sequence has parameters and a source position "
                "to track.", parent=self)
            return

        folder = self.v_folder.get().strip()
        if not folder:
            messagebox.showwarning(
                "No folder", "Browse to a folder of FITS frames first.",
                parent=self)
            return

        p = parent._params()
        if p is None:
            return

        # Warn on unsaved parameter changes — the run uses in-memory state.
        if getattr(parent, "_dirty", False):
            if not messagebox.askyesno(
                    "Unsaved changes",
                    "The configuration has unsaved changes.  The sequence "
                    "will run with the current in-memory parameters.\n\n"
                    "Continue anyway?", parent=self):
                return

        # Positioning mode + target coordinate (WCS only).
        position_mode = self.v_mode.get()
        target_coord = None
        astap_path = None
        if position_mode == "wcs":
            main_wcs = getattr(parent, "_main_wcs", None)
            if main_wcs is None:
                messagebox.showwarning(
                    "No WCS",
                    "WCS positioning needs a solved main frame.  Solve to "
                    "WCS in the main window, or use DAOStarFinder.",
                    parent=self)
                return
            # The stored WCS describes the rotated frame at the angle it was
            # solved at.  The sequence rotates every frame by the CURRENT
            # angle, so a mismatch would place the anchor wrong — refuse rather
            # than extract at stale positions.
            solved_angle = getattr(parent, "_solved_angle", None)
            cur_angle = p["angle"]
            if (solved_angle is None
                    or abs(float(solved_angle) - float(cur_angle)) > 1e-6):
                messagebox.showwarning(
                    "WCS angle mismatch",
                    "The stored WCS was solved at a different rotation angle "
                    f"({solved_angle}) than the current one ({cur_angle}).\n\n"
                    "Re-solve to WCS at the current angle, or use "
                    "DAOStarFinder positioning.", parent=self)
                return
            lx, ly = parent._last_source_xy
            target_coord = main_wcs.pixel_to_world(lx, ly)
            astap_path = parent.v_astap_path.get().strip()

        # Dispersion poly snapshot (reused for all frames).
        poly = parent.get_dispersion_poly()

        # Contaminator mask + sky-column flag from the main analysis (both
        # frozen on the high-SNR working image and reused for every frame, so
        # the low-SNR captures never re-decide cleanliness).
        contam_mask = getattr(parent, "_last_contam_mask", None)
        sky_col_flag = getattr(parent, "_last_sky_col_flag", None)

        def _calibrate(rotated, sx, sy, params, poly_):
            return parent.compute_calibrated_spectrum(
                rotated, sx, sy, params, poly_,
                contam_mask=contam_mask, sky_col_flag=sky_col_flag)

        # Source name for plot titles, if the live solve matched it.
        source_name = None
        idx = parent._selected_source_index()
        if (idx is not None and parent.source_matches is not None
                and idx < len(parent.source_matches)
                and parent.source_matches[idx] is not None):
            mid = getattr(parent.source_matches[idx], "main_id", None)
            if mid and mid != "(unnamed)":
                source_name = mid

        # Smoothing tunables from the dialog.
        try:
            sg_win = int(self.v_savgol_window.get())
            sg_poly = int(self.v_savgol_poly.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning(
                "Invalid smoothing", "SavGol window/polyorder must be "
                "integers.", parent=self)
            return

        self.btn_generate.configure(state="disabled", text="Working…")
        # The run is synchronous and pumps the event queue (_pump), so
        # without a grab the user could re-run the main-window pipeline
        # or edit parameters mid-sequence.  A grab routes all input to
        # this dialog for the duration — closing it mid-run stays
        # possible and is TclError-tolerated throughout.
        # A grab rather than a worker thread; move the pipeline onto a
        # thread only if a synchronous run proves too limiting.
        try:
            self.grab_set()
        except tk.TclError:
            pass
        self._reset_progress()
        self._log("Generating spectrum sequence…")
        try:
            frames = extract_sequence(
                folder, p, p["angle"], poly,
                _calibrate,
                parent._last_source_xy, parent.SOURCE_SNAP_RADIUS,
                log_fn=self._log,
                position_mode=position_mode, target_coord=target_coord,
                astap_path=astap_path,
                savgol_window=sg_win, savgol_polyorder=sg_poly,
                progress_fn=self._on_progress,
            )
        except FileNotFoundError as e:
            messagebox.showwarning("No frames", str(e), parent=self)
            self._reset_progress()
            return
        except Exception as e:
            # Dialog + log only, no re-raise into the Tk callback — same
            # policy as the explorer's _run.
            messagebox.showerror("Sequence error", str(e), parent=self)
            self._log(f"Sequence error: {e}", level="error")
            self._reset_progress()
            return
        finally:
            try:
                self.grab_release()
                self.btn_generate.configure(state="normal", text="▶ Generate")
            except tk.TclError:
                pass  # window closed mid-run

        # The run pumps the event queue, so the window may have been
        # closed while extract_sequence was working — don't animate into
        # destroyed widgets.
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        if not frames:
            messagebox.showwarning(
                "Empty sequence",
                "No frames yielded a usable spectrum.  See the log for "
                "per-frame skip reasons.", parent=self)
            self._reset_progress()
            return

        self._log(f"Sequence: {len(frames)} frame(s) animated.")
        self.frames = frames
        self.source_name = source_name
        self._start_animation()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def _read_norm_args(self):
        """Snapshot the norm percentiles once per start/toggle —
        DoubleVar.get() raises TclError on garbage spinbox input, and
        _frame_flux runs once per animation frame."""
        try:
            self._norm_pcts = dict(plo=float(self.v_norm_lo.get()),
                                   phi=float(self.v_norm_hi.get()))
        except (tk.TclError, ValueError):
            self._norm_pcts = dict(plo=NORM_PLO, phi=NORM_PHI)

    def _frame_flux(self, fr):
        if self.v_normalize.get():
            return _normalize_unit(fr["flux"], **self._norm_pcts)
        return fr["flux"]

    def _start_animation(self):
        self._read_norm_args()
        # Stop any prior animation before rebuilding.
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            self.anim = None

        self._compute_limits()
        self._playing = True
        self.btn_play.configure(state="normal", text="Pause")
        self.btn_save.configure(state="normal")
        self.anim = FuncAnimation(
            self.fig, self._draw_frame, frames=len(self.frames),
            interval=self.INTERVAL_MS, blit=False, repeat=True)
        self.canvas.draw()

    def _compute_limits(self):
        """Fix axes limits across the whole sequence for a stable animation."""
        xmins, xmaxs = [], []
        ymin, ymax = np.inf, -np.inf
        for fr in self.frames:
            wls = fr["wls"]
            flux = self._frame_flux(fr)
            if len(wls):
                xmins.append(np.nanmin(wls))
                xmaxs.append(np.nanmax(wls))
            finite = flux[np.isfinite(flux)]
            if finite.size:
                ymin = min(ymin, float(np.min(finite)))
                ymax = max(ymax, float(np.max(finite)))
        if xmins and xmaxs:
            self.ax.set_xlim(min(xmins), max(xmaxs))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = 0.05 * (ymax - ymin)
            self.ax.set_ylim(ymin - pad, ymax + pad)

    def _draw_frame(self, i):
        fr = self.frames[i % len(self.frames)]
        self.line.set_data(fr["wls"], self._frame_flux(fr))
        parts = []
        if self.source_name:
            parts.append(self.source_name)
        if fr["timestamp"]:
            parts.append(fr["timestamp"])
        parts.append(f"[{i + 1}/{len(self.frames)}]")
        self.ax.set_title("  ".join(parts), color=FG, fontsize=9)
        return (self.line,)

    def _on_normalize_toggle(self):
        """Normalise toggle / percentile change — recompute limits, redraw."""
        if self.frames:
            self._read_norm_args()
            self._compute_limits()
            self.canvas.draw_idle()

    def _toggle_play(self):
        if self.anim is None:
            return
        self._playing = not self._playing
        if self._playing:
            self.anim.event_source.start()
            self.btn_play.config(text="Pause")
        else:
            self.anim.event_source.stop()
            self.btn_play.config(text="Play")

    def _save_gif(self):
        """Write the current sequence to an animated GIF via Pillow (Save-As)."""
        if not self.frames or self.anim is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save animation as GIF", parent=self,
            defaultextension=".gif",
            initialfile="animation.gif",
            filetypes=[("Animated GIF", "*.gif"), ("All files", "*.*")])
        if not path:
            return

        was_playing = self._playing
        if was_playing:
            self.anim.event_source.stop()
        self.btn_save.config(state="disabled", text="Saving…")
        self.update_idletasks()

        fps = max(1, round(1000.0 / self.INTERVAL_MS))
        try:
            self.anim.save(path, writer=PillowWriter(fps=fps))
            messagebox.showinfo(
                "Saved", f"Animation written to:\n{os.path.abspath(path)}",
                parent=self)
        except Exception as e:
            messagebox.showerror(
                "Save failed", f"Could not write GIF:\n{e}", parent=self)
        finally:
            self.btn_save.config(state="normal", text="Save GIF…")
            if was_playing:
                self.anim.event_source.start()

    # ------------------------------------------------------------------
    def _on_close(self):
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
        plt.close(self.fig)
        if getattr(self.parent, "_sequence_dialog", None) is self:
            self.parent._sequence_dialog = None
        self.destroy()
