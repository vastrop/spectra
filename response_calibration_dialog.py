"""
response_calibration_dialog.py
==================
Modal dialog for generating an instrument-response calibration file
from an observed A0V star and a reference template (Pickles or similar).

Workflow:
  1. The dialog reads the currently-extracted spectrum from the parent
     (``column_sums``) and maps it to wavelength using the parent's
     current dispersion solution.
  2. The user loads a reference .dat template (Pickles A0V, etc.).
  3. The dialog computes the response curve in one of two methods:
       * Spline (anchors)  — user-placed anchors on the obs/ref ratio
                             interpolated with a cubic spline.  An
                             Auto-suggest helper places a robust,
                             sigma-clipped starting set.
       * Smoothed ratio    — observed/reference, both Gaussian-smoothed
                             with the same kernel and Balmer-masked
                             (the original method, kept for comparison).
  4. The user tunes the active method's parameters interactively.
  5. Apply installs the curve as the parent's live calibration (an
     in-memory array — no file round-trip); the parent's Save config
     then persists it.  Export writes the same curve as a two-column
     .dat in the format ``load_calibration_file`` consumes, for reuse
     outside this config.

Parent API contract
-------------------
  - ``column_sums`` : 1D ndarray  – raw extracted spectrum (counts/pixel).
  - ``v_dispersion``, ``v_sp_min``, ``v_sp_max`` : tk.StringVar.
  - ``get_dispersion_poly()`` -> ndarray | None — polynomial coefficients
        (highest degree first) for non-linear dispersion, or None for
        the linear fallback.
  - ``_log(msg)`` (optional) — write a status line to the parent's log
        widget if present; falls back to ``print``.
  - ``set_response_curve(wavelengths, factors, path="")`` — install the
        curve as the live calibration (embedded array; ``path`` is
        reference only, blank unless the user exported a .dat).
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt   # style only — never plt.figure
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from spectrum_core import (
    pixels_to_wavelengths,
    save_spectrum_dat,
    load_reference_spectrum,
    compute_instrument_response,
    compute_instrument_response_raw,
    suggest_response_anchors,
    edge_anchors_for_response,
    fit_continuum_spline,
    BALMER_LINES_A,
    TELLURIC_BANDS_A,
)
import tooltip_help as tt


# Fallback values matching DEFAULTS in spectrum_explorer.py.
DISPERSION_FALLBACK    = 7.7
SP_RANGE_MIN_FALLBACK  = 4000
SP_RANGE_MAX_FALLBACK  = 8000

# Default smoothing parameters — sensible starting point for SA100.
DEFAULT_SMOOTH_FWHM_A  = 400.0
DEFAULT_BALMER_HALF_A  = 40.0

# Default off-telluric light smoothing for Raw mode.  0 = off; the user
# dials in a small value (~30-50 Å at 7.6 Å/px) to flatten continuum
# jitter without touching the preserved telluric band depths.
DEFAULT_TELLURIC_SMOOTH_FWHM_A = 0.0

# Default number of anchors for spline-mode auto-suggest.
DEFAULT_N_ANCHORS = 12

# Pixel tolerance for picking an existing anchor under the cursor —
# kept in sync with continuum_dialog.py.
ANCHOR_PICK_RADIUS_PX = 8

# Anchor drag sensitivity, as a fraction of the anchor's own value per
# pixel of vertical mouse travel (multiplicative — see _on_motion).
# 0.002 = 0.2%/px, ×2 per ~350 px, which is the mid-band sensitivity a drag
# in data coordinates would give.
#
# The drag must be multiplicative rather than positional because the response
# factor falls towards zero at the band edges and the pipeline DIVIDES by it.
# A fixed step in data coordinates is a huge relative change near the edges
# (one pixel there is worth ~25% of the value), so the calibrated spectrum
# explodes on a twitch.  Scaling the step by the value being dragged makes
# the sensitivity uniform across the band.
#
# The constant is a feel knob; the proportional law is the load-bearing part.
ANCHOR_DRAG_PER_PX = 0.002


class ResponseCalibrationDialog(tk.Toplevel):
    """
    Modal dialog to build an instrument-response curve from an observed
    A0V and a reference template, save it as a two-column .dat file, and
    optionally also save the raw observed A0V spectrum for later reuse.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Calibrate instrument response")
        self.configure(bg="#0e1014")
        self.geometry("1920x1080")
        self.minsize(960, 540)
        self.transient(parent)
        self.grab_set()

        # State — observed and reference inputs
        self._obs_wls   = None    # wavelength-calibrated observed spectrum
        self._obs_flux  = None
        self._ref_wls   = None    # reference template
        self._ref_flux  = None
        self._ref_path  = None
        self._ref_resampled = None  # reference resampled onto obs grid (display)

        # State — smoothed-ratio method
        self._response  = None    # last computed smoothed response curve
        self._diag      = None    # diagnostics dict from compute_*

        # State — spline method (mirrors continuum_dialog)
        self._anchors          = []    # list of [wavelength_Å, ratio]
        self._dragging_idx     = None
        self._drag_started     = False
        self._drag_y0_px       = 0.0   # drag origin (display px) and the
        self._drag_val0        = 0.0   # value it started from
        self._spline_response  = None  # last spline evaluated on obs_wls

        self._build_ui()
        self.bind("<Escape>", lambda e: self._close())
        self.protocol("WM_DELETE_WINDOW", self._close)

        # Build observed spectrum from parent state, then draw initial view.
        self.after_idle(self._init_observed_and_draw)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        BG, PANEL, FG, ACC, EBG = "#0e1014", "#0f0f1a", "#e6e9ef", "#e0c46c", "#1e232c"
        self._colors = (BG, PANEL, FG, ACC, EBG)

        # Help banner
        tk.Label(
            self,
            text=("Loads the current extraction as an A0V observation. "
                  "Load a reference (e.g. Pickles A0V), then choose a "
                  "method:  Spline — left-click to add anchors, drag to "
                  "adjust, right-click to delete, or Auto-suggest a "
                  "starting set.  Smoothed ratio — set FWHM & Balmer "
                  "mask, then Recompute.  Raw (telluric-preserving) — "
                  "full-resolution obs/ref ratio with the Balmer cores "
                  "interpolated out but telluric bands kept, so they "
                  "divide out of science spectra; the Smooth FWHM field "
                  "is an optional light smoothing (0 = off) applied only "
                  "outside the telluric bands."),
            bg=BG, fg=FG, font=("Courier New", 9), pady=6,
            justify="left", anchor="w", wraplength=1060,
        ).pack(side="top", fill="x", padx=10)

        # ─── Plot ───
        spec_frame = tk.Frame(self, bg=BG)
        spec_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(4, 0))

        plt.style.use("dark_background")
        # Figure(), not plt.figure(): pyplot gives an embedded figure a manager,
        # i.e. a withdrawn second tk.Tk() root.  See spectrum_explorer.py:35.
        self.fig = Figure(facecolor=BG)
        # Two stacked axes: top = obs vs ref overlay, bottom = response curve.
        self.ax_top = self.fig.add_subplot(211)
        self.ax_bot = self.fig.add_subplot(212, sharex=self.ax_top)
        for ax in (self.ax_top, self.ax_bot):
            ax.set_facecolor(PANEL)
            ax.tick_params(labelsize=8, colors="#aab2c0")
            for sp in ax.spines.values():
                sp.set_edgecolor("#262c37")
        self.fig.tight_layout(pad=1.6)

        self.canvas = FigureCanvasTkAgg(self.fig, master=spec_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        tb = tk.Frame(spec_frame, bg=BG)
        tb.pack(side="top", fill="x")
        self._toolbar = NavigationToolbar2Tk(self.canvas, tb)

        # Matplotlib event bindings — spline-mode anchor editing
        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # ─── Controls row ───
        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(side="top", fill="x", padx=10, pady=(8, 0))

        # Reference file column
        ref_col = tk.Frame(ctrl, bg=BG)
        ref_col.pack(side="left", fill="x", expand=True)

        tk.Label(ref_col, text="Reference template",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")
        row1 = tk.Frame(ref_col, bg=BG)
        row1.pack(fill="x")
        self.v_ref_path = tk.StringVar(value="(none loaded)")
        tk.Entry(row1, textvariable=self.v_ref_path, bg=EBG, fg=FG,
                 font=("Courier New", 9), relief="flat", state="readonly",
                 readonlybackground=EBG
                 ).pack(side="left", fill="x", expand=True)
        tk.Button(row1, text="Load…", bg="#1e232c", fg="#e6e9ef",
                  font=("Courier New", 9), relief="flat",
                  activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._load_reference
                  ).pack(side="left", padx=(4, 0))

        # Method selector column
        mode_col = tk.Frame(ctrl, bg=BG)
        mode_col.pack(side="left", padx=(20, 0))
        tk.Label(mode_col, text="Method",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")
        self.v_method = tk.StringVar(value="spline")
        for label, value in (("Spline (anchors)",          "spline"),
                             ("Smoothed ratio",            "smooth"),
                             ("Raw (telluric-preserving)", "raw")):
            _rb = tk.Radiobutton(
                mode_col, text=label, value=value, variable=self.v_method,
                bg=BG, fg=FG, font=("Courier New", 9),
                selectcolor=PANEL, activebackground=BG, activeforeground=ACC,
                command=self._on_method_change,
            )
            _rb.pack(anchor="w")
            tt.attach(_rb, "ResponseCalibrationDialog", "method_selector")

        # Smoothing parameters column
        smooth_col = tk.Frame(ctrl, bg=BG)
        smooth_col.pack(side="left", padx=(20, 0))

        tk.Label(smooth_col, text="Smoothing / mask",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")

        sm_grid = tk.Frame(smooth_col, bg=BG)
        sm_grid.pack(anchor="w", pady=(2, 0))

        # Broad smoothing — Smoothed-ratio method only.  Smooths the whole
        # obs/ref ratio (tellurics included), so it cannot remove tellurics.
        tk.Label(sm_grid, text="Ratio smooth FWHM (Å):",
                 bg=BG, fg=FG, font=("Courier New", 9)
                 ).grid(row=0, column=0, sticky="w")
        self.v_smooth_fwhm = tk.StringVar(value=f"{DEFAULT_SMOOTH_FWHM_A:.0f}")
        self.e_smooth_fwhm = tk.Entry(
            sm_grid, textvariable=self.v_smooth_fwhm, width=6,
            bg=EBG, fg=FG, font=("Courier New", 9), relief="flat")
        self.e_smooth_fwhm.grid(row=0, column=1, padx=(4, 0))
        tt.attach(self.e_smooth_fwhm, "ResponseCalibrationDialog",
                  "Ratio smooth FWHM (Å)")

        # Balmer mask — shared by Smoothed-ratio and Raw methods.
        tk.Label(sm_grid, text="Balmer mask ±Å:",
                 bg=BG, fg=FG, font=("Courier New", 9)
                 ).grid(row=1, column=0, sticky="w")
        self.v_balmer_half = tk.StringVar(value=f"{DEFAULT_BALMER_HALF_A:.0f}")
        _e_balmer = tk.Entry(sm_grid, textvariable=self.v_balmer_half, width=6,
                             bg=EBG, fg=FG, font=("Courier New", 9), relief="flat")
        _e_balmer.grid(row=1, column=1, padx=(4, 0))
        tt.attach(_e_balmer, "ResponseCalibrationDialog", "Balmer mask ±Å")

        # Off-telluric light smoothing — Raw method only.  Smooths the
        # continuum jitter *outside* the telluric bands; band depths are
        # left untouched so they still divide out of science spectra.
        tk.Label(sm_grid, text="Off-telluric smooth FWHM (Å):",
                 bg=BG, fg=FG, font=("Courier New", 9)
                 ).grid(row=2, column=0, sticky="w")
        self.v_telluric_smooth_fwhm = tk.StringVar(
            value=f"{DEFAULT_TELLURIC_SMOOTH_FWHM_A:.0f}")
        self.e_telluric_smooth_fwhm = tk.Entry(
            sm_grid, textvariable=self.v_telluric_smooth_fwhm, width=6,
            bg=EBG, fg=FG, font=("Courier New", 9), relief="flat")
        self.e_telluric_smooth_fwhm.grid(row=2, column=1, padx=(4, 0))
        tt.attach(self.e_telluric_smooth_fwhm, "ResponseCalibrationDialog",
                  "Off-telluric smooth FWHM (Å)")

        tk.Button(smooth_col, text="Recompute",
                  bg="#1e232c", fg="#e6e9ef", font=("Courier New", 9),
                  relief="flat", activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._recompute
                  ).pack(anchor="w", pady=(6, 0))

        # Auto-suggest column (spline mode)
        sug_col = tk.Frame(ctrl, bg=BG)
        sug_col.pack(side="left", padx=(20, 0), anchor="n")
        tk.Label(sug_col, text="Auto-suggest (spline)",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")
        sug_row = tk.Frame(sug_col, bg=BG)
        sug_row.pack(anchor="w", pady=(2, 0))
        tk.Label(sug_row, text="N:", bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(side="left")
        self.v_n_anchors = tk.StringVar(value=str(DEFAULT_N_ANCHORS))
        _e_nanch = tk.Entry(sug_row, textvariable=self.v_n_anchors, width=4,
                            bg=EBG, fg=FG, font=("Courier New", 9), relief="flat")
        _e_nanch.pack(side="left", padx=(2, 8))
        tt.attach(_e_nanch, "ResponseCalibrationDialog", "N: (anchors)")
        tk.Button(sug_col, text="Auto-suggest", bg="#1e232c", fg="#e6e9ef",
                  font=("Courier New", 9), relief="flat",
                  activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._auto_suggest
                  ).pack(anchor="w", pady=(4, 0))
        tk.Button(sug_col, text="Clear anchors", bg="#1e232c", fg="#e6e9ef",
                  font=("Courier New", 9), relief="flat",
                  activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._clear_anchors
                  ).pack(anchor="w", pady=(2, 0))
        self.anchor_count_var = tk.StringVar(value="0 anchors")
        tk.Label(sug_col, textvariable=self.anchor_count_var,
                 bg=BG, fg="#aab2c0", font=("Courier New", 8)
                 ).pack(anchor="w", pady=(2, 0))

        # ─── Bottom row: status + save buttons + close ───
        bottom = tk.Frame(self, bg=BG)
        bottom.pack(side="bottom", fill="x", padx=10, pady=10)

        self.status_var = tk.StringVar(
            value="No reference loaded. Use Load… to choose a Pickles "
                  "A0V .dat (or similar two-column template).")
        tk.Label(bottom, textvariable=self.status_var,
                 bg=BG, fg=FG, font=("Courier New", 9),
                 justify="left", anchor="w", wraplength=520,
                 ).pack(side="left", fill="x", expand=True)

        tk.Button(bottom, text="Save A0V spectrum…",
                  bg="#1e232c", fg="#e6e9ef", font=("Courier New", 9),
                  relief="flat", activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._save_observed
                  ).pack(side="left", padx=(8, 4))
        tk.Button(bottom, text="Export response curve…",
                  bg="#1e232c", fg="#e6e9ef", font=("Courier New", 9),
                  relief="flat", activebackground="#262c37",
                  activeforeground="#e6e9ef",
                  command=self._save_response
                  ).pack(side="left", padx=(0, 4))
        tk.Button(bottom, text="Apply",
                  bg=ACC, fg="#1a1a1a", font=("Courier New", 10, "bold"),
                  relief="flat", padx=10,
                  command=self._apply_response
                  ).pack(side="left", padx=(0, 4))
        tk.Button(bottom, text="Close", bg=ACC, fg="#1a1a1a",
                  font=("Courier New", 10, "bold"), relief="flat",
                  padx=14, command=self._close
                  ).pack(side="right")

        # Grey out the smoothing field(s) that don't apply to the initial
        # method so the interface shows which FWHM is in effect.
        self._sync_field_states()

        # Hover help for the text-keyed buttons; method radios and the
        # colon-suffixed numeric fields were tipped explicitly above.
        tt.attach_tree(self, "ResponseCalibrationDialog")

    # ------------------------------------------------------------------
    # Helpers — observed spectrum from parent
    # ------------------------------------------------------------------

    def _init_observed_and_draw(self):
        col_sums = getattr(self.parent, "column_sums", None)
        if col_sums is None or len(col_sums) == 0:
            self._draw_message("No spectrum extracted yet.\n"
                               "Run extraction in the main window first.")
            return

        # Linear dispersion fallback if poly not available.
        try:
            dispersion = float(self.parent.v_dispersion.get())
        except (ValueError, AttributeError):
            dispersion = DISPERSION_FALLBACK

        poly = None
        if hasattr(self.parent, "get_dispersion_poly"):
            poly = self.parent.get_dispersion_poly()

        pixels = np.arange(len(col_sums))
        wls = pixels_to_wavelengths(pixels, dispersion, poly_coeffs=poly)

        # Clip to the display range so the response covers what the user sees.
        try:
            sp_min = float(self.parent.v_sp_min.get())
            sp_max = float(self.parent.v_sp_max.get())
        except (ValueError, AttributeError):
            sp_min, sp_max = SP_RANGE_MIN_FALLBACK, SP_RANGE_MAX_FALLBACK

        m = (wls >= sp_min) & (wls <= sp_max)
        self._obs_wls  = wls[m]
        self._obs_flux = np.asarray(col_sums, dtype=float)[m]
        # Reference may have been loaded before the observed grid existed
        # or before re-extraction; refresh its resampling.
        self._resample_reference()
        self._redraw()

    # ------------------------------------------------------------------
    # Reference loading and recompute
    # ------------------------------------------------------------------

    def _load_reference(self):
        path = filedialog.askopenfilename(
            title="Select reference spectrum (Pickles or similar)",
            filetypes=[("Data files", "*.dat *.txt *.csv"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            wls, flux = load_reference_spectrum(path)
        except Exception as e:
            messagebox.showerror("Load error", str(e), parent=self)
            return
        self._ref_wls  = wls
        self._ref_flux = flux
        self._ref_path = path
        self.v_ref_path.set(os.path.basename(path))
        self._log(f"Loaded reference: {os.path.basename(path)}  "
                  f"({wls.min():.0f}–{wls.max():.0f} Å, {len(wls)} samples)")
        self._resample_reference()
        # In spline mode, seed the anchors on first reference load so the
        # user sees a useful curve immediately.
        if self.v_method.get() == "spline" and not self._anchors:
            self._auto_suggest(silent=True)
        self._recompute()

    def _recompute(self):
        if self._obs_wls is None or self._ref_wls is None:
            self._redraw()
            return

        if self.v_method.get() == "spline":
            # Re-evaluate spline through current anchors.  Useful when
            # the user has edited the Balmer-mask width and wants a
            # fresh evaluation without re-auto-suggesting.
            self._evaluate_spline()
            n_ok = (0 if self._spline_response is None
                    else int(np.sum(np.isfinite(self._spline_response))))
            self.status_var.set(
                f"Spline response: {n_ok} samples valid, "
                f"{len(self._anchors)} anchor(s). Save when satisfied.")
            self._redraw()
            return

        if self.v_method.get() == "raw":
            try:
                half = float(self.v_balmer_half.get())
                fwhm = float(self.v_telluric_smooth_fwhm.get())
            except ValueError:
                messagebox.showerror("Bad value",
                                     "Balmer half-width and off-telluric "
                                     "smooth FWHM must be numbers.",
                                     parent=self)
                return
            if half < 0 or fwhm < 0:
                messagebox.showerror("Bad value",
                                     "Balmer half-width and off-telluric "
                                     "smooth FWHM must be non-negative.",
                                     parent=self)
                return
            wls, response, diag = compute_instrument_response_raw(
                self._obs_wls, self._obs_flux,
                self._ref_wls, self._ref_flux,
                balmer_half_width_A=half,
                smoothing_fwhm_A=fwhm,
            )
            self._response = response
            self._diag = diag
            n_ok = int(np.sum(np.isfinite(response)))
            smooth_note = (f", off-telluric smooth FWHM={fwhm:.0f} Å"
                           if fwhm > 0 else ", no off-telluric smoothing")
            self.status_var.set(
                f"Raw response: {n_ok}/{len(response)} samples valid "
                f"(Balmer interp ±{half:.0f} Å, tellurics preserved"
                f"{smooth_note}). Save when satisfied.")
            self._redraw()
            return

        # Smoothed-ratio path (unchanged behaviour).
        try:
            fwhm = float(self.v_smooth_fwhm.get())
            half = float(self.v_balmer_half.get())
        except ValueError:
            messagebox.showerror("Bad value",
                                 "Smoothing FWHM and Balmer half-width "
                                 "must be numbers.", parent=self)
            return
        if fwhm < 0 or half < 0:
            messagebox.showerror("Bad value",
                                 "Smoothing FWHM and Balmer half-width "
                                 "must be non-negative.", parent=self)
            return

        wls, response, diag = compute_instrument_response(
            self._obs_wls, self._obs_flux,
            self._ref_wls, self._ref_flux,
            smoothing_fwhm_A=fwhm,
            balmer_half_width_A=half,
        )
        self._response = response
        self._diag = diag
        n_ok = int(np.sum(np.isfinite(response)))
        self.status_var.set(
            f"Response curve: {n_ok}/{len(response)} samples valid "
            f"(FWHM={fwhm:.0f} Å, mask=±{half:.0f} Å). "
            f"Save when satisfied.")
        self._redraw()

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_method_change(self):
        """Refresh the view when the user toggles between methods."""
        # Diagnostics belong to the method that computed them — drop them
        # so the top panel doesn't keep drawing the previous method's
        # overlays until the next Recompute.
        self._diag = None
        if (self.v_method.get() == "spline"
                and self._obs_wls is not None
                and self._ref_wls is not None
                and not self._anchors):
            # First time the user enters spline mode with no anchors yet:
            # quietly auto-suggest so they have something to look at.
            self._auto_suggest(silent=True)
        self._sync_field_states()
        self._evaluate_spline()
        self._redraw()

    def _sync_field_states(self):
        """Enable only the smoothing field that applies to the active
        method, so the interface makes plain which FWHM is in effect:
          * Smoothed ratio → 'Ratio smooth FWHM' (broad, whole-ratio).
          * Raw            → 'Off-telluric smooth FWHM' (light, off-band).
          * Spline         → neither.
        The Balmer-mask field stays editable in all modes (auto-suggest
        and both ratio methods use it)."""
        method = self.v_method.get()
        ratio_state    = "normal" if method == "smooth" else "disabled"
        telluric_state = "normal" if method == "raw"    else "disabled"
        DISABLED_FG = "#5a5a72"
        FG = self._colors[2]
        try:
            self.e_smooth_fwhm.config(
                state=ratio_state,
                fg=(FG if ratio_state == "normal" else DISABLED_FG))
            self.e_telluric_smooth_fwhm.config(
                state=telluric_state,
                fg=(FG if telluric_state == "normal" else DISABLED_FG))
        except (AttributeError, tk.TclError):
            pass

    # ------------------------------------------------------------------
    # Spline-mode helpers
    # ------------------------------------------------------------------

    def _anchors_sorted_arrays(self):
        if not self._anchors:
            return np.array([]), np.array([])
        arr = np.array(self._anchors, dtype=float)
        order = np.argsort(arr[:, 0])
        return arr[order, 0], arr[order, 1]

    def _evaluate_spline(self):
        """Evaluate the spline through anchors on the observed grid."""
        if self._obs_wls is None:
            self._spline_response = None
            return
        aw, af = self._anchors_sorted_arrays()
        if aw.size < 2:
            self._spline_response = None
            return
        self._spline_response = fit_continuum_spline(
            aw, af, self._obs_wls, outside="clamp")

    def _resample_reference(self):
        """Resample the reference onto the observed grid for display.

        Independent of the method choice — gives the top panel a
        consistent 'Reference (resampled)' overlay regardless of
        whether the smoothed-ratio path has been executed.
        """
        if self._obs_wls is None or self._ref_wls is None:
            self._ref_resampled = None
            return
        self._ref_resampled = np.interp(
            self._obs_wls, self._ref_wls, self._ref_flux,
            left=np.nan, right=np.nan,
        )

    def _toolbar_busy(self):
        """True if pan/zoom is active — suppress anchor edits then."""
        try:
            return bool(self._toolbar.mode)
        except Exception:
            return False

    def _pick_anchor(self, event):
        """Return index (into self._anchors) of anchor under cursor, or None."""
        if not self._anchors or event.xdata is None or event.ydata is None:
            return None
        # Convert pick tolerance from pixels to data units in x and y.
        trans = self.ax_bot.transData.inverted()
        x0, y0 = trans.transform((0, 0))
        x1, y1 = trans.transform((ANCHOR_PICK_RADIUS_PX,
                                  ANCHOR_PICK_RADIUS_PX))
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx <= 0 or dy <= 0:
            return None

        best_idx = None
        best_norm = np.inf
        for k, (w, f) in enumerate(self._anchors):
            n = ((w - event.xdata) / dx) ** 2 + \
                ((f - event.ydata) / dy) ** 2
            if n < best_norm:
                best_norm = n
                best_idx = k
        if best_norm > 1.0:
            return None
        return best_idx

    def _on_press(self, event):
        if self.v_method.get() != "spline":
            return
        if self._toolbar_busy() or event.inaxes is not self.ax_bot:
            return
        if event.xdata is None or event.ydata is None:
            return

        idx = self._pick_anchor(event)
        if event.button == 3:  # right-click → delete
            if idx is not None:
                del self._anchors[idx]
                self._evaluate_spline()
                self._redraw()
                self.status_var.set(
                    f"Deleted anchor (now {len(self._anchors)}).")
            return
        if event.button == 1:  # left-click
            if idx is not None:
                self._dragging_idx = idx
                self._drag_started = False
                # Anchor the drag to where it began: the motion handler
                # works from this origin, not from the live cursor value.
                self._drag_y0_px = event.y
                self._drag_val0 = float(self._anchors[idx][1])
                return
            # Empty space → add a new anchor.  Y comes from the click
            # (user's intent), not from the ratio — the user may want
            # to override a noisy point.
            self._anchors.append([float(event.xdata), float(event.ydata)])
            self._evaluate_spline()
            self._redraw()
            self.status_var.set(
                f"Added anchor at {event.xdata:.0f} Å "
                f"(now {len(self._anchors)}).")

    def _on_motion(self, event):
        if self._dragging_idx is None or self._toolbar_busy():
            return
        if event.inaxes is not self.ax_bot or event.y is None:
            return
        self._drag_started = True
        # Wavelength is held constant; only the response value moves.
        # Horizontal moves require delete-and-re-add — same convention as
        # continuum_dialog.
        if 0 <= self._dragging_idx < len(self._anchors):
            self._anchors[self._dragging_idx][1] = self._drag_value(event)
            self._evaluate_spline()
            self._redraw()

    def _drag_value(self, event):
        """New factor for the anchor being dragged.

        Proportional: each pixel of travel scales the value the drag
        started from by ANCHOR_DRAG_PER_PX, so the response can be nudged
        as finely at the near-zero band edges as it can mid-band, and can
        never be dragged to zero or negative (which the pipeline reads as
        a gap).  A non-positive starting value has no scale to work from,
        so it falls back to following the cursor.
        """
        if self._drag_val0 <= 0:
            return float(event.ydata) if event.ydata is not None \
                else self._drag_val0
        dpix = event.y - self._drag_y0_px
        return self._drag_val0 * float(np.exp(ANCHOR_DRAG_PER_PX * dpix))

    def _on_release(self, event):
        if self._dragging_idx is None:
            return
        if self._drag_started and 0 <= self._dragging_idx < len(self._anchors):
            wl, fx = self._anchors[self._dragging_idx]
            self.status_var.set(
                f"Anchor moved to ({wl:.0f} Å, {fx:.3g}).")
        self._dragging_idx = None
        self._drag_started = False

    def _auto_suggest(self, silent=False):
        if self._obs_wls is None or self._ref_wls is None:
            if not silent:
                messagebox.showinfo(
                    "Not ready",
                    "Load both an extracted spectrum and a reference "
                    "template before auto-suggesting anchors.",
                    parent=self)
            return
        try:
            n = int(self.v_n_anchors.get())
        except ValueError:
            messagebox.showerror("Bad value",
                                 "Number of anchors must be an integer.",
                                 parent=self)
            return
        if n < 2:
            messagebox.showerror("Bad value", "Need at least 2 anchors.",
                                 parent=self)
            return
        try:
            half = float(self.v_balmer_half.get())
        except ValueError:
            half = DEFAULT_BALMER_HALF_A

        suggested = suggest_response_anchors(
            self._obs_wls, self._obs_flux,
            self._ref_wls, self._ref_flux,
            n_anchors=n, line_mask_half_A=half,
        )
        if not suggested:
            if not silent:
                self.status_var.set(
                    "Auto-suggest produced no anchors "
                    "(no valid bins in obs/ref ratio).")
            return
        # Add edge anchors so the saved curve covers the full
        # observed range rather than clamping flat past the first/
        # last bin-centre anchor.  Edge anchors look and behave
        # identically to suggested anchors — the user can drag,
        # delete, or replace them like any other.
        edges = edge_anchors_for_response(self._obs_wls, suggested)
        full_set = sorted(list(suggested) + list(edges),
                          key=lambda a: a[0])
        self._anchors = [[wl, fx] for wl, fx in full_set]
        self._evaluate_spline()
        self._redraw()
        if not silent:
            self.status_var.set(
                f"Auto-suggested {len(suggested)} anchor(s).")

    def _clear_anchors(self):
        if not self._anchors:
            return
        if not messagebox.askyesno(
                "Clear anchors",
                f"Remove all {len(self._anchors)} response anchors?",
                parent=self):
            return
        self._anchors = []
        self._spline_response = None
        self._redraw()
        self.status_var.set("All anchors cleared.")

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_observed(self):
        if self._obs_wls is None:
            messagebox.showwarning("No spectrum", "No observed spectrum to save.",
                                   parent=self)
            return
        path = filedialog.asksaveasfilename(
            title="Save observed A0V spectrum",
            defaultextension=".dat",
            filetypes=[("Data files", "*.dat *.txt"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            save_spectrum_dat(
                path, self._obs_wls, self._obs_flux,
                header="Observed A0V spectrum (raw counts, wavelength-calibrated)",
            )
        except Exception as e:
            messagebox.showerror("Save error", str(e), parent=self)
            return
        self._log(f"Saved observed spectrum: {os.path.basename(path)}")

    def _current_curve(self):
        """The active method's response curve + a human-readable method
        description, or (None, None) when nothing is computed yet.
        Silent — callers that need a curve do the complaining."""
        method = self.v_method.get()
        if method == "spline":
            curve = self._spline_response
            method_desc = f"spline through {len(self._anchors)} anchors"
        elif method == "raw":
            curve = self._response
            fwhm = self.v_telluric_smooth_fwhm.get()
            method_desc = (f"raw obs/ref, Balmer-interpolated "
                           f"(±{self.v_balmer_half.get()} Å), tellurics "
                           f"preserved, off-telluric smooth FWHM={fwhm} Å")
        else:
            curve = self._response
            method_desc = (f"smoothed ratio, "
                           f"FWHM={self.v_smooth_fwhm.get()} Å, "
                           f"Balmer mask=±{self.v_balmer_half.get()} Å")
        if curve is None:
            return None, None
        return curve, method_desc

    def _require_curve(self):
        """_current_curve, but complains when there is nothing to use."""
        curve, desc = self._current_curve()
        if curve is None:
            messagebox.showwarning(
                "No response",
                "Load a reference and place/compute the curve first.",
                parent=self)
        return curve, desc

    def _apply_response(self):
        """Install the current curve as the parent's live calibration —
        no .dat round-trip.  Exporting is optional and separate."""
        curve, _desc = self._require_curve()
        if curve is None:
            return
        self.parent.set_response_curve(self._obs_wls, curve)
        self._log(f"Applied response curve ({len(curve)} points) "
                  f"to the live config.")
        self.status_var.set("Applied as the live calibration — Run to see "
                            "it, Save config to persist it.")

    def _save_response(self):
        curve, method_desc = self._require_curve()
        if curve is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save instrument response",
            defaultextension=".dat",
            filetypes=[("Data files", "*.dat *.txt"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        ref_name = os.path.basename(self._ref_path) if self._ref_path else "(unknown)"
        try:
            save_spectrum_dat(
                path, self._obs_wls, curve,
                header=(f"Instrument response (observed / reference). "
                        f"Reference: {ref_name}. Method: {method_desc}."),
            )
        except Exception as e:
            messagebox.showerror("Save error", str(e), parent=self)
            return
        self._log(f"Saved response curve: {os.path.basename(path)}")

        # Exporting also applies: the array goes straight into the live
        # config and the path rides along for reference only — the
        # pipeline reads the array, so the file may later move or vanish.
        self.parent.set_response_curve(self._obs_wls, curve, path)
        self.status_var.set(
            f"Exported {os.path.basename(path)} and applied as the live "
            f"calibration.  Save config to persist.")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_message(self, msg):
        BG, PANEL, FG, ACC, _ = self._colors
        for ax in (self.ax_top, self.ax_bot):
            ax.cla()
            ax.set_facecolor(PANEL)
            ax.tick_params(labelsize=8, colors="#aab2c0")
            for sp in ax.spines.values():
                sp.set_edgecolor("#262c37")
        self.ax_top.text(0.5, 0.5, msg, ha="center", va="center",
                         color="#e94560", fontsize=11,
                         transform=self.ax_top.transAxes)
        self.canvas.draw()

    def _redraw(self):
        BG, PANEL, FG, ACC, _ = self._colors
        for ax in (self.ax_top, self.ax_bot):
            ax.cla()
            ax.set_facecolor(PANEL)
            ax.tick_params(labelsize=8, colors="#aab2c0")
            for sp in ax.spines.values():
                sp.set_edgecolor("#262c37")

        # Top: observed vs reference (overlaid, each normalised to its own
        # max so the shapes can be compared regardless of absolute flux).
        if self._obs_wls is not None:
            obs_n = _norm_max(self._obs_flux)
            self.ax_top.plot(self._obs_wls, obs_n,
                             color="#c0c0d0", linewidth=0.8, label="Observed A0V")
            if self._diag is not None and "obs_smoothed" in self._diag:
                self.ax_top.plot(self._obs_wls,
                                 _norm_max(self._diag["obs_smoothed"]),
                                 color="#88e0ff", linewidth=1.0, alpha=0.9,
                                 label="Observed (smoothed)")
        if self._diag is not None and self._diag.get("ref_resampled") is not None:
            self.ax_top.plot(self._obs_wls,
                             _norm_max(self._diag["ref_resampled"]),
                             color="#e9b04d", linewidth=0.8, alpha=0.7,
                             label="Reference (resampled)")
        elif self._ref_resampled is not None and self._obs_wls is not None:
            # Spline mode (or pre-Recompute smoothed mode): no diag yet,
            # but the orange reference curve is still useful feedback.
            self.ax_top.plot(self._obs_wls,
                             _norm_max(self._ref_resampled),
                             color="#e9b04d", linewidth=0.8, alpha=0.7,
                             label="Reference (resampled)")
        # Sanity check: the observed star divided by the response being
        # built should land on the reference.  In raw mode that is true
        # by construction (the curve IS obs/ref) so the line just confirms
        # the plumbing; in spline mode the gap to the orange curve is the
        # real measure of how well the anchors fit.
        curve, _desc = self._current_curve()
        if curve is not None and self._obs_wls is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                corrected = (np.asarray(self._obs_flux, dtype=float)
                             / np.asarray(curve, dtype=float))
            corrected[~np.isfinite(corrected)] = np.nan
            self.ax_top.plot(self._obs_wls, _norm_max(corrected),
                             color="#7ee787", linewidth=0.9, alpha=0.9,
                             label="Observed / response")

        # Balmer markers
        for line in BALMER_LINES_A:
            self.ax_top.axvline(line, color="#e94560", linestyle="--",
                                linewidth=0.6, alpha=0.35)
        # Telluric bands — shaded only in raw mode, where they are the
        # features being deliberately preserved.
        if self.v_method.get() == "raw":
            for centre, half in TELLURIC_BANDS_A:
                self.ax_top.axvspan(centre - half, centre + half,
                                    color="#4da6ff", alpha=0.10, zorder=0)

        self.ax_top.set_ylabel("Normalised flux", fontsize=9, color="#aab2c0")
        self.ax_top.set_title("Observed vs reference (green = corrected, "
                              "should track the reference)", fontsize=9,
                              color="#aab2c0", pad=4)
        if self._obs_wls is not None:
            self.ax_top.legend(loc="upper right", fontsize=7,
                               facecolor=PANEL, edgecolor="#262c37",
                               labelcolor="#c0c0d0")

        # Bottom: response curve — branches on method.
        method = self.v_method.get()
        if method == "spline":
            aw, af = self._anchors_sorted_arrays()
            if self._spline_response is not None:
                # Response-curve red is the curve's data identity (see
                # response_viewer.CURVE), not the chrome accent.
                self.ax_bot.plot(self._obs_wls, self._spline_response,
                                 color="#e94560", linewidth=1.1,
                                 label="Spline response")
            # Dim overlay of the smoothed-ratio curve, if previously
            # computed — handy for visual comparison.
            if self._response is not None:
                self.ax_bot.plot(self._obs_wls, self._response,
                                 color="#88e0ff", linewidth=0.7,
                                 alpha=0.35, label="Smoothed (ref)")
            if aw.size:
                self.ax_bot.scatter(aw, af, s=42, marker="o",
                                    facecolor="#e94560", edgecolor="#ffffff",
                                    linewidth=0.8, zorder=5)
            if aw.size < 2 and self._spline_response is None:
                self.ax_bot.text(
                    0.5, 0.5,
                    "Place ≥ 2 anchors or click Auto-suggest.",
                    ha="center", va="center", color="#aab2c0",
                    fontsize=10, transform=self.ax_bot.transAxes)
            self.ax_bot.set_title(
                "Instrument response (spline through anchors)",
                fontsize=9, color="#aab2c0", pad=4)
            if self._spline_response is not None or self._response is not None:
                self.ax_bot.legend(loc="upper right", fontsize=7,
                                   facecolor=PANEL, edgecolor="#262c37",
                                   labelcolor="#c0c0d0")
        else:
            if self._response is not None:
                self.ax_bot.plot(self._obs_wls, self._response,
                                 color="#e94560", linewidth=1.0)
                bot_title = (
                    "Instrument response (raw obs / ref, "
                    "Balmer-interpolated, tellurics kept)"
                    if method == "raw"
                    else "Instrument response (obs / ref, smoothed)")
                self.ax_bot.set_title(
                    bot_title, fontsize=9, color="#aab2c0", pad=4)
            else:
                self.ax_bot.text(0.5, 0.5,
                                 "Load a reference to compute the response.",
                                 ha="center", va="center", color="#aab2c0",
                                 fontsize=10, transform=self.ax_bot.transAxes)
        self.ax_bot.set_xlabel("Wavelength (Å)", fontsize=9, color="#aab2c0")
        self.ax_bot.set_ylabel("Response factor", fontsize=9, color="#aab2c0")

        # Anchor count label (spline mode)
        if hasattr(self, "anchor_count_var"):
            self.anchor_count_var.set(
                f"{len(self._anchors)} anchor"
                f"{'s' if len(self._anchors) != 1 else ''}")

        self.fig.tight_layout(pad=1.6)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _log(self, msg):
        if hasattr(self.parent, "_log"):
            try:
                self.parent._log(msg)
                return
            except Exception:
                pass
        print(msg)

    def _close(self):
        # No plt.close(): a Figure() has no pyplot manager to tear down.
        self.destroy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_max(arr):
    a = np.asarray(arr, dtype=float)
    m = np.nanmax(a)
    if not np.isfinite(m) or m <= 0:
        return a
    return a / m
