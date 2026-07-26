"""
continuum_dialog.py
===================
Modal dialog for placing, editing and deleting continuum anchor points
that define a smooth baseline for the wavelength-calibrated spectrum.

The fitted continuum can be applied to the spectrum in two ways:

* **subtract**  →  flux − continuum  (baseline → 0)
* **normalize** →  flux ÷ continuum  (baseline → 1)

The dialog provides a live preview of the corrected spectrum in either
mode while the user edits.  Anchors are committed to ``parent.continuum_anchors``
in place — there is no working copy, matching the established pattern of
``calibration_dialog.py``.

Parent API contract
-------------------
The dialog expects ``parent`` to expose:

  - ``continuum_anchors`` : list of [wavelength_Å, flux] pairs
        Read and mutated in place.

  - ``_calibrated_wls`` : 1D ndarray
        Wavelength axis (Å) for the calibrated spectrum.  Must be set
        (extraction must have run and produced a calibrated spectrum)
        for the dialog to be useful.

  - ``_calibrated_flux`` : 1D ndarray
        Response-corrected, normalised flux for the calibrated spectrum,
        same length as _calibrated_wls.  Anchors live in this space.
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt   # style only — never plt.figure
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from spectrum_core import (
    fit_continuum_spline,
    fit_continuum_auto,
    apply_continuum,
    suggest_continuum_anchors,
    estimate_planck_temperature,
    planck_relative,
    load_reference_library,
    match_reference_templates,
)
from reference_library_viewer import _default_library_dir
import tooltip_help as tt


# Visual identity — kept in sync with calibration_dialog.py
BG, PANEL, FG, ACC, EBG = "#0e1014", "#0f0f1a", "#e6e9ef", "#e0c46c", "#1e232c"

# Match the picker pixel-tolerance used for node selection elsewhere
ANCHOR_PICK_RADIUS_PX = 8


class ContinuumDialog(tk.Toplevel):
    """
    Modal dialog to define a continuum via spline-interpolated anchors,
    with live preview of the corrected spectrum and an auto-suggest
    helper.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Calibrate continuum")
        self.configure(bg=BG)
        self.geometry("1920x1080")
        self.minsize(960, 540)
        self.transient(parent)
        self.grab_set()

        # Interaction state for click/drag on anchors
        self._dragging_idx = None
        self._drag_started = False        # distinguish click from drag

        # Current Planck-temperature fit (dict from
        # estimate_planck_temperature).  Recomputed live from the anchors
        # on every redraw — the grid search is a few ms on ~10 anchors.
        self._planck_fit = None

        # Pull spectrum from parent.  None signals "no spectrum available"
        # and the dialog will draw an instructional message instead.
        wls = getattr(parent, "_calibrated_wls", None)
        flux = getattr(parent, "_calibrated_flux", None)
        if wls is None or flux is None or len(wls) == 0 or len(flux) != len(wls):
            self._wls = None
            self._flux = None
        else:
            self._wls = np.asarray(wls, dtype=float)
            self._flux = np.asarray(flux, dtype=float)

        # Pickles template match — computed once: it depends only on the
        # spectrum, not on the anchors.  Library load + 100-odd template
        # χ² comparisons take well under a second.
        self._template_match = None
        self._template_library = {}
        if self._wls is not None:
            self._template_library = load_reference_library(
                _default_library_dir())
            self._template_match = match_reference_templates(
                self._wls, self._flux, self._template_library)

        self._build_ui()
        # The template panel depends only on the spectrum — drawn once
        # here, never redrawn (the shared canvas repaints it anyway).
        self._draw_template_panel()
        self.bind("<Escape>", lambda e: self._close())
        self.protocol("WM_DELETE_WINDOW", self._close)

        # Initial draw after Tk has measured the layout.
        self.after_idle(self._redraw_all)

    def _close(self):
        # No plt.close(): a Figure() has no pyplot manager to tear down.
        self.destroy()

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Help banner
        tk.Label(
            self,
            text=("Left-click on the spectrum to add anchor points along "
                  "the continuum. Left-drag an anchor vertically to "
                  "override its flux. Right-click an anchor to delete it. "
                  "Use Auto-suggest for a starting set."),
            bg=BG, fg=FG, font=("Courier New", 9), pady=6,
            justify="left", anchor="w", wraplength=1060,
        ).pack(side="top", fill="x", padx=10)

        # ─── Plot area ───
        plot_frame = tk.Frame(self, bg=BG)
        plot_frame.pack(side="top", fill="both", expand=True,
                        padx=10, pady=(4, 0))

        plt.style.use("dark_background")
        # Figure(), not plt.figure(): pyplot gives an embedded figure a manager,
        # i.e. a withdrawn second tk.Tk() root.  See spectrum_explorer.py:35.
        self.fig = Figure(facecolor=BG)
        # Rows: spectrum + continuum overlay + anchors; preview of the
        # corrected spectrum; full Planck curve of the live black-body
        # fit (own x-range, extends past the observed band to show the
        # peak); Pickles template match (spectral-type visual check —
        # unlike Planck it uses the band structure, so it stays honest
        # on cool stars where colour T underreads T_eff).
        self.ax_top = self.fig.add_subplot(411)
        self.ax_bot = self.fig.add_subplot(412, sharex=self.ax_top)
        self.ax_planck = self.fig.add_subplot(413)
        self.ax_tmpl = self.fig.add_subplot(414)
        for ax in (self.ax_top, self.ax_bot, self.ax_planck, self.ax_tmpl):
            ax.set_facecolor(PANEL)
            ax.tick_params(labelsize=8, colors="#aab2c0")
            for sp in ax.spines.values():
                sp.set_edgecolor("#262c37")
        self.fig.tight_layout(pad=1.6)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        tb = tk.Frame(plot_frame, bg=BG)
        tb.pack(side="top", fill="x")
        self._toolbar = NavigationToolbar2Tk(self.canvas, tb)

        # Matplotlib event bindings — left/right clicks, drag, release
        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # ─── Controls row ───
        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(side="top", fill="x", padx=10, pady=(8, 0))

        # Auto-suggest column
        sug = tk.Frame(ctrl, bg=BG)
        sug.pack(side="left", anchor="n")

        tk.Label(sug, text="Auto-suggest",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")

        sug_row = tk.Frame(sug, bg=BG)
        sug_row.pack(anchor="w", pady=(2, 0))
        tk.Label(sug_row, text="N:", bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(side="left")
        self.v_n_anchors = tk.StringVar(value="8")
        _e_nanch = tk.Entry(sug_row, textvariable=self.v_n_anchors, width=4,
                            bg=EBG, fg=FG, font=("Courier New", 9), relief="flat")
        _e_nanch.pack(side="left", padx=(2, 8))
        tt.attach(_e_nanch, "ContinuumDialog", "N: (continuum anchors)")

        self.v_mode = tk.StringVar(value="absorption")
        tk.Radiobutton(sug_row, text="absorption", variable=self.v_mode,
                       value="absorption", bg=BG, fg=FG,
                       selectcolor=PANEL, activebackground=BG,
                       activeforeground=FG, font=("Courier New", 9)
                       ).pack(side="left")
        tk.Radiobutton(sug_row, text="emission", variable=self.v_mode,
                       value="emission", bg=BG, fg=FG,
                       selectcolor=PANEL, activebackground=BG,
                       activeforeground=FG, font=("Courier New", 9)
                       ).pack(side="left")

        tk.Button(sug, text="Suggest anchors",
                  bg="#1e232c", fg="#e6e9ef", font=("Courier New", 9),
                  relief="flat", activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._auto_suggest
                  ).pack(anchor="w", pady=(6, 0))

        # Auto-fit column — fits a Chebyshev continuum via specutils and
        # loads the sampled curve as editable anchors.  Distinct from
        # "Auto-suggest" (which picks per-bin extreme pixels): this is a
        # statistical whole-spectrum fit with line/telluric regions excluded.
        # Both seed the SAME anchor list, so the user refines either result
        # by dragging or deleting points.
        autof = tk.Frame(ctrl, bg=BG)
        autof.pack(side="left", anchor="n", padx=(30, 0))

        tk.Label(autof, text="Auto-fit (specutils)",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")

        tk.Label(autof,
                 text="Chebyshev deg 3, lines/tellurics\nexcluded. Uses N above.",
                 bg=BG, fg="#aab2c0", font=("Courier New", 8),
                 justify="left", anchor="w"
                 ).pack(anchor="w", pady=(2, 0))

        tk.Button(autof, text="Auto-fit continuum",
                  bg="#1e232c", fg="#e6e9ef", font=("Courier New", 9),
                  relief="flat", activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._auto_fit
                  ).pack(anchor="w", pady=(6, 0))

        # Preview-mode column
        prev = tk.Frame(ctrl, bg=BG)
        prev.pack(side="left", anchor="n", padx=(30, 0))

        tk.Label(prev, text="Preview mode",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")

        self.v_preview_mode = tk.StringVar(value="normalize")
        tk.Radiobutton(prev, text="subtract", variable=self.v_preview_mode,
                       value="subtract", bg=BG, fg=FG,
                       selectcolor=PANEL, activebackground=BG,
                       activeforeground=FG, font=("Courier New", 9),
                       command=self._redraw_preview
                       ).pack(anchor="w")
        tk.Radiobutton(prev, text="normalize", variable=self.v_preview_mode,
                       value="normalize", bg=BG, fg=FG,
                       selectcolor=PANEL, activebackground=BG,
                       activeforeground=FG, font=("Courier New", 9),
                       command=self._redraw_preview
                       ).pack(anchor="w")

        # Anchor management column
        mgmt = tk.Frame(ctrl, bg=BG)
        mgmt.pack(side="left", anchor="n", padx=(30, 0))

        tk.Label(mgmt, text="Anchors",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")

        self.anchor_count_var = tk.StringVar(value="0 anchors")
        tk.Label(mgmt, textvariable=self.anchor_count_var,
                 bg=BG, fg=FG, font=("Courier New", 9)
                 ).pack(anchor="w", pady=(2, 4))

        tk.Button(mgmt, text="Clear all",
                  bg="#1e232c", fg="#e6e9ef", font=("Courier New", 9),
                  relief="flat", activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._clear_all
                  ).pack(anchor="w")

        # Temperature column — black-body fit to the continuum anchors
        temp = tk.Frame(ctrl, bg=BG)
        temp.pack(side="left", anchor="n", padx=(30, 0))

        tk.Label(temp, text="Temperature",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")

        tk.Button(temp, text="Evaluate T",
                  bg="#1e232c", fg="#e6e9ef", font=("Courier New", 9),
                  relief="flat", activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._evaluate_temperature
                  ).pack(anchor="w", pady=(2, 4))

        self.temp_result_var = tk.StringVar(value="")
        tk.Label(temp, textvariable=self.temp_result_var,
                 bg=BG, fg=FG, font=("Courier New", 8),
                 justify="left", anchor="w", wraplength=340,
                 ).pack(anchor="w")

        # ─── Footer ───
        footer = tk.Frame(self, bg=BG)
        footer.pack(side="bottom", fill="x", padx=10, pady=10)

        self.status_var = tk.StringVar(value="")
        tk.Label(footer, textvariable=self.status_var,
                 bg=BG, fg=FG, font=("Courier New", 9),
                 justify="left", anchor="w", wraplength=520,
                 ).pack(side="left", fill="x", expand=True)

        tk.Button(footer, text="Close", bg=ACC, fg="#1a1a1a",
                  font=("Courier New", 10, "bold"), relief="flat",
                  padx=14, command=self._close
                  ).pack(side="right")

        # Hover help for the text-keyed controls; the "N:" entry was tipped
        # explicitly above (its JSON key is disambiguated).
        tt.attach_tree(self, "ContinuumDialog")

    # ------------------------------------------------------------------
    # Anchor list helpers — operate on parent.continuum_anchors in place
    # ------------------------------------------------------------------

    def _anchors(self):
        """Return parent.continuum_anchors (live reference, not a copy)."""
        return self.parent.continuum_anchors

    def _anchors_sorted_arrays(self):
        """
        Return (wl, flux) ndarrays sorted by wavelength.  Empty arrays
        if no anchors are present.  Sort is non-destructive — does not
        modify the underlying list.
        """
        anchors = self._anchors()
        if not anchors:
            return np.array([]), np.array([])
        arr = np.asarray(anchors, dtype=float)
        order = np.argsort(arr[:, 0])
        return arr[order, 0], arr[order, 1]

    def _find_anchor_under_cursor(self, event):
        """
        Return the index (into parent.continuum_anchors) of the anchor
        nearest the cursor if it is within ANCHOR_PICK_RADIUS_PX pixels
        on the top axis, else None.
        """
        if event.inaxes is not self.ax_top:
            return None
        anchors = self._anchors()
        if not anchors:
            return None

        # Convert anchor data coords → display (pixel) coords for distance
        # measurement so the pick radius is screen-uniform regardless of
        # axis scaling.
        trans = self.ax_top.transData.transform
        anchor_display = trans(np.asarray(anchors, dtype=float))
        cursor = np.array([event.x, event.y])
        dists = np.hypot(anchor_display[:, 0] - cursor[0],
                         anchor_display[:, 1] - cursor[1])
        idx = int(np.argmin(dists))
        if dists[idx] <= ANCHOR_PICK_RADIUS_PX:
            return idx
        return None

    # ------------------------------------------------------------------
    # Matplotlib event handlers
    # ------------------------------------------------------------------

    def _toolbar_busy(self):
        """True if matplotlib's nav toolbar is in zoom/pan mode."""
        mode = getattr(self._toolbar, "mode", "")
        return mode not in ("", None)

    def _on_press(self, event):
        if self._wls is None or self._toolbar_busy():
            return
        if event.inaxes is not self.ax_top:
            return
        if event.xdata is None or event.ydata is None:
            return

        idx = self._find_anchor_under_cursor(event)

        if event.button == 3:    # right-click: delete
            if idx is not None:
                del self._anchors()[idx]
                self._redraw_all()
                self.status_var.set(f"Deleted anchor (now {len(self._anchors())}).")
            return

        if event.button == 1:    # left-click: select-or-add
            if idx is not None:
                # Begin a potential drag on this anchor.
                self._dragging_idx = idx
                self._drag_started = False
            else:
                # Add a new anchor at the cursor's wavelength.  Flux is
                # read from the spectrum at that wavelength (not from
                # the cursor's y) so the anchor starts on the curve.
                wl = float(event.xdata)
                fx = self._sample_spectrum(wl)
                if not np.isfinite(fx):
                    self.status_var.set(
                        "Click was outside the spectrum range; anchor not added.")
                    return
                self._anchors().append([wl, fx])
                self._redraw_all()
                self.status_var.set(f"Added anchor at {wl:.0f} Å "
                                    f"(now {len(self._anchors())}).")

    def _on_motion(self, event):
        if self._dragging_idx is None or self._toolbar_busy():
            return
        if event.inaxes is not self.ax_top or event.ydata is None:
            return
        # Mark that drag actually moved (so release knows it wasn't a click)
        self._drag_started = True
        # Update the anchor's flux to follow the cursor.  Wavelength is
        # held constant to avoid accidental horizontal drift; the user
        # who wants to move an anchor horizontally should delete and re-add.
        anchors = self._anchors()
        if 0 <= self._dragging_idx < len(anchors):
            anchors[self._dragging_idx][1] = float(event.ydata)
            self._redraw_all(redraw_top=True, redraw_preview=True)

    def _on_release(self, event):
        if self._dragging_idx is None:
            return
        if self._drag_started:
            anchors = self._anchors()
            if 0 <= self._dragging_idx < len(anchors):
                wl, fx = anchors[self._dragging_idx]
                self.status_var.set(
                    f"Anchor moved to ({wl:.0f} Å, {fx:.3f}).")
        self._dragging_idx = None
        self._drag_started = False

    def _sample_spectrum(self, wl):
        """Linear-interpolated flux at wavelength wl, or NaN if out of range."""
        if self._wls is None:
            return float("nan")
        if wl < self._wls[0] or wl > self._wls[-1]:
            return float("nan")
        return float(np.interp(wl, self._wls, self._flux))

    # ------------------------------------------------------------------
    # Controls actions
    # ------------------------------------------------------------------

    def _auto_suggest(self):
        if self._wls is None:
            messagebox.showinfo("No spectrum",
                                "No calibrated spectrum is available.")
            return
        try:
            n = int(self.v_n_anchors.get())
        except ValueError:
            messagebox.showerror("Bad value",
                                 "Number of anchors must be an integer.")
            return
        if n < 2:
            messagebox.showerror("Bad value", "Need at least 2 anchors.")
            return

        mode = self.v_mode.get()
        suggested = suggest_continuum_anchors(
            self._wls, self._flux,
            n_anchors=n, mode=mode, line_mask_half_A=40.0,
        )
        if not suggested:
            self.status_var.set("Auto-suggest produced no anchors "
                                "(no valid pixels in any bin).")
            return

        # Replace existing anchors with the suggestion.  This is what
        # the user wants ~95% of the time and matches the way
        # auto-suggest is presented in the UI.
        self.parent.continuum_anchors = [[wl, fx] for wl, fx in suggested]
        self._redraw_all()
        self.status_var.set(
            f"Auto-suggested {len(suggested)} anchor(s) ({mode} mode).")

    def _auto_fit(self):
        """Fit a continuum with specutils and load the sampled curve as
        editable anchors.  Replaces the current anchor set, exactly like
        _auto_suggest — the user then refines by dragging/deleting points."""
        if self._wls is None:
            messagebox.showinfo("No spectrum",
                                "No calibrated spectrum is available.")
            return
        try:
            n = int(self.v_n_anchors.get())
        except ValueError:
            messagebox.showerror("Bad value",
                                 "Number of anchors must be an integer.")
            return
        if n < 2:
            messagebox.showerror("Bad value", "Need at least 2 anchors.")
            return

        try:
            anchors = fit_continuum_auto(
                self._wls, self._flux, n_anchors=n, line_mask_half_A=40.0)
        except ImportError as exc:
            messagebox.showerror("specutils not available", str(exc))
            return

        if not anchors:
            self.status_var.set("Auto-fit produced no anchors "
                                "(too few finite samples to fit).")
            return

        self.parent.continuum_anchors = [[wl, fx] for wl, fx in anchors]
        self._redraw_all()
        self.status_var.set(
            f"Auto-fit continuum: loaded {len(anchors)} anchor(s) "
            f"(Chebyshev deg 3). Drag or delete to refine.")

    def _evaluate_temperature(self):
        """Fit a Planck curve to the continuum anchors and report the
        colour temperature with an honest (possibly open-ended) interval."""
        aw, af = self._anchors_sorted_arrays()
        if aw.size < 3:
            messagebox.showinfo(
                "Not enough anchors",
                "Place at least 3 continuum anchors first — the fit "
                "needs the continuum shape, not just a slope.")
            return

        fit = self._planck_fit    # kept live by _redraw_all
        if fit is None:
            self.temp_result_var.set(
                "Fit failed: need ≥ 3 anchors with positive flux.")
            return

        regime_txt = {
            "in-band": "Planck peak at {:.0f} Å is inside the spectrum — "
                       "estimate well constrained.",
            "uv-side": "Planck peak at {:.0f} Å is blueward of the spectrum: "
                       "fitting the Rayleigh-Jeans down-slope. The lower "
                       "bound is firm; the upper bound is soft.",
            "ir-side": "Planck peak at {:.0f} Å is redward of the spectrum: "
                       "fitting the Wien rise — good T sensitivity. "
                       "Caveat: below ~4000 K, molecular blanketing (TiO) "
                       "depresses the visible pseudo-continuum, so this "
                       "colour temperature reads several hundred K cooler "
                       "than T_eff — cross-check the template panel.",
        }[fit["regime"]].format(fit["peak_A"])

        if fit["super_rj"]:
            self.temp_result_var.set(
                f"No valid estimate: the continuum falls as "
                f"λ^{fit['loglog_slope']:.2f}, steeper than the λ⁻⁴ "
                f"Rayleigh-Jeans limit — bluer than ANY black body. "
                f"The instrument response correction (or reddening "
                f"handling) is suspect; fix that before trusting T.")
            self.status_var.set("Black-body fit drawn, but flux "
                                "calibration looks unphysical.")
            return

        headline = self._fit_headline(fit)

        loo_txt = ""
        if fit["loo_lo"] is not None:
            span = fit["loo_hi"] / max(fit["loo_lo"], 1e-9)
            loo_txt = (f"Leave-one-out best fits: {fit['loo_lo']:.0f} – "
                       f"{fit['loo_hi']:.0f} K")
            if span > 1.5:
                loo_txt += (" — a single anchor dominates the fit; "
                            "check the extreme blue/red anchors and the "
                            "response correction there.")
            loo_txt += "\n"

        self.temp_result_var.set(
            f"{headline}\n{regime_txt}\n{loo_txt}"
            f"(rms of log-flux residuals: {fit['rms']:.3f}; assumes good "
            f"response correction, ignores reddening)")
        self.status_var.set("Black-body fit drawn over the spectrum.")

    @staticmethod
    def _fit_headline(fit):
        """One-line temperature verdict, shared by the text report and the
        Planck-panel legend."""
        if fit["super_rj"]:
            return (f"no fit: slope λ^{fit['loglog_slope']:.2f} steeper "
                    f"than λ⁻⁴ RJ limit")
        if fit["hi_open"]:
            return f"T ≳ {fit['t_lo']:.0f} K (best grid fit {fit['t_best']:.0f} K)"
        lo = ("<" if fit["lo_open"] else "") + f"{fit['t_lo']:.0f}"
        return f"T ≈ {fit['t_best']:.0f} K  [{lo} – {fit['t_hi']:.0f} K]"

    def _clear_all(self):
        if not self._anchors():
            return
        if not messagebox.askyesno(
                "Clear anchors",
                f"Remove all {len(self._anchors())} continuum anchors?"):
            return
        self.parent.continuum_anchors = []
        self._redraw_all()
        self.status_var.set("All anchors cleared.")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw_all(self, redraw_top=True, redraw_preview=True):
        """Repaint spectrum, preview and Planck panel.  Cheap enough to do
        unconditionally — the Planck grid search is a few ms.

        Every anchor mutation routes through here, so the fit (and its
        top-panel overlay + bottom Planck panel) tracks the anchors live.
        The detailed "Evaluate T" text is cleared here because a redraw
        means the anchors changed and the report is stale; the live
        headline lives on the Planck panel's legend instead.
        """
        aw, af = self._anchors_sorted_arrays()
        self._planck_fit = (estimate_planck_temperature(aw, af)
                            if aw.size >= 3 else None)
        self.temp_result_var.set("")
        if redraw_top:
            self._redraw_top()
        if redraw_preview:
            self._redraw_preview()
        self._redraw_planck()
        self.anchor_count_var.set(
            f"{len(self._anchors())} anchor"
            f"{'s' if len(self._anchors()) != 1 else ''}")
        self.canvas.draw_idle()

    def _redraw_top(self):
        ax = self.ax_top
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.tick_params(labelsize=8, colors="#aab2c0")
        for sp in ax.spines.values():
            sp.set_edgecolor("#262c37")
        ax.set_title("Calibrated spectrum + continuum",
                     fontsize=9, color="#aab2c0", pad=4)
        ax.set_ylabel("Normalised flux", fontsize=9, color="#aab2c0")

        if self._wls is None:
            ax.text(0.5, 0.5,
                    "No calibrated spectrum available.\n"
                    "Run extraction in the main window first.",
                    ha="center", va="center", color="#e94560", fontsize=11,
                    transform=ax.transAxes)
            return

        # The spectrum itself.
        ax.plot(self._wls, self._flux, color="#c0c0d0",
                linewidth=0.8, label="Spectrum")

        # Continuum spline overlay (only if ≥ 2 anchors).
        aw, af = self._anchors_sorted_arrays()
        if aw.size >= 2:
            continuum = fit_continuum_spline(aw, af, self._wls,
                                             outside="clamp")
            ax.plot(self._wls, continuum, color="#f0c040",
                    linewidth=1.1, alpha=0.95, label="Continuum")

        # Best-fit Planck curve from the last "Evaluate T" run.
        if self._planck_fit is not None:
            fit = self._planck_fit
            model = np.exp(fit["log_amp"]) * planck_relative(
                self._wls, fit["t_best"])
            ax.plot(self._wls, model, color="#60d090", linewidth=1.0,
                    linestyle="--", alpha=0.9,
                    label=f"Planck {fit['t_best']:.0f} K")

        # Anchor markers — drawn last so they sit on top.
        if aw.size:
            ax.scatter(aw, af, s=42, marker="o",
                       facecolor="#e94560", edgecolor="#ffffff",
                       linewidth=0.8, zorder=5)

        # Lower right: the upper-right corner is where the last (reddest)
        # anchor is usually being adjusted.
        ax.legend(loc="lower right", fontsize=7,
                  facecolor=PANEL, edgecolor="#262c37",
                  labelcolor="#c0c0d0")

    def _redraw_preview(self):
        ax = self.ax_bot
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.tick_params(labelsize=8, colors="#aab2c0")
        for sp in ax.spines.values():
            sp.set_edgecolor("#262c37")
        ax.set_xlabel("Wavelength (Å)", fontsize=9, color="#aab2c0")

        mode = self.v_preview_mode.get()
        title = ("Preview: continuum-subtracted (flux − continuum)"
                 if mode == "subtract"
                 else "Preview: continuum-normalised (flux ÷ continuum)")
        ax.set_title(title, fontsize=9, color="#aab2c0", pad=4)
        ax.set_ylabel("Subtracted" if mode == "subtract" else "Ratio",
                      fontsize=9, color="#aab2c0")

        if self._wls is None:
            return

        aw, af = self._anchors_sorted_arrays()
        if aw.size < 2:
            ax.text(0.5, 0.5,
                    "Place ≥ 2 anchors to see the preview.",
                    ha="center", va="center", color="#aab2c0", fontsize=10,
                    transform=ax.transAxes)
            return

        continuum = fit_continuum_spline(aw, af, self._wls, outside="clamp")
        corrected = apply_continuum(self._flux, continuum, mode)

        # Reference line: zero for subtract, 1 for normalize.
        ref = 0.0 if mode == "subtract" else 1.0
        ax.axhline(ref, color="#888899", linewidth=0.6,
                   linestyle="--", alpha=0.7)
        ax.plot(self._wls, corrected, color="#88e0ff", linewidth=0.9)

    def _redraw_planck(self):
        """Full Planck curve of the live fit on its own wavelength axis —
        wide enough to show the peak even when it lies outside the
        observed band, as a visual sanity check on the continuum (e.g.
        when the red end is degraded by atmospheric conditions)."""
        ax = self.ax_planck
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.tick_params(labelsize=8, colors="#aab2c0")
        for sp in ax.spines.values():
            sp.set_edgecolor("#262c37")
        ax.set_title("Black-body fit — full Planck curve",
                     fontsize=9, color="#aab2c0", pad=4)
        ax.set_xlabel("Wavelength (Å)", fontsize=9, color="#aab2c0")
        ax.set_ylabel("Relative flux", fontsize=9, color="#aab2c0")

        if self._wls is None:
            return
        fit = self._planck_fit
        if fit is None:
            ax.text(0.5, 0.5,
                    "Place ≥ 3 anchors for a live black-body fit.",
                    ha="center", va="center", color="#aab2c0", fontsize=10,
                    transform=ax.transAxes)
            return

        # X-range: cover the Planck peak with room on both sides, and
        # always include the observed band.
        grid = np.linspace(min(0.3 * fit["peak_A"], self._wls[0]),
                           max(3.0 * fit["peak_A"], 1.3 * self._wls[-1]),
                           600)
        model = np.exp(fit["log_amp"]) * planck_relative(grid, fit["t_best"])
        # Red flags an unphysical (super-Rayleigh-Jeans) fit — reserved
        # warning colour, not the chrome accent.
        curve_color = "#e94560" if fit["super_rj"] else "#60d090"
        ax.plot(grid, model, color=curve_color, linewidth=1.1,
                label=self._fit_headline(fit))

        # Observed band, so the constrained part of the curve is obvious.
        ax.axvspan(self._wls[0], self._wls[-1], color="#88e0ff",
                   alpha=0.08, label="observed band")

        # Wien peak of the fitted black body.
        ax.axvline(fit["peak_A"], color="#f0c040", linewidth=0.8,
                   linestyle=":", alpha=0.9,
                   label=f"peak {fit['peak_A']:.0f} Å")

        # The anchors themselves — they should sit on the curve.
        aw, af = self._anchors_sorted_arrays()
        ax.scatter(aw, af, s=28, marker="o",
                   facecolor="#e94560", edgecolor="#ffffff",
                   linewidth=0.6, zorder=5)

        ax.legend(loc="upper right", fontsize=7,
                  facecolor=PANEL, edgecolor="#262c37",
                  labelcolor="#c0c0d0")

    def _draw_template_panel(self):
        """Pickles template match on the comparison grid — the observed
        band shape vs the best-matching library templates.  Static per
        spectrum (anchors play no role)."""
        ax = self.ax_tmpl
        ax.set_title("Pickles template match (spectral-type check)",
                     fontsize=9, color="#aab2c0", pad=4)
        ax.set_xlabel("Wavelength (Å)", fontsize=9, color="#aab2c0")
        ax.set_ylabel("Flux", fontsize=9, color="#aab2c0")

        match = self._template_match
        if match is None:
            ax.text(0.5, 0.5,
                    "No template match available\n"
                    "(no spectrum, or ReferenceLibrary not found).",
                    ha="center", va="center", color="#aab2c0", fontsize=10,
                    transform=ax.transAxes)
            return

        grid, obs, mask = match["grid"], match["obs"], match["mask"]
        # Observed: faint over the full band, solid where actually compared
        # (Balmer cores and telluric bands are excluded from the fit).
        ax.plot(grid, obs, color="#c0c0d0", linewidth=0.7, alpha=0.35)
        ax.plot(grid, np.where(mask, obs, np.nan), color="#c0c0d0",
                linewidth=0.9, label="observed (smoothed)")

        styles = (("#60d090", "-"), ("#f0c040", "--"))
        for r, (color, ls) in zip(match["ranked"][:2], styles):
            twl, tfx = self._template_library[r["name"]]
            tmpl = r["scale"] * np.interp(grid, twl, tfx)
            ax.plot(grid, tmpl, color=color, linewidth=1.0, linestyle=ls,
                    alpha=0.9,
                    label=f"{r['name']}  (rms {r['rms']:.3f})")

        ax.legend(loc="lower right", fontsize=7,
                  facecolor=PANEL, edgecolor="#262c37",
                  labelcolor="#c0c0d0")
