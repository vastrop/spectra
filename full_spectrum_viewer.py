"""
full_spectrum_viewer.py
=======================
Standalone full-size viewer for the calibrated & normalised spectrum.

Extracted from spectrum_explorer.py so presentation-only features
(annotations, line measurements, export, etc.) can grow here without
bloating the main pipeline class.

Design
------
* Modeless ``Toplevel`` — the user can still adjust extraction
  parameters in the main window while the viewer is open.
* Reads state directly from the parent ``SpectrumExplorer`` instance,
  matching the convention used by ``CalibrationDialog`` and
  ``ResponseDialog``.  The parent is the single source of truth for
  analysis state.
* Single instance per parent — opening the dialog while one is already
  visible focuses and refreshes it rather than spawning a duplicate.
  The parent holds the live reference in ``parent._full_spec_dialog``.
* Continuum-corrected row appears automatically when the parent has
  ≥ 2 continuum anchors.  When fewer anchors exist, the layout falls
  back to the original spectrum + luminance-strip form so empty space
  is never shown.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tooltip_help as tt
from line_overlay import LineOverlayPanel
from spectrum_core import (
    wavelength_to_rgb,
    rainbow_fill,
    fit_continuum_spline,
    apply_continuum,
    write_spectrum_fits,
    rolling_median_nan,
)


# Palette — kept in sync with the parent's chrome.
BG          = "#0e1014"
PANEL       = "#0f0f1a"
SPINE       = "#262c37"
GRID        = "#3d3d6b"
FG          = "#aab2c0"
NODE_COLOUR = "#f0c040"
CONT_COLOUR = "#f0c040"
CORR_COLOUR = "#88e0ff"
CLOSE_BG    = "#e0c46c"

# ±2σ confidence band: width (in columns) of the rolling median applied to
# σ before drawing the band.  The per-column σ is itself a noisy estimate
# (std of ~N sky pixels, with ~σ/√(2N) scatter), while the true background
# noise floor varies smoothly with wavelength — so a modest rolling median
# recovers the smooth floor and tames the jagged band edge without smearing
# real localized structure.  Median (not mean) so a single contaminated sky
# column doesn't bleed into neighbours.  Display-only, tunable; 1 disables.
BAND_SMOOTH_COLS = 9


class FullSpectrumDialog(tk.Toplevel):
    """
    Full-size view of the calibrated & normalised spectrum, optionally
    accompanied by a continuum-corrected panel and a luminance strip.

    Parameters
    ----------
    parent : SpectrumExplorer
        The main application window.  Used both as the Tk parent and as
        the source of all spectrum data and overlay-toggle state.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.configure(bg=BG)
        self.geometry("1920x1080")
        self.minsize(960, 540)
        self.transient(parent)
        # No grab_set() — modeless.
        self.title(self._window_title())

        # ─── State for the new controls ───
        # Continuum-corrected display mode.  Used only when the parent
        # has ≥ 2 anchors; otherwise the radio buttons are disabled.
        self.v_continuum_mode = tk.StringVar(value="normalize")
        # Continuum overlay on the main spectrum panel.  Off by default
        # so the rainbow spectrum stays uncluttered for the common case.
        self.v_show_continuum_overlay = tk.BooleanVar(value=False)
        # ±2σ confidence band on the main spectrum panel.  Mutually
        # exclusive with the per-wavelength RGB fill: when on, the main
        # spectrum is drawn as a neutral line with a shaded background-noise
        # band instead of the coloured fill.  Off by default.
        self.v_show_confidence = tk.BooleanVar(value=False)

        # ─── Top control strip ───
        self._build_controls()

        # ─── Matplotlib figure (built lazily in refresh()) ───
        # Figure(), not plt.figure(): pyplot gives an embedded figure a manager,
        # i.e. a withdrawn second tk.Tk() root.  See spectrum_explorer.py:35.
        self.fig = Figure(facecolor=BG)
        self.ax = None        # main spectrum
        self.ax_corr = None   # continuum-corrected (None when no anchors)
        self.ax_lum = None    # luminance strip

        # Canvas and annotation column share a row, so the column spans the
        # full height of the plot stack rather than the whole window.
        body = tk.Frame(self, bg=BG)
        body.pack(side="top", fill="both", expand=True)

        # Annotation overlay — presentation-only, additive, and discarded
        # when this window closes.  It never writes back to the parent's
        # reference-line toggles, which stay the persistent choice.
        self.lines_panel = LineOverlayPanel(body, on_change=self.refresh)
        self.lines_panel.pack(side="right", fill="y", padx=(8, 10), pady=(4, 4))

        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        tb_frame = tk.Frame(self, bg=BG)
        tb_frame.pack(side="top", fill="x")
        NavigationToolbar2Tk(self.canvas, tb_frame)

        tk.Button(
            self, text="Close", bg=CLOSE_BG, fg="#1a1a1a",
            font=("Courier New", 10, "bold"), relief="flat", padx=14,
            command=self._close,
        ).pack(side="bottom", pady=8)

        self.protocol("WM_DELETE_WINDOW", self._close)

        # Draw once.  Continuum control toggles and the parent both
        # call refresh() to re-render.
        self.refresh()

        tt.attach_tree(self, "FullSpectrumViewer")

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _build_controls(self):
        """
        Build the small Tk control strip above the figure.

        Two controls live here:
          - Continuum overlay checkbox (affects the main spectrum panel)
          - Continuum mode radio (affects the corrected panel)

        Both are always visible.  Their enabled state tracks whether
        the parent has ≥ 2 continuum anchors, refreshed by
        _update_control_state() from refresh().

        Styling note: tk.Checkbutton / tk.Radiobutton indicators render
        inconsistently across platforms (selectcolor sometimes paints
        the indicator regardless of state, hiding the "checked" cue).
        ``indicatoron=False`` sidesteps that: the entire button becomes a
        clickable pill whose background colour shows on/off state via
        _refresh_toggle_styles().
        """
        FG_BRIGHT = "#e6e9ef"
        HEADER    = "#f0c040"

        strip = tk.Frame(self, bg=BG)
        strip.pack(side="top", fill="x", padx=12, pady=(10, 4))

        tk.Label(strip, text="CONTINUUM",
                 bg=BG, fg=HEADER, font=("Courier New", 10, "bold")
                 ).pack(side="left")

        # Common kwargs for the toggle pills.
        pill_kwargs = dict(
            indicatoron=False,
            font=("Courier New", 10),
            borderwidth=1,
            highlightthickness=0,
            padx=8, pady=2,
            disabledforeground="#5a5a72",   # visibly inactive when greyed out
            command=self._on_control_change,
        )

        self.cb_overlay = tk.Checkbutton(
            strip, text="Show overlay",
            variable=self.v_show_continuum_overlay,
            **pill_kwargs,
        )
        self.cb_overlay.pack(side="left", padx=(12, 18))

        # Confidence-band toggle.  Separate concern from the continuum
        # controls (measurement uncertainty, not continuum shape), with its
        # own header.  Always enabled when σ data is present.
        tk.Label(strip, text="UNCERTAINTY",
                 bg=BG, fg=HEADER, font=("Courier New", 10, "bold")
                 ).pack(side="left", padx=(0, 0))
        self.cb_confidence = tk.Checkbutton(
            strip, text="±2σ band",
            variable=self.v_show_confidence,
            **pill_kwargs,
        )
        self.cb_confidence.pack(side="left", padx=(12, 18))

        tk.Label(strip, text="Corrected view:",
                 bg=BG, fg=FG_BRIGHT, font=("Courier New", 10)
                 ).pack(side="left")

        self.rb_subtract = tk.Radiobutton(
            strip, text="subtract", variable=self.v_continuum_mode,
            value="subtract", **pill_kwargs,
        )
        self.rb_subtract.pack(side="left", padx=(8, 2))

        self.rb_normalize = tk.Radiobutton(
            strip, text="normalize", variable=self.v_continuum_mode,
            value="normalize", **pill_kwargs,
        )
        self.rb_normalize.pack(side="left", padx=(2, 0))

        # Status label on the right.
        self.continuum_status_var = tk.StringVar(value="")
        tk.Label(strip, textvariable=self.continuum_status_var,
                 bg=BG, fg=FG_BRIGHT, font=("Courier New", 9, "italic")
                 ).pack(side="right")

        # Export buttons (right side, before the status label visually).
        # Accent-filled like Close: these are the window's data products,
        # and low-key styling makes them hard to find.
        export_kwargs = dict(
            bg=CLOSE_BG, fg="#1a1a1a", activebackground="#d0b25a",
            relief="flat", font=("Courier New", 10, "bold"),
            padx=10, cursor="hand2")
        self.btn_export = tk.Button(
            strip, text="Export FITS…", command=self._on_export_fits,
            **export_kwargs)
        self.btn_export.pack(side="right", padx=(0, 8))
        self.btn_save_png = tk.Button(
            strip, text="Save PNG…", command=self._on_save_png,
            **export_kwargs)
        self.btn_save_png.pack(side="right", padx=(0, 8))

        # Initial style sync.
        self._refresh_toggle_styles()

    def _on_control_change(self):
        """Called by every toggle.  Updates pill styling, then refreshes."""
        self._refresh_toggle_styles()
        self.refresh()

    def _on_export_fits(self):
        """Export the underlying calibrated spectrum to a FITS file.

        Writes the raw calibrated flux and the *unsmoothed* per-column sigma
        (the real data), never the display-smoothed or continuum-operated
        arrays — those are cosmetic.  The continuum mode is recorded as
        metadata only.  This is the data product for downstream analysis and
        for the throughput-calibration loop.
        """
        wls   = self.parent._calibrated_wls
        flux  = self.parent._calibrated_flux
        sigma = getattr(self.parent, "_calibrated_sigma", None)
        p     = self.parent._last_p

        if wls is None or flux is None or p is None:
            messagebox.showwarning(
                "Export FITS",
                "No spectrum available — run an extraction first.",
                parent=self)
            return

        default_name = f"{self._target_stem()}.fits"

        path = filedialog.asksaveasfilename(
            parent=self, title="Export calibrated spectrum",
            defaultextension=".fits", initialfile=default_name,
            filetypes=[("FITS files", "*.fits *.fit"), ("All files", "*.*")])
        if not path:
            return

        # Metadata: the explorer's parameter dict plus the current continuum
        # mode (cosmetic state, recorded for provenance, not applied).
        meta = dict(p)
        meta["continuum_mode"] = self.v_continuum_mode.get()
        meta["n_anchors"] = len(self.parent.continuum_anchors)

        try:
            write_spectrum_fits(path, wls, flux, sigma, meta=meta)
        except Exception as exc:
            messagebox.showerror(
                "Export FITS",
                f"Could not write the FITS file:\n\n{exc}",
                parent=self)
            return

        n_sig = (0 if sigma is None
                 else int(np.count_nonzero(np.isfinite(sigma))))
        messagebox.showinfo(
            "Export FITS",
            f"Wrote {len(wls)} samples to:\n{path}\n\n"
            f"Sigma column: "
            f"{'present' if n_sig else 'absent (NaN)'}.",
            parent=self)

    def _target_stem(self):
        """Filesystem-safe stem derived from the target's basename — the
        suggested-filename base shared by both export buttons.  Basename
        first: sanitizing the full path bakes the drive and every folder
        into the suggested name."""
        p = self.parent._last_p
        target = ((p.get("target") if p else None) or "spectrum").strip() \
            or "spectrum"
        target = os.path.splitext(os.path.basename(target))[0] or "spectrum"
        return "".join(c if (c.isalnum() or c in "-_") else "_"
                       for c in target)

    def _on_save_png(self):
        """Save the figure as a PNG — exactly what is on screen, dark
        theme and current toggles included (unlike Export FITS, which
        writes the raw data product)."""
        path = filedialog.asksaveasfilename(
            parent=self, title="Save figure as PNG",
            defaultextension=".png",
            initialfile=f"{self._target_stem()}.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        if not path:
            return
        try:
            # facecolor: keep the dark chrome instead of savefig's default
            # white, so the file matches the on-screen figure.
            self.fig.savefig(path, dpi=200,
                             facecolor=self.fig.get_facecolor())
        except Exception as exc:
            messagebox.showerror(
                "Save PNG",
                f"Could not write the PNG:\n\n{exc}",
                parent=self)
            return
        messagebox.showinfo("Save PNG", f"Saved figure to:\n{path}",
                            parent=self)

    def _refresh_toggle_styles(self):
        """
        Paint each toggle pill according to its current variable state.

        Selected → bright accent fill on dark text (high contrast).
        Unselected → dark fill on bright text (low-key but readable).
        Disabled state is applied separately by _update_control_state().
        """
        ON_BG,  ON_FG  = "#e0c46c", "#1a1a1a"   # accent fill, dark text
        OFF_BG, OFF_FG = "#262c37", "#e6e9ef"   # subdued fill, bright text

        def _paint(widget, is_on):
            bg, fg = (ON_BG, ON_FG) if is_on else (OFF_BG, OFF_FG)
            try:
                widget.configure(
                    bg=bg, fg=fg,
                    activebackground=bg, activeforeground=fg,
                    selectcolor=bg,    # so indicator-area follows the pill
                )
            except tk.TclError:
                pass

        _paint(self.cb_overlay,   self.v_show_continuum_overlay.get())
        _paint(self.cb_confidence, self.v_show_confidence.get())
        mode = self.v_continuum_mode.get()
        _paint(self.rb_subtract,  mode == "subtract")
        _paint(self.rb_normalize, mode == "normalize")

    def _update_confidence_state(self, have_sigma):
        """Enable the ±2σ toggle only when usable σ data is present."""
        try:
            self.cb_confidence.configure(
                state="normal" if have_sigma else "disabled")
        except tk.TclError:
            pass
        self._refresh_toggle_styles()

    def _update_control_state(self, have_continuum):
        """Enable or disable continuum-related controls."""
        state = "normal" if have_continuum else "disabled"
        for w in (self.cb_overlay, self.rb_subtract, self.rb_normalize):
            try:
                w.configure(state=state)
            except tk.TclError:
                # Widget destroyed (e.g. window closing) — ignore.
                pass
        # Keep the on/off pill colours in sync (especially after a
        # re-enable, where the previously-greyed widget reverts to
        # whatever bg/fg it last had).
        self._refresh_toggle_styles()

        n = len(self.parent.continuum_anchors)
        if have_continuum:
            self.continuum_status_var.set(f"{n} anchors active")
        elif n == 1:
            self.continuum_status_var.set("1 anchor (need ≥ 2)")
        else:
            self.continuum_status_var.set(
                "No continuum — use Calibrate continuum… in the main window")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self):
        """
        Re-render the spectrum from the parent's current cached arrays.
        Called once on open, on every continuum-control toggle, and
        intended for re-use by the parent (or future internal widgets)
        when the underlying data changes.
        """
        wls      = self.parent._calibrated_wls
        norm_int = self.parent._calibrated_flux
        sigma    = getattr(self.parent, "_calibrated_sigma", None)
        p        = self.parent._last_p

        # Caller is expected to verify data presence before opening, but
        # refresh() may be invoked later when the cache has been wiped
        # (e.g. file reload).  Show a friendly placeholder in that case.
        if wls is None or norm_int is None or p is None:
            self._render_empty("No spectrum available — run an extraction first.")
            self._update_control_state(have_continuum=False)
            self._update_confidence_state(have_sigma=False)
            return

        # Cached arrays already reflect a monotonic dispersion fit (or
        # linear fallback) because _display_extraction validated and
        # stored them.  Re-validate only to decide the y-axis label.
        col_n = (len(self.parent.column_sums)
                 if self.parent.column_sums is not None else None)
        poly = self.parent._validate_dispersion_poly(
            self.parent.get_dispersion_poly(), col_n)

        anchors = self.parent.continuum_anchors
        have_continuum = len(anchors) >= 2

        # σ is usable if present, same length as flux, and has any finite
        # value.  When unusable, the band toggle is disabled.
        have_sigma = (
            sigma is not None
            and len(sigma) == len(norm_int)
            and bool(np.any(np.isfinite(sigma))))

        self._update_control_state(have_continuum)
        self._update_confidence_state(have_sigma)
        self._build_axes(have_continuum)
        self._render(wls, norm_int, sigma, p, poly, have_continuum,
                     have_sigma)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_axes(self, have_continuum):
        """
        Build (or rebuild) the figure's axes layout.

        Called on every refresh because the number of rows depends on
        whether continuum anchors exist.  fig.clear() drops all axes
        and any artists; a fresh GridSpec is then attached.

        Row order (top to bottom):
          - main spectrum
          - luminance strip   (acts as a slim visual separator)
          - continuum-corrected (only when have_continuum is True)
        """
        self.fig.clear()

        if have_continuum:
            # main / luminance / corrected
            # Larger hspace than the 2-row case so the corrected panel's
            # title has room to sit above its data without crowding the
            # luminance strip above it.  The uniform hspace also gives
            # the strip a bit of breathing room from the main spectrum,
            # which actually reads better visually.
            gs = self.fig.add_gridspec(3, 1, height_ratios=[6, 1, 3],
                                       hspace=0.25)
            self.ax      = self.fig.add_subplot(gs[0])
            self.ax_lum  = self.fig.add_subplot(gs[1], sharex=self.ax)
            self.ax_corr = self.fig.add_subplot(gs[2], sharex=self.ax)
        else:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[9, 1],
                                       hspace=0.1)
            self.ax      = self.fig.add_subplot(gs[0])
            self.ax_lum  = self.fig.add_subplot(gs[1], sharex=self.ax)
            self.ax_corr = None

        for ax in (self.ax, self.ax_lum, self.ax_corr):
            if ax is None:
                continue
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, wls, norm_int, sigma, p, poly, have_continuum,
                have_sigma):
        """Draw the spectrum and any active overlays/corrections."""
        target_name = self._target_name()

        confidence = have_sigma and self.v_show_confidence.get()

        # --- Main spectrum plot (top) ---
        if confidence:
            # Confidence mode: neutral line + shaded ±2σ band, NO coloured
            # fill (the band occupies the same visual slot as the rainbow
            # fill, so the two are mutually exclusive).  The band is the
            # propagated background-noise term only — see the caption note.
            # σ is smoothed by a NaN-aware rolling median first: the true
            # noise floor is smooth in wavelength, while per-column σ is a
            # noisy estimate, so this recovers the floor and de-jaggs the
            # band edge (display-only; see BAND_SMOOTH_COLS).
            sigma_s = rolling_median_nan(sigma, BAND_SMOOTH_COLS)
            lo = norm_int - 2.0 * sigma_s
            hi = norm_int + 2.0 * sigma_s
            # fill_between tolerates NaN by leaving gaps; mask non-finite
            # explicitly so a NaN σ doesn't blank a finite-flux point.
            band_ok = np.isfinite(lo) & np.isfinite(hi)
            self.ax.fill_between(
                wls, lo, hi, where=band_ok,
                color="#5aa9e6", alpha=0.30, linewidth=0, zorder=2,
                label="±2σ (background noise only)")
            self.ax.plot(wls, norm_int,
                         color="#e8e8f0", linewidth=0.9, alpha=0.95,
                         zorder=3)
        else:
            # Coloured fill per pixel pair + white profile on top — same
            # visual treatment as the inline calibrated panel, just at a
            # larger size.  Single-artist PolyCollection (see rainbow_fill)
            # — a per-segment fill_between loop would dominate refresh
            # cost, which matters because livestack refreshes this window
            # on every stacked frame.
            rainbow_fill(self.ax, wls, norm_int, zorder=2)
            self.ax.plot(wls, norm_int,
                         color="white", linewidth=0.8, alpha=0.7, zorder=3)

        # Continuum overlay (computed once, drawn if requested).
        continuum = None
        if have_continuum:
            aw, af = self._anchor_arrays()
            continuum = fit_continuum_spline(aw, af, wls, outside="clamp")
            if self.v_show_continuum_overlay.get():
                self.ax.plot(wls, continuum,
                             color=CONT_COLOUR, linewidth=1.2, alpha=0.9,
                             zorder=4, label="Continuum")
                # Anchor markers keep the red of the other anchor editors —
                # data markers, and they must stay visible on the gold
                # continuum curve.
                self.ax.scatter(aw, af, s=28, marker="o",
                                facecolor="#e94560", edgecolor="#ffffff",
                                linewidth=0.6, zorder=5)

        self.ax.set_xlim(p["sp_min"], p["sp_max"])
        self.ax.set_ylim(bottom=0)
        self.ax.set_ylabel("Norm. flux", fontsize=11, color=FG)
        self.ax.tick_params(labelsize=9, colors=FG, labelbottom=False)
        self.ax.grid(True, color=GRID, linewidth=0.5, zorder=1)

        # In confidence mode, label what the band includes (so a tight band
        # on a bright source isn't mistaken for completeness — the source
        # Poisson term is not yet included) and give the upper band room.
        if confidence:
            leg = self.ax.legend(
                loc="upper left", fontsize=9, framealpha=0.25,
                facecolor=PANEL, edgecolor=SPINE)
            for txt in leg.get_texts():
                txt.set_color(FG)
            finite_hi = hi[np.isfinite(hi)]
            if finite_hi.size:
                top = max(1.0, float(np.nanpercentile(finite_hi, 99)))
                self.ax.set_ylim(0, top * 1.05)

        # Reference lines on main plot — delegate to the parent's central helper.
        # force_linear=True because this x-axis is already in Å: passing
        # dispersion=1.0 makes the helper plot at xpix = wl/1 = wl.
        cal_ymax = float(np.nanmax(norm_int)) if len(norm_int) else 1.0
        _cal_p = dict(p, dispersion=1.0)
        # One label-slot list across the groups and the overlay, so every
        # label on this axis staggers against every other.
        slots = []
        group_wls = self.parent._draw_reference_line_groups(
            _cal_p, cal_ymax, ax=self.ax,
            force_linear=True, fontsize=12, occupied=slots,
        )

        # Annotation overlay on top, skipping whatever the groups above
        # already marked so a line the user has on both ways is not
        # labelled twice at the same x.
        self.lines_panel.draw(self.ax, cal_ymax, skip=group_wls, fontsize=12,
                              occupied=slots)

        # Calibration node markers — driven by the parent's "Show
        # calibration lines" toggle in the right pane.
        show_nodes = getattr(self.parent, "v_cal_lines", None)
        nodes = self.parent.dispersion_nodes
        if show_nodes is not None and show_nodes.get() and nodes:
            ylims = self.ax.get_ylim()
            yspan = ylims[1] - ylims[0]
            for _px, wl in nodes:
                self.ax.axvline(
                    x=wl, color=NODE_COLOUR, linestyle=":",
                    linewidth=1.0, alpha=0.9,
                )
                self.ax.text(
                    wl, ylims[1] - 0.05 * yspan, f"{wl:.1f}",
                    rotation=90, va="top", ha="right",
                    color=NODE_COLOUR, fontsize=8,
                )

        # --- Luminance strip plot ---
        # When have_continuum is True, the strip is the *middle* panel,
        # sandwiched between the main spectrum and the corrected panel.
        # In that case it should not carry the x-axis labels — the
        # corrected panel below will.  When there's no corrected panel,
        # the strip is the bottom and owns the wavelength axis labels.
        self._render_luminance_strip(wls, norm_int, p,
                                     is_bottom=not have_continuum)

        # Calibration node markers also drawn on the luminance strip.
        if show_nodes is not None and show_nodes.get() and nodes:
            for _px, wl in nodes:
                self.ax_lum.axvline(
                    x=wl, color=NODE_COLOUR, linestyle=":",
                    linewidth=0.8, alpha=0.5,
                )

        # --- Continuum-corrected panel (bottom), if any ---
        # Drawn after the luminance strip so it ends up as the bottom-
        # most row visually (matching the GridSpec layout).  This panel
        # carries the wavelength x-axis when present.
        if have_continuum and continuum is not None:
            self._render_corrected(wls, norm_int, continuum, p)

        # Main title
        self.fig.suptitle(
            f"Calibrated & normalised spectrum — {target_name}",
            color=FG, fontsize=11, y=0.98,
        )

        # Manually adjust outer subplot margins.  hspace is omitted on
        # purpose so the value set by add_gridspec() (which differs
        # between the 2-row and 3-row layouts) is preserved.
        self.fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08,
                                 right=0.95)
        self.canvas.draw_idle()

    def _render_corrected(self, wls, norm_int, continuum, p):
        """
        Draw the continuum-corrected spectrum on self.ax_corr.

        Mode is taken from self.v_continuum_mode (subtract / normalize).
        A horizontal reference line marks the baseline (0 or 1).
        Pixels above/below the baseline get a faint fill to make
        emission vs absorption visually obvious at a glance.

        Theming differs by mode.  Normalised spectra are conventionally
        published on a light background, so "normalize" mode follows that
        convention and matches paper figures.  The
        subtract view stays on the dark theme used by the rest of the
        dialog.

        This panel is the bottom-most when continuum is active, so it
        carries the wavelength axis tick labels and x-label.
        """
        ax = self.ax_corr
        mode = self.v_continuum_mode.get()
        corrected = apply_continuum(norm_int, continuum, mode)

        # Per-mode theme.  All colours pair so that traces and baseline
        # remain visible on the chosen background.
        if mode == "normalize":
            face        = "#dddde2"     # light grey panel
            trace       = "#102040"     # near-black trace
            base_line   = "#404060"
            fill_above  = "#5588cc"
            fill_below  = "#cc6688"
            axis_fg     = FG
            grid_colour = "#a0a0b0"
            grid_alpha  = 1.0
            spine_col   = axis_fg
        else:   # subtract — dark theme matching the rest of the dialog
            face        = PANEL
            trace       = CORR_COLOUR
            base_line   = "#888899"
            fill_above  = CORR_COLOUR
            fill_below  = "#cc6688"
            axis_fg     = FG
            grid_colour = GRID
            grid_alpha  = 1.0
            spine_col   = SPINE

        ax.set_facecolor(face)
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_col)

        # Baseline reference.
        ref = 0.0 if mode == "subtract" else 1.0
        ax.axhline(ref, color=base_line, linewidth=0.6,
                   linestyle="--", alpha=0.85, zorder=1)

        ax.plot(wls, corrected, color=trace, linewidth=0.9, zorder=3)
        ax.fill_between(wls, ref, corrected,
                        where=np.isfinite(corrected) & (corrected > ref),
                        color=fill_above, alpha=0.18, linewidth=0, zorder=2)
        ax.fill_between(wls, ref, corrected,
                        where=np.isfinite(corrected) & (corrected < ref),
                        color=fill_below, alpha=0.18, linewidth=0, zorder=2)

        ax.set_xlim(p["sp_min"], p["sp_max"])
        ax.set_ylabel("Flux − cont." if mode == "subtract" else "Flux / cont.",
                      fontsize=10, color=axis_fg)
        ax.set_xlabel("Wavelength (Å)", fontsize=11, color=axis_fg)
        ax.tick_params(labelsize=9, colors=axis_fg)
        ax.grid(True, color=grid_colour, linewidth=0.5, zorder=0,
                alpha=grid_alpha)

        title = ("Continuum-subtracted  (flux − continuum)"
                 if mode == "subtract"
                 else "Continuum-normalised  (flux ÷ continuum)")
        ax.set_title(title, fontsize=9, color=axis_fg, pad=6, loc="left")

    def _render_luminance_strip(self, wls, norm_int, p, is_bottom):
        """
        Draw the luminance strip on self.ax_lum.

        ``is_bottom`` is True when the strip is the bottom-most panel
        (no corrected panel below it).  Only the bottom-most panel
        carries the wavelength x-tick labels and "Wavelength (Å)"
        title — otherwise they'd duplicate the corrected panel's.
        """
        ax = self.ax_lum

        valid_mask = np.isfinite(norm_int)
        if np.any(valid_mask):
            valid_ints = norm_int[valid_mask]
            int_min = np.min(valid_ints)
            int_max = np.max(valid_ints)
            if int_max > int_min:
                norm_int_normalized = (norm_int - int_min) / (int_max - int_min)
            else:
                norm_int_normalized = np.zeros_like(norm_int)
            norm_int_normalized = np.clip(norm_int_normalized, 0, 1)
        else:
            norm_int_normalized = np.zeros_like(norm_int)

        norm_int_normalized = np.nan_to_num(norm_int_normalized, nan=0.0)
        luminance_strip = norm_int_normalized.reshape(1, -1)

        # imshow spreads its samples evenly across ``extent``, but ``wls``
        # is not evenly spaced — the dispersion polynomial puts a different
        # number of Å in each pixel — and it spans wls[0]..wls[-1] rather
        # than the whole display window.  Stretching the samples across
        # [sp_min, sp_max] therefore slides every feature away from where
        # the curve above draws it, by tens of Å near the middle of a cubic
        # solution: enough to park Hα's band beside its own peak.
        # Resampling onto a uniform axis makes imshow's assumption true,
        # and taking the extent from the data (less half a cell, so sample
        # centres land on their own wavelengths) keeps the two panels
        # registered.  wls is strictly increasing — validate_dispersion_poly
        # rejects any fit that is not — so np.interp is safe here.
        extent = (p["sp_min"], p["sp_max"], 0, 1)
        if len(wls) >= 2:
            uniform = np.linspace(wls[0], wls[-1], len(wls))
            luminance_strip = np.interp(
                uniform, wls, norm_int_normalized).reshape(1, -1)
            half_cell = (wls[-1] - wls[0]) / (2 * (len(wls) - 1))
            extent = (wls[0] - half_cell, wls[-1] + half_cell, 0, 1)

        self.lum_img = ax.imshow(
            luminance_strip,
            aspect='auto',
            extent=extent,
            cmap='gray',
            vmin=0,
            vmax=1,
            interpolation='bilinear',
        )

        ax.set_xlim(p["sp_min"], p["sp_max"])
        ax.set_ylim(0.2, 0.8)
        ax.set_yticks([])
        ax.set_ylabel("Intensity", fontsize=9, color=FG, labelpad=2)
        ax.yaxis.set_label_position("left")
        if is_bottom:
            ax.set_xlabel("Wavelength (Å)", fontsize=11, color=FG)
            ax.tick_params(labelsize=9, colors=FG)
        else:
            # Strip is sandwiched between data panels — suppress its
            # tick labels so they don't duplicate the corrected panel's.
            ax.set_xlabel("")
            ax.tick_params(labelsize=9, colors=FG, labelbottom=False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(SPINE)
            spine.set_linewidth(0.5)

    def _render_empty(self, message):
        """Render a centred placeholder message — used when data is missing."""
        # Rebuild to the 2-row layout so the placeholder doesn't share
        # vertical space with a phantom corrected panel.
        self._build_axes(have_continuum=False)
        self.ax.cla()
        self.ax.set_facecolor(PANEL)
        self.ax.text(
            0.5, 0.5, message,
            ha="center", va="center",
            color="#e94560", fontsize=12,
            transform=self.ax.transAxes,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.ax_lum.cla()
        self.ax_lum.set_facecolor(PANEL)
        self.ax_lum.set_xticks([])
        self.ax_lum.set_yticks([])

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _anchor_arrays(self):
        """
        Return (wavelengths, fluxes) ndarrays from parent's anchor list,
        sorted by wavelength.  The spline requires strictly-increasing
        x, which sorting guarantees.
        """
        arr = np.asarray(self.parent.continuum_anchors, dtype=float)
        order = np.argsort(arr[:, 0])
        return arr[order, 0], arr[order, 1]

    def _target_name(self):
        """Last path component of the target FITS file, '/' and '\\' safe."""
        return (self.parent.v_target.get()
                .strip()
                .split("/")[-1]
                .split(chr(92))[-1])

    def _window_title(self):
        return f"Calibrated spectrum — {self._target_name()}"

    def _close(self):
        """
        Tear down the figure and the dialog, and clear the parent's
        single-instance reference so the next open spawns a fresh one.
        """
        # No plt.close(): a Figure() has no pyplot manager to tear down.
        if getattr(self.parent, "_full_spec_dialog", None) is self:
            self.parent._full_spec_dialog = None
        self.destroy()
