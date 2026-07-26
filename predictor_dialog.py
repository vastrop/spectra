"""
predictor_dialog.py
===================
EXPERIMENTAL standalone viewer for the Pickles spectral-type predictor.

Shows the same template-match panel as the continuum dialog's 4th plot —
the observed band shape vs the best-matching library templates — but as
its own window, so the spectral-type check is reachable without opening
the continuum workflow.

Parent API contract
-------------------
The dialog expects ``parent`` to expose:

  - ``_calibrated_wls``  : 1D ndarray — wavelength axis (Å), or None.
  - ``_calibrated_flux`` : 1D ndarray — response-corrected flux, same
        length as _calibrated_wls.

Design
------
* Modeless ``Toplevel`` — sits alongside the main window; single
  instance per parent, held in ``parent._predictor_dialog``.
* ``refresh()`` re-pulls the spectrum from the parent and re-runs the
  match, so re-clicking the launch button updates an open window.
* The reference library is loaded once per dialog lifetime — it is
  static on disk; only the observed spectrum changes.
"""

from __future__ import annotations

import tkinter as tk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from spectrum_core import load_reference_library, match_reference_templates
from reference_library_viewer import _default_library_dir
import tooltip_help as tt


# Palette — matches the rest of the application's chrome.
BG       = "#0e1014"
PANEL    = "#0f0f1a"
SPINE    = "#262c37"
FG       = "#aab2c0"
ACC      = "#e0c46c"
CLOSE_BG = "#e0c46c"
ERR_FG   = "#e94560"


class PredictorDialog(tk.Toplevel):
    """Pickles template match (spectral-type prediction) as a window."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.configure(bg=BG)
        self.geometry("1280x720")
        self.minsize(800, 450)
        self.transient(parent)
        # No grab_set() — modeless.

        self.title("Predictor — spectral-type match (experimental)")

        tk.Label(
            self,
            text=("EXPERIMENTAL — ranks Pickles library templates by "
                  "whole-band shape match (log-flux χ², Balmer cores and "
                  "telluric bands excluded). Depends on a good response "
                  "correction; treat the result as a suggestion."),
            bg=BG, fg=ACC, font=("Courier New", 9), pady=6,
            justify="left", anchor="w", wraplength=1200,
        ).pack(side="top", fill="x", padx=10)

        # Figure(), not plt.figure(): pyplot gives an embedded figure a manager,
        # i.e. a withdrawn second tk.Tk() root.  See spectrum_explorer.py:35.
        self.fig = Figure(facecolor=BG)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        tb_frame = tk.Frame(self, bg=BG)
        tb_frame.pack(side="top", fill="x")
        NavigationToolbar2Tk(self.canvas, tb_frame)

        _close = tk.Button(
            self, text="Close", bg=CLOSE_BG, fg="#1a1a1a",
            font=("Courier New", 10, "bold"), relief="flat", padx=14,
            command=self._close,
        )
        _close.pack(side="bottom", pady=8)
        tt.attach(_close, "PredictorDialog", "Close")

        self.protocol("WM_DELETE_WINDOW", self._close)

        # Library is static on disk — load once per dialog lifetime.
        self._library = load_reference_library(_default_library_dir())

        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self):
        """Re-pull the calibrated spectrum from the parent and redraw."""
        self.ax.cla()
        self.ax.set_facecolor(PANEL)
        self.ax.tick_params(labelsize=8, colors=FG)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(SPINE)

        wls = getattr(self.parent, "_calibrated_wls", None)
        flux = getattr(self.parent, "_calibrated_flux", None)
        if (wls is None or flux is None or len(wls) == 0
                or len(flux) != len(wls)):
            self._placeholder("No calibrated spectrum available.\n"
                              "Run extraction in the main window first.")
            return
        if not self._library:
            self._placeholder("ReferenceLibrary not found — "
                              "no templates to match against.")
            return

        wls = np.asarray(wls, dtype=float)
        flux = np.asarray(flux, dtype=float)
        match = match_reference_templates(wls, flux, self._library)
        if match is None:
            self._placeholder("Template match failed on this spectrum.")
            return

        self.ax.set_title("Pickles template match (spectral-type check)",
                          fontsize=9, color=FG, pad=4)
        self.ax.set_xlabel("Wavelength (Å)", fontsize=9, color=FG)
        self.ax.set_ylabel("Flux", fontsize=9, color=FG)

        grid, obs, mask = match["grid"], match["obs"], match["mask"]
        # Observed: faint over the full band, solid where actually
        # compared (Balmer cores / telluric bands are excluded).
        self.ax.plot(grid, obs, color="#c0c0d0", linewidth=0.7, alpha=0.35)
        self.ax.plot(grid, np.where(mask, obs, np.nan), color="#c0c0d0",
                     linewidth=0.9, label="observed (smoothed)")

        styles = (("#60d090", "-"), ("#f0c040", "--"), ("#88e0ff", ":"))
        for r, (color, ls) in zip(match["ranked"][:3], styles):
            twl, tfx = self._library[r["name"]]
            tmpl = r["scale"] * np.interp(grid, twl, tfx)
            self.ax.plot(grid, tmpl, color=color, linewidth=1.0,
                         linestyle=ls, alpha=0.9,
                         label=f"{r['name']}  (rms {r['rms']:.3f})")

        self.ax.legend(loc="lower right", fontsize=8,
                       facecolor=PANEL, edgecolor=SPINE,
                       labelcolor="#c0c0d0")
        if match["ranked"]:
            self.title(f"Predictor — best match: "
                       f"{match['ranked'][0]['name']} (experimental)")

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _placeholder(self, message):
        self.ax.text(0.5, 0.5, message,
                     ha="center", va="center", color=ERR_FG, fontsize=11,
                     transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw_idle()

    def _close(self):
        # No plt.close(): a Figure() has no pyplot manager to tear down.
        if getattr(self.parent, "_predictor_dialog", None) is self:
            self.parent._predictor_dialog = None
        self.destroy()
