"""
response_viewer.py
========================
Standalone viewer for the currently active instrument-response curve.

Reads the file referenced by ``parent.v_response_file`` and plots its
wavelength-vs-factor curve.  Useful as a sanity check during normal
work, and especially when calibrating a new response curve — open
this viewer next to the calibration dialog to compare the curve you
are building against the one currently in use.

Design
------
* Modeless ``Toplevel`` — the user can still interact with the main
  window (and the calibration dialog) while this viewer is open.
* Reads from the parent directly, matching ``FullSpectrumDialog``.
* Single instance per parent — opening while one already exists
  refreshes and focuses it.  The parent holds the reference in
  ``parent._response_viewer``.
"""

from __future__ import annotations

import os
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from spectrum_core import load_calibration_file
import tooltip_help as tt


# Palette — matches the rest of the application's chrome.
BG       = "#0e1014"
PANEL    = "#0f0f1a"
SPINE    = "#262c37"
GRID     = "#262c37"
FG       = "#aab2c0"
CURVE    = "#e94560"   # warm red — the active response curve (data identity)
CLOSE_BG = "#e0c46c"
ERR_FG   = "#e94560"   # reserved for genuine errors, not chrome


class ResponseCurveDialog(tk.Toplevel):
    """
    Display the wavelength-vs-factor curve of the active response file.

    Parameters
    ----------
    parent : SpectrumExplorer
        Used as the Tk parent and as the source of the current
        calibration file path (``parent.v_response_file``).
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.configure(bg=BG)
        self.geometry("1920x1080")
        self.minsize(960, 540)
        self.transient(parent)
        # No grab_set() — modeless, can sit alongside ResponseDialog
        # while the user tunes a new curve.

        self.title("Active instrument response")

        # Matplotlib figure + canvas
        # Figure(), not plt.figure(): pyplot gives an embedded figure a manager,
        # i.e. a withdrawn second tk.Tk() root.  See spectrum_explorer.py:35.
        self.fig = Figure(facecolor=BG)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(SPINE)

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
        tt.attach(_close, "ResponseCurveViewer", "Close")

        self.protocol("WM_DELETE_WINDOW", self._close)

        # Initial draw.  Callers (and the parent's
        # "View active response" handler) can call refresh() later to
        # pick up a changed v_response_file without closing/reopening.
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self):
        """
        Redraw the response curve that the pipeline is actually applying.

        Mirrors the precedence in ``SpectrumExplorer._display_extraction``:
        an in-memory ``parent._response_df_cache`` (the array embedded in a
        loaded config) takes priority over the on-disk file at
        ``parent.v_response_file``.  Without this, after loading a config
        whose .dat has moved or been edited, this window — titled
        "Active" — would show *file not found* or a stale curve that
        differs from the one in use.

        Handles these states gracefully:
        * Embedded array present → plot it, title notes it is embedded.
        * No array, no path      → placeholder text, no plot.
        * No array, path broken  → error message in muted red.
        * No array, path valid   → response curve drawn from the file.
        """
        # 1) Embedded calibration array (config-resident) wins, exactly as
        #    the pipeline prefers it.
        cache = getattr(self.parent, "_response_df_cache", None)
        if cache is not None:
            self._render("embedded from config", cache, embedded=True)
            return

        # 2) Otherwise fall back to the file path.
        path = self.parent.v_response_file.get().strip()
        if not path:
            self._render_empty("No calibration file set.")
            self.title("Active instrument response")
            return

        try:
            cal_df = load_calibration_file(path)
        except FileNotFoundError:
            self._render_empty(f"File not found:\n{path}")
            self.title("Active instrument response — (missing)")
            return
        except Exception as e:
            self._render_empty(f"Could not load file:\n{e}")
            self.title("Active instrument response — (error)")
            return

        self._render(path, cal_df)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, path, cal_df, embedded=False):
        """
        Plot the response curve from the loaded dataframe.

        ``embedded`` flags that ``cal_df`` came from the config-resident
        array rather than a file, in which case ``path`` is a descriptive
        label rather than a real path.
        """
        self.ax.cla()
        self.ax.set_facecolor(PANEL)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(SPINE)

        self.ax.plot(
            cal_df["wavelength"], cal_df["factor"],
            color=CURVE, linewidth=1.2,
        )
        self.ax.set_xlabel("Wavelength (Å)", fontsize=9, color=FG)
        self.ax.set_ylabel("Response factor", fontsize=9, color=FG)
        label = path if embedded else os.path.basename(path)
        self.ax.set_title(
            f"Instrument response: {label}",
            color=FG, fontsize=9, pad=6,
        )
        self.ax.tick_params(labelsize=8, colors=FG)
        self.ax.grid(True, color=GRID, linewidth=0.5)

        self.fig.tight_layout(pad=1.5)
        suffix = "(embedded from config)" if embedded else label
        self.title(f"Active instrument response — {suffix}")
        self.canvas.draw_idle()

    def _render_empty(self, message):
        """Show a centred placeholder message in lieu of a plot."""
        self.ax.cla()
        self.ax.set_facecolor(PANEL)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(SPINE)
        self.ax.text(
            0.5, 0.5, message,
            ha="center", va="center",
            color=ERR_FG, fontsize=11,
            transform=self.ax.transAxes,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _close(self):
        """Tear down and clear the parent's single-instance reference."""
        # No plt.close(): a Figure() has no pyplot manager to tear down.
        if getattr(self.parent, "_response_viewer", None) is self:
            self.parent._response_viewer = None
        self.destroy()