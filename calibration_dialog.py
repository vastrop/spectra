"""
calibration_dialog.py
=====================
Modal dialog for adding, editing and deleting non-linear dispersion
calibration nodes.

The dialog runs as a child of ``SpectrumExplorer`` but holds no
hard reference to that class — it interacts only through a small,
explicit attribute/method contract on the parent object.  This keeps
the module standalone and testable in isolation.

Parent API contract
-------------------
The dialog expects ``parent`` to expose:

  - ``column_sums`` : 1D ndarray
        Raw column-summed spectrum for the currently extracted source.
        May be None before the first extraction; the dialog handles that.

  - ``dispersion_nodes`` : list of [pixel, wavelength_Å] pairs
        Read and mutated in place.  The dialog adds, edits and removes
        entries here; the parent persists and re-fits.

  - ``v_dispersion``, ``v_sp_min``, ``v_sp_max`` : tk.StringVar
        Read-only access to the parent's current linear-dispersion and
        wavelength-range settings (used for the secondary x-axis and to
        size the spectrum view).

  - ``get_dispersion_poly()`` -> ndarray | None
        Returns polynomial coefficients (highest degree first) of the
        current dispersion fit, or None when fewer than two nodes exist.

  - ``rebuild_after_node_change()``
        Called after every node mutation so the parent persists the
        nodes to disk and refreshes its own panels.

  - ``_calibrated_wls`` : 1D ndarray or None  (optional)
        Wavelength array (Å) for the calibrated spectrum, set by the
        parent after a successful extraction + response calibration.
        When present, the dialog overlays the response-corrected profile.

  - ``_calibrated_flux`` : 1D ndarray or None  (optional)
        Normalised flux values matching ``_calibrated_wls``.  Used together
        with ``_calibrated_wls`` for the calibrated overlay; ignored if either
        is None.

Fallback constants
------------------
DISPERSION_FALLBACK, SP_RANGE_MIN_FALLBACK and SP_RANGE_MAX_FALLBACK
mirror the corresponding ``DEFAULTS`` entries in ``spectrum_explorer.py``.
They are only used when the parent's StringVars contain unparsable text,
which should be rare in practice.  Keep these in sync with ``DEFAULTS``
if the application-wide defaults ever change.
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402  (style only — never plt.figure)
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402

from spectrum_core import (pixels_to_wavelengths, snap_to_peak, normalize_flux,
                           suggest_dispersion_nodes)  # noqa: E402

from wavelength import SPECTRAL_LINES  # noqa: E402
import tooltip_help as tt  # noqa: E402


# Fallback values matching DEFAULTS in spectrum_explorer.py — used only
# when the parent's StringVar.get() returns text that won't parse to a
# number.  Update alongside DEFAULTS if the app-wide defaults change.
DISPERSION_FALLBACK = 7.7
SP_RANGE_MIN_FALLBACK = 4000
SP_RANGE_MAX_FALLBACK = 8000


class CalibrationDialog(tk.Toplevel):
    """
    Modal dialog providing a pixel-accurate workspace for adding, editing
    and deleting dispersion-calibration nodes.

    The dialog operates directly on the parent ``SpectrumExplorer``'s
    ``dispersion_nodes`` list and calls ``parent.rebuild_after_node_change()``
    after every mutation so the main window reflects the new fit
    immediately.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Dispersion calibration")
        self.configure(bg="#0e1014")
        # The inline assignment panel lives in a fixed-height slot (hint when
        # no node is selected, controls when one is), so the window size is
        # constant — no dynamic resize on node selection.
        self.geometry("1920x1080")
        self.minsize(960, 540)
        self.transient(parent)
        self.grab_set()

        self._inline_frame = None  # bottom assignment panel
        self._click_pixel = None  # pixel chosen by current click
        self._click_marker = None  # vertical line on the spectrum
        self._ax_cal = None  # twin y-axis for calibrated overlay
        self._snap_window = 10  # ±px snap search range
        self._snap_sigma = 4.0  # Gaussian weight width (px)

        self._build_ui()
        self._refresh_nodes_listbox()
        self._update_fit_info()

        # Wire click handler
        self._click_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_spectrum_click)

        self.bind("<Escape>", lambda e: self._close())
        self.protocol("WM_DELETE_WINDOW", self._close)

        # Defer the first spectrum draw until after Tk has finished laying
        # out this Toplevel.  Drawing inside __init__ paints into a
        # zero-sized canvas on some platforms and leaves the dialog blank.
        self.after_idle(self._draw_spectrum)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        BG, PANEL, FG, ACC, EBG = "#0e1014", "#0f0f1a", "#e6e9ef", "#e0c46c", "#1e232c"

        # Help banner
        help_lbl = tk.Label(
            self,
            text=("Click a feature to add a calibration node — snaps to local "
                  "extremum.   Hold Shift to disable snapping.   Pan/zoom "
                  "with the toolbar."),
            bg=BG, fg=FG, font=("Courier New", 9), pady=6, justify="left",
            anchor="w",
        )
        help_lbl.pack(side="top", fill="x", padx=10)

        # Spectrum canvas at the top
        spec_frame = tk.Frame(self, bg=BG)
        spec_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(4, 0))

        plt.style.use("dark_background")
        # Figure(), not plt.figure(): pyplot gives an embedded figure a manager,
        # i.e. a withdrawn second tk.Tk() root.  See spectrum_explorer.py:35.
        self.fig = Figure(facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL)
        self.ax.tick_params(labelsize=8, colors="#aab2c0")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#262c37")

        self.canvas = FigureCanvasTkAgg(self.fig, master=spec_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        tb = tk.Frame(spec_frame, bg=BG)
        tb.pack(side="top", fill="x")
        NavigationToolbar2Tk(self.canvas, tb)

        # Inline assignment slot — a fixed-height container always packed
        # into the window.  Its inner content swaps between a passive
        # hint (no node selected) and the active assignment controls
        # (node selected).  Permanently packing the slot prevents the
        # spectrum frame (which has expand=True) from grabbing the freed
        # space when the assignment controls are torn down, which is
        # what was causing the spectrum to resize on confirm/cancel.
        ASSIGN_SLOT_H = 220
        self._assign_frame = tk.Frame(self, bg=BG, height=ASSIGN_SLOT_H)
        self._assign_frame.pack(side="top", fill="x", padx=10, pady=(4, 0))
        self._assign_frame.pack_propagate(False)

        # Default content: hint label, replaced when a node is clicked.
        self._show_assign_hint()

        # Bottom: nodes list (left) + fit info (right) + close (far right)
        self._bottom_row = tk.Frame(self, bg=BG)
        self._bottom_row.pack(side="bottom", fill="x", padx=10, pady=10)
        bottom = self._bottom_row

        # --- Nodes list ---
        nodes_col = tk.Frame(bottom, bg=BG)
        nodes_col.pack(side="left", fill="both", expand=True)

        tk.Label(nodes_col, text="Nodes",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")

        lb_frame = tk.Frame(nodes_col, bg=BG)
        lb_frame.pack(fill="both", expand=True)

        self.nodes_listbox = tk.Listbox(
            lb_frame, height=6, bg=EBG, fg=FG,
            font=("Courier New", 9), selectbackground=ACC,
            activestyle="none", borderwidth=0, highlightthickness=0,
        )
        sb = ttk.Scrollbar(lb_frame, orient="vertical",
                           command=self.nodes_listbox.yview)
        self.nodes_listbox.configure(yscrollcommand=sb.set)
        self.nodes_listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        node_btns = tk.Frame(nodes_col, bg=BG)
        node_btns.pack(fill="x", pady=(4, 0))
        tk.Button(node_btns, text="Edit selected", bg="#1e232c", fg="#e6e9ef",
                  font=("Courier New", 9), relief="flat",
                  activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._edit_selected
                  ).pack(side="left", padx=(0, 4))
        tk.Button(node_btns, text="Delete selected", bg="#1e232c", fg="#e6e9ef",
                  font=("Courier New", 9), relief="flat",
                  activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._delete_selected
                  ).pack(side="left", padx=(0, 4))
        tk.Button(node_btns, text="Clear all", bg="#1e232c", fg="#e6e9ef",
                  font=("Courier New", 9), relief="flat",
                  activebackground="#262c37", activeforeground="#e6e9ef",
                  command=self._clear_all
                  ).pack(side="left", padx=(0, 4))
        tk.Button(node_btns, text="Suggest (A-star)", bg=ACC, fg="#1a1a1a",
                  font=("Courier New", 9, "bold"), relief="flat",
                  activebackground="#d0b25a", activeforeground="#1a1a1a",
                  command=self._auto_suggest_nodes
                  ).pack(side="left")
        # --- Fit info ---
        fit_col = tk.Frame(bottom, bg=BG)
        fit_col.pack(side="left", fill="y", padx=(20, 0))

        tk.Label(fit_col, text="Fit",
                 bg=BG, fg=ACC, font=("Courier New", 9, "bold")
                 ).pack(anchor="w")
        self.fit_info_var = tk.StringVar(value="No fit yet.")
        tk.Label(fit_col, textvariable=self.fit_info_var,
                 bg=BG, fg=FG, font=("Courier New", 9),
                 justify="left", anchor="nw"
                 ).pack(anchor="nw", pady=(2, 0))

        # --- Close button ---
        tk.Button(bottom, text="Close", bg=ACC, fg="#1a1a1a",
                  font=("Courier New", 10, "bold"), relief="flat",
                  padx=14, command=self._close
                  ).pack(side="right", anchor="se")

        # Hover help for the static node/close buttons. The wavelength-entry
        # panel is built on demand and tipped in _open_assignment_panel.
        tt.attach_tree(self, "CalibrationDialog")

    # ------------------------------------------------------------------
    # Spectrum drawing
    # ------------------------------------------------------------------

    def _draw_spectrum(self):
        """
        Plot the spectrum on a pixel x-axis.

        Primary trace: raw background-subtracted column sums (always available
        after an extraction).  If the parent has already applied instrument-
        response calibration (``_calibrated_wls`` / ``_calibrated_flux`` are set), a
        second normalised trace is overlaid on a right-hand y-axis so both
        share the same pixel x-axis and click / snap behaviour is unaffected.
        """
        if self._ax_cal is not None:
            try:
                self._ax_cal.remove()
            except Exception:
                pass
            self._ax_cal = None
            
        self._click_marker = None  # stale artist reference, the axes were just cleared
        col_sums = self.parent.column_sums
        self.ax.cla()
        self.ax.set_facecolor("#0f0f1a")
        self.ax.tick_params(labelsize=8, colors="#aab2c0")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#262c37")

        if col_sums is None or len(col_sums) == 0:
            self.ax.text(0.5, 0.5, "No spectrum extracted yet.\n"
                                   "Run extraction in the main window first.",
                         ha="center", va="center", color="#e94560",
                         fontsize=11, transform=self.ax.transAxes)
            self.canvas.draw()
            return

        pixels = np.arange(len(col_sums))
        # Raw curve is the primary focus of this dialog — give it the
        # accent colour and full weight.  The calibrated overlay drops
        # to a muted grey so it provides context without competing.
        self.ax.plot(pixels, col_sums, color="#e0c46c", linewidth=1.0,
                     alpha=0.9, label="Raw")

        # ── Dispersion mapping (linear or polynomial) ────────────────
        try:
            dispersion = float(self.parent.v_dispersion.get())
        except (ValueError, AttributeError):
            dispersion = DISPERSION_FALLBACK
        poly = self.parent.get_dispersion_poly()

        # Pre-compute the monotonic pixel→wavelength grid once so the
        # inverse transform below is fast and shape-safe.  Sort it so we
        # can use ``searchsorted`` regardless of polynomial sign.
        n = len(col_sums)
        grid_px = np.linspace(0.0, n - 1, max(2000, n))
        grid_wl = pixels_to_wavelengths(grid_px, dispersion, poly_coeffs=poly)
        sort_idx = np.argsort(grid_wl)
        wl_sorted = grid_wl[sort_idx]
        px_sorted = grid_px[sort_idx]

        def _wl_to_px_scalar(wl):
            """Single-wavelength inverse — used here for xlim setup."""
            i = int(np.clip(np.searchsorted(wl_sorted, wl),
                            1, len(wl_sorted) - 1))
            wl_lo, wl_hi = wl_sorted[i - 1], wl_sorted[i]
            if wl_hi == wl_lo:
                return float(px_sorted[i - 1])
            frac = (wl - wl_lo) / (wl_hi - wl_lo)
            return float(px_sorted[i - 1] + frac * (px_sorted[i] - px_sorted[i - 1]))

        # ── X-range from main-window sp_min/sp_max ───────────────────
        try:
            sp_min = float(self.parent.v_sp_min.get())
            sp_max = float(self.parent.v_sp_max.get())
        except (ValueError, AttributeError):
            sp_min, sp_max = SP_RANGE_MIN_FALLBACK, SP_RANGE_MAX_FALLBACK

        xlo = _wl_to_px_scalar(sp_min)
        xhi = _wl_to_px_scalar(sp_max)
        if xlo > xhi:
            xlo, xhi = xhi, xlo
        # Clip to actual spectrum bounds.
        xlo = max(0, xlo)
        xhi = min(n - 1, xhi)
        if xhi - xlo < 1:  # degenerate fallback
            xlo, xhi = 0, n - 1
        self.ax.set_xlim(xlo, xhi)

        # Y-range from the visible slice only — keeps the zero-order
        # peak (sitting near pixel 0) from dwarfing the spectrum.
        i_lo = int(np.floor(xlo))
        i_hi = int(np.ceil(xhi)) + 1
        visible = col_sums[i_lo:i_hi]
        if len(visible) == 0:
            visible = col_sums
        ymin = float(np.nanmin(visible))
        ymax = float(np.nanmax(visible))
        if ymax == ymin:
            ymax = ymin + 1.0
        self.ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax * 1.08)

        # Pixel axis lives on top — clicks, snap and node positions all
        # operate in pixel space, so it remains the primary, but
        # wavelengths are more natural to read off at the bottom.
        self.ax.xaxis.tick_top()
        self.ax.xaxis.set_label_position("top")
        self.ax.set_xlabel("Pixel index", fontsize=9, color="#aab2c0")
        self.ax.set_ylabel("Raw flux (counts)", fontsize=9, color="#e0c46c")
        self.ax.grid(True, color="#262c37", linewidth=0.5)

        # ── Secondary x-axis (current-fit wavelength) at bottom ──────

        def px_to_wl(px):
            return pixels_to_wavelengths(px, dispersion, poly_coeffs=poly)

        def wl_to_px(wl):
            # matplotlib calls this with arrays of arbitrary shape
            # (sometimes 0-D, sometimes 2-D, sometimes empty).  Flatten,
            # invert via searchsorted, reshape back.
            wl_arr = np.asarray(wl, dtype=float)
            if wl_arr.size == 0:
                return wl_arr
            flat = wl_arr.ravel()
            idx = np.searchsorted(wl_sorted, flat)
            idx = np.clip(idx, 1, len(wl_sorted) - 1)
            # Linear interpolation between bracketing grid points.
            wl_lo = wl_sorted[idx - 1]
            wl_hi = wl_sorted[idx]
            frac = np.where(wl_hi != wl_lo,
                            (flat - wl_lo) / (wl_hi - wl_lo), 0.0)
            px_out = px_sorted[idx - 1] + frac * (px_sorted[idx] - px_sorted[idx - 1])
            return px_out.reshape(wl_arr.shape)

        sec = self.ax.secondary_xaxis("bottom", functions=(px_to_wl, wl_to_px))
        sec.set_xlabel("Current-fit wavelength (Å)",
                       fontsize=9, color="#aab2c0")
        sec.tick_params(labelsize=8, colors="#aab2c0")

        # ── Calibrated & normalised overlay (right y-axis) ───────────
        # Interpolate the parent's calibrated wavelength→flux onto the
        # pixel grid of the current spectrum so the two traces share the
        # same x-axis.  The parent caches these arrays after every
        # successful extraction; they are None until then.
        cal_wls = getattr(self.parent, "_calibrated_wls", None)
        cal_norm_int = getattr(self.parent, "_calibrated_flux", None)

        if cal_wls is not None and cal_norm_int is not None:
            # Map each pixel to its wavelength under the current dispersion
            # model, then interpolate the calibrated flux at those wavelengths.
            # np.interp extrapolates as the nearest endpoint value outside the
            # calibrated range — those pixels are already NaN in cal_norm_int
            # so finite-only masking below keeps them out of the plot.
            pixel_wls = pixels_to_wavelengths(pixels, dispersion, poly_coeffs=poly)
            cal_on_grid = np.interp(pixel_wls, cal_wls,
                                    cal_norm_int, left=np.nan, right=np.nan)

            # Normalise over the full array so the overlay shares one [0, 1]
            # scale regardless of the visible window — panning doesn't change
            # the curve's shape, only which part of it is shown.
            cal_on_grid_norm = normalize_flux(cal_on_grid, min_val=0.0, max_val=1.0)

            self._ax_cal = self.ax.twinx()
            self._ax_cal.set_facecolor("none")
            # Calibrated overlay is contextual — muted so it doesn't
            # compete with the raw curve for attention.
            self._ax_cal.tick_params(labelsize=7, colors="#6a6a6a")
            self._ax_cal.set_ylabel("Norm. flux (calibrated)",
                                    fontsize=9, color="#6a6a6a")

            # Mask NaNs so matplotlib draws clean gaps rather than
            # connecting across uncalibrated regions.
            finite = np.isfinite(cal_on_grid_norm)
            self._ax_cal.plot(pixels[finite], cal_on_grid_norm[finite],
                              color="#6a6a6a", linewidth=0.7, alpha=0.5,
                              label="Cal.")
            self._ax_cal.set_xlim(xlo, xhi)
            self._ax_cal.set_ylim(-0.05, 1.15)
            # Keep the twin axis from adding its own grid on top of the
            # primary grid.
            self._ax_cal.set_zorder(self.ax.get_zorder() - 1)
            self.ax.set_frame_on(False)

        # ── Existing nodes as gold vertical lines ────────────────────
        for px, wl in self.parent.dispersion_nodes:
            self.ax.axvline(x=px, color="#f0c040", linestyle=":",
                            linewidth=1.0, alpha=0.9)
            self.ax.text(px, ymax * 1.02, f"{wl:.1f}",
                         rotation=90, va="bottom", ha="right",
                         color="#f0c040", fontsize=7)

        # Explicit margins — predictable, doesn't trigger the secondary
        # axis transform during layout the way tight_layout() does.
        self.fig.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.12)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _on_spectrum_click(self, event):
        """Left-click on the spectrum opens the inline assignment panel."""
        if event.button != 1:
            return
        if event.inaxes is not self.ax:
            return
        if event.xdata is None:
            return
        col_sums = self.parent.column_sums
        if col_sums is None or len(col_sums) == 0:
            return

        # Snap unless Shift is held
        shift_held = bool(event.key and "shift" in event.key.lower())
        if shift_held:
            pixel = float(event.xdata)
        else:
            pixel = float(snap_to_peak(col_sums, event.xdata,
                                       half_window=self._snap_window,
                                       sigma=self._snap_sigma))

        self._click_pixel = pixel
        self._draw_click_marker(pixel)
        self._open_assignment_panel(pixel, snapped=not shift_held)

    def _draw_click_marker(self, pixel):
        """Draw or move the vertical accent line marking the current click."""
        if self._click_marker is not None:
            try:
                self._click_marker.remove()
            except Exception:
                pass
        self._click_marker = self.ax.axvline(
            x=pixel, color="#e0c46c", linewidth=1.2, alpha=0.85)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Inline assignment panel
    # ------------------------------------------------------------------
    def _show_assign_hint(self):
        """
        Fill the assignment slot with a passive hint when no node is
        selected.  Keeps the slot visually populated so the layout
        stays constant whether or not a node is being edited.  Future
        guidance text can be added here without changing the layout.
        """
        BG, FG = "#0e1014", "#aab2c0"
        for child in self._assign_frame.winfo_children():
            child.destroy()
        tk.Label(
            self._assign_frame,
            text=("Click a feature on the spectrum to add a calibration "
                  "node.\n"
                  "The wavelength assignment controls will appear here."),
            bg=BG, fg=FG, font=("Courier New", 9),
            justify="center",
        ).place(relx=0.5, rely=0.5, anchor="center")

    def _open_assignment_panel(self, pixel, snapped):
        """Build the bottom panel for entering a wavelength."""
        BG, FG, ACC, EBG = "#0e1014", "#e6e9ef", "#e0c46c", "#1e232c"

        # Tear down any previous panel content
        for child in self._assign_frame.winfo_children():
            child.destroy()

        # The slot is permanently packed (see _build_ui); only its contents
        # are swapped.  The destroy loop above cleared the hint label, and
        # the widgets created below repopulate the slot.

        # Hint wavelength under current dispersion
        try:
            dispersion = float(self.parent.v_dispersion.get())
        except (ValueError, AttributeError):
            dispersion = DISPERSION_FALLBACK
        poly = self.parent.get_dispersion_poly()
        hint_wl = float(pixels_to_wavelengths(
            pixel, dispersion, poly_coeffs=poly))

        snap_note = "snapped" if snapped else "no snap (Shift)"
        header = tk.Label(
            self._assign_frame,
            text=(f"Pixel {pixel:.2f}  ({snap_note})   "
                  f"Current-fit λ ≈ {hint_wl:.1f} Å.  "
                  f"Assign true wavelength:"),
            bg=BG, fg=FG, font=("Courier New", 9, "bold"),
            anchor="w", justify="left",
        )
        header.pack(anchor="w", pady=(2, 4))

        # Entry + confirm/cancel row
        entry_row = tk.Frame(self._assign_frame, bg=BG)
        entry_row.pack(anchor="w", fill="x")

        self._entry_var = tk.StringVar(value=f"{hint_wl:.2f}")
        entry = tk.Entry(entry_row, textvariable=self._entry_var, bg=EBG, fg=FG,
                         insertbackground=FG, font=("Courier New", 11),
                         width=14, relief="flat")
        entry.pack(side="left", padx=(0, 8))
        entry.select_range(0, tk.END)
        entry.focus_set()
        tt.attach(entry, "CalibrationDialog", "wavelength_entry")

        _confirm = tk.Button(entry_row, text="Confirm", bg=ACC, fg="#1a1a1a",
                             font=("Courier New", 9, "bold"), relief="flat",
                             padx=10, command=self._confirm_assignment)
        _confirm.pack(side="left", padx=(0, 4))
        tt.attach(_confirm, "CalibrationDialog", "Confirm")
        _cancel = tk.Button(entry_row, text="Cancel", bg=EBG, fg=FG,
                            font=("Courier New", 9), relief="flat",
                            padx=10, command=self._cancel_assignment)
        _cancel.pack(side="left")
        tt.attach(_cancel, "CalibrationDialog", "Cancel")

        # Quick-pick — one row per element group, group-coloured row
        # label.  Only lines flagged quickpick=True in SPECTRAL_LINES get
        # a button; groups with no flagged lines are skipped entirely.
        qp_label = tk.Label(self._assign_frame, text="Quick pick:",
                            bg=BG, fg=ACC, font=("Courier New", 8, "bold"))
        qp_label.pack(anchor="w", pady=(6, 2))

        qp_container = tk.Frame(self._assign_frame, bg=BG)
        qp_container.pack(anchor="w", fill="x")

        # Width of the row label column — matches the longest group name
        # ("Atmospheric" = 11 chars) so all first buttons line up.
        LABEL_W = 11

        for group_name, group in SPECTRAL_LINES.items():
            picks = [(wl, label) for wl, label, qp in group["lines"] if qp]
            if not picks:
                continue
            row = tk.Frame(qp_container, bg=BG)
            row.pack(anchor="w", fill="x", pady=1)
            tk.Label(row, text=group_name,
                     bg=BG, fg=group["colour"],
                     font=("Courier New", 8, "bold"),
                     width=LABEL_W, anchor="w"
                     ).pack(side="left", padx=(0, 4))
            for wl, label in picks:
                _qp = tk.Button(row, text=label, bg="#1e232c", fg="#e6e9ef",
                                font=("Courier New", 8), relief="flat",
                                activebackground="#262c37", activeforeground="#e6e9ef",
                                command=lambda w=wl: self._quick_pick(w))
                _qp.pack(side="left", padx=2)
                tt.attach(_qp, "CalibrationDialog", "quick_pick_button")

        # Bind Enter/Escape only while panel is active
        entry.bind("<Return>", lambda e: self._confirm_assignment())
        # "break" stops the event here — without it Escape continues up
        # the bindtags to the Toplevel's <Escape> → _close() binding and
        # cancelling an assignment closes the whole dialog.
        entry.bind("<Escape>",
                   lambda e: (self._cancel_assignment(), "break")[1])
        # No window resize here — the slot is fixed-height, so the
        # spectrum's allocated space stays constant whether the slot
        # shows the hint or the assignment controls.

    def _quick_pick(self, wl):
        self._entry_var.set(f"{wl:.2f}")

    def _confirm_assignment(self):
        try:
            assigned_wl = float(self._entry_var.get())
        except ValueError:
            return  # silently ignore — user can correct
        if self._click_pixel is None:
            return
        # Add (or replace if same pixel within 0.5 px tolerance)
        existing = self.parent.dispersion_nodes
        replaced = False
        for i, (px, _) in enumerate(existing):
            if abs(px - self._click_pixel) < 0.5:
                existing[i] = [self._click_pixel, assigned_wl]
                replaced = True
                break
        if not replaced:
            existing.append([self._click_pixel, assigned_wl])
        existing.sort(key=lambda n: n[0])

        self.parent.rebuild_after_node_change()
        self._refresh_nodes_listbox()
        self._update_fit_info()
        self._tear_down_assignment()
        self._draw_spectrum()

    def _cancel_assignment(self):
        self._tear_down_assignment()
        if self._click_marker is not None:
            try:
                self._click_marker.remove()
            except Exception:
                pass
            self._click_marker = None
            self.canvas.draw_idle()

    def _tear_down_assignment(self):
        # Don't unpack the slot — just swap content back to the passive
        # hint so the layout stays stable.  This is what prevents the
        # spectrum frame (expand=True) from grabbing the freed 220 px
        # of space on confirm/cancel.
        self._show_assign_hint()
        self._click_pixel = None

    # ------------------------------------------------------------------
    # Node list operations
    # ------------------------------------------------------------------

    def _refresh_nodes_listbox(self):
        self.nodes_listbox.delete(0, tk.END)
        for px, wl in self.parent.dispersion_nodes:
            self.nodes_listbox.insert(
                tk.END, f"px {px:8.2f}  →  {wl:8.2f} Å")

    def _delete_selected(self):
        sel = self.nodes_listbox.curselection()
        if not sel:
            return
        del self.parent.dispersion_nodes[sel[0]]
        self.parent.rebuild_after_node_change()
        self._refresh_nodes_listbox()
        self._update_fit_info()
        self._draw_spectrum()

    def _clear_all(self):
        if not self.parent.dispersion_nodes:
            return
        if messagebox.askyesno("Clear all", "Delete all dispersion nodes?",
                               parent=self):
            self.parent.dispersion_nodes = []
            self.parent.rebuild_after_node_change()
            self._refresh_nodes_listbox()
            self._update_fit_info()
            self._draw_spectrum()

    def _auto_suggest_nodes(self):
        """Auto-suggest dispersion nodes from Balmer lines (A-star only).

        Replaces the current node set after a confirmation guard. Routes
        through the same commit path as a manual click, so the suggestions
        appear in the list and on the plot identically.
        """
        col_sums = self.parent.column_sums
        if col_sums is None or len(col_sums) == 0:
            messagebox.showinfo("No spectrum",
                                "Run an extraction first so there is a "
                                "spectrum to analyse.", parent=self)
            return
        try:
            dispersion = float(self.parent.v_dispersion.get())
        except (ValueError, AttributeError):
            dispersion = DISPERSION_FALLBACK

        nodes, info = suggest_dispersion_nodes(col_sums, dispersion)
        if not nodes:
            messagebox.showwarning(
                "Auto-suggest failed",
                "Could not find a clean Balmer pattern.\n\n"
                f"({info.get('error', 'unknown reason')})\n\n"
                "Check that this is an A-type star with visible Balmer "
                "absorption and that the approximate dispersion is set.",
                parent=self)
            return

        # ── Guard: confirm before discarding existing nodes ──
        existing = self.parent.dispersion_nodes
        if existing:
            if not messagebox.askyesno(
                    "Replace nodes?",
                    f"Replace the existing {len(existing)} node(s) with "
                    f"{len(nodes)} auto-suggested node(s)?",
                    parent=self):
                return

        self.parent.dispersion_nodes = [list(nd) for nd in nodes]
        self.parent.rebuild_after_node_change()
        self._refresh_nodes_listbox()
        self._update_fit_info()
        self._draw_spectrum()

        # ── Report quality so a bad solve is immediately visible ──
        max_resid = max(abs(r) for r in info["residuals"])
        tell = "yes" if info["telluric_added"] else "no"
        messagebox.showinfo(
            "Auto-suggest done",
            f"Placed {len(nodes)} node(s).\n"
            f"Balmer lines: {info['n_balmer']}   Telluric 7594: {tell}\n"
            f"Linear dispersion: {info['dispersion']:.3f} Å/px\n"
            f"Max linear residual: {max_resid:.1f} Å\n\n"
            "Review the nodes and delete/edit any that look wrong.",
            parent=self)

    def _edit_selected(self):
        """Re-open the assignment panel for the selected node, keeping pixel."""
        sel = self.nodes_listbox.curselection()
        if not sel:
            return
        px, wl = self.parent.dispersion_nodes[sel[0]]
        # The node is NOT removed here: _confirm_assignment's same-pixel
        # (< 0.5 px) branch replaces it in place, so Cancel/Escape leaves
        # the original node untouched instead of having silently deleted it.
        self._click_pixel = px
        self._draw_click_marker(px)
        self._open_assignment_panel(px, snapped=False)
        # Pre-fill with the current wavelength so a small correction
        # starts from the value being edited.
        self._entry_var.set(f"{wl:.2f}")

    # ------------------------------------------------------------------
    # Fit info
    # ------------------------------------------------------------------

    def _update_fit_info(self):
        nodes = self.parent.dispersion_nodes
        if not nodes:
            self.fit_info_var.set("No nodes yet.")
            return
        if len(nodes) < 2:
            self.fit_info_var.set("Need ≥ 2 nodes for a fit.")
            return
        arr = np.array(nodes)
        pixels, wls = arr[:, 0], arr[:, 1]
        deg = min(len(nodes) - 1, 3)
        try:
            coeffs = np.polyfit(pixels, wls, deg=deg)
            deriv = np.polyder(coeffs)
            disp_at_nodes = np.abs(np.polyval(deriv, pixels))
            fitted = np.polyval(coeffs, pixels)
            rms = float(np.sqrt(np.mean((wls - fitted) ** 2)))
            self.fit_info_var.set(
                f"Poly deg {deg}  |  {len(wls)} nodes\n"
                f"RMS:        {rms:.3f} Å\n"
                f"Disp range: {disp_at_nodes.min():.2f}–"
                f"{disp_at_nodes.max():.2f} Å/px")
        except Exception as e:
            self.fit_info_var.set(f"Fit error: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _close(self):
        try:
            self.canvas.mpl_disconnect(self._click_cid)
        except Exception:
            pass
        # No plt.close(): a Figure() has no pyplot manager to tear down.
        self.grab_release()
        self.destroy()
