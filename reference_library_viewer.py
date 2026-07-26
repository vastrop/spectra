"""
reference_library_viewer.py
===========================
Browse the local subset of the Pickles reference spectra library.

The library lives in a subfolder named ``ReferenceLibrary`` next to
the application's main script.  Every ``.dat`` file in that folder is
read at first open and cached in memory as ``{stem: (wavelengths,
flux)}``, where ``stem`` is the file name without extension (e.g.
``a0v``, ``b3i``).

The viewer shows a search box at the top and a vertically-scrollable
pane of plot panels below — one panel per matching spectrum.  Typing
in the search box filters the cached spectra by case-insensitive
substring on the stem; an empty search shows the whole library.

Design notes
------------
* Modeless ``Toplevel`` — matches ``ResponseCurveDialog`` and
  ``FullSpectrumDialog``; the user can keep working in the main
  window while browsing references.
* Single instance per parent — opening while one already exists
  refocuses it.  The parent holds the reference in
  ``parent._reference_library_viewer``.
* Reuses ``load_reference_spectrum`` from ``spectrum_core`` so the
  unit-detection and decimal-separator handling stay consistent
  with the response-calibration dialog.
* Each plot is its own small matplotlib figure embedded in a scrollable
  ``tk.Canvas``.  This is easier to size (fixed pixel height per panel,
  so the scrollregion is just ``n × panel_height``) and faster than a
  single very tall multi-subplot figure.
* A small debounce on the search entry keeps things responsive: the
  panel stack is rebuilt ~150 ms after the user stops typing, not on
  every keystroke.
* Panels are *virtualized*: the inner frame is given a fixed height of
  ``n × stride`` so its geometry is known without the panels existing,
  and only the handful of panels near the viewport are actually built
  (as matplotlib figures).  Scrolling reconciles the live set —
  building newly-visible panels and tearing down ones that scrolled
  out.  This keeps the number of open figures small regardless of how
  many spectra match, so there is no "too many open figures" warning
  and no need to cap the result count.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from spectrum_core import load_reference_spectrum
import tooltip_help as tt


# Palette — matches the rest of the application's chrome.
BG       = "#0e1014"
PANEL    = "#0f0f1a"
SPINE    = "#262c37"
GRID     = "#262c37"
FG       = "#aab2c0"
ACC      = "#e0c46c"
CURVE    = "#4ec9b0"   # cool teal — distinguishes from response-curve red
ENTRY_BG = "#1e232c"
ENTRY_FG = "#e6e9ef"

# Fixed pixel height per plot panel — keeps the scrollregion math
# trivial and the layout uniform.
PANEL_HEIGHT_PX = 200

# Vertical gap between panels (formerly the pack pady).  Stride is the
# centre-to-centre distance used for all virtualization arithmetic.
PANEL_GAP_PX = 6
PANEL_STRIDE_PX = PANEL_HEIGHT_PX + PANEL_GAP_PX

# Number of extra panels to keep live just beyond each edge of the
# viewport.  A small buffer means scrolling reveals an already-drawn
# plot rather than a blank that pops in a frame later.
WINDOW_BUFFER = 2

# Debounce delay for the search box, in milliseconds.
SEARCH_DEBOUNCE_MS = 150


def _default_library_dir():
    """Resolve the default ``ReferenceLibrary`` folder.

    Looks next to this module (``__file__``) rather than ``sys.argv[0]``:
    identical in a source checkout, but in a frozen (PyInstaller onedir)
    build argv[0] is the exe at the bundle root while the library ships
    with the modules under ``_internal/`` — argv[0] would miss it.
    Falls back to the current working directory if that can't be
    determined.
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base = os.getcwd()
    return os.path.join(base, "ReferenceLibrary")

# Fallback wavelength range — matches DEFAULTS in spectrum_explorer.py
# (SP_RANGE_MIN / SP_RANGE_MAX).  Used when the parent's StringVars
# aren't readable as floats for whatever reason.
SP_RANGE_MIN_FALLBACK = 4000.0
SP_RANGE_MAX_FALLBACK = 8000.0


def _read_parent_xlim(parent):
    """Read the active wavelength range from the parent.

    Returns ``(xmin, xmax)`` as floats, falling back to the
    application's default 4000–8000 Å if the parent's StringVars are
    missing, unparseable, or inverted.
    """
    try:
        lo = float(parent.v_sp_min.get())
        hi = float(parent.v_sp_max.get())
    except (AttributeError, ValueError, tk.TclError):
        return SP_RANGE_MIN_FALLBACK, SP_RANGE_MAX_FALLBACK
    if hi <= lo:
        return SP_RANGE_MIN_FALLBACK, SP_RANGE_MAX_FALLBACK
    return lo, hi

class ReferenceLibraryDialog(tk.Toplevel):
    """
    Browse the local Pickles reference subset.

    Parameters
    ----------
    parent : SpectrumExplorer
        Used as the Tk parent.  Not otherwise read from — the viewer
        is self-contained.
    library_dir : str or None
        Folder to scan for ``.dat`` files.  If ``None``, uses
        ``ReferenceLibrary`` next to the main script.
    """

    def __init__(self, parent, library_dir=None):
        super().__init__(parent)
        self.parent = parent

        self.configure(bg=BG)
        self.geometry("1920x1080")
        self.minsize(960, 540)
        self.transient(parent)
        # No grab_set() — modeless.

        self.title("Reference spectra library")

        # State
        self._library_dir = library_dir or _default_library_dir()
        self._cache = {}                # {stem: (wl_array, flux_array)}
        self._load_errors = []          # list of (stem, message)
        self.v_query = tk.StringVar()
        self.v_status = tk.StringVar(value="")
        # Cached parent xlim — refreshed at the top of _redraw and
        # read by _build_panel.  Initialised to the application
        # default so it's safe even if _redraw is called via a path
        # that hasn't refreshed it yet, or if a parent-var trace
        # fires before the first manual redraw.
        self._xlim = (SP_RANGE_MIN_FALLBACK, SP_RANGE_MAX_FALLBACK)

        # Virtualization state.  ``_filtered`` is the full ordered list
        # of matches for the current query; ``_live`` maps an index in
        # that list to the widgets currently realised for it.  Only the
        # panels near the viewport are ever built — see _sync_window.
        self._filtered = []             # list of (stem, wl, flux)
        self._live = {}                 # {index: (frame, fig, fig_canvas)}
        self._syncing = False           # re-entrancy guard for _sync_window
        self._sync_pending = False      # request for one more sync pass
        self._placeholder = None        # Label shown in the empty state
        self._debounce_job = None       # after-id for the typing debounce

        self._build_ui()
        tt.attach_tree(self, "ReferenceLibraryViewer")
        self.bind("<Escape>", lambda e: self._close())
        self.protocol("WM_DELETE_WINDOW", self._close)

        # Defer loading until after the window is mapped so the empty
        # frame paints immediately and the user isn't staring at a
        # frozen dialog while the disk is read.
        self.after_idle(self._load_library)

        # Follow the main window's λ min/λ max so panning the range in
        # the parent reflows the panels here.  The trace tokens are kept
        # so _close can detach them: the parent outlives this dialog, and
        # stale traces would fire on a destroyed Toplevel.
        self._xlim_traces = []
        for var in (getattr(self.parent, "v_sp_min", None),
                    getattr(self.parent, "v_sp_max", None)):
            if var is not None:
                token = var.trace_add(
                    "write", lambda *_: self._on_query_changed())
                self._xlim_traces.append((var, token))


    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Top bar: search + folder controls ─────────────────────────
        top = tk.Frame(self, bg=BG)
        top.pack(side="top", fill="x", padx=10, pady=(10, 6))

        tk.Label(top, text="Search:", bg=BG, fg=FG,
                 font=("Courier New", 10)).pack(side="left")

        entry = tk.Entry(
            top, textvariable=self.v_query,
            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG,
            relief="flat", font=("Courier New", 11), width=24,
        )
        entry.pack(side="left", padx=(6, 10))
        # Debounced redraw — see _on_query_changed.
        self.v_query.trace_add("write", lambda *_: self._on_query_changed())
        entry.focus_set()

        tk.Button(
            top, text="Change folder…", bg="#1e232c", fg="#e6e9ef",
            font=("Courier New", 9), relief="flat",
            activebackground="#262c37", activeforeground="#e6e9ef",
            command=self._change_folder,
        ).pack(side="right")

        tk.Button(
            top, text="Reload", bg="#1e232c", fg="#e6e9ef",
            font=("Courier New", 9), relief="flat",
            activebackground="#262c37", activeforeground="#e6e9ef",
            command=self._load_library,
        ).pack(side="right", padx=(0, 6))

        # ── Status line ───────────────────────────────────────────────
        tk.Label(
            self, textvariable=self.v_status,
            bg=BG, fg=FG, font=("Courier New", 9),
            anchor="w", justify="left", wraplength=740,
        ).pack(side="top", fill="x", padx=10, pady=(0, 4))

        # ── Scrollable plot area ──────────────────────────────────────
        # Standard recipe: outer Canvas + Scrollbar, inner Frame placed
        # via create_window.  The inner frame is re-filled on every
        # filter change.
        scroll_holder = tk.Frame(self, bg=BG)
        scroll_holder.pack(side="top", fill="both", expand=True,
                           padx=10, pady=(0, 4))

        self._scroll_canvas = tk.Canvas(
            scroll_holder, bg=BG, highlightthickness=0, borderwidth=0,
        )
        self._scroll_canvas.pack(side="left", fill="both", expand=True)

        self._scrollbar = tk.Scrollbar(
            scroll_holder, orient="vertical",
            command=self._on_scrollbar,
        )
        self._scrollbar.pack(side="right", fill="y")
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)

        self._inner = tk.Frame(self._scroll_canvas, bg=BG)
        self._inner_window = self._scroll_canvas.create_window(
            (0, 0), window=self._inner, anchor="nw",
        )

        # Keep the inner frame's width tied to the canvas's width so
        # plots fill horizontally and never produce a horizontal
        # scrollbar.  Panels are placed with relwidth=1, so they follow.
        # The scrollregion height is *not* derived from the inner
        # frame's natural size any more — with place()-based panels the
        # frame doesn't auto-size, so _redraw sets the scrollregion
        # explicitly to (0, 0, w, n*stride).  A resize also reconciles
        # the live window, since a taller canvas shows more panels.
        self._scroll_canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel — bind only while the pointer is over the dialog
        # to avoid stealing wheel events from other widgets (e.g. the
        # matplotlib toolbars in the main window or in
        # ResponseCurveDialog) when scrolling there.
        # Wheel events bind on this Toplevel (every child widget's
        # bindtags include it), NOT bind_all: the explorer owns the
        # application-wide wheel bindings, and the previous bind_all/
        # unbind_all pair replaced them while the library was open and
        # removed them for good on close — killing side-pane scrolling
        # in the main window until restart.  The explorer's app-level
        # handler routes by ancestry and ignores widgets outside its own
        # panes, so both handlers coexist.  Bindings die with the window;
        # no unbind is needed.
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", lambda e: self._scroll_units(-1))
        self.bind("<Button-5>", lambda e: self._scroll_units(1))

        # ── Close button ──────────────────────────────────────────────
        tk.Button(
            self, text="Close", bg=ACC, fg="#1a1a1a",
            font=("Courier New", 10, "bold"), relief="flat", padx=14,
            command=self._close,
        ).pack(side="bottom", pady=8)

    # ------------------------------------------------------------------
    # Mousewheel handling
    # ------------------------------------------------------------------

    def _on_mousewheel(self, event):
        # Windows / macOS deliver wheel events as <MouseWheel> with
        # event.delta = ±120 per notch (Windows) or smaller floats
        # (macOS).  X11 delivers Button-4 (up) and Button-5 (down),
        # bound separately in __init__.
        # Normalise across platforms: positive delta scrolls up.
        if event.delta:
            step = -1 if event.delta > 0 else 1
            # On macOS delta values are small (±1 or so) — clamp so
            # the user gets a sensible amount of motion per notch.
            self._scroll_units(step)

    def _scroll_units(self, step):
        """Scroll by ``step`` units and reconcile the live panel set."""
        self._scroll_canvas.yview_scroll(step, "units")
        self._sync_window()

    def _on_scrollbar(self, *args):
        """Scrollbar command wrapper that reconciles after the view
        moves.  ``args`` is whatever Tk passes to a yview command
        (``moveto frac`` or ``scroll n what``)."""
        self._scroll_canvas.yview(*args)
        self._sync_window()

    def _on_canvas_configure(self, event):
        """Keep the inner frame's width tied to the canvas, then
        reconcile — a taller canvas exposes more panels."""
        self._scroll_canvas.itemconfigure(self._inner_window, width=event.width)
        self._sync_window()

    # ------------------------------------------------------------------
    # Library loading
    # ------------------------------------------------------------------

    def _load_library(self):
        """Scan ``self._library_dir`` and populate the cache."""
        self._cache.clear()
        self._load_errors.clear()

        folder = self._library_dir
        if not os.path.isdir(folder):
            self.v_status.set(
                f"Folder not found: {folder}.  "
                f"Use Change folder… to point to your local Pickles subset."
            )
            self._redraw()
            return

        try:
            entries = sorted(os.listdir(folder))
        except OSError as e:
            self.v_status.set(f"Could not list folder: {e}")
            self._redraw()
            return

        for name in entries:
            if not name.lower().endswith(".dat"):
                continue
            stem = os.path.splitext(name)[0]
            path = os.path.join(folder, name)
            try:
                wl, fx = load_reference_spectrum(path)
            except Exception as e:
                self._load_errors.append((stem, str(e)))
                continue
            if len(wl) == 0:
                self._load_errors.append((stem, "empty file"))
                continue
            self._cache[stem] = (wl, fx)

        self._redraw()

    def _change_folder(self):
        """Pick a different library folder and reload."""
        new = filedialog.askdirectory(
            title="Select reference library folder",
            initialdir=(self._library_dir
                        if os.path.isdir(self._library_dir)
                        else os.getcwd()),
        )
        if not new:
            return
        self._library_dir = new
        self._load_library()

    # ------------------------------------------------------------------
    # Search-box debounce
    # ------------------------------------------------------------------

    def _on_query_changed(self):
        """Schedule a redraw after the user pauses typing.

        Rebuilding the panel stack involves tearing down matplotlib
        canvases, which is non-trivial.  Debouncing keeps the UI
        smooth when the user types or holds backspace.
        """
        if self._debounce_job is not None:
            try:
                self.after_cancel(self._debounce_job)
            except Exception:
                pass
        self._debounce_job = self.after(SEARCH_DEBOUNCE_MS, self._redraw)

    # ------------------------------------------------------------------
    # Filtering and drawing
    # ------------------------------------------------------------------

    def _matches(self):
        """Return the (sorted) list of (stem, wl, flux) tuples for the
        current query — no length cap applied here so the caller knows
        the full count for the status line.
        """
        q = self.v_query.get().strip().lower()
        items = sorted(self._cache.items())
        if not q:
            picks = items
        else:
            picks = [it for it in items if q in it[0].lower()]
        return [(stem, wl, fx) for stem, (wl, fx) in picks]

    def _clear_panels(self):
        """Tear down all currently live plot panels and the placeholder."""
        for index in list(self._live.keys()):
            self._destroy_panel(index)
        if self._placeholder is not None:
            try:
                self._placeholder.destroy()
            except Exception:
                pass
            self._placeholder = None

    def _destroy_panel(self, index):
        """Dispose of a single live panel by its filtered-list index.

        Order matters to avoid a native access violation (0xC0000005):
        a FigureCanvasTkAgg can have a draw or wheel callback in flight,
        and tearing the figure down out from under it crashes the
        interpreter rather than raising.  Teardown therefore (1) cancels
        any pending idle-draw the Agg backend scheduled, (2) unbinds the
        wheel events attached here, (3) destroys the Tk canvas widget
        explicitly, and only then (4) closes the figure and (5) destroys
        the frame.
        """
        entry = self._live.pop(index, None)
        if entry is None:
            return
        frame, fig, fig_canvas = entry

        # (1) Cancel a pending idle-draw, if the backend left one queued.
        try:
            cb = getattr(fig_canvas, "_idle_draw_id", None)
            if cb is not None:
                fig_canvas.get_tk_widget().after_cancel(cb)
        except Exception:
            pass

        # (2) + (3) Unbind wheel events and destroy the canvas widget
        # before the figure, so no event can dispatch to a freed widget.
        try:
            widget = fig_canvas.get_tk_widget()
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                try:
                    widget.unbind(seq)
                except Exception:
                    pass
            widget.destroy()
        except Exception:
            pass

        # (4) The Figure has no pyplot manager, so there is nothing to close —
        #     dropping the reference with the frame below is the teardown.

        # (5) Destroy the containing frame.
        try:
            frame.destroy()
        except Exception:
            pass

    def _redraw(self):
        """Rebuild the filtered set and reset the virtual viewport.

        This does *not* build every panel — it sizes the inner frame to
        the full match count and then lets _sync_window realise only the
        panels near the (reset-to-top) viewport.
        """
        self._debounce_job = None
        self._xlim = _read_parent_xlim(self.parent)
        self._clear_panels()

        self._filtered = self._matches()
        total = len(self._filtered)

        # ── Status line ───────────────────────────────────────────────
        folder = self._library_dir
        parts = [f"Library: {folder}"]
        parts.append(f"Loaded: {len(self._cache)}")
        if self._load_errors:
            parts.append(f"Skipped: {len(self._load_errors)}")
        q = self.v_query.get().strip()
        if q:
            parts.append(f"Matches: {total}")
        self.v_status.set("    ".join(parts))

        # ── Empty state ───────────────────────────────────────────────
        if total == 0:
            # Collapse the virtual height and show the placeholder.
            self._inner.configure(height=1)
            self._scroll_canvas.configure(scrollregion=(0, 0, 0, 1))
            self._placeholder = tk.Label(
                self._inner,
                text=("No spectra loaded." if not self._cache
                      else "No match for this search."),
                bg=BG, fg=FG, font=("Courier New", 11),
                pady=40,
            )
            self._placeholder.place(x=0, y=0, relwidth=1.0)
            self._scroll_canvas.yview_moveto(0.0)
            return

        # ── Size the virtual canvas and reset to the top ──────────────
        total_height = total * PANEL_STRIDE_PX
        self._inner.configure(height=total_height)
        self._scroll_canvas.configure(scrollregion=(0, 0, 0, total_height))
        # Reset scroll to top so a fresh search starts at the first
        # result rather than wherever the previous scroll left off.
        self._scroll_canvas.yview_moveto(0.0)
        # Realise the panels visible at the top.  update_idletasks first
        # so the canvas reports its true height to _sync_window.
        self._scroll_canvas.update_idletasks()
        self._sync_window()

    def _visible_index_range(self):
        """Return the inclusive ``(first, last)`` filtered indices that
        should be live for the current scroll position, padded by
        ``WINDOW_BUFFER`` on each side and clamped to the match count.
        Returns ``(0, -1)`` (an empty range) when there is nothing to
        show.
        """
        total = len(self._filtered)
        if total == 0:
            return 0, -1

        total_height = total * PANEL_STRIDE_PX
        top_frac, _bottom_frac = self._scroll_canvas.yview()
        offset = top_frac * total_height
        viewport_h = self._scroll_canvas.winfo_height()
        if viewport_h <= 1:                     # not yet mapped
            viewport_h = PANEL_STRIDE_PX

        first = int(offset // PANEL_STRIDE_PX) - WINDOW_BUFFER
        last = int((offset + viewport_h) // PANEL_STRIDE_PX) + WINDOW_BUFFER
        first = max(0, first)
        last = min(total - 1, last)
        return first, last

    def _sync_window(self):
        """Reconcile the live panel set against the visible range:
        build panels that have scrolled into view, destroy ones that
        have scrolled out.  Cheap to call on every scroll tick — it
        only touches the boundary panels.

        Guarded against re-entrancy: a fast wheel burst can deliver the
        next event before this call returns, and nested reconciliation
        could double-destroy a panel.  A call arriving inside an active
        sync is noted and replayed as one more pass when the outer call
        finishes, so the final state still matches the latest scroll
        position.
        """
        if self._syncing:
            self._sync_pending = True
            return
        self._syncing = True
        try:
            while True:
                self._sync_pending = False
                if not self._filtered:
                    break
                first, last = self._visible_index_range()
                wanted = set(range(first, last + 1))

                # Tear down panels no longer wanted.
                for index in list(self._live.keys()):
                    if index not in wanted:
                        self._destroy_panel(index)

                # Build newly-wanted panels.
                for index in wanted:
                    if index not in self._live:
                        self._build_panel(index)

                if not self._sync_pending:
                    break
        finally:
            self._syncing = False

    def _build_panel(self, index):
        """Construct the plot panel for ``self._filtered[index]`` and
        place it at its fixed virtual y-position."""
        stem, wl, fx = self._filtered[index]
        y = index * PANEL_STRIDE_PX

        frame = tk.Frame(self._inner, bg=BG)
        # Absolute placement — the panel's slot is determined entirely
        # by its index, so empty (unbuilt) slots cost nothing.
        frame.place(x=0, y=y, relwidth=1.0, height=PANEL_HEIGHT_PX)

        # Figure(), not plt.figure(): pyplot gives an embedded figure a manager,
        # i.e. a withdrawn second tk.Tk() root — and this one is per panel, so
        # a full library browse would spawn dozens.  See spectrum_explorer.py.
        fig = Figure(facecolor=BG, dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)

        ax.plot(wl, fx, color=CURVE, linewidth=1.0)
        ax.set_title(stem, color=FG, fontsize=9, pad=4, loc="left")
        ax.set_xlabel("Wavelength (Å)", fontsize=8, color=FG)
        ax.set_ylabel("Flux", fontsize=8, color=FG)
        ax.tick_params(labelsize=8, colors=FG)
        ax.grid(True, color=GRID, linewidth=0.5)

        # Match the main window's current λ range, and rescale the y
        # axis to the data that's actually visible — otherwise a
        # strong out-of-window feature would flatten the in-window
        # detail against the y-axis.
        xmin, xmax = self._xlim
        ax.set_xlim(xmin, xmax)
        m = (wl >= xmin) & (wl <= xmax)
        if m.any():
            fxv = fx[m]
            fxv = fxv[np.isfinite(fxv)]
            if fxv.size:
                lo, hi = float(fxv.min()), float(fxv.max())
                pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.05
                ax.set_ylim(lo - pad, hi + pad)

        fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.22)

        fig_canvas = FigureCanvasTkAgg(fig, master=frame)
        widget = fig_canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        # Forward wheel events from the matplotlib canvas widget up
        # to the scroll canvas — otherwise the wheel does nothing
        # when the pointer is directly over a plot.
        widget.bind("<MouseWheel>", self._on_mousewheel)
        widget.bind("<Button-4>", lambda e: self._scroll_units(-1))
        widget.bind("<Button-5>", lambda e: self._scroll_units(1))
        # Draw synchronously rather than via draw_idle(): a fast, large
        # scroll can destroy a panel before a deferred idle-draw fires,
        # and that draw then runs against a freed Tk canvas — a hard
        # access violation (exit 0xC0000005), not a Python exception.
        # The per-panel draw cost is small and bounded (only the few
        # panels near the viewport are ever built).
        fig_canvas.draw()

        self._live[index] = (frame, fig, fig_canvas)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _close(self):
        # Detach traces on the parent's StringVars — they outlive us.
        for var, token in getattr(self, "_xlim_traces", []):
            try:
                var.trace_remove("write", token)
            except Exception:
                pass
        self._xlim_traces = []

        if self._debounce_job is not None:
            try:
                self.after_cancel(self._debounce_job)
            except Exception:
                pass
            self._debounce_job = None
        self._clear_panels()
        if getattr(self.parent, "_reference_library_viewer", None) is self:
            self.parent._reference_library_viewer = None
        self.destroy()
