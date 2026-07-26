"""
first_run_dialog.py
===================
Welcome dialog: get an approximate linear dispersion out of a new user
before they hit the pipeline with someone else's number.

Why this exists
---------------
Å/px is the one parameter a new user cannot leave alone.  It sizes the
extraction strip (``compute_spectrum_width``), so a wrong value puts the
spectrum half off the end of the strip, and it seeds the Balmer scale
search, so a value far from the truth leaves the automatic calibration
with nothing to lock onto.  Both failures look like a broken program
rather than a misconfiguration, and any hard-coded default is one
particular rig's value: plausibly right for a stranger, which is worse
than obviously wrong.

It only has to be approximate.  ``suggest_dispersion_nodes`` searches ±25%
around it and is prior-independent well beyond that, so "in the right
ballpark" is genuinely enough — which is what the welcome text promises.

Parent API contract
-------------------
The dialog expects ``parent`` to expose:

  - ``v_dispersion`` : tk.StringVar — the Å/px field it writes on accept.
  - ``_ui_state_update(key, value)`` : persist a per-machine UI value.
  - ``_ui_state_get(key, default)``  : read one back.

Design
------
* Modal ``Toplevel`` (``grab_set`` + ``wait_window``): the value is needed
  before anything else is meaningful.
* Pre-filled, so OK always produces something usable.  There is no Skip —
  a modal a new user cannot satisfy is a modal they close, which lands
  them back in exactly the broken-looking state this exists to prevent.
  Closing the window accepts whatever is currently shown.
* Geometry inputs persist in the UI-state dotfile, not the analysis config:
  they are per-rig facts that only produce Å/px, and Å/px is already saved.
  That keeps the config format (and its loader) untouched.
* Reusable as a plain calculator — the explorer's "calc" button next to the
  Å/px field opens the same dialog with ``first_run=False``.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from spectrum_core import dispersion_from_geometry
import tooltip_help as tt


# Palette — matches the rest of the application's chrome.
BG    = "#0e1014"
FG    = "#aab2c0"
ACC   = "#e0c46c"
ERR   = "#e94560"

# Groove densities of the two gratings this toolkit is built around.  Custom
# is not a special case — the field stays editable, this is just a shortcut.
GRATINGS = (("Star Analyser 100", 100.0),
            ("Star Analyser 200", 200.0))

# Pre-fill: an SA100 at 50 mm with 2.9 µm pixels (IMX585 class) → 5.8 Å/px.
# 50 mm is the realistic spacing for a Star Analyser screwed into a typical
# imaging train, so a user who accepts the form unchanged already lands in
# the right ballpark rather than a factor of two out.  Every field is visible
# and editable, so this is a starting point rather than a hidden assumption.
DEFAULT_GEOMETRY = {"lines_mm": "100", "distance_mm": "50", "pixel_um": "2.9"}

_UI_KEY = "grating_geometry"


class FirstRunDialog(tk.Toplevel):
    """Approximate-dispersion prompt with a grating geometry calculator."""

    def __init__(self, parent, first_run=True):
        super().__init__(parent)
        self.parent = parent
        self._first_run = first_run
        self.result = None

        self.configure(bg=BG)
        self.title("Welcome — initial dispersion"
                   if first_run else "Dispersion calculator")
        self.transient(parent)
        self.resizable(False, False)

        saved = {}
        try:
            saved = dict(parent._ui_state_get(_UI_KEY, {}) or {})
        except Exception:
            saved = {}
        geo = {**DEFAULT_GEOMETRY, **saved}

        self.v_lines = tk.StringVar(value=str(geo["lines_mm"]))
        self.v_dist = tk.StringVar(value=str(geo["distance_mm"]))
        self.v_pixel = tk.StringVar(value=str(geo["pixel_um"]))
        self.v_result = tk.StringVar()

        pad = {"padx": 12}
        row = 0

        if first_run:
            tk.Label(
                self,
                text=("Welcome — before you start, the toolkit needs your "
                      "approximate linear dispersion in Å per pixel."),
                bg=BG, fg=ACC, font=("Courier New", 10, "bold"),
                justify="left", anchor="w", wraplength=430,
            ).grid(row=row, column=0, columnspan=2, sticky="w",
                   pady=(12, 2), **pad)
            row += 1
            tk.Label(
                self,
                text=("Don't worry about getting it exact. You can change it "
                      "later, and the wavelength calibration will replace it "
                      "with a proper non-linear solution — but starting "
                      "roughly right is what lets the automatic features "
                      "find your spectrum at all."),
                bg=BG, fg=FG, font=("Courier New", 9),
                justify="left", anchor="w", wraplength=430,
            ).grid(row=row, column=0, columnspan=2, sticky="w",
                   pady=(0, 8), **pad)
            row += 1

        ttk.Separator(self, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6, **pad)
        row += 1

        tk.Label(self, text="Work it out from your setup:", bg=BG, fg=ACC,
                 font=("Courier New", 9), anchor="w").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4), **pad)
        row += 1

        # Grating shortcut buttons — they only fill the lines/mm field.
        btns = tk.Frame(self, bg=BG)
        btns.grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        for label, lines in GRATINGS:
            b = ttk.Button(btns, text=label, width=18,
                           command=lambda v=lines: self._set_lines(v))
            b.pack(side="left", padx=(0, 6))
            tt.attach(b, "FirstRunDialog", "grating_preset")
        row += 1

        row = self._entry_row(row, "Grating", self.v_lines, "lines/mm",
                              "lines_mm", pad)
        row = self._entry_row(row, "Grating→sensor", self.v_dist, "mm",
                              "distance_mm", pad)
        row = self._entry_row(row, "Pixel size", self.v_pixel, "µm",
                              "pixel_um", pad)

        tk.Label(self,
                 text=("Distance is grating to SENSOR, not to the filter "
                       "thread — that mix-up is the usual reason this comes "
                       "out a few percent off. A prism/grism on the grating "
                       "does not change this figure meaningfully."),
                 bg=BG, fg=FG, font=("Courier New", 8), justify="left",
                 anchor="w", wraplength=430).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 6), **pad)
        row += 1

        ttk.Separator(self, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6, **pad)
        row += 1

        res = tk.Frame(self, bg=BG)
        res.grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        tk.Label(res, text="Dispersion", bg=BG, fg=ACC,
                 font=("Courier New", 10, "bold"), width=15,
                 anchor="w").pack(side="left")
        e = ttk.Entry(res, textvariable=self.v_result, width=10)
        e.pack(side="left", padx=(4, 4))
        tt.attach(e, "FirstRunDialog", "result")
        tk.Label(res, text="Å / px", bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(side="left")
        row += 1

        self._warn = tk.Label(self, text="", bg=BG, fg=ERR,
                              font=("Courier New", 8), justify="left",
                              anchor="w", wraplength=430)
        self._warn.grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        bar = tk.Frame(self, bg=BG)
        bar.grid(row=row, column=0, columnspan=2, sticky="e",
                 pady=(10, 12), **pad)
        ok = ttk.Button(bar, text="Use this value", command=self._accept)
        ok.pack(side="right")
        tt.attach(ok, "FirstRunDialog", "accept")

        # Geometry edits drive the result field.  Typing directly into the
        # result is allowed and simply is not fed back — a user who already
        # knows their Å/px should not have to invent a geometry to enter it.
        for var in (self.v_lines, self.v_dist, self.v_pixel):
            var.trace_add("write", self._recompute)
        self._recompute()

        self.bind("<Return>", lambda _e: self._accept())
        self.protocol("WM_DELETE_WINDOW", self._accept)
        self.update_idletasks()
        self._centre_on_parent()
        # grab_set raises if the window is not yet viewable, and a grab held
        # by an invisible window freezes the whole application with nothing
        # on screen to dismiss.  The caller is responsible for only opening
        # this over a mapped parent; this is the belt to that braces.  A
        # modeless dialog is a far better failure than a headless app.
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass
        e.focus_set()

    # ------------------------------------------------------------------

    def _entry_row(self, row, label, var, unit, tip_key, pad):
        frm = tk.Frame(self, bg=BG)
        frm.grid(row=row, column=0, columnspan=2, sticky="w", pady=1, **pad)
        tk.Label(frm, text=label, bg=BG, fg=FG, font=("Courier New", 9),
                 width=15, anchor="w").pack(side="left")
        ent = ttk.Entry(frm, textvariable=var, width=10)
        ent.pack(side="left", padx=(4, 4))
        tt.attach(ent, "FirstRunDialog", tip_key)
        tk.Label(frm, text=unit, bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(side="left")
        return row + 1

    def _set_lines(self, value):
        self.v_lines.set(f"{value:g}")

    def _centre_on_parent(self):
        try:
            px, py = self.parent.winfo_rootx(), self.parent.winfo_rooty()
            pw, ph = self.parent.winfo_width(), self.parent.winfo_height()
            w, h = self.winfo_width(), self.winfo_height()
            self.geometry(f"+{px + max(0, (pw - w) // 2)}"
                          f"+{py + max(0, (ph - h) // 3)}")
        except Exception:
            pass  # placement is cosmetic; never fail the dialog over it

    def _recompute(self, *_args):
        disp = dispersion_from_geometry(self.v_lines.get(), self.v_dist.get(),
                                        self.v_pixel.get())
        if disp != disp:            # NaN — incomplete or nonsense input
            self._warn.configure(text="Enter three positive numbers to "
                                      "compute a dispersion.")
            return
        self._warn.configure(text="")
        self.v_result.set(f"{disp:.2f}")

    def _accept(self):
        try:
            value = float(self.v_result.get())
        except (TypeError, ValueError):
            value = None
        if value is None or not (value > 0):
            self._warn.configure(text="Dispersion must be a positive number.")
            return
        self.result = value
        try:
            self.parent.v_dispersion.set(f"{value:g}")
            # Only remember geometry that actually computes.  A half-filled
            # form would otherwise come back as a blank field plus a warning
            # next time, which reads as the dialog having lost the entry.
            geo = {"lines_mm": self.v_lines.get().strip(),
                   "distance_mm": self.v_dist.get().strip(),
                   "pixel_um": self.v_pixel.get().strip()}
            if dispersion_from_geometry(*geo.values()) > 0:
                self.parent._ui_state_update(_UI_KEY, geo)
        except Exception:
            pass  # the value is returned regardless; persistence is a bonus
        self.grab_release()
        self.destroy()


def _selfcheck():
    """The formula, on cases with independently known answers."""
    from spectrum_core import dispersion_from_geometry as d

    # SA100, 100 mm, 3.76 µm — the textbook Star Analyser example.
    assert abs(d(100, 100, 3.76) - 3.76) < 1e-9, d(100, 100, 3.76)
    # Halving the groove density doubles the dispersion; doubling the
    # distance halves it; doubling the pixel doubles it.
    assert abs(d(50, 100, 3.76) - 7.52) < 1e-9
    assert abs(d(100, 200, 3.76) - 1.88) < 1e-9
    assert abs(d(100, 100, 7.52) - 7.52) < 1e-9
    # Inverting for distance reproduces the input.
    assert abs(d(200, 1.0e4 * 3.76 / (200 * 7.7), 3.76) - 7.7) < 1e-9
    # Garbage in → NaN, never an exception or a bogus number.
    for bad in ((0, 100, 3.76), (100, -1, 3.76), (100, 100, 0),
                ("", 100, 3.76), (None, 100, 3.76), (100, "abc", 3.76)):
        got = d(*bad)
        assert got != got, (bad, got)
    print("first_run_dialog self-check OK")


if __name__ == "__main__":
    _selfcheck()
