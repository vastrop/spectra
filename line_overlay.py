"""
line_overlay.py
===============
Presentation-only spectral-line overlay: a checkable tree of the
``wavelength.py`` line library, drawn additively onto one Axes.

Self-contained on purpose.  It owns its selection, reads nothing but
``SPECTRAL_LINES``, and draws through ``spectrum_core``.  It knows
nothing of the explorer, its parent window, or the spectra database, so
the same widget mounts in the full-spectrum viewer today and in the DB
browser when the viewer moves there.

The selection is deliberately ephemeral — it dies with the window, and
nothing is written to any config.  These are per-target presentation
marks; the next star wants a different set.  The window's own
reference-line toggles remain the persistent, analysis-side choice, and
this overlay never writes back to them.

Drawn lines are solid where the catalogue groups are dashed, so which
mark came from where stays readable when both are on.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from spectrum_core import plot_reference_lines
from wavelength import SPECTRAL_LINES

# Palette — matches the viewer's chrome (full_spectrum_viewer.py).
BG      = "#0e1014"
PANEL   = "#0f0f1a"
SPINE   = "#262c37"
FG      = "#aab2c0"
ACC     = "#f0c040"
SELECT  = "#2b3a55"


def build_index():
    """
    Reshape SPECTRAL_LINES into the tree's nesting.

    Returns ``{group: {"colour": str, "subs": {subgroup|None: [(wl, label)]}}}``.
    A line with no fourth element lands under the ``None`` subgroup and
    hangs directly off its group in the tree.  Insertion order is kept,
    so the tree reads in the order the catalogue is written.
    """
    index = {}
    for name, group in SPECTRAL_LINES.items():
        subs = {}
        for wl, label, _qp, *rest in group["lines"]:
            subs.setdefault(rest[0] if rest else None, []).append((wl, label))
        index[name] = {"colour": group["colour"], "subs": subs}
    return index


def lines_to_draw(index, selected, skip=()):
    """
    Resolve ticked keys into ``{colour: {wavelength: label}}`` for drawing.

    ``selected`` holds ``(group, subgroup, wavelength)`` keys.  Anything
    in ``skip`` is dropped: those wavelengths are already on the plot
    from the window's own group toggles, and drawing them again would
    stack a second label at the same x.  Unknown keys are ignored rather
    than raising — the catalogue can be edited under a live selection.
    """
    skipped = {round(float(w), 3) for w in skip}
    out = {}
    for group, sub, wl in selected:
        if round(float(wl), 3) in skipped:
            continue
        entry = index.get(group)
        if entry is None:
            continue
        label = dict(entry["subs"].get(sub, ())).get(wl)
        if label is not None:
            out.setdefault(entry["colour"], {})[wl] = label
    return out


class LineOverlayPanel(ttk.Frame):
    """
    Narrow column of checkable spectral lines, for mounting beside a plot.

    Parameters
    ----------
    master : tk widget
    on_change : callable or None
        Called with no arguments whenever the ticked set changes.  The
        host passes its own refresh so the plot follows the tree.
    width : int
        Requested column width in pixels.
    """

    CHECKED, PARTIAL, UNCHECKED = "☑", "◪", "☐"

    def __init__(self, master, on_change=None, width=240, **kw):
        kw.setdefault("style", "LineOverlay.TFrame")
        super().__init__(master, **kw)
        self.on_change = on_change
        self.index = build_index()
        self.selected = set()
        self._line_key = {}    # iid -> (group, subgroup, wavelength)
        self._label = {}       # iid -> text without the checkbox glyph

        self._configure_style()
        self._build(width)
        self._refresh_glyphs()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _configure_style(self):
        """
        Dark theme for this tree only.

        Named "LineOverlay.Treeview" rather than plain "Treeview": ttk
        styles are per-interpreter, so configuring the base name here
        would repaint the catalogue browsers and the NINA dialog's star
        list as a side effect.
        """
        style = ttk.Style(self)
        style.configure("LineOverlay.TFrame", background=BG)
        style.configure("LineOverlay.Treeview",
                        background=PANEL, fieldbackground=PANEL,
                        foreground=FG, borderwidth=0,
                        font=("Courier New", 9), rowheight=18)
        style.map("LineOverlay.Treeview",
                  background=[("selected", SELECT)])

    def _build(self, width):
        header = tk.Frame(self, bg=BG)
        header.pack(side="top", fill="x", pady=(0, 4))
        tk.Label(header, text="ANNOTATE",
                 bg=BG, fg=ACC, font=("Courier New", 10, "bold")
                 ).pack(side="left")
        tk.Button(header, text="Clear", command=self.clear,
                  bg=PANEL, fg=FG, relief="flat",
                  font=("Courier New", 8), padx=6, cursor="hand2"
                  ).pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(side="top", fill="both", expand=True)

        self.tree = ttk.Treeview(body, show="tree", selectmode="none",
                                 style="LineOverlay.Treeview")
        bar = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.column("#0", width=width, minwidth=140, stretch=True)

        for group, entry in self.index.items():
            tag = f"c{len(self._label)}"
            self.tree.tag_configure(tag, foreground=entry["colour"])
            gid = self._add(self.tree, "", group, tag)
            for sub, lines in entry["subs"].items():
                # A None subgroup means "no middle level" — the lines hang
                # straight off the group rather than under an empty node.
                parent = gid if sub is None else self._add(self.tree, gid, sub, tag)
                for wl, label in lines:
                    lid = self._add(self.tree, parent, label, tag)
                    self._line_key[lid] = (group, sub, wl)

        self.tree.bind("<Button-1>", self._on_click, add="+")

    def _add(self, tree, parent, label, tag):
        iid = tree.insert(parent, "end", text=label, tags=(tag,), open=False)
        self._label[iid] = label
        return iid

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _lines_under(self, iid):
        """Every line iid at or below ``iid`` (a line is its own leaf)."""
        if iid in self._line_key:
            return [iid]
        out = []
        for kid in self.tree.get_children(iid):
            out.extend(self._lines_under(kid))
        return out

    def _on_click(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return None
        # The expander triangle must still expand; only the row body ticks.
        if self.tree.identify_element(event.x, event.y) == "Treeitem.indicator":
            return None
        self._toggle(iid)
        return "break"

    def _toggle(self, iid):
        """Tick or untick a row; a parent applies all-or-nothing below it."""
        leaves = self._lines_under(iid)
        keys = {self._line_key[l] for l in leaves}
        if keys <= self.selected:
            self.selected -= keys
        else:
            self.selected |= keys
        self._refresh_glyphs()
        if self.on_change is not None:
            self.on_change()

    def clear(self):
        if not self.selected:
            return
        self.selected.clear()
        self._refresh_glyphs()
        if self.on_change is not None:
            self.on_change()

    def _refresh_glyphs(self):
        for iid, label in self._label.items():
            leaves = self._lines_under(iid)
            n = sum(1 for l in leaves if self._line_key[l] in self.selected)
            glyph = (self.CHECKED if n and n == len(leaves)
                     else self.PARTIAL if n else self.UNCHECKED)
            self.tree.item(iid, text=f"{glyph} {label}")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, ax, y_max, skip=(), fontsize=6, occupied=None):
        """
        Draw the ticked lines on ``ax``, whose x-axis must be in Ångström.

        dispersion=1.0 with no polynomial makes plot_reference_lines place
        each line at x = wavelength.  ``occupied`` is the shared label-slot
        list: pass the one the host already used for its catalogue groups
        so the annotations stagger against those instead of over them.
        """
        if occupied is None:
            occupied = []
        for colour, lines in lines_to_draw(self.index, self.selected, skip).items():
            plot_reference_lines(ax, lines, 1.0, y_max, colour=colour,
                                 fontsize=fontsize, linestyle="-",
                                 occupied=occupied)


if __name__ == "__main__":
    # Selection maths and catalogue reshaping — the parts that can be
    # wrong without Tk being involved.
    idx = build_index()

    # Flat groups keep a single None subgroup; WN is entirely nested;
    # Balmer mixes the two — its classic five hang off the group while
    # the high series nests, and both shapes must survive together.
    assert list(idx["Oxygen"]["subs"]) == [None], idx["Oxygen"]["subs"]
    assert list(idx["Balmer"]["subs"]) == [None, "high series"], \
        idx["Balmer"]["subs"]
    assert len(idx["Balmer"]["subs"][None]) == 5
    assert None not in idx["Wolf-Rayet WN"]["subs"]
    assert "N III" in idx["Wolf-Rayet WN"]["subs"]
    assert idx["Wolf-Rayet WN"]["subs"]["N III"] == [
        (4634.0, "N III 4634"), (4641.0, "N III 4641")]

    # Every catalogue line survives the reshape.
    for name, group in SPECTRAL_LINES.items():
        n = sum(len(v) for v in idx[name]["subs"].values())
        assert n == len(group["lines"]), (name, n, len(group["lines"]))

    # Resolution: colour comes from the group, label from the catalogue.
    sel = {("Balmer", None, 6562.8), ("Wolf-Rayet WN", "He II", 4685.7)}
    drawn = lines_to_draw(idx, sel)
    assert drawn["#ff6060"] == {6562.8: "Hα 6563"}, drawn
    assert drawn["#7ecfff"] == {4685.7: "He II 4686"}, drawn

    # skip drops a wavelength the host already drew, and only that one.
    drawn = lines_to_draw(idx, sel, skip=[6562.8])
    assert "#ff6060" not in drawn, drawn
    assert drawn["#7ecfff"] == {4685.7: "He II 4686"}, drawn

    # A stale key (catalogue edited under a live selection) is ignored,
    # not raised — the tree outlives any one version of wavelength.py.
    assert lines_to_draw(idx, {("Nonesuch", None, 1.0),
                               ("Balmer", None, 1234.5),
                               ("Balmer", "no such sub", 6562.8)}) == {}

    print("line_overlay self-checks passed")
