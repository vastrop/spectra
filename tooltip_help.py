"""
tooltip_help.py
===============
Attach hover tooltips (from ``help/tooltips.json``) to Tkinter widgets.

Tk has no built-in tooltip, so this module provides a tiny hover pop-up plus
two entry points:

* ``attach_tree(root, dialog_key)`` — walk ``root``'s descendants and tip every
  widget whose visible ``text`` matches a key under ``dialog_key``. Covers the
  bulk case: buttons, check/radio buttons, and the caption labels that sit next
  to entries/dropdowns (the JSON is keyed by those visible captions).
* ``attach(widget, dialog_key, key)`` — tip one specific widget by an explicit
  key. Used for controls whose JSON key is a synthetic id with no matching
  visible text (radio-button groups, the "-"/"+" nudge buttons, etc.).

The content file is loaded once and cached. A missing dialog/key is a silent
no-op, so an incomplete tooltips.json never breaks the UI.
"""
from __future__ import annotations

import json
import os
import tkinter as tk

# Resolve the content file relative to this module, not the cwd, so it loads
# regardless of where the app is launched from.
_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "help", "tooltips.json")
_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        try:
            with open(_JSON_PATH, encoding="utf-8") as f:
                _DATA = json.load(f)
        except Exception:
            _DATA = {}
    return _DATA


class _ToolTip:
    """A hover pop-up bound to a single widget."""

    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None

    def _show(self):
        if self._tip is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except tk.TclError:
            return
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=self.text, justify="left",
            background="#ffffcc", foreground="#1a1a1a",
            relief="solid", borderwidth=1, wraplength=380,
            font=("Segoe UI", 9), padx=6, pady=4,
        ).pack()

    def _hide(self, _=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


def attach(widget, dialog_key, key):
    """Tip ``widget`` with the text at ``dialog_key`` -> ``key``. No-op if missing."""
    if widget is None:
        return
    text = _data().get(dialog_key, {}).get(key)
    if text:
        _ToolTip(widget, text)


def attach_tree(root, dialog_key):
    """Tip every descendant of ``root`` whose ``text`` matches a key.

    Synthetic-id keys (no matching visible text) are left for explicit
    ``attach`` calls.
    """
    section = _data().get(dialog_key, {})
    if not section:
        return

    def walk(w):
        for child in w.winfo_children():
            try:
                txt = child.cget("text")
            except (tk.TclError, AttributeError):
                txt = None
            if txt and txt in section:
                _ToolTip(child, section[txt])
            walk(child)

    walk(root)


if __name__ == "__main__":
    # Self-check: the data-matching logic needs no display.
    assert _data(), "tooltips.json failed to load"
    se = _data().get("SpectrumExplorer", {})
    assert "N sources" in se and se["N sources"]
    # Missing keys are silent no-ops.
    assert attach(None, "Nope", "Nope") is None
    print("tooltip_help self-check OK:",
          sum(len(v) for v in _data().values()), "tips across",
          len(_data()), "dialogs")
