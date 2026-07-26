"""
livestack_feeder.py — test feeder for the Spectrum Explorer live stack.

Pick a source folder of raw captures and the monitored (destination) folder;
this copies the captures in one at a time on an interval (default 10 s), so you
can exercise Livestack without a real camera.  The receiver's size-stability
guard tolerates the in-progress copy, so a plain shutil.copy2 is fine.

Run:  python livestack_feeder.py
"""
import os
import re
import shutil
import tkinter as tk
from tkinter import ttk, filedialog

FITS_EXTS = (".fit", ".fits", ".fts")


def list_captures(folder):
    """FITS files in `folder`, natural-sorted so f2 < f10 (capture order)."""
    def nat(s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", s)]
    try:
        names = [f for f in os.listdir(folder)
                 if f.lower().endswith(FITS_EXTS)]
    except OSError:
        return []
    names.sort(key=nat)
    return [os.path.join(folder, n) for n in names]


class Feeder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Livestack feeder")
        self.resizable(False, False)

        self.v_src = tk.StringVar()
        self.v_dst = tk.StringVar()
        self.v_interval = tk.StringVar(value="10")

        self._files = []
        self._idx = 0
        self._after = None

        pad = dict(padx=6, pady=3)
        row = 0
        for label, var, cmd in (
            ("Source (captures)", self.v_src, self._pick_src),
            ("Monitored (dest)", self.v_dst, self._pick_dst),
        ):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", **pad)
            ttk.Entry(self, textvariable=var, width=44).grid(
                row=row, column=1, **pad)
            ttk.Button(self, text="…", width=3, command=cmd).grid(
                row=row, column=2, **pad)
            row += 1

        ttk.Label(self, text="Interval (s)").grid(
            row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.v_interval, width=8).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        self._btn = ttk.Button(self, text="Start", command=self._toggle)
        self._btn.grid(row=row, column=0, columnspan=3, sticky="ew", **pad)
        row += 1

        self._status = ttk.Label(self, text="Idle.", foreground="#555")
        self._status.grid(row=row, column=0, columnspan=3, sticky="w", **pad)

    def _pick_src(self):
        d = filedialog.askdirectory(title="Source folder of captures")
        if d:
            self.v_src.set(d)

    def _pick_dst(self):
        d = filedialog.askdirectory(title="Monitored destination folder")
        if d:
            self.v_dst.set(d)

    def _toggle(self):
        if self._after is not None:
            self._stop("Stopped.")
            return
        src, dst = self.v_src.get().strip(), self.v_dst.get().strip()
        if not src or not dst:
            self._status.config(text="Pick both folders first.")
            return
        if os.path.abspath(src) == os.path.abspath(dst):
            self._status.config(text="Source and destination must differ.")
            return
        try:
            self._interval_ms = max(1, int(float(self.v_interval.get()))) * 1000
        except ValueError:
            self._status.config(text="Interval must be a number.")
            return
        self._files = list_captures(src)
        if not self._files:
            self._status.config(text="No FITS files in source folder.")
            return
        self._idx = 0
        self._btn.config(text="Stop")
        self._feed_next()   # first file immediately, then on interval

    def _stop(self, msg):
        if self._after is not None:
            self.after_cancel(self._after)
            self._after = None
        self._btn.config(text="Start")
        self._status.config(text=msg)

    def _feed_next(self):
        if self._idx >= len(self._files):
            self._stop(f"Done — delivered {len(self._files)} file(s).")
            return
        src_path = self._files[self._idx]
        name = os.path.basename(src_path)
        try:
            shutil.copy2(src_path, os.path.join(self.v_dst.get(), name))
            self._status.config(
                text=f"Delivered {self._idx + 1}/{len(self._files)}: {name}")
        except OSError as e:
            self._status.config(text=f"Copy failed for {name}: {e}")
        self._idx += 1
        self._after = self.after(self._interval_ms, self._feed_next)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        # Natural sort is the only real logic here, so that is what is checked.
        import tempfile
        d = tempfile.mkdtemp()
        for n in ("f10.fits", "f2.fits", "f1.fits", "notes.txt"):
            open(os.path.join(d, n), "w").close()
        got = [os.path.basename(p) for p in list_captures(d)]
        assert got == ["f1.fits", "f2.fits", "f10.fits"], got
        print("selftest OK:", got)
    else:
        Feeder().mainloop()
