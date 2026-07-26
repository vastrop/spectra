"""
LAMOST DR11 low-resolution spectrum viewer dialog.

A self-contained modeless Toplevel that, given a resolved source name
(from a WCS plate-solve in spectrum_explorer), queries the LAMOST DR11
SSA service for a low-res spectrum at that position and, if one exists,
downloads and plots it.

The query/sanitization logic mirrors the standalone ``lamost_query.py``
probe; the plotting mirrors ``lamost_viewer_simple.py`` (HDU1 row 0,
WAVELENGTH/FLUX columns).  Network and VOTable parsing run on a worker
thread so the Tk event loop stays responsive; results are marshalled
back to the main thread via a ``queue.Queue`` drained by an ``after``
poller created on the Tk thread — the same pump the explorer uses (Tk
calls from worker threads, including ``after`` itself, are not
reliably safe during main-window drawing/teardown).
"""
import io
import os
import re
import gzip
import queue
import tempfile
import threading

import tkinter as tk
from tkinter import ttk

import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse
import tooltip_help as tt

# Dark-theme palette — matches spectrum_explorer.  ACC here is the error
# red (only used for failure text); the app's chrome accent is #e0c46c.
BG, PANEL, ACC, CURVE = "#0e1014", "#0f0f1a", "#e94560", "#4ec9b0"

SSA = "https://www.lamost.org/dr11/v2.0/voservice/ssap"
# Search diameter in degrees; ~20 arcsec cone (fibre is 3.3" but
# catalog/pointing offsets warrant a generous cone).
SSA_SIZE = 0.0056


def _sanitize_votable(raw_bytes):
    """Repair LAMOST's non-standard VOTable so astropy can parse it.

    LAMOST DR11 SSA emits illegal datatype="date" fields, empty
    arraysize="" attributes, and FIELDs missing a datatype entirely.
    These three regex/replace fixes are lifted verbatim from the
    working lamost_query.py probe.
    """
    raw = raw_bytes.decode("utf-8", errors="replace")
    raw = raw.replace('datatype="date"', 'datatype="char" arraysize="*"')
    raw = raw.replace('arraysize=""', 'arraysize="*"')

    def _patch_field(m):
        tag = m.group(0)
        if "datatype=" not in tag:
            tag = tag.replace("<FIELD", '<FIELD datatype="char" arraysize="*"', 1)
        return tag

    raw = re.sub(r"<FIELD\b[^>]*>", _patch_field, raw)
    return raw.encode("utf-8")


def _candidate_urls(vot):
    """Yield candidate download URLs from a LAMOST SSA result table.

    SSA's 'accref' is supposed to be a direct FITS link but LAMOST
    sometimes returns a datalink / landing URL that serves HTML.  We
    therefore collect *all* URL-bearing string columns, accref first,
    and try them in order until one yields valid FITS.
    """
    preferred = ("accref", "access_url", "accessurl", "accessReference")
    cols = list(vot.colnames)
    # accref-like columns first, then any other column whose first value
    # looks like an http URL.
    ordered = [c for c in cols if c.lower() in [p.lower() for p in preferred]]
    ordered += [c for c in cols if c not in ordered
                and isinstance(vot[c][0], (str, bytes))
                and "http" in str(vot[c][0]).lower()]

    seen = set()
    for row in range(len(vot)):
        for c in ordered:
            val = str(vot[c][row]).strip()
            if val.lower().startswith("http") and val not in seen:
                seen.add(val)
                yield c, row, val


def _boxcar(flux, window):
    """Odd-window boxcar smooth that preserves length and edges.

    window <= 1 returns the input unchanged.  Reflection padding avoids
    the cosine roll-off at the spectrum ends.
    """
    window = int(window)
    if window <= 1:
        return flux
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(flux, pad, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _read_lamost_fits(raw_bytes):
    """Parse LAMOST LRS FITS bytes -> (wavelength, flux).

    LAMOST serves the spectrum as gzip-compressed FITS, so we
    transparently decompress when the gzip magic (1f 8b) is present.
    Raises ValueError if the result is not FITS (e.g. an HTML error
    page returned by a datalink endpoint).
    """
    if raw_bytes[:2] == b"\x1f\x8b":
        raw_bytes = gzip.decompress(raw_bytes)
    if raw_bytes[:6] != b"SIMPLE":
        head = raw_bytes[:200].decode("latin-1", "replace")
        raise ValueError(f"not a FITS file (starts with: {head[:80]!r})")
    with fits.open(io.BytesIO(raw_bytes)) as hdul:
        spec = hdul[1].data[0]
        wavelength = np.asarray(spec["WAVELENGTH"], dtype=float)
        flux = np.asarray(spec["FLUX"], dtype=float)
    return wavelength, flux


def _query_lamost(ra, dec, save_path=None):
    """Return (wavelength, flux, access_url) for the first LAMOST LRS hit.

    Tries each candidate URL until one yields valid FITS.  When
    ``save_path`` is given, the bytes of the *last attempted* download
    are written there for inspection (handy when nothing parses).
    Raises RuntimeError with a user-facing message otherwise.  Runs on
    a worker thread.
    """
    params = {"REQUEST": "queryData", "POS": f"{ra},{dec}", "SIZE": SSA_SIZE}
    r = requests.get(SSA, params=params, timeout=90)
    r.raise_for_status()

    content = _sanitize_votable(r.content)
    vot = parse(io.BytesIO(content), verify="ignore").get_first_table().to_table()

    if len(vot) == 0:
        raise RuntimeError(
            "No LAMOST DR11 low-res spectrum at this position.")

    candidates = list(_candidate_urls(vot))
    if not candidates:
        raise RuntimeError("LAMOST result has no downloadable spectrum URL.")

    last_err = None
    last_bytes = None
    for col, row, url in candidates:
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            last_bytes = resp.content
            wl, flux = _read_lamost_fits(last_bytes)
            return wl, flux, url
        except Exception as e:                # noqa: BLE001 — try next URL
            last_err = f"{col}[{row}]: {e}"
            continue

    # Nothing parsed.  Save the last payload so the user can inspect it.
    if save_path and last_bytes is not None:
        try:
            with open(save_path, "wb") as f:
                f.write(last_bytes)
        except OSError:
            pass
    raise RuntimeError(
        f"Downloaded data is not valid FITS. Last attempt — {last_err}. "
        f"Tried {len(candidates)} URL(s)."
        + (f" Raw payload saved to {save_path}." if save_path else ""))


class LamostDialog(tk.Toplevel):
    """Modeless LAMOST low-res spectrum viewer.

    Construction kicks off the query immediately for ``source_name`` (a
    SIMBAD-resolvable identifier).  The window shows a status line while
    the worker runs, then the plotted spectrum or an error message.
    """

    def __init__(self, parent, source_name, ra=None, dec=None, xlim=None):
        super().__init__(parent)
        self._parent = parent
        self._source_name = source_name
        # Display form: SIMBAD main-ids carry a "V* " prefix on variable
        # stars; the explorer's overlay labels drop it (_compact_label),
        # so drop it here too.  Queries keep the full id — guaranteed to
        # resolve.
        self._display_name = re.sub(r"^V\*\s+", "", source_name)
        self._closed = False
        # Worker→Tk pump: the worker only ever puts callables here; the
        # Tk-side poller below runs them.  Workers never touch Tk.
        self._q = queue.Queue()
        self._poll_after = self.after(50, self._poll_queue)
        # Parent's current calibrated-panel wavelength limits (lo, hi) or
        # None — applied to the plot for direct comparison.
        self._xlim = xlim
        # Raw spectrum cache (set on successful load) so the smoothing
        # control can redraw without re-querying.
        self._wl = None
        self._flux = None
        # Where a non-FITS payload gets written for inspection when the
        # download doesn't parse (see _query_lamost).
        self._debug_save_path = os.path.join(
            tempfile.gettempdir(),
            f"lamost_{re.sub(r'[^A-Za-z0-9]+', '_', source_name)}.bin")

        self.title(f"LAMOST DR11 — {self._display_name}")
        self.configure(bg=BG)
        self.geometry("900x520")

        self._build_ui()

        # Resolve coordinates: prefer caller-supplied RA/Dec (from the
        # WCS solve), else fall back to a SIMBAD name resolution.
        self._start_query(ra, dec)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        plt.style.use("dark_background")

        top = ttk.Frame(self, style="Panel.TFrame", padding=6)
        top.pack(side="top", fill="x")
        self._status = ttk.Label(
            top, text=f"Querying LAMOST DR11 for {self._display_name}…",
            style="Header.TLabel")
        self._status.pack(side="left")

        # Smoothing control — boxcar window in pixels (1 = no smoothing).
        # Odd windows only; redraws from the cached raw spectrum.
        ttk.Label(top, text="Smooth (px):", style="Header.TLabel").pack(
            side="left", padx=(16, 4))
        self.v_smooth = tk.IntVar(value=1)
        self._smooth_scale = ttk.Scale(
            top, from_=1, to=51, orient="horizontal", length=140,
            command=self._on_smooth)
        self._smooth_scale.pack(side="left")
        tt.attach(self._smooth_scale, "LamostDialog", "Smooth (px)")
        self._smooth_lbl = ttk.Label(top, text="1", style="Header.TLabel")
        self._smooth_lbl.pack(side="left", padx=(4, 0))
        # set() fires _on_smooth synchronously (0 → 1 is a change), which
        # updates _smooth_lbl — so the label must exist first.
        self._smooth_scale.set(1)

        plot_frame = ttk.Frame(self, style="TFrame")
        plot_frame.pack(side="top", fill="both", expand=True)

        # Figure(), not plt.figure(): pyplot would hand this embedded figure a
        # manager — i.e. a second tk.Tk() root, a whole extra Tcl interpreter
        # living inside the app, withdrawn so you never see it.  Closing the
        # dialog would then tear that root down from inside the main
        # interpreter's event loop.  FigureCanvasTkAgg below is the only
        # canvas this dialog needs.
        self.fig = Figure(facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#262c37")
        self.ax.tick_params(colors="#aab2c0", labelsize=8)
        self.ax.set_xlabel(r"Wavelength (Å)", color="#aab2c0", fontsize=9)
        self.ax.set_ylabel("Flux", color="#aab2c0", fontsize=9)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.12)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        tb = ttk.Frame(plot_frame, style="TFrame")
        tb.pack(side="top", fill="x")
        NavigationToolbar2Tk(self.canvas, tb)

    # ── Query lifecycle ───────────────────────────────────────────────
    def _start_query(self, ra, dec):
        def worker():
            try:
                if ra is None or dec is None:
                    # Resolve by name via SIMBAD (import locally to keep
                    # the dialog importable without astroquery present
                    # until actually used).
                    from astroquery.simbad import Simbad
                    res = Simbad.query_object(self._source_name)
                    if res is None or len(res) == 0:
                        raise RuntimeError(
                            f"SIMBAD could not resolve “{self._source_name}”.")
                    r_ra = float(res["ra"][0])
                    r_dec = float(res["dec"][0])
                else:
                    coord = SkyCoord(ra, dec, unit="deg")
                    r_ra, r_dec = coord.ra.deg, coord.dec.deg

                wl, flux, url = _query_lamost(
                    r_ra, r_dec, save_path=self._debug_save_path)
                self._marshal(lambda: self._on_success(wl, flux))
            except Exception as e:           # noqa: BLE001 — surface to UI
                msg = str(e) or e.__class__.__name__
                self._marshal(lambda: self._on_error(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _marshal(self, fn):
        """Queue fn for the Tk main thread (worker-safe: no Tk calls)."""
        self._q.put(fn)

    def _poll_queue(self):
        """Tk-thread pump for worker results (see module docstring)."""
        if self._closed:
            return
        try:
            while True:
                self._q.get_nowait()()
        except queue.Empty:
            pass
        self._poll_after = self.after(50, self._poll_queue)

    def _on_success(self, wl, flux):
        if self._closed:
            return
        self._wl = wl
        self._flux = flux
        self._redraw()
        # Status line reflects the raw spectrum extent, plus the comparison
        # window when one was supplied.
        msg = (f"{self._display_name}: {len(wl)} points "
               f"({wl.min():.0f}–{wl.max():.0f} Å)")
        if self._xlim is not None:
            msg += f"  ·  showing {self._xlim[0]:.0f}–{self._xlim[1]:.0f} Å"
        self._status.config(text=msg)

    def _redraw(self):
        """Redraw the spectrum from the cached arrays at the current
        smoothing window and (optionally) the parent's λ limits."""
        if self._wl is None or self._flux is None:
            return
        win = int(self.v_smooth.get())
        flux = _boxcar(self._flux, win)

        self.ax.clear()
        self.ax.set_facecolor(PANEL)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#262c37")
        self.ax.tick_params(colors="#aab2c0", labelsize=8)
        self.ax.plot(self._wl, flux, color=CURVE, lw=0.7)
        self.ax.set_xlabel(r"Wavelength (Å)", color="#aab2c0", fontsize=9)
        self.ax.set_ylabel("Flux", color="#aab2c0", fontsize=9)
        title = f"LAMOST DR11 LRS — {self._display_name}"
        if win > 1:
            title += f"  (smoothed ×{win})"
        self.ax.set_title(title, color="#aab2c0", fontsize=9)
        self.ax.grid(True, linestyle="--", alpha=0.25, color="#262c37")

        # Match the parent's wavelength window for direct comparison, and
        # autoscale Y to the data actually visible in that window.
        if self._xlim is not None:
            lo, hi = self._xlim
            self.ax.set_xlim(lo, hi)
            # Only finite visible flux may set limits — a LAMOST segment
            # can be all-NaN in the window, and set_ylim(nan) raises.
            yv = flux[(self._wl >= lo) & (self._wl <= hi)]
            yv = yv[np.isfinite(yv)]
            if yv.size:
                pad = 0.05 * ((yv.max() - yv.min()) or 1.0)
                self.ax.set_ylim(yv.min() - pad, yv.max() + pad)
        self.canvas.draw()

    def _on_smooth(self, _value):
        """Scale callback: snap to an odd window, update label, redraw."""
        win = int(round(float(self._smooth_scale.get())))
        if win % 2 == 0:                # boxcar wants an odd window
            win += 1
        self.v_smooth.set(win)
        self._smooth_lbl.config(text=str(win))
        self._redraw()

    def _on_error(self, msg):
        if self._closed:
            return
        self._status.config(text=f"LAMOST: {msg}")
        self.ax.clear()
        self.ax.set_facecolor(PANEL)
        self.ax.text(0.5, 0.5, msg, color=ACC, ha="center", va="center",
                     transform=self.ax.transAxes, fontsize=10, wrap=True)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def _on_close(self):
        self._closed = True
        try:
            self.after_cancel(self._poll_after)
        except tk.TclError:
            pass
        # No plt.close(): the Figure has no pyplot manager, so destroying the
        # window below is the whole teardown.  The worker thread may still be
        # in flight — anything it queues after this simply never runs.
        # Clear the explorer's single-instance reference so the destroyed
        # dialog (and its cached arrays) can be collected — matches the
        # other modeless dialogs.
        if getattr(self._parent, "_lamost_dialog", None) is self:
            self._parent._lamost_dialog = None
        self.destroy()
