"""
nina_dialog.py
==============

Remote-observing panel for a NINA rig running the Advanced API plugin.

Workflow it supports (NINA's own UI, over VNC, keeps doing the pointing):
slew to a bright A star in NINA → [Run spectral autofocus here] → slew to
the science target, start guiding in NINA → [Start capturing here] into a
local work folder → livestack that folder with the explorer's existing,
untouched livestack machinery → cleanup when moving on.

Sections
--------
* Connection — host/port + Connect (threaded; a dead host must not
  freeze Tk).  A successful connect also reads the camera's gain into
  the capture Gain field (unless the user already typed one).
* Autofocus — calls ``spectral_autofocus.run_autofocus()`` on a worker
  thread, exactly like every other long job in this dialog: the tool's
  own argparse builds the parameters (so the panel cannot drift from the
  CLI), progress arrives through an ``emit`` callback and is pumped into
  the log, and Cancel sets an Event the sweep checks between points —
  the exposure in flight still lands, so no half-written FITS.  It is an
  in-process call rather than a subprocess: in the frozen build
  ``sys.executable`` is the exe, not python, so a subprocess could never
  reach the CLI entry point.  The live (possibly
  unsaved) explorer config is passed straight in as a FocusConfig, so the
  sweep always uses the current angle/dispersion nodes.  A sweep-model
  pane plots the measured points, parabola fit and best/ideal positions,
  plus the best frame's extracted strip once a run completes.
* Capture — two modes writing FITS into the local work folder:
    - snapshot loop: this dialog drives the camera (exposure/gain/count);
    - mirror: NINA's own sequence drives the camera (guiding, dithering);
      the image history is polled and each new LIGHT frame downloaded,
      keeping NINA's filename, into a per-target subfolder named after
      the history entry's TargetName (mirroring NINA's layout).  The
      work-folder field stays the BASE — the active target subfolder is
      only tracked internally, so switching targets never nests folders.
      Local copies double as a backup.
  A status line shows the running livestack total (frames, exposure).
* Livestack this folder / Cleanup — hook the active folder (the target
  subfolder while mirroring, the work folder otherwise) into the
  parent's livestack, and clear it (after confirmation) between targets.

Parent-API contract: reads ``parent._config_dict()`` (autofocus config
snapshot), calls ``parent._start_livestack_at(folder)`` /
``parent._stop_livestack()``, reads ``parent._livestack_dir`` and
``parent._ui_state_path`` (host/port/params persist in the per-machine
UI dotfile, NOT in the analysis config).  Single instance
per parent, held in ``parent._nina_dialog``.
"""

from __future__ import annotations

import gc
import json
import csv
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, angular_separation, SkyCoord, FK5, ICRS
from astropy.time import Time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from nina_client import NinaClient, NinaError, DEFAULT_PORT, is_guiding
import tooltip_help as tt

BG = "#0e1014"
PANEL = "#0f0f1a"
FG = "#aab2c0"
FG_DIM = "#6b7484"
LINE = "#323a47"
ACCENT = "#e0c46c"

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Where the app WRITES.  In a frozen build _ROOT is inside the bundle's
# _internal/, which a reinstall wipes — bundled read-only data (the A-star
# catalog) belongs there, but a focus run's frames must not.  Same rule as
# the spectra DB (db/spectra_db.py).
_DATA_ROOT = (os.path.dirname(os.path.abspath(sys.executable))
              if getattr(sys, "frozen", False) else _ROOT)

MIRROR_POLL_S = 2.0
# Suffixes tried before giving up on a colliding frame name (_write_frame).
MAX_NAME_COLLISIONS = 100
# Re-read the connection status (focuser position especially) this often once
# connected, so an autofocus move NINA already applied shows up without a
# manual reconnect.  A couple of cheap GETs; harmless to run during a sweep.
STATUS_POLL_S = 5.0
# Let the mount settle before handing it back to the guider: the last
# arcseconds of a slew (backlash, dec creep) are still moving when the driver
# clears Slewing, and PHD2 locking on during them calibrates off a star that
# is still drifting.
GUIDE_SETTLE_S = 5.0
ASTAR_CATALOG = os.path.join(_ROOT, "ReferenceLibrary",
                             "astar_catalog.csv")


def _safe_dirname(name):
    """Target name -> usable Windows folder name, or None."""
    return "".join(c for c in name if c not in '<>:"/\\|?*').strip() or None


def _parse_astar_catalog(handle):
    """Parse catalog CSV rows into the numeric form used by the dialog."""
    rows = []
    for row in csv.DictReader(handle):
        rows.append({
            "hip": row["hip"].strip(), "hd": row["hd"].strip(),
            "ra_deg": float(row["ra_deg"]),
            "dec_deg": float(row["dec_deg"]),
            "vmag": float(row["vmag"]), "sptype": row["sptype"].strip(),
        })
    return rows


AF_BASE_EXPOSURE_S = 3.0    # measured baseline: a V=2 A star at 3 s
AF_BASE_VMAG = 2.0
AF_MAX_VMAG = 4.5           # scaled exposure tops out at 30 s there


def _autofocus_exposure(vmag):
    """Exposure scaled from the V=2/3 s baseline by the flux ratio."""
    return AF_BASE_EXPOSURE_S * 10 ** (0.4 * (vmag - AF_BASE_VMAG))


def _nearest_astars(rows, ra_deg, dec_deg, limit=15, max_vmag=AF_MAX_VMAG):
    """Return catalog rows nearest a decimal-degree ICRS pointing.

    Stars fainter than ``max_vmag`` are pruned: the magnitude-scaled
    autofocus exposure would exceed 30 s per sweep point beyond V≈4.5.
    """
    rows = [row for row in rows if row["vmag"] <= max_vmag]
    if not rows:
        return []
    ras = np.deg2rad([row["ra_deg"] for row in rows])
    decs = np.deg2rad([row["dec_deg"] for row in rows])
    sep = np.rad2deg(angular_separation(
        np.deg2rad(ra_deg), np.deg2rad(dec_deg), ras, decs))
    order = np.argsort(sep)[:limit]
    return [dict(rows[i], separation_deg=float(sep[i])) for i in order]


def _to_j2000(ra_deg, dec_deg, epoch):
    """Precess a mount pointing to J2000/ICRS so it is comparable with the
    (J2000) A-star catalog.

    NINA tags each Coordinates block with an Epoch.  A mount configured for
    JNOW reports equinox-of-date, which precession puts ~0.3° from J2000 by
    2026 — enough that a star you are dead-on reads a 0.2–0.3° "separation"
    against the J2000 catalog (the mount is on target; only the readout's
    frame differs).  Only JNOW is converted; J2000 — or any unrecognised
    tag — passes through untouched.  FK5-mean-of-date is the conventional
    JNOW model; the arcsec-level nutation/aberration it omits is far below
    the precession term this removes.
    """
    if str(epoch).upper() != "JNOW":
        return ra_deg, dec_deg
    c = SkyCoord(ra_deg, dec_deg, unit="deg",
                 frame=FK5(equinox=Time.now())).transform_to(ICRS())
    return float(c.ra.deg), float(c.dec.deg)


def _format_mount_pos(ra_deg, dec_deg, epoch):
    """Sexagesimal pointing for the readout, e.g. 'RA 15h59m30.0s  Dec +25°55′12.6″ (JNOW)'."""
    ra = Angle(ra_deg, unit=u.deg).to_string(
        unit=u.hourangle, sep="hms", precision=1)
    dec = Angle(dec_deg, unit=u.deg).to_string(
        unit=u.deg, sep=("°", "′", "″"), precision=1, alwayssign=True)
    return f"Mount: RA {ra}  Dec {dec} ({epoch})"


class NinaDialog(tk.Toplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("NINA remote")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.transient(parent)          # modeless, no grab

        self._q = queue.Queue()         # worker → UI messages
        self._af_thread = None          # running autofocus worker
        self._af_stop = threading.Event()
        self._cap_stop = threading.Event()
        self._cap_thread = None         # snapshot-loop OR mirror thread
        self._foc_thread = None         # manual focuser nudge in flight
        self._cap_mode = None           # "loop" | "mirror" | None
        self._mirror_dir = None         # per-target subfolder frames land in
        self._astar_catalog = None
        self._mount_thread = None
        self._af_return = None
        self._closed = False
        self._pump_after = None         # after-id, cancelled on close
        self._connected = False         # a probe succeeded → poll status
        self._poll_thread = None        # single-flight status refresh
        self._last_poll = 0.0
        self._led_phase = False         # alternates greens: poll heartbeat
        self._resume_guiding = False    # was guiding at focus-run start

        self._load_state()
        # Editing the base folder invalidates the remembered subfolder.
        self.v_workdir.trace_add(
            "write", lambda *a: setattr(self, "_mirror_dir", None))
        self._build_ui()
        self.parent._dock_nina(self)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda e: self._close())
        # Closing the MAIN window destroys us without _close — stop the
        # pump so its pending after() doesn't fire on a dead widget.
        self.bind("<Destroy>", self._on_destroy)
        self._pump()

    def _on_destroy(self, event):
        if event.widget is self and not self._closed:
            self._closed = True
            self._cap_stop.set()
            self._af_stop.set()
            if self._pump_after is not None:
                try:
                    self.after_cancel(self._pump_after)
                except tk.TclError:
                    pass
            # Main-window close skips _close — don't lose the session's
            # settings (work folder, exposure…) on that path either.
            try:
                self._save_state()
            except tk.TclError:
                pass

    # ── persistence (per-machine UI dotfile, shared with last-dirs) ───
    def _load_state(self):
        self._state_path = getattr(
            self.parent, "_ui_state_path",
            os.path.join(os.path.expanduser("~"), ".spectrum_explorer_ui.json"))
        try:
            with open(self._state_path, "r") as f:
                self._ui_state = json.load(f)
        except Exception:
            self._ui_state = {}
        # Valid JSON of the wrong shape (a list, a bare string, null) parses
        # fine and then AttributeErrors on .get() — which happens out here,
        # not in the try, and would leave the dialog permanently unopenable
        # while the app itself runs on.  The parent's loader normalises
        # inside its own try (spectrum_explorer.py _ui_state_path); match it.
        if not isinstance(self._ui_state, dict):
            self._ui_state = {}
        nina = self._ui_state.get("nina", {})
        if not isinstance(nina, dict):
            nina = {}
        self.v_host = tk.StringVar(value=nina.get("host", "localhost"))
        self.v_port = tk.StringVar(value=str(nina.get("port", DEFAULT_PORT)))
        self.v_af_exp = tk.StringVar(value=str(nina.get("af_exposure", 3.0)))
        self.v_af_step = tk.StringVar(value=str(nina.get("af_step", 25)))
        self.v_af_points = tk.StringVar(value=str(nina.get("af_points", 9)))
        self.v_foc_step = tk.StringVar(value=str(nina.get("foc_step", 25)))
        self.v_cap_exp = tk.StringVar(value=str(nina.get("cap_exposure", 30.0)))
        self.v_cap_gain = tk.StringVar(value=str(nina.get("cap_gain", "")))
        self.v_cap_count = tk.StringVar(value=str(nina.get("cap_count", 0)))
        self.v_workdir = tk.StringVar(value=nina.get("work_dir", ""))

    def _save_state(self):
        def _num(var, cast, default):
            try:
                return cast(var.get().strip())
            except (ValueError, TypeError):
                return default
        # Re-read the dotfile before writing: the explorer updates sibling
        # keys (last_dirs, last_config) while this dialog is open, and a
        # stale in-memory copy would clobber them.
        try:
            with open(self._state_path, "r") as f:
                self._ui_state = dict(json.load(f))
        except Exception:
            self._ui_state = {}
        self._ui_state["nina"] = {
            "host": self.v_host.get().strip(),
            "port": _num(self.v_port, int, DEFAULT_PORT),
            "af_exposure": _num(self.v_af_exp, float, 3.0),
            "af_step": _num(self.v_af_step, int, 25),
            "af_points": _num(self.v_af_points, int, 9),
            "foc_step": _num(self.v_foc_step, int, 25),
            "cap_exposure": _num(self.v_cap_exp, float, 30.0),
            "cap_gain": self.v_cap_gain.get().strip(),
            "cap_count": _num(self.v_cap_count, int, 0),
            "work_dir": self.v_workdir.get().strip(),
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(self._ui_state, f, indent=2)
        except OSError:
            pass

    # ── client from the current host/port fields ──────────────────────
    def _client(self):
        host = self.v_host.get().strip()
        try:
            port = int(self.v_port.get().strip())
        except ValueError:
            port = DEFAULT_PORT
        return NinaClient(host, port)

    def _require_nina(self, client):
        """Pre-flight before any action: fail fast and VISIBLY when
        NINA is unreachable — without this, a dead rig only wrote one
        line into the log and the buttons appeared to do nothing.
        Synchronous with a short timeout (same trade-off as Connect,
        but bounded at 3 s instead of the client's default 15)."""
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            client.version(timeout=3)
            return True
        except Exception as e:
            messagebox.showerror(
                "NINA", f"Cannot reach NINA at {client.base}\n\n{e}\n\n"
                        f"Check host/port (and that the Advanced API "
                        f"plugin is running), then try Connect.",
                parent=self)
            return False
        finally:
            self.config(cursor="")

    # ── UI ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = {"padx": 8, "pady": 3}

        # Connection -----------------------------------------------------
        con = ttk.LabelFrame(self, text="Connection")
        con.pack(fill="x", **pad)
        ttk.Label(con, text="Host").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(con, textvariable=self.v_host, width=16).grid(
            row=0, column=1, sticky="w")
        ttk.Label(con, text="Port").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Entry(con, textvariable=self.v_port, width=6).grid(
            row=0, column=3, sticky="w")
        self.btn_probe = ttk.Button(con, text="Connect",
                                    style="Action.TButton",
                                    command=self._probe)
        self.btn_probe.grid(row=0, column=4, padx=8)
        self.led = ttk.Label(con, text="●", foreground="#c0392b")
        self.led.grid(row=0, column=5, padx=(0, 4))
        self.v_status = tk.StringVar(value="Not connected yet.")
        ttk.Label(con, textvariable=self.v_status).grid(
            row=1, column=0, columnspan=6, sticky="w", padx=4, pady=(2, 0))
        self.v_mount = tk.StringVar(
            value="Mount: position unknown — Connect reads it.")
        ttk.Label(con, textvariable=self.v_mount).grid(
            row=2, column=0, columnspan=6, sticky="w", padx=4, pady=(0, 4))
        # Slew outcomes get their own line: the status line above is
        # rewritten by the 5 s poll, so "…complete" flashed and vanished.
        self.v_slew = tk.StringVar(value="")
        ttk.Label(con, textvariable=self.v_slew,
                  foreground=ACCENT).grid(
            row=3, column=0, columnspan=6, sticky="w", padx=4, pady=(0, 4))

        # Autofocus -------------------------------------------------------
        af = ttk.LabelFrame(self, text="Spectral autofocus (point NINA at a "
                                       "bright A star first)")
        af.pack(fill="x", **pad)
        for col, (lbl, var, w) in enumerate((
                ("Exp (s)", self.v_af_exp, 6),
                ("Step", self.v_af_step, 6),
                ("Points", self.v_af_points, 6))):
            lab = ttk.Label(af, text=lbl)
            lab.grid(row=0, column=2 * col, sticky="w", padx=4)
            if lbl == "Exp (s)":
                self._lbl_af_exp = lab
            ttk.Entry(af, textvariable=var, width=w).grid(
                row=0, column=2 * col + 1, sticky="w")
        # Manual focus nudge.  Signed steps, not "in"/"out": which way the
        # drawtube travels for a rising position number is the focuser's
        # business (and reversible in its driver), so naming a direction
        # here would be wrong for some rigs.
        self._lbl_foc = ttk.Label(af, text="Manual focus ±")
        self._lbl_foc.grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(af, textvariable=self.v_foc_step, width=6).grid(
            row=1, column=1, sticky="w")
        self.btn_foc_minus = ttk.Button(af, text="◀  −", width=6,
                                        command=lambda: self._nudge_focus(-1))
        self.btn_foc_minus.grid(row=1, column=2, sticky="w", padx=(8, 2))
        self.btn_foc_plus = ttk.Button(af, text="+  ▶", width=6,
                                       command=lambda: self._nudge_focus(1))
        self.btn_foc_plus.grid(row=1, column=3, sticky="w")
        self.btn_af = ttk.Button(af, text="▶  Autofocus Here",
                                 style="Action.TButton",
                                 command=self._toggle_autofocus)
        self.btn_af.grid(row=1, column=4, columnspan=2, sticky="e",
                         padx=8, pady=4)
        self.v_af_result = tk.StringVar(value="")
        ttk.Label(af, textvariable=self.v_af_result).grid(
            row=2, column=0, columnspan=6, sticky="w", padx=4, pady=(0, 4))

        # Sweep-model plot: measured points, fitted parabola, best/ideal,
        # plus the best frame's extracted strip once a run completes.
        # Fed by the sweep's emit("afdata", …), cleared at each run start.
        self._af_fig = Figure(figsize=(5.6, 2.8), dpi=100, facecolor=BG,
                              layout="constrained")
        self._af_ax = None              # (re)created per _plot_af_data call
        self._af_canvas = FigureCanvasTkAgg(self._af_fig, master=af)
        widget = self._af_canvas.get_tk_widget()
        widget.configure(bg=BG, highlightthickness=0)
        widget.grid(row=3, column=0, columnspan=6, sticky="ew",
                    padx=4, pady=(0, 4))
        af.columnconfigure(5, weight=1)
        self._plot_af_data(None)

        # A-star slew -----------------------------------------------------
        slew = ttk.LabelFrame(self, text="A-star slew (autofocus targets)")
        slew.pack(fill="x", **pad)
        self.btn_astar_find = ttk.Button(
            slew, text="Find nearest", command=self._find_astars)
        self.btn_astar_find.grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.btn_astar_slew = ttk.Button(
            slew, text="Slew", style="Action.TButton", command=self._slew_astar)
        self.btn_astar_slew.grid(row=0, column=1, padx=4, pady=4, sticky="w")
        self.btn_astar_focus = ttk.Button(
            slew, text="Focus on Selected A star and Return",
            style="Action.TButton", command=self._focus_run)
        self.btn_astar_focus.grid(row=0, column=2, padx=4, pady=4, sticky="w")
        # "exp" stays last: _selected_astar indexes the earlier columns.
        columns = ("name", "sep", "vmag", "sptype", "ra", "dec", "exp")
        # 3 visible rows to save vertical room; the full nearest-star list
        # (up to _nearest_astars' limit) is reachable through the scrollbar.
        self.astar_tree = ttk.Treeview(
            slew, columns=columns, show="headings", height=3, selectmode="browse")
        headings = (("name", "Star", 105), ("sep", "Sep (°)", 65),
                    ("vmag", "Vmag", 55), ("sptype", "Type", 80),
                    ("ra", "RA (°)", 80), ("dec", "Dec (°)", 80),
                    ("exp", "Exp (s)", 60))
        for key, text, width in headings:
            self.astar_tree.heading(key, text=text)
            self.astar_tree.column(key, width=width, anchor="center")
        self.astar_tree.grid(row=1, column=0, columnspan=4, padx=(4, 0),
                             pady=(0, 4), sticky="ew")
        astar_scroll = ttk.Scrollbar(slew, orient="vertical",
                                     command=self.astar_tree.yview)
        self.astar_tree.configure(yscrollcommand=astar_scroll.set)
        astar_scroll.grid(row=1, column=4, sticky="ns", padx=(0, 4),
                          pady=(0, 4))
        self.astar_tree.bind("<<TreeviewSelect>>", self._on_astar_select)
        slew.columnconfigure(3, weight=1)

        # Capture ----------------------------------------------------------
        cap = ttk.LabelFrame(self, text="Capture to work folder")
        cap.pack(fill="x", **pad)
        ttk.Label(cap, text="Work folder").grid(row=0, column=0,
                                                sticky="w", padx=4)
        ttk.Entry(cap, textvariable=self.v_workdir, width=42).grid(
            row=0, column=1, columnspan=4, sticky="ew")
        ttk.Button(cap, text="…", width=2, command=self._browse_workdir).grid(
            row=0, column=5, padx=(2, 8))

        for col, (lbl, var, w) in enumerate((
                ("Exp (s)", self.v_cap_exp, 7),
                ("Gain", self.v_cap_gain, 6),
                ("Count (0=∞)", self.v_cap_count, 6))):
            lab = ttk.Label(cap, text=lbl)
            lab.grid(row=1, column=2 * col, sticky="w", padx=4)
            if lbl == "Exp (s)":
                self._lbl_cap_exp = lab
            ttk.Entry(cap, textvariable=var, width=w).grid(
                row=1, column=2 * col + 1, sticky="w")

        self.btn_loop = ttk.Button(cap, text="▶  Start capturing",
                                   style="Action.TButton",
                                   command=self._toggle_loop)
        self.btn_loop.grid(row=2, column=0, columnspan=2, sticky="ew",
                           padx=4, pady=4)
        self.btn_mirror = ttk.Button(cap, text="▶  Mirror NINA sequence",
                                     style="Action.TButton",
                                     command=self._toggle_mirror)
        self.btn_mirror.grid(row=2, column=2, columnspan=2, sticky="ew",
                             padx=4, pady=4)
        self.btn_stack_here = ttk.Button(cap, text="Livestack this folder",
                                         style="Action.TButton",
                                         command=self._toggle_livestack_here)
        self.btn_stack_here.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 4))
        ttk.Button(cap, text="Cleanup work folder…",
                   style="Action.TButton",
                   command=self._cleanup).grid(
            row=3, column=2, columnspan=2, sticky="ew", padx=4, pady=(0, 4))
        self.v_stack = tk.StringVar(value="Livestack: off")
        ttk.Label(cap, textvariable=self.v_stack).grid(
            row=4, column=0, columnspan=6, sticky="w", padx=4, pady=(0, 4))

        # Bottom bar ---------------------------------------------------------
        # Packed before the log so the packer reserves its strip first: the
        # log expands into whatever is left, never over the button.
        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        ttk.Separator(self, orient="horizontal").pack(
            side="bottom", fill="x", padx=8, pady=(4, 4))
        ttk.Button(bar, text="Hide ▸", width=14, style="Action.TButton",
                   command=self._hide).pack(side="right")

        # Log ---------------------------------------------------------------
        self.log = tk.Text(self, height=14, width=76, wrap="word",
                           background=PANEL, foreground=FG,
                           insertbackground=FG, relief="flat",
                           font=("Consolas", 9), state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=(3, 8))

        # Hover help. The autofocus and capture sections both caption an
        # entry "Exp (s)", so those two labels get explicit synthetic keys
        # instead of a tree match (attach_tree would tip both with the
        # same text).
        tt.attach_tree(self, "NinaDialog")
        tt.attach(self._lbl_af_exp, "NinaDialog", "af_exp")
        tt.attach(self._lbl_foc, "NinaDialog", "foc_step")
        tt.attach(self.btn_foc_minus, "NinaDialog", "foc_step")
        tt.attach(self.btn_foc_plus, "NinaDialog", "foc_step")
        tt.attach(self._lbl_cap_exp, "NinaDialog", "cap_exp")

    def _plot_af_data(self, data):
        """Render the sweep model (or the empty placeholder) in the AF pane,
        with the best frame's extracted strip below it when available."""
        # The saved strip of the best frame (written by the tool in the
        # run folder) — loaded first because it decides the layout.
        strip = None
        strip_path = data.get("strip_path") if data else None
        if strip_path and os.path.isfile(strip_path):
            try:
                strip = np.load(strip_path)
                if strip.ndim != 2 or not strip.size:
                    strip = None
            except Exception:
                strip = None

        self._af_fig.clf()
        if strip is not None:
            gs = self._af_fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0])
            ax = self._af_fig.add_subplot(gs[0])
            sax = self._af_fig.add_subplot(gs[1])
            sax.set_facecolor(PANEL)
            lo, hi = np.percentile(strip, [1.0, 99.8])
            sax.imshow(strip, cmap="gray", vmin=lo, vmax=max(hi, lo + 1),
                       aspect="auto", origin="lower", interpolation="nearest")
            sax.set_xticks([])
            sax.set_yticks([])
            for spine in sax.spines.values():
                spine.set_color(LINE)
            sax.set_ylabel(f"best strip\npos {data.get('best', '?')}",
                           color=FG, fontsize=6)
        else:
            ax = self._af_fig.add_subplot(111)
        self._af_ax = ax
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(LINE)
        ax.tick_params(colors=FG, labelsize=7)

        scores = ([float("nan") if s is None else float(s)
                   for s in data.get("scores", [])] if data else [])
        if not data or not any(np.isfinite(scores)):
            ax.text(0.5, 0.5, "No sweep data yet",
                    transform=ax.transAxes, ha="center", va="center",
                    color=FG_DIM, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            self._af_canvas.draw_idle()
            return

        pos = np.asarray(data["positions"], dtype=float)
        sco = np.asarray(scores)
        ax.plot(pos, sco, "o-", ms=4, lw=0.8, color=ACCENT,
                label="measured")
        if data.get("parabola"):
            a, b, c = data["parabola"]
            # Draw the parabola only over its fit window (the 5 measured
            # points nearest the vertex) — outside it the model is
            # meaningless and would squash the y-scale.
            vertex = data.get("ideal")
            if vertex is None:
                vertex = -b / (2 * a)
            near = pos[np.argsort(np.abs(pos - vertex))[:5]]
            px = np.linspace(near.min(), near.max(), 80)
            ax.plot(px, a * px ** 2 + b * px + c, lw=1.2,
                    color="#d06060", alpha=0.9, label="parabola fit")
        if data.get("best") is not None:
            best_pos = data["best"]
            # Label the best line with the score actually measured there (the
            # min FWHM the sweep reached), not just the position.
            idx = int(np.argmin(np.abs(pos - best_pos)))
            unit = {"fwhm": " Å", "spatial": " px"}.get(data.get("metric"), "")
            lbl = f"best {best_pos:g}"
            if np.isfinite(sco[idx]):
                lbl += f" ({sco[idx]:.2f}{unit})"
            ax.axvline(best_pos, color="#4c9f70", lw=1.0, ls="--", label=lbl)
        if data.get("ideal") is not None:
            ax.axvline(data["ideal"], color="#d06060", lw=1.2,
                       label=f"ideal {data['ideal']:.0f}")
        ylabels = {"fwhm": "line FWHM (Å)", "depth": "depth",
                   "gradient": "wing gradient", "spatial": "spatial FWHM (px)"}
        ax.set_ylabel(ylabels.get(data.get("metric"), "score"),
                      color=FG, fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.25, color=FG)
        leg = ax.legend(fontsize=6, loc="best", facecolor=PANEL,
                        edgecolor=LINE, labelcolor=FG)
        leg.get_frame().set_alpha(0.8)
        self._af_canvas.draw_idle()

    def _log(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text.rstrip("\n") + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # ── message pump: workers post, the Tk thread renders ─────────────
    def _pump(self):
        if self._closed:
            return
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "status":
                    self.v_status.set(msg[1])
                elif kind == "af_data":
                    self._plot_af_data(msg[1])
                elif kind == "af_done":
                    self._af_finished(msg[1])
                elif kind == "af_start":
                    if not self._toggle_autofocus():
                        self._start_return_slew()
                elif kind == "mount_pos":
                    self.v_mount.set(msg[1])
                elif kind == "astar_rows":
                    self._show_astars(msg[1])
                elif kind == "mount_done":
                    self._mount_thread = None
                    self.v_status.set(msg[1])
                    self.v_slew.set(msg[1])   # survives the status poll
                    self._log(msg[1])
                elif kind == "foc_done":
                    self._foc_thread = None
                    self.v_af_result.set(msg[1])
                    self._log(msg[1])
                    self._last_poll = 0.0    # refresh the position readout now
                elif kind == "cap_done":
                    self._capture_finished()
                elif kind == "cam_gain":
                    # NINA's actual camera gain, learned on Connect —
                    # fill the field but never clobber a manual value.
                    if not self.v_cap_gain.get().strip():
                        self.v_cap_gain.set(str(msg[1]))
                        self._log(f"Gain {msg[1]} read from the camera.")
                elif kind == "mirror_target":
                    # The work-folder field stays the BASE; writing the
                    # subfolder into it made every restart nest one
                    # level deeper (c:\base\HIP111169\Matar).
                    self._mirror_dir = msg[1]
                    self._log(f"[mirror] target folder: {msg[1]}")
                elif kind == "led":
                    # Green alternates light/dark on each poll so a live
                    # heartbeat is visibly different from a stale green.
                    if msg[1]:
                        self._led_phase = not self._led_phase
                        color = "#2ecc71" if self._led_phase else "#1e8449"
                    else:
                        color = "#c0392b"
                    self.led.config(foreground=color)
                elif kind == "probe_done":
                    self.btn_probe.config(
                        state="normal",
                        text="Polling" if self._connected else "Connect")
                elif kind == "poll_done":
                    self._poll_thread = None
        except queue.Empty:
            pass
        self._update_stack_status()
        self._maybe_poll_status()
        self._pump_after = self.after(150, self._pump)

    def _maybe_poll_status(self):
        """Refresh the connection status line every STATUS_POLL_S once
        connected, so a focuser move NINA applied (autofocus) shows up on its
        own.  Single-flight and off the Tk thread; a transient failure just
        keeps the last good line and retries next tick."""
        if (not self._connected or self._poll_thread is not None
                or self._closed):
            return
        if time.monotonic() - self._last_poll < STATUS_POLL_S:
            return
        self._last_poll = time.monotonic()
        client = self._client()

        def worker():
            try:
                text, _cam = self._status_line(client)
                self._q.put(("status", text))
                self._q.put(("led", True))
                try:
                    self._q.put(("mount_pos",
                                 self._mount_readout(client.mount_info())))
                except Exception:
                    pass    # no mount: keep the last readout / hint
            except Exception as e:
                # Stay "connected" so the poll keeps retrying and recovers on
                # its own; the LED and the line tell the truth meanwhile.
                self._q.put(("status", f"Lost contact: {e}"))
                self._q.put(("led", False))
            finally:
                self._q.put(("poll_done",))

        self._poll_thread = threading.Thread(
            target=worker, daemon=True, name="nina-status-poll")
        self._poll_thread.start()

    def _update_stack_status(self):
        """Running livestack total, read straight off the parent (same
        thread — the pump runs on the Tk loop)."""
        if self.parent._livestack_dir is None:
            text = "Livestack: off"
        else:
            exp = self.parent._stack_total_exp
            text = (f"Livestack: {self.parent._stack_count} frames, "
                    f"{exp:.0f} s total"
                    f" — {os.path.basename(self.parent._livestack_dir)}")
        if self.v_stack.get() != text:
            self.v_stack.set(text)
        # Mirror the main livestack button: yellow + "Stop" while the stack
        # is watching THIS dialog's folder (counts stay on the status line
        # above — this button is too small for them).
        folder = self._mirror_dir or self.v_workdir.get().strip()
        watching = bool(folder) and self.parent._livestack_dir == folder
        btn_text = "■  Stop Livestack" if watching else "Livestack this folder"
        if self.btn_stack_here.cget("text") != btn_text:
            self.btn_stack_here.config(
                text=btn_text,
                style="Run.TButton" if watching else "Action.TButton")

    # ── connect (probe) ────────────────────────────────────────────────
    @staticmethod
    def _status_line(client):
        """Build the connection status line; returns (text, camera_info)."""
        v = client.version()
        cam = client.camera_info()
        foc = client.focuser_info()
        cam_s = "camera OK" if cam.get("Connected") else "camera NOT connected"
        foc_s = (f"focuser OK @ {foc.get('Position')}"
                 if foc.get("Connected") else "focuser NOT connected")
        return f"API {v} — {cam_s}, {foc_s}", cam

    def _probe(self):
        client = self._client()
        self.v_status.set("Connecting…")
        self.btn_probe.config(state="disabled")

        def worker():
            try:
                text, cam = self._status_line(client)
                self._q.put(("status", text))
                self._q.put(("led", True))
                self._connected = True
                if cam.get("Connected") and cam.get("Gain") is not None:
                    self._q.put(("cam_gain", int(cam["Gain"])))
                # Knowing the pointing costs one request — read it on
                # connect and populate the A-star candidates right away
                # instead of waiting for a Find nearest click.
                try:
                    info = client.mount_info()
                    self._q.put(("mount_pos", self._mount_readout(info)))
                    ra, dec = self._mount_coordinates(info)
                    self._q.put(("astar_rows",
                                 _nearest_astars(self._load_astars(),
                                                 ra, dec)))
                except Exception:
                    # No mount / no catalog: the readout keeps its hint and
                    # Find nearest stays the manual path.
                    pass
            except Exception as e:
                self._connected = False
                self._q.put(("status", f"Connect failed: {e}"))
                self._q.put(("led", False))
            finally:
                # Tk must only be touched from the Tk thread — route the
                # button re-enable through the queue like everything else.
                self._q.put(("probe_done",))

        threading.Thread(target=worker, daemon=True, name="nina-probe").start()

    # ── A-star slew ───────────────────────────────────────────────────────
    @staticmethod
    def _mount_coordinates(info):
        coordinates = info.get("Coordinates") or {}
        # Normalise to J2000 (the catalog's frame) using NINA's Epoch tag, so
        # a JNOW mount does not read a ~0.3° precession offset as a real slew
        # distance.  See _to_j2000.  CATALOG MATH ONLY — the on-screen
        # readout is _mount_readout, which stays in the mount's own frame.
        return _to_j2000(float(coordinates["RADegrees"]),
                         float(coordinates["Dec"]),
                         coordinates.get("Epoch", "J2000"))

    @staticmethod
    def _mount_readout(info):
        """Display string straight from NINA's raw report, in the frame the
        mount and NINA themselves show (JNOW on a JNOW mount), so this
        readout, NINA and the mount control program agree digit for digit.
        The J2000 conversion above is for catalog math only; displaying it
        here would read ~0.3° "off" against the mount."""
        coordinates = info.get("Coordinates") or {}
        return _format_mount_pos(float(coordinates["RADegrees"]),
                                 float(coordinates["Dec"]),
                                 coordinates.get("Epoch", "J2000"))

    @staticmethod
    def _guider_active(info):
        """True when the guider is connected and actively guiding — the only
        case where a focus detour (which stops guiding) should resume it."""
        return is_guiding(info)

    @staticmethod
    def _resume_guiding_after_slew(client):
        """Settle, then restart guiding; returns the status-line suffix.

        Runs on a mount worker thread — it sleeps and then blocks until
        PHD2 has settled (nina_client.guider_start), so never call it
        from the GUI thread.
        """
        time.sleep(GUIDE_SETTLE_S)
        try:
            client.guider_start()
            return "  Guiding resumed."
        except Exception as exc:
            return f"  Guiding resume FAILED: {exc}"

    def _nudge_focus(self, sign):
        """Move the focuser by ±step from wherever it is, on a worker thread.

        Reads the position rather than tracking one: NINA, a sequence or an
        autofocus run may have moved the focuser since the panel last looked,
        and a remembered position would nudge from a fiction.
        """
        if (self._af_thread is not None or self._cap_thread is not None
                or self._foc_thread is not None):
            messagebox.showinfo(
                "Busy", "Wait for autofocus, capture or the current focuser "
                "move to finish — one focuser.", parent=self)
            return
        try:
            step = int(self.v_foc_step.get().strip())
        except ValueError:
            messagebox.showinfo("Focus step", "The step must be a whole "
                                              "number of focuser steps.",
                                parent=self)
            return
        client = self._client()
        delta = sign * step

        def worker():
            try:
                start = int(client.focuser_info()["Position"])
                landed = client.focuser_move(start + delta)
                text = f"Focuser {start} → {landed} ({delta:+d})."
            except Exception as exc:
                text = f"Focus move FAILED: {exc}"
            self._q.put(("foc_done", text))

        self._foc_thread = threading.Thread(target=worker, daemon=True,
                                            name="nina-focus-nudge")
        self._foc_thread.start()
        self.v_af_result.set(f"Moving the focuser {delta:+d}…")

    def _load_astars(self):
        if self._astar_catalog is None:
            with open(ASTAR_CATALOG, newline="", encoding="utf-8") as handle:
                self._astar_catalog = _parse_astar_catalog(handle)
        return self._astar_catalog

    def _mount_busy(self, action):
        if self._af_thread is not None or self._cap_thread is not None:
            messagebox.showinfo(
                "Busy", f"Stop autofocus or capture before {action} — one mount, "
                "one camera.", parent=self)
            return True
        if self._mount_thread is not None:
            messagebox.showinfo("Busy", "A mount operation is already running.",
                                parent=self)
            return True
        return False

    def _find_astars(self):
        if self._mount_busy("finding A stars"):
            return
        try:
            rows = self._load_astars()
        except OSError:
            messagebox.showerror(
                "A-star catalog", "astar_catalog.csv is missing. Run\n"
                "tools/build_astar_catalog.py to build it.", parent=self)
            return
        client = self._client()
        self.v_status.set("Reading mount pointing…")

        def worker():
            try:
                info = client.mount_info()
                self._q.put(("mount_pos", self._mount_readout(info)))
                ra, dec = self._mount_coordinates(info)
                self._q.put(("astar_rows", _nearest_astars(rows, ra, dec)))
                self._q.put(("status", f"Nearest A stars to {ra:.3f}°, "
                                       f"{dec:.3f}° (J2000)"))
            except Exception as exc:
                self._q.put(("status", f"Could not find A stars: {exc}"))

        threading.Thread(target=worker, daemon=True,
                         name="nina-astar-find").start()

    def _show_astars(self, rows):
        self.astar_tree.delete(*self.astar_tree.get_children())
        for row in rows:
            name = (f"HIP {row['hip']}" if row["hip"] else
                    f"HD {row['hd']}" if row["hd"] else "unnamed")
            self.astar_tree.insert("", "end", values=(
                name, f"{row['separation_deg']:.1f}", f"{row['vmag']:.2f}",
                row["sptype"], f"{row['ra_deg']:.3f}", f"{row['dec_deg']:.3f}",
                f"{_autofocus_exposure(row['vmag']):.2f}"))

    def _on_astar_select(self, _event=None):
        # Selecting a star forecasts its autofocus exposure into the Exp
        # field; the user can still hand-tweak it before a Focus run.
        selected = self.astar_tree.selection()
        if selected:
            vmag = float(self.astar_tree.item(selected[0], "values")[2])
            self.v_af_exp.set(f"{_autofocus_exposure(vmag):.2f}")

    def _selected_astar(self):
        selected = self.astar_tree.selection()
        if not selected:
            messagebox.showinfo("A-star slew", "Select an A star first.", parent=self)
            return None
        values = self.astar_tree.item(selected[0], "values")
        return (values[0], float(values[4]), float(values[5]),
                float(values[2]))

    def _start_mount_worker(self, worker, name):
        self._mount_thread = threading.Thread(
            target=worker, daemon=True, name=name)
        self._mount_thread.start()

    def _slew_astar(self):
        star = self._selected_astar()
        if star is None:
            return
        name, ra, dec, _vmag = star
        self.slew_to(name, ra, dec)

    def slew_to(self, name, ra_deg, dec_deg):
        """Public goto — also called by the explorer's catalogue browsers.

        Coordinates are J2000 degrees, the same convention as the A-star
        slew (NINA and the mount driver own the epoch handling). Keeps the
        panel's safety conventions: busy check, explicit confirmation, slew
        on the mount worker thread (guiding is stopped inside mount_slew).
        Guiding active before the slew is resumed on arrival (same pattern
        as the focus-run return; the resume waits out PHD2's star pick and
        settle) — a goto from an unguided state stays unguided.
        """
        if self._mount_busy("slewing"):
            return
        if not messagebox.askyesno(
                "Confirm slew", f"Slew the mount to {name}?\n\n"
                f"RA {ra_deg:.3f}°, Dec {dec_deg:.3f}°", parent=self):
            return
        client = self._client()

        def worker():
            try:
                resume = self._guider_active(client.guider_info())
            except Exception:
                resume = False
            try:
                info = client.mount_slew(ra_deg, dec_deg)
                self._q.put(("mount_pos", self._mount_readout(info)))
                text = f"Slew to {name} complete."
                if resume:
                    text += self._resume_guiding_after_slew(client)
            except Exception as exc:
                text = f"Slew to {name} failed: {exc}"
            self._q.put(("mount_done", text))

        self.v_status.set(f"Slewing to {name}…")
        self.v_slew.set(f"Slewing to {name}…")   # replaced by the outcome
        self._start_mount_worker(worker, "nina-goto-slew")

    def _focus_run(self):
        if self._mount_busy("starting a focus run"):
            return
        star = self._selected_astar()
        if star is None:
            return
        name, ra, dec, vmag = star
        # Selection already forecast the scaled exposure into the field;
        # honour a hand-tweaked value rather than overwriting it here.
        exposure = self.v_af_exp.get().strip() or "3"
        if not messagebox.askyesno(
                "Confirm focus run", f"Slew to {name}, run spectral autofocus "
                f"at {exposure} s per point (V={vmag:.2f}), "
                "then slew back to the current position?\n\n"
                f"RA {ra:.3f}°, Dec {dec:.3f}°", parent=self):
            return
        client = self._client()

        def worker():
            try:
                # Note whether guiding was running before the detour, so the
                # return slew can resume it (the outbound slew stops it).
                try:
                    self._resume_guiding = self._guider_active(
                        client.guider_info())
                except Exception:
                    self._resume_guiding = False
                self._af_return = self._mount_coordinates(client.mount_info())
                info = client.mount_slew(ra, dec)
                self._q.put(("mount_pos", self._mount_readout(info)))
                self._q.put(("mount_done", f"Slew to {name} complete."))
                self._q.put(("af_start", None))
            except Exception as exc:
                self._af_return = None
                self._q.put(("mount_done", f"Focus-run slew failed: {exc}"))

        self.v_status.set(f"Slewing to {name} for autofocus…")
        self.v_slew.set(f"Slewing to {name} for autofocus…")
        self._start_mount_worker(worker, "nina-focus-slew")

    # ── autofocus (worker thread, same code as tools/spectral_autofocus) ──
    def _toggle_autofocus(self):
        if self._af_thread is not None:
            self._af_stop.set()
            self._log("Cancelling after the current exposure…")
            return True
        if self._cap_thread is not None:
            messagebox.showinfo("Busy", "Stop capturing before autofocus — "
                                        "one camera.", parent=self)
            return False
        if self._foc_thread is not None:
            messagebox.showinfo("Busy", "A manual focuser move is still "
                                        "running.", parent=self)
            return False
        if not self._require_nina(self._client()):
            return False

        # Imported here, not at module scope: it drags in the whole science
        # stack (focus_analyzer → specutils/photutils), which the dialog has
        # no other use for, and a missing specutils must degrade to a message
        # rather than break the panel.
        try:
            from tools import spectral_autofocus as af
            from focus_analyzer import config_from_dict
        except ImportError as exc:
            messagebox.showerror(
                "Autofocus", f"The autofocus stack is unavailable: {exc}",
                parent=self)
            return False

        # The CLI's parser IS the parameter builder — option defaults and
        # validation live in one place, and the panel cannot drift from the
        # tool.
        try:
            args = af.parse_args([
                "--host", self.v_host.get().strip(),
                "--port", self.v_port.get().strip() or str(DEFAULT_PORT),
                "--exposure", self.v_af_exp.get().strip() or "3",
                "--step", self.v_af_step.get().strip() or "25",
                "--points", self.v_af_points.get().strip() or "9",
            ])
            # The tool defaults the run folder relative to the cwd, which was
            # the repo root when it ran as a subprocess but is not guaranteed
            # for a GUI (least of all a frozen one) — pin it somewhere writable
            # that a reinstall won't erase.
            args.run_dir = os.path.join(_DATA_ROOT, "focus_runs",
                                        time.strftime("%Y%m%d_%H%M%S"))
            # The LIVE config, straight across — a visually tweaked
            # calibration counts, and no temp file has to carry it.
            cfg = config_from_dict(self.parent._config_dict())
        except SystemExit:      # argparse rejected a field
            messagebox.showerror("Autofocus", "Exposure/step/points must be "
                                              "numeric.", parent=self)
            return False
        except Exception as exc:
            messagebox.showerror("Autofocus", f"Could not read the config: "
                                              f"{exc}", parent=self)
            return False

        client = self._client()
        self._af_stop.clear()
        self.btn_af.config(text="■  Cancel autofocus")
        self.v_af_result.set("Autofocus running…")
        self._plot_af_data(None)
        self._log(f"Autofocus: {args.points} points, step {args.step}, "
                  f"{args.exposure}s")

        def worker():
            try:
                rc, msg = af.run_autofocus(
                    args, client, cfg=cfg,
                    emit=lambda kind, payload: self._q.put(
                        (("af_data" if kind == "afdata" else "log"), payload)),
                    should_stop=self._af_stop.is_set)
            except Exception as exc:            # AutofocusError, NinaError, …
                rc, msg = 1, f"Autofocus failed: {exc}"
                self._q.put(("log", msg))
            self._q.put(("af_done", (rc, msg)))

        self._af_thread = threading.Thread(target=worker, daemon=True,
                                           name="nina-autofocus")
        self._af_thread.start()
        return True

    def _af_finished(self, result):
        self._af_thread = None
        self.btn_af.config(text="▶  Autofocus Here")
        _rc, msg = result
        self.v_af_result.set(msg)
        self._start_return_slew()

    def _start_return_slew(self):
        target, self._af_return = self._af_return, None
        resume, self._resume_guiding = self._resume_guiding, False
        if target is None or self._closed:
            return
        client = self._client()

        def worker():
            try:
                info = client.mount_slew(*target)
                self._q.put(("mount_pos", self._mount_readout(info)))
                text = (f"Return slew complete: RA {target[0]:.3f}°, "
                        f"Dec {target[1]:.3f}°.")
                # Resume the guiding the outbound slew stopped.  This blocks
                # on the worker thread until PHD2 has settled (or the client
                # confirms it from the guider state) — a resume is only worth
                # reporting once it is real.
                if resume:
                    text += self._resume_guiding_after_slew(client)
            except Exception as exc:
                text = ("RETURN SLEW FAILED — mount remains on the A star: "
                        f"{exc}")
            self._q.put(("mount_done", text))

        self.v_status.set("Returning to the saved target…")
        self.v_slew.set("Returning to the saved target…")
        self._start_mount_worker(worker, "nina-focus-return")

    # ── capture: shared bits ───────────────────────────────────────────
    def _workdir(self):
        folder = self.v_workdir.get().strip()
        if not folder:
            messagebox.showinfo("Work folder", "Choose a work folder first.",
                                parent=self)
            return None
        os.makedirs(folder, exist_ok=True)
        # The folder counts as "used" the moment an action starts —
        # persist now, so a crash or hard close doesn't forget it
        # (state was otherwise only saved by a clean dialog close).
        self._save_state()
        return folder

    def _browse_workdir(self):
        folder = filedialog.askdirectory(title="Select capture work folder",
                                         parent=self)
        if folder:
            self.v_workdir.set(folder)

    def _start_capture(self, mode, worker):
        if self._af_thread is not None:
            messagebox.showinfo("Busy", "Wait for autofocus to finish — "
                                        "one camera.", parent=self)
            return False
        if self._cap_thread is not None:
            return False
        self._cap_stop.clear()
        self._cap_mode = mode
        self._mirror_dir = None      # fresh session: no subfolder yet
        self._cap_thread = threading.Thread(
            target=worker, daemon=True, name=f"nina-{mode}")
        self._cap_thread.start()
        return True

    def _capture_finished(self):
        self._cap_thread = None
        mode, self._cap_mode = self._cap_mode, None
        self.btn_loop.config(text="▶  Start capturing")
        self.btn_mirror.config(text="▶  Mirror NINA sequence")
        self._log(f"{'Capture' if mode == 'loop' else 'Mirror'} stopped.")
        # A snapshot-loop capture auto-stacks its folder; stop that stack
        # when the sequence ends so the next target starts clean.  Only when
        # the watch is still on this folder — a stack the user re-pointed
        # elsewhere is left alone.
        # Drain first: the last frame(s) may not be stacked yet (the watch
        # ingests one per tick), and stopping outright would drop them.
        if (mode == "loop"
                and self.parent._livestack_dir == self.v_workdir.get().strip()):
            self.parent._drain_then_stop_livestack()
            self._log("Livestack will stop once the last frames are stacked.")

    def _write_frame(self, folder, name, blob):
        """Write one frame, never over an existing one.

        "xb", not "wb": snapshot names carry a per-run counter and a
        second-resolution stamp, and mirrored frames reuse NINA's own
        filenames — a rerun inside the same second, or a re-mirror after a
        history reset, would otherwise silently truncate a real exposure.
        On collision the frame lands beside the existing one under a _1,
        _2 … suffix; astronomical data is never implicitly replaced.
        """
        stem, ext = os.path.splitext(name)
        path = os.path.join(folder, name)
        for attempt in range(MAX_NAME_COLLISIONS):
            try:
                with open(path, "xb") as f:
                    f.write(blob)
                break
            except FileExistsError:
                path = os.path.join(folder, f"{stem}_{attempt + 1}{ext}")
        else:
            self._q.put(("log", f"  NOT saved: {name} and "
                                f"{MAX_NAME_COLLISIONS} alternatives all "
                                f"exist."))
            return
        self._q.put(("log", f"  saved {os.path.basename(path)} "
                            f"({len(blob) / 1e6:.1f} MB)"))

    # ── capture: snapshot loop (this dialog drives the camera) ─────────
    def _toggle_loop(self):
        if self._cap_thread is not None and self._cap_mode == "loop":
            self._cap_stop.set()
            self._log("Stopping after the current exposure…")
            return
        # Refuse BEFORE the clear-leftovers prompt below, not after starting
        # it: a running mirror leaves _cap_mode == "mirror", so the early
        # return above misses it, and the prompt would delete the frames the
        # mirror is still writing before refusing to start.
        if self._af_thread is not None or self._cap_thread is not None:
            messagebox.showinfo("Busy", "Finish autofocus or the running "
                                "capture first — one camera.", parent=self)
            return
        folder = self._workdir()
        if folder is None:
            return
        try:
            exposure = float(self.v_cap_exp.get().strip())
            count = int(self.v_cap_count.get().strip() or 0)
            gain_s = self.v_cap_gain.get().strip()
            gain = int(gain_s) if gain_s else None
        except ValueError:
            messagebox.showerror("Capture", "Exposure/gain/count must be "
                                            "numeric.", parent=self)
            return
        # A locally driven sequence into a fresh folder: clear leftover FITS so the
        # auto-livestack (below) sees only this run.  Confirm first — deleting
        # frames is data loss: Cancel aborts, No keeps them and stacks all.
        existing = [f for f in os.listdir(folder)
                    if f.lower().endswith((".fit", ".fits", ".fts"))]
        if existing:
            ans = messagebox.askyesnocancel(
                "Start capturing",
                f"{len(existing)} FITS already in\n{folder}\n\n"
                "Clear them before starting?\n"
                "(No keeps them and stacks all; Cancel aborts.)", parent=self)
            if ans is None:
                return
            if ans:
                if self.parent._livestack_dir == folder:
                    self.parent._stop_livestack()
                for f in existing:
                    try:
                        os.remove(os.path.join(folder, f))
                    except OSError:
                        pass
                self._log(f"Cleared {len(existing)} old FITS before capture.")
        client = self._client()
        if not self._require_nina(client):
            return

        def worker():
            n = 0
            fails = 0
            # NINA's capture endpoint hiccups now and then (camera busy for a
            # beat) and returns Success:false — the next call works.  Retry the
            # frame instead of ending the run; only give up after MAX_FAILS in
            # a row, which is a real stall (camera gone), not a transient.
            MAX_FAILS = 5
            while not self._cap_stop.is_set():
                try:
                    blob = client.capture_fits(exposure, gain)
                except Exception as e:
                    fails += 1
                    self._q.put(("log", f"[capture] error {fails}/{MAX_FAILS} "
                                        f"(retrying): {e}"))
                    if fails >= MAX_FAILS:
                        self._q.put(("log", "[capture] too many consecutive "
                                            "errors — stopping."))
                        break
                    self._cap_stop.wait(2.0)   # backoff, still interruptible
                    continue
                fails = 0
                name = time.strftime("snap_%Y%m%d_%H%M%S") + f"_{n:04d}.fits"
                self._write_frame(folder, name, blob)
                n += 1
                if count and n >= count:
                    self._q.put(("log", f"[capture] reached {count} frames."))
                    break
            self._q.put(("cap_done",))

        # Start the livestack BEFORE capture: its "no config loaded — load the
        # last one?" prompt then surfaces up front instead of after the first
        # frames land, and the watch is ready for frame one.
        self._livestack_here()
        if self._start_capture("loop", worker):
            self.btn_loop.config(text="■  Stop capturing")
            self._log(f"Capturing {count if count else '∞'} × {exposure}s "
                      f"to {folder}")

    # ── capture: mirror NINA's own sequence ────────────────────────────
    def _toggle_mirror(self):
        if self._cap_thread is not None and self._cap_mode == "mirror":
            self._cap_stop.set()
            return
        folder = self._workdir()
        if folder is None:
            return
        client = self._client()
        if not self._require_nina(client):
            return

        def worker():
            try:
                seen = client.image_count("LIGHT")
            except Exception as e:
                self._q.put(("log", f"[mirror] cannot reach NINA: {e}"))
                self._q.put(("cap_done",))
                return
            self._q.put(("log", f"[mirror] baseline: {seen} LIGHT frames "
                                f"already in history (ignored)."))
            # Frames land in a per-target subfolder named after NINA's
            # TargetName (mirroring NINA's own layout); ``folder`` stays
            # the base — Livestack/Cleanup follow via self._mirror_dir.
            dest, current_target = folder, None
            while not self._cap_stop.wait(MIRROR_POLL_S):
                try:
                    n = client.image_count("LIGHT")
                    if n < seen:
                        # NINA restarted or its history was cleared, so the
                        # baseline now points past the end.  Without this we
                        # ignore every new frame until the count climbs back
                        # past the stale mark, then index from it and skip
                        # the frames before it.
                        self._q.put(("log", f"[mirror] NINA history reset "
                                            f"({seen} → {n} frames); "
                                            f"re-mirroring from the start."))
                        seen = 0
                    if n <= seen:
                        continue
                    # NINA's own filename (from the history entry) keeps
                    # the local copy a faithful backup of the rig files.
                    hist = client.image_history("LIGHT")
                    for i in range(seen, n):
                        entry = hist[i] if i < len(hist) else {}
                        target = _safe_dirname(
                            str(entry.get("TargetName", "") or ""))
                        if target and target != current_target:
                            # Restart case: the chosen work folder may
                            # already BE the target folder — don't nest.
                            new_dest = (folder if os.path.basename(folder)
                                        == target
                                        else os.path.join(folder, target))
                            os.makedirs(new_dest, exist_ok=True)
                            # Latch ONLY once the folder exists.  Assigning
                            # current_target first meant any makedirs failure
                            # stuck it: the retry then saw target ==
                            # current_target, skipped the mkdir, and wrote
                            # into a folder that was never created — every
                            # later frame failing silently on a 2 s loop.
                            dest, current_target = new_dest, target
                            self._q.put(("mirror_target", dest))
                        blob = client.fetch_fits(i, "LIGHT")
                        try:
                            name = os.path.basename(
                                str(entry.get("Filename", "")).replace(
                                    "\\", "/")) or f"mirror_{i:05d}.fits"
                        except Exception:
                            name = f"mirror_{i:05d}.fits"
                        self._write_frame(dest, name, blob)
                    seen = n
                except Exception as e:
                    self._q.put(("log", f"[mirror] error (will retry): {e}"))
            self._q.put(("cap_done",))

        if self._start_capture("mirror", worker):
            self.btn_mirror.config(text="■  Stop mirroring")
            self._log(f"Mirroring NINA's sequence LIGHTs to {folder}")

    # ── livestack hookup + cleanup ─────────────────────────────────────
    def _toggle_livestack_here(self):
        """Button command only — _livestack_here stays start-only because
        _toggle_loop calls it to auto-stack a starting capture (a toggle
        there would kill a stack already watching the folder)."""
        folder = self._mirror_dir or self.v_workdir.get().strip()
        if folder and self.parent._livestack_dir == folder:
            self.parent._stop_livestack()
            return
        self._livestack_here()

    def _livestack_here(self):
        folder = self._mirror_dir or self._workdir()
        if folder is None:
            return
        if self.parent._livestack_dir == folder:
            self._log("Livestack already watching the work folder.")
            return
        if self.parent._livestack_dir is not None:
            self.parent._stop_livestack()
        self.parent._start_livestack_at(folder)
        self._log(f"Livestack watching {folder}.")

    def _cleanup(self):
        # Never delete under a live capture worker.  It only observes the
        # stop flag between exposures, so a frame already in flight lands
        # after the enumeration and deletion, leaving a "cleaned" folder
        # with a file in it, and racing os.remove against the open write.
        # Instead request the stop and let the user re-run Cleanup once
        # cap_done has cleared the worker; that needs no pending state.
        if self._cap_thread is not None:
            self._cap_stop.set()
            messagebox.showinfo(
                "Cleanup",
                "Capture/mirror is stopping after the current exposure.\n"
                "Run Cleanup again once it has stopped.", parent=self)
            return
        folder = self._mirror_dir or self.v_workdir.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo("Cleanup", "No work folder to clean.",
                                parent=self)
            return
        fits = [f for f in os.listdir(folder)
                if f.lower().endswith((".fit", ".fits", ".fts"))]
        if not fits:
            self._log("Work folder already clean.")
            return
        if not messagebox.askyesno(
                "Cleanup work folder",
                f"Delete {len(fits)} FITS file(s) from\n{folder}?\n\n"
                f"(The livestack on this folder will be stopped first.)",
                parent=self):
            return
        # Same-directory test must survive case/slash/relative spelling
        # differences — a string mismatch here would let cleanup delete
        # FITS out from under a still-polling livestack.
        ls_dir = self.parent._livestack_dir
        if ls_dir is not None:
            try:
                same = os.path.samefile(ls_dir, folder)
            except OSError:
                same = (os.path.normcase(os.path.abspath(ls_dir))
                        == os.path.normcase(os.path.abspath(folder)))
            if same:
                self.parent._stop_livestack()
        errors = 0
        for f in fits:
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                errors += 1
        self._log(f"Cleanup: deleted {len(fits) - errors} file(s)"
                  + (f", {errors} could not be removed." if errors else "."))

    # ── lifecycle ──────────────────────────────────────────────────────
    def _hide(self):
        """Withdraw the panel and give the explorer the full desktop back.

        Everything stays alive — pump, status poll, capture/mirror threads,
        livestack — only the window vanishes.  The explorer's NINA button
        turns into the highlighted reopen handle."""
        self.withdraw()
        self.parent._restore_main_layout()
        self.parent._btn_nina.config(text="◈  NINA active — reopen",
                                     style="Run.TButton")

    def _close(self):
        # Closing the whole dialog deliberately abandons any pending
        # return slew; there is no live Tk owner left to supervise it safely.
        self._af_return = None
        self._closed = True
        if self._pump_after is not None:
            try:
                self.after_cancel(self._pump_after)
            except tk.TclError:
                pass
        self._cap_stop.set()
        self._af_stop.set()
        self._save_state()
        self.parent._btn_nina.config(text="◈  NINA…", style="Action.TButton")
        if getattr(self.parent, "_nina_dialog", None) is self:
            self.parent._nina_dialog = None
        self.destroy()
        self.parent._restore_main_layout()
        # Collect this dialog's Variable/canvas cycles NOW, on the Tk
        # thread — leaving them for a capture worker's GC to finalize
        # is the Tcl_AsyncDelete process abort.
        self.parent.after_idle(gc.collect)


def _selfcheck():
    import io

    rows = _parse_astar_catalog(io.StringIO(
        "hip,hd,ra_deg,dec_deg,vmag,sptype\n"
        "1,,1,0,2.0,A0V\n"
        "2,,350,0,3.0,A1III\n"
        "3,,180,0,4.0,A3V\n"
        "4,,359,0,5.0,A2V\n"))
    # hip 4 is the closest star but fainter than AF_MAX_VMAG — pruned.
    nearest = _nearest_astars(rows, 359, 0, limit=3)
    assert [row["hip"] for row in nearest] == ["1", "2", "3"]
    assert abs(nearest[0]["separation_deg"] - 2.0) < 1e-10
    assert abs(nearest[1]["separation_deg"] - 9.0) < 1e-10
    assert len(_nearest_astars(rows, 0, 90, limit=2)) == 2
    # Exposure scaling: 3 s at the V=2 baseline, flux-ratio scaled — 30 s
    # at the V=4.5 prune limit, ~0.13 s on Sirius.
    assert _autofocus_exposure(2.0) == 3.0
    assert abs(_autofocus_exposure(4.5) - 30.0) < 1e-9
    assert abs(_autofocus_exposure(-1.44) - 0.1262) < 5e-4
    # Round-trips the mock server's fake pointing (tools/mock_nina_server.py).
    assert _format_mount_pos((15 + 59 / 60 + 30 / 3600) * 15,
                             25 + 55 / 60 + 12.613 / 3600, "JNOW") \
        == "Mount: RA 15h59m30.0s  Dec +25°55′12.6″ (JNOW)"
    # The readout shows NINA's raw report in its own frame — no conversion.
    assert NinaDialog._mount_readout(
        {"Coordinates": {"RADegrees": 239.875, "Dec": 25.9202,
                         "Epoch": "JNOW"}}).endswith("(JNOW)")

    # Epoch handling: a JNOW readout is precessed back to J2000 so a star you
    # are on reads ~0 separation instead of the precession offset.
    def _sep(a, b):
        return np.rad2deg(float(angular_separation(
            np.deg2rad(a[0]), np.deg2rad(a[1]),
            np.deg2rad(b[0]), np.deg2rad(b[1]))))

    vega = (279.2347, 38.7837)                      # J2000
    jnow = SkyCoord(*vega, unit="deg", frame=ICRS()).transform_to(
        FK5(equinox=Time.now()))
    jnow = (float(jnow.ra.deg), float(jnow.dec.deg))
    assert _sep(jnow, vega) > 0.1                    # precession is real…
    assert _sep(_to_j2000(*jnow, "JNOW"), vega) < 1e-3   # …and removed
    assert _to_j2000(*vega, "J2000") == vega         # J2000 untouched

    # Guider-resume gate: only a connected+guiding guider is resumed.
    assert NinaDialog._guider_active({"Connected": True, "State": "Guiding"})
    assert not NinaDialog._guider_active({"Connected": True, "State": "Stopped"})
    assert not NinaDialog._guider_active({"Connected": False, "State": "Guiding"})

    # _write_frame must never overwrite a frame already on disk: a rerun in
    # the same second, or a re-mirror after a NINA history reset, reuses the
    # name.  Only self._q is touched, so a stub stands in for the dialog.
    import tempfile
    import types
    stub = types.SimpleNamespace(_q=queue.Queue())
    with tempfile.TemporaryDirectory() as d:
        NinaDialog._write_frame(stub, d, "snap.fits", b"first")
        NinaDialog._write_frame(stub, d, "snap.fits", b"second")
        NinaDialog._write_frame(stub, d, "snap.fits", b"third")
        with open(os.path.join(d, "snap.fits"), "rb") as f:
            assert f.read() == b"first", "original exposure was overwritten"
        with open(os.path.join(d, "snap_1.fits"), "rb") as f:
            assert f.read() == b"second"
        with open(os.path.join(d, "snap_2.fits"), "rb") as f:
            assert f.read() == b"third"
        assert sorted(os.listdir(d)) == ["snap.fits", "snap_1.fits",
                                         "snap_2.fits"]

    print("nina_dialog self-check OK")


if __name__ == "__main__":
    _selfcheck()
