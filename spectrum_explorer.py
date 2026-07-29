"""
spectrum_explorer.py
====================
Interactive Tkinter wrapper around the spectrum extraction pipeline.

Layout
------
Left panel  : editable global constants + Run button
Right panel : matplotlib canvas
               [Main Image — raw frame at load, rotated working image
                with sources / edge line / extraction boxes after a run]
               [Extracted aperture strip]
               [Raw coloured spectrum] [Calibrated & normalised spectrum]
"""

import faulthandler
import sys
if sys.stderr is not None:  # windowed frozen build has no stderr — enable() would raise
    faulthandler.enable()   # thread stacks on a fatal interpreter crash

import gc
import hashlib
import json
import logging
import os
import queue
import threading
import webbrowser
import urllib.parse
from datetime import datetime, timezone

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# Embedded figures are Figure(), never plt.figure(): under TkAgg pyplot creates
# a tk.Tk() manager window per figure and withdraws it.  Those windows are
# invisible and never closed, and Tcl_MainLoop only returns once *every* Tk
# main window is gone, so they keep mainloop() spinning after the root is
# destroyed and leave the process alive with no window.
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import PowerStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from scipy.ndimage import rotate, affine_transform
from auto_stretch.stretch import Stretch
from registration import register_pair, detect_stars, _build_triangles

from calibration_dialog import CalibrationDialog
from full_spectrum_viewer import FullSpectrumDialog
from response_viewer import ResponseCurveDialog
from reference_library_viewer import ReferenceLibraryDialog
from response_calibration_dialog import ResponseCalibrationDialog
from continuum_dialog import ContinuumDialog
from sequence_generator import SequenceGeneratorDialog
from lamost_dialog import LamostDialog
from explorer import catalog_browser
from explorer.be_star_dialog import BeStarDialog
from explorer.wr_star_dialog import WRStarDialog
from explorer.quasar_dialog import QuasarDialog
from explorer.carbon_star_dialog import CarbonStarDialog
from explorer.mira_dialog import MiraDialog
from explorer.s_star_dialog import SStarDialog
from explorer.cv_dialog import CVDialog
from explorer.symbiotic_dialog import SymbioticDialog
from explorer.herbig_dialog import HerbigDialog

# Catalogue browsers behind the "Catalogues" button: menu label ->
# (single-instance attribute, dialog class). All share the same shell
# (explorer/catalog_browser.py) and the same goto/Spec plumbing.
CATALOG_BROWSERS = (
    ("Be Stars", "_be_star_dialog", BeStarDialog),
    ("WR Stars", "_wr_star_dialog", WRStarDialog),
    ("Quasars", "_quasar_dialog", QuasarDialog),
    ("Carbon Stars", "_carbon_star_dialog", CarbonStarDialog),
    ("Mira Variables", "_mira_dialog", MiraDialog),
    ("S-Type Stars", "_s_star_dialog", SStarDialog),
    ("Cataclysmic Variables", "_cv_dialog", CVDialog),
    ("Symbiotic Stars", "_symbiotic_dialog", SymbioticDialog),
    ("Herbig / T Tauri", "_herbig_dialog", HerbigDialog),
)
from predictor_dialog import PredictorDialog
# _DATA_ROOT: where the app may WRITE (beside the .exe when frozen, repo root
# otherwise — _internal/ is wiped by a reinstall).  Imported rather than
# redefined so the rule lives in one place; see nina_dialog and db/spectra_db.
from nina_dialog import NinaDialog, _DATA_ROOT
from first_run_dialog import FirstRunDialog
import tooltip_help as tt

# WCS plate-solve + SIMBAD source identification — see source_identification.py
import source_identification as srcid

# Spectra database (schema + identity waterfall + ingest) — see db/spectra_db.py
from db import spectra_db

# Pure computation helpers — see spectrum_core.py
from spectrum_core import (
    compute_spectrum_width,
    custom_formatter,
    normalize_flux,
    load_calibration_file,
    pixels_to_wavelengths,
    apply_calibration,
    apply_calibration_to_sigma,
    spectrum_fully_in_frame,
    estimate_source_fwhm,
    measure_zero_order_x,
    contaminators_from_sources,
    extract_spectrum,
    best_y_shift,
    plot_reference_lines,
    read_fits_image,
    to_mono,
    rainbow_fill,
    _dao_xy,
    build_sky_col_flag,
    fit_dispersion_poly,
    validate_dispersion_poly,
    dispersion_fit_stats,
)

# Spectrum tilt (derotation) angle detection — see rotation.py
from rotation import (
    estimate_angle,
    FIRST_PASS_CHOICES,
    FIRST_PASS_SHORT,
    DEFAULT_FIRST_PASS_LABEL,
)

# Spectral line catalogue — see wavelength.py
from wavelength import (
    BALMER_LINES,
    HELIUM_LINES,
    OXYGEN_LINES,
    CARBON_LINES,
    CALCIUM_LINES,
    ATMOSPHERIC_LINES,
    CARBON_STAR_LINES,
    HERBIG_LINES,
    WR_WN_LINES,
    WR_WC_LINES,
)


def _hdr(title: str, px: int, font) -> str:
    """'FILES' → '── FILES ─────…', its rule padded to `px` pixels.

    Padded by measurement, not by character count: Header.TLabel asks for a
    tuple of mono families, which Tk cannot resolve, so it renders in Arial —
    proportional, and a fixed count leaves the long titles visibly short.
    Measuring also keeps the headers flush with the buttons under any font
    substitution or display scaling.
    """
    text, dash = f"── {title} ", "─"
    step = font.measure(dash)
    while font.measure(text) + step <= px:
        text += dash
    return text


# ---------------------------------------------------------------------------
# Default global constants (exposed in the left panel)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    TARGET               = "",
    CALIBRATION_FILE     = "",
    ASTAP_PATH           = "C:/Program Files/astap/astap.exe",
    DISPERSION           = 7.7,
    ANGLE                = 0.0,
    NSOURCES             = 10,
    FWHM                 = 5.0,
    APERTURE_HALF_HEIGHT = 13,
    SKY_BAND_GAP         = 3,     # low band (below the aperture)
    SKY_BAND_WIDTH       = 20,
    SKY_BAND_GAP_HI      = 3,     # high band (above); independent so the two
    SKY_BAND_WIDTH_HI    = 20,    # can dodge a contaminator separately
    Y_OFFSET             = 0.0,
    SP_RANGE_MIN         = 4000,
    SP_RANGE_MAX         = 8000,
    AUTOCENTER_TRACE     = True,
    PLOT_BALMER_LINES      = True,
    PLOT_HELIUM_LINES      = False,
    PLOT_ATMOSPHERIC_LINES = True,
    PLOT_OXYGEN_LINES      = False,
    PLOT_CARBON_LINES      = False,
    PLOT_CALCIUM_LINES     = False,
    PLOT_CARBON_STAR_LINES = False,
    PLOT_HERBIG_LINES      = False,
    PLOT_WR_WN_LINES       = False,
    PLOT_WR_WC_LINES       = False,
    PLOT_CAL_LINES         = False,
    PLOT_CONTAM_MASK       = False,
)


# Aperture and sky-gap auto-fit multipliers, applied on source switch
# only.  ±2.5 × FWHM captures ~98% of a Gaussian PSF; sky_gap = 0.7 × FWHM
# puts the inner edge of the sky band ~3.2 σ from the centroid, safely
# past the wings.  The user's ± nudges and direct edits override these
# (they're only applied on a fresh source switch, never re-applied).
# edit: unfocused bloated stars may benefit from smaller apertures
APERTURE_FWHM_MULT = 2.5 / 3.0
SKY_GAP_FWHM_MULT  = 0.7

# Ceiling on the auto-fitted aperture half-height.  DAOStarFinder's FWHM runs
# high — worst on the bright stars, where the zero order and the grating
# confuse it — so the multiplier above can fit an aperture far taller than the
# trace.  That costs SNR (every extra row adds background, not signal) and
# gives a misplaced centroid more room to go unnoticed.  A bright or bloated
# star has ample SNR in a short strip anyway, so cap it: above ~12 px FWHM the
# fit stops growing.  Only the auto-fit is capped — a hand-typed aperture is
# still honoured, the field is the user's.
APERTURE_HALF_MAX = 10

# Rows searched either side by "Centre on trace" → 2N+1 trial extractions.
# 5 covers the misplacement DAO produces on bright stars; the button reports
# an edge-pinned peak so a bigger error just needs a second click.
CENTRE_TRACE_MAX_SHIFT = 5

# Half-width of the on-trace zone whose sources are treated as the target's
# own light (emission line, zero order) rather than contaminators, in units
# of the target's spatial FWHM.
#
# A nova Hα peak centroids ~0.06 × FWHM off the trace; the nearest real
# contaminator is ~0.6 × FWHM out.  The two do not scale alike: the line's
# offset is a centroiding error and shrinks with a tighter PSF, while a star
# sits where it sits, in pixels.  A zone expressed in FWHM therefore widens
# exactly where DAOStarFinder overestimates FWHM (bright stars, zero order
# through the grating), which is where it begins swallowing real
# contaminators.  0.2 keeps roughly 3× margin over the line offset without
# spending the pixels an inflated FWHM would take.
# The log names every source this drops — tune it against those numbers.
TRACE_EXCLUDE_FWHM = 0.2

# Mask half-width for in-aperture contaminating stars, in units of the
# target's spatial FWHM.  ±1.5 × FWHM ≈ ±3.5 σ captures ~99.95% of a
# Gaussian PSF.  Slightly conservative — better to over-mask than miss
# the wings of a contaminator and leave a residual bump in the spectrum.
CONTAM_MASK_FWHM_MULT = 1.5

# Number of trailing strip columns used to set the raw-panel y-limits.
# The blue (left) end of the strip carries the zero-order light and the
# strongest continuum; scaling the y-axis to the redder tail keeps faint
# red-end features visible instead of being flattened by the bright blue.
RAW_YLIM_TAIL_COLS = 600

# Spectral line catalogue lives below, next to the legacy per-element dicts
# it now feeds.  See SPECTRAL_LINES.


def _suppress_cursor_data(image):
    """
    Disable matplotlib's toolbar hover value readout for an AxesImage.

    Matplotlib's ``format_cursor_data`` path computes significant digits
    via ``math.log10`` on the pixel value under the cursor; on a NaN pixel
    this raises ``ValueError: cannot convert float NaN to integer`` and
    spams the console on every mouse move.  The working image (rotated, with
    cval-padded / NaN regions) and the strip bands can both contain NaN, so
    the formatter is overridden to return an empty string.  This affects
    only the hover readout text, nothing rendered.
    """
    try:
        image.format_cursor_data = lambda data: ""
    except Exception:
        pass


def _canvas_without_focus_steal(fig, master):
    """Create a FigureCanvasTkAgg whose construction does not grab focus.

    FigureCanvasTk.__init__ ends with ``self._tkcanvas.focus_set()``
    (matplotlib _backend_tk.py) so a fresh figure can receive key events
    without a click.  The strip canvas is display-only and rebuilt on
    every extraction — during a livestack that grab fires once per frame,
    and on Windows moving keyboard focus into the main toplevel
    *activates* it, dropping whatever dialog the user is working in
    (catalogue browser, cal dialog…) behind the main window, and yanking
    the caret out of any Entry mid-edit.  Shadow Canvas.focus_set for the
    duration of the constructor; a click still focuses the canvas
    (button_press_event calls focus_set at click time), so interactive
    canvases lose nothing.
    """
    tk.Canvas.focus_set = lambda self: None
    try:
        return FigureCanvasTkAgg(fig, master=master)
    finally:
        del tk.Canvas.focus_set   # un-shadow; Misc.focus_set resumes


class _TkLogHandler(logging.Handler):
    """Show library warnings (ASTAP, SIMBAD, …) in the GUI's log pane.

    Emitted from worker threads, so the append is marshalled onto the Tk
    thread with ``after(0, …)`` — the same route the solve worker uses.
    """

    def __init__(self, app):
        super().__init__(level=logging.WARNING)
        self.app = app

    def emit(self, record):
        tag = "error" if record.levelno >= logging.ERROR else "warn"
        try:
            # Queue, never after(): emit() runs on worker threads and a
            # cross-thread Tk call can deadlock the app (see _from_thread).
            self.app._from_thread(self.app._log, self.format(record), tag)
        except (tk.TclError, RuntimeError, AttributeError):
            pass  # app closed / not yet constructed


class SpectrumExplorer(tk.Tk):

    # Click-to-snap radius (data pixels) for picking a source on the
    # working image.  Used both when re-snapping after a Run and when
    # handling main-image clicks.
    SOURCE_SNAP_RADIUS = 40

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.title("Spectrum Explorer")
        self.configure(bg="#0e1014")
        self.resizable(True, True)
        self.minsize(960, 540)
        self._restore_main_layout()

        # Worker threads (WCS solve, library log records) must NEVER call Tk
        # directly: a cross-thread Tk call blocks in Tcl's WaitForMainloop
        # holding the interpreter lock, and if the main thread is meanwhile
        # inside a Tk callback (livestack ingest → _run → canvas draw) the two
        # deadlock. So workers put (fn, args) here; _drain_ui_queue runs them
        # on the Tk thread. Created before any handler/thread that feeds it.
        self._ui_queue = queue.Queue()
        self.after(50, self._drain_ui_queue)

        # State initialised before _build_ui.
        self.rotated_data     = None
        self.top_sources      = None
        # Every source DAOStarFinder found on the rotated frame, (x, y) —
        # not just the nsources brightest.  Contaminator matching needs the
        # faint and the frame-clipped ones too (contaminators_from_sources).
        self.all_sources_xy   = None
        # FITS header of the currently loaded target frame, captured at
        # load time in _load_and_run.  Used by the WCS plate-solve to
        # supply ASTAP its position / scale hints.  None until a frame
        # is loaded; cleared in _reset_analysis.
        self._target_header   = None
        # Per-source catalog identifications, a list aligned 1:1 with
        # top_sources (each entry a srcid.SourceMatch or None).  Populated
        # by _solve_to_wcs; None until a solve runs.  These are tied to the
        # source PIXEL POSITIONS, which only move when the rotation angle
        # changes — so they survive extraction / dispersion / y-offset
        # UPDATEs and are only dropped when the angle changes or in
        # _reset_analysis.  See _load_and_run for the remap-vs-clear logic.
        self.source_matches   = None
        # Rotation angle (deg) at which the current source_matches were
        # solved.  Used to decide whether a re-detection at UPDATE time can
        # keep the matches (same angle → remap by position) or must drop
        # them (angle changed → centroids moved).  None when unsolved.
        self._solved_angle    = None
        # The WCS object from the most recent successful _solve_to_wcs, on the
        # ROTATED frame.  Reused by the sequence generator's WCS positioning
        # mode (world_to_pixel of the target).  Tied to _solved_angle: only
        # valid while the current angle matches.  None until a solve runs.
        self._main_wcs        = None
        self.column_sums      = None
        self.dispersion_nodes = []
        # animation
        self._sequence_dialog = None
        self._last_contam_mask = None
        self._last_sky_col_flag = None

        # Continuum anchor points: list of [wavelength_Å, flux] pairs.
        # Used by the continuum-calibration dialog (slice 3) and the
        # full-spectrum view (slice 5) to produce a continuum-corrected
        # spectrum.  Per-target by nature (they describe the specific
        # observation's flux shape), so deliberately NOT persisted to
        # the config file — reloading a config for a different target
        # would otherwise inherit anchors that don't apply.  Cleared
        # automatically when a new FITS is loaded (see _reset_analysis).
        self.continuum_anchors = []
        self.response_status_var = tk.StringVar()
        self.continuum_status_var = tk.StringVar()
        self._main_click_cid  = None
        self._node_markers    = []
        # Displayed image shape on ax_main — _show_image keeps the
        # user's zoom across same-shaped preserve_view redraws (see there).
        self._main_img_shape = None
        # Artists for the numbered source markers (apertures + name labels)
        # on ax_main.  Tracked separately so _redraw_source_labels can
        # remove just these without disturbing the extraction-box patches
        # drawn by _draw_overlay_boxes.
        self._source_marker_artists = []
        # Extraction-box rectangles on ax_main — the mirror-image list, so
        # _draw_overlay_boxes can remove just its own patches without
        # sweeping the source-marker circles.
        self._box_artists = []
        # Warn-once latches, so these two messages fire on state change
        # rather than on every extraction/redraw.  Coverage is keyed on the
        # inputs that define the condition (table range + display window —
        # stable across angle nudges); the non-monotonic-fit flag clears
        # whenever validation passes or the nodes change.
        self._warned_cal_coverage = None
        self._warned_nonmono = False
        self._last_source_xy  = None   # (x, y) of currently extracted source
        self._last_p          = None   # last extraction params (None until a run)
        self._last_select_free = False # True if current selection is a free
                                       # (arbitrary-point) click, not a detected source
        self._calibrated_wls         = None   # cached wavelength array for calibrated panel
        self._calibrated_flux    = None   # cached normalised flux for calibrated panel
        self._calibrated_sigma   = None   # cached normalised 1σ (background term) for confidence band
        self._calibrated_pixels  = None   # strip-pixel index of each calibrated sample,
                                          # so Add-to-DB can join cal values back onto column_sums
        self._unmasked_col_sums  = None   # pre-contamination-mask counts: the DB stores
                                          # masked columns flagged, never dropped
        # (click-xy, SourceMatch-or-None) from the last Add-to-DB cone query
        # on a free selection, so a repeated Add doesn't re-hit SIMBAD.
        self._free_match_cache   = None
        self._target_fwhm     = None   # measured at last source switch

        # ── Zero-order wavelength anchor (colour-robust scale transfer) ─
        # The dispersion solution is calibrated on one star; its stored
        # nodes are in that star's strip-pixel coordinates, whose origin
        # is the star's extraction start column (its DAO centroid).  The
        # DAO centroid carries a small, colour-dependent offset from the
        # true zero-order peak, so applying the same nodes to a different-
        # coloured star shifts the whole wavelength scale by a fraction of
        # a pixel (~7.6 Å/px → visible Balmer offsets).
        #
        # _calib_anchor_resid : (zero_order_x − x_start) measured on the
        #   CALIBRATION star, captured whenever the nodes are modified (the
        #   active source at that moment *is* the calibration source).
        # _current_anchor_resid : the same residual for the currently
        #   extracted source, refreshed every _display_extraction.
        # get_dispersion_poly applies Δ = current − calib by recomposing
        #   the fit to poly(pixel − Δ).  Either residual None/NaN, or the
        #   toggle off → Δ = 0, so a config with no stored residual behaves
        #   as if the anchor correction were absent.
        self._calib_anchor_resid   = None
        self._current_anchor_resid = None
        self._full_spec_dialog = None  # single-instance reference for the full-spectrum viewer
        self._response_viewer = None  # single-instance reference for the active-response viewer
        self._reference_library_viewer = None  # single-instance reference for the reference-library browser
        self._predictor_dialog = None  # single-instance reference for the spectral-type predictor
        self._nina_dialog = None  # single-instance reference for the NINA remote panel

        # Embedded instrument-response calibration cache.  Populated by
        # _load_config when the loaded JSON carries an inline calibration
        # array, and by set_response_curve when the response dialog
        # computes a new one; consumed by _load_and_run in place of
        # reading the file at p["cal_file"].  Invalidated whenever the
        # calibration file path field is edited or browsed (see trace set
        # up after _build_ui creates the StringVar) so a manual path change
        # reads the new file fresh rather than reusing stale embedded data.
        self._response_df_cache    = None

        # ── Config-dirty tracking ─────────────────────────────────────
        # The two Load/Save buttons are visual indicators of whether
        # there are unsaved changes.  States:
        #   Fresh  (no config loaded, _dirty=False) → Load highlighted
        #   Clean  (config loaded, _dirty=False)    → both normal
        #   Dirty  (_dirty=True)                    → Save highlighted
        #
        # _suppress_dirty gates the trace callbacks so programmatic
        # writes during _load_config, _reset_analysis, _browse_target
        # and Auto-derotate do not flip the flag.  Per-user-action sets
        # (typed entry, ±-step buttons) leave the flag clear so the
        # trace fires normally and marks the config dirty.
        self._dirty                = False
        self._suppress_dirty       = False
        self._loaded_config_path   = None
        # A config chosen before any FITS is loaded: parsed and applied to
        # the parameters immediately, but stashed here so _browse_target
        # can trigger the run once a frame exists.  None = nothing pending.
        # (Once loaded, config parameters live in the widget vars and
        # survive frame loads — _reset_analysis leaves them alone — so no
        # persistent config stash is needed.)
        self._pending_config       = None

        # Last-used folders per dialog category ("fits", "config",
        # "livestack"), so each file dialog reopens where the user last
        # was for THAT purpose — Tk's shared dialog memory would otherwise
        # send Load-config to the stacking folder.  Persisted across
        # sessions in a home-dir dotfile: per-machine UI convenience, not
        # analysis state, so deliberately not in the saved config (§4).
        self._ui_state_path = os.path.join(os.path.expanduser("~"),
                                           ".spectrum_explorer_ui.json")
        try:
            with open(self._ui_state_path, "r") as f:
                self._last_dirs = dict(json.load(f).get("last_dirs", {}))
        except Exception:
            self._last_dirs = {}

        # Livestack folder-watch state.  _dir None = off.  _seen holds paths
        # already processed or present at start (so only genuinely new files
        # fire); _pending tracks new files whose size is still changing (a
        # copy in progress) until it stabilises.  _retry counts read
        # failures per path so a transiently locked file is retried but a
        # repeatably unreadable one is eventually given up on.
        self._livestack_dir     = None
        self._livestack_seen    = set()
        self._livestack_pending = {}
        self._livestack_retry   = {}
        self._livestack_after   = None
        # Livestack autosave: the stack is rewritten after every accepted
        # frame, so there is always an on-disk file of record (Add-to-DB
        # hashes it).  It is written to a session folder under _DATA_ROOT,
        # NOT into the watched folder: capture folders are routinely
        # cloud-synced (Drive/OneDrive), and a multi-MB file rewritten every
        # few seconds keeps the sync client uploading all night.
        # _out_dir and _src_dir outlive _stop_livestack() — Add-to-DB may run
        # on a stopped-but-still-analysed stack, and needs both.
        self._livestack_save_path = None
        self._livestack_out_dir   = None   # where livestack.fit is written
        self._livestack_src_dir   = None   # folder the frames came from
        self._stack_mid_jds       = []
        self._stack_total_exp     = 0.0

        # Virtual live stack.  First frame is the registration reference;
        # each later frame is warped into its coords and accumulated as a
        # coverage-weighted running mean (_stack_sum / _stack_wsum).  When
        # _frame_override is set, _load_and_run uses it instead of reading
        # the target file, so the pipeline runs on the growing stack.
        self._stack_ref         = None   # reference mono frame (float64)
        # Reference stars + triangle descriptors, detected once when the
        # stack starts.  Pure memoization: the reference never changes
        # during a session and detection is deterministic, so passing these
        # to register_pair is bit-identical to letting it re-derive them
        # every frame (which cost ~half the per-frame registration time).
        self._stack_ref_stars   = None
        self._stack_ref_tris    = None
        self._stack_sum         = None   # Σ warped frames
        self._stack_wsum        = None   # Σ per-pixel coverage weights
        self._stack_count       = 0
        self._frame_override    = None
        # Pending debounced refresh of the Full-Spectrum window (after id).
        self._fs_sync_after     = None

        self._build_styles()
        self._build_ui()

        # Invalidate the embedded calibration cache whenever the user
        # edits the calibration-file path.  After _build_ui has created
        # self.v_response_file.  The trace is "write" rather than "read" so
        # programmatic .set() during config-load also fires it — which
        # is fine, because _load_config sets the cache *after* setting
        # the path, so the cache survives.
        self.v_response_file.trace_add("write", self._on_response_file_changed)

        # ── Dirty-flag traces on the StringVars that participate in
        # the saved config.  All respect self._suppress_dirty so
        # programmatic writes during config load / file load / reset /
        # auto-derotate do not falsely mark the config as dirty.
        # v_astap_path is persisted too, so a manual edit (Browse… or
        # typed path) must mark the config dirty; its load-time write at
        # _load_config is already inside the _suppress_dirty block.
        # v_zero_anchor and v_first_pass are likewise persisted — same
        # rule (their _apply_config writes are dirty-suppressed too).
        #
        # v_y_offset is deliberately NOT here, though it IS saved:
        # autocentre rewrites it on essentially every new extraction, so
        # tracking it would leave Save permanently lit, and a dirty flag
        # that is always on carries no information.  It is a per-source
        # placement the app re-derives, not a setting the user chose.
        for var in (self.v_response_file, self.v_angle,
                    self.v_dispersion,
                    self.v_astap_path, self.v_zero_anchor,
                    self.v_first_pass):
            var.trace_add("write", self._mark_dirty)

        # ── Initial button state: Fresh (no config loaded).
        self._update_config_buttons()
        self._update_calibration_status_labels()

        # ── Confirm-on-close when there are unsaved changes.
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Python's cyclic GC runs in whichever thread happens to trip
        # the allocation threshold.  A collection inside a worker
        # thread (WCS solve, NINA capture) finalizes leaked Tk
        # Variables/PhotoImages OFF the Tk thread — Tcl then aborts the
        # whole process ("Tcl_AsyncDelete: async handler deleted by the
        # wrong thread"), killing the app mid-session.  Disable the
        # opportunistic collector and collect on
        # the Tk thread instead, on a timer.  Refcounting still frees
        # everything acyclic immediately.  Remember the prior state so
        # destroy() can restore it — the process-global disable must not
        # outlive this window (embedding, tests, a second app instance).
        self._gc_was_enabled = gc.isenabled()
        gc.disable()
        self._gc_tick()

        # Autoload and display the default target's raw frame, if it
        # exists.  Defer until after the window has finished laying out
        # so the canvas has a real size to draw into.  Run nothing else:
        # derotation and extraction wait for the user.
        self.after_idle(self._load_default_target)

        # First-time user: ask for an approximate Å/px.  After the target
        # load, so the modal opens over a laid-out window rather than an
        # empty one, and so a returning user never sees it at all.
        self.after_idle(self._maybe_first_run)

    # ------------------------------------------------------------------
    # Cross-thread callback marshalling
    # ------------------------------------------------------------------
    def _from_thread(self, fn, *args):
        """Schedule fn(*args) to run on the Tk thread. Safe to call from any
        thread — only a Queue.put happens here, never a Tk call."""
        self._ui_queue.put((fn, args))

    def _drain_ui_queue(self):
        """Pump: run queued worker callbacks on the Tk thread, then reschedule.
        The only place worker-originated callbacks touch Tk."""
        try:
            while True:
                fn, args = self._ui_queue.get_nowait()
                try:
                    fn(*args)
                except Exception:
                    logging.getLogger(__name__).warning(
                        "queued UI callback failed", exc_info=True)
        except queue.Empty:
            pass
        finally:
            self.after(50, self._drain_ui_queue)

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # ── palette ────────────────────────────────────────────────────
        # Single warm-yellow accent (matches the aperture colour in
        # _draw_strip) so chrome and data agree.  No saturated red as
        # default chrome — reserved for genuine errors in the log.
        bg = "#0e1014"  # app background
        panel = "#181c24"  # left panel + log
        panel_2 = "#1e232c"  # entry / inner surfaces
        line = "#262c37"  # hairlines / separators
        line_2 = "#323a47"  # input borders
        fg = "#e6e9ef"  # primary text
        fg_2 = "#aab2c0"  # secondary text
        fg_3 = "#6b7484"  # tertiary / section headers
        acc = "#e0c46c"  # primary accent (warm yellow)
        acc_hover = "#d0b25a"
        acc_ink = "#1a1a1a"  # text colour on the accent button

        # ── type ───────────────────────────────────────────────────────
        # Sans for prose labels, mono for values and identifiers.  Falls
        # back gracefully on Linux/Win/macOS without an extra dependency.
        sans = ("Segoe UI", "SF Pro Text", "Helvetica", "sans-serif")
        mono = ("Consolas", "Menlo", "Courier New", "monospace")
        f_label = (sans, 10)
        f_section = (mono, 9, "bold")  # uppercase, used as section header
        f_value = (mono, 10)
        f_button = (sans, 10, "bold")

        # ── frames / surfaces ──────────────────────────────────────────
        style.configure("TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel)
        style.configure("Inner.TFrame", background=panel)  # for nested entry+button rows
        # Labelled section boxes (NINA panel etc.) — clam's default is a
        # light-grey card that clashes hard with the dark chrome.
        style.configure("TLabelframe", background=panel,
                        bordercolor=line_2, lightcolor=line_2,
                        darkcolor=line_2, relief="solid")
        style.configure("TLabelframe.Label", background=panel,
                        foreground=fg_3, font=f_section)

        # ── labels ─────────────────────────────────────────────────────
        style.configure("TLabel",
                        background=panel, foreground=fg_2, font=f_label)
        # Section header: small uppercase mono, tertiary fg.  Apply tracking
        # in the string itself (Tk has no letter-spacing).
        style.configure("Header.TLabel",
                        background=panel, foreground=fg_3, font=f_section)
        # Same spec, resolved: whatever Tk really renders headers in, which
        # is what _hdr must measure against.
        self._hdr_font = tkfont.Font(font=f_section)
        # Optional brand/title label
        style.configure("Title.TLabel",
                        background=panel, foreground=fg,
                        font=(sans, 12, "bold"))

        # ── entries ────────────────────────────────────────────────────
        style.configure("TEntry",
                        fieldbackground=panel_2, foreground=fg,
                        insertcolor=fg, bordercolor=line_2,
                        lightcolor=line_2, darkcolor=line_2,
                        font=f_value, padding=3)
        style.map("TEntry",
                  bordercolor=[("focus", acc)],
                  lightcolor=[("focus", acc)],
                  darkcolor=[("focus", acc)])

        # ── checkbuttons ───────────────────────────────────────────────
        style.configure("TCheckbutton",
                        background=panel, foreground=fg_2,
                        indicatorcolor=panel_2,
                        font=f_label, padding=(2, 1))
        style.map("TCheckbutton",
                  background=[("active", panel)],
                  foreground=[("active", fg)],
                  indicatorcolor=[("selected", acc),
                                  ("active", line_2)])

        # ── radiobuttons ───────────────────────────────────────────────
        # Mirror the checkbutton treatment so the first-pass method
        # selector matches the rest of the panel.
        style.configure("TRadiobutton",
                        background=panel, foreground=fg_2,
                        indicatorcolor=panel_2,
                        font=f_label, padding=(2, 1))
        style.map("TRadiobutton",
                  background=[("active", panel)],
                  foreground=[("active", fg)],
                  indicatorcolor=[("selected", acc),
                                  ("active", line_2)])

        # ── buttons ────────────────────────────────────────────────────
        # Primary (Run / Auto-derotate): yellow fill, dark ink.
        style.configure("Run.TButton",
                        background=acc, foreground=acc_ink,
                        bordercolor=acc, lightcolor=acc, darkcolor=acc,
                        font=f_button, padding=7, relief="flat")
        style.map("Run.TButton",
                  background=[("active", acc_hover), ("pressed", acc_hover)],
                  bordercolor=[("active", acc_hover)],
                  lightcolor=[("active", acc_hover)],
                  darkcolor=[("active", acc_hover)])

        # Secondary (View calibration curve, …): ghost button.
        style.configure("TButton",
                        background=panel_2, foreground=fg_2,
                        bordercolor=line_2, lightcolor=line_2, darkcolor=line_2,
                        font=(sans, 10), padding=5, relief="flat")
        style.map("TButton",
                  background=[("active", line)],
                  foreground=[("active", fg)])

        # Tonal (outlined accent) — for actions that aren't the main pipeline
        # step but still deserve visual weight: View calibration curve,
        # View full spectrum, Dispersion calibration.  Outlined in the accent
        # colour with accent-coloured text; hover fills with a translucent
        # accent tint.
        acc_tint = "#2a2a1f"  # very dark warm tint that reads as "yellow @ 12%"
        acc_bg = "#302823"   # dark warm fill — clearly distinct from panel
        style.configure("Action.TButton",
                        background=acc_bg,
                        foreground=acc,
                        bordercolor=acc,
                        lightcolor=acc, darkcolor=acc,
                        font=(sans, 10, "bold"),
                        padding=6, relief="flat")
        style.map("Action.TButton",
                  background=[("active", acc_tint), ("pressed", acc_tint)],
                  foreground=[("active", acc)],
                  bordercolor=[("active", acc)],
                  lightcolor=[("active", acc)],
                  darkcolor=[("active", acc)])

        # ── separators ─────────────────────────────────────────────────
        style.configure("TSeparator", background=line)

        # ── scrollbar ──────────────────────────────────────────────────
        # Thumb is deliberately a few shades lighter than the trough and
        # given a wider grip so the scrollable side panes don't hide their
        # scrollbar against the dark panel (the previous panel_2-on-panel
        # thumb was nearly invisible).  Accent on hover/drag.
        sb_thumb = "#4a5365"   # clearly lighter than trough; reads as a grip
        style.configure("Vertical.TScrollbar",
                        background=sb_thumb, troughcolor=panel,
                        bordercolor=panel, arrowcolor=fg,
                        lightcolor=sb_thumb, darkcolor=sb_thumb,
                        width=14)
        style.map("Vertical.TScrollbar",
                  background=[("active", acc), ("pressed", acc)],
                  arrowcolor=[("active", bg)])

        # Cache for use in _build_log_pane / _log (severity colours)
        self._palette = dict(
            bg=bg, panel=panel, panel_2=panel_2,
            fg=fg, fg_2=fg_2, fg_3=fg_3,
            acc=acc, warn="#d97757", err="#d05a5a",
        )

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── root grid: three columns × two rows (log pane on row 1) ──
        # col 0 (280px) : left controls — analysis-affecting knobs
        # col 1 (flex)  : plot canvas
        # col 2 (200px) : right pane — viewing / overlay toggles
        # Both side panes are scrollable so they can grow without
        # forcing window resizes.  Log spans all three columns.
        self.columnconfigure(0, weight=0, minsize=280)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0, minsize=200)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # NEW
        # ── left control panel (scrollable) ──────────────────────────
        left_outer = ttk.Frame(self, style="Panel.TFrame")
        left_outer.grid(row=0, column=0, sticky="nsew")
        left_outer.rowconfigure(0, weight=1)
        left_outer.columnconfigure(0, weight=1)

        ctrl_canvas = tk.Canvas(
            left_outer, bg=self._palette["panel"],
            highlightthickness=0, width=280,
        )
        ctrl_canvas.grid(row=0, column=0, sticky="nsew")

        ctrl_sb = ttk.Scrollbar(
            left_outer, orient="vertical", command=ctrl_canvas.yview)
        ctrl_sb.grid(row=0, column=1, sticky="ns")
        ctrl_canvas.configure(yscrollcommand=ctrl_sb.set)

        ctrl = ttk.Frame(ctrl_canvas, style="Panel.TFrame", padding=12)
        ctrl_window = ctrl_canvas.create_window(
            (0, 0), window=ctrl, anchor="nw")

        def _on_ctrl_configure(event):
            ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
            ctrl_canvas.itemconfig(ctrl_window, width=ctrl_canvas.winfo_width())

        ctrl.bind("<Configure>", _on_ctrl_configure)
        ctrl_canvas.bind("<Configure>",
                         lambda e: ctrl_canvas.itemconfig(ctrl_window, width=e.width))

        # Header-rule width = the pane's content width, i.e. exactly what a
        # full-width button spans (canvas width less the frame's padding).
        self._hdr_px_left = ctrl_canvas.winfo_reqwidth() - 2 * 12

        self._build_controls(ctrl)

        # ── right plot area ───────────────────────────────────────────
        plot_frame = ttk.Frame(self, style="TFrame", padding=4)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self._build_canvas(plot_frame)

        # ── right viewing pane (scrollable, mirrors left construction) ──
        right_outer = ttk.Frame(self, style="Panel.TFrame")
        right_outer.grid(row=0, column=2, sticky="nsew")
        right_outer.rowconfigure(0, weight=1)
        right_outer.columnconfigure(0, weight=1)

        right_canvas = tk.Canvas(
            right_outer, bg=self._palette["panel"],
            highlightthickness=0, width=200,
        )
        right_canvas.grid(row=0, column=0, sticky="nsew")

        right_sb = ttk.Scrollbar(
            right_outer, orient="vertical", command=right_canvas.yview)
        right_sb.grid(row=0, column=1, sticky="ns")
        right_canvas.configure(yscrollcommand=right_sb.set)

        right = ttk.Frame(right_canvas, style="Panel.TFrame", padding=12)
        right_window = right_canvas.create_window(
            (0, 0), window=right, anchor="nw")

        def _on_right_configure(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
            right_canvas.itemconfig(right_window, width=right_canvas.winfo_width())

        right.bind("<Configure>", _on_right_configure)
        right_canvas.bind("<Configure>",
                          lambda e: right_canvas.itemconfig(right_window, width=e.width))

        self._hdr_px_right = right_canvas.winfo_reqwidth() - 2 * 12
        self._build_right_pane(right)

        # ── Mouse-wheel scrolling for both panes (§3.2) ────────────────
        # On Windows the wheel event is delivered to the widget under the
        # pointer, so per-widget bindings on the canvas/frame died the
        # moment the cursor sat over any child (entry, checkbutton,
        # label).  One application-level handler instead routes the event
        # by walking the target widget's ancestry to whichever pane
        # contains it; events over the plot area or over dialogs (their
        # ancestry never reaches these panes) fall through untouched, so
        # dialogs keep their own wheel handling.
        def _wheel_pane(widget):
            w = widget
            while w is not None:
                if w is ctrl_canvas:
                    return ctrl_canvas
                if w is right_canvas:
                    return right_canvas
                w = getattr(w, "master", None)
            return None

        def _on_app_mousewheel(event):
            # event.widget is a string for widgets destroyed mid-event.
            if not hasattr(event.widget, "winfo_exists"):
                return
            pane = _wheel_pane(event.widget)
            if pane is None:
                return
            if event.num == 4:      # Linux scroll up
                pane.yview_scroll(-1, "units")
            elif event.num == 5:    # Linux scroll down
                pane.yview_scroll(1, "units")
            else:                   # Windows / macOS
                pane.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.bind_all("<MouseWheel>", _on_app_mousewheel)
        self.bind_all("<Button-4>", _on_app_mousewheel)
        self.bind_all("<Button-5>", _on_app_mousewheel)

        # ── bottom log pane ───────────────────────────────────────────
        log_frame = ttk.Frame(self, style="Panel.TFrame", padding=(8, 4))
        log_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        self._build_log_pane(log_frame)

        # Hover help: tip every control keyed by its visible label. The
        # synthetic-id controls (first-pass radios, ± nudge buttons) were
        # tipped explicitly above.
        tt.attach_tree(self, "SpectrumExplorer")

    def _build_controls(self, parent):
        row = 0

        def section(label):
            nonlocal row
            ttk.Label(parent, text=label, style="Header.TLabel").grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(10, 2))
            row += 1

        # Every parameter row lines its entry up in the same place.  Two
        # things are needed for that and both are load-bearing:
        #  - a fixed label width, or each row sizes its label to its own text;
        #  - every row in its OWN frame spanning both parent columns, never
        #    gridded into the shared column 0.  Anything wide in that column
        #    (the "Auto center" checkbox) stretches it and shoves the rows
        #    that live in it sideways, past the rows that don't.
        # Widths are in characters — LABEL_W fits the longest first label
        # ("Aper half-h"), LABEL_B_W the longest second one ("FWHM (px)").
        LABEL_W, LABEL_B_W, ENTRY_W = 11, 10, 6

        def row_frame(pady=1):
            nonlocal row
            frm = ttk.Frame(parent, style="Panel.TFrame")
            frm.grid(row=row, column=0, columnspan=2, sticky="w", pady=pady)
            row += 1
            return frm

        def field(label, key, width=ENTRY_W):
            frm = row_frame()
            ttk.Label(frm, text=label, width=LABEL_W).grid(
                row=0, column=0, sticky="w")
            var = tk.StringVar(value=str(DEFAULTS[key]))
            ttk.Entry(frm, textvariable=var, width=width).grid(
                row=0, column=1, sticky="w", padx=(4, 0))
            return var

        def field_pair(label_a, key_a, label_b, key_b, width=ENTRY_W):
            """Two related numbers on one line — label, entry, label, entry.

            Both labels stay real Labels carrying their own text, so
            tooltip_help.attach_tree still finds and tips each one
            individually (it matches on widget text, and it walks into
            frames).  Pairing them here is purely a space decision: the pane
            is a tight column and these read as pairs anyway (gap/width,
            min/max).  The fixed label widths keep the first entry in line
            with the plain rows' and the second in line with each other's.
            """
            frm = row_frame()
            var_a = tk.StringVar(value=str(DEFAULTS[key_a]))
            var_b = tk.StringVar(value=str(DEFAULTS[key_b]))
            ttk.Label(frm, text=label_a, width=LABEL_W).grid(
                row=0, column=0, sticky="w")
            ttk.Entry(frm, textvariable=var_a, width=width).grid(
                row=0, column=1, sticky="w", padx=(4, 8))
            ttk.Label(frm, text=label_b, width=LABEL_B_W).grid(
                row=0, column=2, sticky="w")
            ttk.Entry(frm, textvariable=var_b, width=width).grid(
                row=0, column=3, sticky="w", padx=(4, 0))
            return var_a, var_b

        def check(label, key):
            nonlocal row
            var = tk.BooleanVar(value=bool(DEFAULTS[key]))
            ttk.Checkbutton(parent, text=label, variable=var).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=1)
            row += 1
            return var

        ttk.Label(parent, text="SPECTRUM EXPLORER",
                  style="Header.TLabel",
                  font=("Courier New", 12, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(0, 6))
        row += 1

        # Files
        section(_hdr("FILES", self._hdr_px_left, self._hdr_font))

        # NINA remote panel: probe/autofocus/capture against the imaging
        # rig's Advanced API — see nina_dialog.py.  Sits above the file
        # buttons because it is where frames come FROM on a remote night.
        self._btn_nina = ttk.Button(parent, text="◈  NINA…",
                                    style="Action.TButton",
                                    command=self._show_nina)
        self._btn_nina.grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        row += 1

        # Target FITS: full-width button (same look as Start Livestack).
        # v_target stays the path state — set by this browse, Livestack
        # and config load; the loaded file name is reported in the log.
        self.v_target = tk.StringVar(value=str(DEFAULTS["TARGET"]))
        ttk.Button(parent, text="Load Target FITS…", style="Action.TButton",
                   command=self._browse_target).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        row += 1

        # Livestack: monitor a folder and auto-load+run each new FITS as it
        # appears.  Toggle button — text/style flip while active.
        self._btn_livestack = ttk.Button(
            parent, text="Start Livestack", style="Action.TButton",
            command=self._toggle_livestack)
        self._btn_livestack.grid(row=row, column=0, columnspan=2,
                                 sticky="ew", pady=(0, 4))
        row += 1

        # The response/calibration-file row lives in the NON-LINEAR
        # CALIBRATION block ("Legacy response", _build_nodes_panel) — it
        # loads a response curve produced by another program, so it
        # belongs next to "Calibrate instrument response…".

        # "View active response curve…" moved to the right pane next to
        # "View full spectrum" — both are passive viewers, not analysis
        # controls.

        # Save/Load full analysis config — sits with the file controls
        # because it's a file operation whose effects span the whole
        # left panel (calibration file + embedded array, angle, linear
        # dispersion, y-offset, plus the non-linear nodes).  Loading
        # triggers an automatic re-run; saving captures the current
        # state of all five parameter groups.  See _save_config /
        # _load_config.
        cfg_frame = ttk.Frame(parent, style="Panel.TFrame")
        cfg_frame.grid(row=row, column=0, columnspan=2, sticky="ew",
                       pady=(0, 4))
        cfg_frame.columnconfigure(0, weight=1)
        cfg_frame.columnconfigure(1, weight=1)
        self._btn_load_cfg = ttk.Button(
            cfg_frame, text="Load config…",
            style="Action.TButton",
            command=self._load_config)
        self._btn_load_cfg.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self._btn_save_cfg = ttk.Button(
            cfg_frame, text="Save config…",
            style="Action.TButton",
            command=self._save_config)
        self._btn_save_cfg.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        row += 1

        # Auto-derotate — same red button style as Update.  Stays up here
        # with the file controls: it runs the whole pipeline, unlike the
        # Rot. angle field it feeds (now down in Extraction parameters).
        ttk.Button(parent, text="▶ AUTO-DEROT. and PROC", style="Run.TButton",
                   command=self._derotate).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        row += 1

        # First-pass method selector — chooses which whole-frame detector
        # Auto-derotate uses (see rotation.FIRST_PASS_CHOICES).  The label
        # is resolved to a method in _derotate.
        self.v_first_pass = tk.StringVar(value=DEFAULT_FIRST_PASS_LABEL)
        fp_frame = ttk.Frame(parent, style="Panel.TFrame")
        fp_frame.grid(row=row, column=0, columnspan=2, sticky="ew",
                      pady=(0, 4))
        # One line, three columns: the labels are abbreviated for display
        # (FIRST_PASS_SHORT) while the stored value stays the full label, so
        # a saved config's `first_pass` still validates.
        for i, (label, _method) in enumerate(FIRST_PASS_CHOICES):
            fp_frame.columnconfigure(i, weight=1)
            rb = ttk.Radiobutton(fp_frame, text=FIRST_PASS_SHORT.get(label,
                                                                     label),
                                 value=label, variable=self.v_first_pass)
            rb.grid(row=0, column=i, sticky="w", pady=1)
            tt.attach(rb, "SpectrumExplorer", "first_pass_method")
        row += 1

        # Sources
        section(_hdr("SOURCES", self._hdr_px_left, self._hdr_font))

        self.v_nsources, self.v_fwhm = field_pair(
            "N sources", "NSOURCES", "FWHM (px)", "FWHM")

        # Dispersion — entry plus the geometry calculator that seeds it.
        # Same row shape as Rot. angle above (entry then a trailing button).
        # The calculator is the first-run dialog reopened: it is the only
        # place the grating geometry lives, so this is the way back to it.
        section(_hdr("INITIAL LINEAR DISP", self._hdr_px_left, self._hdr_font))
        self.v_dispersion = tk.StringVar(value=str(DEFAULTS["DISPERSION"]))
        disp_frame = row_frame()
        ttk.Label(disp_frame, text="Å / px", width=LABEL_W).grid(
            row=0, column=0, sticky="w")
        ttk.Entry(disp_frame, textvariable=self.v_dispersion,
                  width=ENTRY_W).grid(row=0, column=1, sticky="w", padx=(4, 0))
        _dcalc = ttk.Button(disp_frame, text="calc", width=5,
                            command=self._open_dispersion_calculator)
        _dcalc.grid(row=0, column=2, padx=(4, 0))
        tt.attach(_dcalc, "SpectrumExplorer", "dispersion_calc")
        self.v_sp_min, self.v_sp_max = field_pair(
            "λ min (Å)", "SP_RANGE_MIN", "λ max (Å)", "SP_RANGE_MAX")

        # Extraction
        section(_hdr("EXTRACTION PARAMETERS", self._hdr_px_left, self._hdr_font))

        # Rotation angle: entry and its ± nudges on one line (same shape as
        # the strip y-offset row below).  The step size is not labelled —
        # it lives in the buttons' own tooltips, and a static "step 0.05°"
        # costs a whole row of a pane that has none to spare.  The "Auto"
        # action sits as a primary button above, since it triggers a full
        # pipeline run (not just a derotation) and deserves Update's weight.
        # The ° rides after the box rather than in the label: it keeps the
        # label inside LABEL_W, which is what lines this entry up with the
        # rest of the section.
        self.v_angle = tk.StringVar(value=str(DEFAULTS["ANGLE"]))
        ang_frame = row_frame()
        ttk.Label(ang_frame, text="Rot. angle", width=LABEL_W).grid(
            row=0, column=0, sticky="w")
        ttk.Entry(ang_frame, textvariable=self.v_angle, width=ENTRY_W).grid(
            row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Label(ang_frame, text="°").grid(row=0, column=2, padx=(3, 0))
        _am = ttk.Button(ang_frame, text="−", width=2,
                         command=self._angle_minus)
        _am.grid(row=0, column=3, padx=(4, 0))
        tt.attach(_am, "SpectrumExplorer", "angle_minus")
        _ap = ttk.Button(ang_frame, text="+", width=2,
                         command=self._angle_plus)
        _ap.grid(row=0, column=4, padx=(2, 0))
        tt.attach(_ap, "SpectrumExplorer", "angle_plus")

        self.v_aper_half   = field("Aper half-h",      "APERTURE_HALF_HEIGHT")
        # Two sky bands, independently offset: "high" sits above the aperture,
        # "low" below.  In a crowded field a contaminator on one side can be
        # dodged by widening the gap on that side alone.  Ordered high→low to
        # mirror their on-image layout (high band on top); each band's gap and
        # width share a line, since they are only ever read as a pair.
        self.v_sky_gap_hi, self.v_sky_width_hi = field_pair(
            "Bkg Hi gap", "SKY_BAND_GAP_HI", "width", "SKY_BAND_WIDTH_HI")
        self.v_sky_gap, self.v_sky_width = field_pair(
            "Bkg Lo gap", "SKY_BAND_GAP", "width", "SKY_BAND_WIDTH")

        # Strip y offset — entry + ± fine-tune buttons.  Entry is the
        # primary state holder so the offset survives a Run; the buttons
        # nudge the entry value and re-extract.
        self.v_y_offset = tk.StringVar(value=str(DEFAULTS["Y_OFFSET"]))
        yoff_frame = row_frame(pady=(4, 1))
        ttk.Label(yoff_frame, text="Y offset", width=LABEL_W).grid(
            row=0, column=0, sticky="w")
        ttk.Entry(yoff_frame, textvariable=self.v_y_offset,
                  width=ENTRY_W).grid(row=0, column=1, sticky="w", padx=(4, 0))
        _ym = ttk.Button(yoff_frame, text="−", width=2,
                         command=self._aper_minus)
        _ym.grid(row=0, column=2, padx=(4, 0))
        tt.attach(_ym, "SpectrumExplorer", "aper_minus")
        _yp = ttk.Button(yoff_frame, text="+", width=2,
                         command=self._aper_plus)
        _yp.grid(row=0, column=3, padx=(2, 0))
        tt.attach(_yp, "SpectrumExplorer", "aper_plus")

        # Toggle and button share a line: they are the automatic and manual
        # halves of one action, and the button only earns its space when the
        # toggle is off (it highlights then — _update_centre_button).
        self.v_autocenter = tk.BooleanVar(
            value=bool(DEFAULTS["AUTOCENTER_TRACE"]))
        ttk.Checkbutton(parent, text="Auto center",
                        variable=self.v_autocenter,
                        command=self._update_centre_button).grid(
            row=row, column=0, sticky="w", pady=(4, 1))
        self._btn_centre = ttk.Button(parent, text="Centre on trace",
                                      command=self._centre_on_trace)
        self._btn_centre.grid(row=row, column=1, sticky="ew", padx=(4, 0),
                              pady=(4, 1))
        tt.attach(self._btn_centre, "SpectrumExplorer", "centre_on_trace")
        self._update_centre_button()
        row += 1

        # In-aperture contaminant masking belongs with Extraction — it
        # NaN-masks columns of the extracted spectrum where another
        # source overlaps the aperture row.  This changes data, not
        # display, which is why it lives on the left pane.
        self.v_contam_mask = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_CONTAM_MASK"]))
        ttk.Checkbutton(parent, text="Mask in-aperture stars",
                        variable=self.v_contam_mask,
                        command=self._on_contam_mask_toggle).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(4, 1))
        row += 1

        # Diagnostic toggle, not a second algorithm: an emission line is
        # never a contaminant, so off (the default) is the correct mode.
        # Checked sets trace_exclude=0, letting on-trace WR/nova peaks
        # reappear as contaminants for a visual A/B on emission-line frames.
        self.v_contam_legacy = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Incl. on-trace peaks (diag.)",
                        variable=self.v_contam_legacy,
                        command=self._on_contam_mask_toggle).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 1))
        row += 1

        # Free selection: when on, a left-click anywhere on the working
        # image extracts a spectrum from that exact pixel instead of
        # snapping to the nearest detected source.  Useful for sources
        # not in the detected set (faint companions, the central star
        # of a planetary nebula, an arbitrary point on a nebula, etc.).
        self.v_free_select = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Free selection (click any point)",
                        variable=self.v_free_select).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(4, 1))
        row += 1

        # Zero-order wavelength anchor: when on, the dispersion solution
        # calibrated on one star is re-anchored to each source's measured
        # zero-order peak, cancelling the colour-dependent centroid offset
        # that otherwise shifts the whole wavelength scale by a fraction of
        # a pixel between stars.  Off → the scale stays anchored to the raw
        # extraction start column.  Toggling re-extracts so the
        # effect is visible immediately.
        self.v_zero_anchor = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Zero-order wavelength anchor",
                        variable=self.v_zero_anchor,
                        command=self._on_zero_anchor_toggle).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(4, 1))
        row += 1

        # Reference-line toggles and the View Full Spectrum button now
        # live in the right pane (_build_right_pane) — they are
        # cosmetic / viewing operations, not analysis knobs.  The Plate
        # Solve controls (ASTAP path + Solve to WCS + recovered-source
        # info) also live in the right pane (_build_right_pane).

        # Separator + run button
        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        ttk.Button(parent, text="▶  UPDATE", style="Run.TButton",
                   command=self._run).grid(
            row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        # ── Dispersion calibration nodes ─────────────────────────────
        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        row += 1

        self._build_nodes_panel(parent, start_row=row)

        parent.columnconfigure(1, weight=1)

    def _build_right_pane(self, parent):
        """
        Build the right pane: viewing / overlay controls.

        Holds the reference-line toggles (Balmer, Helium, atmospheric,
        Oxygen, Carbon, Calcium, Carbon Star), the calibration-node-marker
        toggle, and the View Full Spectrum button.  Everything here is
        cosmetic — toggling these controls cannot change extracted
        spectrum values, only what gets drawn on top of them.

        Parent is a Panel.TFrame embedded in a scrollable canvas, so
        nothing here needs to fit in any particular height.
        """
        parent.columnconfigure(0, weight=1)
        row = 0

        ttk.Label(parent, text=_hdr("REFERENCE LINES", self._hdr_px_right, self._hdr_font),
                  style="Header.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        self.v_balmer_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_BALMER_LINES"]))
        ttk.Checkbutton(parent, text="Plot Balmer lines",
                        variable=self.v_balmer_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        self.v_helium_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_HELIUM_LINES"]))
        ttk.Checkbutton(parent, text="Plot Helium lines",
                        variable=self.v_helium_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        self.v_atmospheric_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_ATMOSPHERIC_LINES"]))
        ttk.Checkbutton(parent, text="Plot atmospheric lines",
                        variable=self.v_atmospheric_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        self.v_oxygen_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_OXYGEN_LINES"]))
        ttk.Checkbutton(parent, text="Plot Oxygen lines",
                        variable=self.v_oxygen_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        self.v_carbon_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_CARBON_LINES"]))
        ttk.Checkbutton(parent, text="Plot Carbon lines",
                        variable=self.v_carbon_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        self.v_calcium_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_CALCIUM_LINES"]))
        ttk.Checkbutton(parent, text="Plot Calcium lines",
                        variable=self.v_calcium_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        self.v_carbon_star_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_CARBON_STAR_LINES"]))
        ttk.Checkbutton(parent, text="Plot Carbon Star lines",
                        variable=self.v_carbon_star_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        self.v_herbig_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_HERBIG_LINES"]))
        ttk.Checkbutton(parent, text="Plot Herbig / Fe II lines",
                        variable=self.v_herbig_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        ttk.Label(parent, text=_hdr("WOLF-RAYET", self._hdr_px_right, self._hdr_font),
                  style="Header.TLabel").grid(
            row=row, column=0, sticky="w", pady=(12, 2))
        row += 1

        self.v_wr_wn_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_WR_WN_LINES"]))
        ttk.Checkbutton(parent, text="Plot WR WN lines",
                        variable=self.v_wr_wn_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        self.v_wr_wc_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_WR_WC_LINES"]))
        ttk.Checkbutton(parent, text="Plot WR WC lines",
                        variable=self.v_wr_wc_lines,
                        command=self._on_ref_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=1)
        row += 1

        # Calibration-node markers — display only.  Moved here from
        # the Non-Linear Calibration block on the left pane.
        self.v_cal_lines = tk.BooleanVar(
            value=bool(DEFAULTS["PLOT_CAL_LINES"]))
        ttk.Checkbutton(parent, text="Show calibration lines",
                        variable=self.v_cal_lines,
                        command=self._on_cal_lines_toggle).grid(
            row=row, column=0, sticky="w", pady=(8, 1))
        row += 1

        # ── View full spectrum ───────────────────────────────────────
        # No header on this group, so its top pad is what sets it apart
        # from the checkboxes above.
        ttk.Button(parent, text="⤢  View full spectrum",
                   style="Action.TButton",
                   command=self._show_full_spectrum).grid(
            row=row, column=0, sticky="ew", pady=(16, 0))
        row += 1

        ttk.Button(parent, text="⤢  View active response curve…",
                   style="Action.TButton",
                   command=self._show_cal_curve).grid(
            row=row, column=0, sticky="ew", pady=(4, 0))
        row += 1

        ttk.Button(parent, text="⤢  Browse reference library…",
                   style="Action.TButton",
                   command=self._show_reference_library).grid(
            row=row, column=0, sticky="ew", pady=(4, 0))
        row += 1

        # ── Plate solve / source identification ──────────────────────
        # Plate-solves the working frame (ASTAP) and labels each detected
        # source with its catalog name via a SIMBAD cone on the source's
        # sky position.  The recovered details for the currently selected
        # source are shown in the info box below.  The ASTAP path is
        # per-machine and persists in the analysis config.
        ttk.Label(parent, text=_hdr("PLATE SOLVE", self._hdr_px_right, self._hdr_font),
                  style="Header.TLabel").grid(
            row=row, column=0, sticky="w", pady=(16, 4))
        row += 1

        self.v_astap_path = tk.StringVar(value=str(DEFAULTS["ASTAP_PATH"]))
        astap_frame = ttk.Frame(parent, style="Panel.TFrame")
        astap_frame.grid(row=row, column=0, sticky="ew", pady=1)
        astap_frame.columnconfigure(1, weight=1)
        ttk.Label(astap_frame, text="ASTAP").grid(
            row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(astap_frame, textvariable=self.v_astap_path, width=14).grid(
            row=0, column=1, sticky="ew")
        ttk.Button(astap_frame, text="…", width=2,
                   command=self._browse_astap).grid(
            row=0, column=2, padx=(2, 0))
        row += 1

        self._btn_solve = ttk.Button(parent, text="✦  Solve to WCS",
                                     style="Action.TButton",
                                     command=self._solve_to_wcs)
        self._btn_solve.grid(row=row, column=0, sticky="ew", pady=(4, 4))
        row += 1

        # Hide/show the resolved names on the overlay (numbers + circles
        # always stay — they are the source-switching UI).
        self.v_show_names = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Show source names",
                        variable=self.v_show_names,
                        command=lambda: self._redraw_source_labels(
                            draw=True)).grid(
            row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        # Read-only info box for the recovered catalog details of the
        # currently selected source.  A tk.Text (not ttk) so it can be
        # multi-line and themed; kept disabled so the user can't edit it.
        info_frame = ttk.Frame(parent, style="Panel.TFrame")
        info_frame.grid(row=row, column=0, sticky="ew", pady=(0, 4))
        info_frame.columnconfigure(0, weight=1)
        self.source_info_text = tk.Text(
            info_frame, height=7, width=28, wrap="word",
            background="#0f0f1a", foreground="#aab2c0",
            insertbackground="#aab2c0",
            relief="flat", borderwidth=1,
            font=("Consolas", 9), state="disabled")
        self.source_info_text.grid(row=0, column=0, sticky="ew")
        self._set_source_info("No WCS solve yet.\nPress “Solve to WCS”.")
        row += 1

        # Open the SIMBAD page for the selected source.  Disabled until a
        # source with a catalog match is selected (enabled by
        # _update_source_info).
        self.btn_check_simbad = ttk.Button(
            parent, text="↗  Check SIMBAD",
            style="Action.TButton",
            command=self._open_simbad_page, state="disabled")
        self.btn_check_simbad.grid(row=row, column=0, sticky="ew", pady=(0, 4))
        row += 1

        # Open the LAMOST DR11 low-res spectrum for the selected source.
        # Enabled alongside Check SIMBAD: both depend on a resolved
        # catalog name from the WCS solve.
        self.btn_lamost = ttk.Button(
            parent, text="↗  LAMOST",
            style="Action.TButton",
            command=self._show_lamost, state="disabled")
        self.btn_lamost.grid(row=row, column=0, sticky="ew", pady=(0, 4))
        row += 1

        # Catalogue browsers (target planning) — independent of any loaded
        # frame, so always enabled. Nine of them now, so one button posting
        # a menu instead of a button per catalogue.
        self._btn_catalogs = ttk.Button(
            parent, text="✶  Catalogues",
            style="Action.TButton",
            command=self._post_catalog_menu)
        self._btn_catalogs.grid(row=row, column=0, sticky="ew", pady=(0, 4))
        row += 1

        # Save the current extraction to the spectra database: target
        # identity, raw + calibrated spectrum, and the live config
        # snapshot.  Always enabled; the handler validates and explains.
        ttk.Button(parent, text="🗄  Add Spectrum to DB",
                   style="Action.TButton",
                   command=self._add_spectrum_to_db).grid(
            row=row, column=0, sticky="ew", pady=(0, 4))
        row += 1

        # ── Tools ────────────────────────────────────────────────────
        # Animate = the sequence generator (controls + folder selection
        # live in its own window; the pane keeps only the launch button).
        # Predictor = standalone Pickles template match, the same panel
        # the continuum dialog shows as its 4th plot, reachable without
        # opening the continuum workflow.
        ttk.Label(parent, text=_hdr("TOOLS", self._hdr_px_right, self._hdr_font),
                  style="Header.TLabel").grid(
            row=row, column=0, sticky="w", pady=(16, 4))
        row += 1

        ttk.Button(parent, text="▶  Animate",
                   style="Action.TButton",
                   command=self._show_sequence_generator).grid(
            row=row, column=0, sticky="ew", pady=(0, 4))
        row += 1

        ttk.Button(parent, text="✧  Predictor",
                   style="Action.TButton",
                   command=self._show_predictor).grid(
            row=row, column=0, sticky="ew", pady=(0, 1))
        row += 1

        ttk.Label(parent, text="experimental — Pickles spectral-type match",
                  foreground="#8890a0").grid(
            row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

    def _build_canvas(self, parent):
        plt.style.use("dark_background")
        BG, PANEL = "#1a1a2e", "#0f0f1a"

        parent.rowconfigure(0, weight=1)   # main image
        parent.rowconfigure(1, weight=0)   # strip (fixed height)
        parent.rowconfigure(2, weight=0)   # spectra
        parent.columnconfigure(0, weight=1)

        # ── Top: main image ───────────────────────────────────────────
        main_frame = ttk.Frame(parent, style="TFrame")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self.fig_main = Figure(facecolor=BG)
        self.ax_main = self.fig_main.add_subplot(111)
        self.ax_main.set_facecolor(PANEL)
        # Tight margins — the image is the content, not a plot.  Tick
        # labels are off (see _strip_main_ticks), so the default left/bottom
        # padding matplotlib reserves for them is not needed.
        # `top` reserves just enough room for the title; `bottom` for
        # the navigation toolbar's spine.
        self.fig_main.subplots_adjust(
            left=0.01, right=0.99, top=0.94, bottom=0.02)
        self.ax_main.set_title("No target loaded",
                               color="#a0a0c0", fontsize=8, pad=4)
        self._strip_main_ticks()

        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=main_frame)
        self.canvas_main.draw()
        self._mpl_widget = self.canvas_main.get_tk_widget()
        self._mpl_widget.grid(row=0, column=0, sticky="nsew")
        tb_main = ttk.Frame(main_frame, style="TFrame")
        tb_main.grid(row=1, column=0, sticky="ew")
        self._toolbar_main = NavigationToolbar2Tk(self.canvas_main, tb_main)

        # No overlay canvas — boxes drawn as mpl patches on ax_main with a full redraw

        # ── Middle: scrollable pixel-exact strip ──────────────────────
        strip_outer = ttk.Frame(parent, style="Panel.TFrame")
        strip_outer.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        strip_outer.columnconfigure(0, weight=1)

        ttk.Label(strip_outer, text="Extracted aperture strip",
                  style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=4)

        # Plain frame that centres the strip figure; no scrollbar needed
        # (_draw_strip embeds the mpl widget directly here)
        self.strip_inner = ttk.Frame(strip_outer, style="TFrame")
        self.strip_inner.grid(row=1, column=0, sticky="ew")
        strip_outer.columnconfigure(0, weight=1)

        self.fig_strip    = None
        self.canvas_strip = None

        # ── Bottom: two spectrum plots side by side ───────────────────
        spec_frame = ttk.Frame(parent, style="TFrame")
        spec_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        spec_frame.columnconfigure(0, weight=1)

        self.fig_spec = Figure(figsize=(14, 3), facecolor=BG)
        self.ax_raw, self.ax_cal = self.fig_spec.subplots(
            1, 2, gridspec_kw={"wspace": 0.3})
        self.fig_spec.patch.set_facecolor(BG)
        for ax in (self.ax_raw, self.ax_cal):
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a4e")
        self.ax_raw.set_title("Raw spectrum (background subtracted)",
                              color="#a0a0c0", fontsize=8, pad=4)
        self.ax_cal.set_title("Calibrated & normalised spectrum",
                              color="#a0a0c0", fontsize=8, pad=4)

        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=spec_frame)
        self.canvas_spec.draw()
        self.canvas_spec.get_tk_widget().grid(row=0, column=0, sticky="ew")
        # No toolbar for spectra — zoom not needed

    def _build_log_pane(self, parent):
        """Build the bottom log pane spanning both columns."""
        parent.columnconfigure(0, weight=1)

        ttk.Label(parent, text="LOG", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))

        # Text widget + vertical scrollbar.  Background matches the panel
        # style; foreground is the regular fg colour.  Heights ~5 lines
        # keeps the pane unobtrusive but lets the user see recent history.
        self.log_text = tk.Text(
            parent, height=5,
            bg=self._palette["panel_2"], fg=self._palette["fg_2"],
            insertbackground=self._palette["fg_2"], relief="flat",
            font=("Consolas", 10), wrap="word",
            highlightthickness=0, borderwidth=0,
        )
        # severity tags
        self.log_text.tag_configure("info", foreground=self._palette["fg_2"])
        self.log_text.tag_configure("warn", foreground=self._palette["warn"])
        self.log_text.tag_configure("error", foreground=self._palette["err"])
        self.log_text.tag_configure("time", foreground=self._palette["fg_3"])

        # A windowed frozen build has no stderr, so logging.lastResort drops
        # every library warning on the floor — including ASTAP's exit code and
        # message, which is the only clue when a plate solve fails. Route the
        # root logger into this pane instead.
        logging.getLogger().addHandler(_TkLogHandler(self))

        self.log_text.grid(row=1, column=0, sticky="ew")
        sb = ttk.Scrollbar(parent, orient="vertical",
                           command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.grid(row=1, column=1, sticky="ns")
        # Read-only by default — toggle to "normal" when writing.
        self.log_text.configure(state="disabled")

        self._log("Ready.")

    def _log(self, message, level="info"):
        """
        Append a timestamped line to the log pane and keep it scrolled to
        the bottom.
        """
        from datetime import datetime
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"[{stamp}] ", "time")
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _label_axes(self):
        # Title on ax_main is set contextually by whoever drew it last
        # (raw frame vs. working image); only the spectrum panels need
        # static titles here.
        self.ax_raw.set_title("Raw spectrum (background subtracted)",
                              color="#a0a0c0", fontsize=8, pad=4)
        self.ax_cal.set_title("Calibrated & normalised spectrum",
                              color="#a0a0c0", fontsize=8, pad=4)

    def _strip_main_ticks(self):
        """
        Hide tick marks and tick labels on ax_main.

        The working image is read as a picture (click to switch source),
        not as a plot — numeric pixel coordinates are not interesting
        at a glance.  Position remains available via the navigation
        toolbar's hover-coordinate readout when needed.

        Called whenever ax_main is cleared via ax.cla() since cla()
        resets tick_params back to matplotlib defaults.
        """
        self.ax_main.tick_params(
            left=False, bottom=False, top=False, right=False,
            labelleft=False, labelbottom=False,
            labeltop=False, labelright=False,
        )

    def _draw_strip(self, strip_data):
        """
        Render the extraction zone at 1 px = 1 screen pixel.

        Parameters
        ----------
        strip_data : dict with keys
            "sky_lo"   : 2D ndarray or None
            "aperture" : 2D ndarray
            "sky_hi"   : 2D ndarray or None
        The three bands are stacked vertically (sky_hi on top, aperture in
        the middle, sky_lo at the bottom) with a 2 px gap between them.
        Sky bands are shown with a faint cyan tint border; the aperture with
        yellow.
        """
        DPI = 96
        GAP = 2   # px between bands

        sky_lo  = strip_data["sky_lo"]
        mask_lo = strip_data.get("mask_lo")
        ap      = strip_data["aperture"]
        sky_hi  = strip_data["sky_hi"]
        mask_hi = strip_data.get("mask_hi")
        contam_mask = strip_data.get("contam_mask")
        contam_science = strip_data.get("contam_science")
        spec_start_px = strip_data.get("spec_start_px")

        # Build ordered band list top→bottom: sky_hi, aperture, sky_lo
        # Each entry: (key, data, rejection_mask_or_None, colour, label)
        bands = []
        if sky_hi is not None and sky_hi.shape[0] > 0:
            bands.append(("sky_hi",  sky_hi,  mask_hi, "cyan",   "Sky (high)"))
        bands.append(    ("aperture", ap,      None,    "yellow", "Aperture"))
        if sky_lo is not None and sky_lo.shape[0] > 0:
            bands.append(("sky_lo",  sky_lo,  mask_lo, "cyan",   "Sky (low)"))

        # All bands share the same x slice so width is always ap.shape[1]
        strip_w = ap.shape[1]
        strip_h = sum(b[1].shape[0] for b in bands) + (len(bands) - 1) * GAP

        # Destroy the previous canvas; the Figure carries no pyplot manager,
        # so dropping the reference below is all the teardown it needs.
        if self.canvas_strip is not None:
            self.canvas_strip.get_tk_widget().destroy()

        self.fig_strip = Figure(
            figsize=(strip_w / DPI, strip_h / DPI),
            dpi=DPI, facecolor="#0f0f1a")

        # Compute the stretch limits from the aperture band — that's where
        # the signal of interest is, and basing the stretch on it (rather
        # than on whichever band happens to be drawn first by the shared
        # ImageNormalize) prevents the star core from saturating against
        # the sky bands' narrow dynamic range.  Use robust percentiles so
        # a single hot pixel or cosmic ray can't blow out the stretch.
        ap_finite = ap[np.isfinite(ap)]
        if ap_finite.size:
            vmin = float(np.percentile(ap_finite, 1.0))
            vmax = float(np.percentile(ap_finite, 99.5))
            if vmax <= vmin:    # degenerate (uniform aperture) — widen a bit
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        norm = ImageNormalize(vmin=vmin, vmax=vmax,
                              stretch=PowerStretch(a=0.15))

        # Place axes top-down: y_cursor tracks the bottom of the current band
        # in figure pixels, starting from the top (strip_h) and going down.
        y_cursor = strip_h
        ap_ax = None   # captured during the loop for the contam overlay below
        for key, data, rej_mask, colour, label in bands:
            rh, rw = data.shape
            y_cursor -= rh
            ax = self.fig_strip.add_axes((
                0,             y_cursor / strip_h,
                1.0,           rh / strip_h,
            ))
            ax.imshow(data, cmap="Greys", origin="lower",
                      norm=norm, interpolation="nearest",
                      extent=[0, rw, 0, rh], aspect="auto")
            # Disable matplotlib's hover value readout on this image:
            # the strip bands can contain NaN (edge / masked pixels) and
            # the toolbar's NaN formatter raises (see _suppress_cursor_data).
            _suppress_cursor_data(ax.images[-1])
            ax.set_xlim(0, rw)
            ax.set_ylim(0, rh)
            ax.set_facecolor("#0f0f1a")
            # Tint sky bands with a faint cyan background
            if key != "aperture":
                ax.set_facecolor("#001a1a")
                ax.fill_betweenx([0, rh], 0, rw,
                                 color="cyan", alpha=0.08, zorder=0)
            # Overlay rejection mask in red (origin="lower" so row 0 is bottom)
            if rej_mask is not None and rej_mask.any():
                red_overlay = np.zeros((*rej_mask.shape, 4), dtype=float)
                red_overlay[rej_mask] = [1.0, 0.0, 0.0, 0.5]
                ax.imshow(red_overlay, origin="lower",
                          extent=[0, rw, 0, rh], aspect="auto",
                          interpolation="nearest", zorder=3)
            # Spectrum-start marker: a vertical line at the region column
            # where the science range (sp_min) begins.  Drawn on every band
            # at the same x so it reads as a single full-height divider —
            # everything to its left is the zero-order / buffer zone, not
            # part of the extracted spectrum.
            if (spec_start_px is not None
                    and np.isfinite(spec_start_px)
                    and 0 < spec_start_px < rw):
                ax.axvline(x=spec_start_px, color="#111111",
                           linestyle="--", linewidth=0.8, alpha=0.85,
                           zorder=5)
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(1.2)
            ax.text(0.005, 0.5, label, transform=ax.transAxes,
                    color=colour, fontsize=5, va="center", alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            if key == "aperture":
                ap_ax = ax
            y_cursor -= GAP

        # Contamination overlay on the aperture band — full-height red
        # bars at each masked column.  Drawn after the loop so it sits on
        # top of the aperture imshow; same red treatment as the sky-band
        # rejection overlay (the geometry — aperture vs sky band — tells
        # the user which kind of rejection they're seeing).
        if contam_mask is not None and contam_mask.any() and ap_ax is not None:
            ap_h, ap_w = ap.shape
            mask2d = np.broadcast_to(contam_mask, (ap_h, ap_w))
            red_overlay = np.zeros((ap_h, ap_w, 4), dtype=float)
            # Two strengths, matching the log: full red where the column
            # reaches the extracted spectrum, faint where it is truncated
            # away with the zero-order zone.  The faint bars stay drawn —
            # they explain the count, and a mask sitting just outside the
            # window is worth seeing when sp_min is about to move.
            red_overlay[mask2d] = [1.0, 0.0, 0.0, 0.15]
            science2d = (np.broadcast_to(contam_science, (ap_h, ap_w))
                         if contam_science is not None else mask2d)
            red_overlay[science2d] = [1.0, 0.0, 0.0, 0.5]
            ap_ax.imshow(red_overlay, origin="lower",
                         extent=[0, ap_w, 0, ap_h], aspect="auto",
                         interpolation="nearest", zorder=4)

        # Embed in strip_inner frame
        for child in self.strip_inner.winfo_children():
            child.destroy()

        # Rebuilt per extraction — must not steal focus (see helper).
        self.canvas_strip = _canvas_without_focus_steal(
            self.fig_strip, self.strip_inner)
        widget = self.canvas_strip.get_tk_widget()
        widget.configure(width=strip_w, height=strip_h)
        widget.pack(anchor="center")
        self.canvas_strip.draw()

    # ------------------------------------------------------------------
    # Parameter reading
    # ------------------------------------------------------------------

    def _params(self):
        """Read and validate all control-panel values; return a dict, or
        None after an error dialog that names the offending field (§3.5 —
        a bare "could not convert string to float: 'abc'" told the user
        nothing about where to look)."""
        p = dict(
            target   = self.v_target.get().strip(),
            cal_file = self.v_response_file.get().strip(),
        )
        # (UI label, dict key, cast, variable) — labels match the left panel.
        fields = (
            ("Å / px",           "dispersion",    float, self.v_dispersion),
            ("Rotation angle °", "angle",         float, self.v_angle),
            ("N sources",        "nsources",      int,   self.v_nsources),
            ("FWHM (px)",        "fwhm",          float, self.v_fwhm),
            ("Aperture half-h",  "aperture_half", int,   self.v_aper_half),
            ("Bkg gap low",      "sky_gap",       int,   self.v_sky_gap),
            ("Bkg width low",    "sky_width",     int,   self.v_sky_width),
            ("Bkg gap high",     "sky_gap_hi",    int,   self.v_sky_gap_hi),
            ("Bkg width high",   "sky_width_hi",  int,   self.v_sky_width_hi),
            ("λ min (Å)",        "sp_min",        float, self.v_sp_min),
            ("λ max (Å)",        "sp_max",        float, self.v_sp_max),
        )
        for label, key, cast, var in fields:
            raw = var.get().strip()
            try:
                p[key] = cast(raw)
            except ValueError:
                kind = "a whole number" if cast is int else "a number"
                messagebox.showerror(
                    "Parameter error", f"{label}: '{raw}' is not {kind}.")
                return None

        # Physically meaningless values — catch the common typos with a
        # named message instead of a downstream crash or silent nonsense.
        checks = (
            (p["dispersion"] <= 0, "Å / px must be > 0."),
            (p["nsources"] < 1,    "N sources must be ≥ 1."),
            (p["fwhm"] <= 0,       "FWHM (px) must be > 0."),
            (p["aperture_half"] < 1, "Aperture half-h must be ≥ 1."),
            (p["sky_gap"] < 0,     "Bkg gap low must be ≥ 0."),
            (p["sky_width"] < 1,   "Bkg width low must be ≥ 1."),
            (p["sky_gap_hi"] < 0,  "Bkg gap high must be ≥ 0."),
            (p["sky_width_hi"] < 1, "Bkg width high must be ≥ 1."),
            (p["sp_min"] >= p["sp_max"],
             "λ min (Å) must be smaller than λ max (Å)."),
        )
        for bad, msg in checks:
            if bad:
                messagebox.showerror("Parameter error", msg)
                return None

        p["spectrum_width"] = compute_spectrum_width(p["dispersion"], p["sp_max"])
        return p

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _run(self):
        p = self._params()
        if p is None:
            return

        # Livestack runs on the in-memory virtual stack — saying "Loading
        # FITS" there made readers think the file was re-read every frame.
        self._log("Restacking…" if self._frame_override is not None
                  else "Loading FITS…")
        self.update_idletasks()

        try:
            self._load_and_run(p)
        except FileNotFoundError as e:
            messagebox.showerror("File not found", str(e))
            self._log("Error — see dialog.", level="error")
        except Exception as e:
            messagebox.showerror("Runtime error", str(e))
            self._log("Error — see dialog.", level="error")

    def _load_and_run(self, p):
        # True iff no source was previously extracted — gates the
        # initial-Update auto-fit at the end of this method.
        was_fresh = self._last_source_xy is None
        # ── Load FITS ─────────────────────────────────────────────────
        # read_fits_image owns HDU selection (handles empty primary HDUs in
        # .fits.fz / stacker outputs) and native-order byteswap.  Retain the
        # header for the WCS plate-solve (position / scale hints).
        # Live-stack override: when set, run on the in-memory virtual stack
        # rather than re-reading the target file.  _target_header was set
        # from the reference frame when the stack started.
        if self._frame_override is not None:
            data = self._frame_override
        else:
            raw, self._target_header = read_fits_image(p["target"])
            data = to_mono(raw)
        # Stats on a 4× subsample: median/σ of a megapixel frame taken from
        # 1/16th of its pixels is statistically indistinguishable, and the
        # full-frame sigma-clip costs ~0.5 s of every pipeline run.
        _, median, std = sigma_clipped_stats(data[::4, ::4], sigma=3.0)

        # ── Rotate ────────────────────────────────────────────────────
        rotated = rotate(data, p["angle"], reshape=False, cval=median)
        self.rotated_data = rotated
        h_img, w_img = rotated.shape

        # ── Detect sources ────────────────────────────────────────────
        # Livestack: the reference frame's detection defines the source
        # list (and its numbering) for the whole stack.  Accumulating
        # frames saturate / flat-top bright stars and reshuffle
        # DAOStarFinder's peak ordering, renumbering apertures mid-
        # session; every frame is registered onto the reference, so its
        # centroids stay valid.  Re-detect only when a detection-relevant
        # parameter changed (same staleness pattern as _solved_angle).
        _detect_keys = ("angle", "fwhm", "nsources", "spectrum_width")
        if (self._frame_override is not None
                and self.top_sources is not None
                and self._last_p is not None
                and all(self._last_p[k] == p[k] for k in _detect_keys)):
            detection_ran = False
            top_sources = self.top_sources
            all_sources_xy = self.all_sources_xy
        else:
            detection_ran = True
            self._log("Detecting sources…"); self.update_idletasks()
            daofind = DAOStarFinder(fwhm=p["fwhm"], threshold=5.0 * std)
            sources = daofind(rotated - median)
            if sources is None or len(sources) == 0:
                messagebox.showwarning("No sources", "DAOStarFinder detected no sources.")
                self._log("Error — see dialog.", level="error")
                return
            sources.sort("peak", reverse=True)
            # Every source, before the frame-fit filter and the nsources
            # truncation below: contamination does not care whether a star's
            # own spectrum fits in the frame or whether it made the user's
            # top-N list — only whether its PSF reaches the aperture.  The
            # detection is re-run whenever nsources changes, which is how a
            # deepening livestack picks up sources that were under the
            # threshold on earlier frames.
            all_sources_xy = np.array([_dao_xy(s) for s in sources],
                                      dtype=float)
            # Reject sources whose spectrum, after rotation, would extend
            # into the cval-padded corner triangles.  The naive check
            # `xc + L <= w_img` only catches off-array sources; the geometric
            # check below also rejects sources whose tail falls into the
            # synthetic flat-data corners.
            valid = [s for s in sources
                     if spectrum_fully_in_frame(
                    *_dao_xy(s),
                    p["spectrum_width"], p["angle"], w_img, h_img)]
            top_sources = valid[: p["nsources"]]
            if not top_sources:
                messagebox.showwarning("No valid sources",
                                       "All detected sources are too close to the "
                                       "right edge for a full spectrum.")
                self._log("Error — see dialog.", level="warn")
                return
        # Capture the previous source list (the positions the current
        # matches are aligned to) BEFORE reassigning, so the remap below can
        # map old→new by centroid position.
        prev_sources = self.top_sources
        self.top_sources = top_sources
        self.all_sources_xy = all_sources_xy
        # WCS identifications are tied to source pixel positions, which only
        # move when the rotation angle changes.  If the angle is unchanged
        # since the solve, the freshly-detected centroids are the same stars
        # at (essentially) the same pixels, so remap the existing matches to
        # the new source order by nearest centroid.  If the angle changed
        # (or there was no solve), the matches no longer apply — drop them.
        if (self.source_matches is not None
                and self._solved_angle is not None
                and prev_sources is not None
                and abs(p["angle"] - self._solved_angle) < 1e-6):
            self.source_matches = self._remap_matches_to_sources(
                prev_sources, self.source_matches, top_sources)
        else:
            self.source_matches = None
            self._solved_angle = None
            self._main_wcs = None

        # ── Clear spectrum axes ───────────────────────────────────────
        for ax in (self.ax_raw, self.ax_cal):
            ax.cla()
            ax.set_facecolor("#0f0f1a")
        self._label_axes()

        # ── Plot working (rotated mono) image on ax_main ──────────────
        # Replaces the raw frame shown on ax_main by _show_original at
        # file-load time.
        self._log("Plotting…"); self.update_idletasks()
        # preserve_view: an Update, a parameter nudge, or a livestack frame
        # landing re-runs on the same-shape frame — keep the user's pan/zoom
        # instead of snapping back to full extent.
        # No title: the caption costs a strip of vertical space for a hint
        # that is only useful once.
        self._show_image(rotated, title="", preserve_view=True)
        # NOTE: the validity-boundary indicator line is intentionally not
        # drawn.  A single right-edge line (validity_boundary_line in
        # spectrum_core) only represents the true accept/reject boundary at
        # small rotation angles; at large angles the top/bottom edges of the
        # rotated frame dominate and the line diverges from — and misleads
        # about — the actual decision.  The source-acceptance logic itself
        # (spectrum_fully_in_frame) is correct at all angles, so rejected
        # sources simply don't appear as selectable, which is self-evident.
        # validity_boundary_line is kept in spectrum_core for debugging.

        # Plot numbered apertures for all valid detected sources.  Source
        # labels (catalog names from a WCS solve, if any) are applied by
        # _redraw_source_labels, which is also called standalone after
        # Solve to WCS so names can appear without re-running detection.
        xs = [_dao_xy(s)[0] for s in top_sources]
        ys = [_dao_xy(s)[1] for s in top_sources]
        self._redraw_source_labels(draw=False)

        # Extraction boxes get added by _display_extraction →
        # _draw_overlay_boxes below, which issues its own draw_idle.

        # ── Extract: re-snap to previously selected source if possible ─
        self._last_p = p
        sel_x, sel_y = xs[0], ys[0]  # default: brightest detected centroid
        if self._last_select_free and self._last_source_xy is not None:
            # A free-selected point must survive a Run (angle nudge,
            # dispersion change, etc.) without snapping to a detected
            # source.  Keep the exact anchor.
            sel_x, sel_y = self._last_source_xy
        elif self._last_source_xy is not None:
            lx, ly = self._last_source_xy
            dists = [np.hypot(sx - lx, sy - ly) for sx, sy in zip(xs, ys)]
            nearest_dist = min(dists)
            if nearest_dist <= self.SOURCE_SNAP_RADIUS:
                idx = dists.index(nearest_dist)
                sel_x, sel_y = xs[idx], ys[idx]
        # Store raw centroid — offset is applied via _applied_y() at
        # every extraction call site, so it remains independently adjustable
        # across source switches, rotation changes, and node edits.
        self._last_source_xy = (sel_x, sel_y)

        # On the very first Update after a file load, auto-fit aperture
        # and sky_gap from the source's spatial FWHM.  Subsequent Updates
        # on the same target leave the user's tuned values alone.
        # FWHM is measured on the raw centroid — it is a property of the
        # PSF, not of the user's strip offset.
        if was_fresh:
            fwhm = estimate_source_fwhm(self.rotated_data, sel_x, sel_y)
            if np.isfinite(fwhm):
                self._target_fwhm = fwhm
                new_aper, new_gap, capped = self._autofit_geometry(fwhm)
                self.v_aper_half.set(str(new_aper))
                self.v_sky_gap.set(str(new_gap))
                self.v_sky_gap_hi.set(str(new_gap))  # auto-fit keeps both bands symmetric
                p = self._params()
                if p is None:
                    return
                # Re-stash: _last_p must track the post-auto-fit geometry or
                # every later nudge/toggle re-extracts with the pre-fit values.
                self._last_p = p
                self._log("Initial source:"
                          + self._autofit_msg(fwhm, new_aper, new_gap, capped))
            else:
                self._log(
                    "Initial source: FWHM fit failed; keeping "
                    f"aperture ±{self.v_aper_half.get()}, "
                    f"sky gap {self.v_sky_gap.get()}.")

        self._display_extraction(sel_x, self._applied_y(sel_y), p)
        self._autocentre_if_enabled()

        # ── Wire main-image click handler ────────────────────────────
        if self._main_click_cid is not None:
            self.canvas_main.mpl_disconnect(self._main_click_cid)
        self._main_click_cid = self.canvas_main.mpl_connect(
            "button_press_event", self._on_main_click)

        self._redraw_nodes()
        self.canvas_spec.draw()
        # Detection just reset source_matches to None; reflect that in the
        # info box (prompts the user to solve).  A prior solve's labels are
        # gone because the source set was recomputed.
        self._update_source_info()
        # "detected" vs "kept": the cached branch reuses the reference
        # frame's frozen detection, and saying "detected" there read as a
        # per-frame re-detection during livestack (it isn't — see the
        # _detect_keys staleness guard above).
        n_det = len(top_sources)
        verb = "detected" if detection_ran else "kept (detection frozen)"
        self._log(
            f"Done.  {n_det} valid source(s) {verb}.  "
            f"Click a numbered aperture to switch source.")


    # ------------------------------------------------------------------
    # Overlay box drawing — full redraw on ax_main
    # ------------------------------------------------------------------

    def _draw_overlay_boxes(self, bbox):
        """
        Draw aperture and sky-band rectangles on ax_main.

        Uses a full canvas redraw — simpler and more reliable than a
        blit-based partial redraw.  Called on every source switch,
        parameter change, and toggle, all of which are user-initiated, so
        the redraw cost (tens of ms) is invisible.
        """
        # Remove only the tracked rectangles, NOT every patch on ax_main:
        # the source-marker aperture circles from _redraw_source_labels are
        # patches too, and a blanket sweep makes them vanish on every
        # extraction while surviving a Solve-to-WCS redraw, so the circles
        # appear intermittently.
        for art in self._box_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._box_artists = []

        # Aperture rectangle
        rect = Rectangle(
            (bbox["x_start"], bbox["y_start"]),
            bbox["x_end"] - bbox["x_start"],
            bbox["y_end"] - bbox["y_start"],
            lw=1.2, edgecolor="yellow", facecolor="none",
        )
        self.ax_main.add_patch(rect)
        self._box_artists.append(rect)
        # Sky bands (dashed cyan)
        for y0, y1 in [(bbox["sky_lo_start"], bbox["sky_lo_end"]),
                       (bbox["sky_hi_start"], bbox["sky_hi_end"])]:
            band = Rectangle(
                (bbox["x_start"], y0),
                bbox["x_end"] - bbox["x_start"], y1 - y0,
                lw=0.8, edgecolor="cyan", facecolor="none", linestyle="--",
            )
            self.ax_main.add_patch(band)
            self._box_artists.append(band)

        self.canvas_main.draw_idle()

        # ------------------------------------------------------------------
    # Extraction display — single source
    # ------------------------------------------------------------------
    def compute_calibrated_spectrum(self, rotated, sx, sy, p, poly,
                                    contam_mask=None, sky_col_flag=None):
        """
        Extract and calibrate a single spectrum to (wavelengths, norm_flux).

        Pure-ish helper consumed by the sequence generator.  The
        interactive path (_display_extraction) runs its own inline chain
        because it additionally needs the extraction intermediates
        (region, sky bands, bbox) for display and propagates σ — keep the
        two in step when touching either.  Does NOT touch any axes,
        overlays, caches, or the per-source anchor residual — it takes a
        dispersion polynomial as an explicit argument (the caller decides
        whether that is the live anchor-corrected poly or a fixed one).

        ``contam_mask`` (optional) is a per-region-column boolean mask of
        contaminated columns, computed once on the high-SNR working image by
        the main analysis and reused verbatim here.  Per-frame re-detection on
        the low-SNR sequence captures would yield a different, noisy mask each
        frame, injecting artificial frame-to-frame variation; with good
        tracking the column positions are stable, so a single authoritative
        mask is better.  Length is reconciled to this frame's strip by
        truncate/pad (the ±1 px strip-origin jitter between tracked frames can
        change the column count by one): a longer mask is truncated, a shorter
        one is padded with False.

        ``sky_col_flag`` (optional) is the companion per-column flag for the
        *sky band*: where the high-SNR reference found the sky untrustworthy,
        ``extract_spectrum`` interpolates the per-frame background from clean
        neighbour columns instead of re-deciding cleanliness on the low-SNR
        frame.  The frame still sets its own background level; only the
        cleanliness decision is inherited.

        Returns (wls, norm_flux); wls is empty when the extracted range does
        not overlap [sp_min, sp_max] or the calibration file is missing.
        """
        empty = (np.array([]), np.array([]))
        _, _, _, _, _, col_sums, _, _ = extract_spectrum(
            sx, sy, rotated,
            p["spectrum_width"], p["aperture_half"],
            p["sky_gap"], p["sky_width"],
            sky_col_flag=sky_col_flag,
            sky_gap_hi=p.get("sky_gap_hi"), sky_width_hi=p.get("sky_width_hi"),
        )

        # Apply the stashed contaminator mask, reconciled to this frame's
        # strip length by truncate/pad.
        if contam_mask is not None and contam_mask.any():
            n = len(col_sums)
            if len(contam_mask) >= n:
                m = contam_mask[:n]
            else:
                m = np.zeros(n, dtype=bool)
                m[:len(contam_mask)] = contam_mask
            if m.any():
                col_sums = col_sums.astype(float).copy()
                col_sums[m] = np.nan

        try:
            if self._response_df_cache is not None:
                cal_df = self._response_df_cache
            else:
                cal_df = load_calibration_file(p["cal_file"])
        except FileNotFoundError:
            return empty
        except ValueError as e:      # malformed/unparseable table
            self._log(f"Calibration file unusable: {e}", level="warn")
            return empty

        poly = self._validate_dispersion_poly(poly, len(col_sums))
        all_pixels = np.arange(len(col_sums))
        all_wls = pixels_to_wavelengths(
            all_pixels, p["dispersion"], poly_coeffs=poly)
        mask = (all_wls >= p["sp_min"]) & (all_wls <= p["sp_max"])
        wls = all_wls[mask]
        if len(wls) == 0:
            return empty
        trunc = col_sums[mask]

        cal_int = apply_calibration(wls, trunc, cal_df)
        norm_int = normalize_flux(cal_int)
        return wls, norm_int

    def _display_extraction(self, sx, sy, p):
        """
        Extract and display the spectrum for a single source at (sx, sy).
        Clears ax_raw, ax_cal and all previous extraction boxes on ax_main,
        then redraws the strip and both spectrum panels.
        """
        # Clear spectrum panels
        self.ax_raw.cla()
        self.ax_raw.set_facecolor("#0f0f1a")
        self.ax_cal.cla()
        self.ax_cal.set_facecolor("#0f0f1a")
        self._label_axes()

        region, sky_lo, sky_hi, mask_lo, mask_hi, col_sums, col_var, bbox = \
            extract_spectrum(
                sx, sy, self.rotated_data,
                p["spectrum_width"], p["aperture_half"],
                p["sky_gap"], p["sky_width"],
                sky_gap_hi=p.get("sky_gap_hi"),
                sky_width_hi=p.get("sky_width_hi"),
            )
        # ── Per-column sky flag for the sequence generator ─────────────
        # Collapse the 2-D sky-band rejection masks to a per-column "this
        # column's sky band is untrustworthy" flag, frozen here on the
        # high-SNR working image and reused by the sequence (low-SNR frames
        # can't re-decide sky cleanliness reliably).  A column is flagged when
        # most of its sky pixels were rejected by the sigma-clip.
        self._last_sky_col_flag = build_sky_col_flag(mask_lo, mask_hi)

        # ── In-aperture contaminator detection ────────────────────────
        # Detect point sources sitting inside the aperture (other stars
        # whose direct image overlaps the target's spectrum row).  Always
        # detected so the user can see them in the strip overlay; column
        # masking of col_sums is gated on the v_contam_mask toggle.
        fwhm_for_mask = (self._target_fwhm
                         if self._target_fwhm is not None
                         and np.isfinite(self._target_fwhm)
                         else 4.0)

        # ── Per-source zero-order anchor residual ──────────────────────
        # Measure the sub-pixel zero-order peak on the full rotated image
        # (NOT the truncated strip) and store its offset from the strip
        # origin sx.  get_dispersion_poly turns the difference between this
        # and the calibration star's residual into a wavelength-scale
        # correction.  NaN (no clear peak) → no correction for this source.
        zo_x = measure_zero_order_x(self.rotated_data, sx, sy,
                                    fwhm=fwhm_for_mask)
        # Residual is measured against the strip ORIGIN (bbox["x_start"],
        # the floored centroid that extract_spectrum uses to anchor the
        # pixel axis), NOT the float centroid sx.  The dispersion nodes
        # live in strip-pixel coordinates, where the zero-order peak sits
        # at zo_x - x_start.  Using sx instead would leave a residual
        # frac(sx) term that differs between sources by up to ±1 px
        # (≈ ±7.6 Å), corrupting the very colour-dependent offset this
        # correction exists to cancel.
        self._current_anchor_resid = (zo_x - bbox["x_start"]
                                      if np.isfinite(zo_x) else None)

        # Match the strip against the frame-wide source list rather than
        # re-detecting inside it: DAOStarFinder needs a whole PSF, so a star
        # half inside the band is not detectable there — and that is
        # precisely the star that spills flux into the aperture unnoticed.
        # No zero-order slice is needed: the target's own blob sits on the
        # trace and the on-trace test drops it.  The diagnostic toggle
        # disables that test so emission peaks reappear as contaminants.
        excl = 0.0 if self.v_contam_legacy.get() else TRACE_EXCLUDE_FWHM
        contam_cols, on_trace = contaminators_from_sources(
            self.all_sources_xy if self.all_sources_xy is not None
            else np.empty((0, 2)),
            bbox, fwhm_for_mask, sy, trace_exclude_fwhm=excl)

        # ── Which of those can reach the extracted spectrum ────────────
        # The strip runs from the zero order outwards, so it always holds
        # sources the extraction then truncates away — the target's own
        # zero order at col 0 first among them.  Only a source inside
        # [sp_min, sp_max] can put a bump in the science spectrum, and only
        # that deserves a red line; the rest are stated in plain text so a
        # warning always means something to act on.  The polynomial is
        # recomputed here rather than shared with the calibrated panel
        # below: that copy lives inside a try that can bail out before it,
        # and a polyval over the strip is not worth restructuring for.
        poly_c = self._validate_dispersion_poly(
            self.get_dispersion_poly(), len(col_sums))
        col_wls = pixels_to_wavelengths(
            np.arange(len(col_sums)), p["dispersion"], poly_coeffs=poly_c)
        in_window = (col_wls >= p["sp_min"]) & (col_wls <= p["sp_max"])

        def _in_window(cols):
            """Science-window flag for each strip column in ``cols``."""
            cols = np.atleast_1d(np.asarray(cols, dtype=float))
            if not len(in_window):
                return np.zeros(len(cols), dtype=bool)
            return in_window[np.clip(np.rint(cols).astype(int),
                                     0, len(in_window) - 1)]

        # Say what was passed over and why.  On-trace is the one call this
        # cannot make from geometry — the target's emission line and a star
        # sitting on the trace are the same picture — so it must not be a
        # silent decision: a real contaminant left unmasked shows up as a
        # bump the user would otherwise take for a spectral feature.
        if len(on_trace):
            hit = _in_window(on_trace[:, 0])
            near = (f"within {excl:.2f} × FWHM = "
                    f"{excl * fwhm_for_mask:.1f} px of the trace")
            if hit.any():
                where = ", ".join(f"col {int(round(c))} ({d:+.1f} px)"
                                  for c, d in on_trace[hit])
                self._log(
                    f"Contaminators: {int(hit.sum())} source(s) in the "
                    f"science window ignored as the target's own light "
                    f"({near}): {where}.", level="warn")
            if (~hit).any():
                where = ", ".join(f"col {int(round(c))} ({d:+.1f} px)"
                                  for c, d in on_trace[~hit])
                self._log(
                    f"Contaminators: {int((~hit).sum())} source(s) ignored as "
                    f"the target's own light ({near}), outside the science "
                    f"window — zero order and buffer: {where}.")

        # Build a per-column boolean mask of contaminated regions
        contam_mask = np.zeros(col_sums.shape, dtype=bool)
        if len(contam_cols):
            mask_half = max(2, int(round(CONTAM_MASK_FWHM_MULT * fwhm_for_mask)))
            for cx in contam_cols:
                c0 = max(0, int(round(cx)) - mask_half)
                c1 = min(len(col_sums), int(round(cx)) + mask_half + 1)
                contam_mask[c0:c1] = True

        # Apply masking to col_sums if the toggle is on.  Always cast to
        # float so a downstream NaN assignment works whatever the input
        # dtype is; cheap if already float.
        # Stash the applied contaminator mask for the sequence generator.
        # Only set when the toggle is on AND columns were actually masked —
        # i.e. when this mask was genuinely applied to the science spectrum;
        # otherwise cleared, so a stale mask can't outlive the toggle or a
        # source switch.  The sequence reuses THIS mask (computed on the
        # high-SNR working image) rather than re-detecting per low-SNR
        # frame, where the detection would be noisy and vary
        # frame-to-frame.  Excellent tracking keeps the column positions
        # stable, so one authoritative mask applied uniformly is both
        # cleaner and more honest.
        if self.v_contam_mask.get() and contam_mask.any():
            # Always cast to float so a downstream NaN assignment works
            # whatever the input dtype is; cheap if already float.
            col_sums = col_sums.astype(float).copy()
            # Pre-mask counts kept for the DB: masked data is stored
            # flagged, never dropped (spectra_db guidelines §4).
            self._unmasked_col_sums = col_sums.copy()
            col_sums[contam_mask] = np.nan
            self._last_contam_mask = contam_mask.copy()
        else:
            self._unmasked_col_sums = None
            self._last_contam_mask = None

        if len(contam_cols):
            hit = _in_window(contam_cols)
            mask_state = "masked" if self.v_contam_mask.get() else "shown but not masked"
            if hit.any():
                self._log(
                    f"Detected {int(hit.sum())} in-aperture contaminator(s) "
                    f"in the science window ({len(contam_cols)} in the "
                    f"strip); {mask_state}.", level="warn")
            else:
                self._log(
                    f"Detected {len(contam_cols)} in-aperture "
                    f"contaminator(s), none in the science window; "
                    f"{mask_state}.")

        # Draw extraction boxes via partial ax_main redraw
        self._draw_overlay_boxes(bbox)

        # Raw spectrum — draw directly on ax_raw (no inset needed).
        # Shared with the reference-line toggle path via _draw_raw_panel.
        self._draw_raw_panel(p, col_sums)

        # Region-pixel column where the science spectrum begins (sp_min).
        # Linear dispersion is used here for a simple visual guide, matching
        # the zero-order exclusion above; with a non-linear fit the true
        # column differs by at most a pixel or two at this end.
        spec_start_px = (p["sp_min"] / p["dispersion"]
                         if p["dispersion"] else None)

        self._draw_strip({
            "sky_lo":      sky_lo   if sky_lo.shape[0]  > 0 else None,
            "mask_lo":     mask_lo  if sky_lo.shape[0]  > 0 else None,
            "aperture":    region,
            "sky_hi":      sky_hi   if sky_hi.shape[0]  > 0 else None,
            "mask_hi":     mask_hi  if sky_hi.shape[0]  > 0 else None,
            "contam_mask": contam_mask if contam_mask.any() else None,
            "contam_science": (contam_mask & in_window
                               if contam_mask.any() else None),
            "spec_start_px": spec_start_px,
        })

        # Calibrated spectrum
        self.column_sums = col_sums
        try:
            # Prefer the in-memory calibration cache if one is present
            # (set by _load_config from an embedded JSON array).  This
            # decouples the running pipeline from the on-disk .dat file,
            # so a config remains usable even after the original file is
            # moved or deleted.  The cache is invalidated automatically
            # whenever the cal-file path field is edited or browsed.
            if self._response_df_cache is not None:
                cal_df = self._response_df_cache
            else:
                cal_df = load_calibration_file(p["cal_file"])

            # Map every pixel index to a wavelength: polynomial when there
            # are >= 2 calibration nodes and the fit is monotonic, otherwise
            # fall back to linear dispersion.
            poly = self._validate_dispersion_poly(
                self.get_dispersion_poly(), len(col_sums))
            all_pixels = np.arange(len(col_sums))
            all_wls = pixels_to_wavelengths(
                all_pixels, p["dispersion"], poly_coeffs=poly)

            mask = (all_wls >= p["sp_min"]) & (all_wls <= p["sp_max"])
            wls = all_wls[mask]
            trunc = col_sums[mask]
            trunc_var = col_var[mask]

            # Empty intersection: the extracted strip's wavelength
            # coverage doesn't overlap [sp_min, sp_max].  Common causes:
            # the source is too close to the right edge (strip too
            # short), sp_min/sp_max are inconsistent with the current
            # dispersion, or a non-linear polynomial fit pushes the
            # mapped range entirely outside the display window.  Bail
            # out cleanly with a diagnostic so the user can see what
            # to adjust; the strip and main image remain drawn from
            # earlier in this method.
            if len(wls) == 0:
                pixel_lo = float(all_wls[0])  if len(all_wls) else float("nan")
                pixel_hi = float(all_wls[-1]) if len(all_wls) else float("nan")
                self._log(
                    f"Calibrated panel: extracted range "
                    f"{pixel_lo:.0f}–{pixel_hi:.0f} Å does not overlap "
                    f"display window {p['sp_min']:.0f}–{p['sp_max']:.0f} Å; "
                    f"check dispersion, sp_min/sp_max, or source position.",
                    level="warn")
                self._calibrated_wls = wls
                self._calibrated_flux = np.full_like(wls, np.nan)  # explicit empty flux, not aliased to wls
                self._calibrated_sigma = np.full_like(wls, np.nan)
                self._calibrated_pixels = all_pixels[mask]
                return

            # Warn once per state if the requested display range overruns
            # the calibration table — those wavelengths become NaN gaps.
            # Latched on (table range, display window) so an angle nudge
            # doesn't re-fire it; cleared when the overrun goes away.
            cal_lo = float(cal_df["wavelength"].min())
            cal_hi = float(cal_df["wavelength"].max())
            if wls[0] < cal_lo or wls[-1] > cal_hi:
                cov_key = (cal_lo, cal_hi, p["sp_min"], p["sp_max"])
                if cov_key != self._warned_cal_coverage:
                    self._warned_cal_coverage = cov_key
                    self._log(
                        f"Calibration table covers {cal_lo:.0f}–{cal_hi:.0f} Å; "
                        f"requested {wls[0]:.0f}–{wls[-1]:.0f} Å will be gapped "
                        f"outside that range.")
            else:
                self._warned_cal_coverage = None

            cal_int = apply_calibration(wls, trunc, cal_df)
            norm_int, norm_scale = normalize_flux(cal_int, return_scale=True)

            # Propagate the background-term 1σ through the same two
            # transforms as the flux: divide by the response factor
            # (apply_calibration_to_sigma), then scale by the normalisation
            # factor (the affine offset does not affect σ).  The band is
            # therefore expressed in the same normalised units as the
            # plotted flux.  Source Poisson term still omitted (no gain yet).
            sigma_raw = np.sqrt(trunc_var)
            cal_sigma = apply_calibration_to_sigma(wls, sigma_raw, cal_df)
            norm_sigma = cal_sigma * norm_scale

            # Cache for lightweight toggle redraws
            self._calibrated_wls      = wls
            self._calibrated_flux = norm_int
            self._calibrated_sigma = norm_sigma
            self._calibrated_pixels = all_pixels[mask]

            # Calibrated panel — shared with the toggle path via
            # _draw_cal_panel.  Node markers are redrawn elsewhere on this
            # path, so the helper deliberately does not touch them.
            self._draw_cal_panel(p, wls, norm_int, poly)
        except (FileNotFoundError, ValueError) as e:
            # Clear the toggle-redraw caches (mirrors the empty-overlap
            # path above): column_sums already holds the new source, so
            # leaving the previous source's calibrated arrays here would
            # feed a mismatched spectrum to the full-spectrum viewer and
            # the continuum dialog.  ValueError is a malformed/rejected
            # calibration table (load_calibration_file validates);
            # whatever raised, log the full message so nothing in this
            # block can fail silently.
            self._calibrated_wls = None
            self._calibrated_flux = None
            self._calibrated_sigma = None
            self._calibrated_pixels = None
            if isinstance(e, FileNotFoundError):
                msg = "Calibration file not found"
            else:
                msg = "Calibration file unusable (see log)"
                self._log(f"Calibration failed: {e}", level="warn")
            self.ax_cal.text(0.5, 0.5, msg,
                             ha="center", va="center",
                             color="#e94560", fontsize=9,
                             transform=self.ax_cal.transAxes)

    def _on_main_click(self, event):
        """
        Left-click snaps to the nearest detected (valid) source within
        SNAP_RADIUS pixels and extracts its spectrum.
        Clicks that land far from any aperture are silently ignored.
        """

        if event.button != 1:
            return
        if event.inaxes is not self.ax_main:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.rotated_data is None or self._last_p is None:
            return
        if not self.top_sources:
            return

        # ax_main is fully redrawn on every change, so its transform is
        # current and event.xdata/ydata can be used directly.
        cx, cy = event.xdata, event.ydata
        p = self._last_p

        # ── Free selection ───────────────────────────────────────────
        # When the toggle is on, extract from the exact click point
        # rather than snapping to a detected source.  Reuses the same
        # FWHM auto-fit and extraction path as a normal source switch.
        if self.v_free_select.get():
            self._extract_free_point(cx, cy, p)
            return

        # Find nearest valid source
        best_src = None
        best_dist = float("inf")
        for src in self.top_sources:
            sx_src, sy_src = _dao_xy(src)
            dx = sx_src - cx
            dy = sy_src - cy
            dist = np.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_src = src

        if best_dist > self.SOURCE_SNAP_RADIUS:
            self._log(
                f"No source within {self.SOURCE_SNAP_RADIUS} px — "
                f"click on a numbered aperture.")
            return

        sx, sy = _dao_xy(best_src)

        # Is this genuinely a different source from the one currently
        # extracted?  Within 1 px tolerance (DAO centroids are stable
        # but not bit-identical across runs).
        is_switch = (
                self._last_source_xy is None
                or np.hypot(sx - self._last_source_xy[0],
                            sy - self._last_source_xy[1]) > 1.0
        )

        # Source number (1-based) for the log line
        src_idx = next(
            (i for i, s in enumerate(self.top_sources)
             if _dao_xy(s) == _dao_xy(best_src)),
            None,
        )
        src_num = src_idx + 1 if src_idx is not None else "?"

        self._last_source_xy = (sx, sy)
        self._last_select_free = False

        # On a fresh source switch, auto-fit aperture and sky_gap to the
        # source's spatial FWHM.  User can still nudge afterwards with
        # the ± buttons or by typing.
        fit_msg = ""
        if is_switch:
            fwhm = estimate_source_fwhm(self.rotated_data, sx, sy)
            if np.isfinite(fwhm):
                self._target_fwhm = fwhm
                new_aper, new_gap, capped = self._autofit_geometry(fwhm)
                self.v_aper_half.set(str(new_aper))
                self.v_sky_gap.set(str(new_gap))
                self.v_sky_gap_hi.set(str(new_gap))  # auto-fit keeps both bands symmetric
                p = self._params()
                if p is None:
                    return
                self._last_p = p
                fit_msg = self._autofit_msg(fwhm, new_aper, new_gap, capped)
            else:
                fit_msg = ("  FWHM fit failed; keeping "
                           f"aperture ±{self.v_aper_half.get()}, "
                           f"sky gap {self.v_sky_gap.get()}.")

        self._display_extraction(sx, self._applied_y(sy), p)
        self._autocentre_if_enabled()

        self._redraw_nodes()
        self.canvas_spec.draw()
        self._update_source_info()

        if is_switch:
            self._log(
                f"Switched to source {src_num} at ({sx:.0f}, {sy:.0f}).{fit_msg}")
        else:
            self._log(f"Re-extracting source {src_num} at ({sx:.0f}, {sy:.0f}).")

    def _extract_free_point(self, cx, cy, p):
        """
        Extract a spectrum starting from the arbitrary click point (cx, cy),
        bypassing the nearest-detected-source snap.

        Used when the "Free selection" toggle is on, for sources outside
        the detected set — a faint companion, the central star of a
        planetary nebula, or any point on extended nebulosity.

        Behaviour mirrors a source switch: the click pixel is taken as the
        spectrum anchor, a spatial FWHM is fitted there to auto-size the
        aperture and sky gap, and the same extraction/display path runs.
        Unlike snap, no DAO centroid refinement is applied — the anchor is
        exactly where the user clicked.
        """
        # Clamp to the rotated-image bounds so a click on the canvas
        # margin still yields a valid in-array anchor.
        h, w = self.rotated_data.shape
        sx = float(np.clip(cx, 0, w - 1))
        sy = float(np.clip(cy, 0, h - 1))

        self._last_source_xy = (sx, sy)
        self._last_select_free = True

        # Auto-fit aperture / sky gap from the local PSF at the click.
        # A free point need not sit on a star, so a failed FWHM fit is
        # expected on nebulosity, and leaves the current values in place.
        fwhm = estimate_source_fwhm(self.rotated_data, sx, sy)
        fit_msg = ""
        if np.isfinite(fwhm):
            self._target_fwhm = fwhm
            new_aper, new_gap, capped = self._autofit_geometry(fwhm)
            self.v_aper_half.set(str(new_aper))
            self.v_sky_gap.set(str(new_gap))
            self.v_sky_gap_hi.set(str(new_gap))  # auto-fit keeps both bands symmetric
            p = self._params()
            if p is None:
                return
            self._last_p = p
            fit_msg = self._autofit_msg(fwhm, new_aper, new_gap, capped)
        else:
            # No measurable PSF (e.g. diffuse nebula) — keep the existing
            # aperture so the user's tuned values, or the last source's
            # fit, carry over.
            fit_msg = ("  no PSF fit; keeping "
                       f"aperture ±{self.v_aper_half.get()}, "
                       f"sky gap {self.v_sky_gap.get()}.")

        self._display_extraction(sx, self._applied_y(sy), p)
        self._autocentre_if_enabled()

        self._redraw_nodes()
        self.canvas_spec.draw()
        self._update_source_info()

        self._log(f"Free extraction at ({sx:.0f}, {sy:.0f}).{fit_msg}")

    # ------------------------------------------------------------------
    # Rotation fine-tune controls
    # ------------------------------------------------------------------

    def _angle_step(self, delta):
        """Nudge angle by delta degrees, update field, and re-run."""
        try:
            current = float(self.v_angle.get())
        except ValueError:
            current = 0.0
        new_angle = round(current + delta, 6)
        self.v_angle.set(f"{new_angle:.4f}")
        self._run()

    def _angle_minus(self):
        self._angle_step(-0.05)

    def _angle_plus(self):
        self._angle_step(+0.05)

    def _aper_step(self, delta):
        """
        Nudge the y-offset StringVar by delta pixels and re-extract.

        The offset is the single source of truth — _last_source_xy still
        holds the *applied* y (detected centroid + offset) so click handlers
        and re-snap on Run can locate the source, but the offset itself is
        what survives a parameter change.
        """
        if self._last_source_xy is None or self._last_p is None:
            return
        try:
            current = int(float(self.v_y_offset.get()))
        except ValueError:
            current = 0
        new_offset = current + delta
        self.v_y_offset.set(str(new_offset))
        sx, sy = self._last_source_xy   # raw centroid — not modified
        self._display_extraction(sx, self._applied_y(sy), self._last_p)
        self._redraw_nodes()
        self.canvas_spec.draw()

    def _autofit_geometry(self, fwhm):
        """
        Aperture half-height and sky gap for a source of this spatial FWHM.

        One place, because three paths auto-fit — initial Update, source
        switch, free click — and APERTURE_HALF_MAX has to apply to all of
        them.  A source switch is how a bright star usually arrives, so a
        cap that lives in only one of the three is a cap that never fires.

        Returns (aperture_half, sky_gap, capped).
        """
        fitted = max(3, int(round(APERTURE_FWHM_MULT * fwhm)))
        aperture = min(fitted, APERTURE_HALF_MAX)
        gap = max(2, int(round(SKY_GAP_FWHM_MULT * fwhm)))
        return aperture, gap, aperture < fitted

    @staticmethod
    def _autofit_msg(fwhm, aper, gap, capped):
        return (f"  FWHM ≈ {fwhm:.1f} px → aperture ±{aper}"
                + (" (capped — DAO's FWHM runs high on bright stars)"
                   if capped else "")
                + f", sky gap {gap}.")

    def _update_centre_button(self):
        """
        Highlight "Centre on trace" exactly when it is worth pressing.

        Same vocabulary as the config buttons (_update_config_buttons):
        filled accent = the action to take, outlined accent = available but
        not needed.  With autocentre on, every new extraction has already
        been centred and the button is redundant, so it steps back; turn
        autocentre off and it lights up to say the manual route is there.
        """
        self._btn_centre.configure(
            style="Action.TButton" if self.v_autocenter.get()
            else "Run.TButton")

    def _autocentre_if_enabled(self):
        """Re-centre the strip on the trace after a new extraction.

        Only the three paths that produce a NEW extraction call this — Run,
        source switch, free click.  Not the nudges or display toggles: those
        re-extract too, and auto-centring there would undo the ± buttons the
        moment they were pressed.
        """
        if not self.v_autocenter.get():
            return
        self._centre_on_trace(auto=True)

    def _centre_on_trace(self, auto=False):
        """
        Slide the aperture to the y that extracts the most spectrum.

        Scores the SAME columns the calibrated panel uses (the sp_min–sp_max
        window), which is what keeps the zero order out of it: the strip
        starts at the source, so its first columns are the saturated blob
        DAOStarFinder already mis-centred on — scoring those would just
        re-centre the aperture on the mistake.
        """
        if (self._last_source_xy is None or self._last_p is None
                or self.column_sums is None):
            if not auto:
                self._log("Centre on trace: run Update first.", level="warn")
            return
        p = self._last_p
        sx, sy = self._last_source_xy

        # Same pixel→wavelength mapping as _display_extraction, so "the
        # spectrum" means the same thing to the scan and to the plot.
        # Length comes from the live extraction, NOT from p["spectrum_width"]:
        # extract_spectrum clips the strip at the frame edge, so a source near
        # the right edge yields fewer columns and a full-width boolean mask
        # would not match.  The y shift cannot change this — the strip's
        # columns depend only on x.
        n_cols = len(self.column_sums)
        poly = self._validate_dispersion_poly(self.get_dispersion_poly(),
                                              n_cols)
        all_wls = pixels_to_wavelengths(np.arange(n_cols), p["dispersion"],
                                        poly_coeffs=poly)
        cols = (all_wls >= p["sp_min"]) & (all_wls <= p["sp_max"])
        if not cols.any():
            if not auto:
                self._log("Centre on trace: no columns inside the λ window.",
                          level="warn")
            return

        shift, scores = best_y_shift(
            sx, self._applied_y(sy), self.rotated_data, n_cols,
            p["aperture_half"], p["sky_gap"], p["sky_width"], cols=cols,
            max_shift=CENTRE_TRACE_MAX_SHIFT,
            sky_gap_hi=p.get("sky_gap_hi"), sky_width_hi=p.get("sky_width_hi"))

        if not scores:
            if not auto:
                self._log("Centre on trace: no usable extraction at any "
                          "shift.", level="warn")
            return
        if shift == 0:
            # Silent when automatic: this is the expected outcome on most
            # extractions, and a log line every Run saying nothing happened
            # is how a log stops being read.
            if not auto:
                self._log("Centre on trace: already best of "
                          f"{len(scores)} shifts — nothing to do.")
            return

        try:
            current = int(float(self.v_y_offset.get()))
        except ValueError:
            current = 0
        self.v_y_offset.set(str(current + shift))
        self._display_extraction(sx, self._applied_y(sy), self._last_p)
        self._redraw_nodes()
        self.canvas_spec.draw()

        gain = scores[shift] / scores[0] if scores.get(0, 0) > 0 else None
        msg = (f"{'Autocentre' if auto else 'Centre on trace'}: y offset "
               f"{current:+d} → {current + shift:+d} ({shift:+d} px)")
        if gain is not None:
            msg += f", {gain:.2f}× the flux"
        # An optimum sitting on the scan edge means the real one may be
        # further out — say so rather than let a clipped answer look final.
        if shift in (min(scores), max(scores)):
            msg += " — peak is at the scan edge, click again to go further"
        self._log(msg + ".")

    # ------------------------------------------------------------------
    # WCS plate-solve + SIMBAD source identification
    # ------------------------------------------------------------------

    def _redraw_source_labels(self, draw=True):
        """Draw numbered apertures + (optional) catalog names on ax_main.

        Removes any previously-drawn source markers (tracked in
        _source_marker_artists) and redraws them from self.top_sources,
        appending each source's catalog label from self.source_matches
        when a match exists.  Extraction-box patches are left untouched.

        Called from _load_and_run (draw deferred to the caller's final
        canvas draw) and standalone after _solve_to_wcs (draw immediately).
        """
        if not self.top_sources:
            return

        # Remove only the tracked marker artists, not the extraction boxes.
        for art in self._source_marker_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._source_marker_artists = []

        xs = [_dao_xy(s)[0] for s in self.top_sources]
        ys = [_dao_xy(s)[1] for s in self.top_sources]

        # CircularAperture.plot can disturb axis limits — capture/restore.
        xlim, ylim = self.ax_main.get_xlim(), self.ax_main.get_ylim()
        positions = np.transpose((xs, ys))
        aps = CircularAperture(positions, r=5)
        # aps.plot returns the artists it created (Circle patches in current
        # photutils).  Track them directly — they are added to ax.patches
        # alongside the extraction boxes, so tracking the specific returned
        # artists is the only way to remove just ours on the next redraw.
        # Dark blue, not the palette red: the working image is a negative
        # (cmap="Greys" — dark stars on a light field) and red reads as a
        # pale smear there.
        marker_color = "#1565c0"
        aperture_artists = aps.plot(color=marker_color, lw=1.2, alpha=0.8,
                                    ax=self.ax_main)
        if aperture_artists:
            self._source_marker_artists.extend(aperture_artists)
        self.ax_main.set_xlim(xlim)
        self.ax_main.set_ylim(ylim)

        matches = self.source_matches or []
        show_names = self.v_show_names.get()
        for idx, (sx, sy) in enumerate(zip(xs, ys)):
            m = matches[idx] if idx < len(matches) else None
            label = str(idx + 1)
            if show_names and m and m.label:
                label = f"{idx + 1}  {m.label}"
                if m.sp_type:
                    label += f" ({m.sp_type})"
            ann = self.ax_main.annotate(
                label, (sx, sy), xytext=(3, 3),
                textcoords="offset points",
                color=marker_color, fontweight="bold", fontsize=8)
            self._source_marker_artists.append(ann)

        if draw:
            self.canvas_main.draw_idle()

    def _browse_astap(self):
        """Pick the ASTAP executable; store it in the path field."""
        path = filedialog.askopenfilename(
            title="Locate ASTAP executable",
            filetypes=[("ASTAP executable", "astap*"),
                       ("Executable", "*.exe"),
                       ("All files", "*.*")],
        )
        if path:
            self.v_astap_path.set(path)

    def _solve_to_wcs(self):
        """Plate-solve the working frame and identify sources via SIMBAD.

        Runs ASTAP on the current rotated frame, converts each detected
        source's centroid to sky coordinates through the resulting WCS,
        and resolves each against SIMBAD.  Results are stored in
        self.source_matches and the overlay is redrawn with catalog names.

        The solve + SIMBAD queries run on a background thread so a
        livestack session keeps ingesting frames meanwhile; the result is
        marshalled back to the Tk thread by _apply_solve_result, which
        discards it if the rotation angle changed while solving.

        Guards: a frame must be loaded (Run first) and sources detected.
        Degrades with a clear message on solve failure or missing ASTAP.
        """
        if getattr(self, "_wcs_solving", False):
            return  # a solve is already running
        if self.rotated_data is None or not self.top_sources:
            messagebox.showinfo(
                "Run first",
                "Load a FITS file and press UPDATE to detect sources "
                "before solving to WCS.")
            return

        astap_path = self.v_astap_path.get().strip()
        if not astap_path:
            messagebox.showwarning(
                "ASTAP path not set",
                "Set the path to the ASTAP executable first "
                "(Plate solve section).")
            return

        header = self._target_header
        if header is None:
            messagebox.showwarning(
                "No header",
                "The target frame's FITS header is unavailable; "
                "reload the file and press UPDATE.")
            return

        # Snapshot everything the worker and the result application need
        # NOW: livestack may replace rotated_data / top_sources while the
        # solve runs.  rotated_data is only ever reassigned (never mutated
        # in place), so holding the reference is safe.
        snap_data    = self.rotated_data
        snap_sources = self.top_sources
        try:
            snap_angle = float(self._last_p["angle"])
        except (TypeError, KeyError, ValueError):
            snap_angle = None

        self._wcs_solving = True
        self._btn_solve.configure(state="disabled", text="Solving…")
        self._log("Plate-solving with ASTAP (background)…")

        def worker():
            try:
                wcs = srcid.solve_wcs(snap_data, header,
                                      astap_path=astap_path)
                # ASTAP can take ~20 s, during which the livestack may
                # re-detect sources (e.g. 10 → 20 as the stack deepens).
                # The WCS still describes the reference pixel grid (the
                # angle guard in _apply_solve_result protects geometry),
                # so identify against the NEWEST list, not the click-
                # time snapshot — otherwise a solve started on 10
                # sources reports "10/20 identified" and the extra
                # sources stay anonymous until the next manual solve.
                # top_sources is only ever reassigned, never mutated, so
                # grabbing the reference here is as safe as snapshotting.
                cur_sources = self.top_sources or snap_sources
                centroids = [_dao_xy(s) for s in cur_sources]
                # Wide-cone escalation eligibility: the brightest sources
                # (by DAO peak) are the ones whose saturated / zero-order-
                # smeared centroids can miss the tight cone; the V<=10 gate
                # in identify_sources keeps the wide retry honest.  A star
                # bright enough to bloat is near the frame's peak counts,
                # so "within 20% of the brightest" bounds the eligible set
                # without needing a saturation level from the header.
                try:
                    peaks = [float(s["peak"]) for s in cur_sources]
                    pmax = max(peaks)
                    wide_ok = [p >= 0.8 * pmax for p in peaks] \
                        if pmax > 0 else None
                except (KeyError, TypeError, ValueError):
                    wide_ok = None
                matches = (srcid.identify_sources(wcs, centroids,
                                                  wide_ok=wide_ok)
                           if wcs is not None else None)
                result = (wcs, matches, None)
            except Exception as e:
                cur_sources = snap_sources
                result = (None, None, e)
            # Queue, not after(): this runs on the solve worker thread, and a
            # cross-thread Tk call deadlocks against livestack's main-thread
            # draws (see _from_thread).
            self._from_thread(self._apply_solve_result,
                              result, cur_sources, snap_angle)

        threading.Thread(target=worker, daemon=True,
                         name="wcs-solve").start()

    def _apply_solve_result(self, result, snap_sources, snap_angle):
        """Apply a background solve's outcome on the Tk thread.

        Discards the solution if the rotation angle changed while the
        solve ran (the WCS no longer describes the current pixel grid).
        If the source list was re-detected meanwhile (livestack UPDATE at
        the same angle), matches are remapped by centroid position.
        """
        self._wcs_solving = False
        self._btn_solve.configure(state="normal", text="✦  Solve to WCS")

        wcs, matches, error = result
        if error is not None:
            self._log(f"WCS identification error: {error}", level="error")
            messagebox.showerror("WCS identification error", str(error))
            return
        if wcs is None:
            self._log("WCS solve failed — no solution.", level="warn")
            messagebox.showwarning(
                "Solve failed",
                "ASTAP did not return a solution for this frame.\n\n"
                "Check that the ASTAP path is correct, its star "
                "database is installed, and the frame has usable "
                "header hints (OBJCTRA/OBJCTDEC, XPIXSZ, FOCALLEN).")
            return

        try:
            cur_angle = float(self._last_p["angle"])
        except (TypeError, KeyError, ValueError):
            cur_angle = None
        if snap_angle is None or cur_angle is None \
                or abs(cur_angle - snap_angle) > 1e-6:
            self._log("Solve finished but the rotation angle changed "
                      "meanwhile — result discarded.", level="warn")
            return

        # Matches are aligned to the snapshot source list; if detection
        # re-ran while solving, realign them to the current list.
        if self.top_sources is not snap_sources:
            matches = self._remap_matches_to_sources(
                snap_sources, matches, self.top_sources)
        self.source_matches = matches
        self._main_wcs = wcs
        self._solved_angle = snap_angle

        n_named = sum(1 for m in matches if m is not None)
        self._redraw_source_labels(draw=True)
        # Refresh the info box for whatever source is currently selected.
        self._update_source_info()
        self._log(f"Identified {n_named}/{len(matches)} source(s).")
        if n_named == 0:
            self._log("No catalog matches within the search cone "
                      "(sources may be too faint to be catalogued).",
                      level="warn")

    def _remap_matches_to_sources(self, prev_sources, prev_matches,
                                  new_sources, tol_px=2.0):
        """Realign catalog matches to a freshly-detected source list.

        After a re-detection at the same rotation angle, the source list
        may be reordered or differ slightly in membership, but the stars
        sit at essentially the same pixels.  Build a new matches list,
        aligned 1:1 with new_sources, by assigning each new source the
        match of the nearest previous source within tol_px.  New sources
        with no nearby predecessor get None.

        Returns a list the same length as new_sources.
        """
        prev_xy = [_dao_xy(s) for s in prev_sources]
        out = []
        claimed = set()  # previous-source indices already assigned, so two
        #                  new sources can't both inherit the same match
        for s in new_sources:
            nx, ny = _dao_xy(s)
            best_i, best_d = None, float("inf")
            for i, (px, py) in enumerate(prev_xy):
                if i in claimed:
                    continue
                d = np.hypot(px - nx, py - ny)
                if d < best_d:
                    best_d, best_i = d, i
            if (best_i is not None and best_d <= tol_px
                    and best_i < len(prev_matches)):
                claimed.add(best_i)
                out.append(prev_matches[best_i])
            else:
                out.append(None)
        return out

    def _set_source_info(self, text):
        """Write text into the read-only source-info box."""
        widget = getattr(self, "source_info_text", None)
        if widget is None:
            return
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def _selected_source_index(self):
        """Index into top_sources of the currently extracted source, or None.

        Matches the active _last_source_xy against the detected centroids
        (1 px tolerance, as used by the click handler).  Returns None for a
        free-selected point or when nothing is extracted.
        """
        if (not self.top_sources or self._last_source_xy is None
                or self._last_select_free):
            return None
        lx, ly = self._last_source_xy
        for i, s in enumerate(self.top_sources):
            sx, sy = _dao_xy(s)
            if np.hypot(sx - lx, sy - ly) <= 1.0:
                return i
        return None

    def _update_source_info(self):
        """Refresh the info box from the selected source's catalog match.

        Shows the recovered SIMBAD details (main id, spectral type, object
        type, alternate names) for the currently selected source, or an
        appropriate placeholder when there is no solve, no match, or the
        selection is a free point.
        """
        if getattr(self, "source_info_text", None) is None:
            return

        if self.source_matches is None:
            self._set_source_info("No WCS solve yet.\nPress “Solve to WCS”.")
            self._set_simbad_button(None)
            return

        if self._last_select_free:
            self._set_source_info("Free selection — no catalog ID.")
            self._set_simbad_button(None)
            return

        idx = self._selected_source_index()
        if idx is None:
            self._set_source_info("Select a numbered source.")
            self._set_simbad_button(None)
            return

        m = self.source_matches[idx] if idx < len(self.source_matches) else None
        if m is None:
            self._set_source_info(
                f"Source {idx + 1}: no catalog match\nwithin search cone.")
            self._set_simbad_button(None)
            return

        self._set_source_info(f"Source {idx + 1}\n{m.info_text()}")
        self._set_simbad_button(m.main_id)

    def _set_simbad_button(self, main_id):
        """Enable the Check SIMBAD button for a matched source, else disable.

        Stashes the identifier to open so the command callback needs no
        arguments.  Any falsy / placeholder id disables the button.
        """
        btn = getattr(self, "btn_check_simbad", None)
        if btn is None:
            return
        valid = bool(main_id) and main_id != "(unnamed)"
        self._simbad_open_id = main_id if valid else None
        btn.config(state=("normal" if valid else "disabled"))
        lam = getattr(self, "btn_lamost", None)
        if lam is not None:
            lam.config(state=("normal" if valid else "disabled"))

    def _open_simbad_page(self):
        """Open the SIMBAD page for the selected source's identifier."""
        main_id = getattr(self, "_simbad_open_id", None)
        if not main_id:
            return
        # SIMBAD identifier query by name.  URL-encode the id (it can
        # contain spaces, '+', '*', etc.).
        url = ("https://simbad.cds.unistra.fr/simbad/sim-id?Ident="
               + urllib.parse.quote(main_id))
        try:
            webbrowser.open(url)
            self._log(f"Opened SIMBAD page for {main_id}.")
        except Exception as e:
            self._log(f"Could not open browser: {e}", level="warn")

    # ------------------------------------------------------------------
    # Spectra database
    # ------------------------------------------------------------------

    def _add_spectrum_to_db(self):
        """
        Save the current extraction to the spectra DB (db/spectra.db).

        Gathers: star identity (stored SIMBAD match for a detected source;
        an Add-time cone query at the click position for a free selection;
        position-only when neither yields a match — with a WCS we always
        have ICRS coordinates, the guidelines' ground-truth identity), the
        dataset (FITS file + header block), a run carrying the LIVE config
        snapshot (visually tweaked calibration included, saved or not),
        and one sample row per strip pixel: raw counts (pre-mask), the
        calibrated flux/σ where the display window covers it, and flags
        for contaminated / uncalibrated / bad-sky columns.

        A livestack works too: the autosaved livestack.fit is hashed as
        the dataset of record, stored under the literal path
        "livestacked" (its mid-stack DATE-OBS / total EXPTIME come from
        the autosave header).

        Everything is validated up front; ingest is a single transaction.
        """
        # Livestack: the autosaved livestack.fit (rewritten after every
        # accepted frame) is the file of record.  _frame_override also
        # covers a stopped-but-still-analysed stack.
        is_stack = self._frame_override is not None
        if is_stack and not (self._livestack_save_path
                             and os.path.isfile(self._livestack_save_path)):
            messagebox.showinfo(
                "Livestack not saved",
                "No autosaved livestack.fit is available — stack at least "
                "one frame, or check that the watch folder is writable.")
            return
        if (self.column_sums is None or len(self.column_sums) == 0
                or self._last_source_xy is None or self._last_p is None):
            messagebox.showinfo(
                "Nothing to save",
                "Extract a spectrum first (Run, then select a source).")
            return
        if (self._calibrated_pixels is None or self._calibrated_wls is None
                or len(self._calibrated_wls) == 0):
            messagebox.showinfo(
                "No calibrated spectrum",
                "The calibrated panel is empty — check the calibration "
                "file and the λ display window before adding to the DB.")
            return
        # _load_and_run nulls _main_wcs whenever the rotation angle
        # changes, so a live WCS is guaranteed to describe the current
        # rotated frame and centroids.
        if self._main_wcs is None:
            messagebox.showinfo(
                "No WCS", "Solve to WCS first — the DB needs sky "
                "coordinates as the target's identity.")
            return

        sx, sy = self._last_source_xy
        try:
            ra_meas, dec_meas = (
                float(v) for v in
                self._main_wcs.pixel_to_world_values(float(sx), float(sy)))
        except Exception as e:
            messagebox.showerror(
                "WCS error", f"Could not convert the source position to "
                f"sky coordinates: {e}")
            return

        # ── Identity ──────────────────────────────────────────────────
        # Detected source: reuse the stored match from the solve (no new
        # SIMBAD traffic).  Free selection: cone-query now, only on this
        # click — cached per position so a repeated Add doesn't re-query.
        match = None
        if self._last_select_free:
            if (self._free_match_cache is not None
                    and self._free_match_cache[0] == (sx, sy)):
                match = self._free_match_cache[1]
            else:
                self._log("Querying SIMBAD at the clicked position…")
                self.update_idletasks()
                # Synchronous query — ≤30 s worst case if SIMBAD hangs.
                # Thread it like _solve_to_wcs if that becomes intrusive.
                match = srcid.identify_sources(
                    self._main_wcs, [(sx, sy)])[0]
                # Only cache a hit: a None can mean "SIMBAD unreachable",
                # and that must stay retryable on the next Add.
                if match is not None:
                    self._free_match_cache = ((sx, sy), match)
            if match is None:
                self._log("No catalog object in the cone — saving a "
                          "position-only star (identifiable later by "
                          "coordinates).", level="warn")
        else:
            idx = self._selected_source_index()
            if (idx is not None and self.source_matches
                    and idx < len(self.source_matches)):
                match = self.source_matches[idx]

        header = self._target_header

        def hv(*keys, cast=None):
            """First present header value among keys, optionally cast."""
            for k in keys:
                if header is not None and k in header:
                    try:
                        return cast(header[k]) if cast else header[k]
                    except (TypeError, ValueError):
                        continue
            return None

        date_obs = hv("DATE-OBS")
        # Observation epoch (jyear) — the epoch of a *measured* position.
        obs_jyear = None
        if date_obs:
            try:
                from astropy.time import Time
                obs_jyear = float(Time(str(date_obs)).jyear)
            except Exception:
                pass

        if match is not None:
            has_cat = match.cat_ra_deg is not None
            star = dict(
                gaia_dr3_source_id=spectra_db.gaia_dr3_id(match.all_ids),
                main_id=(match.main_id
                         if match.main_id != "(unnamed)" else None),
                label=match.label or None,
                sp_type=match.sp_type or None,
                otype=match.otype or None,
                # Catalog (SIMBAD ICRS J2000) position when available,
                # best measured position otherwise (see the DB design
                # contract).
                ra_deg=match.cat_ra_deg if has_cat else ra_meas,
                dec_deg=match.cat_dec_deg if has_cat else dec_meas,
                pos_epoch_jyear=2000.0 if has_cat else obs_jyear,
            )
        else:
            star = dict(ra_deg=ra_meas, dec_deg=dec_meas,
                        pos_epoch_jyear=obs_jyear)

        # ── Dataset (file + header block) ─────────────────────────────
        # A livestack's per-frame files are transient (moved/archived
        # after the night), so the stored path is the literal
        # "livestacked" — a marker that there is no original to hunt
        # for.  The sha256 of the autosaved livestack.fit still ties the
        # row to the exact stack state that was analysed.
        hash_path = (self._livestack_save_path if is_stack
                     else self.v_target.get().strip())
        sha = None
        try:
            h = hashlib.sha256()
            with open(hash_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            sha = h.hexdigest()
        except OSError as e:
            self._log(f"Could not hash {os.path.basename(hash_path)}: {e} "
                      f"— dataset stored by path only.", level="warn")
        if is_stack and sha is None:
            # Without a hash, every night's "livestacked" dataset would
            # dedup onto one row — refuse rather than corrupt provenance.
            messagebox.showerror(
                "Add to DB failed",
                "The livestack.fit file could not be hashed; cannot "
                "record a livestack dataset without it.")
            return

        dataset = dict(
            fits_path="livestacked" if is_stack else hash_path,
            fits_sha256=sha,
            date_obs=str(date_obs) if date_obs is not None else None,
            exptime_s=hv("EXPTIME", "EXPOSURE", cast=float),
            telescope=hv("TELESCOP"),
            instrument=hv("INSTRUME"),
            object_name=hv("OBJECT"),
            # Site block: inputs for a future barycentric correction —
            # stored, not computed (≤0.7 Å at this resolution).
            site_lat_deg=hv("SITELAT", "LAT-OBS", "OBSGEO-B", cast=float),
            site_lon_deg=hv("SITELONG", "LONG-OBS", "OBSGEO-L", cast=float),
            site_elev_m=hv("SITEELEV", "ALT-OBS", "OBSGEO-H", cast=float),
        )

        # ── Run provenance: live config + this extraction's parameters ──
        snapshot = {
            "config": self._config_dict(),
            "extraction": dict(
                self._last_p,
                source_x=float(sx), source_y=float(sy),
                y_offset=self.v_y_offset.get(),
                free_selection=bool(self._last_select_free),
                measured_fwhm=(float(self._target_fwhm)
                               if self._target_fwhm is not None
                               and np.isfinite(self._target_fwhm) else None),
            ),
        }
        if is_stack:
            snapshot["livestack"] = dict(
                n_frames=self._stack_count,
                total_exptime_s=float(self._stack_total_exp),
                # The folder the FRAMES came from; not derivable from the
                # save path, which lives under _DATA_ROOT.
                source_folder=self._livestack_src_dir,
            )
        run = dict(
            run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_hash=spectra_db.git_hash(),
            config_json=json.dumps(snapshot),
        )

        # ── Samples: one row per strip pixel ──────────────────────────
        raw = (self._unmasked_col_sums
               if self._unmasked_col_sums is not None else self.column_sums)
        contam = self._last_contam_mask
        badsky = self._last_sky_col_flag
        cal_by_px = {
            int(px): (float(wl), fx, sg)
            for px, wl, fx, sg in zip(self._calibrated_pixels,
                                      self._calibrated_wls,
                                      self._calibrated_flux,
                                      self._calibrated_sigma)
        }
        samples = []
        for i in range(len(raw)):
            flags = 0
            if contam is not None and i < len(contam) and contam[i]:
                flags |= spectra_db.FLAG_CONTAM
            if badsky is not None and i < len(badsky) and badsky[i]:
                flags |= spectra_db.FLAG_BAD_SKY
            rc = float(raw[i])
            rc = rc if np.isfinite(rc) else None
            wl = fx = sg = None
            if i in cal_by_px:
                wl, fxv, sgv = cal_by_px[i]
                fx = float(fxv) if np.isfinite(fxv) else None
                sg = float(sgv) if np.isfinite(sgv) else None
            if fx is None:
                flags |= spectra_db.FLAG_NO_CAL
            samples.append((i, rc, wl, fx, sg, flags))

        spectrum = dict(source_x=float(sx), source_y=float(sy),
                        free_selection=bool(self._last_select_free),
                        match_sep_arcsec=(float(match.sep_arcsec)
                                          if match is not None else None))

        try:
            conn = spectra_db.connect()
            try:
                res = spectra_db.ingest_spectrum(
                    conn, star=star, dataset=dataset, run=run,
                    spectrum=spectrum, samples=samples)
            finally:
                conn.close()
        except Exception as e:
            self._log(f"Add to DB failed: {e}", level="error")
            messagebox.showerror("Add to DB failed", str(e))
            return

        name = (star.get("label") or star.get("main_id")
                or f"RA {star['ra_deg']:.4f} Dec {star['dec_deg']:+.4f}")
        self._log(
            f"DB: spectrum #{res['spectrum_id']} saved for {name} "
            f"(star_id {res['star_id']}, "
            f"{'new star' if res['created_star'] else 'existing star'}, "
            f"run {res['run_id']}, {len(samples)} samples).")

    def _open_dispersion_calculator(self):
        """Reopen the first-run dialog as a plain calculator."""
        FirstRunDialog(self, first_run=False)

    def _maybe_first_run(self, _tries=0):
        """Prompt a first-time user for an approximate dispersion.

        "First time" = the UI dotfile has never recorded a loaded config.
        That key is written whenever a config is loaded or saved, so anyone
        who has ever had a working setup is past this, and it survives a
        reinstall of the app while living outside the analysis config.

        Deliberately blocking (the dialog is modal): Å/px sizes the strip
        and seeds the Balmer scale search, so with the wrong value the very
        first Run produces something that looks like a broken program.  The
        dialog is pre-filled, so dismissing it still leaves a usable number.
        """
        if self._ui_state_get("last_config") is not None:
            return
        if self._ui_state_get("first_run_done"):
            return

        # The root must be ON SCREEN before a modal child is built.
        # after_idle fires before Tk has mapped the main window, and Windows
        # never displays a transient whose owner is unmapped — so the dialog
        # was invisible, held the input grab, and wait_window blocked with
        # the root still unmapped.  The process stayed alive with no window
        # at all (MainWindowHandle 0): "the app just doesn't start".
        # Polled rather than tkwait/wait_visibility so a window that never
        # becomes viewable (started iconified, no display) cannot hang us —
        # it just silently skips the prompt after ~5 s.
        if not self.winfo_viewable():
            if _tries < 50:
                self.after(100, lambda: self._maybe_first_run(_tries + 1))
            return

        try:
            dlg = FirstRunDialog(self, first_run=True)
            self.wait_window(dlg)
            if dlg.result is not None:
                self._log(f"Initial dispersion set to {dlg.result:g} Å/px. "
                          "Calibrate it properly from an A-type star "
                          "(Calibration → auto-suggest) when you have one.")
        except Exception:
            logging.getLogger(__name__).exception("first-run dialog failed")
        # Recorded either way: a user who dismissed it once should not be
        # met by it again every launch.
        self._ui_state_update("first_run_done", True)

    def _aper_minus(self):
        self._aper_step(-1)

    def _aper_plus(self):
        self._aper_step(+1)

    def _applied_y(self, sy):
        """
        Return sy + current v_y_offset.

        Every call site that feeds a raw _last_source_xy centroid into
        _display_extraction must go through this helper.  That keeps the
        raw DAO centroid unmodified in _last_source_xy and makes the offset
        field the single source of truth, independently adjustable across
        source switches, rotation changes, and node edits.
        Falls back to sy unchanged if the field contains non-numeric text.
        """
        try:
            return sy + int(float(self.v_y_offset.get()))
        except ValueError:
            return sy

    # ------------------------------------------------------------------
    # File selection and derotation

    # ------------------------------------------------------------------
    def _reset_analysis(self):
        """
        Wipe all analysis-derived state and clear ax_main, ax_raw, ax_cal.

        Called on file load so a new target doesn't sit alongside the
        previous file's rotated frame, strip, and spectra.  The caller
        (_browse_target) will subsequently render the new raw frame on
        ax_main via _show_original.

        Reset: the continuum anchors, which describe the specific
        target's flux shape and have no meaning carried over to another
        target.

        Deliberately NOT reset: angle, y-offset, dispersion and nodes —
        the loaded/edited configuration describes the optical setup, not
        the target, and must carry unchanged into every newly loaded FITS.
        """
        # Analysis-derived state
        self.rotated_data = None
        self.top_sources = None
        self.all_sources_xy = None
        self._target_header = None
        self.source_matches = None
        self._solved_angle = None
        self._main_wcs = None
        self.column_sums = None
        self._last_source_xy = None
        self._last_select_free = False
        self._last_p = None
        self._calibrated_wls = None
        self._calibrated_flux = None
        self._calibrated_sigma = None
        self._calibrated_pixels = None
        self._unmasked_col_sums = None
        self._free_match_cache = None
        self._target_fwhm = None
        # Per-target extraction by-products: a previous target's mask or
        # sky flag must not leak into the next one (the sequence generator
        # reads _last_contam_mask via getattr).
        self._last_contam_mask = None
        self._last_sky_col_flag = None
        self._current_anchor_resid = None
        self.continuum_anchors = []
        self._update_calibration_status_labels()

        # Disconnect main-image click handler if active
        if self._main_click_cid is not None:
            self.canvas_main.mpl_disconnect(self._main_click_cid)
            self._main_click_cid = None

        # Clear ax_main, ax_raw, ax_cal — and the zoom memory: the next
        # frame is a new context, it gets the full view.
        self._main_img_shape = None
        for ax in (self.ax_main, self.ax_raw, self.ax_cal):
            ax.cla()
            ax.set_facecolor("#0f0f1a")
        self.ax_main.set_title("No target loaded",
                               color="#a0a0c0", fontsize=8, pad=4)
        # cla() above resets tick_params to defaults — re-suppress.
        self._strip_main_ticks()
        self._label_axes()
        self.canvas_main.draw_idle()
        self.canvas_spec.draw_idle()

        # Tear down the strip figure (no pyplot manager to close)
        self.fig_strip = None
        if self.canvas_strip is not None:
            self.canvas_strip.get_tk_widget().destroy()
            self.canvas_strip = None
        for child in self.strip_inner.winfo_children():
            child.destroy()

        # Refresh node-marker bookkeeping: the markers lived on the
        # just-cleared ax_cal, so while the node objects in the parent are
        # untouched, their artist references are now invalid.
        self._node_markers = []
        self._redraw_nodes()

    def _show_image(self, data, *, title, is_rgb_candidate=False,
                    preserve_view=False):
        """
        Render an image array on ax_main with an auto-stretch and a title.

        Used for both the raw frame (at file load) and the rotated working
        image (after Auto/Update/source switch).  All callers operate on
        the same single axis, so a fresh _show_image fully replaces the
        previous image without any cross-state bookkeeping.

        Parameters
        ----------
        data : ndarray
            2D mono array, or 3D RGB array (only when is_rgb_candidate).
        title : str
            Title displayed above the axis.
        is_rgb_candidate : bool
            If True and ``data`` is 3D, render as colour; otherwise force
            a mono greyscale render.
        preserve_view : bool
            Keep the current pan/zoom instead of snapping back to full
            extent.  Set on a re-run of the SAME frame (an Update, a
            parameter nudge, a livestack frame landing) so the user's
            zoom survives; left False on a new file load so a fresh target
            starts at full extent.  Honoured only when the incoming frame
            has the same shape as the one currently displayed — otherwise
            the saved limits describe a different pixel grid and are dropped.
        """
        prev_lim = None
        if (preserve_view and getattr(self, "_main_img_shape", None)
                == data.shape and self.ax_main.images):
            prev_lim = (self.ax_main.get_xlim(), self.ax_main.get_ylim())

        self.ax_main.cla()
        self.ax_main.set_facecolor("#0f0f1a")
        if is_rgb_candidate and data.ndim == 3:
            def _sc(ch):
                return Stretch().stretch(ch.astype(float))
            if data.shape[0] <= 4:
                channels = [_sc(data[c])
                            for c in range(min(data.shape[0], 3))]
            else:
                channels = [_sc(data[:, :, c])
                            for c in range(min(data.shape[2], 3))]
            rgb = np.clip(np.stack(channels, axis=-1), 0, 1)
            im = self.ax_main.imshow(rgb, origin="lower",
                                interpolation="nearest", aspect="equal")
            _suppress_cursor_data(im)
        else:
            mono = Stretch().stretch(data.astype(float))
            finite = mono[np.isfinite(mono)]
            if finite.size:
                vmin = float(finite.min())
                vmax = float(finite.max())
                if vmax <= vmin:  # degenerate (flat frame) — widen a bit
                    vmax = vmin + 1.0
            else:  # all-NaN frame
                vmin, vmax = 0.0, 1.0
            im = self.ax_main.imshow(mono,
                                cmap="Greys", origin="lower",
                                vmin=vmin, vmax=vmax,
                                interpolation="nearest", aspect="equal")
            _suppress_cursor_data(im)

        self.ax_main.set_title(title, color="#a0a0c0", fontsize=8, pad=4)
        # cla() above resets tick_params to defaults — re-suppress.
        self._strip_main_ticks()
        # Restore the pre-cla() pan/zoom on a same-frame re-run (see the
        # preserve_view note); after imshow so it wins over the autoscale.
        if prev_lim is not None:
            self.ax_main.set_xlim(prev_lim[0])
            self.ax_main.set_ylim(prev_lim[1])
        self._main_img_shape = data.shape
        self.canvas_main.draw_idle()

    def _show_original(self, raw):
        """Show the raw (un-rotated) FITS array on ax_main."""
        # New file → full extent (preserve_view left False): a fresh target
        # must not inherit the previous one's zoom.
        self._show_image(raw,
                         title="Original frame — press Auto / Update",
                         is_rgb_candidate=True)

    def _load_default_target(self):
        """
        Show the default target's raw frame at startup.

        Mirrors the preview branch of _browse_target but is silent if
        the file is missing — startup shouldn't pop dialogs about a
        default the user hasn't chosen.
        """
        path = self.v_target.get().strip()
        if not path:
            return
        try:
            raw, _ = read_fits_image(path)
        except FileNotFoundError:
            self._log(f"Default target '{path}' not found — "
                      f"set Target FITS to begin.")
            return
        except Exception as e:
            self._log(f"Default target preview error: {e}")
            return
        self._show_original(raw)
        self._log(f"Loaded: {os.path.basename(path)}  "
                  f"— press Update to run, or Auto to derotate.")

    def _dlg_dir(self, key):
        """initialdir kwargs for a file dialog from the remembered folder."""
        d = self._last_dirs.get(key)
        return {"initialdir": d} if d and os.path.isdir(d) else {}

    def _ui_state_update(self, key, value):
        """Read-merge-write ONE top-level key of the UI dotfile.

        Other writers own sibling keys (the NINA dialog keeps "nina"),
        so dumping a partial in-memory view would clobber them — always
        merge against the file's current contents.
        """
        try:
            with open(self._ui_state_path, "r") as f:
                data = dict(json.load(f))
        except Exception:
            data = {}
        data[key] = value
        try:
            with open(self._ui_state_path, "w") as f:
                json.dump(data, f, indent=1)
        except OSError:
            pass  # convenience only — never fail the action over it

    def _ui_state_get(self, key, default=None):
        """Read ONE top-level key back out of the UI dotfile.

        Counterpart to _ui_state_update.  Always re-reads: the file is the
        shared surface between this window, the NINA dialog and a previous
        session, so an in-memory cache would go stale.
        """
        try:
            with open(self._ui_state_path, "r") as f:
                return json.load(f).get(key, default)
        except Exception:
            return default

    def _remember_dir(self, key, path):
        """Store the folder of a chosen path as the dialog's next start."""
        self._last_dirs[key] = path if os.path.isdir(path) \
            else os.path.dirname(path)
        self._ui_state_update("last_dirs", self._last_dirs)

    def _browse_target(self):
        """Open a file dialog, update the Target FITS field and show the original frame."""
        path = filedialog.askopenfilename(
            title="Select target FITS file",
            filetypes=[("FITS files", "*.fit *.fits *.fts"), ("All files", "*.*")],
            **self._dlg_dir("fits"),
        )
        if not path:
            return
        self._remember_dir("fits", path)
        # Config (angle/dispersion/nodes) survives the load, so apply it
        # immediately — no manual Update needed.
        self._load_target_path(path, run_after=True)

    def _load_target_path(self, path, run_after=False):
        """
        Load ``path`` as the target FITS: reset analysis, show the raw
        preview, and re-apply any config that was deferred pending a frame.

        Shared by the manual browse button and Livestack.  With
        ``run_after`` the pipeline is triggered automatically after the
        preview (Livestack has no user to press Update); a deferred config
        already triggers its own run, so that case must not run twice.
        Returns True if a preview was shown, False on read failure.
        """
        self.v_target.set(path)
        # A single-file load always reads that file — drop any live-stack
        # override left over from a previous stacking session.
        self._frame_override = None
        # Wipe any analysis from the previous target so the new raw frame
        # does not sit alongside a stale rotated frame, strip and spectra.
        # Angle/y-offset/dispersion/nodes survive the reset so the current
        # configuration (loaded or hand-edited) applies to the new frame.
        self._reset_analysis()
        self._log("Loading preview…")
        self.update_idletasks()
        try:
            raw, _ = read_fits_image(path)
            self._show_original(raw)
            suffix = "running…" if run_after else "press Update to run."
            self._log(f"Loaded: {os.path.basename(path)}  — {suffix}")
        except Exception as e:
            self._log(f"Preview error: {e}")
            return False

        # A config chosen before any FITS was deferred.  Now that a frame
        # exists, apply it — this also triggers a run.
        ran = False
        if self._pending_config is not None:
            cfg, cfg_path = self._pending_config
            self._pending_config = None
            self._log("Applying stored config to the loaded FITS…")
            self._apply_config(cfg, cfg_path)
            ran = True

        if run_after and not ran:
            self._run()
        return True

    # ------------------------------------------------------------------
    # Livestack — watch a folder and auto-load+run each new FITS
    # ------------------------------------------------------------------
    _LIVESTACK_POLL_MS = 1000
    _FITS_EXTS = (".fit", ".fits", ".fts")
    # Autosave filename, written to the session folder under _DATA_ROOT (see
    # _autosave_livestack), never into the watched folder.  _list_fits still
    # excludes the name so a stale livestack.fit left in a capture folder by
    # an older build is never stacked into a new stack.
    LIVESTACK_NAME = "livestack.fit"

    def _show_nina(self):
        """Open (or refocus) the single-instance NINA remote panel.

        A hidden (withdrawn) panel is still alive — polling, capture and
        mirror keep running — so reopen re-docks it instead of minting a
        new instance."""
        dlg = self._nina_dialog
        if dlg is not None and dlg.winfo_exists():
            if dlg.state() == "withdrawn":
                dlg.deiconify()
                self._dock_nina(dlg)
                self._btn_nina.config(text="◈  NINA…",
                                      style="Action.TButton")
            dlg.lift()
            dlg.focus_set()
            return
        self._nina_dialog = NinaDialog(self)

    def _usable_window_size(self):
        """Desktop area used by the explorer, capped for large monitors."""
        return (min(self.winfo_screenwidth(), 1920),
                max(self.winfo_screenheight() - 60, 540))

    def _restore_main_layout(self):
        w, h = self._usable_window_size()
        self.geometry(f"{w}x{h}+0+0")

    def _dock_nina(self, dialog):
        """Place the explorer and NINA side by side in the usable area."""
        dialog.update_idletasks()
        total_w, h = self._usable_window_size()
        nina_w = min(dialog.winfo_reqwidth(), max(total_w - 960, 1))
        main_w = total_w - nina_w
        self.geometry(f"{main_w}x{h}+0+0")
        dialog.geometry(f"{nina_w}x{h}+{main_w}+0")

    def _toggle_livestack(self):
        """Start/stop monitoring a folder for new FITS files."""
        if self._livestack_dir is not None:
            self._stop_livestack()
            return
        folder = filedialog.askdirectory(
            title="Select folder to monitor for new FITS",
            **self._dlg_dir("livestack"))
        if not folder:
            return
        self._remember_dir("livestack", folder)
        self._start_livestack_at(folder)

    def _start_livestack_at(self, folder):
        """Start the folder watch on ``folder`` (no dialog).

        The body of the classic Start Livestack flow, split out so the
        NINA panel can point the watch at its capture work folder; the
        button flow above is unchanged.  Caller guarantees no watch is
        currently running.
        """
        # A livestack with no active config runs on default parameters
        # (angle 0, linear dispersion…) and silently extracts garbage —
        # offer the previous session's config if one is remembered.
        # Either answer proceeds: a first-time user has no config yet.
        if self._loaded_config_path is None:
            last = None
            try:
                with open(self._ui_state_path, "r") as f:
                    last = json.load(f).get("last_config")
            except Exception:
                pass
            if last and os.path.isfile(last):
                if messagebox.askyesno(
                        "No config active",
                        "No analysis config is loaded — the livestack "
                        "would run with the current (possibly default) "
                        "parameters.\n\nLoad the last used config?\n\n"
                        f"{last}"):
                    try:
                        with open(last, "r") as f:
                            cfg = json.load(f)
                        if not isinstance(cfg, dict):
                            raise ValueError("not a JSON object")
                        self._apply_config(cfg, last)
                    except Exception as e:
                        self._log(f"Could not load {os.path.basename(last)}"
                                  f": {e} — continuing with current "
                                  f"parameters.", level="warn")
            else:
                messagebox.showwarning(
                    "No config active",
                    "No analysis config is loaded — the livestack will "
                    "run with the current parameters.\n\nLoad or build a "
                    "config for calibrated results.")
        self._livestack_dir = folder
        # Empty seen-set: the first poll treats every FITS already in the
        # folder as pending, so a stack started after capture began picks
        # up the backlog (the size-stability guard promotes one per tick).
        self._livestack_seen = set()
        self._livestack_pending = {}
        self._livestack_retry = {}
        # Fresh stack for this session.
        self._stack_ref = self._stack_sum = self._stack_wsum = None
        self._stack_ref_stars = self._stack_ref_tris = None
        self._stack_count = 0
        self._frame_override = None
        self._livestack_save_path = None
        self._livestack_src_dir = folder
        # One session folder per stack, named like a focus run.  Created
        # lazily on the first autosave, so a watch that never sees a frame
        # leaves nothing behind.
        self._livestack_out_dir = os.path.join(
            _DATA_ROOT, "livestack", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self._stack_mid_jds = []
        self._stack_total_exp = 0.0
        self._btn_livestack.configure(style="Run.TButton")
        self._refresh_livestack_button()
        existing = len(self._list_fits(folder))
        self._log(f"Livestack ON — watching {folder}; "
                  f"{existing} existing FITS will be stacked first, then new "
                  f"arrivals (first frame becomes the reference).")
        self._poll_livestack()

    def _stop_livestack(self):
        if self._livestack_after is not None:
            self.after_cancel(self._livestack_after)
            self._livestack_after = None
        self._livestack_dir = None
        self._livestack_pending = {}
        self._livestack_retry = {}
        self._btn_livestack.configure(text="Start Livestack",
                                      style="Action.TButton")
        self._log("Livestack OFF.")

    def _drain_then_stop_livestack(self, prev=None, stalls=0):
        """Stop the livestack, but only once it has ingested every FITS now in
        the folder.  Capture just wrote its last frame (and may have outrun the
        watch, which stabilises then ingests one file per tick); an immediate
        _stop_livestack would drop that backlog.  Bounded by lack of progress,
        not a fixed count, so a real backlog always drains while a genuinely
        stuck file (0-byte / never-stabilising) still lets the stop through."""
        if self._livestack_dir is None:
            return
        remaining = len([p for p in self._list_fits(self._livestack_dir)
                         if p not in self._livestack_seen])
        if remaining == 0:
            self._stop_livestack()
            return
        # No drop since last check counts as a stall; progress resets it.
        stalls = stalls + 1 if prev is not None and remaining >= prev else 0
        if stalls >= 10:
            self._log(f"Livestack: stopping with {remaining} frame(s) not yet "
                      f"stacked (no progress).", level="warn")
            self._stop_livestack()
            return
        self.after(self._LIVESTACK_POLL_MS,
                   lambda: self._drain_then_stop_livestack(remaining, stalls))

    def _refresh_livestack_button(self):
        """Show the running frame count + total exposure on the active button.

        Called after each accepted frame; a no-op label ("Stop Livestack")
        until the first frame lands.  _stop_livestack resets the text itself.
        """
        if self._livestack_dir is None:
            return
        n = self._stack_count
        exp = self._stack_total_exp
        self._btn_livestack.configure(
            text=f"Stop Livestack  ({n} fr · {exp:.0f} s)" if n
            else "Stop Livestack")

    def _list_fits(self, folder):
        """FITS candidates in capture order — os.listdir order is a
        filesystem accident, and the first backlog file promoted becomes
        the registration reference, so sort by mtime (path as tiebreak).
        A file vanishing between listdir and stat sorts last instead of
        failing the whole listing."""
        def mtime(p):
            try:
                return os.path.getmtime(p)
            except OSError:
                return float("inf")
        try:
            paths = [os.path.join(folder, f) for f in os.listdir(folder)
                     if f.lower().endswith(self._FITS_EXTS)
                     and f.lower() != self.LIVESTACK_NAME]
        except OSError:
            return []
        return sorted(paths, key=lambda p: (mtime(p), p))

    def _poll_livestack(self):
        """
        Tick: detect new FITS, wait for each to stop growing (guards
        against loading a file mid-copy), then load+run the stable one.
        Processes at most one file per tick to keep the UI responsive.
        """
        if self._livestack_dir is None:
            return
        try:
            current = self._list_fits(self._livestack_dir)
            # Register newly-seen files as pending (size unknown yet).
            for p in current:
                if p not in self._livestack_seen and p not in self._livestack_pending:
                    self._livestack_pending[p] = -1

            # Promote the first pending file whose size has stabilised.
            ready = None
            for p, last_size in list(self._livestack_pending.items()):
                try:
                    size = os.path.getsize(p)
                except OSError:          # vanished mid-copy — forget it
                    del self._livestack_pending[p]
                    continue
                if size == last_size and size > 0:
                    ready = p
                    break
                self._livestack_pending[p] = size

            if ready is not None:
                del self._livestack_pending[ready]
                self._log(f"Livestack: new file {os.path.basename(ready)}")
                if self._livestack_ingest(ready):
                    self._livestack_seen.add(ready)
                    self._livestack_retry.pop(ready, None)
                else:
                    # Transient read failure (producer still holding the
                    # file, cloud folder hiccup): send it back through the
                    # size-stability gate for a bounded number of retries;
                    # only a repeatably unreadable file is given up on, so
                    # one bad FITS cannot block the queue forever.
                    n = self._livestack_retry.get(ready, 0) + 1
                    if n >= 3:
                        self._livestack_seen.add(ready)
                        self._livestack_retry.pop(ready, None)
                        self._log(f"Livestack: giving up on "
                                  f"{os.path.basename(ready)} after {n} "
                                  f"failed reads.", level="warn")
                    else:
                        self._livestack_retry[ready] = n
                        self._livestack_pending[ready] = -1
        finally:
            # Reschedule even if this tick raised, so one bad file doesn't
            # silently kill the watch.
            if self._livestack_dir is not None:
                self._livestack_after = self.after(
                    self._LIVESTACK_POLL_MS, self._poll_livestack)

    def _stacked_mean(self):
        """Current virtual stack as a coverage-weighted mean image."""
        return self._stack_sum / np.maximum(self._stack_wsum, 1e-6)

    def _livestack_ingest(self, path):
        """
        Add one incoming FITS to the virtual stack and re-run the pipeline
        on the improved stack.

        First frame becomes the registration reference and seeds the stack
        (a fresh analysis, like a normal file load — the user then picks a
        source).  Each later frame is registered to the reference, warped
        into its coordinates and accumulated; the pipeline re-runs WITHOUT
        resetting analysis, so its existing re-snap keeps the selected
        source active as the stack refreshes.

        Returns True when the file is dealt with (stacked, or skipped for
        a deterministic reason like shape/registration) and False on a
        read failure, which may be transient — the poller retries those.
        """
        name = os.path.basename(path)
        try:
            raw, hdr = read_fits_image(path)
            mono = to_mono(raw).astype(np.float64)
        except Exception as e:
            self._log(f"Livestack: cannot read {name}: {e}", level="warn")
            return False

        if self._stack_count == 0:
            # Reference frame — start the stack, fresh analysis.
            self._stack_ref = mono
            # Detect reference stars + triangles once; reused verbatim by
            # every subsequent register_pair call (see __init__ note).
            self._stack_ref_stars = detect_stars(mono, use_gpu=False)
            self._stack_ref_tris = _build_triangles(self._stack_ref_stars)
            self._stack_sum = mono.copy()
            self._stack_wsum = np.ones_like(mono)
            self._stack_count = 1
            self.v_target.set(path)
            self._reset_analysis()                 # clears source selection
            # After the reset (which nulls _target_header): the reference
            # frame's header serves the whole stack — _load_and_run skips
            # the file read while _frame_override is set, and the WCS
            # solve needs the header's position/scale hints.
            self._target_header = hdr
            self._frame_override = self._stacked_mean()
            self._show_original(self._frame_override)
            self._log(f"Livestack: reference = {name}. "
                      f"Select a source; it stays selected as frames stack.")
            # _reset_analysis preserves angle/y-offset/dispersion/nodes, so
            # the current configuration (loaded or edited) already applies —
            # a plain run derotates/calibrates the reference accordingly.
            self._run()
            self._note_stack_frame(hdr)
            self._autosave_livestack()
            return True

        if mono.shape != self._stack_ref.shape:
            self._log(f"Livestack: {name} shape {mono.shape} ≠ reference "
                      f"{self._stack_ref.shape} — skipped.", level="warn")
            return True

        # Register to the reference (CPU — frame cadence leaves ample time).
        # M maps reference→source coords, which is exactly what
        # affine_transform needs to pull the source into reference space.
        M = register_pair(self._stack_ref, mono, use_gpu=False,
                          ref_stars=self._stack_ref_stars,
                          _ref_triangles=self._stack_ref_tris,
                          debug_callback=lambda m: None)
        if M is None:
            self._log(f"Livestack: {name} failed to register — skipped.",
                      level="warn")
            return True

        # register_pair's M is in (x, y) convention; affine_transform works
        # in (row=y, col=x).  Transpose the linear part and reverse the
        # offset so the source is pulled into reference space correctly.
        # A mismatch here collapses the stack into a double image.
        mat_yx = M[:, :2].T
        off_yx = M[:, 2][::-1]
        shape = self._stack_ref.shape
        warped = affine_transform(mono, mat_yx, offset=off_yx,
                                  output_shape=shape, order=1,
                                  mode="constant", cval=0.0)
        wmap = affine_transform(np.ones_like(mono), mat_yx, offset=off_yx,
                                output_shape=shape, order=1,
                                mode="constant", cval=0.0)
        self._stack_sum += warped
        self._stack_wsum += wmap
        self._stack_count += 1
        self._frame_override = self._stacked_mean()

        angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        self._log(f"Livestack: +{name} "
                  f"(dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} rot={angle:.2f}°); "
                  f"stack = {self._stack_count} frames.")
        self._run()   # re-runs on the new mean; re-snaps to selected source
        self._note_stack_frame(hdr)
        self._autosave_livestack()
        return True

    def _note_stack_frame(self, hdr):
        """Record an accepted frame's exposure + mid-exposure JD.

        Feeds the livestack.fit header: DATE-OBS becomes the mid-stack
        time (mean of frame mid-exposures) and EXPTIME the summed
        exposure.  A frame without a parseable DATE-OBS simply doesn't
        vote on the epoch; its exposure still counts toward the total.
        """
        exp = 0.0
        for k in ("EXPTIME", "EXPOSURE"):
            if k in hdr:
                try:
                    exp = float(hdr[k])
                    break
                except (TypeError, ValueError):
                    pass
        self._stack_total_exp += exp
        try:
            from astropy.time import Time
            self._stack_mid_jds.append(
                Time(str(hdr["DATE-OBS"])).jd + exp / 172800.0)
        except Exception:
            pass
        self._refresh_livestack_button()

    def _autosave_livestack(self):
        """Rewrite the current (unrotated) stack to livestack.fit.

        Called after every accepted frame, so there is always an on-disk
        file matching the analysed stack — Add-to-DB hashes it as the
        dataset of record (stored path: "livestacked").  It is written to
        this session's folder under _DATA_ROOT, never into the watched
        folder: that one is often a cloud-synced capture folder, and
        rewriting the stack there every few seconds would have the sync
        client re-uploading it all night.
        The header is the reference frame's, with DATE-OBS replaced by
        the mid-stack time and EXPTIME by the total stacked exposure, so
        both the plate-solve hints and the DB epoch stay meaningful.
        """
        path = os.path.join(self._livestack_out_dir, self.LIVESTACK_NAME)
        try:
            os.makedirs(self._livestack_out_dir, exist_ok=True)
        except OSError as e:
            self._livestack_save_path = None
            self._log(f"Livestack autosave failed: {e}", level="warn")
            return
        hdr = (self._target_header.copy()
               if self._target_header is not None else fits.Header())
        if self._stack_mid_jds:
            from astropy.time import Time
            mid = Time(float(np.mean(self._stack_mid_jds)), format="jd")
            hdr["DATE-OBS"] = (mid.isot,
                               "mid-stack (mean of frame mid-exposures)")
        if self._stack_total_exp:
            hdr["EXPTIME"] = (float(self._stack_total_exp),
                              "total stacked exposure")
        hdr["NFRAMES"] = (self._stack_count, "frames in livestack")
        try:
            fits.PrimaryHDU(
                np.ascontiguousarray(self._frame_override, dtype=np.float32),
                header=hdr).writeto(path, overwrite=True)
        except Exception as e:
            self._livestack_save_path = None
            self._log(f"Livestack autosave failed: {e}", level="warn")
            return
        if self._livestack_save_path is None:
            self._log(f"Livestack autosaving to {path} "
                      f"(refreshed after every frame).")
        self._livestack_save_path = path
        # Keep the working header in step: the DB dataset block and any
        # later WCS solve read _target_header.
        self._target_header = hdr

    def _browse_cal_file(self):
        """Open a file dialog and update the legacy-response file field."""
        path = filedialog.askopenfilename(
            title="Select legacy response file",
            filetypes=[("Data files", "*.dat *.txt *.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.v_response_file.set(path)
        self._log(f"Calibration file: {os.path.basename(path)}")

    def _show_cal_curve(self):
        """
        Open (or focus + refresh) the active-response-curve viewer.

        Single-instance: clicking again while one is open refreshes it
        from the current v_response_file rather than spawning a duplicate.
        The dialog handles missing-path and broken-file states itself.
        """
        existing = getattr(self, "_response_viewer", None)
        if existing is not None and existing.winfo_exists():
            existing.refresh()
            existing.lift()
            existing.focus_force()
            return

        self._response_viewer = ResponseCurveDialog(self)

    def _show_predictor(self):
        """
        Open (or focus + refresh) the spectral-type predictor window.

        Single-instance: clicking again while one is open re-runs the
        template match on the current calibrated spectrum.  The dialog
        handles the no-spectrum and no-library states itself.
        """
        existing = getattr(self, "_predictor_dialog", None)
        if existing is not None and existing.winfo_exists():
            existing.refresh()
            existing.lift()
            existing.focus_force()
            return

        self._predictor_dialog = PredictorDialog(self)

    def _show_full_spectrum(self):
        """
        Open (or focus + refresh) the full-size calibrated spectrum viewer.

        Single-instance: clicking again while one is open re-focuses
        and refreshes it from the current cache rather than spawning
        a duplicate.  The dialog reads data from this instance directly
        — no arguments needed beyond the parent reference.
        """
        if self._calibrated_wls is None or self._calibrated_flux is None:
            messagebox.showwarning(
                "No spectrum", "Run an extraction first (▶ UPDATE).")
            return

        existing = getattr(self, "_full_spec_dialog", None)
        if existing is not None and existing.winfo_exists():
            existing.refresh()
            existing.lift()
            existing.focus_force()
            return

        self._full_spec_dialog = FullSpectrumDialog(self)

    def _show_reference_library(self):
        """
        Open (or focus) the reference-spectra library browser.

        Single-instance: clicking again while one is open re-focuses it
        rather than spawning a duplicate.  The dialog is self-contained
        and reads only the parent's λ range for panel scaling.
        """
        existing = getattr(self, "_reference_library_viewer", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        self._reference_library_viewer = ReferenceLibraryDialog(self)

    def _show_lamost(self):
        """
        Open (or focus) the LAMOST DR11 low-res spectrum viewer for the
        currently selected, catalog-matched source.

        Forces the TYC designation as the LAMOST query name when the
        source carries one (LAMOST cross-IDs lean on Tycho); the cone
        position itself comes straight from the WCS-resolved ra_deg /
        dec_deg, so no name re-resolution is needed.  Single-instance.
        """
        if not getattr(self, "_simbad_open_id", None):
            messagebox.showwarning(
                "No source", "Solve to WCS and select a named source first.")
            return

        existing = getattr(self, "_lamost_dialog", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        # Resolve the selected SourceMatch to get exact position + IDs.
        ra = dec = None
        name = self._simbad_open_id
        idx = self._selected_source_index()
        if (idx is not None and self.source_matches
                and idx < len(self.source_matches)):
            m = self.source_matches[idx]
            ra, dec = m.ra_deg, m.dec_deg
            # Force a TYC identifier when one is among the aliases.
            tyc = next((i for i in ([m.main_id] + list(m.all_ids))
                        if i.upper().startswith("TYC")), None)
            if tyc:
                name = tyc

        # Current calibrated-panel wavelength limits, for direct overlay
        # comparison (None if no spectrum is drawn yet).
        xlim = None
        if self._calibrated_wls is not None:
            try:
                lo, hi = self.ax_cal.get_xlim()
                if hi > lo:
                    xlim = (float(lo), float(hi))
            except Exception:
                xlim = None

        self._lamost_dialog = LamostDialog(self, name, ra=ra, dec=dec, xlim=xlim)

    def _show_catalog(self, attr, factory):
        """Open (or focus) a catalogue browser dialog. Single-instance."""
        existing = getattr(self, attr, None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
        setattr(self, attr, factory(self, goto=self._nina_goto))

    def _nina_goto(self, name, ra_deg, dec_deg):
        """Catalogue-browser goto: slew the rig to a target (J2000 degrees).

        Routed through the NINA remote panel so its confirmation, busy
        check and mount-worker conventions apply — the panel must be open
        (it holds the host/port and the sticky connection).
        """
        dlg = self._nina_dialog
        if dlg is None or not dlg.winfo_exists():
            messagebox.showinfo(
                "NINA remote not open",
                "Open the NINA remote panel and connect to the rig first — "
                "the goto slews through it.")
            return
        dlg.slew_to(name, ra_deg, dec_deg)

    def _post_catalog_menu(self):
        """Drop the catalogue menu below its button."""
        menu = tk.Menu(self, tearoff=0)
        for label, attr, factory in CATALOG_BROWSERS:
            menu.add_command(
                label=label,
                command=lambda a=attr, f=factory: self._show_catalog(a, f))
        btn = self._btn_catalogs
        menu.tk_popup(btn.winfo_rootx(),
                      btn.winfo_rooty() + btn.winfo_height())

    def _show_sequence_generator(self):
        """
        Open (or focus) the spectrum-sequence generator window.

        The dialog owns its own controls, folder selection, Generate action,
        animation and GIF export, reading analysis state from this instance at
        Generate time.  Single-instance: re-focuses an existing window rather
        than spawning a duplicate, matching the full-spectrum viewer.
        """
        existing = getattr(self, "_sequence_dialog", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        self._sequence_dialog = SequenceGeneratorDialog(self)

    def _derotate(self):
        """
        Load the target FITS, collapse to mono, detect the tilt angle via
        Hough, update the rotation-angle field, and trigger a full Run.
        """
        path = self.v_target.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please set a Target FITS file first.")
            return

        self._log("Loading FITS for derotation…")
        self.update_idletasks()

        try:
            raw, _ = read_fits_image(path)
        except FileNotFoundError:
            messagebox.showerror("File not found", path)
            self._log("Error — file not found.")
            return
        except Exception as e:
            messagebox.showerror("FITS error", str(e))
            self._log("Error reading FITS.")
            return

        self._log("Detecting rotation angle…")
        self.update_idletasks()

        # Pass 2 needs the current parameter dict; None disables refinement
        # (estimate_angle records that in the diagnostics).
        p = self._params()

        # Resolve the selected first-pass method label to its function.
        label = self.v_first_pass.get()
        method = dict(FIRST_PASS_CHOICES).get(label)

        try:
            mono = to_mono(raw)
            angle2, diag = estimate_angle(mono, p, first_pass=method)
        except Exception as e:
            messagebox.showerror("Derotation error", str(e))
            self._log("Derotation failed.")
            return

        # Report pass 1, then whatever pass 2 did, from the diagnostics.
        self._log(
            f"Stage 1 [{diag['method']}]: {diag['angle1']:.4f}° "
            f"(support {diag['support']:.3g}).")
        if diag["stage2_ran"]:
            self._log(
                f"Stage 2: gradient peak {diag['peak_orientation']:.2f}°, "
                f"correction {diag['correction']:+.4f}° → "
                f"final angle {diag['angle2']:.4f}°")
        else:
            self._log(
                f"Stage 2: {diag['stage2_reason']}, keeping stage-1 angle.")

        # Update the angle field and trigger a single full run.
        # Suppressed: Auto-derotate produces a candidate angle that the
        # user may still want to fine-tune (e.g. via ±-step or by typing).
        # Only those subsequent user actions should mark the config dirty.
        self._suppress_dirty = True
        try:
            self.v_angle.set(f"{angle2:.4f}")
        finally:
            self._suppress_dirty = False
        self.update_idletasks()
        self._run()

    # ------------------------------------------------------------------
    # Dispersion calibration nodes — UI panel

    # ------------------------------------------------------------------

    def _build_nodes_panel(self, parent, start_row):
        """Build the compact non-linear calibration block on the left pane."""
        row = start_row

        ttk.Label(parent, text=_hdr("NON-LINEAR CALIBRATION", self._hdr_px_left, self._hdr_font),
                  style="Header.TLabel").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        ttk.Button(parent, text="Calibrate dispersion…", style="Action.TButton",
                   command=self._open_calibration_dialog).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        row += 1

        # One-line summary, kept fresh by _redraw_nodes()
        self.fit_info_var = tk.StringVar(value="No nodes yet.")
        ttk.Label(parent, textvariable=self.fit_info_var,
                  wraplength=200, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        ttk.Button(parent, text="Calibrate instrument response…",
                   style="Action.TButton",
                   command=self._open_response_dialog).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        row += 1

        ttk.Label(parent, textvariable=self.response_status_var,
                  wraplength=200, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        # Legacy response: a response curve file produced by another
        # program, as an alternative to the built-in response dialog.
        self.v_response_file = tk.StringVar(
            value=str(DEFAULTS["CALIBRATION_FILE"]))
        ttk.Label(parent, text="Legacy response").grid(
            row=row, column=0, sticky="w", pady=1)
        legacy_frame = ttk.Frame(parent, style="Panel.TFrame")
        legacy_frame.grid(row=row, column=1, sticky="ew", padx=(4, 0), pady=1)
        legacy_frame.columnconfigure(0, weight=1)
        ttk.Entry(legacy_frame, textvariable=self.v_response_file,
                  width=12).grid(row=0, column=0, sticky="ew")
        ttk.Button(legacy_frame, text="…", width=2,
                   command=self._browse_cal_file).grid(
            row=0, column=1, padx=(2, 0))
        row += 1

        ttk.Button(parent, text="Calibrate continuum…",
                   style="Action.TButton",
                   command=self._open_continuum_dialog).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        row += 1

        ttk.Label(parent, textvariable=self.continuum_status_var,
                  wraplength=200, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1



        # "Show calibration lines" toggle has moved to the right pane
        # (_build_right_pane) — it is a cosmetic overlay, not a
        # calibration action.

    def _update_calibration_status_labels(self):
        """Update status labels for response and continuum calibration."""
        # Instrument response calibration status
        if self.v_response_file.get().strip() or self._response_df_cache is not None:
            self.response_status_var.set("Status: Available")
        else:
            self.response_status_var.set("Status: Not available")

        # Continuum calibration status
        n_anchors = len(self.continuum_anchors)
        if n_anchors >= 2:
            self.continuum_status_var.set(f"Status: Available ({n_anchors} anchors)")
        else:
            self.continuum_status_var.set("Status: Not available (needs ≥ 2 anchors)")

    def _open_calibration_dialog(self):
        """Open the dispersion-calibration dialog window."""
        if not self._require_spectrum("calibrating dispersion"):
            return
        nodes_before = list(self.dispersion_nodes)
        dlg = CalibrationDialog(self)
        self.wait_window(dlg)
        # The dialog calls _rebuild_after_node_change() itself on each
        # mutation, so a fresh rebuild here is only needed defensively, or
        # when the node list differs from the one recorded on entry.
        if self.dispersion_nodes != nodes_before:
            self.rebuild_after_node_change()

    def _require_spectrum(self, action_description):
        """
        Return True if a spectrum has been extracted, else show a
        warning dialog and return False.  Centralises the guard used
        by features that depend on column_sums being populated (which
        in turn requires a FITS file to have been loaded and Run).
        """
        if self.column_sums is None or len(self.column_sums) == 0:
            messagebox.showwarning(
                "No spectrum loaded",
                f"Please load a FITS file and run extraction "
                f"(▶ UPDATE) before {action_description}."
            )
            return False
        return True

    def _open_response_dialog(self):
        """Open the instrument-response calibration dialog."""
        if not self._require_spectrum("calibrating the instrument response"):
            return
        dlg = ResponseCalibrationDialog(self)
        self.wait_window(dlg)

    def _open_continuum_dialog(self):
        """
        Open the continuum-calibration dialog.

        Requires not just an extracted spectrum but a wavelength-
        calibrated, response-corrected one (``_calibrated_wls`` /
        ``_calibrated_flux`` populated by the most recent UPDATE).  The
        continuum acts on that spectrum, so without it the dialog has
        nothing meaningful to show.
        """
        if not self._require_spectrum("calibrating the continuum"):
            return
        if self._calibrated_wls is None or self._calibrated_flux is None \
                or len(self._calibrated_wls) == 0:
            messagebox.showwarning(
                "No calibrated spectrum",
                "The continuum calibration operates on the wavelength-"
                "calibrated, response-corrected spectrum.  Run an "
                "extraction (▶ UPDATE) with a valid dispersion and "
                "calibration file first."
            )
            return
        dlg = ContinuumDialog(self)
        self.wait_window(dlg)
        self._update_calibration_status_labels()

    # ------------------------------------------------------------------
    # Full analysis config — save/load (JSON)
    # ------------------------------------------------------------------
    #
    # A complete configuration bundles the calibration file path AND
    # the inline calibration array, the rotation angle, the linear
    # dispersion, the strip y-offset, and the current non-linear
    # dispersion nodes into one JSON file.  This lets the user keep
    # separate named configurations per instrument setup or per
    # observing session and switch between them on demand.
    #
    # The JSON file is the single source of truth for persisted state:
    # there is no autosave, no shadow .dat file.  The Load/Save buttons
    # visually indicate whether the in-memory state is dirty (Save
    # highlighted) or whether no config is loaded (Load highlighted),
    # and a close-confirm dialog catches forgotten saves.
    #
    # Why embed the calibration array
    # -------------------------------
    # The path alone is fragile: rename or move the .dat file and the
    # config breaks.  Embedding the array (~50 KB for a typical
    # instrument response) makes the JSON self-contained — the config
    # remains usable even if the original .dat is gone.  The path is
    # kept alongside for reference and for the entry field.

    def _on_response_file_changed(self, *_):
        """
        StringVar write-trace on v_response_file.  Invalidates the embedded
        calibration cache so a manual path change forces the pipeline
        to read the new file fresh on the next Run.

        Note: _load_config sets the path *before* the cache, so the
        cache it then writes is not invalidated by its own .set() call.

        This callback fires regardless of self._suppress_dirty — the
        cache-invalidation logic must always run, even during config
        load.  The companion dirty-mark trace on the same variable is
        registered separately and does respect the suppress flag.
        """
        self._response_df_cache = None
        self._update_calibration_status_labels()

    def _fit_window_to_response(self, applied=None):
        """Point the display window at what the response curve can actually
        calibrate.

        The window (sp_min/sp_max) is deliberately not persisted — it is
        "where am I looking", not calibration.  Leaving it at the 4000–8000
        default after loading a config whose curve reaches further would
        hide real data behind a stale constant, so it is derived from the
        loaded curve instead: unpersisted, and always matching the data.

        Rounds INWARD: the window must stay inside the curve's coverage, or
        the edge samples calibrate to NaN and _display_extraction fires its
        coverage warning against a self-inflicted window.
        """
        cal_df = self._response_df_cache
        if cal_df is None or cal_df.empty:
            return
        w_lo = float(cal_df["wavelength"].min())
        w_hi = float(cal_df["wavelength"].max())
        if not (np.isfinite(w_lo) and np.isfinite(w_hi)):
            return
        lo = float(np.ceil(w_lo))
        hi = float(np.floor(w_hi))
        if hi <= lo:
            return
        self.v_sp_min.set(f"{lo:g}")
        self.v_sp_max.set(f"{hi:g}")
        if applied is not None:
            applied.append(f"display window {lo:g}–{hi:g} Å (from the curve)")

    def set_response_curve(self, wavelengths, factors, path=""):
        """
        Install a freshly-computed instrument response as the live
        calibration (response-calibration dialog's parent API).

        The embedded array — not the .dat — is what the pipeline and
        _collect_config read, so a new curve applies on the next Run and
        gets saved by Save config.  Exporting a .dat is an optional
        convenience, not a step on the way in.

        `path` is stored for reference only (blank when the curve was
        never exported).  It is set FIRST because its write-trace clears
        the cache; same ordering as _load_config.
        """
        self.v_response_file.set(path)
        self._response_df_cache = pd.DataFrame(
            {"wavelength": np.asarray(wavelengths, dtype=float),
             "factor": np.asarray(factors, dtype=float)})
        self._mark_dirty()
        self._update_calibration_status_labels()

        # Re-extract so the calibrated panel shows the new response now;
        # requiring an Update click after Apply is exactly the extra trip
        # the embedded array exists to avoid.
        if self._last_p is not None and self._last_source_xy is not None:
            sx, sy = self._last_source_xy
            self._display_extraction(sx, self._applied_y(sy), self._last_p)
            self._redraw_nodes()
            self.canvas_spec.draw()

    # ------------------------------------------------------------------
    # Dirty-flag bookkeeping for the Load/Save config buttons
    # ------------------------------------------------------------------

    def _mark_dirty(self, *_):
        """
        Trace callback fired by any tracked config-persisted variable
        (see the trace_add block in __init__) and called explicitly by
        rebuild_after_node_change for dispersion-node mutations.

        Respects self._suppress_dirty so that programmatic writes
        during _load_config, _reset_analysis, _browse_target and
        Auto-derotate do not falsely mark the config dirty.
        """
        if self._suppress_dirty:
            return
        if not self._dirty:
            self._dirty = True
            self._update_config_buttons()

    def _set_clean(self, loaded_path=None):
        """
        Clear the dirty flag and update the loaded-config path.  Called
        after a successful Save or Load.  ``loaded_path`` is the file
        whose state now matches the in-memory state.
        """
        self._dirty = False
        if loaded_path is not None:
            self._loaded_config_path = loaded_path
            # Remembered across sessions so a config-less livestack start
            # can offer "load the last used config?" (_start_livestack_at).
            self._ui_state_update("last_config", loaded_path)
        self._update_config_buttons()

    def _update_config_buttons(self):
        """
        Repaint the Load/Save config buttons to reflect current state.

        Fresh  (no config loaded, _dirty=False) → Load highlighted
        Clean  (config loaded, _dirty=False)    → both normal
        Dirty  (_dirty=True)                    → Save highlighted

        Highlighted = Run.TButton (filled accent).
        Normal      = Action.TButton (outlined accent).
        """
        if self._dirty:
            load_style, save_style = "Action.TButton", "Run.TButton"
        elif self._loaded_config_path is None:
            load_style, save_style = "Run.TButton", "Action.TButton"
        else:
            load_style, save_style = "Action.TButton", "Action.TButton"
        self._btn_load_cfg.configure(style=load_style)
        self._btn_save_cfg.configure(style=save_style)

    def _on_close(self):
        """
        WM_DELETE_WINDOW handler.  If the config has unsaved changes,
        ask the user whether to save before quitting.

        Yes    → save (cancellable via the file dialog; if cancelled,
                 stay open so no work is lost)
        No     → quit without saving
        Cancel → stay open
        """
        if not self._dirty:
            if self._livestack_dir is not None:
                self._stop_livestack()
            self.destroy()
            return
        answer = messagebox.askyesnocancel(
            "Unsaved changes",
            "The configuration has unsaved changes.\n\n"
            "Save before quitting?",
        )
        if answer is None:
            return                       # Cancel — stay open
        if answer:
            if not self._save_config():  # user cancelled the save dialog
                return                   #   → stay open, don't lose work
        # Stop the watch only once closing is certain — a cancelled close
        # must not silently kill a running livestack.
        if self._livestack_dir is not None:
            self._stop_livestack()
        self.destroy()

    def _gc_tick(self):
        """Periodic cycle collection on the Tk thread (see the
        gc.disable() rationale in __init__)."""
        gc.collect()
        self._gc_after = self.after(15_000, self._gc_tick)

    def destroy(self):
        """Final sweep of Tk-object cycles on the Tk thread while the
        interpreter is still alive (companion to the gc.disable()
        scheme in __init__ — anything left after this is freed at
        interpreter shutdown on the main thread)."""
        gc.collect()
        # A stray pyplot figure would own a withdrawn tk.Tk() manager window
        # that keeps Tcl_MainLoop — and the process — alive after this root
        # is gone; close any as a safety net.
        plt.close("all")
        super().destroy()
        # Only now is it safe to hand collection back to any thread: the
        # Tk objects whose off-thread finalization aborts Tcl are gone.
        if self._gc_was_enabled:
            gc.enable()

    def _config_dict(self):
        """
        Build the live analysis-configuration dict (the on-disk config
        format): calibration file path AND its inline array, rotation
        angle, linear dispersion, strip y-offset, and the full list of
        non-linear dispersion nodes.

        Read from the live widget vars, so it always reflects the current
        (possibly visually tweaked, unsaved) state — used both by
        Save-config and as the Add-to-DB provenance snapshot.
        """
        # Embed the calibration the PIPELINE is using: the in-memory
        # cache when active (same precedence as _display_extraction and the
        # response viewer), the on-disk .dat otherwise.  With an active
        # cache the stored path may point at a deleted file, which is what
        # embedding exists for, so no disk read is attempted and no warning
        # is due.
        cal_array = None
        cal_path = self.v_response_file.get().strip()
        cal_df = self._response_df_cache
        if cal_df is None and cal_path:
            try:
                cal_df = load_calibration_file(cal_path)
            except Exception as e:
                self._log(
                    f"Warning: could not read calibration file "
                    f"'{cal_path}' for embedding ({e}); saving path only.",
                    level="warn",
                )
        if cal_df is not None:
            # Two-column list of [wavelength, factor] pairs — same
            # shape as the on-disk .dat, just JSON-encoded.
            cal_array = [
                [float(w), float(f)]
                for w, f in zip(
                    cal_df["wavelength"].values,
                    cal_df["factor"].values,
                )
            ]

        # Stored as list of [pixel, wavelength] pairs to keep the
        # on-disk shape identical to self.dispersion_nodes.  Built
        # defensively: a malformed node (wrong shape / non-numeric)
        # is skipped with a warning rather than crashing the save.
        nodes_out = []
        for entry in self.dispersion_nodes:
            try:
                px, wl = float(entry[0]), float(entry[1])
                nodes_out.append([px, wl])
            except (TypeError, ValueError, IndexError):
                self._log("Warning: skipping a malformed dispersion node "
                          "during save.", level="warn")

        cfg = {
            "version": 2,
            "calibration_file": cal_path,
            "calibration_array": cal_array,
            # ASTAP executable path — per-machine, unlikely to change on a
            # given system, so it rides along in the analysis config.
            "astap_path": self.v_astap_path.get().strip(),
            "angle": self.v_angle.get(),
            "dispersion": self.v_dispersion.get(),
            "y_offset": self.v_y_offset.get(),
            # Zero-order anchor toggle — saved so a config reproduces the
            # calibrated state it was built in.  A config carrying a
            # calib_anchor_resid but loaded into a session where the toggle
            # was left off would otherwise silently skip the colour-robust
            # transfer the residual exists to provide.
            "zero_anchor": bool(self.v_zero_anchor.get()),
            # First-pass derotation method label — per-setup (trace
            # character / geometry).  Stored as its display label; the
            # loader validates it against FIRST_PASS_CHOICES.
            "first_pass": self.v_first_pass.get(),
            "dispersion_nodes": nodes_out,
            # Calibration star's zero-order anchor residual (strip pixels),
            # so the colour-robust wavelength-scale transfer survives a
            # save/load.  May be null when no nodes were calibrated, or in
            # a config written before this key existed; loaders treat
            # null/missing as "no anchor", i.e. Δ = 0.
            "calib_anchor_resid": (
                float(self._calib_anchor_resid)
                if (self._calib_anchor_resid is not None
                    and np.isfinite(self._calib_anchor_resid))
                else None
            ),
        }
        return cfg

    def _save_config(self):
        """
        Save the current analysis parameters (see _config_dict) to a
        JSON file chosen by the user.
        """
        path = filedialog.asksaveasfilename(
            title="Save analysis config",
            defaultextension=".json",
            initialfile="spectrum_config.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            **self._dlg_dir("config"),
        )
        if not path:
            return False    # user cancelled the file dialog
        self._remember_dir("config", path)

        cfg = self._config_dict()
        # Serialise to a string first so a non-serialisable value (e.g. a
        # stray numpy array in a node) aborts cleanly with a clear message
        # before anything touches the disk.
        try:
            cfg_text = json.dumps(cfg, indent=2)
        except (TypeError, ValueError) as e:
            self._log(f"Config not saved — could not serialise: {e}",
                      level="error")
            messagebox.showerror(
                "Save error",
                f"The configuration could not be serialised to JSON "
                f"({e}).\n\nThe file was not written.")
            return False

        # Write beside the destination and os.replace into place: a
        # disk-full, disconnect or crash mid-write then costs only the
        # temp file, never the last known-good config.
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                f.write(cfg_text)
            os.replace(tmp, path)
            self._log(f"Config saved: {os.path.basename(path)}")
            self._set_clean(loaded_path=path)
            return True
        except OSError as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            messagebox.showerror("Save error", str(e))
            return False

    def _load_config(self):
        """
        Load analysis parameters from a JSON file chosen by the user.

        All keys are optional: any key missing from the file leaves the
        corresponding field at its current value (i.e. current default).
        Nodes, if present, replace the current in-memory list and a
        rebuild is triggered.  The calibration array, if present, is
        cached in memory so the pipeline uses it instead of reading the
        path-referenced .dat file — this makes the config robust to
        moves or renames of the original calibration file.

        A FITS need not be loaded first.  With a target present the config
        is applied and the pipeline re-runs immediately; with no target the
        parameters are still applied to the controls and the config is
        stashed in self._pending_config so _browse_target re-applies it (and
        runs) once a frame exists.  Extraction need not have run either
        (column_sums may be empty).
        """

        path = filedialog.askopenfilename(
            title="Load analysis config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            **self._dlg_dir("config"),
        )
        if not path:
            return
        self._remember_dir("config", path)

        try:
            with open(path, "r") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror("Load error", str(e))
            return

        if not isinstance(cfg, dict):
            messagebox.showerror("Load error",
                                 "Config file does not contain a JSON object.")
            return

        self._apply_config(cfg, path)

        # No FITS loaded yet: the parameters are applied and will survive
        # the frame load; stash the config so the frame provider
        # (_browse_target today, an external loader tomorrow) re-applies it
        # once a frame exists — mainly to trigger the automatic first run.
        if not self.v_target.get().strip():
            self._pending_config = (cfg, path)
            self._log("No FITS loaded yet — config stored; it will be "
                      "applied automatically when a target FITS is loaded.")

    def _apply_config(self, cfg, path):
        """
        Apply an already-parsed config dict to the UI/analysis state and,
        if a target FITS is loaded, re-run the pipeline.  Split out from
        _load_config so a config chosen before any FITS can be re-applied
        by the frame provider (_browse_target) once a frame exists.
        """
        applied = []

        # All programmatic .set() calls in this method must not flip
        # the dirty flag — they reflect on-disk state being copied into
        # memory, not user edits.  rebuild_after_node_change also
        # respects this guard (its _mark_dirty call is a no-op while
        # _suppress_dirty is True).
        self._suppress_dirty = True
        try:
            # Scalar fields — set as strings so the StringVars behave
            # exactly as if the user had typed them in.  v_response_file is
            # set FIRST so its write-trace clears _response_df_cache before
            # the cache is overwritten below; setting it later would let the
            # trace wipe the freshly populated cache.
            if "calibration_file" in cfg:
                self.v_response_file.set(str(cfg["calibration_file"]))
                applied.append("calibration file")
            if "astap_path" in cfg and str(cfg["astap_path"]).strip():
                self.v_astap_path.set(str(cfg["astap_path"]))
                applied.append("ASTAP path")
            if "angle" in cfg:
                self.v_angle.set(str(cfg["angle"]))
                applied.append("angle")
            if "dispersion" in cfg:
                self.v_dispersion.set(str(cfg["dispersion"]))
                applied.append("dispersion")
            if "y_offset" in cfg:
                self.v_y_offset.set(str(cfg["y_offset"]))
                applied.append("y-offset")
            # Zero-order anchor toggle.  Only JSON booleans (and 0/1 from
            # a hand-edit) are accepted: bool("false") is True, so a blind
            # coercion would silently enable the anchor — and change the
            # wavelength solution — on a string value.  Anything else is
            # warned about and leaves the current toggle untouched.
            if "zero_anchor" in cfg:
                za = cfg["zero_anchor"]
                if isinstance(za, bool) or za in (0, 1):
                    self.v_zero_anchor.set(bool(za))
                    applied.append("zero-order anchor toggle")
                else:
                    self._log(f"Config zero_anchor value {za!r} is not a "
                              f"boolean — keeping the current setting.",
                              level="warn")

            # First-pass derotation method.  Validate against the known
            # choice labels — an unrecognised label (stale name, typo,
            # hand-edit) would resolve to None in _derotate and silently
            # disable method selection, so fall back to the default and
            # warn rather than storing a dead label.
            if "first_pass" in cfg:
                fp_label = str(cfg["first_pass"])
                if fp_label in dict(FIRST_PASS_CHOICES):
                    self.v_first_pass.set(fp_label)
                    applied.append("first-pass method")
                else:
                    self._log(
                        f"Config first-pass method '{fp_label}' is not a "
                        f"recognised choice; keeping "
                        f"'{self.v_first_pass.get()}'.", level="warn")

            # Embedded calibration array — populate the in-memory cache
            # AFTER setting the path field (whose trace clears the cache).
            # Shape: list of [wavelength, factor] pairs.  Skips silently if
            # the array is missing, empty, or malformed — the pipeline will
            # then fall back to reading the path-referenced .dat file.
            if cfg.get("calibration_array"):
                try:
                    arr = cfg["calibration_array"]
                    wls = [float(p[0]) for p in arr]
                    facs = [float(p[1]) for p in arr]
                    if len(wls) >= 2:
                        self._response_df_cache = pd.DataFrame(
                            {"wavelength": wls, "factor": facs})
                        applied.append(f"{len(wls)}-row calibration array")
                        self._fit_window_to_response(applied)
                    else:
                        self._response_df_cache = None
                except (TypeError, ValueError, IndexError):
                    self._log("Warning: embedded calibration array is "
                              "malformed; ignoring (pipeline will read "
                              "the file path instead).", level="warn")
                    self._response_df_cache = None

            # Non-linear dispersion nodes.  Validate shape: list of
            # 2-element numeric pairs.  Invalid entries are skipped so a
            # partially-corrupted file still loads what it can.  A missing
            # or empty list is reported per the user request.
            #
            # Note: the calib_anchor_resid restore is deliberately nested
            # inside this block.  The residual is a Δ-shift applied to the
            # node-defined dispersion; with no nodes it has nothing to
            # anchor and is meaningless, so a config carrying a residual but
            # no dispersion_nodes key intentionally drops it.
            nodes_in_cfg = "dispersion_nodes" in cfg
            if nodes_in_cfg:
                raw_nodes = cfg["dispersion_nodes"]
                new_nodes = []
                if isinstance(raw_nodes, list):
                    for entry in raw_nodes:
                        try:
                            px, wl = float(entry[0]), float(entry[1])
                            new_nodes.append([px, wl])
                        except (TypeError, ValueError, IndexError):
                            continue
                self.dispersion_nodes = new_nodes
                if new_nodes:
                    applied.append(f"{len(new_nodes)} dispersion node(s)")
                else:
                    self._log("No non-linear calibration found, using "
                              "linear configuration.")

                # Calibration-star zero-order anchor residual.  Restore it
                # BEFORE rebuild_after_node_change, whose capture branch
                # would otherwise overwrite it with the current source's
                # residual.  Missing/null/invalid → None, i.e. Δ = 0.
                #
                # Version 2 measures the residual against the strip origin
                # (floored centroid) rather than the float centroid.  A v1
                # residual differs by frac(centroid) and would apply a wrong
                # sub-pixel shift, so it is discarded and the user is asked
                # to recalibrate.
                cfg_version = cfg.get("version", 1)
                car = cfg.get("calib_anchor_resid", None)
                if cfg_version < 2 and car is not None:
                    self._log(
                        "Config predates the zero-order anchor convention "
                        "(v2); discarding its stored anchor residual. "
                        "Re-run dispersion calibration on this setup to "
                        "restore the colour-robust wavelength transfer, then "
                        "Save config to upgrade it.", level="warn")
                    car = None
                try:
                    self._calib_anchor_resid = (
                        float(car) if car is not None
                                      and np.isfinite(float(car)) else None)
                except (TypeError, ValueError):
                    self._calib_anchor_resid = None

                # Redraw cal panel + node markers + fit-info label.
                # rebuild_after_node_change's _mark_dirty is suppressed by
                # the enclosing _suppress_dirty block.
                self.rebuild_after_node_change()
            else:
                # No nodes key at all — same user-facing message; leaves
                # the current node list untouched per the "partial config"
                # rule.
                self._log("No non-linear calibration found in config; "
                          "keeping current nodes.")
        finally:
            self._suppress_dirty = False

        self._update_calibration_status_labels()
        fname = os.path.basename(path)
        if applied:
            self._log(f"Config loaded ({fname}): " + ", ".join(applied))
        else:
            self._log(f"Config loaded ({fname}) but contained no "
                      f"recognised parameters.", level="warn")

        # Sanity warning if the path-referenced calibration file no
        # longer exists on disk.  Not blocking — the embedded array
        # (if any) will be used by the pipeline regardless.
        cal_path = self.v_response_file.get().strip()
        if cal_path:
            try:
                with open(cal_path, "r"):
                    pass
            except OSError:
                if self._response_df_cache is not None:
                    self._log(f"Note: calibration file '{cal_path}' "
                              f"not found on disk — using embedded "
                              f"array from the config.", level="warn")
                else:
                    self._log(f"Note: calibration file '{cal_path}' "
                              f"not found on disk and no embedded "
                              f"array — Run will fail until you "
                              f"browse to a valid file.", level="warn")

        # The on-disk file is now mirrored in memory — this is the new
        # "clean" baseline.  Subsequent user edits will re-dirty it.
        self._set_clean(loaded_path=path)

        # Trigger a full re-run so the user immediately sees the new
        # configuration applied.  If no target is set or the target
        # file is missing the existing error handling in _run() will
        # surface a dialog — same behaviour as the manual Update
        # button on a cold start.
        if self.v_target.get().strip():
            self._run()

    # ------------------------------------------------------------------
    # Reference-line drawing
    # ------------------------------------------------------------------
    def _draw_raw_panel(self, p, col_sums):
        """
        Draw the raw (background-subtracted) spectrum on ``self.ax_raw``:
        the grey trace, the per-column rainbow fill, axis limits, the
        wavelength locator/formatter, labels, grid, and reference-line
        groups.

        Assumes the caller has already cleared ``ax_raw`` (``cla()`` +
        facecolor + ``_label_axes()``).  Pure drawing — no I/O, no state
        mutation.  Returns the ``ymax`` used for the y-limit, which is also
        the value passed to the reference-line groups.
        """
        self.ax_raw.plot(col_sums, color="#c0c0d0", linewidth=0.6, zorder=3)
        pix = np.arange(len(col_sums))
        rainbow_fill(self.ax_raw, pix, col_sums, zorder=2,
                     color_wls=pix * p["dispersion"], flux_alpha=False)
        self.ax_raw.set_xlim(p["sp_min"] / p["dispersion"],
                             p["sp_max"] / p["dispersion"])

        # Scale the y-axis to the trailing (redder) columns so the bright
        # blue end doesn't flatten faint red features.  Guard against a
        # fully-masked tail (all-NaN → nanmin/nanmax warn and return NaN,
        # which would poison set_ylim): fall back to the full strip, then
        # to a plain 0..1 window if everything is NaN.
        tail = (col_sums[-RAW_YLIM_TAIL_COLS:]
                if len(col_sums) > RAW_YLIM_TAIL_COLS else col_sums)
        if not np.any(np.isfinite(tail)):
            tail = col_sums
        if np.any(np.isfinite(tail)):
            ymin = np.nanmin(tail)
            ymax = np.nanmax(tail)
        else:
            ymin, ymax = 0.0, 1.0
        self.ax_raw.set_ylim(ymin, ymax * 1.1)

        self.ax_raw.set_facecolor("#0f0f1a")
        self.ax_raw.tick_params(labelsize=6, colors="#a0a0c0")
        self.ax_raw.xaxis.set_major_locator(
            ticker.MultipleLocator(500 / p["dispersion"]))
        self.ax_raw.xaxis.set_major_formatter(
            ticker.FuncFormatter(custom_formatter(p["dispersion"])))
        self.ax_raw.set_xlabel("Wavelength (Å)", fontsize=7, color="#a0a0c0")
        self.ax_raw.set_ylabel("Flux (counts)", fontsize=7, color="#a0a0c0")
        self.ax_raw.grid(True, color="#2a2a4e", linewidth=0.5, zorder=1)
        # force_linear: the raw panel's tick labels come from custom_formatter,
        # which is a plain linear pixel×dispersion mapping.  Placing the
        # reference lines via the (non-linear) polynomial would leave each
        # line off its own labeled gridline whenever a curved fit is active.
        # The raw panel is the linear quick-look view; the calibrated panel
        # is the wavelength-accurate one.  Keep p (real pixel dispersion) so
        # lines land at wavelength/dispersion, matching the ticks.
        self._draw_reference_line_groups(p, ymax, force_linear=True)
        return ymax

    def _draw_cal_panel(self, p, wls, norm_int, poly):
        """
        Draw the calibrated, normalised spectrum on ``self.ax_cal``: the
        NaN-skipping per-segment rainbow fill, the white overlay line, axis
        limits, labels, grid, and reference-line groups (forced linear,
        since this axis is already in Å).

        Assumes the caller has already cleared ``ax_cal`` and has computed
        the final ``wls`` / ``norm_int`` / ``poly``.  Does NOT redraw node
        markers — the caller owns that (only the toggle path needs it on
        this code path).  Pure drawing — no I/O, no state mutation.
        """
        rainbow_fill(self.ax_cal, wls, norm_int, zorder=2)
        self.ax_cal.plot(wls, norm_int, color="white",
                         linewidth=0.6, alpha=0.6, zorder=3)
        self.ax_cal.set_xlim(p["sp_min"], p["sp_max"])
        self.ax_cal.set_ylim(bottom=0)
        self.ax_cal.set_xlabel("Wavelength (Å)", fontsize=7, color="#a0a0c0")
        ylabel = "Norm. flux  (poly λ)" if poly is not None else "Norm. flux  (linear λ)"
        self.ax_cal.set_ylabel(ylabel, fontsize=7, color="#a0a0c0")
        self.ax_cal.tick_params(labelsize=6, colors="#a0a0c0")
        self.ax_cal.grid(True, color="#3d3d6b", linewidth=0.5, zorder=1)

        # Reference lines on the calibrated panel.  Its x-axis is already
        # in Å, so pass dispersion=1.0 / force_linear so each line is placed
        # at xpix = wl.
        cal_ymax = float(np.nanmax(norm_int)) if len(norm_int) else 1.0
        _cal_p = dict(p, dispersion=1.0)
        self._draw_reference_line_groups(_cal_p, cal_ymax,
                                         ax=self.ax_cal, force_linear=True)

        # Keep the detached Full-Spectrum window in sync with this panel.
        self._sync_full_spectrum()

    def _sync_full_spectrum(self):
        """Schedule a refresh of the open Full-Spectrum window so it tracks
        the inline calibrated panel.  No-op when that window isn't open.

        Debounced (200 ms, coalescing): the refresh runs as its own
        event-loop task instead of inline in the pipeline callback, so a
        burst of updates (livestack frames, rapid parameter nudges) costs
        one big-window render, and user events stay responsive in between.
        """
        if getattr(self, "_full_spec_dialog", None) is None:
            return
        if self._fs_sync_after is not None:
            try:
                self.after_cancel(self._fs_sync_after)
            except tk.TclError:
                pass
        self._fs_sync_after = self.after(200, self._do_full_spectrum_sync)

    def _do_full_spectrum_sync(self):
        self._fs_sync_after = None
        dlg = getattr(self, "_full_spec_dialog", None)
        if dlg is None:
            return
        try:
            if dlg.winfo_exists():
                dlg.refresh()
                # refresh() ends in draw_idle(); render synchronously and
                # flush display updates so the window repaints even while
                # livestack after-callbacks keep the event loop busy.
                dlg.canvas.draw()
                dlg.update_idletasks()
        except tk.TclError:
            pass

    def _draw_reference_line_groups(self, p, y_max, ax=None, force_linear=False,
                                    fontsize=6, occupied=None):
        """
        Draw whichever reference-line groups are currently enabled.

        Parameters
        ----------
        p   : parameter dict (must contain ``dispersion``).
        y_max : float – top of the y-range; labels are anchored here.
        ax  : matplotlib Axes to draw on.  Defaults to ``self.ax_raw``.
              Pass ``self.ax_cal`` (with ``dispersion`` effectively 1 and
              ``poly_coeffs=None``) to draw on the calibrated panel whose
              x-axis is already in Å.
        force_linear : bool
            If True, skip the polynomial dispersion lookup and place lines
            via the linear ``p["dispersion"]`` mapping.  Two callers set it:
            • the raw panel, whose ticks use the linear pixel×dispersion
              formatter (``custom_formatter``) — so lines must use the same
              linear mapping to stay on their labelled gridlines, while
              keeping the real pixel ``p["dispersion"]``; and
            • the calibrated panel, whose x-axis is already in Å — there the
              caller also passes ``dispersion=1.0`` so lines land at
              xpix = wl.
        occupied : list or None
            Shared label-slot list (see ``plot_reference_lines``).  One
            list is used across all the enabled groups so their labels
            stagger against each other rather than each group starting
            from a clear axis.  A caller that draws further lines on the
            same axes afterwards — the full-spectrum viewer's annotation
            overlay — passes its own list in and reuses it.

        Returns
        -------
        set of float
            The wavelengths of the enabled groups.  The full-spectrum
            viewer's annotation overlay uses it to avoid stacking a
            second label on a line the user already has switched on.
        """
        if ax is None:
            ax = self.ax_raw
        poly = None if force_linear else self.get_dispersion_poly()
        n_pixels = len(self.column_sums) if self.column_sums is not None else None
        groups = [
            (self.v_balmer_lines,      BALMER_LINES,      "#ff6060"),
            (self.v_helium_lines,      HELIUM_LINES,      "#ffd060"),
            (self.v_atmospheric_lines, ATMOSPHERIC_LINES, "#80d0ff"),
            (self.v_oxygen_lines,      OXYGEN_LINES,      "#90ee90"),
            (self.v_carbon_lines,      CARBON_LINES,      "#ffb347"),
            (self.v_calcium_lines,     CALCIUM_LINES,     "#45e0c8"),
            (self.v_carbon_star_lines, CARBON_STAR_LINES, "#ff9de2"),
            (self.v_herbig_lines,      HERBIG_LINES,      "#b98cff"),
            (self.v_wr_wn_lines,       WR_WN_LINES,       "#7ecfff"),
            (self.v_wr_wc_lines,       WR_WC_LINES,       "#ffb870"),
        ]
        if occupied is None:
            occupied = []
        drawn = set()
        for var, lines, colour in groups:
            if var.get():
                plot_reference_lines(
                    ax, lines, p["dispersion"], y_max,
                    poly_coeffs=poly, n_pixels=n_pixels, colour=colour,
                    fontsize=fontsize, occupied=occupied,
                )
                drawn.update(lines)
        return drawn

    def _refresh_full_spec_dialog(self):
        """
        Re-render the detached full-spectrum window if it is currently open.

        The dialog reads the parent's live reference-line BooleanVars and
        cached spectrum arrays, so a plain refresh() picks up any toggle
        change.  No-op when the window was never opened or has been closed.
        """
        dlg = getattr(self, "_full_spec_dialog", None)
        if dlg is not None and dlg.winfo_exists():
            dlg.refresh()

    def _on_ref_lines_toggle(self):
        """
        Show/hide reference-line groups on both spectrum panels without a
        full pipeline re-run.  Triggered by any of the line-group checkboxes.

        Raw panel  — fully redrawn from self.column_sums (fast, no I/O).
        Cal panel  — redrawn from self._calibrated_wls / self._calibrated_flux cache
                     (set by _display_extraction); also fast, no I/O.
        """
        p = self._last_p
        if p is None or self.column_sums is None:
            self.canvas_spec.draw()
            return

        # ── Raw panel ────────────────────────────────────────────────
        self.ax_raw.cla()
        self.ax_raw.set_facecolor("#0f0f1a")
        self._label_axes()
        col_sums = self.column_sums
        self._draw_raw_panel(p, col_sums)

        # ── Cal panel ────────────────────────────────────────────────
        if self._calibrated_wls is not None and self._calibrated_flux is not None:
            wls = self._calibrated_wls
            norm_int = self._calibrated_flux
            poly = self._validate_dispersion_poly(
                self.get_dispersion_poly(), len(col_sums))
            self.ax_cal.cla()
            self.ax_cal.set_facecolor("#0f0f1a")
            self._label_axes()
            self._draw_cal_panel(p, wls, norm_int, poly)
            self._redraw_nodes()

        # No explicit full-spectrum refresh: _draw_cal_panel already ends
        # with the debounced _sync_full_spectrum, and without calibrated
        # caches the detached window has nothing to show.
        self.canvas_spec.draw()

    def _validate_dispersion_poly(self, poly, n_pixels):
        """
        Return ``poly`` unchanged if the implied wavelength mapping is
        strictly monotonic over the spectrum's pixel range, else log a
        warning and return None (forcing callers to fall back to the linear
        dispersion).

        Thin logging wrapper around spectrum_core.validate_dispersion_poly
        — kept as a method because it is part of the dialogs' parent-API
        contract (full_spectrum_viewer calls it).
        """
        validated, n_bad = validate_dispersion_poly(poly, n_pixels)
        if n_bad:
            # Warn once per bad-fit episode (§3.3): this runs from both the
            # extraction path and the toggle redraw, so an unlatched log
            # turns a single bad fit into a stream.
            if not self._warned_nonmono:
                self._warned_nonmono = True
                self._log(
                    f"Dispersion fit is non-monotonic over {n_bad} pixel(s); "
                    f"falling back to linear dispersion. "
                    f"Add or move calibration nodes to improve the fit.")
        else:
            self._warned_nonmono = False
        return validated

    def get_dispersion_poly(self):
        """
        Return polynomial coefficients (highest degree first) for the
        non-linear dispersion fit if >= 2 nodes are available, else None.

        When the zero-order wavelength anchor is active (toggle on, and
        both the calibration-star and current-source anchor residuals are
        available), the returned polynomial is recomposed so it expects
        ``pixel − Δ`` in place of ``pixel``, where

            Δ = current_source_anchor_resid − calibration_anchor_resid

        is the per-source shift (in strip pixels) of the zero-order peak
        relative to the calibration star.  This cancels the colour-
        dependent centroid offset that otherwise shifts the whole
        wavelength scale between sources.  For the calibration star itself
        Δ = 0, so its solution is untouched; with the toggle off, an
        unset residual, or a missing measurement, Δ = 0 as well, leaving
        the uncorrected solution.

        The stored ``dispersion_nodes`` are never mutated: the shift is a
        display/extraction-time correction expressed purely in the
        returned coefficients.  Fit + recomposition live in
        spectrum_core.fit_dispersion_poly (self-checked in
        test_dispersion_math.py); this method only supplies Δ.
        """
        return fit_dispersion_poly(self.dispersion_nodes,
                                   self._dispersion_anchor_shift())

    def _dispersion_anchor_shift(self):
        """
        Return Δ (strip pixels) to subtract from the pixel axis before
        evaluating the dispersion polynomial, or 0.0 when no correction
        applies.

        Δ = current_source_anchor_resid − calibration_anchor_resid

        Guards: toggle must be on, and both residuals must be present and
        finite.  Any miss → 0.0, i.e. no correction.
        """
        try:
            if not self.v_zero_anchor.get():
                return 0.0
        except Exception:
            return 0.0
        cur = self._current_anchor_resid
        cal = self._calib_anchor_resid
        if cur is None or cal is None:
            return 0.0
        if not (np.isfinite(cur) and np.isfinite(cal)):
            return 0.0
        return float(cur - cal)

    def _on_contam_mask_toggle(self):
        """
        Re-run extraction with/without contaminator masking.  Cannot use
        the cached-redraw path (_on_ref_lines_toggle) because this toggle
        changes the data being plotted, not just an overlay.
        """
        if self._last_p is None or self._last_source_xy is None:
            return
        sx, sy = self._last_source_xy
        self._display_extraction(sx, self._applied_y(sy), self._last_p)
        self._redraw_nodes()
        self.canvas_spec.draw()

    def _on_zero_anchor_toggle(self):
        """
        Checkbox callback for the zero-order wavelength anchor.  The toggle
        changes how get_dispersion_poly maps pixels to wavelengths, so a
        full re-extraction of the current source is needed (not just a node
        redraw).  No-op until a source has been extracted.
        """
        if self._last_p is None or self._last_source_xy is None:
            return
        sx, sy = self._last_source_xy
        self._display_extraction(sx, self._applied_y(sy), self._last_p)
        self._redraw_nodes()
        # The extraction path's _draw_cal_panel already schedules the
        # debounced full-spectrum sync — no explicit refresh needed.
        self.canvas_spec.draw()
        state = "on" if self.v_zero_anchor.get() else "off"
        self._log(f"Zero-order wavelength anchor {state}.")

    def _on_cal_lines_toggle(self):
        """Checkbox callback — redraw nodes with or without calibration lines."""
        self._redraw_nodes()
        self.canvas_spec.draw()
        self._refresh_full_spec_dialog()

    def _redraw_nodes(self):
        """Overlay calibration node markers on ax_cal and update fit info."""
        # Remove old markers
        for artist in self._node_markers:
            try:
                artist.remove()
            except Exception:
                pass
        self._node_markers = []

        if not self.dispersion_nodes:
            self.fit_info_var.set("No nodes yet.")
            return

        try:
            dispersion = float(self.v_dispersion.get())
        except ValueError:
            dispersion = DEFAULTS["DISPERSION"]

        nodes = np.array(self.dispersion_nodes)  # shape (N, 2): [pixel, wl]
        pixels = nodes[:, 0]
        wls    = nodes[:, 1]

        # x-position on ax_cal: when the polynomial fit is active (≥ 2 nodes),
        # the calibrated panel uses poly wavelengths, so a node naturally sits
        # at its assigned wavelength.  With < 2 nodes — or a non-monotonic
        # fit, where the panel falls back to the linear mapping — markers use
        # the same linear estimate pixel * dispersion, so they sit where the
        # panel actually draws.  The silent core validator is used here (not
        # _validate_dispersion_poly) because the extraction path already logs
        # the fallback.
        col_sums = getattr(self, "column_sums", None)
        n_pix = (len(col_sums) if col_sums is not None
                 else int(pixels.max()) + 1)
        poly, _ = validate_dispersion_poly(self.get_dispersion_poly(), n_pix)
        if poly is not None:
            x_positions = wls   # by construction of the fit
        else:
            x_positions = pixels * dispersion

        # Draw vertical dashed lines + labels
        ylims = self.ax_cal.get_ylim()
        yspan = ylims[1] - ylims[0]

        show = getattr(self, "v_cal_lines", None)
        show = show.get() if show is not None else True
        for xp, wl in zip(x_positions, wls):
            if show:
                vl = self.ax_cal.axvline(x=xp, color="#f0c040", linestyle=":",
                                         linewidth=1.0, alpha=0.9)
                txt = self.ax_cal.text(xp, ylims[1] - 0.05 * yspan, f"{wl:.1f}",
                                       rotation=90, va="top", ha="right",
                                       color="#f0c040", fontsize=6)
                self._node_markers.extend([vl, txt])

        # Fit-quality label — the maths lives in
        # spectrum_core.dispersion_fit_stats; this block only formats.
        # The monotonicity in the stats uses the same full-pixel-span check
        # as the panel's fallback decision, so the label always agrees with
        # what is drawn.
        if len(self.dispersion_nodes) >= 2:
            try:
                stats = dispersion_fit_stats(self.dispersion_nodes,
                                             n_pixels=n_pix)
                rms_str = ("exact (N ≤ deg+1)" if stats["exact"]
                           else f"{stats['rms']:.2f} Å")
                status_str = ("Non-linear fit active"
                              if stats["monotonic"]
                              else "Fallback: linear dispersion "
                                   "(fit non-monotonic)")

                self.fit_info_var.set(
                    f"{len(wls)} nodes  |  poly deg {stats['deg']}\n"
                    f"RMS: {rms_str}\n"
                    f"Disp: {stats['disp_min']:.2f}–"
                    f"{stats['disp_max']:.2f} Å/px\n"
                    f"{status_str}")
            except Exception as e:
                self.fit_info_var.set(f"Fit error: {e}")
        else:
            self.fit_info_var.set(
                f"{len(self.dispersion_nodes)} node — need ≥ 2 for a fit.\n"
                f"Using linear dispersion.")

    def rebuild_after_node_change(self):
        """
        Common tail for any action that mutates self.dispersion_nodes:
        mark the config dirty, redraw the calibrated spectrum (its
        wavelength axis depends on the polynomial fit) and the node
        markers, then push to the canvas.

        Persistence to disk is now handled exclusively by Save config…
        Mutations driven by _load_config bypass the dirty mark because
        _suppress_dirty is set around that path.
        """
        self._mark_dirty()

        # Capture the calibration star's zero-order anchor residual.  Node
        # mutations always happen while the calibration source is the
        # active extraction, so _current_anchor_resid is that star's
        # residual.  Storing it now lets get_dispersion_poly compute Δ for
        # every other source.  Cleared when no nodes remain.
        #
        # Skip while _suppress_dirty is set: that flag marks a config load,
        # during which _calib_anchor_resid has already been restored from
        # disk and must not be overwritten by the current source's value.
        if not self._suppress_dirty:
            if self.dispersion_nodes:
                self._calib_anchor_resid = self._current_anchor_resid
            else:
                self._calib_anchor_resid = None

        if self._last_p is not None and self._last_source_xy is not None:
            sx, sy = self._last_source_xy
            self._display_extraction(sx, self._applied_y(sy), self._last_p)
        self._redraw_nodes()
        self.canvas_spec.draw()


if __name__ == "__main__":
    # Runs alongside the UI build, so the first catalogue browser opens as
    # fast as the later ones.
    catalog_browser.warm_coordinates()
    app = SpectrumExplorer()
    app.mainloop()
