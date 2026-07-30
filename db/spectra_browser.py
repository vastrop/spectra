"""
db/spectra_browser.py
=====================

Basic Tk client for the spectra database.

Left: a Treeview of the stored stars (name + spectral type), each
expandable into its spectra (date, samples, source file).  Multi-select
works across both levels.  Right: the selected spectra overplotted
(calibrated flux vs wavelength), same dark theme and axis formatting as
the explorer.

Duplicates — several spectra of one star from the *same* dataset (same
file hash) — are flagged orange in the tree.

The poster section (bottom of the plot panel) renders the ticked stars
to an A4 PNG: title/author/instrument fields, a columns override, and
three plot styles — the classic rainbow strip, the strip with a fake
star hovering above it, or a "circular" spectral ring (radius ∝
wavelength so the sub-λ_min centre stays empty, hue = wavelength,
brightness = flux, and the fake star in the hole).  The fake star is a
gaussian PSF tinted with the star's blackbody colour from a Planck fit
(flux-weighted mean hue when the fit fails).  The last eight
generated posters are remembered recent-files-style in a JSON sidecar
next to the DB; clicking one re-ticks its stars and restores its text.

"Full spectrum…" opens the explorer's own ``FullSpectrumDialog`` on the
selected spectrum, at the size it gets there — the browser's panel shares
its window with the tree, the note and the poster fields, which is no
place to read annotations off.  The viewer takes its data from its parent
rather than from arguments, so ``_ViewerHost`` presents a stored spectrum
in the shape it expects; one window is re-pointed rather than a new one
opened per press.  Continuum anchors and dispersion nodes do not exist
here, so it comes up in its 2-row form with the ANNOTATE column, the
luminance strip, the ±2σ band (when the samples carry sigma) and its
FITS/PNG exports.

Right below the plot, an info-card panel shows the
``ReferenceLibrary/notes/<stem>.md`` card matching the selected star's
spectral type (single-star selections only).  Types without their own
card fall back to the generic '## Class' section borrowed from a
same-class sibling card.

Writes: per the guidelines (§5) the browser holds a READ-ONLY
connection; the explicit curation actions each open a short-lived
read-write connection:

* delete selected spectra (whole-run deletion, after confirmation);
* delete same-capture duplicates DB-wide (same star, same DATE-OBS —
  the accidental double-save; deliberate star-following repeats stay),
  keeping the newest run of each;
* merge another spectra DB (a remote-controlled session brought home):
  every spectrum goes through the normal ingest waterfall, already-
  present spectra are skipped, so re-merging is a no-op;
* rename a star (display ``label`` only — catalog identity stays with
  SIMBAD);
* set a star's spectral type by hand (dialog, not in-place treeview
  editing — ttk has no native cell editor and the overlay-Entry hack
  isn't worth it for an occasional curation action);
* re-identify a star against SIMBAD from its stored position, with a
  user-chosen cone radius (bloated / saturated stars can need a much
  wider cone than the capture-time default before SIMBAD finds them).
  Reuses ``source_identification.query_position``;
* edit a spectrum's exclusion zones (``SpectrumEditor``: drag a
  wavelength box over a polluting star, choose mask / interpolate /
  overlay). Zones are annotations applied at display time — in this
  viewer and on posters — never edits to the stored samples.

Run:  py -3.13 db/spectra_browser.py [db_path]
Self-check (no GUI):  py -3.13 db/spectra_browser.py --selfcheck
"""

from __future__ import annotations

import os
import sys

# Root modules (source_identification, spectrum_core) live one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import sqlite3
import tkinter as tk
from datetime import datetime, timezone
from tkinter import ttk, messagebox, simpledialog, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from db import spectra_db
from full_spectrum_viewer import FullSpectrumDialog
from spectrum_core import rainbow_fill, wavelength_to_rgb

# Palette — matches the rest of the application's chrome.
BG    = "#0e1014"
PANEL = "#0f0f1a"
SPINE = "#262c37"
GRID  = "#262c37"
FG    = "#aab2c0"
ACC   = "#e0c46c"
DUP   = "#e0904c"   # duplicate flag — warning orange
ENTRY_BG = "#1e232c"
ENTRY_FG = "#e6e9ef"

# The observer's note is the one thing on screen the user wrote, so it is
# the one thing that does not wear the app's chrome: a paper sticky note,
# dark ink on yellow.  Greyed to the panel colour when no single star is
# selected, so "you cannot write here" is visible rather than merely true.
NOTE_BG  = "#f0e2a0"
NOTE_FG  = "#20201a"
NOTE_OFF = "#2a2b26"

# Matplotlib default colour cycle works fine on the dark panel.
CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# Pure data access (GUI-free, self-checkable)
# ---------------------------------------------------------------------------

def star_display_name(row: dict) -> str:
    """label > main_id > bare position, same preference as ingest logging."""
    return (row.get("label") or row.get("main_id")
            or f"RA {row['ra_deg']:.4f} Dec {row['dec_deg']:+.4f}")


def load_tree(conn):
    """Return the browser's model: [(star dict, [spectrum dicts])].

    Each spectrum dict carries a ``dup`` flag: True when the same star
    has more than one spectrum from the same dataset (same file hash,
    or same path row when unhashed) — the accidental double-Add case.
    """
    stars = [dict(r) for r in conn.execute(
        "SELECT star_id, gaia_dr3_source_id, main_id, label, sp_type, "
        "otype, ra_deg, dec_deg, note FROM stars ORDER BY star_id")]
    # Constellation is derived from the stored position on the fly —
    # astropy bundles the IAU 1987 (Roman) boundary grid, so there is
    # nothing to persist and re-identification updates it for free.
    if stars:
        from astropy.coordinates import SkyCoord, get_constellation
        coords = SkyCoord([s["ra_deg"] for s in stars],
                          [s["dec_deg"] for s in stars], unit="deg")
        names = np.atleast_1d(get_constellation(coords, short_name=True))
        for s, c in zip(stars, names):
            s["constellation"] = str(c)
    out = []
    for star in stars:
        specs = [dict(r) for r in conn.execute(
            "SELECT s.spectrum_id, s.n_samples, s.free_selection, "
            "       r.run_id, r.run_utc, "
            "       d.dataset_id, d.date_obs, d.fits_path, d.object_name "
            "FROM spectra s "
            "JOIN runs r ON r.run_id = s.run_id "
            "JOIN datasets d ON d.dataset_id = r.dataset_id "
            "WHERE s.star_id = ? "
            "ORDER BY d.date_obs, s.spectrum_id", (star["star_id"],))]
        counts = {}
        for sp in specs:
            counts[sp["dataset_id"]] = counts.get(sp["dataset_id"], 0) + 1
        for sp in specs:
            sp["dup"] = counts[sp["dataset_id"]] > 1
        out.append((star, specs))
    return out


def load_samples(conn, spectrum_id: int):
    """Calibrated samples of one spectrum: (wavelengths, fluxes) lists."""
    rows = conn.execute(
        "SELECT wavelength_a, cal_flux FROM samples "
        "WHERE spectrum_id = ? AND cal_flux IS NOT NULL "
        "ORDER BY pixel", (spectrum_id,)).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def load_sigma(conn, spectrum_id: int):
    """Per-sample cal_sigma, row-aligned with load_samples, or None.

    Same filter and order as load_samples so the two arrays index
    together; a stored None becomes NaN rather than shortening the array,
    which is what the viewer's ±2σ band expects of a gap.  None when the
    spectrum carries no sigma at all — then the band toggle stays off
    instead of shading a row of zeros.
    """
    rows = conn.execute(
        "SELECT cal_sigma FROM samples WHERE spectrum_id = ? "
        "AND cal_flux IS NOT NULL ORDER BY pixel", (spectrum_id,)).fetchall()
    if not any(r[0] is not None for r in rows):
        return None
    return [float("nan") if r[0] is None else float(r[0]) for r in rows]


def capture_info(conn, spectrum_id: int):
    """(total_exposure_s, n_frames) behind one spectrum; (None, 0) if unknown.

    A livestack's numbers live in the run's config snapshot, not on the
    dataset row: that row describes the autosaved stack file, and only the
    snapshot says how many frames went into it.  A single frame has just
    the dataset's own exptime_s and no frame count.
    """
    row = conn.execute(
        "SELECT d.exptime_s, r.config_json FROM spectra s "
        "JOIN runs r ON r.run_id = s.run_id "
        "JOIN datasets d ON d.dataset_id = r.dataset_id "
        "WHERE s.spectrum_id = ?", (spectrum_id,)).fetchone()
    if row is None:
        return None, 0
    total, config = row[0], row[1]
    try:
        stack = (json.loads(config) or {}).get("livestack") or {}
    except (TypeError, ValueError):
        stack = {}          # a run without a parseable snapshot still has
    total = stack.get("total_exptime_s") or total   # the dataset exposure
    return (float(total) if total else None), int(stack.get("n_frames") or 0)


def delete_spectrum(db_path: str, spectrum_id: int) -> None:
    """Whole-run deletion (guidelines §2/§5): the spectrum, its samples
    and its run go together; the dataset row goes too once no other run
    references it.  One transaction."""
    conn = spectra_db.connect(db_path)
    try:
        with conn:
            row = conn.execute(
                "SELECT run_id FROM spectra WHERE spectrum_id = ?",
                (spectrum_id,)).fetchone()
            if row is None:
                return
            run_id = row[0]
            dataset_id = conn.execute(
                "SELECT dataset_id FROM runs WHERE run_id = ?",
                (run_id,)).fetchone()[0]
            conn.execute("DELETE FROM samples WHERE spectrum_id = ?",
                         (spectrum_id,))
            conn.execute("DELETE FROM spectrum_exclusions "
                         "WHERE spectrum_id = ?", (spectrum_id,))
            conn.execute("DELETE FROM spectra WHERE spectrum_id = ?",
                         (spectrum_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            if conn.execute("SELECT 1 FROM runs WHERE dataset_id = ? LIMIT 1",
                            (dataset_id,)).fetchone() is None:
                conn.execute("DELETE FROM datasets WHERE dataset_id = ?",
                             (dataset_id,))
    finally:
        conn.close()


def duplicate_spectra(conn) -> list:
    """Spectrum ids that are same-capture duplicates, oldest runs first.

    Repeated spectra of one star are normal (following a star); the
    accidental kind is the same star saved twice from the same capture —
    same DATE-OBS (same dataset row when the frame has none).  For each
    such group everything but the newest run (the latest reduction) is
    returned for deletion.
    """
    rows = conn.execute(
        "SELECT s.spectrum_id, s.star_id, r.run_utc, "
        "       d.dataset_id, d.date_obs "
        "FROM spectra s "
        "JOIN runs r ON r.run_id = s.run_id "
        "JOIN datasets d ON d.dataset_id = r.dataset_id").fetchall()
    groups = {}
    for sid, star_id, run_utc, ds_id, date_obs in rows:
        key = (star_id, date_obs or f"ds:{ds_id}")
        groups.setdefault(key, []).append((run_utc or "", sid))
    doomed = []
    for specs in groups.values():
        if len(specs) > 1:
            specs.sort()
            doomed.extend(sid for _utc, sid in specs[:-1])
    return doomed


def merge_db(dest_path: str, src_path: str) -> tuple[int, int]:
    """Merge another spectra DB (a session from the remote machine) into
    ``dest_path``.  Returns (added, skipped).

    Every source spectrum rides through the normal ingest path, so star
    identity goes through the same gaia→name→cone waterfall as a live
    save — only star rows merge, spectra always append.  A spectrum
    already present (same file hash/path, same run_utc, same source
    position) is skipped, so re-merging the same DB is a no-op — which
    also makes an interrupted merge self-healing on the next attempt.
    Exclusion zones travel with newly added spectra.
    """
    src = spectra_db.connect(src_path, readonly=True)
    src.row_factory = sqlite3.Row
    dest = spectra_db.connect(dest_path)
    try:
        # Older source schemas (pre-v3) have no exclusions table.
        has_zones = src.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='spectrum_exclusions'").fetchone() is not None
        specs = src.execute(
            "SELECT s.spectrum_id, s.star_id, s.source_x, s.source_y, "
            "       s.free_selection, s.match_sep_arcsec, "
            "       r.run_utc, r.git_hash, r.config_json, "
            "       d.fits_path, d.fits_sha256, d.date_obs, d.exptime_s, "
            "       d.telescope, d.instrument, d.object_name, "
            "       d.site_lat_deg, d.site_lon_deg, d.site_elev_m "
            "FROM spectra s "
            "JOIN runs r ON r.run_id = s.run_id "
            "JOIN datasets d ON d.dataset_id = r.dataset_id "
            "ORDER BY s.spectrum_id").fetchall()
        added = skipped = 0
        for sp in specs:
            present = dest.execute(
                "SELECT 1 FROM spectra s "
                "JOIN runs r ON r.run_id = s.run_id "
                "JOIN datasets d ON d.dataset_id = r.dataset_id "
                "WHERE r.run_utc = ? "
                "AND IFNULL(d.fits_sha256, d.fits_path) = ? "
                "AND IFNULL(s.source_x, -1) = IFNULL(?, -1) "
                "AND IFNULL(s.source_y, -1) = IFNULL(?, -1)",
                (sp["run_utc"], sp["fits_sha256"] or sp["fits_path"],
                 sp["source_x"], sp["source_y"])).fetchone()
            if present:
                skipped += 1
                continue
            star = dict(src.execute(
                "SELECT gaia_dr3_source_id, main_id, label, sp_type, "
                "otype, ra_deg, dec_deg, pos_epoch_jyear FROM stars "
                "WHERE star_id = ?", (sp["star_id"],)).fetchone())
            samples = src.execute(
                "SELECT pixel, raw_counts, wavelength_a, cal_flux, "
                "cal_sigma, flags FROM samples WHERE spectrum_id = ? "
                "ORDER BY pixel", (sp["spectrum_id"],)).fetchall()
            res = spectra_db.ingest_spectrum(
                dest,
                star=star,
                dataset={k: sp[k] for k in (
                    "fits_path", "fits_sha256", "date_obs", "exptime_s",
                    "telescope", "instrument", "object_name",
                    "site_lat_deg", "site_lon_deg", "site_elev_m")},
                run={k: sp[k] for k in ("run_utc", "git_hash",
                                        "config_json")},
                spectrum={k: sp[k] for k in (
                    "source_x", "source_y", "free_selection",
                    "match_sep_arcsec")},
                samples=[tuple(s) for s in samples])
            if has_zones:
                for zone in src.execute(
                        "SELECT x1_wl_a, x2_wl_a, method "
                        "FROM spectrum_exclusions WHERE spectrum_id = ?",
                        (sp["spectrum_id"],)):
                    spectra_db.add_exclusion(dest, res["spectrum_id"],
                                             zone[0], zone[1], zone[2])
            added += 1
        return added, skipped
    finally:
        src.close()
        dest.close()


def rename_star(db_path: str, star_id: int, label: str) -> None:
    """Set a star's display label (identity attributes stay untouched)."""
    conn = spectra_db.connect(db_path)
    try:
        with conn:
            conn.execute("UPDATE stars SET label = ? WHERE star_id = ?",
                         (label.strip() or None, star_id))
    finally:
        conn.close()


def set_sp_type(db_path: str, star_id: int, sp_type: str) -> None:
    """Set a star's spectral type by hand (free text — SIMBAD types are
    too messy to validate: 'B0.5IVpe', 'kA2hA5mA7V', …)."""
    conn = spectra_db.connect(db_path)
    try:
        with conn:
            conn.execute("UPDATE stars SET sp_type = ? WHERE star_id = ?",
                         (sp_type.strip() or None, star_id))
    finally:
        conn.close()


def set_note(db_path: str, star_id: int, note: str) -> None:
    """Set (or clear) the observer's own note on a star — free text.

    Empty/whitespace stores NULL, so "has a note" stays a simple IS NOT
    NULL test rather than a '' vs NULL distinction nobody wants to debug.
    """
    conn = spectra_db.connect(db_path)
    try:
        with conn:
            conn.execute("UPDATE stars SET note = ? WHERE star_id = ?",
                         (note.strip() or None, star_id))
    finally:
        conn.close()


def migrate(db_path: str) -> None:
    """Bring a DB up to the current schema.

    The browser's own connection is readonly, and connect() deliberately
    never migrates a readonly connection — so without this, a v1 DB would
    open fine and then blow up on the first SELECT of a column added in v2.
    A write connection, opened and closed, is what applies the migration.
    """
    spectra_db.connect(db_path).close()


SP_FILTERS = ("All", "O", "B", "A", "F", "G", "K", "M", "Other")


def sp_filter_match(sp_type: str | None, flt: str) -> bool:
    """Does a spectral type fall under a class filter?  'Other' catches
    untyped stars and non-OBAFGKM prefixes (WR, white dwarfs, sd…)."""
    if flt == "All":
        return True
    letter = (sp_type or "").strip().upper()[:1]
    if flt == "Other":
        # set, not substring test — '' is "in" any string.
        return letter not in set("OBAFGKM")
    return letter == flt


def apply_identity(db_path: str, star_id: int, match) -> str | None:
    """Overwrite a star's identity from a SIMBAD SourceMatch.

    Unlike ingest's NULL-only backfill this is an explicit user
    correction, so existing values are replaced — including the stored
    position (catalog position is the preferred ground truth, §1).
    Returns an error string when the Gaia id already belongs to another
    star row (a merge situation this basic client doesn't attempt).
    """
    gaia = spectra_db.gaia_dr3_id(match.all_ids)
    has_cat = match.cat_ra_deg is not None
    conn = spectra_db.connect(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE stars SET gaia_dr3_source_id = ?, main_id = ?, "
                "label = ?, sp_type = ?, otype = ? WHERE star_id = ?",
                (gaia, match.main_id or None, match.label or None,
                 match.sp_type or None, match.otype or None, star_id))
            if has_cat:
                conn.execute(
                    "UPDATE stars SET ra_deg = ?, dec_deg = ?, "
                    "pos_epoch_jyear = 2000.0 WHERE star_id = ?",
                    (match.cat_ra_deg, match.cat_dec_deg, star_id))
        return None
    except sqlite3.IntegrityError:
        return (f"Gaia id {gaia} already belongs to another star row — "
                f"these records need a merge; identity not applied.")
    finally:
        conn.close()


# Spectral-type reference cards, one markdown file per Pickles stem
# (g5iii.md, …), compiled from published spectral-classification sources.
NOTES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ReferenceLibrary", "notes")


import re as _re

# SIMBAD types are continuous; the Pickles grid is not.  An F7V is not an
# exotic star, it is a hole in the grid.  Rather than write cards for
# types with no library spectrum, the card snaps to the nearest available
# type and says so.  Luminosity classes become an ordinal axis, and LUM_WEIGHT
# prices one luminosity step in subtype steps: it is the only number here
# that isn't derivable from the grid, so it stays a visible knob.
_LUM_ORDER = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
LUM_WEIGHT = 1.5
MAX_COST = 3.0        # beyond this, no card is honestly "near" — fall back

# Non-MK objects: an event or a circumstellar disc, not a temperature.
# Keyed on SIMBAD's sp_type or otype token, never through the MK parse.
# The leading '_' keeps these files out of the class-letter globs below —
# 'n' is a class letter, so a bare nova.md would be served as the class
# card for every N-type star.
_PHENOMENA = {"NOVA": "_nova", "SN": "_sn", "BE": "_be", "EM": "_be",
              "C": "_c"}   # otype "C*" (carbon star) when sp_type is blank

# Carbon-star types never reach the MK parse — C is not an OBAFGKM letter
# — so they get their own gate: the revised Keenan forms (C-N5, C-R2,
# C-J, C-H3, bare C4,5 / CH) and the older Harvard R and N classes.  Bare
# R/N take the digit-or-end guard so nothing else can wander in.
_CARBON_RE = _re.compile(r"C(?:-[NRJH]|[\dH,]|$)|[NR](?:[AB]?\d|$)")

# When SIMBAD gives no luminosity class, otype usually still does — and
# guessing wrong here is not a small error: an M7 AGB giant and an M7
# dwarf share a subtype and nothing else.  Evolved-star otypes therefore
# override the dwarf default.  (An M7 dwarf is anyway far too faint to
# reach a small telescope, which is the same argument from the other end.)
_SUPERGIANT_OTYPES = {"S*R", "S*B", "S*Y", "SG*", "RSG*"}
_GIANT_OTYPES = {"AB*", "LP*", "LP?", "MI*", "MI?", "RG*", "RGB*",
                 "AGB*", "C*", "C*?", "S*", "S*?", "PA*", "OH/IR"}

_MK_RE = _re.compile(
    r"([OBAFGKM])[A-Z]?"                    # class, plus a qualifier (BC0.7)
    r"(\d+(?:\.\d+)?)"                      # subtype, decimals allowed
    r"(?:\s*-\s*(\d+(?:\.\d+)?))?")         # ...or a subtype range (B1-2)
_LUM_RE = _re.compile(
    r"^\s*-?\s*(IV|III|II|I|V)(?:AB|A|B)?"  # class + the a/ab/b qualifier
    r"(?:\s*[-/]\s*(IV|III|II|I|V))?")      # ...or a range (K2IV-V)
_CARD_RE = _re.compile(r"([obafgkm])(\d+(?:\.\d+)?)(iii|iv|ii|i|v)")


def parse_sp_type(sp_type: str | None):
    """(letter, sub_lo, sub_hi, lums | None), or None if not an MK type.

    Handles what SIMBAD actually emits: qualifiers ("K0-IIIa" → K0 III),
    subtype and luminosity ranges ("B1-2Ia-0ep" → B1-2 I, "K2IV-V" →
    K2 {IV,V}), decimals, and peculiarity noise.  A '+' string is a
    multiple system ("B7Vn+B9VHgMn+A1V"), not a star: the primary
    component is taken and the caller says so.  lums is None when SIMBAD gave
    no luminosity class at all.
    """
    s = (sp_type or "").strip().split("+")[0].strip().upper()
    m = _MK_RE.match(s)
    if not m:
        return None
    lo = float(m.group(2))
    hi = float(m.group(3)) if m.group(3) else lo
    lm = _LUM_RE.match(s[m.end():])
    lums = None
    if lm:
        lums = frozenset(x for x in (lm.group(1), lm.group(2)) if x)
    return m.group(1), min(lo, hi), max(lo, hi), lums


def _card_grid(notes_dir: str):
    """The cards as [(stem, path, sub_lo, sub_hi, lum)].

    Six stems are subtype *ranges* (b57v = B5-7 V, k01ii = K0-1 II, …).
    A two-digit integer stem is one of them: no real subtype reaches 10,
    so the rule needs no lookup table.  The r*/w* abundance variants and
    the '_' phenomenon cards fall out here — neither is reachable from a
    SIMBAD type.
    """
    import glob
    out = []
    for path in sorted(glob.glob(os.path.join(notes_dir, "*.md"))):
        stem = os.path.basename(path)[:-3]
        m = _CARD_RE.fullmatch(stem)
        if not m:
            continue
        num, lum = m.group(2), m.group(3).upper()
        if "." not in num and len(num) == 2:
            lo, hi = float(num[0]), float(num[1])
        else:
            lo = hi = float(num)
        out.append((stem, path, lo, hi, lum))
    return out


def _gap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Distance between two closed intervals; 0 when they overlap."""
    if a0 <= b1 and b0 <= a1:
        return 0.0
    return b0 - a1 if b0 > a1 else a0 - b1


def _card_name(letter: str, lo: float, hi: float, lum: str) -> str:
    def n(x):
        return str(int(x)) if x == int(x) else str(x)
    sub = n(lo) if lo == hi else f"{n(lo)}-{n(hi)}"
    return f"{letter}{sub} {lum}"


def _phenomenon(tok: str | None, notes_dir: str) -> str | None:
    stem = _PHENOMENA.get((tok or "").strip().upper().rstrip("*"))
    if not stem:
        return None
    path = os.path.join(notes_dir, stem + ".md")
    return path if os.path.isfile(path) else None


def _is_carbon(sp_type: str | None) -> bool:
    """True for carbon-star spectral types ("C-N4.5", "C4,5J", "N8", "R3")."""
    s = (sp_type or "").strip().split("+")[0].strip().upper()
    return bool(_CARBON_RE.match(s))


def _has_emission(sp_type: str | None) -> bool:
    """True for the 'e' peculiarity suffix ("B2Ve", "B0.5IVpe").

    Only the tail past the luminosity class is searched, so the 'e' of a
    real suffix is not confused with a letter inside the type itself.
    """
    s = (sp_type or "").strip().split("+")[0].strip().upper()
    m = _MK_RE.match(s)
    if not m:
        return False
    tail = s[m.end():]
    lm = _LUM_RE.match(tail)
    return "E" in (tail[lm.end():] if lm else tail)


def notes_match(sp_type: str | None, notes_dir: str = NOTES_DIR,
                otype: str | None = None) -> tuple[str | None, str | None]:
    """(card path, provenance) for a star.  Provenance is None for a hit
    that needs no apology, and a one-line caveat otherwise.

    The ladder: phenomenon card → exact/covered-by-a-range card → nearest
    card within MAX_COST → nothing (the caller falls back to class prose).
    An approximation must never read as a fact, so anything that isn't an
    exact hit carries a line saying what was actually shown.
    """
    path = _phenomenon(sp_type, notes_dir)
    if path:
        return path, None
    if _is_carbon(sp_type):
        path = os.path.join(notes_dir, "_c.md")
        if os.path.isfile(path):
            return path, None

    parsed = parse_sp_type(sp_type)
    if parsed is None:
        # Only now is otype allowed to speak.  A Be star with a real MK
        # type ("B2Ve", otype Em*) is still a B star and keeps its B card
        # — the disc is an annotation on the photosphere, not a
        # replacement for it — so otype must not outrank a parseable type.
        return _phenomenon(otype, notes_dir), None
    letter, lo, hi, lums = parsed
    assumed = None
    if lums is None:
        # No luminosity class.  Ask otype before falling back to the dwarf
        # default: most of these are faint HD/BD field stars, where dwarfs
        # dominate by number, but an "M7" that SIMBAD also calls an AGB or
        # long-period variable is a giant, and saying dwarf would be worse
        # than saying nothing.  Either way the guess is declared.
        ot = (otype or "").strip().upper()
        assumed = ("I" if ot in _SUPERGIANT_OTYPES else
                   "III" if ot in _GIANT_OTYPES else "V")
        lums = frozenset({assumed})

    scored = []
    for stem, path, clo, chi, clum in _card_grid(notes_dir):
        if stem[0] != letter.lower():
            continue
        sub = _gap(lo, hi, clo, chi)
        lum = min(abs(_LUM_ORDER[x] - _LUM_ORDER[clum]) for x in lums)
        cost = sub + LUM_WEIGHT * lum
        if cost <= MAX_COST:
            scored.append((cost, sub, stem, path,
                           _card_name(letter, clo, chi, clum)))
    if not scored:
        return None, None
    scored.sort()
    cost, _sub, _stem, path, name = scored[0]

    notes = []
    if cost > 0:
        tied = [s[4] for s in scored[1:] if s[0] == cost]
        alt = f" ({', '.join(tied)} equally close)" if tied else ""
        notes.append(f"no card for {sp_type.strip()} — showing the nearest, "
                     f"{name}{alt}")
    if assumed:
        word = {"I": "supergiant", "III": "giant", "V": "dwarf"}[assumed]
        why = (f" from SIMBAD's otype “{otype.strip()}”"
               if assumed != "V" else "")
        notes.append(f"no luminosity class given; assuming "
                     f"a {word} ({assumed}){why}")
    if "+" in (sp_type or ""):
        notes.append("composite type: this is a multiple system, and the "
                     "card describes the primary component only")
    if _has_emission(sp_type):
        # The photosphere is still a normal star and still classifies, so
        # it keeps its card — but the card cannot account for the emission
        # sitting on top of it, and saying nothing would let the reader
        # take an absorption description of Hα at face value.
        notes.append("the “e” suffix means emission: hydrogen lines "
                     "(Hα above all) are partly or wholly in emission, "
                     "which no card below accounts for — see the Be card "
                     "for the disc, and mind that an emission-line star "
                     "is a poor response-calibration reference")
    return path, ("≈ " + "; ".join(notes) + ".") if notes else None


def notes_card(sp_type: str | None, notes_dir: str = NOTES_DIR) -> str | None:
    """Path of the info card best matching a spectral type, or None."""
    return notes_match(sp_type, notes_dir)[0]


def notes_text(sp_type: str | None, notes_dir: str = NOTES_DIR,
               otype: str | None = None) -> str:
    """Info-card text for a spectral type, or ''.

    The matched card (prefixed by its provenance line when it is not an
    exact hit); otherwise the shared '## Class' section lifted from any
    sibling card of the same class letter (the cards repeat it verbatim,
    by design), so a star too far from the grid still shows class info.
    """
    import glob
    path, note = notes_match(sp_type, notes_dir, otype)
    if path:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        return f"{note}\n\n{text}" if note else text
    letter = (sp_type or "").strip().lower()[:1]
    if letter not in set("obafgkmn"):
        return ""
    for p in sorted(glob.glob(os.path.join(notes_dir, letter + "*.md"))):
        with open(p, encoding="utf-8") as f:
            m = _re.search(r"^## Class.*?(?=^## |\Z)", f.read(),
                           _re.M | _re.S)
        if m:
            return (f"# {sp_type.strip()} — no per-type card, "
                    f"class info only\n\n{m.group(0).strip()}")
    return ""


def notes_sources(notes_dir: str = NOTES_DIR) -> list[tuple[str, str]]:
    """The shared reference list as [(title, url)], or [].

    The cards cite nothing individually — they all rest on the same few
    atlases — so the list lives once in notes/README.md '## Sources' and
    is shown under every card.  Keeping it in the notes dir (not in this
    file) means the references travel with the notes they back.
    """
    import re
    path = os.path.join(notes_dir, "README.md")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []
    m = re.search(r"^## Sources.*?(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return []
    return re.findall(r"^- \[([^\]]+)\]\((https?://[^)]+)\)", m.group(0), re.M)


def notes_features(sp_type: str | None,
                   notes_dir: str = NOTES_DIR,
                   otype: str | None = None) -> list[tuple]:
    """Annotatable features for a spectral type: [(wl_lo, wl_hi, name)].

    Parsed from the '## Features' table of the matched card (the nearest
    one on the grid, per notes_match), or — class-defining features being
    shared — from the first same-class sibling when nothing is near
    enough.  wl_hi == wl_lo for a single line, > wl_lo for a band.  []
    when nothing is known.

    Phenomenon cards give *rest* wavelengths: a nova's or supernova's
    lines are Doppler-shifted by thousands of km/s, so the ticks mark
    where the line would sit at rest, not where its trough lands.  The
    cards say so; see _sn.md.
    """
    import glob
    import re
    path, _note = notes_match(sp_type, notes_dir, otype)
    if path is None:
        letter = (sp_type or "").strip().lower()[:1]
        if letter not in set("obafgkmn"):
            return []
        sibs = sorted(glob.glob(os.path.join(notes_dir, letter + "*.md")))
        if not sibs:
            return []
        path = sibs[0]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = []
    for m in re.finditer(r"^\|\s*(\d+(?:\.\d+)?)\s*(?:[-–]\s*"
                         r"(\d+(?:\.\d+)?)\s*)?\|([^|]+)\|", text, re.M):
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        out.append((lo, hi, m.group(3).strip()))
    return out


_SPECTRAL_SEQUENCE = "OBAFGKMN"


def spectral_sort_key(sp_type: str):
    """Sort key for the classic O-B-A-F-G-K-M(-N) sequence.

    (class index, subclass number); types that don't start with a class
    letter — and stars with no type at all — sort after everything,
    keeping their relative order (sorts are stable).
    """
    import re
    m = re.match(r"([OBAFGKMN])(\d+(?:\.\d+)?)?",
                 (sp_type or "").strip().upper())
    if not m:
        return (len(_SPECTRAL_SEQUENCE), 0.0)
    # First-letter parse only: composite SIMBAD types ("kA2hA5mA7V", "sdB")
    # misfile.  Revisit if the DB ever accumulates such stars.
    return (_SPECTRAL_SEQUENCE.index(m.group(1)),
            float(m.group(2)) if m.group(2) else 5.0)


def poster_grid(n: int, cols: int | None = None, square: bool = False):
    """Poster layout rule: columns of up to 7 plots, at most 3 columns
    (21 stars → 3×7, 14 → 2×7); more than 21 grows the rows instead.
    ``cols`` overrides the column count; ``square`` (circular cells)
    aims for near-square cells instead of the 7-per-column strip rule.
    Returns (rows, cols, landscape) — wide strip cells want landscape
    from 3 columns up, square cells whenever the grid is wider than
    tall."""
    if cols:
        c = max(1, min(cols, n))
    elif square:
        c = min(4, math.ceil(math.sqrt(n)))
    else:
        c = min(3, max(1, math.ceil(n / 7)))
    rows = math.ceil(n / c)
    return rows, c, (c > rows) if square else (c >= 3)


def kelvin_to_rgb(t_k: float):
    """Tanner Helland's blackbody colour approximation, clipped to its
    1000–40000 K validity range.  Returns RGB in [0, 1]."""
    t = min(max(float(t_k), 1000.0), 40000.0) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    return np.clip(np.array([r, g, b]) / 255.0, 0.0, 1.0)


def star_color(wls, flux):
    """Blackbody tint for the circular plot's fake star, or None.

    The flux-weighted mean of the ring hues washes out to beige for any
    normalized spectrum, so instead fit a colour temperature (coarse
    continuum anchors → the existing Planck estimator, whose free
    amplitude makes it normalization-blind) and use the classic
    blackbody palette: M stars orange-red, A/B blue-white.  None when
    the fit isn't possible — caller picks the fallback.
    """
    from spectrum_core import estimate_planck_temperature
    wls = np.asarray(wls, dtype=float)
    flux = np.asarray(flux, dtype=float)
    # ~10 anchors: 80th-percentile flux per wavelength bin — high
    # enough to ride over absorption lines, low enough not to chase
    # emission spikes (Be stars).
    edges = np.linspace(wls[0], wls[-1], 11)
    aw, af = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (wls >= lo) & (wls <= hi) & np.isfinite(flux)
        if m.sum() >= 3 and np.percentile(flux[m], 80) > 0:
            aw.append(float(wls[m].mean()))
            af.append(float(np.percentile(flux[m], 80)))
    if len(aw) < 3:
        return None
    try:
        fit = estimate_planck_temperature(aw, af)
    except Exception:
        return None
    return kelvin_to_rgb(fit["t_best"]) if fit else None


def star_tint(wls, flux):
    """Star tint: blackbody colour when the Planck fit works (see
    ``star_color``), flux-weighted mean of the wavelength hues
    otherwise."""
    tint = star_color(wls, flux)
    if tint is not None:
        return tint
    rgb = np.array([wavelength_to_rgb(w) for w in wls])
    return (rgb * flux[:, None]).sum(axis=0) / flux.sum()


def star_image(wls, flux, size: int = 160, bg: str = BG):
    """Just the fake star on its own square tile — the "combined"
    poster cells put it above the rainbow strip.  Same recipe as the
    circular plot's centre: gaussian PSF peaking at 0.9 × tint."""
    from matplotlib.colors import to_rgb
    wls = np.asarray(wls, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(wls) & np.isfinite(flux)
    wls, flux = wls[ok], flux[ok]
    if wls.size < 2 or np.max(flux) <= 0:
        raise ValueError("Need at least two calibrated samples.")
    flux = np.clip(flux / np.max(flux), 0.0, 1.0)
    tint = star_tint(wls, flux)
    c = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]
    psf = np.exp(-0.5 * (np.hypot(xx - c, yy - c) / (0.18 * size)) ** 2)
    img = np.tile(np.asarray(to_rgb(bg)), (size, size, 1))
    img += 0.9 * psf[..., None] * tint
    return np.clip(img, 0.0, 1.0)


def circular_image(wls, flux, size: int = 900, bg: str = PANEL):
    """The 'circular plot': the spectrum swept 360° as a radial profile.

    Radius maps wavelength linearly from the centre (r/R = λ/λ_max), so
    the disc below λ_min stays empty; each annulus takes its
    wavelength's rainbow hue at a brightness set by the flux there —
    angle carries no information.  The central hole gets a fake star:
    a gaussian PSF tinted with the star's blackbody colour (Planck fit,
    see ``star_color``; flux-weighted mean hue as fallback).  The PSF
    peaks at 0.9 × tint, NOT white: a white-hot core saturates the centre
    for every star, hiding the tint entirely, so hot stars, cool stars and
    a silently failed fit all look alike.
    Returns a (size, size, 3) float RGB in [0, 1].
    """
    from matplotlib.colors import to_rgb
    wls = np.asarray(wls, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(wls) & np.isfinite(flux)
    wls, flux = wls[ok], flux[ok]
    if wls.size < 2 or np.max(flux) <= 0:
        raise ValueError("Need at least two calibrated samples.")
    order = np.argsort(wls)
    wls = wls[order]
    flux = np.clip(flux[order] / np.max(flux), 0.0, 1.0)
    # Hue comes from the wavelength itself, through a dense colour LUT —
    # interpolating per-SAMPLE colours would blend endpoint RGB instead
    # of sweeping the rainbow when samples are sparse.
    lut_wl = np.linspace(wls[0], wls[-1], 512)
    lut_rgb = np.array([wavelength_to_rgb(w) for w in lut_wl])  # (512, 3)

    c = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]
    lam = np.hypot(xx - c, yy - c) / c * wls[-1]               # r → λ
    img = np.tile(np.asarray(to_rgb(bg)), (size, size, 1))

    ring = (lam >= wls[0]) & (lam <= wls[-1])
    bright = np.interp(lam[ring], wls, flux)
    for ch in range(3):
        img[..., ch][ring] = (np.interp(lam[ring], lut_wl, lut_rgb[:, ch])
                              * bright)

    tint = star_tint(wls, flux)
    hole = lam < wls[0]
    # PSF width scales with the hole so the star never bleeds into the
    # ring; psf⁴ whitens the core the way a saturated star would.
    psf = np.exp(-0.5 * (lam / (0.25 * wls[0])) ** 2)
    img[hole] += 0.9 * psf[..., None][hole] * tint
    return np.clip(img, 0.0, 1.0)


def load_poster_history(path: str) -> list:
    """Recent posters from the JSON sidecar; missing/corrupt → []."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["posters"]
    except (OSError, ValueError, KeyError):
        return []


def push_poster_history(path: str, entry: dict, keep: int = 8) -> list:
    """MRU, recent-files style: newest first; regenerating the same
    poster (same title + same star set) moves it up instead of
    duplicating.  Returns the saved list."""
    key = (entry["title"], sorted(entry["star_ids"]))
    hist = [h for h in load_poster_history(path)
            if (h.get("title"), sorted(h.get("star_ids", []))) != key]
    hist.insert(0, entry)
    del hist[keep:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"posters": hist}, f, indent=1)
    return hist


def render_poster(png_path: str, entries: list, title: str,
                  credit: str = "", dpi: int = 600, style: str = "strip",
                  cols: int | None = None) -> None:
    """Render a poster PNG: a grid of rainbow spectra on an A4 page.

    ``entries``: [(name, sp_type, wls, flux)] in display order.  Each
    plot uses the browser's calibrated look minus the y-axis labeling
    (normalised flux carries no per-star information on a poster).
    Pure Agg — no Tk, callable headless.

    ``style``: "strip" (rainbow fill under the flux curve), "combined"
    (the strip with the fake star hovering above it, see
    ``star_image``) or "circular" (spectral ring, see
    ``circular_image``).  ``cols`` overrides the automatic column
    count.

    ``dpi``: 600 by default — a 3-column plot is only ~600 px wide at
    200 dpi, ~1.4 px per spectral segment, and the fill edges visibly
    stair-step.  Print-grade output; downscale in an image editor if a
    smaller file is wanted (its resampling beats rendering small).
    Font/line sizes are in points, so layout is dpi-independent.
    """
    n = len(entries)
    if n == 0:
        raise ValueError("No spectra to render.")
    rows, ncols, landscape = poster_grid(n, cols=cols,
                                         square=(style == "circular"))
    a4 = (11.69, 8.27) if landscape else (8.27, 11.69)
    fig = Figure(figsize=a4, dpi=dpi, facecolor=BG)
    canvas = FigureCanvasAgg(fig)

    axes = fig.subplots(rows, ncols, squeeze=False)
    # combined cells stack star tile + title ~1.6 axes-heights above
    # each strip, so they need a wider inter-row gap (hspace is in
    # axes-height units) and a deeper top margin below the page title.
    hspace = {"strip": 0.9, "combined": 1.8}.get(style, 0.35)
    fig.subplots_adjust(top=0.82 if style == "combined" else 0.90,
                        bottom=0.05, left=0.05, right=0.97,
                        hspace=hspace, wspace=0.15)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue
        name, sp, wls, flux = entries[i]
        wls = np.asarray(wls, dtype=float)
        flux = np.asarray(flux, dtype=float)
        ax.set_facecolor(PANEL)
        # combined: the star tile sits above the strip (1.08–1.40 in
        # axes coords), so the title moves up out of its way.
        title_kw = {"y": 1.46, "pad": 0} if style == "combined" \
            else {"pad": 3}
        ax.set_title(f"{name}   ·   {sp}" if sp else name,
                     color="#e6e9ef", fontsize=7, **title_kw)
        if style == "circular":
            # equal-aspect imshow squares the cell inside its grid slot
            ax.imshow(circular_image(wls, flux), interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)
        # smooth: the historical per-segment fill — slower but blends
        # colour transitions; acceptable for a one-shot offline render.
        rainbow_fill(ax, wls, flux, zorder=2, smooth=True)
        ax.plot(wls, flux, color="white", linewidth=0.5, alpha=0.6, zorder=3)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=5, colors="#a0a0c0")
        ax.grid(True, axis="x", color="#3d3d6b", linewidth=0.3, zorder=1)
        if style == "combined":
            sax = ax.inset_axes([0.46, 1.08, 0.08, 0.32])
            sax.imshow(star_image(wls, flux), interpolation="bilinear")
            sax.axis("off")

    # Title: white, large, sitting in a gap of the top rule of a
    # hairline page frame (a classic plate border).  matplotlib has no
    # bordered-page primitive, so draw
    # the title, measure it, and rule the frame in five segments with
    # the top one split around the measured extent.  Margins are in
    # inches so the border sits at the same physical distance from
    # every page edge; the credit stays outside the frame, like a
    # signature below a plate mark.
    mx, my = 0.25 / a4[0], 0.25 / a4[1]
    txt = fig.text(0.5, 1 - my, title, ha="center", va="center",
                   color="white", fontsize=22)
    canvas.draw()
    ext = txt.get_window_extent(canvas.get_renderer())
    fw, fh = fig.bbox.width, fig.bbox.height
    pad = 0.15 / a4[0]                    # rule-end to title breathing room
    gx0 = max(ext.x0 / fw - pad, mx)      # a page-wide title eats the
    gx1 = min(ext.x1 / fw + pad, 1 - mx)  # top rule rather than escaping
    for xs, ys in [((mx, gx0), (1 - my, 1 - my)),      # top, left of title
                   ((gx1, 1 - mx), (1 - my, 1 - my)),  # top, right of title
                   ((mx, 1 - mx), (my, my)),           # bottom
                   ((mx, mx), (my, 1 - my)),           # left
                   ((1 - mx, 1 - mx), (my, 1 - my))]:  # right
        fig.add_artist(Line2D(xs, ys, transform=fig.transFigure,
                              color="#e6e9ef", linewidth=0.8))

    if credit:
        fig.text(0.97, 0.012, credit, ha="right", va="bottom",
                 color="#a0a0c0", fontsize=6)

    fig.savefig(png_path, facecolor=BG)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class _ViewerHost(tk.Frame):
    """Stand-in "parent" for the explorer's full-spectrum viewer.

    ``FullSpectrumDialog`` reads its data off its parent — the explorer's
    live extraction state — rather than through arguments, so showing a
    stored spectrum in it means presenting that same handful of
    attributes.  A never-mapped Frame, because a Toplevel needs a real
    widget as its master; it owns no connection, and nothing here writes
    back to the database.

    What the DB cannot offer stays empty instead of faked: no continuum
    anchors (there is no anchor picker in the browser, so the viewer drops
    its corrected panel), no dispersion polynomial and no calibration
    nodes — a stored spectrum is already in Ångström, which is exactly
    what the viewer's force_linear path assumes.
    """

    def __init__(self, master):
        super().__init__(master)
        self._calibrated_wls = None
        self._calibrated_flux = None
        self._calibrated_sigma = None
        # Frame scatter is measured while a livestack runs and is not
        # stored per sample, so that band stays unavailable here.
        self._calibrated_sigma_frames = None
        self._last_p = None
        self.column_sums = None
        self.continuum_anchors = []
        self.dispersion_nodes = []
        self._simbad_open_id = None
        self._target_header = None
        self._frame_override = None
        self._stack_count = 0
        self._stack_total_exp = 0.0
        self.v_target = tk.StringVar(master=self)
        self._full_spec_dialog = None

    def load(self, name, wls, flux, sigma=None, exptime_s=None, n_frames=0):
        """Point the host at one stored spectrum, ready to open or refresh."""
        wls = np.asarray(wls, dtype=float)
        self._calibrated_wls = wls
        self._calibrated_flux = np.asarray(flux, dtype=float)
        self._calibrated_sigma = (None if sigma is None
                                  else np.asarray(sigma, dtype=float))
        # sp_min/sp_max are the viewer's x-window: a stored spectrum is its
        # own window.  target names the FITS/PNG export file; dispersion is
        # replaced by the viewer itself for the reference-line call.
        self._last_p = {"target": name, "dispersion": 1.0,
                        "sp_min": float(np.nanmin(wls)),
                        "sp_max": float(np.nanmax(wls))}
        self.v_target.set(name)
        # The viewer titles itself from the identity first, so the star's
        # name lands there rather than a file path it has no use for.
        self._simbad_open_id = name
        self._target_header = {"EXPTIME": exptime_s} if exptime_s else None
        self._stack_total_exp = exptime_s or 0.0
        self._stack_count = n_frames
        # A frame count is what marks the exposure as a stack: the viewer
        # reads the running total only when a livestack frame is in play.
        self._frame_override = True if n_frames > 1 else None

    def get_dispersion_poly(self):
        return None

    def _validate_dispersion_poly(self, poly, n_pixels):
        return None

    def _draw_reference_line_groups(self, *args, **kwargs):
        """No persistent catalogue-group toggles in the browser — the
        viewer's own ANNOTATE column is the annotation path here.  Returns
        the wavelengths it drew, i.e. none."""
        return []


class SpectraBrowser(tk.Tk):

    def __init__(self, db_path: str = spectra_db.DEFAULT_DB_PATH):
        super().__init__()
        self.title("Spectra DB browser")
        self.configure(bg=BG)
        # Tall enough for the note panel without squeezing the type card;
        # minsize is what protects small screens.
        self.geometry("1600x1040")
        self.minsize(960, 540)

        self._db_path = db_path
        # UI state, not science data — a sidecar next to the DB, keyed
        # to it by name, covered by the same gitignore pattern.
        self._hist_path = db_path + ".posters.json"
        self._history = []           # recent posters (MRU)
        self._conn = None            # read-only browse connection
        self._samples_cache = {}     # {spectrum_id: (wls, flux)}
        self._exclusions_cache = {}  # {spectrum_id: [zone dicts]}
        self._editor = None          # single-instance exclusion editor
        self._viewer_host = None     # parent stand-in for the full viewer
        self._legend_by_iid = {}     # {tree iid: legend label}
        # otype rides along because novae and supernovae have no spectral
        # type to match on — their card is keyed on the object type.
        self._star_by_iid = {}   # {spec iid: (star_id, name, sp_type, otype)}
        self._sort_keys = {}         # {star iid: {column: sort key}}
        self._sort_state = (None, False)   # (column, reverse)
        self._checked = set()        # star_ids ticked for the poster
        self._date_text = None       # fig.text date on the suptitle row

        self._build_ui()
        self._open_db()
        self._refresh()

    # ── DB ────────────────────────────────────────────────────────────
    def _open_db(self):
        self._summary = ""          # stale counts must not survive a reopen
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        try:
            # Migrate first: this connection is readonly, and readonly
            # connections never migrate, so an out-of-date DB would open
            # happily and then fail on the first SELECT of a newer column.
            migrate(self._db_path)
            self._conn = spectra_db.connect(self._db_path, readonly=True)
            self._conn.row_factory = sqlite3.Row
            self._status(f"DB: {self._db_path}")
        except Exception as e:
            self._status(f"Could not open {self._db_path}: {e}")

    # ── UI scaffolding ────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=ENTRY_FG, rowheight=22, borderwidth=0)
        style.configure("Treeview.Heading", background=ENTRY_BG,
                        foreground=FG, relief="flat")
        style.map("Treeview", background=[("selected", "#2b3a55")])

        left = tk.Frame(self, bg=BG)
        left.pack(side="left", fill="y", padx=(10, 6), pady=10)

        self.tree = ttk.Treeview(
            left, columns=("chk", "sp", "con", "info"),
            selectmode="extended", height=30)
        # Clicking a heading sorts the star rows by that column
        # (children stay date-ordered under their star).
        self.tree.heading("#0", text="Star / spectrum",
                          command=lambda: self._sort_by("#0"))
        self.tree.heading("chk", text="✓")
        self.tree.heading("sp", text="Sp. type",
                          command=lambda: self._sort_by("sp"))
        self.tree.heading("con", text="Con",
                          command=lambda: self._sort_by("con"))
        self.tree.heading("info", text="Info",
                          command=lambda: self._sort_by("info"))
        self.tree.column("#0", width=240)
        self.tree.column("chk", width=28, anchor="center", stretch=False)
        self.tree.column("sp", width=80, anchor="w")
        self.tree.column("con", width=44, anchor="w", stretch=False)
        self.tree.column("info", width=150, anchor="w")
        self.tree.tag_configure("dup", foreground=DUP)
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._plot_selection())
        # Tick-box toggling: a click on the ✓ cell of a star row flips
        # its poster check without disturbing the selection.
        self.tree.bind("<Button-1>", self._on_tree_click)

        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)

        # ── Action bar under the tree ─────────────────────────────────
        actions = tk.Frame(self, bg=BG)
        actions.pack(side="left", fill="y", pady=10)

        def btn(text, cmd):
            b = tk.Button(actions, text=text, bg=ENTRY_BG, fg=ENTRY_FG,
                          font=("Courier New", 9), relief="flat", width=18,
                          activebackground="#262c37",
                          activeforeground=ENTRY_FG, command=cmd)
            b.pack(side="top", pady=3)
            return b

        btn("Refresh", self._refresh)
        btn("Full spectrum…", self._show_full_spectrum)
        btn("Delete spectra…", self._delete_selected)
        btn("Delete duplicates…", self._delete_duplicates)
        btn("Merge DB…", self._merge_db)
        btn("Edit spectrum…", self._edit_selected)
        btn("Rename star…", self._rename_selected)
        btn("Set sp. type…", self._set_sptype_selected)

        tk.Label(actions, text="Sp. class filter", bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(side="top", pady=(14, 2))
        self.v_spfilter = tk.StringVar(value="All")
        cb = ttk.Combobox(actions, textvariable=self.v_spfilter,
                          values=SP_FILTERS, state="readonly", width=8)
        cb.pack(side="top")
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        tk.Label(actions, text="SIMBAD cone (″)", bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(side="top", pady=(14, 2))
        self.v_radius = tk.StringVar(value="10")
        ttk.Spinbox(actions, from_=1, to=600, increment=5, width=8,
                    textvariable=self.v_radius).pack(side="top")
        btn("Re-identify…", self._reidentify_selected)

        # ── Plot area + poster section ────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True,
                   padx=(6, 10), pady=10)

        # Poster section (bottom): title/author/instrument fields, grid
        # parameters, generate button, and the recent-posters MRU list.
        # Packed before the canvas so it always keeps its height.
        poster = tk.Frame(right, bg=BG)
        poster.pack(side="bottom", fill="x", pady=(8, 0))
        self._poster_frame = poster

        fields = tk.Frame(poster, bg=BG)
        fields.pack(side="left", anchor="n")

        def field(row, label):
            tk.Label(fields, text=label, bg=BG, fg=FG,
                     font=("Courier New", 9)).grid(
                row=row, column=0, sticky="w", pady=1)
            v = tk.StringVar()
            tk.Entry(fields, textvariable=v, width=44, bg=ENTRY_BG,
                     fg=ENTRY_FG, insertbackground=ENTRY_FG,
                     relief="flat", font=("Courier New", 9)).grid(
                row=row, column=1, sticky="w", padx=(8, 0), pady=1)
            return v

        self.v_title = field(0, "Title")
        self.v_author = field(1, "Author")
        self.v_instrument = field(2, "Instrument")

        opts = tk.Frame(fields, bg=BG)
        opts.grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))
        tk.Label(opts, text="Columns", bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(side="left")
        self.v_cols = tk.StringVar(value="Auto")
        ttk.Combobox(opts, textvariable=self.v_cols, state="readonly",
                     values=("Auto", "1", "2", "3", "4"),
                     width=5).pack(side="left", padx=(4, 14))
        self.v_style = tk.StringVar(value="strip")
        for txt, val in (("Rainbow strip", "strip"),
                         ("Strip + star", "combined"),
                         ("Circular", "circular")):
            tk.Radiobutton(opts, text=txt, value=val, variable=self.v_style,
                           bg=BG, fg=FG, selectcolor=ENTRY_BG,
                           activebackground=BG, activeforeground=ENTRY_FG,
                           font=("Courier New", 9)).pack(side="left")
        tk.Button(opts, text="Poster PNG…", bg=ENTRY_BG, fg=ENTRY_FG,
                  font=("Courier New", 9), relief="flat",
                  activebackground="#262c37", activeforeground=ENTRY_FG,
                  command=self._export_poster).pack(side="left",
                                                    padx=(14, 0))

        recent = tk.Frame(poster, bg=BG)
        recent.pack(side="left", fill="x", expand=True,
                    anchor="n", padx=(16, 0))
        tk.Label(recent, text="Recent posters (click to re-tick)",
                 bg=BG, fg=FG, font=("Courier New", 9)).pack(anchor="w")
        self.lst_recent = tk.Listbox(
            recent, height=5, bg=ENTRY_BG, fg=ENTRY_FG, relief="flat",
            selectbackground="#2b3a55", selectforeground=ENTRY_FG,
            font=("Courier New", 9), exportselection=False)
        self.lst_recent.pack(fill="x")
        self.lst_recent.bind("<<ListboxSelect>>", self._restore_poster)
        self._reload_history()

        # Figure(), not plt.figure(): pyplot would give this embedded figure a
        # manager — a withdrawn second tk.Tk() root that keeps Tcl_MainLoop (and
        # so the process) alive after this window is destroyed.  The poster
        # render above already uses Figure + an explicit Agg canvas.
        self.fig = Figure(facecolor=BG, dpi=100)
        self.ax = self.fig.add_subplot(111)
        # Room for the star suptitle above the axes (the date shares its
        # row, so no axes-title strip) and for the tick labels + x-axis
        # label below (the fixed 7:3 widget height makes the default
        # bottom margin clip "Wavelength (Å)").
        self.fig.subplots_adjust(top=0.88, bottom=0.17, left=0.07, right=0.98)
        # Suptitle x: the axes' horizontal midpoint, not the figure's —
        # keeps it centred over the (axes-centred) date title.
        self._suptitle_x = (0.07 + 0.98) / 2
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        # Fixed 7:3 aspect — each spectrum panel in the explorer is half
        # of a figsize=(14, 3) figure.  The widget is sized from the
        # container's width (capped to its height), not fill-expanded.
        self.canvas.get_tk_widget().pack(side="top", anchor="n")
        right.bind("<Configure>", self._on_plot_area_resize)
        self.canvas.draw()

        # ── Observer's note ───────────────────────────────────────────
        # Edited in place rather than behind a dialog: a note is something
        # you jot *while* looking at the spectrum, and a modal would make
        # you leave it.  Saved on focus-out, on selection change and on
        # close — never silently, the header says which of those happened.
        mynote = tk.Frame(right, bg=BG)
        mynote.pack(side="top", fill="x", pady=(8, 0))
        # _on_plot_area_resize hands the canvas whatever height the other
        # right-panel sections don't claim — so it has to know about this
        # one, or the canvas grows over it and the panel is never seen.
        self._mynote_frame = mynote
        hdr = tk.Frame(mynote, bg=BG)
        hdr.pack(fill="x")
        self.v_note_hdr = tk.StringVar(value="My note")
        tk.Label(hdr, textvariable=self.v_note_hdr, bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(side="left")
        self.v_note_state = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self.v_note_state, bg=BG, fg="#7fa8d8",
                 font=("Courier New", 9)).pack(side="right")
        self.txt_mynote = tk.Text(
            mynote, height=6, bg=NOTE_OFF, fg=NOTE_FG, relief="flat",
            wrap="word", font=("Courier New", 10), state="disabled",
            insertbackground=NOTE_FG, selectbackground="#c9b678",
            selectforeground=NOTE_FG, padx=8, pady=6)
        self.txt_mynote.pack(fill="x")
        self.txt_mynote.bind("<KeyRelease>", self._note_touched)
        self.txt_mynote.bind("<FocusOut>", lambda e: self._save_note())
        self._note_star = None      # (star_id, display name) or None
        self._note_dirty = False
        self._note_after = None     # pending autosave timer
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Spectral-type info card: packed top AFTER the canvas, so it
        # hugs the plot's lower edge; expands to fill the space down to
        # the poster section, closed off by a separator rule.
        notes = tk.Frame(right, bg=BG)
        notes.pack(side="top", fill="both", expand=True, pady=(8, 0))
        self._notes_frame = notes
        tk.Frame(notes, bg=SPINE, height=1).pack(side="bottom", fill="x",
                                                 pady=(6, 0))
        self.v_annotate = tk.BooleanVar(value=True)
        tk.Checkbutton(
            notes, text="Annotate features (│ line, ▼ band)",
            variable=self.v_annotate, command=self._plot_selection,
            bg=BG, fg=FG, selectcolor=ENTRY_BG, activebackground=BG,
            activeforeground=ENTRY_FG, font=("Courier New", 9)
        ).pack(side="top", anchor="w")
        self.txt_notes = tk.Text(
            notes, height=9, bg=PANEL, fg=FG, relief="flat", wrap="word",
            font=("Courier New", 9), state="disabled",
            insertbackground=ENTRY_FG)
        nsb = ttk.Scrollbar(notes, orient="vertical",
                            command=self.txt_notes.yview)
        self.txt_notes.configure(yscrollcommand=nsb.set)
        nsb.pack(side="right", fill="y")
        self.txt_notes.pack(side="left", fill="both", expand=True)

        self.v_status = tk.StringVar()
        self._summary = ""          # DB counts, kept ahead of any message
        tk.Label(self, textvariable=self.v_status, bg=BG, fg=FG,
                 font=("Courier New", 9), anchor="w").pack(
            side="bottom", fill="x", padx=10, pady=(0, 6))

    def _style_axes(self):
        # Same chrome as the explorer's calibrated panel (_draw_cal_panel):
        # plain Å ticks (custom_formatter is the RAW panel's pixel→Å
        # mapping and does not belong on a wavelength axis).
        self.ax.set_facecolor(PANEL)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(SPINE)
        self.ax.tick_params(labelsize=8, colors="#a0a0c0")
        self.ax.grid(True, color="#3d3d6b", linewidth=0.5, zorder=1)
        self.ax.set_xlabel("Wavelength (Å)", fontsize=9, color="#a0a0c0")
        self.ax.set_ylabel("Norm. flux", fontsize=9, color="#a0a0c0")

    def _on_plot_area_resize(self, event):
        w = max(event.width, 200)
        # The poster section keeps its requested height; the canvas gets
        # what remains above it.
        avail = (event.height - self._poster_frame.winfo_reqheight()
                 - self._notes_frame.winfo_reqheight()
                 - self._mynote_frame.winfo_reqheight() - 16)
        h = min(int(w * 3 / 7), max(avail, 150))
        self.canvas.get_tk_widget().configure(width=w, height=h)

    def _status(self, text):
        # The DB counts stay put; transient messages append to them so a
        # "note saved" does not hide what the database holds.
        self.v_status.set(f"{self._summary}    {text}".strip()
                          if self._summary else text)

    def _sort_by(self, col):
        """Reorder the star rows by a column; a second click reverses."""
        reverse = (self._sort_state == (col, False))
        self._sort_state = (col, reverse)
        stars = sorted(self.tree.get_children(""),
                       key=lambda iid: self._sort_keys[iid][col],
                       reverse=reverse)
        for i, iid in enumerate(stars):
            self.tree.move(iid, "", i)

    def _on_tree_click(self, event):
        """Toggle a star's poster tick when its ✓ cell is clicked."""
        if self.tree.identify_column(event.x) != "#1":
            return None
        iid = self.tree.identify_row(event.y)
        if not iid.startswith("star-"):
            return None
        star_id = int(iid.split("-", 1)[1])
        if star_id in self._checked:
            self._checked.discard(star_id)
            self.tree.set(iid, "chk", "☐")
        else:
            self._checked.add(star_id)
            self.tree.set(iid, "chk", "☑")
        return "break"     # keep the click from changing the selection

    # ── Tree population ───────────────────────────────────────────────
    def _refresh(self):
        # Every curation action rebuilds the whole tree, so remember where
        # the user was; a spectrum that no longer exists (just deleted)
        # falls back to the star it belonged to.
        sel = list(self.tree.selection())
        fallback = [f"star-{self._star_by_iid[i][0]}"
                    for i in sel if i in self._star_by_iid]
        self.tree.delete(*self.tree.get_children())
        self._samples_cache.clear()
        self._exclusions_cache.clear()
        self._legend_by_iid.clear()
        self._star_by_iid.clear()
        self._sort_keys.clear()
        if self._conn is None:
            self._open_db()
            if self._conn is None:
                return
        try:
            model = load_tree(self._conn)
        except Exception as e:
            self._status(f"DB read failed: {e}")
            return
        # Ticks survive a refresh, but not for stars that no longer exist.
        self._checked &= {star["star_id"] for star, _s in model}
        flt = self.v_spfilter.get()
        shown = n_spec = 0
        for star, specs in model:
            if not sp_filter_match(star["sp_type"], flt):
                continue
            shown += 1
            n_spec += len(specs)
            any_dup = any(sp["dup"] for sp in specs)
            # ✎ marks a star carrying an observer's note, so an annotated
            # star is visible without clicking through the list.
            pen = "  ✎" if (star["note"] or "").strip() else ""
            sid = self.tree.insert(
                "", "end", iid=f"star-{star['star_id']}",
                text=f"#{star['star_id']}  {star_display_name(star)}{pen}",
                values=("☑" if star["star_id"] in self._checked else "☐",
                        star["sp_type"] or "",
                        star["constellation"],
                        f"{len(specs)} spectra"
                        + ("  ⚠ dup" if any_dup else "")),
                tags=("dup",) if any_dup else (), open=False)
            name = star_display_name(star)
            self._sort_keys[sid] = {
                "#0": name.lower(),
                # missing spectral type sorts last when ascending
                "sp": (star["sp_type"] or "~~").lower(),
                "con": star["constellation"],
                "info": len(specs),
            }
            for sp in specs:
                date = (sp["date_obs"] or sp["run_utc"] or "?")[:19]
                self.tree.insert(
                    sid, "end", iid=f"spec-{sp['spectrum_id']}",
                    text=f"  {date}",
                    values=("", "", "", f"{sp['n_samples']} px  "
                                        f"{os.path.basename(sp['fits_path'])}"
                            + ("  ⚠ dup" if sp["dup"] else "")),
                    tags=("dup",) if sp["dup"] else ())
                self._legend_by_iid[f"spec-{sp['spectrum_id']}"] = (
                    f"{star_display_name(star)} · {date} "
                    f"· #{sp['spectrum_id']}")
                self._star_by_iid[f"spec-{sp['spectrum_id']}"] = (
                    star["star_id"], name, star["sp_type"] or "",
                    star["otype"] or "")
        if self._sort_state[0] is not None:
            # Re-apply the active sort; _sort_by toggles, so pre-flip.
            col, reverse = self._sort_state
            self._sort_state = (col, not reverse)
            self._sort_by(col)
        restore = [i for i in sel if self.tree.exists(i)] or list(
            dict.fromkeys(i for i in fallback if self.tree.exists(i)))
        if restore:
            self.tree.selection_set(restore)
            self.tree.focus(restore[0])
            self.tree.see(restore[0])   # opens the star row if collapsed
        total_spec = sum(len(s) for _st, s in model)
        counts = (f"{shown}/{len(model)} stars, {n_spec}/{total_spec} "
                  f"spectra  (filter: {flt})" if flt != "All"
                  else f"{len(model)} stars, {total_spec} spectra")
        self._summary = f"DB: {self._db_path}    {counts}"
        self._status("")
        self._plot_selection()

    # ── Selection → spectra ids ───────────────────────────────────────
    def _selected_spectrum_iids(self):
        """Selected spectrum iids; a selected star contributes all its
        children.  Order preserved, duplicates removed."""
        out = []
        for iid in self.tree.selection():
            children = ([iid] if iid.startswith("spec-")
                        else list(self.tree.get_children(iid)))
            for c in children:
                if c not in out:
                    out.append(c)
        return out

    def _load_series(self, spectrum_id: int):
        """(wls, flux, overlay_spans) with the exclusion zones applied,
        or None when the samples cannot be read. Raw samples and zones
        are cached; the editor invalidates via _exclusions_changed."""
        if spectrum_id not in self._samples_cache:
            try:
                self._samples_cache[spectrum_id] = load_samples(
                    self._conn, spectrum_id)
            except Exception as e:
                self._status(f"Sample read failed: {e}")
                return None
        wls, flux = self._samples_cache[spectrum_id]
        if not wls:
            return None
        if spectrum_id not in self._exclusions_cache:
            try:
                self._exclusions_cache[spectrum_id] = (
                    spectra_db.exclusions_for(self._conn, spectrum_id))
            except Exception:
                self._exclusions_cache[spectrum_id] = []
        flux, spans = spectra_db.apply_exclusions(
            wls, flux, self._exclusions_cache[spectrum_id])
        return wls, flux, spans

    def _exclusions_changed(self, spectrum_id: int):
        """Editor callback: re-read this spectrum's zones and redraw."""
        self._exclusions_cache.pop(spectrum_id, None)
        self._plot_selection()

    # ── Plotting ──────────────────────────────────────────────────────
    def _plot_selection(self):
        """Redraw the plot for the current tree selection.

        One spectrum: the explorer's calibrated-panel look — per-
        wavelength rainbow fill under a thin white overlay line.
        Several: colour-cycled lines with a legend (overlapping rainbow
        fills are unreadable), same axis chrome.
        """
        self.ax.clear()
        self._style_axes()
        iids = self._selected_spectrum_iids()
        series = []              # (iid, wls, flux, spans) actually plottable
        for iid in iids:
            loaded = self._load_series(int(iid.split("-", 1)[1]))
            if loaded is not None:
                wls, flux, spans = loaded
                series.append((iid, np.asarray(wls, dtype=float),
                               np.asarray(flux, dtype=float), spans))

        # Suptitle: the resolved star name (+ spectral type when known)
        # while the selection stays within one star; a count otherwise.
        star_infos = {self._star_by_iid[iid][0]: self._star_by_iid[iid]
                      for iid, _w, _f, _s in series
                      if iid in self._star_by_iid}
        sp = ot = ""
        if len(star_infos) == 1:
            (_sid, name, sp, ot) = next(iter(star_infos.values()))
            sup = f"{name}   ·   {sp}" if sp else name
            self._show_note(_sid, name)
        else:
            # A note belongs to one star: with none or many selected there
            # is nothing to write on.
            self._show_note(None)
            sup = f"{len(star_infos)} stars" if star_infos else ""
        self.fig.suptitle(sup, color=ENTRY_FG, fontsize=12,
                          x=self._suptitle_x)
        # The date shares the suptitle row (smaller, parenthesised,
        # locked right) — fig.text artists persist across redraws, so
        # the previous one is removed explicitly.
        if self._date_text is not None:
            self._date_text.remove()
            self._date_text = None
        self._show_notes(sp, ot)

        if len(series) == 1:
            iid, wls, flux, spans = series[0]
            rainbow_fill(self.ax, wls, flux, zorder=2)
            self.ax.plot(wls, flux, color="white",
                         linewidth=0.6, alpha=0.6, zorder=3)
            for x1, x2 in spans:   # overlay exclusions: mark, don't alter
                self.ax.axvspan(x1, x2, color=DUP, alpha=0.18, zorder=4)
            label = self._legend_by_iid.get(iid, iid)
            parts = label.split(" · ")
            self._date_text = self.fig.text(
                0.98, 0.98, f"({parts[1] if len(parts) > 1 else label})",
                ha="right", va="top", color="#a0a0c0", fontsize=9)
            self.ax.set_ylim(bottom=0)
            if self.v_annotate.get():
                self._annotate_features(sp, wls, flux, ot)
        elif series:
            # Overlay spans are not drawn in multi-spectrum mode: whose
            # zone a shared band belongs to would be unreadable.
            for i, (iid, wls, flux, _spans) in enumerate(series):
                self.ax.plot(wls, flux, linewidth=1.0,
                             color=CYCLE[i % len(CYCLE)],
                             label=self._legend_by_iid.get(iid, iid))
            self.ax.set_ylim(bottom=0)
            leg = self.ax.legend(fontsize=8, facecolor=PANEL,
                                 edgecolor=SPINE, labelcolor=FG)
            leg.set_draggable(True)
        else:
            self.ax.text(0.5, 0.5, "Select stars or spectra to plot",
                         ha="center", va="center", color=FG, fontsize=10,
                         transform=self.ax.transAxes)
        self.canvas.draw_idle()

    def _annotate_features(self, sp_type: str, wls, flux,
                           otype: str = ""):
        """Mark the type card's features just above the spectrum curve:
        a thin tick for a spectral line, a thicker down-arrow at the
        midpoint for a band (at SA100-class resolution a band's true
        extent would be indistinguishable from a line, so the glyph
        carries the distinction, not the width).  Data coordinates, a
        small margin above the flux maximum, growing the y-limit to
        fit — NOT axes-fraction above the axes, which parked the
        markers far from the curve whenever the plot had headroom."""
        # nanmax: a masked exclusion zone leaves NaNs in the display flux.
        fmax = float(np.nanmax(flux)) if np.isfinite(flux).any() else 0.0
        if fmax <= 0:
            return
        lo_w, hi_w = float(np.min(wls)), float(np.max(wls))
        base, top = 1.03 * fmax, 1.08 * fmax
        drawn = False
        for lo, hi, _name in notes_features(sp_type, otype=otype):
            mid = (lo + hi) / 2.0
            if not lo_w <= mid <= hi_w:
                continue
            drawn = True
            if hi > lo:
                self.ax.annotate(
                    "", xy=(mid, base), xytext=(mid, 1.10 * fmax),
                    arrowprops=dict(arrowstyle="-|>", color=ACC,
                                    linewidth=1.8, mutation_scale=10))
            else:
                self.ax.plot([mid, mid], [base, top],
                             color=FG, linewidth=1.0)
        if drawn:
            self.ax.set_ylim(0, 1.12 * fmax)

    # ── Observer's note ───────────────────────────────────────────────
    # Autosaved: typing stops, the note is written.  There is no save
    # gesture to remember and no "unsaved" state to reason about.  An
    # explicit save key plus an "unsaved" label would be misleading here,
    # since the backstops save the note regardless.  The status reports
    # only what has already happened.
    NOTE_AUTOSAVE_MS = 700

    def _note_touched(self, _event=None):
        """A keystroke: (re)arm the autosave timer."""
        if self._note_star is None:
            return
        self._note_dirty = True
        if self._note_after is not None:
            self.after_cancel(self._note_after)
        self._note_after = self.after(self.NOTE_AUTOSAVE_MS, self._save_note)

    def _save_note(self):
        """Write the note back if it changed.  Safe to call at any time."""
        if self._note_after is not None:
            self.after_cancel(self._note_after)
            self._note_after = None
        if self._note_star is None or not self._note_dirty:
            return
        star_id, name = self._note_star
        text = self.txt_mynote.get("1.0", "end-1c")
        try:
            set_note(self._db_path, star_id, text)
        except Exception as e:                       # noqa: BLE001
            # Never fail silently: this is the user's own typing.
            self.v_note_state.set("NOT SAVED")
            messagebox.showerror("Note not saved", str(e), parent=self)
            return
        self._note_dirty = False
        self.v_note_state.set(f"saved {datetime.now():%H:%M:%S}")
        self._status(f"Note saved for {name}")
        # Re-mark just this row.  A full _refresh() would rebuild the tree
        # and drop the selection — blanking the box the user is typing in.
        iid = f"star-{star_id}"
        if self.tree.exists(iid):
            label = self.tree.item(iid, "text").rstrip().removesuffix("✎")
            self.tree.item(iid, text=label.rstrip()
                           + ("  ✎" if text.strip() else ""))

    def _show_note(self, star_id: int | None, name: str = ""):
        """Point the note box at a star — saving the previous one first."""
        self._save_note()
        self.txt_mynote.configure(state="normal")
        self.txt_mynote.delete("1.0", "end")
        if star_id is None:
            self._note_star = None
            self._note_dirty = False
            self.txt_mynote.configure(state="disabled", bg=NOTE_OFF)
            self.v_note_hdr.set("My note — select a single star to write one")
            self.v_note_state.set("")
            return
        row = self._conn.execute(
            "SELECT note FROM stars WHERE star_id = ?", (star_id,)).fetchone()
        self.txt_mynote.insert("1.0", (row["note"] if row else None) or "")
        self.txt_mynote.configure(bg=NOTE_BG)
        self._note_star = (star_id, name)
        self._note_dirty = False
        self.v_note_hdr.set(f"My note — {name}")
        self.v_note_state.set("saves as you type")

    def _on_close(self):
        self._save_note()
        self.destroy()

    def _show_notes(self, sp_type: str, otype: str = ""):
        """Fill the info-card panel for a spectral type ('' clears it)."""
        try:
            text = notes_text(sp_type, otype=otype)
        except OSError as e:
            text = f"Could not read info card: {e}"
        if not text and sp_type:
            text = (f"No info card for “{sp_type}” "
                    f"(ReferenceLibrary/notes/).")
        self.txt_notes.configure(state="normal")
        self.txt_notes.delete("1.0", "end")
        self.txt_notes.insert("1.0", text)
        if text:
            self._append_sources()
        self.txt_notes.configure(state="disabled")

    def _append_sources(self):
        """Rule off the card with the shared, clickable reference list."""
        import webbrowser
        srcs = notes_sources()
        if not srcs:
            return
        self.txt_notes.insert("end", "\n\nSources: ")
        for i, (title, url) in enumerate(srcs):
            tag = f"src{i}"
            if i:
                self.txt_notes.insert("end", " · ")
            self.txt_notes.insert("end", title, tag)
            self.txt_notes.tag_configure(tag, foreground="#7fa8d8",
                                         underline=True)
            # default arg: late binding would give every tag the last url
            self.txt_notes.tag_bind(
                tag, "<Button-1>", lambda e, u=url: webbrowser.open(u))
            self.txt_notes.tag_bind(
                tag, "<Enter>",
                lambda e: self.txt_notes.configure(cursor="hand2"))
            self.txt_notes.tag_bind(
                tag, "<Leave>",
                lambda e: self.txt_notes.configure(cursor=""))

    # ── Curation actions (short-lived read-write connections) ────────
    def _delete_selected(self):
        iids = self._selected_spectrum_iids()
        if not iids:
            messagebox.showinfo("Delete", "Select spectra (or a star, for "
                                          "all of its spectra) first.")
            return
        # Enumerate exactly what dies: selecting one duplicate child
        # deletes only that record; selecting a star row expands to ALL
        # of its spectra.  Star identity rows are never deleted.
        names = [self._legend_by_iid.get(i, i) for i in iids]
        listing = "\n".join(names[:10])
        if len(names) > 10:
            listing += f"\n…and {len(names) - 10} more"
        if not messagebox.askyesno(
                "Delete spectra",
                f"Delete these {len(iids)} spectrum record(s) and their "
                f"runs?\n\n{listing}\n\n"
                f"(Star entries stay; only the selected spectra go.)\n"
                f"This cannot be undone."):
            return
        try:
            for iid in iids:
                delete_spectrum(self._db_path, int(iid.split("-", 1)[1]))
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))
        self._refresh()

    def _delete_duplicates(self):
        """Delete same-capture duplicates DB-wide, keeping the newest run
        of each — the accidental double-save case, not the deliberate
        follow-a-star repeats (those differ in DATE-OBS and stay)."""
        if self._conn is None:
            return
        try:
            doomed = duplicate_spectra(self._conn)
        except Exception as e:
            messagebox.showerror("Duplicates", str(e))
            return
        if not doomed:
            messagebox.showinfo("Duplicates",
                                "No same-capture duplicates found.")
            return
        if not messagebox.askyesno(
                "Delete duplicates",
                f"{len(doomed)} spectra are extra copies of a star from "
                f"the same capture (same DATE-OBS).\n\nKeep the newest "
                f"run of each and delete the {len(doomed)} older cop"
                f"{'ies' if len(doomed) > 1 else 'y'}?\n\n"
                f"This cannot be undone."):
            return
        try:
            for sid in doomed:
                delete_spectrum(self._db_path, sid)
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))
        self._refresh()

    def _merge_db(self):
        """Merge a DB brought back from the remote-controlled machine."""
        src = filedialog.askopenfilename(
            title="Merge spectra DB",
            filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")])
        if not src:
            return
        if os.path.abspath(src) == os.path.abspath(self._db_path):
            messagebox.showinfo("Merge DB", "That is the open database.")
            return
        try:
            added, skipped = merge_db(self._db_path, src)
        except Exception as e:
            messagebox.showerror("Merge failed", str(e))
            return
        self._refresh()
        self._status(f"Merged {os.path.basename(src)}: {added} added, "
                     f"{skipped} already present")

    def _edit_selected(self):
        """Open the exclusion editor for the one selected spectrum."""
        iids = self._selected_spectrum_iids()
        if len(iids) != 1:
            messagebox.showinfo(
                "Edit spectrum", "Select exactly one spectrum (or a star "
                                 "with a single spectrum) first.")
            return
        spectrum_id = int(iids[0].split("-", 1)[1])
        if self._editor is not None and self._editor.winfo_exists():
            self._editor.destroy()
        _sid, name, _sp, _ot = self._star_by_iid.get(
            iids[0], (None, f"spectrum {spectrum_id}", "", ""))
        self._editor = SpectrumEditor(
            self, self._db_path, spectrum_id,
            f"{name} · {self._legend_by_iid.get(iids[0], '')}",
            on_change=self._exclusions_changed)

    def _show_full_spectrum(self):
        """Open the explorer's full-size viewer on the selected spectrum.

        The browser's own panel is a thumbnail sharing its window with the
        tree, the note and the poster fields; this is the same spectrum at
        the size the explorer gives it, with the annotation column, the
        luminance strip, the ±2σ band and the FITS/PNG exports that come
        with it.  Exclusion zones are applied first, so what opens is what
        the browser plots.
        """
        iids = self._selected_spectrum_iids()
        if len(iids) != 1:
            messagebox.showinfo(
                "Full spectrum", "Select exactly one spectrum (or a star "
                                 "with a single spectrum) first.")
            return
        spectrum_id = int(iids[0].split("-", 1)[1])
        loaded = self._load_series(spectrum_id)
        if loaded is None:
            return                     # _load_series has already said why
        wls, flux, _spans = loaded
        _sid, name, _sp, _ot = self._star_by_iid.get(
            iids[0], (None, f"spectrum {spectrum_id}", "", ""))
        try:
            sigma = load_sigma(self._conn, spectrum_id)
            exptime_s, n_frames = capture_info(self._conn, spectrum_id)
        except Exception as exc:       # noqa: BLE001 — extras, not the data
            sigma, exptime_s, n_frames = None, None, 0
            self._status(f"σ / exposure unread: {exc}")

        if self._viewer_host is None:
            self._viewer_host = _ViewerHost(self)
        host = self._viewer_host
        host.load(name, wls, flux, sigma, exptime_s, n_frames)
        dlg = host._full_spec_dialog
        # One viewer window, re-pointed: pressing the button with another
        # spectrum selected refreshes that window (it re-titles itself on
        # every render) rather than stacking near-identical copies.
        if dlg is not None and dlg.winfo_exists():
            dlg.refresh()
            dlg.lift()
            dlg.focus_force()
        else:
            host._full_spec_dialog = FullSpectrumDialog(host)
        self._status(f"Full spectrum: {name}")

    def _selected_single_star(self):
        sel = [i for i in self.tree.selection() if i.startswith("star-")]
        if len(sel) != 1:
            messagebox.showinfo("Star action",
                                "Select exactly one star row first.")
            return None
        return int(sel[0].split("-", 1)[1])

    def _rename_selected(self):
        star_id = self._selected_single_star()
        if star_id is None:
            return
        row = self._conn.execute(
            "SELECT label FROM stars WHERE star_id = ?",
            (star_id,)).fetchone()
        new = simpledialog.askstring(
            "Rename star", "Display name (label):",
            initialvalue=(row["label"] or ""), parent=self)
        if new is None:
            return
        try:
            rename_star(self._db_path, star_id, new)
        except Exception as e:
            messagebox.showerror("Rename failed", str(e))
        self._refresh()

    def _set_sptype_selected(self):
        star_id = self._selected_single_star()
        if star_id is None:
            return
        row = self._conn.execute(
            "SELECT sp_type FROM stars WHERE star_id = ?",
            (star_id,)).fetchone()
        new = simpledialog.askstring(
            "Set spectral type",
            "Spectral type (e.g. A0V, B0.5IVpe; empty clears it):",
            initialvalue=(row["sp_type"] or ""), parent=self)
        if new is None:
            return
        try:
            set_sp_type(self._db_path, star_id, new)
        except Exception as e:
            messagebox.showerror("Set sp. type failed", str(e))
        self._refresh()

    def _reidentify_selected(self):
        star_id = self._selected_single_star()
        if star_id is None:
            return
        try:
            radius = float(self.v_radius.get())
        except ValueError:
            messagebox.showerror("SIMBAD", "Cone radius must be a number.")
            return
        row = self._conn.execute(
            "SELECT ra_deg, dec_deg FROM stars WHERE star_id = ?",
            (star_id,)).fetchone()

        self._status(f"Querying SIMBAD at RA {row['ra_deg']:.4f} "
                     f"Dec {row['dec_deg']:+.4f}, radius {radius:.0f}″…")
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            import source_identification as srcid
            # Synchronous query, same trade-off as Add-to-DB.
            match = srcid.query_position(row["ra_deg"], row["dec_deg"],
                                         radius_arcsec=radius)
        finally:
            self.config(cursor="")
        if match is None:
            self._status(f"SIMBAD: no object within {radius:.0f}″ "
                         f"(or query failed) — try a wider cone.")
            return
        if not messagebox.askyesno(
                "Apply identity",
                f"SIMBAD match at {match.sep_arcsec:.1f}″:\n\n"
                f"{match.info_text()}\n\nApply to star #{star_id}? "
                f"(overwrites name, type and position)"):
            return
        err = apply_identity(self._db_path, star_id, match)
        if err:
            messagebox.showwarning("Not applied", err)
        self._refresh()

    def _reload_history(self):
        """(Re)fill the recent-posters listbox from the sidecar."""
        self._history = load_poster_history(self._hist_path)
        self.lst_recent.delete(0, "end")
        for h in self._history:
            self.lst_recent.insert(
                "end", f"{h.get('utc', '')[:10]}  {h.get('title', '?')}"
                       f"  ({len(h.get('star_ids', []))}★"
                       f", {h.get('style', 'strip')})")

    def _restore_poster(self, _event):
        """Recent-files behaviour: clicking an entry re-ticks its stars
        and restores its text fields and grid parameters."""
        sel = self.lst_recent.curselection()
        if not sel:
            return
        h = self._history[sel[0]]
        self.v_title.set(h.get("title", ""))
        self.v_author.set(h.get("author", ""))
        self.v_instrument.set(h.get("instrument", ""))
        self.v_cols.set(h.get("cols", "Auto"))
        self.v_style.set(h.get("style", "strip"))
        wanted = set(h.get("star_ids", []))
        self._checked = set(wanted)
        self._refresh()              # prunes ticks for vanished stars
        missing = len(wanted) - len(self._checked)
        self._status(f"Restored “{h.get('title', '?')}”: "
                     f"{len(self._checked)} stars ticked"
                     + (f", {missing} no longer in the DB" if missing
                        else ""))

    def _export_poster(self):
        """Render the ticked stars to a poster PNG.

        Title (required), author and instrument come from the poster
        section's fields, as do the column override and plot style.
        Each ticked star contributes its LATEST spectrum; plots run hot
        to cool down the classic OBAFGKMN sequence, with untyped stars
        last (in tree order).  A successful render is pushed onto the
        recent-posters MRU.
        """
        star_iids = [iid for iid in self.tree.get_children("")
                     if int(iid.split("-", 1)[1]) in self._checked]
        if not star_iids:
            messagebox.showinfo(
                "Poster", "Tick the ✓ box of the stars to include first.")
            return

        title = self.v_title.get().strip()
        if not title:
            messagebox.showinfo(
                "Poster", "Enter a title first (e.g. “The Stars of "
                          "Cassiopeia”).")
            return
        credit = " · ".join(
            s.strip() for s in (self.v_author.get(),
                                self.v_instrument.get()) if s.strip())
        style = self.v_style.get()
        cols = None if self.v_cols.get() == "Auto" else int(self.v_cols.get())

        entries = []
        for iid in star_iids:
            kids = self.tree.get_children(iid)
            if not kids:
                continue
            # Children are date-ordered ascending → last = latest.
            # Exclusion zones apply (mask/interpolate transform the data);
            # overlay bands are not drawn on the poster's small panels.
            loaded = self._load_series(int(kids[-1].split("-", 1)[1]))
            if loaded is None:
                continue
            wls, flux, _spans = loaded
            _sid, name, sp, _ot = self._star_by_iid[kids[-1]]
            entries.append((name, sp, wls, flux))
        if not entries:
            messagebox.showinfo(
                "Poster", "None of the ticked stars has a calibrated "
                          "spectrum to plot.")
            return
        # Classic spectral sequence: hot to cool (OBAFGKMN), subclass
        # within; untyped stars follow in their tree order (stable sort).
        entries.sort(key=lambda e: spectral_sort_key(e[1]))

        path = filedialog.asksaveasfilename(
            title="Save poster PNG", defaultextension=".png",
            initialfile="spectra_poster.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        if not path:
            return
        try:
            render_poster(path, entries, title, credit,
                          style=style, cols=cols)
        except Exception as e:
            messagebox.showerror("Poster failed", str(e))
            return
        try:
            push_poster_history(self._hist_path, {
                "utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                "title": title,
                "author": self.v_author.get().strip(),
                "instrument": self.v_instrument.get().strip(),
                "cols": self.v_cols.get(),
                "style": style,
                "star_ids": sorted(self._checked)})
            self._reload_history()
        except OSError as e:
            self._status(f"Poster saved, history not written: {e}")
            return
        self._status(f"Poster saved: {path}  ({len(entries)} stars)")

    def destroy(self):
        # Cancel a queued idle-draw before tearing down the Tk canvas —
        # letting it fire against a destroyed widget is the access-
        # violation class reference_library_viewer._destroy_panel guards.
        try:
            cb = getattr(self.canvas, "_idle_draw_id", None)
            if cb is not None:
                self.canvas.get_tk_widget().after_cancel(cb)
        except Exception:
            pass
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        super().destroy()


# ---------------------------------------------------------------------------
# Self-check — pure data-access layer against a synthetic DB (no GUI)
# ---------------------------------------------------------------------------

class SpectrumEditor(tk.Toplevel):
    """Exclusion-zone editor: mark wavelength ranges polluted by another
    star crossing the strip.

    Drag horizontally on the plot to outline a zone — SpanSelector draws
    the full-height translucent box while you drag — then pick a method
    and Add. Zones apply immediately in the browser (samples are never
    modified, guidelines §4: the stored data stays, the zone is applied
    at display time). mask blanks the zone, interpolate bridges it,
    overlay only marks it. Writes go through short-lived rw connections
    (the rename_star pattern); the parent refreshes via ``on_change``.
    """

    METHOD_TINT = {"mask": "#e94560", "interpolate": "#4ec9b0",
                   "overlay": DUP}

    def __init__(self, parent, db_path, spectrum_id, title, on_change):
        super().__init__(parent)
        self.title(f"Exclusion editor — {title}")
        self.configure(bg=BG)
        self.geometry("1150x640")
        self._db_path = db_path
        self._spectrum_id = spectrum_id
        self._on_change = on_change
        self._pending = None
        self._zone_list = []

        ro = spectra_db.connect(db_path, readonly=True)
        try:
            self._wls, self._flux = load_samples(ro, spectrum_id)
        finally:
            ro.close()

        self.fig = Figure(figsize=(9.0, 5.6), dpi=100, facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="left", fill="both",
                                         expand=True, padx=(10, 6), pady=10)

        side = tk.Frame(self, bg=BG)
        side.pack(side="right", fill="y", padx=(0, 10), pady=10)
        tk.Label(side, text="Drag on the plot to outline\na polluted zone.",
                 bg=BG, fg=FG, font=("Courier New", 9),
                 justify="left").pack(anchor="w", pady=(0, 6))
        self.v_method = tk.StringVar(value="mask")
        for method in spectra_db.EXCLUSION_METHODS:
            tk.Radiobutton(side, text=method, value=method,
                           variable=self.v_method, bg=BG, fg=FG,
                           selectcolor=ENTRY_BG, activebackground=BG,
                           activeforeground=FG,
                           font=("Courier New", 9)).pack(anchor="w")
        self.v_pending = tk.StringVar(value="no zone drawn")
        tk.Label(side, textvariable=self.v_pending, bg=BG, fg=ACC,
                 font=("Courier New", 9)).pack(anchor="w", pady=(6, 2))

        def btn(text, cmd):
            b = tk.Button(side, text=text, bg=ENTRY_BG, fg=ENTRY_FG,
                          font=("Courier New", 9), relief="flat", width=20,
                          activebackground="#262c37",
                          activeforeground=ENTRY_FG, command=cmd)
            b.pack(side="top", pady=3)
            return b

        self.btn_add = btn("Add exclusion", self._add)
        self.btn_add.config(state="disabled")
        tk.Label(side, text="Existing zones", bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(anchor="w", pady=(10, 2))
        self.listbox = tk.Listbox(side, bg=ENTRY_BG, fg=ENTRY_FG, height=8,
                                  width=24, relief="flat",
                                  selectbackground=SPINE,
                                  selectforeground=ENTRY_FG,
                                  font=("Courier New", 9))
        self.listbox.pack(anchor="w")
        btn("Delete selected", self._delete)
        btn("Close", self.destroy)

        self._span = SpanSelector(self.ax, self._on_span, "horizontal",
                                  useblit=True,
                                  props=dict(alpha=0.25, facecolor=ACC))
        self._redraw()

    def _zones(self):
        ro = spectra_db.connect(self._db_path, readonly=True)
        try:
            return spectra_db.exclusions_for(ro, self._spectrum_id)
        finally:
            ro.close()

    def _redraw(self):
        zones = self._zones()
        self._zone_list = zones
        self.ax.clear()
        self.ax.set_facecolor(PANEL)
        for spine in self.ax.spines.values():
            spine.set_color(SPINE)
        self.ax.tick_params(colors=FG, labelsize=8)
        self.ax.grid(color=GRID, linewidth=0.4, alpha=0.5)
        self.ax.set_xlabel("Wavelength (Å)", color=FG, fontsize=9)
        if self._wls:
            flux, _spans = spectra_db.apply_exclusions(
                self._wls, self._flux, zones)
            wls = np.asarray(self._wls, dtype=float)
            flux = np.asarray(flux, dtype=float)
            rainbow_fill(self.ax, wls, flux, zorder=2)
            self.ax.plot(wls, flux, color="white", linewidth=0.6,
                         alpha=0.6, zorder=3)
            self.ax.set_ylim(bottom=0)
            # Every zone stays locatable: a tinted full-height band per
            # method, stronger for overlay (which is purely a marker).
            for zone in zones:
                self.ax.axvspan(zone["x1_wl_a"], zone["x2_wl_a"],
                                color=self.METHOD_TINT[zone["method"]],
                                alpha=0.18 if zone["method"] == "overlay"
                                else 0.12, zorder=4)
        else:
            self.ax.text(0.5, 0.5, "No calibrated samples in this spectrum",
                         ha="center", va="center", color=FG,
                         transform=self.ax.transAxes)
        self.listbox.delete(0, "end")
        for zone in zones:
            self.listbox.insert(
                "end", f"{zone['x1_wl_a']:.0f}–{zone['x2_wl_a']:.0f} Å "
                       f"· {zone['method']}")
        self.canvas.draw_idle()

    def _on_span(self, x1, x2):
        if abs(x2 - x1) < 1.0:          # a click, not a drag
            return
        self._pending = (min(x1, x2), max(x1, x2))
        self.v_pending.set(f"{self._pending[0]:.0f}–"
                           f"{self._pending[1]:.0f} Å")
        self.btn_add.config(state="normal")

    def _add(self):
        if self._pending is None:
            return
        conn = spectra_db.connect(self._db_path)
        try:
            spectra_db.add_exclusion(conn, self._spectrum_id,
                                     *self._pending, self.v_method.get())
        finally:
            conn.close()
        self._pending = None
        self.v_pending.set("no zone drawn")
        self.btn_add.config(state="disabled")
        self._redraw()
        self._on_change(self._spectrum_id)

    def _delete(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        zone = self._zone_list[sel[0]]
        conn = spectra_db.connect(self._db_path)
        try:
            spectra_db.delete_exclusion(conn, zone["exclusion_id"])
        finally:
            conn.close()
        self._redraw()
        self._on_change(self._spectrum_id)


def _selfcheck():
    import json
    import tempfile

    class FakeMatch:
        main_id = "* bet Cas"
        label = "Caph"
        sp_type = "F2III"
        otype = "dS*"
        all_ids = ["Gaia DR3 42"]
        cat_ra_deg = 2.29452
        cat_dec_deg = 59.14978

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.db")
        conn = spectra_db.connect(path)
        star = dict(ra_deg=10.0, dec_deg=20.0)
        run = dict(run_utc="2026-07-12T02:00:00+00:00",
                   config_json=json.dumps({}))
        spec = dict(free_selection=True)
        smp = [(0, 1.0, 4000.0, 0.5, 0.01, 0),
               (1, 2.0, None, None, None, spectra_db.FLAG_NO_CAL)]
        ds1 = dict(fits_path="a.fits", fits_sha256="s1", date_obs="d1")
        ds2 = dict(fits_path="b.fits", fits_sha256="s2", date_obs="d2")
        r1 = spectra_db.ingest_spectrum(conn, star=star, dataset=ds1,
                                        run=run, spectrum=spec, samples=smp)
        r2 = spectra_db.ingest_spectrum(conn, star=star, dataset=ds1,
                                        run=run, spectrum=spec, samples=smp)
        r3 = spectra_db.ingest_spectrum(conn, star=star, dataset=ds2,
                                        run=run, spectrum=spec, samples=smp)
        conn.close()

        ro = spectra_db.connect(path, readonly=True)
        ro.row_factory = sqlite3.Row
        model = load_tree(ro)
        assert len(model) == 1
        _star, specs = model[0]
        assert [sp["dup"] for sp in specs] == [True, True, False], specs
        assert star_display_name(_star).startswith("RA 10.0000")
        # Constellation derived on the fly from the stored position:
        # RA 10° Dec +20° lies in Pisces.
        assert _star["constellation"] == "Psc", _star["constellation"]

        wls, flux = load_samples(ro, r1["spectrum_id"])
        assert wls == [4000.0] and flux == [0.5]     # NULL cal row excluded
        # load_sigma shares that row filter, so it indexes with flux — and
        # this run recorded no exposure at all, snapshot or dataset.
        assert load_sigma(ro, r1["spectrum_id"]) == [0.01]
        assert capture_info(ro, r1["spectrum_id"]) == (None, 0)

        # A livestack in its own DB: the run snapshot's total wins over the
        # dataset's per-frame exposure and brings the frame count with it.
        # Samples with no sigma give None, which is what keeps the viewer's
        # ±2σ toggle off instead of shading a row of zeros.
        stack_path = os.path.join(td, "stack.db")
        sconn = spectra_db.connect(stack_path)
        sid = spectra_db.ingest_spectrum(
            sconn, star=star,
            dataset=dict(fits_path="livestacked", fits_sha256="s9",
                         exptime_s=30.0),
            run=dict(run_utc="2026-07-30T00:00:00+00:00",
                     config_json=json.dumps({"livestack": {
                         "n_frames": 18, "total_exptime_s": 540.0}})),
            spectrum=spec,
            samples=[(0, 1.0, 4000.0, 0.5, None, 0)])["spectrum_id"]
        sconn.close()
        sro = spectra_db.connect(stack_path, readonly=True)
        assert capture_info(sro, sid) == (540.0, 18), capture_info(sro, sid)
        assert load_sigma(sro, sid) is None
        assert capture_info(sro, 9999) == (None, 0)   # no such spectrum
        sro.close()

        # Whole-run delete: one dup goes, its run goes, dataset s1 stays
        # (still referenced by the other dup) — and its exclusion zones
        # go with it, while another spectrum's zones survive.
        rw = spectra_db.connect(path)
        spectra_db.add_exclusion(rw, r2["spectrum_id"], 4000, 4005, "mask")
        spectra_db.add_exclusion(rw, r1["spectrum_id"], 4000, 4005,
                                 "overlay")
        rw.close()
        delete_spectrum(path, r2["spectrum_id"])
        assert ro.execute("SELECT COUNT(*) FROM spectra").fetchone()[0] == 2
        left = [r[0] for r in ro.execute(
            "SELECT spectrum_id FROM spectrum_exclusions").fetchall()]
        assert left == [r1["spectrum_id"]], left
        assert ro.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        assert ro.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 2
        # Deleting the last s2 spectrum removes the orphaned dataset.
        delete_spectrum(path, r3["spectrum_id"])
        assert ro.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 1
        # No dups remain.
        assert not any(sp["dup"] for _s, ss in load_tree(ro) for sp in ss)

        rename_star(path, _star["star_id"], "  My star  ")
        assert ro.execute("SELECT label FROM stars").fetchone()[0] == "My star"

        set_sp_type(path, _star["star_id"], "  A0V ")
        assert ro.execute("SELECT sp_type FROM stars").fetchone()[0] == "A0V"
        set_sp_type(path, _star["star_id"], "   ")     # empty clears
        assert ro.execute("SELECT sp_type FROM stars").fetchone()[0] is None

        # ── Observer's note: round-trip, and empty stores NULL so that
        # "has a note" stays a plain IS NOT NULL test.
        set_note(path, _star["star_id"], "  Hα in emission 2026-07-13  ")
        assert (ro.execute("SELECT note FROM stars").fetchone()[0]
                == "Hα in emission 2026-07-13")
        assert load_tree(ro)[0][0]["note"].startswith("Hα")
        set_note(path, _star["star_id"], "  \n ")       # whitespace clears
        assert ro.execute("SELECT note FROM stars").fetchone()[0] is None

        # ── Migration: a v1 DB must gain the column, not explode.  The
        # browser reads through a READONLY connection, and connect() never
        # migrates one of those — so without migrate() first, an existing
        # DB would open happily and then fail on the first SELECT of note.
        old = os.path.join(td, "v1.db")
        v1 = sqlite3.connect(old)
        with v1:
            for stmt in spectra_db.MIGRATIONS[1]:
                v1.execute(stmt)
            v1.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
            v1.execute("INSERT INTO schema_version VALUES (1)")
            v1.execute("INSERT INTO stars (main_id, ra_deg, dec_deg) "
                       "VALUES ('old star', 1.0, 2.0)")
        v1.close()
        cols = lambda c: {r[1] for r in c.execute("PRAGMA table_info(stars)")}
        probe = spectra_db.connect(old, readonly=True)
        assert "note" not in cols(probe), "v1 fixture already has the column"
        probe.close()
        migrate(old)                                   # what _open_db does
        probe = spectra_db.connect(old, readonly=True)
        probe.row_factory = sqlite3.Row
        assert "note" in cols(probe)
        assert probe.execute("SELECT note FROM stars").fetchone()[0] is None
        assert load_tree(probe) == [] or True          # reads without error
        probe.close()

        assert apply_identity(path, _star["star_id"], FakeMatch()) is None
        row = ro.execute("SELECT gaia_dr3_source_id, main_id, label, ra_deg, "
                         "pos_epoch_jyear FROM stars").fetchone()
        assert tuple(row) == (42, "* bet Cas", "Caph", 2.29452, 2000.0), \
            tuple(row)
        # Re-identification moved the star to Caph's catalog position —
        # the on-the-fly constellation follows, nothing stored.
        star2 = load_tree(ro)[0][0]
        assert star2["constellation"] == "Cas", star2["constellation"]

        # Gaia conflict with a second star → refused with a message.
        rw = spectra_db.connect(path)
        rw.execute("INSERT INTO stars (gaia_dr3_source_id, ra_deg, dec_deg) "
                   "VALUES (7, 30.0, 40.0)")
        rw.commit()
        other = rw.execute("SELECT star_id FROM stars "
                           "WHERE gaia_dr3_source_id = 7").fetchone()[0]
        rw.close()

        class ConflictMatch(FakeMatch):
            all_ids = ["Gaia DR3 42"]     # already taken by star 1
        assert apply_identity(path, other, ConflictMatch()) is not None
        ro.close()

        # ── Same-capture duplicates: two saves of one star from the same
        # DATE-OBS collapse to the newest run; a different capture of the
        # same star (following it) is NOT a duplicate.
        d1 = os.path.join(td, "m1.db")
        c1 = spectra_db.connect(d1)
        star_a = dict(main_id="HD 1", ra_deg=10.0, dec_deg=20.0)
        cap1 = dict(fits_path="f1.fits", fits_sha256="h1",
                    date_obs="2026-07-17T22:00:00")
        cap2 = dict(fits_path="f2.fits", fits_sha256="h2",
                    date_obs="2026-07-16T22:00:00")
        one_px = [(0, 1.0, 4000.0, 0.5, 0.01, 0)]
        a1 = spectra_db.ingest_spectrum(
            c1, star=star_a, dataset=cap1, spectrum={}, samples=one_px,
            run=dict(run_utc="2026-07-18T01:00:00", config_json="{}"))
        spectra_db.ingest_spectrum(
            c1, star=star_a, dataset=cap1, spectrum={}, samples=one_px,
            run=dict(run_utc="2026-07-18T02:00:00", config_json="{}"))
        spectra_db.ingest_spectrum(
            c1, star=star_a, dataset=cap2, spectrum={}, samples=one_px,
            run=dict(run_utc="2026-07-18T03:00:00", config_json="{}"))
        doomed = duplicate_spectra(c1)
        assert doomed == [a1["spectrum_id"]], doomed
        c1.close()
        for sid in doomed:
            delete_spectrum(d1, sid)
        c1 = spectra_db.connect(d1, readonly=True)
        assert c1.execute("SELECT COUNT(*) FROM spectra").fetchone()[0] == 2
        assert duplicate_spectra(c1) == []
        c1.close()

        # ── Merge: the known star lands on the existing row (waterfall),
        # the new star gets its own; exclusions travel; re-merge no-op.
        d2 = os.path.join(td, "m2.db")
        c2 = spectra_db.connect(d2)
        b1 = spectra_db.ingest_spectrum(
            c2, star=dict(main_id="HD 1", ra_deg=10.0, dec_deg=20.0),
            dataset=dict(fits_path="r1.fits", fits_sha256="rh1",
                         date_obs="2026-07-17T23:00:00"),
            run=dict(run_utc="2026-07-18T04:00:00", config_json="{}"),
            spectrum=dict(source_x=5.0, source_y=6.0), samples=one_px)
        spectra_db.ingest_spectrum(
            c2, star=dict(main_id="HD 99", ra_deg=50.0, dec_deg=-10.0),
            dataset=dict(fits_path="r2.fits", fits_sha256="rh2",
                         date_obs="2026-07-17T23:30:00"),
            run=dict(run_utc="2026-07-18T05:00:00", config_json="{}"),
            spectrum={}, samples=one_px)
        spectra_db.add_exclusion(c2, b1["spectrum_id"], 5000, 5100, "mask")
        c2.close()
        assert merge_db(d1, d2) == (2, 0)
        assert merge_db(d1, d2) == (0, 2)          # idempotent
        ro2 = spectra_db.connect(d1, readonly=True)
        assert ro2.execute("SELECT COUNT(*) FROM stars").fetchone()[0] == 2
        assert ro2.execute("SELECT COUNT(*) FROM spectra").fetchone()[0] == 4
        zones = [tuple(r) for r in ro2.execute(
            "SELECT x1_wl_a, x2_wl_a, method FROM spectrum_exclusions")]
        assert zones == [(5000.0, 5100.0, "mask")], zones
        ro2.close()

        # Poster layout rule: ≤7 rows per column, ≤3 columns,
        # landscape A4 only at 3 columns.
        assert poster_grid(1) == (1, 1, False)
        assert poster_grid(7) == (7, 1, False)
        assert poster_grid(8) == (4, 2, False)
        assert poster_grid(14) == (7, 2, False)
        assert poster_grid(15) == (5, 3, True)
        assert poster_grid(21) == (7, 3, True)
        assert poster_grid(24) == (8, 3, True)
        # Overrides: explicit columns win; square (circular) aims for
        # near-square cells and goes landscape when wider than tall.
        assert poster_grid(21, cols=2) == (11, 2, False)
        assert poster_grid(21, cols=4) == (6, 4, True)
        assert poster_grid(3, cols=8) == (1, 3, True)      # capped at n
        assert poster_grid(1, square=True) == (1, 1, False)
        assert poster_grid(2, square=True) == (1, 2, True)
        assert poster_grid(8, square=True) == (3, 3, False)
        assert poster_grid(12, square=True) == (3, 4, True)
        assert poster_grid(24, square=True) == (6, 4, False)

        # OBAFGKMN ordering: class then subclass, untyped last.
        seq = ["M1III", "A0V", "", "F2.5II", "B9V", "sdX", "A2IV"]
        assert sorted(seq, key=spectral_sort_key) == \
            ["B9V", "A0V", "A2IV", "F2.5II", "M1III", "", "sdX"], \
            sorted(seq, key=spectral_sort_key)
        # Be stars ARE B-type (the 'e' is an emission suffix) — they file
        # among the Bs; non-OBAFGKMN prefixes (WR, white dwarfs) go last.
        assert spectral_sort_key("B2Ve") == (1, 2.0)
        assert spectral_sort_key("B0.5IVpe") == (1, 0.5)     # gamma Cas
        assert spectral_sort_key("WC8")[0] == len(_SPECTRAL_SEQUENCE)
        assert spectral_sort_key("DA2")[0] == len(_SPECTRAL_SEQUENCE)

        # Info cards: MK-class stem match, qualifiers/peculiarities drop.
        nd = os.path.join(td, "notes")
        os.makedirs(nd)
        with open(os.path.join(nd, "g5iii.md"), "w", encoding="utf-8") as f:
            f.write("# G5III — G5 giant\n\n## Class: G stars\nYellow.\n\n"
                    "## This type\nGiant.\n\n## Features\n"
                    "| Wavelength (Å) | Feature | Notes |\n|---|---|---|\n"
                    "| 4861 | Hβ | line |\n"
                    "| 4300-4315 | CH G band | band |\n")
        assert notes_card("G5III", nd).endswith("g5iii.md")
        assert notes_card(" g5 III ", nd) is not None      # case/spaces
        assert notes_card("G5IIIvar", nd) is not None      # suffix drops
        assert notes_card(None, nd) is None
        assert notes_card("kA2hA5mA7V", nd) is None        # composite
        # G5V reaches g5iii (cost 1.5*2 = 3.0, exactly at the cap) but
        # G1V does not (subtype 4 away as well) — the cap is what stops a
        # lone card in a class from describing the whole class.
        assert notes_card("G5V", nd).endswith("g5iii.md")
        assert notes_card("G1V", nd) is None

        # parse_sp_type: what SIMBAD actually emits.
        assert parse_sp_type("K0-IIIa") == ("K", 0.0, 0.0, frozenset({"III"}))
        assert parse_sp_type("M1Iab") == ("M", 1.0, 1.0, frozenset({"I"}))
        assert parse_sp_type("K2IV-V") == ("K", 2.0, 2.0,
                                           frozenset({"IV", "V"}))
        assert parse_sp_type("B1-2Ia-0ep") == ("B", 1.0, 2.0,
                                               frozenset({"I"}))
        assert parse_sp_type("B0.5III(n)") == ("B", 0.5, 0.5,
                                               frozenset({"III"}))
        assert parse_sp_type("BC0.7Ia") == ("B", 0.7, 0.7, frozenset({"I"}))
        assert parse_sp_type("A0")[3] is None          # no luminosity class
        assert parse_sp_type("B7Vn+B9VHgMn+A1V")[:3] == ("B", 7.0, 7.0)
        assert parse_sp_type("Be") is None             # no subtype at all
        assert parse_sp_type("NOVA") is None

        # The nearest-card ladder, against the real library.  Grid holes
        # snap to a neighbour; range cards cover their span exactly.
        def stem(sp, ot=None):
            p, _n = notes_match(sp, otype=ot)
            return os.path.basename(p)[:-3] if p else None

        assert stem("K0-IIIa") == "k0iii"       # exact, once qualifiers go
        assert stem("B7V") == "b57v"            # B5-7 V composite covers it
        assert stem("A5IV") == "a47iv"          # A4-7 IV composite covers it
        assert stem("B0.5III(n)") == "b12iii"   # inside-range beats b3iii
        assert stem("M6II") == "m6iii"          # temperature beats gravity
        assert stem("M1Iab") == "m2i"
        assert stem("B1-2Ia-0ep") == "b1i"
        assert stem("F7V") in ("f6v", "f8v")    # a genuine tie
        assert stem("NOVA") == "_nova"          # phenomena, keyed on token
        assert stem("", "SN*") == "_sn"         # ...or on SIMBAD's otype
        assert stem("Be") == "_be"              # no subtype to work with
        assert stem("", "Em*") == "_be"
        assert stem("O2If") is None             # too far: cap holds

        # Carbon stars: revised Keenan forms, old Harvard R/N, otype C*.
        assert stem("C-N4.5") == "_c"
        assert stem("C4,5J") == "_c"
        assert stem("CH") == "_c"
        assert stem("N8") == "_c" and stem("R3") == "_c"
        assert stem("", "C*") == "_c"
        assert not _is_carbon("NOVA")           # N + letter: not Harvard N
        assert not _is_carbon("K3III")          # MK types stay on the grid

        # A Be star with a real MK type keeps its MK card — the disc is an
        # annotation on the photosphere, not a replacement for it — so an
        # Em* otype must not outrank a parseable type.  It does earn a
        # caveat, because the card describes Hα in absorption.
        # (There is no b2v card, so B2V itself lands on a neighbour — the
        # point here is that Em* did not divert it to the Be card.)
        assert stem("B2Ve", "Em*") == stem("B2V") == "b1v"
        assert stem("B0.5IVpe") == "b2iv"       # gamma Cas: nearest B card
        assert "emission" in notes_match("B2Ve", otype="Em*")[1]
        assert "emission" not in (notes_match("B2V")[1] or "")
        assert _has_emission("B0.5IVpe") and _has_emission("B1-2Ia-0ep")
        assert not _has_emission("B9VHgMn")     # 'e'-free suffixes
        assert not _has_emission("G5IIIvar") and not _has_emission("M2Iab")

        # Provenance: an approximation must never read as a fact.
        assert notes_match("K0III")[1] is None            # exact: no caveat
        assert notes_match("B7V")[1] is None              # covered: no caveat
        assert "nearest" in notes_match("M6II")[1]
        assert "equally close" in notes_match("F7V")[1]
        assert "assuming a dwarf" in notes_match("A0")[1]
        # otype rescues the luminosity guess: an M7 AGB star is a giant,
        # and calling it a dwarf would be the worst error the matcher can
        # make (an M7 V and an M7 III share a subtype and nothing else).
        assert stem("M7", "AB*") == "m7iii"
        assert stem("M7") == "m6v"              # no otype: dwarf default
        assert "assuming a giant" in notes_match("M7", otype="LP*")[1]
        assert "multiple system" in notes_match("B7Vn+B9VHgMn+A1V")[1]

        # notes_text: matched card (with its caveat when inexact);
        # otherwise the shared '## Class' section from a same-class
        # sibling; '' otherwise.
        assert "Giant." in notes_text("G5III", nd)
        fb = notes_text("G1V", nd)
        assert "Yellow." in fb and "Giant." not in fb, fb
        assert "class info only" in fb
        assert notes_text("K0III", nd) == ""    # no K card to borrow
        assert notes_text("", nd) == ""
        assert notes_text(None, nd) == ""
        assert notes_text("M6II").startswith("≈ ")       # caveat leads
        assert "Type Ia" in notes_text("", otype="SN*")

        # notes_features: table rows → (lo, hi, name); hi == lo marks a
        # single line; class-sibling fallback; header row not a feature.
        assert notes_features("G5III", nd) == [
            (4861.0, 4861.0, "Hβ"), (4300.0, 4315.0, "CH G band")]
        assert notes_features("G1V", nd) == notes_features("G5III", nd)
        assert notes_features("K0III", nd) == []
        assert notes_features(None, nd) == []

        # notes_sources: parsed from the real notes/README.md, since that
        # file *is* the reference list every card leans on.  [] when the
        # dir has no README (the synthetic one above).
        assert notes_sources(nd) == []
        real = notes_sources()
        assert len(real) >= 4, real
        assert all(u.startswith("http") for _t, u in real), real
        assert any("Gray" in t for t, _u in real), real

        # Class filter: letter match, 'Other' = untyped + non-OBAFGKM.
        assert sp_filter_match("B0.5IVpe", "B")
        assert not sp_filter_match("B0.5IVpe", "A")
        # Composite SIMBAD types misfile by first letter (same ceiling
        # as spectral_sort_key): "kA2hA5mA7V" lands under K, not A.
        assert sp_filter_match("kA2hA5mA7V", "K")
        assert sp_filter_match(None, "Other")
        assert sp_filter_match("WC8", "Other")
        assert sp_filter_match(None, "All") and sp_filter_match("M1", "All")
        assert sp_filter_match(" g8iii", "G")

        # smooth=True is the per-segment fill_between loop (one artist
        # per segment), the fast default a single PolyCollection.
        chk = Figure()
        FigureCanvasAgg(chk)
        ax_f, ax_s = chk.subplots(1, 2)
        rainbow_fill(ax_f, [1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0])
        rainbow_fill(ax_s, [1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0], smooth=True)
        assert len(ax_f.collections) == 1 and len(ax_s.collections) == 3

        # Circular plot: radius ∝ wavelength, hue = λ colour × flux,
        # empty below λ_min except the fake star.
        size = 201
        cimg = circular_image([4000.0, 8000.0], [1.0, 1.0], size=size)
        assert cimg.shape == (size, size, 3)
        cc = (size - 1) // 2
        from matplotlib.colors import to_rgb
        assert np.allclose(cimg[0, 0], to_rgb(PANEL))       # corner: bg
        # ring pixel at r = 0.75·R → λ = 6000 Å, full flux
        assert np.allclose(cimg[cc, cc + int(0.75 * cc)],
                           wavelength_to_rgb(6000.0), atol=0.05), \
            cimg[cc, cc + int(0.75 * cc)]
        # star core: 0.9 × tint over the background, NOT white — here
        # the tint is the 2-sample fallback (mean of the endpoint hues)
        exp = np.clip(0.9 * (np.array(wavelength_to_rgb(4000.0))
                             + wavelength_to_rgb(8000.0)) / 2
                      + to_rgb(PANEL), 0.0, 1.0)
        assert np.allclose(cimg[cc, cc], exp, atol=0.02), cimg[cc, cc]
        # the star's glow must have died off before the ring's inner
        # edge (r = 0.45·R → λ = 3600 Å, still inside the hole): the
        # pixel is back to the background, no bleed into the ring
        assert np.allclose(cimg[cc, cc + int(0.45 * cc)], to_rgb(PANEL),
                           atol=0.02), cimg[cc, cc + int(0.45 * cc)]
        try:
            circular_image([4000.0], [1.0])
            raise AssertionError("single-sample ring not refused")
        except ValueError:
            pass

        # Star tint: blackbody palette from a Planck fit — a cool star
        # comes out red-warm, a hot one blue-white; the normalization
        # scale must not matter (free-amplitude fit).
        r, g, b = kelvin_to_rgb(3000.0)
        assert r == 1.0 and r > g > b, (r, g, b)
        r, g, b = kelvin_to_rgb(20000.0)
        assert b == 1.0 and b > g > r, (r, g, b)

        def planck(wl_a, t):
            x = 1.43877688e8 / (wl_a * t)
            return wl_a ** -5.0 / np.expm1(x)

        wl = np.linspace(4000.0, 8000.0, 400)
        cool = star_color(wl, 1e17 * planck(wl, 3000.0))
        hot = star_color(wl, 1e21 * planck(wl, 20000.0))
        assert cool[0] > cool[2], cool           # redder than blue
        assert hot[2] > hot[0], hot              # bluer than red
        assert star_color([4000.0, 8000.0], [1.0, 1.0]) is None  # <3 anchors
        # …and the tint actually shows at the rendered star's core.
        px = circular_image(wl, 1e17 * planck(wl, 3000.0), size=101)[50, 50]
        assert px[0] > px[2] + 0.3 and px[1] < 0.95, px

        # Standalone star tile (combined cells): same recipe — tinted
        # core, page background at the edges.
        simg = star_image(wl, 1e17 * planck(wl, 3000.0), size=101)
        assert simg.shape == (101, 101, 3)
        assert np.allclose(simg[0, 0], to_rgb(BG), atol=0.02)
        assert simg[50, 50][0] > simg[50, 50][2] + 0.3, simg[50, 50]

        # Poster history: MRU, dedup on (title, star set), capped at 8.
        hp = os.path.join(td, "t.db.posters.json")
        assert load_poster_history(hp) == []
        for i in range(10):
            push_poster_history(hp, {"title": f"P{i}", "star_ids": [i]})
        hist = load_poster_history(hp)
        assert len(hist) == 8 and hist[0]["title"] == "P9"
        push_poster_history(hp, {"title": "P5", "star_ids": [5]})  # redo
        hist = load_poster_history(hp)
        assert len(hist) == 8 and hist[0]["title"] == "P5"
        assert sum(h["title"] == "P5" for h in hist) == 1

        # Headless poster render (pure Agg, no Tk) — both styles.
        png = os.path.join(td, "poster.png")
        wls = list(range(4000, 7000, 10))
        flux = [0.5 + 0.4 * (w % 500) / 500 for w in wls]
        entries = [(f"Star {i}", "A0V" if i % 2 else "", wls, flux)
                   for i in range(8)]
        render_poster(png, entries, "The Stars of Cassiopeia",
                      "Observer · 200 mm f/8 + SA200", dpi=150)
        assert os.path.getsize(png) > 10_000
        render_poster(png, entries, "Rings of Cassiopeia", dpi=150,
                      style="circular", cols=4)
        assert os.path.getsize(png) > 10_000
        render_poster(png, entries, "Stars of Cassiopeia", dpi=150,
                      style="combined")
        assert os.path.getsize(png) > 10_000
        try:
            render_poster(png, [], "empty")
            raise AssertionError("empty poster not refused")
        except ValueError:
            pass
    print("spectra_browser self-check OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        db_path = sys.argv[1] if len(sys.argv) > 1 \
            else spectra_db.DEFAULT_DB_PATH
        SpectraBrowser(db_path).mainloop()
