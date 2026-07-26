"""
db/spectra_db.py
================

Spectra database: schema, star-identity resolution and ingest.

Follows the light-curve DB design contract (spectra_db_guidelines_2026-07-12.md):
integer surrogate keys, nullable external identities with ICRS position as
ground truth, the gaia→name→cone dedup waterfall, append-only run provenance
with a full JSON config snapshot, portable SQL only, numbered migrations, and
a synthetic self-check runnable as ``py -3.13 -m db.spectra_db``.

Pure module: sqlite3 + stdlib only, no tkinter, no astropy.  The GUI gathers
everything (arrays, header fields, config snapshot) and calls
``ingest_spectrum`` — one transaction, all-or-nothing.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import subprocess
import sys

# Single DB file for all sessions/targets, in the app's db/ subfolder
# (photometry-pipeline convention; outside any dataset folder).
#
# Frozen build: __file__ points inside the bundle's _internal/, which a
# reinstall wipes — the observations must NOT live there.  Sit beside the
# .exe instead, where the DB is visible and survives replacing the bundle.
if getattr(sys, "frozen", False):
    DEFAULT_DB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(sys.executable)), "spectra.db")
else:
    DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "spectra.db")

# Dedup waterfall cone radius (guidelines §1).
CONE_RADIUS_ARCSEC = 1.5

# Per-sample flag bits (controlled vocabulary in code, not SQL CHECKs — §5).
FLAG_CONTAM = 1    # column masked as contaminated by an in-aperture star
FLAG_NO_CAL = 2    # outside the calibrated window / no calibrated value
FLAG_BAD_SKY = 4   # sky bands untrustworthy at this column (sigma-clip flag)

# Exclusion-zone rendering methods (controlled vocabulary in code — §5).
#   mask        — blank the zone (NaN gap in plots)
#   interpolate — bridge the zone linearly between its edges
#   overlay     — leave the data, draw a translucent marker band
EXCLUSION_METHODS = ("mask", "interpolate", "overlay")

SCHEMA_VERSION = 3

MIGRATIONS = {
    1: [
        """CREATE TABLE stars (
               star_id            INTEGER PRIMARY KEY,
               gaia_dr3_source_id INTEGER UNIQUE,
               main_id            TEXT,
               label              TEXT,
               sp_type            TEXT,
               otype              TEXT,
               ra_deg             REAL NOT NULL,
               dec_deg            REAL NOT NULL,
               pos_epoch_jyear    REAL
           )""",
        "CREATE INDEX idx_stars_radec ON stars(dec_deg, ra_deg)",
        """CREATE TABLE datasets (
               dataset_id   INTEGER PRIMARY KEY,
               fits_path    TEXT NOT NULL,
               fits_sha256  TEXT,
               date_obs     TEXT,
               exptime_s    REAL,
               telescope    TEXT,
               instrument   TEXT,
               object_name  TEXT,
               site_lat_deg REAL,
               site_lon_deg REAL,
               site_elev_m  REAL
           )""",
        """CREATE TABLE runs (
               run_id      INTEGER PRIMARY KEY,
               dataset_id  INTEGER NOT NULL REFERENCES datasets(dataset_id),
               run_utc     TEXT NOT NULL,
               git_hash    TEXT,
               config_json TEXT NOT NULL
           )""",
        "CREATE INDEX idx_runs_dataset ON runs(dataset_id)",
        """CREATE TABLE spectra (
               spectrum_id      INTEGER PRIMARY KEY,
               star_id          INTEGER NOT NULL REFERENCES stars(star_id),
               run_id           INTEGER NOT NULL REFERENCES runs(run_id),
               source_x         REAL,
               source_y         REAL,
               free_selection   INTEGER NOT NULL DEFAULT 0,
               match_sep_arcsec REAL,
               n_samples        INTEGER NOT NULL,
               UNIQUE(star_id, run_id)
           )""",
        "CREATE INDEX idx_spectra_star ON spectra(star_id)",
        "CREATE INDEX idx_spectra_run ON spectra(run_id)",
        # Bulk per-sample data: composite PK, no other index (§3).
        """CREATE TABLE samples (
               spectrum_id  INTEGER NOT NULL REFERENCES spectra(spectrum_id),
               pixel        INTEGER NOT NULL,
               raw_counts   REAL,
               wavelength_a REAL,
               cal_flux     REAL,
               cal_sigma    REAL,
               flags        INTEGER NOT NULL DEFAULT 0,
               PRIMARY KEY (spectrum_id, pixel)
           )""",
    ],
    # The observer's own note on a star — free text, hand-written in the
    # browser.  It sits with the other hand-editable identity fields
    # (label, sp_type), NOT in the append-only provenance chain (§2): it is
    # an annotation about the star, not a fact derived from a run.  The
    # producer never reads or writes it, and its INSERTs name their columns,
    # so this is invisible to the ingest path.
    2: [
        "ALTER TABLE stars ADD COLUMN note TEXT",
    ],
    # Hand-drawn wavelength exclusion zones (a polluting star crossing the
    # strip): like stars.note these are editor annotations about a product,
    # NOT run provenance — mutable, deletable, outside the append-only
    # chain (§2). Several zones per spectrum are expected; "has exclusion"
    # is an EXISTS query, never a stored flag. Samples stay untouched:
    # the zone is applied at display time (§4: masked data is stored,
    # flagged, never dropped — here the "flag" lives beside the data).
    3: [
        """CREATE TABLE spectrum_exclusions (
               exclusion_id INTEGER PRIMARY KEY,
               spectrum_id  INTEGER NOT NULL REFERENCES spectra(spectrum_id),
               x1_wl_a      REAL NOT NULL,
               x2_wl_a      REAL NOT NULL,
               method       TEXT NOT NULL
           )""",
        "CREATE INDEX idx_exclusions_spectrum "
        "ON spectrum_exclusions(spectrum_id)",
    ],
}


def connect(path: str = DEFAULT_DB_PATH, readonly: bool = False):
    """Open the spectra DB, creating/migrating it unless readonly.

    Refuses to open a DB whose schema_version is newer than this code.
    Readonly connections (viewers) use a mode=ro URI and never migrate.
    """
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='schema_version'").fetchone()
    version = (conn.execute("SELECT version FROM schema_version").fetchone()[0]
               if row else 0)
    if version > SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(
            f"DB schema version {version} is newer than this code "
            f"({SCHEMA_VERSION}) — update the application.")

    if not readonly and version < SCHEMA_VERSION:
        with conn:
            if version == 0:
                conn.execute(
                    "CREATE TABLE schema_version (version INTEGER NOT NULL)")
                conn.execute("INSERT INTO schema_version VALUES (0)")
            for v in range(version + 1, SCHEMA_VERSION + 1):
                for stmt in MIGRATIONS[v]:
                    conn.execute(stmt)
                conn.execute("UPDATE schema_version SET version = ?", (v,))
    return conn


def gaia_dr3_id(all_ids) -> int | None:
    """Extract the Gaia DR3 source_id from a SIMBAD identifier list."""
    for s in all_ids or []:
        m = re.fullmatch(r"Gaia DR3 (\d+)", s.strip())
        if m:
            return int(m.group(1))
    return None


def git_hash() -> str | None:
    """Current repo commit, or None (e.g. frozen PyInstaller build, no git)."""
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # CREATE_NO_WINDOW: a frozen --noconsole build would otherwise
        # flash a console (Windows-only app, same as the ASTAP call).
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    """Small-angle separation — exact enough at the 1.5″ waterfall scale."""
    dra = (ra1 - ra2) * math.cos(math.radians(0.5 * (dec1 + dec2)))
    return math.hypot(dra, dec1 - dec2) * 3600.0


def resolve_star(conn, star: dict) -> tuple[int, bool]:
    """Dedup waterfall (guidelines §1): gaia id → name → 1.5″ cone → insert.

    ``star`` keys: gaia_dr3_source_id, main_id, label, sp_type, otype,
    ra_deg, dec_deg, pos_epoch_jyear (all nullable except ra/dec).
    Returns (star_id, created).  On a match, backfills identity attributes
    the existing row lacks (NULL-only — never overwrites).
    """
    gaia = star.get("gaia_dr3_source_id")
    name = star.get("main_id")

    match_id = None
    if gaia is not None:
        row = conn.execute("SELECT star_id FROM stars "
                           "WHERE gaia_dr3_source_id = ?", (gaia,)).fetchone()
        if row:
            match_id = row[0]

    if match_id is None and name:
        row = conn.execute(
            "SELECT star_id, gaia_dr3_source_id FROM stars WHERE main_id = ?",
            (name,)).fetchone()
        # Rejected if the two sides carry different Gaia ids (close pairs
        # sharing one catalog entry).
        if row and not (gaia is not None and row[1] is not None
                        and row[1] != gaia):
            match_id = row[0]

    if match_id is None:
        ra, dec = star["ra_deg"], star["dec_deg"]
        r_deg = CONE_RADIUS_ARCSEC / 3600.0
        cosd = max(math.cos(math.radians(dec)), 1e-6)
        rows = conn.execute(
            "SELECT star_id, gaia_dr3_source_id, ra_deg, dec_deg FROM stars "
            "WHERE dec_deg BETWEEN ? AND ? AND ra_deg BETWEEN ? AND ?",
            (dec - r_deg, dec + r_deg,
             ra - r_deg / cosd, ra + r_deg / cosd)).fetchall()
        best = None
        for sid, sgaia, sra, sdec in rows:
            # A cone hit with a conflicting Gaia id on either side is two
            # real stars, not a match.
            if gaia is not None and sgaia is not None and sgaia != gaia:
                continue
            sep = _sep_arcsec(ra, dec, sra, sdec)
            if sep <= CONE_RADIUS_ARCSEC and (best is None or sep < best[1]):
                best = (sid, sep)
        if best:
            match_id = best[0]

    if match_id is not None:
        # NULL-only enrichment: a position-only row gains its identity the
        # first time a resolved save matches it.
        for col in ("gaia_dr3_source_id", "main_id", "label",
                    "sp_type", "otype"):
            val = star.get(col)
            if val is not None and val != "":
                conn.execute(
                    f"UPDATE stars SET {col} = ? "
                    f"WHERE star_id = ? AND {col} IS NULL", (val, match_id))
        return match_id, False

    cur = conn.execute(
        "INSERT INTO stars (gaia_dr3_source_id, main_id, label, sp_type, "
        "otype, ra_deg, dec_deg, pos_epoch_jyear) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (gaia, name or None, star.get("label") or None,
         star.get("sp_type") or None, star.get("otype") or None,
         star["ra_deg"], star["dec_deg"], star.get("pos_epoch_jyear")))
    return cur.lastrowid, True


def _resolve_dataset(conn, dataset: dict) -> int:
    """Reuse the dataset row for the same file (by sha256, else path)."""
    sha = dataset.get("fits_sha256")
    if sha:
        row = conn.execute("SELECT dataset_id FROM datasets "
                           "WHERE fits_sha256 = ?", (sha,)).fetchone()
    else:
        row = conn.execute(
            "SELECT dataset_id FROM datasets "
            "WHERE fits_path = ? AND fits_sha256 IS NULL",
            (dataset["fits_path"],)).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO datasets (fits_path, fits_sha256, date_obs, exptime_s, "
        "telescope, instrument, object_name, site_lat_deg, site_lon_deg, "
        "site_elev_m) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (dataset["fits_path"], sha, dataset.get("date_obs"),
         dataset.get("exptime_s"), dataset.get("telescope"),
         dataset.get("instrument"), dataset.get("object_name"),
         dataset.get("site_lat_deg"), dataset.get("site_lon_deg"),
         dataset.get("site_elev_m")))
    return cur.lastrowid


def ingest_spectrum(conn, *, star: dict, dataset: dict, run: dict,
                    spectrum: dict, samples: list) -> dict:
    """Ingest one spectrum: star waterfall + dataset + new run + samples.

    Append-only (§2): every call creates a new run; existing rows are never
    edited.  One transaction — on any error nothing is written.

    ``run``      : {run_utc, git_hash, config_json}
    ``spectrum`` : {source_x, source_y, free_selection, match_sep_arcsec}
    ``samples``  : iterable of (pixel, raw_counts, wavelength_a, cal_flux,
                   cal_sigma, flags); None where a value is absent/masked.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("Refusing to ingest a spectrum with no samples.")
    with conn:
        star_id, created = resolve_star(conn, star)
        dataset_id = _resolve_dataset(conn, dataset)
        cur = conn.execute(
            "INSERT INTO runs (dataset_id, run_utc, git_hash, config_json) "
            "VALUES (?, ?, ?, ?)",
            (dataset_id, run["run_utc"], run.get("git_hash"),
             run["config_json"]))
        run_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO spectra (star_id, run_id, source_x, source_y, "
            "free_selection, match_sep_arcsec, n_samples) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (star_id, run_id, spectrum.get("source_x"),
             spectrum.get("source_y"),
             int(bool(spectrum.get("free_selection"))),
             spectrum.get("match_sep_arcsec"), len(samples)))
        spectrum_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO samples (spectrum_id, pixel, raw_counts, "
            "wavelength_a, cal_flux, cal_sigma, flags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(spectrum_id,) + tuple(s) for s in samples])
    return {"star_id": star_id, "created_star": created,
            "dataset_id": dataset_id, "run_id": run_id,
            "spectrum_id": spectrum_id}


# ---------------------------------------------------------------------------
# Exclusion zones — editor annotations, applied at display time
# ---------------------------------------------------------------------------

def add_exclusion(conn, spectrum_id: int, x1_wl_a: float, x2_wl_a: float,
                  method: str) -> int:
    """Add an exclusion zone; endpoints are ordered for the caller."""
    if method not in EXCLUSION_METHODS:
        raise ValueError(f"unknown exclusion method {method!r}")
    x1, x2 = sorted((float(x1_wl_a), float(x2_wl_a)))
    if not (x2 > x1):
        raise ValueError("exclusion zone must have nonzero width")
    with conn:
        cur = conn.execute(
            "INSERT INTO spectrum_exclusions (spectrum_id, x1_wl_a, "
            "x2_wl_a, method) VALUES (?, ?, ?, ?)",
            (spectrum_id, x1, x2, method))
    return cur.lastrowid


def exclusions_for(conn, spectrum_id: int) -> list:
    """A spectrum's exclusion zones, bluest first."""
    rows = conn.execute(
        "SELECT exclusion_id, x1_wl_a, x2_wl_a, method "
        "FROM spectrum_exclusions WHERE spectrum_id = ? ORDER BY x1_wl_a",
        (spectrum_id,)).fetchall()
    return [{"exclusion_id": r[0], "x1_wl_a": r[1], "x2_wl_a": r[2],
             "method": r[3]} for r in rows]


def delete_exclusion(conn, exclusion_id: int) -> None:
    with conn:
        conn.execute("DELETE FROM spectrum_exclusions WHERE exclusion_id = ?",
                     (exclusion_id,))


def apply_exclusions(wls, flux, exclusions):
    """Apply zones to a (wavelength, flux) series for display.

    Returns (flux_out, overlay_spans): mask zones become NaN, interpolate
    zones a linear bridge between the edge samples (falling back to mask
    when a zone touches the end of the spectrum and has no anchor),
    overlay zones leave the data and come back as (x1, x2) spans for the
    caller to draw. Pure — inputs are not modified.
    """
    flux_out = [float(f) for f in flux]
    spans = []
    for zone in exclusions:
        x1, x2, method = zone["x1_wl_a"], zone["x2_wl_a"], zone["method"]
        if method == "overlay":
            spans.append((x1, x2))
            continue
        inside = [i for i, w in enumerate(wls) if x1 <= w <= x2]
        if not inside:
            continue
        before = inside[0] - 1
        after = inside[-1] + 1
        if (method == "interpolate" and before >= 0 and after < len(wls)):
            w1, f1 = wls[before], flux_out[before]
            w2, f2 = wls[after], flux_out[after]
            for i in inside:
                t = (wls[i] - w1) / (w2 - w1)
                flux_out[i] = f1 + t * (f2 - f1)
        else:                       # mask, or an unanchored interpolation
            for i in inside:
                flux_out[i] = float("nan")
    return flux_out, spans


# ---------------------------------------------------------------------------
# Self-check — synthetic ingest into a temp DB (guidelines §5)
# ---------------------------------------------------------------------------

def _selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.db")
        conn = connect(path)

        caph = dict(gaia_dr3_source_id=395234439752169344,
                    main_id="* bet Cas", label="Caph", sp_type="F2III",
                    otype="dS*", ra_deg=2.29452, dec_deg=59.14978,
                    pos_epoch_jyear=2000.0)
        ds = dict(fits_path="C:/obs/caph.fits", fits_sha256="abc123",
                  date_obs="2026-07-12T01:23:45", exptime_s=30.0,
                  telescope="Test scope", instrument="Test camera",
                  object_name="Caph", site_lat_deg=0.0, site_lon_deg=0.0,
                  site_elev_m=0.0)
        run = dict(run_utc="2026-07-12T02:00:00+00:00", git_hash="deadbee",
                   config_json=json.dumps({"angle": 1.5}))
        spec = dict(source_x=100.0, source_y=200.0, free_selection=True,
                    match_sep_arcsec=2.1)
        smp = [(0, 10.0, None, None, None, FLAG_NO_CAL),
               (1, None, 4000.0, None, None, FLAG_CONTAM),
               (2, 30.0, 4010.0, 0.5, 0.01, 0)]

        r1 = ingest_spectrum(conn, star=caph, dataset=ds, run=run,
                             spectrum=spec, samples=smp)
        assert r1["created_star"]

        # Same star via Gaia id, different position → still one star row;
        # same file → dataset reused; new run appended.
        caph2 = dict(caph, ra_deg=2.30000, dec_deg=59.15100)
        r2 = ingest_spectrum(conn, star=caph2, dataset=ds, run=run,
                             spectrum=spec, samples=smp)
        assert not r2["created_star"] and r2["star_id"] == r1["star_id"]
        assert r2["dataset_id"] == r1["dataset_id"]
        assert r2["run_id"] != r1["run_id"]

        # Position-only star (empty cone case), then enriched by a later
        # resolved save at the same position (cone step + NULL backfill).
        anon = dict(ra_deg=100.0, dec_deg=20.0, pos_epoch_jyear=2026.5)
        r3 = ingest_spectrum(conn, star=anon, dataset=ds, run=run,
                             spectrum=spec, samples=smp)
        assert r3["created_star"]
        named = dict(ra_deg=100.0002, dec_deg=20.0002, main_id="HD 1",
                     gaia_dr3_source_id=42, pos_epoch_jyear=2000.0)
        r4 = ingest_spectrum(conn, star=named, dataset=ds, run=run,
                             spectrum=spec, samples=smp)
        assert not r4["created_star"] and r4["star_id"] == r3["star_id"]
        row = conn.execute("SELECT gaia_dr3_source_id, main_id FROM stars "
                           "WHERE star_id = ?", (r3["star_id"],)).fetchone()
        assert row == (42, "HD 1")

        # Cone hit with conflicting Gaia ids → two real stars, two rows.
        twin = dict(named, gaia_dr3_source_id=43, main_id="HD 2")
        r5 = ingest_spectrum(conn, star=twin, dataset=ds, run=run,
                             spectrum=spec, samples=smp)
        assert r5["created_star"]

        # Name match rejected when Gaia ids differ on both sides.
        imposter = dict(ra_deg=250.0, dec_deg=-30.0, main_id="HD 1",
                        gaia_dr3_source_id=99)
        r6 = ingest_spectrum(conn, star=imposter, dataset=ds, run=run,
                             spectrum=spec, samples=smp)
        assert r6["created_star"]

        # 6 ingests, all appended.
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 6
        assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 18

        # Empty samples refused, nothing written (one-transaction rule).
        n_stars = conn.execute("SELECT COUNT(*) FROM stars").fetchone()[0]
        try:
            ingest_spectrum(conn, star=caph, dataset=ds, run=run,
                            spectrum=spec, samples=[])
            raise AssertionError("empty ingest not refused")
        except ValueError:
            pass
        assert conn.execute(
            "SELECT COUNT(*) FROM stars").fetchone()[0] == n_stars

        # Exclusion zones: vocabulary enforced, endpoints ordered, listed
        # bluest-first, deletable.
        sid = r1["spectrum_id"]
        try:
            add_exclusion(conn, sid, 5000.0, 5100.0, "vaporise")
            raise AssertionError("unknown method not refused")
        except ValueError:
            pass
        e1 = add_exclusion(conn, sid, 5250.0, 5150.0, "mask")   # swapped
        add_exclusion(conn, sid, 4000.0, 4100.0, "overlay")
        zones = exclusions_for(conn, sid)
        assert [z["method"] for z in zones] == ["overlay", "mask"], zones
        assert zones[1]["x1_wl_a"] == 5150.0, zones
        delete_exclusion(conn, e1)
        assert len(exclusions_for(conn, sid)) == 1

        # apply_exclusions: mask NaNs the zone, interpolate bridges it
        # linearly, an edge zone with no anchor falls back to mask,
        # overlay only reports its span. Inputs stay untouched.
        wls = [4000.0, 4010.0, 4020.0, 4030.0, 4040.0]
        flux = [1.0, 5.0, 9.0, 2.0, 3.0]
        out, spans = apply_exclusions(
            wls, flux, [{"x1_wl_a": 4005, "x2_wl_a": 4025, "method": "mask"}])
        assert out[0] == 1.0 and out[3] == 2.0 and flux[1] == 5.0
        assert all(o != o for o in out[1:3])            # NaN != NaN
        out, spans = apply_exclusions(
            wls, flux,
            [{"x1_wl_a": 4005, "x2_wl_a": 4025, "method": "interpolate"}])
        assert abs(out[1] - (1 + (2 - 1) / 3)) < 1e-9, out
        assert abs(out[2] - (1 + 2 * (2 - 1) / 3)) < 1e-9, out
        out, spans = apply_exclusions(
            wls, flux,
            [{"x1_wl_a": 3990, "x2_wl_a": 4015, "method": "interpolate"}])
        assert out[0] != out[0] and out[1] != out[1], out   # no left anchor
        out, spans = apply_exclusions(
            wls, flux,
            [{"x1_wl_a": 4010, "x2_wl_a": 4020, "method": "overlay"}])
        assert out == flux and spans == [(4010, 4020)], (out, spans)
        conn.close()

        # Readers open read-only and cannot write (§5).
        ro = connect(path, readonly=True)
        assert ro.execute("SELECT COUNT(*) FROM stars").fetchone()[0] == 4
        try:
            ro.execute("DELETE FROM stars")
            raise AssertionError("readonly connection allowed a write")
        except sqlite3.OperationalError:
            pass
        ro.close()

        # Refuse a DB newer than the code.
        conn = sqlite3.connect(path)
        conn.execute("UPDATE schema_version SET version = 999")
        conn.commit()
        conn.close()
        try:
            connect(path)
            raise AssertionError("future schema not refused")
        except RuntimeError:
            pass

    assert gaia_dr3_id(["HD 1", "Gaia DR3 12345"]) == 12345
    assert gaia_dr3_id(["Gaia DR2 9", "HDS 1"]) is None
    print("spectra_db self-check OK")


if __name__ == "__main__":
    _selfcheck()
