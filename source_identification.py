"""
source_identification.py
========================

Standalone module for identifying the stellar sources in a spectroscopy frame.

Two independent stages, usable separately:

1. ``solve_wcs(...)``        — run ASTAP on a (rotated) 2-D mono frame and return
                              an astropy ``WCS`` object, or ``None`` on failure.
2. ``identify_sources(...)`` — convert measured zero-order centroids to sky
                              coordinates via that WCS and resolve each against
                              SIMBAD (cone search), returning ``SourceMatch``
                              objects (or ``None`` per source when unmatched).

Design notes
------------
* Mono only.  The Spectrum Explorer pipeline works on a single derotated mono
  frame; RGB/green-channel handling from the astrometric project is not needed.
* The frame passed to ``solve_wcs`` MUST be the *rotated* frame, so the returned
  WCS shares a pixel coordinate system with the ``top_sources`` centroids that
  ``identify_sources`` consumes.  The pipeline's *own* centroids are fed
  through the WCS; ASTAP's internal star list only establishes the plate
  solution.
* SIMBAD access goes through astroquery.  The astroquery >=0.4.8 SIMBAD
  interface renamed several votable fields; column access here is defensive
  (tries known aliases) so it survives 0.4.8 .. 0.4.11 without pinning names.
* Every network / subprocess path degrades gracefully to ``None`` so the GUI
  can fall back to bare source numbers.

This module has no Tkinter / GUI dependencies.
"""

from __future__ import annotations

import os
import re
import sys
import logging
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# astroquery / astropy.coordinates are imported lazily inside the query path so
# that ``solve_wcs`` (and module import) work even if astroquery is absent.

# Failures still degrade to None for the GUI, but they are logged here so
# "SIMBAD is down" and "ASTAP exited 1" stay distinguishable from "no match".
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_ASTAP_PATH = "C:/Program Files/astap/astap.exe"
ASTAP_TIMEOUT = 60          # seconds, subprocess hard timeout
SIMBAD_TIMEOUT = 30         # seconds, network query timeout
DEFAULT_CONE_RADIUS_ARCSEC = 10.0

# Brightness-gated escalation (identify_sources).  A saturated bright star's
# centroid is unreliable — DAO's fit has nothing to grip on a flat-topped
# core, and the SA100's zero order elongates chromatically along the
# dispersion axis — so the projected position can miss the 10" cone.
# Widening the cone for everyone would invite faint Gaia interlopers, but
# the bloat itself certifies brightness: a star that flat-tops a short sub
# must be bright, and bright stars are sparse on the sky (V<10 expectation
# in a 30" cone is ~0.007 even in the galactic plane).  So the wide retry
# accepts only candidates SIMBAD knows to be bright: wide+bright is safer
# than tight+anything.
WIDE_CONE_RADIUS_ARCSEC = 30.0
WIDE_CONE_VMAX = 10.0

# Catalog-prefix preference for the compact overlay label, best first.
_LABEL_PREFIX_PRIORITY = ("HD", "HIP", "TYC", "HR", "BD", "SAO", "GAIA", "TIC")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SourceMatch:
    """A resolved catalog identification for one zero-order centroid."""
    main_id: str                       # SIMBAD primary identifier
    label: str                         # compact label for the overlay (HD/TYC/…)
    ra_deg: float                      # queried sky position (from WCS)
    dec_deg: float
    sep_arcsec: float                  # separation of match from queried point
    sp_type: str = ""                  # spectral type, if known
    otype: str = ""                    # object type, if known
    mag: Optional[float] = None        # representative magnitude (V/G), if known
    all_ids: list[str] = field(default_factory=list)   # alternate denominations
    # Catalog (SIMBAD ICRS J2000) position of the matched object — preferred
    # over the queried WCS position as DB ground-truth identity when present.
    cat_ra_deg: Optional[float] = None
    cat_dec_deg: Optional[float] = None

    def info_text(self) -> str:
        """Multi-line summary for the right-panel info box."""
        lines = [self.main_id]
        if self.sp_type:
            lines.append(f"Sp. type : {self.sp_type}")
        if self.otype:
            lines.append(f"Type     : {self.otype}")
        if self.mag is not None:
            lines.append(f"Mag      : {self.mag:.2f}")
        lines.append(f"Sep      : {self.sep_arcsec:.2f}\"")
        if self.all_ids:
            others = [i for i in self.all_ids if i != self.main_id]
            if others:
                lines.append("Also: " + ", ".join(others[:8]))
        return "\n".join(lines)


# ===========================================================================
# Stage 1 — ASTAP plate solve
# ===========================================================================

def _parse_astap_solution(ini_file: Path) -> Optional[WCS]:
    """Parse ASTAP's ``.ini`` solution file into an astropy WCS.

    The ``.ini`` uses plain ``KEY=value`` lines, with ``PLTSOLVD=T`` marking a
    successful solve.  Values are deliberately stripped of any FITS-style
    ``/ comment`` suffix and surrounding quotes before casting — harmless for
    the numeric keys consumed here, and robust if ASTAP ever carries
    comments over from the FITS keywords.  The file carries the
    CRPIX/CRVAL/CD matrix but no CTYPE, so RA---TAN / DEC--TAN is supplied
    here, which is what ASTAP's TAN solution implies.

    Returns None if the file is missing, the solve flag is not set, the required
    keywords are absent, or the assembled WCS is not celestial.
    """
    if not ini_file.exists():
        return None

    raw: dict = {}
    try:
        with open(ini_file, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.split("/")[0].strip().strip("'").strip()
                cast: object = value
                try:
                    if any(ch in value for ch in ".eE"):
                        cast = float(value)
                    else:
                        cast = int(value)
                except ValueError:
                    cast = value
                raw[key] = cast
    except OSError:
        return None

    # Success flag: PLTSOLVD=T (string 'T' after the casting above).
    solved = str(raw.get("PLTSOLVD", "")).upper().startswith("T")
    if not solved:
        return None

    required = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"]
    if not all(k in raw for k in required):
        return None
    has_cd = all(k in raw for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
    has_cdelt = all(k in raw for k in ("CDELT1", "CDELT2"))
    if not (has_cd or has_cdelt):
        return None

    try:
        hdr = fits.Header()
        hdr["CTYPE1"] = raw.get("CTYPE1", "RA---TAN")
        hdr["CTYPE2"] = raw.get("CTYPE2", "DEC--TAN")
        hdr["CRVAL1"] = float(raw["CRVAL1"])
        hdr["CRVAL2"] = float(raw["CRVAL2"])
        hdr["CRPIX1"] = float(raw["CRPIX1"])
        hdr["CRPIX2"] = float(raw["CRPIX2"])
        if has_cd:
            for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
                hdr[k] = float(raw[k])
        else:
            hdr["CDELT1"] = float(raw["CDELT1"])
            hdr["CDELT2"] = float(raw["CDELT2"])
            if "CROTA2" in raw:
                hdr["CROTA2"] = float(raw["CROTA2"])
        wcs = WCS(hdr)
    except Exception:
        return None

    if not wcs.has_celestial:
        return None
    return wcs


def _sexagesimal(value) -> float:
    """'01 33 28' / '+59:30:02' → decimal (sign taken from the string)."""
    parts = [abs(float(x)) for x in str(value).replace(":", " ").split()]
    dec = parts[0] + parts[1] / 60.0 + (parts[2] if len(parts) > 2 else 0.0) / 3600.0
    return -dec if str(value).strip().startswith("-") else dec


def _astap_hint_flags(header, height_px: int) -> list:
    """NINA-parity ASTAP flags derived from the frame header.

    Mirrors NINA's ASTAPSolver.cs argument builder: -fov (field height,
    degrees), -z (downsample, 0 = auto), -s (star limit), and — when the
    header carries a position — -r/-ra/-spd.  ASTAP unit traps, straight
    from NINA's source: -ra is in HOURS, -spd is south-pole distance
    (Dec + 90°), and values must be dot-decimal (f-strings are
    locale-independent, matching NINA's InvariantCulture).

    Every flag degrades independently: a header without scale keys still
    gets -z/-s, one without a position still gets -fov, and a bare header
    yields ASTAP's own defaults — never worse than the flagless call.
    """
    flags = []
    try:  # field height: pixel scale (arcsec/px) × height, in degrees
        scale = 206.265 * float(header["XPIXSZ"]) / float(header["FOCALLEN"])
        flags += ["-fov", f"{height_px * scale / 3600.0:.6f}"]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    flags += ["-z", "0", "-s", "500"]   # NINA defaults: auto binning, 500 stars
    ra_hours = dec_deg = None
    try:  # numeric keys first (degrees), sexagesimal OBJCT* as fallback
        ra_hours = float(header["RA"]) / 15.0
        dec_deg = float(header["DEC"])
    except (KeyError, TypeError, ValueError):
        try:
            ra_hours = _sexagesimal(header["OBJCTRA"])       # already hours
            dec_deg = _sexagesimal(header["OBJCTDEC"])
        except (KeyError, IndexError, TypeError, ValueError):
            ra_hours = dec_deg = None
    if ra_hours is not None and dec_deg is not None:
        flags += ["-r", "30",
                  "-ra", f"{ra_hours:.6f}",
                  "-spd", f"{dec_deg + 90.0:.6f}"]
    return flags


def solve_wcs(rotated_data: np.ndarray,
              fits_header,
              astap_path: str = DEFAULT_ASTAP_PATH) -> Optional[WCS]:
    """Plate-solve a 2-D mono frame with ASTAP and return an astropy WCS.

    Parameters
    ----------
    rotated_data : 2-D ndarray
        The *rotated* mono frame whose pixel grid the returned WCS describes.
    fits_header : astropy Header
        Source header, used only for position / FOV hints (OBJCTRA/OBJCTDEC,
        RA/DEC, XPIXSZ, FOCALLEN, DATE-OBS, TELESCOP, INSTRUME).
    astap_path : str
        Path to the ASTAP executable.

    Returns
    -------
    astropy.wcs.WCS or None
        A celestial WCS on success; None on any failure (missing executable,
        timeout, non-zero exit, unparseable / non-celestial solution).
    """
    if not os.path.isfile(astap_path):
        log.warning("ASTAP executable not found: %s", astap_path)
        return None
    if rotated_data is None or rotated_data.ndim != 2:
        return None

    temp_dir = None
    try:
        temp_base = Path(tempfile.gettempdir()) / "astap_spectro_solving"
        temp_base.mkdir(exist_ok=True)
        # Any surviving solve_* sibling means a previous run never cleaned up
        # (killed mid-solve, or ASTAP still held the FITS when rmtree ran);
        # sweep them now, before creating this run's dir.
        try:
            for stale in temp_base.glob("solve_*"):
                _safe_rmtree(stale)
        except Exception:
            pass
        temp_dir = Path(tempfile.mkdtemp(prefix="solve_", dir=temp_base))
        temp_fits = temp_dir / "temp_solve.fits"

        hdu = fits.PrimaryHDU(np.ascontiguousarray(rotated_data,
                                                   dtype=np.float32))
        # Copy the position / scale hints ASTAP needs.  These are essential:
        # without OBJCTRA/OBJCTDEC + XPIXSZ/FOCALLEN, ASTAP falls back to a
        # blind solve and fails (RC 1) on a narrow field.  This mirrors what
        # the GUI hands ASTAP (the original header).
        hints = {}
        for key in ("OBJCTRA", "OBJCTDEC", "RA", "DEC",
                    "XPIXSZ", "YPIXSZ", "FOCALLEN", "FOCALRAT",
                    "DATE-OBS", "TELESCOP", "INSTRUME"):
            if key in fits_header:
                try:
                    hdu.header[key] = fits_header[key]
                    hints[key] = fits_header[key]
                except Exception:
                    pass
        hdu.writeto(temp_fits, overwrite=True)

        # ── Build the command ────────────────────────────────────────
        # Each flag is its own list element.  The explicit flags mirror
        # NINA's ASTAPSolver.cs: without -r/-ra/-spd, ASTAP spirals out to a
        # 180° search radius on failure, which costs minutes; without -z it
        # detects at 1x1 on the full-res frame, where 2x binning is what
        # lets it find soft SA100 stars.  The hint units are easy to get
        # wrong and make the solve fail: -ra takes HOURS, -spd takes Dec+90,
        # and both need dot-decimal formatting.  _astap_hint_flags honours
        # all three.  -update writes the solution sidecars.
        cmd = ([astap_path, "-f", str(temp_fits), "-solve", "-update"]
               + _astap_hint_flags(fits_header, rotated_data.shape[0]))
        # For the failure logs: the exact command and the hints that made
        # it into the temp FITS, so a failing solve can be replayed and
        # compared against another ASTAP setup (e.g. NINA's) by hand.
        cmd_str = subprocess.list2cmdline(cmd)
        hints_str = (", ".join(f"{k}={v}" for k, v in hints.items())
                     or "NONE — blind solve")

        try:
            # CREATE_NO_WINDOW: without it a frozen --noconsole build flashes
            # a console per solve (Windows-only app, so unconditional).
            result = subprocess.run(
                cmd, cwd=str(temp_dir),
                capture_output=True, text=True, timeout=ASTAP_TIMEOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
        except subprocess.TimeoutExpired:
            log.warning("ASTAP timed out after %ds | cmd: %s | hints: %s",
                        ASTAP_TIMEOUT, cmd_str, hints_str)
            return None

        # The GUI astap.exe is silent on stdout/stderr; on failure its
        # reason (ERROR=/WARNING= lines, PLTSOLVD=F) lands in the .ini
        # sidecar instead — read it before the finally-rmtree eats it.
        def _ini_tail():
            try:
                text = temp_fits.with_suffix(".ini").read_text(
                    errors="replace")
                return " ".join(text.split())[-300:] or "(empty .ini)"
            except OSError:
                return "(no .ini written)"

        if result.returncode != 0:
            # ASTAP reports its reason on stdout more often than stderr.
            tail = (result.stderr or result.stdout or "").strip()[-300:]
            log.warning("ASTAP solve failed (exit %d): %s | ini: %s | "
                        "cmd: %s | hints: %s",
                        result.returncode, tail, _ini_tail(),
                        cmd_str, hints_str)
            return None

        # ASTAP writes the solution to a .ini (PLTSOLVD=T + WCS keywords).
        wcs = _parse_astap_solution(temp_fits.with_suffix(".ini"))
        if wcs is None:
            log.warning("ASTAP exited 0 but no usable solution | ini: %s | "
                        "cmd: %s | hints: %s",
                        _ini_tail(), cmd_str, hints_str)
        return wcs

    except Exception:
        log.warning("ASTAP solve raised", exc_info=True)
        return None
    finally:
        if temp_dir is not None:
            _safe_rmtree(temp_dir)


def _safe_rmtree(path: Path) -> None:
    try:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


# ===========================================================================
# Stage 2 — SIMBAD cone identification
# ===========================================================================

def _build_simbad():
    """Construct a configured astroquery Simbad client, or None if unavailable.

    Requests sp_type, otype, ids, and a magnitude.  Field names differ across
    astroquery versions, so a generous superset is requested and failures of
    individual add_votable_fields calls are tolerated.
    """
    try:
        from astroquery.simbad import Simbad
    except Exception as e:
        # Not just ImportError: astroquery reads CITATION and simbad's
        # query_criteria_fields.json *at import*, so a frozen build missing
        # those data files raises FileNotFoundError here, which would
        # otherwise surface as a silent "0 identified".
        log.warning("SIMBAD unavailable (astroquery import failed): %s", e)
        return None

    sim = Simbad()
    try:
        sim.TIMEOUT = SIMBAD_TIMEOUT
    except Exception:
        pass
    sim.ROW_LIMIT = 50

    # Modern (astroquery >=0.4.8) field names only.  'V' only populates when
    # the object carries that band, so it stays unusable as a generic
    # ranking key (matches still rank by separation) — but that sparseness
    # is exactly what the wide-cone escalation gate wants: "no known V"
    # reads as "not provably bright" and the candidate is rejected.  Bright
    # stars — the only ones the gate must admit — essentially always have V
    # in SIMBAD.  When present it also feeds SourceMatch.mag for display.
    for fld in ("sp_type", "otype", "ids", "V"):
        try:
            sim.add_votable_fields(fld)
        except Exception:
            continue
    return sim


def _col(row, *names):
    """Fetch the first present, non-masked column value from a table row."""
    for n in names:
        if n in row.colnames:
            val = row[n]
            try:
                if np.ma.is_masked(val):
                    continue
            except Exception:
                pass
            if val is None:
                continue
            return val
    return None


def _to_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", "replace").strip()
        except Exception:
            return ""
    return str(val).strip()


def _compact_label(main_id: str, all_ids: Sequence[str]) -> str:
    """Pick a short, recognisable label: common name first, then HD/HIP/TYC/…"""
    candidates = [main_id] + list(all_ids)
    # SIMBAD carries proper names as "NAME Vega" identifiers — a common
    # name beats any catalog number for at-a-glance recognition.
    for cid in candidates:
        if cid.upper().startswith("NAME "):
            return " ".join(cid.split()[1:])
    # Variable-star designations ("V* HW Cas") are the standard GCVS-style
    # names amateurs (and LAMOST) know a variable by — rank them right
    # after proper names, ahead of any catalog number.  The "V*" prefix is
    # SIMBAD packaging, not part of the designation, so strip it.
    for cid in candidates:
        if cid.upper().startswith("V* "):
            return " ".join(cid.split()[1:])
    for prefix in _LABEL_PREFIX_PRIORITY:
        for cid in candidates:
            cu = cid.upper().replace(" ", "")
            # Prefix must be followed by the id number (optionally signed,
            # Gaia ids interpose a DRn revision) — bare startswith would let
            # e.g. "HDS 123" masquerade as an HD entry.
            if re.match(rf"{prefix}(?:DR\d)?[-+]?\d", cu):
                # normalise whitespace to a single space for display
                return " ".join(cid.split())
    # fall back to the main id, whitespace-normalised
    return " ".join(main_id.split()) if main_id else ""


def identify_sources(wcs: WCS,
                     centroids_xy: Sequence[tuple[float, float]],
                     radius_arcsec: float = DEFAULT_CONE_RADIUS_ARCSEC,
                     wide_ok: Optional[Sequence[bool]] = None
                     ) -> list[Optional[SourceMatch]]:
    """Resolve each pixel centroid against SIMBAD via the supplied WCS.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        A celestial WCS in the same pixel frame as ``centroids_xy``.
    centroids_xy : sequence of (x, y)
        Zero-order centroid pixel positions (e.g. the explorer's top_sources).
    radius_arcsec : float
        Cone search radius.  Keep tight; matches are ranked by separation.
    wide_ok : sequence of bool, aligned with ``centroids_xy``, optional
        Which sources have earned the wide-cone escalation: when the tight
        cone misses AND the flag is True, one retry runs at
        WIDE_CONE_RADIUS_ARCSEC accepting only SIMBAD-bright candidates
        (V <= WIDE_CONE_VMAX; see the constants' rationale).  Callers flag
        sources whose image evidence says "bright and bloated" — the
        explorer uses the DAO peak.  None (the default) keeps every other
        caller on the tight-cone-only behaviour.

    Returns
    -------
    list of (SourceMatch or None)
        One entry per input centroid, in the same order.  None where the WCS is
        missing, the query failed, or no catalog object fell within the cone.
    """
    n = len(centroids_xy)
    if wcs is None or not getattr(wcs, "has_celestial", False) or n == 0:
        return [None] * n

    sim = _build_simbad()
    if sim is None:
        return [None] * n

    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except Exception:
        return [None] * n

    out: list[Optional[SourceMatch]] = []
    for i, (x, y) in enumerate(centroids_xy):
        try:
            ra_deg, dec_deg = wcs.pixel_to_world_values(float(x), float(y))
            ra_deg = float(ra_deg)
            dec_deg = float(dec_deg)
        except Exception:
            out.append(None)
            continue

        match = _query_one(sim, ra_deg, dec_deg, radius_arcsec,
                           SkyCoord, u)
        if match is None and wide_ok is not None and wide_ok[i]:
            match = _query_one(sim, ra_deg, dec_deg,
                               WIDE_CONE_RADIUS_ARCSEC, SkyCoord, u,
                               vmax=WIDE_CONE_VMAX)
        out.append(match)

    return out


def query_position(ra_deg: float, dec_deg: float,
                   radius_arcsec: float = DEFAULT_CONE_RADIUS_ARCSEC
                   ) -> Optional[SourceMatch]:
    """Resolve one sky position against SIMBAD — no WCS involved.

    The coordinate-based entry point to the same cone machinery as
    ``identify_sources``; used by the DB browser to (re-)identify a
    stored star from its ICRS position with a caller-chosen radius.
    """
    sim = _build_simbad()
    if sim is None:
        return None
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except Exception:
        return None
    return _query_one(sim, float(ra_deg), float(dec_deg),
                      radius_arcsec, SkyCoord, u)


def _rank_candidates(cands, vmax):
    """Pick the best (sep_arcsec, vmag_or_None, payload) candidate.

    vmax None — the normal tight cone: nearest wins, V ignored (it is too
    sparsely populated to rank by; see _build_simbad).  vmax set — the
    wide-cone escalation: only candidates with a KNOWN V <= vmax survive,
    ranked brightest-first (separation as tiebreak) — the bloated star we
    are rescuing is almost certainly the brightest thing in the cone, and
    its centroid error makes separation the less trustworthy key.
    """
    if vmax is None:
        pool = sorted(cands, key=lambda c: c[0])
    else:
        pool = sorted((c for c in cands if c[1] is not None and c[1] <= vmax),
                      key=lambda c: (c[1], c[0]))
    return pool[0] if pool else None


def _query_one(sim, ra_deg, dec_deg, radius_arcsec, SkyCoord, u,
               vmax: Optional[float] = None) -> Optional[SourceMatch]:
    """Cone query around one sky position; return the best-ranked match.

    ``vmax`` switches ranking to the wide-cone brightness gate — see
    _rank_candidates.
    """
    center = SkyCoord(ra_deg, dec_deg, unit="deg")
    try:
        table = sim.query_region(center, radius=radius_arcsec * u.arcsec)
    except Exception as e:
        # Distinguish "SIMBAD unreachable" from a genuine empty cone.
        log.warning("SIMBAD query failed at RA %.5f Dec %.5f: %s",
                    ra_deg, dec_deg, e)
        return None
    if table is None or len(table) == 0:
        return None

    # SIMBAD (astroquery >=0.4.8) returns 'ra'/'dec' as decimal degrees.
    cands = []
    for row in table:
        rra = _col(row, "ra", "RA", "RA_d")
        rdec = _col(row, "dec", "DEC", "DEC_d")
        try:
            mc = SkyCoord(float(rra), float(rdec), unit="deg")
            sep = float(center.separation(mc).arcsec)
        except Exception:
            # unrankable row — never let it become the "best" match
            continue
        try:
            vmag = float(_col(row, "V", "FLUX_V"))
        except (TypeError, ValueError):
            vmag = None
        cands.append((sep, vmag, row))

    best = _rank_candidates(cands, vmax)
    if best is None:
        return None

    sep, vmag, row = best
    try:
        cat_ra = float(_col(row, "ra", "RA", "RA_d"))
        cat_dec = float(_col(row, "dec", "DEC", "DEC_d"))
    except (TypeError, ValueError):
        cat_ra = cat_dec = None
    main_id = _to_str(_col(row, "main_id", "MAIN_ID"))
    sp_type = _to_str(_col(row, "sp_type", "SP_TYPE"))
    otype = _to_str(_col(row, "otype", "OTYPE"))
    ids_raw = _to_str(_col(row, "ids", "IDS"))
    all_ids = [s.strip() for s in ids_raw.split("|") if s.strip()] \
        if ids_raw else []

    return SourceMatch(
        main_id=main_id or "(unnamed)",
        label=_compact_label(main_id, all_ids),
        ra_deg=ra_deg, dec_deg=dec_deg,
        sep_arcsec=sep,
        sp_type=sp_type, otype=otype, mag=vmag,
        all_ids=all_ids,
        cat_ra_deg=cat_ra, cat_dec_deg=cat_dec,
    )


# ===========================================================================
# __main__ test harness
# ===========================================================================

def _main(argv) -> int:
    """Run against a real NINA 2-D FITS to validate solve + cone query.

    Usage:
        python source_identification.py <fits_path> [astap_path] [radius_arcsec]

    Detects sources with DAOStarFinder (so this exercises the same centroid
    path as the explorer), solves the frame, and prints identifications.
    """
    # Pure self-check of the label priority (runs even without a FITS):
    # proper name > variable-star designation > catalog prefixes.
    assert _compact_label("NAME Vega", ["V* HW Cas", "HD 1"]) == "Vega"
    assert _compact_label("Gaia DR3 42", ["V* HW Cas", "HD 1"]) == "HW Cas"
    assert _compact_label("Gaia DR3 42",
                          ["HDS 123", "TYC 1-2-3"]) == "TYC 1-2-3"

    # Pure self-check of the cone ranking: tight mode is nearest-wins with
    # V ignored; the wide-cone gate drops unknown/faint V and prefers the
    # brighter candidate over the nearer one (the rescued star's centroid
    # error makes separation the less trustworthy key).
    assert _rank_candidates([(5.0, None, "a"), (2.0, 14.0, "b")],
                            None)[2] == "b"
    assert _rank_candidates([(2.0, None, "faint"), (20.0, 8.5, "bright")],
                            10.0)[2] == "bright"
    assert _rank_candidates([(2.0, 14.0, "faint")], 10.0) is None
    assert _rank_candidates([(20.0, 8.5, "b1"), (10.0, 9.5, "b2")],
                            10.0)[2] == "b1"
    assert _rank_candidates([], 10.0) is None

    if len(argv) < 2:
        print(_main.__doc__)
        return 2

    fits_path = argv[1]
    astap_path = argv[2] if len(argv) > 2 else DEFAULT_ASTAP_PATH
    radius = float(argv[3]) if len(argv) > 3 else DEFAULT_CONE_RADIUS_ARCSEC

    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    if data is None or data.ndim != 2:
        print(f"Expected a 2-D image in HDU 0; got shape "
              f"{None if data is None else data.shape}")
        return 1

    print(f"Frame: {data.shape}  OBJECT={header.get('OBJECT')}  "
          f"OBJCTRA={header.get('OBJCTRA')}  OBJCTDEC={header.get('OBJCTDEC')}")

    # Detect a few bright sources to use as centroids.
    try:
        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        dao = DAOStarFinder(fwhm=4.0, threshold=5.0 * std)
        srcs = dao(data - median)
        if srcs is None or len(srcs) == 0:
            print("No sources detected.")
            return 1
        srcs.sort("peak", reverse=True)
        srcs = srcs[:5]
        try:
            xs = list(srcs["x_centroid"]); ys = list(srcs["y_centroid"])
        except KeyError:
            xs = list(srcs["xcentroid"]); ys = list(srcs["ycentroid"])
        centroids = list(zip(xs, ys))
    except Exception as e:
        print(f"Detection failed: {e}")
        return 1

    print(f"Detected {len(centroids)} centroids; solving with ASTAP…")
    wcs = solve_wcs(data, header, astap_path=astap_path)
    if wcs is None:
        print("Solve FAILED (no WCS).")
        return 1

    cx, cy = data.shape[1] / 2, data.shape[0] / 2
    csky = wcs.pixel_to_world_values(cx, cy)
    print(f"Solve OK.  Image-center sky = "
          f"RA {float(csky[0]):.5f}  Dec {float(csky[1]):.5f}")

    print(f"Querying SIMBAD (radius {radius}\")…")
    matches = identify_sources(wcs, centroids, radius_arcsec=radius)
    for i, (xy, m) in enumerate(zip(centroids, matches), start=1):
        if m is None:
            print(f"  #{i}  ({xy[0]:.1f},{xy[1]:.1f})  — no match")
        else:
            print(f"  #{i}  ({xy[0]:.1f},{xy[1]:.1f})  -> {m.label}"
                  f"   sep={m.sep_arcsec:.2f}\"  sp={m.sp_type}  "
                  f"otype={m.otype}  mag={m.mag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
