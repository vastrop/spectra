"""Be-star catalogue browser dialog (Jaschek & Egret 1982, VizieR III/67A).

Lists the catalogue in the shared sortable browser (catalog_browser.py):
designations (BD, MWC/AS, HD, TYC, TIC), V, spectral type, JNOW position,
constellation (astropy's IAU boundary grid, same as db/spectra_browser.py)
and an Hα-emission-chance rating derived from the secondary spectral
descriptors. J2000 stays in the row data (goto/scheduler), not on screen.

The catalogue ships as ReferenceLibrary/be_star_catalog.csv (tracked, and
bundled by spectrum_explorer.spec). It is regenerated — VizieR fetch, B1950
to ICRS, constellation derivation, SIMBAD TAP crossmatch for TYC/TIC — by
running this module standalone with --refresh.

Run:      py -3.13 explorer/be_star_dialog.py [--refresh]
Check:    py -3.13 explorer/be_star_dialog.py --selfcheck
"""

from __future__ import annotations

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorer import catalog_browser as cb   # noqa: E402

CATALOG = "III/67A"
CACHE = cb.reference_csv("be_star_catalog.csv")
FIELDS = ("bd", "mwc", "hd", "tyc", "tic", "vmag", "sptype", "ra_deg",
          "dec_deg", "constellation")
COLUMNS = ("bd", "mwc", "hd", "tyc", "tic", "vmag", "sptype",
           "ra_jnow", "dec_jnow", "constellation", "ha", "nspec")
HEADINGS = {"bd": ("BD", 110), "mwc": ("MWC/AS", 90), "hd": ("HD", 70),
            "tyc": ("TYC", 105), "tic": ("TIC", 85),
            "vmag": ("V", 55), "sptype": ("SpType", 60),
            "ra_jnow": ("RA JNOW", 85), "dec_jnow": ("Dec JNOW", 85),
            "constellation": ("Const", 60), "ha": ("Hα", 60),
            "nspec": ("Spec", 45)}
TITLE = "Be stars — Jaschek & Egret 1982"
SIZE = "1280x600"


def _table_to_rows(table):
    """Astropy table (III/67A schema, B1950 sexagesimal) -> list of dicts."""
    from astropy import units as u
    from astropy.coordinates import FK4, SkyCoord, get_constellation
    import numpy as np

    keep = [r for r in table if cb.cell(r, "RAB1950") and cb.cell(r, "DEB1950")]
    coords = SkyCoord([cb.cell(r, "RAB1950") for r in keep],
                      [cb.cell(r, "DEB1950") for r in keep],
                      unit=(u.hourangle, u.deg),
                      frame=FK4(equinox="B1950")).icrs
    consts = np.atleast_1d(get_constellation(coords, short_name=True))
    rows = []
    for row, coord, const in zip(keep, coords, consts):
        rows.append({
            "bd": cb.cell(row, "Number"),
            "mwc": cb.cell(row, "BeStar"),
            "hd": cb.cell(row, "HD"),
            "vmag": cb.cell(row, "Vmag"),
            "sptype": cb.cell(row, "SpType"),
            "ra_deg": f"{coord.ra.deg:.4f}",
            "dec_deg": f"{coord.dec.deg:+.4f}",
            "constellation": str(const),
        })
    return _dedupe(rows)


def _dedupe(rows):
    """Collapse III/67A's doubled entries (11 stars appear twice, same
    position, slightly different photometry/type). One row per exact
    position, preferring the entry whose type carries the emission
    descriptor — that is the one this browser exists for."""
    best = {}
    for row in rows:
        key = (row["ra_deg"], row["dec_deg"])
        kept = best.get(key)
        if kept is None or (not ha_chance(kept["sptype"])
                            and ha_chance(row["sptype"])):
            best[key] = row
    return list(best.values())


def _target_id(row):
    """SIMBAD identifier for a catalogue row: HD when present, else BD/CPD.

    The Number column mixes proper designations ("BD+62 11", "CD-28 ...")
    with a bare zero-padded zone form ("+63-00395" = BD+63 395) and "CP-"
    for CPD — normalise those or SIMBAD won't resolve them.
    """
    if row["hd"]:
        return f"HD {row['hd']}"
    bd = re.sub(r"\s+", " ", row["bd"])
    zone = re.match(r"^([+-]\d+)-0*(\d+)$", bd)
    if zone:
        return f"BD{zone.group(1)} {zone.group(2)}"
    if bd.startswith("CP-"):
        return "CPD" + bd[2:]
    return bd


def fetch_catalog():
    from astroquery.vizier import Vizier
    vizier = Vizier(columns=["Number", "BeStar", "HD", "Vmag", "SpType",
                             "RAB1950", "DEB1950"], row_limit=-1)
    tables = vizier.get_catalogs(CATALOG)
    if not tables:
        raise RuntimeError(f"VizieR returned no table for {CATALOG}")
    rows = _table_to_rows(tables[0])
    cb.crossmatch_ids(rows, _target_id)
    return rows


def load_rows(refresh=False):
    return cb.load_rows(CACHE, FIELDS, fetch_catalog, refresh)


def ha_chance(sptype):
    """Hα-emission likelihood from the 1982 secondary descriptors, as stars.

    ne/nne/e+/shell (broad-lined or strong emission) beat plain e/ep, which
    beat intermittent/uncertain (e)/e:; an early subtype (B0-B2) adds one —
    emission fraction and strength peak there. NB the descriptor column is
    incomplete: gamma Cas itself is a bare "B0.5 IV", so blank means
    unrecorded, not absent.
    """
    sp = sptype.replace("shell", "sh")   # keep its 'e' out of the e-match
    if re.search(r"e\+|nn?e|\bsh\b", sp):
        base = 3
    elif re.search(r"\(e\)|e:", sp):
        base = 1
    elif "e" in sp:
        base = 2
    else:
        return ""
    sub = re.match(r"B(\d(?:\.\d)?)", sp)
    early = sub is not None and float(sub.group(1)) <= 2
    return "★" * (base + (1 if early else 0))


def _prepare(rows):
    cb.add_display_cols(rows)
    cb.mark_have_spectra(rows)
    for row in rows:
        row["ha"] = ha_chance(row["sptype"])
    return rows


class BeStarDialog(cb.CatalogDialog):
    """Modeless Be-star browser; contract in catalog_browser.CatalogDialog."""

    def __init__(self, parent, goto=None):
        rows = _prepare(load_rows())
        super().__init__(parent, f"{TITLE}, {len(rows)} entries", SIZE,
                         rows, COLUMNS, HEADINGS,
                         goto=goto, name_of=_target_id)


def _show(rows):
    _prepare(rows)
    cb.run_standalone(f"{TITLE}, {len(rows)} entries", SIZE,
                      rows, COLUMNS, HEADINGS)


def _selfcheck():
    from astropy.table import Table

    # gamma Cas (HD 5394), B1950 00 53 40.2 +60 26 46 -> J2000 ~14.177 +60.717
    table = Table(rows=[("BD+59  144", "MWC     9", "5394", "2.47",
                         "B0.5 IV", "00 53 40.2", "+60 26 46"),
                        ("nocoord", "", "1", "9.99", "B2 V", "", ""),
                        # III/67A-style doubled entry: same position twice,
                        # non-emission type first — dedupe must keep the 'e'.
                        ("BD+46 1203", "MWC   537", "50658", "5.8",
                         "B8 III", "06 48 20", "+46 23 00"),
                        ("BD+46 1203", "MWC   537", "50658", "5.8",
                         "B6 IV-V e", "06 48 20", "+46 23 00")],
                  names=("Number", "BeStar", "HD", "Vmag", "SpType",
                         "RAB1950", "DEB1950"))
    rows = _table_to_rows(table)
    assert len(rows) == 2, "coordinate-less dropped, doubled entry collapsed"
    star, dup = rows
    assert dup["sptype"] == "B6 IV-V e", dup
    assert star["constellation"] == "Cas", star
    assert abs(float(star["ra_deg"]) - 14.177) < 0.01, star
    assert abs(float(star["dec_deg"]) - 60.717) < 0.01, star

    for sptype, expected in (("B0.5 IV", ""),          # gamma Cas: unrecorded
                             ("B2 V,ne", "★★★★"),      # broad emission + early
                             ("B2.5 III ne", "★★★"),   # broad emission, mid
                             ("B7: e sh", "★★★"),      # shell, late
                             ("B2 V P shell", "★★★★"), # shell, early
                             ("B1 V:,ep", "★★★"),      # plain emission + early
                             ("B3 V,e", "★★"),         # plain emission, mid
                             ("B4 V (e)", "★"),        # intermittent
                             ("B1.5 V :e:,nn", "★★")): # uncertain + early
        assert ha_chance(sptype) == expected, (sptype, ha_chance(sptype))

    # SIMBAD id normalisation: HD preferred, zone/CP forms translated.
    assert _target_id({"hd": "5394", "bd": "BD+59  144"}) == "HD 5394"
    assert _target_id({"hd": "", "bd": "BD+62    11"}) == "BD+62 11"
    assert _target_id({"hd": "", "bd": "+63-00395"}) == "BD+63 395"
    assert _target_id({"hd": "", "bd": "CP-56  154"}) == "CPD-56 154"

    # The shipped catalogue must be present and on the current schema —
    # the frozen build has no network fallback worth trusting.
    assert os.path.exists(CACHE), CACHE
    with open(CACHE, newline="", encoding="utf-8") as handle:
        shipped = list(csv.DictReader(handle))
    assert len(shipped) > 1000 and set(FIELDS) <= set(shipped[0]), CACHE
    print("be_star_dialog self-check OK")


if __name__ == "__main__":
    cb.cli(__doc__, load_rows, _show, _selfcheck)
