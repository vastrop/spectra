"""S-type-star catalogue browser (Stephenson 1984 CGSS, VizieR III/168).

The General Catalogue of Galactic S Stars, cut to v <= 10 (rows without a
usable magnitude are dropped with it — unknown is treated as faint here,
documented trade-off). ZrO bands instead of TiO are the SA100 signature.
The catalogue's native positions are 1900.0, but VizieR precomputes ICRS
(_RA.icrs/_DE.icrs), and those are the columns read here.

The catalogue ships as ReferenceLibrary/s_star_catalog.csv (tracked,
bundled by spectrum_explorer.spec); regenerate with --refresh.

Run:      py -3.13 explorer/s_star_dialog.py [--refresh]
Check:    py -3.13 explorer/s_star_dialog.py --selfcheck
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorer import catalog_browser as cb   # noqa: E402

CATALOG = "III/168/catalog"
MAG_LIMIT = 10.0
CACHE = cb.reference_csv("s_star_catalog.csv")
FIELDS = ("css", "names", "hd", "tyc", "tic", "vmag", "sptype",
          "ra_deg", "dec_deg", "constellation")
COLUMNS = ("css", "names", "hd", "tyc", "tic", "vmag", "sptype",
           "ra_jnow", "dec_jnow", "constellation", "nspec")
HEADINGS = {"css": ("CSS", 55), "names": ("Names", 130), "hd": ("HD", 70),
            "tyc": ("TYC", 105), "tic": ("TIC", 85), "vmag": ("v", 55),
            "sptype": ("SpType", 110),
            "ra_jnow": ("RA JNOW", 85), "dec_jnow": ("Dec JNOW", 85),
            "constellation": ("Const", 60), "nspec": ("Spec", 45)}
TITLE = f"S-type stars (v ≤ {MAG_LIMIT:g}) — Stephenson CGSS"
SIZE = "1210x600"


def _bright(row):
    try:
        return float(cb.cell(row, "vmag")) <= MAG_LIMIT
    except ValueError:
        return False


def _table_to_rows(table):
    keep = [r for r in table
            if cb.cell(r, "_RA.icrs") and cb.cell(r, "_DE.icrs")
            and _bright(r)]
    sky = cb.sky_fields([cb.cell(r, "_RA.icrs") for r in keep],
                        [cb.cell(r, "_DE.icrs") for r in keep])
    return [{"css": cb.cell(r, "CSS"),
             "names": cb.cell(r, "Names"),
             "vmag": cb.cell(r, "vmag"),
             "sptype": cb.cell(r, "SpType"),
             **s} for r, s in zip(keep, sky)]


def fetch_catalog():
    from astroquery.vizier import Vizier
    vizier = Vizier(columns=["CSS", "_RA.icrs", "_DE.icrs", "vmag", "SpType",
                             "Names"], row_limit=-1)
    tables = vizier.get_catalogs(CATALOG)
    if not tables:
        raise RuntimeError(f"VizieR returned no table for {CATALOG}")
    rows = _table_to_rows(tables[0])
    # The Names column is mostly bare catalogue numbers ("4350(Md)"), not
    # designations — CSS is what SIMBAD actually resolves here.
    cb.crossmatch_ids(rows, _name_of)
    return rows


def load_rows(refresh=False):
    return cb.load_rows(CACHE, FIELDS, fetch_catalog, refresh)


def _prepare(rows):
    cb.add_display_cols(rows)
    cb.mark_have_spectra(rows)
    return rows


def _name_of(row):
    return f"CSS {row['css']}"


class SStarDialog(cb.CatalogDialog):
    """Modeless S-star browser; contract in catalog_browser."""

    def __init__(self, parent, goto=None):
        rows = _prepare(load_rows())
        super().__init__(parent, f"{TITLE}, {len(rows)} entries", SIZE,
                         rows, COLUMNS, HEADINGS,
                         goto=goto, name_of=_name_of)


def _show(rows):
    _prepare(rows)
    cb.run_standalone(f"{TITLE}, {len(rows)} entries", SIZE,
                      rows, COLUMNS, HEADINGS)


def _selfcheck():
    from astropy.table import Table

    # chi Cyg region star at J2000 19 50 33.9 +32 54 51 -> Cyg; the faint
    # and magnitude-less rows fall to the brightness cut.
    table = Table(rows=[("123", "19 50 33.9", "+32 54 51", "5.2",
                         "S6,2e", "chi Cyg"),
                        ("124", "19 50 00.0", "+32 00 00", "13.0", "S", ""),
                        ("125", "19 51 00.0", "+33 00 00", "", "S", "")],
                  names=("CSS", "_RA.icrs", "_DE.icrs", "vmag", "SpType",
                         "Names"))
    rows = _table_to_rows(table)
    assert len(rows) == 1, "faint and magnitude-less rows must be dropped"
    star = rows[0]
    assert star["constellation"] == "Cyg" and star["names"] == "chi Cyg", star
    assert _name_of(star) == "CSS 123"

    assert os.path.exists(CACHE), CACHE
    with open(CACHE, newline="", encoding="utf-8") as handle:
        shipped = list(csv.DictReader(handle))
    assert len(shipped) > 100 and set(FIELDS) <= set(shipped[0]), CACHE
    # The crossmatch is the point of the HD/TYC/TIC columns: SIMBAD knows
    # most of these, so a wholesale blank means the refresh silently failed.
    matched = sum(1 for r in shipped if r["hd"] or r["tyc"] or r["tic"])
    assert matched > len(shipped) // 3, f"only {matched} rows cross-matched"
    print("s_star_dialog self-check OK")


if __name__ == "__main__":
    cb.cli(__doc__, load_rows, _show, _selfcheck)
