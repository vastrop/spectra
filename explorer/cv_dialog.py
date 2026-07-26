"""Cataclysmic-variable catalogue browser (Downes+ 2006, VizieR V/123A).

The final (2006) edition of the Downes catalog of cataclysmic variables,
cut to magnitude-at-maximum < 12 — CVs live near minimum, so Max is an
outburst magnitude: check recent photometry (AAVSO) before pointing. The
Min column says how faint the quiescent state gets. Outburst vs quiescence
spectra (absorption vs emission lines) are the SA100 story here.

The catalogue ships as ReferenceLibrary/cv_catalog.csv (tracked, bundled
by spectrum_explorer.spec); regenerate with --refresh.

Run:      py -3.13 explorer/cv_dialog.py [--refresh]
Check:    py -3.13 explorer/cv_dialog.py --selfcheck
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorer import catalog_browser as cb   # noqa: E402

CATALOG = "V/123A/cv"
MAG_LIMIT = 12.0
CACHE = cb.reference_csv("cv_catalog.csv")
FIELDS = ("name", "hd", "tyc", "tic", "vartype", "maxmag", "minmag",
          "names", "ra_deg", "dec_deg", "constellation")
COLUMNS = ("name", "hd", "tyc", "tic", "vartype", "maxmag", "minmag",
           "names", "ra_jnow", "dec_jnow", "constellation", "nspec")
HEADINGS = {"name": ("Name", 90), "hd": ("HD", 70), "tyc": ("TYC", 105),
            "tic": ("TIC", 85), "vartype": ("Type", 60),
            "maxmag": ("Max", 55), "minmag": ("Min", 55),
            "names": ("Aliases", 110),
            "ra_jnow": ("RA JNOW", 85), "dec_jnow": ("Dec JNOW", 85),
            "constellation": ("Const", 60), "nspec": ("Spec", 45)}
TITLE = f"Cataclysmic variables (max < {MAG_LIMIT:g}) — Downes+ 2006"
SIZE = "1350x600"


def _table_to_rows(table):
    keep = [r for r in table if cb.cell(r, "RAJ2000") and cb.cell(r, "DEJ2000")]
    sky = cb.sky_fields([cb.cell(r, "RAJ2000") for r in keep],
                        [cb.cell(r, "DEJ2000") for r in keep])
    return [{"name": cb.cell(r, "GCVS"),
             "vartype": cb.cell(r, "VarType"),
             "maxmag": cb.cell(r, "Maxmag"),
             "minmag": cb.cell(r, "Minmag"),
             "names": cb.cell(r, "Names"),
             **s} for r, s in zip(keep, sky)]


def fetch_catalog():
    from astroquery.vizier import Vizier
    vizier = Vizier(columns=["GCVS", "RAJ2000", "DEJ2000", "VarType",
                             "Maxmag", "Minmag", "Names"],
                    column_filters={"Maxmag": f"<{MAG_LIMIT:g}"},
                    row_limit=-1)
    tables = vizier.get_catalogs(CATALOG)
    if not tables:
        raise RuntimeError(f"VizieR returned no table for {CATALOG}")
    rows = _table_to_rows(tables[0])
    cb.crossmatch_ids(rows, _name_of)
    return rows


def load_rows(refresh=False):
    return cb.load_rows(CACHE, FIELDS, fetch_catalog, refresh)


def _name_of(row):
    """GCVS designation ("SS Cyg"); SIMBAD files these under "V* SS Cyg"."""
    return row["name"]


def _prepare(rows):
    cb.add_display_cols(rows)
    cb.mark_have_spectra(rows)
    return rows


class CVDialog(cb.CatalogDialog):
    """Modeless CV browser; contract in catalog_browser."""

    def __init__(self, parent, goto=None):
        rows = _prepare(load_rows())
        super().__init__(parent, f"{TITLE}, {len(rows)} entries", SIZE,
                         rows, COLUMNS, HEADINGS, sort="maxmag",
                         goto=goto, name_of=_name_of)


def _show(rows):
    _prepare(rows)
    cb.run_standalone(f"{TITLE}, {len(rows)} entries", SIZE,
                      rows, COLUMNS, HEADINGS, sort="maxmag")


def _selfcheck():
    from astropy.table import Table

    # SS Cyg at J2000 21 42 42.8 +43 35 10 -> Cyg.
    table = Table(rows=[("SS Cyg", "21 42 42.8", "+43 35 10", "UGSS",
                         "7.7", "12.4", "BD+42 4189"),
                        ("nocoord", "", "", "UG", "11.0", "15.0", "")],
                  names=("GCVS", "RAJ2000", "DEJ2000", "VarType",
                         "Maxmag", "Minmag", "Names"))
    rows = _table_to_rows(table)
    assert len(rows) == 1, "coordinate-less row must be dropped"
    star = rows[0]
    assert star["constellation"] == "Cyg" and star["vartype"] == "UGSS", star

    assert os.path.exists(CACHE), CACHE
    with open(CACHE, newline="", encoding="utf-8") as handle:
        shipped = list(csv.DictReader(handle))
    assert len(shipped) > 200 and set(FIELDS) <= set(shipped[0]), CACHE
    # The crossmatch is the point of the HD/TYC/TIC columns: SIMBAD knows
    # most of these, so a wholesale blank means the refresh silently failed.
    matched = sum(1 for r in shipped if r["hd"] or r["tyc"] or r["tic"])
    assert matched > len(shipped) // 3, f"only {matched} rows cross-matched"
    print("cv_dialog self-check OK")


if __name__ == "__main__":
    cb.cli(__doc__, load_rows, _show, _selfcheck)
