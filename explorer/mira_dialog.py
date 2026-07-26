"""Mira-variable catalogue browser (GCVS 5.1, VizieR B/gcvs, max < 10).

Miras from the General Catalogue of Variable Stars (VarType exactly "M"),
cut to magnitude-at-maximum < 10: the observable-near-maximum set. The Max
column is the GCVS magnitude at maximum — at minimum these stars fade by
up to ~8 mag, so check the period and a recent light curve before slewing.
TiO band evolution plus Balmer emission near maximum are the SA100 payoff.

The catalogue ships as ReferenceLibrary/mira_catalog.csv (tracked, bundled
by spectrum_explorer.spec); regenerate with --refresh.

Run:      py -3.13 explorer/mira_dialog.py [--refresh]
Check:    py -3.13 explorer/mira_dialog.py --selfcheck
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorer import catalog_browser as cb   # noqa: E402

CATALOG = "B/gcvs/gcvs_cat"
MAG_LIMIT = 10.0
CACHE = cb.reference_csv("mira_catalog.csv")
FIELDS = ("name", "hd", "tyc", "tic", "magmax", "period", "sptype",
          "ra_deg", "dec_deg", "constellation")
COLUMNS = ("name", "hd", "tyc", "tic", "magmax", "period", "sptype",
           "ra_jnow", "dec_jnow", "constellation", "nspec")
HEADINGS = {"name": ("Name", 90), "hd": ("HD", 70), "tyc": ("TYC", 105),
            "tic": ("TIC", 85), "magmax": ("Max", 55),
            "period": ("P (d)", 60), "sptype": ("SpType", 110),
            "ra_jnow": ("RA JNOW", 85), "dec_jnow": ("Dec JNOW", 85),
            "constellation": ("Const", 60), "nspec": ("Spec", 45)}
TITLE = f"Mira variables (max < {MAG_LIMIT:g}) — GCVS 5.1"
SIZE = "1230x600"


def _table_to_rows(table):
    keep = [r for r in table if cb.cell(r, "RAJ2000") and cb.cell(r, "DEJ2000")]
    sky = cb.sky_fields([cb.cell(r, "RAJ2000") for r in keep],
                        [cb.cell(r, "DEJ2000") for r in keep])
    return [{"name": cb.cell(r, "GCVS"),
             "magmax": cb.cell(r, "magMax"),
             "period": cb.cell(r, "Period"),
             "sptype": cb.cell(r, "SpType"),
             **s} for r, s in zip(keep, sky)]


def fetch_catalog():
    from astroquery.vizier import Vizier
    vizier = Vizier(columns=["GCVS", "RAJ2000", "DEJ2000", "VarType",
                             "magMax", "Period", "SpType"],
                    column_filters={"VarType": "=M",
                                    "magMax": f"<{MAG_LIMIT:g}"},
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
    """GCVS designation ("R And"); SIMBAD files these under "V* R And"."""
    return row["name"]


def _prepare(rows):
    cb.add_display_cols(rows)
    cb.mark_have_spectra(rows)
    return rows


class MiraDialog(cb.CatalogDialog):
    """Modeless Mira browser; contract in catalog_browser."""

    def __init__(self, parent, goto=None):
        rows = _prepare(load_rows())
        super().__init__(parent, f"{TITLE}, {len(rows)} entries", SIZE,
                         rows, COLUMNS, HEADINGS, sort="magmax",
                         goto=goto, name_of=_name_of)


def _show(rows):
    _prepare(rows)
    cb.run_standalone(f"{TITLE}, {len(rows)} entries", SIZE,
                      rows, COLUMNS, HEADINGS, sort="magmax")


def _selfcheck():
    from astropy.table import Table

    # Mira itself (omi Cet) at J2000 02 19 20.8 -02 58 40 -> Cet.
    table = Table(rows=[("omi Cet", "02 19 20.8", "-02 58 40", "M",
                         "2.0", "331.96", "M5e-M9e"),
                        ("nocoord", "", "", "M", "9.0", "300", "M5e")],
                  names=("GCVS", "RAJ2000", "DEJ2000", "VarType",
                         "magMax", "Period", "SpType"))
    rows = _table_to_rows(table)
    assert len(rows) == 1, "coordinate-less row must be dropped"
    star = rows[0]
    assert star["constellation"] == "Cet" and star["period"] == "331.96", star

    assert os.path.exists(CACHE), CACHE
    with open(CACHE, newline="", encoding="utf-8") as handle:
        shipped = list(csv.DictReader(handle))
    assert len(shipped) > 1000 and set(FIELDS) <= set(shipped[0]), CACHE
    # The crossmatch is the point of the HD/TYC/TIC columns: SIMBAD knows
    # most of these, so a wholesale blank means the refresh silently failed.
    matched = sum(1 for r in shipped if r["hd"] or r["tyc"] or r["tic"])
    assert matched > len(shipped) // 3, f"only {matched} rows cross-matched"
    print("mira_dialog self-check OK")


if __name__ == "__main__":
    cb.cli(__doc__, load_rows, _show, _selfcheck)
