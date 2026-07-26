"""Carbon-star catalogue browser (Alksnis+ 2001, VizieR III/227, V < 10).

The General Catalog of Galactic Carbon Stars, cut to V < 10 — the SA100's
practical spectroscopy range, and where the C2 Swan bands are at their most
spectacular. Rows without a V magnitude are dropped by the same cut (6891
catalogue entries, mostly faint IR stars, become ~440 observable ones).

The catalogue ships as ReferenceLibrary/carbon_star_catalog.csv (tracked,
bundled by spectrum_explorer.spec); regenerate with --refresh.

Run:      py -3.13 explorer/carbon_star_dialog.py [--refresh]
Check:    py -3.13 explorer/carbon_star_dialog.py --selfcheck
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorer import catalog_browser as cb   # noqa: E402

CATALOG = "III/227/catalog"
MAG_LIMIT = 10.0
CACHE = cb.reference_csv("carbon_star_catalog.csv")
FIELDS = ("cgcs", "hd", "tyc", "tic", "vmag", "bmag", "sptype",
          "ra_deg", "dec_deg", "constellation")
COLUMNS = ("cgcs", "hd", "tyc", "tic", "vmag", "bmag", "sptype",
           "ra_jnow", "dec_jnow", "constellation", "nspec")
HEADINGS = {"cgcs": ("CGCS", 70), "hd": ("HD", 70), "tyc": ("TYC", 105),
            "tic": ("TIC", 85), "vmag": ("V", 55), "bmag": ("B", 55),
            "sptype": ("SpType", 110),
            "ra_jnow": ("RA JNOW", 85), "dec_jnow": ("Dec JNOW", 85),
            "constellation": ("Const", 60), "nspec": ("Spec", 45)}
TITLE = f"Carbon stars (V < {MAG_LIMIT:g}) — Alksnis+ 2001"
SIZE = "1195x600"


def _table_to_rows(table):
    keep = [r for r in table if cb.cell(r, "RAJ2000") and cb.cell(r, "DEJ2000")]
    sky = cb.sky_fields([cb.cell(r, "RAJ2000") for r in keep],
                        [cb.cell(r, "DEJ2000") for r in keep])
    return [{"cgcs": cb.cell(r, "CGCS"),
             "vmag": cb.cell(r, "Vmag"),
             "bmag": cb.cell(r, "Bmag"),
             "sptype": cb.cell(r, "Sp"),
             **s} for r, s in zip(keep, sky)]


def fetch_catalog():
    from astroquery.vizier import Vizier
    vizier = Vizier(columns=["CGCS", "RAJ2000", "DEJ2000", "Bmag", "Vmag",
                             "Sp"],
                    column_filters={"Vmag": f"<{MAG_LIMIT:g}"}, row_limit=-1)
    tables = vizier.get_catalogs(CATALOG)
    if not tables:
        raise RuntimeError(f"VizieR returned no table for {CATALOG}")
    rows = _table_to_rows(tables[0])
    cb.crossmatch_ids(rows, _name_of)   # III/227 ships no cross-ids at all
    return rows


def load_rows(refresh=False):
    return cb.load_rows(CACHE, FIELDS, fetch_catalog, refresh)


def _prepare(rows):
    cb.add_display_cols(rows)
    cb.mark_have_spectra(rows)
    return rows


def _name_of(row):
    return f"CGCS {row['cgcs']}"


class CarbonStarDialog(cb.CatalogDialog):
    """Modeless carbon-star browser; contract in catalog_browser."""

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

    # Y CVn (La Superba) at J2000 12 45 07.8 +45 26 25 -> CVn.
    table = Table(rows=[("1234", "12 45 07.8", "+45 26 25", "7.3", "4.9",
                         "C5,4J"),
                        ("9999", "", "", "9.0", "8.0", "C")],
                  names=("CGCS", "RAJ2000", "DEJ2000", "Bmag", "Vmag", "Sp"))
    rows = _table_to_rows(table)
    assert len(rows) == 1, "coordinate-less row must be dropped"
    star = rows[0]
    assert star["constellation"] == "CVn" and star["vmag"] == "4.9", star
    assert _name_of(star) == "CGCS 1234"

    assert os.path.exists(CACHE), CACHE
    with open(CACHE, newline="", encoding="utf-8") as handle:
        shipped = list(csv.DictReader(handle))
    assert len(shipped) > 300 and set(FIELDS) <= set(shipped[0]), CACHE
    # The crossmatch is the point of the HD/TYC/TIC columns: SIMBAD knows
    # most of these, so a wholesale blank means the refresh silently failed.
    matched = sum(1 for r in shipped if r["hd"] or r["tyc"] or r["tic"])
    assert matched > len(shipped) // 2, f"only {matched} rows cross-matched"
    print("carbon_star_dialog self-check OK")


if __name__ == "__main__":
    cb.cli(__doc__, load_rows, _show, _selfcheck)
