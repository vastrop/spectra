"""Herbig Ae/Be and T Tauri catalogue browser (Herbig & Bell 1988, V/73A).

The Herbig-Bell Catalog of pre-main-sequence emission-line stars of the
Orion population, kept in full. The Class column is the catalogue's own
type flag: "tt" = T Tauri, "ae" = Herbig Ae/Be, blank = other/unclassified.
Positions are catalogue 1950.0, but VizieR precomputes ICRS
(_RA.icrs/_DE.icrs), and those are the columns read here. Hα emission is
the SA100 observable (AB Aur, T Tau itself, V633 Cas).

The catalogue ships as ReferenceLibrary/herbig_catalog.csv (tracked,
bundled by spectrum_explorer.spec); regenerate with --refresh.

Run:      py -3.13 explorer/herbig_dialog.py [--refresh]
Check:    py -3.13 explorer/herbig_dialog.py --selfcheck
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorer import catalog_browser as cb   # noqa: E402

CATALOG = "V/73A/catalog"
CACHE = cb.reference_csv("herbig_catalog.csv")
FIELDS = ("hbc", "name", "hd", "tyc", "tic", "vmag", "sptype", "ytype",
          "ra_deg", "dec_deg", "constellation")
COLUMNS = ("hbc", "name", "hd", "tyc", "tic", "vmag", "sptype", "ytype",
           "ra_jnow", "dec_jnow", "constellation", "nspec")
HEADINGS = {"hbc": ("HBC", 55), "name": ("Name", 110), "hd": ("HD", 70),
            "tyc": ("TYC", 105), "tic": ("TIC", 85), "vmag": ("V", 55),
            "sptype": ("SpType", 100), "ytype": ("Class", 50),
            "ra_jnow": ("RA JNOW", 85), "dec_jnow": ("Dec JNOW", 85),
            "constellation": ("Const", 60), "nspec": ("Spec", 45)}
TITLE = "Herbig Ae/Be & T Tauri — Herbig-Bell 1988"
SIZE = "1280x600"


def _table_to_rows(table):
    keep = [r for r in table
            if cb.cell(r, "_RA.icrs") and cb.cell(r, "_DE.icrs")]
    sky = cb.sky_fields([cb.cell(r, "_RA.icrs") for r in keep],
                        [cb.cell(r, "_DE.icrs") for r in keep])
    return [{"hbc": cb.cell(r, "HBC"),
             "name": cb.cell(r, "name"),
             "vmag": cb.cell(r, "Vmag"),
             "sptype": cb.cell(r, "Sp"),
             "ytype": cb.cell(r, "type"),
             **s} for r, s in zip(keep, sky)]


def fetch_catalog():
    from astroquery.vizier import Vizier
    vizier = Vizier(columns=["HBC", "name", "_RA.icrs", "_DE.icrs", "Vmag",
                             "Sp", "type"], row_limit=-1)
    tables = vizier.get_catalogs(CATALOG)
    if not tables:
        raise RuntimeError(f"VizieR returned no table for {CATALOG}")
    rows = _table_to_rows(tables[0])
    cb.crossmatch_ids(rows, _name_of)
    return rows


def load_rows(refresh=False):
    return cb.load_rows(CACHE, FIELDS, fetch_catalog, refresh)


def _prepare(rows):
    cb.add_display_cols(rows)
    cb.mark_have_spectra(rows)
    return rows


def _name_of(row):
    return row["name"] or f"HBC {row['hbc']}"


class HerbigDialog(cb.CatalogDialog):
    """Modeless Herbig/T Tauri browser; contract in catalog_browser."""

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

    # T Tau at J2000 04 21 59.4 +19 32 06 -> Tau, class tt.
    table = Table(rows=[("35", "T Tau", "04 21 59.4", "+19 32 06", "9.6",
                         "K0e", "tt"),
                        ("999", "nocoord", "", "", "12.0", "A0e", "ae")],
                  names=("HBC", "name", "_RA.icrs", "_DE.icrs", "Vmag",
                         "Sp", "type"))
    rows = _table_to_rows(table)
    assert len(rows) == 1, "coordinate-less row must be dropped"
    star = rows[0]
    assert star["constellation"] == "Tau" and star["ytype"] == "tt", star
    assert _name_of(star) == "T Tau"
    assert _name_of({"name": "", "hbc": "7"}) == "HBC 7"

    assert os.path.exists(CACHE), CACHE
    with open(CACHE, newline="", encoding="utf-8") as handle:
        shipped = list(csv.DictReader(handle))
    assert len(shipped) > 500 and set(FIELDS) <= set(shipped[0]), CACHE
    # The crossmatch is the point of the HD/TYC/TIC columns: SIMBAD knows
    # most of these, so a wholesale blank means the refresh silently failed.
    matched = sum(1 for r in shipped if r["hd"] or r["tyc"] or r["tic"])
    assert matched > len(shipped) // 3, f"only {matched} rows cross-matched"
    print("herbig_dialog self-check OK")


if __name__ == "__main__":
    cb.cli(__doc__, load_rows, _show, _selfcheck)
