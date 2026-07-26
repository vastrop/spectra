"""Symbiotic-star catalogue browser (Belczyński+ 2000, VizieR J/A+AS/146/407).

The reference catalogue of symbiotic and suspected symbiotic stars, kept in
full — it is small, and symbiotics brighten unpredictably. Spectral types
come from the companion properties table, joined on the [BMM2000] number.
An M-giant TiO continuum with hydrogen emission on top, in one SA100 shot,
is the payoff (EG And, Z And, CH Cyg, AG Dra).

The catalogue ships as ReferenceLibrary/symbiotic_catalog.csv (tracked,
bundled by spectrum_explorer.spec); regenerate with --refresh.

Run:      py -3.13 explorer/symbiotic_dialog.py [--refresh]
Check:    py -3.13 explorer/symbiotic_dialog.py --selfcheck
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorer import catalog_browser as cb   # noqa: E402

CATALOG = "J/A+AS/146/407"
CACHE = cb.reference_csv("symbiotic_catalog.csv")
FIELDS = ("num", "name", "hd", "tyc", "tic", "vmag", "sptype",
          "ra_deg", "dec_deg", "constellation")
COLUMNS = ("num", "name", "hd", "tyc", "tic", "vmag", "sptype",
           "ra_jnow", "dec_jnow", "constellation", "nspec")
HEADINGS = {"num": ("#", 45), "name": ("Name", 110), "hd": ("HD", 70),
            "tyc": ("TYC", 105), "tic": ("TIC", 85), "vmag": ("V", 55),
            "sptype": ("SpType", 110),
            "ra_jnow": ("RA JNOW", 85), "dec_jnow": ("Dec JNOW", 85),
            "constellation": ("Const", 60), "nspec": ("Spec", 45)}
TITLE = "Symbiotic stars — Belczyński+ 2000"
SIZE = "1205x600"


def _tables_to_rows(cat, prop):
    """Join the position/photometry table with SpType on [BMM2000]."""
    types = {cb.cell(r, "[BMM2000]"): cb.cell(r, "SpType") for r in prop}
    keep = [r for r in cat if cb.cell(r, "RAJ2000") and cb.cell(r, "DEJ2000")]
    sky = cb.sky_fields([cb.cell(r, "RAJ2000") for r in keep],
                        [cb.cell(r, "DEJ2000") for r in keep])
    return [{"num": cb.cell(r, "[BMM2000]"),
             "name": cb.cell(r, "Name"),
             "vmag": cb.cell(r, "Vmag"),
             "sptype": types.get(cb.cell(r, "[BMM2000]"), ""),
             **s} for r, s in zip(keep, sky)]


def fetch_catalog():
    from astroquery.vizier import Vizier
    cat = Vizier(columns=["[BMM2000]", "Name", "RAJ2000", "DEJ2000", "Vmag"],
                 row_limit=-1).get_catalogs(f"{CATALOG}/catalog")
    prop = Vizier(columns=["[BMM2000]", "SpType"],
                  row_limit=-1).get_catalogs(f"{CATALOG}/prop")
    if not cat or not prop:
        raise RuntimeError(f"VizieR returned no tables for {CATALOG}")
    rows = _tables_to_rows(cat[0], prop[0])
    cb.crossmatch_ids(rows, _name_of)
    return rows


def load_rows(refresh=False):
    return cb.load_rows(CACHE, FIELDS, fetch_catalog, refresh)


def _name_of(row):
    """The catalogue's Name column is the SIMBAD-resolvable designation."""
    return row["name"]


def _prepare(rows):
    cb.add_display_cols(rows)
    cb.mark_have_spectra(rows)
    return rows


class SymbioticDialog(cb.CatalogDialog):
    """Modeless symbiotic-star browser; contract in catalog_browser."""

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

    # EG And at J2000 00 44 37.1 +40 40 46 -> And; row 004 has no prop
    # entry -> blank type; the coordinate-less row drops.
    cat = Table(rows=[("003", "EG And", "00 44 37.1", "+40 40 45.7", "7.1"),
                      ("004", "X Foo", "01 00 00.0", "+41 00 00.0", "9.9"),
                      ("005", "nocoord", "", "", "8.0")],
                names=("[BMM2000]", "Name", "RAJ2000", "DEJ2000", "Vmag"))
    prop = Table(rows=[("003", "M3")], names=("[BMM2000]", "SpType"))
    rows = _tables_to_rows(cat, prop)
    assert len(rows) == 2, "coordinate-less row must be dropped"
    eg, x = rows
    assert eg["constellation"] == "And" and eg["sptype"] == "M3", eg
    assert x["sptype"] == "", x

    assert os.path.exists(CACHE), CACHE
    with open(CACHE, newline="", encoding="utf-8") as handle:
        shipped = list(csv.DictReader(handle))
    assert len(shipped) > 150 and set(FIELDS) <= set(shipped[0]), CACHE
    # The crossmatch is the point of the HD/TYC/TIC columns: SIMBAD knows
    # most of these, so a wholesale blank means the refresh silently failed.
    matched = sum(1 for r in shipped if r["hd"] or r["tyc"] or r["tic"])
    assert matched > len(shipped) // 3, f"only {matched} rows cross-matched"
    print("symbiotic_dialog self-check OK")


if __name__ == "__main__":
    cb.cli(__doc__, load_rows, _show, _selfcheck)
