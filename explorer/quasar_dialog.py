"""Bright-quasar catalogue browser (Véron-Cetty & Véron 2010, VizieR VII/258).

Lists the quasars (class Q only — BL Lacs and low-luminosity AGN excluded)
brighter than magnitude 16 from the 13th edition of the Catalogue of Quasars
and Active Nuclei, in the shared sortable browser (catalog_browser.py):
name, redshift, magnitude with its band (blank n_Vmag flag = Johnson V;
R/O/* are red or photographic magnitudes as flagged in the catalogue),
JNOW position and constellation. No stellar designations, obviously.

The catalogue ships as ReferenceLibrary/quasar_catalog.csv (tracked, and
bundled by spectrum_explorer.spec); regenerate with --refresh.

Run:      py -3.13 explorer/quasar_dialog.py [--refresh]
Check:    py -3.13 explorer/quasar_dialog.py --selfcheck
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorer import catalog_browser as cb   # noqa: E402

CATALOG = "VII/258/vv10"
MAG_LIMIT = 16.0
CACHE = cb.reference_csv("quasar_catalog.csv")
FIELDS = ("name", "z", "vmag", "band", "ra_deg", "dec_deg", "constellation")
COLUMNS = ("name", "z", "vmag", "band",
           "ra_jnow", "dec_jnow", "constellation", "nspec")
HEADINGS = {"name": ("Name", 130), "z": ("z", 60),
            "vmag": ("Mag", 55), "band": ("Band", 45),
            "ra_jnow": ("RA JNOW", 85), "dec_jnow": ("Dec JNOW", 85),
            "constellation": ("Const", 60), "nspec": ("Spec", 45)}
TITLE = f"Bright quasars (< mag {MAG_LIMIT:g}) — Véron-Cetty & Véron 2010"
SIZE = "920x600"


def _table_to_rows(table):
    """Astropy table (VII/258 vv10 schema, J2000 sexagesimal) -> dicts."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord, get_constellation
    import numpy as np

    keep = [r for r in table if cb.cell(r, "RAJ2000") and cb.cell(r, "DEJ2000")]
    coords = SkyCoord([cb.cell(r, "RAJ2000") for r in keep],
                      [cb.cell(r, "DEJ2000") for r in keep],
                      unit=(u.hourangle, u.deg))
    consts = np.atleast_1d(get_constellation(coords, short_name=True))
    rows = []
    for row, coord, const in zip(keep, coords, consts):
        z = cb.cell(row, "z")
        # Véron's class-Q table carries a few z=0 locals (M 31, NGC 3031...)
        # that are catalogue errors, not quasars — drop them.
        if not z or float(z) <= 0:
            continue
        rows.append({
            "name": cb.cell(row, "Name"),
            "z": z,
            "vmag": cb.cell(row, "Vmag"),
            "band": cb.cell(row, "n_Vmag") or "V",
            "ra_deg": f"{coord.ra.deg:.4f}",
            "dec_deg": f"{coord.dec.deg:+.4f}",
            "constellation": str(const),
        })
    return rows


def fetch_catalog():
    from astroquery.vizier import Vizier
    vizier = Vizier(columns=["Name", "RAJ2000", "DEJ2000", "z", "Vmag",
                             "n_Vmag"],
                    column_filters={"Cl": "=Q", "Vmag": f"<{MAG_LIMIT:g}"},
                    row_limit=-1)
    tables = vizier.get_catalogs(CATALOG)
    if not tables:
        raise RuntimeError(f"VizieR returned no table for {CATALOG}")
    return _table_to_rows(tables[0])


def load_rows(refresh=False):
    return cb.load_rows(CACHE, FIELDS, fetch_catalog, refresh)


def _prepare(rows):
    cb.add_display_cols(rows)
    cb.mark_have_spectra(rows)
    return rows


class QuasarDialog(cb.CatalogDialog):
    """Modeless quasar browser; contract in catalog_browser.CatalogDialog."""

    def __init__(self, parent, goto=None):
        rows = _prepare(load_rows())
        super().__init__(parent, f"{TITLE}, {len(rows)} entries", SIZE,
                         rows, COLUMNS, HEADINGS,
                         goto=goto, name_of=lambda r: r["name"])


def _show(rows):
    _prepare(rows)
    cb.run_standalone(f"{TITLE}, {len(rows)} entries", SIZE,
                      rows, COLUMNS, HEADINGS)


def _selfcheck():
    from astropy.table import Table

    # 3C 273 at J2000 12 29 06.7 +02 03 09 -> Vir; blank flag reads as V.
    table = Table(rows=[("3C 273", "12 29 06.7", "+02 03 09",
                         "0.158", "12.85", ""),
                        ("PKS 0002-478", "00 04 35.7", "-47 36 18",
                         "0.88", "15.88", "R"),
                        ("nocoord", "", "", "1.0", "15.0", ""),
                        # Catalogue error: a local galaxy classed Q at z=0.
                        ("M 31", "00 42 44.3", "+41 16 08",
                         "0.0", "10.57", "")],
                  names=("Name", "RAJ2000", "DEJ2000", "z", "Vmag", "n_Vmag"))
    rows = _table_to_rows(table)
    assert len(rows) == 2, "coordinate-less and z=0 rows must be dropped"
    q1, q2 = rows
    assert q1["constellation"] == "Vir", q1
    assert abs(float(q1["ra_deg"]) - 187.278) < 0.01, q1
    assert q1["band"] == "V" and q2["band"] == "R", (q1, q2)
    assert q2["constellation"] == "Phe", q2

    # The shipped catalogue must be present and on the current schema.
    assert os.path.exists(CACHE), CACHE
    with open(CACHE, newline="", encoding="utf-8") as handle:
        shipped = list(csv.DictReader(handle))
    assert len(shipped) > 300 and set(FIELDS) <= set(shipped[0]), CACHE
    print("quasar_dialog self-check OK")


if __name__ == "__main__":
    cb.cli(__doc__, load_rows, _show, _selfcheck)
