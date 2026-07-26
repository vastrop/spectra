"""Galactic Wolf-Rayet catalogue browser (van der Hucht 2001, VizieR III/215).

Lists the VIIth Catalogue of Galactic Wolf-Rayet Stars in the shared
sortable browser (catalog_browser.py): WR number, usual name (HD etc., or
the alias when there is none), TYC, TIC, narrowband v magnitude (the WR
photometric system, NOT Johnson V — WR emission lines make broadband V
misleading), spectral type, JNOW position and constellation. Positions come
from table13, spectral types and v from table15, joined on the WR number.

The catalogue ships as ReferenceLibrary/wr_star_catalog.csv (tracked, and
bundled by spectrum_explorer.spec); regenerate with --refresh.

Run:      py -3.13 explorer/wr_star_dialog.py [--refresh]
Check:    py -3.13 explorer/wr_star_dialog.py --selfcheck
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorer import catalog_browser as cb   # noqa: E402

CATALOG = "III/215"
CACHE = cb.reference_csv("wr_star_catalog.csv")
FIELDS = ("wr", "name", "tyc", "tic", "vmag", "sptype", "ra_deg", "dec_deg",
          "constellation")
COLUMNS = ("wr", "name", "tyc", "tic", "vmag", "sptype",
           "ra_jnow", "dec_jnow", "constellation", "nspec")
HEADINGS = {"wr": ("WR", 55), "name": ("Name", 110),
            "tyc": ("TYC", 105), "tic": ("TIC", 85),
            "vmag": ("v", 55), "sptype": ("SpType", 110),
            "ra_jnow": ("RA JNOW", 85), "dec_jnow": ("Dec JNOW", 85),
            "constellation": ("Const", 60), "nspec": ("Spec", 45)}
TITLE = "Wolf-Rayet stars — van der Hucht 2001"
SIZE = "1140x600"


def _tables_to_rows(t13, t15):
    """Join positions (table13) with types/photometry (table15) on WR."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord, get_constellation
    import numpy as np

    pars = {cb.cell(r, "WR"): r for r in t15}
    keep = [r for r in t13 if cb.cell(r, "RAJ2000") and cb.cell(r, "DEJ2000")]
    coords = SkyCoord([cb.cell(r, "RAJ2000") for r in keep],
                      [cb.cell(r, "DEJ2000") for r in keep],
                      unit=(u.hourangle, u.deg))
    consts = np.atleast_1d(get_constellation(coords, short_name=True))
    rows = []
    for row, coord, const in zip(keep, coords, consts):
        wr = cb.cell(row, "WR")
        par = pars.get(wr)
        rows.append({
            "wr": wr,
            "name": cb.cell(row, "Name") or cb.cell(row, "Aname"),
            "vmag": cb.cell(par, "vmag") if par is not None else "",
            "sptype": cb.cell(par, "SpType") if par is not None else "",
            "ra_deg": f"{coord.ra.deg:.4f}",
            "dec_deg": f"{coord.dec.deg:+.4f}",
            "constellation": str(const),
        })
    return rows


def _target_id(row):
    """WR numbers are first-class SIMBAD identifiers ("WR 140")."""
    return f"WR {row['wr']}"


def fetch_catalog():
    from astroquery.vizier import Vizier
    pos = Vizier(columns=["WR", "Name", "Aname", "RAJ2000", "DEJ2000"],
                 row_limit=-1).get_catalogs(f"{CATALOG}/table13")
    par = Vizier(columns=["WR", "SpType", "vmag"],
                 row_limit=-1).get_catalogs(f"{CATALOG}/table15")
    if not pos or not par:
        raise RuntimeError(f"VizieR returned no tables for {CATALOG}")
    rows = _tables_to_rows(pos[0], par[0])
    cb.crossmatch_ids(rows, _target_id)
    return rows


def load_rows(refresh=False):
    return cb.load_rows(CACHE, FIELDS, fetch_catalog, refresh)


def _prepare(rows):
    cb.add_display_cols(rows)
    cb.mark_have_spectra(rows)
    return rows


class WRStarDialog(cb.CatalogDialog):
    """Modeless WR browser; contract in catalog_browser.CatalogDialog."""

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

    # WR 1 (HD 4004) at J2000 00 43 28.40 +64 45 35.4 -> Cas; WR 999 is
    # synthetic with no table15 entry -> blank type/photometry, kept.
    t13 = Table(rows=[("1", "HD 4004", "HIP 3415",
                       "00 43 28.40", "+64 45 35.40"),
                      ("2", "", "HIP 5100", "01 05 23.03", "+60 25 18.90"),
                      ("999", "", "", "", "")],
                names=("WR", "Name", "Aname", "RAJ2000", "DEJ2000"))
    t15 = Table(rows=[("1", "WN4-s", "10.14"), ("2", "WN2-w", "11.33")],
                names=("WR", "SpType", "vmag"))
    rows = _tables_to_rows(t13, t15)
    assert len(rows) == 2, "coordinate-less row must be dropped"
    wr1, wr2 = rows
    assert wr1["name"] == "HD 4004", wr1          # Name preferred over Aname
    assert wr2["name"] == "HIP 5100", wr2         # Aname fallback
    assert wr1["sptype"] == "WN4-s" and wr1["vmag"] == "10.14", wr1
    assert wr1["constellation"] == "Cas", wr1
    assert abs(float(wr1["ra_deg"]) - 10.868) < 0.01, wr1
    assert _target_id(wr1) == "WR 1"

    # The shipped catalogue must be present and on the current schema.
    assert os.path.exists(CACHE), CACHE
    with open(CACHE, newline="", encoding="utf-8") as handle:
        shipped = list(csv.DictReader(handle))
    assert len(shipped) > 200 and set(FIELDS) <= set(shipped[0]), CACHE
    print("wr_star_dialog self-check OK")


if __name__ == "__main__":
    cb.cli(__doc__, load_rows, _show, _selfcheck)
