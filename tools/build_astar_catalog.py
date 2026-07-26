#!/usr/bin/env python3
"""Build the permanent bright early-A-star catalog from Hipparcos.

Hipparcos ICRS positions have epoch 1991.25. Proper motion is deliberately
ignored: even decades of drift remain irrelevant for slewing a degrees-wide,
slitless-spectroscopy field.  Revisit only if this catalog is reused for
narrow-field pointing.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

from astroquery.vizier import Vizier


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(ROOT, "ReferenceLibrary",
                              "astar_catalog.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build astar_catalog.csv from Hipparcos via Vizier")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    vizier = Vizier(
        columns=["HIP", "HD", "Vmag", "SpType", "RAICRS", "DEICRS"],
        column_filters={"Vmag": "<8"}, row_limit=-1)
    tables = vizier.get_catalogs("I/239/hip_main")
    if not tables:
        raise RuntimeError("Vizier returned no Hipparcos table")

    rows = []
    for row in tables[0]:
        sptype = str(row["SpType"]).strip()
        if (getattr(row["SpType"], "mask", False)
                or getattr(row["RAICRS"], "mask", False)
                or getattr(row["DEICRS"], "mask", False)
                or not re.match(r"^A[0-3]", sptype)):
            continue
        rows.append({
            "hip": "" if getattr(row["HIP"], "mask", False) else int(row["HIP"]),
            "hd": "" if getattr(row["HD"], "mask", False) else int(row["HD"]),
            "ra_deg": float(row["RAICRS"]),
            "dec_deg": float(row["DEICRS"]),
            "vmag": float(row["Vmag"]),
            "sptype": sptype,
        })
    rows.sort(key=lambda item: item["vmag"])

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("hip", "hd", "ra_deg", "dec_deg", "vmag",
                                "sptype"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
