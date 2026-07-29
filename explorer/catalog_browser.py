"""Shared machinery for the explorer's catalogue-browser dialogs.

Each concrete dialog (be_star_dialog, wr_star_dialog, quasar_dialog)
supplies its VizieR fetch, its column layout and its tracked CSV under
ReferenceLibrary/; this module owns everything catalogue-agnostic: the CSV
load/regenerate cycle, the sortable zebra-striped Treeview, the display-time
JNOW and live Alt/Az columns, the SIMBAD TAP TYC/TIC crossmatch, and the
standalone --refresh/--selfcheck CLI shared by all of them.

Check:    py -3.13 explorer/catalog_browser.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import threading

import tkinter as tk
from tkinter import ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # subfolder scripts reach root modules (db.*)

# Observer location for the "Visible tonight" filter. Beside the .exe when
# frozen (same rule as the spectra DB — _internal/ is wiped on reinstall).
if getattr(sys, "frozen", False):
    DATA_ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    DATA_ROOT = ROOT
OBSERVER_PATH = os.path.join(DATA_ROOT, "observer.json")
SCHEDULER_PATH = os.path.join(DATA_ROOT, "scheduler_targets.txt")
# Placeholder site, used only until the user sets their own in the filter
# bar (persisted to observer.json).  Mid-northern latitude, prime meridian.
DEFAULT_LAT, DEFAULT_LON = 45.0, 0.0
MIN_ALT_DEG = 30.0   # fixed visibility threshold; make it a field in the
                     # filter bar if it ever needs tuning per target


def reference_csv(name):
    return os.path.join(ROOT, "ReferenceLibrary", name)


def load_observer():
    """(lat_deg, lon_deg) from observer.json, defaults if absent/broken."""
    import json
    try:
        with open(OBSERVER_PATH, encoding="utf-8") as handle:
            d = json.load(handle)
        return float(d["lat_deg"]), float(d["lon_deg"])
    except Exception:
        return DEFAULT_LAT, DEFAULT_LON


def save_observer(lat_deg, lon_deg):
    import json
    try:
        with open(OBSERVER_PATH, "w", encoding="utf-8") as handle:
            json.dump({"lat_deg": lat_deg, "lon_deg": lon_deg}, handle)
    except OSError:
        pass


def schedule_targets(pairs, path=None):
    """Append (name, row) targets to the scheduler outbox; returns #added.

    The outbox is a tab-separated text file, one target per line:
    name, JNOW RA h:m:s, JNOW Dec d:m:s, J2000 ra/dec in degrees, UTC
    added. JNOW is what the scheduler consumes; the J2000 degrees ride
    along so coordinates can be re-derived losslessly at any later epoch.
    Names already present are skipped (re-sending a selection is a no-op).
    """
    from datetime import datetime, timezone

    path = path or SCHEDULER_PATH
    existing = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = {line.split("\t")[0] for line in handle
                        if line.strip() and not line.startswith("#")}
    new = [(name, row) for name, row in pairs if name not in existing]
    if not new:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "a", encoding="utf-8") as handle:
        if not existing:
            handle.write("# name\tra_jnow\tdec_jnow\tra_j2000_deg"
                         "\tdec_j2000_deg\tadded_utc\n")
        for name, row in new:
            handle.write(f"{name}\t{row['ra_jnow']}\t{row['dec_jnow']}"
                         f"\t{row['ra_deg']}\t{row['dec_deg']}\t{stamp}\n")
    return len(new)


def sky_fields(ra_values, dec_values):
    """Sexagesimal (hourangle, deg) ICRS positions -> per-object dicts with
    ra_deg / dec_deg / constellation, ready to merge into catalogue rows."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord, get_constellation
    import numpy as np

    coords = SkyCoord(ra_values, dec_values, unit=(u.hourangle, u.deg))
    consts = np.atleast_1d(get_constellation(coords, short_name=True))
    return [{"ra_deg": f"{c.ra.deg:.4f}",
             "dec_deg": f"{c.dec.deg:+.4f}",
             "constellation": str(k)}
            for c, k in zip(np.atleast_1d(coords), consts)]


def _dark_grid(lat_deg, lon_deg, t0):
    """(times, dark_mask, location) on a 30-min grid over the coming 24 h.

    Darkness is sun < -12° (nautical); summer at mid-latitudes may never
    get there, so it degrades to sun < -6°, then to the darkest hour of
    the night — the numbers stay meaningful in July instead of going empty.
    """
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, get_sun

    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg)
    times = t0 + np.linspace(0.0, 24.0, 49) * u.hour
    sun_alt = get_sun(times).transform_to(
        AltAz(obstime=times, location=loc)).alt.deg
    for limit in (-12.0, -6.0):
        dark = sun_alt < limit
        if dark.any():
            break
    else:
        dark = sun_alt <= sun_alt.min() + 1.0
    return times, dark, loc


def night_altitudes(rows, lat_deg, lon_deg, when=None):
    """Per-row maximum altitude (deg) during tonight's darkness at the site.

    Night duration is inherent: a shorter dark window simply gives objects
    fewer hours in which to peak (darkness rules in _dark_grid).
    """
    import numpy as np
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time

    t0 = when or Time.now()
    times, dark, loc = _dark_grid(lat_deg, lon_deg, t0)
    dark_times = times[dark]
    if len(dark_times) > 16:   # bound the transform cost, keep the spread
        dark_times = dark_times[::int(np.ceil(len(dark_times) / 16))]

    coords = SkyCoord([float(r["ra_deg"]) for r in rows],
                      [float(r["dec_deg"]) for r in rows], unit="deg")
    frame = AltAz(obstime=dark_times[np.newaxis, :], location=loc)
    alt = coords.reshape(-1, 1).transform_to(frame).alt.deg
    return [float(v) for v in alt.max(axis=1)]


def visible_mask(rows, lat_deg, lon_deg, when=None, min_alt_deg=MIN_ALT_DEG):
    """Per-row True when the object clears ``min_alt_deg`` during tonight's
    darkness — see night_altitudes for the darkness-window rules."""
    return [a >= min_alt_deg
            for a in night_altitudes(rows, lat_deg, lon_deg, when=when)]


def dark_reference_time(lat_deg, lon_deg, when=None):
    """The moment the Alt/Az columns are quoted at: now when the sun is
    already down (the observing case), else the first dark point of the
    coming night — so a browser opened in the afternoon shows the sky as
    it will stand at dusk."""
    from astropy.time import Time

    t0 = when or Time.now()
    times, dark, _ = _dark_grid(lat_deg, lon_deg, t0)
    return t0 if dark[0] else times[dark][0]


def current_altaz(rows, lat_deg, lon_deg, when=None):
    """Per-row (alt, az) in degrees at dark_reference_time."""
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time

    t0 = when or Time.now()
    times, dark, loc = _dark_grid(lat_deg, lon_deg, t0)
    ref = t0 if dark[0] else times[dark][0]
    coords = SkyCoord([float(r["ra_deg"]) for r in rows],
                      [float(r["dec_deg"]) for r in rows], unit="deg")
    aa = coords.transform_to(AltAz(obstime=ref, location=loc))
    return [(float(a), float(z)) for a, z in zip(aa.alt.deg, aa.az.deg)]


def warm_coordinates():
    """Build astropy's AltAz machinery on a background thread, at leisure.

    The first AltAz transform in a process spends ~0.5 s assembling the
    coordinate transform graph and loading the IERS table. Without this,
    that cost lands on whichever catalogue browser opens first, and it is
    spent before the browser's window exists — so it reads as the menu
    hanging. Pure computation, no Tk state, safe off the main thread.
    """
    def run():
        try:
            current_altaz([{"ra_deg": "0", "dec_deg": "0"}], 0.0, 0.0)
        except Exception:
            pass   # warm-up only — the real call reports its own failure

    threading.Thread(target=run, daemon=True).start()


def cell(row, key):
    """Astropy table cell as a stripped string, '' when masked."""
    value = row[key]
    return "" if getattr(value, "mask", False) else str(value).strip()


def sort_key(value):
    try:
        return (0, float(value))
    except ValueError:
        pass
    sexa = re.match(r"^([+-]?)(\d+):(\d+):(\d+(?:\.\d+)?)$", value)
    if sexa:   # h:m:s / d:m:s — lexical order breaks on negative declinations
        sign = -1.0 if sexa.group(1) == "-" else 1.0
        return (0, sign * (float(sexa.group(2)) * 3600
                           + float(sexa.group(3)) * 60 + float(sexa.group(4))))
    return (1, value)   # blanks and text sort after numbers


def add_display_cols(rows, equinox=None):
    """Derive the JNOW position for display.

    JNOW drifts ~50"/yr, so it is computed from the stored J2000 degrees
    every time a browser opens — never persisted to the CSVs. J2000 itself
    is data, not display: ra_deg/dec_deg stay in every row for the goto
    contract and the scheduler outbox, without a column of their own.
    """
    from astropy import units as u
    from astropy.coordinates import FK5, SkyCoord
    from astropy.time import Time

    coords = SkyCoord([float(r["ra_deg"]) for r in rows],
                      [float(r["dec_deg"]) for r in rows], unit="deg")
    jnow = coords.transform_to(FK5(equinox=equinox or Time.now()))
    ra_j = jnow.ra.to_string(unit=u.hourangle, sep=":", precision=1, pad=True)
    dec_j = jnow.dec.to_string(unit=u.deg, sep=":", precision=0,
                               alwayssign=True, pad=True)
    for row, rj, dj in zip(rows, ra_j, dec_j):
        row["ra_jnow"] = str(rj)
        row["dec_jnow"] = str(dj)


def id_key(ident):
    """Normalise an identifier so SIMBAD's replies match the local keys.

    SIMBAD resolves loosely but answers in its own canonical form: padded
    to a fixed width ("CSS    3") and carrying the object-class designator
    a variable star is filed under ("T And" comes back as "V* T And"). Ask
    for one form, get another — so both sides of the join go through here,
    or every variable-star catalogue silently matches nothing.
    """
    text = re.sub(r"\s+", " ", ident).strip()
    prefix, _, rest = text.partition(" ")
    if prefix in ("V*", "*", "**", "NAME"):
        text = rest.strip()
    return text.casefold()


def adql_in_list(ids):
    """Quote identifiers for an ADQL IN (...) list, doubling any quote.

    A name like "Ap'ndx" would otherwise close the string literal early and
    make the query unparseable — losing the whole 300-id chunk, not one row.
    """
    return ", ".join("'" + i.replace("'", "''") + "'" for i in ids)


def apply_matches(rows_by_id, pairs):
    """Fill hd/tyc/tic from (target, other) identifier pairs (padding-blind)."""
    index = {}
    for raw, rows in rows_by_id.items():
        index.setdefault(id_key(raw), []).extend(rows)
    for target, other in pairs:
        prefix, _, value = other.partition(" ")
        key = prefix.lower()
        if key not in ("hd", "tyc", "tic"):
            continue
        # SIMBAD zero-pads into a fixed width ("TYC  297-383-1", "HD  5394"),
        # so the tail needs stripping or the value displays shifted right.
        value = value.strip()
        # A shared id (double-star components) maps several rows to one key.
        for row in index.get(id_key(target), ()):
            if not row[key]:
                row[key] = value


def crossmatch_ids(rows, target_id):
    """Fill each row's HD, TYC and TIC identifiers from SIMBAD (batched TAP).

    ``target_id(row)`` must return a SIMBAD-resolvable identifier; rows it
    maps to ids SIMBAD does not know simply stay blank. An HD the source
    catalogue already carries wins over SIMBAD's — it is what target_id
    resolved on, so overwriting it could rename the row out from under the
    match.
    """
    from astroquery.simbad import Simbad
    for row in rows:
        row["tyc"] = row["tic"] = ""
        row.setdefault("hd", "")
    rows_by_id = {}
    for row in rows:
        rows_by_id.setdefault(target_id(row), []).append(row)
    ids = [i for i in rows_by_id if i]
    for start in range(0, len(ids), 300):
        chunk = adql_in_list(ids[start:start + 300])
        table = Simbad.query_tap(
            "SELECT i1.id AS target, i2.id AS other FROM ident i1 "
            "JOIN ident i2 ON i1.oidref = i2.oidref "
            f"WHERE i1.id IN ({chunk}) "
            "AND (i2.id LIKE 'HD %' OR i2.id LIKE 'TYC %' "
            "OR i2.id LIKE 'TIC %')")
        apply_matches(rows_by_id,
                      [(str(r["target"]), str(r["other"])) for r in table])


def mark_have_spectra(rows, db_path=None, radius_arcsec=60.0):
    """Set each row's 'nspec' to the DB's spectrum count for that object.

    Matches by position against the spectra DB's identified star positions,
    so it works across designation systems (and for quasars). 60" absorbs
    proper-motion drift of the older catalogue epochs without risking
    cross-matches — the observation DB is sparse. Missing or unreadable DB
    (e.g. a newer schema) leaves every count blank.
    """
    from db import spectra_db

    for row in rows:
        row["nspec"] = ""
    path = db_path or spectra_db.DEFAULT_DB_PATH
    if not os.path.exists(path):
        return
    try:
        conn = spectra_db.connect(path, readonly=True)
        try:
            stars = conn.execute(
                "SELECT s.ra_deg, s.dec_deg, COUNT(sp.spectrum_id) "
                "FROM stars s JOIN spectra sp ON sp.star_id = s.star_id "
                "GROUP BY s.star_id").fetchall()
        finally:
            conn.close()
    except Exception as exc:
        print(f"Spectra DB unreadable, 'Spec' column left blank: {exc}")
        return
    for row in rows:
        ra, dec = float(row["ra_deg"]), float(row["dec_deg"])
        count = 0
        for sra, sdec, n in stars:
            if abs(sdec - dec) * 3600.0 > radius_arcsec:   # cheap prefilter
                continue
            if spectra_db._sep_arcsec(ra, dec, sra, sdec) <= radius_arcsec:
                count += n
        row["nspec"] = str(count) if count else ""


def load_rows(cache, fields, fetch, refresh=False):
    """Read the tracked CSV, or (re)generate it via ``fetch()``."""
    if not refresh and os.path.exists(cache):
        with open(cache, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if set(fields) <= set(reader.fieldnames or ()):
                return list(reader)
        print("Catalogue file predates the current schema — refetching.")
    rows = fetch()
    # Write beside the cache and os.replace into place: a failure
    # mid-write must not truncate the last good catalogue (the next
    # launch would refetch from the network, or fail offline).
    tmp = cache + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        # extrasaction: FIELDS is the schema, and a shared helper may leave
        # keys a given catalogue does not publish (crossmatch_ids fills hd
        # for everyone, but WR shows it inside its Name column instead).
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, cache)
    print(f"Wrote {len(rows)} rows to {cache}")
    return rows


def populate(container, rows, columns, headings, sort="vmag"):
    """Build the sortable zebra-striped Treeview into ``container``.

    Returns (tree, set_rows, row_of): ``set_rows(new_rows)`` re-renders
    with a new row list, preserving the current sort column/direction —
    that is how the visibility filter swaps the displayed set. ``row_of``
    maps a tree item id back to its full row dict, which carries fields
    (ra_deg/dec_deg) that are not displayed as columns.
    """
    tree = ttk.Treeview(container, columns=columns, show="headings")
    tree.tag_configure("even", background="white")
    tree.tag_configure("odd", background="#f0f0f0")
    scroll = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    state = {"rows": list(rows), "col": sort, "desc": False, "items": {}}

    def render():
        # Re-rendering (sort, filter) replaces every item id — carry the
        # selection across by row identity, not by id.
        keep = {id(state["items"][iid]) for iid in tree.selection()
                if iid in state["items"]}
        tree.delete(*tree.get_children())
        state["items"] = {}
        order = sorted(state["rows"],
                       key=lambda r: sort_key(r[state["col"]]),
                       reverse=state["desc"])
        reselect = []
        for i, row in enumerate(order):
            iid = tree.insert("", "end", values=[row[f] for f in columns],
                              tags=("odd" if i % 2 else "even",))
            state["items"][iid] = row
            if id(row) in keep:
                reselect.append(iid)
        if reselect:
            tree.selection_set(reselect)
            tree.see(reselect[0])

    def on_heading(col):
        state["desc"] = not state["desc"] if col == state["col"] else False
        state["col"] = col
        render()

    def set_rows(new_rows):
        state["rows"] = list(new_rows)
        render()

    for col in columns:
        text, width = headings[col]
        tree.heading(col, text=text, command=lambda c=col: on_heading(c))
        tree.column(col, width=width, anchor="w", stretch=False)

    render()   # brightest first by default
    tree.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    return tree, set_rows, lambda iid: state["items"][iid]


def build_browser(container, rows, columns, headings, sort="vmag",
                  goto=None, name_of=None):
    """Assemble the full browser into ``container``: filter bar + tree.

    The bar holds the "Visible tonight" filter (on by default; observer
    lat/lon persisted in observer.json) and the mutually exclusive
    "Visible now", a shown-count label, "Schedule
    selected" (appends to the scheduler_targets.txt outbox, multi-select
    aware) and — when a ``goto`` callback is given — the "Goto selected"
    button. Live Alt°/Az° and night-max Culm° columns are appended to
    every layout.
    """
    from tkinter import messagebox

    bar = ttk.Frame(container)
    bar.pack(side="bottom", fill="x")
    v_visible = tk.BooleanVar(master=container, value=True)
    v_now = tk.BooleanVar(master=container, value=False)
    lat0, lon0 = load_observer()
    v_lat = tk.StringVar(master=container, value=f"{lat0:g}")
    v_lon = tk.StringVar(master=container, value=f"{lon0:g}")
    v_count = tk.StringVar(master=container)
    cache = {}
    # Live altitudes as numbers, kept beside the formatted Alt° cells: the
    # "Visible now" filter must compare the real value, not the rounded one.
    alt_now = []

    def altitudes_for(lat, lon):   # night-max — drives the visibility filter
        import time as _time
        key = (round(lat, 3), round(lon, 3),
               int(_time.time() // 3600))   # recompute hourly at most
        if key not in cache:
            cache.clear()
            cache[key] = night_altitudes(rows, lat, lon)
        return cache[key]

    def fill_altaz(lat, lon):
        alt_now.clear()
        for row, (alt, az) in zip(rows, current_altaz(rows, lat, lon)):
            row["alt"] = f"{alt:.0f}"
            row["az"] = f"{az:.0f}"
            alt_now.append(alt)
        # Culm is the night-max the filter runs on — the "how good does it
        # get tonight" number the scheduler outbox is chosen by.
        for row, culm in zip(rows, altitudes_for(lat, lon)):
            row["culm"] = f"{culm:.0f}"

    # Alt/Az are the live position — current during darkness, else at the
    # coming dusk — refreshed every minute below. The visibility filter
    # deliberately stays on the night-max ("clears the bar at some point
    # tonight"), so targets rising later don't vanish from the plan.
    try:
        fill_altaz(lat0, lon0)
    except Exception as exc:
        print(f"Alt/Az columns unavailable: {exc}")
        for row in rows:
            row["alt"] = row["az"] = row["culm"] = ""
    columns = tuple(columns) + ("alt", "az", "culm")
    headings = {**headings, "alt": ("Alt°", 50), "az": ("Az°", 50),
                "culm": ("Culm°", 55)}

    tree, set_rows, row_of = populate(container, rows, columns, headings,
                                      sort=sort)

    def refresh():
        shown = rows
        if v_visible.get() or v_now.get():
            try:
                lat, lon = float(v_lat.get()), float(v_lon.get())
            except ValueError:
                messagebox.showwarning(
                    "Observer location", "Lat/Lon must be decimal degrees.",
                    parent=container)
                v_visible.set(False)
                v_now.set(False)
                return
            try:
                fill_altaz(lat, lon)
                # Same threshold, different instant: "now" runs on the live
                # altitude (the Alt° column, i.e. at dusk while it is still
                # day), "tonight" on the night max.
                alts = list(alt_now) if v_now.get() else altitudes_for(lat, lon)
            except Exception as exc:   # filter must not break the browser
                print(f"Visibility filter unavailable: {exc}")
                v_visible.set(False)
                v_now.set(False)
            else:
                save_observer(lat, lon)
                shown = [r for r, a in zip(rows, alts) if a >= MIN_ALT_DEG]
        set_rows(shown)
        v_count.set(f"{len(shown)} of {len(rows)} shown"
                    if len(shown) != len(rows) else "")

    def tick():
        # Cell updates in place — no re-render, so sort order and selection
        # never jump under the user once a minute.
        if not container.winfo_exists():
            return
        try:
            fill_altaz(float(v_lat.get()), float(v_lon.get()))
            # Spec counts go stale the moment a livestack is added to the
            # DB with the browser open — re-query along with the minute
            # tick (one SQL read + sparse position match, cheap).
            if "nspec" in columns:
                mark_have_spectra(rows)
            for iid in tree.get_children():
                row = row_of(iid)
                tree.set(iid, "alt", row["alt"])
                tree.set(iid, "az", row["az"])
                tree.set(iid, "culm", row["culm"])   # hourly cache rollover
                if "nspec" in columns:
                    tree.set(iid, "nspec", row["nspec"])
            # "Visible now" membership turns with the sky, so it has to be
            # re-applied — but only when a row actually crossed the
            # threshold, since a re-render every minute would reset the
            # scroll position under the user.  A simultaneous rise and set
            # hides in the count and waits for the next minute.
            if v_now.get() and (sum(a >= MIN_ALT_DEG for a in alt_now)
                                != len(tree.get_children())):
                refresh()
        except Exception:
            pass   # lat/lon mid-edit or a transient — retry next minute
        container.after(60_000, tick)

    container.after(60_000, tick)

    # The two visibility filters answer different questions ("worth planning
    # for tonight" vs "shoot it right now"), so they are mutually exclusive
    # rather than combined.
    def toggle(chosen, other):
        if chosen.get():
            other.set(False)
        refresh()

    ttk.Checkbutton(bar, text=f"Visible tonight (>{MIN_ALT_DEG:g}°)",
                    variable=v_visible,
                    command=lambda: toggle(v_visible, v_now)).pack(
        side="left", padx=(6, 10), pady=4)
    ttk.Checkbutton(bar, text=f"Visible now (>{MIN_ALT_DEG:g}°)",
                    variable=v_now,
                    command=lambda: toggle(v_now, v_visible)).pack(
        side="left", padx=(0, 10), pady=4)
    for label, var in (("Lat", v_lat), ("Lon", v_lon)):
        ttk.Label(bar, text=label).pack(side="left")
        entry = ttk.Entry(bar, textvariable=var, width=7)
        entry.pack(side="left", padx=(2, 8))
        entry.bind("<Return>", lambda _e: (cache.clear(), refresh()))
    ttk.Label(bar, textvariable=v_count).pack(side="left", padx=8)

    def selected_rows():
        # Full row dicts, not tree cell values — ra_deg/dec_deg have no
        # column any more but goto and the scheduler still need them.
        return [(name_of(row) if name_of else row[columns[0]], row)
                for row in map(row_of, tree.selection())]

    def do_schedule():
        pairs = selected_rows()
        if not pairs:
            return
        added = schedule_targets(pairs)
        v_count.set(f"{added} added to scheduler" if added
                    else "already in scheduler")

    buttons = []
    if goto is not None:
        btn_goto = ttk.Button(bar, text="→  Goto selected",
                              style="Action.TButton", state="disabled")
        btn_goto.pack(side="right", padx=6, pady=4)
        buttons.append(btn_goto)

        def do_goto():
            pairs = selected_rows()
            if not pairs:
                return
            name, row = pairs[0]
            goto(name, float(row["ra_deg"]), float(row["dec_deg"]))

        btn_goto.config(command=do_goto)

    btn_sched = ttk.Button(bar, text="🗓  Schedule selected",
                           style="Action.TButton", state="disabled",
                           command=do_schedule)
    btn_sched.pack(side="right", padx=(6, 0), pady=4)
    buttons.append(btn_sched)

    def on_select(_event):
        state = "normal" if tree.selection() else "disabled"
        for button in buttons:
            button.config(state=state)

    tree.bind("<<TreeviewSelect>>", on_select)
    refresh()   # "Visible tonight" is the default view


class CatalogDialog(tk.Toplevel):
    """Modeless catalogue browser shell.

    Parent-API contract: the explorer parent is only the Tk master — no
    explorer state is read or written. Single-instance is enforced by the
    caller (spectrum_explorer._show_catalog). The optional ``goto``
    callback — (name, ra_deg, dec_deg), J2000 degrees, same convention as
    the A-star slew — adds a "Goto selected" button; the explorer wires it
    to the NINA remote panel, which owns confirmation and mount safety.

    ``name_of`` maps a row dict (all values strings) to the display name
    used in the goto confirmation; defaults to the first column.
    """

    def __init__(self, parent, title, size, rows, columns, headings,
                 sort="vmag", goto=None, name_of=None):
        super().__init__(parent)
        self.title(title)
        self.geometry(size)
        build_browser(self, rows, columns, headings, sort=sort,
                      goto=goto, name_of=name_of)


def run_standalone(title, size, rows, columns, headings, sort="vmag"):
    root = tk.Tk()
    root.title(title)
    root.geometry(size)
    build_browser(root, rows, columns, headings, sort=sort)
    root.mainloop()


def cli(doc, load, show, selfcheck, argv=None):
    """Shared --refresh/--selfcheck entry point for the dialog modules."""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument("--refresh", action="store_true",
                        help="refetch the catalogue from VizieR/SIMBAD")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        selfcheck()
        return
    show(load(refresh=args.refresh))


def _selfcheck():
    assert sort_key("") > sort_key("9.99") > sort_key("2.47")
    assert (sort_key("-80:00:00") < sort_key("-05:00:00")
            < sort_key("+02:30:00") < sort_key("+10:00:00"))
    assert sort_key("02:03:04.5") < sort_key("10:00:00")

    # gamma Cas J2000 -> pinned-equinox JNOW (deterministic; precession
    # moves its RA forward ~3.7 s/yr at that declination).
    from astropy.time import Time
    disp = {"ra_deg": "14.1772", "dec_deg": "+60.7167"}
    add_display_cols([disp], equinox=Time("2026-01-01"))
    assert disp["ra_jnow"].startswith("00:58:1"), disp
    assert disp["dec_jnow"].startswith("+60:5"), disp

    # Crossmatch plumbing, offline: SIMBAD's own padding ignored on both
    # sides of the pair, shared-id rows all filled, unknown targets and
    # non-HD/TYC/TIC pairs ignored, and a catalogue's own HD not overwritten.
    a = {"hd": "", "tyc": "", "tic": ""}
    a2 = {"hd": "", "tyc": "", "tic": ""}
    b = {"hd": "", "tyc": "", "tic": ""}
    c = {"hd": "", "tyc": "", "tic": ""}
    d = {"hd": "99", "tyc": "", "tic": ""}
    e = {"hd": "", "tyc": "", "tic": ""}
    apply_matches({"HD 5394": [a, a2], "BD+62 11": [b],
                   "CGCS 3298": [c], "CGCS 1": [d], "T And": [e]},
                  [("HD   5394", "TYC  4017-2319-1"),
                   ("HD   5394", "TIC 51962733"),
                   ("BD+62    11", "TIC 83224269"),
                   ("BD+62    11", "GAIA DR3 1"),
                   ("CGCS 3298", "HD  111908"),
                   ("CGCS 1", "HD  111908"),
                   ("V* T And", "TIC 437756809"),
                   ("HD 99999", "TIC 1")])
    assert a["tyc"] == "4017-2319-1" and a["tic"] == "51962733", a
    assert a2["tic"] == "51962733", a2
    assert b["tic"] == "83224269" and b["tyc"] == "", b
    assert c["hd"] == "111908", c   # HD gained from SIMBAD
    assert d["hd"] == "99", d       # catalogue's own HD not overwritten
    assert e["tic"] == "437756809", e   # "T And" answered as "V* T And"
    assert id_key("V*  RR  And") == id_key("rr and") == "rr and"

    # An apostrophe must not close the ADQL literal early: one unescaped
    # quote makes the whole query unparseable and loses the entire chunk.
    assert adql_in_list(["Ap'ndx", "Z And"]) == "'Ap''ndx', 'Z And'"

    # "Have spectrum" counts: two runs of a star at gamma Cas' position in a
    # temp DB -> the nearby catalogue row counts 2, the far row stays blank;
    # a missing DB file blanks everything without erroring.
    import json
    import tempfile
    from db import spectra_db

    near = {"ra_deg": "14.1772", "dec_deg": "+60.7167"}
    far = {"ra_deg": "100.0000", "dec_deg": "-20.0000"}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.db")
        conn = spectra_db.connect(path)
        star = dict(gaia_dr3_source_id=1, main_id="* gam Cas", label="",
                    sp_type="B0.5IVe", otype="Be*",
                    ra_deg=14.1772, dec_deg=60.7167, pos_epoch_jyear=2000.0)
        run = dict(run_utc="2026-07-18T00:00:00+00:00",
                   config_json=json.dumps({}))
        for n, sha in enumerate(("s1", "s2")):
            spectra_db.ingest_spectrum(
                conn,
                star=star,
                dataset=dict(fits_path=f"C:/obs/{sha}.fits", fits_sha256=sha),
                run=run,
                spectrum=dict(),
                samples=[(0, 1.0, 4000.0, 1.0, 0.1, 0)])
        conn.close()
        mark_have_spectra([near, far], db_path=path)
        assert near["nspec"] == "2" and far["nspec"] == "", (near, far)
        mark_have_spectra([near], db_path=os.path.join(td, "absent.db"))
        assert near["nspec"] == "", near

    # sky_fields: 3C 273's sexagesimal position -> degrees + Vir.
    sky = sky_fields(["12 29 06.7"], ["+02 03 09"])[0]
    assert abs(float(sky["ra_deg"]) - 187.278) < 0.01, sky
    assert sky["constellation"] == "Vir", sky

    # Visible tonight at a mid-northern site: Polaris circles at ~lat
    # altitude (always in), a dec -60 object never rises. Winter has true
    # nautical darkness; the July call exercises the no-darkness fallback
    # instead of returning nothing.  The latitude is load-bearing: 50.6°
    # sits just above the limit where astronomical night disappears at
    # midsummer, which is what makes the July branch fire.
    lat, lon = 50.6, 0.0
    targets = [{"ra_deg": "37.95", "dec_deg": "+89.26"},   # Polaris
               {"ra_deg": "100.0", "dec_deg": "-60.0"}]    # never up
    for when in ("2026-01-15T12:00:00", "2026-07-17T12:00:00"):
        mask = visible_mask(targets, lat, lon, when=Time(when))
        assert mask == [True, False], (when, mask)
    # Polaris' night-max altitude sits within ~1 deg of the site latitude.
    alts = night_altitudes(targets, lat, lon,
                           when=Time("2026-01-15T12:00:00"))
    assert abs(alts[0] - lat) < 2.0 and alts[1] < 0.0, alts

    # Live Alt/Az reference time: a winter-noon call jumps forward to the
    # coming dusk; a call during darkness is quoted at "now".
    noon = Time("2026-01-15T12:00:00")
    ref = dark_reference_time(lat, lon, when=noon)
    assert 3 * 3600 < (ref - noon).sec < 8 * 3600, ref.iso
    night = Time("2026-01-15T22:00:00")
    assert dark_reference_time(lat, lon, when=night) == night
    # Polaris pins the frame either way: altitude ~site latitude, azimuth
    # within ~1.2° of due north (dec 89.26).
    for when in (noon, night):
        alt, az = current_altaz(targets[:1], lat, lon, when=when)[0]
        assert abs(alt - lat) < 2.0, (when.iso, alt)
        assert min(az, 360.0 - az) < 2.0, (when.iso, az)

    # Scheduler outbox: header once, name-dedup on re-send, JNOW + J2000
    # fields land tab-separated and parsable.
    with tempfile.TemporaryDirectory() as td:
        outbox = os.path.join(td, "sched.txt")
        row = {"ra_jnow": "00:58:17.2", "dec_jnow": "+60:51:12",
               "ra_deg": "14.1772", "dec_deg": "+60.7167"}
        assert schedule_targets([("HD 5394", row)], path=outbox) == 1
        assert schedule_targets([("HD 5394", row),
                                 ("HD 10516", row)], path=outbox) == 1
        with open(outbox, encoding="utf-8") as handle:
            lines = [l.rstrip("\n") for l in handle]
        assert lines[0].startswith("#") and len(lines) == 3, lines
        fields = lines[1].split("\t")
        assert fields[0] == "HD 5394" and fields[1] == "00:58:17.2", fields
        assert fields[3] == "14.1772" and fields[5].endswith("Z"), fields
    print("catalog_browser self-check OK")


if __name__ == "__main__":
    _selfcheck()
