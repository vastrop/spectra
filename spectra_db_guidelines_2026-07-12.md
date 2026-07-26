# Light-curve DB design philosophy — transfer guidelines for the spectra DB

Source of truth: `db/lightcurve_db.py` (DDL + MIGRATIONS) in PhotometryPipeline;
rationale record in `lightcurve_db_design_2026-07-04.md`. This document is the
portable summary: follow these rules and a future client (or merge) can ask
"what do we have for that star?" across both databases.

## 1. Identity model — the part that MUST match

- **Internal integer surrogate key** (`star_id`, autoincrement) is the primary
  key. Never a catalog designation: not every star has one, Gaia source_ids
  can be renumbered between releases, and integers join/index/migrate best.
- **External identities are unique, nullable attributes**:
  `gaia_dr3_source_id` (INTEGER UNIQUE, int64-capable), `vsx_name`, `tic_id`.
- **ICRS position is the ground-truth identity**: `ra_deg`, `dec_deg`,
  `pos_epoch_jyear`. Store catalog (Gaia epoch-propagated) positions when
  matched, best measured position otherwise — and correct measured positions
  onto the catalog frame at ingest if your astrometry has a bulk offset.
- **Dedup waterfall at ingest** (in this order):
  1. match on `gaia_dr3_source_id` when present;
  2. match on catalog name (`vsx_name`) — *rejected* if the two sides carry
     different Gaia ids (close pairs sharing one entry);
  3. positional cone match at **1.5″** via an indexed box query — a cone hit
     with a conflicting Gaia id on either side is rejected (two real stars 1″
     apart stay two rows);
  4. otherwise insert a new star.
  The name step is not decorative: a night with a slightly wrong plate
  solution once duplicated 140 stars that failed both the id and cone steps.

**Cross-DB consequence:** each DB has its own `star_id` namespace. The
cross-database join is on the shared identity attributes — Gaia id first,
name second, position cone last. The spectra DB must therefore store the same
three: Gaia DR3 id (UNIQUE nullable), catalog names, and ICRS ra/dec.

## 2. Provenance hierarchy — append-only

```
stars ──< lightcurves >── runs >── datasets        (photometry)
              └──< points >── frames

stars ──< spectra     >── runs >── datasets        (spectra analog)
              └──< (pixels/orders?)
```

- **`datasets`** = the observing material (field/target + night + full
  instrument block + **site lat/lon/elev** — required for barycentric times,
  read from FITS headers with a prompt fallback).
- **`runs`** = one pipeline execution on a dataset: timestamp, git hash, and
  the **complete config snapshot as JSON** (`config_json`, one TEXT column —
  the cheapest and most valuable provenance field).
- **Product rows** (lightcurves / spectra) reference `(star_id, run_id)` with
  a UNIQUE constraint on the pair.
- **Append-only rule**: re-processing the same dataset creates a *new* run
  and new product rows; existing rows are never edited. "Latest per star" is
  a query, not an overwrite. Deletion happens only at whole-run granularity.
- **Derived-quantity history** (`ephemerides` here; line measurements or
  classifications for spectra): append-only rows with a `source` vocabulary
  and an `is_active` flag; supersede = deactivate old + insert new. A
  **partial unique index** enforces one active row per star.

## 3. Indexing philosophy

- Integer PKs everywhere; `INTEGER PRIMARY KEY` only (SQLite rowid = Postgres
  BIGSERIAL-compatible).
- **Positional index**: `CREATE INDEX idx_stars_radec ON stars(dec_deg,
  ra_deg)` — cone searches are an indexed dec/ra box + exact separation in
  code. Dec first (no wraparound/cos-scaling on the leading column). A
  HEALPix column was deliberately *deferred*: backfillable with one UPDATE at
  Postgres-migration time, when Q3C/pg_healpix takes over. Do the same.
- **Identity indexes**: UNIQUE on `gaia_dr3_source_id`; UNIQUE on
  `(star_id, run_id)` for product rows.
- **FK indexes** on every join path a tool uses: product→star, product→run,
  derived→star.
- **Partial unique index** for one-active-row invariants
  (`... ON ephemerides(star_id) WHERE is_active = 1`) — portable to Postgres.
- Bulk per-sample data (`points`) gets a **composite PK** `(product_id,
  frame_id)` and no other index; millions of rows are routine.

## 4. Physics decided at ingest (cannot be retrofitted)

- **Time standard: BJD_TDB per point**, computed at ingest (astropy `Time` +
  `EarthLocation` + `light_travel_time`), with raw `jd_utc_mid` kept once per
  frame. Requires site coordinates and the star position captured at ingest —
  the correction (±8 min/yr) is unrecoverable later without them. For spectra
  the analog is the barycentric *velocity* correction: store it (or the data
  needed to compute it) at ingest.
- **Store what the human validated** (the differential curve as reviewed)
  *plus* the raw measurement (`flux_net_adu` per point) so a better algorithm
  can be re-run against archived data without the original FITS. Spectra
  analog: store the calibrated product plus enough rawness to re-reduce.
- **Masked data is stored, flagged, never dropped** — a `flags` bitmask per
  sample (bad frame / despiked / non-finite raw). Why a point is absent is
  itself data; the analysis tool decides what to exclude.

## 5. Engine and portability discipline

- **SQLite, single file**, living outside any dataset folder (the DB spans
  fields and sessions). Zero admin, stdlib, transactional, fine to millions
  of rows.
- **Portable SQL only**: no SQLite-only pragmas in schema or queries, JSON as
  TEXT (Postgres reads it as `::jsonb` later), int64-safe ids. The eventual
  server migration is a data copy plus a connection string.
- **`schema_version` table + numbered migrations in code** from day one
  (a `MIGRATIONS = {version: [statements]}` dict applied by `connect()`;
  refuse to open a DB newer than the code).
- **Controlled vocabularies live in code constants, not SQL CHECKs**
  (SQLite can't alter a CHECK; vocabularies must grow without table
  rebuilds). E.g. `LC_STATUSES = pending/confirmed/rejected/followup`,
  category and source lists.
- **Readers open read-only** (`file:...?mode=ro` URI); viewers never edit
  stored values — their only write is whole-run deletion after confirmation.
- Every ingest is **one transaction**: on any error, nothing is written.
- A module-level `_selfcheck()` (synthetic ingest into a temp DB, run via
  `python -m ...`) guards the schema and ingest logic.

## 6. The eventual "what do we have for that star?" query

Whether by merge or by a client over both files, resolution is:

1. take the query star's Gaia DR3 id / name / position;
2. run the §1 waterfall against each DB independently to get each DB's
   `star_id`;
3. pull products by `star_id` in each.

Nothing else needs to align — table names, product schemas, and per-sample
layouts are free to differ. The contract is only: identity attributes (§1),
append-only run provenance (§2), and portable SQL (§5).
