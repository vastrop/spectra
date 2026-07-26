# ASTAP WCS Plate Solving — Implementation Reference

How the **Spectra** project calls **ASTAP** for plate solving (astrometric WCS on FITS
frames). Written as a portable reference so the same approach can be reused elsewhere;
it replaces the earlier description of CometPlanner's version, which this implementation
has since diverged from (see §9 for the back-port list).

Everything lives in `source_identification.py`, stage 1 (`solve_wcs`, `_astap_hint_flags`,
`_parse_astap_solution`). No GUI dependencies; failures always degrade to `None`.

---

## 1. Overview

Given a mono FITS frame with no celestial WCS, run ASTAP and return an astropy `WCS`
object **in memory**. The flow is:

1. Write the frame to a temporary FITS, copying the header's position/scale hints.
2. Call ASTAP as a subprocess with NINA-parity flags.
3. Parse ASTAP's `.ini` sidecar into an astropy `WCS`; validate `has_celestial`.
4. Delete the temp directory.

Two deliberate departures from the CometPlanner design:

- **Mono only.** The pipeline works on a single derotated mono frame, so there is no
  RGB / green-channel path and no `NAXIS3` preservation problem.
- **No write-back.** Nothing is injected into the original FITS header. The `WCS` object
  is consumed live (`spectrum_explorer.py`, `sequence_generator.py`) to convert measured
  zero-order centroids to sky coordinates. Frames on disk stay untouched.

The frame handed to `solve_wcs` MUST be the **rotated** frame, so the returned WCS shares
a pixel grid with the centroids fed through it.

---

## 2. Configuration

```python
DEFAULT_ASTAP_PATH = "C:/Program Files/astap/astap.exe"
ASTAP_TIMEOUT = 60          # seconds, subprocess hard timeout
```

ASTAP is an external prerequisite (not bundled by `build_exe.bat`) and needs its own star
database (H17/H18/D80) configured inside ASTAP itself. A missing executable is a logged
warning and `None`, not an exception.

---

## 3. Detecting whether a WCS already exists

Not done here — this path exists because the frames never carry a solution. If you need
the check, `WCS(header, naxis=2).has_celestial` is the authoritative test (partial or
garbage WCS cards can otherwise look like a solution).

---

## 4. How ASTAP is called

### 4.1 Preparing the temporary FITS

```python
temp_base = Path(tempfile.gettempdir()) / "astap_spectro_solving"
temp_base.mkdir(exist_ok=True)
for stale in temp_base.glob("solve_*"):    # a survivor means a previous run
    _safe_rmtree(stale)                    # died mid-solve; sweep before creating
temp_dir = Path(tempfile.mkdtemp(prefix="solve_", dir=temp_base))
temp_fits = temp_dir / "temp_solve.fits"

hdu = fits.PrimaryHDU(np.ascontiguousarray(rotated_data, dtype=np.float32))
```

The predictable parent folder is for debugging; the stale sweep matters because ASTAP can
still hold the FITS when `rmtree` runs on Windows, so the `finally` cleanup is not
guaranteed to have succeeded last time.

### 4.2 Header hints copied into the temp FITS

A fixed set, when present, copied verbatim — and remembered for the failure log:

```python
for key in ("OBJCTRA", "OBJCTDEC", "RA", "DEC",
            "XPIXSZ", "YPIXSZ", "FOCALLEN", "FOCALRAT",
            "DATE-OBS", "TELESCOP", "INSTRUME"):
```

No `CRVAL1`/`CRVAL2` seeding: the position reaches ASTAP through the command line
(§4.3), which is what NINA does and what ASTAP actually acts on.

### 4.3 The hint flags (`_astap_hint_flags`)

This is the part that was wrong before 2026-07-25 and is worth porting. It mirrors NINA's
`ASTAPSolver.cs` argument builder:

| Flag | Value | Why |
|------|-------|-----|
| `-fov` | field **height** in degrees, `206.265 × XPIXSZ / FOCALLEN × height_px / 3600` | without it ASTAP guesses the scale |
| `-z` | `0` (auto downsample) | flagless ASTAP detects at 1×1 on the full-res frame and misses soft SA100 stars; NINA's auto-binning is why it finds enough |
| `-s` | `500` (star limit) | NINA default |
| `-r` | `30` (search radius, degrees) | without a radius ASTAP spirals **180°** on failure — minutes per failed solve |
| `-ra` | RA in **hours** | position hint |
| `-spd` | **Dec + 90** (south-pole distance) | position hint |

Three unit traps, straight from NINA's source, and the likely cause of the old note that
"explicit flags make the solve fail": `-ra` is in **hours, not degrees**, `-spd` is
**Dec+90, not Dec**, and both must be **dot-decimal** (f-strings are locale-independent,
matching NINA's `InvariantCulture`).

Position comes from numeric `RA`/`DEC` (degrees, so `/15` for hours) first, falling back
to sexagesimal `OBJCTRA`/`OBJCTDEC` (already hours, sign taken from the string).

**Every flag degrades independently** — there is no retry and no second invocation. A
header without scale keys still gets `-z`/`-s`; one without a position still gets `-fov`;
a bare header yields ASTAP's own defaults, i.e. never worse than the flagless call.

### 4.4 The command line

```python
cmd = ([astap_path, "-f", str(temp_fits), "-solve", "-update"]
       + _astap_hint_flags(fits_header, rotated_data.shape[0]))
```

One list element per argument (`"-solve -update"` as a single joined string is the classic
bug). `-update` is what makes ASTAP write the solution sidecars.

### 4.5 Running the subprocess

```python
result = subprocess.run(
    cmd, cwd=str(temp_dir),
    capture_output=True, text=True, timeout=ASTAP_TIMEOUT,
    creationflags=subprocess.CREATE_NO_WINDOW)
```

- `cwd=temp_dir` so sidecars land next to the temp FITS.
- `CREATE_NO_WINDOW` — without it a frozen `--noconsole` build flashes a console per solve.
- `TimeoutExpired` and any non-zero `returncode` → logged warning, `None`.

### 4.6 Failure logging

The GUI `astap.exe` is **silent on stdout/stderr**; on failure its actual reason
(`ERROR=`/`WARNING=` lines, `PLTSOLVD=F`) goes into the `.ini` sidecar. So every failure
path logs the `.ini` tail (last 300 chars, read *before* the `finally` rmtree eats it),
the exact command via `subprocess.list2cmdline`, and the hints that made it into the temp
FITS — enough to replay the solve by hand and diff it against another ASTAP setup.

---

## 5. Reading the solution back (the `.ini` sidecar)

`_parse_astap_solution(temp_fits.with_suffix(".ini"))`. The `.ini`, not the `.wcs`: it
carries the same solution plus the `PLTSOLVD` flag and the error text, so one file covers
both success and diagnosis.

Plain `KEY=value` lines; values are stripped of any FITS-style `/ comment` and quotes,
then cast float (contains `.eE`) → int → string.

Validation order, any failure returning `None`:

1. `PLTSOLVD` starts with `T`.
2. `CRVAL1`, `CRVAL2`, `CRPIX1`, `CRPIX2` all present.
3. Either the full `CD1_1/CD1_2/CD2_1/CD2_2` matrix **or** `CDELT1/CDELT2`
   (with `CROTA2` if present) — ASTAP writes one or the other.
4. The assembled `WCS` is `has_celestial`.

`CTYPE1`/`CTYPE2` are usually absent from the `.ini`; default them to `RA---TAN` /
`DEC--TAN`, which is what ASTAP's TAN solution implies.

---

## 6. Writing the WCS back into a FITS

Not implemented — see §1. If you add it, the CometPlanner rules still apply: whitelist the
WCS keywords, copy any `PC*_*`, and **restore the original `NAXIS*`** afterwards, because
`wcs.to_header()` can silently flatten an RGB cube to 2D.

---

## 7. RGB vs mono handling

Mono only. RGB would be solved on the green channel, as CometPlanner does.

---

## 8. Verifying a change to this path

There is no self-check for `solve_wcs` — it needs a real ASTAP and a real frame. The
practical test is a known-good frame through the explorer with logging on: a solve that
took minutes-then-failed before the flags now lands in ~0.7 s (1.6 s rotated) and matches
NINA's solution digit for digit. `_astap_hint_flags` is pure and cheap to check by hand
against NINA's numbers if you touch the unit conversions.

---

## 9. Porting back to CometPlanner

The changes worth carrying over, in order of payoff:

1. **The hint flags (§4.3).** This is the whole win: `-fov`, `-z 0`, `-s 500`,
   `-r 30 -ra -spd`, with RA in hours and `-spd = Dec+90`. CometPlanner computes
   `fov_estimate` and `search_radius` and then never passes them.
2. **Split the arguments** — `["-solve", "-update"]`, not `["-solve -update"]`.
3. **Read the `.ini`, not the `.wcs` (§5)**, and check `PLTSOLVD`. Same solution, plus the
   failure reason.
4. **Log the command, hints and `.ini` tail on failure (§4.6)** before cleaning up. Nearly
   all the 2026-07-25 debugging time went into not having this.
5. **Accept `CDELT`+`CROTA2` as well as the `CD` matrix (§5)** — a valid solve can arrive
   in either form.
6. **Sweep stale `solve_*` temp dirs (§4.1)** and `CREATE_NO_WINDOW` on Windows.

Keep on the CometPlanner side: the RGB green-channel path, `copy_wcs_to_fits` with its
`NAXIS*` preservation, and the `has_celestial` pre-check for already-solved files — none
of that is needed here, and dropping it would be a regression there.
