# Spectrum Explorer

Amateur stellar spectroscopy from slitless grating spectra. Point it at a FITS
frame containing a star's spectrum and it extracts the trace, calibrates the
wavelength scale, corrects for the instrument response, and estimates a
black-body temperature — with a catalogue browser, a spectra database and a
NINA-driven capture loop around the edges.

Built for a Star Analyser-class transmission grating on a small telescope, but
nothing in the maths assumes a particular instrument.

![Spectra of stars in Cassiopeia](posters/spectra_poster.png)

## Requirements

- Python 3.13
- The packages in `requirements.txt` (version floors there are load-bearing —
  `specutils` 2.0 renamed `Spectrum1D`, `astroquery` 0.4.8 changed SIMBAD's
  column case)
- [ASTAP](https://www.hnsky.org/astap.htm) with a star database, for the
  optional plate-solving and source-identification features. External
  prerequisite, not bundled.

## Install

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

## Run

The programs are independent; run them from the repository root.

```
python spectrum_explorer.py         # the main application
python focus_analyzer.py            # focus analysis on a folder of frames
python db/spectra_browser.py        # browse the spectra database
```

New here? `getting_started.md` walks through calibrating a fresh setup on a
bright A-type star, which is where everything else starts.

## Layout

| Path | What it holds |
|---|---|
| `spectrum_core.py` | all the maths — extraction, calibration, temperature fitting. No GUI. |
| `spectrum_explorer.py` | the Tkinter application and its state |
| `*_dialog.py`, `*_viewer.py` | the calibration, continuum, library and NINA panels |
| `explorer/` | catalogue browsers (Be, WR, carbon, Mira, S-type, CV, symbiotic, Herbig, quasars) |
| `db/` | the spectra database and its browser |
| `tools/` | standalone helpers — exposure estimator, autofocus, throughput checks |
| `tests/` | self-checks |
| `ReferenceLibrary/` | Pickles stellar spectral library, used as the response reference |

## Building a standalone executable

Optional — `build_exe.bat` freezes a windowed onedir build into `dist\` with
PyInstaller, which is not part of `requirements.txt` because it is not needed
to run from source:

```
pip install "pyinstaller>=6.10"
build_exe.bat
```

The script uses `SPECTRA_PYTHON` if set, otherwise `.venv` beside the
repository, otherwise the launcher, and refuses to build against an
environment missing `specutils` or `astroquery` — such a build succeeds while
silently baking in a crippled continuum fit. Close any running
`spectrum_explorer.exe` first or the rebuild fails on locked files.

## Tests

There is no test framework. Checks are `assert`-based and run directly:

```
python tests/test_planck_temperature.py
python tests/test_dispersion_math.py
python -m db.spectra_db
```

Most modules have a `__main__` self-check; `tests/` holds the rest.

## Credits

This program is a thin layer over other people's work. The maths is
elementary; the reference data is not, and none of it was produced here.

### Reference spectra

The `ReferenceLibrary/` templates and the default response reference
`reference/a0v.dat` are the **Pickles stellar spectral flux library**:

> Pickles, A. J. 1998, *A Stellar Spectral Flux Library: 1150–25000 Å*,
> PASP 110, 863. [PDF](https://www.eso.org/sci/observing/tools/standards/IR_spectral_library/hilib.pdf)

Response calibration divides observed counts by these, so the whole
instrument-response chain rests on them.

### Spectral-type notes

The per-type cards under `ReferenceLibrary/notes/` carry no per-card
citations; every classification claim in them comes from this list, shown as
links under each card in the DB browser:

- Gray, [*A Digital Spectral Classification Atlas*](https://ned.ipac.caltech.edu/level5/Gray/TOC.html)
- Morgan, Keenan & Kellman, [*An Atlas of Stellar Spectra*](https://www.ucl.ac.uk/mathematical-physical-sciences/sites/mathematical_physical_sciences/files/mkkbook.pdf)
- Pickles 1998, as above
- [NIST Handbook of Basic Atomic Spectroscopic Data](https://physics.nist.gov/PhysRefData/Handbook/atomic_number.htm)
- Sota et al., [GOSSS O-star classification](https://arxiv.org/abs/1101.4002)
- Kirkpatrick, Henry & McCarthy 1991, [K5–M9 standards](https://articles.adsabs.harvard.edu/pdf/1991ApJS...77..417K)

### Catalogues

The browsers under `explorer/` ship as `ReferenceLibrary/*_catalog.csv`,
regenerated from VizieR with each module's `--refresh`. Every one is somebody
else's catalogue:

| Browser | Catalogue | VizieR |
|---|---|---|
| Be stars | Jaschek & Egret 1982 | `III/67A` |
| Wolf-Rayet | van der Hucht 2001, VIIth Catalogue of Galactic WR Stars | `III/215` |
| Carbon stars | Alksnis et al. 2001, General Catalog of Galactic Carbon Stars | `III/227` |
| S-type stars | Stephenson 1984, General Catalogue of Galactic S Stars | `III/168` |
| Mira variables | General Catalogue of Variable Stars 5.1 | `B/gcvs` |
| Cataclysmic variables | Downes et al. 2006 | `V/123A` |
| Symbiotic stars | Belczyński et al. 2000 | `J/A+AS/146/407` |
| Herbig Ae/Be & T Tauri | Herbig & Bell 1988, HBC | `V/73A` |
| Quasars | Véron-Cetty & Véron 2010, 13th edition | `VII/258` |
| A-star slew list | Hipparcos | `I/239/hip_main` |

### Services queried at runtime

- **SIMBAD** and **VizieR** — object identification, spectral types, and the
  TAP crossmatch that fills in HD/TYC/TIC and Gaia DR3 identifiers.
- **LAMOST** DR11 — survey spectra via its SSAP service.
- **BeSS** — the Be Star Spectra database, the reason Be monitoring by
  amateurs is worth doing.

> This research has made use of the SIMBAD database and the VizieR catalogue
> access tool, CDS, Strasbourg, France (Wenger et al. 2000; Ochsenbein et al.
> 2000). Anyone publishing results obtained with this program should carry
> the same acknowledgement, plus the catalogue citations above.

### External programs

- [**ASTAP**](https://www.hnsky.org/astap.htm) — the Astrometric STAcking
  Program, which does the plate solving behind "Solve to WCS". An external
  prerequisite, not bundled.
- [**N.I.N.A.**](https://nighttime-imaging.eu/) — Nighttime Imaging 'N'
  Astronomy, and its contributors. The capture software the remote rig runs
  on.
- [**ninaAPI**](https://github.com/christian-photo/ninaAPI), the Advanced API
  plugin for N.I.N.A. by christian-photo (MPL-2.0) — the REST and WebSocket
  interface that `nina_client.py` drives for probing, autofocus, capture and
  slewing. Its [API documentation](https://bump.sh/christian-photo/doc/advanced-api/)
  is what the client was written against.

### Libraries

astropy, astroquery, photutils, specutils, numpy, scipy, matplotlib,
scikit-image, pandas, Pillow, auto-stretch. Versions and the load-bearing
floors are in `requirements.txt`.

### AI assistance

Much of this code was written with Anthropic's Claude as a pair-programming
assistant — **Claude Opus 4.8**, **Claude Fable 5** and **Claude Opus 5**.
The spectral-type reference cards under `ReferenceLibrary/notes/` were also
drafted with it, against the classification sources listed above.

The instrument, the observations, the measurements and the decisions about
what was worth building are the author's. Anything wrong here is too.

## Licence

MIT — see `LICENSE`. That covers the code and the instrument-response curves
measured for it (`telluric.dat`, `touptek-response.dat`). It does **not**
cover the bundled third-party data credited above — the Pickles library and
the catalogue CSVs keep their own terms and citation expectations.
