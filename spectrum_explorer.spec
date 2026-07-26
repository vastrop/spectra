# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Spectrum Explorer — onedir, windowed, Windows, Python 3.13.
#
# Build:   py -3.13 -m PyInstaller spectrum_explorer.spec
# Output:  dist/spectrum_explorer/spectrum_explorer.exe
#
# ── Build requirements ─────────────────────────────────────────────────────
# 1. BUILD FROM AN ENVIRONMENT THAT HAS THE FULL STACK, in particular
#    specutils and astroquery. Both are imported behind try/except by design,
#    so a build from an environment missing them still *succeeds* — it just
#    permanently bakes in silent degradations (continuum auto-fit
#    unavailable). build_exe.bat hard-fails when either is absent; do not
#    route around that check.
# 2. PyInstaller is a build-only dependency, not in requirements.txt:
#      python -m pip install "pyinstaller>=6.10"
#    6.10+ is required for Python 3.13. Keep pyinstaller-hooks-contrib
#    current; the astropy-iers-data and asdf hooks live there.
# 3. Prefer a clean virtual environment holding only the packages in the
#    version table below. A shared site-packages tends to carry torch,
#    tensorflow, cv2, pyarrow and similar multi-GB dependencies; the excludes
#    below guard against accidental pull-in, but a minimal environment is the
#    robust route.
#
# ── External prerequisite (not bundled) ─────────────────────────────────────
# ASTAP (plate solver) — user-installed, default C:/Program Files/astap/astap.exe,
# path configurable in the GUI and persisted in the analysis config. Document
# it in the install notes; do not bundle.
#
# ── Dependency inventory ────────────────────────────────────────────────────
# Direct imports of the entry-point graph, with the versions this spec was
# exercised against:
#   numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, matplotlib 3.10.9 (TkAgg forced),
#   astropy 7.2.0, photutils 3.0.0, scikit-image 0.26.0, requests 2.34.2,
#   auto-stretch 1.0.0 (import name auto_stretch; pure Python, no data files)
#   NB: requirements.txt floors astropy>=8.0 and auto-stretch>=1.1, which are
#   higher than the versions above. Build against what is installed rather
#   than upgrading the build environment to match the file.
# Deep / implied:
#   astropy-iers-data 0.2025.2.3 (separate dist since astropy 6 — bundled by
#     the contrib hook; keep pyinstaller-hooks-contrib current),
#   pyerfa 2.0.1.5, PyYAML 6.0.2 (astropy), certifi 2025.1.31 (requests),
#   Pillow 11.0.0 (matplotlib.animation.PillowWriter — GIF export in
#     sequence_generator.py, a top-level import, found automatically),
#   lazy-loader 0.5 + imageio/tifffile/networkx (skimage),
#   tkinter/tcl-tk data (bundled automatically from the python.org install)
# Lazy imports (inside function bodies / try-except — PyInstaller's bytecode
# scan finds them once installed, hiddenimports below are belt-and-braces
# because an ImportError there is silently swallowed by design):
#   specutils(.fitting/.spectra) + astropy.modeling  — spectrum_core.py ~1206
#   astroquery.simbad                                — source_identification.py,
#                                                      lamost_dialog.py
# Once specutils/astroquery are installed they pull gwcs, asdf, asdf-astropy,
# ndcube, pyvo, keyring, beautifulsoup4 — handled by contrib hooks + the
# copy_metadata() calls below (asdf registers extensions via entry points).
# Optional GPU path: cupy (registration.py) — deliberately excluded; the
# ImportError fallback covers it.
#
# Also on the graph: nina_dialog (+ nina_client, stdlib-only),
# predictor_dialog, lamost_dialog, db.spectra_db, and the explorer/ catalogue
# browsers (nine dialogs over catalog_browser; astroquery only on their
# --refresh paths).
# nina_dialog pulls tools/spectral_autofocus.py + focus_analyzer.py lazily,
# inside the autofocus handler. Autofocus runs IN-PROCESS on a worker thread,
# with no subprocess, which is what makes it work in a frozen build.
#
# No multiprocessing in the graph (no freeze_support needed).
#
# WRITES NEXT TO THE EXECUTABLE: the frozen app writes spectra.db (the ingest)
# and focus_runs/<timestamp>/ (an autofocus sweep) into the folder holding the
# .exe — NOT into _internal/, which is where __file__ points and which a
# reinstall wipes. Both resolve off sys.executable when sys.frozen is set; see
# db/spectra_db.py and nina_dialog._DATA_ROOT. Consequences:
#   • install the onedir somewhere user-writable (NOT Program Files);
#   • to carry an existing database over, drop spectra.db beside the .exe (the
#     schema is created/migrated on open, so an older DB is fine);
#   • spectrum_config.json is unaffected — it goes through file dialogs.

from PyInstaller.utils.hooks import (copy_metadata, collect_data_files,
                                     collect_submodules)

hiddenimports = [
    # skimage >= 0.19 resolves submodules through lazy_loader at runtime;
    # rotation.py uses `from skimage import feature` and
    # `from skimage.transform import probabilistic_hough_line`.
    'skimage.feature',
    'skimage.transform',
    'skimage.measure',      # pulled internally by skimage.feature
    'skimage.draw',
    # Lazy, try/except-guarded imports — a miss is silent, so pin them:
    'specutils',
    'specutils.fitting',
    'specutils.spectra',
    'astropy.modeling',
    'astropy.modeling.fitting',
    'astroquery.simbad',
    # db/, tools/ and explorer/ have no __init__.py — they are implicit
    # namespace packages, which PyInstaller's module graph does not always
    # follow through `from db import spectra_db` (spectrum_explorer.py:62),
    # `from tools import spectral_autofocus` (nina_dialog, lazily inside the
    # autofocus handler) or `from explorer.be_star_dialog import ...`.
    # Cheap insurance.
    'db.spectra_db',
    'tools.spectral_autofocus',
    'explorer.catalog_browser',
    'explorer.be_star_dialog',
    'explorer.wr_star_dialog',
    'explorer.quasar_dialog',
    'explorer.carbon_star_dialog',
    'explorer.mira_dialog',
    'explorer.s_star_dialog',
    'explorer.cv_dialog',
    'explorer.symbiotic_dialog',
    'explorer.herbig_dialog',
    'focus_analyzer',           # the sweep's scoring, imported by the above
]

# photutils' Cython extensions cimport sibling compiled modules
# (circular_overlap → geometry.core) invisibly to static analysis, and the
# bundle crashes without them; collect the whole package rather than chase
# individual cimports.
hiddenimports += collect_submodules('photutils')

datas = [
    # tooltip_help.py resolves help/tooltips.json via __file__ — lands in
    # _internal/ where frozen __file__ points. Missing file degrades to a
    # silent no-op (tooltips vanish, no error) — verify presence post-build.
    ('help/tooltips.json', 'help'),
    # Only *.dat files are scanned by the library viewer; Makefile/.HILIB/
    # README are dev artifacts. ReferenceLibrary/notes/*.md is deliberately
    # absent: the cards are read by db/spectra_browser.py, a separate program
    # that is not part of this bundle.
    ('ReferenceLibrary/*.dat', 'ReferenceLibrary'),
    # Catalogues for the explorer/ browser dialogs (tracked in git,
    # regenerated with --refresh; resolved via __file__ -> _internal/).
    # The glob also carries nina_dialog.py's astar_catalog.csv, which
    # lives in ReferenceLibrary too — without it the A-star list is dead:
    # Connect pops "run tools/build_astar_catalog.py".
    ('ReferenceLibrary/*_catalog.csv', 'ReferenceLibrary'),
]

# The asdf ecosystem (specutils deps; astropy 8 imports asdf_astropy at
# top level via astropy.table's IO registration) registers extensions and
# resource mappings via importlib.metadata entry points — metadata must
# ship. Guarded so the spec still builds if a package is absent (the app
# then degrades as it does from source).
# photutils 3.0 additionally reads its own metadata at import
# (_optional_deps → importlib.metadata.requires('photutils')) and crashes
# without it; astroquery resolves its version the same way.
for _dist in ('asdf', 'asdf-astropy', 'asdf-standard',
              'asdf-transform-schemas', 'asdf-coordinates-schemas',
              'asdf-wcs-schemas', 'specutils', 'gwcs', 'ndcube',
              'photutils', 'astroquery'):
    try:
        datas += copy_metadata(_dist)
    except Exception:
        pass

# asdf's vendored _jsonschema reads its draft-N schema .json files at import
# (without them: FileNotFoundError on _jsonschema/schemas/draft3.json), and
# the schema packages ship .yaml resources resolved at runtime.  None are
# covered by the contrib hooks as of pyinstaller-hooks-contrib 2026.6.
for _pkg in ('asdf', 'asdf_astropy', 'asdf_standard',
             'asdf_transform_schemas', 'asdf_coordinates_schemas',
             'asdf_wcs_schemas', 'specutils', 'gwcs'):
    try:
        datas += collect_data_files(_pkg)
    except Exception:
        pass

# The SIMBAD import chain reads package data AT IMPORT, in three places:
#   astroquery/__init__          → CITATION (_get_bibtex)
#   astroquery.simbad.utils      → simbad/data/query_criteria_fields.json
#   pyvo.samp.constants          → samp/data/astropy_icon.png  (pulled in by
#                                  astroquery.simbad.core's `from pyvo.dal …`)
# Miss any one and the import raises — and worse than a plain
# FileNotFoundError: astropy's get_pkg_data_filename falls back to
# DOWNLOADING the missing file from data.astropy.org, which 404s.
# source_identification._build_simbad catches that, so the symptom is a
# frozen app that plate-solves normally and silently identifies 0 sources.
# Scoped deliberately: a bare collect_data_files() on either package would drag
# in ~120 test .vot/.xml fixtures for nothing.
try:
    datas += collect_data_files('astroquery',
                                includes=['CITATION',
                                          'simbad/data/query_criteria_fields.json'])
    datas += collect_data_files('pyvo', excludes=['**/tests/**'])
except Exception:
    pass

a = Analysis(
    ['spectrum_explorer.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        # Backend is hard-set to TkAgg (spectrum_explorer.py:27); skip the rest.
        'matplotlib': {'backends': 'TkAgg'},
    },
    runtime_hooks=[],
    excludes=[
        'cupy', 'cupyx',                    # optional GPU path, has a fallback
        'pyarrow',                          # pandas probes it → ~100 MB for nothing
        'torch', 'torchvision', 'torchaudio',
        'tensorflow', 'tensorboard', 'keras',
        'sklearn', 'cv2', 'sympy', 'h5py',  # site-wide installs, not on the graph
        'IPython', 'pytest',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='spectrum_explorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,   # windowed; ASTAP subprocess carries CREATE_NO_WINDOW
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='spectra.ico',  # none exists in the repo yet
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='spectrum_explorer',
)
