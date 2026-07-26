"""
wavelength.py
=============
Central reference for spectral line wavelengths.

Single source of truth for line wavelengths used throughout the
application.  Two consumers:

  1. Reference-line overlays on the spectrum panels — all entries are
     drawn so the user can identify any feature they see.
  2. Calibration dialog quick-pick — only entries flagged quickpick=True
     get a button.  The criterion is "strong, isolated, well-spaced
     lines worth anchoring a polynomial fit on."  A line being absent
     from quick-pick doesn't prevent calibration on it: the user can
     always type the wavelength manually.

All wavelengths are air values in Ångström as commonly tabulated in
stellar spectroscopy references (NIST ASD for He I; standard Balmer
series).  Telluric bands are quoted at their conventional band-centre.
Wolf-Rayet C III/C IV blends are tagged as such; the wavelength is the
conventional blend centroid used for ID, not a single transition.

Each line is (wavelength_A, label, quickpick).
"""

SPECTRAL_LINES = {
    "Balmer": {
        "colour": "#ff6060",
        "lines": [
            (3970.1, "Hε 3970", True),
            (4101.7, "Hδ 4102", True),
            (4340.5, "Hγ 4340", True),
            (4861.3, "Hβ 4861", True),
            (6562.8, "Hα 6563", True),
        ],
    },
    "Helium": {
        "colour": "#ffd060",
        "lines": [
            (4026.2, "He I 4026",  False),
            (4471.5, "He I 4472",  False),
            (4685.7, "He II 4686", True),   # strongest WN line; WR/Of + hot star classifier
            (4713.1, "He I 4713",  False),
            (4921.9, "He I 4922",  False),
            (5015.7, "He I 5016",  False),
            (5411.5, "He II 5411", True),   # WN subtype ratio: He II 5411 / He I 5876
            (5875.6, "He I 5876",  True),   # strong in B/A stars; WN subtype denominator
            (6560.0, "He II 6560", False),  # blends with Hα in WR; distinguishable in WN
            (6678.2, "He I 6678",  False),
            (7065.2, "He I 7065",  False),
        ],
    },
    "Oxygen": {
        "colour": "#90ee90",
        "lines": [
            (6300.3, "O I 6300", False),
            (6363.8, "O I 6364", False),
            (7771.9, "O I 7772", False),
            (7774.2, "O I 7774", False),
            (7775.4, "O I 7775", False),
            (8446.4, "O I 8446", False),
        ],
    },
    "Carbon": {
        "colour": "#ffb347",
        "lines": [
            (4267.0, "C II 4267",      False),
            (4650.0, "C III/IV 4650",  True),    # WR blend (C III 4647/4650 + C IV 4658)
            (4745.0, "C I 4745",       False),
            (5052.2, "C I 5052",       False),
            (5380.3, "C I 5380",       False),
            (5801.0, "C IV 5801",      True),    # strong in WR/Of spectra
            (6578.1, "C II 6578",      False),
            (6582.9, "C II 6583",      False),
            (7113.2, "C I 7113",       False),
            (8335.1, "C I 8335",       False),
        ],
    },
    "Atmospheric": {
        "colour": "#80d0ff",
        "lines": [
            (6867.0, "O₂ B 6867", True),    # sharp B-band
            (7594.0, "O₂ A 7594", True),    # sharp A-band, anchors red end
            (7186.0, "H₂O 7186",  False),   # broad, less precise
            (8227.0, "H₂O 8227",  False),
        ],
    },
    "Other": {
        "colour": "#c0c0c0",
        "lines": [
            (5892.9, "Na D 5893", True),    # strong, narrow doublet centre
            (5460.7, "Hg 5461",   False),
            (6598.9, "Ne 6599",   False),
        ],
    },
    # ── Wolf-Rayet groups ──────────────────────────────────────────────────
    # Lines are broad emission features (FWHM often 20–100 Å in real WR spectra),
    # so wavelengths here are the conventional blend centroids used for
    # classification (Crowther et al. 2022, MNRAS; Smith et al. 1996, MNRAS 281).
    # Duplicating He II 4686 / 5411 from the Helium group is intentional: when
    # observing a WR star the user will likely enable the WR overlay instead of
    # the general Helium one, and the WR groups should be self-contained.
    "Wolf-Rayet WN": {
        "colour": "#7ecfff",   # cool blue — ionised nitrogen / hot plasma
        "lines": [
            # He II — classification ratios and strongest lines in WN spectra
            (4685.7, "He II 4686",  True),   # dominant WN emission; WR bump centre
            (5411.5, "He II 5411",  True),   # WN subtype: He II 5411 / He I 5876 ratio
            # N III doublet — blue-bump WN component; late WN (WN6–9)
            (4634.0, "N III 4634",  True),   # Crowther et al. 2022 λλ4634,41
            (4641.0, "N III 4641",  True),
            # N IV — prominent in intermediate WN subtypes
            (4057.76, "N IV 4058",  False),  # also seen in WN/C transition stars
            (7109.0, "N IV 7109",   False),  # useful red anchor; classif. diagnostic
            # N V doublet — early / hot WN (WN2–5); weak or absent in late WN
            (4603.0, "N V 4603",    False),  # Crowther et al. 2022 λλ4603,20
            (4619.0, "N V 4619",    False),
            # He I — present in late WN stars (cooler subtypes)
            (5875.6, "He I 5876",   False),  # WN subtype denominator (Smith 1996)
        ],
    },
    "Wolf-Rayet WC": {
        "colour": "#ffb870",   # warm orange — carbon-rich winds
        "lines": [
            # C III — the defining WC diagnostic; absent in WN stars
            (5696.0, "C III 5696",     True),   # Paris Obs. dict.; Royer et al. 1998
            # C III / C IV blue bump — strong in WC; overlaps the WN He II bump
            (4647.0, "C III 4647",     True),   # Crowther et al. 2022 λλ4647,51
            (4651.0, "C III 4651",     False),
            (4658.0, "C IV 4658",      False),  # blends with C III in the 4650 complex
            # C IV yellow/red bump — WC classifier (EW ratio with blue bump)
            (5801.0, "C IV 5801",      True),   # Crowther et al. 2022 λλ5801,12
            (5812.0, "C IV 5812",      False),
            # He II — present in WC, usually weaker than in WN
            (4685.7, "He II 4686",     False),  # blends with C IV 4658 complex
            (5411.5, "He II 5411",     False),
            # O III — present in late WC (WC7–9); used for WC/WO discrimination
            (5592.0, "O III 5592",     False),
        ],
    },
    "Carbon Stars": {
        "colour": "#ff9de2",
        "lines": [
            (3880.0, "CN 3880",        False),
            (4056.0, "C3 4056",        False),
            (4217.0, "CN 4217",        False),
            (4380.0, "C2 Swan 4380",   False),
            (4554.0, "Ba II 4554",     False),
            (4581.0, "SiC2 4581",      False),
            (4607.0, "Sr I 4607",      False),
            (4640.0, "SiC2 4640",      False),
            (4738.0, "C2 Swan 4738",   False),
            (4867.0, "SiC2 4867",      False),
            (4906.0, "SiC2 4906",      False),
            (4977.0, "SiC2 4977",      False),
            (5165.0, "C2 Swan 5165",   False),
            (5635.0, "C2 Swan 5635",   False),
            (6122.0, "C2 Swan 6122",   False),
        ],
    },
}


def flatten_group(group_name):
    """
    Build a flat {wavelength: label} dict for one group from SPECTRAL_LINES.

    Used to derive the legacy per-element dicts (BALMER_LINES, etc.)
    consumed by plot_reference_lines and _draw_reference_line_groups.
    Adding a line in SPECTRAL_LINES automatically updates both the
    overlay (via the legacy dicts) and the calibration dialog quick-pick
    (which reads SPECTRAL_LINES directly).
    """
    return {wl: label for wl, label, _qp in SPECTRAL_LINES[group_name]["lines"]}


# Legacy flat dicts — derived from SPECTRAL_LINES so adding a line in one
# place updates both the overlay and (if quickpick=True) the dialog.
# Consumed by plot_reference_lines (in spectrum_core) and by
# _draw_reference_line_groups (in the main GUI module).
BALMER_LINES      = flatten_group("Balmer")
HELIUM_LINES      = flatten_group("Helium")
OXYGEN_LINES      = flatten_group("Oxygen")
CARBON_LINES      = flatten_group("Carbon")
ATMOSPHERIC_LINES = flatten_group("Atmospheric")
CARBON_STAR_LINES = flatten_group("Carbon Stars")
WR_WN_LINES       = flatten_group("Wolf-Rayet WN")
WR_WC_LINES       = flatten_group("Wolf-Rayet WC")