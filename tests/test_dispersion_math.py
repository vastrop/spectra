"""
Assert-based self-checks for the dispersion maths moved out of the GUI
(review R2.5): fit_dispersion_poly (including the zero-order anchor
recomposition), validate_dispersion_poly, dispersion_fit_stats, and
build_sky_col_flag.

Run:  py -3.13 tests\test_dispersion_math.py
"""

import os
import sys

import numpy as np

# The module under test lives in the repo root, one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectrum_core import (
    fit_dispersion_poly,
    validate_dispersion_poly,
    dispersion_fit_stats,
    build_sky_col_flag,
    suggest_dispersion_nodes,
    AUTO_BALMER_A,
)


TRUE_DISP, TRUE_X0 = 7.684, -4.0     # measured on cornercheck/CIG.fit


def synth_a_star(n=1150, x0=TRUE_X0, disp=TRUE_DISP, seed=0):
    """Synthetic A-type strip with Balmer absorption at a known (x0, Å/px).

    Deliberately includes the DEAD BLUE END a real strip has: the extraction
    starts at the zero order, so the first few hundred columns sit below the
    atmospheric cutoff where the star has no flux and the frame has only
    noise.  That region is what made a free-scale scan misbehave — depth
    normalised by a near-zero continuum outscores any real line — so a
    synthetic without it would not exercise the guard.
    """
    rng = np.random.default_rng(seed)
    px = np.arange(n, dtype=float)
    wl = (px - x0) * disp

    # Smoothstep turn-on, not a linear ramp: a ramp has a hard corner at each
    # end, and _dip_strength fits a straight continuum across its window, so
    # a corner reads as a deep "line".  That is a defect of synthetic data —
    # a real cutoff rolls off smoothly — and it would hand the scan a feature
    # to lock onto that no real spectrum has.
    t = np.clip((wl - 3500.0) / 700.0, 0.0, 1.0)
    turn_on = t * t * (3.0 - 2.0 * t)             # zero slope at both ends
    cont = 1000.0 * turn_on * np.exp(
        -(wl - 4600.0) ** 2 / (2 * 2600.0 ** 2))

    # Depth proportional to the local continuum → constant fractional depth,
    # which is what _dip_strength measures.
    for lam in AUTO_BALMER_A:
        centre = x0 + lam / disp
        cont -= 0.35 * cont * np.exp(-(px - centre) ** 2 / (2 * 4.0 ** 2))

    return cont + rng.normal(0.0, 3.0, n)


def test_fit_dispersion_poly():
    # Too few nodes / bad input → None
    assert fit_dispersion_poly(None) is None
    assert fit_dispersion_poly([[100.0, 4000.0]]) is None

    # Two nodes → exact linear fit
    nodes = [[100.0, 4000.0], [600.0, 7850.0]]
    poly = fit_dispersion_poly(nodes)
    assert poly is not None and len(poly) == 2
    assert abs(np.polyval(poly, 100.0) - 4000.0) < 1e-9
    assert abs(np.polyval(poly, 600.0) - 7850.0) < 1e-9

    # Degree caps at 3 for many nodes
    many = [[float(p), 3800.0 + 7.7 * p + 1e-4 * p * p]
            for p in range(0, 1000, 100)]
    poly = fit_dispersion_poly(many)
    assert len(poly) == 4  # deg 3 → 4 coefficients

    # Anchor recomposition is EXACT: poly'(x) == poly(x − Δ) everywhere.
    # This is the shape-preservation property — the shift must not bend
    # the fit, only translate its argument.
    delta = 0.37
    base = fit_dispersion_poly(many)
    shifted = fit_dispersion_poly(many, delta=delta)
    x = np.linspace(0.0, 1000.0, 501)
    assert np.allclose(np.polyval(shifted, x),
                       np.polyval(base, x - delta), rtol=0, atol=1e-6)

    # Δ = 0 is bit-identical to the unshifted fit
    assert np.array_equal(fit_dispersion_poly(many, delta=0.0), base)


def test_validate_dispersion_poly():
    # Monotonic linear map passes through unchanged
    lin = np.array([7.7, 3800.0])
    poly, n_bad = validate_dispersion_poly(lin, 1000)
    assert poly is lin and n_bad == 0

    # Nothing to check → passthrough
    poly, n_bad = validate_dispersion_poly(None, 1000)
    assert poly is None and n_bad == 0
    poly, n_bad = validate_dispersion_poly(lin, None)
    assert poly is lin and n_bad == 0

    # A downward parabola turns over inside the pixel range → rejected,
    # with a positive bad-step count
    bad = np.array([-0.02, 10.0, 4000.0])  # vertex at px 250
    poly, n_bad = validate_dispersion_poly(bad, 1000)
    assert poly is None and n_bad > 0


def test_dispersion_fit_stats():
    assert dispersion_fit_stats([[1.0, 2.0]]) is None

    # Two nodes: deg 1, exact by construction, dispersion = slope
    nodes = [[100.0, 4000.0], [600.0, 7850.0]]
    s = dispersion_fit_stats(nodes, n_pixels=1000)
    assert s["deg"] == 1 and s["exact"] and s["monotonic"]
    assert abs(s["disp_min"] - 7.7) < 1e-9 and abs(s["disp_max"] - 7.7) < 1e-9
    assert s["rms"] < 1e-9

    # Six near-linear nodes: deg 3, over-determined → not exact, small RMS
    rng = [(p, 3800.0 + 7.7 * p + (1.0 if p == 300 else 0.0))
           for p in (100.0, 200.0, 300.0, 400.0, 500.0, 600.0)]
    s = dispersion_fit_stats(list(map(list, rng)), n_pixels=1000)
    assert s["deg"] == 3 and not s["exact"]
    assert 0.0 < s["rms"] < 1.0
    assert s["monotonic"]


def test_build_sky_col_flag():
    # No masks at all → None
    assert build_sky_col_flag(None, None) is None
    assert build_sky_col_flag(np.zeros((0, 5), bool), None) is None

    # Nothing flagged → None (falls back to per-frame sky)
    clean = np.zeros((4, 6), dtype=bool)
    assert build_sky_col_flag(clean, clean) is None

    # Column 2 fully rejected in both bands → flagged; column 3 rejected
    # in 2 of 8 rows (25% < 50%) → not flagged
    lo = np.zeros((4, 6), dtype=bool)
    hi = np.zeros((4, 6), dtype=bool)
    lo[:, 2] = True
    hi[:, 2] = True
    lo[0, 3] = True
    hi[0, 3] = True
    flag = build_sky_col_flag(lo, hi)
    assert flag is not None
    assert flag[2] and not flag[3] and not flag[0]

    # One band missing: fractions computed over the surviving rows only
    flag = build_sky_col_flag(lo, None)
    assert flag is not None and flag[2] and not flag[3]


def test_suggest_nodes_recovers_scale_from_bad_prior():
    """The scale search is what makes the initial Å/px non-critical.

    A new user cannot supply a correct dispersion — it is the thing the
    calibration measures.  These priors are ±15%, far past the ~1-2% the
    old fixed-scale scan tolerated before it locked onto the wrong lines.
    Kept inside ±25% rather than exactly on it, so the truth sits properly
    within the band and the edge-pinned assertion below means something.
    """
    spec = synth_a_star()

    for prior in (TRUE_DISP * 0.85, TRUE_DISP, TRUE_DISP * 1.15):
        nodes, info = suggest_dispersion_nodes(spec, prior)
        assert nodes, f"prior {prior:.2f}: {info.get('error')}"
        assert abs(info["dispersion"] - TRUE_DISP) < 0.05, (prior, info)
        assert info["n_balmer"] == 4, (prior, info)

    # The prior is reported back untouched, and the winning grid value is
    # distinct from the fitted slope (which comes from refined centroids).
    _nodes, info = suggest_dispersion_nodes(spec, 9.0)
    assert info["dispersion_prior"] == 9.0
    assert abs(info["dispersion_scan"] - TRUE_DISP) < 0.3, info


def test_suggest_nodes_fixed_scale_cannot_recover():
    """search_frac=0 pins the scale to the prior and fails on the same
    20%-off prior the search handles.  This is what establishes that the
    scale search, and not something else, is doing the work; if it ever
    starts passing, that is no longer true."""
    spec = synth_a_star()
    nodes, info = suggest_dispersion_nodes(spec, TRUE_DISP * 0.85,
                                           search_frac=0.0)
    recovered = bool(nodes) and abs(info.get("dispersion", 0.0)
                                    - TRUE_DISP) < 0.05
    assert not recovered, info

    # ...while a correct prior still works with the scale pinned.
    nodes, info = suggest_dispersion_nodes(spec, TRUE_DISP, search_frac=0.0)
    assert nodes and abs(info["dispersion"] - TRUE_DISP) < 0.05, info


def test_suggest_nodes_ignores_dead_blue_end():
    """Regression: columns below the atmospheric cutoff have a near-zero
    continuum, so _dip_strength's depth/continuum is noise over noise there.
    Unguarded, that region outscored every real Balmer line and pinned the
    scan at the top of the band (measured on cornercheck: 9.66 Å/px with all
    four lines parked below 3000 Å)."""
    spec = synth_a_star()
    nodes, _info = suggest_dispersion_nodes(spec, TRUE_DISP)
    assert nodes
    # Hδ, the bluest line, sits at ~530 px; anything under 450 means the
    # scan locked onto the dead end.
    assert min(p for p, _ in nodes) > 450.0, nodes


def test_suggest_nodes_rejects_shifted_line_aliases():
    """The alias the focus runs exposed.

    A shifted assignment — Hδ landing where Hγ really is, Hγ where Hβ is —
    is self-consistent enough to score well, and lands on scales of
    TRUE × 4340.5/4861.3 ≈ 0.893 and TRUE × 4861.3/6562.8 ≈ 0.741.  Those
    were the real attractors: with three lines required and an arithmetic
    mean, they took 69 of 93 frames across focus_runs.  Seeded directly at
    each alias, the scan must still come back to the truth.

    Only aliases that keep the truth inside the ±25% band are testable here:
    the Hβ→Hα alias at 0.741 puts it out of reach (5.69 × 1.25 = 7.11 <
    7.684).  That is the capture-range limit rather than an alias failure —
    see test_suggest_nodes_outside_capture_range_is_not_self_diagnosing.
    """
    spec = synth_a_star()
    for ratio in (4340.5 / 4861.3, 4101.7 / 4340.5,      # aliases from below
                  4861.3 / 4340.5, 4340.5 / 4101.7):     # and from above
        prior = TRUE_DISP * ratio
        _nodes, info = suggest_dispersion_nodes(spec, prior)
        assert abs(info.get("dispersion", 0.0) - TRUE_DISP) < 0.05, \
            (ratio, prior, info)


def test_suggest_nodes_needs_every_balmer_line():
    """A shallow spectrum missing one line must not be rescued by the other
    three averaging it away — that leniency is what let the aliases in.
    Blanking Hβ leaves a strip the scale search should decline rather than
    fit, since three lines cannot pin two free parameters."""
    spec = synth_a_star()
    centre = int(TRUE_X0 + 4861.3 / TRUE_DISP)
    # Replace the Hβ dip with the local continuum level.
    spec[centre - 20:centre + 21] = np.median(
        np.r_[spec[centre - 60:centre - 30], spec[centre + 30:centre + 60]])
    _nodes, info = suggest_dispersion_nodes(spec, TRUE_DISP)
    assert abs(info.get("dispersion", 0.0) - TRUE_DISP) > 0.05 \
        or not _nodes, info


def test_suggest_nodes_outside_capture_range_is_not_self_diagnosing():
    """Pins the honest limit: a prior 40% off puts the truth outside the
    band, and what comes back is a CONFIDENT wrong answer — an interior
    solution, nothing edge-pinned, sub-Å linear residuals.  An edge-pinning
    check does not detect this case, which is why the calibration dialog
    plots the nodes for approval instead of trusting them.  A future
    detector for this failure should be proved here."""
    spec = synth_a_star()
    _nodes, info = suggest_dispersion_nodes(spec, TRUE_DISP * 0.6)
    assert abs(info["dispersion"] - TRUE_DISP) > 1.0, info
    assert max(abs(r) for r in info["residuals"]) < 5.0, info


if __name__ == "__main__":
    test_fit_dispersion_poly()
    test_validate_dispersion_poly()
    test_dispersion_fit_stats()
    test_build_sky_col_flag()
    test_suggest_nodes_recovers_scale_from_bad_prior()
    test_suggest_nodes_fixed_scale_cannot_recover()
    test_suggest_nodes_ignores_dead_blue_end()
    test_suggest_nodes_rejects_shifted_line_aliases()
    test_suggest_nodes_needs_every_balmer_line()
    test_suggest_nodes_outside_capture_range_is_not_self_diagnosing()
    print("test_dispersion_math: all checks passed")
