"""Self-check for contaminators_from_sources — matching an extraction strip
against the frame-wide source list.

The original design ran DAOStarFinder on the aperture cutout itself.  That
could not see a star whose PSF was only partly inside the band — precisely
the star that spills flux into the science spectrum.  Sources are now
detected once on the whole frame and the strip asks which of them land in
it (its own rows, plus one FWHM above and below).

Asserts, with sources given in rotated-frame coordinates:
  - an on-trace emission line (WR/nova peak) is dropped,
  - a far off-trace star inside the band is kept,
  - a NEAR off-trace star (~1 FWHM out, like the real V2014 Aql source-7
    contaminant) is kept too — the exclusion zone must stay tight,
  - a star straddling the band edge, whose PSF is only half inside, is now
    kept: the case a strip-local detection structurally cannot see,
  - a star further than the pad is not,
  - the target's own zero order is dropped,
  - sources outside the strip's columns are ignored,
  - turning the exclusion off brings the emission line back, proving the
    y-discriminant is what removes it.

Run: py -3.13 tests/test_contaminator_detection.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectrum_core import contaminators_from_sources   # noqa: E402

# The library default; the explorer passes its own TRACE_EXCLUDE_FWHM.
TRACE_EXCLUDE_DEFAULT = 0.2

FWHM = 4.0
TARGET_X, TARGET_Y = 100.0, 200.0
APER_HALF = 6
# The bbox extract_spectrum would build for this target.
BBOX = dict(x_start=int(TARGET_X), x_end=int(TARGET_X) + 200,
            y_start=int(TARGET_Y) - APER_HALF,
            y_end=int(TARGET_Y) + APER_HALF + 1)

ZERO_ORDER = (TARGET_X, TARGET_Y)                 # the target itself
EMISSION = (TARGET_X + 60, TARGET_Y)              # its own line, on-trace
FAR = (TARGET_X + 120, TARGET_Y - 5.0)            # another star, in-band
NEAR = (TARGET_X + 150, TARGET_Y + 1.0 * FWHM)    # ~1 FWHM off the trace
STRADDLE = (TARGET_X + 170, TARGET_Y + APER_HALF + 2.0)   # half in the band
OUTSIDE_Y = (TARGET_X + 180, TARGET_Y + APER_HALF + 3.0 * FWHM)  # too far
OUTSIDE_X = (TARGET_X + 400, TARGET_Y)            # past the strip's columns


def cols_for(sources, **kw):
    cols, _on_trace = contaminators_from_sources(sources, BBOX, FWHM,
                                                 TARGET_Y, **kw)
    return cols


def on_trace_for(sources, **kw):
    _cols, on_trace = contaminators_from_sources(sources, BBOX, FWHM,
                                                 TARGET_Y, **kw)
    return on_trace


def near(cols, src):
    """Is this source's column among the returned (strip-relative) ones?"""
    want = src[0] - BBOX["x_start"]
    return bool(len(cols)) and np.any(np.abs(np.asarray(cols) - want) < 1e-6)


ALL = [ZERO_ORDER, EMISSION, FAR, NEAR, STRADDLE, OUTSIDE_Y, OUTSIDE_X]


def test_the_discriminants():
    cols = cols_for(ALL)
    assert near(cols, FAR), f"far off-trace contaminator missed: {cols}"
    assert near(cols, NEAR), f"near (~1 FWHM) contaminator wrongly killed: {cols}"
    assert not near(cols, EMISSION), f"emission line falsely flagged: {cols}"
    assert not near(cols, ZERO_ORDER), f"target's own zero order flagged: {cols}"


def test_straddling_star_is_the_point_of_the_rewrite():
    # 2 px outside a +-6 aperture: its PSF reaches in, but a DAOStarFinder
    # run on the cutout would never have found it (no whole PSF in there).
    cols = cols_for(ALL)
    assert near(cols, STRADDLE), f"straddling contaminator missed: {cols}"


def test_pad_has_a_limit():
    cols = cols_for(ALL)
    assert not near(cols, OUTSIDE_Y), f"star 3 FWHM clear of the band: {cols}"
    # ...and the pad is what admits the straddler, so a zero pad drops it.
    tight = cols_for(ALL, y_pad_fwhm=0.0)
    assert not near(tight, STRADDLE)
    assert near(tight, FAR), "in-band star must survive a zero pad"


def test_columns_outside_the_strip_are_ignored():
    cols = cols_for(ALL)
    assert not near(cols, OUTSIDE_X), f"source past x_end flagged: {cols}"


def test_isolation_the_exclusion_is_what_drops_the_emission_line():
    # With the exclusion disabled the line reappears — so it WAS in range
    # and the y-discriminant is what suppressed it.
    cols0 = cols_for(ALL, trace_exclude_fwhm=0.0)
    assert near(cols0, EMISSION), f"emission not seen with exclusion off: {cols0}"
    assert near(cols0, ZERO_ORDER), "zero order should reappear too"


def test_on_trace_drops_are_reported_not_silent():
    """The on-trace test is a judgement call geometry cannot make — a star
    sitting on the trace looks exactly like an emission line — so what it
    discards has to be visible to the caller."""
    on_trace = on_trace_for(ALL)
    cols = list(on_trace[:, 0])
    assert len(on_trace) == 2, on_trace          # the emission line + zero order
    assert EMISSION[0] - BBOX["x_start"] in cols
    assert ZERO_ORDER[0] - BBOX["x_start"] in cols
    # Each carries its offset from the trace, which is the number to tune the
    # zone against.
    for _col, dy in on_trace:
        assert abs(dy) < TRACE_EXCLUDE_DEFAULT * FWHM, dy
    # Nothing outside the band is reported: these are drops, not a source dump.
    assert not np.any(np.isclose(cols, OUTSIDE_X[0] - BBOX["x_start"]))


def test_degenerate_inputs():
    assert len(cols_for([])) == 0
    assert len(cols_for(np.empty((0, 2)))) == 0
    assert len(on_trace_for([])) == 0
    for bad_fwhm in (np.nan, 0.0):
        cols, on_trace = contaminators_from_sources(ALL, BBOX, bad_fwhm,
                                                    TARGET_Y)
        assert len(cols) == 0 and len(on_trace) == 0


if __name__ == "__main__":
    test_the_discriminants()
    test_straddling_star_is_the_point_of_the_rewrite()
    test_pad_has_a_limit()
    test_columns_outside_the_strip_are_ignored()
    test_isolation_the_exclusion_is_what_drops_the_emission_line()
    test_on_trace_drops_are_reported_not_silent()
    test_degenerate_inputs()
    print("OK - emission and zero order dropped; in-band, near and "
          "straddling contaminators kept")
