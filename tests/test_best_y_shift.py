"""Self-check for best_y_shift — centring the extraction band on the trace.

Synthesises a rotated frame holding a dispersed trace at a known y, plus the
saturated zero-order blob that sits at the start of every real strip, and
asserts:
  - a correctly placed aperture is left alone (shift 0),
  - a misplaced aperture is corrected, both signs,
  - the saturated zero order does not drag the answer, because the score
    ignores those columns — and DOES drag it when they are included, which
    is why `cols` exists,
  - a frame with no trace is left alone rather than shifted onto noise,
  - the scan reports its score curve so a peak pinned at the scan edge is
    visible to the caller.

Run: py -3.13 tests/test_best_y_shift.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectrum_core import best_y_shift   # noqa: E402

H, W = 120, 1200
FWHM = 7.5
SIGMA = FWHM / 2.355
SRC_X = 40                     # zero order sits here; the strip starts here
TRACE_LO, TRACE_HI = 560, 1080  # dispersed part, in image columns
APER_HALF = 6                  # a capped aperture, as the GUI now fits
SKY_GAP, SKY_WIDTH = 5, 20
SPECTRUM_WIDTH = 1100
# Strip-relative columns holding the dispersed trace (the strip starts at x).
DISPERSED = slice(TRACE_LO - SRC_X, TRACE_HI - SRC_X)


def make_frame(trace_y, zero_order=True, noise=2.0, sky=100.0, seed=0,
               amp=200.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), sky, dtype=float)
    on = (xx >= TRACE_LO) & (xx < TRACE_HI)
    prof = amp * (1.0 + 0.5 * np.sin((xx - TRACE_LO) / 90.0))
    img += np.where(on, prof * np.exp(-((yy - trace_y) ** 2)
                                      / (2 * SIGMA ** 2)), 0.0)
    if zero_order:
        # Saturated blob at the source, deliberately 4 rows OFF the trace:
        # the bad DAO centroid this function exists to overrule.
        blob = 60000.0 * np.exp(-(((yy - (trace_y - 4)) ** 2)
                                  + ((xx - SRC_X) ** 2)) / (2 * FWHM ** 2))
        img += np.minimum(blob, 65535.0)
    return img + rng.normal(0.0, noise, img.shape)


def scan(img, y, cols=DISPERSED, max_shift=5):
    return best_y_shift(SRC_X, y, img, SPECTRUM_WIDTH, APER_HALF,
                        SKY_GAP, SKY_WIDTH, cols=cols, max_shift=max_shift)


def test_correct_placement_is_left_alone():
    trace_y = 60
    shift, scores = scan(make_frame(trace_y), trace_y)
    assert shift == 0, (shift, scores)


def test_misplacement_is_corrected_both_signs():
    trace_y = 60
    for err in (-4, -2, 2, 4):
        # The aperture sits at trace_y + err; the fix is -err.
        shift, scores = scan(make_frame(trace_y), trace_y + err)
        assert shift == -err, (err, shift, scores)


def test_zero_order_is_why_cols_exists():
    # Trace at 60, saturated blob at 56, aperture correctly on the trace.
    trace_y = 60
    shift, _ = scan(make_frame(trace_y), trace_y)
    assert shift == 0, shift
    # Scoring every column lets the blob outweigh the whole trace and pulls
    # the aperture off the spectrum and onto the saturated star.
    shift_all, _ = scan(make_frame(trace_y), trace_y, cols=None)
    assert shift_all != 0, (
        "the zero order no longer dominates an unrestricted score - the "
        "test no longer demonstrates why cols= is needed")


def test_no_trace_stays_put():
    # Sky only: nothing to centre on, so the aperture must not move.
    trace_y = 60
    img = make_frame(trace_y, zero_order=False, amp=0.0)
    shift, _ = scan(img, trace_y)
    assert shift == 0, shift


def test_scores_are_reported_for_every_shift():
    trace_y = 60
    shift, scores = scan(make_frame(trace_y), trace_y + 2, max_shift=5)
    assert set(scores) == set(range(-5, 6)), sorted(scores)
    assert shift == -2, shift
    # The score curve must actually peak at the answer, not merely differ.
    assert max(scores, key=scores.get) == shift
    assert scores[shift] > scores[shift + 3] > 0


def test_strip_clipped_by_the_frame_edge():
    # extract_spectrum clips at x_end = min(x + width, w), so a source near
    # the right edge yields fewer columns than spectrum_width.  A caller
    # sizing its mask from spectrum_width would hand in a too-long boolean
    # and get an IndexError; sizing from the real strip works.
    trace_y = 60
    img = make_frame(trace_y)
    src_x = W - 300                      # only 300 columns left, not 1100
    n_real = min(SPECTRUM_WIDTH, W - src_x)
    cols = np.zeros(n_real, dtype=bool)
    cols[50:] = True                     # skip this strip's own zero order
    shift, scores = best_y_shift(src_x, trace_y + 3, img, SPECTRUM_WIDTH,
                                 APER_HALF, SKY_GAP, SKY_WIDTH, cols=cols,
                                 max_shift=5)
    assert scores, "no shift scored on a clipped strip"
    assert shift == -3, (shift, scores)


def test_edge_pinned_peak_is_visible_to_the_caller():
    # Trace 5 rows out but only +-2 searched: the best shift sits ON the
    # scan edge, which the caller can detect from the returned curve.
    trace_y = 60
    shift, scores = scan(make_frame(trace_y), trace_y + 5, max_shift=2)
    assert shift == -2, (shift, scores)
    assert shift == min(scores), "expected the peak pinned at the low edge"


if __name__ == "__main__":
    test_correct_placement_is_left_alone()
    test_misplacement_is_corrected_both_signs()
    test_zero_order_is_why_cols_exists()
    test_no_trace_stays_put()
    test_scores_are_reported_for_every_shift()
    test_strip_clipped_by_the_frame_edge()
    test_edge_pinned_peak_is_visible_to_the_caller()
    print("test_best_y_shift: all checks passed")
