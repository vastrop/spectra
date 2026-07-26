"""rotate_band == rotate(reshape=False)[y0:y1], for a fraction of the cost.

The autofocus reduction leans on that equivalence: once the strip's position is
known it derotates only the strip's rows, so if this drifts, every focus metric
drifts with it silently.

Run:  py -3.13 tests/test_rotate_band.py
"""
import os
import sys
import time

import numpy as np
from scipy.ndimage import rotate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectrum_core import rotate_band


def main():
    rng = np.random.default_rng(20260715)

    # Structure, not just noise: a smooth gradient with stars on top, so an
    # interpolation error shows up instead of averaging away.
    h, w = 600, 900
    yy, xx = np.mgrid[0:h, 0:w]
    img = (1000.0 + 0.4 * xx + 0.2 * yy
           + rng.normal(0, 5.0, size=(h, w)))
    for _ in range(40):
        sy, sx = rng.integers(20, h - 20), rng.integers(20, w - 20)
        img += 4000.0 * np.exp(-(((yy - sy) ** 2 + (xx - sx) ** 2) / (2 * 2.5 ** 2)))

    cval = float(np.median(img))

    for angle in (0.0, 5.18, -3.4, 12.0, 0.25):
        full = rotate(img, angle, reshape=False, cval=cval)
        for y0, y1 in ((250, 350),        # interior band
                       (0, 64),           # top edge — samples off-frame
                       (h - 64, h),       # bottom edge
                       (0, h)):           # the whole frame
            band = rotate_band(img, angle, y0, y1, cval=cval)
            assert band.shape == (y1 - y0, w), (band.shape, y1 - y0, w)
            ref = full[y0:y1]
            err = np.abs(band - ref).max()
            scale = max(1.0, float(np.abs(ref).max()))
            assert err / scale < 1e-6, (
                f"angle {angle}, rows [{y0},{y1}): max abs error {err:.3e} "
                f"(relative {err / scale:.2e})")

    # Bad row ranges are a programming error, not a silent empty band.
    for bad in ((-1, 10), (0, 0), (10, 5), (0, h + 1)):
        try:
            rotate_band(img, 5.0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"rotate_band accepted rows {bad}")

    # The whole point: a narrow band must be much cheaper than the full rotate.
    big = rng.normal(1000, 10, size=(2160, 3840))
    t0 = time.perf_counter()
    rotate(big, 5.18, reshape=False, cval=1000.0)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    rotate_band(big, 5.18, 1000, 1400, cval=1000.0)
    t_band = time.perf_counter() - t0
    assert t_band < t_full / 2, (
        f"400-row band took {t_band*1000:.0f} ms vs {t_full*1000:.0f} ms full — "
        f"no saving")

    print(f"full rotate {t_full*1000:6.0f} ms   400-row band {t_band*1000:6.0f} ms   "
          f"({t_full/t_band:.1f}x)")
    print("rotate_band self-check OK")


if __name__ == "__main__":
    main()
