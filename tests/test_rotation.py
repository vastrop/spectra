import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotation import _hist_peak


def test_hist_peak():
    peak, support = _hist_peak(np.array([0.0, 1.0, 4.0, 1.0]), 4, 0)
    assert peak == 112.5
    assert support == 8 / 3
    assert _hist_peak(np.zeros(4), 4, 0) == (None, 0.0)


if __name__ == "__main__":
    test_hist_peak()
    print("test_rotation OK")
