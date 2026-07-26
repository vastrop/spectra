"""Self-check for estimate_planck_temperature (spectrum_core).

Synthetic continuum anchors drawn from a known Planck curve, over a
typical visible window (3900–7200 Å), for the three regimes:
in-band peak, UV-side (hot star) and IR-side (cool star).
"""

import os
import sys

import numpy as np

# The module under test lives in the repo root, one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectrum_core import estimate_planck_temperature, planck_relative


def _anchors(temp, scale=3.7, noise=0.0, seed=0):
    wl = np.linspace(3900.0, 7200.0, 10)
    fx = scale * planck_relative(wl, temp) / planck_relative(5500.0, temp)
    if noise:
        fx *= np.exp(np.random.default_rng(seed).normal(0, noise, wl.size))
    return wl, fx


def main():
    # In-band peak (6000 K → peak 4830 Å): tight recovery.
    fit = estimate_planck_temperature(*_anchors(6000.0))
    assert fit["regime"] == "in-band", fit["regime"]
    assert abs(fit["t_best"] - 6000) / 6000 < 0.02, fit["t_best"]
    assert not fit["hi_open"] and not fit["lo_open"]

    # IR-side (3200 K → peak 9056 Å): Wien rise, still tight.
    fit = estimate_planck_temperature(*_anchors(3200.0))
    assert fit["regime"] == "ir-side", fit["regime"]
    assert abs(fit["t_best"] - 3200) / 3200 < 0.02, fit["t_best"]

    # UV-side (25000 K → peak 1159 Å) with 2% noise: the lower bound
    # must hold; the upper bound is expected to be soft/open.
    fit = estimate_planck_temperature(*_anchors(25000.0, noise=0.02))
    assert fit["regime"] == "uv-side", fit["regime"]
    assert fit["t_lo"] < 25000 < fit["t_hi"], (fit["t_lo"], fit["t_hi"])
    assert fit["t_lo"] > 10000, fit["t_lo"]   # still rules out A stars

    # Interval honesty: noisy in-band data yields a finite, non-empty
    # interval near the truth (Δχ²=1 ≈ 1σ, so exact bracketing is not
    # guaranteed — only closeness).
    fit = estimate_planck_temperature(*_anchors(6000.0, noise=0.03))
    assert fit["t_lo"] < fit["t_hi"], (fit["t_lo"], fit["t_hi"])
    assert abs(fit["t_best"] - 6000) / 6000 < 0.05, fit["t_best"]
    assert not fit["hi_open"] and not fit["lo_open"]

    # Leverage: a 15000 K star whose reddest anchor is depressed 4%
    # by a response-correction error.  The fit is biased hot; the
    # leave-one-out spread must flag the offending anchor's influence
    # and the widened interval must recover the true temperature.
    wl, fx = _anchors(15000.0)
    fx[-1] *= 0.96
    fit = estimate_planck_temperature(wl, fx)
    assert fit["loo_hi"] > fit["loo_lo"] * 1.02, (fit["loo_lo"], fit["loo_hi"])
    assert fit["t_lo"] <= 15000 * 1.001, fit["t_lo"]   # grid resolution
    assert not fit["super_rj"]

    # Unphysical blue tilt (the Segin symptom): continuum steeper than
    # the λ⁻⁴ Rayleigh-Jeans limit → no black body fits, flag it.
    wl, fx = _anchors(15000.0)
    fx *= (5500.0 / wl) ** 1.5
    fit = estimate_planck_temperature(wl, fx)
    assert fit["super_rj"], fit["loglog_slope"]
    assert fit["loglog_slope"] < -4.0

    # Clean in-band data: leave-one-out best fits stay tight.
    fit = estimate_planck_temperature(*_anchors(6000.0))
    assert fit["loo_hi"] / fit["loo_lo"] < 1.05, (fit["loo_lo"], fit["loo_hi"])

    # Degenerate input rejected.
    assert estimate_planck_temperature([5000, 6000], [1.0, 1.0]) is None
    assert estimate_planck_temperature(
        [4000, 5000, 6000], [1.0, -1.0, float("nan")]) is None

    print("test_planck_temperature: all checks passed")


if __name__ == "__main__":
    main()
