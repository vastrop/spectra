"""
Self-check for the response-calibration round trip.

Guards the orientation the response dialog's "Observed / response" sanity
line depends on: the curve is observed/reference, so dividing the observed
star by it must give the reference back.  A flipped ratio anywhere in the
chain would still plot a plausible-looking curve — this catches it.

Run: py -3.13 tests/test_response_roundtrip.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectrum_core import (          # noqa: E402
    compute_instrument_response_raw,
    apply_calibration,
    load_calibration_file,
)


def _synthetic():
    """An A0V-ish reference and an observation of it seen through a
    smooth, strongly wavelength-dependent instrument response."""
    wls = np.linspace(4000.0, 7000.0, 600)
    ref = 1.0 + 0.5 * np.exp(-((wls - 5200.0) / 600.0) ** 2)
    # Throughput peaking mid-band and falling off at both ends.
    resp = 0.2 + 0.8 * np.exp(-((wls - 5500.0) / 1200.0) ** 2)
    return wls, ref, ref * resp, resp


def test_curve_is_obs_over_ref():
    wls, ref, obs, resp = _synthetic()
    _w, curve, _diag = compute_instrument_response_raw(
        wls, obs, wls, ref, balmer_half_width_A=0.0)
    good = np.isfinite(curve)
    assert good.sum() > 500, "curve should cover essentially the whole band"
    # The recovered curve is the throughput, up to no scaling at all here.
    assert np.allclose(curve[good], resp[good], rtol=1e-6), \
        "response is not observed/reference — the sanity line would be wrong"


def test_dividing_observed_by_curve_recovers_reference():
    wls, ref, obs, _resp = _synthetic()
    _w, curve, _diag = compute_instrument_response_raw(
        wls, obs, wls, ref, balmer_half_width_A=0.0)

    # What the dialog's green line plots (plain division on a shared grid).
    corrected = obs / curve
    good = np.isfinite(corrected)
    assert np.allclose(corrected[good], ref[good], rtol=1e-6), \
        "observed / response must land on the reference"

    # And what the pipeline does, via the interpolating path — same answer,
    # which is why the green line honestly previews a Run.
    cal_df = pd.DataFrame({"wavelength": wls, "factor": curve})
    piped = apply_calibration(wls, obs, cal_df)
    both = np.isfinite(piped) & good
    assert both.sum() > 500
    assert np.allclose(piped[both], corrected[both], rtol=1e-6), \
        "apply_calibration disagrees with the dialog's preview"


def test_calibration_file_validation():
    """load_calibration_file must reject tables np.interp would silently
    mis-handle: duplicate wavelengths, too few rows; and it must drop
    non-finite rows like it already drops NaNs."""
    def load(text):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".dat", delete=False) as f:
            f.write(text)
        try:
            return load_calibration_file(f.name)
        finally:
            os.unlink(f.name)

    # Good file: unsorted + an inf row — loads sorted, inf dropped.
    df = load("5000 1.0\n4000 0.5\n6000 inf\n7000 2.0\n")
    assert list(df["wavelength"]) == [4000.0, 5000.0, 7000.0]

    for bad, why in [
        ("5000 1.0\n5000 1.2\n6000 2.0\n", "duplicate wavelength"),
        ("5000 1.0\n", "single row"),
        ("5000 inf\n6000 inf\n", "all rows non-finite"),
    ]:
        try:
            load(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} table was accepted")


if __name__ == "__main__":
    test_curve_is_obs_over_ref()
    test_dividing_observed_by_curve_recovers_reference()
    test_calibration_file_validation()
    print("response round-trip self-check OK")
