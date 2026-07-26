#!/usr/bin/env python3
"""
estimate_exposure.py
====================

Standalone interactive helper for exposure_estimator.py.  Asks for a target
and observing conditions, then reports the predicted continuum SNR per
resolution element across the configurations under consideration and the
integration time needed to reach a chosen SNR.

Run interactively:
    python estimate_exposure.py

Or non-interactively with flags (skips prompts you supply):
    python estimate_exposure.py --vmag 14.6 --wl 6541 --slope -1.0 \
        --sqm 21.6 --sub 300 --target-snr 5

Depends only on exposure_estimator.py (same folder) and numpy.
"""

from __future__ import annotations

import argparse
import sys

from exposure_estimator import (
    standard_configs, Throughput, CameraNoise, SkyConditions,
    predict_snr, integration_for_target_snr,
)


# ---------------------------------------------------------------------------
# Prompt helpers (used only for values not supplied on the command line)
# ---------------------------------------------------------------------------

def _ask_float(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"  (not a number — using {default})")
        return float(default)


def _fmt_time(seconds):
    if seconds == float("inf"):
        return ">100 h"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds/60:.0f} min"
    return f"{seconds/3600:.1f} h"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Interactive slitless-spectroscopy exposure estimator.")
    ap.add_argument("--vmag", type=float, default=None,
                    help="Target V magnitude.")
    ap.add_argument("--wl", type=float, default=None,
                    help="Wavelength of interest in Angstrom "
                         "(e.g. a redshifted emission line).")
    ap.add_argument("--slope", type=float, default=None,
                    help="Continuum power-law slope alpha (F_lambda ~ "
                         "lambda^alpha). Quasars ~ -1 to -1.5; 0 = flat.")
    ap.add_argument("--sqm", type=float, default=None,
                    help="Sky brightness in mag/arcsec^2 (SQM reading).")
    ap.add_argument("--sub", type=float, default=None,
                    help="Sub-exposure length in seconds.")
    ap.add_argument("--target-snr", type=float, default=None,
                    help="Continuum SNR per resolution element to solve for. "
                         "Default 20 (a useful spectrum, not a bare detection).")
    ap.add_argument("--line-ew", type=float, default=None,
                    help="Observed-frame equivalent width (A) of the emission "
                         "line of interest. If given, the helper reports line "
                         "DETECTION significance, the number that actually "
                         "matters, instead of bare continuum SNR.")
    ap.add_argument("--read-noise", type=float, default=1.5,
                    help="Camera read noise e- (default 1.5, IMX585 HDR ~1).")
    ap.add_argument("--dark", type=float, default=0.01,
                    help="Dark current e-/s/px (default 0.01, cooled).")
    ap.add_argument("--efficiency", type=float, default=0.15,
                    help="Catch-all real-world derating factor (0-1) folding "
                         "in atmospheric extinction, scattered light, "
                         "imperfect sky subtraction and flat-fielding, and "
                         "true vs assumed grating throughput. Default 0.15 — "
                         "the idealised model is ~5-10x optimistic. SET THIS "
                         "from calibrate_throughput() once you have a real "
                         "spectrum; use 1.0 to see the raw idealised numbers.")
    args = ap.parse_args(argv)

    interactive = sys.stdin.isatty()

    print("=" * 64)
    print(" Slitless spectroscopy exposure estimator")
    print("=" * 64)

    # Gather inputs: command-line value wins; otherwise prompt (interactive)
    # or fall back to a sensible default (non-interactive).
    def get(val, prompt, default):
        if val is not None:
            return val
        if interactive:
            return _ask_float(prompt, default)
        return float(default)

    vmag = get(args.vmag, "Target V magnitude", 14.6)
    wl = get(args.wl, "Wavelength of interest (A)", 6541.0)
    slope = get(args.slope, "Continuum slope alpha (0=flat, QSO ~ -1)", -1.0)
    sqm = get(args.sqm, "Sky brightness SQM (mag/arcsec^2)", 21.6)
    sub = get(args.sub, "Sub-exposure length (s)", 300.0)
    target_snr = get(args.target_snr, "Target continuum SNR/res.element", 20.0)
    # Line EW is optional; 0 / blank means "continuum only, no line framing".
    if args.line_ew is not None:
        line_ew = args.line_ew
    elif interactive:
        line_ew = _ask_float("Line equivalent width (A, 0=skip line framing)",
                             0.0)
    else:
        line_ew = 0.0

    thru = Throughput()
    # Fold the catch-all real-world derating into the throughput. filter_trans
    # is a free multiplier in the model, so this scales total throughput
    # without touching the library. SNR ~ sqrt(throughput) in the sky/shot
    # regime, so efficiency 0.15 lowers SNR by ~2.6x; in the read-noise
    # regime the effect is closer to linear.
    thru.filter_trans = max(0.0, min(1.0, args.efficiency))
    noise = CameraNoise(read_noise_e=args.read_noise, dark_e_per_s=args.dark)
    sky = SkyConditions(sqm_mag_arcsec2=sqm)

    print()
    print(f"Target: V={vmag:.1f}, line/continuum at {wl:.0f} A, "
          f"slope {slope:+.1f}")
    print(f"Sky SQM {sqm:.1f}, sub-exposures {sub:.0f} s, "
          f"target SNR {target_snr:.0f}")
    print(f"Camera: read noise {noise.read_noise_e:.1f} e-, "
          f"dark {noise.dark_e_per_s:.3f} e-/s/px")
    print(f"Real-world efficiency factor: {thru.filter_trans:.2f} "
          f"{'(idealised — set from calibration!)' if thru.filter_trans >= 1.0 else ''}")
    print()
    print("SNR is per resolution element (continuum). A broad emission line")
    print("with observed EW ~ tens-to-hundreds of A rises ON TOP of this.")
    print()

    # ── Per-configuration table ──
    header = (f"{'Configuration':<28}{'Res':>6}{'SNR':>7}{'SNR':>7}"
              f"{'SNR':>7}{'t to':>9}")
    sub2 = (f"{'':<28}{'(A)':>6}{'/1h':>7}{'/4h':>7}{'/10h':>7}"
            f"{'SNR'+str(int(target_snr)):>9}")
    print(header)
    print(sub2)
    print("-" * 64)

    best = None
    for cfg in standard_configs():
        r1 = predict_snr(wl, vmag, cfg, thru, noise, sky, 3600, sub,
                         cont_slope=slope)
        r4 = predict_snr(wl, vmag, cfg, thru, noise, sky, 14400, sub,
                         cont_slope=slope)
        r10 = predict_snr(wl, vmag, cfg, thru, noise, sky, 36000, sub,
                          cont_slope=slope)
        t = integration_for_target_snr(target_snr, wl, vmag, cfg, thru,
                                       noise, sky, sub, cont_slope=slope)
        print(f"{cfg.name:<28}{r1['resolution_element_A']:>6.0f}"
              f"{r1['snr']:>7.1f}{r4['snr']:>7.1f}{r10['snr']:>7.1f}"
              f"{_fmt_time(t):>9}")
        # Track the config reaching target SNR fastest as the recommendation.
        if t != float("inf") and (best is None or t < best[1]):
            best = (cfg, t, r1)

    print("-" * 64)

    # ── Recommendation ──
    if best is None:
        print("\nNo configuration reaches the target SNR within 100 h.")
        print("Consider: brighter target, larger aperture, or lower target SNR.")
        return 0

    cfg, t, r = best
    n_subs = max(1, round(t / sub))
    print(f"\nRECOMMENDATION (fastest to continuum SNR {target_snr:.0f} "
          f"at {wl:.0f} A):")
    print(f"  {cfg.name}")
    print(f"  Total integration : {_fmt_time(t)}  "
          f"(~{n_subs} x {sub:.0f}s sub-exposures)")
    print(f"  Resolution element: {r['resolution_element_A']:.0f} A "
          f"({cfg.spatial_fwhm_px:.0f} px spatial FWHM)")
    print(f"  Plate scale       : {r['plate_scale_as_px']:.2f} arcsec/px")

    # ── Line detection framing (the number that actually matters) ──
    # A line of observed EW sits on the continuum. Detecting it means its
    # integrated excess beats the continuum noise. Over a resolution element,
    # the line's contrast is ~ EW / Delta-lambda_res; the detection
    # significance is roughly that contrast times the continuum SNR. This is
    # an approximation (assumes the line is comparable to a resolution
    # element and the continuum is well determined either side), good enough
    # to answer "is this worth attempting".
    if line_ew > 0:
        res = r["resolution_element_A"]
        print(f"\n  LINE DETECTION (observed EW {line_ew:.0f} A):")
        for hours in (1, 4, 10):
            rr = predict_snr(wl, vmag, cfg, thru, noise, sky, hours * 3600,
                             sub, cont_slope=slope)
            contrast = line_ew / res
            sig = rr["snr"] * contrast
            verdict = ("confident" if sig >= 5 else
                       "marginal" if sig >= 3 else "not detectable")
            print(f"    {hours:>2}h : continuum SNR {rr['snr']:>5.0f}  ->  "
                  f"line significance ~{sig:>4.1f} sigma  ({verdict})")
        print("    (>=5 sigma confident, 3-5 marginal, <3 not detectable)")

    # Read-noise vs shot-noise regime note (drives sub-exposure choice).
    import math
    S = r["star_e_per_s_bin"]
    shot_1h = math.sqrt(S * 3600)
    rn_1h = math.sqrt(r["npix_per_bin"] * (3600 / sub) * noise.read_noise_e**2)
    if rn_1h > shot_1h:
        print(f"  NOTE: read-noise limited (read {rn_1h:.0f} e- > "
              f"shot {shot_1h:.0f} e- at 1h). Longer sub-exposures help — "
              f"try increasing --sub.")
    else:
        print(f"  NOTE: shot/sky limited (shot {shot_1h:.0f} e- >= "
              f"read {rn_1h:.0f} e- at 1h). Sub-exposure length matters less; "
              f"keep subs short enough to avoid saturation.")

    print()
    print("Caveat: idealised model (no extinction, perfect extraction/sky).")
    print("Treat absolute SNR as an upper bound; the ratios between configs")
    print("are the reliable part. Calibrate against a real spectrum via")
    print("exposure_estimator.measure_continuum_snr / calibrate_throughput.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
