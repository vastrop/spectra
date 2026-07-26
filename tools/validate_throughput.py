#!/usr/bin/env python3
"""
validate_throughput.py
======================

Closes the exposure-estimator validation loop on a REAL exported spectrum:

    export from the viewer  ->  predict SNR  ->  measure SNR  ->  derive the
    true end-to-end throughput factor.

Reads a FITS spectrum written by spectrum_core.write_spectrum_fits (binary
table HDU 'SPECTRUM' with WAVELENGTH / FLUX / SIGMA columns and the
extraction metadata in the primary header), measures the achieved continuum
SNR in a line-free window, predicts what the idealised model expects for the
same target/configuration/integration, and reports the multiplicative
efficiency correction to feed back into estimate_exposure.py via
--efficiency.

Depends only on exposure_estimator.py (same folder), numpy, and astropy.

Usage
-----
    python validate_throughput.py spectrum.fits \
        --vmag 10.2 --window 5300 5600 \
        --aperture 115 --fratio 7 --total 3600 --sub 60 --sqm 20.0

The optical defaults below match the 115 mm f/7 refractor; override for the
configuration the spectrum was actually taken with.  SQM should reflect the
real sky at capture (e.g. moon present) -- the point of the loop is to absorb
real conditions into the throughput number.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
from astropy.io import fits

from exposure_estimator import (
    OpticalConfig, Throughput, CameraNoise, SkyConditions,
    predict_snr, measure_continuum_snr, calibrate_throughput,
)


def load_spectrum(path):
    """Return (wavelengths, flux, sigma, header_dict) from an exported FITS.
    Raises a clear error if the file is the known-corrupt structural form."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path) as hdul:
            if "SPECTRUM" not in [h.name for h in hdul]:
                raise ValueError(
                    f"{path!r} has no SPECTRUM extension; not an exported "
                    f"spectrum from this pipeline.")
            t = hdul["SPECTRUM"].data
            wl = np.asarray(t["WAVELENGTH"], dtype=float)
            fl = np.asarray(t["FLUX"], dtype=float)
            sg = np.asarray(t["SIGMA"], dtype=float)
            meta = dict(hdul[0].header)

    # Sanity guard against a corrupt write: a valid
    # spectrum has finite, physically-scaled values.  Astronomically absurd
    # magnitudes (1e+300) mean the binary payload was mis-aligned on write.
    finite = np.isfinite(wl) & np.isfinite(fl)
    if np.count_nonzero(finite) < 8:
        raise ValueError("Too few finite samples -- file likely corrupt.")
    if np.nanmax(np.abs(wl[finite])) > 1e6 or np.nanmax(np.abs(fl[finite])) > 1e30:
        raise ValueError(
            "Wavelength/flux values are out of physical range (>1e6 A or "
            ">1e30). The FITS binary payload is corrupt -- re-export with the "
            "hardened writer (the one that verifies on write).")
    return wl, fl, sg, meta


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validate predicted vs measured SNR on a real spectrum.")
    ap.add_argument("fits", help="Exported spectrum FITS file.")
    ap.add_argument("--vmag", type=float, required=True,
                    help="Target V magnitude at time of capture.")
    ap.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"),
                    required=True,
                    help="Line-free wavelength window (A) for the SNR "
                         "measurement, e.g. --window 5300 5600.")
    ap.add_argument("--aperture", type=float, default=115.0,
                    help="Aperture diameter mm (default 115).")
    ap.add_argument("--fratio", type=float, default=7.0,
                    help="Focal ratio f/D (default 7).")
    ap.add_argument("--obstruction", type=float, default=0.0,
                    help="Central obstruction areal fraction (default 0).")
    ap.add_argument("--dispersion", type=float, default=None,
                    help="A/px; default read from FITS header DISPERSI.")
    ap.add_argument("--spatial-fwhm", type=float, default=None,
                    help="Spatial FWHM px; default read from header FWHM.")
    ap.add_argument("--total", type=float, required=True,
                    help="Total integration time (s), e.g. 3600 for 60x60s.")
    ap.add_argument("--sub", type=float, default=60.0,
                    help="Sub-exposure length (s), default 60.")
    ap.add_argument("--sqm", type=float, default=21.0,
                    help="Sky brightness mag/arcsec^2 AT CAPTURE (default "
                         "21.0; lower if the moon was up).")
    ap.add_argument("--slope", type=float, default=0.0,
                    help="Continuum slope alpha; 0 flat is fine for a star.")
    ap.add_argument("--read-noise", type=float, default=1.5)
    ap.add_argument("--dark", type=float, default=0.01)
    args = ap.parse_args(argv)

    wl, fl, sg, meta = load_spectrum(args.fits)

    disp = args.dispersion
    if disp is None:
        disp = float(meta.get("DISPERSI", 7.6))
    sfwhm = args.spatial_fwhm
    if sfwhm is None:
        sfwhm = float(meta.get("FWHM", 10.0))

    # Reference wavelength = centre of the measurement window.
    ref_wl = 0.5 * (args.window[0] + args.window[1])

    cfg = OpticalConfig(
        name=f"{args.aperture:.0f}mm f/{args.fratio:.0f}",
        aperture_mm=args.aperture, focal_ratio=args.fratio,
        obstruction_frac=args.obstruction,
        dispersion_A_per_px=disp, spatial_fwhm_px=sfwhm)
    thru = Throughput()              # idealised; the correction is derived
    noise = CameraNoise(read_noise_e=args.read_noise, dark_e_per_s=args.dark)
    sky = SkyConditions(sqm_mag_arcsec2=args.sqm)

    # ── Measured continuum SNR in the line-free window ──
    meas = measure_continuum_snr(wl, fl, tuple(args.window), detrend=True)

    # ── Predicted continuum SNR for the same config / integration ──
    pred = predict_snr(ref_wl, args.vmag, cfg, thru, noise, sky,
                       args.total, args.sub,
                       bin_width_A=cfg.resolution_element_A(),
                       cont_slope=args.slope)

    print("=" * 60)
    print(" Throughput validation:", args.fits)
    print("=" * 60)
    print(f"Target V={args.vmag}, window {args.window[0]:.0f}-{args.window[1]:.0f} A "
          f"(ref {ref_wl:.0f} A)")
    print(f"Config: {cfg.name}, dispersion {disp:.2f} A/px, "
          f"spatial FWHM {sfwhm:.1f} px, res element {cfg.resolution_element_A():.0f} A")
    print(f"Integration {args.total:.0f}s ({args.total/args.sub:.0f} x "
          f"{args.sub:.0f}s), SQM {args.sqm}")
    print()
    print(f"MEASURED continuum SNR : {meas['snr']:.1f}  "
          f"(over {meas['n']} samples, detrended)")
    print(f"PREDICTED (idealised)  : {pred['snr']:.1f}")
    print()

    if meas["snr"] <= 0 or not np.isfinite(meas["snr"]):
        print("Measured SNR is unusable -- check the window is line-free and "
              "on real continuum.")
        return 1

    cal = calibrate_throughput(pred["snr"], meas["snr"], thru)
    print(f"SNR ratio (meas/pred)  : {cal['snr_ratio_meas_over_pred']:.3f}")
    print(f"Throughput factor      : {cal['throughput_factor_sky_limited']:.3f}")
    print()
    print(f"=> Use  --efficiency {cal['throughput_factor_sky_limited']:.2f}  "
          f"in estimate_exposure.py for this system/sky.")
    print()
    print("Note:", cal["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
