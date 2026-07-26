"""
exposure_estimator.py
======================

Theoretical exposure / SNR estimator for slitless (SA100/SA200) point-source
spectroscopy.  Predicts the continuum signal-to-noise ratio reached per
spectral bin for a given target, optical configuration, and total
integration, via the standard CCD equation.  Includes a companion helper to
measure the *achieved* continuum SNR in an already-reduced spectrum, so the
theoretical throughput can be calibrated against real data
(predict -> measure -> derive true end-to-end efficiency).

Design notes
------------
* Pure computation; no GUI, no I/O beyond the optional FITS-measurement
  helper.  Mirrors the spectrum_core separation of concerns.
* Every physical assumption is an explicit, overridable field on a dataclass
  with its source/justification in a comment, so the model can be challenged
  and recalibrated rather than trusted blindly.
* f/number enters in TWO independent places, deliberately separated so the
  f/4-vs-f/8 trade is visible:
    1. Collecting area depends ONLY on aperture diameter D (not f/D).  For a
       point source spread over a slitless spectrum, total star photons is
       set by D alone.
    2. Image quality / resolution depends on f/D: a faster system gives a
       smaller spot (fewer pixels, less sky per resolution element) BUT on a
       fast Newtonian spectral coma broadens the line-spread function.  This
       is captured by an explicit per-configuration spatial FWHM that the
       USER supplies (measured or estimated), NOT an invented coma model.
  The estimator therefore reports SNR *and* the resolution it is quoted at,
  so more-photons-but-worse-PSF (f/4) vs fewer-spread-but-cleaner-PSF
  (f/8 Barlowed) can be compared on equal footing.

References for defaults (IMX585, retrieved 2026):
  - 2.9 um pixels, full well ~47 ke-, peak QE ~91% visible, read noise ~1 e-
    (HDR), dark current ~0.005 e-/s/pix at 0C (cooled).  A cooled mono unit
    matches or beats these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import math

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_H_PLANCK = 6.62607015e-34       # J s
_C_LIGHT = 2.99792458e8          # m / s

# Zero-point photon flux of a V=0 source above the atmosphere, per unit
# wavelength, hitting a collecting area.  The standard Johnson-V flux density
# for V=0 is ~3.63e-9 erg/s/cm^2/Angstrom (Bessell); this yields the
# textbook ~1000 photons/s/cm^2/Angstrom at 5500 A.  This is the single most
# important calibratable constant; the validation loop folds any residual
# error here into the empirical throughput factor.
_V0_FLAM_ERG = 3.63e-9           # erg / s / cm^2 / Angstrom at V=0


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OpticalConfig:
    """One telescope + camera + grating configuration."""
    name: str
    aperture_mm: float            # clear aperture diameter D
    focal_ratio: float            # f/D (image scale / sky-per-pixel only)
    obstruction_frac: float = 0.0  # central obstruction (areal frac), e.g.
    #                                Newtonian secondary; reduces collecting
    #                                area. 0 for refractor.
    pixel_um: float = 2.9          # IMX585
    dispersion_A_per_px: float = 7.6  # measured on the 115 f/7; scales with
    #                                   grating-to-sensor distance.
    spatial_fwhm_px: float = 10.0  # cross-dispersion / LSF width in pixels at
    #                                best focus FOR THIS CONFIG.  This is where
    #                                f/4 spectral coma shows up: supply the
    #                                measured (or estimated) value per setup.
    #                                115 f/7 measured ~10 px (Delta-lambda~77 A).

    def collecting_area_cm2(self) -> float:
        """Effective light-collecting area in cm^2, accounting for any
        central obstruction.  Depends on D only -- NOT on f/D."""
        r_cm = (self.aperture_mm / 10.0) / 2.0
        geo = math.pi * r_cm * r_cm
        return geo * (1.0 - self.obstruction_frac)

    def plate_scale_arcsec_per_px(self) -> float:
        """Arcsec per pixel = 206265 * pixel_size / focal_length.
        focal_length = aperture * focal_ratio.  This DOES depend on f/D and
        sets sky background per pixel (the real f/4 advantage)."""
        focal_len_mm = self.aperture_mm * self.focal_ratio
        return 206265.0 * (self.pixel_um * 1e-3) / focal_len_mm

    def resolution_element_A(self) -> float:
        """Delta-lambda in Angstrom = spatial_fwhm_px * dispersion.
        The resolution the SNR is quoted per (and what coma degrades)."""
        return self.spatial_fwhm_px * self.dispersion_A_per_px


@dataclass
class Throughput:
    """End-to-end efficiency factors, wavelength-dependent where it matters.
    All overridable; the validation loop collapses them into one empirical
    number, so treat the product as the thing to calibrate, not each term."""
    optics: float = 0.80          # mirror/lens reflectivity*transmission;
    #                               refractor ~0.85, Newtonian 2-mirror ~0.80.
    grating_order1: float = 0.25  # SA100 first-order efficiency. Most light
    #                               goes to 0th order; ~20-30% reaches 1st.
    #                               Flat default; supply a curve if you have it.
    filter_trans: float = 1.0     # any blocking/order-sorting filter; 1.0 none
    # QE is supplied separately (wavelength-dependent), see qe_at().

    # IMX585 QE: rough piecewise model, peak ~0.91 in the visible, falling to
    # the blue and into the red/NIR.  Tunable anchor points (wl_A: QE).
    qe_curve: Dict[float, float] = field(default_factory=lambda: {
        3500.0: 0.30, 4000.0: 0.55, 4500.0: 0.72, 5000.0: 0.85,
        5500.0: 0.91, 6000.0: 0.90, 6500.0: 0.85, 7000.0: 0.78,
        8000.0: 0.60, 9000.0: 0.40, 10000.0: 0.18,
    })

    def qe_at(self, wl_A: float) -> float:
        ks = sorted(self.qe_curve)
        if wl_A <= ks[0]:
            return self.qe_curve[ks[0]]
        if wl_A >= ks[-1]:
            return self.qe_curve[ks[-1]]
        return float(np.interp(wl_A, ks, [self.qe_curve[k] for k in ks]))

    def total_at(self, wl_A: float) -> float:
        return (self.optics * self.grating_order1 * self.filter_trans
                * self.qe_at(wl_A))


@dataclass
class CameraNoise:
    """Detector noise terms (IMX585 cooled mono defaults)."""
    read_noise_e: float = 1.5     # HDR ~1 e-; 1.5 conservative for unity/HCG
    dark_e_per_s: float = 0.01    # cooled; ~0.005 at 0C, 0.01 conservative


@dataclass
class SkyConditions:
    """Sky background. SQM (mag/arcsec^2) is the practical input; converted
    to a V-band surface brightness.  Theoretical-clean assumption: dark site,
    no moon.  Real data with moon/low elevation will be far worse -- which is
    exactly why the validation targets should be good-sky frames."""
    sqm_mag_arcsec2: float = 21.6  # the user's dark site


# ---------------------------------------------------------------------------
# Core photon-rate computation
# ---------------------------------------------------------------------------

def _v0_photons_per_s_per_cm2_per_A(wl_A: float) -> float:
    """Photons/s/cm^2/Angstrom from a V=0 source, at wavelength wl_A.
    Photon energy E = h c / lambda.  Flux density is taken flat at the V
    zero-point value as a first approximation across the optical (the target
    spectral shape is applied separately)."""
    wl_m = wl_A * 1e-10
    e_phot_J = _H_PLANCK * _C_LIGHT / wl_m
    flam_erg = _V0_FLAM_ERG                  # erg/s/cm^2/A
    flam_J = flam_erg * 1e-7                  # -> J/s/cm^2/A
    return flam_J / e_phot_J                  # photons/s/cm^2/A


def star_electrons_per_s_in_bin(wl_A: float, v_mag: float,
                                cfg: OpticalConfig, thru: Throughput,
                                bin_width_A: float,
                                cont_slope: float = 0.0,
                                ref_wl_A: float = 5500.0) -> float:
    """Star electrons/s collected into one spectral bin of width bin_width_A
    centred at wl_A.

    cont_slope : power-law continuum slope alpha for F_lambda ~ lambda^alpha,
                 normalised at ref_wl_A so v_mag stays the V-band anchor.
                 0.0 = flat in F_lambda.  Quasar optical continua are roughly
                 alpha ~ -1 to -1.5 in F_lambda; supply if you want the blue
                 boost / red droop reflected.
    """
    phot0 = _v0_photons_per_s_per_cm2_per_A(wl_A)
    flux_scale = 10.0 ** (-0.4 * v_mag)
    shape = (wl_A / ref_wl_A) ** cont_slope
    photons_per_s_per_A = phot0 * flux_scale * shape * cfg.collecting_area_cm2()
    e_per_s_per_A = photons_per_s_per_A * thru.total_at(wl_A)
    return e_per_s_per_A * bin_width_A


def sky_electrons_per_s_per_px(wl_A: float, cfg: OpticalConfig,
                               thru: Throughput, sky: SkyConditions) -> float:
    """Sky electrons/s in ONE pixel.  Sky surface brightness (mag/arcsec^2)
    over the solid angle subtended by a pixel gives a V=mag-equivalent point
    flux per pixel, dispersed across the spectrum like the star.  The sky
    flux is spread over the dispersion the same way (per Angstrom *
    dispersion per pixel)."""
    scale = cfg.plate_scale_arcsec_per_px()
    px_area_arcsec2 = scale * scale
    # Effective V mag of the sky seen by one pixel:
    sky_mag_in_px = sky.sqm_mag_arcsec2 - 2.5 * math.log10(px_area_arcsec2)
    phot0 = _v0_photons_per_s_per_cm2_per_A(wl_A)
    flux_scale = 10.0 ** (-0.4 * sky_mag_in_px)
    photons_per_s_per_A = phot0 * flux_scale * cfg.collecting_area_cm2()
    e_per_s_per_A = photons_per_s_per_A * thru.total_at(wl_A)
    # One pixel spans dispersion_A_per_px Angstrom of the dispersed sky.
    return e_per_s_per_A * cfg.dispersion_A_per_px


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------

def predict_snr(wl_A: float, v_mag: float, cfg: OpticalConfig,
                thru: Throughput, noise: CameraNoise, sky: SkyConditions,
                total_integration_s: float, sub_exposure_s: float,
                bin_width_A: Optional[float] = None,
                extraction_height_px: Optional[float] = None,
                cont_slope: float = 0.0) -> Dict[str, float]:
    """Predict continuum SNR per bin via the CCD equation.

    SNR = S t / sqrt( S t + npix (B t + D t) + npix N_sub Rn^2 )

    where S, B in e-/s, t total integration, npix pixels summed per bin,
    N_sub = number of sub-exposures (read noise does NOT average away).

    bin_width_A defaults to one resolution element (spatial_fwhm * dispersion)
    -- the natural unit for a broad emission line.  extraction_height_px
    defaults to ~2x the spatial FWHM (a sensible aperture).
    """
    if bin_width_A is None:
        bin_width_A = cfg.resolution_element_A()
    if extraction_height_px is None:
        extraction_height_px = 2.0 * cfg.spatial_fwhm_px

    bin_width_px = bin_width_A / cfg.dispersion_A_per_px
    npix = max(1.0, bin_width_px) * max(1.0, extraction_height_px)
    n_sub = max(1.0, total_integration_s / max(1e-6, sub_exposure_s))

    S = star_electrons_per_s_in_bin(wl_A, v_mag, cfg, thru, bin_width_A,
                                    cont_slope=cont_slope)
    B = sky_electrons_per_s_per_px(wl_A, cfg, thru, sky)
    D = noise.dark_e_per_s

    t = total_integration_s
    signal = S * t
    var = (signal
           + npix * (B * t + D * t)
           + npix * n_sub * noise.read_noise_e ** 2)
    snr = signal / math.sqrt(var) if var > 0 else 0.0

    return {
        "snr": snr,
        "star_e_per_s_bin": S,
        "sky_e_per_s_px": B,
        "npix_per_bin": npix,
        "n_sub": n_sub,
        "signal_e": signal,
        "noise_e": math.sqrt(var),
        "bin_width_A": bin_width_A,
        "resolution_element_A": cfg.resolution_element_A(),
        "plate_scale_as_px": cfg.plate_scale_arcsec_per_px(),
        "collecting_area_cm2": cfg.collecting_area_cm2(),
    }


def integration_for_target_snr(target_snr: float, wl_A: float, v_mag: float,
                               cfg: OpticalConfig, thru: Throughput,
                               noise: CameraNoise, sky: SkyConditions,
                               sub_exposure_s: float,
                               bin_width_A: Optional[float] = None,
                               extraction_height_px: Optional[float] = None,
                               cont_slope: float = 0.0,
                               t_max_s: float = 360000.0) -> float:
    """Total integration (seconds) needed to reach target_snr at wl_A.
    Solved by bisection on the monotonic SNR(t).  Returns inf if t_max is
    insufficient."""
    def f(t):
        return predict_snr(wl_A, v_mag, cfg, thru, noise, sky, t,
                           sub_exposure_s, bin_width_A, extraction_height_px,
                           cont_slope)["snr"] - target_snr
    lo, hi = 1.0, t_max_s
    if f(hi) < 0:
        return float("inf")
    if f(lo) >= 0:
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# Validation helper: measured SNR from a reduced spectrum
# ---------------------------------------------------------------------------

def measure_continuum_snr(wls_A: np.ndarray, flux: np.ndarray,
                          window_A: tuple,
                          detrend: bool = True) -> Dict[str, float]:
    """Measure achieved continuum SNR in a line-free window of an already-
    reduced 1-D spectrum, for the predict-vs-measured calibration loop.

    SNR is mean(flux)/std(flux) over the window, optionally after removing a
    linear continuum slope (recommended -- a sloped continuum inflates the
    raw std and understates the true SNR).  Choose a window with no emission
    or absorption features and away from telluric bands.

    Returns dict with snr, mean, std, n.
    """
    wls_A = np.asarray(wls_A, dtype=float)
    flux = np.asarray(flux, dtype=float)
    lo, hi = window_A
    sel = np.isfinite(wls_A) & np.isfinite(flux) & (wls_A >= lo) & (wls_A <= hi)
    if np.count_nonzero(sel) < 8:
        return {"snr": float("nan"), "mean": float("nan"),
                "std": float("nan"), "n": int(np.count_nonzero(sel))}
    w = wls_A[sel]
    y = flux[sel]
    mean = float(np.mean(y))
    if detrend and w.size >= 3:
        a, b = np.polyfit(w, y, 1)
        resid = y - (a * w + b)
        std = float(np.std(resid, ddof=1))
    else:
        std = float(np.std(y, ddof=1))
    snr = mean / std if std > 0 else float("nan")
    return {"snr": snr, "mean": mean, "std": std, "n": int(y.size)}


def calibrate_throughput(predicted_snr: float, measured_snr: float,
                         thru: Throughput) -> Dict[str, float]:
    """Given a predicted and a measured continuum SNR for the SAME frame,
    derive the empirical correction to the assumed throughput.

    SNR scales as sqrt(throughput) in the sky/shot-limited regime (signal and
    sky both scale linearly with throughput, so SNR ~ sqrt(thru) when sky- or
    dark-limited; ~linear-ish only in the read-noise floor).  The
    sky-limited estimate is reported as the primary correction, flagged as
    a lower bound when the frame was read-noise limited.
    """
    if measured_snr <= 0 or predicted_snr <= 0:
        return {"throughput_factor": float("nan")}
    ratio = measured_snr / predicted_snr
    factor_sky_limited = ratio ** 2      # SNR ~ sqrt(thru)
    return {
        "snr_ratio_meas_over_pred": ratio,
        "throughput_factor_sky_limited": factor_sky_limited,
        "suggested_total_throughput": (thru.optics * thru.grating_order1
                                       * thru.filter_trans) * factor_sky_limited,
        "note": ("Multiply assumed (optics*grating*filter) by "
                 "throughput_factor_sky_limited. Valid if sky/dark-limited; "
                 "if read-noise-limited the true factor is closer to "
                 "ratio**1, so this is a lower bound on efficiency."),
    }


# ---------------------------------------------------------------------------
# Convenience: standard configurations for this project
# ---------------------------------------------------------------------------

def standard_configs() -> List[OpticalConfig]:
    """The four configurations under consideration.  spatial_fwhm_px encodes
    the image-quality trade: f/4 inflated for spectral coma, f/8 (Barlowed)
    restored, refractor at its measured ~10 px.  THESE ARE ESTIMATES -- the
    whole point is to replace them with measured values per setup."""
    return [
        OpticalConfig("115mm f/7 refractor", 115, 7.0,
                      obstruction_frac=0.0, spatial_fwhm_px=10.0),
        OpticalConfig("235mm f/10 SCT", 235, 10.0,
                      obstruction_frac=0.13, spatial_fwhm_px=9.0),
        OpticalConfig("305mm f/4 Newtonian", 305, 4.0,
                      obstruction_frac=0.20, spatial_fwhm_px=14.0),  # coma!
        OpticalConfig("305mm f/8 Newtonian+Barlow", 305, 8.0,
                      obstruction_frac=0.20, spatial_fwhm_px=9.0),   # tamed
    ]
