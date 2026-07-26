# Spectrum Explorer — Getting Started

Spectrum Explorer turns a FITS image containing a stellar spectrum into a
wavelength-calibrated, instrument-response-corrected spectrum. For a new
equipment setup, begin by creating a reusable configuration from a bright
A-type star.

## First-time setup

1. **Capture a bright A1–A3 star.** Choose a bright, well-exposed A-type star.
   Its strong Balmer absorption lines make it suitable for wavelength and
   instrument-response calibration.

2. **Load the FITS image.** Start Spectrum Explorer and select the calibration
   star's FITS file. The original image appears in the main plot.

3. **Derotate and extract.** Press **Auto** to estimate the rotation angle,
   then **Update**. Confirm that the extraction aperture is centred on the
   spectral trace and that its two background bands avoid nearby stars.

4. **Enter an approximate dispersion.** Enter an approximate value in
   **Å/pixel**. It need not be exact yet; it only needs to make the extraction
   long enough to contain the useful spectrum.

5. **Calibrate the dispersion.** Open **Calibrate dispersion** and try the
   automatic calibration. The Balmer lines of a suitable A1–A3 spectrum should
   provide a reasonable initial wavelength solution. Check the proposed line
   positions and correct or add nodes manually when necessary.

6. **Calibrate the instrument response.** Open **Calibrate instrument
   response**, select a matching A-type reference spectrum from the supplied
   library, and use the automatic response calculation as a starting point.
   Inspect the corrected spectrum for obvious distortions.

7. **Save the configuration.** Save the completed analysis configuration. It
   records the rotation, dispersion solution, and instrument-response curve
   for this optical setup.

## Analysing subsequent targets

Load the target FITS and the saved configuration, then press **Update**.
Detected sources are numbered on the image; click a numbered source to inspect
it, or click an arbitrary position if the desired source was not detected.

Check the aperture and background bands before trusting the result. Centre the
aperture on the trace, adjust its height or background bands if needed, and
enable contaminant masking when another star crosses the spectrum.

Open **Full spectrum** for a larger calibrated view. From there, export the
spectrum as FITS or save the displayed figure as PNG. Continuum calibration is
optional and normally follows wavelength and instrument-response calibration.

Recalibration is normally required only when the camera, grating, focus,
orientation, or optical arrangement changes.

## Optional tools

- **Solve to WCS** identifies detected sources and enables catalogue and
  database features.
- **Reference Library** provides standard spectra for comparison.
- **Predictor** offers an experimental template match.
- **Sequence Generator** displays spectral changes across a folder of FITS
  frames.
- **Livestack** watches a capture folder and improves the working image as new
  frames arrive.
