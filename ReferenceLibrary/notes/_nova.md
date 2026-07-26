# Nova — classical nova (no MK type)

## Class: eruptive variables, not a spectral class
A nova is not a temperature class and has no MK type: it is a thermonuclear runaway on the surface of a white dwarf that has accreted hydrogen from a close companion. What you record is an *event caught at an epoch*, not a star of a type, and the spectrum changes beyond recognition over days to months. Two consequences for this library: there is no Pickles reference spectrum to compare against, and the same object will match different rows of the table below depending on when you observed it.

The one constant is emission. From a few days after maximum the spectrum is dominated by broad emission lines — Hα above all, with equivalent widths of hundreds of Å. In a slitless frame this is unmistakable: the zero-order star looks ordinary while the spectrum carries a bright, isolated blob at Hα, far brighter than any absorption-line star's continuum there. That alone is enough to flag a nova candidate.

Lines are Doppler-broadened by the expanding ejecta, so **read the widths as velocities**. At Hα, 1000 km/s ≈ 22 Å — the reason these lines are easy targets at slitless resolution rather than in spite of it.

## This type: which nova, and at what epoch
**Phase** (the axis that matters most):

- *Fireball / pre-maximum*, hours to days: blue, nearly featureless continuum with P Cygni profiles — a blueshifted absorption trough on the blue wing of each emission line, the signature of an optically thick expanding shell seen against itself.
- *Early decline*, days to weeks: P Cygni absorption fades, leaving pure emission. Balmer dominates; the Fe II or He/N character (below) is legible here. This is the phase most amateur spectra catch.
- *Nebular*, weeks to months: the ejecta thin out and forbidden lines appear — [O III] 4959/5007, then [N II] 5755. Balmer weakens *relative* to these. A nova showing [O III] as strong as Hβ is well past maximum.
- *Coronal*, in some novae: [Fe X] 6375 and [Fe XIV] 5303, from gas ionized far beyond anything in a normal stellar photosphere.

**Spectroscopic class**, read during early decline:

- *Fe II novae* — the more common, slower kind. Ejection velocities ~1000-2500 km/s, so lines are relatively narrow (Hα FWHM ~20-60 Å). Fe II 4924/5018/5169 in emission, often with P Cygni profiles, is the marker. These evolve slowly and stay observable for weeks.
- *He/N novae* — fast, energetic. Velocities ~2500-5000 km/s give visibly broader, flat-topped or castellated lines (Hα FWHM ~60-110 Å). He I 5876 and the N III 4640 blend stand out; Fe II is weak or absent. These fade fast, so an early spectrum matters.

Do not confuse a classical nova with a **dwarf nova** (a disc-instability outburst in a cataclysmic variable, no thermonuclear event): far fainter, and its Balmer emission is narrow and often double-peaked from the rotating accretion disc, not the thousands of km/s of an expanding shell.

## Features
Rest wavelengths. In the ejecta these lines are broadened by 1000-5000 km/s and, before the P Cygni absorption fades, their troughs sit blueward of the values below.

| Wavelength (Å) | Feature | Notes |
|---|---|---|
| 4340 | Hγ | emission; P Cygni profile before/near maximum |
| 4640-4650 | N III / C III blend | He/N novae; weak or absent in Fe II novae |
| 4686 | He II | high-ionization; strengthens as the ejecta thin |
| 4861 | Hβ | strong emission; compare against [O III] to place the epoch |
| 4924 | Fe II | Fe II-class marker (multiplet 42), often P Cygni |
| 4959 | [O III] | nebular phase only — forbidden, so absent while the shell is dense |
| 5007 | [O III] | nebular phase; eventually rivals or beats Hβ |
| 5018 | Fe II | Fe II-class marker |
| 5169 | Fe II | Fe II-class marker |
| 5303 | [Fe XIV] | coronal phase, in some novae only |
| 5755 | [N II] | nebular phase |
| 5876 | He I | prominent in He/N novae |
| 6375 | [Fe X] | coronal phase, in some novae only |
| 6563 | Hα | the dominant feature; FWHM ~500-5000 km/s (11-110 Å) |
