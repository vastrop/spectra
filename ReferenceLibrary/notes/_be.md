# Be star — emission-line B star (a phenomenon, not a class)

## Class: a disc, not a temperature
The `e` in "B2Ve" is not a temperature class and does not move the star along the spectral sequence. A classical Be star is an ordinary, rapidly rotating B star (luminosity class V to III — **not** a supergiant) that has thrown off an equatorial **decretion disc**: gas spun out of the star, orbiting it in a flattened Keplerian ring. The photosphere underneath is a perfectly normal B star, and its He I lines, Balmer wings and continuum still classify it. The disc simply adds emission on top.

So a Be star has *two* spectra superimposed. The card for its underlying MK type (the He I/Mg II ratio, the Balmer widths, the luminosity criteria) still applies. What no card can give you is the emission — and there is no Pickles Be reference in this library, because there is no such thing as a standard Be spectrum. **Do not use a Be star as a response-calibration reference**: its Hα is in emission, and dividing by it would carve a spurious hole in your instrument response.

**The emission is transient.** A disc builds up over weeks to months, can persist for years, and can dissipate entirely — at which point the star looks like an ordinary B star until it does it again. The same object genuinely looks different from one observing season to the next, which is why Be stars are worth monitoring rather than merely observing, and why amateur spectra of them are scientifically useful (the [BeSS database](http://basebe.obspm.fr/) exists to collect exactly this).

## This type: what to expect, and what to rule out
**Hα is the feature.** At slitless resolution it is unmistakable: a strong emission peak where every other B star shows an absorption trough. Equivalent widths run from a few Å to −40 Å or more in an active disc. The **profile encodes the viewing angle**:

- *Pole-on* (disc face-on): a single, fairly narrow emission peak.
- *Intermediate inclination*: the classic **double-peaked** profile — two emission humps split by a central absorption dip, the signature of a rotating disc, one arm approaching and one receding. The peak separation measures the disc's projected rotation.
- *Edge-on*: a **Be-shell star** — the disc is seen against the photosphere, so a sharp, deep absorption core is cut into the emission, and shell absorption lines appear (Fe II, Ca II).

Hβ may be filled in or weakly in emission when the disc is strong; the higher Balmer lines usually stay in absorption. He I lines stay photospheric absorption throughout — they belong to the star, not the disc. Fe II emission appears in the strongest cases.

**Rule out the look-alikes.** Emission at Hα is not by itself a Be diagnosis:

- **P Cygni / LBV** (e.g. P Cyg itself, `B1-2Ia-0ep`): a *supergiant* with a dense radiatively driven wind. The profile is a P Cygni — emission with a **blueshifted absorption trough on its blue wing** — not a symmetric double peak. The wind is expanding outward; a Be disc is rotating. The luminosity class is the tell: Be stars are not supergiants.
- **Herbig Ae/Be**: a *young* star still accreting, with a circumstellar disc and a strong infrared excess. Emission looks similar; the object is pre-main-sequence.
- **Nova / symbiotic**: far stronger, broader emission on a hot continuum, with forbidden lines — see the nova card.
- **Interacting binary / mass transfer**: emission from an accretion stream rather than a decretion disc.

γ Cas is the prototype of the whole class, and its emission was the first ever seen in a stellar spectrum (Secchi, 1866).

## Features
The underlying photosphere is a normal B star — use its MK card for the absorption spectrum. The rows below are what the *disc* adds.

| Wavelength (Å) | Feature | Notes |
|---|---|---|
| 4026 | He I | photospheric absorption — belongs to the star, not the disc |
| 4102 | Hδ | absorption; rarely affected |
| 4340 | Hγ | absorption, occasionally partly filled in |
| 4471 | He I | photospheric absorption; classifies the underlying B subtype |
| 4861 | Hβ | absorption normally; filled in or in emission when the disc is strong |
| 5169 | Fe II | emission in strong discs; a shell absorption line when seen edge-on |
| 5876 | He I | photospheric absorption |
| 6563 | Hα | **the feature**: emission, EW a few to −40 Å; single-peaked pole-on, double-peaked at intermediate inclination, shell-cored edge-on |
