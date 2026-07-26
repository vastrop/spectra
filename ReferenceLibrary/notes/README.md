# Spectral-type cards — how to read them

One card per Pickles reference spectrum (`../<stem>.dat` ↔ `<stem>.md`). Each has
the shared class prose, what is specific to this subtype/luminosity class, and a
table of annotatable features.

Two caveats apply to every card:

**These describe stellar physics, not necessarily what the template shows.**
The Pickles library is sampled at 5 Å and its spectra are flux-calibrated
*composites* assembled from several sources — excellent for spectral energy
distributions and broad features, but not a homogeneous high-resolution MK
atlas. Descriptions of line cores, wing widths and "sharper metal lines" are
expected higher-resolution behaviour; do not assume they are measurable in the
loaded template at this sampling.

**The wavelength column mixes three kinds of thing**, all approximate air
values: single atomic lines (4226 = Ca I), molecular bandheads (4954 = TiO), and
band or index intervals (6814-6846 = CaH2). A range means the feature is
extended, not that the line is uncertain.

The `r` and `w` filename prefixes are Pickles' metal-rich and metal-weak
abundance groups. They are empirical composites of the same *nominal* spectral
and luminosity class — not controlled pairs at identical effective temperature
and gravity — so treat their metallicity trends as expectations, not guarantees.
Metal-weak is also not the same thing as MK luminosity class VI (subdwarf).

## Sources

The cards carry no per-card citations: every classification claim in them comes
from this same short list, so it is kept here once and shown as clickable links
under each card in the DB browser. Check any claim against these.

- [Gray, A Digital Spectral Classification Atlas](https://ned.ipac.caltech.edu/level5/Gray/TOC.html)
- [Morgan, Keenan & Kellman, An Atlas of Stellar Spectra](https://www.ucl.ac.uk/mathematical-physical-sciences/sites/mathematical_physical_sciences/files/mkkbook.pdf)
- [Pickles 1998, A Stellar Spectral Flux Library](https://www.eso.org/sci/observing/tools/standards/IR_spectral_library/hilib.pdf)
- [NIST Handbook of Basic Atomic Spectroscopic Data](https://physics.nist.gov/PhysRefData/Handbook/atomic_number.htm)
- [Sota et al., GOSSS O-star classification](https://arxiv.org/abs/1101.4002)
- [Kirkpatrick, Henry & McCarthy 1991, K5-M9 standards](https://articles.adsabs.harvard.edu/pdf/1991ApJS...77..417K)
