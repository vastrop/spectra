"""Self-check for match_reference_templates (spectrum_core).

Uses the real ReferenceLibrary: a library template restricted to a
typical observed band (plus noise) must be recovered as the best match,
for both a hot star (a0v) and a cool giant (m4iii — the T CrB case the
matcher exists for, where the Planck colour temperature underreads).
"""

import os
import sys

import numpy as np

# The module under test and ReferenceLibrary/ live in the repo root, one
# level up from tests/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from spectrum_core import load_reference_library, match_reference_templates

LIB_DIR = os.path.join(_ROOT, "ReferenceLibrary")


def _observed_from(library, name, noise=0.02, seed=0):
    """Simulate an observation of a library star over 3900–7200 Å."""
    twl, tfx = library[name]
    wls = np.linspace(3900.0, 7200.0, 1000)
    flux = np.interp(wls, twl, tfx)
    flux *= np.exp(np.random.default_rng(seed).normal(0, noise, wls.size))
    return wls, 3.7 * flux    # arbitrary amplitude — must not matter


def main():
    library = load_reference_library(LIB_DIR)
    assert len(library) > 50, f"library too small: {len(library)}"

    for name in ("a0v", "m4iii"):
        match = match_reference_templates(*_observed_from(library, name),
                                          library=library)
        assert match is not None
        best = match["ranked"][0]["name"]
        assert best == name, f"{name}: best match was {best}"
        # A clean self-match must beat the runner-up clearly.
        assert (match["ranked"][0]["rms"]
                < 0.8 * match["ranked"][1]["rms"]), match["ranked"][:2]

    # The mask must exclude telluric bands (7594 O2 A sits in-band when
    # the range extends; here check Balmer: 6563 must be masked out).
    wls, flux = _observed_from(library, "a0v")
    match = match_reference_templates(wls, flux, library)
    grid, mask = match["grid"], match["mask"]
    assert not mask[np.abs(grid - 6563) < 30].any(), "Balmer core not masked"

    # Degenerate inputs.
    assert match_reference_templates([5000], [1.0], library) is None
    assert match_reference_templates(wls, flux, {}) is None

    print("test_template_match: all checks passed")


if __name__ == "__main__":
    main()
