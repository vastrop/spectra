"""
test_reference_line_labels.py
=============================
Label placement in ``plot_reference_lines``.

The stagger is what keeps a crowded group readable: neighbours that would
overprint step up a level, and an isolated line drops back to the
baseline.  Renders onto an explicit Agg Figure (never pyplot — see the
figure convention in spectrum_explorer.py) and reads the text artists'
y-positions back.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib.figure import Figure           # noqa: E402

from spectrum_core import (                    # noqa: E402
    LABEL_LEVELS,
    LABEL_MIN_GAP_FRAC,
    plot_reference_lines,
)


def label_levels(lines, xlim=(4000.0, 8000.0), y_max=1.0):
    """Draw on a throwaway axes; return [(x, level_index)] in x-order."""
    ax = Figure().add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, y_max)
    plot_reference_lines(ax, lines, 1.0, y_max)
    out = []
    for txt in ax.texts:
        x, y = txt.get_position()
        # Levels are exact multiples of y_max, so the lookup is safe.
        out.append((x, min(range(len(LABEL_LEVELS)),
                           key=lambda i: abs(LABEL_LEVELS[i] * y_max - y))))
    return sorted(out)


def main():
    span = 8000.0 - 4000.0
    gap = LABEL_MIN_GAP_FRAC * span     # bump threshold, in Å here

    # Well-separated lines all sit on the baseline: no gratuitous stagger.
    got = label_levels({4500.0: "a", 5500.0: "b", 6500.0: "c"})
    assert [lvl for _x, lvl in got] == [0, 0, 0], got

    # A tight cluster steps up level by level, then wraps — three levels
    # is the documented ceiling, not a promise of infinite headroom.
    tight = {5000.0 + i * gap * 0.5: f"l{i}" for i in range(4)}
    got = label_levels(tight)
    assert [lvl for _x, lvl in got] == [0, 1, 2, 0], got

    # A gap resets the stagger rather than carrying the level onward.
    mixed = {5000.0: "a", 5000.0 + gap * 0.5: "b", 7000.0: "c"}
    got = label_levels(mixed)
    assert [lvl for _x, lvl in got] == [0, 1, 0], got

    # Headroom follows the highest level used, so the tallest label has
    # room above it instead of being clipped at the old fixed 1.25.
    ax = Figure().add_subplot(111)
    ax.set_xlim(4000.0, 8000.0)
    ax.set_ylim(0.0, 1.0)
    plot_reference_lines(ax, tight, 1.0, 1.0)
    assert ax.get_ylim()[1] >= LABEL_LEVELS[2] + 0.2, ax.get_ylim()

    # Lines outside the visible range are dropped entirely, and an
    # all-out-of-range call must not touch the y-limits.
    ax = Figure().add_subplot(111)
    ax.set_xlim(4000.0, 8000.0)
    ax.set_ylim(0.0, 1.0)
    plot_reference_lines(ax, {100.0: "far", 99000.0: "farther"}, 1.0, 1.0)
    assert not ax.texts
    assert ax.get_ylim() == (0.0, 1.0), ax.get_ylim()

    print("reference-line label self-checks passed")


if __name__ == "__main__":
    main()
