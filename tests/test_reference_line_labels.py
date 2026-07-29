"""
test_reference_line_labels.py
=============================
Label placement in ``plot_reference_lines``.

Every line in view gets a rule; a label is dropped when a neighbour
already holds the space, measured off the rendered text rather than
estimated from the font size.  Renders onto an explicit Agg Figure
(never pyplot — see the figure convention in spectrum_explorer.py) and
reads the artists back.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib.backends.backend_agg import FigureCanvasAgg   # noqa: E402
from matplotlib.figure import Figure                          # noqa: E402

from spectrum_core import (                                    # noqa: E402
    LABEL_LEVEL,
    LABEL_MAX_LEVELS,
    plot_reference_lines,
)

XLIM = (4000.0, 8000.0)


def panel():
    """An axes with a real Agg canvas, so text can actually be measured."""
    fig = Figure(figsize=(16.0, 6.0))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(*XLIM)
    ax.set_ylim(0.0, 1.0)
    return ax


def label_width_data(ax, label, fontsize):
    """Rendered width of one label, in data units on ``ax``."""
    txt = ax.text(5000.0, 0.5, label, rotation=90, fontsize=fontsize)
    bb = txt.get_window_extent(ax.figure.canvas.get_renderer())
    txt.remove()
    inv = ax.transData.inverted()
    return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]


def drawn(ax):
    return sorted(t.get_text() for t in ax.texts)


def main():
    # Well-separated lines all keep their labels, all at one height.
    ax = panel()
    plot_reference_lines(ax, {4500.0: "a", 5500.0: "b", 6500.0: "c"},
                         1.0, 1.0, fontsize=10)
    assert drawn(ax) == ["a", "b", "c"], drawn(ax)
    assert {round(t.get_position()[1], 6) for t in ax.texts} == {LABEL_LEVEL}

    # Spacing follows the *measured* label, not a font-size guess: set
    # over one rendered width apart, both stay on the bottom row.
    ax = panel()
    w = label_width_data(ax, "Hβ 4861", 10)
    plot_reference_lines(ax, {5000.0: "Hβ 4861", 5000.0 + w * 1.4: "Hγ 4340"},
                         1.0, 1.0, fontsize=10)
    assert drawn(ax) == ["Hβ 4861", "Hγ 4340"], drawn(ax)
    assert len({round(t.get_position()[1], 6) for t in ax.texts}) == 1

    # Too close, and the second is raised a row rather than dropped.
    ax = panel()
    plot_reference_lines(ax, {5000.0: "Hβ 4861", 5000.0 + w * 0.3: "Hγ 4340"},
                         1.0, 1.0, fontsize=10)
    assert drawn(ax) == ["Hβ 4861", "Hγ 4340"], drawn(ax)
    ys = {t.get_text(): t.get_position()[1] for t in ax.texts}
    assert ys["Hγ 4340"] > ys["Hβ 4861"], ys

    # Rows run out eventually: past LABEL_MAX_LEVELS piled on one spot,
    # the surplus keeps its rule and loses its label.
    ax = panel()
    n = LABEL_MAX_LEVELS + 2
    plot_reference_lines(ax, {5000.0 + i * w * 0.05: f"l{i}" for i in range(n)},
                         1.0, 1.0, fontsize=10)
    assert len(ax.lines) == n, len(ax.lines)
    assert len(ax.texts) == LABEL_MAX_LEVELS, drawn(ax)

    # Partial overlap is tolerated up to LABEL_OVERLAP_FRAC before a raise
    # is spent: interleaved glyph columns stay readable, and rows are a
    # scarce resource worth saving for real collisions.
    ax = panel()
    plot_reference_lines(ax, {5000.0: "Hβ 4861", 5000.0 + w * 0.8: "Hγ 4340"},
                         1.0, 1.0, fontsize=10)
    assert len({round(t.get_position()[1], 6) for t in ax.texts}) == 1, \
        [t.get_position() for t in ax.texts]

    # A rotated label's x-footprint comes from the font, not the string —
    # its length runs vertically.  Worth pinning, because it is why the
    # spacing is one measured width per font size rather than anything
    # per-label, while the y-headroom below *does* scale with length.
    ax = panel()
    assert abs(label_width_data(ax, "Hε + Ca II H 3970", 10) - w) < 0.01 * w
    assert label_width_data(ax, "Hβ 4861", 20) > w * 1.5    # tracks font size

    # A shared `occupied` list is what stops two separately drawn groups
    # from printing over each other: with it the second group's label is
    # raised, without it both land on the same row at the same spot.
    ax = panel()
    slots = []
    plot_reference_lines(ax, {5000.0: "first"}, 1.0, 1.0, fontsize=10,
                         occupied=slots)
    plot_reference_lines(ax, {5000.0 + w * 0.3: "second"}, 1.0, 1.0,
                         fontsize=10, occupied=slots)
    ys = {t.get_text(): t.get_position()[1] for t in ax.texts}
    assert ys["second"] > ys["first"], ys

    ax = panel()
    plot_reference_lines(ax, {5000.0: "first"}, 1.0, 1.0, fontsize=10)
    plot_reference_lines(ax, {5000.0 + w * 0.3: "second"}, 1.0, 1.0,
                         fontsize=10)
    ys = {t.get_text(): t.get_position()[1] for t in ax.texts}
    assert ys["second"] == ys["first"], ys

    # Headroom tracks label length, so a wordy label is not clipped.
    def top_for(label):
        ax = panel()
        plot_reference_lines(ax, {5000.0: label}, 1.0, 1.0, fontsize=12)
        return ax.get_ylim()[1]

    assert top_for("Hα 6563") > LABEL_LEVEL
    assert top_for("Hε + Ca II H 3970") > top_for("Hα 6563"), \
        (top_for("Hε + Ca II H 3970"), top_for("Hα 6563"))

    # The label masks its own rule, so it is never read through the line.
    ax = panel()
    plot_reference_lines(ax, {5000.0: "a"}, 1.0, 1.0, fontsize=10)
    bbox = ax.texts[0].get_bbox_patch()
    assert bbox is not None and bbox.get_facecolor()[3] == 1.0, bbox

    # Lines outside the visible range are dropped entirely, and an
    # all-out-of-range call must not touch the y-limits.
    ax = panel()
    plot_reference_lines(ax, {100.0: "far", 99000.0: "farther"}, 1.0, 1.0)
    assert not ax.texts and not ax.lines
    assert ax.get_ylim() == (0.0, 1.0), ax.get_ylim()

    print("reference-line label self-checks passed")


if __name__ == "__main__":
    main()
