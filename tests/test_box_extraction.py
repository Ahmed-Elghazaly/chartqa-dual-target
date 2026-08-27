"""Exact bounding-box extraction from matplotlib artists.

`PLAN.md` 3.5: *"Box extraction must be exact ... Never estimate a box by eye or
by formula."* A generator with subtly wrong boxes poisons every training example,
is invisible in the loss, and is nearly undetectable once training has begun. So
the technique is proven against rendered pixels, in CI, before the generator that
depends on it exists.

The proof is deliberately adversarial. Checking that a box *contains* its bar is
not enough — an over-large box passes that trivially. It must also exclude its
neighbours, exclude the strip immediately above it, and a box shifted by most of
its own width must **fail**, or the check cannot tell exact from approximate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))
pytest.importorskip("matplotlib")
prove = pytest.importorskip("prove_box_extraction")


@pytest.fixture(scope="module")
def rendered():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = [(200, 30, 30), (30, 140, 60), (40, 60, 200), (220, 160, 20)]
    values = [37, 82, 55, 19]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    bars = list(ax.bar(list("ABCD"), values, color=[tuple(c / 255 for c in col) for col in colours]))
    ax.set_ylim(0, 100)
    img = prove.render_rgb(fig)
    boxes = prove.bar_boxes_in_pixels(fig, ax, bars)
    plt.close(fig)
    return img, boxes, colours, values


def test_each_box_is_filled_by_its_own_bar(rendered):
    img, boxes, colours, _ = rendered
    for i, (box, colour) in enumerate(zip(boxes, colours)):
        frac = prove.fraction_of_colour(img, box, colour)
        assert frac > 0.97, f"bar {i}: only {100 * frac:.1f}% of its box is its own colour"


def test_no_box_contains_a_neighbouring_bar(rendered):
    img, boxes, colours, _ = rendered
    for i, box in enumerate(boxes):
        for j, other in enumerate(colours):
            if i == j:
                continue
            bleed = prove.fraction_of_colour(img, box, other)
            assert bleed <= 0.01, f"bar {i}'s box contains {100 * bleed:.1f}% of bar {j}"


def test_an_approximately_right_box_fails(rendered):
    """Without this, the suite would accept a box that is merely close."""
    img, boxes, colours, _ = rendered
    for i, (box, colour) in enumerate(zip(boxes, colours)):
        width = box[2] - box[0]
        shifted = (box[0] + width * 0.6, box[1], box[2] + width * 0.6, box[3])
        frac = prove.fraction_of_colour(img, shifted, colour)
        assert frac < 0.5, f"bar {i}: a box shifted by 60% of its width still scores {100 * frac:.1f}%"


def test_the_top_edge_is_the_top_of_the_bar(rendered):
    """Catches a box that is correct in x but too short or too tall in y."""
    img, boxes, colours, _ = rendered
    for i, (box, colour) in enumerate(zip(boxes, colours)):
        strip = max(2, int(0.02 * (box[3] - box[1])))
        above = (box[0], max(0, box[1] - strip * 3), box[2], max(0, box[1] - strip))
        frac = prove.fraction_of_colour(img, above, colour)
        assert frac <= 0.05, f"bar {i}: the strip above its box is {100 * frac:.1f}% bar colour"


def test_box_heights_are_proportional_to_the_plotted_values(rendered):
    """Real geometry scales with the data; an estimate would not, exactly."""
    _, boxes, _, values = rendered
    ratios = [(b[3] - b[1]) / v for b, v in zip(boxes, values)]
    spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios))
    assert spread < 0.02, f"height/value ratios vary by {spread:.3f}; geometry is not exact"


def test_y_axis_is_flipped_into_image_coordinates(rendered):
    """Display space has origin bottom-left; images have origin top-left.

    A missing flip is the single easiest way to get boxes that look plausible and
    are vertically mirrored. Taller bars must have SMALLER y1 (start higher up).
    """
    _, boxes, _, values = rendered
    tallest = max(range(len(values)), key=lambda i: values[i])
    shortest = min(range(len(values)), key=lambda i: values[i])
    assert boxes[tallest][1] < boxes[shortest][1], (
        "the taller bar's box does not start higher in the image; the y flip is wrong"
    )
    for box in boxes:
        assert box[1] < box[3], "y1 must be above y2 in image coordinates"
        assert box[0] < box[2], "x1 must be left of x2"


def test_the_proof_script_passes_end_to_end():
    assert prove.main() == 0
