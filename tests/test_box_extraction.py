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


# ---------------------------------------- line, pie and scatter (PLAN.md 3.5)

all_types = pytest.importorskip("prove_box_extraction_all_types")


def test_line_vertex_boxes_are_exact():
    """`Line2D.get_window_extent()` covers the whole polyline, which is not what
    grounding needs; a per-vertex box comes from `transData.transform` plus the
    marker radius. A disc inscribed in its bounding square fills pi/4 = 78.5%, so
    a correct box lands near that — too high means the box is too small, too low
    means it is too large or misplaced."""
    assert all_types.prove_line() == []


def test_pie_wedge_boxes_contain_and_are_tight():
    """A wedge is a sector, so its bounding box necessarily includes background
    and slivers of neighbours. 'No other colour inside' would be wrong by
    construction; containment plus tightness is the correct pair."""
    assert all_types.prove_pie() == []


def test_scatter_marker_boxes_use_points_squared():
    """`s` is an AREA in points squared, so diameter is sqrt(s) points. Treating
    `s` as a diameter would make every scatter box far too large, and the box
    would still contain its marker — which is why the check has an upper bound."""
    assert all_types.prove_scatter() == []


def test_points_to_pixels_conversion():
    """72 points per inch, dpi pixels per inch. Guessing this is how marker boxes
    end up plausible and wrong."""
    assert all_types.prove_points_to_pixels() == []
    assert all_types.points_to_pixels(72, 100) == pytest.approx(100.0)
    assert all_types.points_to_pixels(10, 100) == pytest.approx(13.888888888888889)


def test_a_scatter_box_that_is_too_small_would_be_rejected():
    """Guards the guard: the upper bound must actually bite."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = (150, 40, 160)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.scatter([3.0], [50.0], s=400.0, c=[tuple(c / 255 for c in colour)])
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 100)
    img = all_types.render_rgb(fig)
    correct = all_types.scatter_point_box(fig, ax, (3.0, 50.0), 400.0)
    plt.close(fig)

    w = correct[2] - correct[0]
    h = correct[3] - correct[1]
    too_small = (correct[0] + 0.3 * w, correct[1] + 0.3 * h,
                 correct[2] - 0.3 * w, correct[3] - 0.3 * h)
    assert all_types.fraction_inside(img, too_small, colour) > 0.90, (
        "a box entirely inside the marker should be ~100% marker colour, "
        "which is exactly why the scatter check has an upper bound"
    )
