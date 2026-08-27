"""The RefChartQA audit pre-screen: does a box contain any chart ink at all?

`PLAN.md` 3.4 gates RefChartQA training data on 200 human judgements. This does
not replace them — it adds one objective signal computable for **every** box:
a box on blank canvas contains no evidence, whatever the question asked.

Validated against synthetic charts whose true boxes are known by construction, so
the detector is proven before it is pointed at data whose labels are the very
thing under question.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))
pytest.importorskip("matplotlib")
ink = pytest.importorskip("prove_ink_prescreen")


@pytest.fixture(scope="module")
def chart():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    bars = list(ax.bar(list("ABCD"), [37, 82, 55, 19], color="#3060c8"))
    ax.set_ylim(0, 100)
    img = ink.render(fig)
    boxes = ink.bar_boxes(fig, bars)
    plt.close(fig)
    return img, boxes


def test_true_boxes_are_ink_rich(chart):
    img, boxes = chart
    for i, box in enumerate(boxes):
        assert ink.ink_fraction(img, box) > 0.90, f"true box {i} is not ink-rich"


def test_blank_regions_are_ink_poor(chart):
    """The failure the pre-screen exists to catch: a box on empty canvas."""
    img, boxes = chart
    top = min(b[1] for b in boxes)
    left, right = min(b[0] for b in boxes), max(b[2] for b in boxes)
    blank = (left, top - 55, right, top - 10)
    assert ink.ink_fraction(img, blank) < 0.05


def test_the_two_populations_separate_widely(chart):
    """A threshold is only usable if there is room between the classes."""
    img, boxes = chart
    top = min(b[1] for b in boxes)
    left, right = min(b[0] for b in boxes), max(b[2] for b in boxes)
    blank = (left, top - 55, right, top - 10)
    worst_true = min(ink.ink_fraction(img, b) for b in boxes)
    assert worst_true - ink.ink_fraction(img, blank) > 0.50


def test_a_partially_overlapping_box_scores_in_between(chart):
    """The signal must be graded, not binary, or it cannot rank borderline boxes."""
    img, boxes = chart
    box = boxes[1]
    w = box[2] - box[0]
    half = (box[0] + w * 0.5, box[1], box[2] + w * 0.5, box[3])
    assert 0.10 < ink.ink_fraction(img, half) < 0.95


def test_background_is_not_assumed_to_be_white():
    """RefChartQA contains dark and tinted charts; assuming white would invert
    the signal on every one of them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100, facecolor="#20242b")
    ax.set_facecolor("#20242b")
    bars = list(ax.bar(list("XY"), [60, 30], color="#e0a020"))
    ax.set_ylim(0, 100)
    img = ink.render(fig)
    boxes = ink.bar_boxes(fig, bars)
    plt.close(fig)
    assert ink.ink_fraction(img, boxes[0]) > 0.90


def test_degenerate_boxes_do_not_crash(chart):
    img, _ = chart
    for box in [(0, 0, 0, 0), (10, 10, 5, 5), (-50, -50, -10, -10),
                (10_000, 10_000, 10_100, 10_100)]:
        assert ink.ink_fraction(img, box) == 0.0


def test_the_proof_script_passes_end_to_end():
    assert ink.main() == 0
