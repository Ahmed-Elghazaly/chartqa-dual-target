"""The synthetic pipeline's boxes must be exact, and the verifier must be able to reject.

A verifier that accepts everything is worse than none: it would certify wrong boxes and
we would train on them. Every acceptance test here is paired with an adversarial one.
"""

from __future__ import annotations

import random

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chartqa_dt.synth import generator as G
from chartqa_dt.synth.artists import clip_to_canvas, is_degenerate, point_box, points_to_pixels
from chartqa_dt.synth.curriculum import LEVELS
from chartqa_dt.synth.verify import (
    GEOMETRY_THRESHOLDS,
    check_box_for,
    containment,
    ink_bbox_iou,
    render_rgb,
)


@pytest.fixture(params=G.CHART_TYPES)
def scene(request):
    """A drawn chart plus its sentinel-coloured render, per chart type."""
    ct = request.param
    style = G.Style.sample(3)
    series, _q, _u = G.sample_series(random.Random(2001))
    fig, _ax, box_fns, recolour = G._draw(ct, series, style, random.Random(2001), "t")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    sentinels = [G.SENTINELS[i % len(G.SENTINELS)] for i in range(len(series))]
    recolour(sentinels)
    img = render_rgb(fig)
    labels = [lab for lab, _ in series]
    boxes = {}
    for lab in labels:
        b = clip_to_canvas(box_fns[lab](), width, height)
        if not is_degenerate(b, G.MIN_BOX_SIDE_PX):
            idx = 0 if ct in ("line", "multi_line", "area") else labels.index(lab)
            h = sentinels[idx].lstrip("#")
            boxes[lab] = (b, (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
    yield ct, img, boxes, (width, height)
    plt.close(fig)


def test_exact_boxes_are_accepted(scene):
    ct, img, boxes, _ = scene
    assert boxes, f"{ct} produced no usable boxes"
    for lab, (box, rgb) in boxes.items():
        r = check_box_for(img, box, rgb, lab, G.GEOMETRY_OF[ct])
        assert r.ok, f"{ct}/{lab}: exact box rejected — {r.reason}"


@pytest.mark.parametrize("kind", ["shift", "shrink", "grow", "far"])
def test_wrong_boxes_are_rejected(scene, kind):
    """The verifier must discriminate, or it certifies nothing."""
    ct, img, boxes, (width, height) = scene
    for lab, (b, rgb) in boxes.items():
        bw, bh = b[2] - b[0], b[3] - b[1]
        wrong = {
            "shift": (b[0] + bw * 0.5, b[1], b[2] + bw * 0.5, b[3]),
            "shrink": (b[0] + bw * 0.2, b[1] + bh * 0.2, b[2] - bw * 0.2, b[3] - bh * 0.2),
            "grow": (b[0] - bw * 0.4, b[1] - bh * 0.4, b[2] + bw * 0.4, b[3] + bh * 0.4),
            "far": (1.0, 1.0, 1.0 + bw, 1.0 + bh),
        }[kind]
        r = check_box_for(img, clip_to_canvas(wrong, width, height), rgb, lab,
                          G.GEOMETRY_OF[ct])
        assert not r.ok, f"{ct}/{lab}: {kind} box was ACCEPTED (fill {r.fill:.3f})"


def test_ink_iou_separates_exact_from_wrong(scene):
    """The margin, not just the verdict — a floor with no headroom is fragile."""
    ct, img, boxes, _ = scene
    for lab, (b, rgb) in boxes.items():
        bw, bh = b[2] - b[0], b[3] - b[1]
        exact = ink_bbox_iou(img, b, rgb)
        worst = max(ink_bbox_iou(img, w, rgb) for w in (
            (b[0] + bw * 0.5, b[1], b[2] + bw * 0.5, b[3]),
            (b[0] - bw * 0.4, b[1] - bh * 0.4, b[2] + bw * 0.4, b[3] + bh * 0.4),
        ))
        floor = GEOMETRY_THRESHOLDS[G.GEOMETRY_OF[ct]]["min_ink_iou"]
        assert exact > floor >= worst, f"{ct}/{lab}: exact {exact:.3f}, worst wrong {worst:.3f}"


def test_containment_is_unusable_for_shared_colour_markers():
    """Documents *why* line markers get their own geometry class.

    If this ever starts passing, the class distinction is obsolete and should go —
    the test exists so that stays visible rather than being rediscovered.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    vals = [5, 9, 3, 7, 6]
    ax.plot(range(5), vals, marker="o", markersize=12, color="#ff00ff",
            markerfacecolor="#ff00ff", markeredgecolor="#ff00ff", markeredgewidth=1.5)
    ax.set_ylim(0, 11.25)
    img = render_rgb(fig)
    worst = max(containment(img, point_box(fig, ax, i, v, 12, 1.5), (255, 0, 255))
                for i, v in enumerate(vals))
    plt.close(fig)
    assert worst < 0.5, f"containment now reads {worst:.3f} on shared-colour markers"


def test_marker_box_includes_the_stroke():
    """matplotlib centres a stroke on its path, so half of it lies outside the marker.

    Asserted on the box geometry rather than on containment: `containment` floors and
    ceils the box, and that ~1px of slack can absorb a thin stroke at low dpi, hiding a
    real error at higher dpi or heavier linewidths.
    """
    from chartqa_dt.synth.artists import points_to_pixels, scatter_point_box
    for dpi in (100, 200):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.scatter(range(3), [5, 9, 3], s=300, c=["#ff00ff", "#00ff00", "#ff0000"],
                   linewidths=1.5)
        ax.set_xlim(-0.6, 2.4)
        ax.set_ylim(0, 11.25)
        img = render_rgb(fig)
        padded = scatter_point_box(fig, ax, 0, 5, 300, 1.5)
        unpadded = scatter_point_box(fig, ax, 0, 5, 300, 0.0)
        grew = (padded[2] - padded[0]) - (unpadded[2] - unpadded[0])
        assert grew == pytest.approx(points_to_pixels(1.5, dpi), abs=1e-6), \
            f"dpi {dpi}: box grew by {grew}px, expected one full stroke width"
        assert containment(img, padded, (255, 0, 255)) == pytest.approx(1.0, abs=1e-6)
        plt.close(fig)


def test_labels_map_to_the_right_element():
    """A box around the *wrong* element of the right colour is invisible to pixel checks.

    Guarded structurally instead: with strictly increasing values, each successive box
    must sit strictly higher in the image (smaller y) than the last.
    """
    style = G.Style.sample(1)
    series = [("a", 10.0), ("b", 20.0), ("c", 30.0), ("d", 40.0)]
    for ct in ("vbar", "line", "scatter", "area"):
        fig, _ax, box_fns, _rc = G._draw(ct, series, style, random.Random(0), None)
        fig.canvas.draw()
        centres = [sum(box_fns[lab]()[1::2]) / 2 for lab, _ in series]
        plt.close(fig)
        assert centres == sorted(centres, reverse=True), \
            f"{ct}: label->element mapping is scrambled ({centres})"


def test_holdout_is_decided_in_exactly_one_place(tmp_path):
    assert G.is_holdout(next(iter(G.HOLDOUT_STYLE_SEEDS)), 0)
    assert G.is_holdout(0, G.HOLDOUT_SEED_START)
    assert not G.is_holdout(0, G.HOLDOUT_SEED_START - 1)
    train = G.generate_batch(6, tmp_path / "train", seed=1, holdout=False)
    held = G.generate_batch(6, tmp_path / "held", seed=1, holdout=True)
    assert train and held
    assert not any(e.holdout for e in train)
    assert all(e.holdout for e in held)
    assert not ({e.style_seed for e in train} & G.HOLDOUT_STYLE_SEEDS)


def test_points_to_pixels_uses_the_figure_dpi():
    assert points_to_pixels(72.0, 100.0) == pytest.approx(100.0)
    assert points_to_pixels(36.0, 200.0) == pytest.approx(100.0)


def test_unknown_geometry_raises_rather_than_defaulting():
    import numpy as np
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown geometry"):
        check_box_for(img, (0, 0, 5, 5), (0, 0, 0), "x", "triangle")


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_generates_on_every_chart_type(level, tmp_path):
    for ct in G.CHART_TYPES:
        ex = G.generate_example(chart_type=ct, level=level, style_seed=3,
                                data_seed=2001, out_dir=tmp_path)
        assert ex is not None, f"{ct}/{level} produced nothing"
        assert ex.level == level and ex.chart_type == ct


def test_saved_image_never_contains_sentinel_colours(tmp_path):
    """The verification recolour must be undone before the image is written."""
    import numpy as np
    from PIL import Image
    sentinels = {tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for c in G.SENTINELS}
    for ct in G.CHART_TYPES:
        ex = G.generate_example(chart_type=ct, level="L1", style_seed=3,
                                data_seed=2001, out_dir=tmp_path)
        assert ex is not None
        arr = np.asarray(Image.open(ex.image_path).convert("RGB")).reshape(-1, 3)
        present = {tuple(int(v) for v in c) for c in np.unique(arr, axis=0)}
        assert not (present & sentinels), f"{ct}: sentinel colour leaked into the PNG"


def test_element_colours_are_distinct_beyond_the_match_tolerance():
    for palette in G.PALETTES:
        rgbs = [tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                for c in G.element_colours(palette, 7)]
        for i, a in enumerate(rgbs):
            for b in rgbs[i + 1:]:
                assert max(abs(x - y) for x, y in zip(a, b)) > 12, \
                    f"{a} and {b} are within the verifier's tolerance"


def test_boxes_are_normalised_into_the_0_1000_range(tmp_path):
    for ct in G.CHART_TYPES:
        ex = G.generate_example(chart_type=ct, level="L2", style_seed=5,
                                data_seed=2002, out_dir=tmp_path)
        assert ex is not None
        for item in ex.evidence:
            x1, y1, x2, y2 = item["bbox"]
            assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000, item
