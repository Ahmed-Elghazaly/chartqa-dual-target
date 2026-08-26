"""Coordinate mathematics.

Two tests here are load-bearing beyond ordinary correctness:

* :func:`test_clamp_rescues_the_edge_touching_box` guards `DECISIONS.md` 0004.
  The official evaluator discards any box containing a coordinate of 1000, with
  no error. Charts produce those constantly.
* :func:`test_our_smart_resize_matches_transformers` guards `DECISIONS.md` 0008.
  If our port ever drifts from the implementation that actually runs, every
  token count and every sub-token statistic in the report becomes fiction.
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

from chartqa_dt.vision.coords import (
    OFFICIAL_MAX_COORD,
    QWEN2VL_FACTOR,
    QWEN3VL_FACTOR,
    VisualGeometry,
    clamp_for_official_evaluator,
    norm1000_to_px,
    px_to_norm1000,
    remap_crop_box_to_original,
    smart_resize,
    smart_resize_appendix_c,
)

QWEN3VL = VisualGeometry(
    factor=QWEN3VL_FACTOR, min_pixels=65536, max_pixels=16777216, patch_size=16, merge_size=2
)

SHAPES = [
    (557, 800), (600, 850), (507, 858), (836, 800), (224, 224),
    (1080, 1920), (100, 100), (28, 28), (4000, 3000), (33, 512),
]


# --------------------------------------------------------------- smart_resize


@pytest.mark.parametrize(("h", "w"), SHAPES)
def test_smart_resize_output_is_divisible_by_factor(h, w):
    rh, rw = QWEN3VL.resize(h, w)
    assert rh % QWEN3VL.factor == 0 and rw % QWEN3VL.factor == 0


@pytest.mark.parametrize(("h", "w"), SHAPES)
def test_smart_resize_respects_pixel_bounds(h, w):
    rh, rw = QWEN3VL.resize(h, w)
    assert rh * rw <= QWEN3VL.max_pixels * 1.02
    assert rh * rw >= min(QWEN3VL.min_pixels * 0.98, h * w * 4)


# Shapes where both sides comfortably exceed the factor. Chart images are all
# in this regime (RefChartQA is dominated by 800x557).
NORMAL_SHAPES = [s for s in SHAPES if min(s) >= 4 * QWEN3VL_FACTOR]


@pytest.mark.parametrize(("h", "w"), NORMAL_SHAPES)
def test_smart_resize_preserves_aspect_ratio_closely(h, w):
    rh, rw = QWEN3VL.resize(h, w)
    assert abs(math.log((rw / rh) / (w / h))) < 0.25


def test_smart_resize_distorts_thin_images_and_that_is_expected():
    """Aspect ratio is only preserved "as closely as possible", and for a thin
    image that is not very close at all.

    33x512 has 16,896 pixels, below min_pixels, so it is scaled up by
    beta = sqrt(65536/16896) = 1.9695 and each side is then rounded UP to a
    multiple of 32. On the 33-pixel side that rounding is brutal:
    ceil(33*1.9695/32)*32 = 96, against an ideal of ~65. The result is 96x1024,
    a 1.5x vertical stretch.

    This is real behaviour of the algorithm, not a defect in our port. It does
    not affect box mapping, because horizontal and vertical scale factors are
    tracked separately, but it does mean the model sees a distorted picture.
    Recorded here so a future reader does not mistake it for a bug.
    """
    rh, rw = QWEN3VL.resize(33, 512)
    assert (rh, rw) == (96, 1024)
    distortion = abs(math.log((rw / rh) / (512 / 33)))
    assert distortion > 0.35, "the distortion is large, and that is the point"

    # Chart-shaped images are nowhere near this regime: 557 -> 544 is a 2.3%
    # distortion, against 45% for the thin case above.
    ch, cw = QWEN3VL.resize(557, 800)
    assert abs(math.log((cw / ch) / (800 / 557))) < 0.03


def test_smart_resize_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError, match="aspect ratio"):
        smart_resize(1, 500, 32, 65536, 16777216)


def test_smart_resize_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError, match="positive"):
        smart_resize(0, 100, 32, 65536, 16777216)


@pytest.mark.official
def test_our_smart_resize_matches_transformers():
    """The port must equal the implementation that actually runs (decision 0008)."""
    tfm = pytest.importorskip(
        "transformers.models.qwen2_vl.image_processing_qwen2_vl",
        reason="transformers not installed (CPU test env)",
    )
    for h, w in SHAPES:
        for factor in (QWEN3VL_FACTOR, QWEN2VL_FACTOR):
            ours = smart_resize(h, w, factor, 65536, 16777216)
            theirs = tfm.smart_resize(h, w, factor=factor, min_pixels=65536, max_pixels=16777216)
            assert ours == tuple(theirs), f"{h}x{w} factor={factor}: {ours} != {theirs}"


def test_appendix_c_is_kept_only_for_comparison_and_does_diverge():
    """Document, by test, that the plan's transcription is not what we run.

    Appendix C puts the `max(factor, ...)` guard on the initial rounding rather
    than in the downscale branch. On a tall thin image that is downscaled hard,
    the two formulations disagree.
    """
    h, w, factor = 33, 512, 32
    max_px = 64 * 64
    ours = smart_resize(h, w, factor, 4 * 28 * 28, max_px)
    theirs = smart_resize_appendix_c(h, w, factor, 4 * 28 * 28, max_px)
    assert ours != theirs, "expected the two formulations to diverge on this shape"
    assert min(ours) >= factor, "our version never returns a dimension below one factor"
    assert min(theirs) == 0, "Appendix C's version can return a zero dimension here"


def test_appendix_c_default_factor_is_the_wrong_one_for_qwen3vl():
    """The whole point of decision 0008, pinned as a test."""
    assert smart_resize_appendix_c.__defaults__[0] == QWEN2VL_FACTOR == 28
    assert QWEN3VL_FACTOR == 32
    assert QWEN3VL.factor == 32


# ------------------------------------------------------------ visual tokens


def test_token_counts_match_the_recorded_measurements():
    """These exact numbers appear in verification/phase0.md F11 and decision 0008."""
    assert QWEN3VL.n_visual_tokens(557, 800) == 425                      # native
    assert QWEN3VL.with_max_pixels(512 * 512).n_visual_tokens(557, 800) == 247


def test_sub_token_detection_on_a_real_refchartqa_box():
    """RefChartQA_human_val_0: 800x386 image, box x=276 y=277 w=60 h=23."""
    box = (276.0, 277.0, 276.0 + 60.0, 277.0 + 23.0)
    tw, th = QWEN3VL.box_in_tokens(box, img_h=386, img_w=800)
    assert th < 1.0 < tw, "this box is sub-token vertically but not horizontally"
    assert QWEN3VL.is_sub_token(box, 386, 800, rule="axis") is True
    assert QWEN3VL.is_sub_token(box, 386, 800, rule="area") is False


def test_sub_token_gets_worse_at_lower_resolution():
    box = (276.0, 277.0, 336.0, 300.0)
    native = QWEN3VL.box_in_tokens(box, 557, 800)
    small = QWEN3VL.with_max_pixels(448 * 448).box_in_tokens(box, 557, 800)
    assert small[0] < native[0] and small[1] < native[1]


def test_unknown_sub_token_rule_raises():
    with pytest.raises(ValueError, match="unknown rule"):
        QWEN3VL.is_sub_token((0, 0, 10, 10), 100, 100, rule="diagonal")


def test_geometry_refuses_to_guess_from_a_bare_processor():
    class Bare:
        pass

    with pytest.raises(ValueError, match="refusing to guess"):
        VisualGeometry.from_processor(Bare())


def test_geometry_reads_a_processor_like_object():
    class IP:
        patch_size = 16
        merge_size = 2
        size: ClassVar[dict] = {"shortest_edge": 65536, "longest_edge": 16777216}

    class Proc:
        image_processor = IP()

    g = VisualGeometry.from_processor(Proc())
    assert (g.factor, g.min_pixels, g.max_pixels) == (32, 65536, 16777216)
    assert "IP" in g.source


def test_geometry_accepts_legacy_min_max_pixels_attributes():
    class IP:
        patch_size = 14
        merge_size = 2
        size: ClassVar[dict] = {}
        min_pixels = 3136
        max_pixels = 12845056

    g = VisualGeometry.from_processor(IP())
    assert g.factor == 28 and g.max_pixels == 12845056


# ------------------------------------------------------- normalised 0-1000


@pytest.mark.parametrize(
    ("box", "w", "h"),
    [
        ((0, 0, 100, 50), 800, 600),
        ((12.5, 33.25, 700.75, 599.0), 800, 600),
        ((0, 0, 800, 600), 800, 600),
        ((399, 299, 401, 301), 800, 600),
    ],
)
def test_norm1000_roundtrip_is_identity(box, w, h):
    back = norm1000_to_px(px_to_norm1000(box, w, h), w, h)
    for a, b in zip(box, back):
        assert abs(a - b) < 1e-9


def test_full_image_box_maps_to_the_full_normalised_range():
    assert px_to_norm1000((0, 0, 800, 600), 800, 600) == [0.0, 0.0, 1000.0, 1000.0]


# ---- the silent-discard guard (decision 0004) ----


def test_clamp_rescues_the_edge_touching_box():
    """A box on the right/bottom edge normalises to 1000, which the official
    evaluator throws away entirely. This is the whole reason clamping exists."""
    full = px_to_norm1000((0, 0, 800, 600), 800, 600)
    assert 1000.0 in full, "precondition: an edge-touching box does produce a 1000"
    clamped = clamp_for_official_evaluator(full)
    assert clamped == [0, 0, 999, 999]
    assert all(0 <= v <= OFFICIAL_MAX_COORD for v in clamped)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([0, 0, 1000, 1000], [0, 0, 999, 999]),
        ([-5, -0.4, 1200, 1000.6], [0, 0, 999, 999]),
        ([12.4, 12.6, 100.5, 200.49], [12, 13, 100, 200]),
        ([999.9, 999.9, 1000, 1000], [999, 999, 999, 999]),
    ],
)
def test_clamp_cases(raw, expected):
    assert clamp_for_official_evaluator(raw) == expected


def test_clamped_boxes_would_survive_the_official_extractor():
    """Replicates the exact acceptance test in the released evaluate.py."""
    bins = 1000
    for raw in ([0, 0, 1000, 1000], [500, 500, 1000, 800], [1000, 1000, 1000, 1000]):
        assert not all(0 <= v <= bins - 1 for v in raw) or raw[2] < 1000
        clamped = clamp_for_official_evaluator(raw)
        assert all(0 <= v <= bins - 1 for v in clamped), "clamped box must survive"


# ------------------------------------------------------------- crop remapping


def test_crop_remap_on_a_hand_computed_case():
    """Original 1000x1000. Crop is the bottom-right quadrant (500,500)-(1000,1000).
    A box at the exact centre of that crop, normalised 250..750 within it, covers
    original pixels 625..875, i.e. normalised 625..875 of the original."""
    out = remap_crop_box_to_original((250, 250, 750, 750), (500, 500, 1000, 1000), 1000, 1000)
    for a, b in zip(out, [625.0, 625.0, 875.0, 875.0]):
        assert abs(a - b) < 1e-9


def test_crop_remap_of_the_whole_crop_returns_the_crop_itself():
    crop = (100.0, 200.0, 500.0, 400.0)
    out = remap_crop_box_to_original((0, 0, 1000, 1000), crop, 800, 600)
    for a, b in zip(out, px_to_norm1000(crop, 800, 600)):
        assert abs(a - b) < 1e-9


def test_crop_remap_of_a_full_frame_crop_is_the_identity():
    out = remap_crop_box_to_original((120, 340, 560, 780), (0, 0, 800, 600), 800, 600)
    for a, b in zip(out, [120, 340, 560, 780]):
        assert abs(a - b) < 1e-9
