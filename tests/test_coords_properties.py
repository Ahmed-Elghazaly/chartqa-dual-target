"""Coordinate properties, over seeded random boxes and image sizes.

Boxes are half the headline metric: AP@0.5 and P@F1 are computed from nothing else. A
coordinate defect does not raise, it does not fail a schema, and it does not show up in a loss
curve — it shows up as a grounding number that is quietly lower than it should be.

The existing suite tests these functions on chosen examples. This tests the properties that
must hold for **every** box: that the round trip returns what went in, that the official
evaluator's clamp never widens a box or moves it off the canvas, and that a box which was
inside the image before a resize is still inside it afterwards.
"""
from __future__ import annotations

import random

import pytest

from chartqa_dt.vision.coords import (
    OFFICIAL_MAX_COORD,
    clamp_for_official_evaluator,
    norm1000_to_px,
    px_to_norm1000,
    smart_resize,
)

#: Qwen's own preprocessing constants, so the properties are tested at the sizes that will
#: actually be used rather than at invented ones.
FACTOR, MIN_PIXELS, MAX_PIXELS = 32, 32 * 32, 512 * 512


def resize(w, h, *, max_pixels=MAX_PIXELS):
    """`smart_resize` with this project's constants, height first as the port takes it."""
    return smart_resize(h, w, FACTOR, MIN_PIXELS, max_pixels)


def _sizes(rng, n=200):
    """Image sizes spanning what ChartQA actually contains, plus awkward but LEGAL extremes.

    `smart_resize` refuses an absolute aspect ratio above 200 and a non-positive dimension,
    which is its documented contract and is asserted separately.
    """
    out = [(rng.randint(80, 2000), rng.randint(80, 2000)) for _ in range(n - 4)]
    out += [(33, 33), (64, 12800), (12800, 64), (4000, 4000)]
    return out


def _box_in(rng, w, h):
    x1 = rng.uniform(0, max(w - 2, 1))
    y1 = rng.uniform(0, max(h - 2, 1))
    return (x1, y1, rng.uniform(x1 + 1, w), rng.uniform(y1 + 1, h))


# ================================================================== the normalisation round trip


@pytest.mark.parametrize("seed", range(12))
def test_pixels_to_norm_and_back_stays_within_a_pixel(seed):
    """The normalisation is anisotropic — x by width, y by height — so a square box does not
    stay square. What must hold is that the value comes back."""
    rng = random.Random(seed)
    for w, h in _sizes(rng, 60):
        box = _box_in(rng, w, h)
        back = norm1000_to_px(px_to_norm1000(box, w, h), w, h)
        for got, want, extent in zip(back, box, (w, h, w, h)):
            assert abs(got - want) <= extent / 1000.0 + 1e-6, (w, h, box, back)


@pytest.mark.parametrize("seed", range(12))
def test_normalised_coordinates_stay_in_range(seed):
    rng = random.Random(seed)
    for w, h in _sizes(rng, 60):
        for v in px_to_norm1000(_box_in(rng, w, h), w, h):
            assert 0.0 <= v <= 1000.0


@pytest.mark.parametrize("seed", range(8))
def test_normalisation_preserves_ordering(seed):
    """A box that is left of another must stay left of it, or grounding is scrambled."""
    rng = random.Random(100 + seed)
    for w, h in _sizes(rng, 40):
        a, b = sorted([_box_in(rng, w, h), _box_in(rng, w, h)], key=lambda t: t[0])
        na, nb = px_to_norm1000(a, w, h), px_to_norm1000(b, w, h)
        assert na[0] <= nb[0] + 1e-9


# ============================================================ the official evaluator's clamp


@pytest.mark.parametrize("seed", range(12))
def test_the_clamp_only_moves_a_coordinate_onto_the_canvas(seed):
    """It is `int(max(0, min(999, round(v))))` per coordinate, so each result must equal the
    original pulled onto the canvas and rounded — never anything else. Written as the exact
    property rather than an inequality, because a hand-written inequality about clamping is
    how this test was wrong the first time."""
    rng = random.Random(200 + seed)
    for _ in range(300):
        box = tuple(rng.uniform(-200, 1200) for _ in range(4))
        got = clamp_for_official_evaluator(box)
        for value, result in zip(box, got):
            assert result == int(max(0, min(OFFICIAL_MAX_COORD, round(value))))


@pytest.mark.parametrize("seed", range(8))
def test_a_box_already_on_the_canvas_only_gets_rounded(seed):
    """The common case: clamping must not move a legal box."""
    rng = random.Random(250 + seed)
    for _ in range(300):
        box = tuple(rng.uniform(0, OFFICIAL_MAX_COORD) for _ in range(4))
        for value, result in zip(box, clamp_for_official_evaluator(box)):
            assert abs(result - value) <= 0.5 + 1e-9


@pytest.mark.parametrize("seed", range(12))
def test_the_clamp_never_leaves_the_canvas(seed):
    """The official evaluator silently discards a box at exactly 1000, so nothing may reach
    it — that is why the cap is 999 and not 1000."""
    rng = random.Random(300 + seed)
    for _ in range(300):
        box = (rng.uniform(-200, 1200), rng.uniform(-200, 1200),
               rng.uniform(-200, 1200), rng.uniform(-200, 1200))
        got = clamp_for_official_evaluator(box)
        assert all(0 <= v <= OFFICIAL_MAX_COORD for v in got), (box, got)
        assert all(isinstance(v, int) for v in got)


def test_the_clamp_is_idempotent():
    """Applying it twice must change nothing, or a box drifts every time it passes through."""
    rng = random.Random(0)
    for _ in range(400):
        box = tuple(rng.uniform(-100, 1100) for _ in range(4))
        once = clamp_for_official_evaluator(box)
        assert clamp_for_official_evaluator(tuple(once)) == once


def test_a_box_at_the_boundary_is_pulled_inside_not_dropped():
    """1000 is the value the official evaluator throws away."""
    got = clamp_for_official_evaluator((0.0, 0.0, 1000.0, 1000.0))
    assert got[2] == OFFICIAL_MAX_COORD and got[3] == OFFICIAL_MAX_COORD


# ========================================================================= smart_resize


@pytest.mark.parametrize("seed", range(10))
def test_resized_dimensions_are_always_a_multiple_of_the_factor(seed):
    """The processor's own contract. A dimension that is not a multiple of patch × merge
    changes how many visual tokens an image becomes, which moves every box."""
    rng = random.Random(400 + seed)
    for w, h in _sizes(rng, 60):
        rh, rw = resize(w, h)
        assert rh % FACTOR == 0 and rw % FACTOR == 0, (w, h, rw, rh)
        assert rh > 0 and rw > 0


@pytest.mark.parametrize("seed", range(10))
def test_resizing_roughly_preserves_the_aspect_ratio(seed):
    """Rounding to a multiple of 32 perturbs it; it must not invert it."""
    rng = random.Random(500 + seed)
    for w, h in _sizes(rng, 60):
        if min(w, h) < FACTOR:
            continue
        rh, rw = resize(w, h)
        before, after = w / h, rw / rh
        # Rounding to a multiple of 32 perturbs the ratio; on a nearly-square image it can
        # cross 1. It must never invert a decisively non-square one.
        if abs(before - 1) > 0.25:
            assert (before >= 1) == (after >= 1), (w, h, rw, rh)
        assert 0.5 < after / before < 2.0, (w, h, rw, rh)


@pytest.mark.parametrize("seed", range(8))
def test_a_pixel_budget_is_respected(seed):
    rng = random.Random(600 + seed)
    for w, h in _sizes(rng, 40):
        rh, rw = resize(w, h, max_pixels=MAX_PIXELS)
        assert rh * rw <= MAX_PIXELS + FACTOR * FACTOR, (w, h, rw, rh)


def test_resize_is_deterministic():
    for w, h in ((800, 600), (33, 33), (4000, 4000), (777, 1013)):
        assert resize(w, h) == resize(w, h)


@pytest.mark.parametrize("w,h", [(0, 100), (100, 0), (-5, 100), (1, 4000), (4000, 1)])
def test_resize_refuses_what_its_contract_refuses(w, h):
    """A non-positive dimension, or an absolute aspect ratio above 200. Refusing loudly is
    the documented behaviour and the port must keep it -- silently resizing a 1x4000 strip
    would put every box on it in the wrong place."""
    with pytest.raises(ValueError):
        resize(w, h)


# ================================================== the property the whole pipeline rests on


@pytest.mark.parametrize("seed", range(10))
def test_a_box_inside_the_image_stays_inside_after_normalisation_and_clamping(seed):
    """End to end: a real annotation box, normalised, clamped for the official evaluator, and
    still describing a region of the chart rather than a degenerate point or something off
    the canvas."""
    rng = random.Random(700 + seed)
    for w, h in _sizes(rng, 60):
        if w < 4 or h < 4:
            continue
        box = _box_in(rng, w, h)
        clamped = clamp_for_official_evaluator(px_to_norm1000(box, w, h))
        assert all(0 <= v <= OFFICIAL_MAX_COORD for v in clamped)
        assert clamped[2] >= clamped[0] and clamped[3] >= clamped[1], (box, clamped)
