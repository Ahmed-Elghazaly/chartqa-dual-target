"""Properties of the metrics, over seeded random boxes and answers.

These functions produce every number the project will report. A metric that is subtly wrong
does not fail anything — it publishes a result, and the result is wrong in a direction nobody
can see. `relaxed_correctness` is byte-faithful to the official ChartQA implementation and
must stay that way; the geometry functions are ours and must obey the mathematics they claim.

Written as properties because a metric tested on chosen examples is tested where its author
already believed it worked.
"""
from __future__ import annotations

import math
import random

import pytest

from chartqa_dt.eval.metrics import (
    average_precision_coco,
    bootstrap_ci,
    exact_match,
    f1_of_boxes,
    grounding_is_perfect,
    iou,
    p_at_f1,
    relaxed_correctness,
    to_float,
)


def a_box(rng, span=1000):
    x1, y1 = rng.uniform(0, span - 2), rng.uniform(0, span - 2)
    return [x1, y1, rng.uniform(x1 + 1, span), rng.uniform(y1 + 1, span)]


# ============================================================================== IoU


@pytest.mark.parametrize("seed", range(15))
def test_iou_is_bounded_symmetric_and_one_on_itself(seed):
    rng = random.Random(seed)
    for _ in range(200):
        a, b = a_box(rng), a_box(rng)
        got = iou(a, b)
        assert 0.0 <= got <= 1.0 + 1e-9, got
        assert math.isclose(got, iou(b, a), abs_tol=1e-12), "IoU must be symmetric"
        assert math.isclose(iou(a, a), 1.0, abs_tol=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_disjoint_boxes_have_zero_overlap(seed):
    rng = random.Random(100 + seed)
    for _ in range(200):
        a = a_box(rng, 400)
        b = [a[2] + rng.uniform(1, 50), a[1], a[2] + rng.uniform(51, 100), a[3]]
        assert iou(a, b) == 0.0


def test_a_contained_box_scores_the_ratio_of_the_areas():
    outer, inner = [0, 0, 100, 100], [25, 25, 75, 75]
    assert math.isclose(iou(outer, inner), (50 * 50) / (100 * 100), abs_tol=1e-9)


@pytest.mark.parametrize("box", [[0, 0, 0, 10], [0, 0, 10, 0], [5, 5, 5, 5]])
def test_a_degenerate_box_never_produces_a_score(box):
    """A zero-area box would otherwise divide by zero or claim a perfect match."""
    assert iou(box, [0, 0, 10, 10]) == 0.0
    assert iou(box, box) in (0.0, 1.0)


# =============================================================== average precision


@pytest.mark.parametrize("seed", range(12))
def test_average_precision_is_bounded(seed):
    rng = random.Random(200 + seed)
    for _ in range(60):
        gts = {f"i{i}": [a_box(rng) for _ in range(rng.randint(0, 3))] for i in range(4)}
        preds = [(k, 1.0, a_box(rng)) for k in gts for _ in range(rng.randint(0, 3))]
        got = average_precision_coco(preds, gts)
        assert 0.0 <= got <= 1.0 + 1e-9, got


def test_perfect_predictions_score_one():
    gts = {"a": [[0, 0, 10, 10]], "b": [[5, 5, 20, 20]]}
    preds = [(k, 1.0, list(b)) for k, boxes in gts.items() for b in boxes]
    assert math.isclose(average_precision_coco(preds, gts), 1.0, abs_tol=1e-9)


def test_a_spurious_box_costs_nothing_after_the_true_one_and_everything_before_it():
    """COCO's precision envelope is computed right-to-left, so a false positive ranked
    *after* recall has already reached 1.0 cannot lower it — and one ranked *before* a true
    positive lowers precision at the recall level that matters.

    Our predictions are all tied at score 1.0, so their **emitted order** is what decides
    this. That is the fragility behind `DECISIONS.md` 0014's "few boxes, best first": the
    ordering half is what the metric rewards.
    """
    gts = {"a": [[0, 0, 10, 10]]}
    true, false = [0, 0, 10, 10], [500, 500, 600, 600]
    assert average_precision_coco([("a", 1.0, true)], gts) == pytest.approx(1.0)
    assert average_precision_coco(
        [("a", 1.0, true), ("a", 1.0, false)], gts) == pytest.approx(1.0), "trailing: free"
    assert average_precision_coco(
        [("a", 1.0, false), ("a", 1.0, true)], gts) < 1.0, "leading: not free"


def test_a_missed_ground_truth_always_costs():
    """Recall is the half no ordering can rescue."""
    gts = {"a": [[0, 0, 10, 10], [50, 50, 60, 60]]}
    assert average_precision_coco([("a", 1.0, [0, 0, 10, 10])], gts) < 1.0


def test_predicting_nothing_scores_zero_not_one():
    assert average_precision_coco([], {"a": [[0, 0, 10, 10]]}) == 0.0


def test_an_image_with_no_ground_truth_cannot_be_scored_perfectly_by_guessing():
    rng = random.Random(0)
    gts = {"a": [[0, 0, 10, 10]], "b": []}
    got = average_precision_coco([("b", 1.0, a_box(rng))], gts)
    assert got == 0.0


# ================================================================== P@F1 and F1


@pytest.mark.parametrize("seed", range(12))
def test_f1_is_bounded_and_one_only_on_an_exact_cover(seed):
    rng = random.Random(300 + seed)
    for _ in range(120):
        gt = [a_box(rng) for _ in range(rng.randint(1, 3))]
        pred = [list(b) for b in gt] if rng.random() > 0.5 else [a_box(rng)]
        got = f1_of_boxes(pred, gt)
        assert 0.0 <= got <= 1.0 + 1e-9
        if pred == [list(b) for b in gt]:
            assert math.isclose(got, 1.0, abs_tol=1e-9)


def test_p_at_f1_follows_the_official_predicate_not_an_actual_f1():
    """The official helper says *"F_1 score = 1.0"* in its docstring and computes something
    else — COCO AP on the single image, tested for `== 1.0`. The rule that falls out is:
    **every ground truth matched, and every false positive after every true positive.**

    Each row here is characterised against the vendored evaluator. They are surprising, and
    a test written from the name rather than the behaviour asserts the opposite.
    """
    gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
    spurious, other = [900, 900, 950, 950], [700, 700, 750, 750]

    assert grounding_is_perfect([list(b) for b in gt], gt), "exactly right"
    assert grounding_is_perfect([*gt, spurious], gt), "a TRAILING false positive is free"
    assert grounding_is_perfect([*gt, spurious, other], gt), "two of them are free too"
    assert not grounding_is_perfect([spurious, *gt], gt), "a LEADING one is not"
    assert not grounding_is_perfect([gt[0], spurious, gt[1]], gt), "nor one in the middle"
    assert not grounding_is_perfect([gt[0]], gt), "and a missed ground truth never is"
    assert not grounding_is_perfect([], gt)
    assert not grounding_is_perfect([list(b) for b in gt], [])


def test_p_at_f1_averages_the_predicate_over_samples():
    gt = [[0, 0, 10, 10]]
    assert p_at_f1([([list(b) for b in gt], gt), ([[900, 900, 950, 950]], gt)]) == \
        pytest.approx(0.5)


# ================================================= the official answer metric, unchanged


@pytest.mark.parametrize("target,prediction,expected", [
    ("245", "245", True),
    ("245", "245.0", True),
    ("245", "260", False),   # 6.1% apart, outside the 5% margin
    ("100", "104", True),       # inside 5%
    ("100", "106", False),      # outside it
    ("Nigeria", "Nigeria", True),
    ("Nigeria", "nigeria", True),
    ("Nigeria", "Egypt", False),
    ("81.9%", "0.819", True),   # the official parser divides a trailing percent
    ("0", "0.0", False),        # the truthiness quirk, faithful to upstream
])
def test_relaxed_correctness_matches_the_official_behaviour(target, prediction, expected):
    assert relaxed_correctness(target, prediction) is expected


@pytest.mark.parametrize("seed", range(10))
def test_relaxed_correctness_never_raises_on_arbitrary_text(seed):
    """It is called on whatever a model produced. A crash here loses a whole evaluation."""
    rng = random.Random(400 + seed)
    junk = ["", " ", "nan", "inf", "-", "%", "1e400", "٣", "None", "[]", "{}", "1,2,3"]
    for _ in range(300):
        a = rng.choice(junk) if rng.random() < 0.5 else str(rng.uniform(-1e6, 1e6))
        b = rng.choice(junk) if rng.random() < 0.5 else str(rng.uniform(-1e6, 1e6))
        assert isinstance(relaxed_correctness(a, b), bool)
        assert isinstance(exact_match(a, b), bool)


@pytest.mark.parametrize("seed", range(8))
def test_a_prediction_equal_to_the_target_is_always_correct(seed):
    rng = random.Random(500 + seed)
    for _ in range(200):
        text = rng.choice([str(rng.randint(-999, 999)), f"{rng.uniform(0, 100):.2f}",
                           "Nigeria", "2019", "Yes"])
        assert relaxed_correctness(text, text)


def test_to_float_is_the_official_parser_and_stays_narrow():
    """It must NOT be improved: a more generous parser makes our numbers incomparable with
    the literature while looking better (`DECISIONS.md` 0045)."""
    assert to_float("81.9%") == pytest.approx(0.819)
    assert to_float("1,234") is None, "the official parser does not strip commas"
    assert to_float("1 234") is None, "nor spaces"
    assert to_float("abc") is None


# ================================================================== intervals


@pytest.mark.parametrize("seed", range(10))
def test_a_bootstrap_interval_contains_its_own_mean(seed):
    rng = random.Random(600 + seed)
    scores = [rng.choice([0.0, 1.0]) for _ in range(rng.randint(20, 200))]
    got = bootstrap_ci(scores, n_resamples=400, seed=seed)
    mean = sum(scores) / len(scores)
    assert got.lo <= got.mean <= got.hi
    assert got.mean == pytest.approx(mean)
    assert 0.0 <= got.lo <= got.hi <= 1.0
    assert got.n == len(scores)


def test_a_bootstrap_interval_is_reproducible_from_its_seed():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert bootstrap_ci(scores, n_resamples=300, seed=7) == \
        bootstrap_ci(scores, n_resamples=300, seed=7)


def test_a_unanimous_sample_has_a_degenerate_interval():
    got = bootstrap_ci([1.0] * 50, n_resamples=300, seed=0)
    assert got.lo == got.hi == got.mean == 1.0


def test_an_empty_sample_is_a_zero_interval_rather_than_a_crash():
    """An evaluation that produced no scorable items must report nothing, not divide by
    zero half an hour into a run."""
    got = bootstrap_ci([])
    assert (got.mean, got.lo, got.hi, got.n) == (0.0, 0.0, 0.0, 0)


def test_an_interval_prints_its_uncertainty_not_just_its_point():
    """`Cell` refuses a point estimate without an interval; the string form is what a table
    actually shows, so it has to carry both."""
    got = bootstrap_ci([1.0, 0.0, 1.0, 1.0], n_resamples=200, seed=0)
    assert "[" in str(got) and "n=4" in str(got)
    assert "%" in got.as_percent
