"""The evaluator regression suite — `PLAN.md` 4.3, every case in the table.

The plan names twelve cases and, for four of them, says only "define and test the
behaviour explicitly". Those four are the interesting ones: there is no single right
answer, and the failure mode is not getting them wrong but *never deciding*, so that the
behaviour silently changes when someone edits a helper. Each is decided here, in a test,
with the reasoning attached.

None of this is the scorer of record — `DECISIONS.md` 0003 gives that to the vendored
official evaluator, and `scripts/crosscheck_evaluators.py` (`PLAN.md` 4.2) is what proves
these agree with it.
"""

from __future__ import annotations

import pytest

from chartqa_dt.eval.metrics import (
    average_precision_coco,
    bootstrap_ci,
    exact_match,
    f1_of_boxes,
    grounding_is_perfect,
    iou,
    normalise_prediction,
    p_at_f1,
    relaxed_correctness,
    to_float,
)

UNIT = (0.0, 0.0, 10.0, 10.0)


# ------------------------------------------------ the numeric tolerance, and its edges


@pytest.mark.parametrize(("target", "pred", "expected"), [
    ("10", "10.4", True),      # within 5%
    ("10", "10.6", False),     # 6% away
    ("10", "10.5", True),      # exactly 5% — the boundary is inclusive (<=)
    ("10", "9.5", True),
    ("10", "9.4", False),
    ("100", "105", True),
    ("100", "105.1", False),
])
def test_the_five_percent_tolerance(target, pred, expected):
    assert relaxed_correctness(target, pred) is expected


def test_zero_target_never_takes_the_numeric_path():
    """`PLAN.md` 4.3's zero case, resolved the way the official metric resolves it.

    The official guard is `if prediction_float is not None and target_float` — a
    *truthiness* test — so a gold "0" is falsy and the comparison falls through to string
    equality. The plan's table asks for "0" vs "0" correct and "0" vs "0.1" incorrect, and
    that follows. So does something the plan does not mention: "0" vs "0.0" is **wrong**,
    because the strings differ (`DECISIONS.md` 0015).

    Appendix D's explicit `t == 0` guard would call "0" vs "0.0" correct. `PLAN.md` 4.2
    settles it — the official wins — and `DECISIONS.md` 0052 records the change.
    """
    assert relaxed_correctness("0", "0") is True
    assert relaxed_correctness("0", "0.1") is False
    assert relaxed_correctness("0", "0.0") is False, "string branch: '0.0' != '0'"
    assert relaxed_correctness("0", "-0") is False
    assert relaxed_correctness("0.0", "0.0") is True


# -------------------------------------------------- cases the plan says to DEFINE


def test_percent_against_decimal_is_correct_because_both_sides_convert():
    """`target="50%"`, `pred="0.5"` → **correct**. Decided, not inherited.

    `to_float` divides by 100 whenever a string ends in `%`, and it does so on *both*
    arguments. So "50%" and "0.5" both become 0.5 and agree.

    The alternative — converting only the target — would make the metric asymmetric:
    "50%" vs "0.5" would differ from "0.5" vs "50%". A metric that depends on which
    argument a value arrives in is worse than either consistent choice.
    """
    assert relaxed_correctness("50%", "0.5") is True
    assert relaxed_correctness("0.5", "50%") is True, "the metric must be symmetric here"
    assert relaxed_correctness("50%", "50") is False, "50 is not 0.5"
    assert relaxed_correctness("50%", "51%") is True, "still within 5%"
    assert relaxed_correctness("50%", "60%") is False


def test_a_trailing_period_makes_a_non_numeric_answer_incorrect():
    """`target="Yes"`, `pred="Yes."` → **incorrect**. Decided.

    Non-numeric answers compare exactly after `.strip().lower()`, and stripping does not
    remove interior or trailing punctuation. "Yes." therefore fails.

    This is deliberate rather than an oversight. The official ChartQA metric does exactly
    this, and the number it produces is what published results mean; adding punctuation
    tolerance would make our score incomparable with every number the project is measured
    against. The right place to normalise a trailing period is the model's output
    contract, not the metric.
    """
    assert relaxed_correctness("Yes", "yes") is True, "case-insensitive"
    assert relaxed_correctness("Yes", "YES") is True
    assert relaxed_correctness("Yes", "Yes.") is False
    assert relaxed_correctness("Yes", "Yes!") is False
    # The official metric does NOT strip whitespace either. Normalising is the pipeline's
    # job, in one visible place, rather than a quiet loosening of the shared metric.
    assert relaxed_correctness("Yes", " Yes ") is False
    assert relaxed_correctness("Yes", normalise_prediction(" Yes\n")) is True


def test_a_list_answer_is_compared_as_one_string():
    """List-valued answers → **compared whole, after strip and lower**. Decided.

    ChartQA gold answers that look like lists are stored as one string, and the official
    metric never splits them: `_to_float` fails on "A, B", so it falls to a full-string
    comparison. Splitting on commas here would score answers the official evaluator scores
    as wrong, and inflate every reported number relative to the literature.

    Consequence, asserted below so it is not a surprise later: order matters, and
    whitespace after the separator matters.
    """
    assert relaxed_correctness("Yes, No", "yes, no") is True
    assert relaxed_correctness("Yes, No", "No, Yes") is False, "order is significant"
    assert relaxed_correctness("Yes, No", "Yes,No") is False, "spacing is significant"
    assert to_float("Yes, No") is None


def test_the_unanswerable_representation():
    """Unanswerable → the **empty string** in `model_answer`, with `answerable: false`.

    The schema carries answerability as its own boolean field (`OUTPUT_SCHEMA`), so the
    answer string does not have to encode it — and must not, because a sentinel like
    "unanswerable" would be scored as a literal answer by a metric that knows nothing
    about the convention. An empty prediction is wrong against any real gold answer, which
    is the behaviour we want, and right against an empty gold answer.
    """
    assert relaxed_correctness("", "") is True
    assert relaxed_correctness("42", "") is False
    assert relaxed_correctness("", "42") is False
    assert to_float("") is None


def test_thousands_separators_are_not_stripped():
    """Another Appendix D divergence resolved in the official's favour.

    Appendix D strips commas, so it reads "1,234" as 1234.0. The canonical implementation
    (`pix2struct/metrics.py`, vendored verbatim by RefChartQA) calls plain `float(text)`,
    which raises on the comma — so "1,234" is a *string* and only matches another "1,234".

    Being more generous here would score answers the official evaluator scores as wrong
    and inflate every number relative to the literature.
    """
    assert to_float("1,234") is None
    assert relaxed_correctness("1,234", "1234") is False
    assert relaxed_correctness("1,234", "1,234") is True


# --------------------------------------------------------------------- IoU and boxes


def test_the_iou_threshold_boundary_is_inclusive():
    """`PLAN.md` 4.3: define whether IoU exactly 0.5 counts. It **does** — `>=`.

    This follows the official evaluator, which uses torchmetrics with
    `iou_thresholds=[0.5]`, and COCO convention: a detection matches at
    `IoU >= threshold`. Constructed exactly rather than approximately: two unit-area boxes
    overlapping on a third of their union give IoU 1/3, so the pair below is built to land
    on 0.5 to machine precision.
    """
    a = (0.0, 0.0, 1.0, 1.0)
    b = (0.0, 0.0, 1.0, 0.5)   # nested: intersection 0.5, union 1.0 -> IoU exactly 0.5
    assert iou(a, b) == pytest.approx(0.5)
    gt = {"img": [a]}
    assert average_precision_coco([("img", 1.0, b)], gt, 0.5) == pytest.approx(1.0)
    assert average_precision_coco([("img", 1.0, b)], gt, 0.5 + 1e-9) == 0.0


def test_an_empty_prediction_set_scores_zero_without_crashing():
    gt = {"img": [UNIT]}
    assert average_precision_coco([], gt, 0.5) == 0.0
    assert grounding_is_perfect([], [UNIT]) is False
    assert f1_of_boxes([], [UNIT]) == 0.0
    assert p_at_f1([([], [UNIT])]) == 0.0
    assert p_at_f1([]) == 0.0
    assert bootstrap_ci([]).mean == 0.0


def test_no_ground_truth_scores_zero_rather_than_dividing_by_zero():
    assert average_precision_coco([("img", 1.0, UNIT)], {}, 0.5) == 0.0
    assert grounding_is_perfect([UNIT], []) is False
    assert f1_of_boxes([UNIT], []) == 0.0


def test_degenerate_boxes_have_zero_iou_not_nan():
    point = (5.0, 5.0, 5.0, 5.0)
    inverted = (10.0, 10.0, 0.0, 0.0)
    assert iou(point, UNIT) == 0.0
    assert iou(inverted, UNIT) == 0.0
    assert iou(point, point) == 0.0


def test_out_of_bounds_boxes_are_clamped_consistently():
    """`PLAN.md` 4.3: rejected or clamped, **consistently**.

    Clamped, and in exactly one place: `clamp_for_official_evaluator`
    (`DECISIONS.md` 0004), which follows the official evaluator's own
    `ensure_xyxy_bbox_within_bounds` — clamp to [0, 999] — rather than dropping the box.
    Dropping would change the *count* of predictions and therefore the precision
    denominator, which silently flatters a model that emits nonsense.
    """
    from chartqa_dt.vision.coords import OFFICIAL_MAX_COORD, clamp_for_official_evaluator

    assert clamp_for_official_evaluator((-5.0, -5.0, 2000.0, 2000.0)) == \
        [0, 0, OFFICIAL_MAX_COORD, OFFICIAL_MAX_COORD]
    assert clamp_for_official_evaluator((10.0, 20.0, 30.0, 40.0)) == [10, 20, 30, 40]
    assert all(0 <= v <= OFFICIAL_MAX_COORD
               for v in clamp_for_official_evaluator((-1.0, 1500.0, 999.4, 1000.0)))


# ---------------------------------------------------------------------- AP behaviour


def test_an_extra_box_is_free_on_one_image_and_costly_across_a_dataset():
    """The measurement behind `DECISIONS.md` 0014's "few boxes, best first".

    On a *single* image a false positive ranked after the true positives costs nothing —
    recall has already reached 1.0 and the precision envelope is taken right to left. It
    is tempting to conclude extra boxes are harmless. They are not: across a dataset the
    official evaluator ties every score at 1.0, so one image's spurious box is ranked
    among other images' true positives and drags precision down at every recall level
    below 1.0.

    Measured against `torchmetrics` (the official path): five images with one true box
    each score 1.00; add one spurious box to each and the score is 0.68.
    """
    gt_one = {"img": [UNIT]}
    assert average_precision_coco([("img", 1.0, UNIT)], gt_one) == pytest.approx(1.0)
    assert average_precision_coco(
        [("img", 1.0, UNIT), ("img", 1.0, UNIT)], gt_one) == pytest.approx(1.0), \
        "a trailing duplicate is free within one image"

    spurious = (500.0, 500.0, 600.0, 600.0)
    gts = {f"i{k}": [UNIT] for k in range(5)}
    clean = [(f"i{k}", 1.0, UNIT) for k in range(5)]
    noisy = [x for k in range(5)
             for x in ((f"i{k}", 1.0, UNIT), (f"i{k}", 1.0, spurious))]
    assert average_precision_coco(clean, gts) == pytest.approx(1.0)
    assert average_precision_coco(noisy, gts) == pytest.approx(0.68, abs=0.01), \
        "across a dataset, one spurious box per image is expensive"


def test_ap_is_order_invariant_given_scores():
    gt = {"img": [UNIT, (20.0, 20.0, 30.0, 30.0)]}
    preds = [("img", 0.9, UNIT), ("img", 0.4, (20.0, 20.0, 30.0, 30.0))]
    assert average_precision_coco(preds, gt, 0.5) == \
        pytest.approx(average_precision_coco(list(reversed(preds)), gt, 0.5))


def test_perfect_predictions_score_one_everywhere():
    gt_boxes = [UNIT, (20.0, 20.0, 30.0, 30.0)]
    gt = {"img": gt_boxes}
    preds = [("img", 1.0, b) for b in gt_boxes]
    assert average_precision_coco(preds, gt, 0.5) == pytest.approx(1.0)
    assert grounding_is_perfect(gt_boxes, gt_boxes) is True
    assert p_at_f1([(gt_boxes, gt_boxes)]) == pytest.approx(1.0)


def test_p_at_f1_needs_every_ground_truth_matched():
    gt_boxes = [UNIT, (20.0, 20.0, 30.0, 30.0)]
    assert f1_of_boxes([UNIT], gt_boxes) == pytest.approx(2 / 3)
    assert grounding_is_perfect([UNIT], gt_boxes) is False, "a missed box is fatal"
    assert p_at_f1([([UNIT], gt_boxes)]) == 0.0


def test_a_trailing_false_positive_is_free_but_a_leading_one_is_fatal():
    """The official P@F1 predicate is not an F1 — characterised in `grounding_is_perfect`.

    Measured against the vendored evaluator, which computes COCO AP on the single image
    and tests `map == 1.0`. Because the precision envelope is taken right to left, a false
    positive after recall has already reached 1.0 costs nothing; one before a true
    positive destroys precision at that recall level and cannot be recovered.
    """
    bad = (500.0, 500.0, 600.0, 600.0)
    assert grounding_is_perfect([UNIT, bad], [UNIT]) is True, "trailing FP is free"
    assert grounding_is_perfect([UNIT, bad, (700.0, 700.0, 800.0, 800.0)], [UNIT]) is True
    assert grounding_is_perfect([bad, UNIT], [UNIT]) is False, "leading FP is fatal"
    other = (20.0, 20.0, 30.0, 30.0)
    assert grounding_is_perfect([UNIT, other, bad], [UNIT, other]) is True
    assert grounding_is_perfect([UNIT, bad, other], [UNIT, other]) is False
    # ... and F1 disagrees with all of that, which is why both exist.
    assert f1_of_boxes([UNIT, bad], [UNIT]) == pytest.approx(2 / 3)


def test_p_at_f1_dataset_counts_skipped_items_in_the_denominator():
    """The official evaluator divides by every item, including ones it skips."""
    good = ([UNIT], [UNIT])
    empty = ([], [UNIT])
    assert p_at_f1([good, good]) == pytest.approx(1.0)
    assert p_at_f1([good, empty]) == pytest.approx(0.5)


# ---------------------------------------------------------------------- statistics


def test_bootstrap_interval_brackets_the_mean_and_is_seeded():
    scores = [1.0] * 70 + [0.0] * 30
    a = bootstrap_ci(scores, n_resamples=2000, seed=0)
    b = bootstrap_ci(scores, n_resamples=2000, seed=0)
    assert a.mean == pytest.approx(0.7)
    assert a.lo < a.mean < a.hi
    assert (a.lo, a.hi) == (b.lo, b.hi), "same seed must give the same interval"
    # Continuous scores, because with 0/1 data the 2.5th percentile of the resampled mean
    # lands on the same discrete value for most seeds and the check proves nothing.
    continuous = [i / 97 for i in range(97)]
    assert bootstrap_ci(continuous, n_resamples=2000, seed=1).lo != \
        bootstrap_ci(continuous, n_resamples=2000, seed=0).lo


def test_a_unanimous_score_has_a_degenerate_interval():
    one = bootstrap_ci([1.0] * 50, n_resamples=500)
    assert (one.mean, one.lo, one.hi) == (1.0, 1.0, 1.0)


def test_exact_match_is_stricter_than_relaxed():
    assert exact_match("10", "10") is True
    assert exact_match("10", "10.4") is False
    assert relaxed_correctness("10", "10.4") is True
