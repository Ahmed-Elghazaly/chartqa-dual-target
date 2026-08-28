"""Emitting for the official RefChartQA evaluator — `PLAN.md` 5.4.

5.4 requires the **released evaluator**, not ours. The first draft of the zero-shot stage
scored grounding with our own AP implementation, which agrees with the official to within
0.07 percentage points but is still not the same thing; `DECISIONS.md` 0003 makes the
official code the scorer of record. Caught in a design pass before the GPU run rather
than after it.

These tests exist because the official parser has three quiet failure modes, each of which
scores an item zero for a reason unrelated to its answer.
"""

from __future__ import annotations

import pytest

from chartqa_dt.eval.official_format import (
    GROUNDING_SEPARATOR,
    build_rows,
    format_box,
    format_prediction,
    score_with_official,
)

pytestmark = pytest.mark.official

pytest.importorskip("torch")
pytest.importorskip("torchmetrics")

W, H = 800, 600
GT = [{"x": 100.0, "y": 200.0, "w": 50.0, "h": 40.0}]
EXACT = [[1000 * 100 / W, 1000 * 200 / H, 1000 * 150 / W, 1000 * 240 / H]]


def item(pred_boxes, answer="42", label="42", gt=None):
    return {"pred_boxes": pred_boxes, "answer": answer, "label": label,
            "image_size": (W, H), "grounding_bboxes": gt if gt is not None else GT,
            "question_kind": "human"}


def test_a_perfect_prediction_scores_one_through_the_official_evaluator():
    """The end-to-end check that the emitter speaks the evaluator's language."""
    got = score_with_official(build_rows([item(EXACT)]))
    assert got["accuracy"] == pytest.approx(1.0)
    assert got["AP_50"] == pytest.approx(1.0)
    assert got["P_at_FI"] == pytest.approx(1.0)


def test_boxes_are_emitted_as_clamped_integers():
    """`DECISIONS.md` 0004: the official parser discards anything outside [0, 999].

    Silently — `extract_bounding_boxes` filters on `0 <= v <= bins - 1` and drops the box
    with no error, so an out-of-range coordinate costs a detection rather than raising.
    """
    assert format_box([10.4, 20.6, 30.0, 40.0]) == "<box>10,21,30,40</box>"
    assert format_box([-5.0, 0.0, 1000.0, 2000.0]) == "<box>0,0,999,999</box>"
    for part in format_box([1.7, 2.2, 3.9, 4.1]).strip("<box>/").split(","):
        assert part.isdigit() or part.lstrip("-").isdigit()


def test_an_answer_containing_the_separator_would_break_the_parser():
    """The evaluator splits on `<grounding-sep>` and requires EXACTLY two parts.

    A three-part split scores the item zero for both accuracy and grounding, whatever the
    answer said, so the separator is stripped from the answer before emission.
    """
    text = format_prediction(EXACT, f"42{GROUNDING_SEPARATOR}extra")
    assert text.count(GROUNDING_SEPARATOR) == 1

    got = score_with_official(build_rows([
        item(EXACT, answer=f"42{GROUNDING_SEPARATOR}extra")]))
    # The answer really is wrong — "42 extra" is not "42" — and it scores zero, which is
    # correct. What the strip protects is the *grounding*: a three-part split makes the
    # evaluator discard the boxes too, losing a correct detection to a formatting
    # accident.
    assert got["accuracy"] == pytest.approx(0.0)
    assert got["AP_50"] == pytest.approx(1.0), \
        "the boxes must still be read even when the answer is junk"


def test_no_boxes_still_produces_a_well_formed_string():
    """A record with no evidence must score zero on grounding, not crash the evaluator."""
    text = format_prediction([], "42")
    assert text == f"{GROUNDING_SEPARATOR}42"
    got = score_with_official(build_rows([item([])]))
    assert got["accuracy"] == pytest.approx(1.0), "the answer is still right"
    assert got["AP_50"] == pytest.approx(0.0)
    assert got["P_at_FI"] == pytest.approx(0.0)


def test_a_wrong_box_scores_zero_grounding_and_a_right_answer_still_counts():
    """The two halves are scored independently, which is the point of a dual target."""
    far = [[900.0, 900.0, 950.0, 950.0]]
    got = score_with_official(build_rows([item(far)]))
    assert got["accuracy"] == pytest.approx(1.0)
    assert got["AP_50"] == pytest.approx(0.0)


def test_a_wrong_answer_scores_zero_accuracy_with_perfect_grounding():
    got = score_with_official(build_rows([item(EXACT, answer="99", label="42")]))
    assert got["accuracy"] == pytest.approx(0.0)
    assert got["AP_50"] == pytest.approx(1.0)


def test_ground_truth_boxes_are_passed_through_unconverted():
    """The evaluator quantises `{x, y, w, h}` itself; converting first would double-apply."""
    rows = build_rows([item(EXACT)])
    assert rows[0]["grounding_bboxes"] is GT
    assert rows[0]["width"] == W and rows[0]["height"] == H


def test_extra_boxes_cost_grounding_but_not_the_answer():
    """`DECISIONS.md` 0014, measured through the official evaluator this time."""
    spurious = [*EXACT, [900.0, 900.0, 950.0, 950.0]]
    clean = score_with_official(build_rows([item(EXACT)] * 4))
    noisy = score_with_official(build_rows([item(spurious)] * 4))
    assert clean["AP_50"] > noisy["AP_50"], "a spurious box must cost AP"
    assert noisy["accuracy"] == pytest.approx(clean["accuracy"])
