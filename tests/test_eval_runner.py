"""The evaluation runner and stratified reporting — `PLAN.md` 4.5 and 4.6.

The property worth guarding hardest is the one that was wrong first: **a perfect
prediction set must score 100% in every stratum**, not only overall. The first
implementation restricted ground truths to a bucket but scored every prediction against
them, so a prediction matching a large target became a false positive in the small-target
bucket — perfect predictions reported 78% and 94% while the overall score was 100%. That
is the shape of a bug that gets reported as a finding.
"""

from __future__ import annotations

import json

import pytest

from chartqa_dt.eval.runner import (
    EvalResult,
    evaluate_predictions,
    score_item,
    write_results,
)
from chartqa_dt.eval.stratified import (
    box_area_in_tokens,
    is_subtoken_by_area,
    is_subtoken_by_axis,
    stratify,
)

BIG = [100.0, 100.0, 300.0, 300.0]        # ~16 tokens at 512 px
SMALL = [500.0, 500.0, 520.0, 520.0]      # sub-token at 512 px
OTHER = [600.0, 600.0, 800.0, 800.0]
SIZE = (512.0, 512.0)


# ------------------------------------------------------------------- stratification


def _item(pred, gt):
    return {"pred_boxes": pred, "gt_boxes": gt, "resized_size": SIZE}


def test_perfect_predictions_score_one_in_every_bucket():
    """The regression this module exists for."""
    items = [_item([BIG, SMALL], [BIG, SMALL]) for _ in range(20)]
    report = stratify(items)
    assert report.overall_ap50 == pytest.approx(1.0)
    for bucket in report.buckets:
        assert bucket.n_targets > 0, f"{bucket.name} is empty; the split proves nothing"
        assert bucket.ap50 == pytest.approx(1.0), \
            f"{bucket.name} scored {bucket.ap50} on perfect predictions"
        assert bucket.p_at_f1 == pytest.approx(1.0)


def test_a_target_outside_a_bucket_is_ignored_not_missed():
    """COCO area-range semantics: out-of-range targets neither help nor hurt."""
    items = [_item([BIG], [BIG, SMALL]) for _ in range(10)]
    report = stratify(items)
    big_bucket = next(b for b in report.buckets if b.name != "sub-token")
    sub_bucket = next(b for b in report.buckets if b.name == "sub-token")
    assert big_bucket.ap50 == pytest.approx(1.0), \
        "the large target was found; the missed small one must not count against it"
    assert sub_bucket.ap50 == pytest.approx(0.0), "the small target was genuinely missed"


def test_the_two_subtoken_definitions_are_different_and_both_kept():
    """Measured on real data: 24.8% by area, 66.7% by axis. Not interchangeable."""
    sliver = [0.0, 0.0, 8.0, 500.0]        # 4 x 256 px at 512: two tokens of area
    assert box_area_in_tokens(sliver, *SIZE, *SIZE, 32.0) > 1.0
    assert is_subtoken_by_area(sliver, *SIZE, 32.0) is False
    assert is_subtoken_by_axis(sliver, *SIZE, 32.0) is True, \
        "a sliver has the area of two tokens and still cannot be localised across it"


def test_the_bucket_boundary_is_one_visual_token_of_area():
    exactly_one = [0.0, 0.0, 1000.0 * 32.0 / 512.0, 1000.0 * 32.0 / 512.0]
    assert box_area_in_tokens(exactly_one, *SIZE, *SIZE, 32.0) == pytest.approx(1.0)
    assert is_subtoken_by_area(exactly_one, *SIZE, 32.0) is False, "boundary is >= 1"


def test_stratify_handles_empty_and_boxless_items():
    report = stratify([])
    assert report.overall_ap50 == 0.0 and report.subtoken_fraction == 0.0
    report = stratify([_item([], []), _item([], [BIG])])
    assert report.overall_ap50 == 0.0


# --------------------------------------------------------------------- the runner


def test_scoring_normalises_the_prediction_but_not_the_metric():
    item = score_item("x", "Yes", " Yes\n")
    assert item.correct is True and item.exact is True
    assert item.prediction == "Yes", "normalisation happens once, visibly, at scoring"


def test_headline_numbers_carry_intervals():
    items = [score_item(f"i{k}", "10", "10.1" if k < 7 else "99",
                        pred_boxes=[BIG], gt_boxes=[BIG]) for k in range(10)]
    result = evaluate_predictions(items, seeds=[0, 1, 2])
    assert result.n_items == 10
    assert result.relaxed_accuracy.mean == pytest.approx(0.7)
    assert result.relaxed_accuracy.lo < result.relaxed_accuracy.mean < \
        result.relaxed_accuracy.hi
    assert result.ap50.mean == pytest.approx(1.0)
    assert result.seeds == [0, 1, 2]


def test_the_seed_spread_is_reported_rather_than_hidden():
    """`PLAN.md` 4.6 asks for three seeds. On a fixed prediction file the point estimates
    cannot move, so what varies is the bootstrap — and reporting that spread is what makes
    the interval's precision honest rather than decorative.
    """
    items = [score_item(f"i{k}", "10", "10.1" if k % 3 else "99",
                        pred_boxes=[BIG], gt_boxes=[BIG]) for k in range(60)]
    result = evaluate_predictions(items, seeds=[0, 1, 2], ap_resamples=50)
    assert set(result.seed_spread) == {"relaxed_accuracy", "ap50_ci_width"}
    assert all(v >= 0.0 for v in result.seed_spread.values())


def test_the_same_seed_gives_the_same_intervals():
    items = [score_item(f"i{k}", "10", "10.1" if k % 2 else "99") for k in range(40)]
    a = evaluate_predictions(items, seeds=[7], ap_resamples=20)
    b = evaluate_predictions(items, seeds=[7], ap_resamples=20)
    assert (a.relaxed_accuracy.lo, a.relaxed_accuracy.hi) == \
        (b.relaxed_accuracy.lo, b.relaxed_accuracy.hi)


def test_ap_is_bootstrapped_by_resampling_items_not_by_averaging():
    """AP has no per-item value to average; the interval comes from recomputing it."""
    items = [score_item(f"i{k}", "1", "1", pred_boxes=[BIG] if k % 2 else [OTHER],
                        gt_boxes=[BIG]) for k in range(40)]
    result = evaluate_predictions(items, seeds=[0], ap_resamples=80)
    assert 0.0 < result.ap50.mean < 1.0
    assert result.ap50.lo < result.ap50.hi, "a mixed set must have a non-degenerate CI"


def test_per_subset_numbers_are_reported():
    items = ([score_item(f"h{k}", "1", "1", pred_boxes=[BIG], gt_boxes=[BIG],
                         subset="human") for k in range(5)]
             + [score_item(f"m{k}", "1", "2", pred_boxes=[OTHER], gt_boxes=[BIG],
                           subset="machine") for k in range(5)])
    result = evaluate_predictions(items, seeds=[0], ap_resamples=20)
    assert result.by_subset["human"]["relaxed_accuracy"] == pytest.approx(1.0)
    assert result.by_subset["machine"]["relaxed_accuracy"] == pytest.approx(0.0)
    assert result.by_subset["human"]["ap50"] == pytest.approx(1.0)
    assert result.by_subset["machine"]["ap50"] == pytest.approx(0.0)


def test_an_empty_prediction_set_does_not_crash():
    result = evaluate_predictions([], seeds=[0])
    assert result.n_items == 0 and result.ap50.mean == 0.0
    assert isinstance(result, EvalResult)


def test_results_json_is_structured_and_complete(tmp_path):
    """The Phase 4 acceptance criterion: a structured results JSON."""
    items = [score_item(f"i{k}", "1", "1", pred_boxes=[BIG], gt_boxes=[BIG])
             for k in range(6)]
    result = evaluate_predictions(items, seeds=[0], ap_resamples=20)
    report = stratify([_item([BIG], [BIG])] * 6)
    path = write_results(result, tmp_path / "results.json",
                         meta={"source": "test"}, stratified=report.to_dict())
    data = json.loads(path.read_text())
    assert data["meta"]["source"] == "test"
    for key in ("relaxed_accuracy", "exact_match", "p_at_f1", "ap50"):
        assert set(data["results"][key]) == {"mean", "ci_lo", "ci_hi", "n"}
    assert data["stratified"]["subtoken_fraction"] == 0.0
    assert "subtoken_fraction_by_axis" in data["stratified"]


def test_the_dev_fixture_runs_end_to_end():
    """`cdt-eval --dev` must work with no model and no network."""
    from chartqa_dt.cli.evaluate import _dev_rows

    rows = _dev_rows()
    items = [score_item(r["id"], r["gold"], r["prediction"],
                        pred_boxes=r.get("pred_boxes"), gt_boxes=r.get("gt_boxes"),
                        subset=r.get("subset", "")) for r in rows]
    result = evaluate_predictions(items, seeds=[0, 1, 2], ap_resamples=50)
    assert result.n_items == len(rows)
    assert 0.0 < result.relaxed_accuracy.mean < 1.0, "the fixture must exercise both"
    assert set(result.by_subset) == {"human", "machine", "pot"}


# --- `PLAN.md` 9.2's other two axes: chart type and question kind ------------------

CATEGORICAL_ITEMS = [
    {"id": "a", "chart_type": "v_bar", "kind": "human",
     "pred_boxes": [[10, 10, 50, 50]], "gt_boxes": [[10, 10, 50, 50]], "correct": True},
    {"id": "b", "chart_type": "v_bar", "kind": "machine",
     "pred_boxes": [[0, 0, 5, 5]], "gt_boxes": [[90, 90, 99, 99]], "correct": False},
    {"id": "c", "chart_type": "pie", "kind": "human",
     "pred_boxes": [[10, 10, 50, 50]], "gt_boxes": [[10, 10, 50, 50]], "correct": True},
]


def test_stratify_by_scores_each_group_independently():
    from chartqa_dt.eval.stratified import stratify_by

    out = stratify_by(CATEGORICAL_ITEMS, "chart_type")
    assert out["pie"]["ap50"] == 1.0
    assert out["pie"]["n"] == 1 and out["v_bar"]["n"] == 2


def test_stratify_by_regroups_the_same_items_on_a_different_field():
    from chartqa_dt.eval.stratified import stratify_by

    out = stratify_by(CATEGORICAL_ITEMS, "kind")
    assert set(out) == {"human", "machine"}
    assert out["human"]["n"] == 2 and out["human"]["ap50"] == 1.0


def test_stratify_by_keeps_items_missing_the_field_as_unknown():
    """A stratification that quietly loses records misstates its own population."""
    from chartqa_dt.eval.stratified import stratify_by

    out = stratify_by([*CATEGORICAL_ITEMS, {"id": "d", "pred_boxes": [], "gt_boxes": []}],
                      "chart_type")
    assert out["unknown"]["n"] == 1
    assert sum(g["n"] for g in out.values()) == 4


def test_stratify_by_excludes_boxless_records_from_p_at_f1_but_not_accuracy():
    from chartqa_dt.eval.stratified import stratify_by

    out = stratify_by([{"id": "x", "chart_type": "pie", "pred_boxes": [],
                        "gt_boxes": [], "correct": True}], "chart_type")
    assert out["pie"]["n"] == 1 and out["pie"]["n_with_boxes"] == 0
    assert out["pie"]["p_at_f1"] == 0.0 and out["pie"]["accuracy"] == 1.0


def test_stratify_by_recomputes_ap_per_group_rather_than_averaging():
    """AP is not a mean of per-item scores; averaging group APs would silently produce a
    different quantity from the one the metric names."""
    from chartqa_dt.eval.stratified import stratify_by

    out = stratify_by(CATEGORICAL_ITEMS, "chart_type")
    naive_mean = (out["pie"]["ap50"] + out["v_bar"]["ap50"]) / 2
    whole = stratify_by([{**i, "chart_type": "all"} for i in CATEGORICAL_ITEMS],
                        "chart_type")["all"]["ap50"]
    assert whole != naive_mean
