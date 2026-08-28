"""Phase 9's analyses composed, not just tested in isolation.

Each module has its own tests. This one runs a single prediction set through all of them,
because the defect that survives per-module tests is the interface between them — a key one
module writes and another does not read, which is exactly the shape of `DECISIONS.md` 0071.
"""

from __future__ import annotations

import json

import pytest

from chartqa_dt.eval.calibration import assert_headline_is_full_coverage, calibrate
from chartqa_dt.eval.calibrator import evaluate as evaluate_calibrator
from chartqa_dt.eval.calibrator import fit
from chartqa_dt.eval.crop import run_crop_pass
from chartqa_dt.eval.diagnostics import diagnose
from chartqa_dt.eval.figures import coverage_report, select_figures
from chartqa_dt.eval.oracle import OracleItem, decompose
from chartqa_dt.eval.stratified import stratify_by

GOLD_EVIDENCE = [{"label": "2019", "value": 245.0, "unit": "units"},
                 {"label": "2018", "value": 210.0, "unit": "units"}]
GOLD_PLAN = {"op": "difference", "args": ["2019", "2018"]}


def _raw(plan: dict, answer: str, evidence=None) -> str:
    ev = evidence or [{**e, "bbox": [100, 100, 300, 300]} for e in GOLD_EVIDENCE]
    return json.dumps({"answerable": True, "evidence": ev, "plan": plan,
                       "model_answer": answer}, separators=(",", ":"))


@pytest.fixture
def predictions() -> list[dict]:
    """A small mixed set: correct, visually wrong, reasoning-wrong, and unparseable."""
    wrong_values = [{"label": "2019", "value": 100.0, "unit": "units",
                     "bbox": [100, 100, 300, 300]},
                    {"label": "2018", "value": 90.0, "unit": "units",
                     "bbox": [400, 400, 600, 600]}]
    items = []
    for i in range(12):
        kind = i % 4
        if kind == 0:                              # correct
            raw, correct, source = _raw(GOLD_PLAN, "35"), True, "synthetic"
        elif kind == 1:                            # misread values
            raw, correct, source = _raw(GOLD_PLAN, "10", wrong_values), False, "synthetic"
        elif kind == 2:                            # wrong operation
            raw = _raw({"op": "sum", "args": ["2019", "2018"]}, "455")
            correct, source = False, "chartqa"
        else:                                      # unparseable
            raw, correct, source = "I think it is 35", False, "chartqa"
        items.append({
            "id": f"r{i}", "raw": raw, "source": source, "correct": correct,
            "gold": "35", "gold_plan": GOLD_PLAN, "gold_evidence": GOLD_EVIDENCE,
            "chart_type": "v_bar" if i % 2 else "pie",
            "kind": "human" if i % 3 else "machine",
            "pred_boxes": [[100, 100, 300, 300]] if kind != 3 else [],
            "gt_boxes": [[100, 100, 300, 300]],
            "image_size": (800, 600),
        })
    return items


def test_diagnostics_and_transfer_run_over_the_same_set(predictions) -> None:
    report = diagnose(predictions)
    assert report["overall"]["n"] == 12
    assert set(report["by_source"]) == {"synthetic", "chartqa"}
    assert report["transfer"]["measurable"] is True
    # Synthetic items here are always parseable and real ones are not, so the gap is real.
    assert report["transfer"]["drop_points"]["schema_valid"] > 0


def test_the_oracle_decomposition_separates_the_two_error_kinds(predictions) -> None:
    from chartqa_dt.prompting.parsing import parse_record

    items = []
    for p in predictions:
        parsed = parse_record(p["raw"])
        record = parsed.record if parsed.ok else None
        items.append(OracleItem(
            record_id=p["id"], gold_answer=p["gold"],
            pred_evidence=list((record or {}).get("evidence") or []),
            gold_evidence=p["gold_evidence"],
            pred_plan=(record or {}).get("plan"), gold_plan=p["gold_plan"]))
    result = decompose(items)
    assert result["n_eligible"] == 12
    assert {c["n"] for c in result["cells"].values()} == {12}
    # Substituting both gold halves must recover every record.
    assert result["attribution"]["executor_ceiling_pct"] == pytest.approx(100.0)
    assert result["attribution"]["visual_error_points"] > 0
    assert result["attribution"]["reasoning_error_points"] > 0


def test_stratification_partitions_the_whole_set(predictions) -> None:
    for key in ("chart_type", "kind", "source"):
        groups = stratify_by(predictions, key)
        assert sum(g["n"] for g in groups.values()) == len(predictions)


def test_calibration_reports_the_headline_at_full_coverage(predictions) -> None:
    from chartqa_dt.eval.diagnostics import diagnose_one

    rows = []
    for p in predictions:
        one = diagnose_one(p["raw"], gold_plan=p["gold_plan"])
        rows.append({**one, "correct": p["correct"]})
    report = calibrate(rows)
    assert report["headline_accuracy"] == pytest.approx(0.25)
    assert report["ece"] is None, "no probability was supplied, so none is invented"
    assert_headline_is_full_coverage(report, 0.25)


def test_a_fitted_calibrator_supplies_the_probability_calibration_declined(
        predictions) -> None:
    from chartqa_dt.eval.diagnostics import diagnose_one

    rows, labels = [], []
    for _ in range(30):                              # repeat for a fittable sample
        for p in predictions:
            one = diagnose_one(p["raw"], gold_plan=p["gold_plan"])
            rows.append({"mean_logprob": -0.3 if p["correct"] else -1.2,
                         "min_logprob": -1.0 if p["correct"] else -3.0,
                         "schema_valid": float(one["schema_valid"]),
                         "executor_agrees": float(one["agrees"]),
                         "evidence_area": 40000.0, "unit_ok": 1.0, "range_ok": 1.0,
                         "n_evidence": 2.0})
            labels.append(p["correct"])
    report = evaluate_calibrator(fit(rows, labels), rows, labels)
    assert report["auc"] > 0.9 and 0.0 <= report["ece"] <= 1.0
    assert not report["skip_crop"]


def test_the_crop_pass_consumes_calibrator_reliabilities(predictions) -> None:
    """The interface that matters: the calibrator's probability is the crop's trigger."""
    items = [{**p, "reliability": 0.9 if p["correct"] else 0.2,
              "focus_box": [200, 200, 700, 700]} for p in predictions]
    final, outcome = run_crop_pass(
        items, lambda item, region: {"schema_valid": True, "agrees": True,
                                     "correct": True})
    assert outcome.requested == 9, "only the unreliable records are offered a crop"
    assert outcome.helped == 9 and outcome.harmed == 0
    assert len(final) == len(items), "every record survives the pass"


def test_figures_cover_distinct_modes_and_respect_the_licence(predictions) -> None:
    from chartqa_dt.eval.diagnostics import diagnose_one
    from chartqa_dt.eval.figures import COMMITTABLE_SOURCES

    items = [{**p, **diagnose_one(p["raw"], gold_plan=p["gold_plan"],
                                  gold_evidence=p["gold_evidence"])}
             for p in predictions]
    figures = select_figures(items, per_mode=2)
    report = coverage_report(figures)
    assert report["n_modes"] >= 3, report
    # Where a mode has both, the committable source is chosen.
    correct = [f for f in figures if f.mode == "correct"]
    assert correct[0].source in COMMITTABLE_SOURCES
