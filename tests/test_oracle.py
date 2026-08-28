"""`PLAN.md` 9.1. The table only means something if all four cells share their records."""

from __future__ import annotations

import pytest

from chartqa_dt.eval.oracle import OracleItem, decompose, describe, run_one

GOLD_EVIDENCE = [{"label": "2019", "value": 245.0, "unit": None},
                 {"label": "2018", "value": 210.0, "unit": None}]
GOLD_PLAN = {"op": "difference", "args": ["2019", "2018"]}


def _item(**kw) -> OracleItem:
    base = {"record_id": "r", "gold_answer": "35",
            "pred_evidence": [dict(e) for e in GOLD_EVIDENCE],
            "gold_evidence": [dict(e) for e in GOLD_EVIDENCE],
            "pred_plan": dict(GOLD_PLAN), "gold_plan": dict(GOLD_PLAN)}
    return OracleItem(**{**base, **kw})


class TestRunOne:
    def test_a_perfect_record_is_correct_in_every_configuration(self) -> None:
        item = _item()
        for gold_e in (False, True):
            for gold_p in (False, True):
                assert run_one(item, gold_evidence=gold_e, gold_plan=gold_p)[0]

    def test_misread_values_fail_with_predicted_evidence_and_pass_with_gold(self) -> None:
        """The definition of visual error: right plan, wrong numbers."""
        item = _item(pred_evidence=[{"label": "2019", "value": 100.0, "unit": None},
                                    {"label": "2018", "value": 90.0, "unit": None}])
        assert not run_one(item, gold_evidence=False, gold_plan=False)[0]
        assert run_one(item, gold_evidence=True, gold_plan=False)[0]

    def test_a_wrong_operation_fails_until_the_gold_plan_replaces_it(self) -> None:
        """The definition of reasoning error: right numbers, wrong operation."""
        item = _item(pred_plan={"op": "sum", "args": ["2019", "2018"]})
        assert not run_one(item, gold_evidence=False, gold_plan=False)[0]
        assert run_one(item, gold_evidence=False, gold_plan=True)[0]

    def test_a_plan_that_does_not_fit_the_gold_evidence_is_a_failure_not_a_skip(self) -> None:
        """Substituting truth can break a prediction. That is a result, not a gap."""
        item = _item(pred_plan={"op": "lookup", "args": ["Wednesday"]})
        ok, why = run_one(item, gold_evidence=True, gold_plan=False)
        assert not ok and why == "executor_refused"

    def test_a_missing_plan_is_distinguished_from_a_refusing_executor(self) -> None:
        assert run_one(_item(pred_plan=None), gold_evidence=False,
                       gold_plan=False) == (False, "no_plan")


class TestDecompose:
    def test_all_four_cells_are_computed_on_identical_records(self) -> None:
        items = [_item(record_id=f"r{i}") for i in range(5)]
        items.append(_item(record_id="no-gold-plan", gold_plan=None))
        result = decompose(items)
        counts = {c["n"] for c in result["cells"].values()}
        assert counts == {5}, "a cell computed on a different set is not comparable"
        assert result["n_total"] == 6
        assert result["n_excluded_no_gold_plan"] == 1

    def test_records_without_gold_evidence_are_excluded_too(self) -> None:
        assert decompose([_item(gold_evidence=[])])["n_eligible"] == 0

    def test_visual_error_is_the_gain_from_substituting_gold_evidence(self) -> None:
        misread = _item(record_id="a",
                        pred_evidence=[{"label": "2019", "value": 1.0, "unit": None},
                                       {"label": "2018", "value": 1.0, "unit": None}])
        result = decompose([misread, _item(record_id="b")])
        assert result["cells"]["pred_pred"]["accuracy"] == pytest.approx(0.5)
        assert result["cells"]["gold_pred"]["accuracy"] == pytest.approx(1.0)
        assert result["attribution"]["visual_error_points"] == pytest.approx(50.0)

    def test_reasoning_error_is_the_gain_from_substituting_the_gold_plan(self) -> None:
        wrong_op = _item(record_id="a", pred_plan={"op": "sum", "args": ["2019", "2018"]})
        result = decompose([wrong_op, _item(record_id="b")])
        assert result["attribution"]["reasoning_error_points"] == pytest.approx(50.0)
        assert result["attribution"]["visual_error_points"] == pytest.approx(0.0)

    def test_the_executor_ceiling_is_reported_from_gold_and_gold(self) -> None:
        """If this is not ~100%, the executor itself is losing records."""
        result = decompose([_item(record_id=f"r{i}") for i in range(4)])
        assert result["attribution"]["executor_ceiling_pct"] == pytest.approx(100.0)

    def test_an_empty_set_does_not_divide_by_zero(self) -> None:
        result = decompose([])
        assert result["n_eligible"] == 0
        assert result["cells"]["pred_pred"]["accuracy"] == 0.0

    def test_the_description_names_what_was_excluded(self) -> None:
        text = describe(decompose([_item(), _item(gold_plan=None)]))
        assert "1 have no gold plan" in text
        assert "the executor's own ceiling" in text
